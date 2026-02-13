"""FOV: per-FOV stateful processor with fluent API."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from starfinder.dataset.logging import log_step
from starfinder.dataset.paths import FOVPaths
from starfinder.dataset.types import Codebook, ImageArray, Shift3D

if TYPE_CHECKING:
    from starfinder.dataset.dataset import STARMapDataset
    from starfinder.dataset.types import ChannelOrder, LayerState


@dataclass
class FOV:
    """Per-FOV processing state and methods.

    Mutable. NOT thread-safe. One instance per Snakemake job.
    Delegates to dataset for layers, codebook, and channel_order.
    All processing methods return ``self`` for fluent chaining.
    """

    dataset: STARMapDataset
    fov_id: str

    # Mutable state
    images: dict[str, ImageArray] = field(default_factory=dict)
    metadata: dict[str, dict] = field(default_factory=dict)
    global_shifts: dict[str, Shift3D] = field(default_factory=dict)
    local_registered: set[str] = field(default_factory=set)
    all_spots: pd.DataFrame | None = None
    good_spots: pd.DataFrame | None = None

    # --- Delegated properties ---

    @property
    def layers(self) -> LayerState:
        return self.dataset.layers

    @property
    def codebook(self) -> Codebook | None:
        return self.dataset.codebook

    # --- Path helpers ---

    @property
    def paths(self) -> FOVPaths:
        return FOVPaths(self.dataset.output_root, self.fov_id)

    def input_dir(self, round_name: str) -> Path:
        """Input directory for a specific round."""
        return self.dataset.input_root / round_name / self.fov_id

    # --- Image loading ---

    @log_step
    def load_raw_images(
        self,
        rounds: list[str] | None = None,
        channel_order: ChannelOrder | None = None,
        *,
        convert_uint8: bool = True,
        subdir: str = "",
        layer_slot: Literal["seq", "other"] = "seq",
    ) -> FOV:
        """Load raw TIFF stacks for specified rounds.

        Delegates to ``starfinder.io.load_image_stacks()`` per round.
        """
        from starfinder.io import load_image_stacks

        if rounds is None:
            rounds = (
                self.layers.seq if layer_slot == "seq" else self.layers.other
            )
        if channel_order is None:
            channel_order = self.dataset.channel_order

        for round_name in rounds:
            img, meta = load_image_stacks(
                self.input_dir(round_name),
                channel_order=channel_order,
                subdir=subdir,
                convert_uint8=convert_uint8,
            )
            self.images[round_name] = img
            self.metadata[round_name] = meta
        return self

    # --- Preprocessing ---

    def _apply_to_layers(self, func, layers: list[str] | None) -> None:
        """Apply a function to images for the given layers (or all)."""
        if layers is None:
            layers = self.layers.all_layers
        for name in layers:
            if name in self.images:
                self.images[name] = func(self.images[name])

    @log_step
    def enhance_contrast(self, layers: list[str] | None = None) -> FOV:
        """Per-channel min-max normalization."""
        from starfinder.preprocessing import min_max_normalize

        self._apply_to_layers(min_max_normalize, layers)
        return self

    @log_step
    def hist_equalize(
        self,
        ref_channel: int = 0,
        nbins: int = 64,
        layers: list[str] | None = None,
    ) -> FOV:
        """Histogram matching to reference layer's channel."""
        from starfinder.preprocessing import histogram_match

        reference = self.images[self.layers.ref][:, :, :, ref_channel]
        if layers is None:
            layers = self.layers.all_layers
        for name in layers:
            if name in self.images:
                self.images[name] = histogram_match(
                    self.images[name], reference, nbins=nbins
                )
        return self

    @log_step
    def morph_recon(
        self, radius: int = 3, layers: list[str] | None = None
    ) -> FOV:
        """Background removal via morphological reconstruction."""
        from starfinder.preprocessing import morphological_reconstruction

        self._apply_to_layers(
            lambda v: morphological_reconstruction(v, radius=radius), layers
        )
        return self

    @log_step
    def tophat(
        self, radius: int = 3, layers: list[str] | None = None
    ) -> FOV:
        """White tophat filtering."""
        from starfinder.preprocessing import tophat_filter

        self._apply_to_layers(
            lambda v: tophat_filter(v, radius=radius), layers
        )
        return self

    @log_step
    def make_projection(
        self, method: Literal["max", "sum"] = "max"
    ) -> FOV:
        """Apply Z-projection to ALL images."""
        from starfinder.utils import make_projection as _make_projection

        for name in list(self.images):
            self.images[name] = _make_projection(
                self.images[name], method=method
            )
        return self

    # --- Registration ---

    def _make_ref_3d(
        self,
        round_name: str,
        mode: Literal["merged", "single-channel"],
        channel: int,
    ) -> np.ndarray:
        """Create 3D reference/moving image for registration."""
        img = self.images[round_name]
        if mode == "merged":
            return np.sum(img, axis=-1)
        else:
            return img[:, :, :, channel]

    @log_step
    def global_registration(
        self,
        *,
        layers_to_register: list[str] | None = None,
        ref_img: Literal["merged", "single-channel"] = "merged",
        mov_img: Literal["merged", "single-channel"] = "merged",
        ref_channel: int = 0,
        save_shifts: bool = True,
    ) -> FOV:
        """Global (rigid) registration using phase correlation.

        Stores shifts in ``self.global_shifts`` and optionally writes
        a shift log CSV.
        """
        from starfinder.registration import register_volume

        if layers_to_register is None:
            layers_to_register = self.layers.to_register

        ref_round = self.layers.ref
        ref_3d = self._make_ref_3d(ref_round, ref_img, ref_channel)

        for round_name in layers_to_register:
            if round_name not in self.images:
                continue
            mov_3d = self._make_ref_3d(round_name, mov_img, ref_channel)
            registered, shifts = register_volume(
                self.images[round_name], ref_3d, mov_3d
            )
            self.images[round_name] = registered
            self.global_shifts[round_name] = shifts

        if save_shifts and self.global_shifts:
            self._save_shift_log()

        return self

    def _save_shift_log(self) -> None:
        """Write global shifts to CSV in MATLAB-compatible format."""
        path = self.paths.shift_log()
        path.parent.mkdir(parents=True, exist_ok=True)

        rows = []
        for round_name, (dz, dy, dx) in self.global_shifts.items():
            rows.append(
                {
                    "fov_id": self.fov_id,
                    "round": round_name,
                    "row": dy,
                    "col": dx,
                    "z": dz,
                }
            )
        pd.DataFrame(rows).to_csv(path, index=False)

    @log_step
    def local_registration(
        self,
        *,
        ref_channel: int = 0,
        layers_to_register: list[str] | None = None,
        iterations: list[int] | None = None,
        smoothing_sigma: float = 1.0,
        method: str = "demons",
        pyramid_mode: str = "antialias",
    ) -> FOV:
        """Local (non-rigid) registration using demons algorithm.

        Displacement fields are ephemeral (applied then discarded).
        """
        from starfinder.registration import register_volume_local

        if layers_to_register is None:
            layers_to_register = self.layers.to_register

        ref_round = self.layers.ref
        ref_3d = self.images[ref_round][:, :, :, ref_channel]

        for round_name in layers_to_register:
            if round_name not in self.images:
                continue
            mov_3d = self.images[round_name][:, :, :, ref_channel]
            registered, _ = register_volume_local(
                self.images[round_name],
                ref_3d,
                mov_3d,
                iterations=iterations,
                smoothing_sigma=smoothing_sigma,
                method=method,
                pyramid_mode=pyramid_mode,
            )
            self.images[round_name] = registered
            self.local_registered.add(round_name)

        return self

    # --- Spot finding & barcode ---

    @log_step
    def spot_finding(
        self,
        *,
        intensity_estimation: Literal["adaptive", "global"] = "adaptive",
        intensity_threshold: float = 0.2,
    ) -> FOV:
        """Detect spots on the reference round."""
        from starfinder.spotfinding import find_spots_3d

        ref_image = self.images[self.layers.ref]
        self.all_spots = find_spots_3d(
            ref_image,
            intensity_estimation=intensity_estimation,
            intensity_threshold=intensity_threshold,
        )
        return self

    @log_step
    def reads_extraction(
        self,
        voxel_size: tuple[int, int, int] = (1, 2, 2),
        layers: list[str] | None = None,
    ) -> FOV:
        """Extract color sequences from spot locations across rounds.

        Adds ``{round}_color``, ``{round}_score`` columns per round,
        and a concatenated ``color_seq`` column.
        """
        from starfinder.barcode import extract_from_location

        if layers is None:
            layers = self.layers.seq

        for round_name in layers:
            color, score = extract_from_location(
                self.images[round_name], self.all_spots, voxel_size
            )
            self.all_spots[f"{round_name}_color"] = color
            self.all_spots[f"{round_name}_score"] = score

        # Concatenate per-round colors into single color_seq string
        color_cols = [f"{r}_color" for r in layers]
        self.all_spots["color_seq"] = self.all_spots[color_cols].apply(
            lambda row: "".join(str(v) for v in row), axis=1
        )
        return self

    @log_step
    def reads_filtration(
        self,
        *,
        end_bases: str | None = None,
        start_base: str = "C",
    ) -> FOV:
        """Filter reads against codebook."""
        from starfinder.barcode import filter_reads

        if self.codebook is None:
            raise ValueError(
                "Codebook not loaded. Call dataset.load_codebook() first."
            )

        good, _stats = filter_reads(
            self.all_spots,
            self.codebook.seq_to_gene,
            end_bases=end_bases,
            start_base=start_base,
        )
        self.good_spots = good
        return self

    # --- Output ---

    def save_ref_merged(self) -> Path:
        """Save reference merged image as TIFF."""
        from starfinder.io import save_stack
        from starfinder.utils import make_projection

        ref_image = self.images[self.layers.ref]
        if self.dataset.maximum_projection:
            ref_image = make_projection(ref_image)

        path = self.paths.ref_merged_tif
        path.parent.mkdir(parents=True, exist_ok=True)
        save_stack(ref_image, path)
        return path

    def save_signal(
        self,
        slot: Literal["allSpots", "goodSpots"] = "goodSpots",
        columns: list[str] | None = None,
    ) -> Path:
        """Save spots to CSV with 1-based coordinates.

        Converts internal 0-based (z, y, x) to CSV 1-based (x, y, z, gene).
        """
        spots = self.good_spots if slot == "goodSpots" else self.all_spots
        if spots is None or spots.empty:
            raise ValueError(f"No spots in '{slot}' to save.")

        if columns is None:
            base = ["x", "y", "z"]
            if "gene" in spots.columns:
                base.append("gene")
            columns = base

        out = spots[columns].copy()
        # Convert 0-based → 1-based for coordinate columns
        for col in ("x", "y", "z"):
            if col in out.columns:
                out[col] = out[col] + 1

        path = self.paths.signal_csv(slot)
        path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(path, index=False)
        return path

    # --- Subtile operations ---

    def create_subtiles(
        self,
        *,
        out_dir: Path | None = None,
    ) -> pd.DataFrame:
        """Partition FOV into overlapping subtiles and save as NPZ.

        Returns subtile coordinates DataFrame with 1-based coords
        for ``stitch_subtile.py`` compatibility.
        """
        if self.dataset.subtile is None:
            raise ValueError("SubtileConfig not set on dataset.")

        subtile_cfg = self.dataset.subtile
        if out_dir is None:
            out_dir = self.paths.subtile_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        coord_rows = []
        for t, window in enumerate(subtile_cfg.windows):
            sy, sx = window.to_slice()

            # Extract cropped images for each round
            arrays = {}
            for round_name, img in self.images.items():
                if img.ndim == 4:
                    arrays[f"images_{round_name}"] = img[:, sy, sx, :]
                else:
                    # 2D projected (Y, X, C) or (Y, X)
                    arrays[f"images_{round_name}"] = img[sy, sx]

            # Save NPZ
            npz_path = out_dir / f"subtile_{t:05d}.npz"
            np.savez_compressed(
                npz_path,
                **arrays,
                fov_id=self.fov_id,
                subtile_id=t,
                layers_seq=self.layers.seq,
                layers_ref=self.layers.ref,
            )

            # 1-based coordinates for stitch_subtile.py
            coord_rows.append(
                {
                    "t": t,
                    "scoords_x": window.x_start + 1,
                    "scoords_y": window.y_start + 1,
                    "ecoords_x": window.x_end,  # exclusive→inclusive + 0→1
                    "ecoords_y": window.y_end,
                }
            )

        coords_df = pd.DataFrame(coord_rows)
        coords_df.to_csv(out_dir / "subtile_coords.csv", index=False)
        return coords_df

    @classmethod
    def from_subtile(
        cls,
        subtile_path: Path,
        dataset: STARMapDataset,
        fov_id: str,
    ) -> FOV:
        """Load FOV state from a saved NPZ subtile."""
        data = np.load(subtile_path, allow_pickle=True)

        fov = cls(dataset=dataset, fov_id=fov_id)
        for key in data.files:
            if key.startswith("images_"):
                round_name = key[len("images_") :]
                fov.images[round_name] = data[key]
        return fov
