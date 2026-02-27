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

    def _rotate_round(self, round_name: str, angle: float) -> None:
        """Rotate a single round's image in-place."""
        vol = self.images[round_name]
        k_90 = round(angle / 90)
        if abs(angle - k_90 * 90) < 1e-6:
            yx_axes = (1, 2) if vol.ndim == 4 else (0, 1)
            # np.rot90 and imrotate share sign convention: k=1 is CCW, k=-1 is CW
            self.images[round_name] = np.ascontiguousarray(
                np.rot90(vol, k=k_90, axes=yx_axes)
            )
        else:
            from scipy.ndimage import rotate as ndimage_rotate

            yx_axes = (1, 2) if vol.ndim == 4 else (0, 1)
            self.images[round_name] = ndimage_rotate(
                vol, angle, axes=yx_axes, reshape=False, order=1
            ).astype(vol.dtype)

    @log_step
    def rotate(self, *, angle: float) -> FOV:
        """Rotate all loaded volumes by angle degrees in the YX plane.

        Applied after loading, before any other processing.
        For exact 90-degree multiples, uses np.rot90 (zero-copy, instant).
        For other angles, uses scipy.ndimage.rotate with bilinear interpolation.
        """
        for round_name in list(self.images.keys()):
            self._rotate_round(round_name, angle)
        return self

    @log_step
    def enhance_contrast(
        self,
        layers: list[str] | None = None,
        snr_threshold: float | None = None,
    ) -> FOV:
        """Per-channel min-max normalization.

        Parameters
        ----------
        snr_threshold : float or None
            If set, channels with max/mean < snr_threshold are not
            normalized (raw values kept). Prevents noise inflation.
        """
        from starfinder.preprocessing import min_max_normalize

        self._apply_to_layers(
            lambda v: min_max_normalize(v, snr_threshold=snr_threshold),
            layers,
        )
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
            # uint16 is sufficient: max sum of 4 uint8 channels = 1020
            return np.sum(img, axis=-1, dtype=np.uint16)
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
        method: str = "demons",
        fallback: bool = True,
        # Demons parameters
        iterations: list[int] | None = None,
        smoothing_sigma: float = 1.0,
        pyramid_mode: str = "antialias",
        # TPS parameters
        detection_threshold: float = 3.0,
        match_distance: float = 10.0,
        min_matches: int = 50,
        max_control_points: int = 1000,
        tps_smoothing: float = 1.0,
        grid_spacing: int = 32,
        # CPD parameters
        beta: float | None = None,
        lmbda: float = 2.0,
        cpd_w: float = 0.15,
        affine_first: bool = True,
        candidate_radius: float = 15.0,
        k_neighbors: int = 3,
    ) -> FOV:
        """Local (non-rigid) registration.

        Supports three methods:
        - ``"demons"`` (default): Iterative voxel-level optimization via SimpleITK.
        - ``"tps"``: Spot-based Thin Plate Spline — fits a smooth displacement
          field from matched spot correspondences. No SimpleITK dependency.
        - ``"cpd"``: Coherent Point Drift — simultaneous correspondence and
          transformation via EM on GMM. No SimpleITK dependency.

        When ``method="tps"`` or ``method="cpd"`` and ``fallback=True``,
        falls back to demons if insufficient spot matches.

        Displacement fields are ephemeral (applied then discarded).
        """
        if layers_to_register is None:
            layers_to_register = self.layers.to_register

        ref_round = self.layers.ref
        ref_3d = self.images[ref_round][:, :, :, ref_channel]

        for round_name in layers_to_register:
            if round_name not in self.images:
                continue
            mov_3d = self.images[round_name][:, :, :, ref_channel]

            if method == "cpd":
                try:
                    from starfinder.registration import register_volume_cpd

                    registered, _ = register_volume_cpd(
                        self.images[round_name],
                        ref_3d,
                        mov_3d,
                        detection_threshold=detection_threshold,
                        max_control_points=max_control_points,
                        beta=beta,
                        lmbda=lmbda,
                        w=cpd_w,
                        affine_first=affine_first,
                        grid_spacing=grid_spacing,
                        candidate_radius=candidate_radius,
                        k_neighbors=k_neighbors,
                    )
                except ValueError:
                    if not fallback:
                        raise
                    # Fall back to demons
                    from starfinder.registration import register_volume_local

                    registered, _ = register_volume_local(
                        self.images[round_name],
                        ref_3d,
                        mov_3d,
                        iterations=iterations,
                        smoothing_sigma=smoothing_sigma,
                        pyramid_mode=pyramid_mode,
                    )
            elif method == "tps":
                try:
                    from starfinder.registration import register_volume_tps

                    registered, _ = register_volume_tps(
                        self.images[round_name],
                        ref_3d,
                        mov_3d,
                        detection_threshold=detection_threshold,
                        match_distance=match_distance,
                        min_matches=min_matches,
                        max_control_points=max_control_points,
                        smoothing=tps_smoothing,
                        grid_spacing=grid_spacing,
                    )
                except ValueError:
                    if not fallback:
                        raise
                    # Fall back to demons
                    from starfinder.registration import register_volume_local

                    registered, _ = register_volume_local(
                        self.images[round_name],
                        ref_3d,
                        mov_3d,
                        iterations=iterations,
                        smoothing_sigma=smoothing_sigma,
                        pyramid_mode=pyramid_mode,
                    )
            else:
                from starfinder.registration import register_volume_local

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
        intensity_estimation: Literal[
            "noise", "adaptive", "adaptive_round", "global"
        ] = "noise",
        intensity_threshold: float = 5.0,
        min_distance: int = 1,
    ) -> FOV:
        """Detect spots on the reference round."""
        from starfinder.spotfinding import find_spots_3d

        ref_image = self.images[self.layers.ref]
        self.all_spots = find_spots_3d(
            ref_image,
            intensity_estimation=intensity_estimation,
            intensity_threshold=intensity_threshold,
            min_distance=min_distance,
        )
        return self

    def _extract_round(
        self,
        round_name: str,
        voxel_size: tuple[int, int, int] = (1, 2, 2),
    ) -> None:
        """Extract colors for a single round, adding columns to all_spots."""
        from starfinder.barcode import extract_from_location

        color, score = extract_from_location(
            self.images[round_name], self.all_spots, voxel_size
        )
        self.all_spots[f"{round_name}_color"] = color
        self.all_spots[f"{round_name}_score"] = score

    def _build_color_seq(self, layers: list[str] | None = None) -> None:
        """Concatenate per-round color columns into color_seq string."""
        if layers is None:
            layers = self.layers.seq
        color_cols = [f"{r}_color" for r in layers]
        self.all_spots["color_seq"] = self.all_spots[color_cols].astype(str).agg(
            "".join, axis=1
        )

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
        if layers is None:
            layers = self.layers.seq

        for round_name in layers:
            self._extract_round(round_name, voxel_size)
        self._build_color_seq(layers)
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

    # --- Streaming pipeline ---

    @log_step
    def run_streaming(
        self,
        *,
        rotate_angle: float | None = None,
        snr_threshold: float | None = None,
        intensity_estimation: Literal[
            "noise", "adaptive", "adaptive_round", "global"
        ] = "noise",
        intensity_threshold: float = 5.0,
        voxel_size: tuple[int, int, int] = (1, 2, 2),
        end_bases: str | None = None,
        start_base: str = "C",
        local_method: str | None = None,
        local_kwargs: dict | None = None,
    ) -> FOV:
        """Memory-efficient streaming pipeline. Peak memory = 2 x round_size.

        Processes one round at a time instead of loading all rounds
        simultaneously. Produces identical results to the batch flow
        (load_all -> enhance_all -> register_all -> spot_find ->
        extract_all -> filter).

        Parameters
        ----------
        local_method : str or None
            If set (e.g. ``"tps"`` or ``"demons"``), applies local
            registration after global registration for each non-ref round.
        local_kwargs : dict or None
            Extra keyword arguments passed to ``local_registration()``.

        The spot DataFrame (all_spots) accumulates per-round color columns
        throughout. Only image volumes are loaded and discarded per-round.
        Filtering runs at the end on the complete color_seq column.
        """
        ref = self.layers.ref

        # --- Phase 1: Reference round (kept in memory throughout) ---
        self.load_raw_images(rounds=[ref])
        if rotate_angle is not None:
            self._rotate_round(ref, rotate_angle)
        self.enhance_contrast(layers=[ref], snr_threshold=snr_threshold)
        self.spot_finding(
            intensity_estimation=intensity_estimation,
            intensity_threshold=intensity_threshold,
        )
        self._extract_round(ref, voxel_size)

        # --- Phase 2: Non-ref rounds (one at a time) ---
        for round_name in self.layers.to_register:
            self.load_raw_images(rounds=[round_name])
            if rotate_angle is not None:
                self._rotate_round(round_name, rotate_angle)
            self.enhance_contrast(
                layers=[round_name], snr_threshold=snr_threshold
            )
            self.global_registration(
                layers_to_register=[round_name], save_shifts=False
            )
            if local_method is not None:
                self.local_registration(
                    layers_to_register=[round_name],
                    method=local_method,
                    **(local_kwargs or {}),
                )
            self._extract_round(round_name, voxel_size)
            del self.images[round_name]

        # Save shift log once (not per-round)
        if self.global_shifts:
            self._save_shift_log()

        # --- Phase 3: Finalize (spot DataFrame only, images released) ---
        self._build_color_seq()
        self.reads_filtration(end_bases=end_bases, start_base=start_base)
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
