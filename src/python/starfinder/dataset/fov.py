"""FOV: per-FOV stateful processor with fluent API."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import json
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from starfinder.image import ImageMetadata, _validate_image
from starfinder.spot_finding import LocalMaximaConfig, SpotFindingResult
from starfinder.io import ImageLoadConfig
from starfinder.preprocessing import (MinMaxNormalizationConfig, HistogramMatchingConfig, ReconstructionConfig, TophatConfig, ProjectionConfig)
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
    Image-processing methods return ``self`` for fluent chaining; output
    methods return paths/tables. See individual methods.

    Parameters
    ----------
    dataset : STARMapDataset
        Shared STARMapDataset.
    fov_id : str
        FOV identifier.
    images : dict[str, ImageArray]
        Round to (Z,Y,X,C) numeric array mapping; default empty dict.
    metadata : dict[str, ImageMetadata]
        Round to loader metadata mapping; default empty dict.
    global_shifts : dict[str, Shift3D]
        Round to detected (dz,dy,dx) voxel displacement; default empty dict.
    local_registered : set[str]
        Names of locally registered rounds; default empty set.
    spot_result : SpotFindingResult | None
        Detection table, geometry, identity namespace and effective config.
    subtile_id : int | None
        One-based saved subtile ID, or None for the full FOV.
    all_spots : pd.DataFrame | None
        Detected/extracted zero-based spot DataFrame, default None.
    good_spots : pd.DataFrame | None
        Filtered spot DataFrame, default None.

    """

    dataset: STARMapDataset
    fov_id: str

    # Mutable state
    images: dict[str, ImageArray] = field(default_factory=dict)
    metadata: dict[str, ImageMetadata] = field(default_factory=dict)
    global_shifts: dict[str, Shift3D] = field(default_factory=dict)
    local_registered: set[str] = field(default_factory=set)
    spot_result: SpotFindingResult | None = None
    subtile_id: int | None = None
    all_spots: pd.DataFrame | None = None
    good_spots: pd.DataFrame | None = None

    load_diagnostics: dict[str, dict] = field(default_factory=dict)

    # --- Delegated properties ---

    @property
    def layers(self) -> LayerState:
        """Shared dataset LayerState.
        """
        return self.dataset.layers

    @property
    def codebook(self) -> Codebook | None:
        """Shared dataset Codebook, or None before loading.
        """
        return self.dataset.codebook

    # --- Path helpers ---

    @property
    def paths(self) -> FOVPaths:
        """FOVPaths for this field of view; no directories are created.
        """
        return FOVPaths(self.dataset.output_root, self.fov_id)

    def input_dir(self, round_name: str) -> Path:
        """Input directory for a specific round.

        Parameters
        ----------
        round_name : str
            Round name appended to input_root before fov_id.

        Returns
        -------
        pathlib.Path
            Input directory; does not create it.
        """
        return self.dataset.input_root / round_name / self.fov_id

    # --- Image loading ---

    @log_step
    def load_raw_images(
        self,
        rounds: list[str] | None = None,
        channel_order: ChannelOrder | None = None,
        *,
        config: ImageLoadConfig | None = None,
        subdir: str = "",
        layer_slot: Literal["seq", "other"] = "seq",
    ) -> FOV:
        """Load raw TIFF stacks for specified rounds.

        Delegates to ``starfinder.io.load_round()`` per round.

        Parameters
        ----------
        rounds : list[str] | None
            Round names to load; None uses the dataset layers selected by layer_slot.
        channel_order : ChannelOrder | None
            Filename channel patterns in output C order; None uses dataset.channel_order.
        config : ImageLoadConfig | None
            Explicit loading/conversion policy; None preserves source dtype.
        subdir : str
            Optional subdirectory beneath each round/FOV input directory.
        layer_slot : Literal['seq', 'other']
            Round category used when rounds is None: seq (default) or other.

        Returns
        -------
        FOV
            This instance, with processing state updated in place.
        """
        from starfinder.io import load_round

        if config is not None and (channel_order is not None or subdir):
            raise ValueError("select channels/subdir through config or arguments, not both")
        if rounds is None:
            rounds = (
                self.layers.seq if layer_slot == "seq" else self.layers.other
            )
        if channel_order is None:
            channel_order = self.dataset.channel_order

        for round_name in rounds:
            load_config = config or ImageLoadConfig(
                channel_labels=tuple(channel_order), subdir=subdir,
            )
            loaded = load_round(self.input_dir(round_name), config=load_config)
            self.images[round_name] = loaded.image
            self.metadata[round_name] = loaded.metadata
            self.load_diagnostics[round_name] = loaded.diagnostics
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
        vol = _validate_image(self.images[round_name])
        source = self.metadata.get(round_name, ImageMetadata(f"{self.fov_id}/{round_name}"))
        rotated_metadata = source.rotated(vol.shape[:3], angle, frame_id=f"{source.frame_id}/rotate:{angle}")
        k_90 = round(angle / 90)
        if abs(angle - k_90 * 90) < 1e-6:
            yx_axes = (1, 2)
            # np.rot90 and imrotate share sign convention: k=1 is CCW, k=-1 is CW
            self.images[round_name] = np.ascontiguousarray(
                np.rot90(vol, k=k_90, axes=yx_axes)
            )
        else:
            from scipy.ndimage import rotate as ndimage_rotate

            yx_axes = (1, 2)
            self.images[round_name] = ndimage_rotate(
                vol, angle, axes=yx_axes, reshape=False, order=1
            ).astype(vol.dtype)

        self.metadata[round_name] = rotated_metadata
        self.load_diagnostics.setdefault(round_name, {})["rotation"] = {
            "source_metadata": asdict(source), "source_shape_zyx": vol.shape[:3],
            "output_shape_zyx": self.images[round_name].shape[:3], "angle_degrees": angle,
            "mapping": "source = R @ (output - output_center) + source_center",
        }

    @log_step
    def rotate(self, *, angle: float) -> FOV:
        """Rotate all loaded volumes by angle degrees in the YX plane.

        Applied after loading, before any other processing.
        For exact 90-degree multiples, uses np.rot90 followed by a contiguous copy.
        For other angles, uses scipy.ndimage.rotate with bilinear interpolation.

        Parameters
        ----------
        angle : float
            Rotation angle in degrees, positive counterclockwise in the YX plane.

        Returns
        -------
        FOV
            This instance, with processing state updated in place.
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
            If set, nonconstant channels with max/mean < snr_threshold keep
            raw values clipped to uint8. Constant channels always map to zero.
            Prevents noise inflation without overriding the constant policy.

        Parameters
        ----------
        layers : list[str] | None
            Rounds to process; None uses all configured layers for preprocessing, sequencing layers for extraction.

        Returns
        -------
        FOV
            This instance, with processing state updated in place.
        """
        from starfinder.preprocessing import normalize_intensity

        self._apply_to_layers(
            lambda v: normalize_intensity(v, config=MinMaxNormalizationConfig('uint8', (0, 255), snr_threshold=snr_threshold)),
            layers,
        )
        return self

    @log_step
    def hist_equalize(
        self,
        ref_channel: int = 0,
        layers: list[str] | None = None,
    ) -> FOV:
        """Histogram matching to reference layer's channel.

        Parameters
        ----------
        ref_channel : int
            Zero-based reference channel index; also selects moving channel in single-channel registration.
        layers : list[str] | None
            Rounds to process; None uses all configured layers for preprocessing, sequencing layers for extraction.

        Returns
        -------
        FOV
            This instance, with processing state updated in place.
        """
        from starfinder.preprocessing import match_histogram

        reference = self.images[self.layers.ref][:, :, :, ref_channel]
        if layers is None:
            layers = self.layers.all_layers
        for name in layers:
            if name in self.images:
                self.images[name] = match_histogram(
                    self.images[name], reference, config=HistogramMatchingConfig()
                )
        return self

    @log_step
    def morph_recon(
        self, radius: int = 3, layers: list[str] | None = None
    ) -> FOV:
        """Background removal via morphological reconstruction.

        Parameters
        ----------
        radius : int
            Nonnegative structuring-element radius in voxels for each channel.
        layers : list[str] | None
            Rounds to process; None uses all configured layers for preprocessing, sequencing layers for extraction.

        Returns
        -------
        FOV
            This instance, with processing state updated in place.
        """
        from starfinder.preprocessing import reconstruct_background

        self._apply_to_layers(
            lambda v: reconstruct_background(v, config=ReconstructionConfig(radius_yx=radius)), layers
        )
        return self

    @log_step
    def tophat(
        self, radius: int = 3, layers: list[str] | None = None
    ) -> FOV:
        """White tophat filtering.

        Parameters
        ----------
        radius : int
            Nonnegative structuring-element radius in voxels for each channel.
        layers : list[str] | None
            Rounds to process; None uses all configured layers for preprocessing, sequencing layers for extraction.

        Returns
        -------
        FOV
            This instance, with processing state updated in place.
        """
        from starfinder.preprocessing import filter_tophat

        self._apply_to_layers(
            lambda v: filter_tophat(v, config=TophatConfig(radius_yx=radius)), layers
        )
        return self

    @log_step
    def project_image(
        self, method: Literal["max", "sum"] = "max"
    ) -> FOV:
        """Apply Z-projection to ALL images.

        Parameters
        ----------
        method : Literal['max', 'sum']
            Projection method max or sum (retains singleton Z; sum does not rescale).

        Returns
        -------
        FOV
            This instance, with processing state updated in place.
        """
        from starfinder.preprocessing import project_image as _project_image

        for name in list(self.images):
            source_shape = self.images[name].shape[:3]
            self.images[name] = _project_image(
                self.images[name], config=ProjectionConfig(method=method)
            )
            source = self.metadata.get(name, ImageMetadata(f"{self.fov_id}/{name}"))
            self.metadata[name] = source.projected(method=method)
            self.load_diagnostics.setdefault(name, {})["projection_source_metadata"] = asdict(source)
            self.load_diagnostics[name]["projection_source_shape_zyx"] = source_shape
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

        Parameters
        ----------
        layers_to_register : list[str] | None
            Rounds to register; None uses all configured non-reference layers. Unloaded rounds are skipped.
        ref_img : Literal['merged', 'single-channel']
            Reference representation: merged sums channels as uint16; single-channel selects ref_channel.
        mov_img : Literal['merged', 'single-channel']
            Moving representation: merged sums channels as uint16; single-channel selects ref_channel.
        ref_channel : int
            Zero-based reference channel index; also selects moving channel in single-channel registration.
        save_shifts : bool
            Whether to write detected displacements to the FOV shift-log CSV.

        Returns
        -------
        FOV
            This instance, with processing state updated in place.

        Notes
        -----
        global_shifts and the row/col/z log contain detected displacement, not
        the correction translation. See :func:`starfinder.registration.register_volume`.
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
        boundary_mode: str = "constant",
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
        falls back to demons on any ValueError from that registration call.

        Displacement fields are ephemeral (applied then discarded).

        Parameters
        ----------
        boundary_mode : str
            How to handle out-of-bounds source coordinates during warping:
            ``"constant"`` (default) fills with 0; ``"nearest"`` extends edges.

        Parameters
        ----------
        ref_channel : int
            Zero-based reference channel index; also selects moving channel in single-channel registration.
        layers_to_register : list[str] | None
            Rounds to register; None uses all configured non-reference layers. Unloaded rounds are skipped.
        method : str
            Local registration backend: demons, diffeomorphic, symmetric, fast_symmetric, tps, or cpd.
        fallback : bool
            For TPS/CPD, True catches any ValueError and retries demons; False propagates it. Fallback requires SimpleITK.
        iterations : list[int] | None
            Demons iterations per pyramid level; None uses [100, 50, 25].
        smoothing_sigma : float
            Demons displacement smoothing standard deviation in voxel units.
        pyramid_mode : str
            Demons pyramid: antialias (default) or sitk.
        detection_threshold : float
            TPS/CPD noise-floor detection k in median + k * MAD * 1.4826.
        match_distance : float
            TPS maximum correspondence distance in voxel-index Euclidean units.
        min_matches : int
            TPS minimum matched pairs; insufficient matches raise ValueError.
        max_control_points : int
            Maximum TPS control points or target CPD point count.
        tps_smoothing : float
            Smoothing passed to tps_register as smoothing.
        grid_spacing : int
            Coarse dense-field grid stride in voxel indices for TPS/CPD.
        beta : float | None
            CPD kernel width in voxels; None estimates it from moving-point neighbor distances.
        lmbda : float
            CPD regularization weight; larger values favor smoother transformations.
        cpd_w : float
            CPD expected outlier fraction in [0, 1), passed as w.
        affine_first : bool
            Whether CPD performs affine alignment before nonrigid fitting.
        candidate_radius : float
            CPD moving-candidate search radius in voxel units.
        k_neighbors : int
            CPD neighbors retained per fixed-point sampling anchor.

        Returns
        -------
        FOV
            This instance, with processing state updated in place.
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
                        boundary_mode=boundary_mode,
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
                        boundary_mode=boundary_mode,
                    )
            elif method == "tps":
                try:
                    from starfinder.registration import register_volume_tps

                    registered, _ = register_volume_tps(
                        self.images[round_name],
                        ref_3d,
                        mov_3d,
                        boundary_mode=boundary_mode,
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
                        boundary_mode=boundary_mode,
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
                    boundary_mode=boundary_mode,
                )

            self.images[round_name] = registered
            self.local_registered.add(round_name)

        return self

    # --- Spot finding & barcode ---

    @log_step
    def find_spots(self, *, config: LocalMaximaConfig = LocalMaximaConfig()) -> FOV:
        """Detect reference-round spots with explicit config and FOV identity.

        The namespace encodes dataset/sample/FOV and, for saved subtiles, the
        one-based subtile ID. IDs are retained in extraction/filtering tables.
        """
        from starfinder.spot_finding import find_spots

        if not isinstance(config, LocalMaximaConfig):
            raise TypeError("FOV detection requires LocalMaximaConfig")
        if config.channel_labels is None and self.dataset.channel_order:
            config = replace(config, channel_labels=tuple(self.dataset.channel_order))
        namespace = json.dumps([self.dataset.dataset_id, self.dataset.sample_id,
                                self.fov_id, self.subtile_id], separators=(",", ":"))
        metadata = self.metadata.get(self.layers.ref, ImageMetadata(f"{self.fov_id}/{self.layers.ref}"))
        self.spot_result = find_spots(self.images[self.layers.ref], config=config,
                                     metadata=metadata, spot_namespace=namespace)
        self.all_spots = self.spot_result.spots.copy()
        self.all_spots["spot_namespace"] = pd.Series(namespace, index=self.all_spots.index, dtype="string")
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

        Parameters
        ----------
        voxel_size : tuple[int, int, int]
            Extraction half-widths (dz, dy, dx) in voxel indices, not physical spacing.
        layers : list[str] | None
            Rounds to process; None uses all configured layers for preprocessing, sequencing layers for extraction.

        Returns
        -------
        FOV
            This instance, with processing state updated in place.
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
        """Filter reads against codebook.

        Parameters
        ----------
        end_bases : str | None
            Optional decoded sequence suffix for filtering; None disables suffix filtering.
        start_base : str
            Starting nucleotide for color decoding when end_bases is supplied.

        Returns
        -------
        FOV
            This instance, with processing state updated in place.

        Raises
        ------
        ValueError
            Dataset codebook has not been loaded.
        """
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
        """Streaming pipeline retaining the reference and one moving round.

        Processes one round at a time instead of loading all rounds
        simultaneously. Temporary registration/extraction buffers also consume
        memory. Operations are load, optional rotation, enhancement, registration,
        spot finding, extraction, and filtering.

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

        Parameters
        ----------
        rotate_angle : float | None
            Optional rotation in degrees before enhancement; None skips rotation (does not read dataset.rotate_angle).
        snr_threshold : float | None
            Optional max/mean threshold for skipping normalization; None disables the gate.
        intensity_estimation : Literal['noise', 'adaptive', 'adaptive_round', 'global']
            Spot threshold mode: noise, adaptive, adaptive_round, or global; pass together with intensity_threshold.
        intensity_threshold : float
            Noise k-sigma multiplier, or intensity fraction for adaptive/global modes.
        voxel_size : tuple[int, int, int]
            Extraction half-widths (dz, dy, dx) in voxel indices, not physical spacing.
        end_bases : str | None
            Optional decoded sequence suffix for filtering; None disables suffix filtering.
        start_base : str
            Starting nucleotide for color decoding when end_bases is supplied.

        Returns
        -------
        FOV
            This instance, with processing state updated in place.
        """
        ref = self.layers.ref

        # --- Phase 1: Reference round (kept in memory throughout) ---
        self.load_raw_images(rounds=[ref])
        if rotate_angle is not None:
            self._rotate_round(ref, rotate_angle)
        self.enhance_contrast(layers=[ref], snr_threshold=snr_threshold)
        self.find_spots(config=LocalMaximaConfig(threshold_mode=intensity_estimation, threshold_value=intensity_threshold))
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

    @log_step
    def run_streaming_gr(
        self,
        *,
        rotate_angle: float | None = None,
        snr_threshold: float | None = None,
        ref_img: Literal["merged", "single-channel"] = "merged",
        mov_img: Literal["merged", "single-channel"] = "merged",
        ref_channel: int = 0,
        hist_equalize: bool = False,
        hist_equalize_ref_channel: int = 0,
        morph_recon: bool = False,
        morph_recon_radius: int = 3,
    ) -> FOV:
        """Streaming global registration only (for subtile workflow).

        Processes one non-ref round at a time to reduce peak memory.
        After this method, the FOV has all images loaded with global
        shifts applied, ready for ``create_subtiles()``.

        Parameters
        ----------
        rotate_angle : float | None
            Optional rotation in degrees before enhancement; None skips rotation (does not read dataset.rotate_angle).
        snr_threshold : float | None
            Optional max/mean threshold for skipping normalization; None disables the gate.
        ref_img : Literal['merged', 'single-channel']
            Reference representation: merged sums channels as uint16; single-channel selects ref_channel.
        mov_img : Literal['merged', 'single-channel']
            Moving representation: merged sums channels as uint16; single-channel selects ref_channel.
        ref_channel : int
            Zero-based reference channel index; also selects moving channel in single-channel registration.
        hist_equalize : bool
            Whether to histogram-match non-reference rounds to the reference channel.
        hist_equalize_ref_channel : int
            Zero-based reference channel used for histogram matching.
        morph_recon : bool
            Whether to remove background by morphological reconstruction.
        morph_recon_radius : int
            Morphological reconstruction radius in voxels.

        Returns
        -------
        FOV
            This instance, with processing state updated in place.
        """
        ref = self.layers.ref

        # --- Phase 1: Load and preprocess ref round ---
        self.load_raw_images(rounds=[ref])
        if rotate_angle is not None:
            self._rotate_round(ref, rotate_angle)
        self.enhance_contrast(layers=[ref], snr_threshold=snr_threshold)
        if hist_equalize:
            # For hist_equalize we need to save the reference for later rounds
            pass  # ref round is the reference itself
        if morph_recon:
            self.morph_recon(radius=morph_recon_radius, layers=[ref])

        # --- Phase 2: Non-ref rounds (one at a time) ---
        for round_name in self.layers.to_register:
            self.load_raw_images(rounds=[round_name])
            if rotate_angle is not None:
                self._rotate_round(round_name, rotate_angle)
            self.enhance_contrast(
                layers=[round_name], snr_threshold=snr_threshold
            )
            if hist_equalize:
                from starfinder.preprocessing import match_histogram

                reference = self.images[ref][:, :, :, hist_equalize_ref_channel]
                self.images[round_name] = match_histogram(
                    self.images[round_name], reference
                )
            if morph_recon:
                self.morph_recon(
                    radius=morph_recon_radius, layers=[round_name]
                )
            self.global_registration(
                layers_to_register=[round_name],
                ref_img=ref_img,
                mov_img=mov_img,
                ref_channel=ref_channel,
                save_shifts=False,
            )

        # Save shift log once
        if self.global_shifts:
            self._save_shift_log()

        return self

    # --- Output ---

    def save_ref_merged(self) -> Path:
        """Save reference merged image as TIFF.

        Returns
        -------
        pathlib.Path
            Written TIFF path; maximum_projection optionally reduces Z to one plane. Does not sum channels.
        """
        from starfinder.io import save_volume
        from starfinder.preprocessing import project_image

        ref_image = self.images[self.layers.ref]
        metadata = self.metadata.get(self.layers.ref)
        if self.dataset.maximum_projection:
            ref_image = project_image(ref_image)
            if metadata is not None:
                metadata = metadata.projected(method="max")

        path = self.paths.ref_merged_tif
        path.parent.mkdir(parents=True, exist_ok=True)
        save_volume(ref_image, path, metadata=metadata)
        return path

    def save_signal(
        self,
        slot: Literal["allSpots", "goodSpots"] = "goodSpots",
        columns: list[str] | None = None,
    ) -> Path:
        """Save spots to CSV with 1-based coordinates.

        Converts internal 0-based (z, y, x) to CSV 1-based (x, y, z, gene).

        Parameters
        ----------
        slot : Literal['allSpots', 'goodSpots']
            goodSpots selects filtered spots; allSpots selects all detected spots.
        columns : list[str] | None
            Columns to write; None uses x, y, z and gene when present. Included x/y/z are incremented by one.

        Returns
        -------
        pathlib.Path
            Written CSV path. The in-memory DataFrame is unchanged.

        Raises
        ------
        ValueError
            Selected spots are absent or empty.
        KeyError
            Requested columns are missing.
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

    def save_log(self, log_type: Literal["rsf", "gr"] = "rsf") -> Path:
        """Write a pipeline log file (summary of steps run).

        Parameters
        ----------
        log_type : Literal['rsf', 'gr']
            rsf selects the read/spot log; gr selects the global-registration log.

        Returns
        -------
        pathlib.Path
            Written summary text-file path.
        """
        import time

        path = self.paths.rsf_log() if log_type == "rsf" else self.paths.gr_log()
        path.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            f"FOV: {self.fov_id}",
            f"Backend: python",
            f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Rounds: {self.layers.seq}",
            f"Ref round: {self.layers.ref}",
            f"Global shifts: {dict(self.global_shifts)}",
            f"Local registered: {self.local_registered}",
            f"All spots: {len(self.all_spots) if self.all_spots is not None else 0}",
            f"Good spots: {len(self.good_spots) if self.good_spots is not None else 0}",
        ]
        path.write_text("\n".join(lines) + "\n")
        return path

    def save_score_log(self, suffix: str = "") -> Path:
        """Write spot-finding score log (spot count summary).

        Parameters
        ----------
        suffix : str
            Optional text appended to fov_id in the score-log filename.

        Returns
        -------
        pathlib.Path
            Written spot-count summary path.
        """
        path = self.paths.score_log(suffix)
        path.parent.mkdir(parents=True, exist_ok=True)

        n_all = len(self.all_spots) if self.all_spots is not None else 0
        n_good = len(self.good_spots) if self.good_spots is not None else 0

        lines = [
            f"FOV: {self.fov_id}",
            f"Total spots detected: {n_all}",
            f"Good spots (codebook matched): {n_good}",
            f"Match rate: {n_good / n_all:.4f}" if n_all > 0 else "Match rate: N/A",
        ]
        if self.good_spots is not None and "gene" in self.good_spots.columns:
            lines.append(
                f"Unique genes: {self.good_spots['gene'].nunique()}"
            )
        path.write_text("\n".join(lines) + "\n")
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

        Parameters
        ----------
        out_dir : Path | None
            Output directory for compressed subtile NPZs and coordinates CSV; None uses paths.subtile_dir.

        Returns
        -------
        pd.DataFrame
            Subtile table with t (one-based ID), scoords_x/y (one-based starts), ecoords_x/y (one-based inclusive ends). Also writes NPZs and subtile_coords.csv.

        Raises
        ------
        ValueError
            Dataset subtile configuration is absent. Compute its windows first.
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

            # Singleton Z is retained for both volume and projected images.
            arrays = {}
            for round_name, img in self.images.items():
                _validate_image(img)
                if not (0 <= window.y_start < window.y_end <= img.shape[1]
                        and 0 <= window.x_start < window.x_end <= img.shape[2]):
                    raise ValueError("subtile crop must lie within the image")
                arrays[f"images_{round_name}"] = img[:, sy, sx, ...]
                source = self.metadata.get(round_name, ImageMetadata(f"{self.fov_id}/{round_name}"))
                geometry = source.cropped((0, window.y_start, window.x_start),
                    frame_id=f"{source.frame_id}/subtile:{t + 1}")
                arrays[f"metadata_{round_name}"] = json.dumps(asdict(geometry))
                arrays[f"source_mapping_{round_name}"] = json.dumps({
                    "source_metadata": asdict(source),
                    "source_start_zyx": [0, window.y_start, window.x_start],
                })

            # Save NPZ — 1-based naming to match Snakemake wildcard {n_subtile}
            subtile_id = t + 1
            npz_path = out_dir / f"subtile_data_{subtile_id}.npz"
            np.savez_compressed(
                npz_path,
                **arrays,
                fov_id=self.fov_id,
                subtile_id=subtile_id,
                layers_seq=self.layers.seq,
                layers_ref=self.layers.ref,
            )

            # 1-based coordinates for stitch_subtile.py
            coord_rows.append(
                {
                    "t": subtile_id,
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
        """Load FOV state from a saved NPZ subtile.

        Parameters
        ----------
        subtile_path : Path
            Path to a trusted NPZ written by create_subtiles (loaded with allow_pickle=True).
        dataset : STARMapDataset
            Dataset providing shared configuration, layers and codebook.
        fov_id : str
            Field-of-view identifier used in output paths.

        Returns
        -------
        FOV
            New FOV with image arrays loaded; dataset state is supplied by caller. Geometry is restored; spots and shifts are not restored.
        """
        data = np.load(subtile_path, allow_pickle=True)

        fov = cls(dataset=dataset, fov_id=fov_id)
        if "subtile_id" not in data:
            raise ValueError("saved subtile requires subtile_id for spot namespace")
        fov.subtile_id = int(data["subtile_id"])
        for key in data.files:
            if key.startswith("images_"):
                round_name = key[len("images_") :]
                fov.images[round_name] = data[key]
                metadata_key = f"metadata_{round_name}"
                fov.metadata[round_name] = (ImageMetadata(**json.loads(str(data[metadata_key])))
                    if metadata_key in data else ImageMetadata(f"{fov_id}/{round_name}/unknown-subtile"))
                mapping_key = f"source_mapping_{round_name}"
                if mapping_key in data:
                    fov.load_diagnostics[round_name] = json.loads(str(data[mapping_key]))
        data.close()
        return fov
