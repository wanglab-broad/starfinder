"""FOV: per-FOV stateful processor with fluent API."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from contextlib import nullcontext
import json
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from starfinder.image import ImageMetadata, _validate_image
from starfinder.spot_finding import LocalMaximaConfig, SpotFindingResult
from starfinder.io import ImageCheckpoint, ImageLoadConfig
from starfinder.preprocessing import (MinMaxNormalizationConfig, HistogramMatchingConfig, ReconstructionConfig, TophatConfig, ProjectionConfig)
from starfinder.dataset._logging import _log_step
from starfinder.dataset._paths import _FovPaths
from starfinder.dataset.config import PipelineConfig, ExecutionConfig, RegistrationStep
from starfinder.registration import RegistrationResult
from starfinder.barcode import (Codebook, IntensityExtractionResult, NeighborhoodSumConfig,
    WtaDecoderConfig, ReadFilterConfig, BarcodeDecodingResult, ReadFilteringResult)

if TYPE_CHECKING:
    from starfinder.dataset.dataset import Dataset
    from starfinder.dataset.types import RoundState


@dataclass
class FOV:
    """Mutable per-FOV coordinator. Public operations own the algorithms.

    Stores round images/metadata, structured scientific results, ordered
    registration results and attempts. One instance per job; not thread-safe.
    """

    dataset: Dataset
    fov_id: str

    # Mutable state
    images: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, ImageMetadata] = field(default_factory=dict)
    registration_results: dict[str, list[RegistrationResult]] = field(default_factory=dict)
    registration_attempts: dict[str, list[dict]] = field(default_factory=dict)
    spot_result: SpotFindingResult | None = None
    subtile_id: int | None = None

    intensity_result: IntensityExtractionResult | None = None
    decoding_result: BarcodeDecodingResult | None = None
    filtering_result: ReadFilteringResult | None = None
    _round_intensities: dict = field(default_factory=dict)

    load_diagnostics: dict[str, dict] = field(default_factory=dict)
    image_checkpoint: ImageCheckpoint | None = None
    candidate_checkpoint_save: object | None = None
    decoded_checkpoint_path: Path | None = None
    final_checkpoint_path: Path | None = None

    # --- Delegated properties ---

    @property
    def rounds(self) -> RoundState:
        """Shared dataset RoundState.
        """
        return self.dataset.rounds

    @property
    def codebook(self) -> Codebook | None:
        """Shared barcode Codebook, or None before loading.
        """
        return self.dataset.codebook

    # --- Path helpers ---

    @property
    def paths(self) -> _FovPaths:
        """_FovPaths for this field of view; no directories are created.
        """
        return _FovPaths(self.dataset.output_root, self.fov_id)

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

    @_log_step
    def load_images(
        self,
        rounds: list[str] | None = None,
        channel_order: tuple[str, ...] | None = None,
        *,
        config: ImageLoadConfig | None = None,
        subdir: str = "",
        round_category: Literal["sequencing", "other"] = "sequencing",
    ) -> FOV:
        """Load raw TIFF stacks for specified rounds.

        Delegates to ``starfinder.io.load_round()`` per round.

        Parameters
        ----------
        rounds : list[str] | None
            Round names to load; None uses the dataset rounds selected by round_category.
        channel_order : tuple[str, ...] | None
            Filename channel patterns in output C order; None uses dataset.channel_order.
        config : ImageLoadConfig | None
            Explicit loading/conversion policy; None preserves source dtype.
        subdir : str
            Optional subdirectory beneath each round/FOV input directory.
        round_category : Literal['seq', 'other']
            Round category used when rounds is None: seq (default) or other.

        Returns
        -------
        FOV
            This instance, with processing state updated in place.
        """
        from starfinder.io import load_round

        if round_category not in ("sequencing", "other"):
            raise ValueError("invalid round_category")
        if config is not None and tuple(config.channel_labels) != self.dataset.channel_order:
            raise ValueError("load channel labels differ from dataset channel_order")
        if config is not None and (channel_order is not None or subdir):
            raise ValueError("select channels/subdir through config or arguments, not both")
        if rounds is None:
            rounds = (
                self.rounds.sequencing_rounds if round_category == "sequencing" else self.rounds.other_rounds
            )
        if channel_order is None:
            channel_order = self.dataset.channel_order

        if tuple(channel_order) != self.dataset.channel_order:
            raise ValueError("channel_order differs from dataset")
        if len(set(rounds)) != len(rounds) or not set(rounds) <= set(self.rounds.all_rounds):
            raise ValueError("load rounds must be unique configured rounds")
        for round_name in rounds:
            load_config = config or ImageLoadConfig(
                channel_labels=tuple(channel_order), subdir=subdir,
            )
            loaded = load_round(self.input_dir(round_name), config=load_config)
            self.images[round_name] = loaded.image
            self.metadata[round_name] = loaded.metadata
            self.load_diagnostics[round_name] = loaded.diagnostics
            recorder = getattr(self, '_provenance', None)
            if recorder:
                recorder._loaded_sources(round_name, loaded, load_config)
        return self

    # --- Preprocessing ---

    def load_image_checkpoint(self, path, *, require_registered=False, sha256=None) -> FOV:
        """Load an explicit prepared/registered artifact into an empty FOV.

        Identity, round order and sequencing labels must match this dataset.
        Registered sequencing geometry must agree; prepared frames may differ
        before explicit registration. All rounds must be present; partial artifacts are
        inspectable through ``io.load_image_checkpoint`` instead. No operation
        or transform is reapplied. Call downstream methods directly to continue
        from registered images; ``run`` remains an explicit full pipeline call.
        Existing images/results raise ValueError rather than being overwritten.
        """
        from starfinder.io import load_image_checkpoint

        if (self.images or self.metadata or self.registration_results or self.registration_attempts or
                self.spot_result is not None or self.intensity_result is not None or
                self.decoding_result is not None or self.filtering_result is not None or self._round_intensities):
            raise ValueError('load_image_checkpoint requires an empty FOV')
        saved = load_image_checkpoint(path, sha256=sha256)
        identity = saved.artifact
        if (identity['dataset_id'], identity['sample_id'], identity['FOV'], identity['subtile']) != (
                self.dataset.dataset_id, self.dataset.sample_id, self.fov_id, self.subtile_id):
            raise ValueError('checkpoint dataset/sample/FOV/subtile identity mismatch')
        if saved.rounds != self.rounds or {layer.round_label for layer in saved.layers} != set(self.rounds.all_rounds):
            raise ValueError('checkpoint rounds differ or are incomplete')
        if require_registered or saved.artifact['stage'] == 'registered_images':
            images = saved.sequencing_images(require_registered=True)
        else:
            images = {layer.round_label: layer.loaded for layer in saved.layers
                      if layer.round_label in self.rounds.sequencing_rounds}
            if any(loaded.image.ndim != 4 for loaded in images.values()):
                raise ValueError('FOV sequencing requires explicit ZYXC layers')
        if any(loaded.channel_labels != self.dataset.channel_order for loaded in images.values()):
            raise ValueError('checkpoint channel labels differ from dataset')
        if any(layer.processing.terminal_state in ('failed', 'skipped') for layer in saved.layers):
            raise ValueError('checkpoint has unavailable rounds')
        self.images = {layer.round_label: layer.loaded.image for layer in saved.layers}
        self.metadata = {layer.round_label: layer.loaded.metadata for layer in saved.layers}
        self.load_diagnostics = {layer.round_label: layer.loaded.diagnostics for layer in saved.layers}
        self.registration_results = {layer.round_label: list(layer.processing.registrations)
                                     for layer in saved.layers if layer.processing.registrations}
        self.registration_attempts = {layer.round_label: list(layer.processing.attempts)
                                      for layer in saved.layers if layer.processing.attempts}
        self.image_checkpoint = saved
        return self

    def _apply_to_rounds(self, func, rounds: list[str] | None) -> None:
        """Apply a function to images for the given rounds (or all)."""
        if rounds is None:
            rounds = self.rounds.all_rounds
        for name in rounds:
            self.images[name] = func(self.images[name])

    @_log_step
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

    @_log_step
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

    @_log_step
    def normalize_intensity(self, *, config=MinMaxNormalizationConfig('uint8', (0, 255)), rounds=None):
        """Normalize selected rounds with an explicit output policy."""
        from starfinder.preprocessing import normalize_intensity
        self._apply_to_rounds(lambda image: normalize_intensity(image, config=config), rounds)
        return self

    @_log_step
    def match_histogram(self, *, config=HistogramMatchingConfig(), reference_channel=0, rounds=None, reference=None):
        """Match selected rounds to one reference channel; retain its pre-match copy."""
        from starfinder.preprocessing import match_histogram
        if reference is None:
            reference = self.images[self.rounds.reference_round][..., reference_channel].copy()
        self._apply_to_rounds(lambda image: match_histogram(image, reference, config=config), rounds)
        return self

    @_log_step
    def reconstruct_background(self, *, config=ReconstructionConfig(), rounds=None):
        """Apply public reconstruction to selected rounds."""
        from starfinder.preprocessing import reconstruct_background
        self._apply_to_rounds(lambda image: reconstruct_background(image, config=config), rounds)
        return self

    @_log_step
    def filter_tophat(self, *, config=TophatConfig(), rounds=None):
        """Apply public tophat to selected rounds."""
        from starfinder.preprocessing import filter_tophat
        self._apply_to_rounds(lambda image: filter_tophat(image, config=config), rounds)
        return self

    @_log_step
    def project_image(self, *, config=ProjectionConfig(), rounds=None):
        """Project selected rounds, retaining singleton Z and source mapping."""
        from starfinder.preprocessing import project_image
        for name in list(self.images) if rounds is None else rounds:
            source_shape = self.images[name].shape[:3]
            self.images[name] = project_image(self.images[name], config=config)
            source = self.metadata.get(name, ImageMetadata(f"{self.fov_id}/{name}"))
            self.metadata[name] = source.projected(method=config.method)
            self.load_diagnostics.setdefault(name, {})['projection_source_metadata'] = asdict(source)
            self.load_diagnostics[name]['projection_source_shape_zyx'] = source_shape
        return self

    # --- Registration ---

    def _registration_image(self, name, mode, channel):
        image = _validate_image(self.images[name], ndim=(4,))
        if mode == 'merged':
            # Preserve signed/high-bit-depth input instead of overflowing uint16.
            return image.sum(axis=-1, dtype=np.float64)
        if channel >= image.shape[-1]:
            raise ValueError('registration channel outside image')
        return image[..., channel]

    @_log_step
    def register(self, step: RegistrationStep, *, rounds=None):
        """Estimate then apply; only opted-in estimation failures can recover.

        Ordered attempts include requested/actual method, effective config,
        outcome and failure. Apply errors propagate and are recorded too.
        """
        from starfinder.registration import estimate_transform, apply_transform
        recorder = getattr(self, '_provenance', None)
        step.__post_init__()
        ref = self.rounds.reference_round
        reference = self._registration_image(ref, step.reference_image, step.reference_channel)
        for name in self.rounds.moving_rounds if rounds is None else rounds:
            moving = self._registration_image(name, step.moving_image, step.reference_channel)
            configs = (step.config,) + (tuple(step.recovery.alternatives) if step.recovery else ())
            recovered_errors = []
            for index, config in enumerate(configs):
                attempt = dict(requested_method=step.config.method, actual_method=config.method,
                               config=asdict(config), failure=None, outcome='estimating')
                self.registration_attempts.setdefault(name, []).append(attempt)
                try:
                    context = recorder._operation('estimate_transform', config, round_name=name,
                        requested=step.config.method, actual=config.method) if recorder else nullcontext()
                    with context as diagnostics:
                        result = estimate_transform(reference, moving, config=config,
                            reference_metadata=self.metadata[ref], moving_metadata=self.metadata[name])
                        if recorder:
                            diagnostics['result'] = recorder._encode(result)
                except Exception as error:
                    attempt.update(outcome='failed', failure={'type': type(error).__name__, 'message': str(error)})
                    if step.recovery and isinstance(error, step.recovery.allowed_errors) and index + 1 < len(configs):
                        recovered_errors.append(error)
                        continue
                    raise
                try:
                    warp = step.warp or result.application_config
                    context = recorder._operation('apply_transform', warp, round_name=name,
                        requested=step.config.method, actual=config.method) if recorder else nullcontext()
                    with context:
                        registered = apply_transform(self.images[name], result.transform, config=warp)
                except Exception as error:
                    attempt.update(outcome='application_failed', failure={'type': type(error).__name__, 'message': str(error)})
                    raise
                attempt.update(outcome='succeeded', application_config=asdict(warp))
                self.images[name] = registered
                self.metadata[name] = result.transform.reference_metadata
                self.registration_results.setdefault(name, []).append(replace(result, application_config=warp))
                if recorder:
                    for error in recovered_errors:
                        recorder._recover(error, 'estimate_transform', round_name=name,
                            requested=step.config.method, actual=config.method)
                break
        return self

    def _save_shift_log(self):
        """Preserve MATLAB detected-displacement row/col/z columns."""
        from starfinder.registration import TranslationTransform
        rows = []
        for name, results in self.registration_results.items():
            for result in results:
                if isinstance(result.transform, TranslationTransform):
                    dz, dy, dx = (-v for v in result.transform.correction_zyx)
                    rows.append(dict(fov_id=self.fov_id, round=name, row=dy, col=dx, z=dz))
        path = self.paths.shift_log()
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows, columns=['fov_id', 'round', 'row', 'col', 'z']).to_csv(path, index=False)

    @_log_step
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
        metadata = self.metadata.get(self.rounds.reference_round, ImageMetadata(f"{self.fov_id}/{self.rounds.reference_round}"))
        self.spot_result = find_spots(self.images[self.rounds.reference_round], config=config,
                                     metadata=metadata, spot_namespace=namespace)
        return self

    @_log_step
    def _extract_round(self, round_name, config=NeighborhoodSumConfig()):
        from starfinder.barcode import extract_intensities
        from starfinder.io import ImageLoadResult
        loaded = ImageLoadResult(self.images[round_name],
            self.metadata.get(round_name, ImageMetadata(f"{self.fov_id}/{round_name}")),
            tuple(self.dataset.channel_order), (), {})
        self._round_intensities[round_name] = extract_intensities(
            {round_name: loaded}, self.spot_result,
            config=config)

    @_log_step
    def _assemble_intensities(self, rounds=None):
        rounds = self.rounds.sequencing_rounds if rounds is None else rounds
        results = [self._round_intensities[r] for r in rounds]
        first = results[0]
        if any(r.spot_ids != first.spot_ids or r.spot_namespace != first.spot_namespace or r.metadata != first.metadata or
               r.channel_labels != first.channel_labels or r.config != first.config or
               r.diagnostics['source_shape_zyx'] != first.diagnostics['source_shape_zyx'] for r in results):
            raise ValueError("inconsistent round extraction results")
        self.intensity_result = IntensityExtractionResult(
            np.concatenate([r.values for r in results], axis=2), first.spot_ids,
            first.spot_namespace, first.channel_labels, tuple(rounds), first.metadata,
            first.config, np.concatenate([r.valid for r in results], axis=1),
            {'rounds': {r: self._round_intensities[r].diagnostics for r in rounds}})
        self._round_intensities.clear()

    @_log_step
    def extract_intensities(self, *, config=NeighborhoodSumConfig(), rounds=None):
        """Extract labeled intensities; decoding is a separate reusable stage."""
        if not isinstance(config, NeighborhoodSumConfig):
            raise TypeError("config must be NeighborhoodSumConfig")
        config.__post_init__()
        rounds = self.rounds.sequencing_rounds if rounds is None else rounds
        if not rounds or len(set(rounds)) != len(rounds):
            raise ValueError("extraction rounds must be nonempty and unique")
        self._round_intensities.clear()
        for round_name in rounds:
            self._extract_round(round_name, config)
        self._assemble_intensities(rounds)
        return self

    @_log_step
    def decode_barcodes(self, *, config=WtaDecoderConfig(diagnostics=True)):
        """Decode the stored intensity result and retain every spot identity."""
        from starfinder.barcode import decode_barcodes
        if self.codebook is None:
            raise ValueError("Codebook not loaded. Call dataset.load_codebook() first.")
        self.decoding_result = decode_barcodes(self.intensity_result, self.codebook, config=config)
        return self

    @_log_step
    def filter_reads(self, *, config=ReadFilterConfig()):
        """Rerun explicit read predicates without decoding or image access."""
        from starfinder.barcode import filter_reads
        self.filtering_result = filter_reads(self.decoding_result, config=config)
        return self

    @_log_step
    def run(self, config: PipelineConfig, *, execution: ExecutionConfig = ExecutionConfig(),
            provenance=None):
        """Run one scientific sequence with batch or streaming residency.

        Reference first, then moving rounds in declared order. Histogram targets
        are captured before downstream reference processing. Loading may be
        disabled for resident/subtile data. Images without registration must
        already declare the same frame/grid for extraction.
        Optional ``provenance`` is a single-use RunRecorder writing to a fresh
        external directory, including failed/interrupted stages and diagnostics.
        Its provisional candidate/signal saving default runs after extraction,
        before decoding/QC; disable on the recorder explicitly. Without a
        recorder, results remain in-memory and no destination is inferred.
        """
        if not isinstance(config, PipelineConfig) or not isinstance(execution, ExecutionConfig):
            raise TypeError('run requires PipelineConfig and ExecutionConfig')
        config.__post_init__()
        execution.__post_init__()
        self.rounds.validate()
        ref = self.rounds.reference_round
        if ref is None:
            raise ValueError('reference_round is required')
        if config.extraction and not (config.detection or self.spot_result is not None):
            raise ValueError('extraction requires detections')
        if config.decoding and not (config.extraction or self.intensity_result is not None):
            raise ValueError('decoding requires intensities')
        if config.filtering and not (config.decoding or self.decoding_result is not None):
            raise ValueError('filtering requires decoding')
        if config.decoding and self.codebook is None:
            raise ValueError('decoding requires a loaded codebook')
        if config.load and execution.mode == 'batch':
            self.load_images(rounds=self.rounds.all_rounds, config=config.load)
        histogram_reference = None
        if config.extraction:
            self._round_intensities.clear()
        for name in [ref] + self.rounds.moving_rounds:
            if config.load and execution.mode == 'streaming':
                self.load_images(rounds=[name], config=config.load)
            if name not in self.images or name not in self.metadata:
                raise ValueError(f'missing image/metadata for {name}')
            if config.rotation_degrees is not None:
                self._rotate_round(name, config.rotation_degrees)
            if config.normalization:
                self.normalize_intensity(config=config.normalization, rounds=[name])
            if config.histogram:
                if name == ref:
                    histogram_reference = self.images[ref][..., config.histogram_reference_channel].copy()
                self.match_histogram(config=config.histogram, rounds=[name], reference=histogram_reference)
            if config.reconstruction and not config.reconstruction_after_registration:
                self.reconstruct_background(config=config.reconstruction, rounds=[name])
            if config.tophat:
                self.filter_tophat(config=config.tophat, rounds=[name])
            if config.projection:
                self.project_image(config=config.projection, rounds=[name])
            if name != ref:
                processed_reference = self.images[ref]
                if config.reconstruction and config.reconstruction_after_registration:
                    self.images[ref] = registration_reference
                try:
                    for step in config.registration:
                        self.register(step, rounds=[name])
                finally:
                    self.images[ref] = processed_reference
            if config.reconstruction and config.reconstruction_after_registration:
                # Keep the registration reference before the post-registration
                # operation; use its snapshot for each moving round below.
                if name == ref:
                    registration_reference = self.images[ref].copy()
                self.reconstruct_background(config=config.reconstruction, rounds=[name])
            if name == ref and config.detection:
                self.find_spots(config=config.detection)
            if config.extraction and name in self.rounds.sequencing_rounds:
                self._extract_round(name, config.extraction)
            if execution.mode == 'streaming' and not execution.retain_images and name != ref:
                del self.images[name]
        if config.extraction:
            self._assemble_intensities()
            if provenance is not None:
                provenance._save_candidates(self)
            else:
                from starfinder.io import CandidateSaveResult
                self.candidate_checkpoint_save = CandidateSaveResult(None, 0, 'no_persistent_run_destination')
        if config.decoding:
            self.decode_barcodes(config=config.decoding)
            if provenance is not None:
                provenance._save_molecular(self, 'decoded_pre_qc')
        if config.filtering:
            if provenance is not None and not config.decoding:
                provenance._save_molecular(self, 'decoded_pre_qc')
            self.filter_reads(config=config.filtering)
            if provenance is not None:
                provenance._save_molecular(self, 'final_accepted')
        return self

    # --- Output ---

    def save_reference_image(self, *, projection: ProjectionConfig | None = None) -> Path:
        """Save reference TIFF using the unchanged shared filename."""
        from starfinder.io import save_volume
        from starfinder.preprocessing import project_image
        ref = self.rounds.reference_round
        image, metadata = self.images[ref], self.metadata[ref]
        if projection is not None:
            image = project_image(image, config=projection)
            metadata = metadata.projected(method=projection.method)
        path = self.paths.ref_merged_tif
        path.parent.mkdir(parents=True, exist_ok=True)
        save_volume(image, path, metadata=metadata)
        return path

    def save_spots(self, slot='goodSpots', columns=None, *, path=None):
        """Export identities and coordinates once; shared slot filenames retained."""
        from starfinder.io import export_spots
        if slot not in ('goodSpots', 'allSpots'):
            raise ValueError('slot must be goodSpots or allSpots')
        reads = self.filtering_result if slot == 'goodSpots' else self.decoding_result
        return export_spots(self.spot_result, reads, path or self.paths.signal_csv(slot),
                            accepted_only=slot == 'goodSpots', columns=columns)

    def save_processing_log(self, log_type='rsf'):
        """Persist ordered registration attempts and stage counts."""
        if log_type not in ('rsf', 'gr'):
            raise ValueError('invalid log_type')
        path = self.paths.rsf_log() if log_type == 'rsf' else self.paths.gr_log()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(fov_id=self.fov_id, backend='python',
            rounds=asdict(self.rounds), registration_attempts=self.registration_attempts,
            detected=len(self.spot_result.spots) if self.spot_result is not None else None,
            filtering=self.filtering_result.counts if self.filtering_result is not None else None), indent=2))
        self._save_shift_log()
        return path

    def save_diagnostics(self, suffix=''):
        """Write counts, undefined fractions and explicit registration attempts."""
        path = self.paths.score_log(suffix)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(
            detected=len(self.spot_result.spots) if self.spot_result is not None else None,
            counts=self.filtering_result.counts if self.filtering_result is not None else None,
            fractions=self.filtering_result.fractions if self.filtering_result is not None else None,
            registration_attempts=self.registration_attempts), indent=2))
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
                layers_seq=self.rounds.sequencing_rounds,
                layers_ref=self.rounds.reference_round,
                other_rounds=self.rounds.other_rounds,
                dataset_id=self.dataset.dataset_id, sample_id=self.dataset.sample_id,
                channel_labels=self.dataset.channel_order,
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
        dataset: Dataset,
        fov_id: str,
    ) -> FOV:
        """Load FOV state from a saved NPZ subtile.

        Parameters
        ----------
        subtile_path : Path
            Path to a trusted NPZ written by create_subtiles (loaded with allow_pickle=True).
        dataset : Dataset
            Dataset providing shared configuration, rounds and codebook.
        fov_id : str
            Field-of-view identifier used in output paths.

        Returns
        -------
        FOV
            New FOV with image arrays loaded; dataset state is supplied by caller. Geometry is restored; spots and shifts are not restored.
        """
        data = np.load(subtile_path, allow_pickle=True)

        if str(data["fov_id"]) != fov_id or str(data["dataset_id"]) != dataset.dataset_id or str(data["sample_id"]) != dataset.sample_id:
            data.close()
            raise ValueError("subtile identity differs from dataset/FOV")
        if (list(data["layers_seq"]) != dataset.rounds.sequencing_rounds or
            str(data["layers_ref"]) != dataset.rounds.reference_round or
            list(data["other_rounds"]) != dataset.rounds.other_rounds or
            tuple(data["channel_labels"]) != dataset.channel_order):
            data.close()
            raise ValueError("subtile round/channel labels differ from dataset")
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
