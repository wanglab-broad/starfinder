"""FOV: per-FOV stateful processor with fluent API."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from starfinder.image import ImageMetadata, _validate_image
from starfinder.spot_finding import LocalMaximaConfig, SpotFindingPlan, SpotFindingResult
from starfinder.spot_finding._methods import SpotFindingConfig
from starfinder.io import ImageLoadConfig
from starfinder.preprocessing import (MinMaxNormalizationConfig, HistogramMatchingConfig, ReconstructionConfig, TophatConfig, ProjectionConfig,
    StepContext, read_supplied_statistics, run_step, step_spec)
from starfinder.dataset._logging import _log_step
from starfinder.dataset._paths import _FovPaths
from starfinder._registry import _version
from starfinder.dataset.config import (CheckpointConfig, PipelineConfig, ExecutionConfig, ExternalReference,
    RegistrationRecipe)
from starfinder.registration import (RegistrationRejectedError, RegistrationResult, TransformChain,
    TranslationTransform, WarpConfig)
from starfinder.barcode import (DECODING_METHODS, Codebook, IntensityExtractionResult, NeighborhoodSumConfig,
    WtaDecoderConfig, ReadFilterConfig, BarcodeDecodingResult, ReadFilteringResult, ReadScoreConfig,
    ReadScoringResult, DeduplicationConfig, ReadDeduplicationResult)
from starfinder.barcode.decoding import _mode_mismatch

# Raised in readout mode multiplexed when a candidate set with a round column would be decoded as barcodes.
MULTI_ROUND_DECODING = ("decoding candidates from several detection rounds needs a readout mode (§2.8): "
                        "set readout_mode='direct' on the dataset for direct readout")
# Raised in readout mode direct when the candidates have no round column.
DIRECT_NEEDS_ROUNDS = ("readout_mode='direct' assigns each candidate from its own round and needs candidates "
                       "with a round column: detect with a SpotFindingPlan with explicit rounds")

if TYPE_CHECKING:
    from starfinder.dataset.dataset import Dataset
    from starfinder.dataset.types import RoundState
    from starfinder.segmentation import ReferenceGrid, SegmentationPlan, SegmentationResult


def _recipe_record(recipe):
    """The recorded summary of a registration recipe: step method names, signals, warp, reference and QC."""
    from starfinder.registration import REGISTRATION_METHODS
    return dict(steps=[REGISTRATION_METHODS[type(step.config)].name for step in recipe.steps],
                signal=asdict(recipe.signal),
                step_signals=[None if step.signal is None else asdict(step.signal) for step in recipe.steps],
                warp=None if recipe.warp is None else asdict(recipe.warp), reference_round=recipe.reference_round,
                qc=asdict(recipe.qc))


def _rotated(volume, source, angle):
    """(image, metadata, diagnostics) of a ZYX(C) volume rotated by angle degrees in YX.

    Exact 90-degree multiples use np.rot90 and a contiguous copy; other angles
    scipy.ndimage.rotate with bilinear interpolation, keeping the shape and dtype.
    """
    vol = _validate_image(volume)
    metadata = source.rotated(vol.shape[:3], angle, frame_id=f"{source.frame_id}/rotate:{angle}")
    k_90 = round(angle / 90)
    if abs(angle - k_90 * 90) < 1e-6:
        # np.rot90 and imrotate share sign convention: k=1 is CCW, k=-1 is CW
        image = np.ascontiguousarray(np.rot90(vol, k=k_90, axes=(1, 2)))
    else:
        from scipy.ndimage import rotate as ndimage_rotate
        image = ndimage_rotate(vol, angle, axes=(1, 2), reshape=False, order=1).astype(vol.dtype)
    return image, metadata, {
        "source_metadata": asdict(source), "source_shape_zyx": vol.shape[:3],
        "output_shape_zyx": image.shape[:3], "angle_degrees": angle,
        "mapping": "source = R @ (output - output_center) + source_center",
    }


def _default_warp(chain):
    """The derived final resampling: translation for a chain of translations, else linear SciPy."""
    return WarpConfig() if chain.translation() is not None else WarpConfig(backend='scipy')


def _signal_warp(chain, warp):
    """Float64 resampling of a step signal through chain with the recipe's interpolation and boundary.

    A chain of translations under constant zero fill uses the translation
    path; otherwise the recipe's dense backend (SciPy when warp is None or
    the translation backend). Nothing is rounded.
    """
    base = warp if warp is not None and warp.backend != 'translation' else WarpConfig(backend='scipy')
    if chain.translation() is not None and base.boundary_mode == 'constant' and base.fill_value == 0:
        workers = warp.fft_workers if warp is not None and warp.backend == 'translation' else 1
        return WarpConfig(fft_workers=workers, output_dtype='float64')
    return replace(base, output_dtype='float64')


def _qc_record(qc):
    """A registration_qc result for the attempt records, in the form a registered checkpoint restores.

    Sequences are tuples and undefined floats None; projections are not kept.
    """
    from starfinder.io._checkpoint import _jsonable, _tuples
    return _tuples(_jsonable(dict(status=qc.status, values=qc.values, units=qc.units, counts=qc.counts,
                                  reasons=qc.reasons, config=qc.config,
                                  details={k: v for k, v in qc.details.items() if k != 'projections'})))


def _check_qc(qc, config, transform):
    """Raise RegistrationRejectedError for the first configured criterion the step fails.

    An undefined value (for example an undefined NCC gain) rejects nothing;
    the QC record keeps its reason.
    """
    checks = [('min_coverage', qc.values['coverage'], config.min_coverage, lambda v, b: v < b),
              ('min_ncc_gain', qc.values['ncc_gain'], config.min_ncc_gain, lambda v, b: v < b),
              ('max_fold_fraction', qc.details['transform'].get('fold_fraction'), config.max_fold_fraction,
               lambda v, b: v > b),
              ('max_translation_voxels',
               float(np.linalg.norm(transform.displacement_zyx)) if isinstance(transform, TranslationTransform) else None,
               config.max_translation_voxels, lambda v, b: v > b)]
    for criterion, value, bound, fails in checks:
        if bound is not None and value is not None and fails(value, bound):
            relation = 'below' if criterion.startswith('min_') else 'above'
            error = RegistrationRejectedError(f'{criterion}: {value!r} is {relation} the bound {bound!r}')
            error.criterion = criterion
            raise error


@dataclass
class FOV:
    """Mutable per-FOV coordinator. Public operations own the algorithms.

    Stores round images/metadata, recipe snapshots (per round, snapshot name
    to image, in the round's current coordinates), structured scientific
    results, ordered registration results and attempts, the TransformChain
    applied to each moving round, registration_record (semantics
    ``recipe``, or ``sequential`` for a loaded version-1 checkpoint; the
    recipe summary; the WarpConfig applied per round), and
    preprocessing_record (the recipe, per-round step records and, per round
    and snapshot, the transforms composed, of the last run with a recipe),
    and segmentation_results (run name to SegmentationResult, from segment()).
    One instance per job; not thread-safe.
    The repr summarizes image geometry, channels and completed stages without
    array values or table rows; results maps completed stages by name.
    """

    dataset: Dataset
    fov_id: str

    # Mutable state
    images: dict[str, np.ndarray] = field(default_factory=dict)
    snapshots: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    metadata: dict[str, ImageMetadata] = field(default_factory=dict)
    registration_results: dict[str, list[RegistrationResult]] = field(default_factory=dict)
    registration_attempts: dict[str, list[dict]] = field(default_factory=dict)
    registration_chains: dict[str, TransformChain] = field(default_factory=dict)
    registration_record: dict = field(default_factory=dict)
    spot_result: SpotFindingResult | None = None
    subtile_id: int | None = None

    intensity_result: IntensityExtractionResult | None = None
    decoding_result: BarcodeDecodingResult | None = None
    filtering_result: ReadFilteringResult | None = None
    scoring_result: ReadScoringResult | None = None
    deduplication_result: ReadDeduplicationResult | None = None
    _round_intensities: dict = field(default_factory=dict)

    load_diagnostics: dict[str, dict] = field(default_factory=dict)
    preprocessing_record: dict = field(default_factory=dict)
    segmentation_results: dict[str, SegmentationResult] = field(default_factory=dict)
    _run_record: object | None = field(default=None, init=False, repr=False, compare=False)

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

    def encoding_table(self) -> pd.DataFrame:
        """The barcode encoding table of the dataset codebook (:meth:`Dataset.encoding_table`).

        Raises
        ------
        ValueError
            readout_mode is ``direct`` or no codebook is loaded.
        """
        return self.dataset.encoding_table()

    # --- Summaries ---

    @property
    def results(self) -> Mapping[str, object]:
        """Read-only mapping of the stages that have run, in pipeline order.

        Keys are ``registration``, ``spot_finding``, ``extraction``,
        ``decoding``, ``scoring``, ``deduplication`` and ``filtering``; a stage
        that has not run is absent. Values are the stored result objects:
        ``registration`` is a read-only view of registration_results (round
        label to that round's ordered RegistrationResult list), and the others
        are spot_result, intensity_result, decoding_result, scoring_result,
        deduplication_result and filtering_result. The read diagnostics
        (summarize_reads, explain_read) accept this mapping.
        """
        stages = {}
        if self.registration_results:
            stages["registration"] = MappingProxyType(self.registration_results)
        for name, result in (("spot_finding", self.spot_result), ("extraction", self.intensity_result),
                             ("decoding", self.decoding_result), ("scoring", self.scoring_result),
                             ("deduplication", self.deduplication_result), ("filtering", self.filtering_result)):
            if result is not None:
                stages[name] = result
        return MappingProxyType(stages)

    def _images_summary(self) -> str:
        ref = self.rounds.reference_round
        configured = self.rounds.all_rounds
        loaded = [r for r in configured if r in self.images] + [r for r in self.images if r not in configured]
        missing = [r for r in configured if r not in self.images]

        def mark(rounds):
            return ", ".join(r + "*" if r == ref else r for r in rounds)

        def geometry(image):
            axes = {3: "ZYX", 4: "ZYXC"}.get(image.ndim, f"{image.ndim}D")
            return f"{tuple(image.shape)} {image.dtype} {axes}"

        if not loaded:
            text = f"not loaded ({mark(configured)})" if configured else "not loaded"
            shown = configured
        else:
            geometries = [geometry(self.images[r]) for r in loaded]
            if len(set(geometries)) == 1:
                text = f"{mark(loaded)} — {geometries[0]}"
            else:
                text = ", ".join(f"{mark([r])} {g}" for r, g in zip(loaded, geometries))
            if missing:
                text += f"; not loaded: {mark(missing)}"
            shown = loaded + missing
        return text + ("   (* reference)" if ref in shown else "")

    def __repr__(self):
        ds = self.dataset
        subtile = "" if self.subtile_id is None else f" subtile {self.subtile_id}"
        results = self.results
        lines = [
            f"FOV {self.fov_id!r}{subtile} of Dataset {ds.dataset_id!r} (sample {ds.sample_id!r})",
            f"    images:   {self._images_summary()}",
            f"    channels: {', '.join(ds.channel_order) or 'none'}",
            f"    results:  {', '.join(results) or 'none'}",
        ]
        for name, result in results.items():
            if name == "registration":
                methods = dict.fromkeys(r.diagnostics.method for rs in result.values() for r in rs)
                n = len(result)
                summary = f"{n} moving round{'' if n == 1 else 's'}, {', '.join(methods)}"
            else:
                summary = result._summary()
            lines.append(f"      {name:<14}{summary}")
        return "\n".join(lines)

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
            Filename channel patterns in output C order; None uses each
            round's dataset.channel_labels (dataset.channel_order, or an other
            round's own labels). When given, it must equal them.
        config : ImageLoadConfig | None
            Explicit loading/conversion policy; None preserves source dtype.
            Its channel_labels must equal each loaded round's labels.
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
        if config is not None and (channel_order is not None or subdir):
            raise ValueError("select channels/subdir through config or arguments, not both")
        if rounds is None:
            rounds = (
                self.rounds.sequencing_rounds if round_category == "sequencing" else self.rounds.other_rounds
            )
        if len(set(rounds)) != len(rounds) or not set(rounds) <= set(self.rounds.all_rounds):
            raise ValueError("load rounds must be unique configured rounds")
        for round_name in rounds:
            labels = self.dataset.channel_labels(round_name)
            if config is not None and tuple(config.channel_labels) != labels:
                raise ValueError("load channel labels differ from dataset channel_order")
            if channel_order is not None and tuple(channel_order) != labels:
                raise ValueError("channel_order differs from dataset")
        for round_name in rounds:
            load_config = config or ImageLoadConfig(
                channel_labels=self.dataset.channel_labels(round_name), subdir=subdir,
            )
            loaded = load_round(self.input_dir(round_name), config=load_config)
            self.images[round_name] = loaded.image
            self.metadata[round_name] = loaded.metadata
            self.load_diagnostics[round_name] = loaded.diagnostics
            if self._run_record is not None:
                self._run_record.add_inputs(loaded.source_paths)
        return self

    # --- Preprocessing ---

    def _apply_to_rounds(self, func, rounds: list[str] | None) -> None:
        """Apply a function to images for the given rounds (or all)."""
        if rounds is None:
            rounds = self.rounds.all_rounds
        for name in rounds:
            self.images[name] = func(self.images[name])

    @_log_step
    def _rotate_round(self, round_name: str, angle: float) -> None:
        """Rotate a single round's image in-place."""
        source = self.metadata.get(round_name, ImageMetadata(f"{self.fov_id}/{round_name}"))
        self.images[round_name], self.metadata[round_name], diagnostics = _rotated(
            self.images[round_name], source, angle)
        self.load_diagnostics.setdefault(round_name, {})["rotation"] = diagnostics

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
            self._rotate_round(round_name=round_name, angle=angle)
        return self

    @_log_step
    def normalize_intensity(self, *, config=MinMaxNormalizationConfig('uint8', (0, 255)), rounds=None):
        """Normalize selected rounds with an explicit output policy."""
        from starfinder.preprocessing import normalize_intensity
        self._apply_to_rounds(lambda image: normalize_intensity(image, config=config), rounds)
        return self

    @_log_step
    def match_histogram(self, *, config=HistogramMatchingConfig(), rounds=None, reference=None):
        """Match selected rounds to one reference channel; retain its pre-match copy.

        None uses channel config.reference_channel of the reference round.
        """
        from starfinder.preprocessing import match_histogram
        if reference is None:
            reference = self.images[self.rounds.reference_round][..., config.reference_channel].copy()
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
        if config.axis != 'z':
            raise ValueError('FOV rounds stay ZYXC; only a z projection applies to them')
        for name in list(self.images) if rounds is None else rounds:
            source_shape = self.images[name].shape[:3]
            self.images[name] = project_image(self.images[name], config=config)
            source = self.metadata.get(name, ImageMetadata(f"{self.fov_id}/{name}"))
            self.metadata[name] = source.projected(method=config.method)
            self.load_diagnostics.setdefault(name, {})['projection_source_metadata'] = asdict(source)
            self.load_diagnostics[name]['projection_source_shape_zyx'] = source_shape
        return self

    @_log_step
    def _preprocess(self, config, *, round_name, index, phase, stage, references, supplied=None, save_as=None):
        """Run one recipe step on one round through the enforcement wrapper and record it.

        stage is the step name, so a failure names the step in the run record.
        supplied is the validated supplied-statistics document; a step with
        fit="supplied" receives its section and requires the file's dtype.
        save_as keeps the output as that snapshot of the round.
        """
        spec = step_spec(config)
        ref = self.rounds.reference_round
        reference = section = None
        if getattr(config, 'fit', None) == 'supplied':
            section = supplied['steps'][stage]
            dtype = np.asarray(self.images[round_name]).dtype
            if dtype.name != supplied['dtype']:
                raise ValueError(f'step {stage!r} input is {dtype}, but the supplied statistics are for {supplied["dtype"]}')
        elif spec.scope == 'needs_reference':
            key = (phase, index)
            if key not in references:
                if round_name != ref:
                    raise ValueError(f'step {stage!r} needs the reference round {ref!r} to pass it first')
                image = self.images[ref]
                channel = config.reference_channel
                if image.ndim != 4 or channel >= image.shape[-1]:
                    raise ValueError(f'step {stage!r} reference_channel {channel} is outside the reference image')
                references[key] = image[..., channel].copy()
            reference = references[key]
        image = self.images[round_name]
        metadata = self.metadata.get(round_name, ImageMetadata(f"{self.fov_id}/{round_name}"))
        result = run_step(image, config, StepContext(round_name, ref, metadata, reference, section))
        self.preprocessing_record.setdefault('rounds', {}).setdefault(round_name, []).append(dict(
            index=index, stage=phase, step=stage, config=asdict(config), fitted=dict(result.fitted),
            diagnostics=dict(result.diagnostics), input_dtype=str(np.asarray(image).dtype),
            output_dtype=str(result.image.dtype), save_as=save_as))
        self.images[round_name] = np.asarray(result.image)
        if save_as is not None:
            # Steps never mutate their input, so the snapshot can share the array.
            self.snapshots.setdefault(round_name, {})[save_as] = self.images[round_name]

    # --- Registration ---

    def _signal_channel(self, name, signal, role):
        """The channel index that mode "channel" of signal selects in round name; None for other modes.

        role is "reference" or "moving" (moving_channel, None: reference_channel).
        A label is looked up in the round's channel labels: dataset.channel_order,
        or an other round's own labels in dataset.other_channel_order. An
        unknown label or an index outside the round's image is a ValueError.
        """
        if signal.mode != 'channel':
            return None
        channel = signal.moving_channel if role == 'moving' and signal.moving_channel is not None else signal.reference_channel
        if isinstance(channel, str):
            labels = self.dataset.other_channel_order.get(name, self.dataset.channel_order)
            if channel not in labels:
                raise ValueError(f'registration channel {channel!r} is not a channel label of round {name!r}')
            channel = tuple(labels).index(channel)
        if channel >= np.shape(self.images[name])[-1]:
            raise ValueError(f'registration channel {channel} is outside the image of round {name!r}')
        return channel

    def _registration_image(self, name, signal, role, source=None):
        """The float64 ZYX registration signal of round name (RegistrationSignalConfig signal).

        role is "reference" or "moving" and selects the channel of mode
        "channel" (see _signal_channel).
        """
        if source is not None and source not in self.snapshots.get(name, {}):
            raise ValueError(f'registration source {source!r} is not a snapshot of round {name!r}')
        image = _validate_image(self.images[name] if source is None else self.snapshots[name][source], ndim=(4,))
        if signal.mode == 'max':
            return image.max(axis=-1).astype(np.float64)
        if signal.mode == 'sum':
            # Preserve signed/high-bit-depth input instead of overflowing the input dtype.
            return image.sum(axis=-1, dtype=np.float64)
        channel = self._signal_channel(name, signal, role)
        if channel >= image.shape[-1]:
            raise ValueError(f'registration channel {channel} is outside the image of round {name!r}')
        return image[..., channel].astype(np.float64)

    @_log_step
    def register(self, recipe: RegistrationRecipe, *, rounds=None, source: str | None = None):
        """Estimate the recipe's steps per moving round, then resample the round once.

        Signals are built with the recipe's (or a step's own) signal from
        snapshot source of the reference and each moving round (None: their
        images). Step k is estimated on the moving signal resampled in float64
        through steps 1 to k-1; each successful estimate is checked by
        registration_qc against recipe.qc, and a failed criterion raises
        RegistrationRejectedError. Only opted-in estimation failures recover.
        The step transforms compose into one TransformChain, and the moving
        round's image and every one of its snapshots are resampled once from
        their pre-registration arrays with the recipe's warp (None: derived
        from the chain), so they stay aligned; the reference round is not
        transformed. registration_attempts gets one estimation entry per
        attempt and one application entry per round; registration_results
        the step results (application_config: the round's WarpConfig),
        registration_chains the chain and registration_record the recipe and
        the application policies. Apply errors propagate and are recorded
        too, never recovered. A round is registered at most once. During a
        preprocessing recipe run, preprocessing_record lists the composed
        results per round and snapshot.
        """
        if not isinstance(recipe, RegistrationRecipe):
            raise TypeError('register requires a RegistrationRecipe')
        recipe.__post_init__()
        ref = self.rounds.reference_round
        if recipe.reference_round is not None and recipe.reference_round != ref:
            raise ValueError(f'recipe reference round {recipe.reference_round!r} differs from the dataset reference {ref!r}')
        summary = _recipe_record(recipe)
        if self.registration_record.get('semantics') == 'sequential':
            raise ValueError('this FOV holds a sequential (version-1) registration; it cannot be extended by a recipe')
        if self.registration_record.get('recipe') not in (None, summary):
            raise ValueError('this FOV was registered with another registration recipe')
        rounds = self.rounds.moving_rounds if rounds is None else rounds
        registered = [name for name in rounds if self.registration_results.get(name) or name in self.registration_chains]
        if registered:
            raise ValueError(f'rounds {registered} are already registered; a round is registered once')
        signals = list(dict.fromkeys(step.signal or recipe.signal for step in recipe.steps)) + [recipe.signal]
        references = {signal: self._registration_image(ref, signal, 'reference', source) for signal in dict.fromkeys(signals)}
        self.registration_record = dict(self.registration_record, semantics='recipe', recipe=summary)
        self.registration_record.setdefault('application', {})
        for name in rounds:
            self._register_round(recipe, name, source, references)
        return self

    @_log_step
    def register_rounds(self, recipe: RegistrationRecipe, *, rounds: Sequence[str],
                        reference: str | ExternalReference | None = None) -> FOV:
        """Register loaded rounds, such as morphology rounds, to a reference round or an external reference.

        The typical recipe signal is ``mode="channel"`` naming a shared stain
        in each round: reference_channel in the reference round and
        moving_channel in every moving round, by index or by the round's own
        channel labels (dataset.channel_labels). Any allowed step sequence
        works; MATLAB nuclei_registration corresponds to one translation step.
        Label and grid errors are raised before any estimator runs.

        Parameters
        ----------
        recipe : RegistrationRecipe
            Steps, signal, warp and QC, as in register.
        rounds : Sequence[str]
            Loaded rounds to register, each at most once.
        reference : str | ExternalReference | None
            None uses recipe.reference_round (default: the dataset reference
            round). A round name may be any loaded round, other rounds
            included. An ExternalReference supplies the reference signal
            directly (its image, whatever the signal mode) and must have the
            moving rounds' grid; the SHA-256 of its image's C-order bytes is
            recorded.

        Returns
        -------
        FOV
            This instance. Each round is registered as in register: the steps
            compose into one TransformChain estimated on the stain, and the
            round's image (every channel) and each of its snapshots are
            resampled once through it. Every estimation attempt records the
            reference (round name or external label) and reference_sha256
            (None for a round); registration_record["rounds"][round] keeps the
            recipe summary and the reference.

        Raises
        ------
        ValueError
            An unknown channel label or a channel index outside a round, a
            round that is not loaded or is already registered, or a reference
            that differs from recipe.reference_round.
        IncompatibleGeometryError
            A moving round and the reference are not on one grid.
        """
        from starfinder.registration._types import _geometry
        if not isinstance(recipe, RegistrationRecipe):
            raise TypeError('register_rounds requires a RegistrationRecipe')
        recipe.__post_init__()
        if isinstance(rounds, str) or not isinstance(rounds, Sequence) or not rounds:
            raise ValueError('rounds must be a nonempty sequence of round names')
        rounds = list(rounds)
        if len(set(rounds)) != len(rounds):
            raise ValueError('rounds must be unique')
        missing = [name for name in rounds if name not in self.images or name not in self.metadata]
        if missing:
            raise ValueError(f'rounds {missing} are not loaded')
        external = isinstance(reference, ExternalReference)
        if external:
            if recipe.reference_round is not None:
                raise ValueError('an external reference replaces the recipe reference_round, which must be None')
            reference.__post_init__()
            reference_shape, reference_metadata = reference.image.shape, reference.metadata
        else:
            if reference is None:
                reference = recipe.reference_round or self.rounds.reference_round
            elif not isinstance(reference, str):
                raise TypeError('reference must be a round name, an ExternalReference or None')
            if recipe.reference_round not in (None, reference):
                raise ValueError(f'reference {reference!r} differs from the recipe reference round {recipe.reference_round!r}')
            if reference not in self.images or reference not in self.metadata:
                raise ValueError(f'reference round {reference!r} is not loaded')
            if reference in rounds:
                raise ValueError(f'the reference round {reference!r} cannot be registered to itself')
            reference_shape, reference_metadata = self.images[reference].shape[:3], self.metadata[reference]
        if self.registration_record.get('semantics') == 'sequential':
            raise ValueError('this FOV holds a sequential (version-1) registration; it cannot be extended by a recipe')
        registered = [name for name in rounds if self.registration_results.get(name) or name in self.registration_chains]
        if registered:
            raise ValueError(f'rounds {registered} are already registered; a round is registered once')
        signals = list(dict.fromkeys([step.signal or recipe.signal for step in recipe.steps] + [recipe.signal]))
        # Every label and grid error is raised here, before any estimator runs.
        for name in rounds:
            _geometry(reference_shape, _validate_image(self.images[name], ndim=(4,)).shape[:3],
                      reference_metadata, self.metadata[name])
            for signal in signals:
                self._signal_channel(name, signal, 'moving')
        if external:
            image = np.asarray(reference.image)
            described = (reference.label, hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest(),
                         reference.metadata)
            signal_image = image.astype(np.float64)
            references = {signal: signal_image for signal in signals}
        else:
            described = (reference, None, reference_metadata)
            references = {signal: self._registration_image(reference, signal, 'reference') for signal in signals}
        summary = _recipe_record(recipe)
        self.registration_record = dict(self.registration_record, semantics='recipe')
        self.registration_record.setdefault('application', {})
        self.registration_record.setdefault('rounds', {})
        for name in rounds:
            self._register_round(recipe, name, None, references, described)
            self.registration_record['rounds'][name] = dict(recipe=summary, reference=described[0],
                                                            reference_sha256=described[1])
        return self

    def _register_round(self, recipe, name, source, references, described=None):
        """Estimate every step of recipe for round name, then resample its images once.

        described is (round name or external label, SHA-256 or None, metadata)
        of the reference signal; None is the dataset reference round.
        """
        from starfinder import registration
        from starfinder.evaluation.registration import registration_qc
        from starfinder.registration._chain import resample
        ref = self.rounds.reference_round
        ref, ref_sha256, ref_metadata = described or (ref, None, self.metadata[ref])
        originals = {signal: self._registration_image(name, signal, 'moving', source) for signal in references}
        attempts = self.registration_attempts.setdefault(name, [])
        results = []
        # The moving signal of each signal config at the start of the current step.
        current = dict(originals)
        for index, step in enumerate(recipe.steps):
            signal = step.signal or recipe.signal
            reference, before = references[signal], current[signal]
            moving_metadata = self.metadata[name] if not results else results[-1].transform.reference_metadata
            configs = (step.config,) + (tuple(step.recovery.alternatives) if step.recovery else ())
            for number, config in enumerate(configs):
                spec = registration.REGISTRATION_METHODS[type(config)]
                attempt = dict(record='estimation', step=index, attempt=number, requested_method=step.config.method,
                               actual_method=config.method, fallback=number > 0, backend=None,
                               backend_versions={d.distribution: _version(d.distribution) for d in spec.requires},
                               reference=ref, reference_sha256=ref_sha256, config=asdict(config),
                               outcome='estimating', failure=None, qc=None)
                attempts.append(attempt)
                try:
                    result = registration.estimate_transform(reference, before, config=config,
                        reference_metadata=ref_metadata, moving_metadata=moving_metadata)
                    attempt.update(backend=result.diagnostics.backend)
                    if result.diagnostics.backend_versions is not None:
                        attempt.update(backend_versions=dict(result.diagnostics.backend_versions))
                    chain = TransformChain(tuple(r.transform for r in results) + (result.transform,))
                    after = resample([originals[signal]], chain, _signal_warp(chain, recipe.warp))[0]
                    qc = registration_qc(reference, before, after, result.transform, config=recipe.qc,
                                         diagnostics=result.diagnostics)
                    attempt.update(qc=_qc_record(qc))
                    _check_qc(qc, recipe.qc, result.transform)
                except Exception as error:
                    failure = {'type': type(error).__name__, 'message': str(error)}
                    if isinstance(error, RegistrationRejectedError):
                        failure['criterion'] = getattr(error, 'criterion', None)
                    attempt.update(outcome='rejected' if isinstance(error, RegistrationRejectedError) else 'failed',
                                   failure=failure)
                    if step.recovery and isinstance(error, step.recovery.allowed_errors) and number + 1 < len(configs):
                        continue
                    raise
                attempt.update(outcome='succeeded')
                results.append(result)
                current = {key: after if key == signal else None for key in current}
                break
            if index + 1 < len(recipe.steps):
                chain = TransformChain(tuple(r.transform for r in results))
                current = {key: value if value is not None else
                           resample([originals[key]], chain, _signal_warp(chain, recipe.warp))[0]
                           for key, value in current.items()}
        chain = TransformChain(tuple(r.transform for r in results))
        warp = recipe.warp or _default_warp(chain)
        application = dict(record='application', outcome='applying', application_config=asdict(warp), failure=None,
                           qc=None)
        attempts.append(application)
        keys = list(self.snapshots.get(name, {}))
        try:
            outputs = resample([self.images[name], *(self.snapshots[name][key] for key in keys)], chain, warp)
        except Exception as error:
            application.update(outcome='application_failed', failure={'type': type(error).__name__, 'message': str(error)})
            raise
        after = current[recipe.signal]
        if after is None:
            after = resample([originals[recipe.signal]], chain, _signal_warp(chain, recipe.warp))[0]
        application.update(outcome='succeeded', qc=_qc_record(registration_qc(
            references[recipe.signal], originals[recipe.signal], after, chain, config=recipe.qc)))
        self.images[name] = outputs[0]
        if keys:
            self.snapshots[name] = dict(zip(keys, outputs[1:]))
        self.metadata[name] = chain.reference_metadata
        self.registration_results[name] = [replace(result, application_config=warp) for result in results]
        self.registration_chains[name] = chain
        self.registration_record['application'][name] = warp
        self._record_transforms(name)

    def _record_transforms(self, name):
        """List the composed results of round name for each of its images in the recipe record.

        result indexes registration_results[name] (and the round's transforms.json entries).
        """
        from starfinder.registration._chain import transform_kind
        if 'recipe' not in self.preprocessing_record:
            return
        entries = [dict(result=i, method=r.diagnostics.method, kind=transform_kind(r.transform))
                   for i, r in enumerate(self.registration_results[name])]
        applied = self.preprocessing_record.setdefault('transforms', {}).setdefault(name, {})
        for key in ('detection', *self.snapshots.get(name, {})):
            applied.setdefault(key, []).extend(dict(entry) for entry in entries)

    def _save_shift_log(self, suffix='', rounds=None):
        """Preserve MATLAB detected-displacement row/col/z columns (rounds None: every registered round)."""
        from starfinder.registration import TranslationTransform
        rows = []
        for name, results in self.registration_results.items():
            if rounds is not None and name not in rounds:
                continue
            for result in results:
                if isinstance(result.transform, TranslationTransform):
                    dz, dy, dx = result.transform.displacement_zyx
                    rows.append(dict(fov_id=self.fov_id, round=name, row=dy, col=dx, z=dz))
        path = self.paths.shift_log(suffix)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows, columns=['fov_id', 'round', 'row', 'col', 'z']).to_csv(path, index=False)

    def _detection_plan(self, config):
        """(config for one round, the plan with its rounds or None) of a FOV detection.

        The method must be a pipeline method; channel_labels None is filled
        from dataset.channel_order; the rounds of a plan must be labels of
        RoundState.all_rounds. The first item is the bare config, or the plan
        without rounds, so a detection without rounds runs as before.
        """
        from starfinder.spot_finding import SPOT_FINDING_METHODS

        plan = config if type(config) is SpotFindingPlan else None
        base = plan.config if plan is not None else config
        spec = SPOT_FINDING_METHODS.get(type(base))
        if spec is None or not spec.pipeline:
            accepted = " or ".join(t.__name__ for t, s in SPOT_FINDING_METHODS.items() if s.pipeline)
            raise TypeError(f"FOV detection requires {accepted}")
        if base.channel_labels is None and self.dataset.channel_order:
            base = replace(base, channel_labels=tuple(self.dataset.channel_order))
        single = base if plan is None else SpotFindingPlan(base, plan.channel_overrides)
        if plan is None or plan.rounds is None:
            return single, None
        unknown = [r for r in plan.rounds if r not in self.rounds.all_rounds]
        if unknown:
            raise ValueError(f"detection rounds {unknown} are not rounds of this FOV ({self.rounds.all_rounds})")
        return single, SpotFindingPlan(base, plan.channel_overrides, plan.rounds)

    def _detection_order(self, rounds):
        """The listed rounds in FOV.run order: the reference first, then the moving rounds in declared order."""
        return [r for r in [self.rounds.reference_round] + self.rounds.moving_rounds if r in rounds]

    def _detect_round(self, config, round_name, device):
        """Detect one round's image with a config or a plan without rounds; return its SpotFindingResult.

        A round other than the reference must have the reference round's
        metadata (and ZYX shape when the reference image is resident), so
        its coordinates are on the reference grid; otherwise
        IncompatibleGeometryError.
        """
        from starfinder.image import IncompatibleGeometryError
        from starfinder.spot_finding import _detect

        reference = self.rounds.reference_round
        metadata = self.metadata.get(round_name, ImageMetadata(f"{self.fov_id}/{round_name}"))
        if round_name != reference:
            expected = self.metadata.get(reference, ImageMetadata(f"{self.fov_id}/{reference}"))
            if metadata != expected:
                raise IncompatibleGeometryError(
                    f"detection round {round_name!r} has metadata {metadata}, unlike the reference round "
                    f"{reference!r} ({expected}); a round is detected on the reference grid, after its registration")
            shape = np.shape(self.images[round_name])[:3]
            reference_shape = np.shape(self.images[reference])[:3] if reference in self.images else None
            if reference_shape is not None and shape != reference_shape:
                raise IncompatibleGeometryError(f"detection round {round_name!r} has ZYX shape {shape}, unlike the "
                                                f"reference round {reference!r} ({reference_shape})")
        namespace = json.dumps([self.dataset.dataset_id, self.dataset.sample_id,
                                self.fov_id, self.subtile_id], separators=(",", ":"))
        return _detect(self.images[round_name], config, metadata, namespace, device, round_name)

    @staticmethod
    def _detection_entry(config, result):
        """The provenance entry of one detection: the registry's entry, the weights artifacts and execution."""
        from starfinder._registry import provenance
        from starfinder.spot_finding import SPOT_FINDING_METHODS, _model_artifacts

        base = config.config if type(config) is SpotFindingPlan else config
        entry = provenance(SPOT_FINDING_METHODS[type(base)], base, "spot_finding")
        if "model" in result.diagnostics:
            entry["artifacts"] = _model_artifacts(result.diagnostics["model"])
        entry["execution"] = result.diagnostics["execution"]
        return entry

    @_log_step
    def _find_round_spots(self, config, round_name, device="cpu"):
        """FOV.run's detection of one listed round of a plan with rounds; returns the round's SpotFindingResult.

        While FOV.run records run.json, the step record gets the detection's
        provenance entry under methods.
        """
        result = self._detect_round(config, round_name, device)
        if self._run_record is not None:
            self._run_record.add_methods([self._detection_entry(config, result)])
        return result

    @_log_step
    def find_spots(self, *, config: SpotFindingConfig | SpotFindingPlan = LocalMaximaConfig(),
                   device: str = "cpu") -> FOV:
        """Detect reference-round spots, or the spots of a plan's rounds, with explicit config and FOV identity.

        config is a config of a SPOT_FINDING_METHODS method with
        pipeline=True, or a SpotFindingPlan of one; channel_labels None is
        filled from dataset.channel_order, and override channels must be
        among them. A plan with rounds detects each listed round's resident
        image (registered to the reference grid: a round whose metadata or
        shape differs from the reference round's raises
        IncompatibleGeometryError) and gives one result with a ``round``
        column, the rounds in FOV.run order (reference first), spot_id
        running over the combined table and the per-round diagnostics under
        diagnostics['rounds']. The namespace encodes dataset/sample/FOV and,
        for saved subtiles, the one-based subtile ID. IDs are retained in
        extraction/filtering tables. While FOV.run records run.json, the
        step record gets the detection's provenance entry under methods (one
        per detected round, with its ``round``, for a plan with rounds).
        """
        from starfinder.spot_finding import _combine_rounds

        single, plan = self._detection_plan(config)
        if plan is None:
            self.spot_result = self._detect_round(single, self.rounds.reference_round, device)
            entries = [self._detection_entry(single, self.spot_result)]
        else:
            results = {name: self._detect_round(single, name, device) for name in self._detection_order(plan.rounds)}
            self.spot_result = _combine_rounds(results, plan)
            entries = [dict(self._detection_entry(single, result), round=name) for name, result in results.items()]
        if self._run_record is not None:
            self._run_record.add_methods(entries)
        return self

    @_log_step
    def _extract_round(self, round_name, config=NeighborhoodSumConfig(), source=None):
        """Extract one round from its image or, when source is set, from that snapshot."""
        from starfinder.barcode import extract_intensities
        from starfinder.io import ImageLoadResult
        if source is not None and source not in self.snapshots.get(round_name, {}):
            raise ValueError(f'extraction source {source!r} is not a snapshot of round {round_name!r}')
        image = self.images[round_name] if source is None else self.snapshots[round_name][source]
        loaded = ImageLoadResult(image,
            self.metadata.get(round_name, ImageMetadata(f"{self.fov_id}/{round_name}")),
            tuple(self.dataset.channel_order), (), {})
        self._round_intensities[round_name] = extract_intensities(
            {round_name: loaded}, self.spot_result,
            config=config, readout_mode=self.dataset.readout_mode)

    def _assemble_intensities(self, rounds=None):
        rounds = self.rounds.sequencing_rounds if rounds is None else rounds
        results = [self._round_intensities[r] for r in rounds]
        first = results[0]
        if any(r.spot_ids != first.spot_ids or r.spot_namespace != first.spot_namespace or r.metadata != first.metadata or
               r.channel_labels != first.channel_labels or r.config != first.config or
               r.diagnostics['source_shape_zyx'] != first.diagnostics['source_shape_zyx'] for r in results):
            raise ValueError("inconsistent round extraction results")
        boxes = [r.box_voxels for r in results]
        # Local background measurements per (spot, channel, round), and per (channel, round) for the image.
        measured = {}
        if all(r.background is not None for r in results):
            measured = {name: np.concatenate([getattr(r, name) for r in results], axis=getattr(first, name).ndim - 1)
                        for name in ('background', 'noise', 'background_voxels', 'image_background', 'image_noise')}
        self.intensity_result = IntensityExtractionResult(
            np.concatenate([r.values for r in results], axis=2), first.spot_ids,
            first.spot_namespace, first.channel_labels, tuple(rounds), first.metadata,
            first.config, np.concatenate([r.valid for r in results], axis=1),
            {'rounds': {r: self._round_intensities[r].diagnostics for r in rounds}},
            None if any(b is None for b in boxes) else np.concatenate(boxes, axis=1), **measured)
        self._round_intensities.clear()

    @_log_step
    def extract_intensities(self, *, config=NeighborhoodSumConfig(), rounds=None):
        """Extract labeled intensities; decoding is a separate reusable stage.

        In readout mode direct each candidate is read in its own round only
        (see :func:`starfinder.barcode.extract_intensities`).
        """
        if not isinstance(config, NeighborhoodSumConfig):
            raise TypeError("config must be NeighborhoodSumConfig")
        config.__post_init__()
        rounds = self.rounds.sequencing_rounds if rounds is None else rounds
        if not rounds or len(set(rounds)) != len(rounds):
            raise ValueError("extraction rounds must be nonempty and unique")
        self._round_intensities.clear()
        for round_name in rounds:
            self._extract_round(round_name=round_name, config=config)
        self._assemble_intensities(rounds)
        return self

    def _check_readout(self, decoder, has_round):
        """Mode checks of decoding in dataset.readout_mode (docs/readout-contract.md, "Readout modes").

        multiplexed: a round column raises ValueError (MULTI_ROUND_DECODING);
        a decoder without the mode raises TypeError naming both; direct: no
        round column raises ValueError (DIRECT_NEEDS_ROUNDS).
        """
        from starfinder._registry import spec_for
        mode = self.dataset.readout_mode
        spec = spec_for(DECODING_METHODS, decoder, 'decoding method', TypeError, 'unsupported decoder config')
        if mode == 'multiplexed' and has_round:
            raise ValueError(MULTI_ROUND_DECODING)
        if mode not in spec.modes:
            raise TypeError(_mode_mismatch(mode, spec))
        if mode == 'direct' and not has_round:
            raise ValueError(DIRECT_NEEDS_ROUNDS)

    def _check_reference(self):
        """The readout mode's reference must be loaded: the codebook, or the direct panel."""
        if self.dataset.readout_mode == 'direct':
            if self.dataset.direct_panel is None:
                raise ValueError("readout_mode='direct' requires a loaded direct panel. "
                                 "Call dataset.load_direct_panel() first.")
        elif self.codebook is None:
            raise ValueError("Codebook not loaded. Call dataset.load_codebook() first.")

    @_log_step
    def decode_barcodes(self, *, config=WtaDecoderConfig(diagnostics=True)):
        """Decode (multiplexed) or assign (direct) the stored intensities; retain every spot identity.

        In readout mode multiplexed a spot table with a ``round`` column
        (detection in several rounds) raises ValueError naming the readout
        mode (§2.8). In readout mode direct, config must be a
        DirectAssignmentConfig and the candidates must have a ``round``
        column; reads come from assign_direct with dataset.direct_panel. A
        decoder that does not support the dataset's mode raises TypeError.
        """
        from starfinder.barcode import assign_direct, decode_barcodes
        self._check_readout(config, self.spot_result is not None and 'round' in self.spot_result.spots)
        self._check_reference()
        # A score and a deduplication belong to the reads they were computed on.
        self.scoring_result = self.deduplication_result = None
        if self.dataset.readout_mode == 'direct':
            self.decoding_result = assign_direct(self.intensity_result, self.spot_result,
                                                 self.dataset.direct_panel, config=config)
        else:
            self.decoding_result = decode_barcodes(self.intensity_result, self.codebook, config=config)
        return self

    @_log_step
    def score_reads(self, *, config=ReadScoreConfig()):
        """Add the shared read-QC score to the stored reads, from the stored intensities and background.

        The reference is the dataset codebook (multiplexed) or direct panel
        (direct). Identities are never changed (see
        :func:`starfinder.barcode.score_reads`). Intensities without background
        measurements raise ValueError naming extraction, the stage to rerun.
        """
        from starfinder.barcode import score_reads
        from starfinder.barcode.scoring import NO_BACKGROUND
        if self.decoding_result is None:
            raise ValueError('scoring requires decoding')
        if self.intensity_result is None or self.intensity_result.background is None:
            raise ValueError(NO_BACKGROUND)
        self._check_reference()
        reference = self.dataset.direct_panel if self.dataset.readout_mode == 'direct' else self.codebook
        self.scoring_result = score_reads(self.decoding_result, self.intensity_result, reference=reference,
                                          config=config)
        self.deduplication_result = None
        return self

    @_log_step
    def deduplicate_reads(self, *, config=DeduplicationConfig()):
        """Mark cross-channel reads of one amplicon as duplicates of one representative read.

        The reads are the scored reads when scoring ran, otherwise the decoded
        reads; the candidates without a ``round`` column were detected in the
        reference round. Nothing is removed or changed: the deduplication columns
        are added (see :func:`starfinder.barcode.deduplicate_reads`), and
        filter_reads rejects the duplicates by default. Readout mode direct
        raises ValueError.
        """
        from starfinder.barcode import deduplicate_reads
        from starfinder.barcode.deduplication import DIRECT_MODE
        if self.dataset.readout_mode == 'direct':
            raise ValueError(DIRECT_MODE)
        if self.decoding_result is None:
            raise ValueError('deduplication requires decoding')
        if self.spot_result is None or self.intensity_result is None:
            raise ValueError('deduplication requires the candidates and their intensities '
                             '(load the candidates checkpoint)')
        reads = self.scoring_result if self.scoring_result is not None else self.decoding_result
        self.deduplication_result = deduplicate_reads(reads, self.spot_result, self.intensity_result, config=config,
                                                      detection_round=self.rounds.reference_round)
        return self

    @_log_step
    def filter_reads(self, *, config=ReadFilterConfig()):
        """Rerun explicit read predicates without decoding or image access.

        The reads are the deduplicated reads when deduplication ran, else the
        scored reads when scoring ran (their score and deduplication columns are
        kept), otherwise the decoded reads. The dataset codebook, when loaded,
        supplies the segment ends of its layout; it is not used for reads of
        readout mode direct.
        """
        from starfinder.barcode import filter_reads
        reads = next(r for r in (self.deduplication_result, self.scoring_result, self.decoding_result)
                     if r is not None)
        codebook = self.codebook if getattr(reads, 'readout_mode', None) != 'direct' else None
        self.filtering_result = filter_reads(reads, config=config, codebook=codebook)
        return self

    @_log_step
    def run(self, config: PipelineConfig, *, execution: ExecutionConfig = ExecutionConfig(),
            checkpoints: CheckpointConfig | None = None):
        """Run one scientific sequence with batch or streaming residency.

        Reference first, then moving rounds in declared order. Each round runs
        the preprocessing recipe's steps in order, then registration, then the
        recipe's post_registration steps. For a needs_reference step (histogram
        matching) with fit="fov" the reference round's input to that step,
        restricted to the configured channel, is retained until every moving
        round has passed the step; this holds in streaming mode too. Steps
        with fit="supplied" read the recipe's supplied_statistics file, which
        is validated against the recipe, the dataset channel_order and the
        processed rounds before any step runs. Steps with save_as keep
        snapshots in snapshots[round]. Registration signals are built from the
        recipe's registration_source snapshot (default: the detection image),
        and the registration recipe's steps compose into one transform per
        moving round that resamples the round's detection image and every one
        of its snapshots once, so they stay aligned; the reference round is
        not transformed (see register). Detection runs on the reference
        round; a SpotFindingPlan with rounds detects each listed round after
        its registration and post-registration steps (a step record per
        round), combines them as FOV.find_spots does, and then extracts
        every sequencing round at every candidate (batch mode or
        retain_images is then required for extraction). The dataset's
        readout_mode decides the readout (docs/readout-contract.md): in
        ``multiplexed`` mode decoding such a candidate set raises ValueError
        naming the readout mode; in ``direct`` mode each candidate is
        extracted in its own round only, its other rounds are valid=False,
        and decoding is DirectAssignmentConfig, which needs such a candidate
        set and dataset.direct_panel. A decoder that does not support the
        mode raises TypeError naming the mode and the decoder. Extraction reads
        the extraction_source snapshot (default: the detection image; without
        a recipe, the source recorded in preprocessing_record, as after
        load_checkpoint). In streaming mode without retain_images a moving
        round's snapshots are dropped with its image, and only the reference
        round's registration source is kept until the moving rounds are
        registered. Per-round step records, the transforms composed per round
        and snapshot, and the supplied file's path and SHA-256, are kept in
        preprocessing_record and written to run.json and the registered
        checkpoint, which also stores the extraction source snapshot of each
        round. The pipeline never projects. Loading may be disabled for
        resident/subtile data. Images without registration must already
        declare the same frame/grid for extraction. Without image
        operations or resident images (after load_checkpoint), the round loop
        is skipped.

        After decoding (or assignment), scoring adds the shared read-QC score
        from the retained values and background (scoring_result), and
        filtering then reads the scored reads. Scoring intensities without
        background measurements raises ValueError naming extraction before
        any processing. A run that neither decodes nor holds decoded reads
        skips scoring. Deduplication (off unless config.deduplication is set)
        runs after scoring and before filtering, on the candidates and
        intensities of the run or resident ones; it raises ValueError in readout
        mode direct or without them, before any processing. A run that decodes
        or scores again without deduplication drops an earlier deduplication.
        The pre_qc checkpoint holds the reads after scoring and deduplication.
        While checkpoints are written, run.json records the population summary
        (summarize_reads) under counts when the run scored or deduplicated.

        checkpoints=None writes nothing. Otherwise the selected stages and
        run.json are written to the FOV checkpoint directory; see
        :doc:`/checkpoints`. The directory, overwrite policy and Parquet support
        are checked before any processing. On an exception the record gets
        failed (or interrupted for other BaseExceptions) with the failing step
        and round, and the original exception is re-raised.
        """
        if not isinstance(config, PipelineConfig) or not isinstance(execution, ExecutionConfig):
            raise TypeError('run requires PipelineConfig and ExecutionConfig')
        if checkpoints is not None and not isinstance(checkpoints, CheckpointConfig):
            raise TypeError('checkpoints must be CheckpointConfig or None')
        config.__post_init__()
        execution.__post_init__()
        self.rounds.validate()
        ref = self.rounds.reference_round
        if ref is None:
            raise ValueError('reference_round is required')
        if config.registration is not None and config.registration.reference_round not in (None, ref):
            raise ValueError(f'registration reference round {config.registration.reference_round!r} differs '
                             f'from the dataset reference {ref!r}')
        # A plan with rounds detects each listed round in the loop; the rounds are combined after it.
        single, detection_plan = self._detection_plan(config.spot_finding) if config.spot_finding else (None, None)
        has_round = detection_plan is not None or (
            not config.spot_finding and self.spot_result is not None and 'round' in self.spot_result.spots)
        if config.decoding:
            self._check_readout(config.decoding, has_round)
        if config.extraction and self.dataset.readout_mode == 'direct' and not has_round:
            raise ValueError(DIRECT_NEEDS_ROUNDS)
        if config.extraction and not (config.spot_finding or self.spot_result is not None):
            raise ValueError('extraction requires detections')
        if config.decoding and not (config.extraction or self.intensity_result is not None):
            raise ValueError('decoding requires intensities')
        if config.filtering and not (config.decoding or self.decoding_result is not None):
            raise ValueError('filtering requires decoding')
        # Scoring scores the reads of this run (decoded now, or resident): without reads it has
        # nothing to score and is skipped, as when the workflow's downstream stages are disabled.
        scoring = config.scoring if (config.decoding or self.decoding_result is not None) else None
        if scoring:
            from starfinder.barcode.scoring import NO_BACKGROUND
            # Scoring reads the background of this run's extraction, or of the resident intensities.
            background = (config.extraction.background is not None if config.extraction else
                          self.intensity_result is not None and self.intensity_result.background is not None)
            if not background:
                raise ValueError(NO_BACKGROUND)
        if config.deduplication:
            from starfinder.barcode.deduplication import DIRECT_MODE
            if self.dataset.readout_mode == 'direct':
                raise ValueError(DIRECT_MODE)
            if not (config.decoding or self.decoding_result is not None):
                raise ValueError('deduplication requires decoding')
            if not ((config.spot_finding or self.spot_result is not None)
                    and (config.extraction or self.intensity_result is not None)):
                raise ValueError('deduplication requires the candidates and their intensities '
                                 '(load the candidates checkpoint)')
        if config.decoding and self.dataset.readout_mode == 'direct' and self.dataset.direct_panel is None:
            raise ValueError("readout_mode='direct' requires a loaded direct panel (dataset.load_direct_panel)")
        if config.decoding and self.dataset.readout_mode == 'multiplexed' and self.codebook is None:
            raise ValueError('decoding requires a loaded codebook')
        if (detection_plan is not None and config.extraction and execution.mode == 'streaming'
                and not execution.retain_images):
            raise ValueError('extraction of candidates from several detection rounds runs after the last detected '
                             'round and reads every round; use batch mode or retain_images=True')
        detection_rounds = self._detection_order(detection_plan.rounds) if detection_plan is not None else []
        round_detections = {}
        record = self._start_run_record(checkpoints, config, execution) if checkpoints is not None else None
        current = None
        self._run_record = record
        recipe = config.preprocessing
        steps = recipe.steps if recipe is not None else ()
        post = recipe.post_registration if recipe is not None else ()
        registration_source = recipe.registration_source if recipe is not None else None
        if recipe is not None:
            self.snapshots.clear()
            self.preprocessing_record = {'recipe': {
                'steps': [step_spec(s.config).name for s in steps],
                'post_registration': [step_spec(s.config).name for s in post],
                'extraction_source': recipe.extraction_source, 'registration_source': registration_source},
                'rounds': {}, 'transforms': {}, 'supplied_statistics': {'path': None, 'sha256': None}}
        # Without a recipe (for example after load_checkpoint) extraction keeps the recorded source.
        extraction_source = self.preprocessing_record.get('recipe', {}).get('extraction_source')
        supplied = None
        try:
            if any(getattr(s.config, 'fit', None) == 'supplied' for s in steps):
                if not self.dataset.channel_order:
                    raise ValueError('fit="supplied" steps require the dataset channel_order')
                path = recipe.supplied_statistics
                supplied = read_supplied_statistics(path, recipe, channel_labels=self.dataset.channel_order,
                                                    rounds=[ref] + self.rounds.moving_rounds)
                self.preprocessing_record['supplied_statistics'] = {
                    'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            image_stages = any((config.load, config.rotation_degrees is not None, steps, post,
                config.registration, config.spot_finding, config.extraction))
            stages = checkpoints.stages if checkpoints is not None else ()
            if config.load and execution.mode == 'batch':
                self.load_images(rounds=self.rounds.all_rounds, config=config.load)
            # Reference-round inputs of needs_reference steps, kept until every round passed them.
            references = {}
            if config.extraction:
                self._round_intensities.clear()
            # A run resumed from candidates or pre_qc has no images to process.
            loop_rounds = ([ref] + self.rounds.moving_rounds) if image_stages or self.images else []
            for name in loop_rounds:
                current = name
                if config.load and execution.mode == 'streaming':
                    self.load_images(rounds=[name], config=config.load)
                if name not in self.images or name not in self.metadata:
                    raise ValueError(f'missing image/metadata for {name}')
                if config.rotation_degrees is not None:
                    self._rotate_round(round_name=name, angle=config.rotation_degrees)
                for index, step in enumerate(steps):
                    self._preprocess(step.config, round_name=name, index=index, phase='steps',
                                     stage=step_spec(step.config).name, references=references, supplied=supplied,
                                     save_as=step.save_as)
                if recipe is not None:
                    self.preprocessing_record['transforms'][name] = {
                        key: [] for key in ('detection', *self.snapshots.get(name, {}))}
                if name != ref:
                    processed_reference = self.images[ref]
                    if post:
                        self.images[ref] = registration_reference
                    try:
                        if config.registration is not None:
                            self.register(config.registration, rounds=[name], source=registration_source)
                    finally:
                        self.images[ref] = processed_reference
                if post:
                    # Keep the registration reference before the post-registration
                    # steps; use its snapshot for each moving round below.
                    if name == ref:
                        registration_reference = self.images[ref].copy()
                    for index, step in enumerate(post):
                        self._preprocess(step.config, round_name=name, index=index, phase='post_registration',
                                         stage=step_spec(step.config).name, references=references)
                if 'registered' in stages:
                    self._write_checkpoint(stage='registered', directory=record.directory,
                                           table_format=checkpoints.table_format, round_name=name)
                if name == ref and config.spot_finding and detection_plan is None:
                    self.find_spots(config=config.spot_finding, device=execution.device)
                if name in detection_rounds:
                    round_detections[name] = self._find_round_spots(single, round_name=name, device=execution.device)
                if config.extraction and name in self.rounds.sequencing_rounds and detection_plan is None:
                    self._extract_round(round_name=name, config=config.extraction, source=extraction_source)
                if execution.mode == 'streaming' and not execution.retain_images:
                    if name != ref:
                        del self.images[name]
                        self.snapshots.pop(name, None)
                    elif name in self.snapshots:
                        # Only the registration source crosses rounds.
                        self.snapshots[ref] = {key: image for key, image in self.snapshots[ref].items()
                                               if key == registration_source}
            references.clear()
            if execution.mode == 'streaming' and not execution.retain_images and loop_rounds:
                self.snapshots.pop(ref, None)
            current = None
            if 'registered' in stages and record.data['checkpoints'].get('registered'):
                self._write_checkpoint(stage='registered', directory=record.directory,
                                       table_format=checkpoints.table_format, image_rounds=[ref] + self.rounds.moving_rounds)
            if detection_plan is not None:
                from starfinder.spot_finding import _combine_rounds
                self.spot_result = _combine_rounds(round_detections, detection_plan)
                # Every round is extracted at every candidate (multiplexed) or at the candidates
                # detected in it (direct); either needs the candidates of all rounds.
                for name in loop_rounds if config.extraction else ():
                    if name in self.rounds.sequencing_rounds:
                        current = name
                        self._extract_round(round_name=name, config=config.extraction, source=extraction_source)
                current = None
            if config.extraction:
                self._assemble_intensities()
            if 'candidates' in stages and (config.spot_finding or config.extraction):
                self._write_checkpoint(stage='candidates', directory=record.directory,
                                       table_format=checkpoints.table_format)
            if config.decoding:
                self.decode_barcodes(config=config.decoding)
            if scoring:
                self.score_reads(config=scoring)
            if config.deduplication:
                self.deduplicate_reads(config=config.deduplication)
            if 'pre_qc' in stages and (config.decoding or scoring or config.deduplication):
                self._write_checkpoint(stage='pre_qc', directory=record.directory,
                                       table_format=checkpoints.table_format)
            if config.filtering:
                self.filter_reads(config=config.filtering)
            if record is not None:
                record.finish('succeeded')
        except BaseException as error:
            if record is not None and record.data['status'] == 'running':
                record.finish('failed' if isinstance(error, Exception) else 'interrupted', error, current)
            raise
        finally:
            self._run_record = None
        return self

    # --- Checkpoints ---

    def _checkpoint_dir(self, checkpoints: CheckpointConfig) -> Path:
        base = self.paths.checkpoint_dir if checkpoints.directory is None else Path(checkpoints.directory) / self.fov_id
        return base if self.subtile_id is None else base / f'subtile_{self.subtile_id}'

    def _checkpoint_header(self) -> dict:
        return dict(dataset_id=self.dataset.dataset_id, sample_id=self.dataset.sample_id,
                    fov_id=self.fov_id, subtile_id=self.subtile_id, rounds=asdict(self.rounds),
                    channel_labels=list(self.dataset.channel_order))

    def _start_run_record(self, checkpoints, config, execution):
        from starfinder.dataset._run_record import _RunRecord
        from starfinder.io._checkpoint import _require_parquet, clear_stages
        checkpoints.__post_init__()
        directory = self._checkpoint_dir(checkpoints)
        if directory.exists() and not checkpoints.overwrite:
            raise FileExistsError(f'checkpoint directory {directory} exists; pass overwrite=True to replace its files')
        if checkpoints.table_format == 'parquet':
            _require_parquet()
        if directory.exists():
            # Stages this run does not rewrite must not survive from an earlier run.
            clear_stages(directory)
        directory.mkdir(parents=True, exist_ok=True)
        record = _RunRecord(self, directory, checkpoints, config, execution)
        record.write()
        return record

    def _checkpoint_snapshots(self) -> list[str]:
        """Snapshots used downstream besides the detection image: the recorded extraction source."""
        source = self.preprocessing_record.get('recipe', {}).get('extraction_source')
        return [] if source is None else [source]

    def _write_registered_round(self, directory, round_name):
        """Write a round's image and its downstream snapshots; return their relative paths."""
        from starfinder.io import _checkpoint as io
        missing = [s for s in self._checkpoint_snapshots() if s not in self.snapshots.get(round_name, {})]
        if missing:
            raise ValueError(f'registered checkpoint requires snapshots {missing} of round {round_name!r}')
        paths = [io.write_registered_round(directory, round_name, self.images[round_name], self.metadata[round_name])]
        paths += [io.write_registered_round(directory, round_name, self.snapshots[round_name][s],
                                            self.metadata[round_name], snapshot=s) for s in self._checkpoint_snapshots()]
        return [path.relative_to(directory).as_posix() for path in paths]

    @_log_step
    def _write_checkpoint(self, stage, directory, table_format, *, round_name=None, image_rounds=None):
        """Write one stage (or one registered round) and record its files."""
        from starfinder.io import _checkpoint as io
        directory = Path(directory)
        header = self._checkpoint_header()
        if stage == 'registered' and round_name is not None:
            files = self._write_registered_round(directory, round_name)
        elif stage == 'registered':
            if image_rounds is None:
                image_rounds = self.rounds.all_rounds
                missing = [r for r in image_rounds if r not in self.images or r not in self.metadata]
                if missing:
                    raise ValueError(f'registered checkpoint requires resident images for {missing}')
                files = [f for r in image_rounds for f in self._write_registered_round(directory, r)]
            else:
                files = []
            header.update(image_rounds=list(image_rounds), snapshots=self._checkpoint_snapshots(),
                          registration_attempts=self.registration_attempts,
                          preprocessing=self.preprocessing_record or None)
            files += [f'registered/{name}' for name in
                      io.write_registered_header(directory, header, self.registration_results,
                                                 self.registration_record)]
        elif stage == 'candidates':
            if self.spot_result is None:
                raise ValueError('candidates checkpoint requires spot_result')
            header.update(readout_mode=self.dataset.readout_mode)
            files = io.write_candidates(directory, header, self.spot_result, self.intensity_result, table_format)
        elif stage == 'pre_qc':
            if self.decoding_result is None:
                raise ValueError('pre_qc checkpoint requires decoding_result')
            from starfinder.barcode.codebook import _encoding_record
            recorded = self.decoding_result.readout_mode == 'multiplexed' and self.codebook is not None
            # The encoding that decoded the reads is recorded beside the layout (null in direct mode).
            header.update(layout=self.codebook.layout if recorded else None,
                          encoding=_encoding_record(self.codebook.encoding) if recorded else None)
            files = io.write_pre_qc(directory, header, self.decoding_result, table_format,
                                    scoring_result=self.scoring_result,
                                    deduplication_result=self.deduplication_result)
        else:
            io._check_stage(stage)
        if self._run_record is not None:
            self._run_record.add_checkpoint(stage, files)
        return files

    def save_checkpoint(self, stage: str, *, checkpoints: CheckpointConfig = CheckpointConfig()) -> Path:
        """Write one checkpoint stage from the current results.

        Parameters
        ----------
        stage : str
            ``registered`` (all round images, and their extraction source
            snapshots, must be resident), ``candidates``
            (spot_result, with intensity_result when present) or ``pre_qc``
            (decoding_result, or the scored and deduplicated reads when
            scoring and deduplication ran).
        checkpoints : CheckpointConfig
            Directory, table_format and overwrite policy; stages and
            hash_inputs are not used here.

        Returns
        -------
        pathlib.Path
            The per-FOV checkpoint directory.

        Raises
        ------
        FileExistsError
            The stage exists and overwrite is False.
        ValueError
            Unknown stage or missing results.
        ImportError
            Parquet was requested without pyarrow.
        """
        from starfinder.io._checkpoint import _check_stage, _require_parquet, header_path
        _check_stage(stage)
        checkpoints.__post_init__()
        directory = self._checkpoint_dir(checkpoints)
        if header_path(directory, stage).exists() and not checkpoints.overwrite:
            raise FileExistsError(f'{stage} checkpoint exists in {directory}; pass overwrite=True')
        if checkpoints.table_format == 'parquet':
            _require_parquet()
        self._write_checkpoint(stage=stage, directory=directory, table_format=checkpoints.table_format)
        return directory

    @_log_step
    def load_checkpoint(self, stage: str, *, checkpoints: CheckpointConfig = CheckpointConfig()) -> FOV:
        """Restore one checkpoint stage so later stages can run without images.

        ``registered`` restores images, the stored snapshots, metadata,
        registration results, attempts, chains and record (a version-1
        checkpoint: no chains, semantics ``sequential``), and the
        preprocessing record; ``candidates`` restores
        spot_result and intensity_result (with its background measurements when
        they were stored); ``pre_qc`` restores decoding_result and, when the
        checkpoint was scored or deduplicated, scoring_result and
        deduplication_result. Continue with run() and a PipelineConfig that
        starts after the loaded stage: for example decoding, scoring,
        deduplication and filtering after ``candidates``, or rescoring and
        deduplication after ``candidates`` and ``pre_qc``.

        Parameters
        ----------
        stage : str
            ``registered``, ``candidates`` or ``pre_qc``.
        checkpoints : CheckpointConfig
            Only directory is used; the table format is read from the header.

        Returns
        -------
        FOV
            This instance, with the stage's results set.

        Raises
        ------
        ValueError
            This FOV already has results at or after the stage, or the saved
            FOV id, round labels or channel order differ from this FOV, or a
            candidates or pre_qc checkpoint's readout mode (multiplexed when
            the header has none) differs from the dataset's, or a pre_qc
            checkpoint's recorded encoding (method, reverse_bases and table)
            differs from the loaded codebook's. A checkpoint without the
            encoding key, or a dataset without a codebook, is not checked.
        FileNotFoundError
            The stage was not written.
        """
        from starfinder.io._checkpoint import _check_stage, _jsonable, read_checkpoint, read_header
        from starfinder.io._checkpoint import readout_mode as io_readout_mode
        _check_stage(stage)
        later = ['spot_result', 'intensity_result', 'decoding_result', 'scoring_result', 'deduplication_result',
                 'filtering_result']
        later = {'registered': ['images', 'snapshots', 'registration_results', 'registration_attempts',
                                'registration_chains', 'registration_record'] + later,
                 'candidates': later, 'pre_qc': later[2:]}[stage]
        occupied = [name for name in later if getattr(self, name) is not None and getattr(self, name) != {}]
        if occupied:
            raise ValueError(f'load_checkpoint({stage!r}) requires an FOV without {", ".join(occupied)}')
        directory = self._checkpoint_dir(checkpoints)
        header = read_header(directory, stage)
        expected = _jsonable(self._checkpoint_header())
        for key, label in (('fov_id', 'FOV id'), ('subtile_id', 'subtile id'),
                           ('rounds', 'round labels'), ('channel_labels', 'channel order')):
            if header.get(key) != expected[key]:
                raise ValueError(f'{stage} checkpoint {label} {header.get(key)!r} differs from this FOV ({expected[key]!r})')
        if stage != 'registered' and io_readout_mode(header) != self.dataset.readout_mode:
            raise ValueError(f'{stage} checkpoint readout mode {io_readout_mode(header)!r} differs from the '
                             f'dataset ({self.dataset.readout_mode!r})')
        if header.get('encoding') is not None and self.codebook is not None:
            from starfinder.barcode.codebook import _encoding_record
            current = _jsonable(_encoding_record(self.codebook.encoding))
            if header['encoding'] != current:
                raise ValueError(f"{stage} checkpoint encoding {header['encoding']} differs from the dataset "
                                 f"codebook's encoding ({current}); load the codebook with the recorded encoding")
        for name, value in read_checkpoint(directory, stage).items():
            setattr(self, name, value)
        return self

    # --- Segmentation ---

    def reference_grid(self) -> ReferenceGrid:
        """The grid of the resident reference round, on which every label image of this FOV lies.

        A :class:`~starfinder.segmentation.ReferenceGrid` with
        ``images[reference_round].shape[:3]``, ``metadata[reference_round]``, source
        ``"fov:<reference round>"`` and the SHA-256 of the reference image's C-order
        bytes (as ``reference_sha256`` of register_rounds). It is available after run() (whose
        streaming mode keeps the reference round) or after
        ``load_checkpoint("registered")``. See docs/segmentation-contract.md.

        Raises
        ------
        ValueError
            The reference round's image or metadata is not resident.
        """
        from starfinder.segmentation import ReferenceGrid, SegmentationPlan, SegmentationResult
        from starfinder.segmentation._labels import grid_sha256
        ref = self.rounds.reference_round
        if not ref or ref not in self.images or ref not in self.metadata:
            raise ValueError(f'the reference round {ref!r} is not resident; call run() or '
                             'load_checkpoint("registered") first')
        image = self.images[ref]
        return ReferenceGrid(np.shape(image)[:3], self.metadata[ref], f'fov:{ref}', grid_sha256(image))

    def segment(self, plan: SegmentationPlan, *, device: str = 'cpu',
                checkpoints: CheckpointConfig | None = None) -> FOV:
        """Run a segmentation plan on this FOV's resident reference-frame images.

        Each run, in order, takes the reference grid (:meth:`reference_grid`, or its
        Z projection for a run with ``projection``), assembles its
        :class:`~starfinder.segmentation.SegmentationInput` from the resident images
        (each :class:`~starfinder.segmentation.InputChannel` from the reference round,
        its channel maximum or a registered morphology round, with its ``prepare``
        function applied), calls :func:`~starfinder.segmentation.segment` with the
        run's seeds (or :func:`~starfinder.segmentation.import_labels` with that
        grid), applies the run's label operations, and keys the result by the run's
        name. The results are stored in ``segmentation_results`` together once every
        run has finished. A label's namespace is the JSON list ``[dataset_id,
        sample_id, fov_id, subtile_id, run name]``. Each record also holds, under
        ``upstream``, the SHA-256 of preprocessing_record and registration_record.
        It never runs registration, detection or decoding (docs/segmentation-contract.md,
        "Coordination per FOV").

        Parameters
        ----------
        plan : SegmentationPlan
            The runs; a run's seeds name an earlier run of the plan.
        device : str
            ``"cpu"`` (default) or ``"cuda"``, passed to every method.
        checkpoints : None
            Only None: results stay in memory (the saved format comes later).

        Returns
        -------
        FOV
            This instance.

        Raises
        ------
        ValueError
            An input round that is not loaded, a morphology round without an entry
            in registration_record["rounds"] (or a sequencing round without a
            registration), a round with metadata other than the reference round's,
            an unknown channel, an unknown device, or checkpoints other than None;
            and every error of segment and import_labels.
        IncompatibleGeometryError
            An input round whose ZYX shape differs from the reference grid, or seeds
            on another grid.
        """
        from starfinder.segmentation._plan import segment_fov
        if checkpoints is not None:
            raise ValueError('FOV.segment keeps its results in memory; checkpoints must be None')
        self.segmentation_results.update(segment_fov(self, plan, device=device))
        return self

    # --- Output ---

    def save_reference_image(self, *, projection: ProjectionConfig | None = None,
                             reference_image: str = 'merged', reference_channel: int = 0) -> Path:
        """Save the reference merged image under the shared ``ref_merged`` name.

        Writes the reference round's current image, which after run() is its
        detection image (the preprocessing output before registration), as
        the MATLAB workflow scripts write ``sdata.registration{ref}``:

        * ``reference_image="merged"``: the channel maximum, ZYX;
        * ``reference_image="single-channel"``: channel ``reference_channel``, ZYX;
        * then, if ``projection`` (a z projection) is given, YX.

        The dtype is kept. ZYX is a tifffile TIFF written by save_volume; YX is
        one page with the same JSON axes and metadata description, so
        load_volume reads either. The name keeps ``.tif`` for both backends.

        Raises
        ------
        ValueError
            An unknown reference_image, a channel outside the image or a
            projection that is not along z.
        """
        import tifffile
        from starfinder.io import save_volume
        from starfinder.preprocessing import project_image
        if reference_image not in ('merged', 'single-channel'):
            raise ValueError('reference_image must be merged or single-channel')
        if projection is not None and projection.axis != 'z':
            raise ValueError('the reference image projection must be along z')
        ref = self.rounds.reference_round
        image, metadata = _validate_image(self.images[ref], ndim=(4,)), self.metadata[ref]
        if reference_image == 'merged':
            image = project_image(image, config=ProjectionConfig(axis='channel'))
        else:
            if isinstance(reference_channel, bool) or not 0 <= reference_channel < image.shape[-1]:
                raise ValueError(f'reference_channel {reference_channel} is outside the reference image')
            image = image[..., reference_channel]
        path = self.paths.ref_merged_tif
        path.parent.mkdir(parents=True, exist_ok=True)
        if projection is None:
            save_volume(image, path, metadata=metadata)
            return path
        image = project_image(image, config=projection)[0]
        metadata = metadata.projected(method=projection.method)
        tifffile.imwrite(path, image, photometric='minisblack',
                         metadata={'axes': 'YX', 'starfinder_metadata': asdict(metadata)})
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
        """Persist ordered registration attempts and stage counts.

        log_type ``rsf`` or ``gr`` writes ``log/<fov>_<log_type>.txt`` and
        ``log/gr_shifts/<fov>.txt`` for every registered round; ``nr`` (the
        nuclei_registration names) writes ``log/<fov>_nr.txt`` and
        ``log/gr_shifts/<fov>_nr.txt`` for the rounds registered by
        register_rounds, with their attempts.
        """
        if log_type not in ('rsf', 'gr', 'nr'):
            raise ValueError('invalid log_type')
        path = {'rsf': self.paths.rsf_log, 'gr': self.paths.gr_log, 'nr': self.paths.nr_log}[log_type]()
        path.parent.mkdir(parents=True, exist_ok=True)
        rounds = list(self.registration_record.get('rounds', {})) if log_type == 'nr' else None
        attempts = (self.registration_attempts if rounds is None else
                    {name: self.registration_attempts[name] for name in rounds})
        path.write_text(json.dumps(dict(fov_id=self.fov_id, backend='python',
            rounds=asdict(self.rounds), registration_attempts=attempts,
            detected=len(self.spot_result.spots) if self.spot_result is not None else None,
            filtering=self.filtering_result.counts if self.filtering_result is not None else None), indent=2))
        self._save_shift_log('_nr' if log_type == 'nr' else '', rounds)
        return path

    def save_diagnostics(self, suffix=''):
        """Write counts, undefined fractions, the population summary and explicit registration attempts.

        summary is :func:`starfinder.barcode.summarize_reads` of the stored
        results (None before decoding).
        """
        from starfinder.barcode import summarize_reads
        from starfinder.io._checkpoint import _jsonable
        path = self.paths.score_log(suffix)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(
            detected=len(self.spot_result.spots) if self.spot_result is not None else None,
            counts=self.filtering_result.counts if self.filtering_result is not None else None,
            fractions=self.filtering_result.fractions if self.filtering_result is not None else None,
            summary=_jsonable(summarize_reads(self.results)) if self.decoding_result is not None else None,
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
