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
    StepContext, load_supplied_statistics, run_step, step_spec)
from starfinder.preprocessing._rotation import _rotated
from starfinder.dataset._logging import _log_step
from starfinder import _checkpoint
from starfinder import _layout
from starfinder._layout import _FovPaths
from starfinder._records import file_sha256
from starfinder._registry import _version
from starfinder.dataset.config import CheckpointConfig, PipelineConfig, ExecutionConfig, MorphologyConfig, _check_rotation
from starfinder.dataset.types import REFERENCE_STAIN
from starfinder.registration import (ExternalReference, RegistrationRecipe, RegistrationRejectedError, RegistrationResult,
    RegistrationSignalConfig, RegistrationStep, TransformChain, TranslationConfig, TranslationTransform, WarpConfig)
from starfinder.readout import (DECODING_METHODS, Codebook, IntensityExtractionResult, NeighborhoodSumConfig,
    WtaDecoderConfig, ReadFilterConfig, ReadDecodingResult, ReadFilteringResult, ReadScoreConfig,
    ReadScoringResult, DeduplicationConfig, ReadDeduplicationResult)
from starfinder.readout.decoding import _mode_mismatch

# Raised in readout mode multiplexed when a candidate set with a round column would be decoded as barcodes.
MULTI_ROUND_DECODING = ("decoding candidates from several detection rounds needs a readout mode: "
                        "set readout_mode='direct' on the dataset for direct readout")
# Raised in readout mode direct when the candidates have no round column.
DIRECT_NEEDS_ROUNDS = ("readout_mode='direct' assigns each candidate from its own round and needs candidates "
                       "with a round column: detect with a SpotFindingPlan with explicit rounds")

if TYPE_CHECKING:
    from starfinder.dataset.dataset import Dataset
    from starfinder.assignment import AssignmentConfig, AssignmentResult
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
    from starfinder._records import jsonable, tuples
    return tuples(jsonable(dict(status=qc.status, values=qc.values, units=qc.units, counts=qc.counts,
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

    Parameters
    ----------
    dataset : Dataset
        Shared sample configuration and reference data.
    fov_id : str
        FOV identifier used in input paths, output paths and spot namespaces.
    images : dict[str, numpy.ndarray]
        Resident ZYXC images keyed by round name; the default is empty.
    snapshots : dict[str, dict[str, numpy.ndarray]]
        Round names mapped to named ZYXC preprocessing images in that round's
        current coordinates; the default is empty.
    metadata : dict[str, ImageMetadata]
        Spatial metadata keyed by resident image name; the default is empty.
    registration_results : dict[str, list[RegistrationResult]]
        Ordered registration results keyed by moving round; the default is
        empty.
    registration_attempts : dict[str, list[dict]]
        Registration attempt records keyed by moving round; the default is
        empty.
    registration_chains : dict[str, TransformChain]
        Applied transform chains keyed by moving round; the default is empty.
    registration_record : dict
        Registration semantics, recipe summary and per-round warp records; the
        default is empty.
    spot_result : SpotFindingResult or None
        Detected spots with their identities; None represents absent detections.
    subtile_id : int or None
        One-based subtile identifier; None denotes the whole FOV.
    intensity_result : IntensityExtractionResult or None
        Extracted spot intensities; None represents an absent extraction result.
    decoding_result : ReadDecodingResult or None
        Decoded reads; None represents an absent decoding result.
    filtering_result : ReadFilteringResult or None
        Read filtering result; None represents an absent filtering result.
    scoring_result : ReadScoringResult or None
        Read quality scores; None represents an absent scoring result.
    deduplication_result : ReadDeduplicationResult or None
        Read duplicate annotations; None represents an absent deduplication
        result.
    load_diagnostics : dict[str, dict]
        Image loading and geometry-operation diagnostics keyed by image name;
        the default is empty.
    preprocessing_record : dict
        Preprocessing recipe, step records and snapshot transform records; the
        default is empty.
    segmentation_results : dict[str, SegmentationResult]
        Segmentation results keyed by run name; the default is empty.
    assignment_results : dict[str, AssignmentResult]
        Molecule assignment results keyed by assignment name; the default is
        empty.

    Notes
    -----
    Stores round images/metadata, recipe snapshots (per round, snapshot name
    to image, in the round's current coordinates), structured scientific
    results, ordered registration results and attempts, the TransformChain
    applied to each moving round, registration_record (semantics
    ``recipe``, or ``sequential`` for a loaded version-1 checkpoint; the
    recipe summary; the WarpConfig applied per round), and
    preprocessing_record (the recipe, per-round step records and, per round
    and snapshot, the transforms composed, of the last run with a recipe),
    segmentation_results (run name to SegmentationResult, from segment()) and
    assignment_results (assignment name to AssignmentResult, from assign()).
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
    decoding_result: ReadDecodingResult | None = None
    filtering_result: ReadFilteringResult | None = None
    scoring_result: ReadScoringResult | None = None
    deduplication_result: ReadDeduplicationResult | None = None
    _round_intensities: dict = field(default_factory=dict)

    load_diagnostics: dict[str, dict] = field(default_factory=dict)
    preprocessing_record: dict = field(default_factory=dict)
    segmentation_results: dict[str, SegmentationResult] = field(default_factory=dict)
    assignment_results: dict[str, AssignmentResult] = field(default_factory=dict)
    _run_record: object | None = field(default=None, init=False, repr=False, compare=False)
    # run.json of the FOV's latest run, when that run had checkpoints and succeeded; export_spots records its
    # tables there (outputs.<slot>).
    _record_path: Path | None = field(default=None, init=False, repr=False, compare=False)
    # The saved reference merged image restored by load_reference_image: (ZYX image, its grid, path).
    _reference_file: tuple | None = field(default=None, init=False, repr=False, compare=False)

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

        Returns
        -------
        pandas.DataFrame
            Loaded codebook encoding table with bases, color and channel columns.

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

    def _sequencing_moving_rounds(self) -> list[str]:
        """The sequencing rounds other than the reference round, in declared order: FOV.register's default."""
        return [r for r in self.rounds.sequencing_rounds if r != self.rounds.reference_round]

    def _pipeline_rounds(self) -> list[str]:
        """The rounds FOV.run processes: the reference round (also an other round), then the sequencing rounds."""
        return [self.rounds.reference_round] + self._sequencing_moving_rounds()

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
            Round name; the folder is ``<input_root>/<round>/<fov_id>``, or
            ``<input_root>/<fov_id>/<round>`` with the dataset's
            image_dir_template ``"{fov}/{round}"``.

        Returns
        -------
        pathlib.Path
            Input directory; does not create it.
        """
        return _layout.image_dir(self.dataset.input_root, round_name, self.fov_id, self.dataset.image_dir_template)

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
        rotation_degrees: float | None = None,
    ) -> FOV:
        """Load raw TIFF stacks for specified rounds, optionally rotated.

        Delegates to ``starfinder.io.load_round()`` per round.

        Parameters
        ----------
        rounds : list[str] | None
            Round names to load; None uses the dataset rounds selected by
            round_category. ``reference_stain`` (when dataset.reference_stains
            is set) loads the reference stains from the reference round's
            folder as the image ``reference_stain``, one channel per stain.
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
        rotation_degrees : float | None
            Rotate each loaded round in YX by this angle with the code FOV.run
            uses for PipelineConfig.rotation_degrees (None: no rotation); the
            rotation diagnostics go to load_diagnostics[round]["rotation"].

        Returns
        -------
        FOV
            This instance, with processing state updated in place.
        """
        from starfinder.io import load_round

        if round_category not in ("sequencing", "other"):
            raise ValueError("invalid round_category")
        _check_rotation(rotation_degrees)
        if config is not None and (channel_order is not None or subdir):
            raise ValueError("select channels/subdir through config or arguments, not both")
        if rounds is None:
            rounds = (
                self.rounds.sequencing_rounds if round_category == "sequencing" else self.rounds.other_rounds
            )
        loadable = set(self.rounds.all_rounds) | ({REFERENCE_STAIN} if self.dataset.reference_stains else set())
        if len(set(rounds)) != len(rounds) or not set(rounds) <= loadable:
            raise ValueError("load rounds must be unique configured rounds (or reference_stain when the dataset "
                             "has reference stains)")
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
            # The reference stains are files of the reference round's folder.
            folder = self.rounds.reference_round if round_name == REFERENCE_STAIN else round_name
            loaded = load_round(self.input_dir(folder), config=load_config)
            self.images[round_name] = loaded.image
            self.metadata[round_name] = loaded.metadata
            self.load_diagnostics[round_name] = loaded.diagnostics
            if self._run_record is not None:
                self._run_record.add_inputs(loaded.source_paths)
            if rotation_degrees is not None:
                self._rotate_round(round_name=round_name, angle=rotation_degrees)
        return self

    # --- Preprocessing ---

    def _apply_to_rounds(self, func, rounds: list[str] | None) -> None:
        """Apply a function to images for the given rounds (or all)."""
        if rounds is None:
            rounds = self.rounds.all_rounds
        for name in rounds:
            self.images[name] = func(self.images[name])

    def _check_unrotated(self, round_name):
        """A round is rotated once: ValueError when its load_diagnostics already hold a rotation."""
        done = self.load_diagnostics.get(round_name, {}).get("rotation")
        if done is not None:
            raise ValueError(f"round {round_name!r} is already rotated by {done['angle_degrees']} degrees; "
                             "a round is rotated once")

    @_log_step
    def _rotate_round(self, round_name: str, angle: float) -> None:
        """Rotate a single round's image in-place; a round that already carries a rotation raises ValueError."""
        self._check_unrotated(round_name)
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

        A round is rotated once: a loaded round that already carries a
        rotation (from load_images, FOV.run or an earlier rotate) raises
        ValueError before any round is rotated.

        Parameters
        ----------
        angle : float
            Rotation angle in degrees, positive counterclockwise in the YX plane.

        Returns
        -------
        FOV
            This instance, with processing state updated in place.
        """
        for round_name in self.images:
            self._check_unrotated(round_name)
        for round_name in list(self.images.keys()):
            self._rotate_round(round_name=round_name, angle=angle)
        return self

    @_log_step
    def normalize_min_max(self, *, config=MinMaxNormalizationConfig('uint8', (0, 255)), rounds=None):
        """Min-max normalize selected rounds with an explicit output policy (normalize_min_max).

        Parameters
        ----------
        config : MinMaxNormalizationConfig
            Intensity normalization and output policy.
        rounds : list[str] or None
            Round names to process; None selects dataset.rounds.all_rounds.

        Returns
        -------
        FOV
            This instance, with processing state updated in place.
        """
        from starfinder.preprocessing import normalize_min_max
        self._apply_to_rounds(lambda image: normalize_min_max(image, config=config), rounds)
        return self

    @_log_step
    def match_histogram(self, *, config=HistogramMatchingConfig(), rounds=None, reference=None):
        """Match selected rounds to one reference channel; retain its pre-match copy.

        Parameters
        ----------
        config : HistogramMatchingConfig
            Histogram matching settings, including the reference channel.
        rounds : list[str] or None
            Round names to process; None selects dataset.rounds.all_rounds.
        reference : numpy.ndarray or None
            ZYX reference intensity image. None takes a copy of
            config.reference_channel from the resident reference round.

        Returns
        -------
        FOV
            This instance, with processing state updated in place.

        Notes
        -----
        None uses channel config.reference_channel of the reference round.
        """
        from starfinder.preprocessing import match_histogram
        if reference is None:
            reference = self.images[self.rounds.reference_round][..., config.reference_channel].copy()
        self._apply_to_rounds(lambda image: match_histogram(image, reference, config=config), rounds)
        return self

    @_log_step
    def correct_background(self, *, config=ReconstructionConfig(), rounds=None):
        """Correct the background of selected rounds by morphological reconstruction (correct_background).

        Parameters
        ----------
        config : ReconstructionConfig
            Morphological reconstruction settings for background correction.
        rounds : list[str] or None
            Round names to process; None selects dataset.rounds.all_rounds.

        Returns
        -------
        FOV
            This instance, with processing state updated in place.
        """
        from starfinder.preprocessing import correct_background
        self._apply_to_rounds(lambda image: correct_background(image, config=config), rounds)
        return self

    @_log_step
    def filter_tophat(self, *, config=TophatConfig(), rounds=None):
        """Apply public tophat to selected rounds.

        Parameters
        ----------
        config : TophatConfig
            Tophat filtering settings.
        rounds : list[str] or None
            Round names to process; None selects dataset.rounds.all_rounds.

        Returns
        -------
        FOV
            This instance, with processing state updated in place.
        """
        from starfinder.preprocessing import filter_tophat
        self._apply_to_rounds(lambda image: filter_tophat(image, config=config), rounds)
        return self

    @_log_step
    def project_image(self, *, config=ProjectionConfig(), rounds=None):
        """Project selected rounds, retaining singleton Z and source mapping.

        Parameters
        ----------
        config : ProjectionConfig
            Z projection settings for the resident round images.
        rounds : list[str] or None
            Resident image names to project; None selects the keys of images.

        Returns
        -------
        FOV
            This instance, with processing state updated in place.
        """
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
        A string is a channel pattern or name of the round, resolved by
        dataset.channel_index. An unknown or repeated name or an index outside
        the round's image is a ValueError.
        """
        if signal.mode != 'channel':
            return None
        channel = signal.moving_channel if role == 'moving' and signal.moving_channel is not None else signal.reference_channel
        if isinstance(channel, str):
            channel = self.dataset._resident_channel_index(name, channel)
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
    def register(self, recipe: RegistrationRecipe, *, rounds: Sequence[str] | None = None, source: str | None = None,
                 reference: str | ExternalReference | None = None,
                 checkpoints: CheckpointConfig | None = None) -> FOV:
        """Register moving rounds with a recipe: the sequencing rounds, or other rounds such as morphology rounds.

        The route follows the rounds' category, decided before any estimator
        runs:

        * ``rounds`` None: the sequencing rounds other than the reference
          round, on the sequencing route.
        * Every named round is a sequencing round: the sequencing route.
          ``reference`` and ``checkpoints`` must be None.
        * No named round is a sequencing round (other rounds and loaded rounds
          the dataset does not configure): the other-round route. ``source``
          must be None.
        * Both categories in one call, an empty sequence, or a parameter of
          the other route: ValueError.

        On both routes each moving round is registered once by the same
        estimation: step k is estimated on the moving signal resampled in
        float64 through steps 1 to k-1; each successful estimate is checked by
        registration_qc against recipe.qc, and a failed criterion raises
        RegistrationRejectedError. Only opted-in estimation failures recover.
        The step transforms compose into one TransformChain, and the moving
        round's image (every channel) and every one of its snapshots are
        resampled once from their pre-registration arrays with the recipe's
        warp (None: derived from the chain), so they stay aligned; the
        reference is not transformed. registration_attempts gets one
        estimation entry per attempt and one application entry per round;
        registration_results the step results (application_config: the
        round's WarpConfig), registration_chains the chain and
        registration_record the application policies. Apply errors propagate
        and are recorded too, never recovered. During a preprocessing recipe
        run, preprocessing_record lists the composed results per round and
        snapshot.

        The sequencing route registers to the dataset reference round, with
        signals built with the recipe's (or a step's own) signal from snapshot
        ``source`` of the reference and each moving round (None: their
        images). One recipe registers the sequencing rounds of a FOV:
        registration_record["recipe"] keeps its summary. FOV.run calls this
        route once per moving round.

        The other-round route registers loaded rounds to a reference round or
        an external reference. The typical recipe signal is
        ``mode="channel"`` naming a shared stain in each round:
        reference_channel in the reference round and moving_channel in every
        moving round, by index or by the round's own channel labels
        (dataset.channel_labels). MATLAB nuclei_registration corresponds to
        one translation step. Label, grid and folder errors are raised before
        any estimator runs. Every estimation attempt records the reference
        (round name or external label) and reference_sha256 (None for a
        round); registration_record["rounds"][round] keeps the recipe summary
        and the reference.

        Parameters
        ----------
        recipe : RegistrationRecipe
            Steps, signal, warp and QC.
        rounds : Sequence[str] | None
            The moving rounds, each registered at most once; None: the
            sequencing rounds other than the reference round.
        source : str | None
            Sequencing route only: the snapshot of the reference and the
            moving rounds whose signal is registered (None: their images).
        reference : str | ExternalReference | None
            Other-round route only. None uses recipe.reference_round
            (default: the dataset reference round). A round name may be any
            loaded round, other rounds included. An ExternalReference supplies
            the reference signal directly (its image, whatever the signal
            mode) and must have the moving rounds' grid; the SHA-256 of its
            image's C-order bytes is recorded.
        checkpoints : CheckpointConfig | None
            Other-round route only. None (default) writes nothing. Otherwise
            each registered round is saved to
            ``<FOV folder>/other_rounds/<round>/``
            (``image.ome.tif`` and ``registration.json``; see
            :doc:`/checkpoints`, "Prepared morphology images"), which
            :meth:`load_registered_round` reads. Only directory and overwrite
            are used.

        Returns
        -------
        FOV
            This instance.

        Raises
        ------
        ValueError
            Sequencing and other rounds in one call, an empty sequence, a
            parameter of the other route; a recipe reference round that
            differs from the reference, the reference round named in
            ``rounds`` (it cannot be registered to itself), a round that is
            already registered or (other rounds) not loaded, an unknown
            channel label or a channel index outside a round.
        IncompatibleGeometryError
            Other rounds: a moving round and the reference are not on one grid.
        FileExistsError
            With checkpoints, a round's folder exists and overwrite is False
            (checked before any estimator runs).
        """
        if not isinstance(recipe, RegistrationRecipe):
            raise TypeError('register requires a RegistrationRecipe')
        if rounds is None:
            rounds, sequencing = self._sequencing_moving_rounds(), True
        else:
            if isinstance(rounds, str) or not isinstance(rounds, Sequence) or not rounds:
                raise ValueError('rounds must be a nonempty sequence of round names')
            rounds = list(rounds)
            named = [name for name in rounds if name in self.rounds.sequencing_rounds]
            others = [name for name in rounds if name not in self.rounds.sequencing_rounds]
            if named and others:
                raise ValueError(f'rounds {named} are sequencing rounds and rounds {others} are not; register the '
                                 'sequencing rounds and the other rounds in separate calls')
            sequencing = bool(named)
        if sequencing:
            given = [name for name, value in dict(reference=reference, checkpoints=checkpoints).items()
                     if value is not None]
            if given:
                raise ValueError(f'{" and ".join(given)} must be None for the sequencing rounds {rounds}: reference '
                                 'and checkpoints are parameters of the other rounds')
            return self._register_sequencing(recipe, rounds, source)
        if source is not None:
            raise ValueError(f'source must be None for the other rounds {rounds}: source is a parameter of the '
                             'sequencing rounds')
        return self._register_other_rounds(recipe, rounds, reference, checkpoints)

    def _register_sequencing(self, recipe, rounds, source):
        """The sequencing route of register: to the dataset reference round, one recipe per FOV."""
        recipe.__post_init__()
        ref = self.rounds.reference_round
        if recipe.reference_round is not None and recipe.reference_round != ref:
            raise ValueError(f'recipe reference round {recipe.reference_round!r} differs from the dataset reference {ref!r}')
        if ref in rounds:
            raise ValueError(f'the reference round {ref!r} cannot be registered to itself')
        summary = _recipe_record(recipe)
        if self.registration_record.get('semantics') == 'sequential':
            raise ValueError('this FOV holds a sequential (version-1) registration; it cannot be extended by a recipe')
        if self.registration_record.get('recipe') not in (None, summary):
            raise ValueError('this FOV was registered with another registration recipe')
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

    def _register_other_rounds(self, recipe, rounds, reference, checkpoints):
        """The other-round route of register: loaded rounds to a reference, one recipe per round."""
        from starfinder.registration._types import _geometry
        if checkpoints is not None and not isinstance(checkpoints, CheckpointConfig):
            raise TypeError('checkpoints must be a CheckpointConfig or None')
        recipe.__post_init__()
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
        if checkpoints is not None:
            root = self._prepared_root(checkpoints, rounds)
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
        if checkpoints is not None:
            for name in rounds:
                self._save_prepared(root, name)
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

        Parameters
        ----------
        config : SpotFindingConfig or SpotFindingPlan
            Pipeline spot-finding settings, optionally wrapped in a plan with
            channel overrides and detection rounds.
        device : str
            Execution device for spot finding: ``"cpu"``.

        Returns
        -------
        FOV
            This instance, with processing state updated in place.

        Notes
        -----
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
        from starfinder.readout import extract_intensities
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

        Parameters
        ----------
        config : NeighborhoodSumConfig
            Spot-neighborhood intensity extraction settings.
        rounds : list[str] or None
            Round names in extraction order; None selects
            dataset.rounds.sequencing_rounds.

        Returns
        -------
        FOV
            This instance, with processing state updated in place.

        Notes
        -----
        In readout mode direct each candidate is read in its own round only
        (see :func:`starfinder.readout.extract_intensities`).
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
        """Mode checks of decoding in dataset.readout_mode (design/specs/readout.md, "Readout modes").

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
    def decode_reads(self, *, config=WtaDecoderConfig(diagnostics=True)):
        """Decode (multiplexed) or assign (direct) the stored intensities; retain every spot identity.

        Parameters
        ----------
        config : WtaDecoderConfig, CodebookAwareDecoderConfig or DirectDecoderConfig
            Read decoding settings for the dataset's readout mode.

        Returns
        -------
        FOV
            This instance, with processing state updated in place.

        Notes
        -----
        In readout mode multiplexed a spot table with a ``round`` column
        (detection in several rounds) raises ValueError naming the readout
        mode. In readout mode direct, config must be a
        DirectDecoderConfig and the candidates must have a ``round``
        column; reads come from decode_direct with dataset.direct_panel. A
        decoder that does not support the dataset's mode raises TypeError.
        """
        # §2.8
        from starfinder.readout import decode_direct, decode_reads
        self._check_readout(config, self.spot_result is not None and 'round' in self.spot_result.spots)
        self._check_reference()
        # A score and a deduplication belong to the reads they were computed on.
        self.scoring_result = self.deduplication_result = None
        if self.dataset.readout_mode == 'direct':
            self.decoding_result = decode_direct(self.intensity_result, self.spot_result,
                                                 self.dataset.direct_panel, config=config)
        else:
            self.decoding_result = decode_reads(self.intensity_result, self.codebook, config=config)
        return self

    @_log_step
    def score_reads(self, *, config=ReadScoreConfig()):
        """Add the shared read-QC score to the stored reads, from the stored intensities and background.

        Parameters
        ----------
        config : ReadScoreConfig
            Read quality scoring settings.

        Returns
        -------
        FOV
            This instance, with processing state updated in place.

        Notes
        -----
        The reference is the dataset codebook (multiplexed) or direct panel
        (direct). Identities are never changed (see
        :func:`starfinder.readout.score_reads`). Intensities without background
        measurements raise ValueError naming extraction, the stage to rerun.
        """
        from starfinder.readout import score_reads
        from starfinder.readout.scoring import NO_BACKGROUND
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

        Parameters
        ----------
        config : DeduplicationConfig
            Cross-channel read deduplication settings.

        Returns
        -------
        FOV
            This instance, with processing state updated in place.

        Notes
        -----
        The reads are the scored reads when scoring ran, otherwise the decoded
        reads; the candidates without a ``round`` column were detected in the
        reference round. Nothing is removed or changed: the deduplication columns
        are added (see :func:`starfinder.readout.deduplicate_reads`), and
        filter_reads rejects the duplicates by default. Readout mode direct
        raises ValueError.
        """
        from starfinder.readout import deduplicate_reads
        from starfinder.readout.deduplication import DIRECT_MODE
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

        Parameters
        ----------
        config : ReadFilterConfig
            Read acceptance predicates.

        Returns
        -------
        FOV
            This instance, with processing state updated in place.

        Notes
        -----
        The reads are the deduplicated reads when deduplication ran, else the
        scored reads when scoring ran (their score and deduplication columns are
        kept), otherwise the decoded reads. The dataset codebook, when loaded,
        supplies the segment ends of its layout; it is not used for reads of
        readout mode direct.
        """
        from starfinder.readout import filter_reads
        reads = next(r for r in (self.deduplication_result, self.scoring_result, self.decoding_result)
                     if r is not None)
        codebook = self.codebook if getattr(reads, 'readout_mode', None) != 'direct' else None
        self.filtering_result = filter_reads(reads, config=config, codebook=codebook)
        return self

    @_log_step
    def run(self, config: PipelineConfig, *, execution: ExecutionConfig = ExecutionConfig(),
            checkpoints: CheckpointConfig | None = None):
        """Run one scientific sequence with batch or streaming residency.

        Parameters
        ----------
        config : PipelineConfig
            Scientific processing stages and their settings.
        execution : ExecutionConfig
            Image residency and execution device settings.
        checkpoints : CheckpointConfig or None
            Checkpoint and run-record persistence settings. None disables these
            writes.

        Returns
        -------
        FOV
            This instance, with processing state updated in place.

        Notes
        -----
        The run processes the reference round and the sequencing rounds only:
        reference first, then the other sequencing rounds in declared order. It
        neither loads nor changes an other round (a reference round that is an
        other round is still processed as the reference); other rounds and the
        reference stain are prepared by :meth:`prepare_morphology`. A
        SpotFindingPlan whose rounds name an other round raises ValueError.
        Each round runs
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
        readout_mode decides the readout (design/specs/readout.md): in
        ``multiplexed`` mode decoding such a candidate set raises ValueError
        naming the readout mode; in ``direct`` mode each candidate is
        extracted in its own round only, its other rounds are valid=False,
        and decoding is DirectDecoderConfig, which needs such a candidate
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
        (summarize_reads) under counts whenever reads were decoded, and the
        filter's fractions under counts.filtering.

        checkpoints=None writes nothing. Otherwise the selected stages and
        run.json (version 2) are written to the FOV folder
        (``fovs/<fov_id>/run.json`` and its ``checkpoints/``); see
        :doc:`/checkpoints`. The directory, overwrite
        policy and Parquet support are checked before any processing. On an exception the record gets
        failed (or interrupted for other BaseExceptions) with the failing step
        and round, and the original exception is re-raised. Its ``succeeded``
        means that this processing succeeded; the files written afterwards
        (export_spots, save_reference_image, save_subtiles) are added to its
        ``outputs`` as each is written.
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
            from starfinder.readout.scoring import NO_BACKGROUND
            # Scoring reads the background of this run's extraction, or of the resident intensities.
            background = (config.extraction.background is not None if config.extraction else
                          self.intensity_result is not None and self.intensity_result.background is not None)
            if not background:
                raise ValueError(NO_BACKGROUND)
        if config.deduplication:
            from starfinder.readout.deduplication import DIRECT_MODE
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
        pipeline_rounds = self._pipeline_rounds()
        if detection_plan is not None:
            outside = [r for r in detection_plan.rounds if r not in pipeline_rounds]
            if outside:
                raise ValueError(f'detection rounds {outside} are not processed by FOV.run, which processes the '
                                 f'reference and sequencing rounds {pipeline_rounds} only')
        detection_rounds = self._detection_order(detection_plan.rounds) if detection_plan is not None else []
        round_detections = {}
        record = self._start_run_record(checkpoints, config, execution) if checkpoints is not None else None
        current = None
        self._run_record = record
        self._record_path = None
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
                supplied = load_supplied_statistics(path, recipe, channel_labels=self.dataset.channel_order,
                                                    rounds=pipeline_rounds)
                self.preprocessing_record['supplied_statistics'] = {
                    'path': str(path), 'sha256': file_sha256(path)}
            image_stages = any((config.load, config.rotation_degrees is not None, steps, post,
                config.registration, config.spot_finding, config.extraction))
            stages = checkpoints.stages if checkpoints is not None else ()
            if config.load and execution.mode == 'batch':
                self.load_images(rounds=pipeline_rounds, config=config.load)
            # Reference-round inputs of needs_reference steps, kept until every round passed them.
            references = {}
            if config.extraction:
                self._round_intensities.clear()
            # A run resumed from candidates or pre_qc has no images to process (prepared morphology
            # images are not processed by run).
            loop_rounds = pipeline_rounds if image_stages or any(r in self.images for r in pipeline_rounds) else []
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
                    self._write_checkpoint(stage='registered', directory=record.checkpoint_directory,
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
            if 'registered' in stages and record.written.get('registered'):
                self._write_checkpoint(stage='registered', directory=record.checkpoint_directory,
                                       table_format=checkpoints.table_format, image_rounds=pipeline_rounds)
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
                self._write_checkpoint(stage='candidates', directory=record.checkpoint_directory,
                                       table_format=checkpoints.table_format)
            if config.decoding:
                self.decode_reads(config=config.decoding)
            if scoring:
                self.score_reads(config=scoring)
            if config.deduplication:
                self.deduplicate_reads(config=config.deduplication)
            if 'pre_qc' in stages and (config.decoding or scoring or config.deduplication):
                self._write_checkpoint(stage='pre_qc', directory=record.checkpoint_directory,
                                       table_format=checkpoints.table_format)
            if config.filtering:
                self.filter_reads(config=config.filtering)
            if record is not None:
                record.finish('succeeded')
                self._record_path = record.path
        except BaseException as error:
            if record is not None and record.data['status'] == 'running':
                record.finish('failed' if isinstance(error, Exception) else 'interrupted', error, current)
            raise
        finally:
            self._run_record = None
        return self

    # --- Morphology rounds ---

    def _stain_index(self, key):
        """The reference stain a registration signal's reference channel names (pattern or name), else None."""
        if not self.dataset.reference_stains or not isinstance(key, str):
            return None
        try:
            return self.dataset.channel_index(REFERENCE_STAIN, key)
        except ValueError:
            return None

    def _morphology_recipe(self, recipe, name):
        """(recipe, reference stain index or None) that prepare_morphology registers other round name with.

        recipe None is one translation step on the shared stain: the single
        reference stain against the round's one channel of the same name. A
        given recipe registers against the reference stain when its signals
        name a stain channel, else (index None) against the resident reference
        round.
        """
        stains = self.dataset.reference_stains
        if recipe is None:
            ask = 'give MorphologyConfig(recipe=...) to register it another way'
            if len(stains) != 1:
                found = 'no reference stain' if not stains else f'{len(stains)} reference stains'
                raise ValueError(f'prepare_morphology without a recipe registers round {name!r} on the single '
                                 f'reference stain, but the dataset has {found}; {ask}')
            stain = stains[0]
            matches = [c for c in self.dataset.channel_info(name) if stain.name is not None and c.name == stain.name]
            if len(matches) != 1:
                raise ValueError(f'prepare_morphology without a recipe registers round {name!r} on its one channel '
                                 f'named like the reference stain {stain.channel!r} ({stain.name!r}), but the round '
                                 f'has {len(matches)} such channels; {ask}')
            signal = RegistrationSignalConfig('channel', reference_channel=stain.channel,
                                              moving_channel=matches[0].channel)
            return RegistrationRecipe((RegistrationStep(TranslationConfig()),), signal=signal), 0
        signals = list(dict.fromkeys([step.signal or recipe.signal for step in recipe.steps] + [recipe.signal]))
        named = [self._stain_index(s.reference_channel) if s.mode == 'channel' else None for s in signals]
        if all(index is None for index in named):
            ref = self.rounds.reference_round
            if ref not in self.images or ref not in self.metadata:
                raise ValueError(f'the recipe of round {name!r} registers against the reference round {ref!r}, which '
                                 'is not resident; call FOV.run first, or name a reference stain in its signal')
            return recipe, None
        if len(set(named)) != 1:
            raise ValueError(f'the recipe of round {name!r} names a reference stain in some signals only, or several '
                             'stains; a recipe on the reference stain names one stain channel in every signal')
        return recipe, named[0]

    def _prepared_root(self, checkpoints, names):
        """The FOV folder; FileExistsError when a folder of names exists and overwrite is False."""
        from starfinder.dataset._prepared import check_writable, image_directory
        checkpoints.__post_init__()
        root = self._fov_dir(checkpoints)
        for name in names:
            check_writable(image_directory(root, name), checkpoints.overwrite)
        return root

    def _save_prepared(self, root, name):
        """Write the saved form of the prepared image name under root/other_rounds/name/."""
        from starfinder.dataset._prepared import image_directory, write_image
        from starfinder._records import jsonable
        entry = self.registration_record['rounds'][name]
        registered = name in self.registration_chains
        write_image(image_directory(root, name),
                    identity=dict(dataset_id=self.dataset.dataset_id, sample_id=self.dataset.sample_id,
                                  fov_id=self.fov_id, subtile_id=self.subtile_id),
                    name=name, image=self.images[name], metadata=self.metadata[name],
                    channels=[c.record() for c in self.dataset._resident_channels(name)],
                    rotation=jsonable(self.load_diagnostics.get(name, {}).get('rotation')),
                    entry={k: v for k, v in entry.items() if k != 'saved'},
                    chain_results=self.registration_results[name] if registered else None,
                    application=self.registration_record['application'][name] if registered else None,
                    attempts=self.registration_attempts.get(name) if registered else None)

    @_log_step
    def prepare_morphology(self, config: MorphologyConfig = MorphologyConfig(), *,
                           checkpoints: CheckpointConfig | None = None) -> FOV:
        """Prepare the reference stain and the other rounds in the reference frame (beside FOV.run, not a stage).

        In order:

        1. when dataset.reference_stains is not empty, load those files from
           the reference round's folder as the image ``reference_stain``
           (ZYXC, one channel per stain) and rotate it; it is recorded without
           a transform: registration_record["rounds"]["reference_stain"]
           states ``same acquisition as the reference round``, and no
           registration attempt is recorded;
        2. for each other round (config.rounds, default every configured other
           round except the reference round), load it with its own channels,
           rotate it and register it with :meth:`register`. With
           config.recipe None the recipe is one translation step on the shared
           stain, against an ExternalReference of the reference stain labelled
           ``<reference round>:<stain pattern>`` (as the nuclei_registration
           rule); a given recipe is used as register uses it, against
           the reference stain when its signal names a stain channel, else
           against the resident reference round (after FOV.run);
        3. with checkpoints, save each prepared image to
           ``<FOV folder>/other_rounds/<name>/`` (``image.ome.tif``
           and ``registration.json``; see :doc:`/checkpoints`), which
           :meth:`load_registered_round` reads.

        The loading, the rotation and register are the public calls
        (load_images with rotation_degrees, register), so calling them
        step by step gives the same images, chains and records. Without a
        recipe on the reference round, the entry needs the raw files and the
        angle only, not the results of FOV.run; pass FOV.run's
        rotation_degrees so the images lie on its reference grid, where
        :meth:`segment` reads them.

        Parameters
        ----------
        config : MorphologyConfig
            rotation_degrees, recipe and rounds.
        checkpoints : CheckpointConfig, optional
            None (default) writes nothing; otherwise its directory and
            overwrite place the folders.

        Returns
        -------
        FOV
            This instance.

        Raises
        ------
        ValueError
            No reference round, a round that is not an other round (or is the
            reference round), nothing to prepare, an image that is already
            prepared, no recipe with no reference stain, several reference
            stains or no single channel of the stain's name, a recipe on the
            reference round that is not resident, and the errors of
            load_images and register.
        TypeError
            config or checkpoints of another type.
        FileExistsError
            With checkpoints, a folder exists and overwrite is False (checked
            before anything is loaded).
        """
        from starfinder.dataset._prepared import SAME_ACQUISITION
        if not isinstance(config, MorphologyConfig):
            raise TypeError('prepare_morphology requires a MorphologyConfig')
        config.__post_init__()
        if checkpoints is not None and not isinstance(checkpoints, CheckpointConfig):
            raise TypeError('checkpoints must be a CheckpointConfig or None')
        ref = self.rounds.reference_round
        if ref is None:
            raise ValueError('reference_round is required')
        rounds = (list(config.rounds) if config.rounds is not None
                  else [r for r in self.rounds.other_rounds if r != ref])
        outside = [r for r in rounds if r not in self.rounds.other_rounds or r == ref]
        if outside:
            raise ValueError(f'rounds {outside} are not other rounds of this dataset (the reference round is '
                             'processed by FOV.run)')
        names = ([REFERENCE_STAIN] if self.dataset.reference_stains else []) + rounds
        if not names:
            raise ValueError('there is nothing to prepare: the dataset has no reference stain and no other round')
        prepared = [n for n in names if n in self.registration_record.get('rounds', {}) or n in self.registration_chains]
        if prepared:
            raise ValueError(f'{prepared} are already prepared; an image is prepared once')
        if self.registration_record.get('semantics') == 'sequential':
            raise ValueError('this FOV holds a sequential (version-1) registration; it cannot be extended by a recipe')
        recipes = {name: self._morphology_recipe(config.recipe, name) for name in rounds}
        root = self._prepared_root(checkpoints, names) if checkpoints is not None else None
        angle = config.rotation_degrees
        if self.dataset.reference_stains:
            self.load_images(rounds=[REFERENCE_STAIN], rotation_degrees=angle)
            self.registration_record = dict(self.registration_record)
            self.registration_record.setdefault('rounds', {})[REFERENCE_STAIN] = dict(
                relation=SAME_ACQUISITION, reference=ref, reference_sha256=None, recipe=None)
        for name in rounds:
            recipe, stain = recipes[name]
            self.load_images(rounds=[name], rotation_degrees=angle)
            reference = None
            if stain is not None:
                pattern = self.dataset.reference_stains[stain].channel
                reference = ExternalReference(np.ascontiguousarray(self.images[REFERENCE_STAIN][..., stain]),
                                              self.metadata[REFERENCE_STAIN], f'{ref}:{pattern}')
            self.register(recipe, rounds=[name], reference=reference)
        if root is not None:
            for name in names:
                self._save_prepared(root, name)
        return self

    @_log_step
    def load_registered_round(self, name: str, *, checkpoints: CheckpointConfig = CheckpointConfig()) -> FOV:
        """Restore one image saved by :meth:`prepare_morphology` or :meth:`register` with checkpoints.

        Reads ``<FOV folder>/other_rounds/<name>/`` and restores
        images[name], metadata[name], the rotation diagnostics and
        registration_record["rounds"][name], with the key ``saved`` (the
        image file's path relative to the FOV folder and its
        SHA-256), which :meth:`segment` links in its records. A registered
        round also gets its registration_results, registration_chains,
        registration_attempts and application WarpConfig back; the reference
        stain has none.

        Parameters
        ----------
        name : str
            An other round, or ``reference_stain``.
        checkpoints : CheckpointConfig
            Only directory is used.

        Returns
        -------
        FOV
            This instance.

        Raises
        ------
        FileNotFoundError
            The image has no ``registration.json``.
        ValueError
            A file that is missing or whose SHA-256 differs from the record
            (naming the path and both hashes), a record of another version or
            image, a dataset, sample, FOV or subtile other than this FOV's,
            channels other than the dataset's, an image that is already
            prepared, or, when the reference round is resident, metadata other
            than the reference grid's.
        """
        from starfinder.dataset._prepared import IMAGE_FILE, image_directory, read_image
        from starfinder._records import tuples
        from starfinder.registration import TransformChain
        if name in self.registration_record.get('rounds', {}) or name in self.registration_chains:
            raise ValueError(f'{name!r} is already prepared in this FOV')
        if self.registration_record.get('semantics') == 'sequential':
            raise ValueError('this FOV holds a sequential (version-1) registration; it cannot be extended by a recipe')
        root = self._fov_dir(checkpoints)
        directory = image_directory(root, name)
        identity = dict(dataset_id=self.dataset.dataset_id, sample_id=self.dataset.sample_id, fov_id=self.fov_id,
                        subtile_id=self.subtile_id)
        saved = read_image(directory, name, identity, self.dataset.channel_labels(name))
        ref = self.rounds.reference_round
        if ref in self.metadata and saved['metadata'] != self.metadata[ref]:
            raise ValueError(f'saved image {name!r} has metadata {saved["metadata"]!r}, unlike the reference round '
                             f'{ref!r} ({self.metadata[ref]!r})')
        record = saved['record']
        self.images[name], self.metadata[name] = saved['image'], saved['metadata']
        self.load_diagnostics[name] = {} if record['rotation'] is None else {'rotation': tuples(record['rotation'])}
        self.registration_record = dict(self.registration_record)
        if saved['results'] is not None:
            registration = record['registration']
            self.registration_record['semantics'] = 'recipe'
            self.registration_record.setdefault('application', {})[name] = saved['application']
            self.registration_results[name] = saved['results']
            self.registration_chains[name] = TransformChain(tuple(r.transform for r in saved['results']))
            self.registration_attempts[name] = [tuples(a) for a in registration['attempts']]
        self.registration_record.setdefault('rounds', {})[name] = dict(
            saved['entry'], saved={'path': (directory / IMAGE_FILE).relative_to(root).as_posix(),
                                   'sha256': record['image']['file_sha256']})
        return self

    # --- Checkpoints ---

    def _reference_image_path(self) -> Path:
        """The file of the FOV's reference image (_layout.reference_image_path of the dataset's settings)."""
        return _layout.reference_image_path(self.dataset.output_root, self.fov_id, self.dataset.reference_image,
                                            self.dataset.reference_image_import)

    def _fov_dir(self, checkpoints: CheckpointConfig) -> Path:
        """The FOV folder of the durable results: fovs/<fov_id>[/subtile_<n>], or under CheckpointConfig.directory."""
        return _layout.fov_dir(self.dataset.output_root, self.fov_id, self.subtile_id, parent=checkpoints.directory)

    def _checkpoint_dir(self, checkpoints: CheckpointConfig) -> Path:
        """The restart folder of the checkpoint stages: <FOV folder>/checkpoints."""
        return _layout.checkpoint_dir(self._fov_dir(checkpoints))

    def _checkpoint_header(self) -> dict:
        return dict(dataset_id=self.dataset.dataset_id, sample_id=self.dataset.sample_id,
                    fov_id=self.fov_id, subtile_id=self.subtile_id, rounds=asdict(self.rounds),
                    channel_labels=list(self.dataset.channel_order))

    def _start_run_record(self, checkpoints, config, execution):
        from starfinder.dataset._run_record import _RunRecord
        from starfinder._records import _require_parquet
        checkpoints.__post_init__()
        # The guard covers the run's own files, run.json and checkpoints/; other writers share the FOV folder.
        folder = self._fov_dir(checkpoints)
        directory = _layout.checkpoint_dir(folder)
        existing = [path for path in (folder / _layout.RUN_RECORD, directory) if path.exists()]
        if existing and not checkpoints.overwrite:
            raise FileExistsError(f'{" and ".join(map(str, existing))} exist{"s" if len(existing) == 1 else ""}; '
                                  'pass overwrite=True to replace the run record and the checkpoint files')
        if checkpoints.table_format == 'parquet':
            _require_parquet()
        if directory.exists():
            # Stages this run does not rewrite must not survive from an earlier run.
            _checkpoint.clear_stages(directory)
        directory.mkdir(parents=True, exist_ok=True)
        record = _RunRecord(self, folder, checkpoints, config, execution)
        record.write()
        return record

    def _checkpoint_snapshots(self) -> list[str]:
        """Snapshots used downstream besides the detection image: the recorded extraction source."""
        source = self.preprocessing_record.get('recipe', {}).get('extraction_source')
        return [] if source is None else [source]

    def _write_registered_round(self, directory, round_name):
        """Write a round's image and its downstream snapshots; return their relative paths."""
        missing = [s for s in self._checkpoint_snapshots() if s not in self.snapshots.get(round_name, {})]
        if missing:
            raise ValueError(f'registered checkpoint requires snapshots {missing} of round {round_name!r}')
        paths = [_checkpoint.write_registered_round(directory, round_name, self.images[round_name],
                                                    self.metadata[round_name])]
        paths += [_checkpoint.write_registered_round(directory, round_name, self.snapshots[round_name][s],
                                                     self.metadata[round_name], snapshot=s)
                  for s in self._checkpoint_snapshots()]
        return [path.relative_to(directory).as_posix() for path in paths]

    @_log_step
    def _write_checkpoint(self, stage, directory, table_format, *, round_name=None, image_rounds=None):
        """Write one stage (or one registered round) and record its files.

        A registered round's images are hashed as they are written; the
        registered header names them with the field files (``files``).
        """
        directory = Path(directory)
        header = self._checkpoint_header()
        registered = directory / _layout.REGISTERED
        if stage == 'registered' and round_name is not None:
            files = self._write_registered_round(directory, round_name)
            self._run_record.registered += _checkpoint.payload(
                registered, [Path(f).relative_to(_layout.REGISTERED).as_posix() for f in files])
        elif stage == 'registered':
            if image_rounds is None:
                image_rounds = self._pipeline_rounds()
                missing = [r for r in image_rounds if r not in self.images or r not in self.metadata]
                if missing:
                    raise ValueError(f'registered checkpoint requires resident images for {missing}')
                files = [f for r in image_rounds for f in self._write_registered_round(directory, r)]
                images = _checkpoint.payload(registered, [Path(f).relative_to(_layout.REGISTERED).as_posix()
                                                          for f in files])
            else:
                # The rounds were written one by one during the run, which hashed them.
                files, images = [], list(self._run_record.registered)
            # The checkpoint holds the reference and sequencing rounds only; an other round is saved by
            # prepare_morphology or register (other rounds) under other_rounds/.
            kept = set(image_rounds)
            record = dict(self.registration_record)
            if 'application' in record:
                record['application'] = {k: v for k, v in record['application'].items() if k in kept}
            header.update(image_rounds=list(image_rounds), snapshots=self._checkpoint_snapshots(),
                          registration_attempts={k: v for k, v in self.registration_attempts.items() if k in kept},
                          preprocessing=self.preprocessing_record or None,
                          channels={r: [c.record() for c in self.dataset.channel_info(r)] for r in image_rounds})
            files += [f'registered/{name}' for name in
                      _checkpoint.write_registered_header(directory, header,
                                                          {k: v for k, v in self.registration_results.items()
                                                           if k in kept}, record, images)]
        elif stage == 'candidates':
            if self.spot_result is None:
                raise ValueError('candidates checkpoint requires spot_result')
            header.update(readout_mode=self.dataset.readout_mode)
            files = _checkpoint.write_candidates(directory, header, self.spot_result, self.intensity_result,
                                                 table_format)
        elif stage == 'pre_qc':
            if self.decoding_result is None:
                raise ValueError('pre_qc checkpoint requires decoding_result')
            from starfinder.readout.codebook import _encoding_record
            recorded = self.decoding_result.readout_mode == 'multiplexed' and self.codebook is not None
            # The encoding that decoded the reads is recorded beside the layout (null in direct mode).
            header.update(layout=self.codebook.layout if recorded else None,
                          encoding=_encoding_record(self.codebook.encoding) if recorded else None)
            files = _checkpoint.write_pre_qc(directory, header, self.decoding_result, table_format,
                                             scoring_result=self.scoring_result,
                                             deduplication_result=self.deduplication_result)
        else:
            _checkpoint._check_stage(stage)
        if self._run_record is not None:
            self._run_record.add_checkpoint(stage, files)
        return files

    def save_checkpoint(self, stage: str, *, checkpoints: CheckpointConfig = CheckpointConfig()) -> Path:
        """Write one checkpoint stage from the current results.

        Parameters
        ----------
        stage : str
            ``registered`` (the images of the reference and sequencing rounds,
            and their extraction source snapshots, must be resident; other
            rounds are not written), ``candidates``
            (spot_result, with intensity_result when present) or ``pre_qc``
            (decoding_result, or the scored and deduplicated reads when
            scoring and deduplication ran).
        checkpoints : CheckpointConfig
            Directory, table_format and overwrite policy; stages and
            hash_inputs are not used here.

        Returns
        -------
        pathlib.Path
            The FOV's checkpoint folder, ``<FOV folder>/checkpoints``.

        Raises
        ------
        FileExistsError
            The stage exists and overwrite is False.
        ValueError
            Unknown stage or missing results.
        ImportError
            Parquet was requested without pyarrow.
        """
        from starfinder._records import _require_parquet
        _checkpoint._check_stage(stage)
        checkpoints.__post_init__()
        directory = self._checkpoint_dir(checkpoints)
        if _checkpoint.header_path(directory, stage).exists() and not checkpoints.overwrite:
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
            encoding key, or a dataset without a codebook, is not checked. A
            payload file (table, image or field file) that a version-3 stage
            record names is missing or its SHA-256 differs from the record
            (the message names the file); records of versions 1 and 2 hold no
            hashes.
        FileNotFoundError
            The stage was not written.
        """
        from starfinder._records import jsonable
        _checkpoint._check_stage(stage)
        later = ['spot_result', 'intensity_result', 'decoding_result', 'scoring_result', 'deduplication_result',
                 'filtering_result']
        later = {'registered': ['images', 'snapshots', 'registration_results', 'registration_attempts',
                                'registration_chains', 'registration_record'] + later,
                 'candidates': later, 'pre_qc': later[2:]}[stage]
        occupied = [name for name in later if getattr(self, name) is not None and getattr(self, name) != {}]
        if occupied:
            raise ValueError(f'load_checkpoint({stage!r}) requires an FOV without {", ".join(occupied)}')
        directory = self._checkpoint_dir(checkpoints)
        header = _checkpoint.read_header(directory, stage)
        expected = jsonable(self._checkpoint_header())
        for key, label in (('fov_id', 'FOV id'), ('subtile_id', 'subtile id'),
                           ('rounds', 'round labels'), ('channel_labels', 'channel order')):
            if header.get(key) != expected[key]:
                raise ValueError(f'{stage} checkpoint {label} {header.get(key)!r} differs from this FOV ({expected[key]!r})')
        if stage != 'registered' and _checkpoint.readout_mode(header) != self.dataset.readout_mode:
            raise ValueError(f'{stage} checkpoint readout mode {_checkpoint.readout_mode(header)!r} differs from the '
                             f'dataset ({self.dataset.readout_mode!r})')
        if header.get('encoding') is not None and self.codebook is not None:
            from starfinder.readout.codebook import _encoding_record
            current = jsonable(_encoding_record(self.codebook.encoding))
            if header['encoding'] != current:
                raise ValueError(f"{stage} checkpoint encoding {header['encoding']} differs from the dataset "
                                 f"codebook's encoding ({current}); load the codebook with the recorded encoding")
        for name, value in _checkpoint.load_checkpoint(directory, stage).items():
            setattr(self, name, value)
        return self

    # --- Segmentation ---

    def _reference_resident(self) -> bool:
        ref = self.rounds.reference_round
        return bool(ref) and ref in self.images and ref in self.metadata

    def reference_grid(self) -> ReferenceGrid:
        """The grid of the resident reference round, on which every label image of this FOV lies.

        Returns
        -------
        ReferenceGrid
            Reference shape in ZYX order, spatial metadata, source and image hash.

        Notes
        -----
        A :class:`~starfinder.segmentation.ReferenceGrid` with
        ``images[reference_round].shape[:3]``, ``metadata[reference_round]``, source
        ``"fov:<reference round>"`` and the SHA-256 of the reference image's C-order
        bytes (as ``reference_sha256`` of register). It is available after run() (whose
        streaming mode keeps the reference round) or after
        ``load_checkpoint("registered")``. Without the reference round, after
        :meth:`load_reference_image`, it is the grid of the saved reference image
        as :func:`~starfinder.segmentation.reference_grid_from_file` reads it
        (source ``"file:<path>"``; a YX file gives a Z=1 grid). Either is the
        same grid by the grid rule (equal shape and metadata). See
        design/specs/segmentation.md.

        Raises
        ------
        ValueError
            Neither the reference round nor the saved reference image is
            resident, or both are and disagree.
        """
        from starfinder.segmentation import ReferenceGrid, SegmentationPlan, SegmentationResult
        from starfinder.segmentation._labels import grid_sha256
        ref = self.rounds.reference_round
        if not self._reference_resident():
            if self._reference_file is not None:
                return self._reference_file[1]
            raise ValueError(f'the reference round {ref!r} is not resident; call run(), '
                             'load_checkpoint("registered") or load_reference_image() first')
        if self._reference_file is not None:
            self._check_reference_file(*self._reference_file)
        image = self.images[ref]
        return ReferenceGrid(np.shape(image)[:3], self.metadata[ref], f'fov:{ref}', grid_sha256(image))

    def _merged_reference(self):
        """(ZYX reference merged image, its saved file or None): the resident reference round's channel
        maximum, else the image load_reference_image restored."""
        if self._reference_resident():
            return np.asarray(self.images[self.rounds.reference_round]).max(axis=3), None
        if self._reference_file is not None:
            return self._reference_file[0], self._reference_file[2]
        raise ValueError(f'the reference merged image needs the reference round {self.rounds.reference_round!r} '
                         'or the saved reference image; call run() or load_reference_image() first')

    def _check_reference_file(self, image, grid, path):
        """ValueError unless the saved reference image is the resident reference round's channel maximum.

        A ZYX file must be the channel maximum with the reference metadata; a YX
        file (a Z=1 grid in the reference frame's projection) its Z projection
        by that method, as save_reference_image writes them.
        """
        from starfinder.preprocessing import project_image
        ref = self.rounds.reference_round
        expected, metadata = np.asarray(self.images[ref]).max(axis=3), self.metadata[ref]
        method = next((m for m in ('max', 'sum') if grid.metadata == metadata.projected(method=m)), None)
        if method is not None and grid.shape_zyx[0] == 1:
            expected = project_image(expected, config=ProjectionConfig(method=method))
            metadata = metadata.projected(method=method)
        if (grid.shape_zyx != expected.shape or grid.metadata != metadata or image.dtype != expected.dtype
                or not np.array_equal(image, expected)):
            raise ValueError(f'the saved reference image {path} is not the channel maximum of the resident reference '
                             f'round {ref!r}: the file has {grid.shape_zyx} {image.dtype} {grid.metadata!r}, the '
                             f'round gives {expected.shape} {expected.dtype} {metadata!r}'
                             + ('' if grid.shape_zyx != expected.shape or grid.metadata != metadata
                                or image.dtype != expected.dtype else ' with other values'))

    @_log_step
    def load_reference_image(self) -> FOV:
        """Restore the reference merged image that :meth:`save_reference_image` wrote, without the reference round.

        Reads the file of the dataset's ``reference_image`` (default
        ``fovs/{fov_id}/reference.tif``), the file
        ``save_reference_image(reference_image="merged")`` writes, or the file
        ``reference_image_import`` names: ZYX, or YX after a projection, with its
        stored ``ImageMetadata``. The image becomes
        resident as the reference merged image: without the reference round,
        :meth:`reference_grid` returns the grid of that file (as
        :func:`~starfinder.segmentation.reference_grid_from_file`: source
        ``"file:<path>"`` and the SHA-256 of the array's C-order bytes; a YX
        file gives a Z=1 grid), and ``InputChannel(reference_merged=True)``, also
        as the amplicon of a ``CompositeConfig``, reads it. With
        :meth:`load_registered_round`, a process can then segment with no
        sequencing round resident (design/specs/segmentation.md, "Coordination
        per FOV"). A channel of the reference round other than the merged image
        still needs that round. When the reference round is resident too, the
        round is read, and the file must agree with its channel maximum
        (checked here and by reference_grid).

        Returns
        -------
        FOV
            This instance.

        Raises
        ------
        FileNotFoundError
            The file does not exist.
        ValueError
            The file has no stored metadata, or it disagrees with the resident
            reference round (shape, metadata, dtype or values).
        """
        from starfinder.io import load_volume
        from starfinder.segmentation import reference_grid_from_file
        path = self._reference_image_path()
        grid = reference_grid_from_file(path)
        image = np.ascontiguousarray(load_volume(path).image)
        if self._reference_resident():
            self._check_reference_file(image, grid, path)
        self._reference_file = (image, grid, path)
        return self

    def segment(self, plan: SegmentationPlan, *, device: str = 'cpu',
                checkpoints: CheckpointConfig | None = None) -> FOV:
        """Run a segmentation plan on this FOV's resident reference-frame images.

        Each run, in order, takes the reference grid (:meth:`reference_grid`, or its
        Z projection for a run with ``projection``), assembles its
        :class:`~starfinder.segmentation.SegmentationInput` from the resident images
        (each :class:`~starfinder.segmentation.InputChannel` from the reference round,
        its channel maximum (or, without the round, the saved reference image
        restored by :meth:`load_reference_image`), a registered sequencing round,
        or ``reference_stain`` or
        an other round with an entry of ``registration_record["rounds"]`` (prepared by
        :meth:`prepare_morphology` or :meth:`register` in this process, or
        reloaded by :meth:`load_registered_round`), with its ``prepare`` function
        applied), calls :func:`~starfinder.segmentation.segment` with the
        run's seeds (or :func:`~starfinder.segmentation.import_labels` with that
        grid), applies the run's label operations, and keys the result by the run's
        name. The results are stored in ``segmentation_results`` together once every
        run has finished. A label's namespace is the JSON list ``[dataset_id,
        sample_id, fov_id, subtile_id, run name]``. Each record also holds, under
        ``upstream``, the SHA-256 of preprocessing_record and registration_record.
        The record of an input channel names its round, channel name and
        registration entry; for an image reloaded by load_registered_round the
        entry links the saved file (relative path and SHA-256), and a channel
        whose reference merged image (its own, or a composite's amplicon) was
        read from the saved reference image links that file under
        ``reference_image`` (path relative to the dataset output root and
        SHA-256). A process that restores the reference image and the prepared
        images it needs segments with no sequencing round resident.
        It never runs registration, detection or decoding (design/specs/segmentation.md,
        "Coordination per FOV").

        With ``checkpoints``, each run is also written to
        ``<FOV folder>/segmentation/<run>/`` (option F1 of
        design/specs/segmentation.md, "Saved format"): ``labels.tif`` (ZYX
        ``uint32``, zlib, with the grid's ImageMetadata), ``input.ome.tif`` (the
        segmentation input, ZYXC OME-TIFF; not for an import) and
        ``segmentation.json`` (the run record, with each file's path and SHA-256).
        The stored results carry that saved record. ``run.json`` and the ``FOV.run``
        stages are not touched; ``stages`` and ``hash_inputs`` are not used.

        Parameters
        ----------
        plan : SegmentationPlan
            The runs; a run's seeds name an earlier run of the plan.
        device : str
            ``"cpu"`` (default) or ``"cuda"``, passed to every method.
        checkpoints : CheckpointConfig, optional
            None (default) keeps the results in memory only; otherwise its
            ``directory`` and ``overwrite`` place the run folders.

        Returns
        -------
        FOV
            This instance.

        Raises
        ------
        ValueError
            No reference round and no saved reference image, an input round that
            is not loaded (a channel of the reference round other than its merged
            image needs that round, not the saved reference image), a morphology
            round without an entry in registration_record["rounds"] (or a
            sequencing round without a registration), a round with metadata other
            than the reference round's, an unknown channel or an unknown device;
            and every error of segment and import_labels.
        TypeError
            checkpoints is neither None nor a CheckpointConfig.
        FileExistsError
            A run folder holds saved files and overwrite is False (checked before
            any run).
        IncompatibleGeometryError
            An input round whose ZYX shape differs from the reference grid, or seeds
            on another grid.
        """
        from starfinder.segmentation import SegmentationPlan
        from starfinder.segmentation._persist import check_writable, run_directory, write_run
        from starfinder.segmentation._plan import segment_fov
        if checkpoints is not None and not isinstance(checkpoints, CheckpointConfig):
            raise TypeError('checkpoints must be a CheckpointConfig or None')
        if checkpoints is None:
            self.segmentation_results.update(segment_fov(self, plan, device=device))
            return self
        checkpoints.__post_init__()
        root = self._fov_dir(checkpoints)
        if isinstance(plan, SegmentationPlan):
            for run in plan.runs:
                check_writable(run_directory(root, run.name), checkpoints.overwrite)
        inputs = {}
        results = segment_fov(self, plan, device=device, inputs=inputs)
        for name, result in results.items():
            directory = run_directory(root, name)
            results[name] = write_run(directory, result, inputs.get(name))
        self.segmentation_results.update(results)
        return self

    def load_segmentation(self, name: str, *, checkpoints: CheckpointConfig = CheckpointConfig()) -> SegmentationResult:
        """Read a segmentation run saved by :meth:`segment` with checkpoints.

        Reads ``<FOV folder>/segmentation/<name>/``: ``segmentation.json``
        and ``labels.tif``, checking the label file's SHA-256 and its array's
        SHA-256 against the record, the stored metadata against the recorded
        grid, the input file's SHA-256 when the run has one, and the record's
        FOV identity against this FOV. The result (its record is the saved
        record; diagnostics are not saved) is stored in
        ``segmentation_results[name]``. Like each result :meth:`segment` saved, it
        counts as saved under its run for :meth:`assign_molecules` with the same checkpoint
        root, on any FOV object of this FOV, also after the run is loaded again,
        as long as ``labels.tif`` still holds its labels.

        Parameters
        ----------
        name : str
            The run name.
        checkpoints : CheckpointConfig
            Only directory is used.

        Returns
        -------
        SegmentationResult

        Raises
        ------
        FileNotFoundError
            The run has no ``segmentation.json``.
        ValueError
            A file that is missing or whose SHA-256 differs from the record
            (naming the path and both hashes), a record of another version, run
            or FOV, or stored metadata other than the recorded grid's.
        """
        from starfinder._records import jsonable
        from starfinder.segmentation._persist import read_run, run_directory
        directory = run_directory(self._fov_dir(checkpoints), name)
        result = read_run(directory, name)
        expected = jsonable(self._checkpoint_header())
        for key in ('dataset_id', 'sample_id', 'fov_id', 'subtile_id'):
            if result.record.get(key) != expected[key]:
                raise ValueError(f'segmentation run {name!r} has {key} {result.record.get(key)!r}, this FOV '
                                 f'{expected[key]!r}')
        self.segmentation_results[name] = result
        return result

    def _label_run(self, value, what):
        from starfinder.segmentation import SegmentationResult
        if isinstance(value, SegmentationResult):
            return value
        if not isinstance(value, str):
            raise TypeError(f'{what} must name a segmentation run or be a SegmentationResult')
        if value not in self.segmentation_results:
            raise ValueError(f'{what} run {value!r} is not in segmentation_results; call segment() first '
                             f'(runs: {sorted(self.segmentation_results)})')
        return self.segmentation_results[value]

    def assign_molecules(self, config: AssignmentConfig | None = None, *, cells: str | SegmentationResult = 'cell',
                         nuclei: str | SegmentationResult | None = None, name: str = 'default',
                         population: str = 'final', correspondence: pd.DataFrame | None = None,
                         checkpoints: CheckpointConfig | None = None) -> FOV:
        """Assign this FOV's molecules to the cells of a segmentation run (the third call per FOV).

        ``cells`` and ``nuclei`` name runs of ``segmentation_results`` or are
        :class:`~starfinder.segmentation.SegmentationResult` objects the caller made
        (an ``import_labels`` result, for example). The molecules are
        :func:`~starfinder.assignment.molecule_table` of ``spot_result`` and the
        read result of ``population``: ``final`` takes the accepted reads of
        ``filtering_result``; ``called`` the reads called ``assigned`` in
        ``filtering_result`` (or in ``decoding_result`` before filtering). The
        genes are the loaded codebook's (the direct panel's in direct readout)
        and the grid is :meth:`reference_grid`: the reference round's, or the
        grid of the saved reference image after :meth:`load_reference_image`
        (one grid by the grid rule of design/specs/assignment.md). Calls
        :func:`~starfinder.assignment.assign_molecules` and stores the result in
        ``assignment_results[name]``; its record names it and holds
        :func:`~starfinder.assignment.summarize_assignment` under ``counts``. It
        never runs registration, detection, decoding or segmentation
        (design/specs/assignment.md, "Coordination per FOV").

        Parameters
        ----------
        config : AssignmentConfig, optional
            Expansion, correspondence and exclusion; ``AssignmentConfig()`` by default.
        cells, nuclei : str or SegmentationResult
            The territories (default the run ``cell``) and the optional nuclei.
        name : str
            snake_case key of ``assignment_results``.
        population : str
            ``final`` (default) or ``called``.
        correspondence : pandas.DataFrame, optional
            Supplied ``nucleus_id``, ``cell_id`` pairs.
        checkpoints : CheckpointConfig, optional
            None (default) writes nothing. Otherwise the result is also written to
            ``<FOV folder>/assignment/<name>/`` (option L1 of
            design/specs/assignment.md, "Persistence"): ``molecules``, ``cells``,
            ``counts`` and, with nuclei, ``nuclei`` in ``table_format``,
            ``assignment.json``, ``territories.tif`` with an expansion, and
            ``cell_labels.tif`` and ``nucleus_labels.tif`` for inputs that are not
            saved under their run in this root (a saved input's
            ``segmentation/<run>/labels.tif`` is linked, not copied). The stored
            result carries the saved record. Nothing else is written.

        Returns
        -------
        FOV
            This instance.

        Raises
        ------
        ValueError
            A name that is not snake_case, a run that is not in
            segmentation_results, no spot or read result, no gene list, a grid
            from a Z-projected saved reference image without the reference
            round, and every error of assign_molecules and molecule_table.
        TypeError
            checkpoints is neither None nor a CheckpointConfig, cells or nuclei of
            another type, and the type errors of assign_molecules.
        FileExistsError
            The assignment folder holds saved files and overwrite is False
            (checked before assigning).
        ImportError
            Parquet was requested without pyarrow.
        IncompatibleGeometryError
            The grid errors of assign_molecules.
        """
        from starfinder._registry import check_name
        from starfinder.assignment import AssignmentConfig, assign_molecules, molecule_table, summarize_assignment
        if checkpoints is not None and not isinstance(checkpoints, CheckpointConfig):
            raise TypeError('checkpoints must be a CheckpointConfig or None')
        check_name(name, 'assignment')
        if checkpoints is not None:
            from starfinder.assignment._persist import assignment_directory, check_writable
            from starfinder._records import _require_parquet
            checkpoints.__post_init__()
            check_writable(assignment_directory(self._fov_dir(checkpoints), name), checkpoints.overwrite)
            if checkpoints.table_format == 'parquet':
                _require_parquet()
        config = AssignmentConfig() if config is None else config
        cell_run = self._label_run(cells, 'cells')
        nucleus_run = None if nuclei is None else self._label_run(nuclei, 'nuclei')
        if self.spot_result is None:
            raise ValueError('FOV.assign_molecules needs spot_result; call run() or load_checkpoint() first')
        reads = (self.filtering_result if population == 'final' or self.filtering_result is not None
                 else self.decoding_result)
        if reads is None:
            raise ValueError(f'FOV.assign_molecules with population {population!r} needs '
                             + ('filtering_result' if population == 'final' else 'a decoding or filtering result'))
        if self.dataset.readout_mode == 'direct' and self.dataset.direct_panel is not None:
            genes = tuple(dict.fromkeys(self.dataset.direct_panel.genes))
        elif self.codebook is not None:
            genes = tuple(self.codebook.genes)
        else:
            raise ValueError('FOV.assign_molecules needs the gene list: load the codebook (or the direct panel)')
        grid = self.reference_grid()
        projected = grid.shape_zyx[0] == 1 and grid.metadata.frame_id.endswith(('/projection:max', '/projection:sum'))
        if projected and not self._reference_resident():
            raise ValueError('the saved reference image is a Z projection, so its grid is not the grid of the molecules '
                             'of FOV.run; call assign_molecules with the molecule run\'s grid, or save the reference '
                             'image without a projection')
        molecules = molecule_table(self.spot_result, reads, genes=genes, population=population)
        result = assign_molecules(molecules, cell_run, grid=grid, nuclei=nucleus_run,
                                  correspondence=correspondence, config=config)
        record = dict(result.record, name=name, counts=summarize_assignment(result))
        result = replace(result, record=record)
        if checkpoints is not None:
            from starfinder.assignment._persist import assignment_directory, write_assignment
            from starfinder.segmentation._persist import saved_labels
            root = self._fov_dir(checkpoints)
            result = write_assignment(assignment_directory(root, name), result, metadata=cell_run.grid.metadata,
                                      cells_saved=saved_labels(cell_run, root),
                                      nuclei_saved=None if nucleus_run is None else saved_labels(nucleus_run, root),
                                      same_run=nucleus_run is cell_run, table_format=checkpoints.table_format)
        self.assignment_results[name] = result
        return self

    def load_assignment(self, name: str, *, checkpoints: CheckpointConfig = CheckpointConfig()) -> AssignmentResult:
        """Read an assignment saved by :meth:`assign_molecules` with checkpoints.

        Reads ``<FOV folder>/assignment/<name>/`` and the linked
        ``segmentation/<run>/labels.tif`` files, checks every recorded SHA-256
        (each file's, and each label array's against the record) and the
        record's FOV identity, rebuilds the tables with their recorded dtypes
        (CSV or Parquet), stores the result in ``assignment_results[name]`` and
        returns it. Its record is the saved record.

        Parameters
        ----------
        name : str
            The assignment name.
        checkpoints : CheckpointConfig
            Only directory is used.

        Returns
        -------
        AssignmentResult

        Raises
        ------
        FileNotFoundError
            The assignment has no ``assignment.json``.
        ValueError
            A written or linked file that is missing or whose SHA-256 differs
            from the record (naming the path and both hashes), a record of
            another version, name or FOV.
        ImportError
            A Parquet table is read without pyarrow.
        """
        from starfinder.assignment._persist import assignment_directory, read_assignment
        from starfinder._records import jsonable
        result = read_assignment(assignment_directory(self._fov_dir(checkpoints), name), name)
        expected = jsonable(self._checkpoint_header())
        for key in ('dataset_id', 'sample_id', 'fov_id', 'subtile_id'):
            if result.record.get(key) != expected[key]:
                raise ValueError(f'assignment {name!r} has {key} {result.record.get(key)!r}, this FOV '
                                 f'{expected[key]!r}')
        self.assignment_results[name] = result
        return result

    # --- Output ---

    def save_reference_image(self, *, projection: ProjectionConfig | None = None,
                             reference_image: str = 'merged', reference_channel: int = 0) -> Path:
        """Save the reference merged image to the file of the dataset's ``reference_image``.

        Parameters
        ----------
        projection : ProjectionConfig or None
            Z projection settings for the saved reference view. None saves the ZYX
            volume without projection.
        reference_image : str
            Reference view: ``"merged"`` for the channel maximum or
            ``"single-channel"`` for reference_channel.
        reference_channel : int
            Zero-based channel index for the ``"single-channel"`` reference view.

        Returns
        -------
        pathlib.Path
            Path of the written reference-image TIFF.

        Notes
        -----
        The file is ``fovs/{fov_id}/reference.tif`` by default; a dataset that
        declares ``reference_image_import`` reads its reference image from that
        file, which is never written, so the call raises. Writes the reference
        round's current image, which after run() is its
        detection image (the preprocessing output before registration), as
        the MATLAB workflow scripts write ``sdata.registration{ref}``:

        * ``reference_image="merged"``: the channel maximum, ZYX;
        * ``reference_image="single-channel"``: channel ``reference_channel``, ZYX;
        * then, if ``projection`` (a z projection) is given, YX.

        The dtype is kept. ZYX is a tifffile TIFF written by save_volume; YX is
        one page with the same JSON axes and metadata description, so
        load_volume reads either. After a checkpointed run that succeeded,
        the FOV's run.json gains ``outputs.reference_image`` (path relative to
        the FOV folder and SHA-256).

        Raises
        ------
        ValueError
            An unknown reference_image, a channel outside the image, a
            projection that is not along z, or a dataset whose reference image
            is imported (``reference_image_import``).
        """
        import tifffile
        from starfinder.io import save_volume
        from starfinder.preprocessing import project_image
        if reference_image not in ('merged', 'single-channel'):
            raise ValueError('reference_image must be merged or single-channel')
        if self.dataset.reference_image_import is not None:
            raise ValueError(f'the reference image of this dataset is imported from '
                             f'{self.dataset.reference_image_import!r} ({_layout.REFERENCE_IMAGE_IMPORT_KEY}), which '
                             'is read where it is and never written')
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
        path = self._reference_image_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        if projection is None:
            save_volume(image, path, metadata=metadata)
        else:
            image = project_image(image, config=projection)[0]
            metadata = metadata.projected(method=projection.method)
            tifffile.imwrite(path, image, photometric='minisblack',
                             metadata={'axes': 'YX', 'starfinder_metadata': asdict(metadata)})
        self._record_output('reference_image', path)
        return path

    def export_spots(self, slot='molecules', columns=None, *, path=None):
        """Write the FOV's molecule or read table as a typed, zero-based CSV; return its path.

        Parameters
        ----------
        slot : str
            Table to write: ``"molecules"`` for accepted reads or ``"reads"`` for
            the detection/read join.
        columns : sequence of str or None
            Output column names in order. None uses the molecule columns for
            ``"molecules"`` or the complete detection/read join for ``"reads"``.
        path : pathlib.Path, str or None
            Destination CSV path. None uses the FOV output folder's molecules.csv
            for ``"molecules"`` or reads.csv for ``"reads"``.

        Returns
        -------
        pathlib.Path
            Path of the written CSV table.

        Notes
        -----
        ``slot="molecules"`` writes the accepted reads of the filtering result
        (``spot_namespace, spot_id, z, y, x, gene_id, entry_id, qc_score``);
        ``slot="reads"``, on request, every read of the decoding result joined
        to its detection (:func:`starfinder.readout.export_spots`). The default
        path is ``<output_root>/fovs/<fov_id>/molecules.csv`` or ``reads.csv``;
        ``columns`` selects columns. When the FOV's latest ``run`` had
        checkpoints and succeeded, its ``run.json`` gains
        ``outputs.<slot>``, rewritten atomically: the path relative to the
        record's folder, the file's SHA-256, the keyed ``molecules_sha256`` of
        the molecule columns (null for reads, or when the file leaves one
        out), the column dtypes and the row count.
        """
        from starfinder.readout.spots import _spot_table, _write_spot_table
        if slot not in ('molecules', 'reads'):
            raise ValueError(f"slot must be molecules or reads; got {slot!r}")
        reads = self.filtering_result if slot == 'molecules' else self.decoding_result
        table = _spot_table(self.spot_result, reads, accepted_only=slot == 'molecules', columns=columns)
        path = Path(path) if path is not None else self.paths.spot_table(slot)
        dtypes = _write_spot_table(table, path)
        if self._recording():
            from starfinder.assignment._molecules import COLUMNS, molecules_sha256
            keyed = (molecules_sha256(table) if slot == 'molecules' and set(COLUMNS) <= set(table) else None)
            self._record_output(slot, path, molecules_sha256=keyed, dtypes=dtypes, rows=len(table))
        return path

    def _recording(self) -> bool:
        """Whether the FOV's latest run had checkpoints, succeeded and left its run.json, which records the outputs."""
        return self._record_path is not None and self._record_path.is_file()

    def _record_output(self, slot, path, **extra):
        """Add ``outputs.<slot>`` (path relative to the FOV folder, SHA-256, ``extra``) to the latest run.json."""
        if self._recording():
            from starfinder.dataset._run_record import add_output, output_entry
            add_output(self._record_path, slot, output_entry(self._record_path, path, **extra))

    def save_registration_record(self, path: Path | str | None = None) -> Path:
        """Write the registration record of the other rounds as JSON; return its path.

        Parameters
        ----------
        path : pathlib.Path, str or None
            Destination JSON path. None uses
            ``<output_root>/fovs/<fov_id>/nuclei_registration.json``.

        Returns
        -------
        pathlib.Path
            Path of the written registration record.

        Notes
        -----
        The default path is ``<output_root>/fovs/<fov_id>/nuclei_registration.json``.
        The record holds ``fov_id``; ``rounds``, the round roles
        (RoundState); ``registered_rounds``, the entries of
        registration_record["rounds"] of the other rounds with registration
        attempts (registered by prepare_morphology or by register on other
        rounds; the reference stain has no registration and is left out); and
        ``registration``, those rounds' registration attempts, unchanged,
        whose estimation entries hold each translation as
        ``qc.details.transform.displacement_zyx``. The prepared images are
        saved only through the ``checkpoints`` of prepare_morphology or
        register.
        """
        from starfinder._layout import nuclei_registration_record
        from starfinder._records import write_json
        names = [r for r in self.registration_record.get('rounds', {}) if r in self.registration_attempts]
        path = Path(path) if path is not None else nuclei_registration_record(self.dataset.output_root, self.fov_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(dict(fov_id=self.fov_id, rounds=asdict(self.rounds),
                        registered_rounds={name: self.registration_record['rounds'][name] for name in names},
                        registration={name: self.registration_attempts[name] for name in names}), path)
        return path

    # --- Subtile operations ---

    def save_subtiles(
        self,
        *,
        out_dir: Path | None = None,
    ) -> pd.DataFrame:
        """Partition FOV into overlapping subtiles and save as NPZ.

        Returns subtile coordinates DataFrame with 1-based coords
        for ``stitch_subtile.py`` compatibility. After a checkpointed run that
        succeeded, the FOV's run.json gains ``outputs.subtile_coords`` (path
        relative to the FOV folder, SHA-256, rows) and ``outputs.subtile_data``
        (per subtile ID, path and SHA-256 of its NPZ).

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
        if self._recording():
            from starfinder.dataset._run_record import add_output, output_entry
            add_output(self._record_path, 'subtile_data', {
                str(row["t"]): output_entry(self._record_path, out_dir / f"subtile_data_{row['t']}.npz")
                for row in coord_rows})
            self._record_output('subtile_coords', out_dir / "subtile_coords.csv", rows=len(coords_df))
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
            Path to a trusted NPZ written by save_subtiles (loaded with allow_pickle=True).
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
