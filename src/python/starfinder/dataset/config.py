"""Validated scientific stages, separate from image residency and workflow keys."""
from dataclasses import dataclass
import math
from pathlib import Path

from starfinder.io import ImageLoadConfig
from starfinder.preprocessing import (MinMaxNormalizationConfig, HistogramMatchingConfig,
    ReconstructionConfig, TophatConfig, ProjectionConfig)
from starfinder.registration import (TranslationConfig, DemonsConfig, TpsConfig, CpdConfig,
    RegistrationEstimationError, InsufficientLandmarksError, WarpConfig)
from starfinder.spot_finding import LocalMaximaConfig
from starfinder.barcode import (NeighborhoodSumConfig, WtaDecoderConfig,
    CodebookAwareDecoderConfig, ReadFilterConfig)

_REGISTRATION_CONFIGS = (TranslationConfig, DemonsConfig, TpsConfig, CpdConfig)


@dataclass(frozen=True)
class RecoveryConfig:
    """Opt-in ordered alternatives, only for explicitly allowed estimation errors.

    Validation, geometry, application and dependency errors never recover.
    """
    allowed_errors: tuple[type[RegistrationEstimationError], ...]
    alternatives: tuple[TranslationConfig | DemonsConfig | TpsConfig | CpdConfig, ...]

    def __post_init__(self):
        if not self.allowed_errors or any(e not in (RegistrationEstimationError, InsufficientLandmarksError) for e in self.allowed_errors):
            raise ValueError('recovery allows only explicit estimation error categories')
        if not self.alternatives:
            raise ValueError('recovery requires ordered alternatives')
        for config in self.alternatives:
            if not isinstance(config, _REGISTRATION_CONFIGS):
                raise TypeError('invalid recovery configuration')
            config.__post_init__()


@dataclass(frozen=True)
class RegistrationStep:
    """One estimate/apply stage with explicit channel reduction and recovery."""
    config: TranslationConfig | DemonsConfig | TpsConfig | CpdConfig
    reference_image: str = 'merged'
    moving_image: str = 'merged'
    reference_channel: int = 0
    recovery: RecoveryConfig | None = None
    warp: WarpConfig | None = None

    def __post_init__(self):
        if not isinstance(self.config, _REGISTRATION_CONFIGS):
            raise TypeError('unsupported registration config')
        self.config.__post_init__()
        if self.reference_image not in ('merged', 'single-channel') or self.moving_image not in ('merged', 'single-channel'):
            raise ValueError('registration image must be merged or single-channel')
        if isinstance(self.reference_channel, bool) or not isinstance(self.reference_channel, int) or self.reference_channel < 0:
            raise ValueError('reference_channel must be a nonnegative integer')
        if self.recovery is not None:
            self.recovery.__post_init__()
        if self.warp is not None:
            self.warp.__post_init__()


@dataclass(frozen=True)
class ExecutionConfig:
    """Batch preloads rounds; streaming loads one at a time.

    retain_images keeps processed rounds (required for subtile creation).
    Otherwise only the reference image remains after streaming.
    """
    mode: str = 'batch'
    retain_images: bool = False

    def __post_init__(self):
        if self.mode not in ('batch', 'streaming') or not isinstance(self.retain_images, bool):
            raise ValueError('invalid execution policy')


@dataclass(frozen=True)
class CheckpointConfig:
    """Opt-in per-FOV checkpoints and run record written by FOV.run.

    Stages are registered images, candidates with signals and pre-QC decoding.
    Files go to ``<directory>/<fov_id>/`` (``subtile_<n>/`` below it for
    subtiles); None uses ``<output_root>/checkpoints``. Tables are CSV or
    Parquet (requires pyarrow). hash_inputs streams SHA-256 of loaded TIFFs.
    Without overwrite, an existing FOV directory is an error before processing.
    """
    stages: tuple[str, ...] = ('registered', 'candidates', 'pre_qc')
    directory: Path | str | None = None
    table_format: str = 'csv'
    hash_inputs: bool = True
    overwrite: bool = False

    def __post_init__(self):
        stages = tuple(self.stages)
        if len(set(stages)) != len(stages) or any(s not in ('registered', 'candidates', 'pre_qc') for s in stages):
            raise ValueError('stages must be unique names among registered, candidates and pre_qc')
        object.__setattr__(self, 'stages', stages)
        if self.directory is not None:
            object.__setattr__(self, 'directory', Path(self.directory))
        if self.table_format not in ('csv', 'parquet'):
            raise ValueError('table_format must be csv or parquet')
        if not isinstance(self.hash_inputs, bool) or not isinstance(self.overwrite, bool):
            raise ValueError('hash_inputs and overwrite must be Boolean')


@dataclass(frozen=True)
class PipelineConfig:
    """One processing sequence. None disables an operation, including loading.

    Order: load, rotate, normalize, histogram match, reconstruct, tophat,
    project, ordered registration, detect, extract, decode, filter.
    Reconstruction can explicitly follow registration for legacy subtile jobs.
    All operation parameters are passed intact to public functions.
    """
    load: ImageLoadConfig | None = None
    rotation_degrees: float | None = None
    normalization: MinMaxNormalizationConfig | None = None
    histogram: HistogramMatchingConfig | None = None
    histogram_reference_channel: int = 0
    reconstruction: ReconstructionConfig | None = None
    reconstruction_after_registration: bool = False
    tophat: TophatConfig | None = None
    projection: ProjectionConfig | None = None
    registration: tuple[RegistrationStep, ...] = ()
    detection: LocalMaximaConfig | None = None
    extraction: NeighborhoodSumConfig | None = None
    decoding: WtaDecoderConfig | CodebookAwareDecoderConfig | None = None
    filtering: ReadFilterConfig | None = None

    def __post_init__(self):
        types = {'load': ImageLoadConfig, 'normalization': MinMaxNormalizationConfig,
            'histogram': HistogramMatchingConfig, 'reconstruction': ReconstructionConfig,
            'tophat': TophatConfig, 'projection': ProjectionConfig, 'detection': LocalMaximaConfig,
            'extraction': NeighborhoodSumConfig, 'decoding': (WtaDecoderConfig, CodebookAwareDecoderConfig),
            'filtering': ReadFilterConfig}
        for name, kind in types.items():
            value = getattr(self, name)
            if value is not None:
                if not isinstance(value, kind):
                    raise TypeError(f'{name} requires its typed operation config')
                value.__post_init__()
        if not isinstance(self.reconstruction_after_registration, bool):
            raise ValueError('reconstruction_after_registration must be Boolean')
        if self.rotation_degrees is not None and (isinstance(self.rotation_degrees, bool) or not math.isfinite(self.rotation_degrees)):
            raise ValueError('rotation_degrees must be finite')
        if isinstance(self.histogram_reference_channel, bool) or not isinstance(self.histogram_reference_channel, int) or self.histogram_reference_channel < 0:
            raise ValueError('histogram_reference_channel must be a nonnegative integer')
        for step in self.registration:
            if not isinstance(step, RegistrationStep):
                raise TypeError('registration requires RegistrationStep entries')
            step.__post_init__()
