"""Validated scientific stages, separate from image residency and workflow keys."""
from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np

from starfinder.io import ImageLoadConfig
from starfinder.preprocessing import PreprocessingRecipe
from starfinder.registration import RegistrationRecipe
from starfinder._execution import check_device
from starfinder.spot_finding import SPOT_FINDING_METHODS, SpotFindingPlan
from starfinder.spot_finding._methods import SpotFindingConfig
from starfinder.readout import (DECODING_METHODS, NeighborhoodSumConfig, WtaDecoderConfig,
    CodebookAwareDecoderConfig, DirectAssignmentConfig, ReadFilterConfig, ReadScoreConfig, DeduplicationConfig)


@dataclass(frozen=True)
class ExecutionConfig:
    """Batch preloads rounds; streaming loads one at a time.

    retain_images keeps processed rounds (required for subtile creation).
    Otherwise only the reference image remains after streaming. device is
    the cross-stage execution device of the methods; §2.7 accepts only
    "cpu" (any other value raises ValueError).
    """
    mode: str = 'batch'
    retain_images: bool = False
    device: str = 'cpu'

    def __post_init__(self):
        if self.mode not in ('batch', 'streaming') or not isinstance(self.retain_images, bool):
            raise ValueError('invalid execution policy')
        check_device(self.device)


@dataclass(frozen=True)
class CheckpointConfig:
    """Opt-in per-FOV checkpoints and run record written by FOV.run.

    Stages are registered images, candidates with signals and pre-QC decoding.
    Files go to ``<directory>/<fov_id>/`` (``subtile_<n>/`` below it for
    subtiles); None uses ``<output_root>/checkpoints``. Tables are CSV or
    Parquet (requires pyarrow). hash_inputs streams SHA-256 of loaded TIFFs.
    Without overwrite, an existing FOV directory is an error before processing;
    with it, run first removes that directory's earlier checkpoint files.
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


def _check_rotation(value):
    """A rotation angle in degrees is None or a finite real number (not a bool)."""
    if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating))
                              or not math.isfinite(value)):
        raise ValueError('rotation_degrees must be finite')


@dataclass(frozen=True)
class MorphologyConfig:
    """The other rounds and the reference stain prepared by FOV.prepare_morphology.

    rotation_degrees rotates every loaded image in YX as FOV.run's
    PipelineConfig.rotation_degrees does (None: no rotation); give FOV.run's
    angle so the prepared images lie on its reference grid. recipe None
    registers each other round by one translation step on the shared stain:
    the single reference stain against the round's one channel with the same
    name. A given recipe is used as FOV.register_rounds uses it, against the
    reference stain when its signal names a stain channel, else against the
    resident reference round. rounds None prepares every configured other
    round except the reference round; otherwise a tuple of other round names.
    """
    rotation_degrees: float | None = None
    recipe: RegistrationRecipe | None = None
    rounds: tuple[str, ...] | None = None

    def __post_init__(self):
        _check_rotation(self.rotation_degrees)
        if self.recipe is not None:
            if not isinstance(self.recipe, RegistrationRecipe):
                raise TypeError('recipe requires a RegistrationRecipe or None')
            self.recipe.__post_init__()
        if self.rounds is not None:
            if isinstance(self.rounds, str) or not isinstance(self.rounds, (tuple, list)):
                raise TypeError('rounds must be a tuple of round names or None')
            rounds = tuple(self.rounds)
            if not rounds or any(not isinstance(r, str) or not r for r in rounds) or len(set(rounds)) != len(rounds):
                raise ValueError('rounds must be nonempty unique round names')
            object.__setattr__(self, 'rounds', rounds)


def _pipeline_spot_finding(value):
    """Whether value is a config, or a SpotFindingPlan of a config, of a method with pipeline=True.

    The lookup uses the exact type, so a subclass of a detection config is not accepted.
    """
    config = value.config if type(value) is SpotFindingPlan else value
    spec = SPOT_FINDING_METHODS.get(type(config))
    return spec is not None and spec.pipeline


@dataclass(frozen=True)
class PipelineConfig:
    """One processing sequence. None disables an operation, including loading.

    Order: load, rotate, the preprocessing recipe's steps, the registration
    recipe (None: no registration), the preprocessing recipe's
    post_registration steps, detect, extract, decode, score, deduplicate, filter. spot_finding is a
    config of a SPOT_FINDING_METHODS method with pipeline=True, or a
    SpotFindingPlan of one. decoding is a DECODING_METHODS config of the
    dataset's readout mode: wta or codebook_aware (multiplexed), or
    DirectAssignmentConfig (direct). scoring (ReadScoreConfig, opt-in) adds
    the shared read-QC score to the reads after decoding; it needs the local
    background of the extraction. deduplication (DeduplicationConfig, off by
    default) marks cross-channel reads of one amplicon as duplicates of one
    representative after scoring; it is not available in readout mode direct.
    The pipeline processes ZYX(C) volumes and never projects;
    projection is an output view. All operation parameters are passed intact
    to public functions.
    """
    load: ImageLoadConfig | None = None
    rotation_degrees: float | None = None
    preprocessing: PreprocessingRecipe | None = None
    registration: RegistrationRecipe | None = None
    spot_finding: SpotFindingConfig | SpotFindingPlan | None = None
    extraction: NeighborhoodSumConfig | None = None
    decoding: WtaDecoderConfig | CodebookAwareDecoderConfig | DirectAssignmentConfig | None = None
    filtering: ReadFilterConfig | None = None
    scoring: ReadScoreConfig | None = None
    deduplication: DeduplicationConfig | None = None

    def __post_init__(self):
        types = {'load': ImageLoadConfig, 'preprocessing': PreprocessingRecipe, 'registration': RegistrationRecipe,
            'spot_finding': None,
            'extraction': NeighborhoodSumConfig, 'decoding': tuple(DECODING_METHODS),
            'filtering': ReadFilterConfig, 'scoring': ReadScoreConfig,
            'deduplication': DeduplicationConfig}
        for name, kind in types.items():
            value = getattr(self, name)
            if value is not None:
                if not (_pipeline_spot_finding(value) if name == 'spot_finding' else isinstance(value, kind)):
                    raise TypeError(f'{name} requires its typed operation config')
                value.__post_init__()
        if self.rotation_degrees is not None and (isinstance(self.rotation_degrees, bool) or not math.isfinite(self.rotation_degrees)):
            raise ValueError('rotation_degrees must be finite')
