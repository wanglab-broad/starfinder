"""Preprocessing steps, their explicit registry, the enforcement wrapper and recipes.

See the preprocessing contract page for the interface. Steps are registered in
one mapping keyed by the exact frozen config type; the lookup from step name to
config type is derived from it.
"""
from collections.abc import Callable, Mapping
import copy
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

import numpy as np

from starfinder.image import ImageMetadata, _validate_image
from starfinder.preprocessing.background import (Background3DConfig, ScalarBackgroundConfig, _background_3d,
    _scalar_background)
from starfinder.preprocessing.morphology import ReconstructionConfig, TophatConfig, filter_tophat, reconstruct_background
from starfinder.preprocessing.normalization import (HistogramMatchingConfig, MinMaxNormalizationConfig,
    PercentileNormalizationConfig, _match_counts, _normalize, _percentile_normalize, match_histogram)

_CATEGORIES = ("background", "intensity", "contrast")
_SCOPES = ("per_channel", "per_round", "needs_reference")
_DTYPE_POLICIES = ("preserve", "declared")


@dataclass(frozen=True)
class StepContext:
    """What a step may read besides its own round's image.

    reference is the reference round's input to this step, restricted to the
    configured channel (ZYX); it is set only for needs_reference steps with
    fit="fov". supplied is this step's section of the supplied-statistics
    file ({"summarized_after", "params", "fitted"}); it is set only for
    steps with fit="supplied".
    """
    round_name: str
    reference_round: str
    metadata: ImageMetadata
    reference: np.ndarray | None = None
    supplied: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class StepResult:
    """Step output with the input's shape and axes, plus JSON-serializable records."""
    image: np.ndarray
    fitted: Mapping[str, Any]
    diagnostics: Mapping[str, Any]


@dataclass(frozen=True)
class StepSpec:
    """Registered step: stable snake_case name, implementation and declared policies.

    run(volume, config, context) returns a StepResult. category documents
    intent and imposes no order. dtype_policy "preserve" requires the input
    dtype; "declared" requires the config's output_dtype (legacy min-max only).
    """
    name: str
    run: Callable[..., StepResult]
    category: str
    scope: str
    dtype_policy: str = "preserve"

    def __post_init__(self):
        if not isinstance(self.name, str) or not re.fullmatch(r"[a-z][a-z0-9]*(_[a-z0-9]+)*", self.name):
            raise ValueError("step name must be lowercase snake_case")
        if not callable(self.run):
            raise TypeError("step run must be callable")
        if self.category not in _CATEGORIES or self.scope not in _SCOPES or self.dtype_policy not in _DTYPE_POLICIES:
            raise ValueError(f"invalid category, scope or dtype_policy for step {self.name!r}")


def _min_max(volume, config, context):
    image, groups = _normalize(volume, config)
    return StepResult(image, {"groups": groups}, {})


def _supplied_round(context, round_name):
    fitted = context.supplied["fitted"]
    if round_name not in fitted:
        raise ValueError(f"the supplied statistics have no values for round {round_name!r}")
    return fitted[round_name]


def _histogram(volume, config, context):
    if config.fit == "supplied":
        entry = _supplied_round(context, context.reference_round)
        image = _match_counts(volume, entry["values"], entry["counts"], config)
    else:
        image = match_histogram(volume, context.reference, config=config)
    return StepResult(image, {"reference_round": context.reference_round,
                              "reference_channel": config.reference_channel}, {})


def _percentile(volume, config, context):
    fitted = _supplied_round(context, context.round_name) if config.fit == "supplied" else None
    image, fitted, diagnostics = _percentile_normalize(volume, config, fitted)
    return StepResult(image, fitted, diagnostics)


def _scalar(volume, config, context):
    fitted = _supplied_round(context, context.round_name) if config.fit == "supplied" else None
    image, fitted, diagnostics = _scalar_background(volume, config, fitted)
    return StepResult(image, fitted, diagnostics)


def _volumetric(volume, config, context):
    image, fitted, diagnostics = _background_3d(volume, config, context.metadata)
    return StepResult(image, fitted, diagnostics)


def _reconstruction(volume, config, context):
    return StepResult(reconstruct_background(volume, config=config), {}, {})


def _tophat(volume, config, context):
    return StepResult(filter_tophat(volume, config=config), {}, {})


STEPS: dict[type, StepSpec] = {
    MinMaxNormalizationConfig: StepSpec("min_max_normalization", _min_max, "intensity", "per_channel", "declared"),
    HistogramMatchingConfig: StepSpec("histogram_matching", _histogram, "intensity", "needs_reference"),
    ReconstructionConfig: StepSpec("reconstruction", _reconstruction, "background", "per_channel"),
    TophatConfig: StepSpec("white_tophat", _tophat, "background", "per_channel"),
    ScalarBackgroundConfig: StepSpec("scalar_background", _scalar, "background", "per_channel"),
    Background3DConfig: StepSpec("background_3d", _volumetric, "background", "per_channel"),
    PercentileNormalizationConfig: StepSpec("percentile_normalization", _percentile, "intensity", "per_channel"),
}


def _supplied(config) -> bool:
    return getattr(config, "fit", None) == "supplied"


def step_spec(config) -> StepSpec:
    """Registered StepSpec for type(config) exactly; subclasses are not matched.

    Raises
    ------
    TypeError
        No step is registered for this exact config type.
    """
    spec = STEPS.get(type(config))
    if spec is None:
        raise TypeError(f"no preprocessing step is registered for {type(config).__qualname__} "
                        "(lookup uses the exact config type)")
    return spec


def step_config_type(name: str) -> type:
    """Config type of the step called name, derived from STEPS.

    Raises
    ------
    ValueError
        No registered step, or more than one, has this name.
    """
    matches = [config_type for config_type, spec in STEPS.items() if spec.name == name]
    if len(matches) != 1:
        raise ValueError(f"unknown preprocessing step {name!r}" if not matches else
                         f"preprocessing step name {name!r} is registered more than once")
    return matches[0]


def _jsonable(value, what, name):
    if not isinstance(value, Mapping):
        raise ValueError(f"step {name!r} returned {what} that is not a mapping")
    try:
        json.dumps(dict(value), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"step {name!r} returned {what} that are not JSON-serializable: {error}") from None


def run_step(volume: np.ndarray, config, context: StepContext) -> StepResult:
    """Run the step registered for config and enforce the step contract.

    Checks that the output has the input's shape (and so its axis order), the
    input dtype ("preserve") or the config's output_dtype ("declared"), only
    finite values, that context.metadata is unchanged, and that fitted and
    diagnostics are JSON-serializable. The input is validated as a finite
    nonempty ZYX or ZYXC array. needs_reference steps with fit="fov" require
    context.reference; other steps must not receive one. Steps with
    fit="supplied" require context.supplied (their validated section); other
    steps must not receive one.

    Raises
    ------
    TypeError
        Unregistered config type or a result that is not a StepResult.
    ValueError
        A contract violation; the message names the step.
    """
    spec = step_spec(config)
    config.__post_init__()
    if not isinstance(context, StepContext):
        raise TypeError("context must be StepContext")
    volume = _validate_image(volume)
    wants_reference = spec.scope == "needs_reference" and not _supplied(config)
    if (context.reference is not None) != wants_reference:
        raise ValueError(f"step {spec.name!r} {'requires' if wants_reference else 'does not take'} a reference image")
    if (context.supplied is not None) != _supplied(config):
        raise ValueError(f"step {spec.name!r} {'requires' if _supplied(config) else 'does not take'} supplied statistics")
    metadata = copy.deepcopy(context.metadata)
    result = spec.run(volume, config, context)
    if not isinstance(result, StepResult):
        raise TypeError(f"step {spec.name!r} must return StepResult")
    image = np.asarray(result.image)
    if context.metadata != metadata:
        raise ValueError(f"step {spec.name!r} modified the image metadata")
    if image.shape != volume.shape:
        raise ValueError(f"step {spec.name!r} changed the shape {volume.shape} to {image.shape}")
    expected = volume.dtype if spec.dtype_policy == "preserve" else np.dtype(config.output_dtype)
    if image.dtype != expected:
        raise ValueError(f"step {spec.name!r} returned dtype {image.dtype}, expected {expected} ({spec.dtype_policy})")
    if image.dtype.kind not in "uif" or not np.isfinite(image).all():
        raise ValueError(f"step {spec.name!r} returned nonfinite or nonnumeric values")
    _jsonable(result.fitted, "fitted values", spec.name)
    _jsonable(result.diagnostics, "diagnostics", spec.name)
    return result


@dataclass(frozen=True)
class RecipeStep:
    """One recipe entry: a frozen config whose exact type is registered in STEPS."""
    config: Any

    def __post_init__(self):
        step_spec(self.config)
        self.config.__post_init__()


@dataclass(frozen=True)
class PreprocessingRecipe:
    """Ordered steps run on each round before registration.

    The output of the last step is the detection image; with no steps it is
    the loaded image. post_registration runs after registration and may
    contain only ReconstructionConfig (the legacy path for resident subtiles).
    supplied_statistics is the JSON file read by steps with fit="supplied";
    it is required when such a step is present, and a step name may occur
    only once with fit="supplied".
    """
    steps: tuple[RecipeStep, ...] = ()
    post_registration: tuple[RecipeStep, ...] = ()
    supplied_statistics: Path | None = None

    def __post_init__(self):
        for name in ("steps", "post_registration"):
            entries = tuple(getattr(self, name))
            for entry in entries:
                if not isinstance(entry, RecipeStep):
                    raise TypeError(f"{name} requires RecipeStep entries")
                entry.__post_init__()
            object.__setattr__(self, name, entries)
        for entry in self.post_registration:
            if type(entry.config) is not ReconstructionConfig:
                raise ValueError("post_registration may contain only ReconstructionConfig steps, "
                                 f"not {type(entry.config).__qualname__}")
        supplied = [step_spec(entry.config).name for entry in self.steps if _supplied(entry.config)]
        repeated = sorted({name for name in supplied if supplied.count(name) > 1})
        if repeated:
            raise ValueError(f'steps {repeated} occur more than once with fit="supplied"')
        if self.supplied_statistics is not None:
            if not isinstance(self.supplied_statistics, (str, Path)) or not str(self.supplied_statistics):
                raise TypeError("supplied_statistics must be a path or None")
            object.__setattr__(self, "supplied_statistics", Path(self.supplied_statistics))
        elif supplied:
            raise ValueError(f'steps {supplied} use fit="supplied" but supplied_statistics is not set')
