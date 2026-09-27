"""Preprocessing steps, their explicit registry, the enforcement wrapper and recipes.

See the preprocessing contract page for the interface. Steps are registered in
one mapping keyed by the exact frozen config type; the lookup from step name to
config type is derived from it.
"""
from collections.abc import Callable, Mapping
import copy
from dataclasses import dataclass
import json
import re
from typing import Any

import numpy as np

from starfinder.image import ImageMetadata, _validate_image
from starfinder.preprocessing.morphology import ReconstructionConfig, TophatConfig, filter_tophat, reconstruct_background
from starfinder.preprocessing.normalization import HistogramMatchingConfig, MinMaxNormalizationConfig, _normalize, match_histogram

_CATEGORIES = ("background", "intensity", "contrast")
_SCOPES = ("per_channel", "per_round", "needs_reference")
_DTYPE_POLICIES = ("preserve", "declared")


@dataclass(frozen=True)
class StepContext:
    """What a step may read besides its own round's image.

    reference is the reference round's input to this step, restricted to the
    configured channel (ZYX); it is set only for needs_reference steps.
    """
    round_name: str
    reference_round: str
    metadata: ImageMetadata
    reference: np.ndarray | None = None


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


def _histogram(volume, config, context):
    image = match_histogram(volume, context.reference, config=config)
    return StepResult(image, {"reference_round": context.reference_round,
                              "reference_channel": config.reference_channel}, {})


def _reconstruction(volume, config, context):
    return StepResult(reconstruct_background(volume, config=config), {}, {})


def _tophat(volume, config, context):
    return StepResult(filter_tophat(volume, config=config), {}, {})


STEPS: dict[type, StepSpec] = {
    MinMaxNormalizationConfig: StepSpec("min_max_normalization", _min_max, "intensity", "per_channel", "declared"),
    HistogramMatchingConfig: StepSpec("histogram_matching", _histogram, "intensity", "needs_reference"),
    ReconstructionConfig: StepSpec("reconstruction", _reconstruction, "background", "per_channel"),
    TophatConfig: StepSpec("white_tophat", _tophat, "background", "per_channel"),
}


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
    nonempty ZYX or ZYXC array. needs_reference steps require
    context.reference; other steps must not receive one.

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
    if (context.reference is not None) != (spec.scope == "needs_reference"):
        raise ValueError(f"step {spec.name!r} {'requires' if spec.scope == 'needs_reference' else 'does not take'} "
                         "a reference image")
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
    """
    steps: tuple[RecipeStep, ...] = ()
    post_registration: tuple[RecipeStep, ...] = ()

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
