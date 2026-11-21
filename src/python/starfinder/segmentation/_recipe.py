"""The input recipe of a segmentation run (design/specs/segmentation-input-recipe.md).

A recipe turns a run's source channels into the image its method receives, in five
fixed stages: ``channels`` (registered preprocessing steps on each source channel),
``combine`` (source channels to model channels, :data:`COMBINE_METHODS`),
``projection`` (Z), ``after_projection`` (steps on the model channels) and
``model_input`` (the pixel size the model sees). :func:`apply_input_recipe` is the one
function that applies it, for ``FOV.segment``, ``segment_fused`` and the preview.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import KW_ONLY, dataclass, replace
from numbers import Real

import numpy as np

from starfinder._registry import Dependency, check_shared, spec_for
from starfinder.preprocessing import (PREPROCESSING_METHODS, ClaheConfig, GaussianConfig, PercentileNormalizationConfig,
                                      ProjectionConfig, ReconstructionConfig, StepContext, TophatConfig, run_step,
                                      step_spec)
from starfinder.preprocessing._lengths import SPACING_KEYS, config_record, yx_pixels
from starfinder.preprocessing.contrast import clahe_used, gaussian_used

from ._inputs import CompositeConfig, FlamingoEnhancementConfig, composite_nuclei_amplicon, enhance_with_flamingo
from ._labels import ReferenceGrid, _json, array_sha256
from ._methods import ROLES, SEGMENTATION_METHODS
from ._segment import SegmentationInput

#: The registered steps an input recipe admits: scope ``per_channel``, not fitted from supplied statistics, lengths in
#: micrometres. A later step is admitted by the same rule.
ADMITTED_STEPS = (GaussianConfig, ClaheConfig, TophatConfig, ReconstructionConfig, PercentileNormalizationConfig)
#: The output dtypes of a ``sum`` or ``mean`` projection in a recipe (float32 by default).
PROJECTION_DTYPES = ("float32", "float64")
#: How a value outside the projection's output range is reported, and the bounds of the ``exact`` flag.
_EXACT_BOUND = {"float32": 2 ** 24, "float64": 2 ** 53}
#: Where the recipe takes its spacing from, recorded with it.
SPACING_SOURCE = "run grid before projection"
#: The migration row every replaced key names.
MIGRATION = "docs/migration.md, 'Segmentation input recipe'"
_STEP_RULE = ("an input recipe admits a registered step of scope per_channel that is not fitted from supplied "
              "statistics and whose lengths are in micrometres")


#: The config fields an input recipe refuses, per config type: lengths in pixels (a recipe's lengths are in µm) and
#: the projection's display conversion and axis.
REFUSED_FIELDS = {TophatConfig: ("radius_yx",), ReconstructionConfig: ("radius_yx",),
                  FlamingoEnhancementConfig: ("median_radius_px",), ProjectionConfig: ("conversion", "axis")}


def recipe_fields(config_type) -> tuple[str, ...]:
    """The init fields of a step, combine or projection config that an input recipe takes, in field order."""
    from dataclasses import fields
    refused = REFUSED_FIELDS.get(config_type, ())
    return tuple(f.name for f in fields(config_type) if f.init and f.name not in refused)


def admitted_steps() -> tuple[str, ...]:
    """The registered names of :data:`ADMITTED_STEPS`, in order."""
    return tuple(PREPROCESSING_METHODS[config_type].name for config_type in ADMITTED_STEPS)


def _check_step(config, where):
    """A step config the recipe admits: registered, admitted, not supplied, with lengths in µm."""
    if isinstance(config, RecipeStep):
        config = config.config
    spec = step_spec(config)   # TypeError for an unregistered config
    config.__post_init__()
    if type(config) not in ADMITTED_STEPS:
        raise ValueError(f"{where}: step {spec.name!r} is refused: {_STEP_RULE} ({', '.join(admitted_steps())})")
    if getattr(config, "fit", None) == "supplied":
        raise ValueError(f"{where}: step {spec.name!r} with fit: supplied is refused: {_STEP_RULE}; use fit: fov")
    if isinstance(config, (TophatConfig, ReconstructionConfig)) and config.radius_um is None:
        raise ValueError(f"{where}: step {spec.name!r} gives radius_yx (pixels), which an input recipe refuses: "
                         f"{_STEP_RULE}; give radius_um")


@dataclass(frozen=True)
class RecipeStep:
    """One step of an input recipe: a registered preprocessing step config and the channels it applies to.

    Parameters
    ----------
    config : object
        A config of a ``PREPROCESSING_METHODS`` step the recipe admits:
        ``GaussianConfig``, ``ClaheConfig``, ``TophatConfig`` or
        ``ReconstructionConfig`` with ``radius_um``, or
        ``PercentileNormalizationConfig`` with ``fit="fov"``.
    roles : tuple of str or None
        The roles of the channels the step applies to (source roles in
        ``channels``, model roles in ``after_projection``); ``None`` applies
        it to every channel of its stage.

    Raises
    ------
    TypeError
        An unregistered config, or roles that are not a tuple of strings.
    ValueError
        A step the recipe does not admit (naming the step and the rule), or an
        unknown or repeated role.
    """

    config: object
    roles: tuple[str, ...] | None = None

    def __post_init__(self):
        _check_step(self.config, "input recipe")
        if self.roles is not None:
            if isinstance(self.roles, list):
                object.__setattr__(self, "roles", tuple(self.roles))
            if not isinstance(self.roles, tuple) or not self.roles or not all(isinstance(r, str) for r in self.roles):
                raise TypeError(f"roles of step {step_spec(self.config).name!r} must be a nonempty tuple of roles "
                                "or None")
            unknown = [r for r in self.roles if r not in ROLES]
            if unknown or len(set(self.roles)) != len(self.roles):
                raise ValueError(f"roles of step {step_spec(self.config).name!r} must be distinct roles of "
                                 f"{list(ROLES)}; got {list(self.roles)}")


def _positive_values(value, name):
    if isinstance(value, (tuple, list)):
        if len(value) != 3:
            raise ValueError(f"{name} must be a number (Y and X, Z factor 1) or a ZYX 3-tuple; got {value!r}")
        return tuple(_positive(v, name) for v in value)
    return _positive(value, name)


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number; got {value!r}")
    return float(value)


@dataclass(frozen=True)
class ModelInputConfig:
    """The pixel size the model sees, from which each grid's model scale factor is derived.

    Parameters
    ----------
    pixel_size_um : float or tuple[float, float, float] or None
        The pixel size the model sees, in micrometres: a number for Y and X
        (Z factor 1), or a ZYX 3-tuple; ``None`` gives ``factor`` instead.
    factor : float or tuple[float, float, float] or None
        The scale factor itself, for a grid without spacing: a number for Y
        and X (Z factor 1), or a ZYX 3-tuple; ``None`` gives ``pixel_size_um``
        instead.

    Notes
    -----
    Exactly one of the two is given. On a grid of spacing ``s`` the factor is
    ``s / pixel_size_um`` per axis, rounded to six significant digits (both
    values are recorded), so 0.094635 µm and ``pixel_size_um`` 0.37854 give
    0.25. ``factor`` is used as given per FOV and multiplied by the fusion
    factor ``k`` in Y and X after fusion. There is no default: a run whose
    method names a model needs a model input.

    Raises
    ------
    ValueError
        Both or neither given, or a value that is not positive and finite or
        a tuple of another length.
    """

    pixel_size_um: float | tuple[float, float, float] | None = None
    factor: float | tuple[float, float, float] | None = None

    def __post_init__(self):
        if (self.pixel_size_um is None) == (self.factor is None):
            raise ValueError("input_recipe.model_input holds exactly one of pixel_size_um and factor; got "
                             f"{'both' if self.factor is not None else 'neither'}")
        if self.pixel_size_um is not None:
            object.__setattr__(self, "pixel_size_um",
                               _positive_values(self.pixel_size_um, "input_recipe.model_input.pixel_size_um"))
        else:
            object.__setattr__(self, "factor", _positive_values(self.factor, "input_recipe.model_input.factor"))

    @property
    def sets_z(self) -> bool:
        """Whether the model input sets a Z factor (a ZYX 3-tuple)."""
        return isinstance(self.pixel_size_um if self.pixel_size_um is not None else self.factor, tuple)


def _steps(value, stage):
    if isinstance(value, list):
        value = tuple(value)
    if not isinstance(value, tuple):
        raise TypeError(f"input recipe stage {stage} must be a tuple of steps")
    return tuple(item if isinstance(item, RecipeStep) else RecipeStep(item) for item in value)


@dataclass(frozen=True, kw_only=True)
class InputRecipe:
    """How a segmentation run turns its source channels into the image its method receives.

    Parameters
    ----------
    channels : tuple of RecipeStep
        Steps on each source channel, in order; a bare step config is taken as
        ``RecipeStep(config)``. The empty default runs no step.
    combine : CompositeConfig or FlamingoEnhancementConfig or None
        The combine method from source channels to model channels
        (:data:`COMBINE_METHODS`); ``None`` is ``single``, each source channel
        one model channel.
    projection : ProjectionConfig or None
        The Z projection of the model channels: axis ``z``, method ``max``,
        ``sum`` or ``mean``, and for ``sum`` and ``mean`` an ``output_dtype``
        of ``float32`` (the default when None) or ``float64``; ``None``
        projects nothing.
    after_projection : tuple of RecipeStep
        Steps on each model channel after the projection (or after ``combine``
        when there is no projection). The empty default runs no step.
    model_input : ModelInputConfig or None
        The pixel size the model sees; ``None`` for a method without a model.
    name : str or None
        The name of the recipe in the configuration (``segmentation.input_recipes``),
        recorded beside it and not part of its identity; ``None`` for an inline or
        unnamed recipe.

    Notes
    -----
    The stages run in the order of the fields, and a stage that is absent is
    the identity: the recipe with every stage absent is the empty recipe, whose
    model input is the source channels stacked as ZYXC in their dtype. A
    channel is converted with ``img_as_float32`` before its first step, so
    every step works in float32. The recipe's identity is the SHA-256 of its
    canonical JSON (``recipe_sha256``). See
    design/specs/segmentation-input-recipe.md.

    Raises
    ------
    TypeError
        A field of the wrong type, or a step config that is not registered.
    ValueError
        A step the recipe does not admit, a Flamingo enhancement with
        ``median_radius_px``, a projection along the channels, with a
        conversion, with an ``output_dtype`` other than float32 and float64,
        or with an ``output_dtype`` for ``max``.
    """

    channels: tuple[RecipeStep, ...] = ()
    combine: object | None = None
    projection: ProjectionConfig | None = None
    after_projection: tuple[RecipeStep, ...] = ()
    model_input: ModelInputConfig | None = None
    name: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "channels", _steps(self.channels, "channels"))
        object.__setattr__(self, "after_projection", _steps(self.after_projection, "after_projection"))
        for item in (*self.channels, *self.after_projection):
            item.__post_init__()
        combine_spec(self.combine)
        if isinstance(self.combine, FlamingoEnhancementConfig) and self.combine.median_radius_um is None:
            raise ValueError("input recipe combine flamingo_enhance gives median_radius_px (pixels), which an input "
                             "recipe refuses: a recipe's lengths are in micrometres; give median_radius_um")
        projection = self.projection
        if projection is not None:
            if not isinstance(projection, ProjectionConfig):
                raise TypeError("input recipe projection must be a ProjectionConfig or None")
            projection.__post_init__()
            if projection.axis != "z":
                raise ValueError("an input recipe's projection is along z")
            if projection.conversion is not None:
                raise ValueError("an input recipe's projection refuses conversion: a display rescale is not a "
                                 "projection; give output_dtype float32 or float64 for sum and mean")
            if projection.method == "max" and projection.output_dtype is not None:
                raise ValueError("an input recipe's projection max keeps the dtype and takes no output_dtype")
            if projection.output_dtype is not None and np.dtype(projection.output_dtype).name not in PROJECTION_DTYPES:
                raise ValueError(f"an input recipe's projection {projection.method} has output_dtype float32 (the "
                                 f"default) or float64; got {projection.output_dtype!r}")
        if self.model_input is not None and not isinstance(self.model_input, ModelInputConfig):
            raise TypeError("input recipe model_input must be a ModelInputConfig or None")
        if self.name is not None and (not isinstance(self.name, str) or not self.name):
            raise TypeError("input recipe name must be a nonempty string or None")


# --- The combine stage -------------------------------------------------------------------------------------

@dataclass(frozen=True)
class CombineSpec:
    """Registered combine method of an input recipe: from prepared source channels to model channels.

    Parameters
    ----------
    name : str
        Stable snake_case name of the combine method.
    run : callable
        ``run(images, config, metadata)`` with ``images`` mapping each source
        role to its ZYX array, returning the model channels (a list of ZYX
        arrays in order), their record (a mapping, or None) and the values used
        on the grid (a mapping).
    source_roles : frozenset of str or None
        The source roles the method takes, exactly; ``None`` takes any roles,
        each at most once.
    output_roles : tuple of str or None
        The roles of the model channels in order; ``None`` keeps the source
        roles.
    requires : tuple of Dependency
        Optional dependencies; the empty default declares none.
    min_shape_zyx : tuple of int
        Smallest accepted ZYX shape, the shared registry field.
    """

    name: str
    run: Callable[..., tuple[list, Mapping | None, Mapping]]
    source_roles: frozenset[str] | None
    output_roles: tuple[str, ...] | None
    _: KW_ONLY
    requires: tuple[Dependency, ...] = ()
    min_shape_zyx: tuple[int, int, int] = (1, 1, 1)

    def __post_init__(self):
        check_shared(self, "combine method")


def _single(images, config, metadata):
    return list(images.values()), None, {}


def _composite(images, config, metadata):
    output, record = composite_nuclei_amplicon(images["nuclear"], images["amplicon"], config=config)
    return [output], record, {}


def _flamingo(images, config, metadata):
    radius = yx_pixels(config.median_radius_um, metadata, "FlamingoEnhancementConfig.median_radius_um")
    used = replace(config, median_radius_px=radius, median_radius_um=None)
    output, record = enhance_with_flamingo(images["nuclear"], images["cytoplasm"], config=used)
    return [output], record, {"median_radius_px": radius}


#: The combine registry of an input recipe, mapping each exact config type (``NoneType`` for ``single``, the
#: default) to its CombineSpec. A new composite is a new method here.
COMBINE_METHODS: dict[type, CombineSpec] = {
    type(None): CombineSpec("single", _single, None, None),
    CompositeConfig: CombineSpec("composite", _composite, frozenset({"nuclear", "amplicon"}), ("composite",)),
    FlamingoEnhancementConfig: CombineSpec("flamingo_enhance", _flamingo, frozenset({"nuclear", "cytoplasm"}),
                                           ("nuclear",)),
}


def combine_spec(config) -> CombineSpec:
    """The CombineSpec registered for ``type(config)`` exactly (None: ``single``); TypeError otherwise."""
    return spec_for(COMBINE_METHODS, config, "combine method", TypeError,
                    f"no combine method is registered for {type(config).__qualname__}; an input recipe's combine is "
                    "None (single), a CompositeConfig (composite) or a FlamingoEnhancementConfig (flamingo_enhance)")


def model_roles(recipe: InputRecipe | None, source_roles: Sequence[str], *, run: str | None = None) -> tuple[str, ...]:
    """The model roles of a run whose sources have ``source_roles``, after checking the recipe's roles.

    The source roles must be exactly those of the combine method (``single``:
    any), and each step's ``roles`` must be roles of its stage (source roles in
    ``channels``, model roles in ``after_projection``); ValueError naming the
    run, the method or the step and the roles otherwise.
    """
    where = "input recipe" if run is None else f"run {run!r}"
    source_roles = tuple(source_roles)
    recipe = recipe if recipe is not None else InputRecipe()
    spec = combine_spec(recipe.combine)
    if spec.source_roles is not None and set(source_roles) != spec.source_roles or \
            len(set(source_roles)) != len(source_roles):
        raise ValueError(f"{where}: the combine method {spec.name!r} takes the source roles "
                         f"{sorted(spec.source_roles or [])} exactly; the run's sources have {list(source_roles)}")
    output = source_roles if spec.output_roles is None else spec.output_roles
    for stage, roles in (("channels", source_roles), ("after_projection", output)):
        for item in getattr(recipe, stage):
            missing = [r for r in item.roles or () if r not in roles]
            if missing:
                raise ValueError(f"{where}: step {step_spec(item.config).name!r} of stage {stage} names the role "
                                 f"{missing[0]!r}, which the {'source' if stage == 'channels' else 'model'} channels "
                                 f"{list(roles)} do not have")
    return tuple(output)


# --- Identity ------------------------------------------------------------------------------------------------

def _step_entry(item):
    entry = {"method": step_spec(item.config).name, **config_record(item.config)}
    if item.roles is not None:
        entry["roles"] = list(item.roles)
    return entry


def recipe_config(recipe: InputRecipe | None) -> dict:
    """The recipe as given, in the layout of the configuration (µm, the configs' fields), as JSON values."""
    recipe = recipe if recipe is not None else InputRecipe()
    combine = None if recipe.combine is None else {"method": combine_spec(recipe.combine).name,
                                                   **config_record(recipe.combine)}
    projection = recipe.projection
    model_input = recipe.model_input
    return _json({
        "channels": [_step_entry(item) for item in recipe.channels], "combine": combine,
        "projection": None if projection is None else {"method": projection.method,
                                                       "output_dtype": projection.output_dtype},
        "after_projection": [_step_entry(item) for item in recipe.after_projection],
        "model_input": None if model_input is None else {"pixel_size_um": model_input.pixel_size_um,
                                                         "factor": model_input.factor}})


def recipe_sha256(recipe: InputRecipe | None) -> str:
    """SHA-256 of the recipe's canonical JSON (sorted keys, the configs as given); None is the empty recipe.

    The recipe's ``name`` is not part of it, so one recipe given by name and inline has one SHA-256.
    """
    text = json.dumps(recipe_config(recipe), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


# --- The model input --------------------------------------------------------------------------------------

def _six(value: float) -> float:
    """``value`` rounded to six significant digits."""
    return float(f"{value:.6g}")


def model_factor(model_input: ModelInputConfig | None, metadata, *, k: int = 1) -> dict | None:
    """The model input's record on a grid of ``metadata``: the factor ZYX, unrounded and rounded.

    ``pixel_size_um`` gives ``spacing / pixel_size_um`` per axis (Z factor 1 for
    a number), rounded to six significant digits; ``factor`` gives itself, times
    ``k`` in Y and X. None without a model input. ValueError naming
    ``model_input.pixel_size_um`` and the spacing keys on a grid without spacing.
    """
    if model_input is None:
        return None
    spacing = None if metadata is None else metadata.spacing_zyx
    if model_input.pixel_size_um is not None:
        if spacing is None:
            raise ValueError(f"input_recipe.model_input.pixel_size_um needs the grid's spacing, which it does not "
                             f"have: set {SPACING_KEYS} (micrometres), or give model_input.factor")
        size = model_input.pixel_size_um
        sizes = size if isinstance(size, tuple) else (None, size, size)
        exact = [1.0 if p is None else float(s) / p for s, p in zip(spacing, sizes)]
        rounded = [1.0 if p is None else _six(e) for e, p in zip(exact, sizes)]
    else:
        factor = model_input.factor
        fz, fy, fx = factor if isinstance(factor, tuple) else (1.0, factor, factor)
        exact = rounded = [float(fz), float(fy) * k, float(fx) * k]
    return {"pixel_size_um": _json(model_input.pixel_size_um), "factor": _json(model_input.factor), "k": int(k),
            "spacing_yx": None if spacing is None else [float(spacing[1]), float(spacing[2])],
            "factor_zyx_exact": list(exact), "factor_zyx": list(rounded)}


def check_model_input(method, recipe: InputRecipe | None, *, run: str, factor_zyx=None) -> None:
    """The model input of a run against its method, at configuration (design/specs/segmentation-input-recipe.md).

    A method that scales its input for its model (``SegmentationSpec.scales``)
    needs ``model_input``, and one that does not refuses it. With the factor
    known (``factor_zyx``; a ``factor`` model input gives it without a grid),
    Cellpose needs equal Y and X factors, a 3D Cellpose run (``do_3d``) whose
    model input sets a Z factor needs it equal to ``anisotropy × f_yx``, the Z
    factor Cellpose applies (amendment A), and StarDist's block-wise prediction
    needs a factor of 1. Each failure raises ValueError naming the run.
    """
    from ._learned import CellposeConfig, StarDistConfig
    spec = spec_for(SEGMENTATION_METHODS, method, "segmentation method")
    model_input = None if recipe is None else recipe.model_input
    if spec.scales and model_input is None:
        raise ValueError(f"run {run!r}: method {spec.name!r} names a model and needs input_recipe.model_input, the "
                         "pixel size the model sees (model_input.pixel_size_um; model_input.factor on a grid "
                         "without spacing); it has no default")
    if not spec.scales and model_input is not None:
        reason = ("cell outlines are never computed on a shrunk grid" if spec.name == "seeded_watershed"
                  else "the method does not scale its input")
        raise ValueError(f"run {run!r}: method {spec.name!r} refuses input_recipe.model_input: {reason}")
    if model_input is None:
        return
    if factor_zyx is None and model_input.factor is not None:
        factor_zyx = model_factor(model_input, None)["factor_zyx"]
    if factor_zyx is None:
        return
    fz, fy, fx = (float(f) for f in factor_zyx)
    if isinstance(method, CellposeConfig):
        if not math.isclose(fy, fx, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f"run {run!r}: Cellpose scales Y and X by one factor (diameter 30 / f_yx); the model "
                             f"input gives Y {fy} and X {fx}")
        if method.do_3d and model_input.sets_z:
            applied = float(method.anisotropy) * fy
            if not math.isclose(fz, applied, rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError(f"run {run!r}: input_recipe.model_input sets the Z factor {fz!r}, but a 3D Cellpose "
                                 f"run scales Z by anisotropy × f_yx = {applied!r}; give a number for Y and X only, "
                                 "or a Z factor equal to it")
    if isinstance(method, StarDistConfig) and method.block_size is not None and any(f != 1 for f in (fz, fy, fx)):
        # W-309
        raise ValueError(f"run {run!r}: block-wise prediction (block_size) together with a model input factor "
                         f"other than 1 ({[fz, fy, fx]}) is not supported: nothing has tested that combination")


# --- Applying a recipe ---------------------------------------------------------------------------------------

def _used(config, metadata, shape):
    """The pixel values of a step on the run grid (``used`` of its record)."""
    if isinstance(config, GaussianConfig):
        return gaussian_used(config, metadata, shape)
    if isinstance(config, ClaheConfig):
        return clahe_used(config, metadata, shape)
    if isinstance(config, (TophatConfig, ReconstructionConfig)):
        return {"radius_yx": yx_pixels(config.radius_um, metadata, f"{type(config).__name__}.radius_um")}
    return {}


def _working(image):
    """WT1: a channel's image before its first step, converted with ``img_as_float32``."""
    from skimage.util import img_as_float32
    return np.ascontiguousarray(img_as_float32(image))


def _run_steps(stage, steps, channels, roles, metadata, run, mask, records):
    """Each step of a stage on the channels it names, in order; returns the channels."""
    channels = list(channels)
    stepped = [False] * len(channels)
    for index, item in enumerate(steps):
        name = step_spec(item.config).name
        for c, role in enumerate(roles):
            if item.roles is not None and role not in item.roles:
                continue
            before = channels[c]
            image = before if stepped[c] else _working(before)
            used = _used(item.config, metadata, image.shape[:3])
            context = StepContext(role, run or "", metadata, fit_mask=mask)
            result = run_step(image, item.config, context)
            channels[c], stepped[c] = np.ascontiguousarray(result.image), True
            records.append({"stage": stage, "index": index, "role": role, "method": name,
                            "config": _json(config_record(item.config)), "used": _json(used),
                            "fitted": _json(dict(result.fitted)), "diagnostics": _json(dict(result.diagnostics)),
                            "input": array_sha256(before), "output": array_sha256(channels[c]),
                            "dtype": str(channels[c].dtype)})
    return channels


def _project(channels, projection, run, projected):
    """The projection stage on each model channel (ZYX) in its own dtype: the projected channels and the record entry.

    ``max`` keeps each channel's dtype; ``sum`` and ``mean`` accumulate in float64 and are rounded once to the output
    dtype, after the overflow check over every channel. ``exact`` is true when every sum of integer channels is
    held by the output dtype (|sum| ≤ 2²⁴ for float32, 2⁵³ for float64), and null for ``max``, ``mean`` and a float
    channel.
    """
    method = projection.method
    output = np.dtype(projection.output_dtype or "float32")
    entry = {"stage": "projection", "method": method, "config": {"method": method,
                                                                 "output_dtype": projection.output_dtype},
             "inputs": [array_sha256(image) for image in channels]}
    if projected is not None:
        # After fusion the sources are already Z maxima (or single planes): the stage is not applied.
        results = list(channels) if method == "max" else [image.astype(output) for image in channels]
        entry.update(applied=False, reason=projected, exact=None,
                     rounding=None if method == "max" else f"float64 result rounded once to {output.name}",
                     max_abs=None)
    elif method == "max":
        results = [np.max(image, axis=0, keepdims=True) for image in channels]
        entry.update(applied=True, exact=None, rounding=None, max_abs=None)
    else:
        if any(image.dtype.kind in "ui" and image.dtype.itemsize > 4 for image in channels):
            raise ValueError(f"run {run!r}: the input recipe's projection {method} refuses 64-bit integer input")
        values = []
        for image in channels:
            total = np.sum(image, axis=0, keepdims=True, dtype=np.float64)
            values.append(total if method == "sum" else total / image.shape[0])
        max_abs = max((float(np.max(np.abs(v))) for v in values if v.size), default=0.0)
        limit = float(np.finfo(output).max)
        if not math.isfinite(max_abs) or max_abs > limit:
            raise ValueError(f"run {run!r}: the input recipe's projection {method} reaches the magnitude "
                             f"{max_abs!r}, outside the finite range of its output dtype {output.name} (magnitude at "
                             f"most {limit!r}); set the projection's output_dtype: float64")
        results = [v.astype(output) for v in values]
        exact = None
        if method == "sum" and all(image.dtype.kind in "ui" for image in channels):
            exact = bool(max_abs <= _EXACT_BOUND[output.name])
        entry.update(applied=True, exact=exact, rounding=f"float64 result rounded once to {output.name}",
                     max_abs=max_abs)
    results = [np.ascontiguousarray(image) for image in results]
    entry["dtype"] = str(np.result_type(*results))
    entry["outputs"] = [array_sha256(image) for image in results]
    return results, entry


def apply_input_recipe(sources: Sequence[tuple[str, np.ndarray, Mapping]], grid: ReferenceGrid, *,
                       recipe: InputRecipe | None, fit_mask: np.ndarray | None = None, k: int = 1,
                       run: str | None = None, projected: str | None = None,
                       source: str = "given") -> tuple[SegmentationInput, np.ndarray | None, dict]:
    """Apply an input recipe to a run's source channels; return the model input, the unprojected channels and the
    recipe record.

    The one function that applies a recipe: ``FOV.segment`` (per FOV),
    ``starfinder.export.segment_fused`` (after fusion) and
    :func:`preview_recipes` call it, and it reads and writes no file. The
    stages run in order: ``channels`` (each step through ``run_step`` on the
    source channels it names, converted with ``img_as_float32`` before their
    first step), ``combine`` (:data:`COMBINE_METHODS`), ``projection``
    (``max`` keeps the dtype; ``sum`` and ``mean`` accumulate in float64 and
    are rounded once to the output dtype, refused with ValueError when a value
    lies outside its finite range), ``after_projection`` and ``model_input``
    (``model_factor``). Every length in µm is converted with the run grid's
    spacing before any projection. Each step's ``StepContext`` has the run
    grid's metadata, ``round_name`` the source or model role and
    ``reference_round`` the run's name.

    Parameters
    ----------
    sources : sequence of (str, numpy.ndarray, Mapping)
        ``(source role, ZYX array, source record)`` per source channel, in the
        order of the run's inputs.
    grid : ReferenceGrid
        The run grid of the sources.
    recipe : InputRecipe or None
        The recipe; None is the empty recipe.
    fit_mask : numpy.ndarray or None
        Boolean ZYX mask of the fit region, the pixels the global statistics of
        ``percentile_normalization`` and ``clahe`` are taken from; None is the
        whole image.
    k : int
        The fusion factor a ``factor`` model input is multiplied by in Y and X
        (1 per FOV).
    run : str or None
        The run's name, for the steps' context and the messages.
    projected : str or None
        None when the sources are not projected (per FOV); otherwise why they
        are already Z projections, and the projection stage is recorded as not
        applied with that reason and keeps the grid: ``fused_maximum`` (after
        fusion of tiles with Z > 1, which refuses a ``channels`` step and a
        ``sum`` or ``mean`` projection), ``single_plane`` (after fusion of
        tiles with one plane) or ``input_file`` (the legacy rule's projected
        input file).
    source : str
        Where the recipe comes from, recorded: ``given`` (a recipe of the run,
        or the empty recipe) or ``legacy`` (the implicit recipe of
        ``rules.stardist_segmentation``).

    Returns
    -------
    tuple[SegmentationInput, numpy.ndarray or None, dict]
        The model input (its image, its grid: the run grid or its projection,
        the model roles, a source mapping per model channel and
        ``model_scale_zyx``), the model channels before the projection as ZYXC
        (for ``extend_labels_through_z``; None without an applied projection)
        and the recipe record of design/specs/segmentation-input-recipe.md
        ("Records").

    Raises
    ------
    TypeError
        A recipe that is not an InputRecipe, or a grid that is not a ReferenceGrid.
    ValueError
        Source roles that differ from the combine method's, a step role the
        channels do not have, a length in µm on a grid without spacing (naming
        the field and the keys ``voxel_size_xy`` and ``voxel_size_z``), a
        projection value outside its output dtype's finite range (naming
        ``output_dtype: float64``), a step or projection the projected sources
        refuse, or a source of another shape than the grid.
    """
    if recipe is not None and not isinstance(recipe, InputRecipe):
        raise TypeError("recipe must be an InputRecipe or None")
    if not isinstance(grid, ReferenceGrid):
        raise TypeError("grid must be a ReferenceGrid")
    if projected not in (None, "fused_maximum", "single_plane", "input_file"):
        raise ValueError(f"projected must be None, fused_maximum, single_plane or input_file; got {projected!r}")
    given = recipe
    recipe = recipe if recipe is not None else InputRecipe()
    recipe.__post_init__()
    where = "input recipe" if run is None else f"run {run!r}"
    roles = tuple(role for role, _, _ in sources)
    arrays = [np.ascontiguousarray(np.asarray(image)) for _, image, _ in sources]
    records = [dict(record) for _, _, record in sources]
    for role, image in zip(roles, arrays):
        if image.shape != grid.shape_zyx:
            raise ValueError(f"{where}: the {role} source has shape {image.shape}, the run grid {grid.shape_zyx}")
    out_roles = model_roles(recipe, roles, run=run)
    if projected == "fused_maximum":
        if recipe.channels:
            raise ValueError(f"{where}: the tiles have more than one plane, and after fusion a step of stage channels "
                             "cannot run before their Z maximum; give 2D steps in stage after_projection")
        if recipe.projection is not None and recipe.projection.method != "max":
            raise ValueError(f"{where}: the tiles have more than one plane, and after fusion a projection "
                             f"{recipe.projection.method} cannot be recovered from their Z maximum; give projection "
                             "max or none, and 2D steps in stage after_projection")
    metadata = grid.metadata
    spacing = metadata.spacing_zyx
    model_input = model_factor(recipe.model_input, metadata, k=k)
    if fit_mask is not None:
        fit_mask = np.asarray(fit_mask)
        if fit_mask.dtype != bool or fit_mask.shape != grid.shape_zyx:
            raise ValueError(f"{where}: fit_mask must be a boolean array of the grid's ZYX shape {grid.shape_zyx}")
    stages = []
    # channels
    channels = _run_steps("channels", recipe.channels, arrays, roles, metadata, run, fit_mask, stages)
    # combine
    spec = combine_spec(recipe.combine)
    model, combined, used = spec.run(dict(zip(roles, channels)), recipe.combine, metadata)
    model = [np.ascontiguousarray(image) for image in model]
    stages.append({"stage": "combine", "method": spec.name,
                   "config": None if recipe.combine is None else _json(config_record(recipe.combine)),
                   "used": _json(dict(used)), "record": None if combined is None else _json(dict(combined)),
                   "roles_in": list(roles), "roles_out": list(out_roles),
                   "inputs": [array_sha256(image) for image in channels],
                   "outputs": [array_sha256(image) for image in model]})
    # projection
    unprojected, run_grid, mask = None, grid, fit_mask
    if recipe.projection is not None:
        projected_channels, entry = _project(model, recipe.projection, run, projected)
        if projected is None:
            unprojected = np.ascontiguousarray(np.stack(model, axis=-1))
            run_grid = grid.projected(method=recipe.projection.method)
            mask = None if fit_mask is None else fit_mask.any(axis=0, keepdims=True)
        entry["grid"] = {"shape_zyx": list(run_grid.shape_zyx), "frame_id": run_grid.metadata.frame_id}
        stages.append(entry)
        model = projected_channels
    # after_projection
    model = _run_steps("after_projection", recipe.after_projection, model, out_roles, metadata, run, mask, stages)
    image = np.ascontiguousarray(np.stack(model, axis=-1))
    if spec.output_roles is None:
        model_sources = tuple(records)
    else:
        model_sources = ({"combine": spec.name,
                          "sources": [{"role": role, **record} for role, record in zip(roles, records)]},)
    factor = None if model_input is None else tuple(model_input["factor_zyx"])
    segmentation_input = SegmentationInput(image, run_grid, out_roles, model_sources, model_scale_zyx=factor)
    record = {"name": recipe.name, "source": source, "sha256": recipe_sha256(given),
              "config": recipe_config(given),
              "spacing_zyx": None if spacing is None else [float(s) for s in spacing],
              "spatial_unit": None if spacing is None else metadata.spatial_unit,
              "spacing_source": SPACING_SOURCE,
              "fit_region": "image" if fit_mask is None else (
                  "footprints" if projected in ("fused_maximum", "single_plane") else "mask"),
              "fit_pixels": int(np.prod(grid.shape_zyx)) if fit_mask is None else int(np.count_nonzero(fit_mask)),
              "stages": stages, "model_input": model_input}
    return segmentation_input, unprojected, record


def method_value(record: Mapping) -> dict | None:
    """The model scale a method applied, from its ``effective`` record (``scale_zyx`` and its library value)."""
    effective = ((record.get("methods") or [{}])[0] or {}).get("effective") or {}
    if "scale_zyx" not in effective:
        return None
    keys = ("scale_zyx", "library_scale", "diameter", "anisotropy")
    return {key: effective[key] for key in keys if key in effective}
