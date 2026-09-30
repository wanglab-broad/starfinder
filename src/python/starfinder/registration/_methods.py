"""The registration method registry: exact config type -> RegistrationSpec."""

from collections.abc import Callable
from dataclasses import KW_ONLY, dataclass

from starfinder._registry import Dependency, check_shared
from starfinder.image import IncompatibleGeometryError

from ._config import CpdConfig, DemonsConfig, TpsConfig, TranslationConfig
from ._cpd import estimate_cpd
from ._demons import estimate_demons
from ._tps import estimate_tps
from ._translation import estimate_translation

_STEP_KINDS = ("global", "local")
_TRANSFORM_KINDS = ("translation", "affine", "bspline", "dense")
_SPACES = ("index", "physical")


@dataclass(frozen=True)
class RegistrationSpec:
    """Registered method: stable snake_case name, private estimator and declared capabilities.

    run(reference, moving, config, geometry) returns the transform, the
    backend that ran and its application WarpConfig; estimate_transform calls
    it after the checks below and callers never call it directly.
    step_kind is "global" or "local" (the allowed recipe sequences).
    dimensions holds 2 when a Z=1 input is estimated as 2D and 3 when Z>1 is
    estimated in 3D. transform_kind is the kind of the returned transform
    ("translation", "affine", "bspline" or "dense"); space is "index" or
    "physical" estimation coordinates. requires lists optional dependencies,
    imported when the method runs; min_shape_zyx is the smallest accepted size
    of each axis (a Z=1 input is checked against its last two entries).
    """

    name: str
    run: Callable[..., tuple]
    _: KW_ONLY
    step_kind: str
    dimensions: frozenset[int]
    transform_kind: str
    space: str
    requires: tuple[Dependency, ...] = ()
    min_shape_zyx: tuple[int, int, int] = (1, 1, 1)

    def __post_init__(self):
        check_shared(self, "registration method")
        if self.step_kind not in _STEP_KINDS:
            raise ValueError(f"step_kind of {self.name!r} must be one of {_STEP_KINDS}")
        if (not isinstance(self.dimensions, frozenset) or not self.dimensions
                or not self.dimensions <= {2, 3}):
            raise ValueError(f"dimensions of {self.name!r} must be a nonempty frozenset of 2 and/or 3")
        if self.transform_kind not in _TRANSFORM_KINDS:
            raise ValueError(f"transform_kind of {self.name!r} must be one of {_TRANSFORM_KINDS}")
        if self.space not in _SPACES:
            raise ValueError(f"space of {self.name!r} must be one of {_SPACES}")


# The registration method registry (exported with its documentation by starfinder.registration).
REGISTRATION_METHODS: dict[type, RegistrationSpec] = {
    TranslationConfig: RegistrationSpec(
        "translation", estimate_translation, step_kind="global", dimensions=frozenset({2, 3}),
        transform_kind="translation", space="index"),
    DemonsConfig: RegistrationSpec(
        "demons", estimate_demons, step_kind="local", dimensions=frozenset({3}), transform_kind="dense",
        space="index", requires=(Dependency("SimpleITK", "SimpleITK", "local-registration"),),
        min_shape_zyx=(4, 4, 4)),
    TpsConfig: RegistrationSpec(
        "tps", estimate_tps, step_kind="local", dimensions=frozenset({3}), transform_kind="dense",
        space="index", min_shape_zyx=(2, 2, 2)),
    CpdConfig: RegistrationSpec(
        "cpd", estimate_cpd, step_kind="local", dimensions=frozenset({3}), transform_kind="dense",
        space="index", min_shape_zyx=(2, 2, 2)),
}

# Annotation alias for a registered config; a test keeps its members equal to the registry keys.
RegistrationConfig = TranslationConfig | DemonsConfig | TpsConfig | CpdConfig


def _check_shape(spec, shape):
    """Reject a ZYX shape the method does not accept, before it runs."""
    if shape[0] == 1 and 2 not in spec.dimensions:
        raise IncompatibleGeometryError(f"{spec.name} requires 3D input; Z=1 is not supported")
    if shape[0] > 1 and 3 not in spec.dimensions:
        raise IncompatibleGeometryError(f"{spec.name} supports only Z=1 input")
    minimum = spec.min_shape_zyx if shape[0] > 1 else (1, *spec.min_shape_zyx[1:])
    if any(n < m for n, m in zip(shape, minimum)):
        raise IncompatibleGeometryError(
            f"{spec.name} requires 3D input with every axis at least {minimum}, not {tuple(shape)}"
            if shape[0] > 1 else
            f"{spec.name} requires Y and X of at least {minimum[1:]}, not {tuple(shape[1:])}")
