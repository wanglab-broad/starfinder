"""Explicit index-space transform and result contracts."""

from dataclasses import dataclass

import numpy as np

from starfinder.image import ImageMetadata, IncompatibleGeometryError

from ._config import CpdConfig, DemonsConfig, TpsConfig, TranslationConfig, WarpConfig
from ._errors import UnsupportedTransformOperationError


def _geometry(reference_shape, moving_shape, reference_metadata, moving_metadata):
    shapes = []
    for shape in (reference_shape, moving_shape):
        if (
            not isinstance(shape, (tuple, list))
            or len(shape) != 3
            or any(
                isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n <= 0 for n in shape
            )
        ):
            raise IncompatibleGeometryError("shapes must be positive integer ZYX triples")
        shapes.append(tuple(int(n) for n in shape))
    if shapes[0] != shapes[1]:
        raise IncompatibleGeometryError("only equal-shaped grids are supported")
    if not isinstance(reference_metadata, ImageMetadata) or not isinstance(
        moving_metadata, ImageMetadata
    ):
        raise IncompatibleGeometryError("both ImageMetadata values are required")
    # Distinct frame identifiers are expected for registration, but physical
    # grids must agree; no rescaling/reorientation or inferred calibration.
    for name in ("spacing_zyx", "origin_zyx", "direction_zyx", "spatial_unit"):
        if getattr(reference_metadata, name) != getattr(moving_metadata, name):
            raise IncompatibleGeometryError(f"unsupported grid conversion: {name}")
    return shapes


@dataclass(frozen=True)
class TranslationTransform:
    """Compact correction: pull source index = reference index - correction."""

    correction_zyx: tuple[float, float, float]
    reference_shape_zyx: tuple[int, int, int]
    moving_shape_zyx: tuple[int, int, int]
    reference_metadata: ImageMetadata
    moving_metadata: ImageMetadata
    direction: str = "moving_to_reference"
    units: str = "voxel_index"

    def __post_init__(self):
        _validate_transform(self, "moving_to_reference")
        a = np.asarray(self.correction_zyx, dtype=float)
        if a.shape != (3,) or not np.isfinite(a).all():
            raise IncompatibleGeometryError("correction_zyx must be a finite triple")
        object.__setattr__(self, "correction_zyx", tuple(float(x) for x in a))


@dataclass(frozen=True)
class DenseDisplacementTransform:
    """Pull field: moving index = reference index + displacement_zyx[index].

    Components are ZYX, in voxel indices on the reference grid. No inversion
    or physical-unit conversion is implicit. The owned field is read-only.
    """

    displacement_zyx: np.ndarray
    reference_shape_zyx: tuple[int, int, int]
    moving_shape_zyx: tuple[int, int, int]
    reference_metadata: ImageMetadata
    moving_metadata: ImageMetadata
    direction: str = "reference_to_moving"
    units: str = "voxel_index"

    def __post_init__(self):
        _validate_transform(self, "reference_to_moving")
        a = np.asarray(self.displacement_zyx)
        if (
            a.shape != (*self.reference_shape_zyx, 3)
            or a.dtype not in (np.dtype("float32"), np.dtype("float64"))
            or not np.isfinite(a).all()
        ):
            raise IncompatibleGeometryError(
                "displacement_zyx must be finite float32/64 reference ZYX3"
            )
        a = a.copy()
        a.flags.writeable = False
        object.__setattr__(self, "displacement_zyx", a)


def _validate_transform(transform, direction):
    if transform.direction != direction or transform.units != "voxel_index":
        raise UnsupportedTransformOperationError(
            "unsupported transform direction/units; no implicit inversion or conversion"
        )
    shapes = _geometry(
        transform.reference_shape_zyx,
        transform.moving_shape_zyx,
        transform.reference_metadata,
        transform.moving_metadata,
    )
    for name, shape in zip(("reference_shape_zyx", "moving_shape_zyx"), shapes):
        object.__setattr__(transform, name, shape)


@dataclass(frozen=True)
class RegistrationDiagnostics:
    """Estimator identity and measured diagnostics; unknown values stay None."""

    method: str
    backend: str
    effective_config: TranslationConfig | DemonsConfig | TpsConfig | CpdConfig
    converged: bool | None = None
    iterations_completed: int | None = None
    reference_landmark_count: int | None = None
    moving_landmark_count: int | None = None
    matched_landmark_count: int | None = None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class RegistrationResult:
    """Estimated transform and its matching application policy.

    Applied arrays have transform.reference_metadata, never moving metadata.
    """

    transform: TranslationTransform | DenseDisplacementTransform
    diagnostics: RegistrationDiagnostics
    application_config: WarpConfig
