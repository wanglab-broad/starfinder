"""Explicit index-space transform and result contracts."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from starfinder.image import ImageMetadata, IncompatibleGeometryError

from ._config import WarpConfig
from ._errors import UnsupportedTransformOperationError

if TYPE_CHECKING:
    from ._methods import RegistrationConfig


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
    """Compact pull map: moving index = reference index + displacement_zyx.

    Parameters
    ----------
    displacement_zyx : tuple[float, float, float]
        Detected displacement of moving content in ZYX voxel indices.
    reference_shape_zyx : tuple[int, int, int]
        Reference-grid shape as positive voxel counts in ZYX order.
    moving_shape_zyx : tuple[int, int, int]
        Moving-grid shape as positive voxel counts in ZYX order.
    reference_metadata : ImageMetadata
        Spatial geometry of the reference grid.
    moving_metadata : ImageMetadata
        Spatial geometry of the moving grid.
    direction : str
        Pull-map direction label, ``reference_to_moving``.
    units : str
        Pull-map coordinate unit, ``voxel_index``.

    Notes
    -----
    displacement_zyx is the detected displacement of the moving content, ZYX
    in voxel indices, like every other transform's pull displacement.
    """

    displacement_zyx: tuple[float, float, float]
    reference_shape_zyx: tuple[int, int, int]
    moving_shape_zyx: tuple[int, int, int]
    reference_metadata: ImageMetadata
    moving_metadata: ImageMetadata
    direction: str = "reference_to_moving"
    units: str = "voxel_index"

    def __post_init__(self):
        _validate_transform(self, "reference_to_moving")
        a = np.asarray(self.displacement_zyx, dtype=float)
        if a.shape != (3,) or not np.isfinite(a).all():
            raise IncompatibleGeometryError("displacement_zyx must be a finite triple")
        object.__setattr__(self, "displacement_zyx", tuple(float(x) for x in a))


@dataclass(frozen=True)
class DenseDisplacementTransform:
    """Pull field: moving index = reference index + displacement_zyx[index].

    Parameters
    ----------
    displacement_zyx : numpy.ndarray
        Finite float32 or float64 pull displacement on the reference grid, shaped ``(Z, Y, X, 3)`` with components ``(dz, dy, dx)`` in voxel indices.
    reference_shape_zyx : tuple[int, int, int]
        Reference-grid shape as positive voxel counts in ZYX order.
    moving_shape_zyx : tuple[int, int, int]
        Moving-grid shape as positive voxel counts in ZYX order.
    reference_metadata : ImageMetadata
        Spatial geometry of the reference grid.
    moving_metadata : ImageMetadata
        Spatial geometry of the moving grid.
    direction : str
        Pull-map direction label, ``reference_to_moving``.
    units : str
        Pull-map coordinate unit, ``voxel_index``.

    Notes
    -----
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


def _readonly(value, dtype, shape, name):
    a = np.array(value, dtype=dtype)
    if (shape is not None and a.shape != shape) or not np.isfinite(a).all():
        raise IncompatibleGeometryError(f"{name} must be finite with shape {shape}")
    a.flags.writeable = False
    return a


def _embed_field(field_xyz, shape, spacing_zyx):
    """Physical XYZ displacement of a 2D or 3D grid to a ZYX voxel field of shape (*shape, 3)."""
    dim = field_xyz.shape[-1]
    spacing = np.asarray(spacing_zyx, dtype=np.float64)
    out = np.zeros((*shape, 3), dtype=np.float64)
    out[..., 3 - dim:] = (field_xyz[..., ::-1] / spacing[3 - dim:]).reshape((*shape, dim))
    return out


@dataclass(frozen=True, eq=False)
class AffineTransform:
    """Linear pull map: moving index = A @ reference index + b (ZYX voxel indices).

    Parameters
    ----------
    matrix_zyx : numpy.ndarray
        Homogeneous 4-by-4 index-space pull matrix in ZYX order, stored as read-only float64.
    reference_shape_zyx : tuple[int, int, int]
        Reference-grid shape as positive voxel counts in ZYX order.
    moving_shape_zyx : tuple[int, int, int]
        Moving-grid shape as positive voxel counts in ZYX order.
    reference_metadata : ImageMetadata
        Spatial geometry of the reference grid.
    moving_metadata : ImageMetadata
        Spatial geometry of the moving grid.
    physical : dict or None
        Recorded physical-space estimation parameters; ``None`` means no physical parameter record was supplied.
    direction : str
        Pull-map direction label, ``reference_to_moving``.
    units : str
        Pull-map coordinate unit, ``voxel_index``.

    Notes
    -----
    matrix_zyx is the 4x4 float64 index-space matrix [[A, b], [0, 0, 0, 1]];
    it is read-only. physical holds the parameters of a physical-space
    estimate (matrix_xyz, center_xyz, translation_xyz of the pull map
    x -> M (x - c) + c + t, the backend's parameters and parameter_names, its
    transform name and spacing_zyx), or None. A Z=1 estimate has Z row and
    column (1, 0, 0) and b_z = 0.
    """

    matrix_zyx: np.ndarray
    reference_shape_zyx: tuple[int, int, int]
    moving_shape_zyx: tuple[int, int, int]
    reference_metadata: ImageMetadata
    moving_metadata: ImageMetadata
    physical: dict | None = None
    direction: str = "reference_to_moving"
    units: str = "voxel_index"

    def __post_init__(self):
        _validate_transform(self, "reference_to_moving")
        matrix = _readonly(self.matrix_zyx, np.float64, (4, 4), "matrix_zyx")
        if not np.array_equal(matrix[3], [0, 0, 0, 1]):
            raise IncompatibleGeometryError("the last row of matrix_zyx must be (0, 0, 0, 1)")
        object.__setattr__(self, "matrix_zyx", matrix)
        if self.physical is not None and not isinstance(self.physical, dict):
            raise IncompatibleGeometryError("physical must be a dict or None")

    def dense(self):
        """The pull displacement A p + b - p on the reference grid as a float64 DenseDisplacementTransform.

        Returns
        -------
        DenseDisplacementTransform
            Float64 ZYX pull displacement in voxel indices on the reference grid.
        """
        q = np.moveaxis(np.indices(self.reference_shape_zyx, dtype=np.float64), 0, -1)
        a, b = self.matrix_zyx[:3, :3], self.matrix_zyx[:3, 3]
        return DenseDisplacementTransform(q @ (a - np.eye(3)).T + b, **_geometry_fields(self))


@dataclass(frozen=True, eq=False)
class BSplineTransform:
    """Cubic B-spline pull map on a physical control grid: p + S^-1 P v(P S p).

    Parameters
    ----------
    grid_size_xyz : tuple[int, ...]
        Positive control-point counts in XYZ order.
    grid_origin_xyz : tuple[float, ...]
        Control-grid origin in XYZ physical coordinates.
    grid_spacing_xyz : tuple[float, ...]
        Positive control-grid spacings in XYZ physical units.
    grid_direction_xyz : tuple[float, ...]
        Row-major direction matrix of the control grid in XYZ order.
    coefficients : numpy.ndarray
        Control-point displacements in physical units, stored as read-only float64 with shape ``(dimension, *grid_size_xyz[::-1])`` and XYZ component order.
    spacing_zyx : tuple[float, float, float]
        Positive image voxel spacings in ZYX physical units used to convert the pull map to voxel indices.
    reference_shape_zyx : tuple[int, int, int]
        Reference-grid shape as positive voxel counts in ZYX order.
    moving_shape_zyx : tuple[int, int, int]
        Moving-grid shape as positive voxel counts in ZYX order.
    reference_metadata : ImageMetadata
        Spatial geometry of the reference grid.
    moving_metadata : ImageMetadata
        Spatial geometry of the moving grid.
    order : int
        B-spline degree, ``3`` for a cubic spline.
    direction : str
        Pull-map direction label, ``reference_to_moving``.
    units : str
        Pull-map coordinate unit, ``voxel_index``.

    Notes
    -----
    S is diag(spacing_zyx), P reverses ZYX to the XYZ order of ITK and v is
    the B-spline displacement in physical XYZ units. The grid fields are the
    ITK fixed parameters (XYZ order, dimension 2 for a Z=1 estimate):
    grid_size_xyz, grid_origin_xyz, grid_spacing_xyz and grid_direction_xyz
    (row-major). coefficients has shape ``(dimension, *grid_size_xyz[::-1])``:
    component d (X first) on the control grid, so its C-order ravel is the ITK
    parameter vector. Arrays are float64 and read-only; dense() evaluates the
    grid with SimpleITK, so a stored B-spline never needs elastix.
    """

    grid_size_xyz: tuple[int, ...]
    grid_origin_xyz: tuple[float, ...]
    grid_spacing_xyz: tuple[float, ...]
    grid_direction_xyz: tuple[float, ...]
    coefficients: np.ndarray
    spacing_zyx: tuple[float, float, float]
    reference_shape_zyx: tuple[int, int, int]
    moving_shape_zyx: tuple[int, int, int]
    reference_metadata: ImageMetadata
    moving_metadata: ImageMetadata
    order: int = 3
    direction: str = "reference_to_moving"
    units: str = "voxel_index"

    def __post_init__(self):
        _validate_transform(self, "reference_to_moving")
        dim = len(self.grid_size_xyz)
        if dim not in (2, 3) or dim != (2 if self.reference_shape_zyx[0] == 1 else 3):
            raise IncompatibleGeometryError("a B-spline has dimension 2 for Z=1 grids and 3 otherwise")
        if self.order != 3:
            raise UnsupportedTransformOperationError("only cubic B-splines are supported")
        size = tuple(int(n) for n in self.grid_size_xyz)
        if any(n < 1 for n in size):
            raise IncompatibleGeometryError("grid_size_xyz must be positive")
        object.__setattr__(self, "grid_size_xyz", size)
        for name, n in (("grid_origin_xyz", dim), ("grid_spacing_xyz", dim), ("grid_direction_xyz", dim * dim),
                        ("spacing_zyx", 3)):
            value = _readonly(getattr(self, name), np.float64, (n,), name)
            object.__setattr__(self, name, tuple(float(v) for v in value))
        if any(v <= 0 for v in self.grid_spacing_xyz + self.spacing_zyx):
            raise IncompatibleGeometryError("grid and image spacings must be positive")
        object.__setattr__(self, "coefficients",
                           _readonly(self.coefficients, np.float64, (dim, *size[::-1]), "coefficients"))

    def dense(self):
        """Evaluate the grid on the reference grid with SimpleITK; a float64 DenseDisplacementTransform.

        Returns
        -------
        DenseDisplacementTransform
            Float64 ZYX pull displacement in voxel indices evaluated on the reference grid.
        """
        from ._demons import _import_sitk

        sitk = _import_sitk()
        dim = len(self.grid_size_xyz)
        transform = sitk.BSplineTransform(dim, self.order)
        transform.SetFixedParameters([float(v) for v in (*self.grid_size_xyz, *self.grid_origin_xyz,
                                                         *self.grid_spacing_xyz, *self.grid_direction_xyz)])
        transform.SetParameters(self.coefficients.ravel().tolist())
        shape = self.reference_shape_zyx
        field = sitk.TransformToDisplacementField(
            transform, sitk.sitkVectorFloat64, [int(n) for n in shape[::-1][:dim]], [0.0] * dim,
            list(self.spacing_zyx[::-1][:dim]), np.eye(dim).ravel().tolist())
        return DenseDisplacementTransform(_embed_field(sitk.GetArrayFromImage(field), shape, self.spacing_zyx),
                                          **_geometry_fields(self))


def _geometry_fields(transform):
    return {name: getattr(transform, name) for name in
            ("reference_shape_zyx", "moving_shape_zyx", "reference_metadata", "moving_metadata")}


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
    """Estimator identity and measured diagnostics; unknown values stay None.

    Parameters
    ----------
    method : str
        Registration method identifier.
    backend : str
        Identifier of the backend that performed estimation.
    effective_config : RegistrationConfig
        Method configuration used for estimation.
    converged : bool or None
        Recorded convergence status; ``None`` means convergence is unknown.
    iterations_completed : int or tuple[int, ...] or None
        Recorded iteration count as a scalar or per-level tuple; ``None`` means unreported.
    reference_landmark_count : int or None
        Recorded number of reference landmarks; ``None`` means unreported.
    moving_landmark_count : int or None
        Recorded number of moving landmarks; ``None`` means unreported.
    matched_landmark_count : int or None
        Recorded number of matched landmarks; ``None`` means unreported.
    warnings : tuple[str, ...]
        Estimator warning messages; an empty tuple records no warnings.
    final_metric_value : tuple[float, ...] or None
        Recorded final similarity metric values per pyramid level; ``None`` means unreported.
    stop_condition : tuple[str, ...] or None
        Recorded stopping-condition descriptions per pyramid level; ``None`` means unreported.
    elapsed_iterations : tuple[int, ...] or None
        Recorded demons iteration counts per pyramid level; ``None`` means unreported.
    final_rms_change : tuple[float, ...] or None
        Recorded demons final RMS displacement changes in voxel units per pyramid level; ``None`` means unreported.
    backend_versions : dict[str, str or None] or None
        Recorded backend distribution versions, with ``None`` for an unavailable version; a ``None`` mapping means unreported.
    backend_parameters : dict[str, str] or None
        Recorded effective backend parameter map; ``None`` means unreported.
    spacing_source : str or None
        Recorded spacing provenance, ``metadata`` or ``unknown_unit``; ``None`` means unreported.

    Notes
    -----
    The elastix methods (rigid, affine, bspline) record converged=None (fixed
    iterations, no convergence test), iterations_completed, final_metric_value
    and stop_condition per pyramid level, backend_versions, the effective
    backend_parameters (the elastix parameter map, values space-separated) and
    spacing_source ("metadata", or "unknown_unit" with a warning when unit
    spacing replaced unknown spacing). Demons records elapsed_iterations and
    final_rms_change per level.
    """

    method: str
    backend: str
    effective_config: "RegistrationConfig"
    converged: bool | None = None
    iterations_completed: int | tuple[int, ...] | None = None
    reference_landmark_count: int | None = None
    moving_landmark_count: int | None = None
    matched_landmark_count: int | None = None
    warnings: tuple[str, ...] = ()
    final_metric_value: tuple[float, ...] | None = None
    stop_condition: tuple[str, ...] | None = None
    elapsed_iterations: tuple[int, ...] | None = None
    final_rms_change: tuple[float, ...] | None = None
    backend_versions: dict[str, str | None] | None = None
    backend_parameters: dict[str, str] | None = None
    spacing_source: str | None = None


@dataclass(frozen=True)
class RegistrationResult:
    """Estimated transform and its matching application policy.

    Parameters
    ----------
    transform : TranslationTransform or AffineTransform or BSplineTransform or DenseDisplacementTransform
        Estimated pull transform in voxel indices.
    diagnostics : RegistrationDiagnostics
        Estimator identity, settings and recorded diagnostics.
    application_config : WarpConfig
        Matching resampling policy returned by the estimator.

    Notes
    -----
    Applied arrays have transform.reference_metadata, never moving metadata.
    The repr is a one-line method and transform summary without field values.
    """

    transform: TranslationTransform | AffineTransform | BSplineTransform | DenseDisplacementTransform
    diagnostics: RegistrationDiagnostics
    application_config: WarpConfig

    def __repr__(self):
        transform = self.transform
        if isinstance(transform, TranslationTransform):
            detail = "displacement_zyx (" + ", ".join(f"{v + 0.0:g}" for v in transform.displacement_zyx) + ")"
        elif isinstance(transform, AffineTransform):
            detail = "affine matrix_zyx"
        elif isinstance(transform, BSplineTransform):
            detail = f"B-spline grid {transform.grid_size_xyz} (XYZ)"
        else:
            field_ = transform.displacement_zyx
            detail = f"dense field {tuple(field_.shape)} {field_.dtype}"
        converged = self.diagnostics.converged
        if converged is not None:
            detail += ", converged" if converged else ", not converged"
        return f"RegistrationResult: {self.diagnostics.method} ({self.diagnostics.backend}), {detail}"
