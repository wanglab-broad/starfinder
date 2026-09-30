"""Shared estimation/application without fallback or benchmark dependencies."""

import numpy as np

from starfinder._registry import require, spec_for
from starfinder.image import ImageMetadata, IncompatibleGeometryError, _validate_image

from ._config import WarpConfig
from ._errors import (
    InvalidRegistrationConfigError,
    RegistrationBackendUnavailableError,
    RegistrationEstimationError,
    UnsupportedTransformOperationError,
)
from ._chain import TransformChain, resample
from ._methods import REGISTRATION_METHODS, RegistrationConfig, _check_shape
from ._resampling import _cast_warp_output, _output_dtype, apply_tps_deformation
from ._types import (
    AffineTransform,
    BSplineTransform,
    DenseDisplacementTransform,
    RegistrationDiagnostics,
    RegistrationResult,
    TranslationTransform,
    _geometry,
)


def estimate_transform(
    reference_image: np.ndarray,
    moving_image: np.ndarray,
    *,
    config: RegistrationConfig,
    reference_metadata: ImageMetadata,
    moving_metadata: ImageMetadata,
) -> RegistrationResult:
    """Estimate from finite ZYX arrays on equal grids without mutating inputs.

    The exact config type selects the method in REGISTRATION_METHODS. Its
    declared dimensions and min_shape_zyx are checked before it runs:
    translation supports singleton axes; rigid, affine, B-spline and demons
    estimate Z=1 as 2D and reject 1 < Z < 4 (rigid, affine and B-spline need
    Y and X of at least 16, demons 4); TPS and CPD require 3D with every axis
    at least 2. Declared optional dependencies are imported before the
    estimator runs. Unknown physical geometry is accepted when explicitly
    unknown in both metadata values; the physical-space methods then use unit
    spacing and record spacing_source="unknown_unit". No algorithm
    substitution occurs, including on insufficient landmarks.
    """
    spec = spec_for(REGISTRATION_METHODS, config, "registration method", InvalidRegistrationConfigError,
                    "expected a typed registration config")
    reference_image = _validate_image(reference_image, ndim=(3,))
    moving_image = _validate_image(moving_image, ndim=(3,))
    _geometry(reference_image.shape, moving_image.shape, reference_metadata, moving_metadata)
    geometry = dict(
        reference_shape_zyx=reference_image.shape,
        moving_shape_zyx=moving_image.shape,
        reference_metadata=reference_metadata,
        moving_metadata=moving_metadata,
    )
    _check_shape(spec, reference_image.shape)
    require(spec, "registration method", RegistrationBackendUnavailableError)
    try:
        transform, backend, application, *details = spec.run(reference_image, moving_image, config, geometry)
    except ImportError as exc:
        raise RegistrationBackendUnavailableError(str(exc)) from exc
    except (np.linalg.LinAlgError, RuntimeError, ValueError) as exc:
        if isinstance(exc, RegistrationEstimationError):
            raise
        raise RegistrationEstimationError(f"{config.method} estimation failed: {exc}") from exc
    return RegistrationResult(
        transform, RegistrationDiagnostics(config.method, backend, config, **(details[0] if details else {})),
        application,
    )


def apply_transform(
    moving_image: np.ndarray,
    transform: TranslationTransform | AffineTransform | BSplineTransform | DenseDisplacementTransform | TransformChain,
    *,
    config: WarpConfig,
) -> np.ndarray:
    """Apply once to ZYX/ZYXC, reusing the transform across channels.

    Returns an array on transform.reference_metadata/grid, preserving channel
    order and input dtype by default. Translations stay compact; SciPy uses
    slice-sized coordinate arrays; SimpleITK prepares one field/resampler.
    Affine and B-spline transforms are applied through their dense() pull
    field with the scipy or simpleitk backend. A TransformChain is resampled
    once at its composite pull points: the translation backend applies only a
    chain of translations (its summed displacement); scipy (linear, plane by
    plane) or simpleitk (its pull_field()) sample any chain with their
    boundary policy.
    """
    if not isinstance(config, WarpConfig):
        raise InvalidRegistrationConfigError("expected WarpConfig")
    image = _validate_image(moving_image)
    if isinstance(transform, TransformChain):
        return resample([image], transform, config)[0]
    if not isinstance(transform, (TranslationTransform, AffineTransform, BSplineTransform,
                                  DenseDisplacementTransform)):
        raise UnsupportedTransformOperationError("unsupported transform type")
    if image.shape[:3] != transform.moving_shape_zyx:
        raise IncompatibleGeometryError("moving image does not match transform grid")
    if isinstance(transform, TranslationTransform) != (config.backend == "translation"):
        raise UnsupportedTransformOperationError("transform incompatible with application backend")
    if isinstance(transform, (AffineTransform, BSplineTransform)):
        transform = transform.dense()
    channels = image[..., None] if image.ndim == 3 else image
    output = np.empty(channels.shape, dtype=_output_dtype(image.dtype, config.output_dtype))
    if config.backend == "simpleitk":
        from ._demons import _import_sitk

        sitk = _import_sitk()
        field_image = sitk.GetImageFromArray(
            np.require(transform.displacement_zyx[..., ::-1], dtype=np.float64, requirements="C"),
            isVector=True,
        )
        prepared = sitk.DisplacementFieldTransform(field_image)
        resampler = sitk.ResampleImageFilter()
        resampler.SetTransform(prepared)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(config.fill_value)
        if config.boundary_mode == "nearest":
            resampler.UseNearestNeighborExtrapolatorOn()
    for c in range(channels.shape[-1]):
        volume = channels[..., c]
        if config.backend == "translation":
            from ._translation import apply_shift

            # apply_shift moves content by its shift: the negated pull displacement.
            output[..., c] = apply_shift(
                volume,
                tuple(-v for v in transform.displacement_zyx),
                workers=config.fft_workers,
                output_dtype=config.output_dtype,
            )
        elif config.backend == "scipy":
            output[..., c] = apply_tps_deformation(
                volume,
                transform.displacement_zyx,
                boundary_mode=config.boundary_mode,
                output_dtype=config.output_dtype,
                fill_value=config.fill_value,
            )
        else:
            source = sitk.GetImageFromArray(np.require(volume, dtype=np.float64, requirements="C"))
            resampler.SetReferenceImage(source)
            output[..., c] = _cast_warp_output(
                sitk.GetArrayFromImage(resampler.Execute(source)), output.dtype
            )
    return output[..., 0] if image.ndim == 3 else output
