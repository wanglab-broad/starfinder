"""Shared estimation/application without fallback or benchmark dependencies."""

import numpy as np

from starfinder.image import ImageMetadata, IncompatibleGeometryError, _validate_image

from ._config import CpdConfig, DemonsConfig, TpsConfig, TranslationConfig, WarpConfig
from ._errors import (
    InvalidRegistrationConfigError,
    RegistrationBackendUnavailableError,
    RegistrationEstimationError,
    UnsupportedTransformOperationError,
)
from ._resampling import _cast_warp_output, _output_dtype, apply_tps_deformation
from ._types import (
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
    config: TranslationConfig | DemonsConfig | TpsConfig | CpdConfig,
    reference_metadata: ImageMetadata,
    moving_metadata: ImageMetadata,
) -> RegistrationResult:
    """Estimate from finite ZYX arrays on equal grids without mutating inputs.

    Configuration identity selects the method. Translation supports singleton
    axes; current local estimators require 3D (demons axes >=4). Unknown physical
    geometry is accepted when explicitly unknown in both metadata values.
    No algorithm substitution occurs, including on insufficient landmarks.
    """
    if type(config) not in (TranslationConfig, DemonsConfig, TpsConfig, CpdConfig):
        raise InvalidRegistrationConfigError("expected a typed registration config")
    reference_image = _validate_image(reference_image, ndim=(3,))
    moving_image = _validate_image(moving_image, ndim=(3,))
    _geometry(reference_image.shape, moving_image.shape, reference_metadata, moving_metadata)
    geometry = dict(
        reference_shape_zyx=reference_image.shape,
        moving_shape_zyx=moving_image.shape,
        reference_metadata=reference_metadata,
        moving_metadata=moving_metadata,
    )
    if not isinstance(config, TranslationConfig) and min(reference_image.shape) < (
        4 if isinstance(config, DemonsConfig) else 2
    ):
        raise IncompatibleGeometryError("local estimator requires 3D; demons axes must be >=4")
    try:
        if isinstance(config, TranslationConfig):
            if config.backend == "scipy_fft":
                from ._translation import phase_correlate

                shift = phase_correlate(reference_image, moving_image, workers=config.fft_workers)
            else:
                from ._skimage_backend import phase_correlate_skimage

                shift = phase_correlate_skimage(reference_image, moving_image)
            transform = TranslationTransform(tuple(-s for s in shift), **geometry)
            backend = config.backend
            application = WarpConfig(fft_workers=config.fft_workers)
        elif isinstance(config, DemonsConfig):
            from ._demons import demons_register

            field = demons_register(
                reference_image,
                moving_image,
                iterations=config.iterations,
                smoothing_sigma=config.smoothing_sigma,
                method=config.variant,
                pyramid_mode=config.pyramid_mode,
            )
            transform = DenseDisplacementTransform(field, **geometry)
            backend = "simpleitk"
            application = WarpConfig(backend=backend)
        else:
            common = dict(
                detection_threshold=config.detection_noise_sigma,
                max_control_points=config.max_control_points,
                grid_spacing=config.grid_spacing_voxels,
                zoom_order=config.interpolation_order,
                field_smooth_sigma=config.field_smoothing_sigma,
                clamp_sampling_coordinates=config.clamp_sampling_coordinates,
            )
            if isinstance(config, TpsConfig):
                from ._tps import tps_register

                field = tps_register(
                    reference_image,
                    moving_image,
                    match_distance=config.match_distance_voxels,
                    min_matches=config.min_matches,
                    smoothing=config.smoothing,
                    **common,
                )
            else:
                from ._cpd import cpd_register

                field = cpd_register(
                    reference_image,
                    moving_image,
                    beta=config.kernel_width_voxels,
                    lmbda=config.regularization_weight,
                    w=config.outlier_fraction,
                    affine_first=config.affine_first,
                    candidate_radius=config.candidate_radius_voxels,
                    k_neighbors=config.neighbors_per_anchor,
                    **common,
                )
            transform = DenseDisplacementTransform(field, **geometry)
            backend = "scipy"
            application = WarpConfig(backend=backend)
    except ImportError as exc:
        raise RegistrationBackendUnavailableError(str(exc)) from exc
    except (np.linalg.LinAlgError, RuntimeError, ValueError) as exc:
        if isinstance(exc, RegistrationEstimationError):
            raise
        raise RegistrationEstimationError(f"{config.method} estimation failed: {exc}") from exc
    return RegistrationResult(
        transform, RegistrationDiagnostics(config.method, backend, config), application
    )


def apply_transform(
    moving_image: np.ndarray,
    transform: TranslationTransform | DenseDisplacementTransform,
    *,
    config: WarpConfig,
) -> np.ndarray:
    """Apply once to ZYX/ZYXC, reusing the transform across channels.

    Returns an array on transform.reference_metadata/grid, preserving channel
    order and input dtype by default. Translations stay compact; SciPy uses
    slice-sized coordinate arrays; SimpleITK prepares one field/resampler.
    """
    if not isinstance(config, WarpConfig):
        raise InvalidRegistrationConfigError("expected WarpConfig")
    image = _validate_image(moving_image)
    if not isinstance(transform, (TranslationTransform, DenseDisplacementTransform)):
        raise UnsupportedTransformOperationError("unsupported transform type")
    if image.shape[:3] != transform.moving_shape_zyx:
        raise IncompatibleGeometryError("moving image does not match transform grid")
    if isinstance(transform, TranslationTransform) != (config.backend == "translation"):
        raise UnsupportedTransformOperationError("transform incompatible with application backend")
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

            output[..., c] = apply_shift(
                volume,
                transform.correction_zyx,
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
