"""Non-rigid registration using demons algorithm (SimpleITK)."""

from __future__ import annotations

import numpy as np


def _import_sitk():
    """Lazy import SimpleITK with helpful error message."""
    try:
        import SimpleITK as sitk
        return sitk
    except ImportError:
        raise ImportError(
            "SimpleITK required for local registration. "
            "Install with: uv add 'starfinder[local-registration]'"
        )


def demons_register(
    fixed: np.ndarray,
    moving: np.ndarray,
    iterations: list[int] | None = None,
    smoothing_sigma: float = 0.5,
    method: str = "diffeomorphic",
) -> np.ndarray:
    """Register moving image to fixed using demons algorithm.

    Performs non-rigid registration using SimpleITK's demons filters
    with multi-resolution pyramid for robustness.

    Parameters
    ----------
    fixed : np.ndarray
        Fixed (reference) volume with shape (Z, Y, X).
    moving : np.ndarray
        Moving volume to register, with shape (Z, Y, X).
    iterations : list[int] | None, optional
        Number of iterations at each pyramid level (coarse to fine).
        Default is [50] (single-level, no pyramid). Single-level works best
        for sparse fluorescence images; multi-level pyramids can degrade quality.
    smoothing_sigma : float, optional
        Standard deviation for Gaussian smoothing of the displacement field.
        Lower values (0.5) work better for sparse images. Default is 0.5.
    method : str, optional
        Demons variant to use. Options:
        - "diffeomorphic" (default): Topology-preserving, more stable
        - "symmetric": Standard symmetric forces demons
        - "fast_symmetric": Faster symmetric forces variant

    Returns
    -------
    np.ndarray
        Displacement field with shape (Z, Y, X, 3) where the last dimension
        contains (dz, dy, dx) displacement vectors.
    """
    sitk = _import_sitk()

    if iterations is None:
        # Single-level registration performs best for sparse fluorescence images.
        # Multi-resolution pyramid can degrade quality due to upsampling artifacts.
        iterations = [50]

    # Convert numpy arrays to SimpleITK images
    fixed_sitk = sitk.GetImageFromArray(fixed.astype(np.float32))
    moving_sitk = sitk.GetImageFromArray(moving.astype(np.float32))

    # Create demons registration filter based on method
    if method == "diffeomorphic":
        demons = sitk.DiffeomorphicDemonsRegistrationFilter()
        demons.SetStandardDeviations(smoothing_sigma)
        demons.SetUpdateFieldStandardDeviations(smoothing_sigma * 0.5)
    elif method == "fast_symmetric":
        demons = sitk.FastSymmetricForcesDemonsRegistrationFilter()
        demons.SetStandardDeviations(smoothing_sigma)
        demons.SetUpdateFieldStandardDeviations(smoothing_sigma)
    else:  # symmetric
        demons = sitk.SymmetricForcesDemonsRegistrationFilter()
        demons.SetStandardDeviations(smoothing_sigma)

    # Multi-resolution registration
    # Compute shrink factors, ensuring minimum dimension stays >= 2
    num_levels = len(iterations)
    min_dim = min(fixed.shape)
    max_shrink = max(1, min_dim // 2)  # Ensure coarsest level has at least 2 voxels
    shrink_factors = []
    for i in range(num_levels):
        ideal_shrink = 2 ** (num_levels - 1 - i)
        shrink_factors.append(min(ideal_shrink, max_shrink))

    # Initialize displacement field at coarsest level
    displacement_field = None

    for level, (shrink, num_iter) in enumerate(zip(shrink_factors, iterations)):
        # Shrink images for this pyramid level
        if shrink > 1:
            fixed_level = sitk.Shrink(fixed_sitk, [shrink] * 3)
            moving_level = sitk.Shrink(moving_sitk, [shrink] * 3)
        else:
            fixed_level = fixed_sitk
            moving_level = moving_sitk

        demons.SetNumberOfIterations(num_iter)

        if displacement_field is None:
            # First level: run demons directly
            displacement_field = demons.Execute(fixed_level, moving_level)
        else:
            # Subsequent levels: upsample previous field and use as initial
            # Expand displacement field to current level size
            current_size = fixed_level.GetSize()
            prev_size = displacement_field.GetSize()

            # Scale factors for upsampling
            scale = [c / p for c, p in zip(current_size, prev_size)]

            # Resample displacement field to current level size
            resampler = sitk.ResampleImageFilter()
            resampler.SetSize(current_size)
            resampler.SetOutputSpacing(fixed_level.GetSpacing())
            resampler.SetOutputOrigin(fixed_level.GetOrigin())
            resampler.SetOutputDirection(fixed_level.GetDirection())
            resampler.SetInterpolator(sitk.sitkLinear)

            displacement_field = resampler.Execute(displacement_field)

            # Scale the displacement vectors by the upsample factor
            displacement_field = sitk.Compose([
                sitk.VectorIndexSelectionCast(displacement_field, i) * scale[i]
                for i in range(3)
            ])

            # Run demons with initial displacement field
            displacement_field = demons.Execute(fixed_level, moving_level, displacement_field)

    # Convert back to numpy array
    # SimpleITK displacement field: GetArrayFromImage returns (Z, Y, X, 3)
    # where vector components are in SimpleITK order: (dx, dy, dz)
    field_array = sitk.GetArrayFromImage(displacement_field)

    # Convert from SimpleITK (dx, dy, dz) to numpy convention (dz, dy, dx)
    field_array = field_array[..., ::-1]

    return field_array


def apply_deformation(
    volume: np.ndarray,
    displacement_field: np.ndarray,
) -> np.ndarray:
    """Apply displacement field to warp a volume.

    Parameters
    ----------
    volume : np.ndarray
        Input volume with shape (Z, Y, X).
    displacement_field : np.ndarray
        Displacement field with shape (Z, Y, X, 3) where the last dimension
        contains (dz, dy, dx) displacement vectors.

    Returns
    -------
    np.ndarray
        Warped volume with same shape as input.
    """
    sitk = _import_sitk()

    # Preserve input dtype
    input_dtype = volume.dtype

    # Convert volume to SimpleITK image
    volume_sitk = sitk.GetImageFromArray(volume.astype(np.float64))

    # Convert displacement field from numpy convention (dz, dy, dx) to
    # SimpleITK convention (dx, dy, dz)
    field_sitk_order = displacement_field[..., ::-1]

    # Convert displacement field to SimpleITK (isVector=True for vector image)
    # DisplacementFieldTransform requires float64 vectors
    field_sitk = sitk.GetImageFromArray(
        field_sitk_order.astype(np.float64), isVector=True
    )

    # Create displacement field transform
    transform = sitk.DisplacementFieldTransform(field_sitk)

    # Resample the volume using the displacement field
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(volume_sitk)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(transform)

    warped_sitk = resampler.Execute(volume_sitk)

    # Convert back to numpy and preserve input dtype
    warped = sitk.GetArrayFromImage(warped_sitk)

    return warped.astype(input_dtype)


def register_volume_local(
    images: np.ndarray,
    ref_image: np.ndarray,
    mov_image: np.ndarray,
    iterations: list[int] | None = None,
    smoothing_sigma: float = 0.5,
    method: str = "diffeomorphic",
) -> tuple[np.ndarray, np.ndarray]:
    """Register multi-channel volume using demons.

    This function computes the displacement field between ref_image and mov_image,
    then applies that field to all channels in the images volume.

    Parameters
    ----------
    images : np.ndarray
        Multi-channel volume with shape (Z, Y, X, C).
    ref_image : np.ndarray
        Reference image with shape (Z, Y, X) for field calculation.
    mov_image : np.ndarray
        Moving image with shape (Z, Y, X) for field calculation.
    iterations : list[int] | None, optional
        Number of iterations per pyramid level.
        Defaults to [50] (single-level, no pyramid - best for sparse images).
    smoothing_sigma : float, optional
        Standard deviation for displacement field smoothing.
        Lower values (0.5) work better for sparse images. Default is 0.5.
    method : str, optional
        Demons variant: "diffeomorphic" (default), "symmetric", "fast_symmetric".

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Tuple of (registered_images, displacement_field).
        - registered_images: Warped volume with shape (Z, Y, X, C)
        - displacement_field: Computed field with shape (Z, Y, X, 3)
    """
    # Compute displacement field from reference and moving images
    displacement_field = demons_register(
        ref_image, mov_image, iterations=iterations,
        smoothing_sigma=smoothing_sigma, method=method
    )

    # Apply the displacement field to each channel
    n_channels = images.shape[-1]
    registered = np.empty_like(images)

    for c in range(n_channels):
        registered[:, :, :, c] = apply_deformation(images[:, :, :, c], displacement_field)

    return registered, displacement_field
