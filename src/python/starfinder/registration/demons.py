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


def _create_demons_filter(sitk, method: str, smoothing_sigma: float):
    """Create a SimpleITK demons registration filter.

    Parameters
    ----------
    sitk : module
        SimpleITK module.
    method : str
        Demons variant name.
    smoothing_sigma : float
        Standard deviation for displacement field smoothing.

    Returns
    -------
    SimpleITK filter
        Configured demons registration filter.
    """
    if method == "demons":
        demons = sitk.DemonsRegistrationFilter()
        demons.SetStandardDeviations(smoothing_sigma)
    elif method == "diffeomorphic":
        demons = sitk.DiffeomorphicDemonsRegistrationFilter()
        demons.SetStandardDeviations(smoothing_sigma)
        demons.SetUpdateFieldStandardDeviations(smoothing_sigma * 0.5)
    elif method == "fast_symmetric":
        demons = sitk.FastSymmetricForcesDemonsRegistrationFilter()
        demons.SetStandardDeviations(smoothing_sigma)
        demons.SetUpdateFieldStandardDeviations(smoothing_sigma)
    elif method == "symmetric":
        demons = sitk.SymmetricForcesDemonsRegistrationFilter()
        demons.SetStandardDeviations(smoothing_sigma)
    else:
        raise ValueError(
            f"Unknown method: '{method}'. "
            f"Options: 'demons', 'diffeomorphic', 'symmetric', 'fast_symmetric'"
        )
    return demons


def _run_sitk_pyramid(sitk, fixed, moving, demons, iterations):
    """Run demons with SimpleITK's built-in (naive) multi-resolution pyramid.

    This is the original implementation. It uses ``sitk.Shrink`` for
    downsampling which does naive subsampling (every Nth voxel) with no
    anti-aliasing. Works acceptably for single-level registration but
    degrades quality for multi-level pyramids on sparse fluorescence images.
    """
    fixed_sitk = sitk.GetImageFromArray(fixed.astype(np.float32))
    moving_sitk = sitk.GetImageFromArray(moving.astype(np.float32))

    num_levels = len(iterations)
    min_dim = min(fixed.shape)
    max_shrink = max(1, min_dim // 2)
    shrink_factors = []
    for i in range(num_levels):
        ideal_shrink = 2 ** (num_levels - 1 - i)
        shrink_factors.append(min(ideal_shrink, max_shrink))

    displacement_field = None

    for shrink, num_iter in zip(shrink_factors, iterations):
        if shrink > 1:
            fixed_level = sitk.Shrink(fixed_sitk, [shrink] * 3)
            moving_level = sitk.Shrink(moving_sitk, [shrink] * 3)
        else:
            fixed_level = fixed_sitk
            moving_level = moving_sitk

        demons.SetNumberOfIterations(num_iter)

        if displacement_field is None:
            displacement_field = demons.Execute(fixed_level, moving_level)
        else:
            current_size = fixed_level.GetSize()
            prev_size = displacement_field.GetSize()
            scale = [c / p for c, p in zip(current_size, prev_size)]

            resampler = sitk.ResampleImageFilter()
            resampler.SetSize(current_size)
            resampler.SetOutputSpacing(fixed_level.GetSpacing())
            resampler.SetOutputOrigin(fixed_level.GetOrigin())
            resampler.SetOutputDirection(fixed_level.GetDirection())
            resampler.SetInterpolator(sitk.sitkLinear)
            displacement_field = resampler.Execute(displacement_field)

            displacement_field = sitk.Compose([
                sitk.VectorIndexSelectionCast(displacement_field, i) * scale[i]
                for i in range(3)
            ])

            displacement_field = demons.Execute(
                fixed_level, moving_level, displacement_field
            )

    field_array = sitk.GetArrayFromImage(displacement_field)
    field_array = field_array[..., ::-1]  # (dx,dy,dz) -> (dz,dy,dx)
    return field_array


def _run_antialias_pyramid(sitk, fixed, moving, demons, iterations):
    """Run demons with MATLAB-matching anti-aliased multi-resolution pyramid.

    Replaces SimpleITK's naive ``Shrink`` with Butterworth-filtered
    downsampling, matching MATLAB's ``imregdemons`` internal
    ``antialiasResize``. Uses float64 precision (matching MATLAB's
    ``double``).

    For each pyramid level (coarse to fine):
    1. Downsample fixed/moving with anti-aliased resize
    2. Upsample previous displacement field (if any) and scale magnitudes
    3. Run single-level SimpleITK demons with initial field
    """
    from starfinder.registration.pyramid import (
        antialias_resize,
        crop_padding,
        pad_for_pyramiding,
    )

    num_levels = len(iterations)

    # Pad volumes for clean 2x downsampling at every level
    fixed_padded, pad_widths = pad_for_pyramiding(fixed, num_levels)
    moving_padded, _ = pad_for_pyramiding(moving, num_levels)

    # Convert to float64 (matching MATLAB's double precision)
    fixed_padded = fixed_padded.astype(np.float64)
    moving_padded = moving_padded.astype(np.float64)

    # Displacement field in numpy convention (Z, Y, X, 3) with (dz, dy, dx)
    displacement_np = None

    for level_idx, num_iter in enumerate(iterations):
        # Downsample factor: coarsest level first
        # Level 0 (coarsest): factor = 0.5^(num_levels-1)
        # Level N-1 (finest): factor = 1.0
        power = num_levels - 1 - level_idx
        factor = 0.5 ** power if power > 0 else 1.0

        if factor < 1.0:
            fixed_level = antialias_resize(fixed_padded, factor)
            moving_level = antialias_resize(moving_padded, factor)
        else:
            fixed_level = fixed_padded
            moving_level = moving_padded

        # Convert to SimpleITK for this level
        fixed_sitk = sitk.GetImageFromArray(fixed_level.astype(np.float64))
        moving_sitk = sitk.GetImageFromArray(moving_level.astype(np.float64))

        demons.SetNumberOfIterations(num_iter)

        if displacement_np is None:
            # First level: run demons directly
            disp_sitk = demons.Execute(fixed_sitk, moving_sitk)
        else:
            # Upsample displacement field from previous level
            # Scale each component by 2 (since spatial dimensions doubled)
            upsampled = np.empty((*fixed_level.shape, 3), dtype=np.float64)
            for d in range(3):
                upsampled[..., d] = antialias_resize(
                    displacement_np[..., d], 2.0
                ) * 2.0

            # Crop/pad to match current level size exactly
            for d in range(3):
                comp = upsampled[..., d]
                target_shape = fixed_level.shape
                # Handle size mismatches from rounding
                slices = tuple(
                    slice(0, min(cs, ts))
                    for cs, ts in zip(comp.shape, target_shape)
                )
                tmp = np.zeros(target_shape, dtype=np.float64)
                src_slices = tuple(
                    slice(0, min(cs, ts))
                    for cs, ts in zip(comp.shape, target_shape)
                )
                tmp[slices] = comp[src_slices]
                upsampled[..., d] = tmp

            # Convert to SimpleITK (dz,dy,dx) -> (dx,dy,dz)
            field_sitk_order = upsampled[..., ::-1].astype(np.float64)
            disp_sitk = sitk.GetImageFromArray(field_sitk_order, isVector=True)
            disp_sitk.CopyInformation(fixed_sitk)

            disp_sitk = demons.Execute(fixed_sitk, moving_sitk, disp_sitk)

        # Convert result back to numpy (dz, dy, dx)
        displacement_np = sitk.GetArrayFromImage(disp_sitk)[..., ::-1].copy()

    # Crop padding from displacement field
    field_cropped = crop_padding(displacement_np[..., 0], pad_widths)
    result = np.empty((*field_cropped.shape, 3), dtype=displacement_np.dtype)
    for d in range(3):
        result[..., d] = crop_padding(displacement_np[..., d], pad_widths)

    return result


def demons_register(
    fixed: np.ndarray,
    moving: np.ndarray,
    iterations: list[int] | None = None,
    smoothing_sigma: float = 1.0,
    method: str = "demons",
    pyramid_mode: str = "antialias",
) -> np.ndarray:
    """Register moving image to fixed using demons algorithm.

    Performs non-rigid registration using SimpleITK's demons filters
    with anti-aliased multi-resolution pyramid for robustness.

    Parameters
    ----------
    fixed : np.ndarray
        Fixed (reference) volume with shape (Z, Y, X).
    moving : np.ndarray
        Moving volume to register, with shape (Z, Y, X).
    iterations : list[int] | None, optional
        Number of iterations at each pyramid level (coarse to fine).
        Default is [100, 50, 25] (3-level anti-aliased pyramid).
    smoothing_sigma : float, optional
        Standard deviation for Gaussian smoothing of the displacement field.
        Default is 1.0 (matches MATLAB's AccumulatedFieldSmoothing).
    method : str, optional
        Demons variant to use. Options:
        - "demons" (default): Classic Thirion demons. Best speed/memory,
          matches MATLAB's imregdemons algorithm.
        - "diffeomorphic": Topology-preserving. Better quality on large
          deformations but ~37% more memory.
        - "symmetric": Standard symmetric forces demons.
        - "fast_symmetric": Faster symmetric forces variant.
    pyramid_mode : str, optional
        Multi-resolution pyramid strategy:
        - "antialias" (default): MATLAB-matching Butterworth-filtered
          downsampling. Use with multi-level pyramids.
        - "sitk": SimpleITK's built-in shrink (naive subsampling).
          Degrades quality for multi-level pyramids on sparse images.

    Returns
    -------
    np.ndarray
        Displacement field with shape (Z, Y, X, 3) where the last dimension
        contains (dz, dy, dx) displacement vectors.
    """
    sitk = _import_sitk()

    if iterations is None:
        iterations = [100, 50, 25]

    demons = _create_demons_filter(sitk, method, smoothing_sigma)

    if pyramid_mode == "antialias" and len(iterations) > 1:
        return _run_antialias_pyramid(sitk, fixed, moving, demons, iterations)
    else:
        return _run_sitk_pyramid(sitk, fixed, moving, demons, iterations)


def matlab_compatible_config() -> dict:
    """Return config matching MATLAB imregdemons defaults.

    Uses classic Thirion demons with anti-aliased 3-level pyramid,
    matching MATLAB's ``imregdemons(moving, fixed, [100 50 25])``
    with ``AccumulatedFieldSmoothing=1.0``.

    Returns
    -------
    dict
        Keyword arguments for :func:`demons_register`.

    Examples
    --------
    >>> field = demons_register(fixed, moving, **matlab_compatible_config())
    """
    return {
        "iterations": [100, 50, 25],
        "smoothing_sigma": 1.0,
        "method": "demons",
        "pyramid_mode": "antialias",
    }


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
    smoothing_sigma: float = 1.0,
    method: str = "demons",
    pyramid_mode: str = "antialias",
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
        Default is [100, 50, 25] (3-level anti-aliased pyramid).
    smoothing_sigma : float, optional
        Standard deviation for displacement field smoothing.
        Default is 1.0 (matches MATLAB's AccumulatedFieldSmoothing).
    method : str, optional
        Demons variant: "demons" (default), "diffeomorphic", "symmetric", "fast_symmetric".
    pyramid_mode : str, optional
        Pyramid strategy: "antialias" (default) or "sitk".

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
        smoothing_sigma=smoothing_sigma, method=method,
        pyramid_mode=pyramid_mode,
    )

    # Apply the displacement field to each channel
    n_channels = images.shape[-1]
    registered = np.empty_like(images)

    for c in range(n_channels):
        registered[:, :, :, c] = apply_deformation(images[:, :, :, c], displacement_field)

    return registered, displacement_field
