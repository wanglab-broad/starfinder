"""DFT-based phase correlation registration using NumPy/SciPy."""

from __future__ import annotations

import numpy as np


def phase_correlate(
    fixed: np.ndarray,
    moving: np.ndarray,
) -> tuple[float, float, float]:
    """
    Compute shift to align moving image to fixed using phase correlation.

    Args:
        fixed: Reference volume with shape (Z, Y, X).
        moving: Volume to align with shape (Z, Y, X).

    Returns:
        Tuple of (dz, dy, dx) shift values.
    """
    from scipy.fft import fftn, ifftn

    # Cast to float32 for faster FFT (complex64 vs complex128)
    fixed = np.asarray(fixed, dtype=np.float32)
    moving = np.asarray(moving, dtype=np.float32)

    nz, ny, nx = moving.shape

    # Cross-correlation in frequency domain
    fixed_fft = fftn(fixed)
    moving_fft = fftn(moving)
    cc = ifftn(fixed_fft * np.conj(moving_fft))

    # Find peak
    peak_idx = np.argmax(np.abs(cc))
    iz, iy, ix = np.unravel_index(peak_idx, cc.shape)

    # Convert to signed shifts (handle wrap-around)
    # Negate to get shift of moving relative to fixed
    dz = float(-(iz if iz < nz // 2 else iz - nz))
    dy = float(-(iy if iy < ny // 2 else iy - ny))
    dx = float(-(ix if ix < nx // 2 else ix - nx))

    return (dz, dy, dx)


def apply_shift(
    volume: np.ndarray,
    shift: tuple[float, float, float],
) -> np.ndarray:
    """
    Apply shift to volume and zero out wrapped regions.

    Args:
        volume: Input volume with shape (Z, Y, X).
        shift: Tuple of (dz, dy, dx) shift values.

    Returns:
        Shifted volume with same shape.
    """
    from scipy.fft import fftn, ifftn
    from scipy.ndimage import fourier_shift

    # Cast to float32 for faster FFT (complex64 vs complex128)
    volume_f32 = np.asarray(volume, dtype=np.float32)

    # Apply shift in frequency domain
    shifted_fft = fourier_shift(fftn(volume_f32), shift)
    result = np.abs(ifftn(shifted_fft))

    # Zero out wrapped regions
    nz, ny, nx = volume.shape
    dz, dy, dx = shift

    if dz >= 0:
        result[: int(np.ceil(dz)), :, :] = 0
    else:
        result[int(nz + np.floor(dz) + 1) :, :, :] = 0

    if dy >= 0:
        result[:, : int(np.ceil(dy)), :] = 0
    else:
        result[:, int(ny + np.floor(dy) + 1) :, :] = 0

    if dx >= 0:
        result[:, :, : int(np.ceil(dx))] = 0
    else:
        result[:, :, int(nx + np.floor(dx) + 1) :] = 0

    return result.astype(volume.dtype)


def register_volume(
    images: np.ndarray,
    ref_image: np.ndarray,
    mov_image: np.ndarray,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """
    Register multi-channel volume using phase correlation.

    Args:
        images: Multi-channel volume with shape (Z, Y, X, C).
        ref_image: Reference image with shape (Z, Y, X) for shift calculation.
        mov_image: Moving image with shape (Z, Y, X) for shift calculation.

    Returns:
        Tuple of (registered_images, shifts).
    """
    # Calculate shift
    shifts = phase_correlate(ref_image, mov_image)

    # Apply shift to each channel
    n_channels = images.shape[-1]
    registered = np.zeros_like(images)

    for c in range(n_channels):
        registered[:, :, :, c] = apply_shift(images[:, :, :, c], shifts)

    return registered, shifts
