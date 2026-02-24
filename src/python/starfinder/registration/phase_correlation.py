"""DFT-based phase correlation registration using NumPy/SciPy."""

from __future__ import annotations

import numpy as np


def phase_correlate(
    fixed: np.ndarray,
    moving: np.ndarray,
    workers: int | None = None,
) -> tuple[float, float, float]:
    """
    Compute shift to align moving image to fixed using phase correlation.

    Args:
        fixed: Reference volume with shape (Z, Y, X).
        moving: Volume to align with shape (Z, Y, X).
        workers: Number of parallel workers for FFT. None for single-threaded,
            -1 for all available CPUs.

    Returns:
        Tuple of (dz, dy, dx) shift values.
    """
    from scipy.fft import irfftn, rfftn

    # Cast to float32 for faster FFT (complex64 vs complex128)
    fixed = np.asarray(fixed, dtype=np.float32)
    moving = np.asarray(moving, dtype=np.float32)

    nz, ny, nx = moving.shape

    # Cross-correlation in frequency domain using real FFT
    # rfftn exploits conjugate symmetry of real input → ~50% less memory
    fixed_fft = rfftn(fixed, workers=workers)
    moving_fft = rfftn(moving, workers=workers)
    cc = irfftn(
        fixed_fft * np.conj(moving_fft), s=(nz, ny, nx), workers=workers
    )

    # Find peak (cc is real-valued from irfftn, abs handles numerical noise)
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
    workers: int | None = None,
) -> np.ndarray:
    """
    Apply shift to volume and zero out wrapped regions.

    Uses np.roll for integer-pixel shifts (no FFT needed), falling back
    to Fourier-domain shifting for sub-pixel shifts.

    Args:
        volume: Input volume with shape (Z, Y, X).
        shift: Tuple of (dz, dy, dx) shift values.
        workers: Number of parallel workers for FFT. None for single-threaded,
            -1 for all available CPUs.

    Returns:
        Shifted volume with same shape and dtype.
    """
    # Fast path: integer shifts via np.roll (no FFT overhead)
    int_shift = tuple(int(round(s)) for s in shift)
    if all(abs(s - i) < 1e-6 for s, i in zip(shift, int_shift)):
        return _apply_integer_shift(volume, int_shift)

    # Sub-pixel path: FFT-based shift
    from scipy.fft import fftn, ifftn
    from scipy.ndimage import fourier_shift

    volume_f32 = np.asarray(volume, dtype=np.float32)
    shifted_fft = fourier_shift(fftn(volume_f32, workers=workers), shift)
    result = np.abs(ifftn(shifted_fft, workers=workers))

    _zero_wrapped_edges(result, shift)
    return result.astype(volume.dtype)


def _apply_integer_shift(volume: np.ndarray, shift: tuple[int, int, int]) -> np.ndarray:
    """Apply integer-pixel shift using np.roll + zero-fill. No FFT overhead."""
    dz, dy, dx = shift
    if dz == 0 and dy == 0 and dx == 0:
        return volume.copy()

    result = np.roll(volume, shift=(dz, dy, dx), axis=(0, 1, 2))
    _zero_wrapped_edges(result, shift)
    return result


def _zero_wrapped_edges(
    result: np.ndarray, shift: tuple[float, float, float]
) -> None:
    """Zero out wrapped boundary regions after a circular shift."""
    nz, ny, nx = result.shape
    dz, dy, dx = shift

    if dz > 0:
        result[: int(np.ceil(dz)), :, :] = 0
    elif dz < 0:
        result[nz + int(np.floor(dz)) :, :, :] = 0

    if dy > 0:
        result[:, : int(np.ceil(dy)), :] = 0
    elif dy < 0:
        result[:, ny + int(np.floor(dy)) :, :] = 0

    if dx > 0:
        result[:, :, : int(np.ceil(dx))] = 0
    elif dx < 0:
        result[:, :, nx + int(np.floor(dx)) :] = 0


def register_volume(
    images: np.ndarray,
    ref_image: np.ndarray,
    mov_image: np.ndarray,
    workers: int | None = None,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """
    Register multi-channel volume using phase correlation.

    Args:
        images: Multi-channel volume with shape (Z, Y, X, C).
        ref_image: Reference image with shape (Z, Y, X) for shift calculation.
        mov_image: Moving image with shape (Z, Y, X) for shift calculation.
        workers: Number of parallel workers for FFT. None for single-threaded,
            -1 for all available CPUs.

    Returns:
        Tuple of (registered_images, shifts).
    """
    # Calculate shift (how much mov_image is shifted from ref_image)
    shifts = phase_correlate(ref_image, mov_image, workers=workers)

    # Apply NEGATIVE shift to correct the alignment
    correction = tuple(-s for s in shifts)

    # Apply correction to each channel
    n_channels = images.shape[-1]
    registered = np.zeros_like(images)

    for c in range(n_channels):
        registered[:, :, :, c] = apply_shift(images[:, :, :, c], correction, workers=workers)

    return registered, shifts
