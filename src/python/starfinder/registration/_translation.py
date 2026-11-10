"""DFT-based phase correlation registration using NumPy/SciPy."""

from __future__ import annotations

import numpy as np

from ..image import _validate_image, _triple
from ._resampling import _cast_warp_output, _output_dtype


def phase_correlate(
    fixed: np.ndarray,
    moving: np.ndarray,
    workers: int | None = None,
) -> tuple[float, float, float]:
    """
    Detect the displacement of moving relative to fixed by FFT correlation.

    Args:
        fixed: Reference volume with shape (Z, Y, X).
        moving: Volume to align with shape (Z, Y, X).
        workers: Number of parallel workers for FFT. None for single-threaded,
            -1 for all available CPUs.

    Returns:
        Detected displacement ``(dz, dy, dx)`` in voxels (integer-valued floats).
        Correct alignment with ``apply_shift(moving, tuple(-s for s in shift))``.
        Inputs must be equal-shaped 3D numeric volumes; FFTs use float32.
        Singleton axes return zero. Odd-axis peaks at n//2 are nonwrapped;
        even half-period displacement is reported as +n/2 (ambiguous sign).
    """
    from scipy.fft import irfftn, rfftn

    fixed = _validate_image(fixed, ndim=(3,))
    moving = _validate_image(moving, ndim=(3,))
    if fixed.shape != moving.shape:
        raise ValueError("fixed and moving must have equal shapes")

    nz, ny, nx = moving.shape

    # Cross-correlation in frequency domain using real FFT (float32 → complex64;
    # rfftn keeps half the spectrum). Each float32 copy is released after its
    # transform, and the cross-power spectrum fixed_fft * conj(moving_fft) is
    # formed in place in fixed_fft (the same ufunc on the same operands in the
    # same order), so at most three volume-sized float32/complex64 arrays are
    # alive at once (W-742).
    fixed_fft = rfftn(np.asarray(fixed, dtype=np.float32), workers=workers)
    moving_fft = rfftn(np.asarray(moving, dtype=np.float32), workers=workers)
    np.conjugate(moving_fft, out=moving_fft)
    np.multiply(fixed_fft, moving_fft, out=fixed_fft)
    del moving_fft
    cc = irfftn(fixed_fft, s=(nz, ny, nx), workers=workers)
    del fixed_fft

    # Find peak (cc is real-valued from irfftn, abs handles numerical noise)
    peak_idx = np.argmax(np.abs(cc, out=cc))
    iz, iy, ix = np.unravel_index(peak_idx, cc.shape)

    # Convert to signed shifts (handle wrap-around)
    # Singleton zero and odd n//2 stay nonnegative; even n/2 unwraps
    # to -n/2, giving the declared positive half-period displacement.
    # Negate to get shift of moving relative to fixed
    dz = float(-(iz if iz < (nz + 1) // 2 else iz - nz))
    dy = float(-(iy if iy < (ny + 1) // 2 else iy - ny))
    dx = float(-(ix if ix < (nx + 1) // 2 else ix - nx))

    return (dz, dy, dx)


def apply_shift(
    volume: np.ndarray,
    shift: tuple[float, float, float],
    workers: int | None = None,
    *,
    output_dtype: str = "input",
) -> np.ndarray:
    """
    Apply shift to volume and zero out wrapped regions.

    Uses np.roll for integer-pixel shifts (no FFT needed), falling back
    to Fourier-domain shifting for sub-pixel shifts.

    Args:
        volume: Input volume with shape (Z, Y, X).
        shift: Translation to apply, ``(dz, dy, dx)`` in voxels. Positive
            components move content toward larger indices. Negate the detected
            displacement returned by :func:`starfinder.registration.phase_correlate`.
        workers: Number of parallel workers for FFT. None for single-threaded,
            -1 for all available CPUs.
        output_dtype: "input" (default), "float32" or "float64". Explicit float
            output avoids final integer quantization; it does not select the
            calculation precision.

    Returns:
        Shifted volume with the same shape; dtype is preserved by default.
        Integer outputs are rounded nearest-even, clipped to the dtype range,
        then cast once. Floating outputs retain signed Fourier values.
    """
    volume = _validate_image(volume, ndim=(3,))
    shift = _triple(shift, "shift")
    dtype = _output_dtype(volume.dtype, output_dtype)
    int_shift = _integer_shift(shift)
    if int_shift is None:
        return _fourier_shift(volume, shift, dtype, workers)
    output = np.empty(volume.shape, dtype=dtype)
    _integer_shift_into(volume, int_shift, output)
    return output


def _shift_into(volume, shift, output, workers=None):
    """apply_shift(volume, shift, workers, output_dtype) written into ``output``, a ZYX array (or view) of the
    output dtype: the same validation and values, without a volume-sized intermediate on the integer path."""
    volume = _validate_image(volume, ndim=(3,))
    shift = _triple(shift, "shift")
    int_shift = _integer_shift(shift)
    if int_shift is None:
        output[...] = _fourier_shift(volume, shift, output.dtype, workers)
    else:
        _integer_shift_into(volume, int_shift, output)


def _integer_shift(shift):
    """The integer shift that shift rounds to when every component is within 1e-6 of it; None otherwise."""
    int_shift = tuple(int(round(s)) for s in shift)
    return int_shift if all(abs(s - i) < 1e-6 for s, i in zip(shift, int_shift)) else None


def _fourier_shift(volume, shift, dtype, workers):
    """Sub-pixel path: FFT-based shift, wrapped regions zeroed, cast once to dtype."""
    from scipy.fft import fftn, ifftn
    from scipy.ndimage import fourier_shift

    calculation_dtype = np.float64 if volume.dtype == np.float64 else np.float32
    volume_float = np.asarray(volume, dtype=calculation_dtype)
    shifted_fft = fourier_shift(fftn(volume_float, workers=workers), shift)
    # Real projection retains signed intensities (and discards the Nyquist
    # imaginary component for fractional shifts of even-length real signals).
    result = ifftn(shifted_fft, workers=workers).real

    _zero_wrapped_edges(result, shift)
    return _cast_warp_output(result, dtype)


def _integer_shift_into(volume: np.ndarray, shift: tuple[int, int, int], output: np.ndarray) -> None:
    """Integer-pixel shift with zero fill, written into output (no FFT, no intermediate copy).

    The values of a circular shift (np.roll) with the wrapped regions zeroed: the overlapping block is copied
    (cast to output's dtype as astype casts) and the rest of output is zero.
    """
    if any(abs(s) >= n for s, n in zip(shift, volume.shape)):
        output.fill(0)
        return
    target = tuple(slice(s, n) if s >= 0 else slice(0, n + s) for s, n in zip(shift, volume.shape))
    source = tuple(slice(0, n - s) if s >= 0 else slice(-s, n) for s, n in zip(shift, volume.shape))
    output[target] = volume[source]
    _zero_wrapped_edges(output, shift)


def _zero_wrapped_edges(
    result: np.ndarray, shift: tuple[float, float, float]
) -> None:
    """Zero out wrapped boundary regions after a circular shift."""
    nz, ny, nx = result.shape
    if any(abs(s) >= n for s, n in zip(shift, result.shape)):
        result.fill(0)
        return
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



def estimate_translation(reference, moving, config, geometry):
    """Registered estimator of TranslationConfig: transform, backend and application policy."""
    from ._config import WarpConfig
    from ._types import TranslationTransform

    if config.backend == "scipy_fft":
        shift = phase_correlate(reference, moving, workers=config.fft_workers)
    else:
        from ._skimage_backend import phase_correlate_skimage

        shift = phase_correlate_skimage(reference, moving)
    transform = TranslationTransform(shift, **geometry)
    return transform, config.backend, WarpConfig(fft_workers=config.fft_workers)
