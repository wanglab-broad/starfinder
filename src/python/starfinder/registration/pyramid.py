"""MATLAB-matching anti-aliased multi-resolution pyramid utilities.

Implements Butterworth-filtered downsampling matching MATLAB's imregdemons
internal ``antialiasResize`` / ``butterwth`` functions. This is critical for
sparse fluorescence images: naive subsampling (SimpleITK ``Shrink``) drops
bright spots at coarse pyramid levels, while anti-aliased downsampling
preserves them via low-pass filtering before decimation.

Reference: MATLAB ``MultiResolutionDemons3D.m`` (R2023b).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import zoom


def butterworth_3d(
    shape: tuple[int, ...],
    cutoff: float,
    order: int = 2,
) -> np.ndarray:
    """3D Butterworth low-pass filter in frequency domain.

    Matches MATLAB's ``butterwth(shape, cutoff, order)`` used inside
    ``antialiasResize``. The filter is centered at DC (zero-frequency)
    using ``fftfreq`` convention.

    Parameters
    ----------
    shape : tuple[int, ...]
        Spatial dimensions (Z, Y, X) of the volume.
    cutoff : float
        Normalized cutoff frequency in [0, 1]. For downsampling by factor f,
        use ``cutoff = 0.5 * f`` (Nyquist of the target resolution).
    order : int, optional
        Butterworth filter order. Higher = sharper rolloff. Default 2
        matches MATLAB.

    Returns
    -------
    np.ndarray
        Filter in frequency domain with shape ``shape``, values in [0, 1].
    """
    if cutoff <= 0:
        return np.zeros(shape, dtype=np.float64)

    # Separable Butterworth: product of per-dimension 1D filters.
    # This matches MATLAB's butterwth which filters each axis independently,
    # avoiding the over-attenuation of an isotropic L2-distance filter.
    result = np.ones(shape, dtype=np.float64)
    for dim, s in enumerate(shape):
        freq = np.fft.fftfreq(s)
        h_1d = 1.0 / (1.0 + (np.abs(freq) / cutoff) ** (2 * order))
        # Broadcast to full shape
        slicing = [np.newaxis] * len(shape)
        slicing[dim] = slice(None)
        result *= h_1d[tuple(slicing)]
    return result


def antialias_resize(volume: np.ndarray, factor: float) -> np.ndarray:
    """Anti-aliased 3D resize matching MATLAB's ``antialiasResize``.

    For downsampling (factor < 1): applies Butterworth low-pass filter
    before linear interpolation to prevent aliasing.
    For upsampling (factor >= 1): linear interpolation only.

    Parameters
    ----------
    volume : np.ndarray
        3D volume (Z, Y, X). Can be float or integer type.
    factor : float
        Resize factor. 0.5 = halve each dimension, 2.0 = double.

    Returns
    -------
    np.ndarray
        Resized volume (float64).
    """
    vol = volume.astype(np.float64)

    if factor < 1.0:
        # Low-pass filter before downsampling to prevent aliasing
        cutoff = 0.5 * factor  # Nyquist of target resolution
        filt = butterworth_3d(vol.shape, cutoff, order=2)
        vol = np.real(np.fft.ifftn(np.fft.fftn(vol) * filt))

    # Resize with linear interpolation
    new_shape = tuple(max(1, int(round(s * factor))) for s in vol.shape)
    zoom_factors = tuple(n / o for n, o in zip(new_shape, vol.shape))
    return zoom(vol, zoom_factors, order=1)


def pad_for_pyramiding(
    volume: np.ndarray,
    pyramid_levels: int,
) -> tuple[np.ndarray, list[int]]:
    """Pad volume so dimensions are divisible by 2^(levels-1).

    Uses replicate-border (edge) padding, matching MATLAB's
    ``padForPyramiding``.

    Parameters
    ----------
    volume : np.ndarray
        3D volume (Z, Y, X).
    pyramid_levels : int
        Number of pyramid levels. Dimensions are padded to be divisible
        by ``2^(pyramid_levels - 1)``.

    Returns
    -------
    tuple[np.ndarray, list[int]]
        ``(padded_volume, pad_widths)`` where ``pad_widths[i]`` is the
        number of voxels added to the end of dimension ``i``.
    """
    divisor = 2 ** (pyramid_levels - 1)
    pad_widths = []
    for s in volume.shape:
        remainder = s % divisor
        pad_widths.append((divisor - remainder) % divisor)

    if all(p == 0 for p in pad_widths):
        return volume, pad_widths

    # Edge (replicate-border) padding — pad only at the end of each axis
    pad_spec = [(0, p) for p in pad_widths]
    padded = np.pad(volume, pad_spec, mode="edge")
    return padded, pad_widths


def crop_padding(volume: np.ndarray, pad_widths: list[int]) -> np.ndarray:
    """Remove padding added by :func:`pad_for_pyramiding`.

    Parameters
    ----------
    volume : np.ndarray
        Padded 3D volume.
    pad_widths : list[int]
        Padding widths returned by :func:`pad_for_pyramiding`.

    Returns
    -------
    np.ndarray
        Cropped volume with original dimensions.
    """
    slices = tuple(
        slice(0, s - p) for s, p in zip(volume.shape, pad_widths)
    )
    return volume[slices]
