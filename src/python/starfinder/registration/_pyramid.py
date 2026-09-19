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
    rfft: bool = False,
    dtype: np.dtype = np.float64,
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
    rfft : bool, optional
        If True, produce half-spectrum filter for use with ``rfftn``.
        Last axis uses ``rfftfreq`` (length ``X//2+1``). Default False.
    dtype : np.dtype, optional
        Output dtype. Default ``np.float64``.

    Returns
    -------
    np.ndarray
        Filter in frequency domain, values in [0, 1].
        Shape is ``shape`` when rfft=False, or
        ``(*shape[:-1], shape[-1]//2+1)`` when rfft=True.
    """
    # Determine output shape
    if rfft:
        out_shape = (*shape[:-1], shape[-1] // 2 + 1)
    else:
        out_shape = shape

    if cutoff <= 0:
        return np.zeros(out_shape, dtype=dtype)

    # Separable Butterworth: product of per-dimension 1D filters.
    # This matches MATLAB's butterwth which filters each axis independently,
    # avoiding the over-attenuation of an isotropic L2-distance filter.
    result = np.ones(out_shape, dtype=dtype)
    for dim, s in enumerate(shape):
        if rfft and dim == len(shape) - 1:
            freq = np.fft.rfftfreq(s)
        else:
            freq = np.fft.fftfreq(s)
        h_1d = 1.0 / (1.0 + (np.abs(freq) / cutoff) ** (2 * order))
        # Broadcast to full output shape
        slicing = [np.newaxis] * len(out_shape)
        slicing[dim] = slice(None)
        result *= h_1d.astype(dtype)[tuple(slicing)]
    return result


def antialias_resize(
    volume: np.ndarray,
    factor: float,
    filt: np.ndarray | None = None,
) -> np.ndarray:
    """Anti-aliased 3D resize matching MATLAB's ``antialiasResize``.

    For downsampling (factor < 1): applies Butterworth low-pass filter
    before linear interpolation to prevent aliasing. Uses ``rfftn``/``irfftn``
    for memory efficiency (~50% less than ``fftn``/``ifftn`` for real input).

    For upsampling (factor >= 1): linear interpolation only.

    Parameters
    ----------
    volume : np.ndarray
        3D volume (Z, Y, X). Can be float or integer type.
    factor : float
        Resize factor. 0.5 = halve each dimension, 2.0 = double.
    filt : np.ndarray or None, optional
        Pre-computed Butterworth rfft filter. If None, computed
        automatically. Pass pre-computed filter to avoid recomputing
        for fixed/moving pairs at the same pyramid level.

    Returns
    -------
    np.ndarray
        Resized volume. Float32 for integer input, preserves floating dtype.
    """
    # Use float32 for integer input (sufficient for image processing);
    # preserve floating-point dtype (important for displacement fields).
    if np.issubdtype(volume.dtype, np.floating):
        vol = np.asarray(volume)
    else:
        vol = volume.astype(np.float32)

    if factor < 1.0:
        # Low-pass filter before downsampling to prevent aliasing.
        # rfftn/irfftn exploits conjugate symmetry of real input:
        # last axis halved from X to X//2+1, ~50% less memory.
        if filt is None:
            cutoff = 0.5 * factor  # Nyquist of target resolution
            filt = butterworth_3d(
                vol.shape, cutoff, order=2, rfft=True, dtype=vol.dtype,
            )
        axes = tuple(range(vol.ndim))
        vol = np.fft.irfftn(np.fft.rfftn(vol, axes=axes) * filt, s=vol.shape, axes=axes)

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
