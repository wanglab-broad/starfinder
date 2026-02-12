"""Intensity normalization and histogram matching for STARfinder.

Ports MATLAB MinMaxNorm.m and STARMapDataset.HistEqualize.
"""

import numpy as np
from skimage.exposure import match_histograms


def min_max_normalize(volume: np.ndarray) -> np.ndarray:
    """Per-channel min-max normalization to uint8.

    Matches MATLAB ``stretchlim(ch, 0)`` + ``imadjustn(ch, [min, max])``:
    compute global min/max per channel across all Z slices, then linearly
    rescale to [0, 255].

    Parameters
    ----------
    volume : np.ndarray
        Input volume with shape (Z, Y, X) or (Z, Y, X, C).

    Returns
    -------
    np.ndarray
        Normalized volume, dtype uint8, same shape as input.
    """
    is_3d = volume.ndim == 3
    if is_3d:
        volume = volume[..., np.newaxis]

    result = np.empty_like(volume, dtype=np.uint8)
    for c in range(volume.shape[3]):
        ch = volume[:, :, :, c].astype(np.float64)
        lo = ch.min()
        hi = ch.max()
        if lo == hi:
            result[:, :, :, c] = 0
        else:
            result[:, :, :, c] = ((ch - lo) / (hi - lo) * 255).astype(np.uint8)

    if is_3d:
        result = result[..., 0]
    return result


def histogram_match(
    volume: np.ndarray,
    reference: np.ndarray,
    nbins: int = 64,
) -> np.ndarray:
    """Match histogram of each channel to a reference volume.

    Ports MATLAB ``imhistmatchn``. Uses scikit-image exact CDF matching
    (``nbins`` accepted for API compatibility but unused — the difference
    is negligible for uint8 data).

    Parameters
    ----------
    volume : np.ndarray
        Input volume with shape (Z, Y, X) or (Z, Y, X, C).
    reference : np.ndarray
        Reference volume with shape (Z, Y, X). Each channel of *volume*
        is matched to this single reference.
    nbins : int
        Accepted for API compatibility; not used by skimage.

    Returns
    -------
    np.ndarray
        Histogram-matched volume, same shape and dtype as input.
    """
    is_3d = volume.ndim == 3
    if is_3d:
        volume = volume[..., np.newaxis]

    result = np.empty_like(volume)
    for c in range(volume.shape[3]):
        result[:, :, :, c] = match_histograms(volume[:, :, :, c], reference)

    if is_3d:
        result = result[..., 0]
    return result
