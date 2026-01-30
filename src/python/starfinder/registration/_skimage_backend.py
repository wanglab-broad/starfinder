"""scikit-image based phase correlation (for comparison/benchmarking)."""

from __future__ import annotations

import numpy as np
from skimage.registration import phase_cross_correlation


def phase_correlate_skimage(
    fixed: np.ndarray,
    moving: np.ndarray,
) -> tuple[float, float, float]:
    """
    Compute shift using scikit-image phase_cross_correlation.

    Args:
        fixed: Reference volume with shape (Z, Y, X).
        moving: Volume to align with shape (Z, Y, X).

    Returns:
        Tuple of (dz, dy, dx) shift values.
    """
    shift, error, diffphase = phase_cross_correlation(fixed, moving)
    # Negate to match our convention: shift to apply to moving to align with fixed
    return (float(-shift[0]), float(-shift[1]), float(-shift[2]))
