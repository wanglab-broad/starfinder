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
        Detected displacement ``(dz, dy, dx)`` in voxels, matching
        :func:`starfinder.registration.phase_correlate`. Negate this tuple
        before applying it to ``moving``. Inputs must be equal-shaped 3D arrays.
    """
    shift, error, diffphase = phase_cross_correlation(fixed, moving)
    # Negate scikit-image correction to return detected displacement.
    return (float(-shift[0]), float(-shift[1]), float(-shift[2]))
