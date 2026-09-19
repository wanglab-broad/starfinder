"""Private registration numerical implementation."""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

@dataclass(frozen=True)
class _FieldProcessingResult:
    field: np.ndarray
    fold_mask: np.ndarray | None

def sanitize_displacement_field(
    field: np.ndarray,
    *,
    clamp: bool = True,
    margin: int = 0,
    smooth_sigma: float | None = None,
    detect_folds: bool = False,
) -> _FieldProcessingResult:
    """Post-process a displacement field to reduce interpolation artifacts.

    Parameters
    ----------
    field : np.ndarray
        Displacement field with shape (Z, Y, X, 3), (dz, dy, dx).
    clamp : bool
        If True, clip ``position + displacement`` to ``[margin, dim-1-margin]``
        so that source coordinates never go out of bounds. Eliminates black
        bands from ``map_coordinates(cval=0)`` or SimpleITK zero-fill.
    margin : int
        Safety margin for clamping (pixels from each edge). Default 0.
    smooth_sigma : float or None
        If set, apply Gaussian smoothing to each displacement component.
        Removes cubic-zoom ringing and softens sharp gradients that cause
        field folding.
    detect_folds : bool
        If True, also return a boolean mask where det(J) < 0 (non-injective
        mapping). Diagnostic only — the caller decides what to do.

    Returns
    -------
    _FieldProcessingResult
        Sanitized field and optional fold_mask (ZYX), None when not measured.
    """
    from scipy.ndimage import gaussian_filter

    field = field.copy()
    shape = field.shape[:3]  # (Z, Y, X)

    # Gaussian smoothing (before clamping so clamp sees smoothed values)
    if smooth_sigma is not None and smooth_sigma > 0:
        for d in range(3):
            field[..., d] = gaussian_filter(field[..., d], sigma=smooth_sigma)

    # Coordinate clamping
    if clamp:
        for d in range(3):
            # Build coordinate grid for this axis
            ax_len = shape[d]
            lo = float(margin)
            hi = float(ax_len - 1 - margin)

            # Create position array matching field shape
            slices = [None, None, None]
            slices[d] = slice(None)
            pos = np.arange(ax_len, dtype=np.float32)
            # Reshape for broadcasting: e.g. for d=1 → (1, Y, 1)
            bcast_shape = [1, 1, 1]
            bcast_shape[d] = ax_len
            pos = pos.reshape(bcast_shape)

            src = pos + field[..., d]
            field[..., d] = np.clip(src, lo, hi) - pos

    # Fold detection via Jacobian determinant
    if detect_folds:
        fold_mask = _detect_folds(field)
        return _FieldProcessingResult(field, fold_mask)

    return _FieldProcessingResult(field, None)


def _detect_folds(field: np.ndarray) -> np.ndarray:
    """Compute fold mask where det(Jacobian of deformation) < 0.

    The deformation is phi(x) = x + d(x). The Jacobian of phi is
    J = I + grad(d), so det(J) < 0 means the mapping folds (non-injective).
    Uses finite differences on the displacement field.

    Parameters
    ----------
    field : np.ndarray
        Displacement field, shape (Z, Y, X, 3).

    Returns
    -------
    np.ndarray
        Boolean mask, shape (Z, Y, X), True where det(J) < 0.
    """
    # Gradient of each displacement component w.r.t. each spatial axis
    # grad[i][j] = d(field[..., i]) / d(axis_j)
    grad = np.empty((3, 3) + field.shape[:3], dtype=np.float32)
    for i in range(3):
        for j in range(3):
            grad[i, j] = np.gradient(field[..., i], axis=j)

    # Jacobian of deformation = I + grad(displacement)
    for i in range(3):
        grad[i, i] += 1.0

    # det(J) via explicit 3x3 formula
    det = (
        grad[0, 0] * (grad[1, 1] * grad[2, 2] - grad[1, 2] * grad[2, 1])
        - grad[0, 1] * (grad[1, 0] * grad[2, 2] - grad[1, 2] * grad[2, 0])
        + grad[0, 2] * (grad[1, 0] * grad[2, 1] - grad[1, 1] * grad[2, 0])
    )

    return det < 0
