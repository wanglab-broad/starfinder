"""General-purpose utilities for STARfinder."""

import numpy as np


def make_projection(volume: np.ndarray, method: str = "max") -> np.ndarray:
    """Project a 3D/4D volume along the Z axis (axis 0).

    Parameters
    ----------
    volume : np.ndarray
        Input volume with shape (Z, Y, X) or (Z, Y, X, C).
    method : str
        Projection method: "max" (maximum intensity) or "sum" (sum with
        uint8 rescaling, matching MATLAB im2uint8 behavior).

    Returns
    -------
    np.ndarray
        Projected image with shape (Y, X) or (Y, X, C), dtype uint8.
    """
    if method == "max":
        return np.max(volume, axis=0)
    elif method == "sum":
        summed = np.sum(volume.astype(np.uint32), axis=0)
        max_val = summed.max()
        if max_val == 0:
            return np.zeros(summed.shape, dtype=np.uint8)
        return (summed / max_val * 255).astype(np.uint8)
    else:
        raise ValueError(f"Unknown projection method: {method!r}. Use 'max' or 'sum'.")
