"""Color vector extraction from spot locations.

Reference: src/matlab/ExtractFromLocation.m

Reads per-channel intensities in a voxel neighborhood around each detected
spot, L2-normalizes, and performs winner-take-all channel assignment.
"""

import numpy as np
import pandas as pd


def extract_from_location(
    image: np.ndarray,
    spots: pd.DataFrame,
    voxel_size: tuple[int, int, int] = (1, 2, 2),
) -> tuple[np.ndarray, np.ndarray]:
    """Extract color vectors from spot neighborhoods.

    For each spot, sums intensities in a voxel neighborhood per channel,
    L2-normalizes, and assigns a winner-take-all channel label.

    Parameters
    ----------
    image : np.ndarray
        4D array with shape (Z, Y, X, C).
    spots : pd.DataFrame
        Must have columns [z, y, x] with 0-based coordinates.
    voxel_size : tuple[int, int, int]
        Half-widths (dz, dy, dx) for the extraction neighborhood.
        Default (1, 2, 2) matches MATLAB's [dx, dy, dz] = [2, 2, 1].

    Returns
    -------
    color_seq : np.ndarray
        1D string array of length N. Values: "1"-"4" (1-based channel),
        "M" (tie), or "N" (no signal).
    color_score : np.ndarray
        1D float array of length N. Score = -log(max_normalized_value).
        inf for "M" or "N" assignments.
    """
    if image.ndim != 4:
        raise ValueError(f"Expected 4D (Z, Y, X, C) image, got {image.ndim}D")

    n_points = len(spots)
    n_channels = image.shape[3]
    dz, dy, dx = voxel_size
    dim_z, dim_y, dim_x = image.shape[:3]

    color_seq = np.empty(n_points, dtype=object)
    color_score = np.zeros(n_points, dtype=np.float64)

    for i in range(n_points):
        z = int(spots.iloc[i]["z"])
        y = int(spots.iloc[i]["y"])
        x = int(spots.iloc[i]["x"])

        # Compute extents with boundary clipping
        z0 = max(0, z - dz)
        z1 = min(dim_z, z + dz + 1)
        y0 = max(0, y - dy)
        y1 = min(dim_y, y + dy + 1)
        x0 = max(0, x - dx)
        x1 = min(dim_x, x + dx + 1)

        # Extract voxel neighborhood and sum across spatial dims
        voxel = image[z0:z1, y0:y1, x0:x1, :]
        color_vec = voxel.sum(axis=(0, 1, 2)).astype(np.float64)

        # L2 normalize
        norm = np.sqrt(np.sum(color_vec**2)) + 1e-6
        color_vec = color_vec / norm

        color_max = np.max(color_vec)

        if np.isnan(color_max):
            color_seq[i] = "N"
            color_score[i] = np.inf
        else:
            max_indices = np.where(color_vec == color_max)[0]
            if len(max_indices) != 1:
                # Tie between multiple channels
                color_seq[i] = "M"
                color_score[i] = np.inf
            else:
                # 1-based channel string for codebook compatibility
                color_seq[i] = str(max_indices[0] + 1)
                color_score[i] = -np.log(color_max)

    return color_seq, color_score
