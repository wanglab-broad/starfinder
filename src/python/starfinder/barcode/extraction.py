"""Color vector extraction from spot locations.

Reference: src/matlab/ExtractFromLocation.m

Reads per-channel intensities in a voxel neighborhood around each detected
spot, L2-normalizes, and performs winner-take-all channel assignment.
"""

import numpy as np
import pandas as pd


def _sum_neighborhood_intensities(
    image: np.ndarray,
    spots: pd.DataFrame,
    voxel_size: tuple[int, int, int],
) -> np.ndarray:
    """Sum per-channel intensities around each spot."""
    if image.ndim != 4:
        raise ValueError(f"Expected 4D (Z, Y, X, C) image, got {image.ndim}D")

    n_points = len(spots)
    n_channels = image.shape[3]
    dz, dy, dx = voxel_size

    if n_points == 0:
        return np.empty((0, n_channels), dtype=np.float64)

    # Pad image with zeros so boundary spots use zero-padded neighborhoods.
    # Padding zeros contribute nothing to the sum, matching the original
    # clipping behavior without per-spot boundary checks.
    if dz > 0 or dy > 0 or dx > 0:
        padded = np.pad(
            image, ((dz, dz), (dy, dy), (dx, dx), (0, 0)), mode="constant"
        )
    else:
        padded = image

    # Spot coordinates index directly into padded image
    z_arr = spots["z"].values.astype(int)
    y_arr = spots["y"].values.astype(int)
    x_arr = spots["x"].values.astype(int)

    # Build neighborhood offset grid
    nz, ny, nx = 2 * dz + 1, 2 * dy + 1, 2 * dx + 1
    oz, oy, ox = np.mgrid[0:nz, 0:ny, 0:nx]
    oz_flat = oz.ravel()  # (nz*ny*nx,)
    oy_flat = oy.ravel()
    ox_flat = ox.ravel()

    # Compute all voxel indices for all spots: (N, neighborhood_size)
    z_idx = z_arr[:, None] + oz_flat[None, :]
    y_idx = y_arr[:, None] + oy_flat[None, :]
    x_idx = x_arr[:, None] + ox_flat[None, :]

    # Extract all neighborhoods at once: (N, neighborhood_size, C)
    neighborhoods = padded[z_idx, y_idx, x_idx, :]

    # Sum over spatial dims -> (N, C) per-spot color vectors
    return neighborhoods.sum(axis=1).astype(np.float64)


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

    if n_points == 0:
        return np.empty(0, dtype=object), np.empty(0, dtype=np.float64)

    color_vecs = _sum_neighborhood_intensities(image, spots, voxel_size)

    # L2 normalize per spot
    norms = np.sqrt((color_vecs**2).sum(axis=1, keepdims=True)) + 1e-6
    color_vecs /= norms

    # Winner-take-all channel assignment
    color_max = color_vecs.max(axis=1)  # (N,)
    argmax = color_vecs.argmax(axis=1)  # (N,)

    # Detect ties (multiple channels share the max) and NaN
    tie_mask = (color_vecs == color_max[:, None]).sum(axis=1) > 1
    nan_mask = np.isnan(color_max)
    normal_mask = ~nan_mask & ~tie_mask

    # Build color_seq
    channel_labels = np.array([str(i + 1) for i in range(n_channels)])
    color_seq = np.empty(n_points, dtype=object)
    color_seq[nan_mask] = "N"
    color_seq[tie_mask] = "M"
    color_seq[normal_mask] = channel_labels[argmax[normal_mask]]

    # Build color_score (inf for ties and NaN, -log(max) for normal)
    color_score = np.full(n_points, np.inf, dtype=np.float64)
    if normal_mask.any():
        color_score[normal_mask] = -np.log(color_max[normal_mask])

    return color_seq, color_score


def extract_intensity_tensor(
    images: dict[str, np.ndarray],
    spots: pd.DataFrame,
    round_order: list[str],
    voxel_size: tuple[int, int, int] = (1, 2, 2),
) -> np.ndarray:
    """Return raw per-(spot, channel, round) intensities.

    Uses the same zero-padded voxel-neighborhood sum as
    ``extract_from_location()``, but skips L2 normalization and
    winner-take-all assignment. Output shape is ``(N, C, R)`` for direct
    use by probabilistic decoders such as Postcode.
    """
    if not round_order:
        return np.empty((len(spots), 0, 0), dtype=np.float64)

    missing = [round_name for round_name in round_order if round_name not in images]
    if missing:
        raise KeyError(f"Missing images for rounds: {missing}")

    first = images[round_order[0]]
    if first.ndim != 4:
        raise ValueError(f"Expected 4D (Z, Y, X, C) image, got {first.ndim}D")

    n_points = len(spots)
    n_channels = first.shape[3]
    tensor = np.empty((n_points, n_channels, len(round_order)), dtype=np.float64)

    for round_idx, round_name in enumerate(round_order):
        image = images[round_name]
        if image.ndim != 4:
            raise ValueError(
                f"Expected 4D (Z, Y, X, C) image for {round_name}, "
                f"got {image.ndim}D"
            )
        if image.shape[3] != n_channels:
            raise ValueError(
                f"All rounds must have {n_channels} channels; "
                f"{round_name} has {image.shape[3]}"
            )
        tensor[:, :, round_idx] = _sum_neighborhood_intensities(
            image, spots, voxel_size
        )

    return tensor
