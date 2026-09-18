"""Private registration numerical implementation."""
from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree
from ._errors import InsufficientLandmarksError
def detect_and_match_spots(
    fixed: np.ndarray,
    moving: np.ndarray,
    detection_threshold: float = 3.0,
    match_distance: float = 10.0,
    min_matches: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect spots in both volumes and match by nearest-neighbor.

    Parameters
    ----------
    fixed : np.ndarray
        Fixed (reference) volume, shape (Z, Y, X).
    moving : np.ndarray
        Moving volume, shape (Z, Y, X).
    detection_threshold : float
        ``k`` for noise-floor detection: ``median + k * MAD * 1.4826``.
        Lower values detect more spots (more control points).
    match_distance : float
        Maximum Euclidean distance (pixels) for a valid match.
    min_matches : int
        Minimum number of matched pairs required.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(positions, displacements)`` where:

        - positions: (N, 3) matched spot positions in **fixed** image coords

        - displacements: (N, 3) displacement vectors (dz, dy, dx) at each position

    Raises
    ------
    ValueError
        If fewer than ``min_matches`` pairs are found.
    """
    from starfinder.spot_finding import find_spots, NoiseLandmarkConfig
    from starfinder.image import ImageMetadata

    fixed_spots = find_spots(
        fixed, config=NoiseLandmarkConfig(noise_sigma=detection_threshold),
        metadata=ImageMetadata("registration/fixed"), spot_namespace="registration/fixed",
    ).spots[["z", "y", "x"]].to_numpy()
    moving_spots = find_spots(
        moving, config=NoiseLandmarkConfig(noise_sigma=detection_threshold),
        metadata=ImageMetadata("registration/moving"), spot_namespace="registration/moving",
    ).spots[["z", "y", "x"]].to_numpy()

    if len(fixed_spots) == 0 or len(moving_spots) == 0:
        raise InsufficientLandmarksError(
            f"Spot detection found {len(fixed_spots)} fixed, "
            f"{len(moving_spots)} moving spots — need at least {min_matches}."
        )

    # Greedy nearest-neighbor matching via KDTree
    tree = cKDTree(moving_spots)
    distances, indices = tree.query(fixed_spots, k=1)

    # Filter by match distance
    valid = distances <= match_distance
    matched_fixed = fixed_spots[valid]
    matched_moving = moving_spots[indices[valid]]

    # Remove duplicate moving matches (keep closest)
    used_moving: set[int] = set()
    keep = []
    # Sort by distance so we keep closest matches
    order = np.argsort(distances[valid])
    matched_indices = indices[valid]
    for i in order:
        mov_idx = matched_indices[i]
        if mov_idx not in used_moving:
            used_moving.add(mov_idx)
            keep.append(i)
    keep = np.array(keep) if keep else np.array([], dtype=int)

    if len(keep) < min_matches:
        raise InsufficientLandmarksError(
            f"Only {len(keep)} spot matches found (need {min_matches}). "
            f"Detected {len(fixed_spots)} fixed, {len(moving_spots)} moving spots. "
            f"Try lowering detection_threshold or increasing match_distance."
        )

    positions = matched_fixed[keep]
    # Backward-mapping convention: displacement points from fixed-space to
    # moving-space so map_coordinates(moving, pos + disp) recovers fixed.
    displacements = matched_moving[keep] - matched_fixed[keep]

    return positions, displacements


def subsample_control_points(
    positions: np.ndarray,
    displacements: np.ndarray,
    max_points: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """Subsample to max_points using farthest-point sampling.

    Greedy farthest-point sampling ensures uniform spatial coverage,
    unlike random sampling which can leave gaps.

    Parameters
    ----------
    positions : np.ndarray
        (N, 3) point positions.
    displacements : np.ndarray
        (N, 3) displacement vectors at each position.
    max_points : int
        Maximum number of control points to keep.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Subsampled ``(positions, displacements)``.
    """
    n = len(positions)
    if n <= max_points:
        return positions, displacements

    # Farthest-point sampling: O(N * max_points)
    selected = np.zeros(max_points, dtype=np.intp)
    # Start from the point closest to the centroid (most central)
    centroid = positions.mean(axis=0)
    selected[0] = np.argmin(np.sum((positions - centroid) ** 2, axis=1))

    # Track minimum distance from each point to the selected set
    min_dist = np.full(n, np.inf)
    for i in range(1, max_points):
        # Update min distances with the last selected point
        d = np.sum((positions - positions[selected[i - 1]]) ** 2, axis=1)
        min_dist = np.minimum(min_dist, d)
        # Select the farthest point from the current set
        selected[i] = np.argmax(min_dist)

    return positions[selected], displacements[selected]
