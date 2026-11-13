"""Private registration numerical implementation."""
from __future__ import annotations
import numpy as np
from ._errors import InsufficientLandmarksError, RegistrationEstimationError
# The keys of a channel's noise record that a registration's landmarks record keeps (W-836).
_NOISE_KEYS = ("threshold", "median", "mad", "mad_floored", "estimation", "n_estimation")


def detect_landmarks(
    image: np.ndarray, detection_threshold: float, which: str,
) -> tuple[np.ndarray, dict, tuple[str, ...]]:
    """Noise landmarks of one registration image, their record and their notes.

    Parameters
    ----------
    image : np.ndarray
        Registration image, shape (Z, Y, X) or (Z, Y, X, C).
    detection_threshold : float
        ``noise_sigma`` of the NoiseLandmarkConfig.
    which : str
        ``reference`` or ``moving``, the image named in the notes and errors.

    Returns
    -------
    tuple[np.ndarray, dict, tuple[str, ...]]
        ``(points, record, notes)``: the (N, 3) ZYX landmarks after the
        cross-channel deduplication; the record ``noise`` (per channel label:
        threshold, median, mad, mad_floored, estimation, n_estimation) and
        ``candidates`` (N); the spot-finding notes (above-minimum estimation,
        MAD floor), each prefixed with the image.

    Raises
    ------
    RegistrationEstimationError
        If a floating-point channel with signal has a noise MAD of 0
        (SpotFindingError), naming the image and the channel.

    Notes
    -----
    The threshold is local maxima's noise threshold (W-836): the median and
    MAD over the voxels above the minimum when more than half sit at it,
    and the one-grey-level MAD floor of an integer image. Each note is also
    emitted as a SpotFindingWarning.
    """
    import warnings
    from starfinder.spot_finding import NoiseLandmarkConfig, SpotFindingError, SpotFindingWarning, find_spots
    from starfinder.image import ImageMetadata

    name = "fixed" if which == "reference" else which
    with warnings.catch_warnings():
        # Re-emitted below with the image named.
        warnings.simplefilter("ignore", SpotFindingWarning)
        try:
            result = find_spots(
                image, config=NoiseLandmarkConfig(noise_sigma=detection_threshold),
                metadata=ImageMetadata(f"registration/{name}"), spot_namespace=f"registration/{name}",
            )
        except SpotFindingError as exc:
            raise RegistrationEstimationError(f"{which} image landmarks: {exc}") from exc
    points = result.spots[["z", "y", "x"]].to_numpy()
    record = dict(noise={label: {key: noise[key] for key in _NOISE_KEYS}
                         for label, noise in result.diagnostics["noise"].items()},
                  candidates=len(points))
    notes = tuple(f"{which} image landmarks: {note}" for note in result.diagnostics["warnings"])
    for note in notes:
        warnings.warn(note, SpotFindingWarning, stacklevel=2)
    return points, record, notes


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
        ``k`` for noise-floor detection: ``median + k * MAD * 1.4826``, with
        local maxima's noise rule (see detect_landmarks).
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
    return _match_landmarks(fixed, moving, detection_threshold, match_distance, min_matches)[:2]


def _match_landmarks(fixed, moving, detection_threshold, match_distance, min_matches):
    """detect_and_match_spots, also returning the landmarks record (reference, moving, matched) and the notes."""
    fixed_spots, fixed_record, fixed_notes = detect_landmarks(fixed, detection_threshold, "reference")
    moving_spots, moving_record, moving_notes = detect_landmarks(moving, detection_threshold, "moving")

    if len(fixed_spots) == 0 or len(moving_spots) == 0:
        raise InsufficientLandmarksError(
            f"Spot detection found {len(fixed_spots)} fixed, "
            f"{len(moving_spots)} moving spots — need at least {min_matches}."
        )

    from starfinder.evaluation.matching import match_points
    from starfinder.image import ImageMetadata
    # Both index arrays are searched in the same voxel-coordinate domain;
    # these candidate correspondences do not assert aligned biological frames.
    metadata = ImageMetadata("registration/candidate-search")
    matches = match_points(fixed_spots, moving_spots, policy="nearest_candidate",
        threshold=match_distance, units="voxel", reference_metadata=metadata,
        observed_metadata=metadata)
    pairs = matches.details["matched_pairs"]
    if len(pairs) < min_matches:
        raise InsufficientLandmarksError(
            f"Only {len(pairs)} spot matches found (need {min_matches}). "
            f"Detected {len(fixed_spots)} fixed, {len(moving_spots)} moving spots. "
            f"Try lowering detection_threshold or increasing match_distance."
        )

    positions = fixed_spots[[i for i, _, _ in pairs]]
    # Backward-mapping convention: displacement points from fixed-space to
    # moving-space so map_coordinates(moving, pos + disp) recovers fixed.
    displacements = moving_spots[[j for _, j, _ in pairs]] - positions

    record = dict(reference=fixed_record, moving=moving_record, matched=len(pairs))
    return positions, displacements, record, fixed_notes + moving_notes


def landmark_details(record, control_points, notes):
    """The RegistrationDiagnostics fields of a landmark estimator: the three counts, landmarks and warnings."""
    return dict(reference_landmark_count=record["reference"]["candidates"],
                moving_landmark_count=record["moving"]["candidates"],
                matched_landmark_count=record["matched"],
                landmarks=dict(reference=record["reference"], moving=record["moving"],
                               control_points=int(control_points)),
                warnings=notes)


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


def _landmark_settings(config):
    """Keyword arguments shared by the TPS and CPD estimators."""
    return dict(
        detection_threshold=config.detection_noise_sigma,
        max_control_points=config.max_control_points,
        grid_spacing=config.grid_spacing_voxels,
        zoom_order=config.interpolation_order,
        field_smooth_sigma=config.field_smoothing_sigma,
        clamp_sampling_coordinates=config.clamp_sampling_coordinates,
    )
