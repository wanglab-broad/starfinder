"""3D spot detection using local maxima finding.

Reference: src/matlab/SpotFindingMax3D.m

Detects fluorescent spots (mRNA molecules) in multi-channel 3D volumes by
finding local intensity maxima above a threshold in each channel independently.
"""

import numpy as np
import pandas as pd
from skimage.feature import peak_local_max

SPOT_COLUMNS = ["z", "y", "x", "intensity", "channel"]


def find_spots_3d(
    image: np.ndarray,
    intensity_estimation: str = "adaptive",
    intensity_threshold: float = 0.2,
) -> pd.DataFrame:
    """Detect fluorescent spots in a multi-channel 3D volume.

    For each channel, finds local intensity maxima exceeding a threshold.
    Matches MATLAB SpotFindingMax3D behavior.

    Parameters
    ----------
    image : np.ndarray
        4D array with shape (Z, Y, X, C). Must be integer dtype (uint8/uint16).
    intensity_estimation : str
        Thresholding mode:
        - "adaptive": threshold = channel_max * intensity_threshold
        - "global": threshold = dtype_max * intensity_threshold
    intensity_threshold : float
        Fraction for threshold computation (0-1).

    Returns
    -------
    pd.DataFrame
        Detected spots with columns [z, y, x, intensity, channel].
        Coordinates are 0-based. Channel is 0-based index.
        Empty DataFrame with correct schema if no spots found.
    """
    if image.ndim != 4:
        raise ValueError(f"Expected 4D (Z, Y, X, C) image, got {image.ndim}D")

    n_channels = image.shape[3]
    all_spots = []

    for c in range(n_channels):
        channel = image[:, :, :, c]

        # Compute threshold
        if intensity_estimation == "adaptive":
            threshold_abs = float(channel.max()) * intensity_threshold
        elif intensity_estimation == "global":
            if image.dtype == np.uint8:
                threshold_abs = 255.0 * intensity_threshold
            elif image.dtype == np.uint16:
                threshold_abs = 65535.0 * intensity_threshold
            else:
                raise ValueError(f"Global mode requires uint8/uint16, got {image.dtype}")
        else:
            raise ValueError(
                f"intensity_estimation must be 'adaptive' or 'global', got '{intensity_estimation}'"
            )

        # Skip if threshold would be zero (no signal)
        if threshold_abs <= 0:
            continue

        # Find local maxima above threshold
        # min_distance=1 gives 26-connectivity, matching MATLAB imregionalmax
        coords = peak_local_max(
            channel,
            min_distance=1,
            threshold_abs=threshold_abs,
        )

        if len(coords) == 0:
            continue

        # Extract intensities at detected positions
        intensities = channel[coords[:, 0], coords[:, 1], coords[:, 2]]

        # Build per-channel DataFrame
        channel_spots = pd.DataFrame(
            {
                "z": coords[:, 0],
                "y": coords[:, 1],
                "x": coords[:, 2],
                "intensity": intensities,
                "channel": c,
            }
        )
        all_spots.append(channel_spots)

    if not all_spots:
        return pd.DataFrame(columns=SPOT_COLUMNS)

    return pd.concat(all_spots, ignore_index=True)
