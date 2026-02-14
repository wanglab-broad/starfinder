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
    intensity_estimation: str = "noise",
    intensity_threshold: float = 5.0,
    min_distance: int = 1,
) -> pd.DataFrame:
    """Detect fluorescent spots in a multi-channel 3D volume.

    For each channel, finds local intensity maxima exceeding a threshold.
    Based on MATLAB SpotFindingMax3D.

    Parameters
    ----------
    image : np.ndarray
        4D array with shape (Z, Y, X, C). Must be integer dtype (uint8/uint16).
    intensity_estimation : str
        Thresholding mode:
        - "adaptive": threshold = channel_max * intensity_threshold
        - "adaptive_round": threshold = round_max * intensity_threshold
          (max across ALL channels, not per-channel)
        - "global": threshold = dtype_max * intensity_threshold
        - "noise": threshold = median + intensity_threshold * MAD * 1.4826
          (noise-floor based, intensity_threshold acts as k-sigma)
    intensity_threshold : float
        Fraction for threshold computation (0-1) in adaptive/global modes.
        Number of sigma above noise floor in "noise" mode (typically 3-5).
    min_distance : int
        Minimum number of pixels separating detected peaks.
        Higher values suppress nearby false positives. MATLAB's
        ``imregionalmax`` + ``regionprops3`` effectively merges nearby
        peaks via connected-component analysis; ``min_distance >= 2``
        approximates this behavior.

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

    # Pre-compute round-level max for adaptive_round mode
    if intensity_estimation == "adaptive_round":
        round_max = float(max(
            image[:, :, :, c].max() for c in range(n_channels)
        ))

    for c in range(n_channels):
        channel = image[:, :, :, c]

        # Compute threshold
        if intensity_estimation == "adaptive":
            threshold_abs = float(channel.max()) * intensity_threshold
        elif intensity_estimation == "adaptive_round":
            threshold_abs = round_max * intensity_threshold
        elif intensity_estimation == "global":
            if image.dtype == np.uint8:
                threshold_abs = 255.0 * intensity_threshold
            elif image.dtype == np.uint16:
                threshold_abs = 65535.0 * intensity_threshold
            else:
                raise ValueError(f"Global mode requires uint8/uint16, got {image.dtype}")
        elif intensity_estimation == "noise":
            ch_float = channel.astype(np.float64)
            med = np.median(ch_float)
            mad = np.median(np.abs(ch_float - med))
            # 1.4826 converts MAD to σ under Gaussian assumption
            threshold_abs = med + intensity_threshold * mad * 1.4826
        else:
            raise ValueError(
                f"intensity_estimation must be 'adaptive', 'adaptive_round', "
                f"'global', or 'noise', got '{intensity_estimation}'"
            )

        # Skip if threshold is negative (shouldn't happen) or if
        # channel has no signal at all
        if channel.max() == 0:
            continue

        # Find local maxima above threshold
        coords = peak_local_max(
            channel,
            min_distance=min_distance,
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
