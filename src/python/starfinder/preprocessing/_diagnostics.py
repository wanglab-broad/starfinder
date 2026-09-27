"""Per-channel diagnostics of the shared numerical policy (preprocessing algorithms page)."""
import numpy as np

#: Sigma multiple of the default local-maxima noise threshold.
NOISE_SIGMA = 5


def channel_diagnostics(image):
    """Zero fraction, median, MAD, noise threshold and mad_zero per channel of a ZYX(C) image.

    The noise threshold is computed as local-maxima detection computes it in
    noise mode with the default threshold_value (median + 5 * MAD * 1.4826),
    in float64. Values are lists with one entry per channel.
    """
    channels = image[..., None] if image.ndim == 3 else image
    result = {"zero_fraction": [], "median": [], "mad": [], "noise_threshold": [], "mad_zero": []}
    for c in range(channels.shape[-1]):
        values = channels[..., c].astype(np.float64)
        median = np.median(values)
        mad = np.median(np.abs(values - median))
        result["zero_fraction"].append(float(np.count_nonzero(values == 0) / values.size))
        result["median"].append(float(median))
        result["mad"].append(float(mad))
        result["noise_threshold"].append(float(median + NOISE_SIGMA * mad * 1.4826))
        result["mad_zero"].append(bool(mad == 0))
    return result
