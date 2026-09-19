"""Finite-array intensity processing with explicit output policies."""
from dataclasses import dataclass

import numpy as np
from skimage.exposure import match_histograms

from starfinder.image import _validate_image
from starfinder.io.conversion import ImageConversionConfig, _cast, _dtype, convert_image


@dataclass(frozen=True)
class MinMaxNormalizationConfig:
    """Declared output dtype/range, global or per-channel min-max scaling.

    Integer rounding defaults to truncate, preserving historical uint8 behavior.
    Nonconstant low-SNR groups (max/mean < threshold; nonpositive mean gives
    SNR=0) retain raw values clipped to output_range. Constant groups always
    map to its lower endpoint, including a nonzero lower endpoint when
    explicitly requested, regardless of the SNR gate.
    """

    output_dtype: str
    output_range: tuple[float, float]
    scope: str = "per_channel"
    snr_threshold: float | None = None
    rounding: str = "truncate"

    def __post_init__(self):
        ImageConversionConfig(self.output_dtype, "rescale", output_range=self.output_range, range_policy="data", scope=self.scope, rounding=self.rounding)
        if self.snr_threshold is not None and (not np.isfinite(self.snr_threshold) or self.snr_threshold < 0):
            raise ValueError("snr_threshold must be finite and nonnegative")


def normalize_intensity(volume: np.ndarray, *, config: MinMaxNormalizationConfig) -> np.ndarray:
    """Normalize finite nonempty ZYX/ZYXC data without input mutation.

    Allocates the output and float64 work per channel (or whole volume for
    global scope). No spatial boundary operation. Constants map to the declared
    lower endpoint before SNR gating, for both global and per-channel scope.
    """
    volume = _validate_image(volume)
    result = np.empty(volume.shape, dtype=config.output_dtype)
    groups = range(volume.shape[-1]) if volume.ndim == 4 and config.scope == "per_channel" else [None]
    for c in groups:
        key = (..., c) if c is not None else (...,)
        channel = volume[key]
        mode = "rescale"
        if channel.min() != channel.max():
            mean = channel.mean(dtype=np.float64)
            snr = float(channel.max()) / mean if mean > 0 else 0.0
            if config.snr_threshold is not None and snr < config.snr_threshold:
                mode = "clip"
        conversion = ImageConversionConfig(config.output_dtype, mode, output_range=config.output_range, range_policy="data" if mode == "rescale" else "declared", rounding=config.rounding)
        result[key] = convert_image(channel, config=conversion)
    return result


@dataclass(frozen=True)
class HistogramMatchingConfig:
    """Exact CDF matching; input dtype retained by default.

    Integer values truncate by default. Out-of-range reference values raise
    rather than wrap; an explicit floating output_dtype can retain them.
    """

    output_dtype: str | None = None
    rounding: str = "truncate"

    def __post_init__(self):
        if self.output_dtype is not None:
            _dtype(self.output_dtype)
        if self.rounding not in ("truncate", "nearest_even"):
            raise ValueError("unsupported rounding")


def match_histogram(volume: np.ndarray, reference: np.ndarray, *, config: HistogramMatchingConfig = HistogramMatchingConfig()) -> np.ndarray:
    """Match each ZYX channel to one finite ZYX reference CDF.

    Allocates output plus float64/skimage CDF work for one channel at a time;
    inputs are unchanged. Reference spatial shape may differ. Empty/nonfinite
    inputs error; constant inputs use skimage's exact CDF mapping. No bins or
    spatial padding are used.
    """
    volume = _validate_image(volume)
    reference = _validate_image(reference, ndim=(3,))
    channels = volume[..., None] if volume.ndim == 3 else volume
    dtype = config.output_dtype or volume.dtype
    result = np.empty(channels.shape, dtype=dtype)
    for c in range(channels.shape[-1]):
        matched = match_histograms(channels[..., c], reference)
        result[..., c] = _cast(matched, dtype, config.rounding)
    return result[..., 0] if volume.ndim == 3 else result
