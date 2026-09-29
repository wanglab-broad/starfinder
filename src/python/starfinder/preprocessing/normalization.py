"""Finite-array intensity processing with explicit output policies."""
from collections.abc import Mapping
from dataclasses import dataclass
import math

import numpy as np
from skimage.exposure import match_histograms

from starfinder.image import _validate_image
from starfinder.io.conversion import ImageConversionConfig, _cast, _dtype, convert_image
from starfinder.preprocessing._diagnostics import channel_diagnostics
from starfinder.preprocessing.histograms import histogram_percentile

_FITS = ("fov", "supplied")


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
    return _normalize(volume, config)[0]


def _normalize(volume, config):
    """Normalized image and, per group, the fitted data range and applied mode."""
    volume = _validate_image(volume)
    result = np.empty(volume.shape, dtype=config.output_dtype)
    groups = range(volume.shape[-1]) if volume.ndim == 4 and config.scope == "per_channel" else [None]
    fitted = []
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
        fitted.append({"channel": c, "min": channel.min().item(), "max": channel.max().item(), "mode": mode})
    return result, fitted


@dataclass(frozen=True)
class HistogramMatchingConfig:
    """Exact CDF matching; input dtype retained by default.

    Integer values truncate by default. Out-of-range reference values raise
    rather than wrap; an explicit floating output_dtype can retain them.
    As a recipe step with fit="fov", the reference is channel
    reference_channel of the reference round's input to this step. With
    fit="supplied" it is the merged count vector of that round and channel,
    read from the recipe's supplied-statistics file; this requires unsigned
    integer input. match_histogram itself takes the reference volume
    explicitly and reads neither reference_channel nor fit.
    """

    output_dtype: str | None = None
    rounding: str = "truncate"
    reference_channel: int = 0
    fit: str = "fov"

    def __post_init__(self):
        if isinstance(self.reference_channel, bool) or not isinstance(self.reference_channel, int) or self.reference_channel < 0:
            raise ValueError("reference_channel must be a nonnegative integer")
        if self.fit not in _FITS:
            raise ValueError('fit must be "fov" or "supplied"')
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


def _match_counts(volume, values, counts, config):
    """match_histogram against a reference given by its nonzero values and their counts.

    Reproduces scikit-image's CDF mapping for unsigned input, which reads the
    template only through numpy.bincount, so the result equals matching
    against any reference volume with these counts.
    """
    volume = _validate_image(volume)
    if volume.dtype.kind != "u":
        raise ValueError(f"a supplied histogram reference requires unsigned integer input, not {volume.dtype}")
    values, counts = np.asarray(values, dtype=np.int64), np.asarray(counts, dtype=np.int64)
    tmpl_quantiles = np.cumsum(counts) / counts.sum()
    channels = volume[..., None] if volume.ndim == 3 else volume
    dtype = config.output_dtype or volume.dtype
    result = np.empty(channels.shape, dtype=dtype)
    for c in range(channels.shape[-1]):
        source = channels[..., c].reshape(-1)
        src_quantiles = np.cumsum(np.bincount(source)) / source.size
        matched = np.interp(src_quantiles, tmpl_quantiles, values)[source].reshape(channels.shape[:-1])
        result[..., c] = _cast(matched, dtype, config.rounding)
    return result[..., 0] if volume.ndim == 3 else result


@dataclass(frozen=True)
class PercentileNormalizationConfig:
    """Linear map of a per-channel percentile range onto the full output range.

    p_low and p_high are intensity percents, 0 <= p_low < p_high <= 100
    (provisional defaults 1 and 99.9), estimated by the inverted-CDF
    definition. fit="fov" fits them on each round of each FOV; fit="supplied"
    reads low and high per round from the recipe's supplied-statistics file.
    The output keeps the input dtype.
    """

    p_low: float = 1.0
    p_high: float = 99.9
    fit: str = "fov"

    def __post_init__(self):
        for name in ("p_low", "p_high"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if not 0 <= self.p_low < self.p_high <= 100:
            raise ValueError("percentiles require 0 <= p_low < p_high <= 100")
        if self.fit not in _FITS:
            raise ValueError('fit must be "fov" or "supplied"')


def _channel_percentile(channel, p):
    if channel.dtype.name in ("uint8", "uint16"):
        counts = np.bincount(channel.ravel(), minlength=np.iinfo(channel.dtype).max + 1)
        return histogram_percentile(counts, p)
    return np.percentile(channel, p, method="inverted_cdf").item()


def _supplied_range(fitted, n_channels):
    if not isinstance(fitted, Mapping) or not {"low", "high"} <= set(fitted):
        raise ValueError('supplied percentile range requires "low" and "high"')
    low, high = list(fitted["low"]), list(fitted["high"])
    if len(low) != n_channels or len(high) != n_channels:
        raise ValueError(f"supplied percentile range has {len(low)}/{len(high)} channels, the image has {n_channels}")
    for a, b in zip(low, high):
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in (a, b)) or a > b:
            raise ValueError("supplied percentile ranges must be finite with low <= high")
    return low, high


def _percentile_normalize(volume, config, fitted=None):
    """Normalized image, fitted {"low", "high"} and per-channel diagnostics."""
    volume = _validate_image(volume)
    channels = volume[..., None] if volume.ndim == 3 else volume
    n = channels.shape[-1]
    if config.fit == "supplied":
        if fitted is None:
            raise ValueError('fit="supplied" requires the supplied low and high values')
        low, high = _supplied_range(fitted, n)
    else:
        if fitted is not None:
            raise ValueError('fitted values are accepted only with fit="supplied"')
        low = [_channel_percentile(channels[..., c], config.p_low) for c in range(n)]
        high = [_channel_percentile(channels[..., c], config.p_high) for c in range(n)]
    result = np.empty(channels.shape, dtype=volume.dtype)
    degenerate, constant = [], []
    for c in range(n):
        x = channels[..., c].astype(np.float64)
        constant.append(bool(x.min() == x.max()))
        # A constant channel is degenerate in every fit mode, even with supplied low != high.
        degenerate.append(bool(high[c] == low[c]) or constant[c])
        y = np.zeros_like(x) if degenerate[c] else np.clip((x - low[c]) / (high[c] - low[c]), 0, 1)
        if volume.dtype.kind in "ui":
            y = np.rint(y * np.iinfo(volume.dtype).max)
        result[..., c] = y.astype(volume.dtype)
    result = result[..., 0] if volume.ndim == 3 else result
    diagnostics = {**channel_diagnostics(result), "degenerate_range": degenerate, "constant_channel": constant}
    return result, {"low": low, "high": high}, diagnostics


def normalize_percentile(volume: np.ndarray, *, config: PercentileNormalizationConfig = PercentileNormalizationConfig(),
                         fitted: Mapping | None = None) -> np.ndarray:
    """Map each channel's [low, high] percentile range linearly onto the output range.

    Unsigned and signed integer output is
    rint(clip((x - low) / (high - low), 0, 1) * dtype_max) (round half to
    even) in the input dtype; float input maps to [0, 1] in its own dtype.
    Values above high saturate. With fit="fov" the range is fitted per
    channel of this volume (uint8 and uint16 from their integer histogram);
    with fit="supplied", fitted must give {"low": [...], "high": [...]} with
    one value per channel. A constant channel or high == low gives zeros and
    a degenerate_range diagnostic, also with supplied low != high.
    Works in float64 one channel at a time; the input is not modified.

    Raises
    ------
    ValueError
        Empty or nonfinite input, missing or mismatched supplied values, or
        fitted values given with fit="fov".
    """
    return _percentile_normalize(volume, config, fitted)[0]
