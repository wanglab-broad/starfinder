"""The configs of the registered spot-finding methods: frozen dataclasses validated at construction.

Each config's method discriminator equals its SPOT_FINDING_METHODS spec name.
"""
from dataclasses import dataclass, field
from numbers import Integral, Real

import numpy as np


def _number(value, name, lower, upper=None):
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not np.isfinite(value) or value < lower
            or (upper is not None and value > upper)):
        raise ValueError(f"{name} must be finite and in [{lower}, {upper or 'inf'}]")


def _distance(value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError("min_distance_voxels must be a positive integer")


def _labels(labels):
    if labels is not None and (not isinstance(labels, tuple) or not labels
            or any(not isinstance(s, str) or not s for s in labels)
            or len(set(labels)) != len(labels)):
        raise ValueError("channel_labels must be a nonempty tuple of unique strings")


@dataclass(frozen=True)
class LocalMaximaConfig:
    """Per-channel pipeline peaks; no spatial deduplication across channels.

    noise uses median + value * MAD * 1.4826 (sigma units); adaptive uses
    channel maximum, adaptive_round the image maximum, global the uint8/uint16
    maximum (fraction units [0,1]). Border exclusion uses min_distance_voxels.
    Singleton Z is treated as a 2D plane, excluding only the YX border.
    Optional peak_intensity is the original pixel value, without normalization.
    """
    threshold_mode: str = "noise"
    threshold_value: float = 5.0
    min_distance_voxels: int = 1
    exclude_border: bool = True
    channel_labels: tuple[str, ...] | None = None
    measure_peak_intensity: bool = True
    method: str = field(default="local_maxima", init=False)

    def __post_init__(self):
        if self.threshold_mode not in ("noise", "adaptive", "adaptive_round", "global"):
            raise ValueError("unknown threshold_mode")
        _number(self.threshold_value, "threshold_value", 0,
                None if self.threshold_mode == "noise" else 1)
        _distance(self.min_distance_voxels)
        _labels(self.channel_labels)
        if type(self.exclude_border) is not bool or type(self.measure_peak_intensity) is not bool:
            raise ValueError("exclude_border and measure_peak_intensity must be Boolean")


@dataclass(frozen=True)
class NoiseLandmarkConfig:
    """Registration MAD peaks with the original per-channel radius deduplication.

    Keep the lower concatenated index for every pair within min_distance_voxels.
    This is intentionally distinct from pipeline channel-local detections.
    """
    noise_sigma: float = 5.0
    min_distance_voxels: int = 1
    channel_labels: tuple[str, ...] | None = None
    method: str = field(default="noise_landmark", init=False)

    def __post_init__(self):
        _number(self.noise_sigma, "noise_sigma", 0)
        _distance(self.min_distance_voxels)
        _labels(self.channel_labels)


@dataclass(frozen=True)
class PercentileCentroidConfig:
    """Intensity-weighted centroids of face-connected voxels above percentile.

    Channels are summed before thresholding; percentile is in [0,100].
    No channel or peak measurement is assigned to a multichannel centroid.
    """
    threshold_percentile: float = 99.5
    channel_labels: tuple[str, ...] | None = None
    method: str = field(default="percentile_centroid", init=False)

    def __post_init__(self):
        _number(self.threshold_percentile, "threshold_percentile", 0, 100)
        _labels(self.channel_labels)
