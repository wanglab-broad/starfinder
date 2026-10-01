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


def _positive(value):
    return not isinstance(value, bool) and isinstance(value, Real) and bool(np.isfinite(value)) and value > 0


def _sigma(value, name):
    """A positive finite number or a ZYX 3-tuple of them; returns the per-axis values."""
    if isinstance(value, tuple) and len(value) == 3 and all(_positive(v) for v in value):
        return np.asarray(value, dtype=float)
    if _positive(value):
        return np.full(3, float(value))
    raise ValueError(f"{name} must be a positive finite number or a ZYX 3-tuple of them")


@dataclass(frozen=True)
class LocalMaximaConfig:
    """Per-channel pipeline peaks; no spatial deduplication across channels.

    noise uses median + value * MAD * 1.4826 (sigma units); adaptive uses
    channel maximum, adaptive_round the image maximum, global the uint8/uint16
    maximum (fraction units [0,1]). Border exclusion uses min_distance_voxels.
    Singleton Z is treated as a 2D plane, excluding only the YX border.
    Optional peak_intensity is the original pixel value, without normalization.

    merge_radius_zyx (None: off, the legacy result) is the opt-in W-218
    within-channel merge: after border exclusion a channel's maxima are
    ordered by decreasing pixel value, then increasing z, y, x, and a maximum
    is dropped when an earlier kept maximum of the same channel lies within
    the ellipsoid sum((d_i / r_i) ** 2) <= 1 (radii in voxels; the Z radius
    is unused for Z=1). Kept maxima are unchanged and nothing is averaged;
    diagnostics['merged'] holds the number removed per channel.
    """
    threshold_mode: str = "noise"
    threshold_value: float = 5.0
    min_distance_voxels: int = 1
    exclude_border: bool = True
    channel_labels: tuple[str, ...] | None = None
    measure_peak_intensity: bool = True
    merge_radius_zyx: tuple[float, float, float] | None = None
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
        if self.merge_radius_zyx is not None and not (
                isinstance(self.merge_radius_zyx, tuple) and len(self.merge_radius_zyx) == 3
                and all(_positive(r) for r in self.merge_radius_zyx)):
            raise ValueError("merge_radius_zyx must be None or a ZYX 3-tuple of positive finite radii in voxels")


@dataclass(frozen=True)
class StarfishLogConfig:
    """Native starfish BlobDetector: Laplacian-of-Gaussian blobs of one channel at a time.

    Reproduces starfish BlobDetector(is_volume=True, detector_method="blob_log")
    without a reference image (starfish 1fb00cbc). Integer images are scaled by
    their dtype maximum to float32 [0, 1] (skimage img_as_float32); float images
    must already lie in [0, 1]. min_sigma and max_sigma are sigma in voxels, a
    number or a ZYX 3-tuple (per axis, min_sigma <= max_sigma); num_sigma is the
    number of scales; threshold is absolute on the scale-normalized LoG response
    of the scaled image. starfish has no defaults for these four, so they are
    required. overlap and exclude_border are the blob_log arguments (the
    starfish defaults). A Z=1 image is detected as a YX plane (z=0), where a
    3-tuple sigma raises IncompatibleGeometryError. There is no tiling.
    """
    min_sigma: float | tuple[float, float, float]
    max_sigma: float | tuple[float, float, float]
    num_sigma: int
    threshold: float
    overlap: float = 0.5
    exclude_border: bool | int = False
    channel_labels: tuple[str, ...] | None = None
    method: str = field(default="starfish_log", init=False)

    def __post_init__(self):
        if (_sigma(self.min_sigma, "min_sigma") > _sigma(self.max_sigma, "max_sigma")).any():
            raise ValueError("min_sigma must not exceed max_sigma on any axis")
        if isinstance(self.num_sigma, bool) or not isinstance(self.num_sigma, Integral) or self.num_sigma < 1:
            raise ValueError("num_sigma must be a positive integer")
        _number(self.threshold, "threshold", 0)
        _number(self.overlap, "overlap", 0, 1)
        if not isinstance(self.exclude_border, Integral) or self.exclude_border < 0:
            raise ValueError("exclude_border must be Boolean or a nonnegative integer")
        _labels(self.channel_labels)


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
