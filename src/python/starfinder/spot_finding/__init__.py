"""Explicit pipeline and registration landmark detection policies.

Coordinates are zero-based voxel indices. IDs are stable within a result's
namespace, including through reordering/subsetting; changing detection settings
or the image does not promise the same identities.
"""
from dataclasses import dataclass, field
from numbers import Integral, Real

import numpy as np
import pandas as pd
from scipy.ndimage import center_of_mass, label
from scipy.spatial import cKDTree
from skimage.feature import peak_local_max

from starfinder.image import ImageMetadata, _validate_image

__all__ = ["LocalMaximaConfig", "NoiseLandmarkConfig", "PercentileCentroidConfig",
           "SpotFindingResult", "find_spots"]


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


@dataclass(frozen=True)
class SpotFindingResult:
    """Typed spot table, source geometry, identity scope and effective policy.

    Required columns: spot_id (pandas string), z/y/x (float64). Optional channel
    (int64) indexes diagnostics['channel_labels']; peak_intensity (float64)
    means the original sampled pixel value. No universal detection score is
    invented. Namespace must include dataset/sample/FOV/subtile when applicable.
    Consumers must join on namespace and spot_id, never row position.
    """
    spots: pd.DataFrame
    metadata: ImageMetadata
    spot_namespace: str
    config: LocalMaximaConfig | NoiseLandmarkConfig | PercentileCentroidConfig
    diagnostics: dict

    def __post_init__(self):
        if not isinstance(self.metadata, ImageMetadata):
            raise TypeError("metadata must be ImageMetadata")
        if not isinstance(self.spot_namespace, str) or not self.spot_namespace.strip():
            raise ValueError("spot_namespace must be nonempty")
        if not isinstance(self.config, (LocalMaximaConfig, NoiseLandmarkConfig, PercentileCentroidConfig)):
            raise TypeError("unsupported detection config")
        table = self.spots
        if not {"spot_id", "z", "y", "x"}.issubset(table.columns) or not table.columns.is_unique:
            raise ValueError("spots require unique spot_id/z/y/x columns")
        if not isinstance(table.spot_id.dtype, pd.StringDtype):
            raise ValueError("spot_id must have pandas string dtype")
        if table.spot_id.isna().any() or (table.spot_id.str.len() == 0).any() or not table.spot_id.is_unique:
            raise ValueError("spot_id must be nonempty and unique within namespace")
        if any(table[c].dtype != np.dtype('float64') for c in ('z', 'y', 'x')) or not np.isfinite(table[['z', 'y', 'x']]).all().all():
            raise ValueError("coordinates must be finite float64")
        if 'channel' in table and (table.channel.dtype != np.dtype('int64') or (table.channel < 0).any()):
            raise ValueError("channel must be nonnegative int64")
        for name in ('peak_intensity', 'integrated_intensity', 'detection_score'):
            if name in table and (table[name].dtype != np.dtype('float64') or not np.isfinite(table[name]).all()):
                raise ValueError(f"{name} must be finite float64")


def _peaks(channel, distance, threshold, border=True):
    # Explicit singleton-Z support; volumetric ordering/policy is unchanged.
    plane = channel.shape[0] == 1
    coords = peak_local_max(channel[0] if plane else channel,
                            min_distance=distance, threshold_abs=threshold,
                            exclude_border=border)
    return np.column_stack((np.zeros(len(coords), dtype=int), coords)) if plane else coords


def find_spots(
    image: np.ndarray,
    *,
    config: LocalMaximaConfig | NoiseLandmarkConfig | PercentileCentroidConfig,
    metadata: ImageMetadata,
    spot_namespace: str,
) -> SpotFindingResult:
    """Detect finite ZYX/ZYXC images using the exact typed detector policy.

    Returns a SpotFindingResult, including typed empty success. Does not match
    landmarks or evaluate registration. Calculation uses float64 for MAD and
    centroid weighting; input pixels are never modified. Global thresholds
    require uint8/uint16. Unknown physical geometry remains unknown.
    """
    image = _validate_image(image)
    if not isinstance(config, (LocalMaximaConfig, NoiseLandmarkConfig, PercentileCentroidConfig)):
        raise TypeError("unsupported detection config")
    if not isinstance(metadata, ImageMetadata):
        raise TypeError("metadata must be ImageMetadata")
    if not isinstance(spot_namespace, str) or not spot_namespace.strip():
        raise ValueError("spot_namespace must be nonempty")
    n_channels = image.shape[3] if image.ndim == 4 else 1
    labels = config.channel_labels
    if labels is not None and len(labels) != n_channels:
        raise ValueError("channel_labels must match the channel axis")
    thresholds = []
    if isinstance(config, PercentileCentroidConfig):
        volume = image if image.ndim == 3 else image.sum(axis=-1)
        threshold = np.percentile(volume, config.threshold_percentile)
        thresholds.append(float(threshold))
        components, count = label(volume > threshold)
        coords = np.asarray(center_of_mass(volume, components, range(1, count + 1)), dtype=float).reshape(-1, 3) if count else np.empty((0, 3))
        table = pd.DataFrame(coords, columns=['z', 'y', 'x'], dtype='float64')
    else:
        pipeline = isinstance(config, LocalMaximaConfig)
        mode = config.threshold_mode if pipeline else 'noise'
        value = config.threshold_value if pipeline else config.noise_sigma
        if mode == 'global' and image.dtype not in (np.dtype('uint8'), np.dtype('uint16')):
            raise ValueError("global thresholds require uint8/uint16")
        rows = []
        for c in range(n_channels):
            channel = image[..., c] if image.ndim == 4 else image
            if mode == 'noise':
                values = channel.astype(np.float64)
                median = np.median(values)
                threshold = median + value * np.median(np.abs(values - median)) * 1.4826
            elif mode == 'global':
                threshold = np.iinfo(image.dtype).max * value
            elif mode == 'adaptive_round':
                threshold = float(image.max()) * value
            else:
                threshold = float(channel.max()) * value
            thresholds.append(float(threshold))
            if channel.max() == 0:
                continue
            coords = _peaks(channel, config.min_distance_voxels, threshold,
                            config.exclude_border if pipeline else True)
            for z, y, x in coords:
                rows.append((z, y, x, c, float(channel[z, y, x])))
        table = pd.DataFrame(rows, columns=['z', 'y', 'x', 'channel', 'peak_intensity']).astype(
            {'z': 'float64', 'y': 'float64', 'x': 'float64', 'channel': 'int64', 'peak_intensity': 'float64'})
        if not pipeline:
            if len(table) > 1 and image.ndim == 4:
                pairs = cKDTree(table[['z', 'y', 'x']]).query_pairs(r=config.min_distance_voxels)
                table = table.drop(index={max(i, j) for i, j in pairs}).reset_index(drop=True)
            table = table[['z', 'y', 'x']]
        elif not config.measure_peak_intensity:
            table = table.drop(columns='peak_intensity')
    table.insert(0, 'spot_id', pd.array([str(i) for i in range(len(table))], dtype='string'))
    diagnostics = {'method': config.method, 'channel_labels': labels,
                   'thresholds': tuple(thresholds), 'coordinate_units': 'voxel_index',
                   'singleton_z_policy': 'YX plane; Z=0',
                   'measurements': ({'peak_intensity': 'original pixel intensity at the detected channel maximum'}
                                    if 'peak_intensity' in table else {})}
    return SpotFindingResult(table, metadata, spot_namespace, config, diagnostics)
