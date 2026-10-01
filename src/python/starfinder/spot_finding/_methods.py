"""The spot-finding method registry: exact config type -> SpotFindingSpec, and the private method functions."""
from collections.abc import Callable, Mapping
from dataclasses import KW_ONLY, dataclass, field
import re
import sys

import numpy as np
import pandas as pd
from scipy.ndimage import center_of_mass, label
from scipy.spatial import cKDTree
from skimage.feature import peak_local_max

from starfinder._registry import Dependency, check_shared, require
from starfinder.image import IncompatibleGeometryError

from ._config import (LocalMaximaConfig, NoiseLandmarkConfig, PercentileCentroidConfig, PiscisConfig, SpotiflowConfig,
    StarfishLogConfig)
from ._errors import SpotFindingBackendUnavailableError
from ._piscis import piscis
from ._spotiflow import spotiflow
from ._starfish_log import starfish_log

_COLUMN = r"[a-z][a-z0-9_]*\??"
# The learned-detector extras install nothing on Python 3.14 and later (torch 2.7.1 has no cp314 wheel).
_PY314 = sys.version_info >= (3, 14)


@dataclass(frozen=True)
class MethodContext:
    """What find_spots passes to a method function besides the image and config.

    channels are the indices of the channel axis to detect (all channels
    unless a plan overrides some of them; methods that combine channels
    receive every channel); device is the checked execution device.
    """
    channels: tuple[int, ...]
    device: str = "cpu"


@dataclass(frozen=True)
class SpotFindingSpec:
    """Registered method: stable snake_case name, private method function and declared capabilities.

    run(image, config, context) receives the validated ZYX or ZYXC image
    (a ZYX image has one channel) and returns the method's table (the declared columns,
    without spot_id) and its diagnostics: ``thresholds``, one value per
    detected channel in context.channels order (one value for a method that
    combines channels), and for local maxima ``noise``, one record per
    detected channel, and ``merged``, the removed count per detected channel
    when the merge is on. A method may add ``geometry`` (a mapping),
    ``measurements`` (column -> meaning), ``effective`` (one config per
    detected channel with the native defaults that None resolved) and
    ``model`` (the weights record of a learned method). find_spots calls it
    after the checks below; callers never call it directly. pipeline is True when FOV.find_spots and
    PipelineConfig.detection accept the method. dimensions holds 2 when a
    Z=1 input is detected as a YX plane and 3 when Z>1 is detected in 3D.
    output_columns are the spot-table columns besides spot_id, in order; a
    trailing ``?`` marks an optional column. column_fields maps each
    optional column to the Boolean config field that adds it, so the columns
    a config produces follow from the spec. weights is True when the config
    names pretrained weights from KNOWN_WEIGHTS. requires lists optional
    dependencies, imported when the method runs; min_shape_zyx is the
    smallest accepted size of each axis (a Z=1 input is checked against its
    last two entries).
    """

    name: str
    run: Callable[..., tuple]
    _: KW_ONLY
    pipeline: bool
    dimensions: frozenset[int]
    output_columns: tuple[str, ...]
    column_fields: Mapping[str, str] = field(default_factory=dict)
    weights: bool = False
    requires: tuple[Dependency, ...] = ()
    min_shape_zyx: tuple[int, int, int] = (1, 1, 1)

    def __post_init__(self):
        check_shared(self, "spot-finding method")
        if not isinstance(self.pipeline, bool) or not isinstance(self.weights, bool):
            raise TypeError(f"pipeline and weights of {self.name!r} must be Boolean")
        if (not isinstance(self.dimensions, frozenset) or not self.dimensions
                or not self.dimensions <= {2, 3}):
            raise ValueError(f"dimensions of {self.name!r} must be a nonempty frozenset of 2 and/or 3")
        columns = self.output_columns
        if (not isinstance(columns, tuple) or not columns
                or any(not isinstance(c, str) or not re.fullmatch(_COLUMN, c) for c in columns)
                or len({c.rstrip("?") for c in columns}) != len(columns)
                or "spot_id" in {c.rstrip("?") for c in columns}
                or [c.rstrip("?") for c in columns[:3]] != ["z", "y", "x"]):
            raise ValueError(f"output_columns of {self.name!r} must be unique names starting with z, y, x, "
                             "without spot_id")
        optional = {c.rstrip("?") for c in columns if c.endswith("?")}
        if (not isinstance(self.column_fields, Mapping) or set(self.column_fields) != optional
                or not all(isinstance(f, str) and f for f in self.column_fields.values())):
            raise ValueError(f"column_fields of {self.name!r} must map each optional column to a config field")


def _peaks(channel, distance, threshold, border=True):
    # Explicit singleton-Z support; volumetric ordering/policy is unchanged.
    plane = channel.shape[0] == 1
    coords = peak_local_max(channel[0] if plane else channel,
                            min_distance=distance, threshold_abs=threshold,
                            exclude_border=border)
    return np.column_stack((np.zeros(len(coords), dtype=int), coords)) if plane else coords


def _maxima(image, config, context, mode, value, border):
    """Per-channel maxima above each channel's threshold; rows ordered by channel, then intensity.

    Returns the rows, the thresholds and, per channel, the noise record:
    zero fraction, median, MAD (unscaled) and threshold, computed in float64.
    """
    if mode == 'global' and image.dtype not in (np.dtype('uint8'), np.dtype('uint16')):
        raise ValueError("global thresholds require uint8/uint16")
    rows, thresholds, noise = [], [], []
    for c in context.channels:
        channel = image[..., c] if image.ndim == 4 else image
        values = channel.astype(np.float64)
        median = np.median(values)
        mad = np.median(np.abs(values - median))
        if mode == 'noise':
            threshold = median + value * mad * 1.4826
        elif mode == 'global':
            threshold = np.iinfo(image.dtype).max * value
        elif mode == 'adaptive_round':
            threshold = float(image.max()) * value
        else:
            threshold = float(channel.max()) * value
        thresholds.append(float(threshold))
        noise.append({'zero_fraction': float(np.mean(values == 0)), 'median': float(median), 'mad': float(mad),
                      'threshold': float(threshold)})
        if channel.max() == 0:
            continue
        coords = _peaks(channel, config.min_distance_voxels, threshold, border)
        for z, y, x in coords:
            rows.append((z, y, x, c, float(channel[z, y, x])))
    table = pd.DataFrame(rows, columns=['z', 'y', 'x', 'channel', 'peak_intensity']).astype(
        {'z': 'float64', 'y': 'float64', 'x': 'float64', 'channel': 'int64', 'peak_intensity': 'float64'})
    return table, tuple(thresholds), noise


def _merge(table, radius, channels):
    """The W-218 within-channel merge of the maxima rows; returns the kept rows (in their order) and removed counts.

    Per channel, maxima are visited by decreasing pixel value (peak_intensity),
    then increasing z, y, x; one is dropped when an earlier kept maximum lies
    within the ellipsoid sum((d_i / r_i) ** 2) <= 1. A KD-tree on the
    radius-scaled coordinates proposes neighbours, which the ellipsoid test
    then decides. Z=1 maxima all have z=0, so the Z radius has no effect.
    """
    keep = np.ones(len(table), dtype=bool)
    removed = []
    for c in channels:
        rows = np.flatnonzero(table['channel'].to_numpy() == c)
        if len(rows) < 2:
            removed.append(0)
            continue
        zyx = table[['z', 'y', 'x']].to_numpy()[rows]
        value = table['peak_intensity'].to_numpy()[rows]
        order = np.lexsort((zyx[:, 2], zyx[:, 1], zyx[:, 0], -value))
        near = cKDTree(zyx / radius).query_ball_point(zyx / radius, r=1.0 + 1e-9)
        kept = np.zeros(len(rows), dtype=bool)
        for i in order:
            kept[i] = not any(kept[j] and (((zyx[i] - zyx[j]) / radius) ** 2).sum() <= 1 for j in near[i] if j != i)
        keep[rows[~kept]] = False
        removed.append(int((~kept).sum()))
    return table[keep].reset_index(drop=True), removed


def _local_maxima(image, config, context):
    """Pipeline peaks per channel; see LocalMaximaConfig."""
    table, thresholds, noise = _maxima(image, config, context, config.threshold_mode, config.threshold_value,
                                       config.exclude_border)
    details = {'thresholds': thresholds, 'noise': noise}
    if config.merge_radius_zyx is not None:
        table, details['merged'] = _merge(table, np.asarray(config.merge_radius_zyx, dtype=float), context.channels)
    if not config.measure_peak_intensity:
        table = table.drop(columns='peak_intensity')
    return table, details


def _noise_landmark(image, config, context):
    """Registration landmarks: noise-mode peaks with the border excluded, deduplicated across channels."""
    table, thresholds, _ = _maxima(image, config, context, 'noise', config.noise_sigma, True)
    if len(table) > 1 and image.ndim == 4:
        pairs = cKDTree(table[['z', 'y', 'x']]).query_pairs(r=config.min_distance_voxels)
        table = table.drop(index={max(i, j) for i, j in pairs}).reset_index(drop=True)
    return table[['z', 'y', 'x']], {'thresholds': thresholds}


def _percentile_centroid(image, config, context):
    """Intensity-weighted centroids of the channel sum above a percentile; see PercentileCentroidConfig."""
    volume = image if image.ndim == 3 else image.sum(axis=-1)
    threshold = np.percentile(volume, config.threshold_percentile)
    components, count = label(volume > threshold)
    coords = (np.asarray(center_of_mass(volume, components, range(1, count + 1)), dtype=float).reshape(-1, 3)
              if count else np.empty((0, 3)))
    return pd.DataFrame(coords, columns=['z', 'y', 'x'], dtype='float64'), {'thresholds': (float(threshold),)}


# The spot-finding method registry (exported with its documentation by starfinder.spot_finding).
SPOT_FINDING_METHODS: dict[type, SpotFindingSpec] = {
    LocalMaximaConfig: SpotFindingSpec(
        "local_maxima", _local_maxima, pipeline=True, dimensions=frozenset({2, 3}),
        output_columns=("z", "y", "x", "channel", "peak_intensity?"),
        column_fields={"peak_intensity": "measure_peak_intensity"}),
    NoiseLandmarkConfig: SpotFindingSpec(
        "noise_landmark", _noise_landmark, pipeline=False, dimensions=frozenset({2, 3}),
        output_columns=("z", "y", "x")),
    PercentileCentroidConfig: SpotFindingSpec(
        "percentile_centroid", _percentile_centroid, pipeline=False, dimensions=frozenset({2, 3}),
        output_columns=("z", "y", "x")),
    StarfishLogConfig: SpotFindingSpec(
        "starfish_log", starfish_log, pipeline=True, dimensions=frozenset({2, 3}),
        output_columns=("z", "y", "x", "channel", "peak_intensity", "radius")),
    SpotiflowConfig: SpotFindingSpec(
        "spotiflow", spotiflow, pipeline=True, dimensions=frozenset({2, 3}),
        output_columns=("z", "y", "x", "channel", "peak_intensity", "probability"), weights=True,
        requires=(Dependency("spotiflow", "spotiflow", "spotiflow"), Dependency("torch", "torch", "spotiflow")),
        min_shape_zyx=(7, 6, 6)),
    PiscisConfig: SpotFindingSpec(
        "piscis", piscis, pipeline=True, dimensions=frozenset({2, 3}),
        output_columns=("z", "y", "x", "channel", "peak_intensity"), weights=True,
        requires=(Dependency("piscis", "piscis", "piscis"), Dependency("torch", "torch", "piscis")),
        min_shape_zyx=(2, 1, 1)),
}

# Annotation alias for a registered config; a test keeps its members equal to the registry keys.
SpotFindingConfig = (LocalMaximaConfig | NoiseLandmarkConfig | PercentileCentroidConfig | StarfishLogConfig
                     | SpotiflowConfig | PiscisConfig)


def per_channel(spec) -> bool:
    """Whether the method detects each channel on its own (its table has a channel column)."""
    return "channel" in [c.rstrip("?") for c in spec.output_columns]


def require_method(spec) -> None:
    """Import the method's optional dependencies; SpotFindingBackendUnavailableError names the module and extra."""
    try:
        require(spec, "spot-finding method", SpotFindingBackendUnavailableError)
    except SpotFindingBackendUnavailableError as error:
        if _PY314 and any(d.extra for d in spec.requires):
            raise SpotFindingBackendUnavailableError(
                f"{error} (the extra is not available on Python 3.14 and later)") from error.__cause__
        raise


def check_shape(spec, shape):
    """Reject a ZYX shape the method does not accept, before it runs."""
    if shape[0] == 1 and 2 not in spec.dimensions:
        raise IncompatibleGeometryError(f"{spec.name} requires 3D input; Z=1 is not supported")
    if shape[0] > 1 and 3 not in spec.dimensions:
        raise IncompatibleGeometryError(f"{spec.name} supports only Z=1 input")
    minimum = spec.min_shape_zyx if shape[0] > 1 else (1, *spec.min_shape_zyx[1:])
    if any(n < m for n, m in zip(shape, minimum)):
        raise IncompatibleGeometryError(
            f"{spec.name} requires 3D input with every axis at least {minimum}, not {tuple(shape)}"
            if shape[0] > 1 else
            f"{spec.name} requires Y and X of at least {minimum[1:]}, not {tuple(shape[1:])}")


def check_columns(spec, table):
    """Raise ValueError unless the method returned exactly its declared columns, in order."""
    columns = list(table.columns)
    required = [c for c in spec.output_columns if not c.endswith("?")]
    declared = [c.rstrip("?") for c in spec.output_columns]
    if [c for c in declared if c in columns] != columns or any(c not in columns for c in required):
        raise ValueError(f"spot-finding method {spec.name!r} returned columns {columns}, "
                         f"not its declared {list(spec.output_columns)}")
