"""The spot-finding method registry: exact config type -> SpotFindingSpec, and the private method functions."""
from collections.abc import Callable, Mapping
from dataclasses import KW_ONLY, dataclass, field
from itertools import product
import re
import sys

import numpy as np
import pandas as pd
from scipy.ndimage import center_of_mass, find_objects, label, maximum_filter
from scipy.spatial import cKDTree
from skimage._shared.coord import ensure_spacing
from skimage.feature import peak_local_max
from skimage.morphology import local_maxima

from starfinder._execution import DEVICES
from starfinder._registry import Dependency, check_shared, require
from ._config import (LocalMaximaConfig, NoiseLandmarkConfig, PercentileCentroidConfig, PiscisConfig, SpotiflowConfig,
    StarfishLogConfig)
from ._errors import SpotFindingBackendUnavailableError, SpotFindingError
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
    receive every channel); device is the checked execution device;
    round_name is the round FOV.find_spots detects (None for find_spots),
    which a refusal names.
    """
    channels: tuple[int, ...]
    device: str = "cpu"
    round_name: str | None = None


@dataclass(frozen=True)
class SpotFindingSpec:
    """Registered method: stable snake_case name, private method function and declared capabilities.

    Parameters
    ----------
    name : str
        Stable snake_case method identifier.
    run : callable
        Method function taking image, config and context, and returning a spot table
        and diagnostics.
    pipeline : bool
        Whether the method is available for pipeline spot finding.
    dimensions : frozenset of int
        Supported image dimensionalities drawn from 2 and 3.
    output_columns : tuple of str
        Ordered method-output column names, with a trailing ``?`` marking an
        optional column.
    column_fields : mapping of str to str
        Optional-column names mapped to the Boolean config fields that enable them;
        an empty mapping is the default.
    weights : bool
        Whether the config names pretrained weights; defaults to false.
    requires : tuple of Dependency
        Optional dependencies for the method; the default empty tuple declares no
        dependencies.
    min_shape_zyx : tuple of int
        Minimum Y and X sizes, and the minimum Z size of 3D input, as positive integers in ZYX voxel counts. Defaults to (1, 1, 1).
    devices : frozenset of str
        Execution devices the method runs on, from ``cpu`` and ``cuda``. Defaults to ``cpu`` only.

    Notes
    -----
    run(image, config, context) receives the validated ZYX or ZYXC image
    (a ZYX image has one channel) and returns the method's table (the declared columns,
    without spot_id) and its diagnostics: ``thresholds``, one value per
    detected channel in context.channels order (one value for a method that
    combines channels), and for local maxima ``noise``, one record per
    detected channel, and ``merged``, the removed count per detected channel
    when the merge is on. Local maxima and noise landmarks add ``plateaus``,
    per channel the ``maxima`` and ``removed_voxels`` counts before exclusions;
    find_spots omits the key when every count is zero. A method may add ``geometry`` (a mapping),
    ``measurements`` (column -> meaning), ``effective`` (one config per
    detected channel with the native defaults that None resolved) and
    ``model`` (the weights record of a learned method). find_spots calls it
    after the checks below; callers never call it directly. pipeline is True when FOV.find_spots and
    PipelineConfig.spot_finding accept the method. dimensions holds 2 when a
    Z=1 input is detected as a YX plane and 3 when Z>1 is detected in 3D.
    output_columns are the spot-table columns besides spot_id, in order; a
    trailing ``?`` marks an optional column. column_fields maps each
    optional column to the Boolean config field that adds it, so the columns
    a config produces follow from the spec. weights is True when the config
    names pretrained weights from KNOWN_MODELS. requires lists optional
    dependencies, imported when the method runs; min_shape_zyx is the
    smallest accepted size of each axis (a Z=1 input is checked against its
    last two entries). devices lists the execution devices find_spots accepts
    for the method (check_method_device); Spotiflow and Piscis add ``cuda``.
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
    devices: frozenset[str] = frozenset({"cpu"})

    def __post_init__(self):
        check_shared(self, "spot-finding method")
        if not isinstance(self.pipeline, bool) or not isinstance(self.weights, bool):
            raise TypeError(f"pipeline and weights of {self.name!r} must be Boolean")
        if (not isinstance(self.devices, frozenset) or "cpu" not in self.devices
                or not self.devices <= set(DEVICES)):
            raise ValueError(f"devices of {self.name!r} must be a frozenset of {', '.join(DEVICES)} with cpu")
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


def _regional_maxima(channel, threshold, *, include_constant=False):
    """Yield full-connectivity maxima above threshold, as lexicographically ordered ZYX voxels."""
    plane = channel.shape[0] == 1
    image = channel[0] if plane else channel
    if include_constant and image.min() == image.max() and image.flat[0] > threshold:
        yield np.argwhere(np.ones(channel.shape, dtype=bool))
        return
    mask = local_maxima(image, connectivity=image.ndim, allow_borders=True) & (image > threshold)
    components, _ = label(mask, structure=np.ones((3,) * image.ndim, dtype=bool))
    for number, box in enumerate(find_objects(components), 1):
        points = np.argwhere(components[box] == number) + np.array([s.start for s in box])
        yield np.column_stack((np.zeros(len(points), dtype=int), points)) if plane else points


def _representative(points):
    """Centroid rounded half up if in the maximum, otherwise nearest with ZYX tie-break."""
    centroid = points.mean(axis=0)
    rounded = np.floor(centroid + 0.5).astype(int)
    if np.any(np.all(points == rounded, axis=1)):
        return rounded
    # Squared distances times n, omitting the common centroid term: n * ||p||² - 2 p·sum(points).
    # Integer arithmetic makes distance ties exact, independent of axis summation order.
    count, largest = len(points), int(points.max())
    if 9 * count * largest ** 2 <= np.iinfo(np.int64).max:
        scores = count * np.sum(points * points, axis=1) - 2 * (points @ points.sum(axis=0))
        return points[np.argmin(scores)]
    totals = tuple(sum(map(int, points[:, axis])) for axis in range(3))
    nearest = min(range(count), key=lambda i: count * sum(int(v) ** 2 for v in points[i])
                  - 2 * sum(int(v) * total for v, total in zip(points[i], totals)))
    return points[nearest]


def _peaks(channel, distance, threshold, border=True):
    plane = channel.shape[0] == 1
    image = channel[0] if plane else channel
    # Probe before border/window/spacing exclusions, so suppressed plateaus still get counted.
    candidates = peak_local_max(image, min_distance=1, threshold_abs=threshold, exclude_border=False)
    candidates = np.column_stack((np.zeros(len(candidates), dtype=int), candidates)) if plane else candidates
    tied = False
    values = channel[tuple(candidates.T)]
    for offset in product((0,) if plane else (-1, 0, 1), (-1, 0, 1), (-1, 0, 1)):
        if not any(offset):
            continue
        neighbours = candidates + offset
        inside = np.all((neighbours >= 0) & (neighbours < channel.shape), axis=1)
        if np.any(channel[tuple(neighbours[inside].T)] == values[inside]):
            tied = True
            break
    if not tied:
        # Use the original numerical window and spacing path verbatim on untied maxima.
        # This also preserves scipy's source-dtype window behaviour for large integers.
        coords = peak_local_max(image, min_distance=distance, threshold_abs=threshold, exclude_border=border)
        coords = np.column_stack((np.zeros(len(coords), dtype=int), coords)) if plane else coords
        return coords, {'maxima': 0, 'removed_voxels': 0}
    regions = list(_regional_maxima(channel, threshold))
    plateaus = {'maxima': sum(len(p) > 1 for p in regions),
                'removed_voxels': sum(len(p) - 1 for p in regions)}
    coords = np.asarray([_representative(p) for p in regions], dtype=int).reshape(-1, 3)
    if not len(coords):
        return coords, plateaus
    axes = slice(1, None) if plane else slice(None)
    if border:
        active = coords[:, axes]
        shape = np.array(channel.shape)[axes]
        coords = coords[np.all((active >= distance) & (active < shape - distance), axis=1)]
    if distance > 1:
        window = maximum_filter(channel[0] if plane else channel, size=2 * distance + 1, mode='nearest')
        maxima = window[tuple(coords[:, axes].T)]
        coords = coords[channel[tuple(coords.T)] == maxima]
    # Sort in the source dtype, as peak_local_max did (uint64 can exceed float64 precision).
    values = channel[tuple(coords.T)]
    coords = coords[np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0], -values))]
    if distance > 1:
        coords = ensure_spacing(coords, spacing=distance, p_norm=np.inf)
    return coords, plateaus


def _noise_record(image, channel, mode, value, *, landmark=False):
    """One channel's noise record: zero fraction, median, MAD (unscaled) and threshold in float64, and the
    estimation set of the median and MAD with its voxel count.

    The estimation set is every voxel ('all') except in local-maxima noise mode on a channel with signal of which
    more than half the voxels equal its minimum: then it is the voxels above the minimum, which a truncation did
    not clip ('above_minimum', W-789). Noise landmarks (landmark) keep every voxel.

    On an integer image the MAD of a noise-mode channel with signal has a floor of one grey level (W-848): the
    threshold is median + value x 1.4826 x max(MAD, 1), 'mad' stays the measured MAD and 'mad_floored' says
    whether the floor replaced it. A constant channel has no spot, so its record is not floored.
    """
    values = channel.astype(np.float64)
    estimation, sample = 'all', values
    if mode == 'noise' and not landmark:
        above = values > values.min()
        count = int(np.count_nonzero(above))
        if count and 2 * count < values.size:
            estimation, sample = 'above_minimum', values[above]
    median = np.median(sample)
    mad = np.median(np.abs(sample - median))
    floored = False
    if mode == 'noise':
        scale = mad
        if (not landmark and np.issubdtype(image.dtype, np.integer) and mad < 1
                and channel.max() != channel.min()):
            scale, floored = 1.0, True
        threshold = median + value * scale * 1.4826
    elif mode == 'global':
        threshold = np.iinfo(image.dtype).max * value
    elif mode == 'absolute':
        threshold = float(value)
    elif mode == 'adaptive_round':
        threshold = float(image.max()) * value
    else:
        threshold = float(channel.max()) * value
    return {'zero_fraction': float(np.mean(values == 0)), 'median': float(median), 'mad': float(mad),
            'threshold': float(threshold), 'estimation': estimation, 'n_estimation': int(sample.size),
            'mad_floored': floored}


def check_noise(record, channel, label, round_name):
    """Raise SpotFindingError for a noise-mode float channel with signal whose MAD over its estimation set is 0.

    Its threshold would be the median, and every regional maximum above it a spot (W-789); there is no fallback.
    One grey level is not defined for floating point, so the floor of an integer image (W-848) does not apply.
    An integer channel never raises: its MAD is floored at one grey level.
    """
    minimum = channel.min()
    if record['mad'] != 0 or channel.max() == minimum or np.issubdtype(channel.dtype, np.integer):
        return
    where = f"round {round_name!r}, " if round_name is not None else ""
    at_minimum = int(np.count_nonzero(channel == minimum))
    raise SpotFindingError(
        f"spot finding: {where}channel {label!r}: the noise MAD is 0 over its estimation set "
        f"{record['estimation']!r} ({record['n_estimation']} voxels); {at_minimum} of its {channel.size} voxels "
        f"({at_minimum / channel.size:.1%}) are at its minimum {float(minimum)!r}, so threshold_mode 'noise' has no "
        "noise to scale; use threshold_mode 'adaptive' or 'absolute'")


def _maxima(image, config, context, mode, value, border, landmark=False):
    """Per-channel maxima above each channel's threshold; rows ordered by channel, then intensity.

    Returns the rows, thresholds and per-channel noise and plateau records.
    Noise holds zero fraction, median, MAD (unscaled) and threshold in float64, the estimation set and whether the
    MAD was floored; local maxima (not landmark) in noise mode refuse a float channel with signal and a MAD of 0
    (check_noise) and floor the MAD of an integer channel at one grey level.
    """
    if mode == 'global' and image.dtype not in (np.dtype('uint8'), np.dtype('uint16')):
        raise ValueError("global thresholds require uint8/uint16")
    rows, thresholds, noise, plateaus = [], [], [], []
    labels = config.channel_labels
    for c in context.channels:
        channel = image[..., c] if image.ndim == 4 else image
        record = _noise_record(image, channel, mode, value, landmark=landmark)
        if mode == 'noise' and not landmark:
            check_noise(record, channel, labels[c] if labels is not None else str(c), context.round_name)
        thresholds.append(record['threshold'])
        noise.append(record)
        if channel.max() == 0:
            plateaus.append({'maxima': 0, 'removed_voxels': 0})
            continue
        coords, collapsed = _peaks(channel, config.min_distance_voxels, record['threshold'], border)
        plateaus.append(collapsed)
        for z, y, x in coords:
            rows.append((z, y, x, c, float(channel[z, y, x])))
    table = pd.DataFrame(rows, columns=['z', 'y', 'x', 'channel', 'peak_intensity']).astype(
        {'z': 'float64', 'y': 'float64', 'x': 'float64', 'channel': 'int64', 'peak_intensity': 'float64'})
    return table, tuple(thresholds), noise, plateaus


def _merge(table, radius, channels):
    """The within-channel merge of the maxima rows; returns the kept rows (in their order) and removed counts.

    Per channel, maxima are visited by decreasing pixel value (peak_intensity),
    then increasing z, y, x; one is dropped when an earlier kept maximum lies
    within the ellipsoid sum((d_i / r_i) ** 2) <= 1. A KD-tree on the
    radius-scaled coordinates proposes neighbours, which the ellipsoid test
    then decides. Z=1 maxima all have z=0, so the Z radius has no effect.
    """
    # W-218
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
    table, thresholds, noise, plateaus = _maxima(image, config, context, config.threshold_mode, config.threshold_value,
                                       config.exclude_border)
    details = {'thresholds': thresholds, 'noise': noise, 'plateaus': plateaus}
    if config.merge_radius_zyx is not None:
        table, details['merged'] = _merge(table, np.asarray(config.merge_radius_zyx, dtype=float), context.channels)
    if not config.measure_peak_intensity:
        table = table.drop(columns='peak_intensity')
    return table, details


def _noise_landmark(image, config, context):
    """Registration landmarks: noise-mode peaks with the border excluded, deduplicated across channels."""
    table, thresholds, _, plateaus = _maxima(image, config, context, 'noise', config.noise_sigma, True, landmark=True)
    if len(table) > 1 and image.ndim == 4:
        pairs = cKDTree(table[['z', 'y', 'x']]).query_pairs(r=config.min_distance_voxels)
        table = table.drop(index={max(i, j) for i, j in pairs}).reset_index(drop=True)
    return table[['z', 'y', 'x']], {'thresholds': thresholds, 'plateaus': plateaus}


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
        min_shape_zyx=(7, 6, 6), devices=frozenset({"cpu", "cuda"})),
    PiscisConfig: SpotFindingSpec(
        "piscis", piscis, pipeline=True, dimensions=frozenset({2, 3}),
        output_columns=("z", "y", "x", "channel", "peak_intensity"), weights=True,
        requires=(Dependency("piscis", "piscis", "piscis"), Dependency("torch", "torch", "piscis")),
        min_shape_zyx=(2, 1, 1), devices=frozenset({"cpu", "cuda"})),
}

# Annotation alias for a registered config; a test keeps its members equal to the registry keys.
SpotFindingConfig = (LocalMaximaConfig | NoiseLandmarkConfig | PercentileCentroidConfig | StarfishLogConfig
                     | SpotiflowConfig | PiscisConfig)


def per_channel(spec) -> bool:
    """Whether the method detects each channel on its own (its table has a channel column)."""
    return "channel" in [c.rstrip("?") for c in spec.output_columns]


def output_columns(spec, config) -> list[str]:
    """The columns (besides spot_id) a config of the method produces: its required and enabled optional columns."""
    return [c.rstrip("?") for c in spec.output_columns
            if not c.endswith("?") or getattr(config, spec.column_fields[c.rstrip("?")])]


def empty_table(spec, config) -> pd.DataFrame:
    """The typed empty table of a config: its output columns, channel int64 and the others float64."""
    return pd.DataFrame({c: pd.Series(dtype="int64" if c == "channel" else "float64")
                         for c in output_columns(spec, config)})


def _local_maxima_constant(image, config, channels, verified):
    if config.threshold_mode == 'global' and image.dtype not in (np.dtype('uint8'), np.dtype('uint16')):
        raise ValueError("global thresholds require uint8/uint16")
    noise = [_noise_record(image, image[..., c] if image.ndim == 4 else image, config.threshold_mode,
                           config.threshold_value) for c in channels]
    details = {'thresholds': tuple(r['threshold'] for r in noise), 'noise': noise,
               'plateaus': [{'maxima': 0, 'removed_voxels': 0} for _ in channels]}
    if config.merge_radius_zyx is not None:
        details['merged'] = [0] * len(channels)
    return details


def _starfish_log_constant(image, config, channels, verified):
    from ._starfish_log import MEASUREMENTS, check_plane, check_range, memory_geometry
    check_plane(image, config)
    for c in channels:
        check_range(image[..., c] if image.ndim == 4 else image)
    return {'thresholds': tuple(float(config.threshold) for _ in channels),
            'geometry': memory_geometry(image, config), 'measurements': MEASUREMENTS}


def _learned_constant(method, image, config, verified, threshold):
    """Thresholds and, unless this call has verified the weights already, the geometry check, the weights
    verification and the model record, which a run reports even when no channel reaches the model."""
    from ._learned import model_record, verified_folder
    from ._spotiflow import check_geometry
    from ._weights import known_weights
    details = {'thresholds': threshold}
    if not verified:
        if method == "spotiflow":
            check_geometry(known_weights(method, config.model), image.shape[:3])
        details['model'] = model_record(method, config.model, verified_folder(method, config.model))
    return details


def _spotiflow_constant(image, config, channels, verified):
    from ._weights import known_weights
    threshold = (config.prob_thresh if config.prob_thresh is not None
                 else known_weights("spotiflow", config.model).native_threshold)
    return _learned_constant("spotiflow", image, config, verified, tuple(float(threshold) for _ in channels))


def _piscis_constant(image, config, channels, verified):
    return _learned_constant("piscis", image, config, verified, tuple(float(config.threshold) for _ in channels))


# What find_spots records for a constant channel, which it never passes to the method function. A
# per-channel method without an entry receives its constant channels like any other channel.
_CONSTANT = {LocalMaximaConfig: _local_maxima_constant, StarfishLogConfig: _starfish_log_constant,
             SpotiflowConfig: _spotiflow_constant, PiscisConfig: _piscis_constant}


def skips_constant(config) -> bool:
    """Whether find_spots records the config's constant channels without passing them to the method."""
    return type(config) in _CONSTANT


def constant_details(image, config, channels, verified) -> dict:
    """The thresholds of constant channels, without running the method (a method with an entry in _CONSTANT).

    Local maxima adds the noise records (and merged and plateau counts 0) and the LoG its
    memory estimate (geometry) and measurements, which need no detection.
    Spotiflow and Piscis still check the geometry and verify the weights
    and add the model record, unless verified (the method function has
    verified them in this call), so a missing or changed weights file raises
    as on any other image.
    """
    return _CONSTANT[type(config)](image, config, channels, verified)


def require_method(spec) -> None:
    """Import the method's optional dependencies; SpotFindingBackendUnavailableError names the module and extra."""
    try:
        require(spec, "spot-finding method", SpotFindingBackendUnavailableError)
    except SpotFindingBackendUnavailableError as error:
        if _PY314 and any(d.extra for d in spec.requires):
            raise SpotFindingBackendUnavailableError(
                f"{error} (the extra is not available on Python 3.14 and later)") from error.__cause__
        raise


def check_method_device(spec, device, context="device") -> None:
    """Raise ValueError unless the method runs on device (spec.devices), naming the methods that accept it."""
    if device not in spec.devices:
        accepting = sorted(other.name for other in SPOT_FINDING_METHODS.values()
                           if other.pipeline and device in other.devices)
        raise ValueError(f"{context} {device!r}: spot-finding method {spec.name!r} runs on "
                         f"{', '.join(sorted(spec.devices))} only; {device} is for {', '.join(accepting)}")


def require_gpu() -> None:
    """Raise ValueError when torch sees no GPU, as segmentation does for device 'cuda'."""
    import torch
    if not torch.cuda.is_available():
        raise ValueError("device 'cuda' was requested, but torch sees no GPU")


def check_columns(spec, table):
    """Raise ValueError unless the method returned exactly its declared columns, in order."""
    columns = list(table.columns)
    required = [c for c in spec.output_columns if not c.endswith("?")]
    declared = [c.rstrip("?") for c in spec.output_columns]
    if [c for c in declared if c in columns] != columns or any(c not in columns for c in required):
        raise ValueError(f"spot-finding method {spec.name!r} returned columns {columns}, "
                         f"not its declared {list(spec.output_columns)}")
