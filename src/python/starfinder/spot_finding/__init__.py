"""Explicit pipeline and registration landmark detection policies.

Coordinates are zero-based voxel indices. IDs are stable within a result's
namespace, including through reordering/subsetting; changing detection settings
or the image does not promise the same identities.
"""
from dataclasses import dataclass, replace
import warnings

import numpy as np
import pandas as pd

from starfinder._execution import check_device, execution_record
from starfinder._registry import spec_for
from starfinder.image import ImageMetadata, _validate_image

from ._config import (LocalMaximaConfig, NoiseLandmarkConfig, PercentileCentroidConfig, PiscisConfig, SpotiflowConfig,
    StarfishLogConfig)
from ._errors import MissingWeightsError, SpotFindingBackendUnavailableError, SpotFindingWarning, WeightsHashMismatchError
from ._methods import SPOT_FINDING_METHODS as _SPOT_FINDING_METHODS
from ._methods import (MethodContext, SpotFindingConfig, SpotFindingSpec, check_columns, check_shape, constant_details,
    empty_table, per_channel, require_method, skips_constant)
from ._plan import ChannelOverride, SpotFindingPlan
from ._plot import plot_detections
from ._weights import KNOWN_WEIGHTS as _KNOWN_WEIGHTS
from ._weights import KnownWeights, WeightsFile, fetch_weights, resolve_weights

#: The spot-finding method registry, mapping each exact frozen config type to its
#: SpotFindingSpec. find_spots, SpotFindingResult, SpotFindingPlan, PipelineConfig.detection,
#: FOV.find_spots, the workflow adapter and the checkpoint reader derive their method sets from it.
SPOT_FINDING_METHODS = _SPOT_FINDING_METHODS

#: The known-weights table, mapping (method, model) to its KnownWeights entry. It lists the
#: pretrained weights Starfinder can fetch and verify, and sets no default model.
KNOWN_WEIGHTS = _KNOWN_WEIGHTS

__all__ = ["LocalMaximaConfig", "NoiseLandmarkConfig", "PercentileCentroidConfig", "StarfishLogConfig",
           "SpotiflowConfig", "PiscisConfig",
           "SpotFindingResult", "find_spots", "plot_detections", "SPOT_FINDING_METHODS", "SpotFindingSpec",
           "SpotFindingPlan", "ChannelOverride", "SpotFindingBackendUnavailableError", "SpotFindingWarning",
           "KNOWN_WEIGHTS", "KnownWeights", "WeightsFile", "resolve_weights", "fetch_weights",
           "MissingWeightsError", "WeightsHashMismatchError"]

_WHAT = "spot-finding method"
# Native score and size columns, summarised per channel in diagnostics['native'].
_NATIVE_COLUMNS = ("probability", "radius")


@dataclass(frozen=True)
class SpotFindingResult:
    """Typed spot table, source geometry, identity scope and effective policy.

    Required columns: spot_id (pandas string), z/y/x (float64). Optional channel
    (int64) indexes diagnostics['channel_labels']; peak_intensity (float64)
    means the original sampled pixel value. No universal detection score is
    invented. Namespace must include dataset/sample/FOV/subtile when applicable.
    Consumers must join on namespace and spot_id, never row position.
    config is the registered config of the method (a plan's base config; the
    per-channel configs are in diagnostics['effective_settings']). plan is
    the SpotFindingPlan the result was detected with; None means
    SpotFindingPlan(config), and plan.config must equal config. A result of
    a plan with rounds has a ``round`` column (pandas string): every row's
    detection round, with spot_id running over the combined table.
    The repr is a one-line spot count and column names without table rows.
    """
    spots: pd.DataFrame
    metadata: ImageMetadata
    spot_namespace: str
    config: SpotFindingConfig
    diagnostics: dict
    plan: SpotFindingPlan | None = None

    def __post_init__(self):
        if not isinstance(self.metadata, ImageMetadata):
            raise TypeError("metadata must be ImageMetadata")
        if not isinstance(self.spot_namespace, str) or not self.spot_namespace.strip():
            raise ValueError("spot_namespace must be nonempty")
        spec_for(SPOT_FINDING_METHODS, self.config, _WHAT, TypeError, "unsupported detection config")
        if self.plan is None:
            object.__setattr__(self, "plan", SpotFindingPlan(self.config))
        if not isinstance(self.plan, SpotFindingPlan) or self.plan.config != self.config:
            raise ValueError("plan must be a SpotFindingPlan whose config equals the result's config")
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
        if 'round' in table and (not isinstance(table['round'].dtype, pd.StringDtype) or table['round'].isna().any()
                                 or (table['round'].str.len() == 0).any()):
            raise ValueError("round must be nonempty labels of pandas string dtype")

    def _summary(self):
        columns = [str(c) for c in self.spots.columns]
        shown = ", ".join(columns[:4] + (["..."] if len(columns) > 4 else []))
        return f"{len(self.spots)} spots × [{shown}]"

    def __repr__(self):
        return f"SpotFindingResult: {self._summary()}"


def find_spots(
    image: np.ndarray,
    *,
    config: SpotFindingConfig | SpotFindingPlan,
    metadata: ImageMetadata,
    spot_namespace: str,
    device: str = "cpu",
) -> SpotFindingResult:
    """Detect finite ZYX/ZYXC images with a registered config or a SpotFindingPlan.

    The method is the SPOT_FINDING_METHODS entry of the config's exact type;
    a plan's overrides replace the whole config of their channels (a plan
    with overrides needs config.channel_labels). device must be "cpu".
    Returns a SpotFindingResult, including typed empty success. Does not match
    landmarks or evaluate registration. Calculation uses float64 for MAD and
    centroid weighting; input pixels are never modified. Global thresholds
    require uint8/uint16. Unknown physical geometry remains unknown.

    Local maxima records, per channel and in every threshold mode, the noise
    record diagnostics['noise'] (zero fraction, median, MAD and threshold)
    and emits a SpotFindingWarning, also listed in diagnostics['warnings'],
    when a channel's MAD is 0 or more than half of its voxels are zero; the
    threshold is unchanged. With merge_radius_zyx set, diagnostics['merged']
    holds the number of maxima the within-channel merge removed per channel.
    The Starfish LoG records its scale-space memory estimate in
    diagnostics['geometry'] (scale_space_bytes_estimate, 10.4 bytes x
    num_sigma x voxels). Spotiflow and Piscis need their extras and named
    weights from the Starfinder cache, which every call re-hashes before the
    model is built (nothing is downloaded); they record the loaded files in
    diagnostics['model'] (method, model, artifacts with path, SHA-256, source
    and revision, and the training pixel size) and their tiling in
    diagnostics['geometry'] (Spotiflow n_tiles; Piscis mode, tile size,
    overlap and keep-boundaries per lateral axis).
    diagnostics['effective_settings'] holds every channel's effective config,
    with the native defaults that None resolved (such as Spotiflow's stored
    prob_thresh), and diagnostics['execution'] the execution entry (device,
    framework, threads). The pipeline methods never receive a constant
    channel: it yields no candidates, and its threshold (and the local-maxima
    noise record) is still recorded; Spotiflow and Piscis still verify their
    weights. diagnostics['counts'] gives
    the candidates per channel, diagnostics['outcomes'] 'ok', 'empty' (the
    method ran and found nothing) or 'constant', diagnostics['native'] the
    minimum, median and maximum of each native score or size column (radius,
    probability; None without candidates) and diagnostics['software'] the
    versions of starfinder, NumPy, SciPy, scikit-image and the method's
    optional dependencies. An empty result has exactly the method's declared
    columns and dtypes. The plan must not name rounds: FOV.find_spots and
    FOV.run detect a plan's rounds.
    """
    return _detect(image, config, metadata, spot_namespace, device)


def _channel_configs(plan, n_channels):
    """The effective config of every channel: the plan's config or a channel's override, with the plan's labels."""
    configs = [plan.config] * n_channels
    labels = plan.config.channel_labels
    for override in plan.channel_overrides:
        configs[labels.index(override.channel)] = replace(override.config, channel_labels=labels)
    return configs


def _noise_warning(record, label, round_name):
    reasons = []
    if record['mad'] == 0:
        reasons.append("its noise MAD is 0")
    if record['zero_fraction'] > 0.5:
        reasons.append(f"{record['zero_fraction']:.1%} of its voxels are zero")
    where = f"round {round_name!r}, " if round_name is not None else ""
    return (f"spot finding: {where}channel {label!r}: {' and '.join(reasons)}, so its median "
            f"({record['median']!r}) and MAD do not measure its noise; the threshold {record['threshold']!r} "
            "is unchanged")


def _model_record(models, keys):
    """diagnostics['model']: the first group's record; overridden channels with another model add theirs."""
    (_, record), *others = models
    record = dict(record)
    overrides = {keys[c]: other for channels, other in others for c in channels
                 if other['model'] != record['model']}
    if overrides:
        record['channel_overrides'] = overrides
    return record


def _model_artifacts(record):
    """The provenance artifacts entries of a diagnostics['model'] record, each loaded file once."""
    entries = list(record['artifacts'])
    for other in record.get('channel_overrides', {}).values():
        entries += [a for a in other['artifacts'] if a not in entries]
    return entries


def _detect(image, config, metadata, spot_namespace, device="cpu", round_name=None):
    """find_spots; round_name (from FOV.find_spots) is named in the warnings."""
    from starfinder.io._checkpoint import _jsonable, _tuples
    image = _validate_image(image)
    base = config.config if isinstance(config, SpotFindingPlan) else config
    spec = spec_for(SPOT_FINDING_METHODS, base, _WHAT, TypeError, "unsupported detection config")
    if not isinstance(metadata, ImageMetadata):
        raise TypeError("metadata must be ImageMetadata")
    if not isinstance(spot_namespace, str) or not spot_namespace.strip():
        raise ValueError("spot_namespace must be nonempty")
    check_device(device)
    plan = config if isinstance(config, SpotFindingPlan) else SpotFindingPlan(base)
    plan.__post_init__()
    if plan.rounds is not None:
        raise ValueError("find_spots detects one image, so its plan takes no rounds; FOV.find_spots and FOV.run "
                         "detect the rounds of a plan")
    n_channels = image.shape[3] if image.ndim == 4 else 1
    labels = base.channel_labels
    if labels is not None and len(labels) != n_channels:
        raise ValueError("channel_labels must match the channel axis")
    if plan.channel_overrides and labels is None:
        raise ValueError("a plan with channel overrides needs config.channel_labels")
    if plan.channel_overrides and not per_channel(spec):
        raise ValueError(f"{_WHAT} {spec.name!r} combines channels and takes no channel overrides")
    require_method(spec)
    check_shape(spec, image.shape[:3])
    configs = _channel_configs(plan, n_channels)
    overridden = {labels.index(o.channel) for o in plan.channel_overrides}
    groups = [(base, tuple(c for c in range(n_channels) if c not in overridden))]
    groups = [g for g in groups if g[1]] + [(configs[c], (c,)) for c in sorted(overridden)]
    # A constant channel of a per-channel method yields no candidates without running the method.
    constant = ({c for c in range(n_channels) if _is_constant(image[..., c] if image.ndim == 4 else image)}
                if per_channel(spec) and skips_constant(base) else set())
    tables, thresholds, noise, merged, geometry, measurements = [], {}, {}, {}, [], None
    effective, models = {}, []
    for group_config, channels in groups:
        ran = tuple(c for c in channels if c not in constant)
        details = {}
        if ran:
            table, details = spec.run(image, group_config, MethodContext(ran, device))
            check_columns(spec, table)
            tables.append(table)
            if per_channel(spec):
                thresholds.update(zip(ran, details['thresholds']))
            else:
                thresholds = dict(enumerate(details['thresholds']))
            noise.update(zip(ran, details.get('noise', ())))
            merged.update(zip(ran, details.get('merged', ())))
            effective.update(zip(ran, details.get('effective', ())))
            if 'model' in details:
                models.append((ran, details['model']))
            if 'geometry' in details:
                geometry.append(details['geometry'])
            measurements = details.get('measurements', measurements)
        still = tuple(c for c in channels if c in constant)
        if still:
            skipped = constant_details(image, group_config, still, verified=bool(ran))
            thresholds.update(zip(still, skipped['thresholds']))
            noise.update(zip(still, skipped.get('noise', ())))
            merged.update(zip(still, skipped.get('merged', ())))
            if 'geometry' in skipped:
                geometry.append(skipped['geometry'])
            if 'model' in skipped:
                models.append((still, skipped['model']))
            measurements = measurements if measurements is not None else skipped.get('measurements')
            effective.update((c, details['effective'][0]) for c in still if details.get('effective'))
    table = tables[0] if tables else empty_table(spec, base)
    if len(tables) > 1:
        table = pd.concat(tables, ignore_index=True).sort_values('channel', kind='stable').reset_index(drop=True)
    table.insert(0, 'spot_id', pd.array([str(i) for i in range(len(table))], dtype='string'))
    keys = labels if labels is not None else tuple(str(c) for c in range(n_channels))
    diagnostics = {'method': base.method, 'channel_labels': labels,
                   'thresholds': tuple(thresholds[k] for k in sorted(thresholds)), 'coordinate_units': 'voxel_index',
                   'singleton_z_policy': 'YX plane; Z=0',
                   'measurements': (measurements if measurements is not None else
                                    {'peak_intensity': 'original pixel intensity at the detected channel maximum'}
                                    if 'peak_intensity' in table else {})}
    if per_channel(spec):
        counts = table['channel'].value_counts()
        diagnostics['counts'] = {keys[c]: int(counts.get(c, 0)) for c in range(n_channels)}
        diagnostics['outcomes'] = {keys[c]: 'constant' if c in constant else
                                   'ok' if diagnostics['counts'][keys[c]] else 'empty' for c in range(n_channels)}
        native = [column for column in _NATIVE_COLUMNS if column in table]
        if native:
            diagnostics['native'] = {keys[c]: {column: _summary(table.loc[table['channel'] == c, column])
                                               for column in native} for c in range(n_channels)}
    if merged:
        diagnostics['merged'] = {keys[c]: merged[c] for c in sorted(merged)}
    if geometry:
        # Channels are detected one at a time, so the largest estimate of the channels' configs is the peak.
        diagnostics['geometry'] = max(geometry, key=lambda g: g.get('scale_space_bytes_estimate', 0))
    messages = []
    if noise:
        diagnostics['noise'] = {keys[c]: noise[c] for c in sorted(noise)}
        for c in sorted(noise):
            if noise[c]['mad'] == 0 or noise[c]['zero_fraction'] > 0.5:
                messages.append(_noise_warning(noise[c], keys[c], round_name))
                warnings.warn(messages[-1], SpotFindingWarning, stacklevel=3)
    if models:
        diagnostics['model'] = _model_record(models, keys)
    diagnostics['effective_settings'] = {keys[c]: _tuples(_jsonable(effective.get(c, configs[c])))
                                         for c in range(n_channels)}
    diagnostics['warnings'] = tuple(messages)
    diagnostics['software'] = _software(spec)
    diagnostics['execution'] = execution_record(device, framework=any(d.module == "torch" for d in spec.requires))
    return SpotFindingResult(table, metadata, spot_namespace, base, diagnostics, plan)


def _is_constant(channel):
    return channel.min() == channel.max()


def _summary(values):
    """Minimum, median and maximum of one channel's native column (None for a channel without candidates)."""
    if not len(values):
        return {'min': None, 'median': None, 'max': None}
    return {'min': float(values.min()), 'median': float(np.median(values)), 'max': float(values.max())}


def _software(spec):
    """Versions of starfinder, NumPy, SciPy, scikit-image and the method's optional dependencies."""
    import scipy
    import skimage

    import starfinder
    from starfinder._registry import _version
    versions = {'starfinder': starfinder.__version__, 'numpy': np.__version__, 'scipy': scipy.__version__,
                'scikit-image': skimage.__version__}
    versions.update({d.distribution: _version(d.distribution) for d in spec.requires})
    return versions


# Per-round diagnostics, which a result of several rounds keeps under diagnostics['rounds'][<round>].
_ROUND_KEYS = ('thresholds', 'counts', 'outcomes', 'noise', 'merged')


def _combine_rounds(results, plan):
    """One SpotFindingResult of several rounds: the rounds' tables in the given order, with a ``round`` column.

    results maps each round label (in FOV.run order, reference first) to its
    single-round result of the same plan without rounds. spot_id runs "0"..
    "N-1" over the combined table, so the first round's identities are its
    own; rows are never merged. The per-round diagnostics move under
    diagnostics['rounds']; the warnings of every round are kept, and native
    summarises every round's rows per channel.
    """
    first = next(iter(results.values()))
    frames = []
    for name, result in results.items():
        frame = result.spots.drop(columns='spot_id')
        frame['round'] = pd.array([name] * len(frame), dtype='string')
        frames.append(frame)
    table = pd.concat(frames, ignore_index=True)
    table.insert(0, 'spot_id', pd.array([str(i) for i in range(len(table))], dtype='string'))
    diagnostics = {key: value for key, value in first.diagnostics.items() if key not in _ROUND_KEYS}
    diagnostics['rounds'] = {name: {key: result.diagnostics[key] for key in _ROUND_KEYS if key in result.diagnostics}
                             for name, result in results.items()}
    diagnostics['warnings'] = tuple(m for result in results.values() for m in result.diagnostics['warnings'])
    if 'native' in diagnostics:
        keys = tuple(diagnostics['native'])
        diagnostics['native'] = {key: {column: _summary(table.loc[table['channel'] == c, column])
                                       for column in diagnostics['native'][key]} for c, key in enumerate(keys)}
    geometry = [r.diagnostics['geometry'] for r in results.values() if 'geometry' in r.diagnostics]
    if geometry:
        diagnostics['geometry'] = max(geometry, key=lambda g: g.get('scale_space_bytes_estimate', 0))
    for key in ('model', 'measurements'):
        found = [r.diagnostics[key] for r in results.values() if r.diagnostics.get(key)]
        if found:
            diagnostics[key] = found[0]
    return SpotFindingResult(table, first.metadata, first.spot_namespace, first.config, diagnostics, plan)
