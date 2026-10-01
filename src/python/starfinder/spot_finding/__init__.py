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

from ._config import LocalMaximaConfig, NoiseLandmarkConfig, PercentileCentroidConfig, StarfishLogConfig
from ._errors import MissingWeightsError, SpotFindingBackendUnavailableError, SpotFindingWarning, WeightsHashMismatchError
from ._methods import SPOT_FINDING_METHODS as _SPOT_FINDING_METHODS
from ._methods import MethodContext, SpotFindingConfig, SpotFindingSpec, check_columns, check_shape, per_channel, require_method
from ._plan import ChannelOverride, SpotFindingPlan
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
           "SpotFindingResult", "find_spots", "SPOT_FINDING_METHODS", "SpotFindingSpec",
           "SpotFindingPlan", "ChannelOverride", "SpotFindingBackendUnavailableError", "SpotFindingWarning",
           "KNOWN_WEIGHTS", "KnownWeights", "WeightsFile", "resolve_weights", "fetch_weights",
           "MissingWeightsError", "WeightsHashMismatchError"]

_WHAT = "spot-finding method"


@dataclass(frozen=True)
class SpotFindingResult:
    """Typed spot table, source geometry, identity scope and effective policy.

    Required columns: spot_id (pandas string), z/y/x (float64). Optional channel
    (int64) indexes diagnostics['channel_labels']; peak_intensity (float64)
    means the original sampled pixel value. No universal detection score is
    invented. Namespace must include dataset/sample/FOV/subtile when applicable.
    Consumers must join on namespace and spot_id, never row position.
    config is the registered config of the method (a plan's base config; the
    per-channel configs are in diagnostics['effective_settings']).
    The repr is a one-line spot count and column names without table rows.
    """
    spots: pd.DataFrame
    metadata: ImageMetadata
    spot_namespace: str
    config: SpotFindingConfig
    diagnostics: dict

    def __post_init__(self):
        if not isinstance(self.metadata, ImageMetadata):
            raise TypeError("metadata must be ImageMetadata")
        if not isinstance(self.spot_namespace, str) or not self.spot_namespace.strip():
            raise ValueError("spot_namespace must be nonempty")
        spec_for(SPOT_FINDING_METHODS, self.config, _WHAT, TypeError, "unsupported detection config")
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
    num_sigma x voxels). diagnostics['effective_settings'] holds every
    channel's effective config, and diagnostics['execution'] the execution
    entry (device, framework, threads).
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
    tables, thresholds, noise, merged, geometry, measurements = [], {}, {}, {}, [], None
    for group_config, channels in groups:
        table, details = spec.run(image, group_config, MethodContext(channels, device))
        check_columns(spec, table)
        tables.append(table)
        if per_channel(spec):
            thresholds.update(zip(channels, details['thresholds']))
        else:
            thresholds = dict(enumerate(details['thresholds']))
        noise.update(zip(channels, details.get('noise', ())))
        merged.update(zip(channels, details.get('merged', ())))
        if 'geometry' in details:
            geometry.append(details['geometry'])
        measurements = details.get('measurements', measurements)
    table = tables[0]
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
    diagnostics['effective_settings'] = {keys[c]: _tuples(_jsonable(configs[c])) for c in range(n_channels)}
    diagnostics['warnings'] = tuple(messages)
    diagnostics['execution'] = execution_record(device, framework=any(d.module == "torch" for d in spec.requires))
    return SpotFindingResult(table, metadata, spot_namespace, base, diagnostics)
