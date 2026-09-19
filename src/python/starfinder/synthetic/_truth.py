"""Scene tables and explicit limits of historical processed-image truth."""
from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd
from starfinder.image import ImageMetadata
from ._config import SyntheticConfig

LIMITATIONS = (
    'Processed-image Gaussian scenes, not qualified molecular truth or calibrated acquisition physics; W-93.',
    'Integer sampled centers; deformation uses nearest-even rounded centers and drops out-of-bounds spots.',
    'Registration preset/deformation seeds use process-dependent hash(); cross-process reproducibility unqualified (W-93).',
)

@dataclass
class SyntheticDataset:
    """In-memory generation result, with no disk/reporting side effects.

    ``rounds[fov_id][round_label]`` preserves generation order; arrays are
    ZYXC for sequencing and ZYX for registration. ``metadata`` uses the same
    keys. ``spot_truth`` records all identities, including unrendered rows.
    ``molecular_truth`` is None: historical generators do not supply qualified
    molecular truth. ``historical_truth`` preserves their original records.
    Forward scene perturbations are not registration pull transforms.
    """
    rounds: dict[str, dict[str, np.ndarray]]
    metadata: dict[str, dict[str, ImageMetadata]]
    channel_labels: tuple[str, ...]
    spot_truth: pd.DataFrame
    molecular_truth: pd.DataFrame | None
    config: SyntheticConfig | dict[str, Any]
    provenance: dict[str, Any]
    perturbations: dict[str, Any]
    historical_truth: dict[str, Any]
    codebook: list[tuple[str, str]] | None = None


def _scene_table(spots):
    """Convert internal historical rows to the public scene schema."""
    result = pd.DataFrame(spots, columns=['z', 'y', 'x', 'intensity', 'sigma'])
    result.insert(0, 'spot_id', np.arange(len(result), dtype=np.int64))
    return result.astype({'z': float, 'y': float, 'x': float,
                          'intensity': float, 'sigma': float})


def _render_rows(spots):
    if not isinstance(spots, pd.DataFrame):
        raise TypeError('spots must be a scene DataFrame with spot_id/z/y/x/intensity/sigma')
    required = ['spot_id', 'z', 'y', 'x', 'intensity', 'sigma']
    if any(c not in spots for c in required):
        raise ValueError('scene requires spot_id/z/y/x/intensity/sigma')
    if spots.spot_id.isna().any() or spots.spot_id.duplicated().any():
        raise ValueError('scene spot_id must be non-null and unique')
    values = spots[required[1:]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values[:, 4] <= 0).any():
        raise ValueError('scene values must be finite and sigma positive')
    for z, y, x, intensity, sigma in values:
        # Integer centers keep the exact historical kernel slicing path.
        yield (int(z) if z.is_integer() else z,
               int(y) if y.is_integer() else y,
               int(x) if x.is_integer() else x, intensity, sigma)


def _truth_rows(spots, *, namespace, round_label, shape, shift=(0, 0, 0), field=None):
    """Track available continuous displaced centers before historical rounding."""
    records = []
    for spot_id, (z, y, x, intensity, sigma) in enumerate(spots):
        position = np.array((z, y, x)) + np.array(shift)
        continuous = position.astype(float)
        rendered = bool(((position >= 0) & (position < shape)).all())
        reason = None if rendered else 'outside_after_shift'
        if rendered and field is not None:
            displacement = field[tuple(position.astype(int))]
            continuous = np.array([int(v) + d for v, d in zip(position, displacement)])
            position = np.array([int(round(v)) for v in continuous])
            rendered = bool(((position >= 0) & (position < shape)).all())
            reason = None if rendered else 'outside_after_deformation_rounding'
        records.append(dict(spot_namespace=namespace, spot_id=spot_id,
            round_label=round_label, z=float(position[0]), y=float(position[1]), x=float(position[2]),
            continuous_z=float(continuous[0]), continuous_y=float(continuous[1]), continuous_x=float(continuous[2]),
            intensity=intensity, sigma=sigma, rendered=rendered, eligible=rendered,
            eligibility_reason=reason, frame_id=f'{namespace}/{round_label}', units='voxel_index',
            perturbation_direction='reference_to_moving', molecular_truth_eligible=False))
    return records


def _truth_table(records):
    columns = ['spot_namespace', 'spot_id', 'round_label', 'z', 'y', 'x',
               'continuous_z', 'continuous_y', 'continuous_x', 'intensity', 'sigma',
               'rendered', 'eligible', 'eligibility_reason', 'frame_id', 'units',
               'perturbation_direction', 'molecular_truth_eligible']
    table = pd.DataFrame(records, columns=columns + ['gene', 'barcode', 'color_seq', 'channel_label'])
    return table.astype({'spot_id': 'int64', **{c: 'float64' for c in columns[3:11]},
                         'rendered': bool, 'eligible': bool, 'molecular_truth_eligible': bool})
