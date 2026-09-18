"""Identity-based spot persistence; the sole MATLAB coordinate boundary."""
from __future__ import annotations
from pathlib import Path
import starfinder
import pandas as pd

from starfinder.spot_finding import SpotFindingResult



def _join_spots(detection, reads=None, *, accepted_only=False):
    from starfinder.barcode import BarcodeDecodingResult, ReadFilteringResult
    if not isinstance(detection, SpotFindingResult):
        raise TypeError('detection must be SpotFindingResult')
    detection.__post_init__()
    coordinates = detection.spots.copy()
    if 'spot_namespace' in coordinates and not coordinates.spot_namespace.eq(detection.spot_namespace).all():
        raise ValueError('detection namespace column differs from result')
    coordinates['spot_namespace'] = pd.Series(detection.spot_namespace, index=coordinates.index, dtype='string')
    keys = ['spot_namespace', 'spot_id']
    if reads is None:
        if accepted_only:
            raise ValueError('accepted_only requires filtering results')
        return coordinates
    if not isinstance(reads, (BarcodeDecodingResult, ReadFilteringResult)):
        raise TypeError('reads must be decoding or filtering results')
    table = reads.table
    if not table.columns.is_unique or any(k not in table for k in keys) or table[keys].isna().any().any() or table.duplicated(keys).any():
        raise ValueError('read keys must be present, nonnull and unique')
    if reads.spot_namespace != detection.spot_namespace or not table.spot_namespace.eq(reads.spot_namespace).all():
        raise ValueError('spot namespaces differ')
    if set(map(tuple, table[keys].to_numpy())) != set(map(tuple, coordinates[keys].to_numpy())):
        raise ValueError('missing or unexpected spot keys')
    if any(c in table for c in ('z', 'y', 'x')):
        raise ValueError('coordinates belong to detection results')
    joined = coordinates.merge(table, on=keys, how='left', validate='one_to_one', sort=False)
    if accepted_only:
        if not isinstance(reads, ReadFilteringResult):
            raise ValueError('accepted_only requires filtering results')
        if 'accepted' not in joined or joined.accepted.dtype != bool or joined.accepted.isna().any():
            raise ValueError('filter acceptance must be a complete Boolean column')
        joined = joined.loc[joined.accepted].copy()
    return joined


def export_spots(detection: SpotFindingResult, reads: starfinder.barcode.BarcodeDecodingResult | starfinder.barcode.ReadFilteringResult | None,
                 path: Path | str, *, accepted_only: bool = False,
                 columns: list[str] | None = None) -> Path:
    """Join complete results by namespace/ID, then optionally select accepted reads.

    Write 1-based XYZ CSV without changing source tables. Default columns are
    x,y,z,gene when reads exist, otherwise x,y,z. Empty results write headers.
    Full decoding/filtering results must cover exactly the detection identities;
    reordered rows are allowed, missing/duplicate/foreign keys are errors.
    """
    table = _join_spots(detection, reads, accepted_only=accepted_only)
    if reads is not None:
        table = table.rename(columns={'gene_id': 'gene', 'observed_color_sequence': 'color_seq'})
    columns = columns if columns is not None else ['x', 'y', 'z'] + (['gene'] if reads is not None else [])
    out = table[columns].copy()
    for axis in ('x', 'y', 'z'):
        if axis in out:
            out[axis] = out[axis] + 1
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return path
