"""Identity-based spot tables: the zero-based molecule and read exports, and the legacy one-based writer."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from starfinder.spot_finding import SpotFindingResult
from .decoding import ReadDecodingResult
from .filtering import ReadFilteringResult

#: The columns of molecules.csv and their dtypes: the accepted reads, zero-based ZYX.
MOLECULE_DTYPES = {"spot_namespace": "string", "spot_id": "string", "z": "float64", "y": "float64",
                   "x": "float64", "gene_id": "string", "entry_id": "string", "qc_score": "float64"}
_KEYS = ['spot_namespace', 'spot_id']


def _channel_labels(detection, reads):
    labels = detection.diagnostics.get('channel_labels') or getattr(reads, 'channel_labels', None)
    if labels is None:
        raise ValueError('the detection channel indices and the read channel labels cannot be compared without '
                         'channel labels (detection diagnostics or the read result)')
    return tuple(labels)


def _shared_columns(coordinates, detection, reads, table):
    """The columns both tables hold (direct readout: round and channel), checked to agree and left to the reads.

    A detection ``channel`` that is an index into the channel axis is compared
    through the channel labels with a read ``channel`` that is a label.
    """
    shared = [c for c in table if c in coordinates and c not in _KEYS]
    if not shared:
        return coordinates
    aligned = coordinates[_KEYS].merge(table[_KEYS + shared], on=_KEYS, how='left', validate='one_to_one', sort=False)
    for column in shared:
        mine, theirs = coordinates[column], aligned[column]
        if column == 'channel' and pd.api.types.is_integer_dtype(mine) and not pd.api.types.is_integer_dtype(theirs):
            labels = _channel_labels(detection, reads)
            index = mine.to_numpy(dtype=np.int64)
            if len(index) and (index.min() < 0 or index.max() >= len(labels)):
                raise ValueError('detection channel outside the channel labels')
            mine = pd.Series([labels[i] for i in index], index=coordinates.index, dtype='string')
        left = mine.astype('string').to_numpy(dtype=object, na_value=None)
        right = theirs.astype('string').to_numpy(dtype=object, na_value=None)
        if not all(a == b for a, b in zip(left, right)):
            first = next(i for i, (a, b) in enumerate(zip(left, right)) if a != b)
            raise ValueError(f'the detection and the reads disagree on {column!r} for spot '
                             f'{coordinates.spot_id.iloc[first]!r} ({left[first]!r} and {right[first]!r})')
    return coordinates.drop(columns=shared)


def _join_spots(detection, reads=None, *, accepted_only=False):
    if not isinstance(detection, SpotFindingResult):
        raise TypeError('detection must be SpotFindingResult')
    detection.__post_init__()
    coordinates = detection.spots.copy()
    if 'spot_namespace' in coordinates and not coordinates.spot_namespace.eq(detection.spot_namespace).all():
        raise ValueError('detection namespace column differs from result')
    coordinates['spot_namespace'] = pd.Series(detection.spot_namespace, index=coordinates.index, dtype='string')
    keys = _KEYS
    if reads is None:
        if accepted_only:
            raise ValueError('accepted_only requires filtering results')
        return coordinates
    if not isinstance(reads, (ReadDecodingResult, ReadFilteringResult)):
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
    coordinates = _shared_columns(coordinates, detection, reads, table)
    joined = coordinates.merge(table, on=keys, how='left', validate='one_to_one', sort=False)
    if accepted_only:
        if not isinstance(reads, ReadFilteringResult):
            raise ValueError('accepted_only requires filtering results')
        if 'accepted' not in joined or joined.accepted.dtype != bool or joined.accepted.isna().any():
            raise ValueError('filter acceptance must be a complete Boolean column')
        joined = joined.loc[joined.accepted].copy()
    return joined


def _molecule_frame(joined):
    """The molecules.csv columns of the accepted join; entry_id and qc_score are missing where absent."""
    n = len(joined)
    frame = {}
    for column, dtype in MOLECULE_DTYPES.items():
        if dtype == 'float64':
            frame[column] = joined[column].to_numpy(np.float64) if column in joined else np.full(n, np.nan)
        else:
            frame[column] = (joined[column].astype('string').array if column in joined
                             else pd.array([pd.NA] * n, dtype='string'))
    return pd.DataFrame(frame)


def _spot_table(detection: SpotFindingResult, reads: ReadDecodingResult | ReadFilteringResult | None, *,
               accepted_only: bool = False, columns: list[str] | None = None) -> pd.DataFrame:
    """The table export_spots writes, in the detection's row order, with zero-based ``z, y, x``."""
    joined = _join_spots(detection, reads, accepted_only=accepted_only)
    if accepted_only:
        table = _molecule_frame(joined)
    else:
        first = ['spot_namespace', 'spot_id', 'z', 'y', 'x']
        table = joined[first + [c for c in joined if c not in first]]
    if columns is not None:
        missing = [c for c in columns if c not in table]
        if missing:
            raise ValueError(f'columns {missing} are not columns of the table ({list(table.columns)})')
        table = table[list(columns)]
    return table.reset_index(drop=True)


def _write_spot_table(table, path):
    """Write a typed CSV (``_records._write_table``); return its dtype map."""
    from starfinder._records import _write_table
    path = Path(path)
    if path.suffix != '.csv':
        raise ValueError(f'the spot table is a typed CSV; the path must end in .csv, not {path.name!r}')
    path.parent.mkdir(parents=True, exist_ok=True)
    return _write_table(table, path.parent, path.stem, 'csv')[1]


def export_spots(detection: SpotFindingResult, reads: ReadDecodingResult | ReadFilteringResult | None,
                 path: Path | str, *, accepted_only: bool = False,
                 columns: list[str] | None = None) -> Path:
    """Join complete results by namespace/ID and write a typed, zero-based CSV.

    With ``accepted_only`` (a filtering result) the file is the molecule table
    (``molecules.csv``): the accepted reads with the columns ``spot_namespace,
    spot_id, z, y, x, gene_id, entry_id, qc_score`` (``entry_id`` and
    ``qc_score`` are ``<NA>`` where the reads have none, ``qc_score`` for
    unscored reads). Otherwise it is the read table (``reads.csv``): every read
    joined to its detection, ``spot_namespace, spot_id, z, y, x``, then the
    other detection columns, then the read columns; a column that both hold
    (``round`` and ``channel`` in direct readout) is written once, from the
    reads, after a check that the two agree (a detection channel index is
    compared through the channel labels). Without reads it is the detection
    table. ``columns`` selects and orders columns of that table. Coordinates
    are the detection's zero-based ``z, y, x``, written with
    ``_records._write_table`` (``<NA>`` for missing values, 17 significant
    digits); empty results write headers. Full decoding/filtering results must
    cover exactly the detection identities; reordered rows are allowed,
    missing/duplicate/foreign keys and disagreeing shared columns are errors.
    """
    table = _spot_table(detection, reads, accepted_only=accepted_only, columns=columns)
    _write_spot_table(table, path)
    return Path(path)


def _export_legacy_spots(detection: SpotFindingResult, reads: ReadFilteringResult, path: Path | str) -> Path:
    """The legacy route's one-based ``x, y, z, gene`` CSV of the accepted reads, without identities.

    Only the resident subtile and deep rules write it (``subtile_goodSpots_<n>.csv``,
    which ``stitch_subtile`` reads); the coordinates are the detection's plus one.
    """
    table = _join_spots(detection, reads, accepted_only=True).rename(columns={'gene_id': 'gene'})
    out = table[['x', 'y', 'z', 'gene']].copy()
    for axis in ('x', 'y', 'z'):
        out[axis] = out[axis] + 1
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return path
