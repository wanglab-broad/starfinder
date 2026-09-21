"""Lossless pre-rejection Parquet candidates/signals (starfinder.artifacts/1)."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import uuid
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from starfinder.io.checkpoints import _encode, _decode, _validate_context
from starfinder.provenance import _component, _hash, _header, _is_hash, _record

if TYPE_CHECKING:
    from starfinder.barcode import Codebook, IntensityExtractionResult
    from starfinder.spot_finding import SpotFindingResult


@dataclass(frozen=True)
class CandidateCheckpoint:
    """Complete pre-rejection results and source context, validated on load.

    ``spots`` and ``intensities`` are existing SpotFindingResult and
    IntensityExtractionResult objects. No detection, extraction or decoding is
    performed by loading. ``codebook`` is optional until decoding is requested.
    """

    artifact: dict
    spots: SpotFindingResult
    intensities: IntensityExtractionResult
    codebook: Codebook | None

    def source_trace(self, *, run_id: str, candidate_artifact_id: str,
                     spot_namespace: str, spot_id: str) -> dict:
        """Resolve an exact artifact/identity locator; never select a latest FOV.

        Unknown or mismatched identities raise ValueError. Returned numerical
        values are copies, with C×R signals and R validity, including invalid
        measurements' stored finite values.
        """
        a, result = self.artifact, self.intensities
        _require((run_id, candidate_artifact_id, spot_namespace) ==
                 (a['run_id'], a['artifact_id'], result.spot_namespace), 'source locator mismatch')
        _require(spot_id in result.spot_ids, 'unknown source spot_id')
        i = result.spot_ids.index(spot_id)
        return dict(candidate=self.spots.spots.iloc[i].copy(), values=result.values[i].copy(),
                    valid=result.valid[i].copy(), metadata=result.metadata,
                    round_labels=result.round_labels, channel_labels=result.channel_labels,
                    codebook=self.codebook, context=a['payload']['context'])


@dataclass(frozen=True)
class CandidateSaveResult:
    """Actual persisted bytes, or an explicit disabled/no-destination reason."""

    path: Path | None
    size_bytes: int
    reason: str | None = None


def _require(condition, message):
    if not condition:
        raise ValueError(f'candidate checkpoint: {message}')


def _backend():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError('Parquet checkpoints require the starfinder[checkpoint] extra (pyarrow)') from exc
    return pa, pq


def _pack(value):
    return _encode(value, None, '', '', [])


def _unpack(value):
    # Metadata cannot reference arrays/HDF5 datasets in a table checkpoint.
    if isinstance(value, dict):
        _require('array' not in value, 'array metadata is not supported')
        for item in value.values():
            _check_metadata(item)
    return _decode(value, None, '', '', set())


def _check_metadata(value):
    if isinstance(value, dict):
        _require('array' not in value, 'array metadata is not supported')
        for item in value.values():
            _check_metadata(item)
    elif isinstance(value, list):
        for item in value:
            _check_metadata(item)


def _schema(table):
    _require(table.columns.is_unique and all(isinstance(c, str) for c in table.columns), 'unique string columns required')
    result = []
    for name in table:
        dtype = table[name].dtype
        # Explicit scalar tabular types only; no pickle, nested/object inference.
        allowed = (isinstance(dtype, pd.StringDtype) or
                   str(dtype) in {'bool', 'boolean', 'Int8', 'Int16', 'Int32', 'Int64',
                                  'UInt8', 'UInt16', 'UInt32', 'UInt64', 'Float32', 'Float64'} or
                   isinstance(dtype, np.dtype) and
                   (dtype.kind in 'biu' and dtype.itemsize <= 8 or dtype.kind == 'f' and dtype.itemsize in (4, 8)))
        _require(allowed, f'unsupported column dtype {name}: {dtype}')
        result.append(dict(name=name, dtype=str(dtype),
                           storage=dtype.storage if isinstance(dtype, pd.StringDtype) else None,
                           nullable=bool(table[name].isna().any()) or isinstance(dtype, pd.api.extensions.ExtensionDtype)))
    return result


def _write_table(root, name, table, artifact_id):
    pa, pq = _backend()
    schema = _schema(table)
    arrow = pa.Table.from_pandas(table, preserve_index=False)
    metadata = dict(arrow.schema.metadata or {})
    metadata[b'starfinder.artifact_id'] = artifact_id.encode()
    metadata[b'starfinder.component'] = name.encode()
    arrow = arrow.replace_schema_metadata(metadata)
    path = root / f'{name}.parquet'
    pq.write_table(arrow, path, compression='zstd', row_group_size=65536)
    return dict(component_id=name, path=path.name, format='parquet',
                size=path.stat().st_size, sha256=_hash(path)), schema


def _read_table(root, descriptor, schema, artifact_id):
    pa, pq = _backend()
    path = _component(root, descriptor)
    _require(descriptor['format'] == 'parquet', 'component format')
    # ParquetFile does not infer Hive partition columns from parent directories.
    file = pq.ParquetFile(path)
    for i in range(file.metadata.num_row_groups):
        group = file.metadata.row_group(i)
        _require(group.num_rows <= 65536, 'row group size')
        _require(all(group.column(j).compression == 'ZSTD' for j in range(group.num_columns)), 'codec')
    arrow = file.read(use_threads=False)
    metadata = arrow.schema.metadata or {}
    _require(metadata.get(b'starfinder.artifact_id') == artifact_id.encode() and
             metadata.get(b'starfinder.component') == descriptor['component_id'].encode(), 'component binding')
    _require(arrow.column_names == [s['name'] for s in schema], 'column schema mismatch')
    # Restore pandas nullable types explicitly, while checking physical widths
    # before any conversion can conceal malformed input.
    result = arrow.to_pandas()
    for s in schema:
        dtype = pd.StringDtype(storage=s['storage']) if s['storage'] else pd.api.types.pandas_dtype(s['dtype'])
        expected = pa.Schema.from_pandas(pd.DataFrame({s['name']: pd.Series([], dtype=dtype)}), preserve_index=False).field(0).type
        _require(arrow.schema.field(s['name']).type == expected, f"physical dtype {s['name']}")
        _require(s['nullable'] or not result[s['name']].isna().any(), f"null {s['name']}")
        result[s['name']] = result[s['name']].astype(dtype)
    _require(_schema(result) == schema, 'declared pandas schema mismatch')
    return result


def _validate(spots, intensities, codebook):
    from starfinder.barcode import Codebook, IntensityExtractionResult
    from starfinder.spot_finding import SpotFindingResult
    _require(isinstance(spots, SpotFindingResult) and isinstance(intensities, IntensityExtractionResult), 'result types')
    spots.__post_init__()
    intensities.__post_init__()
    _require(spots.spot_namespace == intensities.spot_namespace and spots.metadata == intensities.metadata,
             'namespace/geometry mismatch')
    _require(set(spots.spots.spot_id) == set(intensities.spot_ids), 'candidate/signal identity mismatch')
    _require(not {'candidate_index', 'spot_namespace'} & set(spots.spots.columns), 'reserved candidate columns')
    _schema(spots.spots)
    if codebook is not None:
        _require(isinstance(codebook, Codebook), 'codebook type')
        codebook.__post_init__()
        _require(codebook.round_labels == intensities.round_labels and codebook.channel_labels == intensities.channel_labels,
                 'codebook acquisition mismatch')
        _require('codebook_index' not in codebook.table, 'reserved codebook column')
        _schema(codebook.table)


def save_candidate_checkpoint(directory: str | Path, spots, intensities, *,
                              dataset_id: str, sample_id: str, FOV: str, run_id: str,
                              config: dict, code: dict, codebook=None, sources=(), parents=(),
                              provenance=None, subtile: int | None = None,
                              enabled: bool = True) -> CandidateSaveResult:
    """Save the complete pre-decoding/QC population in a fresh directory.

    Saving defaults on; enabled=False returns an explicit omission and writes
    nothing (no backend required). Size is actual component plus manifest bytes.
    Candidates, long NCR signals, NR validity and an optional codebook are
    checksummed Parquet components. The complete manifest is published last.
    Existing directories are never overwritten. Input row order may differ:
    signals join by identity; candidates define restored order.
    """
    _require(type(enabled) is bool, 'enabled must be Boolean')
    if not enabled:
        return CandidateSaveResult(None, 0, 'candidates_signals_disabled')
    _validate(spots, intensities, codebook)
    source_ids = _validate_context(config, code, sources, parents, provenance)
    pa, _ = _backend()
    context = dict(config=config, code=code, sources=sources, provenance=provenance)
    metadata = _pack(dict(metadata=spots.metadata, detector_config=spots.config,
                         detector_diagnostics=spots.diagnostics, extraction_config=intensities.config,
                         extraction_diagnostics=intensities.diagnostics,
                         encoding=codebook.encoding if codebook is not None else None))
    candidates = spots.spots.reset_index(drop=True).copy()
    ids = tuple(candidates.spot_id)
    n, c, r = len(ids), len(intensities.channel_labels), len(intensities.round_labels)
    candidates['candidate_index'] = np.arange(n, dtype=np.int64)
    candidates['spot_namespace'] = pd.Series([spots.spot_namespace] * n, dtype='string')
    ni, ci, ri = np.indices((n, c, r)).reshape(3, -1)
    identity_index = {identity: i for i, identity in enumerate(intensities.spot_ids)}
    order = [identity_index[x] for x in ids]
    signals = pd.DataFrame(dict(spot_namespace=pd.Series([spots.spot_namespace] * len(ni), dtype='string'),
        spot_id=pd.Series([ids[i] for i in ni], dtype='string'), channel_index=ci, round_index=ri,
        value=intensities.values[order].reshape(-1)))
    ni, ri = np.indices((n, r)).reshape(2, -1)
    validity = pd.DataFrame(dict(spot_namespace=pd.Series([spots.spot_namespace] * len(ni), dtype='string'),
        spot_id=pd.Series([ids[i] for i in ni], dtype='string'), round_index=ri,
        valid=intensities.valid[order].reshape(-1)))
    tables = dict(candidates=candidates, signals=signals, validity=validity)
    if codebook is not None:
        tables['codebook'] = codebook.table.reset_index(drop=True).copy()
        tables['codebook']['codebook_index'] = np.arange(len(codebook.table), dtype=np.int64)
    artifact_id = uuid.uuid4().hex
    artifact = dict(_header('artifact'), artifact_id=artifact_id, run_id=run_id,
        stage='candidates_signals', status='complete', dataset_id=dataset_id, sample_id=sample_id,
        FOV=FOV, subtile=subtile, parents=list(parents), config_ref='payload/context/config',
        source_refs=source_ids, components=[{}], payload={}, omission_reason=None, failure_id=None)
    _record(artifact, 'artifact')
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    components, schemas = [], {}
    for name, table in tables.items():
        descriptor, schemas[name] = _write_table(root, name, table, artifact_id)
        components.append(descriptor)
    artifact['components'] = components
    artifact['payload'] = dict(metadata=metadata, context={k: _pack(v) for k, v in context.items()},
        schemas=schemas, shape=[n, c, r], axes='NCR', validity_axes='NR',
        row_counts={name: len(table) for name, table in tables.items()},
        spot_namespace=spots.spot_namespace, channel_labels=list(intensities.channel_labels),
        round_labels=list(intensities.round_labels), color_to_channel=codebook.color_to_channel if codebook else None,
        storage=dict(engine='pyarrow', version=pa.__version__, compression='zstd', row_group_size=65536))
    path = root / 'artifact.json'
    temporary = root / 'artifact.json.tmp'
    temporary.write_text(json.dumps(artifact, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)
    return CandidateSaveResult(path, path.stat().st_size + sum(x['size'] for x in components))


def load_candidate_checkpoint(path: str | Path, *, sha256: str | None = None) -> CandidateCheckpoint:
    """Verify all components and reconstruct a whole FOV by stable identity.

    Missing/duplicate/extra identities, rows or indices, nonfinite signals,
    unsupported schema and corrupted components fail without partial success.
    An optional trusted manifest checksum also pins metadata and source lookup.
    """
    from starfinder.barcode import Codebook, IntensityExtractionResult
    from starfinder.spot_finding import SpotFindingResult
    path = Path(path)
    if path.is_dir():
        path /= 'artifact.json'
    if sha256 is not None:
        _require(_is_hash(sha256) and _hash(path) == sha256, 'manifest checksum mismatch')
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    def constant(value):
        raise ValueError(f'candidate checkpoint: nonfinite JSON {value}')
    try:
        a = json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=pairs, parse_constant=constant)
        _record(a, 'artifact')
        _require(a['stage'] == 'candidates_signals' and a['status'] == 'complete', 'stage/status')
        _require(a['config_ref'] == 'payload/context/config', 'config reference')
        p = a['payload']
        _require(set(p) == {'metadata', 'context', 'schemas', 'row_counts', 'shape', 'axes', 'validity_axes', 'spot_namespace',
                           'channel_labels', 'round_labels', 'color_to_channel', 'storage'}, 'payload fields')
        _require(p['axes'] == 'NCR' and p['validity_axes'] == 'NR', 'tensor axes')
        _require(isinstance(p['shape'], list) and len(p['shape']) == 3 and
                 all(type(x) is int and x >= 0 for x in p['shape']), 'dimensions')
        n, c, r = p['shape']
        _require(c > 0 and r > 0 and c == len(p['channel_labels']) and r == len(p['round_labels']), 'label dimensions')
        s = p['storage']
        _require(s['engine'] == 'pyarrow' and isinstance(s['version'], str) and s['compression'] == 'zstd'
                 and s['row_group_size'] == 65536, 'storage settings')
        required = {'candidates', 'signals', 'validity'} | ({'codebook'} if p['color_to_channel'] is not None else set())
        names = [d['component_id'] for d in a['components']]
        _require(len(names) == len(required) and set(names) == required and set(p['schemas']) == required, 'component inventory')
        tables = {d['component_id']: _read_table(path.parent, d, p['schemas'][d['component_id']], a['artifact_id']) for d in a['components']}
        _require(set(p['row_counts']) == required and all(type(p['row_counts'][name]) is int and
                 p['row_counts'][name] == len(table) for name, table in tables.items()), 'component row counts')
        meta = _unpack(p['metadata'])
        _require(set(p['context']) == {'config', 'code', 'sources', 'provenance'}, 'context fields')
        context = {k: _unpack(v) for k, v in p['context'].items()}
        source_ids = _validate_context(context['config'], context['code'], context['sources'], a['parents'], context['provenance'])
        _require(source_ids == a['source_refs'], 'source references')
        candidates = tables['candidates']
        _require(candidates.candidate_index.dtype == np.int64 and len(candidates) == n and
                 sorted(candidates.candidate_index) == list(range(n)), 'candidate indices')
        candidates = candidates.sort_values('candidate_index').reset_index(drop=True)
        _require(isinstance(candidates.spot_namespace.dtype, pd.StringDtype) and
                 not candidates.spot_namespace.isna().any(), 'candidate namespace dtype/null')
        _require(candidates.spot_namespace.eq(p['spot_namespace']).all(), 'candidate namespace')
        spots = SpotFindingResult(candidates.drop(columns=['candidate_index', 'spot_namespace']), meta['metadata'],
                                  p['spot_namespace'], meta['detector_config'], meta['detector_diagnostics'])
        ids = tuple(spots.spots.spot_id)
        lookup = {identity: i for i, identity in enumerate(ids)}
        values = np.empty((n, c, r), dtype=np.float64)
        valid = np.empty((n, r), dtype=bool)
        for name, dims, target, column in [('signals', (c, r), values, 'value'), ('validity', (r,), valid, 'valid')]:
            table = tables[name]
            indices = ['channel_index', 'round_index'] if name == 'signals' else ['round_index']
            _require(set(table) == {'spot_namespace', 'spot_id', column, *indices}, f'{name} columns')
            _require(isinstance(table.spot_id.dtype, pd.StringDtype) and isinstance(table.spot_namespace.dtype, pd.StringDtype), f'{name} ID dtype')
            _require(not table.isna().any().any() and table.spot_namespace.eq(p['spot_namespace']).all(), f'{name} null/namespace')
            _require(len(table) == n * int(np.prod(dims)) and table.spot_id.isin(ids).all(), f'{name} cardinality/identity')
            _require(not table.duplicated(['spot_id', *indices]).any(), f'{name} duplicate keys')
            for index, size in zip(indices, dims):
                _require(table[index].dtype == np.int64 and table[index].between(0, size - 1).all(), f'{name} index range/type')
            _require(table[column].dtype == target.dtype, f'{name} value dtype')
            if name == 'signals':
                _require(np.isfinite(table[column]).all(), 'nonfinite signal')
            rows = np.array([lookup[x] for x in table.spot_id], dtype=np.int64)
            target[(rows, *(table[index].to_numpy() for index in indices))] = table[column].to_numpy()
        intensities = IntensityExtractionResult(values, ids, spots.spot_namespace, tuple(p['channel_labels']),
            tuple(p['round_labels']), spots.metadata, meta['extraction_config'], valid, meta['extraction_diagnostics'])
        codebook = None
        if 'codebook' in tables:
            table = tables['codebook']
            _require(table.codebook_index.dtype == np.int64 and sorted(table.codebook_index) == list(range(len(table))), 'codebook indices')
            codebook = Codebook(table.sort_values('codebook_index').drop(columns='codebook_index').reset_index(drop=True),
                                intensities.round_labels, intensities.channel_labels, p['color_to_channel'], meta['encoding'])
        _validate(spots, intensities, codebook)
        # In-memory payload exposes decoded context for source trace consumers.
        a['payload']['context'] = context
        return CandidateCheckpoint(a, spots, intensities, codebook)
    except (KeyError, TypeError, AttributeError, OSError) as exc:
        if isinstance(exc, FileNotFoundError):
            raise
        raise ValueError(f'candidate checkpoint: malformed/corrupt artifact: {exc}') from exc
