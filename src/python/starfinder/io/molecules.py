"""Decoded/QC artifacts and ordered sample access, without spatial assembly."""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
from functools import wraps
import json
from pathlib import Path
from typing import TYPE_CHECKING
import uuid

import numpy as np
import pandas as pd

from starfinder.io.candidates import (
    _backend, _read_table, _write_table, _schema, load_candidate_checkpoint,
)
from starfinder.io.checkpoints import _types, _validate_context
from starfinder.provenance import _component, _hash, _header, _is_hash, _record

if TYPE_CHECKING:
    from starfinder.barcode import BarcodeDecodingResult, Codebook, ReadFilteringResult
    from starfinder.spot_finding import SpotFindingResult


def _require(condition, message):
    if not condition:
        raise ValueError(f'molecular checkpoint: {message}')


def _json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    def constant(value):
        raise ValueError(f'molecular checkpoint: nonfinite JSON {value}')
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=pairs, parse_constant=constant)


def _reader(function):
    @wraps(function)
    def read(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except (KeyError, TypeError, AttributeError, OSError, IndexError, EOFError) as exc:
            if isinstance(exc, FileNotFoundError):
                raise
            raise ValueError(f'molecular checkpoint: malformed/corrupt artifact: {exc}') from exc
    return read


@dataclass(frozen=True)
class ArtifactReference:
    """Exact manifest identity. Paths are explicit source locators, not components."""

    path: Path
    sha256: str
    run_id: str
    artifact_id: str

    def __post_init__(self):
        object.__setattr__(self, 'path', Path(self.path).absolute())
        _require(_is_hash(self.sha256) and all(isinstance(x, str) and x for x in
                 (self.run_id, self.artifact_id)), 'reference identity/checksum')


@_reader
def checkpoint_reference(path: str | Path) -> ArtifactReference:
    """Pin a completed artifact manifest by stored-byte SHA-256 and identity."""
    path = Path(path)
    if path.is_dir():
        path /= 'artifact.json'
    a = _json(path)
    _record(a, 'artifact')
    _require(a['status'] == 'complete', 'reference requires complete artifact')
    return ArtifactReference(path, _hash(path), a['run_id'], a['artifact_id'])


def _reference_dict(ref):
    return dict(path=str(ref.path), sha256=ref.sha256, run_id=ref.run_id, artifact_id=ref.artifact_id)


def _reference(value):
    _require(isinstance(value, dict) and set(value) == {'path', 'sha256', 'run_id', 'artifact_id'}, 'reference fields')
    return ArtifactReference(**value)


def _resolve(ref, loader):
    _require(isinstance(ref, ArtifactReference), 'ArtifactReference required')
    result = loader(ref.path, sha256=ref.sha256)
    _require((result.artifact['run_id'], result.artifact['artifact_id']) ==
             (ref.run_id, ref.artifact_id), 'reference identity mismatch')
    return result


class _Codec:
    """Whitelisted typed metadata, scalar tables and non-pickled diagnostics."""

    def __init__(self, root, artifact_id, components=()):
        self.root, self.artifact_id = root, artifact_id
        self.components = list(components)
        self.used = set()

    def encode(self, value):
        if isinstance(value, (pd.DataFrame, np.ndarray)):
            name = f'payload{len(self.components):04d}'
            if isinstance(value, pd.DataFrame):
                descriptor, schema = _write_table(self.root, name, value, self.artifact_id)
                record = dict(table=name, schema=schema)
            else:
                _require(value.dtype.kind in 'biuf', 'diagnostic array dtype')
                path = self.root / f'{name}.npy'
                with path.open('xb') as handle:
                    np.save(handle, value, allow_pickle=False)
                descriptor = dict(component_id=name, path=path.name, format='npy',
                    size=path.stat().st_size, sha256=_hash(path))
                record = dict(array=name, shape=list(value.shape), dtype=value.dtype.str)
            self.components.append(descriptor)
            return record
        if is_dataclass(value):
            _require(type(value).__name__ in _types(), 'unsupported typed metadata')
            return dict(type=type(value).__name__, fields={f.name: self.encode(getattr(value, f.name)) for f in fields(value)})
        if isinstance(value, dict):
            _require(all(isinstance(k, str) for k in value), 'metadata keys')
            return {'mapping': {k: self.encode(v) for k, v in value.items()}}
        if isinstance(value, tuple):
            return {'tuple': [self.encode(v) for v in value]}
        if isinstance(value, list):
            return [self.encode(v) for v in value]
        if isinstance(value, Path):
            return {'path': str(value)}
        if isinstance(value, np.generic):
            return self.encode(value.item())
        if isinstance(value, float) and not np.isfinite(value):
            return {'float_special': 'nan' if np.isnan(value) else '+inf' if value > 0 else '-inf'}
        _require(value is None or type(value) in (str, bool, int, float), 'unsupported metadata')
        return value

    def decode(self, value):
        if isinstance(value, list):
            return [self.decode(v) for v in value]
        if not isinstance(value, dict):
            _require(value is None or type(value) in (str, bool, int, float), 'metadata scalar')
            _require(not isinstance(value, float) or np.isfinite(value), 'nonfinite scalar')
            return value
        keys = set(value)
        if keys == {'mapping'}:
            return {k: self.decode(v) for k, v in value['mapping'].items()}
        if keys == {'tuple'}:
            return tuple(self.decode(v) for v in value['tuple'])
        if keys == {'path'}:
            return Path(value['path'])
        if keys == {'float_special'}:
            _require(value['float_special'] in ('nan', '+inf', '-inf'), 'special float tag')
            return float(value['float_special'])
        if keys in ({'table', 'schema'}, {'array', 'shape', 'dtype'}):
            name = value.get('table', value.get('array'))
            matches = [d for d in self.components if d['component_id'] == name]
            _require(len(matches) == 1 and name not in self.used, 'component reference')
            self.used.add(name)
            descriptor = matches[0]
            if 'table' in value:
                return _read_table(self.root, descriptor, value['schema'], self.artifact_id)
            path = _component(self.root, descriptor)
            _require(descriptor['format'] == 'npy', 'array format')
            array = np.load(path, allow_pickle=False)
            _require(array.dtype.kind in 'biuf' and array.dtype.str == value['dtype'] and
                     list(array.shape) == value['shape'], 'array schema')
            return array
        _require(keys == {'type', 'fields'} and value['type'] in _types(), 'typed metadata')
        cls = _types()[value['type']]
        _require(set(value['fields']) == {f.name for f in fields(cls)}, 'typed metadata fields')
        decoded = {k: self.decode(v) for k, v in value['fields'].items()}
        result = cls(**{f.name: decoded[f.name] for f in fields(cls) if f.init})
        _require(all(getattr(result, f.name) == decoded[f.name] for f in fields(cls) if not f.init), 'config discriminator')
        return result


def _equal_table(left, right, message):
    try:
        pd.testing.assert_frame_equal(left.reset_index(drop=True), right.reset_index(drop=True), check_exact=True)
    except AssertionError as exc:
        raise ValueError(f'molecular checkpoint: {message}') from exc


def _ordered(table, ids):
    _require(table.spot_id.is_unique and set(table.spot_id) == set(ids), 'population identity mismatch')
    return table.set_index('spot_id', drop=False).loc[list(ids)].reset_index(drop=True)


def _indexed(table):
    _require('candidate_index' not in table, 'reserved candidate_index')
    result = table.reset_index(drop=True).copy()
    result['candidate_index'] = np.arange(len(result), dtype=np.int64)
    return result


def _restore(table):
    _require(table.candidate_index.dtype == np.int64 and
             sorted(table.candidate_index) == list(range(len(table))), 'candidate indices')
    return table.sort_values('candidate_index').drop(columns='candidate_index').reset_index(drop=True)


def _write(directory, stage, payload, *, dataset_id, sample_id, FOV, run_id,
           config, code, sources=(), parents=(), provenance=None, subtile=None):
    source_ids = _validate_context(config, code, sources, parents, provenance)
    pa, _ = _backend()
    a = dict(_header('artifact'), artifact_id=uuid.uuid4().hex, run_id=run_id,
        stage=stage, status='complete', dataset_id=dataset_id, sample_id=sample_id, FOV=FOV,
        subtile=subtile, parents=list(parents), config_ref='payload/context/config',
        source_refs=source_ids, components=[{}], payload={}, omission_reason=None, failure_id=None)
    _record(a, 'artifact')
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    codec = _Codec(root, a['artifact_id'])
    payload = dict(payload, context=dict(config=config, code=code, sources=sources, provenance=provenance),
                   storage=dict(engine='pyarrow', version=pa.__version__, compression='zstd', row_group_size=65536))
    a['payload'] = {k: codec.encode(v) for k, v in payload.items()}
    a['components'] = codec.components
    path = root/'artifact.json'
    # Full-file publication last, including on network artifact storage.
    path.write_text(json.dumps(a, indent=2, allow_nan=False), encoding='utf-8')
    return path


def _read(path, stage, sha256):
    path = Path(path)
    if path.is_dir():
        path /= 'artifact.json'
    if sha256 is not None:
        _require(_is_hash(sha256) and _hash(path) == sha256, 'manifest checksum mismatch')
    a = _json(path)
    _record(a, 'artifact')
    _require(a['stage'] == stage and a['status'] == 'complete', 'stage/status mismatch')
    _require(a['config_ref'] == 'payload/context/config', 'config reference')
    codec = _Codec(path.parent, a['artifact_id'], a['components'])
    payload = {k: codec.decode(v) for k, v in a['payload'].items()}
    _require(len(codec.used) == len(a['components']), 'unreferenced/duplicate components')
    context = payload['context']
    ids = _validate_context(**context, parents=a['parents'])
    _require(ids == a['source_refs'], 'source references')
    storage = payload['storage']
    _require(storage['engine'] == 'pyarrow' and isinstance(storage['version'], str) and
             storage['compression'] == 'zstd' and storage['row_group_size'] == 65536, 'storage settings')
    a['payload'] = payload
    return a


def _check_source(spots, decoded, codebook, source):
    source = _resolve(source, load_candidate_checkpoint)
    _require(source.spots.spot_namespace == spots.spot_namespace and source.spots.metadata == spots.metadata,
             'source namespace/geometry mismatch')
    _equal_table(_ordered(source.spots.spots, tuple(spots.spots.spot_id)), spots.spots, 'source candidates mismatch')
    _require(source.intensities.channel_labels == decoded.channel_labels and
             source.intensities.round_labels == decoded.round_labels, 'source acquisition mismatch')
    if source.codebook is not None:
        _equal_table(source.codebook.table, codebook.table, 'source codebook mismatch')
        _require(source.codebook.color_to_channel == codebook.color_to_channel and
                 source.codebook.encoding == codebook.encoding, 'source codebook mapping mismatch')
    return source


@dataclass(frozen=True)
class DecodedCheckpoint:
    """Whole-FOV pre-QC result with candidate coordinates and a codebook snapshot."""

    artifact: dict
    spots: SpotFindingResult
    decoded: BarcodeDecodingResult
    codebook: Codebook
    candidate_source: ArtifactReference | None
    trace_unavailable_reason: str | None

    def source_trace(self, *, spot_namespace: str, spot_id: str) -> dict:
        """Read the exact source checkpoint; omitted traces report a reason."""
        _require(spot_namespace == self.spots.spot_namespace and spot_id in set(self.spots.spots.spot_id),
                 'unknown source identity')
        if self.candidate_source is None:
            return dict(available=False, reason=self.trace_unavailable_reason)
        saved = _check_source(self.spots, self.decoded, self.codebook, self.candidate_source)
        trace = saved.source_trace(run_id=self.candidate_source.run_id,
            candidate_artifact_id=self.candidate_source.artifact_id, spot_namespace=spot_namespace, spot_id=spot_id)
        return dict(available=True, reason=None, **trace)

    def comparison_metadata(self) -> dict:
        """Explicit backend-neutral axis/index/unit/codebook mapping, no conversion."""
        return dict(coordinate_axes='ZYX', coordinate_index_base=0, coordinate_unit='voxel_index',
            metadata=self.spots.metadata, signal_axes='NCR', trace_axes='CR', validity_axes='NR',
            channel_index_base=0, round_index_base=0, round_labels=self.decoded.round_labels,
            channel_labels=self.decoded.channel_labels, color_to_channel=dict(self.codebook.color_to_channel),
            codebook=self.codebook.table.copy(), encoding=self.codebook.encoding,
            transform_policy='source/run links retain actual transforms; no assembly transform applied')


def save_decoded_checkpoint(directory: str | Path, spots, decoded, codebook, *,
                            candidate_source: ArtifactReference | None = None,
                            trace_unavailable_reason: str | None = None, links: dict | None = None,
                            **context) -> Path:
    """Persist every pre-QC call, coordinates, codebook, configs and diagnostics.

    Supply a pinned candidate_source or a nonempty trace_unavailable_reason.
    links preserves explicit assignment/H5AD/registered-image references without
    interpreting their scientific content. Coordinates retain the supplied FOV
    frame; no assembly or assignment is performed. Context matches the candidate
    writer (dataset_id, sample_id, FOV, run_id, config, code and optional sources,
    parents, provenance, subtile). Returns the complete manifest path.
    """
    from starfinder.barcode import BarcodeDecodingResult, Codebook
    from starfinder.spot_finding import SpotFindingResult
    _require(isinstance(spots, SpotFindingResult) and isinstance(decoded, BarcodeDecodingResult)
             and isinstance(codebook, Codebook), 'decoded input types')
    spots.__post_init__()
    decoded.__post_init__()
    codebook.__post_init__()
    decoded.config.__post_init__()
    _require(spots.spot_namespace == decoded.spot_namespace and
             decoded.channel_labels == codebook.channel_labels and decoded.round_labels == codebook.round_labels,
             'decoded namespace/acquisition mismatch')
    _require(set(spots.spots.spot_id) == set(decoded.table.spot_id), 'pre-QC population mismatch')
    _schema(spots.spots)
    _schema(decoded.table)
    _schema(codebook.table)
    _require((candidate_source is not None and trace_unavailable_reason is None) or
             (candidate_source is None and isinstance(trace_unavailable_reason, str) and trace_unavailable_reason),
             'source or explicit unavailable reason required')
    _require(links is None or isinstance(links, dict), 'links must be a mapping')
    if candidate_source is not None:
        source = _check_source(spots, decoded, codebook, candidate_source)
        _require(all(source.artifact[k] == context.get(k) for k in ('dataset_id', 'sample_id', 'FOV')) and
                 source.artifact['subtile'] == context.get('subtile'), 'source FOV identity mismatch')
    # Candidate order is authoritative; array diagnostics share the decoder N axis.
    ids = tuple(spots.spots.spot_id)
    order = [tuple(decoded.table.spot_id).index(s) for s in ids]
    diagnostics = dict(decoded.diagnostics)
    for key, shape in [('probabilities', (len(ids), len(decoded.channel_labels), len(decoded.round_labels))),
                       ('wta_round_l2_nll', (len(ids), len(decoded.round_labels)))]:
        if key in diagnostics:
            _require(isinstance(diagnostics[key], np.ndarray) and diagnostics[key].shape == shape,
                     f'{key} axes/shape')
            diagnostics[key] = diagnostics[key][order]
    decoded = replace(decoded, table=_ordered(decoded.table, ids), diagnostics=diagnostics)
    payload = dict(calls=_indexed(decoded.table), spots=_indexed(spots.spots), metadata=spots.metadata,
        namespace=spots.spot_namespace, detector_config=spots.config, detector_diagnostics=spots.diagnostics,
        decoder_config=decoded.config, diagnostics=decoded.diagnostics, codebook=_indexed(codebook.table),
        encoding=codebook.encoding, channel_labels=decoded.channel_labels, round_labels=decoded.round_labels,
        color_to_channel=codebook.color_to_channel, candidate_source=_reference_dict(candidate_source) if candidate_source else None,
        trace_unavailable_reason=trace_unavailable_reason, links=links or {})
    context = dict(context)
    if candidate_source:
        context['parents'] = (*context.get('parents', ()), _reference_dict(candidate_source))
    return _write(directory, 'decoded_pre_qc', payload, **context)


@_reader
def load_decoded_checkpoint(path: str | Path, *, sha256: str | None = None) -> DecodedCheckpoint:
    """Reload pre-QC without images or signals; source resolution is explicit."""
    from starfinder.barcode import BarcodeDecodingResult, Codebook
    from starfinder.spot_finding import SpotFindingResult
    a = _read(path, 'decoded_pre_qc', sha256)
    p = a['payload']
    _require(set(p) == {'calls', 'spots', 'metadata', 'namespace', 'detector_config', 'detector_diagnostics',
        'decoder_config', 'diagnostics', 'codebook', 'encoding', 'channel_labels', 'round_labels',
        'color_to_channel', 'candidate_source', 'trace_unavailable_reason', 'links', 'context', 'storage'}, 'decoded payload fields')
    spots = SpotFindingResult(_restore(p['spots']), p['metadata'], p['namespace'], p['detector_config'], p['detector_diagnostics'])
    decoded = BarcodeDecodingResult(_restore(p['calls']), p['namespace'], p['channel_labels'], p['round_labels'],
                                    p['decoder_config'], p['diagnostics'])
    book = Codebook(_restore(p['codebook']), p['round_labels'], p['channel_labels'], p['color_to_channel'], p['encoding'])
    _require(tuple(spots.spots.spot_id) == tuple(decoded.table.spot_id), 'pre-QC population/order mismatch')
    decoded.config.__post_init__()
    for key, shape in [('probabilities', (len(decoded.table), len(decoded.channel_labels), len(decoded.round_labels))),
                       ('wta_round_l2_nll', (len(decoded.table), len(decoded.round_labels)))]:
        if key in decoded.diagnostics:
            _require(isinstance(decoded.diagnostics[key], np.ndarray) and decoded.diagnostics[key].shape == shape,
                     f'{key} axes/shape')
    ref = _reference(p['candidate_source']) if p['candidate_source'] is not None else None
    reason = p['trace_unavailable_reason']
    _require(ref is not None and reason is None or ref is None and isinstance(reason, str) and reason,
             'trace availability mismatch')
    if ref:
        _require(_reference_dict(ref) in a['parents'], 'candidate parent binding')
    return DecodedCheckpoint(a, spots, decoded, book, ref, reason)


@dataclass(frozen=True)
class FinalCheckpoint:
    """Accepted molecules plus complete QC decisions and pinned pre-QC source."""

    artifact: dict
    pre_qc: DecodedCheckpoint
    filtering: ReadFilteringResult

    def molecule_table(self, *, accepted_only: bool = True) -> pd.DataFrame:
        """Return zero-based FOV coordinates, calls, QC and exact source identities."""
        _require(type(accepted_only) is bool, 'accepted_only must be Boolean')
        table = self.filtering.accepted if accepted_only else self.filtering.table.copy()
        coordinates = self.pre_qc.spots.spots[['spot_id', 'z', 'y', 'x']]
        _require(not {'z', 'y', 'x'} & set(table), 'ambiguous coordinate columns')
        table = table.merge(coordinates, on='spot_id', how='left', validate='one_to_one', sort=False)
        source = self.pre_qc.candidate_source
        values = dict(dataset_id=self.artifact['dataset_id'], sample_id=self.artifact['sample_id'],
            FOV=self.artifact['FOV'], frame_id=self.pre_qc.spots.metadata.frame_id,
            source_run_id=source.run_id if source else None,
            candidate_artifact_id=source.artifact_id if source else None,
            decoded_run_id=self.pre_qc.artifact['run_id'], decoded_artifact_id=self.pre_qc.artifact['artifact_id'],
            trace_unavailable_reason=self.pre_qc.trace_unavailable_reason)
        for name, value in values.items():
            _require(name not in table, f'reserved molecule column {name}')
            table[name] = pd.Series([value] * len(table), dtype='string')
        return table.reset_index(drop=True)


def _validate_filter(decoded, filtered):
    from starfinder.barcode import ReadFilteringResult, filter_reads
    _require(isinstance(filtered, ReadFilteringResult) and filtered.spot_namespace == decoded.spot_namespace,
             'filter result/namespace')
    # Reuse the existing policy, never duplicate its predicates or invent scores.
    expected = filter_reads(decoded, config=filtered.config)
    actual = _ordered(filtered.table, tuple(decoded.table.spot_id))
    _equal_table(actual, expected.table, 'QC decisions differ from supplied filter configuration')
    _require(filtered.counts == expected.counts and filtered.fractions == expected.fractions and
             filtered.diagnostics == expected.diagnostics, 'QC summary mismatch')
    return actual


def save_final_checkpoint(directory: str | Path, filtered, *, decoded_source: ArtifactReference,
                          config: dict, code: dict, links: dict | None = None) -> Path:
    """Save accepted and complete QC tables; preserve source frame and references.

    Validation reruns only the existing filter predicates against pinned pre-QC
    calls. It never redecodes, reloads images or assembles FOVs. Source candidate
    traces are not duplicated. Optional links retain supplied H5AD/assignment
    references unchanged, without defining new cell/population policy.
    """
    saved = _resolve(decoded_source, load_decoded_checkpoint)
    table = _validate_filter(saved.decoded, filtered)
    a = saved.artifact
    _require(links is None or isinstance(links, dict), 'links must be a mapping')
    payload = dict(qc=_indexed(table), accepted=_indexed(table.loc[table.accepted]),
        filter_config=filtered.config, counts=filtered.counts, fractions=filtered.fractions,
        diagnostics=filtered.diagnostics, decoded_source=_reference_dict(decoded_source), links=links or {})
    return _write(directory, 'final_accepted', payload,
        **{k: a[k] for k in ('dataset_id', 'sample_id', 'FOV', 'run_id', 'subtile')},
        config=config, code=code, parents=(_reference_dict(decoded_source),))


@_reader
def load_final_checkpoint(path: str | Path, *, sha256: str | None = None) -> FinalCheckpoint:
    """Reload all QC decisions and accepted rows, verifying their exact agreement."""
    from starfinder.barcode import ReadFilteringResult
    a = _read(path, 'final_accepted', sha256)
    p = a['payload']
    _require(set(p) == {'qc', 'accepted', 'filter_config', 'counts', 'fractions', 'diagnostics',
                       'decoded_source', 'links', 'context', 'storage'}, 'final payload fields')
    ref = _reference(p['decoded_source'])
    _require(_reference_dict(ref) in a['parents'], 'decoded parent binding')
    saved = _resolve(ref, load_decoded_checkpoint)
    _require(all(a[k] == saved.artifact[k] for k in ('dataset_id', 'sample_id', 'FOV', 'subtile')), 'decoded FOV identity')
    filtered = ReadFilteringResult(_restore(p['qc']), saved.decoded.spot_namespace, p['filter_config'],
                                   p['counts'], p['fractions'], p['diagnostics'])
    _validate_filter(saved.decoded, filtered)
    _require(tuple(filtered.table.spot_id) == tuple(saved.decoded.table.spot_id), 'QC candidate order')
    _equal_table(_restore(p['accepted']), filtered.accepted, 'accepted population mismatch')
    return FinalCheckpoint(a, saved, filtered)


@dataclass(frozen=True)
class MoleculeBatch:
    """FOV-scoped rows; metadata/codebook/links remain explicit for every batch."""

    table: pd.DataFrame
    source: ArtifactReference
    comparison: dict
    links: dict


@dataclass(frozen=True)
class MoleculeIndex:
    """Ordered FOV references for one sample/optional section; no common frame."""

    dataset_id: str
    sample_id: str
    section_id: str | None
    sources: tuple[ArtifactReference, ...]

    def iter_batches(self, *, batch_size: int = 65536, accepted_only: bool = True):
        """Yield bounded row batches, loading at most one whole FOV at a time.

        Empty FOVs yield one typed empty batch. Acquisition/geometry metadata are
        never silently stacked. Memory is bounded by the largest saved FOV, not
        by batch_size; this is not a row-group streaming signal reader.
        """
        _require(type(batch_size) is int and batch_size > 0, 'positive batch_size required')
        _require(type(accepted_only) is bool, 'accepted_only must be Boolean')
        for ref in self.sources:
            saved = _resolve(ref, load_final_checkpoint)
            _require((saved.artifact['dataset_id'], saved.artifact['sample_id']) ==
                     (self.dataset_id, self.sample_id), 'sample identity mismatch')
            table = saved.molecule_table(accepted_only=accepted_only)
            links = dict(decoded=saved.pre_qc.artifact['payload']['links'], final=saved.artifact['payload']['links'])
            for start in range(0, max(1, len(table)), batch_size):
                yield MoleculeBatch(table.iloc[start:start+batch_size].reset_index(drop=True).copy(), ref,
                                    saved.pre_qc.comparison_metadata(), links)
            del saved, table, links

    def read_table(self, *, accepted_only: bool = True) -> pd.DataFrame:
        """Concatenate FOV rows in index order; reject incompatible column schemas."""
        batches = list(self.iter_batches(accepted_only=accepted_only))
        schemas = [_schema(batch.table) for batch in batches]
        # Null masks can vary by batch; declared pandas types/column order cannot.
        signatures = [[(s['name'], s['dtype'], s['storage']) for s in schema] for schema in schemas]
        _require(all(s == signatures[0] for s in signatures), 'incompatible molecule schemas; use FOV batches')
        return pd.concat([b.table for b in batches], ignore_index=True)

    def source_trace(self, *, run_id: str, candidate_artifact_id: str,
                     spot_namespace: str, spot_id: str) -> dict:
        """Resolve a complete candidate locator, rejecting missing/ambiguous sources."""
        matches = []
        for ref in self.sources:
            saved = _resolve(ref, load_final_checkpoint).pre_qc
            source = saved.candidate_source
            if source and (source.run_id, source.artifact_id, saved.spots.spot_namespace) == (
                    run_id, candidate_artifact_id, spot_namespace):
                matches.append(saved)
        _require(len(matches) == 1, 'missing or ambiguous source reference')
        return matches[0].source_trace(spot_namespace=spot_namespace, spot_id=spot_id)


def save_molecule_index(path: str | Path, sources: tuple[ArtifactReference, ...], *,
                        dataset_id: str, sample_id: str, section_id: str | None = None) -> Path:
    """Save an ordered sample/section index, retaining distinct FOV coordinate frames.

    section_id is only a supplied grouping label. Duplicate FOV/subtile sources
    and duplicate candidate sources are ambiguous and rejected; no overlap
    reconciliation, spatial transform or cell assignment is inferred.
    """
    index = MoleculeIndex(dataset_id, sample_id, section_id, tuple(sources))
    _validate_index(index)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(dict(schema_name='starfinder.molecule_index', schema_version=1,
            contract_id='starfinder.artifacts/1', dataset_id=dataset_id, sample_id=sample_id,
            section_id=section_id, sources=[_reference_dict(s) for s in sources]), handle, indent=2, allow_nan=False)
    return path


def _validate_index(index):
    _require(all(isinstance(x, str) and x for x in (index.dataset_id, index.sample_id)) and
             (index.section_id is None or isinstance(index.section_id, str) and index.section_id), 'index identity')
    _require(bool(index.sources), 'at least one FOV artifact required (empty FOVs are supported)')
    seen, candidates = set(), set()
    for ref in index.sources:
        saved = _resolve(ref, load_final_checkpoint)
        a = saved.artifact
        _require((a['dataset_id'], a['sample_id']) == (index.dataset_id, index.sample_id), 'sample identity mismatch')
        key = (a['FOV'], a['subtile'])
        _require(key not in seen, 'ambiguous duplicate FOV/subtile')
        seen.add(key)
        source = saved.pre_qc.candidate_source
        if source:
            key = (source.run_id, source.artifact_id)
            _require(key not in candidates, 'ambiguous duplicate candidate source')
            candidates.add(key)
        del saved


@_reader
def load_molecule_index(path: str | Path, *, sha256: str | None = None) -> MoleculeIndex:
    """Verify the index and every pinned FOV reference before returning access."""
    path = Path(path)
    if sha256 is not None:
        _require(_is_hash(sha256) and _hash(path) == sha256, 'index checksum mismatch')
    p = _json(path)
    _require(set(p) == {'schema_name', 'schema_version', 'contract_id', 'dataset_id', 'sample_id', 'section_id', 'sources'}
             and p['schema_name'] == 'starfinder.molecule_index' and type(p['schema_version']) is int
             and p['schema_version'] == 1 and p['contract_id'] == 'starfinder.artifacts/1', 'index schema')
    index = MoleculeIndex(p['dataset_id'], p['sample_id'], p['section_id'], tuple(_reference(s) for s in p['sources']))
    _validate_index(index)
    return index
