"""Opt-in, lossless per-FOV image artifacts (starfinder.artifacts/1).

Each fresh directory contains artifact.json and images.h5. The manifest is
published last; an interrupted directory is never a loadable checkpoint.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
import json
from pathlib import Path
from typing import TYPE_CHECKING
import uuid

import h5py
import numpy as np

from starfinder.image import ImageMetadata, _validate_image
from starfinder.io.tiff import ImageLoadResult
from starfinder.provenance import _component, _hash, _header, _is_hash, _record
from starfinder.registration import (
    CpdConfig, DemonsConfig, DenseDisplacementTransform, RegistrationDiagnostics,
    RegistrationResult, TpsConfig, TranslationConfig, TranslationTransform, WarpConfig,
)

if TYPE_CHECKING:
    from starfinder.dataset.types import RoundState


@dataclass(frozen=True)
class ImageProcessingState:
    """Actual per-layer history, supplied explicitly by the processing caller.

    ``operations`` is an ordered tuple of objects with operation and config
    fields. Configurations include their effective defaults. ``registrations``
    contains only successfully applied results, in application order; failed
    attempts belong in ``attempts``. Terminal state is prepared, applied,
    reference_unchanged, skipped, failed or not_registered. A non-applied state
    needs a reason. Source geometry is retained independently of output geometry.
    """

    source_metadata: ImageMetadata
    operations: tuple[dict, ...] = ()
    registrations: tuple[RegistrationResult, ...] = ()
    attempts: tuple[dict, ...] = ()
    terminal_state: str = "prepared"
    reason: str | None = "supplied input; no processing applied"


@dataclass(frozen=True)
class ImageLayer:
    """One round's array and roles, with its original loader result and history.

    Roles are sequencing, stain and/or registration_reference. Each round has
    one layer; roles may overlap. Different layers can have different geometry,
    channel counts and labels. Storage never aligns them.
    """

    round_label: str
    roles: tuple[str, ...]
    loaded: ImageLoadResult
    processing: ImageProcessingState


@dataclass(frozen=True)
class ImageCheckpoint:
    """Validated artifact and ordered layers, with no processing on reload."""

    artifact: dict
    rounds: RoundState
    layers: tuple[ImageLayer, ...]

    def sequencing_images(self, *, require_registered: bool = False) -> dict[str, ImageLoadResult]:
        """Return ordered complete ZYXC sequencing inputs for existing algorithms.

        Missing/failed/skipped rounds, differing labels or incompatible output
        geometry raise ValueError. A prepared checkpoint can be inspected/used
        explicitly, but cannot satisfy require_registered=True.
        """
        if require_registered and self.artifact['stage'] != 'registered_images':
            raise ValueError('registered_images stage required')
        selected = {layer.round_label: layer for layer in self.layers}
        result = {}
        for name in self.rounds.sequencing_rounds:
            if name not in selected:
                raise ValueError(f'missing sequencing round: {name}')
            layer = selected[name]
            state = layer.processing.terminal_state
            if state in ('failed', 'skipped') or require_registered and state not in ('applied', 'reference_unchanged'):
                raise ValueError(f'round {name} is not available for processing: {state}')
            if layer.loaded.image.ndim != 4:
                raise ValueError('sequencing extraction requires explicit ZYXC layers')
            if result:
                first = next(iter(result.values()))
                if (layer.loaded.metadata != first.metadata or
                        layer.loaded.image.shape[:3] != first.image.shape[:3] or
                        layer.loaded.channel_labels != first.channel_labels):
                    raise ValueError('sequencing layer geometry/channel labels differ')
            result[name] = layer.loaded
        if not result:
            raise ValueError('no sequencing rounds')
        return result


def _require(condition, message):
    if not condition:
        raise ValueError(f'image checkpoint: {message}')


def _text(value):
    return isinstance(value, str) and bool(value)


def _types():
    # Local import avoids the dataset -> io import cycle. No arbitrary imports
    # or constructors are selected by file content.
    from starfinder.dataset.types import RoundState
    from starfinder.barcode import NeighborhoodSumConfig, EncodingConfig
    from starfinder.spot_finding import LocalMaximaConfig, NoiseLandmarkConfig, PercentileCentroidConfig
    return {cls.__name__: cls for cls in (
        ImageMetadata, ImageProcessingState, RoundState, RegistrationResult,
        RegistrationDiagnostics, TranslationTransform, DenseDisplacementTransform,
        TranslationConfig, DemonsConfig, TpsConfig, CpdConfig, WarpConfig,
        NeighborhoodSumConfig, EncodingConfig, LocalMaximaConfig,
        NoiseLandmarkConfig, PercentileCentroidConfig,
    )}


def _write_array(handle, key, array, artifact_id, layer_id, *, field=False):
    chunks = tuple(min(n, cap) for n, cap in zip(array.shape[:3], (8, 64, 64)))
    if array.ndim == 4:
        chunks += (3 if field else 1,)
    dataset = handle.create_dataset(key, data=array, chunks=chunks, compression='gzip',
                                    compression_opts=4, shuffle=True)
    dataset.attrs['artifact_id'] = artifact_id
    dataset.attrs['layer_id'] = layer_id
    return dict(key=key, shape=list(array.shape), dtype=array.dtype.str,
                chunks=list(dataset.chunks), compression=dataset.compression,
                compression_opts=dataset.compression_opts, shuffle=dataset.shuffle,
                artifact_id=artifact_id, layer_id=layer_id)


def _read_array(handle, descriptor, artifact_id, layer_id):
    _require(isinstance(descriptor, dict), 'array descriptor required')
    required = {'key', 'shape', 'dtype', 'chunks', 'compression', 'compression_opts',
                'shuffle', 'artifact_id', 'layer_id'}
    _require(set(descriptor) == required, 'array descriptor fields')
    for name in ('shape', 'chunks'):
        _require(isinstance(descriptor[name], list) and len(descriptor[name]) in (3, 4) and
                 all(type(n) is int and n > 0 for n in descriptor[name]),
                 f'invalid array {name}')
    _require(isinstance(descriptor['dtype'], str), 'invalid array dtype')
    key = descriptor['key']
    _require(_text(key) and key.startswith(f'/layers/{layer_id}/') and '..' not in key.split('/'), 'unsafe dataset key')
    # Reject external/soft links at every level, even in a checksummed file.
    group = handle
    for part in key.strip('/').split('/'):
        _require(isinstance(group.get(part, getlink=True), h5py.HardLink), 'missing dataset or nonlocal HDF5 link')
        group = group[part]
    dataset = group
    _require(isinstance(dataset, h5py.Dataset) and not dataset.is_virtual and not dataset.external,
             'image must be a local dataset')
    _require(descriptor['artifact_id'] == artifact_id and descriptor['layer_id'] == layer_id and
             dataset.attrs.get('artifact_id') == artifact_id and dataset.attrs.get('layer_id') == layer_id,
             'dataset artifact/layer binding mismatch')
    _require(list(dataset.shape) == descriptor['shape'] and dataset.dtype.str == descriptor['dtype'], 'shape/dtype mismatch')
    _require(list(dataset.chunks or ()) == descriptor['chunks'] and
             dataset.compression == descriptor['compression'] == 'gzip' and
             dataset.compression_opts == descriptor['compression_opts'] == 4 and
             dataset.shuffle == descriptor['shuffle'] is True and dataset.scaleoffset is None,
             'unsupported or mismatched storage settings')
    return dataset[...]


def _encode(value, handle, artifact_id, layer_id, arrays):
    if is_dataclass(value) and type(value).__name__ in _types():
        return {'type': type(value).__name__, 'fields': {
            f.name: _encode(getattr(value, f.name), handle, artifact_id, layer_id, arrays) for f in fields(value)}}
    if isinstance(value, np.ndarray):
        _require(value.ndim == 4 and value.shape[-1] == 3 and value.dtype in (np.dtype('float32'), np.dtype('float64')),
                 'only dense ZYX3 transform arrays are metadata components')
        key = f'/layers/{layer_id}/fields/{len(arrays)}'
        descriptor = _write_array(handle, key, value, artifact_id, layer_id, field=True)
        arrays.append(descriptor)
        return {'array': descriptor}
    if isinstance(value, tuple):
        return {'tuple': [_encode(v, handle, artifact_id, layer_id, arrays) for v in value]}
    if isinstance(value, list):
        return [_encode(v, handle, artifact_id, layer_id, arrays) for v in value]
    if isinstance(value, dict):
        _require(all(isinstance(k, str) for k in value), 'metadata keys must be strings')
        return {'mapping': {k: _encode(v, handle, artifact_id, layer_id, arrays) for k, v in value.items()}}
    if isinstance(value, Path):
        return {'path': str(value)}
    if isinstance(value, np.generic):
        return _encode(value.item(), handle, artifact_id, layer_id, arrays)
    _require(value is None or type(value) in (str, int, float, bool), f'unsupported metadata type {type(value).__name__}')
    _require(not isinstance(value, float) or np.isfinite(value), 'nonfinite metadata')
    return value


def _decode(value, handle, artifact_id, layer_id, arrays):
    if isinstance(value, list):
        return [_decode(v, handle, artifact_id, layer_id, arrays) for v in value]
    if not isinstance(value, dict):
        _require(value is None or type(value) in (str, int, float, bool), 'invalid metadata value')
        _require(not isinstance(value, float) or np.isfinite(value), 'nonfinite metadata')
        return value
    decode = lambda v: _decode(v, handle, artifact_id, layer_id, arrays)
    if set(value) == {'tuple'}:
        _require(isinstance(value['tuple'], list), 'tuple encoding')
        return tuple(decode(v) for v in value['tuple'])
    if set(value) == {'mapping'}:
        _require(isinstance(value['mapping'], dict), 'mapping encoding')
        return {k: decode(v) for k, v in value['mapping'].items()}
    if set(value) == {'path'}:
        _require(isinstance(value['path'], str), 'path encoding')
        return Path(value['path'])
    if set(value) == {'array'}:
        descriptor = value['array']
        _require(descriptor['key'] not in arrays, 'duplicate field key')
        arrays.add(descriptor['key'])
        return _read_array(handle, descriptor, artifact_id, layer_id)
    _require(set(value) == {'type', 'fields'} and value['type'] in _types(), 'unknown typed metadata')
    cls = _types()[value['type']]
    _require(isinstance(value['fields'], dict) and set(value['fields']) == {f.name for f in fields(cls)}, 'missing/unknown typed fields')
    decoded = {k: decode(v) for k, v in value['fields'].items()}
    result = cls(**{f.name: decoded[f.name] for f in fields(cls) if f.init})
    _require(all(getattr(result, f.name) == decoded[f.name] for f in fields(cls) if not f.init),
             'fixed configuration discriminator mismatch')
    return result


def _validate_layers(layers, rounds, stage):
    _require(isinstance(rounds, _types()['RoundState']), 'RoundState required')
    rounds.validate()
    _require(stage in ('prepared_input', 'registered_images'), 'unsupported image stage')
    _require(bool(layers), 'empty layer inventory')
    seen = set()
    for layer in layers:
        _require(isinstance(layer, ImageLayer) and layer.round_label in rounds.all_rounds and layer.round_label not in seen, 'duplicate/unknown round')
        seen.add(layer.round_label)
        roles = layer.roles
        _require(isinstance(roles, tuple) and bool(roles) and len(set(roles)) == len(roles) and
                 set(roles) <= {'sequencing', 'stain', 'registration_reference'}, 'invalid layer roles')
        _require(('sequencing' in roles) == (layer.round_label in rounds.sequencing_rounds), 'sequencing role mismatch')
        _require(('registration_reference' in roles) == (layer.round_label == rounds.reference_round), 'reference role mismatch')
        loaded = layer.loaded
        _require(isinstance(loaded, ImageLoadResult) and isinstance(loaded.metadata, ImageMetadata), 'ImageLoadResult and metadata required')
        array = _validate_image(loaded.image, ndim=(3, 4))
        labels = loaded.channel_labels
        _require(isinstance(labels, tuple) and all(_text(x) for x in labels) and len(set(labels)) == len(labels) and
                 len(labels) == (array.shape[-1] if array.ndim == 4 else 1), 'channel labels/axes mismatch')
        _require(isinstance(loaded.source_paths, tuple) and all(isinstance(p, Path) for p in loaded.source_paths) and isinstance(loaded.diagnostics, dict), 'source paths/diagnostics')
        state = layer.processing
        _require(isinstance(state, ImageProcessingState) and isinstance(state.source_metadata, ImageMetadata), 'source metadata required')
        _require(isinstance(state.operations, tuple) and all(isinstance(op, dict) and _text(op.get('operation')) and 'config' in op for op in state.operations), 'operation/config history required')
        _require(isinstance(state.attempts, tuple) and all(isinstance(a, dict) and a.get('outcome') in ('succeeded', 'failed', 'application_failed', 'estimating') and
                 _text(a.get('requested_method')) and _text(a.get('actual_method')) and 'config' in a and 'failure' in a for a in state.attempts), 'registration attempts malformed')
        _require(isinstance(state.registrations, tuple) and all(isinstance(r, RegistrationResult) for r in state.registrations), 'registration results required')
        terminal = state.terminal_state
        _require(terminal in ('prepared', 'applied', 'reference_unchanged', 'skipped', 'failed', 'not_registered'), 'invalid terminal state')
        _require(terminal == 'applied' and state.reason is None or terminal != 'applied' and _text(state.reason), 'terminal state reason')
        if stage == 'prepared_input':
            _require(terminal == 'prepared' and not state.registrations and not state.attempts and not state.operations and state.source_metadata == loaded.metadata,
                     'prepared input cannot contain processing history')
        else:
            _require(terminal != 'prepared', 'registered checkpoint needs actual terminal state')
            _require((terminal == 'reference_unchanged') == (layer.round_label == rounds.reference_round), 'reference terminal state mismatch')
            _require(terminal != 'applied' or bool(state.registrations), 'applied round requires transform')
            _require(not state.registrations or terminal in ('applied', 'failed'), 'transform contradicts terminal state')
            _require(bool(state.operations) or state.source_metadata == loaded.metadata,
                     'geometry changed without operation history')
            if state.registrations:
                applications = [op for op in state.operations if op['operation'] == 'apply_transform']
                _require(len(applications) == len(state.registrations) and
                         all(op['config'] == result.application_config
                             for op, result in zip(applications, state.registrations)),
                         'application configs/history do not match applied results')
                for previous, current in zip(state.registrations, state.registrations[1:]):
                    _require(previous.transform.reference_metadata == current.transform.moving_metadata, 'transform chain geometry mismatch')
                transform = state.registrations[-1].transform
                _require(transform.reference_metadata == loaded.metadata and transform.reference_shape_zyx == array.shape[:3], 'transform/output geometry mismatch')
                _require(any(op['operation'] == 'apply_transform' for op in state.operations), 'applied transform missing from operation history')


def _validate_context(config, code, sources, parents, provenance):
    _require(isinstance(config, dict) and isinstance(code, dict) and
             (_text(code.get('commit')) or _is_hash(code.get('patch_sha256')) or _text(code.get('unknown_reason'))),
             'code revision/patch and config required')
    _require(provenance is None or isinstance(provenance, dict) and
             _text(provenance.get('uri')) and _is_hash(provenance.get('sha256')),
             'provenance URI/checksum')
    _require(isinstance(sources, (tuple, list)), 'source inventory required')
    source_ids = []
    for source in sources:
        _require(isinstance(source, dict) and
                 {'source_id', 'catalog', 'uri', 'sha256', 'unverified_reason', 'selection'} <= source.keys() and
                 _text(source['source_id']) and _text(source['uri']) and isinstance(source['selection'], dict) and
                 (_is_hash(source['sha256']) or source['sha256'] is None and _text(source['unverified_reason'])),
                 'source identity/selection/checksum')
        source_ids.append(source['source_id'])
    _require(len(set(source_ids)) == len(source_ids), 'duplicate source ID')
    for parent in parents:
        _require(isinstance(parent, dict) and _text(parent.get('run_id')) and
                 _text(parent.get('artifact_id')) and _is_hash(parent.get('sha256')),
                 'parent identity/checksum')
    return source_ids


def save_image_checkpoint(directory: str | Path, layers: tuple[ImageLayer, ...], *,
                          rounds: RoundState, stage: str, dataset_id: str, sample_id: str,
                          FOV: str, run_id: str, config: dict, code: dict,
                          sources: tuple[dict, ...] = (), parents: tuple[dict, ...] = (),
                          provenance: dict | None = None, subtile: int | None = None) -> Path:
    """Save supplied layers without conversion to a fresh per-FOV directory.

    ``stage`` is prepared_input or registered_images. ``code`` supplies commit
    or patch_sha256 identity; ``config`` records the effective caller settings.
    ``sources`` follow the v1 source-record contract (including selection and
    checksum or an explicit unverified reason). ``provenance``, when supplied,
    is a URI and SHA-256 link to a saved run; it is preserved, not dereferenced.
    Parent references bind run/artifact IDs and manifest SHA-256. Images are
    opt-in; this function is never called automatically by processing.

    Existing directories raise FileExistsError. Invalid inputs raise ValueError.
    The manifest is written last; failed writes remain inspectable but incomplete.
    Returns the artifact.json path. No run manifest is mutated.
    """
    _validate_layers(layers, rounds, stage)
    source_ids = _validate_context(config, code, sources, parents, provenance)
    artifact_id = uuid.uuid4().hex
    artifact = dict(_header('artifact'), artifact_id=artifact_id, run_id=run_id, stage=stage,
        status='complete', dataset_id=dataset_id, sample_id=sample_id, FOV=FOV, subtile=subtile,
        parents=list(parents), config_ref='payload/config', source_refs=source_ids,
        components=[{}], payload={}, omission_reason=None, failure_id=None)
    _record(artifact, 'artifact')
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    arrays = []
    with h5py.File(root / 'images.h5', 'x') as handle:
        inventory = []
        encode = lambda value, key: _encode(value, handle, artifact_id, key, arrays)
        for index, layer in enumerate(layers):
            key = f'layer{index:04d}'
            image = layer.loaded
            inventory.append(dict(layer_id=key, round_label=layer.round_label, roles=list(layer.roles),
                axes='ZYXC' if image.image.ndim == 4 else 'ZYX',
                image=_write_array(handle, f'/layers/{key}/image', image.image, artifact_id, key),
                metadata=encode(image.metadata, key), channel_labels=list(image.channel_labels),
                source_paths=[str(p) for p in image.source_paths], diagnostics=encode(image.diagnostics, key),
                processing=encode(layer.processing, key)))
        artifact['payload'] = dict(rounds=encode(rounds, 'metadata'), layers=inventory,
            config=encode(config, 'metadata'), code=encode(code, 'metadata'),
            sources=encode(sources, 'metadata'), provenance=encode(provenance, 'metadata'),
            storage=dict(format='hdf5', h5py_version=h5py.__version__, compression='gzip', compression_opts=4, shuffle=True))
    component = root / 'images.h5'
    artifact['components'] = [dict(component_id='images', path='images.h5', format='hdf5',
                                   size=component.stat().st_size, sha256=_hash(component))]
    text = json.dumps(artifact, indent=2, allow_nan=False)
    temporary = root / 'artifact.json.tmp'
    temporary.write_text(text, encoding='utf-8')
    temporary.replace(root / 'artifact.json')
    return root / 'artifact.json'


def load_image_checkpoint(path: str | Path, *, expected_stage: str | None = None,
                          sha256: str | None = None) -> ImageCheckpoint:
    """Validate and reload every saved layer/transform exactly; never apply it.

    Reads artifact.json (or a directory containing it), verifies HDF5 checksum,
    schema, bindings, geometry and state before returning any images. Missing
    files raise FileNotFoundError; corrupt/mismatched/incomplete metadata raises
    ValueError. A partial round inventory remains inspectable; use
    sequencing_images to require a complete downstream input. An optional
    manifest SHA-256 binds the metadata itself to a trusted parent/handoff.
    """
    path = Path(path)
    if path.is_dir():
        path = path / 'artifact.json'
    if sha256 is not None:
        _require(_is_hash(sha256) and _hash(path) == sha256, 'manifest checksum mismatch')
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    def constant(value):
        raise ValueError(f'image checkpoint: nonfinite JSON number {value}')
    try:
        artifact = json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=pairs,
                              parse_constant=constant)
        _record(artifact, 'artifact')
        _require(artifact['status'] == 'complete' and artifact['stage'] in ('prepared_input', 'registered_images'), 'incomplete/unsupported stage')
        _require(expected_stage is None or artifact['stage'] == expected_stage, 'stage mismatch')
        _require(artifact['config_ref'] == 'payload/config', 'config reference mismatch')
        components = artifact['components']
        _require(len(components) == 1 and components[0]['component_id'] == 'images' and components[0]['format'] == 'hdf5', 'HDF5 component required')
        component = _component(path.parent, components[0])
        payload = artifact['payload']
        _require(set(payload) == {'rounds', 'layers', 'config', 'code', 'sources', 'provenance', 'storage'}, 'payload fields')
        layers = []
        seen = set()
        with h5py.File(component, 'r') as handle:
            decode = lambda value, key: _decode(value, handle, artifact['artifact_id'], key, set())
            rounds = decode(payload['rounds'], 'metadata')
            for row in payload['layers']:
                _require(set(row) == {'layer_id', 'round_label', 'roles', 'axes', 'image', 'metadata', 'channel_labels', 'source_paths', 'diagnostics', 'processing'}, 'layer fields')
                key = row['layer_id']
                _require(all(isinstance(row[name], list) and all(_text(v) for v in row[name])
                             for name in ('roles', 'channel_labels', 'source_paths')),
                         'layer roles/labels/source paths must be string lists')
                _require(isinstance(key, str) and key.startswith('layer') and key[5:].isdigit() and key not in seen, 'unsafe/duplicate layer ID')
                seen.add(key)
                _require(row['image']['key'] == f'/layers/{key}/image', 'image key mismatch')
                array = _read_array(handle, row['image'], artifact['artifact_id'], key)
                _require(row['axes'] == ('ZYXC' if array.ndim == 4 else 'ZYX'), 'axes mismatch')
                layers.append(ImageLayer(row['round_label'], tuple(row['roles']),
                    ImageLoadResult(array, decode(row['metadata'], key), tuple(row['channel_labels']),
                        tuple(Path(p) for p in row['source_paths']), decode(row['diagnostics'], key)),
                    decode(row['processing'], key)))
            for key in ('config', 'code', 'sources', 'provenance'):
                payload[key] = decode(payload[key], 'metadata')
        _validate_layers(layers, rounds, artifact['stage'])
        source_ids = _validate_context(payload['config'], payload['code'], payload['sources'],
                                      artifact['parents'], payload['provenance'])
        _require(source_ids == artifact['source_refs'], 'source references mismatch')
        storage = payload['storage']
        _require(isinstance(storage, dict) and storage.get('format') == 'hdf5' and
                 storage.get('compression') == 'gzip' and storage.get('compression_opts') == 4 and
                 storage.get('shuffle') is True and _text(storage.get('h5py_version')), 'storage metadata mismatch')
        return ImageCheckpoint(artifact, rounds, tuple(layers))
    except (KeyError, TypeError, AttributeError, OSError) as exc:
        if isinstance(exc, FileNotFoundError):
            raise
        raise ValueError(f'image checkpoint: malformed or corrupt artifact: {exc}') from exc
