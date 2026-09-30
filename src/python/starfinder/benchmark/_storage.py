"""Strict JSON and checksum-verified artifact storage."""
import hashlib
import json
from pathlib import Path
import numpy as np

SCHEMA_VERSION = 1


def _digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _write(path, value):
    # Exclusive creation: even interrupted attempts are never overwritten.
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)


def _read(path):
    with Path(path).open() as stream:
        return json.load(stream)


def _inside(root, name):
    path = (Path(root) / name).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError(f'artifact escapes root: {name}')
    return path


def _reference(root, path):
    return {'path': str(Path(path).relative_to(root)), 'sha256': _digest(path)}


def _verified(root, ref):
    path = _inside(root, ref['path'])
    if not path.is_file():
        raise FileNotFoundError(f'required artifact missing: {path}')
    if _digest(path) != ref['sha256']:
        raise ValueError(f'artifact checksum mismatch: {path}')
    return path


def _array(path):
    if path.suffix == '.npy':
        return np.load(path, allow_pickle=False)
    import tifffile
    return tifffile.imread(path)


def _save_array(root, directory, name, value):
    path = directory / (name + '.npy')
    with path.open('xb') as stream:
        np.save(stream, value, allow_pickle=False)
    return _reference(root, path)


def _save_transform(root, directory, name, result, artifacts):
    """Persist a registration result without copying a dense field in asdict."""
    from dataclasses import asdict
    transform = result.transform
    arrays = {'displacement_zyx': ('field', 'displacement_artifact'),
              'coefficients': ('coefficients', 'coefficients_artifact')}
    record = {k: v for k, v in vars(transform).items() if k not in arrays and not k.startswith('_')}
    for key in ('reference_metadata', 'moving_metadata'):
        record[key] = asdict(record[key])
    if 'matrix_zyx' in record:
        record['matrix_zyx'] = record['matrix_zyx'].tolist()
    for attribute, (suffix, key) in arrays.items():
        if hasattr(transform, attribute):
            array_name = suffix if name == 'transform' else f'{name}-{suffix}'
            artifacts[array_name] = _save_array(root, directory, array_name, getattr(transform, attribute))
            record[key] = artifacts[array_name]
    record['application_config'] = asdict(result.application_config)
    record['diagnostics'] = asdict(result.diagnostics)
    path = directory / (name + '.json')
    _write(path, record)
    artifacts[name] = _reference(root, path)
