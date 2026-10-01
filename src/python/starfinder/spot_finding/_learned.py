"""Shared parts of the learned detectors (Spotiflow, Piscis): the model cache, the model record and the table.

The run order is the contract's: find_spots has already imported the
library through require(); each method then verifies every listed weights
file with resolve_weights (verified_folder) and only then constructs the
model from the verified files, through an absolute path.
Nothing here uses the network or the libraries' own caches.
"""
import numpy as np
import pandas as pd

from ._weights import known_weights, listed_files, resolve_weights, weights_artifacts

# Loaded models of this process, keyed by (method, model, SHA-256 of the loaded weights file), so the
# channels and rounds of a run load a model once. Verification still runs on every call, before the lookup.
_MODELS: dict[tuple[str, str, str], object] = {}


def verified_folder(method, model):
    """The model's cache folder after resolve_weights has recomputed the SHA-256 of every file KNOWN_WEIGHTS
    lists for it (its files and all extracted files); the same for every method, with nothing skipped."""
    return resolve_weights(method, model, extracted=listed_files(method, model))


def absolute(path, library):
    """path as a string, after checking that it is absolute: a library never receives a relative path."""
    if not path.is_absolute():
        raise ValueError(f"{library} would receive the relative weights path {str(path)!r}; it must be absolute")
    return str(path)


def cached_model(method, model, build):
    """The loaded model of (method, model), built by build() on first use."""
    key = (method, model, known_weights(method, model).files[0].sha256)
    if key not in _MODELS:
        _MODELS[key] = build()
    return _MODELS[key]


def model_record(method, model, folder):
    """diagnostics['model']: method, model, the provenance artifacts entries (one per verified file) and the
    training pixel size."""
    entry = known_weights(method, model)
    return {'method': method, 'model': model,
            'artifacts': weights_artifacts(method, model, folder, listed_files(method, model)),
            'training_pixel_size': entry.training_pixel_size,
            'training_pixel_size_provenance': entry.training_pixel_size_provenance}


def channels_of(image, context):
    """(channel index, ZYX channel) for each channel in context.channels."""
    return [(c, image[..., c] if image.ndim == 4 else image) for c in context.channels]


def is_constant(channel):
    """A constant channel yields no candidates without running the model."""
    return channel.min() == channel.max()


def frame(channel, c, zyx, columns, **extra):
    """The rows of one channel: coordinates as float64 and peak_intensity, the original pixel value at the
    coordinate rounded half to even (np.rint) and clipped into the image."""
    zyx = np.asarray(zyx, dtype=np.float64).reshape(-1, 3)
    index = np.clip(np.rint(zyx).astype(np.int64), 0, np.asarray(channel.shape) - 1)
    data = {'z': zyx[:, 0], 'y': zyx[:, 1], 'x': zyx[:, 2], 'channel': np.full(len(zyx), c, dtype=np.int64),
            'peak_intensity': channel[tuple(index.T)].astype(np.float64)}
    data.update({name: np.asarray(values, dtype=np.float64) for name, values in extra.items()})
    return pd.DataFrame(data)[list(columns)]


def table(frames, columns):
    """The channels' rows in channel order, typed even when empty."""
    types = {name: 'int64' if name == 'channel' else 'float64' for name in columns}
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=list(columns))).astype(types)
