"""Analytic forward geometry for formed scenes, independent of registration."""
from dataclasses import dataclass

import numpy as np

from ._common import _array, _json


@dataclass(frozen=True)
class GeometryConfig:
    """Absolute reference-to-round translations and Gaussian local displacements.

    Supplied translations have shape (R,3); alternatively translation_max_zyx
    sets componentwise uniform half-ranges. Local centers are (K,3), scales (K,)
    default to 8 voxels, and supplied vectors are (R,K,3). Alternatively strength
    is the normal vector-component SD. Supplied values and random controls are
    mutually exclusive. Local control IDs are control-0, control-1, etc.
    Disabled controls retain validated requests but apply identity. Z=1 requires
    zero requested Z motion/centers; random local Z is identically zero.
    Instances are not hashable because array-valued fields are accepted.
    """

    __hash__ = None

    translation_enabled: bool = False
    translations_zyx: object = None
    translation_max_zyx: object = None
    local_enabled: bool = False
    centers_zyx: object = ()
    scales: object = None
    vectors_zyx: object = None
    strength: float = 0.0


def _prepare_geometry(config, shape, labels, stream):
    if not isinstance(config, GeometryConfig):
        raise TypeError('geometry must be GeometryConfig')
    for flag in (config.translation_enabled, config.local_enabled):
        if type(flag) is not bool:
            raise ValueError('geometry enable flags must be Boolean')
    r = len(labels)
    centers = _array(config.centers_zyx, 'centers_zyx')
    if centers.shape == (0,):
        centers = centers.reshape(0, 3)
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError('centers_zyx must be K by 3')
    k = len(centers)

    def array(value, name, dims, default=0):
        a = np.full(dims, default, dtype=float) if value is None else _array(value, name)
        if a.shape != dims:
            raise ValueError(f'{name} must have shape {dims}')
        return a

    translations = array(config.translations_zyx, 'translations_zyx', (r, 3))
    ranges = array(config.translation_max_zyx, 'translation_max_zyx', (3,))
    scales = array(config.scales, 'scales', (k,), 8)
    vectors = array(config.vectors_zyx, 'vectors_zyx', (r, k, 3))
    strength = _array(config.strength, 'strength')
    if strength.shape != () or strength < 0 or (ranges < 0).any() or (scales <= 0).any():
        raise ValueError('strength/ranges must be nonnegative; scales must be positive')
    if config.translations_zyx is not None and config.translation_max_zyx is not None:
        raise ValueError('select supplied translations or uniform ranges')
    if config.vectors_zyx is not None and strength != 0:
        raise ValueError('select supplied vectors or strength')
    if shape[0] == 1 and (np.any(translations[:, 0]) or ranges[0] != 0
                          or np.any(centers[:, 0]) or np.any(vectors[..., 0])):
        raise ValueError('Z=1 requires zero Z translation, vectors and centers')
    if config.translation_enabled and config.translation_max_zyx is not None:
        for i, label in enumerate(labels):
            # Multiplication avoids overflow in high-low for large finite ranges.
            translations[i] = (2 * stream('geometry.translation', None, label).random(3) - 1) * ranges
    if config.local_enabled and config.vectors_zyx is None and strength != 0:
        for i, label in enumerate(labels):
            for j in range(k):
                vectors[i, j] = stream('geometry.local', _json([f'control-{j}', 'vector']), label).normal(size=3) * strength
        if shape[0] == 1:
            vectors[..., 0] = 0
    # Check supplied requests even when disabled; never rescale or retry a draw.
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        bounds = np.sum(np.hypot.reduce(vectors, axis=-1) / scales / np.sqrt(np.e), axis=1)
    if not np.isfinite(vectors).all() or not np.isfinite(bounds).all() or (bounds > .5).any():
        raise ValueError('local invertibility bound must be <= 0.5')
    if not config.translation_enabled:
        translations = np.zeros_like(translations)
    if not config.local_enabled:
        vectors = np.zeros_like(vectors)
        bounds = np.zeros_like(bounds)
    maps = [dict(translation_zyx=t.tolist(), centers_zyx=centers.tolist(),
                 scales=scales.tolist(), vectors_zyx=v.tolist(), lipschitz_bound=float(b))
            for t, v, b in zip(translations, vectors, bounds)]
    # The label follows the effective maps, not the requested enable flags.
    kinds = [_kind(m) for m in maps]
    effective = dict(type='GeometryConfig', translation_enabled=config.translation_enabled,
                     local_enabled=config.local_enabled,
                     kind='identity' if set(kinds) <= {'identity'} else 'gaussian_rbf_translation',
                     rounds=dict(zip(labels, maps)))
    return maps, effective


def _kind(mapping):
    """Transform label from effective coefficients; zero maps are identity."""
    moving = np.any(mapping['translation_zyx']) or np.any(mapping['vectors_zyx'])
    return 'gaussian_rbf_translation' if moving else 'identity'


def _displacement(coordinates, mapping):
    result = np.zeros_like(coordinates, dtype=np.float64)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        for center, scale, vector in zip(mapping['centers_zyx'], mapping['scales'], mapping['vectors_zyx']):
            delta = (coordinates - center) / scale
            weight = np.exp(-.5 * np.sum(delta * delta, axis=-1))
            result += weight[..., None] * vector
    return result


def _forward(coordinates, mapping):
    result = coordinates + _displacement(coordinates, mapping) + mapping['translation_zyx']
    if not np.isfinite(result).all():
        raise ValueError('nonfinite geometry output')
    return result


def _inverse(coordinates, mapping):
    """Contractive inverse; return reference coordinates and measured diagnostics."""
    target = coordinates - mapping['translation_zyx']
    q = target.copy()
    for iteration in range(1, 101):
        next_q = target - _displacement(q, mapping)
        update = float(np.max(np.abs(next_q - q), initial=0))
        q = next_q
        if not np.isfinite(q).all():
            raise ValueError('nonfinite inverse geometry')
        if update <= 1e-10:
            residual = float(np.max(np.abs(_forward(q, mapping) - coordinates), initial=0))
            if residual > 2e-10:
                raise ValueError('inverse geometry residual exceeds 2e-10 voxels')
            return q, dict(iterations=iteration, max_update=update, max_residual=residual,
                           tolerance=1e-10, max_iterations=100)
    raise ValueError('inverse geometry did not converge in 100 iterations')
