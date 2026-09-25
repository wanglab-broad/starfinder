"""Analytic forward geometry for formed scenes, independent of registration."""
from dataclasses import dataclass

import numpy as np

from ._common import _array, _json

# Grid points per inverse-geometry block; grids up to this size use one block.
_BLOCK_VOXELS = 1 << 18
# Second-order monomials of normalized ZYX coordinates, in column order.
POLYNOMIAL_TERMS = ('zz', 'yy', 'xx', 'zy', 'zx', 'yx')
_TERM_AXES = tuple(('zyx'.index(t[0]), 'zyx'.index(t[1])) for t in POLYNOMIAL_TERMS)


@dataclass(frozen=True)
class GeometryConfig:
    """Absolute reference-to-round translations plus local, affine and polynomial maps.

    Supplied translations have shape (R,3); alternatively translation_max_zyx
    sets componentwise uniform half-ranges. Local centers are (K,3), scales (K,)
    default to 8 voxels, and supplied vectors are (R,K,3). Alternatively strength
    is the normal vector-component SD, or local_magnitude gives each control a
    uniformly random direction with that length. Local control IDs are
    control-0, control-1, etc.

    Affine and polynomial terms act about the grid centre c=(shape-1)/2.
    Affine matrices (R,3,3) give A(q-c) in voxels; polynomial coefficients
    (R,3,6) multiply the monomials of POLYNOMIAL_TERMS (zz, yy, xx, zy, zx, yx)
    of u=(q-c)/h, in voxels, where the isotropic scale h is the largest grid
    half extent (at least 1), so thin Z stacks have small u_z.
    Alternatively affine_max_zyx/polynomial_max_zyx draw uniform coefficients,
    scaled so each output axis displacement is at most that bound on the grid
    (exact at a grid corner for affine terms).

    Supplied values and random controls are mutually exclusive. reference_round
    names a round held at identity (no draws). Disabled controls retain
    validated requests but apply identity. Z=1 requires zero requested Z motion,
    centers and Z output rows; random Z terms are identically zero. The summed
    Lipschitz bound (local terms globally, polynomial terms on the grid) must be
    at most 0.5. Instances are not hashable because array-valued fields are accepted.
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
    local_magnitude: float | None = None
    affine_enabled: bool = False
    affine_zyx: object = None
    affine_max_zyx: object = None
    polynomial_enabled: bool = False
    polynomial_zyx: object = None
    polynomial_max_zyx: object = None
    reference_round: str | None = None


def _frame(shape):
    """Grid centre, isotropic normalization and half extents in voxel-index units."""
    extent = (np.asarray(shape, dtype=np.float64) - 1) / 2
    return extent.copy(), np.full(3, max(float(extent.max()), 1.0)), extent


def _polynomial_jacobian_bound(coefficients, half, extent):
    """Frobenius bound of the polynomial Jacobian over the grid box |u_j| <= extent_j/h."""
    reach = extent / half
    bound = np.zeros((3, 3))
    for t, (a, b) in enumerate(_TERM_AXES):
        c = np.abs(coefficients[:, t])
        if a == b:
            bound[:, a] += c * 2 * reach[a] / half[a]
        else:
            bound[:, a] += c * reach[b] / half[a]
            bound[:, b] += c * reach[a] / half[b]
    return float(np.sqrt(np.sum(bound**2)))


def _prepare_geometry(config, shape, labels, stream):
    if not isinstance(config, GeometryConfig):
        raise TypeError('geometry must be GeometryConfig')
    flags = dict(translation=config.translation_enabled, local=config.local_enabled,
                 affine=config.affine_enabled, polynomial=config.polynomial_enabled)
    if any(type(flag) is not bool for flag in flags.values()):
        raise ValueError('geometry enable flags must be Boolean')
    if config.reference_round is not None and config.reference_round not in labels:
        raise ValueError('reference_round must be a codebook round label')
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
    affine = array(config.affine_zyx, 'affine_zyx', (r, 3, 3))
    affine_max = array(config.affine_max_zyx, 'affine_max_zyx', (3,))
    polynomial = array(config.polynomial_zyx, 'polynomial_zyx', (r, 3, len(POLYNOMIAL_TERMS)))
    polynomial_max = array(config.polynomial_max_zyx, 'polynomial_max_zyx', (3,))
    strength = _array(config.strength, 'strength')
    magnitude = None if config.local_magnitude is None else _array(config.local_magnitude, 'local_magnitude')
    if (strength.shape != () or strength < 0 or (ranges < 0).any() or (scales <= 0).any()
            or (affine_max < 0).any() or (polynomial_max < 0).any()
            or (magnitude is not None and (magnitude.shape != () or magnitude < 0))):
        raise ValueError('strength/magnitude/ranges must be nonnegative; scales must be positive')
    if config.translations_zyx is not None and config.translation_max_zyx is not None:
        raise ValueError('select supplied translations or uniform ranges')
    if config.vectors_zyx is not None and (strength != 0 or magnitude is not None) or (
            strength != 0 and magnitude is not None):
        raise ValueError('select supplied vectors, strength or local_magnitude')
    if config.affine_zyx is not None and config.affine_max_zyx is not None:
        raise ValueError('select supplied affine matrices or affine_max_zyx')
    if config.polynomial_zyx is not None and config.polynomial_max_zyx is not None:
        raise ValueError('select supplied polynomial coefficients or polynomial_max_zyx')
    if shape[0] == 1 and (np.any(translations[:, 0]) or ranges[0] != 0
                          or np.any(centers[:, 0]) or np.any(vectors[..., 0])
                          or np.any(affine[:, 0]) or affine_max[0] != 0
                          or np.any(polynomial[:, 0]) or polynomial_max[0] != 0):
        raise ValueError('Z=1 requires zero Z translation, vectors, centers and Z output rows')
    center, half, extent = _frame(shape)
    moving = [label != config.reference_round for label in labels]
    if config.translation_enabled and config.translation_max_zyx is not None:
        for i, label in enumerate(labels):
            if moving[i]:
                # Multiplication avoids overflow in high-low for large finite ranges.
                translations[i] = (2 * stream('geometry.translation', None, label).random(3) - 1) * ranges
    if config.local_enabled and config.vectors_zyx is None and (strength != 0 or magnitude is not None):
        for i, label in enumerate(labels):
            for j in range(k if moving[i] else 0):
                draw = stream('geometry.local', _json([f'control-{j}', 'vector']), label).normal(size=3)
                if magnitude is None:
                    vectors[i, j] = draw * strength
                else:
                    if shape[0] == 1:
                        draw[0] = 0
                    norm = np.sqrt(np.sum(draw * draw))
                    vectors[i, j] = draw / norm * magnitude if norm > 0 else 0
        if shape[0] == 1:
            vectors[..., 0] = 0
    for enabled, values, bound, component in (
            (config.affine_enabled and config.affine_max_zyx is not None, affine, affine_max, 'affine'),
            (config.polynomial_enabled and config.polynomial_max_zyx is not None,
             polynomial, polynomial_max, 'polynomial')):
        if not enabled:
            continue
        reach = extent if component == 'affine' else np.array(
            [extent[a] * extent[b] / (half[a] * half[b]) for a, b in _TERM_AXES])
        for i, label in enumerate(labels):
            if not moving[i]:
                continue
            draw = stream('geometry.' + component, None, label).uniform(-1, 1, values.shape[1:])
            if shape[0] == 1:
                draw[0] = 0
            # Row k attains sum |c| * reach at a grid corner (a bound for monomials).
            peak = np.abs(draw) @ reach
            values[i] = draw * np.divide(bound, peak, out=np.zeros(3), where=peak > 0)[:, None]
    for values in (translations, vectors, affine, polynomial):
        values[[not m for m in moving]] = 0
    # Check supplied requests even when disabled; never rescale or retry a draw.
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        bounds = np.sum(np.hypot.reduce(vectors, axis=-1) / scales / np.sqrt(np.e), axis=1)
        bounds = bounds + np.sqrt(np.sum(affine**2, axis=(1, 2)))
        bounds = bounds + [_polynomial_jacobian_bound(p, half, extent) for p in polynomial]
    if (not np.isfinite(vectors).all() or not np.isfinite(affine).all()
            or not np.isfinite(polynomial).all() or not np.isfinite(bounds).all() or (bounds > .5).any()):
        raise ValueError('local invertibility bound must be <= 0.5')
    if not config.translation_enabled:
        translations = np.zeros_like(translations)
    if not config.local_enabled:
        vectors = np.zeros_like(vectors)
    if not config.affine_enabled:
        affine = np.zeros_like(affine)
    if not config.polynomial_enabled:
        polynomial = np.zeros_like(polynomial)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        bounds = (np.sum(np.hypot.reduce(vectors, axis=-1) / scales / np.sqrt(np.e), axis=1)
                  + np.sqrt(np.sum(affine**2, axis=(1, 2)))
                  + [_polynomial_jacobian_bound(p, half, extent) for p in polynomial])
    maps = []
    for t, v, a, p, b in zip(translations, vectors, affine, polynomial, bounds):
        mapping = dict(translation_zyx=t.tolist(), centers_zyx=centers.tolist(),
                       scales=scales.tolist(), vectors_zyx=v.tolist(), lipschitz_bound=float(b))
        # Only nonzero affine/polynomial terms are recorded, keeping local-only records unchanged.
        if np.any(a) or np.any(p):
            mapping.update(grid_center_zyx=center.tolist(), grid_half_zyx=half.tolist(),
                           affine_zyx=a.tolist(), polynomial_zyx=p.tolist(),
                           polynomial_terms=list(POLYNOMIAL_TERMS))
        maps.append(mapping)
    # The label follows the effective maps, not the requested enable flags.
    kinds = {_kind(m) for m in maps}
    kind = next((k for k in ('polynomial_affine_gaussian_rbf_translation', 'gaussian_rbf_translation')
                 if k in kinds), 'identity')
    effective = dict(type='GeometryConfig', **{f'{k}_enabled': v for k, v in flags.items()},
                     reference_round=config.reference_round, kind=kind, rounds=dict(zip(labels, maps)))
    return maps, effective


def _kind(mapping):
    """Transform label from effective coefficients; zero maps are identity."""
    if np.any(mapping.get('affine_zyx', 0)) or np.any(mapping.get('polynomial_zyx', 0)):
        return 'polynomial_affine_gaussian_rbf_translation'
    moving = np.any(mapping['translation_zyx']) or np.any(mapping['vectors_zyx'])
    return 'gaussian_rbf_translation' if moving else 'identity'


def _translation_only(mapping):
    """True when the effective map is q + t (no local, affine or polynomial term)."""
    return not (np.any(mapping['vectors_zyx']) or np.any(mapping.get('affine_zyx', 0))
                or np.any(mapping.get('polynomial_zyx', 0)))


def _displacement(coordinates, mapping):
    # Explicit three-term sums, in-place temporaries and skipped all-zero terms
    # give the same bits as the direct expressions with fewer passes.
    result = np.zeros_like(coordinates, dtype=np.float64)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore', under='ignore'):
        for center, scale, vector in zip(mapping['centers_zyx'], mapping['scales'], mapping['vectors_zyx']):
            if not np.any(vector):
                continue
            delta = coordinates - center
            delta /= scale
            np.multiply(delta, delta, out=delta)
            weight = delta[..., 0] + delta[..., 1]
            weight += delta[..., 2]
            weight *= -.5
            np.exp(weight, out=weight)
            for k in range(3):
                result[..., k] += weight * vector[k]
        if 'affine_zyx' in mapping:
            offset = coordinates - mapping['grid_center_zyx']
            affine = np.asarray(mapping['affine_zyx'])
            if np.any(affine):
                result += offset @ affine.T
            polynomial = np.asarray(mapping['polynomial_zyx'])
            if np.any(polynomial):
                u = offset / mapping['grid_half_zyx']
                terms = np.empty(u.shape[:-1] + (len(_TERM_AXES),))
                for t, (a, b) in enumerate(_TERM_AXES):
                    np.multiply(u[..., a], u[..., b], out=terms[..., t])
                result += terms @ polynomial.T
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


def _inverse_per_point(coordinates, mapping):
    """_inverse with per-point stopping: a point stops once its own update is <= 1e-10.

    Same fixed point, update tolerance and residual limit as _inverse. Converged
    points are removed once they are at least 1/8 of those still iterating
    (until then they keep iterating, staying within the tolerance), so values
    can differ from the whole-block iteration by less than the tolerance.
    """
    shape = coordinates.shape
    # Component-major (3, N) storage keeps gathers and per-axis passes contiguous.
    target = np.ascontiguousarray(np.moveaxis(coordinates - mapping['translation_zyx'], -1, 0)).reshape(3, -1)
    q = np.empty_like(target)
    index = np.arange(target.shape[1])
    current, goal = target, target
    max_update = 0.0
    for iteration in range(1, 101):
        step = _displacement(current.T, mapping).T
        np.subtract(goal, step, out=step)
        diff = step - current
        np.abs(diff, out=diff)
        update = np.maximum(diff[0], diff[1])
        np.maximum(update, diff[2], out=update)
        if not np.isfinite(update).all():
            raise ValueError('nonfinite inverse geometry')
        done = update <= 1e-10
        finished = int(np.count_nonzero(done))
        if finished == len(index):
            q[:, index] = step
            max_update = max(max_update, float(update.max(initial=0)))
            q = np.moveaxis(q.reshape(3, *shape[:-1]), 0, -1)
            residual = float(np.max(np.abs(_forward(q, mapping) - coordinates), initial=0))
            if residual > 2e-10:
                raise ValueError('inverse geometry residual exceeds 2e-10 voxels')
            return q, dict(iterations=iteration, max_update=max_update, max_residual=residual,
                           tolerance=1e-10, max_iterations=100, stopping='per_point')
        if finished >= len(index) // 8:
            q[:, index[done]] = step[:, done]
            max_update = max(max_update, float(update[done].max(initial=0)))
            keep = ~done
            index, current, goal = index[keep], step[:, keep], goal[:, keep]
        else:
            current = step
    raise ValueError('inverse geometry did not converge in 100 iterations')


def _blocks(shape):
    """ZYX slices tiling the grid; grids up to _BLOCK_VOXELS voxels are one block."""
    z, y, x = (int(n) for n in shape)
    if z * y * x <= _BLOCK_VOXELS:
        return [(slice(0, z), slice(0, y), slice(0, x))]
    by, bx = min(y, 256), min(x, 256)
    bz = max(1, min(z, _BLOCK_VOXELS // (by * bx)))
    return [(slice(a, min(a + bz, z)), slice(b, min(b + by, y)), slice(c, min(c + bx, x)))
            for a in range(0, z, bz) for b in range(0, y, by) for c in range(0, x, bx)]


def _inverse_grid(shape, mapping, evaluate, dtype=np.float64):
    """Evaluate evaluate(F_inverse(p)) on every output voxel p, block by block.

    Each block iterates to the same update tolerance and residual limit as
    _inverse; diagnostics report the worst block. float32 results (the
    benchmark tier) stop each point separately (_inverse_per_point); float64
    results (oracle fixtures) iterate whole blocks.
    """
    result = np.empty(tuple(int(n) for n in shape), dtype=dtype)
    per_point = np.dtype(dtype) == np.float32
    inverse = _inverse_per_point if per_point else _inverse
    diagnostics = dict(iterations=0, max_update=0.0, max_residual=0.0, tolerance=1e-10,
                       max_iterations=100, evaluated_on='output_grid', blocks=0,
                       stopping='per_point' if per_point else 'per_block')
    for block in _blocks(shape):
        start = np.array([s.start for s in block], dtype=np.float64)
        dims = tuple(s.stop - s.start for s in block)
        grid = np.moveaxis(np.indices(dims, dtype=np.float64), 0, -1) + start
        reference, diag = inverse(grid, mapping)
        result[block] = evaluate(reference)
        diagnostics.update(iterations=max(diagnostics['iterations'], diag['iterations']),
                           max_update=max(diagnostics['max_update'], diag['max_update']),
                           max_residual=max(diagnostics['max_residual'], diag['max_residual']),
                           blocks=diagnostics['blocks'] + 1)
    return result, diagnostics
