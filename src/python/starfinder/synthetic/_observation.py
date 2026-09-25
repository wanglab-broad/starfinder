"""Analytic latent backgrounds and independent residual observation streams."""
from dataclasses import dataclass, field

import numpy as np

from ._common import (ScalarDistribution, _array, _distribution, _draw, _integer,
                      _json, _placement, _position)


@dataclass(frozen=True)
class TextureConfig:
    """Persistent Gaussian blobs, independent of formed molecules.

    Count/density/coordinates and placement follow FormedSceneConfig. Default
    enabled count is four. Width distributions are axial/lateral (YX shared),
    in voxel indices; brightness is peak intensity. Supplied values use blob-N
    IDs. Enable this component with BackgroundConfig.texture_enabled.
    max_count is a user-settable allocation guard, not a model parameter.
    Instances are not hashable (see FormedSceneConfig).
    """

    __hash__ = None

    count: int | None = None
    density: float | None = None
    coordinates: object = None
    max_count: int = 1024
    placement: str = "uniform"
    spatial_weights: object = None
    cluster_centers: object = None
    cluster_weights: object = None
    spread_zyx: tuple[float, float, float] = (1, 2, 2)
    axial_width: ScalarDistribution = field(default_factory=ScalarDistribution)
    lateral_width: ScalarDistribution = field(default_factory=lambda: ScalarDistribution(parameters=(3,)))
    brightness: ScalarDistribution = field(default_factory=lambda: ScalarDistribution(parameters=(5,)))


@dataclass(frozen=True)
class BackgroundConfig:
    """Analytic reference background and destination-frame R×C baseline.

    All flags default false. Gradient is max(0, intercept + slopes dot
    normalized ZYX). Regions are supplied K×3 centers, positive sigma_zyx
    triples and nonnegative heights (None gives (2,8,8), 10 per region).
    Tissue weights and baselines are nonnegative R×C arrays, default zero
    (C is the codebook channel count).
    Disabled parameters are validated and retained, but contribute zero.
    Instances are not hashable (see FormedSceneConfig).
    """

    __hash__ = None

    baseline_enabled: bool = False
    baseline: object = None
    gradient_enabled: bool = False
    gradient_intercept: float = 0.0
    gradient_slopes_zyx: tuple[float, float, float] = (0, 0, 0)
    regions_enabled: bool = False
    region_centers: object = ()
    region_sigma_zyx: object = None
    region_heights: object = None
    texture_enabled: bool = False
    texture: TextureConfig = field(default_factory=TextureConfig)
    tissue_weights: object = None


@dataclass(frozen=True)
class NoiseConfig:
    """Separate residuals: signal-dependent first, then sigma*Z_ind (read noise).

    ``model`` selects the dependent residual: "gaussian" adds sqrt(alpha*J)*Z_dep;
    "poisson" replaces J by alpha*Poisson(J/alpha), with the same mean and
    variance (alpha is intensity per detected count). J includes signal,
    weighted tissue and baseline. Both flags default false; alpha/sigma are
    finite nonnegative scalars (zero by default). Draws come from the
    round/channel keyed noise streams; Gaussian standardized draws persist when
    strengths change. No calibrated photon claim.
    Instances are not hashable (see FormedSceneConfig).
    """

    __hash__ = None

    dependent_enabled: bool = False
    alpha: float = 0.0
    independent_enabled: bool = False
    sigma: float = 0.0
    model: str = "gaussian"


def _parameter(value, name, shape, default=0, *, positive=False):
    result = np.full(shape, default, dtype=np.float64) if value is None else _array(value, name)
    if result.shape != shape or (result <= 0 if positive else result < 0).any():
        raise ValueError(f"invalid {name}: expected {shape} and {'positive' if positive else 'nonnegative'} values")
    return result


def _flags(config, names):
    flags = {name + '_enabled': getattr(config, name + '_enabled') for name in names}
    if any(type(v) is not bool for v in flags.values()):
        raise ValueError('observation enable flags must be Boolean')
    return flags


def _centers(value, name):
    points = _array(value, name)
    if points.shape == (0,):
        points = points.reshape(0, 3)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f'{name} must be N by 3')
    return points


def _texture(config, shape, enabled, stream):
    if not isinstance(config, TextureConfig):
        raise TypeError('texture must be TextureConfig')
    _integer(config.max_count, 'texture max_count', 0)
    if sum(x is not None for x in (config.count, config.density, config.coordinates)) > 1:
        raise ValueError('select texture coordinates/count/density')
    weights, centers, spread = _placement(config, shape)
    points = None if config.coordinates is None else _centers(config.coordinates, 'texture coordinates')
    if points is not None and shape[0] == 1 and np.any(points[:, 0] != 0):
        raise ValueError('Z=1 requires texture z=0')
    if config.density is not None:
        density = _parameter(config.density, 'texture density', ())
        mean = float(density) * int(shape.prod())
        if not np.isfinite(mean):
            raise ValueError('nonfinite expected texture count')
        n = int(stream('background.count').poisson(mean)) if enabled else 0
    else:
        n = len(points) if points is not None else (4 if config.count is None else config.count)
        _integer(n, 'texture count', 0)
    if n > config.max_count:
        raise ValueError('texture count exceeds max_count')
    ids = tuple(f'blob-{i}' for i in range(n))
    properties = [('width.axial', config.axial_width), ('width.lateral', config.lateral_width),
                  ('brightness', config.brightness)]
    for name, spec in properties:
        _distribution(spec, name, ids)
    if not enabled:
        return []

    def property_stream(name, identity):
        if name.startswith('width.'):
            return stream('background.width', _json([identity, name.split('.')[1]]))
        return stream('background.brightness', identity)

    result = []
    for i, identity in enumerate(ids):
        point = points[i] if points is not None else _position(
            config, shape, stream('background.placement', identity), weights, centers, spread)
        sz, sl, height = [_draw(spec, name, identity, property_stream) for name, spec in properties]
        if not np.isfinite([*point, sz, sl, height]).all() or min(sz, sl) <= 0:
            raise ValueError('nonfinite or invalid texture properties')
        result.append(dict(component_id=identity, kind='texture', center_zyx=point.tolist(),
                           sigma_zyx=[sz, sl, sl], height=height))
    return result


def _prepare_background(config, shape, rounds, stream, channels=4):
    if not isinstance(config, BackgroundConfig):
        raise TypeError('background must be BackgroundConfig')
    flags = _flags(config, ('baseline', 'gradient', 'regions', 'texture'))
    baseline = _parameter(config.baseline, 'baseline', (rounds, channels))
    weights = _parameter(config.tissue_weights, 'tissue_weights', (rounds, channels))
    intercept = _parameter(_array(config.gradient_intercept, 'gradient_intercept'), 'gradient_intercept', ())
    slopes = _array(config.gradient_slopes_zyx, 'gradient_slopes_zyx')
    if slopes.shape != (3,):
        raise ValueError('gradient slopes must be ZYX')
    centers = _centers(config.region_centers, 'region_centers')
    if shape[0] == 1 and np.any(centers[:, 0] != 0):
        raise ValueError('Z=1 requires region z=0')
    widths = _parameter(np.tile([2, 8, 8], (len(centers), 1)) if config.region_sigma_zyx is None
                        else config.region_sigma_zyx, 'region_sigma_zyx', (len(centers), 3), positive=True)
    heights = _parameter(config.region_heights, 'region_heights', (len(centers),), 10)
    components = []
    if config.gradient_enabled:
        components.append(dict(component_id='gradient', kind='gradient', intercept=float(intercept),
                               slopes_zyx=slopes.tolist(), shape_zyx=shape.tolist()))
    if config.regions_enabled:
        components.extend(dict(component_id=f'region-{i}', kind='region', center_zyx=p.tolist(),
                               sigma_zyx=w.tolist(), height=float(h))
                          for i, (p, w, h) in enumerate(zip(centers, widths, heights)))
    components.extend(_texture(config.texture, shape, config.texture_enabled, stream))
    # Latents are analytic, reference-frame records; no noisy image is sampled.
    for component in components:
        component.update(units='voxel_index/intensity', frame='reference', support='untruncated')
    effective = dict(type='BackgroundConfig', **flags,
                     baseline=(baseline if config.baseline_enabled else np.zeros_like(baseline)).tolist(),
                     tissue_weights=(weights if components else np.zeros_like(weights)).tolist(),
                     components=components)
    return components, effective


def evaluate_background(components, coordinates):
    """Evaluate generated latent records at real reference coordinates (..., ZYX).

    Geometry callers supply inverse-mapped output coordinates. Gaussian tails
    and extrapolated normalized gradients have no periodic wrap or image padding.
    This internal evaluator consumes records validated by _prepare_background.
    """
    result = np.zeros(coordinates.shape[:-1], dtype=np.float64)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        for component in components:
            if component['kind'] == 'gradient':
                shape = np.asarray(component['shape_zyx'])
                normalized = coordinates / np.maximum(shape - 1, 1)
                normalized = np.where(shape == 1, 0, normalized)
                result += np.maximum(0, component['intercept'] + normalized @ component['slopes_zyx'])
            else:
                delta = (coordinates - component['center_zyx']) / component['sigma_zyx']
                result += component['height'] * np.exp(-.5 * np.sum(delta * delta, axis=-1))
    if not np.isfinite(result).all():
        raise ValueError('nonfinite structured background')
    return result


def _noise(config):
    """Validate NoiseConfig once; return the effective record used by every round."""
    if not isinstance(config, NoiseConfig):
        raise TypeError('noise must be NoiseConfig')
    flags = _flags(config, ('dependent', 'independent'))
    if config.model not in ('gaussian', 'poisson'):
        raise ValueError('noise model must be gaussian or poisson')
    alpha = float(_parameter(config.alpha, 'alpha', ())) if config.alpha is not None else None
    sigma = float(_parameter(config.sigma, 'sigma', ())) if config.sigma is not None else None
    if alpha is None or sigma is None:
        raise ValueError('noise strengths must be nonnegative scalars')
    alpha = alpha if config.dependent_enabled else 0.0
    sigma = sigma if config.independent_enabled else 0.0
    return dict(type='NoiseConfig', **flags, alpha=alpha, sigma=sigma, model=config.model)


def evaluate_background_translated(components, shape, translation, dtype=np.float64):
    """Evaluate latent records on a grid moved by a pure translation, plane by plane.

    Reference coordinates are p - t on every axis, so each component is
    separable: gradients are sums and Gaussians are outer products of per-axis
    factors. Equal to evaluate_background on the same coordinates up to
    floating-point reassociation; returns the array and the maximum axis residual.
    """
    shape = tuple(int(n) for n in shape)
    axes = [np.arange(n, dtype=np.float64) - t for n, t in zip(shape, translation)]
    residual = max(float(np.max(np.abs(a + t - np.arange(len(a))), initial=0))
                   for a, t in zip(axes, translation))
    result = np.empty(shape, dtype=dtype)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore', under='ignore'):
        terms = []
        for component in components:
            if component['kind'] == 'gradient':
                size = np.asarray(component['shape_zyx'])
                factors = [np.where(n == 1, 0, a / max(n - 1, 1)) * slope
                           for a, n, slope in zip(axes, size, component['slopes_zyx'])]
                terms.append(('gradient', component['intercept'], factors))
            else:
                factors = [np.exp(-.5 * ((a - c) / w) ** 2)
                           for a, c, w in zip(axes, component['center_zyx'], component['sigma_zyx'])]
                terms.append(('gaussian', component['height'], factors))
        for z in range(shape[0]):
            plane = np.zeros(shape[1:], dtype=np.float64)
            for kind, value, (fz, fy, fx) in terms:
                if kind == 'gradient':
                    plane += np.maximum(0, value + fz[z] + fy[:, None] + fx[None, :])
                else:
                    plane += value * fz[z] * (fy[:, None] * fx[None, :])
            result[z] = plane
    if not np.isfinite(result).all():
        raise ValueError('nonfinite structured background')
    return result, residual


# Noise draws per call; chunked draws equal one full-plane draw in C order.
_NOISE_CHUNK = 1 << 22


def _observe(plane, tissue, background, r, c, noise, label, channel, stream):
    """Add tissue/baseline and residual noise to channel c of round r, in place.

    plane is one contiguous ZYX accumulation plane; tissue is the reference
    background sampled on this round's grid, or None without latent components.
    Draws come from the (round, channel) keyed streams in flat chunks, so
    rounds and channels may be generated one at a time.
    """
    with np.errstate(over='ignore', invalid='ignore'):
        if tissue is not None:
            plane += tissue * np.asarray(background['tissue_weights'])[r, c]
        plane += np.asarray(background['baseline'])[r, c]
    if not np.isfinite(plane).all() or (plane < 0).any():
        raise ValueError('nonfinite or negative pre-noise total')
    alpha, sigma = noise['alpha'], noise['sigma']
    dependent = stream('noise.dependent', None, label, channel) if noise['dependent_enabled'] else None
    independent = stream('noise.independent', None, label, channel) if noise['independent_enabled'] else None
    flat = plane.reshape(-1)
    for start in range(0, flat.size, _NOISE_CHUNK):
        part = flat[start:start + _NOISE_CHUNK]
        with np.errstate(over='ignore', invalid='ignore'):
            # Dependent scale uses J before either residual has been added.
            if dependent is not None and noise['model'] == 'poisson':
                if alpha > 0:
                    part[...] = alpha * dependent.poisson(part / alpha)
            elif dependent is not None:
                part += np.sqrt(alpha) * np.sqrt(part) * dependent.standard_normal(part.size)
            if independent is not None:
                part += sigma * independent.standard_normal(part.size)
