"""Analytic latent backgrounds and independent residual observation streams."""
import hashlib

import numpy as np

from ._formed import (BackgroundConfig, NoiseConfig, TextureConfig, _array,
                      _distribution, _draw, _integer, _json, _placement,
                      _position)


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
    _integer(config.max_count, 'texture max_count', 0, 1024)
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
        _integer(n, 'texture count', 0, config.max_count)
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


def _prepare_background(config, shape, rounds, stream):
    if not isinstance(config, BackgroundConfig):
        raise TypeError('background must be BackgroundConfig')
    flags = _flags(config, ('baseline', 'gradient', 'regions', 'texture'))
    baseline = _parameter(config.baseline, 'baseline', (rounds, 4))
    weights = _parameter(config.tissue_weights, 'tissue_weights', (rounds, 4))
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


def _observe(images, tissue, background, config, channels, stream):
    if not isinstance(config, NoiseConfig):
        raise TypeError('noise must be NoiseConfig')
    flags = _flags(config, ('dependent', 'independent'))
    alpha = float(_parameter(config.alpha, 'alpha', ())) if config.alpha is not None else None
    sigma = float(_parameter(config.sigma, 'sigma', ())) if config.sigma is not None else None
    if alpha is None or sigma is None:
        raise ValueError('noise strengths must be nonnegative scalars')
    alpha = alpha if config.dependent_enabled else 0.0
    sigma = sigma if config.independent_enabled else 0.0
    hashes = {}
    pre_noise_hashes = {}
    for r, (label, image) in enumerate(images.items()):
        with np.errstate(over='ignore', invalid='ignore'):
            image += tissue[..., None] * np.asarray(background['tissue_weights'])[r]
            image += np.asarray(background['baseline'])[r]
        if not np.isfinite(image).all() or (image < 0).any():
            raise ValueError('nonfinite or negative pre-noise total')
        pre_noise_hashes[label] = hashlib.sha256(image.tobytes()).hexdigest()
        for c, channel in enumerate(channels):
            plane = image[..., c]
            # Dependent scale uses J before either residual has been added.
            for kind, enabled, strength in [('dependent', config.dependent_enabled, alpha),
                                             ('independent', config.independent_enabled, sigma)]:
                if enabled:
                    draws = stream('noise.' + kind, None, label, channel).standard_normal(plane.shape)
                    hashes[_json([kind, label, channel])] = hashlib.sha256(draws.tobytes()).hexdigest()
                    with np.errstate(over='ignore', invalid='ignore'):
                        scale = np.sqrt(strength) * np.sqrt(plane) if kind == 'dependent' else strength
                        plane += scale * draws
    effective = dict(type='NoiseConfig', **flags, alpha=alpha, sigma=sigma)
    return dict(noise=effective, standardized_noise_sha256=hashes,
                pre_noise_sha256=pre_noise_hashes, noise_draw_axes='ZYX/C-order',
                background_frame='reference', baseline_frame='destination'), effective
