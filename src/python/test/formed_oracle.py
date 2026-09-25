"""Independent formed-scene oracle: stream keys and the development fixtures.

Imports nothing from starfinder.synthetic. Constants restate the documented
development preset; draws come from the documented SHA-256/PCG64 stream key,
and the geometry inverse is a scalar bisection, not the producer's fixed point.
"""
import hashlib
import json
import math
import unicodedata

import numpy as np
import pandas as pd

NAMESPACE = 'starfinder.synthetic/1'
COMPONENTS = (
    'count', 'placement', 'identity', 'brightness', 'width.axial',
    'width.lateral', 'elongation', 'angle', 'round.dropout', 'round.weakening',
    'round.loss', 'geometry.translation', 'geometry.local', 'background.count',
    'background.placement', 'background.width', 'background.brightness',
    'noise.dependent', 'noise.independent',
)
SIZES = {'z1': (1, 32, 32), 'small': (9, 32, 32)}
FIXTURES = ('z1-clean', 'z1-combined', 'small-clean', 'small-combined')
FACTORS = ('brightness', 'axial_width', 'lateral_width', 'elongation', 'placement',
           'dropout', 'weakening', 'trend', 'loss', 'gain', 'mixing', 'baseline',
           'gradient', 'regions', 'texture', 'dependent_noise', 'independent_noise',
           'translation', 'local')
COMBINED = set(FACTORS[5:]) - {'dropout', 'loss'}
ROUNDS = ('round10', 'round2', 'round1')
CHANNELS = ('ch02', 'ch00', 'ch03', 'ch01')
IDS = ('gt-A', 'gt-B')
# Persistent shape literals: gt-B is elongated and rotated (theta != 0).
ELONGATION = (1.0, 1.5)
THETA = (0.0, math.pi / 6)


def stream_descriptor(component, *, split='development', seed=42, scene='formed-v1',
                      entity=None, round_label=None, channel_label=None):
    """Reference UTF-8 compact-JSON descriptor bytes for one stream key."""
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError('seed must be an unsigned 64-bit integer')
    if component not in COMPONENTS:
        raise ValueError('unknown component')
    for label in (split, scene):
        if not isinstance(label, str) or not label or unicodedata.normalize('NFC', label) != label:
            raise ValueError('split and scene must be nonempty NFC strings')
    for label in (entity, round_label, channel_label):
        if label is not None and (not isinstance(label, str) or not label
                                  or unicodedata.normalize('NFC', label) != label):
            raise ValueError('labels must be nonempty NFC strings or null')
    return json.dumps([NAMESPACE, split, seed, scene, component, entity, round_label, channel_label],
                      ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')


def stream_digest(component, **keys):
    return hashlib.sha256(stream_descriptor(component, **keys)).hexdigest()


def generator(component, **keys):
    return np.random.Generator(np.random.PCG64(int.from_bytes(
        hashlib.sha256(stream_descriptor(component, **keys)).digest(), 'big')))


def rng(component, entity=None, round_label=None, channel=None):
    return generator(component, scene='controlled-development-v1', entity=entity,
                     round_label=round_label, channel_label=channel)


def inverse_grid(points, center, vector, translation):
    """Scalar bisection for the preset's single RBF control per round.

    q = p - t - v*a with a = exp(-||q-c||²/128); a is bracketed by [0, 1].
    """
    base = points - translation
    if not np.any(vector):
        return base
    lo = np.zeros(points.shape[:-1])
    hi = np.ones_like(lo)
    for _ in range(55):
        mid = (lo + hi) / 2
        q = base - mid[..., None] * vector
        residual = mid - np.exp(-np.sum((q - center)**2, axis=-1) / 128)
        hi = np.where(residual >= 0, mid, hi)
        lo = np.where(residual < 0, mid, lo)
    result = base - ((lo + hi) / 2)[..., None] * vector
    residual = result + np.exp(-np.sum((result - center)**2, axis=-1) / 128)[..., None] * vector \
        + translation - points
    assert np.max(np.abs(residual)) <= 2e-10
    return result


def kernel_radius(points, center, sz, sl, e, theta):
    """Squared normalized radius of the rotated ellipsoidal Gaussian."""
    d = points - center
    c, s = math.cos(theta), math.sin(theta)
    u = (c * d[..., 1] + s * d[..., 2]) / (sl * e)
    v = (-s * d[..., 1] + c * d[..., 2]) / sl
    return (d[..., 0] / sz)**2 + u**2 + v**2


def expected_case(size, condition):
    """Full-grid float64 images, signals and literal truth for one preset."""
    if condition == 'placement':
        raise ValueError('placement draws are not part of this oracle')
    shape = SIZES[size]
    factors = COMBINED if condition == 'combined' else set() if condition == 'clean' else {condition}
    anchor = np.array([shape[0] // 2, 10, 10], dtype=float)
    centers = np.array([anchor, [shape[0] // 2, 22, 22]])
    amplitude = 16 if 'brightness' in factors else 8
    sz = 1.5 if 'axial_width' in factors else 1.0
    sl = 2.0 if 'lateral_width' in factors else 1.25
    elongation = (2.0, 2.0) if 'elongation' in factors else ELONGATION
    intended = np.zeros((2, 4, 3))
    for i, active in enumerate(((1, 0, 3), (0, 1, 2))):
        intended[i, active, np.arange(3)] = amplitude
    pre_mix = intended.copy()
    if 'trend' in factors:
        pre_mix *= [1, .5, .25]
    if 'weakening' in factors:
        pre_mix *= [1, .25, 1]
    if 'dropout' in factors:
        pre_mix *= [1, 0, 1]
    if 'loss' in factors:
        pre_mix *= [1, 0, 0]
    if 'gain' in factors:
        pre_mix *= .5
    realized = pre_mix.copy()
    if 'mixing' in factors:
        realized[:, 0] += .25 * pre_mix[:, 1]
        realized[:, 2] += .25 * pre_mix[:, 3]
    points = np.moveaxis(np.indices(shape, dtype=float), 0, -1)
    upper = np.array(shape) - 1
    images, histories = {}, []
    for r, label in enumerate(ROUNDS):
        translation = np.array(((0, 0, 0), (0, .5, -.5), (0, -1, 1))[r], float) \
            if 'translation' in factors else np.zeros(3)
        vector = np.array(((0, 0, 0), (0, 0, .5), (0, .25, 0))[r], float) \
            if 'local' in factors else np.zeros(3)
        moved = centers + np.exp(-np.sum((centers - anchor)**2, axis=1) / 128)[:, None] * vector + translation
        image = np.zeros((*shape, 4))
        for i, center in enumerate(moved):
            e, theta = elongation[i], THETA[i]
            radius = kernel_radius(points, center, sz, sl, e, theta)
            image += np.where(radius <= 16, np.exp(-radius / 2), 0)[..., None] * realized[i, :, r]
            extent = 4 * np.array([sz, sl * math.hypot(e * math.cos(theta), math.sin(theta)),
                                   sl * math.hypot(e * math.sin(theta), math.cos(theta))])
            histories.append(dict(
                amplicon_id=IDS[i], round_label=label, round_index=r,
                z=center[0], y=center[1], x=center[2],
                dropped='dropout' in factors and r == 1,
                weakened='weakening' in factors and r == 1,
                lost='loss' in factors and r >= 1,
                emitting=bool(realized[i, :, r].any()),
                center_in_bounds=bool(np.all(center >= 0) and np.all(center <= upper)),
                support_intersects=bool(np.any(radius <= 16)),
                support_truncated=bool(np.any(center - extent < 0) or np.any(center + extent > upper)),
                trend_multiplier=.5**r if 'trend' in factors else 1.0,
                weak_multiplier=.25 if 'weakening' in factors and r == 1 else 1.0))
        reference = inverse_grid(points, anchor, vector, translation)
        background = np.zeros(shape)
        if 'gradient' in factors:
            background += np.maximum(0, 1 + 2 * reference[..., 2] / (shape[2] - 1))
        if 'regions' in factors:
            background += 3 * np.exp(-.5 * np.sum(((reference - anchor) / [2, 5, 5])**2, axis=-1))
        if 'texture' in factors:
            for i in range(2):
                center = rng('background.placement', f'blob-{i}').uniform(0, upper, 3)
                background += 5 * np.exp(-.5 * np.sum(((reference - center) / [1, 3, 3])**2, axis=-1))
        image += background[..., None] * [1, .5, .25, 0]
        if 'baseline' in factors:
            image += [1, 2, 3, 4]
        for c, channel in enumerate(CHANNELS):
            if 'dependent_noise' in factors:
                image[..., c] += np.sqrt(.25 * image[..., c]) * rng(
                    'noise.dependent', round_label=label, channel=channel).standard_normal(shape)
            if 'independent_noise' in factors:
                image[..., c] += .5 * rng(
                    'noise.independent', round_label=label, channel=channel).standard_normal(shape)
        images[label] = image
    formed = pd.DataFrame(dict(
        amplicon_id=IDS, gene_id=('gene-A', 'gene-B'), codeword=('123', '214'),
        formed_index=(0, 1), z=centers[:, 0], y=centers[:, 1], x=centers[:, 2],
        A=[float(amplitude)] * 2, sz=[sz] * 2, sl=[sl] * 2, e=elongation, theta=THETA))
    return dict(images=images, formed=formed, intended=intended, pre_mix=pre_mix,
                realized=realized, histories=pd.DataFrame(histories))
