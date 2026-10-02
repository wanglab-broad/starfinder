"""Correlated noise (analytic kernel oracle), stream isolation, codebooks and calibrated presets (W-241).

The correlated field is reproduced here from the documented definition: the
keyed noise.correlated stream, a white draw on the grid padded by the kernel
radius, and a separable Gaussian kernel exp(-k^2/L^2) truncated at
ceil(4L/sqrt(2)) and scaled to unit sum of squares. No producer helper is used.
"""
from collections import Counter
from dataclasses import replace
import hashlib
import json
import math

import numpy as np
import pandas as pd
import pytest

from starfinder.synthetic import (BackgroundConfig, CALIBRATED_CONDITIONS, NoiseConfig, ScalarDistribution,
                                  TextureConfig, calibrated_scene_preset, development_codebook,
                                  development_scene_preset, formed_scene_preset, generate_formed_scene)

pytestmark = pytest.mark.synthetic

LENGTHS = (1.0, 3.0, 2.0)


def kernel(length):
    radius = math.ceil(4 * length / math.sqrt(2))
    k = np.arange(-radius, radius + 1, dtype=float)
    w = np.exp(-k ** 2 / length ** 2) if length > 0 else (k == 0).astype(float)
    return w / np.sqrt(np.sum(w ** 2))


def autocorrelation(w, lag):
    """rho(lag) of the unit-variance field smoothed by w (sum of w^2 is 1)."""
    return float(np.sum(w[:w.size - lag] * w[lag:])) if lag < w.size else 0.0


def keyed(component, round_label, channel, *, split='development', seed=42, scene_key='formed-v1'):
    descriptor = ['starfinder.synthetic/1', split, seed, scene_key, component, None, round_label, channel]
    digest = hashlib.sha256(json.dumps(descriptor, ensure_ascii=False, separators=(',', ':')).encode()).digest()
    return np.random.Generator(np.random.PCG64(int.from_bytes(digest, 'big')))


def expected_field(round_label, channel, shape, lengths):
    weights = [kernel(length) for length in lengths]
    field = keyed('noise.correlated', round_label, channel).standard_normal(
        tuple(n + w.size - 1 for n, w in zip(shape, weights)))
    for axis, w in enumerate(weights):
        field = np.apply_along_axis(lambda v: np.convolve(v, w, mode='valid'), axis, field)
    assert field.shape == tuple(shape)
    return field


def scene(noise, shape=(32, 64, 64), **kwargs):
    book, config = formed_scene_preset()
    return generate_formed_scene(book, config=replace(config, shape_zyx=shape, dtype='float64', noise=noise,
                                                      **{'count': 0, **kwargs}))


def test_correlated_disabled_by_default_and_draws_nothing():
    assert NoiseConfig().correlated_enabled is False
    base = scene(NoiseConfig(True, 2, True, 1), shape=(3, 7, 9), count=4)
    disabled = scene(NoiseConfig(True, 2, True, 1, correlated_sigma=5, correlation_length_zyx=(2, 3, 4)),
                     shape=(3, 7, 9), count=4)
    for label in base.round_labels:
        np.testing.assert_array_equal(base.rounds[label], disabled.rounds[label])
    assert base.provenance['stream_scheme'] == disabled.provenance['stream_scheme']
    assert base.provenance['image_sha256'] == disabled.provenance['image_sha256']
    effective = disabled.provenance['effective_config']['noise']
    assert effective['correlated_enabled'] is False and effective['correlated_sigma'] == 0
    assert effective['correlation_length_zyx'] == [2, 3, 4]
    requested = disabled.provenance['requested_config']['noise']
    assert requested['correlated_sigma'] == 5 and requested['correlation_length_zyx'] == [2, 3, 4]


@pytest.mark.parametrize('bad', [dict(correlated_sigma=-1), dict(correlation_length_zyx=(1, 2)),
                                 dict(correlation_length_zyx=(1, -1, 1)), dict(correlated_enabled=1),
                                 dict(correlation_length_zyx=(1, np.inf, 1)), dict(correlated_sigma=True)])
def test_correlated_parameters_are_validated(bad):
    with pytest.raises(ValueError):
        scene(NoiseConfig(**bad), shape=(1, 4, 4))


def test_correlated_variance_and_length_match_the_kernel():
    """Variance sigma^2 and per-axis autocorrelation rho_a(d) of the separable kernel.

    With A = prod_a sum_d rho_a(d)^2, the zero-mean variance estimator mean(x^2)
    has standard error at most sigma^2 sqrt(2A/N) over N voxels (Isserlis; edge
    terms only reduce it). A lag-d product mean has standard error at most
    sigma^2 sqrt(2A/N_d), so the autocorrelation ratio is within
    sqrt(2A/N_d) + rho sqrt(2A/N) of rho. Every tolerance is five of these
    standard errors. rho_a(L_a) of the discrete kernel equals exp(-1/2), the
    continuous value that defines L as the correlation length, within the
    Poisson-summation bound for sampling Gaussians of SD L/2 on the integers:
    relative error (1 + 2e)/(1 - 2e) - 1 with e = sum_m exp(-2 pi^2 m^2 L^2/4),
    plus 1e-6 for the four-sigma truncation.
    """
    sigma = 3.0
    result = scene(NoiseConfig(correlated_enabled=True, correlated_sigma=sigma, correlation_length_zyx=LENGTHS))
    fields = np.stack([result.rounds[label][..., c] for label in result.round_labels for c in range(4)])
    n = fields.size
    weights = [kernel(length) for length in LENGTHS]
    area = math.prod(sum(autocorrelation(w, abs(d)) ** 2 for d in range(-w.size + 1, w.size)) for w in weights)
    variance = float(np.mean(fields ** 2))
    se = math.sqrt(2 * area / n)
    assert abs(variance / sigma ** 2 - 1) <= 5 * se, (variance, se)
    for axis, (length, w) in enumerate(zip(LENGTHS, weights)):
        aliasing = sum(math.exp(-2 * math.pi ** 2 * m ** 2 * length ** 2 / 4) for m in range(1, 20))
        bound = math.exp(-.5) * ((1 + 2 * aliasing) / (1 - 2 * aliasing) - 1) + 1e-6
        assert autocorrelation(w, round(length)) == pytest.approx(math.exp(-.5), abs=bound)
        for lag in sorted({1, round(length), 2 * round(length)}):
            head = fields[(slice(None),) * (axis + 1) + (slice(None, -lag),)]
            tail = fields[(slice(None),) * (axis + 1) + (slice(lag, None),)]
            estimate = float(np.mean(head * tail)) / variance
            expected = autocorrelation(w, lag)
            tolerance = 5 * (math.sqrt(2 * area / head.size) + expected * se)
            assert abs(estimate - expected) <= tolerance, (axis, lag, estimate, expected, tolerance)


def test_correlated_field_matches_independent_reconstruction():
    result = scene(NoiseConfig(correlated_enabled=True, correlated_sigma=1.5, correlation_length_zyx=(0, 2, 1)),
                   shape=(3, 12, 10))
    for label in result.round_labels:
        for c, channel in enumerate(result.channel_labels):
            np.testing.assert_allclose(result.rounds[label][..., c],
                                       1.5 * expected_field(label, channel, (3, 12, 10), (0, 2, 1)),
                                       rtol=0, atol=1e-12)
    streams = result.provenance['stream_scheme']['streams']
    assert {s[4] for s in streams} == {'noise.correlated'} and len(streams) == 12


def test_enabling_correlated_changes_no_other_draws():
    """Signal, background, dependent (Poisson) and read noise are identical with the term on or off.

    The on-off difference of every voxel is exactly sigma times the independently
    reconstructed correlated field, so every other component is unchanged.
    """
    background = BackgroundConfig(baseline_enabled=True, baseline=[[3, 4, 5, 6]] * 3, gradient_enabled=True,
                                  gradient_slopes_zyx=(0, 1, 2), texture_enabled=True, texture=TextureConfig(count=3),
                                  tissue_weights=np.ones((3, 4)))
    other = NoiseConfig(dependent_enabled=True, alpha=.5, model='poisson', independent_enabled=True, sigma=.7)
    kwargs = dict(shape=(4, 16, 16), count=6, background=background, brightness=ScalarDistribution(parameters=(40,)))
    off = scene(other, **kwargs)
    on = scene(replace(other, correlated_enabled=True, correlated_sigma=2.0, correlation_length_zyx=(1, 2, 2)),
               **kwargs)
    pd.testing.assert_frame_equal(off.formed, on.formed)
    pd.testing.assert_frame_equal(off.round_truth, on.round_truth)
    for name in ('intended', 'pre_mix', 'realized'):
        np.testing.assert_array_equal(getattr(off, name), getattr(on, name))
    assert (off.provenance['effective_config']['background'] == on.provenance['effective_config']['background'])
    streams_off = off.provenance['stream_scheme']['streams']
    streams_on = on.provenance['stream_scheme']['streams']
    assert [s for s in streams_on if s[4] != 'noise.correlated'] == streams_off
    assert len(streams_on) - len(streams_off) == 12
    for label in off.round_labels:
        for c, channel in enumerate(off.channel_labels):
            np.testing.assert_allclose(on.rounds[label][..., c] - off.rounds[label][..., c],
                                       2.0 * expected_field(label, channel, (4, 16, 16), (1, 2, 2)),
                                       rtol=0, atol=1e-9)


def test_existing_presets_do_not_enable_the_correlated_term():
    for condition in ('clean', 'combined', 'dependent_noise', 'independent_noise'):
        assert development_scene_preset(condition)[1].noise.correlated_enabled is False


def counts(book):
    return [Counter(sequence[r] for sequence in book.table.color_sequence) for r in range(len(book.round_labels))]


def test_balanced_codebook():
    book = development_codebook()
    assert book.round_labels == ('round1', 'round2', 'round3', 'round4')
    assert len(book.channel_labels) == 4 and book.n_genes >= 16
    for round_counts in counts(book):
        assert set(round_counts) == set('1234')              # every channel occupied in every round
        assert max(round_counts.values()) - min(round_counts.values()) <= 1
    sequences = list(book.table.color_sequence)
    assert min(sum(a != b for a, b in zip(s, t)) for i, s in enumerate(sequences) for t in sequences[i + 1:]) >= 3
    assert all(gene.startswith('balanced-') for gene in book.genes)


def test_unbalanced_codebook_is_labelled_and_unbalanced():
    book = development_codebook('unbalanced')
    assert all(gene.startswith('unbalanced-') for gene in book.genes)
    assert book.n_genes == 16 and len(book.round_labels) == 4
    for r, round_counts in enumerate(counts(book)):
        assert sorted(round_counts.values()) == [1, 3, 5, 7]
        assert round_counts[str(r + 1)] == 7
    with pytest.raises(ValueError):
        development_codebook('other')


def test_calibrated_uint16_is_sixteen_times_uint8():
    for condition in CALIBRATED_CONDITIONS:
        _, low = calibrated_scene_preset(condition, 'uint8')
        _, high = calibrated_scene_preset(condition, 'uint16')
        assert (low.shape_zyx, low.count, low.seed, low.scene_key) == (high.shape_zyx, high.count, 0, high.scene_key)
        mu8, sd8 = low.brightness.parameters
        mu16, sd16 = high.brightness.parameters
        assert sd8 == sd16 and mu16 - mu8 == pytest.approx(math.log(16))
        assert high.readout == low.readout and high.geometry == low.geometry
        for name in ('alpha', 'sigma', 'correlated_sigma'):
            assert getattr(high.noise, name) == pytest.approx(16 * getattr(low.noise, name))
        assert high.noise.correlation_length_zyx == low.noise.correlation_length_zyx
        np.testing.assert_allclose(high.background.baseline, 16 * np.asarray(low.background.baseline))
        np.testing.assert_allclose(high.background.region_heights, 16 * np.asarray(low.background.region_heights))
        np.testing.assert_allclose(high.background.gradient_slopes_zyx,
                                   16 * np.asarray(low.background.gradient_slopes_zyx))
        assert high.background.tissue_weights == low.background.tissue_weights
        mu8, _ = low.background.texture.brightness.parameters
        mu16, _ = high.background.texture.brightness.parameters
        assert mu16 - mu8 == pytest.approx(math.log(16))


def test_calibrated_conditions_carry_the_baseline_and_add_factors():
    _, clean = calibrated_scene_preset('clean')
    noise = clean.noise
    assert noise.dependent_enabled and noise.independent_enabled and noise.correlated_enabled
    assert clean.brightness.mode == 'lognormal'
    assert clean.readout.gain_enabled and clean.readout.trend_enabled and clean.background.baseline_enabled
    assert len(set(np.asarray(clean.readout.gains)[0])) == 4     # non-uniform channel gains
    assert clean.readout.trend_base < 1
    for condition, factors in CALIBRATED_CONDITIONS.items():
        _, config = calibrated_scene_preset(condition, seed=3)
        assert config.noise == noise and config.brightness == clean.brightness and config.seed == 3
        assert config.shape_zyx == clean.shape_zyx and config.scene_key == clean.scene_key
        for name in ('weakening', 'mixing'):
            assert getattr(config.readout, name + '_enabled') == (name in factors)
        for name in ('gradient', 'regions', 'texture'):
            assert getattr(config.background, name + '_enabled') == (name in factors)
        assert config.geometry.translation_enabled == ('translation' in factors)
        assert config.geometry.local_enabled == ('local' in factors)
        assert (config.background.baseline != clean.background.baseline) == ('baseline' in factors)
        assert (config.readout.gains != clean.readout.gains) == bool({'gain', 'trend'} & set(factors))
        assert (config.readout.trend_base != clean.readout.trend_base) == ('trend' in factors)
    with pytest.raises(ValueError):
        calibrated_scene_preset('unknown')
    with pytest.raises(ValueError):
        calibrated_scene_preset('clean', 'float32')


@pytest.mark.parametrize('condition', ['clean', 'combined_geometry'])
def test_calibrated_generation_stays_within_bounds(condition):
    book, config = calibrated_scene_preset(condition, 'uint8', codebook='unbalanced')
    shape = config.shape_zyx
    assert shape[0] <= 32 and shape[1] <= 64 and shape[2] <= 64 and config.count <= 80
    with pytest.warns(RuntimeWarning, match='clipped'):
        result = generate_formed_scene(book, config=config)
    assert len(result.rounds) == 4 and len(result.channel_labels) == 4
    for image in result.rounds.values():
        assert image.shape == (*shape, 4) and image.dtype == np.uint8
    assert result.provenance['effective_config']['noise']['correlated_enabled'] is True
    assert set(result.formed.gene_id) <= set(book.genes)
