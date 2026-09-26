"""Preset isolation and literal history expectations, independent of rendering."""
from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from starfinder.synthetic import (DEVELOPMENT_FACTORS, DEVELOPMENT_FIXTURES, DEVELOPMENT_SIZES,
    development_preset_factors, development_scene_preset, generate_formed_scene)


def scene(condition='clean', size='small'):
    book, config = development_scene_preset(condition, size=size)
    return generate_formed_scene(book, config=config)


def ext(value):
    return value.provenance


def components(value):
    return value.provenance['effective_config']['background']['components']


@pytest.mark.parametrize('factor', DEVELOPMENT_FACTORS)
@pytest.mark.parametrize('size', DEVELOPMENT_SIZES)
def test_single_factor_preserves_unrelated_latents(factor, size):
    clean, changed = scene(size=size), scene(factor, size)
    columns = {'brightness': ['A'], 'axial_width': ['sz'],
               'lateral_width': ['sl'], 'elongation': ['e'],
               'placement': ['z', 'y', 'x']}.get(factor, [])
    pd.testing.assert_frame_equal(clean.formed.drop(columns=['namespace', 'frame_id', *columns]),
                                  changed.formed.drop(columns=['namespace', 'frame_id', *columns]), check_exact=True)
    if factor != 'brightness':
        np.testing.assert_array_equal(clean.intended, changed.intended)
    flags = {k.removesuffix('_enabled') for group in ('readout','background','noise','geometry')
             for k, v in ext(changed)['effective_config'][group].items() if k.endswith('_enabled') and v}
    expected = {'dependent_noise': 'dependent', 'independent_noise': 'independent'}.get(factor, factor)
    assert flags == ({expected} if factor not in DEVELOPMENT_FACTORS[:5] else set())
    assert development_preset_factors(factor) == (factor,)
    assert changed.formed.frame_id.eq(changed.formed.namespace + '/reference').all()
    if size == 'z1' and factor == 'axial_width':
        assert changed.formed.sz.eq(1.5).all()
        for label in clean.rounds:
            np.testing.assert_array_equal(clean.rounds[label], changed.rounds[label])
    else:
        assert any(not np.array_equal(clean.rounds[r], changed.rounds[r]) for r in clean.rounds)


@pytest.mark.parametrize('size', DEVELOPMENT_SIZES)
def test_combined_order_and_shared_background(size):
    clean, combined = scene(size=size), scene('combined', size)
    pd.testing.assert_frame_equal(clean.formed.drop(columns=['namespace', 'frame_id']), combined.formed.drop(columns=['namespace', 'frame_id']))
    # A=8; trend [1,1/2,1/4], middle weakening 1/4, source gain 1/2.
    expected = np.zeros((2, 4, 3))
    expected[0, [1, 0, 3], [0, 1, 2]] = [4, .5, 1]
    expected[1, [0, 1, 2], [0, 1, 2]] = [4, .5, 1]
    np.testing.assert_array_equal(combined.pre_mix, expected)
    expected[0, 0, 0] = 1
    expected[0, 2, 2] = .25
    expected[1, 0, 1] = .125
    np.testing.assert_array_equal(combined.realized, expected)
    book, config = development_scene_preset('combined', size=size)
    stationary = generate_formed_scene(book, config=replace(config,
        geometry=replace(config.geometry, local_enabled=False, translation_enabled=False)))
    assert components(stationary) == components(combined)
    assert components(combined)
    assert ext(stationary)['stream_scheme'] == ext(combined)['stream_scheme']
    # The first round's preset map is identity; later rounds move the scene.
    assert [np.array_equal(stationary.rounds[r], combined.rounds[r]) for r in combined.rounds] == [True, False, False]
    assert ext(stationary)['effective_config']['geometry']['kind'] == 'identity'
    assert ext(combined)['effective_config']['geometry']['kind'] == 'gaussian_rbf_translation'
    assert all(t['inverse']['max_residual'] <= 2e-10 for t in ext(combined)['transforms'].values())
    assert next(iter(clean.rounds.values())).shape == (*DEVELOPMENT_SIZES[size], 4)
    assert config.split == 'development'
    if size == 'z1':
        assert combined.round_truth.z.eq(0).all()
        assert combined.round_truth.support_truncated.all()


def test_loss_dropout_and_noise_isolation():
    for name, emitting in [('dropout', [True, False, True]), ('loss', [True, False, False])]:
        value = scene(name)
        for identity in value.amplicon_ids:
            assert value.round_truth[value.round_truth.amplicon_id == identity].emitting.tolist() == emitting
    book, config = development_scene_preset('combined')
    changed = generate_formed_scene(book, config=replace(config, noise=replace(config.noise, sigma=2)))
    original = scene('combined')
    pd.testing.assert_frame_equal(original.round_truth, changed.round_truth)
    assert ext(original)['transforms'] == ext(changed)['transforms']
    assert components(original) == components(changed)
    assert ext(original)['stream_scheme'] == ext(changed)['stream_scheme']
    np.testing.assert_array_equal(original.realized, changed.realized)


def test_invalid_names():
    with pytest.raises(ValueError):
        development_scene_preset('evaluation')
    with pytest.raises(ValueError):
        development_scene_preset(size='large')
    with pytest.raises(ValueError):
        development_preset_factors('unknown')


def test_packaged_fixtures_and_rotated_amplicon():
    assert DEVELOPMENT_FIXTURES == {'z1-clean': ('clean', 'z1'), 'z1-combined': ('combined', 'z1'),
                                    'small-clean': ('clean', 'small'),
                                    'small-combined': ('combined', 'small')}
    assert set(DEVELOPMENT_SIZES) == {'z1', 'small'}
    for condition, size in DEVELOPMENT_FIXTURES.values():
        formed = scene(condition, size).formed.set_index('amplicon_id')
        assert formed.loc['gt-A', ['e', 'theta']].tolist() == [1, 0]
        assert formed.loc['gt-B', ['e', 'theta']].tolist() == [1.5, np.pi / 6]
