"""Independent analytic observation oracles, keyed draws and component isolation."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import runpy

import numpy as np
import pandas as pd
import pytest

from starfinder.synthetic import (BackgroundConfig, NoiseConfig, TextureConfig, ScalarDistribution,
                                  ReadoutEffectsConfig, formed_scene_preset, generate_formed_scene)
from starfinder.synthetic._observation import _observe, evaluate_background


def generate(background=BackgroundConfig(), noise=NoiseConfig(), **kwargs):
    book, config = formed_scene_preset()
    return generate_formed_scene(book, config=replace(config, shape_zyx=(3, 7, 9), dtype='float64',
                                                     background=background, noise=noise, **kwargs))


def payload(scene):
    return scene.provenance['extensions']['starfinder.synthetic']


@pytest.mark.parametrize('depth', [1, 3])
def test_example(depth):
    path = Path(__file__).resolve().parents[3] / 'docs/examples/background_noise.py'
    runpy.run_path(str(path))['check_background'](depth)


def test_a6_literal_injected_draws():
    class Draw:
        def __init__(self, value):
            self.value = value
        def standard_normal(self, shape):
            return np.full(shape, self.value)
    def stream(component, *args):
        return Draw(-.5 if component == 'noise.dependent' else .25)
    background = dict(tissue_weights=[[0]*4], baseline=[[0]*4])
    for value, enabled, expected in [(9., True, 6.5), (9., False, 9.), (0., True, .5)]:
        images = {'r': np.full((1, 1, 1, 4), value)}
        _observe(images, np.zeros((1, 1, 1)), background,
                 NoiseConfig(enabled, 4, enabled, 2), ('a', 'b', 'c', 'd'), stream)
        np.testing.assert_array_equal(images['r'], expected)


def test_a5_baseline_after_mixing():
    matrix = np.repeat(np.eye(4)[None], 3, axis=0)
    matrix[0, 0, 1] = .25  # preset gene A uses source index 1 in round10
    scene = generate(BackgroundConfig(baseline_enabled=True, baseline=[[2, 1, 3, 4]]*3),
        coordinates=((1, 2, 2),), gene_ids={'amplicon-0': 'gene-A'},
        brightness=ScalarDistribution(parameters=(8,)),
        readout=ReadoutEffectsConfig(gain_enabled=True, gains=np.full((3, 4), .5),
                                     mixing_enabled=True, mixing=matrix))
    np.testing.assert_array_equal(scene.rounds['round10'][1, 2, 2], [3, 5, 3, 4])
    np.testing.assert_array_equal(scene.realized[0, :, 0], [1, 4, 0, 0])


def test_disabled_zero_controls_no_hidden_noise():
    background = BackgroundConfig(baseline=[[9]*4]*3, gradient_intercept=9,
        gradient_slopes_zyx=(1, 2, 3), region_centers=((0, 2, 2),),
        texture=TextureConfig(count=2), tissue_weights=np.ones((3, 4)))
    clean = generate(count=0)
    disabled = generate(background, NoiseConfig(alpha=4, sigma=2), count=0)
    zero = generate(BackgroundConfig(baseline_enabled=True, gradient_enabled=True,
        regions_enabled=True, texture_enabled=True, texture=TextureConfig(density=0)),
        NoiseConfig(True, 0, True, 0), count=0)
    for scene in (clean, disabled, zero):
        assert all(not image.any() for image in scene.rounds.values())
    assert payload(clean)['streams'] == payload(disabled)['streams']
    assert payload(disabled)['effective_config']['noise']['alpha'] == 0
    assert payload(disabled)['requested_config']['noise']['alpha'] == 4
    assert payload(disabled)['background_components'] == []


def independent_draw(component, entity=None, round_label=None, channel=None):
    descriptor = ['starfinder.synthetic/1', 'development', 42, 'formed-v1',
                  component, entity, round_label, channel]
    seed = int.from_bytes(hashlib.sha256(json.dumps(descriptor, ensure_ascii=False,
                          separators=(',', ':')).encode()).digest(), 'big')
    return np.random.Generator(np.random.PCG64(seed))


def test_noise_pixel_oracle_and_isolation():
    background = BackgroundConfig(texture_enabled=True, tissue_weights=np.ones((3, 4)),
        baseline_enabled=True, baseline=[[9]*4]*3,
        texture=TextureConfig(count=3, axial_width=ScalarDistribution('uniform', (.5, 2)),
                              brightness=ScalarDistribution('lognormal', (1, .3))))
    clean = generate(background)
    noisy = generate(background, NoiseConfig(True, 4, True, 2))
    changed = generate(background, NoiseConfig(True, 1, True, 5))
    for other in (noisy, changed):
        pd.testing.assert_frame_equal(clean.formed, other.formed)
        pd.testing.assert_frame_equal(clean.round_truth, other.round_truth)
        for name in ('intended', 'pre_mix', 'realized'):
            np.testing.assert_array_equal(getattr(clean, name), getattr(other, name))
        assert payload(clean)['background_components'] == payload(other)['background_components']
        assert payload(clean)['transforms'] == payload(other)['transforms']
    assert payload(noisy)['observation']['standardized_noise_sha256'] == payload(changed)['observation']['standardized_noise_sha256']
    for label, image in clean.rounds.items():
        for c, channel in enumerate(clean.channel_labels):
            dep = independent_draw('noise.dependent', None, label, channel).standard_normal(image.shape[:3])
            ind = independent_draw('noise.independent', None, label, channel).standard_normal(image.shape[:3])
            j = image[..., c]
            expected = (j + 2*np.sqrt(j)*dep) + 2*ind
            np.testing.assert_array_equal(noisy.rounds[label][..., c], expected)
    # Width/brightness changes cannot redraw centers; molecule changes cannot redraw texture.
    other = generate(replace(background, texture=replace(background.texture,
                     brightness=ScalarDistribution(parameters=(20,)))), count=0)
    for a, b in zip(payload(clean)['background_components'], payload(other)['background_components']):
        assert a['center_zyx'] == b['center_zyx'] and a['sigma_zyx'] == b['sigma_zyx']


def test_texture_laws_and_reference_evaluation():
    texture = TextureConfig(density=.01, axial_width=ScalarDistribution('uniform', (.5, 2)),
                            lateral_width=ScalarDistribution('lognormal', (0, .2)),
                            brightness=ScalarDistribution('uniform', (2, 6)))
    scene = generate(BackgroundConfig(texture_enabled=True, texture=texture, tissue_weights=[[1]*4]*3), count=0)
    components = payload(scene)['background_components']
    assert len(components) == independent_draw('background.count').poisson(.01*3*7*9)
    for i, component in enumerate(components):
        identity = f'blob-{i}'
        np.testing.assert_array_equal(component['center_zyx'], independent_draw('background.placement', identity).uniform(0, [2, 6, 8], 3))
        axial = independent_draw('background.width', json.dumps([identity, 'axial'], separators=(',', ':'))).uniform(.5, 2)
        assert component['sigma_zyx'][0] == axial
        assert component['height'] == independent_draw('background.brightness', identity).uniform(2, 6)
    # Analytic tails exist beyond the image; gradients extrapolate without wrap.
    scene = generate(BackgroundConfig(gradient_enabled=True, gradient_intercept=1,
        gradient_slopes_zyx=(0, 0, 8), tissue_weights=[[1]*4]*3), count=0)
    np.testing.assert_array_equal(evaluate_background(payload(scene)['background_components'],
        np.array([[0., 0, -2], [0, 0, 10]])), [0, 11])


@pytest.mark.parametrize('kwargs', [
    {'baseline_enabled': 1}, {'baseline': [[1]*4]}, {'baseline': [[-1]*4]*3},
    {'tissue_weights': [[float('inf')]*4]*3}, {'gradient_slopes_zyx': (1, 2)},
    {'gradient_intercept': -1}, {'region_centers': (1, 2, 3)},
    {'region_centers': ((0, 1, 1),), 'region_sigma_zyx': ((1, 0, 1),)},
    {'texture': TextureConfig(count=True)}, {'texture': TextureConfig(count=2, density=1)},
    {'texture': TextureConfig(density=-1)}, {'texture': TextureConfig(count=1025)},
    {'texture': TextureConfig(axial_width=ScalarDistribution(parameters=(0,)))},
])
def test_invalid_background_even_disabled_empty(kwargs):
    with pytest.raises(ValueError):
        generate(BackgroundConfig(**kwargs), count=0)


@pytest.mark.parametrize('kwargs', [{'alpha': -1}, {'sigma': float('nan')}, {'alpha': True},
    {'sigma': [2]}, {'dependent_enabled': 1}, {'sigma': None}])
def test_invalid_noise_even_disabled_empty(kwargs):
    with pytest.raises(ValueError):
        generate(noise=NoiseConfig(**kwargs), count=0)


def test_noise_statistics_and_signed_output():
    book, base = formed_scene_preset()
    scene = generate_formed_scene(book, config=replace(base, count=0, shape_zyx=(8, 32, 32), dtype='float64',
        background=BackgroundConfig(baseline_enabled=True, baseline=[[9]*4]*3),
        noise=NoiseConfig(True, 4, True, 2)))
    values = np.concatenate([a.ravel() for a in scene.rounds.values()])
    # N=98304 independent residuals, E=9, variance=4*9+2**2=40.
    # Fixed deterministic test, loose >6-SE bounds; not empirical calibration.
    assert abs(values.mean() - 9) < .15
    assert abs(values.var() - 40) < 1.2
    assert (values < 0).any()


def test_texture_placement_and_supplied_properties():
    weights = np.zeros((3, 7, 9))
    weights[1, 2, 3] = 1
    texture = TextureConfig(count=1, placement='weighted', spatial_weights=weights,
        brightness=ScalarDistribution('supplied', (), {'blob-0': 8}),
        axial_width=ScalarDistribution('supplied', (), {'blob-0': 2}))
    scene = generate(BackgroundConfig(texture_enabled=True, texture=texture))
    blob = payload(scene)['background_components'][0]
    assert blob['height'] == 8 and blob['sigma_zyx'] == [2, 3, 3]
    assert np.all(np.abs(np.array(blob['center_zyx']) - [1, 2, 3]) <= .5)
    assert blob['frame_id'] == scene.metadata.frame_id
    clustered = TextureConfig(count=2, placement='clustered', cluster_centers=((1, 3, 4),),
                              cluster_weights=(1,), spread_zyx=(.1, .1, .1))
    scene = generate(BackgroundConfig(texture_enabled=True, texture=clustered))
    for blob in payload(scene)['background_components']:
        rng = independent_draw('background.placement', blob['component_id'])
        rng.choice(1, p=[1.])
        np.testing.assert_array_equal(blob['center_zyx'], np.array([1, 3, 4]) + rng.normal(0, .1, 3))
    with pytest.raises(ValueError, match='exceeds max_count'):
        generate(BackgroundConfig(texture_enabled=True, texture=TextureConfig(density=1, max_count=0)))
    with pytest.raises(ValueError):
        generate(BackgroundConfig(gradient_intercept=None))


def test_noise_cast_once_and_round_channel_streams():
    book, base = formed_scene_preset()
    config = replace(base, count=0, shape_zyx=(1, 3, 3), dtype='float64',
        background=BackgroundConfig(baseline_enabled=True, baseline=[[0, .5, 1.5, 255.5]]*3))
    scene = generate_formed_scene(book, config=replace(config, dtype='uint8'))
    np.testing.assert_array_equal(scene.rounds['round10'][0, 0, 0], [0, 0, 2, 255])
    assert payload(scene)['observation']['clipping_counts']['round10']['above'] == 9
    config = replace(config, noise=NoiseConfig(independent_enabled=True, sigma=10))
    signed = generate_formed_scene(book, config=config)
    cast = generate_formed_scene(book, config=replace(config, dtype='uint8'))
    for label, image in signed.rounds.items():
        np.testing.assert_array_equal(cast.rounds[label], np.clip(np.rint(image), 0, 255).astype('uint8'))
    hashes = payload(signed)['observation']['standardized_noise_sha256']
    assert len(hashes) == len(set(hashes.values())) == 12
    assert payload(signed)['observation']['standardized_noise_sha256'] == payload(cast)['observation']['standardized_noise_sha256']


def test_observation_cross_process_repeatability():
    import os
    import subprocess
    import sys
    code = '''
import hashlib,json
from dataclasses import replace
from starfinder.synthetic import *
b,c=formed_scene_preset()
s=generate_formed_scene(b,config=replace(c,shape_zyx=(1,7,9),
    background=BackgroundConfig(texture_enabled=True,tissue_weights=[[1]*4]*3),
    noise=NoiseConfig(True,4,True,2)))
h=hashlib.sha256()
for a in s.rounds.values(): h.update(a.tobytes())
h.update(json.dumps(s.provenance,sort_keys=True).encode())
print(h.hexdigest())
'''
    outputs = [subprocess.check_output([sys.executable, '-c', code],
        env=dict(os.environ, PYTHONHASHSEED=value), text=True) for value in ('1', '987')]
    assert outputs[0] == outputs[1]
