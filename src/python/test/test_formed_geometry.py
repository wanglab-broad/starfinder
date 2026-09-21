"""Independent A7 geometry expectations; no registration estimator is an oracle."""
from dataclasses import replace
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from starfinder.synthetic import (BackgroundConfig, GeometryConfig, NoiseConfig,
    ReadoutEffectsConfig, ScalarDistribution, formed_scene_preset, generate_formed_scene)
from starfinder.synthetic._geometry import _forward, _inverse


def payload(scene):
    return scene.provenance['extensions']['starfinder.synthetic']


def make(depth=3, **kwargs):
    cb, cfg = formed_scene_preset()
    return generate_formed_scene(cb, config=replace(cfg, shape_zyx=(depth, 7, 9),
        dtype='float64', **kwargs))


@pytest.mark.parametrize('depth', [3, 1])
def test_a7_fractional_translation_boundary_and_absolute_rounds(depth):
    z = depth // 2
    translation = (.5 if depth > 1 else 0, -1, 2)
    geometry = GeometryConfig(translation_enabled=True,
        translations_zyx=(translation, (0, 0, 0), (0, 0, -3.5)))
    s = make(depth, coordinates=((z, 2, 3), (z, 2, -20)),
             amplicon_ids=('gt-A', 'outside'), gene_ids={'gt-A': 'gene-A', 'outside': 'gene-A'},
             brightness=ScalarDistribution(parameters=(8,)), geometry=geometry)
    truth = s.round_truth.query('amplicon_id == "gt-A"')
    np.testing.assert_array_equal(truth[['z','y','x']],
        [[z + translation[0], 1, 5], [z, 2, 3], [z, 2, -.5]])
    assert truth.center_in_bounds.tolist() == [True, True, False]
    assert truth.support_intersects.all()
    assert len(s.formed) == 2 and len(s.round_truth) == 6
    assert not s.round_truth.query('amplicon_id == "outside"').support_intersects.any()
    # Last round codeword 3 maps to channel 3. Out-of-frame center still contributes.
    assert s.rounds['round1'][z, 2, 0, 3] == pytest.approx(7.059975220676764, abs=1e-12)
    expected_peak = 7.059975220676764 if depth > 1 else 8
    assert s.rounds['round10'][z, 1, 5, 1] == pytest.approx(expected_peak, abs=1e-12)
    for record in payload(s)['transforms'].values():
        label = record['round_label']
        assert record['source_frame'] == s.metadata.frame_id
        assert record['destination_frame'] == s.round_metadata[label].frame_id
        assert record['direction'] == 'reference_to_round'
        assert record['units'] == 'voxel_index'
        assert record['inverse']['max_residual'] <= 2e-10
    np.testing.assert_array_equal(s.formed[['z','y','x']], [[z,2,3],[z,2,-20]])


@pytest.mark.parametrize('depth', [3, 1])
def test_local_composition_shared_background_and_independent_inverse(depth):
    z = depth // 2
    center = (z,2,3)
    geometry = GeometryConfig(translation_enabled=True, translations_zyx=((0,0,.5),)*3,
        local_enabled=True, centers_zyx=(center,), scales=(2,), vectors_zyx=(((0,0,.25),),)*3)
    background = BackgroundConfig(regions_enabled=True, region_centers=(center,),
        region_sigma_zyx=((1,1,1),), region_heights=(8,), tissue_weights=((1,0,0,0),)*3)
    s = make(depth, coordinates=(center,), amplicon_ids=('gt-A',), gene_ids={'gt-A':'gene-A'},
             brightness=ScalarDistribution(parameters=(8,)), geometry=geometry, background=background)
    np.testing.assert_array_equal(s.round_truth[['z','y','x']], [[z,2,3.75]]*3)
    mapping = next(iter(payload(s)['transforms'].values()))
    # Evaluate displacement at q, not at q+t: two independent off-center literals.
    q = np.array([[z,2,5], [z,4,3]], float)
    expected = q + [[0,0,.6516326649281584]]*2
    np.testing.assert_allclose(_forward(q, mapping), expected, atol=1e-12, rtol=0)
    recovered, diag = _inverse(expected, mapping)
    np.testing.assert_allclose(recovered, q, atol=2e-10, rtol=0)
    assert diag['max_residual'] <= 2e-10
    # Independent scalar bisection (not the production fixed-point method).
    lo, hi = 2., 4.
    for _ in range(60):
        mid = (lo+hi)/2
        if mid + .25*np.exp(-(mid-3)**2/8) + .5 < 4:
            lo = mid
        else:
            hi = mid
    ref_x = (lo+hi)/2
    expected_background = 8*np.exp(-.5*(ref_x-3)**2)
    assert s.rounds['round10'][z,2,4,0] == pytest.approx(expected_background, abs=2e-10)
    # Molecule and background share transformed landmark 3.75; distinct shape semantics.
    assert s.rounds['round10'][z,2,4,1] == pytest.approx(8*np.exp(-.5*.25**2), abs=1e-12)
    np.testing.assert_allclose(_inverse(np.array([[z,2,3.75]]), mapping)[0], [center], atol=2e-10)
    assert payload(s)['background_components'][0]['center_zyx'] == list(center)


def test_identity_disable_and_unrelated_latents_preserved():
    readout = ReadoutEffectsConfig(dropout_enabled=True, dropout_probability=(.2,.4,.3),
                                  loss_enabled=True, loss_probability=.2)
    background = BackgroundConfig(texture_enabled=True, tissue_weights=((1,1,1,1),)*3)
    noise = NoiseConfig(dependent_enabled=True, alpha=.3, independent_enabled=True, sigma=.2)
    base = make(count=4, readout=readout, background=background, noise=noise)
    geometry = GeometryConfig(translation_max_zyx=(.2,.3,.4), centers_zyx=((1,3,4),), strength=.05)
    disabled = make(count=4, readout=readout, background=background, noise=noise, geometry=geometry)
    moved = make(count=4, readout=readout, background=background, noise=noise,
                 geometry=replace(geometry, translation_enabled=True, local_enabled=True))
    for label in base.rounds:
        np.testing.assert_array_equal(base.rounds[label], disabled.rounds[label])
    for other in (disabled, moved):
        pd.testing.assert_frame_equal(base.formed, other.formed)
        for name in ('intended','pre_mix','realized'):
            np.testing.assert_array_equal(getattr(base,name), getattr(other,name))
        cols = ['namespace','amplicon_id','round_label','dropped','lost','first_loss_round']
        pd.testing.assert_frame_equal(base.round_truth[cols], other.round_truth[cols])
        assert payload(base)['background_components'] == payload(other)['background_components']
        assert payload(base)['observation']['standardized_noise_sha256'] == payload(other)['observation']['standardized_noise_sha256']
    # Independent frozen descriptor and draw, including vector entity and ZYX order.
    descriptor = ['starfinder.synthetic/1','development',42,'formed-v1','geometry.local',
                  '["control-0","vector"]','round10',None]
    seed = int.from_bytes(hashlib.sha256(json.dumps(descriptor,separators=(',',':')).encode()).digest(),'big')
    vector = np.random.Generator(np.random.PCG64(seed)).normal(size=3)*.05
    record = next(iter(payload(moved)['transforms'].values()))
    np.testing.assert_array_equal(record['vectors_zyx'], [vector])


@pytest.mark.parametrize('geometry', [
    GeometryConfig(translation_enabled=1), GeometryConfig(translations_zyx=((0,0,0),)),
    GeometryConfig(translation_max_zyx=(-1,0,0)), GeometryConfig(strength=-1),
    GeometryConfig(strength=True), GeometryConfig(scales=(0,), centers_zyx=((0,0,0),)),
    GeometryConfig(centers_zyx=((0,0),)), GeometryConfig(strength=float('nan')),
    GeometryConfig(centers_zyx=((0,0,0),), vectors_zyx=(((0,0,20),),)*3),
    GeometryConfig(translations_zyx=((0,0,0),)*3, translation_max_zyx=(1,1,1)),
    GeometryConfig(vectors_zyx=(), strength=1),
])
def test_invalid_geometry(geometry):
    with pytest.raises((ValueError, TypeError)):
        make(count=0, geometry=geometry)


@pytest.mark.parametrize('geometry', [
    GeometryConfig(translations_zyx=((1,0,0),)*3),
    GeometryConfig(translation_max_zyx=(1,0,0)),
    GeometryConfig(centers_zyx=((1,0,0),)),
    GeometryConfig(centers_zyx=((0,0,0),), vectors_zyx=(((.1,0,0),),)*3),
])
def test_singleton_rejects_out_of_plane_even_disabled(geometry):
    with pytest.raises(ValueError, match='Z=1'):
        make(1, count=0, geometry=geometry)


def test_random_singleton_and_cross_process_repeat():
    code = '''
import hashlib,json
from dataclasses import replace
from starfinder.synthetic import *
cb,cfg=formed_scene_preset('formed-z1-v1')
s=generate_formed_scene(cb,config=replace(cfg,shape_zyx=(1,7,9),geometry=GeometryConfig(
 translation_enabled=True,translation_max_zyx=(0,.2,.3),local_enabled=True,
 centers_zyx=((0,2,3),),strength=.03)))
p=s.provenance['extensions']['starfinder.synthetic']
assert all(t['vectors_zyx'][0][0]==0 for t in p['transforms'].values())
print(json.dumps(dict(provenance=s.provenance,truth=s.round_truth.to_json(),
 images={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in s.rounds.items()}),sort_keys=True))
'''
    results = [subprocess.check_output([sys.executable,'-c',code], env=dict(os.environ,PYTHONHASHSEED=str(seed)))
               for seed in (1,991)]
    assert results[0] == results[1]


def test_multiple_3d_controls_scales_and_vector_components():
    geometry = GeometryConfig(local_enabled=True, centers_zyx=((1,2,3),(1,2,5)),
        scales=(2,1), vectors_zyx=(((.1,.2,-.1),(-.2,.1,.2)),)*3,
        translation_enabled=True, translations_zyx=((.25,-.5,.75),)*3)
    s = make(coordinates=((1,2,3),), geometry=geometry)
    # At first center: first Gaussian=1; second exp(-2). All three axes move.
    expected = np.array([1,2,3]) + [.1,.2,-.1] + np.exp(-2)*np.array([-.2,.1,.2]) + [.25,-.5,.75]
    np.testing.assert_allclose(s.round_truth[['z','y','x']], np.tile(expected,(3,1)), atol=1e-12, rtol=0)
    record = next(iter(payload(s)['transforms'].values()))
    np.testing.assert_allclose(_inverse(expected[None], record)[0], [[1,2,3]], atol=2e-10, rtol=0)
    expected_bound = (np.sqrt(.06)/2 + .3)/np.sqrt(np.e)
    assert record['lipschitz_bound'] == pytest.approx(expected_bound, abs=1e-15)
    # Disabled retained requests must not masquerade as effective transforms.
    disabled = make(count=0, geometry=replace(geometry,local_enabled=False,translation_enabled=False))
    effective = payload(disabled)['effective_config']['geometry']
    assert all(not np.any(t['vectors_zyx']) and not np.any(t['translation_zyx'])
               for t in effective['rounds'].values())
