"""Independent literals and complete-population checks for formed scenes."""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from starfinder.barcode import Codebook
from starfinder.image import ImageMetadata
from starfinder.synthetic import (BackgroundConfig, FormedSceneConfig, GeometryConfig,
                                  NoiseConfig, ReadoutEffectsConfig, ScalarDistribution,
                                  TextureConfig, formed_scene_preset, generate_formed_scene)


def book():
    return Codebook(pd.DataFrame(dict(gene_id=['a', 'b'], color_sequence=['123', '214'])),
                    ('round10', 'round2', 'round1'), ('ch02', 'ch00', 'ch03', 'ch01'),
                    {'1': 1, '2': 0, '3': 3, '4': 2})


def scene(**kwargs):
    return generate_formed_scene(book(), config=FormedSceneConfig(**kwargs))


def scalar(value):
    return ScalarDistribution(parameters=(value,))


def literal(coordinates, **kwargs):
    ids = tuple(f'p{i}' for i in range(len(coordinates)))
    return scene(coordinates=coordinates, amplicon_ids=ids,
                 gene_ids={i: 'a' for i in ids}, brightness=scalar(8), **kwargs)


@pytest.mark.parametrize('shape,population', [
    ((3, 7, 9), dict(count=0)), ((3, 7, 9), dict(density=0)),
    ((3, 7, 9), dict(coordinates=())), ((1, 7, 9), dict(coordinates=())),
])
def test_empty_typed_truth(shape, population):
    result = scene(shape_zyx=shape, **population)
    assert result.formed.empty and result.round_truth.empty
    assert result.intended.shape == (0, 4, 3)
    assert result.intended.dtype == np.float64
    assert result.formed.amplicon_id.dtype == pd.StringDtype()
    assert result.formed.formed_index.dtype == np.int64
    assert result.round_truth.support_intersects.dtype == bool
    assert str(result.round_truth.first_loss_round.dtype) == 'Int64'
    assert all(v.shape == (*shape, 4) and not v.any() for v in result.rounds.values())


@pytest.mark.parametrize('dtype,atol', [('float64', 1e-12), ('float32', 1e-6)])
@pytest.mark.parametrize('depth,z', [(3, 1), (1, 0)])
def test_a2_integer_fractional_kernel(dtype, atol, depth, z):
    result = literal(((z, 2, 2),), shape_zyx=(depth, 9, 9), dtype=dtype)
    image = result.rounds['round10'][..., 1]
    np.testing.assert_allclose(image[z, 2, [2, 3, 6, 7]],
                               [8, 4.852245277701067, .002683701023220095, 0], rtol=0, atol=atol)
    np.testing.assert_array_equal(result.intended[0], [[0, 8, 0], [8, 0, 0], [0, 0, 0], [0, 0, 8]])
    np.testing.assert_array_equal(result.realized, result.intended)
    np.testing.assert_array_equal(result.pre_mix, result.intended)
    assert not np.shares_memory(result.realized, result.intended)
    fractional = literal(((z, 2, 2.5),), shape_zyx=(depth, 9, 9), dtype=dtype)
    np.testing.assert_allclose(fractional.rounds['round10'][z, 2, [2, 3], 1],
                               [7.059975220676764]*2, rtol=0, atol=atol)
    if depth == 1:
        # Z=1 samples z=0 only, so its four-sigma axial extent is always truncated.
        assert result.round_truth.support_truncated.all()


def test_a3_shape_rotated_and_boundary():
    result = literal(((2, 4, 4),), shape_zyx=(5, 9, 9), dtype='float64',
                     axial_width=scalar(2), elongation=scalar(2))
    a = result.rounds['round10'][..., 1]
    np.testing.assert_allclose([a[4, 4, 4], a[2, 6, 4], a[2, 4, 6]],
                               [4.852245277701067, 4.852245277701067, 1.0826822658929016], atol=1e-12, rtol=0)
    rotated = literal(((2, 4, 4),), shape_zyx=(5, 9, 9), dtype='float64',
                      elongation=scalar(2), angle=scalar(np.pi/2))
    np.testing.assert_allclose(rotated.rounds['round10'][2, 4, 6, 1], 4.852245277701067, atol=1e-12, rtol=0)
    outside = literal(((1, 2, -.5), (1, 2, -10)), shape_zyx=(3, 9, 9), dtype='float64')
    np.testing.assert_allclose(outside.rounds['round10'][1, 2, 0, 1], 7.059975220676764, atol=1e-12, rtol=0)
    truth = outside.round_truth.set_index(['amplicon_id', 'round_label'])
    assert not truth.loc[('p0', 'round10'), 'center_in_bounds']
    assert truth.loc[('p0', 'round10'), 'support_intersects']
    assert not truth.loc[('p1', 'round10'), 'support_intersects']
    assert len(outside.formed) == 2 and len(truth) == 6
    assert 'eligible' not in truth and truth.emitting.all()


def test_oblique_angle_kernel_literal():
    # theta=pi/6, e=2, sl=sz=1 at (1,4,4). Offset (dy,dx)=(0,2): u=sin*2/2=1/2,
    # v=cos*2=sqrt(3), r2=13/4. Offset (1,0): u=cos/2=sqrt(3)/4, v=-sin=-1/2,
    # r2=7/16. Values are 8*exp(-r2/2), independent of the renderer.
    result = literal(((1, 4, 4),), shape_zyx=(3, 9, 9), dtype='float64',
                     elongation=scalar(2), angle=scalar(np.pi/6))
    image = result.rounds['round10'][..., 1]
    assert image[1, 4, 6] == pytest.approx(8*np.exp(-13/8), abs=1e-12)
    assert image[1, 5, 4] == pytest.approx(8*np.exp(-7/32), abs=1e-12)
    # Unrotated e=2 would give the X step exp(-2) instead: theta is effective.
    assert image[1, 4, 6] != pytest.approx(8*np.exp(-2), abs=1e-3)
    assert result.formed.theta.tolist() == [np.pi/6]


def test_coincident_and_dark_population_stable_ids():
    result = scene(coordinates=((1, 2, 2),)*3, amplicon_ids=('z', 'a', 'dark'),
                   gene_ids={'dark': 'b', 'a': 'a', 'z': 'a'}, dtype='float64',
                   brightness=ScalarDistribution('supplied', (), {'a': 8, 'dark': 0, 'z': 8}))
    assert result.rounds['round10'][1, 2, 2, 1] == 16
    assert len(result.formed) == 3 and len(result.round_truth) == 9
    assert result.formed.set_index('amplicon_id').loc['dark', 'gene_id'] == 'b'
    dark = result.round_truth.query('amplicon_id == "dark"')
    assert not dark.emitting.any() and dark.support_intersects.all()
    assert dark.first_loss_round.isna().all()
    assert not result.round_truth[['dropped', 'weakened', 'lost']].any().any()
    assert result.amplicon_ids == ('z', 'a', 'dark')


def test_component_and_append_isolation():
    cb, config = formed_scene_preset()
    config = replace(config, brightness=ScalarDistribution('lognormal', (3, .2)),
                     lateral_width=ScalarDistribution('uniform', (.5, 2)))
    original = generate_formed_scene(cb, config=config)
    changed = generate_formed_scene(cb, config=replace(config, lateral_width=scalar(2), abundances=(1, 0)))
    pd.testing.assert_frame_equal(original.formed[['amplicon_id', 'z', 'y', 'x', 'A', 'sz', 'e', 'theta']],
                                  changed.formed[['amplicon_id', 'z', 'y', 'x', 'A', 'sz', 'e', 'theta']])
    assert set(changed.formed.gene_id) == {'gene-A'}
    more = generate_formed_scene(cb, config=replace(config, count=9))
    pd.testing.assert_frame_equal(original.formed, more.formed.iloc[:8].reset_index(drop=True))
    reordered = generate_formed_scene(cb, config=replace(config, amplicon_ids=original.amplicon_ids[::-1]))
    cols = [x for x in original.formed if x != 'formed_index']
    pd.testing.assert_frame_equal(original.formed[cols].sort_values('amplicon_id').reset_index(drop=True),
                                  reordered.formed[cols].sort_values('amplicon_id').reset_index(drop=True))
    for r in original.rounds:
        np.testing.assert_array_equal(original.rounds[r], reordered.rounds[r])


def test_weighted_clustered_and_density_laws():
    # A single supported voxel fixes a half-cell interval; no statistical threshold.
    weights = np.zeros((1, 5, 7)); weights[0, 2, 3] = 1
    weighted = scene(shape_zyx=(1, 5, 7), count=4, placement='weighted', spatial_weights=weights)
    p = weighted.formed[['z', 'y', 'x']].to_numpy()
    assert np.all(p[:, 0] == 0)
    assert np.all((p[:, 1:] >= [1.5, 2.5]) & (p[:, 1:] <= [2.5, 3.5]))
    # All singleton axes make the cluster's expected coordinates exact.
    clustered = scene(shape_zyx=(1, 1, 1), count=3, placement='clustered',
                      cluster_centers=((0, 0, 0),), cluster_weights=(1,))
    np.testing.assert_array_equal(clustered.formed[['z', 'y', 'x']], np.zeros((3, 3)))
    assert len(clustered.round_truth) == 9
    general = scene(count=5, placement='clustered', cluster_centers=((1, 2, 3), (3, 4, 5)), cluster_weights=(1, 2))
    points = general.formed[['z', 'y', 'x']].to_numpy()
    assert np.all((points >= 0) & (points <= [7, 31, 31]))
    density = scene(shape_zyx=(1, 2, 3), density=2)
    # Independent frozen descriptor, not a call to production seed derivation.
    descriptor = b'["starfinder.synthetic/1","development",42,"formed-v1","count",null,null,null]'
    seed = int.from_bytes(hashlib.sha256(descriptor).digest(), 'big')
    expected = np.random.Generator(np.random.PCG64(seed)).poisson(12)
    assert len(density.formed) == expected
    with pytest.raises(ValueError, match='max_count'):
        scene(shape_zyx=(1, 2, 3), density=2, max_count=0)


@pytest.mark.parametrize('distribution', [ScalarDistribution('constant', (2,)),
    ScalarDistribution('uniform', (1, 2)), ScalarDistribution('lognormal', (0, .1))])
def test_property_laws_finite(distribution):
    result = scene(count=3, brightness=distribution, axial_width=distribution, lateral_width=distribution,
                   elongation=ScalarDistribution('folded_lognormal', (0, .3)),
                   angle=ScalarDistribution('uniform', (0, np.pi)))
    assert (result.formed[['A', 'sz', 'sl']] > 0).all().all()
    assert (result.formed.e >= 1).all()
    assert ((result.formed.theta >= 0) & (result.formed.theta < np.pi)).all()


@pytest.mark.parametrize('kwargs', [
    dict(count=True), dict(count=-1), dict(count=1.5), dict(count=1025),
    dict(count=0, density=0), dict(count=0, coordinates=()), dict(density=-1), dict(density=np.inf), dict(density=True),
    dict(seed=True), dict(seed=-1), dict(seed=2**64), dict(split=''), dict(split='é'),
    dict(max_count=-1), dict(max_count=True),
    dict(shape_zyx=(0, 2, 3)), dict(shape_zyx=(1, True, 3)), dict(shape_zyx=(1, 2)), dict(shape_zyx=(1, 2, 3.0)),
    dict(coordinates=((1, 2),)), dict(coordinates=((np.nan, 2, 3),)),
    dict(shape_zyx=(1, 3, 3), coordinates=((1, 2, 2),)),
    dict(count=2, amplicon_ids=('same', 'same')), dict(count=1, amplicon_ids=('é',)),
    dict(count=1, gene_ids={'amplicon-0': 'unknown'}), dict(abundances=(0, 0)), dict(abundances=(-1, 2)),
    dict(placement='unknown'), dict(placement='weighted', spatial_weights=np.zeros((8, 32, 32))),
    dict(placement='clustered', cluster_centers=((0, 0, 0),), cluster_weights=(0,)),
    dict(placement='clustered', cluster_centers=((-1, 0, 0),), cluster_weights=(1,)),
    dict(spread_zyx=(0, 1, 1)), dict(dtype='int64'),
    dict(brightness=scalar(-1)), dict(axial_width=scalar(0)), dict(elongation=scalar(.5)),
    dict(brightness=ScalarDistribution('uniform', (2, 1))),
    dict(brightness=ScalarDistribution('lognormal', (1, -1))),
    dict(brightness=ScalarDistribution('folded_lognormal', (1, 1))),
    dict(elongation=ScalarDistribution('lognormal', (1, 1))),
    dict(angle=ScalarDistribution('uniform', (0, 1))),
    dict(angle=scalar(-.1)), dict(angle=scalar(np.pi)), dict(angle=scalar(4)),
    dict(count=1, angle=ScalarDistribution('supplied', (), {'amplicon-0': 3.5})),
    dict(angle=ScalarDistribution('lognormal', (0, 1))),
    dict(brightness=ScalarDistribution('supplied', (), {})),
    dict(brightness=ScalarDistribution('unknown', (1,))),
])
def test_invalid_inputs(kwargs):
    with pytest.raises((ValueError, TypeError)):
        scene(**kwargs)


def test_angle_domain_endpoints_accepted():
    for angle in (0.0, np.nextafter(np.pi, 0)):
        assert scene(count=1, angle=scalar(angle)).formed.theta.tolist() == [angle]


def test_invalid_codebook_metadata_and_overflow():
    cb = book()
    cb.table.loc[1, 'color_sequence'] = '123'
    with pytest.raises(ValueError, match='collision'):
        generate_formed_scene(cb)
    with pytest.raises(ValueError):
        ImageMetadata('ref', spacing_zyx=(1, 0, 1))
    with pytest.raises(ValueError, match='nonfinite generated'):
        scene(brightness=ScalarDistribution('lognormal', (1000, 0)))
    with pytest.raises(ValueError, match='overflow'):
        scene(coordinates=((1, 2, 2),), brightness=scalar(1e100))


def test_quantization_and_provenance_payload():
    metadata = ImageMetadata('ref', (2, 3, 4), (10, 20, 30), ((1, 0, 0), (0, 1, 0), (0, 0, 1)), 'um')
    for brightness, expected in [(.5, 0), (1.5, 2), (255.5, 255)]:
        result = generate_formed_scene(book(), config=FormedSceneConfig(coordinates=((1, 2, 2),),
                                      brightness=scalar(brightness), dtype='uint8'), metadata=metadata)
        assert result.rounds['round10'].max() == expected
        assert result.metadata == metadata
        assert all(m.spacing_zyx == (2., 3., 4.) for m in result.round_metadata.values())
        provenance = result.provenance
        assert set(provenance) == {'generator', 'generator_version', 'requested_config',
                                   'effective_config', 'seed', 'stream_scheme', 'codebook',
                                   'image_sha256', 'clipping_counts', 'transforms'}
        assert provenance['clipping_counts']['round10']['above'] == int(brightness > 255)
        assert provenance['seed'] == 42
        assert provenance['stream_scheme']['key'] == [
            'namespace', 'split', 'seed', 'scene_key', 'component', 'entity', 'round_label', 'channel_label']
        assert provenance['codebook']['round_labels'] == ['round10', 'round2', 'round1']
        for label, image in result.rounds.items():
            assert provenance['image_sha256'][label] == hashlib.sha256(image.tobytes()).hexdigest()
        assert json.loads(result.formed.namespace.iloc[0]) == ['formed-small-v1', 'sample', 'FOV_001', 'formed']
        text = json.dumps(provenance, allow_nan=False)
        assert 'contract_id' not in text and 'catalog' not in text


def test_configs_are_not_hashable():
    for config in (FormedSceneConfig(), ScalarDistribution(), ReadoutEffectsConfig(),
                   BackgroundConfig(), TextureConfig(), NoiseConfig(), GeometryConfig()):
        with pytest.raises(TypeError, match='unhashable'):
            hash(config)
    # Equality still compares values; frozen fields still reject assignment.
    assert FormedSceneConfig(seed=1) == FormedSceneConfig(seed=1)
    with pytest.raises(AttributeError):
        FormedSceneConfig().seed = 1


def test_shapes_rounds_and_counts_above_former_limits():
    # 8x128x128 with five rounds exceeds the former 32x64x64 / 1-4 round limits.
    rounds = tuple(f'r{i}' for i in range(5))
    cb = Codebook(pd.DataFrame(dict(gene_id=['a', 'b'], color_sequence=['12341', '43214'])),
                  rounds, ('c0', 'c1', 'c2', 'c3'))
    config = FormedSceneConfig(shape_zyx=(8, 128, 128), coordinates=((3, 100, 120), (4, 10, 5)),
                               amplicon_ids=('far', 'near'), gene_ids={'far': 'a', 'near': 'b'},
                               brightness=scalar(8), dtype='float64', split='any-label',
                               readout=ReadoutEffectsConfig(trend_enabled=True, trend_base=.5),
                               geometry=GeometryConfig(translation_enabled=True,
                                                       translations_zyx=((0, 0, 0),)*4 + ((0, 1, -1),)))
    result = generate_formed_scene(cb, config=config)
    assert result.round_labels == rounds
    assert all(image.shape == (8, 128, 128, 4) for image in result.rounds.values())
    assert result.intended.shape == result.realized.shape == (2, 4, 5)
    np.testing.assert_array_equal(result.realized[0].sum(axis=0), 8 * .5**np.arange(5))
    # 'far' is gene a = 12341: round index 4 uses color 1 -> channel 0; brightness 8/16.
    assert result.rounds['r4'][3, 101, 119, 0] == 8 * .5**4
    assert result.rounds['r0'][3, 100, 120, 0] == 8
    assert len(result.round_truth) == 10 and result.round_truth.center_in_bounds.all()
    assert result.provenance['effective_config']['split'] == 'any-label'
    # Counts above the former 1024 limit only require an explicit max_count.
    many = generate_formed_scene(cb, config=FormedSceneConfig(shape_zyx=(1, 40, 40), count=1100,
                                                              max_count=1100))
    assert len(many.formed) == 1100 and len(many.round_truth) == 5500
    with pytest.raises(ValueError, match='max_count'):
        generate_formed_scene(cb, config=FormedSceneConfig(shape_zyx=(1, 4, 4), count=1100))


def test_background_and_geometry_counts_above_former_limits():
    cb, config = formed_scene_preset('formed-z1-v1')
    texture = TextureConfig(count=1100, max_count=1100)
    result = generate_formed_scene(cb, config=replace(config, shape_zyx=(1, 4, 4), count=0,
        background=BackgroundConfig(texture_enabled=True, texture=texture, tissue_weights=np.ones((3, 4)))))
    assert len(result.provenance['effective_config']['background']['components']) == 1100
    controls = tuple((0, i % 4, i // 400) for i in range(1100))
    result = generate_formed_scene(cb, config=replace(config, shape_zyx=(1, 4, 4), count=0,
        geometry=GeometryConfig(local_enabled=True, centers_zyx=controls)))
    assert len(next(iter(result.provenance['transforms'].values()))['centers_zyx']) == 1100


def test_split_is_a_stream_label():
    cb, config = formed_scene_preset()
    base = generate_formed_scene(cb, config=config)
    other = generate_formed_scene(cb, config=replace(config, split='evaluation'))
    assert not np.array_equal(base.formed[['z', 'y', 'x']], other.formed[['z', 'y', 'x']])
    descriptor = other.provenance['stream_scheme']['streams'][0]
    assert descriptor[:4] == ['starfinder.synthetic/1', 'evaluation', 42, 'formed-v1']


def test_actual_cross_process_repeats_and_no_files(tmp_path):
    code = '''
import hashlib,json
from starfinder.synthetic import (DEVELOPMENT_FIXTURES, development_scene_preset,
                                  formed_scene_preset, generate_formed_scene)
output=[]
configs=[formed_scene_preset(name) for name in ('formed-small-v1','formed-z1-v1')]
configs+=[development_scene_preset(c,size=s) for c,s in DEVELOPMENT_FIXTURES.values()]
for cb,cfg in configs:
    s=generate_formed_scene(cb,config=cfg)
    output.append(dict(provenance=s.provenance, formed=s.formed.to_json(), rounds=s.round_truth.to_json(),
                       images={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in s.rounds.items()},
                       signals=hashlib.sha256(s.realized.tobytes()).hexdigest()))
print(json.dumps(output,sort_keys=True))
'''
    outputs = [subprocess.check_output([sys.executable, '-c', code], cwd=tmp_path, timeout=120,
                                       env=dict(os.environ, PYTHONHASHSEED=str(seed))) for seed in (1, 991)]
    assert outputs[0] == outputs[1]
    assert list(tmp_path.iterdir()) == []
    data = json.loads(outputs[0])
    assert len(data) == 6
    # Clean fixtures have explicit coordinates and constant/supplied properties,
    # so they draw nothing; every recorded stream uses the documented namespace.
    streams = [s for preset in data for s in preset['provenance']['stream_scheme']['streams']]
    assert streams and all(s[0] == 'starfinder.synthetic/1' for s in streams)
    assert not data[2]['provenance']['stream_scheme']['streams']


def test_bounded_downstream_example():
    import runpy
    example = Path(__file__).resolve().parents[3] / 'docs/examples/formed_scene.py'
    check = runpy.run_path(str(example))['check_scene']
    for depth in (3, 1):
        check(depth)


def test_pinned_independent_component_draws():
    # Independently evaluated from the canonical descriptor, SHA-256 and PCG64
    # laws, without importing the production stream helper. Uniform draws are
    # plain IEEE arithmetic and must match exactly. Lognormal properties go
    # through exp, whose result NumPy dispatches per CPU (e.g. AVX-512 vs libm),
    # so they may differ by one ULP between hosts and are compared to rounding.
    # These literals also detect accidentally sharing a stream across properties.
    law = ScalarDistribution('lognormal', (.2, .1))
    result = scene(count=1, brightness=law, axial_width=law, lateral_width=law,
                   elongation=ScalarDistribution('folded_lognormal', (.2, .1)),
                   angle=ScalarDistribution('uniform', (0, np.pi)))
    np.testing.assert_array_equal(result.formed[['z', 'y', 'x']].to_numpy()[0],
                                  [5.52290136568776, 20.31117785750813, 18.81900376172695])
    np.testing.assert_allclose(result.formed[['A', 'sz', 'sl', 'e']].to_numpy()[0],
                               [1.438641251074506, 1.1970397659789114, 1.2626294282763126,
                                1.2690413185721265], rtol=1e-15, atol=0)
    assert result.formed.theta.iloc[0] == 1.5668133208553605
    assert result.formed.gene_id.tolist() == ['b']
