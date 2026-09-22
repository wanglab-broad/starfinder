"""Independent frozen W-155 literals and complete-population checks for W-157."""
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
from starfinder.synthetic import (FormedSceneConfig, ScalarDistribution,
                                  formed_scene_preset, generate_formed_scene)


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
        assert result.round_truth.support_truncated.all()
        assert result.provenance['extensions']['starfinder.synthetic']['singleton_z_sampling']


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
    dict(seed=True), dict(seed=-1), dict(seed=2**64), dict(split='evaluation'),
    dict(shape_zyx=(1, 2, 65)), dict(shape_zyx=(0, 2, 3)), dict(shape_zyx=(1, True, 3)),
    dict(coordinates=((1, 2),)), dict(coordinates=((np.nan, 2, 3),)),
    dict(shape_zyx=(1, 3, 3), coordinates=((1, 2, 2),)),
    dict(count=2, amplicon_ids=('same', 'same')), dict(count=1, amplicon_ids=('e\u0301',)),
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
    dict(brightness=ScalarDistribution('supplied', (), {})),
    dict(brightness=ScalarDistribution('unknown', (1,))),
])
def test_invalid_inputs(kwargs):
    with pytest.raises((ValueError, TypeError)):
        scene(**kwargs)


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


def test_quantization_and_provenance_geometry():
    metadata = ImageMetadata('ref', (2, 3, 4), (10, 20, 30), ((1, 0, 0), (0, 1, 0), (0, 0, 1)), 'um')
    for brightness, expected in [(.5, 0), (1.5, 2), (255.5, 255)]:
        result = generate_formed_scene(book(), config=FormedSceneConfig(coordinates=((1, 2, 2),),
                                      brightness=scalar(brightness), dtype='uint8'), metadata=metadata)
        assert result.rounds['round10'].max() == expected
        assert result.metadata == metadata
        payload = result.provenance['extensions']['starfinder.synthetic']
        assert payload['geometry']['spacing_zyx'] == (2., 3., 4.)
        assert payload['observation']['clipping_counts']['round10']['above'] == int(brightness > 255)
        assert json.loads(payload['truth_namespace']) == ['formed-small-v1', 'sample', 'FOV_001', 'formed']
        assert payload['contract_revision'] == 'f9512694a0960c10ce5236efbaaf9d6f425c1d8a'
        json.dumps(result.provenance, allow_nan=False)


def test_actual_cross_process_repeats_and_no_files(tmp_path):
    code = '''
import hashlib,json
from starfinder.synthetic import formed_scene_preset,generate_formed_scene
output=[]
for name in ('formed-small-v1','formed-z1-v1'):
    cb,cfg=formed_scene_preset(name)
    s=generate_formed_scene(cb,config=cfg)
    output.append(dict(provenance=s.provenance, formed=s.formed.to_json(), rounds=s.round_truth.to_json(),
                       images={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in s.rounds.items()},
                       signals=hashlib.sha256(s.realized.tobytes()).hexdigest()))
print(json.dumps(output,sort_keys=True))
'''
    outputs = [subprocess.check_output([sys.executable, '-c', code], cwd=tmp_path,
                                       env=dict(os.environ, PYTHONHASHSEED=str(seed))) for seed in (1, 991)]
    assert outputs[0] == outputs[1]
    assert list(tmp_path.iterdir()) == []
    data = json.loads(outputs[0])
    for preset in data:
        streams = preset['provenance']['extensions']['starfinder.synthetic']['streams']
        for stream in streams:
            serialized = json.dumps(stream['descriptor'], ensure_ascii=False, separators=(',', ':')).encode()
            assert hashlib.sha256(serialized).hexdigest() == stream['sha256']


def test_bounded_downstream_example():
    import runpy
    example = Path(__file__).resolve().parents[3] / 'docs/examples/formed_scene.py'
    check = runpy.run_path(str(example))['check_scene']
    for depth in (3, 1):
        check(depth)


def test_pinned_independent_component_draws():
    # Independently evaluated from the W-155 canonical descriptor, SHA-256 and
    # PCG64 laws in NumPy 2.2.6 and the exporter profile's NumPy 1.26.4,
    # without importing the production stream helper. Scalar exp differs by one
    # ULP for brightness; the contract promises exact replay within a pinned
    # environment, not bitwise equality between NumPy releases.
    # These literals also detect accidentally sharing a stream across properties.
    law = ScalarDistribution('lognormal', (.2, .1))
    result = scene(count=1, brightness=law, axial_width=law, lateral_width=law,
                   elongation=ScalarDistribution('folded_lognormal', (.2, .1)),
                   angle=ScalarDistribution('uniform', (0, np.pi)))
    np.testing.assert_array_equal(result.formed[['z', 'y', 'x']].to_numpy()[0],
                                  [5.52290136568776, 20.31117785750813, 18.81900376172695])
    expected_brightness = 1.4386412510745061 if np.__version__ == '1.26.4' else 1.438641251074506
    np.testing.assert_array_equal(result.formed[['A', 'sz', 'sl', 'e', 'theta']].to_numpy()[0],
                                  [expected_brightness, 1.1970397659789114, 1.2626294282763126,
                                   1.2690413185721265, 1.5668133208553605])
    assert result.formed.gene_id.tolist() == ['b']


def test_source_record_integrates_with_run_provenance(tmp_path):
    from starfinder.provenance import RunRecorder, read_run
    result = scene(count=0)
    recorder = RunRecorder(tmp_path / 'run', dataset_id='formed-small-v1', sample_id='sample',
                           sources=[result.provenance], owner='Jiahao')
    saved = read_run(recorder.path)
    source = saved['sources'][0]
    assert source['source_id'] == result.provenance['source_id']
    assert source['uri'] is None and source['sha256'] is None and source['unverified_reason']
    assert source['selection']['axes'] == 'ZYXC'
    payload = source['extensions']['starfinder.synthetic']
    assert payload['requested_config']['type'] == 'FormedSceneConfig'
    assert payload['requested_config']['brightness']['type'] == 'ScalarDistribution'
