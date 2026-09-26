"""Unified generator contracts: datasets, streams, dtypes, noise, geometry and presets.

Expectations are literals, closed forms or replays of the documented stream key
(formed_oracle); none is derived from the producer's rendering path.
"""
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

import starfinder.synthetic as synthetic
from starfinder.synthetic import (BENCHMARK_PRESETS, BackgroundConfig, DEFORMATION_PRESETS,
                                  GeometryConfig, NoiseConfig, SCENE_PRESETS, benchmark_scene_preset,
                                  deformation_geometry, formed_scene_preset, forward_displacement,
                                  generate_codebook, generate_dataset, generate_formed_scene,
                                  generate_registration_pair)
from starfinder.synthetic import _geometry, _observation
from starfinder.synthetic._formed import _prepare
from starfinder.synthetic._presets import _estimate_peak_bytes, registration_scene_preset

from .formed_oracle import generator

ROOT = Path(__file__).resolve().parents[1]


def bounded(name='tiny', *, preset_dtype='uint16', **changes):
    """A benchmark preset reduced to a bounded grid; every appearance default kept."""
    book, config = benchmark_scene_preset(name, dtype=preset_dtype)
    return book, replace(config, shape_zyx=(4, 48, 48), count=6, **changes)


def digest(result):
    h = hashlib.sha256()
    for fov in sorted(result.rounds):
        for label, image in result.rounds[fov].items():
            h.update(label.encode())
            h.update(image.tobytes())
    for table in (result.formed, result.round_truth):
        h.update(table.to_json(orient='table').encode())
    return h.hexdigest()


def test_old_generator_api_is_removed():
    for name in ('SyntheticConfig', 'get_preset_config', 'generate_registration_pairs',
                 'render_spots', 'generate_volume', 'generate_displacement_field'):
        assert not hasattr(synthetic, name), name
    for module in ('_config', '_generation', '_rendering', '_fields', '_perturbations', '_truth'):
        assert not (ROOT / 'starfinder/synthetic' / f'{module}.py').exists(), module
    subprocess.run([sys.executable, '-c', '''
import sys, importlib.util
import starfinder.synthetic
assert not any(n == 'starfinder.benchmark' or n.startswith('starfinder.benchmark.') for n in sys.modules)
import starfinder.benchmark as benchmark
assert importlib.util.find_spec('starfinder.benchmark.synthetic') is None
for name in ['SyntheticConfig', 'generate_synthetic_dataset', 'create_test_volume', 'SIZE_PRESETS']:
    assert not hasattr(benchmark, name), name
'''], check=True, timeout=120)


def test_one_registry_holds_every_tier():
    tiers = {name: record['tier'] for name, record in SCENE_PRESETS.items()}
    assert {n for n, t in tiers.items() if t == 'benchmark'} == set(BENCHMARK_PRESETS)
    assert {n for n, t in tiers.items() if t == 'fixture'} == {'formed-small-v1', 'formed-z1-v1'}
    assert {n for n, t in tiers.items() if t == 'development_fixture'} == set(synthetic.DEVELOPMENT_FIXTURES)
    for name in SCENE_PRESETS:
        book, config = formed_scene_preset(name)
        assert tuple(config.shape_zyx) == tuple(SCENE_PRESETS[name]['shape_zyx'])
    with pytest.raises(ValueError, match='unknown'):
        formed_scene_preset('xlarge')


@pytest.mark.parametrize('name,shape,count,seed,z,yx', [
    ('tiny', (8, 128, 128), 10, 42, 2, 5), ('small', (16, 256, 256), 50, 42, 2, 5),
    ('medium', (32, 512, 512), 400, 42, 8, 50), ('large', (30, 1024, 1024), 1500, 123, 7, 100),
    ('tissue', (30, 3072, 3072), 14000, 456, 7, 300), ('thick_medium', (100, 1024, 1024), 5200, 789, 25, 100),
])
def test_benchmark_presets_keep_historical_sizes_counts_and_shifts(name, shape, count, seed, z, yx):
    book, config = benchmark_scene_preset(name)
    assert tuple(config.shape_zyx) == shape and config.count == count and config.seed == seed
    assert config.geometry.translation_max_zyx == (z, yx, yx)
    assert config.geometry.reference_round == 'round1' == book.round_labels[0]
    assert book.channel_labels == ('ch00', 'ch01', 'ch02', 'ch03')
    assert config.dtype == 'uint16' and config.accumulation is None
    assert config.noise.model == 'poisson' and config.background.regions_enabled
    assert config.dataset_version == f'{synthetic.PRESET_VERSION}-{name}'
    # Every color is used in every round, so no channel is empty by construction.
    for r in range(len(book.round_labels)):
        assert set(book.table.color_sequence.str[r]) == set('1234')


@pytest.mark.parametrize('name', ['large', 'tissue', 'thick_medium'])
def test_large_presets_validate_by_configuration_and_memory_estimate(name):
    """Configuration and memory estimate only; these presets are never rendered here."""
    for book, config in (benchmark_scene_preset(name), registration_scene_preset(name, 'multi_point')):
        plan = _prepare(book, config, None, {})
        assert plan.n == BENCHMARK_PRESETS[name]['count'] and plan.accumulation == 'float32'
        assert len(plan.maps) == len(book.round_labels)
    estimate = SCENE_PRESETS[name]['peak_bytes_estimate']
    voxels = int(np.prod(BENCHMARK_PRESETS[name]['shape_zyx']))
    # Output round (uint16 ZYXC), the writer's one-channel copy and two float32
    # planes dominate; far below tens of GB.
    assert estimate >= voxels * (5 * 2 + 2 * 4)
    assert estimate < 6 * 2**30
    print(f'{name}: estimated peak working set {estimate / 2**30:.2f} GiB')


def test_dataset_fovs_own_stream_namespaces_and_appending_ids_changes_nothing():
    book, config = bounded()
    one = generate_dataset(book, config, fov_ids=('FOV_001',))
    two = generate_dataset(book, config, fov_ids=('FOV_001', 'FOV_002'))
    for label, image in one.rounds['FOV_001'].items():
        np.testing.assert_array_equal(image, two.rounds['FOV_001'][label])
    assert one.provenance['FOV_001']['image_sha256'] == two.provenance['FOV_001']['image_sha256']
    first = two.formed[two.formed.namespace.str.contains('FOV_001')].reset_index(drop=True)
    pd.testing.assert_frame_equal(one.formed, first)
    other = two.formed[two.formed.namespace.str.contains('FOV_002')]
    assert not np.allclose(first[['z', 'y', 'x']], other[['z', 'y', 'x']])
    for fov in ('FOV_001', 'FOV_002'):
        key = two.provenance[fov]['requested_config']['scene_key']
        assert json.loads(key) == [config.scene_key, fov]
    with pytest.raises(ValueError, match='unique'):
        generate_dataset(book, config, fov_ids=('FOV_001', 'FOV_001'))


def test_toggling_one_control_changes_no_other_draw():
    book, config = bounded()
    base = generate_dataset(book, config).rounds['FOV_001']
    # Disabling mixing changes amplitudes only: placements, noise draws and the
    # identity-held reference geometry are untouched.
    unmixed = generate_dataset(book, replace(config, readout=replace(config.readout, mixing_enabled=False)))
    pd.testing.assert_frame_equal(generate_dataset(book, config).formed, unmixed.formed)
    # Noise-free variants isolate the shared noise draws: toggling noise model or
    # strengths leaves every standardized draw of the other component fixed.
    clean = replace(config, noise=NoiseConfig(), dtype='float64')
    read = replace(clean, noise=NoiseConfig(independent_enabled=True, sigma=3.))
    read_double = replace(clean, noise=NoiseConfig(independent_enabled=True, sigma=6.))
    images = [generate_dataset(book, c).rounds['FOV_001'] for c in (clean, read, read_double)]
    for label in images[0]:
        np.testing.assert_allclose(images[2][label] - images[0][label],
                                   2 * (images[1][label] - images[0][label]), rtol=0, atol=1e-9)
    # A new geometry translation range redraws only the geometry stream.
    moved = replace(config, geometry=replace(config.geometry, translation_max_zyx=(1, 9, 9)))
    other = generate_dataset(book, moved).rounds['FOV_001']
    np.testing.assert_array_equal(other['round1'], base['round1'])
    assert not np.array_equal(other['round2'], base['round2'])


def test_cross_process_byte_repeatability():
    code = '''
import sys
sys.path.insert(0, sys.argv[1])
from test.test_synthetic_contracts import bounded, digest
from starfinder.synthetic import generate_dataset, generate_registration_pair
book, config = bounded()
print(digest(generate_dataset(book, config, fov_ids=('FOV_001', 'FOV_002'))), hash('probe'))
'''
    outputs = [subprocess.check_output([sys.executable, '-c', code, str(ROOT)], text=True, timeout=300,
                                       env=dict(os.environ, PYTHONHASHSEED=value)).split()
               for value in ('3', '4242')]
    assert outputs[0][1] != outputs[1][1]
    assert outputs[0][0] == outputs[1][0]


@pytest.mark.parametrize('dtype', ['uint8', 'uint16'])
def test_integer_outputs_are_rounded_clipped_float32_accumulation(dtype):
    book, config = bounded(preset_dtype=dtype)
    floating = generate_dataset(book, replace(config, dtype='float32')).rounds['FOV_001']
    quantized = generate_dataset(book, config).rounds['FOV_001']
    limit = np.iinfo(dtype).max
    for label, image in quantized.items():
        assert image.dtype == np.dtype(dtype) and image.shape == (4, 48, 48, 4)
        np.testing.assert_array_equal(image, np.clip(np.rint(floating[label]), 0, limit).astype(dtype))
    assert quantized['round1'].max() > 0


def test_uint8_preset_scales_intensities_and_stays_mostly_unclipped():
    book, config = benchmark_scene_preset('tiny', dtype='uint8')
    assert config.dtype == 'uint8'
    assert config.brightness.parameters[0] == pytest.approx(np.log(1500 / 16))
    scene = generate_formed_scene(book, config=replace(config, shape_zyx=(4, 48, 48), count=6))
    total = sum(sum(c.values()) for c in scene.provenance['clipping_counts'].values())
    assert total <= .01 * 4 * 4 * 48 * 48 * 4


def test_accumulation_default_and_float64_option():
    book, config = bounded(dtype='float32', noise=NoiseConfig())
    single = generate_formed_scene(book, config=config)
    double = generate_formed_scene(book, config=replace(config, accumulation='float64'))
    assert single.provenance['effective_config']['accumulation'] == 'float32'
    assert double.provenance['effective_config']['accumulation'] == 'float64'
    for label, image in single.rounds.items():
        np.testing.assert_allclose(image, double.rounds[label], rtol=1e-5, atol=1e-3)
    assert generate_formed_scene(book, config=replace(config, dtype='float64')).provenance[
        'effective_config']['accumulation'] == 'float64'
    with pytest.raises(ValueError, match='accumulation'):
        generate_formed_scene(book, config=replace(config, accumulation='float16'))


def test_poisson_noise_replays_the_dependent_stream_and_read_noise():
    book, base = formed_scene_preset()
    baseline = np.array([[100., 50., 20., 0.]] * 3)
    config = replace(base, count=0, shape_zyx=(4, 32, 32), dtype='float64',
                     background=BackgroundConfig(baseline_enabled=True, baseline=baseline),
                     noise=NoiseConfig(dependent_enabled=True, alpha=2., model='poisson',
                                       independent_enabled=True, sigma=.5))
    scene = generate_formed_scene(book, config=config)
    assert scene.provenance['effective_config']['noise']['model'] == 'poisson'
    for r, label in enumerate(scene.round_labels):
        for c, channel in enumerate(scene.channel_labels):
            counts = generator('noise.dependent', round_label=label, channel_label=channel).poisson(
                np.full((4, 32, 32), baseline[r, c] / 2))
            read = generator('noise.independent', round_label=label, channel_label=channel).standard_normal(
                (4, 32, 32))
            np.testing.assert_allclose(scene.rounds[label][..., c], 2 * counts + .5 * read, rtol=0, atol=1e-12)
    # Mean J and variance alpha*J + sigma^2 within five standard errors (N = 12288).
    values = np.stack([scene.rounds[label][..., 0] for label in scene.round_labels])
    n = values.size
    assert abs(values.mean() - 100) < 5 * np.sqrt(200.25 / n)
    assert abs(values.var() - 200.25) < 5 * 200.25 * np.sqrt(2 / n)
    with pytest.raises(ValueError, match='gaussian or poisson'):
        generate_formed_scene(book, config=replace(config, noise=NoiseConfig(model='uniform')))


def test_clipping_warning_threshold_and_counts():
    book, base = formed_scene_preset()
    over = replace(base, count=0, shape_zyx=(1, 10, 10), dtype='uint8',
                   background=BackgroundConfig(baseline_enabled=True, baseline=[[300, 0, 0, 0]] * 3))
    with pytest.warns(RuntimeWarning, match=r'100 of 400 voxel values \(25.0%\)'):
        scene = generate_formed_scene(book, config=over)
    assert scene.provenance['clipping_counts']['round1'] == dict(below=0, above=100)
    # Exactly 1% clipped (4 of 400) stays silent; the count is still recorded.
    one = replace(over, count=None, coordinates=((0, 5, 5),), amplicon_ids=('a',), gene_ids={'a': 'gene-A'},
                  background=BackgroundConfig(), brightness=synthetic.ScalarDistribution(parameters=(300,)))
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        quiet = generate_formed_scene(book, config=one)
    assert sum(c['above'] for c in quiet.provenance['clipping_counts'].values()) >= 1


def test_chunked_noise_and_blockwise_inverse_match_whole_grid(monkeypatch):
    book, config = bounded(dtype='float64')
    config = replace(config, geometry=deformation_geometry('gaussian_large', config.shape_zyx,
                                                           reference_round='round1'))
    whole = generate_formed_scene(book, config=config)
    monkeypatch.setattr(_observation, '_NOISE_CHUNK', 1000)
    monkeypatch.setattr(_geometry, '_BLOCK_VOXELS', 3000)
    blocked = generate_formed_scene(book, config=config)
    transforms = list(blocked.provenance['transforms'].values())
    assert max(t['inverse']['blocks'] for t in transforms if t['inverse'].get('blocks')) > 1
    for label, image in whole.rounds.items():
        # Chunked draws are bitwise equal; block-wise inverse agrees within its tolerance.
        np.testing.assert_allclose(blocked.rounds[label], image, rtol=0, atol=1e-6)
    assert all(t['inverse']['max_residual'] <= 2e-10 for t in transforms)


def test_affine_and_polynomial_forward_maps_are_literal():
    affine = ((0, 0, 0), (.01, .02, 0), (0, -.01, .03))
    polynomial = ((0,) * 6, (.5, 0, 0, 0, 0, 1.), (0, 0, -.5, 0, .25, 0))
    book, base = formed_scene_preset()
    geometry = GeometryConfig(affine_enabled=True, affine_zyx=(affine,) * 3,
                              polynomial_enabled=True, polynomial_zyx=(polynomial,) * 3,
                              reference_round='round10')
    config = replace(base, shape_zyx=(5, 21, 41), coordinates=((1, 3, 30),), amplicon_ids=('p',),
                     gene_ids={'p': 'gene-A'}, dtype='float64', geometry=geometry)
    scene = generate_formed_scene(book, config=config)
    q = np.array([1., 3., 30.])
    center, h = np.array([2., 10., 20.]), 20.
    u = (q - center) / h
    monomials = np.array([u[0]**2, u[1]**2, u[2]**2, u[0]*u[1], u[0]*u[2], u[1]*u[2]])
    expected = q + np.array(affine) @ (q - center) + np.array(polynomial) @ monomials
    truth = scene.round_truth.set_index('round_label')
    np.testing.assert_allclose(truth.loc['round2', ['z', 'y', 'x']], expected, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(truth.loc['round10', ['z', 'y', 'x']], q)
    record = next(t for t in scene.provenance['transforms'].values() if t['round_label'] == 'round2')
    assert record['kind'] == 'polynomial_affine_gaussian_rbf_translation'
    field = forward_displacement(record, (5, 21, 41))
    np.testing.assert_allclose(field[1, 3, 30], expected - q, rtol=0, atol=1e-6)
    np.testing.assert_allclose(forward_displacement(record, (5, 21, 41), z=slice(1, 2))[0], field[1])


def test_random_affine_polynomial_and_magnitude_bounds():
    book, base = formed_scene_preset()
    shape = (6, 40, 60)
    extent = (np.array(shape) - 1) / 2
    geometry = GeometryConfig(affine_enabled=True, affine_max_zyx=(.1, 1., 2.),
                              polynomial_enabled=True, polynomial_max_zyx=(.1, 1., 2.),
                              local_enabled=True, centers_zyx=((2, 20, 30),), scales=(10,),
                              local_magnitude=1.5, reference_round='round10')
    scene = generate_formed_scene(book, config=replace(base, shape_zyx=shape, count=0, geometry=geometry))
    rounds = scene.provenance['effective_config']['geometry']['rounds']
    for label, record in rounds.items():
        if label == 'round10':
            assert not np.any(record['translation_zyx']) and not np.any(record['vectors_zyx'])
            assert 'affine_zyx' not in record
            continue
        # Affine rows attain their bound exactly at a grid corner.
        np.testing.assert_allclose(np.abs(record['affine_zyx']) @ extent, [.1, 1., 2.], rtol=1e-12)
        assert np.linalg.norm(record['vectors_zyx'][0]) == pytest.approx(1.5, abs=1e-12)
        # Affine and polynomial bounds add; the RBF adds at most its magnitude.
        displacement = np.abs(forward_displacement(record, shape)).reshape(-1, 3).max(axis=0)
        assert (displacement <= np.array([.1, 1., 2.]) * 2 + 1.5 + 1e-6).all()
        assert record['lipschitz_bound'] <= .5
    with pytest.raises(ValueError, match='select supplied vectors, strength or local_magnitude'):
        generate_formed_scene(book, config=replace(base, geometry=GeometryConfig(
            local_enabled=True, centers_zyx=((1, 2, 2),), strength=.1, local_magnitude=.1)))
    with pytest.raises(ValueError, match='reference_round'):
        generate_formed_scene(book, config=replace(base, geometry=GeometryConfig(reference_round='nope')))
    with pytest.raises(ValueError, match='Z=1'):
        generate_formed_scene(book, config=replace(base, shape_zyx=(1, 8, 8), geometry=GeometryConfig(
            affine_enabled=True, affine_zyx=(((.1, 0, 0), (0, 0, 0), (0, 0, 0)),) * 3)))
    with pytest.raises(ValueError, match='invertibility'):
        generate_formed_scene(book, config=replace(base, geometry=GeometryConfig(
            affine_enabled=True, affine_zyx=((((.6, 0, 0), (0, 0, 0), (0, 0, 0)),) * 3))))


@pytest.mark.parametrize('name', ['shift', *DEFORMATION_PRESETS])
def test_deformation_presets_are_invertible_percent_of_size(name):
    for shape in ((8, 128, 128), (32, 512, 512), (30, 3072, 3072)):
        geometry = deformation_geometry(name, shape, reference_round='reference')
        cb, config = registration_scene_preset('tiny', name)
        # Validation includes the conservative invertibility bound for this shape.
        _prepare(cb, replace(config, shape_zyx=shape, count=0, geometry=geometry,
                             background=BackgroundConfig()), None, {})
        if name != 'shift':
            spec = DEFORMATION_PRESETS[name]
            lateral = min(spec['percent'] * min(shape[1:]) / 100, spec['cap_px'])
            if spec['kind'] == 'rbf':
                assert geometry.local_magnitude <= lateral
                assert geometry.scales[0] == spec['radius_percent'] * min(shape[1:]) / 100
            else:
                key = 'polynomial_max_zyx' if spec['kind'] == 'polynomial' else 'affine_max_zyx'
                assert getattr(geometry, key) == (spec['percent'] * shape[0] / 100, lateral, lateral)


@pytest.mark.parametrize('preset', ['tiny', 'small', 'medium'])
def test_random_polynomial_and_affine_draws_always_meet_the_invertibility_bound(preset):
    # Seeds that exceeded the bound before random draws were capped to the
    # remaining budget (tiny 366 crashed the registration CLI), plus a range.
    known = {'tiny': [366], 'small': [290], 'medium': [64, 78, 226]}[preset]
    names = [n for n, spec in DEFORMATION_PRESETS.items() if spec['kind'] in ('polynomial', 'affine')]
    for name in names:
        cb, config = registration_scene_preset(preset, name)
        for seed in known + list(range(40)):
            state = _prepare(cb, replace(config, seed=seed, count=0, background=BackgroundConfig()), None, {})
            assert len(state.maps) == 2
            assert all(m['lipschitz_bound'] <= .5 for m in state.maps), (name, seed)


def test_documented_effective_multi_point_magnitudes():
    # Keep the table in docs/api/synthetic.rst in step with deformation_geometry.
    documented = {'tiny': 1.32, 'small': 2.64, 'medium': 5.27, 'large': 10.54,
                  'thick_medium': 10.54, 'tissue': 20.0}
    for name, used in documented.items():
        geometry = deformation_geometry('multi_point', BENCHMARK_PRESETS[name]['shape_zyx'])
        assert round(geometry.local_magnitude, 2) == used, name
    for name in ('gaussian_small', 'gaussian_large'):
        spec = DEFORMATION_PRESETS[name]
        for preset in documented:
            shape = BENCHMARK_PRESETS[preset]['shape_zyx']
            requested = min(spec['percent'] * min(shape[1:]) / 100, spec['cap_px'])
            assert deformation_geometry(name, shape).local_magnitude == requested


def test_registration_pairs_share_one_scene_and_record_forward_maps():
    first = generate_registration_pair('tiny', deformation='shift')
    second = generate_registration_pair('tiny', deformation='linear_small')
    np.testing.assert_array_equal(first.rounds['tiny']['reference'], second.rounds['tiny']['reference'])
    pd.testing.assert_frame_equal(first.formed, second.formed)
    shift = np.array(first.historical_truth['pairs']['shift']['shift_zyx'])
    truth = first.round_truth.set_index(['amplicon_id', 'round_label'])
    for identity in first.formed.amplicon_id:
        np.testing.assert_allclose(truth.loc[(identity, 'shift'), ['z', 'y', 'x']].to_numpy()
                                   - truth.loc[(identity, 'reference'), ['z', 'y', 'x']].to_numpy(),
                                   shift, rtol=0, atol=1e-12)
    assert np.all(np.abs(shift) <= [2, 10, 10])
    skipped = generate_registration_pair('tiny', deformation='linear_small', include_reference=False)
    assert list(skipped.rounds['tiny']) == ['linear_small']
    np.testing.assert_array_equal(skipped.rounds['tiny']['linear_small'], second.rounds['tiny']['linear_small'])
    pd.testing.assert_frame_equal(skipped.round_truth, second.round_truth)
    image = second.rounds['tiny']['reference']
    assert image.dtype == np.uint16 and image.shape == (8, 128, 128, 4)


def test_historical_truth_is_derived_from_round_truth():
    book, config = bounded()
    result = generate_dataset(book, config, fov_ids=('FOV_001', 'FOV_002'), preset='tiny')
    truth = result.historical_truth
    assert truth['version'] == '2.0' and truth['n_rounds'] == 4 and truth['n_channels'] == 4
    assert truth['image_shape'] == [4, 48, 48] and truth['n_genes'] == 12
    for fov, record in truth['fovs'].items():
        rows = result.round_truth[result.round_truth.namespace.str.contains(fov)]
        reference = rows[rows.round_label == 'round1'].set_index('amplicon_id')
        for spot in record['spots']:
            np.testing.assert_array_equal(spot['position'], reference.loc[spot['id'], ['z', 'y', 'x']])
            assert book.gene_to_seq[spot['gene']] == spot['color_seq']
        for label, shift in record['shifts'].items():
            moved = rows[rows.round_label == label].set_index('amplicon_id')
            np.testing.assert_allclose(moved[['z', 'y', 'x']] - reference[['z', 'y', 'x']],
                                       np.tile(shift, (len(moved), 1)), rtol=0, atol=1e-12)
    spots = result.spot_truth
    assert len(spots) == len(result.round_truth)
    assert set(spots.channel_label) <= set(book.channel_labels)
    assert not spots.molecular_truth_eligible.any()


def test_dataset_reference_round_never_moves_and_other_draws_are_unchanged():
    book, config = bounded()
    # Without a reference round in the geometry, the first round is the
    # reference and is held at identity; every other round keeps its draws.
    unset = replace(config, geometry=replace(config.geometry, reference_round=None))
    implicit = generate_dataset(book, unset, fov_ids=('FOV_001',))
    explicit = generate_dataset(book, config, fov_ids=('FOV_001',))
    assert implicit.historical_truth['reference_round'] == 'round1'
    assert implicit.historical_truth['fovs']['FOV_001']['shifts']['round1'] == [0.0, 0.0, 0.0]
    truth = implicit.round_truth.set_index(['amplicon_id', 'round_label'])
    formed = implicit.formed.set_index('amplicon_id')
    for identity in formed.index:
        moved = truth.loc[(identity, 'round1'), ['z', 'y', 'x']].to_numpy(dtype=float)
        np.testing.assert_array_equal(moved, formed.loc[identity, ['z', 'y', 'x']].to_numpy(dtype=float))
    pd.testing.assert_frame_equal(implicit.round_truth, explicit.round_truth)
    assert digest(implicit) == digest(explicit)


def test_generate_codebook_returns_balanced_codebook():
    book = generate_codebook(64)
    assert book.n_genes == 64 and book.round_labels == ('round1', 'round2', 'round3', 'round4')
    assert book.encoding.reverse_bases
    assert generate_codebook(8).table.base_sequence.str.fullmatch('C[ACGT]{3}C').all()
    with pytest.raises(ValueError, match='unique color sequences'):
        generate_codebook(65)
    with pytest.raises(ValueError, match='rounds'):
        generate_codebook(4, rounds=5)


def test_peak_estimate_formula():
    assert _estimate_peak_bytes((1, 1, 1)) > 0
    medium = _estimate_peak_bytes((32, 512, 512))
    # Working memory only (the interpreter and libraries add their own RSS).
    assert 200 * 2**20 < medium < 1024 * 2**20


@pytest.mark.parametrize('name', ['multi_point', 'polynomial_large', 'linear_small'])
def test_per_point_inverse_matches_whole_block_iteration(name):
    shape = (4, 64, 64)
    cb, config = registration_scene_preset('tiny', name)
    config = replace(config, shape_zyx=shape, count=0,
                     geometry=deformation_geometry(name, shape, reference_round='reference'))
    mapping = _prepare(cb, config, None, {}).maps[1]
    grid = np.moveaxis(np.indices(shape, dtype=np.float64), 0, -1)
    block, block_diag = _geometry._inverse(grid, mapping)
    point, point_diag = _geometry._inverse_per_point(grid, mapping)
    # Same fixed point and stopping tolerance; per-point stopping may differ by < 1e-10.
    np.testing.assert_allclose(point, block, rtol=0, atol=2e-10)
    assert point_diag['stopping'] == 'per_point' and point_diag['max_residual'] <= 2e-10
    assert point_diag['iterations'] == block_diag['iterations']
    np.testing.assert_allclose(_geometry._forward(point, mapping), grid, rtol=0, atol=2e-10)
    scene = generate_formed_scene(cb, config=config)
    moving = next(t for t in scene.provenance['transforms'].values() if t['round_label'] == name)
    assert moving['inverse']['stopping'] == 'per_point'
