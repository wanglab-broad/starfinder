"""Bounded independent scene/result contracts; no scientific qualification."""
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from starfinder.synthetic import (
    SyntheticConfig, generate_dataset, generate_registration_pairs,
    generate_volume, render_spots,
)
from starfinder.synthetic import _generation, _presets, _rendering
from starfinder.synthetic._truth import _scene_table, _truth_rows


def test_independent_import_and_removed_namespace():
    subprocess.run([sys.executable, '-c', '''
import sys, importlib.util
import starfinder.synthetic as synthetic
assert not any(n == 'starfinder.benchmark' or n.startswith('starfinder.benchmark.') for n in sys.modules)
import starfinder.benchmark as benchmark
assert importlib.util.find_spec('starfinder.benchmark.synthetic') is None
for name in ['SyntheticConfig', 'generate_synthetic_dataset', 'create_test_volume',
             'generate_codebook', 'SIZE_PRESETS', 'get_size_preset', 'SpotTuple']:
    assert not hasattr(benchmark, name), name
assert not hasattr(synthetic, 'SpotTuple')
'''], check=True)


def test_dataset_labels_identity_eligibility_and_config_copy(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = SyntheticConfig(n_z=8, height=32, width=32, n_fovs=2,
        n_spots_per_fov=8, seed=17, max_shift_xy=30, max_shift_z=7,
        deformation='polynomial_small')
    result = generate_dataset(config)
    assert list(tmp_path.iterdir()) == []
    assert result.molecular_truth is None
    assert result.config == config and result.config is not config
    assert result.channel_labels == ('ch00', 'ch01', 'ch02', 'ch03')
    assert result.provenance['scientific_owner'] == 'W-93'
    table = result.spot_truth
    assert len(table) == 2 * 4 * 8
    assert not table.duplicated(['spot_namespace', 'round_label', 'spot_id']).any()
    assert set(table.spot_id) == set(range(8))
    assert not table.rendered.all()
    assert table.loc[~table.rendered, 'eligibility_reason'].notna().all()
    assert np.array_equal(table.rendered, table.eligible)
    assert not table.molecular_truth_eligible.any()
    for fov, rounds in result.rounds.items():
        assert list(rounds) == ['round1', 'round2', 'round3', 'round4']
        assert result.perturbations[fov]['direction'] == 'reference_to_moving'
        for label, image in rounds.items():
            assert image.shape == (8, 32, 32, 4) and image.dtype == np.uint8
            metadata = result.metadata[fov][label]
            assert metadata.frame_id == f'{fov}/{label}'
            assert metadata.spacing_zyx is None and metadata.spatial_unit is None


def test_one_renderer_for_all_generators(monkeypatch):
    calls = []
    original = render_spots
    def observed(*args, **kwargs):
        calls.append(kwargs['shape'])
        return original(*args, **kwargs)
    monkeypatch.setattr(_generation, 'render_spots', observed)
    monkeypatch.setattr(_rendering, 'render_spots', observed)
    monkeypatch.setitem(_presets.SIZE_PRESETS, 'tiny', (8, 32, 32))
    generate_volume((8, 32, 32), n_spots=2, seed=4)
    assert len(calls) == 1
    generate_dataset(SyntheticConfig(n_z=8, height=32, width=32, n_fovs=1, n_spots_per_fov=2))
    assert len(calls) == 17
    pairs = generate_registration_pairs(['tiny'], seed=4)['tiny']
    assert len(calls) == 25  # reference plus 7 independent moving images
    assert pairs.provenance['effective_seeds']['shift'] == 4 + hash('tiny') % 10000
    for label, perturbation in pairs.perturbations.items():
        assert perturbation['direction'] == 'reference_to_moving'
        assert perturbation['units'] == 'voxel_index'
    assert pairs.spot_truth.shape[0] == 80
    assert pairs.molecular_truth is None


def test_continuous_displacement_rounding_and_dropout():
    field = np.full((8, 16, 16, 3), 0.5, dtype=np.float32)
    rows = _truth_rows([(2, 3, 4, 200, 1.5), (7, 15, 15, 200, 1.5)],
        namespace='scene', round_label='moving', shape=(8, 16, 16), field=field)
    assert [rows[0][c] for c in ('continuous_z', 'continuous_y', 'continuous_x')] == [2.5, 3.5, 4.5]
    assert [rows[0][c] for c in ('z', 'y', 'x')] == [2, 4, 4]
    assert not rows[1]['rendered'] and rows[1]['eligibility_reason'] == 'outside_after_deformation_rounding'


def test_scene_validation_fractional_and_empty_results():
    table = _scene_table([(4.25, 8.5, 8.75, 200, 1.5)])
    image = render_spots((8, 16, 16), table, seed=4, dtype='uint16')
    assert image.dtype == np.uint16 and image.max() > 100
    with pytest.raises(TypeError, match='DataFrame'):
        render_spots((8, 16, 16), [], seed=4)
    with pytest.raises(ValueError, match='unique'):
        render_spots((8, 16, 16), pd.concat([table, table]), seed=4)
    result = generate_dataset(SyntheticConfig(n_z=8, height=16, width=16, n_fovs=1, n_spots_per_fov=0))
    assert result.spot_truth.empty
    assert result.spot_truth.spot_id.dtype == np.int64
    assert result.spot_truth.rendered.dtype == bool


@pytest.mark.parametrize(('dtype', 'expected'), [
    ('uint8', '5e82d9085b16fc26e10ad2eb313d8b49a7a9f510a8607e9887111979ff799401'),
    ('uint16', '9ccefcfe70f9e9abb9e8f9ca7a5a4a6482cc9e173b30f252f9285adb4c9ade33'),
])
def test_historical_integer_renderer_fixture(dtype, expected):
    # Captured from 0710a803 benchmark/synthetic.py; bounded source-preservation
    # fixture, not qualification of molecular truth or cross-process hash seeds.
    import hashlib
    spots = _scene_table([(0, 0, 0, 220, 1.5), (4, 16, 16, 200, .9), (7, 31, 31, 240, 2.2)])
    image = render_spots((8, 32, 32), spots, seed=17, dtype=dtype)
    assert hashlib.sha256(image.tobytes()).hexdigest() == expected


def test_missing_color_channel_remains_explicitly_ineligible():
    result = generate_dataset(SyntheticConfig(n_z=8, height=16, width=16,
        n_fovs=1, n_channels=1, n_spots_per_fov=2, codebook=[('GeneA', 'CACGC')]))
    truth = result.spot_truth
    assert len(truth) == 8
    assert not truth.rendered.any()  # reversed CACGC -> 4422, none in channel 1
    assert set(truth.eligibility_reason) == {'channel_not_rendered'}
    assert truth[['z', 'y', 'x', 'intensity', 'sigma']].isna().all().all()
