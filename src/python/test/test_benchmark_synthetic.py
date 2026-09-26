"""CLI generation in both modes, the saved layout and the benchmark code that reads it."""
import json
import re
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import tifffile

from starfinder.__main__ import main
from starfinder.synthetic import BENCHMARK_PRESETS, DEFORMATION_PRESETS, SCENE_PRESETS

E2E_FILES = {'codebook.csv', 'formed.csv', 'generation.json', 'ground_truth.json',
             'manifest.json', 'round_truth.csv', 'scene_truth.csv'}
PAIR_FILES = {'ref.tif', 'mov_shift.tif', 'formed.csv', 'round_truth.csv', 'scene_truth.csv',
              'ground_truth.json', 'generation.json',
              *(f'mov_deform_{d}.tif' for d in DEFORMATION_PRESETS),
              *(f'field_{d}.npy' for d in DEFORMATION_PRESETS)}


def generate(tmp_path, mode, preset, *extra):
    output = tmp_path / f'{preset}-{mode}'
    assert main(['synthetic', 'generate', '--mode', mode, '--preset', preset, '--seed', '42',
                 '--owner', 'pytest', '--output', str(output), *extra]) == 0
    return output


def check_e2e_layout(root, preset, dtype='uint16'):
    shape = BENCHMARK_PRESETS[preset]['shape_zyx']
    assert {p.name for p in root.iterdir() if p.is_file()} == E2E_FILES
    fovs = sorted(p.name for p in root.iterdir() if p.is_dir())
    assert fovs == [f'FOV_{i + 1:03d}' for i in range(BENCHMARK_PRESETS[preset]['fovs'])]
    for fov in fovs:
        assert sorted(p.name for p in (root / fov).iterdir()) == ['round1', 'round2', 'round3', 'round4']
        for r in range(1, 5):
            names = sorted(p.name for p in (root / fov / f'round{r}').iterdir())
            assert names == ['ch00.tif', 'ch01.tif', 'ch02.tif', 'ch03.tif']
        image = tifffile.imread(root / fov / 'round2' / 'ch01.tif')
        assert image.shape == shape and image.dtype == np.dtype(dtype)
    codebook = pd.read_csv(root / 'codebook.csv')
    assert list(codebook.columns) == ['gene', 'barcode'] and len(codebook) == BENCHMARK_PRESETS[preset]['genes']
    truth = json.loads((root / 'ground_truth.json').read_text())
    assert truth['version'] == '2.0' and truth['preset'] == preset
    assert truth['image_shape'] == list(shape) and truth['n_rounds'] == 4 and truth['n_channels'] == 4
    formed = pd.read_csv(root / 'formed.csv')
    rounds = pd.read_csv(root / 'round_truth.csv')
    assert len(formed) == BENCHMARK_PRESETS[preset]['count'] * len(fovs)
    assert len(rounds) == 4 * len(formed) == len(pd.read_csv(root / 'scene_truth.csv'))
    for fov, record in truth['fovs'].items():
        assert record['shifts']['round1'] == [0.0, 0.0, 0.0]
        z, yx = BENCHMARK_PRESETS[preset]['e2e_shift']
        assert all(abs(s[0]) <= z and max(abs(s[1]), abs(s[2])) <= yx for s in record['shifts'].values())
        positions = np.array([s['position'] for s in record['spots']])
        ours = formed[formed.namespace.str.contains(f'"{fov}"')][['z', 'y', 'x']].to_numpy()
        np.testing.assert_allclose(positions, ours, rtol=0, atol=1e-9)
    generation = json.loads((root / 'generation.json').read_text())
    assert generation['mode'] == 'e2e' and generation['seed'] == 42 and generation['molecular_truth'] is None
    assert generation['generator_version'] == '6' and generation['preset_version'] == 'benchmark-presets-v1'
    manifest = json.loads((root / 'manifest.json').read_text())
    files = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()} - {'manifest.json'}
    assert {a['path'] for a in manifest['artifacts']} == files


def check_registration_layout(output, preset):
    root = output / 'synthetic' / preset
    shape = BENCHMARK_PRESETS[preset]['shape_zyx']
    assert {p.name for p in root.iterdir()} == PAIR_FILES
    for name in ('ref', 'mov_shift', *(f'mov_deform_{d}' for d in DEFORMATION_PRESETS)):
        image = tifffile.imread(root / f'{name}.tif')
        assert image.shape == shape and image.dtype == np.uint16, name
    for deformation in DEFORMATION_PRESETS:
        field = np.load(root / f'field_{deformation}.npy', mmap_mode='r')
        assert field.shape == (*shape, 3) and field.dtype == np.float32
    truth = json.loads((root / 'ground_truth.json').read_text())
    assert set(truth['pairs']) == {'shift', *DEFORMATION_PRESETS}
    assert truth['n_spots'] == BENCHMARK_PRESETS[preset]['count'] and truth['shape'] == list(shape)
    rounds = pd.read_csv(root / 'round_truth.csv')
    assert rounds.groupby('round_label').size().eq(truth['n_spots']).all()
    assert set(rounds.round_label) == {'reference', 'shift', *DEFORMATION_PRESETS}
    summary = json.loads((output / 'synthetic' / 'summary.json').read_text())
    assert summary['presets'][preset]['n_pairs'] == 7
    return root, truth, rounds


def test_session_small_dataset_uses_the_new_generator(small_dataset, small_ground_truth):
    check_e2e_layout(small_dataset, 'small')
    assert small_ground_truth['seed'] == 42
    assert json.loads((small_dataset / 'manifest.json').read_text())['command']['preset'] == 'small'


def test_tiny_e2e_cli_uint16_and_uint8(tmp_path):
    root = generate(tmp_path, 'e2e', 'tiny')
    check_e2e_layout(root, 'tiny')
    small = generate(tmp_path / 'u8', 'e2e', 'tiny', '--dtype', 'uint8')
    check_e2e_layout(small, 'tiny', 'uint8')
    clipping = json.loads((small / 'generation.json').read_text())['provenance']['FOV_001']['clipping_counts']
    assert all(c['below'] + c['above'] <= .01 * 8 * 128 * 128 * 4 for c in clipping.values())
    quiet = generate(tmp_path / 'clean', 'e2e', 'tiny', '--no-noise')
    noise = json.loads((quiet / 'generation.json').read_text())['provenance']['FOV_001'][
        'effective_config']['noise']
    assert not noise['dependent_enabled'] and not noise['independent_enabled']


def test_tiny_registration_cli_layout_fields_and_benchmark_run(tmp_path):
    from starfinder.benchmark import BenchmarkCase, evaluate_benchmark, run_benchmark
    output = generate(tmp_path, 'registration', 'tiny')
    root, truth, rounds = check_registration_layout(output, 'tiny')
    reference = rounds[rounds.round_label == 'reference'].set_index('amplicon_id')
    for deformation in DEFORMATION_PRESETS:
        # Nearest-voxel field agrees with moved centers to first order (Lipschitz <= 0.5).
        field = np.load(root / f'field_{deformation}.npy')
        moved = rounds[rounds.round_label == deformation].set_index('amplicon_id')
        for identity, row in reference.iterrows():
            q = row[['z', 'y', 'x']].to_numpy(float)
            voxel = tuple(np.clip(np.rint(q).astype(int), 0, np.array(field.shape[:3]) - 1))
            offset = np.abs(q - voxel).sum()
            expected = moved.loc[identity, ['z', 'y', 'x']].to_numpy(float) - q
            assert np.abs(field[voxel] - expected).max() <= .5 * offset + 1e-4, deformation
    shift = np.array(truth['pairs']['shift']['shift_zyx'])
    (root / 'correction.json').write_text(json.dumps((-shift).tolist()))
    case = BenchmarkCase('tiny-shift', 'registration', {'reference': 'ref.tif', 'moving': 'mov_shift.tif'},
        {'registration': {'method': 'translation'}, 'reference_metadata': {'frame_id': 'reference'},
         'moving_metadata': {'frame_id': 'moving'},
         'evaluation': {'ncc': True, 'translation': {'tolerance': 1.0}}},
        truth={'correction': 'correction.json'})
    run = run_benchmark([case], input_root=root, output_root=tmp_path / 'runs', owner='pytest')
    evaluation = evaluate_benchmark(run)
    record = json.loads((evaluation / 'results.json').read_text())[0]
    assert record['status']['processing'] == 'success'
    assert record['metrics']['translation']['values']['passed'] is True


def test_cli_rejects_unknown_preset_and_existing_output(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        main(['synthetic', 'generate', '--mode', 'e2e', '--preset', 'huge', '--seed', '1',
              '--owner', 'pytest', '--output', str(tmp_path / 'x')])
    assert error.value.code == 2 and 'unknown preset' in capsys.readouterr().err
    (tmp_path / 'used').mkdir()
    with pytest.raises(SystemExit):
        main(['synthetic', 'generate', '--mode', 'e2e', '--preset', 'tiny', '--seed', '1',
              '--owner', 'pytest', '--output', str(tmp_path / 'used')])


def test_generation_json_records_stream_counts_not_descriptors(tmp_path):
    for mode in ('e2e', 'registration'):
        root = generate(tmp_path, mode, 'tiny')
        path = next(root.rglob('generation.json'))
        for key, provenance in json.loads(path.read_text())['provenance'].items():
            scheme = provenance['stream_scheme']
            assert 'streams' not in scheme, (mode, key)
            assert scheme['stream_count'] > 0
            assert sum(scheme['streams_per_component'].values()) == scheme['stream_count']
            assert scheme['bit_generator'] == 'PCG64' and scheme['key']


def test_small_registration_cli(tmp_path):
    check_registration_layout(generate(tmp_path, 'registration', 'small'), 'small')


def timed_cli(tmp_path, mode, preset):
    """Run the CLI in a child under /usr/bin/time -v; return (output, seconds, max RSS KiB)."""
    output = tmp_path / f'{preset}-{mode}'
    completed = subprocess.run(
        ['/usr/bin/time', '-v', sys.executable, '-m', 'starfinder', 'synthetic', 'generate', '--mode', mode,
         '--preset', preset, '--seed', '42', '--owner', 'pytest', '--output', str(output)],
        capture_output=True, text=True, timeout=1200)
    assert completed.returncode == 0, completed.stderr[-2000:]
    clock = re.search(r'Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): ([\d:.]+)', completed.stderr).group(1)
    seconds = sum(float(part) * 60**i for i, part in enumerate(reversed(clock.split(':'))))
    rss = int(re.search(r'Maximum resident set size \(kbytes\): (\d+)', completed.stderr).group(1))
    return output, seconds, rss


@pytest.mark.extended
def test_medium_cli_both_modes_time_and_peak_rss(tmp_path, capsys):
    """The one medium generation check: each mode within about one minute on one thread.

    Both modes must also stay far below the 4 GiB RSS stop target. Each output
    is checked and deleted before the next mode runs; measurements for both
    modes are printed uncaptured, so they appear in the check log.
    """
    import shutil
    measured = {}
    for mode, check in (('e2e', check_e2e_layout), ('registration', check_registration_layout)):
        output, seconds, rss = timed_cli(tmp_path, mode, 'medium')
        check(output, 'medium')
        shutil.rmtree(output)
        measured[mode] = seconds, rss
    with capsys.disabled():
        for mode, (seconds, rss) in measured.items():
            print(f'\n  medium {mode}: wall {seconds:.1f} s, peak RSS {rss} KiB (target <= 60 s, < 4 GiB; '
                  f'estimate {SCENE_PRESETS["medium"]["peak_bytes_estimate"] // 1024} KiB working memory)')
    for mode, (seconds, rss) in measured.items():
        assert seconds <= 60, (mode, seconds)
        assert rss < 4 * 2**20, (mode, rss)
