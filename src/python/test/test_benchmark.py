"""Bounded persisted lifecycle, failure, resume, and CLI contracts."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tracemalloc

import numpy as np
import pytest

from starfinder.benchmark import (BenchmarkCase, BenchmarkTrialResult, run_benchmark,
                                 evaluate_benchmark, report_benchmark)


def read(path):
    return json.loads(path.read_text())


@pytest.fixture
def case(tmp_path):
    inputs = tmp_path / 'inputs'
    inputs.mkdir()
    reference = np.zeros((8, 16, 16), dtype=np.float32)
    reference[3, 7, 8] = 10
    np.save(inputs / 'reference.npy', reference)
    np.save(inputs / 'moving.npy', np.roll(reference, (1, -2, 1), (0, 1, 2)))
    (inputs / 'correction.json').write_text('[ -1.0, 2.0, -1.0 ]')
    return BenchmarkCase('tiny', 'registration', {'reference': 'reference.npy', 'moving': 'moving.npy'},
        {'registration': {'method': 'translation'},
         'reference_metadata': {'frame_id': 'reference'}, 'moving_metadata': {'frame_id': 'moving'},
         'evaluation': {'ncc': True, 'ssim': {'data_range': 10, 'policy': 'mip'},
                        'translation': {'tolerance': 0.01}}}, truth={'correction': 'correction.json'})


def run(case, tmp_path, **kwargs):
    return run_benchmark([case], input_root=tmp_path / 'inputs', output_root=tmp_path / 'outputs', owner='test-owner', **kwargs)


def trial(root):
    return read(root / 'tiny/0000/trial.json')


def test_saved_lifecycle_and_roundtrip(case, tmp_path, monkeypatch):
    root = run(case, tmp_path)
    original = trial(root)
    assert original['status'] == {'processing': 'success', 'evaluation': 'not_run'}
    assert original == BenchmarkTrialResult.from_dict(original).to_dict()
    assert case == BenchmarkCase.from_dict(case.to_dict())
    transform = read(root / 'tiny/0000/transform.json')
    assert transform['correction_zyx'] == [-1.0, 2.0, -1.0]
    assert transform['reference_metadata']['spacing_zyx'] is None
    for key in ('wall', 'cpu', 'python_peak_allocation'):
        assert original['resources'][key]['value'] >= 0
        assert set(original['resources'][key]) == {'value', 'unit', 'source', 'scope'}
    assert original['resources']['process_peak_rss']['value'] is None
    def forbidden(*args, **kwargs):
        raise AssertionError('saved evaluation/report invoked processing')
    import starfinder.registration as registration
    import starfinder.benchmark._lifecycle as lifecycle
    monkeypatch.setattr(registration, 'estimate_transform', forbidden)
    monkeypatch.setattr(registration, 'apply_transform', forbidden)
    monkeypatch.setattr(lifecycle, '_process', forbidden)
    # Original inputs can disappear: saved evaluation is self-contained.
    import shutil
    shutil.rmtree(tmp_path / 'inputs')
    evaluation = evaluate_benchmark(root)
    results = read(evaluation / 'results.json')
    assert results[0]['metrics']['ncc_registered']['values']['ncc'] == pytest.approx(1)
    assert results[0]['metrics']['translation']['values']['passed'] is True
    assert trial(root) == original
    monkeypatch.setattr(lifecycle, '_evaluate', forbidden)
    report = report_benchmark(evaluation)
    assert read(report / 'trials.json') == results
    assert 'translation.mean_error_l2' in (report / 'summary.csv').read_text()
    assert report_benchmark(evaluation) != report


def test_resume_and_integrity(case, tmp_path, monkeypatch):
    root = run(case, tmp_path, run_id='fixed')
    before = (root / 'tiny/0000/trial.json').read_bytes()
    import starfinder.benchmark._lifecycle as lifecycle
    monkeypatch.setattr(lifecycle, '_process', lambda *a: pytest.fail('resume reran processing'))
    assert run(case, tmp_path, run_id='fixed', resume=True) == root
    assert (root / 'tiny/0000/trial.json').read_bytes() == before
    with pytest.raises(FileExistsError):
        run(case, tmp_path, run_id='fixed')
    with pytest.raises(ValueError, match='identity'):
        run(replace(case, config={**case.config, 'registration': {'method': 'translation', 'backend': 'skimage'}}), tmp_path, run_id='fixed', resume=True)
    with pytest.raises(ValueError, match='run_id'):
        run(case, tmp_path, resume=True)
    manifest_path = root / 'manifest.json'
    manifest = read(manifest_path)
    manifest_path.write_text(json.dumps({**manifest, 'schema_version': 99}))
    with pytest.raises(ValueError, match='schema'):
        run(case, tmp_path, run_id='fixed', resume=True)
    manifest_path.write_text(json.dumps(manifest))
    path = root / 'tiny/0000/registered.npy'
    content = path.read_bytes()
    path.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='checksum'):
        evaluate_benchmark(root)
    path.write_bytes(content)
    path.unlink()
    with pytest.raises(FileNotFoundError, match='required artifact missing'):
        evaluate_benchmark(root)


def test_failures_fallback_and_undefined(case, tmp_path):
    np.save(tmp_path / 'inputs/reference.npy', np.zeros((8, 16, 16), dtype=np.float32))
    np.save(tmp_path / 'inputs/moving.npy', np.zeros((8, 16, 16), dtype=np.float32))
    config = {**case.config, 'registration': {'method': 'tps'}, 'evaluation': {'ncc': True}}
    failed = run(replace(case, config=config), tmp_path)
    assert trial(failed)['errors']['processing']['type'] == 'InsufficientLandmarksError'
    assert read(evaluate_benchmark(failed) / 'results.json')[0]['status']['evaluation'] == 'skipped'
    config['fallback'] = {'on_errors': ['InsufficientLandmarksError'], 'configs': [{'method': 'translation'}]}
    recovered = run(replace(case, config=config), tmp_path)
    record = trial(recovered)
    assert record['requested_method'] == 'tps' and record['actual_method'] == 'translation'
    assert record['status']['processing'] == 'fallback_success'
    assert [a['status'] for a in record['attempts']] == ['failed', 'success']
    evaluated = read(evaluate_benchmark(recovered) / 'results.json')[0]
    assert evaluated['status']['evaluation'] == 'undefined'
    assert evaluated['metrics']['ncc_registered']['values']['ncc'] is None
    assert evaluated['metrics']['ncc_registered']['reasons']['ncc']
    assert not tracemalloc.is_tracing()
    # Invalid image errors do not trigger even explicitly configured fallback.
    np.save(tmp_path / 'inputs/moving.npy', np.full((8, 16, 16), np.nan))
    invalid = trial(run(replace(case, config=config), tmp_path))
    assert invalid['status']['processing'] == 'failed' and len(invalid['attempts']) == 1


def test_evaluation_failure_is_distinct(case, tmp_path):
    (tmp_path / 'inputs/correction.json').write_text('[1, 2]')
    root = run(case, tmp_path)
    result = read(evaluate_benchmark(root) / 'results.json')[0]
    assert result['status'] == {'processing': 'success', 'evaluation': 'failed'}
    assert result['errors']['evaluation']['type'] == 'ValueError'
    assert 'processing' not in result['errors']


def test_float_transform_and_dense_artifacts(case, tmp_path, monkeypatch):
    from starfinder.registration import (RegistrationResult, RegistrationDiagnostics,
        TranslationTransform, TranslationConfig, DenseDisplacementTransform, WarpConfig)
    from starfinder.image import ImageMetadata
    import starfinder.registration as registration
    metadata = ImageMetadata('reference', spacing_zyx=(2, 1, 1))
    config = TranslationConfig()
    transform = TranslationTransform((0.125, -0.75, 1.5), (8, 16, 16), (8, 16, 16), metadata, metadata)
    monkeypatch.setattr(registration, 'estimate_transform', lambda *a, **k:
        RegistrationResult(transform, RegistrationDiagnostics('translation', 'fixture', config), WarpConfig(output_dtype='float32')))
    root = run(case, tmp_path)
    saved = read(root / 'tiny/0000/transform.json')
    assert saved['correction_zyx'] == [0.125, -0.75, 1.5]
    assert saved['reference_metadata']['spacing_zyx'] == [2, 1, 1]
    transform = DenseDisplacementTransform(np.full((8, 16, 16, 3), 0.25, dtype=np.float32),
        (8, 16, 16), (8, 16, 16), metadata, metadata)
    monkeypatch.setattr(registration, 'estimate_transform', lambda *a, **k:
        RegistrationResult(transform, RegistrationDiagnostics('translation', 'fixture', config), WarpConfig(backend='scipy')))
    root = run(case, tmp_path)
    assert trial(root)['status']['processing'] == 'success'
    np.testing.assert_array_equal(np.load(root / 'tiny/0000/field.npy'), transform.displacement_zyx)


def test_cli_lifecycle_and_help(case, tmp_path):
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'schema_version': 1, 'cases': [case.to_dict()], 'repetitions': 1}))
    def cli(*args, code=0):
        result = subprocess.run([sys.executable, '-m', 'starfinder', *map(str, args)], capture_output=True, text=True)
        assert result.returncode == code, result.stderr
        return result.stdout.strip()
    root = Path(cli('benchmark', 'run', '--config', config, '--input-root', tmp_path / 'inputs',
        '--output-root', tmp_path / 'outputs', '--owner', 'test'))
    evaluation = Path(cli('benchmark', 'evaluate', '--run-dir', root))
    report = Path(cli('benchmark', 'report', '--evaluation-dir', evaluation))
    assert (report / 'summary.csv').is_file()
    for command in [('synthetic', 'generate'), ('benchmark', 'run'), ('benchmark', 'evaluate'), ('benchmark', 'report')]:
        assert 'usage:' in cli(*command, '--help')
    cli('benchmark', 'run', code=2)
    cli('benchmark', 'evaluate', '--run-dir', tmp_path / 'absent', code=2)
    # Exercise the persisted processing-failure exit status.
    np.save(tmp_path / 'inputs/moving.npy', np.full((8, 16, 16), np.nan))
    root = Path(cli('benchmark', 'run', '--config', config, '--input-root', tmp_path / 'inputs',
        '--output-root', tmp_path / 'outputs', '--owner', 'test', code=1))
    cli('benchmark', 'evaluate', '--run-dir', root, code=1)


def test_removed_surface_and_invalid_configs(case, tmp_path):
    import starfinder.benchmark as benchmark
    for name in ('BenchmarkSuite', 'BenchmarkResult', 'RegistrationBenchmarkRunner', 'RegistrationResult', 'benchmark', 'measure', 'run_comparison'):
        assert not hasattr(benchmark, name)
    for config in ({**case.config, 'typo': True}, {**case.config, 'registration': {'method': 'translation', 'typo': 1}}):
        with pytest.raises((ValueError, TypeError)):
            run(replace(case, config=config), tmp_path)
    with pytest.raises(ValueError, match='escapes'):
        run(replace(case, inputs={'reference': '../outside.npy', 'moving': 'moving.npy'}), tmp_path)


from starfinder.synthetic._presets import SIZE_PRESETS
from starfinder.synthetic import get_preset_config


class TestPresets:
    """Tests for benchmark presets."""

    def test_size_presets_exist(self):
        """SIZE_PRESETS contains expected presets."""
        assert "tiny" in SIZE_PRESETS
        assert "small" in SIZE_PRESETS
        assert "medium" in SIZE_PRESETS

    def test_removed_presets(self):
        """xlarge and thick_large are no longer in SIZE_PRESETS."""
        assert "xlarge" not in SIZE_PRESETS
        assert "thick_large" not in SIZE_PRESETS

    def test_get_size_preset(self):
        """get_size_preset() returns correct shape."""
        shape = get_preset_config("tiny").shape_zyx
        assert len(shape) == 3
        assert all(isinstance(s, int) for s in shape)

    def test_get_size_preset_invalid(self):
        """get_size_preset() raises for unknown preset."""
        import pytest
        with pytest.raises(ValueError, match="Unknown preset"):
            get_preset_config("nonexistent")


def test_demons_saved_field(case, tmp_path):
    # Installed SimpleITK is required by the validation environment.
    import SimpleITK
    assert SimpleITK.Version_VersionString()
    config = {**case.config, 'registration': {'method': 'demons', 'iterations': [1]},
              'evaluation': {'ncc': True}}
    root = run(replace(case, config=config), tmp_path)
    record = trial(root)
    assert record['status']['processing'] == 'success', record['errors']
    assert record['actual_method'] == 'demons'
    assert record['effective_configs']['application']['backend'] == 'simpleitk'
    field = np.load(root / 'tiny/0000/field.npy')
    assert field.shape == (8, 16, 16, 3) and np.isfinite(field).all()
    assert read(evaluate_benchmark(root) / 'results.json')[0]['status']['evaluation'] == 'success'


def test_interrupted_trial_preserved(case, tmp_path):
    root = run(case, tmp_path, run_id='interrupted')
    path = root / 'tiny/0000/trial.json'
    path.unlink()  # Simulate interruption before durable completion.
    contents = (root / 'tiny/0000/registered.npy').read_bytes()
    with pytest.raises(FileExistsError):
        run(case, tmp_path, run_id='interrupted', resume=True)
    assert (root / 'tiny/0000/registered.npy').read_bytes() == contents
