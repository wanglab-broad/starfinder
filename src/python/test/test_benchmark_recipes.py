"""Migrated configuration, pipeline parity and saved-only reporting contracts."""
from dataclasses import replace
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile

from starfinder.benchmark import run_benchmark, evaluate_benchmark, report_benchmark
from starfinder.benchmark._adapters import _validate

ROOT = Path(__file__).resolve().parents[3]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'benchmarks' / (name + '.py'))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


recipes = module('recipes')
read = lambda path: json.loads(path.read_text())


def test_profiles_and_complete_source_map():
    sources = read(ROOT / 'benchmarks/configs/source-map.json')
    assert len(sources['files']) == 56
    assert len({r['source'] for r in sources['files']}) == 56
    references = [r for r in sources['files'] if r['disposition'] == 'reference-only']
    assert len(references) == 8 and all(r['maintained_target'] is None for r in references)
    for row in sources['files']:
        assert len(row['source_sha256']) == 64
        if row['maintained_target']:
            assert (ROOT / row['maintained_target'].split('#')[0]).is_file()
    for name in read(ROOT / 'benchmarks/configs/registration.json')['profiles']:
        case = recipes.registration_case(name, case_id='fixture', reference='r.npy', moving='m.npy',
            evaluation={'ncc': True}, reference_metadata={'frame_id': 'r'}, moving_metadata={'frame_id': 'm'})
        _validate(case)
    assert recipes.profile('registration', 'tps-small')['registration']['min_matches'] == 5
    assert recipes.profile('registration', 'cpd-small')['registration']['grid_spacing_voxels'] == 8
    for name, profile in read(ROOT / 'benchmarks/configs/pipeline.json')['profiles'].items():
        n = profile['parameters']['n_rounds'] or 4
        sources = {f'round{i}': {c: f'{i}/{c}.tif' for c in profile['parameters']['channel_order']}
                   for i in range(1, n + 1)}
        _validate(recipes.pipeline_case(name, case_id='fixture', fov_id='FOV_001',
            sources=sources, codebook='codebook.csv', n_rounds=n))


def fixture_case(tmp_path, profile='large-batch'):
    inputs = tmp_path / 'inputs'
    inputs.mkdir(exist_ok=True)
    sources = {}
    for i in range(1, 5):
        sources[f'round{i}'] = {}
        for j in range(4):
            a = np.zeros((8, 16, 16), dtype=np.uint8)
            a[3, 7, 8] = 200 if j == 0 else 20
            a[5, 10, 11] = 150 if j == 1 else 10
            path = f'r{i}-ch{j:02}.tif'
            tifffile.imwrite(inputs / path, a, metadata={'axes': 'ZYX'}, photometric='minisblack')
            sources[f'round{i}'][f'ch{j:02}'] = path
    (inputs / 'codebook.csv').write_text('gene,barcode\ngeneA,CCCCC\ngeneB,CACAC\n')
    return recipes.pipeline_case(profile, case_id='fixture', fov_id='FOV_001',
        sources=sources, codebook='codebook.csv', n_rounds=4)


def test_pipeline_shared_lifecycle_and_saved_only(tmp_path, monkeypatch):
    case = fixture_case(tmp_path)
    def run(c, **kw):
        return run_benchmark([c], input_root=tmp_path/'inputs', output_root=tmp_path/'outputs', owner='test', **kw)
    batch = run(case)
    stream_case = fixture_case(tmp_path, 'large-streaming')
    stream = run(stream_case, run_id='stream')
    for root in (batch, stream):
        trial = read(root / 'fixture/0000/trial.json')
        assert trial['status']['processing'] == 'success', trial['errors']
        assert len(trial['attempts']) == 3
        assert trial['effective_configs']['pipeline']['normalization']['output_dtype'] == 'uint8'
        transform = read(root / 'fixture/0000/transform-round2-0.json')
        assert transform['correction_zyx'] == [0.0, 0.0, 0.0]
    pd.testing.assert_frame_equal(pd.read_csv(batch/'fixture/0000/reads.csv'),
                                  pd.read_csv(stream/'fixture/0000/reads.csv'))
    assert read(stream/'fixture/0000/pipeline.json')['retained_rounds'] == ['round1']
    assert len(read(batch/'fixture/0000/pipeline.json')['retained_rounds']) == 4
    from starfinder.dataset import FOV
    monkeypatch.setattr(FOV, 'run', lambda *a, **kw: pytest.fail('saved path ran pipeline'))
    assert run(stream_case, run_id='stream', resume=True) == stream
    import shutil
    shutil.rmtree(tmp_path/'inputs')
    evaluation = evaluate_benchmark(stream)
    result = read(evaluation/'results.json')[0]
    assert result['status']['evaluation'] == 'success', result['errors']
    assert result['metrics']['counts']['values']['detected'] > 0
    report = report_benchmark(evaluation)
    assert 'counts.detected' in (report/'summary.csv').read_text()
    (stream/'fixture/0000/reads.csv').unlink()
    with pytest.raises(FileNotFoundError):
        evaluate_benchmark(stream)


def test_pipeline_rejects_missing_sources_and_unsafe_paths(tmp_path):
    case = fixture_case(tmp_path)
    config = {**case.config, 'fov_id': '../escape'}
    with pytest.raises(ValueError, match='safe'):
        _validate(replace(case, config=config))
    with pytest.raises(ValueError, match='distinct'):
        _validate(replace(case, inputs={}))


def test_pipeline_records_explicit_recovery_and_failures(tmp_path):
    case = fixture_case(tmp_path)
    params = case.config['workflow']['rules']['rsf_single_fov']['parameters']
    params['global_registration'] = {'run': True, 'method': 'tps', 'recovery': {
        'allowed_errors': ['InsufficientLandmarksError'],
        'alternatives': [{'method': 'translation'}]}}
    def run():
        return run_benchmark([case], input_root=tmp_path/'inputs', output_root=tmp_path/'outputs', owner='test')
    root = run()
    trial = read(root/'fixture/0000/trial.json')
    assert trial['status']['processing'] == 'fallback_success', trial['errors']
    assert len(trial['attempts']) == 6
    assert {a['actual_method'] for a in trial['attempts']} == {'tps', 'translation'}
    assert trial['effective_configs']['pipeline']['registration'][0]['recovery']['allowed_errors'] == ['InsufficientLandmarksError']
    del params['global_registration']['recovery']
    root = run()
    trial = read(root/'fixture/0000/trial.json')
    assert trial['status']['processing'] == 'failed'
    assert trial['errors']['processing']['type'] == 'InsufficientLandmarksError'
    assert len(trial['attempts']) == 1
    assert read(evaluate_benchmark(root)/'results.json')[0]['status']['evaluation'] == 'skipped'


def test_historical_report_preserves_scopes_missing_and_failures(tmp_path):
    report = module('report_saved')
    quality, timing = tmp_path/'q.csv', tmp_path/'t.csv'
    quality.write_text('dataset,pair,backend,ncc,status\nfixture,shift,python,1,success\nother,real,matlab,,failed\n')
    timing.write_text('dataset,pair,backend,wall_seconds,peak_rss_kib\nfixture,shift,python,2,100\nthird,shift,python,3,200\n')
    kwargs = dict(output_dir=tmp_path/'report', keys=['dataset','pair','backend'], variant='local-v2',
        timing_path=timing, timing_scope='GNU time whole subprocess, seconds',
        memory_scope='GNU time max RSS, KiB; not tracemalloc')
    output = report.report_saved(quality, **kwargs)
    table = pd.read_csv(output/'comparison.csv')
    assert len(table) == 3
    assert pd.isna(table.loc[table.dataset=='other','wall_seconds']).all()
    assert table.loc[table.dataset=='other','status'].item() == 'failed'
    assert set(table._merge) == {'both','left_only','right_only'}
    assert read(output/'manifest.json')['sources'][0]['sha256']
    with pytest.raises(FileExistsError):
        report.report_saved(quality, **kwargs)
    timing.write_text(timing.read_text()+'fixture,shift,python,9,999\n')
    with pytest.raises(ValueError, match='unique'):
        report.report_saved(quality, **{**kwargs,'output_dir':tmp_path/'invalid'})
