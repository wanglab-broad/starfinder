"""P1/X1: literal 1/3 x 7 x 9, four channels, two rounds; no randomness."""
from dataclasses import replace
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from starfinder.dataset import Dataset, RoundState, PipelineConfig, ExecutionConfig, RegistrationStep, RecoveryConfig
from starfinder.image import ImageMetadata
from starfinder.barcode import Codebook, NeighborhoodSumConfig, WtaDecoderConfig, ReadFilterConfig
from starfinder.spot_finding import LocalMaximaConfig
from starfinder.registration import TpsConfig, TranslationConfig, InsufficientLandmarksError, DemonsConfig
from starfinder.provenance import RunRecorder, read_run


def fixture(root, z=3, empty=False):
    ds = Dataset(root, root/'out', 'literal', 'sample', 'output',
        RoundState(['round10', 'round2'], reference_round='round10'), ('b', 'a', 'd', 'c'))
    ds.codebook = Codebook(pd.DataFrame({'gene_id': ['gene'], 'color_sequence': ['11']}),
        tuple(ds.rounds.sequencing_rounds), ds.channel_order)
    fov = ds.fov('FOV')
    a = np.zeros((z, 7, 9, 4), dtype=np.uint16)
    if not empty:
        a[z//2, 3, 4, 0] = 7
    fov.images = {'round10': a, 'round2': a.copy()}
    fov.metadata = {name: ImageMetadata('common') for name in fov.images}
    config = PipelineConfig(detection=LocalMaximaConfig('adaptive', .1),
        extraction=NeighborhoodSumConfig((0, 0, 0)), decoding=WtaDecoderConfig(diagnostics=True),
        filtering=ReadFilterConfig())
    return fov, config


def recorder(root):
    return RunRecorder(root, dataset_id='literal', sample_id='sample', owner='test', retention='test fixture',
                       save_candidates_signals=False)


def state(run):
    return run['extensions']['starfinder.provenance']['final_state']


@pytest.mark.parametrize('z', [1, 3])
@pytest.mark.parametrize('empty', [False, True])
def test_success_empty_and_residency(tmp_path, z, empty):
    runs, results = [], []
    for mode in ('batch', 'streaming'):
        fov, config = fixture(tmp_path, z, empty)
        rec = recorder(tmp_path/mode)
        assert fov.run(config, execution=ExecutionConfig(mode), provenance=rec) is fov
        run = read_run(rec.path, sha256=hashlib.sha256(rec.path.read_bytes()).hexdigest())
        assert run['status'] == 'succeeded' and run['failures'] == []
        assert run['dataset_id'] == 'literal' and run['sample_id'] == 'sample'
        assert state(run)['detected'] == (0 if empty else 1)
        assert state(run)['geometry']['round10']['fields']['spacing_zyx'] is None
        assert run['code']['commit'] is None and run['code']['unknown_reason']
        assert run['environment']['seed']['value'] is None
        assert state(run)['detection_config']['fields']['channel_labels'] == ['b', 'a', 'd', 'c']
        for artifact in run['artifacts']:
            if artifact['stage'] in ('decoded_pre_qc', 'final_accepted'):
                assert artifact['status'] == 'complete' and artifact['components']
                assert artifact['omission_reason'] is None
            else:
                assert artifact['status'] == 'omitted' and not artifact['components']
                assert artifact['omission_reason']
        assert [e['sequence'] for e in run['events']] == list(range(len(run['events'])))
        expected = [('find_spots', 'round10'), ('_extract_round', 'round10'),
                    ('_extract_round', 'round2'), ('_assemble_intensities', None),
                    ('decode_barcodes', None), ('save_decoded_checkpoint', None),
                    ('filter_reads', None), ('save_final_checkpoint', None)]
        assert [(e['operation'], e['round']) for e in run['events'] if e['outcome'] == 'succeeded'] == expected
        if not empty:
            np.testing.assert_array_equal(fov.intensity_result.values, [[[7, 7], [0, 0], [0, 0], [0, 0]]])
            assert fov.filtering_result.accepted['gene_id'].tolist() == ['gene']
        else:
            assert fov.intensity_result.values.shape == (0, 4, 2)
        with pytest.raises(ValueError, match='single-use'):
            fov.run(config, provenance=rec)
        with pytest.raises(FileExistsError):
            recorder(tmp_path/mode)
        runs.append(run)
        results.append(fov)
    pd.testing.assert_frame_equal(results[0].filtering_result.table, results[1].filtering_result.table)
    assert state(runs[0])['counts'] == state(runs[1])['counts']


def test_failed_estimation_and_explicit_recovery(tmp_path):
    for recover in (False, True):
        fov, config = fixture(tmp_path)
        rec = recorder(tmp_path/str(recover))
        step = RegistrationStep(TpsConfig(), recovery=RecoveryConfig(
            (InsufficientLandmarksError,), (TranslationConfig(),)) if recover else None)
        config = replace(config, registration=(step,))
        if recover:
            fov.run(config, provenance=rec)
        else:
            with pytest.raises(InsufficientLandmarksError):
                fov.run(config, provenance=rec)
        run = read_run(rec.path)
        assert run['status'] == ('succeeded' if recover else 'failed')
        failures = [f for f in run['failures'] if f['type'] == 'InsufficientLandmarksError']
        assert failures and all(f['category'] == 'estimation' for f in failures)
        attempts = state(run)['registration_attempts']['round2']
        assert attempts[0]['requested_method'] == attempts[0]['actual_method'] == 'tps'
        assert attempts[0]['outcome'] == 'failed'
        if recover:
            assert failures[0]['recovery_event_id']
            assert attempts[-1]['actual_method'] == 'translation'
            result = state(run)['registration_results']['round2'][0]['fields']
            assert result['transform']['fields']['correction_zyx'] == [0., 0., 0.]
        else:
            assert state(run)['partial_round_signals'] == ['round10']
            assert run['extensions']['starfinder.provenance']['stage_state']['candidates_signals'] == 'partial'
            candidates = next(a for a in run['artifacts'] if a['stage'] == 'candidates_signals')
            assert candidates['status'] == 'omitted' and candidates['omission_reason'] == 'incomplete_stage'
            assert any(a['status'] == 'failed' for a in run['artifacts'])
            assert not any(e['operation'] == 'decode_barcodes' for e in run['events'])


@pytest.mark.parametrize('error', [ValueError('invalid input'), RuntimeError('application failed'), KeyboardInterrupt()])
def test_failures_and_interruption(tmp_path, monkeypatch, error):
    import starfinder.registration as registration
    fov, config = fixture(tmp_path)
    config = replace(config, registration=(RegistrationStep(TranslationConfig()),))
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(registration, 'apply_transform', fail)
    rec = recorder(tmp_path/'run')
    with pytest.raises(type(error)):
        fov.run(config, provenance=rec)
    run = read_run(rec.path)
    assert run['status'] == ('interrupted' if isinstance(error, KeyboardInterrupt) else 'failed')
    assert any(f['type'] == type(error).__name__ for f in run['failures'])
    assert not any(a['status'] == 'complete' for a in run['artifacts'])
    assert fov._provenance is None


def test_dense_transform_component_and_integrity(tmp_path):
    fov, config = fixture(tmp_path, z=4)
    rec = recorder(tmp_path/'run')
    config = replace(config, registration=(RegistrationStep(DemonsConfig(iterations=(1,))),))
    fov.run(config, provenance=rec)
    run = read_run(rec.path)
    transform = state(run)['registration_results']['round2'][0]['fields']['transform']['fields']
    original = fov.registration_results['round2'][0].transform.displacement_zyx
    assert transform['displacement_zyx'].dtype == original.dtype
    assert transform['displacement_zyx'].tobytes() == original.tobytes()
    component = next((tmp_path/'run').glob('*.npy'))
    saved = component.read_bytes()
    component.write_bytes(saved[:-1])
    with pytest.raises(ValueError, match='integrity'):
        read_run(rec.path)
    component.unlink()
    with pytest.raises(FileNotFoundError, match='component'):
        read_run(rec.path)


@pytest.mark.parametrize('mutate', [
    lambda r: r.update(schema_version=2),
    lambda r: r.pop('environment'),
    lambda r: r.update(status='complete'),
    lambda r: r['events'][0].update(stage='typo'),
    lambda r: r['events'][0].update(sequence=True),
    lambda r: r['events'][0].update(config_ref='missing'),
    lambda r: r['events'][1].update(event_id=r['events'][0]['event_id']),
    lambda r: next(a for a in r['artifacts'] if a['stage'] == 'candidates_signals').update(status='complete'),
    lambda r: r['artifacts'][0].update(source_refs=['missing']),
    lambda r: r['artifacts'][0].update(sample_id='foreign'),
])
def test_malformed_records(tmp_path, mutate):
    fov, config = fixture(tmp_path)
    rec = recorder(tmp_path/'run')
    fov.run(config, provenance=rec)
    run = json.loads(rec.path.read_text())
    mutate(run)
    rec.path.write_text(json.dumps(run))
    with pytest.raises(ValueError, match='provenance'):
        read_run(rec.path)


def test_persistence_error_visible_and_running_record_not_success(tmp_path, monkeypatch):
    fov, config = fixture(tmp_path)
    rec = recorder(tmp_path/'run')
    original = rec._publish
    calls = 0
    def broken():
        nonlocal calls
        calls += 1
        if calls > 2:
            raise OSError('intentional disk failure')
        original()
    monkeypatch.setattr(rec, '_publish', broken)
    with pytest.raises(OSError, match='intentional disk failure'):
        fov.run(config, provenance=rec)
    run = read_run(rec.path)
    assert run['status'] == 'running'
    assert run['events'][-1]['outcome'] == 'started'
    assert not any(a['status'] == 'complete' for a in run['artifacts'])


def test_invalid_config_serialization_visible(tmp_path):
    fov, config = fixture(tmp_path)
    rec = recorder(tmp_path/'run')
    with pytest.raises(TypeError, match='unsupported provenance'):
        fov.run(object(), provenance=rec)
    run = read_run(rec.path)
    assert run['status'] == 'failed'
    assert run['failures'][0]['category'] == 'serialization'


def test_checksum_and_secret_boundary(tmp_path, monkeypatch):
    monkeypatch.setenv('SECRET_TEST_TOKEN', 'must-not-be-recorded')
    fov, config = fixture(tmp_path)
    rec = recorder(tmp_path/'run')
    fov.run(config, provenance=rec)
    assert 'must-not-be-recorded' not in rec.path.read_text()
    with pytest.raises(ValueError, match='manifest integrity'):
        read_run(rec.path, sha256='0'*64)
    rec.path.write_text('{"schema_name":"starfinder.run","schema_name":"other"}')
    with pytest.raises(ValueError, match='duplicate JSON'):
        read_run(rec.path)


def test_known_geometry_tiff_sources_and_warnings(tmp_path):
    from starfinder.io import ImageLoadConfig, save_volume
    fov, config = fixture(tmp_path)
    geometry = ImageMetadata('physical', (2, 3, 4), (10, 20, 30),
                             ((1, 0, 0), (0, -1, 0), (0, 0, -1)), 'um')
    hashes = []
    for name, image in fov.images.items():
        folder = tmp_path/name/'FOV'
        folder.mkdir(parents=True)
        for index, label in enumerate(fov.dataset.channel_order):
            path = folder/f'{label}.tif'
            save_volume(image[..., index], path, metadata=geometry)
            hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
    fov.images.clear()
    fov.metadata.clear()
    rec = recorder(tmp_path/'record')
    fov.run(replace(config, load=ImageLoadConfig(channel_labels=fov.dataset.channel_order)), provenance=rec)
    run = read_run(rec.path)
    assert [s['sha256'] for s in run['sources']] == hashes
    assert all(s['selection']['original_geometry']['spacing_zyx'] == [2, 3, 4] for s in run['sources'])
    assert state(run)['geometry']['round2']['fields']['origin_zyx'] == [10, 20, 30]
    original = json.loads(rec.path.read_text())
    original['extensions']['starfinder.provenance']['final_state']['geometry']['round2']['fields']['spacing_zyx'] = [0, 3, 4]
    rec.path.write_text(json.dumps(original))
    with pytest.raises(ValueError, match='geometry'):
        read_run(rec.path)


def test_warning_visibility_and_special_diagnostics(tmp_path, monkeypatch):
    import warnings
    import starfinder.spot_finding as spot_finding
    original = spot_finding.find_spots
    def warning(*args, **kwargs):
        warnings.warn('intentional detector diagnostic', UserWarning)
        result = original(*args, **kwargs)
        result.diagnostics['special'] = [float('inf'), -float('inf'), float('nan'), None]
        return result
    monkeypatch.setattr(spot_finding, 'find_spots', warning)
    fov, config = fixture(tmp_path)
    rec = recorder(tmp_path/'run')
    with pytest.warns(UserWarning, match='intentional detector diagnostic'):
        fov.run(config, provenance=rec)
    run = read_run(rec.path)
    event = next(e for e in run['events'] if e['operation'] == 'find_spots' and e['outcome'] == 'succeeded')
    assert event['diagnostics']['warnings'] == [{'type': 'UserWarning', 'message': 'intentional detector diagnostic'}]
    values = state(run)['detection_diagnostics']['special']
    assert values[:2] == [float('inf'), -float('inf')]
    assert np.isnan(values[2]) and values[3] is None
    assert '"float_special"' in rec.path.read_text()
    # Recording must not turn warnings configured as errors into success.
    fov, config = fixture(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        with pytest.raises(UserWarning):
            fov.run(config, provenance=recorder(tmp_path/'error'))
    assert read_run(tmp_path/'error')['status'] == 'failed'


def test_complete_artifact_link_and_immutability(tmp_path):
    # Metadata integration contract for future format-specific checkpoint writers.
    rec = recorder(tmp_path/'run')
    path = rec.directory/'empty.npy'
    np.save(path, np.empty((0,), dtype=np.float64), allow_pickle=False)
    descriptor = dict(component_id='empty', path='empty.npy', format='npy',
        size=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), shape=[0], dtype='<f8')
    artifact = dict(schema_name='starfinder.artifact', schema_version=1, contract_id='starfinder.artifacts/1',
        artifact_id='writer-validated-example', run_id=rec.run_id, stage='prepared_input', status='complete',
        dataset_id='literal', sample_id='sample', FOV='FOV', subtile=None, parents=[],
        config_ref='config/effective', source_refs=[], components=[descriptor],
        payload={'example_only': True}, omission_reason=None, failure_id=None)
    rec.record_artifact(artifact)
    with pytest.raises(ValueError, match='duplicate'):
        rec.record_artifact(artifact)
    fov, config = fixture(tmp_path)
    fov.run(config, provenance=rec)
    assert read_run(rec.path)['artifacts'][0] == artifact
    with pytest.raises(ValueError, match='terminal'):
        rec.record_artifact(dict(artifact, artifact_id='other'))
    original = json.loads(rec.path.read_text())
    for bad_path in ('../outside.npy', '/absolute.npy'):
        malformed = json.loads(json.dumps(original))
        malformed['artifacts'][0]['components'][0]['path'] = bad_path
        rec.path.write_text(json.dumps(malformed))
        with pytest.raises(ValueError, match='escapes'):
            read_run(rec.path)
    rec.path.write_text(json.dumps(original))
    (rec.directory/'escape.npy').symlink_to(tmp_path/'outside.npy')
    original['artifacts'][0]['components'][0]['path'] = 'escape.npy'
    rec.path.write_text(json.dumps(original))
    with pytest.raises(ValueError, match='escapes'):
        read_run(rec.path)


def test_missing_input_keeps_failure_attribution(tmp_path):
    from starfinder.io import ImageLoadConfig
    fov, config = fixture(tmp_path)
    fov.images.clear()
    rec = recorder(tmp_path/'run')
    with pytest.raises(FileNotFoundError):
        fov.run(replace(config, load=ImageLoadConfig(channel_labels=fov.dataset.channel_order)), provenance=rec)
    run = read_run(rec.path)
    assert run['status'] == 'failed'
    assert run['failures'][0]['category'] == 'missing_input'
    assert run['failures'][0]['FOV'] == 'FOV'
    assert run['failures'][0]['type'] == 'FileNotFoundError'


def test_invalid_pipeline_type_is_not_a_serialization_failure(tmp_path):
    fov, _ = fixture(tmp_path)
    rec = recorder(tmp_path/'run')
    with pytest.raises(TypeError, match='PipelineConfig'):
        fov.run('invalid pipeline', provenance=rec)
    run = read_run(rec.path)
    assert run['status'] == 'failed'
    assert run['failures'][0]['category'] == 'invalid_input'
