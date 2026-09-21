"""C1/C2/C3/X1: literal W-154 signals; no random seeds or historical TIFFs."""
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from starfinder.barcode import decode_barcodes, filter_reads, WtaDecoderConfig
from starfinder.io import save_candidate_checkpoint, load_candidate_checkpoint
from starfinder.provenance import RunRecorder, read_run
from test.test_provenance import fixture

spec = importlib.util.spec_from_file_location('oracle', Path(__file__).resolve().parents[3]/'docs/examples/artifact_contracts.py')
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)


def save(path, spots, result, book=None, **kwargs):
    return save_candidate_checkpoint(path, spots, result, codebook=book,
        dataset_id='artifact-contract-v1', sample_id='sample', FOV='FOV', run_id='run',
        config={'literal': (1, 2)}, code={'commit': 'test-revision'}, **kwargs)


def rewrite(path, component, change):
    a = json.loads(path.read_text())
    d = next(d for d in a['components'] if d['component_id'] == component)
    target = path.parent/d['path']
    pq.write_table(change(pq.ParquetFile(target).read()), target, compression='zstd', row_group_size=65536)
    d.update(size=target.stat().st_size, sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    path.write_text(json.dumps(a))


@pytest.mark.parametrize('depth', [1, 3])
@pytest.mark.parametrize('variant', ['normal', 'empty', 'invalid', 'reordered'])
def test_roundtrip(tmp_path, depth, variant):
    spots, result, book, *_ = oracle.signal_example(depth)
    table = spots.spots.copy()
    table['optional_int'] = pd.Series([3, None], dtype='Int16')
    table['optional_bool'] = pd.Series([None, False], dtype='boolean')
    table['optional_text'] = pd.Series(['', None], dtype='string')
    table['narrow'] = np.array([1, 2], dtype=np.uint8)
    table['score'] = np.array([-0., np.nan], dtype=np.float32)
    spots = replace(spots, spots=table)
    result.values[1, 0, 0] = -0.
    if variant == 'empty':
        spots = replace(spots, spots=table.iloc[:0])
        result = replace(result, values=result.values[:0], valid=result.valid[:0], spot_ids=())
    if variant == 'invalid': result.valid[0, 1] = False
    if variant == 'reordered':
        result = replace(result, spot_ids=result.spot_ids[::-1], values=result.values[::-1], valid=result.valid[::-1])
    report = save(tmp_path/'saved', spots, result, book)
    assert report.size_bytes == sum(p.stat().st_size for p in report.path.parent.iterdir())
    for component in ('candidates', 'signals', 'validity', 'codebook'):
        rewrite(report.path, component, lambda a: a.take(np.arange(len(a), dtype=np.int64)[::-1]))
    loaded = load_candidate_checkpoint(report.path)
    pd.testing.assert_frame_equal(loaded.spots.spots, spots.spots.reset_index(drop=True), check_exact=True)
    assert loaded.spots.metadata == spots.metadata and loaded.spots.config == spots.config
    assert loaded.spots.diagnostics == spots.diagnostics
    order = [result.spot_ids.index(s) for s in spots.spots.spot_id]
    assert loaded.intensities.values.tobytes() == result.values[order].tobytes()
    np.testing.assert_array_equal(loaded.intensities.valid, result.valid[order], strict=True)
    assert loaded.intensities.config == result.config and loaded.intensities.diagnostics == result.diagnostics
    assert loaded.intensities.round_labels == ('round10', 'round2')
    assert loaded.codebook.color_to_channel == {'1': 1, '2': 0, '3': 3, '4': 2}
    before = filter_reads(decode_barcodes(result, book, config=WtaDecoderConfig()))
    after = filter_reads(decode_barcodes(loaded.intensities, loaded.codebook, config=WtaDecoderConfig()))
    pd.testing.assert_frame_equal(before.table.sort_values('spot_id').reset_index(drop=True),
        after.table.sort_values('spot_id').reset_index(drop=True), check_exact=True)
    assert before.counts == after.counts and before.fractions == after.fractions
    locator = dict(run_id='run', candidate_artifact_id=loaded.artifact['artifact_id'], spot_namespace=result.spot_namespace)
    if variant != 'empty':
        trace = loaded.source_trace(**locator, spot_id='A')
        np.testing.assert_array_equal(trace['values'], result.values[result.spot_ids.index('A')])
        assert trace['context']['config'] == {'literal': (1, 2)}
    with pytest.raises(ValueError, match='locator'):
        loaded.source_trace(**dict(locator, run_id='other'), spot_id='A')
    with pytest.raises(ValueError, match='unknown'):
        loaded.source_trace(**locator, spot_id='missing')


@pytest.mark.parametrize('component', ['candidates', 'signals', 'validity', 'codebook'])
@pytest.mark.parametrize('mutation', ['duplicate', 'missing'])
def test_missing_duplicate(tmp_path, component, mutation):
    spots, result, book, *_ = oracle.signal_example(1)
    path = save(tmp_path/'saved', spots, result, book).path
    rewrite(path, component, lambda a: pa.concat_tables([a, a.slice(0, 1)]) if mutation == 'duplicate' else a.slice(1))
    with pytest.raises(ValueError): load_candidate_checkpoint(path)


@pytest.mark.parametrize('field,value', [('spot_id', 'foreign'), ('spot_namespace', 'foreign'),
    ('channel_index', 4), ('round_index', -1), ('value', float('inf')), ('value', None)])
def test_signal_corruption(tmp_path, field, value):
    spots, result, book, *_ = oracle.signal_example(1)
    path = save(tmp_path/'saved', spots, result, book).path
    def change(a):
        values = a[field].to_pylist()
        values[0] = value
        return a.set_column(a.column_names.index(field), field, pa.array(values, type=a[field].type))
    rewrite(path, 'signals', change)
    with pytest.raises(ValueError): load_candidate_checkpoint(path)


@pytest.mark.parametrize('mutation', ['version', 'stage', 'axes', 'shape', 'checksum', 'missing_file', 'binding'])
def test_integrity(tmp_path, mutation):
    spots, result, book, *_ = oracle.signal_example(1)
    path = save(tmp_path/'saved', spots, result, book).path
    a = json.loads(path.read_text())
    if mutation == 'version': a['schema_version'] = 2
    if mutation == 'stage': a['stage'] = 'decoded_pre_qc'
    if mutation == 'axes': a['payload']['axes'] = 'NRC'
    if mutation == 'shape': a['payload']['shape'] = [3, 4, 2]
    if mutation == 'checksum': a['components'][0]['sha256'] = '0'*64
    if mutation == 'missing_file': (path.parent/a['components'][0]['path']).unlink()
    if mutation == 'binding': a['artifact_id'] = 'different'
    path.write_text(json.dumps(a))
    with pytest.raises(FileNotFoundError if mutation == 'missing_file' else ValueError): load_candidate_checkpoint(path)


def test_save_rejection_disable_no_codebook(tmp_path, monkeypatch):
    spots, result, book, *_ = oracle.signal_example(1)
    spots.spots.loc[1, 'spot_id'] = 'A'
    with pytest.raises(ValueError, match='unique'): save(tmp_path/'bad', spots, result, book)
    assert not (tmp_path/'bad').exists()
    spots, result, *_ = oracle.signal_example(1)
    saved = save(tmp_path/'saved', spots, result)
    assert load_candidate_checkpoint(saved.path).codebook is None
    with pytest.raises(FileExistsError): save(tmp_path/'saved', spots, result)
    with pytest.raises(ValueError, match='checksum'): load_candidate_checkpoint(saved.path, sha256='0'*64)
    import starfinder.io.candidates as module
    def missing(): raise ImportError('missing engine')
    monkeypatch.setattr(module, '_backend', missing)
    omitted = save(tmp_path/'disabled', spots, result, enabled=False)
    assert omitted.path is None and omitted.size_bytes == 0 and omitted.reason == 'candidates_signals_disabled'
    assert not (tmp_path/'disabled').exists()
    with pytest.raises(ImportError): save(tmp_path/'missing', spots, result)


@pytest.mark.parametrize('mode', ['batch', 'streaming'])
@pytest.mark.parametrize('enabled', [False, True])
def test_coordinated_save_before_qc(tmp_path, mode, enabled):
    from starfinder.dataset import ExecutionConfig
    fov, config = fixture(tmp_path)
    rec = RunRecorder(tmp_path/'run', dataset_id='literal', sample_id='sample',
                      **({} if enabled else {'save_candidates_signals': False}))
    fov.run(config, execution=ExecutionConfig(mode), provenance=rec)
    run = read_run(rec.path)
    assert run['saving_policy']['candidates_signals'] is enabled
    report = fov.candidate_checkpoint_save
    if enabled:
        saved = load_candidate_checkpoint(report.path)
        np.testing.assert_array_equal(saved.intensities.values, fov.intensity_result.values)
        assert report.size_bytes > 0
        names = [e['operation'] for e in run['events'] if e['outcome'] == 'succeeded']
        assert names.index('save_candidate_checkpoint') < names.index('decode_barcodes')
        event = next(e for e in run['events'] if e['operation'] == 'save_candidate_checkpoint' and e['outcome'] == 'succeeded')
        assert event['output_artifacts'] == [saved.artifact['artifact_id']]
        pd.testing.assert_frame_equal(filter_reads(decode_barcodes(saved.intensities, saved.codebook,
            config=config.decoding)).table, fov.filtering_result.table, check_exact=True)
        assert next(a for a in run['artifacts'] if a['stage'] == 'candidates_signals')['status'] == 'complete'
    else:
        assert report.path is None and report.reason == 'candidates_signals_disabled'
        assert not (rec.directory/'candidates-signals').exists()


def test_save_failure(tmp_path, monkeypatch):
    import starfinder.io.candidates as module
    fov, config = fixture(tmp_path)
    rec = RunRecorder(tmp_path/'run', dataset_id='literal', sample_id='sample')
    def fail(*args): raise OSError('simulated storage failure')
    monkeypatch.setattr(module, '_write_table', fail)
    with pytest.raises(OSError, match='storage failure'): fov.run(config, provenance=rec)
    assert not (rec.directory/'candidates-signals/artifact.json').exists()
    assert fov.decoding_result is None
    run = read_run(rec.path)
    assert run['status'] == 'failed' and run['failures']
    assert run['failures'][0]['category'] == 'serialization'


def test_geometry_and_input_identity_alignment(tmp_path):
    from starfinder.image import ImageMetadata
    spots, result, book, *_ = oracle.signal_example(3)
    geometry = ImageMetadata('reference', (2, 3, 4), (10, 20, 30),
                             ((1, 0, 0), (0, -1, 0), (0, 0, -1)), 'um')
    spots = replace(spots, metadata=geometry)
    result = replace(result, metadata=geometry)
    loaded = load_candidate_checkpoint(save(tmp_path/'geometry', spots, result, book).path)
    assert loaded.spots.metadata == geometry
    np.testing.assert_array_equal(loaded.spots.metadata.index_to_world((1, 2, 3)), (12, 14, 18))
    with pytest.raises(ValueError, match='identity'):
        save(tmp_path/'bad_ids', spots, replace(result, spot_ids=('A', 'foreign')), book)
    with pytest.raises(ValueError, match='namespace'):
        save(tmp_path/'bad_namespace', spots, replace(result, spot_namespace='foreign'), book)
