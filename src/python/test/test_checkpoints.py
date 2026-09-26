"""Checkpoint round trips and run records on hand-built 4-channel, 2-round cases."""
from dataclasses import replace
import json
import shutil
import sys
from time import perf_counter

import numpy as np
import pandas as pd
import pytest

from starfinder.barcode import (Codebook, CodebookAwareDecoderConfig, NeighborhoodSumConfig,
    ReadFilterConfig, WtaDecoderConfig)
from starfinder.dataset import (CheckpointConfig, Dataset, ExecutionConfig, PipelineConfig,
    RegistrationStep, RoundState)
from starfinder.image import ImageMetadata
from starfinder.io import ImageLoadConfig, load_volume_zyxc, read_checkpoint, save_volume
from starfinder.io._checkpoint import _read_table, _write_table, candidates_frame, parse_candidates
from starfinder.registration import DemonsConfig, TranslationConfig
from starfinder.spot_finding import LocalMaximaConfig

ROUNDS = ('round1', 'round2')
CHANNELS = ('a', 'b', 'c', 'd')


def dataset(tmp_path, rounds=RoundState(list(ROUNDS), reference_round='round1'), channels=CHANNELS):
    ds = Dataset(tmp_path, tmp_path / 'out', 'data', 'sample', 'run',
                 RoundState(list(rounds.sequencing_rounds), reference_round=rounds.reference_round),
                 channels)
    ds.codebook = Codebook(pd.DataFrame({'gene_id': ['gene', 'other'], 'color_sequence': ['11', '22']}),
                           tuple(ds.rounds.sequencing_rounds), ds.channel_order)
    return ds


def images(z):
    """Assigned, unmatched and tied (ambiguous, infinite score) spots."""
    first = np.zeros((z, 12, 14, 4), dtype=np.uint16)
    zs = [0, 0, 0] if z == 1 else [1, 2, 1]
    first[zs[0], 4, 5, 0] = 60000
    first[zs[1], 8, 9, 0] = 30000
    first[zs[2], 3, 10, 1] = 20000
    first[zs[2], 3, 10, 2] = 20000
    second = first.copy()
    second[zs[1], 8, 9, 0] = 0
    second[zs[1], 8, 9, 3] = 30000
    return {'round1': first, 'round2': second}


def resident(ds, z=4):
    fov = ds.fov('FOV')
    fov.images = images(z)
    fov.metadata = {r: ImageMetadata('common', spacing_zyx=(2, .5, .5)) for r in ds.rounds.all_rounds}
    return fov


DETECT = dict(detection=LocalMaximaConfig('adaptive', .1), extraction=NeighborhoodSumConfig((0, 1, 1)))
DECODE = dict(decoding=WtaDecoderConfig(diagnostics=True), filtering=ReadFilterConfig())


def full(**kwargs):
    return PipelineConfig(registration=(RegistrationStep(TranslationConfig()),), **DETECT, **DECODE, **kwargs)


def assert_spots_equal(a, b):
    pd.testing.assert_frame_equal(a.spots, b.spots)
    assert (a.metadata, a.spot_namespace, a.config, a.diagnostics) == (b.metadata, b.spot_namespace, b.config, b.diagnostics)


def assert_intensities_equal(a, b):
    np.testing.assert_array_equal(a.values, b.values)
    np.testing.assert_array_equal(a.valid, b.valid)
    assert a.values.dtype == b.values.dtype and a.valid.dtype == b.valid.dtype
    for name in ('spot_ids', 'spot_namespace', 'channel_labels', 'round_labels', 'metadata', 'config', 'diagnostics'):
        assert getattr(a, name) == getattr(b, name), name


def assert_decoding_equal(a, b):
    pd.testing.assert_frame_equal(a.table, b.table)
    for name in ('spot_namespace', 'channel_labels', 'round_labels', 'config'):
        assert getattr(a, name) == getattr(b, name), name
    # Probability arrays and per-round/candidate tables are not saved in checkpoints.
    kept = [{k: v for k, v in r.diagnostics.items() if not isinstance(v, (np.ndarray, pd.DataFrame))} for r in (a, b)]
    assert kept[0] == kept[1]


def assert_downstream_equal(a, b):
    assert_decoding_equal(a.decoding_result, b.decoding_result)
    pd.testing.assert_frame_equal(a.filtering_result.table, b.filtering_result.table)
    assert (a.filtering_result.counts, a.filtering_result.fractions) == (b.filtering_result.counts, b.filtering_result.fractions)


def record(fov, directory=None):
    path = fov.paths.checkpoint_dir if directory is None else directory / fov.fov_id
    return json.loads((path / 'run.json').read_text())


@pytest.mark.parametrize('table_format', ['csv', 'parquet'])
@pytest.mark.parametrize('mode', ['batch', 'streaming'])
@pytest.mark.parametrize('z', [1, 4])
def test_each_stage_round_trips_and_continues(tmp_path, z, mode, table_format):
    ds = dataset(tmp_path)
    execution = ExecutionConfig(mode)
    reference = resident(ds, z).run(full())
    saved = resident(ds, z).run(full(), execution=execution,
                                checkpoints=CheckpointConfig(table_format=table_format))
    assert saved.decoding_result.table.call_status.tolist() == ['assigned', 'unmatched', 'ambiguous', 'ambiguous']
    assert np.isinf(saved.decoding_result.table.wta_l2_nll).sum() == 2
    assert saved.decoding_result.table.gene_id.isna().sum() == 3
    assert_downstream_equal(saved, reference)
    directory = saved.paths.checkpoint_dir
    assert (directory / f'candidates.{table_format}').is_file() and (directory / f'pre_qc.{table_format}').is_file()
    assert record(saved)['status'] == 'succeeded'

    # registered -> spot finding + extraction (+ decode/filter)
    fov = ds.fov('FOV').load_checkpoint('registered')
    for name in ROUNDS:
        assert fov.images[name].dtype == reference.images[name].dtype
        np.testing.assert_array_equal(fov.images[name], reference.images[name])
        assert fov.metadata[name] == reference.metadata[name]
    assert fov.registration_results == reference.registration_results
    assert fov.registration_attempts == reference.registration_attempts
    fov.run(PipelineConfig(**DETECT, **DECODE), execution=execution)
    assert_spots_equal(fov.spot_result, reference.spot_result)
    assert_intensities_equal(fov.intensity_result, reference.intensity_result)
    assert_downstream_equal(fov, reference)

    # candidates -> decode + filter, without images
    fov = ds.fov('FOV').load_checkpoint('candidates')
    assert not fov.images
    assert_spots_equal(fov.spot_result, reference.spot_result)
    assert_intensities_equal(fov.intensity_result, reference.intensity_result)
    fov.run(PipelineConfig(**DECODE), execution=execution)
    assert_downstream_equal(fov, reference)

    # pre_qc -> filter, with a changed predicate
    fov = ds.fov('FOV').load_checkpoint('pre_qc')
    assert_decoding_equal(fov.decoding_result, reference.decoding_result)
    rerun = ReadFilterConfig(call_statuses=('assigned', 'unmatched'))
    fov.run(PipelineConfig(filtering=rerun), execution=execution)
    expected = resident(ds, z).run(replace(full(), filtering=rerun))
    pd.testing.assert_frame_equal(fov.filtering_result.table, expected.filtering_result.table)


def test_dense_transforms_and_codebook_aware_tables_round_trip(tmp_path):
    ds = dataset(tmp_path)
    config = PipelineConfig(registration=(RegistrationStep(DemonsConfig(iterations=(1,))),
                                          RegistrationStep(TranslationConfig())), **DETECT,
                            decoding=CodebookAwareDecoderConfig(diagnostics=True), filtering=ReadFilterConfig())
    saved = resident(ds).run(config, checkpoints=CheckpointConfig())
    directory = saved.paths.checkpoint_dir
    assert sorted(p.name for p in (directory / 'registered').iterdir()) == [
        'round1.tif', 'round2.tif', 'round2_field.npz', 'transforms.json']
    with np.load(directory / 'registered' / 'round2_field.npz') as fields:
        assert fields.files == ['result_0']
    loaded = read_checkpoint(directory, 'registered')
    restored = loaded['registration_results']['round2']
    original = saved.registration_results['round2']
    np.testing.assert_array_equal(restored[0].transform.displacement_zyx, original[0].transform.displacement_zyx)
    assert restored[0].transform.displacement_zyx.dtype == original[0].transform.displacement_zyx.dtype
    assert [r.diagnostics for r in restored] == [r.diagnostics for r in original]
    assert [r.application_config for r in restored] == [r.application_config for r in original]
    assert restored[1] == original[1]
    fov = ds.fov('FOV').load_checkpoint('pre_qc')
    assert_decoding_equal(fov.decoding_result, saved.decoding_result)
    fov.run(PipelineConfig(filtering=ReadFilterConfig()))
    assert_downstream_equal(fov, saved)


def test_wide_table_layout_and_vectorized_parse(tmp_path):
    ds = dataset(tmp_path)
    fov = resident(ds).run(full())
    frame = candidates_frame(fov.spot_result, fov.intensity_result)
    assert list(frame.columns) == (['spot_namespace', 'spot_id', 'z', 'y', 'x', 'channel', 'peak_intensity']
        + [f'sig_{r}_{c}' for r in ROUNDS for c in CHANNELS] + ['valid_round1', 'valid_round2'])
    values = fov.intensity_result.values
    np.testing.assert_array_equal(frame['sig_round2_d'], values[:, 3, 1])
    spots, parsed, valid = parse_candidates(frame, ROUNDS, CHANNELS)
    np.testing.assert_array_equal(parsed, values)
    assert valid.dtype == bool and valid.all()
    with pytest.raises(ValueError, match='duplicate'):
        candidates_frame(fov.spot_result, replace(fov.intensity_result, round_labels=('x_y', 'x'),
                                                  channel_labels=('b', 'y_b', 'c', 'd')))


def test_empty_spot_table_and_all_invalid_signals_round_trip(tmp_path):
    ds = dataset(tmp_path)
    empty = resident(ds)
    empty.images = {k: np.zeros_like(v) for k, v in empty.images.items()}
    empty.run(full(), checkpoints=CheckpointConfig())
    assert len(empty.spot_result.spots) == 0
    fov = ds.fov('FOV').load_checkpoint('candidates')
    assert_spots_equal(fov.spot_result, empty.spot_result)
    assert fov.intensity_result.values.shape == (0, 4, 2)
    assert_intensities_equal(fov.intensity_result, empty.intensity_result)
    fov.run(PipelineConfig(**DECODE))
    assert_downstream_equal(fov, empty)
    fov = ds.fov('FOV').load_checkpoint('pre_qc')
    assert_decoding_equal(fov.decoding_result, empty.decoding_result)

    ds = dataset(tmp_path / 'invalid')
    source = resident(ds).run(PipelineConfig(**DETECT))
    source.intensity_result = replace(source.intensity_result,
                                      valid=np.zeros_like(source.intensity_result.valid))
    source.save_checkpoint('candidates', checkpoints=CheckpointConfig(table_format='parquet'))
    source.run(PipelineConfig(**DECODE))
    source.save_checkpoint('pre_qc')
    assert source.decoding_result.table.failure_reason.eq('invalid_measurement').all()
    fov = ds.fov('FOV').load_checkpoint('candidates')
    assert_intensities_equal(fov.intensity_result, source.intensity_result)
    assert not fov.intensity_result.valid.any()
    fov.run(PipelineConfig(**DECODE))
    assert_downstream_equal(fov, source)
    fov = ds.fov('FOV').load_checkpoint('pre_qc')
    assert_decoding_equal(fov.decoding_result, source.decoding_result)


def test_csv_and_parquet_reload_identically(tmp_path):
    ds = dataset(tmp_path)
    csv = resident(ds).run(full(), checkpoints=CheckpointConfig(directory=tmp_path / 'csv'))
    resident(ds).run(full(), checkpoints=CheckpointConfig(directory=tmp_path / 'parquet', table_format='parquet'))
    for stage in ('candidates', 'pre_qc'):
        a = read_checkpoint(tmp_path / 'csv' / 'FOV', stage)
        b = read_checkpoint(tmp_path / 'parquet' / 'FOV', stage)
        assert a.keys() == b.keys()
        if stage == 'candidates':
            assert_spots_equal(a['spot_result'], b['spot_result'])
            assert_intensities_equal(a['intensity_result'], b['intensity_result'])
            assert_spots_equal(a['spot_result'], csv.spot_result)
        else:
            pd.testing.assert_frame_equal(a['decoding_result'].table, b['decoding_result'].table)
            assert a['decoding_result'].diagnostics == b['decoding_result'].diagnostics
    header = json.loads((tmp_path / 'csv' / 'FOV' / 'pre_qc.json').read_text())
    assert header['dtypes']['gene_id'] == 'string' and header['dtypes']['wta_l2_nll'] == 'float64'
    # Empty strings, missing values and infinities survive CSV text.
    table = _read_table(tmp_path / 'csv' / 'FOV' / 'pre_qc.csv', header['dtypes'])
    assert table.failure_reason.eq('').sum() == 1 and table.gene_id.isna().sum() == 3


# Printable strings pandas reads as missing by default, the escape itself, and CSV syntax.
TRICKY_STRINGS = ['', '<NA>', 'NA', 'NaN', 'nan', 'null', 'NULL', 'N/A', 'n/a', 'None', '#N/A',
                  '#NA', '-NaN', '-1.#IND', '1.#QNAN', '<NA', ' <NA> ', '\\', '\\<NA>', '\\\\x',
                  'a,b', 'q"uote', '  pad ']


@pytest.mark.parametrize('dtype', ['string', 'str'])
def test_string_columns_are_lossless_and_identical_in_csv_and_parquet(tmp_path, dtype):
    values = TRICKY_STRINGS + [None]
    frame = pd.DataFrame({'text': pd.array(values, dtype='string').astype(dtype),
                          'other': pd.array([None] + TRICKY_STRINGS, dtype='string'),
                          'score': np.r_[np.arange(len(TRICKY_STRINGS), dtype=float), np.nan]})
    tables = {}
    for table_format in ('csv', 'parquet'):
        name, dtypes = _write_table(frame, tmp_path, 'table', table_format)
        tables[table_format] = _read_table(tmp_path / name, dtypes)
        pd.testing.assert_frame_equal(tables[table_format], frame)
    pd.testing.assert_frame_equal(tables['csv'], tables['parquet'])
    csv = tables['csv']
    assert csv.text.isna().tolist() == [False] * len(TRICKY_STRINGS) + [True]
    assert csv.text.iloc[1] == '<NA>' and csv.other.iloc[2] == '<NA>' and pd.isna(csv.other.iloc[0])
    # Only the missing value is written as the bare token; the literal is escaped.
    raw = pd.read_csv(tmp_path / 'table.csv', dtype=str, keep_default_na=False)
    assert (raw.text.iloc[1], raw.text.iloc[-1], raw.text.iloc[0]) == ('\\<NA>', '<NA>', '')
    assert raw.text.iloc[TRICKY_STRINGS.index('\\<NA>')] == '\\\\<NA>'


@pytest.mark.parametrize('control', ['\x00', '\t', '\n', '\r', '\x1f', '\x7f'])
def test_csv_rejects_control_characters_without_writing(tmp_path, control):
    frame = pd.DataFrame({'spot_id': pd.array(['ok', f'a{control}b'], dtype='string'),
                          'score': np.array([1.0, 2.0])})
    with pytest.raises(ValueError, match="column 'spot_id'.*parquet"):
        _write_table(frame, tmp_path, 'table', 'csv')
    assert list(tmp_path.iterdir()) == []
    name, dtypes = _write_table(frame, tmp_path, 'table', 'parquet')
    pd.testing.assert_frame_equal(_read_table(tmp_path / name, dtypes), frame)


def test_nul_gene_id_fails_csv_and_round_trips_through_parquet(tmp_path):
    ds = dataset(tmp_path)
    ds.codebook = Codebook(pd.DataFrame({'gene_id': ['a\x00b', 'other'], 'color_sequence': ['11', '22']}),
                           ROUNDS, CHANNELS)
    fov = resident(ds)
    with pytest.raises(ValueError, match="column 'gene_id'.*parquet"):
        fov.run(full(), checkpoints=CheckpointConfig(directory=tmp_path / 'csv'))
    assert not (tmp_path / 'csv' / 'FOV' / 'pre_qc.csv').exists()
    assert record(fov, tmp_path / 'csv')['error']['step'] == 'write_checkpoint:pre_qc'
    saved = resident(ds).run(full(), checkpoints=CheckpointConfig(directory=tmp_path / 'parquet',
                                                                    table_format='parquet'))
    assert saved.decoding_result.table.gene_id.iloc[0] == 'a\x00b'
    loaded = read_checkpoint(tmp_path / 'parquet' / 'FOV', 'pre_qc')['decoding_result']
    assert_decoding_equal(loaded, saved.decoding_result)
    assert loaded.table.gene_id.iloc[0] == 'a\x00b'


def test_literal_na_gene_id_round_trips_in_csv_and_parquet(tmp_path):
    ds = dataset(tmp_path)
    ds.codebook = Codebook(pd.DataFrame({'gene_id': ['<NA>', 'NA'], 'color_sequence': ['11', '22']}),
                           ROUNDS, CHANNELS)
    saved = resident(ds).run(full(), checkpoints=CheckpointConfig(directory=tmp_path / 'csv'))
    resident(ds).run(full(), checkpoints=CheckpointConfig(directory=tmp_path / 'parquet', table_format='parquet'))
    genes = saved.decoding_result.table.gene_id
    assert genes.iloc[0] == '<NA>' and genes.isna().sum() == 3
    loaded = {f: read_checkpoint(tmp_path / f / 'FOV', 'pre_qc')['decoding_result'] for f in ('csv', 'parquet')}
    assert_decoding_equal(loaded['csv'], saved.decoding_result)
    assert_decoding_equal(loaded['csv'], loaded['parquet'])
    for table_format in ('csv', 'parquet'):
        fov = ds.fov('FOV').load_checkpoint('pre_qc', checkpoints=CheckpointConfig(directory=tmp_path / table_format))
        fov.run(PipelineConfig(filtering=ReadFilterConfig()))
        assert_downstream_equal(fov, saved)
        assert fov.filtering_result.accepted.gene_id.tolist() == ['<NA>']


def test_parquet_without_pyarrow_fails_before_image_load(tmp_path, monkeypatch):
    import starfinder.io as io
    ds = dataset(tmp_path)
    fov = resident(ds).run(PipelineConfig(**DETECT))
    calls = []
    monkeypatch.setattr(io, 'load_round', lambda *a, **k: calls.append(a))
    # pandas itself may use pyarrow for strings, so it is hidden only around the calls.
    monkeypatch.setitem(sys.modules, 'pyarrow', None)
    config = replace(full(), load=ImageLoadConfig(channel_labels=CHANNELS))
    with pytest.raises(ImportError, match='pyarrow'):
        ds.fov('FOV').run(config, checkpoints=CheckpointConfig(table_format='parquet'))
    assert calls == []
    assert not (tmp_path / 'out' / 'checkpoints').exists()
    with pytest.raises(ImportError, match='pyarrow'):
        fov.save_checkpoint('candidates', checkpoints=CheckpointConfig(table_format='parquet'))
    assert not (tmp_path / 'out' / 'checkpoints').exists()


def test_injected_registration_failure_is_recorded_and_propagates(tmp_path, monkeypatch):
    import starfinder.registration as registration
    injected = RuntimeError('injected estimation failure')

    def fail(*args, **kwargs):
        raise injected
    monkeypatch.setattr(registration, 'estimate_transform', fail)
    ds = dataset(tmp_path)
    fov = resident(ds)
    with pytest.raises(RuntimeError) as raised:
        fov.run(full(), checkpoints=CheckpointConfig())
    assert raised.value is injected
    data = record(fov)
    assert data['status'] == 'failed' and data['ended_at'] is not None
    error = data['error']
    assert (error['step'], error['round'], error['type'], error['message']) == (
        'register', 'round2', 'RuntimeError', 'injected estimation failure')
    assert 'injected estimation failure' in error['traceback']
    assert data['steps'][-1] == dict(data['steps'][-1], name='register', round='round2', status='failed')
    assert data['registration']['round2'][0]['outcome'] == 'failed'
    assert fov._run_record is None


def test_keyboard_interrupt_is_recorded_as_interrupted(tmp_path, monkeypatch):
    import starfinder.barcode as barcode

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt
    monkeypatch.setattr(barcode, 'decode_barcodes', interrupt)
    fov = resident(dataset(tmp_path))
    with pytest.raises(KeyboardInterrupt):
        fov.run(full(), checkpoints=CheckpointConfig())
    data = record(fov)
    assert data['status'] == 'interrupted'
    assert (data['error']['step'], data['error']['round'], data['error']['type']) == (
        'decode_barcodes', None, 'KeyboardInterrupt')
    # Earlier stages remain usable after the interruption.
    assert data['checkpoints']['candidates'] == ['candidates.csv', 'candidates.json']
    assert 'pre_qc' not in data['checkpoints']


def test_failed_final_write_never_replaces_original_error(tmp_path, monkeypatch, caplog):
    import starfinder.dataset._run_record as run_record
    import starfinder.registration as registration
    injected = ValueError('original')
    monkeypatch.setattr(registration, 'estimate_transform', lambda *a, **k: (_ for _ in ()).throw(injected))
    original_write = run_record.write_json

    def write(data, path):
        if data['status'] != 'running':
            raise OSError('disk full')
        original_write(data, path)
    monkeypatch.setattr(run_record, 'write_json', write)
    fov = resident(dataset(tmp_path))
    with pytest.raises(ValueError) as raised:
        fov.run(full(), checkpoints=CheckpointConfig())
    assert raised.value is injected
    assert 'Could not write' in caplog.text
    assert record(fov)['status'] == 'running'


def test_run_record_fields_inputs_and_hashes(tmp_path):
    import hashlib
    ds = dataset(tmp_path)
    for round_name, image in images(4).items():
        for index, channel in enumerate(CHANNELS):
            save_volume(image[..., index], tmp_path / round_name / 'FOV' / f'{channel}.tif',
                        metadata=ImageMetadata('common'))
    config = replace(full(), load=ImageLoadConfig(channel_labels=CHANNELS))
    fov = ds.fov('FOV').run(config, execution=ExecutionConfig('streaming'), checkpoints=CheckpointConfig())
    data = record(fov)
    assert set(data) == {'format_version', 'dataset_id', 'sample_id', 'fov_id', 'subtile_id', 'status',
        'started_at', 'ended_at', 'error', 'code', 'environment', 'config', 'inputs', 'steps',
        'registration', 'counts', 'checkpoint_directory', 'checkpoints'}
    assert (data['dataset_id'], data['sample_id'], data['fov_id'], data['error']) == ('data', 'sample', 'FOV', None)
    assert set(data['code']) == {'version', 'git_commit', 'git_dirty'}
    assert data['environment']['packages']['numpy'] == np.__version__
    assert data['config']['execution']['mode'] == 'streaming'
    assert data['config']['checkpoints']['stages'] == ['registered', 'candidates', 'pre_qc']
    assert data['config']['pipeline']['registration'][0]['config']['method'] == 'translation'
    assert len(data['inputs']) == 8
    first = data['inputs'][0]
    assert first['sha256'] == hashlib.sha256(open(first['path'], 'rb').read()).hexdigest()
    assert data['counts'] == {'spots': 4, 'intensities': 4,
        'call_status': {'ambiguous': 2, 'assigned': 1, 'unmatched': 1},
        'filtering': {'total': 4, 'accepted': 1, 'rejected': 3}}
    names = [(s['name'], s['round']) for s in data['steps']]
    assert ('load_images', 'round2') in names and ('register', 'round2') in names
    assert ('write_checkpoint:registered', 'round2') in names and ('extract_round', 'round1') in names
    assert all(s['status'] == 'succeeded' and s['seconds'] >= 0 for s in data['steps'])
    assert data['checkpoints']['registered'][:2] == ['registered/round1.tif', 'registered/round2.tif']
    fov = ds.fov('FOV').run(config, checkpoints=CheckpointConfig(hash_inputs=False, overwrite=True))
    assert {entry['sha256'] for entry in record(fov)['inputs']} == {None}


def test_disabled_checkpoints_write_nothing_and_existing_directory_is_protected(tmp_path, monkeypatch):
    import starfinder.io as io
    ds = dataset(tmp_path)
    resident(ds).run(full())
    assert not (tmp_path / 'out').exists()
    resident(ds).run(full(), checkpoints=CheckpointConfig(stages=('pre_qc',)))
    assert sorted(p.name for p in (tmp_path / 'out' / 'checkpoints' / 'FOV').iterdir()) == [
        'pre_qc.csv', 'pre_qc.json', 'run.json']
    calls = []
    monkeypatch.setattr(io, 'load_round', lambda *a, **k: calls.append(a))
    config = replace(full(), load=ImageLoadConfig(channel_labels=CHANNELS))
    with pytest.raises(FileExistsError, match='overwrite'):
        ds.fov('FOV').run(config, checkpoints=CheckpointConfig())
    assert calls == []
    fov = resident(ds)
    with pytest.raises(FileExistsError):
        fov.run(full(), checkpoints=CheckpointConfig())
    assert fov.spot_result is None
    fov.run(full(), checkpoints=CheckpointConfig(overwrite=True))
    with pytest.raises(FileExistsError):
        fov.save_checkpoint('pre_qc')
    fov.save_checkpoint('pre_qc', checkpoints=CheckpointConfig(overwrite=True))
    for bad in (dict(stages=('final',)), dict(table_format='hdf5'), dict(overwrite=1)):
        with pytest.raises(ValueError):
            CheckpointConfig(**bad)


def test_overwrite_run_never_leaves_a_stale_stage_loadable(tmp_path, monkeypatch):
    import starfinder.barcode as barcode
    ds = dataset(tmp_path)
    resident(ds).run(full(), checkpoints=CheckpointConfig(hash_inputs=False))
    directory = tmp_path / 'out' / 'checkpoints' / 'FOV'
    (directory / 'notes.txt').write_text('kept')
    injected = RuntimeError('injected decoding failure')

    def fail(*args, **kwargs):
        raise injected
    monkeypatch.setattr(barcode, 'decode_barcodes', fail)
    with pytest.raises(RuntimeError):
        resident(ds).run(full(), checkpoints=CheckpointConfig(hash_inputs=False, overwrite=True))
    data = record(ds.fov('FOV'))
    assert data['status'] == 'failed' and 'pre_qc' not in data['checkpoints']
    # The earlier run's pre_qc files are gone, so they cannot be loaded as this run's.
    assert not (directory / 'pre_qc.json').exists() and not (directory / 'pre_qc.csv').exists()
    with pytest.raises(FileNotFoundError):
        ds.fov('FOV').load_checkpoint('pre_qc')
    assert (directory / 'notes.txt').read_text() == 'kept'
    ds.fov('FOV').load_checkpoint('candidates')


def test_load_checkpoint_rejects_mismatches_and_existing_results(tmp_path):
    ds = dataset(tmp_path)
    saved = resident(ds).run(full(), checkpoints=CheckpointConfig())
    for stage in ('registered', 'candidates', 'pre_qc'):
        with pytest.raises(ValueError, match='requires an FOV without'):
            saved.load_checkpoint(stage)
    fov = ds.fov('FOV').load_checkpoint('candidates')
    with pytest.raises(ValueError, match='spot_result'):
        fov.load_checkpoint('registered')
    fov.load_checkpoint('pre_qc')
    shutil.copytree(saved.paths.checkpoint_dir, tmp_path / 'out' / 'checkpoints' / 'OTHER')
    with pytest.raises(ValueError, match='FOV id'):
        ds.fov('OTHER').load_checkpoint('candidates')
    swapped = dataset(tmp_path, rounds=RoundState(list(ROUNDS), reference_round='round2'))
    with pytest.raises(ValueError, match='round labels'):
        swapped.fov('FOV').load_checkpoint('registered')
    reordered = dataset(tmp_path, channels=tuple(reversed(CHANNELS)))
    with pytest.raises(ValueError, match='channel order'):
        reordered.fov('FOV').load_checkpoint('pre_qc')
    with pytest.raises(FileNotFoundError):
        ds.fov('MISSING').load_checkpoint('pre_qc')
    with pytest.raises(ValueError, match='unknown checkpoint stage'):
        ds.fov('FOV').load_checkpoint('final')


def test_subtile_checkpoints_use_their_own_directory(tmp_path):
    ds = dataset(tmp_path)
    fov = resident(ds)
    fov.subtile_id = 3
    fov.run(full(), checkpoints=CheckpointConfig())
    directory = tmp_path / 'out' / 'checkpoints' / 'FOV' / 'subtile_3'
    assert (directory / 'run.json').is_file()
    loaded = ds.fov('FOV')
    loaded.subtile_id = 3
    loaded.load_checkpoint('candidates')
    assert_spots_equal(loaded.spot_result, fov.spot_result)


@pytest.mark.parametrize('shape,dtype', [((1, 5, 6, 1), np.float32), ((3, 5, 6, 2), np.uint16),
                                         ((1, 4, 4, 3), np.float64)])
def test_load_volume_zyxc_preserves_shape_dtype_and_geometry(tmp_path, shape, dtype):
    image = np.arange(np.prod(shape)).reshape(shape).astype(dtype)
    metadata = ImageMetadata('frame', spacing_zyx=(1.5, .25, .25), spatial_unit='um')
    save_volume(image, tmp_path / 'v.tif', metadata=metadata)
    loaded = load_volume_zyxc(tmp_path / 'v.tif', channel_labels=tuple('abc'[:shape[3]]))
    assert loaded.image.shape == shape and loaded.image.dtype == dtype
    np.testing.assert_array_equal(loaded.image, image)
    assert loaded.metadata == metadata and loaded.channel_labels == tuple('abc'[:shape[3]])
    with pytest.raises(ValueError, match='channel'):
        load_volume_zyxc(tmp_path / 'v.tif', channel_labels=('a', 'b', 'c', 'd'))
    save_volume(image[..., 0], tmp_path / 'zyx.tif')
    with pytest.raises(ValueError, match='ZYXC'):
        load_volume_zyxc(tmp_path / 'zyx.tif')


@pytest.mark.extended
def test_checkpoint_timing_on_small_synthetic_dataset(small_dataset, tmp_path, capsys):
    """Wall time of a 16x256x256, 4-round run with all checkpoints and of table build/parse."""
    from starfinder.io._checkpoint import read_header
    from starfinder.preprocessing import MinMaxNormalizationConfig
    for round_dir in (small_dataset / 'FOV_001').iterdir():
        if round_dir.is_dir():
            target = tmp_path / round_dir.name / 'FOV_001'
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(round_dir)
    ds = Dataset(tmp_path, tmp_path / 'output', 'test', 'small', 'out',
                 RoundState(['round1', 'round2', 'round3', 'round4'], reference_round='round1'),
                 ('ch00', 'ch01', 'ch02', 'ch03'))
    ds.load_codebook(small_dataset / 'codebook.csv')
    config = PipelineConfig(load=ImageLoadConfig(channel_labels=ds.channel_order),
        normalization=MinMaxNormalizationConfig('uint8', (0, 255), snr_threshold=5.0),
        registration=(RegistrationStep(TranslationConfig()),), detection=LocalMaximaConfig(),
        extraction=NeighborhoodSumConfig(), decoding=WtaDecoderConfig(diagnostics=True),
        filtering=ReadFilterConfig())
    start = perf_counter()
    plain = ds.fov('FOV_001').run(config)
    plain_seconds = perf_counter() - start
    start = perf_counter()
    saved = ds.fov('FOV_001').run(config, checkpoints=CheckpointConfig())
    checkpoint_seconds = perf_counter() - start
    assert_downstream_equal(saved, plain)
    directory = saved.paths.checkpoint_dir
    start = perf_counter()
    frame = candidates_frame(saved.spot_result, saved.intensity_result)
    build_seconds = perf_counter() - start
    header = read_header(directory, 'candidates')
    start = perf_counter()
    parsed = _read_table(directory / 'candidates.csv', header['dtypes'])
    parse_candidates(parsed, header['signals']['round_labels'], header['signals']['channel_labels'])
    parse_seconds = perf_counter() - start
    pd.testing.assert_frame_equal(parsed, frame)
    resumed = ds.fov('FOV_001').load_checkpoint('candidates').run(PipelineConfig(
        decoding=WtaDecoderConfig(diagnostics=True), filtering=ReadFilterConfig()))
    assert_downstream_equal(resumed, plain)
    steps = json.loads((directory / 'run.json').read_text())['steps']
    written = sum(p.stat().st_size for p in directory.rglob('*') if p.is_file())
    with capsys.disabled():
        print(f"\n[checkpoint timing] spots={len(frame)} columns={frame.shape[1]} "
              f"run_without={plain_seconds:.2f}s run_with_checkpoints={checkpoint_seconds:.2f}s "
              f"checkpoint_writes={sum(s['seconds'] for s in steps if s['name'].startswith('write_checkpoint')):.2f}s "
              f"table_build={build_seconds:.4f}s table_read_parse={parse_seconds:.4f}s bytes={written}")
