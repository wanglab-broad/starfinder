"""D1/S1/X1: independent literal population and source-identity oracles."""
from dataclasses import replace
import json
import numpy as np
import pandas as pd
import pytest
from starfinder.barcode import decode_barcodes, filter_reads, WtaDecoderConfig, CodebookAwareDecoderConfig, ReadFilterConfig
from starfinder.io import (checkpoint_reference, save_candidate_checkpoint, save_decoded_checkpoint,
    load_decoded_checkpoint, save_final_checkpoint, load_final_checkpoint, save_molecule_index, load_molecule_index)
from starfinder.provenance import RunRecorder, read_run
from test.test_candidate_checkpoints import oracle, rewrite
from test.test_provenance import fixture


def create(root, *, depth=1, variant='normal', FOV='FOV', namespace=None, decoder=None, omitted=False):
    spots, intensities, book, *_ = oracle.signal_example(depth)
    if namespace:
        spots = replace(spots, spot_namespace=namespace)
        intensities = replace(intensities, spot_namespace=namespace)
    if variant == 'empty':
        spots = replace(spots, spots=spots.spots.iloc[:0])
        intensities = replace(intensities, values=intensities.values[:0], valid=intensities.valid[:0], spot_ids=())
    if variant == 'invalid': intensities.valid[0, 1] = False
    if variant == 'tie': intensities.values[0, 0, 0] = 7
    context = dict(dataset_id='artifact-contract-v1', sample_id='sample', FOV=FOV,
        run_id=f'run-{FOV}', config={'literal': (1, 2)}, code={'commit': 'test-revision'})
    source = None
    if not omitted:
        candidate = save_candidate_checkpoint(root/'candidates', spots, intensities, codebook=book, **context)
        for name in ('candidates', 'signals', 'validity'):
            rewrite(candidate.path, name, lambda a: a.take(np.arange(len(a), dtype=np.int64)[::-1]))
        source = checkpoint_reference(candidate.path)
    decoded = decode_barcodes(intensities, book, config=decoder or WtaDecoderConfig(diagnostics=True))
    filtered = filter_reads(decoded, config=ReadFilterConfig(call_statuses=()) if variant == 'rejected' else ReadFilterConfig())
    links = {'h5ad': {'uri': 'existing.h5ad', 'unverified_reason': 'external link only'},
             'assignment': {'population': 'caller-defined', 'uri': 'assignment.csv'}}
    pre = save_decoded_checkpoint(root/'decoded', spots, decoded, book, candidate_source=source,
        trace_unavailable_reason='candidates_signals_disabled' if omitted else None, links=links, **context)
    final = save_final_checkpoint(root/'final', filtered, decoded_source=checkpoint_reference(pre),
                                   config={}, code=context['code'], links=links)
    return spots, intensities, decoded, filtered, pre, final, source


@pytest.mark.parametrize('depth', [1, 3])
@pytest.mark.parametrize('variant', ['normal', 'empty', 'invalid', 'rejected', 'tie'])
def test_populations_roundtrip(tmp_path, depth, variant):
    spots, intensities, decoded, filtered, pre, final, source = create(tmp_path, depth=depth, variant=variant)
    saved = load_decoded_checkpoint(pre)
    pd.testing.assert_frame_equal(saved.decoded.table, decoded.table, check_exact=True)
    assert saved.decoded.config == decoded.config
    for key, value in decoded.diagnostics.items():
        actual = saved.decoded.diagnostics[key]
        if isinstance(value, pd.DataFrame): pd.testing.assert_frame_equal(actual, value, check_exact=True)
        elif isinstance(value, np.ndarray): assert actual.dtype == value.dtype and actual.tobytes() == value.tobytes()
        else: assert actual == value
    loaded = load_final_checkpoint(final)
    pd.testing.assert_frame_equal(loaded.filtering.table, filtered.table, check_exact=True)
    pd.testing.assert_frame_equal(filter_reads(saved.decoded, config=filtered.config).table, filtered.table, check_exact=True)
    assert loaded.filtering.counts == filtered.counts and loaded.filtering.fractions == filtered.fractions
    if variant == 'normal':
        assert loaded.filtering.counts == {'total': 2, 'accepted': 1, 'rejected': 1}
        assert loaded.filtering.table.rejection_reasons.tolist() == ['', 'call_status']
        assert loaded.molecule_table().spot_id.tolist() == ['A']
        assert loaded.molecule_table().gene_id.tolist() == ['gene-A']
        trace = saved.source_trace(spot_namespace=spots.spot_namespace, spot_id='A')
        np.testing.assert_array_equal(trace['values'], [[0, 9], [7, 0], [0, 0], [0, 0]])
        assert trace['round_labels'] == ('round10', 'round2')
        assert trace['channel_labels'] == ('ch02', 'ch00', 'ch03', 'ch01')
        assert trace['metadata'] == spots.metadata
    elif variant == 'empty':
        assert loaded.filtering.counts == {'total': 0, 'accepted': 0, 'rejected': 0}
        assert loaded.filtering.fractions == {'accepted': None}
        assert loaded.filtering.diagnostics['undefined_fraction_reasons'] == {'accepted': 'empty_population'}
    else: assert loaded.filtering.counts['accepted'] == 0
    comparison = saved.comparison_metadata()
    assert comparison['coordinate_axes'] == 'ZYX' and comparison['coordinate_index_base'] == 0
    assert comparison['coordinate_unit'] == 'voxel_index'
    assert comparison['color_to_channel'] == {'1': 1, '2': 0, '3': 3, '4': 2}
    assert comparison['metadata'].spacing_zyx is None
    assert loaded.artifact['payload']['links'] == saved.artifact['payload']['links']
    assert not any('signal' in x.name for x in pre.parent.iterdir())


def test_codebook_aware(tmp_path):
    *_, pre, final, source = create(tmp_path, decoder=CodebookAwareDecoderConfig(diagnostics=True))
    saved = load_decoded_checkpoint(pre)
    assert isinstance(saved.decoded.config, CodebookAwareDecoderConfig)
    assert 'probability_nll' in saved.decoded.table and 'wta_l2_nll' not in saved.decoded.table
    assert load_final_checkpoint(final).molecule_table().spot_id.tolist() == ['A']


def test_sample_order_batches_collision_and_lookup(tmp_path):
    first = create(tmp_path/'one', FOV='one', namespace='namespace-one')
    second = create(tmp_path/'two', FOV='two', namespace='namespace-two', depth=3)
    third = create(tmp_path/'empty', FOV='empty', variant='empty')
    refs = tuple(checkpoint_reference(case[5]) for case in (second, third, first))
    path = save_molecule_index(tmp_path/'sample.json', refs, dataset_id='artifact-contract-v1', sample_id='sample', section_id='section')
    index = load_molecule_index(path)
    for accepted in (True, False):
        batches = list(index.iter_batches(batch_size=1, accepted_only=accepted))
        pd.testing.assert_frame_equal(pd.concat([b.table for b in batches], ignore_index=True), index.read_table(accepted_only=accepted), check_exact=True)
        assert any(b.table.empty for b in batches)
        assert index.read_table(accepted_only=accepted).FOV.tolist() == (['two', 'one'] if accepted else ['two', 'two', 'one', 'one'])
    for case in (first, second):
        source = case[6]
        trace = index.source_trace(run_id=source.run_id, candidate_artifact_id=source.artifact_id,
            spot_namespace=case[0].spot_namespace, spot_id='A')
        assert trace['candidate']['z'] == (0 if case is first else 1)
    with pytest.raises(ValueError, match='missing or ambiguous'):
        index.source_trace(run_id='missing', candidate_artifact_id='missing', spot_namespace='namespace-one', spot_id='A')
    with pytest.raises(ValueError, match='duplicate FOV'):
        save_molecule_index(tmp_path/'bad.json', (refs[0], refs[0]), dataset_id='artifact-contract-v1', sample_id='sample')
    first[6].path.unlink()
    with pytest.raises(FileNotFoundError):
        index.source_trace(run_id=first[6].run_id, candidate_artifact_id=first[6].artifact_id, spot_namespace='namespace-one', spot_id='A')


def test_omitted_and_source_binding(tmp_path):
    *_, pre, final, _ = create(tmp_path/'omitted', omitted=True)
    saved = load_final_checkpoint(final)
    assert saved.pre_qc.source_trace(spot_namespace=saved.pre_qc.spots.spot_namespace, spot_id='A') == {'available': False, 'reason': 'candidates_signals_disabled'}
    assert saved.molecule_table().candidate_artifact_id.isna().all()
    assert saved.molecule_table().trace_unavailable_reason.tolist() == ['candidates_signals_disabled']
    with pytest.raises(ValueError, match='unknown'):
        saved.pre_qc.source_trace(spot_namespace='wrong', spot_id='A')
    a = json.loads(pre.read_text())
    a['payload']['trace_unavailable_reason'] = None
    pre.write_text(json.dumps(a))
    with pytest.raises(ValueError, match='availability'): load_decoded_checkpoint(pre)


@pytest.mark.parametrize('mutation', ['version', 'stage', 'missing', 'checksum', 'truncated', 'binding', 'qc'])
def test_corrupt_artifacts(tmp_path, mutation):
    *_, pre, final, _ = create(tmp_path)
    a = json.loads(final.read_text())
    if mutation == 'version': a['schema_version'] = 7
    if mutation == 'stage': a['stage'] = 'decoded_pre_qc'
    if mutation == 'missing': (final.parent/a['components'][0]['path']).unlink()
    if mutation == 'checksum': a['components'][0]['sha256'] = '0'*64
    if mutation == 'truncated': (final.parent/a['components'][0]['path']).write_bytes(b'PAR1')
    if mutation == 'binding': a['artifact_id'] = 'other'
    if mutation == 'qc': a['payload']['counts']['mapping']['accepted'] = 2
    final.write_text(json.dumps(a))
    with pytest.raises(FileNotFoundError if mutation == 'missing' else ValueError): load_final_checkpoint(final)


@pytest.mark.parametrize('mode', ['batch', 'streaming'])
@pytest.mark.parametrize('enabled', [True, False])
def test_pipeline_saved_stages(tmp_path, mode, enabled):
    from starfinder.dataset import ExecutionConfig
    fov, config = fixture(tmp_path)
    rec = RunRecorder(tmp_path/'run', dataset_id='literal', sample_id='sample', save_candidates_signals=enabled)
    fov.run(config, provenance=rec, execution=ExecutionConfig(mode))
    run = read_run(rec.path)
    assert run['status'] == 'succeeded'
    pre = load_decoded_checkpoint(fov.decoded_checkpoint_path)
    final = load_final_checkpoint(fov.final_checkpoint_path)
    pd.testing.assert_frame_equal(final.filtering.table, fov.filtering_result.table, check_exact=True)
    assert (pre.candidate_source is not None) is enabled
    for stage in ('decoded_pre_qc', 'final_accepted'):
        assert next(a for a in run['artifacts'] if a['stage'] == stage)['status'] == 'complete'
    names = [e['operation'] for e in run['events'] if e['outcome'] == 'succeeded']
    assert names.index('decode_barcodes') < names.index('save_decoded_checkpoint') < names.index('filter_reads') < names.index('save_final_checkpoint')


def test_final_write_failure_retains_pre_qc(tmp_path, monkeypatch):
    import starfinder.io as io
    fov, config = fixture(tmp_path)
    rec = RunRecorder(tmp_path/'run', dataset_id='literal', sample_id='sample')
    def fail(*args, **kwargs): raise OSError('final storage failure')
    monkeypatch.setattr(io, 'save_final_checkpoint', fail)
    with pytest.raises(OSError, match='final storage failure'): fov.run(config, provenance=rec)
    run = read_run(rec.path)
    assert run['status'] == 'failed' and run['failures'][-1]['category'] == 'serialization'
    assert next(a for a in run['artifacts'] if a['stage'] == 'decoded_pre_qc')['status'] == 'complete'
    assert next(a for a in run['artifacts'] if a['stage'] == 'final_accepted')['status'] == 'failed'
    assert len(load_decoded_checkpoint(fov.decoded_checkpoint_path).decoded.table) == 1


def test_reordered_decoder_and_physical_tables(tmp_path):
    spots, intensities, book, *_ = oracle.signal_example(3)
    reversed_intensities = replace(intensities, spot_ids=intensities.spot_ids[::-1],
        values=intensities.values[::-1], valid=intensities.valid[::-1])
    decoded = decode_barcodes(reversed_intensities, book, config=WtaDecoderConfig(diagnostics=True))
    expected = decode_barcodes(intensities, book, config=WtaDecoderConfig(diagnostics=True))
    pre = save_decoded_checkpoint(tmp_path/'decoded', spots, decoded, book,
        trace_unavailable_reason='not saved', dataset_id='literal', sample_id='sample',
        FOV='FOV', run_id='run', config={}, code={'commit': 'test'})
    raw = json.loads(pre.read_text())
    # Shuffle only indexed payload tables; diagnostic tables retain their own order.
    for key in ('calls', 'spots', 'codebook'):
        name = raw['payload'][key]['table']
        rewrite(pre, name, lambda a: a.take(np.arange(len(a), dtype=np.int64)[::-1]))
    saved = load_decoded_checkpoint(pre)
    pd.testing.assert_frame_equal(saved.decoded.table, expected.table, check_exact=True)
    for key in ('probabilities', 'wta_round_l2_nll'):
        np.testing.assert_array_equal(saved.decoded.diagnostics[key], expected.diagnostics[key], strict=True)
    final = save_final_checkpoint(tmp_path/'final', filter_reads(decoded),
        decoded_source=checkpoint_reference(pre), config={}, code={'commit': 'test'})
    raw = json.loads(final.read_text())
    for key in ('qc', 'accepted'):
        rewrite(final, raw['payload'][key]['table'], lambda a: a.take(np.arange(len(a), dtype=np.int64)[::-1]))
    pd.testing.assert_frame_equal(load_final_checkpoint(final).filtering.table, filter_reads(expected).table, check_exact=True)


def test_same_namespace_different_artifact_and_geometry(tmp_path):
    # Even a legacy namespace collision cannot redirect the full source locator.
    first = create(tmp_path/'one', FOV='one', namespace='same')
    second = create(tmp_path/'two', FOV='two', namespace='same', depth=3)
    path = save_molecule_index(tmp_path/'index.json', tuple(checkpoint_reference(c[5]) for c in (first, second)),
        dataset_id='artifact-contract-v1', sample_id='sample')
    index = load_molecule_index(path)
    assert index.read_table().spot_id.tolist() == ['A', 'A']
    for case, z in ((first, 0.), (second, 1.)):
        source = case[6]
        assert index.source_trace(run_id=source.run_id, candidate_artifact_id=source.artifact_id,
            spot_namespace='same', spot_id='A')['candidate']['z'] == z


def test_actual_transform_and_nullable_fields(tmp_path):
    from starfinder.image import ImageMetadata
    from starfinder.registration import DenseDisplacementTransform
    spots, intensities, book, *_ = oracle.signal_example(3)
    metadata = ImageMetadata('reference', (2, 3, 4), (10, 20, 30), ((1, 0, 0), (0, -1, 0), (0, 0, -1)), 'um')
    spots = replace(spots, metadata=metadata)
    decoded = decode_barcodes(intensities, book, config=WtaDecoderConfig())
    decoded.table['cell_id'] = pd.Series([None, 4], dtype='Int16')
    decoded.table['optional_flag'] = pd.Series([False, None], dtype='boolean')
    decoded.table['optional_score'] = np.array([-0., np.nan], dtype=np.float32)
    field = np.zeros((3, 4, 5, 3), dtype=np.float32)
    field[..., 2] = .5
    transform = DenseDisplacementTransform(field, (3, 4, 5), (3, 4, 5), metadata, metadata)
    pre = save_decoded_checkpoint(tmp_path/'decoded', spots, decoded, book,
        trace_unavailable_reason='not saved', links={'actual_transform': transform},
        dataset_id='literal', sample_id='sample', FOV='FOV', run_id='run', config={}, code={'commit': 'test'})
    saved = load_decoded_checkpoint(pre)
    pd.testing.assert_frame_equal(saved.decoded.table, decoded.table, check_exact=True)
    assert np.signbit(saved.decoded.table.optional_score.iloc[0])
    assert saved.spots.metadata == metadata
    np.testing.assert_array_equal(saved.artifact['payload']['links']['actual_transform'].displacement_zyx, field, strict=True)
    final = save_final_checkpoint(tmp_path/'final', filter_reads(decoded),
        decoded_source=checkpoint_reference(pre), config={}, code={'commit': 'test'})
    assert load_final_checkpoint(final).molecule_table().cell_id.isna().all()


@pytest.mark.parametrize('change', ['missing_field', 'bad_config', 'bad_probability_shape', 'parent', 'unknown_field'])
def test_malformed_decoded_metadata(tmp_path, change):
    *_, pre, final, source = create(tmp_path)
    a = json.loads(pre.read_text())
    if change == 'missing_field': del a['payload']['calls']
    if change == 'bad_config': a['payload']['decoder_config']['fields']['method'] = 'foreign'
    if change == 'bad_probability_shape': a['payload']['diagnostics']['mapping']['probabilities']['shape'] = [1, 4, 2]
    if change == 'parent': a['parents'] = []
    if change == 'unknown_field': a['payload']['unknown'] = 'foreign'
    pre.write_text(json.dumps(a))
    with pytest.raises(ValueError): load_decoded_checkpoint(pre)
