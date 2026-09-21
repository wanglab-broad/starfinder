"""I1/I2/X1: bounded literal oracles; no historical TIFFs or random seeds."""
from dataclasses import replace
import hashlib
import json

import h5py
import numpy as np
import pandas as pd
import pytest

from starfinder.barcode import Codebook, NeighborhoodSumConfig, WtaDecoderConfig, decode_barcodes, extract_intensities, filter_reads
from starfinder.dataset import RoundState
from starfinder.image import ImageMetadata
from starfinder.io import (ImageLayer, ImageLoadConfig, ImageLoadResult, ImageProcessingState,
    load_image_checkpoint, load_round, save_image_checkpoint, save_volume)
from starfinder.registration import (DenseDisplacementTransform, RegistrationDiagnostics,
    RegistrationResult, TranslationConfig, TranslationTransform, WarpConfig, apply_transform)
from starfinder.spot_finding import LocalMaximaConfig, SpotFindingResult

CHANNELS = ('ch02', 'ch00', 'ch03', 'ch01')
ROUNDS = RoundState(['round10', 'round2'], [], 'round10')


def save(root, layers, *, rounds=ROUNDS, stage='prepared_input', **kwargs):
    return save_image_checkpoint(root, tuple(layers), rounds=rounds, stage=stage,
        dataset_id='image-contract-v1', sample_id='sample', FOV='FOV', run_id='run',
        config={'fixture': 'literal-v1'}, code={'commit': 'test-literal'}, **kwargs)


def layer(name, array, *, metadata=None, state=None, channels=CHANNELS, roles=None):
    metadata = metadata or ImageMetadata('reference')
    return ImageLayer(name, roles or (('sequencing', 'registration_reference') if name == 'round10' else ('sequencing',)),
        ImageLoadResult(array, metadata, channels, ()), state or ImageProcessingState(metadata))


def equal_image(a, b):
    assert a.image.dtype == b.image.dtype and a.image.shape == b.image.shape
    assert a.image.tobytes() == b.image.tobytes()  # includes signed zero
    assert a.metadata == b.metadata and a.channel_labels == b.channel_labels
    assert a.source_paths == b.source_paths and a.diagnostics == b.diagnostics


@pytest.mark.parametrize('depth', [1, 3])
@pytest.mark.parametrize('dtype', ['uint8', 'uint16', 'int16', 'float32', 'float64'])
def test_prepared_tiff_hdf5_exact(tmp_path, depth, dtype):
    metadata = ImageMetadata('reference', (2, 3, 4), (10, 20, 30),
                             ((1, 0, 0), (0, -1, 0), (0, 0, -1)), 'um')
    array = np.arange(depth * 4 * 5 * 4).reshape(depth, 4, 5, 4).astype(dtype)
    if array.dtype.kind in 'if':
        array.flat[1] = -3
    if array.dtype.kind == 'f':
        array.flat[0] = -0.0
        array.flat[2] = 0.25
    paths = []
    for c, label in enumerate(CHANNELS):
        name = f'{label}.tif'
        save_volume(array[..., c], tmp_path / name, metadata=metadata)
        paths.append(name)
    loaded = load_round(tmp_path, config=ImageLoadConfig(channel_labels=CHANNELS, source_paths=tuple(paths)))
    assert loaded.image.tobytes() == array.tobytes()
    stain = layer('stain', np.zeros((1, 2, 3), dtype=np.uint8), metadata=ImageMetadata('stain'),
                  channels=('DAPI',), roles=('stain',))
    reference = layer('registration', np.ones((2, 3, 4), dtype=np.float32),
                      metadata=ImageMetadata('other-reference'), channels=('ref',), roles=('registration_reference',))
    rounds = RoundState(['round10'], ['stain', 'registration'], 'registration')
    layers = (ImageLayer('round10', ('sequencing',), loaded, ImageProcessingState(metadata)), stain, reference)
    source = dict(source_id='tiff0', catalog='image-contract-v1', uri=str(tmp_path / paths[0]),
                  sha256=hashlib.sha256((tmp_path / paths[0]).read_bytes()).hexdigest(),
                  unverified_reason=None, selection={'channel': CHANNELS[0], 'series': 0})
    path = save(tmp_path / 'prepared', layers, rounds=rounds, sources=(source,))
    result = load_image_checkpoint(path, expected_stage='prepared_input', sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    assert result.rounds == rounds and result.artifact['payload']['sources'] == (source,)
    for before, after in zip(layers, result.layers):
        equal_image(before.loaded, after.loaded)
        assert before.processing == after.processing and before.roles == after.roles
    assert result.layers[1].loaded.metadata.spacing_zyx is None
    with pytest.raises(ValueError, match='registered_images stage required'):
        result.sequencing_images(require_registered=True)
    with h5py.File(path.parent / 'images.h5') as handle:
        ds = handle['/layers/layer0000/image']
        assert ds.chunks == (depth, 4, 5, 1)
        assert ds.compression == 'gzip' and ds.compression_opts == 4 and ds.shuffle


def registered_layers(depth, dense=False):
    shape = (depth, 4, 5)
    z = depth // 2
    reference, moving = ImageMetadata('reference'), ImageMetadata('moving')
    first = np.zeros((*shape, 4), dtype=np.uint16)
    first[z, 2, 1, 1] = 7
    second = np.zeros_like(first)
    second[z, 1, 2, 0] = 9
    if dense:
        field = np.zeros((*shape, 3), dtype=np.float32)
        field[..., 1] = -1
        field[..., 2] = 1
        transform = DenseDisplacementTransform(field, shape, shape, reference, moving)
        warp = WarpConfig(backend='scipy')
    else:
        transform = TranslationTransform((0, 1, -1), shape, shape, reference, moving)
        warp = WarpConfig()
    result = RegistrationResult(transform, RegistrationDiagnostics('translation', 'analytic_fixture', TranslationConfig()), warp)
    output = apply_transform(second, transform, config=warp)
    expected = np.zeros_like(first)
    expected[z, 2, 1, 0] = 9
    np.testing.assert_array_equal(output, expected, strict=True)
    state = ImageProcessingState(moving, operations=({'operation': 'apply_transform', 'config': warp},),
        registrations=(result,), terminal_state='applied', reason=None)
    return (layer('round10', first, state=ImageProcessingState(reference,
                  terminal_state='reference_unchanged', reason='reference selected; no warp')),
            layer('round2', output, state=state))


def downstream(images, depth):
    spots = SpotFindingResult(pd.DataFrame({'spot_id': pd.Series(['A', 'B'], dtype='string'),
        'z': [float(depth // 2)] * 2, 'y': [2., 2.], 'x': [1., 3.]}),
        ImageMetadata('reference'), '["image-contract-v1","sample","FOV",null]', LocalMaximaConfig(), {})
    signals = extract_intensities(images, spots, config=NeighborhoodSumConfig((0, 0, 0)))
    expected = np.array([[[0, 9], [7, 0], [0, 0], [0, 0]], np.zeros((4, 2))])
    np.testing.assert_array_equal(signals.values, expected, strict=True)
    book = Codebook(pd.DataFrame({'gene_id': ['gene-A'], 'color_sequence': ['12']}),
                    tuple(ROUNDS.sequencing_rounds), CHANNELS, {'1': 1, '2': 0, '3': 3, '4': 2})
    decoded = decode_barcodes(signals, book, config=WtaDecoderConfig())
    filtered = filter_reads(decoded)
    assert decoded.table.call_status.tolist() == ['assigned', 'no_signal']
    assert filtered.accepted.spot_id.tolist() == ['A']
    assert filtered.counts == {'total': 2, 'accepted': 1, 'rejected': 1}
    return signals, decoded, filtered


@pytest.mark.parametrize('depth', [1, 3])
@pytest.mark.parametrize('dense', [False, True])
def test_registered_downstream_exact(tmp_path, depth, dense):
    layers = registered_layers(depth, dense)
    before = downstream({x.round_label: x.loaded for x in layers}, depth)
    path = save(tmp_path / 'registered', layers, stage='registered_images',
                provenance={'uri': 'saved-run/run.json', 'sha256': 'a' * 64},
                parents=({'run_id': 'parent', 'artifact_id': 'prepared', 'sha256': 'b' * 64},))
    result = load_image_checkpoint(path)
    after = downstream(result.sequencing_images(require_registered=True), depth)
    for old, new in zip(layers, result.layers):
        equal_image(old.loaded, new.loaded)
    np.testing.assert_array_equal(before[0].valid, after[0].valid, strict=True)
    pd.testing.assert_frame_equal(before[1].table, after[1].table, check_exact=True)
    pd.testing.assert_frame_equal(before[2].table, after[2].table, check_exact=True)
    assert before[2].counts == after[2].counts and before[2].fractions == after[2].fractions
    a, b = layers[1].processing.registrations[0], result.layers[1].processing.registrations[0]
    assert a.application_config == b.application_config and a.diagnostics == b.diagnostics
    assert a.transform.reference_metadata == b.transform.reference_metadata
    assert a.transform.moving_metadata == b.transform.moving_metadata
    if dense:
        assert a.transform.displacement_zyx.dtype == b.transform.displacement_zyx.dtype
        assert a.transform.displacement_zyx.tobytes() == b.transform.displacement_zyx.tobytes()
    else:
        assert a.transform == b.transform


@pytest.mark.parametrize('dtype', ['float32', 'float64'])
def test_dense_half_index_rounding(tmp_path, dtype):
    shape = (1, 3, 5)
    metadata = ImageMetadata('reference')
    field = np.zeros((*shape, 3), dtype=dtype)
    field[..., 2] = .5
    transform = DenseDisplacementTransform(field, shape, shape, metadata, metadata)
    source = np.broadcast_to(np.arange(5, dtype=np.uint16), shape).copy()
    warp = WarpConfig(backend='scipy')
    output = apply_transform(source, transform, config=warp)
    assert output[0, 1].tolist() == [0, 2, 2, 4, 0]
    state = ImageProcessingState(metadata, ({'operation': 'apply_transform', 'config': warp},),
        (RegistrationResult(transform, RegistrationDiagnostics('translation', 'analytic_fixture', TranslationConfig()), warp),),
        terminal_state='applied', reason=None)
    original = layer('round2', output, state=state, channels=('only',))
    path = save(tmp_path / 'half', [original], stage='registered_images')
    equal_image(load_image_checkpoint(path).layers[0].loaded, original.loaded)


def test_partial_and_failed_state(tmp_path):
    layers = registered_layers(1)
    failed = replace(layers[1], loaded=replace(layers[1].loaded, metadata=ImageMetadata('moving')),
        processing=ImageProcessingState(ImageMetadata('moving'),
        terminal_state='failed', reason='estimation failed', attempts=({'outcome': 'failed',
            'requested_method': 'translation', 'actual_method': 'translation', 'config': {},
            'failure': {'type': 'RuntimeError', 'message': 'fixture'}},)))
    result = load_image_checkpoint(save(tmp_path / 'failed', [layers[0], failed], stage='registered_images'))
    assert result.layers[1].processing.attempts == failed.processing.attempts
    with pytest.raises(ValueError, match='not available'):
        result.sequencing_images(require_registered=True)
    path = save(tmp_path / 'partial', [layers[0]], stage='registered_images')
    with pytest.raises(ValueError, match='missing sequencing round'):
        load_image_checkpoint(path).sequencing_images(require_registered=True)


def corrupt_manifest(path, change):
    data = json.loads(path.read_text())
    change(data)
    path.write_text(json.dumps(data))


@pytest.mark.parametrize('mutation', [
    lambda a: a.update(extensions={'invalid': float('nan')}),
    lambda a: a.update(schema_version=2),
    lambda a: a.update(status='omitted', components=[], omission_reason='disabled'),
    lambda a: a.update(config_ref='elsewhere'),
    lambda a: a['payload']['layers'][0].pop('metadata'),
    lambda a: a['payload']['layers'][0].update(axes='YX'),
    lambda a: a['payload']['layers'][0]['image'].update(dtype='<f4'),
    lambda a: a['payload']['layers'][0]['image'].update(artifact_id='other'),
    lambda a: a['payload']['layers'].append(a['payload']['layers'][0]),
    lambda a: a['components'][0].update(path='../images.h5'),
    lambda a: a['payload']['layers'][0]['metadata']['fields'].pop('spatial_unit'),
    lambda a: a['payload']['layers'][1]['metadata']['fields'].update(frame_id='false-frame'),
    lambda a: a['payload']['layers'][1]['processing']['fields'].update(terminal_state='invented'),
    lambda a: a['payload']['layers'][1]['processing']['fields']['registrations']['tuple'][0]['fields']['application_config']['fields'].update(backend='invalid'),
])
def test_malformed_manifest(tmp_path, mutation):
    path = save(tmp_path / 'bad', registered_layers(1), stage='registered_images')
    corrupt_manifest(path, mutation)
    with pytest.raises(ValueError):
        load_image_checkpoint(path)


@pytest.mark.parametrize('mode', ['missing', 'truncated', 'corrupt', 'binding', 'external'])
def test_bad_component(tmp_path, mode):
    path = save(tmp_path / 'bad', registered_layers(1), stage='registered_images')
    component = path.parent / 'images.h5'
    if mode == 'missing':
        component.unlink()
    elif mode == 'truncated':
        component.write_bytes(component.read_bytes()[:100])
    elif mode == 'corrupt':
        data = bytearray(component.read_bytes())
        data[-1] ^= 1
        component.write_bytes(data)
    else:
        with h5py.File(component, 'r+') as handle:
            if mode == 'binding':
                handle['/layers/layer0000/image'].attrs['layer_id'] = 'other'
            else:
                del handle['/layers/layer0000/image']
                handle['/layers/layer0000/image'] = h5py.ExternalLink('/missing.h5', '/image')
        corrupt_manifest(path, lambda a: a['components'][0].update(
            size=component.stat().st_size, sha256=hashlib.sha256(component.read_bytes()).hexdigest()))
    with pytest.raises((ValueError, FileNotFoundError)):
        load_image_checkpoint(path)


def test_incomplete_existing_and_stage(tmp_path):
    path = save(tmp_path / 'ok', registered_layers(1), stage='registered_images')
    with pytest.raises(FileExistsError):
        save(path.parent, registered_layers(1), stage='registered_images')
    with pytest.raises(ValueError, match='stage mismatch'):
        load_image_checkpoint(path, expected_stage='prepared_input')
    with pytest.raises(ValueError, match='manifest checksum'):
        load_image_checkpoint(path, sha256='0' * 64)
    path.unlink()
    with pytest.raises(FileNotFoundError):
        load_image_checkpoint(path.parent)


def test_invalid_save_and_incompatible_consumer(tmp_path):
    layers = registered_layers(1)
    with pytest.raises(ValueError, match='prepared input cannot'):
        save(tmp_path / 'invalid', layers)
    assert not (tmp_path / 'invalid').exists()
    array = np.zeros((1, 4, 5, 4), dtype=np.uint16)
    path = save(tmp_path / 'different', [layer('round10', array), layer('round2', array, metadata=ImageMetadata('other'))])
    with pytest.raises(ValueError, match='geometry/channel'):
        load_image_checkpoint(path).sequencing_images()


def test_FOV_checkpoint_identity_and_no_reapplication(tmp_path):
    from starfinder.dataset import Dataset
    dataset = Dataset(tmp_path, tmp_path / 'out', 'image-contract-v1', 'sample', 'out', ROUNDS, CHANNELS)
    path = save(tmp_path / 'saved', registered_layers(1), stage='registered_images')
    fov = dataset.fov('FOV').load_image_checkpoint(path, require_registered=True)
    assert fov.images['round2'][0, 2, 1, 0] == 9
    assert fov.registration_results['round2'][0].transform.correction_zyx == (0, 1, -1)
    assert fov.image_checkpoint.artifact['stage'] == 'registered_images'
    with pytest.raises(ValueError, match='empty FOV'):
        fov.load_image_checkpoint(path)
    wrong = dataset.fov('other')
    with pytest.raises(ValueError, match='identity mismatch'):
        wrong.load_image_checkpoint(path)
    assert not wrong.images
    dataset.channel_order = tuple(reversed(CHANNELS))
    with pytest.raises(ValueError, match='channel labels'):
        dataset.fov('FOV').load_image_checkpoint(path)


def test_write_failure_leaves_no_manifest(tmp_path, monkeypatch):
    import starfinder.io.checkpoints as module
    def fail(*args, **kwargs):
        raise OSError('injected write failure')
    monkeypatch.setattr(module, '_write_array', fail)
    with pytest.raises(OSError, match='injected'):
        save(tmp_path / 'incomplete', registered_layers(1), stage='registered_images')
    assert not (tmp_path / 'incomplete/artifact.json').exists()
    with pytest.raises(FileNotFoundError):
        load_image_checkpoint(tmp_path / 'incomplete')
