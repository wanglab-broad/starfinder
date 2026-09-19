"""Bounded W-142 contracts: 4x12x14, four channels and two rounds."""
from dataclasses import replace
from pathlib import Path
import copy
import runpy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from starfinder.dataset import (Dataset, FOV, RoundState, PipelineConfig, ExecutionConfig,
    RegistrationStep, RecoveryConfig, SubtileConfig, from_workflow_config)
from starfinder.image import ImageMetadata, IncompatibleGeometryError
from starfinder.io import ImageLoadConfig, save_volume, export_spots
from starfinder.preprocessing import (MinMaxNormalizationConfig, HistogramMatchingConfig,
    ReconstructionConfig, TophatConfig, ProjectionConfig)
from starfinder.registration import (TranslationConfig, TpsConfig, DemonsConfig, CpdConfig,
    InsufficientLandmarksError, RegistrationBackendUnavailableError, InvalidRegistrationConfigError)
from starfinder.spot_finding import LocalMaximaConfig
from starfinder.barcode import (Codebook, NeighborhoodSumConfig, WtaDecoderConfig,
    ReadFilterConfig, filter_reads)


def dataset(tmp_path):
    ds = Dataset(tmp_path, tmp_path/'out', 'data', 'sample', 'run',
                 RoundState(['round1', 'round2'], reference_round='round1'),
                 ('a', 'b', 'c', 'd'))
    ds.codebook = Codebook(pd.DataFrame({'gene_id': ['gene'], 'color_sequence': ['11']}),
                           ('round1', 'round2'), ds.channel_order)
    return ds


def images():
    a = np.zeros((4, 12, 14, 4), dtype=np.uint16)
    a[1, 4, 5, 0] = 60000
    a[2, 8, 9, 0] = 30000
    return {'round1': a, 'round2': a.copy()}


def resident(ds):
    fov = ds.fov('FOV')
    fov.images = images()
    fov.metadata = {r: ImageMetadata('common') for r in ds.rounds.all_rounds}
    return fov


def complete(**kwargs):
    return PipelineConfig(detection=LocalMaximaConfig('adaptive', .1),
        extraction=NeighborhoodSumConfig((0, 0, 0)), decoding=WtaDecoderConfig(),
        filtering=ReadFilterConfig(), **kwargs)


@pytest.mark.parametrize('options', [
    {}, {'rotation_degrees': 90},
    {'normalization': MinMaxNormalizationConfig('uint8', (0, 255), snr_threshold=2)},
    {'histogram': HistogramMatchingConfig(output_dtype='float32'), 'histogram_reference_channel': 0},
    {'reconstruction': ReconstructionConfig(radius_yx=1)},
    {'tophat': TophatConfig(radius_yx=1)},
    {'projection': ProjectionConfig(method='sum')},
    {'registration': (RegistrationStep(TranslationConfig(), 'single-channel', 'single-channel'),)},
    {'registration': (RegistrationStep(DemonsConfig(iterations=(1,))),)},
    {'reconstruction': ReconstructionConfig(radius_yx=1), 'reconstruction_after_registration': True,
     'registration': (RegistrationStep(TranslationConfig()),)},
    {'normalization': MinMaxNormalizationConfig('uint8', (0, 255)),
     'histogram': HistogramMatchingConfig(), 'reconstruction': ReconstructionConfig(radius_yx=1),
     'tophat': TophatConfig(radius_yx=1), 'projection': ProjectionConfig(),
     'registration': (RegistrationStep(TranslationConfig()),)},
])
def test_scientific_parity(tmp_path, options):
    ds = dataset(tmp_path)
    config = complete(**options)
    batch = resident(ds).run(config)
    stream = resident(ds).run(config, execution=ExecutionConfig('streaming'))
    np.testing.assert_array_equal(batch.intensity_result.values, stream.intensity_result.values)
    pd.testing.assert_frame_equal(batch.spot_result.spots, stream.spot_result.spots)
    pd.testing.assert_frame_equal(batch.decoding_result.table, stream.decoding_result.table)
    pd.testing.assert_frame_equal(batch.filtering_result.table, stream.filtering_result.table)
    assert batch.metadata == stream.metadata
    assert batch.registration_attempts == stream.registration_attempts
    assert set(stream.images) == {'round1'}
    assert set(batch.images) == {'round1', 'round2'}


@pytest.mark.parametrize('config', [PipelineConfig(), PipelineConfig(detection=LocalMaximaConfig()),
    PipelineConfig(detection=LocalMaximaConfig(), extraction=NeighborhoodSumConfig()),
    PipelineConfig(detection=LocalMaximaConfig(), extraction=NeighborhoodSumConfig(), decoding=WtaDecoderConfig())])
def test_disabled_stages(tmp_path, config):
    ds = dataset(tmp_path)
    batch = resident(ds).run(config)
    stream = resident(ds).run(config, execution=ExecutionConfig('streaming', retain_images=True))
    for name in ds.rounds.all_rounds:
        np.testing.assert_array_equal(batch.images[name], stream.images[name])
    for name, enabled in [('spot_result', config.detection), ('intensity_result', config.extraction),
                          ('decoding_result', config.decoding), ('filtering_result', config.filtering)]:
        assert (getattr(batch, name) is not None) == (enabled is not None)
        assert (getattr(stream, name) is not None) == (enabled is not None)


def test_loading_residency_and_label_validation(tmp_path, monkeypatch):
    ds = dataset(tmp_path)
    for round_name, image in images().items():
        for index, channel in enumerate(ds.channel_order):
            path = tmp_path / round_name / 'FOV' / f'{channel}.tif'
            path.parent.mkdir(parents=True, exist_ok=True)
            save_volume(image[..., index], path, metadata=ImageMetadata('common'))
    config = complete(load=ImageLoadConfig(channel_labels=ds.channel_order))
    batch = ds.fov('FOV').run(config)
    stream = ds.fov('FOV').run(config, execution=ExecutionConfig('streaming'))
    np.testing.assert_array_equal(batch.intensity_result.values, stream.intensity_result.values)
    assert set(stream.images) == {'round1'}
    with pytest.raises(ValueError, match='channel'):
        ds.fov('FOV').run(replace(config, load=ImageLoadConfig(channel_labels=tuple(reversed(ds.channel_order)))))


def test_recovery_is_opt_in_and_records_order(tmp_path):
    ds = dataset(tmp_path)
    fov = resident(ds)
    with pytest.raises(InsufficientLandmarksError):
        fov.register(RegistrationStep(TpsConfig()))
    assert len(fov.registration_attempts['round2']) == 1
    fov = resident(ds)
    step = RegistrationStep(TpsConfig(), recovery=RecoveryConfig(
        (InsufficientLandmarksError,), (TpsConfig(min_matches=4), TranslationConfig())))
    fov.register(step)
    attempts = fov.registration_attempts['round2']
    assert [a['actual_method'] for a in attempts] == ['tps', 'tps', 'translation']
    assert [a['outcome'] for a in attempts] == ['failed', 'failed', 'succeeded']
    assert all(a['requested_method'] == 'tps' for a in attempts)
    assert attempts[0]['failure']['type'] == 'InsufficientLandmarksError'
    assert fov.registration_results['round2'][0].diagnostics.method == 'translation'


@pytest.mark.parametrize('error', [IncompatibleGeometryError('geometry'),
    InvalidRegistrationConfigError('config'), RegistrationBackendUnavailableError('dependency')])
def test_invalid_input_and_dependency_never_recover(tmp_path, monkeypatch, error):
    import starfinder.registration as registration
    calls = []
    def fail(*args, **kwargs):
        calls.append(kwargs['config'])
        raise error
    monkeypatch.setattr(registration, 'estimate_transform', fail)
    fov = resident(dataset(tmp_path))
    step = RegistrationStep(TpsConfig(), recovery=RecoveryConfig((InsufficientLandmarksError,), (TranslationConfig(),)))
    with pytest.raises(type(error)):
        fov.register(step)
    assert len(calls) == 1
    assert fov.registration_attempts['round2'][0]['failure']['type'] == type(error).__name__
    with pytest.raises(ValueError):
        RecoveryConfig((ValueError,), (TranslationConfig(),))


def test_signed_high_range_merged_registration(tmp_path):
    fov = resident(dataset(tmp_path))
    fov.images['round1'] = np.full((2, 3, 4, 4), 60000, dtype=np.uint16)
    assert fov._registration_image('round1', 'merged', 0).min() == 240000
    fov.images['round1'] = np.full((2, 3, 4, 4), -3.5, dtype=np.float32)
    assert fov._registration_image('round1', 'merged', 0).min() == -14


def test_export_reordered_subset_empty_and_bad_keys(tmp_path):
    fov = resident(dataset(tmp_path)).run(complete())
    decoded = replace(fov.decoding_result, table=fov.decoding_result.table.iloc[::-1].copy())
    table = decoded.table.copy()
    table.loc[table.index[0], 'call_status'] = 'unmatched'
    table.loc[table.index[0], 'gene_id'] = pd.NA
    filtered = filter_reads(replace(decoded, table=table))
    before = fov.spot_result.spots.copy()
    out = pd.read_csv(export_spots(fov.spot_result, filtered, tmp_path/'subset.csv', accepted_only=True))
    selected = before.merge(filtered.accepted[['spot_id']], on='spot_id', validate='one_to_one')
    np.testing.assert_array_equal(out[['x', 'y', 'z']], selected[['x', 'y', 'z']] + 1)
    pd.testing.assert_frame_equal(before, fov.spot_result.spots)
    for bad in [pd.concat([filtered.table, filtered.table.iloc[:1]]), filtered.table.iloc[:1],
                filtered.table.assign(spot_namespace='foreign')]:
        with pytest.raises(ValueError):
            export_spots(fov.spot_result, replace(filtered, table=bad), tmp_path/'bad.csv')
    all_rejected = replace(filtered, table=filtered.table.assign(accepted=False))
    assert (export_spots(fov.spot_result, all_rejected, tmp_path/'rejected.csv', accepted_only=True)).read_text() == 'x,y,z,gene\n'
    empty = resident(dataset(tmp_path))
    empty.images = {k: np.zeros_like(v) for k, v in empty.images.items()}
    empty.run(complete())
    assert empty.save_spots().read_text() == 'x,y,z,gene\n'
    assert empty.filtering_result.fractions


@pytest.mark.parametrize('shape', [(9, 14), (13, 7)])
def test_rectangular_subtiles_cover_remainder_and_restore_geometry(tmp_path, shape):
    ds = dataset(tmp_path)
    ds.subtile = SubtileConfig(3, .2)
    ds.subtile.compute_windows(*shape)
    coverage = np.zeros(shape, bool)
    for window in ds.subtile.windows:
        coverage[window.to_slice()] = True
    assert coverage.all()
    fov = resident(ds)
    fov.images = {r: np.zeros((2, *shape, 4), dtype=np.uint8) for r in ds.rounds.all_rounds}
    geometry = ImageMetadata('grid', spacing_zyx=(2, 3, 4), origin_zyx=(10, 20, 30),
        direction_zyx=((1, 0, 0), (0, 1, 0), (0, 0, 1)), spatial_unit='um')
    fov.metadata = dict.fromkeys(ds.rounds.all_rounds, geometry)
    fov.create_subtiles()
    loaded = FOV.from_subtile(fov.paths.subtile_dir/'subtile_data_9.npz', ds, 'FOV')
    w = ds.subtile.windows[-1]
    assert loaded.metadata['round1'].origin_zyx == (10, 20 + 3*w.y_start, 30 + 4*w.x_start)
    loaded.find_spots()
    assert loaded.spot_result.spot_namespace == '["data","sample","FOV",9]'
    assert loaded.metadata['round1'] == loaded.metadata['round2']
    with pytest.raises(ValueError, match='identity'):
        FOV.from_subtile(fov.paths.subtile_dir/'subtile_data_9.npz', ds, 'OTHER')


def workflow_config(tmp_path):
    return dict(root_input_path=str(tmp_path), root_output_path=str(tmp_path/'out'),
        dataset_id='data', sample_id='sample', output_id='run', n_rounds=2,
        ref_round='round1', seq_channel_order=['a','b','c','d'], rotate_angle=0,
        img_row=12, img_col=14, rules={})


def test_workflow_translation_rejects_unknowns_and_preserves_effective_settings(tmp_path):
    config = workflow_config(tmp_path)
    params = dict(enhance_contrast={'run': False}, global_registration={'run': False},
        local_registration={'run': True, 'method': 'cpd'}, streaming=True)
    config['rules']['rsf_single_fov'] = {'parameters': params}
    original = copy.deepcopy(config)
    adapted = from_workflow_config(config)
    assert config == original
    assert adapted.pipeline.normalization is None
    assert len(adapted.pipeline.registration) == 1
    cpd = adapted.pipeline.registration[0].config
    assert cpd.detection_noise_sigma == 3 and cpd.grid_spacing_voxels == 32
    assert adapted.execution.mode == 'streaming'
    params['local_registration'] = {'run': True, 'method': 'tps', 'min_matches': 4,
        'grid_spacing': 2, 'tps_smoothing': .5, 'ref_channel': 2, 'boundary_mode': 'nearest'}
    step = from_workflow_config(config).pipeline.registration[0]
    assert step.config.min_matches == 4 and step.config.smoothing == .5
    assert step.reference_channel == 2 and step.warp.boundary_mode == 'nearest'
    params['typo'] = True
    with pytest.raises(ValueError, match='unknown'):
        from_workflow_config(config)
    with pytest.raises(TypeError):
        PipelineConfig(enhance_contrast=True)


def test_maintained_workflow_examples_translate():
    root = Path(__file__).resolve().parents[3]
    for path in [root/'docs/examples/workflow-full.yaml', root/'docs/examples/workflow-minimal.yaml',
                 root/'tests/minimal_config.yaml', root/'tests/tissue_2D_test.yaml']:
        config = yaml.safe_load(path.read_text())
        for rule in config['rules']:
            if rule in ('rsf_single_fov','gr_single_fov_subtile','lrsf_single_fov_subtile','deep_create_subtile','deep_rsf_subtile'):
                from_workflow_config(config, rule)


@pytest.mark.parametrize('streaming', [False, True])
def test_five_workflow_adapters_and_shared_filenames(tmp_path, streaming):
    config = workflow_config(tmp_path)
    root = Path(__file__).resolve().parents[3]
    config['starfinder_path'] = str(root)
    config['fov_id_pattern'] = '%s'
    for name, image in images().items():
        for c, channel in enumerate(config['seq_channel_order']):
            path = tmp_path/'data'/'sample'/name/'FOV'/f'{channel}.tif'
            path.parent.mkdir(parents=True, exist_ok=True)
            save_volume(image[..., c], path, metadata=ImageMetadata('common'))
    codebook = tmp_path/'genes.csv'
    codebook.write_text('gene,barcode\ngene,AAA\n')
    rsf = dict(streaming=streaming, spot_finding={'run': True, 'intensity_estimation': 'adaptive', 'intensity_threshold': .1},
               reads_extraction={'run': True, 'voxel_size': [0, 0, 0]}, reads_filtration={'run': True})
    creation = dict(streaming=streaming, create_subtiles={'run': True, 'sqrt_pieces': 2})
    for rule, parameters in [('rsf_single_fov', rsf), ('gr_single_fov_subtile', creation),
                             ('lrsf_single_fov_subtile', rsf), ('deep_create_subtile', creation),
                             ('deep_rsf_subtile', rsf)]:
        config['rules'][rule] = {'parameters': parameters}
    for rule in ['rsf_single_fov', 'gr_single_fov_subtile', 'lrsf_single_fov_subtile', 'deep_create_subtile', 'deep_rsf_subtile']:
        adapted = from_workflow_config(config, rule)
        fov = adapted.dataset.fov('FOV')
        subtile = fov.paths.subtile_dir/'subtile_data_1.npz'
        snakemake = SimpleNamespace(config=config, input=['unused', str(codebook), str(subtile)],
                                   wildcards=SimpleNamespace(fovID='FOV', n_subtile='1'))
        runpy.run_path(str(root/f'workflow/scripts/{rule}.py'), init_globals={'snakemake': snakemake})
        if rule == 'rsf_single_fov':
            assert fov.paths.signal_csv('goodSpots').name == 'FOV_goodSpots.csv'
            assert list(pd.read_csv(fov.paths.signal_csv('goodSpots'))) == ['x', 'y', 'z', 'gene']
            assert fov.paths.ref_merged_tif.is_file()
        elif 'rsf_subtile' in rule or rule == 'lrsf_single_fov_subtile':
            assert list(pd.read_csv(subtile.parent/'subtile_goodSpots_1.csv')) == ['x', 'y', 'z', 'gene']
        else:
            assert subtile.is_file()
            assert (subtile.parent/'subtile_coords.csv').is_file()


def test_removed_python_surface():
    import starfinder
    import starfinder.dataset as module
    for name in ('STARMapDataset','LayerState','FOVPaths','log_step','ImageArray','ChannelOrder'):
        assert not hasattr(module, name)
        assert not hasattr(starfinder, name)
    for name in ('run_streaming','run_streaming_gr','load_raw_images','enhance_contrast','hist_equalize',
                 'morph_recon','tophat','all_spots','good_spots','global_shifts','local_registered','save_signal','save_log'):
        assert not hasattr(FOV, name)
    assert not hasattr(Dataset, 'from_config')
