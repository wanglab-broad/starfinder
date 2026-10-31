"""Bounded coordination contracts: 4x12x14, four channels and two rounds."""
from dataclasses import replace
from pathlib import Path
import copy

import numpy as np
import pandas as pd
import pytest
import yaml

from starfinder.dataset import (Dataset, FOV, RoundState, PipelineConfig, ExecutionConfig,
    from_workflow_config)
from starfinder.registration import RecoveryConfig, RegistrationRecipe, RegistrationStep
from starfinder.image import ImageMetadata, IncompatibleGeometryError
from starfinder.readout import export_spots
from starfinder.io import ImageLoadConfig, save_volume
from starfinder.preprocessing import (MinMaxNormalizationConfig, HistogramMatchingConfig,
    ReconstructionConfig, TophatConfig, PreprocessingRecipe, PreprocessingStep)
from starfinder.registration import (TranslationConfig, TpsConfig, DemonsConfig, InsufficientLandmarksError, RegistrationBackendUnavailableError, InvalidRegistrationConfigError,
    RegistrationSignalConfig)
from starfinder.spot_finding import LocalMaximaConfig
from starfinder.readout import (Codebook, NeighborhoodSumConfig, WtaDecoderConfig,
    ReadFilterConfig, filter_reads)

pytestmark = [pytest.mark.integration, pytest.mark.contract]


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


def recipe(*steps, post=()):
    return PreprocessingRecipe(tuple(map(PreprocessingStep, steps)), tuple(map(PreprocessingStep, post)))


def complete(**kwargs):
    return PipelineConfig(spot_finding=LocalMaximaConfig('adaptive', .1),
        extraction=NeighborhoodSumConfig((0, 0, 0)), decoding=WtaDecoderConfig(),
        filtering=ReadFilterConfig(), **kwargs)


@pytest.mark.parametrize('options', [
    {}, {'rotation_degrees': 90},
    {'preprocessing': recipe(MinMaxNormalizationConfig('uint8', (0, 255), snr_threshold=2))},
    {'preprocessing': recipe(HistogramMatchingConfig(reference_channel=0))},
    {'preprocessing': recipe(ReconstructionConfig(radius_yx=1))},
    {'preprocessing': recipe(TophatConfig(radius_yx=1))},
    {'registration': RegistrationRecipe((RegistrationStep(TranslationConfig()),),
                                        signal=RegistrationSignalConfig('channel', 0))},
    {'registration': RegistrationRecipe((RegistrationStep(DemonsConfig(iterations=(1,))),))},
    {'preprocessing': recipe(post=(ReconstructionConfig(radius_yx=1),)),
     'registration': RegistrationRecipe((RegistrationStep(TranslationConfig()),))},
    {'preprocessing': recipe(MinMaxNormalizationConfig('uint8', (0, 255)), HistogramMatchingConfig(),
                             ReconstructionConfig(radius_yx=1), TophatConfig(radius_yx=1)),
     'registration': RegistrationRecipe((RegistrationStep(TranslationConfig()),))},
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


@pytest.mark.parametrize('config', [PipelineConfig(), PipelineConfig(spot_finding=LocalMaximaConfig()),
    PipelineConfig(spot_finding=LocalMaximaConfig(), extraction=NeighborhoodSumConfig()),
    PipelineConfig(spot_finding=LocalMaximaConfig(), extraction=NeighborhoodSumConfig(), decoding=WtaDecoderConfig())])
def test_disabled_stages(tmp_path, config):
    ds = dataset(tmp_path)
    batch = resident(ds).run(config)
    stream = resident(ds).run(config, execution=ExecutionConfig('streaming', retain_images=True))
    for name in ds.rounds.all_rounds:
        np.testing.assert_array_equal(batch.images[name], stream.images[name])
    for name, enabled in [('spot_result', config.spot_finding), ('intensity_result', config.extraction),
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
        fov.register(RegistrationRecipe((RegistrationStep(TpsConfig()),)))
    assert len(fov.registration_attempts['round2']) == 1
    fov = resident(ds)
    step = RegistrationStep(TpsConfig(), recovery=RecoveryConfig(
        (InsufficientLandmarksError,), (TpsConfig(min_matches=4), DemonsConfig(iterations=(1,)))))
    fov.register(RegistrationRecipe((step,)))
    *attempts, application = fov.registration_attempts['round2']
    assert [a['actual_method'] for a in attempts] == ['tps', 'tps', 'demons']
    assert [a['outcome'] for a in attempts] == ['failed', 'failed', 'succeeded']
    assert all(a['requested_method'] == 'tps' for a in attempts)
    assert attempts[0]['failure']['type'] == 'InsufficientLandmarksError'
    assert application['record'] == 'application' and application['outcome'] == 'succeeded'
    assert fov.registration_results['round2'][0].diagnostics.method == 'demons'


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
    step = RegistrationStep(TpsConfig(), recovery=RecoveryConfig((InsufficientLandmarksError,), (DemonsConfig(),)))
    with pytest.raises(type(error)):
        fov.register(RegistrationRecipe((step,)))
    assert len(calls) == 1
    assert fov.registration_attempts['round2'][0]['failure']['type'] == type(error).__name__
    with pytest.raises(ValueError):
        RecoveryConfig((ValueError,), (TranslationConfig(),))


def test_signed_high_range_merged_registration(tmp_path):
    fov = resident(dataset(tmp_path))
    fov.images['round1'] = np.full((2, 3, 4, 4), 60000, dtype=np.uint16)
    assert fov._registration_image('round1', RegistrationSignalConfig('sum'), 'reference').min() == 240000
    fov.images['round1'] = np.full((2, 3, 4, 4), -3.5, dtype=np.float32)
    assert fov._registration_image('round1', RegistrationSignalConfig('sum'), 'reference').min() == -14


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
    np.testing.assert_array_equal(out[['x', 'y', 'z']], selected[['x', 'y', 'z']] + 0)
    pd.testing.assert_frame_equal(before, fov.spot_result.spots)
    for bad in [pd.concat([filtered.table, filtered.table.iloc[:1]]), filtered.table.iloc[:1],
                filtered.table.assign(spot_namespace='foreign')]:
        with pytest.raises(ValueError):
            export_spots(fov.spot_result, replace(filtered, table=bad), tmp_path/'bad.csv')
    all_rejected = replace(filtered, table=filtered.table.assign(accepted=False))
    assert (export_spots(fov.spot_result, all_rejected, tmp_path/'rejected.csv', accepted_only=True)).read_text() == 'spot_namespace,spot_id,z,y,x,gene_id,entry_id,qc_score\n'
    empty = resident(dataset(tmp_path))
    empty.images = {k: np.zeros_like(v) for k, v in empty.images.items()}
    empty.run(complete())
    assert empty.export_spots().read_text() == 'spot_namespace,spot_id,z,y,x,gene_id,entry_id,qc_score\n'
    assert empty.filtering_result.fractions


@pytest.mark.parametrize('shape', [(9, 14), (13, 7)])
def test_rectangular_tiles_cover_remainder_and_restore_geometry(tmp_path, shape):
    # W-735: the tiled entries replace the retired subtile API (SubtileConfig, save_subtiles, from_subtile).
    from starfinder.dataset.tiled import tile_grid
    from starfinder.io import ChunkedCheckpointWriter, read_window
    ds = dataset(tmp_path)
    tiles = tile_grid(shape, 5, 2)
    coverage = np.zeros(shape, int)
    for tile in tiles:
        y0, y1, x0, x1 = tile.core_yx
        coverage[y0:y1, x0:x1] += 1
    assert (coverage == 1).all()
    geometry = ImageMetadata('grid', spacing_zyx=(2, 3, 4), origin_zyx=(10, 20, 30),
        direction_zyx=((1, 0, 0), (0, 1, 0), (0, 0, 1)), spatial_unit='um')
    store = tmp_path/'registered.h5'
    with ChunkedCheckpointWriter(store, identity={'fov_id': 'FOV'}, channel_labels=ds.channel_order) as writer:
        for r in ds.rounds.all_rounds:
            writer.write_round(r, np.zeros((2, *shape, 4), dtype=np.uint8), geometry)
    w = tiles[-1].window_yx
    loaded = resident(ds)
    loaded.tile = tiles[-1].id
    for r in ds.rounds.all_rounds:
        window = read_window(store, r, w, frame_id='grid/tile')
        loaded.images[r], loaded.metadata[r] = window.image, window.metadata
    assert loaded.metadata['round1'].origin_zyx == (10, 20 + 3*w[0], 30 + 4*w[2])
    loaded.find_spots()
    assert loaded.spot_result.spot_namespace == f'["data","sample","FOV","{tiles[-1].id}"]'
    assert loaded.metadata['round1'] == loaded.metadata['round2']


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
    assert adapted.pipeline.preprocessing is None
    assert len(adapted.pipeline.registration.steps) == 1
    cpd = adapted.pipeline.registration.steps[0].config
    assert cpd.detection_noise_sigma == 3 and cpd.grid_spacing_voxels == 32
    assert adapted.execution.mode == 'streaming'
    params['local_registration'] = {'run': True, 'method': 'tps', 'min_matches': 4,
        'grid_spacing': 2, 'tps_smoothing': .5, 'ref_channel': 2, 'boundary_mode': 'nearest'}
    registration = from_workflow_config(config).pipeline.registration
    step = registration.steps[0]
    assert step.config.min_matches == 4 and step.config.smoothing == .5
    # The local ref_img/mov_img default is merged-image (the channel maximum); ref_channel selects nothing then.
    assert registration.signal == RegistrationSignalConfig('max') and registration.warp.boundary_mode == 'nearest'
    params['typo'] = True
    with pytest.raises(ValueError, match='unknown'):
        from_workflow_config(config)
    with pytest.raises(TypeError):
        PipelineConfig(enhance_contrast=True)


def test_maintained_workflow_examples_translate():
    root = Path(__file__).resolve().parents[3]
    for path in [root/'docs/examples/workflow-full.yaml', root/'docs/examples/workflow-minimal.yaml',
                 root/'docs/examples/workflow-recipe-2.yaml',
                 root/'tests/minimal_config.yaml', root/'tests/tissue_2D_test.yaml']:
        config = yaml.safe_load(path.read_text())
        for rule in config['rules']:
            if rule in ('rsf_single_fov','gr_single_fov_subtile','lrsf_single_fov_subtile','deep_create_subtile','deep_rsf_subtile'):
                from_workflow_config(config, rule)


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
