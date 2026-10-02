"""Projection views (W-235): z and channel projection, and the saved reference merged image.

MATLAB writes ``images/ref_merged`` with SaveSingleStack, one YX page per Z of
``sdata.registration{ref}`` (the channel maximum, or the selected channel), or
one page of its Z maximum when ``maximum_projection`` is true. MATLAB is not
run here; the page layout is mirrored with tifffile.
"""
from dataclasses import replace
from pathlib import Path
import runpy
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

from starfinder.dataset import Dataset, PipelineConfig, RoundState, from_workflow_config
from starfinder.image import ImageMetadata, InvalidImageError
from starfinder.io import load_volume, save_volume
from starfinder.preprocessing import (MinMaxNormalizationConfig, PreprocessingRecipe, ProjectionConfig,
    PreprocessingStep, project_image)

pytestmark = pytest.mark.preprocessing

ROOT = Path(__file__).resolve().parents[3]
CHANNELS = ('a', 'b', 'c', 'd')
METADATA = ImageMetadata('common', spacing_zyx=(2, .5, .5))


def reference_round(dtype, shape=(3, 6, 7, 4), seed=0):
    """Channels differ everywhere, so a channel maximum differs from every channel."""
    high = np.iinfo(dtype).max if np.dtype(dtype).kind == 'u' else 1
    image = np.random.default_rng(seed).uniform(0, high, size=shape)
    return image.astype(dtype)


def fov_with(tmp_path, image):
    ds = Dataset(tmp_path, tmp_path / 'out', 'data', 'sample', 'run',
                 RoundState(['round1', 'round2'], reference_round='round1'), CHANNELS)
    fov = ds.fov('FOV')
    fov.images = {'round1': image, 'round2': image.copy()}
    fov.metadata = {name: METADATA for name in fov.images}
    return fov


def matlab_single_stack(image, path):
    """Mirror SaveSingleStack: append one YX page per Z plane, no axes metadata."""
    planes = image if image.ndim == 3 else image[np.newaxis]
    for plane in planes:
        tifffile.imwrite(path, plane, append=True, metadata=None)


def read_back(path):
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        return series.asarray(), series.axes


# --- ProjectionConfig axis ---------------------------------------------------------

def test_z_and_channel_projections_default_to_maximum():
    image = reference_round(np.uint16)
    assert ProjectionConfig().axis == 'z' and ProjectionConfig().method == 'max'
    z = project_image(image)
    np.testing.assert_array_equal(z, image.max(axis=0, keepdims=True), strict=True)
    assert z.shape == (1, 6, 7, 4)
    channel = project_image(image, config=ProjectionConfig(axis='channel'))
    np.testing.assert_array_equal(channel, image.max(axis=-1), strict=True)
    assert channel.shape == (3, 6, 7)
    summed = project_image(image, config=ProjectionConfig('sum', axis='channel'))
    np.testing.assert_array_equal(summed, image.sum(axis=-1, dtype=np.uint64), strict=True)


@pytest.mark.parametrize('axis', ['y', 'x', 'c', 'Z', 0, None])
def test_invalid_projection_axis_raises(axis):
    with pytest.raises(ValueError, match='projection axis must be z or channel'):
        ProjectionConfig(axis=axis)


def test_channel_projection_requires_zyxc():
    with pytest.raises(InvalidImageError):
        project_image(np.ones((2, 3, 4), np.uint8), config=ProjectionConfig(axis='channel'))


def test_fov_projection_is_along_z_only(tmp_path):
    fov = fov_with(tmp_path, reference_round(np.uint8))
    with pytest.raises(ValueError, match='only a z projection'):
        fov.project_image(config=ProjectionConfig(axis='channel'))


# --- Saved reference merged image --------------------------------------------------

@pytest.mark.parametrize('dtype', [np.uint8, np.uint16, np.float32])
@pytest.mark.parametrize('mode', ['merged', 'single-channel'])
@pytest.mark.parametrize('maximum_projection', [False, True])
def test_ref_merged_content_axes_and_dtype(tmp_path, dtype, mode, maximum_projection):
    image = reference_round(dtype)
    expected = image.max(axis=-1) if mode == 'merged' else image[..., 2]
    if maximum_projection:
        expected = expected.max(axis=0)
    fov = fov_with(tmp_path, image)
    path = fov.save_reference_image(projection=ProjectionConfig() if maximum_projection else None,
                                    reference_image=mode, reference_channel=2)
    assert path == tmp_path / 'out' / 'images' / 'ref_merged' / 'FOV.tif'
    saved, axes = read_back(path)
    np.testing.assert_array_equal(saved, expected, strict=True)
    assert axes == ('YX' if maximum_projection else 'ZYX')
    # The same array written as MATLAB's SaveSingleStack reads back with the same shape.
    matlab_single_stack(expected, tmp_path / 'matlab.tif')
    assert tifffile.imread(tmp_path / 'matlab.tif').shape == saved.shape == expected.shape
    loaded = load_volume(path)
    np.testing.assert_array_equal(loaded.image, expected.reshape((-1,) + expected.shape[-2:]), strict=True)
    assert loaded.metadata == (METADATA.projected(method='max') if maximum_projection else METADATA)


def test_ref_merged_is_the_reference_detection_image(tmp_path):
    raw = reference_round(np.uint16)
    fov = fov_with(tmp_path, raw)
    fov.run(PipelineConfig(preprocessing=PreprocessingRecipe((PreprocessingStep(MinMaxNormalizationConfig('uint8', (0, 255))),))))
    detection = fov.images['round1']
    assert detection.dtype == np.uint8
    saved, _ = read_back(fov.save_reference_image())
    np.testing.assert_array_equal(saved, detection.max(axis=-1), strict=True)


def test_ref_merged_rejects_invalid_views(tmp_path):
    fov = fov_with(tmp_path, reference_round(np.uint8))
    with pytest.raises(ValueError, match='reference_image'):
        fov.save_reference_image(reference_image='sum')
    with pytest.raises(ValueError, match='reference_channel 4 is outside'):
        fov.save_reference_image(reference_image='single-channel', reference_channel=4)
    with pytest.raises(ValueError, match='along z'):
        fov.save_reference_image(projection=ProjectionConfig(axis='channel'))


# --- Workflow adapter: the view the MATLAB script of the same rule saves -------------

def workflow_config(tmp_path, rule='rsf_single_fov', maximum_projection=False, **parameters):
    return dict(root_input_path=str(tmp_path), root_output_path=str(tmp_path / 'out'), dataset_id='data',
                sample_id='sample', output_id='run', n_rounds=2, ref_round='round1', seq_channel_order=list(CHANNELS),
                rotate_angle=0, img_row=6, img_col=7, maximum_projection=maximum_projection,
                rules={rule: {'parameters': parameters}})


SINGLE = {'run': True, 'ref_img': 'single-channel', 'mov_img': 'single-channel', 'ref_channel': 2}


@pytest.mark.parametrize('rule, parameters, view', [
    # rsf_single_fov.m: GlobalRegistration(ref_img from config), else channel maximum.
    ('rsf_single_fov', {'global_registration': SINGLE}, ('single-channel', 2)),
    ('rsf_single_fov', {'global_registration': {'run': True}}, ('merged', 0)),
    ('rsf_single_fov', {}, ('merged', 0)),
    # rsf_single_fov.m: LocalRegistration(ref_layer only) resets it to the channel maximum.
    ('rsf_single_fov', {'global_registration': SINGLE, 'local_registration': {'run': True}}, ('merged', 0)),
    # gr_single_fov_subtile.m passes no ref_img; deep_create_subtile.m sets the channel maximum.
    ('gr_single_fov_subtile', {'global_registration': SINGLE}, ('merged', 0)),
    ('deep_create_subtile', {'global_registration': SINGLE}, ('merged', 0)),
])
def test_adapter_reference_view_follows_the_matlab_script(tmp_path, rule, parameters, view):
    adapted = from_workflow_config(workflow_config(tmp_path, rule, **parameters), rule)
    assert (adapted.reference_image, adapted.reference_channel) == view
    assert adapted.reference_projection is None
    projected = from_workflow_config(workflow_config(tmp_path, rule, maximum_projection=True, **parameters), rule)
    assert projected.reference_projection == ProjectionConfig()


@pytest.mark.parametrize('maximum_projection', [False, True])
def test_rsf_single_fov_script_writes_the_detection_view(tmp_path, maximum_projection):
    raw = reference_round(np.uint16, seed=1)
    for name in ('round1', 'round2'):
        for c, channel in enumerate(CHANNELS):
            save_volume(raw[..., c], tmp_path / 'data' / 'sample' / name / 'FOV' / f'{channel}.tif', metadata=METADATA)
    codebook = tmp_path / 'genes.csv'
    codebook.write_text('gene,barcode\ngene,AAA\n')
    config = workflow_config(tmp_path, maximum_projection=maximum_projection, enhance_contrast={'run': True},
        global_registration=SINGLE, spot_finding={'run': True, 'intensity_estimation': 'adaptive', 'intensity_threshold': .1},
        reads_extraction={'run': True, 'voxel_size': [0, 0, 0]}, reads_filtration={'run': True})
    config.update(starfinder_path=str(ROOT), fov_id_pattern='%s')
    snakemake = SimpleNamespace(config=config, input=['unused', str(codebook)], wildcards=SimpleNamespace(fovID='FOV'))
    runpy.run_path(str(ROOT / 'workflow' / 'scripts' / 'rsf_single_fov.py'), init_globals={'snakemake': snakemake})
    adapted = from_workflow_config(config)
    fov = adapted.dataset.fov('FOV')
    fov.run(replace(adapted.pipeline, registration=None, spot_finding=None, extraction=None, decoding=None, filtering=None))
    expected = fov.images['round1'][..., 2]
    expected = expected.max(axis=0) if maximum_projection else expected
    saved, axes = read_back(fov.paths.ref_merged_tif)
    np.testing.assert_array_equal(saved, expected, strict=True)
    assert saved.dtype == np.uint8 and axes == ('YX' if maximum_projection else 'ZYX')
