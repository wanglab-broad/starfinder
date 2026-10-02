"""ZYXC volumes as OME-TIFF: page layout, OME-XML, exact round trips and the earlier layout."""
from dataclasses import asdict, replace
import json
from xml.etree import ElementTree

import numpy as np
import pytest
import tifffile

from starfinder.dataset import CheckpointConfig
from starfinder.image import ImageMetadata
from starfinder.io import ImageLoadConfig, load_volume, load_volume_zyxc, read_checkpoint, save_volume
from starfinder.io.tiff import METADATA_NAMESPACE

from .test_checkpoints import ROUNDS, dataset, full, resident

pytestmark = pytest.mark.io

OME = '{http://www.openmicroscopy.org/Schemas/OME/2016-06}'
OME_TYPES = {np.uint8: 'uint8', np.uint16: 'uint16', np.float32: 'float', np.float64: 'double'}
METADATA = ImageMetadata('frame', spacing_zyx=(1.5, .25, .125), origin_zyx=(-2.0, 0.1, 3.0),
                         spatial_unit='um')


def volume(shape, dtype):
    image = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
    if np.dtype(dtype).kind == 'f':
        image = image / 7 + 1e-3  # values that are not exact in float32
    return image.astype(dtype)


def old_layout(image, path, metadata=None):
    """A ZYXC file as save_volume wrote it before OME-TIFF: tifffile JSON, pages of X×C."""
    info = {'axes': 'ZYXC'}
    if metadata is not None:
        info['starfinder_metadata'] = asdict(metadata)
    tifffile.imwrite(path, image, photometric='minisblack', metadata=info)


def pixels(path):
    with tifffile.TiffFile(path) as tif:
        assert tif.is_ome
        return ElementTree.fromstring(tif.ome_metadata).find(f'{OME}Image/{OME}Pixels')


@pytest.mark.parametrize('dtype', list(OME_TYPES))
@pytest.mark.parametrize('shape', [(3, 5, 6, 2), (1, 5, 6, 3), (4, 5, 6, 1), (1, 5, 6, 1)])
def test_zyxc_is_ome_tiff_with_yx_pages_and_exact_round_trip(tmp_path, shape, dtype):
    image = volume(shape, dtype)
    original = image.copy()
    path = tmp_path / 'v.ome.tif'
    save_volume(image, path, metadata=METADATA)
    np.testing.assert_array_equal(image, original)
    z, y, x, c = shape
    with tifffile.TiffFile(path) as tif:
        assert len(tif.pages) == z * c
        assert all(page.shape == (y, x) and page.dtype == image.dtype for page in tif.pages)
    element = pixels(path)
    assert {k: element.get(k) for k in ('SizeZ', 'SizeC', 'SizeY', 'SizeX', 'SizeT', 'Type', 'DimensionOrder')} == dict(
        SizeZ=str(z), SizeC=str(c), SizeY=str(y), SizeX=str(x), SizeT='1', Type=OME_TYPES[dtype],
        DimensionOrder='XYCZT')
    labels = tuple(f'ch{i}' for i in range(c))
    loaded = load_volume_zyxc(path, channel_labels=labels)
    assert loaded.image.shape == shape and loaded.image.dtype == image.dtype and loaded.image.flags.c_contiguous
    np.testing.assert_array_equal(loaded.image, image, strict=True)
    assert loaded.metadata == METADATA and loaded.channel_labels == labels
    assert loaded.diagnostics['metadata_source'] == 'stored'
    for index in range(c):
        one = load_volume(path, config=ImageLoadConfig(channel_index=index))
        np.testing.assert_array_equal(one.image, image[..., index], strict=True)
        assert one.metadata == METADATA


def test_metadata_is_an_ome_comment_annotation_and_optional(tmp_path):
    image = volume((2, 3, 4, 2), np.uint16)
    save_volume(image, tmp_path / 'a.ome.tif', metadata=METADATA, compress=True)
    with tifffile.TiffFile(tmp_path / 'a.ome.tif') as tif:
        root = ElementTree.fromstring(tif.ome_metadata)
    notes = [e for e in root.iter(f'{OME}CommentAnnotation') if e.get('Namespace') == METADATA_NAMESPACE]
    assert len(notes) == 1 and ImageMetadata(**json.loads(notes[0].find(f'{OME}Value').text)) == METADATA
    np.testing.assert_array_equal(load_volume_zyxc(tmp_path / 'a.ome.tif').image, image)
    save_volume(image, tmp_path / 'b.ome.tif')
    loaded = load_volume_zyxc(tmp_path / 'b.ome.tif')
    assert loaded.metadata == ImageMetadata(str((tmp_path / 'b.ome.tif').resolve()))
    assert loaded.diagnostics['metadata_source'] == 'unknown'


@pytest.mark.parametrize('dtype', list(OME_TYPES))
@pytest.mark.parametrize('shape', [(3, 5, 6, 2), (1, 5, 6, 1)])
def test_earlier_zyxc_layout_loads_identically(tmp_path, shape, dtype):
    image = volume(shape, dtype)
    old_layout(image, tmp_path / 'old.tif', METADATA)
    with tifffile.TiffFile(tmp_path / 'old.tif') as tif:
        assert not tif.is_ome and tif.shaped_metadata[0]['axes'] == 'ZYXC'
        assert shape[3] == 1 or tif.pages[0].shape == shape[2:]  # tifffile drops singleton C
    save_volume(image, tmp_path / 'new.ome.tif', metadata=METADATA)
    old, new = load_volume_zyxc(tmp_path / 'old.tif'), load_volume_zyxc(tmp_path / 'new.ome.tif')
    np.testing.assert_array_equal(old.image, new.image, strict=True)
    np.testing.assert_array_equal(old.image, image, strict=True)
    assert old.metadata == new.metadata == METADATA and old.diagnostics == new.diagnostics
    for index in range(shape[3]):
        config = ImageLoadConfig(channel_index=index)
        a, b = load_volume(tmp_path / 'old.tif', config=config), load_volume(tmp_path / 'new.ome.tif', config=config)
        np.testing.assert_array_equal(a.image, b.image, strict=True)
        assert a.metadata == b.metadata == METADATA


def test_zyx_files_keep_the_tifffile_layout(tmp_path):
    image = volume((3, 5, 6), np.uint16)
    save_volume(image, tmp_path / 'zyx.tif', metadata=METADATA)
    with tifffile.TiffFile(tmp_path / 'zyx.tif') as tif:
        assert not tif.is_ome
        assert tif.shaped_metadata[0] == {'shape': [3, 5, 6], 'axes': 'ZYX',
                                          'starfinder_metadata': json.loads(json.dumps(asdict(METADATA)))}
    loaded = load_volume(tmp_path / 'zyx.tif')
    np.testing.assert_array_equal(loaded.image, image, strict=True)
    assert loaded.metadata == METADATA


def test_zyxc_ome_rejects_extra_axes(tmp_path):
    tifffile.imwrite(tmp_path / 't.ome.tif', np.zeros((2, 3, 4, 5), np.uint8), ome=True,
                     photometric='minisblack', metadata={'axes': 'TCYX'})
    with pytest.raises(ValueError, match='OME axes'):
        load_volume_zyxc(tmp_path / 't.ome.tif')


def test_registered_checkpoint_is_ome_and_reloads_exactly(tmp_path):
    """Notebook-style: save the registered stage, then reload it into a fresh FOV."""
    ds = dataset(tmp_path)
    saved = resident(ds).run(full())
    for name in ROUNDS:  # float64 geometry and a float image as well as uint16
        saved.metadata[name] = replace(METADATA, frame_id=f'registered/{name}')
    saved.images['round2'] = saved.images['round2'].astype(np.float64) / 3
    directory = saved.save_checkpoint('registered', checkpoints=CheckpointConfig())
    for name in ROUNDS:
        path = directory / 'registered' / f'{name}.ome.tif'
        assert pixels(path).get('SizeC') == '4' and not (directory / 'registered' / f'{name}.tif').exists()
    fov = ds.fov('FOV').load_checkpoint('registered')
    for name in ROUNDS:
        np.testing.assert_array_equal(fov.images[name], saved.images[name], strict=True)
        assert fov.metadata[name] == saved.metadata[name]
    assert fov.registration_results == saved.registration_results


def test_checkpoint_saved_under_the_earlier_tif_name_reloads(tmp_path):
    ds = dataset(tmp_path)
    saved = resident(ds).run(full(), checkpoints=CheckpointConfig())
    registered = saved.paths.checkpoint_dir / 'registered'
    for name in ROUNDS:
        (registered / f'{name}.ome.tif').unlink()
        old_layout(saved.images[name], registered / f'{name}.tif', saved.metadata[name])
    fov = ds.fov('FOV').load_checkpoint('registered')
    for name in ROUNDS:
        np.testing.assert_array_equal(fov.images[name], saved.images[name], strict=True)
        assert fov.metadata[name] == saved.metadata[name]
    assert read_checkpoint(saved.paths.checkpoint_dir, 'registered')['registration_results'] == saved.registration_results
    # overwrite clears the earlier files as well as the current ones
    resident(ds).run(full(), checkpoints=CheckpointConfig(overwrite=True))
    assert sorted(p.name for p in registered.glob('*.tif')) == ['round1.ome.tif', 'round2.ome.tif']


def test_formed_scene_images_are_ome_tiff(tmp_path):
    from starfinder.synthetic import development_scene_preset, generate_formed_scene, save_formed_scene
    book, config = development_scene_preset('clean', size='z1')
    scene = generate_formed_scene(book, config=config)
    save_formed_scene(scene, tmp_path / 'scene')
    for label in scene.round_labels:
        path = tmp_path / 'scene' / 'images' / f'{label}.ome.tif'
        image = scene.rounds[label]
        assert pixels(path).get('SizeC') == str(image.shape[3])
        loaded = load_volume_zyxc(path, channel_labels=scene.channel_labels)
        np.testing.assert_array_equal(loaded.image, image, strict=True)
        assert loaded.metadata == scene.round_metadata[label]


def test_reference_image_keeps_shared_name_with_merged_zyx_content(tmp_path):
    ds = dataset(tmp_path)
    fov = resident(ds)
    path = fov.save_reference_image()
    assert path.name == 'FOV.tif'
    with tifffile.TiffFile(path) as tif:
        assert not tif.is_ome and tif.series[0].axes == 'ZYX'
    loaded = load_volume(path)
    np.testing.assert_array_equal(loaded.image, fov.images['round1'].max(axis=-1), strict=True)
    assert loaded.metadata == fov.metadata['round1']
