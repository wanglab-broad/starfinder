"""ZYXC volumes as OME-TIFF: page layout, OME-XML, exact round trips, the earlier layout and the name-based UUID."""
from dataclasses import asdict, replace
import json
import os
import subprocess
import sys
import time
import uuid
from xml.etree import ElementTree

import numpy as np
import pytest
import tifffile

from starfinder.dataset import CheckpointConfig
from starfinder.image import ImageMetadata
from starfinder.dataset import read_checkpoint
from starfinder.io import ImageLoadConfig, load_volume, load_volume_zyxc, save_volume
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


# --- The OME UUID is a function of the pixel data and the metadata written (W-380) ----------------------------------

SAVE_IN_A_PROCESS = """
import hashlib, json, sys
from pathlib import Path
import numpy as np
from starfinder.image import ImageMetadata
from starfinder.io import save_volume
source, root = Path(sys.argv[1]), Path(sys.argv[2])
for name, metadata in json.loads((source / 'cases.json').read_text()).items():
    save_volume(np.load(source / f'{name}.npy'), root / f'{name}.ome.tif',
                metadata=None if metadata is None else ImageMetadata(**metadata))
print(json.dumps({p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.iterdir())}))
"""


def uuid_cases():
    """name -> (array, metadata): a ZYXC volume with metadata, a one-plane (2D) image without, a ZYX volume on an
    ``.ome.tif`` name, the volume with one voxel changed and the volume with another spacing."""
    base = volume((3, 5, 6, 2), np.uint16)
    changed = base.copy()
    changed[1, 2, 3, 1] += 1
    return {'volume': (base, METADATA), 'plane': (volume((1, 5, 6, 1), np.uint8), None),
            'zyx': (volume((3, 5, 6), np.float32), METADATA), 'voxel': (changed, METADATA),
            'spacing': (base, replace(METADATA, spacing_zyx=(1.5, .25, .25)))}


def ome_uuid(path):
    with tifffile.TiffFile(path) as tif:
        assert tif.is_ome
        return ElementTree.fromstring(tif.ome_metadata).get('UUID')


def starting_commit_save(image, path, metadata):
    """save_volume's tifffile call at the starting commit (59c8d0c), whose UUID tifffile draws from the time."""
    if image.ndim == 3:
        info = {'axes': 'ZYX'}
        if metadata is not None:
            info['starfinder_metadata'] = asdict(metadata)
        tifffile.imwrite(path, image, compression=None, photometric='minisblack', metadata=info)
        return
    info = {'axes': 'ZCYX'}
    if metadata is not None:
        info['CommentAnnotation'] = {'Namespace': METADATA_NAMESPACE, 'Value': json.dumps(asdict(metadata))}
    tifffile.imwrite(path, np.moveaxis(image, 3, 1), ome=True, compression=None, photometric='minisblack',
                     metadata=info)


def loaded(path, image):
    """load_volume of each channel: (array, metadata, diagnostics)."""
    configs = [ImageLoadConfig()] if image.ndim == 3 else [ImageLoadConfig(channel_index=c)
                                                           for c in range(image.shape[3])]
    return [(r.image, r.metadata, r.diagnostics) for r in (load_volume(path, config=c) for c in configs)]


def test_equal_content_gives_equal_bytes_in_two_processes(tmp_path):
    cases = uuid_cases()
    source = tmp_path / 'source'
    source.mkdir()
    for name, (image, _) in cases.items():
        np.save(source / f'{name}.npy', image)
    (source / 'cases.json').write_text(json.dumps({n: None if m is None else asdict(m) for n, (_, m) in cases.items()}))
    digests, started = [], []
    for seed in ('1', '2'):
        if started:  # tifffile's time-based UUID differs between two writes at least a second apart
            time.sleep(max(0.0, 1.0 - (time.monotonic() - started[-1])))
        started.append(time.monotonic())
        digests.append(json.loads(subprocess.check_output(
            [sys.executable, '-c', SAVE_IN_A_PROCESS, str(source), str(tmp_path / seed)],
            env=dict(os.environ, PYTHONHASHSEED=seed), text=True, timeout=300)))
    assert started[1] - started[0] >= 1.0
    # The two processes write identical files.
    assert digests[0] == digests[1] and sorted(digests[0]) == sorted(f'{n}.ome.tif' for n in cases)
    uuids = {n: ome_uuid(tmp_path / '1' / f'{n}.ome.tif') for n in cases}
    assert uuids == {n: ome_uuid(tmp_path / '2' / f'{n}.ome.tif') for n in cases}
    for value in uuids.values():
        assert value.startswith('urn:uuid:') and str(uuid.UUID(value[len('urn:uuid:'):])) == value[len('urn:uuid:'):]
        assert uuid.UUID(value[len('urn:uuid:'):]).version == 5
    # Another voxel or another spacing gives another UUID (each case is distinct).
    assert len(set(uuids.values())) == len(uuids) and len(set(digests[0].values())) == len(cases)
    # Against the starting commit's writer at the same path: the same bytes but the UUID, and the same loads.
    for name, (image, metadata) in cases.items():
        path = tmp_path / 'start' / f'{name}.ome.tif'
        path.parent.mkdir(exist_ok=True)
        starting_commit_save(image, path, metadata)
        before, start_uuid = path.read_bytes(), ome_uuid(path)
        expected = loaded(path, image)
        save_volume(image, path, metadata=metadata)
        assert path.read_bytes() == (tmp_path / '1' / f'{name}.ome.tif').read_bytes()
        assert before.count(start_uuid.encode()) == 1
        assert before.replace(start_uuid.encode(), uuids[name].encode()) == path.read_bytes(), name
        for (a, m, d), (b, n, e) in zip(expected, loaded(path, image), strict=True):
            np.testing.assert_array_equal(a, b, strict=True)
            assert m == n and d == e
        if metadata is not None and image.ndim == 4:
            assert loaded(path, image)[0][1] == metadata


def test_compressed_volume_without_metadata_has_a_content_uuid(tmp_path):
    image = volume((2, 4, 5, 1), np.uint16)
    for name in ('a', 'b'):
        save_volume(image, tmp_path / f'{name}.ome.tif', compress=True)
    assert (tmp_path / 'a.ome.tif').read_bytes() == (tmp_path / 'b.ome.tif').read_bytes()
    save_volume(image, tmp_path / 'raw.ome.tif')
    assert ome_uuid(tmp_path / 'raw.ome.tif') != ome_uuid(tmp_path / 'a.ome.tif')
    np.testing.assert_array_equal(load_volume_zyxc(tmp_path / 'a.ome.tif').image, image, strict=True)
