"""Small, value-preserving ImageJ exports for the saved synthetic example only."""
import json
from pathlib import Path

import numpy as np
import tifffile


def export_images(root, images):
    """Export each saved ZYXC round as an ImageJ ZCYX float32 hyperstack."""
    destination = root / 'inspection'
    destination.mkdir()
    records = []
    for index, (name, loaded) in enumerate(images.items()):
        array = loaded.image
        assert array.dtype == np.float32
        labels = [f'{name} | {channel} | z={z}' for z in range(array.shape[0])
                  for channel in loaded.channel_labels]
        path = destination / f'{name}.tif'
        tifffile.imwrite(path, array.transpose(0, 3, 1, 2), imagej=True,
                         photometric='minisblack', metadata={'axes': 'ZCYX', 'Labels': labels,
                         'Info': f'Sequencing round {name}; channels: {", ".join(loaded.channel_labels)}; physical calibration unknown.'})
        records.append(dict(round=name, hdf5_dataset=f'/layers/layer{index:04d}/image',
                            tiff=path.name, channels=list(loaded.channel_labels), shape_zyxc=list(array.shape)))
    (destination / 'mapping.json').write_text(json.dumps(dict(
        source='../registered/images.h5', hdf5_layout='zyxc', tiff_storage='ZCYX',
        imagej_indices='C and Z are 1-based; canonical coordinates are 0-based',
        calibration='unknown; ImageJ unit grid is pixel indices, not measured physical spacing',
        rounds=records), indent=2)+'\n')
    verify_exports(root, images)


def verify_exports(root, images):
    """Check every TIFF value, metadata label and stored source mapping."""
    mapping = json.loads((root / 'inspection/mapping.json').read_text())
    assert [r['round'] for r in mapping['rounds']] == list(images)
    for record, (name, loaded) in zip(mapping['rounds'], images.items(), strict=True):
        assert record['channels'] == list(loaded.channel_labels)
        assert record['shape_zyxc'] == list(loaded.image.shape)
        with tifffile.TiffFile(root / 'inspection' / record['tiff']) as handle:
            assert handle.is_imagej and handle.imagej_metadata['channels'] == 4
            assert handle.imagej_metadata.get('slices', 1) == loaded.image.shape[0]
            array = handle.asarray().reshape(loaded.image.shape[0], 4, 32, 32).transpose(0, 2, 3, 1)
            assert array.dtype == loaded.image.dtype and array.tobytes() == loaded.image.tobytes()
            assert handle.imagej_metadata['Labels'] == [f'{name} | {c} | z={z}'
                for z in range(loaded.image.shape[0]) for c in loaded.channel_labels]


def export_recipe(directory):
    """Keep the tested Fiji script and a standalone import guide with the data."""
    import html
    import shutil
    script = Path(__file__).with_name('inspect_saved_synthetic_fiji.py')
    shutil.copyfile(script, directory / script.name)
    (directory / 'fiji-import.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8">
<title>Fiji import · saved fixture v2</title><style>body{font:17px/1.6 system-ui;max-width:900px;margin:40px auto;padding:20px}td,th{padding:8px;border:1px solid #aaa}pre{white-space:pre-wrap}</style>
<h1>Inspect the saved rounds in Fiji</h1>
<p>Open the TIFF files under z9/inspection or z1/inspection directly with File → Open. Each is float32, 32×32 XY, four channels, nine or one Z slices, one frame. Plane labels show round, channel and zero-based Z.</p>
<p>For canonical HDF5, choose File → Import → HDF5, open z9/registered/images.h5 or z1/registered/images.h5, select only the three image datasets, choose individual hyperstacks (custom layout), and enter <strong>zyxc</strong>.</p>
<table><tr><th>Dataset</th><th>Sequencing round</th></tr><tr><td>/layers/layer0000/image</td><td>round10</td></tr><tr><td>/layers/layer0001/image</td><td>round2</td></tr><tr><td>/layers/layer0002/image</td><td>round1</td></tr></table>
<p>Channels, in ImageJ C=1–4 order: ch02, ch00, ch03, ch01. Canonical coordinates are zero-based; ImageJ C/Z selectors are one-based. The center slice is Z=5 for z9, Z=1 for z1. Spot centers are XY=(10,10) and (22,22). Set every channel display range to 0–8.</p>
<p><strong>Calibration is unknown.</strong> HDF5_Vibez 1.1.1 assumes 1 µm when its calibration attribute is absent. Do not interpret this as measured spacing. Use Image → Properties to set unit pixel and widths/heights/depths 1, or run the supplied script, which removes the plugin default explicitly.</p>
<p>Open <a href="inspect_saved_synthetic_fiji.py">the verification/import script</a> in Fiji Script Editor (Python). Run it, select this delivery folder and enable Show images. It labels HDF5 planes explicitly, reopens both formats and checks all voxel values, dimensions and mappings. Show images opens and verifies the existing delivery without writing files. Headless verification writes evidence and requires a fresh delivery without an existing fiji-verification.json; preserve the original evidence.</p>
<p>For headless application verification:</p><pre>/path/to/Fiji.app/ImageJ-linux64 --headless --console --mem=1024m \\
  --run /path/to/inspect_saved_synthetic_fiji.py \\
  'delivery="/path/to/delivery",show_images=false'</pre>
<p>The runtime writes fiji-verification.json and center-plane PNGs in inspection/. The packet records its actual versions, mode and checks separately. No setup or update is implicit.</p>
<p><a href="https://github.com/fiji/HDF5_Vibez#load-data-sets">HDF5_Vibez import documentation</a></p>
<details><summary>Exact script</summary><pre>'''+html.escape(script.read_text())+'</pre></details></html>')
