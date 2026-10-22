"""The name-based OME UUID of the sample-output OME-TIFF writer (W-380): equal content gives equal bytes.

``write_ome_tiff`` is called on a 4 × 200 × 240 uint16 stack (and its plane 0) with one
level, as numpy and as dask arrays; the fused image and the label image are written by two
exports of ``export_folder`` (``fusion_fixtures``), each in its own process with its own
``PYTHONHASHSEED``, at least a second apart (``slow``: about 12 s). The export needs the
``multiview-stitcher`` and ``anndata`` extras and is skipped without them; dask runs with the
synchronous scheduler.
"""
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pytest
import tifffile

from starfinder.image import ImageMetadata
from starfinder.io import load_volume
from starfinder.io._ome import PLACEHOLDER_UUID, read_ome_layout, write_ome_tiff

pytestmark = [pytest.mark.stitching, pytest.mark.contract]
EXTRAS = "the multiview-stitcher and anndata extras (starfinder[multiview-stitcher,anndata]) are not installed"
needs_extras = pytest.mark.skipif(importlib.util.find_spec("multiview_stitcher") is None
                                  or importlib.util.find_spec("anndata") is None, reason=EXTRAS)
SRC = Path(__file__).resolve().parents[1]
METADATA = ImageMetadata("sample:s1", (2.0, 0.5, 0.5), (0.0, 1.0, 2.0), ((1, 0, 0), (0, 1, 0), (0, 0, 1)), "µm")


def ome_uuid(path):
    with tifffile.TiffFile(path) as tif:
        assert tif.is_ome
        return ElementTree.fromstring(tif.ome_metadata).get("UUID")


def stack():
    return np.random.default_rng(380).integers(0, 1000, (4, 200, 240)).astype(np.uint16)


def test_write_ome_tiff_uuid_is_a_function_of_the_content(tmp_path):
    import dask.array as da
    image = stack()
    level = image[:, ::2, ::2].copy()
    annotation = {"sample": "s1", "image": "reference_stain"}
    write_ome_tiff(tmp_path / "a.ome.tif", image, metadata=METADATA, annotation=annotation, levels=(level,))
    write_ome_tiff(tmp_path / "b.ome.tif", da.from_array(image, chunks=(1, 64, 96)), metadata=METADATA,
                   annotation=annotation, levels=(da.from_array(level, chunks=(1, 32, 48)),))
    # The same content, from a numpy or a lazy array, gives the same bytes.
    assert (tmp_path / "a.ome.tif").read_bytes() == (tmp_path / "b.ome.tif").read_bytes()
    value = ome_uuid(tmp_path / "a.ome.tif")
    assert value.startswith("urn:uuid:") and value != PLACEHOLDER_UUID
    assert uuid.UUID(value[len("urn:uuid:"):]).version == 5
    assert PLACEHOLDER_UUID.encode() not in (tmp_path / "a.ome.tif").read_bytes()
    # Another voxel of a level, another annotation or another spacing gives another UUID.
    changed = level.copy()
    changed[2, 10, 20] += 1
    write_ome_tiff(tmp_path / "voxel.ome.tif", image, metadata=METADATA, annotation=annotation, levels=(changed,))
    write_ome_tiff(tmp_path / "annotation.ome.tif", image, metadata=METADATA, annotation=dict(annotation, image="x"),
                   levels=(level,))
    spacing = ImageMetadata("sample:s1", (2.0, 0.5, 0.25), METADATA.origin_zyx, METADATA.direction_zyx, "µm")
    write_ome_tiff(tmp_path / "spacing.ome.tif", image, metadata=spacing, annotation=annotation, levels=(level,))
    values = [ome_uuid(tmp_path / f"{n}.ome.tif") for n in ("a", "voxel", "annotation", "spacing")]
    assert len(set(values)) == 4
    # The layout, the pixels and the metadata read back as before.
    layout = read_ome_layout(tmp_path / "a.ome.tif")
    assert layout["is_bigtiff"] and layout["is_ome"] and layout["tile_yx"] == [512, 512]
    assert layout["level_shapes"] == [[4, 100, 120]]
    restored = load_volume(tmp_path / "a.ome.tif")
    np.testing.assert_array_equal(restored.image, image, strict=True)
    assert restored.metadata == METADATA
    np.testing.assert_array_equal(tifffile.imread(tmp_path / "a.ome.tif", level=1), level, strict=True)


def test_write_ome_tiff_2d(tmp_path):
    plane = stack()[0]
    for name in ("a", "b"):
        write_ome_tiff(tmp_path / f"{name}.ome.tif", plane, metadata=METADATA, annotation={"image": "p"})
    assert (tmp_path / "a.ome.tif").read_bytes() == (tmp_path / "b.ome.tif").read_bytes()
    assert uuid.UUID(ome_uuid(tmp_path / "a.ome.tif")[len("urn:uuid:"):]).version == 5
    np.testing.assert_array_equal(tifffile.imread(tmp_path / "a.ome.tif"), plane, strict=True)


EXPORT_IN_A_PROCESS = """
import hashlib, json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import dask
from starfinder.sample_output import ExportConfig
from test.assembly_fixtures import SAMPLE
from test.fusion_fixtures import export_folder
folder = export_folder(Path(sys.argv[2]))
with dask.config.set(scheduler="synchronous"):
    folder.dataset.export(SAMPLE, placement=folder.placement_dir,
                          config=ExportConfig(route="per_fov", fuse_labels=True),
                          output_dir=folder.export_dir)
print(json.dumps({"root": str(folder.root), "files": {
    p.relative_to(folder.root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in sorted(folder.root.rglob("*.ome.tif"))}}))
"""


@pytest.mark.slow
@needs_extras
def test_two_exports_in_two_processes_write_identical_ome_tiffs(tmp_path):
    outputs, started = [], []
    for seed in ("1", "2"):
        if started:  # tifffile's time-based UUID differs between two writes at least a second apart
            time.sleep(max(0.0, 1.0 - (time.monotonic() - started[-1])))
        started.append(time.monotonic())
        outputs.append(json.loads(subprocess.check_output(
            [sys.executable, "-c", EXPORT_IN_A_PROCESS, str(SRC), str(tmp_path / seed)],
            env=dict(os.environ, PYTHONHASHSEED=seed), text=True, timeout=600)))
    assert started[1] - started[0] >= 1.0
    digests = [output["files"] for output in outputs]
    exported = {"export/sample1/reference_stain_projection.ome.tif", "export/sample1/cell_labels.ome.tif"}
    assert exported <= set(digests[0])
    # The fused image, the label image and each FOV's reference_stain (save_volume) are identical files.
    assert digests[0] == digests[1]
    for name in sorted(digests[0]):
        first, second = (Path(output["root"]) / name for output in outputs)
        assert hashlib.sha256(first.read_bytes()).hexdigest() == digests[0][name]
        assert ome_uuid(first) == ome_uuid(second) and uuid.UUID(ome_uuid(first)[9:]).version == 5
