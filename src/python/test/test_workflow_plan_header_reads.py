"""W-905: the plan's geometry of a tiled FOV takes its shape and dtype from one unbuffered read of one header.

``Graph.image_geometry`` (tiled execution's grid and the tile memory prediction) read the reference file twice: through
``io.tiff._volume_header`` (unbuffered, the bytes tifffile asks for) and through a buffered ``tifffile.TiffFile`` for
the dtype alone, which on an SMB share (block size 1 MiB) read 73 MB of a 151 MB file. Now ``_volume_header_dtype``
gives both from the one unbuffered open. The expected outcomes: each FOV's reference file is opened once, unbuffered,
and no page is decoded; the dtype is the file's own, the preprocessing's ``output_dtype`` overriding it; the bytes
read (``rchar``) of the header read are a fraction of a strip-compressed file's, where a buffered open reads more than
the file; the plan of the ``after-fusion`` variant equals its committed manifest
(``data/expected/test_workflow_plan/after-fusion.json``, which test_workflow_plan.py also compares).
"""
import builtins
import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

from starfinder.workflow import plan

from . import workflow_examples as X

pytestmark = [pytest.mark.workflow, pytest.mark.io, pytest.mark.contract]

CODE = {"version": "0.0.0+test", "version_source": "dist", "git_commit": None, "git_dirty": False}


@pytest.fixture(scope="module")
def fixture_work(tmp_path_factory):
    work = tmp_path_factory.mktemp("tiny") / "work"
    X.golden().write_fixture(work)
    X.write_supplied(work)
    return work


@pytest.fixture(autouse=True)
def fixed_code(monkeypatch):
    import starfinder.workflow._plan as P
    monkeypatch.setattr(P, "package_identity", lambda: dict(CODE))


class Opens:
    """The opens of files under ``root`` through ``builtins.open``: (path, buffering) of each, and the pages decoded."""

    def __init__(self, monkeypatch, root):
        self.root, self.opened, self.pages = Path(root).resolve(), [], 0
        real = builtins.open

        def counting(file, mode="r", buffering=-1, *args, **kwargs):
            if isinstance(file, (str, Path)) and Path(file).resolve().is_relative_to(self.root):
                self.opened.append((Path(file).resolve(), buffering))
            return real(file, mode, buffering, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", counting)
        for name in ("asarray", "decode"):
            for cls in (tifffile.TiffPage, tifffile.TiffFrame):
                self._wrap(monkeypatch, cls, name)

    def _wrap(self, monkeypatch, cls, name):
        real = getattr(cls, name, None)
        if real is None:
            return

        def counted(*args, **kwargs):
            self.pages += 1
            return real(*args, **kwargs)

        monkeypatch.setattr(cls, name, counted)

    def of(self, suffix):
        return [item for item in self.opened if item[0].name == suffix or str(item[0]).endswith(suffix)]


def test_each_fov_s_reference_file_is_opened_once_unbuffered(fixture_work, monkeypatch):
    """The plan of the tiled variant (4 FOVs) opens each FOV's reference file (``round1/<fov>/ch00.ome.tif``, float32) once,
    with buffering 0, and decodes no page."""
    config = X.variants(fixture_work)["after-fusion"]
    root = X.input_root(fixture_work)
    references = {root / "round1" / fov / "ch00.ome.tif" for fov in X.FOVS}
    assert all(p.is_file() for p in references), references
    opens = Opens(monkeypatch, root)
    plan(X.write_config(fixture_work, "after-fusion-reads", config))
    for path in sorted(references):
        assert [b for p, b in opens.opened if p == path.resolve()] == [0], path
    assert opens.pages == 0


def test_the_plan_equals_the_committed_manifest(fixture_work):
    """The manifest of the tiled variant equals its committed expected form, as before the change."""
    config = X.variants(fixture_work)["after-fusion"]
    manifest = plan(X.write_config(fixture_work, "after-fusion", config))
    expected = json.loads((X.EXPECTED / "after-fusion.json").read_text())
    assert X.normalized(manifest, fixture_work) == expected


def _write(path, dtype, shape=(30, 64, 64), **kwargs):
    rng = np.random.default_rng(905)
    tifffile.imwrite(path, rng.integers(0, 200, shape).astype(dtype), metadata=None, photometric="minisblack",
                     **kwargs)
    return path


@pytest.mark.parametrize("dtype", ["uint8", "uint16", "float32"])
def test_the_dtype_is_the_files_and_the_shape_and_flag_are_the_header_reader_s(tmp_path, dtype):
    """``_volume_header_dtype`` gives ``_volume_header``'s result and the series' dtype."""
    from starfinder.io.tiff import _volume_header, _volume_header_dtype
    path = _write(tmp_path / "a.tif", dtype)
    assert _volume_header_dtype(path, floating=True) == ((30, 64, 64), False, np.dtype(dtype))
    assert _volume_header(path, floating=True) == ((30, 64, 64), False)
    shape, stored, found = _volume_header_dtype(path)
    assert (shape, stored, found) == ((30, 64, 64) if dtype != "float32" else None, False, np.dtype(dtype))


def test_a_file_of_several_series_gives_none_and_a_missing_file_raises(tmp_path):
    from starfinder.io.tiff import _volume_header_dtype
    path = tmp_path / "two.tif"
    with tifffile.TiffWriter(path) as writer:
        writer.write(np.zeros((2, 8, 8), np.uint8), metadata=None)
        writer.write(np.zeros((3, 4, 4), np.uint16), metadata=None)
    assert _volume_header_dtype(path) is None
    with pytest.raises(FileNotFoundError):
        _volume_header_dtype(tmp_path / "missing.tif")


def rchar():
    with open("/proc/self/io") as handle:
        return next(int(line.split()[1]) for line in handle if line.startswith("rchar"))


@pytest.mark.skipif(not Path("/proc/self/io").is_file(), reason="needs the Linux I/O counters of /proc/self/io")
def test_the_header_read_with_the_dtype_reads_only_the_header(tmp_path):
    """The bytes read (rchar) for shape and dtype of a file of 30 pages in 64 strips each are a fraction of the
    file's size, where the buffered open the plan used for the dtype read more than the file."""
    from starfinder.io.tiff import _volume_header_dtype
    path = _write(tmp_path / "strips.tif", "uint8", compression="packbits", rowsperstrip=1)
    before = rchar()
    assert _volume_header_dtype(path)[2] == np.dtype("uint8")
    read = rchar() - before
    before = rchar()
    with tifffile.TiffFile(path) as tif:
        assert tif.series[0].dtype == np.dtype("uint8")
    buffered = rchar() - before
    assert read < path.stat().st_size / 4 < buffered


@pytest.mark.parametrize("steps, expected", [
    ([{"method": "white_tophat", "radius_yx": 3}], 4),
    ([{"method": "min_max_normalization", "output_dtype": "uint8", "output_range": [0, 255], "rounding": "truncate"}], 1),
    ([{"method": "min_max_normalization", "output_dtype": "uint16", "output_range": [0, 65535],
       "rounding": "truncate"}], 2)], ids=["file", "uint8", "uint16"])
def test_the_bytes_per_voxel_follow_the_file_and_the_preprocessing_override(fixture_work, monkeypatch, steps, expected):
    """image_geometry's second value is the file's itemsize (float32 here), or the output_dtype of a preprocessing
    step that sets one; the shape is the header's (5 x 36 x 36, no rotation)."""
    from starfinder.workflow import _plan
    config = X.variants(fixture_work)["after-fusion"]
    config["processing"]["preprocessing"] = {"steps": steps}
    seen = {}
    real = _plan.Graph.image_geometry

    def spy(self, fov):
        seen[fov] = real(self, fov)
        return seen[fov]

    monkeypatch.setattr(_plan.Graph, "image_geometry", spy)
    plan(X.write_config(fixture_work, "bytes", config))
    assert seen["FOV_001"] == ((5, 36, 36), expected)
