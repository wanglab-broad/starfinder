"""How the fusion chunks a source file stored in several strips per page (W-432).

Fixture: the 3 × 3 grid of ``reading_fixtures`` with tiles of 4×96×96 uint16, each written
twice as zlib-compressed pages: in strips of 8 rows (12 strips per page, as the LN files are
stored in strips of 23 rows) and in one strip per page. A file whose pages are its Z planes is
one chunk per page, whatever its strips: the source's array and the fused array's graph of the
file in strips count as many chunks and tasks as those of its one-strip twin, tifffile's Zarr
view of such a file is not opened, and the fused images are equal. The guarantees of W-397
(each page decoded once, nothing decoded before the writer, fewer than all tiles' worth of
pages held) are checked on the files in strips. A file whose pages are not its Z planes (one
volumetric page, tiled) is one task for the whole file in the chunks of tifffile's Zarr view,
as before. dask runs with the synchronous scheduler.
"""
import importlib.util
from contextlib import ExitStack

import dask
import numpy as np
import pytest
import tifffile

from starfinder.io import _ome
from starfinder.export import FusionConfig, FusionInput, _fuse, fuse_images

from . import reading_fixtures as rf

pytestmark = [pytest.mark.stitching, pytest.mark.validation]

needs_library = pytest.mark.skipif(importlib.util.find_spec("multiview_stitcher") is None,
                                   reason="the multiview-stitcher extra (starfinder[multiview-stitcher]) is not "
                                          "installed")

SHAPE = (4, 96, 96)
ROWS_PER_STRIP = 8
CONFIG = FusionConfig(images=("reference_merged",), dimensionality="stack", scale_yx=0.5)


@pytest.fixture
def synchronous():
    with dask.config.set(scheduler="synchronous"):
        yield


def write_tiles(root, form, rowsperstrip):
    """FOV → its tile as zlib-compressed pages (``pages``: no metadata, ``ZYX``: ZYX axes) in strips of ``rowsperstrip``
    rows."""
    paths = {}
    for fov in rf.GRID_IDS:
        path = root / f"{form}_{rowsperstrip}" / f"{fov}.tif"
        path.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(path, rf.tile(fov, SHAPE), metadata=None if form == "pages" else {"axes": "ZYX"},
                         photometric="minisblack", compression="zlib", rowsperstrip=rowsperstrip)
        with tifffile.TiffFile(path) as tif:
            assert tif.series[0].axes == rf.FORMS[form] and len(tif.pages) == SHAPE[0]
            assert len(tif.pages[0].dataoffsets) == -(-SHAPE[1] // rowsperstrip)
        paths[fov] = path
    return paths


def inputs(paths, grid):
    return [FusionInput(fov, grid, {"reference_merged": paths[fov]}) for fov in rf.GRID_IDS]


def zarr_views(monkeypatch):
    """The paths tifffile's Zarr view is opened for."""
    opened = []
    real = tifffile.imread
    monkeypatch.setattr(tifffile, "imread", lambda *a, **k: (opened.append(a[0]) if k.get("aszarr") else None) or
                        real(*a, **k))
    return opened


def lazy_graph(paths, placed, grid):
    """(chunks of each source's array, tasks of each source's graph, tasks of the fused array's graph), built as
    ``fuse_images`` builds them."""
    from dask.utils import ensure_dict
    plan = _fuse.check_fusion(placed, inputs(paths, grid), config=CONFIG)
    with ExitStack() as stack:
        reader = _fuse._Reader()
        lazies = {key: _fuse._lazy(source, stack, reader) for key, source in plan.sources.items()}
        fused, _, _ = _fuse._fuse_file(plan, lazies, "reference_merged", "stack", CONFIG, reader, stack)
        return ({key[0]: lazy.chunks for key, lazy in lazies.items()},
                {key[0]: len(ensure_dict(lazy.__dask_graph__())) for key, lazy in lazies.items()},
                len(ensure_dict(fused.data.__dask_graph__())))


@needs_library
@pytest.mark.parametrize("form", sorted(rf.FORMS))
def test_a_file_in_strips_is_one_chunk_per_page(tmp_path, monkeypatch, form):
    placed, grid = rf.grid_placement(SHAPE)
    strips = write_tiles(tmp_path, form, ROWS_PER_STRIP)
    twins = write_tiles(tmp_path, form, SHAPE[1])
    opened = zarr_views(monkeypatch)
    chunks, source_tasks, fused_tasks = lazy_graph(strips, placed, grid)
    strips_opened = list(opened)
    twin_chunks, twin_source_tasks, twin_fused_tasks = lazy_graph(twins, placed, grid)
    n_z, n_y, n_x = SHAPE
    # One chunk per page: the page tasks as they are, not the 12 strips of each page.
    assert all(c == ((1,) * n_z, (n_y,), (n_x,)) for c in chunks.values()), chunks
    assert twin_chunks == chunks
    # As many tasks as the one-strip twin, for each source and for the fused array.
    assert source_tasks == twin_source_tasks, (source_tasks, twin_source_tasks)
    assert fused_tasks == twin_fused_tasks, (fused_tasks, twin_fused_tasks)
    # tifffile's Zarr view of a file read by pages is not opened.
    assert strips_opened == [] and opened == []


@needs_library
@pytest.mark.parametrize("form", sorted(rf.FORMS))
def test_a_file_in_strips_is_read_once_and_fuses_as_its_twin(tmp_path, monkeypatch, synchronous, form):
    """W-397's guarantees on files in strips, and the fused stack of its one-strip twin."""
    placed, grid = rf.grid_placement(SHAPE)
    strips = write_tiles(tmp_path, form, ROWS_PER_STRIP)
    twins = write_tiles(tmp_path, form, SHAPE[1])
    inside, peak = set(), {"pages": 0}
    real_get = _fuse._Reader._get

    def counted(self, key, load):
        if key not in self._held:
            inside.add(key)
            peak["pages"] = max(peak["pages"], sum("-page-" in k for k in inside))
        try:
            return real_get(self, key, load)
        finally:
            if key not in self._held:
                inside.discard(key)
    monkeypatch.setattr(_fuse._Reader, "_get", counted)
    started = []
    real_write = _ome.write_ome_tiff
    images = {}
    with rf.DecodeCounter(strips) as counter:
        monkeypatch.setattr(_ome, "write_ome_tiff", lambda *a, **k: started.append(len(counter.log)) or
                            real_write(*a, **k))
        record = fuse_images(placed, inputs(strips, grid), config=CONFIG, output_dir=tmp_path / "out_strips")
        assert record.status == "written"
        n_pages = len(rf.GRID_IDS) * SHAPE[0]
        # Every strip of every page decoded exactly once, none before the writer starts.
        assert ("unknown", -1, -1) not in counter.counts
        assert len(counter.counts) == n_pages * -(-SHAPE[1] // ROWS_PER_STRIP)
        assert set(counter.counts.values()) == {1} and started == [0]
    # Fewer than all nine tiles' worth of pages held at one time.
    assert peak["pages"] < n_pages, peak
    images["strips"] = tifffile.imread(tmp_path / "out_strips" / record.section["outputs"][0]["path"])
    record = fuse_images(placed, inputs(twins, grid), config=CONFIG, output_dir=tmp_path / "out_twins")
    images["twins"] = tifffile.imread(tmp_path / "out_twins" / record.section["outputs"][0]["path"])
    assert images["strips"].dtype == images["twins"].dtype and np.array_equal(images["strips"], images["twins"])


def test_a_file_read_whole_keeps_the_chunks_of_its_zarr_view(tmp_path, monkeypatch):
    """A file whose pages are not its Z planes (one volumetric page in tiles) is one task for the whole file, in the
    chunks of tifffile's Zarr view, as before W-432."""
    import dask.array as da
    import zarr
    from dask.utils import ensure_dict
    value = rf.tile("T00", SHAPE)
    path = tmp_path / "volumetric.tif"
    tifffile.imwrite(path, value, photometric="minisblack", tile=(SHAPE[0], 32, 32), volumetric=True,
                     metadata={"axes": "ZYX"})
    with tifffile.imread(path, aszarr=True) as store:
        expected = da.from_zarr(zarr.open(store, mode="r")).chunks
    assert expected == ((SHAPE[0],), (32,) * 3, (32,) * 3)
    shape, dtype = _fuse._source_shape(path)
    opened = zarr_views(monkeypatch)
    with ExitStack() as stack:
        reader = _fuse._Reader()
        lazy = _fuse._lazy(_fuse._Source("T00", "reference_merged", path, path, shape, dtype), stack, reader)
        assert opened == [path]
        assert lazy.chunks == expected
        keys = [k for k in ensure_dict(lazy.__dask_graph__()) if isinstance(k, str) and k.startswith(reader._prefix)]
        assert len(keys) == 1 and keys[0].endswith("-file")
        assert np.array_equal(lazy.compute(scheduler="synchronous"), value)
