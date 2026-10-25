"""How the fusion reads its sources (W-397): each page decoded at most once, nothing read before the fused
array is computed, the same fused arrays as the fusion of the tiles given as arrays, and a bounded number of tiles
in memory.

Fixture: ``reading_fixtures`` — a synthetic 3 × 3 grid of 3D uint16 tiles (3×192×192 in the default
tier, 16×256×256 in the ``slow`` test), saved as zlib-compressed pages without axis metadata
(``IYX``) and with ZYX axes, with sub-voxel jitter so that each output plane interpolates two pages
of a tile. For each form, ``fuse_images`` runs four times: the stack and the projection, at scale 1 and
at ``scale_yx`` 0.5. The decodes are counted by wrapping tifffile's ``TiffPage.decode`` (every tifffile
reader calls it on the compressed bytes of one segment) and attributed to a page by those bytes.
Each fused array of a file form equals, pixel for pixel, the same fusion of the tiles given as
in-memory arrays (``in_memory_fusions``; an array source is not read through the reader), so the two
file forms also equal each other. Until W-574 the expectation was the SHA-256 of each fused array
computed with the base revision 75fa680 by the W-397 run script ``measure_reads.py``, which also
counted up to 8 decodes of a page for the stack there, and every page decoded before the writer for
the projection. Tiles held: from the reads and releases of the fusion's reader (``_Reader._get``),
in pages, and as whole-tile equivalents (pages held / planes per tile). dask runs with the
synchronous scheduler.
"""
import importlib.util

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

FUSIONS = (("stack", 1.0), ("stack", 0.5), ("projection", 1.0), ("projection", 0.5))


@pytest.fixture
def synchronous():
    with dask.config.set(scheduler="synchronous"):
        yield


def held_counter(monkeypatch):
    """The pages (and projections) the reader holds or is reading, and the largest number at one time."""
    state = {"inside": set(), "peak_pages": 0, "peak_tiles": 0, "reads": 0, "releases": 0}
    real = _fuse._Reader._get

    def counted(self, key, load):
        if key not in self._held:
            state["inside"].add(key)
            state["reads"] += 1
            pages = [k for k in state["inside"] if "-page-" in k]
            state["peak_pages"] = max(state["peak_pages"], len(pages))
            state["peak_tiles"] = max(state["peak_tiles"], len({k.rsplit("-", 4)[1] for k in pages}))
        try:
            return real(self, key, load)
        finally:
            if key not in self._held:
                state["inside"].discard(key)
                state["releases"] += 1
    monkeypatch.setattr(_fuse._Reader, "_get", counted)
    return state


def fusion_config(dimensionality, scale):
    return FusionConfig(images=("reference_merged",), dimensionality=dimensionality, scale_yx=scale)


def in_memory_fusions(tmp_path, shape):
    """The four fusions of the grid's tiles given as in-memory arrays: the fused array of each."""
    placed, grid = rf.grid_placement(shape)
    tiles = {fov: rf.tile(fov, shape) for fov in rf.GRID_IDS}
    fused = {}
    for dimensionality, scale in FUSIONS:
        folder = tmp_path / f"arrays_{dimensionality}_{scale}"
        inputs = [FusionInput(fov, grid, {"reference_merged": tiles[fov]}) for fov in rf.GRID_IDS]
        record = fuse_images(placed, inputs, config=fusion_config(dimensionality, scale), output_dir=folder)
        assert record.status == "written"
        fused[(dimensionality, scale)] = tifffile.imread(folder / record.section["outputs"][0]["path"])
    return fused


@pytest.fixture(scope="module")
def fused_arrays(tmp_path_factory):
    """in_memory_fusions of the default tier's grid of 3×192×192, shared by its two forms."""
    with dask.config.set(scheduler="synchronous"):
        return in_memory_fusions(tmp_path_factory.mktemp("arrays"), (3, 192, 192))


def run_fusions(tmp_path, monkeypatch, form, shape):
    """The four fusions of one form: per fusion the decodes per page, the decodes before the writer started, the
    fused array and the pages held."""
    placed, grid = rf.grid_placement(shape)
    paths = rf.write_tiles(tmp_path / "tiles", form, shape)
    held = held_counter(monkeypatch)
    started = []
    real_write = _ome.write_ome_tiff
    results = {}
    with rf.DecodeCounter(paths) as counter:
        monkeypatch.setattr(_ome, "write_ome_tiff", lambda *a, **k: started.append(len(counter.log)) or
                            real_write(*a, **k))
        for dimensionality, scale in FUSIONS:
            counter.reset()
            started.clear()
            held.update(inside=set(), peak_pages=0, peak_tiles=0)
            config = fusion_config(dimensionality, scale)
            folder = tmp_path / f"{form}_{dimensionality}_{scale}"
            record = fuse_images(placed, [FusionInput(fov, grid, {"reference_merged": paths[fov]}) for fov in rf.GRID_IDS],
                                 config=config, output_dir=folder)
            assert record.status == "written"
            assert ("unknown", -1, -1) not in counter.counts
            per_page = counter.per_page()
            before = started[0]
            image = tifffile.imread(folder / record.section["outputs"][0]["path"])
            results[(dimensionality, scale)] = {
                "pages": per_page, "before": before, "image": image,
                "peak_pages": held["peak_pages"], "peak_tiles": held["peak_tiles"]}
    return results


def check(results, shape, fused):
    n_pages = len(rf.GRID_IDS) * shape[0]
    for key, result in results.items():
        # Each page of each source file is decoded at most once in the fusion, and every page is read.
        assert len(result["pages"]) == n_pages and max(result["pages"].values()) == 1, key
        # No source file is decoded before the fused array is computed (the writer starts); headers only.
        assert result["before"] == 0, key
        # The fused array is the fusion of the tiles given as arrays, pixel for pixel.
        image, expected = result["image"], fused[key]
        assert image.dtype == expected.dtype and np.array_equal(image, expected), key
    for scale in (1.0, 0.5):
        stack = results[("stack", scale)]
        # Fewer than all nine tiles in memory at one time: pages held / planes per tile.
        assert stack["peak_pages"] / shape[0] < len(rf.GRID_IDS), (scale, stack["peak_pages"])
        assert stack["peak_pages"] < n_pages


@needs_library
@pytest.mark.parametrize("form", sorted(rf.FORMS))
def test_each_page_is_decoded_once_in_each_fusion(tmp_path, monkeypatch, fused_arrays, synchronous, form):
    shape = (3, 192, 192)
    results = run_fusions(tmp_path, monkeypatch, form, shape)
    check(results, shape, fused_arrays)
    assert results[("stack", 1.0)]["image"].shape == (3, 538, 538)    # two bands of the writer per plane


@needs_library
@pytest.mark.slow
def test_each_page_is_decoded_once_on_the_16_plane_grid(tmp_path, monkeypatch, synchronous):
    """The resource bound's 3 × 3 grid of 16×256×256: both forms equal the fusion of the arrays and each other, and
    the tiles held reported."""
    shape = (16, 256, 256)
    fused = in_memory_fusions(tmp_path, shape)
    held, images = {}, {}
    for form in sorted(rf.FORMS):
        results = run_fusions(tmp_path / form, monkeypatch, form, shape)
        check(results, shape, fused)
        held[form] = {scale: (results[("stack", scale)]["peak_pages"], results[("stack", scale)]["peak_tiles"])
                      for scale in (1.0, 0.5)}
        images[form] = {key: result["image"] for key, result in results.items()}
    for key in FUSIONS:
        assert np.array_equal(images["pages"][key], images["ZYX"][key]), key
    print("stack, largest number of pages held at one time (pages, tiles with a held page):", held)
    for form, values in held.items():
        for scale, (pages, _) in values.items():
            assert pages <= 4 * len(rf.GRID_IDS), (form, scale, pages)   # at most four pages of each tile


def test_a_source_given_as_an_array_is_read_as_before():
    """An array source is not read through the reader: a dask view of the array itself."""
    from contextlib import ExitStack
    value = np.arange(2 * 3 * 4, dtype=np.uint16).reshape(2, 3, 4)
    with ExitStack() as stack:
        lazy = _fuse._lazy(_fuse._Source("F", "reference_merged", value, None, value.shape, value.dtype), stack)
        assert np.array_equal(lazy.compute(), value)
        plane = _fuse._lazy(_fuse._Source("F", "reference_merged", value[0], None, (1, 3, 4), value.dtype), stack)
        assert plane.shape == (1, 3, 4)
