"""How the fusion reads its sources (W-397): each page decoded at most once, nothing read before the fused
array is computed, the fused arrays of the base revision, and a bounded number of tiles in memory.

Fixture: ``reading_fixtures`` — a synthetic 3 × 3 grid of 3D uint16 tiles (3×192×192 in the default
tier, 16×256×256 in the ``slow`` test), saved as zlib-compressed pages without axis metadata
(``IYX``) and with ZYX axes, with sub-voxel jitter so that each output plane interpolates two pages
of a tile. For each form, ``fuse`` runs four times: the stack and the projection, at scale 1 and at
``scale_yx`` 0.5. The decodes are counted by wrapping tifffile's ``TiffPage.decode`` (every tifffile
reader calls it on the compressed bytes of one segment) and attributed to a page by those bytes.
The pinned SHA-256 of each fused array were computed with the base revision 75fa680 (its reader:
tifffile's Zarr view, and an eager projection) by the W-397 run script ``measure_reads.py``, which
also counted up to 8 decodes of a page for the stack there, and every page decoded before the
writer for the projection. Tiles held: from the reads and releases of the fusion's reader
(``_Reader._get``), in pages, and as whole-tile equivalents (pages held / planes per tile).
dask runs with the synchronous scheduler.
"""
import hashlib
import importlib.util

import dask
import numpy as np
import pytest
import tifffile

from starfinder.io import _ome
from starfinder.sample_output import FusionConfig, FusionInput, _fuse, fuse

from . import reading_fixtures as rf

pytestmark = [pytest.mark.stitching, pytest.mark.validation]

needs_library = pytest.mark.skipif(importlib.util.find_spec("multiview_stitcher") is None,
                                   reason="the multiview-stitcher extra (starfinder[multiview-stitcher]) is not "
                                          "installed")

FUSIONS = (("stack", 1.0), ("stack", 0.5), ("projection", 1.0), ("projection", 0.5))
#: SHA-256 of the fused array's C-order bytes from the base revision 75fa680 (results/reads_base_*.json of W-397).
BASE_SHA256 = {
    (3, 192, 192): {("stack", 1.0): "25ba9ef3b4a8d7df3d44eb1c98ae5580bf88e91e73346bd54482ddc658528a01",
                    ("stack", 0.5): "b890d64ae9e466aa8189778459c6e5827597c8dd7989f676d1944967d7a918ee",
                    ("projection", 1.0): "6a800887f4bd3a96056ae2a0f9311a3634548683ada551ae78300fd1ec4018fa",
                    ("projection", 0.5): "0f67dccb154f66b4648acd8886fafa5b9c92a970ca2d8011143e10219558bbbd"},
    (16, 256, 256): {("stack", 1.0): "6d3d4d2e38f4b973649d9a1611985bc62623f7e4d2252604105f5bb7467f2638",
                     ("stack", 0.5): "60d79c0c1398b49fec4475aa8ead7c503dbaf254f0f117a9f8a5434b3840206d",
                     ("projection", 1.0): "c98929931e35f7f95dac74d58031294ae9be358f4ef33854da6757c425b84f3f",
                     ("projection", 0.5): "4d44532212fb371a6a29a99bedb4fed27f9102c7f08afbad754dbab1a8343fb4"},
}


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


def run_fusions(tmp_path, monkeypatch, form, shape):
    """The four fusions of one form: per fusion the decodes per page, the decodes before the writer started, the
    fused array's SHA-256 and the pages held."""
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
            config = FusionConfig(images=("reference_merged",), dimensionality=dimensionality, scale_yx=scale)
            folder = tmp_path / f"{form}_{dimensionality}_{scale}"
            record = fuse(placed, [FusionInput(fov, grid, {"reference_merged": paths[fov]}) for fov in rf.GRID_IDS],
                          config=config, output_dir=folder)
            assert record.status == "written"
            assert ("unknown", -1, -1) not in counter.counts
            per_page = counter.per_page()
            before = started[0]
            image = tifffile.imread(folder / record.section["outputs"][0]["path"])
            results[(dimensionality, scale)] = {
                "pages": per_page, "before": before, "image": image,
                "sha256": hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest(),
                "peak_pages": held["peak_pages"], "peak_tiles": held["peak_tiles"]}
    return results


def check(results, shape):
    n_pages = len(rf.GRID_IDS) * shape[0]
    for key, result in results.items():
        # Each page of each source file is decoded at most once in the fusion, and every page is read.
        assert len(result["pages"]) == n_pages and max(result["pages"].values()) == 1, key
        # No source file is decoded before the fused array is computed (the writer starts); headers only.
        assert result["before"] == 0, key
        # The fused array is the base revision's, pixel for pixel.
        assert result["sha256"] == BASE_SHA256[shape][key], key
    for scale in (1.0, 0.5):
        stack = results[("stack", scale)]
        # Fewer than all nine tiles in memory at one time: pages held / planes per tile.
        assert stack["peak_pages"] / shape[0] < len(rf.GRID_IDS), (scale, stack["peak_pages"])
        assert stack["peak_pages"] < n_pages


@needs_library
@pytest.mark.parametrize("form", sorted(rf.FORMS))
def test_each_page_is_decoded_once_in_each_fusion(tmp_path, monkeypatch, synchronous, form):
    shape = (3, 192, 192)
    results = run_fusions(tmp_path, monkeypatch, form, shape)
    check(results, shape)
    assert results[("stack", 1.0)]["image"].shape == (3, 538, 538)    # two bands of the writer per plane


@needs_library
@pytest.mark.slow
def test_each_page_is_decoded_once_on_the_16_plane_grid(tmp_path, monkeypatch, synchronous):
    """The resource bound's 3 × 3 grid of 16×256×256: both forms equal, and the tiles held reported."""
    shape = (16, 256, 256)
    held = {}
    for form in sorted(rf.FORMS):
        results = run_fusions(tmp_path / form, monkeypatch, form, shape)
        check(results, shape)
        held[form] = {scale: (results[("stack", scale)]["peak_pages"], results[("stack", scale)]["peak_tiles"])
                      for scale in (1.0, 0.5)}
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
