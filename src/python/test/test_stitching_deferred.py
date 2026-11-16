"""W-857: deferred stitching images (``TileLoader``) and the binned registration of ``multiview_stitcher``.

``StitchingInput.image`` takes a ``TileLoader`` (a callable and its dtype; the grid declares the
shape). The method reads each tile of a registered pair once, bins it with the library's own
binning, keeps the binned image only and recomputes each link's ``bbox`` from the full-resolution
geometry with the library's private ``registration._get_overlap_bboxes`` (Jiahao's decision of
2026-11-15), so that the placement equals the library binning each pair itself, bit for bit.

Fixtures: the ``cut_synthetic`` cases of ``stitching_cut_fixtures`` (W-324's recipe; 3 × 3 tiles of
14×120×120, ``z1_3d`` and ``plane_2d`` at Z=1, ``disconnected`` with tile 8 moved 200 px away). The
checks that run the library skip, naming the extra, where ``multiview_stitcher`` is not importable;
they run at one thread with dask's synchronous scheduler, and those that run the method or build the fixtures
are ``slow`` (the first pays the fixture scene, about 5 s), as in ``test_stitching_multiview``.
"""
import gc
import hashlib
import importlib.util
import inspect
import weakref

import numpy as np
import pandas as pd
import pytest

from starfinder.image import ImageMetadata, IncompatibleGeometryError
from starfinder.segmentation import ReferenceGrid
from starfinder.stitching import (BigStitcherConfig, MultiviewStitcherConfig, StitchingInput, TileLoader,
                                  export_bigstitcher, positions_from_grid, stitch)

from .stitching_cut_fixtures import IDENTITY, case

pytestmark = [pytest.mark.stitching, pytest.mark.contract]

EXTRA = "the multiview-stitcher extra (starfinder[multiview-stitcher]) is not installed"
needs_library = pytest.mark.skipif(importlib.util.find_spec("multiview_stitcher") is None, reason=EXTRA)
GRID = ReferenceGrid((2, 16, 16), ImageMetadata("fov", (1.0, 0.5, 0.5), (0.0, 0.0, 0.0), IDENTITY, "µm"), "declared")
PLANE = ReferenceGrid((1, 16, 16), ImageMetadata("fov", (1.0, 0.5, 0.5), (0.0, 0.0, 0.0), IDENTITY, "µm"),
                      "declared")


def loader_of(image):
    return TileLoader(lambda: image, image.dtype)


# --- The loader and StitchingInput (no library) ---------------------------------------------------

def test_a_loader_declares_its_dtype_and_is_not_called_at_construction():
    def refuse():
        raise AssertionError("the loader must not be called before a method needs the pixels")
    item = StitchingInput("a", TileLoader(refuse, ">u2"), GRID, "reference_merged")
    assert item.image.dtype == np.dtype("=u2") and item.image.read is refuse
    with pytest.raises(TypeError, match="read must be callable"):
        TileLoader(np.zeros(3), np.uint8)
    for dtype in (np.bool_, object, "U3"):
        with pytest.raises(TypeError, match="dtype must be a real numeric dtype"):
            TileLoader(refuse, dtype)
    with pytest.raises(TypeError, match=r"FOV 'a' is a callable; pass it as TileLoader\(read, dtype\)"):
        StitchingInput("a", refuse, GRID, "reference_merged")


def test_read_returns_the_checked_tile_and_reads_on_each_call():
    image = np.arange(2 * 16 * 16, dtype=np.uint16).reshape(2, 16, 16)
    calls = []
    item = StitchingInput("a", TileLoader(lambda: calls.append(1) or image, np.uint16), GRID, "reference_merged")
    assert np.array_equal(item.read(), image) and np.array_equal(item.read(), image) and len(calls) == 2
    plane = StitchingInput("p", loader_of(image[0]), PLANE, "reference_merged")
    assert plane.read().shape == (1, 16, 16) and np.array_equal(plane.read()[0], image[0])
    eager = StitchingInput("e", image, GRID, "reference_merged")
    assert eager.read() is eager.image
    assert StitchingInput("m", None, GRID, "reference_merged").read() is None


def test_a_loader_that_raises_or_returns_another_tile_is_named():
    def broken():
        raise OSError("the share went away")
    with pytest.raises(RuntimeError, match=r"reading the stitching image of FOV 'tile7' failed: OSError: the share "
                                           r"went away") as caught:
        StitchingInput("tile7", TileLoader(broken, np.uint8), GRID, "reference_merged").read()
    assert isinstance(caught.value.__cause__, OSError)
    wrong_shape = StitchingInput("b", loader_of(np.zeros((2, 16, 15), np.uint8)), GRID, "reference_merged")
    with pytest.raises(IncompatibleGeometryError, match=r"FOV 'b' has shape \(2, 16, 15\); its grid \(2, 16, 16\)"):
        wrong_shape.read()
    wrong_dtype = StitchingInput("c", TileLoader(lambda: np.zeros((2, 16, 16), np.uint16), np.uint8), GRID,
                                 "reference_merged")
    with pytest.raises(TypeError, match="the loader of FOV 'c' returned uint16; it declares uint8"):
        wrong_dtype.read()


def test_export_bigstitcher_writes_the_same_tiles_from_loaders(tmp_path):
    import tifffile
    table = positions_from_grid(columns=2, rows=2, order="column_major_down_right", overlap=0.25,
                                tile_shape_yx=(16, 16), spacing_yx=(0.5, 0.5), fov_ids=["a", "b", "c", "d"])
    images = {fov: np.full((2, 16, 16), k + 1, np.uint16) for k, fov in enumerate(table.fov_ids)}
    written = {}
    for form, wrap in (("eager", lambda image: image), ("deferred", loader_of)):
        inputs = [StitchingInput(fov, wrap(images[fov]) if fov != "c" else None, GRID, "reference_merged")
                  for fov in table.fov_ids]
        text, view_map = export_bigstitcher(table, inputs, config=BigStitcherConfig(), tile_dir=tmp_path / form / "t",
                                            save_dir=tmp_path / form / "s")
        written[form] = (text.replace(form, ""), view_map,
                         {p.name: p.read_bytes() for p in sorted((tmp_path / form / "t").iterdir())})
        assert tifffile.imread(tmp_path / form / "t" / "tile_0.tif").dtype == np.uint16
    assert written["eager"][0] == written["deferred"][0] and written["eager"][2] == written["deferred"][2]
    pd.testing.assert_frame_equal(written["eager"][1], written["deferred"][1], check_exact=True)


# --- The method on loaders (the library) -----------------------------------------------------------

class Tracked:
    """Loaders that return a fresh copy of each tile and see, at each call, how many earlier copies are alive.

    Only reference counting frees a copy here (no ``gc.collect``), so a copy kept by a reference cycle counts.
    """

    def __init__(self, fixture, raising=None):
        self.alive, self.calls, self.most = [], [], 0
        self.fixture, self.raising = fixture, raising

    def loader(self, tile):
        def read():
            if tile.fov_id == self.raising:
                raise ValueError(f"cannot decode {tile.fov_id}.tif")
            self.most = max(self.most, sum(r() is not None for r in self.alive))
            copy = tile.image.copy()
            self.alive.append(weakref.ref(copy))
            self.calls.append(tile.fov_id)
            return copy
        return TileLoader(read, tile.image.dtype)

    def inputs(self):
        return [StitchingInput(item.fov_id, self.loader(tile), item.grid, item.source)
                for item, tile in zip(self.fixture.inputs(), self.fixture.tiles)]


def run(inputs, fixture, **fields):
    import dask
    with dask.config.set(scheduler="synchronous"):
        return stitch(inputs, fixture.table(), config=MultiviewStitcherConfig(**fields), sample="cut")


def assert_same_placement(a, b):
    pd.testing.assert_frame_equal(a.fovs, b.fovs, check_exact=True)
    pd.testing.assert_frame_equal(a.links, b.links, check_exact=True)
    assert a.record["method_diagnostics"] == b.record["method_diagnostics"]
    assert a.record["inputs"] == b.record["inputs"]


@pytest.mark.slow
@needs_library
@pytest.mark.parametrize("name", ["cut_synthetic", "z1_3d", "plane_2d", "disconnected"])
def test_deferred_and_eager_inputs_give_the_same_placement(name):
    """Bit for bit: translations, origins, statuses, links (shift, quality, kept, reason, residual) and diagnostics,
    on 3D tiles, Z=1 tiles given as ZYX and as YX, and a disconnected tile (tile 8, which stays unplaced)."""
    fixture = case(name)
    eager = run(fixture.inputs(), fixture)
    tracked = Tracked(fixture)
    deferred = run(tracked.inputs(), fixture)
    assert_same_placement(eager, deferred)
    if name in ("z1_3d", "plane_2d"):
        assert eager.record["method_diagnostics"]["registration_dims"] == "yx"
        assert (deferred.fovs.method_translation_z == 0.0).all()
    if name == "disconnected":
        # Tile 8 has no overlap, so it is in no registered pair: its loader is never called.
        assert deferred.fovs.set_index("fov_id").loc["tile8", "status"] == "failed"
        assert "tile8" not in tracked.calls and sorted(tracked.calls) == [f"tile{k}" for k in range(8)]
    else:
        assert sorted(tracked.calls) == sorted(fixture.fov_ids)


@pytest.mark.slow
@needs_library
def test_at_most_one_full_resolution_tile_is_alive_at_a_time():
    """Each loader is called once, in table order, and no earlier tile is alive when the next is read; none is
    alive after stitch returns."""
    fixture = case("cut_synthetic")
    tracked = Tracked(fixture)
    inputs = tracked.inputs()
    result = run(inputs, fixture)
    assert tracked.calls == fixture.fov_ids and tracked.most == 0
    assert all(r() is None for r in tracked.alive)
    del result
    gc.collect()
    assert all(r() is None for r in tracked.alive)


@pytest.mark.slow
@needs_library
def test_a_loader_that_raises_stops_stitch_naming_the_tile():
    fixture = case("cut_synthetic")
    with pytest.raises(RuntimeError, match=r"reading the stitching image of FOV 'tile4' failed: ValueError: "
                                           r"cannot decode tile4\.tif") as caught:
        run(Tracked(fixture, raising="tile4").inputs(), fixture)
    assert isinstance(caught.value.__cause__, ValueError)


# --- The library's binning and its private bbox function -------------------------------------------

def _library_graph(fixture, config):
    """The base revision's calls: full-resolution images, the pruned overlap graph and the library binning each
    pair (registration_binning = binning_zyx)."""
    from multiview_stitcher import mv_graph, registration
    from multiview_stitcher import spatial_image_utils as si_utils
    from starfinder.stitching._multiview import _NOMINAL, _msim
    plane = fixture.inputs()[0].grid.shape_zyx[0] == 1
    axes = (1, 2) if plane else (0, 1, 2)
    dims = [("z", "y", "x")[k] for k in axes]
    nominal = fixture.table().nominal_zyx()
    geometries, msims = [], []
    for i, item in enumerate(fixture.inputs()):
        geometries.append({"dims": dims, "transform_key": _NOMINAL,
                           "scale": {dims[j]: float(fixture.spacing[k]) for j, k in enumerate(axes)},
                           "translation": {dims[j]: float(nominal[i, k]) for j, k in enumerate(axes)}})
        msims.append(_msim(si_utils.get_sim_from_array(item.image[0] if plane else item.image, **geometries[-1])))
    graph = mv_graph.prune_view_adjacency_graph(
        mv_graph.build_view_adjacency_graph_from_msims(msims, transform_key=_NOMINAL, overlap_tolerance=0.0),
        method=config.pruning)
    computed = registration.compute_pairwise_registrations(
        msims, graph, transform_key=_NOMINAL, registration_binning={dims[j]: config.binning_zyx[k]
                                                                    for j, k in enumerate(axes)},
        overlap_tolerance=0.0, pairwise_reg_func=registration.phase_correlation_registration)
    return plane, dims, geometries, msims, graph, computed


@pytest.mark.slow
@needs_library
@pytest.mark.parametrize("name, binning", [("cut_synthetic", (4, 4, 4)), ("cut_synthetic", (2, 3, 2)),
                                           ("plane_2d", (4, 4, 4))])
def test_binning_once_equals_the_library_binning_each_pair(name, binning):
    """Every registered link's transform, quality and bbox equal the library's own, bit for bit: the tile binned once
    with the library's coarsen and registered at binning 1, with the bbox recomputed at full resolution."""
    import dask
    from multiview_stitcher import msi_utils

    from starfinder.stitching._multiview import _binned_msim, _pairwise_registrations
    fixture, config = case(name), MultiviewStitcherConfig(binning_zyx=binning)
    with dask.config.set(scheduler="synchronous"):
        plane, dims, geometries, msims, graph, library = _library_graph(fixture, config)
        bins = {d: b for d, b in zip(dims, binning[3 - len(dims):])}
        binned = [_binned_msim(item.image, plane, bins, geometry)
                  for item, geometry in zip(fixture.inputs(), geometries)]
        ours = _pairwise_registrations(msims, binned, graph, config)
    for k, msim in enumerate(binned):   # the binned tile is 1/prod(binning) of the pixels, computed (not lazy)
        sim = msi_utils.get_sim_from_msim(msim, scale="scale0")
        full = msi_utils.get_sim_from_msim(msims[k], scale="scale0")
        assert isinstance(sim.data, np.ndarray)
        assert [sim.sizes[d] for d in dims] == [full.sizes[d] // bins[d] for d in dims]
    assert len(library.edges) and set(library.edges) == set(ours.edges)
    for edge in library.edges:
        for key in ("transform", "quality", "bbox"):
            a, b = np.asarray(library.edges[edge][key]), np.asarray(ours.edges[edge][key])
            assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), (edge, key)


@needs_library
@pytest.mark.parametrize("dtype, shape, factors", [
    ("uint8", (14, 61, 58), (4, 4, 4)), ("uint16", (9, 33, 40), (2, 3, 2)), ("int16", (5, 20, 21), (1, 1, 4)),
    ("uint32", (6, 17, 18), (3, 2, 2)), ("uint16", (37, 50), (4, 4)), ("uint8", (3, 16, 16), (4, 4, 4))],
    ids=["uint8", "uint16-mixed", "int16-negative", "uint32", "uint16-plane", "fewer-planes-than-a-bin"])
def test_integer_means_equal_the_library_binning(dtype, shape, factors):
    """The window means of an integer tile, from integer sums, equal the library's coarsen(...).mean().astype(dtype) bit
    for bit, on shapes that the windows do not divide (trimmed), with negative values, in 2D, and with no whole window
    along z; a floating-point tile has none (the method then uses the library's reduction)."""
    import dask
    from multiview_stitcher import spatial_image_utils as si_utils

    from starfinder.stitching._multiview import _integer_means
    info = np.iinfo(dtype)
    rng = np.random.default_rng(857)
    image = rng.integers(info.min, int(info.max) + 1, size=shape, dtype=dtype)
    dims = ["z", "y", "x"][3 - len(shape):]
    sim = si_utils.get_sim_from_array(image, dims=dims, transform_key="nominal")
    with dask.config.set(scheduler="synchronous"):
        library = sim.coarsen(dict(zip(dims, factors)), boundary="trim").mean().astype(sim.dtype).compute()
    ours = _integer_means(image, factors)
    assert ours.dtype == image.dtype and ours.shape == tuple(n // f for n, f in zip(shape, factors))
    assert ours.tobytes() == np.ascontiguousarray(np.asarray(library.data).reshape(ours.shape)).tobytes()
    assert _integer_means(image.astype(np.float32), factors) is None


GET_OVERLAP_BBOXES_SHA256 ="2d1261a6c1d15cc802219481ae1e4e658596ce9ffb4084ee15e2bafa6bf03a57"


@needs_library
def test_the_private_bbox_function_is_pinned():
    """The placement depends on multiview-stitcher's private ``registration._get_overlap_bboxes`` (0.1.62): its
    signature, its source (SHA-256 of ``inspect.getsource``; any change of the function fails here, by decision)
    and a known answer. Boxes span voxel centres, (n − 1) × spacing from the origin, grown by the tolerance."""
    from multiview_stitcher import registration
    from multiview_stitcher import spatial_image_utils as si_utils
    function = registration._get_overlap_bboxes
    assert str(inspect.signature(function)) == ("(sim1, sim2, input_transform_key=None, output_transform_key=None, "
                                                "overlap_tolerance=None)")
    assert hashlib.sha256(inspect.getsource(function).encode()).hexdigest() == GET_OVERLAP_BBOXES_SHA256

    def sim(ty, tx):   # 20 × 40 voxels of 0.5 × 0.25 µm: y spans 9.5 µm, x 9.75 µm
        return si_utils.get_sim_from_array(np.zeros((20, 40), np.uint8), dims=["y", "x"], scale={"y": 0.5, "x": 0.25},
                                           translation={"y": ty, "x": tx}, transform_key="nominal")
    for tolerance, lower, upper in ((0.0, (2.0, 7.5), (9.5, 9.75)), (0.25, (1.75, 7.25), (9.75, 10.0))):
        box = function(sim(0.0, 0.0), sim(2.0, 7.5), input_transform_key="nominal", output_transform_key="nominal",
                       overlap_tolerance={"y": tolerance, "x": tolerance})
        for k in (0, 1):   # float64 computed values (halfspace intersections): the tolerance of docs/contributing.md
            np.testing.assert_allclose(box["lowers"][k], lower, rtol=1e-9, atol=1e-12)
            np.testing.assert_allclose(box["uppers"][k], upper, rtol=1e-9, atol=1e-12)
