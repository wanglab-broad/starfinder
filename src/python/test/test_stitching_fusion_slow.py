"""The ``slow`` parts of FI1, FI2, FI3, FI4, FI6, FI8 and FI9 of docs/stitching-assembly-validation.md.

Fixtures, rebuilt in ``fusion_fixtures``: ``fuse_band`` (six FOVs of 16×256×256 uint8, fusion
grid 16 × 1408 × 256), ``overlap_pair`` (W-354's overlap fixture, union grid 40 × 456, in
uint8, uint16 and float32) and ``position_grid`` (W-354's position fixture, seed 354). Each
is inside the ``slow`` bound of a 3 × 3 grid of 16×256×256 tiles; dask runs with the
synchronous scheduler. Every test needs the ``multiview-stitcher`` extra and is skipped
without it.
"""
import importlib.util
import json
import math

import numpy as np
import pytest
import tifffile

from starfinder.sample_output import FusionConfig, check_export, fuse, fused_grid
from starfinder.io._ome import read_ome_layout

from .fusion_fixtures import OVERLAP_VALUES, fuse_band, overlap_pair, position_grid

EXTRA = "the multiview-stitcher extra (starfinder[multiview-stitcher]) is not installed"
pytestmark = [pytest.mark.stitching, pytest.mark.validation, pytest.mark.slow,
              pytest.mark.skipif(importlib.util.find_spec("multiview_stitcher") is None, reason=EXTRA)]


@pytest.fixture(autouse=True)
def synchronous():
    import dask
    with dask.config.set(scheduler="synchronous"):
        yield


def run(case, folder, **fields):
    record = fuse(case.placement, case.inputs(), config=FusionConfig(**fields), output_dir=folder)
    assert record.status == "written"
    return record.section


def counting(store):
    """Record the keys a Zarr store is asked for (its chunks, in a region read)."""
    keys, get = [], store.get

    async def counted(key, *args, **kwargs):
        keys.append(key)
        return await get(key, *args, **kwargs)

    store.get = counted
    return keys


def output(section, dimensionality, fmt="ome_tiff"):
    return next(o for o in section["outputs"] if o["dimensionality"] == dimensionality and o["format"] == fmt)


# --- fuse_band ---------------------------------------------------------------------------------------------------

def test_fuse_band_grid_layout_and_bands(tmp_path):
    """FI1 (16 × 1408 × 256), FI8 (projection and stack at k = 1 with two SubIFD levels; bands read one at a time)
    and FI6 at k = 2."""
    import zarr
    case = fuse_band()
    assert fused_grid(case.placement, case.grids).shape_zyx == (16, 1408, 256)
    section = run(case, tmp_path / "k1", dimensionality="both", scale_yx=1, pyramid_levels=2)
    assert section["grid"]["shape_zyx"] == [16, 1408, 256]
    for entry in section["outputs"]:
        assert entry["library_grid"] == entry["output_stack_properties"]
        path = tmp_path / "k1" / entry["path"]
        layout = read_ome_layout(path)
        assert layout["is_bigtiff"] and layout["tile_yx"] == [512, 512]
        assert [s[-2:] for s in layout["level_shapes"]] == [[704, 128], [352, 64]]
        with tifffile.TiffFile(path) as tif:
            assert len(tif.pages[0].dataoffsets) == 3      # three 512-row bands of one tile each per plane
            whole = tif.series[0].asarray()
            plane = whole if whole.ndim == 2 else whole[0]
            store = tif.series[0].aszarr(level=0)
            keys = counting(store)
            array = zarr.open_array(store=store, mode="r")
            for band in range(3):
                rows = slice(512 * band, min(512 * (band + 1), 1408))
                keys.clear()
                part = array[rows] if array.ndim == 2 else array[0, rows]
                assert np.array_equal(part, plane[rows])
                assert len({k for k in keys if not k.endswith(("zarr.json", ".zarray", ".zattrs"))}) == 1, keys
            store.close()
    check_export(tmp_path / "k1")
    section = run(case, tmp_path / "k2", dimensionality="stack", scale_yx=0.5)
    entry = output(section, "stack")
    assert entry["shape_zyx"] == [16, 704, 128]
    assert entry["spacing_zyx"] == [1.0, 2.0, 2.0] and entry["origin_zyx"] == [0.0, 0.5, 0.5]
    check_export(tmp_path / "k2")


@pytest.mark.parametrize(("version", "zarr_format"), [("0.5", 3), ("0.4", 2)])
def test_fuse_band_ome_zarr(tmp_path, version, zarr_format):
    """FI9: the library's levels, chunks of (1, 1, 1024, 1024) and (1, 1, 16, 512, 512), one chunk per region read."""
    import zarr
    case = fuse_band()
    section = run(case, tmp_path, dimensionality="both", scale_yx=1, ome_zarr=True, ome_zarr_version=version)
    projection = output(section, "projection", "ome_zarr")
    stack = output(section, "stack", "ome_zarr")
    assert [level["shape"][-2:] for level in projection["levels"]] == [[1408, 256], [704, 128], [352, 128], [176, 128]]
    assert [level["shape"][-3:] for level in stack["levels"]] == [[16, 1408, 256], [16, 704, 128], [16, 352, 128],
                                                                    [16, 176, 128]]
    # The requested chunks (1, 1, 1024, 1024) and (1, 1, 16, 512, 512), as dask clips them to the 256 px of X:
    # the library's writer takes its chunks from the dask chunks (A12).
    assert projection["chunks"] == [1, 1, 1024, 256] and stack["chunks"] == [1, 1, 16, 512, 256]
    for entry, n_chunks in ((projection, 2), (stack, 3)):
        path = tmp_path / entry["path"]
        store = zarr.storage.LocalStore(path, read_only=True)
        keys = counting(store)
        array = zarr.open_array(store=store, path="0", mode="r")
        assert math.prod(math.ceil(n / c) for n, c in zip(array.shape, array.chunks)) == n_chunks
        keys.clear()
        region = array[(0, 0) + (slice(0, 1),) * (array.ndim - 4) + (slice(600, 700), slice(10, 50))]
        chunk_keys = [k for k in keys if not k.endswith(("zarr.json", ".zarray", ".zattrs", ".zgroup"))]
        assert len(set(chunk_keys)) == 1, keys
        assert region.size
        for level in entry["levels"]:
            f = level["factors"]
            assert abs(level["spacing"]["y"] - f["y"] * 1.0) <= 1e-6
    check_export(tmp_path)


# --- overlap_pair --------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", ["uint8", "uint16", "float32"])
def test_overlap_pair_values(tmp_path, dtype):
    """FI2 (F4, F2) and FI3 on the two uncovered corners of the union grid."""
    case = overlap_pair(dtype)
    section = run(case, tmp_path, dimensionality="projection", scale_yx=1)
    entry = output(section, "projection")
    data = tifffile.imread(tmp_path / entry["path"])
    assert data.shape == (40, 456) and data.dtype == np.dtype(dtype) and entry["dtype"] == dtype
    a, b = OVERLAP_VALUES[dtype]
    a, b = np.array(a, dtype), np.array(b, dtype)
    y, x = np.meshgrid(np.arange(40), np.arange(456), indexing="ij")
    in_a = (y <= 31) & (x <= 255)
    in_b = (y >= 8) & (x >= 200)
    assert np.all(data[in_a & ~in_b] == a) and np.all(data[in_b & ~in_a] == b)
    overlap = data[in_a & in_b]
    assert overlap.min() >= a and overlap.max() <= b
    if dtype != "float32":
        assert not np.any(overlap == b)                   # truncation: 203 and 3001 never occur in the overlap
    corners = data[~in_a & ~in_b]
    assert corners.size and np.all(corners == 0) and not np.isnan(corners.astype(float)).any()
    print(json.dumps({"dtype": dtype, "overlap_min": overlap.min().item(), "overlap_max": overlap.max().item()}))
    check_export(tmp_path)


# --- position_grid -------------------------------------------------------------------------------------------------

def test_position_of_a_bright_voxel(tmp_path):
    """FI4 (F5, condition 4) on position_grid: scale 1, then k = 2 and k = 4 by the level1 route."""
    case = position_grid()
    bright = case.extra["bright"]
    found = {}
    for k in (1, 2, 4):
        section = run(case, tmp_path / str(k), dimensionality="stack", scale_yx=1 / k)
        entry = output(section, "stack")
        data = tifffile.imread(tmp_path / str(k) / entry["path"]).astype(float)
        spacing, origin = np.array(entry["spacing_zyx"]), np.array(entry["origin_zyx"])
        for fov, (_, q) in bright.items():
            j = (q - origin) / spacing
            v = np.floor(j + 0.5).astype(int)
            lo = np.maximum(v - 3, 0)
            hi = np.minimum(v + 4, data.shape)
            window = data[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
            peak = np.array(np.unravel_index(int(np.argmax(window)), window.shape)) + lo
            value = window.max()
            position = origin + peak * spacing
            if k == 1:
                assert np.array_equal(peak, v), (fov, peak, v)
                n_lo, n_hi = np.maximum(peak - 1, 0), np.minimum(peak + 2, data.shape)
                block = data[n_lo[0]:n_hi[0], n_lo[1]:n_hi[1], n_lo[2]:n_hi[2]]
                grids = np.meshgrid(*[np.arange(n_lo[a], n_hi[a]) for a in range(3)], indexing="ij")
                centroid = origin + np.array([(block * g).sum() for g in grids]) / block.sum() * spacing
                assert np.linalg.norm(centroid - q) <= 0.001, (fov, centroid, q)
                found[(k, fov)] = float(np.linalg.norm(centroid - q))
            else:
                bound = ((k - 1) / (2 * k) + 0.5) * spacing
                assert 0 < value <= math.floor(1000 / k ** 2), (fov, value)
                assert np.all(np.abs(position[1:] - q[1:]) <= bound[1:]), (fov, position, q, bound)
                found[(k, fov)] = float(np.abs(position[1:] - q[1:]).max())
        check_export(tmp_path / str(k))
    print(json.dumps({f"{k}:{fov}": d for (k, fov), d in found.items()}))
