"""FI3, FI5 to FI11 (their routine parts) and EX5 of docs/stitching-assembly-validation.md: the fused files.

``fuse`` is called alone on ``quad`` (seed 2102), ``zstack``, ``plane_pair`` and
``checker_pair``, built in ``fusion_fixtures``. The files are compared with a direct call of
``multiview_stitcher.fusion.fuse`` on the inputs and arguments the ``fusion`` section
records (FI5, assumption A6), with the formulas of the output levels (FI6), with each
other (FI7), with the OME-TIFF and OME-Zarr layout checks (FI8, FI9) and with the
middle-plane rule (FI11). Every test needs the ``multiview-stitcher`` extra and is
skipped without it; dask runs with the synchronous scheduler (one thread).
"""
import importlib.util
import json
import math

import numpy as np
import pytest
import tifffile

from starfinder.io import load_volume
from starfinder.preprocessing.projection import ProjectionConfig, project_image
from starfinder.export import FusionConfig, check_export, fuse
from starfinder.io._ome import ANNOTATION_NAMESPACE, read_ome_layout

from .fusion_fixtures import checker_pair, make_case, plane_pair, quad, zstack

EXTRA = "the multiview-stitcher extra (starfinder[multiview-stitcher]) is not installed"
pytestmark = [pytest.mark.stitching, pytest.mark.validation,
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


def output(section, dimensionality, fmt="ome_tiff"):
    return next(o for o in section["outputs"] if o["dimensionality"] == dimensionality and o["format"] == fmt)


def read(folder, entry):
    return tifffile.imread(folder / entry["path"])


def direct(case, section, entry):
    """The array a direct call of ``fusion.fuse`` returns for the inputs and arguments the section records."""
    from multiview_stitcher import fusion, msi_utils
    from multiview_stitcher import spatial_image_utils as si_utils
    library, grid = section["library"], section["grid"]
    k = entry["factor_yx"]
    dims = ["y", "x"] if entry["inputs_2d"] else ["z", "y", "x"]
    sims = []
    for fov_entry in section["fovs"]:
        if not fov_entry["included"]:
            continue
        array = case.arrays[fov_entry["fov_id"]]
        if entry["dimensionality"] == "projection":
            method = entry["projection"]
            array = project_image(array, config=ProjectionConfig(method=method, axis="z",
                                                                 output_dtype="float32" if method == "sum" else None))
        if len(dims) == 2:
            array = array[0]
        spacing = dict(zip("zyx", fov_entry["grid"]["metadata"]["spacing_zyx"]))
        origin = dict(zip("zyx", fov_entry["origin_zyx"]))
        sim = si_utils.get_sim_from_array(array, dims=dims, scale={a: spacing[a] for a in dims},
                                          translation={a: origin[a] for a in dims},
                                          transform_key=library["transform_key"])
        if library["route"] == "level1":
            sim = msi_utils.get_sim_from_msim(msi_utils.get_msim_from_sim(sim, scale_factors=[{"y": k, "x": k}]),
                                              scale="scale1")
        sims.append(sim)
    assert library["fusion_func"] == "weighted_average_fusion" and grid["factor_yx"] == k
    fused = fusion.fuse(sims, transform_key=library["transform_key"], fusion_func=fusion.weighted_average_fusion,
                        interpolation_order=library["interpolation_order"],
                        output_stack_properties=entry["output_stack_properties"],
                        output_chunksize=entry["output_chunksize"])
    return np.asarray(fused.data[0, 0].compute())


def uncovered(section, entry):
    """Voxels whose centre lies outside every included FOV's voxel-centre extent on a fused axis (F3)."""
    spacing, origin, shape = entry["spacing_zyx"], entry["origin_zyx"], entry["shape_zyx"]
    axes = (1, 2) if entry["inputs_2d"] else (0, 1, 2)
    s0 = section["grid"]["spacing_zyx"]
    covered = np.zeros(shape, bool)
    for fov in section["fovs"]:
        if not fov["included"]:
            continue
        masks = []
        for a in range(3):
            c = origin[a] + np.arange(shape[a]) * spacing[a]
            lo, hi = fov["origin_zyx"][a], fov["origin_zyx"][a] + (fov["grid"]["shape_zyx"][a] - 1) * s0[a]
            margin = 1e-6 * s0[a]
            masks.append(np.ones(shape[a], bool) if a not in axes else (c >= lo - margin) & (c <= hi + margin))
        covered |= masks[0][:, None, None] & masks[1][None, :, None] & masks[2][None, None, :]
    return ~covered


# --- FI5: library consistency, and EX5 -------------------------------------------------------------------------

@pytest.mark.parametrize("k", [1, 2])
def test_files_equal_a_direct_library_call(tmp_path, k):
    case = quad()
    section = run(case, tmp_path, dimensionality="both", scale_yx=1 / k)
    for dimensionality in ("projection", "stack"):
        entry = output(section, dimensionality)
        written = read(tmp_path, entry)
        expected = direct(case, section, entry)
        assert written.dtype == expected.dtype == np.uint16
        assert np.array_equal(written.reshape(expected.shape), expected)
        library = entry["library_grid"]
        properties = entry["output_stack_properties"]
        assert library == properties                     # FI1: the grid the library records is the grid passed
    assert section["library"]["route"] == ("level1" if k > 1 else "none")
    check_export(tmp_path)


def test_fusion_section_items(tmp_path):
    """EX5: every item of "The fusion section of export.json"."""
    case = quad()
    section = run(case, tmp_path, scale_yx=0.5)
    assert section["status"] == "written" and section["assembly_sha256"] is None
    assert set(section["placement"]) == {"method", "gauge", "review", "approval", "fovs_sha256", "links_sha256"}
    assert section["placement"]["approval"] is not None
    for entry in section["fovs"]:
        assert set(entry) >= {"fov_id", "placement_status", "included", "reason", "origin_zyx", "grid", "images"}
        assert set(entry["grid"]) >= {"shape_zyx", "metadata", "sha256"}
        assert set(entry["images"]["reference_stain"]) >= {"path", "sha256"}
    library = section["library"]
    assert library["multiview_stitcher"] == "0.1.62" and library["fusion_func"] == "weighted_average_fusion"
    assert library["interpolation_order"] == 1 and library["blending_widths_um"] == {"z": 3, "y": 10, "x": 10}
    assert library["route"] == "level1" and library["inputs_2d"] is False
    assert section["config"] == {"images": ["reference_stain"], "dimensionality": "projection", "projection": "max",
                                 "scale_yx": 0.5, "ome_zarr": False, "ome_zarr_version": "0.5", "pyramid_levels": 0}
    assert section["grid"]["shape_zyx"] == [4, 58, 58] and section["grid"]["spacing_zyx"] == [1.0, 1.0, 1.0]
    assert section["grid"]["origin_zyx"] == [0.0, 0.0, 0.0]
    entry = section["outputs"][0]
    assert set(entry) >= {"path", "sha256", "format", "image", "dimensionality", "projection", "factor_yx",
                          "shape_zyx", "dtype", "spacing_zyx", "origin_zyx", "slab_z", "levels", "tile_yx",
                          "compression"}
    assert entry["path"] == "reference_stain_projection.ome.tif" and entry["factor_yx"] == 2
    assert entry["shape_zyx"] == [1, 29, 29] and entry["tile_yx"] == [512, 512] and entry["compression"] == "zlib"
    assert set(section["software"]) and all(section[k] >= 0 for k in ("wall_s", "cpu_s", "peak_rss_kib"))
    assert section["viewer_qualification"] == "deferred"
    exported = json.loads((tmp_path / "export.json").read_text())
    assert exported["viewer_qualification"] == "deferred" and exported["fusion"] == section


def test_one_included_fov_is_its_source(tmp_path):
    """FI1: one included FOV gives its own grid, and at scale 1 the fused stack equals the source image."""
    case = quad()
    first = make_case("quad1", {"Q0": (0.0, 0.0, 0.0)}, {"Q0": case.arrays["Q0"]})
    section = run(first, tmp_path, dimensionality="stack", scale_yx=1)
    assert section["grid"]["shape_zyx"] == [4, 32, 32] and section["grid"]["origin_zyx"] == [0.0, 0.0, 0.0]
    assert np.array_equal(read(tmp_path, output(section, "stack")), case.arrays["Q0"])


# --- FI3: uncovered voxels ---------------------------------------------------------------------------------------

def test_uncovered_voxels_hold_zero(tmp_path):
    case = zstack("different")
    section = run(case, tmp_path, dimensionality="stack", scale_yx=1)
    entry = output(section, "stack")
    data = read(tmp_path, entry)
    assert data.shape == (5, 8, 14)
    mask = uncovered(section, entry)
    assert mask.any() and np.all(data[mask] == 0)
    for dtype in ("float32",):
        float_case = zstack("different", dtype)
        section = run(float_case, tmp_path / dtype, dimensionality="stack", scale_yx=1)
        data = read(tmp_path / dtype, output(section, "stack"))
        assert data.dtype == np.float32 and not np.isnan(data).any() and np.all(data[mask] == 0.0)


def test_plane_pair_values_and_slab(tmp_path):
    """FI3 and FI7 on plane_pair: 50 and 120 inside each FOV, 0 at x 8 and 9; PhysicalSizeZ 3, PositionZ 1."""
    case = plane_pair()
    section = run(case, tmp_path, dimensionality="both", scale_yx=1)
    projection, stack = output(section, "projection"), output(section, "stack")
    p, s = read(tmp_path, projection), read(tmp_path, stack)
    assert p.shape == s.shape == (8, 18) and np.array_equal(p, s)
    assert np.all(p[:, :8] == 50) and np.all(p[:, 10:] == 120) and np.all(p[:, 8:10] == 0)
    assert section["library"]["inputs_2d"] is True and section["grid"]["origins_z"] == {"P0": 0.0, "P1": 2.0}
    for entry in (projection, stack):
        layout = read_ome_layout(tmp_path / entry["path"])
        assert float(layout["pixels"]["PhysicalSizeZ"]) == 3.0 and float(layout["planes"][0]["PositionZ"]) == 1.0
        assert layout["axes"] == "YX"
    check_export(tmp_path)


# --- FI6: the scale factor and the level1 route -------------------------------------------------------------------

@pytest.mark.parametrize("case_name", ["3d", "plane"])
def test_checker_pair_block_means(tmp_path, case_name):
    case = checker_pair(case_name)
    values = {}
    for k in (1, 2, 4):
        section = run(case, tmp_path / str(k), dimensionality="stack", scale_yx=1 / k)
        entry = output(section, "stack")
        grid = section["grid"]
        s, o, n0 = grid["spacing_zyx"], grid["origin_zyx"], grid["shape_zyx"]
        for a in (1, 2):
            assert abs(entry["spacing_zyx"][a] - k * s[a]) <= 1e-9
            assert abs(entry["origin_zyx"][a] - (o[a] + s[a] * (k - 1) / 2)) <= 1e-9
            assert entry["shape_zyx"][a] == math.ceil(n0[a] / k)
        if case_name == "3d":
            assert (entry["spacing_zyx"][0], entry["origin_zyx"][0], entry["shape_zyx"][0]) == (s[0], o[0], n0[0])
        data = read(tmp_path / str(k), entry)
        plane = data if data.ndim == 2 else data[0]
        cy = entry["origin_zyx"][1] + np.arange(entry["shape_zyx"][1]) * entry["spacing_zyx"][1]
        cx = entry["origin_zyx"][2] + np.arange(entry["shape_zyx"][2]) * entry["spacing_zyx"][2]
        a_only = plane[np.ix_((cy >= 0.2) & (cy < 1.8), (cx >= 0.2) & (cx < 7.8))]
        b_only = plane[np.ix_((cy > 13.2) & (cy < 14.4), (cx > 13.2) & (cx < 20.4))]
        if k > 1:
            assert a_only.size and np.all(a_only == 100)
        values[k] = (a_only, sorted(set(b_only.ravel().tolist())))
        check_export(tmp_path / str(k))
    print(json.dumps({"case": case_name, "b_only_values": {k: v[1] for k, v in values.items()}}))


def test_quad_partial_last_block(tmp_path):
    case = quad()
    section = run(case, tmp_path, dimensionality="both", scale_yx=0.25, pyramid_levels=3)
    for dimensionality in ("projection", "stack"):
        entry = output(section, dimensionality)
        assert entry["shape_zyx"][1:] == [15, 15]
        assert np.array_equal(read(tmp_path, entry).reshape(entry["shape_zyx"][0], 15, 15).squeeze(),
                              direct(case, section, entry).squeeze())
    check_export(tmp_path)


# --- FI7: projection, stack and one-plane FOVs ------------------------------------------------------------------

@pytest.mark.parametrize("name", ["equal", "different", "plane"])
def test_projection_and_stack(tmp_path, name):
    case = zstack(name)
    section = run(case, tmp_path, dimensionality="both", scale_yx=0.5)
    projection, stack = output(section, "projection"), output(section, "stack")
    for key in ("spacing_zyx", "origin_zyx", "shape_zyx"):
        assert projection[key][1:] == stack[key][1:]
    z_lo, z_hi = projection["slab_z"]
    S_z, O_z = projection["spacing_zyx"][0], projection["origin_zyx"][0]
    q_z = np.linspace(z_lo, z_hi, 50, endpoint=False)
    assert np.all(np.floor((q_z - O_z) / S_z + 0.5) == 0)
    p = read(tmp_path, projection)
    assert np.array_equal(p, direct(case, section, projection))
    grid = section["grid"]
    if name == "plane":
        assert section["library"]["inputs_2d"] is True and grid["origins_z"] == {"Z0": 0.0, "Z1": 0.0}
        assert np.array_equal(p, read(tmp_path, stack))
        assert stack["spacing_zyx"][0] == z_hi - z_lo and stack["origin_zyx"][0] == (z_lo + z_hi) / 2
    else:
        n0, s, o = grid["shape_zyx"][0], grid["spacing_zyx"][0], grid["origin_zyx"][0]
        assert abs(S_z - n0 * s) <= 1e-9 and abs(O_z - (o + s * (n0 - 1) / 2)) <= 1e-9
    check_export(tmp_path)


# --- FI8: OME-TIFF layout -----------------------------------------------------------------------------------------

@pytest.mark.parametrize(("k", "levels", "shapes"), [(1, 2, [[29, 29], [14, 14]]), (4, 3, [[7, 7], [3, 3], [1, 1]])])
def test_ome_tiff_layout(tmp_path, k, levels, shapes):
    from multiview_stitcher import msi_utils
    from multiview_stitcher import spatial_image_utils as si_utils
    case = quad()
    section = run(case, tmp_path, dimensionality="both", scale_yx=1 / k, pyramid_levels=levels)
    one_voxel_axes = []
    for entry in section["outputs"]:
        layout = read_ome_layout(tmp_path / entry["path"])
        stack = entry["dimensionality"] == "stack"
        assert layout["is_bigtiff"] and layout["is_ome"] and layout["n_series"] == 1 and layout["channels"] == 1
        assert layout["axes"] == ("ZYX" if stack else "YX") and layout["dtype"] == "uint16"
        assert layout["pixels"]["Type"] == "uint16" and layout["tile_yx"] == [512, 512]
        assert layout["compression"] in ("adobe_deflate", "deflate")
        assert [s[-2:] for s in layout["level_shapes"]] == shapes
        spacing, origin = entry["spacing_zyx"], entry["origin_zyx"]
        physical = [float(layout["pixels"][f"PhysicalSize{a}"]) for a in "ZYX"]
        assert np.allclose(physical, spacing, rtol=0, atol=1e-9)
        for i, plane in enumerate(layout["planes"]):
            assert abs(float(plane["PositionZ"]) - (origin[0] + i * spacing[0])) <= 1e-9
            assert abs(float(plane["PositionY"]) - origin[1]) <= 1e-9
            assert abs(float(plane["PositionX"]) - origin[2]) <= 1e-9
        restored = load_volume(tmp_path / entry["path"])
        assert restored.diagnostics["metadata_source"] == "stored"
        assert np.allclose(restored.metadata.spacing_zyx, spacing, rtol=0, atol=1e-9)
        assert np.allclose(restored.metadata.origin_zyx, origin, rtol=0, atol=1e-9)
        annotation = layout["annotations"][ANNOTATION_NAMESPACE]
        assert annotation["sample"] == "quad" and annotation["image"] == "reference_stain"
        assert annotation["dimensionality"] == entry["dimensionality"] and annotation["factor_yx"] == k
        assert annotation["projection"] == ("max" if not stack else None)
        assert annotation["placement_fovs_sha256"] == section["placement"]["fovs_sha256"]
        # The levels are the library's multiscale of the output level (A7): shapes, geometry and values.
        data = tifffile.imread(tmp_path / entry["path"])
        dims = ["z", "y", "x"] if stack else ["y", "x"]
        sim = si_utils.get_sim_from_array(data, dims=dims, scale=dict(zip(dims, spacing[-len(dims):])),
                                          translation=dict(zip(dims, origin[-len(dims):])), transform_key="placed")
        msim = msi_utils.get_msim_from_sim(sim, scale_factors=[{"y": 2, "x": 2}] * levels)
        with tifffile.TiffFile(tmp_path / entry["path"]) as tif:
            for level, recorded in enumerate(entry["levels"], start=1):
                library = msi_utils.get_sim_from_msim(msim, scale=f"scale{level}")
                lib_spacing = si_utils.get_spacing_from_sim(library)
                lib_origin = si_utils.get_origin_from_sim(library)
                shape = tif.series[0].levels[level].shape
                assert list(shape)[-2:] == recorded["shape_zyx"][-2:]
                for i in range(3):
                    # The record carries the page's formula on every axis, one-voxel axes included.
                    f = recorded["factors_zyx"][i]
                    assert f == (1 if i == 0 else 2 ** level)
                    assert abs(recorded["spacing_zyx"][i] - f * spacing[i]) <= 1e-9
                    assert abs(recorded["origin_zyx"][i] - (origin[i] + spacing[i] * (f - 1) / 2)) <= 1e-9
                for a in dims:
                    i = "zyx".index(a)
                    assert abs(lib_origin[a] - recorded["origin_zyx"][i]) <= 1e-9
                    if library.sizes[a] > 1:
                        assert abs(lib_spacing[a] - recorded["spacing_zyx"][i]) <= 1e-9
                    else:
                        # A7 at a one-voxel axis: the library derives spacing from coordinates and reports 1.0,
                        # not the page's f·S; asserted so that a change of the library shows.
                        one_voxel_axes.append((entry["path"], level, a))
                        assert lib_spacing[a] == 1.0 and recorded["spacing_zyx"][i] != 1.0
                assert np.array_equal(tif.series[0].levels[level].asarray(),
                                      np.asarray(library.data[0, 0].compute()).reshape(
                                          tif.series[0].levels[level].shape))
    # quad at k = 4 with three levels ends at 1 × 1 (15 → 7 → 3 → 1); at k = 1 no level has a one-voxel axis.
    assert len(one_voxel_axes) == (4 if k == 4 else 0), one_voxel_axes
    check_export(tmp_path)


# --- FI9: standalone OME-Zarr ---------------------------------------------------------------------------------------

@pytest.mark.parametrize(("version", "zarr_format"), [("0.5", 3), ("0.4", 2)])
def test_standalone_ome_zarr(tmp_path, version, zarr_format):
    import zarr
    case = quad()
    section = run(case, tmp_path, dimensionality="both", scale_yx=1, ome_zarr=True, ome_zarr_version=version)
    # The requested chunks (1, 1, 1024, 1024) and (1, 1, 16, 512, 512), clipped by dask to the 58 px of the grid.
    for dimensionality, chunks in (("projection", [1, 1, 58, 58]), ("stack", [1, 1, 4, 58, 58])):
        entry = output(section, dimensionality, "ome_zarr")
        assert entry["zarr_format"] == zarr_format and entry["ngff_version"] == version
        assert entry["axes"] == ["t", "c"] + (["z"] if dimensionality == "stack" else []) + ["y", "x"]
        assert len(entry["levels"]) == 1 and entry["chunks"] == chunks
        group = zarr.open_group(str(tmp_path / entry["path"]), mode="r")
        assert group.metadata.zarr_format == zarr_format
        level0 = np.asarray(group["0"][...]).squeeze()
        assert np.array_equal(level0, read(tmp_path, output(section, dimensionality)).squeeze())   # F9
    check_export(tmp_path)


@pytest.mark.parametrize(("version", "zarr_format"), [("0.5", 3), ("0.4", 2)])
def test_check_export_counts_chunk_reads(tmp_path, version, zarr_format):
    """OME-Zarr layout check 3 in check_export: a region inside one chunk touches that chunk only.

    Two FOVs of 20×8×8 (a hand-built stack taller than 16 planes) give a stack of two requested
    chunks (1, 1, 16, 512, 512) in Z, clipped to 16 × 8 × 14. check_export passes on the files the
    export writes; after level 0 is rewritten with chunks of 8 planes (and the record made to
    match, so that layout checks 1 and 2 and R2 pass), the first requested chunk spans two
    stored chunks and check_export fails naming layout check 3.
    """
    import zarr

    from starfinder.export._fuse import _codec
    from starfinder.export._record import directory_files, section_sha256
    z, y, x = np.meshgrid(np.arange(20), np.arange(8), np.arange(8), indexing="ij")
    arrays = {f"T{f}": (1 + 64 * z + 8 * y + x + 100 * f).astype(np.uint16) for f in range(2)}
    case = make_case("tall", {"T0": (0.0, 0.0, 0.0), "T1": (0.0, 0.0, 6.25)}, arrays)
    section = run(case, tmp_path, dimensionality="stack", scale_yx=1, ome_zarr=True, ome_zarr_version=version)
    entry = output(section, "stack", "ome_zarr")
    assert entry["zarr_format"] == zarr_format and entry["chunks"] == [1, 1, 16, 8, 14]
    check_export(tmp_path)
    path = tmp_path / entry["path"]
    group = zarr.open_group(str(path), mode="r+")
    data = group["0"][...]
    rewritten = group.create_array("0", shape=data.shape, dtype=data.dtype, chunks=(1, 1, 8, 8, 14), overwrite=True)
    rewritten[...] = data
    record = json.loads((tmp_path / "export.json").read_text())
    stored = next(o for o in record["fusion"]["outputs"] if o["path"] == entry["path"])
    stored["levels"][0]["chunks"] = stored["chunks"] = [1, 1, 8, 8, 14]
    stored["levels"][0]["codec"] = stored["compression"] = _codec(zarr.open_array(str(path / "0"), mode="r"))
    stored["files"] = directory_files(path)
    stored["sha256"] = section_sha256(stored["files"])
    (tmp_path / "export.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="OME-Zarr layout check 3 fails at level 0: .* touches 2 chunks"):
        check_export(tmp_path)


# --- FI10: dtypes written ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", ["uint8", "float32"])
def test_written_in_the_input_dtype(tmp_path, dtype):
    case = quad(dtype)
    section = run(case, tmp_path, dimensionality="both", scale_yx=0.5)
    for entry in section["outputs"]:
        assert entry["dtype"] == dtype and read(tmp_path, entry).dtype == np.dtype(dtype)
    check_export(tmp_path)


@pytest.mark.parametrize("dtype", ["uint16", "float32"])
def test_sum_projection_is_float32(tmp_path, dtype):
    case = zstack("equal", dtype)
    section = run(case, tmp_path, dimensionality="projection", projection="sum", scale_yx=1)
    entry = output(section, "projection")
    data = read(tmp_path, entry)
    assert entry["dtype"] == "float32" and data.dtype == np.float32
    assert np.array_equal(data, direct(case, section, entry))
    # Where only FOV Z0 covers, the value is its exact sum over Z.
    expected = case.arrays["Z0"].astype(np.float64).sum(axis=0)
    assert np.array_equal(data[:, :6], expected[:, :6].astype(np.float32))
    check_export(tmp_path)


# --- FI11: the stack alone and the middle plane --------------------------------------------------------------------

@pytest.mark.parametrize(("name", "n_z", "middle"), [("equal", 4, 2), ("different", 5, 2), ("plane", 1, 0)])
def test_stack_alone_and_middle_plane(tmp_path, name, n_z, middle):
    case = zstack(name)
    for projection in ("max", "sum"):
        folder = tmp_path / projection
        section = run(case, folder, dimensionality="stack", projection=projection)
        assert [o["path"] for o in section["outputs"]] == ["reference_stain_stack.ome.tif"]
        assert sorted(p.name for p in folder.iterdir()) == ["export.json", "reference_stain_stack.ome.tif"]
        entry = output(section, "stack")
        grid = section["grid"]
        assert entry["factor_yx"] == 2 and entry["dtype"] == "uint16"
        assert entry["shape_zyx"] == [n_z, math.ceil(grid["shape_zyx"][1] / 2), math.ceil(grid["shape_zyx"][2] / 2)]
        if n_z > 1:
            assert (entry["spacing_zyx"][0], entry["origin_zyx"][0]) == (grid["spacing_zyx"][0], grid["origin_zyx"][0])
        path = folder / entry["path"]
        whole = tifffile.imread(path)
        i = entry["shape_zyx"][0] // 2
        assert i == middle
        if n_z == 1:
            assert whole.ndim == 2 and read_ome_layout(path)["axes"] == "YX"
            assert np.array_equal(tifffile.imread(path, key=0), whole)
        else:
            assert np.array_equal(tifffile.imread(path, key=i), whole[i])
        plane = read_ome_layout(path)["planes"][i]
        assert abs(float(plane["PositionZ"]) - (entry["origin_zyx"][0] + i * entry["spacing_zyx"][0])) <= 1e-9
        assert abs(float(plane["PositionY"]) - entry["origin_zyx"][1]) <= 1e-9
        assert abs(float(plane["PositionX"]) - entry["origin_zyx"][2]) <= 1e-9
        check_export(folder)
    assert np.array_equal(tifffile.imread(tmp_path / "max" / "reference_stain_stack.ome.tif"),
                          tifffile.imread(tmp_path / "sum" / "reference_stain_stack.ome.tif"))
