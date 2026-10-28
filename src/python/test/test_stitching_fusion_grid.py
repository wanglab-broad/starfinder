"""FI1, FI6 and FI10 of design/validation/stitching-assembly.md on hand-built geometry: the fusion grid, the output
levels and the configuration refusals, without multiview-stitcher.

Fixtures: ``quad`` (seed 2102), ``zstack`` (equal Z, different Z, the one-plane pair),
``plane_pair``, ``checker_pair``, ``fuse_band``, ``quad``'s first FOV alone and
``status_cases``' placement with every FOV excluded (``fuse_images`` called alone: status
``empty``, F8). The expected ``n0`` is the formula evaluated here. ``import
starfinder.export`` is checked in a subprocess.
"""
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from starfinder.assembly import AssemblyConfig
from starfinder.export import (
    ExportConfig,
    ExportIncompleteWarning,
    FusionConfig,
    FusionInput,
    LabelConfig,
    SpatialDataExportConfig,
    check_export,
    fuse_images,
    fused_grid,
)
from starfinder.export._fuse import sum_bound

from .assembly_fixtures import approved, golden_placement
from .fusion_fixtures import checker_pair, fuse_band, grid_of, make_case, plane_pair, quad, zstack

pytestmark = [pytest.mark.stitching, pytest.mark.validation]

SRC = Path(__file__).resolve().parents[1]


def formula_n0(case):
    """``floor((max_f (origin_f + (n_f − 1)·s) − o) / s + 1e-9) + 1`` per axis, over the included FOVs."""
    ids = case.included
    s = case.grids[ids[0]].metadata.spacing_zyx
    out = []
    for a in range(3):
        o = min(case.origin(f)[a] for f in ids)
        last = max(case.origin(f)[a] + (case.grids[f].shape_zyx[a] - 1) * s[a] for f in ids)
        out.append(int(math.floor((last - o) / s[a] + 1e-9)) + 1)
    return out


@pytest.mark.parametrize(("build", "expected"), [
    (lambda: quad(), (4, 58, 58)),
    (lambda: zstack("equal"), (4, 8, 14)),
    (lambda: zstack("different"), (5, 8, 14)),
    (lambda: fuse_band(), (16, 1408, 256)),
])
def test_fused_grid_multi_plane(build, expected):
    case = build()
    grid = fused_grid(case.placement, case.grids)
    assert grid.shape_zyx == expected
    assert list(grid.shape_zyx) == formula_n0(case)
    assert grid.spacing_zyx == case.grids[case.ids[0]].metadata.spacing_zyx
    for a in range(3):
        assert grid.origin_zyx[a] == min(case.origin(f)[a] for f in case.ids)       # bitwise
    assert not grid.one_plane and grid.fov_ids == tuple(case.ids)
    assert grid.metadata.frame_id == f"sample:{case.placement.sample}"


def test_fused_grid_one_plane_slab():
    case = plane_pair()
    grid = fused_grid(case.placement, case.grids)
    assert grid.one_plane and grid.shape_zyx == (1, 8, 18)
    assert grid.slab_z == (-0.5, 2.5)
    for dimensionality in ("projection", "stack"):
        spacing, origin, shape = grid.level(dimensionality)
        assert abs(spacing[0] - 3.0) <= 1e-9 and abs(origin[0] - 1.0) <= 1e-9 and shape[0] == 1
    pair = zstack("plane")
    grid = fused_grid(pair.placement, pair.grids)
    assert grid.one_plane and grid.shape_zyx == (1, 8, 14) and grid.origins_z == (0.0, 0.0)
    spacing, origin, _ = grid.level("stack")
    assert abs(spacing[0] - 1.0) <= 1e-9 and abs(origin[0] - 0.0) <= 1e-9


def test_fused_grid_one_fov_and_excluded_fovs():
    case = quad()
    first = make_case("quad1", {"Q0": (0.0, 0.0, 0.0)}, {"Q0": case.arrays["Q0"]})
    grid = fused_grid(first.placement, first.grids)
    assert grid.shape_zyx == (4, 32, 32) and grid.origin_zyx == (0.0, 0.0, 0.0)
    # Excluded FOVs do not extend the grid.
    for status in ("placeholder", "missing", "failed"):
        statuses = {"Q3": status}
        excluded = make_case("quad_x", dict(zip(case.ids, (case.origin(f) for f in case.ids))), case.arrays,
                             statuses=statuses)
        grid = fused_grid(excluded.placement, excluded.grids)
        assert grid.fov_ids == ("Q0", "Q1", "Q2")
        assert list(grid.shape_zyx) == formula_n0(excluded)


def test_fused_grid_refusals():
    case = quad()
    with pytest.raises(ValueError, match="no included FOV"):
        fused_grid(make_case("x", {"Q0": (0.0, 0.0, 0.0)}, {"Q0": case.arrays["Q0"]},
                             statuses={"Q0": "failed"}).placement, case.grids)
    grids = dict(case.grids, Q1=grid_of((4, 32, 32), (1.0, 0.5, 0.5)))
    with pytest.raises(ValueError, match="share one spacing"):
        fused_grid(case.placement, grids)
    grids = dict(case.grids, Q1=grid_of((1, 32, 32)))
    with pytest.raises(ValueError, match="one-plane FOVs cannot be fused beside multi-plane"):
        fused_grid(case.placement, grids)


@pytest.mark.parametrize(("k", "n"), [(1, 58), (2, 29), (4, 15)])
def test_output_level_geometry(k, n):
    """FI6, conditions 1 and 3: S = k·s, O = o + s·(k − 1)/2, N = ceil(n0/k); index identities."""
    case = quad()
    grid = fused_grid(case.placement, case.grids, config=FusionConfig(scale_yx=1 / k))
    for dimensionality in ("stack", "projection"):
        spacing, origin, shape = grid.level(dimensionality)
        assert shape[1:] == (n, n)
        for a in (1, 2):
            assert abs(spacing[a] - k * grid.spacing_zyx[a]) <= 1e-9
            assert abs(origin[a] - (grid.origin_zyx[a] + grid.spacing_zyx[a] * (k - 1) / 2)) <= 1e-9
        if dimensionality == "stack":
            assert (spacing[0], origin[0], shape[0]) == (grid.spacing_zyx[0], grid.origin_zyx[0], grid.shape_zyx[0])
        else:
            assert abs(spacing[0] - 4 * 1.0) <= 1e-9 and abs(origin[0] - 1.5) <= 1e-9 and shape[0] == 1
        # j = (j0 − (k − 1)/2)/k and v = floor(v0/k) at positions away from the fusion grid's voxel boundaries.
        q = np.array([0.37, 3.21, 11.6, 25.04, 40.33, 57.2])
        j0 = (q - grid.origin_zyx[1]) / grid.spacing_zyx[1]
        j = (q - origin[1]) / spacing[1]
        assert np.all(np.abs(j - (j0 - (k - 1) / 2) / k) <= 1e-9)
        assert np.array_equal(np.floor(j + 0.5), np.floor(np.floor(j0 + 0.5) / k))
    checker = checker_pair("3d")
    grid = fused_grid(checker.placement, checker.grids, config=FusionConfig(scale_yx=1 / k))
    spacing, origin, shape = grid.level("stack")
    assert shape[1:] == tuple(math.ceil(v / k) for v in grid.shape_zyx[1:])


def test_fuse_alone_without_an_included_fov(tmp_path):
    """F8 on status_cases' placement with every FOV excluded: no image, status empty, no_included_fov."""
    statuses = {fov: "failed" for fov in ("Position001", "Position003")} | {"Position002": "placeholder"}
    result = approved(golden_placement(statuses=statuses, review="required"))
    record = fuse_images(result, [], output_dir=tmp_path / "export")
    assert record.status == "empty" and record.paths == ()
    section = json.loads((tmp_path / "export" / "export.json").read_text())["fusion"]
    assert section["status"] == "empty" and section["reason"] == "no_included_fov" and section["outputs"] == []
    assert [entry["reason"] for entry in section["fovs"]] == ["failed", "placeholder", "failed"]
    assert sorted(p.name for p in (tmp_path / "export").iterdir()) == ["export.json"]
    check_export(tmp_path / "export")


# --- FI10: configuration refusals and the sum bound ------------------------------------------------------------

@pytest.mark.parametrize("make", [
    lambda: FusionConfig(scale_yx=0.3),
    lambda: FusionConfig(pyramid_levels=-1),
    lambda: FusionConfig(images=()),
    lambda: FusionConfig(images=("a=file:a_{fov_id}.tif", "A=file:b_{fov_id}.tif")),
    lambda: FusionConfig(images=("my image=file:x_{fov_id}.tif",)),
    lambda: ExportConfig(export_spatialdata=True, fusion=FusionConfig(ome_zarr=True)),
    lambda: FusionConfig(ome_zarr_version="0.6"),
    lambda: SpatialDataExportConfig(zarr_format=4),
    lambda: LabelConfig(scale_yx=0.3),
    lambda: FusionConfig(dimensionality="volume"),
    lambda: FusionConfig(projection="mean"),
    lambda: ExportConfig(table_format="tsv"),
    lambda: ExportConfig(assembly=AssemblyConfig(), fuse="yes"),
])
def test_configuration_refusals(make):
    with pytest.raises(ValueError):
        make()


def test_configuration_accepts():
    assert FusionConfig().factor_yx == 2 and FusionConfig(scale_yx=0.25).factor_yx == 4
    assert FusionConfig(scale_yx=1 / 3).factor_yx == 3
    assert FusionConfig(images=["reference_stain", "dapi=file:x/{fov_id}.tif"]).image_names == ("reference_stain",
                                                                                                 "dapi")
    assert FusionConfig(dimensionality="both").dimensionalities == ("projection", "stack")
    assert ExportConfig().fuse_labels is False and ExportConfig().export_spatialdata is False


@pytest.mark.parametrize(("dtype", "planes", "ok"), [("uint16", 256, True), ("uint16", 257, False),
                                                      ("uint8", 65793, True), ("uint8", 65794, False),
                                                      ("float32", 10 ** 6, True)])
def test_sum_bound(dtype, planes, ok):
    """The integer sum bound n_planes × max ≤ 2^24, evaluated on declared plane counts (no array is read)."""
    if ok:
        sum_bound(dtype, planes)
    else:
        with pytest.raises(ValueError, match="2\\^24"):
            sum_bound(dtype, planes)


def test_refusals_of_fuse_before_any_file(tmp_path):
    """Grid, dtype and pyramid refusals of fuse_images raise before anything is written (no library needed)."""
    case = quad()
    out = tmp_path / "export"
    for dtype in ("int8", "int16", "uint32", "int32", "float64", "uint64", "bool"):
        bad = quad(dtype)
        with pytest.raises(ValueError, match=f"dtype {dtype}.*\\['uint8', 'uint16', 'float32'\\]"):
            fuse_images(bad.placement, bad.inputs(), output_dir=out)
    mixed = quad(dtypes={"Q1": "uint8"})
    with pytest.raises(ValueError, match="mixes dtypes"):
        fuse_images(mixed.placement, mixed.inputs(), output_dir=out)
    with pytest.raises(ValueError, match="pyramid_levels 4 exceeds .* = 3"):
        fuse_images(case.placement, case.inputs(), config=FusionConfig(scale_yx=0.25, pyramid_levels=4), output_dir=out)
    with pytest.raises(ValueError, match="no FusionInput|FOVs without one"):
        fuse_images(case.placement, case.inputs()[1:], output_dir=out)
    other = FusionInput("Other", case.grids["Q0"], {"reference_stain": case.arrays["Q0"]})
    with pytest.raises(ValueError, match="not included FOVs"):
        fuse_images(case.placement, case.inputs() + [other], output_dir=out)
    assert not out.exists()


def test_missing_extra_records_not_run(tmp_path, monkeypatch):
    """With multiview_stitcher not importable fuse_images writes no image and records not_run, extra_missing."""
    monkeypatch.setitem(sys.modules, "multiview_stitcher", None)
    case = quad()
    with pytest.warns(ExportIncompleteWarning, match="extra_missing"):
        record = fuse_images(case.placement, case.inputs(), output_dir=tmp_path)
    assert record.status == "not_run" and record.section["reason"] == "extra_missing"
    assert record.section["extra"] == "multiview-stitcher" and record.section["outputs"] == []
    assert sorted(p.name for p in tmp_path.iterdir()) == ["export.json"]
    exported = json.loads((tmp_path / "export.json").read_text())
    assert exported["complete"] is False and exported["parts"]["fuse"] == "not_run"
    check_export(tmp_path)


def test_import_loads_no_optional_library():
    code = ("import sys, starfinder.export; "
            "print([m for m in ('multiview_stitcher', 'zarr', 'spatialdata') if m in sys.modules])")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                         env={"PYTHONPATH": str(SRC), "PATH": "/usr/bin:/bin"})
    assert out.stdout.strip() == "[]"
