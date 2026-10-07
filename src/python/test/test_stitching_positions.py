"""PL1 of docs/stitching-assembly-validation.md: the tile position table and its three converters.

Fixtures: the stage files of ``hand_records`` (``maf_3x2_gap``, ``maf_2x3_jitter``),
``golden_stitching``'s stage file, supplied tables in µm, nm and mm, and the
``positions.csv`` of ``sample-tiny-v1`` (seed 42). The expected cells come from the frozen
legacy helper ``run_stitching_preparation`` of ``test_workflow_stitching_golden.py``.
"""
import hashlib
import math

import numpy as np
import pandas as pd
import pytest

from starfinder.stitching import (GridSnap, StageAxisMapping, TilePositionTable, positions_from_grid,
                                  positions_from_leica, positions_from_table)

from .stitching_fixtures import MAF_2X3_JITTER, MAF_3X2_GAP, maf_text, position_ids
from .test_workflow_stitching_golden import POSITIONS, run_stitching_preparation, stage_file

pytestmark = [pytest.mark.stitching, pytest.mark.validation]

SNAP = GridSnap(overlap=0.1, tile_shape_yx=(16, 24), spacing_yx=(0.25, 0.25))
STEP_Y, STEP_X = (1 - 0.1) * 16 * 0.25, (1 - 0.1) * 24 * 0.25
D_M = 2.0 ** -12          # a stage step of 244.140625 µm, exact in metres and in µm
X0_M, Y0_M = 0.0625, 0.046875


def _write(tmp_path, text, name="stage.maf"):
    path = tmp_path / name
    path.write_text(text)
    return path


def _three_points():
    """Position 1 at the corner, 2 moved by +X, 3 moved by +2·Y."""
    return ((1, repr(X0_M), repr(Y0_M)), (2, repr(X0_M + D_M), repr(Y0_M)), (3, repr(X0_M), repr(Y0_M + 2 * D_M)))


MAPPINGS = [("+x", "+y"), ("+x", "-y"), ("-x", "+y"), ("-x", "-y"),
            ("+y", "+x"), ("+y", "-x"), ("-y", "+x"), ("-y", "-x")]


@pytest.mark.parametrize("x,y", MAPPINGS)
def test_stage_axis_mapping_moves_stage_axes_where_it_names(tmp_path, x, y):
    points = _three_points()
    table = positions_from_leica(_write(tmp_path, maf_text(points)), stage_axes=StageAxisMapping(x=x, y=y),
                                 fov_ids=position_ids(points))
    frame = table.positions
    position = {pid: np.array([z, b, a]) for pid, z, b, a in
                zip(frame.position_id, frame.nominal_z, frame.nominal_y, frame.nominal_x)}
    step = D_M * 1e6
    axis = {"z": 0, "y": 1, "x": 2}
    expected_x = np.zeros(3)
    expected_x[axis[x[1]]] = step if x[0] == "+" else -step
    expected_y = np.zeros(3)
    expected_y[axis[y[1]]] = 2 * step if y[0] == "+" else -2 * step
    assert np.abs(position[2] - position[1] - expected_x).max() <= 1e-9
    assert np.abs(position[3] - position[1] - expected_y).max() <= 1e-9
    assert frame.nominal_y.min() == 0.0 and frame.nominal_x.min() == 0.0
    assert (frame.nominal_z == 0.0).all()
    assert table.record["stage_axes"] == {"x": x, "y": y, "source": "declared"}


def test_unsnapped_stage_table_keeps_the_stage_positions(tmp_path):
    path = _write(tmp_path, maf_text(MAF_2X3_JITTER))
    table = positions_from_leica(path, stage_axes=StageAxisMapping(x="+x", y="+y"),
                                 fov_ids=position_ids(MAF_2X3_JITTER))
    frame = table.positions
    assert tuple(frame.columns) == ("fov_id", "position_id", "nominal_z", "nominal_y", "nominal_x", "source",
                                    "grid_column", "grid_row")
    assert list(frame.position_id) == [1, 2, 3, 4, 5, 6]
    assert list(frame.fov_id) == [f"Position{k:03d}" for k in range(1, 7)]
    assert set(frame.source) == {"stage"} and frame.grid_column.isna().all() and frame.grid_row.isna().all()
    stage_x = np.array([float(x) for _, x, _ in MAF_2X3_JITTER]) * 1e6
    stage_y = np.array([float(y) for _, _, y in MAF_2X3_JITTER]) * 1e6
    np.testing.assert_allclose(frame.nominal_x, stage_x - stage_x.min(), rtol=0, atol=1e-9)
    np.testing.assert_allclose(frame.nominal_y, stage_y - stage_y.min(), rtol=0, atol=1e-9)
    assert table.source == f"leica:{path.resolve()}" and table.unit == "µm"
    assert table.record["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert table.record["snap_to_grid"] is None
    assert table.record["stage_positions"][0]["attributes"]["Sections"] == "12"


def _legacy_cells(tmp_path, monkeypatch, text):
    out = run_stitching_preparation(tmp_path, monkeypatch, text)
    grid = pd.read_csv(out / "grid.csv", index_col=0)
    return {int(i): (int(c), int(r)) for c, r, i in zip(grid.col, grid.row, grid["id"]) if i != 0}


@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive")
@pytest.mark.parametrize("text,step", [
    pytest.param(stage_file(), 244.1, id="golden_stitching"),
    pytest.param(maf_text(MAF_3X2_GAP), 261.0, id="maf_3x2_gap"),
    pytest.param(maf_text(MAF_2X3_JITTER), 261.0, id="maf_2x3_jitter"),
])
def test_snapped_cells_equal_the_legacy_grid(tmp_path, monkeypatch, text, step):
    expected = _legacy_cells(tmp_path / "legacy", monkeypatch, text)
    pids = sorted(expected)
    table = positions_from_leica(_write(tmp_path, text), stage_axes=StageAxisMapping(x="+x", y="+y"),
                                 fov_ids={pid: f"Position{pid:03d}" for pid in pids}, snap_to_grid=SNAP)
    frame = table.positions
    cells = {int(p): (int(c), int(r)) for p, c, r in zip(frame.position_id, frame.grid_column, frame.grid_row)}
    assert cells == expected
    assert set(frame.source) == {"grid"}
    assert list(frame.nominal_y) == [r * STEP_Y for r in frame.grid_row]
    assert list(frame.nominal_x) == [c * STEP_X for c in frame.grid_column]
    assert (frame.nominal_z == 0.0).all()
    snap = table.record["snap_to_grid"]
    assert snap["cell_step_um"] == step and snap["cell_step_axis"] == "y"
    assert snap["tile_shape_yx"] == [16, 24] and snap["spacing_yx"] == [0.25, 0.25] and snap["overlap_yx"] == [0.1, 0.1]


def test_golden_stage_file_covers_its_eight_positions():
    assert sorted(POSITIONS) == list(range(1, 9))


def test_a_remaining_step_tie_goes_to_the_smaller_step(tmp_path):
    # y differences 100 µm and 200 µm, once each: the step is 100 µm.
    points = ((1, "0.05", "0.02"), (2, "0.05", "0.0201"), (3, "0.05", "0.0203"))
    table = positions_from_leica(_write(tmp_path, maf_text(points)), stage_axes=StageAxisMapping(x="+x", y="+y"),
                                 fov_ids=position_ids(points), snap_to_grid=SNAP)
    assert table.record["snap_to_grid"]["cell_step_um"] == 100.0
    assert list(table.positions.grid_row) == [0, 1, 3]


ORDERS = {
    "column_major_down_right": [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],
    "column_major_down_right_serpentine": [(0, 0), (0, 1), (0, 2), (1, 2), (1, 1), (1, 0)],
    "row_major_right_down": [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2)],
    "row_major_right_down_serpentine": [(0, 0), (1, 0), (1, 1), (0, 1), (0, 2), (1, 2)],
}


@pytest.mark.parametrize("order", sorted(ORDERS))
def test_regular_grid_orders_on_a_2_by_3_grid(order):
    ids = [f"FOV{k}" for k in range(6)]
    table = positions_from_grid(columns=2, rows=3, order=order, tile_shape_yx=(16, 24), spacing_yx=(0.25, 0.5),
                                overlap=(0.25, 0.125), fov_ids=ids)
    frame = table.positions
    assert list(zip(frame.grid_column, frame.grid_row)) == ORDERS[order]
    step_y, step_x = (1 - 0.25) * 16 * 0.25, (1 - 0.125) * 24 * 0.5
    assert list(frame.nominal_y) == [r * step_y for _, r in ORDERS[order]]
    assert list(frame.nominal_x) == [c * step_x for c, _ in ORDERS[order]]
    assert (frame.nominal_z == 0.0).all() and frame.position_id.isna().all() and set(frame.source) == {"grid"}
    assert list(frame.fov_id) == ids and table.source == "grid"


def test_column_major_puts_fov_k_at_column_k_div_rows():
    ids = [f"FOV{k}" for k in range(7)]
    table = positions_from_grid(columns=3, rows=3, order="column_major_down_right", tile_shape_yx=(16, 24),
                                spacing_yx=(0.25, 0.25), overlap=0.25, fov_ids=ids)
    frame = table.positions
    assert list(frame.grid_column) == [k // 3 for k in range(7)]
    assert list(frame.grid_row) == [k % 3 for k in range(7)]
    assert table.record["step_yx_um"] == [(1 - 0.25) * 16 * 0.25, (1 - 0.25) * 24 * 0.25]


def test_regular_grid_reproduces_the_scene_nominal_grid(tmp_path):
    from starfinder.synthetic import generate_sample_scene, sample_scene_preset, save_sample_scene
    book, config = sample_scene_preset("sample-tiny-v1")
    scene = generate_sample_scene(book, config=config)
    save_sample_scene(scene, tmp_path / "scene")
    acquisition = config.acquisition
    grid = positions_from_grid(columns=acquisition.columns, rows=acquisition.rows, order=acquisition.order,
                               tile_shape_yx=acquisition.fov_shape_zyx[1:], spacing_yx=acquisition.spacing_zyx[1:],
                               overlap=acquisition.overlap, fov_ids=list(scene.positions.fov_id))
    supplied = positions_from_table(tmp_path / "scene" / "positions.csv")
    for column in ("nominal_z", "nominal_y", "nominal_x"):
        assert grid.positions[column].to_numpy().tobytes() == scene.positions[column].to_numpy().tobytes()
        assert supplied.positions[column].to_numpy().tobytes() == scene.positions[column].to_numpy().tobytes()
    assert list(supplied.positions.fov_id) == list(scene.positions.fov_id)


def test_supplied_tables_convert_units(tmp_path):
    um = pd.DataFrame({"fov_id": ["a", "b"], "nominal_z": [0.0, 1.5], "nominal_y": [12.25, -3.0],
                       "nominal_x": [100.125, 0.0], "note": ["x", "y"]})
    expected = um[["nominal_z", "nominal_y", "nominal_x"]].to_numpy()
    for unit, factor in (("µm", 1.0), ("um", 1.0), ("nm", 1000.0), ("mm", 1e-3)):
        frame = um.copy()
        frame[["nominal_z", "nominal_y", "nominal_x"]] = expected * factor
        table = positions_from_table(frame, unit=unit)
        assert np.abs(table.nominal_zyx() - expected).max() <= 1e-9
        assert table.unit == "µm" and table.record["unit"] == unit and table.source == "table:frame"
    path = tmp_path / "positions.csv"
    um.to_csv(path, index=False)
    table = positions_from_table(path)
    assert table.nominal_zyx().tobytes() == expected.tobytes()
    frame = table.positions
    assert set(frame.source) == {"supplied"}
    assert frame.position_id.isna().all() and frame.grid_column.isna().all() and frame.grid_row.isna().all()
    assert table.source == f"table:{path.resolve()}"
    assert table.record["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_leica_failures_name_the_item(tmp_path):
    axes = StageAxisMapping(x="+x", y="+y")
    repeated = MAF_3X2_GAP + ((2, "0.077", "0.049"),)
    with pytest.raises(ValueError, match="repeats PositionID 2"):
        positions_from_leica(_write(tmp_path, maf_text(repeated)), stage_axes=axes, fov_ids=position_ids(repeated))
    ids = position_ids(MAF_3X2_GAP)
    del ids[4]
    with pytest.raises(ValueError, match="PositionID 4 .*no FOV ID"):
        positions_from_leica(_write(tmp_path, maf_text(MAF_3X2_GAP)), stage_axes=axes, fov_ids=ids)
    bad = ((1, "0.076", "0.048"), (2, "nan", "0.048261"))
    with pytest.raises(ValueError, match="PositionID 2 has a non-finite"):
        positions_from_leica(_write(tmp_path, maf_text(bad)), stage_axes=axes, fov_ids=position_ids(bad))
    close = ((1, "0.076", "0.048"), (2, "0.076", "0.048261"), (3, "0.0760005", "0.048"))
    with pytest.raises(ValueError, match="two positions in one cell: PositionID 1, 3"):
        positions_from_leica(_write(tmp_path, maf_text(close)), stage_axes=axes, fov_ids=position_ids(close),
                             snap_to_grid=SNAP)
    with pytest.raises(TypeError, match="stage_axes"):
        positions_from_leica(_write(tmp_path, maf_text(MAF_3X2_GAP)), stage_axes=None,
                             fov_ids=position_ids(MAF_3X2_GAP))
    with pytest.raises(FileNotFoundError):
        positions_from_leica(tmp_path / "absent.maf", stage_axes=axes, fov_ids={})


def test_stage_axis_mapping_refuses_one_axis_twice():
    with pytest.raises(ValueError, match="names image axis 'x' twice"):
        StageAxisMapping(x="+x", y="-x")
    with pytest.raises(ValueError, match="must be one of"):
        StageAxisMapping(x="+z", y="+y")


@pytest.mark.parametrize("change,match", [
    (dict(overlap=1.0), r"overlap \(y\) must satisfy 0 ≤ overlap < 1"),
    (dict(overlap=-0.1), r"overlap \(y\) must satisfy"),
    (dict(overlap=(0.1, 1.0)), r"overlap \(x\) must satisfy"),
    (dict(tile_shape_yx=(0, 24)), r"tile_shape_yx \(y\) must be an integer ≥ 1"),
    (dict(spacing_yx=(0.25, 0.0)), r"spacing_yx \(x\) must be finite and > 0"),
    (dict(spacing_yx=(math.nan, 0.25)), r"spacing_yx \(y\) must be finite and > 0"),
])
def test_grid_parameter_rules(change, match):
    snap = dict(overlap=0.1, tile_shape_yx=(16, 24), spacing_yx=(0.25, 0.25)) | change
    with pytest.raises(ValueError, match="GridSnap " + match):
        GridSnap(**snap)
    grid = dict(columns=2, rows=2, order="column_major_down_right", fov_ids=["a", "b"]) | snap
    with pytest.raises(ValueError, match="positions_from_grid " + match):
        positions_from_grid(**grid)


def test_regular_grid_failures():
    grid = dict(columns=2, rows=2, tile_shape_yx=(16, 24), spacing_yx=(0.25, 0.25), overlap=0.1)
    with pytest.raises(ValueError, match="5 FOVs for 2 × 2 = 4 cells"):
        positions_from_grid(order="column_major_down_right", fov_ids=list("abcde"), **grid)
    with pytest.raises(ValueError, match="order must be one of"):
        positions_from_grid(order="spiral", fov_ids=list("ab"), **grid)
    with pytest.raises(ValueError, match="fov_id must be unique; repeated: a"):
        positions_from_grid(order="column_major_down_right", fov_ids=list("aa"), **grid)


def test_supplied_table_failures(tmp_path):
    base = pd.DataFrame({"fov_id": ["a", "b"], "nominal_z": [0.0, 0.0], "nominal_y": [0.0, 1.0],
                         "nominal_x": [0.0, 2.0]})
    with pytest.raises(ValueError, match="repeated: a"):
        positions_from_table(base.assign(fov_id=["a", "a"]))
    with pytest.raises(ValueError, match="nonempty string; rows 1"):
        positions_from_table(base.assign(fov_id=["a", ""]))
    with pytest.raises(ValueError, match=r"lacks the columns \['nominal_y'\]"):
        positions_from_table(base.drop(columns="nominal_y"))
    with pytest.raises(ValueError, match="nominal_x must be a finite number; FOVs b"):
        positions_from_table(base.assign(nominal_x=[0.0, np.inf]))
    with pytest.raises(ValueError, match="unit must be one of"):
        positions_from_table(base, unit="px")
    path = tmp_path / "positions.csv"
    path.write_text("fov_id,nominal_z,nominal_y,nominal_x\na,0,0,0\nb,0,,1\n")
    with pytest.raises(ValueError, match="nominal_y must be a finite number; FOVs b"):
        positions_from_table(path)


def test_table_rules_hold_for_direct_construction():
    frame = positions_from_grid(columns=1, rows=2, order="column_major_down_right", tile_shape_yx=(16, 24),
                                spacing_yx=(0.25, 0.25), overlap=0.25, fov_ids=["a", "b"]).positions
    with pytest.raises(ValueError, match="unit"):
        TilePositionTable(frame, "grid", unit="nm")
    with pytest.raises(ValueError, match="row source"):
        TilePositionTable(frame.assign(source=["grid", "guess"]), "grid")
    with pytest.raises(ValueError, match="columns"):
        TilePositionTable(frame.drop(columns="grid_row"), "grid")
    empty = TilePositionTable(frame.iloc[:0], "grid")
    assert empty.fov_ids == []
