"""Stage-position selection before conversion (W-566, OM-14): a whole-acquisition stage file and its per-sample subset.

``positions_from_leica(..., position_ids=...)`` selects the sample's positions before the axis mapping,
the shift to the smallest y and x, the snap step, the cells and the clash check, so that a stage file that
holds every sample's positions gives the tile position table, and ``Dataset.stitch`` the placement, of the
file that holds only the sample's. Fixtures, all written in the test: sample A is ``MAF_2X3_JITTER`` of
``hand_records`` (PositionIDs 1 to 6); sample B is seven positions at smaller stage X and Y in one column
with a 300 µm step (PositionIDs 7 to 13), so that the shift and the most frequent step differ when both
samples are read together (snapped with B's step, A's positions clash). The whole file interleaves them. ``Dataset.stitch`` runs for real up to the
method, which ``placement_from_table`` stands in for (no extra needed).
"""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import starfinder.stitching as stitching
from starfinder.dataset import Dataset, RoundState, from_workflow_config
from starfinder.io import save_volume
from starfinder.stitching import GridSnap, StageAxisMapping, placement_from_table, positions_from_leica

from .stitching_fixtures import MAF_2X3_JITTER, STAGE_SPACING, grid, maf_text

pytestmark = [pytest.mark.stitching, pytest.mark.contract]

SAMPLE_A = MAF_2X3_JITTER
SAMPLE_B = tuple((7 + k, "0.0495", repr(0.0195 + 0.0003 * k)) for k in range(7))
WHOLE = (SAMPLE_B[0], *SAMPLE_A[:3], *SAMPLE_B[1:], *SAMPLE_A[3:])
AXES = StageAxisMapping(x="+x", y="+y")
SHAPE = (2, 16, 24)
SNAP = GridSnap(overlap=0.1, tile_shape_yx=SHAPE[1:], spacing_yx=STAGE_SPACING[1:])
IDS_A = {pid: f"Position{pid:03d}" for pid, *_ in SAMPLE_A}


def write(tmp_path, name, points):
    path = tmp_path / name
    path.write_text(maf_text(points))
    return path


@pytest.mark.parametrize("snap", [None, SNAP], ids=["stage", "snapped"])
def test_a_whole_acquisition_file_gives_the_table_of_its_sample_subset(tmp_path, snap):
    whole, subset = write(tmp_path, "acquisition.maf", WHOLE), write(tmp_path, "sample_a.maf", SAMPLE_A)
    selected = positions_from_leica(whole, stage_axes=AXES, fov_ids=IDS_A, position_ids=list(IDS_A),
                                    snap_to_grid=snap)
    alone = positions_from_leica(subset, stage_axes=AXES, fov_ids=IDS_A, snap_to_grid=snap)
    pd.testing.assert_frame_equal(selected.positions, alone.positions, check_exact=True)
    for key in ("fov_ids", "stage_positions", "stage_axes", "snap_to_grid"):
        assert selected.record[key] == alone.record[key], key
    assert selected.record["path"] == str(whole.resolve()) and selected.record["sha256"] != alone.record["sha256"]
    # Read together, the other sample's positions move the shift, and snapped B's 300 µm step puts two of A's
    # positions in one cell: selection before the conversion is what makes the whole file usable.
    every = {pid: f"Position{pid:03d}" for pid, *_ in WHOLE}
    if snap is not None:
        with pytest.raises(ValueError, match="puts two positions in one cell"):
            positions_from_leica(whole, stage_axes=AXES, snap_to_grid=snap, fov_ids=every)
        return
    together = positions_from_leica(whole, stage_axes=AXES, fov_ids=every)
    rows = together.positions.set_index("fov_id").loc[alone.positions.fov_id].reset_index()
    assert not rows[["nominal_y", "nominal_x"]].equals(alone.positions[["nominal_y", "nominal_x"]])


@pytest.mark.parametrize("position_ids, match", [
    ([1, 2, 99], r"PositionID 99 is not in stage file"),
    ([1, 1], r"position_ids repeats PositionID 1"),
    ([], "nonempty list of integer PositionIDs"),
    (["1"], "nonempty list of integer PositionIDs"),
])
def test_the_selected_ids_are_validated(tmp_path, position_ids, match):
    whole = write(tmp_path, "acquisition.maf", WHOLE)
    with pytest.raises(ValueError, match=match):
        positions_from_leica(whole, stage_axes=AXES, fov_ids={**IDS_A, 99: "Position099"}, position_ids=position_ids)


def test_a_selected_id_without_a_fov_id_raises(tmp_path):
    whole = write(tmp_path, "acquisition.maf", WHOLE)
    with pytest.raises(ValueError, match=r"PositionID 7 of .* has no FOV ID"):
        positions_from_leica(whole, stage_axes=AXES, fov_ids=IDS_A, position_ids=[1, 7])


def dataset(tmp_path):
    """A dataset with the default {i} pattern and a reference image per FOV of sample A."""
    ds = Dataset(tmp_path / "input", tmp_path / "output", "data", "sample", "out",
                 rounds=RoundState(sequencing_rounds=["round1"], reference_round="round1"), channel_order=("ch00",))
    assert ds.fov_pattern == "Position{i:03d}"
    for fov in IDS_A.values():
        save_volume(np.zeros(SHAPE, np.uint16), tmp_path / "output" / "fovs" / fov / "reference.tif",
                    metadata=grid(SHAPE, STAGE_SPACING).metadata)
    return ds


@pytest.mark.parametrize("named", ["pattern", "position_ids"])
@pytest.mark.parametrize("snap", [None, {"overlap": 0.1}], ids=["stage", "snapped"])
def test_dataset_stitch_places_a_whole_acquisition_file_as_its_subset(tmp_path, monkeypatch, named, snap):
    """Dataset.stitch with a brace pattern (PositionID p is Position{p:03d}) or with each FOV's PositionID: the whole
    file and the sample's subset give equal placement tables."""
    monkeypatch.setattr(stitching, "stitch", lambda inputs, table, *, sample, **kw: placement_from_table(
        table, {item.fov_id: item.grid for item in inputs}, sample=sample))
    ds = dataset(tmp_path)
    fov_ids = list(IDS_A.values())
    placements = {}
    for name, points in (("acquisition.maf", WHOLE), ("sample_a.maf", SAMPLE_A)):
        positions = {"source": "stage_file", "stage_axes": {"x": "+x", "y": "+y"}, "snap_to_grid": snap,
                     "stage_file": str(write(tmp_path, name, points))}
        position_ids = {fov: pid for pid, fov in IDS_A.items()} if named == "position_ids" else None
        placements[name] = ds.stitch("sampleA", fov_ids, table=positions, config=None, position_ids=position_ids)
    whole, alone = placements["acquisition.maf"], placements["sample_a.maf"]
    pd.testing.assert_frame_equal(whole.fovs, alone.fovs, check_exact=True)
    assert whole.fovs.fov_id.tolist() == fov_ids


def test_a_workflow_built_dataset_names_and_places_by_its_brace_pattern(tmp_path, monkeypatch):
    """from_workflow_config's Dataset holds the default {i} pattern: fov_ids names the FOVs from it, and
    Dataset.stitch(table={"source": "stage_file", ...}) maps PositionID p to Position{p:03d} and places the
    sample's positions of the whole file as those of its subset."""
    monkeypatch.setattr(stitching, "stitch", lambda inputs, table, *, sample, **kw: placement_from_table(
        table, {item.fov_id: item.grid for item in inputs}, sample=sample))
    config = {"root_input_path": str(tmp_path / "input"), "root_output_path": str(tmp_path), "dataset_id": "data",
              "sample_id": "sample", "output_id": "out", "n_rounds": 1, "ref_round": "round1", "rotate_angle": 0,
              "seq_channel_order": ["ch00"]}
    ds = from_workflow_config(config).dataset
    assert ds.fov_pattern == "Position{i:03d}"
    fov_ids = ds.fov_ids(6, start=1)
    assert fov_ids == list(IDS_A.values())
    for fov in fov_ids:
        save_volume(np.zeros(SHAPE, np.uint16), ds.output_root / "fovs" / fov / "reference.tif",
                    metadata=grid(SHAPE, STAGE_SPACING).metadata)
    placements = [ds.stitch("sampleA", fov_ids, config=None, table={
        "source": "stage_file", "stage_axes": {"x": "+x", "y": "+y"}, "snap_to_grid": {"overlap": 0.1},
        "stage_file": str(write(tmp_path, name, points))})
        for name, points in (("acquisition.maf", WHOLE), ("sample_a.maf", SAMPLE_A))]
    pd.testing.assert_frame_equal(placements[0].fovs, placements[1].fovs, check_exact=True)
    assert placements[0].record["table"]["fov_ids"] == {str(p): f for p, f in IDS_A.items()}


def test_dataset_stitch_reports_a_fov_the_stage_file_does_not_name(tmp_path, monkeypatch):
    monkeypatch.setattr(stitching, "stitch", lambda *a, **k: pytest.fail("stitch must not be called"))
    ds = dataset(tmp_path)
    positions = {"source": "stage_file", "stage_axes": {"x": "+x", "y": "+y"},
                 "stage_file": str(write(tmp_path, "acquisition.maf", (*SAMPLE_A[:3], *SAMPLE_B)))}
    with pytest.raises(ValueError, match=r"missing \['Position004', 'Position005', 'Position006'\]"):
        ds.stitch("sampleA", list(IDS_A.values()), table=positions, config=None)
    with pytest.raises(ValueError, match=r"PositionID 4, 5, 6 is not in stage file"):
        ds.stitch("sampleA", list(IDS_A.values()), table=positions, config=None,
                  position_ids={fov: pid for pid, fov in IDS_A.items()})
    with pytest.raises(ValueError, match=r"no PositionID p of .* names one of the FOVs .* as 'tile_\{i\}'"):
        replace(ds, fov_pattern="tile_{i}").stitch("sampleA", list(IDS_A.values()), table=positions, config=None)
    with pytest.raises(ValueError, match=r"position_ids gives two FOVs one PositionID"):
        ds.stitch("sampleA", ["Position001", "Position002"], table=positions, config=None,
                  position_ids={"Position001": 1, "Position002": 1})


def write_table(tmp_path, name, rows):
    """A supplied position table (positions_from_table's columns, µm) of (fov_id, nominal_y, nominal_x) rows."""
    path = tmp_path / name
    pd.DataFrame([{"fov_id": f, "nominal_z": 0.0, "nominal_y": y, "nominal_x": x} for f, y, x in rows]).to_csv(
        path, index=False)
    return path


def test_dataset_stitch_selects_its_fovs_from_a_supplied_table(tmp_path, monkeypatch):
    """A supplied table that also holds other FOVs gives, for fov_ids, the placement of a table that holds only them;
    a FOV of fov_ids the table does not list still raises."""
    monkeypatch.setattr(stitching, "stitch", lambda inputs, table, *, sample, **kw: placement_from_table(
        table, {item.fov_id: item.grid for item in inputs}, sample=sample))
    ds = dataset(tmp_path)
    rows = [("Position001", 0.0, 0.0), ("Position002", 259.2, 0.0), ("Position003", 0.0, 259.2)]
    placements = [ds.stitch("sampleA", ["Position003", "Position002"], config=None,
                            table={"source": "table", "table": str(write_table(tmp_path, name, table_rows))})
                  for name, table_rows in (("all.csv", rows), ("selected.csv", rows[1:]))]
    pd.testing.assert_frame_equal(placements[0].fovs, placements[1].fovs, check_exact=True)
    assert placements[0].fovs.fov_id.tolist() == ["Position002", "Position003"]          # table order
    with pytest.raises(ValueError, match=r"missing \['Position004'\]"):
        ds.stitch("sampleA", ["Position002", "Position004"], config=None,
                  table={"source": "table", "table": str(write_table(tmp_path, "all.csv", rows))})
