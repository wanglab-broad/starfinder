"""The step times of segment, assign_molecules, stitch and assembly, and the acquisition count of the Summary (W-879).

Records: ``wall_s`` in the run record of ``FOV.segment``, in ``assign_molecules`` and in ``stitch``, saved in
``segmentation.json``, ``assignment.json`` and ``placement.json``. The report reads them into ``steps.csv``; a record
written before has none and its time stays ``not_recorded``. The Summary counts a reference stain taken in the
reference round's acquisition (tissue-2D's form: ``reference_stain`` is a channel of round1) with that round, and a
stain registered as an image of its own as a round. The measured seconds of the real runs are checked against bounds
only (positive, under a minute); the report's reading is checked against the dyadic seconds of ``STEP_SECONDS``.
"""
from __future__ import annotations

import json

import pytest

from starfinder.report._values import Unavailable, acquisition
from starfinder.stitching import STITCHING_METHODS, save_placement, stitch

from .report_fixtures import STEP_SECONDS, report_cells, report_records
from .stitching_fixtures import FRACTIONAL, STUB_SPEC, StubConfig, grids_for, inputs_for, stub_table
from .test_assignment import assign_boxes
from .test_report_collection import anndata_or_skip, collect, keyed
from .test_segmentation_persistence import golden, import_run  # noqa: F401 (the golden fixture)

pytestmark = [pytest.mark.report, pytest.mark.segmentation, pytest.mark.stitching, pytest.mark.contract]

LONG = 3600.0   # a bound, not a measurement: the runs here take well under a second
FOVS = ("F1", "F2", "F3", "F4")


# --- The records ----------------------------------------------------------------------------------------------

def test_a_segmentation_run_records_its_wall_time(golden, tmp_path):  # noqa: F811
    from starfinder.dataset import CheckpointConfig
    from starfinder.segmentation import SegmentationPlan
    fov, path, _ = golden
    checkpoints = CheckpointConfig(directory=tmp_path / "checkpoints")
    fov.segment(SegmentationPlan((import_run("nucleus", path),)), checkpoints=checkpoints)
    saved = fov.segmentation_results["nucleus"].record["wall_s"]
    written = json.loads((tmp_path / "checkpoints" / "FOV_001" / "segmentation" / "nucleus" / "segmentation.json")
                         .read_text())["wall_s"]
    assert 0 < saved < LONG and written == saved


def test_an_assignment_records_its_wall_time():
    assert 0 < assign_boxes().record["wall_s"] < LONG


def test_a_placement_records_its_wall_time_and_save_placement_keeps_it(monkeypatch, tmp_path):
    monkeypatch.setitem(STITCHING_METHODS, StubConfig, STUB_SPEC)
    table = stub_table()
    result = stitch(inputs_for(table, grids_for(table)), table, config=StubConfig(FRACTIONAL), sample="s1")
    assert 0 < result.record["wall_s"] < LONG
    folder = save_placement(result, tmp_path / "stitching")
    assert json.loads((folder / "placement.json").read_text())["wall_s"] == result.record["wall_s"]


# --- The report's steps table ---------------------------------------------------------------------------------

def test_the_report_shows_the_four_step_times_from_the_records(tmp_path):
    anndata_or_skip()
    case = report_cells(tmp_path / "root", times=True)
    _, _, tables = collect(case)
    steps = tables["steps"]
    for name, scope, rows in (("segment", "fov", len(FOVS)), ("assign_molecules", "fov", len(FOVS)),
                              ("stitch", "sample", 1), ("assemble", "sample", 1)):
        # The export's part ``segment`` (off here) is a sample step of the same name.
        found = steps[steps["name"].eq(name).to_numpy(bool) & steps["scope"].eq(scope).to_numpy(bool)]
        assert len(found) == rows, name
        assert found["seconds"].tolist() == [STEP_SECONDS[name]] * rows, name
    assert not [u for u in keyed(tables["unavailable"]) if u[0] == "workflow.time"]


def test_records_without_wall_s_keep_their_times_not_recorded(tmp_path):
    anndata_or_skip()
    case = report_cells(tmp_path / "root")
    _, _, tables = collect(case)
    rows = [u for u in keyed(tables["unavailable"]) if u[0] == "workflow.time"]
    # segment and assign_molecules of each FOV, stitch and assemble.
    assert len(rows) == 2 * len(FOVS) + 2 and {u[4] for u in rows} == {"not_recorded"}
    assert tables["steps"]["seconds"].isna().sum() >= len(rows)


def test_a_placement_with_a_time_and_no_export_shows_the_stitch_time(tmp_path):
    case = report_records(tmp_path / "root", times=True)
    _, _, tables = collect(case)
    steps = tables["steps"]
    assert steps.loc[steps["name"].eq("stitch").to_numpy(bool), "seconds"].tolist() == [STEP_SECONDS["stitch"]]


# --- The Summary's acquisition --------------------------------------------------------------------------------

ROUND_NAMES = ("round1", "round2", "round3", "round4", "protein")


def channels(count):
    return [{"channel": f"ch{i:02d}", "name": None, "wavelength": "unavailable"} for i in range(count)]


def tissue_record(*, stain_registered=False, reference="round1"):
    """The run record of a tissue-2D FOV: four rounds of 4 channels, a one-channel protein round, and the reference
    stain, one channel; ``stain_registered`` gives the stain a registration entry, as an image of its own has."""
    counts = {"round1": 4, "round2": 4, "round3": 4, "round4": 4, "protein": 1, "reference_stain": 1}
    registration = {"round2": [{"reference": reference}]}
    if stain_registered:
        registration["reference_stain"] = [{"reference": reference}]
    return {"channels": {name: channels(n) for name, n in counts.items()}, "registration": registration,
            "rounds": {"reference_round": reference}}


def test_a_stain_taken_in_the_reference_rounds_acquisition_counts_with_that_round():
    found = acquisition({"F1": tissue_record(), "F2": tissue_record()}, 1)
    assert found["rounds"] == 5 and found["channels"] == [1, 5]
    assert found["text"] == "5 rounds × 1–5 channels, 2D (Z = 1)"


def test_a_stain_registered_as_an_image_of_its_own_stays_a_round():
    found = acquisition({"F1": tissue_record(stain_registered=True)}, None)
    assert found["rounds"] == 6 and found["channels"] == [1, 4]
    assert found["text"] == "6 rounds × 1–4 channels"


def test_the_reference_round_is_read_from_the_registration_entries_of_an_older_record():
    record = tissue_record()
    del record["rounds"]
    assert acquisition({"F1": record}, None)["rounds"] == 5
    del record["registration"]["round2"]
    # No registration entry and no round: the reference round is unknown, the stain stays a round of its own.
    assert acquisition({"F1": record}, None)["rounds"] == 6


def test_the_stain_of_a_record_whose_reference_round_has_no_channels_stays_a_round():
    assert acquisition({"F1": tissue_record(reference="roundX")}, None)["rounds"] == 6


def test_no_record_has_no_acquisition():
    assert acquisition({"F1": None}, None) == Unavailable("no_run_record", "no run record holds channels")


def test_a_run_without_a_stain_counts_as_before():
    record = tissue_record()
    del record["channels"]["reference_stain"]
    assert acquisition({"F1": record}, 3)["text"] == "5 rounds × 1–4 channels, 3D"
