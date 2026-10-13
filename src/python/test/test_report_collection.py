"""The collection of one sample's records into the report's tables and record (W-452).

Checks of docs/report-algorithms.md ("Validation design") on the hand-written fixtures
``report_records`` and ``report_cells`` of ``report_fixtures.py``: RC1 to RC5, RC11 to RC13
for the columns the collection copies from the records, the record of unavailable values
(RC12), the inputs and their hashes, the entry checks and the output folder (PG7), the
schema (PG6) and the determinism of two single-thread processes (DT1). Expected values are
written here by hand from the fixtures' own numbers.

The computed values of the later definitions (the fractions, ``run_seconds``, the shifts,
residuals and scores, the area a FOV owns and the density, the reads in cells and the
``counted`` population; ``report.json`` ``summary`` and ``rules``) are not made by the
collection: such a cell is null here exactly when every input it is computed from is
available (``PENDING`` below), and every other null cell is not applicable or named by one
row of ``unavailable.csv``. The tests that pin those placeholders say so.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pandas as pd
import pytest

from starfinder.io._checkpoint import _read_table
from starfinder.report import (PROBLEM_RULES, REPORT_PARTS, REPORT_TABLES, UNAVAILABLE_REASONS, MarkerGroup,
                               ReportConfig, ReportLimits, RuleLimit, read_report)

from .report_fixtures import (CELLS_RETAINED, CHANNELS, COUNTS, ENDED, FOVS, MOVING, NOMINAL, ORIGINS, RECORD_VARIANTS,
                              SAMPLE, STARTED, raw_path, report_cells, report_records, write_inputs)

pytestmark = [pytest.mark.report, pytest.mark.validation]

COLLECTED = ("fovs", "registration", "steps", "readout_totals", "population", "problematic_fovs", "unavailable")
EXPORT_COLUMNS = ("n_cells", "n_cells_retained", "n_molecules", "n_molecules_retained", "reads_in_cells",
                  "reads_in_cells_fraction", "owned_area_um2", "cell_density_per_mm2")
PLACEMENT_COLUMNS = ("placement_status", "correction_z", "correction_y", "correction_x", "footprint_y0",
                     "footprint_y1", "footprint_x0", "footprint_x1", "centre_y", "centre_x")
READ_COLUMNS = ("spots", "correct_form", "matched", "merged_by_deduplication", "passed_filter",
                "correct_form_fraction", "matched_fraction", "merged_fraction", "passed_filter_fraction")
STATUS_COLUMNS = ("status", "failed_step", "failed_round", "started_at", "ended_at", "run_seconds")
REGISTRATION_COLUMNS = ("outcome", "fallback", "methods", "transform_source", "shift_z", "shift_y", "shift_x",
                        "residual_z", "residual_y", "residual_x", "alignment_score")
EXPORT = "export/s1/export.json"


def anndata_or_skip():
    pytest.importorskip("anndata", reason="report_cells writes sample.h5ad, which needs the anndata extra")


def same(actual, expected):
    """Frames equal cell for cell, dtypes included: the fixtures' floats are dyadic, so exactly."""
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)


def tables_of(folder, record):
    """Every table of the record's files, read back with the dtypes it lists."""
    return {Path(e["path"]).stem: _read_table(Path(folder) / e["path"], e["dtypes"]) for e in record["files"]}


def keyed(rows):
    """(part, table, column, row_key, reason) of the rows of unavailable.csv."""
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    return [tuple(None if pd.isna(v) else v for v in r)
            for r in frame[["part", "table", "column", "row_key", "reason"]].itertuples(index=False)]


def collect(case, **changes):
    folder, record = case.collect(**changes)
    return folder, record, tables_of(folder, record)


@pytest.fixture
def records(tmp_path):
    return report_records(tmp_path / "root")


# --- The contract's lists, written here from docs/report-contract.md -------------------------------------

CONTRACT_REASONS = ["no_run_record", "run_not_finished", "step_failed", "not_configured", "not_recorded",
                    "no_placement", "fov_not_placed", "no_export", "part_not_written", "no_segmentation",
                    "no_markers", "no_stack", "input_missing", "input_changed", "field_not_saved", "replay_differs",
                    "undefined"]
# "Parts, their inputs and their reasons": the reasons each part can have (in the order of the contract's rows).
PART_REASONS = {
    "summary.identity": {"no_run_record", "not_recorded"},
    "summary.key_numbers": {"no_run_record", "run_not_finished", "step_failed", "not_configured", "no_export",
                            "part_not_written", "no_segmentation", "undefined"},
    "workflow.steps": {"no_run_record", "no_placement", "no_export"},
    "workflow.fov_status": {"no_run_record"},
    "workflow.time": {"no_run_record", "run_not_finished", "not_recorded"},
    "registration.maps": {"no_run_record", "step_failed", "not_configured", "not_recorded", "no_placement",
                          "fov_not_placed", "undefined", "run_not_finished"},
    "registration.heatmap": {"no_run_record", "step_failed", "not_configured", "not_recorded", "undefined",
                             "run_not_finished"},
    "readout.totals": {"no_run_record", "run_not_finished", "step_failed", "not_configured", "not_recorded",
                       "no_export", "no_segmentation"},
    "readout.per_fov": {"no_run_record", "run_not_finished", "step_failed", "not_configured", "not_recorded",
                        "no_placement", "fov_not_placed", "undefined"},
    "assembly.cells_per_fov": {"no_export", "part_not_written", "no_segmentation", "no_placement", "undefined",
                               "fov_not_placed", "input_missing", "input_changed"},
    "assembly.placement": {"no_placement", "fov_not_placed", "input_missing", "input_changed"},
    "assembly.accounting": {"no_export", "part_not_written", "no_segmentation"},
    "files.input": {"no_run_record", "not_recorded", "no_placement"},
    "files.configuration": {"no_run_record", "not_recorded"},
}
# "The tables": each column's type, and whether its null rule is "never".
CONTRACT_COLUMNS = {
    "fovs": [("fov_id", "string", True), ("list_index", "int64", True), ("example", "string", False),
             ("run_record", "boolean", True), ("status", "string", False), ("failed_step", "string", False),
             ("failed_round", "string", False), ("started_at", "string", False), ("ended_at", "string", False),
             ("run_seconds", "float64", False), ("placement_status", "string", False),
             *((c, "float64", False) for c in ("correction_z", "correction_y", "correction_x", "footprint_y0",
                                                "footprint_y1", "footprint_x0", "footprint_x1", "centre_y",
                                                "centre_x")),
             *((c, "int64", False) for c in ("spots", "correct_form", "matched", "merged_by_deduplication",
                                              "passed_filter")),
             *((c, "float64", False) for c in ("correct_form_fraction", "matched_fraction", "merged_fraction",
                                                "passed_filter_fraction")),
             *((c, "int64", False) for c in ("n_cells", "n_cells_retained", "n_molecules", "n_molecules_retained",
                                              "reads_in_cells")),
             ("reads_in_cells_fraction", "float64", False), ("owned_area_um2", "float64", False),
             ("cell_density_per_mm2", "float64", False)],
    "registration": [("fov_id", "string", True), ("round", "string", True), ("outcome", "string", False),
                     ("fallback", "boolean", False), ("methods", "string", False),
                     ("transform_source", "string", False),
                     *((c, "float64", False) for c in ("shift_z", "shift_y", "shift_x", "residual_z", "residual_y",
                                                        "residual_x", "alignment_score"))],
    "steps": [("scope", "string", True), ("fov_id", "string", False), ("index", "int64", True),
              ("name", "string", True), ("round", "string", False), ("method", "string", False),
              ("status", "string", True), ("seconds", "float64", False)],
    "readout_totals": [("level", "string", True), ("count", "int64", False), ("fraction_of_spots", "float64", False),
                       ("fovs", "int64", True), ("overlaps_counted_per_fov", "boolean", True)],
    "population": [("population", "string", True), ("cells", "int64", False), ("molecules", "int64", False)],
    "problematic_fovs": [("fov_id", "string", True), ("section", "string", True), ("rule", "string", True),
                         ("round", "string", False), ("axis", "string", False), ("measured", "float64", False),
                         ("sample_median", "float64", False), ("limit", "float64", False),
                         ("unit", "string", False), ("reason", "string", True)],
    "unavailable": [("part", "string", True), ("table", "string", False), ("column", "string", False),
                    ("row_key", "string", True), ("reason", "string", True), ("detail", "string", False)],
}
KEYS = {"fovs": ["fov_id"], "registration": ["fov_id", "round"], "steps": ["scope", "fov_id", "index"],
        "readout_totals": ["level"], "population": ["population"],
        "problematic_fovs": ["fov_id", "rule", "round", "axis"], "unavailable": ["part", "table", "column", "row_key"]}
RECORD_KEYS = ["format_version", "stage", "sample", "dataset_id", "sample_id", "fov_ids", "route", "assignment",
               "segmentation", "generated_at", "software", "config", "stages", "inputs", "files", "unavailable",
               "rules", "summary", "wall_s", "cpu_s", "peak_rss_kib"]


def pandas_dtype(kind, never):
    """The checkpoint dtype of a contract type: a nullable dtype exactly when the column can be null."""
    return {"string": "string", "float64": "float64", "int64": "int64" if never else "Int64",
            "boolean": "bool" if never else "boolean"}[kind]


def key_of(table, row):
    return ";".join(f"{k}={'' if pd.isna(row[k]) else row[k]}" for k in KEYS[table])


# --- RC1: a normal sample ---------------------------------------------------------------------------------

def expected_fovs():
    rows = []
    for i, fov in enumerate(FOVS):
        (oz, oy, ox), (nz, ny, nx) = ORIGINS[fov], NOMINAL[fov]
        rows.append({"fov_id": fov, "list_index": i, "example": None, "run_record": True, "status": "succeeded",
                     "failed_step": None, "failed_round": None, "started_at": STARTED[fov], "ended_at": ENDED[fov],
                     "run_seconds": None, "placement_status": "image_supported",
                     "correction_z": oz - nz, "correction_y": oy - ny, "correction_x": ox - nx,
                     # footprint_yx: [o − s/2, o + (n − ½)·s), centre o + (n − 1)/2 · s; s = 0.25, n = 16.
                     "footprint_y0": oy - 0.125, "footprint_y1": oy + 15.5 * 0.25,
                     "footprint_x0": ox - 0.125, "footprint_x1": ox + 15.5 * 0.25,
                     "centre_y": oy + 7.5 * 0.25, "centre_x": ox + 7.5 * 0.25,
                     "spots": COUNTS["spots"][fov], "correct_form": COUNTS["endpoint_valid"][fov],
                     "matched": COUNTS["assigned"][fov], "merged_by_deduplication": COUNTS["merged_reads"][fov],
                     "passed_filter": COUNTS["accepted"][fov], "correct_form_fraction": None,
                     "matched_fraction": None, "merged_fraction": None, "passed_filter_fraction": None,
                     **{c: None for c in EXPORT_COLUMNS}})
    return rows


def frame_of(table, rows):
    columns = CONTRACT_COLUMNS[table]
    return pd.DataFrame({c: pd.Series([r[c] for r in rows], dtype=object).astype(pandas_dtype(k, n))
                         for c, k, n in columns}, columns=[c for c, _, _ in columns])


def expected_registration(source="run.json"):
    return [{"fov_id": fov, "round": name, "outcome": "succeeded", "fallback": False, "methods": "translation",
             "transform_source": source, **{c: None for c in REGISTRATION_COLUMNS[4:]}}
            for fov in FOVS for name in MOVING]


def expected_steps(*, deduplication=True, registration=True, placement=True):
    rows = []
    for fov in FOVS:
        steps = [("load_images", None, None, 0.5), ("preprocess:min_max_normalization", "round1",
                                                     "min_max_normalization", 0.25),
                 ("preprocess:white_tophat", "round1", "white_tophat", 0.25),
                 ("find_spots", None, "local_maxima", 1.0), ("extract_round", "round1", None, 0.125)]
        for name in MOVING:
            steps += [("preprocess:min_max_normalization", name, "min_max_normalization", 0.25),
                      ("preprocess:white_tophat", name, "white_tophat", 0.25)]
            if registration:
                steps.append(("register", name, "translation", 0.5))
            steps.append(("extract_round", name, None, 0.125))
        steps += [("decode_barcodes", None, "wta (multiplexed)", 0.25), ("score_reads", None, None, 0.125)]
        if deduplication:
            steps.append(("deduplicate_reads", None, None, 0.125))
        steps.append(("filter_reads", None, "filter", 0.125))
        rows += [{"scope": "fov", "fov_id": fov, "index": i, "name": n, "round": r, "method": m,
                  "status": "succeeded", "seconds": s} for i, (n, r, m, s) in enumerate(steps)]
    if placement:
        rows.append({"scope": "sample", "fov_id": None, "index": 0, "name": "stitch", "round": None,
                     "method": "constructed", "status": "succeeded", "seconds": None})
    return rows


def expected_totals(fovs=FOVS, merged=True):
    rows = []
    for level, key in (("spots", "spots"), ("correct_form", "endpoint_valid"), ("matched", "assigned"),
                       ("merged_by_deduplication", "merged_reads"), ("passed_filter", "accepted")):
        present = merged or level != "merged_by_deduplication"
        rows.append({"level": level, "count": sum(COUNTS[key][f] for f in fovs) if present else None,
                     "fraction_of_spots": None, "fovs": len(fovs) if present else 0,
                     "overlaps_counted_per_fov": True})
    rows.append({"level": "reads_in_cells", "count": None, "fraction_of_spots": None, "fovs": 0,
                 "overlaps_counted_per_fov": False})
    return rows


def no_export_rows(fovs=FOVS):
    """The rows of every value of the collected tables that needs the export, in the contract's order."""
    rows = [("readout.totals", "readout_totals", c, "level=reads_in_cells", "no_export")
            for c in ("count", "fraction_of_spots")]
    rows += [("assembly.cells_per_fov", "fovs", c, f"fov_id={f}", "no_export") for c in EXPORT_COLUMNS for f in fovs]
    cells = ("input", "retained", "excluded_not_owner", "excluded_fov_not_placed", "excluded_duplicate_observation",
             "counted")
    molecules = ("input", "retained", "excluded_not_owner", "excluded_fov_not_placed", "excluded_cell_not_owner",
                 "excluded_cell_duplicate_observation", "excluded_outside_grid", "counted")
    rows += [("assembly.accounting", "population", "cells", f"population={p}", "no_export") for p in cells]
    rows += [("assembly.accounting", "population", "molecules", f"population={p}", "no_export") for p in molecules]
    return rows


STITCH_TIME = ("workflow.time", "steps", "seconds", "scope=sample;fov_id=;index=0", "not_recorded")


def ordered(rows):
    """Rows in the contract's order: parts, rows without a table first, tables, columns, rows."""
    tables = {name: i for i, name in enumerate(REPORT_TABLES)}
    columns = {t: [c for c, _, _ in cols] for t, cols in CONTRACT_COLUMNS.items()}
    return sorted(rows, key=lambda r: (REPORT_PARTS.index(r[0]), -1 if r[1] is None else tables[r[1]],
                                       0 if r[1] is None else columns[r[1]].index(r[2])))


def test_rc1_normal_sample(records):
    folder, record, tables = collect(records)
    same(tables["fovs"], frame_of("fovs", expected_fovs()))
    same(tables["registration"], frame_of("registration", expected_registration()))
    same(tables["steps"], frame_of("steps", expected_steps()))
    same(tables["readout_totals"], frame_of("readout_totals", expected_totals()))
    assert tables["readout_totals"]["count"].tolist()[:5] == [4000, 2960, 2800, 40, 2590]
    # Placeholder of W-453 (the rules): the table is written with its header only.
    assert list(tables["problematic_fovs"].columns) == [c for c, _, _ in CONTRACT_COLUMNS["problematic_fovs"]]
    assert tables["problematic_fovs"].empty
    assert keyed(tables["unavailable"]) == ordered([STITCH_TIME] + no_export_rows())
    assert all(isinstance(d, str) and d for d in tables["unavailable"]["detail"])


@pytest.mark.parametrize("variant", ["stage_off", "stage_off_registration"])
def test_rc4_stage_not_run(tmp_path, variant):
    case = report_records(tmp_path / "root", variant)
    _, _, tables = collect(case)
    rows = keyed(tables["unavailable"])
    fovs, totals = tables["fovs"], tables["readout_totals"].set_index("level")
    if variant == "stage_off":
        assert fovs["merged_by_deduplication"].isna().all() and fovs["merged_fraction"].isna().all()
        for column in ("merged_by_deduplication", "merged_fraction"):
            assert [r for r in rows if r[2] == column] == [
                ("readout.per_fov", "fovs", column, f"fov_id={f}", "not_configured") for f in FOVS]
        assert pd.isna(totals.loc["merged_by_deduplication", "count"])
        assert totals.loc["merged_by_deduplication", "fovs"] == 0
        assert ("readout.totals", "readout_totals", "count", "level=merged_by_deduplication",
                "not_configured") in rows
        same(tables["readout_totals"], frame_of("readout_totals",
                                                                         expected_totals(merged=False)))
        same(tables["steps"], frame_of("steps", expected_steps(deduplication=False)))
        same(tables["registration"], frame_of("registration", expected_registration()))
    else:
        assert tables["registration"].empty
        assert ("registration.maps", None, None, "", "not_configured") in rows
        assert ("registration.heatmap", None, None, "", "not_configured") in rows
        assert not any(r[1] == "registration" for r in rows)
        same(tables["steps"], frame_of("steps", expected_steps(registration=False)))
        same(tables["fovs"], frame_of("fovs", expected_fovs()))


# --- RC2, RC13, RC3, RC5, RC11 -----------------------------------------------------------------------------

def test_rc2_failed_fov(tmp_path):
    case = report_records(tmp_path / "root", "failed")
    _, _, tables = collect(case)
    f3 = tables["fovs"].set_index("fov_id").loc["F3"]
    assert (f3["status"], f3["failed_step"], f3["failed_round"]) == ("failed", "register", "round3")
    registration = tables["registration"].set_index(["fov_id", "round"])
    assert registration.loc[("F3", "round2"), "outcome"] == "succeeded"
    assert registration.loc[("F3", "round2"), "transform_source"] == "run.json"
    assert registration.loc[("F3", "round3")].isna().all()
    rows = keyed(tables["unavailable"])
    assert [r for r in rows if r[3] == "fov_id=F3;round=round3"] == [
        ("registration.heatmap", "registration", c, "fov_id=F3;round=round3", "step_failed")
        for c in REGISTRATION_COLUMNS]
    assert [r for r in rows if r[3] == "fov_id=F3" and r[0] == "readout.per_fov"] == [
        ("readout.per_fov", "fovs", c, "fov_id=F3", "step_failed") for c in READ_COLUMNS]
    totals = tables["readout_totals"].set_index("level")
    assert totals.loc["spots", "count"] == 1000 + 1200 + 1000 and totals.loc["spots", "fovs"] == 3
    steps = tables["steps"]
    f3_steps = steps[steps["fov_id"].eq("F3").fillna(False).to_numpy(bool)]
    assert f3_steps["name"].iloc[-1] == "register" and f3_steps["status"].iloc[-1] == "failed"


@pytest.mark.parametrize("variant,outcome,methods", [("failed_applying", "application_failed", "translation"),
                                                     ("failed_estimating", "failed", None),
                                                     ("failed_succeeded", "failed", "translation")])
def test_rc13_round_that_did_not_finish(tmp_path, variant, outcome, methods):
    case = report_records(tmp_path / "root", variant)
    _, _, tables = collect(case)
    row = tables["registration"].set_index(["fov_id", "round"]).loc[("F3", "round3")]
    assert row["outcome"] == outcome
    assert (None if pd.isna(row["methods"]) else row["methods"]) == methods
    assert bool(row["fallback"]) is False
    rows = keyed(tables["unavailable"])
    key = "fov_id=F3;round=round3"
    named = [r[2] for r in rows if r[3] == key]
    assert "outcome" not in named and "fallback" not in named and "methods" not in named
    assert pd.isna(row["alignment_score"])
    assert [r for r in rows if r[2] == "alignment_score"] == [
        ("registration.heatmap", "registration", "alignment_score", key, "step_failed")]
    assert named == ["transform_source", "shift_z", "shift_y", "shift_x", "residual_z", "residual_y", "residual_x",
                     "alignment_score"]
    assert {r[4] for r in rows if r[3] == key} == {"step_failed"}


@pytest.mark.parametrize("variant", ["not_run", "only_f1"])
def test_rc3_fov_not_run(tmp_path, variant):
    case = report_records(tmp_path / "root", variant)
    _, record, tables = collect(case)
    absent = ["F4"] if variant == "not_run" else ["F2", "F3", "F4"]
    present = [f for f in FOVS if f not in absent]
    fovs = tables["fovs"].set_index("fov_id")
    rows = keyed(tables["unavailable"])
    for fov in absent:
        assert not fovs.loc[fov, "run_record"]
        assert fovs.loc[fov, list(STATUS_COLUMNS + READ_COLUMNS)].isna().all()
        assert [r for r in rows if r[3] == f"fov_id={fov}" and r[0] != "assembly.cells_per_fov"] == [
            ("workflow.fov_status", "fovs", c, f"fov_id={fov}", "no_run_record") for c in STATUS_COLUMNS] + [
            ("readout.per_fov", "fovs", c, f"fov_id={fov}", "no_run_record") for c in READ_COLUMNS]
        for name in MOVING:
            key = f"fov_id={fov};round={name}"
            assert [r for r in rows if r[3] == key] == [
                ("registration.heatmap", "registration", c, key, "no_run_record") for c in REGISTRATION_COLUMNS]
        assert fov not in record["stages"]["run_records"]
    same(tables["readout_totals"], frame_of("readout_totals", expected_totals(present)))
    totals = tables["readout_totals"].set_index("level")
    assert (totals.loc["spots", "count"], totals.loc["matched", "count"], totals.loc["spots", "fovs"]) == (
        (3000, 2100, 3) if variant == "not_run" else (1000, 700, 1))


def test_rc5_no_placement(tmp_path):
    case = report_records(tmp_path / "root", "no_placement")
    _, record, tables = collect(case)
    fovs = tables["fovs"]
    assert fovs[list(PLACEMENT_COLUMNS)].isna().all().all()
    rows = keyed(tables["unavailable"])
    for column in PLACEMENT_COLUMNS:
        assert [r for r in rows if r[2] == column] == [
            ("assembly.placement", "fovs", column, f"fov_id={f}", "no_placement") for f in FOVS]
    assert ("registration.maps", None, None, "", "no_placement") in rows
    assert ("readout.per_fov", None, None, "", "no_placement") in rows
    assert ("files.input", None, None, "entry=positions", "no_placement") in rows
    # The heatmap and the tables are present; the area is no_placement (before no_export), the density too.
    assert ("registration.heatmap", None, None, "", "no_placement") not in rows
    assert not tables["registration"].empty
    assert [r[4] for r in rows if r[2] in ("owned_area_um2", "cell_density_per_mm2")] == ["no_placement"] * 8
    assert not any(r[0] == "workflow.time" and r[3].startswith("scope=sample") for r in rows)
    assert record["stages"]["placement"] is False


@pytest.mark.parametrize("variant", ["older", "older_transforms"])
def test_rc11_older_record(tmp_path, variant):
    case = report_records(tmp_path / "root", variant)
    _, record, tables = collect(case)
    registration = tables["registration"].set_index(["fov_id", "round"])
    rows = keyed(tables["unavailable"])
    with_file = {"F1"} if variant == "older_transforms" else set()
    for fov in FOVS:
        for name in MOVING:
            key = f"fov_id={fov};round={name}"
            if fov in with_file:
                assert registration.loc[(fov, name), "transform_source"] == "transforms.json"
                assert not [r for r in rows if r[3] == key]
            else:
                assert pd.isna(registration.loc[(fov, name), "transform_source"])
                assert [r for r in rows if r[3] == key] == [
                    ("registration.heatmap", "registration", c, key, "not_recorded")
                    for c in REGISTRATION_COLUMNS[3:10]]
    assert [r for r in rows if r[2] == "correct_form"] == [
        ("readout.per_fov", "fovs", "correct_form", f"fov_id={f}", "not_recorded") for f in FOVS]
    assert ("files.input", None, None, "entry=codebook", "not_recorded") in rows
    assert ("files.configuration", None, None, "entry=workflow", "not_recorded") in rows
    kinds = {e["path"]: e["kind"] for e in record["inputs"]}
    assert ("checkpoints/F1/registered/transforms.json" in kinds) == (variant == "older_transforms")


# --- RC12: every null cell is n/a or named once -------------------------------------------------------------

def _pending(table, row, frame, has_export):
    """Whether a null cell is a computed value of a later definition whose inputs are all available."""
    def available(column):
        return not pd.isna(row[column])
    if table == "fovs":
        fractions = {"correct_form_fraction": "correct_form", "matched_fraction": "matched",
                     "merged_fraction": "merged_by_deduplication", "passed_filter_fraction": "passed_filter"}
        owned = available("correction_z") and has_export
        return {"run_seconds": available("started_at") and available("ended_at"),
                **{c: available(fractions[c]) and available("spots") for c in fractions},
                "reads_in_cells": available("n_cells"),
                "reads_in_cells_fraction": available("n_cells") and available("n_molecules_retained"),
                "owned_area_um2": owned, "cell_density_per_mm2": owned and available("n_cells_retained")}
    if table == "registration":
        return {c: available("transform_source") for c in REGISTRATION_COLUMNS[4:10]} | {
            "alignment_score": row["outcome"] == "succeeded"}
    if table == "readout_totals":
        spots = frame.set_index("level").loc["spots", "count"]
        in_cells = row["level"] == "reads_in_cells"
        return {"count": in_cells and has_export,
                "fraction_of_spots": (available("count") or (in_cells and has_export)) and not pd.isna(spots)}
    if table == "population":
        return {"cells": row["population"] == "counted" and has_export,
                "molecules": row["population"] == "counted" and has_export}
    return {}


WITHOUT_METHOD = ("load_images", "extract_round", "score_reads", "deduplicate_reads")


def _not_applicable(table, column, row, case):
    """The column's null rule names this case n/a (docs/report-contract.md, "The tables"), read from the records."""
    if table == "fovs":
        record = case.run_record(row["fov_id"]) if row["run_record"] else None
        error = (record or {}).get("error")
        running = record is not None and record["status"] == "running"
        return {"example": True, "failed_step": record is not None and error is None,
                "failed_round": record is not None and (error is None or error.get("round") is None),
                "ended_at": running, "run_seconds": running}.get(column, False)
    if table == "registration":
        if column != "methods" or not row["run_record_present"]:
            return False
        entries = case.run_record(row["fov_id"])["registration"].get(row["round"]) or []
        return bool(entries) and not any(e["record"] == "estimation" and e["outcome"] == "succeeded"
                                         for e in entries)
    if table == "steps":
        if row["scope"] == "sample":
            return {"fov_id": True, "round": True, "method": row["name"] != "stitch",
                    "seconds": row["status"] in ("off", "not_run")}.get(column, False)
        if row["name"] in ("segment", "assign"):
            return column == "round"
        record = case.run_record(row["fov_id"])
        step = record["steps"][row["index"]]
        if column == "method" and row["name"] == "register":
            # A round with no succeeded estimation has no method (registration.csv methods n/a or unavailable).
            entries = record["registration"].get(step["round"]) or []
            return not any(e["record"] == "estimation" and e["outcome"] == "succeeded" for e in entries)
        return {"round": step["round"] is None, "method": row["name"] in WITHOUT_METHOD}.get(column, False)
    if table == "population":
        population = row["population"]
        cells_only = population in ("excluded_cell_not_owner", "excluded_cell_duplicate_observation",
                                    "excluded_outside_grid")
        return (column == "cells" and cells_only) or (column == "molecules" and
                                                       population == "excluded_duplicate_observation")
    return False


CASES = [("report_records", None)] + [("report_records", v) for v in RECORD_VARIANTS] + [
    ("report_cells", None), ("report_cells", "nocal"), ("report_cells", "mismatch"), ("report_cells", "no_markers")]


@pytest.mark.parametrize("maker,variant", CASES, ids=[f"{m}-{v}" for m, v in CASES])
def test_rc12_null_cells_and_closed_lists(tmp_path, maker, variant):
    if maker == "report_cells":
        anndata_or_skip()
    case = (report_cells if maker == "report_cells" else report_records)(tmp_path / "root", variant)
    _, record, tables = collect(case)
    unavailable = tables["unavailable"]
    named = {}
    for row in unavailable.to_dict("records"):
        assert row["reason"] in UNAVAILABLE_REASONS
        assert row["part"] in REPORT_PARTS or (row["part"].startswith("rule:") and row["part"][5:] in PROBLEM_RULES)
        assert row["reason"] in PART_REASONS[row["part"]], row
        if not pd.isna(row["table"]):
            key = (row["table"], row["column"], row["row_key"])
            assert key not in named, f"two rows name {key}"
            named[key] = row
    assert len(set(keyed(unavailable))) == len(unavailable)
    found = set()
    has_export = (case.root / EXPORT).is_file()
    present = set(record["stages"]["run_records"])
    for table in ("fovs", "registration", "steps", "readout_totals", "population"):
        frame = tables[table]
        for row in frame.to_dict("records"):
            key = key_of(table, row)
            pending = _pending(table, row, frame, has_export)
            row["run_record_present"] = row.get("fov_id") in present
            for column in frame.columns:
                if not pd.isna(row[column]):
                    assert (table, column, key) not in named, f"{table}.{column} [{key}] holds a value and a row"
                    continue
                if (table, column, key) in named:
                    found.add((table, column, key))
                elif pending.get(column, False):
                    continue  # computed by a later definition from available inputs
                else:
                    assert _not_applicable(table, column, row, case), (
                        f"{table}.{column} [{key}] is null, not applicable by no rule and named by no row")
    assert found == set(named), set(named) - found
    # Placeholders of W-453 and W-455 in report.json, pinned here: no summary key and no rule yet.
    assert record["summary"] == {} and record["rules"] == {}


# --- Inputs and their hashes ----------------------------------------------------------------------------------

def change_byte(path: Path):
    data = bytearray(path.read_bytes())
    data[-2] = (data[-2] + 1) % 256
    path.write_bytes(bytes(data))


def write_assignment_folder(case, fov):
    """A saved assignment folder of the route per FOV, as assignment/_persist.py lays it out (hand-written)."""
    folder = case.checkpoint(fov) / "assignment" / "default"
    folder.mkdir(parents=True)
    (folder / "cells.csv").write_text("cell_id,status\n1,kept\n")
    digest = hashlib.sha256((folder / "cells.csv").read_bytes()).hexdigest()
    record = {"format_version": 1, "stage": "assignment", "name": "default",
              "inputs": {"cells": {"run": "cell", "file": {"path": "cell_labels.tif", "sha256": None}},
                         "nuclei": None},
              "files": {"cells": {"path": "cells.csv", "sha256": digest, "dtypes": {}}}}
    (folder / "assignment.json").write_text(json.dumps(record))
    run = case.checkpoint(fov) / "segmentation" / "cell"
    run.mkdir(parents=True)
    (run / "segmentation.json").write_text(json.dumps({"run": "cell", "methods": [{"method": "watershed"}]}))
    return folder


def inputs_case(root, edit=None):
    """report_cells with its raw inputs and an assignment folder per FOV, one file edited; its collection."""
    case = report_cells(root)
    write_inputs(case.root)
    for fov in FOVS:
        write_assignment_folder(case, fov)
    if edit is not None:
        edit(case)
    return case, *collect(case)


@pytest.fixture(scope="module")
def inputs_base(tmp_path_factory):
    anndata_or_skip()
    return inputs_case(tmp_path_factory.mktemp("inputs") / "root")


def test_inputs_lists_every_file_read(inputs_base):
    base, _, record, tables = inputs_base
    for entry in record["inputs"]:
        path = base.root / entry["path"]
        assert set(entry) == {"path", "kind", "sha256", "bytes", "verified"}
        assert entry["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert entry["bytes"] == path.stat().st_size
    kinds = {e["path"]: (e["kind"], e["verified"]) for e in record["inputs"]}
    assert kinds["export/s1/tables/fovs.csv"] == ("export_table", True)
    assert kinds["stitching/s1/fovs.csv"] == ("placement_table", True)
    assert kinds["codebook.csv"] == ("codebook", True)
    assert kinds["checkpoints/F1/run.json"] == ("run_record", False)
    assert kinds["checkpoints/F1/assignment/default/assignment.json"] == ("assignment_record", False)
    assert kinds["checkpoints/F1/segmentation/cell/segmentation.json"] == ("segmentation_record", False)
    steps = tables["steps"]
    lane = steps[steps["scope"].eq("fov").to_numpy(bool) & steps["name"].isin(["segment", "assign"]).to_numpy(bool)]
    assert lane[["fov_id", "name", "method"]].values.tolist() == [
        [f, n, m] for f in FOVS for n, m in (("segment", "watershed"), ("assign", "cell"))]


@pytest.mark.parametrize("reason,edit", [
    ("input_changed", lambda c: change_byte(c.root / "export/s1/tables/fovs.csv")),
    ("input_missing", lambda c: (c.root / "export/s1/tables/fovs.csv").unlink())], ids=["changed", "missing"])
def test_inputs_export_table(tmp_path, inputs_base, reason, edit):
    """The export's fovs table: the columns copied from it and the values computed from them, on no other part."""
    _, _, _, base_tables = inputs_base
    case, _, record, tables = inputs_case(tmp_path / "root", edit)
    base_rows, rows = keyed(base_tables["unavailable"]), keyed(tables["unavailable"])
    expected = [("assembly.cells_per_fov", "fovs", c, f"fov_id={f}", reason)
                for c in ("n_cells", "n_cells_retained", "n_molecules", "n_molecules_retained",
                          "reads_in_cells_fraction", "cell_density_per_mm2") for f in FOVS]
    # The footprints and centres read the export's fovs table first (the placement status, the placement).
    expected += [("assembly.placement", "fovs", c, f"fov_id={f}", reason) for c in PLACEMENT_COLUMNS[4:]
                 for f in FOVS]
    assert [r for r in rows if r not in base_rows] == expected
    assert [r for r in base_rows if r not in rows] == []
    assert tables["fovs"][["n_cells", "n_cells_retained", "n_molecules", "n_molecules_retained",
                           *PLACEMENT_COLUMNS[4:]]].isna().all().all()
    same(tables["fovs"][list(PLACEMENT_COLUMNS[:4])],
                                  base_tables["fovs"][list(PLACEMENT_COLUMNS[:4])])
    for other in ("registration", "steps", "readout_totals", "population"):
        same(tables[other], base_tables[other])
    entry = {e["path"]: e for e in record["inputs"]}.get("export/s1/tables/fovs.csv")
    if reason == "input_changed":
        data = (case.root / "export/s1/tables/fovs.csv").read_bytes()
        assert entry == {"path": "export/s1/tables/fovs.csv", "kind": "export_table",
                         "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), "verified": True}
    else:
        assert entry is None


@pytest.mark.parametrize("edit", [
    lambda c: change_byte(raw_path(c.root, "round1", "F1", CHANNELS[0])),
    lambda c: raw_path(c.root, "round1", "F1", CHANNELS[0]).unlink(),
    lambda c: change_byte(c.checkpoint("F1") / "assignment/default/cells.csv"),
    lambda c: (c.checkpoint("F1") / "assignment/default/cells.csv").unlink()],
    ids=["raw_changed", "raw_missing", "assignment_changed", "assignment_missing"])
def test_inputs_files_no_collected_part_reads(tmp_path, inputs_base, edit):
    """A run record's raw input and an assignment folder's table: no collected part reads them (the replay and the
    segmentation examples do), so the tables, the rows and the inputs are those of the unchanged records."""
    _, _, base_record, base_tables = inputs_base
    _, _, record, tables = inputs_case(tmp_path / "root", edit)
    assert keyed(tables["unavailable"]) == keyed(base_tables["unavailable"])
    for table in ("fovs", "registration", "steps", "readout_totals", "population"):
        same(tables[table], base_tables[table])
    assert [(e["path"], e["kind"]) for e in record["inputs"]] == [(e["path"], e["kind"])
                                                                 for e in base_record["inputs"]]


def test_inputs_placement_table(tmp_path):
    """The placement table, read without an export: the placement columns, on no other part."""
    case = report_records(tmp_path / "root")
    _, _, base = collect(report_records(tmp_path / "base"))
    change_byte(case.root / "stitching/s1/fovs.csv")
    _, _, tables = collect(case)
    rows, base_rows = keyed(tables["unavailable"]), keyed(base["unavailable"])
    assert [r for r in rows if r not in base_rows] == [
        ("assembly.placement", "fovs", c, f"fov_id={f}", "input_changed") for c in PLACEMENT_COLUMNS for f in FOVS]
    # The area a FOV owns needs the export first (no_export), so its row and the density's keep their reason.
    assert [r for r in base_rows if r not in rows] == []


def test_codebook_changed_falls_back_to_the_export_genes(tmp_path):
    anndata_or_skip()
    case = report_cells(tmp_path / "root")
    change_byte(case.root / "codebook.csv")
    _, record, _ = collect(case)
    assert {e["path"]: e["verified"] for e in record["inputs"]}["codebook.csv"] is True
    case.config = ReportConfig(markers=(MarkerGroup("x", ("G9",)),))
    with pytest.raises(ValueError, match=r"'G9' is not in the gene list of .*export\.json"):
        collect(case)


# --- PG7: the entry checks and the output folder -----------------------------------------------------------------

def snapshot(folder: Path):
    return {p.relative_to(folder).as_posix(): p.read_bytes() for p in sorted(folder.rglob("*")) if p.is_file()} \
        if folder.exists() else None


def _check3(case):
    record = case.run_record("F2")
    record["dataset_id"] = "other"
    case.write_run_record("F2", record)


def _check4(case):
    record = json.loads((case.root / EXPORT).read_text())
    record["sample"] = "other"
    (case.root / EXPORT).write_text(json.dumps(record))


def _check7(case):
    (case.folder / "notes.txt").write_text("mine")


# One violation per check of "Checks before anything is written": (check, error, message, files changed,
# arguments changed). Check 2's needs a FOV list without run records and no export, so it replaces check 3's and
# check 4's (whose records it then never reads); check 5's configuration also names an example outside the list.
CHECKS = [
    (1, TypeError, "config must be a ReportConfig", None, dict(config="defaults")),
    (2, FileNotFoundError, "nothing to report", lambda c: shutil.rmtree(c.root / "export"),
     dict(fov_ids=["X1", "X2"])),
    (3, ValueError, r"checkpoints/F2/run\.json: dataset_id is 'other'", _check3, {}),
    (4, ValueError, r"export\.json: sample is 'other'", _check4, {}),
    (5, ValueError, r"marker gene 'Zz' is not in the gene list of .*codebook\.csv", None,
     dict(config=ReportConfig(markers=(MarkerGroup("m", ("Zz",)),), example_fovs=("F1", "F2", "F9")))),
    (6, ValueError, r"example_fovs \['F9'\] are not FOVs of the list", None,
     dict(config=ReportConfig(example_fovs=("F1", "F2", "F9")))),
    (7, FileExistsError, r"notes\.txt is not a file of the report", _check7, {}),
]


@pytest.mark.parametrize("number", [c[0] for c in CHECKS])
def test_pg7_entry_checks_in_order(tmp_path, number):
    anndata_or_skip()
    case = report_cells(tmp_path / "root")
    case.collect()
    arguments = {}
    # This check's violation and every later one's (the earliest argument change kept), so that each check raises
    # before the later ones and the error is the first check's.
    for later, _, _, edit, changes in CHECKS[number - 1:]:
        if number <= 2 and later in (3, 4):
            continue
        if edit is not None:
            edit(case)
        arguments = changes | arguments
    _, error, message, _, _ = CHECKS[number - 1]
    earlier = snapshot(case.folder)
    with pytest.raises(error, match=message):
        case.collect(**arguments)
    assert snapshot(case.folder) == earlier
    assert [p.name for p in case.folder.parent.iterdir()] == [SAMPLE]


@pytest.mark.parametrize("arguments,error", [(dict(fov_ids=[]), ValueError), (dict(fov_ids=["F1", "F1"]), ValueError),
                                             (dict(fov_ids="F1"), ValueError), (dict(sample=""), ValueError),
                                             (dict(config=ReportLimits()), TypeError)])
def test_pg7_check1_types_and_values(records, arguments, error):
    with pytest.raises(error):
        records.collect(**arguments)
    assert not (records.root / "report").exists()


def test_pg7_check4_export_and_placement(tmp_path):
    anndata_or_skip()
    case = report_cells(tmp_path / "root")
    record = json.loads((case.root / EXPORT).read_text())
    record["fusion"] = {"status": "written", "assembly_sha256": "0" * 64}
    (case.root / EXPORT).write_text(json.dumps(record))
    with pytest.raises(ValueError, match="fails its hashes"):
        case.collect()
    case = report_cells(tmp_path / "fovs")
    with pytest.raises(ValueError, match=r"assembly\.fovs holds the FOVs"):
        case.collect(fov_ids=["F1", "F2", "F3"])
    with pytest.raises(ValueError, match="assembled the assignment"):
        case.collect(assignment="other")
    case = report_records(tmp_path / "placement")
    with pytest.raises(ValueError, match="the placement holds the FOVs"):
        case.collect(fov_ids=["F1", "F2", "F3"])


def test_pg7_markers_without_a_gene_list(tmp_path):
    case = report_records(tmp_path / "root", "older")
    case.config = ReportConfig(marker_pairs=(("G1", "G2"),))
    with pytest.raises(ValueError, match="no gene list to check them against"):
        case.collect()
    assert not case.folder.exists()


def test_pg7_folder_rules_and_replacement(records, monkeypatch):
    folder, first, _ = collect(records)
    assert sorted(snapshot(folder)) == sorted(["report.json"] + [f"tables/{t}.csv" for t in COLLECTED])
    # A second collection replaces the first as a whole, once it is complete.
    record = records.run_record("F1")
    record["counts"]["spots"] = 1001
    records.write_run_record("F1", record)
    _, second, tables = collect(records)
    assert tables["fovs"].set_index("fov_id").loc["F1", "spots"] == 1001
    assert sorted(snapshot(folder)) == sorted(["report.json"] + [f"tables/{t}.csv" for t in COLLECTED])
    assert read_report(folder) == second
    # An error after the checks leaves the earlier report as it was, and no staging folder.
    earlier = snapshot(folder)
    import starfinder.io._checkpoint as checkpoint
    monkeypatch.setattr(checkpoint, "_write_table", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError, match="disk full"):
        records.collect()
    monkeypatch.undo()
    assert snapshot(folder) == earlier
    assert [p.name for p in folder.parent.iterdir()] == [SAMPLE]
    # A folder without report.json is not the report's; an empty folder is used as it is.
    other = records.root / "elsewhere"
    other.mkdir()
    (other / "data.csv").write_text("x\n")
    with pytest.raises(FileExistsError, match=r"data\.csv is not a file of a report"):
        records.collect(output_dir=other)
    assert snapshot(other) == {"data.csv": b"x\n"}
    empty = records.root / "empty"
    empty.mkdir()
    folder, _ = records.collect(output_dir=empty)
    assert folder == empty and (empty / "report.json").is_file()


def test_read_report_checks_the_listed_files(records):
    folder, record = records.collect()
    assert read_report(folder) == record
    change_byte(folder / "tables" / "fovs.csv")
    with pytest.raises(ValueError, match=r"fovs\.csv has SHA-256"):
        read_report(folder)
    (folder / "tables" / "fovs.csv").unlink()
    with pytest.raises(ValueError, match="is missing"):
        read_report(folder)
    with pytest.raises(FileNotFoundError):
        read_report(folder / "tables")


# --- PG6: the schema, with no verdict and no score -------------------------------------------------------------

@pytest.mark.parametrize("maker,variant", [("report_records", None), ("report_records", "failed"),
                                           ("report_records", "no_placement"), ("report_cells", None)])
def test_pg6_schema(tmp_path, maker, variant):
    if maker == "report_cells":
        anndata_or_skip()
    case = (report_cells if maker == "report_cells" else report_records)(tmp_path / "root", variant)
    folder, record, tables = collect(case)
    assert [Path(e["path"]).stem for e in record["files"]] == list(COLLECTED)
    for name in COLLECTED:
        columns = CONTRACT_COLUMNS[name]
        assert list(tables[name].columns) == [c for c, _, _ in columns]
        dtypes = {e["path"]: e["dtypes"] for e in record["files"]}[f"tables/{name}.csv"]
        assert dtypes == {c: pandas_dtype(k, n) for c, k, n in columns}
        assert {c: str(d) for c, d in tables[name].dtypes.items()} == dtypes
        keys = [key_of(name, row) for row in tables[name].to_dict("records")]
        assert len(keys) == len(set(keys)), name
    text = (folder / "report.json").read_text()
    strict = json.loads(text, parse_constant=lambda c: pytest.fail(f"report.json holds {c}"))
    assert list(strict) == RECORD_KEYS
    assert (strict["format_version"], strict["stage"]) == (1, "report")
    assert set(strict["stages"]) == {"run_records", "placement", "export", "parts"}
    for entry in strict["unavailable"]:
        assert list(entry) == ["part", "table", "column", "row_key", "reason", "detail"]
    same(pd.DataFrame(strict["unavailable"], columns=list(tables["unavailable"].columns))
                                  .astype(tables["unavailable"].dtypes.to_dict()), tables["unavailable"])
    for entry in strict["files"]:
        data = (folder / entry["path"]).read_bytes()
        assert (entry["sha256"], entry["bytes"]) == (hashlib.sha256(data).hexdigest(), len(data))
    words = json.dumps(strict).lower()
    assert not any(w in words for w in ('"verdict"', '"score"', '"pass"', '"grade"'))


def test_closed_lists_equal_the_contract():
    assert list(UNAVAILABLE_REASONS) == CONTRACT_REASONS
    assert len(REPORT_PARTS) == 27 and len(set(REPORT_PARTS)) == 27
    assert REPORT_PARTS[:3] == ("summary.identity", "summary.key_numbers", "summary.problematic_fovs")
    assert REPORT_PARTS[-3:] == ("files.input", "files.configuration", "files.output")
    assert REPORT_TABLES == ("fovs", "registration", "steps", "readout_totals", "population", "problematic_fovs",
                             "cells", "mecr", "reads_along_z", "intensity", "unavailable")
    assert PROBLEM_RULES == ("fov_failed", "fov_not_run", "shift_residual", "alignment_score",
                             "registration_rejected", "registration_fallback", "correct_form_fraction",
                             "matched_fraction", "spots_near_zero", "not_placed", "stage_position_fallback",
                             "reads_no_cell")


# --- Configuration ---------------------------------------------------------------------------------------------

def test_configuration_defaults_and_checks():
    config = ReportConfig()
    assert (config.markers, config.marker_pairs, config.positive_threshold, config.example_fovs,
            config.figure_width_px, config.jpeg_quality) == ((), (), 1, None, 1600, 85)
    limits = config.limits
    assert limits.shift_residual == RuleLimit(k=5.0, floor=5.0)
    assert limits.alignment_score == RuleLimit()
    assert limits.correct_form_fraction == limits.matched_fraction == RuleLimit(k=3.0, floor=0.05)
    assert limits.spots_near_zero == 0.05
    config = ReportConfig(markers=[MarkerGroup("a", ["G1", "G2"])], marker_pairs=[["G1", "G2"]],
                          example_fovs=["F1", "F2", "F3"])
    assert config.markers[0].genes == ("G1", "G2") and config.marker_pairs == (("G1", "G2"),)
    assert config.example_fovs == ("F1", "F2", "F3")
    for build, error in [(lambda: MarkerGroup("", ("G1",)), ValueError),
                         (lambda: MarkerGroup("a", ()), ValueError),
                         (lambda: MarkerGroup("a", ("G1", "G2", "G3", "G4", "G5")), ValueError),
                         (lambda: MarkerGroup("a", ("G1", "G1")), ValueError),
                         (lambda: MarkerGroup("a", "G1"), TypeError),
                         (lambda: RuleLimit(k=-1.0), ValueError), (lambda: RuleLimit(floor=float("nan")), ValueError),
                         (lambda: RuleLimit(absolute=True), TypeError),
                         (lambda: ReportLimits(spots_near_zero=0.0), ValueError),
                         (lambda: ReportLimits(spots_near_zero=1.0), ValueError),
                         (lambda: ReportLimits(shift_residual=(5, 5)), TypeError),
                         (lambda: ReportConfig(markers=(MarkerGroup("a", ("G1",)), MarkerGroup("a", ("G2",)))),
                          ValueError),
                         (lambda: ReportConfig(marker_pairs=(("G1", "G1"),)), ValueError),
                         (lambda: ReportConfig(marker_pairs=(("G1",),)), ValueError),
                         (lambda: ReportConfig(positive_threshold=0), ValueError),
                         (lambda: ReportConfig(positive_threshold=1.0), TypeError),
                         (lambda: ReportConfig(example_fovs=("F1", "F1", "F2")), ValueError),
                         (lambda: ReportConfig(example_fovs=("F1", "F2")), ValueError),
                         (lambda: ReportConfig(limits=RuleLimit()), TypeError),
                         (lambda: ReportConfig(figure_width_px=0), ValueError),
                         (lambda: ReportConfig(jpeg_quality=96), ValueError)]:
        with pytest.raises(error):
            build()


# --- DT1: determinism ---------------------------------------------------------------------------------------------

DT1_SCRIPT = """
import sys
from pathlib import Path
from test.report_fixtures import ReportCase, cells_config
ReportCase(Path(sys.argv[1]), config=cells_config()).collect()
"""
LEFT_OUT = ("generated_at", "wall_s", "cpu_s", "peak_rss_kib")


def test_dt1_two_single_thread_processes(tmp_path):
    anndata_or_skip()
    case = report_cells(tmp_path / "root")
    source = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(source), os.environ.get("PYTHONPATH", "")]),
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
               NUMBA_NUM_THREADS="1")
    copies = []
    for run in range(2):
        subprocess.run([sys.executable, "-c", DT1_SCRIPT, str(case.root)], check=True, env=env, cwd=source)
        aside = tmp_path / f"run{run}"
        shutil.copytree(case.folder, aside)
        copies.append(aside)
    first, second = (snapshot(c) for c in copies)
    assert sorted(first) == sorted(second) == sorted(["report.json"] + [f"tables/{t}.csv" for t in COLLECTED])
    for name in first:
        if name != "report.json":
            assert first[name] == second[name], name
    records = [json.loads(s["report.json"]) for s in (first, second)]
    assert records[0]["generated_at"] != records[1]["generated_at"]
    for record in records:
        for key in LEFT_OUT:
            del record[key]
        # No report.html is written yet, so the files list has no entry of it to leave out.
        assert not any(e["path"] == "report.html" for e in record["files"])
    assert records[0] == records[1]
    assert {c: CELLS_RETAINED[c] for c in FOVS} == dict(
        tables_of(case.folder, json.loads(second["report.json"]))["fovs"][["fov_id", "n_cells_retained"]].values)
