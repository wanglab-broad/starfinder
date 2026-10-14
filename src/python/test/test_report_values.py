"""The values the report computes (W-453): read levels, shifts, residuals, alignment scores, areas, densities,
summary keys, the other values of the page and the example FOVs.

Checks of docs/report-algorithms.md ("Validation design"): DF1 to DF6, DF12, the fractions and residuals of RC1,
RC2, RC8 to RC10, RC13's medians and alignment score, EF1 and EF2 (without its ``scene_no_run_json`` part), on
``hand_tables`` built here and on ``report_records`` and ``report_cells`` of ``report_fixtures.py``. Expected
values are written by hand or computed here by the page's formula from the fixtures' own numbers; the area a FOV
owns is counted here with the package's ``ownership_map``. A float the page calls exact is compared within
1e-9 absolute (the project guidance), and within 1e-12 where DF2 names it.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from starfinder.report import ReportConfig
from starfinder.report._values import (NA, Unavailable, acquisition, alignment_score, chain_shift, choose_examples,
                                       configuration_check, density, fraction, residuals, run_seconds, software,
                                       time_per_step, z_size)

from .report_fixtures import (CELLS_RETAINED, COUNTS, FOVS, MOVING, NCC_AFTER, ORIGINS, SHAPE_ZYX, SPACING_ZYX,
                              TRANSLATIONS, application_entry, estimation_entry, jsonable, metadata, report_cells,
                              report_records)
from .test_report_collection import anndata_or_skip, collect, keyed

pytestmark = [pytest.mark.report, pytest.mark.validation]

TOL = 1e-9
LEVELS = {"correct_form_fraction": "correct_form", "matched_fraction": "matched",
          "merged_fraction": "merged_by_deduplication", "passed_filter_fraction": "passed_filter"}


def by_key(frame, *keys):
    return frame.set_index(list(keys))


# --- DF1, RC1, RC8: the read levels ----------------------------------------------------------------------------

def test_df1_fraction_definition():
    # A hand table: matched above correct_form is kept (the levels are not nested); each fraction has spots as its
    # denominator.
    counts = {"spots": 1000, "correct_form": 700, "matched": 900, "merged_by_deduplication": 10, "passed_filter": 650}
    shares = {c: fraction(counts[level], counts["spots"], "spots") for c, level in LEVELS.items()}
    assert shares == {c: counts[level] / 1000 for c, level in LEVELS.items()}
    assert shares["matched_fraction"] > shares["correct_form_fraction"]
    assert fraction(0, 0, "spots") == Unavailable("undefined", "spots")
    missing = Unavailable("not_configured", "config.pipeline.deduplication is null")
    assert fraction(missing, 1000, "spots") is missing
    assert fraction(10, Unavailable("step_failed", "x"), "spots") == Unavailable("step_failed", "x")


def test_df1_rc1_fractions_and_totals(tmp_path):
    case = report_records(tmp_path / "root")
    _, record, tables = collect(case)
    fovs = by_key(tables["fovs"], "fov_id")
    for fov in FOVS:
        for column, level in LEVELS.items():
            assert fovs.loc[fov, column] == pytest.approx(fovs.loc[fov, level] / fovs.loc[fov, "spots"], abs=TOL)
        assert fovs.loc[fov, "run_seconds"] == 8.0
    totals = by_key(tables["readout_totals"], "level")
    expected = {"spots": (4000, 1.0), "correct_form": (2960, 0.74), "matched": (2800, 0.7),
                "merged_by_deduplication": (40, 0.01), "passed_filter": (2590, 0.6475)}
    for level, (count, share) in expected.items():
        assert totals.loc[level, "count"] == count and totals.loc[level, "fovs"] == 4
        assert totals.loc[level, "fraction_of_spots"] == pytest.approx(share, abs=TOL)
    assert totals.loc["reads_in_cells", "fovs"] == 0


@pytest.mark.parametrize("variant,column,reason", [("stage_off", "merged_fraction", "not_configured"),
                                                   ("older", "correct_form_fraction", "not_recorded")])
def test_df1_null_count_gives_its_reason(tmp_path, variant, column, reason):
    _, _, tables = collect(report_records(tmp_path / "root", variant))
    assert tables["fovs"][column].isna().all()
    assert [r for r in keyed(tables["unavailable"]) if r[2] == column] == [
        ("readout.per_fov", "fovs", column, f"fov_id={f}", reason) for f in FOVS]


def test_rc8_fov_with_no_spot(tmp_path):
    _, _, tables = collect(report_records(tmp_path / "root", "no_spot"))
    f2 = by_key(tables["fovs"], "fov_id").loc["F2"]
    assert [f2[c] for c in ("spots", "correct_form", "matched", "merged_by_deduplication", "passed_filter")] == [0] * 5
    rows = tables["unavailable"]
    named = rows[rows["row_key"].eq("fov_id=F2").to_numpy(bool) & rows["column"].isin(list(LEVELS)).to_numpy(bool)]
    assert named[["column", "reason", "detail"]].values.tolist() == [[c, "undefined", "spots"] for c in LEVELS]
    totals = by_key(tables["readout_totals"], "level")
    # F2's zero spots count in the denominator of every level, as the spots of the FOVs with the level.
    assert totals.loc["matched", "fraction_of_spots"] == pytest.approx((700 + 560 + 700) / 2800, abs=TOL)


# --- DF2: the shift of the global steps -------------------------------------------------------------------------

def result_entry(transform, index, field_name=None):
    from starfinder.io._checkpoint import _transform_json
    from starfinder.registration import TranslationConfig, WarpConfig
    from starfinder.registration._types import RegistrationDiagnostics, RegistrationResult
    diagnostics = RegistrationDiagnostics("translation", "scipy_fft", TranslationConfig())
    return jsonable(_transform_json(RegistrationResult(transform, diagnostics, WarpConfig()), index, field_name))


def translation(displacement):
    from starfinder.registration._types import TranslationTransform
    return TranslationTransform(tuple(float(v) for v in displacement), SHAPE_ZYX, SHAPE_ZYX, metadata(), metadata())


def affine():
    from starfinder.registration import AffineTransform
    matrix = np.eye(4)
    matrix[2, 2] = 1.1
    matrix[:3, 3] = (0.0, 0.5, -0.75)
    return AffineTransform(matrix, SHAPE_ZYX, SHAPE_ZYX, metadata(), metadata())


def dense():
    from starfinder.registration import DenseDisplacementTransform
    return DenseDisplacementTransform(np.zeros((*SHAPE_ZYX, 3)), SHAPE_ZYX, SHAPE_ZYX, metadata(), metadata())


def chain(*transforms):
    return [result_entry(t, i, "round2_field.npz" if t.__class__.__name__.startswith("Dense") else None)
            for i, t in enumerate(transforms)]


def test_df2_shift_of_the_global_steps():
    assert chain_shift(chain(translation((0, 1, 2)), translation((0, -1, 0)))) == (0.0, 0.0, 2.0)
    # The affine diag(1, 1, 1.1), b = (0, 0.5, −0.75) on the 4 × 16 × 16 grid moves the centre (1.5, 7.5, 7.5).
    for steps, expected in [((affine(),), (0.0, 0.5, 0.0)),
                            ((translation((0, 1, 2)), affine()), (0.0, 1.5, 2.0)),
                            ((affine(), translation((0, 1, 2))), (0.0, 1.5, 2.2))]:
        assert chain_shift(chain(*steps)) == pytest.approx(expected, abs=1e-12)
    # A dense last step is not a global step; a dense step alone gives n/a.
    assert chain_shift(chain(translation((0, 1, 2)), dense())) == (0.0, 1.0, 2.0)
    assert chain_shift(chain(dense())) is NA


def test_df2_affine_and_dense_through_the_collection(tmp_path):
    case = report_records(tmp_path / "root")
    record = case.run_record("F1")
    record["transforms"]["round2"]["results"] = chain(translation((0, 1, 2)), affine())
    record["transforms"]["round3"]["results"] = chain(dense())
    case.write_run_record("F1", record)
    _, _, tables = collect(case)
    registration = by_key(tables["registration"], "fov_id", "round")
    assert registration.loc[("F1", "round2"), ["shift_z", "shift_y", "shift_x"]].tolist() == pytest.approx(
        [0.0, 1.5, 2.0], abs=1e-12)
    assert registration.loc[("F1", "round3"), ["shift_z", "shift_y", "shift_x", "residual_z", "residual_y",
                                               "residual_x"]].isna().all()
    # The n/a shift of a dense step alone is named by no row, and the round's median is over F2 to F4.
    assert not [r for r in keyed(tables["unavailable"]) if r[3] == "fov_id=F1;round=round3"]
    assert registration.loc[("F4", "round3"), "residual_x"] == 8.0


# --- DF3, RC1, RC2, RC13: the shift residual --------------------------------------------------------------------

def rows_of(round_name, shifts, outcomes=None):
    """Registration rows of one round: shifts in x only (z and y 0), each outcome succeeded unless given."""
    outcomes = outcomes or {}
    rows = []
    for fov, shift in shifts.items():
        outcome = outcomes.get(fov, "succeeded")
        value = shift if outcome == "succeeded" else Unavailable("step_failed", f"outcome {outcome}")
        rows.append({"fov_id": fov, "round": round_name, "outcome": outcome, "shift_z": 0.0 if outcome ==
                     "succeeded" else value, "shift_y": 0.0 if outcome == "succeeded" else value, "shift_x": value})
    return rows


def test_df3_residual_definition():
    odd = rows_of("r", {"A": 1.0, "B": 4.0, "C": 2.0})
    assert residuals(odd)[("r", "x")] == 2.0
    assert [r["residual_x"] for r in odd] == [-1.0, 2.0, 0.0]
    even = rows_of("r", {"A": 1.0, "B": 4.0, "C": 2.0, "D": 10.0})
    assert residuals(even)[("r", "x")] == 3.0
    assert [r["residual_x"] for r in even] == [-2.0, 1.0, -1.0, 7.0]
    one = rows_of("r", {"A": 3.25})
    residuals(one)
    assert one[0]["residual_x"] == 0.0
    two = rows_of("r", {"A": 1.0, "B": 4.0})
    residuals(two)
    assert [r["residual_x"] for r in two] == [-1.5, 1.5]
    # A FOV whose round did not succeed has no residual and is not in the median.
    mixed = rows_of("r", {"A": 1.0, "B": 4.0, "C": 2.0, "D": 50.0, "E": 60.0, "F": 70.0},
                    {"D": "rejected", "E": "failed", "F": "application_failed"})
    assert residuals(mixed)[("r", "x")] == 2.0
    assert [r["residual_x"] for r in mixed][:3] == [-1.0, 2.0, 0.0]
    assert [r["residual_x"] for r in mixed][3:] == [Unavailable("step_failed", f"outcome {o}")
                                                   for o in ("rejected", "failed", "application_failed")]
    # No succeeded FOV: every residual of the round null with undefined.
    none = rows_of("r", {"A": 1.0, "B": 2.0}, {"A": "failed", "B": "rejected"})
    assert residuals(none) == {}
    assert [r[f"residual_{a}"] for r in none for a in "zyx"] == [Unavailable("undefined", "no succeeded FOV")] * 6


def test_rc1_residuals(tmp_path):
    _, _, tables = collect(report_records(tmp_path / "root"))
    registration = by_key(tables["registration"], "fov_id", "round")
    nonzero = {("F2", "round2", "x"): 1.0, ("F3", "round2", "y"): 1.0, ("F3", "round3", "z"): 1.0,
               ("F4", "round3", "x"): 8.0}
    for fov in FOVS:
        for name in MOVING:
            for axis in "zyx":
                assert registration.loc[(fov, name), f"residual_{axis}"] == nonzero.get((fov, name, axis), 0.0)
                assert registration.loc[(fov, name), f"shift_{axis}"] == TRANSLATIONS[name][fov]["zyx".index(axis)]


@pytest.mark.parametrize("variant", ["failed", "failed_applying", "failed_estimating", "failed_succeeded"])
def test_rc2_rc13_round3_medians_over_the_succeeded_fovs(tmp_path, variant):
    _, _, tables = collect(report_records(tmp_path / "root", variant))
    registration = by_key(tables["registration"], "fov_id", "round")
    # Round3 over F1, F2 and F4 only: medians (0, −1, 0), so F4's residual x is 8.
    assert registration.loc[("F4", "round3"), "residual_x"] == pytest.approx(8.0, abs=TOL)
    for fov in ("F1", "F2", "F4"):
        for axis in "zyx":
            assert registration.loc[(fov, "round3"), f"residual_{axis}"] == pytest.approx(
                TRANSLATIONS["round3"][fov]["zyx".index(axis)] - (0, -1, 0)["zyx".index(axis)], abs=TOL)
    assert pd.isna(registration.loc[("F3", "round3"), "alignment_score"])
    rows = [r for r in keyed(tables["unavailable"]) if r[2] == "alignment_score"]
    assert rows == ([("registration.heatmap", "registration", "alignment_score", "fov_id=F3;round=round3",
                      "step_failed")])
    # F3's round2 shift is kept, and enters round2's median.
    assert registration.loc[("F3", "round2"), "residual_y"] == 1.0


# --- DF4: the alignment score -----------------------------------------------------------------------------------

def test_df4_alignment_score():
    entries = [estimation_entry(0.81), application_entry(0.9)]
    assert alignment_score(entries, "succeeded") == 0.9  # the application entry's, not the estimation's 0.81
    null = application_entry(None)
    null["qc"]["reasons"] = {"ncc_after": "empty mask"}
    assert alignment_score([estimation_entry(0.8), null], "succeeded") == Unavailable(
        "undefined", "qc.values.ncc_after is null (empty mask)")
    assert alignment_score([estimation_entry(0.8), application_entry(None, qc=False)], "succeeded") == Unavailable(
        "undefined", "application qc is null")
    # A round without an application entry did not succeed.
    assert alignment_score([estimation_entry(0.8, "rejected")], "rejected") == Unavailable(
        "step_failed", "outcome rejected")


def test_df4_rc1_scores_and_a_null_qc(tmp_path):
    case = report_records(tmp_path / "root")
    record = case.run_record("F2")
    record["registration"]["round2"][-1]["qc"]["values"]["ncc_after"] = None
    record["registration"]["round2"][-1]["qc"]["reasons"] = {"ncc_after": "empty mask"}
    case.write_run_record("F2", record)
    _, _, tables = collect(case)
    registration = by_key(tables["registration"], "fov_id", "round")
    for fov in FOVS:
        for name in MOVING:
            if (fov, name) != ("F2", "round2"):
                assert registration.loc[(fov, name), "alignment_score"] == NCC_AFTER[name][fov]
    unavailable = tables["unavailable"]
    row = unavailable[unavailable["column"].eq("alignment_score").fillna(False).to_numpy(bool)]
    assert row[["row_key", "reason", "detail"]].values.tolist() == [
        ["fov_id=F2;round=round2", "undefined", "qc.values.ncc_after is null (empty mask)"]]


# --- DF5, DF6: the area a FOV owns and the cell density ---------------------------------------------------------

def own_count(case):
    """Each FOV's pixels whose centre it owns, counted here with the package's ownership_map."""
    from starfinder.assembly import AssemblyConfig, ownership_map
    from starfinder.segmentation import ReferenceGrid
    from starfinder.stitching import load_placement
    grids = {f: ReferenceGrid(SHAPE_ZYX, metadata(), "declared") for f in FOVS}
    owner = ownership_map(load_placement(case.root / "stitching" / "s1"), grids, config=AssemblyConfig())
    i, j = np.meshgrid(np.arange(SHAPE_ZYX[1]), np.arange(SHAPE_ZYX[2]), indexing="ij")
    counts = {}
    for fov, (oz, oy, ox) in ORIGINS.items():
        points = np.stack([np.full(i.size, oz), oy + SPACING_ZYX[1] * i.ravel(), ox + SPACING_ZYX[2] * j.ravel()], 1)
        counts[fov] = int((owner.owner(points) == fov).sum())
    return counts


def test_df5_df6_area_and_density(tmp_path):
    anndata_or_skip()
    case = report_cells(tmp_path / "root")
    _, _, tables = collect(case)
    fovs = by_key(tables["fovs"], "fov_id")
    counts = own_count(case)
    assert counts == {"F1": 210, "F2": 200, "F3": 196, "F4": 194}
    pixel = SPACING_ZYX[1] * SPACING_ZYX[2]
    assert pixel == 0.0625
    for fov, area in zip(FOVS, (13.125, 12.5, 12.25, 12.125)):
        assert fovs.loc[fov, "owned_area_um2"] == pytest.approx(counts[fov] * pixel, abs=TOL)
        assert fovs.loc[fov, "owned_area_um2"] == pytest.approx(area, abs=TOL)
        assert fovs.loc[fov, "cell_density_per_mm2"] == pytest.approx(
            CELLS_RETAINED[fov] / (counts[fov] * pixel) * 1e6, abs=TOL)
    assert fovs.loc["F1", "cell_density_per_mm2"] == pytest.approx(1.6e6, abs=TOL)
    assert not [r for r in keyed(tables["unavailable"]) if r[2] in ("owned_area_um2", "cell_density_per_mm2")]


def test_df5_df6_unplaced_placeholder_and_zero_area(tmp_path):
    _, _, tables = collect(report_records(tmp_path / "states", "placement_states"))
    rows = keyed(tables["unavailable"])
    for column in ("owned_area_um2", "cell_density_per_mm2"):
        assert ("assembly.cells_per_fov", "fovs", column, "fov_id=F4", "fov_not_placed") in rows
    # A placeholder owns no point: its area and density are n/a, named by no row.
    anndata_or_skip()
    from starfinder.stitching import save_placement
    from .report_fixtures import placement_result
    case = report_cells(tmp_path / "placeholder")
    save_placement(placement_result(FOVS, statuses={"F4": "placeholder"}), case.root / "stitching" / "s1")
    _, _, tables = collect(case)
    f4 = by_key(tables["fovs"], "fov_id").loc["F4"]
    assert pd.isna(f4["owned_area_um2"]) and pd.isna(f4["cell_density_per_mm2"])
    assert not [r for r in keyed(tables["unavailable"]) if r[3] == "fov_id=F4" and r[0] == "assembly.cells_per_fov"]
    # A placeholder is no owning FOV of reads_no_cell.
    assert ("F4", "reads_no_cell") not in set(zip(tables["problematic_fovs"]["fov_id"], tables["problematic_fovs"]["rule"]))
    # An area of 0 gives an undefined density; n/a as the area.
    assert density(5, 0.0) == Unavailable("undefined", "the area")
    assert density(5, NA) is NA
    assert density(21, 13.125) == 21 / 13.125 * 1e6


# --- DF12: summary keys and the other values of the page --------------------------------------------------------

def records_of(case):
    return {f: case.run_record(f) for f in case.fov_ids}


def test_df12_summary_keys(tmp_path):
    _, record, _ = collect(report_records(tmp_path / "root"))
    summary = record["summary"]
    assert (summary["fovs_succeeded"], summary["spots_median_per_fov"]) == (4, 1000.0)
    assert summary["matched_fraction"] == pytest.approx(2800 / 4000, abs=TOL)
    assert summary["matched_fraction"] == pytest.approx(0.7, abs=TOL)
    _, record, _ = collect(report_records(tmp_path / "not_run", "not_run"))
    assert (record["summary"]["spots_median_per_fov"], record["summary"]["matched_fraction"]) == (
        1000.0, pytest.approx(2100 / 3000, abs=TOL))


def test_df12_cells_median_per_fov(tmp_path):
    anndata_or_skip()
    _, record, _ = collect(report_cells(tmp_path / "root"))
    assert record["summary"]["cells_median_per_fov"] == float(np.median(list(CELLS_RETAINED.values())))


def test_df12_configuration_acquisition_software_and_time(tmp_path):
    case = report_records(tmp_path / "root")
    _, _, tables = collect(case)
    records = records_of(case)
    assert configuration_check(records) == {"text": "identical for all FOVs", "groups": [list(FOVS)],
                                            "differing_keys": []}
    changed = json.loads(json.dumps(records))
    changed["F2"]["config"]["pipeline"]["spot_finding"]["threshold_value"] = 0.3
    assert configuration_check(changed) == {"text": "differs: 2 groups", "groups": [["F1", "F3", "F4"], ["F2"]],
                                            "differing_keys": ["pipeline"]}
    assert configuration_check(dict.fromkeys(FOVS)) == Unavailable("no_run_record",
                                                                   "no FOV of the list has a run record")
    placement = json.loads((case.root / "stitching" / "s1" / "placement.json").read_text())
    assert z_size(None, placement) == 4
    assert acquisition(records, z_size(None, placement))["text"] == "3 rounds × 4 channels, 3D"
    assert acquisition(records, 1)["text"] == "3 rounds × 4 channels, 2D (Z = 1)"
    assert software(records) == [{"version": "0.1.0", "git_commit": "0" * 40, "git_dirty": False, "fovs": 4}]
    # Each FOV's time per step name is the sum of its rows of that name (the rounds of a per-round step add up).
    steps = tables["steps"].astype(object).where(tables["steps"].notna(), None).to_dict("records")
    times = time_per_step(steps)
    for fov in FOVS:
        record = records[fov]
        for name in dict.fromkeys(s["name"] for s in record["steps"]):
            assert times[name][fov] == sum(s["seconds"] for s in record["steps"] if s["name"] == name)
    assert times["preprocess:min_max_normalization"]["F1"] == 0.75 and times["register"]["F1"] == 1.0
    assert "stitch" not in times  # its time is not recorded
    assert run_seconds("2026-10-12T10:00:00+00:00", "2026-10-12T10:03:08+00:00") == 188.0


def test_df12_acquisition_from_the_export(tmp_path):
    anndata_or_skip()
    case = report_cells(tmp_path / "root")
    export = json.loads((case.root / "export" / "s1" / "export.json").read_text())
    assert z_size(export, None) == 4
    export["fusion"] = {"fovs": [{"grid": {"shape_zyx": [1, 36, 36]}}]}
    assert z_size(export, None) == 1


# --- EF1, EF2, RC9, RC10: the example FOVs ----------------------------------------------------------------------

def test_ef1_automatic_choice(tmp_path):
    _, record, tables = collect(report_records(tmp_path / "records"))
    assert record["examples"] == {
        "example_1": {"fov_id": "F1", "chosen_by": "automatic", "rule": "median_reads", "value": 650, "seed": 0,
                      "near_median": 1, "median": 645.0},
        "example_2": {"fov_id": "F3", "chosen_by": "automatic", "rule": "fewest_reads", "value": 500, "seed": 0},
        "example_3": {"fov_id": "F2", "chosen_by": "automatic", "rule": "most_reads", "value": 800, "seed": 0}}
    assert tables["fovs"]["example"].tolist() == ["example_1", "example_3", "example_2", pd.NA]
    anndata_or_skip()
    _, record, tables = collect(report_cells(tmp_path / "cells"))
    assert record["examples"]["example_1"]["fov_id"] == "F1"
    assert record["examples"]["example_2"] == {"fov_id": "F2", "chosen_by": "automatic", "rule": "fewest_cells",
                                               "value": 0, "seed": 0}
    assert record["examples"]["example_3"] == {"fov_id": "F4", "chosen_by": "automatic", "rule": "most_cells",
                                               "value": 14, "seed": 0}
    assert tables["fovs"]["example"].tolist() == ["example_1", "example_2", pd.NA, "example_3"]


def test_ef1_ten_fovs():
    fovs = [f"T{k}" for k in range(10)]
    reads = {f: 100 * (k + 1) for k, f in enumerate(fovs)}
    chosen = choose_examples(fovs, reads, {})
    # N holds the FOVs of 500 and 600 (q = 2), in list order; example_1 is N[default_rng(0).integers(2)].
    near = ["T4", "T5"]
    assert chosen[0].fov_id == near[int(np.random.default_rng(0).integers(2))]
    assert (chosen[0].near_median, chosen[0].median, chosen[0].seed) == (2, 550.0, 0)
    assert [(e.role, e.fov_id, e.rule, e.value) for e in chosen[1:]] == [("example_2", "T0", "fewest_reads", 100),
                                                                         ("example_3", "T9", "most_reads", 1000)]
    # Ties of distance go by list order; q = max(1, ceil(n / 5)).
    assert math.ceil(10 / 5) == 2


def test_ef2_rc9_one_fov(tmp_path):
    _, record, tables = collect(report_records(tmp_path / "root", "one"))
    assert tables["fovs"]["example"].tolist() == ["example_1;example_2;example_3"]
    assert [e["fov_id"] for e in record["examples"].values()] == ["F1"] * 3
    registration = tables["registration"]
    assert (registration[[f"residual_{a}" for a in "zyx"]] == 0.0).all().all()
    assert tables["problematic_fovs"].empty


def test_ef2_two_fovs(tmp_path):
    import shutil
    case = report_records(tmp_path / "root")
    shutil.rmtree(case.root / "stitching")
    for fov in ("F3", "F4"):
        shutil.rmtree(case.checkpoint(fov))
    case.fov_ids = ["F1", "F2"]
    _, record, tables = collect(case)
    # Median 725: F1 and F2 both at 75, F1 first in list order; F2 is the fewest left, and the most over both.
    assert [(e["fov_id"], e["rule"]) for e in record["examples"].values()] == [
        ("F1", "median_reads"), ("F2", "fewest_reads"), ("F2", "most_reads")]
    assert tables["fovs"]["example"].tolist() == ["example_1", "example_2;example_3"]


def test_ef2_named_examples(tmp_path):
    case = report_records(tmp_path / "root")
    case.config = ReportConfig(example_fovs=("F4", "F2", "F3"))
    _, record, tables = collect(case)
    assert record["examples"] == {"example_1": {"fov_id": "F4", "chosen_by": "configuration"},
                                  "example_2": {"fov_id": "F2", "chosen_by": "configuration"},
                                  "example_3": {"fov_id": "F3", "chosen_by": "configuration"}}
    assert tables["fovs"]["example"].tolist() == [pd.NA, "example_2", "example_3", "example_1"]


def test_ef2_no_read_count():
    reasons = {"A": Unavailable("no_run_record", "a"), "B": Unavailable("not_configured", "b")}
    assert choose_examples(["A", "B"], reasons, {}) == Unavailable("no_run_record", "a")


def test_rc10_equal_values(tmp_path):
    _, record, tables = collect(report_records(tmp_path / "root", "equal"))
    assert (tables["registration"][[f"residual_{a}" for a in "zyx"]] == 0.0).all().all()
    # ceil(4/5) = 1 FOV near the median, F1 first in the order (distance, list_index).
    assert record["examples"]["example_1"]["fov_id"] == "F1"
    assert not set(tables["problematic_fovs"]["rule"]) & {"shift_residual", "alignment_score",
                                                          "correct_form_fraction", "matched_fraction",
                                                          "spots_near_zero"}
    assert COUNTS["accepted"]["F1"] == record["examples"]["example_1"]["value"]
