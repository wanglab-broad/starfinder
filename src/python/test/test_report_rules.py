"""The rules of the problematic FOVs (W-453): the one form of a rule on values and every rule of the first delivery.

Checks of docs/report-algorithms.md ("Validation design"): PR1 to PR5, RC1's two rows, RC8 to RC10, the reason
texts of "The reason texts", and PG6 and RC12 on the completed tables, on ``hand_tables`` built here and on
``report_records`` and ``report_cells`` of ``report_fixtures.py``. Comparisons are strict, as the page states. A
float the page calls exact is compared within 1e-9 absolute (the project guidance); PR2's limit within 1e-12.
"""
from __future__ import annotations

import json
import pandas as pd
import pytest

from starfinder.report import PROBLEM_RULES, ReportConfig, ReportLimits, RuleLimit
from starfinder.report._rules import RULE_SECTIONS, RuleInputs, evaluate_rules, rule_form
from starfinder.report._schema import TABLE_DTYPES
from starfinder.report._values import NA, Unavailable, mad, median

from .report_fixtures import FOVS, estimation_entry, report_cells, report_records
from .test_report_collection import (CONTRACT_COLUMNS, CONTRACT_RECORD_KEYS, anndata_or_skip, check_null_values,
                                     collect, keyed, pandas_dtype)

pytestmark = [pytest.mark.report, pytest.mark.validation]

TOL = 1e-9
VALUE_RULES = ("shift_residual", "alignment_score", "correct_form_fraction", "matched_fraction", "spots_near_zero")


def rows_of(tables):
    """(fov_id, rule, round, axis) of the rows of problematic_fovs.csv, in order."""
    return [tuple(None if pd.isna(v) else v for v in r)
            for r in tables["problematic_fovs"][["fov_id", "rule", "round", "axis"]].itertuples(index=False)]


def completed(case, record, tables):
    """PG6 and RC12 on the completed tables: the problematic table's schema, the record's keys, the null rules."""
    frame = tables["problematic_fovs"]
    assert list(frame.columns) == [c for c, _, _ in CONTRACT_COLUMNS["problematic_fovs"]]
    assert {c: str(d) for c, d in frame.dtypes.items()} == {c: pandas_dtype(k, n)
                                                           for c, k, n in CONTRACT_COLUMNS["problematic_fovs"]}
    assert list(record) == [k for k in CONTRACT_RECORD_KEYS if k != "replay"]  # replay is W-454's
    assert list(record["examples"]) == ["example_1", "example_2", "example_3"]
    for entry in record["examples"].values():
        assert entry is None or set(entry) <= {"fov_id", "chosen_by", "rule", "value", "seed", "near_median",
                                               "median"}
    for rule, entry in record["rules"].items():
        assert list(entry) == ["limit", "unit", "evaluated", "fovs_marked"]
        if entry["evaluated"]:
            assert entry["fovs_marked"] == len(set(frame["fov_id"][frame["rule"].eq(rule).to_numpy(bool)]))
    for row in frame.to_dict("records"):
        categorical = row["rule"] not in VALUE_RULES
        assert row["section"] == RULE_SECTIONS[row["rule"]]
        assert all(pd.isna(row[c]) == categorical for c in ("measured", "limit", "unit"))
        assert pd.isna(row["round"]) == (row["rule"] not in ("shift_residual", "alignment_score",
                                                             "registration_rejected", "registration_fallback"))
        assert pd.isna(row["axis"]) == (row["rule"] != "shift_residual")
    words = json.dumps(record).lower()
    assert not any(w in words for w in ('"verdict"', '"score"', '"pass"', '"grade"'))
    check_null_values(case, record, tables)


# --- hand_tables -------------------------------------------------------------------------------------------------

def hand_fovs(column, values):
    """fovs rows of a hand table: ``column`` holds the values (None: unavailable), every other column n/a."""
    rows = []
    for i, value in enumerate(values):
        row = {c: NA for c in TABLE_DTYPES["fovs"]}
        row.update(fov_id=f"H{i + 1}", list_index=i, owned_area_um2=NA)
        row[column] = Unavailable("step_failed", f"H{i + 1}") if value is None else value
        rows.append(row)
    return rows


def evaluate(column, values, limits=None):
    """Every rule on a hand fovs table, without run records or registration (those rules not evaluated)."""
    fovs = hand_fovs(column, values)
    ids = [r["fov_id"] for r in fovs]
    inputs = RuleInputs(ids, {f: {"status": "succeeded"} for f in ids}, Unavailable("no_placement", "x"),
                        Unavailable("not_configured", "no registration"), {})
    return evaluate_rules(fovs, [], limits or ReportLimits(), inputs)


def test_pr1_rule_form():
    values = list(zip("abcd", (0.80, 0.80, 0.50, 0.80)))
    form = rule_form(values, RuleLimit(k=3, floor=0.05), "low")
    assert (form.marked, form.limit, form.absolute) == (["c"], pytest.approx(0.05, abs=TOL), False)
    assert form.median == pytest.approx(0.8, abs=TOL)
    form = rule_form(values, RuleLimit(absolute=0.6), "low")
    assert (form.marked, form.limit, form.absolute) == (["c"], 0.6, True)
    assert rule_form(values, RuleLimit(absolute=0.5), "low").marked == []  # strict
    assert rule_form(values, RuleLimit(), "low") is None


def test_pr1_rows_and_records(tmp_path):
    rows, rules, part_rows = evaluate("correct_form_fraction", [0.80, 0.80, 0.50, 0.80])
    assert [(r["fov_id"], r["rule"], r["measured"], r["sample_median"], r["limit"], r["unit"]) for r in rows] == [
        ("H3", "correct_form_fraction", 0.5, pytest.approx(0.8, abs=TOL), pytest.approx(0.05, abs=TOL), "fraction")]
    rows, _, _ = evaluate("correct_form_fraction", [0.80, 0.80, 0.50, 0.80],
                          ReportLimits(correct_form_fraction=RuleLimit(absolute=0.6)))
    assert [(r["fov_id"], r["sample_median"], r["limit"]) for r in rows] == [("H3", NA, 0.6)]
    rows, _, _ = evaluate("correct_form_fraction", [0.80, 0.80, 0.50, 0.80],
                          ReportLimits(correct_form_fraction=RuleLimit(absolute=0.5)))
    assert rows == []
    rows, rules, part_rows = evaluate("correct_form_fraction", [0.80, 0.80, 0.50, 0.80],
                                      ReportLimits(correct_form_fraction=RuleLimit()))
    assert rows == [] and rules["correct_form_fraction"] == {"evaluated": False, "fovs_marked": NA}
    assert ("rule:correct_form_fraction", Unavailable("not_configured",
                                                      "config.limits.correct_form_fraction sets no limit"), "") \
        in part_rows
    # Through the collection: the absolute limit's row has no sample median; no limit is one rule: row.
    case = report_records(tmp_path / "root")
    case.config = ReportConfig(limits=ReportLimits(correct_form_fraction=RuleLimit(absolute=0.6),
                                                   matched_fraction=RuleLimit()))
    _, record, tables = collect(case)
    frame = tables["problematic_fovs"]
    row = frame[frame["rule"].eq("correct_form_fraction").to_numpy(bool)]
    assert row[["fov_id", "measured", "limit"]].values.tolist() == [["F3", 0.5, 0.6]]
    assert row["sample_median"].isna().all()
    assert record["rules"]["matched_fraction"] == {"limit": {"k": None, "floor": None, "absolute": None},
                                                   "unit": "fraction", "evaluated": False, "fovs_marked": None}
    assert ("rule:matched_fraction", None, None, "", "not_configured") in keyed(tables["unavailable"])
    completed(case, record, tables)


def test_pr2_edge_cases():
    assert rule_form([("a", 0.5)], RuleLimit(k=0.1, floor=0), "low").marked == []
    two = [("a", 0.2), ("b", 0.8)]
    form = rule_form(two, RuleLimit(k=1, floor=0), "low")
    assert form.marked == [] and form.median == 0.5
    # The two deviations as their exact float64 values, and their median, the MAD.
    assert [abs(v - 0.5) for _, v in two] == [0.3, 0.30000000000000004]
    assert mad([0.2, 0.8]) == 0.30000000000000004
    assert form.limit == pytest.approx(0.44478, abs=1e-12)
    assert rule_form(two, RuleLimit(k=1, floor=0), "high").marked == []
    assert rule_form(two, RuleLimit(k=0, floor=0), "low").marked == ["a"]
    assert rule_form(two, RuleLimit(k=0, floor=0), "high").marked == ["b"]
    equal = [(k, 0.5) for k in "abcd"]
    assert rule_form(equal, RuleLimit(k=0, floor=0), "low").marked == []
    assert rule_form(equal, RuleLimit(absolute=0.6), "low").marked == list("abcd")
    assert rule_form([(k, 0.7) for k in "abcd"], RuleLimit(absolute=0.6), "low").marked == []
    assert rule_form([(k, 0.5) for k in "abcd"], RuleLimit(absolute=0.6), "high").marked == []
    # A spread of zero and one outlier: the limit is the floor.
    assert rule_form([("a", 0.8), ("b", 0.8), ("c", 0.8), ("d", 0.5)], RuleLimit(k=3, floor=0.05), "low").marked \
        == ["d"]
    assert median([0.8, 0.8, 0.5]) == 0.8


def test_pr2_null_values():
    # A null value does not enter the median and is not evaluated; the rule needs no row of its own.
    rows, rules, part_rows = evaluate("correct_form_fraction", [0.8, 0.8, None, 0.5])
    assert [(r["fov_id"], r["sample_median"]) for r in rows] == [("H4", 0.8)]
    assert rules["correct_form_fraction"] == {"evaluated": True, "fovs_marked": 1}
    assert not [p for p in part_rows if p[0] == "rule:correct_form_fraction"]
    # Every value null: one rule: row with the values' first reason.
    rows, rules, part_rows = evaluate("correct_form_fraction", [None, None])
    assert rules["correct_form_fraction"]["evaluated"] is False
    assert ("rule:correct_form_fraction", Unavailable("step_failed", "H1"), "") in part_rows


def test_pr5_spots_near_zero():
    rows, rules, _ = evaluate("spots", [1000, 1200, 40, 1000])
    assert [(r["fov_id"], r["measured"], r["sample_median"], r["limit"], r["unit"], r["reason"]) for r in rows] == [
        ("H3", 40.0, 1000.0, 50.0, "reads", "spot count close to zero")]
    rows, rules, _ = evaluate("spots", [0, 0, 0])
    assert rows == [] and rules["spots_near_zero"] == {"evaluated": True, "fovs_marked": 0}


# --- report_records: the rules of the first delivery ----------------------------------------------------------------

def test_rc1_rows(tmp_path):
    case = report_records(tmp_path / "root")
    _, record, tables = collect(case)
    assert rows_of(tables) == [("F3", "correct_form_fraction", None, None), ("F4", "shift_residual", "round3", "x")]
    assert record["rules"]["shift_residual"]["fovs_marked"] == 1
    assert record["rules"]["reads_no_cell"]["evaluated"] is False
    completed(case, record, tables)


SHIFT_ROWS = [("F2", "shift_residual", "round2", "x"), ("F3", "shift_residual", "round2", "y"),
              ("F3", "shift_residual", "round3", "z"), ("F4", "shift_residual", "round3", "x")]


@pytest.mark.parametrize("limit,expected", [
    (RuleLimit(k=5, floor=5), [(8, 0, 5)]),
    (RuleLimit(k=0, floor=0.5), [(3, 2, 0.5), (2, 1, 0.5), (1, 0, 0.5), (8, 0, 0.5)]),
    (RuleLimit(absolute=0.5), [(3, 2, 0.5), (2, 1, 0.5), (1, 0, 0.5), (8, 0, 0.5)]),
    (RuleLimit(absolute=8), [])], ids=["default", "k0_floor0.5", "absolute0.5", "absolute8"])
def test_pr3_shift_residual(tmp_path, limit, expected):
    case = report_records(tmp_path / "root")
    case.config = ReportConfig(limits=ReportLimits(shift_residual=limit))
    _, record, tables = collect(case)
    frame = tables["problematic_fovs"]
    mine = frame[frame["rule"].eq("shift_residual").to_numpy(bool)]
    keys = [r for r in rows_of(tables) if r[1] == "shift_residual"]
    assert keys == ([SHIFT_ROWS[-1]] if len(expected) == 1 else SHIFT_ROWS if expected else [])
    assert len(mine) == len(expected)
    for (_, row), (measured, centre, bound) in zip(mine.iterrows(), expected):
        assert (row["measured"], row["sample_median"], row["limit"]) == pytest.approx((measured, centre, bound),
                                                                                       abs=TOL)
        # measured − sample_median is the residual of registration.csv.
        residual = tables["registration"].set_index(["fov_id", "round"]).loc[(row["fov_id"], row["round"]),
                                                                             f"residual_{row['axis']}"]
        assert row["measured"] - row["sample_median"] == pytest.approx(residual, abs=TOL)
        assert (row["unit"], row["reason"]) == ("voxels", f"shift residual above the limit ({row['axis']}, "
                                                          f"{row['round']})")
    completed(case, record, tables)


def test_pr4_failed_and_running(tmp_path):
    case = report_records(tmp_path / "root")
    f2 = case.run_record("F2")
    f2.update(status="interrupted", error={"step": "register", "round": "round3", "type": "KeyboardInterrupt",
                                           "message": "", "traceback": ""})
    case.write_run_record("F2", f2)
    f3 = case.run_record("F3")
    f3.update(status="running", ended_at=None, registration={}, transforms={}, counts={}, steps=f3["steps"][:3])
    case.write_run_record("F3", f3)
    _, record, tables = collect(case)
    failed = tables["problematic_fovs"][tables["problematic_fovs"]["rule"].eq("fov_failed").to_numpy(bool)]
    assert failed[["fov_id", "section", "reason"]].values.tolist() == [
        ["F2", "Workflow overview", "run interrupted at register, round3"]]
    completed(case, record, tables)


def test_pr4_rc2_failed_reason(tmp_path):
    case = report_records(tmp_path / "root", "failed")
    _, record, tables = collect(case)
    frame = tables["problematic_fovs"]
    assert frame[frame["rule"].eq("fov_failed").to_numpy(bool)][["fov_id", "reason"]].values.tolist() == [
        ["F3", "run failed at register, round3"]]
    completed(case, record, tables)


@pytest.mark.parametrize("variant", ["not_run", "only_f1"])
def test_pr4_fov_not_run(tmp_path, variant):
    case = report_records(tmp_path / "root", variant)
    _, record, tables = collect(case)
    absent = ["F4"] if variant == "not_run" else ["F2", "F3", "F4"]
    frame = tables["problematic_fovs"]
    assert frame[frame["rule"].eq("fov_not_run").to_numpy(bool)][["fov_id", "reason"]].values.tolist() == [
        [f, "expected and not run"] for f in absent]
    completed(case, record, tables)


def test_pr4_no_run_record_at_all(tmp_path):
    anndata_or_skip()
    case = report_cells(tmp_path / "root")
    for fov in FOVS:
        (case.checkpoint(fov) / "run.json").unlink()
    _, record, tables = collect(case)
    rows = keyed(tables["unavailable"])
    for rule in ("fov_failed", "fov_not_run", "shift_residual", "registration_rejected", "registration_fallback",
                 "correct_form_fraction", "matched_fraction", "spots_near_zero"):
        assert (f"rule:{rule}", None, None, "", "no_run_record") in rows, rule
        assert record["rules"][rule]["evaluated"] is False
    # The rules of the export still hold: F2 holds retained reads and no cell.
    assert rows_of(tables) == [("F2", "reads_no_cell", None, None)]
    assert record["examples"] == {"example_1": None, "example_2": None, "example_3": None}
    assert tables["fovs"]["example"].isna().all()
    completed(case, record, tables)


def test_pr4_rejected_then_fallback(tmp_path):
    case = report_records(tmp_path / "root")
    record = case.run_record("F1")
    entries = record["registration"]["round2"]
    rejected = estimation_entry(0.2, "rejected")
    recovered = estimation_entry(0.89, fallback=True, method="phase_correlation")
    recovered["attempt"] = 1
    record["registration"]["round2"] = [rejected, recovered, entries[-1]]
    case.write_run_record("F1", record)
    _, record, tables = collect(case)
    frame = tables["problematic_fovs"]
    mine = frame[frame["fov_id"].eq("F1").to_numpy(bool)]
    assert mine[["rule", "round", "section", "reason"]].values.tolist() == [
        ["registration_rejected", "round2", "Registration", "registration rejected (round2)"],
        ["registration_fallback", "round2", "Registration",
         "registration succeeded with a fallback method (round2: phase_correlation)"]]
    completed(case, record, tables)


@pytest.mark.parametrize("variant", ["placement_states", "placement_supplied"])
def test_pr4_placement_rules(tmp_path, variant):
    case = report_records(tmp_path / "root", variant)
    _, record, tables = collect(case)
    frame = tables["problematic_fovs"]
    placement = frame[frame["section"].eq("Segmentation and assembly").to_numpy(bool)]
    expected = [["F3", "stage_position_fallback", "fell back to the stage position"],
                ["F4", "not_placed", "not placed (failed)"]] if variant == "placement_states" else []
    assert placement[["fov_id", "rule", "reason"]].values.tolist() == expected
    completed(case, record, tables)


def test_rc5_placement_rules_not_evaluated(tmp_path):
    case = report_records(tmp_path / "root", "no_placement")
    _, record, tables = collect(case)
    rows = keyed(tables["unavailable"])
    for rule in ("not_placed", "stage_position_fallback"):
        assert (f"rule:{rule}", None, None, "", "no_placement") in rows
        assert record["rules"][rule]["evaluated"] is False
    completed(case, record, tables)


def test_pr4_reads_no_cell(tmp_path):
    anndata_or_skip()
    case = report_cells(tmp_path / "root")
    _, record, tables = collect(case)
    assert rows_of(tables) == [("F2", "reads_no_cell", None, None), ("F3", "correct_form_fraction", None, None),
                               ("F4", "shift_residual", "round3", "x")]
    frame = tables["problematic_fovs"]
    assert frame["reason"].tolist() == ["reads but no cell", "correct-form fraction below the limit",
                                        "shift residual above the limit (x, round3)"]
    assert record["rules"]["reads_no_cell"] == {"limit": None, "unit": None, "evaluated": True, "fovs_marked": 1}
    completed(case, record, tables)


def test_pr_alignment_score_with_a_limit(tmp_path):
    case = report_records(tmp_path / "root")
    case.config = ReportConfig(limits=ReportLimits(alignment_score=RuleLimit(absolute=0.5)))
    _, record, tables = collect(case)
    frame = tables["problematic_fovs"]
    mine = frame[frame["rule"].eq("alignment_score").to_numpy(bool)]
    assert mine[["fov_id", "round", "measured", "limit", "unit", "reason"]].values.tolist() == [
        ["F4", "round3", 0.3, 0.5, "dimensionless", "alignment score below the limit (round3)"]]
    assert mine["sample_median"].isna().all()
    case.config = ReportConfig(limits=ReportLimits(alignment_score=RuleLimit(k=3, floor=0.1)))
    _, _, tables = collect(case)
    frame = tables["problematic_fovs"]
    mine = frame[frame["rule"].eq("alignment_score").to_numpy(bool)]
    assert mine[["fov_id", "round", "measured", "sample_median"]].values.tolist() == [
        ["F4", "round3", 0.3, pytest.approx(0.875, abs=TOL)]]


@pytest.mark.parametrize("variant", ["no_spot", "one", "equal"])
def test_rc8_rc9_rc10(tmp_path, variant):
    case = report_records(tmp_path / "root", variant)
    _, record, tables = collect(case)
    marked = rows_of(tables)
    if variant == "no_spot":
        # 0 < 0.05 × 900, the median of 1000, 0, 800, 1000; F2 enters no median of a fraction rule.
        assert ("F2", "spots_near_zero", None, None) in marked
        row = tables["problematic_fovs"][tables["problematic_fovs"]["rule"].eq("spots_near_zero").to_numpy(bool)]
        assert row[["measured", "sample_median", "limit"]].values.tolist() == [[0.0, 900.0, pytest.approx(45.0)]]
        fraction_rows = tables["problematic_fovs"][tables["problematic_fovs"]["rule"].isin(
            ["correct_form_fraction", "matched_fraction"]).to_numpy(bool)]
        assert (fraction_rows["sample_median"] == 0.8).all()
    else:
        assert not [r for r in marked if r[1] in VALUE_RULES]
        if variant == "one":
            assert marked == []
    completed(case, record, tables)


def test_reason_text_of_the_matched_fraction():
    # The other texts of "The reason texts" are checked by the tests of their rules above.
    rows, _, _ = evaluate("matched_fraction", [0.8, 0.8, 0.5, 0.8])
    assert [(r["section"], r["reason"]) for r in rows] == [("Readout", "matched fraction below the limit")]
    assert list(RULE_SECTIONS) == list(PROBLEM_RULES)
