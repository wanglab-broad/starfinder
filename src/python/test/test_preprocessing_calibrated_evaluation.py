"""Focused tests of the calibrated rerun design in benchmarks/preprocessing_synthetic.py (W-239).

The rules are those of docs/preprocessing-algorithms.md, *Evaluation design amendment for the
calibrated rerun (Accepted, W-243)*: item 2 (threshold modes), item 3 (precondition, including the
C5 set-specific multi-FOV check), item 4 (revised low-benefit rule), item 5 (added conditions) and
C7 (histogram matching's harm test).
"""
import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from starfinder.synthetic import CALIBRATED_CONDITIONS, calibrated_scene_preset, development_codebook

ROOT = Path(__file__).resolve().parents[3]


def module():
    spec = importlib.util.spec_from_file_location("preprocessing_synthetic",
                                                  ROOT / "benchmarks" / "preprocessing_synthetic.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


evaluation = module()
E = evaluation.RULE_ENDPOINTS


# --- Condition builders (item 1, item 5, C1, C4, C6, C8) -------------------------------------------

def test_calibrated_conditions_are_the_preset_as_returned_with_the_balanced_codebook():
    balanced = development_codebook("balanced").table
    for condition in CALIBRATED_CONDITIONS:
        for dtype in ("uint8", "uint16"):
            book, config = evaluation.calibrated_scene_config(condition, dtype=dtype, seed=101)
            _, preset = calibrated_scene_preset(condition, dtype, seed=101, codebook="balanced")
            assert config == preset
            pd.testing.assert_frame_equal(book.table, balanced)
    book, config = evaluation.calibrated_scene_config("clean_unbalanced", dtype="uint8", seed=3)
    _, preset = calibrated_scene_preset("clean", "uint8", seed=3, codebook="unbalanced")
    pd.testing.assert_frame_equal(book.table, development_codebook("unbalanced").table)
    assert replace(config, dataset_version=preset.dataset_version) == preset
    assert config.dataset_version == "calibrated-development-v1-clean_unbalanced-unbalanced"


@pytest.mark.parametrize("dtype, median", [("uint8", 352.0), ("uint16", 352.0 * 16)])
def test_bright_outliers_are_four_blobs_at_four_times_the_brightness_median(dtype, median):
    book, config = evaluation.calibrated_scene_config("bright_outliers", dtype=dtype, seed=0)
    _, clean = calibrated_scene_preset("clean", dtype, seed=0)
    texture = config.background.texture
    assert config.background.texture_enabled and texture.count == 4
    assert (texture.axial_width.mode, texture.axial_width.parameters) == ("constant", (1.5,))
    assert (texture.lateral_width.mode, texture.lateral_width.parameters) == ("constant", (2.0,))
    assert texture.brightness.mode == "lognormal" and texture.brightness.parameters[1] == 0.1
    assert np.exp(texture.brightness.parameters[0]) == pytest.approx(median, rel=1e-12)
    assert np.exp(clean.brightness.parameters[0]) * 4 == pytest.approx(median, rel=1e-12)
    assert np.array_equal(config.background.tissue_weights, np.ones((4, 4)))
    unchanged = replace(config, dataset_version=clean.dataset_version, background=clean.background)
    assert unchanged == clean


def test_saturation_scales_the_uint16_intensity_parameters_by_k():
    _, clean = calibrated_scene_preset("clean", "uint8", seed=1)
    _, one = evaluation.calibrated_scene_config("saturation", dtype="uint8", seed=1, k=1.0)
    assert replace(one, dataset_version=clean.dataset_version) == clean
    k = 2.0 ** (5 / 4)
    _, config = evaluation.calibrated_scene_config("saturation", dtype="uint8", seed=1, k=k)
    assert np.exp(config.brightness.parameters[0]) == pytest.approx(88.0 * k, rel=1e-12)
    assert config.brightness.parameters[1] == clean.brightness.parameters[1]
    np.testing.assert_allclose(config.background.baseline, np.asarray(clean.background.baseline) * k)
    for field in ("alpha", "sigma", "correlated_sigma"):
        assert getattr(config.noise, field) == pytest.approx(getattr(clean.noise, field) * k)
    rest = replace(config, dataset_version=clean.dataset_version, brightness=clean.brightness,
                   background=clean.background, noise=clean.noise)
    assert rest == clean
    with pytest.raises(ValueError, match="searched k"):
        evaluation.calibrated_scene_config("saturation", dtype="uint8", seed=1)


def test_saturation_k_is_the_smallest_grid_value_reaching_the_fraction_on_development_seeds():
    calls = []

    def fraction(dtype, seed, k):
        calls.append((dtype, seed, k))
        return 1e-3 * k / 2 * (1 + (seed - 1) * 0.01)  # the mean over seeds 0-2 is 1e-3 * k / 2

    result = evaluation.saturation_k("uint8", fraction=fraction)
    assert (result["j"], result["k"]) == (4, 2.0)
    assert result["mean_clipped_fraction"] >= evaluation.SATURATION_FRACTION
    assert all(r["mean_clipped_fraction"] < 1e-3 for r in result["trace"][:-1])
    assert {seed for _, seed, _ in calls} == set(evaluation.DEV_SEEDS)
    assert set(result["per_seed_clipped_fraction"]) == {"0", "1", "2"}
    with pytest.raises(RuntimeError, match="no k"):
        evaluation.saturation_k("uint8", fraction=lambda dtype, seed, k: 0.0)


def test_clipped_fraction_counts_voxels_above_the_dtype_range():
    class Scene:
        provenance = {"clipping_counts": {"round1": dict(below=7, above=3), "round2": dict(below=0, above=5)}}
        rounds = {"round1": np.zeros((2, 4, 4, 4)), "round2": np.zeros((2, 4, 4, 4))}
    assert evaluation.clipped_fraction(Scene()) == 8 / 256


def test_gain_strong_has_the_ln_channel_spread_and_keeps_the_geometric_mean():
    _, clean = calibrated_scene_preset("clean", "uint8", seed=0)
    _, config = evaluation.calibrated_scene_config("gain_strong", dtype="uint8", seed=0)
    gains, base = np.asarray(config.readout.gains), np.asarray(clean.readout.gains)
    for row, clean_row in zip(gains, base):
        assert row.max() / row.min() == pytest.approx(3.15, rel=1e-12)
        assert np.all(np.diff(row) < 0)
        assert np.exp(np.mean(np.log(row))) == pytest.approx(np.exp(np.mean(np.log(clean_row))), rel=1e-12)
    assert replace(config, dataset_version=clean.dataset_version, readout=clean.readout) == clean


def test_multi_fov_fovs_replace_count_gains_and_scene_key():
    _, clean = calibrated_scene_preset("combined", "uint8", seed=100)
    _, config = evaluation.calibrated_scene_config("combined", dtype="uint8", seed=100, fov_id="Position003",
                                                   gain=0.5, count=2, scene_key="k/Position003")
    assert (config.count, config.FOV_id, config.scene_key) == (2, "Position003", "k/Position003")
    assert config.coordinates is None and config.amplicon_ids is None and config.gene_ids is None
    np.testing.assert_allclose(config.readout.gains, np.asarray(clean.readout.gains) * 0.5)


def test_background_radius_caps_r_z_at_three_on_eight_planes():
    _, config = evaluation.calibrated_scene_config("regions", dtype="uint8", seed=0)
    assert config.shape_zyx[0] == 8
    assert evaluation.calibrated_background_radius(config) == ((3, 5, 5), (6, 5, 5))


def test_plan_follows_c2_and_the_pilot_covers_every_new_path():
    full = evaluation.calibrated_plan("full")
    uint8 = [c for c, d in full["single_fov"] if d == "uint8"]
    assert uint8 == list(CALIBRATED_CONDITIONS) + ["bright_outliers", "saturation", "gain_strong", "clean_unbalanced"]
    assert sorted(c for c, d in full["single_fov"] if d == "uint16") == ["bright_outliers", "clean", "combined"]
    assert {d for _, d in full["multi_fov"]} == {"uint8"} and len(full["multi_fov"]) == 3
    assert full["seeds"] == (0, 1, 2, 100, 101, 102)
    reduced = evaluation.calibrated_plan("full", ("R3",))
    assert {d for _, d in reduced["single_fov"]} == {"uint8"} and len(reduced["single_fov"]) == 17
    with pytest.raises(ValueError, match="R3"):
        evaluation.calibrated_plan("full", ("R1",))
    pilot = evaluation.calibrated_plan("pilot")
    assert set(evaluation.ADDED_CONDITIONS) <= {c for c, d in pilot["single_fov"] if d == "uint8"}
    assert {d for _, d in pilot["single_fov"]} == {"uint8", "uint16"}
    assert {s for s, _ in pilot["multi_fov"]} == {"mf_density", "mf_density_clean"}
    assert pilot["seeds"] == (0, 100)


# --- Threshold modes and operating points (item 2) -------------------------------------------------

def test_threshold_mode_grids_and_fixed_points():
    modes = evaluation.THRESHOLD_MODES
    assert modes["noise"] == dict(grid=(2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0), fixed=(5.0,))
    assert modes["adaptive"] == dict(grid=(0.1, 0.15, 0.2, 0.25, 0.3, 0.4), fixed=(0.2, 0.4))


def _dev(values):
    return pd.DataFrame([dict(seed=s, threshold=t, f1=f) for t, row in values.items() for s, f in zip((2, 0, 1), row)])


def test_operating_point_ties_go_to_the_smallest_value_and_grid_edges_are_marked():
    grid = evaluation.THRESHOLD_MODES["noise"]["grid"]
    values = {t: (0.5, 0.5, 0.5) for t in grid}
    values[3.0] = values[6.0] = (0.7, 0.8, 0.9)  # exact tie in the float64 mean
    assert evaluation.select_operating_point(_dev(values), grid) == (3.0, pytest.approx(0.8), False)
    values[2.0] = (0.9, 0.9, 0.9)
    value, mean, edge = evaluation.select_operating_point(_dev(values), grid)
    assert (value, edge) == (2.0, True)
    adaptive = evaluation.THRESHOLD_MODES["adaptive"]["grid"]
    values = {t: (0.1 * i, 0.1 * i, 0.1 * i) for i, t in enumerate(adaptive)}
    assert evaluation.select_operating_point(_dev(values), adaptive)[::2] == (0.4, True)
    values[0.4] = (1.0, float("nan"), 1.0)  # undefined on one seed: not selectable
    assert evaluation.select_operating_point(_dev(values), adaptive)[::2] == (0.3, False)
    # A plain tie at every value selects the smallest (the lower grid edge).
    assert evaluation.select_operating_point(_dev({t: (0.4, 0.4, 0.4) for t in adaptive}), adaptive)[::2] == (0.1, True)


def test_summarize_modes_reports_each_mode_with_its_own_selection_and_fixed_points():
    rows = []
    for mode, spec in evaluation.THRESHOLD_MODES.items():
        for seed in (0, 1, 2, 100, 101, 102):
            for i, value in enumerate(spec["grid"]):
                f1 = 0.9 if (mode == "noise" and value == 4.0) or (mode == "adaptive" and value == 0.1) else 0.5
                rows.append(dict(condition="clean", dtype="uint8", arm="none", seed=seed, threshold_mode=mode,
                                 threshold=value, f1=f1, precision=f1, recall=f1, localization_error=0.1,
                                 correct_fraction=f1 / 2, reads_correct=1, reads_wrong_gene=0,
                                 reads_false_detection=i, color_call_agreement=1.0))
    per_seed, points, selected, ends = evaluation.summarize_modes(pd.DataFrame(rows), ["condition", "dtype", "arm"])
    assert selected[("noise", "clean", "uint8", "none")] == dict(value=4.0, dev_mean_f1=pytest.approx(0.9), at_grid_edge=False)
    assert selected[("adaptive", "clean", "uint8", "none")] == dict(value=0.1, dev_mean_f1=pytest.approx(0.9), at_grid_edge=True)
    table = points.set_index(["threshold_mode", "operating_point", "threshold"])
    assert set(points.threshold_mode) == {"noise", "adaptive"}
    assert sorted(points[points.operating_point == "fixed"].threshold) == [0.2, 0.4, 5.0]
    assert bool(table.loc[("adaptive", "dev_max_f1", 0.1), "at_grid_edge"])
    ends = ends.set_index(["threshold_mode", "seed"])
    assert ends.loc[("noise", 100), "correct_fraction"] == 0.45 and ends.loc[("noise", 100), "max_f1"] == 0.9
    assert ends.loc[("adaptive", 101), "f1_fixed1"] == 0.5 and ends.loc[("adaptive", 101), "fixed2_threshold"] == 0.4
    assert np.isnan(ends.loc[("noise", 102), "f1_fixed2"])
    assert set(per_seed.split) == {"development", "evaluation"}


# --- Precondition (item 3, C5) ---------------------------------------------------------------------

def test_precondition_is_strict_after_rounding_and_undefined_endpoints_do_not_degrade():
    # clean - condition = 0.1 on every seed and both ranges are 0.1; in float64 the mean difference
    # (0.10000000000000002) exceeds the range (0.1), so only the 10-decimal rounding makes this the
    # equality case, which does not degrade.
    clean, condition = [0.1, 0.1, 0.2], [0.0, 0.0, 0.1]
    assert np.mean(np.subtract(clean, condition)) > max(np.ptp(clean), np.ptp(condition))
    result = evaluation.precondition_check({"max_f1": clean, "correct_fraction": clean},
                                           {"max_f1": condition, "correct_fraction": condition})
    assert (result["g_max_f1"], result["range_max_f1"]) == (0.1, 0.1)
    assert not result["degrades_max_f1"] and not result["valid"]
    strict = evaluation.precondition_check({"max_f1": [0.9, 0.9, 0.9], "correct_fraction": clean},
                                           {"max_f1": [0.6, 0.65, 0.7], "correct_fraction": condition})
    assert strict["degrades_max_f1"] and strict["valid"] and strict["g_max_f1"] == 0.25
    assert (strict["range_max_f1"], strict["degrades_correct_fraction"]) == (0.1, False)
    undefined = evaluation.precondition_check({"max_f1": [0.9, 0.9, 0.9], "correct_fraction": [0.9, 0.9, 0.9]},
                                              {"max_f1": [0.1, None, 0.1], "correct_fraction": [0.9, 0.9, 0.9]})
    assert undefined["g_max_f1"] is None and not undefined["degrades_max_f1"] and not undefined["valid"]
    assert "max_f1 undefined" in undefined["note"]
    unpaired = evaluation.precondition_check({"max_f1": [0.9, 0.9, 0.9]}, {"max_f1": [0.1, 0.1]})
    assert not unpaired["valid"]


def _ends(rows):
    return pd.DataFrame([dict(condition=c, dtype=d, arm=a, threshold_mode=m, seed=s, max_f1=f, correct_fraction=q)
                         for c, d, a, m, values in rows for s, (f, q) in zip((100, 101, 102), values)])


def test_fixture_check_scope_and_fixture_gap_rows():
    good = [(0.9, 0.8)] * 3
    ends = _ends([("clean", "uint8", "none", "noise", good),
                  ("baseline", "uint8", "none", "noise", [(0.5, 0.4)] * 3),   # degrades both endpoints
                  ("gain", "uint8", "none", "noise", good),                   # does not degrade: fixture gap
                  ("combined", "uint8", "none", "noise", [(0.1, 0.1)] * 3),   # reported only, no check
                  ("clean_unbalanced", "uint8", "none", "noise", [(0.1, 0.1)] * 3),
                  ("background_only", "uint8", "none", "noise", [(0.1, 0.1)] * 3),  # not targeted
                  ("baseline", "uint8", "scalar", "noise", good)])
    table = evaluation.preconditions_table(ends).set_index("condition")
    assert sorted(table.index) == ["baseline", "gain"]
    assert table.loc["baseline", "valid"] and not table.loc["gain", "valid"]
    assert table.loc["gain", "targeted_by"] == "histogram_matching,min_max_normalization,percentile_normalization"
    gaps = evaluation.fixture_gaps(evaluation.preconditions_table(ends))
    assert list(gaps.condition) == ["gain"] and {"g_max_f1", "range_max_f1", "g_correct_fraction",
                                                  "range_correct_fraction"} <= set(gaps)


def test_set_specific_check_gates_each_recipes_sample_level_flag():
    per_fov = pd.DataFrame([dict(multi_fov="mf_density", dtype="uint8", arm=arm, threshold_mode="adaptive", seed=s,
                                 fov_id=f"Position00{i}", role=role, max_f1=f, correct_fraction=f)
                            for arm, near_empty in (("r1", 0.2), ("r2_scalar", 0.9), ("r1_sample", 0.9))
                            for s in (100, 101, 102)
                            for i, (role, f) in enumerate((("dense", 0.9), ("sparse", 0.9),
                                                           ("near_empty", near_empty)), 1)])
    # The fixture check (none on the set against mf_density_clean) fails here, and is reported only.
    mf_ends = _ends([(name, "uint8", arm, "adaptive", [(0.8, 0.7)] * 3)
                     for name in ("mf_density", "mf_density_clean") for arm in ("none", "r1", "r1_sample",
                                                                                 "r2_scalar", "r2_scalar_sample")])
    table = evaluation.preconditions_table(pd.DataFrame(), mf_ends, per_fov)
    assert sorted(zip(table.check, table.arm)) == [("fixture", "none"), ("set_specific", "r1"),
                                                   ("set_specific", "r2_scalar")]
    checks = table.set_index(["check", "arm"])
    assert not checks.loc[("fixture", "none"), "valid"] and not checks.loc[("fixture", "none"), "gates_flag"]
    assert checks.loc[("set_specific", "r1"), "valid"] and checks.loc[("set_specific", "r1"), "g_max_f1"] == 0.7
    assert not checks.loc[("set_specific", "r2_scalar"), "valid"]
    assert (checks.loc[("set_specific", "r1"), "reference"], checks.loc[("set_specific", "r1"), "degraded"]) == (
        "dense", "near_empty")
    rows = pd.DataFrame([dict(method="sample_level_fitting", comparison=f"ablation_{r}", before=r, after=f"{r}_sample",
                              condition="mf_density", role="targeted", dtype="uint8", threshold_mode="adaptive")
                         for r in ("r1", "r2_scalar")])
    lookups = dict(single_fov={}, multi_fov=evaluation._endpoint_lookup(mf_ends))
    attached = evaluation.attach_rule(rows, lookups, table).set_index("before")
    assert attached.loc["r1", "eligible"] and not attached.loc["r2_scalar", "eligible"]
    assert "set_specific failed" in attached.loc["r2_scalar", "precondition"]
    assert "fixture failed (reported only, C5)" in attached.loc["r1", "precondition"]
    assert attached.loc["r1", "rule_delta_max_f1"] == 0.0
    flags = evaluation.revised_flags(attached.reset_index())
    method = flags[flags.level == "method"].set_index("before")
    assert method.loc["r2_scalar", "result"] == "not_assessable"
    assert "fixture gap mf_density" in method.loc["r2_scalar", "reason"]
    assert method.loc["r1", "result"] == "low_benefit"


# --- Revised low-benefit rule (item 4) and harm test (C7) -------------------------------------------

def _row(condition, role, max_f1, correct, eligible=True, method="background_3d", comparison="isolated",
         before="none", after="bg3d", dtype="uint8", mode="noise"):
    """A hand-made comparison row: (delta, range) per endpoint, as attach_rule writes them."""
    row = dict(method=method, comparison=comparison, before=before, after=after, condition=condition, role=role,
               dtype=dtype, threshold_mode=mode)
    for endpoint, (delta, span) in zip(E, (max_f1, correct)):
        row.update({f"rule_delta_{endpoint}": delta, f"rule_range_{endpoint}": span})
    if role == "targeted":
        row.update(eligible=eligible, fixture_gap=not eligible,
                   precondition="fixture passed" if eligible else "fixture failed")
    return row


def _result(rows):
    flags = evaluation.revised_flags(pd.DataFrame(rows))
    return flags[flags.level == "method"].iloc[0], flags[flags.level == "targeted_condition"].set_index("condition")


def test_the_w233_case_where_one_endpoint_rose_and_the_other_fell_is_low_benefit():
    # W-233, 3D background on gradient (uint8, isolated): the correct-decode fraction rose by 0.53 while
    # max-F1 fell by 0.87. The old flag counted this as a benefit.
    rows = [_row("gradient", "targeted", (-0.87, 0.03), (0.53, 0.05)),
            _row("regions", "targeted", (0.0, 0.01), (0.01, 0.01)),
            _row("clean", "harm_check", (0.0, 0.02), (0.0, 0.02))]
    method, targeted = _result(rows)
    assert method.result == "low_benefit" and method.low_benefit and method.provisional
    assert method.threshold_points == 0.02
    assert not targeted.loc["gradient", "benefit"]
    assert targeted.loc["gradient", "clause"] == ("correct_fraction improved but max_f1 worsened beyond its seed "
                                                  "range (-0.8700 < -0.0300)")
    assert targeted.loc["regions", "clause"] == "no endpoint improved by at least max(seed range, 0.02)"


def test_benefit_bound_is_inclusive_and_clean_equal_to_the_range_holds():
    rows = [_row("gradient", "targeted", (0.02, 0.01), (-0.03, 0.03)),  # delta = max(R, 0.02); the other = -R
            _row("clean", "harm_check", (-0.02, 0.02), (0.0, 0.0)),
            _row("combined", "combined", (-0.9, 0.0), (-0.9, 0.0))]      # reported, not in the flag
    method, targeted = _result(rows)
    assert targeted.loc["gradient", "benefit"] and targeted.loc["gradient", "clause"] == ""
    assert method.result == "not_flagged" and method.clean_holds
    assert method.reason == "benefit on gradient; clean holds"


def test_clean_harm_and_undefined_values_are_low_benefit_with_named_clauses():
    rows = [_row("gradient", "targeted", (0.3, 0.05), (0.2, 0.05)),
            _row("clean", "harm_check", (-0.1, 0.05), (0.0, 0.0))]
    method, _ = _result(rows)
    assert method.result == "low_benefit"
    assert method.reason == "clean does not hold: max_f1 worsened on clean beyond its seed range (-0.1000 < -0.0500)"
    rows = [_row("gradient", "targeted", (0.3, 0.05), (None, None)),
            _row("clean", "harm_check", (0.0, 0.0), (0.0, 0.0))]
    method, targeted = _result(rows)
    assert method.result == "low_benefit"
    assert targeted.loc["gradient", "clause"] == "max_f1 improved but correct_fraction is undefined"
    assert "gradient: undefined correct_fraction" in method.anomalies


def test_empty_eligible_targets_are_not_assessable_and_still_report_clean():
    rows = [_row("gradient", "targeted", (0.5, 0.0), (0.5, 0.0), eligible=False),
            _row("clean", "harm_check", (-0.5, 0.01), (0.0, 0.0))]
    method, targeted = _result(rows)
    assert method.result == "not_assessable" and not method.low_benefit and method.eligible_targets == 0
    assert "fixture gap gradient (fixture failed)" in method.reason
    assert "clean does not hold" in method.reason
    assert targeted.loc["gradient", "clause"] == "fixture gap, not in T*: fixture failed"
    rows = [_row("clean", "harm_check", (0.0, 0.0), (0.0, 0.0), dtype="uint16")]
    method, _ = _result(rows)
    assert method.reason == "T* is empty: no targeted condition was run in this dtype"


def test_histogram_matching_harm_test_is_reported_beside_its_flag():
    common = dict(method="histogram_matching", comparison="isolated", before="none", after="hist")
    rows = [_row("gain", "targeted", (0.1, 0.02), (0.1, 0.02), **common),
            _row("clean", "harm_check", (0.0, 0.01), (-0.01, 0.01), **common),
            _row("clean_unbalanced", "harm_test", (-0.2, 0.05), (-0.01, 0.05), **common),
            _row("bright_outliers", "reported", (-0.5, 0.0), (-0.5, 0.0), **common),
            _row("gain", "targeted", (0.1, 0.02), (0.1, 0.02), method="min_max_normalization", after="minmax"),
            _row("clean", "harm_check", (0.0, 0.0), (0.0, 0.0), method="min_max_normalization", after="minmax")]
    comparisons = pd.DataFrame(rows)
    flags = evaluation.revised_flags(comparisons)
    harm = evaluation.harm_test_table(comparisons, flags)
    assert len(harm) == 1
    record = harm.iloc[0]
    assert record.harm and record.harm_endpoints == "max_f1" and not record.enters_flag
    assert record.flag_result == "not_flagged"  # the harm test does not change the flag (C7)
    assert (record.unbalanced_delta_max_f1, record.clean_delta_correct_fraction) == (-0.2, -0.01)
    assert set(flags[flags.method == "histogram_matching"].condition) == {"gain", "clean", "all targeted"}


def test_extra_roles_add_reported_and_harm_test_rows():
    assert evaluation.calibrated_extra_roles("histogram_matching", "ablation") == [
        ("bright_outliers", "reported"), ("clean_unbalanced", "harm_test")]
    assert evaluation.calibrated_extra_roles("min_max_normalization", "isolated") == [("bright_outliers", "reported")]
    assert evaluation.calibrated_extra_roles("percentile_normalization", "isolated") == []
    targets = evaluation.calibrated_targets()
    assert targets["bright_outliers"] == ["percentile_normalization"]
    assert targets["saturation"] == ["extraction_source"]
    assert "texture" not in [c for m, _, _, _, t in evaluation.CALIBRATED_COMPARISONS
                             if m == "percentile_normalization" for c in t]


# --- Default-tier smoke test ------------------------------------------------------------------------

def test_calibrated_smoke_run_in_both_threshold_modes(tmp_path):
    output = tmp_path / "evaluation"
    manifest = evaluation.run_calibrated(output, scope="smoke", shape=(8, 32, 32), count=12, log=lambda m: None)
    assert manifest["schema"] == evaluation.CALIBRATED_SCHEMA and manifest["design"] == "calibrated"
    for mode, spec in evaluation.THRESHOLD_MODES.items():
        record = manifest["threshold_subset_verification"][mode]
        assert record["threshold_value"] == spec["fixed"][0] and record["every_arm_verified"]
        assert record["runs_verified"] == 2 * len(evaluation.PRIMARY_ARMS)
    assert {tuple(s["background_radius_voxels_zyx"]) for s in manifest["scenes"]} == {(3, 5, 5)}
    assert manifest["plan"]["approved_matrix"]["uint16_conditions"] == ["bright_outliers", "clean", "combined"]
    curves = pd.read_csv(output / "curves" / "pr_curves.csv")
    for mode, spec in evaluation.THRESHOLD_MODES.items():
        part = curves[curves.threshold_mode == mode]
        assert sorted(set(part.threshold)) == sorted(spec["grid"])
        assert len(part) == 2 * len(evaluation.PRIMARY_ARMS) * len(spec["grid"])
    points = pd.read_csv(output / "tables" / "operating_points.csv")
    assert {"threshold_mode", "at_grid_edge"} <= set(points)
    fixed = points[points.operating_point == "fixed"].groupby("threshold_mode").threshold.unique()
    assert sorted(fixed["noise"]) == [5.0] and sorted(fixed["adaptive"]) == [0.2, 0.4]
    statistics = pd.read_csv(output / "tables" / "image_statistics.csv")
    assert set(statistics.statistic) == set(evaluation.TARGET_RANGES)
    assert statistics.target_min.notna().all() and set(statistics.condition) == {"clean"}
    flags = pd.read_csv(output / "tables" / "low_benefit_flags.csv")
    assert set(flags[flags.level == "method"].result) == {"not_assessable"}
    for entry in manifest["files"]:
        assert (output / entry["path"]).stat().st_size == entry["bytes"]
    assert json.loads((output / "manifest.json").read_text())["artifact_bytes"] == manifest["artifact_bytes"]
