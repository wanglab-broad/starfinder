"""Default-tier smoke test of benchmarks/preprocessing_synthetic.py (W-233) on one tiny condition."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]


def module():
    spec = importlib.util.spec_from_file_location("preprocessing_synthetic",
                                                  ROOT / "benchmarks" / "preprocessing_synthetic.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


evaluation = module()


def test_smoke_run_writes_tables_and_manifest(tmp_path):
    output = tmp_path / "evaluation"
    manifest = evaluation.run(output, scope="smoke", shape=(10, 32, 32), count=12, log=lambda message: None)
    assert manifest["schema"] == evaluation.SCHEMA and manifest["issue"] == "W-233"
    assert manifest["parameters"]["thresholds"] == list(evaluation.THRESHOLDS)
    assert manifest["parameters"]["matching"]["function"] == "starfinder.evaluation.match_points"
    assert {s["seed"] for s in manifest["scenes"]} == {0, 100}
    assert {r["arm"] for r in manifest["runs"]} == set(evaluation.PRIMARY_ARMS)
    assert all(len(r["output_sha256"]) == 64 for r in manifest["runs"])
    assert manifest["threshold_subset_verification"]["runs_verified"] == 2 * len(evaluation.PRIMARY_ARMS)
    for entry in manifest["files"]:
        assert (output / entry["path"]).stat().st_size == entry["bytes"]
    curves = pd.read_csv(output / "curves" / "pr_curves.csv")
    assert len(curves) == 2 * len(evaluation.PRIMARY_ARMS) * len(evaluation.THRESHOLDS)
    for column in ("precision", "recall", "localization_error", "reads_correct", "reads_wrong_gene",
                   "reads_false_detection"):
        assert column in curves
    clean = curves[curves.arm == "none"]
    assert (clean.n_truth > 0).all() and clean.recall.between(0, 1).all()
    points = pd.read_csv(output / "tables" / "operating_points.csv")
    assert set(points.operating_point) == {"dev_max_f1", "default"}
    direct = pd.read_csv(output / "tables" / "direct_metrics.csv")
    assert direct.set_index("arm").loc[["none", "scalar", "bg3d", "r1", "r1_recon", "recon"], "bg_rmse"].notna().all()
    assert pd.read_csv(output / "tables" / "recipe_per_seed.csv").auprc.notna().all()
    diagnostics = pd.read_csv(output / "tables" / "diagnostics.csv")
    assert {"zero_fraction", "median", "mad", "noise_threshold", "mad_zero"} <= set(diagnostics)
    assert json.loads((output / "manifest.json").read_text())["artifact_bytes"] == manifest["artifact_bytes"]


def test_shared_arms_equal_direct_recipe_runs(tmp_path):
    book, config = evaluation.scene_config("baseline", dtype="uint16", seed=1, shape=(10, 32, 32), count=12)
    scene, *_ = evaluation.generate(book, config)
    radius = evaluation.background_radius(config)
    recipes = evaluation.arms(radius)
    shared = {arm: fov for arm, _, fov in evaluation.processed_arms(scene, book, config.FOV_id, tmp_path, radius, False)
              if arm in evaluation.SHARED}
    for arm, fov in shared.items():
        direct = evaluation.preprocess(evaluation.make_fov(scene, book, config.FOV_id, tmp_path), recipes[arm], False)
        for name in book.round_labels:
            np.testing.assert_array_equal(fov.images[name], direct.images[name])
        assert fov.preprocessing_record.get("recipe", {}).get("extraction_source") is None


def test_auprc_integrates_in_threshold_order_and_skips_undefined_precision():
    # Hand-computed: from the highest threshold down, recall 0 (no detections), 0.5 at precision 1.0,
    # 0.5 again at precision 0.5 (no new area), then 1.0 at precision 0.25:
    # 0.5 * 1.0 + 0.5 * 0.25 = 0.625. Sorting equal recalls by precision would give 0.375.
    rows = [dict(threshold=2.0, recall=1.0, precision=0.25), dict(threshold=5.0, recall=0.5, precision=0.5),
            dict(threshold=10.0, recall=0.5, precision=1.0), dict(threshold=15.0, recall=0.0, precision=float("nan"))]
    assert evaluation._auprc(rows) == 0.625
    assert evaluation._auprc([dict(threshold=t, recall=0.0, precision=float("nan")) for t in (2.0, 5.0)]) == 0.0


def test_normalized_truth_follows_the_fitted_value_map():
    raw = np.array([0, 10, 20, 10], dtype=np.uint8).reshape(1, 1, 4, 1)
    normalized = (raw * 2).astype(np.uint8)
    background = np.array([5.0, 15.0, 30.0, 0.0]).reshape(1, 1, 4, 1)
    np.testing.assert_array_equal(evaluation._normalized_truth(raw, normalized, background).ravel(), [10, 30, 40, 0])


def test_comparison_ranges_are_across_seeds_after_channel_aggregation():
    ends = pd.DataFrame([dict(condition=c, dtype="uint8", arm=a, seed=s, **{m: 0.5 for m in evaluation.ENDPOINTS})
                         for c in ("baseline", "clean", "combined") for a in ("none", "scalar")
                         for s in evaluation.EVAL_SEEDS])
    diagnostics = pd.DataFrame([dict(condition=c, dtype="uint8", arm=a, seed=s, channel=k, zero_fraction=0.0,
                                     median=1.0, mad=mad, noise_threshold=1.0, mad_zero=False)
                                for c in ("baseline", "clean", "combined") for a in ("none", "scalar")
                                for s in evaluation.EVAL_SEEDS for k, mad in enumerate((9.0, 10.0))])
    spec = [("scalar_background", "isolated", "none", "scalar", ("baseline",))]
    table = evaluation.comparisons_table(ends, None, diagnostics, spec, "clean", "combined")
    row = table[(table.condition == "baseline") & (table.dtype == "uint8")].iloc[0]
    assert (row.mad_after_mean, row.mad_after_min, row.mad_after_max) == (9.5, 9.5, 9.5)
    assert row.reads_wrong_gene_after_mean == 0.5 and row.reads_false_detection_t5_delta_mean == 0.0


def test_background_ablation_rows_have_before_and_after_background_error(tmp_path):
    # pct has no background step: its estimate is 0, measured in input units before normalization,
    # the stage of r2's bg_corrected snapshot, so pct -> r2 ablations report both sides.
    book, config = evaluation.scene_config("baseline", dtype="uint8", seed=100, shape=(10, 32, 32), count=12)
    scene, background, signal, truth = evaluation.generate(book, config)
    radius = evaluation.background_radius(config)
    rows = []
    for arm, recipe, fov in evaluation.processed_arms(scene, book, config.FOV_id, tmp_path, radius, False):
        rows.append(dict(condition="baseline", dtype="uint8", seed=100, arm=arm,
                         **evaluation.direct_metrics(fov, scene.rounds, background, signal, truth, book, arm, recipe)))
    direct = pd.DataFrame(rows).set_index("arm")
    assert direct.loc["pct", "bg_units"] == direct.loc["r2_scalar", "bg_units"] == "input intensity"
    assert direct.loc["pct", "bg_bias"] == -direct.loc["pct", "bg_truth_mean"]
    ends = pd.DataFrame([dict(r, **{m: 0.5 for m in evaluation.ENDPOINTS}) for r in rows])
    specs = [c for c in evaluation.COMPARISONS if c[0] in ("scalar_background", "background_3d") and c[1] == "ablation"]
    table = evaluation.comparisons_table(ends, pd.DataFrame(rows), None, [(m, c, b, a, ("baseline",))
                                         for m, c, b, a, _ in specs], "baseline", "baseline")
    assert len(table) and table[[f"bg_{m}_{side}_mean" for m in ("rmse", "bias") for side in ("before", "after")]
                                ].notna().all().all()
