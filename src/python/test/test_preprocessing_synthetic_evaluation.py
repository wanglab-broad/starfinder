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
    assert direct.set_index("arm").loc[["none", "scalar", "bg3d"], "bg_rmse"].notna().all()
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
