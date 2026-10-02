"""localization_errors and classify_detections against the saved W-266 and W-267 records
(W-270, docs/spot-finding-algorithms.md, "Metrics that must be added"; data in data/spot_finding_metrics)."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from starfinder.evaluation.spot_finding import classify_detections, evaluate_spots, localization_errors
from starfinder.image import ImageMetadata

pytestmark = [pytest.mark.spot_finding, pytest.mark.evaluation]

DATA = Path(__file__).parent / "data" / "spot_finding_metrics"
W266_RUNS = ["0076-isolated-iso3d-100-spotiflow-smfish_3d", "0078-isolated-iso3d-100-piscis-20251212"]
# W-266's matching on the isolated-spot scenes and W-267's on the W-218 scene.
W266_MATCH = dict(policy="greedy", threshold=3.0, units="voxel", boundary="inclusive")
W218_MATCH = dict(policy="greedy", threshold=5.0, units="voxel", boundary="exclusive")
METRICS = ["abs_z_max", "abs_z_p95", "abs_y_max", "abs_y_p95", "abs_x_max", "abs_x_p95",
           "lateral_max", "lateral_p95", "dist_max", "dist_p95"]


def isolated_positions(case, seed):
    """W-266 w266_common.isolated_positions for the iso3d and iso_z1 cases (truth centres, ZYX voxels)."""
    rng = np.random.default_rng([266, seed, {"iso3d": 0, "iso_z1": 1}[case]])
    yx = np.arange(5, 60, 6, dtype=float)
    layers = (8.0, 16.0, 24.0) if case == "iso3d" else (0.0, 0.0, 0.0)
    grid = np.array([(layers[(i + j) % 3], y, x) for i, y in enumerate(yx) for j, x in enumerate(yx)])
    jitter = rng.uniform(-0.5, 0.5, size=grid.shape)
    if case == "iso_z1":
        jitter[:, 0] = 0.0
    return grid + jitter


def evaluate(detected, truth, **match):
    frame = ImageMetadata("saved-records")
    return evaluate_spots(detected, truth, reference_metadata=frame, observed_metadata=frame, **match)


@pytest.mark.parametrize("run_id", W266_RUNS)
def test_localization_errors_reproduce_w266(run_id):
    with open(DATA / "localization-per-axis.csv", newline="") as handle:
        expected = {row["run_id"]: row for row in csv.DictReader(handle)}[run_id]
    detected = np.load(DATA / f"{run_id}.npz")["points"].reshape(-1, 3)
    truth = isolated_positions("iso3d", 100)
    assert (len(detected), len(truth)) == (int(expected["n_detections"]), int(expected["n_truth"]))
    errors = localization_errors(evaluate(detected, truth, **W266_MATCH), detected, truth)
    assert list(errors.values) == METRICS and errors.status == "ok"
    assert set(errors.units.values()) == {"voxel"}
    # The CSV prints each value with repr, which round-trips. The comparison allows 1e-12 relative,
    # because the last bit of hypot and of the percentile interpolation depends on the CPU.
    for name in METRICS:
        assert errors.values[name] == pytest.approx(float(expected[name]), rel=1e-12, abs=1e-12), name
    assert errors.counts == {"matched": int(expected["matched"]), "n_abs_z_gt_1": int(expected["n_abs_z_gt_1"])}


def test_localization_errors_without_matches():
    truth = isolated_positions("iso_z1", 100)
    errors = localization_errors(evaluate(np.empty((0, 3)), truth, **W266_MATCH), np.empty((0, 3)), truth)
    assert errors.counts == {"matched": 0, "n_abs_z_gt_1": 0} and errors.status == "undefined"
    assert all(errors.values[name] is None for name in METRICS)


def w218_records():
    run = json.loads((DATA / "w218-remeasure-notebook-run.json").read_text())["notebook_run"]
    rows = run["rows"]
    centres = {row["amplicon"]: (row["z"] - row["dz"], row["y"] - row["dy"], row["x"] - row["dx"])
               for row in rows if row["cls"] == "matched"}
    centres.update({missed["amplicon"]: (missed["z"], missed["y"], missed["x"]) for missed in run["missed"]})
    detected = np.array([(row["z"], row["y"], row["x"]) for row in rows])
    channels = np.array([row["channel"] for row in rows])
    return run, detected, np.array(list(centres.values())), channels


def test_classify_detections_reproduces_the_w218_counts():
    run, detected, truth, channels = w218_records()
    assert (len(detected), len(truth)) == (run["candidates"], run["truth"]) == (68, 50)
    match = evaluate(detected, truth, **W218_MATCH)
    assert match.counts["matched"] == run["counts"]["matched"] == 45
    counts = classify_detections(match, detected, truth, radius=5.0, groups=channels).counts
    assert counts == {"matched": 45, "duplicate": 22, "spurious": 1, "duplicate_same_group": 0,
                      "duplicate_other_group": 22}
    assert {k: counts[k] for k in ("matched", "duplicate", "spurious")} == run["geometry_classes"]
    assert run["duplicate_kinds"] == {"cross_channel": 22}
    assert classify_detections(match, detected, truth, radius=5.0).counts == {
        "matched": 45, "duplicate": 22, "spurious": 1}


def test_classify_detections_per_detection_classes_match_the_saved_rows():
    run, detected, truth, channels = w218_records()
    match = evaluate(detected, truth, **W218_MATCH)
    matched = {j for _, j, _ in match.details["matched_pairs"]}
    assert matched == {j for j, row in enumerate(run["rows"]) if row["cls"] == "matched"}


def test_classify_detections_rules():
    truth = np.array([[5.0, 5.0, 5.0], [5.0, 20.0, 20.0]])
    detected = np.array([[5.0, 5.0, 5.0], [5.0, 5.0, 7.0], [5.0, 5.0, 10.0], [9.0, 30.0, 30.0], [5.0, 20.0, 20.5]])
    groups = np.array([0, 0, 1, 0, 1])
    match = evaluate(detected, truth, **W218_MATCH)
    # Detection 2 lies exactly 5 voxels from truth 0: a duplicate only with an inclusive boundary.
    assert classify_detections(match, detected, truth, radius=5.0, groups=groups).counts == {
        "matched": 2, "duplicate": 1, "spurious": 2, "duplicate_same_group": 1, "duplicate_other_group": 0}
    assert classify_detections(match, detected, truth, radius=5.0, groups=groups, boundary="inclusive").counts == {
        "matched": 2, "duplicate": 2, "spurious": 1, "duplicate_same_group": 1, "duplicate_other_group": 1}
    with pytest.raises(ValueError, match="groups"):
        classify_detections(match, detected, truth, radius=5.0, groups=groups[:3])
    with pytest.raises(ValueError, match="populations"):
        classify_detections(match, detected[:4], truth, radius=5.0)
    with pytest.raises(ValueError, match="radius"):
        classify_detections(match, detected, truth, radius=-1.0)
