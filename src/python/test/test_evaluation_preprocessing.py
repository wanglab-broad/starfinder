"""W-770 metric known answers and exact same-environment extraction parity.

The scenes and recipes are the explicit package ones of ``preprocessing_scenes`` (W-781 retired the
benchmarks/preprocessing_synthetic.py script that held them); the pre-move algorithms stay in
``evaluation_preprocessing_oracle``.
"""
from dataclasses import replace
import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from starfinder.evaluation.preprocessing import (detected_spot_intensity_spread, direct_metrics, fov_variation)
from starfinder.evaluation.readout import correct_fraction
from starfinder.evaluation.spot_finding import auprc, f1_score, scored_detection_auprc
from starfinder.image import ImageMetadata
from starfinder.synthetic import calibrated_preprocessing_scene_config, preprocessing_scene_config
from . import evaluation_preprocessing_oracle as oracle
from . import preprocessing_scenes as scenes

pytestmark = [pytest.mark.evaluation, pytest.mark.preprocessing, pytest.mark.benchmark, pytest.mark.spot_finding]


def matching():
    metadata = ImageMetadata("w770")
    return dict(policy="greedy", threshold=0.5, boundary="inclusive", units="voxel",
                reference_metadata=metadata, observed_metadata=metadata)


def _scene(calibrated, condition, dtype, seed):
    """A bounded scene as the existing preprocessing tests used it, its truth and its 3D background radius."""
    if calibrated:
        book, config = calibrated_preprocessing_scene_config(condition, dtype=dtype, seed=seed, shape=(8, 32, 32),
                                                             count=12)
        radius = scenes.calibrated_background_radius(config)[0]
    else:
        book, config = preprocessing_scene_config(condition, dtype=dtype, seed=seed, shape=(10, 32, 32), count=12)
        radius = scenes.background_radius(config)
    return book, config, scenes.generate(book, config), radius


@pytest.mark.slow
@pytest.mark.parametrize("calibrated,condition,dtype,seed", [
    (False, "clean", "uint8", 0), (False, "baseline", "uint8", 100),
    (False, "baseline", "uint16", 1), (True, "combined", "uint8", 100),
])
def test_moved_metrics_equal_pre_move_scenes(tmp_path, calibrated, condition, dtype, seed):
    """Every arm's package direct metrics equal the pre-move implementation exactly, on the same FOV."""
    book, config, (scene, background, signal, truth), radius = _scene(calibrated, condition, dtype, seed)
    arms = set()
    for arm, recipe, fov in scenes.processed_arms(scene, book, config.FOV_id, tmp_path, radius):
        args = (fov, scene.rounds, background, signal, truth, book, arm, recipe)
        assert direct_metrics(*args) == oracle.direct_metrics(*args), arm
        arms.add(arm)
    assert arms == set(scenes.arms(radius))


@pytest.mark.parametrize("empty", [False, True])
def test_read_classification_parity_with_wrong_false_rejected_and_color_calls(monkeypatch, empty):
    """Package M10 (correct_fraction) and the color-call agreement computed from its own matched pairs equal the
    pre-move read loop on the same detections and reads, with wrong-gene, false, rejected and color calls."""
    metadata = ImageMetadata("read-parity")
    fov = SimpleNamespace(rounds=SimpleNamespace(reference_round="r"), metadata={"r": metadata},
                          images={"r": np.zeros((1, 1, 1, 1))})
    truth = pd.DataFrame(dict(z=[0., 0., 0.], y=[0., 0., 0.], x=[0., 10., 20.],
                              gene_id=["a", "b", "c"], color_sequence=["1", "2", "3"]))
    if empty:
        truth = truth.iloc[:0]
    spots = pd.DataFrame(dict(z=[0.] * 5, y=[0.] * 5, x=[0., 10., 20., 30., 40.],
                              channel=[0] * 5, peak_intensity=[20., 12., 6., 9., 3.]))
    reads = pd.DataFrame(dict(gene_id=["a", "wrong", "c", "a", None], accepted=[True, True, False, True, True],
                              observed_color_sequence=["1", "2", "wrong", "1", "1"]))
    # The oracle's read loop at one threshold that keeps every detection; no detection is rerun.
    monkeypatch.setattr(oracle, "_pipeline", lambda *args: (spots, reads, [2.0]), raising=False)
    monkeypatch.setattr(oracle, "_cutoffs", lambda image, value, mode: [value], raising=False)
    (row,) = oracle.sweep(fov, truth, verify=False, grid=(2.0,))
    detected = pd.concat([spots[["z", "y", "x"]], reads[["gene_id", "accepted"]]], axis=1)
    result = correct_fraction(detected, truth, reference_metadata=metadata, observed_metadata=metadata,
                              **oracle.MATCHING)
    for key in ("reads_accepted", "reads_correct", "reads_wrong_gene", "reads_false_detection"):
        assert result.counts[key] == row[key], key
    assert result.values["correct_fraction"] == row["correct_fraction"]
    pairs = result.details["matched_pairs"]
    agree = sum(str(reads.observed_color_sequence.iat[j]) == str(truth.color_sequence.iat[i]) for i, j, _ in pairs)
    agreement = agree / len(pairs) if pairs else None
    assert agreement == row["color_call_agreement"]
    if not empty:
        assert (result.counts["reads_correct"], result.counts["reads_wrong_gene"],
                result.counts["reads_false_detection"]) == (1, 1, 2)
        assert agreement == pytest.approx(2 / 3, rel=1e-9, abs=1e-12)
    else:
        assert result.values["correct_fraction"] is None and agreement is None


def test_shared_arms_equal_direct_recipe_runs(tmp_path):
    """An arm rebuilt from another arm's images equals its own recipe run directly by FOV.run."""
    book, config, (scene, *_), radius = _scene(False, "baseline", "uint16", 1)
    recipes = scenes.arms(radius)
    shared = {arm: fov for arm, _, fov in scenes.processed_arms(scene, book, config.FOV_id, tmp_path, radius)
              if arm in scenes.SHARED}
    assert set(shared) == set(scenes.SHARED)
    for arm, fov in shared.items():
        direct = scenes.preprocess(scenes.make_fov(scene, book, config.FOV_id, tmp_path), recipes[arm])
        for name in book.round_labels:
            np.testing.assert_array_equal(fov.images[name], direct.images[name])
        assert direct.preprocessing_record.get("recipe", {}).get("extraction_source") is None


def test_normalized_truth_follows_the_fitted_value_map():
    """Recipe 1's background truth is mapped through the observed raw -> normalized value map (interpolated between
    observed raw values) and reported in normalized units, through public direct_metrics."""
    raw = np.array([0, 10, 20, 10], dtype=np.uint8).reshape(1, 1, 4, 1)
    normalized = (raw * 2).astype(np.uint8)
    background = np.array([5.0, 15.0, 30.0, 0.0]).reshape(1, 1, 4, 1)
    book = SimpleNamespace(round_labels=("r",), color_to_channel={"1": 0})
    fov = SimpleNamespace(images={"r": normalized}, snapshots={})
    truth = pd.DataFrame(dict(z=[], y=[], x=[], color_sequence=[]))
    out = direct_metrics(fov, {"r": raw}, {"r": background}, {"r": np.zeros(raw.shape)}, truth, book, "r1", None)
    # Mapped truth [10, 30, 40, 0]; recipe 1's estimate is 0 in normalized units.
    assert out["bg_units"] == "recipe 1 normalized"
    assert out["bg_truth_mean"] == pytest.approx(20.0, rel=1e-9, abs=1e-12)
    assert out["bg_bias"] == pytest.approx(-20.0, rel=1e-9, abs=1e-12)
    assert out["bg_rmse"] == pytest.approx(math.sqrt((100 + 900 + 1600 + 0) / 4), rel=1e-9, abs=1e-12)


def test_background_ablation_rows_have_before_and_after_background_error(tmp_path):
    """pct has no background step: its estimate is 0, measured in input units before normalization, the stage of
    r2's bg_corrected snapshot, so both sides of each background ablation (pct -> r2_scalar, pct -> r2_3d) report a
    background error in the same units."""
    book, config, (scene, background, signal, truth), radius = _scene(False, "baseline", "uint8", 100)
    rows = {arm: direct_metrics(fov, scene.rounds, background, signal, truth, book, arm, recipe)
            for arm, recipe, fov in scenes.processed_arms(scene, book, config.FOV_id, tmp_path, radius)}
    assert rows["pct"]["bg_bias"] == -rows["pct"]["bg_truth_mean"]
    assert rows["pct"]["bg_stage"].startswith("before normalization, in input units")
    for before, after in (("pct", "r2_scalar"), ("pct", "r2_3d"), ("none", "scalar"), ("none", "bg3d")):
        for arm in (before, after):
            assert rows[arm]["bg_units"] == "input intensity", arm
            assert all(math.isfinite(rows[arm][key]) for key in ("bg_rmse", "bg_bias", "bg_truth_mean")), arm
    assert rows["r2_scalar"]["bg_stage"] == "background step output (bg_corrected), before normalization"
    assert rows["r1"]["bg_units"] == rows["r1_recon"]["bg_units"] == "recipe 1 normalized"
    # Arms without a background estimate keep the stage distinction by reporting none.
    assert "bg_rmse" not in rows["minmax"] and "bg_rmse" not in rows["hist"]


def test_background_radius_caps_r_z_at_three_on_eight_planes():
    _, config = calibrated_preprocessing_scene_config("regions", dtype="uint8", seed=0)
    assert config.shape_zyx[0] == 8
    assert scenes.calibrated_background_radius(config) == ((3, 5, 5), (6, 5, 5))
    with pytest.raises(ValueError, match="capped Z footprint 7 exceeds 6"):
        scenes.calibrated_background_radius(replace(config, shape_zyx=(6, 32, 32)))


def test_fov_spread_strata_and_fixed_conditions():
    """M16 per fixed condition and per FOV stratum: each fixed threshold value and each population (dense, edge)
    is its own group, never pooled with another; there is no development-selected point."""
    # Per-FOV correct fractions of two FOVs (dense, edge) at two fixed adaptive values, as W-239's fixture had them.
    values = {0.2: {"dense": 0.8, "edge": 0.7}, 0.4: {"dense": 0.9, "edge": 0.4}}
    spread = {value: fov_variation(list(per_fov.values())) for value, per_fov in values.items()}
    # Population std / mean: 0.05 / 0.75 and 0.25 / 0.65.
    assert spread[0.2] == pytest.approx(1 / 15, rel=1e-9, abs=1e-12)
    assert spread[0.4] == pytest.approx(5 / 13, rel=1e-9, abs=1e-12)
    pooled = fov_variation([v for per_fov in values.values() for v in per_fov.values()])
    assert pooled not in (pytest.approx(spread[0.2]), pytest.approx(spread[0.4]))
    # One FOV per stratum and fixed value: no spread to report within a stratum.
    assert {fov_variation([per_fov[s]]) for per_fov in values.values() for s in ("dense", "edge")} == {0.0}
    # A stratum missing its value at a fixed condition stays undefined, not dropped.
    assert fov_variation([values[0.2]["dense"], None]) is None


def test_f1_known_answer_and_empty_populations():
    truth = np.array([[0., 0., 0.], [0., 0., 10.]])
    detected = np.array([[0., 0., 0.], [0., 0., 20.], [0., 0., 30.]])
    result = f1_score(detected, truth, **matching())
    # One match: P=1/3, R=1/2, F1=2/5.
    assert result.values["f1"] == pytest.approx(0.4, rel=1e-9, abs=1e-12)
    assert result.counts["matched"] == 1
    assert f1_score(np.empty((0, 3)), truth, **matching()).values["f1"] == 0.0
    assert f1_score(detected, np.empty((0, 3)), **matching()).values["f1"] is None
    assert f1_score(detected, truth, eligible_reference=[False, False], **matching()).values["f1"] is None


def test_scored_auprc_known_answer_subsets_and_undefined():
    truth = np.array([[0., 0., 0.], [0., 0., 10.]])
    detected = np.array([[0., 0., 0.], [0., 0., 20.], [0., 0., 30.], [0., 0., 10.]])
    # Thresholds 15,10,5,2 give (R,P)=(0,None),(.5,1),(.5,.5),(1,.5).
    result = scored_detection_auprc(detected, truth, [12., 8., 4., 4.],
                                    thresholds=[2., 5., 10., 15.], **matching())
    assert result.values["auprc"] == pytest.approx(0.75, rel=1e-9, abs=1e-12)
    assert [r["recall"] for r in result.details["curve"]] == [1., 0.5, 0.5, 0.]
    masked = scored_detection_auprc(detected, truth, [12., 8., 4., 4.], thresholds=[2., 5., 10., 15.],
                                    eligible_observed=[True, False, False, True], **matching())
    assert masked.values["auprc"] == pytest.approx(1., rel=1e-9, abs=1e-12)
    empty = np.empty((0, 3))
    assert scored_detection_auprc(empty, truth, [], thresholds=[2.], **matching()).values["auprc"] == 0.0
    assert scored_detection_auprc(detected, empty, [12., 8., 4., 4.],
                                   thresholds=[2.], **matching()).values["auprc"] is None
    rows = [dict(threshold=2., recall=1., precision=.25), dict(threshold=5., recall=.5, precision=.5),
            dict(threshold=10., recall=.5, precision=1.), dict(threshold=15., recall=0., precision=float("nan"))]
    assert auprc(rows) == oracle._auprc(rows)
    assert auprc(rows) == pytest.approx(.625, rel=1e-9, abs=1e-12)
    # Recall 0 at every threshold, precision undefined (no detections): no area, not undefined.
    assert auprc([dict(threshold=v, recall=0., precision=float("nan")) for v in (2., 5.)]) == 0.0
    assert auprc([dict(threshold=2., recall=None, precision=None)]) is None
    with pytest.raises(ValueError, match="undefined precision"):
        auprc([dict(threshold=2., recall=1., precision=None)])


def test_detected_intensity_spread_known_group_medians_and_fov_variation():
    book = SimpleNamespace(round_labels=("r1", "r2"), color_to_channel={"1": 0, "2": 1})
    spots = pd.DataFrame(dict(z=[0.] * 3, y=[0.] * 3, x=[0., 3., 6.],
                              observed_color_sequence=["11", "11", "22"]))
    # Radius-zero sums, round/channel group medians: 2, 4, 6, 8.
    images = {name: np.zeros((1, 1, 7, 2), dtype=np.uint16) for name in book.round_labels}
    images["r1"][0, 0, [0, 3], 0] = [1, 3]
    images["r2"][0, 0, [0, 3], 0] = [3, 5]
    images["r1"][0, 0, 6, 1] = 6
    images["r2"][0, 0, 6, 1] = 8
    result = detected_spot_intensity_spread(images, spots, book, neighborhood_radius_zyx=(0, 0, 0))
    assert result == pytest.approx(np.sqrt(5) / 5, rel=1e-9, abs=1e-12)
    assert detected_spot_intensity_spread(images, spots.iloc[:0], book) is None
    assert fov_variation([1., 3.]) == pytest.approx(.5, rel=1e-9, abs=1e-12)
    assert fov_variation([2.]) == 0.0
    assert fov_variation([0., 0.]) is None
    assert fov_variation([]) is None
    assert fov_variation([1., None]) is None
    with pytest.raises(ValueError, match="nonnegative"):
        fov_variation([-1., 1.])
