"""W-770 metric known answers and exact same-environment extraction parity."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from starfinder.evaluation.preprocessing import (detected_spot_intensity_spread, direct_metrics, fov_variation)
from starfinder.evaluation.spot_finding import auprc, f1_score, scored_detection_auprc
from starfinder.image import ImageMetadata
from . import evaluation_preprocessing_oracle as oracle

pytestmark = [pytest.mark.evaluation, pytest.mark.preprocessing, pytest.mark.benchmark, pytest.mark.spot_finding]
ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("preprocessing_w770", ROOT / "benchmarks/preprocessing_synthetic.py")
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)


def matching():
    metadata = ImageMetadata("w770")
    return dict(policy="greedy", threshold=0.5, boundary="inclusive", units="voxel",
                reference_metadata=metadata, observed_metadata=metadata)


def bind_oracle(monkeypatch):
    for name in ("_pipeline", "_cutoffs", "_key"):
        monkeypatch.setattr(oracle, name, getattr(script, name), raising=False)


@pytest.mark.slow
@pytest.mark.parametrize("calibrated,condition,dtype,seed", [
    (False, "clean", "uint8", 0), (False, "baseline", "uint8", 100),
    (False, "baseline", "uint16", 1), (True, "combined", "uint8", 100),
])
def test_moved_metrics_and_sweep_equal_pre_move_scenes(tmp_path, monkeypatch, calibrated, condition, dtype, seed):
    # The scenes and dimensions used by the existing preprocessing tests.
    configure = script.calibrated_scene_config if calibrated else script.scene_config
    shape = (8, 32, 32) if calibrated else (10, 32, 32)
    book, config = configure(condition, dtype=dtype, seed=seed, shape=shape, count=12)
    scene, background, signal, truth = script.generate(book, config)
    radius = script.calibrated_background_radius(config)[0] if calibrated else script.background_radius(config)
    bind_oracle(monkeypatch)
    for arm, recipe, fov in script.processed_arms(scene, book, config.FOV_id, tmp_path, radius, False):
        args = (fov, scene.rounds, background, signal, truth, book, arm, recipe)
        assert direct_metrics(*args) == oracle.direct_metrics(*args)
        if arm in ("none", "r1", "r2_scalar_xsrc"):
            for mode, grid, verify_value in (("noise", script.THRESHOLDS, 5.0),
                                              ("adaptive", script.THRESHOLD_MODES["adaptive"]["grid"], 0.2)):
                before = oracle.sweep(fov, truth, mode=mode, grid=grid, verify_value=verify_value)
                after = script.sweep(fov, truth, mode=mode, grid=grid, verify_value=verify_value)
                pd.testing.assert_frame_equal(pd.DataFrame(after), pd.DataFrame(before), check_exact=True)
                assert auprc(after) == oracle._auprc(before)


@pytest.mark.parametrize("empty", [False, True])
def test_sweep_read_classification_parity_with_wrong_false_rejected_and_color_calls(monkeypatch, empty):
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
    monkeypatch.setattr(script, "_pipeline", lambda *args: (spots, reads, [2.0]))
    monkeypatch.setattr(script, "_cutoffs", lambda image, value, mode: [value])
    bind_oracle(monkeypatch)
    before = oracle.sweep(fov, truth, verify=False)
    after = script.sweep(fov, truth, verify=False)
    pd.testing.assert_frame_equal(pd.DataFrame(after), pd.DataFrame(before), check_exact=True)
    if not empty:
        assert after[0]["reads_correct"] == 1
        assert after[0]["reads_wrong_gene"] == 1
        assert after[0]["reads_false_detection"] == 2
        assert after[0]["color_call_agreement"] == pytest.approx(2 / 3, rel=1e-9, abs=1e-12)


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
