"""Bounded known-value and failure contracts for pure evaluation."""
from dataclasses import asdict
import ast
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from starfinder.image import ImageMetadata, IncompatibleGeometryError
from starfinder.evaluation.matching import match_points
from starfinder.evaluation.spot_finding import evaluate_spots
from starfinder.evaluation.barcode import evaluate_decoding
from starfinder.evaluation.registration import (
    evaluate_translation, normalized_cross_correlation, structural_similarity,
    evaluate_mask_overlap, evaluate_registration,
)

META = ImageMetadata("test/reference")
CONFIG = dict(policy="greedy", threshold=2., units="voxel",
              reference_metadata=META, observed_metadata=META)


def test_matching_policies_and_eligibility():
    ref = np.array([[0, 0, 0], [0, 0, .2]])
    obs = np.array([[0, 0, .1], [0, 0, 1.]])
    greedy = match_points(ref, obs, **CONFIG)
    nearest = match_points(ref, obs, **{**CONFIG, "policy": "nearest_candidate"})
    assert greedy.counts["matched"] == 2
    assert nearest.counts["matched"] == 1
    excluded = match_points(ref, obs, **CONFIG, eligible_reference=[True, False])
    assert excluded.values["recall"] == 1
    assert excluded.values["precision"] == .5
    assert excluded.counts["total_reference"] == 2
    assert excluded.counts["eligible_reference"] == 1
    assert excluded.details["matched_pairs"] == [(0, 0, .1)]


def test_matching_boundary_and_subpixel():
    ref, obs = np.zeros((1, 3)), np.array([[0, 0, .5]])
    inclusive = evaluate_spots(obs, ref, **{**CONFIG, "threshold": .5})
    exclusive = evaluate_spots(obs, ref, **{**CONFIG, "threshold": .5}, boundary="exclusive")
    assert inclusive.values["mean_distance"] == .5
    assert exclusive.counts["matched"] == 0
    assert exclusive.values["recall"] == 0
    assert exclusive.values["mean_distance"] is None


@pytest.mark.parametrize("nr,no", [(0, 0), (0, 1), (1, 0)])
def test_empty_populations(nr, no):
    result = match_points(np.zeros((nr, 3)), np.zeros((no, 3)), **CONFIG)
    assert result.values["recall"] == (0 if nr else None)
    assert result.values["precision"] == (0 if no else None)
    assert result.values["mean_distance"] is None
    assert result.status == "undefined"
    json.dumps(asdict(result), allow_nan=False)


@pytest.mark.parametrize("change", [dict(observed_metadata=ImageMetadata("other")), dict(units="um")])
def test_incompatible_frames_units(change):
    with pytest.raises(IncompatibleGeometryError):
        match_points(np.zeros((1, 3)), np.zeros((1, 3)), **{**CONFIG, **change})


def test_physical_coordinates_require_calibration():
    m = ImageMetadata("physical", (2, 1, 1), (0, 0, 0), tuple(map(tuple, np.eye(3))), "um")
    r = match_points([[0, 0, 0]], [[.5, 0, 0]], policy="greedy", threshold=1, units="um",
                     reference_metadata=m, observed_metadata=m)
    assert r.values["mean_distance"] == .5


def test_translation_known_missing_failed_empty():
    config = dict(reference_metadata=META, observed_metadata=META, units="voxel", tolerance=.3)
    r = evaluate_translation({"r": (.25, 0, 0)}, {"r": (0, 0, 0)}, **config)
    assert r.values == {"max_error": .25, "mean_error_l2": .25, "passed": True}
    missing = evaluate_translation({}, {"r": (0, 0, 0)}, **config)
    assert missing.status == "missing" and missing.values["passed"] is None
    failed = evaluate_translation({}, {"r": (0, 0, 0)}, failed_rounds=["r"], **config)
    assert failed.status == "failed" and failed.counts["failed"] == 1
    empty = evaluate_translation({}, {}, **config)
    assert empty.status == "undefined" and empty.values["max_error"] is None
    partial = evaluate_translation({"r": (0, 0, 0)}, {"r": (0, 0, 0), "s": (0, 0, 0)}, **config)
    assert partial.values["max_error"] == 0 and partial.values["passed"] is None


def test_decoding_known_missing_and_zero_denominator():
    truth = pd.DataFrame({"gene": ["a", "b"], "color_seq": ["12", "21"]})
    decoded = pd.DataFrame({"gene": ["a", None], "color_seq": ["12", "12"]})
    coords = np.array([[0, 0, 0], [0, 0, 5.]])
    matches = match_points(coords, coords, **CONFIG)
    result = evaluate_decoding(decoded, truth, matches=matches)
    assert result.values == {"gene_accuracy": .5, "color_seq_accuracy": .5}
    missing = evaluate_decoding(decoded.drop(columns="gene"), truth, matches=matches)
    assert missing.values["gene_accuracy"] is None
    empty_matches = match_points(coords[:0], coords[:0], **CONFIG)
    empty = evaluate_decoding(decoded.iloc[:0], truth.iloc[:0], matches=empty_matches)
    assert all(v is None for v in empty.values.values())
    with pytest.raises(ValueError):
        evaluate_decoding(decoded.iloc[:1], truth, matches=matches)


def test_image_metrics_and_explicit_policies():
    image = np.arange(5 * 9 * 9, dtype=float).reshape(5, 9, 9)
    assert normalized_cross_correlation(image, image * 2).values["ncc"] == pytest.approx(1)
    assert normalized_cross_correlation(image * 0, image * 0).values["ncc"] is None
    for policy in ("volume", "mip", "slice"):
        opts = {"slice_index": 2} if policy == "slice" else {}
        r = structural_similarity(image, image, data_range=404, policy=policy, **opts)
        assert r.values["ssim"] == 1
        assert r.config["policy"] == policy
    assert structural_similarity(image[:1], image[:1], data_range=404, policy="volume").values["ssim"] is None
    with pytest.raises(ValueError):
        structural_similarity(image, image, data_range=0, policy="volume")
    with pytest.raises(ValueError):
        structural_similarity(image, image, data_range=404, policy="slice")
    with pytest.raises(ValueError):
        normalized_cross_correlation(image, image[:1])
    mask = np.array([True, False, True])
    r = evaluate_mask_overlap(mask, np.array([True, True, False]))
    assert r.values == {"iou": 1/3, "dice": .5}
    assert evaluate_mask_overlap(mask & False, mask & False).values["iou"] is None


def test_pure_report_no_detection_or_io(monkeypatch, tmp_path):
    import starfinder.spot_finding as detection
    import starfinder.registration as registration
    def forbidden(*args, **kwargs):
        raise AssertionError("algorithm rerun")
    monkeypatch.setattr(detection, "find_spots", forbidden)
    monkeypatch.setattr(registration, "estimate_transform", forbidden)
    image = np.arange(3 * 9 * 9, dtype=float).reshape(3, 9, 9)
    original = image.copy()
    points = np.zeros((1, 3))
    mask = image > 100
    r = evaluate_registration(image, image, image, reference_spots=points,
        before_spots=points, after_spots=points, reference_mask=mask, before_mask=mask,
        after_mask=mask, reference_metadata=META, before_metadata=META, after_metadata=META,
        data_range=242, ssim_policy="volume", matching_policy="greedy", match_threshold=1, units="voxel")
    assert r.values["ncc_after"] == pytest.approx(1)
    assert r.values["match_rate_after"] == 1
    np.testing.assert_array_equal(image, original)
    assert list(tmp_path.iterdir()) == []
    json.dumps(asdict(r), allow_nan=False)
    root = Path(__file__).parents[1] / "starfinder" / "evaluation"
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith(("starfinder.benchmark", "starfinder.spot_finding", "starfinder.registration"))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in ("open", "print", "find_spots", "estimate_transform")


def test_benchmark_adapter_uses_canonical_metrics():
    from starfinder.benchmark._legacy_evaluation import _evaluate_images
    image = np.arange(3 * 9 * 9, dtype=float).reshape(3, 9, 9)
    for mip in (False, True):
        r = _evaluate_images(image, image, image, use_mip=mip)
        assert r["ncc_after"] == pytest.approx(1)
        assert r["ssim_after"] == 1
        assert r["evaluation"]["config"]["structure_after"]["policy"] == ("mip" if mip else "volume")


def test_nearest_candidate_preserves_registration_policy():
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(144)
    ref, obs = rng.random((12, 3)), rng.random((8, 3))
    distances, indices = cKDTree(obs).query(ref, k=1)
    valid = distances <= .5
    used, expected = set(), []
    for i in np.flatnonzero(valid)[np.argsort(distances[valid])]:
        if indices[i] not in used:
            used.add(indices[i])
            expected.append((int(i), int(indices[i]), float(distances[i])))
    actual = match_points(ref, obs, **{**CONFIG, "policy": "nearest_candidate", "threshold": .5})
    assert actual.details["matched_pairs"] == expected


@pytest.mark.parametrize("change", [dict(threshold=-1), dict(policy="optimal"),
                                       dict(boundary="unknown"), dict(eligible_reference=[1])])
def test_invalid_matching_config(change):
    with pytest.raises(ValueError):
        match_points(np.zeros((1, 3)), np.zeros((1, 3)), **{**CONFIG, **change})


def test_error_only_translation_and_undefined_reporting(capsys):
    from starfinder.benchmark._reporting import _print_quality_report
    r = evaluate_translation({"r": (0.125, 0, 0)}, {"r": (0, 0, 0)},
        reference_metadata=META, observed_metadata=META, units="voxel", tolerance=None)
    assert r.values["mean_error_l2"] == .125
    assert r.values["passed"] is None
    assert r.reasons["passed"] == "tolerance gate not requested"
    _print_quality_report(r)
    assert "passed: undefined" in capsys.readouterr().out
