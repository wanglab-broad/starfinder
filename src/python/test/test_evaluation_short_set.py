"""Hand-computed §2.14 metrics and same-environment greedy oracle equivalence."""
from dataclasses import asdict
import json

import numpy as np
import pandas as pd
import pytest
from .evaluation_matching_oracle import dense_greedy_oracle as _oracle

from starfinder.evaluation.matching import evaluate_molecules, match_points
from starfinder.evaluation.readout import correct_fraction
from starfinder.evaluation.registration import shift_difference
from starfinder.image import ImageMetadata

pytestmark = [pytest.mark.evaluation, pytest.mark.contract]
META = ImageMetadata("short-set")
GEOMETRY = dict(reference_metadata=META, observed_metadata=META)


def _table(points, genes):
    table = pd.DataFrame(points, columns=["z", "y", "x"])
    table["gene_id"] = genes
    return table


def test_m10_accepted_reads_and_truth_denominator():
    truth = _table([[0, 0, i * 10] for i in range(4)], ["a", "b", "c", "d"])
    detected = _table([[0, 0, 0], [0, 0, 10], [0, 0, 20], [0, 0, 100]], ["a", "wrong", "c", "a"])
    detected["accepted"] = [True, True, False, True]
    result = correct_fraction(detected, truth, policy="greedy", threshold=1, boundary="inclusive",
                              units="voxel", **GEOMETRY)
    assert result.values["correct_fraction"] == pytest.approx(.25, rel=1e-9, abs=1e-12)
    assert {k: result.counts[k] for k in ("reads_correct", "reads_wrong_gene", "reads_false_detection",
                                        "reads_accepted", "matched")} == dict(
        reads_correct=1, reads_wrong_gene=1, reads_false_detection=1, reads_accepted=3, matched=3)
    empty = correct_fraction(detected.iloc[:0], truth.iloc[:0], policy="greedy", threshold=1,
                             boundary="inclusive", units="voxel", **GEOMETRY)
    assert empty.status == "undefined" and empty.values["correct_fraction"] is None


def test_m34_chebyshev_any_neighbour_both_directions():
    direct = _table([[0, 0, 0], [0, 0, .5], [10, 0, 0], [20, 0, 0]], ["a", "a", "b", "c"])
    tiled = _table([[1, 1, 1], [10, 0, 0], [10, 0, 0]], ["a", "wrong", "b"])
    result = evaluate_molecules(direct, tiled, policy="any_neighbor", threshold=1 + 1e-9,
                                units="voxel", **GEOMETRY)
    assert result.counts["matched_reference"] == 3  # both a's use the same neighbour
    assert result.counts["matched_observed"] == 2
    for key, expected in dict(reference_fraction=.75, observed_fraction=2/3,
                               count_difference=.25, count_ratio=.75).items():
        assert result.values[key] == pytest.approx(expected, rel=1e-9, abs=1e-12)
    edge = _table([[0, 0, 1 + .5e-9]], ["a"])
    assert evaluate_molecules(direct.iloc[:1], edge, policy="any_neighbor", threshold=1 + 1e-9,
                               units="voxel", **GEOMETRY).counts["matched_reference"] == 1


def test_m70_m71_legacy_boundary_and_one_to_one(tmp_path):
    from starfinder.assignment import import_legacy_molecules_csv
    path = tmp_path / "goodSpots.csv"
    pd.DataFrame(dict(x=[31, 41, 51], y=[11, 21, 26], z=[3, 5, 7], gene=["a", "b", "c"])).to_csv(path, index=False)
    matlab = import_legacy_molecules_csv(path, spot_namespace='["d","s","f",null]', genes=("a", "b", "c")).table
    np.testing.assert_allclose(matlab[["z", "y", "x"]], [[2, 10, 30], [4, 20, 40], [6, 25, 50]],
                               rtol=1e-9, atol=1e-12)
    python = _table([[2, 10, 30.25], [2, 10, 30.5], [4, 20, 40], [6, 25, 50]], ["a", "a", "b", "wrong"])
    result = evaluate_molecules(matlab, python, policy="greedy", threshold=.5, units="voxel", **GEOMETRY)
    assert result.counts["matched_reference"] == result.counts["matched_observed"] == 2
    for key, expected in dict(reference_fraction=2/3, observed_fraction=.5, count_ratio=4/3).items():
        assert result.values[key] == pytest.approx(expected, rel=1e-9, abs=1e-12)
    unconverted = _table([[3, 11, 31], [5, 21, 41], [7, 26, 51]], ["a", "b", "c"])
    assert evaluate_molecules(unconverted, python, policy="greedy", threshold=.5,
                              units="voxel", **GEOMETRY).counts["matched_reference"] == 0
    empty = evaluate_molecules(matlab.iloc[:0], python.iloc[:0], policy="greedy", threshold=1,
                                units="voxel", **GEOMETRY)
    assert empty.status == "undefined" and all(x is None for x in empty.values.values())
    json.dumps(asdict(result), allow_nan=False)


def test_m69_matlab_axis_sign_and_one_based_cancellation():
    # DFTApply3D.m:21 multiplies F(moving) by exp(-i*2*pi*shift*f),
    # translating content by the correction. Python contracts.md:126 samples
    # moving[p + displacement], so detected ZYX is -(Z,row,column).
    correction_yxz = np.array([2., -3., 1.])
    converted = -correction_yxz[[2, 0, 1]]
    xyz_start = np.array([10., 20., 30.])
    xyz_end = xyz_start - correction_yxz[[1, 0, 2]]
    np.testing.assert_allclose((xyz_end + 1) - (xyz_start + 1), xyz_end - xyz_start,
                               rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose((xyz_end - xyz_start)[::-1], converted, rtol=1e-9, atol=1e-12)
    reference = {"r1": converted, "r2": (0, 0, 0)}
    result = shift_difference(reference, {"r1": converted, "r2": (1, 2, 2)}, **GEOMETRY)
    assert result.values == pytest.approx(dict(max_axis_difference=2, mean_difference_l2=1.5), rel=1e-9, abs=1e-12)
    assert result.details["per_round"]["r2"]["difference_zyx"] == [1, 2, 2]
    for mistake in (correction_yxz[[2, 0, 1]], -correction_yxz):
        assert shift_difference({"r1": converted}, {"r1": mistake}, **GEOMETRY).values["max_axis_difference"] > 0
    missing = shift_difference(reference, {"r1": converted}, **GEOMETRY)
    assert missing.status == "missing" and missing.counts["missing"] == 1


@pytest.mark.parametrize("boundary", ["inclusive", "exclusive"])
@pytest.mark.parametrize("masked", [False, True])
def test_greedy_matches_dense_oracle_seeded_with_ties(boundary, masked):
    rng = np.random.default_rng(769)
    reference = rng.integers(0, 100, (2000, 3)).astype(float)
    observed = rng.integers(0, 100, (2500, 3)).astype(float)
    # Duplicates, equal-distance alternatives, and exact threshold distances.
    reference[:4] = [[0, 0, 0], [0, 0, 0], [0, 0, 4], [0, 0, 8]]
    observed[:4] = [[0, 0, 2], [0, 0, 2], [0, 0, 6], [0, 0, 10]]
    er = rng.random(2000) > .2 if masked else np.ones(2000, dtype=bool)
    eo = rng.random(2500) > .2 if masked else np.ones(2500, dtype=bool)
    er[:4] = eo[:4] = True
    expected = _oracle(reference, observed, 2., boundary, er, eo)
    result = match_points(reference, observed, policy="greedy", threshold=2., units="voxel", boundary=boundary,
                          eligible_reference=er, eligible_observed=eo, **GEOMETRY)
    # Same-environment identity is the explicit acceptance contract, including distances.
    assert result.details["matched_pairs"] == expected
    assert result.counts["matched"] == len(expected)
    assert result.values["recall"] == len(expected) / er.sum()
    assert result.values["precision"] == len(expected) / eo.sum()
    assert result.values["mean_distance"] == np.mean([p[2] for p in expected])


@pytest.mark.parametrize("threshold", [0., .5, 2.])
@pytest.mark.parametrize("boundary", ["inclusive", "exclusive"])
def test_greedy_existing_matching_inputs_and_float_boundary(threshold, boundary):
    cases = [(np.array([[0, 0, 0], [0, 0, .2]]), np.array([[0, 0, .1], [0, 0, 1.]])),
             (np.zeros((1, 3)), np.array([[0, 0, .5]])),
             (np.zeros((0, 3)), np.zeros((0, 3))),
             (np.zeros((0, 3)), np.zeros((1, 3))),
             (np.zeros((1, 3)), np.zeros((0, 3))),
             (np.array([[0, 0, 0], [0, 0, 5.]]), np.array([[0, 0, 0], [0, 0, 5.]])),
             (np.zeros((1, 3)), np.array([[.5, 0, 0]])),
             (np.zeros((1, 3)), np.array([[.3, .4, 0], [0, 0, np.nextafter(.5, 0)],
                                         [0, 0, np.nextafter(.5, np.inf)]]))]
    for reference, observed in cases:
        for masked in (False, True):
            er, eo = np.ones(len(reference), bool), np.ones(len(observed), bool)
            if masked and len(er) > 1:
                er[-1] = False
            expected = _oracle(reference, observed, threshold, boundary, er, eo)
            actual = match_points(reference, observed, policy="greedy", threshold=threshold, boundary=boundary,
                                  units="voxel", eligible_reference=er, eligible_observed=eo, **GEOMETRY)
            assert actual.details["matched_pairs"] == expected
