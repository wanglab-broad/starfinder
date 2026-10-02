"""ranking_quality and evaluate_deduplication against analytic answers on hand-built inputs (W-292).

The two measures of the §2.8 validation design (docs/readout-algorithms.md, "Engineering
validation design"), in the form W-278 computed them for scores.csv and duplicates.csv.
"""
import math

import numpy as np
import pytest

from starfinder.evaluation.barcode import evaluate_deduplication, ranking_quality

pytestmark = [pytest.mark.evaluation, pytest.mark.contract]


def hanley_mcneil(a, n_correct, n_incorrect):
    """Hanley and McNeil (1982), equation (2), written out independently of the implementation."""
    q1 = a / (2 - a)
    q2 = 2 * a ** 2 / (1 + a)
    return math.sqrt((a * (1 - a) + (n_correct - 1) * (q1 - a ** 2) + (n_incorrect - 1) * (q2 - a ** 2))
                     / (n_correct * n_incorrect))


@pytest.mark.parametrize("correct_scores, incorrect_scores, expected", [
    ([1.0, 2.0, 3.0], [4.0, 5.0], 1.0),     # separated: every correct call ranks above
    ([4.0, 5.0], [1.0, 2.0, 3.0], 0.0),     # reversed
    ([2.0, 2.0, 2.0], [2.0, 2.0], 0.5),     # tied: ties count one half
], ids=["separated", "reversed", "tied"])
def test_auroc_of_separated_reversed_and_tied_scores(correct_scores, incorrect_scores, expected):
    score = np.array(correct_scores + incorrect_scores)
    correct = np.array([True] * len(correct_scores) + [False] * len(incorrect_scores))
    result = ranking_quality(score, correct, orientation="lower")
    assert result.values["auroc"] == expected
    assert result.status == "ok"
    assert (result.counts["correct"], result.counts["incorrect"]) == (len(correct_scores), len(incorrect_scores))
    # The same ranking with the opposite orientation of the negated scores.
    assert ranking_quality(-score, correct, orientation="higher").values["auroc"] == expected


def test_hanley_mcneil_standard_error():
    # Correct 1, 3, 5, 7 and incorrect 2, 6, 8 (lower is better): 8 of 12 pairs ordered, AUROC 2/3.
    score = np.array([1.0, 3.0, 5.0, 7.0, 2.0, 6.0, 8.0])
    correct = np.array([True] * 4 + [False] * 3)
    result = ranking_quality(score, correct, orientation="lower")
    assert abs(result.values["auroc"] - 2 / 3) <= 1e-12
    assert abs(result.values["auroc_se"] - hanley_mcneil(2 / 3, 4, 3)) <= 1e-12
    separated = ranking_quality(np.array([1.0, 2.0, 3.0]), np.array([True, True, False]), orientation="lower")
    assert abs(separated.values["auroc_se"] - hanley_mcneil(1.0, 2, 1)) <= 1e-12


def test_error_at_retention_on_ten_calls():
    # Ten calls scored 1..10 (lower is better); the calls scored 3, 6, 9 and 10 are incorrect.
    score = np.arange(1.0, 11.0)
    correct = ~np.isin(score, [3, 6, 9, 10])
    values = ranking_quality(score, correct, orientation="lower").values
    assert values["error_at_50"] == 1 / 5      # best 5: one incorrect (3)
    assert values["error_at_80"] == 2 / 8      # best 8: 3 and 6
    assert values["error_at_90"] == 3 / 9      # best 9: 3, 6 and 9
    assert values["error_at_100"] == 4 / 10    # all: the base error
    # A tie across the 50 % boundary: two calls scored 5, one incorrect; one of them is kept.
    tied = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 7.0, 8.0, 9.0, 10.0])
    correct = np.ones(10, dtype=bool)
    correct[5] = False
    values = ranking_quality(tied, correct, orientation="lower", retention=(0.5,)).values
    assert set(values) == {"auroc", "auroc_se", "error_at_50"} and values["error_at_50"] == 0.5 / 5


@pytest.mark.parametrize("correct, reason", [
    ([True, True, True], "no incorrect calls"),
    ([False, False, False], "no correct calls"),
], ids=["no_incorrect", "no_correct"])
def test_an_empty_class_gives_an_undefined_auroc_with_a_reason(correct, reason):
    result = ranking_quality(np.array([1.0, 2.0, 3.0]), np.array(correct), orientation="lower")
    assert result.values["auroc"] is None and result.values["auroc_se"] is None
    assert result.reasons == {"auroc": reason, "auroc_se": reason}
    assert result.status == "undefined"
    assert result.values["error_at_100"] == (0.0 if all(correct) else 1.0)


def test_no_scored_calls_and_nan_scores():
    result = ranking_quality(np.array([np.nan, np.nan]), np.array([True, False]), orientation="higher")
    assert all(v is None for v in result.values.values())
    assert set(result.reasons.values()) == {"no scored calls"}
    assert result.counts["score_undefined"] == 2
    result = ranking_quality(np.array([1.0, np.nan, 2.0]), np.array([True, True, False]), orientation="lower")
    assert (result.values["auroc"], result.counts["scored"], result.counts["score_undefined"]) == (1.0, 2, 1)


def test_ranking_quality_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="orientation"):
        ranking_quality([1.0], [True], orientation="up")
    with pytest.raises(ValueError, match="Boolean"):
        ranking_quality([1.0, 2.0], [1, 0], orientation="lower")
    with pytest.raises(ValueError, match="retention"):
        ranking_quality([1.0], [True], orientation="lower", retention=(0.0,))


def test_missed_duplicates_and_false_merges_on_five_pairs():
    # Reads 0-5; reads with one group label were merged, None was not grouped.
    groups = ["g0", "g0", "g2", None, "g4", "g4"]
    source = ["a", "a", "b", "b", "c", "d"]
    pairs = [(0, 1),   # one source, merged: found
             (2, 3),   # one source, not merged: a missed duplicate
             (4, 5),   # two sources, merged: a false merge
             (0, 2),   # two sources, not merged
             (1, 4)]   # two sources, not merged
    result = evaluate_deduplication(groups, source, pairs=pairs)
    assert result.values == {"missed_duplicates": 1, "false_merges": 1, "missed_duplicate_rate": 1 / 2,
                             "false_merge_rate": 1 / 3}
    assert result.counts == {"pairs": 5, "true_duplicate_pairs": 2, "distinct_pairs": 3, "unattributed_pairs": 0,
                             "merged_pairs": 2}
    assert result.details == {"missed_pairs": [(2, 3)], "false_merge_pairs": [(4, 5)]}
    assert result.status == "ok"


def test_an_empty_pair_class_gives_an_undefined_rate_with_a_reason():
    # No true duplicate pair (the W-278 held-out case) and one unattributed read.
    result = evaluate_deduplication(["g", "g", None], ["a", "b", None], pairs=[(0, 1), (1, 2)])
    assert result.values["missed_duplicate_rate"] is None
    assert result.reasons == {"missed_duplicate_rate": "no true duplicate pairs in the population"}
    assert (result.values["false_merges"], result.values["false_merge_rate"]) == (1, 1.0)
    assert result.counts["unattributed_pairs"] == 1
    result = evaluate_deduplication(["g", "g"], ["a", "a"], pairs=[(0, 1)])
    assert result.values["false_merge_rate"] is None and result.values["missed_duplicate_rate"] == 0.0
    with pytest.raises(ValueError, match="distinct read indices"):
        evaluate_deduplication(["g"], ["a"], pairs=[(0, 0)])
    with pytest.raises(ValueError, match="unique"):
        evaluate_deduplication(["g", "g"], ["a", "a"], pairs=[(0, 1), (1, 0)])
