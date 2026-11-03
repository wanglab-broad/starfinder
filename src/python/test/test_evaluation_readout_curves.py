"""M41 evaluates supplied cutoffs without choosing an optimum."""
import numpy as np
import pandas as pd
import pytest

pytestmark = [pytest.mark.evaluation, pytest.mark.readout, pytest.mark.contract, pytest.mark.validation]


def rows():
    return pd.DataFrame(dict(score=[4., 3., 2., 1.],
                             call_status=["assigned", "unmatched", "assigned", "assigned"],
                             gene_id=["a", None, "b", "a"], correct=[True, False, False, True]))


def test_cutoff_curve_known_rows():
    from starfinder.evaluation.readout import cutoff_curve

    result = cutoff_curve(rows(), score_column="score", cutoffs=[5, 3, 1],
                          orientation="higher", correct_column="correct")
    curve = result.details["curve"]
    assert [r["retained"] for r in curve] == [0, 2, 4]
    assert [r["retention"] for r in curve] == pytest.approx([0, .5, 1], rel=1e-9, abs=1e-12)
    assert curve[0]["unmatched_rate"] is None and curve[0]["truth_error"] is None
    assert curve[0]["reasons"]["unmatched_rate"] and curve[0]["reasons"]["truth_error"]
    assert [r["unmatched_rate"] for r in curve[1:]] == pytest.approx([.5, .25], rel=1e-9, abs=1e-12)
    assert [r["truth_error"] for r in curve[1:]] == pytest.approx([.5, .5], rel=1e-9, abs=1e-12)
    assert [r["gene_counts"] for r in curve] == [{}, {"a": 1}, {"a": 2, "b": 1}]
    assert curve[1]["gene_retained_fraction"] == pytest.approx(dict(a=.5, b=0), rel=1e-9, abs=1e-12)
    assert "optimum" not in result.values and "optimum" not in result.details
    lower = rows().assign(score=-rows().score)
    flipped = cutoff_curve(lower, score_column="score", cutoffs=[-5, -3, -1],
                           orientation="lower", correct_column="correct")
    for a, b in zip(curve, flipped.details["curve"]):
        assert {k: v for k, v in a.items() if k != "cutoff"} == {
            k: v for k, v in b.items() if k != "cutoff"}


def test_ties_missing_scores_and_optional_truth():
    from starfinder.evaluation.readout import cutoff_curve

    reads = rows().assign(score=[3., 3., np.nan, 1.])
    result = cutoff_curve(reads, score_column="score", cutoffs=[3], orientation="higher")
    assert result.counts == dict(total=4, scored=3, score_undefined=1)
    assert result.reasons["score_undefined"]
    row = result.details["curve"][0]
    assert row["retained"] == 2
    assert row["retention"] == pytest.approx(.5, rel=1e-9, abs=1e-12)
    assert "truth_error" not in row
    assert row["gene_retained_fraction"] == pytest.approx(dict(a=.5, b=0), rel=1e-9, abs=1e-12)
    empty = cutoff_curve(reads.iloc[:0], score_column="score", cutoffs=[3], orientation="higher")
    assert empty.details["curve"][0]["retention"] is None
    assert empty.details["curve"][0]["reasons"]["retention"]
    assert empty.status == "undefined"
    all_missing = cutoff_curve(reads.assign(score=np.nan), score_column="score", cutoffs=[3], orientation="higher")
    assert all_missing.counts["score_undefined"] == 4
    assert all_missing.details["curve"][0]["retention"] is None


def test_missing_truth_is_counted_without_inventing_errors():
    from starfinder.evaluation.readout import cutoff_curve

    reads = rows().assign(correct=pd.array([True, None, False, True], dtype="boolean"))
    result = cutoff_curve(reads, score_column="score", cutoffs=[3, 1],
                          orientation="higher", correct_column="correct")
    a, b = result.details["curve"]
    assert (a["truth_defined"], a["truth_undefined"]) == (1, 1)
    assert a["truth_error"] == pytest.approx(0., rel=1e-9, abs=1e-12)
    assert (b["truth_defined"], b["truth_undefined"]) == (3, 1)
    assert b["truth_error"] == pytest.approx(1 / 3, rel=1e-9, abs=1e-12)


@pytest.mark.parametrize("change", [dict(orientation="best"), dict(cutoffs=[np.nan]), dict(cutoffs=[]),
                                    dict(correct_column="gene_id")])
def test_invalid_curve_inputs(change):
    from starfinder.evaluation.readout import cutoff_curve

    config = dict(score_column="score", cutoffs=[3], orientation="higher")
    with pytest.raises(ValueError):
        cutoff_curve(rows(), **(config | change))
