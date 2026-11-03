"""M17/M18 count decoded statuses before filtering."""
import pandas as pd
import pytest

pytestmark = [pytest.mark.evaluation, pytest.mark.readout, pytest.mark.contract, pytest.mark.validation]


def test_status_counts_and_unmatched_denominator():
    from starfinder.evaluation.readout import read_status_summary

    reads = pd.DataFrame(dict(call_status=["assigned", "unmatched", "assigned", "unmatched"],
                              accepted=[True, False, False, True]))
    result = read_status_summary(reads)
    assert result.values["unmatched_rate"] == pytest.approx(.5, rel=1e-9, abs=1e-12)
    assert result.counts == dict(total=4, assigned=2, unmatched=2, ambiguous=0, no_signal=0)
    assert result.values["assigned"] == 2
    empty = read_status_summary(reads.iloc[:0])
    assert empty.values == dict(unmatched_rate=None, assigned=0)
    assert empty.reasons["unmatched_rate"]
    assert empty.counts["total"] == 0


def test_status_summary_rejects_unknown_or_missing_status():
    from starfinder.evaluation.readout import read_status_summary

    for status in [None, "accepted"]:
        with pytest.raises(ValueError, match="call_status"):
            read_status_summary(pd.DataFrame(dict(call_status=[status])))
    result = read_status_summary(pd.DataFrame(dict(call_status=["assigned", "ambiguous", "no_signal"])))
    assert result.counts["assigned"] == 1
    assert result.values["unmatched_rate"] == pytest.approx(0., rel=1e-9, abs=1e-12)
