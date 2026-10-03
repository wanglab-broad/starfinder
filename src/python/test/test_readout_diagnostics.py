"""§2.8 diagnostics: read inspection, population summaries and decision inspection (W-295).

docs/readout-contract.md, "Diagnostics". On the golden fixture of test_readout_golden.py:
summarize_reads counts equal the pinned statuses, explain_read lists spot 6's rescue to
GeneA with its corrected-round margin against the limit, and inspect_read shows spot 4's
tied round 2. FOV.run never calls inspect_read, plot_read or explain_read, and records the
population summary in run.json.
"""
import json

import numpy as np
import pandas as pd
import pytest

import starfinder.barcode as barcode
from starfinder.barcode import (DeduplicationConfig, ReadFilterConfig, ReadScoreConfig, decode_barcodes,
                                explain_read, filter_reads, inspect_read, plot_read, score_reads, summarize_reads)
from starfinder.barcode import diagnostics as diagnostics_module
from starfinder.io._checkpoint import _jsonable

from . import readout_fixtures as fx
from .test_readout_deduplication import crosstalk_pipeline
from .test_readout_direct import direct_dataset, direct_fov
from .test_readout_direct import pipeline as direct_pipeline
from .test_readout_golden import (CHANNELS, PINNED_STATUS, ROUNDS, SPOTS, TIED_ROUND, TIED_SPOT, extract,
                                  golden_codebook, readout_config)
from .test_readout_scoring import golden_run

pytestmark = [pytest.mark.barcode]


def golden_reads(tmp_path, decoder, **options):
    book = golden_codebook(tmp_path)
    intensities = extract()
    decoded = decode_barcodes(intensities, book, config=readout_config("decoding", decoder=decoder, **options))
    return book, intensities, decoded


@pytest.mark.parametrize("decoder", ["wta", "codebook_aware"])
def test_summarize_reads_counts_equal_the_golden_statuses(tmp_path, decoder):
    book, intensities, decoded = golden_reads(tmp_path, decoder)
    expected = {"wta": {"assigned": 6, "ambiguous": 1, "no_signal": 1, "unmatched": 2},
                "codebook_aware": {"assigned": 8, "ambiguous": 0, "no_signal": 1, "unmatched": 1}}[decoder]
    pinned = pd.Series([status for status, _ in PINNED_STATUS[decoder].values()]).value_counts()
    assert expected == {s: int(pinned.get(s, 0)) for s in expected}
    summary = summarize_reads(decoded, intensity_result=intensities)
    assert summary["call_status"] == expected and summary["reads"] == len(SPOTS)
    reasons = pd.Series([r for _, r in PINNED_STATUS[decoder].values() if r]).value_counts()
    assert summary["failure_reason"] == {k: int(v) for k, v in sorted(reasons.items())}
    table = decoded.table
    assigned = table[table.call_status.eq("assigned")]
    assert summary["genes"] == {k: int(v) for k, v in sorted(assigned.gene_id.value_counts().items())}
    assert summary["entries"] == {k: int(v) for k, v in sorted(assigned.entry_id.value_counts().items())}
    assert sum(summary["genes"].values()) == sum(summary["entries"].values()) == expected["assigned"]
    assert summary["call_type"] == {k: int(v) for k, v in sorted(table.call_type.value_counts().items())}
    assert (summary["qc_score"], summary["scoring"], summary["deduplication"], summary["filtering"]) == (
        None, None, None, None)
    # Per round and channel medians of the valid sums, background and noise; valid counts.
    medians = summary["intensities"]["medians"]
    assert medians["round2"]["ch01"]["sum"] == float(np.median(intensities.values[:, 1, 1]))
    assert medians["round1"]["ch00"]["background"] == float(np.median(intensities.background[:, 0, 0]))
    assert summary["intensities"]["valid"] == {label: len(SPOTS) for label in ROUNDS}
    assert summary["intensities"]["background_unavailable"] == {label: 0 for label in ROUNDS}
    # The scored and filtered reads add the score quantiles and the filtering counts.
    scored = score_reads(decoded, intensities, reference=book)
    filtered = filter_reads(scored)
    summary = summarize_reads({"decoding": decoded, "scoring": scored, "filtering": filtered,
                               "extraction": intensities})
    assert summary["call_status"] == expected
    assert summary["scoring"] == {"scored": expected["assigned"], "no_assignment": len(SPOTS) - expected["assigned"],
                                  "background_unavailable": 0}
    exact = scored.table.qc_score[scored.table.call_type.eq("exact")]
    assert summary["qc_score"]["exact"]["n"] == len(exact)
    assert summary["qc_score"]["exact"]["q50"] == pytest.approx(float(np.median(exact)), abs=0, rel=1e-15)
    assert summary["filtering"] == {"total": len(SPOTS), "accepted": expected["assigned"],
                                    "rejected": len(SPOTS) - expected["assigned"],
                                    "rejection_reasons": {"call_status": len(SPOTS) - expected["assigned"]}}
    json.dumps(_jsonable(summary), allow_nan=False)


@pytest.mark.parametrize("diagnostics", [True, False])
def test_explain_read_lists_the_rescue_of_spot_6(tmp_path, diagnostics):
    book, intensities, decoded = golden_reads(tmp_path, "codebook_aware", diagnostics=diagnostics)
    explained = explain_read(decoded, "6", intensity_result=intensities, reference=book)
    assert explained.step.tolist() == list(range(1, len(explained) + 1))
    rows = explained.set_index("item")
    assert rows.loc["decoder", "detail"] == "codebook_aware"
    assert rows.loc["observed_color_sequence", "detail"] == "1244"
    assert not rows.loc["exact_match", "passed"]
    candidate = explained[explained.item == "candidate"]
    assert len(candidate) == 1 and candidate.detail.iloc[0].startswith("rank 0: 1234 gene GeneA")
    margin = rows.loc["max_corrected_round_margin"]
    assert round(margin.value, 3) == 0.068 and margin.limit == 0.20 and margin.relation == "<=" and margin.passed
    assert margin.value == decoded.table.corrected_round_margin[6]
    for gate in ("max_hamming", "min_score_delta", "max_correction_penalty", "min_geomean_probability"):
        assert rows.loc[gate, "passed"], gate
    call = rows.loc["call"]
    assert call.passed and "rescued_h1 -> GeneA" in call.detail and "corrected rounds 2" in call.detail
    # Decision order: the decoder first, the call last of the decoding stage.
    decoding = explained[explained.stage == "decoding"].item.tolist()
    assert decoding[0] == "decoder" and decoding[-1] == "call"
    # An exact call lists no gates.
    exact = explain_read(decoded, "0", intensity_result=intensities, reference=book)
    assert "max_corrected_round_margin" not in exact.item.tolist()
    assert exact.set_index("item").loc["exact_match", "passed"]


def test_explain_read_follows_the_read_through_every_stage(tmp_path):
    _, fov, _ = golden_run(tmp_path, decoder="codebook_aware")
    explained = explain_read(fov.results, "6", reference=fov.codebook)
    assert list(dict.fromkeys(explained.stage)) == ["decoding", "scoring", "filtering"]
    rows = explained.set_index("item")
    assert rows.loc["max_corrected_round_margin", "limit"] == 0.20
    assert rows.loc["qc_score", "value"] == fov.scoring_result.table.qc_score[6]
    assert rows.loc["accepted", "passed"] and rows.loc["call_status", "passed"]
    tied = explain_read(fov.results, TIED_SPOT, reference=fov.codebook).set_index("item")
    assert not tied.loc["tied_or_unknown_rounds", "passed"] and "rounds 2 tie" in tied.loc["tied_or_unknown_rounds",
                                                                                               "detail"]
    zero = explain_read(fov.results, "5", reference=fov.codebook).set_index("item")
    assert zero.loc["zero_signal_rounds", "value"] == 1 and not zero.loc["zero_signal_rounds", "passed"]
    assert not zero.loc["accepted", "passed"] and zero.loc["accepted", "detail"] == "call_status"
    # A filtering result alone has no decoder config: the gate limits are missing.
    alone = explain_read(fov.filtering_result, "6").set_index("item")
    assert np.isnan(alone.loc["max_corrected_round_margin", "limit"])
    # Deduplication and end-base decisions.
    crosstalk = fx.crosstalk_fov(tmp_path / "crosstalk").run(crosstalk_pipeline())
    copy = explain_read(crosstalk.results, "1", reference=crosstalk.codebook).set_index("item")
    assert not copy.loc["representative", "passed"] and copy.loc["representative", "detail"] == "duplicate of 0"
    assert not copy.loc["duplicate", "passed"] and copy.loc["accepted", "detail"] == "duplicate"
    two_seg = fx.two_seg_fov(tmp_path / "two_seg").run(crosstalk_pipeline(
        deduplication=None, decoder=barcode.WtaDecoderConfig()))
    wrong = explain_read(two_seg.results, str(len(two_seg.spot_result.spots) - 1), reference=two_seg.codebook)
    ends = wrong[wrong.stage == "end_bases"].set_index("item")
    assert ends.loc["segment A", "passed"] and not ends.loc["segment B", "passed"]


def test_inspect_read_shows_the_tied_round_of_spot_4(tmp_path):
    book, intensities, decoded = golden_reads(tmp_path, "wta")
    table = inspect_read(intensities, decoded, TIED_SPOT, reference=book)
    assert len(table) == len(ROUNDS) * len(CHANNELS)
    assert table[["round", "channel"]].apply(tuple, axis=1).tolist() == [(r, c) for r in ROUNDS for c in CHANNELS]
    tied = table[table["round"] == ROUNDS[TIED_ROUND]]
    assert tied.observed_color.unique().tolist() == ["M"]
    top = tied[tied["sum"] == tied["sum"].max()]
    assert top.channel.tolist() == ["ch01", "ch02"] and top.probability.nunique() == 1
    assert not tied.observed.any()
    other = table[table["round"] != ROUNDS[TIED_ROUND]]
    assert other.groupby("round")["observed"].sum().tolist() == [1, 1, 1]
    i = intensities.spot_ids.index(TIED_SPOT)
    np.testing.assert_array_equal(table["sum"].to_numpy().reshape(4, 4), intensities.values[i].T)
    np.testing.assert_array_equal(table.background_sum.to_numpy().reshape(4, 4),
                                  (intensities.background[i] * intensities.box_voxels[i][None, :]).T)
    np.testing.assert_allclose(table.subtracted, table["sum"] - table.background_sum, rtol=0, atol=0)
    np.testing.assert_allclose(table.groupby("round").probability.sum(), 1.0, rtol=0, atol=1e-12)
    # The codebook-aware call of spot 4 assigns GeneE (2222); the colors and bases of its entry.
    rescued = decode_barcodes(intensities, book, config=readout_config("decoding", decoder="codebook_aware"))
    table = inspect_read(intensities, rescued, TIED_SPOT, reference=book)
    assert table.assigned_color[table.assigned].tolist() == list("2222")
    assert table.assigned.sum() == len(ROUNDS) and table.segment.unique().tolist() == ["A"]
    assert table.assigned_bases.unique().tolist() == ["CACAC"] and table.observed_bases.isna().all()
    clean = inspect_read(intensities, rescued, "0", reference=book)
    assert clean.observed_bases.unique().tolist() == ["C:" + "CGACC"[::-1]]


def test_plot_read_draws_the_inspection(tmp_path):
    from matplotlib.figure import Figure
    book, intensities, decoded = golden_reads(tmp_path, "wta")
    figure = plot_read(intensities, decoded, "6", reference=book)
    assert isinstance(figure, Figure)
    ax = figure.axes[0]
    assert len(ax.patches) == len(ROUNDS) * len(CHANNELS)
    assert [t.get_text() for t in ax.get_xticklabels()] == list(ROUNDS)
    figure.savefig(tmp_path / "read.png")
    assert (tmp_path / "read.png").stat().st_size > 0


def test_fov_run_records_the_summary_and_never_calls_the_on_demand_diagnostics(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("FOV.run must not call this diagnostic")

    for name in ("inspect_read", "plot_read", "explain_read"):
        monkeypatch.setattr(barcode, name, forbidden)
        monkeypatch.setattr(diagnostics_module, name, forbidden)
    fov = fx.crosstalk_fov(tmp_path)
    from starfinder.dataset import CheckpointConfig
    fov.run(crosstalk_pipeline(), checkpoints=CheckpointConfig(stages=("candidates", "pre_qc"),
                                                               directory=tmp_path / "checkpoints"))
    record = json.loads((tmp_path / "checkpoints" / "FOV_001" / "run.json").read_text())
    summary = record["counts"]["summary"]
    assert summary == json.loads(json.dumps(_jsonable(summarize_reads(fov.results))))
    assert summary["deduplication"] == fov.deduplication_result.counts == record["counts"]["deduplication"]
    assert summary["filtering"]["rejection_reasons"] == {"duplicate": 12}
    assert summary["cross_channel_pairs"] == {"distance_voxels": 1.0, "pairs": 19}
    # save_diagnostics writes the summary beside the existing counts.
    path = fov.save_diagnostics()
    written = json.loads(path.read_text())
    assert written["counts"] == fov.filtering_result.counts and written["summary"] == summary


def test_direct_mode_summary_counts_cross_channel_pairs(tmp_path):
    fov = direct_fov(direct_dataset(tmp_path)).run(direct_pipeline(scoring=ReadScoreConfig()))
    summary = summarize_reads(fov.results)
    assert summary["readout_mode"] == "direct" and summary["deduplication"] is None
    # direct2's two candidates at (3, 8, 8) are in different rounds; the round1 pair is 11.3 voxels
    # apart and the round2 pair 8 voxels.
    assert summary["cross_channel_pairs"] == {"distance_voxels": DeduplicationConfig().distance_voxels, "pairs": 0}
    assert summarize_reads(fov.results, distance_voxels=8.0)["cross_channel_pairs"]["pairs"] == 1
    assert summarize_reads(fov.results, distance_voxels=20.0)["cross_channel_pairs"]["pairs"] == 2
    assert summary["intensities"]["invalid"] == {"round1": 2, "round2": 2}
    explained = explain_read(fov.results, "0", reference=fov.dataset.direct_panel).set_index("item")
    assert explained.loc["panel", "detail"] == "round1/ch00 -> Gfap" and explained.loc["own_round_valid", "passed"]
    table = inspect_read(fov.intensity_result, fov.results, "0", reference=fov.dataset.direct_panel)
    assert table.assigned.sum() == 1 and table[table.assigned][["round", "channel"]].values.tolist() == [
        ["round1", "ch00"]]
    with pytest.raises(TypeError, match="DirectPanel"):
        inspect_read(fov.intensity_result, fov.results, "0", reference=fov.dataset.codebook)


def test_inputs_are_checked(tmp_path):
    book, intensities, decoded = golden_reads(tmp_path, "wta")
    with pytest.raises(ValueError, match="not a read"):
        explain_read(decoded, "99")
    with pytest.raises(TypeError, match="reads must be"):
        summarize_reads(decoded.table)
    with pytest.raises(ValueError, match="no read result"):
        summarize_reads({"extraction": intensities})
    with pytest.raises(TypeError, match="Codebook"):
        inspect_read(intensities, decoded, "0", reference=None)
    with pytest.raises(TypeError, match="IntensityExtractionResult"):
        inspect_read(decoded, decoded, "0", reference=book)
    assert ReadFilterConfig().exclude_duplicates
