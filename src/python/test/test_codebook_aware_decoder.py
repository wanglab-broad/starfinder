"""Numerical regression and explicit correction policies for public decoding."""

from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from starfinder.barcode import (
    decode_barcodes,
    WtaDecoderConfig,
    CodebookAwareDecoderConfig,
    InvalidIntensityError,
    filter_reads,
    ReadFilterConfig,
)
from .barcode_cases import codebook, intensity, tensor


def decode(values, mapping, **options):
    return decode_barcodes(
        intensity(values), codebook(mapping), config=CodebookAwareDecoderConfig(**options)
    )


def test_known_wta_and_probability_score_semantics():
    values = np.array([[[3.0], [4.0], [0.0], [0.0]]])
    ext = intensity(values)
    book = codebook({"2": "gene"})
    wta = decode_barcodes(ext, book, config=WtaDecoderConfig(diagnostics=True))
    aware = decode_barcodes(ext, book, config=CodebookAwareDecoderConfig(diagnostics=True))
    assert wta.table.wta_l2_nll[0] == pytest.approx(-np.log(4 / (5 + 1e-6)))
    assert aware.table.probability_nll[0] == pytest.approx(-np.log((4 + 1e-6) / (7 + 4e-6)))
    assert wta.table.gene_id[0] == aware.table.gene_id[0] == "gene"
    assert "probability_nll" not in wta.table and "wta_l2_nll" not in aware.table
    np.testing.assert_allclose(aware.diagnostics["probabilities"].sum(axis=1), 1)


def test_exact_and_unmatched_preserve_ids():
    result = decode(
        tensor(["1234", "2222", "1111"]), {"1234": "A", "2222": "B"}, allow_rescue=False
    )
    assert result.table.call_status.tolist() == ["assigned", "assigned", "unmatched"]
    assert result.table.spot_id.tolist() == ["s0", "s1", "s2"]
    assert result.table.failure_reason[2] == "rescue_disabled"
    assert filter_reads(result).accepted.gene_id.tolist() == ["A", "B"]


def test_rescue_margin_boundary_and_probability_gate():
    values = tensor(["4422"])
    values[0, :, 1] = [1, 1, 51, 50]
    kwargs = dict(diagnostics=True, min_geomean_probability=0.3)
    result = decode(values, {"4422": "A"}, **kwargs)
    row = result.table.iloc[0]
    assert row.observed_color_sequence == "4322" and row.decoded_color_sequence == "4422"
    assert row.call_type == "rescued_h1" and row.corrected_rounds == "1"
    margin = row.corrected_round_margin
    assert (
        decode(
            values, {"4422": "A"}, max_corrected_round_margin=margin, **kwargs
        ).table.call_status[0]
        == "assigned"
    )
    assert (
        decode(
            values, {"4422": "A"}, max_corrected_round_margin=np.nextafter(margin, 0), **kwargs
        ).table.failure_reason[0]
        == "corrected_round_margin_too_high"
    )
    assert (
        decode(values, {"4422": "A"}, min_geomean_probability=1).table.failure_reason[0]
        == "geomean_prob_too_low"
    )
    assert (
        decode(values, {"4422": "A"}, max_correction_penalty=0).table.failure_reason[0]
        == "correction_penalty_too_high"
    )
    assert decode(values, {"4422": "A"}, max_hamming=0).table.failure_reason[0] == "no_candidate"
    candidate = result.diagnostics["candidates"].iloc[0]
    assert candidate.color_sequence == "4422" and candidate.spot_id == "s0"
    assert candidate.probability_nll == row.probability_nll


def test_high_margin_and_rescue_disabled():
    values = tensor(["4422"])
    values[0, :, 1] = [1, 1, 90, 10]
    assert (
        decode(values, {"4422": "A"}, min_geomean_probability=0.1).table.failure_reason[0]
        == "corrected_round_margin_too_high"
    )
    assert (
        decode(values, {"4422": "A"}, allow_rescue=False).table.failure_reason[0]
        == "rescue_disabled"
    )
    assert (
        decode(tensor(["4422"]), {"4422": "A"}, allow_exact=False).table.call_status[0]
        == "unmatched"
    )


def test_equal_candidate_scores_ambiguous_even_zero_delta_gate():
    values = tensor(["4422"])
    values[0, :, 1] = [1, 40, 41, 40]
    result = decode(
        values,
        {"4422": "A", "4222": "B"},
        min_score_delta=0,
        min_geomean_probability=0.1,
        diagnostics=True,
    )
    assert result.table.call_status[0] == "ambiguous" and pd.isna(result.table.gene_id[0])
    assert result.diagnostics["candidates"].color_sequence.tolist() == ["4222", "4422"]


def test_zero_signal_is_not_rescued_but_nonzero_tie_can_be():
    values = tensor(["4422"])
    values[0, :, 1] = 0
    result = decode(values, {"4422": "A"}, min_geomean_probability=0.1)
    assert result.table.call_status[0] == "no_signal" and pd.isna(result.table.gene_id[0])
    assert result.table.failure_reason[0] == "zero_signal_round"
    values[0, :, 1] = [1, 1, 50, 50]
    rescued = decode(values, {"4422": "A"}, min_geomean_probability=0.1)
    assert rescued.table.call_type[0] == "rescued_unknown"
    ambiguous = decode(values, {"4422": "A", "4322": "B"}, min_geomean_probability=0.1)
    assert ambiguous.table.call_status[0] == "ambiguous"
    wta = decode_barcodes(intensity(values), codebook({"4422": "A"}), config=WtaDecoderConfig())
    assert wta.table.call_status[0] == "ambiguous"


@pytest.mark.parametrize("config", [WtaDecoderConfig(), CodebookAwareDecoderConfig()])
def test_negatives_separate_from_zero_and_ties(config):
    values = np.array([[[3.0], [-2.0], [0.0], [-1.0]]])
    original = values.copy()
    with pytest.raises(InvalidIntensityError):
        decode_barcodes(intensity(values), codebook({"1": "A"}), config=config)
    result = decode_barcodes(
        intensity(values),
        codebook({"1": "A"}),
        config=replace(config, negative_policy="clip_negative"),
    )
    assert result.table.call_status[0] == "assigned"
    assert result.diagnostics["clipped_count"] == 2 and result.diagnostics["clipped_range"] == (
        -2.0,
        -1.0,
    )
    np.testing.assert_array_equal(values, original)


def test_labels_mapping_invalid_masks_and_empty_schema():
    ext = intensity(tensor(["1234"]))
    book = codebook({"1234": "A"})
    with pytest.raises(ValueError, match="labels"):
        decode_barcodes(
            ext, replace(book, round_labels=book.round_labels[::-1]), config=WtaDecoderConfig()
        )
    mapping = {"1": 1, "2": 0, "3": 2, "4": 3}
    remapped = replace(book, color_to_channel=mapping)
    vals = ext.values[:, [1, 0, 2, 3], :]
    mapped = decode_barcodes(
        replace(ext, values=vals), remapped, config=CodebookAwareDecoderConfig(diagnostics=True)
    )
    assert mapped.table.gene_id[0] == "A"
    assert mapped.diagnostics["probabilities"][0, :, 0].argmax() == 1
    invalid = decode_barcodes(
        replace(ext, valid=np.zeros((1, 4), bool)), book, config=WtaDecoderConfig()
    )
    assert invalid.table.failure_reason[0] == "invalid_measurement"
    for config in (
        WtaDecoderConfig(diagnostics=True),
        CodebookAwareDecoderConfig(diagnostics=True),
    ):
        empty = decode_barcodes(intensity(np.empty((0, 4, 4))), book, config=config)
        assert empty.table.empty and isinstance(empty.table.spot_id.dtype, pd.StringDtype)
        filtered = filter_reads(empty)
        assert filtered.counts == {"total": 0, "accepted": 0, "rejected": 0}
        assert filtered.fractions["accepted"] is None
        assert filtered.diagnostics["undefined_fraction_reasons"]["accepted"] == "empty_population"
        assert filtered.accepted.dtypes.equals(filtered.table.dtypes)


def test_filter_rerun_scores_endpoints_and_all_rejected():
    result = decode(tensor(["12", "11"]), {"12": "A", "11": "B"})
    original = result.table.copy(deep=True)
    diagnostic = filter_reads(result, config=ReadFilterConfig(end_bases="CC"))
    assert diagnostic.counts["accepted"] == 2
    excluding = filter_reads(
        result, config=ReadFilterConfig(end_bases="CC", exclude_invalid_endpoints=True)
    )
    assert excluding.accepted.gene_id.tolist() == ["B"]
    rejected = filter_reads(
        result, config=ReadFilterConfig(score_bounds={"probability_nll": (None, -1)})
    )
    assert len(rejected.table) == 2 and rejected.accepted.empty
    assert rejected.table.spot_id.tolist() == result.table.spot_id.tolist()
    pd.testing.assert_frame_equal(result.table, original)
    with pytest.raises(ValueError, match="unavailable"):
        filter_reads(result, config=ReadFilterConfig(score_bounds={"wta_l2_nll": (0, 1)}))


def test_wta_exact_tie_vs_probability_tolerance_is_explicit():
    values = np.array([[[1.0], [1.0 + 1e-13], [0.0], [0.0]]])
    book = codebook({"1": "A", "2": "B"})
    wta = decode_barcodes(intensity(values), book, config=WtaDecoderConfig())
    aware = decode_barcodes(
        intensity(values), book, config=CodebookAwareDecoderConfig(min_score_delta=0)
    )
    assert wta.table.call_status[0] == "assigned" and wta.table.gene_id[0] == "B"
    assert (
        aware.table.observed_color_sequence[0] == "M" and aware.table.call_status[0] == "ambiguous"
    )


def test_rescue_geomean_penalty_and_separation_boundaries():
    values = tensor(["4422"])
    values[0, :, 1] = [1, 2, 51, 50]
    baseline = decode(
        values, {"4422": "A", "4222": "B"}, diagnostics=True, min_geomean_probability=0
    )
    scores = baseline.diagnostics["candidates"]
    probability = float(scores.geomean_probability.iloc[0])
    delta = float(scores.probability_nll.iloc[1] - scores.probability_nll.iloc[0])
    assert (
        decode(
            values,
            {"4422": "A", "4222": "B"},
            min_geomean_probability=probability,
            min_score_delta=delta,
        ).table.call_status[0]
        == "assigned"
    )
    assert (
        decode(
            values, {"4422": "A", "4222": "B"}, min_score_delta=np.nextafter(delta, np.inf)
        ).table.failure_reason[0]
        == "ambiguous_candidate"
    )
    assert (
        decode(
            values, {"4422": "A"}, min_geomean_probability=np.nextafter(probability, np.inf)
        ).table.failure_reason[0]
        == "geomean_prob_too_low"
    )
    # The penalty compares changed-round -log probabilities; probe either side
    # of its mathematical value with tolerance for the two logarithm operations.
    penalty = np.log((51 + 1e-6) / (50 + 1e-6))
    assert (
        decode(values, {"4422": "A"}, max_correction_penalty=penalty + 1e-12).table.call_status[0]
        == "assigned"
    )
    assert (
        decode(values, {"4422": "A"}, max_correction_penalty=penalty - 1e-12).table.failure_reason[
            0
        ]
        == "correction_penalty_too_high"
    )


def test_result_validation_rejects_nonfinite_and_duplicate_ids():
    from .barcode_cases import intensity

    values = tensor(["1", "2"])
    with pytest.raises(ValueError, match="unique"):
        intensity(values, ids=("same", "same"))
    values[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        intensity(values)
