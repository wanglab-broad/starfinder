"""Tests for the codebook-aware barcode decoder."""

import numpy as np
import pandas as pd
import pytest

from starfinder.barcode import filter_reads
from starfinder.barcode.codebook_aware import (
    build_one_error_index,
    candidate_sequences,
    channel_probabilities,
    decode_codebook_aware,
    wta_color_sequences,
)


def _tensor_for_sequences(sequences, n_channels=4, high=100.0, low=1.0):
    n_spots = len(sequences)
    n_rounds = len(sequences[0]) if sequences else 0
    tensor = np.full((n_spots, n_channels, n_rounds), low, dtype=np.float64)
    for spot_idx, seq in enumerate(sequences):
        for round_idx, color in enumerate(seq):
            tensor[spot_idx, int(color) - 1, round_idx] = high
    return tensor


def test_channel_probabilities_sum_to_one():
    tensor = np.array(
        [
            [[1, 0], [2, 0], [3, 0], [4, 0]],
            [[0, 0], [0, 0], [0, 0], [0, 0]],
        ],
        dtype=np.float64,
    )

    probs = channel_probabilities(tensor)

    np.testing.assert_allclose(probs.sum(axis=1), 1.0)
    np.testing.assert_allclose(probs[1, :, :], 0.25)


def test_wta_exact_matches_filter_reads():
    seq_to_gene = {"1234": "GeneA", "2222": "GeneB"}
    tensor = _tensor_for_sequences(["1234", "2222", "1111"])

    decoded = decode_codebook_aware(tensor, seq_to_gene, allow_rescue=False)

    probs = channel_probabilities(tensor)
    color_seq, _ = wta_color_sequences(probs)
    spots = pd.DataFrame({"spot_id": [0, 1, 2], "color_seq": color_seq})
    good, _ = filter_reads(spots, seq_to_gene)
    expected_genes = spots["color_seq"].map(seq_to_gene)

    assert list(decoded["color_seq_wta"]) == list(spots["color_seq"])
    assert list(decoded["gene_wta"]) == list(expected_genes)
    assert set(decoded.loc[decoded["call_type"] == "exact", "spot_id"]) == set(
        good["spot_id"]
    )
    assert decoded.loc[2, "call_type"] == "no_call"
    assert decoded.loc[2, "reject_reason"] == "rescue_disabled"


def test_one_error_index_returns_expected_candidate():
    seq_to_gene = {"4422": "GeneA"}
    index = build_one_error_index(seq_to_gene, n_channels=4, n_rounds=4)

    assert index["4322"] == ["4422"]
    assert candidate_sequences("4322", index, seq_to_gene, max_hamming=1) == ["4422"]
    assert candidate_sequences("4321", index, seq_to_gene, max_hamming=1) == []


def test_low_margin_one_error_is_rescued():
    seq_to_gene = {"4422": "GeneA"}
    tensor = _tensor_for_sequences(["4422"])
    # Round 1 WTA is channel 3 by a tiny margin, making observed seq 4322.
    tensor[0, :, 1] = [1.0, 1.0, 51.0, 50.0]

    decoded = decode_codebook_aware(
        tensor,
        seq_to_gene,
        min_corrected_round_margin=0.02,
        min_geomean_prob=0.30,
    )

    row = decoded.iloc[0]
    assert row["color_seq_wta"] == "4322"
    assert row["decoded_seq"] == "4422"
    assert row["gene"] == "GeneA"
    assert row["call_type"] == "rescued_h1"
    assert row["corrected_rounds"] == "1"


def test_high_margin_wrong_sequence_is_rejected():
    seq_to_gene = {"4422": "GeneA"}
    tensor = _tensor_for_sequences(["4422"])
    # Round 1 WTA is confidently channel 3; the decoder should not override it.
    tensor[0, :, 1] = [1.0, 1.0, 90.0, 10.0]

    decoded = decode_codebook_aware(
        tensor,
        seq_to_gene,
        min_corrected_round_margin=0.20,
        min_geomean_prob=0.10,
    )

    row = decoded.iloc[0]
    assert row["color_seq_wta"] == "4322"
    assert row["call_type"] == "no_call"
    assert row["reject_reason"] == "corrected_round_margin_too_high"


def test_ambiguous_candidates_are_rejected():
    seq_to_gene = {"4422": "GeneA", "4222": "GeneB"}
    tensor = _tensor_for_sequences(["4422"])
    # Observed 4322 is one edit from both codebook entries with equal support.
    tensor[0, :, 1] = [1.0, 40.0, 41.0, 40.0]

    decoded = decode_codebook_aware(
        tensor,
        seq_to_gene,
        min_corrected_round_margin=0.20,
        min_score_delta=0.25,
        min_geomean_prob=0.10,
    )

    row = decoded.iloc[0]
    assert row["color_seq_wta"] == "4322"
    assert row["call_type"] == "no_call"
    assert row["reject_reason"] == "ambiguous_candidate"


def test_unknown_round_wildcard_rescue():
    seq_to_gene = {"4422": "GeneA"}
    tensor = _tensor_for_sequences(["4422"])
    # Round 1 has no signal, producing an M wildcard: observed seq 4M22.
    tensor[0, :, 1] = 0.0

    decoded = decode_codebook_aware(
        tensor,
        seq_to_gene,
        max_hamming=1,
        min_geomean_prob=0.30,
    )

    row = decoded.iloc[0]
    assert row["color_seq_wta"] == "4M22"
    assert row["decoded_seq"] == "4422"
    assert row["gene"] == "GeneA"
    assert row["call_type"] == "rescued_unknown"
    assert row["corrected_rounds"] == "1"


def test_unknown_round_multiple_candidates_rejected():
    seq_to_gene = {"4422": "GeneA", "4322": "GeneB"}
    tensor = _tensor_for_sequences(["4422"])
    tensor[0, :, 1] = 0.0

    decoded = decode_codebook_aware(
        tensor,
        seq_to_gene,
        max_hamming=1,
        min_score_delta=0.25,
        min_geomean_prob=0.10,
    )

    row = decoded.iloc[0]
    assert row["color_seq_wta"] == "4M22"
    assert row["call_type"] == "no_call"
    assert row["reject_reason"] == "ambiguous_candidate"


def test_output_schema_for_empty_input():
    decoded = decode_codebook_aware(np.empty((0, 4, 4)), {"4422": "GeneA"})

    assert decoded.empty
    assert list(decoded.columns) == [
        "spot_id",
        "color_seq_wta",
        "gene_wta",
        "decoded_seq",
        "gene",
        "call_type",
        "reject_reason",
        "hamming_to_wta",
        "corrected_rounds",
        "score",
        "score_delta",
        "geomean_prob",
        "min_round_margin",
        "corrected_round_margin",
        "mean_total_intensity",
    ]


def test_invalid_shapes_raise():
    with pytest.raises(ValueError, match="Expected intensity tensor"):
        decode_codebook_aware(np.zeros((4, 4)), {"4422": "GeneA"})

    with pytest.raises(ValueError, match="expected 4"):
        decode_codebook_aware(np.zeros((1, 4, 4)), {"442": "GeneA"})

    with pytest.raises(ValueError, match="outside 1..4"):
        decode_codebook_aware(np.zeros((1, 4, 4)), {"5522": "GeneA"})
