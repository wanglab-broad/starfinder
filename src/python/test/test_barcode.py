"""Tests for barcode encoding, decoding, codebook loading, and filtering."""

from pathlib import Path

import pandas as pd
import pytest

from starfinder.barcode.encoding import (
    BASE_PAIR_TO_COLOR,
    COLOR_TO_BASE_PAIRS,
    decode_color_seq,
    encode_bases,
)
from starfinder.barcode.codebook import load_codebook
from starfinder.barcode.filtering import filter_reads
from starfinder.testdata.synthetic import TEST_CODEBOOK

# Resolve path relative to this file → repo root / tests/fixtures/...
_REPO_ROOT = Path(__file__).resolve().parents[3]
MINI_CODEBOOK = _REPO_ROOT / "tests" / "fixtures" / "synthetic" / "mini" / "codebook.csv"


# --- Encoding ---


class TestEncodeBases:
    def test_known_sequence(self):
        """encode_bases('CGCAC') should produce '4422' (no reversal)."""
        assert encode_bases("CGCAC") == "4422"

    def test_output_length(self):
        assert len(encode_bases("CGCAC")) == 4
        assert len(encode_bases("AC")) == 1

    def test_all_same_bases(self):
        """Same-base pairs always map to color '1'."""
        assert encode_bases("AAAA") == "111"
        assert encode_bases("CCCC") == "111"


# --- Decoding ---


class TestDecodeColorSeq:
    def test_known_decode(self):
        """decode_color_seq('4422', 'C') should produce 'CGCAC'."""
        assert decode_color_seq("4422", "C") == "CGCAC"

    def test_encode_decode_roundtrip(self):
        """encode -> decode with correct start base recovers original."""
        for barcode in ["CGCAC", "ACGTG", "TTTTT", "ATCGA"]:
            encoded = encode_bases(barcode)
            decoded = decode_color_seq(encoded, barcode[0])
            assert decoded == barcode, f"Roundtrip failed for {barcode}"

    def test_all_codebook_entries_roundtrip(self):
        """All 8 mini codebook barcodes survive encode->decode roundtrip."""
        for gene, barcode in TEST_CODEBOOK:
            # The codebook uses reversed barcodes for encoding
            reversed_bc = barcode[::-1]
            encoded = encode_bases(reversed_bc)
            decoded = decode_color_seq(encoded, reversed_bc[0])
            assert decoded == reversed_bc, (
                f"{gene}: {reversed_bc} -> {encoded} -> {decoded}"
            )

    def test_single_color(self):
        """Single-color sequence produces 2-base barcode."""
        result = decode_color_seq("1", "A")
        assert result == "AA"
        assert len(result) == 2


# --- Codebook ---


class TestLoadCodebook:
    def test_load_mini_codebook(self):
        gene_to_seq, seq_to_gene = load_codebook(MINI_CODEBOOK)
        assert len(gene_to_seq) == 8
        assert len(seq_to_gene) == 8

    def test_gene_lookup(self):
        gene_to_seq, _ = load_codebook(MINI_CODEBOOK)
        # CACGC reversed = CGCAC, encode -> 4422
        assert gene_to_seq["GeneA"] == "4422"

    def test_seq_lookup(self):
        _, seq_to_gene = load_codebook(MINI_CODEBOOK)
        assert seq_to_gene["4422"] == "GeneA"

    def test_bidirectional_consistency(self):
        """gene_to_seq and seq_to_gene are inverses."""
        gene_to_seq, seq_to_gene = load_codebook(MINI_CODEBOOK)
        for gene, seq in gene_to_seq.items():
            assert seq_to_gene[seq] == gene

    def test_no_reverse(self):
        """With do_reverse=False, encoding uses barcode as-is."""
        gene_to_seq_rev, _ = load_codebook(MINI_CODEBOOK, do_reverse=True)
        gene_to_seq_fwd, _ = load_codebook(MINI_CODEBOOK, do_reverse=False)
        # Results should differ (unless a barcode is a palindrome)
        assert gene_to_seq_rev != gene_to_seq_fwd


# --- Filtering ---


class TestFilterReads:
    @pytest.fixture()
    def codebook(self):
        _, seq_to_gene = load_codebook(MINI_CODEBOOK)
        return seq_to_gene

    def test_basic_filtering(self, codebook):
        """Valid color sequences get gene assigned."""
        spots = pd.DataFrame({
            "z": [1, 2], "y": [10, 20], "x": [30, 40],
            "color_seq": ["4422", "4242"],
        })
        good, stats = filter_reads(spots, codebook)

        assert len(good) == 2
        assert good.iloc[0]["gene"] == "GeneA"
        assert good.iloc[1]["gene"] == "GeneB"
        assert stats["in_codebook"] == 1.0

    def test_invalid_filtered_out(self, codebook):
        """Non-codebook sequences are excluded."""
        spots = pd.DataFrame({
            "z": [1, 2, 3], "y": [10, 20, 30], "x": [30, 40, 50],
            "color_seq": ["4422", "1111", "9999"],
        })
        good, stats = filter_reads(spots, codebook)

        assert len(good) == 1
        assert good.iloc[0]["gene"] == "GeneA"
        assert stats["n_in_codebook"] == 1
        assert stats["n_total"] == 3

    def test_stats_keys(self, codebook):
        spots = pd.DataFrame({
            "z": [1], "y": [10], "x": [30],
            "color_seq": ["4422"],
        })
        _, stats = filter_reads(spots, codebook)
        assert "n_total" in stats
        assert "n_in_codebook" in stats
        assert "in_codebook" in stats

    def test_end_bases_diagnostic(self, codebook):
        """end_bases adds diagnostic stats without changing filtering."""
        spots = pd.DataFrame({
            "z": [1, 2], "y": [10, 20], "x": [30, 40],
            "color_seq": ["4422", "1111"],
        })
        good, stats = filter_reads(spots, codebook, end_bases="CC", start_base="C")

        # Filtering result unchanged — only codebook match matters
        assert len(good) == 1
        # Diagnostic keys present
        assert "correct_form" in stats
        assert "validated" in stats

    def test_empty_input(self, codebook):
        spots = pd.DataFrame({"z": [], "y": [], "x": [], "color_seq": []})
        good, stats = filter_reads(spots, codebook)
        assert len(good) == 0
        assert stats["in_codebook"] == 0.0

    def test_gene_column_added(self, codebook):
        """Output DataFrame has 'gene' column."""
        spots = pd.DataFrame({
            "z": [1], "y": [10], "x": [30],
            "color_seq": ["4422"],
        })
        good, _ = filter_reads(spots, codebook)
        assert "gene" in good.columns


# --- End-to-end ---


class TestEndToEnd:
    def test_ground_truth_pipeline(self):
        """Codebook + filtering recovers gene names from known color sequences."""
        gene_to_seq, seq_to_gene = load_codebook(MINI_CODEBOOK)

        # Simulate extracted spots with known color sequences from all 8 genes
        spots_data = []
        expected_genes = []
        for gene, barcode in TEST_CODEBOOK:
            seq = gene_to_seq[gene]
            spots_data.append({"z": 1, "y": 10, "x": 20, "color_seq": seq})
            expected_genes.append(gene)

        # Add some invalid reads
        spots_data.append({"z": 1, "y": 10, "x": 20, "color_seq": "1111"})
        spots_data.append({"z": 1, "y": 10, "x": 20, "color_seq": "XXXX"})

        spots = pd.DataFrame(spots_data)
        good, stats = filter_reads(spots, seq_to_gene)

        assert len(good) == 8
        assert list(good["gene"]) == expected_genes
        assert stats["n_in_codebook"] == 8
        assert stats["n_total"] == 10
