"""Tests for two-base color-space encoding."""

import pytest

from starfinder.testdata.synthetic import encode_barcode_to_colors, BASE_PAIR_TO_COLOR


class TestTwoBaseEncoding:
    """Tests for the two-base color-space encoding scheme."""

    def test_encoding_table_completeness(self):
        """Verify all 16 base pairs are mapped."""
        bases = "ACGT"
        for b1 in bases:
            for b2 in bases:
                pair = b1 + b2
                assert pair in BASE_PAIR_TO_COLOR, f"Missing pair: {pair}"

    def test_encoding_symmetry(self):
        """Verify encoding follows expected color groups."""
        # Color 1: same bases
        assert BASE_PAIR_TO_COLOR["AA"] == "1"
        assert BASE_PAIR_TO_COLOR["CC"] == "1"
        assert BASE_PAIR_TO_COLOR["GG"] == "1"
        assert BASE_PAIR_TO_COLOR["TT"] == "1"

        # Color 2: A<->C, G<->T
        assert BASE_PAIR_TO_COLOR["AC"] == "2"
        assert BASE_PAIR_TO_COLOR["CA"] == "2"
        assert BASE_PAIR_TO_COLOR["GT"] == "2"
        assert BASE_PAIR_TO_COLOR["TG"] == "2"

        # Color 3: A<->G, C<->T
        assert BASE_PAIR_TO_COLOR["AG"] == "3"
        assert BASE_PAIR_TO_COLOR["GA"] == "3"
        assert BASE_PAIR_TO_COLOR["CT"] == "3"
        assert BASE_PAIR_TO_COLOR["TC"] == "3"

        # Color 4: A<->T, C<->G
        assert BASE_PAIR_TO_COLOR["AT"] == "4"
        assert BASE_PAIR_TO_COLOR["TA"] == "4"
        assert BASE_PAIR_TO_COLOR["CG"] == "4"
        assert BASE_PAIR_TO_COLOR["GC"] == "4"

    @pytest.mark.parametrize(
        "barcode,expected_color_seq",
        [
            # Test codebook entries (barcode reversed first, then encoded)
            ("CACGC", "4422"),  # CGCAC -> CG=4, GC=4, CA=2, AC=2
            ("CATGC", "4242"),  # CGTAC -> CG=4, GT=2, TA=4, AC=2
            ("CGAAC", "2134"),  # CAAGC -> CA=2, AA=1, AG=3, GC=4
            ("CGTAC", "2424"),  # CATGC -> CA=2, AT=4, TG=2, GC=4
            ("CTGAC", "2323"),  # CAGTC -> CA=2, AG=3, GT=2, TC=3
            ("CTAGC", "4343"),  # CGATC -> CG=4, GA=3, AT=4, TC=3
            ("CCATC", "3421"),  # CTACC -> CT=3, TA=4, AC=2, CC=1
            ("CGCTC", "3344"),  # CTCGC -> CT=3, TC=3, CG=4, GC=4
        ],
    )
    def test_barcode_encoding(self, barcode: str, expected_color_seq: str):
        """Verify barcode encoding matches expected color sequence."""
        result = encode_barcode_to_colors(barcode)
        assert result == expected_color_seq, (
            f"Barcode {barcode} -> reversed {barcode[::-1]} -> "
            f"expected {expected_color_seq}, got {result}"
        )

    def test_encoding_output_length(self):
        """Verify color sequence is one less than barcode length."""
        barcode = "CACGC"
        result = encode_barcode_to_colors(barcode)
        assert len(result) == len(barcode) - 1
