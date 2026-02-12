"""Tests for starfinder.barcode.extraction module."""

import numpy as np
import pandas as pd
import pytest

from starfinder.barcode import extract_from_location


def _make_spots(*positions):
    """Helper: create spots DataFrame from (z, y, x) tuples."""
    return pd.DataFrame(positions, columns=["z", "y", "x"])


class TestExtractFromLocation:
    """Tests for extract_from_location function."""

    def test_single_spot_extraction(self):
        """Extracts correct channel from single-channel signal."""
        image = np.zeros((5, 32, 32, 4), dtype=np.uint8)
        # Put strong signal in channel 2 (0-based) at spot location
        image[2, 16, 16, 2] = 200
        spots = _make_spots((2, 16, 16))

        color_seq, color_score = extract_from_location(image, spots)

        assert len(color_seq) == 1
        assert color_seq[0] == "3"  # channel 2 (0-based) → "3" (1-based)

    def test_winner_take_all(self):
        """Selects channel with maximum normalized intensity."""
        image = np.zeros((5, 32, 32, 4), dtype=np.uint8)
        # Channel 0: weak, Channel 3: strong
        image[2, 16, 16, 0] = 50
        image[2, 16, 16, 3] = 200
        spots = _make_spots((2, 16, 16))

        color_seq, _ = extract_from_location(image, spots, voxel_size=(0, 0, 0))

        assert color_seq[0] == "4"  # channel 3 (0-based) → "4" (1-based)

    def test_tie_returns_M(self):
        """Equal max in multiple channels returns 'M'."""
        image = np.zeros((5, 32, 32, 4), dtype=np.uint8)
        image[2, 16, 16, 0] = 100
        image[2, 16, 16, 1] = 100
        spots = _make_spots((2, 16, 16))

        color_seq, color_score = extract_from_location(image, spots, voxel_size=(0, 0, 0))

        assert color_seq[0] == "M"
        assert color_score[0] == np.inf

    def test_zero_signal_handling(self):
        """All-zero neighborhood handled gracefully (epsilon prevents NaN)."""
        image = np.zeros((5, 32, 32, 4), dtype=np.uint8)
        spots = _make_spots((2, 16, 16))

        color_seq, color_score = extract_from_location(image, spots)

        # All channels equal after normalization (all zero → epsilon only) → tie
        assert color_seq[0] == "M"
        assert color_score[0] == np.inf

    def test_voxel_neighborhood(self):
        """voxel_size controls extraction window size."""
        image = np.zeros((5, 32, 32, 4), dtype=np.uint8)
        # Place signal around the spot but not at center
        image[2, 14, 16, 1] = 200  # 2 pixels away in Y
        spots = _make_spots((2, 16, 16))

        # voxel_size=(0, 0, 0) → only center pixel → no signal → tie
        seq_narrow, _ = extract_from_location(image, spots, voxel_size=(0, 0, 0))
        assert seq_narrow[0] == "M"

        # voxel_size=(0, 2, 0) → ±2 in Y → includes signal
        seq_wide, _ = extract_from_location(image, spots, voxel_size=(0, 2, 0))
        assert seq_wide[0] == "2"  # channel 1 → "2"

    def test_boundary_clipping(self):
        """Spots near edges don't crash, extents clipped."""
        image = np.zeros((5, 32, 32, 4), dtype=np.uint8)
        image[0, 0, 0, 0] = 200  # corner spot
        spots = _make_spots((0, 0, 0))

        color_seq, _ = extract_from_location(image, spots)
        assert color_seq[0] == "1"  # channel 0 → "1"

    def test_multiple_spots(self):
        """Batch processing of multiple spots."""
        image = np.zeros((5, 32, 32, 4), dtype=np.uint8)
        image[1, 5, 5, 0] = 200  # spot 0 → channel 1
        image[2, 15, 15, 2] = 200  # spot 1 → channel 3
        image[3, 25, 25, 3] = 200  # spot 2 → channel 4
        spots = _make_spots((1, 5, 5), (2, 15, 15), (3, 25, 25))

        color_seq, color_score = extract_from_location(image, spots, voxel_size=(0, 0, 0))

        assert len(color_seq) == 3
        assert color_seq[0] == "1"
        assert color_seq[1] == "3"
        assert color_seq[2] == "4"
        # All scores should be finite for clean single-channel signals
        assert all(np.isfinite(color_score))

    def test_score_computation(self):
        """Score = -log(max_normalized) ≈ 0 for single-channel signal."""
        image = np.zeros((5, 32, 32, 4), dtype=np.uint8)
        image[2, 16, 16, 0] = 200  # only one channel has signal
        spots = _make_spots((2, 16, 16))

        _, color_score = extract_from_location(image, spots, voxel_size=(0, 0, 0))

        # With only one channel having signal, normalized max ≈ 1.0
        # so -log(1.0) ≈ 0 (with small epsilon correction)
        assert color_score[0] < 0.01
