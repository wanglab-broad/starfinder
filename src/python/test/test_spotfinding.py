"""Tests for starfinder.spotfinding module."""

import numpy as np
import pandas as pd
import pytest

from starfinder.spotfinding import find_spots_3d
from starfinder.spotfinding.local_maxima import SPOT_COLUMNS


class TestFindSpots3D:
    """Tests for find_spots_3d function."""

    def test_finds_known_spots(self, mini_dataset, mini_ground_truth):
        """Detects spots in mini synthetic dataset, count sanity check."""
        from starfinder.io import load_image_stacks

        images, _ = load_image_stacks(
            mini_dataset / "FOV_001" / "round1",
            channel_order=["ch00", "ch01", "ch02", "ch03"],
        )
        spots = find_spots_3d(images)

        # Mini dataset has 20 ground truth spots across 4 channels in round 1
        # Detection may find more (noise peaks) or fewer (dim spots), but
        # should be in a reasonable range
        assert len(spots) > 0
        assert len(spots) >= 10  # should find at least half

    def test_returns_correct_schema(self):
        """DataFrame columns = [z, y, x, intensity, channel]."""
        image = np.zeros((5, 32, 32, 2), dtype=np.uint8)
        image[2, 16, 16, 0] = 200  # one bright spot
        spots = find_spots_3d(image)
        assert list(spots.columns) == SPOT_COLUMNS

    def test_empty_image(self):
        """Blank image returns empty DataFrame with correct schema."""
        image = np.zeros((5, 32, 32, 2), dtype=np.uint8)
        spots = find_spots_3d(image)
        assert len(spots) == 0
        assert list(spots.columns) == SPOT_COLUMNS

    def test_adaptive_threshold(self):
        """Adaptive mode: threshold = max * fraction."""
        image = np.zeros((5, 32, 32, 1), dtype=np.uint8)
        image[2, 16, 16, 0] = 100  # bright spot
        image[2, 10, 10, 0] = 10  # dim spot

        # threshold = 100 * 0.05 = 5 → both spots detected
        spots_low = find_spots_3d(
            image, intensity_estimation="adaptive", intensity_threshold=0.05
        )
        assert len(spots_low) == 2

        # threshold = 100 * 0.5 = 50 → only bright spot
        spots_high = find_spots_3d(
            image, intensity_estimation="adaptive", intensity_threshold=0.5
        )
        assert len(spots_high) == 1
        assert spots_high.iloc[0]["intensity"] == 100

    def test_global_threshold(self):
        """Global mode: threshold = dtype_max * fraction."""
        image = np.zeros((5, 32, 32, 1), dtype=np.uint8)
        image[2, 16, 16, 0] = 200  # bright spot
        image[2, 10, 10, 0] = 40  # dim spot

        # global threshold = 255 * 0.1 = 25.5 → both detected
        spots = find_spots_3d(image, intensity_estimation="global", intensity_threshold=0.1)
        assert len(spots) == 2

        # global threshold = 255 * 0.5 = 127.5 → only bright spot
        spots = find_spots_3d(image, intensity_estimation="global", intensity_threshold=0.5)
        assert len(spots) == 1

    def test_multichannel(self):
        """Spots in different channels detected with correct channel index."""
        image = np.zeros((5, 32, 32, 4), dtype=np.uint8)
        image[1, 10, 10, 0] = 200  # channel 0
        image[2, 20, 20, 2] = 200  # channel 2
        image[3, 15, 15, 3] = 200  # channel 3

        spots = find_spots_3d(image)
        assert len(spots) == 3

        # Check each channel has exactly one spot
        channels = sorted(spots["channel"].tolist())
        assert channels == [0, 2, 3]

    def test_coordinate_values(self):
        """Spot at known (z, y, x) returns correct coordinates."""
        image = np.zeros((10, 64, 64, 1), dtype=np.uint8)
        image[3, 25, 40, 0] = 255

        spots = find_spots_3d(image)
        assert len(spots) == 1
        assert spots.iloc[0]["z"] == 3
        assert spots.iloc[0]["y"] == 25
        assert spots.iloc[0]["x"] == 40
        assert spots.iloc[0]["intensity"] == 255
        assert spots.iloc[0]["channel"] == 0

    def test_noise_threshold(self):
        """Noise mode: threshold = median + k * MAD * 1.4826.

        With Gaussian-like background (mean=20, std≈5) and bright spots,
        k=5 should detect only the bright spots above the noise floor.
        """
        rng = np.random.RandomState(42)
        image = rng.normal(20, 5, (5, 32, 32, 1)).clip(0, 255).astype(np.uint8)
        image[2, 16, 16, 0] = 200  # bright spot well above noise

        # k=5: threshold ≈ 20 + 5*5*1.4826 ≈ 57 → bright spot detected
        spots = find_spots_3d(image, intensity_estimation="noise", intensity_threshold=5.0)
        assert len(spots) >= 1
        # The bright spot should be among the detected
        bright = spots[spots["intensity"] >= 150]
        assert len(bright) == 1
        assert bright.iloc[0]["z"] == 2
        assert bright.iloc[0]["y"] == 16

    def test_adaptive_round_threshold(self):
        """adaptive_round uses max across all channels for threshold.

        Channel 0 is dim (max=50), channel 1 is bright (max=255).
        With adaptive_round, ch0 threshold = 255*0.2 = 51 > 50 → no spots.
        With adaptive, ch0 threshold = 50*0.2 = 10 → spots detected.
        """
        image = np.zeros((5, 32, 32, 2), dtype=np.uint8)
        image[2, 16, 16, 0] = 50   # dim spot in ch0
        image[2, 10, 10, 1] = 255  # bright spot in ch1

        # adaptive_round: threshold = 255 * 0.2 = 51 → ch0 spot (50) suppressed
        spots_round = find_spots_3d(
            image, intensity_estimation="adaptive_round", intensity_threshold=0.2
        )
        ch0_spots = spots_round[spots_round["channel"] == 0]
        assert len(ch0_spots) == 0
        assert len(spots_round[spots_round["channel"] == 1]) == 1

        # adaptive: ch0 threshold = 50 * 0.2 = 10 → ch0 spot detected
        spots_adaptive = find_spots_3d(
            image, intensity_estimation="adaptive", intensity_threshold=0.2
        )
        ch0_spots = spots_adaptive[spots_adaptive["channel"] == 0]
        assert len(ch0_spots) == 1
