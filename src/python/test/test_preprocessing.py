"""Tests for starfinder.preprocessing module."""

import numpy as np
import pytest

from starfinder.preprocessing import (
    histogram_match,
    min_max_normalize,
    morphological_reconstruction,
    tophat_filter,
)


class TestMinMaxNormalize:
    """Tests for min_max_normalize."""

    def test_rescales_to_uint8(self):
        """Output spans [0, 255] for non-constant input."""
        vol = np.random.RandomState(42).randint(50, 200, (5, 16, 16), dtype=np.uint8)
        result = min_max_normalize(vol)
        assert result.dtype == np.uint8
        assert result.min() == 0
        assert result.max() == 255

    def test_per_channel_independence(self):
        """Each channel is normalized independently."""
        vol = np.zeros((3, 8, 8, 2), dtype=np.uint8)
        vol[:, :, :, 0] = 100  # constant → zeros
        vol[:, :, :, 1] = np.arange(192, dtype=np.uint8).reshape(3, 8, 8)
        result = min_max_normalize(vol)
        # Channel 0: constant → all zeros
        assert result[:, :, :, 0].max() == 0
        # Channel 1: variable → spans full range
        assert result[:, :, :, 1].max() == 255

    def test_constant_channel_zeros(self):
        """Constant-valued channel maps to all zeros."""
        vol = np.full((3, 8, 8), 42, dtype=np.uint8)
        result = min_max_normalize(vol)
        assert np.all(result == 0)

    def test_3d_input(self):
        """Handles (Z, Y, X) input without extra channel dim."""
        vol = np.random.RandomState(0).randint(0, 256, (5, 16, 16), dtype=np.uint8)
        result = min_max_normalize(vol)
        assert result.ndim == 3
        assert result.shape == vol.shape

    def test_4d_input(self):
        """Handles (Z, Y, X, C) input."""
        vol = np.random.RandomState(1).randint(0, 256, (5, 16, 16, 4), dtype=np.uint8)
        result = min_max_normalize(vol)
        assert result.ndim == 4
        assert result.shape == vol.shape


class TestHistogramMatch:
    """Tests for histogram_match."""

    def test_dtype_preserved(self):
        """Output dtype matches input dtype."""
        vol = np.random.RandomState(42).randint(0, 128, (3, 16, 16), dtype=np.uint8)
        ref = np.random.RandomState(99).randint(128, 256, (3, 16, 16), dtype=np.uint8)
        result = histogram_match(vol, ref)
        assert result.dtype == vol.dtype

    def test_shape_preserved(self):
        """Output shape matches input shape for 3D and 4D."""
        vol3d = np.random.RandomState(0).randint(0, 256, (3, 16, 16), dtype=np.uint8)
        ref = np.random.RandomState(1).randint(0, 256, (3, 16, 16), dtype=np.uint8)
        assert histogram_match(vol3d, ref).shape == vol3d.shape

        vol4d = np.random.RandomState(2).randint(0, 256, (3, 16, 16, 2), dtype=np.uint8)
        assert histogram_match(vol4d, ref).shape == vol4d.shape

    def test_histogram_shifts_toward_reference(self):
        """Matched volume mean shifts toward reference mean."""
        vol = np.random.RandomState(42).randint(0, 50, (5, 32, 32), dtype=np.uint8)
        ref = np.random.RandomState(99).randint(200, 256, (5, 32, 32), dtype=np.uint8)
        result = histogram_match(vol, ref)
        # Result mean should be closer to ref mean than original was
        assert abs(result.mean() - ref.mean()) < abs(vol.mean() - ref.mean())


class TestMorphologicalReconstruction:
    """Tests for morphological_reconstruction."""

    def test_uint8_output(self):
        """Output is uint8."""
        vol = np.random.RandomState(42).randint(0, 256, (3, 32, 32), dtype=np.uint8)
        result = morphological_reconstruction(vol, radius=3)
        assert result.dtype == np.uint8

    def test_shape_preserved(self):
        """Output shape matches input for 3D and 4D."""
        vol3d = np.random.RandomState(0).randint(0, 256, (3, 32, 32), dtype=np.uint8)
        assert morphological_reconstruction(vol3d).shape == vol3d.shape

        vol4d = np.random.RandomState(1).randint(0, 256, (3, 32, 32, 2), dtype=np.uint8)
        assert morphological_reconstruction(vol4d).shape == vol4d.shape

    def test_background_removed(self):
        """Smooth background is reduced; small features preserved."""
        # Create image with uniform background + bright spot
        img = np.full((1, 64, 64), 100, dtype=np.uint8)
        img[0, 32, 32] = 255
        result = morphological_reconstruction(img, radius=5)
        # Background should be suppressed (lower than original 100)
        bg_val = result[0, 0, 0]
        assert bg_val < 100


class TestTophatFilter:
    """Tests for tophat_filter."""

    def test_uint8_output(self):
        """Output is uint8."""
        vol = np.random.RandomState(42).randint(0, 256, (3, 32, 32), dtype=np.uint8)
        result = tophat_filter(vol, radius=3)
        assert result.dtype == np.uint8

    def test_shape_preserved(self):
        """Output shape matches input for 3D and 4D."""
        vol3d = np.random.RandomState(0).randint(0, 256, (3, 32, 32), dtype=np.uint8)
        assert tophat_filter(vol3d).shape == vol3d.shape

        vol4d = np.random.RandomState(1).randint(0, 256, (3, 32, 32, 2), dtype=np.uint8)
        assert tophat_filter(vol4d).shape == vol4d.shape

    def test_removes_large_structures(self):
        """Uniform background is removed; small features remain."""
        img = np.full((1, 64, 64), 100, dtype=np.uint8)
        img[0, 32, 32] = 255  # small bright spot
        result = tophat_filter(img, radius=5)
        # Background should be ~0 (tophat removes structures larger than SE)
        assert result[0, 0, 0] < 10
        # Bright spot should be preserved (relative to background)
        assert result[0, 32, 32] > 100
