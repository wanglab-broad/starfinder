"""Tests for starfinder.preprocessing module."""

from starfinder.preprocessing import MinMaxNormalizationConfig
from starfinder.preprocessing import ReconstructionConfig
from starfinder.preprocessing import TophatConfig

import numpy as np
import pytest

from starfinder.preprocessing import (
    match_histogram,
    normalize_intensity,
    reconstruct_background,
    filter_tophat,
)


class TestMinMaxNormalize:
    """Tests for normalize_intensity."""

    def test_rescales_to_uint8(self):
        """Output spans [0, 255] for non-constant input."""
        vol = np.random.RandomState(42).randint(50, 200, (5, 16, 16), dtype=np.uint8)
        result = normalize_intensity(vol, config=MinMaxNormalizationConfig('uint8', (0, 255)))
        assert result.dtype == np.uint8
        assert result.shape == vol.shape
        assert result.min() == 0
        assert result.max() == 255

    def test_per_channel_independence(self):
        """Each channel is normalized independently."""
        vol = np.zeros((3, 8, 8, 2), dtype=np.uint8)
        vol[:, :, :, 0] = 100  # constant → zeros
        vol[:, :, :, 1] = np.arange(192, dtype=np.uint8).reshape(3, 8, 8)
        result = normalize_intensity(vol, config=MinMaxNormalizationConfig('uint8', (0, 255)))
        assert result.shape == vol.shape
        assert result.dtype == vol.dtype
        # Channel 0: constant → all zeros
        assert result[:, :, :, 0].max() == 0
        # Channel 1: variable → spans full range
        assert result[:, :, :, 1].max() == 255

    def test_constant_channel_zeros(self):
        """Constant-valued channel maps to all zeros."""
        vol = np.full((3, 8, 8), 42, dtype=np.uint8)
        result = normalize_intensity(vol, config=MinMaxNormalizationConfig('uint8', (0, 255)))
        assert np.all(result == 0)




class TestSNRGating:
    """Tests for SNR-gated normalization."""

    def test_snr_gating_skips_low_snr_channel(self):
        """Low-SNR channel is NOT rescaled; high-SNR channel IS rescaled."""
        vol = np.zeros((3, 32, 32, 2), dtype=np.uint8)
        # Channel 0: nonconstant low SNR (max ~= mean, SNR ≈ 1)
        vol[:, :, :, 0] = 50
        vol[0, 0, 0, 0] = 49
        # Channel 1: high SNR (background=5, bright spot=200, SNR >> 5)
        vol[:, :, :, 1] = 5
        vol[1, 16, 16, 1] = 200

        result = normalize_intensity(vol, config=MinMaxNormalizationConfig('uint8', (0, 255), snr_threshold=5.0))

        # Nonconstant low-SNR values are retained exactly.
        np.testing.assert_array_equal(result[:, :, :, 0], vol[:, :, :, 0])
        # Channel 1: should be rescaled to [0, 255]
        assert result[:, :, :, 1].max() == 255

    def test_snr_gating_none_is_backward_compat(self):
        """With snr_threshold=None, all channels are normalized (default)."""
        vol = np.zeros((3, 32, 32, 2), dtype=np.uint8)
        vol[:, :, :, 0] = 50  # low SNR
        vol[:, :, :, 1] = 5
        vol[1, 16, 16, 1] = 200

        result = normalize_intensity(vol, config=MinMaxNormalizationConfig('uint8', (0, 255), snr_threshold=None))

        # Both channels should be rescaled to [0, 255]
        # Channel 0: constant → all zeros (lo == hi case)
        assert result[:, :, :, 0].max() == 0
        # Channel 1: rescaled to 255
        assert result[:, :, :, 1].max() == 255


class TestHistogramMatch:
    """Tests for match_histogram."""


    def test_shape_preserved(self):
        """Output shape matches input shape for 3D and 4D."""
        vol3d = np.random.RandomState(0).randint(0, 256, (3, 16, 16), dtype=np.uint8)
        ref = np.random.RandomState(1).randint(0, 256, (3, 16, 16), dtype=np.uint8)
        assert match_histogram(vol3d, ref).shape == vol3d.shape

        vol4d = np.random.RandomState(2).randint(0, 256, (3, 16, 16, 2), dtype=np.uint8)
        assert match_histogram(vol4d, ref).shape == vol4d.shape

    def test_histogram_shifts_toward_reference(self):
        """Matched volume mean shifts toward reference mean."""
        vol = np.random.RandomState(42).randint(0, 50, (5, 32, 32), dtype=np.uint8)
        ref = np.random.RandomState(99).randint(200, 256, (5, 32, 32), dtype=np.uint8)
        result = match_histogram(vol, ref)
        assert result.dtype == vol.dtype
        assert result.shape == vol.shape
        # Result mean should be closer to ref mean than original was
        assert abs(result.mean() - ref.mean()) < abs(vol.mean() - ref.mean())


class TestMorphologicalReconstruction:
    """Tests for reconstruct_background."""


    def test_shape_preserved(self):
        """Output shape matches input for 3D and 4D."""
        vol3d = np.random.RandomState(0).randint(0, 256, (3, 32, 32), dtype=np.uint8)
        assert reconstruct_background(vol3d, config=ReconstructionConfig(radius_yx=3)).shape == vol3d.shape

        vol4d = np.random.RandomState(1).randint(0, 256, (3, 32, 32, 2), dtype=np.uint8)
        assert reconstruct_background(vol4d, config=ReconstructionConfig(radius_yx=3)).shape == vol4d.shape

    def test_background_removed(self):
        """Smooth background is reduced; small features preserved."""
        # Create image with uniform background + bright spot
        img = np.full((1, 64, 64), 100, dtype=np.uint8)
        img[0, 32, 32] = 255
        result = reconstruct_background(img, config=ReconstructionConfig(radius_yx=5))
        assert result.dtype == img.dtype
        assert result.shape == img.shape
        # Background should be suppressed (lower than original 100)
        bg_val = result[0, 0, 0]
        assert bg_val < 100


class TestTophatFilter:
    """Tests for filter_tophat."""


    def test_shape_preserved(self):
        """Output shape matches input for 3D and 4D."""
        vol3d = np.random.RandomState(0).randint(0, 256, (3, 32, 32), dtype=np.uint8)
        assert filter_tophat(vol3d, config=TophatConfig(radius_yx=3)).shape == vol3d.shape

        vol4d = np.random.RandomState(1).randint(0, 256, (3, 32, 32, 2), dtype=np.uint8)
        assert filter_tophat(vol4d, config=TophatConfig(radius_yx=3)).shape == vol4d.shape

    def test_removes_large_structures(self):
        """Uniform background is removed; small features remain."""
        img = np.full((1, 64, 64), 100, dtype=np.uint8)
        img[0, 32, 32] = 255  # small bright spot
        result = filter_tophat(img, config=TophatConfig(radius_yx=5))
        assert result.dtype == img.dtype
        assert result.shape == img.shape
        # Background should be ~0 (tophat removes structures larger than SE)
        assert result[0, 0, 0] < 10
        # Bright spot should be preserved (relative to background)
        assert result[0, 32, 32] > 100
