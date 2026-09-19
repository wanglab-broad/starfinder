"""Tests for starfinder.preprocessing module."""

from starfinder.preprocessing import ProjectionConfig

import numpy as np
import pytest

from starfinder.preprocessing import project_image


class TestMakeProjection:
    """Tests for project_image."""

    def test_max_3d(self):
        """Max projection of (Z, Y, X) retains (1, Y, X)."""
        vol = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.uint8)
        result = project_image(vol, config=ProjectionConfig(method="max"))
        assert result.shape == (1, 2, 2)
        np.testing.assert_array_equal(result, [[[5, 6], [7, 8]]])

    def test_max_4d(self):
        """Max projection of (Z, Y, X, C) retains (1, Y, X, C)."""
        vol = np.zeros((3, 4, 4, 2), dtype=np.uint8)
        vol[2, 1, 1, 0] = 200
        vol[0, 2, 2, 1] = 150
        result = project_image(vol, config=ProjectionConfig(method="max"))
        assert result.shape == (1, 4, 4, 2)
        assert result[0, 1, 1, 0] == 200
        assert result[0, 2, 2, 1] == 150

    def test_sum_accumulates(self):
        """Sum projection retains the unscaled sum."""
        vol = np.ones((10, 4, 4), dtype=np.uint8) * 100
        result = project_image(vol, config=ProjectionConfig(method="sum"))
        assert result.dtype == np.uint64
        # No implicit display normalization
        assert result.max() == 1000

    def test_invalid_method_raises(self):
        """Unknown method raises ValueError."""
        vol = np.zeros((2, 4, 4), dtype=np.uint8)
        with pytest.raises(ValueError, match="projection method"):
            project_image(vol, config=ProjectionConfig(method="median"))
