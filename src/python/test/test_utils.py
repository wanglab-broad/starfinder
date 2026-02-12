"""Tests for starfinder.utils module."""

import numpy as np
import pytest

from starfinder.utils import make_projection


class TestMakeProjection:
    """Tests for make_projection."""

    def test_max_3d(self):
        """Max projection of (Z, Y, X) returns (Y, X)."""
        vol = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.uint8)
        result = make_projection(vol, "max")
        assert result.shape == (2, 2)
        np.testing.assert_array_equal(result, [[5, 6], [7, 8]])

    def test_max_4d(self):
        """Max projection of (Z, Y, X, C) returns (Y, X, C)."""
        vol = np.zeros((3, 4, 4, 2), dtype=np.uint8)
        vol[2, 1, 1, 0] = 200
        vol[0, 2, 2, 1] = 150
        result = make_projection(vol, "max")
        assert result.shape == (4, 4, 2)
        assert result[1, 1, 0] == 200
        assert result[2, 2, 1] == 150

    def test_sum_accumulates(self):
        """Sum projection accumulates and rescales to uint8."""
        vol = np.ones((10, 4, 4), dtype=np.uint8) * 100
        result = make_projection(vol, "sum")
        assert result.dtype == np.uint8
        # All voxels identical → sum is uniform → rescaled max = 255
        assert result.max() == 255

    def test_invalid_method_raises(self):
        """Unknown method raises ValueError."""
        vol = np.zeros((2, 4, 4), dtype=np.uint8)
        with pytest.raises(ValueError, match="Unknown projection method"):
            make_projection(vol, "median")
