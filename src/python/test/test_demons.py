"""Tests for starfinder.registration.demons module."""

import numpy as np
import pytest

# Skip all tests if SimpleITK not installed
sitk = pytest.importorskip("SimpleITK", reason="SimpleITK required for local registration")


class TestDemonsRegister:
    """Tests for demons_register function."""

    def test_identity(self, mini_dataset):
        """Identical images produce near-zero displacement field."""
        from starfinder.io import load_multipage_tiff
        from starfinder.registration.demons import demons_register

        vol = load_multipage_tiff(mini_dataset / "FOV_001" / "round1" / "ch00.tif")
        field = demons_register(vol, vol)

        # Shape should be (Z, Y, X, 3) for displacement vectors
        assert field.shape == (*vol.shape, 3)
        # Displacement should be near zero for identical images
        assert np.abs(field).max() < 1.0, "Displacement should be near-zero for identical images"


class TestApplyDeformation:
    """Tests for apply_deformation function."""

    def test_identity_field(self, mini_dataset):
        """Zero displacement field returns original volume."""
        from starfinder.io import load_multipage_tiff
        from starfinder.registration.demons import apply_deformation

        vol = load_multipage_tiff(mini_dataset / "FOV_001" / "round1" / "ch00.tif")

        # Zero displacement field
        field = np.zeros((*vol.shape, 3), dtype=np.float32)
        result = apply_deformation(vol, field)

        assert result.shape == vol.shape
        # Should be nearly identical (interpolation may introduce tiny differences)
        np.testing.assert_allclose(result, vol, rtol=1e-4, atol=1e-4)
