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


class TestRegisterVolumeLocal:
    """Tests for register_volume_local function."""

    def test_multichannel(self, mini_dataset):
        """Registers all channels using computed field."""
        from starfinder.io import load_image_stacks
        from starfinder.registration.demons import register_volume_local

        images, _ = load_image_stacks(
            mini_dataset / "FOV_001" / "round1",
            ["ch00", "ch01", "ch02", "ch03"],
        )

        # Use ch00 as ref/mov (identity case)
        ref_img = images[:, :, :, 0]
        mov_img = images[:, :, :, 0]

        registered, field = register_volume_local(images, ref_img, mov_img)

        assert registered.shape == images.shape
        assert field.shape == (*images.shape[:3], 3)
        # For identity case, registered should be similar to input
        np.testing.assert_allclose(registered, images, rtol=0.1, atol=1.0)
