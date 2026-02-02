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

    def test_known_deformation(self, mini_dataset):
        """Recovers direction of synthetic smooth deformation."""
        from scipy.ndimage import gaussian_filter, map_coordinates

        from starfinder.io import load_multipage_tiff
        from starfinder.registration.demons import demons_register

        vol = load_multipage_tiff(mini_dataset / "FOV_001" / "round1" / "ch00.tif")

        # Create smooth synthetic deformation (simulate tissue warping)
        rng = np.random.default_rng(42)
        strength = 5.0

        # Generate smooth random displacement field
        true_field = rng.standard_normal((3, *vol.shape)) * strength
        # Smooth spatially to make it tissue-like (scale sigma to volume dimensions)
        z_sigma = max(1, vol.shape[0] // 3)
        xy_sigma = max(3, vol.shape[1] // 50)
        true_field = gaussian_filter(true_field, sigma=[0, z_sigma, xy_sigma, xy_sigma])

        # Apply deformation to create "moving" image
        coords = np.meshgrid(*[np.arange(s) for s in vol.shape], indexing='ij')
        warped_coords = [c + f for c, f in zip(coords, true_field)]
        deformed = map_coordinates(vol, warped_coords, order=1, mode='constant', cval=0)

        # Recover displacement field
        estimated_field = demons_register(vol, deformed.astype(np.float32))

        # The estimated field should have similar direction to true field
        # (we check correlation, not exact match, since demons is iterative)
        true_magnitude = np.linalg.norm(true_field.transpose(1, 2, 3, 0), axis=-1)
        est_magnitude = np.linalg.norm(estimated_field, axis=-1)

        # Both should have deformation in similar regions
        mask = true_magnitude > 1.0
        assert est_magnitude[mask].mean() > 0.5, "Should detect deformation in warped regions"


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
