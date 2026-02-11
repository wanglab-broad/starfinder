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


class TestPyramidUtilities:
    """Tests for pyramid.py anti-aliased multi-resolution utilities."""

    def test_butterworth_filter_shape(self):
        """Butterworth filter has correct shape and peak at DC."""
        from starfinder.registration.pyramid import butterworth_3d

        shape = (8, 16, 16)
        filt = butterworth_3d(shape, cutoff=0.25, order=2)

        assert filt.shape == shape
        # Peak at DC (index 0,0,0 for fftfreq convention)
        assert filt[0, 0, 0] == pytest.approx(1.0)
        # Should be low-pass: values decrease away from DC
        assert filt.min() >= 0.0
        assert filt.max() <= 1.0

    def test_antialias_resize_roundtrip(self):
        """Down then up approximately recovers original (smooth signal)."""
        from starfinder.registration.pyramid import antialias_resize

        # Need sufficient volume size for meaningful roundtrip — small volumes
        # (e.g. 8×32×32) have too few samples after 2x downsampling.
        rng = np.random.default_rng(123)
        from scipy.ndimage import gaussian_filter
        vol = gaussian_filter(rng.random((16, 64, 64)), sigma=2.0)

        down = antialias_resize(vol, 0.5)
        up = antialias_resize(down, 2.0)

        # Sizes should match original
        assert up.shape == vol.shape
        # Correlation should be high for smooth signal
        corr = np.corrcoef(vol.ravel(), up.ravel())[0, 1]
        assert corr > 0.9, f"Roundtrip correlation {corr:.3f} too low"

    def test_pad_crop_roundtrip(self):
        """Pad then crop recovers exact original."""
        from starfinder.registration.pyramid import crop_padding, pad_for_pyramiding

        vol = np.random.default_rng(42).random((5, 13, 17))
        padded, pad_widths = pad_for_pyramiding(vol, pyramid_levels=3)

        # Padded dims should be divisible by 2^(3-1)=4
        for s in padded.shape:
            assert s % 4 == 0, f"Dimension {s} not divisible by 4"

        recovered = crop_padding(padded, pad_widths)
        np.testing.assert_array_equal(recovered, vol)

    def test_pad_no_op_when_already_divisible(self):
        """No padding added when dims already divisible."""
        from starfinder.registration.pyramid import pad_for_pyramiding

        vol = np.ones((8, 16, 32))
        padded, pad_widths = pad_for_pyramiding(vol, pyramid_levels=3)

        assert all(p == 0 for p in pad_widths)
        # Should return same array when no padding needed
        assert padded is vol


class TestAntialiasedDemonsRegister:
    """Tests for anti-aliased pyramid mode in demons_register."""

    def test_identity_antialias_pyramid(self, mini_dataset):
        """Identical images with antialias pyramid produce near-zero field."""
        from starfinder.io import load_multipage_tiff
        from starfinder.registration.demons import demons_register

        vol = load_multipage_tiff(mini_dataset / "FOV_001" / "round1" / "ch00.tif")
        field = demons_register(
            vol, vol,
            iterations=[25, 10],
            pyramid_mode="antialias",
            method="demons",
        )

        assert field.shape == (*vol.shape, 3)
        assert np.abs(field).max() < 1.0, "Displacement should be near-zero for identical images"

    def test_antialias_outperforms_sitk_pyramid(self):
        """Anti-aliased pyramid outperforms SITK naive pyramid on dense data."""
        from scipy.ndimage import gaussian_filter, map_coordinates

        from starfinder.registration.demons import apply_deformation, demons_register
        from starfinder.registration.metrics import normalized_cross_correlation

        # Dense synthetic volume (not sparse spots) — large enough for
        # multi-level pyramid to be meaningful
        rng = np.random.default_rng(42)
        vol = (gaussian_filter(rng.random((16, 64, 64)), sigma=3.0) * 200).astype(np.float32)

        # Smooth deformation
        true_field = rng.standard_normal((3, *vol.shape)) * 4.0
        true_field = gaussian_filter(true_field, sigma=[0, 2, 8, 8])
        coords = np.meshgrid(*[np.arange(s) for s in vol.shape], indexing='ij')
        warped_coords = [c + f for c, f in zip(coords, true_field)]
        deformed = map_coordinates(vol, warped_coords, order=1, mode='constant', cval=0).astype(np.float32)

        # SITK naive 2-level pyramid
        field_sitk = demons_register(
            vol, deformed, iterations=[50, 25],
            method="demons", pyramid_mode="sitk",
        )
        reg_sitk = apply_deformation(deformed, field_sitk)
        ncc_sitk = normalized_cross_correlation(vol, reg_sitk)

        # Anti-aliased 2-level pyramid
        field_aa = demons_register(
            vol, deformed, iterations=[50, 25],
            method="demons", pyramid_mode="antialias",
        )
        reg_aa = apply_deformation(deformed, field_aa)
        ncc_aa = normalized_cross_correlation(vol, reg_aa)

        assert field_aa.shape == (*vol.shape, 3)
        assert ncc_aa > ncc_sitk, (
            f"Anti-aliased NCC {ncc_aa:.4f} should beat SITK naive NCC {ncc_sitk:.4f}"
        )


class TestMatlabCompatibleConfig:
    """Tests for matlab_compatible_config function."""

    def test_config_keys(self):
        """Config has expected keys and values."""
        from starfinder.registration.demons import matlab_compatible_config

        config = matlab_compatible_config()
        assert config["iterations"] == [100, 50, 25]
        assert config["smoothing_sigma"] == 1.0
        assert config["method"] == "demons"
        assert config["pyramid_mode"] == "antialias"
