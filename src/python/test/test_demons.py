"""Tests for starfinder.registration.demons module."""

from starfinder.image import ImageMetadata
from starfinder.registration import estimate_transform, apply_transform, DemonsConfig, DenseDisplacementTransform, WarpConfig
from starfinder.io import ImageLoadConfig

import numpy as np
import pytest

pytestmark = pytest.mark.registration

# Skip all tests if SimpleITK not installed
sitk = pytest.importorskip("SimpleITK", reason="SimpleITK required for local registration")


class TestDemonsRegister:
    """Tests for demons_register function."""

    def test_identity(self, small_dataset):
        """Identical images produce near-zero displacement field."""
        from starfinder.io import load_volume
        from starfinder.registration._demons import demons_register

        vol = load_volume(small_dataset / "FOV_001" / "round1" / "ch00.tif").image
        # Use fast single-level config for unit test speed
        field = demons_register(vol, vol, iterations=[25], pyramid_mode="sitk")

        # Shape should be (Z, Y, X, 3) for displacement vectors
        assert field.shape == (*vol.shape, 3)
        # Displacement should be near zero for identical images
        assert np.abs(field).max() < 1.0, "Displacement should be near-zero for identical images"

    def test_known_deformation(self, small_dataset):
        """Recovers direction of synthetic smooth deformation."""
        from scipy.ndimage import gaussian_filter, map_coordinates

        from starfinder.io import load_volume
        from starfinder.registration._demons import demons_register

        vol = load_volume(small_dataset / "FOV_001" / "round1" / "ch00.tif").image

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

        # Recover displacement field (fast single-level for unit test speed)
        estimated_field = demons_register(
            vol, deformed.astype(np.float32),
            iterations=[25], pyramid_mode="sitk",
        )

        # The estimated field should have similar direction to true field
        # (we check correlation, not exact match, since demons is iterative)
        true_magnitude = np.linalg.norm(true_field.transpose(1, 2, 3, 0), axis=-1)
        est_magnitude = np.linalg.norm(estimated_field, axis=-1)

        # Both should have deformation in similar regions (use relative threshold)
        mask = true_magnitude > np.percentile(true_magnitude, 75)
        assert mask.any(), "True deformation field should have non-trivial regions"
        assert est_magnitude[mask].mean() > 0.1, "Should detect deformation in warped regions"


class TestApplyDeformation:
    """Tests for apply_deformation function."""

    def test_identity_field(self, small_dataset):
        """Zero displacement field returns original volume."""
        from starfinder.io import load_volume

        vol = load_volume(small_dataset / "FOV_001" / "round1" / "ch00.tif").image

        # Zero displacement field
        field = np.zeros((*vol.shape, 3), dtype=np.float32)
        result = apply_transform(vol, DenseDisplacementTransform(field, vol.shape, vol.shape, ImageMetadata("test/reference"), ImageMetadata("test/moving")), config=WarpConfig(backend="simpleitk"))

        assert result.shape == vol.shape
        # Should be nearly identical (interpolation may introduce tiny differences)
        np.testing.assert_allclose(result, vol, rtol=1e-4, atol=1e-4)


class TestRegisterVolumeLocal:
    """Tests for register_volume_local function."""

    def test_multichannel(self, small_dataset):
        """Registers all channels using computed field."""
        from starfinder.io import load_round

        loaded_round = load_round(small_dataset / "FOV_001" / "round1", config=ImageLoadConfig(channel_labels=tuple(["ch00", "ch01", "ch02", "ch03"])))
        images = loaded_round.image
        _ = loaded_round.diagnostics

        # Use ch00 as ref/mov (identity case)
        ref_img = images[:, :, :, 0]
        mov_img = images[:, :, :, 0]

        # Use fast single-level config for unit test speed
        _registration = estimate_transform(ref_img, mov_img, config=DemonsConfig(iterations=[25], pyramid_mode='sitk'), reference_metadata=ImageMetadata("test/reference"), moving_metadata=ImageMetadata("test/moving"))
        registered = apply_transform(images, _registration.transform, config=_registration.application_config)
        field = _registration.transform.displacement_zyx

        assert registered.shape == images.shape
        assert field.shape == (*images.shape[:3], 3)
        # For identity case, registered should be similar to input
        np.testing.assert_allclose(registered, images, rtol=0.1, atol=1.0)


class TestPyramidUtilities:
    """Tests for pyramid.py anti-aliased multi-resolution utilities."""

    def test_butterworth_filter_shape(self):
        """Butterworth filter has correct shape and peak at DC."""
        from starfinder.registration._pyramid import butterworth_3d

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
        from starfinder.registration._pyramid import antialias_resize

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
        from starfinder.registration._pyramid import crop_padding
        from starfinder.registration._pyramid import pad_for_pyramiding

        vol = np.random.default_rng(42).random((5, 13, 17))
        padded, pad_widths = pad_for_pyramiding(vol, pyramid_levels=3)

        # Padded dims should be divisible by 2^(3-1)=4
        for s in padded.shape:
            assert s % 4 == 0, f"Dimension {s} not divisible by 4"

        recovered = crop_padding(padded, pad_widths)
        np.testing.assert_array_equal(recovered, vol)

    def test_pad_no_op_when_already_divisible(self):
        """No padding added when dims already divisible."""
        from starfinder.registration._pyramid import pad_for_pyramiding

        vol = np.ones((8, 16, 32))
        padded, pad_widths = pad_for_pyramiding(vol, pyramid_levels=3)

        assert all(p == 0 for p in pad_widths)
        # Should return same array when no padding needed
        assert padded is vol


class TestAntialiasedDemonsRegister:
    """Tests for anti-aliased pyramid mode in demons_register."""

    def test_identity_antialias_pyramid(self, small_dataset):
        """Identical images with antialias pyramid produce near-zero field."""
        from starfinder.io import load_volume
        from starfinder.registration._demons import demons_register

        vol = load_volume(small_dataset / "FOV_001" / "round1" / "ch00.tif").image
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

        from starfinder.registration._demons import demons_register
        from starfinder.evaluation.registration import normalized_cross_correlation

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
        reg_sitk = apply_transform(deformed, DenseDisplacementTransform(field_sitk, deformed.shape, deformed.shape, ImageMetadata("test/reference"), ImageMetadata("test/moving")), config=WarpConfig(backend="simpleitk"))
        ncc_sitk = normalized_cross_correlation(vol, reg_sitk).values["ncc"]

        # Anti-aliased 2-level pyramid
        field_aa = demons_register(
            vol, deformed, iterations=[50, 25],
            method="demons", pyramid_mode="antialias",
        )
        reg_aa = apply_transform(deformed, DenseDisplacementTransform(field_aa, deformed.shape, deformed.shape, ImageMetadata("test/reference"), ImageMetadata("test/moving")), config=WarpConfig(backend="simpleitk"))
        ncc_aa = normalized_cross_correlation(vol, reg_aa).values["ncc"]

        assert field_aa.shape == (*vol.shape, 3)
        assert ncc_aa > ncc_sitk, (
            f"Anti-aliased NCC {ncc_aa:.4f} should beat SITK naive NCC {ncc_sitk:.4f}"
        )


def _one_iteration(fixed, moving, smoothing_sigma=1.0):
    """The demons field after one iteration from a zero field, through DemonsConfig."""
    registration = estimate_transform(
        fixed, moving, config=DemonsConfig(iterations=(1,), smoothing_sigma=smoothing_sigma),
        reference_metadata=ImageMetadata("test/reference"), moving_metadata=ImageMetadata("test/moving"))
    return registration.transform.displacement_zyx


# MATLAB's fspecial('gaussian', [1 7], 1): the sampled Gaussian exp(-x^2 / 2), x = -3..3, divided by its sum.
MATLAB_KERNEL_SIGMA_1 = np.array([0.004433048175243745, 0.054005582622414484, 0.2420362293761143,
                                  0.3990502796524549, 0.2420362293761143, 0.054005582622414484,
                                  0.004433048175243745])
FLOAT64 = {"rtol": 1e-9, "atol": 1e-12}


class TestMatlabOrderUpdate:
    """W-835: the classic variant smooths after each update with MATLAB's kernel, keeping SimpleITK's boundary rules."""

    def test_one_iteration_returns_the_smoothed_update(self):
        """From a zero field, one iteration returns S(u), not the raw Thirion update u."""
        from scipy.ndimage import correlate1d

        rng = np.random.default_rng(835)
        fixed = (100 + 20 * rng.standard_normal((12, 24, 24))).astype(np.float32)
        moving = (100 + 20 * rng.standard_normal((12, 24, 24))).astype(np.float32)
        field = _one_iteration(fixed, moving)

        f, m = fixed.astype(np.float64), moving.astype(np.float64)
        gradient = np.gradient(f)  # central inside; 0 on the two faces normal to each axis, as SimpleITK's
        for axis, component in enumerate(gradient):
            faces = [slice(None)] * 3
            faces[axis] = [0, -1]
            component[tuple(faces)] = 0.0
        diff = f - m
        denominator = sum(g * g for g in gradient) + diff ** 2
        scale = np.where((np.abs(diff) < 0.001) | (denominator < 1e-9), 0.0, diff / denominator)
        raw = np.stack([scale * g for g in gradient], axis=-1)
        smoothed = raw.copy()
        for axis in range(3):
            smoothed = correlate1d(smoothed, MATLAB_KERNEL_SIGMA_1, axis=axis, mode="nearest")

        assert field.shape == (12, 24, 24, 3) and field.dtype == np.float64
        np.testing.assert_allclose(field, smoothed, **FLOAT64)
        assert np.abs(field - raw).max() > 0.1

    def test_kernel_is_matlabs_sampled_gaussian(self):
        """A one-voxel update spreads as the outer product of MATLAB's 7-tap kernel for sigma = 1."""
        z, y, x = np.indices((15, 15, 15), dtype=np.float64)
        fixed = (100 + 2 * x).astype(np.float32)  # dF/dx = 2 inside, 0 on the X faces
        moving = fixed.copy()
        moving[7, 7, 7] -= 1  # F - M = 1 at the centre only
        field = _one_iteration(fixed, moving)

        update = 1 * 2 / (2 ** 2 + 1 ** 2)  # (F - M) g / (g^2 + (F - M)^2) along X
        expected = np.zeros_like(field)
        expected[4:11, 4:11, 4:11, 2] = update * np.einsum(
            "i,j,k->ijk", MATLAB_KERNEL_SIGMA_1, MATLAB_KERNEL_SIGMA_1, MATLAB_KERNEL_SIGMA_1)
        np.testing.assert_allclose(field, expected, **FLOAT64)

    def test_face_gradient_is_zero(self):
        """On a face, the gradient component normal to it is 0, as SimpleITK's (MATLAB's is one-sided)."""
        z, _, x = np.indices((8, 16, 16), dtype=np.float64)
        fixed = (100 + 10 * z ** 2 + 2 * x).astype(np.float32)
        moving = fixed.copy()
        moving[0, 8, 8] -= 1  # F - M = 1 on the Z = 0 face (one-sided dF/dz there: 10)
        moving[7, 4, 4] -= 1  # and on the Z = 7 face (one-sided dF/dz there: 130)
        field = _one_iteration(fixed, moving, smoothing_sigma=0.0)

        expected = np.zeros_like(field)
        expected[0, 8, 8, 2] = expected[7, 4, 4, 2] = 1 * 2 / (0 ** 2 + 2 ** 2 + 1 ** 2)  # gradient (0, 0, 2)
        np.testing.assert_allclose(field, expected, **FLOAT64)

    def test_a_sample_half_a_voxel_outside_takes_the_edge_value(self):
        """A sample within half a voxel outside the grid counts as inside and takes the edge voxel's value."""
        from starfinder.registration._demons import _create_demons_filter

        x = np.indices((8, 16, 16), dtype=np.float64)[2]
        fixed = (100 + 2 * x).astype(np.float32)
        moving = fixed.copy()
        moving[0, 8, 8] -= 1
        moving[0, 4, 4] -= 1
        initial = np.zeros((8, 16, 16, 3))  # SimpleITK components (x, y, z)
        initial[0, 8, 8, 2] = -0.3  # samples Z = -0.3: inside, the edge voxel's value M[0, 8, 8] = F - 1
        initial[0, 4, 4, 2] = -0.7  # samples Z = -0.7: outside, no update
        fixed_image, moving_image = sitk.GetImageFromArray(fixed), sitk.GetImageFromArray(moving)
        initial_image = sitk.GetImageFromArray(initial, isVector=True)
        initial_image.CopyInformation(fixed_image)
        demons = _create_demons_filter(sitk, "demons", 0.0)
        demons.SetNumberOfIterations(1)
        field = sitk.GetArrayFromImage(demons.Execute(fixed_image, moving_image, initial_image))

        expected = initial.copy()
        expected[0, 8, 8, 0] = 1 * 2 / (2 ** 2 + 1 ** 2)  # F - M = 1 (not 1.3, as extrapolation would give)
        np.testing.assert_allclose(field, expected, **FLOAT64)


class TestMatlabCompatibleConfig:
    """Tests for matlab_compatible_config function."""

    def test_config_keys(self):
        """Config has expected keys and values."""

        config = DemonsConfig()
        assert config.iterations == (100, 50, 25)
        assert config.smoothing_sigma == 1.0
        assert config.variant == "demons"
        assert config.pyramid_mode == "antialias"
