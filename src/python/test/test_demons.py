"""Tests for starfinder.registration.demons module."""

from starfinder.image import ImageMetadata
from starfinder.registration import estimate_transform, apply_transform, DemonsConfig, DenseDisplacementTransform, WarpConfig
from starfinder.io import ImageLoadConfig

import numpy as np
import pytest
import SimpleITK as sitk

pytestmark = pytest.mark.registration


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


def _numpy_thirion_iterations(fixed, moving, field, iterations, kernel):
    """The NumPy and SciPy loop the fused kernels replaced (_demons._thirion_iterations at 76cd058), as a reference."""
    from scipy import ndimage

    ndim = fixed.ndim
    grad = []
    for axis, n in enumerate(fixed.shape):  # central inside, 0 on the two faces normal to each axis
        component = np.zeros_like(fixed)
        if n > 2:
            inner = [slice(None)] * ndim
            inner[axis] = slice(1, -1)
            ahead, behind = list(inner), list(inner)
            ahead[axis], behind[axis] = slice(2, None), slice(None, -2)
            component[tuple(inner)] = (fixed[tuple(ahead)] - fixed[tuple(behind)]) / 2
        grad.append(component)
    grad_sq = sum(g * g for g in grad)
    axes = [np.arange(n, dtype=np.float64).reshape([-1 if a == axis else 1 for a in range(ndim)])
            for axis, n in enumerate(fixed.shape)]
    field = field.copy()
    rms = 0.0
    for _ in range(iterations):
        coords = np.stack([field[axis] + axes[axis] for axis in range(ndim)])
        inside = np.all([(coords[axis] >= -0.5) & (coords[axis] < n - 0.5) for axis, n in enumerate(fixed.shape)],
                        axis=0)
        warped = ndimage.map_coordinates(moving, coords, order=1, mode="nearest", prefilter=False)
        diff = fixed - warped
        denominator = grad_sq + diff * diff
        scale = np.zeros_like(diff)
        np.divide(diff, denominator, out=scale, where=inside & (np.abs(diff) >= 0.001) & (denominator >= 1e-9))
        for axis in range(ndim):
            field[axis] += scale * grad[axis]
        count = np.count_nonzero(inside)
        rms = float(np.sqrt((scale * scale * grad_sq)[inside].sum() / count)) if count else 0.0
        for axis in range(1, ndim + 1):
            field = ndimage.correlate1d(field, kernel, axis=axis, mode="nearest")
    return field, rms


def _warped_pair(shape, seed):
    """A smooth random fixed image, the moving image shifted by a smooth field, and a start field of up to 1.5 voxels.

    The start field puts samples beyond the half-voxel band of the faces, inside it and inside the grid.
    """
    from scipy.ndimage import gaussian_filter, map_coordinates

    rng = np.random.default_rng(seed)
    fixed = 100 + 400 * gaussian_filter(rng.random(shape), 1.5)
    shift = np.stack([gaussian_filter(rng.standard_normal(shape), 3) * 8 for _ in shape])
    grid = np.indices(shape, dtype=np.float64)
    moving = map_coordinates(fixed, grid - shift, order=1, mode="nearest")
    start = np.stack([gaussian_filter(rng.standard_normal(shape), 2) * 3 for _ in shape]).clip(-1.5, 1.5)
    return fixed, moving, start


class TestFusedKernels:
    """W-882: the fused numba update gives the NumPy and SciPy loop's field and RMS change."""

    @pytest.mark.parametrize("shape, sigma", [((10, 20, 24), 1.0), ((10, 20, 24), 0.0), ((20, 24), 1.0)])
    def test_field_and_rms_equal_the_numpy_loop(self, shape, sigma):
        """Five iterations from a start field that samples outside the grid, inside the half-voxel band and inside."""
        from starfinder.registration._demons import _sampled_gaussian, _thirion_iterations

        fixed, moving, start = _warped_pair(shape, 882)
        kernel = _sampled_gaussian(sigma)
        expected, expected_rms = _numpy_thirion_iterations(fixed, moving, start, 5, kernel)
        fixed.setflags(write=False)  # read-only inputs, as SimpleITK's array views of float64 images
        moving.setflags(write=False)
        field, rms = _thirion_iterations(fixed, moving, start.copy(), 5, kernel)

        coords = start + np.indices(shape)
        outside = np.any([(c < -0.5) | (c >= n - 0.5) for c, n in zip(coords, shape)], axis=0)
        band = ~outside & np.any([(c < 0) | (c > n - 1) for c, n in zip(coords, shape)], axis=0)
        assert outside.any() and band.any()
        assert field.shape == (len(shape), *shape) and field.dtype == np.float64
        np.testing.assert_allclose(field, expected, **FLOAT64)
        assert rms == pytest.approx(expected_rms, rel=1e-9, abs=1e-12)
        assert np.abs(expected - start).max() > 0.1 and expected_rms > 0.01

    def test_no_voxel_inside_gives_rms_0_and_leaves_the_field(self):
        """Every sample outside the grid: no update, RMS 0, the field only smoothed (here not at all: sigma 0)."""
        from starfinder.registration._demons import _thirion_iterations

        fixed, moving, _ = _warped_pair((6, 8, 8), 1)
        start = np.full((3, 6, 8, 8), 20.0)
        field, rms = _thirion_iterations(fixed, moving, start.copy(), 2, np.ones(1))
        assert rms == 0.0
        np.testing.assert_array_equal(field, start)

    def test_without_numba_the_classic_variant_names_the_extra(self, monkeypatch):
        """numba missing: RegistrationBackendUnavailableError naming the local-registration extra."""
        import sys

        import starfinder.registration as registration
        from starfinder.registration._demons import _thirion_iterations
        from starfinder.registration._errors import RegistrationBackendUnavailableError

        monkeypatch.setitem(sys.modules, "numba", None)
        monkeypatch.delitem(sys.modules, "starfinder.registration._demons_kernels", raising=False)
        monkeypatch.delattr(registration, "_demons_kernels", raising=False)
        fixed, moving, start = _warped_pair((4, 8, 8), 2)
        with pytest.raises(RegistrationBackendUnavailableError, match=r"numba.*local-registration"):
            _thirion_iterations(fixed, moving, start, 1, np.ones(1))

    def test_kernels_compile_without_a_writable_cache(self, tmp_path):
        """A read-only install with no writable cache location (the container image): no cache, no error."""
        import os
        import shutil
        import subprocess
        import sys
        from pathlib import Path

        import starfinder.registration._demons_kernels as kernels

        package = tmp_path / "readonly"
        package.mkdir()
        shutil.copy(Path(kernels.__file__), package / "_demons_kernels.py")
        home = tmp_path / "home"
        home.mkdir()
        for folder in (package, home):
            folder.chmod(0o555)
        code = ("import numpy as np, _demons_kernels as k; "
                "f = np.zeros((3, 4, 5, 6)); i = np.arange(120.0).reshape(4, 5, 6); "
                "print(k.thirion_update_3d(i, i + 1, np.ones((3, 4, 5, 6)), np.full((4, 5, 6), 3.0), f), "
                "k.thirion_update_3d._cache.__class__.__name__)")
        env = {key: value for key, value in os.environ.items() if key not in ("NUMBA_CACHE_DIR", "XDG_CACHE_HOME")}
        env.update(HOME=str(home), PYTHONPATH=str(package), PYTHONDONTWRITEBYTECODE="1")
        try:
            done = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                                  cwd=package, timeout=600)
        finally:
            for folder in (package, home):
                folder.chmod(0o755)
        assert done.returncode == 0, done.stderr
        assert done.stdout.split() == ["(120,", "22.5)", "NullCache"]
        assert list(package.iterdir()) == [package / "_demons_kernels.py"]


class TestMatlabCompatibleConfig:
    """Tests for matlab_compatible_config function."""

    def test_config_keys(self):
        """Config has expected keys and values."""

        config = DemonsConfig()
        assert config.iterations == (100, 50, 25)
        assert config.smoothing_sigma == 1.0
        assert config.variant == "demons"
        assert config.pyramid_mode == "antialias"
