"""Tests for starfinder.registration.pointset (TPS and CPD local registration)."""

import numpy as np
import pytest
from scipy.ndimage import map_coordinates


class TestTPSIdentity:
    """Identical images should produce near-zero displacement."""

    def test_identity_no_displacement(self, small_dataset):
        from starfinder.io import load_multipage_tiff
        from starfinder.registration.pointset import tps_register

        vol = load_multipage_tiff(
            small_dataset / "FOV_001" / "round1" / "ch00.tif"
        )

        field = tps_register(
            vol, vol,
            detection_threshold=2.0,
            min_matches=10,
            smoothing=5.0,
        )

        assert field.shape == (*vol.shape, 3)
        assert field.dtype == np.float32
        # Displacement should be near-zero for identical images
        assert np.abs(field).mean() < 1.0, (
            f"Mean displacement {np.abs(field).mean():.3f} too large for identity"
        )


class TestTPSKnownDeformation:
    """TPS should recover a known smooth deformation."""

    def test_known_local_deformation(self):
        # Create a synthetic volume with bright spots on a dark background
        rng = np.random.default_rng(42)
        shape = (16, 128, 128)
        vol = np.zeros(shape, dtype=np.float32)

        # Place ~200 bright spots (enough for matching)
        n_spots = 200
        zz = rng.integers(2, shape[0] - 2, n_spots)
        yy = rng.integers(10, shape[1] - 10, n_spots)
        xx = rng.integers(10, shape[2] - 10, n_spots)
        for z, y, x in zip(zz, yy, xx):
            vol[z, max(0, y - 1):y + 2, max(0, x - 1):x + 2] = 200.0

        # Apply a known smooth polynomial deformation (YX only, no Z)
        coords = np.meshgrid(
            np.arange(shape[0]),
            np.arange(shape[1]),
            np.arange(shape[2]),
            indexing="ij",
        )
        # Quadratic deformation: max ~5px displacement at edges
        cy, cx = shape[1] / 2, shape[2] / 2
        dy_field = 5.0 * ((coords[1] - cy) / cy) ** 2
        dx_field = 3.0 * ((coords[2] - cx) / cx) ** 2
        dz_field = np.zeros_like(dy_field)

        warped_coords = [
            coords[0] + dz_field,
            coords[1] + dy_field,
            coords[2] + dx_field,
        ]
        deformed = map_coordinates(vol, warped_coords, order=1, mode="constant", cval=0)
        deformed = deformed.astype(np.float32)

        from starfinder.registration.pointset import (
            apply_tps_deformation,
            tps_register,
        )

        field = tps_register(
            vol, deformed,
            detection_threshold=2.0,
            match_distance=15.0,
            min_matches=20,
            smoothing=0.5,
            grid_spacing=8,
        )

        # Apply recovered field to deformed → should approximate original
        recovered = apply_tps_deformation(deformed, field)

        # Check in the region where spots exist (not edges)
        inner = (slice(2, -2), slice(20, -20), slice(20, -20))
        # Spots in the original should approximately match spots in the recovered
        mask = vol[inner] > 100
        if mask.any():
            diff = np.abs(vol[inner].astype(float) - recovered[inner].astype(float))
            mean_error = diff[mask].mean()
            assert mean_error < 100.0, (
                f"Mean spot error {mean_error:.1f} too large (expected <100)"
            )


class TestTPSTooFewSpots:
    """ValueError when insufficient matches."""

    def test_too_few_spots_raises(self):
        from starfinder.registration.pointset import tps_register

        # Uniform dark volume — no spots to detect
        vol = np.ones((8, 32, 32), dtype=np.float32) * 10.0

        with pytest.raises(ValueError, match="spot"):
            tps_register(vol, vol, min_matches=50)


class TestRegisterVolumeTPS:
    """register_volume_tps should produce correct output shape."""

    def test_register_volume_tps_shape(self, small_dataset):
        from starfinder.io import load_image_stacks
        from starfinder.registration.pointset import register_volume_tps

        images, _ = load_image_stacks(
            small_dataset / "FOV_001" / "round1",
            ["ch00", "ch01", "ch02", "ch03"],
        )

        ref_3d = images[:, :, :, 0]

        registered, field = register_volume_tps(
            images, ref_3d, ref_3d,
            detection_threshold=2.0,
            min_matches=10,
            smoothing=5.0,
        )

        assert registered.shape == images.shape
        assert registered.dtype == images.dtype
        assert field.shape == (*images.shape[:3], 3)
        assert field.dtype == np.float32


# ─── CPD Tests ───────────────────────────────────────────────────────────


class TestCPDRegistration:
    """Tests for Coherent Point Drift registration."""

    def test_cpd_nonrigid_identity(self):
        """Identical point clouds should produce near-zero displacements."""
        from starfinder.registration.pointset import cpd_nonrigid

        rng = np.random.default_rng(42)
        points = rng.uniform(0, 100, (50, 3))

        T, W, G = cpd_nonrigid(points, points.copy(), beta=10.0, lmbda=2.0)

        # W should be near-zero → T ≈ Y
        rms_w = np.sqrt(np.mean(W**2))
        assert rms_w < 1.0, f"RMS(W) = {rms_w:.4f}, expected near 0"
        rms_disp = np.sqrt(np.mean((T - points) ** 2))
        assert rms_disp < 1.0, f"RMS displacement = {rms_disp:.4f}, expected near 0"

    def test_cpd_affine_recovery(self):
        """Affine CPD should recover a known affine transform."""
        from starfinder.registration.pointset import cpd_affine

        rng = np.random.default_rng(123)
        Y = rng.uniform(10, 90, (80, 3))

        # Known affine: small rotation + translation
        B_true = np.array([
            [1.0, 0.05, 0.0],
            [-0.05, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        t_true = np.array([2.0, -3.0, 1.0])
        X = Y @ B_true.T + t_true

        T, B_est, t_est = cpd_affine(X, Y, w=0.0, max_iter=200)

        # Check recovered transform
        rms_pos = np.sqrt(np.mean((T - X) ** 2))
        assert rms_pos < 2.0, (
            f"RMS position error = {rms_pos:.4f}, expected < 2.0"
        )
        b_err = np.abs(B_est - B_true).max()
        assert b_err < 0.1, f"Max B error = {b_err:.4f}, expected < 0.1"
        t_err = np.abs(t_est - t_true).max()
        assert t_err < 2.0, f"Max t error = {t_err:.4f}, expected < 2.0"

    def test_cpd_nonrigid_polynomial(self):
        """Non-rigid CPD should reduce error for polynomial deformation."""
        from starfinder.registration.pointset import cpd_nonrigid

        rng = np.random.default_rng(99)
        # Generate points in a 100×100×16 space
        Y = np.column_stack([
            rng.uniform(2, 14, 100),
            rng.uniform(10, 90, 100),
            rng.uniform(10, 90, 100),
        ])

        # Apply polynomial deformation in YX
        cy, cx = 50.0, 50.0
        dy = 5.0 * ((Y[:, 1] - cy) / cy) ** 2
        dx = 3.0 * ((Y[:, 2] - cx) / cx) ** 2
        X = Y.copy()
        X[:, 1] += dy
        X[:, 2] += dx

        rms_before = np.sqrt(np.mean((Y - X) ** 2))
        T, W, G = cpd_nonrigid(X, Y, beta=15.0, lmbda=1.0, w=0.0, max_iter=200)
        rms_after = np.sqrt(np.mean((T - X) ** 2))

        # CPD should reduce RMS error by > 50%
        assert rms_after < rms_before * 0.5, (
            f"RMS after ({rms_after:.3f}) not < 50% of before ({rms_before:.3f})"
        )

    def test_cpd_register_improves_ncc(self):
        """Full CPD pipeline should improve NCC on deformed synthetic volume."""
        from starfinder.registration.metrics import normalized_cross_correlation
        from starfinder.registration.pointset import (
            apply_tps_deformation,
            cpd_register,
        )

        rng = np.random.default_rng(42)
        shape = (16, 128, 128)
        vol = np.zeros(shape, dtype=np.float32)

        # Place bright spots
        n_spots = 200
        zz = rng.integers(2, shape[0] - 2, n_spots)
        yy = rng.integers(10, shape[1] - 10, n_spots)
        xx = rng.integers(10, shape[2] - 10, n_spots)
        for z, y, x in zip(zz, yy, xx):
            vol[z, max(0, y - 1):y + 2, max(0, x - 1):x + 2] = 200.0

        # Apply polynomial deformation
        coords = np.meshgrid(
            np.arange(shape[0]),
            np.arange(shape[1]),
            np.arange(shape[2]),
            indexing="ij",
        )
        cy, cx = shape[1] / 2, shape[2] / 2
        dy_field = 4.0 * ((coords[1] - cy) / cy) ** 2
        dx_field = 3.0 * ((coords[2] - cx) / cx) ** 2
        warped_coords = [coords[0], coords[1] + dy_field, coords[2] + dx_field]
        deformed = map_coordinates(
            vol, warped_coords, order=1, mode="constant", cval=0
        ).astype(np.float32)

        ncc_before = normalized_cross_correlation(vol, deformed)

        field = cpd_register(
            vol, deformed,
            detection_threshold=2.0,
            grid_spacing=8,
        )
        recovered = apply_tps_deformation(deformed, field)
        ncc_after = normalized_cross_correlation(vol, recovered)

        assert ncc_after > ncc_before, (
            f"NCC did not improve: {ncc_before:.4f} → {ncc_after:.4f}"
        )

    def test_cpd_register_too_few_spots(self):
        """CPD should raise ValueError on volume with too few spots."""
        from starfinder.registration.pointset import cpd_register

        vol = np.ones((8, 32, 32), dtype=np.float32) * 10.0

        with pytest.raises(ValueError, match="Too few spots"):
            cpd_register(vol, vol)


# ─── Sanitize / Boundary Mode / Zoom Order Tests ────────────────────────


class TestSanitizeDisplacementField:
    """Tests for sanitize_displacement_field."""

    def test_clamp_prevents_oob(self):
        """Clamping should ensure position + displacement stays in [0, dim-1]."""
        from starfinder.registration.pointset import sanitize_displacement_field

        shape = (8, 32, 32)
        field = np.zeros((*shape, 3), dtype=np.float32)
        # Push everything 50px in Y — way beyond the volume
        field[..., 1] = 50.0

        sanitized = sanitize_displacement_field(field, clamp=True)

        # After clamping: position + displacement should be in [0, 31]
        yy = np.arange(shape[1]).reshape(1, -1, 1).astype(np.float32)
        src_y = yy + sanitized[..., 1]
        assert src_y.min() >= 0.0
        assert src_y.max() <= shape[1] - 1

    def test_clamp_with_margin(self):
        """Margin should tighten the valid range."""
        from starfinder.registration.pointset import sanitize_displacement_field

        shape = (4, 16, 16)
        field = np.zeros((*shape, 3), dtype=np.float32)
        field[..., 2] = -20.0  # push X far negative

        sanitized = sanitize_displacement_field(field, clamp=True, margin=2)

        xx = np.arange(shape[2]).reshape(1, 1, -1).astype(np.float32)
        src_x = xx + sanitized[..., 2]
        assert src_x.min() >= 2.0
        assert src_x.max() <= shape[2] - 1 - 2

    def test_smooth_reduces_gradient(self):
        """Smoothing should reduce the max gradient of the field."""
        from starfinder.registration.pointset import sanitize_displacement_field

        shape = (4, 64, 64)
        field = np.zeros((*shape, 3), dtype=np.float32)
        # Sharp step: half of Y-axis has +5px displacement
        field[:, 32:, :, 1] = 5.0

        smoothed = sanitize_displacement_field(
            field, clamp=False, smooth_sigma=2.0
        )

        # Gradient across the step should be smaller after smoothing
        grad_orig = np.abs(np.diff(field[:, :, :, 1], axis=1)).max()
        grad_smooth = np.abs(np.diff(smoothed[:, :, :, 1], axis=1)).max()
        assert grad_smooth < grad_orig, (
            f"Smoothing did not reduce gradient: {grad_orig:.2f} → {grad_smooth:.2f}"
        )

    def test_fold_detection_finds_known_fold(self):
        """Fold detection should flag regions with crossing displacements."""
        from starfinder.registration.pointset import sanitize_displacement_field

        shape = (4, 32, 32)
        field = np.zeros((*shape, 3), dtype=np.float32)
        # Create a fold: displacement gradient in Y exceeds -1.
        # d_y(y) = 40*(0.5 - y/31), so ∂d_y/∂y = -40/31 ≈ -1.29.
        # J_yy = 1 + (-1.29) = -0.29 < 0 → fold everywhere.
        for y in range(32):
            field[:, y, :, 1] = 40.0 * (0.5 - y / 31.0)

        _, fold_mask = sanitize_displacement_field(
            field, clamp=False, detect_folds=True
        )

        assert fold_mask.shape == shape
        assert fold_mask.any(), "Expected folds but none detected"

    def test_no_fold_on_zero_field(self):
        """Zero displacement should have no folds."""
        from starfinder.registration.pointset import sanitize_displacement_field

        field = np.zeros((4, 16, 16, 3), dtype=np.float32)
        _, fold_mask = sanitize_displacement_field(
            field, clamp=False, detect_folds=True
        )
        assert not fold_mask.any(), "No folds expected on zero field"

    def test_original_not_modified(self):
        """sanitize_displacement_field should not modify the input array."""
        from starfinder.registration.pointset import sanitize_displacement_field

        field = np.full((4, 16, 16, 3), 50.0, dtype=np.float32)
        original = field.copy()

        sanitize_displacement_field(field, clamp=True)
        np.testing.assert_array_equal(field, original)


class TestBoundaryModeNearest:
    """boundary_mode='nearest' should eliminate black bands."""

    def test_nearest_no_black_band(self):
        """Shifting a bright volume should not create zero-filled edges."""
        from starfinder.registration.pointset import apply_tps_deformation

        # Uniform bright volume
        shape = (8, 32, 32)
        vol = np.full(shape, 100.0, dtype=np.float32)

        # Displacement field: src_y = y + 10, so rows y=22..31 map to
        # source y=32..41 (OOB). With constant mode → black band at bottom.
        field = np.zeros((*shape, 3), dtype=np.float32)
        field[..., 1] = 10.0

        # With constant mode: bottom rows (y >= 22) should be 0
        result_const = apply_tps_deformation(vol, field, boundary_mode="constant")
        assert (result_const[:, -5:, :] == 0).all(), (
            "Expected black band at bottom with constant boundary"
        )

        # With nearest mode: no zeros
        result_nearest = apply_tps_deformation(vol, field, boundary_mode="nearest")
        assert (result_nearest > 0).all(), (
            "Expected no black band with nearest boundary"
        )

    def test_nearest_preserves_dtype(self):
        """Result dtype should match input regardless of boundary_mode."""
        from starfinder.registration.pointset import apply_tps_deformation

        vol = np.full((4, 16, 16), 50, dtype=np.uint8)
        field = np.zeros((4, 16, 16, 3), dtype=np.float32)

        result = apply_tps_deformation(vol, field, boundary_mode="nearest")
        assert result.dtype == np.uint8


class TestZoomOrder:
    """zoom_order parameter controls spline interpolation for field upsampling."""

    def test_zoom_order_1_produces_valid_field(self):
        """Linear zoom should produce a valid displacement field."""
        from starfinder.registration.pointset import tps_displacement_field

        rng = np.random.default_rng(42)
        n_pts = 30
        positions = np.column_stack([
            rng.uniform(2, 14, n_pts),
            rng.uniform(10, 118, n_pts),
            rng.uniform(10, 118, n_pts),
        ])
        displacements = rng.uniform(-3, 3, (n_pts, 3)).astype(np.float32)
        shape = (16, 128, 128)

        field_cubic = tps_displacement_field(
            positions, displacements, shape,
            smoothing=1.0, grid_spacing=16, zoom_order=3,
        )
        field_linear = tps_displacement_field(
            positions, displacements, shape,
            smoothing=1.0, grid_spacing=16, zoom_order=1,
        )

        assert field_cubic.shape == field_linear.shape
        assert field_linear.dtype == np.float32
        # Both should be similar — linear is just less smooth
        diff = np.abs(field_cubic - field_linear).mean()
        assert diff < 2.0, f"Mean difference {diff:.3f} too large"

    def test_field_smooth_sigma_smooths_field(self):
        """field_smooth_sigma should produce a smoother field."""
        from starfinder.registration.pointset import tps_displacement_field

        rng = np.random.default_rng(42)
        n_pts = 30
        positions = np.column_stack([
            rng.uniform(2, 14, n_pts),
            rng.uniform(10, 118, n_pts),
            rng.uniform(10, 118, n_pts),
        ])
        displacements = rng.uniform(-3, 3, (n_pts, 3)).astype(np.float32)
        shape = (16, 128, 128)

        field_raw = tps_displacement_field(
            positions, displacements, shape,
            smoothing=1.0, grid_spacing=16,
        )
        field_smooth = tps_displacement_field(
            positions, displacements, shape,
            smoothing=1.0, grid_spacing=16, field_smooth_sigma=2.0,
        )

        # Smoothed field should have smaller max gradient
        grad_raw = np.abs(np.diff(field_raw[..., 1], axis=1)).max()
        grad_smooth = np.abs(np.diff(field_smooth[..., 1], axis=1)).max()
        assert grad_smooth <= grad_raw, (
            f"Smoothed field gradient ({grad_smooth:.3f}) not <= raw ({grad_raw:.3f})"
        )
