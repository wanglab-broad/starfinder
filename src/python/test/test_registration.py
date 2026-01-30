"""Tests for starfinder.registration module."""

import numpy as np
import pytest

from starfinder.registration import (
    phase_correlate,
    apply_shift,
    register_volume,
    phase_correlate_skimage,
)


class TestPhaseCorrelate:
    """Tests for phase_correlate function."""

    def test_zero_shift(self, mini_dataset):
        """Identical images return (0, 0, 0)."""
        from starfinder.io import load_multipage_tiff

        vol = load_multipage_tiff(mini_dataset / "FOV_001" / "round1" / "ch00.tif")
        shift = phase_correlate(vol, vol)

        assert np.allclose(shift, (0, 0, 0), atol=0.1)

    def test_known_shift(self, mini_dataset):
        """Recovers integer shift applied via np.roll."""
        from starfinder.io import load_multipage_tiff

        vol = load_multipage_tiff(mini_dataset / "FOV_001" / "round1" / "ch00.tif")
        moved = np.roll(vol, (2, -3, 5), axis=(0, 1, 2))
        shift = phase_correlate(vol, moved)

        assert np.allclose(shift, (2, -3, 5), atol=0.5)


class TestApplyShift:
    """Tests for apply_shift function."""

    def test_roundtrip(self, mini_dataset):
        """shift -> apply -> inverse shift preserves non-zero data."""
        from starfinder.io import load_multipage_tiff

        vol = load_multipage_tiff(mini_dataset / "FOV_001" / "round1" / "ch00.tif")
        original_sum = vol.sum()

        shifted = apply_shift(vol, (3, -2, 4))
        restored = apply_shift(shifted, (-3, 2, -4))

        # Restored should have some data (not all zeroed out)
        assert restored.sum() > 0
        # Shape preserved
        assert restored.shape == vol.shape


class TestRegisterVolume:
    """Tests for register_volume function."""

    def test_registers_multichannel(self, mini_dataset):
        """Registers all channels and returns shifts."""
        from starfinder.io import load_image_stacks

        images, _ = load_image_stacks(
            mini_dataset / "FOV_001" / "round1",
            ["ch00", "ch01", "ch02", "ch03"],
        )

        # Create shifted version
        shifted = np.roll(images, (2, -3, 5, 0), axis=(0, 1, 2, 3))

        # Use ch00 as ref/mov
        ref_img = images[:, :, :, 0]
        mov_img = shifted[:, :, :, 0]

        registered, shifts = register_volume(shifted, ref_img, mov_img)

        assert registered.shape == images.shape
        assert np.allclose(shifts, (2, -3, 5), atol=0.5)


class TestBackendParity:
    """NumPy vs scikit-image produce same results."""

    def test_backends_match(self, mini_dataset):
        """Both backends return same shift for same input."""
        from starfinder.io import load_multipage_tiff

        vol = load_multipage_tiff(mini_dataset / "FOV_001" / "round1" / "ch00.tif")
        moved = np.roll(vol, (2, 3, -1), axis=(0, 1, 2))

        shift_np = phase_correlate(vol, moved)
        shift_sk = phase_correlate_skimage(vol, moved)

        assert np.allclose(shift_np, shift_sk, atol=0.5)
