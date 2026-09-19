"""Tests for starfinder.registration module."""

from starfinder.image import ImageMetadata
from starfinder.registration import estimate_transform, apply_transform, TranslationConfig
from starfinder.io import ImageLoadConfig

import numpy as np
import pytest

from starfinder.registration._translation import phase_correlate
from starfinder.registration._translation import apply_shift
pass
from starfinder.registration._skimage_backend import phase_correlate_skimage


class TestPhaseCorrelate:
    """Tests for phase_correlate function."""

    def test_zero_shift(self, small_dataset):
        """Identical images return (0, 0, 0)."""
        from starfinder.io import load_volume

        vol = load_volume(small_dataset / "FOV_001" / "round1" / "ch00.tif").image
        shift = phase_correlate(vol, vol)

        assert np.allclose(shift, (0, 0, 0), atol=0.1)

    def test_known_shift(self, small_dataset):
        """Recovers integer shift applied via np.roll."""
        from starfinder.io import load_volume

        vol = load_volume(small_dataset / "FOV_001" / "round1" / "ch00.tif").image
        moved = np.roll(vol, (2, -3, 5), axis=(0, 1, 2))
        shift = phase_correlate(vol, moved)

        assert np.allclose(shift, (2, -3, 5), atol=0.5)


class TestApplyShift:
    """Tests for apply_shift function."""

    def test_roundtrip(self, small_dataset):
        """shift -> apply -> inverse shift preserves non-zero data."""
        from starfinder.io import load_volume

        vol = load_volume(small_dataset / "FOV_001" / "round1" / "ch00.tif").image
        original_sum = vol.sum()

        shifted = apply_shift(vol, (3, -2, 4))
        restored = apply_shift(shifted, (-3, 2, -4))

        # Restored should have some data (not all zeroed out)
        assert restored.sum() > 0
        # Shape preserved
        assert restored.shape == vol.shape


class TestRegisterVolume:
    """Tests for register_volume function."""

    def test_registers_multichannel(self, small_dataset):
        """Registers all channels and returns shifts."""
        from starfinder.io import load_round

        loaded_round = load_round(small_dataset / "FOV_001" / "round1", config=ImageLoadConfig(channel_labels=tuple(["ch00", "ch01", "ch02", "ch03"])))
        images = loaded_round.image
        _ = loaded_round.diagnostics

        # Create shifted version
        shifted = np.roll(images, (2, -3, 5, 0), axis=(0, 1, 2, 3))

        # Use ch00 as ref/mov
        ref_img = images[:, :, :, 0]
        mov_img = shifted[:, :, :, 0]

        _registration = estimate_transform(ref_img, mov_img, config=TranslationConfig(), reference_metadata=ImageMetadata("test/reference"), moving_metadata=ImageMetadata("test/moving"))
        registered = apply_transform(shifted, _registration.transform, config=_registration.application_config)
        shifts = tuple(-x for x in _registration.transform.correction_zyx)

        assert registered.shape == images.shape
        assert np.allclose(shifts, (2, -3, 5), atol=0.5)


class TestBackendParity:
    """NumPy vs scikit-image produce same results."""

    def test_backends_match(self, small_dataset):
        """Both backends return same shift for same input."""
        from starfinder.io import load_volume

        vol = load_volume(small_dataset / "FOV_001" / "round1" / "ch00.tif").image
        moved = np.roll(vol, (2, 3, -1), axis=(0, 1, 2))

        shift_np = phase_correlate(vol, moved)
        shift_sk = phase_correlate_skimage(vol, moved)

        assert np.allclose(shift_np, shift_sk, atol=0.5)
