"""Tests for starfinder.preprocessing module."""

from starfinder.preprocessing import MinMaxNormalizationConfig
from starfinder.preprocessing import ReconstructionConfig
from starfinder.preprocessing import TophatConfig

import numpy as np
import pytest

from starfinder.preprocessing import (
    match_histogram,
    normalize_min_max,
    correct_background,
    filter_tophat,
)

pytestmark = pytest.mark.preprocessing


class TestMinMaxNormalize:
    """Tests for normalize_min_max."""

    def test_rescales_to_uint8(self):
        """Output spans [0, 255] for non-constant input."""
        vol = np.random.RandomState(42).randint(50, 200, (5, 16, 16), dtype=np.uint8)
        result = normalize_min_max(vol, config=MinMaxNormalizationConfig('uint8', (0, 255)))
        assert result.dtype == np.uint8
        assert result.shape == vol.shape
        assert result.min() == 0
        assert result.max() == 255

    def test_per_channel_independence(self):
        """Each channel is normalized independently."""
        vol = np.zeros((3, 8, 8, 2), dtype=np.uint8)
        vol[:, :, :, 0] = 100  # constant → zeros
        vol[:, :, :, 1] = np.arange(192, dtype=np.uint8).reshape(3, 8, 8)
        result = normalize_min_max(vol, config=MinMaxNormalizationConfig('uint8', (0, 255)))
        assert result.shape == vol.shape
        assert result.dtype == vol.dtype
        # Channel 0: constant → all zeros
        assert result[:, :, :, 0].max() == 0
        # Channel 1: variable → spans full range
        assert result[:, :, :, 1].max() == 255

    def test_constant_channel_zeros(self):
        """Constant-valued channel maps to all zeros."""
        vol = np.full((3, 8, 8), 42, dtype=np.uint8)
        result = normalize_min_max(vol, config=MinMaxNormalizationConfig('uint8', (0, 255)))
        assert np.all(result == 0)


class TestSNRGating:
    """Tests for SNR-gated normalization."""

    def test_snr_gating_skips_low_snr_channel(self):
        """Low-SNR channel is NOT rescaled; high-SNR channel IS rescaled."""
        vol = np.zeros((3, 32, 32, 2), dtype=np.uint8)
        # Channel 0: nonconstant low SNR (max ~= mean, SNR ≈ 1)
        vol[:, :, :, 0] = 50
        vol[0, 0, 0, 0] = 49
        # Channel 1: high SNR (background=5, bright spot=200, SNR >> 5)
        vol[:, :, :, 1] = 5
        vol[1, 16, 16, 1] = 200

        result = normalize_min_max(vol, config=MinMaxNormalizationConfig('uint8', (0, 255), snr_threshold=5.0))

        # Nonconstant low-SNR values are retained exactly.
        np.testing.assert_array_equal(result[:, :, :, 0], vol[:, :, :, 0])
        # Channel 1: should be rescaled to [0, 255]
        assert result[:, :, :, 1].max() == 255

    def test_snr_gating_none_is_backward_compat(self):
        """With snr_threshold=None, all channels are normalized (default)."""
        vol = np.zeros((3, 32, 32, 2), dtype=np.uint8)
        vol[:, :, :, 0] = 50  # low SNR
        vol[:, :, :, 1] = 5
        vol[1, 16, 16, 1] = 200

        result = normalize_min_max(vol, config=MinMaxNormalizationConfig('uint8', (0, 255), snr_threshold=None))

        # Both channels should be rescaled to [0, 255]
        # Channel 0: constant → all zeros (lo == hi case)
        assert result[:, :, :, 0].max() == 0
        # Channel 1: rescaled to 255
        assert result[:, :, :, 1].max() == 255


class TestHistogramMatch:
    """Tests for match_histogram."""

    def test_shape_preserved(self):
        """Output shape matches input shape for 3D and 4D."""
        vol3d = np.random.RandomState(0).randint(0, 256, (3, 16, 16), dtype=np.uint8)
        ref = np.random.RandomState(1).randint(0, 256, (3, 16, 16), dtype=np.uint8)
        assert match_histogram(vol3d, ref).shape == vol3d.shape

        vol4d = np.random.RandomState(2).randint(0, 256, (3, 16, 16, 2), dtype=np.uint8)
        assert match_histogram(vol4d, ref).shape == vol4d.shape

    def test_histogram_shifts_toward_reference(self):
        """Matched volume mean shifts toward reference mean."""
        vol = np.random.RandomState(42).randint(0, 50, (5, 32, 32), dtype=np.uint8)
        ref = np.random.RandomState(99).randint(200, 256, (5, 32, 32), dtype=np.uint8)
        result = match_histogram(vol, ref)
        assert result.dtype == vol.dtype
        assert result.shape == vol.shape
        # Result mean should be closer to ref mean than original was
        assert abs(result.mean() - ref.mean()) < abs(vol.mean() - ref.mean())


def _sort_based_correction(volume, radius):
    """The reconstruction step before W-886, with skimage's sort-based reconstruction: the tests' oracle."""
    from skimage.morphology import black_tophat, disk, erosion, reconstruction, white_tophat
    channels = volume[..., None] if volume.ndim == 3 else volume
    result = np.empty_like(channels)
    se = disk(radius)
    limits = np.iinfo(volume.dtype) if volume.dtype.kind in "ui" else np.finfo(volume.dtype)
    for c in range(channels.shape[-1]):
        for z in range(channels.shape[0]):
            image = channels[z, :, :, c].astype(np.float64)
            image = image - reconstruction(erosion(image, se), image, method="dilation")
            image = image + white_tophat(image, se) - black_tophat(image, se)
            result[z, :, :, c] = np.clip(image, limits.min, limits.max).astype(volume.dtype)
    return result[..., 0] if volume.ndim == 3 else result


def _scaled(values, dtype):
    """values in [0, 1] on the dtype's range: integers fill it to the maximum; floats span -1000 to 1000."""
    if dtype.kind in "ui":
        info = np.iinfo(dtype)
        return np.round(info.min + values * (int(info.max) - int(info.min))).astype(dtype)
    return (values * 2000 - 1000).astype(dtype)


def _ties(dtype):
    """Four grey levels only, so most neighbours tie (4 x 64 x 64 x 2)."""
    return _scaled(np.random.default_rng(886).integers(0, 4, (4, 64, 64, 2)) / 3, dtype)


def _plateaus(dtype):
    """Flat cells on a flat background, with single bright spots and a cell at the maximum (4 x 64 x 64 x 2)."""
    values = np.full((4, 64, 64, 2), 0.1)
    values[:, 8:24, 8:30] = 0.4
    values[:, 30:50, 20:44] = 0.6
    values[:, 40:60, 48:60] = 1.0
    values[:, 16, 16] = values[:, 36, 30] = 0.9
    values[1, ..., 1] = values[1, ..., 1][::-1]
    return _scaled(values, dtype)


def _constant(dtype):
    """Every slice constant, one per Z plane and channel, at levels including the maximum (4 x 64 x 64 x 2)."""
    levels = np.array([[0.0, 1.0], [0.5, 0.25], [1.0, 1.0], [0.75, 0.0]])
    return _scaled(np.broadcast_to(levels[:, None, None, :], (4, 64, 64, 2)).copy(), dtype)


def _zeros(dtype):
    """All-zero slices (4 x 64 x 64 x 2)."""
    return np.zeros((4, 64, 64, 2), dtype=dtype)


def _noisy_spots(dtype):
    """Noise around a mid level with spots, non-integer for floats, on 32 x 64 x 64 (the fixture limit)."""
    rng = np.random.default_rng(887)
    values = np.clip(rng.normal(0.3, 0.05, (32, 64, 64, 1)), 0, 1)
    values[:, rng.integers(0, 64, 40), rng.integers(0, 64, 40)] = 1.0
    return _scaled(values, dtype)


_RECONSTRUCTION_SCENES = {"ties": _ties, "plateaus": _plateaus, "constant": _constant, "zeros": _zeros,
                          "noisy_spots": _noisy_spots}


class TestMorphologicalReconstruction:
    """Tests for correct_background."""

    def test_shape_preserved(self):
        """Output shape matches input for 3D and 4D."""
        vol3d = np.random.RandomState(0).randint(0, 256, (3, 32, 32), dtype=np.uint8)
        assert correct_background(vol3d, config=ReconstructionConfig(radius_yx=3)).shape == vol3d.shape

        vol4d = np.random.RandomState(1).randint(0, 256, (3, 32, 32, 2), dtype=np.uint8)
        assert correct_background(vol4d, config=ReconstructionConfig(radius_yx=3)).shape == vol4d.shape

    def test_background_removed(self):
        """Smooth background is reduced; small features preserved."""
        # Create image with uniform background + bright spot
        img = np.full((1, 64, 64), 100, dtype=np.uint8)
        img[0, 32, 32] = 255
        result = correct_background(img, config=ReconstructionConfig(radius_yx=5))
        assert result.dtype == img.dtype
        assert result.shape == img.shape
        # Background should be suppressed (lower than original 100)
        bg_val = result[0, 0, 0]
        assert bg_val < 100

    @pytest.mark.parametrize("scene", sorted(_RECONSTRUCTION_SCENES))
    @pytest.mark.parametrize("dtype", ["uint8", "uint16", ">u2", "int16", "float32", "float64"])
    def test_equals_the_sort_based_reconstruction_bit_for_bit(self, scene, dtype):
        """W-886: SimpleITK's downhill filter gives the step's earlier skimage output exactly, in every dtype."""
        volume = _RECONSTRUCTION_SCENES[scene](np.dtype(dtype))
        original = volume.copy()
        result = correct_background(volume, config=ReconstructionConfig(radius_yx=3))
        expected = _sort_based_correction(volume, 3)
        assert result.dtype == volume.dtype and result.shape == volume.shape
        assert result.tobytes() == expected.tobytes()
        np.testing.assert_array_equal(volume, original)

    def test_zyx_input_and_another_radius_equal_the_sort_based_reconstruction(self):
        volume = _RECONSTRUCTION_SCENES["plateaus"](np.dtype(np.uint16))[..., 0]
        result = correct_background(volume, config=ReconstructionConfig(radius_yx=5))
        assert result.tobytes() == _sort_based_correction(volume, 5).tobytes()

    @pytest.mark.parametrize("dtype, work", [("uint8", "uint8"), ("uint16", "uint16"), (">u2", "uint16"),
                                             ("int32", "int32"), ("float32", "float64"), ("float64", "float64")])
    def test_the_reconstruction_runs_in_the_source_integer_type_or_float64(self, monkeypatch, dtype, work):
        import SimpleITK as sitk
        seen = []
        to_image = sitk.GetImageFromArray

        def spy(array, *args, **kwargs):
            seen.append(array.dtype)
            return to_image(array, *args, **kwargs)

        monkeypatch.setattr(sitk, "GetImageFromArray", spy)
        correct_background(_RECONSTRUCTION_SCENES["ties"](np.dtype(dtype)))
        assert seen and set(seen) == {np.dtype(work)}


class TestTophatFilter:
    """Tests for filter_tophat."""

    def test_shape_preserved(self):
        """Output shape matches input for 3D and 4D."""
        vol3d = np.random.RandomState(0).randint(0, 256, (3, 32, 32), dtype=np.uint8)
        assert filter_tophat(vol3d, config=TophatConfig(radius_yx=3)).shape == vol3d.shape

        vol4d = np.random.RandomState(1).randint(0, 256, (3, 32, 32, 2), dtype=np.uint8)
        assert filter_tophat(vol4d, config=TophatConfig(radius_yx=3)).shape == vol4d.shape

    def test_removes_large_structures(self):
        """Uniform background is removed; small features remain."""
        img = np.full((1, 64, 64), 100, dtype=np.uint8)
        img[0, 32, 32] = 255  # small bright spot
        result = filter_tophat(img, config=TophatConfig(radius_yx=5))
        assert result.dtype == img.dtype
        assert result.shape == img.shape
        # Background should be ~0 (tophat removes structures larger than SE)
        assert result[0, 0, 0] < 10
        # Bright spot should be preserved (relative to background)
        assert result[0, 32, 32] > 100
