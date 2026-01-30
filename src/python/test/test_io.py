"""Tests for starfinder.io module."""

import numpy as np
import pytest
import tifffile
from pathlib import Path

from starfinder.io import load_multipage_tiff, load_image_stacks, save_stack


class TestLoadMultipageTiff:
    """Tests for load_multipage_tiff function."""

    def test_load_returns_zyx_shape(self, tmp_path: Path):
        """Loading a multi-page TIFF returns (Z, Y, X) array."""
        # Create test TIFF: 5 slices, 64x32 pixels
        test_data = np.random.randint(0, 255, (5, 64, 32), dtype=np.uint8)
        tiff_path = tmp_path / "test.tif"
        tifffile.imwrite(tiff_path, test_data)

        result = load_multipage_tiff(tiff_path)

        assert result.shape == (5, 64, 32)

    def test_load_converts_to_uint8_by_default(self, tmp_path: Path):
        """Loading converts to uint8 by default."""
        test_data = np.random.randint(0, 65535, (3, 32, 32), dtype=np.uint16)
        tiff_path = tmp_path / "test16.tif"
        tifffile.imwrite(tiff_path, test_data)

        result = load_multipage_tiff(tiff_path)

        assert result.dtype == np.uint8

    def test_load_preserves_dtype_when_convert_false(self, tmp_path: Path):
        """Loading preserves original dtype when convert_uint8=False."""
        test_data = np.random.randint(0, 65535, (3, 32, 32), dtype=np.uint16)
        tiff_path = tmp_path / "test16.tif"
        tifffile.imwrite(tiff_path, test_data)

        result = load_multipage_tiff(tiff_path, convert_uint8=False)

        assert result.dtype == np.uint16

    def test_load_nonexistent_file_raises(self):
        """Loading non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_multipage_tiff("/nonexistent/path.tif")


class TestSaveStack:
    """Tests for save_stack function."""

    def test_save_3d_roundtrip(self, tmp_path: Path):
        """Saving and reloading 3D array preserves data."""
        original = np.random.randint(0, 255, (5, 64, 32), dtype=np.uint8)
        tiff_path = tmp_path / "output.tif"

        save_stack(original, tiff_path)
        result = load_multipage_tiff(tiff_path, convert_uint8=False)

        np.testing.assert_array_equal(result, original)

    def test_save_overwrites_existing(self, tmp_path: Path):
        """Saving overwrites existing file."""
        tiff_path = tmp_path / "output.tif"

        # Write first file
        data1 = np.zeros((3, 32, 32), dtype=np.uint8)
        save_stack(data1, tiff_path)

        # Overwrite with different data
        data2 = np.ones((5, 64, 64), dtype=np.uint8) * 255
        save_stack(data2, tiff_path)

        result = load_multipage_tiff(tiff_path, convert_uint8=False)
        assert result.shape == (5, 64, 64)

    def test_save_with_compression(self, tmp_path: Path):
        """Saving with compression creates smaller file."""
        # Use patterned data (not random) since random data doesn't compress well
        data = np.zeros((10, 128, 128), dtype=np.uint8)
        data[:, 20:100, 20:100] = 128  # Add compressible pattern
        path_uncompressed = tmp_path / "uncompressed.tif"
        path_compressed = tmp_path / "compressed.tif"

        save_stack(data, path_uncompressed, compress=False)
        save_stack(data, path_compressed, compress=True)

        size_uncompressed = path_uncompressed.stat().st_size
        size_compressed = path_compressed.stat().st_size

        # Compressed should be smaller for patterned/real image data
        assert size_compressed < size_uncompressed


class TestLoadImageStacks:
    """Tests for load_image_stacks function."""

    def test_load_returns_zyxc_shape(self, tmp_path: Path):
        """Loading multiple channels returns (Z, Y, X, C) array."""
        # Create 4 channel files
        for i, ch in enumerate(["ch00", "ch01", "ch02", "ch03"]):
            data = np.full((5, 64, 32), i * 50, dtype=np.uint8)
            tifffile.imwrite(tmp_path / f"img_{ch}.tif", data)

        result, metadata = load_image_stacks(
            tmp_path, ["ch00", "ch01", "ch02", "ch03"]
        )

        assert result.shape == (5, 64, 32, 4)
        assert result.dtype == np.uint8

    def test_load_respects_channel_order(self, tmp_path: Path):
        """Channels are stacked in the order specified."""
        # ch00 = all 0s, ch01 = all 100s
        tifffile.imwrite(tmp_path / "img_ch00.tif", np.zeros((3, 32, 32), dtype=np.uint8))
        tifffile.imwrite(tmp_path / "img_ch01.tif", np.full((3, 32, 32), 100, dtype=np.uint8))

        result, _ = load_image_stacks(tmp_path, ["ch00", "ch01"])

        assert result[0, 0, 0, 0] == 0    # ch00 is first
        assert result[0, 0, 0, 1] == 100  # ch01 is second

    def test_load_with_size_mismatch_crops_and_warns(self, tmp_path: Path):
        """Size mismatch between channels crops to minimum and warns."""
        # ch00: 5x64x32, ch01: 5x60x30
        tifffile.imwrite(tmp_path / "ch00.tif", np.zeros((5, 64, 32), dtype=np.uint8))
        tifffile.imwrite(tmp_path / "ch01.tif", np.zeros((5, 60, 30), dtype=np.uint8))

        with pytest.warns(UserWarning, match="size mismatch"):
            result, metadata = load_image_stacks(tmp_path, ["ch00", "ch01"])

        assert result.shape == (5, 60, 30, 2)  # Cropped to minimum
        assert metadata["cropped"] is True

    def test_load_missing_channel_raises(self, tmp_path: Path):
        """Missing channel file raises ValueError."""
        tifffile.imwrite(tmp_path / "ch00.tif", np.zeros((3, 32, 32), dtype=np.uint8))

        with pytest.raises(ValueError, match="ch01"):
            load_image_stacks(tmp_path, ["ch00", "ch01"])

    def test_load_nonexistent_dir_raises(self):
        """Non-existent directory raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_image_stacks("/nonexistent/dir", ["ch00"])

    def test_load_with_subdir(self, tmp_path: Path):
        """Loading with subdir searches in subdirectory."""
        subdir = tmp_path / "images"
        subdir.mkdir()
        tifffile.imwrite(subdir / "ch00.tif", np.zeros((3, 32, 32), dtype=np.uint8))

        result, _ = load_image_stacks(tmp_path, ["ch00"], subdir="images")

        assert result.shape == (3, 32, 32, 1)
