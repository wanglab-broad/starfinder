"""Tests for unified synthetic data generation (starfinder.synthetic)."""

import json

import numpy as np
import pytest

from starfinder.synthetic import (
    SyntheticConfig, generate_displacement_field, render_spots, generate_volume,
    generate_codebook, generate_dataset, generate_registration_pairs, get_preset_config,
)
from starfinder.synthetic._perturbations import _apply_deformation_to_spots, _apply_shift_to_spots
from starfinder.synthetic._truth import _scene_table
from starfinder.benchmark._synthetic_io import _write_dataset



class TestApplyShiftToSpots:
    """Tests for coordinate-level global shift."""

    def test_spots_shift_correctly(self):
        shape = (10, 100, 100)
        spots = [(5, 50, 50, 200, 1.5), (3, 30, 30, 180, 1.5)]
        shifted = _apply_shift_to_spots(spots, (1, 2, -3), shape)
        assert len(shifted) == 2
        assert shifted[0][:3] == (6, 52, 47)
        assert shifted[1][:3] == (4, 32, 27)

    def test_boundary_spots_dropped(self):
        shape = (10, 100, 100)
        spots = [(0, 0, 0, 200, 1.5), (9, 99, 99, 180, 1.5)]
        # Shift pushes (0,0,0) to (-1,-1,-1) — out of bounds
        shifted = _apply_shift_to_spots(spots, (-1, -1, -1), shape)
        assert len(shifted) == 1
        assert shifted[0][:3] == (8, 98, 98)

    def test_zero_shift_preserves_spots(self):
        shape = (10, 100, 100)
        spots = [(5, 50, 50, 200, 1.5)]
        shifted = _apply_shift_to_spots(spots, (0, 0, 0), shape)
        assert shifted == spots

    def test_preserves_intensity_and_sigma(self):
        shape = (10, 100, 100)
        spots = [(5, 50, 50, 220, 1.8)]
        shifted = _apply_shift_to_spots(spots, (1, 1, 1), shape)
        assert shifted[0][3] == 220  # intensity
        assert shifted[0][4] == 1.8  # sigma


class TestApplyDeformationToSpots:
    """Tests for coordinate-level local deformation."""

    def test_uniform_field(self):
        shape = (10, 100, 100)
        # Constant displacement: all spots shift equally
        field = np.full((*shape, 3), [1.0, 2.0, -1.0], dtype=np.float32)
        spots = [(5, 50, 50, 200, 1.5)]
        deformed = _apply_deformation_to_spots(spots, field, shape)
        assert len(deformed) == 1
        assert deformed[0][:3] == (6, 52, 49)

    def test_boundary_spots_dropped(self):
        shape = (10, 100, 100)
        # Push first spot out of bounds
        field = np.zeros((*shape, 3), dtype=np.float32)
        field[..., 0] = -20  # large negative dz pushes all z<20 out
        spots = [(5, 50, 50, 200, 1.5), (9, 50, 50, 180, 1.5)]
        deformed = _apply_deformation_to_spots(spots, field, shape)
        assert len(deformed) == 0

    def test_zero_field_preserves_spots(self):
        shape = (10, 100, 100)
        field = np.zeros((*shape, 3), dtype=np.float32)
        spots = [(5, 50, 50, 200, 1.5)]
        deformed = _apply_deformation_to_spots(spots, field, shape)
        assert deformed == spots




class TestCreateTestImageStack:
    """Tests for per-spot sigma rendering."""

    def test_per_spot_sigma(self):
        """Different sigma values produce different spot sizes."""
        shape = (10, 64, 64)
        spot_narrow = [(5, 32, 20, 200, 0.8)]
        spot_wide = [(5, 32, 44, 200, 3.0)]
        img = render_spots(shape, _scene_table(spot_narrow + spot_wide), seed=42)
        # Both spots should be visible
        assert img[5, 32, 20] > 50
        assert img[5, 32, 44] > 50

    def test_empty_spots(self):
        """No spots → only background noise."""
        shape = (4, 32, 32)
        img = render_spots(shape, _scene_table([]), seed=42)
        assert img.shape == shape
        assert img.dtype == np.uint8


class TestGenerateSyntheticDataset:
    """Tests for the multi-round E2E dataset generator."""

    def test_tiny_preset(self, tmp_path):
        config = get_preset_config("tiny")
        assert len(config.shape_zyx) == 3
        assert all(isinstance(size, int) for size in config.shape_zyx)
        result = generate_dataset(
            config=config,
            preset="tiny",
        )

        _write_dataset(result, tmp_path, annotations=False)
        gt = result.historical_truth
        assert gt["version"] == "2.0"

        # Verify structure
        assert (tmp_path / "codebook.csv").exists()
        assert (tmp_path / "ground_truth.json").exists()
        assert (tmp_path / "FOV_001").exists()
        assert (tmp_path / "FOV_002").exists()

        # Verify ground truth
        assert gt["n_rounds"] == 4
        assert gt["n_channels"] == 4
        assert "FOV_001" in gt["fovs"]
        fov = gt["fovs"]["FOV_001"]
        assert len(fov["spots"]) == 10
        assert "shifts" in fov
        assert "round1" in fov["shifts"]

        # Verify image files
        import tifffile
        img = tifffile.imread(tmp_path / "FOV_001" / "round1" / "ch00.tif")
        assert img.shape == (8, 128, 128)
        assert img.dtype == np.uint8



class TestPresetConfigs:
    """Tests for preset configuration lookup."""

    def test_all_presets_exist(self):
        for name in ["tiny", "small", "medium", "large", "tissue", "thick_medium"]:
            config = get_preset_config(name)
            assert isinstance(config, SyntheticConfig)

    def test_invalid_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown preset"):
            get_preset_config("nonexistent")

    def test_large_preset_has_64_gene_codebook(self):
        config = get_preset_config("large")
        assert config.codebook is not None
        assert len(config.codebook) == 64


class TestGenerateCodebook:
    """Tests for codebook generation."""

    def test_small_codebook(self):
        codebook = generate_codebook(8)
        assert len(codebook) == 8
        assert all(len(barcode) == 5 for _, barcode in codebook)

    def test_max_codebook(self):
        codebook = generate_codebook(64)
        assert len(codebook) == 64

    def test_exceeds_max_raises(self):
        with pytest.raises(ValueError, match="unique color sequences"):
            generate_codebook(100)


class TestCreateTestVolume:
    """Tests for single-channel volume convenience function."""

    def test_basic_creation(self):
        vol = generate_volume((8, 64, 64), n_spots=5, seed=42)
        assert vol.shape == (8, 64, 64)
        assert vol.dtype == np.uint8
        assert vol.max() > vol.min()
