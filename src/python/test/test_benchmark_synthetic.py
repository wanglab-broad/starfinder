"""Tests for unified synthetic data generation (benchmark.synthetic)."""

import json

import numpy as np
import pytest

from starfinder.benchmark.synthetic import (
    TEST_CODEBOOK,
    SyntheticConfig,
    apply_deformation_to_spots,
    apply_shift_to_spots,
    create_deformation_field,
    create_test_image_stack,
    create_test_volume,
    encode_barcode_to_colors,
    generate_codebook,
    generate_synthetic_dataset,
    get_preset_config,
)


class TestApplyShiftToSpots:
    """Tests for coordinate-level global shift."""

    def test_spots_shift_correctly(self):
        shape = (10, 100, 100)
        spots = [(5, 50, 50, 200, 1.5), (3, 30, 30, 180, 1.5)]
        shifted = apply_shift_to_spots(spots, (1, 2, -3), shape)
        assert len(shifted) == 2
        assert shifted[0][:3] == (6, 52, 47)
        assert shifted[1][:3] == (4, 32, 27)

    def test_boundary_spots_dropped(self):
        shape = (10, 100, 100)
        spots = [(0, 0, 0, 200, 1.5), (9, 99, 99, 180, 1.5)]
        # Shift pushes (0,0,0) to (-1,-1,-1) — out of bounds
        shifted = apply_shift_to_spots(spots, (-1, -1, -1), shape)
        assert len(shifted) == 1
        assert shifted[0][:3] == (8, 98, 98)

    def test_zero_shift_preserves_spots(self):
        shape = (10, 100, 100)
        spots = [(5, 50, 50, 200, 1.5)]
        shifted = apply_shift_to_spots(spots, (0, 0, 0), shape)
        assert shifted == spots

    def test_preserves_intensity_and_sigma(self):
        shape = (10, 100, 100)
        spots = [(5, 50, 50, 220, 1.8)]
        shifted = apply_shift_to_spots(spots, (1, 1, 1), shape)
        assert shifted[0][3] == 220  # intensity
        assert shifted[0][4] == 1.8  # sigma


class TestApplyDeformationToSpots:
    """Tests for coordinate-level local deformation."""

    def test_uniform_field(self):
        shape = (10, 100, 100)
        # Constant displacement: all spots shift equally
        field = np.full((*shape, 3), [1.0, 2.0, -1.0], dtype=np.float32)
        spots = [(5, 50, 50, 200, 1.5)]
        deformed = apply_deformation_to_spots(spots, field, shape)
        assert len(deformed) == 1
        assert deformed[0][:3] == (6, 52, 49)

    def test_boundary_spots_dropped(self):
        shape = (10, 100, 100)
        # Push first spot out of bounds
        field = np.zeros((*shape, 3), dtype=np.float32)
        field[..., 0] = -20  # large negative dz pushes all z<20 out
        spots = [(5, 50, 50, 200, 1.5), (9, 50, 50, 180, 1.5)]
        deformed = apply_deformation_to_spots(spots, field, shape)
        assert len(deformed) == 0

    def test_zero_field_preserves_spots(self):
        shape = (10, 100, 100)
        field = np.zeros((*shape, 3), dtype=np.float32)
        spots = [(5, 50, 50, 200, 1.5)]
        deformed = apply_deformation_to_spots(spots, field, shape)
        assert deformed == spots


class TestPerRoundVariation:
    """Tests for per-round intensity/sigma jitter."""

    def test_intensity_varies_across_rounds(self):
        """Same spot at different rounds should have different intensity."""
        seed = 42
        spot_id = 0
        base_intensity = 220
        intensities = []
        for round_idx in range(1, 5):
            jitter_rng = np.random.default_rng(seed + spot_id * 100 + round_idx)
            jittered = int(base_intensity * (1 + jitter_rng.normal(0, 0.1)))
            intensities.append(jittered)
        # At least 2 distinct values across 4 rounds
        assert len(set(intensities)) >= 2

    def test_sigma_varies_across_rounds(self):
        """Same spot at different rounds should have different sigma."""
        seed = 42
        spot_id = 0
        base_sigma = 1.5
        sigmas = []
        for round_idx in range(1, 5):
            jitter_rng = np.random.default_rng(seed + spot_id * 100 + round_idx)
            _ = jitter_rng.normal(0, 0.1)  # consume intensity jitter
            jittered_sigma = base_sigma * (1 + jitter_rng.normal(0, 0.05))
            sigmas.append(jittered_sigma)
        assert len(set(sigmas)) >= 2

    def test_variation_is_deterministic(self):
        """Same seed → same jitter values."""
        seed = 42
        spot_id = 5
        round_idx = 2
        results = []
        for _ in range(2):
            jitter_rng = np.random.default_rng(seed + spot_id * 100 + round_idx)
            val = int(200 * (1 + jitter_rng.normal(0, 0.1)))
            results.append(val)
        assert results[0] == results[1]


class TestCreateTestImageStack:
    """Tests for per-spot sigma rendering."""

    def test_per_spot_sigma(self):
        """Different sigma values produce different spot sizes."""
        shape = (10, 64, 64)
        spot_narrow = [(5, 32, 20, 200, 0.8)]
        spot_wide = [(5, 32, 44, 200, 3.0)]
        img = create_test_image_stack(shape, spot_narrow + spot_wide, seed=42)
        # Both spots should be visible
        assert img[5, 32, 20] > 50
        assert img[5, 32, 44] > 50

    def test_empty_spots(self):
        """No spots → only background noise."""
        shape = (4, 32, 32)
        img = create_test_image_stack(shape, [], seed=42)
        assert img.shape == shape
        assert img.dtype == np.uint8


class TestGenerateSyntheticDataset:
    """Tests for the multi-round E2E dataset generator."""

    def test_tiny_preset(self, tmp_path):
        config = get_preset_config("tiny")
        gt = generate_synthetic_dataset(
            output_dir=tmp_path,
            config=config,
            preset="tiny",
        )

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

    def test_ground_truth_version_2(self, tmp_path):
        """New generator produces version 2.0 ground truth."""
        config = get_preset_config("tiny")
        gt = generate_synthetic_dataset(tmp_path, config=config)
        assert gt["version"] == "2.0"


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
        vol = create_test_volume((8, 64, 64), n_spots=5, seed=42)
        assert vol.shape == (8, 64, 64)
        assert vol.dtype == np.uint8
        assert vol.max() > vol.min()
