"""Tests for synthetic dataset generation and fixtures."""

from pathlib import Path

import numpy as np
import tifffile


def test_small_dataset_exists(small_dataset: Path):
    """Verify small dataset directory exists."""
    assert small_dataset.exists()
    assert (small_dataset / "codebook.csv").exists()
    assert (small_dataset / "ground_truth.json").exists()
    assert (small_dataset / "FOV_001").exists()


def test_small_ground_truth_structure(small_ground_truth: dict):
    """Verify ground truth JSON has expected structure."""
    assert small_ground_truth["version"] == "1.0"
    assert small_ground_truth["n_rounds"] == 4
    assert small_ground_truth["n_channels"] == 4
    assert "FOV_001" in small_ground_truth["fovs"]

    fov = small_ground_truth["fovs"]["FOV_001"]
    assert "shifts" in fov
    assert "spots" in fov
    assert len(fov["spots"]) > 0

    # Verify spot structure
    spot = fov["spots"][0]
    assert "gene" in spot
    assert "barcode" in spot
    assert "color_seq" in spot
    assert "position" in spot
    assert len(spot["position"]) == 3  # z, y, x


def test_small_image_loadable(small_dataset: Path, small_ground_truth: dict):
    """Verify generated TIFF files are valid and loadable."""
    shape = small_ground_truth["image_shape"]  # [z, y, x]

    # Load a sample image
    tiff_path = small_dataset / "FOV_001" / "round1" / "ch00.tif"
    image = tifffile.imread(tiff_path)

    assert image.shape == tuple(shape)
    assert image.dtype == np.uint8


def test_codebook_matches_ground_truth(small_codebook: dict, small_ground_truth: dict):
    """Verify codebook genes match those in ground truth."""
    fov = small_ground_truth["fovs"]["FOV_001"]
    genes_in_spots = {spot["gene"] for spot in fov["spots"]}

    # All genes in spots should be in codebook
    for gene in genes_in_spots:
        assert gene in small_codebook


def test_color_sequence_length(small_ground_truth: dict):
    """Verify color sequences have correct length (n_rounds - 1)."""
    n_rounds = small_ground_truth["n_rounds"]
    fov = small_ground_truth["fovs"]["FOV_001"]

    for spot in fov["spots"]:
        # Color sequence should be n_rounds characters
        # (each pair of consecutive bases gives one color)
        assert len(spot["color_seq"]) == n_rounds
