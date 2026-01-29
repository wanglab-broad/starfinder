"""Tests for synthetic dataset generation and fixtures."""

from pathlib import Path

import numpy as np
import tifffile


def test_mini_dataset_exists(mini_dataset: Path):
    """Verify mini dataset directory exists."""
    assert mini_dataset.exists()
    assert (mini_dataset / "codebook.csv").exists()
    assert (mini_dataset / "ground_truth.json").exists()
    assert (mini_dataset / "FOV_001").exists()


def test_mini_ground_truth_structure(mini_ground_truth: dict):
    """Verify ground truth JSON has expected structure."""
    assert mini_ground_truth["version"] == "1.0"
    assert mini_ground_truth["n_rounds"] == 4
    assert mini_ground_truth["n_channels"] == 4
    assert "FOV_001" in mini_ground_truth["fovs"]

    fov = mini_ground_truth["fovs"]["FOV_001"]
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


def test_mini_image_loadable(mini_dataset: Path, mini_ground_truth: dict):
    """Verify generated TIFF files are valid and loadable."""
    shape = mini_ground_truth["image_shape"]  # [z, y, x]

    # Load a sample image
    tiff_path = mini_dataset / "FOV_001" / "round1" / "ch00.tif"
    image = tifffile.imread(tiff_path)

    assert image.shape == tuple(shape)
    assert image.dtype == np.uint16


def test_codebook_matches_ground_truth(mini_codebook: dict, mini_ground_truth: dict):
    """Verify codebook genes match those in ground truth."""
    fov = mini_ground_truth["fovs"]["FOV_001"]
    genes_in_spots = {spot["gene"] for spot in fov["spots"]}

    # All genes in spots should be in codebook
    for gene in genes_in_spots:
        assert gene in mini_codebook


def test_color_sequence_length(mini_ground_truth: dict):
    """Verify color sequences have correct length (n_rounds - 1)."""
    n_rounds = mini_ground_truth["n_rounds"]
    fov = mini_ground_truth["fovs"]["FOV_001"]

    for spot in fov["spots"]:
        # Color sequence should be n_rounds characters
        # (each pair of consecutive bases gives one color)
        assert len(spot["color_seq"]) == n_rounds
