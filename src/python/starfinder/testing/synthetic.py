"""Synthetic dataset generator for STARfinder testing.

This module generates synthetic spatial transcriptomics datasets with known
ground truth for testing the STARfinder pipeline components.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
import json

import numpy as np
from scipy.ndimage import gaussian_filter


# Two-base color-space encoding table
# Consecutive base pairs map to colors 1-4
BASE_PAIR_TO_COLOR = {
    "AA": "1", "CC": "1", "GG": "1", "TT": "1",
    "AC": "2", "CA": "2", "GT": "2", "TG": "2",
    "AG": "3", "CT": "3", "GA": "3", "TC": "3",
    "AT": "4", "CG": "4", "GC": "4", "TA": "4",
}

# Color to channel mapping
COLOR_TO_CHANNEL = {"1": 0, "2": 1, "3": 2, "4": 3}

# Test codebook: 8 genes with barcodes starting/ending with C
TEST_CODEBOOK = [
    ("GeneA", "CACGC"),
    ("GeneB", "CATGC"),
    ("GeneC", "CGAAC"),
    ("GeneD", "CGTAC"),
    ("GeneE", "CTGAC"),
    ("GeneF", "CTAGC"),
    ("GeneG", "CCATC"),
    ("GeneH", "CGCTC"),
]


@dataclass
class SyntheticConfig:
    """Configuration for synthetic dataset generation."""

    # Image dimensions
    height: int = 256
    width: int = 256
    n_z: int = 10

    # Dataset structure
    n_fovs: int = 2
    n_rounds: int = 4
    n_channels: int = 4

    # Spot generation
    n_spots_per_fov: int = 50
    spot_sigma: float = 1.5
    spot_intensity: tuple[int, int] = (200, 255)

    # Noise and background
    background_mean: int = 20
    background_std: int = 5
    noise_std: int = 10

    # Registration shifts (for testing registration)
    max_shift_xy: int = 5
    max_shift_z: int = 2

    # Random seed for reproducibility
    seed: int = 42


def get_preset_config(preset: Literal["mini", "standard"]) -> SyntheticConfig:
    """Get predefined configuration for a preset.

    Parameters
    ----------
    preset : {"mini", "standard"}
        - "mini": 1 FOV, 256x256x5, 20 spots (fast unit tests)
        - "standard": 4 FOVs, 512x512x10, 100 spots (integration tests)

    Returns
    -------
    SyntheticConfig
        Configuration for the specified preset
    """
    presets = {
        "mini": SyntheticConfig(
            height=256,
            width=256,
            n_z=5,
            n_fovs=1,
            n_spots_per_fov=20,
            seed=42,
        ),
        "standard": SyntheticConfig(
            height=512,
            width=512,
            n_z=10,
            n_fovs=4,
            n_spots_per_fov=100,
            seed=42,
        ),
    }
    if preset not in presets:
        raise ValueError(f"Unknown preset: {preset}. Choose from: {list(presets.keys())}")
    return presets[preset]


def encode_barcode_to_colors(barcode: str) -> str:
    """Encode a barcode to color sequence using two-base encoding.

    The barcode is first reversed, then consecutive base pairs are
    mapped to colors 1-4.

    Parameters
    ----------
    barcode : str
        5-character barcode (e.g., "CACGC")

    Returns
    -------
    str
        4-character color sequence (e.g., "4422")
    """
    reversed_barcode = barcode[::-1]
    colors = []
    for i in range(len(reversed_barcode) - 1):
        pair = reversed_barcode[i : i + 2]
        colors.append(BASE_PAIR_TO_COLOR[pair])
    return "".join(colors)


def create_test_image_stack(
    shape: tuple[int, int, int],
    spots: list[tuple[int, int, int, int]],
    background: int = 20,
    noise_std: int = 10,
    spot_sigma: float = 1.5,
    seed: int | None = None,
) -> np.ndarray:
    """Create a single 3D image stack with spots at specified locations.

    Parameters
    ----------
    shape : tuple[int, int, int]
        Image shape as (z, y, x)
    spots : list[tuple[int, int, int, int]]
        List of (z, y, x, intensity) tuples for spot locations
    background : int
        Mean background intensity
    noise_std : int
        Standard deviation of additive noise
    spot_sigma : float
        Gaussian sigma for spot blur
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    np.ndarray
        3D uint16 image stack
    """
    rng = np.random.default_rng(seed)

    # Create background with slight variation
    image = rng.normal(background, background / 4, shape).astype(np.float32)

    # Add spots
    for z, y, x, intensity in spots:
        if 0 <= z < shape[0] and 0 <= y < shape[1] and 0 <= x < shape[2]:
            # Create a small patch around the spot
            spot_image = np.zeros(shape, dtype=np.float32)
            spot_image[z, y, x] = intensity
            # Apply Gaussian blur
            spot_blurred = gaussian_filter(spot_image, sigma=spot_sigma)
            image += spot_blurred

    # Add noise
    image += rng.normal(0, noise_std, shape)

    # Clip and convert to uint16
    image = np.clip(image, 0, 65535).astype(np.uint16)

    return image


def create_shifted_stack(
    base_stack: np.ndarray,
    shift: tuple[int, int, int],
) -> np.ndarray:
    """Create a shifted version of a stack for registration testing.

    Parameters
    ----------
    base_stack : np.ndarray
        Original 3D image stack
    shift : tuple[int, int, int]
        Shift as (dz, dy, dx)

    Returns
    -------
    np.ndarray
        Shifted image stack (same dtype as input)
    """
    dz, dy, dx = shift
    result = np.zeros_like(base_stack)

    # Calculate source and destination slices
    src_z = slice(max(0, -dz), min(base_stack.shape[0], base_stack.shape[0] - dz))
    src_y = slice(max(0, -dy), min(base_stack.shape[1], base_stack.shape[1] - dy))
    src_x = slice(max(0, -dx), min(base_stack.shape[2], base_stack.shape[2] - dx))

    dst_z = slice(max(0, dz), min(base_stack.shape[0], base_stack.shape[0] + dz))
    dst_y = slice(max(0, dy), min(base_stack.shape[1], base_stack.shape[1] + dy))
    dst_x = slice(max(0, dx), min(base_stack.shape[2], base_stack.shape[2] + dx))

    result[dst_z, dst_y, dst_x] = base_stack[src_z, src_y, src_x]

    return result


def generate_synthetic_dataset(
    output_dir: Path,
    config: SyntheticConfig | None = None,
    preset: Literal["mini", "standard"] = "mini",
) -> dict:
    """Generate a complete synthetic dataset with ground truth.

    Parameters
    ----------
    output_dir : Path
        Directory to write generated files
    config : SyntheticConfig, optional
        Custom configuration. If None, uses preset defaults.
    preset : {"mini", "standard"}
        Preset configuration (ignored if config is provided)

    Returns
    -------
    dict
        Ground truth metadata including spot positions and expected barcodes
    """
    import tifffile

    if config is None:
        config = get_preset_config(preset)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(config.seed)

    # Prepare ground truth structure
    ground_truth = {
        "version": "1.0",
        "preset": preset if config is None else "custom",
        "seed": config.seed,
        "image_shape": [config.n_z, config.height, config.width],
        "n_rounds": config.n_rounds,
        "n_channels": config.n_channels,
        "fovs": {},
    }

    # Generate each FOV
    for fov_idx in range(config.n_fovs):
        fov_id = f"FOV_{fov_idx + 1:03d}"
        fov_dir = output_dir / fov_id
        fov_dir.mkdir(exist_ok=True)

        # Generate random shifts for each round (round1 is reference, no shift)
        shifts = {"round1": [0, 0, 0]}
        for r in range(2, config.n_rounds + 1):
            shifts[f"round{r}"] = [
                int(rng.integers(-config.max_shift_z, config.max_shift_z + 1)),
                int(rng.integers(-config.max_shift_xy, config.max_shift_xy + 1)),
                int(rng.integers(-config.max_shift_xy, config.max_shift_xy + 1)),
            ]

        # Generate random spot positions and gene assignments
        spots_info = []
        for spot_idx in range(config.n_spots_per_fov):
            gene, barcode = TEST_CODEBOOK[rng.integers(0, len(TEST_CODEBOOK))]
            color_seq = encode_barcode_to_colors(barcode)

            # Random position (with margin from edges, scaled by dimension)
            margin_z = min(1, config.n_z // 4)
            margin_xy = min(10, config.height // 10)
            z = int(rng.integers(margin_z, max(margin_z + 1, config.n_z - margin_z)))
            y = int(rng.integers(margin_xy, config.height - margin_xy))
            x = int(rng.integers(margin_xy, config.width - margin_xy))
            intensity = int(rng.integers(config.spot_intensity[0], config.spot_intensity[1] + 1))

            spots_info.append({
                "id": spot_idx,
                "gene": gene,
                "barcode": barcode,
                "color_seq": color_seq,
                "position": [z, y, x],
                "intensity": intensity,
            })

        # Generate images for each round and channel
        for round_idx in range(1, config.n_rounds + 1):
            round_id = f"round{round_idx}"
            round_dir = fov_dir / round_id
            round_dir.mkdir(exist_ok=True)

            shift = tuple(shifts[round_id])

            # Create image for each channel
            for ch in range(config.n_channels):
                # Collect spots that should appear in this channel for this round
                channel_spots = []
                for spot in spots_info:
                    # Color sequence index (0-3 for rounds 1-4)
                    color = spot["color_seq"][round_idx - 1]
                    spot_channel = COLOR_TO_CHANNEL[color]
                    if spot_channel == ch:
                        z, y, x = spot["position"]
                        channel_spots.append((z, y, x, spot["intensity"]))

                # Create base image with spots
                base_image = create_test_image_stack(
                    shape=(config.n_z, config.height, config.width),
                    spots=channel_spots,
                    background=config.background_mean,
                    noise_std=config.noise_std,
                    spot_sigma=config.spot_sigma,
                    seed=config.seed + fov_idx * 1000 + round_idx * 100 + ch,
                )

                # Apply shift (except for round1)
                if round_idx > 1:
                    image = create_shifted_stack(base_image, shift)
                else:
                    image = base_image

                # Save as TIFF
                tiff_path = round_dir / f"ch{ch:02d}.tif"
                tifffile.imwrite(tiff_path, image, imagej=True)

        # Store FOV ground truth
        ground_truth["fovs"][fov_id] = {
            "shifts": shifts,
            "spots": spots_info,
        }

    # Write codebook
    codebook_path = output_dir / "codebook.csv"
    with open(codebook_path, "w") as f:
        f.write("gene,barcode\n")
        for gene, barcode in TEST_CODEBOOK:
            f.write(f"{gene},{barcode}\n")

    # Write ground truth
    gt_path = output_dir / "ground_truth.json"
    with open(gt_path, "w") as f:
        json.dump(ground_truth, f, indent=2)

    return ground_truth
