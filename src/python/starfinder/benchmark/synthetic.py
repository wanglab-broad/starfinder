"""Unified synthetic data generation for STARfinder testing and benchmarking.

This module provides:
- Multi-round, multi-channel FOV datasets for E2E pipeline testing
- Single-channel ref/moving pairs for registration benchmarking
- Coordinate-first rendering: transforms are applied to spot positions
  before rendering, so images always contain clean analytical Gaussians

Key design: both global shifts and local deformations are coordinate
transforms applied to spot positions *before* rendering. This avoids
interpolation blur from warping rendered images.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Literal

import numpy as np

from starfinder.barcode.encoding import BASE_PAIR_TO_COLOR, COLOR_TO_CHANNEL, encode_bases
from starfinder.benchmark.presets import SIZE_PRESETS, SPOT_COUNTS, SHIFT_RANGES


# ---------------------------------------------------------------------------
# Codebook constants and helpers
# ---------------------------------------------------------------------------

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


def generate_codebook(n_genes: int) -> list[tuple[str, str]]:
    """Generate a codebook with n_genes CNNNNC barcodes and unique color sequences.

    Enumerates all 5-base barcodes of the form C-{A,C,G,T}^3-C,
    filters to those with unique color sequences, and returns the
    first n_genes entries.

    Parameters
    ----------
    n_genes : int
        Number of genes (max 64 for 4-round, 4-channel).

    Returns
    -------
    list[tuple[str, str]]
        List of (gene_name, barcode) tuples.

    Raises
    ------
    ValueError
        If n_genes exceeds the number of unique color sequences.
    """
    bases = "ACGT"
    all_barcodes = [f"C{''.join(mid)}C" for mid in product(bases, repeat=3)]

    # Filter to unique color sequences
    seen_colors: dict[str, str] = {}
    unique_entries: list[tuple[str, str]] = []
    for barcode in all_barcodes:
        color_seq = encode_barcode_to_colors(barcode)
        if color_seq not in seen_colors:
            seen_colors[color_seq] = barcode
            unique_entries.append((barcode, color_seq))

    if n_genes > len(unique_entries):
        raise ValueError(
            f"Requested {n_genes} genes but only {len(unique_entries)} "
            f"unique color sequences available with CNNNNC barcodes."
        )

    codebook = []
    for i, (barcode, _) in enumerate(unique_entries[:n_genes]):
        gene_name = f"Gene{i + 1:03d}"
        codebook.append((gene_name, barcode))

    return codebook


def encode_barcode_to_colors(barcode: str) -> str:
    """Encode a barcode to color sequence using two-base encoding.

    The barcode is first reversed, then consecutive base pairs are
    mapped to colors 1-4. This is a convenience wrapper around
    encode_bases() that handles the STARmap reversal convention.

    Parameters
    ----------
    barcode : str
        5-character barcode (e.g., "CACGC")

    Returns
    -------
    str
        4-character color sequence (e.g., "4422")
    """
    return encode_bases(barcode[::-1])


# ---------------------------------------------------------------------------
# Deformation field generation (moved from benchmark/data.py)
# ---------------------------------------------------------------------------

# Deformation configurations (as percentage of smallest XY dimension)
# These will be scaled to actual pixels based on image size, with optional caps
DEFORMATION_CONFIGS = {
    "polynomial_small": {"type": "polynomial", "max_displacement_pct": 3.0, "cap_px": 15.0},
    "polynomial_large": {"type": "polynomial", "max_displacement_pct": 6.0, "cap_px": 30.0},
    "gaussian_small": {"type": "gaussian", "max_displacement_pct": 3.0, "radius_pct": 6.0, "cap_px": 15.0},
    "gaussian_large": {"type": "gaussian", "max_displacement_pct": 6.0, "radius_pct": 10.0, "cap_px": 30.0},
    "multi_point": {"type": "multi_point", "max_displacement_pct": 4.0, "n_points": 4, "radius_pct": 5.0, "cap_px": 20.0},
    "linear_small": {"type": "linear", "max_displacement_pct": 1.0, "cap_px": 10.0},
}


def scale_deformation_config(config: dict, shape: tuple[int, int, int]) -> dict:
    """Scale deformation config from percentages to absolute pixels.

    Parameters
    ----------
    config : dict
        Deformation config with _pct suffix fields and optional cap_px.
    shape : tuple[int, int, int]
        Volume shape (Z, Y, X).

    Returns
    -------
    dict
        Config with absolute pixel values.
    """
    min_xy = min(shape[1], shape[2])
    scaled = {"type": config["type"]}

    if "max_displacement_pct" in config:
        displacement = config["max_displacement_pct"] * min_xy / 100.0
        # Apply cap if specified
        if "cap_px" in config:
            displacement = min(displacement, config["cap_px"])
        scaled["max_displacement"] = displacement

    if "radius_pct" in config:
        scaled["radius"] = config["radius_pct"] * min_xy / 100.0

    if "n_points" in config:
        scaled["n_points"] = config["n_points"]

    return scaled


def create_deformation_field(
    shape: tuple[int, int, int],
    deform_type: Literal["polynomial", "gaussian", "multi_point", "linear"],
    max_displacement: float,
    seed: int = 42,
    **kwargs,
) -> np.ndarray:
    """Create a displacement field for local deformation.

    Parameters
    ----------
    shape : tuple[int, int, int]
        Volume shape as (Z, Y, X).
    deform_type : str
        Type of deformation: "polynomial", "gaussian", "multi_point", or "linear".
    max_displacement : float
        Maximum displacement in pixels.
    seed : int
        Random seed.
    **kwargs
        Additional parameters (radius for gaussian, n_points for multi_point).

    Returns
    -------
    np.ndarray
        Displacement field with shape (Z, Y, X, 3), where last axis is (dz, dy, dx).
    """
    rng = np.random.default_rng(seed)
    z_size, y_size, x_size = shape

    # Create coordinate grids normalized to [-1, 1]
    z_coords = np.linspace(-1, 1, z_size)
    y_coords = np.linspace(-1, 1, y_size)
    x_coords = np.linspace(-1, 1, x_size)
    zz, yy, xx = np.meshgrid(z_coords, y_coords, x_coords, indexing="ij")

    field = np.zeros((*shape, 3), dtype=np.float32)

    if deform_type == "polynomial":
        coeffs = rng.uniform(-1, 1, size=(3, 6))
        for axis in range(3):
            c = coeffs[axis]
            displacement = (
                c[0]
                + c[1] * xx
                + c[2] * yy
                + c[3] * zz
                + c[4] * xx * yy
                + c[5] * yy * zz
            )
            displacement = displacement / np.abs(displacement).max() * max_displacement
            field[..., axis] = displacement

    elif deform_type == "gaussian":
        radius = kwargs.get("radius", 30)
        margin = 0.3
        center_z = rng.uniform(-1 + margin, 1 - margin)
        center_y = rng.uniform(-1 + margin, 1 - margin)
        center_x = rng.uniform(-1 + margin, 1 - margin)
        direction = rng.normal(size=3)
        direction = direction / np.linalg.norm(direction)
        dist_sq = (
            ((zz - center_z) * z_size / 2) ** 2
            + ((yy - center_y) * y_size / 2) ** 2
            + ((xx - center_x) * x_size / 2) ** 2
        )
        gaussian = np.exp(-dist_sq / (2 * radius**2))
        for axis in range(3):
            field[..., axis] = gaussian * direction[axis] * max_displacement

    elif deform_type == "multi_point":
        n_points = kwargs.get("n_points", 4)
        radius = kwargs.get("radius", 25)
        for _ in range(n_points):
            margin = 0.3
            center_z = rng.uniform(-1 + margin, 1 - margin)
            center_y = rng.uniform(-1 + margin, 1 - margin)
            center_x = rng.uniform(-1 + margin, 1 - margin)
            direction = rng.normal(size=3)
            direction = direction / np.linalg.norm(direction)
            dist_sq = (
                ((zz - center_z) * z_size / 2) ** 2
                + ((yy - center_y) * y_size / 2) ** 2
                + ((xx - center_x) * x_size / 2) ** 2
            )
            gaussian = np.exp(-dist_sq / (2 * radius**2))
            for axis in range(3):
                field[..., axis] += gaussian * direction[axis] * max_displacement

    elif deform_type == "linear":
        coeffs = rng.uniform(-1, 1, size=(3, 5))
        for axis in range(3):
            c = coeffs[axis]
            displacement = c[0] + c[1] * xx + c[2] * yy + c[3] * zz + c[4] * xx * yy
            displacement = displacement / np.abs(displacement).max() * max_displacement
            field[..., axis] = displacement

    return field


# ---------------------------------------------------------------------------
# Coordinate-level transforms
# ---------------------------------------------------------------------------

# Spot tuple format: (z, y, x, intensity, sigma)
SpotTuple = tuple[int, int, int, int, float]


def apply_shift_to_spots(
    spots: list[SpotTuple],
    shift: tuple[int, int, int],
    shape: tuple[int, int, int],
) -> list[SpotTuple]:
    """Shift spot coordinates. Drop spots that move out of bounds.

    Parameters
    ----------
    spots : list[tuple[int, int, int, int, float]]
        List of (z, y, x, intensity, sigma) spot tuples.
    shift : tuple[int, int, int]
        Shift as (dz, dy, dx).
    shape : tuple[int, int, int]
        Volume bounds (Z, Y, X).

    Returns
    -------
    list[tuple[int, int, int, int, float]]
        Shifted spots with out-of-bounds spots removed.
    """
    dz, dy, dx = shift
    z_size, y_size, x_size = shape
    result = []
    for z, y, x, intensity, sigma in spots:
        nz, ny, nx = z + dz, y + dy, x + dx
        if 0 <= nz < z_size and 0 <= ny < y_size and 0 <= nx < x_size:
            result.append((nz, ny, nx, intensity, sigma))
    return result


def apply_deformation_to_spots(
    spots: list[SpotTuple],
    field: np.ndarray,
    shape: tuple[int, int, int],
) -> list[SpotTuple]:
    """Move spots by sampling displacement field at their positions.

    Parameters
    ----------
    spots : list[tuple[int, int, int, int, float]]
        List of (z, y, x, intensity, sigma) spot tuples.
    field : np.ndarray
        Displacement field with shape (Z, Y, X, 3), last axis is (dz, dy, dx).
    shape : tuple[int, int, int]
        Volume bounds (Z, Y, X).

    Returns
    -------
    list[tuple[int, int, int, int, float]]
        Deformed spots with out-of-bounds spots removed.
    """
    z_size, y_size, x_size = shape
    result = []
    for z, y, x, intensity, sigma in spots:
        # Clamp to valid field indices (spot should already be in bounds)
        fz = min(max(z, 0), z_size - 1)
        fy = min(max(y, 0), y_size - 1)
        fx = min(max(x, 0), x_size - 1)
        dz, dy, dx = field[fz, fy, fx]
        nz = int(round(z + dz))
        ny = int(round(y + dy))
        nx = int(round(x + dx))
        if 0 <= nz < z_size and 0 <= ny < y_size and 0 <= nx < x_size:
            result.append((nz, ny, nx, intensity, sigma))
    return result


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

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
    add_noise: bool = True

    # Output dtype
    dtype: Literal["uint8", "uint16"] = "uint8"

    # Registration shifts (for testing registration)
    max_shift_xy: int = 5
    max_shift_z: int = 2

    # Random seed for reproducibility
    seed: int = 42

    # Custom codebook (None = use TEST_CODEBOOK)
    codebook: list[tuple[str, str]] | None = None

    # Deformation (optional, for rounds 2+)
    # None = no deformation, or a key from DEFORMATION_CONFIGS
    deformation: str | None = None


def get_preset_config(
    preset: Literal["tiny", "small", "medium", "large", "tissue", "thick_medium"],
) -> SyntheticConfig:
    """Get predefined configuration for a preset.

    Parameters
    ----------
    preset : {"tiny", "small", "medium", "large", "tissue", "thick_medium"}
        - "tiny": 2 FOVs, 128x128x8, 10 spots (quick tests)
        - "small": 2 FOVs, 256x256x16, 50 spots (unit tests)
        - "medium": 2 FOVs, 512x512x32, 400 spots (integration tests)
        - "large": 2 FOVs, 1024x1024x30, 1500 spots (e2e benchmarking)
        - "tissue": 2 FOVs, 3072x3072x30, 14000 spots (tissue-2D scale)
        - "thick_medium": 2 FOVs, 1024x1024x100, 5200 spots (thick tissue)

    Returns
    -------
    SyntheticConfig
        Configuration for the specified preset.
    """
    presets = {
        "tiny": SyntheticConfig(
            height=128,
            width=128,
            n_z=8,
            n_fovs=2,
            n_spots_per_fov=10,
            max_shift_xy=5,
            max_shift_z=2,
            seed=42,
        ),
        "small": SyntheticConfig(
            height=256,
            width=256,
            n_z=16,
            n_fovs=2,
            n_spots_per_fov=50,
            seed=42,
        ),
        "medium": SyntheticConfig(
            height=512,
            width=512,
            n_z=32,
            n_fovs=2,
            n_spots_per_fov=400,
            max_shift_xy=50,
            max_shift_z=8,
            seed=42,
        ),
        "large": SyntheticConfig(
            height=1024,
            width=1024,
            n_z=30,
            n_fovs=2,
            n_spots_per_fov=1500,
            max_shift_xy=100,
            max_shift_z=7,
            seed=123,
            codebook=generate_codebook(64),
        ),
        "tissue": SyntheticConfig(
            height=3072,
            width=3072,
            n_z=30,
            n_fovs=2,
            n_spots_per_fov=14000,
            max_shift_xy=300,
            max_shift_z=7,
            seed=456,
            codebook=generate_codebook(64),
        ),
        "thick_medium": SyntheticConfig(
            height=1024,
            width=1024,
            n_z=100,
            n_fovs=2,
            n_spots_per_fov=5200,
            max_shift_xy=100,
            max_shift_z=25,
            seed=789,
            codebook=generate_codebook(64),
        ),
    }
    if preset not in presets:
        raise ValueError(f"Unknown preset: {preset}. Choose from: {list(presets.keys())}")
    return presets[preset]


# ---------------------------------------------------------------------------
# Image rendering
# ---------------------------------------------------------------------------

def create_test_image_stack(
    shape: tuple[int, int, int],
    spots: list[tuple[int, int, int, int, float]],
    background: int = 20,
    noise_std: int = 10,
    seed: int | None = None,
    add_noise: bool = True,
    dtype: Literal["uint8", "uint16"] = "uint8",
) -> np.ndarray:
    """Create a single 3D image stack with spots at specified locations.

    Each spot carries its own sigma, enabling per-round PSF variation.

    Parameters
    ----------
    shape : tuple[int, int, int]
        Image shape as (z, y, x).
    spots : list[tuple[int, int, int, int, float]]
        List of (z, y, x, intensity, sigma) tuples for spot locations.
    background : int
        Mean background intensity.
    noise_std : int
        Standard deviation of additive noise.
    seed : int, optional
        Random seed for reproducibility.
    add_noise : bool
        Whether to add Gaussian noise (default True).
    dtype : {"uint8", "uint16"}
        Output data type (default "uint8").

    Returns
    -------
    np.ndarray
        3D image stack with specified dtype.
    """
    rng = np.random.default_rng(seed)

    # Create background with slight variation
    image = rng.normal(background, background / 4, shape).astype(np.float32)

    # Add spots using localized Gaussian kernels
    for z, y, x, intensity, sigma in spots:
        if 0 <= z < shape[0] and 0 <= y < shape[1] and 0 <= x < shape[2]:
            kernel_radius = int(np.ceil(sigma * 4))
            z0, z1 = max(0, z - kernel_radius), min(shape[0], z + kernel_radius + 1)
            y0, y1 = max(0, y - kernel_radius), min(shape[1], y + kernel_radius + 1)
            x0, x1 = max(0, x - kernel_radius), min(shape[2], x + kernel_radius + 1)

            zz, yy, xx = np.ogrid[z0 - z : z1 - z, y0 - y : y1 - y, x0 - x : x1 - x]
            dist_sq = zz**2 + yy**2 + xx**2
            gaussian_spot = intensity * np.exp(-dist_sq / (2 * sigma**2))
            image[z0:z1, y0:y1, x0:x1] += gaussian_spot

    # Add noise (optional)
    if add_noise and noise_std > 0:
        image += rng.normal(0, noise_std, shape)

    # Clip and convert to output dtype
    if dtype == "uint8":
        image = np.clip(image, 0, 255).astype(np.uint8)
    else:
        image = np.clip(image, 0, 65535).astype(np.uint16)

    return image


def create_test_volume(
    shape: tuple[int, int, int],
    n_spots: int = 20,
    spot_intensity: int = 200,
    background: int = 20,
    noise_std: int = 5,
    seed: int | None = None,
) -> np.ndarray:
    """Create a single 3D volume with Gaussian spots for registration testing.

    Parameters
    ----------
    shape : tuple[int, int, int]
        Volume shape as (Z, Y, X).
    n_spots : int
        Number of spots to place (default 20).
    spot_intensity : int
        Peak intensity of spots (default 200).
    background : int
        Background intensity level (default 20).
    noise_std : int
        Standard deviation of Gaussian noise (default 5).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Volume with shape (Z, Y, X) containing spots, dtype uint8.
    """
    rng = np.random.default_rng(seed)
    spot_sigma = 1.5

    z_size, y_size, x_size = shape
    margin_z = max(1, z_size // 8)
    margin_xy = max(5, min(y_size, x_size) // 16)

    spots = []
    for _ in range(n_spots):
        z = int(rng.integers(margin_z, max(margin_z + 1, z_size - margin_z)))
        y = int(rng.integers(margin_xy, max(margin_xy + 1, y_size - margin_xy)))
        x = int(rng.integers(margin_xy, max(margin_xy + 1, x_size - margin_xy)))
        intensity = int(spot_intensity + rng.integers(-20, 21))
        spots.append((z, y, x, intensity, spot_sigma))

    return create_test_image_stack(
        shape=shape,
        spots=spots,
        background=background,
        noise_std=noise_std,
        seed=seed,
        add_noise=True,
        dtype="uint8",
    )


# ---------------------------------------------------------------------------
# Multi-round E2E dataset generation
# ---------------------------------------------------------------------------

def generate_synthetic_dataset(
    output_dir: Path,
    config: SyntheticConfig | None = None,
    preset: str = "small",
) -> dict:
    """Generate a complete synthetic dataset with ground truth.

    Uses coordinate-first rendering: shifts and deformations are applied
    to spot positions before rendering, so images always contain clean
    Gaussian spots without interpolation artifacts.

    Per-round spot variation: each spot gets ~10% intensity jitter and
    ~5% sigma jitter per round, making the synthetic data more realistic.

    Parameters
    ----------
    output_dir : Path
        Directory to write generated files.
    config : SyntheticConfig, optional
        Custom configuration. If None, uses preset defaults.
    preset : str
        Preset configuration name (ignored if config is provided).

    Returns
    -------
    dict
        Ground truth metadata including spot positions and expected barcodes.
    """
    import tifffile

    if config is None:
        config = get_preset_config(preset)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(config.seed)
    shape = (config.n_z, config.height, config.width)

    # Resolve codebook
    codebook = config.codebook if config.codebook is not None else TEST_CODEBOOK

    # Resolve deformation field if configured
    deformation_field = None
    if config.deformation and config.deformation in DEFORMATION_CONFIGS:
        deform_config = DEFORMATION_CONFIGS[config.deformation]
        scaled = scale_deformation_config(deform_config, shape)
        deformation_field = create_deformation_field(
            shape=shape,
            deform_type=scaled["type"],
            max_displacement=scaled["max_displacement"],
            seed=config.seed + 99999,
            **{k: v for k, v in scaled.items() if k not in ["type", "max_displacement"]},
        )

    # Prepare ground truth structure
    ground_truth = {
        "version": "2.0",
        "preset": preset,
        "seed": config.seed,
        "image_shape": list(shape),
        "n_rounds": config.n_rounds,
        "n_channels": config.n_channels,
        "n_genes": len(codebook),
        "fovs": {},
    }

    # Generate each FOV
    for fov_idx in range(config.n_fovs):
        fov_id = f"FOV_{fov_idx + 1:03d}"
        fov_dir = output_dir / fov_id
        fov_dir.mkdir(exist_ok=True)

        # Generate random shifts for each round (round1 is reference)
        shifts = {"round1": [0, 0, 0]}
        for r in range(2, config.n_rounds + 1):
            shifts[f"round{r}"] = [
                int(rng.integers(-config.max_shift_z, config.max_shift_z + 1)),
                int(rng.integers(-config.max_shift_xy, config.max_shift_xy + 1)),
                int(rng.integers(-config.max_shift_xy, config.max_shift_xy + 1)),
            ]

        # Generate random spot positions and gene assignments
        spots_info = []
        margin_z = min(1, config.n_z // 4)
        margin_xy = min(10, config.height // 10)

        for spot_idx in range(config.n_spots_per_fov):
            gene, barcode = codebook[rng.integers(0, len(codebook))]
            color_seq = encode_barcode_to_colors(barcode)

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

        # Generate images for each round and channel using coordinate-first rendering
        for round_idx in range(1, config.n_rounds + 1):
            round_id = f"round{round_idx}"
            round_dir = fov_dir / round_id
            round_dir.mkdir(exist_ok=True)

            shift = tuple(shifts[round_id])

            for ch in range(config.n_channels):
                # Collect spots for this channel with per-round jitter
                channel_spots: list[SpotTuple] = []
                for spot in spots_info:
                    color = spot["color_seq"][round_idx - 1]
                    spot_channel = COLOR_TO_CHANNEL[color]
                    if spot_channel == ch:
                        z, y, x = spot["position"]

                        # Per-round intensity/sigma jitter (deterministic per spot+round)
                        jitter_rng = np.random.default_rng(
                            config.seed + spot["id"] * 100 + round_idx
                        )
                        jittered_intensity = max(1, int(
                            spot["intensity"] * (1 + jitter_rng.normal(0, 0.1))
                        ))
                        jittered_sigma = max(0.5, float(
                            config.spot_sigma * (1 + jitter_rng.normal(0, 0.05))
                        ))

                        channel_spots.append((z, y, x, jittered_intensity, jittered_sigma))

                # Apply coordinate transforms (shift, then deformation)
                if round_idx > 1:
                    channel_spots = apply_shift_to_spots(channel_spots, shift, shape)
                    if config.deformation and deformation_field is not None:
                        channel_spots = apply_deformation_to_spots(
                            channel_spots, deformation_field, shape
                        )

                # Render clean Gaussians at transformed positions
                image = create_test_image_stack(
                    shape=shape,
                    spots=channel_spots,
                    background=config.background_mean,
                    noise_std=config.noise_std,
                    seed=config.seed + fov_idx * 1000 + round_idx * 100 + ch,
                    add_noise=config.add_noise,
                    dtype=config.dtype,
                )

                tiff_path = round_dir / f"ch{ch:02d}.tif"
                tifffile.imwrite(
                    tiff_path,
                    image,
                    imagej=True,
                    metadata={"axes": "ZYX"},
                )

        # Build FOV ground truth
        fov_gt: dict = {
            "shifts": shifts,
            "spots": spots_info,
        }
        if config.deformation and deformation_field is not None:
            # Save deformation field
            field_path = fov_dir / "deformation_field.npy"
            np.save(field_path, deformation_field)
            fov_gt["deformations"] = {
                f"round{r}": {
                    "type": config.deformation,
                    "field_file": f"{fov_id}/deformation_field.npy",
                    "max_displacement": float(np.max(np.linalg.norm(
                        deformation_field, axis=-1
                    ))),
                }
                for r in range(2, config.n_rounds + 1)
            }

        ground_truth["fovs"][fov_id] = fov_gt

    # Write codebook
    codebook_path = output_dir / "codebook.csv"
    with open(codebook_path, "w") as f:
        f.write("gene,barcode\n")
        for gene, barcode in codebook:
            f.write(f"{gene},{barcode}\n")

    # Write ground truth
    gt_path = output_dir / "ground_truth.json"
    with open(gt_path, "w") as f:
        json.dump(ground_truth, f, indent=2)

    # Generate annotated visualization for each FOV
    for fov_id, fov_data in ground_truth["fovs"].items():
        fov_dir = output_dir / fov_id
        _generate_annotated_visualization(
            output_dir=output_dir,
            fov_id=fov_id,
            fov_dir=fov_dir,
            spots=fov_data["spots"],
            image_shape=shape,
            n_channels=config.n_channels,
        )

    return ground_truth


def _generate_annotated_visualization(
    output_dir: Path,
    fov_id: str,
    fov_dir: Path,
    spots: list[dict],
    image_shape: tuple[int, int, int],
    n_channels: int,
) -> None:
    """Generate annotated max projection visualization with spot bounding boxes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import tifffile

    round1_dir = fov_dir / "round1"
    max_proj = None

    for ch in range(n_channels):
        img = tifffile.imread(round1_dir / f"ch{ch:02d}.tif")
        ch_max = img.max(axis=0)
        if max_proj is None:
            max_proj = ch_max.astype(np.float32)
        else:
            max_proj = np.maximum(max_proj, ch_max)

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(max_proj, cmap="gray", vmin=0, vmax=max_proj.max())
    ax.set_title(f"{fov_id} - Round 1 Max Projection (all channels)", fontsize=14)

    unique_genes = sorted(set(s["gene"] for s in spots))
    cmap = plt.cm.get_cmap("tab20", max(len(unique_genes), 8))
    gene_colors = {g: cmap(i % cmap.N) for i, g in enumerate(unique_genes)}

    box_size = 12
    annotate = len(spots) <= 50
    for spot in spots:
        _, y, x = spot["position"]
        gene = spot["gene"]
        color_seq = spot["color_seq"]
        color = gene_colors[gene]

        rect = patches.Rectangle(
            (x - box_size // 2, y - box_size // 2),
            box_size,
            box_size,
            linewidth=1.5,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)

        if annotate:
            label = f"{gene}\n{color_seq}"
            ax.annotate(
                label,
                (x, y - box_size // 2 - 2),
                fontsize=6,
                color=color,
                ha="center",
                va="bottom",
                weight="bold",
            )

    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    ax.set_xlim(0, image_shape[2])
    ax.set_ylim(image_shape[1], 0)

    plt.tight_layout()
    output_path = output_dir / f"ground_truth_annotation_{fov_id}.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Registration benchmark generation (thin wrapper)
# ---------------------------------------------------------------------------

def generate_registration_benchmark(
    output_dir: Path,
    presets: list[str] | None = None,
    seed: int = 42,
    add_noise: bool = True,
) -> dict:
    """Generate synthetic benchmark dataset with ref/mov pairs for registration.

    For each preset, creates a reference volume and multiple moving volumes
    (one with global shift, and one per deformation config). All using
    coordinate-first rendering for clean Gaussian spots.

    Parameters
    ----------
    output_dir : Path
        Output directory for generated data.
    presets : list[str], optional
        List of presets to generate. Defaults to all 6 canonical presets.
    seed : int
        Random seed for reproducibility.
    add_noise : bool
        Whether to add noise to synthetic images.

    Returns
    -------
    dict
        Summary of generated data.
    """
    import tifffile

    output_dir = Path(output_dir)

    if presets is None:
        presets = list(SIZE_PRESETS.keys())

    summary: dict = {"presets": {}, "seed": seed}

    for preset in presets:
        if preset not in SIZE_PRESETS:
            print(f"Warning: Unknown preset '{preset}', skipping")
            continue

        print(f"\nGenerating preset: {preset}")
        shape = SIZE_PRESETS[preset]
        n_spots = SPOT_COUNTS.get(preset, 100)
        shift_range = SHIFT_RANGES.get(preset, {"z": (-5, 5), "yx": (-20, 20)})

        preset_dir = output_dir / "synthetic" / preset
        preset_dir.mkdir(parents=True, exist_ok=True)

        # Generate spot positions (reference)
        spot_sigma = 1.5
        rng_spots = np.random.default_rng(seed)
        z_size, y_size, x_size = shape
        margin_z = 2
        margin_xy = 5

        spot_positions: list[SpotTuple] = []
        for _ in range(n_spots):
            z = int(rng_spots.integers(margin_z, max(margin_z + 1, z_size - margin_z)))
            y = int(rng_spots.integers(margin_xy, max(margin_xy + 1, y_size - margin_xy)))
            x = int(rng_spots.integers(margin_xy, max(margin_xy + 1, x_size - margin_xy)))
            intensity = int(200 + rng_spots.integers(-20, 21))
            spot_positions.append((z, y, x, intensity, spot_sigma))

        # Render and save reference
        print(f"  Creating reference volume {shape}...")
        ref = create_test_image_stack(
            shape=shape,
            spots=spot_positions,
            background=20,
            noise_std=5,
            seed=seed,
            add_noise=add_noise,
        )
        tifffile.imwrite(
            preset_dir / "ref.tif", ref,
            imagej=True, metadata={"axes": "ZYX"},
        )

        ground_truth: dict = {
            "preset": preset,
            "shape": list(shape),
            "n_spots": n_spots,
            "spot_positions": [(z, y, x) for z, y, x, _, _ in spot_positions],
            "seed": seed,
            "pairs": {},
        }

        # Generate shifted moving image using coordinate transform
        print("  Creating shifted moving image...")
        preset_seed = seed + hash(preset) % 10000
        rng_shift = np.random.default_rng(preset_seed)

        z_low, z_high = shift_range["z"]
        yx_low, yx_high = shift_range["yx"]
        z_options = [v for v in range(z_low, z_high + 1) if v != 0]
        z_shift = int(rng_shift.choice(z_options)) if z_options else int(rng_shift.integers(z_low, z_high + 1))
        y_shift = int(rng_shift.integers(yx_low, yx_high + 1))
        x_shift = int(rng_shift.integers(yx_low, yx_high + 1))
        shift = (z_shift, y_shift, x_shift)

        shifted_spots = apply_shift_to_spots(spot_positions, shift, shape)
        mov_shift = create_test_image_stack(
            shape=shape,
            spots=shifted_spots,
            background=20,
            noise_std=5,
            seed=seed + 1,
            add_noise=add_noise,
        )
        tifffile.imwrite(
            preset_dir / "mov_shift.tif", mov_shift,
            imagej=True, metadata={"axes": "ZYX"},
        )
        ground_truth["pairs"]["shift"] = {
            "type": "global_shift",
            "shift_zyx": list(shift),
        }

        # Import inspection image generator
        from starfinder.benchmark.data import generate_inspection_image
        generate_inspection_image(
            ref, mov_shift,
            {"preset": preset, "shift_zyx": list(shift)},
            preset_dir / "inspection_shift.png",
        )

        # Generate deformed moving images
        for deform_name, deform_config in DEFORMATION_CONFIGS.items():
            scaled_config = scale_deformation_config(deform_config, shape)
            print(f"  Creating deformed moving image: {deform_name} (max_disp={scaled_config['max_displacement']:.1f}px)...")

            deform_field = create_deformation_field(
                shape=shape,
                deform_type=scaled_config["type"],
                max_displacement=scaled_config["max_displacement"],
                seed=seed + hash(deform_name) % 10000,
                **{k: v for k, v in scaled_config.items() if k not in ["type", "max_displacement"]},
            )

            deformed_spots = apply_deformation_to_spots(spot_positions, deform_field, shape)
            mov_deform = create_test_image_stack(
                shape=shape,
                spots=deformed_spots,
                background=20,
                noise_std=5,
                seed=seed + 2,
                add_noise=add_noise,
            )

            tifffile.imwrite(
                preset_dir / f"mov_deform_{deform_name}.tif", mov_deform,
                imagej=True, metadata={"axes": "ZYX"},
            )
            np.save(preset_dir / f"field_{deform_name}.npy", deform_field)

            ground_truth["pairs"][deform_name] = {
                "type": "local_deformation",
                "deformation_type": deform_name,
                "max_displacement": round(scaled_config["max_displacement"], 1),
                "field_file": f"field_{deform_name}.npy",
            }

            generate_inspection_image(
                ref, mov_deform,
                {"preset": preset, "deformation_type": deform_name,
                 "max_displacement": round(scaled_config["max_displacement"], 1)},
                preset_dir / f"inspection_deform_{deform_name}.png",
            )

        # Save ground truth
        with open(preset_dir / "ground_truth.json", "w") as f:
            json.dump(ground_truth, f, indent=2)

        summary["presets"][preset] = {
            "shape": list(shape),
            "n_spots": n_spots,
            "n_pairs": 1 + len(DEFORMATION_CONFIGS),
        }
        print(f"  Done: {preset_dir}")

    # Save summary
    summary_dir = output_dir / "synthetic"
    summary_dir.mkdir(parents=True, exist_ok=True)
    with open(summary_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary
