"""Benchmark data generation for registration testing.

This module generates synthetic and real benchmark datasets with known
ground truth for evaluating registration algorithms.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import tifffile
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt

from starfinder.benchmark.presets import SIZE_PRESETS, SPOT_COUNTS, SHIFT_RANGES


# Deformation configurations (as percentage of smallest XY dimension)
# These will be scaled to actual pixels based on image size, with optional caps
DEFORMATION_CONFIGS = {
    "polynomial_small": {"type": "polynomial", "max_displacement_pct": 3.0, "cap_px": 15.0},
    "polynomial_large": {"type": "polynomial", "max_displacement_pct": 6.0, "cap_px": 30.0},
    "gaussian_small": {"type": "gaussian", "max_displacement_pct": 3.0, "radius_pct": 6.0, "cap_px": 15.0},
    "gaussian_large": {"type": "gaussian", "max_displacement_pct": 6.0, "radius_pct": 10.0, "cap_px": 30.0},
    "multi_point": {"type": "multi_point", "max_displacement_pct": 4.0, "n_points": 4, "radius_pct": 5.0, "cap_px": 20.0},
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


@dataclass
class BenchmarkDataConfig:
    """Configuration for benchmark data generation."""

    output_dir: Path
    presets: list[str] = field(default_factory=lambda: ["tiny", "small", "medium"])
    seed: int = 42
    spot_intensity: int = 200
    background: int = 20
    noise_std: int = 5
    add_noise: bool = True


def create_benchmark_volume(
    shape: tuple[int, int, int],
    n_spots: int,
    seed: int = 42,
    spot_intensity: int = 200,
    background: int = 20,
    noise_std: int = 5,
    add_noise: bool = True,
) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    """Create a synthetic 3D volume with Gaussian spots.

    Parameters
    ----------
    shape : tuple[int, int, int]
        Volume shape as (Z, Y, X).
    n_spots : int
        Number of spots to generate.
    seed : int
        Random seed for reproducibility.
    spot_intensity : int
        Peak intensity of spots.
    background : int
        Background intensity level.
    noise_std : int
        Standard deviation of Gaussian noise.
    add_noise : bool
        Whether to add noise.

    Returns
    -------
    volume : np.ndarray
        Volume with shape (Z, Y, X), dtype uint8.
    spot_positions : list[tuple[int, int, int]]
        List of (z, y, x) spot center positions.
    """
    rng = np.random.default_rng(seed)
    z_size, y_size, x_size = shape

    # Generate random spot positions with fixed margin from edges
    margin_z = 2
    margin_xy = 5

    spot_positions = []
    for _ in range(n_spots):
        z = int(rng.integers(margin_z, max(margin_z + 1, z_size - margin_z)))
        y = int(rng.integers(margin_xy, max(margin_xy + 1, y_size - margin_xy)))
        x = int(rng.integers(margin_xy, max(margin_xy + 1, x_size - margin_xy)))
        spot_positions.append((z, y, x))

    # Create background
    volume = rng.normal(background, background / 4, shape).astype(np.float32)

    # Add spots using localized Gaussian kernels
    spot_sigma = 1.5
    kernel_radius = int(np.ceil(spot_sigma * 4))

    for z, y, x in spot_positions:
        # Add intensity variation
        intensity = spot_intensity + rng.integers(-20, 21)

        # Define local patch bounds
        z0, z1 = max(0, z - kernel_radius), min(z_size, z + kernel_radius + 1)
        y0, y1 = max(0, y - kernel_radius), min(y_size, y + kernel_radius + 1)
        x0, x1 = max(0, x - kernel_radius), min(x_size, x + kernel_radius + 1)

        # Create coordinate grids relative to spot center
        zz, yy, xx = np.ogrid[z0 - z : z1 - z, y0 - y : y1 - y, x0 - x : x1 - x]

        # Compute 3D Gaussian
        dist_sq = zz**2 + yy**2 + xx**2
        gaussian_spot = intensity * np.exp(-dist_sq / (2 * spot_sigma**2))

        volume[z0:z1, y0:y1, x0:x1] += gaussian_spot

    # Add noise
    if add_noise and noise_std > 0:
        volume += rng.normal(0, noise_std, shape)

    # Clip and convert to uint8
    volume = np.clip(volume, 0, 255).astype(np.uint8)

    return volume, spot_positions


def apply_global_shift(
    volume: np.ndarray,
    shift: tuple[int, int, int],
) -> np.ndarray:
    """Apply a global shift to a volume with zero-padding (no wrap-around).

    Parameters
    ----------
    volume : np.ndarray
        Input volume (Z, Y, X).
    shift : tuple[int, int, int]
        Shift as (dz, dy, dx). Positive values shift content toward higher indices.

    Returns
    -------
    np.ndarray
        Shifted volume (same shape and dtype). Regions that shift out of bounds
        are replaced with zeros.
    """
    dz, dy, dx = shift
    z_size, y_size, x_size = volume.shape

    # Create output array filled with zeros
    shifted = np.zeros_like(volume)

    # Compute source and destination slices for each axis
    # For positive shift: content moves to higher indices, low indices become zero
    # For negative shift: content moves to lower indices, high indices become zero

    # Z axis
    if dz >= 0:
        src_z = slice(0, max(0, z_size - dz))
        dst_z = slice(dz, z_size)
    else:
        src_z = slice(-dz, z_size)
        dst_z = slice(0, max(0, z_size + dz))

    # Y axis
    if dy >= 0:
        src_y = slice(0, max(0, y_size - dy))
        dst_y = slice(dy, y_size)
    else:
        src_y = slice(-dy, y_size)
        dst_y = slice(0, max(0, y_size + dy))

    # X axis
    if dx >= 0:
        src_x = slice(0, max(0, x_size - dx))
        dst_x = slice(dx, x_size)
    else:
        src_x = slice(-dx, x_size)
        dst_x = slice(0, max(0, x_size + dx))

    # Copy the overlapping region
    shifted[dst_z, dst_y, dst_x] = volume[src_z, src_y, src_x]

    return shifted


def create_deformation_field(
    shape: tuple[int, int, int],
    deform_type: Literal["polynomial", "gaussian", "multi_point"],
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
        Type of deformation: "polynomial", "gaussian", or "multi_point".
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
        # Smooth polynomial warping
        # Randomly generate polynomial coefficients
        coeffs = rng.uniform(-1, 1, size=(3, 6))  # 6 terms per axis

        for axis in range(3):
            c = coeffs[axis]
            # Polynomial: c0 + c1*x + c2*y + c3*z + c4*x*y + c5*y*z
            displacement = (
                c[0]
                + c[1] * xx
                + c[2] * yy
                + c[3] * zz
                + c[4] * xx * yy
                + c[5] * yy * zz
            )
            # Normalize to max_displacement
            displacement = displacement / np.abs(displacement).max() * max_displacement
            field[..., axis] = displacement

    elif deform_type == "gaussian":
        # Single localized Gaussian bump
        radius = kwargs.get("radius", 30)

        # Random center position (away from edges)
        margin = 0.3
        center_z = rng.uniform(-1 + margin, 1 - margin)
        center_y = rng.uniform(-1 + margin, 1 - margin)
        center_x = rng.uniform(-1 + margin, 1 - margin)

        # Random direction
        direction = rng.normal(size=3)
        direction = direction / np.linalg.norm(direction)

        # Compute distance from center
        dist_sq = (
            ((zz - center_z) * z_size / 2) ** 2
            + ((yy - center_y) * y_size / 2) ** 2
            + ((xx - center_x) * x_size / 2) ** 2
        )

        # Gaussian falloff
        gaussian = np.exp(-dist_sq / (2 * radius**2))

        for axis in range(3):
            field[..., axis] = gaussian * direction[axis] * max_displacement

    elif deform_type == "multi_point":
        # Multiple independent Gaussian bumps
        n_points = kwargs.get("n_points", 4)
        radius = kwargs.get("radius", 25)

        for _ in range(n_points):
            # Random center
            margin = 0.3
            center_z = rng.uniform(-1 + margin, 1 - margin)
            center_y = rng.uniform(-1 + margin, 1 - margin)
            center_x = rng.uniform(-1 + margin, 1 - margin)

            # Random direction
            direction = rng.normal(size=3)
            direction = direction / np.linalg.norm(direction)

            # Compute distance from center
            dist_sq = (
                ((zz - center_z) * z_size / 2) ** 2
                + ((yy - center_y) * y_size / 2) ** 2
                + ((xx - center_x) * x_size / 2) ** 2
            )

            gaussian = np.exp(-dist_sq / (2 * radius**2))

            for axis in range(3):
                field[..., axis] += gaussian * direction[axis] * max_displacement

    return field


def apply_deformation_field(
    volume: np.ndarray,
    field: np.ndarray,
) -> np.ndarray:
    """Apply a displacement field to warp a volume.

    Parameters
    ----------
    volume : np.ndarray
        Input volume (Z, Y, X).
    field : np.ndarray
        Displacement field (Z, Y, X, 3).

    Returns
    -------
    np.ndarray
        Warped volume (same shape and dtype as input).
    """
    from scipy.ndimage import map_coordinates

    z_size, y_size, x_size = volume.shape

    # Create coordinate grids
    z_coords, y_coords, x_coords = np.meshgrid(
        np.arange(z_size),
        np.arange(y_size),
        np.arange(x_size),
        indexing="ij",
    )

    # Add displacement
    new_z = z_coords + field[..., 0]
    new_y = y_coords + field[..., 1]
    new_x = x_coords + field[..., 2]

    # Interpolate
    coords = np.array([new_z, new_y, new_x])
    warped = map_coordinates(
        volume.astype(np.float32),
        coords,
        order=1,  # Linear interpolation
        mode="constant",
        cval=0,
    )

    return warped.astype(volume.dtype)


def generate_inspection_image(
    ref: np.ndarray,
    mov: np.ndarray,
    ground_truth: dict,
    output_path: Path,
) -> None:
    """Generate a green-magenta composite inspection image.

    Parameters
    ----------
    ref : np.ndarray
        Reference volume (Z, Y, X).
    mov : np.ndarray
        Moving volume (Z, Y, X).
    ground_truth : dict
        Ground truth metadata.
    output_path : Path
        Path to save the inspection image.
    """
    # Maximum intensity projections
    ref_mip = np.max(ref, axis=0).astype(np.float32)
    mov_mip = np.max(mov, axis=0).astype(np.float32)

    # Normalize to [0, 1]
    ref_norm = ref_mip / ref_mip.max() if ref_mip.max() > 0 else ref_mip
    mov_norm = mov_mip / mov_mip.max() if mov_mip.max() > 0 else mov_mip

    # Green-magenta composite (green=ref, magenta=mov)
    composite = np.stack([mov_norm, ref_norm, mov_norm], axis=-1)

    # Create figure
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(ref_mip, cmap="gray")
    axes[0].set_title(f"Reference MIP\nShape: {ref.shape}")
    axes[0].axis("off")

    axes[1].imshow(mov_mip, cmap="gray")
    axes[1].set_title(f"Moving MIP\nShape: {mov.shape}")
    axes[1].axis("off")

    axes[2].imshow(composite)
    axes[2].set_title("Overlay (G=ref, M=mov)\nWhite=aligned, Color=misaligned")
    axes[2].axis("off")

    # Ground truth info
    info_lines = [f"Dataset: {ground_truth.get('preset', 'N/A')}"]
    if "shift_zyx" in ground_truth:
        info_lines.append(f"Shift (Z,Y,X): {ground_truth['shift_zyx']}")
    if "deformation_type" in ground_truth:
        info_lines.append(f"Deformation: {ground_truth['deformation_type']}")
        info_lines.append(f"Max disp: {ground_truth.get('max_displacement', 'N/A')} px")

    info_text = "\n".join(info_lines)
    axes[3].text(
        0.1, 0.5, info_text,
        fontsize=12, family="monospace",
        verticalalignment="center",
        transform=axes[3].transAxes,
    )
    axes[3].axis("off")
    axes[3].set_title("Ground Truth")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_synthetic_benchmark(
    output_dir: Path,
    presets: list[str] | None = None,
    seed: int = 42,
    add_noise: bool = True,
) -> dict:
    """Generate synthetic benchmark dataset with all ref/mov pairs.

    Parameters
    ----------
    output_dir : Path
        Output directory for generated data.
    presets : list[str], optional
        List of presets to generate. Defaults to all except thick_large.
    seed : int
        Random seed for reproducibility.
    add_noise : bool
        Whether to add noise to synthetic images.

    Returns
    -------
    dict
        Summary of generated data.
    """
    output_dir = Path(output_dir)

    if presets is None:
        # Default: all except thick_large (too large for quick testing)
        presets = ["tiny", "small", "medium", "large", "xlarge", "tissue", "thick_medium"]

    summary = {"presets": {}, "seed": seed}

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

        # Generate reference volume
        print(f"  Creating reference volume {shape}...")
        ref, spot_positions = create_benchmark_volume(
            shape=shape,
            n_spots=n_spots,
            seed=seed,
            add_noise=add_noise,
        )

        # Save reference
        tifffile.imwrite(
            preset_dir / "ref.tif",
            ref,
            imagej=True,
            metadata={"axes": "ZYX"},
        )

        # Ground truth base
        ground_truth = {
            "preset": preset,
            "shape": list(shape),
            "n_spots": n_spots,
            "spot_positions": spot_positions,
            "seed": seed,
            "pairs": {},
        }

        # Generate shifted moving image
        print("  Creating shifted moving image...")
        # Use preset-specific seed to get different shifts for each preset
        preset_seed = seed + hash(preset) % 10000
        rng = np.random.default_rng(preset_seed)

        # Generate non-zero shifts (exclude 0 from range if possible)
        z_low, z_high = shift_range["z"]
        yx_low, yx_high = shift_range["yx"]

        # For Z: pick from non-zero values if range allows
        z_options = [v for v in range(z_low, z_high + 1) if v != 0]
        if z_options:
            z_shift = int(rng.choice(z_options))
        else:
            z_shift = int(rng.integers(z_low, z_high + 1))

        # For Y and X: standard random integers
        y_shift = int(rng.integers(yx_low, yx_high + 1))
        x_shift = int(rng.integers(yx_low, yx_high + 1))

        shift = (z_shift, y_shift, x_shift)
        mov_shift = apply_global_shift(ref, shift)

        tifffile.imwrite(
            preset_dir / "mov_shift.tif",
            mov_shift,
            imagej=True,
            metadata={"axes": "ZYX"},
        )

        ground_truth["pairs"]["shift"] = {
            "type": "global_shift",
            "shift_zyx": list(shift),
        }

        # Generate inspection image for shift
        generate_inspection_image(
            ref, mov_shift,
            {"preset": preset, "shift_zyx": list(shift)},
            preset_dir / "inspection_shift.png",
        )

        # Generate deformed moving images
        for deform_name, deform_config in DEFORMATION_CONFIGS.items():
            # Scale deformation parameters to image size
            scaled_config = scale_deformation_config(deform_config, shape)
            print(f"  Creating deformed moving image: {deform_name} (max_disp={scaled_config['max_displacement']:.1f}px)...")

            field = create_deformation_field(
                shape=shape,
                deform_type=scaled_config["type"],
                max_displacement=scaled_config["max_displacement"],
                seed=seed + hash(deform_name) % 10000,
                **{k: v for k, v in scaled_config.items() if k not in ["type", "max_displacement"]},
            )

            mov_deform = apply_deformation_field(ref, field)

            # Save deformed image and field
            tifffile.imwrite(
                preset_dir / f"mov_deform_{deform_name}.tif",
                mov_deform,
                imagej=True,
                metadata={"axes": "ZYX"},
            )
            np.save(preset_dir / f"field_{deform_name}.npy", field)

            ground_truth["pairs"][deform_name] = {
                "type": "local_deformation",
                "deformation_type": deform_name,
                "max_displacement": round(scaled_config["max_displacement"], 1),
                "field_file": f"field_{deform_name}.npy",
            }

            # Generate inspection image
            generate_inspection_image(
                ref, mov_deform,
                {"preset": preset, "deformation_type": deform_name,
                 "max_displacement": round(scaled_config["max_displacement"], 1)},
                preset_dir / f"inspection_deform_{deform_name}.png",
            )

        # Save ground truth JSON
        with open(preset_dir / "ground_truth.json", "w") as f:
            json.dump(ground_truth, f, indent=2)

        summary["presets"][preset] = {
            "shape": list(shape),
            "n_spots": n_spots,
            "n_pairs": 1 + len(DEFORMATION_CONFIGS),  # 1 shift + N deformations
        }

        print(f"  Done: {preset_dir}")

    # Save summary
    with open(output_dir / "synthetic" / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def generate_overview_grid(output_dir: Path) -> None:
    """Generate an overview grid of all inspection images.

    Parameters
    ----------
    output_dir : Path
        Output directory containing synthetic/ subdirectory.
    """
    from glob import glob

    synthetic_dir = Path(output_dir) / "synthetic"
    inspection_files = sorted(glob(str(synthetic_dir / "*" / "inspection_*.png")))

    if not inspection_files:
        print("No inspection images found")
        return

    # Load images
    images = []
    labels = []
    for f in inspection_files:
        img = plt.imread(f)
        images.append(img)
        # Extract preset and type from path
        parts = Path(f).parts
        preset = parts[-2]
        name = Path(f).stem.replace("inspection_", "")
        labels.append(f"{preset}\n{name}")

    # Create grid
    n_images = len(images)
    n_cols = min(4, n_images)
    n_rows = (n_images + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = np.atleast_2d(axes)

    for idx, (img, label) in enumerate(zip(images, labels)):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].imshow(img)
        axes[row, col].set_title(label, fontsize=8)
        axes[row, col].axis("off")

    # Hide unused axes
    for idx in range(n_images, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].axis("off")

    plt.tight_layout()
    fig.savefig(output_dir / "overview.png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved overview: {output_dir / 'overview.png'}")


# Real dataset configurations
# Structure: dataset_path/round{N}/fov/*_ch0{N}.tif
REAL_DATASETS = {
    "cell_culture_3D": {
        "path": "/home/unix/jiahao/wanglab/Data/Processed/sample-dataset/cell-culture-3D",
        "fov": "Position351",
        "n_channels": 4,  # ch00-ch03
        "expected_shape": (30, 1496, 1496),
    },
    "tissue_2D": {
        "path": "/home/unix/jiahao/wanglab/Data/Processed/sample-dataset/tissue-2D",
        "fov": "tile_1",
        "n_channels": 4,  # ch00-ch03
        "expected_shape": (30, 3072, 3072),
    },
    "LN": {
        "path": "/home/unix/jiahao/wanglab/Data/Processed/sample-dataset/LN",
        "fov": "Position001",
        "n_channels": 4,  # ch00-ch03
        "expected_shape": (50, 1496, 1496),
    },
}


def _load_round_mip(round_dir: Path, n_channels: int) -> np.ndarray:
    """Load a round and compute MIP across channels.

    Parameters
    ----------
    round_dir : Path
        Directory containing channel TIFF files (*_ch0N.tif pattern).
    n_channels : int
        Number of channels to load.

    Returns
    -------
    np.ndarray
        Maximum intensity projection across channels (Z, Y, X).
    """
    from glob import glob

    stacks = []
    for ch in range(n_channels):
        # Find channel file using glob pattern
        pattern = str(round_dir / f"*_ch{ch:02d}.tif")
        matches = glob(pattern)
        if not matches:
            raise FileNotFoundError(f"No file matching {pattern}")
        if len(matches) > 1:
            print(f"  Warning: Multiple files match {pattern}, using first")

        stack = tifffile.imread(matches[0])
        stacks.append(stack)

    # Stack channels and compute MIP
    multi_channel = np.stack(stacks, axis=-1)  # (Z, Y, X, C)
    mip = np.max(multi_channel, axis=-1)  # (Z, Y, X)

    return mip


def extract_real_benchmark_data(
    output_dir: Path,
    datasets: list[str] | None = None,
) -> dict:
    """Extract real dataset round1/round2 pairs for benchmarking.

    Structure expected: dataset_path/round{N}/fov/*_ch0{N}.tif

    Parameters
    ----------
    output_dir : Path
        Output directory for extracted data.
    datasets : list[str], optional
        List of dataset names to extract. Defaults to all available.

    Returns
    -------
    dict
        Summary of extracted data.
    """
    output_dir = Path(output_dir)
    real_dir = output_dir / "real"
    real_dir.mkdir(parents=True, exist_ok=True)

    if datasets is None:
        datasets = list(REAL_DATASETS.keys())

    summary = {"datasets": {}}

    for dataset_name in datasets:
        if dataset_name not in REAL_DATASETS:
            print(f"Warning: Unknown dataset '{dataset_name}', skipping")
            continue

        config = REAL_DATASETS[dataset_name]
        print(f"\nExtracting: {dataset_name}")

        dataset_dir = real_dir / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        base_path = Path(config["path"])
        fov = config["fov"]
        n_channels = config["n_channels"]

        try:
            # Load round1 (reference)
            # Structure: dataset/round1/fov/
            round1_path = base_path / "round1" / fov
            print(f"  Loading round1 from {round1_path}...")
            ref = _load_round_mip(round1_path, n_channels)
            print(f"  Reference shape: {ref.shape}, dtype: {ref.dtype}")

            # Load round2 (moving)
            round2_path = base_path / "round2" / fov
            print(f"  Loading round2 from {round2_path}...")
            mov = _load_round_mip(round2_path, n_channels)
            print(f"  Moving shape: {mov.shape}, dtype: {mov.dtype}")

            # Convert to uint8 if needed
            if ref.dtype != np.uint8:
                ref = (ref / ref.max() * 255).astype(np.uint8) if ref.max() > 0 else ref.astype(np.uint8)
            if mov.dtype != np.uint8:
                mov = (mov / mov.max() * 255).astype(np.uint8) if mov.max() > 0 else mov.astype(np.uint8)

            # Save as TIFF
            tifffile.imwrite(
                dataset_dir / "ref.tif",
                ref,
                imagej=True,
                metadata={"axes": "ZYX"},
            )
            tifffile.imwrite(
                dataset_dir / "mov.tif",
                mov,
                imagej=True,
                metadata={"axes": "ZYX"},
            )

            # Save metadata
            metadata = {
                "dataset": dataset_name,
                "fov": fov,
                "ref_round": "round1",
                "mov_round": "round2",
                "shape": list(ref.shape),
                "n_channels": n_channels,
                "ground_truth_shift": None,  # Unknown for real data
            }
            with open(dataset_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            # Generate inspection image
            generate_inspection_image(
                ref, mov,
                {"preset": dataset_name, "type": "real", "rounds": "round1→round2"},
                dataset_dir / "inspection.png",
            )

            summary["datasets"][dataset_name] = {
                "shape": list(ref.shape),
                "status": "success",
            }
            print(f"  Done: {dataset_dir}")

        except FileNotFoundError as e:
            print(f"  Error: {e}")
            summary["datasets"][dataset_name] = {"status": f"error: {e}"}
        except Exception as e:
            print(f"  Error: {e}")
            summary["datasets"][dataset_name] = {"status": f"error: {e}"}

    # Save summary
    with open(real_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary
