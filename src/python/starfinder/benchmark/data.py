"""Benchmark data utilities: inspection images and real data extraction.

Synthetic data generation has moved to ``starfinder.benchmark.synthetic``.
This module retains visualization helpers and real dataset extraction.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tifffile
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt


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
                dataset_dir / "ref.tif", ref,
                imagej=True, metadata={"axes": "ZYX"},
            )
            tifffile.imwrite(
                dataset_dir / "mov.tif", mov,
                imagej=True, metadata={"axes": "ZYX"},
            )

            # Save metadata
            metadata = {
                "dataset": dataset_name,
                "fov": fov,
                "ref_round": "round1",
                "mov_round": "round2",
                "shape": list(ref.shape),
                "n_channels": n_channels,
                "ground_truth_shift": None,
            }
            with open(dataset_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            # Generate inspection image
            generate_inspection_image(
                ref, mov,
                {"preset": dataset_name, "type": "real", "rounds": "round1\u2192round2"},
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
