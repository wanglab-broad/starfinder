"""Private benchmark report formatting; undefined metrics stay undefined."""
from pathlib import Path
import numpy as np
from starfinder.evaluation import EvaluationResult


def _print_quality_report(report: EvaluationResult) -> None:
    """Print supplied metrics without adding thresholds or scientific claims."""
    print("REGISTRATION QUALITY REPORT")
    for name, value in report.values.items():
        rendered = "undefined" if value is None else f"{value:.4f}"
        reason = report.reasons.get(name, "")
        print(f"{name}: {rendered} {report.units[name]} {reason}".rstrip())


def _e2e_summary(shifts, spots, decoding):
    """Select supplied canonical results for a benchmark summary."""
    return {"shift_max_error": shifts.values["max_error"],
            "shift_passed": shifts.values["passed"],
            "spot_recall": spots.values["recall"],
            "spot_precision": spots.values["precision"],
            "spot_mean_distance": spots.values["mean_distance"],
            **decoding.values}


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
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
