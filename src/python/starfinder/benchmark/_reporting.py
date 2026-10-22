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
