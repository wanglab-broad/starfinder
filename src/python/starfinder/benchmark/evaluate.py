"""Unified evaluation for registration benchmark results (Phase 2).

Loads reference + registered images from disk, computes quality metrics
using the same code path for all backends, and generates inspection artifacts.

This module ensures fair comparison between backends (Python, MATLAB, etc.)
by applying identical metric computation to all registered images.

Usage:
    # Evaluate a single registered image
    metrics = evaluate_registration(ref, mov_before, registered)

    # Evaluate all results in a backend directory tree
    results = evaluate_directory(result_dir, data_dir)

    # CLI: evaluate a backend tree
    uv run python -m starfinder.benchmark.evaluate <result_dir> [--data-dir ...]
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tifffile


# Default benchmark data location
DEFAULT_BENCHMARK_DIR = Path(
    "/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark"
)

# Known real datasets (to distinguish from synthetic presets)
REAL_DATASETS = {"cell_culture_3D", "tissue_2D", "LN"}

# Synthetic presets
SYNTHETIC_PRESETS = {
    "tiny", "small", "medium", "large", "xlarge", "tissue", "thick_medium",
}


def evaluate_registration(
    ref: np.ndarray,
    mov_before: np.ndarray,
    registered: np.ndarray,
    use_mip: bool = False,
) -> dict:
    """Compute registration quality metrics.

    This is the core Phase 2 function. It computes all quality metrics
    between the reference and registered images, using the same code path
    regardless of which backend produced the registered image.

    Args:
        ref: Reference volume (Z, Y, X).
        mov_before: Original moving volume before registration.
        registered: Registered volume (Z, Y, X).
        use_mip: If True, compute SSIM and spot metrics on 2D MIP instead
            of full 3D volume. Much faster for large volumes.

    Returns:
        Flat dict with before/after metrics:
        - ncc_before, ncc_after
        - ssim_before, ssim_after
        - ssim_method: "mip" if computed on MIP, "3d" if full volume
        - spot_iou_before, spot_iou_after
        - spot_dice_before, spot_dice_after
        - match_rate_before, match_rate_after
        - match_distance_before, match_distance_after
        - n_spots_ref, n_spots_before, n_spots_after
    """
    from starfinder.registration.metrics import registration_quality_report

    if use_mip:
        # Compute metrics on 2D MIP (fast path for large volumes)
        from starfinder.registration.metrics import (
            detect_spots,
            normalized_cross_correlation,
            spot_colocalization,
            spot_matching_accuracy,
            structural_similarity,
        )

        ncc_before = normalized_cross_correlation(ref, mov_before)
        ncc_after = normalized_cross_correlation(ref, registered)

        # MIP: collapse Z axis to get (Y, X) images
        ref_mip = np.max(ref, axis=0)
        mov_mip = np.max(mov_before, axis=0)
        reg_mip = np.max(registered, axis=0)

        # SSIM on MIP
        ssim_before = structural_similarity(ref_mip, mov_mip)
        ssim_after = structural_similarity(ref_mip, reg_mip)

        # Spot metrics on MIP (avoids 3D label/center_of_mass on huge volumes)
        coloc_before = spot_colocalization(ref_mip, mov_mip)
        coloc_after = spot_colocalization(ref_mip, reg_mip)

        ref_spots = detect_spots(ref_mip)
        before_spots = detect_spots(mov_mip)
        after_spots = detect_spots(reg_mip)

        match_before = spot_matching_accuracy(ref_spots, before_spots)
        match_after = spot_matching_accuracy(ref_spots, after_spots)

        return {
            "ncc_before": ncc_before,
            "ncc_after": ncc_after,
            "ssim_before": ssim_before,
            "ssim_after": ssim_after,
            "ssim_method": "mip",
            "spot_iou_before": coloc_before["iou"],
            "spot_iou_after": coloc_after["iou"],
            "spot_dice_before": coloc_before["dice"],
            "spot_dice_after": coloc_after["dice"],
            "match_rate_before": match_before["match_rate"],
            "match_rate_after": match_after["match_rate"],
            "match_distance_before": match_before["mean_distance"],
            "match_distance_after": match_after["mean_distance"],
            "n_spots_ref": len(ref_spots),
            "n_spots_before": len(before_spots),
            "n_spots_after": len(after_spots),
            "spot_method": "mip",
        }

    # Full report including SSIM
    report = registration_quality_report(ref, mov_before, registered)

    # Flatten nested structure: {'ncc': {'before': 0.5}} -> {'ncc_before': 0.5}
    flat = {}
    for metric, values in report.items():
        if isinstance(values, dict):
            for key, val in values.items():
                flat[f"{metric}_{key}"] = val
        else:
            flat[metric] = values

    flat["ssim_method"] = "3d"
    flat["spot_method"] = "3d"
    return flat


def generate_inspection(
    ref: np.ndarray,
    mov: np.ndarray,
    registered: np.ndarray,
    metadata: dict,
    output_path: Path,
) -> None:
    """Generate a before/after registration inspection image.

    Creates a 5-panel figure:
    1. Before overlay (green=ref, magenta=mov)
    2. After overlay (green=ref, magenta=registered)
    3. Difference before (hot colormap)
    4. Difference after (hot colormap)
    5. Metadata text panel

    Args:
        ref: Reference volume (Z, Y, X).
        mov: Original moving volume (Z, Y, X).
        registered: Registered volume (Z, Y, X).
        metadata: Dict with preset, method, metrics info for text panel.
        output_path: Path to save the inspection PNG.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    # Maximum intensity projections
    ref_mip = np.max(ref, axis=0).astype(np.float32)
    mov_mip = np.max(mov, axis=0).astype(np.float32)
    reg_mip = np.max(registered, axis=0).astype(np.float32)

    # Normalize to [0, 1]
    def normalize(img):
        return img / img.max() if img.max() > 0 else img

    ref_norm = normalize(ref_mip)
    mov_norm = normalize(mov_mip)
    reg_norm = normalize(reg_mip)

    # Green-magenta composites
    before_composite = np.stack([mov_norm, ref_norm, mov_norm], axis=-1)
    after_composite = np.stack([reg_norm, ref_norm, reg_norm], axis=-1)

    # Create figure with GridSpec for tight layout
    fig = plt.figure(figsize=(18, 4))
    gs = GridSpec(1, 5, figure=fig, width_ratios=[1, 1, 1, 1, 0.6], wspace=0.02)

    # Panel 1: Before registration
    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(before_composite)
    ax0.set_title("Before\n(G=ref, M=mov)", fontsize=9)
    ax0.axis("off")

    # Panel 2: After registration
    ax1 = fig.add_subplot(gs[1])
    ax1.imshow(after_composite)
    ax1.set_title("After\n(G=ref, M=reg)", fontsize=9)
    ax1.axis("off")

    # Panel 3: Difference before
    diff_before = np.abs(ref_norm - mov_norm)
    ax2 = fig.add_subplot(gs[2])
    ax2.imshow(diff_before, cmap="hot", vmin=0, vmax=0.5)

    threshold = np.percentile(np.maximum(ref_norm, mov_norm), 90)
    signal_mask = (ref_norm > threshold) | (mov_norm > threshold)
    n_signal = signal_mask.sum()

    if n_signal > 0:
        mad_bright_before = diff_before[signal_mask].mean()
        ax2.set_title(f"Diff Before\nMAD(bright)={mad_bright_before:.3f}", fontsize=9)
    else:
        ax2.set_title(f"Diff Before\nMAD={diff_before.mean():.4f}", fontsize=9)
    ax2.axis("off")

    # Panel 4: Difference after
    diff_after = np.abs(ref_norm - reg_norm)
    ax3 = fig.add_subplot(gs[3])
    ax3.imshow(diff_after, cmap="hot", vmin=0, vmax=0.5)

    if n_signal > 0:
        mad_bright_after = diff_after[signal_mask].mean()
        if mad_bright_before > 0:
            improvement = (mad_bright_before - mad_bright_after) / mad_bright_before * 100
            ax3.set_title(
                f"Diff After\nMAD(bright)={mad_bright_after:.3f} ({improvement:+.0f}%)",
                fontsize=9,
            )
        else:
            ax3.set_title(f"Diff After\nMAD(bright)={mad_bright_after:.3f}", fontsize=9)
    else:
        ax3.set_title(f"Diff After\nMAD={diff_after.mean():.4f}", fontsize=9)
    ax3.axis("off")

    # Panel 5: Metadata text
    ax4 = fig.add_subplot(gs[4])
    ax4.axis("off")

    info_lines = [
        f"Dataset: {metadata.get('dataset', 'N/A')}",
        f"Backend: {metadata.get('backend', 'N/A')}",
        f"Method:  {metadata.get('method', 'N/A')}",
        f"Pair:    {metadata.get('pair_type', 'N/A')}",
        f"Status:  {metadata.get('status', 'N/A')}",
    ]
    if metadata.get("time_seconds") is not None:
        info_lines.append(f"Time:    {metadata['time_seconds']:.2f}s")
    if metadata.get("ncc_after") is not None:
        ncc_before = metadata.get("ncc_before", 0) or 0
        info_lines.append(f"NCC:     {ncc_before:.3f} -> {metadata['ncc_after']:.3f}")
    if metadata.get("ssim_after") is not None:
        ssim_before = metadata.get("ssim_before", 0) or 0
        info_lines.append(f"SSIM:    {ssim_before:.3f} -> {metadata['ssim_after']:.3f}")
    if metadata.get("spot_iou_after") is not None:
        iou_before = metadata.get("spot_iou_before", 0) or 0
        info_lines.append(f"IoU:     {iou_before:.3f} -> {metadata['spot_iou_after']:.3f}")
    if metadata.get("match_rate_after") is not None:
        mr_before = metadata.get("match_rate_before", 0) or 0
        info_lines.append(f"MR:      {mr_before:.3f} -> {metadata['match_rate_after']:.3f}")

    ax4.text(
        0.05, 0.95, "\n".join(info_lines),
        transform=ax4.transAxes,
        fontsize=8,
        family="monospace",
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def _resolve_data_paths(
    registered_path: Path,
    data_dir: Path,
) -> tuple[Path, Path, str, str, str]:
    """Resolve ref/mov paths from a registered image path.

    Infers dataset name from directory structure and maps to the
    corresponding data directory.

    Args:
        registered_path: Path to a registered_*.tif file.
        data_dir: Root data directory (containing synthetic/ and real/).

    Returns:
        (ref_path, mov_path, dataset, pair_type, data_category)

    Raises:
        FileNotFoundError: If ref/mov files cannot be found.
    """
    dataset = registered_path.parent.name

    # Determine data category
    if dataset in REAL_DATASETS:
        data_category = "real"
    elif dataset in SYNTHETIC_PRESETS:
        data_category = "synthetic"
    else:
        # Try checking filesystem
        if (data_dir / "real" / dataset).exists():
            data_category = "real"
        elif (data_dir / "synthetic" / dataset).exists():
            data_category = "synthetic"
        else:
            raise FileNotFoundError(
                f"Cannot find dataset '{dataset}' in {data_dir}/synthetic/ or {data_dir}/real/"
            )

    dataset_data_dir = data_dir / data_category / dataset
    ref_path = dataset_data_dir / "ref.tif"

    # Determine pair_type from run JSON if available
    pair_type = "shift"  # default for global registration
    run_json = _find_run_json(registered_path)
    if run_json is not None:
        with open(run_json) as f:
            run_data = json.load(f)
        pair_type = run_data.get("pair_type", pair_type)

    # Resolve moving image path
    if data_category == "real":
        mov_path = dataset_data_dir / "mov.tif"
    elif pair_type == "shift":
        mov_path = dataset_data_dir / "mov_shift.tif"
    else:
        mov_path = dataset_data_dir / f"mov_deform_{pair_type}.tif"

    if not ref_path.exists():
        raise FileNotFoundError(f"Reference not found: {ref_path}")
    if not mov_path.exists():
        raise FileNotFoundError(f"Moving image not found: {mov_path}")

    return ref_path, mov_path, dataset, pair_type, data_category


def _find_run_json(registered_path: Path) -> Path | None:
    """Find the run JSON corresponding to a registered TIFF.

    Looks for run_*.json matching the backend label in the filename.
    e.g., registered_python.tif -> run_python.json

    Also checks legacy naming: result_*.json
    """
    stem = registered_path.stem  # e.g., "registered_python"
    backend = stem.replace("registered_", "")
    parent = registered_path.parent

    # New naming convention
    run_path = parent / f"run_{backend}.json"
    if run_path.exists():
        return run_path

    # Legacy naming convention
    result_path = parent / f"result_{backend}.json"
    if result_path.exists():
        return result_path

    return None


def _metrics_path_for(registered_path: Path) -> Path:
    """Get the expected metrics JSON path for a registered TIFF."""
    stem = registered_path.stem
    backend = stem.replace("registered_", "")
    return registered_path.parent / f"metrics_{backend}.json"


def _inspection_path_for(registered_path: Path) -> Path:
    """Get the expected inspection PNG path for a registered TIFF."""
    stem = registered_path.stem
    backend = stem.replace("registered_", "")
    return registered_path.parent / f"inspection_{backend}.png"


def evaluate_single(
    registered_path: Path,
    data_dir: Path,
    force: bool = False,
    use_mip: bool = False,
    generate_insp: bool = True,
) -> dict | None:
    """Evaluate a single registered image from disk.

    Loads ref, mov, and registered from disk, computes metrics,
    and saves metrics JSON + inspection PNG alongside the registered TIFF.

    Args:
        registered_path: Path to registered_*.tif.
        data_dir: Root data directory (containing synthetic/ and real/).
        force: If True, re-evaluate even if metrics already exist.
        use_mip: If True, compute SSIM and spot metrics on 2D MIP.
        generate_insp: If True, generate inspection PNG.

    Returns:
        Metrics dict, or None if skipped.
    """
    metrics_path = _metrics_path_for(registered_path)
    inspection_path = _inspection_path_for(registered_path)

    # Skip if already evaluated
    if not force and metrics_path.exists():
        return None

    # Resolve data paths
    ref_path, mov_path, dataset, pair_type, data_category = _resolve_data_paths(
        registered_path, data_dir
    )

    # Extract backend from filename
    backend = registered_path.stem.replace("registered_", "")

    print(f"  Evaluating {dataset}/{backend}...", end=" ", flush=True)

    # Load images
    ref = tifffile.imread(str(ref_path))
    mov = tifffile.imread(str(mov_path))
    registered = tifffile.imread(str(registered_path))

    # Compute metrics
    metrics = evaluate_registration(ref, mov, registered, use_mip=use_mip)

    # Load run metadata if available
    run_json = _find_run_json(registered_path)
    run_data = {}
    if run_json is not None:
        with open(run_json) as f:
            run_data = json.load(f)

    # Build full metrics output
    output = {
        "dataset": dataset,
        "backend": backend,
        "pair_type": pair_type,
        "data_category": data_category,
        **metrics,
        "time_seconds": run_data.get("internal_time") or run_data.get("time_seconds"),
        "memory_mb": run_data.get("memory_mb"),
        "shifts_zyx": run_data.get("shifts_zyx") or run_data.get("shift_detected"),
        "status": run_data.get("status", "success"),
    }

    # Save metrics JSON
    with open(metrics_path, "w") as f:
        json.dump(output, f, indent=2, default=_json_default)

    # Generate inspection image
    if generate_insp:
        generate_inspection(ref, mov, registered, output, inspection_path)

    print("OK")
    return output


def evaluate_directory(
    result_dir: Path,
    data_dir: Path = DEFAULT_BENCHMARK_DIR / "data",
    force: bool = False,
    use_mip_above: int = 100_000_000,
    generate_insp: bool = True,
) -> list[dict]:
    """Batch-evaluate all registered images in a backend directory tree.

    Scans result_dir/{dataset}/ for registered_*.tif files.
    For each, computes metrics and saves metrics JSON + inspection PNG.

    Args:
        result_dir: Backend result directory (e.g., global_python/).
        data_dir: Root data directory containing synthetic/ and real/.
        force: If True, re-evaluate even if metrics already exist.
        use_mip_above: Use 2D MIP for SSIM and spot metrics on volumes
            larger than this many voxels.
        generate_insp: If True, generate inspection PNGs.

    Returns:
        List of metrics dicts for all evaluated files.
    """
    result_dir = Path(result_dir)
    data_dir = Path(data_dir)

    # Find all registered TIFFs
    registered_files = sorted(result_dir.glob("*/registered_*.tif"))

    if not registered_files:
        print(f"No registered_*.tif files found in {result_dir}/*/")
        return []

    print(f"Found {len(registered_files)} registered images in {result_dir.name}/")

    results = []
    for reg_path in registered_files:
        try:
            # Check volume size for SSIM skip decision
            # Read shape without loading full array
            with tifffile.TiffFile(str(reg_path)) as tif:
                shape = tif.pages[0].shape
                n_pages = len(tif.pages)
                n_voxels = n_pages * shape[0] * shape[1]

            use_mip = n_voxels > use_mip_above

            result = evaluate_single(
                reg_path, data_dir,
                force=force,
                use_mip=use_mip,
                generate_insp=generate_insp,
            )
            if result is not None:
                results.append(result)
        except Exception as e:
            print(f"  ERROR evaluating {reg_path.parent.name}/{reg_path.name}: {e}")

    # Generate summary CSV
    if results:
        _save_summary(results, result_dir / "summary.csv")

    skipped = len(registered_files) - len(results)
    print(f"\nEvaluated: {len(results)}, Skipped: {skipped}")
    return results


def _save_summary(results: list[dict], output_path: Path) -> None:
    """Save evaluation results to a summary CSV."""
    import pandas as pd

    # Select key columns for summary
    columns = [
        "dataset", "backend", "pair_type", "data_category", "status",
        "time_seconds", "memory_mb",
        "ncc_before", "ncc_after",
        "ssim_before", "ssim_after",
        "spot_iou_before", "spot_iou_after",
        "match_rate_before", "match_rate_after",
        "n_spots_ref", "n_spots_after",
    ]

    records = []
    for r in results:
        records.append({k: r.get(k) for k in columns})

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print(f"Saved summary: {output_path}")


def _json_default(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate registration results (Phase 2)",
        usage="uv run python -m starfinder.benchmark.evaluate <result_dir> [options]",
    )
    parser.add_argument(
        "result_dir", type=Path,
        help="Backend result directory containing {dataset}/registered_*.tif",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=DEFAULT_BENCHMARK_DIR / "data",
        help="Root data directory with synthetic/ and real/ subdirs",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-evaluate even if metrics already exist",
    )
    parser.add_argument(
        "--no-inspection", action="store_true",
        help="Skip inspection PNG generation",
    )
    parser.add_argument(
        "--use-mip-above", type=int, default=100_000_000,
        help="Use 2D MIP for SSIM and spot metrics above this voxel count (default: 100M)",
    )

    args = parser.parse_args()

    results = evaluate_directory(
        args.result_dir,
        data_dir=args.data_dir,
        force=args.force,
        use_mip_above=args.use_mip_above,
        generate_insp=not args.no_inspection,
    )

    if results:
        print(f"\nDone. {len(results)} results evaluated.")
    else:
        print("\nNo new results to evaluate (use --force to re-evaluate).")
