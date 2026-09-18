"""Unified evaluation for registration benchmark results (Phase 2).

Loads reference + registered images from disk, computes quality metrics
using the same code path for all backends, and generates inspection artifacts.

This module ensures fair comparison between backends (Python, MATLAB, etc.)
by applying identical metric computation to all registered images.

Usage:
    # Evaluate a single registered image
    # Pure metrics: starfinder.evaluation.registration.evaluate_registration

    # Evaluate all results in a backend directory tree
    results = evaluate_directory(result_dir, data_dir)

    # CLI: evaluate a backend tree
    uv run python -m starfinder.benchmark.evaluate <result_dir> [--data-dir ...]
"""

from __future__ import annotations

import json
from pathlib import Path

from starfinder.spot_finding import find_spots, PercentileCentroidConfig
from starfinder.image import ImageMetadata

import numpy as np
import tifffile

from starfinder.benchmark.presets import BENCHMARK_TASK, DEFAULT_BENCHMARK_DIR

# Default task-specific benchmark directory (derived from shared constants)
DEFAULT_BENCHMARK_TASK_DIR = DEFAULT_BENCHMARK_DIR / BENCHMARK_TASK

# Known real datasets (to distinguish from synthetic presets)
REAL_DATASETS = {"cell_culture_3D", "tissue_2D", "LN"}

# Synthetic presets
SYNTHETIC_PRESETS = {
    "tiny", "small", "medium", "large", "xlarge", "tissue", "thick_medium",
}


def _evaluate_images(ref, mov_before, registered, use_mip=False):
    """Benchmark adapter: explicitly prepare legacy detections, then evaluate.

    Historical percentile choices remain here, outside pure evaluation. Volume
    NCC is retained for both policies. Constant reference range uses explicit 1.
    """
    from dataclasses import asdict
    from starfinder.evaluation.registration import evaluate_registration

    images = [ref, mov_before, registered]
    domain = [x.max(axis=0)[None, ...] for x in images] if use_mip else images
    percentile = 99.0 if use_mip else 99.5
    metadata = ImageMetadata("benchmark/reference-grid")
    spots = [find_spots(x, config=PercentileCentroidConfig(99.5),
                       metadata=metadata, spot_namespace=f"evaluation/{i}")
             .spots[["z", "y", "x"]].to_numpy() for i, x in enumerate(domain)]
    masks = [x > np.percentile(x, percentile) for x in domain]
    data_range = float(np.ptp(domain[0])) or 1.0
    report = evaluate_registration(
        *images, reference_spots=spots[0], before_spots=spots[1], after_spots=spots[2],
        reference_mask=masks[0], before_mask=masks[1], after_mask=masks[2],
        reference_metadata=metadata, before_metadata=metadata, after_metadata=metadata,
        data_range=data_range, ssim_policy="mip" if use_mip else "volume",
        matching_policy="greedy", match_threshold=2.0, units="voxel")
    return {**report.values,
            **{k: v for k, v in report.counts.items() if k.startswith("n_spots")},
            "ssim_method": "mip" if use_mip else "3d",
            "spot_method": "mip" if use_mip else "3d",
            "evaluation": asdict(report),
            "detection_config": {"percentile": 99.5, "mask_percentile": percentile, "method": "percentile_centroid"}}


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
    def metric_text(value):
        return "undefined" if value is None else f"{value:.3f}"
    for key, label in (("ncc", "NCC"), ("ssim", "SSIM"),
                       ("spot_iou", "IoU"), ("match_rate", "MR")):
        info_lines.append(f"{label}: {metric_text(metadata.get(key + '_before'))} -> "
                          f"{metric_text(metadata.get(key + '_after'))}")

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
    metrics = _evaluate_images(ref, mov, registered, use_mip=use_mip)

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
    data_dir: Path = DEFAULT_BENCHMARK_TASK_DIR / "data",
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
        "--data-dir", type=Path, default=DEFAULT_BENCHMARK_TASK_DIR / "data",
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
