"""Private historical real-data extraction; explicit dataset configuration.

No extraction is executed by run/evaluate/report. Scientific qualification and
any revision of this legacy conversion recipe remain outside housekeeping.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tifffile
from ._reporting import generate_inspection_image


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
    datasets: dict[str, dict],
) -> dict:
    """Extract real dataset round1/round2 pairs for benchmarking.

    Structure expected: ``dataset_path/round{N}/fov/*_ch0{N}.tif``

    Parameters
    ----------
    output_dir : Path
        Output directory for extracted data.
    datasets : dict
        Explicit name-to-path/FOV/channel configuration; no institutional defaults.

    Returns
    -------
    dict
        Summary of extracted data.
    """
    output_dir = Path(output_dir)
    real_dir = output_dir / "real"
    real_dir.mkdir(parents=True, exist_ok=True)

    summary = {"datasets": {}}

    for dataset_name, config in datasets.items():
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
