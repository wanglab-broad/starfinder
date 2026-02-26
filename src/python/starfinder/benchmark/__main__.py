"""CLI entry point for synthetic dataset and benchmark generation.

Usage:
    # E2E multi-round dataset (default mode)
    uv run python -m starfinder.benchmark --preset small --output tests/fixtures/synthetic/small

    # Registration benchmark pairs
    uv run python -m starfinder.benchmark --mode registration --output benchmark_data/
"""

import argparse
from pathlib import Path

from .synthetic import (
    generate_synthetic_dataset,
    generate_registration_benchmark,
    get_preset_config,
)
from .presets import SIZE_PRESETS


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic test datasets and benchmarks for STARfinder"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["e2e", "registration"],
        default="e2e",
        help="Generation mode: 'e2e' for multi-round pipeline datasets, "
        "'registration' for ref/mov benchmark pairs (default: e2e)",
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=list(SIZE_PRESETS.keys()),
        default="small",
        help="Preset configuration (default: small)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for generated dataset",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--no-noise",
        action="store_true",
        help="Disable background noise for cleaner spot visualization",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["uint8", "uint16"],
        default="uint8",
        help="Output image dtype (default: uint8)",
    )

    args = parser.parse_args()

    if args.mode == "e2e":
        print(f"Generating {args.preset} synthetic dataset (e2e mode)...")
        config = get_preset_config(args.preset)
        config.seed = args.seed
        config.add_noise = not args.no_noise
        config.dtype = args.dtype

        ground_truth = generate_synthetic_dataset(
            output_dir=args.output,
            config=config,
            preset=args.preset,
        )

        n_fovs = len(ground_truth["fovs"])
        n_spots = sum(len(fov["spots"]) for fov in ground_truth["fovs"].values())
        print(f"Generated {n_fovs} FOV(s) with {n_spots} total spots")
        print(f"Output: {args.output}")
        print(f"Ground truth: {args.output / 'ground_truth.json'}")

    elif args.mode == "registration":
        print(f"Generating registration benchmark (preset: {args.preset})...")
        summary = generate_registration_benchmark(
            output_dir=args.output,
            presets=[args.preset],
            seed=args.seed,
            add_noise=not args.no_noise,
        )
        print(f"Output: {args.output}")
        for preset, info in summary.get("presets", {}).items():
            print(f"  {preset}: {info['shape']}, {info['n_pairs']} pairs")


if __name__ == "__main__":
    main()
