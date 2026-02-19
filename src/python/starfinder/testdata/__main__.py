"""CLI entry point for synthetic dataset generation.

Usage:
    python -m starfinder.testdata --preset small --output tests/fixtures/synthetic/small
    python -m starfinder.testdata --preset medium --output tests/fixtures/synthetic/medium
"""

import argparse
from pathlib import Path

from .synthetic import generate_synthetic_dataset, get_preset_config


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic test datasets for STARfinder"
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=["small", "medium", "large", "tissue", "thick_medium"],
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

    print(f"Generating {args.preset} synthetic dataset...")
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


if __name__ == "__main__":
    main()
