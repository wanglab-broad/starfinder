"""CLI entry point for synthetic dataset generation.

Usage:
    python -m starfinder.testing.generate --preset mini --output tests/fixtures/synthetic/mini
    python -m starfinder.testing.generate --preset standard --output tests/fixtures/synthetic/standard
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
        choices=["mini", "standard"],
        default="mini",
        help="Preset configuration (default: mini)",
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

    args = parser.parse_args()

    print(f"Generating {args.preset} synthetic dataset...")
    config = get_preset_config(args.preset)
    config.seed = args.seed

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
