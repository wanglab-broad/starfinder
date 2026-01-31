"""Registration-specific benchmark utilities.

This module provides convenience functions for benchmarking registration
methods using the generic starfinder.benchmark framework.
"""

from __future__ import annotations

import numpy as np

from starfinder.benchmark import (
    BenchmarkResult,
    SIZE_PRESETS,
    measure,
)
from starfinder.registration import phase_correlate, phase_correlate_skimage


def benchmark_registration(
    sizes: list[tuple[int, int, int]] | None = None,
    methods: list[str] | None = None,
    n_runs: int = 5,
    seed: int = 42,
) -> list[BenchmarkResult]:
    """
    Benchmark registration methods with synthetic images.

    Args:
        sizes: List of (Z, Y, X) sizes. Defaults to tiny, small, medium.
        methods: List of method names ("numpy", "skimage"). Defaults to both.
        n_runs: Number of runs per measurement.
        seed: Random seed for reproducibility.

    Returns:
        List of BenchmarkResult objects.
    """
    from starfinder.testdata import create_test_volume

    if sizes is None:
        sizes = [
            SIZE_PRESETS["tiny"],
            SIZE_PRESETS["small"],
            SIZE_PRESETS["medium"],
        ]

    if methods is None:
        methods = ["numpy", "skimage"]

    method_funcs = {
        "numpy": phase_correlate,
        "skimage": phase_correlate_skimage,
    }

    rng = np.random.default_rng(seed)
    results = []

    for size in sizes:
        # Generate synthetic volume
        fixed = create_test_volume(
            shape=size,
            n_spots=20,
            spot_intensity=200,
            background=20,
            seed=seed,
        )

        # Apply known shift
        known_shift = (
            int(rng.integers(-5, 6)),
            int(rng.integers(-10, 11)),
            int(rng.integers(-10, 11)),
        )
        moving = np.roll(fixed, known_shift, axis=(0, 1, 2))

        for method_name in methods:
            func = method_funcs[method_name]

            # Warm-up
            _ = func(fixed, moving)

            # Timed runs
            times = []
            memories = []
            detected_shift = None

            for _ in range(n_runs):
                detected_shift, elapsed, mem = measure(lambda: func(fixed, moving))
                times.append(elapsed)
                memories.append(mem)

            # Compute shift error
            error = np.sqrt(
                sum((d - e) ** 2 for d, e in zip(detected_shift, known_shift))
            )

            results.append(
                BenchmarkResult(
                    method=method_name,
                    operation="phase_correlate",
                    size=size,
                    time_seconds=float(np.mean(times)),
                    memory_mb=float(np.mean(memories)),
                    metrics={
                        "shift_error": error,
                        "known_shift": known_shift,
                        "detected_shift": detected_shift,
                    },
                )
            )

    return results


# Re-export for backwards compatibility
def run_benchmark(*args, **kwargs):
    """Deprecated: Use benchmark_registration() instead."""
    return benchmark_registration(*args, **kwargs)


def print_benchmark_table(results: list[BenchmarkResult]) -> None:
    """Print registration benchmark results with shift error."""
    print()
    print("| Method  | Size           | Time (s) | Memory (MB) | Shift Error |")
    print("|---------|----------------|----------|-------------|-------------|")

    for r in results:
        size_str = f"{r.size[1]}×{r.size[2]}×{r.size[0]}"
        error = r.metrics.get("shift_error", 0.0)
        print(
            f"| {r.method:<7} | {size_str:<14} | {r.time_seconds:>8.3f} | "
            f"{r.memory_mb:>11.1f} | {error:>11.2f} |"
        )

    print()
