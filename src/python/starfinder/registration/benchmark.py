"""Benchmark utilities for registration methods."""

from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from starfinder.registration import phase_correlate, phase_correlate_skimage


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    method: str
    size: tuple[int, int, int]
    time_sec: float
    memory_peak_mb: float
    shift_error: float


def _measure_registration(
    func: Callable,
    fixed: np.ndarray,
    moving: np.ndarray,
    expected_shift: tuple[float, float, float],
    n_runs: int = 5,
) -> tuple[float, float, float]:
    """
    Measure time, memory, and accuracy for a registration function.

    Returns:
        Tuple of (mean_time_sec, peak_memory_mb, shift_error).
    """
    # Warm-up run
    _ = func(fixed, moving)

    # Timed runs
    times = []
    for _ in range(n_runs):
        tracemalloc.start()
        start = time.perf_counter()

        detected = func(fixed, moving)

        elapsed = time.perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        times.append(elapsed)

    mean_time = np.mean(times)
    peak_mb = peak / (1024 * 1024)

    # Compute shift error (L2 distance)
    error = np.sqrt(sum((d - e) ** 2 for d, e in zip(detected, expected_shift)))

    return mean_time, peak_mb, error


def run_benchmark(
    sizes: list[tuple[int, int, int]] | None = None,
    methods: list[str] | None = None,
    n_runs: int = 5,
    seed: int = 42,
) -> list[BenchmarkResult]:
    """
    Run registration benchmark with synthetic images.

    Args:
        sizes: List of (Z, Y, X) sizes to test. Defaults to standard set.
        methods: List of methods ("numpy", "skimage"). Defaults to both.
        n_runs: Number of runs per measurement.
        seed: Random seed for reproducibility.

    Returns:
        List of BenchmarkResult objects.
    """
    from starfinder.testdata import create_test_volume

    if sizes is None:
        sizes = [
            (5, 128, 128),    # tiny
            (10, 256, 256),   # small
            (30, 512, 512),   # medium
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
        nz, ny, nx = size

        # Generate synthetic volume with spots
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

        for method in methods:
            func = method_funcs[method]
            mean_time, peak_mb, error = _measure_registration(
                func, fixed, moving, known_shift, n_runs
            )

            results.append(
                BenchmarkResult(
                    method=method,
                    size=size,
                    time_sec=mean_time,
                    memory_peak_mb=peak_mb,
                    shift_error=error,
                )
            )

    return results


def print_benchmark_table(results: list[BenchmarkResult]) -> None:
    """Print benchmark results as formatted table."""
    print()
    print("| Method  | Size           | Time (s) | Memory (MB) | Shift Error |")
    print("|---------|----------------|----------|-------------|-------------|")

    for r in results:
        size_str = f"{r.size[1]}×{r.size[2]}×{r.size[0]}"
        print(
            f"| {r.method:<7} | {size_str:<14} | {r.time_sec:>8.3f} | {r.memory_peak_mb:>11.1f} | {r.shift_error:>11.2f} |"
        )

    print()
