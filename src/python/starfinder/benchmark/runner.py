"""Benchmark runner for comparing multiple methods."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from starfinder.benchmark.core import BenchmarkResult, measure


def run_comparison(
    methods: dict[str, Callable],
    inputs: list[Any],
    operation: str,
    n_runs: int = 1,
    warmup: bool = False,
) -> list[BenchmarkResult]:
    """
    Run multiple methods on multiple inputs and collect results.

    Args:
        methods: Dict mapping method name to callable.
        inputs: List of inputs to pass to each method.
        operation: Name of the operation being benchmarked.
        n_runs: Number of runs per (method, input) pair for averaging.
        warmup: If True, run each method once before timing.

    Returns:
        List of BenchmarkResult objects.

    Example:
        >>> results = run_comparison(
        ...     methods={"numpy": np.sum, "python": sum},
        ...     inputs=[list(range(100)), list(range(1000))],
        ...     operation="sum",
        ... )
    """
    results = []

    for inp in inputs:
        for method_name, func in methods.items():
            # Optional warmup
            if warmup:
                _ = func(inp)

            # Collect runs
            times = []
            memories = []
            return_value = None

            for _ in range(n_runs):
                ret, elapsed, mem = measure(lambda f=func, i=inp: f(i))
                times.append(elapsed)
                memories.append(mem)
                return_value = ret

            # Average results
            avg_time = float(np.mean(times))
            avg_mem = float(np.mean(memories))

            # Extract size if input has shape
            size: tuple[int, ...] = ()
            if hasattr(inp, "shape"):
                size = tuple(inp.shape)

            results.append(
                BenchmarkResult(
                    method=method_name,
                    operation=operation,
                    size=size,
                    time_seconds=avg_time,
                    memory_mb=avg_mem,
                    metrics={"return_value": return_value},
                )
            )

    return results


class BenchmarkSuite:
    """Collection of benchmark results with aggregation utilities.

    Attributes:
        name: Name of the benchmark suite.
        results: List of collected BenchmarkResult objects.
    """

    def __init__(self, name: str):
        self.name = name
        self.results: list[BenchmarkResult] = []

    def add(self, result: BenchmarkResult) -> None:
        """Add a result to the suite."""
        self.results.append(result)

    def summary(self) -> dict[str, float]:
        """Compute summary statistics across all results.

        Returns:
            Dict with mean_time, min_time, max_time, mean_memory.
        """
        if not self.results:
            return {}

        times = [r.time_seconds for r in self.results]
        memories = [r.memory_mb for r in self.results]

        return {
            "mean_time": float(np.mean(times)),
            "min_time": float(np.min(times)),
            "max_time": float(np.max(times)),
            "std_time": float(np.std(times)),
            "mean_memory": float(np.mean(memories)),
        }

    def filter(
        self, method: str | None = None, operation: str | None = None
    ) -> list[BenchmarkResult]:
        """Filter results by method and/or operation."""
        filtered = self.results
        if method is not None:
            filtered = [r for r in filtered if r.method == method]
        if operation is not None:
            filtered = [r for r in filtered if r.operation == operation]
        return filtered
