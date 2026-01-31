"""Benchmark utilities for performance measurement and comparison."""

from starfinder.benchmark.core import (
    BenchmarkResult,
    benchmark,
    measure,
)
from starfinder.benchmark.runner import (
    BenchmarkSuite,
    run_comparison,
)

__all__ = [
    "BenchmarkResult",
    "BenchmarkSuite",
    "benchmark",
    "measure",
    "run_comparison",
]
