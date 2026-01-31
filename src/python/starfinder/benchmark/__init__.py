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
from starfinder.benchmark.report import (
    print_table,
    save_csv,
    save_json,
)

__all__ = [
    "BenchmarkResult",
    "BenchmarkSuite",
    "benchmark",
    "measure",
    "print_table",
    "run_comparison",
    "save_csv",
    "save_json",
]
