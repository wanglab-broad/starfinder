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
from starfinder.benchmark.presets import (
    SIZE_PRESETS,
    get_size_preset,
)

__all__ = [
    "BenchmarkResult",
    "BenchmarkSuite",
    "SIZE_PRESETS",
    "benchmark",
    "get_size_preset",
    "measure",
    "print_table",
    "run_comparison",
    "save_csv",
    "save_json",
]
