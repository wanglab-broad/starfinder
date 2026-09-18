"""Benchmark utilities for performance measurement and comparison."""

from starfinder.benchmark.core import (
    BenchmarkResult,
    benchmark,
    measure,
)
from starfinder.benchmark.runner import (
    BenchmarkPair,
    BenchmarkSuite,
    PRESET_ORDER,
    RegistrationBenchmarkRunner,
    RegistrationResult,
    run_comparison,
    timeout_handler,
)
from starfinder.benchmark.report import (
    print_table,
    save_csv,
    save_json,
)
from starfinder.benchmark.presets import (
    DEFAULT_BENCHMARK_DIR,
    BENCHMARK_TASK,
)
from starfinder.benchmark.evaluate import (
    evaluate_registration,
    evaluate_directory,
    evaluate_single,
    generate_inspection,
)
from starfinder.benchmark.data import (
    generate_inspection_image,
    generate_overview_grid,
    extract_real_benchmark_data,
    REAL_DATASETS,
)
from starfinder.benchmark.validation import (
    compare_shifts,
    compare_spots,
    compare_genes,
    e2e_summary,
)

from ._registration import run_benchmark

__all__ = [
    "run_benchmark",
    # Core
    "BenchmarkResult",
    "BenchmarkSuite",
    "benchmark",
    "measure",
    "run_comparison",
    # Registration benchmark runner
    "RegistrationBenchmarkRunner",
    "RegistrationResult",
    "BenchmarkPair",
    "PRESET_ORDER",
    "timeout_handler",
    # Presets
    "DEFAULT_BENCHMARK_DIR",
    "BENCHMARK_TASK",
    # Visualization & real data
    "generate_inspection_image",
    "generate_overview_grid",
    "extract_real_benchmark_data",
    "REAL_DATASETS",
    # Evaluation (Phase 2)
    "evaluate_registration",
    "evaluate_directory",
    "evaluate_single",
    "generate_inspection",
    # Validation (e2e comparison)
    "compare_shifts",
    "compare_spots",
    "compare_genes",
    "e2e_summary",
    # Reporting
    "print_table",
    "save_csv",
    "save_json",
]
