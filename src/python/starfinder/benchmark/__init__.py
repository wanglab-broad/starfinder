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
    SIZE_PRESETS,
    SPOT_COUNTS,
    SHIFT_RANGES,
    get_size_preset,
)
from starfinder.benchmark.evaluate import (
    evaluate_registration,
    evaluate_directory,
    evaluate_single,
    generate_inspection,
)
from starfinder.benchmark.data import (
    create_benchmark_volume,
    apply_global_shift,
    create_deformation_field,
    apply_deformation_field,
    generate_inspection_image,
    generate_synthetic_benchmark,
    generate_overview_grid,
    extract_real_benchmark_data,
    DEFORMATION_CONFIGS,
    REAL_DATASETS,
)

__all__ = [
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
    "SIZE_PRESETS",
    "SPOT_COUNTS",
    "SHIFT_RANGES",
    "DEFORMATION_CONFIGS",
    "get_size_preset",
    # Data generation
    "create_benchmark_volume",
    "apply_global_shift",
    "create_deformation_field",
    "apply_deformation_field",
    "generate_inspection_image",
    "generate_synthetic_benchmark",
    "generate_overview_grid",
    "extract_real_benchmark_data",
    "REAL_DATASETS",
    # Evaluation (Phase 2)
    "evaluate_registration",
    "evaluate_directory",
    "evaluate_single",
    "generate_inspection",
    # Reporting
    "print_table",
    "save_csv",
    "save_json",
]
