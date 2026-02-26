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
    generate_inspection_image,
    generate_overview_grid,
    extract_real_benchmark_data,
    REAL_DATASETS,
)
from starfinder.benchmark.synthetic import (
    SyntheticConfig,
    TEST_CODEBOOK,
    DEFORMATION_CONFIGS,
    generate_codebook,
    encode_barcode_to_colors,
    get_preset_config,
    generate_synthetic_dataset,
    generate_registration_benchmark,
    create_test_image_stack,
    create_test_volume,
    create_deformation_field,
    apply_shift_to_spots,
    apply_deformation_to_spots,
    scale_deformation_config,
)
from starfinder.benchmark.validation import (
    compare_shifts,
    compare_spots,
    compare_genes,
    e2e_summary,
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
    "DEFAULT_BENCHMARK_DIR",
    "BENCHMARK_TASK",
    "SIZE_PRESETS",
    "SPOT_COUNTS",
    "SHIFT_RANGES",
    "DEFORMATION_CONFIGS",
    "get_size_preset",
    # Synthetic data generation
    "SyntheticConfig",
    "TEST_CODEBOOK",
    "generate_codebook",
    "encode_barcode_to_colors",
    "get_preset_config",
    "generate_synthetic_dataset",
    "generate_registration_benchmark",
    "create_test_image_stack",
    "create_test_volume",
    "create_deformation_field",
    "apply_shift_to_spots",
    "apply_deformation_to_spots",
    "scale_deformation_config",
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
