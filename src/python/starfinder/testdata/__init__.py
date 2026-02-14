"""Synthetic test data generation for STARfinder."""

from .synthetic import (
    SyntheticConfig,
    generate_synthetic_dataset,
    get_preset_config,
    create_test_image_stack,
    create_shifted_stack,
    create_test_volume,
)
from .validation import (
    compare_shifts,
    compare_spots,
    compare_genes,
    e2e_summary,
)

__all__ = [
    "SyntheticConfig",
    "generate_synthetic_dataset",
    "get_preset_config",
    "create_test_image_stack",
    "create_shifted_stack",
    "create_test_volume",
    "compare_shifts",
    "compare_spots",
    "compare_genes",
    "e2e_summary",
]
