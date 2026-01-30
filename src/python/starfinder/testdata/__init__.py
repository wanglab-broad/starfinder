"""Synthetic test data generation for STARfinder."""

from .synthetic import (
    SyntheticConfig,
    generate_synthetic_dataset,
    get_preset_config,
    create_test_image_stack,
    create_shifted_stack,
)

__all__ = [
    "SyntheticConfig",
    "generate_synthetic_dataset",
    "get_preset_config",
    "create_test_image_stack",
    "create_shifted_stack",
]
