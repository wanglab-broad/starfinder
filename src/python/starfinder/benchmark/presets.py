"""Benchmark presets for standard test configurations."""

from __future__ import annotations

from pathlib import Path

# Root benchmark directory and current task
DEFAULT_BENCHMARK_DIR = Path(
    "/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark"
)
BENCHMARK_TASK = "registration"

# Standard volume size presets (Z, Y, X)
SIZE_PRESETS: dict[str, tuple[int, int, int]] = {
    "tiny": (8, 128, 128),
    "small": (16, 256, 256),
    "medium": (32, 512, 512),
    "large": (30, 1024, 1024),
    "tissue": (30, 3072, 3072),        # tissue-2D size
    "thick_medium": (100, 1024, 1024),  # thick tissue, medium XY
}

# Spot density: approximately 50 spots per 10^6 voxels
SPOT_COUNTS: dict[str, int] = {
    "tiny": 10,
    "small": 50,
    "medium": 400,
    "large": 1500,
    "tissue": 14000,
    "thick_medium": 5200,
}

# Shift ranges for global registration testing (≤25% of each dimension)
SHIFT_RANGES: dict[str, dict[str, tuple[int, int]]] = {
    "tiny": {"z": (-2, 2), "yx": (-10, 10)},
    "small": {"z": (-4, 4), "yx": (-25, 25)},
    "medium": {"z": (-8, 8), "yx": (-50, 50)},
    "large": {"z": (-7, 7), "yx": (-100, 100)},
    "tissue": {"z": (-7, 7), "yx": (-300, 300)},
    "thick_medium": {"z": (-25, 25), "yx": (-100, 100)},
}


def get_size_preset(name: str) -> tuple[int, int, int]:
    """
    Get volume size for a preset name.

    Args:
        name: Preset name (tiny, small, medium, large, xlarge, tissue).

    Returns:
        Tuple of (Z, Y, X) dimensions.

    Raises:
        ValueError: If preset name is unknown.
    """
    if name not in SIZE_PRESETS:
        raise ValueError(
            f"Unknown size preset: '{name}'. "
            f"Available: {list(SIZE_PRESETS.keys())}"
        )
    return SIZE_PRESETS[name]
