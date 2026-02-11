"""Benchmark presets for standard test configurations."""

from __future__ import annotations

# Standard volume size presets (Z, Y, X)
SIZE_PRESETS: dict[str, tuple[int, int, int]] = {
    "tiny": (8, 128, 128),
    "small": (16, 256, 256),
    "medium": (32, 512, 512),
    "large": (30, 1024, 1024),
    "xlarge": (30, 1496, 1496),      # cell-culture-3D size
    "tissue": (30, 3072, 3072),      # tissue-2D size
    "thick_medium": (100, 1024, 1024),  # thick tissue, medium XY
    "thick_large": (200, 2722, 2722),   # thick tissue, large XY
}

# Spot density: approximately 50 spots per 10^6 voxels
SPOT_COUNTS: dict[str, int] = {
    "tiny": 10,
    "small": 50,
    "medium": 400,
    "large": 1500,
    "xlarge": 3400,
    "tissue": 14000,
    "thick_medium": 5200,
    "thick_large": 74000,
}

# Shift ranges for global registration testing (≤25% of each dimension)
SHIFT_RANGES: dict[str, dict[str, tuple[int, int]]] = {
    "tiny": {"z": (-2, 2), "yx": (-10, 10)},
    "small": {"z": (-4, 4), "yx": (-25, 25)},
    "medium": {"z": (-8, 8), "yx": (-50, 50)},
    "large": {"z": (-7, 7), "yx": (-100, 100)},
    "xlarge": {"z": (-7, 7), "yx": (-150, 150)},
    "tissue": {"z": (-7, 7), "yx": (-300, 300)},
    "thick_medium": {"z": (-25, 25), "yx": (-100, 100)},
    "thick_large": {"z": (-50, 50), "yx": (-270, 270)},
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
