"""Benchmark presets for standard test configurations."""

from __future__ import annotations

# Standard volume size presets (Z, Y, X)
SIZE_PRESETS: dict[str, tuple[int, int, int]] = {
    "tiny": (5, 128, 128),
    "small": (10, 256, 256),
    "medium": (30, 512, 512),
    "large": (30, 1024, 1024),
    "xlarge": (30, 1496, 1496),  # cell-culture-3D size
    "tissue": (30, 3072, 3072),  # tissue-2D size
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
