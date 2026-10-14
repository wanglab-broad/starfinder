"""The figure sizes of the report page, written from docs/report-algorithms.md ("Figure sizes", W-475).

The slots are the widths in px at which the page shows a figure in a window 1,200 px wide: the
full column, a half column, a third, and the column of the comparison page. A plot (PNG) is
drawn at twice the width of its slot; tiles side by side are separated by the smallest whole
gap that is at least 8 px as shown. The numbers are written here from the page, never read
from the package.
"""
from __future__ import annotations

import math

FULL, HALF, THIRD, COMPARISON = 932, 442, 300, 1160
MIN_TEXT_PX, TILE_GAP_PX, MARK_PX, ROW_PX, MAX_ROWS = 11, 8, 2, 14, 64
PREPROCESSING_TILE, EXAMPLE_TILE = 256, 512

# The PNG plots: (width, height) or the width alone where the height follows the content.
TIME_WIDTH = 2 * HALF                                   # "Time per step", a half column
INTENSITY_SIZE = (2 * FULL, 2 * 360)                    # the intensity panels, 360 px high as shown
HEATMAP_WIDTH, READS_Z_WIDTH = 2 * FULL, 2 * HALF       # "All FOVs and rounds"; "Reads along Z"
DISTRIBUTION_THIRD = (2 * THIRD, round(2 * THIRD * 340 / 600))   # 600 × 340
DISTRIBUTION_HALF = (2 * HALF, round(2 * HALF * 340 / 600))      # 884 × 501
BOX_SIZE = (2 * COMPARISON, round(2 * COMPARISON / 3))           # 2,320 × 773


def tile_gap(n, tile, slot=FULL):
    """The smallest whole gap ``g`` with ``g · slot / (n · tile + (n − 1) · g) ≥ 8``."""
    return 0 if n < 2 else math.ceil(TILE_GAP_PX * n * tile / (slot - TILE_GAP_PX * (n - 1)))


def tiles_width(n, tile, slot=FULL):
    """The width of ``n`` tiles side by side with their gaps."""
    return n * tile + (n - 1) * tile_gap(n, tile, slot)


def time_height_at_least(names):
    """"Time per step": at least 280 px as shown and 26 px as shown per step name."""
    return max(2 * 280, 2 * 26 * names)


def grid_slot(n):
    """The slot of a figure in a row of ``n`` figures of a three-figure grid."""
    return THIRD if n >= 3 else HALF
