"""The figure sizes of the report page, written from design/specs/report-algorithms.md ("Figure sizes", W-475, W-478).

The slots are the widths in px at which the page shows a figure in a window 1,200 px wide: the
full column, a half column, a third, and the column of the comparison page. A plot (PNG) is
drawn at twice the width of its slot; tiles side by side are separated by the smallest whole
gap that is at least 8 px as shown; no figure is shown wider than two thirds of its drawn
width. The numbers are written here from the page, never read from the package.
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


# W-478: no figure is shown wider than two thirds of its drawn width.
SHOWN_AT_MOST = 2 / 3

# The table of figures of "Figure sizes" (W-478): per figure, its slot, its shown widths and the leading numbers of its
# drawn width, as many as the checks read ("Distributions" in a third, then in a half; the Preprocessing grid's width,
# then its tile). test_report_docs_contract.py checks that the page's table says this.
FIGURES = {
    "Time per step": ("half", [442], [884]),
    "Intensity after each step": ("full", [932], [1864]),
    "All FOVs and rounds": ("full", [932], [1864]),
    "Reads along Z": ("half", [442], [884]),
    "Distributions across FOVs and per cell": ("third or half", [300, 442], [600, 884]),
    "Box plots by sample (`index.html`)": ("comparison", [1160], [2320]),
    "Preprocessing example grid": ("full", [932], [1398, 256]),
    "Registration and segmentation examples": ("full", [932], [1564]),
    "Whole-sample marker view": ("full", [932], [1600]),
    "Marker zoom-in tile, XY and XZ": ("third", [300], [512]),
}


def preprocessing_tile(columns, slot=FULL):
    """The tile of a Preprocessing grid of ``columns`` states: 256 px, or the smallest whole size above it whose grid
    with its gaps is at least 3/2 of the slot wide (at least 1,398 px for the full column)."""
    tile = PREPROCESSING_TILE
    while tiles_width(columns, tile, slot) < 3 * slot / 2:
        tile += 1
    return tile


def time_height_at_least(names):
    """"Time per step": at least 280 px as shown and 26 px as shown per step name."""
    return max(2 * 280, 2 * 26 * names)


def grid_slot(n):
    """The slot of a figure in a row of ``n`` figures of a three-figure grid."""
    return THIRD if n >= 3 else HALF
