"""Hand-built segmentation fixtures of the §2.9 validation design (docs/assignment-algorithms.md, "Fixtures").

``boxes`` holds the cell and nucleus labels of the fixture of that name (its molecules belong
to the assignment tests); ``seeded`` adds the watershed stain of rows L9 and L13. Every
position is stated here, so expected values follow from the geometry.
"""
import numpy as np

from starfinder.image import ImageMetadata

BOXES_SHAPE = (8, 32, 32)
BOXES_METADATA = ImageMetadata("boxes", spacing_zyx=(0.35, 0.1, 0.1), spatial_unit="micrometer")

# Half-open (z, y, x) ranges. Every box includes the plane z = 4.
CELL_BOXES = {
    1: ((1, 7), (1, 7), (1, 7)),
    2: ((1, 7), (1, 7), (9, 17)),
    3: ((1, 7), (9, 16), (1, 6)),      # cells 3 and 4 share the border x = 5 | 6
    4: ((1, 7), (9, 16), (6, 11)),
    5: ((1, 7), (9, 16), (14, 20)),
    6: ((1, 7), (19, 25), (1, 10)),    # cells 6 and 7 share the border x = 9 | 10
    7: ((1, 7), (19, 25), (10, 16)),
    8: ((1, 7), (19, 25), (19, 25)),   # no nucleus
}
NUCLEUS_BOXES = {
    11: ((3, 6), (2, 5), (2, 5)),      # inside cell 1
    21: ((3, 6), (2, 5), (10, 12)),    # cell 2 holds two nuclei
    22: ((3, 6), (2, 5), (14, 16)),
    31: ((2, 7), (10, 15), (4, 8)),    # 100 voxels: x 4-5 in cell 3, x 6-7 in cell 4 (50 / 50)
    51: ((4, 5), (12, 13), (14, 24)),  # 10 voxels: x 14-19 in cell 5, x 20-23 in the background (6 / 4)
    61: ((4, 5), (22, 23), (2, 12)),   # 10 voxels: x 2-9 in cell 6, x 10-11 in cell 7 (8 / 2)
    91: ((3, 6), (29, 31), (29, 31)),  # background; 5 voxels in Y and X from cell 8, farther from the rest
}


def paint(boxes, shape=BOXES_SHAPE):
    """A uint32 label image with each box set to its value."""
    labels = np.zeros(shape, np.uint32)
    for value, ranges in boxes.items():
        labels[tuple(slice(*r) for r in ranges)] = value
    return labels


def boxes():
    """(cell labels, nucleus labels), both uint32 8×32×32."""
    return paint(CELL_BOXES), paint(NUCLEUS_BOXES)


def seeded_stain():
    """200 inside the boxes cells and 20 elsewhere, plus Gaussian noise of standard deviation 5 (seed 101), uint8."""
    cells, _ = boxes()
    rng = np.random.default_rng(101)
    values = np.where(cells > 0, 200.0, 20.0) + rng.normal(0, 5, BOXES_SHAPE)
    return np.clip(np.rint(values), 0, 255).astype(np.uint8)
