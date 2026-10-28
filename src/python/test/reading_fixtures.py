"""A 3 × 3 grid of 3D tiles saved as compressed pages, and a counter of page decodes (W-397).

The fixture of design/validation/stitching-assembly.md for how the fusion reads its sources:
nine tiles of ``SHAPE`` uint16 (smooth random fields, seeded), on a 3 × 3 grid with 10 %
overlap and sub-voxel jitter in Z, Y and X, so that each output plane interpolates two pages
of a tile. Each tile is written as zlib-compressed pages without axis metadata (``IYX``, the
form of the share's tiles) or with ZYX axes. ``DecodeCounter`` wraps tifffile's per-page
decoding function and attributes each call to (FOV, page, segment) by the compressed bytes it
receives, so it counts the decodes of any reader, the base revision's included. Nothing here
uses a name that W-397 added.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from starfinder.image import ImageMetadata
from starfinder.segmentation import ReferenceGrid
from starfinder.stitching import approve_placement, placement_from_table, positions_from_table

IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
GRID_IDS = tuple(f"T{row}{column}" for row in range(3) for column in range(3))
SHAPE = (6, 256, 256)
SPACING = (1.0, 0.5, 0.5)
OVERLAP = 0.1
FORMS = {"pages": "IYX", "ZYX": "ZYX"}
#: Sub-voxel offsets (z, y, x) in µm of each tile from its grid position.
JITTER = {fov: (0.25 * (k % 3), 0.15 * ((k * 5) % 4) - 0.2, 0.1 * ((k * 7) % 5) - 0.2)
          for k, fov in enumerate(GRID_IDS)}


def tile(fov, shape=SHAPE):
    """A smooth random field of uint16 (seeded by the FOV), with values 100 to about 4,000."""
    k = GRID_IDS.index(fov)
    rng = np.random.default_rng(397 + k)
    z, y, x = np.meshgrid(*[np.arange(n, dtype=np.float64) for n in shape], indexing="ij")
    field = np.zeros(shape)
    for _ in range(4):
        fz, fy, fx = rng.uniform(0.05, 0.3), rng.uniform(0.01, 0.06), rng.uniform(0.01, 0.06)
        field += rng.uniform(300, 900) * (1 + np.sin(fz * z + fy * y + fx * x + rng.uniform(0, 6.28)))
    field += rng.normal(0, 20, shape)
    return np.clip(field + 100, 0, 65535).astype(np.uint16)


def write_tiles(root, form, shape=SHAPE) -> dict[str, Path]:
    """FOV → its tile written as zlib-compressed pages without metadata (``pages``) or with ZYX axes (``ZYX``)."""
    paths = {}
    for fov in GRID_IDS:
        path = Path(root) / form / f"{fov}.tif"
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = None if form == "pages" else {"axes": "ZYX"}
        tifffile.imwrite(path, tile(fov, shape), metadata=metadata, photometric="minisblack", compression="zlib")
        with tifffile.TiffFile(path) as tif:
            assert tif.series[0].axes == FORMS[form] and len(tif.pages) == shape[0]
        paths[fov] = path
    return paths


def grid_placement(shape=SHAPE):
    """The approved placement of the grid (origins = nominal positions + JITTER) and the one reference grid."""
    grid = ReferenceGrid(tuple(shape), ImageMetadata("tile", SPACING, (0.0, 0.0, 0.0), IDENTITY, "µm"), "declared")
    step_y, step_x = [(1 - OVERLAP) * n * s for n, s in zip(shape[1:], SPACING[1:])]
    rows = [(fov, 0.0, (k // 3) * step_y, (k % 3) * step_x) for k, fov in enumerate(GRID_IDS)]
    table = positions_from_table(pd.DataFrame(rows, columns=["fov_id", "nominal_z", "nominal_y", "nominal_x"]))
    origins = {fov: tuple(n + j for n, j in zip(nominal, JITTER[fov])) for fov, *nominal in rows}
    placed = placement_from_table(table, {fov: grid for fov in GRID_IDS}, sample="grid", origins=origins)
    return approve_placement(placed, reviewer="test", note="reading"), grid


def segments(paths) -> dict[bytes, tuple[str, int, int]]:
    """The compressed bytes of every segment of every page → (FOV, page, segment); each must be unique."""
    found = {}
    for fov, path in paths.items():
        with tifffile.TiffFile(path) as tif:
            fh = tif.filehandle
            for p, page in enumerate(tif.pages):
                for s, (offset, count) in enumerate(zip(page.dataoffsets, page.databytecounts)):
                    fh.seek(offset)
                    data = fh.read(count)
                    assert data not in found, "two segments with equal bytes cannot be told apart"
                    found[data] = (fov, p, s)
    return found


class DecodeCounter:
    """Counts the calls of tifffile's page decoding function per (FOV, page, segment) while active.

    ``TiffPage.decode`` is the function every tifffile reader calls on the
    compressed bytes of one segment (``TiffPage.asarray`` and the Zarr store
    alike; a ``TiffFrame`` uses its key frame's). The wrapper looks the bytes
    up in :func:`segments`; ``log`` keeps the order of the calls.
    """

    def __init__(self, paths):
        self.segments = segments(paths)
        self.counts = Counter()
        self.log = []
        self._original = None

    def __enter__(self):
        original = tifffile.TiffPage.__dict__["decode"]
        counter = self

        def wrapped(page):
            decode = original.__get__(page, type(page))

            def counted(data, *args, **kwargs):
                key = counter.segments.get(bytes(data) if data is not None else None, ("unknown", -1, -1))
                counter.counts[key] += 1
                counter.log.append(key)
                return decode(data, *args, **kwargs)
            return counted

        self._original = original
        tifffile.TiffPage.decode = property(wrapped)
        return self

    def __exit__(self, *exc):
        tifffile.TiffPage.decode = self._original
        return False

    def reset(self):
        self.counts.clear()
        self.log.clear()

    def per_page(self) -> Counter:
        """Decodes per (FOV, page): the largest count over the page's segments."""
        pages = Counter()
        for (fov, page, _), n in self.counts.items():
            pages[(fov, page)] = max(pages[(fov, page)], n)
        return pages
