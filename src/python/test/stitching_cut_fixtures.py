"""The ``cut_synthetic`` fixture of docs/stitching-assembly-validation.md ("Fixtures"), rebuilt in the test.

The recipe is W-324's (``scripts/fixture_data.py`` and ``scripts/fixtures.py``), restated
here from its constants: nothing is read from a run directory. One scene of 16×256×256
(seed 20261006: a smoothed-noise texture and 70 Gaussian nuclei) is cut into a 3 × 3 grid
of tiles of 14×120×120 at a nominal step of 64 px from a nominal origin of z 1, y and x 4;
each tile is cut at its true origin, the nominal one plus an integer jitter (z in [−1, 1],
y and x in [−4, 4], seed 324), with noise N(0, 3) per tile (seed 1000 + tile), rounded to
uint16. Tile ``k`` sits at column ``k // 3`` and row ``k % 3`` ("Down & Right"). Spacing is
the LN calibration (0.346255, 0.194398, 0.194398) µm unless a case says otherwise.

Positions are integers in voxels; the nominal positions enter ``stitch`` through
``positions_from_table`` as voxels times spacing (µm). The derived cases are those of W-324
``fixtures.py`` (``two_tiles``, ``z1_3d``, ``plane_2d``, ``spacing_1_2_2``,
``no_signal_zero``, ``no_signal_noise`` with centre noise seed 1050, ``disconnected``) and
``non_square`` (the same scene cut into 3 × 3 tiles of 14×104×120 at nominal steps 56 in y
and 64 in x, with the same jitter draws).
"""
from dataclasses import dataclass
from functools import cache

import numpy as np
import pandas as pd

from starfinder.image import ImageMetadata
from starfinder.segmentation import ReferenceGrid
from starfinder.stitching import StitchingInput, positions_from_table

SEED_SCENE, SEED_JITTER, SEED_NOISE = 20261006, 324, 1000
SEED_CENTRE_NOISE = SEED_NOISE + 50
LN_SPACING = (0.346255, 0.194398, 0.194398)
IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
ROWS = COLUMNS = 3
TILE = (14, 120, 120)


@dataclass(frozen=True)
class Tile:
    """One tile: grid cell, nominal and true origin (z, y, x) in voxels, and its image."""

    tile: int
    column: int
    row: int
    nominal: tuple[int, int, int]
    true: tuple[int, int, int]
    image: np.ndarray

    @property
    def fov_id(self):
        return f"tile{self.tile}"


@dataclass(frozen=True)
class Case:
    """A fixture case: its tiles in table order and the spacing (µm) of every tile."""

    name: str
    tiles: tuple[Tile, ...]
    spacing: tuple[float, float, float] = LN_SPACING

    @property
    def fov_ids(self):
        return [t.fov_id for t in self.tiles]

    def table(self):
        s = np.asarray(self.spacing)
        frame = pd.DataFrame({"fov_id": self.fov_ids,
                              **{f"nominal_{a}": [t.nominal[i] * s[i] for t in self.tiles]
                                 for i, a in enumerate("zyx")}})
        return positions_from_table(frame)

    def inputs(self):
        items = []
        for t in self.tiles:
            shape = t.image.shape if t.image.ndim == 3 else (1, *t.image.shape)
            grid = ReferenceGrid(shape, ImageMetadata("fov", self.spacing, (0.0, 0.0, 0.0), IDENTITY, "µm"),
                                 "declared")
            items.append(StitchingInput(t.fov_id, t.image, grid, "reference_merged"))
        return items

    def true_zyx(self):
        return np.array([t.true for t in self.tiles], dtype=np.float64)


@cache
def scene():
    """The 16×256×256 float32 scene of W-324 ``synthetic_scene`` (seed 20261006)."""
    from scipy import ndimage
    nz, ny, nx = 16, 256, 256
    rng = np.random.default_rng(SEED_SCENE)
    texture = ndimage.gaussian_filter(rng.normal(0, 1, (nz, ny, nx)), (1, 2, 2))
    out = 20 + 60 * texture / texture.std()
    zz, yy, xx = np.mgrid[0:nz, 0:ny, 0:nx]
    for _ in range(70):
        cz, cy, cx = rng.uniform(0, nz), rng.uniform(0, ny), rng.uniform(0, nx)
        sz, sy, sx = rng.uniform(1.5, 3), rng.uniform(4, 9), rng.uniform(4, 9)
        out += rng.uniform(80, 200) * np.exp(-((zz - cz) ** 2 / (2 * sz ** 2) + (yy - cy) ** 2 / (2 * sy ** 2)
                                               + (xx - cx) ** 2 / (2 * sx ** 2)))
    return np.clip(out, 0, None).astype(np.float32)


def layout(step_yx=(64, 64)):
    """Grid cells, nominal and true origins in voxels (W-324 ``grid_layout``, seed 324)."""
    rng = np.random.default_rng(SEED_JITTER)
    cells = []
    for k in range(ROWS * COLUMNS):
        column, row = k // ROWS, k % ROWS
        nominal = (1, 4 + step_yx[0] * row, 4 + step_yx[1] * column)
        jitter = (int(rng.integers(-1, 2)), int(rng.integers(-4, 5)), int(rng.integers(-4, 5)))
        cells.append((k, column, row, nominal, tuple(n + j for n, j in zip(nominal, jitter))))
    return cells


def cut(true_zyx, tile, size):
    """The block at ``true_zyx`` plus N(0, 3) noise of seed 1000 + tile, rounded and clipped to uint16."""
    z, y, x = true_zyx
    block = scene()[z:z + size[0], y:y + size[1], x:x + size[2]].astype(np.float32)
    assert block.shape == tuple(size), (block.shape, true_zyx)
    block = block + np.random.default_rng(SEED_NOISE + tile).normal(0, 3.0, block.shape)
    return np.clip(np.rint(block), 0, np.iinfo(np.uint16).max).astype(np.uint16)


def _grid_tiles(step_yx=(64, 64), size=TILE):
    return tuple(Tile(k, c, r, nominal, true, cut(true, k, size)) for k, c, r, nominal, true in layout(step_yx))


@cache
def case(name):
    """The named case of ``cut_synthetic``."""
    tiles = _grid_tiles()
    if name == "cut_synthetic":
        return Case(name, tiles)
    if name == "two_tiles":
        return Case(name, tuple(t for t in tiles if t.row == 0 and t.column < 2))
    four = tuple(t for t in tiles if t.row < 2 and t.column < 2)
    if name == "spacing_1_2_2":
        return Case(name, four, (1.0, 2.0, 2.0))
    if name in ("z1_3d", "plane_2d"):
        # Plane z 8 of the scene at the y, x true origins of tiles 0, 1, 3, 4; nominal and true z 0.
        planes = []
        for t in four:
            image = cut((8, t.true[1], t.true[2]), t.tile, (1, *TILE[1:]))
            planes.append(Tile(t.tile, t.column, t.row, (0, *t.nominal[1:]), (0, *t.true[1:]),
                               image if name == "z1_3d" else image[0]))
        return Case(name, tuple(planes))
    if name == "no_signal_zero":
        return Case(name, tuple(_replace(t, np.zeros_like(t.image)) if t.tile == 4 else t for t in tiles))
    if name == "no_signal_noise":
        rng = np.random.default_rng(SEED_CENTRE_NOISE)
        centre = tiles[4].image
        noise = np.clip(np.rint(20 + rng.normal(0, 3, centre.shape)), 0, None).astype(centre.dtype)
        return Case(name, tuple(_replace(t, noise) if t.tile == 4 else t for t in tiles))
    if name == "disconnected":
        moved = tiles[8]
        moved = Tile(8, moved.column, moved.row, (moved.nominal[0], moved.nominal[1], moved.nominal[2] + 200),
                     moved.true, moved.image)
        return Case(name, (*tiles[:8], moved))
    if name == "non_square":
        return Case(name, _grid_tiles((56, 64), (14, 104, 120)))
    if name == "no_overlap":
        # MV5: two tiles whose nominal boxes do not overlap (tile 3 moved by +200 px in x).
        a, b = tiles[0], tiles[3]
        return Case(name, (a, Tile(3, b.column, b.row, (b.nominal[0], b.nominal[1], b.nominal[2] + 200), b.true,
                                   b.image)))
    raise KeyError(name)


def _replace(tile, image):
    return Tile(tile.tile, tile.column, tile.row, tile.nominal, tile.true, image)


def errors_voxels(result, case_, fovs=None):
    """Per FOV and axis: (origin − mean) − (true − mean) in voxels over ``fovs`` (default: all FOVs)."""
    ids = case_.fov_ids
    keep = [ids.index(f) for f in (ids if fovs is None else fovs)]
    rows = result.fovs.set_index("fov_id").loc[[ids[i] for i in keep]]
    origin = rows[["origin_z", "origin_y", "origin_x"]].to_numpy() / np.asarray(case_.spacing)
    true = case_.true_zyx()[keep]
    return (origin - origin.mean(axis=0)) - (true - true.mean(axis=0))
