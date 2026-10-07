"""Label-export fixtures of docs/stitching-assembly-validation.md ("Fixtures"), rebuilt in the test.

``label_quad`` (and its volume variant), ``label_pair`` (W-359's two-FOV fixture, rebuilt
from the constants of W-359 ``scripts/labels_retained.py``: ``TWO_FOV_CELLS``,
``TWO_FOV_SHAPE``, ``TWO_FOV_ORIGINS``; read, not copied) with its variants
``label_pair_conflict``, ``label_pair_empty`` and ``label_pair_merged`` (W-362's case "A
cell merged by its owner", its worker notes section 7), ``label_scene`` (W-354's label
fixture rebuilt from its recipe, W-354 ``scripts/fixtures.py``, ``label_scene`` and
``labels``, seeds 354 and 355) and a one-FOV sample whose territory is all one cell.

Each fixture is written as a dataset that ``Dataset.export`` reads: per FOV its territory
label file, imported with ``import_labels(…, relabel=False)``, one molecule per cell
assigned with ``assign_molecules``, the §2.9 assignment folder saved, the reference grid
``images/ref_merged/<fov>.tif``, and the placement (``placement_from_table`` of the
origins, ``nominal_z`` 0, approved) saved under ``stitching/<sample>/``; ``assemble`` is
then given the assignment folders by path, so the ``assembly`` section records each
territory file's path. Each FOV labels the cells it sees with the fixture's cell number
``g`` as ``cell_id``. Nothing is read from a run directory.
"""
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from starfinder.assembly import AssemblyConfig, AssemblyInput, assemble
from starfinder.assignment import assign_molecules
from starfinder.assignment._persist import assignment_directory, write_assignment
from starfinder.dataset import Dataset
from starfinder.io import save_volume
from starfinder.segmentation import import_labels
from starfinder.stitching import approve_placement, placement_from_table, positions_from_table, save_placement

from .assembly_fixtures import cell_namespace, molecules, unit_grid, with_record
from .fusion_fixtures import golden_codebook

LN_XY = 0.194398
GENES = ("A",)


@dataclass
class LabelCase:
    """A label fixture written as a dataset: the placement, the assembly and each FOV's territory."""
    name: str
    dataset: Dataset
    placement: object
    assembly: object
    territories: dict          # fov → uint32 ZYX territory as saved
    origins: dict              # fov → (z, y, x) µm
    grids: dict                # fov → ReferenceGrid
    true_px: dict = field(default_factory=dict)  # (fov, cell_id) → pixels of the uncut cell

    @property
    def root(self):
        return self.dataset.output_root

    @property
    def ids(self):
        return list(self.placement.fovs.fov_id)

    @property
    def placement_dir(self):
        return self.root / "stitching" / self.name

    @property
    def export_dir(self):
        return self.root / "export" / self.name

    def assignment_dir(self, fov, name="default"):
        return assignment_directory(self.root / "checkpoints" / fov, name)

    def territory_path(self, fov, name="default"):
        return self.assignment_dir(fov, name) / "cell_labels.tif"

    def spacing(self, fov):
        return np.asarray(self.grids[fov].metadata.spacing_zyx, float)


def _table(ids, origins):
    return positions_from_table(pd.DataFrame({"fov_id": ids, "nominal_z": [0.0] * len(ids),
                                              "nominal_y": [origins[f][1] for f in ids],
                                              "nominal_x": [origins[f][2] for f in ids]}))


def build(tmp_path, name, territories, origins, spacing=(1.0, 1.0, 1.0), *, grid_shapes=None, statuses=None,
          conflicts=True, true_px=None, assignment="default"):
    """Write a label fixture as a dataset and assemble it from the saved assignment folders.

    ``territories`` maps each FOV (in placement table order) to its uint32 ZYX
    territory (1×Y×X for ``plane``); ``grid_shapes`` the FOV grids (default the
    territories' shapes); ``statuses`` FOVs given another placement status
    (``placeholder``: a constructed placement with the same origins, approved).
    """
    ids = list(territories)
    root = tmp_path / name / "output"
    dataset = Dataset(tmp_path / name / "input", root, "data", "sample", "output", codebook=golden_codebook())
    grids = {f: unit_grid((grid_shapes or {}).get(f, territories[f].shape), spacing) for f in ids}
    result = placement_from_table(_table(ids, origins), grids, sample=name, origins=origins)
    if statuses:
        fovs = result.fovs.copy()
        for fov, status in statuses.items():
            fovs.loc[fovs.fov_id == fov, "status"] = status
        result = replace(result, fovs=fovs)
    placement = approve_placement(result, reviewer="test", note=name)
    save_placement(placement, root / "stitching" / name)
    inputs = []
    for fov in ids:
        grid = grids[fov]
        labels = np.ascontiguousarray(territories[fov], np.uint32)
        source = tmp_path / name / "labels" / f"{fov}.tif"
        plane = labels.shape[0] == 1
        label_grid = grid.projected(method="max") if plane and grid.shape_zyx[0] > 1 else grid
        save_volume(labels, source, metadata=label_grid.metadata)
        seg = import_labels(source, grid=label_grid, target="cell", geometry="plane" if plane else "volume",
                            relabel=False, label_namespace=cell_namespace(fov))
        rows = []
        for value in np.unique(labels[labels > 0]):
            z, y, x = np.argwhere(labels == value)[0]
            rows.append(((0.0 if plane else float(z), float(y), float(x)), "A"))
        assigned = assign_molecules(molecules(fov, rows, GENES), seg, grid=grid)
        folder = assignment_directory(root / "checkpoints" / fov, assignment)
        write_assignment(folder, with_record(assigned, name=assignment), metadata=grid.metadata)
        save_volume(np.zeros(grid.shape_zyx, np.uint8), root / "images" / "ref_merged" / f"{fov}.tif",
                    metadata=grid.metadata)
        inputs.append(AssemblyInput(fov, grid, assignment=folder))
    result = assemble(placement, inputs, config=AssemblyConfig(conflicts=conflicts))
    saved = {f: np.ascontiguousarray(territories[f], np.uint32) for f in ids}
    return LabelCase(name, dataset, placement, result, saved, dict(origins), grids, dict(true_px or {}))


# --- label_quad ---------------------------------------------------------------------------------------------

QUAD_IDS = ["L1", "L2", "L3", "L4"]
QUAD_ORIGINS = {"L1": (0.0, 0.0, 0.0), "L2": (0.0, 30.25, 0.5), "L3": (0.0, 0.5, 30.25), "L4": (0.0, 30.25, 30.25)}
QUAD_STARTS = (4, 16, 28)
QUAD_DOT = (34, 34)
QUAD_EXTRA = (slice(35, 39), slice(4, 8))      # FOV 1's 4 × 4 px cell, in the overlap with FOV 2, which owns it
QUAD_DOT_ID, QUAD_EXTRA_ID = 10, 11


def quad_territory(fov, n_z=1):
    labels = np.zeros((40, 40), np.uint32)
    g = 0
    for r in QUAD_STARTS:
        for c in QUAD_STARTS:
            g += 1
            labels[r:r + 6, c:c + 6] = g
    labels[QUAD_DOT] = QUAD_DOT_ID
    if fov == "L1":
        labels[QUAD_EXTRA] = QUAD_EXTRA_ID
    return np.broadcast_to(labels, (n_z, 40, 40)).copy()


def label_quad(tmp_path, *, volume=False, placeholder=False, conflicts=True, name=None):
    """Four FOVs of 1×40×40 (``volume``: 4×40×40, each box through all planes), unit spacing; grid 1 × 70 × 70."""
    n_z = 4 if volume else 1
    territories = {f: quad_territory(f, n_z) for f in QUAD_IDS}
    name = name or ("label_quad_volume" if volume else "label_quad") + ("_placeholder" if placeholder else "") \
        + ("" if conflicts else "_noconflicts")
    statuses = {"L4": "placeholder"} if placeholder else None
    return build(tmp_path, name, territories, QUAD_ORIGINS, statuses=statuses, conflicts=conflicts)


# --- label_pair (W-359's two-FOV fixture) -------------------------------------------------------------------

#: (g, A's disc or None, B's disc or None), each (cy, cx, r) in px of the sample frame (W-359 TWO_FOV_CELLS).
PAIR_CELLS = [
    (1, (16, 20, 5), None),
    (2, None, (16, 140, 5)),
    (3, (48, 70, 4), (48, 70, 4)),
    (4, (48, 90, 4), (48, 90, 4)),
    (5, (36, 79.5, 4.6), (36, 79.5, 4.6)),
    (6, (8, 77, 4), (8, 82, 4)),
    (7, (58, 83, 4), (58, 76, 4)),
    (8, (24, 74, 5), (24, 74, 3)),
    (9, (24, 85, 3), (24, 84, 5)),
    (10, None, (-1, 124, 5)),
    (11, (52, 62, 4), (52, 62, 4)),
    (12, (40, 30, 6), None),
    (13, None, (44, 130, 6)),
]
PAIR_SHAPE = (64, 96)
PAIR_ORIGINS_PX = [(0.0, 0.0), (0.0, 64.0)]
PAIR_IDS = ["A", "B"]
#: W-362's case "A cell merged by its owner": A segments 31 and 32 apart, B segments both as one cell, 32.
MERGED_CELLS = [(31, (32, 86, 3), (32, 86, 3), 32), (32, (32, 92, 3), (32, 92, 3), None)]


def disc(shape, cy, cx, r, oy=0.0, ox=0.0):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    return (yy + oy - cy) ** 2 + (xx + ox - cx) ** 2 <= r * r


def pair_territories(cells, *, merged=False):
    territories, true_px = {}, {}
    for t, (fov, (oy, ox)) in enumerate(zip(PAIR_IDS, PAIR_ORIGINS_PX)):
        labels = np.zeros(PAIR_SHAPE, np.uint32)
        for cell in cells:
            g, view = cell[0], cell[1 + t]
            if view is None:
                continue
            m = disc(PAIR_SHAPE, view[0], view[1], view[2], oy, ox)
            if not m.any():
                continue
            value = g if not merged or t == 0 or cell[3] is None else cell[3]
            if not merged:
                assert not (labels[m] > 0).any(), (fov, g)       # one segmentation never overlaps itself
            labels[m] = value
            big = disc((PAIR_SHAPE[0] + 40, PAIR_SHAPE[1] + 40), view[0], view[1], view[2], oy - 20, ox - 20)
            true_px[(fov, int(value))] = int(big.sum())
        territories[fov] = labels[None]
    return territories, true_px


def pair_origins():
    return {f: (0.0, oy * LN_XY, ox * LN_XY) for f, (oy, ox) in zip(PAIR_IDS, PAIR_ORIGINS_PX)}


def label_pair(tmp_path, variant="label_pair", *, conflicts=True):
    """W-359's two FOVs of 1×64×96 at 0.194398 µm, A at (0, 0) px and B at (0, 64) px; 13 cells, 21 views.

    Variants: ``label_pair_conflict`` (B's view of cell 6 at x 80), ``label_pair_empty``
    (every label 0) and ``label_pair_merged`` (W-362's cells 31 and 32 alone).
    """
    cells = PAIR_CELLS
    if variant == "label_pair_conflict":
        cells = [(g, a, (8, 80, 4) if g == 6 else b) for g, a, b in PAIR_CELLS]
    if variant == "label_pair_merged":
        territories, true_px = pair_territories(MERGED_CELLS, merged=True)
    else:
        territories, true_px = pair_territories(cells)
    if variant == "label_pair_empty":
        territories, true_px = {f: np.zeros_like(v) for f, v in territories.items()}, {}
    name = variant + ("" if conflicts else "_noconflicts")
    return build(tmp_path, name, territories, pair_origins(), (LN_XY, LN_XY, LN_XY), conflicts=conflicts,
                 true_px=true_px)


# --- label_scene (W-354's label fixture) --------------------------------------------------------------------

SCENE_N, SCENE_STEP = 112, 96


def scene_labels(rng, size):
    """W-354's 2D scene of discs: 399 draws of centre and radius 3.5 to 9 px, each painting only free pixels."""
    labels = np.zeros((size, size), np.uint32)
    yy, xx = np.mgrid[0:size, 0:size]
    for g in range(1, 400):
        cy, cx, r = rng.uniform(0, size), rng.uniform(0, size), rng.uniform(3.5, 9)
        m = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
        labels[m & (labels == 0)] = g
    return labels


def label_scene(tmp_path, *, placeholder=False):
    """3 × 3 FOVs of 112 × 112 cut at steps of 96 px from one 304 × 304 scene (seed 354), jittered (seed 355)."""
    rng, jitter = np.random.default_rng(354), np.random.default_rng(355)
    scene = scene_labels(rng, 2 * SCENE_STEP + SCENE_N)
    scene_px = {int(g): int((scene == g).sum()) for g in np.unique(scene[scene > 0])}
    territories, origins, true_px = {}, {}, {}
    for r in range(3):
        for c in range(3):
            fov = f"S{len(territories)}"
            cut = scene[r * SCENE_STEP:r * SCENE_STEP + SCENE_N, c * SCENE_STEP:c * SCENE_STEP + SCENE_N]
            oy, ox = r * SCENE_STEP + jitter.uniform(-0.4, 0.4), c * SCENE_STEP + jitter.uniform(-0.4, 0.4)
            territories[fov] = np.ascontiguousarray(cut, np.uint32)[None]
            origins[fov] = (0.0, oy * LN_XY, ox * LN_XY)
            for g in np.unique(cut[cut > 0]):
                true_px[(fov, int(g))] = scene_px[int(g)]
    statuses = {"S4": "placeholder"} if placeholder else None
    name = "label_scene" + ("_placeholder" if placeholder else "")
    return build(tmp_path, name, territories, origins, (LN_XY, LN_XY, LN_XY), statuses=statuses, true_px=true_px)


# --- one FOV, all one cell ----------------------------------------------------------------------------------

def one_cell(tmp_path):
    """One owning FOV of 1×16×16 whose territory is all cell 1."""
    return build(tmp_path, "one_cell", {"O1": np.ones((1, 16, 16), np.uint32)}, {"O1": (0.0, 0.0, 0.0)})
