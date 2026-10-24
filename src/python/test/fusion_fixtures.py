"""Fused-image and export fixtures of docs/stitching-assembly-validation.md ("Fixtures"), rebuilt in the test.

``quad`` (seed 2102), ``zstack``, ``plane_pair``, ``checker_pair``, ``overlap_pair`` and
``position_grid`` (seed 354) follow the rows of that table; ``checker_pair``,
``overlap_pair`` and ``position_grid`` are W-354's fixtures ``reduced``, ``overlap`` and
``position`` (W-354 ``scripts/fixtures.py``, read, not copied), rebuilt from their
recipes and constants. ``fuse_band`` is six FOVs in one column. ``export_folder`` writes
``assembly_golden`` as a dataset that ``Dataset.export`` reads: the saved placement, the
saved §2.9 assignment folders, the reference images and a ``reference_stain`` per FOV.
Each fixture's placement is a ``PlacementResult`` built here and approved. Nothing is
read from a run directory.
"""
import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from starfinder.assignment._persist import assignment_directory, write_assignment
from starfinder.readout import Codebook
from starfinder.dataset import Dataset
from starfinder.image import ImageMetadata
from starfinder.io import save_volume
from starfinder.export import FusionInput
from starfinder.segmentation import ReferenceGrid
from starfinder.stitching import approve_placement, save_placement

from .assembly_fixtures import (
    GOLDEN_GENES,
    GOLDEN_IDS,
    SAMPLE,
    TILE_ZYX,
    golden_placement,
    golden_results,
    placement,
    spot_namespace,
    unit_grid,
    with_record,
)
from .coordination_helpers import write_molecules_csv

IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
LN_Z, LN_XY = 0.346255, 0.194398
UNIT = (1.0, 1.0, 1.0)


def grid_of(shape, spacing=UNIT):
    return ReferenceGrid(tuple(shape), ImageMetadata("fov", tuple(spacing), (0.0, 0.0, 0.0), IDENTITY, "µm"),
                         "declared")


@dataclass
class Case:
    """A fusion fixture: an approved placement, each FOV's grid and source image, in placement order."""
    placement: object
    grids: dict
    arrays: dict
    extra: dict = field(default_factory=dict)

    @property
    def ids(self):
        return list(self.placement.fovs.fov_id)

    def inputs(self, name="reference_stain", fovs=None):
        return [FusionInput(f, self.grids[f], {name: self.arrays[f]}) for f in (fovs or self.included)]

    @property
    def included(self):
        fovs = self.placement.fovs
        return list(fovs.fov_id[fovs.status.isin(["image_supported", "metadata_only"])])

    def origin(self, fov):
        row = self.placement.fovs.set_index("fov_id").loc[fov]
        return np.array([row.origin_z, row.origin_y, row.origin_x])


def make_case(sample, origins, arrays, spacing=UNIT, *, statuses=None, grids=None, extra=None):
    ids = list(origins)
    grids = grids or {f: grid_of(arrays[f].shape if arrays[f].ndim == 3 else (1,) + arrays[f].shape, spacing)
                      for f in ids}
    result = placement(sample, ids, grids, origins, origins, statuses=statuses)
    return Case(approve_placement(result, reviewer="test", note=sample), grids, arrays, extra or {})


# --- quad -----------------------------------------------------------------------------------------------------

QUAD_ORIGINS = {"Q0": (0.0, 0.0, 0.0), "Q1": (0.0, 26.3, 0.4), "Q2": (0.5, 0.2, 26.7), "Q3": (0.3, 26.1, 26.9)}


def quad_values():
    rng = np.random.default_rng(2102)
    return {f: rng.integers(0, 4096, (4, 32, 32)) for f in QUAD_ORIGINS}


def quad(dtype="uint16", dtypes=None):
    """Four FOVs of 4×32×32 (values of seed 2102 in table order), unit spacing, fractional offsets; grid 4×58×58.

    uint8 takes the values modulo 256; ``dtypes`` maps a FOV to another dtype (mixed dtypes).
    """
    values = quad_values()
    arrays = {}
    for f, v in values.items():
        kind = (dtypes or {}).get(f, dtype)
        arrays[f] = (v % 256 if kind in ("uint8", "int8") else v).astype(kind)
    return make_case("quad", QUAD_ORIGINS, arrays)


# --- zstack and plane_pair ------------------------------------------------------------------------------------

ZSTACK_ORIGINS = {"equal": ((0.0, 0.0, 0.0), (0.0, 0.0, 6.25)), "different": ((0.0, 0.0, 0.0), (1.4, 0.0, 6.25)),
                  "plane": ((0.0, 0.0, 0.0), (0.0, 0.0, 6.25))}


def zstack(case="equal", dtype="uint16"):
    """Two FOVs of 4×8×8 (one-plane 1×8×8 for ``plane``), values ``1 + 64·z + 8·y + x + 100·f``."""
    n_z = 1 if case == "plane" else 4
    z, y, x = np.meshgrid(np.arange(n_z), np.arange(8), np.arange(8), indexing="ij")
    arrays = {f"Z{f}": (1 + 64 * z + 8 * y + x + 100 * f).astype(dtype) for f in range(2)}
    return make_case(f"zstack_{case}", dict(zip(arrays, ZSTACK_ORIGINS[case])), arrays)


def plane_pair():
    """Two one-plane FOVs of 1×8×8 uint8, constant 50 and 120, at (0, 0, 0) and (2, 0, 10): slab [−0.5, 2.5)."""
    arrays = {"P0": np.full((1, 8, 8), 50, np.uint8), "P1": np.full((1, 8, 8), 120, np.uint8)}
    return make_case("plane_pair", {"P0": (0.0, 0.0, 0.0), "P1": (2.0, 0.0, 10.0)}, arrays)


# --- checker_pair (W-354 "reduced") ---------------------------------------------------------------------------

CHECKER_SPACING = (0.5, 0.2, 0.2)
CHECKER_B_PX = (0.0, 10.4, 40.7)


def checker_pair(case="3d"):
    """A one-voxel checkerboard of 0 and 200 (uint8) in Y and X, FOV A at the origin, B at (0, 10.4, 40.7) px."""
    shape = (4, 64, 64) if case == "3d" else (1, 64, 64)
    yy, xx = np.meshgrid(np.arange(64), np.arange(64), indexing="ij")
    board = np.where((yy + xx) % 2 == 0, 200, 0).astype(np.uint8)
    arrays = {"A": np.broadcast_to(board, shape).copy(), "B": np.broadcast_to(board, shape).copy()}
    origins = {"A": (0.0, 0.0, 0.0), "B": tuple(p * s for p, s in zip(CHECKER_B_PX, CHECKER_SPACING))}
    return make_case(f"checker_{case}", origins, arrays, CHECKER_SPACING)


# --- overlap_pair (W-354 "overlap") ---------------------------------------------------------------------------

OVERLAP_VALUES = {"uint8": (100, 203), "uint16": (1000, 3001), "float32": (0.25, 0.8)}


def overlap_pair(dtype):
    """Two constant 2D FOVs of 32×256 px at 0.194398 µm, the second at (8, 200) px; union grid 40 × 456."""
    a, b = OVERLAP_VALUES[dtype]
    arrays = {"A": np.full((1, 32, 256), a, dtype), "B": np.full((1, 32, 256), b, dtype)}
    origins = {"A": (0.0, 0.0, 0.0), "B": (0.0, 8 * LN_XY, 200 * LN_XY)}
    return make_case(f"overlap_{dtype}", origins, arrays, (LN_Z, LN_XY, LN_XY))


# --- position_grid (W-354 "position") -------------------------------------------------------------------------

def position_grid():
    """3 × 3 FOVs of 16×128×128 uint16, step 115.2 px jittered (seed 354), one voxel of 1000 per FOV.

    The draws are W-354's, in its order: per FOV (row-major) the origin in Z, Y and X
    (uniform ±2 slices, ±3 px around the step), then the bright voxel's index. ``extra``
    holds each FOV's bright index and its sample position ``q*`` (µm).
    """
    rng = np.random.default_rng(354)
    spacing = (LN_Z, LN_XY, LN_XY)
    origins, arrays, bright = {}, {}, {}
    for r in range(3):
        for c in range(3):
            o = (rng.uniform(-2, 2), r * 115.2 + rng.uniform(-3, 3), c * 115.2 + rng.uniform(-3, 3))
            p = (int(rng.integers(4, 12)), int(rng.integers(30, 98)), int(rng.integers(30, 98)))
            fov = f"G{len(origins)}"
            array = np.zeros((16, 128, 128), np.uint16)
            array[p] = 1000
            origins[fov] = tuple(v * s for v, s in zip(o, spacing))
            arrays[fov] = array
            bright[fov] = (p, np.array([(o[i] + p[i]) * spacing[i] for i in range(3)]))
    return make_case("position", origins, arrays, spacing, extra={"bright": bright})


# --- fuse_band ------------------------------------------------------------------------------------------------

def fuse_band():
    """Six FOVs of 16×256×256 uint8 in one column at Y origins 230.4·r µm, values ``(z + y + x) mod 251``."""
    z, y, x = np.meshgrid(np.arange(16), np.arange(256), np.arange(256), indexing="ij")
    image = ((z + y + x) % 251).astype(np.uint8)
    origins = {f"B{r}": (0.0, 230.4 * r, 0.0) for r in range(6)}
    return make_case("band", origins, {f: image for f in origins})


# --- export_folder --------------------------------------------------------------------------------------------

STAIN_FILE = ("other_rounds", "reference_stain", "image.ome.tif")


def stain_image(shape):
    z, y, x = np.meshgrid(*[np.arange(n) for n in shape], indexing="ij")
    return ((z + y + x) % 251).astype(np.uint8)


def merged_image(shape):
    z, y, x = np.meshgrid(*[np.arange(n) for n in shape], indexing="ij")
    return ((3 * z + y + 2 * x) % 200 + 1).astype(np.uint8)


def golden_codebook():
    """A codebook whose genes are the golden genes in order (for the molecule-table path)."""
    sequences = [f"{a}{b}{c}" for a in "1234" for b in "1234" for c in "1234"][:len(GOLDEN_GENES)]
    return Codebook(pd.DataFrame({"entry_id": sequences, "gene_id": list(GOLDEN_GENES), "color_sequence": sequences}),
                    round_labels=("round1", "round2", "round3"), channel_labels=("ch00", "ch01", "ch02", "ch03"))


def plane_results():
    """The golden FOVs as one-plane FOVs: labels max-projected, every molecule at z 0, grids of 1×48×64."""
    from starfinder.assignment import assign_molecules

    from .assembly_fixtures import golden_pieces, label_run, molecules
    grid = unit_grid((1,) + TILE_ZYX[1:])
    results = {}
    for fov, (labels, _, local) in golden_pieces("3d").items():
        rows = [((0.0, p[1], p[2]), gene) for p, gene in local]
        results[fov] = assign_molecules(molecules(fov, rows, GOLDEN_GENES), label_run(labels.max(axis=0)[None], fov,
                                                                                      grid), grid=grid)
    return results


@dataclass
class ExportFolder:
    """``assembly_golden`` written as a dataset for ``Dataset.export``."""
    dataset: Dataset
    placement: object
    results: dict
    grids: dict

    @property
    def root(self):
        return self.dataset.output_root

    @property
    def placement_dir(self):
        return self.root / "stitching" / SAMPLE

    @property
    def export_dir(self):
        return self.root / "export" / SAMPLE

    def stain_path(self, fov):
        return self.root / "checkpoints" / fov / "/".join(STAIN_FILE)

    def assignment_dir(self, fov, name="default"):
        return assignment_directory(self.root / "checkpoints" / fov, name)


def export_folder(tmp_path, *, dimensions="3d", placement_result=None, conflicts_name="default"):
    """Write ``assembly_golden`` (3D, or one-plane FOVs with ``dimensions="plane"``) as a dataset.

    The saved placement under ``stitching/sample1/``, each FOV's §2.9 assignment folder
    ``checkpoints/<fov>/assignment/default/``, ``images/ref_merged/<fov>.tif`` (the grid, uint8
    values ``(3z + y + 2x) mod 200 + 1``), the ``reference_stain`` (uint8 ``(z + y + x) mod
    251``) and each FOV's ``fovs/<fov>/molecules.csv`` (the assignment's molecules, their identities and zero-based
    z, y, x, as ``FOV.export_spots`` writes them).
    """
    shape = TILE_ZYX if dimensions == "3d" else (1,) + TILE_ZYX[1:]
    grids = {fov: unit_grid(shape) for fov in GOLDEN_IDS}
    origins = None
    if dimensions != "3d":
        from .assembly_fixtures import golden_origins
        origins = golden_origins()
    result = placement_result or golden_placement(grids=grids, origins=origins)
    results = golden_results("3d") if dimensions == "3d" else plane_results()
    dataset = Dataset(tmp_path / "input", tmp_path / "output", "data", "sample", "output", codebook=golden_codebook())
    root = dataset.output_root
    save_placement(result, root / "stitching" / SAMPLE)
    for fov in GOLDEN_IDS:
        metadata = grids[fov].metadata
        save_volume(merged_image(shape), root / "images" / "ref_merged" / f"{fov}.tif", metadata=metadata)
        save_volume(stain_image(shape)[..., None], root / "checkpoints" / fov / "/".join(STAIN_FILE), metadata=metadata)
        write_assignment(assignment_directory(root / "checkpoints" / fov, conflicts_name),
                         with_record(results[fov], name=conflicts_name), metadata=metadata)
        molecules = results[fov].molecules
        write_molecules_csv(root / "fovs" / fov / "molecules.csv", spot_namespace(fov),
                            molecules[["z", "y", "x"]].to_numpy(), molecules.gene_id.astype(str),
                            molecules.spot_id.astype(str))
    return ExportFolder(dataset, result, results, grids)


def listing(folder):
    """Every file under a folder with its SHA-256, relative POSIX path → digest."""
    from starfinder._records import file_sha256
    return {p.relative_to(folder).as_posix(): file_sha256(p) for p in sorted(folder.rglob("*")) if p.is_file()}


def canonical(section):
    return json.dumps(section, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
