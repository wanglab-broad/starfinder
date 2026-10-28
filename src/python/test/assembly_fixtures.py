"""Hand-built assembly fixtures of design/validation/stitching-assembly.md ("Fixtures").

``assembly_golden``: the three FOVs of ``test_workflow_assembly_golden.py`` (8×48×64 voxels,
unit spacing, nine scene cells, 24 molecules, 3D and 2D) assigned per FOV in memory, with a
``PlacementResult`` constructed here: every FOV ``image_supported`` at the float64 sums of
the fixture's ``STITCHING`` translations and grid steps (FOV2 at (1.3, 37.4, 0.7), FOV3 at
(−0.6, −0.3, 55.4) µm), nominal positions of ``positions_from_grid`` (``nominal_z`` 0),
review ``not_needed``. ``own_cases``: two FOVs of 4×16×16 at x 0 and 12 µm (centres x 7.5
and 19.5, boundary x 13.5) with box cells and molecules placed so that every status, reason
and flag and an exact tie at x 13.5 occur, a retained ``excluded_no_nucleus`` cell (FOV1
has nuclei) and an expanded FOV (FOV2, 1 µm planar); its expected rows are written below.
``status_cases``, ``empty_cases`` and ``placeholder_grid`` are built from these pieces.
Nothing is read from a run directory, and no random number is used.
"""
import json
from dataclasses import replace

import numpy as np
import pandas as pd

from starfinder.assembly import AssemblyInput
from starfinder.assignment import AssignmentConfig, MoleculeTable, assign_molecules
from starfinder.image import ImageMetadata
from starfinder.segmentation import ExpandLabelsConfig, ReferenceGrid, SegmentationResult
from starfinder.stitching import PlacementResult, approve_placement, placement_from_table, positions_from_grid
from starfinder.stitching._placement import FOV_COLUMNS, empty_links, frame_record, input_record

from .test_assignment_golden import GENES as GOLDEN_GENE_ROWS
from .test_workflow_assembly_golden import FOVS, SCENE_CELLS, STEP_X, STEP_Y, STITCHING, TILE_ZYX, fov_inputs

IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
DATASET, NAMESPACE_SAMPLE = "data", "sample"
SAMPLE = "sample1"
GOLDEN_GENES = tuple(g for g, _ in GOLDEN_GENE_ROWS)
GOLDEN_IDS = [FOVS[k] for k in sorted(FOVS)]
# The integer offsets at which test_workflow_assembly_golden.py cuts each FOV (legacy_tile_config's offsets).
CUT = {1: (0, 0, 0), 2: (1, 37, 0), 3: (0, 0, 55)}
# The setup (grid row of GRID_CSV) of each FOV number: FOV1 setup 0 (column 0, row 0), FOV2 setup 1
# (column 0, row 1), FOV3 setup 2 (column 1, row 0).
SETUP = {1: 0, 2: 1, 3: 2}


def ns(*parts):
    return json.dumps(list(parts), separators=(",", ":"))


def spot_namespace(fov):
    return ns(DATASET, NAMESPACE_SAMPLE, fov, None)


def cell_namespace(fov, run="cell"):
    return ns(DATASET, NAMESPACE_SAMPLE, fov, None, run)


def metadata(spacing=(1.0, 1.0, 1.0), *, unit="µm", frame="fov"):
    return ImageMetadata(frame, spacing, (0.0, 0.0, 0.0), IDENTITY, unit)


def unit_grid(shape, spacing=(1.0, 1.0, 1.0)):
    return ReferenceGrid(tuple(shape), metadata(spacing), "declared")


def molecules(fov, rows, genes, *, prefix="m"):
    """A MoleculeTable of ((z, y, x), gene) rows of one FOV with the spot IDs m0, m1, ..."""
    namespace = spot_namespace(fov)
    table = pd.DataFrame({
        "spot_namespace": pd.array([namespace] * len(rows), dtype="string"),
        "spot_id": pd.array([f"{prefix}{i}" for i in range(len(rows))], dtype="string"),
        "z": np.array([float(p[0]) for p, _ in rows], np.float64),
        "y": np.array([float(p[1]) for p, _ in rows], np.float64),
        "x": np.array([float(p[2]) for p, _ in rows], np.float64),
        "gene_id": pd.array([g for _, g in rows], dtype="string")})
    return MoleculeTable(table, tuple(genes), "final", {"spot_namespace": namespace})


def label_run(labels, fov, grid, *, target="cell", run="cell"):
    """A SegmentationResult of a ZYX label array on grid (a plane on its max projection)."""
    labels = np.ascontiguousarray(labels, np.uint32)
    record = {"run": run, "target": target, "operations": [], "seeds": None, "input": None}
    if labels.shape[0] == 1 and grid.shape_zyx[0] > 1:
        grid = grid.projected(method="max")
        record["input"] = {"projection": {"axis": "z", "method": "max"}}
    geometry = "plane" if labels.shape[0] == 1 else "volume"
    return SegmentationResult(labels, grid, target, geometry, ns(DATASET, NAMESPACE_SAMPLE, fov, None, run), record)


def placement(sample, ids, grids, origins, nominal, *, statuses=None, table=None, review="not_needed"):
    """A PlacementResult constructed in the test: one row per FOV in ids order."""
    statuses = statuses or {}
    rows = []
    for k, fov in enumerate(ids):
        status = statuses.get(fov, "image_supported")
        origin = origins[fov] if status not in ("missing", "failed") else (np.nan, np.nan, np.nan)
        rows.append({"fov_id": fov, "status": status, **{f"origin_{a}": v for a, v in zip("zyx", origin)},
                     **{f"nominal_{a}": v for a, v in zip("zyx", nominal[fov])},
                     **{f"correction_{a}": o - n for a, o, n in zip("zyx", origin, nominal[fov])},
                     **{f"method_translation_{a}": np.nan for a in "zyx"},
                     "component": -1 if status == "missing" else 0, "n_links_kept": 0, "flags": ""})
    record = {"format_version": 1, "stage": "stitching", "sample": sample, "frame": frame_record(sample),
              "gauge": "G1", "method": {"stage": "stitching", "method": "constructed", "config_type": "",
                                        "implementation": "test", "config": {}, "requires": {}, "artifacts": []},
              "table": {} if table is None else table.record,
              "inputs": {fov: input_record("constructed", None, grids[fov]) for fov in ids},
              "fallback": None,
              "review": {"state": review, "reasons": [] if review == "not_needed" else ["unplaced_fovs"],
                         "policy": None, "flagged": {}, "approvals": []}}
    return PlacementResult(sample, pd.DataFrame(rows, columns=list(FOV_COLUMNS)), empty_links(), record)


# --- assembly_golden --------------------------------------------------------------------------------------

def golden_table():
    """positions_from_grid of the 2 × 2 grid: steps 36 (Y) and 56 (X) µm, FOV k at column k // 2, row k % 2."""
    return positions_from_grid(columns=2, rows=2, order="column_major_down_right", tile_shape_yx=TILE_ZYX[1:],
                               spacing_yx=(1.0, 1.0), overlap=(0.25, 0.125), fov_ids=GOLDEN_IDS)


def golden_origins():
    """FOV → origin (z, y, x): STITCHING translation plus the grid step of its setup, float64 sums."""
    origins = {}
    for number, fov in FOVS.items():
        sx, sy, sz = STITCHING[SETUP[number]]
        column, row = divmod(SETUP[number], 2)
        origins[fov] = (sz, sy + STEP_Y * row, sx + STEP_X * column)
    return origins


def golden_placement(*, statuses=None, review="not_needed", origins=None, grids=None):
    table = golden_table()
    grids = grids or {fov: unit_grid(TILE_ZYX) for fov in GOLDEN_IDS}
    nominal = {fov: tuple(v) for fov, v in zip(table.fov_ids, table.nominal_zyx().tolist())}
    return placement(SAMPLE, GOLDEN_IDS, grids, origins or golden_origins(), nominal, statuses=statuses,
                     table=table, review=review)


def golden_pieces(dimensions="3d"):
    """FOV → (label image (ZYX or 1×Y×X), the scene cells in local label order, local molecules)."""
    frame = pd.DataFrame({"id": list(CUT), "z": [c[0] for c in CUT.values()], "y": [c[1] for c in CUT.values()],
                          "x": [c[2] for c in CUT.values()]})
    out = {}
    for number, (labels, present, local) in fov_inputs(frame).items():
        out[FOVS[number]] = (labels if dimensions == "3d" else labels.max(axis=0)[None], present, local)
    return out


def golden_results(dimensions="3d", *, expand=(), labels=None, molecules_of=None, nuclei=None):
    """FOV → AssignmentResult of the golden FOVs, assigned in memory.

    expand: FOVs assigned with a 2 µm planar expansion; labels / molecules_of / nuclei map a
    FOV to replacement labels, local molecules or nucleus labels.
    """
    grid = unit_grid(TILE_ZYX)
    results = {}
    for fov, (own_labels, _, local) in golden_pieces(dimensions).items():
        own_labels = (labels or {}).get(fov, own_labels)
        rows = (molecules_of or {}).get(fov, local)
        config = AssignmentConfig(expansion=ExpandLabelsConfig(2.0, "um", "planar") if fov in expand else None)
        nucleus = (nuclei or {}).get(fov)
        results[fov] = assign_molecules(molecules(fov, rows, GOLDEN_GENES), label_run(own_labels, fov, grid),
                                        grid=grid, config=config,
                                        nuclei=None if nucleus is None else label_run(nucleus, fov, grid,
                                                                                      target="nucleus",
                                                                                      run="nucleus"))
    return results


def inputs_of(results, grids=None, *, kind="assignment"):
    grids = grids or {}
    items = []
    for fov, result in results.items():
        grid = grids.get(fov, unit_grid(TILE_ZYX))
        items.append(AssemblyInput(fov, grid, assignment=result) if kind == "assignment"
                     else AssemblyInput(fov, grid, molecules=result))
    return items


def golden_molecule_tables():
    return {fov: molecules(fov, local, GOLDEN_GENES) for fov, (_, _, local) in golden_pieces().items()}


def scene_cell(fov, cell_id):
    """The scene cell of a golden FOV's local label."""
    number = {v: k for k, v in FOVS.items()}[fov]
    return SCENE_CELLS[number][int(cell_id) - 1]


# --- own_cases --------------------------------------------------------------------------------------------

OWN_SHAPE = (4, 16, 16)
OWN_GENES = ("A", "B", "C")
OWN_IDS = ["FOV1", "FOV2"]
# Box cells: label -> ((z0, z1), (y0, y1), (x0, x1)) inclusive, local index coordinates.
OWN_CELLS = {
    "FOV1": {1: ((0, 1), (1, 3), (1, 3)),      # c1: touches z 0 (z_edge); retained
             2: ((1, 2), (1, 3), (12, 14)),    # A: centroid x 13 (FOV1); conflicts with FOV2's B
             3: ((1, 2), (6, 8), (11, 13)),    # E1: the FOV1 view of the cell FOV2 sees as E2; retained
             4: ((1, 2), (12, 14), (2, 4))},   # N: no nucleus -> excluded_no_nucleus, retained by assembly
    "FOV2": {1: ((1, 2), (1, 3), (0, 4)),      # B: global x 12-16, expanded centroid x 14.32 -> FOV2
             2: ((1, 2), (6, 8), (0, 1)),      # E2: global x 12-13, expanded centroid x 12.85 -> FOV1 (E1 there)
             3: ((1, 2), (11, 13), (0, 0)),    # C: global x 12, expanded centroid x 12.375 -> FOV1, nothing there
             4: ((1, 2), (6, 8), (6, 8))},     # D: global x 18-20 -> FOV2
}
OWN_NUCLEI = {"FOV1": {1: ((0, 1), (2, 2), (2, 2)), 2: ((1, 2), (2, 2), (13, 13)), 3: ((1, 2), (7, 7), (12, 12))}}
# Local molecules ((z, y, x), gene) per FOV; spot IDs m0, m1, ... in this order.
OWN_MOLECULES = {
    "FOV1": [((1, 2, 2), "A"),          # m0 in c1
             ((1, 2, 14), "B"),         # m1 in A, position x 14 -> owned by FOV2: outside_owner_region
             ((1, 7, 12), "C"),         # m2 in E1
             ((1, 13, 3), "A"),         # m3 in N: §2.9 excluded_cell, retained with its cell
             ((2, 10, 13.5), "B"),      # m4 unassigned on the exact tie x 13.5 -> FOV1 (table order)
             ((2, 10, 14.2), "C"),      # m5 unassigned, owned by FOV2 -> not_owner
             ((-0.7, 5, 5), "A")],      # m6 outside_grid (z)
    "FOV2": [((1, 2, 2), "A"),          # m0 in B
             ((1, 2, 0.6), "B"),        # m1 in B, global x 12.6 -> owned by FOV1: outside_owner_region
             ((1, 7, 0), "C"),          # m2 in E2 -> cell_not_owner
             ((1, 12, 0), "A"),         # m3 in C -> cell_not_owner, cell_unclaimed
             ((1, 7, 7), "B"),          # m4 in D
             ((2, 14, 8), "C"),         # m5 unassigned, global x 20 -> FOV2, retained
             ((2, 10, 1.5), "A"),       # m6 unassigned on the exact tie (global x 13.5) -> FOV1: not_owner
             ((1, 7, 16.2), "B")],      # m7 outside_grid (x), global x 28.2: no owner
}
# Expected cell rows: (fov, cell_id) -> (§2.9 status, assembly_status, reason, owner, flags, conflict_with)
OWN_EXPECTED_CELLS = {
    ("FOV1", 1): ("kept", "retained", None, "FOV1", "z_edge", ""),
    ("FOV1", 2): ("kept", "retained", None, "FOV1", "conflict", "s1_FOV2_1"),
    ("FOV1", 3): ("kept", "retained", None, "FOV1", "", ""),
    ("FOV1", 4): ("excluded_no_nucleus", "retained", None, "FOV1", "", ""),
    ("FOV2", 1): ("kept", "retained", None, "FOV2", "fov_edge;conflict", "s1_FOV1_2"),
    ("FOV2", 2): ("kept", "excluded", "not_owner", "FOV1", "fov_edge", ""),
    ("FOV2", 3): ("kept", "excluded", "not_owner", "FOV1", "fov_edge;unclaimed", ""),
    ("FOV2", 4): ("kept", "retained", None, "FOV2", "", ""),
}
# Expected molecule rows: (fov, spot_id) -> (§2.9 status, cell_id, assembly_status, reason, owner, flags)
OWN_EXPECTED_MOLECULES = {
    ("FOV1", "m0"): ("assigned", 1, "retained", None, "FOV1", ""),
    ("FOV1", "m1"): ("assigned", 2, "retained", None, "FOV2", "outside_owner_region"),
    ("FOV1", "m2"): ("assigned", 3, "retained", None, "FOV1", ""),
    ("FOV1", "m3"): ("excluded_cell", 4, "retained", None, "FOV1", ""),
    ("FOV1", "m4"): ("unassigned", None, "retained", None, "FOV1", ""),
    ("FOV1", "m5"): ("unassigned", None, "excluded", "not_owner", "FOV2", ""),
    ("FOV1", "m6"): ("outside_grid", None, "excluded", "outside_grid", "FOV1", ""),
    ("FOV2", "m0"): ("assigned", 1, "retained", None, "FOV2", ""),
    ("FOV2", "m1"): ("assigned", 1, "retained", None, "FOV1", "outside_owner_region"),
    ("FOV2", "m2"): ("assigned", 2, "excluded", "cell_not_owner", "FOV1", ""),
    ("FOV2", "m3"): ("assigned", 3, "excluded", "cell_not_owner", "FOV1", "cell_unclaimed"),
    ("FOV2", "m4"): ("assigned", 4, "retained", None, "FOV2", ""),
    ("FOV2", "m5"): ("unassigned", None, "retained", None, "FOV2", ""),
    ("FOV2", "m6"): ("unassigned", None, "excluded", "not_owner", "FOV1", ""),
    ("FOV2", "m7"): ("outside_grid", None, "excluded", "outside_grid", None, ""),
}

# Under the default duplicate rule (duplicate_threshold 0.5): FOV1's A (18 voxels) lies whole inside FOV2's B
# (56 voxels of its expanded territory), a share of 1.0 of the smaller, so A is excluded duplicate_observation naming
# B and its molecule m1 with it; B keeps no conflict. The other rows are those above (the rule off).
OWN_DUPLICATE_CELLS = {
    ("FOV1", 2): ("kept", "excluded", "duplicate_observation", "FOV1", "", ""),
    ("FOV2", 1): ("kept", "retained", None, "FOV2", "fov_edge", ""),
}
OWN_DUPLICATE_MOLECULES = {
    ("FOV1", "m1"): ("assigned", 2, "excluded", "cell_duplicate_observation", "FOV2", ""),
}


def paint(shape, boxes):
    labels = np.zeros(shape, np.uint32)
    for value, ((z0, z1), (y0, y1), (x0, x1)) in boxes.items():
        labels[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1] = value
    return labels


def own_table(nominal_z=(0.0, 0.0)):
    from starfinder.stitching import positions_from_table
    return positions_from_table(pd.DataFrame({"fov_id": OWN_IDS, "nominal_z": list(nominal_z),
                                              "nominal_y": [0.0, 0.0], "nominal_x": [0.0, 12.0]}))


def own_grids():
    return {fov: unit_grid(OWN_SHAPE) for fov in OWN_IDS}


def own_results():
    grid = unit_grid(OWN_SHAPE)
    results = {}
    for fov in OWN_IDS:
        nuclei = OWN_NUCLEI.get(fov)
        config = AssignmentConfig(expansion=ExpandLabelsConfig(1.0, "um", "planar") if fov == "FOV2" else None)
        results[fov] = assign_molecules(
            molecules(fov, OWN_MOLECULES[fov], OWN_GENES), label_run(paint(OWN_SHAPE, OWN_CELLS[fov]), fov, grid),
            grid=grid, config=config,
            nuclei=None if nuclei is None else label_run(paint(OWN_SHAPE, nuclei), fov, grid, target="nucleus",
                                                         run="nucleus"))
    return results


def own_placement(origins=None, nominal_z=(0.0, 0.0), approve=True):
    result = placement_from_table(own_table(nominal_z), own_grids(), sample="s1", origins=origins)
    return approve_placement(result, reviewer="test", note="own_cases") if approve else result


def own_inputs():
    return inputs_of(own_results(), {fov: unit_grid(OWN_SHAPE) for fov in OWN_IDS})


# --- status_cases -----------------------------------------------------------------------------------------

# The placeholder FOV2's own input: box cells and molecules in its local coordinates (origin (1.3, 37.4, 0.7)).
PLACEHOLDER_CELLS = {1: ((3, 5), (2, 4), (38, 40)),     # centroid global (5.3, 40.4, 39.7): FOV1 owns, retains cell 5
                     2: ((3, 5), (20, 22), (10, 12))}   # centroid global y 58.4: no owning FOV covers it
PLACEHOLDER_MOLECULES = [((4, 3, 39), "A"),       # in cell 1 -> cell_not_owner
                         ((4, 21, 11), "B"),      # in cell 2 -> cell_not_owner, cell_unclaimed
                         ((4, 5, 20), "C"),       # unassigned at global y 42.4: FOV1 owns -> not_owner
                         ((4, 30, 20), "D"),      # unassigned at global y 67.4: no owner -> not_owner
                         ((4, 30, 70), "E")]      # outside_grid (x 70 >= 64)


def placeholder_result():
    grid = unit_grid(TILE_ZYX)
    fov = "Position002"
    return assign_molecules(molecules(fov, PLACEHOLDER_MOLECULES, GOLDEN_GENES),
                            label_run(paint(TILE_ZYX, PLACEHOLDER_CELLS), fov, grid), grid=grid)


def approved(result):
    return approve_placement(result, reviewer="test", note="status_cases")


# --- placeholder_grid -------------------------------------------------------------------------------------

GRID3_SHAPE = (4, 16, 16)
GRID3_IDS = [f"G{k}" for k in range(9)]                  # column-major: G k at column k // 3, row k % 3
GRID3_PLACEHOLDERS = {"G0": (0, 0), "G5": (2, 1)}       # (row, column)
# The six owning FOVs that overlap a placeholder: (row, column) -> box cell (local, inclusive), whose
# ownership centroid lies in the placeholder's footprint and is nearest to its own centre.
GRID3_CELLS = {(0, 1): ((1, 2), (4, 6), (0, 1)),     # centroid (5, 12.5) in G0's footprint
               (1, 0): ((1, 2), (0, 1), (4, 6)),     # (12.5, 5) in G0's
               (1, 1): ((1, 2), (13, 14), (6, 8)),   # (25.5, 19) in G5's
               (1, 2): ((1, 2), (11, 13), (2, 4)),   # (24, 27) in G5's (and in G4's and G8's footprints)
               (2, 0): ((1, 2), (7, 9), (12, 14)),   # (32, 13) in G5's
               (2, 2): ((1, 2), (7, 9), (1, 3))}     # (32, 26) in G5's


def grid3_table():
    return positions_from_grid(columns=3, rows=3, order="column_major_down_right", tile_shape_yx=GRID3_SHAPE[1:],
                               spacing_yx=(1.0, 1.0), overlap=0.25, fov_ids=GRID3_IDS)


def grid3_cell(fov):
    k = GRID3_IDS.index(fov)
    return k % 3, k // 3


def grid3_placement(*, all_placeholder=False):
    table = grid3_table()
    grids = {fov: unit_grid(GRID3_SHAPE) for fov in GRID3_IDS}
    nominal = {fov: tuple(v) for fov, v in zip(table.fov_ids, table.nominal_zyx().tolist())}
    statuses = {fov: "placeholder" for fov in (GRID3_IDS if all_placeholder else GRID3_PLACEHOLDERS)}
    result = placement("grid3", GRID3_IDS, grids, nominal, nominal, statuses=statuses, table=table)
    return approve_placement(result, reviewer="test", note="placeholder_grid")


def grid3_results():
    grid = unit_grid(GRID3_SHAPE)
    results = {}
    for fov in GRID3_IDS:
        if fov in GRID3_PLACEHOLDERS:
            continue
        cell = GRID3_CELLS.get(grid3_cell(fov))
        boxes = {} if cell is None else {1: cell}
        rows = [] if cell is None else [((1.5, (cell[1][0] + cell[1][1]) / 2, (cell[2][0] + cell[2][1]) / 2), "A")]
        results[fov] = assign_molecules(molecules(fov, rows, ("A",)), label_run(paint(GRID3_SHAPE, boxes), fov, grid),
                                        grid=grid)
    return results


def grid3_raster():
    values = np.arange(-0.5, 39.5 + 1e-9, 0.25)
    y, x = np.meshgrid(values, values, indexing="ij")
    return np.stack([np.zeros(y.size), y.ravel(), x.ravel()], axis=1)


def with_record(result, **changes):
    """An AssignmentResult with record keys changed (for example its saved name)."""
    return replace(result, record=dict(result.record, **changes))


# --- Count accounting, evaluated against the inputs ---------------------------------------------------------

def _none(value):
    return None if pd.isna(value) else value


def check_accounting(result, inputs, placement):
    """Identities 1 to 5 of design/specs/assembly.md ("Count accounting"), each recomputed from the inputs.

    inputs maps a fov_id to its AssignmentResult (``cells`` mode) or MoleculeTable
    (``molecules_only``); nothing here reads AssemblyResult.__post_init__ or its helpers.
    """
    cells, mols, counts = result.cells, result.molecules, result.counts
    placed = {r.fov_id: r.status in ("image_supported", "metadata_only", "placeholder")
              for r in placement.fovs.itertuples()}
    owning = {r.fov_id: r.status in ("image_supported", "metadata_only") for r in placement.fovs.itertuples()}
    cell_mode = all(not isinstance(v, MoleculeTable) for v in inputs.values())
    # 1. Anti-joins of the input keys against the sample tables, both ways; no key repeats.
    input_cells, input_molecules = [], []
    for fov, given in inputs.items():
        table = given.molecules if cell_mode else given.table
        input_molecules += list(zip(table.spot_namespace, table.spot_id))
        if cell_mode:
            input_cells += [(given.cell_namespace, int(c)) for c in given.cells.cell_id]
    sample_cells = list(zip(cells.cell_namespace, cells.cell_id.astype(int)))
    sample_molecules = list(zip(mols.spot_namespace, mols.spot_id))
    assert len(set(sample_cells)) == len(sample_cells) and len(set(sample_molecules)) == len(sample_molecules)
    assert set(sample_cells) == set(input_cells) and set(sample_molecules) == set(input_molecules)
    assert len(sample_cells) == len(input_cells) and len(sample_molecules) == len(input_molecules)
    assert set(cells.assembly_status) <= {"retained", "excluded"} and set(mols.assembly_status) <= {"retained",
                                                                                                       "excluded"}
    # 2. Per FOV and in total: inputs = retained + Σ excluded by reason; the record holds the totals.
    for fov in inputs:
        for frame, n in ((cells, len(inputs[fov].cells) if cell_mode else 0),
                         (mols, len(inputs[fov].molecules if cell_mode else inputs[fov].table))):
            rows = frame[frame.fov_id == fov]
            reasons = rows[rows.assembly_status == "excluded"].assembly_reason.value_counts()
            assert n == len(rows) == int((rows.assembly_status == "retained").sum()) + int(reasons.sum())
    record = result.record["counts"]
    # The duplicate rule's reasons are counted when the rule is on (duplicate_threshold not None).
    duplicates = result.record["config"].get("duplicate_threshold") is not None
    for name, frame, reasons in (("cells", cells, ("not_owner", "fov_not_placed")
                                  + (("duplicate_observation",) if duplicates else ())),
                                 ("molecules", mols, ("cell_not_owner", "not_owner", "outside_grid", "fov_not_placed")
                                  + (("cell_duplicate_observation",) if duplicates else ()))):
        excluded = frame[frame.assembly_status == "excluded"]
        assert record[name] == {"input": len(frame), "retained": int((frame.assembly_status == "retained").sum()),
                                "excluded": {r: int((excluded.assembly_reason == r).sum()) for r in reasons}}
        assert set(excluded.assembly_reason) <= set(reasons) and frame[frame.assembly_status == "retained"
                                                                        ].assembly_reason.isna().all()
    # 2 (continued): a cell excluded duplicate_observation, and only such a cell, names a retained cell of another
    # FOV in duplicate_of.
    if "duplicate_of" in cells:
        state = {r.obs_name: (r.fov_id, r.assembly_status) for r in cells.itertuples()}
        for r in cells.itertuples():
            named = _none(r.duplicate_of)
            if _none(r.assembly_reason) == "duplicate_observation":
                assert named in state and state[named][1] == "retained" and state[named][0] != r.fov_id
            else:
                assert named is None
    # 3. Molecules follow their cell; unassigned ones their own position; outside_grid is excluded.
    cell_of = {(r.fov_id, int(r.cell_id)): (r.assembly_status, _none(r.assembly_reason)) for r in cells.itertuples()}
    for r in mols.itertuples():
        status, reason, owner = r.assembly_status, _none(r.assembly_reason), _none(r.owner_fov_id)
        a_status = _none(r.assignment_status)
        if not placed[r.fov_id]:
            assert (status, reason) == ("excluded", "fov_not_placed")
        elif a_status in ("assigned", "excluded_cell"):
            cell_status, cell_reason = cell_of[(r.fov_id, int(r.cell_id))]
            assert status == cell_status
            assert reason == (None if cell_status == "retained" else
                              "cell_not_owner" if cell_reason == "not_owner" else
                              "cell_duplicate_observation" if cell_reason == "duplicate_observation" else cell_reason)
        elif a_status == "outside_grid" or reason == "outside_grid":
            assert (status, reason) == ("excluded", "outside_grid")
        else:
            assert (status == "retained") == (owner == r.fov_id)
            if not owning[r.fov_id]:
                assert status == "excluded"
    # 3 (continued): an owner is always an owning FOV with an input, never a placeholder.
    for frame in (cells, mols):
        owners = {o for o in frame.owner_fov_id.astype(object) if not pd.isna(o)}
        assert owners <= {f for f in inputs if owning[f]}, owners
    if not cell_mode:
        return
    # 4. The counted cells' counts equal their per-FOV §2.9 counts; Σ whole = retained assigned = Σ n_molecules.
    counted = cells[(cells.assembly_status == "retained") & (cells.status == "kept")]
    counted_keys = {(r.fov_id, int(r.cell_id)) for r in counted.itertuples()}
    expected = sorted((fov, int(r.cell_id), r.gene_id, r.compartment, int(r.count))
                      for fov, given in inputs.items() for r in given.counts.itertuples()
                      if (fov, int(r.cell_id)) in counted_keys)
    got = sorted((r.fov_id, int(r.cell_id), r.gene_id, r.compartment, int(r.count)) for r in counts.itertuples())
    assert got == expected
    whole = sum(c for *_, compartment, c in got if compartment == "whole")
    retained_assigned = int(((mols.assignment_status == "assigned") & (mols.assembly_status == "retained")).sum())
    assert whole == retained_assigned == int(counted.n_molecules.sum())
    # 5. The §2.9 compartment identities over the counted cells, for every compartments state, and per molecule.
    state = {(r.fov_id, int(r.cell_id)): r.compartments for r in counted.itertuples()}
    available = {k for k, v in state.items() if v == "available"}
    split = {}
    for fov, cell, gene, compartment, n in got:
        if compartment != "whole":
            assert (fov, cell) in available, (fov, cell, state.get((fov, cell)))
            split[(fov, cell, gene)] = split.get((fov, cell, gene), 0) + n
    assert split == {(f, c, g): n for f, c, g, comp, n in got if comp == "whole" and (f, c) in available}
    given = {}
    for fov, result_29 in inputs.items():
        for r in result_29.molecules.itertuples():
            given[(r.spot_namespace, r.spot_id)] = _none(r.compartment)
    retained_assigned = mols[(mols.assignment_status == "assigned") & (mols.assembly_status == "retained")]
    populations = {}
    held = {}
    for r in mols.itertuples():
        compartment = _none(r.compartment)
        assert compartment == given[(r.spot_namespace, r.spot_id)]        # copied from the input molecule
        assert (compartment is not None) == (_none(r.assignment_status) == "assigned")
    for r in retained_assigned.itertuples():
        cell_state = state[(r.fov_id, int(r.cell_id))]
        assert r.compartment in (("nucleus", "cytoplasm") if cell_state == "available" else (cell_state,))
        populations[r.compartment] = populations.get(r.compartment, 0) + 1
        if r.compartment in ("nucleus", "cytoplasm"):
            key = (r.fov_id, int(r.cell_id), r.gene_id, r.compartment)
            held[key] = held.get(key, 0) + 1
    assert sum(populations.values()) == len(retained_assigned)
    assert {(f, c, g, comp): n for f, c, g, comp, n in got if comp != "whole"} == held


# --- own_states: own_cases with counted cells in every compartments state ---------------------------------

# FOV1 with the exclusion off (cell 4 is kept: no_nucleus) and nucleus 3 one voxel past E1's border
# (x 14 lies outside E1: nucleus_outside_cell, so E1 is withheld); cells 1 and 2 stay available, FOV2 unavailable.
OWN_STATES_NUCLEI = {1: ((0, 1), (2, 2), (2, 2)), 2: ((1, 2), (2, 2), (13, 13)), 3: ((1, 2), (7, 7), (12, 14))}


def own_states_results():
    results = own_results()
    grid = unit_grid(OWN_SHAPE)
    fov = "FOV1"
    results[fov] = assign_molecules(
        molecules(fov, OWN_MOLECULES[fov], OWN_GENES), label_run(paint(OWN_SHAPE, OWN_CELLS[fov]), fov, grid),
        grid=grid, config=AssignmentConfig(exclude_cells_without_nucleus=False),
        nuclei=label_run(paint(OWN_SHAPE, OWN_STATES_NUCLEI), fov, grid, target="nucleus", run="nucleus"))
    return results


# --- The duplicate rule recomputed in the test, on the whole sample grid --------------------------------------

def placed_on_grid(territory, origin, spacing, grid_origin, grid_shape):
    """A ZYX territory placed on a sample grid by the nearest voxel, computed here on the whole grid.

    Grid voxel ``v`` (at ``grid_origin + v·s``) shows the FOV voxel ``floor(c + ½)`` with ``c = (grid_origin +
    v·s − origin)/s`` where ``0 ≤ c ≤ n − 1`` (a margin of 1e-9 voxel), and 0 elsewhere; a one-plane territory is
    placed in Y and X on a grid of one plane.
    """
    out = np.zeros(grid_shape, np.int64)
    grid_index, fov_index = [], []
    for a in range(3):
        if a == 0 and territory.shape[0] == 1:
            grid_index.append(np.zeros(1, np.int64))
            fov_index.append(np.zeros(1, np.int64))
            continue
        v = np.arange(grid_shape[a])
        c = (grid_origin[a] + v * spacing[a] - origin[a]) / spacing[a]
        inside = (c >= -1e-9) & (c <= territory.shape[a] - 1 + 1e-9)
        grid_index.append(v[inside])
        fov_index.append(np.floor(c[inside] + 0.5).astype(np.int64))
    out[np.ix_(*grid_index)] = territory[np.ix_(*fov_index)]
    return out


def duplicate_oracle(territories, retained, origins, spacing, order, threshold=0.5):
    """The duplicate rule of design/specs/assembly-algorithms.md recomputed from the territories, for the tests.

    ``territories`` maps each FOV with a retained cell to its ZYX territory (1×Y×X for ``plane``), ``retained`` each
    of them to its retained ``cell_id`` set, ``origins`` every included FOV (``image_supported``,
    ``metadata_only``) to its origin, ``order`` the FOVs in placement table order. Each FOV's retained cells are
    placed alone on one grid (spacing ``spacing``, origin the smallest included origin per axis, up to the last voxel
    centre), and two cells of two FOVs share the grid voxels where both show. Returns ``(sizes, shares, excluded)``:
    voxels per (fov, cell), ``{((f, a), (g, b)): (shared voxels, share of the smaller)}`` for every pair with a shared
    voxel, ``f`` before ``g`` in ``order``, and ``{excluded (fov, cell): kept (fov, cell)}``.
    """
    plane = all(t.shape[0] == 1 for t in territories.values())
    grid_origin = [min(o[a] for o in origins.values()) for a in range(3)]
    shape = []
    for a in range(3):
        last = max(origins[f][a] + (territories[f].shape[a] - 1) * spacing[a] for f in territories)
        shape.append(1 if a == 0 and plane else int(np.floor((last - grid_origin[a]) / spacing[a] + 1e-9)) + 1)
    fovs = [f for f in order if f in territories]
    canvases = {f: placed_on_grid(np.where(np.isin(territories[f], sorted(retained[f])), territories[f], 0),
                                  origins[f], spacing, grid_origin, shape) for f in fovs}
    sizes = {(f, c): int((canvases[f] == c).sum()) for f in fovs for c in retained[f]}
    shares = {}
    for k, f in enumerate(fovs):
        for g in fovs[k + 1:]:
            both = (canvases[f] > 0) & (canvases[g] > 0)
            pairs, counts = np.unique(np.stack([canvases[f][both], canvases[g][both]]), axis=1, return_counts=True)
            for (a, b), n in zip(pairs.T.tolist(), counts.tolist()):
                shares[((f, a), (g, b))] = (n, n / min(sizes[(f, a)], sizes[(g, b)]))
    partners = {}
    for (a, b), (_, share) in shares.items():
        if share >= threshold:
            partners.setdefault(a, set()).add(b)
            partners.setdefault(b, set()).add(a)
    rank = sorted(sizes, key=lambda cell: (-sizes[cell], order.index(cell[0]), cell[1]))
    kept, excluded = [], {}
    for cell in rank:
        staying = [p for p in kept if p in partners.get(cell, ())]
        if staying:
            excluded[cell] = staying[0]
        else:
            kept.append(cell)
    return sizes, shares, excluded
