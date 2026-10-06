# Assembly contract: sample coordinates, ownership, retained and excluded populations and the sample tables

Status: Proposed

This page proposes the §2.10 assembly contract: a sample-level assemble entry separate
from stitch and from export, the rule that puts molecules and cells into sample
coordinates, ownership regions computed from the placement, the retention rule agreed on
2026-09-27, assembly statuses and reasons, flags for partial and conflicting cells,
sample-level identities, count accounting, the sample tables, the sample-level AnnData,
persistence and the workflow translation. It builds on two contracts and redefines
neither:

* the placement of {doc}`stitching-contract`: `PlacementResult` (per-FOV `status`,
  float64 `origin_z/y/x` in µm, `nominal_*`, links, record, review), the sample frame, the
  index-to-sample rule and `placed_metadata`, the failure policy and
  `PlacementReviewRequiredError`;
* the per-FOV tables of {doc}`assignment-contract`: `AssignmentResult` (molecule table,
  cell table, long counts, nuclei, label arrays, record), its statuses
  (`ASSIGNMENT_STATUSES`, `CELL_STATUSES`, `COMPARTMENT_STATES`), its keys
  `(spot_namespace, spot_id)` and `(cell_namespace, cell_id)`, `MoleculeTable`, the grid
  rule and its count identities.

Where this page lists a column of those tables, it copies it unchanged. The current
behavior is recorded in {doc}`assembly-baseline`; the numerical rules, parameters and
resources are in {doc}`assembly-algorithms`. Nothing here is implemented and nothing is
renamed. The fused image and the SpatialData export belong to W-328.

Evidence, cited by path:

* W-324, run directory
  `/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/runs/W-324/20261006T063540Z-5cb2f421`
  (found with `timeout 30 find …/runs/W-324 -maxdepth 2 -name stitching-manifest.json`):
  the LN parity outputs `parity/script_tile_config_ln.csv` and
  `parity/prototype/tile_config_ln/T2_sum.csv`.
* W-326, run directory
  `/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/runs/W-326/20261006T092110Z-48452974`:
  `tables/ln_ownership.json` (ownership areas of each option on LN, from the two records
  above, `scripts/ln_ownership.py`) and `tables/option_effects.json` (each option on the
  golden fixture, `scripts/option_effects.py`).

## Settled decisions this page follows

Decided by Jiahao in the §2.10 planning session (W-152 comment "§2.10 planning decisions
(2026-10-05)" and its reply "Amendment on the sample outputs"), and in the W-152 comments
of 2026-09-27 on §2.10:

* **Three sample-level entries** outside `PipelineConfig` and `FOV.run`: stitch,
  assemble, and fuse and export (D3). **Ownership regions are computed in assemble from
  the placement**, not stored in it.
* **Placement from either backend** enters as one `PlacementResult` (D1, D2).
* **Retention** (2026-09-27): a cell is retained by its centroid; all molecules of a
  retained cell are kept wherever they lie; an unassigned molecule is retained by its own
  position. Exclusions are preserved, not dropped. Cells at a FOV border are flagged.
  There is no matching or merging of cells across FOVs, no recovery of complete cells
  across borders and no reassignment of molecules. The workflow is FOV-first: per-FOV
  results are produced first, and assembly follows per sample.
* **The sample-level AnnData is a required default output of every assembled sample**
  (Jiahao, 2026-10-05, amendment), written from the sample tables through the §2.9
  AnnData code, without a second exporter. The fused image and SpatialData are W-328's.

Excluded: cross-FOV cell matching or merging, recovery of complete cells, reassignment
of molecules, the fused image and label export, and any change to the placement, label or
assignment contracts.

## Terms

* A **sample**, the **sample frame** (ZYX, µm, identity direction, `frame_id`
  `sample:<sample>`), a FOV's **reference grid**, **origin** and **placed** FOV are those
  of {doc}`stitching-contract`. A FOV is **placed** when its origin is finite
  (`image_supported`, `metadata_only`, `placeholder`).
* The **footprint** of a placed FOV is the region of the sample frame its reference grid
  covers: in Y and X the union of its voxel supports, `[origin − s/2, origin + (n − ½)·s)`
  per axis, with `n` the grid size and `s` the spacing.
* An **observation** is one row of a FOV's cell or molecule table. The same biological
  cell or molecule may be observed by several FOVs where they overlap.
* The **owner** of a sample-frame point is the placed FOV that ownership assigns it to
  ("Ownership regions"); a FOV's **ownership region** is the set of points it owns.
* An observation is **retained** when it represents the sample, and **excluded**
  otherwise; the **assembly status** records which, with a reason. It is distinct from the
  §2.9 assignment status (`assigned`, …) and cell status (`kept`, …), which assemble copies
  and never changes.

## Names

| Option | Names | Effect on the golden values | Effect on existing files and outputs |
| --- | --- | --- | --- |
| **A1. A `starfinder.assembly` module (recommended)** | Module `starfinder.assembly`; entry `assemble`; result `AssemblyResult`; input `AssemblyInput`; config `AssemblyConfig`; ownership `ownership_map`, `OwnershipMap`; vocabularies `ASSEMBLY_STATUSES`, `CELL_ASSEMBLY_REASONS`, `MOLECULE_ASSEMBLY_REASONS`, `CELL_ASSEMBLY_FLAGS`, `MOLECULE_ASSEMBLY_FLAGS`; persistence `save_assembly`, `load_assembly`; AnnData `sample_anndata`; coordination `Dataset.assemble` | None: `test_workflow_assembly_golden.py` pins the legacy scripts and the adapter through frozen helpers, which no option edits. | New files only, plus one method on `Dataset`; `docs/api/assembly.rst`, `docs/api/inventory.rst` and `docs/api/python-index.rst` gain the exports, which `docs/check_reference.py` requires. `tile_config_{sample}.csv` and the legacy per-FOV and sample files stay as they are while the legacy rules run. |
| A2. Inside `starfinder.stitching` | `starfinder.stitching.assemble` returning `SampleAssembly`; `Dataset.assemble` | None. | The same files; one module then holds placement and the observation populations, which D3 separates, and the stitching module gains a dependency on `starfinder.assignment`. |
| A3. One `starfinder.sample` module for assemble and export | `starfinder.sample.assemble`, `AssembledSample`; W-328's export beside it | None. | The same files; the module grows a second responsibility when W-328 adds the fused image and SpatialData export. |

**Recommendation: A1.** It mirrors the stage split of D3 (stitch, assemble, fuse and
export), follows W-325's recommendation that assemble keeps its own module
({doc}`stitching-contract`, "Names"), and keeps `starfinder.stitching` free of the §2.9
tables.

New names under A1:

| Concept | Name | Kind |
| --- | --- | --- |
| Entry | `assemble(placement, inputs, *, config=AssemblyConfig())` | function |
| Result | `AssemblyResult` | frozen dataclass |
| Input per FOV | `AssemblyInput(fov_id, grid, assignment=None, molecules=None)` | frozen dataclass |
| Config | `AssemblyConfig` | frozen dataclass |
| Ownership | `ownership_map(placement, grids, *, config)` returning `OwnershipMap` with `owner(points_zyx)` | function, frozen dataclass |
| Vocabularies | `ASSEMBLY_STATUSES = ("retained", "excluded")`; `CELL_ASSEMBLY_REASONS = ("not_owner", "fov_not_placed")`; `MOLECULE_ASSEMBLY_REASONS = ("cell_not_owner", "not_owner", "outside_grid", "fov_not_placed")`; `CELL_ASSEMBLY_FLAGS = ("fov_edge", "z_edge", "conflict", "unclaimed")`; `MOLECULE_ASSEMBLY_FLAGS = ("outside_owner_region", "cell_unclaimed")` | tuples |
| AnnData | `sample_anndata(result, *, annotation=None)`, through `counts_anndata` of `starfinder.assignment` ("Sample-level AnnData") | functions |
| Persistence | `save_assembly(result, folder, *, table_format="csv")`, `load_assembly(folder)` | functions |
| Coordination | `Dataset.assemble(sample, *, placement, assignment="default", config=AssemblyConfig(), output_dir=None)` | method |

## The assemble entry

### Function

```python
def assemble(placement: PlacementResult, inputs: Sequence[AssemblyInput], *,
             config: AssemblyConfig = AssemblyConfig()) -> AssemblyResult: ...

@dataclass(frozen=True)
class AssemblyInput:
    fov_id: str
    grid: ReferenceGrid                                  # the FOV's reference grid (the molecule run's)
    assignment: AssignmentResult | str | Path | None = None   # a §2.9 result, or its saved folder
    molecules: MoleculeTable | None = None               # a molecule table alone (no segmentation)

@dataclass(frozen=True)
class AssemblyConfig:
    ownership: str = "nearest_center"   # "nearest_center" (O1) or "midpoint_box" (O2)
    centroid: str = "territory"         # "territory" (the territory assign sampled) or "original"
    tie: str = "table_order"            # T1
    conflicts: bool = True              # compute the conflict and unclaimed flags (reads territories)
```

`assemble` checks its arguments, maps every observation into sample coordinates, computes
the ownership map from the placement, assigns one assembly status per cell and molecule,
computes the flags, copies the §2.9 counts of the retained cells, and returns an
`AssemblyResult`. It never reads or writes files except to load a saved assignment named
by path, never runs stitch, assign or any per-FOV stage, matches no cell across FOVs and
moves no molecule to another cell. Like `stitch`, it is a sample-level computation, not a
multi-FOV runner ({doc}`stitching-contract`, "Not the multi-FOV runner").

**Inputs.**

* **Placement**: a `PlacementResult` from `stitch` (`multiview_stitcher`) or from
  `import_bigstitcher`; the two are one type, so assemble does not know the backend except
  from the method record, which it copies.
* **Per FOV, one of two kinds**, the same kind for every FOV of the sample:
  * `assignment`: the §2.9 `AssignmentResult` of that FOV, in memory, or the path of its
    saved folder (`<checkpoint dir>/<fov_id>/assignment/<name>/`), loaded with the §2.9
    loader, which checks every recorded SHA-256. Assemble reads its molecule table, cell
    table, counts, record and, for the flags, its territories;
  * `molecules`: a `MoleculeTable` alone, for a sample without segmentation. Every
    molecule is then retained or excluded by its own position, and the sample has no
    cells ("Retention").
* `grid`: the FOV's reference grid, which places its index coordinates.

`Dataset.assemble(sample, *, placement, assignment="default", config, output_dir)` loads
the saved placement (`load_placement`), builds one `AssemblyInput` per FOV of the
placement from the saved assignment folders named `assignment` (or, with
`assignment=None`, from the saved molecule tables through `molecule_table`) and the
reference grids (`reference_grid_from_file`), calls `assemble`, and with `output_dir`
saves the result ("Persistence"). It keeps the tables of every FOV and at most two label
images in memory at a time ({doc}`assembly-algorithms`, "Resources").

### Checks the entry applies

In this order, before any coordinate is computed; each raises and names the FOVs:

1. **Config.** `AssemblyConfig` validates its four fields in `__post_init__`.
2. **Review.** A placement whose `record["review"]` is `required` and that holds no
   approval raises `PlacementReviewRequiredError` with the reasons
   ({doc}`stitching-contract`, "Failure policy", item 2).
3. **FOV identity.** Every `fov_id` is unique among the inputs and is a row of
   `placement.fovs` (`ValueError`). Every FOV whose placement status is
   `image_supported` or `metadata_only` must have an input (`ValueError`); an input of a
   `missing` or `failed` FOV is accepted and its observations are excluded with the reason
   `fov_not_placed`.
4. **Kind.** Each input gives exactly one of `assignment`, `molecules`, and all inputs give
   the same kind (`ValueError`). The kind is recorded as `mode`: `cells` or
   `molecules_only`.
5. **Observation identity.** The first four entries of each input's `spot_namespace` (and
   of its `cell_namespace`) are `[dataset_id, sample_id, fov_id, subtile_id]` with this
   `fov_id`, one `dataset_id` and `sample_id` for the sample, and `subtile_id` null:
   subtiles are not assembled (`ValueError`).
6. **Grid.** The input's `grid` equals, by the grid rule of {doc}`assignment-contract`,
   the grid the placement recorded for the FOV (`record["inputs"][fov_id]`) and the grid
   of the assignment record; otherwise `IncompatibleGeometryError` naming both. Its
   `ImageMetadata` must have known spacing, unit `µm` and identity direction, as
   `stitch` requires; assemble never assumes a pixel size.
7. **Genes.** Every input has the same `genes` tuple (`ValueError` naming the
   differences), which becomes the sample's gene list.
8. **One layer.** Every placed FOV has the same `nominal_z` (option Z1, "Ownership
   regions"); otherwise `ValueError("assemble handles one layer of tiles; FOVs … have
   nominal z …")`.

### What assemble records

The record of the result (`assembly.json` when saved) holds:

* the placement: its `sample`, frame, gauge, method record (backend, config), review
  state and approval, and the SHA-256 of its `fovs` and `links` tables as
  `placement.json` records them;
* per FOV: `fov_id`, placement status, origin, input kind, the assignment name and the
  SHA-256 of its `assignment.json` and of its molecule, cell and count tables, the
  molecule table's `sha256` (`MoleculeTable`), the SHA-256 of the territory label file read
  for the flags, the grid (shape, metadata, SHA-256) and its `calibration_source`
  (`grid` for a grid read from an image's metadata, `declared` for a grid the legacy
  adapter declared from `voxel_size_*`, never upgraded);
* the config, the ownership geometry with each FOV's footprint and centre, the counts of
  "Count accounting", and the SHA-256 of every table and file it writes;
* `format_version` 1 and `software` as in `run.json` ({doc}`checkpoints`).

```json
{
  "format_version": 1, "stage": "assembly", "sample": "sample1", "mode": "cells",
  "frame": {"frame_id": "sample:sample1", "unit": "µm", "direction": "identity"},
  "placement": {"method": "bigstitcher", "gauge": "G1", "review": "not_needed", "approval": null,
                "fovs_sha256": "...", "links_sha256": "..."},
  "config": {"ownership": "nearest_center", "centroid": "territory", "tie": "table_order", "conflicts": true},
  "fovs": [{"fov_id": "Position001", "placement_status": "image_supported", "origin_zyx": [0.0, 0.0, 0.0],
            "kind": "assignment", "assignment": {"name": "default", "record_sha256": "...", "tables": {}},
            "molecules_sha256": "...", "territories_sha256": "...",
            "grid": {"shape_zyx": [50, 1496, 1496], "sha256": "...", "calibration_source": "grid"},
            "footprint_yx": [[-0.0972, 290.7222], [-0.0972, 290.7222]], "centre_yx": [145.3125, 145.3125]}],
  "counts": {"cells": {"input": 0, "retained": 0, "excluded": {"not_owner": 0, "fov_not_placed": 0}},
             "molecules": {"input": 0, "retained": 0,
                           "excluded": {"cell_not_owner": 0, "not_owner": 0, "outside_grid": 0, "fov_not_placed": 0}},
             "flags": {"fov_edge": 0, "z_edge": 0, "conflict": 0, "unclaimed": 0, "outside_owner_region": 0}},
  "files": {"cells": {"path": "cells.csv", "sha256": "..."}, "molecules": {"path": "molecules.csv", "sha256": "..."},
            "counts": {"path": "counts.csv", "sha256": "..."}, "fovs": {"path": "fovs.csv", "sha256": "..."},
            "anndata": {"path": "sample.h5ad", "sha256": "..."}},
  "software": {"starfinder": "...", "git_commit": "...", "packages": {}}
}
```

The numbers are an illustration of the layout, not results.

### Result

```python
@dataclass(frozen=True)
class AssemblyResult:
    sample: str
    fovs: pd.DataFrame        # one row per FOV of the placement ("Sample tables")
    cells: pd.DataFrame       # one row per cell of every input FOV
    molecules: pd.DataFrame   # one row per molecule of every input FOV
    counts: pd.DataFrame      # the §2.9 long counts of the retained kept cells
    genes: tuple[str, ...]
    record: Mapping[str, Any]
```

`__post_init__` checks the identities of "Count accounting".

## Sample coordinates

An observation's sample coordinates are its zero-based index position on its FOV's
reference grid, pushed into the sample frame by the rule of {doc}`stitching-contract`
("Placement record", "Index to sample coordinates"), used unchanged:
`sample_zyx = origin_zyx + D · (p ⊙ spacing_zyx)`, equivalently
`placed_metadata(placement, fov_id).index_to_world(p)`.

* **Molecules**: `p` is the molecule's `z, y, x` from the §2.9 molecule table, float64,
  sub-voxel positions included; never the sampled voxel. An `outside_grid` molecule is
  mapped too (its position is finite) and keeps its coordinates while excluded.
* **Cells**: `p` is the float centroid of the §2.9 cell table, `centroid_*` for the
  original territory and `expanded_centroid_*` for the expanded one; never truncated. For
  a cell of a `plane` label image, `centroid_z` is 0.0 by the §2.9 table but is not a
  measured position, so its `sample_centroid_z` is NaN.
* **Fractional placement is preserved**: the origin is used as the placement record holds
  it (float64 µm, never cast); so are the fractional centroids and molecule positions.
* **Unknown calibration stays unknown**: a grid without known spacing cannot be placed
  (`stitch` and the import refuse it) and assemble refuses it too (check 6). No pixel size
  is assumed, the YAML `voxel_size_*` is never read by assemble, and a §2.9 `size_physical`
  that is NaN stays NaN. A grid declared from YAML values by the legacy adapter is recorded
  with `calibration_source: declared`.
* **No integer coordinates.** The integer `global_x/y/z` of the legacy files are not
  produced by assemble; they stay in the legacy files while the legacy rules run
  ("Workflow configuration").

Example: FOV origin `(1.3, 37.4, 0.7)` µm, spacing `(1.0, 1.0, 1.0)` µm (the golden
fixture's units), a cell centroid `(3.0, 8.0, 20.0)` → `(4.3, 45.4, 20.7)` µm.

## Ownership regions

Ownership is computed in assemble from the placement's origins and each placed FOV's
grid shape and spacing (D3); it is not stored in the placement, and it uses no row or
column index, no padding view and no YAML tile size. `ownership_map(placement, grids, *,
config)` returns an `OwnershipMap` whose `owner(points_zyx)` gives, for each sample-frame
point, the owning `fov_id` or none. Only placed FOVs own points. A point in no footprint
has no owner. Ownership is decided in Y and X only (option Z1 below).

### Geometry

| Option | Rule | Effect on the golden values | Effect on existing outputs |
| --- | --- | --- | --- |
| **O1. Nearest footprint centre (recommended)** | A point is owned by the placed FOV whose footprint contains it and whose footprint centre `origin + (n − 1)/2 · s` is nearest in Y and X (Euclidean, µm). For tiles of one shape on a regular grid this is the midline of every overlap, the legacy midpoint without the cast; for irregular positions (stage files) and unequal shapes it is the nearest-centre partition restricted to the footprints. Every point covered by a footprint has exactly one owner. | None for the pins (frozen helpers). The assemble entry on the fixture (`tables/option_effects.json`): boundaries near x 59.2 (59.1 to 59.3: slightly tilted, because FOV3's origin is 0.3 µm higher than FOV1's) and y 42.2, instead of 60 and 51. 3D: scene cell 4 is retained once (by FOV2), not twice, and the background molecule at y 45 once; cell 9's molecule at x 64 stays unretained (excluded `cell_not_owner`, its retained copy flagged `fov_edge`). 2D: cell 9 is retained by FOV1 and FOV3 (FOV3's projected centroid lies at x 59.5, past the boundary at 59.1 there), and both copies are flagged `conflict`. | LN (`tables/ln_ownership.json`, 4 px raster, fractional origins of the 2024 placement): the owner differs from the 2024 rectangles on 159,040 px² (0.13 % of the 119.4 million px² covered); no covered point is unowned and none is owned twice (legacy: about 33,600 px² twice, 3,000 px² unowned). |
| O2. Midpoint boxes of neighbour pairs | Today's rule written in µm: per FOV an axis-aligned box whose edge towards each neighbour is the middle of their overlap, with fractional origins, each FOV's own grid shape (`img_row` for Y) and the edges of neighbour pairs from the nominal grid. | None for the pins. On the fixture the same retained sets as O1 (box edges at x 59.2 and y 42.2). | LN: owner differs from 2024 on 76,608 px²; 33,760 px² owned twice and 2,576 px² unowned, at blank cells and corners, as today. Needs grid neighbours, which a stage file or supplied table without a grid does not give. |
| O3. The legacy rectangles | Read `start/end_*_norm` of `tile_config` | Reproduces every pinned value, including the doubly owned band. | Keeps discrepancies 1, 2, 4, 8 and 12 of {doc}`assembly-baseline`; works only with BigStitcher's padded grid. |

**Recommendation: O1.** It is the only option that gives every covered point exactly one
owner whatever the positions (stage, grid or supplied), reduces to the legacy midline
where the legacy rule is right, and needs nothing but origins and grid shapes. Its
boundaries are not axis-aligned for irregular positions, which costs nothing because
ownership is a point test ({doc}`assembly-algorithms`, "Ownership").

### Ties and boundary inclusivity

Footprints are half-open per axis, `[lower, upper)`, as the voxel supports of
{doc}`assignment-contract` ("Coordinate sampling") are. A point whose distances to two
centres are exactly equal (float64 comparison, no tolerance) is a tie.

| Option | Rule | Effect on the golden values | Effect on existing outputs |
| --- | --- | --- | --- |
| **T1. Table order (recommended)** | The tie goes to the FOV that comes first in the tile position table order of the placement. | None: no observation of the fixture lies on an O1 boundary (the legacy edge case, a centroid at exactly x = 60, is a legacy pin and stays). | None measured: no exact tie among the LN raster points (`exact_ties_points` 0). |
| T2. Half-open by axis | On an axis-aligned boundary the point belongs to the FOV whose region starts there (the legacy `range(start, end)`). | None for the pins; it reproduces the legacy edge case under O2 and O3. | Defined only for axis-aligned boundaries (O2, O3); undefined at the corners of O1. |
| T3. Both own | A tied point is owned by every tied FOV and flagged. | None for the pins. | Breaks "exactly one owner" and adds duplicates on purpose. |

**Recommendation: T1.** It is deterministic for any geometry, keeps one owner per point,
and the order it uses is part of the placement record.

### The Z axis

| Option | Rule | Effect on the golden values | Effect on existing outputs |
| --- | --- | --- | --- |
| **Z1. Y and X, one layer (recommended)** | Ownership is decided in Y and X; z is carried into the sample coordinates but never decides ownership. Every placed FOV must have the same `nominal_z` (check 8): one layer of tiles, which is what the three converters give (z 0) and what BigStitcher's macro sets (`tiles_z=1`). | None (today's filter is XY only). | None for the three samples, each one layer. A placement's z correction moves sample z, not ownership. |
| Z2. 3D footprints | Ownership by nearest centre in Z, Y and X among footprints containing the point in all three axes. | None for the pins; for one layer the same owners as Z1 except points outside a FOV's z range, which today's data does not have (such molecules are `outside_grid`). | Needed only for stacked tile layers, which §2.10 does not acquire; a z correction can then change ownership of points near a FOV's top or bottom. |
| Z3. Y and X without the layer check | Today's rule. | None. | A multi-layer acquisition would be assembled with layers owning each other's points silently. |

**Recommendation: Z1.** It keeps today's decision axis, makes the single-layer
assumption explicit and refuses the case it cannot handle.

## Retention

As agreed on 2026-09-27, with the reasons of "Statuses and reasons":

1. **Cells.** A cell of a placed FOV is retained when the owner of its sample centroid
   (the territory assign sampled, expanded when assign expanded, under
   `centroid="territory"`; the original territory under `"original"`) is its own FOV;
   otherwise it is excluded with the reason `not_owner`. A cell centroid always lies in its
   own footprint, so it always has an owner.
2. **Molecules of cells.** An `assigned` or `excluded_cell` molecule (§2.9) takes the
   assembly status of its cell: retained with a retained cell, wherever its own position
   lies, even in another FOV's ownership region; excluded with the reason
   `cell_not_owner` when its cell is excluded.
3. **Unassigned molecules** are retained when the owner of their own sample position is
   their FOV, else excluded `not_owner`.
4. **`outside_grid` molecules** (§2.9) are excluded with the reason `outside_grid`: their
   FOV did not sample them, so no FOV claims them by position.
5. **FOVs without an origin** (`missing`, `failed`): every observation is excluded with the
   reason `fov_not_placed`.
6. **A sample without segmentation** (`mode` `molecules_only`) applies rules 3 to 5 to
   every molecule: its molecules have no cell, so they are retained by their own position.
7. **§2.9 statuses are kept as they are.** A §2.9 cell with status `excluded_no_nucleus` is
   retained or excluded by ownership like any cell; it stays out of the counts and of the
   AnnData because of its §2.9 status, not because of assembly.

There is no other outcome, and no branch: a FOV whose molecules all miss its cells, a FOV
without molecules and a FOV without cells go through the same rules.

## Statuses and reasons

Every cell and every molecule of every input FOV has exactly one assembly status:

| Status | Cells: reason | Molecules: reason |
| --- | --- | --- |
| `retained` | null | null |
| `excluded` | `not_owner`: its centroid is owned by another FOV; `fov_not_placed`: its FOV has no origin | `cell_not_owner`: its cell is excluded `not_owner`; `not_owner`: unassigned and its position is owned by another FOV; `outside_grid`: §2.9 `outside_grid`; `fov_not_placed` |

Excluded observations are preserved: they stay in the sample cell and molecule tables
with their sample coordinates, owner, reason and flags, and with every §2.9 column. Only
the counts table and the AnnData leave them out. A molecule of an excluded cell keeps its
§2.9 `assignment_status` and `cell_id`; assembly never moves it to another cell.

## Flags

Flags make visible what assembly cannot decide; they never change a status.

| Flag | On | Raised when |
| --- | --- | --- |
| `fov_edge` | cell | Its territory touches the first or last index of its grid in Y or X: the FOV may see only part of it. |
| `z_edge` | cell | A `volume` or `extended` territory touches the first or last Z index. |
| `conflict` | retained cell | Its sample centroid lies inside the territory of a retained cell of another FOV, or the centroid of a retained cell of another FOV lies inside its territory; `conflict_with` lists those cell keys. The same biological cell may then be retained twice. |
| `unclaimed` | excluded cell (`not_owner`) | At its sample centroid the owner FOV has no retained cell (its territory value there is 0 or a cell the owner excluded): the cell may be missing from the sample. |
| `outside_owner_region` | retained molecule | Retained with its cell, but its own position is owned by another FOV, which may retain the same molecule as unassigned. |
| `cell_unclaimed` | excluded molecule | Its cell is flagged `unclaimed`. |

"Inside a territory" is the §2.9 sampling rule applied in the other FOV: the sample point
is mapped into that FOV's index coordinates by the inverse rule of {doc}`stitching-contract`
and sampled at `floor(c + 0.5)` (Y and X for a `plane`, Z, Y and X otherwise); a point
outside its grid is in no territory. With `conflicts=False` the `conflict` and `unclaimed`
flags are not computed and the record says so.

**What this policy does not guarantee.** Assembly retains observations; it does not
identify biological cells. Where neighbouring FOVs segment the same cell differently, or
see different parts of it, the two observations' centroids can fall on opposite sides of
an ownership boundary: then the cell is retained twice (both centroids on their own
side) or by no FOV (both on the other side), and a cell cut by a FOV border is retained
with only the part and the molecules that FOV saw. **The sample tables therefore do not
guarantee one observation per biological cell where neighbouring segmentations disagree**;
`conflict` marks the first case, `unclaimed` the second, and `fov_edge` with
`outside_owner_region` the third. The golden fixture shows each: a cell retained twice in
2D under O1, and a molecule of a retained cell that its owner did not image. No matching,
merging or reassignment repairs them (excluded by the planning decisions).

## Sample-level identities

* **Cells** keep the §2.9 key `(cell_namespace, cell_id)`. `cell_namespace` already names
  `[dataset_id, sample_id, fov_id, subtile_id, run]`, so the key is unique in the sample and
  traces back to the FOV, the segmentation run and the label value in its label file.
* **Molecules** keep `(spot_namespace, spot_id)`, unique in the sample, tracing back to the
  FOV's read and spot rows; their cell link is the §2.9 nullable `(cell_namespace, cell_id)`.
* **Readable names**: `obs_name = "<sample>_<fov_id>_<cell_id>"`, the format of today's
  `unique_index`, used as the AnnData obs names and as a column of the cell table;
  `assemble` checks it is unique. Background molecules no longer share a key.
* **FOV**: `fov_id`, as in the placement and the inputs.

## Count accounting

The identities are checked exactly by `AssemblyResult.__post_init__` and by the tests.
With `F` the input FOVs, `C_f` and `M_f` the cell and molecule tables of FOV `f`:

1. **One assembly status each.** Every row of every `C_f` appears exactly once in the
   sample cell table, keyed by `(cell_namespace, cell_id)`, with one status from
   `ASSEMBLY_STATUSES`; likewise every row of every `M_f` in the sample molecule table,
   keyed by `(spot_namespace, spot_id)`.
2. **The populations sum to the inputs.**
   `Σ_f |C_f| = n_cells_retained + Σ_reason n_cells_excluded(reason)` and
   `Σ_f |M_f| = n_molecules_retained + Σ_reason n_molecules_excluded(reason)`; per FOV
   too.
3. **Molecules follow their cell.** Every `assigned` or `excluded_cell` molecule has its
   cell's assembly status (and reason `cell_not_owner` when the cell's is `not_owner`, or
   `fov_not_placed`); every `unassigned` molecule of a placed FOV is retained exactly when
   its position's owner is its FOV; every `outside_grid` molecule is excluded.
4. **Retained counts equal the per-FOV counts.** For every retained cell whose §2.9 status
   is `kept`, its rows in the sample `counts` equal its rows in its FOV's §2.9 `counts`,
   compartment by compartment; no other cell has rows. Hence
   `Σ counts(whole) = #{retained molecules with assignment_status assigned}
   = Σ n_molecules over retained kept cells`.
5. **§2.9 identities carry over.** Each retained cell keeps its §2.9 counts, so
   `whole = nucleus + cytoplasm` holds for every retained cell whose `compartments` is
   `available`, cells whose compartments are `withheld`, `no_nucleus` or `unavailable`
   have no compartment rows, and the per-molecule compartment identity of
   {doc}`assignment-contract` ("Count accounting", 3) holds over the retained molecules.
6. **No identity about biological cells.** The number of distinct biological cells or
   molecules is not known and is not claimed ("Flags").

`record["counts"]` holds the totals of 2 per status and reason, and the flag counts.

## Sample tables

Every assembled sample has these tables, also a sample without segmentation (then the
cell and count tables have no row).

**`fovs`**, one row per FOV of the placement: `fov_id`; `placement_status`; `origin_z/y/x`
(µm, copied); `kind` (`assignment`, `molecules`, or null without input); `footprint_y0,
footprint_y1, footprint_x0, footprint_x1` and `centre_y, centre_x` (µm); `n_cells`,
`n_cells_retained`, `n_molecules`, `n_molecules_retained`.

**`cells`**, one row per cell of every input FOV, in placement table order, then
increasing `cell_id`:

| Column | Type | Content |
| --- | --- | --- |
| `fov_id` | string | The FOV. |
| `cell_namespace`, `cell_id` | string, UInt32 | The §2.9 key. |
| `obs_name` | string | `<sample>_<fov_id>_<cell_id>`. |
| every column of the §2.9 cell table | as there | `status`, `exclusion_reason`, sizes, `centroid_*`, `expanded_centroid_*`, `n_molecules`, `n_nuclei`, `correspondence`, `correspondence_flags`, `compartments`: copied unchanged. |
| `sample_centroid_z/y/x` | float64, µm | The original territory's centroid in sample coordinates; `sample_centroid_z` NaN for a plane. |
| `sample_expanded_centroid_z/y/x` | float64, µm | The same for the expanded territory; NaN unless assign expanded. |
| `owner_fov_id` | string | The owner of the centroid used for ownership; null for `fov_not_placed`. |
| `assembly_status`, `assembly_reason` | string | "Statuses and reasons". |
| `assembly_flags` | string | `;`-joined `CELL_ASSEMBLY_FLAGS`; empty when none; `conflict` and `unclaimed` null with `conflicts=False`. |
| `conflict_with` | string | `;`-joined `obs_name`s of the conflicting retained cells; empty when none. |

**`molecules`**, one row per molecule of every input FOV, in placement table order, then
input order: `fov_id`; every column of the §2.9 molecule table, copied (for a
`molecules_only` sample, the `MoleculeTable` columns and `assignment_status` null);
`sample_z`, `sample_y`, `sample_x` (float64, µm); `owner_fov_id`; `assembly_status`;
`assembly_reason`; `assembly_flags`.

**`counts`**, the long table of {doc}`assignment-contract` with `fov_id` and
`cell_namespace` added: one row per nonzero (retained kept cell, gene, compartment), with
`count` int64; `matrix()` on the result gives the dense form in `genes` order.

## Persistence

| Option | Layout | Effect on the golden values | Effect on existing outputs |
| --- | --- | --- | --- |
| **S1. A folder per sample beside the placement (recommended)** | `<OUTPUT>/assembly/<sample>/`: `assembly.json` (the record, with the SHA-256 of every file), `fovs.<fmt>`, `cells.<fmt>`, `molecules.<fmt>`, `counts.<fmt>` and `sample.h5ad`. `<fmt>` is CSV by default or Parquet, through the checkpoint table writer ({doc}`checkpoints`, "Table formats"), as for the §2.9 tables. | None: the golden test pins values, not files. | New folder beside `stitching/<sample>/` (W-325 S1). `output/tile_config_{sample}.csv`, `expr/{fovID}/*` and the legacy sample files are untouched while the legacy rules run. |
| S2. The legacy sample file names as the store | `expr/{sample}_raw.h5ad` (the AnnData) and `expr/{sample}_reads_assignment.csv` (the molecule table), with new content; the record in `uns`. | None. | The same names hold different populations and columns depending on the configuration; readers of earlier runs cannot tell which they read. The cell table of excluded cells has no file. |
| S3. One AnnData as the only store | `sample.h5ad` with every cell (retained and excluded) as `obs`, the molecule table in `uns`. | None. | Breaks "one row per retained cell" of the AnnData, needs anndata to read any table, and stores a large molecule table inside an H5AD. |

**Recommendation: S1.** It follows the run-record pattern of the placement and of the
§2.9 assignment folders, keeps every population in a plain table, and needs anndata only
for the one file that is an AnnData. `save_assembly` writes the files atomically;
`load_assembly` checks every SHA-256 against the record and raises `ValueError` on a
mismatch.

## Sample-level AnnData

Every assembled sample has `sample.h5ad`, written by `sample_anndata(result, *,
annotation=None)` from the sample tables. It is not a second exporter: the AnnData is
built by the §2.9 AnnData code, factored out of the workflow adapter's `_write_raw_h5ad`
and `_cell_obs` (`src/python/starfinder/dataset/workflow.py:1361-1430`) into one function,
`counts_anndata(cells, counts, genes, *, uns)` in `starfinder.assignment`, which the adapter
then calls for `raw.h5ad` with its legacy columns (its output unchanged) and
`sample_anndata` calls for the sample. Its content:

| Part | Content |
| --- | --- |
| `obs` | One row per cell with `assembly_status == "retained"` and §2.9 `status == "kept"`, in the cell table's order. `obs_names` are the cells' `obs_name`. Columns: `sample`, `fov_id`, `cell_namespace`, `cell_id`, the §2.9 cell columns that `raw.h5ad` carries (`size_voxels`, `expanded_size_voxels`, `size_physical`, `centroid_*`, `n_molecules`, `n_nuclei`, `correspondence`, `correspondence_flags`, `compartments`), `sample_centroid_z/y/x`, `sample_expanded_centroid_z/y/x`, `assembly_flags`, `conflict_with`, and with `annotation` the columns of that sample's annotation row (the workflow passes `documents/sample-annotation.csv` without `fov_start`, `fov_end`). |
| `var` | The sample's `genes`, in order (`var_names`). |
| `X` | float64 whole-cell counts: row `i` equals the `whole` rows of cell `i` in the sample `counts`, which equal its §2.9 counts (identity 4). |
| `layers` | When the inputs gave nuclei: `nucleus` and `cytoplasm`, float64, 0.0 for a measured zero and NaN for every gene of a cell whose `compartments` is not `available`, as in `raw.h5ad`; a FOV assigned without nuclei contributes NaN rows. Without nuclei in any FOV: no layer. |
| `obsm["spatial"]` | `(n, 2)` float64, `[sample_centroid_x, sample_centroid_y]` in µm, of the centroid used for ownership: the X, Y order of the scanpy and squidpy convention. The full ZYX values are the `obs` columns. |
| `uns` | `assembly`: the record as JSON text (the per-FOV assignment record hashes included); `spatial_unit`: `"µm"`; `frame_id`. |

**A test can check it against the sample tables**: `adata.n_obs` equals the number of cell
rows with `assembly_status == "retained"` and `status == "kept"`; `obs_names` and
`obs[["fov_id", "cell_namespace", "cell_id"]]` equal those rows, in order;
`adata.X[i, g]` equals the `whole` count of that cell and gene in `counts` (0 where there
is no row), and the layers likewise; `obs` sample coordinates and `obsm["spatial"]` equal
the cell table's `sample_*` columns exactly; `var_names` equal `genes`.

**No segmentation** (`mode` `molecules_only`): the AnnData has no row (`X` of shape
`(0, n_genes)`), `var["n_molecules_retained"]` holds the retained molecules per gene, and
`uns["assembly"]` the record. The molecule table holds the molecules themselves.

**Dependency.** `sample_anndata` imports `anndata`, from the existing `anndata` extra
(`starfinder[anndata]`: `anndata==0.12.6` on Python 3.11 and later, `0.11.4` on 3.10),
lazily; without it, it raises `ModuleNotFoundError` naming the extra. `assemble` and the
tables need only the base dependencies. No dependency or lock change.

**Where it differs from `create_sample_h5ad.py`**:

* built from the sample tables, so the retained cells follow the new ownership rule, and
  FOVs without cells are part of the record rather than skipped;
* needs no `scanpy` and runs under pandas 3 (strings written as anndata expects);
* exists for every assembled sample, with no row when no cell is retained (today `concat`
  of an empty list raises);
* sample coordinates in µm (`sample_centroid_*`, `obsm["spatial"]`) instead of integer
  pixel `global_*`, and no legacy `volume`, `fov_x/y/z`, `seg_label` columns (the cell key
  replaces `seg_label`);
* the assembly record and the per-FOV assignment hashes in `uns` (today each FOV's `uns`
  is dropped by `concat`);
* the compartment layers are kept when some FOVs lack nuclei (NaN rows), where `concat`'s
  inner join drops them;
* flags and `conflict_with` columns; the same `obs_names` format.

## Workflow configuration

A Python-only `assembly` block (`backend: python`, and a `stitching` block, whose placement
it reads), executed by a new per-sample rule `assemble_sample` through `_run_assembly` in
the workflow adapter:

```yaml
assembly:
  assignment: default          # the assignment name read per FOV; null: molecules only
  ownership: nearest_center    # nearest_center | midpoint_box
  centroid: territory          # territory | original
  tie: table_order
  conflicts: true
  table_format: csv            # csv | parquet
```

Its keys are the fields of `AssemblyConfig` and of the `Dataset.assemble` call; a
default-tier test keeps the static schema equal to them, as for the other blocks.

**Replacing the legacy filter.**

| Option | With an `assembly` block | Without one | Effect on the golden values | Effect on existing outputs |
| --- | --- | --- | --- | --- |
| **W1. Replace per configuration (recommended)** | `reads_assignment` writes the per-FOV files without the filter and without `global_*` (every kept cell and every molecule, with their §2.9 columns), and saves the FOV's assignment folder; `assemble_sample` runs `Dataset.assemble` per sample and writes `assembly/<sample>/`; `create_sample_h5ad` and `create_sample_reads_assignment` do not run; `tile_config_{sample}.csv` is still written by the legacy writer and read by nothing. | The four legacy rules run unchanged, with the adapter's filter. | None: the pins are legacy. `test_assignment_workflow.py` runs without the block and passes unchanged. | Configurations without the block are unchanged; with it, the per-FOV files change content (named in `docs/migration.md`) and the sample outputs move to `assembly/<sample>/`. |
| W2. Both side by side | The legacy chain runs as today and `assemble_sample` runs as well. | Unchanged. | None. | Two sample populations for one run (box-filtered and owned), which disagree where the rectangles do. |
| W3. Replace everywhere | The adapter drops the filter in every configuration and the legacy sample rules read assembly outputs. | The same as with the block. | None for the pins; `test_assignment_workflow.py` assertions on the legacy columns would need edits. | Every existing configuration changes its outputs at once. |

**Recommendation: W1.** It keeps legacy configurations reproducible, gives the new
populations one home, and follows the stitching block's pattern ({doc}`stitching-contract`,
"Workflow configuration").

The rule `assemble_sample` reads `stitching/{sample}/placement.json`, the saved assignment
folders (or molecule tables) of the sample's FOVs and their reference grids, and writes
the files of "Persistence". It runs once per sample, after `reads_assignment` of every FOV
of the sample and after the placement exists; unlike `reads_assignment` today, no rule
reads another sample's placement.

## Tests the implementation keeps and adds

* `test_workflow_assembly_golden.py`, `test_workflow_stitching_golden.py`,
  `test_assignment_golden.py` and `test_assignment_workflow.py` pass unchanged; their
  helpers stay as the frozen legacy reference.
* New tests: each check of the entry with its error; the coordinate rule against
  `placed_metadata(…).index_to_world` on fractional origins; ownership under O1 on the
  golden fixture's placement (boundaries near x 59.2 and y 42.2 for the fractional origins;
  every covered point one owner; T1 on a constructed exact tie); the retained sets of
  `tables/option_effects.json` on the fixture (3D: FOV1 retains scene cells 1, 2, 5, 9;
  FOV2 4, 6; FOV3 3, 7, 8; 2D: FOV3 also 9, with both copies flagged `conflict`); every
  identity of "Count accounting" on the fixture and on a `molecules_only` variant; the
  `fov_not_placed` and review refusal cases; the AnnData checks above; the save and load
  round trip with its SHA-256 check; the workflow block and its schema.

## `docs/migration.md` entries

The implementation adds the `starfinder.assembly` module and its names; the `assembly`
YAML block and the `assemble_sample` rule; `counts_anndata` in `starfinder.assignment`;
and, for configurations with an `assembly` block only, the intentional changes: the
per-FOV `raw.h5ad` and `reads_assignment.csv` are not box-filtered and have no `global_*`
columns, the sample outputs are `assembly/<sample>/` instead of
`expr/{sample}_raw.h5ad` and `expr/{sample}_reads_assignment.csv`, cells and molecules are
retained by ownership regions of the placement instead of the `tile_config` rectangles,
and every excluded observation is kept with its reason.

## Exclusions

No implementation, rename, dependency or lock change; no change to the placement contract
(W-325) or to the §2.9 label and assignment contracts; no cross-FOV cell matching or
merging, no recovery of complete cells across borders, no reassignment of molecules; no
fused image, label export or SpatialData (W-328); no real data (every measured number is
from the W-324 records and the W-326 derivations cited above).
