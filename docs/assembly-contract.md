# Assembly contract: sample coordinates, ownership, retained and excluded populations and the sample tables

Status: Accepted (W-329, 2026-10-07, at 9033d46)

This page specifies the §2.10 assembly contract: a sample-level assemble entry separate
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
resources are in {doc}`assembly-algorithms`. It is implemented in `starfinder.assembly`, and
nothing is renamed. The fused image and the SpatialData export belong to W-328.

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

* **Two sample-level entries** outside `PipelineConfig` and `FOV.run`: `Dataset.stitch`
  and `Dataset.export` (D3, as amended at the W-329 review of 2026-10-06);
  `Dataset.export` calls `assemble`, `fuse_images`, `fuse_labels` and `export_spatialdata` in
  order ({doc}`sample-output-contract`, "The export entry"). **Ownership regions are
  computed in assemble from the placement**, not stored in it.
* **Placement from either backend** enters as one `PlacementResult` (D1, D2).
* **Retention** (2026-09-27): a cell is retained by its centroid; all molecules of a
  retained cell are kept wherever they lie; an unassigned molecule is retained by its own
  position. This holds for the route per FOV; on the route after fusion
  ({doc}`sample-segmentation-contract`) every cell of the sample segmentation is retained
  and molecules are retained by their position. Exclusions are preserved, not dropped. Cells at a FOV border are flagged.
  Cells are not merged across FOVs, complete cells are not recovered across borders and
  molecules are not reassigned; two retained observations of one cell in two FOVs are
  matched only by the duplicate rule of 2026-10-09 (below). The workflow is FOV-first:
  per-FOV results are produced first, and assembly follows per sample.
* **The sample-level AnnData is a required default output of every assembled sample**
  (Jiahao, 2026-10-05, amendment), written from the sample tables through the §2.9
  AnnData code, without a second exporter. The fused image and SpatialData are W-328's.
* **Amended at the W-329 review (2026-10-06):** placeholder FOVs own no points, and only a
  placed FOV that has an input owns points ("Placeholder FOVs"); the shared AnnData
  function is named `write_h5ad` and builds the AnnData and writes the file ("Sample-level
  AnnData"); the record holds each territory label file's path beside its SHA-256 ("What
  assemble records"); and `ownership_map` takes an optional `candidates` argument
  ("Ownership regions"). A placement from `placement_from_table`
  ({doc}`stitching-contract`, "Supplied placement") is a `PlacementResult` like any other.
* **Decided on 2026-10-09** (Jiahao, on {doc}`cells-across-fovs-proposal`, question 3,
  option b; implemented by W-400): the route per FOV keeps one of two observations of a
  cell, as a part of that route for good. Two retained cells of different FOVs whose
  territories share at least `duplicate_threshold` (default 0.5) of the smaller territory
  on the sample grid are one candidate cell: the larger observation stays retained, and
  the other is excluded `duplicate_observation` with its molecules ("Retention", rule 8).
  The rule removes a second observation; it does not complete a cell cut by a FOV edge.

Excluded: merging of cells across FOVs (the duplicate rule of "Retention", rule 8, excludes
one of two observations and merges nothing), recovery of complete cells, reassignment
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
* An **owning FOV** is a placed FOV that has an input and whose placement status is
  `image_supported` or `metadata_only`. A `placeholder` is placed but never an owning FOV,
  with or without an input ("Placeholder FOVs").
* The **owner** of a sample-frame point is the owning FOV that ownership assigns it to
  ("Ownership regions"); a FOV's **ownership region** is the set of points it owns.
* An observation is **retained** when it represents the sample, and **excluded**
  otherwise; the **assembly status** records which, with a reason. It is distinct from the
  §2.9 assignment status (`assigned`, …) and cell status (`kept`, …), which assemble copies
  and never changes.

## Names

| Option | Names | Effect on the golden values | Effect on existing files and outputs |
| --- | --- | --- | --- |
| **A1. A `starfinder.assembly` module (recommended)** | Module `starfinder.assembly`; entry `assemble`; result `AssemblyResult`; input `AssemblyInput`; config `AssemblyConfig`; ownership `ownership_map`, `OwnershipMap`; vocabularies `ASSEMBLY_STATUSES`, `CELL_ASSEMBLY_REASONS`, `MOLECULE_ASSEMBLY_REASONS`, `CELL_ASSEMBLY_FLAGS`, `MOLECULE_ASSEMBLY_FLAGS`; persistence `save_assembly`, `load_assembly`; AnnData `write_sample_h5ad`; coordination through the assemble part of `Dataset.export` ({doc}`sample-output-contract`) | None: `test_workflow_assembly_golden.py` pins the legacy scripts and the adapter through frozen helpers, which no option edits. | New files only; `Dataset.export` belongs to {doc}`sample-output-contract`; the exports go into `__all__` and the autosummary list of `docs/api/assembly.rst`, which `docs/check_reference.py` requires. `tile_config_{sample}.csv` and the legacy per-FOV and sample files stay as they are while the legacy rules run. |
| A2. Inside `starfinder.stitching` | `starfinder.stitching.assemble` returning `SampleAssembly`; the same `Dataset.export` | None. | The same files; one module then holds placement and the observation populations, which D3 separates, and the stitching module gains a dependency on `starfinder.assignment`. |
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
| Ownership | `ownership_map(placement, grids, *, config, candidates=None)` returning `OwnershipMap` with `owner(points_zyx)` | function, frozen dataclass |
| Vocabularies | `ASSEMBLY_STATUSES = ("retained", "excluded")`; `CELL_ASSEMBLY_REASONS = ("not_owner", "fov_not_placed", "duplicate_observation")`; `MOLECULE_ASSEMBLY_REASONS = ("cell_not_owner", "not_owner", "outside_grid", "fov_not_placed", "cell_duplicate_observation")`; `CELL_ASSEMBLY_FLAGS = ("fov_edge", "z_edge", "conflict", "unclaimed")`; `MOLECULE_ASSEMBLY_FLAGS = ("outside_owner_region", "cell_unclaimed")` | tuples |
| AnnData | `write_sample_h5ad(result, path, *, annotation=None)`, through `write_h5ad` of `starfinder.assignment` ("Sample-level AnnData") | functions |
| Persistence | `save_assembly(result, folder, *, table_format="csv")`, `load_assembly(folder)` | functions |
| Coordination | The assemble part of `Dataset.export(sample, *, placement, assignment="default", config=ExportConfig(), output_dir)` ({doc}`sample-output-contract`, "The export entry") | method |

## The assemble entry

### Function

```python
def assemble(placement: PlacementResult, inputs: Sequence[AssemblyInput], *,
             config: AssemblyConfig = AssemblyConfig(),
             segmentation: SampleSegmentation | None = None) -> AssemblyResult: ...

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
    duplicate_threshold: float | None = 0.5   # the duplicate rule ("Retention", rule 8); None: off
```

`assemble` checks its arguments, maps every observation into sample coordinates, computes
the ownership map from the placement, assigns one assembly status per cell and molecule,
computes the flags, copies the §2.9 counts of the counted cells ("Statuses and reasons"),
and returns an
`AssemblyResult`. It never reads or writes files except to load a saved assignment named
by path, never runs stitch, assign or any per-FOV stage (with a `segmentation`, the route
after fusion of {doc}`sample-segmentation-contract`, it calls `assign_molecules` once on
the sample), merges no cell across FOVs,
matches observations of two FOVs only by the duplicate rule ("Retention", rule 8), and
moves no molecule to another cell. Like `stitch`, it is a sample-level computation, not a
multi-FOV runner ({doc}`stitching-contract`, "Not the multi-FOV runner").

**Inputs.**

* **Placement**: a `PlacementResult` from `stitch` (`multiview_stitcher`) or from
  `import_bigstitcher`; the two are one type, so assemble does not know the backend except
  from the method record, which it copies.
* **Per FOV, one of two kinds**, the same kind for every FOV of the sample (with a
  `segmentation`, the inputs are molecule tables):
  * `assignment`: the §2.9 `AssignmentResult` of that FOV, in memory, or the path of its
    saved folder (`fovs/<fov_id>/assignment/<name>/`), loaded with the §2.9
    loader, which checks every recorded SHA-256. Assemble reads its molecule table, cell
    table, counts, record and, for the flags, its territories;
  * `molecules`: a `MoleculeTable` alone, for a sample without segmentation. Every
    molecule is then retained or excluded by its own position, and the sample has no
    cells ("Retention").
* `grid`: the FOV's reference grid, which places its index coordinates.

`assemble` is the first part of the one sample-level export entry
`Dataset.export(sample, *, placement, assignment="default", config=ExportConfig(),
output_dir)` ({doc}`sample-output-contract`, "The export entry"), which replaces a
separate `Dataset.assemble` (W-329 review, 2026-10-06, amending D3). Its assemble part
loads the saved placement (`load_placement`), builds one `AssemblyInput` per FOV of the
placement from the saved assignment folders named `assignment` (or, with
`assignment=None`, from the saved molecule tables through `molecule_table`) and the
reference grids (the stored metadata of each FOV's reference image, the file the dataset's
`reference_image` names, default `fovs/{fov_id}/reference.tif`, read as
`reference_grid_from_file` reads it, or, for a file that stores none, the grid the
dataset's `voxel_size_xy` and `voxel_size_z` declare), calls `assemble` with `config.assembly`, and
saves the result in `output_dir`, the sample's export folder ("Persistence"). It keeps the
tables of every FOV and at most two label images in memory at a time
({doc}`assembly-algorithms`, "Resources"). With `assemble: false` the entry loads the
saved result instead (`load_assembly`). `assemble` stays callable alone.

### Checks the entry applies

In this order, before any coordinate is computed; each raises and names the FOVs:

1. **Config.** `AssemblyConfig` validates its five fields in `__post_init__`
   (`duplicate_threshold` is null or a number above 0 and at most 1; `ValueError` naming
   the field).
2. **Review.** A placement whose `record["review"]` is `required` and that holds no
   approval raises `PlacementReviewRequiredError` with the reasons
   ({doc}`stitching-contract`, "Failure policy", item 2).
3. **FOV identity.** Every `fov_id` is unique among the inputs and is a row of
   `placement.fovs` (`ValueError`). Every FOV whose placement status is
   `image_supported` or `metadata_only` must have an input (`ValueError`); an input of a
   `missing` or `failed` FOV is accepted and its observations are excluded with the reason
   `fov_not_placed`. An input of a `placeholder` FOV is accepted too; the placeholder owns
   no point ("Placeholder FOVs"), so its observations are excluded `not_owner` (rules 1
   and 3 of "Retention").
4. **Kind.** Each input gives exactly one of `assignment`, `molecules`, and all inputs give
   the same kind (`ValueError`). The kind is recorded as `mode`: `cells` or
   `molecules_only`.
5. **Observation identity.** The first four entries of each input's `spot_namespace` (and
   of its `cell_namespace`) are `[dataset_id, sample_id, fov_id, subtile_id]` with this
   `fov_id`, one `dataset_id` and `sample_id` for the sample, and `subtile_id` null:
   subtiles are not assembled (`ValueError`).
6. **Grid.** The input's `grid` equals, by the grid rule of {doc}`assignment-contract`,
   the grid the placement recorded for the FOV (`record["inputs"][fov_id]`) and the grid
   of the assignment record, reading `µm`, `um` and `micrometer` as one unit and a grid
   without a direction as having the identity direction; otherwise
   `IncompatibleGeometryError` naming both. Its `ImageMetadata` must have known spacing,
   unit `µm` (spelled `µm`, `um` or `micrometer`) and identity direction (or none), as
   `stitch` requires; assemble never assumes a pixel size.
7. **Genes.** Every input has the same `genes` tuple (`ValueError` naming the
   differences), which becomes the sample's gene list.
8. **One layer.** Every placed FOV has the same `nominal_z` (option Z1, "Ownership
   regions"); otherwise `ValueError("assemble handles one layer of tiles; FOVs … have
   nominal z …")`.

### What assemble records

The record of the result (the `assembly` section of `export.json` when saved, "Persistence") holds:

* the placement: its `sample`, frame, gauge, method record (backend, config), review
  state and approval, and the SHA-256 of its `fovs` and `links` tables as
  `placement.json` records them;
* per FOV: `fov_id`, placement status, origin, input kind, the assignment name and the
  SHA-256 of its `assignment.json` and of its molecule, cell and count tables, the
  molecule table's `sha256` (`MoleculeTable`), the territory label file read for the flags
  as `territories`: its path (absolute, as the §2.9 loader resolved it) beside its SHA-256,
  both null when no file was read (`conflicts=False`, a `molecules_only` input, or an
  assignment given in memory, whose territories' SHA-256 its §2.9 record already holds),
  the grid (shape, metadata, SHA-256) and its `calibration_source`
  (`grid` for a grid read from an image's metadata, `declared` for a grid declared from
  `voxel_size_*`, by the legacy adapter or for a reference image that stores no metadata,
  never upgraded);
* the config, the ownership geometry with each FOV's footprint, centre and whether it is an
  owning FOV (`owns_points`), the counts of
  "Count accounting", and the SHA-256 of every table and file it writes;
* `format_version` 1 and `software` as in `run.json` ({doc}`checkpoints`);
* `route` (`per_fov`, or `after_fusion` with a `segmentation`) and `route_source` (`set` or
  `default` as `Dataset.export` resolved the route; null when `assemble` is called alone),
  and on the route after fusion `segmentation_files` (per run the label file's name and
  SHA-256 that it read, and `territories.ome.tif` when it wrote one) and the sample
  assignment's §2.9 record as `assignment` ({doc}`sample-segmentation-contract`, "The
  record").

```json
{
  "format_version": 1, "stage": "assembly", "sample": "sample1", "mode": "cells",
  "frame": {"frame_id": "sample:sample1", "unit": "µm", "direction": "identity"},
  "placement": {"method": "bigstitcher", "gauge": "G1", "review": "not_needed", "approval": null,
                "fovs_sha256": "...", "links_sha256": "..."},
  "config": {"ownership": "nearest_center", "centroid": "territory", "tie": "table_order", "conflicts": true,
             "duplicate_threshold": 0.5},
  "fovs": [{"fov_id": "Position001", "placement_status": "image_supported", "origin_zyx": [0.0, 0.0, 0.0],
            "kind": "assignment", "assignment": {"name": "default", "record_sha256": "...", "tables": {}},
            "molecules_sha256": "...",
            "territories": {"path": "/.../Position001/assignment/default/territories.tif", "sha256": "..."},
            "grid": {"shape_zyx": [50, 1496, 1496], "sha256": "...", "calibration_source": "grid"},
            "footprint_yx": [[-0.0972, 290.7222], [-0.0972, 290.7222]], "centre_yx": [145.3125, 145.3125],
            "owns_points": true}],
  "counts": {"cells": {"input": 0, "retained": 0,
                       "excluded": {"not_owner": 0, "fov_not_placed": 0, "duplicate_observation": 0}},
             "molecules": {"input": 0, "retained": 0,
                           "excluded": {"cell_not_owner": 0, "not_owner": 0, "outside_grid": 0, "fov_not_placed": 0,
                                        "cell_duplicate_observation": 0}},
             "flags": {"fov_edge": 0, "z_edge": 0, "conflict": 0, "unclaimed": 0, "outside_owner_region": 0}},
  "files": {"cells": {"path": "tables/cells.csv", "sha256": "..."}, "molecules": {"path": "tables/molecules.csv", "sha256": "..."},
            "counts": {"path": "tables/counts.csv", "sha256": "..."}, "fovs": {"path": "tables/fovs.csv", "sha256": "..."},
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
    counts: pd.DataFrame      # the §2.9 long counts of the counted cells
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
  sub-voxel positions included; never the sampled voxel. On a FOV whose reference grid has
  one plane a molecule is sampled at its `y, x` only ({doc}`assignment-contract`,
  "Coordinate sampling"), so its `p_z` is 0, the plane: its `sample_z` is the plane's,
  and its recorded `z` stays in the column `z`. An `outside_grid` molecule is
  mapped too (its position is finite) and keeps its coordinates while excluded.
* **Cells**: `p` is the float centroid of the §2.9 cell table, `centroid_*` for the
  original territory and `expanded_centroid_*` for the expanded one; never truncated. For
  a cell of a `plane` label image, `centroid_z` is 0.0 by the §2.9 table but is not a
  measured position, so its `sample_centroid_z` is NaN.
* **The ownership centroid** is the centroid that ownership tests ("Retention", rule 1):
  the expanded territory's when `centroid="territory"` and assign expanded the FOV's
  cells, the original territory's otherwise (`centroid="original"`, or no expansion). The
  cell table records which in `ownership_centroid` (`expanded` or `original`) and copies
  that centroid's sample coordinates into `sample_ownership_centroid_z/y/x`. Each name
  therefore means one thing: `sample_centroid_*` is always the original territory's
  centroid, `sample_expanded_centroid_*` always the expanded territory's (NaN without
  expansion), and `sample_ownership_centroid_*` always equals the pair that
  `ownership_centroid` names.
* **Fractional placement is preserved**: the origin is used as the placement record holds
  it (float64 µm, never cast); so are the fractional centroids and molecule positions.
* **Unknown calibration stays unknown**: a grid without known spacing cannot be placed
  (`stitch` and the import refuse it) and assemble refuses it too (check 6). No pixel size
  is assumed, the YAML `voxel_size_*` is never read by assemble, and a §2.9 `size_physical`
  that is NaN stays NaN. A grid declared from YAML values, by the legacy adapter or by the
  export entry for a reference image that stores no metadata, is recorded with
  `calibration_source: declared`.
* **No integer coordinates.** The integer `global_x/y/z` of the legacy files are not
  produced by assemble; they stay in the legacy files while the legacy rules run
  ("Workflow configuration").

Example: FOV origin `(1.3, 37.4, 0.7)` µm, spacing `(1.0, 1.0, 1.0)` µm (the golden
fixture's units), a cell centroid `(3.0, 8.0, 20.0)` → `(4.3, 45.4, 20.7)` µm.

## Ownership regions

Ownership is computed in assemble from the placement's origins and each placed FOV's
grid shape and spacing (D3); it is not stored in the placement, and it uses no row or
column index, no padding view and no YAML tile size. `ownership_map(placement, grids, *,
config, candidates=None)` returns an `OwnershipMap` whose `owner(points_zyx)` gives, for
each sample-frame point, the owning `fov_id` or none. Only owning FOVs own points: a
placed FOV that has an input and is `image_supported` or `metadata_only` ("Terms"). In
`ownership_map` these are the FOVs of those two statuses that have a grid in `grids`;
assemble passes the grids of its inputs, and check 3 gives every FOV of those statuses an
input. A point in no owning FOV's footprint has no owner. Ownership is decided in Y and X
only (option Z1 below).

**Candidates.** `candidates`, an optional collection of `fov_id`s, restricts the FOVs that
may own a point: with it, a point goes to the nearest-centre owning FOV among the
candidates whose footprint contains it, or to none, by the same geometry and tie rule.
The default `None` means every owning FOV, which is what assemble uses. A caller that needs
the owner among a subset, for example the FOVs that have a source image, passes that
subset. An empty collection, or a `fov_id` that is not an owning FOV of the placement (not
in it, not placed, a `placeholder`, or without a grid), raises `ValueError` naming it.

### Placeholder FOVs

A `placeholder` ({doc}`stitching-contract`, "Placement record") is placed, so it has a
footprint and a centre, which the `fovs` table and the record keep, but it owns no point.
A point in a placeholder's footprint goes to the nearest owning FOV whose footprint covers
it, by the geometry and tie rule below, or to none when no owning FOV covers it. Its
effects:

* **On the ownership options.** O1 decides among owning FOVs only, so the placeholder's
  overlap with a neighbour belongs to the neighbour, and the part of its footprint that no
  owning FOV covers has no owner. Under O2 a placeholder neighbour sets no box edge: the
  edge of an owning FOV towards a placeholder stays at its own footprint edge. T1 and Z1
  are unchanged; check 8 still reads the `nominal_z` of every placed FOV.
* **On the `unclaimed` flag.** A neighbour's cell whose ownership centroid lies in the
  placeholder's footprint is owned by the neighbour that covers it, and is therefore no
  longer excluded `not_owner` and flagged `unclaimed` because of a FOV that has no
  observations. A cell of a placeholder with an input is excluded `not_owner`; it is
  flagged `unclaimed` when no FOV owns its ownership centroid or when the owner has no
  retained cell there ("Flags").
* **On the count identities.** They are unchanged: every observation still has exactly one
  assembly status, and the populations still sum to the inputs. Identity 3 reads "its
  position's owner is its FOV", which is never true for a placeholder, so all of a
  placeholder's observations are counted under the excluded reasons.
* **On the golden values.** None. `test_workflow_assembly_golden.py` still pins the legacy
  behavior, in which the rectangles of `tile_config` include every grid cell, blanks
  included; neither its fixture nor the LN records have a placeholder (LN's eight blank
  cells are padding views, which have no row in the placement).

This follows the W-329 review (2026-10-06): no observation is lost to a FOV that has
none.

### Geometry

| Option | Rule | Effect on the golden values | Effect on existing outputs |
| --- | --- | --- | --- |
| **O1. Nearest footprint centre (recommended)** | A point is owned by the owning FOV whose footprint contains it and whose footprint centre `origin + (n − 1)/2 · s` is nearest in Y and X (Euclidean, µm). For tiles of one shape on a regular grid this is the midline of every overlap, the legacy midpoint without the cast; for irregular positions (stage files) and unequal shapes it is the nearest-centre partition restricted to the footprints. Every point covered by an owning FOV's footprint has exactly one owner. | None for the pins (frozen helpers). The assemble entry on the fixture (`tables/option_effects.json`): boundaries near x 59.2 (59.1 to 59.3: slightly tilted, because FOV3's origin is 0.3 µm higher than FOV1's) and y 42.2, instead of 60 and 51. 3D: scene cell 4 is retained once (by FOV2), not twice, and the background molecule at y 45 once; cell 9's molecule at x 64 stays unretained (excluded `cell_not_owner`, its retained copy flagged `fov_edge`). 2D: cell 9 is retained by FOV1 and FOV3 (FOV3's projected centroid lies at x 59.5, past the boundary at 59.1 there), and both copies are flagged `conflict` (with the duplicate rule off; the default rule keeps FOV1's copy, "Retention", rule 8). | LN (`tables/ln_ownership.json`, 4 px raster, fractional origins of the 2024 placement): the owner differs from the 2024 rectangles on 159,040 px² (0.13 % of the 119.4 million px² covered); no covered point is unowned and none is owned twice (legacy: about 33,600 px² twice, 3,000 px² unowned). |
| O2. Midpoint boxes of neighbour pairs | Today's rule written in µm: per FOV an axis-aligned box whose edge towards each neighbour is the middle of their overlap, with fractional origins, each FOV's own grid shape (`img_row` for Y) and the edges of neighbour pairs from the nominal grid. | None for the pins. On the fixture the same retained sets as O1 (box edges at x 59.2 and y 42.2). | LN: owner differs from 2024 on 76,608 px²; 33,760 px² owned twice and 2,576 px² unowned, at blank cells and corners, as today. Needs grid neighbours, which a stage file or supplied table without a grid does not give. |
| O3. The legacy rectangles | Read `start/end_*_norm` of `tile_config` | Reproduces every pinned value, including the doubly owned band. | Keeps discrepancies 1, 2, 4, 8 and 12 of {doc}`assembly-baseline`; works only with BigStitcher's padded grid. |

**Recommendation: O1.** It is the only option that gives every point covered by an owning
FOV exactly one owner whatever the positions (stage, grid or supplied), reduces to the
legacy midline where the legacy rule is right, and needs nothing but origins and grid
shapes. Its
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

1. **Cells.** A cell of a placed FOV is retained when the owner of its ownership centroid
   (`sample_ownership_centroid_*`, "Sample coordinates": the expanded territory's under
   `centroid="territory"` when assign expanded, the original territory's otherwise) is its
   own FOV; otherwise it is excluded with the reason `not_owner`. Either centroid lies in
   its own FOV's footprint, so the cell of an owning FOV always has an owner. A cell of a
   `placeholder` with an input is excluded `not_owner`, because a placeholder owns no
   point; its `owner_fov_id` is the neighbour that owns the centroid, or null when none
   does.
2. **Molecules of cells.** An `assigned` or `excluded_cell` molecule (§2.9) takes the
   assembly status of its cell: retained with a retained cell, wherever its own position
   lies, even in another FOV's ownership region; excluded with the reason
   `cell_not_owner` when its cell is excluded `not_owner`, and `cell_duplicate_observation`
   when its cell is excluded `duplicate_observation` (rule 8).
3. **Unassigned molecules** are retained when the owner of their own sample position is
   their FOV, else excluded `not_owner` (also when the position has no owner).
4. **`outside_grid` molecules** (§2.9) are excluded with the reason `outside_grid`: their
   FOV did not sample them, so no FOV claims them by position.
5. **FOVs without an origin** (`missing`, `failed`): every observation is excluded with the
   reason `fov_not_placed`.
6. **A sample without segmentation** (`mode` `molecules_only`) applies rules 3 to 5 to
   every molecule: its molecules have no cell, so they are retained by their own position.
7. **§2.9 statuses are kept as they are.** A §2.9 cell with status `excluded_no_nucleus` is
   retained or excluded by ownership like any cell. Retained, it is not a counted cell
   ("Statuses and reasons"): it has no row in the counts table or in the AnnData, because
   of its §2.9 status, not because of assembly; it stays in the sample cell table with its
   assembly status, and its molecules (§2.9 `excluded_cell`) take that status by rule 2.
8. **Two observations of one cell** (the duplicate rule, decided on 2026-10-09). With
   `duplicate_threshold` set (default 0.5), two cells of different FOVs that rule 1
   retained are two observations of one candidate cell when their territories share at
   least that share of the smaller territory: the voxels both territories cover on the
   sample grid, divided by the voxels of the smaller, each territory (the one assign
   sampled) placed as the label export places it ({doc}`assembly-algorithms`, "Duplicate
   rule"). The retained cells are taken in the order of decreasing territory size, ties by
   placement table order and then `cell_id`; a cell stays retained unless it is paired at
   or above the threshold with a cell of another FOV that already stays, and then it is
   excluded with the reason `duplicate_observation` and names in `duplicate_of` the first
   such cell in that order. The result does not depend on the order of the inputs. Its
   molecules are excluded with it (rule 2) and are not given to the kept cell. The rule
   removes a second observation; it does not complete a cell cut by a FOV edge, whose kept
   observation is still the part its FOV saw (`fov_edge`). `duplicate_threshold=None` turns
   the rule off and gives the result of rules 1 to 7 alone.

There is no other outcome, and no branch: a FOV whose molecules all miss its cells, a FOV
without molecules and a FOV without cells go through the same rules.

## Statuses and reasons

Every cell and every molecule of every input FOV has exactly one assembly status:

| Status | Cells: reason | Molecules: reason |
| --- | --- | --- |
| `retained` | null | null |
| `excluded` | `not_owner`: its ownership centroid is not owned by its own FOV (another FOV owns it, or, for a cell of a `placeholder`, possibly none); `fov_not_placed`: its FOV has no origin; `duplicate_observation`: a second observation of a cell that another FOV's retained cell keeps ("Retention", rule 8) | `cell_not_owner`: its cell is excluded `not_owner`; `not_owner`: unassigned and its position is not owned by its own FOV; `outside_grid`: §2.9 `outside_grid`; `fov_not_placed`; `cell_duplicate_observation`: its cell is excluded `duplicate_observation` |

**Counted cells.** The sample's **counted cells** are the cells with
`assembly_status == "retained"` and §2.9 `status == "kept"`. They are the only cells with
rows in the sample counts table and in the sample-level AnnData. A retained cell whose
§2.9 status is `excluded_no_nucleus` is not a counted cell: §2.9 gives it no counts, so it
has no row in either; it stays in the sample cell table with its assembly status, and its
molecules keep the assembly status of the cell.

Excluded observations are preserved: they stay in the sample cell and molecule tables
with their sample coordinates, owner, reason and flags, and with every §2.9 column. The
counts table and the AnnData hold only counted cells, so they leave out every excluded
cell, and also the retained cells that §2.9 excluded. A molecule of an excluded cell keeps
its §2.9 `assignment_status` and `cell_id`; assembly never moves it to another cell.

## Flags

Flags make visible what assembly cannot decide; they never change a status.

| Flag | On | Raised when |
| --- | --- | --- |
| `fov_edge` | cell | Its territory touches the first or last index of its grid in Y or X: the FOV may see only part of it. |
| `z_edge` | cell | A `volume` or `extended` territory touches the first or last Z index. |
| `conflict` | retained cell | Its ownership centroid (`sample_ownership_centroid_*`) lies inside the territory of a retained cell of another FOV, or the ownership centroid of a retained cell of another FOV lies inside its territory; `conflict_with` lists those cell keys. The same biological cell may then be retained twice. |
| `unclaimed` | excluded cell (`not_owner`) | No FOV owns its ownership centroid (only possible for a cell of a `placeholder`), or the owner FOV has no retained cell there (its territory value there is 0 or a cell the owner excluded): the cell may be missing from the sample. |
| `outside_owner_region` | retained molecule | Retained with its cell, but its own position is owned by another FOV, which may retain the same molecule as unassigned. |
| `cell_unclaimed` | excluded molecule | Its cell is flagged `unclaimed`. |

"Inside a territory" is the §2.9 sampling rule applied in the other FOV: the sample point
is mapped into that FOV's index coordinates by the inverse rule of {doc}`stitching-contract`
and sampled at `floor(c + 0.5)` (Y and X for a `plane`, Z, Y and X otherwise); a point
outside its grid is in no territory. With `conflicts=False` the `conflict` and `unclaimed`
flags are not computed and the record says so. "Retained" in the rules of `conflict` and
`unclaimed` means retained after the duplicate rule ("Retention", rule 8): a cell it
excludes is not a retained cell for either flag, so the kept observation of a pair no
longer carries `conflict` for it. `fov_edge` and `z_edge` are not changed by the rule.

**What this policy does not guarantee.** Assembly retains observations; it does not
identify biological cells. Where neighbouring FOVs segment the same cell differently, or
see different parts of it, the two observations' centroids can fall on opposite sides of
an ownership boundary: then the cell is retained twice (both centroids on their own
side) or by no FOV (both on the other side), and a cell cut by a FOV border is retained
with only the part and the molecules that FOV saw. **The sample tables therefore do not
guarantee one observation per biological cell where neighbouring segmentations disagree**;
`conflict` marks some of the cells retained twice and `unclaimed` some of the cells
retained by no FOV, because both are centroid tests ("Flags"), and `fov_edge` with
`outside_owner_region` marks the third case. The golden fixture shows each: a cell retained twice in
2D under O1 (with the duplicate rule off), and a molecule of a retained cell that its owner did not image.
The duplicate rule ("Retention", rule 8) keeps one observation of a cell retained twice
whose two territories share at least `duplicate_threshold` of the smaller one (the 2D
golden cell's share 0.90); it removes the second observation and does not complete a cell
cut by a FOV border. No merging or reassignment repairs the other cases (excluded by the
planning decisions).

## Sample-level identities

* **Cells** keep the §2.9 key `(cell_namespace, cell_id)`. `cell_namespace` already names
  `[dataset_id, sample_id, fov_id, subtile_id, run]`, so the key is unique in the sample and
  traces back to the FOV, the segmentation run and the label value in its label file. On
  the route after fusion it names `[dataset_id, sample_id, null, null, run]`, the run of
  the sample segmentation, with a null FOV.
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
   keyed by `(spot_namespace, spot_id)`. On the route after fusion the cells are compared
   with the sample segmentation's labels instead: one row per label of the cell run.
2. **The populations sum to the inputs.**
   `Σ_f |C_f| = n_cells_retained + Σ_reason n_cells_excluded(reason)` and
   `Σ_f |M_f| = n_molecules_retained + Σ_reason n_molecules_excluded(reason)`; per FOV
   too.
3. **Molecules follow their cell.** Every `assigned` or `excluded_cell` molecule has the
   assembly status of its cell, the row of the sample cell table keyed by the molecule's
   own `(cell_namespace, cell_id)` (and reason `cell_not_owner` when the cell's is `not_owner`,
   `cell_duplicate_observation` when it is `duplicate_observation`, or `fov_not_placed`); every `unassigned` molecule of a placed FOV is retained exactly when
   its position's owner is its FOV, which is never the case for a `placeholder` ("Placeholder
   FOVs"); every `outside_grid` molecule is excluded. On the route after fusion molecules
   follow their own position only, by the rule of the `molecules_only` assembly (a
   retained molecule outside the segmentation grid in Y or X is excluded `outside_grid`),
   and only a retained molecule has a sample assignment.
4. **Counted cells' counts equal the per-FOV counts.** For every counted cell
   ("Statuses and reasons"), keyed by `(cell_namespace, cell_id)`, its rows in the sample
   `counts` equal its rows in its FOV's §2.9 `counts`, compartment by compartment; no
   other cell has rows, so a retained cell that §2.9 excluded has none. Hence
   `Σ counts(whole) = #{retained molecules with assignment_status assigned}
   = Σ n_molecules over counted cells`, also cell by cell, each molecule counted in the
   cell of its own `(cell_namespace, cell_id)`, and the AnnData has one row per counted cell
   ("Sample-level AnnData").
5. **§2.9 identities carry over.** Each counted cell keeps its §2.9 counts, so
   `whole = nucleus + cytoplasm` holds for every counted cell whose `compartments` is
   `available`, counted cells whose compartments are `withheld`, `no_nucleus` or
   `unavailable` have no compartment rows, and the per-molecule compartment identity of
   {doc}`assignment-contract` ("Count accounting", 3) holds over the retained `assigned`
   molecules.
6. **No identity about biological cells.** The number of distinct biological cells or
   molecules is not known and is not claimed ("Flags").

`record["counts"]` holds the totals of 2 per status and reason, and the flag counts; the
reasons `duplicate_observation` and `cell_duplicate_observation` are counted when the
duplicate rule is on (`duplicate_threshold` not null), so that a record made with the rule
off has the counts of a record made before the rule. Identity 2 also holds that a cell
excluded `duplicate_observation`, and only such a cell, names in `duplicate_of` a retained
cell of another FOV.

## Sample tables

Every assembled sample has these tables, also a sample without segmentation (then the
cell and count tables have no row).

**`fovs`**, one row per FOV of the placement: `fov_id`; `placement_status`; `origin_z/y/x`
(µm, copied); `kind` (`assignment`, `molecules`, or null without input); `footprint_y0,
footprint_y1, footprint_x0, footprint_x1` and `centre_y, centre_x` (µm); `owns_points`
(boolean: whether it is an owning FOV, false for a `placeholder`); `n_cells`,
`n_cells_retained`, `n_molecules`, `n_molecules_retained`.

**`cells`**, one row per cell of every input FOV, in placement table order, then
increasing `cell_id` (on the route after fusion, one row per cell of the sample
segmentation, in increasing `cell_id`, whose `fov_id` is the owning FOV of its ownership
centroid, or where no owning footprint holds that point the owning FOV with the nearest
footprint centre: {doc}`sample-segmentation-contract`, "The cells"):

| Column | Type | Content |
| --- | --- | --- |
| `fov_id` | string | The FOV. |
| `cell_namespace`, `cell_id` | string, UInt32 | The §2.9 key. |
| `obs_name` | string | `<sample>_<fov_id>_<cell_id>`. |
| every column of the §2.9 cell table | as there | `status`, `exclusion_reason`, sizes, `centroid_*`, `expanded_centroid_*`, `n_molecules`, `n_nuclei`, `correspondence`, `correspondence_flags`, `compartments`: copied unchanged. |
| `sample_centroid_z/y/x` | float64, µm | The original territory's centroid in sample coordinates; `sample_centroid_z` NaN for a plane. |
| `sample_expanded_centroid_z/y/x` | float64, µm | The same for the expanded territory; NaN unless assign expanded. |
| `ownership_centroid` | string | Which centroid ownership tests: `expanded` or `original` ("Sample coordinates"). |
| `sample_ownership_centroid_z/y/x` | float64, µm | A copy of `sample_expanded_centroid_*` when `ownership_centroid` is `expanded`, of `sample_centroid_*` when it is `original`. |
| `owner_fov_id` | string | The owner of `sample_ownership_centroid_*`; null for `fov_not_placed` and where no FOV owns that point. |
| `assembly_status`, `assembly_reason` | string | "Statuses and reasons". |
| `assembly_flags` | string | `;`-joined `CELL_ASSEMBLY_FLAGS`; empty when none; `conflict` and `unclaimed` null with `conflicts=False`. |
| `conflict_with` | string | `;`-joined `obs_name`s of the conflicting retained cells; empty when none. |
| `duplicate_of` | string | For a cell excluded `duplicate_observation`, the `obs_name` of the retained cell of another FOV that keeps it ("Retention", rule 8); null otherwise, and null on every row with the rule off. |

**`molecules`**, one row per molecule of every input FOV, in placement table order, then
input order: `fov_id`; every column of the §2.9 molecule table, copied (for a
`molecules_only` sample, the `MoleculeTable` columns and `assignment_status` null);
`cell_namespace` (string: the namespace of the molecule's cell, so that
`(cell_namespace, cell_id)` is the cell key of a row of `cells`; null for a molecule
without a `cell_id`, so null on every row of a `molecules_only` sample; `load_assembly`
fills it for a table saved before the column existed from the `cells` row of the
molecule's `fov_id` and `cell_id`, null where no single row holds them); `sample_z`, `sample_y`, `sample_x` (float64, µm); `owner_fov_id`;
`assembly_status`; `assembly_reason`; `assembly_flags`.

**`counts`**, the long table of {doc}`assignment-contract` with `fov_id` and
`cell_namespace` added: one row per nonzero (counted cell, gene, compartment), with
`count` int64; `matrix()` on the result gives the dense form in `genes` order.

## Persistence

| Option | Layout | Effect on the golden values | Effect on existing outputs |
| --- | --- | --- | --- |
| **S1. The sample's export folder beside the placement (recommended)** | `<OUTPUT>/export/<sample>/`: the section `assembly` of `export.json` (the record, with the SHA-256 of every file), `tables/fovs.<fmt>`, `tables/cells.<fmt>`, `tables/molecules.<fmt>`, `tables/counts.<fmt>` and `sample.h5ad`. `<fmt>` is CSV by default or Parquet, through the checkpoint table writer ({doc}`checkpoints`, "Table formats"), as for the §2.9 tables. The other parts of `Dataset.export` write their files and sections in the same folder ({doc}`sample-output-contract`, "The export folder and the output profiles"). | None: the golden test pins values, not files. | New folder beside `stitching/<sample>/` (W-325 S1), shared by every part of the export entry (W-329 review, 2026-10-06). `output/tile_config_{sample}.csv`, `expr/{fovID}/*` and the legacy sample files are untouched while the legacy rules run. |
| S2. The legacy sample file names as the store | `expr/{sample}_raw.h5ad` (the AnnData) and `expr/{sample}_reads_assignment.csv` (the molecule table), with new content; the record in `uns`. | None. | The same names hold different populations and columns depending on the configuration; readers of earlier runs cannot tell which they read. The cell table of excluded cells has no file. |
| S3. One AnnData as the only store | `sample.h5ad` with every cell (retained and excluded) as `obs`, the molecule table in `uns`. | None. | Breaks "one row per counted cell" of the AnnData, needs anndata to read any table, and stores a large molecule table inside an H5AD. |

**Recommendation: S1.** It follows the run-record pattern of the placement and of the
§2.9 assignment folders, keeps every population in a plain table, and needs anndata only
for the one file that is an AnnData. `save_assembly(result, folder, *, table_format)`
writes the files atomically and replaces only the `assembly` section of
`folder/export.json`, creating the record when it does not exist; `load_assembly(folder)`
reads that section, checks every SHA-256 against it and raises `ValueError` on a
mismatch.

## Sample-level AnnData

Every assembled sample has `sample.h5ad`, written by `write_sample_h5ad(result, path, *,
annotation=None)` from the sample tables. It is not a second exporter: the AnnData is
built and written by the §2.9 AnnData code, which the implementation factors out of the
workflow adapter's `_write_raw_h5ad` (`src/python/starfinder/dataset/workflow.py:1395-1430`)
into one function of `starfinder.assignment`:

```python
def write_h5ad(path, obs: pd.DataFrame, keys: pd.DataFrame, counts: pd.DataFrame,
               genes: Sequence[str], *, layers: bool, uns: Mapping[str, str],
               var: pd.DataFrame | None = None,
               obsm: Mapping[str, np.ndarray] | None = None) -> Path: ...
```

`write_h5ad` builds the AnnData and writes the file at the given `path`, creating its
folder, and returns the path. The caller prepares `obs`, one row per cell to write, in
order, with the `obs_names` as its index, and `keys`, the same rows' key columns that
`counts` also has (`cell_id` within one FOV; `cell_namespace` and `cell_id` for a sample).
From the long counts of {doc}`assignment-contract` it builds `X`, float64 whole-cell
counts in `genes` order with 0 where a cell has no row, and, with `layers=True`, the
`nucleus` and `cytoplasm` layers: float64, 0.0 for a measured zero and NaN for every gene
of a row whose `obs["compartments"]` is not `available`. `var_names` are `genes`, with the
columns of `var` when given; `uns` and `obsm` are stored as given. Strings are written as
anndata expects (object strings, not the pandas Arrow strings that pandas 3 infers), as
`_write_raw_h5ad` does today. With no row in `obs` it writes an AnnData whose `X` has
shape `(0, len(genes))` (and layers of the same shape with `layers=True`).

It has two callers, and each prepares its own `obs`:

* **The workflow adapter**, for `expr/{fovID}/raw.h5ad`. `_cell_obs`
  (`starfinder/_snakemake.py`) prepares the legacy and cell-table columns of the kept cells;
  the adapter applies its tile-box filter to them as today and calls `write_h5ad` with the
  FOV result's counts, `layers` when the result has nuclei, and `uns["assignment"]`.
  `raw.h5ad` keeps the content it has today: the same rows, `obs` columns, `X`, layers and
  `uns`, which A18 of {doc}`assignment-algorithms` compares with the frozen helper. (W-568:
  the adapter writes `raw.h5ad` without an `assembly` block, or with molecules only; on the
  route of a saved assignment it writes the saved folder alone.)
* **`write_sample_h5ad`**, for `sample.h5ad`. It prepares the sample's `obs` from the sample
  cell table (the rows and columns below), and its `keys`, `obsm["spatial"]`, `uns` and,
  in `molecules_only` mode, `var["n_molecules_retained"]`, and calls `write_h5ad` with the
  sample `counts`. `save_assembly` calls it for `sample.h5ad`.

The content of `sample.h5ad`:

| Part | Content |
| --- | --- |
| `obs` | One row per counted cell ("Statuses and reasons": `assembly_status == "retained"` and §2.9 `status == "kept"`), in the cell table's order. `obs_names` are the cells' `obs_name`. Columns: `sample`, `fov_id`, `cell_namespace`, `cell_id`, the §2.9 cell columns that `raw.h5ad` carries (`size_voxels`, `expanded_size_voxels`, `size_physical`, `centroid_*`, `n_molecules`, `n_nuclei`, `correspondence`, `correspondence_flags`, `compartments`), `sample_centroid_z/y/x`, `sample_expanded_centroid_z/y/x`, `ownership_centroid`, `sample_ownership_centroid_z/y/x`, `assembly_flags`, `conflict_with`, and with `annotation` the columns of that sample's annotation row (`save_assembly`, which `export_sample` calls, passes none, so the workflow's `sample.h5ad` has no annotation columns). |
| `var` | The sample's `genes`, in order (`var_names`). |
| `X` | float64 whole-cell counts: row `i` equals the `whole` rows of cell `i` in the sample `counts`, which equal its §2.9 counts (identity 4). |
| `layers` | When the inputs gave nuclei (on the route after fusion: when the sample segmentation has a nucleus run): `nucleus` and `cytoplasm`, float64, 0.0 for a measured zero and NaN for every gene of a cell whose `compartments` is not `available`, as in `raw.h5ad`; a FOV assigned without nuclei contributes NaN rows. Without nuclei in any FOV: no layer. |
| `obsm["spatial"]` | `(n, 2)` float64 in µm, the X and Y of the ownership centroid: `[sample_ownership_centroid_x, sample_ownership_centroid_y]` of the cell table. On a row whose `ownership_centroid` is `expanded` (`centroid="territory"` and assign expanded) these equal `sample_expanded_centroid_x, sample_expanded_centroid_y`; on a row where it is `original` (`centroid="original"`, or no expansion) they equal `sample_centroid_x, sample_centroid_y`. X, Y is the order of the scanpy and squidpy convention; the ZYX values are the `obs` columns. |
| `uns` | `assembly`: the record as JSON text (the per-FOV assignment record hashes included); `spatial_unit`: `"µm"`; `frame_id`; `spatial_centroid`: which centroid `obsm["spatial"]` holds, as JSON text: the columns it copies (`sample_ownership_centroid_x`, `sample_ownership_centroid_y`), the `centroid` setting, and the number of rows whose `ownership_centroid` is `expanded` and `original`. |

**Which cells have a row.** The rows are the counted cells and no others. A retained cell
whose §2.9 status is `excluded_no_nucleus` has no row in the counts table and none in the
AnnData; it stays in the sample cell table with its assembly status, and its molecules keep
the assembly status of the cell. The identity a test checks: `adata.n_obs` equals the
number of rows of the sample cell table with `assembly_status == "retained"` and
`status == "kept"`.

**A test can check it against the sample tables**, on the counted cells (the cell-table
rows with `assembly_status == "retained"` and `status == "kept"`), in table order:

* `adata.n_obs` equals their number, and `obs_names` and
  `obs[["fov_id", "cell_namespace", "cell_id"]]` equal those rows; no other row of the
  cell table, in particular no retained row whose `status` is `excluded_no_nucleus`, has an
  `obs_name` in the AnnData;
* `adata.X[i, g]` equals the `whole` count of that cell and gene in `counts` (0 where there
  is no row), and the `nucleus` and `cytoplasm` layers likewise;
* each `obs` sample-coordinate column, `sample_centroid_z/y/x`,
  `sample_expanded_centroid_z/y/x` and `sample_ownership_centroid_z/y/x`, and
  `obs["ownership_centroid"]` equal the cell-table column of the same name on those rows
  exactly (NaN where the cell table holds NaN);
* `obsm["spatial"][:, 0]` equals `sample_ownership_centroid_x` and `obsm["spatial"][:, 1]`
  equals `sample_ownership_centroid_y`; on rows whose `ownership_centroid` is `expanded`
  they equal `sample_expanded_centroid_x` and `sample_expanded_centroid_y`, and on rows
  where it is `original`, `sample_centroid_x` and `sample_centroid_y`;
* `uns["spatial_centroid"]` names those two columns and the `centroid` setting;
* `var_names` equal `genes`.

**No segmentation** (`mode` `molecules_only`): the AnnData has no row (`X` of shape
`(0, n_genes)`), `var["n_molecules_retained"]` holds the retained molecules per gene, and
`uns["assembly"]` the record. The molecule table holds the molecules themselves.

**Dependency.** `write_sample_h5ad` imports `anndata`, from the existing `anndata` extra
(`starfinder[anndata]`: `anndata==0.12.6`, its pin for Python 3.11 and later, which are
the versions the package supports once the dependency plan of {doc}`stitching-contract` is
applied), lazily; `write_h5ad` imports it the same way. Without it, either raises
`ModuleNotFoundError` naming the extra. `assemble` and the tables need only the base
dependencies. No dependency or lock change.

**Where it differs from `create_sample_h5ad.py`**:

* built from the sample tables, so its rows are the counted cells of the new ownership
  rule, and FOVs without cells are part of the record rather than skipped;
* needs no `scanpy` and runs under pandas 3 (strings written as anndata expects);
* exists for every assembled sample, with no row when no cell is counted (today `concat`
  of an empty list raises);
* sample coordinates in µm (`sample_centroid_*`, `sample_expanded_centroid_*`,
  `sample_ownership_centroid_*`, `obsm["spatial"]`) instead of integer
  pixel `global_*`, and no legacy `volume`, `fov_x/y/z`, `seg_label` columns (the cell key
  replaces `seg_label`);
* the assembly record and the per-FOV assignment hashes in `uns` (today each FOV's `uns`
  is dropped by `concat`);
* the compartment layers are kept when some FOVs lack nuclei (NaN rows), where `concat`'s
  inner join drops them;
* flags and `conflict_with` columns; the same `obs_names` format.

## Workflow configuration

A Python-only `assembly` block (`backend: python`, and a `stitching` block, whose placement
it reads), the configuration of the assemble part of `Dataset.export`, which the one
per-sample rule `export_sample` runs through `_run_export` in the workflow adapter
({doc}`sample-output-contract`, "Workflow configuration"; its `export` block turns the
parts on and off):

```yaml
assembly:
  assignment: default          # the assignment name read per FOV; null: molecules only
  ownership: nearest_center    # nearest_center | midpoint_box
  centroid: territory          # territory | original
  tie: table_order
  conflicts: true
  duplicate_threshold: 0.5     # the duplicate rule; null: off
  table_format: csv            # csv | parquet
```

Its keys are the fields of `AssemblyConfig`, the `assignment` argument of
`Dataset.export` and its `table_format`; a default-tier test keeps the static schema
equal to them, as for the other blocks.

**Replacing the legacy filter.**

| Option | With an `assembly` block | Without one | Effect on the golden values | Effect on existing outputs |
| --- | --- | --- | --- | --- |
| **W1. Replace per configuration (recommended)** | `reads_assignment` writes the per-FOV files without the filter and without `global_*` (every kept cell and every molecule, with their §2.9 columns), and saves the FOV's assignment folder; `export_sample` runs `Dataset.export` per sample and writes `export/<sample>/`; `create_sample_h5ad` and `create_sample_reads_assignment` do not run; `tile_config_{sample}.csv` is still written by the legacy writer and read by nothing. | The four legacy rules run unchanged, with the adapter's filter. | None: the pins are legacy. `test_assignment_workflow.py` runs without the block and passes unchanged. | Configurations without the block are unchanged; with it, the per-FOV files change content (named in `docs/migration.md`) and the sample outputs move to `export/<sample>/`. |
| W2. Both side by side | The legacy chain runs as today and `export_sample` runs as well. | Unchanged. | None. | Two sample populations for one run (box-filtered and owned), which disagree where the rectangles do. |
| W3. Replace everywhere | The adapter drops the filter in every configuration and the legacy sample rules read assembly outputs. | The same as with the block. | None for the pins; `test_assignment_workflow.py` assertions on the legacy columns would need edits. | Every existing configuration changes its outputs at once. |

**Recommendation: W1.** It keeps legacy configurations reproducible, gives the new
populations one home, and follows the stitching block's pattern ({doc}`stitching-contract`,
"Workflow configuration").

Note (W-568): with an assignment name, `reads_assignment` now writes the saved folder alone
(no per-FOV `raw.h5ad`, `reads_assignment.csv` or `log.txt`), and with an `assembly`
block `stitch_sample` neither writes nor declares `tile_config_{sample}.csv`; molecules
only keeps the per-FOV files.

**The label grid with an `assembly` block.** `reads_assignment` imports a legacy label file
on the grid of the FOV's reference image, the file `reference_image` names (default
`fovs/{fov_id}/reference.tif`; its stored metadata, or for a file that stores none the
grid `voxel_size_xy` and `voxel_size_z` declare), the grid that `Dataset.export` gives
`assemble`, so that the saved assignment
passes assemble's grid check. A ZYX label file must have the reference image's shape, and a YX one the shape of
its Y and X, as a plane on the Z maximum projection of that grid; a file of any other shape
raises `IncompatibleGeometryError`. Without the block the file is imported, as before, on
a grid declared from `voxel_size_xy` and `voxel_size_z`.

The rule `export_sample` reads `stitching/{sample}/placement.json`, the saved assignment
folders (or molecule tables) of the sample's FOVs and their reference grids, and writes
the files of "Persistence" and of the other parts it runs. It runs once per sample, after
`reads_assignment` of every FOV of the sample on the route per FOV
({doc}`sample-segmentation-contract`, "The route setting") and after the placement exists; unlike
`reads_assignment` today, no rule reads another sample's placement.

**The assignment figures with ownership** (W-330, added after the implementation; opt-in
since W-568). With an assignment name (`assembly.assignment` not null), `assembly.previews:
true` and an assembly in the folder (the assemble part written or reused), `export_sample`
then draws `export/<sample>/previews/{fovID}/assignment.png`, and
`assignment_single_layer.png` for a ZYX label image, for each FOV of the sample whose
saved assignment is an input of the rule, with the FOV's ownership region and the
observations the assembly excluded (`plot_assignment` with the record of `fov_ownership`,
{doc}`assignment-contract`, "Diagnostics"), through the helper `reads_assignment` uses
for the image and the views. `reads_assignment` draws its own figures, without ownership,
into `fovs/{fovID}/previews/`: each kind has one writer and its own folder, and each rule
declares its folder as a directory output. `export.json` does not list `previews/`, and
`check_export` skips it (R2). By default (`previews: false`) neither rule draws a figure.
Without an `assembly` block the figures stay as `reads_assignment` draws them, in
`expr/{fovID}/`: the tile configuration is not an ownership source.

## Tests the implementation keeps and adds

* `test_workflow_assembly_golden.py`, `test_workflow_stitching_golden.py`,
  `test_assignment_golden.py` and `test_assignment_workflow.py` pass unchanged; their
  helpers stay as the frozen legacy reference.
* New tests: each check of the entry with its error; the coordinate rule against
  `placed_metadata(…).index_to_world` on fractional origins; ownership under O1 on the
  golden fixture's placement (boundaries near x 59.2 and y 42.2 for the fractional origins;
  every covered point one owner; T1 on a constructed exact tie); the retained sets of
  `tables/option_effects.json` on the fixture (3D: FOV1 retains scene cells 1, 2, 5, 9;
  FOV2 4, 6; FOV3 3, 7, 8; 2D: FOV3 also 9, with both copies flagged `conflict`, with the
  duplicate rule off); every
  identity of "Count accounting" on the fixture and on a `molecules_only` variant; the
  `fov_not_placed` and review refusal cases; the AnnData checks above, with a retained
  `excluded_no_nucleus` cell that has no AnnData row and an expanded FOV whose
  `obsm["spatial"]` holds the expanded centroids; the save and load
  round trip with its SHA-256 check; the workflow block and its schema.
* For the amendments of the W-329 review: a constructed placement with a `placeholder` at
  a corner and one at an edge of the grid, where every point of a placeholder's footprint
  that an owning FOV covers is owned by the nearest such FOV, the rest has no owner, no
  placeholder appears as an owner, and a neighbour's cell whose ownership centroid lies in
  the placeholder's footprint is retained; a placement whose FOVs are all placeholders,
  which raises `ValueError` because there is no owning FOV; `candidates` restricting the
  owner on that placement, its default equal to every owning FOV, and its `ValueError` for
  a placeholder or unknown `fov_id`; the territory path in the `assembly` section of `export.json` resolving to the
  file whose SHA-256 is recorded beside it; and `write_h5ad` with no cell, which writes an
  AnnData of shape `(0, n_genes)`, while `raw.h5ad` written through it keeps A18's values.

## `docs/migration.md` entries

The implementation adds the `starfinder.assembly` module and its names; the `assembly`
YAML block, read by the `export_sample` rule of {doc}`sample-output-contract`; `write_h5ad`
in `starfinder.assignment`, which the adapter calls for `raw.h5ad`;
and, for configurations with an `assembly` block only, the intentional changes: the
per-FOV `raw.h5ad` and `reads_assignment.csv` are not box-filtered and have no `global_*`
columns, the sample outputs are in `export/<sample>/` instead of
`expr/{sample}_raw.h5ad` and `expr/{sample}_reads_assignment.csv`, cells and molecules are
retained by ownership regions of the placement instead of the `tile_config` rectangles,
and every excluded observation is kept with its reason.

## Exclusions

No implementation, rename, dependency or lock change; no change to the placement contract
(W-325) or to the §2.9 label and assignment contracts; no merging of cells across FOVs
(the duplicate rule of "Retention", rule 8, decided on 2026-10-09, excludes one of two
observations and merges nothing), no recovery of complete cells across borders, no reassignment of molecules; no
fused image, label export or SpatialData (W-328); no real data (every measured number is
from the W-324 records and the W-326 derivations cited above).
