# Assignment contract: molecules to cells, correspondence, compartments and counts

Status: Accepted (W-309, 2026-10-05, at 3550723)

This page proposes the §2.9 assignment contract: an assign entry separate from
`FOV.run`, the rule that turns a molecule position into a voxel, direct assignment with
an explicit status for every molecule, one recorded label expansion, nucleus–cell
correspondence, compartments, the exclusion of cells without a nucleus, the cell table,
the count accounting, persistence, the boundary with §2.10 and the workflow translation
of the legacy keys. It builds on the label contract of {doc}`segmentation-contract`
("Label image", "What assign receives from segment") and does not redefine it: the label
image, its dtype, grid, target, geometry, namespace, record and saved files are as
specified there. The current behavior is recorded in {doc}`assignment-baseline`; the
numerical rules, resources and the validation design for the whole of §2.9 are in
{doc}`assignment-algorithms`. Nothing here is implemented and nothing is renamed.

Assignment has no model. Its behavior is fully determined by the label images, the
molecule table and the config, so every rule on this page can be pinned by hand-built
fixtures in the default test tier.

## Settled decisions this page follows

From the W-152 comment "§2.9 planning decisions (2026-10-03)", quoted in full in
{doc}`segmentation-contract` ("Settled decisions this page follows"):

* **D1 Separate entries.** Assign is its own entry, not a field of `PipelineConfig` and
  not a stage of `FOV.run`. Per FOV there are three calls: `FOV.run` (molecules), segment
  (label mask), assign (cell table and counts). Coordination may live on `FOV`. Assign
  checks the grid and records the hashes of the mask and the molecule table, and the
  documentation gives a notebook example of the three calls.
* **D3 External masks.** Masks from other tools reach assign through `import_labels`, as
  any other `SegmentationResult`.
* **D4 Reusable functions.** Label expansion is a plain function (`expand_labels`), not
  a method; assign calls it and does not reimplement it.
* **D7 Two specification issues.** Segmentation is specified by W-307 up to the label
  contract; this page takes over from there.
* **D2, D5 and D6** concern segmentation methods, their environments and the GPU. Assign
  is not a method and has no registry, needs only the base dependencies (no separate
  environment) and runs on CPU.

Proposed in the same comment and not objected to, and followed here: nucleus and cell
masks come from two runs; nucleus–cell correspondence belongs to assignment; the second
expansion of `reads_assignment.py` folds into one recorded expansion; the configuration
is a plan, not a recipe.

The development outline §2.9 (the Linear document "Chapter II development outline and
priorities", copied by the operator session on 2026-10-05) is settled scope. Its task
group 4, "Assignment and compartments" (default direct territory assignment, optional
recorded expansion, explicit unassigned molecules, validated nucleus–cell correspondence,
optional whole-cell/nuclear/cytoplasmic counts and the agreed configurable exclusion), and
the assignment part of task group 5, "Persistence and downstream outputs", are what this
page specifies. Its rules, and where this page follows each:

* Direct assignment uses supplied cell territories and keeps molecules outside them as
  unassigned ("Direct assignment and statuses").
* Optional label expansion is a separate operation applied once, with the original and
  the expanded masks preserved; physical distance when calibrated, explicit pixel or voxel
  units otherwise; volumetric expansion is distinguished from the legacy per-plane one;
  how the territories were obtained is recorded ("Label expansion").
* Ambiguous correspondence and nuclei extending outside their cells are flagged;
  whole-cell counts are kept and the affected compartment counts withheld pending
  resolution; no automatic clipping or repair ("Nucleus–cell correspondence",
  "Compartments").
* With nuclear segmentation, cells without a matched detected nucleus are excluded from
  the filtered cell table and the expression matrices by default, with an explicit reason
  and an option to keep them, and never without nuclear masks. Ambiguous correspondence is
  a separate status, not absence of a detected nucleus. Excluded cells keep their masks,
  metadata and molecule assignments; their status reaches their molecules, which stay
  distinct from molecules outside all territories. Possible cell residue is the filtering
  rationale, not a confirmed biological identity. A cell without a detected nucleus never
  has all its molecules classified as cytoplasmic ("Exclusion of cells without a nucleus",
  "Compartments").

The issue adds: there is no size or count filter, no probabilistic or transcript-based
assignment and no second exporter; assignment is FOV-local, and stitching, global
coordinates, overlap ownership and cross-FOV matching are §2.10's.

## Terms

* A **molecule** is one row of the final molecule table: a read with its spot identity
  `(spot_namespace, spot_id)`, its zero-based position `z, y, x` and its `gene_id`.
* A **territory** is the set of voxels of one positive value of the label image that
  assign samples: the cell run's labels (the **original territories**), or their expansion
  by assign (the **expanded territories**).
* A **cell** is one territory, identified by `(cell_namespace, cell_id)`: the cell run's
  `label_namespace` and the label value.
* A **nucleus** is one object of the nucleus run, identified by the nucleus run's
  `label_namespace` and its value.
* **Correspondence** relates nuclei to cells. A nucleus is **matched** to at most one
  cell; a cell has zero or more matched nuclei. Each cell has one **correspondence
  status**: `matched`, `ambiguous`, `no_nucleus` or `unavailable`.
* A **compartment** is the whole cell, its nuclear part (the territory inside its matched
  nuclei) or its cytoplasmic part (the rest of the territory).
* An **assignment status** says what assign decided for one molecule; a **cell status**
  says whether a cell is kept or excluded.

## Names

| Option | Names | Effect on the golden digests | Effect on the existing outputs |
| --- | --- | --- | --- |
| **N1. A `starfinder.assignment` module (recommended)** | Module `starfinder.assignment`; function `assign_molecules`; result `AssignmentResult`; config `AssignmentConfig` with `CorrespondenceConfig`; input `MoleculeTable` built by `molecule_table`; coordination `FOV.assign(config, *, cells, nuclei, …)`, `FOV.assignment_results`, `FOV.load_assignment(name)`; status vocabularies `ASSIGNMENT_STATUSES`, `CELL_STATUSES`. | None: the names are new. `test_assignment_golden.py` keeps every digest; its named edits ("Tests the implementation changes") add package calls beside the frozen helper. | None from the names: the files keep the names of "Persistence". `docs/api/` gains an `assignment.rst` page and the inventory entries that `docs/check_reference.py` requires. |
| N2. Inside `starfinder.segmentation` | `starfinder.segmentation.assign` returning `CellTable`; `FOV.assign`. | None. | The same files. One module would hold the label contract and the counts, which D7 split, and `assign` sits next to §2.8's `starfinder.barcode.assign_direct` (direct readout), a different meaning of "assign". |
| N3. A `starfinder.cells` module | `count_molecules` returning `CellCounts`; `FOV.count_cells`. | None. | The same files. The D1 call is named "assign"; a different verb in the API breaks the three-call vocabulary of the notebook. |

**Recommendation: N1.** It mirrors the D7 split (one module per specification), keeps
D1's verb on `FOV`, and `assign_molecules` cannot be confused with direct readout's
`assign_direct`.

New names under N1:

| Concept | Name | Kind |
| --- | --- | --- |
| Entry | `assign_molecules(molecules, cells, *, grid, nuclei=None, correspondence=None, config=AssignmentConfig())` | function |
| Result | `AssignmentResult` | frozen dataclass |
| Molecule input | `MoleculeTable`; `molecule_table(detection, reads, *, genes, population="final")`; `molecule_table_from_csv(path, *, spot_namespace, genes)` (legacy one-based CSV) | frozen dataclass, functions |
| Config | `AssignmentConfig`, `CorrespondenceConfig` | frozen dataclasses |
| Expansion | `ExpandLabelsConfig` and `expand_labels` of {doc}`segmentation-contract`, unchanged | reused |
| Statuses | `ASSIGNMENT_STATUSES = ("assigned", "unassigned", "excluded_cell", "outside_grid")`; `CELL_STATUSES = ("kept", "excluded_no_nucleus")`; `CELL_CORRESPONDENCE = ("matched", "ambiguous", "no_nucleus", "unavailable")`; `NUCLEUS_STATUSES = ("matched", "ambiguous", "no_cell")`; `COMPARTMENT_STATES = ("available", "withheld", "no_nucleus", "unavailable")` | tuples |
| Sampling | `sample_labels(labels, positions_zyx, *, geometry)` | function |
| Correspondence | `match_nuclei(nuclei, cells, *, config)` | function |
| Coordination | `FOV.assign(config=AssignmentConfig(), *, cells="cell", nuclei=None, name="default", population="final", correspondence=None, checkpoints=None)`, `FOV.assignment_results`, `FOV.load_assignment(name)` | method, attribute, method |
| Diagnostics | `summarize_assignment(result)`, `plot_assignment(result, *, image=None, z=None)` | functions |

## The assign entry

### Function

```python
def assign_molecules(molecules: MoleculeTable, cells: SegmentationResult, *,
                     grid: ReferenceGrid, nuclei: SegmentationResult | None = None,
                     correspondence: pd.DataFrame | None = None,
                     config: AssignmentConfig = AssignmentConfig()) -> AssignmentResult: ...

@dataclass(frozen=True)
class AssignmentConfig:
    expansion: ExpandLabelsConfig | None = None       # applied once, here, to the cell territories
    legacy_pixel_expansion: bool = False               # allow a pixel distance on a calibrated grid (legacy adapter only)
    correspondence: CorrespondenceConfig = CorrespondenceConfig()
    exclude_cells_without_nucleus: bool | None = None  # None: True when nuclei are given

@dataclass(frozen=True)
class CorrespondenceConfig:
    match_fraction: float = 0.5        # a nucleus is matched when one cell holds more than this share
    outside_tolerance: float = 0.1     # share of a matched nucleus allowed outside its cell (option C2, provisional)
```

`assign_molecules` applies the checks below, samples every molecule, derives or validates
the correspondence, partitions compartments, applies the exclusion, counts and returns an
`AssignmentResult`. It never reads or writes files, never segments and never reads a tile
configuration. `cells` is the label image whose objects become cells: a `cell` run, or a
`nucleus` run whose objects serve as territories (the legacy nucleus-only case), usually
with `expansion`. The record says how the territories were obtained
(`territory_source`): the run's name and target, whether it was segmented or imported
(its record's `methods` or `import` entry), and whether assign expanded it.

### Inputs

* **Molecules.** A `MoleculeTable` (frozen): `table` with exactly the columns
  `spot_namespace`, `spot_id` (string), `z`, `y`, `x` (float64, zero-based voxel
  coordinates of the molecule run's reference grid) and `gene_id` (string); `genes`, the
  ordered gene list of the count matrix (`Codebook.genes`, first-appearance order); the
  `population`; the `source` record; and `sha256`, the SHA-256 of those six columns in
  canonical form (sorted by key, each column's dtype and values), so the hash does not
  depend on row order.
  * `molecule_table(detection, reads, *, genes, population="final")` joins a
    `SpotFindingResult` with a read result by `(spot_namespace, spot_id)`, as
    `export_spots` does. `population="final"` takes the accepted reads of a
    `ReadFilteringResult` (the goodSpots population); `"called"` takes the reads whose
    `call_status` is `assigned` from any read result, before filtering. The source record
    names the result types, the filter config and the population.
  * `molecule_table_from_csv(path, *, spot_namespace, genes)` reads a legacy one-based
    `x, y, z, gene` CSV (MATLAB or `export_spots`, integer or float), subtracts 1, and
    gives each row the identity `spot_id = "csv:<row>"`; its source record names the file
    and its SHA-256. It exists for the workflow adapter.
  * Validation, in both constructors: unique, non-null keys; one FOV identity (the
    `spot_namespace` list `[dataset_id, sample_id, fov_id, subtile_id]`); finite
    coordinates (`ValueError` naming the first bad row); every `gene_id` non-null and in
    `genes` (`ValueError` naming the unknown genes; today they are dropped silently,
    {doc}`assignment-baseline` discrepancy 11).
* **Labels.** `cells` and the optional `nuclei` are `SegmentationResult`s
  ({doc}`segmentation-contract`), in memory or loaded with `FOV.load_segmentation(name)`.
* **Grid.** `grid` is the molecule run's reference grid (`ReferenceGrid`): in a FOV,
  `FOV.reference_grid()`, or `reference_grid_from_file("images/ref_merged/{fovID}.tif")`
  when the reference image is not resident.
* **Correspondence.** An optional supplied table with the columns `nucleus_id` and
  `cell_id` ("Nucleus–cell correspondence").

### Checks the entry applies

In this order, before any sampling; each uses only the arguments:

1. **Config.** `AssignmentConfig` and `CorrespondenceConfig` validate in `__post_init__`:
   `match_fraction` in [0.5, 1) (below 0.5 a nucleus could be matched to two cells),
   `outside_tolerance` in [0, 1), `expansion` an `ExpandLabelsConfig` or `None`.
2. **Molecules.** A `MoleculeTable` (`TypeError` otherwise).
3. **Targets.** `cells.target` is `cell` or `nucleus`; `nuclei.target` is `nucleus`
   (`ValueError`).
4. **Grid.** For `volume` and `extended` labels, `cells.grid.shape_zyx == grid.shape_zyx`
   and `cells.grid.metadata == grid.metadata`; for `plane` labels on a volume grid, the
   cell grid must equal `grid.projected(method=m)`, where `m` is the projection recorded in
   the cell run's record (`input.projection`) or, for an import, declared by it; a `plane`
   label image on a Z=1 grid must equal the grid. Otherwise `IncompatibleGeometryError`
   naming both shapes and frames. A `declared` grid of an import ({doc}`segmentation-contract`,
   "External-mask import") is checked here against the molecule run, and the record says so.
5. **Nuclei on the cell grid.** `nuclei.grid` equals `cells.grid` (shape and metadata),
   else `IncompatibleGeometryError`. A plane nucleus image with an `extended` cell image is
   refused: extend the nuclei too (`extend_labels_through_z`).
6. **Identity.** The first four entries of `cells.label_namespace` (and of
   `nuclei.label_namespace`) equal the molecules' FOV identity, else `ValueError` naming
   both.
7. **One expansion, with both masks.** The cell run's `record["operations"]` must list
   no `expand_labels`, whether or not `config.expansion` is set, else
   `ValueError("labels of run 'cell' were expanded by segmentation, which keeps no
   original mask; run segment without the expansion and set AssignmentConfig.expansion")`.
   An imported mask has no operations and is a supplied territory as it stands.
   With `config.expansion`: `unit="um"` needs a calibrated grid (`ValueError` otherwise);
   `unit="pixel"` on a calibrated grid needs `legacy_pixel_expansion=True` (`ValueError`
   otherwise) ("Label expansion").
8. **Supplied correspondence.** Validated as in "Nucleus–cell correspondence".

### What assign records

* the molecule table's `sha256`, `population` and `source`;
* for each label run: its name, target, geometry, `label_namespace`, the labels' SHA-256
  from its record, its `operations`, its full record, whether it is saved under its run,
  and the path and SHA-256 of the label file that holds it (linked or written; "Label images
  of a checkpointed assignment");
* the grid (shape, metadata, source, hash) and the outcome of the grid check
  (`checked` or `declared_checked`);
* the supplied correspondence table's SHA-256, when given;
* the config, the sampling rule, the expansion entry and the counts;
* the SHA-256 of every table and image it writes.

### Run record

A `FOV.assign` call with checkpoints writes one `assignment.json`, and every call keeps
the same mapping in `AssignmentResult.record` (without checkpoints, the `file` entries name
no path):

```json
{
  "format_version": 1,
  "stage": "assignment",
  "dataset_id": "...", "sample_id": "...", "fov_id": "FOV_001", "subtile_id": null,
  "name": "default",
  "cell_namespace": "[\"dataset\", \"sample\", \"FOV_001\", null, \"cell\"]",
  "territory_source": {"run": "cell", "target": "cell", "origin": "segmented", "expanded_by_assign": true},
  "grid": {"shape_zyx": [50, 512, 512], "metadata": {"frame_id": "..."}, "source": "fov:round1",
           "sha256": "...", "check": "checked"},
  "inputs": {
    "molecules": {"population": "final", "n": 1234, "sha256": "...",
                  "source": {"detection": "SpotFindingResult", "reads": "ReadFilteringResult", "filter_config": {}}},
    "cells": {"run": "cell", "target": "cell", "geometry": "volume", "labels_sha256": "...",
              "record_sha256": "...", "operations": [], "saved_under_run": true,
              "file": {"path": "../../segmentation/cell/labels.tif", "sha256": "..."}, "record": {}},
    "nuclei": {"run": "nucleus", "target": "nucleus", "geometry": "volume", "labels_sha256": "...",
               "record_sha256": "...", "operations": [], "saved_under_run": false,
               "file": {"path": "nucleus_labels.tif", "sha256": "..."}, "record": {}},
    "correspondence": {"source": "overlap", "sha256": null}
  },
  "config": {"expansion": {"distance": 0.7776, "unit": "um", "mode": "planar"}, "legacy_pixel_expansion": false,
             "correspondence": {"match_fraction": 0.5, "outside_tolerance": 0.1},
             "exclude_cells_without_nucleus": true, "exclusion_source": "default",
             "exclusion_rationale": "possible cell residue; not a biological identity"},
  "sampling": {"rule": "floor(c + 0.5)", "sampled_axes": "zyx"},
  "expansion": {"function": "expand_labels", "mode": "planar", "unit": "um", "distance": 0.7776,
                "spacing_yx": [0.1944, 0.1944], "voxels_added": 51234,
                "original": {"path": "../../segmentation/cell/labels.tif", "sha256": "..."},
                "expanded": {"path": "territories.tif", "sha256": "..."}},
  "calibration": "known", "calibration_source": "grid", "size_unit": "micrometer^3",
  "counts": {"molecules": 1234, "assigned": 1000, "unassigned": 200, "excluded_cell": 30, "outside_grid": 4,
             "cells": 425, "cells_kept": 410, "cells_excluded": 15, "nuclei": 430,
             "correspondence": {"matched": 400, "ambiguous": 10, "no_nucleus": 15, "unavailable": 0},
             "compartments": {"available": 370, "withheld": 40, "no_nucleus": 0, "unavailable": 0},
             "assigned_by_compartment": {"nucleus": 400, "cytoplasm": 520, "withheld": 80, "no_nucleus": 0,
                                         "unavailable": 0}},
  "outcome": "ok",
  "files": {"molecules": {"path": "molecules.csv", "sha256": "..."}, "cells": {"path": "cells.csv", "sha256": "..."},
            "counts": {"path": "counts.csv", "sha256": "..."}, "nuclei": {"path": "nuclei.csv", "sha256": "..."},
            "territories": {"path": "territories.tif", "sha256": "..."},
            "nucleus_labels": {"path": "nucleus_labels.tif", "sha256": "..."}},
  "software": {"starfinder": "...", "git_commit": "...", "packages": {}}
}
```

`counts.compartments` covers the kept cells; the excluded cells are the `no_nucleus` ones.
In this illustration the cell run was saved under its run and is linked, while the nucleus
run was not and is written into the assignment folder ("Persistence", "Label images of a
checkpointed assignment"); `inputs.<run>.record` holds the run's full record, so its
`label_namespace`, grid, target and operations resolve from `assignment.json` alone.
`outcome` is `ok`, or `empty` when the territory image has no object; `software` has the
content of `run.json`'s `code` and `environment` ({doc}`checkpoints`). The numbers above
are an illustration of the layout, not results.

### Coordination per FOV

`FOV.assign(config=AssignmentConfig(), *, cells="cell", nuclei=None, name="default",
population="final", correspondence=None, checkpoints=None)` is the third of D1's three
calls. `cells` and `nuclei` name segmentation runs, taken from `FOV.segmentation_results`
or loaded with `FOV.load_segmentation`, or are `SegmentationResult` objects the caller
made (an `import_labels` result, for example). It takes the molecules from
`FOV.spot_result` and
`FOV.filtering_result` (or the `candidates` and `pre_qc` checkpoints with the filter
re-applied) through `molecule_table`, the genes from the loaded codebook, and the grid
from `FOV.reference_grid()`; calls `assign_molecules`; stores the result in
`FOV.assignment_results[name]`; and, with `checkpoints`, writes the files of
"Persistence", including every label image it used that is not already saved under its run
there ("Label images of a checkpointed assignment"). Without `checkpoints` it writes
nothing. It never runs registration, detection, decoding or segmentation, and
`PipelineConfig` gains no field (D1).

```python
fov.run(pipeline, checkpoints=checkpoints)                   # molecules
fov.segment(plan, checkpoints=checkpoints)                   # label masks (W-307)
fov.assign(AssignmentConfig(), cells="cell", nuclei="nucleus",
           checkpoints=checkpoints)                          # cell table and counts
```

### Result

```python
@dataclass(frozen=True)
class AssignmentResult:
    molecules: pd.DataFrame      # one row per input molecule ("Molecule table")
    cells: pd.DataFrame          # one row per territory ("Cell table")
    counts: pd.DataFrame         # long counts ("Count accounting")
    nuclei: pd.DataFrame | None  # one row per nucleus ("Nucleus–cell correspondence")
    cell_labels: np.ndarray            # uint32 ZYX original territories: the cell run's labels
    territories: np.ndarray | None     # uint32 ZYX expanded territories; None without expansion
    nucleus_labels: np.ndarray | None  # uint32 ZYX nucleus labels; None without nuclei
    genes: tuple[str, ...]
    cell_namespace: str
    record: Mapping[str, Any]

    def matrix(self, compartment="whole", *, cells="kept") -> tuple[np.ndarray, pd.DataFrame]: ...
```

`__post_init__` checks the count identities of "Count accounting". `matrix` returns a
dense cells×genes count matrix (`int64`) in `genes` order with its cell table rows, for
`whole`, `nucleus` or `cytoplasm`; for a compartment, rows of cells whose compartments are
not `available` are absent, never zero. `cells="all"` adds the excluded cells to the
`whole` matrix for the before/after totals.

## Coordinate sampling

A molecule's `z, y, x` are zero-based voxel coordinates of the reference grid: voxel `i`
on an axis has its centre at `i` and covers `[i − 0.5, i + 0.5)`. This is the convention of
the spot table ({doc}`checkpoints`) and of the W-168 sample-export contract. The sampled
voxel is chosen per axis:

| Option | Rule per axis | Effect on the golden digests | Effect on the existing outputs |
| --- | --- | --- | --- |
| **P1. The voxel whose support holds the position (recommended)** | `i = floor(c + 0.5)`; inside when `0 ≤ i < n` | None: on integer coordinates `i = c`, so the per-molecule labels of the in-grid golden molecules are unchanged. | Python goodSpots files with float coordinates (`8.0`), which raise today, are read; for subpixel detectors (Spotiflow, Piscis) a position at `c = k + 0.5` goes to `k + 1`. |
| P2. Truncation | `i = floor(c)`; inside when `0 ≤ i < n` | None on integers. | Subpixel positions move up to one voxel towards the origin: a position at 3.9 samples voxel 3, whose centre is 0.9 away, instead of voxel 4. |
| P3. Integers only | Non-integer coordinates raise `ValueError` | None on integers. | Spotiflow and Piscis outputs, and any future subpixel refinement, cannot be assigned without rounding by the caller, which hides the rule again. |

**Recommendation: P1.** It is the nearest voxel centre, it agrees with the voxel-support
convention the sample-export contract already uses, and it equals today's indexing on the
integer coordinates the golden test pins. `np.rint` (half to even) is not used: it would
send 2.5 to 2 and 3.5 to 4, so the rule would depend on parity.

Rules that hold for every option:

* **Bounds.** A molecule whose sampled index lies outside `[0, n)` on any sampled axis
  gets the status `outside_grid` and no cell; it is kept in the molecule table. Nothing
  wraps (today one-based 0 reads the far edge) and nothing raises (today one beyond the
  grid raises). A position at exactly `−0.5` is inside (voxel 0); one at `n − 0.5` is
  outside.
* **Non-finite coordinates** are refused when the `MoleculeTable` is built.
* **Plane labels.** For a label image with geometry `plane` (Z=1), the territory is read
  at `y, x`; `z` does not select it, and the record says `"sampled_axes": "yx"`. `z` is
  still checked against the molecule run's grid (its Z size, before any projection), so a
  molecule off that grid in Z is `outside_grid`. For `volume` and `extended` labels,
  `z, y, x` are sampled (`"zyx"`).
* **Z=1 grids.** When the molecule run's grid itself has Z=1 (a projected FOV), molecules
  have `z = 0` and the labels are a `plane`; the rule is the same.
* **Nuclei** are sampled at the same voxel as the cells.

## Direct assignment and statuses

Assignment is direct: the molecule belongs to the territory that holds its sampled voxel.
Every molecule of the input table appears once in the result's molecule table with exactly
one status:

| Status | When | `cell_id` |
| --- | --- | --- |
| `assigned` | The voxel lies in a territory of a kept cell. | the cell |
| `unassigned` | The voxel is in the grid and its territory value is 0 (outside every territory). | null |
| `excluded_cell` | The voxel lies in the territory of an excluded cell ("Exclusion"). | the excluded cell |
| `outside_grid` | The sampled index is outside the grid ("Coordinate sampling"). | null |

There is no other outcome: no probabilistic or nearest-cell assignment, no distance
threshold, and no molecule is dropped. A molecule that lies in a nucleus but outside every
territory is `unassigned` and keeps its `nucleus_id`; it is not moved to that nucleus's
cell.

### Molecule table

`AssignmentResult.molecules`, one row per input molecule, in the input's order:

| Column | Type | Content |
| --- | --- | --- |
| `spot_namespace`, `spot_id` | string | Identity, as in the input. |
| `z`, `y`, `x` | float64 | As in the input. |
| `gene_id` | string | As in the input. |
| `voxel_z`, `voxel_y`, `voxel_x` | Int64 | The sampled voxel; null on an axis that is outside. `voxel_z` is 0 for a plane. |
| `assignment_status` | string | One of `ASSIGNMENT_STATUSES`. |
| `cell_id` | UInt32 | The territory value, or null. |
| `in_expansion` | boolean | True when the territory value comes only from assign's expansion (the original territory value is 0); null when assign did not expand. |
| `original_cell_id` | UInt32 | The original territory value at the voxel (0 in the expansion band); null when assign did not expand or outside the grid. |
| `nucleus_id` | UInt32 | The nucleus value at the voxel, 0 outside every nucleus; null without nuclei or outside the grid. |
| `compartment` | string | `nucleus` or `cytoplasm` for `assigned` molecules of cells whose compartments are available; otherwise the cell's state, `withheld`, `no_nucleus` or `unavailable` ("Compartments"); null when not `assigned`. |

The cell key is `(cell_namespace, cell_id)`; `cell_namespace` is the cell run's
`label_namespace`, stored once in the record.

## Label expansion

The expansion is optional and is applied once, by assign only
(`AssignmentConfig.expansion`), through `expand_labels` of {doc}`segmentation-contract`
with an `ExpandLabelsConfig` (`distance`, `unit` `pixel` or `um`, `mode` `planar` or
`volumetric`; no default distance). Assign is the place where the original label image is
always at hand: it is the cell run's `labels`. So both masks exist in every case this
contract allows, in memory and, for a checkpointed assignment, on disk by the one rule of
"Persistence" ("Label images of a checkpointed assignment"):

| Mask | Array | File of a checkpointed assignment | Identity | Hash |
| --- | --- | --- | --- | --- |
| Original territories | `AssignmentResult.cell_labels` (the cell run's labels) | the cell run's `segmentation/<run>/labels.tif` when it is saved under its run, else `assignment/<name>/cell_labels.tif` | `(cell_namespace, value)` | the labels' SHA-256 from the cell run's record |
| Expanded territories | `AssignmentResult.territories` | `assignment/<name>/territories.tif` | the same `(cell_namespace, value)`: `expand_labels` never creates, removes or renumbers a value, so a cell keeps its `cell_id` in both | SHA-256 in `assignment.json` |

Rules:

* **No expansion in segmentation for assignment.** A cell run whose record lists
  `expand_labels` is refused (check 7), because W-307's saved format keeps only the label
  image after its operations, so its original mask would be lost. W-307's `expand_labels`
  operation stays available for other uses of a label image; its results are not
  assignment territories. An imported mask is a supplied territory as it stands; if another
  tool expanded it, that is the import's provenance, recorded with the import.
* **Units.** On a calibrated grid (the cell grid's metadata with `spacing_zyx` and
  `spatial_unit`, or, for a plane on a projected grid, the molecule grid it was projected
  from, as for the sizes in "Cell table"), the distance is physical (`unit="um"`): `planar` converts it with the Y and X spacing,
  `volumetric` with the ZYX spacing. On an uncalibrated grid the unit must be explicit
  pixels (`unit="pixel"`; `um` raises). A `pixel` distance on a calibrated grid is accepted
  only with `legacy_pixel_expansion=True`, which the workflow adapter sets for the legacy
  `dilation_distance`, and the record then holds the distance's physical equivalent in Y
  and X.
* **Mode.** `planar` is the legacy per-plane expansion (Y and X only, plane by plane);
  `volumetric` grows in 3D. The record names the mode.
* **Record.** `"expansion"` holds the config, the unit and mode, the spacing used and the
  physical equivalent, the number of voxels added, the paths and SHA-256 of both masks,
  and `territory_source` says the territories were expanded by assign.
* **Results.** Every original territory is contained in its expanded territory. Each
  molecule records `in_expansion` and `original_cell_id`; each cell records
  `size_voxels` and the centroid of the original territory, and `expanded_size_voxels` and
  `expanded_centroid_*` of the expanded one. Nuclei are never expanded. Without
  `expansion`, the territories are the cell run's labels and the expanded columns are null.

`expand_labels(mode="planar", unit="pixel", distance=d)` reproduces today's
`reads_assignment.py` expansion, and the golden digests with expansion stay. The legacy
pair of expansions (segmentation's `distance`, then assignment's `dilation_distance`) is
not one expansion (the golden test shows 4 then 2 differs from 6), and "Workflow
configuration" states what the adapter does with a legacy expansion in segmentation.

## Nucleus–cell correspondence

Correspondence is available only when `nuclei` is given. It is either derived from overlap
(the default) or supplied.

**Derived from overlap.** For nucleus `n` with `s_n` voxels and cell `c`, let `o(n, c)` be
the number of voxels in both (cell territories as sampled, so after any expansion), and
`f(n, c) = o(n, c) / s_n` the share of the nucleus inside the cell.

* `n` is **matched** to the cell `c*` with the largest share when `f(n, c*) >
  match_fraction`; `match_fraction ≥ 0.5` makes `c*` unique.
* Otherwise, if `n` overlaps at least one cell, it is **ambiguous**: no cell holds enough
  of it. It is matched to none.
* Otherwise it is **no_cell** (entirely outside every territory).
* A matched nucleus whose share outside its cell, `1 − f(n, c*)`, exceeds
  `outside_tolerance` is flagged **outside** (it extends outside its cell, into the
  background or another cell).

Per cell: `n_nuclei` counts its matched nuclei, and several nuclei per cell are allowed
(`several_nuclei` flag; multinucleated cells and merged territories both look like this,
and assign does not decide which). The flags that make a cell's compartments unreliable
are:

| Flag | Raised on cell `c` when |
| --- | --- |
| `ambiguous_nucleus` | an ambiguous nucleus overlaps `c` |
| `nucleus_outside_cell` | a nucleus matched to `c` is flagged outside |
| `foreign_nucleus` | a voxel of `c` lies in a nucleus matched to another cell |

A cell with any of these three flags has its compartment counts **withheld**: its
whole-cell counts stay, its nuclear and cytoplasmic counts are not produced, and nothing
moves, trims, merges or re-matches a nucleus or a cell to repair it. `several_nuclei` does
not withhold anything.

**Correspondence status of a cell** (`CELL_CORRESPONDENCE`), one per cell:

| Status | When |
| --- | --- |
| `matched` | at least one nucleus is matched to the cell |
| `ambiguous` | no nucleus is matched to it, and at least one ambiguous nucleus overlaps it |
| `no_nucleus` | no nucleus is matched to it and no ambiguous nucleus overlaps it: no detected nucleus belongs to it. It may still hold voxels of a nucleus matched to another cell, which raises `foreign_nucleus`. |
| `unavailable` | no `nuclei` were given |

Ambiguous correspondence is a separate status, not the absence of a detected nucleus: an
`ambiguous` cell is never treated as `no_nucleus`, never excluded by the no-nucleus rule,
and always carries `ambiguous_nucleus`, so its compartments are withheld.

**Thresholds.**

| Option | `match_fraction`, `outside_tolerance` | Effect on the golden digests | Effect on the existing outputs |
| --- | --- | --- | --- |
| C1. Majority and containment | 0.5 (strict majority), 0.0 (any voxel outside flags) | None: the golden test has no nuclei, and the legacy path has no correspondence. | None of today's files has correspondence. In a new two-run plan, compartment counts are withheld for every cell touched by a nucleus that leaves its cell, even by one voxel; the bounded real examples report how many. |
| **C2. Majority with a tolerance (chosen, 2026-10-05; provisional)** | 0.5, 0.1 (a share strictly greater than 0.1 outside flags) | None. | As C1, with fewer withheld cells: up to a tenth of a nucleus may lie in the background without flagging its own cell. A part inside another cell still raises `foreign_nucleus` on that cell. Its basis is one culture crop of W-320, below. |
| C3. By label value for seeded runs | none: when the cell run's record shows a seeded method whose label rule gives each cell its seed's value, nucleus `k` is matched to cell `k`. | None. | Cheaper and exact for `seeded_watershed`; it says nothing for imported or independently segmented masks, so overlap would still be needed for them, and a seed masked out of the foreground gives a cell without its nucleus silently. |

**Decision: C2, provisional (Jiahao, W-321 review, 2026-10-05).** The default
`outside_tolerance` was raised from 0.0 (C1, the option this page recommended at W-309) to
0.1; `match_fraction` stays 0.5, the strict rule and the flags are unchanged. A matched
nucleus is flagged `outside` only when its share outside its cell is strictly greater
than 0.1, and a part of a nucleus inside another cell still raises `foreign_nucleus` on
that cell, whatever the tolerance. The evidence is the bounded culture example of W-320
(one crop, 18 matched nuclei, cells and nuclei segmented independently): with 0.0, 10 of
the 11 matched nuclei of the reference labels and 7 of the 8 of the Cellpose labels were
flagged `nucleus_outside_cell`, so the compartment counts of nearly every cell were
withheld, while the largest share of a nucleus outside its cell was 0.073. A tolerance of
0.05 would still have flagged 2 of 10 and 2 of 7; 0.1 flags none. The value is
provisional: it rests on one crop without annotation, is not an accuracy statement and
is not claimed to suit other samples. 0.5 remains the smallest share that makes the match
unique. With C2, the value agreement of a seeded run (C3) is recorded as a diagnostic
(`seed_value_agrees`), not used.

**Supplied.** A table with columns `nucleus_id` and `cell_id` (unsigned integers):
every value must exist in its image (`ValueError` naming the missing values), a nucleus
appears at most once, and a cell may appear several times. Supplied rows replace the
majority rule; the outside share is still computed from overlap, so the `outside` flag and
the three cell flags apply as above. A nucleus absent from the table is matched to no
cell, and its status follows from overlap exactly as for a derived one: `ambiguous` when it
overlaps a cell, `no_cell` otherwise. A table equal to the derived matches therefore gives
the derived result. The record holds `"correspondence": {"source": "supplied", "sha256":
…}`.

**Nucleus table.** `AssignmentResult.nuclei`, one row per nucleus: `nucleus_id` (UInt32),
`size_voxels` (int64), `status` (`matched`, `ambiguous`, `no_cell`), `cell_id` (UInt32,
null unless matched), `share_in_cell` (float64, `f(n, c*)` of the matched or largest
overlapping cell, 0 for `no_cell`), `share_background` (float64), `n_cells_overlapped`
(int64), `outside` (boolean) and `seed_value_agrees` (boolean, null unless the cell run is
seeded).

## Compartments

| Compartment | Voxels of cell `c` | Molecules |
| --- | --- | --- |
| whole | the territory of `c` (expanded when assign expanded) | every `assigned` molecule of `c` |
| nucleus | the territory of `c` inside the nuclei matched to `c` | `assigned` molecules of `c` whose `nucleus_id` is a nucleus matched to `c` |
| cytoplasm | the territory of `c` outside those nuclei | the other `assigned` molecules of `c` |

When each is available, recorded per cell in `compartments` (`COMPARTMENT_STATES`), in
this order of precedence:

* `unavailable`: no `nuclei` were given. Only whole-cell counts exist; every `assigned`
  molecule has `compartment = "unavailable"`.
* `no_nucleus`: nuclei were given and the cell's correspondence is `no_nucleus` (it is
  kept only when the exclusion is off). Only whole-cell counts exist. Assign does not infer
  that the cell's molecules are cytoplasmic: a cell without a detected nucleus has no
  nuclear and no cytoplasmic count, and its molecules have `compartment = "no_nucleus"`.
* `withheld`: the cell's correspondence is `matched` or `ambiguous` and it carries
  `ambiguous_nucleus`, `nucleus_outside_cell` or `foreign_nucleus`. Only whole-cell counts
  exist, pending resolution; its molecules have `compartment = "withheld"`.
* `available`: the cell's correspondence is `matched` and none of the three flags applies.
  Nuclear and cytoplasmic counts exist and sum to the whole-cell counts.

Because a cell without `foreign_nucleus` contains no voxel of another cell's nucleus, every
nuclear voxel inside an `available` cell belongs to one of its own nuclei, so the partition
is exact.

**Not available is never zero** (W-168: missing expression is never a measured zero):

* In `counts`, an `available` cell has `nucleus` and `cytoplasm` rows for its nonzero
  entries, so a gene without a row there is a measured zero. A cell whose `compartments` is
  `withheld`, `no_nucleus` or `unavailable` has no `nucleus` or `cytoplasm` row at all, and
  its `compartments` value in the cell table says why.
* `matrix("nucleus")` and `matrix("cytoplasm")` return only `available` cells (with zeros
  for measured zeros); the other cells are absent, not zero rows.
* In `raw.h5ad`, the `nucleus` and `cytoplasm` layers hold 0.0 for a measured zero and NaN
  for every gene of a cell whose compartments are not available; the `compartments` and
  `correspondence` obs columns say why.
* Per molecule, `compartment` holds the cell's state (`withheld`, `no_nucleus`,
  `unavailable`) instead of `nucleus` or `cytoplasm`.

`withheld` and `no_nucleus` stay apart: `withheld` follows a correspondence problem (an
ambiguous nucleus, a nucleus outside its cell, a foreign nucleus) that a later resolution
may settle; `no_nucleus` records that no detected nucleus belongs to the cell.

## Exclusion of cells without a nucleus

When `nuclei` are given, a cell whose correspondence is `no_nucleus` (no matched detected
nucleus, and no ambiguous nucleus over it) is excluded by default from the filtered cell
table, the counts and the expression matrices, with the reason `no_matched_nucleus`. A cell
whose correspondence is `ambiguous` is not excluded: it is kept with its whole-cell counts
and withheld compartments. Without nuclei nothing is excluded and the rule is not applied.
The reason is a filtering rationale (the territory may be cell residue), not a confirmed
biological identity, and the record says so (`"exclusion_rationale": "possible cell residue;
not a biological identity"`).

| Option | Default | Effect on the golden digests | Effect on the existing outputs |
| --- | --- | --- | --- |
| **X1. On when nuclei are given (recommended)** | `exclude_cells_without_nucleus=None` resolves to `True` with nuclei and `False` without; recorded with `"exclusion_source": "default"`. | None: the golden test and every legacy configuration use one label image and no nuclei, so nothing is excluded. | None for the legacy rule. In a two-run plan, `no_nucleus` cells leave the count tables and `raw.h5ad`; they stay in the complete cell table with their reason, and their molecules are `excluded_cell`. |
| X2. Off by default | `False`; exclusion only on request. | None. | None for the legacy rule; two-run plans keep `no_nucleus` cells in the counts, with whole-cell counts only (`compartments = "no_nucleus"`). |
| X3. Required when nuclei are given | `None` with nuclei raises `ValueError` asking for an explicit value. | None. | None for the legacy rule; every two-run configuration must state it. |

**Recommendation: X1**, the default the planning decision and the outline set. `False`
keeps the cells and is recorded as `"exclusion_source": "config"`.

How the status reaches molecules and counts: the cell's `status` becomes
`excluded_no_nucleus` and its `exclusion_reason` `no_matched_nucleus`; every molecule in
its territory becomes `excluded_cell` (with its `cell_id`), not `assigned`, and stays
distinct from `unassigned` molecules outside all territories; the cell has no row in the
count tables and no row in `X`. Its masks (original and expanded territories) are kept
unchanged, its row stays in the complete cell table with its metadata and `n_molecules`,
and its molecules keep their assignment, so the totals before and after the exclusion can
be reported. The filtered cell table is the rows with `status == "kept"`. No size or count
filter exists, here or anywhere in assign.

## Cell table

`AssignmentResult.cells`, one row per positive value of the territory image, in
increasing `cell_id`:

| Column | Type | Content |
| --- | --- | --- |
| `cell_id` | UInt32 | Label value. |
| `status` | string | `kept` or `excluded_no_nucleus`. |
| `exclusion_reason` | string | `no_matched_nucleus` for an excluded cell; null otherwise. |
| `size_voxels` | int64 | Voxels of the original territory (pixels for a plane). |
| `expanded_size_voxels` | Int64 | Voxels of the expanded territory; null unless assign expanded. |
| `size_physical`, `expanded_size_physical` | float64 | The same times the voxel volume (`spacing_z × spacing_y × spacing_x`), or the pixel area (`spacing_y × spacing_x`) for a plane; NaN when the calibration is unknown. |
| `centroid_z`, `centroid_y`, `centroid_x` | float64 | Mean voxel index of the original territory, zero-based on the reference grid (0.0 in Z for a plane); not truncated. |
| `expanded_centroid_z`, `expanded_centroid_y`, `expanded_centroid_x` | float64 | The same for the expanded territory; NaN unless assign expanded. |
| `n_molecules` | int64 | Molecules in the territory (`assigned` or `excluded_cell`). |
| `n_nuclei` | Int64 | Matched nuclei; null without nuclei. |
| `correspondence` | string | One of `CELL_CORRESPONDENCE`. |
| `correspondence_flags` | string | `;`-joined flags (`several_nuclei`, `ambiguous_nucleus`, `nucleus_outside_cell`, `foreign_nucleus`); empty when none; null without nuclei. |
| `compartments` | string | One of `COMPARTMENT_STATES`. |

The record holds `size_unit` (for example `micrometer^3`, or `micrometer^2` for a plane)
and `calibration`: `known` when the grid metadata has `spacing_zyx` and `spatial_unit`,
`unknown` otherwise. Unknown calibration is kept unknown: no pixel size is assumed. For a
plane on a projected grid, whose metadata carries no physical fields
(`ImageMetadata.projected`), the Y and X spacing come from the molecule run's grid, the
projection's source, and the record says `"calibration_source": "projection_source"`.

## Count accounting

The counts are an identity the tests check exactly, on every result (`__post_init__`) and
on every fixture of the validation design. With `M` the input molecules:

1. **One status each.** Every molecule of `M` appears once in `molecules`, with exactly one
   `assignment_status` from `ASSIGNMENT_STATUSES`:
   `n_assigned + n_unassigned + n_excluded_cell + n_outside_grid = |M|`.
2. **Whole-cell counts.** `counts` holds, for each kept cell `c`, gene `g` and compartment
   `whole`, the number of `assigned` molecules of `c` with gene `g`. Summed over cells and
   genes it equals `n_assigned`; per cell it equals the cell's `n_molecules`.
3. **Compartments.** For each kept cell with `compartments == "available"` and each gene,
   `whole = nucleus + cytoplasm`. Cells whose compartments are `withheld`, `no_nucleus` or
   `unavailable` have no `nucleus` or `cytoplasm` rows. Hence
   `Σ nucleus + Σ cytoplasm + Σ_{c kept, not available} whole(c) = n_assigned`, and per
   molecule `#nucleus + #cytoplasm + #withheld + #no_nucleus + #unavailable = n_assigned`.
4. **Exclusion.** Excluded cells have no count rows, and the sum of their `n_molecules`
   equals `n_excluded_cell`.
5. **Genes.** Every counted gene is in `genes`; no molecule is counted twice and none is
   counted without being `assigned`.

`counts` is a long table (`cell_id` UInt32, `gene_id` string, `compartment` string,
`count` int64) holding the nonzero entries only; `matrix()` gives the dense form.
`record["counts"]` holds the totals of 1 to 4, which `summarize_assignment` reports.

## Persistence

| Option | Layout | Effect on the golden digests | Effect on the existing outputs |
| --- | --- | --- | --- |
| **L1. Tables and a JSON record beside the segmentation runs (recommended)** | `<checkpoint dir>/<fov_id>/assignment/<name>/`: `molecules.<fmt>`, `cells.<fmt>` (every cell, kept and excluded), `counts.<fmt>`, `nuclei.<fmt>` (with nuclei), `territories.tif` (with an expansion), `cell_labels.tif` and `nucleus_labels.tif` (each only when that input is not saved under its run; see the table below) and `assignment.json`. Label images are `uint32` ZYX with `ImageMetadata`, written by `save_volume`. `<fmt>` is CSV or Parquet through the checkpoint table writer ({doc}`checkpoints`, "Table formats"), so tables round-trip exactly. | None: the golden test pins arrays and tables, not files. | `run.json`, the `FOV.run` checkpoints and W-307's `segmentation/<run>/` folders are untouched (linked, never written). The workflow keeps `expr/{fovID}/raw.h5ad` and `expr/{fovID}/reads_assignment.csv`, written by the adapter from the result, with the changes listed under "Workflow configuration". |
| L2. Per-FOV AnnData as the store | `assignment/<name>/assignment.h5ad`: `X` whole-cell counts, `layers` `nucleus` and `cytoplasm`, `obs` the cell table, `uns` the record; molecules and nuclei as CSV beside it. | None. | One file per FOV for cells, but `anndata` becomes a dependency of the assign entry (today it is locked only through the `spatialdata` extra), and the H5AD duplicates what `raw.h5ad` already carries. |
| L3. Per-FOV SpatialData | A Zarr store with the territories, points and table. | None. | `spatialdata` becomes a dependency of the assign entry, and the W-168 sample-export contract rules out a per-FOV SpatialData API and export fan-out. |

**Recommendation: L1.** It needs only the base dependencies, reuses the checkpoint table
format and the W-307 label files, and leaves AnnData and SpatialData to the outputs that
already own them. `FOV.load_assignment(name)` reads the folder and the linked
segmentation files, checks every recorded SHA-256 and returns the `AssignmentResult`.

**Label images of a checkpointed assignment.** One rule: a checkpointed assignment
(`FOV.assign(checkpoints=…)`) keeps every label image it used. A label image that is
**saved under its run** is linked, not copied; every other label image it used is written
into the assignment folder. A `SegmentationResult` is saved under its run when it was
written by `FOV.segment(checkpoints=…)` or loaded by `FOV.load_segmentation` from
`<checkpoint dir>/<fov_id>/segmentation/<run>/labels.tif` in the same checkpoint root as the
assignment, and that file's array SHA-256 equals the labels' SHA-256 when `FOV.assign` runs
(it checks). Every other input is unsaved: a direct `segment` result, a `FOV.segment` run
without checkpoints or under another root, a caller-made result, and an `import_labels`
result (its source file lies outside the layout and may hold another dtype). Without
checkpoints nothing is written and the arrays stay in `AssignmentResult`;
`assign_molecules` itself never writes.

| Mask | Exists when | Input | Without checkpoints | File with checkpoints | Links that point to it |
| --- | --- | --- | --- | --- | --- |
| Cell labels (original territories) | always | saved under its run | `AssignmentResult.cell_labels` | `segmentation/<cell run>/labels.tif`, linked | `inputs.cells.file`; `cells.cell_id`, `molecules.cell_id` and `molecules.original_cell_id` are values in it; `expansion.original` when expanded |
| Cell labels (original territories) | always | unsaved, including imports | `AssignmentResult.cell_labels` | `assignment/<name>/cell_labels.tif`, written | the same links, pointing to `cell_labels.tif` |
| Expanded territories | `expansion` set | computed by assign | `AssignmentResult.territories` | `assignment/<name>/territories.tif`, written | `expansion.expanded`; `cells.cell_id` and `molecules.cell_id` are values in it |
| Expanded territories | no `expansion` | none | `territories` is `None` | no file | `expansion` is `null`; the territories are the cell labels |
| Nucleus labels | `nuclei` given | saved under its run | `AssignmentResult.nucleus_labels` | `segmentation/<nucleus run>/labels.tif`, linked | `inputs.nuclei.file`; `nuclei.nucleus_id` and `molecules.nucleus_id` are values in it |
| Nucleus labels | `nuclei` given | unsaved, including imports | `AssignmentResult.nucleus_labels` | `assignment/<name>/nucleus_labels.tif`, written | the same links, pointing to `nucleus_labels.tif` |
| Nucleus labels | no `nuclei` | none | `nucleus_labels` is `None` | no file | `inputs.nuclei` is `null`; no `nuclei` table; `molecules.nucleus_id` is null |

Each mask's file depends only on its own row, so every combination of the cell input
(saved, unsaved, imported), the nucleus input (absent, saved, unsaved), the expansion (with,
without) and the checkpoints (with, without) is the product of these rows. When the cell
run and the nucleus run are the same result (nucleus territories, the legacy nucleus-only
case), one file serves both and both links name it. A written label image is `uint32` ZYX
with the run's grid `ImageMetadata` (`save_volume`), and `inputs.<run>.record` holds the
run's full record, so `(label_namespace, value)` resolves from the assignment folder alone.
The linked `segmentation/<run>/labels.tif` files are prerequisites of
`FOV.load_assignment`: a linked file that is missing or whose SHA-256 differs makes the
reload raise `ValueError` naming the path and both hashes.

**Identities and links.**

| Object | Key | Link |
| --- | --- | --- |
| Molecule | `(spot_namespace, spot_id)` | to its read and spot rows in `pre_qc` and `candidates`; to its cell by `(cell_namespace, cell_id)`, nullable |
| Cell | `(cell_namespace, cell_id)`; `cell_namespace` is the cell run's `label_namespace` | to its original territory in the cell-label file and to its expanded territory in `territories.tif`, by the same value (files as in the table above) |
| Nucleus | `(nucleus run label_namespace, nucleus_id)` | to the nucleus-label file of the table above; to its cell by `cell_id` |
| Assignment | `assignment.json` | the SHA-256 of the molecule table, of every label file in the table above (linked or written), of both segmentation records, of the grid and of every written file |

**AnnData and SpatialData.** Assignment adds no exporter:

* The only AnnData file §2.9 writes is the existing per-FOV `expr/{fovID}/raw.h5ad` of
  the `reads_assignment` rule, written by the workflow adapter from `AssignmentResult`
  (`anndata` is imported there, in the workflow environment). `create_sample_h5ad.py`
  keeps aggregating it unchanged.
* The §2.4 sample export is the W-168 contract `starfinder.sample_export/1`. Its
  partial exporter (W-169/W-170) is on commit `4748a7d`, which is not on this branch, so
  the connection below is a mapping of vocabularies, not code that runs here. The saved
  tables are exactly what that contract asks for: "Assignment links" are the molecule
  table's `(spot_namespace, spot_id) → (cell_namespace, cell_id)`, with its
  `assignment_status` (`assigned`; `unassigned` and `outside_grid` become W-168's
  `unassigned` with the status as the reason; `excluded_cell` becomes `unassigned` with
  the reason `excluded_cell`, and its cell a mask-only cell); "Cells and expression" are
  `raw.h5ad` joined by the explicit cell key; "Mask/table relation" is `(label
  namespace, local label)` from the label files of the table above; "Zero-count cells" are kept cells
  with zero rows; "Mask-only cells" are excluded cells. W-168's `unavailable` means that
  no assignment was supplied and is never written by assign.

## Boundary with §2.10

Assignment is FOV-local: it reads one FOV's molecules and label images on that FOV's
reference grid, and nothing else. It reads no tile configuration, computes no global
coordinate, decides no overlap ownership, matches no cell across FOVs and filters nothing
by position.

What §2.10 receives from assign, per FOV:

* the cell table: `(cell_namespace, cell_id)`, `centroid_z/y/x` (and
  `expanded_centroid_*`) in zero-based index coordinates of the FOV's reference grid, sizes,
  `status` and `exclusion_reason`, `correspondence`, `n_nuclei`, the flags and
  `compartments`;
* the grid (`ReferenceGrid`: shape and `ImageMetadata`, whose origin, spacing and
  direction place the FOV), with which §2.10 maps centroids and territories into sample
  coordinates;
* the original and expanded territory label files and their SHA-256, for overlap and
  cross-FOV matching;
* the molecule table with each molecule's status and cell key, and the count tables.

What stays in the legacy script until §2.10 replaces it: the tile-configuration read
(`reads_assignment.py:39-42`), the global coordinates (`:78-80`, `:141-149`), the
overlap filter of cells and molecules (`:155-165`) and the tile rectangle in the plots.
The workflow adapter applies them to the package result exactly as the script does today
("Workflow configuration"), outside the package.

## Diagnostics

On demand, never by `FOV.assign`:

* `summarize_assignment(result)`: molecules per status, cells per status, per flag and per
  compartment state, nuclei per status, quantiles of `size_voxels`, `size_physical`,
  `n_molecules` and `n_nuclei`, and the totals before and after the exclusion. `FOV.assign`
  stores it in `record["counts"]`.
* `plot_assignment(result, *, image=None, z=None)`: the territory outlines (Z maximum, or
  plane `z`) over `image`, with molecules coloured by status, and the size, count and
  nucleus-count histograms. It replaces the four legacy PNGs and their hard-coded plane.

## Workflow configuration

A Python-only top-level `assignment` block:

```yaml
assignment:
  cells: cell              # a segmentation run name
  nuclei: nucleus          # a segmentation run name, or null
  population: final        # final (accepted reads) or called
  expansion: null          # or {distance: 0.78, unit: um, mode: planar}
  correspondence: {match_fraction: 0.5, outside_tolerance: 0.1}
  exclude_cells_without_nucleus: null   # null: on when nuclei is set
```

Its keys are the fields of `AssignmentConfig` and of the `FOV.assign` call; a
default-tier test keeps the schema equal to them. The block is rejected unless
`backend: python`. It is executed by the §2.13 rule that runs the three calls; until then
it serves the adapter.

**Translation of the legacy keys.** When the block is absent, the adapter
(`dataset/workflow.py`, a new `_reads_assignment`) turns the `reads_assignment` rule's
inputs and `rules.reads_assignment.parameters` into one assign call. The rule keeps its
name, inputs and outputs.

| Legacy input or key | Translation |
| --- | --- |
| `images/stardist_segmentation/{fovID}.tif` | `import_labels` with the target the segmentation adapter infers (W-307: `cell` for the `overlay` input, else `nucleus`); its grid is declared from the file's shape and the configured `voxel_size_z`, `voxel_size_xy`, recorded as declared. When `rules.stardist_segmentation.parameters.expand_labels` is true, the file was expanded in segmentation and no original mask exists: the adapter raises `ValueError` naming that key and the replacement (set it to false and give the distance as `reads_assignment.parameters.dilation_distance` with `expand_labels: true`, so assign expands once and keeps both masks). |
| `signal/{fovID}_goodSpots.csv` | `molecule_table_from_csv` with the FOV's `spot_namespace` and the codebook's genes; integer and float coordinates are both accepted. |
| `documents/genes.csv` | Checked against the codebook's genes; a gene of the CSV absent from the codebook, or the reverse, raises `ValueError` naming them. |
| `parameters.expand_labels: false` | `expansion: null`; `dilation_distance` is ignored and recorded. |
| `parameters.expand_labels: true`, `dilation_distance: d` | `expansion: {distance: d, unit: pixel, mode: planar}` with `legacy_pixel_expansion=True` (the declared grid is calibrated), when `rules.stardist_segmentation.parameters.expand_labels` is false. |
| both `expand_labels` keys true | `ValueError` naming both keys (the row of the label file above): the legacy pair of expansions is not one expansion (golden test), and a label image is expanded once, by assign. |
| `images/DAPI/{fovID}.tif` | Read only for the diagnostic plots. |
| `output/tile_config_{sample}.csv` | Not read by the package; the adapter applies the legacy overlap filter and global coordinates to the result (§2.10 boundary). |

The outputs keep their names. `raw.h5ad` keeps its legacy `obs` columns with their
legacy meaning, computed on the territories assign samples (the expanded ones when assign
expanded, as today): `sample`, `fov_id`, `volume` = `expanded_size_voxels` (or
`size_voxels` without expansion), `fov_x/y/z` = the truncated `expanded_centroid_*` (or
`centroid_*`), `seg_label` = `cell_id`, `global_*`. It gains `size_voxels`,
`expanded_size_voxels`, `size_physical`, `centroid_*`, `n_molecules`, `n_nuclei`,
`correspondence`, `correspondence_flags`, `compartments`, the layers `nucleus` and
`cytoplasm` (float64; 0.0 for a measured zero, NaN rows for cells whose compartments are
not available) when nuclei are given, and the record in `uns["assignment"]`. `X` stays
float64 whole-cell counts of the kept cells. `reads_assignment.csv` keeps its columns and
gains `spot_id`, `assignment_status`, `cell_id`, `in_expansion`, `original_cell_id`,
`nucleus_id` and `compartment`.

## Tests the implementation changes

* `test/test_assignment_golden.py`: every pinned digest stays. `legacy_assignment` stays
  as the frozen legacy reference with its cited lines, and its legacy-behavior tests stay
  (the empty branch for a FOV whose molecules miss every cell, the far-edge read, the two
  `IndexError` cases, the two expansions). Named edits, all additions: tests that build a
  `MoleculeTable` from the golden CSV without the gene-`Z` molecule (unknown genes now
  raise), run `assign_molecules` on the fixture's uncalibrated grid with
  `ExpandLabelsConfig(distance=4, unit="pixel", mode="planar")` or no expansion, and assert that the `cell_id` of every
  molecule equals `SEG_LABELS` (0 as null), that the whole-cell matrix equals the pinned
  counts (the gene-`Z` molecule was never counted), and that the pinned `volume` and
  `fov_*` columns equal `size_voxels` and the truncated `centroid_*` without expansion, and
  `expanded_size_voxels` and the truncated `expanded_centroid_*` with it (the legacy
  metadata is computed after the expansion); and tests of the new behavior on the same
  fixture (cells kept when no molecule is assigned, `outside_grid` for one-based 0 and
  beyond the grid, float coordinates accepted). When `reads_assignment.py` becomes an
  adapter call, `test_the_helper_follows_the_script` is removed, because the cited lines
  leave the script; the helper and its pins stay.
* New tests: each check of the entry with its error, the sampling rule at the bounds,
  the statuses and the count identities, correspondence and compartments on the
  hand-built fixtures of the validation design, the exclusion, the persistence round
  trip, `molecule_table` against `export_spots`, and the workflow key.
* `test_segmentation_golden.py`, `test_checkpoints.py`, `test_io.py`, `test_fov.py`,
  `test_e2e.py`, `test_coordination_contract.py`, `test_workflow_scripts.py`, the
  `test_readout_*` modules and every other existing test pass unchanged.
* The subsystem marker `segmentation` proposed by {doc}`segmentation-contract` covers
  assignment tests too; the golden test keeps `workflow` and gains it.

## `docs/migration.md` entries

The implementation adds: the `starfinder.assignment` module and its names; the
`assignment` YAML block; and these intentional changes of the legacy rule's outputs:
cells are kept when no molecule lands in them; float goodSpots coordinates are accepted;
molecules outside the grid are kept as `outside_grid` instead of wrapping or raising;
genes outside the codebook raise; a label file expanded by `stardist_segmentation`
(`expand_labels: true` there) raises, with the distance moved to
`reads_assignment.parameters.dilation_distance`; the new columns of `raw.h5ad` and
`reads_assignment.csv`. The notebook example of the three calls per FOV
that D1 asks for is added with them.

## Exclusions

No segmentation and no change to the label contract (W-307); no stitching, global
coordinates, overlap ownership or cross-FOV matching (§2.10); no probabilistic or
transcript-based assignment; no size or count filter; no automatic compartment repair; no
new exporter (§2.4's are reused); no real data and no accuracy claim (E04, W-123).
