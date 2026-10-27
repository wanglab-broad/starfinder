# Assembly baseline: tile rectangles, global coordinates, the overlap filter and the sample tables

Status: Accepted (W-329, 2026-10-07, at 9033d46)

Note (W-583): {doc}`assembly-contract` and {doc}`assembly-algorithms` were accepted at W-329 (2026-10-07) and are implemented in `starfinder.assembly`; this page keeps the text of its revision.

This page records how sample assembly behaves at revision `0f5cbd8` (branch
`runner/s210-spec-20261006`, on `dev` after the §2.9 work and W-325), before the Chapter II
§2.10 work changes it. It is the reference that the golden test
`src/python/test/test_workflow_assembly_golden.py` pins. The proposed replacement is
described in {doc}`assembly-contract` and {doc}`assembly-algorithms`; neither is accepted.
The placement that assembly starts from is recorded in {doc}`stitching-baseline` and
proposed in {doc}`stitching-contract`; the per-FOV tables it reads are those of
{doc}`assignment-contract`. Paths are relative to the repository root, and line numbers
are at `0f5cbd8` unless a revision is named.

Evidence, cited by path and not repeated:

* W-324, run directory
  `/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/runs/W-324/20261006T063540Z-5cb2f421`
  (found with `timeout 30 find …/runs/W-324 -maxdepth 2 -name stitching-manifest.json`):
  the parity outputs `parity/script_tile_config_ln.csv` (the 2024 LN `tile_config`,
  reproduced byte for byte by the unchanged script) and
  `parity/prototype/tile_config_ln/T2_sum.csv` (the fractional sums before the cast).
* W-326, run directory
  `/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/runs/W-326/20261006T092110Z-48452974`:
  `tables/ln_ownership.json`, the ownership areas of the LN rectangles derived from those
  two records by `scripts/ln_ownership.py` (records only, no image).

## Where assembly lives today

The Python package has no sample-level code. Assembly is spread over a stitching script,
the §2.9 workflow adapter and two sample scripts:

| Piece | Where | Engine |
| --- | --- | --- |
| Tile offsets and non-overlap rectangles | rule `create_tile_config` (`workflow/rules/stitching.smk:60-70`), `workflow/scripts/create_tile_config.py` (T1 to T7 of {doc}`stitching-baseline`) | workflow environment (plotly) |
| Global coordinates and the overlap filter, per FOV | rule `reads_assignment` (`workflow/rules/reads-assignment.smk:8-22`), `workflow/scripts/reads_assignment.py`, which since W-317 calls `_run_reads_assignment` in `src/python/starfinder/dataset/workflow.py`; before W-317 the script itself (revision `6b384cd`, blob `26a7eab`) | Python package, in the workflow environment (anndata) |
| Sample molecule table | rule `create_sample_reads_assignment` (`reads-assignment.smk:45-54`), `workflow/scripts/create_sample_reads_assignment.py` | workflow environment |
| Sample H5AD | rule `create_sample_h5ad` (`reads-assignment.smk:29-38`), `workflow/scripts/create_sample_h5ad.py` | workflow environment (scanpy, anndata) |
| Spot merge on a Grid/Collection placement | `example/sequential_workflow/reads_stitching.m` with `src/matlab/ParseFijiTileConfiguration.m`; run by hand | MATLAB |

Since §2.9, assignment is FOV-local ({doc}`assignment-contract`, "Boundary with §2.10"):
the package's `assign_molecules` reads no tile configuration, and the workflow adapter
applies the legacy global coordinates and overlap filter to its result, outside the
package, as the script did.

## The rules and their files

Paths are relative to `OUTPUT = {root_output_path}/{dataset_id}/{output_id}`.

| Rule | Reads | Writes |
| --- | --- | --- |
| `create_tile_config` | `images/fused/{sample}/DAPI/dataset.xml`, `images/fused/{sample}/grid.csv` | `output/tile_config_{sample}.csv`, `output/tile_config_{sample}.html` |
| `reads_assignment` | `documents/sample-annotation.csv`; `images/DAPI/{fovID}.tif`; `images/stardist_segmentation/{fovID}.tif`; `signal/{fovID}_goodSpots.csv`; `documents/genes.csv`; `output/tile_config_{sample}.csv` **for every sample** (`expand` over `SAMPLE`, line 15) | declared: `expr/{fovID}/raw.h5ad`, `expr/{fovID}/reads_assignment.csv`; besides: `assignment.png`, `assignment_single_layer.png` (ZYX labels only) and `log.txt` in `expr/{fovID}/` |
| `create_sample_h5ad` | the annotation and the `raw.h5ad` of every FOV of the sample (`SAMPLE_TO_FOV`, lines 26-27) | `expr/{sample}_raw.h5ad` |
| `create_sample_reads_assignment` | the annotation and the `reads_assignment.csv` of every FOV of the sample (lines 42-43) | `expr/{sample}_reads_assignment.csv` |

W-567 later removed `documents/genes.csv` from `reads_assignment`'s inputs: the matrix's genes and their order are the codebook's ({doc}`migration`).

Because every FOV declares every sample's tile configuration, a change in one sample's
placement reruns `reads_assignment` for the FOVs of all samples. No rule reads a placement
status, a link or a review state: none exists.

## The rectangles of `create_tile_config.py`

Steps T1 to T5 (which view is which tile, the sum of the two translations, the integer
cast towards zero, the shift by `|min|` and the join with `grid.csv`) are recorded in
{doc}`stitching-baseline`. T6 and T7 are assembly:

| Step | Lines | Behavior |
| --- | --- | --- |
| Neighbours | 81-84, 86, 92, 98, 104 | For tile `tile_{row}_{col}`, the four names `tile_{row}_{col±1}` and `tile_{row±1}_{col}`. A neighbour counts only if its name is in `grid.csv` and its `id` is not 0 (not blank). Diagonal tiles are never neighbours. |
| Left and upper edge | 86-96 | Towards a counted left neighbour at `x_l`: `start_x = int((x_l + img_col − x)/2 + 0.5) + x`, the middle of the overlap rounded half up; otherwise `start_x = x`. The upper edge is the same in y with `img_row`. |
| Right and lower edge | 98-108 | Towards a counted right neighbour at `x_r`: `end_x = int((x + img_col − x_r)/2 + 0.5 + x_r)`; otherwise `x + img_col`. **The lower edge uses `img_col`, not `img_row`** (lines 106 and 108), both towards a lower neighbour and without one. |
| Blank tiles | 69-73, 127-130 | `start/end_*` and the normalized box are 0. A blank keeps its offsets `x, y, z`, and those offsets entered the minimum of T4. |
| Normalized box | 122-125 | `start_x_norm = start_x − x`, likewise for the other three: the box in the tile's own pixel indices. |
| Z | – | No box in z. The z offset is the truncated sum of the two z translations, in units of the xy pixel size (W-323 finding 3, {doc}`stitching-baseline`), plus `|min|`. |

The box is used half-open (`range(start, end)`), in integer pixels. With square tiles
(LN: `img_col = img_row = 1496`) the `img_col` error has no effect; with non-square tiles
the lower edge moves by `(img_col − img_row)/2` towards a lower neighbour and by `img_col −
img_row` without one.

**What the rectangles do on LN sample1.** From the 2024 `tile_config` (W-324 parity) at a
4 px raster, 119.4 million px² are covered by the 64 acquired tiles; about 33,600 px² are
owned by two rectangles (in 14 pieces, all but 80 px² within one tile of a blank cell,
whose neighbours own their full edge) and about 3,000 px² by none (13 pieces, about half
near blank cells and the rest at corners where four tiles meet, because each edge is set
by one neighbour pair and the edges of adjacent pairs need not meet); no rectangle reaches
outside its own tile there (`tables/ln_ownership.json` in the W-326 run directory, and the
note on its derivation in the W-326 notes).

## Global coordinates and the overlap filter

### `reads_assignment.py` before §2.9 (revision `6b384cd`, blob `26a7eab`)

| Step | Lines | Behavior |
| --- | --- | --- |
| Tile record | 39-42 | `output/tile_config_{sample}.csv` with `index_col=0`; the row whose `id` is the FOV number (`parse` of `fov_id_pattern`). |
| Molecule coordinates | 75-80 | `x, y, z` minus 1 (zero-based), then `global_x = x + tile.x`, likewise y and z: integer pixels and slices, XYZ order. `global_z` is written for 2D label images too. |
| Empty branch | 111-124 | When the label image has no cell, or when no molecule lies in any cell (including a FOV without molecules), the script writes an empty cell table with the `global_*` columns and **every molecule, unfiltered**, with `global_*` and `seg_label`, and nothing else: no plot, no log. A FOV with cells but no assigned molecule loses all its cells and keeps every background molecule, inside its box or not. |
| Cell coordinates | 141-149 | `regionprops` centroids `astype(int)` (truncation) into `fov_x, fov_y` (`fov_z` in 3D); `global_* = fov_* + tile offset`. |
| Cell filter | 155-158 | Keep the cells whose integer `fov_x` is in `range(start_x_norm, end_x_norm)` and `fov_y` in `range(start_y_norm, end_y_norm)`; z is never tested. The obs index is reset to `0 … n−1`. |
| Molecule filter | 160-165 | Keep the molecules whose `seg_label` is a kept cell, wherever they lie, then the background molecules (`seg_label == 0`) whose zero-based `x, y` are in the box. The molecules of dropped cells disappear, with no record. |
| Diagnostics | 167-237 | The four PNGs draw every cell and molecule and the kept ones; `log.txt` reports the assignment ratio and `total_cells` from before the filter. |
| Outputs | 240, 243 | `raw.h5ad` (kept cells) and `reads_assignment.csv` (kept molecules). |

### The §2.9 workflow adapter (`src/python/starfinder/dataset/workflow.py`)

`_run_reads_assignment` (lines 1474-1522) runs `assign_molecules` on the legacy label file
and then applies the same global coordinates and filter to the result, outside the
package:

| Function | Lines | Behavior |
| --- | --- | --- |
| `_fov_sample` | 1275-1283 | The sample and FOV number of `fovID` from the annotation's `fov_start`–`fov_end` ranges; a FOV in no range raises `ValueError`. |
| `_tile_record` | 1286-1293 | The `tile_config` row whose `id` is the FOV number; none raises `ValueError`. |
| `_in_box` | 1355-1358 | `start_x_norm ≤ x < end_x_norm` and `start_y_norm ≤ y < end_y_norm` on integers: the script's `range` test. |
| `_cell_obs` | 1361-1392 | `fov_x/y/z` = the truncated (expanded) centroid; `global_* = fov_* + tile offset` (lines 1379-1382); plane cells have no `fov_z`, `global_z`. |
| `_write_raw_h5ad` | 1395-1430 | Keeps the cells whose `fov_x, fov_y` are in the box (1421-1423); `X` and the compartment layers lose the other rows. Returns the kept `seg_label` values. |
| `_write_reads_assignment` | 1433-1463 | `global_* = x + tile offset` for every molecule (1449-1451). Keeps the `assigned` molecules of the kept cells, then the `unassigned` and `excluded_cell` molecules whose sampled voxel `x, y` is in the box, then every `outside_grid` molecule, unfiltered (1457-1463). |
| Diagnostics | 1506-1522 | `plot_assignment` draws the whole result; `log.txt` gives the assignment counts of the whole FOV, before the filter, and adds "{n} cells inside the tile box". |

The adapter has no empty branch: a FOV whose molecules all miss the cells keeps its cells
(with zero counts) and filters its molecules like any other FOV ({doc}`migration`,
"Segmentation and assignment workflow rules"). Otherwise its `raw.h5ad` and `reads_assignment.csv` carry
the script's values (golden test, `test_the_adapter_writes_the_legacy_values`).

## The sample tables

`create_sample_reads_assignment.py`:

1. Reads the annotation, finds the sample's `fov_start`, `fov_end` (lines 13-15) and loops
   over `fov_id_pattern.format(i=…)` (17-33). A missing file is printed and skipped; an
   empty file is printed and **still appended**, without `fov_id` and `sample` (25-31).
2. `pd.concat` (35); when every file is missing, `concat` of an empty list raises.
3. Merges the annotation columns except `fov_start`, `fov_end` by `sample` (38-39).
4. `unique_index = sample + '_' + fov_id + '_' + seg_label` (40): every background
   molecule of a FOV shares the key `<sample>_<fov>_0`, and a molecule is identified by
   nothing else (the §2.9 `spot_id` column is carried but not used).
5. Writes with the pandas row index (42).

`create_sample_h5ad.py`:

1. The same loop (19-32) with `scanpy.read_h5ad`; a FOV whose `raw.h5ad` has no row is
   printed and **skipped**; `var_names_make_unique()` on each.
2. `anndata.concat(adata_list, index_unique="_")` (34): an inner join of `var` and of the
   layers present in every FOV; `uns` is not merged, so each FOV's `uns["assignment"]`
   record is not in the sample file. When every FOV is empty, `concat` of an empty list
   raises.
3. Merges the annotation into `obs` by `sample` (37-38) and sets the obs index to
   `unique_index = sample_fov_id_seg_label` (39-40), which is unique because `fov_id` is.
4. `write_h5ad` (42).

Two facts of the locked environment (Python 3.12.12, pandas 3.0.0, anndata 0.12.6), shown
by the golden test: `scanpy` is not in the lock, and **under pandas 3.0.0 the script fails
at `write_h5ad`**: the string concatenation of line 39 gives an Arrow string column, which
anndata 0.12.6 cannot write (`IORegistryError`). The adapter avoids this for `raw.h5ad`
with `pd.option_context('future.infer_string', False)` (line 1419); the sample script does
not. The golden test runs it unchanged with that option off, the pandas 2 behavior the
script was written for, and pins the failure without it.

Neither script reads a coordinate, a placement or a status: the sample tables are the
concatenation of what each FOV kept, with the integer `global_*` columns of the filter.

## The legacy MATLAB merge: `reads_stitching.m`

For the Grid/Collection placement (`TileConfiguration.registered.txt`, parsed by
`ParseFijiTileConfiguration.m`: `|min|` added per axis, then `int32`, which rounds to
nearest), lines 36-120:

* The tiles are visited in the order of the tile configuration file.
* Each tile's spot coordinates are cast to `int32` and shifted by its offset (one-based).
  Spots beyond the fused image's width, height or depth are dropped; spots at or below 0
  are not checked.
* The tile's region is the **bounding box of its own spots** (inclusive), not its
  footprint.
* A spot is kept if it lies in its tile's region and not in the union of the regions of
  the tiles visited before: first come, first owned. The first tile keeps everything.
* Spots only: there are no cells, no labels and no record of the dropped spots. The
  result is `signal/fused_goodSpots.csv` and a plot.

It is not called by any rule; it is recorded because the policy differs in kind from the
workflow's (order-dependent, spot-extent regions) and was used for the two sample
datasets' reference outputs.

## What the §2.9 assign entry hands over per FOV

From {doc}`assignment-contract` ("Boundary with §2.10", "Persistence"), per FOV and per
assignment name:

* the cell table: `(cell_namespace, cell_id)`, `status` and `exclusion_reason`,
  `size_voxels`, `expanded_size_voxels`, `size_physical`, float `centroid_z/y/x` and
  `expanded_centroid_*` in zero-based index coordinates of the FOV's reference grid,
  `n_molecules`, `n_nuclei`, `correspondence`, `correspondence_flags`, `compartments`;
* the molecule table: `(spot_namespace, spot_id)`, float `z, y, x`, `gene_id`, the sampled
  voxel, `assignment_status` (`assigned`, `unassigned`, `excluded_cell`, `outside_grid`),
  `cell_id`, `in_expansion`, `original_cell_id`, `nucleus_id`, `compartment`;
* the long `counts` table (`whole`, and `nucleus`, `cytoplasm` where available), the
  record (`assignment.json`) with the SHA-256 of every table and label file, and the
  grid (`ReferenceGrid`: shape and `ImageMetadata`);
* the original and expanded territory label files.

The workflow adapter turns that result into the two legacy per-FOV files and applies the
filter; the package never sees a tile configuration.

## Discrepancies with the agreed §2.10 scope

Each line names the current behavior and the proposed resolution in
{doc}`assembly-contract` or {doc}`assembly-algorithms`.

| # | Current behavior | Proposed resolution |
| --- | --- | --- |
| 1 | **Integer tile offsets.** `global_*` add the truncated, shifted offsets of `tile_config` (T3, T4); the fraction of the placement is lost (up to 0.998 px in x and 0.926 px in y on LN, `tables/ln_ownership.json`). | Sample coordinates come from the placement's float64 origins in µm by the index rule of {doc}`stitching-contract`; nothing is cast ("Sample coordinates"). |
| 2 | **Centroids truncated to integers before the ownership test** (script line 141; adapter line 1379). With integer box edges and non-negative centroids the decision is the same as for the float centroid (`floor(c) < e` exactly when `c < e`), but the reported `global_*` lose up to one pixel, and the truncation would decide once the edges are fractional. | Ownership is tested on the float centroid in sample coordinates; the centroids of the cell table are carried unrounded. |
| 3 | **Ownership tested in XY only**; z is never tested, although tiles carry their own z offsets. | Ownership in Y and X with a stated single-layer rule: a sample whose FOVs are not one layer is refused ("Ownership regions", option Z1). |
| 4 | **`img_col` used for the Y extent of a tile** (`create_tile_config.py:106, 108`). In the golden fixture the band `y ∈ [43, 51)` is owned by two FOVs and one cell is kept twice. | Each FOV's footprint comes from its `ReferenceGrid` shape and spacing; no YAML tile size is read. |
| 5 | **The empty branch bypasses the overlap filter** (script lines 111-124): every molecule is written unfiltered and the cells are dropped. The adapter removed the branch but keeps the filter. | One rule for every FOV: no branch. A FOV without assigned molecules keeps its cells, and a sample without segmentation is assembled from molecules alone by the same position rule ("Retention"). |
| 6 | **Molecules of dropped cells are discarded without a record** (script lines 160-165; adapter 1457-1463). | Every molecule of every FOV stays in the sample molecule table with an assembly status and a reason ("Statuses and reasons"). |
| 7 | **Diagnostics computed before the filter**: the script's log and plots, and the adapter's plot and assignment counts, describe the FOV before ownership; only "cells inside the tile box" follows it. | Assembly diagnostics are computed from the sample tables after retention, per status and reason; the §2.9 per-FOV diagnostics keep describing the FOV. |
| 8 | **Ownership coupled to grid indices and blank tiles**: neighbours by `tile_{row}_{col±1}` names, diagonals ignored, a blank neighbour gives the tile its full edge, blank offsets enter the minimum. On LN about 33,600 px² are owned twice and about 3,000 px² by no tile. | Ownership regions computed in assemble from the placed footprints (origins and grid shapes), with no row or column index; padding views are not FOVs ({doc}`stitching-contract`, "Placement record"). |
| 9 | **No status or reason for an excluded observation**: a dropped cell or molecule leaves no trace, so the sample tables cannot be reconciled with the FOVs. | Assembly statuses `retained`, `excluded` with reasons, kept distinct from the §2.9 assignment statuses; count identities over both populations ("Count accounting"). |
| 10 | **The filter is applied in the workflow adapter, outside the package** (`_write_raw_h5ad`, `_write_reads_assignment`). | A package entry, `assemble`, separate from stitch and from export (D3); the adapter calls it ("Workflow configuration"). |
| 11 | Global coordinates are integer pixels and slices in XYZ order; z comes from translations in xy-pixel units (W-323 finding 3). | ZYX in µm in the sample frame of {doc}`stitching-contract`. |
| 12 | Next to a blank or at the sample edge, a rectangle reaches past its own tile (the fixture's FOV3 owns `y` up to 64 in a 48-pixel tile). | Ownership is bounded by the footprints; a point in no footprint has no owner. |
| 13 | Sample identities are `sample_fov_id_seg_label`; all background molecules of a FOV share `<sample>_<fov>_0`. | Cells keep `(cell_namespace, cell_id)` and molecules `(spot_namespace, spot_id)`; a readable `obs_name` is derived and checked unique ("Sample-level identities"). |
| 14 | Duplicates and losses across FOVs are invisible: in the fixture one cell is kept by two FOVs, and one molecule of a kept cell (outside its owner's tile) is kept by none. | Flags on cells and molecules make such cases visible, with no matching or merging ("Flags"); the page states that one observation per biological cell is not guaranteed. |
| 15 | `create_sample_h5ad.py` needs `scanpy` (not in the lock), fails under pandas 3.0.0, skips empty FOVs, raises when every FOV is empty, and drops each FOV's `uns` record. | The sample-level AnnData is written from the sample tables through the §2.9 AnnData code, needs only the `anndata` extra, exists for every assembled sample (with no row when no cell is retained) and holds the assembly record ("Sample-level AnnData"). |
| 16 | The adapter keeps every `outside_grid` molecule unfiltered, with `global_*` computed from coordinates outside the grid. | `outside_grid` molecules are excluded with the reason `outside_grid` and keep their sample coordinates. |
| 17 | `reads_assignment` declares every sample's `tile_config` as input. | The assemble rule runs once per sample on that sample's placement and FOVs. |
| 18 | The MATLAB merge keeps spots by first-come, spot-extent regions in file order. | Not carried forward; ownership does not depend on FOV order except for exact ties ("Ownership regions"). |
| 19 | No placement status or review is read: a failed or unreviewed placement is assembled like any other. | Assemble refuses a placement whose review is `required` without approval and excludes the observations of FOVs without an origin (`fov_not_placed`) ({doc}`stitching-contract`, "Failure policy"). |

## Golden test

`src/python/test/test_workflow_assembly_golden.py` (markers `workflow`, `dataset`,
`golden`, default tier) pins the current behavior on one sample built in the test:

* **Placement.** A 2 × 2 grid scanned down each column with the cell (row 1, column 1)
  blank, tiles of 8 × 48 × 64 voxels (`img_row` 48, `img_col` 64: not square), and the
  transforms of a BigStitcher `dataset.xml` with fractional stitching translations (sums
  37.4 and 55.4 cast to 37 and 55). `legacy_tile_config` of the W-325 test gives the
  offsets and rectangles, which are pinned.
* **Scene.** Nine ellipsoidal cells in one global label volume, cut at the integer offsets
  into one ZYX label image per FOV, each FOV numbering its own cells; their Z maximum is
  the 2D case. 24 molecules, written per FOV as one-based goodSpots CSVs for every FOV that
  images them. The cells cover each case the policy decides: a cell inside one FOV, a cell
  whole in two FOVs with its centroid on either side of an edge, a centroid exactly on an
  edge (`x = 60.0`, kept by the FOV whose box starts there), a cell in the band that
  `img_col` makes doubly owned, a cell cut by the tile's lower edge, and a cell wider than
  the overlap. Background molecules lie in each overlap.
* **Legacy per FOV.** `legacy_fov_assembly` follows `reads_assignment.py` at `6b384cd`
  (lines 78-80, 111-124, 141-165, cited in `SCRIPT_LINES`) on `legacy_assignment` of the
  §2.9 golden test. The script left the tree at W-317, so the cited lines were checked
  against blob `26a7eab` (`git show 6b384cd:workflow/scripts/reads_assignment.py`), not by
  the test. Pinned exactly, in 3D and 2D: the kept cells with `fov_*` and `global_*`, the
  kept molecules with `global_*` and `seg_label`, which scene cells and molecules each FOV
  keeps, and the empty branch (FOV3 with background molecules only: every molecule kept,
  no cell).
* **Adapter.** `_run_reads_assignment` through a stub `snakemake` object writes the same
  legacy values in `raw.h5ad` and `reads_assignment.csv`; in the empty-branch case it keeps
  the cells and filters the molecules (pinned).
* **Sample tables.** `create_sample_reads_assignment.py` unchanged and
  `create_sample_h5ad.py` unchanged, with a stand-in `scanpy` whose `read_h5ad` is
  `anndata.read_h5ad` and pandas' `future.infer_string` off; the obs index, the rows per
  `unique_index`, the counts against the molecule rows, and the two observations of one
  scene cell are pinned; without the option the H5AD script raises.
* **Sensitivity.** Moving one cell's centroid from `x = 60` to `x = 59` changes which FOV
  keeps it; changing one stitching translation (setup 2, x from −0.6 to 0.7) changes that
  FOV's offset and every pinned `global_x` of its cells by 1.

The values are identical over three separate single-thread runs in the locked environment
(Python 3.12.12, pandas 3.0.0, NumPy 2.2.6, scikit-image 0.26.0, anndata 0.12.6; W-326
notes). The 21 tests take about 7 s together; the slowest step, the module fixture that
runs the adapter on the three FOVs in 3D, takes about 3.4 s.
