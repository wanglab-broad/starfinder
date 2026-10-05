# Assignment baseline: molecules to cells, counts and the sample tables

Status: Proposed

This page records how molecule-to-cell assignment behaves at revision `6b384cd` (branch
`runner/s29-spec-20261004`, on `dev` after the §2.8 work), before the Chapter II §2.9
work changes it. It is the reference that the golden test
`src/python/test/test_assignment_golden.py` pins. The proposed replacement is described
in {doc}`assignment-contract` and {doc}`assignment-algorithms`; neither is accepted. The
label images assignment reads are specified by {doc}`segmentation-contract`, and their
current production by {doc}`segmentation-baseline`. Paths are relative to the
repository root, and line numbers are at `6b384cd`.

Assignment has no model and no measured behavior beyond what the golden test pins: the
model-dependent part of the labels it reads (StarDist inference) is pinned by the W-306
parity outputs, as {doc}`segmentation-baseline` ("Golden test") describes.

## Where assignment lives today

The Python package has no assignment code. Everything below is a Snakemake script or an
example run by hand:

| Piece | Where | Engine |
| --- | --- | --- |
| Per-FOV assignment, counts, cell table, overlap filter, plots | `workflow/scripts/reads_assignment.py` (rule `reads_assignment`, `workflow/rules/reads-assignment.smk:8-22`) | workflow environment |
| Sample H5AD | `workflow/scripts/create_sample_h5ad.py` (rule `create_sample_h5ad`, `reads-assignment.smk:29-38`) | workflow environment |
| Sample molecule table | `workflow/scripts/create_sample_reads_assignment.py` (rule `create_sample_reads_assignment`, `reads-assignment.smk:45-54`) | workflow environment |
| Watershed from PI nuclei and assignment on the stitched 2D tissue sample | `example/sequential_workflow/reads_assignment.py` | by hand |
| Whole-cell, nuclear and cytoplasmic counts on the stitched culture sample | `example/sequential_workflow/reads_assignment_cell_culture.py` | by hand |

## `reads_assignment.py` step by step

| Step | Lines | Behavior at `6b384cd` | Section |
| --- | --- | --- | --- |
| FOV and sample | 17-28 | `parse.parse(fov_id_pattern, fovID)['i']` gives the FOV number; the loop over `sample-annotation.csv` picks the first sample whose `fov_start`–`fov_end` range holds it. A FOV in no range leaves `current_sample` at the last sample of the table, silently. | FOV identity (the `Dataset`'s `sample_id` in the package) |
| Paths | 31-37, 44-46 | `root_output_path/dataset_id/output_id`; creates `expr/` and `expr/{fovID}/`. | plumbing |
| Tile configuration | 39-42 | Reads `output/tile_config_{sample}.csv` and the row whose `id` is the FOV number: offsets `x, y, z` and the non-overlap box `start_x_norm, end_x_norm, start_y_norm, end_y_norm`. | §2.10 |
| Images | 49-50 | `imread(snakemake.input[1])` (the DAPI image `images/DAPI/{fovID}.tif`, used only for plots and coverage) and `imread(snakemake.input[2])` (the label image `images/stardist_segmentation/{fovID}.tif`, which `stardist_segmentation` writes as uint16; the dtype is whatever the file holds). No shape, frame or dtype check between the two, or with the molecules. | assignment |
| Expansion, 3D | 52-57 | When `expand_labels` is true, `skimage.segmentation.expand_labels(…, distance=dilation_distance)` on each Z plane in place, so labels grow in Y and X only, in pixels. This is the second expansion when `stardist_segmentation` also expanded (its own `distance`). | assignment |
| Expansion, 2D | 66-67 | When `expand_labels` is true, one `expand_labels` call on the YX image. | assignment |
| Plot inputs and coverage | 53, 59-62, 64, 69-71 | 3D: the DAPI Z maximum, the label plane `img_z // 2` coloured over the DAPI plane with `label2rgb`, the label projection `max(labels > 0, axis=0)`; 2D: the DAPI image, the coloured labels and the label image itself. "Segmentation coverage" = labelled voxels (after the optional expansion) / DAPI voxels above the grey level 40, times 100. | diagnostics |
| Molecules | 74-77 | `pd.read_csv(goodSpots)`; `x, y, z` minus 1 (one-based to zero-based). Integer columns stay int64; a float column (`8.0`, as `starfinder.io.export_spots` writes it) stays float64. | assignment |
| Global coordinates | 78-80 | `global_x = x + tile.x`, likewise `y`, `z`. | §2.10 |
| Sampling | 82-91 | `points = [x, y, z]`; 3D: `labels[z, y, x]`; 2D: `labels[y, x]`, with `z` unused. No rounding, no bounds check: a negative index reads from the far edge, an index at or beyond the size raises `IndexError`, and float coordinates raise `IndexError` ("arrays used as indices must be of integer (or boolean) type"). The label value is written to `seg_label` (dtype of the label image). With no molecule, `reads_assignment = [0]` and no `seg_label` column is added. | assignment |
| Gene list | 102-109 | `documents/genes.csv`, headerless `gene,barcode`; column order of the matrix. | assignment |
| Empty branch | 111-124 | When the label image has no cell, **or when no molecule landed in any cell** (including a FOV with no molecule at all), the count matrix is 0×genes, the cell metadata is empty (with the `global_*` columns), and `raw.h5ad` and `reads_assignment.csv` (every molecule, unfiltered, with `global_*` and `seg_label` when there are molecules) are written. Everything after line 126 (cell loop, overlap filter, plots, `log.txt`) is in the `else` branch, so this branch writes no plot and no log. A FOV with cells but no assigned molecule loses its cells. | assignment |
| Count matrix | 127-139 | `cell_by_gene` float64, cells×genes; row `i` is the `i`-th region of `regionprops` (increasing label value); each molecule whose `seg_label` equals the region's label adds 1 to its gene's column, **if the gene is in `genes.csv`**; other genes are dropped without a count or a record. A cell with no molecule keeps a zero row. | assignment |
| Cell metadata | 131-149 | Per region of the label image after the optional expansion: `volume` = `region.area` (voxels in 3D, pixels in 2D, float64), the centroid truncated with `astype(int)` into `fov_x, fov_y` (`fov_z` in 3D), `seg_label`, `sample`, `fov_id`, and `global_x, global_y` (`global_z`) = centroid + tile offset. No unit and no spacing. | assignment; global columns §2.10 |
| AnnData | 150-153 | `AnnData(X=cell_by_gene, obs=current_meta, var=DataFrame(index=genes))`. | assignment output |
| Overlap filter | 155-165 | Keeps cells whose integer `fov_x`, `fov_y` lie in the tile's half-open non-overlap box (`range(start_*_norm, end_*_norm)`), resets the obs index to `0…n−1`; keeps the molecules of kept cells (wherever they lie) plus background molecules (`seg_label == 0`) whose `x`, `y` lie in the box. Molecules of dropped cells disappear. | §2.10 |
| Plots | 167-230 | `cell_centers_on_label.png`, `cell_centers_on_dapi.png`, `reads_on_label.png`, `reads_on_label_middle_layer.png`; the last selects molecules with `z == 15`, a hard-coded plane, while it draws the plane `img_z // 2`. | diagnostics |
| Log | 234-237 | `log.txt`: `"{:.2%} percent [{} out of {}] reads were assigned to {} cells"` with `cell_by_gene.sum() / len(bases)`, `cell_by_gene.sum()`, `len(bases)` and `total_cells`, then the coverage. The numerator is the count matrix's total: molecules in a cell whose gene is in `genes.csv` (lines 138-139), printed as a float (`15.0`). The denominator is every molecule of the FOV, of any gene, in a cell or not. Both and `total_cells` are taken before the overlap filter. | diagnostics |
| Outputs | 240, 243 | `expr/{fovID}/raw.h5ad` (the filtered AnnData) and `expr/{fovID}/reads_assignment.csv` (the filtered molecules without the row index: the goodSpots columns with zero-based `x, y, z`, then `global_x, global_y, global_z` and `seg_label`; `global_z` is written in 2D too). | assignment output |

The script imports `parse` and `anndata`, which the locked project environment does not
install (`parse` is not in `uv.lock`; `anndata` is locked only through the `spatialdata`
extra), so it runs only in a workflow environment that adds them.

### Configuration keys: `reads_assignment_config`

`workflow/schemas/config.schema.yaml:1418-1436`, referenced from `rules.reads_assignment`
(`:269-270`):

| Key | Schema | Script |
| --- | --- | --- |
| `run` | boolean, required | rule enabled |
| `resources` | `#/$defs/resources` | `mem_mb` of the rule |
| `parameters.expand_labels` | boolean | per-plane expansion (lines 55, 66) |
| `parameters.dilation_distance` | integer ≥ 0 | expansion distance in pixels |

No key has a schema default. The script always reads `expand_labels` (a missing key raises
`KeyError`) and reads `dilation_distance` only when `expand_labels` is true. `docs/examples/workflow-full.yaml` sets `expand_labels: false,
dilation_distance: 0`. The script also reads the top-level `fov_id_pattern`,
`root_output_path`, `dataset_id`, `output_id` and, for 3D labels, `img_z`.

## The rules and their files

Paths are relative to the output directory.

| Rule | Inputs | Outputs |
| --- | --- | --- |
| `reads_assignment` | `documents/sample-annotation.csv`; `images/DAPI/{fovID}.tif`; `images/stardist_segmentation/{fovID}.tif`; `signal/{fovID}_goodSpots.csv`; `documents/genes.csv`; `output/tile_config_{sample}.csv` for every sample | `expr/{fovID}/raw.h5ad`, `expr/{fovID}/reads_assignment.csv` (declared); `log.txt` and four PNGs in `expr/{fovID}/` (not declared, and written only outside the empty branch) |
| `create_sample_h5ad` | the annotation and the `raw.h5ad` of every FOV of the sample | `expr/{sample}_raw.h5ad` |
| `create_sample_reads_assignment` | the annotation and the `reads_assignment.csv` of every FOV of the sample | `expr/{sample}_reads_assignment.csv` |

`create_sample_h5ad.py` reads each FOV's `raw.h5ad` with `scanpy.read_h5ad`, skips
missing and empty ones with a message, calls `var_names_make_unique`, concatenates with
`anndata.concat(…, index_unique="_")`, merges the annotation columns by `sample`, and sets
the index to `sample_fov_id_seg_label`. When every FOV is empty, `concat` receives an empty
list; that case was not run here (`anndata` is not installed in the locked environment). It needs `scanpy`, which the lock holds only through the `spatialdata` extra (as a dependency of `spatialdata-io`).

`create_sample_reads_assignment.py` concatenates the FOV tables, adds `fov_id` and
`sample` (only to non-empty tables; an empty table is still appended), merges the
annotation and adds `unique_index = sample_fov_id_seg_label`, so every background
molecule of a FOV shares the key `…_0`. It writes with the pandas row index.

{doc}`workflow-downstream` ("Tile geometry and sample aggregation") describes the same
rules from the workflow side.

## The legacy examples

`example/sequential_workflow/reads_assignment.py` (stitched 2D tissue):

1. Reads `signal/fused_goodSpots.csv` (minus 1 on `x, y, z`), `images/fused/overlay.tif`
   and `images/fused/PI_label.tif`.
2. Seeds from the PI label centroids: `markers[x-1, y-1] = 1` where `(x, y)` is the
   centroid's (row, column) truncated to integers, so each seed sits one pixel up and
   left of the centroid; seeds outside the image are skipped.
3. A watershed of the dilated (disk 10) Otsu mask of the Gaussian-blurred (σ 5) overlay,
   with `watershed_line=True`, written as `labeled_cells.tif` (`uint16`).
4. Assignment `labels[y, x]`, the same count loop as the workflow script, metadata
   `area, x, y, seg_label`, and `cell_barcode_count.csv`, `cell_barcode_names.csv`,
   `meta.csv` and a dated H5AD.

`example/sequential_workflow/reads_assignment_cell_culture.py` (stitched culture):

1. Reads `Cell.tif`, `Nuclei.tif` and `Cyto.tif`, written by `create_3d_segmentation.m`,
   where `Cyto = Cell − Nuclei` on label values ({doc}`segmentation-baseline`, "The culture
   path"). Each is assigned independently with `labels[z, y, x]`.
2. The number of cells is that of `Cell.tif`; row `i` counts label value `i + 1`, so the
   rows assume consecutive labels. The metadata rows come from `regionprops` of each image
   and are not aligned with the count rows when a label value is missing.
3. The H5AD has `X` = whole cell and `layers['nucleus']`, `layers['cytoplasm']`. There is
   no nucleus–cell correspondence: row `k − 1` of the nuclear layer counts the molecules
   at which `Nuclei.tif` holds `k`, whatever `Cell.tif` holds there, and likewise for
   `Cyto.tif`, whose arithmetic is not a mask where the two values differ.

## The §2.8 molecule outputs assignment reads

{doc}`readout-contract` (accepted at W-280) and {doc}`checkpoints` define them:

| Element | Rule |
| --- | --- |
| Identity | `(spot_namespace, spot_id)`; `spot_namespace` is the JSON list `[dataset_id, sample_id, fov_id, subtile_id]` (`FOV._detect_round`, `src/python/starfinder/dataset/fov.py:845-846`). Joins use the pair, never row position. |
| Coordinates | `z, y, x` float64, zero-based voxel coordinates of the reference grid after `FOV.run`'s rotation, in the spot table (`SpotFindingResult.spots`, the `candidates` checkpoint). Local-maxima detectors give integer values; subpixel detectors (Spotiflow, Piscis) give fractions. |
| Read table | `BarcodeDecodingResult`, then `ReadScoringResult` and `ReadDeduplicationResult` (`pre_qc`), then `ReadFilteringResult` with `accepted` and `rejection_reasons`. A read carries `gene_id` (nullable), `entry_id`, `call_status` (`assigned`, `ambiguous`, `no_signal`, `unmatched`), `call_type`, score columns and, after deduplication, `is_representative`. |
| Final population | The accepted reads of `ReadFilteringResult`; the default filter keeps every `assigned` read with no score bound. `FOV.save_spots("goodSpots")` writes them through `export_spots` as `signal/{fovID}_goodSpots.csv`: exactly `x, y, z, gene`, one-based, MATLAB-compatible, with float coordinates (`8.0`). |

So the package's own goodSpots CSV cannot be read by `reads_assignment.py` (float
indices raise; golden test `test_float_coordinates_raise`), and the CSV drops the
identity, which the package tables keep.

## Discrepancies with the agreed §2.9 scope

| # | Discrepancy | Proposed resolution |
| --- | --- | --- |
| 1 | Coordinates index the label image without a bounds check (`:87`, `:89`): one-based 0 reads the far edge, a coordinate beyond the grid raises `IndexError` (golden test). | Every molecule is sampled through one rule with explicit bounds; a molecule off the grid gets the status `outside_grid` and is kept ({doc}`assignment-contract`, "Coordinate sampling"). |
| 2 | The subpixel-to-voxel rule is implicit: integers index directly and floats raise, including the package's own goodSpots output. | A stated rule for zero-based subpixel ZYX positions (recommended: the voxel whose support `[i − 0.5, i + 0.5)` holds the position), applied to the package tables, never to the one-based CSV. |
| 3 | Expansion is applied here (`:55-57`, `:66-67`) after an earlier expansion in `stardist_segmentation.py`, with its own distance; two expansions are not one (4 then 2 differs from 6 on the golden fixture by 585 voxels). | One expansion through W-307's `expand_labels`, applied once, by assign, and recorded; assign keeps the original and the expanded territories and refuses a label image already expanded in segmentation, whose original is not kept ({doc}`assignment-contract`, "Label expansion"). |
| 4 | Expansion of 3D labels is per slice, in pixels: labels never grow in Z, and a plane without labels stays empty. | `expand_labels` with an explicit mode (`planar` reproduces today; `volumetric` in physical units) and unit, recorded ({doc}`segmentation-contract`). |
| 5 | Assignment is coupled to the tile configuration, global coordinates and the overlap filter (`:39-42`, `:78-80`, `:141-165`), and the rule needs every sample's tile configuration. | Assignment is FOV-local and reads no tile configuration; §2.10 receives cell identities, centroids and statuses ({doc}`assignment-contract`, "Boundary with §2.10"). The workflow adapter keeps today's overlap filter outside the package until §2.10 replaces it. |
| 6 | No status for unassigned or excluded molecules: label 0 is the only record, the overlap filter drops molecules of dropped cells, and the empty branch writes a different population. | Every molecule keeps exactly one assignment status (`assigned`, `unassigned`, `excluded_cell`, `outside_grid`) in one table, whatever happens to its cell. |
| 7 | No nucleus–cell correspondence and no compartment counts; the culture example derives compartments from label arithmetic. | A correspondence table, supplied or derived from overlap, with flags and a correspondence status per cell; whole-cell counts always, nuclear and cytoplasmic counts only where correspondence allows, withheld for flagged cells and not produced for a cell without a detected nucleus, with no automatic repair. |
| 8 | Cell size is reported as `volume` without units; in 2D it is an area, and it is the expanded territory's size. | `size_voxels` of the original and the expanded territory, and a physical size with its unit when the grid is calibrated, unknown otherwise. |
| 9 | The scripts need packages outside the base package: `parse` (not locked), `anndata` and `scanpy` (locked only through the `spatialdata` extra, which the project environment does not install). | The assignment entry needs only the base dependencies (NumPy, pandas, scikit-image, tifffile); AnnData is written only by the existing per-FOV H5AD output, behind an optional dependency ({doc}`assignment-contract`, "Persistence"). |
| 10 | A FOV whose molecules all miss the cells loses every cell (`:111`, golden test). | Cells are kept whatever their counts; a cell with no molecule has a zero row. |
| 11 | Molecules whose gene is not in `genes.csv` keep a label but are not counted and not recorded; the log's numerator leaves them out while its denominator counts them. | The gene list is the codebook's; a molecule outside it is rejected before assignment with an error naming the genes. |
| 12 | Centroids are truncated to integers (`:141`). | Float centroids, zero-based index coordinates of the reference grid. |
| 13 | No check that the label image, the DAPI image and the molecules share a grid or frame, and nothing records which files were used. | Grid and identity checks before sampling; the hashes of the labels, the grid and the molecule table in the assignment record. |
| 14 | A 2D label image ignores `z` without saying so. | The territory of a `plane` label image is read at Y and X, `z` is still bounds-checked against the molecule grid, and the record says so; a volume samples Z too. |
| 15 | Diagnostics have hard-coded constants: the plane `z == 15`, the grey level 40 of the coverage, and `img_z // 2`. | On-demand diagnostics with stated inputs; no coverage against a fixed grey level. |
| 16 | A FOV outside every sample range is silently given the last sample (`:23-28`). | FOV identity comes from the `Dataset` (`sample_id`), as for the molecules. |
| 17 | The sample tables index cells and molecules by `sample_fov_id_seg_label`, which gives every background molecule the same key. | Cell keys are `(cell_namespace, cell_id)`; molecules keep `(spot_namespace, spot_id)` and a nullable cell key, as the W-168 sample-export contract requires. |

## Golden test

`src/python/test/test_assignment_golden.py` (markers `workflow`, `golden`) pins the
current FOV-local behavior on a hand-built fixture: a 16×64×64 `uint16` label image with
five ellipsoidal cells (labels 3, 7, 12, 20 and 25; cells 3 and 7 are 2 voxels apart and
cell 25 touches `z = 0`), its Z maximum as the 2D case, 19 molecules written as a
one-based `x,y,z,gene` CSV (inside cells, in the background, in the expansion band, one
of a gene absent from `genes.csv`, one on the `x = 0` face, one duplicate position and one
above a cell), and a five-gene `genes.csv`. It pins with exact SHA-256 digests the
per-molecule label, the count matrix and the FOV-local cell metadata (`volume`, `fov_x`,
`fov_y`, `fov_z`, `seg_label`), in 3D and 2D, with and without expansion (distance 4),
and the empty metadata of a FOV without cells. It also checks that every count follows
the per-molecule labels, and documents legacy behaviors that §2.9 changes: the empty
branch for a FOV whose molecules all miss the cells, the far-edge read for one-based 0,
`IndexError` beyond the grid and for float coordinates, and that two expansions are not
one. A digest-change test shows that changing the expansion distance or one molecule's
coordinate changes the per-molecule-label and count digests.

`reads_assignment.py` cannot run in the locked environment (`parse` and `anndata` are not
installed there), so the test runs one helper, `legacy_assignment`, whose lines are cited
against the script; a test checks that the 36 cited lines are still in the script. The
tile configuration, global coordinates, overlap filter, plots and file writing are not
pinned: they are §2.10 or output plumbing. Three separate single-thread processes
(`taskset -c 0`, every thread variable 1) recomputed every pinned value with byte-identical
output (`scripts/w308_compute_pins.py` and `scripts/pins-run{1,2,3}.json` in the W-308 run
directory). The 19 cases take about 0.25 s together.
