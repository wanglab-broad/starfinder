# Validation design: segmentation after fusion, the route setting and the consistency check

Status: Accepted (W-377, 2026-10-09, at 779ea77)

## Amendments after acceptance

* 2026-10-09: the row RS4 refuses `nuclei` naming a run of the target `cell` and `cells`
  naming no run of the configuration, in place of "`cells` naming a nucleus-only run of
  another target", which check 7 of {doc}`sample-segmentation-contract` allows (`cells`
  may name a nucleus run as territories). Found by W-426. Operator decision, pending
  Jiahao's confirmation.
* 2026-10-09: the row SG1 calls `segment_fused` with `identity=(dataset_id, sample_id)`,
  the keyword the contract gained, and its expected namespace follows. Found by W-426.
  Operator decision, pending Jiahao's confirmation.
* 2026-10-10: the row SG5 expects `empty` (`no_included_fov`) for a sample without an
  included FOV with the parts called alone, and the refusal of such a placement by
  `Dataset.export` before anything is written, as the contract's amended "The entry"
  states. Found by W-427 and its review. Operator decision, pending Jiahao's confirmation.
* 2026-10-10: the row SG4 adds a reuse with changed assignment settings: the
  `segmentation` section's `assignment` holds the new settings and the tables are made
  with them, as the contract's amended "The record" states. Found by W-427 and its
  review. Operator decision, pending Jiahao's confirmation.
* 2026-10-10: the row SO4 checks L6 by the rule of the route after fusion (after
  `segment` and before `assemble` writes on a fresh export, before anything is written
  with a reused segmentation), as the contract's amended "The label images" states, in
  place of "L6 holds". Found by W-428 and its review. Operator decision, pending Jiahao's
  confirmation.
* 2026-10-10: the row WF1 states what the default RD1 implies for an existing configuration
  without the new keys, in place of "existing configurations without the new keys give the
  same jobs as before": one that RD1 accepts plans the jobs it planned before; one that RD1
  refuses (a projection with an assignment and no `export.segmentation`) is refused when the
  workflow is read, naming `export.segmentation` and `route: per_fov`, and plans its earlier
  jobs once `route: per_fov` is added. The two could not both hold as written, and RD1 was
  decided. Found by W-430 and its review. Operator decision, pending Jiahao's confirmation.

This page is the validation design of {doc}`sample-segmentation-contract`: the fixtures
and the checks an implementation of the route after fusion, the route setting and
`compare_routes` must pass, in the form of {doc}`stitching-assembly-validation`. It
redefines nothing: every rule, column, status, identity and condition a check names is
used as that page and the pages it builds on state it. The central property is that on
synthetic fixtures with known cells across seams, **the route after fusion gives each cell
once and whole**: one row per cell, with the retained molecules of every FOV that owns a
part of it.

What this design is not: no check measures the accuracy of a segmentation, ranks a method
or a scale, or reads real data; no learned model is called, and no GPU is used. The
learned methods enter only through the conversion of their settings, which is checked
without loading a model. The fixtures are synthetic and small; the whole-sample costs of
{doc}`cells-across-fovs-proposal` are not reproduced here.

## Fixtures

Every fixture is generated in the test from the values stated here. No test reads the
network share, a run directory or a real image. Unit spacing means 1 µm on each axis. The
routine fixtures stay inside the §2.10 bound of the routine tier (synthetic tiles cut from
one scene of at most 16 × 256 × 256); the one marked `slow` stays inside the `slow` bound.

| Fixture | Construction | Seeds | Needs |
| --- | --- | --- | --- |
| `fused_pair` | The fixture of W-401's grounding calls (`scripts/grounding_calls.py`), rebuilt: two FOVs of 3 × 24 × 24 voxels (uint16, written as ZYX TIFFs with their axes, because `fuse` refuses a three-page TIFF without axis metadata, whose axes tifffile reports as SYX: W-401 call C4b), spacing (1.0, 0.5, 0.5) µm, origins (0, 0, 0) and (0, 0, 8) µm (4 µm overlap, O1 boundary at x 9.75 µm); one 2D scene of 24 × 40 pixels with three disc cells: cell 1 at (6, 10) µm of radius 2.5 µm across the seam, cells 2 and 3 inside one FOV each; nuclei as smaller discs; stains `dapi` (nuclei) and `flamingo` (cells), each FOV the scene cut at its origin and repeated over its three planes; molecules as in the grounding calls (one molecule of cell 1 seen by both FOVs, one by FOV1 only, one by FOV2 only); the placement from `placement_from_table`, approved. Variants at `k = 1` and `k = 2`. | none | `multiview-stitcher` extra |
| `fused_pair_seams` | The two seam cases of the comparison, on `fused_pair`'s two FOVs (3 × 24 × 24, spacing (1.0, 0.5, 0.5) µm, origins x 0 and 8 µm, FOV1 seeing scene columns 0 to 23, FOV2 columns 16 to 39, O1 boundary x 9.75 µm), with a scene of 24 × 40 pixels (pixel `i` at x 0.5·i µm) holding two cells, the same in every plane. **Case high**, cell 1: a disc of radius 5 pixels at (6, 19.5) pixels, (3, 9.75) µm, 78 pixels; FOV1 sees 73 of them, ownership centroid x 9.5959 µm, FOV2 sees 73, x 9.9041 µm; each centroid lies on its own FOV's side, so both observations are retained by rule 1; they share 68 pixels, 0.9315 of the smaller. **Case low**, cell 2: a box of rows 15 to 18 and columns 2 to 37 (y 7.5 to 9.0 µm, x 1.0 to 18.5 µm), 144 pixels, cut by both FOV edges; FOV1 sees 88 pixels, centroid x 6.25 µm, FOV2 sees 88, x 13.25 µm; both retained; they share 32 pixels, 0.3636 of the smaller. Molecules (y, x in µm, z 1 µm, gene), each seen by every FOV whose grid holds it: `m_a` (3, 8.0, A), `m_b` (3, 11.5, A), `m_c` (3, 12.0, B), `m_d` (8, 2.0, A), `m_e` (8, 10.0, B), `m_f` (8, 17.0, A), `m_g` (11, 10.0, B); FOV1 sees `m_a`, `m_b`, `m_d`, `m_e`, `m_g`, FOV2 sees `m_a`, `m_b`, `m_c`, `m_e`, `m_f`, `m_g` (11 molecule rows). Expected outcomes, computed with the present package for the route per FOV and with present functions as the contract specifies the route after fusion (W-401 `scripts/seam_cases.py`, record `calls/seam_cases.json`; `compare_routes` does not exist yet, so the comparison is computed there by the contract's rules): **route per FOV** (duplicate rule at 0.5): FOV1's cell 1 retained, FOV2's cell 1 excluded `duplicate_observation` with `duplicate_of` `s1_FOV1_1`; FOV1's and FOV2's cell 2 both retained; counted cells FOV1:1 (A 2), FOV1:2 (A 1, B 1), FOV2:2 (A 1, B 1). **Route after fusion** (`truth_import` at `k = 1`, grid 1 × 24 × 40): cells 1 (78 pixels; A 2, B 1) and 2 (144 pixels; A 2, B 1), one row each. **Comparison** (threshold 0.5): pairs FOV1:1 with 1 (share 1.0, matched), FOV1:2 with 2 (1.0, matched: the tie with FOV2:2 goes to the smaller cell key), FOV2:2 with 2 (1.0, not matched); summary: 2 matched pairs; cells in no matched pair: 1 of the route per FOV (FOV2:2) and 0 of the route after fusion; 1 cell of the route after fusion (cell 2) sharing at least the threshold with two cells of the route per FOV; counts per matched pair: FOV1:1 against 1, whole 2 and 3, per-gene difference 1; FOV1:2 against 2, whole 2 and 3, per-gene difference 1; molecule categories: `same_cell` 2 (FOV1 `m_a`, `m_d`), `different_cells` 2 (FOV2 `m_e`, `m_f`, in FOV2:2, which is not matched), `cell_on_one_route` 0, `no_cell_on_both` 1 (FOV2 `m_g`), `retained_on_one_route` 4 (FOV1 `m_b`, `m_e` on the route per FOV; FOV2 `m_b`, `m_c` after fusion), `excluded_on_both` 2 (FOV1 `m_g`, FOV2 `m_a`). | none | `multiview-stitcher` extra for the fused images |
| `fused_quad` | Four FOVs of 3 × 40 × 40 voxels at spacing (1.0, 0.5, 0.5) µm and origins (0, 0, 0), (0, 15.25, 0.25), (0, 0.25, 15.25) and (0, 15.25, 15.25) µm (fractional offsets, a four-FOV overlap), cut from one scene of 72 × 72 pixels of disc cells written in the fixture: one cell at the crossing of the four FOVs, two across a side seam, one cut by the sample's outer border, one wholly inside each FOV; per cell four molecules at fixed positions, each rendered by every FOV whose footprint holds it. Variant `fused_quad_notch`: the fourth FOV `placeholder`, so the fused image has an L-shaped covered region, with an L-shaped cell around the notch whose ownership centroid lies where no owning footprint covers it. | none | `multiview-stitcher` extra |
| `scene_plane` | The fixture of {doc}`stitching-assembly-validation` ("Fixtures"): `sample-tiny-v1` with one-plane FOVs of 1 × 36 × 36 and jitter (0, 1.5, 1.5) voxels, its truth placement, its stain. Its cells across seams are the truth cells whose `cell_fovs` rows name two or more FOVs that retain or exclude an observation; the test asserts that there is at least one. | 42 | generator addition; `multiview-stitcher` extra |
| `scene_plane_small` | The same change applied to `sample-small-v1` (3 × 3 FOVs of 1 × 96 × 96, jitter (0, 4.5, 4.5) voxels), by `dataclasses.replace`; no preset is added. It has cells at the crossing of four FOVs. | 42 | generator addition; `multiview-stitcher` extra; `slow` |
| `truth_import` | For a fixture and a factor `k`: the scene's cell and nucleus labels on the segmentation grid, each pixel taking the scene label (the sample label grid for the generated scenes, the scene array for the hand-built ones) at the pixel's centre by the nearest voxel, written as a uint32 OME-TIFF with the grid's metadata, imported by a run with `method: import`, `relabel: false`, so that `cell_id` is the truth ID (`import_labels` keeps the file's values with `relabel=False` and takes the grid's metadata as a declaration for a file without stored metadata: W-401 call C17). | none | hand-built |
| `seeded_runs` | The runs `nucleus` (`method: import` of the `truth_import` nuclei) and `cell` (`seeded_watershed` with `sigma_um` 1.0 on the fused `flamingo`, or the scene's stain, seeded by `nucleus`): a method that runs on the CPU without a model. | none | hand-built |
| `per_fov_routes` | The same fixture through the route per FOV: per FOV, the scene labels cut at its origin imported as in {doc}`synthetic-sample-scene` ("The path that needs no segmenter"), assigned with the same `AssignmentConfig`, saved as §2.9 assignment folders, and exported with `route: per_fov` into a second folder. `fused_pair_seams` states, with its numbers, the two seam cells both FOVs retain: one pair above 0.5 of the smaller observation (one excluded by the duplicate rule) and one below (both kept). | none | hand-built |
| `route_configs` | `ExportConfig` and workflow `export` blocks for every combination of `route` (null, `per_fov`, `after_fusion`), `dimensionality` (`projection`, `stack`, `both`), `segment` (true, false), `fuse` (true, false), with and without `segmentation`, and `assignment` set or null; the repository's existing workflow test configurations, and copies with `stitching`, `assembly` and `export` blocks. | none | hand-built |
| `length_configs` | `CellposeConfig(diameter=240, min_size=15, model_path=<a path written by the test>)`, `StarDistConfig(scale=0.25, model_path=…)`, `StarDistConfig(scale=1.0, block_size=(1, 256, 256), min_overlap=(0, 64, 64), context=(0, 52, 52), model_path=…)` and `SeededWatershedConfig(sigma_um=1.5)`; no model file is read beyond its existence and hash. | none | hand-built |

## Checks

"Needs" says what a check requires beyond the routine locked environment: **G** the
generator addition of {doc}`synthetic-sample-scene`; **H** nothing but hand-built
fixtures; **A** the `anndata` extra; **MV** the `multiview-stitcher` extra; **SD** the
`spatialdata` extra; **S** `slow`. A test that needs an extra is skipped without it,
naming the extra; a test of the case without an extra makes the import fail inside the
test. A check that names a condition T1 to T5 of {doc}`sample-segmentation-contract`
("The record") evaluates that condition in the test itself; that `check_export` evaluates
the same conditions is SO2. SG1 to SG3 and SG6 call `segment_fused` on a folder in which
`fuse` was called alone, so they need no part of `Dataset.export` beyond `fuse`.

### The route setting (RS)

| # | Check | Fixture | Metric | Pass/fail tolerance | Needs |
| --- | --- | --- | --- | --- | --- |
| RS1 | Defaults and the recorded route | `route_configs` exported | The recorded `route` and `route_source`; `segment` | With `route` null: `projection` (`max`) gives `after_fusion`, `stack` and `both` give `per_fov`, `route_source` `default`; a set value is kept with `route_source` `set`. With `assignment` null the export runs as `molecules_only` on either route, `segment` `off` with the reason `no_cells`. Exact. | H, MV, A |
| RS2 | A stack and a sum are refused | `route_configs` (`after_fusion` with `stack`; `after_fusion` with `projection: sum`; route null with `projection` and `projection: sum`) | Error; folder | `ValueError` naming `dimensionality` `stack` and `route: per_fov`, or naming `projection` and `route`, raised before anything is written: the SHA-256 of every file of the folder is unchanged and no file is added. Exact. | H |
| RS3 | Default without settings (RD1) | `route_configs` (route null, `projection`, an `assignment` name, no `segmentation`) | Error; folder | `ValueError` naming `export.segmentation` and `route: per_fov`; the folder unchanged. Exact. | H |
| RS4 | Check 7 | `route_configs` with one violation each: an image name not in `images`, a seed naming a later run, `nuclei` naming a run of the target `cell`, `cells` naming no run of the configuration, an unknown method key, `fuse: false` with `segment: true`, `assemble: false` with `segment: true`, a missing model path, a model file whose SHA-256 differs from `model_sha256` | Errors; folder | Each raises its named error (`ValueError`, `MissingModelError`, `ModelHashMismatchError`) naming the key, in the order of check 7; the folder unchanged; a valid configuration passes. Exact. | H |
| RS5 | Configuration schema | `route_configs` | Schema keys | The static workflow schema's `export` keys equal the fields of `ExportConfig` with `route`, `segment` and `segmentation`; a run's keys besides `name`, `target`, `inputs`, `seeds` and `method` equal the init fields of the method's config, kept equal to `SEGMENTATION_METHODS` as the §2.9 schema test does. Exact. | H |

### The segment part (SG)

| # | Check | Fixture | Metric | Pass/fail tolerance | Needs |
| --- | --- | --- | --- | --- | --- |
| SG1 | The input is the fused projection | `fused_pair` (`k` 1 and 2), `fused_quad` with `seeded_runs` | The `SegmentationInput` passed to `segment` (captured by wrapping the call), the grid | The image equals the fused projection file of each input image, channels in the run's role order; the grid's shape, spacing and origin equal the `fusion` section's output exactly, frame `sample:<sample>`; with `segment_fused` called with `identity=(dataset_id, sample_id)`, the label namespace is `[dataset_id, sample_id, null, null, <run>]`. Exact. | H, MV |
| SG2 | Lengths in pixels (LS2) | `length_configs` at `k` 1, 2 and 4 | The configs the route passes, from its conversion called directly; the record | `diameter / k`, `max(1, round(min_size / k²))`, `scale · k` in Y and X (accepted by `StarDistConfig` for every positive product), a block run refused at `k > 1` because `StarDistConfig` refuses block fields with a scale other than 1, `n_tiles` and `sigma_um` unchanged; the record's `lengths` holds each field as given and as used. A test keeps the table of pixel fields equal to the registered methods' fields. Exact. | H |
| SG3 | Label files and the section | `fused_pair`, `fused_quad` with `seeded_runs` and with `truth_import` | Files, `segmentation` section | One `segmentation_<run>.ome.tif` per run, uint32, passing the OME-TIFF layout checks of {doc}`sample-output-contract`, with the recorded grid; every field of "The record" present; T1 holds. Exact. | H, MV |
| SG4 | Reuse, by the rule of {doc}`sample-segmentation-contract` ("What is reused when a part has already run") | `fused_pair` exported; then again with `segment: false` and `fuse: true`; with `segment: false` and `fuse: false`; with `segment: false` after a label file is edited; after the placement is replaced by one with another origin; with `scale_yx` changed; with a source image edited so that the new fusion differs; with `segment: false` and no saved segmentation; with `segment: false` and changed assignment settings | Status; error; files; T1 and T5 | With `fuse: true`: `segment` `reused`, the fused files byte-equal the first export's, the tables equal it, T1 compares the recorded input SHA-256 with the fused files and holds, T5 holds. With `fuse: false`: no fused file remains, `segment` `reused`, T1 checks the label files only and holds, T5 holds. Refused before anything is written, the folder unchanged: an edited label file (`ValueError` naming the path and both SHA-256 values), another placement, another `scale_yx` (another grid), no saved segmentation. An edited source image: `segment` `failed`, reason `fused_input_changed` naming the image and both SHA-256 values, the error raised, the files of `fuse` present. With changed assignment settings: the `segmentation` section's `assignment` holds the new settings and the tables are made with them. Exact. | H, MV, A |
| SG5 | A part that cannot run | `fused_pair` with `multiview_stitcher` made unimportable; with the segmentation method's backend made unimportable (a registered method whose dependency import fails); with every FOV `failed`; with a method that raises | Statuses; files | `fuse` `not_run` gives `segment` `not_run` (`part_not_run`), `assemble` writes the `molecules_only` tables and `sample.h5ad` without a row, `complete` false, and `ExportIncompleteWarning` names both parts; a missing backend gives `not_run` (`extra_missing` naming the extra); no included FOV gives `empty` (`no_included_fov`) with the parts called alone, and `Dataset.export` refuses such a placement before anything is written; a method that raises gives `failed` with its type and message, the error is raised again, and the files of `fuse` stay. Exact. | H, MV, A |
| SG6 | Import runs | `fused_pair` with `truth_import`; the same file written on a grid one pixel larger | Result; error | The import is accepted, `relabel: false` keeps the truth IDs; the file on another grid raises `IncompatibleGeometryError` before anything is written. Exact. | H, MV |

### The sample assignment (SA)

| # | Check | Fixture | Metric | Pass/fail tolerance | Needs |
| --- | --- | --- | --- | --- | --- |
| SA1 | The conversion (MT1) | `fused_pair`, `fused_quad`, `scene_plane` | The `MoleculeTable` passed to `assign_molecules` (captured) | One row per retained molecule of the `molecules_only` assembly, in its order; `spot_namespace` `[dataset_id, sample_id, null, null]` on every row; `z, y, x` equal `(sample_zyx − O) ⊘ S` bitwise; no excluded molecule is passed. Exact. | H, MV |
| SA2 | Mapping back | the same | The sample molecule table | Every row keeps its original `fov_id`, `spot_namespace`, `spot_id`, `z`, `y`, `x`; the §2.9 assignment columns of each retained row equal `assign_molecules`' row of the same position; every excluded row has null assignment columns; the temporary keys occur in no table. Exact. | H, MV |
| SA3 | **Each cell once and whole** | `fused_pair` and `fused_quad` with `truth_import`; `scene_plane` with `truth_import` | Cell rows; cell territories; counts | T2 holds. Every truth cell whose territory on the segmentation grid has a pixel has exactly one row, with its truth ID. Every cell across a seam, the cell at the four-FOV crossing and the cells of `scene_plane` across seams are one row each, and the territory of each equals its truth on the grid pixel for pixel (no cut at a seam). For the hand-built fixtures, each cell's `n_molecules` and whole counts equal the fixture's written values, which hold the retained molecules of every FOV that owns a part of the cell. Exact. | H, MV (G for `scene_plane`) |
| SA4 | Once and whole with a method | `fused_pair`, `fused_quad` with `seeded_runs` | Cell rows; territories | One row per seed; the region of each seed across a seam has pixels on both sides of the O1 boundary; no row is empty. Not an accuracy: the regions are not compared with the truth beyond these conditions. Exact. | H, MV |
| SA5 | Assigned where it lies | every fixture of SA3 | Molecules | T3 holds for every retained molecule. On `scene_plane` the share of retained molecules whose `cell_id` equals their `true_cell_id` is reported, not required, and every molecule where they differ is one whose sampled pixel's imported label differs from its true cell (recomputed in the test). Exact for T3. | H, MV (G) |
| SA6 | `fov_id` of a cell | `fused_quad`, `fused_quad_notch` | `fov_id`, `owner_fov_id` | T4 holds; the cell at the crossing names the owner of its ownership centroid; the L-shaped cell of `fused_quad_notch` names the owning FOV of the nearest footprint centre with `owner_fov_id` null; `obs_name` is unique. Exact. | H, MV |
| SA7 | Identities | every fixture, both routes | `AssemblyResult` and `load_assembly` | The identities of {doc}`assembly-contract` as restated by {doc}`sample-segmentation-contract` hold (keyed by the cell key; identity 1 against the labels; identity 3 by position); a table edited so that a molecule names a cell that does not hold its pixel fails identity 3 or T3. Exact. | H, MV |
| SA8 | Molecules keep their rule | `fused_pair`, `fused_quad` | Molecule statuses | `assembly_status` and `assembly_reason` of every molecule equal those of the `molecules_only` assembly of the same inputs, except a retained molecule outside the segmentation grid, which is `outside_grid` (the fixture `fused_quad` places one within the far voxel of a footprint). Exact. | H, MV |

### The same outputs (SO)

| # | Check | Fixture | Metric | Pass/fail tolerance | Needs |
| --- | --- | --- | --- | --- | --- |
| SO1 | Same columns, keys and files | `fused_pair` exported on both routes (`per_fov_routes`) | The four tables, `sample.h5ad`, `export.json`, the file list | Column names, order and dtypes of every table equal; `obs` columns, `var`, the presence of `layers` (with nuclei), `obsm` keys and `uns` keys of `sample.h5ad` equal; `export.json`'s top-level keys equal and each section's keys equal except the section `segmentation`, the `assembly` section's `segmentation_files` and `assignment`, and the `labels` section's `segmentation_files`; the files equal except `segmentation_<run>.ome.tif` and `territories.ome.tif`. Every row of "Outputs, column by column" holds on the route after fusion: each "the same" column by the rule it names, each other column by its stated content. Exact. | H, MV, A |
| SO2 | Readers | the same | `load_assembly`, `sample_anndata`, `check_export`, `lookup_source`, `export_spatialdata` and its round trip | Each runs without error on both routes; `check_export` passes R1 to R3, F, L, S and T1 to T5, and fails T1 on a folder whose fused file was rewritten with other bytes after the segmentation and T5 on one whose label file the `assembly` section does not record; `lookup_source(cell=…)` on the route after fusion returns `assignment_folder` null, the row in the cell table, `label_file` and `label_value`; the SpatialData round trip (S1 to S8) passes with `cell_instance_id` from the cell key and radii from the segmentation grid's spacing; `layers` present when the sample segmentation has a nucleus run. Exact. | H, MV, A, SD |
| SO3 | The new column on the route per FOV (CK1) | the assemblies of {doc}`stitching-assembly-validation` (`assembly_golden`, `own_cases`, `empty_cases`, `status_cases`) | Sample molecule tables before and after | `cell_namespace` equals the FOV's cell namespace for a molecule with a `cell_id` and is null otherwise; every other column of every table equals the result before the change (`assert_frame_equal(check_exact=True)`); the implementation lists every pinned value that changes with the pair of tables that explains it, and a pinned value that changes for another reason is a defect. Exact. | H, A |
| SO4 | The label export on the route after fusion | `fused_pair`, `fused_quad` with `truth_import`, `fuse_labels: true` at `labels_scale_yx` 1, 0.5 and 0.25 | `cell_labels.ome.tif`, `labels` section | L1, L2 and L4 hold; L6 by the rule of the route after fusion ({doc}`sample-segmentation-contract`, "The label images"): with the bound lowered, a fresh export is refused after `segment` and before `assemble` writes (the files of `fuse` and `segment` present with their recorded SHA-256, no table, `sample.h5ad`, label image or store), and a call with `segment: false` on a saved segmentation leaves the folder byte for byte unchanged; L3 with the sample's territories; L5 with the grid of "The label images"; at scale 1 and `k = 1` each cell shows one ID on every pixel of its territory; the rule is `segmentation`. Exact. | H, MV |

### The consistency check (CR)

| # | Check | Fixture | Metric | Pass/fail tolerance | Needs |
| --- | --- | --- | --- | --- | --- |
| CR1 | Two routes from one truth | `fused_pair`, `fused_pair_seams`, `fused_quad` and `scene_plane`, each on both routes with `truth_import` and `per_fov_routes` | The report | Every counted cell of the route per FOV that is the only observation of its truth cell is matched to that cell of the route after fusion. `fused_pair_seams` gives exactly its written outcome: case high as one matched pair (FOV1:1 with 1) and no row for FOV2's excluded observation; case low as one cell of the route after fusion (cell 2) sharing at least the threshold with two cells of the route per FOV, FOV1:2 matched and FOV2:2 not; the summary's cells in no matched pair as 1 of the route per FOV (FOV2:2) and 0 of the route after fusion; the counts per matched pair and the totals per molecule category as written. On every fixture each molecule key has exactly one of the six categories, and the totals sum to the molecule rows of the two tables' key union (11 in `fused_pair_seams`). Exact. | H, MV, A (G) |
| CR2 | The report decides nothing | the same | Files; report | The SHA-256 of every file of both folders is unchanged by `compare_routes`; without `output` nothing is written (a read-only working directory); with `threshold` 0.25, 0.5 and 0.75 the pairs table differs only in the column `matched` and the summary only in the counts that depend on it. Exact. | H, MV, A |
| CR3 | Refusals | the same with one difference each: another placement, another gene list, another molecule key set, two folders of the same route, `output` inside an export folder | Errors | `ValueError` naming the difference, before anything is written. Exact. | H, MV, A |
| CR4 | Determinism | `fused_quad`, both folders | The report's tables | Two calls give equal tables; reversing the order of the cells in each folder's tables (by rewriting them in the test and recomputing their SHA-256) gives the same pairs and categories. Exact. | H, MV, A |
| CR5 | The whole of a larger scene | `scene_plane_small` on both routes | The report | CR1's conditions on a 3 × 3 grid with cells at the crossing of four FOVs. Exact. | G, MV, A, S |

### Workflow (WF)

| # | Check | Fixture | Metric | Pass/fail tolerance | Needs |
| --- | --- | --- | --- | --- | --- |
| WF1 | The rule's inputs by route | `route_configs` workflow copies | Snakemake dry run (`-n`) with the starfinder-v9 Snakemake, read-only | With `route: per_fov` `export_sample` needs `reads_assignment` of every FOV of the sample, as today; with `route: after_fusion` it needs the molecule tables and the source images and no `reads_assignment` output; an existing configuration without the new keys that the default RD1 accepts (a stack; a projection without an assignment; any configuration with `route: per_fov`) plans exactly the jobs it planned at the base revision; one that RD1 refuses is refused when the workflow is read, before any job is planned, with the error that names `export.segmentation` and `route: per_fov`, and the same configuration with `route: per_fov` added plans exactly the jobs it planned at the base revision. Exact. | H |
| WF2 | The assignment figures | the same | Files | On the route after fusion `export_sample` draws no `expr/{fovID}/assignment.png` again; on the route per FOV it does, as W-330 specifies. Exact. | H |

### What the default tier covers and what it does not

The routine tier covers every check except CR5. Not covered by any check: a learned
segmentation method on a fused image (no model is called; the conversion of its settings
is checked by SG2), a whole sample, a real sample, the GPU, and any accuracy. Whether the
route after fusion gives the right cells is not a question these checks answer; they
answer whether it gives each cell of its segmentation once, whole and with the molecules
of every FOV.

### Resource plan

| Fixture | Largest array | Tier |
| --- | --- | --- |
| `fused_pair` | 3 × 24 × 24 per FOV; fused projection 1 × 24 × 40 | routine |
| `fused_pair_seams` | 3 × 24 × 24 per FOV; fused projection 1 × 24 × 40 (W-401's computation of its outcomes: 1.23 s, 169,428 KiB) | routine |
| `fused_quad` | 3 × 40 × 40 per FOV; fused projection about 1 × 71 × 71 | routine |
| `scene_plane` | 1 × 36 × 36 per FOV; sample grid 1 × 63 × 63 | routine |
| `scene_plane_small` | 1 × 96 × 96 per FOV, 3 × 3 FOVs | `slow` |

One CPU and one thread per test; peak RSS below the 4 GiB stop target (the grounding calls
of W-401 on `fused_pair`, fusion, segmentation and two `Dataset.export` calls included,
peaked at 434,276 KiB in 8.2 s, interpreter and imports included). Each new test is
timed with `--durations=0`; a test that takes 5 s or more is marked `slow`.

## Real examples

None is part of this design. When the route is implemented, the operator may propose a
real example under its own authorization (real data, the GPU): both routes on the cell
culture's 2 × 2 block of W-382 and W-376, compared with `compare_routes`. It would report,
not decide, and would not be an accuracy.

## Implementation order

The checks above are assigned to implementation issues in the W-401 worker notes, each
naming the checks it must pass.

## Implemented checks

One line per implemented check: its identifier and the test that implements it, under
`src/python/test/`.

| # | Test |
| --- | --- |
| SO3 | `test_sample_segmentation_cell_key.py::test_so3_the_column_on_the_route_per_fov`, with the tests `test_so3_*` of the same file for the saved and reloaded column, a folder saved before the column, and the readers that link a molecule to its cell by the cell key |
| RS2 | `test_sample_segmentation_segment.py::test_rs2_a_stack_and_a_sum_are_refused` |
| RS3 | `test_sample_segmentation_segment.py::test_rs3_the_default_without_settings` |
| RS4 | `test_sample_segmentation_segment.py::test_rs4_check_7`, with the tests `test_rs4_*` of the same file for the refusals at construction, an unknown method key of the workflow block, the order of check 7 and a valid configuration |
| SG1 | `test_sample_segmentation_segment.py::test_sg1_the_input_is_the_fused_projection`, with `test_sg1_channels_in_the_run_s_role_order` |
| SG2 | `test_sample_segmentation_segment.py::test_sg2_lengths_in_pixels`, with `test_sg2_the_table_of_pixel_fields_equals_the_registered_methods`; the assignment's expansion distance (the last row of the LS2 table): `test_sample_segmentation_route.py::test_sg2_the_assignment_expansion_at_factor_k` and `test_sg2_the_assignment_expansion_in_the_export` |
| SG3 | `test_sample_segmentation_segment.py::test_sg3_label_files_and_the_section`, with the tests `test_sg3_*` of the same file for `load_sample_segmentation`'s checks, a second call and a folder without a fusion |
| SG6 | `test_sample_segmentation_segment.py::test_sg6_import_runs` |
| RS1 | `test_sample_segmentation_route.py::test_rs1_defaults_and_the_recorded_route` |
| SG4 | `test_sample_segmentation_route.py::test_sg4_reuse`, with `test_sg4_reuse_refused_before_anything_is_written`, `test_sg4_reuse_refused_for_tables_of_another_segmentation`, `test_sg4_a_changed_fused_input` and `test_sg4_reuse_with_changed_assignment_settings` |
| SG5 | `test_sample_segmentation_route.py::test_sg5_fuse_cannot_run`, with `test_sg5_the_method_backend_cannot_run`, `test_sg5_no_included_fov` (the parts called alone; `Dataset.export` refuses such a placement by `assemble`'s own check) and `test_sg5_a_method_that_raises` |
| SA1 | `test_sample_segmentation_route.py::test_sa1_the_conversion` |
| SA2 | `test_sample_segmentation_route.py::test_sa2_mapping_back` |
| SA3 | `test_sample_segmentation_route.py::test_sa3_each_cell_once_and_whole`, with `test_sa3_seam_cells_on_both_routes` |
| SA4 | `test_sample_segmentation_route.py::test_sa4_once_and_whole_with_a_method` |
| SA5 | `test_sample_segmentation_route.py::test_sa5_assigned_where_it_lies` |
| SA6 | `test_sample_segmentation_route.py::test_sa6_fov_id_of_a_cell` |
| SA7 | `test_sample_segmentation_route.py::test_sa7_identities` |
| SA8 | `test_sample_segmentation_route.py::test_sa8_molecules_keep_their_rule` |
| SO1 | `test_sample_segmentation_route.py::test_so1_same_columns_keys_and_files`, with `test_so1_the_saved_territories_reload_and_save_again` |
| SO2 | `test_sample_segmentation_readers.py::test_so2_readers`, with `test_so2_check_export_fails_t1_on_a_fused_input_fused_again`, `test_so2_check_export_fails_t5_on_a_label_file_the_assembly_does_not_record`, `test_so2_check_export_after_a_reuse` and `test_so2_lookup_source_of_identifiers_at_and_above_2_31` |
| SO4 | `test_sample_segmentation_readers.py::test_so4_the_label_export_on_the_route_after_fusion`, with `test_so4_with_the_expanded_territories` and `test_so4_l6_the_id_bound` |
| CR1 | `test_sample_segmentation_compare.py::test_cr1_two_routes_from_one_truth` (also on `fused_pair` at `k = 2`), with `test_cr1_the_placement_at_factor_2` |
| CR2 | `test_sample_segmentation_compare.py::test_cr2_the_report_decides_nothing` |
| CR3 | `test_sample_segmentation_compare.py::test_cr3_refusals`, with `test_cr3_a_threshold_outside_0_1_is_refused` |
| CR4 | `test_sample_segmentation_compare.py::test_cr4_determinism` |
| CR5 | `test_sample_segmentation_compare.py::test_cr5_the_whole_of_a_larger_scene` (`slow`) |
| CR, the measure | `test_sample_segmentation_compare.py::test_cr_the_measure_is_the_duplicate_rule_s`: the share of `compare_routes` equals the duplicate rule's for the same two territories; both take it from `starfinder.assembly._duplicates.share` |
| RS5 | `test_sample_segmentation_workflow.py::test_rs5_configuration_schema`, with `test_rs5_a_run_s_method_keys_equal_the_config_fields`, `test_rs5_every_init_field_of_a_method_config_translates`, `test_rs5_route_configs_validate_and_translate`, `test_rs5_invalid_route_keys_fail_the_schema` and `test_rs5_the_example_configuration_of_each_route` |
| WF1 | `test_sample_segmentation_workflow.py::test_wf1_the_rule_s_inputs_by_route`, with `test_wf1_the_inputs_follow_reference_image_and_the_named_images` and `test_wf1_existing_configurations_without_the_new_keys` and `test_wf1_a_configuration_rd1_refuses_plans_its_earlier_jobs_with_route_per_fov`; the Snakemake dry run (`-n`, starfinder-v9 Snakemake) is recorded in the W-430 worker notes |
| WF2 | `test_sample_segmentation_workflow.py::test_wf2_the_assignment_figures` |
