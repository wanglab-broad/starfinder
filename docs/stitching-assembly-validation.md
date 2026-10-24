# §2.10 validation design: placement, assembly and sample outputs

Status: Accepted (W-329, 2026-10-07, at 9033d46)

This page is the task-group-6 validation design for the whole of §2.10 (for the route after
fusion, see {doc}`sample-segmentation-validation`): the engineering
checks of placement ({doc}`stitching-contract`, {doc}`stitching-algorithms`), of assembly
({doc}`assembly-contract`, {doc}`assembly-algorithms`) and of the export entry with its
fused image, label export and SpatialData export ({doc}`sample-output-contract`), the
fixtures they run on, including the synthetic sample scene of {doc}`synthetic-sample-scene`,
and the plan for the whole-sample real examples. It builds on those six pages and
redefines nothing in them: every rule, column, status, identity, tolerance and file a check
names is used as those pages state it. Where a check needs something they do not give, the
W-357 worker notes say so. The engineering checks are implemented, and the section
"Implemented checks (W-373)" names the tests of each row.

Revised on 2026-10-06 after the W-329 review (comment "Approval record — 2026-10-06") and
the amended pages of W-355 and W-356: the checks of the former `Dataset.assemble` and
`Dataset.sample_outputs` methods, of `write_sample_outputs`, of the folders
`assembly/<sample>/` and `fused/<sample>/` with their records `assembly.json`, `fused.json`
and `labels.json`, and of `check_fused` and `check_spatialdata_export` became checks of the
one `Dataset.export`, of `export/<sample>/` with its one `export.json`, and of `check_export`;
`counts_anndata` became `write_h5ad`; the fused-image checks no longer test a fusion of
Starfinder's own (the overlap rule V1 and exact equality with an oracle of its voxel values)
but multiview-stitcher's `fuse` against the conditions and tolerances of the revised
contract; the label export moved into the first delivery; and the checks that read the LN
records or W-324's Fiji outputs moved from the test tiers to the real-example plan.

Revised again on 2026-10-06 after the second round of the W-329 review (comment
"Second-round outcome — 2026-10-06") and the pages W-358 and W-359 amended: the label
checks test the retained-cell rule of {doc}`sample-output-contract` ("Assembled label
export") in place of owner masks, with W-359's two fixtures; a fused-image check covers a
dataset that sets `dimensionality: stack` alone and the middle plane; the checks of
`placement_from_table` cover origins supplied apart from the nominal positions, so the
truth placement of a scene with Z jitter now goes through `assemble`; the SpatialData
metadata check gains its refusal case; and the real examples write LN sample1 as a stack
alone from both placements, draw every LN figure from its middle plane and record how
often the flags `conflict`, `unclaimed` and `fov_edge` occur.

## What this design is and is not

* **Engineering validation.** Each check is a known-answer test with a pass/fail
  tolerance fixed before any implementation runs, as in §2.7 to §2.9
  ({doc}`assignment-algorithms`, "Engineering validation design"). A tolerance is not
  adjusted during a run; a correct implementation that cannot meet one goes to Jiahao.
* **Hand-specified fixtures test engineering behavior, not cellular truth.** Hand-built
  placements, label boxes and single bright voxels show that a rule is implemented as
  written. The synthetic sample scene adds placement truth and the knowledge of which FOVs
  see each generated cell and molecule; its cells are power-diagram territories around
  ellipsoids, not a tissue model ({doc}`synthetic-sample-scene`, "What this addition is
  not"), so a result on it is a statement about the rules, not about cells.
* **The fused values are the library's.** multiview-stitcher's `fuse` computes every fused
  and label voxel ({doc}`sample-output-contract`, "Fusion by multiview-stitcher"). The
  checks test what the contract states about them: the grid, the geometry recorded in each
  file, the values where one FOV covers, the range in overlaps, the uncovered voxels, the
  position of a bright voxel, the absence of new label values, and the equality of a file
  with a direct call of the library. No check recomputes a blended value.
* **No backend ranking, no sweep, no accuracy claim.** The two binnings of the
  `multiview_stitcher` checks (MV1, MV2) are the contract's starting configuration and the
  alternative setting the contract names; each has its own bound and they are not compared.
  No check prefers one backend, sets a default from comparative data or calls a placement
  accurate. Real data has no placement truth: agreement between BigStitcher and
  multiview-stitcher on LN is monitored in the real examples and gates nothing.
* **E07 keeps the systematic comparison.** Comparing the backends over datasets, settings
  and conditions, with accuracy, robustness and cost, is the work of E07 (W-124). This page
  takes none of it over.
* **No real-data record in the repository.** Default-tier and `slow` tests use fixtures
  built in the test and read no file outside the repository. The comparisons with the LN
  `dataset.xml`, stage file, grid and tile configuration and with W-324's parity and Fiji
  outputs read those files on the share, in the real-example issue ("Checks against the
  records on the share").

## Evidence for the tolerances

Every measured number is cited by path; nothing was run for this page.

* W-354, run directory
  `/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/runs/W-354/20261006T175525Z-174cf769`
  (found with `timeout 30 find …/runs/W-354 -maxdepth 2 -name fusion-manifest.json`):
  `worker-notes.md` sections 2 to 7, `tables/position_summary.csv`,
  `tables/overlap_dtype.csv`, `tables/reduced.csv`, `tables/projection_fixture.csv`,
  `tables/labels_summary.csv`, `tables/labels_new_values.csv`, `readback/readback.csv`,
  `readback/readback.json`, `readback/spatialdata_read.json`, `tables/runs.csv` and
  `scripts/fixtures.py` (the recipes of the fixtures `overlap_pair`, `position_grid`,
  `checker_pair` and `label_scene` below), and `tables/ln_two_placements_grids.csv` for the
  real examples.
* W-359, run directory
  `/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/runs/W-359/20261007T020704Z-40025a5f`
  (found with `timeout 30 find …/runs/W-359 -maxdepth 2 -name worker-notes.md`):
  `worker-notes.md` section 3 (the fixtures and the per-cell measure),
  `tables/w354_summary.csv`, `tables/two_fov_summary.csv`, `tables/two_fov_cells.csv`,
  `tables/cases.csv`, `tables/case_placeholder_summary.csv` and
  `scripts/labels_retained.py` (the recipe of the fixture `label_pair` below, `TWO_FOV_CELLS`,
  `TWO_FOV_SHAPE` and `TWO_FOV_ORIGINS`, and the per-cell measure, function `per_cell`).
* W-324, run directory
  `/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/runs/W-324/20261006T063540Z-5cb2f421`
  (found with `timeout 30 find …/runs/W-324 -maxdepth 2 -name stitching-manifest.json`):
  `notes.md` sections 2.3 to 2.6, 3 and 5, `tables/fixtures_summary.csv`,
  `tables/comparison_summary.csv`, `tables/pairs_between_runs.csv`, `scripts/fixture_data.py`
  (the cut-fixture recipe) and `scripts/parity_fixtures.py` (the values of the hand-built
  stage files and `dataset.xml` files of the fixture `hand_records`).
* W-323, `/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/runs/W-323/`:
  `20261005T1930Z-fiji-baseline/inputs/` and `20261005T2110Z-input-inventory/inputs.md`, read
  only by the real examples.
* W-326 `tables/option_effects.json`, in the run directory that {doc}`assembly-algorithms`
  names.

### Tolerances and where they come from

| Quantity | Tolerance | Source or justification |
| --- | --- | --- |
| Statuses, reasons, flags, keys, counts, row sets, ID sets, file lists, hashes, dtypes, shapes | Exact | Integers and strings; every expected value follows from the fixture by hand or by a formula in the test. |
| Positions computed by two different operation orders (an origin plus an index times a spacing, against `placed_metadata(…).index_to_world`; the fusion grid and its slab; the geometry Starfinder writes into an OME-TIFF and the label grid's formula) | 1e-9 µm absolute | Positions below 10⁴ µm, a few float64 operations each with relative error at most 2⁻⁵³, so the difference stays below 1e-11 µm; 1e-9 µm is 1e-8 of the smallest spacing in use (0.0946 µm). {doc}`sample-output-contract` uses the same bound for the OME-TIFF ("Sample position to image position", condition 1) and the label grid (L5). |
| The same quantity computed by the same operations | Bitwise equal | Float64 results of one formula are deterministic on one host. |
| Geometry of the library's OME-Zarr (per-level `scale` and `translation`) | 1e-6 µm | {doc}`sample-output-contract`, condition 1 and the OME-Zarr layout checks; W-354 `readback/readback.csv`, `origin_matches_grid`. |
| Equivalent radius of a `cell_centroids` circle | 1e-12 relative | Two float64 evaluations of one formula (a cube or square root of a product) that may differ in operation order; a few rounding steps of 2⁻⁵³ each. |
| Fused values where one FOV covers and its voxel centres coincide with the output's | Exact | {doc}`sample-output-contract`, "What a fused image holds" and F4: the input values unchanged. |
| Fused values in an overlap | Between the two inputs, inclusive; an integer value never above the truncation of the float mean | F4; W-354 `tables/overlap_dtype.csv` (`overlap_min`, `overlap_max`). |
| Uncovered voxels | Exactly 0, coverage decided with a margin of 1e-6 fusion-grid voxel | F3; W-354 `tables/overlap_dtype.csv` (`uncovered_*`, `uncovered_is_nan`); assumption A13. |
| A bright voxel at scale 1 | Maximum on the output voxel nearest its true position; intensity centroid within 0.001 µm | Condition 4; W-354 `tables/position_summary.csv`: at most 0.0008183085523003642 µm. |
| A bright voxel at `k = 2` and `k = 4` (`level1`) | A local maximum above 0 and at most `floor(1000/k²)`, within `((k − 1)/(2k) + ½)·S_a` of the true position on Y and on X | Condition 4; W-354: 9 of 9 at both factors, largest distances 0.2904285101109904 µm (bound 0.291597) and 0.5848581960387487 µm (bound 0.680393). |
| A file against a direct call of `fusion.fuse` with the recorded inputs and arguments | Exactly equal | F6 and assumption A6. |
| Labels drawn by retained cells on W-359's two fixtures, at scale 1 and 0.25 | Exact: every drawn voxel is one its FOV's territory holds for that cell, and the per-cell counts (whole, incomplete, two or more IDs, absent, values in no input) and `ids_input`, `ids_output`, `absent_ids` equal W-359's | L3 and L4; W-359 `tables/w354_summary.csv` and `tables/two_fov_summary.csv` (rows `retained_cells`), `tables/two_fov_cells.csv`. The fixtures are deterministic and the rule involves no blending, so a rebuilt fixture gives the same counts. |
| SpatialData element transforms | 1e-9 | S6. |
| `multiview_stitcher` on the cut fixture, starting configuration (binning 4) | Max absolute error at most binning / 2 = 2 voxels per axis, after removing the mean over the FOVs | W-324 `fixtures_summary.csv`: largest 1.0946 voxels (y, synthetic_3x3, binning 4 with and without the 0.7 filter); notes section 5 proposes "binning / 2". |
| `multiview_stitcher` on the cut fixture, binning 2 | At most 1 voxel per axis | W-324: below 1.8e-13 on both cut fixtures. |
| `multiview_stitcher` on the synthetic scene with fractional jitter, binning 1 | At most 0.5 voxel per axis, against the truth in gauge G1 | binning / 2 as above. A fractional true offset adds a rounding of at most binning / (2 × `upsample_factor`) = 0.25 voxel per link (library default `upsample_factor` 2 in 3D, W-324 notes section 3), which W-324's integer offsets did not exercise; W-324 measured binning 1 at 0.19 voxel at most (real_2x2, z). Binning 1 is used because the scene's overlaps after the jitter can be as narrow as 15 voxels: 3.75 binned pixels at binning 4 and 7.5 at binning 2, below the 14 binned pixels of W-324's narrowest measured overlap (synthetic_3x3 at binning 4), and 15 at binning 1. |
| `multiview_stitcher` on the W-324 hand-built cases (two tiles, Z = 1, anisotropic spacing, no-signal centre, disconnected tile) | binning / 2 = 2 voxels per axis for the FOVs of the reference component; statuses exact | W-324 notes section 2.6: at most 0.50 px (two tiles), 0.2500118 px (Z = 1), 0.75 voxel (anisotropic), 1.6108 and 1.2501 px for the other tiles of the no-signal cases. |
| Repetitions and thread counts | Identical translations | W-324 `pairs_between_runs.csv` and `fixtures/rerun_vs_attempt1.csv`: largest difference 0.0. |
| Review defaults (`max_correction_fraction` 0.5, `max_residual_um` 1.0) | Tested at their boundary; not tuned | {doc}`stitching-algorithms`, "Placement-quality checks". For scale, W-324 notes section 5: LN grid-start corrections reach 44.5 px, 0.30 of the 149.6 px overlap, and pass; stage-start corrections reach 149.9 px, 1.0 of it, and are flagged. |
| Real examples only: BigStitcher's result on the W-324 cut fixtures, read through the import | Max absolute error at most 1.05 voxels per axis | W-324 notes section 5: largest 1.0000000020 (z, synthetic_3x3), 0.6896 (z, real_2x2); a bound of exactly 1 voxel would fail by 2e-9. |
| Real examples only: the LN import against the W-324 parity outputs | Exact (float64 equality, byte-equal CSV) | {doc}`stitching-contract`, "What the import of the LN `dataset.xml` must reproduce". |
| Real examples only: agreement of the two backends on LN | Monitored, never gating: residual after the linear trend at most 3.5 px in x, 1.5 px in y, 1.5 slices in z; trend in x at most 2.5 px per column; absolute difference at most 10 px in x, 2.5 px in y, 3.5 slices in z | W-324 notes section 5, regression bounds over the four grid-start configurations; W-324 choice 2 recommends monitoring them, not gating. |
| Memory of the export parts | No bound is tested; the peak RSS of each part is recorded | {doc}`sample-output-contract`, "Memory", promises no bounded memory for a whole-sample stack; lazy loading (A1) is verified on LN in the real examples. |

## Fixtures

Every fixture is generated in the test from the values stated here. No test reads the
network share, a W-323, W-324 or W-354 run directory, real images or Fiji, and no record of
those run directories is copied into `src/python/test/data/`. Every fixture stays inside
the test bounds of the §2.10 resource exceptions ("Resource plan"); the ones above the
routine bound are marked `slow`. Unit spacing means 1 µm on each axis.

| Fixture | Construction | Seeds | Needs |
| --- | --- | --- | --- |
| `stub_grid` | 2 rows × 3 columns of FOV grids of 4×16×24 voxels (Z, Y, X; not square), spacing (0.25, 0.25, 0.25) µm (dyadic, so the review boundaries are exact), identity direction, overlap 0.25; zero images; positions from `positions_from_grid` (`column_major_down_right`). A test-only method, registered in `STITCHING_METHODS` for the test, returns supplied translations (µm), components, links with quality and kept flags, and diagnostics. | none | hand-built |
| `stub_variants` | `stub_grid` with: fractional translations (multiples of 2⁻⁶ µm, so the expected sums are exact); spacing (1, 2, 2) µm and (0.346255, 0.194398, 0.194398) µm; Z = 1 grids from YX inputs; one FOV of 4×16×20 beside 4×16×24 (unequal shapes); the unsnapped stage positions of `hand_records`' `maf_2x3_jitter` (irregular); the five positions of its `maf_3x2_gap` (an L-shaped union); one FOV whose links all have NaN quality; one FOV whose nominal x is moved by two tile widths (no overlap); two components of two FOVs each (a tie for the reference component); corrections at exactly `0.5 × w` and `0.5 × w + 2⁻²⁰` µm, residuals at exactly 1.0 and `1.0 + 2⁻²⁰` µm. | none | hand-built |
| `hand_records` | Written by the test as text from the constant values of W-324 `scripts/parity_fixtures.py`, which the test states as literals (nothing is copied): `xml_2x2_blank` (2 × 2 grid, tile 1 blank, fractional translations of both signs) and `xml_3x2_positive` (3 columns × 2 rows, every translation positive), each a `dataset.xml` with its `grid.csv`; `maf_3x2_gap` (3 × 2 stage positions in metres, one missing) and `maf_2x3_jitter` (2 × 3, x jitter up to 1.1 µm). The expected legacy outputs are computed in the test by the frozen helpers of `test_workflow_stitching_golden.py`: `legacy_tile_config` on each `dataset.xml` and `run_stitching_preparation` on each stage file. The W-324 `dataset.xml` files hold no `ViewSetup`, which the import needs to map a view to its tile ({doc}`stitching-contract`, "BigStitcher"), so for the import the test derives `hand_records_setups`: each file's text with one `ViewSetup` per setup `j` inside `<SequenceDescription><ViewSetups>`, inserted after the `SpimData` element: `<id>` and `<name>` `j`, `<attributes>` with illumination 0, channel 0, `tile` `j` and angle 0, `<size>` and `<voxelSize>` of the fixture's tiles (the import does not read them), and the matching `<Attributes name="tile">` list; the tile of setup `j` is row `j` of the file's `grid.csv`, the order in which BigStitcher reads `tile_{j}.tif`. The import gets the view map as its `view_map` argument, built in the test from that `grid.csv` (`view` `j`, `grid_column`, `grid_row`, `fov_id` for a nonzero `id`, `kind` `fov` or `padding` for `id` 0). The frozen helpers still read the files without `ViewSetups`. | none | hand-built |
| `xml_links` | A `dataset.xml` written in the test for a 2 × 2 grid of 4×16×24 tiles: `ViewSetups` with tile attributes, `ViewRegistrations` (`Stitching Transform`, `Translation to Regular Grid` with fractional values, `calibration` with `c_z` 1.7894736842105263), and one `PairwiseResult` per neighbour pair (four side and two diagonal pairs) with shifts consistent with the translations and correlations 0.80 to 0.95; the view map passed as `view_map`, built from its 2 × 2 grid; copies with the `ViewRegistration` elements in permuted order, with a 1e-6 rad rotation in one view, with `c_x ≠ c_y`, without the `PairwiseResult` entries of view 3, and without view 3's `Stitching Transform`. | none | hand-built |
| `golden_stitching` | The stage file and `dataset.xml` of `test_workflow_stitching_golden.py` (`stage_file`, `dataset_xml`, `GRID_CSV`, `TILE_CONFIG`), imported from that module, which is not edited. Its `dataset_xml()` holds only `ViewRegistrations` (nine setups, setup 4 the blank cell), so every row that imports it uses `golden_stitching_setups`, derived in the test: the text of `dataset_xml()` with one `ViewSetup` per setup `j` inside `<SequenceDescription><ViewSetups>`, inserted after the `SpimData` element: `<id>` and `<name>` `j`, `<attributes>` with illumination 0, channel 0, `tile` `j` and angle 0, `<size>` and `<voxelSize>` of the fixture's tiles (the import does not read them), and the matching `<Attributes name="tile">` list (`<size>` 100 80 1, `TILE_COL` and `TILE_ROW`; `<voxelSize>` 0.19 0.19 0.34 µm, the module's `VOXEL_SIZE_XY` and `VOXEL_SIZE_Z`). The tile of setup `j` is row `j` of `GRID_CSV`. The import gets the view map as its `view_map` argument, built in the test from `GRID_CSV` (`view` `j`, `grid_column`, `grid_row`, `fov_id` `Position<id>` for a nonzero `id`, `kind` `fov`, and `kind` `padding` without `fov_id` for setup 4), with the table of the eight acquired FOVs and their grids of 1×80×100 at the module's voxel sizes. `legacy_tile_config` and `TILE_CONFIG` are used unchanged. | none | hand-built |
| `cut_synthetic` | The W-324 recipe rebuilt in the test: scene 16×256×256 (seed 20261006: texture and 70 Gaussian nuclei), 3×3 tiles of 14×120×120, nominal step 64 px, true origin nominal plus integer jitter (seed 324), noise N(0, 3) per tile (seed 1000 + tile), uint16, spacing (0.346255, 0.194398, 0.194398) µm. Derived cases as W-324 `scripts/fixtures.py` builds them: `two_tiles`, `z1_3d`, `plane_2d`, `spacing_1_2_2`, `no_signal_zero`, `no_signal_noise` (centre noise seed 1050), `disconnected`; and `non_square`: the same scene cut into 3×3 tiles of 14×104×120 at nominal steps 56 (y) and 64 (x), with the same jitter draws. | 20261006, 324, 1000 + tile, 1050 | `multiview-stitcher` extra; `slow` |
| `assembly_golden` | `build_sample` of `test_workflow_assembly_golden.py`: three FOVs of 8×48×64 on a 2 × 2 grid with a blank cell, the fractional `STITCHING` translations, nine scene cells, 24 molecules, 3D and 2D; its placement is a `PlacementResult` constructed in the test, because the golden `dataset.xml` holds no `PairwiseResult` and its import would leave no FOV `image_supported`: every FOV `image_supported`, nominal positions of its grid (`nominal_z` 0 for every FOV, one layer), origins the float64 sums of the `STITCHING` translations and the grid steps as the import arithmetic gives them (FOV2 at y 37.4 and x 0.7, FOV3 at y −0.3 and x 55.4), `record["inputs"]` holding each FOV's grid, review `not_needed`. | none | hand-built |
| `own_cases` | Two FOVs of 4×16×16, spacing 1 µm, origins (0, 0, 0) and (0, 0, 12) µm (centres at x 7.5 and 19.5, boundary x 13.5), with box cells and molecules placed, by hand, so that every assembly status, every reason, every flag and one exact tie at x 13.5 occur at least once, including a retained `excluded_no_nucleus` cell and an expanded FOV; the expected status, reason, owner and flags are written in the fixture; its placement is `placement_from_table` of those origins (`nominal_z` 0), approved. | none | hand-built |
| `edge_points` | For one FOV of 4×16×24 at a fractional origin in Y and X (Z origin 0): points at each footprint bound `origin − s/2` and `origin + (n − ½)·s`, 1e-9 µm inside and outside each; for two FOVs of one shape at X origins 0 and 10 µm, a point exactly on the bisector (float-exact). | none | hand-built |
| `empty_cases` | `assembly_golden` variants: a FOV without molecules, a FOV without cells, a FOV whose molecules miss every cell, a sample whose cells are all `excluded_no_nucleus` (no counted cell), a `molecules_only` sample, a `cells` sample whose FOVs have no cell row, a sample whose FOVs have no molecule, a placement whose FOVs are all `failed`, and one whose FOVs are all `placeholder` (approved); and `no_fov_assembly`, an `AssemblyResult` built by hand on the all-`failed` placement (every row `fov_not_placed`, which `__post_init__` accepts), for the export cases without an included FOV, because `assemble` refuses a placement with no owning FOV. | none | hand-built |
| `status_cases` | `assembly_golden` with FOV2 `placeholder`, once without an input and once with an input holding box cells and molecules (one cell whose ownership centroid lies where no owning FOV covers it); FOV3 `missing`, then FOV3 `failed` with an input; review `required` without and with an approval. | none | hand-built |
| `placeholder_grid` | 3 × 3 FOVs of 4×16×16, unit spacing, nominal step 12 µm (overlap 4 µm), as a `PlacementResult` built in the test with every FOV `image_supported` except the corner FOV (row 0, column 0) and the edge FOV (row 2, column 1), which are `placeholder` (the BigStitcher import with `pad_missing=True` gives the same statuses), `nominal_z` 0 for every FOV, approved; one box cell per owning FOV whose footprint overlaps a placeholder's footprint, with its ownership centroid inside that overlap (with the placeholders at (row 0, column 0) and (row 2, column 1), the six FOVs (0, 1), (1, 0), (1, 1), (1, 2), (2, 0) and (2, 2)); a raster of points every 0.25 µm over the union of the footprints. | none | hand-built |
| `scene_tiny` | `sample_scene_preset("sample-tiny-v1")` ({doc}`synthetic-sample-scene`, "Presets and resources"): 2 × 2 FOVs of 5×36×36, fractional J1 jitter in Z, Y and X, seed 42, split `development`. Its truth placement is `placement_from_table` of {doc}`synthetic-sample-scene` ("The path that needs no segmenter"): the table `positions_from_table` reads from `positions.csv` (the nominal grid, `nominal_z` 0 for every FOV), `origins` mapping each `fov_id` to `(true_origin_z, true_origin_y, true_origin_x)` of its row of `placement_truth.csv`, and each FOV's reference grid, then `approve_placement`. Every FOV is then `metadata_only` at its true origin, Z jitter included, with `correction_*` equal to `true_correction_*`; its FOVs share one `nominal_z`, so check 8 of {doc}`assembly-contract` ("Checks the entry applies") passes and the placement is the input of `assemble` in EE2 to EE5. | 42 | generator addition |
| `scene_tiny_flat` | The same configuration with `jitter_max_zyx` set to `(0.0, 1.5, 1.5)` by `dataclasses.replace` (no preset added): J1 draws z, y and x in that order from each FOV's stream, so the Y and X draws are those of `scene_tiny` and every true Z offset is exactly 0; its truth placement is built as for `scene_tiny`. `scene_tiny_flat_inward`: every FOV moved by the jitter bound towards the grid centre in Y and X (a supplied jitter, Z 0), the least-covered case of the page in Y and X. Beside `scene_tiny`, `scene_tiny_flat` is the case with the same Y and X origins and no Z jitter. | 42 | generator addition |
| `scene_plane` | The single-plane case of {doc}`synthetic-sample-scene` ("Tests the implementation keeps and adds"): `dataclasses.replace` of the `sample-tiny-v1` configuration with `fov_shape_zyx` and the template's `shape_zyx` set to `(1, 36, 36)` and `jitter_max_zyx` to `(0.0, 1.5, 1.5)`, everything else unchanged; no preset is added. | 42 | generator addition |
| `scene_small` | `sample_scene_preset("sample-small-v1")`: 3 × 3 FOVs of 14×96×96 at the LN calibration, overlap 0.25, jitter bound (0.8, 4.5, 4.5) voxels, seed 42. | 42 | generator addition; `slow` |
| `quad` | Four FOVs of 4×32×32 uint16 with values `default_rng(2102).integers(0, 4096)` drawn in table order, unit spacing, at (Z, Y, X) origins (0, 0, 0), (0, 26.3, 0.4), (0.5, 0.2, 26.7) and (0.3, 26.1, 26.9) µm: fractional offsets in every axis, a four-FOV overlap; fusion grid 4 × 58 × 58. | 2102 | `multiview-stitcher` extra |
| `zstack` | Two FOVs of 4×8×8 uint16, unit spacing, values `1 + 64·z + 8·y + x + 100·f` for FOV `f` (at most 356), at origins (0, 0, 0) and (0, 0, 6.25) (equal Z; grid 4 × 8 × 14) and (0, 0, 0) and (1.4, 0, 6.25) (different Z; grid 5 × 8 × 14); two one-plane FOVs of 1×8×8 at (0, 0, 0) and (0, 0, 6.25); a float32 copy. | none | `multiview-stitcher` extra |
| `plane_pair` | The example of F3 of {doc}`sample-output-contract`: two one-plane FOVs of 1×8×8 uint8 with constant values 50 and 120, unit spacing, at origins (0, 0, 0) and (2, 0, 10): one plane, slab `[−0.5, 2.5)` (`S_z` 3, `O_z` 1), fusion grid 1 × 8 × 18 whose voxels x 8 and 9 lie outside both FOVs. | none | `multiview-stitcher` extra |
| `checker_pair` | W-354's reduced fixture rebuilt (`scripts/fixtures.py`, `reduced`): two FOVs with a one-voxel checkerboard of 0 and 200 (uint8) in Y and X, spacing (0.5, 0.2, 0.2) µm, FOV A at the origin and FOV B at (0, 10.4, 40.7) px, as 4×64×64 and as one-plane 1×64×64. | none | `multiview-stitcher` extra |
| `overlap_pair` | W-354's overlap fixture rebuilt (`overlap`): two constant 2D FOVs of 32×256 px at 0.194398 µm, the second at (8, 200) px, overlapping over 56 px in X, with two uncovered corners of the union grid; values 100 and 203 (uint8), 1000 and 3001 (uint16), 0.25 and 0.8 (float32). The union grid, 40 × 456, is above the routine bound. | none | `multiview-stitcher` extra; `slow` |
| `position_grid` | W-354's position fixture rebuilt (`position`): 3 × 3 FOVs of 16×128×128 uint16 at spacing (0.346255, 0.194398, 0.194398) µm, origins at a step of 115.2 px in Y and X jittered by up to 3 px and in Z by up to 2 slices, one voxel of 1000 per FOV in the part only that FOV covers, all drawn from one generator in W-354's order. For the store row SD6 only: `placement_from_table` of the nominal positions of the 3 × 3 grid (`nominal_z` 0) with these origins as `origins`, approved, and a `molecules_only` assembly on it with one molecule at each bright voxel; its one `nominal_z` passes check 8 of `assemble`, whatever the Z jitter of the origins. | 354 | `multiview-stitcher` extra; `slow` |
| `fuse_band` | Six FOVs of 16×256×256 uint8 in one column, overlap 0.1 (Y origins `230.4·r` µm), unit spacing, values `(z + y + x) mod 251`: fusion grid 16 × 1408 × 256, three 512-row bands of OME-TIFF tiles, two OME-Zarr projection chunks of `(1, 1, 1024, 1024)` and three stack chunks of `(1, 1, 16, 512, 512)`. For the store rows (SD7): a `molecules_only` assembly with one molecule at each FOV's centre, on `placement_from_table` of the same origins (`nominal_z` 0), approved. | none | `multiview-stitcher` extra; `slow`; inside the `slow` bound of a 3 × 3 grid of 16×256×256 tiles |
| `label_quad` | Four owning FOVs of 1×40×40 with uint32 `plane` territories, unit spacing, at origins (0, 0, 0), (0, 30.25, 0.5), (0, 0.5, 30.25) and (0, 30.25, 30.25) µm: per FOV nine box cells of 6 × 6 px starting at rows and columns 4, 16 and 28, and one cell of 1 × 1 px at (34, 34) that sampling at `k = 4` can miss; in FOV 1 also a box cell of 4 × 4 px at rows 35 to 38 and columns 4 to 7, in the overlap with FOV 2, which FOV 2 owns; built into a `cells` assembly through `import_labels(…, relabel=False)`, `assign_molecules` with one molecule per cell, the §2.9 assignment folders saved and passed to `assemble` by path (so the `assembly` section records each territory file's path), `placement_from_table` of the origins (`nominal_z` 0), approved, and `assemble`. Fusion grid 1 × 70 × 70; label grid at 0.25: 18 × 18. Its volume variant: the same FOVs as 4×40×40 with `volume` territories (each box through all four planes), Z origins 0. | none | `multiview-stitcher` extra |
| `label_scene` | W-354's label fixture rebuilt (`label_scene`, `labels`): one 2D scene of discs, 3 × 3 FOVs of 112 × 112 cut at steps of 96 px (16 px overlap), placed at the cut positions plus a jitter of up to 0.4 px in Y and X (Z origins 0), spacing 0.194398 µm; per FOV the cut as its `plane` territory, assembled as `label_quad` is. The scene, 304 × 304, is above the routine bound. | 354, 355 | `multiview-stitcher` extra; `slow` |
| `label_pair` | W-359's two-FOV fixture rebuilt from its recipe (W-359 `scripts/labels_retained.py`, `TWO_FOV_CELLS`, `TWO_FOV_SHAPE`, `TWO_FOV_ORIGINS`): two one-plane FOVs of 1×64×96, spacing 0.194398 µm, FOV A at (0, 0) px and FOV B at (0, 64) px, overlapping over 32 px in X (O1 boundary at x 79.5 px, an exact tie going to A); 13 cells, each a disc drawn by each FOV that sees it (21 observations): cell 6 drawn by A at x 77 and by B at x 82, so both FOVs retain it; cell 7 drawn by A at x 83 and by B at x 76, so neither retains it; cell 5 symmetric about x 79.5 (the tie); cells 8 and 9 segmented with different sizes, so that A's cell 8 and B's cell 9, each retained, share one voxel; cell 10 at B's top border (35 of its 81 px imaged); cell 11 cut by B's left border and whole in A; the others alike in A only, in B only or in the overlap. Built into a `cells` assembly as `label_quad` is (`plane` territories through `import_labels(…, relabel=False)`, one molecule per cell, `placement_from_table` of the origins (0, 0, 0) and (0, 0, 64 × 0.194398) µm, approved, `assemble` with `conflicts=True`). Under the assembly contract's rules A retains cells 1, 3, 5, 6, 8, 11, 12 and B cells 2, 4, 6, 9, 10, 13; both observations of cell 7 are `unclaimed`; no observation is `conflict`, because neither view of cell 6 holds the other's ownership centroid; `fov_edge` is on B's cells 10 and 11 (these follow from the recipe by {doc}`assembly-contract`, "Retention" and "Flags", and the test writes them into the fixture). Variants: `label_pair_conflict`, B's view of cell 6 at x 80, so both views of cell 6 are retained and flagged `conflict`; `label_pair_empty`, every label 0. | none | `multiview-stitcher` extra |
| `label_pair_merged` | A variant of `label_pair` after the case "A cell merged by its owner" of the W-362 worker notes (section 7, `scripts/label_pair_overlap.py`): `label_pair`'s two one-plane FOVs of 1×64×96 at 0.194398 µm, A at (0, 0) px and B at (0, 64) px, holding two cells only: FOV A segments cells 31 and 32 apart, as discs of radius 3 px at (y 32, x 86) and (y 32, x 92), drawn in that order; FOV B draws the same two discs and labels both 32, one merged cell. Built into a `cells` assembly as `label_pair` is (`conflicts=True`). A's view of cell 31 has its ownership centroid at x 85.893 px, which B owns, and B's territory there holds B's retained cell 32, so A's row of 31 is excluded `not_owner` and not flagged `unclaimed`; A's row of 32 is excluded too, and B retains its merged 32 (these follow from the recipe by {doc}`assembly-contract`, "Retention" and "Flags"). | none | `multiview-stitcher` extra |
| `id_bound` | A stand-in for the entry's check 6: a sample cell table with one `cell_id` column (uint32) of 16,777,217 rows (64 MiB) and one of 16,777,216 rows (2^24); no image. | none | hand-built |
| `export_folder` | `assembly_golden` (3D) written as a dataset: the saved placement (`save_placement` of its constructed placement), the saved §2.9 assignment folders, the reference grids, and a `reference_stain` image per FOV (uint8, `(z + y + x) mod 251`, 8×48×64), as `Dataset.export` reads them. Variants: one FOV without its `reference_stain` file; the `multiview_stitcher` and `spatialdata` imports made to fail (monkeypatched); and, after a first `Dataset.export` has written and saved the assembly (once with the default `conflicts: true`, whose `assembly` section records each territory file's path and SHA-256, and once with `conflicts: false`, whose section records neither), one owning FOV's territory file removed, or replaced by a file with another SHA-256 (the territory files are inputs of the assembly, not files of the export folder). | none | `anndata`; `multiview-stitcher` and `spatialdata` extras where a part runs |
| `export_cases` | The assemblies of `assembly_golden` (3D, 2D, with an expansion), `empty_cases` and `status_cases`, with images fused by `Dataset.export` from `reference_stain` images as in `export_folder`, on each assembly's own placement, and, for the label element, the label export of their territories: one sample per row of the table "Empty populations" of {doc}`sample-output-contract`. The row "No included FOV" uses `no_fov_assembly` with the `fusion` section of `fuse_images` called alone on its placement (status `empty`), and `export_spatialdata` called alone, since `Dataset.export` refuses that placement in its assemble part. | none | `multiview-stitcher` and `spatialdata` extras |
| `wf_configs` | The repository's existing test configurations (legacy, without the new blocks), and copies with a `stitching`, an `assembly` and an `export` block. | none | hand-built |

Seeds: 324 (W-324 jitter), 20261006 (W-324 scene), 1000 + tile and 1050 (W-324 noise), 42
(the sample presets), 210 for the row permutation of AS14, 2102 for the values of `quad`,
and 354 and 355 (W-354's position and label fixtures); in the real examples only, 325 and
1100 + tile for W-324's real_2x2 cut fixture. No other seed and no other random number is
used. Fixtures marked "none" use no random number.

## Checks

"Needs" says what a check requires beyond the routine locked environment: **G** the
generator addition of {doc}`synthetic-sample-scene` (W-327); **H** nothing but
hand-built geometry; **A** the `anndata` extra; **MV** the `multiview-stitcher` extra;
**SD** the `spatialdata` extra; **S** `slow`. A test that needs an extra is skipped
without it, naming the extra ({doc}`sample-output-contract`, "Tests the implementation
keeps"); the batch that implements it installs the extra, and no such test is skipped
there.

### Placement (PL)

| # | Check | Fixture (seeds) | Metric | Pass/fail tolerance | Needs |
| --- | --- | --- | --- | --- | --- |
| PL1 | Converters | `hand_records` stage files; `golden_stitching`; supplied tables in µm, nm, mm (none) | `TilePositionTable` columns and record | Every `StageAxisMapping` of the eight sign and axis combinations moves stage +X and +Y where it names; the smallest y and x are 0 and z is 0; with `GridSnap(overlap=0.1, tile_shape_yx=(16, 24), spacing_yx=(0.25, 0.25))` on `golden_stitching`'s stage file and on the two stage files of `hand_records`, the cells equal the columns and rows that the frozen legacy helper `run_stitching_preparation` writes for the same file, including `maf_2x3_jitter`, whose last-bit tie the 0.1 µm rounding removes (a remaining tie goes to the smaller step); `positions_from_grid` puts FOV `k` at column `k // rows`, row `k % rows` with the step `(1 − overlap) × tile × spacing`, and on a 2 × 3 grid the orders `row_major_right_down` and the two serpentine forms put each FOV in the cell their names define; units convert within 1e-9 µm; `position_id` holds the Leica `PositionID` and is missing for a grid or a supplied table, `source` is `stage`, `grid` or `supplied` by converter, and `grid_column`, `grid_row` are set by `positions_from_grid` and a snapped stage file and missing otherwise; with `snap_to_grid=None` the stage positions are kept unsnapped; each failure of {doc}`stitching-algorithms` ("Converters") raises `ValueError` naming the item, and so do the parameter rules of its tables: `GridSnap` or `positions_from_grid` with `overlap` 1.0 or −0.1, a tile size of 0, a spacing of 0 or NaN, and `columns × rows` below the number of FOVs. Exact otherwise. A `StageAxisMapping` that names one image axis twice raises `ValueError`. The LN stage file is RX1. | H |
| PL2 | Neighbour pairs | `stub_grid`, `stub_variants` (irregular, unequal shapes, L shape); an 8 × 8 regular grid of 1496 × 1496 px grids at 0.194398 µm and overlap 0.1, built by `positions_from_grid` (grids only, no image) (none) | Pair set and overlap boxes | The pairs are those whose boxes intersect with positive area (and positive z length for Z > 1), diagonals included; touching boxes are not a pair at tolerance 0 and are at a positive tolerance; no pair depends on a row or column index (permuting the grid cells of a supplied table changes no pair); each pair lists first the FOV earlier in table order; a negative `tolerance_um` raises `ValueError`; the 8 × 8 grid has 210 pairs: 56 in rows, 56 in columns and 98 diagonal. Exact. The LN grid start (213 pairs) is RX2. | H |
| PL3 | Transform direction and composition | `stub_variants` (fractional); `golden_stitching_setups` with its view map; `xml_links` with permuted `ViewRegistration` order and with a rotation of 1e-6 rad (none) | Sample coordinates, inverse, composed affines | `origin + p ⊙ spacing` (push) equals `placed_metadata(result, fov).index_to_world(p)` within 1e-9 µm for integer and fractional `p`, including the contract's example `(0.69251, 252.44398, 100.25)` µm; the inverse `(sample − origin) ⊘ spacing` returns `p` within 1e-9 voxel; a cropped grid maps `q` to `q + start` first; the import composes every transform except `calibration` (newest first), its result equals `Stitching Transform` plus `Translation to Regular Grid`, a linear part off the identity by 5e-10 per entry is accepted, the links carry each `PairwiseResult`'s shift converted to µm as `shift_*` and its correlation as `quality`, the views map through the `ViewSetup` tile attribute (the permuted file gives the same result), and the rotated file raises `ValueError` naming the view. Exact or 1e-9 as stated. | H |
| PL4 | Gauge, fractional placement and the record | `stub_variants` (fractional); `scene_tiny` (42) | Origins, corrections, record | Origins equal `nominal + c − mean(c)` over the `image_supported` FOVs, bitwise against the same formula in the test; the mean correction is 0 within 1e-9 µm; no origin is rounded (fractions of 2⁻⁶ µm survive); the record assertions listed in {doc}`stitching-contract` ("Method record": `fov_id` order, one of the five statuses, float64 unrounded origins, finite exactly for the three placed statuses, zero mean correction, `placed_metadata`) hold, and `record["inputs"]` holds per FOV the grid shape, metadata and SHA-256 that assemble's grid check reads; `nominal_*` equal the table's values bitwise, `correction_*` equals `origin − nominal` (NaN where the origin is NaN) and `n_links_kept` equals the FOV's kept links. On `scene_tiny`, the test-only method returning the true corrections from `placement_truth.csv` gives `origin_*` equal to `true_origin_g1_*` within 1e-9 µm for every FOV. The inputs' own `ImageMetadata` are unchanged after `stitch` (equal before and after). | H; G for `scene_tiny` |
| PL5 | Anisotropic spacing and units | `stub_variants` (spacing (1, 2, 2) and the LN calibration); `golden_stitching_setups` with its view map; `xml_links` with `c_x ≠ c_y` (none) | Origins in µm; import units | Method translations in µm stay in µm; the import divides world z by the calibration's z scale (1.7894736842105263 for 0.34 / 0.19) and multiplies by each FOV's reference spacing, never by the macro's `voxel_size_*`; a calibration with `c_x ≠ c_y` raises `ValueError`. Within 1e-9 µm. | H |
| PL6 | Z = 1 | `stub_variants` (Z = 1 from YX inputs); a Z = 1 FOV beside a Z = 4 FOV (none) | Grids, origins, errors | YX inputs give Z = 1 grids; origins keep z from the method (0 for the stub); mixed Z sizes, spacings, directions or units raise `IncompatibleGeometryError` naming the FOVs, and so does a grid with unknown spacing, a unit other than µm or a non-identity direction; a spec without `2` in `dimensions` refuses Z = 1 inputs. Exact. | H |
| PL7 | Non-square tiles | `stub_grid` (16 × 24); `golden_stitching_setups` with its view map (none) | Boxes, origins, legacy view | Every box and origin uses the grid's own Y and X sizes; the import's legacy view (T3 to T7) of `golden_stitching_setups` reproduces `TILE_CONFIG` of `test_workflow_stitching_golden.py`, `img_col` for the Y extent included, while the record never uses `img_col`. Exact. | H |
| PL8 | Irregular footprint | `stub_variants` (unsnapped `maf_2x3_jitter`, unequal shapes, the L shape of `maf_3x2_gap`) (none) | Pairs, statuses, components | Pairs follow PL2's rule; every FOV of the L shape is placed and joined; the missing position has no row; the union of footprints is not a rectangle and nothing assumes one. Exact. | H |
| PL9 | A tile with no signal | `stub_variants` (one FOV with only NaN-quality links) (none) | Statuses, flags, links | Under L1 every NaN link has `kept` false and reason `nan_quality`, the FOV carries `nan_quality_link`; it is `failed` with `fallback=None` and `metadata_only` at `origin = nominal` with `fallback="nominal"`; review is `required`. Exact. | H |
| PL10 | Disconnected tile, components and the review policy | `stub_variants` (moved FOV, two components, review boundaries) (none) | Statuses, review, approval | The reference component is the one with the most FOVs, and on a tie the one with the first FOV in table order; FOVs outside it are `failed` (fallback `None`) or `metadata_only` (fallback `"nominal"`), `component` −1 only for `missing`; a correction of exactly `0.5 × w` is not flagged and `0.5 × w + 2⁻²⁰` µm is flagged `correction_above_bound`; a kept-link residual of exactly 1.0 µm is not flagged and `1.0 + 2⁻²⁰` µm is flagged `residual_above_tolerance` on both FOVs; a method-reported removed link gives `optimizer_removed_link`; each makes review `required` with its reason; `approve_placement` records reviewer, UTC time, note and the SHA-256 of both tables and changes no origin; `assemble` refuses the unapproved result with `PlacementReviewRequiredError` (AS10). An input whose image is `None` gives status `missing`, NaN origin and `component` −1, and review `required`. Exact. | H |
| PL11 | Stage-position fallback | `stub_variants` (moved FOV); `xml_links` without the `PairwiseResult` entries of view 3 (none) | Statuses, origins, legacy view | With `fallback="nominal"` the FOV is `metadata_only`, `origin` equals `nominal` exactly, `correction` is 0, flag `fallback`, review `required`; with `fallback=None` it is `failed` with NaN origin; in the import, BigStitcher's translation for that view stays in `method_translation_*`, unrounded, in both cases; the legacy view (`tile_config`) is byte-identical for both values. Exact. | H |
| PL12 | Import of the hand-built records | `hand_records_setups` (`xml_2x2_blank`, `xml_3x2_positive`) and `golden_stitching_setups`, each with its view map (none) | Raw translations, views, legacy view | The raw translation of every view equals the float64 sum of the two translations written in the file, computed in the test; the blank cell of `xml_2x2_blank` is a padding view with no row; the legacy view, with the fixture's `img_col` and `img_row`, equals the CSV that `legacy_tile_config` computes from the file without `ViewSetups`, byte for byte, and on `golden_stitching_setups` equals `TILE_CONFIG`; a missing file and a view without a view-map entry raise `ValueError` naming the path or the view. Exact. The five items of the LN `dataset.xml` are RX3. | H |
| PL14 | Persistence | results of PL4, PL10 (with an approval), PL12 (`xml_2x2_blank` of `hand_records_setups`) and PL18 (none) | Files and tables | `save_placement` writes `placement.json`, `fovs.csv`, `links.csv` (and for the import `view_map.csv` with the macro's and `dataset.xml`'s SHA-256); `load_placement` returns tables equal under `assert_frame_equal(check_exact=True)` (float64 origins bit-identical through CSV) and an equal record; a changed table raises `ValueError` naming it. `placement.json` holds `frame`, `gauge`, the table record, per FOV the input source, file SHA-256, grid shape, metadata and grid SHA-256, `review` and `format_version` 1; a save interrupted after its first file (a test hook raising) leaves no partial folder. Exact. | H |
| PL15 | Entry, registry and errors | `stub_grid` (none) | Raised errors, record | `stitch` refuses an unknown config type (`TypeError`), inputs that do not match the table, and incompatible grids, before calling the method; it never reads or writes a file (a test with a read-only temporary directory as the working directory); `STITCHING_METHODS` uses the shared mechanism (exact-type lookup, names, provenance entry with `stage` `stitching`); a spec whose dependency is missing raises `StitchingBackendUnavailableError` naming the module and the extra; importing `starfinder.stitching` imports no `multiview_stitcher`; `devices` is `("cpu",)`; `transform_kind` is `translation`; `supplied` is not a key of `STITCHING_METHODS`; `placement_from_table` reads and writes no file (the same read-only working directory). Exact. | H |
| PL16 | Determinism of the stub path | `stub_variants` (none) | SHA-256 of `fovs.csv` and `links.csv` in three single-thread processes | Identical. | H; S |
| PL17 | `Dataset.stitch` | a synthetic dataset folder written in the test: four FOVs of 4×16×24 with `images/ref_merged/{fovID}.tif` carrying reference metadata, one copy with a FOV of another tile size; the same FOVs with a saved `reference_stain` checkpoint image each and with a `file:` pattern, one file of another shape (none) | Table, inputs, saved files, errors | Given the workflow `positions` mapping without `tile_shape_yx` and `spacing_yx`, the table equals `positions_from_grid` with the grids' `shape_zyx[1:]` and `spacing_zyx[1:]`, recorded in the table's record; the inputs are built in table order; with `output_dir` the files of PL14 are written; the copy with another tile size raises `IncompatibleGeometryError` naming the FOVs and both values. `input="reference_stain"` reads `<checkpoint dir>/<fov_id>/other_rounds/reference_stain/image.ome.tif` and `input="file:<pattern>"` accepts a file of the grid's shape and raises `IncompatibleGeometryError` for another shape; the FOV images are loaded one at a time (counted through the reader); no `FOV.run`, `FOV.segment` or `FOV.assign_molecules` is called (each monkeypatched to raise). Exact. | H |
| PL18 | `placement_from_table`, without and with `origins` | `stub_grid`'s table and grids; `own_cases`' table and inputs (one layer, `nominal_z` 0); a one-FOV table; Z = 1 grids (1×16×24); one grid of 4×16×20 beside 4×16×24; `own_cases`' table with `origins` (0.3, 0.25, 1.5) and (−0.7, −0.125, 15.0) µm, that is corrections fractional in Z, Y and X, one above `max_residual_um` (1.5 µm in X) and one above `max_correction_fraction` of the 4 µm nominal overlap (3.0 µm in X, above 0.5 × 4 µm); `stub_grid`'s table with `origins` equal to its nominal positions; `scene_tiny`'s table from `positions.csv` with `origins` from `true_origin_z/y/x` of `placement_truth.csv` (42); a table whose `nominal_z` differ (0 and 1 µm), without and with `origins` whose z are equal; for the failures: an empty table, a table FOV without a grid, a grid key outside the table, a grid with unknown spacing, one in nm, one with a non-identity direction, grids of Z size 4 and 5; with `origins`: a table FOV without an origin, an origin for a `fov_id` the table lacks, an origin with NaN, one with +inf, one of two values, and `origins={}` | Per-FOV table, links, record, review, persistence, `assemble`, errors | In every case the result is a `PlacementResult`; its `fovs` rows equal the table's `fov_id`s in order, each `metadata_only` with `flags` exactly `supplied`, `nominal_*` equal to the table's bitwise, `method_translation_*` NaN, `n_links_kept` 0 and `component` the row index (one FOV: one row, component 0); `links` has the columns of "Links" and no row; `record["method"]` has `stage` `stitching`, `method` `supplied`, `implementation` `starfinder.stitching.placement_from_table` and empty `config_type`, `config`, `requires`, `artifacts`; `record["gauge"]` is `none`; `record["frame"]` has `frame_id` `sample:<sample>`; `record["inputs"]` holds per FOV the source `supplied`, a null file SHA-256 and the grid shape, metadata and SHA-256. **Without `origins`:** `origin_*` equal `nominal_*` bitwise (z included for Z = 1), `correction_*` 0.0 and `record["origins"]` is `nominal`. **With `origins`:** `origin_*` equal the supplied values bitwise (no rounding, cast or shift; fractions survive), `correction_*` equals `origin − nominal` computed in the test bitwise, also above 1.0 µm and above the correction bound, and `record["origins"]` is `supplied`; origins equal to the nominal positions give a `fovs` table equal under `assert_frame_equal(check_exact=True)` to the one without `origins` (equal SHA-256 of `fovs.csv`), and their `placement.json` differ only in `record["origins"]`. **Review:** `required` with the reasons exactly `[supplied_placement]` in both cases; `review_placement` keeps them and adds no flag (no `residual_above_tolerance`, no `correction_above_bound`) and turns no FOV `failed`, for the corrections above 1.0 µm and above the bound too. **Persistence:** the save and load round trip returns equal tables and record, `fovs.csv` holding the nominal, origin and correction columns (float64 bit-identical through CSV), `placement.json` holding `origins`, `links.csv` the header only. **`assemble`:** on `own_cases` without and with the Z-jittered `origins`, `assemble` raises `PlacementReviewRequiredError` until `approve_placement` has run and accepts the approved result, whose origins and corrections are bitwise those before; its one `nominal_z` passes check 8 whatever the z of the origins; the table whose `nominal_z` differ raises `PlacementReviewRequiredError` unapproved and, approved, `ValueError` from check 8 ("assemble handles one layer of tiles"), without and with `origins`. **`scene_tiny`:** the approved result's origins equal the `true_origin_*` read from `placement_truth.csv` bitwise and its corrections equal `true_correction_*` within 1e-9 µm, Z included; it is the truth placement of EE1 to EE5. **Errors:** the table FOV without a grid and the extra grid key raise `ValueError` naming the FOVs, the empty table raises `ValueError`, and the four geometry cases raise `IncompatibleGeometryError`; unequal Y and X sizes are accepted; each faulty `origins` raises `ValueError` naming the FOVs (`origins={}` names every FOV; only `None` omits the origins). Exact otherwise. | H; G for `scene_tiny` |

PL13 of the first version (BigStitcher's result on W-324's Fiji outputs) reads files on the
share and is RX4.

### `multiview_stitcher` (MV)

Starting configuration of {doc}`stitching-algorithms`: phase correlation, translation,
binning 4 in every axis, links with quality below 0.7 or NaN dropped (L1),
`alternating_pattern` pruning, global optimization with library tolerances.

| # | Check | Fixture (seeds) | Metric | Pass/fail tolerance | Needs |
| --- | --- | --- | --- | --- | --- |
| MV1 | Known-answer cut fixture, starting configuration | `cut_synthetic` 3 × 3 (20261006, 324, 1000 + tile) | Error of `origin_*` against the true origins, after removing the mean, per axis, in voxels | All nine FOVs `image_supported`, one component; max absolute error at most 2 voxels per axis. | MV, S |
| MV2 | The same at `binning_zyx=(2, 2, 2)` | `cut_synthetic` 3 × 3 | As MV1 | At most 1 voxel per axis. | MV, S |
| MV3 | Fractional jitter on the synthetic scene | `scene_small` (42); input per FOV the reference-round image, maximum over channels (the `reference_merged` analogue); `binning_zyx=(1, 1, 1)`, with the link filter off (`quality_threshold=-1.0`, NaN-quality links still dropped), because this row tests precision at fractional offsets and the link rule is tested by MV4 and MV5 | `origin_*` against `true_origin_g1_*` of `placement_truth.csv`, in voxels | All nine FOVs `image_supported`; max absolute error at most 0.5 voxel per axis. | G, MV, S |
| MV4 | Hand-built cases | `cut_synthetic`: `two_tiles`, `z1_3d`, `plane_2d`, `spacing_1_2_2`, `non_square`, `no_signal_zero`, `no_signal_noise` (1050), `disconnected` | Statuses and error per axis | Every FOV of the reference component within 2 voxels per axis; Z = 1 cases have z correction exactly 0 (registered in 2D); `spacing_1_2_2` errors in each axis's own voxels; the centre tile of both no-signal cases and tile 8 of `disconnected` are `failed` with `fallback=None` and `metadata_only` at nominal with `fallback="nominal"`; review `required` in those three cases; in the others the review state and its reasons are recorded (with spacing (1, 2, 2) µm a residual of half a voxel already reaches the 1.0 µm default, so no state is asserted there). | MV, S |
| MV5 | Wrapper rules | `cut_synthetic` 3 × 3; two tiles with no overlap | Links, errors | The registered pairs are a subset of `neighbor_pairs` and every other pair is recorded `pruned`; no NaN-quality link is kept; translations are in µm, view to common frame; with no link left, every FOV is `failed` (the library's `NotEnoughOverlapError`), and the result is still returned and saved. Links with quality below 0.7 have `kept` false and reason `quality_below_threshold`; a tile below 16 px in Y or X is refused (`min_shape_zyx`); `MultiviewStitcherConfig().method` is `multiview_stitcher`, the spec's name; `MultiviewStitcherConfig` refuses, at construction, `binning_zyx` with a 0, `quality_threshold` 1.5, `upsample_factor` 0, a negative `overlap_tolerance_um`, and a `pairwise`, `pruning` or `groupwise` value its table does not list. Exact. | MV, S |
| MV6 | Determinism | `cut_synthetic` 3 × 3 | Translations of two runs at one thread, each in its own process | Bitwise identical. The independence of the thread count, which W-324 measured (`pairs_between_runs.csv`), is checked in the real placement examples (RX7), whose bound allows four threads; the `slow` tier allows one CPU. | MV, S |

### Assembly (AS)

Each accounting check cites the identity of {doc}`assembly-contract`, "Count
accounting", by its number. An owning FOV is one of that page's "Terms": placed, with an
input, `image_supported` or `metadata_only`.

| # | Check | Fixture (seeds) | Metric | Pass/fail tolerance | Needs |
| --- | --- | --- | --- | --- | --- |
| AS1 | Sample coordinates | `assembly_golden`; `stub_variants` (LN calibration) (none) | `sample_*`, `sample_centroid_*`, `sample_expanded_centroid_*`, `sample_ownership_centroid_*` | Each equals `placed_metadata(…).index_to_world(p)` within 1e-9 µm for the molecule's `z, y, x` and the float centroids (never truncated); `sample_centroid_z` is NaN for a `plane` cell; `sample_ownership_centroid_*` equals the pair `ownership_centroid` names, bitwise; the example `(1.3, 37.4, 0.7) + (3, 8, 20) = (4.3, 45.4, 20.7)` µm holds; with `centroid="original"` every `ownership_centroid` is `original`, and with `territory` it is `expanded` exactly on the cells of the expanded FOV. No integer `global_*` column is produced. | H |
| AS2 | Ownership at region edges and ties | `edge_points`; `assembly_golden`; a 0.25-voxel raster over `assembly_golden`'s footprints (none) | `owner()` | A point at the lower footprint bound is inside and one 1e-9 µm below is not; one at the upper bound is outside and one 1e-9 µm below is inside; an exact tie goes to the FOV first in table order, and reversing the order of the two FOVs in the placement changes the owner; every raster point inside an owning FOV's footprint has exactly one owner and every other point none; the FOV1/FOV3 boundary lies between x 59.1 and 59.3 and the FOV1/FOV2 boundary near y 42.2 ({doc}`assembly-algorithms`, "O1"); `midpoint_box` on a placement without grid cells raises `ValueError`. With `ownership="midpoint_box"` on `assembly_golden` the box edges lie at x 59.2 and y 42.2 and the retained sets equal AS3's ({doc}`assembly-algorithms`, "O2"); moving a point in z changes no owner (Z1); with `midpoint_box` and `tie="half_open"` a point on a box edge belongs to the FOV whose box starts there (T2). Exact. | H |
| AS3 | Retained sets on the golden fixture | `assembly_golden` (3D and 2D) (none) | Retained scene cells per FOV, flags | 3D: FOV1 retains scene cells 1, 2, 5, 9; FOV2 4, 6; FOV3 3, 7, 8; 2D: FOV3 also 9, both copies flagged `conflict` with each other in `conflict_with`; cell 9's molecule at x 64 is excluded `cell_not_owner` in 3D and retained in 2D (W-326 `tables/option_effects.json`). Exact. | H |
| AS4 | Retention rules and reasons | `own_cases`; `status_cases` (none) | `assembly_status`, `assembly_reason`, `owner_fov_id` per row | Every row equals the status, reason and owner written in the fixture: rules 1 to 7 of "Retention", reasons `not_owner`, `fov_not_placed` (cells) and `cell_not_owner`, `not_owner`, `outside_grid`, `fov_not_placed` (molecules); a retained `excluded_no_nucleus` cell keeps its §2.9 status and its molecules follow it; no molecule changes `cell_id` or `assignment_status`. The cell rows are in placement table order, then increasing `cell_id`, and the molecule rows in placement table order, then input order; the cell and molecule tables have the columns of "Sample tables", every §2.9 column equal to the input's, and `obs_name` is `<sample>_<fov_id>_<cell_id>`. Exact. | H |
| AS5 | Flags | `own_cases`; `assembly_golden`; `status_cases` (placeholder with an input) (none) | `assembly_flags`, `conflict_with` | `fov_edge`, `z_edge`, `conflict`, `unclaimed`, `outside_owner_region`, `cell_unclaimed` each on exactly the rows written in the fixture; in 3D FOV1's cell 6 is `fov_edge` and FOV3's cell 5 is not `unclaimed` ({doc}`assembly-algorithms`, "Flags"); the placeholder's cell whose ownership centroid no owning FOV covers is `unclaimed` with a null `owner_fov_id`; with `conflicts=False` the `conflict` and `unclaimed` columns are null and the record says so; no flag changes a status (the statuses with and without `conflicts` are equal). Exact. | H |
| AS6 | Accounting identities | `assembly_golden` (3D, 2D, expanded), `own_cases`, `empty_cases`, `status_cases` (none) | The sample tables against the inputs | **Identity 1**: anti-joins of the input keys `(cell_namespace, cell_id)` and `(spot_namespace, spot_id)` against the sample tables are empty both ways and no key repeats; **2**: per FOV and in total, input rows = retained + Σ excluded by reason, for cells and molecules, and `record["counts"]` equals these totals; **3**: every `assigned` or `excluded_cell` molecule has its cell's status (reason `cell_not_owner` where the cell's is `not_owner`), every `unassigned` molecule of a placed FOV is retained exactly when `owner_fov_id` equals its `fov_id`, which never holds for a `placeholder`, every `outside_grid` molecule is excluded; **4**: the sorted `(fov_id, cell_id, gene_id, compartment, count)` rows of the sample `counts` equal those of the per-FOV §2.9 `counts` restricted to the counted cells, and `Σ whole` = retained `assigned` molecules = Σ `n_molecules` over the counted cells; **5**: `whole = nucleus + cytoplasm` per counted cell with `compartments` `available`, no compartment rows otherwise. Each identity is evaluated independently of `AssemblyResult.__post_init__`, and a result built with one row removed or one status changed is refused by `__post_init__` with `ValueError` naming the identity. `obs_name` is unique; `matrix()` equals the dense `whole` counts in `genes` order. Exact. | H |
| AS7 | Molecules-only case | `empty_cases` (`molecules_only`) (none) | Tables, AnnData | `mode` `molecules_only`; cell and count tables have no row; every molecule is retained or excluded by its own position (`outside_grid` positions excluded); identities 1 to 3; `sample.h5ad` has `X` of shape `(0, n_genes)`, `var["n_molecules_retained"]` equal to the retained molecules per gene, and `uns["assembly"]` the record. Exact. | H, A |
| AS8 | Empty inputs | `empty_cases` (none) | Tables, record, errors | A FOV without molecules, a FOV without cells and a FOV whose molecules miss every cell go through the same rules without a branch (identities 1 to 5 hold, the `fovs` table has their rows); a sample with no counted cell writes `sample.h5ad` with no row; a placement with no placed FOV, and an approved placement whose FOVs are all `placeholder`, raise `ValueError` because no FOV owns a point. Exact. | H, A |
| AS9 | Placeholder, missing and failed FOVs | `status_cases` (none) | Statuses, ownership, errors | `image_supported` and `metadata_only` FOVs without an input raise `ValueError`; a `missing` or `failed` FOV's input is accepted and all its rows are `fov_not_placed` with NaN sample coordinates and null owner; a `placeholder`'s input is accepted, every one of its cells is excluded `not_owner` with `owner_fov_id` the neighbour that owns the ownership centroid or null where none does, its molecules are excluded (`cell_not_owner`, `not_owner` or `outside_grid`), and `fovs.owns_points` is false for it and true for every owning FOV; without an input the placeholder's region costs no neighbour an observation (AS16). Exact. | H |
| AS10 | Review refusal and entry checks | `status_cases`; `assembly_golden` with one violation per check (none) | Raised errors | Checks 1 to 8 of "Checks the entry applies" each raise their named error (`PlacementReviewRequiredError`, `ValueError`, `IncompatibleGeometryError`) naming the FOVs, in that order (an input that violates two raises the earlier); check 1 refuses unknown values of the four fields and `tie="half_open"` with `ownership="nearest_center"`; an approved placement passes. `assemble` writes no file (a read-only working directory) when its inputs are in memory. Exact. | H |
| AS11 | Sample-level AnnData against the sample tables | `assembly_golden` (3D with an expanded FOV), `own_cases` (with a retained `excluded_no_nucleus` cell) (none) | `sample.h5ad` written by `write_sample_h5ad(result, path)` | Every condition of "A test can check it against the sample tables" in {doc}`assembly-contract`: `n_obs` equals the counted cells and `obs_names` and `obs[["fov_id", "cell_namespace", "cell_id"]]` equal those rows in order; no other row's `obs_name` appears; `X[i, g]` equals the `whole` count (0 where none) and the `nucleus` and `cytoplasm` layers likewise, NaN rows where `compartments` is not `available`, `X` and the layers dense float64; with `annotation` the columns of the annotation row appear in `obs`; every sample-coordinate column and `ownership_centroid` equal the cell table's exactly; `obsm["spatial"]` columns equal `sample_ownership_centroid_x` and `_y` (the expanded centroids on the expanded FOV); `uns["spatial_centroid"]` names those columns and the setting, and `uns` holds `spatial_unit` `"µm"` and `frame_id`; `var_names` equal `genes`; the file is at the given path. Exact (NaN equals NaN). Without `anndata` importable (monkeypatched), `write_sample_h5ad` raises `ModuleNotFoundError` naming the `anndata` extra. | H, A |
| AS12 | Persistence in the export folder | AS3 and AS7 results, CSV and Parquet; a folder whose `export.json` already holds a `fusion` section (none) | Files, tables, record | `save_assembly(result, folder, table_format=…)` writes `tables/fovs.<fmt>`, `tables/cells.<fmt>`, `tables/molecules.<fmt>`, `tables/counts.<fmt>` and `sample.h5ad`, and replaces only the `assembly` section of `folder/export.json`, creating the record when it is absent (the existing `fusion` section is unchanged in canonical JSON); the section lists every file with its SHA-256; `load_assembly(folder)` returns tables equal under `assert_frame_equal(check_exact=True)` and an equal record; a changed file raises `ValueError` naming the path and both hashes. The section holds the placement's method, gauge, review and table hashes, per FOV the placement status, origin, kind, assignment hashes, `territories` with `path` and `sha256` (the path resolves to a file whose SHA-256 is the one recorded; both null with `conflicts=False`, for a `molecules_only` input and for an assignment given in memory), the grid and `calibration_source`, the config, footprints, centres and `owns_points`, the counts and `format_version` 1. Every file is written atomically: a save interrupted between two files (a test hook raising) leaves each file complete or absent and no temporary file. Exact. | H, A |
| AS13 | `write_h5ad` and the legacy adapter | `test_assignment_workflow.py`'s configurations and the A18 `raw.h5ad` comparison; `write_h5ad` called with no row in `obs`, with and without `layers`, and with `var` (none) | `raw.h5ad`; the written AnnData; errors | After `write_h5ad` is factored out of the adapter's `_write_raw_h5ad`, `raw.h5ad`'s rows, `obs` columns, `X`, layers and `uns` equal the frozen helper's values, as A18 of {doc}`assignment-algorithms` states; with no row it writes `X` of shape `(0, len(genes))`, and layers of that shape with `layers=True`; `var` columns are written when given; it returns the path, creating the folder; a `keys` row count that differs from `obs`, or a key column missing from `counts`, raises `ValueError`; without `anndata` it raises `ModuleNotFoundError` naming the extra. Exact. | H, A |
| AS14 | Determinism and row order | `assembly_golden` (3D) (210) | SHA-256 of every sample table in three single-thread processes; per-key results with the input molecule rows permuted by `numpy.random.default_rng(210).permutation` | Identical hashes; identical per-key status, reason, owner and flags. | H; S for the processes |
| AS15 | The assemble part of `Dataset.export` | `export_folder` with `ExportConfig(fuse=False)`; the same with `assignment=None` and saved molecule tables (none) | Result, saved folder | The tables equal those of `assemble` called directly on the same inputs (`assert_frame_equal(check_exact=True)`) and the records are equal except their timing fields; `export/<sample>/` holds `export.json` (section `assembly` `written`, `fusion` `off`), the four tables under `tables/` and `sample.h5ad`; a rerun with `assemble: false` loads the saved assembly, checks every SHA-256 and records `reused`; with `assemble: false`, no saved assembly and no later part on, the part is `off`; a changed assignment file raises naming the path; a missing territory file and one with another SHA-256 raise `ValueError` naming the path and both hashes; at most two territory images are loaded at a time (counted through the loader). Exact. | H, A |
| AS16 | Placeholder FOVs own no points | `placeholder_grid` with its raster; its variant whose FOVs are all `placeholder` (none) | `owner()` over the raster; retained sets; `fovs` | No raster point has a placeholder owner; every point inside an owning FOV's footprint has exactly one owner, the nearest-centre owning FOV with T1 on ties as the test recomputes it over the owning FOVs alone; the points of each placeholder's footprint that no owning FOV covers (its middle, and for the corner and the edge FOV the strip along the grid's border) have no owner; each owning FOV's box cell whose ownership centroid lies in a placeholder's footprint is retained, neither `not_owner` nor `unclaimed`; under `ownership="midpoint_box"` the box edge of an owning FOV towards a placeholder lies at its own footprint edge; `fovs` keeps the placeholders' footprints and centres with `owns_points` false; the all-placeholder variant raises `ValueError` (no owning FOV). Exact. | H |
| AS17 | `candidates` of `ownership_map` | `assembly_golden`'s placement and raster; `placeholder_grid` (none) | `owner()` with and without `candidates`; errors | For every single owning FOV, every pair and the set of all owning FOVs, the owner of each raster point equals the test's nearest-centre rule restricted to those candidates whose footprint contains it, or none; `candidates=None` gives the same owners as the set of all owning FOVs, which is what `assemble` uses; an empty collection, a `placeholder`'s `fov_id`, an unknown `fov_id`, a `missing` FOV and an owning FOV without a grid in `grids` each raise `ValueError` naming it. Exact. | H |

### Synthetic sample scene (SC)

The generator addition's own checks are those of {doc}`synthetic-sample-scene` ("Tests the
implementation keeps and adds"), delivered by its implementation issue; the scene enters
this design through PL4, PL18, MV3 and EE1 to EE6. One case is a row here because the
review of 2026-10-06 asked for it:

| # | Check | Fixture (seeds) | Metric | Pass/fail tolerance | Needs |
| --- | --- | --- | --- | --- | --- |
| SC1 | A successful single-plane scene | `scene_plane` (42) | Generation, shapes, z values, flags, identities, label kind, presets | Generation succeeds without an error; the sample grid has one plane and every FOV image, stain and label image has the shape `(1, 36, 36)`; every nucleus centre, cell anchor and molecule has z exactly 0, and every FOV's true offset in Z is 0; no `cell_fovs` row has `z_edge`; the cellular identities 1 to 5 of "Identities a test can check" hold; `import_labels` reads one FOV's saved labels as a `plane` label image; `SAMPLE_PRESETS` keeps its two keys. Exact. | G |

### End-to-end path of the synthetic scene (EE)

The path of {doc}`synthetic-sample-scene` ("The path that needs no segmenter"): per FOV,
the truth labels enter through `import_labels(…, relabel=False)` and the molecules
through `import_legacy_molecules_csv`; `assign_molecules` runs without expansion; then
`assemble`. The placement is the truth in EE2 to EE5: `placement_from_table` on the
nominal grid of `positions.csv` with `origins` from `true_origin_z/y/x` of
`placement_truth.csv`, approved with `approve_placement`, so every FOV is `metadata_only`
at its true origin, flagged `supplied`, with the grid's one `nominal_z`, which check 8 of
{doc}`assembly-contract` accepts. EE2 to EE5 therefore run on `scene_tiny`, whose true
origins carry the J1 jitter in Z, Y and X, as well as on `scene_tiny_flat` (no Z jitter)
and `scene_tiny_flat_inward`; Z jitter goes through `assemble` on a truth placement. EE1
needs no placement. In EE6 the placement is the result of `stitch`, whose `nominal_z` are
those of the grid (0), so the Z-jittered `scene_small` also goes through `assemble` on a
stitched placement. The oracle of EE2 is computed in the test
from the truth tables, the FOV truth label images and the rules of
{doc}`assembly-algorithms`, never from `assemble`'s output.

| # | Check | Fixture (seeds) | Metric | Pass/fail tolerance | Needs |
| --- | --- | --- | --- | --- | --- |
| EE1 | Per-FOV assignment equals the sampled truth | `scene_tiny`, `scene_tiny_flat`, `scene_tiny_flat_inward` (42) | Per `molecule_fovs` row: §2.9 status and `cell_id` | Rows with `in_footprint`: `cell_id` (null read as 0 for `unassigned`) equals `fov_cell_label`; rows without: `outside_grid`; the `label_agrees` false count per FOV is recorded (a measurement of the §2.9 sampling rule, not a failure). Exact. | G |
| EE2 | Assembly equals an independent oracle | `scene_tiny`, `scene_tiny_flat`, `scene_tiny_flat_inward` (42) | Per row: `assembly_status`, `assembly_reason`, `owner_fov_id`, `fov_edge`, `z_edge` | Equal to the oracle (O1 and T1 recomputed from the true origins and grid shapes, in Y and X only (Z1); centroids recomputed from the truth label voxels); `fov_edge` and `z_edge` equal `cell_fovs`' columns of the same name for every (cell, FOV); on `scene_tiny`, every `sample_z` and `sample_centroid_z` equals `true_origin_z + p_z·s_z` of its FOV within 1e-9 µm, so the Z jitter reaches the sample tables. Exact otherwise. | G |
| EE3 | Accounting on the scene | as EE2 | Identities 1 to 5 of "Count accounting" | As AS6. Exact. | G |
| EE4 | Guarantees against sample-level truth | as EE2 | Sample molecules (`in_sample` true) and their copies | (a) every sample molecule has at least one copy that is not `outside_grid` (identity 3 of {doc}`synthetic-sample-scene`); (b) every sample molecule whose copy in the FOV owning its sample position samples the label 0 has exactly one retained `unassigned` copy, in that FOV; (c) a molecule with `r ≥ 2` retained copies has at least `r − 1` of them flagged `outside_owner_region`; (d) every retained `assigned` copy whose `cell_id` differs from `true_cell_id` is a row with `label_agrees` false, and the converse holds on retained `assigned` copies. Exact. | G |
| EE5 | Comparison with sample-level truth | as EE2 | Per sample cell: retained copies (0, 1, 2 or more) by flag, against `n_molecules` of `cells.csv` (the molecules whose `true_cell_id` is the cell); per sample molecule: retained copies; the same for halo content, reported apart | The table is written to the test's output and equals the oracle's recount exactly. Duplicates and losses are the agreed behavior of retention without matching ({doc}`assembly-contract`, "What this policy does not guarantee"); their numbers are measurements, not failures, and no accuracy is claimed. | G |
| EE6 | The full chain | `scene_small` (42), placement from MV3, approved with `approve_placement` when its review is `required` (the state and reasons recorded) | MV3's bound; EE2 and EE3 on the stitched placement; differences from the truth-placement assembly | MV3 passes; EE2 and EE3 hold exactly with the stitched origins in the oracle; the observations whose status differs from the assembly on the truth placement in gauge G1 (`true_origin_g1_*` through `stitch` with the test-only method of PL4, whose `nominal_z` are 0) are listed (a measurement). | G, MV, S |

### The export entry (EX)

`Dataset.export` and the folder `export/<sample>/` of {doc}`sample-output-contract` ("The
export entry", "The export folder and the output profiles", "Read-back checks", R1 to
R3). Each row runs `check_export` on the folder it writes, unless the row tests a refusal.

| # | Check | Fixture (seeds) | Metric | Pass/fail tolerance | Needs |
| --- | --- | --- | --- | --- | --- |
| EX1 | The parts, in order and alone | `export_folder` with the default `ExportConfig`, and with `fuse_labels: true` and `export_spatialdata: true` (none) | Spied part calls; files under `export/<sample>/`; `export.json` | `Dataset.export` calls `assemble`, `fuse_images`, `fuse_labels` and `export_spatialdata` in this order, each only when its key is on; with the defaults the folder holds exactly `export.json`, `tables/fovs.csv`, `tables/cells.csv`, `tables/molecules.csv`, `tables/counts.csv`, `sample.h5ad` and `reference_stain_projection.ome.tif` (the maximum projection at scale 0.5, factor 2, in the input dtype), with sections `assembly` and `fusion` `written` and `labels` and `spatialdata` `off`, and `complete: true`; with `images: [reference_stain, reference_merged]` (a `ref_merged` image per FOV added to the folder) one projection file per image name is written on the same grid; no file is written outside `export/<sample>/`, in particular none under `images/fused/{sample}/`; each part called alone with the saved files of the parts before it writes its files and its section and leaves the other sections unchanged in canonical JSON; each section records the SHA-256 of the sections it was computed from; `check_export` passes. Exact. | H, A, MV; SD for `export_spatialdata` |
| EX2 | Checks before anything is written | `export_folder`; one violation per check: an unknown `dimensionality`, a placement whose review is `required` without approval, `anndata` and (with `table_format: parquet`) `pyarrow` made unimportable, `assemble: false` with `fuse_labels: true` and no saved assembly and with a saved assembly of another placement, a `FusionInput` missing and one naming another FOV, a grid that differs from the placement's, a `<name>=file:<pattern>` source image of another shape than its grid, two spacings, one-plane beside multi-plane FOVs; `fuse_images` called alone with an assembly of another placement; `id_bound` with `fuse_labels: true` (none) | Raised errors; the folder before and after | Checks 1 to 6 of "Checks before anything is written" raise in this order (`ValueError` from the configs, `PlacementReviewRequiredError`, `ModuleNotFoundError` naming the extra, `ValueError`, `ValueError` or `IncompatibleGeometryError`, `ValueError` naming the count), each naming the FOVs or keys; an input that violates two raises the earlier; `fuse_images` called alone with an assembly whose placement SHA-256 values differ from the placement's raises `ValueError`; the folder's listing and every file's SHA-256 are the same before and after; `id_bound`'s 16,777,216 rows (2^24) pass check 6. Exact. | H, A |
| EX3 | `Dataset.export` when a part cannot run | `export_folder` variants, each with the part or check that meets the fault first under the order of the parts: (a) `multiview_stitcher` not importable, with `fuse_labels: true` and `export_spatialdata: true`: `assemble` needs only `anndata` and runs, `fuse` meets the fault; (b) one FOV without its `reference_stain` file, with `fuse_labels: true`: `assemble` reads no source image, `fuse` meets the fault; (c) after a first export with the default `conflicts: true`, whose `assembly` section records each territory file's path and SHA-256, one owning FOV's territory file removed, then `Dataset.export` again with `assemble: false`, `fuse_labels: true` and `export_spatialdata: true`: check 4 of "Checks before anything is written" loads the saved assembly with `load_assembly`, which checks every SHA-256 the section holds, the territory files' included, and meets the fault; (d) `spatialdata` not importable, with `export_spatialdata: true`: `export_spatialdata` meets the fault; (e) as (c) with the territory file replaced by one with another SHA-256: check 4 meets it; (f) the change of (c) with `assemble: true`: the assemble part meets it first, when the §2.9 loader checks the recorded SHA-256 of the assignment folder it loads; (g) a first export with `conflicts: false` in the `assembly` block, whose section records no territory path or SHA-256 (both null, {doc}`assembly-contract`, "What assemble records"), then one owning FOV's territory file removed and `Dataset.export` again with `assemble: false`, `fuse_labels: true` and `export_spatialdata: true`: the reused assembly loads, and `fuse_labels`, which then resolves the territory file from the recorded assignment folder through the §2.9 loader ({doc}`sample-output-contract`, "Assembled label export", inputs), meets the fault; (h) as (g) with the territory file replaced by one with another SHA-256: `fuse_labels` meets it (none) | Return value, warnings, `export.json`, the folder | (a), (b), (d) and (g) raise nothing and issue one `ExportIncompleteWarning` per part that could not run, naming the part and the reason; the statuses are those of the table "When a part cannot run": (a) `fuse` and `fuse_labels` `not_run`, `extra_missing`, `extra: "multiview-stitcher"`, and `export_spatialdata` `written` with the image and `cell_labels` elements omitted `part_not_run`; (b) `fuse` `not_run`, `source_image_missing` with that FOV and image name, no image file for any image name, and `fuse_labels` `written`; (d) `export_spatialdata` `not_run`, `extra_missing`, `extra: "spatialdata"`, and no `sample.zarr`; (g) `assemble` `reused`, `fuse_labels` `not_run`, `territory_missing` with that FOV and path, and `cell_labels` omitted `part_not_run`; in each of these `complete` is false, the files of the parts that ran exist and are listed with their SHA-256, a part `not_run` lists no file, and `check_export` passes. (c) and (e) raise `ValueError` from `load_assembly` naming the territory path (and, in (e), both hashes) before anything is written, and the folder's listing and every file's SHA-256 are as before the call. (f) raises `ValueError` naming the path, as AS15 states, and no new table is written. (h) records `fuse_labels` `failed` with the exception's type and message, writes `export.json` and raises the error again; the earlier files stay listed. Exact. | H, A; MV for the parts that run in (b) and (g), (h); SD in (a) and (g) |
| EX4 | The folder equals its record | `export_folder` run with `dimensionality: both`, then with `dimensionality: projection`; an unrelated file placed in the folder; a run interrupted between two files of the fuse part (a test hook raising); one file of a finished folder deleted, one added, and the `assembly` section edited (none) | Files, `export.json`, `check_export` | The second run removes `reference_stain_stack.ome.tif`, which the first record listed, and keeps the unrelated file; after the interruption `export.json` lists the parts that finished, every listed file exists and no temporary file remains; `check_export` fails, naming the condition, on the unlisted file (R2), on the listed missing file (R2), and on the `fusion` section whose recorded `assembly` SHA-256 no longer equals the section's (R1), and passes on the unchanged folder; with `assemble` `off` no table or `sample.h5ad` exists and R3 holds. Exact. | H, A, MV |
| EX5 | The fusion section of `export.json` | the folders of EX1 and FI1 to FI7 (none) | Section content | Every item of "The fusion section of `export.json`" is present: status, the placement's method, gauge, review, approval and table SHA-256, the `assembly` section's SHA-256 when an assembly was given; per FOV `fov_id`, placement status, included or the reason it is not, origin, grid, and per source image its path and SHA-256; the library version, `fusion_func` `weighted_average_fusion`, `interpolation_order` 1, `blending_widths_um` `{z: 3, y: 10, x: 10}`, the route (`level1` for `k > 1`, else `none`) and `inputs_2d`; the config and the fusion grid (`n0`, `s`, `o`, and the slab for one-plane FOVs); per output file path, SHA-256, format, image, dimensionality, projection, factor, shape, dtype, spacing, origin, levels, tile or chunk shape and compression; `software`, `wall_s`, `cpu_s` and `peak_rss_kib` (non-negative numbers, recorded, not bounded); `viewer_qualification: "deferred"`. Exact. | H, MV |

### Fused image (FI)

Each row names the condition of {doc}`sample-output-contract` it implements: conditions 1
to 5 of "Sample position to image position" and F1 to F9 of "Read-back checks".
Multi-plane FOVs are fused as 3D and one-plane FOVs as 2D ("The fusion grid"). The placement of each fixture is `placement_from_table` of its origins, approved with
`approve_placement`, and `fuse_images` is called alone with `assembly=None`; fusion has no one-layer
check, so Z origins may differ there.

| # | Check | Fixture (seeds) | Metric | Pass/fail tolerance | Needs |
| --- | --- | --- | --- | --- | --- |
| FI1 | The fusion grid and F8 | `quad` (2102), `zstack` (equal and different Z; the one-plane pair), `plane_pair`, `fuse_band`; `quad`'s first FOV alone; `status_cases`'s placement with every FOV excluded, `fuse_images` called alone | `fused_grid` and the `fusion` section | Spacing is the FOVs' shared spacing; `o` is the smallest included origin per axis, bitwise; `n0` per axis equals `floor((max_f (origin_f + (n_f − 1)·s) − o) / s + 1e-9) + 1` evaluated in the test: `quad` 4 × 58 × 58, `zstack` 4 × 8 × 14 and 5 × 8 × 14, `fuse_band` 16 × 1408 × 256; one included FOV gives its own grid; excluded FOVs do not extend it; one-plane FOVs give `n0_z = 1` with the slab `[z_lo, z_hi)` and `S_z = z_hi − z_lo`, `O_z = (z_lo + z_hi)/2` within 1e-9 µm (`plane_pair`: 3 and 1; the `zstack` one-plane pair: 1 and 0); the grid the library records equals the grid passed; with no included FOV no image is written and the section has status `empty`, `outputs: []` and `reason: "no_included_fov"` (F8), and only then. Exact. | H for `fused_grid`; MV for the files; S for `fuse_band` |
| FI2 | Values of the library's blending (F4) and the dtype written (F2) | `overlap_pair` in uint8, uint16 and float32, at scale 1 | Values where one FOV covers, in the overlap, and the file's dtype | Where one FOV covers, every value equals that FOV's exactly; in the overlap every value lies between the two inputs inclusive (W-354: 100 to 202, 1000 to 3000, 0.25000232458114624 to 0.7999976873397827), and no voxel of the uint8 overlap holds 203 and none of the uint16 overlap holds 3001 (truncation of the float mean); the file's dtype is the input dtype and equals the record's. Exact. | MV, S |
| FI3 | Uncovered voxels (F3) | `overlap_pair` (its two corners); `zstack` with different Z as a stack; `plane_pair` (none) | Voxels uncovered in the axes fused | A voxel is uncovered when, for every included FOV, its centre lies outside that FOV's voxel-centre extent on at least one fused axis by more than 1e-6 of a fusion-grid voxel; every such voxel holds exactly 0 (0.0, not NaN, in float32), in Y and X for a projection and for one-plane FOVs and in Z, Y and X for the stack; on `plane_pair` the voxels inside the first FOV in Y and X hold 50, those inside the second 120, and the voxels at x 8 and 9 hold 0 (the slab is not a coverage axis). Exact. Assumption A13. | MV; S for `overlap_pair` |
| FI4 | Position of a bright voxel (F5, condition 4) | `position_grid` (354) | Per FOV: the output voxel of the maximum, the intensity centroid, the local maximum at `k = 2` and `k = 4` (within 3 output voxels of `v(q*)` in each axis) | At scale 1 the maximum near `q*` lies on the output voxel nearest `q*` and the intensity centroid of its 3 × 3 × 3 neighbourhood lies within 0.001 µm of `q*`; at `k = 2` and `k = 4` (`level1`) each of the nine bright voxels gives a local maximum above 0 and at most `floor(1000/k²)` (250 and 62), within `((k − 1)/(2k) + ½)·S_a` of `q*` on Y and on X (0.291597 and 0.680393 µm). | MV, S |
| FI5 | Library consistency (F6) | `quad` (2102) at `k` 1 and 2, as the projection and the stack | The output level read from each OME-TIFF against a direct `fusion.fuse` | Equal exactly to the array that a direct call of `fusion.fuse` returns for the inputs and arguments the `fusion` section records (the `level1` inputs at `k = 2`, the per-FOV maxima for the projection). Assumption A6. | MV |
| FI6 | The scale factor and the `level1` route (conditions 1 and 3) | `checker_pair` (3D and one-plane) at `k` 1, 2 and 4; `quad` at `k` 1, 2 and 4 (`N` 58, 29 and 15, a partial last block); `fuse_band` at `k = 2` (none) | Recorded `S`, `O`, `N`; index identities; values of `checker_pair` | In Y and X `S = k·s`, `O = o + s·(k − 1)/2`, `N = ceil(n0/k)`; the stack keeps `s_z`, `o_z`, `n0_z`; the projection's Z is the slab `n0_z·s_z`, `o_z + s_z·(n0_z − 1)/2` (one-plane FOVs: the slab of FI1), all within 1e-9 µm in the OME-TIFF; `j = (j⁰ − (k − 1)/2)/k` within 1e-9 and `v = floor(v⁰/k)` exactly at positions at least 1e-6 voxel from a fusion-grid voxel boundary; where only FOV A of `checker_pair` covers, every output voxel holds exactly 100 at `k = 2` and `k = 4` (the block mean of 0 and 200), and the one-plane case's plane equals the 3D case's in Y and X there; FOV B's values are recorded (W-354 found 100). A `scale_yx` whose reciprocal is not whole within 1e-9 is refused (FI10). | MV; S for `fuse_band` |
| FI7 | Projection, stack and one-plane FOVs (F7, condition 2) | `zstack` (equal and different Z, `dimensionality: both`; the one-plane pair); `plane_pair` (none) | Grids of the two files; the projection route; one-plane record | The projection and the stack have equal Y and X `S`, `O` and `N` at the same `k`, bitwise; their values are not compared; for every `q_z` in `[z_lo, z_hi)` the projection's `v_z` is 0; the projection equals the 2D fusion of the per-FOV `project_image` maxima as F6 checks it (FI5); with one-plane FOVs the record says `inputs_2d` true and lists each FOV's `origin_z`, the stack is the one plane and equals the projection's values exactly, and both files have `S_z = z_hi − z_lo` and `O_z = (z_lo + z_hi)/2` (`plane_pair`: `PhysicalSizeZ` 3 and `PositionZ` 1). Exact or 1e-9 µm as stated. | MV |
| FI8 | OME-TIFF layout (F1) | the files of FI1 to FI7, with `quad` at `k = 1` also written with `pyramid_levels: 2` (SubIFDs 29 × 29 and 14 × 14) and at `k = 4` with `pyramid_levels: 3`; `fuse_band` projection and stack at `k = 1` with `pyramid_levels: 2` (none) | Layout checks 1 to 3 of "OME-TIFF layout" | BigTIFF and OME, one series, axes `YX` or `ZYX`, one channel; shape and dtype equal the record's, the OME pixel type is the dtype's; 512 × 512 tiles and `zlib`; the number and shapes of SubIFD levels equal the record's, with `S^(ℓ) = 2^ℓ·S` and `O^(ℓ) = O + S·(2^ℓ − 1)/2` in Y and X and Z kept, within 1e-9 µm (assumption A7 for the odd sizes); `PhysicalSize*` and the `Plane` positions `(O_z + i·S_z, O_y, O_x)` equal the record's within 1e-9 µm; `load_volume` restores that `ImageMetadata`; the `starfinder.sample_output` annotation holds the sample, image name, dimensionality, projection rule, factor and the placement tables' SHA-256; on `fuse_band` the three 512-row bands of tiles are read one at a time. Exact or 1e-9 µm as stated. | MV; S for `fuse_band` |
| FI9 | Standalone OME-Zarr (layout checks and F9) | `quad` and `fuse_band` with `ome_zarr: true`, `ome_zarr_version` `"0.5"` (Z2) and `"0.4"` (Z1) (none) | Layout checks 1 to 3 of "OME-Zarr layout"; values | Zarr format 3 and NGFF 0.5 under Z2, format 2 and 0.4 under Z1; axes `t, c, (z), y, x` with unit `micrometer`; one dataset per recorded level, the levels those of the library's rule (an axis halved while `floor(n/2) > 100`: `fuse_band`'s projection 1408 × 256, 704 × 128, 352 × 128 and 176 × 128; `quad` one level), each dataset's `scale` and `translation` equal to the record's within 1e-6 µm; shape, dtype, chunks (the requested `(1, 1, 1024, 1024)` for a projection and `(1, 1, 16, 512, 512)` for a stack, each clipped to the array's size on each axis) and codec equal the record's; on `fuse_band` a region inside one chunk touches that chunk only (chunk keys counted); the output level equals the OME-TIFF's exactly (F9). | MV; S for `fuse_band` |
| FI10 | Input dtypes and configuration refusals | `quad` copies in uint8 and float32, and in int8, int16, uint32, int32, float64, uint64 and bool; `quad` with one FOV in uint8 and the others in uint16; `zstack` with `projection: sum` (uint16 and float32); the part's sum bound evaluated on grids that declare 256 and 257 planes (uint16) and 65,793 and 65,794 planes (uint8), a check that reads no array; configurations with `scale_yx` 0.3, `pyramid_levels` −1, `images` empty, `images` `("a", "A")`, an image name with a space, `ome_zarr` with `export_spatialdata`, `ome_zarr_version` `"0.6"`, `zarr_format` 4, `labels_scale_yx` 0.3, and `quad` at `k = 4` with `pyramid_levels` 4 (none) | Files, dtypes, raised errors | uint8, uint16 and float32 are fused and written in the input dtype; every other dtype, and mixed dtypes, raise `ValueError` naming the dtype and the allowed list before any file is written; the sum projection is written as float32; its integer bound passes 256 planes of uint16 and 65,793 of uint8 (256 × 65,535 and 65,793 × 255, at most 2^24) and raises `ValueError` naming the bound for 257 and 65,794 planes; each configuration refusal raises `ValueError` from `__post_init__`, and `pyramid_levels` 4 at `N = 15` from `fuse_images` once the grid is known (`floor(log2 15) = 3`), before any file is written; `quad` at `k = 4` with `pyramid_levels` 3 is accepted. Exact. | H; MV for the written files |
| FI11 | The stack alone (`dimensionality: stack`) and the middle plane | `export_folder` (3D) exported by `Dataset.export` with `dimensionality: stack` in its `export` block, once with `projection: max` and once with `projection: sum`, and once with `images: [reference_stain, reference_merged]` and `export_spatialdata: true`; `export_cases`' 2D `assembly_golden` (one-plane FOVs) with `dimensionality: stack` and `export_spatialdata: true`; `zstack` with `fuse_images` alone and `dimensionality: stack`: equal Z (`n_z` 4, even), different Z (`n_z` 5, odd) and the one-plane pair (`n_z` 1) (none) | Files under `export/<sample>/`; the `fusion` section; the store's image elements; plane `i = n_z // 2` read alone | **Files:** for every image name `<image>_stack.ome.tif` exists and no `<image>_projection.ome.tif` exists; the folder holds exactly the default files of EX1 with the stack in place of the projection, and `check_export` passes (R2). With `projection: sum` the stack holds the same array, in the input dtype (not float32): the key has no effect. **Record:** the `fusion` section records `dimensionality` `stack` and lists the stack files only, each with its image name, dimensionality `stack`, factor 2 (the default `scale_yx` 0.5), shape `(n0_z, ceil(n0_y/2), ceil(n0_x/2))` and in Z the fusion grid's `s_z`, `o_z` and `n0_z` (Z is not reduced), within 1e-9 µm. **Store:** with `export_spatialdata: true` the store holds one image element `<image>_stack` per image name and no `<image>_projection`; for multi-plane FOVs it is an `Image3DModel` (`c, z, y, x`) and the file's axes are `ZYX`; for one-plane FOVs it is an `Image2DModel` (`c, y, x`) and the file's axes are `YX` ("The fused image by dataset"). **Middle plane:** `i = n_z // 2` is 2 for the even `zstack` (4 planes), 2 for the odd one (5 planes) and 0 for the one-plane pair, whose file is 2D and is its only plane; `tifffile.imread(path, key=i)` equals plane `i` of the whole stack read from the same file, exactly; on `export_folder` (3D) plane `i = n0_z // 2` of the file equals `[0, i]` of its store element, and on the 2D sample the file's only plane equals `[0]` of its 2D element, exactly; the plane's `PositionZ` equals `O_z + i·S_z` of the record and its Y and X spacing and origin equal the file's, within 1e-9 µm. Exact otherwise. | H, A, MV; SD for the store |

### Assembled label export (LB)

The label export of {doc}`sample-output-contract` ("Assembled label export"), part of the
first delivery: turned on with `fuse_labels: true` (its default is off, "Choices decided
at the gate", 8), at the default `labels_scale_yx` of 0.25. The label image is drawn by
retained cells (LR4, "Choices decided at the gate", 2): each FOV's territory keeps the
cells the assembly retained, as `instance_id`s and whole, and the first positive view in
placement table order is shown, at order 0 and without blending. Each row checks the file
of the `labels` section and, where named, conditions L1 to L6; LB7 checks each case of the
table "What the label image shows".

| # | Check | Fixture (seeds) | Metric | Pass/fail tolerance | Needs |
| --- | --- | --- | --- | --- | --- |
| LB1 | The label export at scale 0.25 (L1, L5) | `label_quad` with `fuse_labels: true` and no `labels_scale_yx` given; the same with `labels_scale_yx: 1`; `label_quad`'s volume variant (none) | `cell_labels.ome.tif` and the `labels` section | The section records the rule `retained_cells`, `labels_scale_yx` 0.25, `fusion_func` `_first_positive`, `interpolation_order` 0 and the route `output_spacing`; the file passes the layout checks of "OME-TIFF layout" with dtype uint32 (L1); its grid in Y and X has spacing `4·s`, origin `o` (not `o + s·(k − 1)/2`) and size `floor((max_f (origin_f + (n_f − 1)·s) − o) / (4·s) + 1e-9) + 1` = 18 × 18, and at scale 1 70 × 70, and its Z is the slab (`S_z` 1, `O_z` 0 for these `plane` territories), all within 1e-9 µm in the file and in the record; the grid the library returned equals the formula grid (L5); the volume variant's label image is ZYX with the fusion grid's Z (`s_z`, `o_z`, `n0_z` 4). | MV |
| LB2 | No new value (L2) | `label_quad` at 0.25 and 1; `label_pair` at 1 and 0.25; `label_scene` at 1 and 0.25 (354, 355) | Nonzero values of the label image | Every nonzero value is the `instance_id` of a cell row whose `assembly_status` is `retained` and belongs to `ids_input`; no value is the `instance_id` of an excluded row, and no value occurs in no input (W-359 `tables/w354_summary.csv` and `tables/two_fov_summary.csv`, rows `retained_cells`, column `values_in_no_input`: 0 on both fixtures at both scales). Exact. | MV; S for `label_scene` |
| LB3 | Drawn by retained cells (L3) | `label_pair` at 1 and 0.25; `label_scene` at 1 and 0.25 (354, 355); `label_quad` and its volume variant at 1 and 0.25 | (a) At each drawn voxel of value `v`: the territory of `v`'s FOV sampled at the voxel's centre by `floor(c + ½)` (Y and X for `plane`, Z, Y and X for `volume`); (b) per scene cell, the measure of W-359 `worker-notes.md` section 3.3 (whole with one ID, incomplete and its three reasons, two or more IDs, absent), computed by the test from the fixture's recipe; (c) on `label_pair`, the outcome of each cell | (a) The sample holds `v`'s cell, the row whose `instance_id` is `v`, at every drawn voxel. (b) Equal to W-359's counts: `label_scene` at 1, 391 cells, 370 whole, 21 incomplete (3 in part under a retained cell of another FOV, 18 beyond their FOV's voxel-centre extent, 0 cut), 0 with two or more IDs, 0 absent; at 0.25, 366, 11 (1, 10, 0), 0, 14; `label_pair` at 1, 13 cells, 9 whole, 2 incomplete, 1 with two IDs, 1 absent; at 0.25, 10, 1, 1, 1 (W-359 `tables/w354_summary.csv` and `tables/two_fov_summary.csv`, rows `retained_cells`). (c) The outcomes of W-359 `tables/two_fov_cells.csv`, rows `retained_cells`: cell 6 two IDs, cell 7 absent, cell 5 one ID and whole, cell 9 with 1 of its 81 voxels under cell 8 at scale 1, cell 10 cut, every other cell whole. Exact; a count that differs goes to Jiahao, not to a looser bound. | MV; S for `label_scene` |
| LB4 | Absent IDs (L4) | `label_quad` at 0.25 (with its 1 × 1 px cells); `label_pair` at 1 and 0.25; `label_scene` at 1 and 0.25 (354, 355); a one-FOV `cells` sample whose territory is all one retained cell of `instance_id` 1, at 0.25 | `ids_input`, `ids_output`, `absent_ids` | Each equals its recomputation: `ids_input` the positive values of the retained-cell images, recomputed from the territory files, the `instance_id` map and the cell table's `assembly_status`; `ids_output` the nonzero values of the label file; `absent_ids = ids_input − ids_output` with its count; `ids_output ⊆ ids_input`; a cell retained by no FOV (`label_pair`'s cell 7) is in neither. On the one-FOV sample `ids_input = ids_output = {1}` and `absent_ids` is empty; the counts of `ids_input`, `ids_output` and `absent_ids` are 391, 391, 0 for `label_scene` at 1 and 391, 377, 14 at 0.25, and 13, 13, 0 for `label_pair` at both scales (W-359 `tables/w354_summary.csv` and `tables/two_fov_summary.csv`, rows `retained_cells`). Exact. | MV; S for `label_scene` |
| LB5 | Empty populations, the conflict count and the section | `empty_cases` (`molecules_only`, the `cells` sample with no cell row); `no_fov_assembly` with `fuse_labels` called alone on its placement; `label_pair_empty` at 1 and 0.25; `label_pair` and `label_pair_conflict` at 1; `label_quad` (none) | The label image, the `labels` section | A `molecules_only` sample writes no file and records `empty`, `no_cells`; a `cells` sample without a cell row, and `label_pair_empty`, write an image of zeros with `ids_input`, `ids_output` and `absent_ids` empty (W-359 `tables/cases.csv`); no included FOV records `empty`, `no_included_fov`; the conflict count, recounted by the test from the cell table's `conflict` flags and the image, is 0 on `label_pair` (cell 6 is retained twice, but neither view holds the other's ownership centroid, so the assembly does not flag it) and 2 on `label_pair_conflict` (A's view of cell 6 whole, B's on the voxels A's does not cover); the section holds the `assembly` section's SHA-256, the rule `retained_cells`, per FOV with a retained cell the territory path and SHA-256 read and its number of retained cells, and no entry for a FOV without one, the library version, `fusion_func`, `interpolation_order`, the route, `labels_scale_yx`, the grid returned and the formula grid, the file's path, SHA-256, shape and dtype, the counts of `ids_input` and `ids_output`, `absent_ids`, the conflict count, and wall time, CPU time and peak RSS. Exact. | MV |
| LB6 | ID bound (L6) | `id_bound` (none) | Check 6 of the entry | 16,777,217 cell rows raise `ValueError` naming the count before anything is written; 16,777,216 rows (2^24, the largest `instance_id` float32 holds exactly, with no shift) pass. Exact. | H |
| LB7 | The cases of "What the label image shows" | the fixtures of the table below | as below | as below; every case of that table has one line. | MV; S for `label_scene` |

Cases of {doc}`sample-output-contract`, "Assembled label export", table "What the label
image shows", at scale 1 unless stated; `F_v` is the set of output voxels whose nearest
voxel of `v`'s FOV, over its footprint, holds `v`'s cell (W-359 `worker-notes.md` section
3.3).

| # | Case | Fixture (seeds) | Metric | Pass/fail tolerance |
| --- | --- | --- | --- | --- |
| LB7.1 | A retained cell that no retained cell of another FOV overlaps | `label_pair` (cells 1, 2, 3, 4, 12, 13); `label_quad` (every retained cell but FOV 4's box 1, the one box that FOV 1's retained cell overlaps, which follows LB7.3) | Voxels showing its `instance_id` | Exactly `F_v`: one ID, drawn whole. Exact. |
| LB7.2 | An excluded cell | `label_pair` (A's views of cells 4 and 9, B's of 3, 5, 8 and 11); `label_quad` (FOV 1's 4 × 4 px cell, owned by FOV 2) | The `instance_id` of the excluded row; its territory's voxels | The `instance_id` appears at no voxel and is not in `ids_input`; on its voxels the image shows the cell another FOV retained there where one lies (on A's view of cell 4, B's cell 4; on FOV 1's 4 × 4 px cell, FOV 2's box), and 0 elsewhere. Exact. |
| LB7.3 | Two retained cells of different FOVs whose territories overlap | `label_pair` (A's cell 8, B's cell 9) | The voxels they share; each cell's voxels | The shared voxel shows A's cell 8, the view first in placement table order; B's cell 9 shows on 80 of its 81 voxels, incomplete (W-359 `tables/two_fov_cells.csv`). Exact. |
| LB7.4 | A cell retained by two FOVs | `label_pair` (cell 6) at 1 and 0.25; `label_pair_conflict` (cell 6) at 1 | IDs showing the cell | Two `instance_id`s: A's on all of `F_v` (49 voxels at scale 1), B's on its voxels where A's does not lie (37 at scale 1, W-359 `tables/two_fov_cells.csv`); two IDs at 0.25 too. On `label_pair_conflict` both rows carry `conflict` and both IDs show. Exact. |
| LB7.5 | A cell retained by no FOV | `label_pair` (cell 7) at 1 and 0.25 | IDs showing the cell; flags | Neither `instance_id` appears; both rows are excluded `not_owner` and flagged `unclaimed`; the cell's area holds 0. Exact. |
| LB7.6 | A cell whose ownership centroid lies exactly on an ownership boundary | `label_pair` (cell 5, centroid x 79.5 px) | Status of both rows; IDs showing the cell | A retains it (the tie goes to the FOV first in table order), B excludes it `not_owner`; one ID, whole on its 62 voxels, also on those beyond x 79.5 in B's ownership region. Exact. |
| LB7.7 | A retained cell cut by the border of its FOV's image | `label_pair` (cell 10 of B; cell 11) | Voxels showing it; flags | Cell 10 shows on its 35 imaged pixels only and is `fov_edge`; nothing is drawn for it from another FOV. Cell 11, cut by B's left border, is excluded in B (`fov_edge`) and drawn whole from A. Exact. |
| LB7.8 | A retained cell that reaches the first or last voxel of its FOV, at a fractional offset | `label_scene` (354, 355) | The cells incomplete beyond their FOV's voxel-centre extent; the missing voxels | 18 such cells at scale 1 and 10 at 0.25 (W-359 `tables/w354_summary.csv`, column `incomplete_fov_edge_outside_sampling`; the cells in `tables/w354_incomplete_edges.csv`); every voxel of `F_v` that does not show `v` lies in one row or column just beyond the FOV's voxel-centre extent, within half a voxel of its footprint. Exact. |
| LB7.9 | A placeholder FOV | `label_quad` with FOV 4 a `placeholder` (a `PlacementResult` constructed in the test with the same origins, `nominal_z` 0, approved; its other FOVs `metadata_only`); `label_scene` with its centre FOV a placeholder, at 1 (354, 355) | The image, the territory files loaded, `ids_input` | The placeholder's territory is not loaded (counted through the loader); no `instance_id` of its rows appears; every output voxel whose centre lies only in its footprint holds 0; a retained cell of FOVs 1 to 3 that reaches into its footprint is drawn as its FOV holds it. On `label_scene`: 13 cells absent, each retained by no FOV, 0 with two or more IDs, 0 values in no input, and 378 retained observations (W-359 `tables/case_placeholder_summary.csv`, row `retained_cells` at 1). Exact. |
| LB7.10 | Background | `label_pair`; `label_pair_empty`; `label_quad` | Voxels where no FOV draws a retained cell; voxels outside every included FOV | 0 at each; on `label_quad`, every voxel uncovered by F3's rule applied to the label grid holds 0. Exact. |
| LB7.11 | An assembly made with `conflicts=False` | `label_pair` and `label_quad` assembled with `conflicts=False` | The image; the territory files read; the conflict count | The image equals the one from the assembly with `conflicts=True` exactly; with the territory path null in the `assembly` section, the file the §2.9 loader resolves from the recorded assignment folder is read and its SHA-256 checked; the conflict count is null. Exact. |
| LB7.12 | A cell retained by no FOV that carries no flag | `label_pair_merged` | The label image; `assembly_status`, `assembly_reason` and `assembly_flags` of the rows of cell 31 | No `instance_id` of cell 31 appears in the label image; its rows are excluded `not_owner` and none carries `unclaimed`; its area shows B's merged cell 32. Exact. |

### SpatialData export (SD)

The eight round-trip conditions S1 to S8 of "Read-back checks", SpatialData, of
{doc}`sample-output-contract`; each row names the one it implements. Image and labels
elements come from the files of the FI and LB rows.

| # | Check | Fixture (seeds) | Metric | Pass/fail tolerance | Needs |
| --- | --- | --- | --- | --- | --- |
| SD1 | Elements, omissions and format (S1) | `export_cases`, one per row of "Empty populations" (the row "No included FOV" through `no_fov_assembly` and `export_spatialdata` called alone), with `zarr_format` 3 (Z2) and 2 (Z1); a sample name with a space (none) | Store contents against the `spatialdata` section | Names and kinds equal the `written` entries; no two names equal ignoring case; every element the configuration asks for is `written` or `omitted`, and each `omitted` reason (`empty`, `no_cells`, `no_included_fov`, `not_requested`, `part_not_run`) holds when recomputed from the sample tables, the `fusion` and `labels` sections and the configuration, the element being absent; `store: null` exactly when every element is omitted, and then no `sample.zarr` exists; one image element per fused image and no standalone OME-Zarr under `export/<sample>/`; Zarr format 3 with raster version string `0.5-dev-spatialdata` under Z2 and format 2 with `0.4-dev-spatialdata` under Z1 (assumption A10); the sample name outside the element-name rule is refused before anything is written. Exact. | SD, MV |
| SD2 | Points (S2) | `export_cases` (none) | `molecules` against the sample molecule table | Joined on `(spot_namespace, spot_id)`: `x, y, z` equal `sample_x/y/z` exactly, `fov_index_z/y/x` equal the §2.9 `z, y, x`, every other source column equals its source (NaN equals NaN, null equals null), no `sample_x/y/z` column remains, the dtypes equal those the section lists; with `molecules_unplaced`, every source row appears exactly once; a sample molecule table without rows has both omitted. Exact. | SD |
| SD3 | Tables preserve their source (S3) | `export_cases`; a cell table with a column `Size` beside `size` (none) | `cells`, `cells_unplaced`, `molecules_unplaced`, `expression` | Every source column of the cell table and every row exactly once in `cells` and `cells_unplaced`; `molecules_unplaced` likewise; `expression` equals `sample.h5ad` in `X`, `layers`, `var` and `obsm` exactly, in every source `obs` column and in every source `uns` key; each column's dtype after the round trip equals the one the section records; when `expression` is omitted, `sample.h5ad` has no row; the table with two column names equal ignoring case is refused naming the key. Exact. | SD |
| SD4 | Export metadata, separately (S4), and the refusal of a table that already has it | `export_cases`; for the refusal, an `export_cases` folder (3D `assembly_golden`, `fuse: false`) whose `sample.h5ad` the test rewrites with `uns["spatialdata_attrs"]` set to `{"region": "cell_centroids", "region_key": "region", "instance_key": "instance_id"}` and whose `assembly` section records the rewritten file's SHA-256, so that `load_assembly` accepts it, then `export_spatialdata` (none) | Added columns and keys; the raised error, the folder | `region` (categorical, every value `cell_centroids`) and `instance_id` (`uint32`) on `cells` and `expression`, `uns["spatialdata_attrs"]` equal to `{"region": "cell_centroids", "region_key": "region", "instance_key": "instance_id"}`, and no other key or column their source lacks; the `_unplaced` tables carry only `instance_id` or `cell_instance_id` and no `spatialdata_attrs`; points, shapes and raster elements carry no feature or instance key. **Refusal:** the source key reaches `TableModel.parse`, which refuses a table that already has `uns["spatialdata_attrs"]` ({doc}`sample-output-contract`, "Metadata spatialdata adds": `spatialdata/models/models.py` lines 1138 to 1142 of 0.6.1); the export raises that error (its type and message recorded by the test), does not overwrite or drop the source key, and leaves no `sample.zarr` and no staging folder, the other files of the folder unchanged. Exact. | SD |
| SD5 | Links, accounting and the `cell_centroids` element (S5) | `export_cases` (3D and 2D `assembly_golden`, expanded); a copy of the 3D result with one retained cell's `size_voxels` set to 0, passed to `export_spatialdata` (none) | IDs, counts, circle centres and radii, errors | `instance_id` of `cells` and `cells_unplaced` together is `1..N` in the cell table's order; `cells` and `cell_centroids` are written or omitted together and the index of `cell_centroids` equals the `instance_id` of `cells`; each circle's centre equals `(sample_ownership_centroid_x, sample_ownership_centroid_y)` of its row exactly, and its radius equals, within 1e-12 relative, `(3·size_voxels·s_z·s_y·s_x / (4π))^(1/3)` for a `volume` territory and `(size_voxels·s_y·s_x / π)^(1/2)` for a `plane` territory, with its FOV's grid spacing; every molecule's `cell_instance_id` equals its cell's `instance_id`, null without a cell; every `expression` row is a `cells` row with `assembly_status == "retained"` and `status == "kept"`; the molecule counts by `assignment_status` and by `assembly_status` and reason equal the section's and the `assembly` section's; the cell with `size_voxels` 0 makes the export raise `ValueError` naming the cell, and no `sample.zarr` and no staging folder remain. Exact or 1e-12 relative as stated. | SD |
| SD6 | Calibration and alignment (S6, condition 5) | `export_cases` (`assembly_golden` 3D exported with `dimensionality: both` at `scale_yx` 1 and 0.5, and `assembly_golden` 2D, whose one-plane FOVs give a slab, with `dimensionality: stack`); `label_quad` exported with `fuse_labels: true`; `position_grid` with its store placement and assembly (354) | Element transforms, extents, point lookup | The transform of each image and labels element equals `Scale(S)` then `Translation(O − S/2)` on its axes within 1e-9, with `S` and `O` of its file's output level (for the one-plane stack element, `Scale(z_hi − z_lo)` and `Translation(z_lo)` in Z, from its record); `get_extent` gives `[O − S/2, O + (N − ½)·S)` per axis; points and shapes have the identity; a point at `q` falls in intrinsic voxel `floor(j + ½)`; on `position_grid` the intrinsic voxel of each `q*` holds the file's value there. | SD, MV; S for `position_grid` |
| SD7 | Values (S7) | `export_cases` (`assembly_golden` 3D with `dimensionality: both`); `fuse_band` with `dimensionality: both` and its store assembly; `label_quad` exported with `fuse_labels: true` (none) | Each image and labels element against its OME-TIFF's output level | Equal exactly, in value and dtype; one level per element. | SD, MV; S for `fuse_band` |
| SD8 | Hashes, staging and lookup (S8) | `export_cases`; an export interrupted after its first element (a test hook raising); a folder whose `assembly` section was edited after the export (none) | Store files, the folder, `lookup_source` | Every store file's SHA-256 equals the section's; the interrupted export leaves no `sample.zarr` and no staging folder, and the files of the earlier parts are unchanged; `lookup_source(folder, spot=…)` and `(cell=…)` return the FOV, the assignment folder, its `assignment.json` SHA-256 and the row's position in the §2.9 table, from the `assembly` section, and raise `ValueError` on the edited folder; the `spatialdata` section holds every item of "The `spatialdata` section of `export.json`" (status, coordinate system, the SHA-256 of the sections it read, the spatialdata version and formats, `store`, per element its status, reason or path, rows or shape, transform, `spatialdata_attrs` and column dtypes, the counts, the column renames and every store file's SHA-256). Exact. | SD |

### Workflow (WF)

| # | Check | Fixture (seeds) | Metric | Pass/fail tolerance | Needs |
| --- | --- | --- | --- | --- | --- |
| WF1 | Schema equals the configs | `wf_configs` (none) | `workflow/schemas/config.schema.yaml` | The `stitching` method list equals `STITCHING_METHODS` plus `bigstitcher`; the `assembly` keys equal the `AssemblyConfig` fields plus `assignment` and `table_format`; the `export` keys equal the fields of `ExportConfig`, `FusionConfig`, `LabelConfig` and `SpatialDataExportConfig` as the block names them (`labels_scale_yx`, `spatialdata_zarr_format`); the configurations validate. Exact. | H |
| WF2 | Legacy configurations unchanged | `wf_configs` without the new blocks (none) | Rules, outputs | The four legacy stitching rules and the legacy assembly rules run unchanged (`test_assignment_workflow.py`, `test_workflow_stitching_golden.py`, `test_workflow_assembly_golden.py` pass unchanged); the adapter's legacy translation gives `StageAxisMapping(x="+x", y="+y", source="legacy")`. Exact. | H |
| WF3 | The new blocks | `wf_configs` with the blocks, through a stub `snakemake` object; `xml_links` and its copy without view 3's `Stitching Transform`; `golden_stitching_setups` and `hand_records_setups` imported with their view maps (none) | Adapter calls, files, errors | `_run_stitching` calls `Dataset.stitch` with the configured values (spied; its result is not used) and `_run_export` runs `Dataset.export` on `assembly_golden`'s constructed placement saved beforehand in `stitching/<sample>/`, writing `export/<sample>/`; with an `assembly` block the per-FOV files have no overlap filter and no `global_*` (W1); a stage file without `stage_axes` raises naming the key; `method: bigstitcher` without `bigstitcher_save_dir` raises; without an `export` block the defaults run; the `export_sample` rule's inputs are the sample's per-FOV assignment outputs and its `placement.json`, and its declared output is `export/{sample}/export.json`; with `multiview_stitcher` not importable the rule succeeds and the record says the fuse part did not run; parsing the rules without `fiji_path` succeeds, and the BigStitcher engine rule raises naming `fiji_path` only when it runs; its success check accepts `xml_links` and fails on the copy, naming the view, and on a missing file, naming the path, whatever Fiji's exit status; the legacy `tile_config` writer, fed from the record of the import of `golden_stitching_setups` and `hand_records_setups` (each with its view map), writes byte for byte what `legacy_tile_config` computes for `golden_stitching` and `hand_records`, for `fallback` `None` and `"nominal"` alike. Exact. | H |

### What the default tier covers and what it does not

PL1 to PL12, PL14, PL15, PL17, PL18, AS1 to AS13, AS15 to AS17, EX1 to EX5, FI1 and FI3 (their
routine parts), FI5 to FI11 (their routine parts), LB1 to LB7 (their `label_quad` and
`label_pair` parts), SD1 to SD8 (their routine parts) and WF1 to WF3 run in the routine tier
on hand-built geometry. PL4's and PL18's scene parts, SC1 and EE1 to EE5 run in the routine
tier once the generator addition exists (`sample-tiny-v1`, its variant without Z jitter and its
single-plane variant are inside the routine bound, {doc}`synthetic-sample-scene`). PL16, AS14's three processes,
MV1 to MV6, EE6, FI2, FI4, the `overlap_pair`, `fuse_band`, `label_scene` and
`position_grid` parts of the other rows, and any test measured at 5 s or more are `slow`.
MV1 to MV6, EE6 and every FI, LB, SD and EX row that writes a fused or label file need the
`multiview-stitcher` extra; SD1 to SD8 need the `spatialdata` extra; AS7, AS8, AS11 to AS13,
AS15, EX1 to EX4 need the `anndata` extra. Nothing here measures how well a backend places
real tiles, how well a fused image represents a sample or how well a sample's cells are
represented; those are questions for E07 and for annotation.

**Which checks need the generator addition (W-327):** PL4 and PL18 (their scene parts),
MV3, SC1, EE1 to EE6. **Which run on hand-built geometry alone:** every PL, AS, EX, FI, LB,
SD and WF row except those scene parts. **Which read a file outside the repository:** none;
the records on the share are read by RX1 to RX7 of the real examples.

### Resource plan

The bounds are the §2.10 resource exceptions recorded in W-323 (agreed with Jiahao on
2026-10-06 as the limits the implementation batches may use) for tests:

* **Routine tests**: synthetic tiles cut from one scene of at most 16×256×256; one CPU,
  4 GiB, 1800 s. Every routine fixture is below it: the largest are `checker_pair` (union
  4 × 75 × 105), `quad` (4 × 58 × 58), `label_quad` (1 × 70 × 70), `label_pair` (1 × 64 × 160), `export_folder` (three
  FOVs of 8×48×64), the scene of `sample-tiny-v1` (sample grid 6×63×63) and its
  single-plane variant (1×63×63); `id_bound` holds one column of 16,777,217 uint32 values (64 MiB) and no image. Each
  check is expected well under a second beyond the one-time import of an extra.
* **`slow` tests**: a synthetic 3 × 3 grid of tiles of 16×256×256; one CPU, 4 GiB,
  1800 s. `cut_synthetic` holds one 16×256×256 float32 scene (4 MiB) and nine tiles of
  14×120×120 uint16; W-324 measured 12 to 78.5 s wall and 0.29 to 1.02 GiB peak RSS per
  fixture row at one thread, most of it the import of multiview-stitcher (W-324 notes
  section 2.5). `scene_small` is nine FOVs of 14×96×96, estimated below 1 GiB
  ({doc}`synthetic-sample-scene`, "Presets and resources"). `position_grid` is nine FOVs
  of 16×128×128 uint16, `label_scene` nine of 1×112×112 uint32, `overlap_pair` two of
  1×32×256, and `fuse_band` six of 16×256×256 uint8 (6 MiB); each is at most a 3 × 3 grid
  of tiles of at most 16×256×256. W-354 fused its fixtures at one thread in 0.6 to 1.5 GiB
  peak RSS (`tables/runs.csv`, rows `fixture-*`). Every `slow` check runs at one thread,
  MV6 included.
* Each test run records wall time and peak RSS with `/usr/bin/time -v`; a run above a
  bound is a failure to report, not a reason to raise the bound.

## Whole-sample real examples

After the implementation, the three whole samples of W-323 are run through the new path
once, as inspections, and the checks against the records on the share run there. They are
not a comparison and rank nothing; no truth exists for them. Every input is a file named in
W-323 `20261005T2110Z-input-inventory/inputs.md` or `20261005T1930Z-fiji-baseline/inputs/`,
or in the W-324 run directory, read-only, with its SHA-256 checked against those records
where they list one. Nothing is written into a data directory or a run directory of an
earlier issue.

| Sample | Placement | Source image for fusion | Volumetric | Fused image written | Figures drawn from |
| --- | --- | --- | --- | --- | --- |
| LN sample1 (64 FOVs of 50×1496×1496 uint8, DAPI, plus 8 blank grid cells) | The 2024 `dataset.xml` through `import_bigstitcher` (no Fiji rerun), and `multiview_stitcher` in the starting configuration from the grid start | the stitching input tiles, `<name>=file:<pattern>` | yes | the 3D stack alone (`dimensionality: stack`, dense 3D tissue) at `scale_yx` 0.5, from each of the two placements; no projection | the middle plane of its stack, `n_z // 2` ({doc}`sample-output-contract`, "The fused image by dataset"), never a maximum projection; its stack for the seam sections |
| tissue-2D (56 FOVs of 3072×3072 uint8, `ref_merged`) | `multiview_stitcher`, starting configuration, grid start (no BigStitcher reference exists; the Grid/Collection file is monitored only) | `reference_merged` | no (one plane) | the default: the 2D maximum projection at `scale_yx` 0.5 | its projection |
| cell-culture-3D (70 FOVs of 30×1496×1496 uint8, `ref_merged`) | as tissue-2D | `reference_merged` | yes | the default projection at 0.5; the stack of one 2 × 2 block of FOVs at `scale_yx` 1 | its projection; the block's stack for the seam views |

**Grids.** The tile files carry no calibration (W-323 `inputs.md`), so each FOV's grid is
declared from the raw-acquisition calibration recorded there (0.194398 µm in x and y for LN
and cell-culture-3D, 0.094635 µm for tissue-2D, 0.346255 µm in z), recorded as
`declared`, and the package functions are called with these `StitchingInput`s directly
rather than through `Dataset.stitch`, which reads grids from saved reference images.

### Checks against the records on the share

These run in the real-example issue, through the package's public functions, from a script
in its run directory; no test in the repository reads these files. Paths are relative to
`/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/runs/`; W-324 means
`W-324/20261006T063540Z-5cb2f421/` and W-323 means `W-323/20261005T1930Z-fiji-baseline/`.

| # | Check | Files read | Pass/fail tolerance |
| --- | --- | --- | --- |
| RX1 | The converter on the LN stage file | W-323 `inputs/sample1.maf`, `inputs/grid.csv`; W-324 `parity/script_grid_ln.csv` | With `StageAxisMapping(x="+x", y="+y")` and `GridSnap(overlap=0.1, tile_shape_yx=(1496, 1496), spacing_yx=(0.194398, 0.194398))` the cells equal the column and row of `script_grid_ln.csv` for every `PositionID`; the stage step is 262.0 µm. Exact. |
| RX2 | Neighbour pairs of the LN grid start | W-324 `parity/script_grid_ln.csv` | 213 pairs, diagonals included, the number multiview-stitcher's overlap graph found (W-324 notes section 2.1). Exact. |
| RX3 | The import of the LN `dataset.xml` | W-323 `inputs/dataset_2024.xml`, `inputs/grid.csv`, `inputs/tile_config_sample1.csv`; W-324 `parity/prototype/tile_config_ln/T2_sum.csv`, `parity/script_grid_ln.csv`, `parity/script_tile_config_ln.csv` | The five items of {doc}`stitching-contract` ("What the import of the LN `dataset.xml` must reproduce"): (1) raw translations of all 72 setups equal `T2_sum.csv` as float64; (2) setup `i` maps to row `i` of `script_grid_ln.csv`, 64 acquired, 8 padding views with no row; (3) the legacy view with `img_col = img_row = 1496` equals `script_tile_config_ln.csv` and `tile_config_sample1.csv` in all 72 rows and 16 columns, minima −5460, −6135 and −69 before the shift; (4) 145 links, all kept, none on a padding view, one component, 64 `image_supported` FOVs; (5) z in slices is world z ÷ 1.7894736842105263 and origins use the reference spacing. The legacy `tile_config` writer fed from the record writes `script_tile_config_ln.csv` byte for byte; the saved import holds `view_map.csv` and the SHA-256 of the macro and of `dataset.xml`. The review state and the residual of each of the 145 links are recorded (they decide whether the 2024 placement needs review under V1). Exact. |
| RX4 | BigStitcher's result on W-324's cut fixtures, through the import | W-324 `fiji-fixtures/synthetic_3x3/dataset.xml`, `grid.csv`; `fiji-fixtures/real_2x2/dataset.xml`, `grid.csv`; `scripts/fixture_data.py` | Every view has a `Stitching Transform`; the true integer origins recomputed by the `grid_layout` rule (synthetic seed 324, step 64, base 4, z nominal 1, jitter ±4 and ±1; real seed 325, step 684, base 26, z nominal 2, jitter ±8 and ±2); max absolute error at most 1.05 voxels per axis after removing the mean. |
| RX5 | The import's fallback on a Fiji file | W-324 `fiji-fixtures/synthetic_3x3/dataset.xml`, copied to the issue's run directory with every `PairwiseResult` of one corner view removed | As PL11: `fallback="nominal"` gives `metadata_only` at nominal, flag `fallback`; `fallback=None` gives `failed`; `method_translation_*` keeps BigStitcher's translation; the legacy view is byte-identical for both. Exact. |
| RX6 | The hand-built records against W-324's outputs | W-324 `parity/fixtures/` (the two stage files, the two `dataset.xml` and their grids), `parity/script_grid_maf_2x3_jitter.csv`, `parity/script_grid_maf_3x2_gap.csv`, `parity/script_tile_config_xml_2x2_blank.csv`, `parity/script_tile_config_xml_3x2_positive.csv` | The files of `hand_records` that the tests write parse to the same values as W-324's fixtures, and the frozen helpers' outputs on them equal W-324's `script_*` files: the in-test expectations of PL1, PL12 and WF3 are those W-324 recorded. Exact. |
| RX7 | Thread count of `multiview_stitcher` | the LN stitching inputs | The LN placement at one thread and at four gives bitwise identical translations (the part of MV6 that the `slow` tier cannot run). |

**Step 1: inventory of the assembly inputs.** W-323 inventoried the stitching inputs only;
the molecule and cell tables of the three samples are not inventoried. They are expected
among each dataset's analysed per-FOV outputs: for LN sample1 under
`Data/Analyzed/2024-03-08-Hongyu-Covid_LN/` (the outputs of the 2024 workflow run whose
stitching inputs W-323 lists), for tissue-2D and cell-culture-3D under
`Data/Analyzed/sample-dataset-output/<dataset>/`, beside the `images/ref_merged/` inputs:
the per-FOV molecule files the legacy workflow writes (the reads or goodSpots tables) and,
for cells, any per-FOV label images. None of these paths was opened for this page. The
first step of the issue that runs the examples is their inventory: per FOV, the path, the
row and column counts and the SHA-256 of each file, recorded before any assembly runs, and
read only where its authorization names them. Until then every assembly figure below is
an estimate, and an assembly without inputs is reported as not run. Cell inputs: the
whole sample in `molecules_only` mode, plus cellular assembly on a 2 × 2 block of FOVs at
one seam of each volumetric sample from existing label files imported with the §2.9 mask
import, where the datasets hold them (W-331 choice 11, taken as recommended). Cells
therefore exist only in those regions: one 2 × 2 block of LN sample1 and one of
cell-culture-3D, each only if the inventory finds its label files; tissue-2D has no cell
input, and every whole sample is assembled without cells.

**Steps and their limits.** The §2.10 resource exceptions recorded in W-323 set a separate
limit for each step; W-323 states that its numbers for the real examples are estimates
from file sizes, not measurements, and that placement uses the CPU only.

| Step | What runs | Limit |
| --- | --- | --- |
| Placement | Per sample and backend: LN sample1 through the import of the 2024 `dataset.xml` and through `multiview_stitcher`; tissue-2D and cell-culture-3D through `multiview_stitcher`; once, W-324's real_2x2 cut fixture of one LN tile; the LN `multiview_stitcher` run repeated at one thread (RX7) | 4 CPU threads, 32 GiB RSS, 7200 s per backend and sample; CPU only |
| Assembly | Per sample: the assemble part of `Dataset.export` in `molecules_only` mode from the inventoried molecule tables, and with cells (`conflicts: true`, the default, so that `conflict` and `unclaimed` are computed) on the 2 × 2 blocks where cell inputs exist, on each placement of the placement step (their `nominal_z` are those of the tile position table, one layer), approved with `approve_placement` where its review is `required`, the state and reasons recorded | one CPU, 16 GiB RSS, 3600 s per sample |
| Sample outputs | `Dataset.export` with the default output (projection at scale 0.5 and `sample.h5ad`) of tissue-2D and cell-culture-3D; LN sample1 with `dimensionality: stack` and `scale_yx` 0.5 and `sample.h5ad`, once from the BigStitcher placement, with `export_spatialdata: true` so that its stack is also the OME-Zarr image element of the SpatialData store, and once from the `multiview_stitcher` placement, the stack OME-TIFF alone; no LN projection; the stack of the cell-culture-3D 2 × 2 block; the label export at 0.25 on the cellular 2 × 2 blocks, granted | 4 CPU threads, 32 GiB RSS; new artifacts at most 20 GiB per run, written to the artifact mount; no time limit is listed, and each run is timed |

The exceptions grant the 3D stack of one whole sample, LN sample1. Since that stack is
LN's only fused image, it is fused once from each of its two placements; this is the
operator's reading of the grant (the same sample and the same kind of output twice),
which Jiahao confirms at the gate. The seam views through depth of cell-culture-3D come
from the 3D stack of a 2 × 2 block of its FOVs around one grid corner, fused by the same
call from those four FOVs' placement rows (a bounded region of the sample-outputs step),
not from a whole-sample stack. The label export on real labels runs on the cellular 2 × 2
blocks only, as a bounded region of the sample-outputs step, as granted at the second
round of the W-329 review.

**Assumptions the real examples verify** ({doc}`sample-output-contract`, "Assumptions"):
A1, by fusing the LN stack from lazily loaded tiles, from each placement, and recording
the peak RSS beside W-354's 12,946,284 KiB with every tile loaded; A2, by the label export on real labels of
the 2 × 2 blocks (LB2 and LB4 on its file); A3 and A8, by writing the LN stack OME-TIFF
(2,247,808,158 bytes uncompressed in W-354's run) with the tiled zlib writer and reading it
back with `check_export`; A9, by the store's `cell_labels` element where the label export
ran. A4 stays open: no whole-sample stack of cell-culture-3D is granted.

**Outputs to inspect**, per sample, in the run directory (at most ten PNGs in total; nine
are planned: a placement overview and an ownership-regions figure per sample, the seam
views of LN and of cell-culture-3D, and the LN seam sections). Every figure of LN sample1
other than its seam sections is drawn from the middle plane of its stack, plane
`n_z // 2` (plane 33 of a 67-plane stack), and none from a maximum projection; the
figures of tissue-2D and cell-culture-3D use their projection.

* **Placement overview** (one PNG per sample): the FOV footprints at their origins,
  coloured by status, links coloured by quality with dropped links dashed, correction
  arrows (scaled and labelled), and the review state with its reasons, over the sample's
  fused image (for LN the middle plane of the stack from the same placement); the
  placement record's `fovs.csv` and `links.csv`.
* **Seam views** (LN and cell-culture-3D, one PNG each): at three seams (a column seam, a
  row seam and a corner), with the ownership boundary drawn. For LN, XY crops of the middle
  plane of its whole stack at `scale_yx` 0.5. For cell-culture-3D, through depth: XY crops
  of the 2 × 2 block's stack at `scale_yx` 1 at the first, middle and last quarter planes,
  and one XZ and one YZ section across the seam.
* **LN seam sections** (LN sample1 only, one PNG): sections through the depth of the LN
  stack, not its middle plane, to show a displacement in Z between neighbouring FOVs,
  which no single plane can show. One XZ section across the column seam and one YZ
  section across the row seam of the LN seam views, each between the same two acquired
  FOVs in both placements: one row (XZ) or one column (YZ) of voxels of the stack at
  `scale_yx` 0.5, with every plane and no projection across the section, the one
  nearest the Y (XZ) or X (YZ) of the midpoint of the two FOVs' footprint centres, over
  the overlap of their footprints and 256 voxels of the stack on each side of it. Both
  placements side by side: the sections of the stack fused from the BigStitcher import
  and of the stack fused from the `multiview_stitcher` placement, each at its own
  placement's origins, four panels drawn at one physical scale in Z and along the section
  (µm). Marked on each panel: the O1 ownership boundary between the two FOVs (a vertical
  line, ownership being decided in Y and X, {doc}`assembly-contract`, "The Z axis"), the
  outline of each FOV's footprint in the section (along the section and in Z, from its
  origin, so that a difference of the two FOVs' Z origins shows as offset outlines), and
  the middle plane `n_z // 2` of that stack as a horizontal line labelled with its index.
* **Ownership regions with retained and excluded observations** (one PNG per sample): the
  O1 regions over the sample's image (for LN the middle plane of its stack, for tissue-2D
  and cell-culture-3D the default projection), molecules coloured by assembly status and
  reason, retained and excluded cell centroids where cells exist, and the `conflict`,
  `unclaimed`, `fov_edge` and `outside_owner_region` flags.
* **Accounting tables**: the `fovs` table and `record["counts"]` (inputs, retained, excluded
  by reason, flag counts) per sample and placement, as CSV and JSON.
* **Flag frequencies** (one CSV and the table below in the issue's report): how often the
  flags `conflict`, `unclaimed` and `fov_edge` occur, and over which cells, and the number
  of overlapping retained pairs defined below, per sample, placement and region where
  cells exist. These are the first measurement of these flags on real samples (W-326
  notes); with them Jiahao decides whether cells with several observations need more
  than the present rule, which keeps one observation by centroid ownership and matches or
  merges nothing. They are recorded, never gating.
* **The sample outputs**: each sample's `export/<sample>/` folder with `export.json`,
  `tables/`, `sample.h5ad` and its fused image, with `check_export` passing: for tissue-2D
  and cell-culture-3D the projection OME-TIFF at scale 0.5; for LN sample1 the stack
  OME-TIFF at `scale_yx` 0.5 from each placement and, in the export from the BigStitcher
  placement, `sample.zarr/` with the stack as its one image element; the label files of
  the 2 × 2 blocks.
* **Monitored agreement** (LN only): the per-axis differences between the imported
  BigStitcher placement and the `multiview_stitcher` placement, against the monitored
  bounds of "Tolerances and where they come from"; reported, never gating. The real_2x2
  cut fixture of W-324 (one LN tile, seed 325, noise seed 1100 + tile) is run once with
  `multiview_stitcher` against MV1's bound.

The flag table the real-example issue fills in. `conflict` is counted over the retained
cells, `unclaimed` over the cells excluded `not_owner`, and `fov_edge` over every cell row
and over the retained cells ({doc}`assembly-contract`, "Flags"); each count is given with
the number of cells it is counted over. The last column is not a flag: it counts the
overlapping retained pairs defined below.

| Sample, placement | Where cells exist | Cell rows | Retained | Excluded `not_owner` | `conflict` (of retained) | `unclaimed` (of `not_owner`) | `fov_edge` (of all rows; of retained) | Overlapping retained pairs (of different FOVs) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LN sample1, BigStitcher import | one 2 × 2 block at a seam, if the inventory finds its label files | counted | counted | counted | counted | counted | counted | counted |
| LN sample1, `multiview_stitcher` | the same block | counted | counted | counted | counted | counted | counted | counted |
| cell-culture-3D, `multiview_stitcher` | one 2 × 2 block around a grid corner, if the inventory finds its label files | counted | counted | counted | counted | counted | counted | counted |
| tissue-2D, `multiview_stitcher` | none: no cell input; the sample is assembled `molecules_only` | 0 | 0 | 0 | not applicable | not applicable | not applicable | not applicable |

A block without label files gets the row "none" with the reason, as tissue-2D does.

**Overlapping retained pairs.** The flags `conflict` and `unclaimed` are tests at
ownership centroids, so a cell can be retained by two FOVs, or by none, without a flag
({doc}`sample-output-contract`, "Assembled label export"). This count shows how often
retained cells of different FOVs cover the same place:

* **Cells compared:** the retained cells (`assembly_status == "retained"`) of the
  cellular assembly of one row of the table, that is of one sample, one placement and its
  2 × 2 block, for each two FOVs of the block whose footprints overlap. Excluded cells are
  not compared, and no cell is compared with a cell of its own FOV.
* **Grid:** the label grid at scale 1 (`labels_scale_yx` 1: the fusion grid's spacing and
  origin in Y and X, and its Z for `volume` and `extended` territories). Each FOV's
  retained-cell image (step 1 of the placement by retained cells,
  {doc}`sample-output-contract`, "Assembled label export") is placed on it alone, at
  interpolation order 0, as the label export places it: an output voxel takes the value
  of the FOV's voxel nearest its centre, inside the FOV's voxel-centre extent. Only
  voxels where both FOVs' images are defined can contribute, so the count is computed two
  FOVs at a time on the box where their footprints overlap.
* **Overlap:** a cell `a` of FOV `f` and a cell `b` of FOV `g ≠ f` overlap when at least
  one output voxel shows `a` in `f`'s placed image and `b` in `g`'s. Each such pair counts
  once, however many voxels the two share.
* **What it counts:** a cell retained by two FOVs whose two observations share a voxel
  counts one pair, flagged `conflict` or not. Two different neighbouring cells whose
  territories only touch, with no voxel in common, count nothing; two different cells
  whose segmentations by two FOVs disagree so that they share a voxel count one pair. The
  count is therefore an upper bound on the cells retained twice whose two observations
  share a voxel; a cell retained twice whose observations share none (for instance under a
  placement error larger than the cell) is not counted, and a cell retained by three or
  four FOVs at a corner counts one pair for each two of them whose observations share a
  voxel. A cell retained by no FOV is in no pair. On `label_pair` the count is 2: cell 6,
  retained by both FOVs (LB7.4), and A's cell 8 with B's cell 9, two different cells that
  share one voxel (LB7.3); the flag `conflict` marks neither, and cell 7 (LB7.5) is in no
  pair.
* **Where:** each row of the table that has cells: LN sample1 from each of its two
  placements and cell-culture-3D from `multiview_stitcher`, on their cellular 2 × 2
  blocks; "not applicable" for tissue-2D and for a block without label files.

The count is a measurement of the real-example issue, computed by that issue's own script
from the cell table and the territory files that the label export of the same block reads
({doc}`sample-output-contract`, "Assembled label export"), within the limits of the
sample-outputs step, with its wall time and peak RSS recorded. It is not a flag: it
changes no status, gates nothing and decides nothing, and it is reported beside the flag
counts for Jiahao. The product records no such count: no column of the cell table, no
field of `export.json` and no output of `Dataset.export` holds it.

**Resources per sample and step.** Placement figures are W-324 measurements and output
figures W-354 measurements ({doc}`sample-output-contract`, "Resources": four threads,
every tile loaded first, the library's uncompressed TIFF writer); assembly figures are
estimates from {doc}`assembly-algorithms` ("Cost": about 100 bytes per molecule row with
the §2.9 columns, and at most two territory images in memory at a time), pending the
inventory of step 1. Every run is measured with `/usr/bin/time -v`; a run above its step's
limit is recorded as `resource_stop` and not retried with a higher limit.

| Sample | Placement (limit 4 threads, 32 GiB, 7200 s) | Assembly (limit one CPU, 16 GiB, 3600 s), an estimate | Default outputs (limit 4 threads, 32 GiB), W-354 | Stack and export (same limit; artifacts at most 20 GiB per run) |
| --- | --- | --- | --- | --- |
| LN sample1 | import: one XML parse; `multiview_stitcher`: 145 to 182 s, 7.0 GiB peak RSS, loading 58 to 81 s | molecule rows unknown until step 1: about 1 GB per 10⁷ rows, plus two territory images of 50×1496×1496 uint32 (0.45 GB each) when cells are assembled; time measured | none: LN writes no projection | the stack alone at `scale_yx` 0.5 from each placement. From `multiview_stitcher`: 67 × 6,166 × 5,441 uint8, 241.65 s and 12,946,284 KiB (12.35 GiB) peak RSS with every tile loaded (A1 to verify lazily), 2,247,808,158 bytes written uncompressed (W-354 `tables/runs.csv`, row `ln-mvs-stack-s05-a2`). From the BigStitcher import, not measured: its grid at 0.5 is 6,167 × 5,435 in Y and X (W-354 `tables/ln_two_placements_grids.csv`), so 2,245,682,215 bytes of voxels with the same 67 planes (the Z extent depends on the placement's z corrections), and time and memory are expected near the measured run. The store with the stack as one level, written once: 156.48 s, 1,662,724 KiB (1.59 GiB) peak, 678,925,650 bytes (W-354 `tables/runs.csv`, row `sd-write-ln-stack-single-aszarr`). About 5.2 GB of artifacts before compression in total, 2.9 GB in the larger run |
| cell-culture-3D | 65 to 78 s, 4.69 GiB | as LN, with territory images of 30×1496×1496 uint32 (0.27 GB each) | 6,818 × 4,788: 22.9 s, 0.62 GiB | the 2 × 2 block's stack at `scale_yx` 1, about 45 × 2,842 × 2,842 (0.34 GiB uncompressed; the Z extent depends on the placement's z corrections); under 0.5 GiB of artifacts |
| tissue-2D | 44 to 49 s, 0.83 GiB | as LN, with territory images of 3072×3072 uint32 (38 MB each) | 11,291 × 9,845: 24.2 s, 1.42 GiB | none (one plane) |

By the assembly estimate a sample fits in its 16 GiB limit up to about 10⁸ molecule rows;
the inventory of step 1 replaces the estimate with the row counts. The tiled zlib OME-TIFF
writer adds to W-354's times (A8). At most ten PNGs are written, nine are planned, and
temporary files go to local scratch; nothing is written into a data directory.

## Implemented checks (W-373)

The checks are in the modules of `src/python/test/` listed below. The issues that delivered
the code added most of them with it; `test_stitching_validation.py` adds EE6, the full
chain on `scene_small`, and the test that keeps this table true. That test reads the rows
of the tables of "Checks" from this page, with each case line LB7.1 to LB7.12 of LB7 as a
row of its own, and checks that this table has exactly those rows and the RX rows of the
real examples; that every row other than an RX row names at least one test of its module
that pytest collects (a name ending in `*` stands for every test whose name starts so, a
name with `[…]` for one parametrized case); that none of them is marked as an expected
failure; that none is skipped where the `multiview-stitcher`, `spatialdata` and `anndata`
extras are importable, as in the batch environment; and that a row whose tier is
`default` names no `slow` test and a row whose tier is `slow` names only `slow` tests. A
test of a module that is not named here is an additional check of the code it tests.

Every tolerance is the one of the table "Checks", unchanged. The rows that need an extra
(column "Needs") skip without it, naming the extra; the batch environment holds the extras,
so no named test is skipped there. The `slow` tests run in the `slow` tier of the
`stitching` subsystem (`slow and not learned and not (benchmark or synthetic or
registration or spot_finding)`), except the `slow` tests of `test_synthetic_sample_scene.py`,
which carry the `synthetic` marker. PL13 of the first version is RX4. The RX rows read the
records on the share and are run by the real-example issue (W-376), not by a test.

| # | Module and tests | Tier |
| --- | --- | --- |
| PL1 | `test_stitching_positions.py`, `test_stage_axis_mapping_*`, `test_unsnapped_stage_table_keeps_the_stage_positions`, `test_snapped_cells_equal_the_legacy_grid`, `test_golden_stage_file_covers_its_eight_positions`, `test_a_remaining_step_tie_goes_to_the_smaller_step`, `test_regular_grid_*`, `test_column_major_puts_fov_k_at_column_k_div_rows`, `test_supplied_tables_convert_units`, `test_leica_failures_name_the_item`, `test_grid_parameter_rules`, `test_supplied_table_failures`, `test_table_rules_hold_for_direct_construction` | default |
| PL2 | `test_stitching_neighbors.py`, `test_stub_grid_pairs_and_boxes_use_each_grid_size`, `test_touching_boxes_pair_only_with_a_tolerance`, `test_z_ranges_must_intersect_for_volumes_only`, `test_pairs_do_not_depend_on_grid_cells_or_row_order`, `test_eight_by_eight_grid_has_210_pairs`, `test_irregular_stage_positions_follow_the_rule`, `test_unequal_shapes_use_each_grid` | default |
| PL3 | `test_stitching_placement.py`, `test_index_to_sample_rule_and_its_inverse`, `test_contract_example_of_the_index_rule`; `test_stitching_bigstitcher.py`, `test_pl3_*` | default |
| PL4 | `test_stitching_placement.py`, `test_fractional_translations_give_the_g1_gauge_bitwise`, `test_scene_origins_equal_the_g1_truth` | default |
| PL5 | `test_stitching_placement.py`, `test_method_translations_stay_in_micrometres`; `test_stitching_bigstitcher.py`, `test_pl5_units_use_the_calibration_and_the_reference_spacing` | default |
| PL6 | `test_stitching_placement.py`, `test_yx_inputs_give_z1_grids_and_keep_the_method_z`, `test_mixed_geometry_is_refused_naming_the_fovs`, `test_grids_outside_the_sample_frame_are_refused`, `test_a_spec_without_dimension_2_refuses_z1_inputs` | default |
| PL7 | `test_stitching_placement.py`, `test_non_square_tiles_place_with_their_own_sizes`; `test_stitching_neighbors.py`, `test_stub_grid_pairs_and_boxes_use_each_grid_size`; `test_stitching_bigstitcher.py`, `test_pl7_golden_legacy_view_keeps_img_col_for_the_y_extent` | default |
| PL8 | `test_stitching_placement.py`, `test_irregular_footprints_are_placed_and_joined`; `test_stitching_neighbors.py`, `test_irregular_stage_positions_follow_the_rule`, `test_unequal_shapes_use_each_grid` | default |
| PL9 | `test_stitching_placement.py`, `test_a_tile_with_only_nan_quality_links` | default |
| PL10 | `test_stitching_placement.py`, `test_components_and_the_reference_component`, `test_an_absent_image_is_missing`, `test_correction_bound_boundary`, `test_residual_boundary_and_optimizer_removed_links`, `test_approval_records_the_tables_and_changes_no_origin`, `test_review_placement_is_idempotent_and_keeps_matching_approvals`, `test_a_present_fov_with_a_non_finite_translation_is_failed_and_keeps_a_component`, `test_a_method_without_a_finite_translation_fails_every_fov`, `test_a_disconnected_fov_with_a_finite_translation_keeps_the_fallback_rule`, `test_the_reference_component_holds_placed_fovs_only`, `test_present_fovs_joined_by_no_kept_link_have_their_own_component`, `test_every_stub_variant_follows_the_component_rule`; `test_stitching_assembly_retention.py`, `test_as9_review_required_is_refused_until_approved` | default |
| PL11 | `test_stitching_placement.py`, `test_stage_position_fallback_for_a_disconnected_tile`; `test_stitching_bigstitcher.py`, `test_pl11_fallback_in_the_import` | default |
| PL12 | `test_stitching_bigstitcher.py`, `test_pl12_*` | default |
| PL14 | `test_stitching_entry.py`, `test_save_and_load_round_trip`, `test_a_changed_table_is_refused_naming_it`, `test_an_interrupted_save_leaves_no_partial_folder`; `test_stitching_bigstitcher.py`, `test_pl14_import_persistence` | default |
| PL15 | `test_stitching_entry.py`, `test_stitch_refuses_bad_input_before_calling_the_method`, `test_stitch_and_the_supplied_placement_touch_no_file`, `test_registry_uses_the_shared_mechanism`, `test_the_package_registers_only_multiview_stitcher`, `test_spec_rules_and_a_missing_dependency`, `test_importing_the_module_imports_no_backend` | default |
| PL16 | `test_stitching_entry.py`, `test_stub_path_is_deterministic_across_processes` | `slow` |
| PL17 | `test_stitching_entry.py`, `test_dataset_stitch_*` | default |
| PL18 | `test_stitching_supplied.py`, `test_without_origins_each_fov_sits_at_its_nominal_position`, `test_with_origins_the_supplied_values_are_kept_bitwise`, `test_origins_equal_to_the_nominal_positions_give_the_same_table`, `test_scene_truth_placement`, `test_a_table_whose_nominal_z_differ_is_still_a_placement`, `test_failures_name_the_fovs`; `test_stitching_assembly_retention.py`, `test_pl18_*` | default |
| MV1 | `test_stitching_multiview.py`, `test_known_answer_cut_fixture[MV1]` | `slow` |
| MV2 | `test_stitching_multiview.py`, `test_known_answer_cut_fixture[MV2]` | `slow` |
| MV3 | `test_stitching_multiview.py`, `test_fractional_jitter_on_the_synthetic_scene` | `slow` |
| MV4 | `test_stitching_multiview.py`, `test_hand_built_cases` | `slow` |
| MV5 | `test_stitching_multiview.py`, `test_wrapper_rules_on_the_cut_fixture`, `test_no_nan_quality_link_is_kept_and_nan_is_recorded`, `test_with_no_link_left_every_fov_is_failed_and_the_result_is_saved`, `test_config_defaults_are_the_starting_configuration`, `test_config_refuses_each_invalid_field_at_construction`, `test_spec_declares_the_contract_capabilities`, `test_a_tile_below_16_px_in_y_or_x_is_refused` | `slow`; the configuration, specification and shape parts default |
| MV6 | `test_stitching_multiview.py`, `test_translations_are_bitwise_identical_in_two_processes` | `slow` |
| AS1 | `test_stitching_assembly_ownership.py`, `test_as1_*` | default |
| AS2 | `test_stitching_assembly_ownership.py`, `test_as2_*` | default |
| AS3 | `test_stitching_assembly_retention.py`, `test_as3_retained_scene_cells` | default |
| AS4 | `test_stitching_assembly_retention.py`, `test_as4_*` | default |
| AS5 | `test_stitching_assembly_retention.py`, `test_as5_*` | default |
| AS6 | `test_stitching_assembly_retention.py`, `test_as6_*`; `test_stitching_assembly_accounting.py`, `test_as6_a_broken_result_is_refused`, `test_the_valid_results_pass_both`, `test_own_states_has_counted_cells_in_every_compartments_state`, `test_the_changes_leave_the_valid_results_alone` | default |
| AS7 | `test_stitching_assembly_retention.py`, `test_as7_*` | default |
| AS8 | `test_stitching_assembly_retention.py`, `test_as8_*` | default |
| AS9 | `test_stitching_assembly_retention.py`, `test_as9_*` | default |
| AS10 | `test_stitching_assembly_retention.py`, `test_as10_*` | default |
| AS11 | `test_stitching_assembly_outputs.py`, `test_as11_*` | default |
| AS12 | `test_stitching_assembly_outputs.py`, `test_as12_*` | default |
| AS13 | `test_stitching_assembly_outputs.py`, `test_as13_*`, `test_as11_and_as13_without_anndata` | default |
| AS14 | `test_stitching_assembly_outputs.py`, `test_as14_permuted_molecule_rows_give_the_same_per_key_results`, `test_as14_three_single_thread_processes_give_identical_tables` | default; the three processes `slow` |
| AS15 | `test_stitching_export_entry.py`, `test_assemble_part`, `test_assemble_off_without_a_saved_assembly`, `test_assemble_part_input_files`, `test_assemble_part_from_molecule_tables`, `test_molecule_tables_are_the_molecules_csv_rows` | default |
| AS16 | `test_stitching_assembly_ownership.py`, `test_as16_*` | default |
| AS17 | `test_stitching_assembly_ownership.py`, `test_as17_*` | default |
| SC1 | `test_synthetic_sample_scene.py`, `test_single_plane_scene_sc1` | default |
| EE1 | `test_stitching_assembly_scene.py`, `test_ee1_per_fov_assignment_equals_the_sampled_truth` | default |
| EE2 | `test_stitching_assembly_scene.py`, `test_ee2_*` | default |
| EE3 | `test_stitching_assembly_scene.py`, `test_ee3_accounting_on_the_scene` | default |
| EE4 | `test_stitching_assembly_scene.py`, `test_ee4_guarantees_against_sample_level_truth` | default |
| EE5 | `test_stitching_assembly_scene.py`, `test_ee5_comparison_with_sample_level_truth` | default |
| EE6 | `test_stitching_validation.py`, `test_ee6_*` | `slow` |
| EX1 | `test_stitching_export_entry.py`, `test_default_export`, `test_two_image_names`, `test_parts_called_alone`; `test_stitching_labels.py`, `test_ex1_the_label_part_runs_third`; `test_stitching_spatialdata.py`, `test_ex1_*` | default |
| EX2 | `test_stitching_export_entry.py`, `test_checks_before_writing`, `test_check_saved_assembly_of_another_placement`, `test_check_fusion_inputs`, `test_check_file_source_shape`, `test_check_spacing_and_planes`; `test_stitching_labels.py`, `test_lb6_id_bound_through_the_entry` | default |
| EX3 | `test_stitching_export_entry.py`, `test_fuse_without_the_extra`, `test_fuse_with_a_source_image_missing`; `test_stitching_labels.py`, `test_ex3_*`; `test_stitching_spatialdata.py`, `test_ex3_*` | default |
| EX4 | `test_stitching_export_entry.py`, `test_folder_equals_its_record`, `test_interrupted_export`, `test_assemble_off_and_fuse_on` | default |
| EX5 | `test_stitching_fusion_files.py`, `test_fusion_section_items` | default |
| FI1 | `test_stitching_fusion_grid.py`, `test_fused_grid_*`, `test_fuse_alone_without_an_included_fov`; `test_stitching_fusion_files.py`, `test_one_included_fov_is_its_source`; `test_stitching_fusion_slow.py`, `test_fuse_band_grid_layout_and_bands` | default; `fuse_band` `slow` |
| FI2 | `test_stitching_fusion_slow.py`, `test_overlap_pair_values`; `test_stitching_fusion_files.py`, `test_written_in_the_input_dtype` | `slow`; the dtype written on `quad` default |
| FI3 | `test_stitching_fusion_files.py`, `test_uncovered_voxels_hold_zero`, `test_plane_pair_values_and_slab`; `test_stitching_fusion_slow.py`, `test_overlap_pair_values` | default; `overlap_pair` `slow` |
| FI4 | `test_stitching_fusion_slow.py`, `test_position_of_a_bright_voxel` | `slow` |
| FI5 | `test_stitching_fusion_files.py`, `test_files_equal_a_direct_library_call` | default |
| FI6 | `test_stitching_fusion_grid.py`, `test_output_level_geometry`, `test_configuration_refusals`; `test_stitching_fusion_files.py`, `test_checker_pair_block_means`, `test_quad_partial_last_block`; `test_stitching_fusion_slow.py`, `test_fuse_band_grid_layout_and_bands` | default; `fuse_band` `slow` |
| FI7 | `test_stitching_fusion_files.py`, `test_projection_and_stack`, `test_plane_pair_values_and_slab` | default |
| FI8 | `test_stitching_fusion_files.py`, `test_ome_tiff_layout`; `test_stitching_fusion_slow.py`, `test_fuse_band_grid_layout_and_bands` | default; `fuse_band` `slow` |
| FI9 | `test_stitching_fusion_files.py`, `test_standalone_ome_zarr`, `test_check_export_counts_chunk_reads`; `test_stitching_fusion_slow.py`, `test_fuse_band_ome_zarr` | default; `fuse_band` `slow` |
| FI10 | `test_stitching_fusion_grid.py`, `test_configuration_refusals`, `test_configuration_accepts`, `test_sum_bound`, `test_refusals_of_fuse_before_any_file`; `test_stitching_fusion_files.py`, `test_written_in_the_input_dtype`, `test_sum_projection_is_float32` | default |
| FI11 | `test_stitching_fusion_files.py`, `test_stack_alone_and_middle_plane`; `test_stitching_export_entry.py`, `test_stack_alone_through_the_entry`, `test_stack_of_one_plane_fovs`; `test_stitching_spatialdata.py`, `test_fi11_the_stack_alone_in_the_store` | default |
| LB1 | `test_stitching_labels.py`, `test_lb1_*` | default |
| LB2 | `test_stitching_labels.py`, `test_lb2_lb3_label_quad`, `test_lb2_lb3_lb4_label_pair`, `test_label_scene` | default; `label_scene` `slow` |
| LB3 | `test_stitching_labels.py`, `test_lb2_lb3_label_quad`, `test_lb2_lb3_lb4_label_pair`, `test_label_scene` | default; `label_scene` `slow` |
| LB4 | `test_stitching_labels.py`, `test_lb4_*`, `test_lb2_lb3_lb4_label_pair`, `test_label_scene` | default; `label_scene` `slow` |
| LB5 | `test_stitching_labels.py`, `test_lb5_*` | default |
| LB6 | `test_stitching_labels.py`, `test_lb6_*` | default |
| LB7 | `test_stitching_labels.py`, `test_lb7_*`, `test_label_scene` (the case lines below) | default; `label_scene` `slow` |
| LB7.1 | `test_stitching_labels.py`, `test_lb7_1_a_retained_cell` | default |
| LB7.2 | `test_stitching_labels.py`, `test_lb7_2_an_excluded_cell` | default |
| LB7.3 | `test_stitching_labels.py`, `test_lb7_3_two_retained_cells_whose_territories_overlap` | default |
| LB7.4 | `test_stitching_labels.py`, `test_lb7_4_a_cell_retained_by_two_fovs` | default |
| LB7.5 | `test_stitching_labels.py`, `test_lb7_5_a_cell_retained_by_no_fov` | default |
| LB7.6 | `test_stitching_labels.py`, `test_lb7_6_a_centroid_on_an_ownership_boundary` | default |
| LB7.7 | `test_stitching_labels.py`, `test_lb7_7_a_cell_cut_by_its_fovs_border` | default |
| LB7.8 | `test_stitching_labels.py`, `test_label_scene` | `slow` |
| LB7.9 | `test_stitching_labels.py`, `test_lb7_9_a_placeholder_fov`, `test_lb7_9_label_scene_with_a_placeholder` | default; `label_scene` `slow` |
| LB7.10 | `test_stitching_labels.py`, `test_lb7_10_background` | default |
| LB7.11 | `test_stitching_labels.py`, `test_lb7_11_an_assembly_made_without_conflicts` | default |
| LB7.12 | `test_stitching_labels.py`, `test_lb7_12_a_cell_retained_by_no_fov_without_a_flag` | default |
| SD1 | `test_stitching_spatialdata.py`, `test_sd1_*` | default; the case `fov_not_placed` with Zarr format 2 `slow` |
| SD2 | `test_stitching_spatialdata.py`, `test_sd2_points` | default |
| SD3 | `test_stitching_spatialdata.py`, `test_sd3_*` | default |
| SD4 | `test_stitching_spatialdata.py`, `test_sd4_*` | default |
| SD5 | `test_stitching_spatialdata.py`, `test_sd5_*` | default |
| SD6 | `test_stitching_spatialdata.py`, `test_sd6_sd7_golden_3d_both`, `test_sd6_golden_2d_stack_slab`, `test_sd6_sd7_label_quad`, `test_sd6_position_grid` | default; `position_grid` `slow` |
| SD7 | `test_stitching_spatialdata.py`, `test_sd6_sd7_golden_3d_both`, `test_sd6_sd7_label_quad`, `test_sd7_fuse_band` | default; `fuse_band` `slow` |
| SD8 | `test_stitching_spatialdata.py`, `test_sd8_*`, `test_check_export_finds_a_store_that_differs_from_its_section` | default |
| WF1 | `test_stitching_workflow.py`, `test_wf1_*` | default |
| WF2 | `test_stitching_workflow.py`, `test_wf2_*`; `test_assignment_workflow.py`, `test_a18_the_legacy_outputs_equal_the_frozen_helper`; `test_workflow_stitching_golden.py`, `test_stage_file_grid`, `test_script_writes_the_helper_csv`; `test_workflow_assembly_golden.py`, `test_the_adapter_writes_the_legacy_values`, `test_the_empty_branch` | default |
| WF3 | `test_stitching_workflow.py`, `test_wf3_*` | default |
| RX1 | Run by the real-example issue (W-376), not by a test | real examples |
| RX2 | Run by the real-example issue (W-376), not by a test | real examples |
| RX3 | Run by the real-example issue (W-376), not by a test | real examples |
| RX4 | Run by the real-example issue (W-376), not by a test | real examples |
| RX5 | Run by the real-example issue (W-376), not by a test | real examples |
| RX6 | Run by the real-example issue (W-376), not by a test | real examples |
| RX7 | Run by the real-example issue (W-376), not by a test | real examples |

## Results of the real examples

W-376 ran the plan of "Whole-sample real examples" on 2026-10-08, at revision `70ad3d6`, in
the run directory
`/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/runs/W-376/20261008T044605Z-9a24f906`
(`worker-notes.md`, `results/manifest.json`, `results/tables/`). These are inspections:
nothing is ranked, no accuracy is claimed and no default changed. W-389 ran the tissue-2D
example on 2026-10-08, at revision `b4bf7bb`, which holds W-388's tolerance of the
adapter, in the run directory
`/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/runs/W-389/20261008T093243Z-b342bf8d`
(`worker-notes.md`, `results/manifest.json`, `results/tables/`); the tissue-2D rows and
sentences below are from that run. W-391 ran the three samples again through the unchanged
`Dataset.export` from the files as saved on 2026-10-08, at revision `f6ffdd9`, which holds
W-390, in the run directory
`/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/runs/W-391/20261008T203138Z-905169ae`
(`worker-notes.md`, `results/manifest.json`, `results/comparison/`), from the saved
placements of W-389 (tissue-2D) and W-376 (cell-culture-3D, LN sample1 BigStitcher); no
placement was made again. The sentences marked W-391 are from that run. W-398 ran LN
sample1 through the unchanged `Dataset.export` from the dataset's own folder, and the
cell-culture-3D projection again, on 2026-10-09, at revision `5ffee25`, which holds W-397,
in the run directory
`/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/runs/W-398/20261009T045906Z-2d4054c3`
(`worker-notes.md`, `results/manifest.json`, `results/comparison/`), from the same saved
placements of W-376; the sentences marked W-398 are from that run. W-431 measured where the
LN fusion's time goes and tried one change to how `fuse_images` builds a source's array, on
2026-10-10, at revision `1803299`, in the run directory
`/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/runs/W-431/20261010T090737Z-e03e2c65`
(`worker-notes.md`, `results/manifest.json`, `results/profile/`); the change did not meet
that issue's rule and was taken out again, so the package is as before; the sentences marked
W-431 are from that run. W-432 tried the change W-431's measurements point to, one task and
one chunk per page with no split into strips, under the same rule, on 2026-10-10, at revision
`afe8272` with the change applied, in the run directory
`/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/runs/W-432/20261010T173323Z-ecdf51b6`
(`worker-notes.md`, `results/manifest.json`, `results/profile/`); the change met the rule and
stays in the package; the sentences marked W-432 are from that run.

* **Inventory.** The 190 molecule tables (LN sample1 64, 3,782,580 rows; tissue-2D 56,
  5,353,714; cell-culture-3D 70, 2,458,865) equal the operator's inventory of 2026-10-07 in
  path, row count and SHA-256. No dataset holds a label image per FOV, so cells exist only
  on the two 2 × 2 blocks W-382 segmented, and none for tissue-2D.
* **Records.** RX1 to RX7 pass as their rows state: 213 pairs; the import's five items
  (145 links, one component, 64 `image_supported`, review `not_needed`, residuals 0.038 to
  0.943 µm); BigStitcher's cut-fixture results within 1.05 voxels (largest 1.0000000020, z,
  synthetic_3x3); the bitwise identical LN placement at one and four threads.
* **Placements.** LN sample1 from both backends and cell-culture-3D: every FOV
  `image_supported`, review `not_needed`. In W-376 `stitch` raised on tissue-2D, because the
  `multiview_stitcher` adapter refused a pairwise 2D affine whose linear part is off the
  identity by 1.1e-16. With W-388's tolerance, tissue-2D from the grid start places every
  one of its 56 FOVs `image_supported`, one component: 181 pairs, 84 pruned, 97 registered,
  of which 87 are kept, 3 dropped below 0.7 and 7 removed by the optimizer. Its review is
  `required` (`optimizer_removed_link`, 13 FOVs). The placement was approved with
  `approve_placement`, with the state and reasons recorded, so that the export could run.
* **tissue-2D assembly (`molecules_only`).** Identities 1 to 5 hold. Of 5,353,714
  molecules, 160,257 are retained, 20,750 excluded `not_owner` and 5,172,707 (96.6 %)
  excluded `outside_grid`. Every `outside_grid` molecule is outside in z only. The tiles
  are single-plane maximum projections, while each FOV's molecule table spans 30 planes
  (one-based z 1 to 30), and only the 181,007 molecules of the first plane fall on the
  one-plane grid. W-391, with the molecules of a one-plane grid assigned by y and x (W-390):
  identities 1 to 5 hold; 4,703,832 retained, 649,882 excluded `not_owner`, 0 `outside_grid`
  (none outside in y or x). Every changed row is one of the 5,172,707 molecules that were
  `outside_grid` in z alone, with its sample y and x unchanged: 4,543,575 became retained
  and 629,132 `not_owner`; the 181,007 molecules of the first plane keep their status. The
  projection equals W-389's pixel for pixel.
* **Monitored LN agreement.** Within every monitored bound: |d| at most 8.04 px in x, 1.77 px
  in y, 2.77 slices in z; trend in x 1.69 px per column; residual at most 3.14, 1.13 and
  1.31. Of the 111 registered `multiview_stitcher` links, the filter of 0.7 drops 14, all
  of which agree with the BigStitcher placement (qualities 0.634 to 0.690), and keeps none
  that disagrees. real_2x2: at most 0.75 voxel, within MV1's 2.
* **Inputs prepared for the package, without changing it.** The W-382 assignment grids say
  `micrometer` with no direction, which `assemble` refuses; the blocks were assembled from
  copies whose `assignment.json` says µm with the identity direction, every other file
  byte-identical. The LN and cell-culture-3D tiles read as axes `IYX`, which the fusion
  refuses; it read ZYX copies of the same pixels on local scratch. The tissue-2D tiles are
  named by FOV and read as `YX`, so its fusion read the original files; only its grids were
  declared, as for the other samples. In W-391 none of this was needed: no function of the
  package was replaced, the grids came from the dataset's `voxel_size_xy` and
  `voxel_size_z`, the fusion read the culture's `IYX` tiles and the 64 LN images under their
  FOV names as saved (each equal byte for byte to its stitching tile), and the W-382 grid
  metadata was accepted as saved.
* **From the files as saved (W-391).** tissue-2D and cell-culture-3D ran from their own
  folders: `check_export` passes, and the cell-culture-3D tables equal W-376's row for row in
  every column (sample coordinates within 0.0 µm) and its projection pixel for pixel. Two
  inputs are not where the export's layout looks, so those parts are not run as saved:
  `Dataset.export` reads `checkpoints/<fov>/assignment/<name>/` under the dataset's root,
  while W-382 saved the folders under `checkpoints/<sample>/<fov>/` of its run directory;
  and it reads every grid from `images/ref_merged/<fov>.tif`, which LN sample1 does not have.
  From the dataset roots the export refuses both blocks and LN sample1. Through a root of
  symbolic links to the saved files (nothing copied or rewritten), both blocks and the LN
  assembly equal W-376's in counts, flags and every table row for row. The LN stack
  (BigStitcher placement, `scale_yx` 0.5) was not written: the fusion read the 64 images
  lazily from the share, 295 GB for 4.7 GB of images, and the worker stopped the run after
  2,911 s, having misjudged its progress from the writer's bursts, with about 45 % of
  W-376's file size written; no limit was reached. By the operator's decision it is not run
  again in this issue, so its comparison with W-376's stack is not made.
* **LN sample1 from its own folder (W-398).** With W-397, the dataset configuration names
  the reference image (`reference_image` set to `images/flamingo/DAPI/{fov_id}.tif`) and the
  voxel sizes, and `Dataset.export` ran with the LN folder as its output root: no function
  replaced, no input copied, rewritten or linked. Each of the 64 images equals byte for byte
  the SHA-256 W-391 recorded for it. The assembly (`molecules_only`, BigStitcher placement):
  identities 1 to 5 hold, `check_export` passes, the counts equal W-376's (3,782,580 →
  3,204,425 retained, 578,155 `not_owner`), and the four tables equal W-376's byte for byte.
  The stack (`scale_yx` 0.5, no projection, no store), fused from the images where they are
  saved, each tile read once, was written: 66 × 6,167 × 5,435, 662,701,545 bytes, equal to
  W-376's pixel for pixel and with the same recorded SHA-256; the folder passes
  `check_export`. While the fusion ran, the record said `fuse: not_run` and
  `complete: false`. The run ended by itself after 6,142 s, below its limit of 10,800 s. The
  process read 41.7 GB (W-391's stopped run: about 295 GB), but the fusion part took 5,903 s
  against W-376's 1,468 s from local copies, and the stack file reached the sizes W-391's run
  had reached at about the same times. The cell-culture-3D projection from its tiles as
  saved took 41.2 s (W-391 229.9 s, W-376 26.8 s), equal to W-376's pixel for pixel, and its
  tables equal W-376's byte for byte.
* **Where the LN fusion's time goes (W-431).** The 64 LN images are PackBits pages stored in
  strips of 23 rows, and tifffile's Zarr view gives them chunks of 1 × 23 × 1,496. The
  fusion's reader rebuilds these chunks from its page tasks, so the graph of the fused stack
  holds 4,093,517 tasks, one output chunk depends on 6,267 of them, and one band of the writer
  runs about 7,200 after dask's optimisation. On one CPU the 26 bands of planes 33 and 34 took
  163.9 s: 60 % outside the tasks (dask's handling of each band's graph), 22 % in the
  library's level-1 reduction of the strips and 13 % in its fusion. Building the graph read
  10.2 GB from the share, besides the 14.9 GB that the export reads for the FOVs' grids before
  the fusion; most of the 31.6 GB that W-398 read in its first seven minutes is these two
  reads. With each source file given to the fusion as one piece, the graph holds 111,255
  tasks (47 for one output chunk) and the bands are equal pixel for pixel, but they took
  638.4 s, because each band computes the level-1 reduction of every whole tile it touches,
  and all 64 tiles are held at once. The whole stack with that change, from the share: fusion
  part 5,010 s and 15,237 CPU s, 13.13 GiB at peak, 36.6 GB read during the export call
  (W-398: 5,903 s, 7,363 CPU s, 4.74 GiB); the stack equals W-376's pixel for pixel. On the
  synthetic grid of the reading tests the change holds all nine tiles at once. The rule of
  W-431 is not met (the fusion part above 2,950 s, and not fewer than all tiles' worth of
  pages held), so the change was taken out. A file written as W-376 wrote its local copies
  (uncompressed, axes `ZYX`) has one Zarr chunk per page, so W-376's fusion most likely
  read one chunk per page, not one per file; that run's copies were not read again. With the
  change, the cell-culture-3D projection, whose reading the change does not touch, took
  36.3 s and equals W-376's pixel for pixel.
* **One chunk per page (W-432).** The fusion's reader gives the library each page of a file
  read by pages as one chunk, its one task, without the rechunk to the 23-row strips, and
  opens no Zarr view of such a file; a file read as one task for the whole file is read as
  before. On the same 26 bands, one CPU, run one after the other: the graph of the fused stack
  holds 197,455 tasks against the present code's 4,093,517 (171 for one output chunk against
  6,267), and the bands took 91.9 s against 172.7 s (W-431's present code: 163.9 s), of which
  28.2 s outside the tasks against 104.5 s; the bands are equal pixel for pixel, and building
  the graph read 5.1 GB instead of 10.2 GB. Each page is still decoded once, and the pages
  held at one time are those of the present code (at most 207 on the LN stack by the plan's
  walk; 28 and 25 on the 16-plane grid of the reading tests). The whole stack from the share:
  fusion part 1,511 s and 2,469 CPU s, 3.27 GiB at peak, the run 6.92 GiB, 36.6 GB read during
  the export call (W-398: 5,903 s; W-431, one piece per file: 5,010 s; W-376 from local copies:
  1,468 s); the stack equals W-376's pixel for pixel. The rule is met, so the change stays.
  The cell-culture-3D projection, whose reading the change does not touch, took 39.4 s and
  equals W-376's pixel for pixel.

Measured resources, in place of the estimates of "Resources per sample and step" (the wall
time each run recorded for its work, without its closing `check_export`, and the run's peak
RSS; every run's full figures are in `results/tables/resources.csv`; assembly at one CPU,
the others at four threads):

| Sample | Placement | Assembly (`molecules_only`; cellular block) | Default outputs | Stack and export |
| --- | --- | --- | --- | --- |
| LN sample1 | import: under 2 s; `multiview_stitcher`: 140.6 s, 7.57 GiB (one thread: 151.1 s) | BigStitcher: 121.7 s, 2.90 GiB; `multiview_stitcher`: 124.0 s, 2.91 GiB; blocks 46.7 and 47.2 s, 1.96 GiB; W-391 (BigStitcher, link root): 180.4 s, 2.91 GiB; block 51.8 s, 1.97 GiB; W-398 (BigStitcher, own folder): 183.6 s, 2.90 GiB | none: LN writes no projection | the stack OME-TIFF from each placement, zlib, with `check_export`: BigStitcher 66 × 6,167 × 5,435, fusion 1,468 s, 2.53 GiB, 662,701,545 bytes, and the store 141.5 s, 729,704,694 bytes (run 1,955 s, 6.00 GiB); `multiview_stitcher` 68 × 6,166 × 5,441, fusion 1,564 s, 2.53 GiB, 669,758,230 bytes (run 1,708 s, 6.08 GiB); the two ran side by side; the blocks' label exports 54 s, 4.6 GiB each; W-391, from the FOV-named images read lazily from the share: stopped by the worker after 2,911 s (295 GB read, about 45 % of the stack's size written, 3.82 GiB), not retried; W-398, from the same images, each read once: fusion 5,903 s, 4.74 GiB, 662,701,545 bytes (run 6,142 s, 7.62 GiB, 41.7 GB read); W-431, each source file as one piece (tried, then taken out): fusion 5,010 s, 13.13 GiB, 662,701,545 bytes (run 5,251 s, 13.13 GiB, 36.6 GB read); W-432, one chunk per page (kept): fusion 1,511 s, 3.27 GiB, 662,701,545 bytes (run 1,752 s, 6.92 GiB, 36.6 GB read) |
| cell-culture-3D | 55.2 s, 4.77 GiB | 79.1 s, 2.18 GiB; block 28.4 s, 1.02 GiB; W-391: 92.3 s, 2.18 GiB; block (link root) 31.1 s, 1.02 GiB | 6,818 × 4,788 projection: 52.0 s, 2.15 GiB; W-391, from the `IYX` tiles as saved: 229.9 s, 2.21 GiB; W-398, each tile read once: 41.2 s, 2.21 GiB; W-431, with the change: 36.3 s, 2.39 GiB; W-432, with its change: 39.4 s, 2.38 GiB | the block's stack, 33 × 2,844 × 2,842, with its label export: 66.6 s, 2.98 GiB, 31,008,462 bytes |
| tissue-2D | `multiview_stitcher`: 37.4 s (loading 8.0 s), 0.88 GiB | 160.7 s, 5.38 GiB; W-391: 164.9 s, 3.86 GiB | 11,291 × 9,845 projection (max, 0.5), 77,776,028 bytes: 86.4 s, 4.76 GiB; W-391: 89.1 s, 3.59 GiB | none |

No run reached its limit; the largest export folder holds 2.13 GB. Assumptions
({doc}`sample-output-contract`, "Assumptions"): A1 holds, the LN stack fused from lazily
loaded tiles peaking at 2.53 GiB in the fusion part against W-354's 12.35 GiB with every
tile loaded; A2 holds on the three blocks (L2 and L4 pass, every retained ID drawn); A3 and
A8: both stack files are below 4 GiB, so the BigTIFF case was not reached, they read back
through `check_export` in 294 s (with the store) and 100 s, and the tiled zlib writer took
1,468 and 1,564 s against W-354's 242 s uncompressed with every tile loaded; A9 is not
verified, because the one store written (LN sample1, BigStitcher) has no label export;
A4 stays open.

| Sample, placement | Where cells exist | Cell rows | Retained | Excluded `not_owner` | `conflict` (of retained) | `unclaimed` (of `not_owner`) | `fov_edge` (of all rows; of retained) | Overlapping retained pairs (of different FOVs) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LN sample1, BigStitcher import | the 2 × 2 block of `Position048`, `049`, `057`, `058` | 9,102 | 8,194 | 908 | 8 of 8,194 | 361 of 908 | 196 of 9,102; 98 of 8,194 | 93 |
| LN sample1, `multiview_stitcher` | the same block | 9,102 | 8,197 | 905 | 12 of 8,197 | 360 of 905 | 196 of 9,102; 98 of 8,197 | 82 |
| cell-culture-3D, `multiview_stitcher` | the 2 × 2 block of `Position397`, `398`, `407`, `408` | 226 | 185 | 41 | 16 of 185 | 6 of 41 | 82 of 226; 46 of 185 | 34 |
| tissue-2D, `multiview_stitcher` | none: no cell input | 0 | 0 | 0 | not applicable | not applicable | not applicable | not applicable |

The overlap count is computed by the issue's script, which gives 2 on `label_pair`; it is an
upper bound on the cells retained twice whose observations share a voxel, recorded for
Jiahao's decision on cells with several observations, not acted on.

**What these examples do not show.** The export of the files as saved is shown for
tissue-2D and the cell-culture-3D whole sample (W-391) and for LN sample1, its assembly and
its stack, from its own folder with the configured reference image (W-398). Both cellular
blocks went through links: the export finds an assignment folder only at
`checkpoints/<fov>/assignment/<name>/` under the dataset's root, so the route from the
W-382 folders where they are saved is not shown. The LN fusion from the share took about
four times W-376's time from local copies while the reader rebuilt the chunks of 23 rows
(W-398, W-431); with one chunk per page it took about W-376's time (W-432: 1,511 s against
1,468 s), in one run on one host. No store was written from the files as
saved. No cells exist for tissue-2D. A9 is not verified, because no store holds
`cell_labels`. A4 stays open.

**Both routes on the cell-culture block (W-435).** W-435 ran both routes to the cells of the
cell-culture-3D 2 × 2 block on 2026-10-11, at revision `d0fcaf4`, with the package unchanged, in
the run directory
`/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/runs/W-435/20261011T015921Z-1f511925`
(`worker-notes.md`, `results/manifest.json`, `results/compare/`). The route per FOV gave
W-434's cells: 177 retained, 8 excluded `duplicate_observation`, 150 counted. The route after
fusion ran by its parts, because no environment holds both multiview-stitcher and a GPU: the
fusion, `segment_fused` alone on the GPU, then `Dataset.export` with `segment` off.
`check_export` passes on all three folders. After fusion, Cellpose found 175 cells at scale 1
and 177 at 0.5, of which 149 and 148 are counted, each cell once. `compare_routes` at its
default threshold matched 140 pairs at scale 1 and 139 at 0.5. At scale 1, 10 cells per FOV
and 9 after fusion are in no matched pair. For 16 of these 19 the other route has the same
cell (share 0.95 or more) but does not count it: 14 times because no nucleus matched it, twice
because the route per FOV excluded it `not_owner`. Of the other 3, two are part of one cell
after fusion that the route per FOV holds as two, and one has no counterpart. The unmatched cells do not lie closer to the seams than the matched ones. 4 of the 5
cells that W-434 found missing from the per-FOV `sample.h5ad` are counted after fusion. The
limits: one 2 × 2 block; no annotation says
which route is right; Jiahao considers this dataset's nucleus labels faulty, and they decide
which cells are counted; the two routes segment different images (each FOV's own Z maxima
against the fused projection) as well as fusing at different times. None of this is an
accuracy, a statement about a whole sample or a preferred route.

## Limitations

* The tolerances for `multiview_stitcher` come from two cut fixtures with integer offsets
  and one host; MV3 extends them to fractional offsets by an argument about quantization,
  not a measurement, and the non-square case of MV4 was not measured by W-324.
* The fusion bounds are W-354's values on its own fixtures and the label counts W-359's on
  its two synthetic 2D fixtures, one host and multiview-stitcher 0.1.62 on Python 3.12;
  FI3, FI5 and SD1 rest on assumptions A13, A6 and A10 of {doc}`sample-output-contract`,
  and the label rows on A2 (real and 3D label images), which the implementation and the
  real examples verify.
* The flag frequencies of the real examples exist only where cells are assembled, on the
  cellular 2 × 2 blocks; the whole samples are assembled without cells.
* RX3's review state of the 2024 LN placement under V1 is not known before the import
  exists: `dataset.xml` does not mark the two links BigStitcher's optimization removed.
* The scene fixtures exercise the rules on generated geometry; duplicates and losses on
  real samples are only inspected, not measured against truth.
* The molecule and cell tables of the real samples are not inventoried; their assembly
  figures stay estimates until step 1 of the real examples.
* Viewer qualification stays deferred; no check opens a file in a viewer.
