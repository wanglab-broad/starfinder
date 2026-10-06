# §2.10 validation design: placement, assembly and sample outputs

Status: Proposed

This page is the task-group-6 validation design for the whole of §2.10: the engineering
checks of placement ({doc}`stitching-contract`, {doc}`stitching-algorithms`), of assembly
({doc}`assembly-contract`, {doc}`assembly-algorithms`) and of the sample outputs
({doc}`sample-output-contract`), the fixtures they run on, including the synthetic sample
scene of {doc}`synthetic-sample-scene`, and the plan for the whole-sample real examples.
It builds on those five pages and redefines nothing in them: every rule, column, status,
identity and file a check names is used as those pages state it. Where a check needs
something they do not give, the W-331 worker notes say so. Nothing here is implemented.

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
* **No backend ranking, no sweep, no accuracy claim.** The two binnings of the
  `multiview_stitcher` checks (MV1, MV2) are the contract's starting configuration and the
  alternative setting the contract names; each has its own bound and they are not compared.
  No check prefers one backend, sets a default from comparative data or calls a placement
  accurate. Real data has no placement truth: agreement between BigStitcher and
  multiview-stitcher on LN is monitored in the real examples and gates nothing.
* **E07 keeps the systematic comparison.** Comparing the backends over datasets, settings
  and conditions, with accuracy, robustness and cost, is the work of E07 (W-124). This page
  takes none of it over.

## Evidence for the tolerances

Every measured number is cited by path; nothing was run for this page except one count
derived from a W-324 record (marked below).

* W-324, run directory
  `/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/runs/W-324/20261006T063540Z-5cb2f421`
  (found with `timeout 30 find …/runs/W-324 -maxdepth 2 -name stitching-manifest.json`):
  `notes.md` sections 2.3 to 2.6, 3 and 5, `tables/fixtures_summary.csv`,
  `tables/comparison_summary.csv`, `tables/pairs_between_runs.csv`, `fiji-fixtures/`,
  `parity/`, `fusion/fusion_probe.json` and `scripts/fixture_data.py` (the cut-fixture
  recipe).
* W-323, `/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/runs/W-323/`:
  `20261005T1930Z-fiji-baseline/` (the 2024 `dataset_2024.xml`, `sample1.maf`, `grid.csv`,
  `tile_config_sample1.csv`), `20261005T2110Z-input-inventory/inputs.md`.
* W-326 `tables/option_effects.json` and W-328 `tables/resource_estimate.csv`, in the run
  directories that {doc}`assembly-algorithms` and {doc}`sample-output-contract` name.

### Tolerances and where they come from

| Quantity | Tolerance | Source or justification |
| --- | --- | --- |
| Statuses, reasons, flags, keys, counts, row sets, file lists, hashes, dtypes, shapes | Exact | Integers and strings; every expected value follows from the fixture by hand or by a formula in the test. |
| Positions computed by two different operation orders (an origin plus an index times a spacing, against `placed_metadata(…).index_to_world`; recorded spacings and origins of a fused file) | 1e-9 µm absolute | Positions below 10⁴ µm, a few float64 operations each with relative error at most 2⁻⁵³, so the difference stays below 1e-11 µm; 1e-9 µm is 1e-8 of the smallest spacing in use (0.0946 µm). The same bound as identity "Placement truth" of {doc}`synthetic-sample-scene`. |
| The same quantity computed by the same operations | Bitwise equal | Float64 results of one formula are deterministic on one host. |
| `multiview_stitcher` on the cut fixture, starting configuration (binning 4) | Max absolute error at most binning / 2 = 2 voxels per axis, after removing the mean over the FOVs | W-324 `fixtures_summary.csv`: largest 1.0946 voxels (y, synthetic_3x3, binning 4 with and without the 0.7 filter); notes section 5 proposes "binning / 2". |
| `multiview_stitcher` on the cut fixture, binning 2 | At most 1 voxel per axis | W-324: below 1.8e-13 on both cut fixtures. |
| `multiview_stitcher` on the synthetic scene with fractional jitter, binning 1 | At most 0.5 voxel per axis, against the truth in gauge G1 | binning / 2 as above. A fractional true offset adds a rounding of at most binning / (2 × `upsample_factor`) = 0.25 voxel per link (library default `upsample_factor` 2 in 3D, W-324 notes section 3), which W-324's integer offsets did not exercise; W-324 measured binning 1 at 0.19 voxel at most (real_2x2, z). Binning 1 is used because the scene's overlaps after the jitter can be as narrow as 15 voxels: 3.75 binned pixels at binning 4 and 7.5 at binning 2, below the 14 binned pixels of W-324's narrowest measured overlap (synthetic_3x3 at binning 4), and 15 at binning 1. |
| `multiview_stitcher` on the W-324 hand-built cases (two tiles, Z = 1, anisotropic spacing, no-signal centre, disconnected tile) | binning / 2 = 2 voxels per axis for the FOVs of the reference component; statuses exact | W-324 notes section 2.6: at most 0.50 px (two tiles), 0.2500118 px (Z = 1), 0.75 voxel (anisotropic), 1.6108 and 1.2501 px for the other tiles of the no-signal cases. |
| BigStitcher's result on the cut fixtures, read through the import | Max absolute error at most 1.05 voxels per axis | W-324 notes section 5: largest 1.0000000020 (z, synthetic_3x3), 0.6896 (z, real_2x2); a bound of exactly 1 voxel would fail by 2e-9. |
| Repetitions and thread counts | Identical translations | W-324 `pairs_between_runs.csv` and `fixtures/rerun_vs_attempt1.csv`: largest difference 0.0. |
| LN import against the W-324 parity outputs | Exact (float64 equality, byte-equal CSV) | {doc}`stitching-contract`, "What the import of the LN `dataset.xml` must reproduce". |
| Review defaults (`max_correction_fraction` 0.5, `max_residual_um` 1.0) | Tested at their boundary; not tuned | {doc}`stitching-algorithms`, "Placement-quality checks". For scale, W-324 notes section 5: LN grid-start corrections reach 44.5 px, 0.30 of the 149.6 px overlap, and pass; stage-start corrections reach 149.9 px, 1.0 of it, and are flagged. |
| Agreement of the two backends on LN (real examples only) | Monitored, never gating: residual after the linear trend at most 3.5 px in x, 1.5 px in y, 1.5 slices in z; trend in x at most 2.5 px per column; absolute difference at most 10 px in x, 2.5 px in y, 3.5 slices in z | W-324 notes section 5, regression bounds over the four grid-start configurations; W-324 choice 2 recommends monitoring them, not gating. |
| Memory of fusion, of the store writer and of the label export | Traced peak below the bound each check states (FI7, SD9, SD10, LB4), derived from "Bounded memory" of {doc}`sample-output-contract` | A whole-sample array, or two FOV volumes at once, exceeds the bound by construction of the fixture. |

## Fixtures

Every fixture is generated in the test from the values stated here, or is a named copy of
a record. No test reads the network share, a W-323 or W-324 run directory, real images or
Fiji. Every fixture stays inside the test bounds of the §2.10 resource exceptions
("Resource plan"); the ones above the routine size are marked `slow`.

| Fixture | Construction | Seeds | Needs |
| --- | --- | --- | --- |
| `stub_grid` | 2 rows × 3 columns of FOV grids of 4×16×24 voxels (Z, Y, X; not square), spacing (0.25, 0.25, 0.25) µm (dyadic, so the review boundaries are exact), identity direction, overlap 0.25; zero images; positions from `positions_from_grid` (`column_major_down_right`). A test-only method, registered in `STITCHING_METHODS` for the test, returns supplied translations (µm), components, links with quality and kept flags, and diagnostics. | none | hand-built |
| `stub_variants` | `stub_grid` with: fractional translations (multiples of 2⁻⁶ µm, so the expected sums are exact); spacing (1, 2, 2) µm and (0.346255, 0.194398, 0.194398) µm; Z = 1 grids from YX inputs; one FOV of 4×16×20 beside 4×16×24 (unequal shapes); the unsnapped stage positions of `maf_2x3_jitter` (irregular); the five positions of `maf_3x2_gap` (an L-shaped union); one FOV whose links all have NaN quality; one FOV whose nominal x is moved by two tile widths (no overlap); two components of two FOVs each (a tie for the reference component); corrections at exactly `0.5 × w` and `0.5 × w + 2⁻²⁰` µm, residuals at exactly 1.0 and `1.0 + 2⁻²⁰` µm. | none | hand-built |
| `parity_records` | Copies, in `src/python/test/data/stitching_parity/` with a SHA-256 manifest, of W-323's `dataset_2024.xml` (145 KB), `sample1.maf` (36 KB), `grid.csv`, `tile_config_sample1.csv`, and of W-324's `parity/script_grid_*.csv`, `parity/script_tile_config_*.csv`, `parity/prototype/tile_config_ln/T2_sum.csv` and `parity/fixtures/` (two stage files, two `dataset.xml`, two grids); about 260 KB of text, no image. | none | copied records |
| `golden_stitching` | The stage file and `dataset.xml` of `test_workflow_stitching_golden.py` (`stage_file`, `dataset_xml`, `GRID_CSV`, `TILE_CONFIG`), imported from that module. | none | hand-built |
| `fiji_cut` | Copies of W-324 `fiji-fixtures/synthetic_3x3/` and `fiji-fixtures/real_2x2/` `dataset.xml` and `grid.csv` (26 KB). The true integer origins are recomputed in the test by the `grid_layout` rule of W-324 `scripts/fixture_data.py` (no image needed): synthetic seed 324, step 64, base 4, z nominal 1, jitter ±4 (y, x) and ±1 (z); real seed 325, step 684, base 26, z nominal 2, jitter ±8 and ±2. | 324, 325 | copied records |
| `cut_synthetic` | The W-324 recipe rebuilt in the test: scene 16×256×256 (seed 20261006: texture and 70 Gaussian nuclei), 3×3 tiles of 14×120×120, nominal step 64 px, true origin nominal plus integer jitter (seed 324), noise N(0, 3) per tile (seed 1000 + tile), uint16, spacing (0.346255, 0.194398, 0.194398) µm. Derived cases as W-324 `scripts/fixtures.py` builds them: `two_tiles`, `z1_3d`, `plane_2d`, `spacing_1_2_2`, `no_signal_zero`, `no_signal_noise` (centre noise seed 1050), `disconnected`; and `non_square`, new: the same scene cut into 3×3 tiles of 14×104×120 at nominal steps 56 (y) and 64 (x), with the same jitter draws. | 20261006, 324, 1000 + tile, 1050 | `multiview-stitcher` extra; `slow` |
| `assembly_golden` | `build_sample` of `test_workflow_assembly_golden.py`: three FOVs of 8×48×64 on a 2 × 2 grid with a blank cell, the fractional `STITCHING` translations, nine scene cells, 24 molecules, 3D and 2D; its placement converted into a `PlacementResult` by the import (or, before the import exists, built directly from the same origins). | none | hand-built |
| `own_cases` | Two FOVs of 4×16×16, spacing 1 µm, origins (0, 0, 0) and (0, 0, 12) µm (centres at x 7.5 and 19.5, boundary x 13.5), with box cells and molecules placed, by hand, so that every assembly status, every reason, every flag and one exact tie at x 13.5 occur at least once, including a retained `excluded_no_nucleus` cell and an expanded FOV; the expected status, reason, owner and flags are written in the fixture. | none | hand-built |
| `edge_points` | For one FOV of 4×16×24 at a fractional origin: points at each footprint bound `origin − s/2` and `origin + (n − ½)·s`, 1e-9 µm inside and outside each; for two FOVs of one shape at X origins 0 and 10 µm, a point exactly on the bisector (float-exact). | none | hand-built |
| `empty_cases` | `assembly_golden` variants: a FOV without molecules, a FOV without cells, a FOV whose molecules miss every cell, a sample whose cells are all `excluded_no_nucleus` (no counted cell), a `molecules_only` sample, and a placement whose FOVs are all `failed`. | none | hand-built |
| `status_cases` | `assembly_golden` with FOV2 `placeholder` (no input), FOV3 `missing`, then FOV3 `failed` with an input; review `required` without and with an approval. | none | hand-built |
| `scene_tiny` | `sample_scene_preset("sample-tiny-v1")` ({doc}`synthetic-sample-scene`, "Presets and resources"): 2 × 2 FOVs of 5×36×36, fractional J1 jitter, seed 42, split `development`; and `scene_tiny_inward`, its variant with every FOV moved by the jitter bound towards the grid centre in Y and X and by `+J_z` in Z (the least-covered case of that page). | 42 | generator addition |
| `scene_small` | `sample_scene_preset("sample-small-v1")`: 3 × 3 FOVs of 14×96×96 at the LN calibration, overlap 0.25, jitter bound (0.8, 4.5, 4.5) voxels, seed 42. | 42 | generator addition; `slow` |
| `voxel_fused` | Source images that are zero except one voxel of value `A`, unit spacing: the pair of FC1 (two FOVs of 1×4×4 at X origins 0 and 1.4 µm, the worked counterexample of {doc}`sample-output-contract`; `A` 200 in uint8) and the quad of FC3 (four FOVs of 3×8×8 at origins that are not half a voxel off the grid; `A` 200 in uint8 for FI2, and for FI6 one copy per allowed input dtype with the `A` of the table "FI6 dtype cases", which every dtype can hold). | none | hand-built |
| `zstack` | Two FOVs of 4×8×8 uint16, unit spacing, values `1 + 64·z + 8·y + x + 100·f` for FOV `f`, at the origins of FC6 (equal `origin_z`) and FC7 (`origin_z` 0 and 1.4 µm), and two Z = 1 FOVs of 1×8×8 (FC8); a float32 copy for FC10. | none | hand-built |
| `fuse_band` | Six FOVs of 16×256×256 uint8 in one column, overlap 0.1 (Y origins `230.4·r` µm), unit spacing, values `(z + y + x) mod 251`: fusion grid 16 × 1408 × 256 (three 512-row bands, three OME-TIFF tiles, two projection chunks and three stack chunks of OME-Zarr, one automatic level), the FC11 of the fusion configurations. | none | `slow`; inside the `slow` bound of a 3 × 3 grid of 16×256×256 tiles |
| `export_cases` | The assemblies of `assembly_golden` (3D, 2D, with an expansion), `empty_cases` and `status_cases`, with the fused images of FC1 or FC3; one sample per row of the table "Empty populations" of {doc}`sample-output-contract`. | none | `spatialdata` extra |
| `wf_configs` | The repository's existing test configurations (legacy, without the new blocks), and copies with a `stitching`, an `assembly` and a `sample_outputs` block. | none | hand-built |

Seeds: 324 and 325 (W-324 jitter), 20261006 (W-324 scene), 1000 + tile and 1050 (W-324
noise), 42 (the sample presets), and 210 for the row permutation of AS14; in the real
examples only, 325 and 1100 + tile for W-324's real_2x2 cut fixture. No other seed and no
other random number is used. Fixtures marked "none" use no random number.

## Checks

"Needs" says what a check requires beyond the routine locked environment: **G** the
generator addition of {doc}`synthetic-sample-scene` (W-327); **H** nothing but
hand-built geometry; **C** the copied records (`parity_records`, `fiji_cut`); **MV** the
`multiview-stitcher` extra; **SD** the `spatialdata` extra (also needed for standalone
OME-Zarr, which uses `ome-zarr`); **S** `slow`.

### Placement (PL)

| # | Check | Fixture (seeds) | Metric | Pass/fail tolerance | Needs |
| --- | --- | --- | --- | --- | --- |
| PL1 | Converters | `parity_records` stage files and `sample1.maf`; `golden_stitching`; supplied tables in µm, nm, mm (none) | `TilePositionTable` columns and record | Every `StageAxisMapping` of the eight sign and axis combinations moves stage +X and +Y where it names; the smallest y and x are 0 and z is 0; with `GridSnap(overlap=0.1, tile_shape_yx=(1496, 1496), spacing_yx=(0.194398, 0.194398))` on `sample1.maf` the cells equal the column and row of `script_grid_ln.csv` for every `PositionID`, and on the two W-324 stage files those of `script_grid_maf_*.csv`, including `maf_2x3_jitter`, whose last-bit tie the 0.1 µm rounding removes (a remaining tie goes to the smaller step); `positions_from_grid` puts FOV `k` at column `k // rows`, row `k % rows` with the step `(1 − overlap) × tile × spacing`; units convert within 1e-9 µm; each failure of {doc}`stitching-algorithms` ("Converters") raises `ValueError` naming the item. Exact otherwise. A `StageAxisMapping` that names one image axis twice raises `ValueError`. | C, H |
| PL2 | Neighbour pairs | `stub_grid`, `stub_variants` (irregular, unequal shapes, L shape); `parity_records` (the LN grid start) | Pair set and overlap boxes | The pairs are those whose boxes intersect with positive area (and positive z length for Z > 1), diagonals included; touching boxes are not a pair at tolerance 0 and are at a positive tolerance; no pair depends on a row or column index (permuting the grid cells of a supplied table changes no pair); on the LN grid start (64 FOVs, step 0.9 × 1496 px) there are 213 pairs, the number multiview-stitcher's overlap graph found (W-324 notes section 2.1; also derived for this page from `parity/script_grid_ln.csv`). Exact. | C, H |
| PL3 | Transform direction and composition | `stub_variants` (fractional); `golden_stitching`; a `dataset.xml` written in the test whose `ViewRegistration` elements are in permuted order; one with a rotation of 1e-6 rad | Sample coordinates, inverse, composed affines | `origin + p ⊙ spacing` (push) equals `placed_metadata(result, fov).index_to_world(p)` within 1e-9 µm for integer and fractional `p`, including the contract's example `(0.69251, 252.44398, 100.25)` µm; the inverse `(sample − origin) ⊘ spacing` returns `p` within 1e-9 voxel; a cropped grid maps `q` to `q + start` first; the import composes every transform except `calibration` (newest first), its result equals `Stitching Transform` plus `Translation to Regular Grid`, the views map through the `ViewSetup` tile attribute (the permuted file gives the same result), and the rotated file raises `ValueError` naming the view. Exact or 1e-9 as stated. | H |
| PL4 | Gauge, fractional placement and the record | `stub_variants` (fractional); `scene_tiny` (seed 42) | Origins, corrections, record | Origins equal `nominal + c − mean(c)` over the `image_supported` FOVs, bitwise against the same formula in the test; the mean correction is 0 within 1e-9 µm; no origin is rounded (fractions of 2⁻⁶ µm survive); the record assertions listed in {doc}`stitching-contract` ("Method record": `fov_id` order, one of the five statuses, float64 unrounded origins, finite exactly for the three placed statuses, zero mean correction, `placed_metadata`) hold. On `scene_tiny`, the test-only method returning the true corrections from `placement_truth.csv` gives `origin_*` equal to `true_origin_g1_*` within 1e-9 µm for every FOV. The inputs' own `ImageMetadata` are unchanged after `stitch` (equal before and after). | H; G for `scene_tiny` |
| PL5 | Anisotropic spacing and units | `stub_variants` (spacing (1, 2, 2) and the LN calibration); `golden_stitching` | Origins in µm; import units | Method translations in µm stay in µm; the import divides world z by the calibration's z scale (1.7894736842105263 for 0.34 / 0.19) and multiplies by each FOV's reference spacing, never by the macro's `voxel_size_*`; a calibration with `c_x ≠ c_y` raises. Within 1e-9 µm. | H |
| PL6 | Z = 1 | `stub_variants` (Z = 1 from YX inputs); a Z = 1 FOV beside a Z = 4 FOV | Grids, origins, errors | YX inputs give Z = 1 grids; origins keep z from the method (0 for the stub); mixed Z sizes, spacings, directions or units raise `IncompatibleGeometryError` naming the FOVs; a spec without `2` in `dimensions` refuses Z = 1 inputs. Exact. | H |
| PL7 | Non-square tiles | `stub_grid` (16 × 24); `golden_stitching` | Boxes, origins, legacy view | Every box and origin uses the grid's own Y and X sizes; the import's legacy view (T3 to T7) reproduces `TILE_CONFIG` of `test_workflow_stitching_golden.py`, `img_col` for the Y extent included, while the record never uses `img_col`. Exact. | H |
| PL8 | Irregular footprint | `stub_variants` (unsnapped `maf_2x3_jitter`, unequal shapes, the L shape of `maf_3x2_gap`) | Pairs, statuses, components | Pairs follow PL2's rule; every FOV of the L shape is placed and joined; the missing position has no row; the union of footprints is not a rectangle and nothing assumes one. Exact. | H, C |
| PL9 | A tile with no signal | `stub_variants` (one FOV with only NaN-quality links) | Statuses, flags, links | Under L1 every NaN link has `kept` false and reason `nan_quality`, the FOV carries `nan_quality_link`; it is `failed` with `fallback=None` and `metadata_only` at `origin = nominal` with `fallback="nominal"`; review is `required`. Exact. | H |
| PL10 | Disconnected tile, components and the review policy | `stub_variants` (moved FOV, two components, review boundaries) | Statuses, review, approval | The reference component is the one with the most FOVs, and on a tie the one with the first FOV in table order; FOVs outside it are `failed` (fallback `None`) or `metadata_only` (fallback `"nominal"`), `component` −1 only for `missing`; a correction of exactly `0.5 × w` is not flagged and `0.5 × w + 2⁻²⁰` µm is flagged `correction_above_bound`; a kept-link residual of exactly 1.0 µm is not flagged and `1.0 + 2⁻²⁰` µm is flagged `residual_above_tolerance` on both FOVs; a method-reported removed link gives `optimizer_removed_link`; each makes review `required` with its reason; `approve_placement` records reviewer, UTC time, note and the SHA-256 of both tables and changes no origin; `assemble` refuses the unapproved result with `PlacementReviewRequiredError` (AS10). An input whose image is `None` gives status `missing`, NaN origin and `component` −1, and review `required`. Exact. | H |
| PL11 | Stage-position fallback | `stub_variants` (moved FOV); a copy of `fiji_cut`'s synthetic `dataset.xml` with every `PairwiseResult` of one corner view removed (the hand-built `dataset.xml` files hold no `PairwiseResult`) | Statuses, origins, legacy view | With `fallback="nominal"` the FOV is `metadata_only`, `origin` equals `nominal` exactly, `correction` is 0, flag `fallback`, review `required`; with `fallback=None` it is `failed` with NaN origin; in the import, BigStitcher's translation for that view stays in `method_translation_*`, unrounded, in both cases; the legacy view (`tile_config`) is byte-identical for both values. Exact. | H, C |
| PL12 | Import of the LN `dataset.xml` against the W-324 parity outputs | `parity_records` | The five items of {doc}`stitching-contract` ("What the import of the LN `dataset.xml` must reproduce") | (1) raw translations of all 72 setups equal `T2_sum.csv` as float64; (2) setup `i` maps to row `i` of `script_grid_ln.csv`, 64 acquired, 8 padding views with no row; (3) the legacy view with `img_col = img_row = 1496` equals `script_tile_config_ln.csv` and the 2024 `tile_config_sample1.csv` in all 72 rows and 16 columns, minima −5460, −6135 and −69 before the shift; (4) 145 links, all kept, none on a padding view, one component, 64 `image_supported` FOVs; (5) z in slices is world z ÷ 1.7894736842105263 and origins use the reference spacing. The review state and the residual of each of the 145 links are recorded in the test output (they decide whether the 2024 placement needs review under V1; worker notes). Exact. The same import on the two W-324 hand-built `dataset.xml` reproduces `script_tile_config_xml_2x2_blank.csv` and `script_tile_config_xml_3x2_positive.csv`, and on `golden_stitching` reproduces `TILE_CONFIG`. | C |
| PL13 | BigStitcher's result on the cut fixtures, through the import | `fiji_cut` (324, 325) | Error of the imported origins against the true integer origins, after removing the mean | Every view has a `Stitching Transform`; max absolute error at most 1.05 voxels per axis on both fixtures. | C |
| PL14 | Persistence | results of PL4, PL10 (with an approval) and PL12 | Files and tables | `save_placement` writes `placement.json`, `fovs.csv`, `links.csv` (and for the import `view_map.csv` with the macro's and `dataset.xml`'s SHA-256); `load_placement` returns tables equal under `assert_frame_equal(check_exact=True)` (float64 origins bit-identical through CSV) and an equal record; a changed table raises `ValueError` naming it. `placement.json` holds `frame`, `gauge`, the table record, per FOV the input source, file SHA-256, grid shape, metadata and grid SHA-256, `review` and `format_version` 1; a save interrupted after its first file (a test hook raising) leaves no partial folder. Exact. | H, C |
| PL15 | Entry, registry and errors | `stub_grid` | Raised errors, record | `stitch` refuses an unknown config type (`TypeError`), inputs that do not match the table, and incompatible grids, before calling the method; it never reads or writes a file (a test with a read-only temporary directory as the working directory); `STITCHING_METHODS` uses the shared mechanism (exact-type lookup, names, provenance entry with `stage` `stitching`); a spec whose dependency is missing raises `StitchingBackendUnavailableError` naming the module and the extra; importing `starfinder.stitching` imports no `multiview_stitcher`; `devices` is `("cpu",)`. `transform_kind` is `translation`. Exact. | H |
| PL16 | Determinism of the stub path | `stub_variants` | SHA-256 of `fovs.csv` and `links.csv` in three single-thread processes | Identical. | H; S |
| PL17 | `Dataset.stitch` | a synthetic dataset folder written in the test: four FOVs of 4×16×24 with `images/ref_merged/{fovID}.tif` carrying reference metadata, one copy with a FOV of another tile size; the same FOVs with a saved `reference_stain` checkpoint image each and with a `file:` pattern, one file of another shape (none) | Table, inputs, saved files, errors | Given the workflow `positions` mapping without `tile_shape_yx` and `spacing_yx`, the table equals `positions_from_grid` with the grids' `shape_zyx[1:]` and `spacing_zyx[1:]`, recorded in the table's record; the inputs are built in table order; with `output_dir` the files of PL14 are written; the copy with another tile size raises `IncompatibleGeometryError` naming the FOVs and both values. `input="reference_stain"` reads `<checkpoint dir>/<fov_id>/other_rounds/reference_stain/image.ome.tif` and `input="file:<pattern>"` accepts a file of the grid's shape and raises `IncompatibleGeometryError` for another shape; the FOV images are loaded one at a time (counted through the reader); no `FOV.run`, `FOV.segment` or `FOV.assign` is called (each monkeypatched to raise). Exact. | H |

### `multiview_stitcher` (MV)

Starting configuration of {doc}`stitching-algorithms`: phase correlation, translation,
binning 4 in every axis, links with quality below 0.7 or NaN dropped (L1),
`alternating_pattern` pruning, global optimization with library tolerances.

| # | Check | Fixture (seeds) | Metric | Pass/fail tolerance | Needs |
| --- | --- | --- | --- | --- | --- |
| MV1 | Known-answer cut fixture, starting configuration | `cut_synthetic` 3 × 3 (20261006, 324, 1000 + tile) | Error of `origin_*` against the true origins, after removing the mean, per axis, in voxels | All nine FOVs `image_supported`, one component; max absolute error at most 2 voxels per axis. | MV, S |
| MV2 | The same at `binning_zyx=(2, 2, 2)` | `cut_synthetic` 3 × 3 | As MV1 | At most 1 voxel per axis. | MV, S |
| MV3 | Fractional jitter on the synthetic scene | `scene_small` (42); input per FOV the reference-round image, maximum over channels (the `reference_merged` analogue); `binning_zyx=(1, 1, 1)` | `origin_*` against `true_origin_g1_*` of `placement_truth.csv`, in voxels | All nine FOVs `image_supported`; max absolute error at most 0.5 voxel per axis. | G, MV, S |
| MV4 | Hand-built cases | `cut_synthetic`: `two_tiles`, `z1_3d`, `plane_2d`, `spacing_1_2_2`, `non_square`, `no_signal_zero`, `no_signal_noise` (1050), `disconnected` | Statuses and error per axis | Every FOV of the reference component within 2 voxels per axis; Z = 1 cases have z correction exactly 0 (registered in 2D); `spacing_1_2_2` errors in each axis's own voxels; the centre tile of both no-signal cases and tile 8 of `disconnected` are `failed` with `fallback=None` and `metadata_only` at nominal with `fallback="nominal"`; review `required` in those three cases; in the others the review state and its reasons are recorded (with spacing (1, 2, 2) µm a residual of half a voxel already reaches the 1.0 µm default, so no state is asserted there). | MV, S |
| MV5 | Wrapper rules | `cut_synthetic` 3 × 3; two tiles with no overlap | Links, errors | The registered pairs are a subset of `neighbor_pairs` and every other pair is recorded `pruned`; no NaN-quality link is kept; translations are in µm, view to common frame; with no link left, every FOV is `failed` (the library's `NotEnoughOverlapError`), and the result is still returned and saved. Links with quality below 0.7 have `kept` false and reason `quality_below_threshold`; a tile below 16 px in Y or X is refused (`min_shape_zyx`). Exact. | MV, S |
| MV6 | Determinism | `cut_synthetic` 3 × 3 | Translations of two runs at one thread, each in its own process | Bitwise identical. The independence of the thread count, which W-324 measured (`pairs_between_runs.csv`), is checked in the real placement examples, whose bound allows four threads; the `slow` tier allows one CPU. | MV, S |

### Assembly (AS)

Each accounting check cites the identity of {doc}`assembly-contract`, "Count
accounting", by its number.

| # | Check | Fixture (seeds) | Metric | Pass/fail tolerance | Needs |
| --- | --- | --- | --- | --- | --- |
| AS1 | Sample coordinates | `assembly_golden`; `stub_variants` (LN calibration) | `sample_*`, `sample_centroid_*`, `sample_expanded_centroid_*`, `sample_ownership_centroid_*` | Each equals `placed_metadata(…).index_to_world(p)` within 1e-9 µm for the molecule's `z, y, x` and the float centroids (never truncated); `sample_centroid_z` is NaN for a `plane` cell; `sample_ownership_centroid_*` equals the pair `ownership_centroid` names, bitwise; the example `(1.3, 37.4, 0.7) + (3, 8, 20) = (4.3, 45.4, 20.7)` µm holds. No integer `global_*` column is produced. | H |
| AS2 | Ownership at region edges and ties | `edge_points`; `assembly_golden`; a 0.25-voxel raster over `assembly_golden`'s footprints | `owner()` | A point at the lower footprint bound is inside and one 1e-9 µm below is not; one at the upper bound is outside and one 1e-9 µm below is inside; an exact tie goes to the FOV first in table order, and reversing the order of the two FOVs in the placement changes the owner; every raster point inside a footprint has exactly one owner and every other point none; the FOV1/FOV3 boundary lies between x 59.1 and 59.3 and the FOV1/FOV2 boundary near y 42.2 ({doc}`assembly-algorithms`, "O1"); `midpoint_box` on a placement without grid cells raises `ValueError`. With `ownership="midpoint_box"` on `assembly_golden` the box edges lie at x 59.2 and y 42.2 and the retained sets equal AS3's ({doc}`assembly-algorithms`, "O2"); moving a point in z changes no owner (Z1). Exact. | H |
| AS3 | Retained sets on the golden fixture | `assembly_golden` (3D and 2D) | Retained scene cells per FOV, flags | 3D: FOV1 retains scene cells 1, 2, 5, 9; FOV2 4, 6; FOV3 3, 7, 8; 2D: FOV3 also 9, both copies flagged `conflict` with each other in `conflict_with`; cell 9's molecule at x 64 is excluded `cell_not_owner` in 3D and retained in 2D (W-326 `tables/option_effects.json`). Exact. | H |
| AS4 | Retention rules and reasons | `own_cases`; `status_cases` | `assembly_status`, `assembly_reason`, `owner_fov_id` per row | Every row equals the status, reason and owner written in the fixture: rules 1 to 7 of "Retention", reasons `not_owner`, `fov_not_placed` (cells) and `cell_not_owner`, `not_owner`, `outside_grid`, `fov_not_placed` (molecules); a retained `excluded_no_nucleus` cell keeps its §2.9 status and its molecules follow it; no molecule changes `cell_id` or `assignment_status`. The cell rows are in placement table order, then increasing `cell_id`, and the molecule rows in placement table order, then input order. Exact. | H |
| AS5 | Flags | `own_cases`; `assembly_golden` | `assembly_flags`, `conflict_with` | `fov_edge`, `z_edge`, `conflict`, `unclaimed`, `outside_owner_region`, `cell_unclaimed` each on exactly the rows written in the fixture; in 3D FOV1's cell 6 is `fov_edge` and FOV3's cell 5 is not `unclaimed` ({doc}`assembly-algorithms`, "Flags"); with `conflicts=False` the `conflict` and `unclaimed` columns are null and the record says so; no flag changes a status (the statuses with and without `conflicts` are equal). Exact. | H |
| AS6 | Accounting identities | `assembly_golden` (3D, 2D, expanded), `own_cases`, `empty_cases`, `status_cases` | The sample tables against the inputs | **Identity 1**: anti-joins of the input keys `(cell_namespace, cell_id)` and `(spot_namespace, spot_id)` against the sample tables are empty both ways and no key repeats; **2**: per FOV and in total, input rows = retained + Σ excluded by reason, for cells and molecules, and `record["counts"]` equals these totals; **3**: every `assigned` or `excluded_cell` molecule has its cell's status (reason `cell_not_owner` where the cell's is `not_owner`), every `unassigned` molecule of a placed FOV is retained exactly when `owner_fov_id` equals its `fov_id`, every `outside_grid` molecule is excluded; **4**: the sorted `(fov_id, cell_id, gene_id, compartment, count)` rows of the sample `counts` equal those of the per-FOV §2.9 `counts` restricted to the counted cells, and `Σ whole` = retained `assigned` molecules = Σ `n_molecules` over the counted cells; **5**: `whole = nucleus + cytoplasm` per counted cell with `compartments` `available`, no compartment rows otherwise. Each identity is evaluated independently of `AssemblyResult.__post_init__`, and a result built with one row removed or one status changed is refused by `__post_init__` with `ValueError` naming the identity. `obs_name` is unique; `matrix()` equals the dense `whole` counts in `genes` order. Exact. | H |
| AS7 | Molecules-only case | `empty_cases` (`molecules_only`) | Tables, AnnData | `mode` `molecules_only`; cell and count tables have no row; every molecule is retained or excluded by its own position (`outside_grid` positions excluded); identities 1 to 3; `sample.h5ad` has `X` of shape `(0, n_genes)`, `var["n_molecules_retained"]` equal to the retained molecules per gene, and `uns["assembly"]` the record. Exact. | H |
| AS8 | Empty inputs | `empty_cases` | Tables, record, errors | A FOV without molecules, a FOV without cells and a FOV whose molecules miss every cell go through the same rules without a branch (identities 1 to 5 hold, the `fovs` table has their rows); a sample with no counted cell writes `sample.h5ad` with no row; a placement with no placed FOV raises `ValueError`. Exact. | H |
| AS9 | Placeholder and missing FOVs | `status_cases` | Statuses, ownership, errors | `image_supported` and `metadata_only` FOVs without an input raise `ValueError`; a `missing` or `failed` FOV's input is accepted and all its rows are `fov_not_placed` with NaN sample coordinates and null owner; a `placeholder` FOV is placed and owns points by the contract as written, so a neighbour's observation in its region is excluded `not_owner` and such a cell is flagged `unclaimed` (worker notes, gap 1, propose a change at the gate). Exact. | H |
| AS10 | Review refusal and entry checks | `status_cases`; `assembly_golden` with one violation per check | Raised errors | Checks 1 to 8 of "Checks the entry applies" each raise their named error (`PlacementReviewRequiredError`, `ValueError`, `IncompatibleGeometryError`) naming the FOVs, in that order (an input that violates two raises the earlier); an approved placement passes. `assemble` writes no file (a read-only working directory) when its inputs are in memory. Exact. | H |
| AS11 | Sample-level AnnData against the sample tables | `assembly_golden` (3D with an expanded FOV), `own_cases` (with a retained `excluded_no_nucleus` cell) | `sample.h5ad` | Every condition of "A test can check it against the sample tables" in {doc}`assembly-contract`: `n_obs` equals the counted cells and `obs_names` and `obs[["fov_id", "cell_namespace", "cell_id"]]` equal those rows in order; no other row's `obs_name` appears; `X[i, g]` equals the `whole` count (0 where none) and the `nucleus` and `cytoplasm` layers likewise, NaN rows where `compartments` is not `available`; every sample-coordinate column and `ownership_centroid` equal the cell table's exactly; `obsm["spatial"]` columns equal `sample_ownership_centroid_x` and `_y` (the expanded centroids on the expanded FOV); `uns["spatial_centroid"]` names those columns and the setting; `var_names` equal `genes`. Exact (NaN equals NaN). Without `anndata` importable (monkeypatched), `sample_anndata` raises `ModuleNotFoundError` naming the `anndata` extra. | H |
| AS12 | Persistence round trip | AS3 and AS7 results, CSV and Parquet | Files, tables, record | `save_assembly` writes the files of S1 with their SHA-256 in `assembly.json`; `load_assembly` returns tables equal under `assert_frame_equal(check_exact=True)` and an equal record; a changed file raises `ValueError` naming the path and both hashes. `assembly.json` holds the placement's method, gauge, review and table hashes, per FOV the placement status, origin, kind, assignment hashes, grid and `calibration_source`, the config, footprints and centres, the counts and `format_version` 1; a save interrupted after its first file leaves no partial folder. Exact. | H |
| AS13 | Legacy adapter unchanged | `test_assignment_workflow.py`'s configurations and the A18 `raw.h5ad` comparison | `raw.h5ad` | After `counts_anndata` is factored out of the adapter, `raw.h5ad`'s `obs` columns and `X` equal the frozen helper's values, as A18 of {doc}`assignment-algorithms` states. Exact. | H |
| AS14 | Determinism and row order | `assembly_golden` (3D) | SHA-256 of every sample table in three single-thread processes; per-key results with the input molecule rows permuted by `numpy.random.default_rng(210).permutation` | Identical hashes; identical per-key status, reason, owner and flags. | H; S for the processes |
| AS15 | `Dataset.assemble` | `assembly_golden` written as a dataset with saved placement (PL14) and saved §2.9 assignment folders; the same with `assignment=None` and saved molecule tables (none) | Result, saved folder, memory | The result equals `assemble` called on the same inputs (tables equal under `assert_frame_equal(check_exact=True)`, equal records); with `output_dir` the files of S1 are written; a changed assignment file raises naming the path; at most two territory images are loaded at a time (counted through the loader). Exact. | H |

### End-to-end path of the synthetic scene (EE)

The path of {doc}`synthetic-sample-scene` ("The path that needs no segmenter"): per FOV,
the truth labels enter through `import_labels(…, relabel=False)` and the molecules
through `molecule_table_from_csv`; `assign_molecules` runs without expansion; then
`assemble`. The placement is the truth (a `PlacementResult` holding `true_origin_g1_*`,
statuses `image_supported`, built as the worker notes' choice 2 decides) in EE1 to EE5,
and the result of `stitch` in EE6. The oracle of EE2 is computed in the test from the
truth tables, the FOV truth label images and the rules of {doc}`assembly-algorithms`,
never from `assemble`'s output.

| # | Check | Fixture (seeds) | Metric | Pass/fail tolerance | Needs |
| --- | --- | --- | --- | --- | --- |
| EE1 | Per-FOV assignment equals the sampled truth | `scene_tiny`, `scene_tiny_inward` (42) | Per `molecule_fovs` row: §2.9 status and `cell_id` | Rows with `in_footprint`: `cell_id` (null read as 0 for `unassigned`) equals `fov_cell_label`; rows without: `outside_grid`; the `label_agrees` false count per FOV is recorded (a measurement of the §2.9 sampling rule, not a failure). Exact. | G |
| EE2 | Assembly equals an independent oracle | `scene_tiny`, `scene_tiny_inward` | Per row: `assembly_status`, `assembly_reason`, `owner_fov_id`, `fov_edge`, `z_edge` | Equal to the oracle (O1 and T1 recomputed from the true origins and grid shapes; centroids recomputed from the truth label voxels); `fov_edge` and `z_edge` equal `cell_fovs`' columns of the same name for every (cell, FOV). Exact. | G |
| EE3 | Accounting on the scene | as EE2 | Identities 1 to 5 of "Count accounting" | As AS6. Exact. | G |
| EE4 | Guarantees against sample-level truth | as EE2 | Sample molecules (`in_sample` true) and their copies | (a) every sample molecule has at least one copy that is not `outside_grid` (identity 3 of {doc}`synthetic-sample-scene`); (b) every sample molecule whose copy in the FOV owning its sample position samples the label 0 has exactly one retained `unassigned` copy, in that FOV; (c) a molecule with `r ≥ 2` retained copies has at least `r − 1` of them flagged `outside_owner_region`; (d) every retained `assigned` copy whose `cell_id` differs from `true_cell_id` is a row with `label_agrees` false, and the converse holds on retained `assigned` copies. Exact. | G |
| EE5 | Comparison with sample-level truth | as EE2 | Per sample cell: retained copies (0, 1, 2 or more) by flag; per sample molecule: retained copies; the same for halo content, reported apart | The table is written to the test's output and equals the oracle's recount exactly. Duplicates and losses are the agreed behavior of retention without matching ({doc}`assembly-contract`, "What this policy does not guarantee"); their numbers are measurements, not failures, and no accuracy is claimed. | G |
| EE6 | The full chain | `scene_small` (42), placement from MV3 | MV3's bound; EE2 and EE3 on the stitched placement; differences from the truth-placement assembly | MV3 passes; EE2 and EE3 hold exactly with the stitched origins in the oracle; the observations whose status differs from the truth-placement assembly are listed (a measurement). | G, MV, S |

### Fused image (FI)

Each read-back check cites "Sample position to image position" (conditions 1 to 4) or
"Read-back checks" of {doc}`sample-output-contract`. The files that are read back come
from the accepted configurations FC1 to FC11 below; the refused configurations FR1 to
FR5 write nothing and are tested only for the error. For each configuration the table
gives the fusion grid `n0` (Z × Y × X, from the shape formula of "The fusion grid"), the
output level `N` (Y × X, `ceil(n0 / k)`), the largest `pyramid_levels` that `fuse`
accepts, `floor(log2(max(N_y, N_x)))`, and the tiles and chunks of the output level.
Every fixture has unit spacing (1 µm on each axis) unless stated.

| Config | Fixture and geometry | `k_yx` (`k_z`), `pyramid_levels` | `n0`; `N`; largest accepted levels | OME-TIFF tiles; OME-Zarr chunks and automatic levels | Accepted |
| --- | --- | --- | --- | --- | --- |
| FC1 | `voxel_fused` pair: two FOVs of 1×4×4 at X origins 0 and 1.4, Y and Z origins 0 | 1, 0 | 1 × 4 × 6; 4 × 6; 2 | one tile; one chunk, no automatic level | yes |
| FC2 | the same | 1, 1 | as FC1; level 1 is 2 × 3 | as FC1 | yes |
| FC3 | `voxel_fused` quad: four FOVs of 3×8×8 at (Z, Y, X) origins (0, 0, 0), (0, 6.25, 0.25), (0, 0.25, 6.25), (0, 6.25, 6.25) | 1, 2 | 3 × 15 × 15; 15 × 15, padded to 16 × 16; 3 (levels 8 × 8, 4 × 4) | one tile per plane; one chunk, no automatic level | yes |
| FC4 | the same | 2, 2 | 8 × 8 (partial last block); 3 (levels 4 × 4, 2 × 2) | as FC3 | yes |
| FC5 | the same | 4, 1 | 4 × 4 (partial last block); 2 (level 2 × 2) | as FC3 | yes |
| FC6 | `zstack`, equal Z: two FOVs of 4×8×8 uint16 at origins (0, 0, 0) and (0, 0, 6.25) | 1 (`k_z` 1 and 2), 0 | 4 × 8 × 15; stack 4 × 8 × 15 at `k_z` 1 and 2 × 8 × 15 at `k_z` 2, projection 1 × 8 × 15; 3 | one tile per plane; one chunk | yes |
| FC7 | `zstack`, different Z offsets: origins (0, 0, 0) and (1.4, 0, 6.25) | 1, 0 | 6 × 8 × 15; projection 1 × 8 × 15; 3 | as FC6 | yes |
| FC8 | `zstack`, Z = 1: two FOVs of 1×8×8 at origins (0, 0, 0) and (0, 0, 6.25) | 1 (`k_z` 1 and 2), 0 | 1 × 8 × 15; stack one plane at both `k_z`; 3 | as FC6 | yes |
| FC9 | `zstack`, equal Z, as FC6 | 2, 0 | 4 × 4 × 8 (partial last block in X); 3 | as FC6 | yes |
| FC10 | `zstack`, equal Z, `projection: sum`, on the uint16 fixture and on its float32 copy | 1, 0 | projection 1 × 8 × 15 in float64; 3 | as FC6 | yes |
| FC11 | `fuse_band`: six FOVs of 16×256×256 uint8 in one column at Y origins `230.4·r` (overlap 0.1) | 1, 2 | 16 × 1408 × 256; 1408 × 256; 10 (levels 704 × 128, 352 × 64) | projection 3 × 1 tiles, stack 3 × 1 tiles per plane; OME-Zarr projection chunks `(1, 1024, 1024)`: 2 × 1, stack chunks `(1, 16, 512, 512)`: 1 × 3 × 1; one automatic level (704 × 128) | yes (`slow`) |
| FR1 | FC1's pair | 2, 2 | `N` 2 × 3; 1 | – | refused (`pyramid_levels` above 1) |
| FR2 | FC3's quad | 4, 3 | `N` 4 × 4; 2 | – | refused (`pyramid_levels` above 2) |
| FR3 | FC3's quad | `scale_yx` 0.3 | – | – | refused (reciprocal not whole) |
| FR4 | FC6, `dimensionality: projection` | `scale_z` 0.5 | – | – | refused (`scale_z` with a projection) |
| FR5 | FC3's quad in uint64, int64, float16, bool and complex64 | 1, 0 | – | – | refused (dtype not allowed) |

| # | Check | Fixture (seeds) | Metric | Pass/fail tolerance | Needs |
| --- | --- | --- | --- | --- | --- |
| FI1 | Fusion grid | FC1, FC3, FC7, FC8, FC11; two Z = 1 FOVs of 1×4×4 at `origin_z` 0 and 1.4; `status_cases` (none) | `fused_grid` and `fused.json` | `o` is the smallest included origin per axis; `n0` equals the table's value for each configuration (the shape formula with its 1e-9 tolerance); one included FOV gives its own grid; excluded FOVs do not extend it; the two Z = 1 FOVs give `n0_z = 3`; with no included FOV no image is written and `fused.json` has `outputs: []` and `reason: "no_included_fov"`, which `check_fused` accepts only then. Exact. | H; S for FC11 |
| FI2 | The value of a voxel | FC1 (the counterexample), FC3 (none) | Whole stack and projection at the output level against condition 4's oracle | The image equals the oracle exactly: `A` (200, uint8) at `v*` when V1 chooses FOV `g` at the voxel centre `c*`, 0 elsewhere; on FC1 with FOV 2's voxel `(0, 1, 1)` bright the stack is 0 everywhere; on FC3, with each FOV's bright voxel in turn, `A` appears where the oracle says; a `placeholder` owner is skipped for the nearest-centre candidate with an image; for the stack a candidate must cover `c` in Z; the uncovered count equals the record's. Exact. | H |
| FI3 | OME-TIFF read-back | the files of FC1 to FC10, those of FC3 to FC5 in every dtype of the table "FI6 dtype cases", and, `slow`, FC11 (none) | Conditions 1 to 5 of "Read-back checks", OME-TIFF | BigTIFF and OME, one series, axes `YX` or `ZYX`; shape (the table's `N`, padded where levels are written), dtype and OME pixel type (the output dtype of "Dtypes written": the input dtype for the stack and a `max` projection, float64 for a `sum` projection, as the FI6 table lists), 512 × 512 tiles, `zlib`, the SubIFD count and shapes of the table (FC2: 2 × 3; FC3: 8 × 8, 4 × 4; FC4: 4 × 4, 2 × 2; FC5: 2 × 2; FC11: 704 × 128, 352 × 64) equal the record; `PhysicalSize*` and `Plane` positions equal the record's `S` and `O` within 1e-9 µm and `load_volume` restores that `ImageMetadata`; FC3 to FC5 and FC11 hold a region spanning an ownership boundary and uncovered voxels, FC4, FC5 and FC9 a partial last block, and each equals the oracle exactly at the output level and every SubIFD level (for FC3 to FC5 the values of the FI6 table); FC11's three tiles in Y are read one at a time; the file's SHA-256 equals the record's. The `starfinder.sample_output` annotation holds the sample, image name, dimensionality, projection rule, factors and the placement tables' SHA-256. | H; S for FC11 |
| FI4 | OME-Zarr read-back | FC1, FC3 and, `slow`, FC11 written with `formats: [ome_zarr]` (none) | Conditions 1 to 4, OME-Zarr | Zarr format 2, `multiscales` version `0.4`, axes, units; one dataset per level: FC1 and FC3 one (no automatic level, since 6 and 15 are at most 1024), FC11 two (1408 × 256, then 704 × 128); each dataset's `scale` and `translation` equal `S^(ℓ)` and `O^(ℓ)` within 1e-9; shape, dtype, chunks and compressor equal the record; values equal the OME-TIFF oracle exactly; on FC11 a region inside one chunk reads that chunk only (2 projection chunks and 3 stack chunks, keys counted). The group attribute `starfinder` holds the output level's `ImageMetadata`, the sample, `frame_id` and the placement tables' SHA-256. | SD; S for FC11 |
| FI5 | The 2D projection against the 3D stack | FC6, FC7, FC8 (none) | Projection and stack values | FC6 at `k_z` 1: `projection[0, y, x] == stack[:, y, x].max()` for every covered `(y, x)`; FC7: the projection equals the maximum over the owning FOV's own planes (oracle), and at every voxel where it differs from the stack's maximum a non-owner covers planes the owner does not or the Z resampling moves a plane (checked voxel by voxel); FC8: the stack's one plane equals the projection's values, and at `k_z` 2 the stack has `S_z = 2`, `O_z = 0.5` while the projection has `S_z = 1`, `O_z = 0`. Exact. | H |
| FI6 | Scale factor, levels, recorded geometry and dtypes | FC3, FC4, FC5 (the quad with one bright voxel, one copy per allowed input dtype, written as the stack, the `max` projection and the `sum` projection), FC9 (rounding), FC10 (sum on `zstack`) (none) | Conditions 1 to 3; the resampling rule; "Dtypes written" | `S = k·s`, `O = o + s·(k − 1)/2` per axis and the projection's Z slab (`S_z = n0_z·s_z`, `O_z = o_z + s_z·(n0_z − 1)/2`) within 1e-9 µm; `j = (j⁰ − (k − 1)/2)/k` within 1e-9 and `v = floor(v⁰/k)` exactly at positions at least 1e-6 voxel from a boundary, also for each written level `ℓ` with `k·2^ℓ`; the bright voxel's level-0, output-level and SubIFD-level values and dtypes equal the table "FI6 dtype cases" for every allowed dtype, and every other voxel is 0; FC9's values equal the rounding case of that table; FC10's projection equals `388 + 32·y + 4·x + 400·f` in float64 for both inputs, with `(y, x)` the owning FOV's own indices; levels padded to `2^L` as the configuration table states. Exact. | H |
| FI7 | Bounded memory of `fuse` | FC11 (none) | `tracemalloc` peak during `fuse` (NumPy reports its allocations to `tracemalloc`); peak RSS from `/usr/bin/time -v` recorded | Stack at scale 1: the traced peak is below two planes per FOV plus one 512-row band in float64 and one in uint8 (1.9 MiB), about a third of the level-0 stack (5.5 MiB), so holding the stack or all FOV volumes fails; projection: below the six FOV projections plus one and a half FOV volumes (1.9 MiB), so holding two FOV volumes at once (2.4 MiB with the projections) fails, while one volume with its projection (1.45 MiB) or the projections with one band in float64 and one in uint8 (1.5 MiB) pass; `fused.json` records the band size and the measured peak. | H, S |
| FI8 | Entry and config checks | FR1 to FR5; FC1, `status_cases` (none) | Raised errors | Checks 1 to 6 of "The fuse and export entry" raise in order (review, assembly placement hashes, inputs, grid, images with the dtype list, config); FR3, FR4 and FR5 raise `ValueError` from `FusionConfig` or check 5, and FR1 and FR2 from `fuse` once the grid is known, before any file is written; `FusionConfig` also refuses negative levels, empty lists and image names equal ignoring case; the sum-projection bound is evaluated before computing (tested on the bound's formula with a declared plane count of 2²¹ + 1 for uint32). An image name with a character outside letters, digits, `_`, `-` and `.` is refused; `formats: [ome_zarr]` or `spatialdata: true` without the `spatialdata` extra (import monkeypatched) raises `ModuleNotFoundError` naming the extra. Exact. | H |
| FI9 | Run record and the output profile | the files of FC1 to FC10; FC3 with a second source image (`reference_merged`, the same zero-and-one-voxel content); FC3 with `fuse: false` (none) | `fused.json`, the folder | Every written file is listed with SHA-256, format, image, dimensionality, projection, factors, shape, dtype, spacing, origin, levels, tiles or chunks and compression; a file present but unlisted, or listed but missing, fails `check_fused`; `formats: [ome_zarr]` with `spatialdata: true` writes no `fused/<sample>/*.ome.zarr`; the record says `viewer_qualification: "deferred"`. `fused.json` records the ownership source (`assembly` with its SHA-256, or `defaults`); `images: [reference_stain, reference_merged]` writes one file per image name and dimensionality on the same grid; `fuse: false` writes no `fused/<sample>/` folder; nothing is written under `images/fused/`. Exact. | H |
| FI10 | `Dataset.sample_outputs` | FC3 written as a dataset with saved placement and assembly (PL14, AS12); one FOV without its `reference_stain` file (none) | Files, errors | The files equal those of `write_sample_outputs` on the same inputs (identical SHA-256 except the record's timing fields); the FOV without its source image raises `ValueError` naming the FOV and the key `images`. `write_sample_outputs` calls `fuse`, `fuse_labels` and `export_spatialdata` in this order (spied, each only when its key is on) and returns the paths of the records it wrote. Exact. | H |

**FI6 dtype cases.** The bright voxel is FOV 1's voxel `p* = (1, 1, 1)` of the quad (FOV 1 at
origin `(0, 0, 0)`), so `q* = (1, 1, 1)` µm and `v* = (1, 1, 1)`. FOV 1 owns every voxel
centre of the 4 × 4 block in Y and X that holds `v*` (its footprint centre is at
`(3.5, 3.5)`, the nearest other centre at `(3.75, 9.75)` or farther), and every voxel of
that block is covered, so each block mean is `A` divided by the block's voxel count. `A`
is divisible by 64, so every value below is exact in its dtype and no rounding occurs;
the rounding rule is tested by FC9. Values are those of the bright voxel's block; every
other voxel is 0. Levels are the SubIFD (or OME-Zarr) levels the configuration writes.

| Input dtype `d` | `A` (`d` holds it) | Stack and `max` projection: dtype written | FC3 (`k_yx` 1): level 0, level 1, level 2 | FC4 (`k_yx` 2): output, level 1, level 2 | FC5 (`k_yx` 4): output, level 1 | `sum` projection: dtype written and values |
| --- | --- | --- | --- | --- | --- | --- |
| uint8 | 192 | uint8 | 192, 48, 12 | 48, 12, 3 | 12, 3 | float64; the same numbers as float64 |
| int8 | 64 | int8 | 64, 16, 4 | 16, 4, 1 | 4, 1 | float64; the same |
| uint16 | 64000 | uint16 | 64000, 16000, 4000 | 16000, 4000, 1000 | 4000, 1000 | float64; the same |
| int16 | 32000 | int16 | 32000, 8000, 2000 | 8000, 2000, 500 | 2000, 500 | float64; the same |
| uint32 | 4,000,000 | uint32 | 4,000,000; 1,000,000; 250,000 | 1,000,000; 250,000; 62,500 | 250,000; 62,500 | float64; the same |
| int32 | 4,000,000 | int32 | as uint32 | as uint32 | as uint32 | float64; the same |
| float32 | 1.5 | float32 | 1.5, 0.375, 0.09375 | 0.375, 0.09375, 0.0234375 | 0.09375, 0.0234375 | float64; the same |
| float64 | 1.5 | float64 | as float32 | as float32 | as float32 | float64; the same |
| Rounding (FC9, `zstack` uint16, `k_yx` 2, stack) | the fixture's values | uint16 | – | every full 2 × 2 block inside one FOV's ownership region has the mean `n + 0.5` with `n = 5 + 64·z + 16·i + 2·j + 100·f` odd, `(i, j)` the block's position in the owning FOV's own indices; it is written as `n + 1` (half to even) | – | – |

Every `A` lies within its dtype (int8 holds 64 of at most 127, int16 32000 of at most
32767, int32 4,000,000 of at most 2³¹ − 1), and every value of the stack, the `max`
projection and FC9 lies within its dtype. The `sum` projection's values are float64 for
every input dtype, as "Dtypes written" requires, and the one-plane sum equals `A` because
only one voxel of the column is nonzero.

### SpatialData export (SD)

The eight round-trip conditions are those of "Read-back checks", SpatialData, of
{doc}`sample-output-contract`; each row names the one it implements. SD9 and SD10 are the
measurement and the fallback that "Bounded memory" of that page asks for inside the
store. Image elements come from the accepted fusion configurations above.

| # | Check | Fixture (seeds) | Metric | Pass/fail tolerance | Needs |
| --- | --- | --- | --- | --- | --- |
| SD1 | Elements and omissions (condition 1) | `export_cases`, one per row of "Empty populations", with FC1 or FC3 images (none) | Store contents against `export.json` | Names and kinds equal the `written` entries; no two names equal ignoring case; each `omitted` reason recomputed from the tables, the fusion record and the configuration; `store: null` exactly when every element is omitted; one image element per fused image and no other OME-Zarr group of it; Zarr format 2, `multiscales` version `0.4-dev-spatialdata`. A sample name outside the element-name rule is refused before anything is written. Exact. | SD |
| SD2 | Points (condition 2) | `export_cases` (none) | `molecules` against the sample molecule table | Joined on `(spot_namespace, spot_id)`: `x, y, z` equal `sample_x/y/z` exactly, `fov_index_z/y/x` equal the §2.9 `z, y, x`, every other column equal (NaN equals NaN), no `sample_x/y/z` column left; with `molecules_unplaced`, every source row exactly once; dtypes as `export.json` lists. Exact. | SD |
| SD3 | Tables preserve their source (condition 3) | `export_cases` (none) | `cells`, `cells_unplaced`, `molecules_unplaced`, `expression` | Every source column of the cell table and every row exactly once; `expression` equals `sample.h5ad` in `X`, `layers`, `var`, `obsm`, every source `obs` column and every source `uns` key; dtypes after the round trip as recorded. A table with a column name or key outside the element-name rule, or two equal ignoring case, is refused naming the key. Exact. | SD |
| SD4 | Export metadata, separately (condition 4) | `export_cases` (none) | Added columns and keys | `region` (categorical, `cell_centroids`) and `instance_id` (uint32) on `cells` and `expression`, `uns["spatialdata_attrs"]` as specified, nothing else added; the `_unplaced` tables carry only `instance_id` or `cell_instance_id` and no `spatialdata_attrs`. Exact. | SD |
| SD5 | Links and accounting (condition 5) | `export_cases` (none) | IDs and counts | `instance_id` is `1..N` in the cell table's order; `cells` and `cell_centroids` written together with equal IDs; every molecule's `cell_instance_id` matches its cell key, null without one; every `expression` row is a counted cell; molecule counts by `assignment_status` and by `assembly_status` and reason equal `export.json` and identity 2 of {doc}`assembly-contract`. Exact. | SD |
| SD6 | Calibration and alignment (condition 6; "Sample position to image position", condition 5) | FC1, FC3 exported (one level each); FC11 exported, `dimensionality: both` (two levels: 1408 × 256 and 704 × 128) (none) | Element transforms, extents, point lookup | Each image transform equals `Scale(S)` then `Translation(O − S/2)` within 1e-9, and FC11's level 1 `Scale(2·S)` then the same translation; extents `[O − S/2, O + (N_padded − ½)·S)`; points and shapes have the identity; on FC1 and FC3 a point at `q*` falls in the intrinsic voxel that holds `A` exactly when condition 4's oracle says so. | SD; S for FC11 |
| SD7 | Values, written once (condition 7) | `export_cases` with FC3 images, and FC11 with `dimensionality: both`, each with and without `formats: [ome_tiff]` (none) | Store image against the OME-TIFF (or the oracle without it) | Output levels equal on the shared extent, 0 in the padding; same dtype; FC11's level 1 equals the 2 × 2 block mean of its output level, as FI4; FC11's projection spans two chunks and its stack three. Exact. | SD; S for FC11 |
| SD8 | Hashes, staging and lookup (condition 8) | `export_cases`; an export interrupted after its first element (a test hook raising) (none) | Files, `lookup_source` | Every store file's SHA-256 equals `export.json`; the interrupted export leaves no `export/<sample>/` folder; `lookup_source(spot=…)` and `(cell=…)` return the FOV, the assignment folder, its `assignment.json` SHA-256 and the row's position. Exact. | SD |
| SD9 | Bounded memory of the store writer | FC11 with `dimensionality: both`, `spatialdata: true`, in one process with one `dask` thread (none) | `tracemalloc` peak while `export_spatialdata` writes the image elements (spatialdata's one `dask` computation of every level); peak RSS from `/usr/bin/time -v`; the path used | The traced peak during the image write is below the level-0 stack's bytes (5.5 MiB), so no path holds the whole output level of the stack at once; `export.json` records the peak, the RSS and which path wrote the levels (`lazy` or `scratch`). | SD, S |
| SD10 | The scratch fallback | FC11 as SD9, with the fallback forced by the export's test switch (none) | Values, temporary file, traced peak | The output level is written to a temporary array under the local scratch directory (`TMPDIR`), never on the artifact mount, and the levels are computed from it; the store's values and levels equal SD7's exactly; the temporary file is removed after the export, also when the export fails; the traced peak is below SD9's bound; `export.json` records `scratch`. | SD, S |

### Assembled label export (LB), for the later delivery

Only if the gate decides the label export ({doc}`sample-output-contract`, option LD2
recommends a later delivery). `quad_labels` is FC3's quad geometry with hand-built uint32
territories: box cells, one of which lies in the overlap of all four FOVs with a
different territory in each, and one cell of one FOV that a neighbour's territory holds as
background.

| # | Check | Fixture (seeds) | Metric | Pass/fail tolerance | Needs |
| --- | --- | --- | --- | --- | --- |
| LB1 | Values and placement | `quad_labels`, fusion grid 3 × 15 × 15 at `k_yx` 1, no level (none) | Label voxels against the oracle of "The value of a voxel" on territories | Every nonzero value is an `instance_id`; every fusion-grid voxel equals the chosen FOV's territory value mapped to its `instance_id`. Exact. | H; SD for LP1 |
| LB2 | Reduction and lost IDs | `quad_labels` at `k_yx` 2 (`N` 8 × 8, largest accepted levels 3) with one level (4 × 4), and at `k_yx` 3 (`N` 5 × 5, largest 2) with one level (padded to 6 × 6, level 3 × 3) (none) | Block mode, ties to the smallest, lost IDs | Each level equals the block mode; every ID missing from a level is listed in `labels.json`. Exact. | H |
| LB3 | Conflicts and the no-cell cases | `quad_labels`; `assembly_golden` (2D) at `k_yx` 1, no level; `empty_cases` (none) | `labels.json` | The conflict pairs and background-over-cell counts recomputed from the territories equal the record; `molecules_only` and no included FOV record their reasons and write no image. Exact. | H |
| LB4 | Bounded memory of `fuse_labels` | FC11's geometry with uint32 territories of 16×256×256 per FOV (none) | `tracemalloc` peak during `fuse_labels` | Below two planes per FOV plus two 512-row bands in uint32 (4 MiB), about a sixth of the level-0 label volume (22 MiB), so no full-sample mask is held in memory. | H, S |

### Workflow (WF)

| # | Check | Fixture (seeds) | Metric | Pass/fail tolerance | Needs |
| --- | --- | --- | --- | --- | --- |
| WF1 | Schema equals the configs | `wf_configs` | `workflow/schemas/config.schema.yaml` | The `stitching` method list equals `STITCHING_METHODS` plus `bigstitcher`; the `assembly` keys equal the `AssemblyConfig` fields and `Dataset.assemble` arguments; the `sample_outputs` keys equal `FusionConfig` and `SampleOutputConfig`; the configurations validate. Exact. | H |
| WF2 | Legacy configurations unchanged | `wf_configs` without the new blocks | Rules, outputs | The four legacy stitching rules and the legacy assembly rules run unchanged (`test_assignment_workflow.py`, `test_workflow_stitching_golden.py`, `test_workflow_assembly_golden.py` pass unchanged); the adapter's legacy translation gives `StageAxisMapping(x="+x", y="+y", source="legacy")`. Exact. | H |
| WF3 | The new blocks | `wf_configs` with the blocks, through a stub `snakemake` object | Adapter calls and files | `_run_stitching`, `_run_assembly` and `_run_sample_outputs` call the package entries with the configured values and write the files of the three persistence layouts; with an `assembly` block the per-FOV files have no overlap filter and no `global_*` (W1); a stage file without `stage_axes` raises naming the key; `method: bigstitcher` without `bigstitcher_save_dir` raises. Without a `sample_outputs` block the defaults run; the `assemble_sample` rule's inputs are the sample's per-FOV assignment outputs and its `placement.json`; parsing the rules without `fiji_path` succeeds, and the BigStitcher engine rule raises naming `fiji_path` only when it runs. Exact. | H |

### What the default tier covers and what it does not

PL1 to PL15, PL17, AS1 to AS13, AS15, FI1 to FI3 (their FC1 to FC10 parts), FI5, FI6, FI8 to FI10, LB1
to LB3 (when built) and WF1 to WF3 run in the routine tier on hand-built geometry or the
copied records. PL4's scene part and EE1 to EE5 run in the routine tier once the generator
addition exists (`sample-tiny-v1` is inside the routine bound, {doc}`synthetic-sample-scene`).
PL16, AS14's three processes, the FC11 parts of FI1, FI3, FI4, SD6 and SD7, FI7, SD9,
SD10, LB4, MV1 to MV6 and EE6 are `slow`. MV1 to MV6 and EE6 need the
`multiview-stitcher` extra; FI4 and SD1 to SD10 need the `spatialdata` extra. Nothing here
measures how well a backend places real tiles or how well a sample's cells are
represented; those are questions for E07 and for annotation.

**Which checks need the generator addition (W-327):** PL4 (its scene part), MV3, EE1 to
EE6. **Which run on hand-built geometry alone:** every PL row except PL1's LN files,
PL4's scene part, PL11's and PL14's import parts, PL12 and PL13; every AS, FI, LB and WF
row. **Which run on the cut fixtures or copied records alone:** PL1 (LN stage file),
PL11 and PL14 (import parts), PL12, PL13 (copied records); MV1, MV2, MV4 to MV6 (the
cut-fixture recipe, rebuilt in the test).

### Resource plan

The bounds are the §2.10 resource exceptions recorded in W-323 (agreed with Jiahao on
2026-10-05) for tests:

* **Routine tests**: synthetic tiles cut from one scene of at most 16×256×256; one CPU,
  4 GiB, 1800 s. Every routine fixture here is far below it: the largest are the scene of
  `sample-tiny-v1` (sample grid 6×63×63), `assembly_golden` (three FOVs of 8×48×64) and
  `stub_grid`; each check is expected well under a second.
* **`slow` tests**: a synthetic 3 × 3 grid of tiles of 16×256×256; one CPU, 4 GiB,
  1800 s. `cut_synthetic` holds one 16×256×256 float32 scene (4 MiB) and nine tiles of
  14×120×120 uint16; W-324 measured 12 to 78.5 s wall and 0.29 to 1.02 GiB peak RSS per
  fixture row at one thread, most of it the import of multiview-stitcher (W-324 notes
  section 2.5). `scene_small` is nine FOVs of 14×96×96, estimated below 1 GiB
  ({doc}`synthetic-sample-scene`, "Presets and resources"). `fuse_band` and LB4 hold six
  tiles of 16×256×256 (6 MiB in uint8, 24 MiB in uint32). Every `slow` check runs at one
  thread, MV6 included.
* Each test run records wall time and peak RSS with `/usr/bin/time -v`; a run above a
  bound is a failure to report, not a reason to raise the bound.

## Whole-sample real examples

After the implementation, the three whole samples of W-323 are run through the new path
once, as inspections. They are not a comparison and rank nothing; no truth exists for
them. Every input is a file named in W-323 `20261005T2110Z-input-inventory/inputs.md` or
`20261005T1930Z-fiji-baseline/inputs/`, read-only, with its SHA-256 checked against those
records.

| Sample | Placement | Source image for fusion | Volumetric |
| --- | --- | --- | --- |
| LN sample1 (64 FOVs of 50×1496×1496 uint8, DAPI, plus 8 blank grid cells) | The 2024 `dataset.xml` through `import_bigstitcher` (no Fiji rerun), and `multiview_stitcher` in the starting configuration from the grid start | the stitching input tiles, `file:` pattern | yes |
| tissue-2D (56 FOVs of 3072×3072 uint8, `ref_merged`) | `multiview_stitcher`, starting configuration, grid start (no BigStitcher reference exists; the Grid/Collection file is monitored only) | `reference_merged` | no (one plane) |
| cell-culture-3D (70 FOVs of 30×1496×1496 uint8, `ref_merged`) | as tissue-2D | `reference_merged` | yes |

**Grids.** The tile files carry no calibration (W-323 `inputs.md`), so each FOV's grid is
declared from the raw-acquisition calibration recorded there (0.194398 µm in x and y for LN
and cell-culture-3D, 0.094635 µm for tissue-2D, 0.346255 µm in z), recorded as
`declared`, and the package functions are called with these `StitchingInput`s directly
rather than through `Dataset.stitch`, which reads grids from saved reference images.

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
an estimate, and an assembly without inputs is reported as not run. Which cell inputs to
use is choice 11 of the W-331 worker notes.

**Steps and their limits.** The §2.10 resource exceptions recorded in W-323 set a separate
limit for each step; W-323 states that its numbers for the real examples are estimates
from file sizes, not measurements, and that placement uses the CPU only.

| Step | What runs | Limit |
| --- | --- | --- |
| Placement | Per sample and backend: LN sample1 through the import of the 2024 `dataset.xml` and through `multiview_stitcher`; tissue-2D and cell-culture-3D through `multiview_stitcher`; once, W-324's real_2x2 cut fixture of one LN tile; the LN `multiview_stitcher` run repeated at one thread, for the thread-count part of MV6 | 4 CPU threads, 32 GiB RSS, 7200 s per backend and sample; CPU only |
| Assembly | Per sample: assembly in `molecules_only` mode from the inventoried molecule tables, and with cells where cell inputs exist | one CPU, 16 GiB RSS, 3600 s per sample |
| Sample outputs | The default output (projection at scale 0.5 and `sample.h5ad`) of all three samples; the 3D stack of LN sample1 in both formats, as OME-TIFF and as OME-Zarr inside its SpatialData store (`dimensionality: both`, `spatialdata: true`, `scale_yx` 0.5); the SpatialData export of LN sample1 | 4 CPU threads, 32 GiB RSS; new artifacts at most 20 GiB per run, written to the artifact mount; no time limit is listed, and each run is timed |

The exceptions grant the 3D stack of one whole sample only. The seam views through depth
of cell-culture-3D therefore come from the 3D stack of a 2 × 2 block of its FOVs around one
grid corner, fused by the same rule from those four FOVs' placement rows (a bounded region
of the sample-outputs step), not from a whole-sample stack; those of LN sample1 come from
its whole stack.

**Outputs to inspect**, per sample, in the run directory (at most ten PNGs in total; eight
are planned):

* **Placement overview** (one PNG per sample): the FOV footprints at their origins,
  coloured by status, links coloured by quality with dropped links dashed, correction
  arrows (scaled and labelled), and the review state with its reasons; the placement
  record's `fovs.csv` and `links.csv`.
* **Seam views through depth** (LN and cell-culture-3D, one PNG each): at three seams (a
  column seam, a row seam and a corner), XY crops of the 3D stack at the first, middle
  and last quarter planes and one XZ and one YZ section across the seam, with the
  ownership boundary drawn; for LN from its whole stack at `scale_yx` 0.5, for
  cell-culture-3D from the stack of the 2 × 2 block at `scale_yx` 1.
* **Ownership regions with retained and excluded observations** (one PNG per sample): the
  O1 regions over the default projection, molecules coloured by assembly status and
  reason, retained and excluded cell centroids, and the `conflict`, `unclaimed` and
  `outside_owner_region` flags; their frequencies are the first measurement of these flags
  on real samples (W-326 notes).
* **Accounting tables**: the `fovs` table and `record["counts"]` (inputs, retained, excluded
  by reason, flag counts) per sample, as CSV and JSON.
* **The default sample outputs**: `sample.h5ad`, `fused/<sample>/fused.json` and the
  projection OME-TIFF at scale 0.5 of each sample, with `check_fused` passing; for LN
  sample1 also the stack OME-TIFF and the SpatialData export with its stack and projection
  image elements, with `check_spatialdata_export` passing and the SD9 measurement of the
  store writer's peak memory recorded.
* **Monitored agreement** (LN only): the per-axis differences between the imported
  BigStitcher placement and the `multiview_stitcher` placement, against the monitored
  bounds of "Tolerances and where they come from"; reported, never gating. The real_2x2
  cut fixture of W-324 (one LN tile, seed 325, noise seed 1100 + tile) is run once with
  `multiview_stitcher` against MV1's bound.

**Resources per sample and step.** Placement figures are W-324 measurements; assembly
figures are estimates from {doc}`assembly-algorithms` ("Cost": about 100 bytes per molecule
row with the §2.9 columns, and at most two territory images in memory at a time), pending
the inventory of step 1; output figures are W-328 estimates
({doc}`sample-output-contract`, "Resources"). Every run is measured with
`/usr/bin/time -v`; a run above its step's limit is recorded as `resource_stop` and not
retried with a higher limit.

| Sample | Placement (limit 4 threads, 32 GiB, 7200 s) | Assembly (limit one CPU, 16 GiB, 3600 s), an estimate | Default outputs (limit 4 threads, 32 GiB) | Stack and export (same limit; artifacts at most 20 GiB per run) |
| --- | --- | --- | --- | --- |
| LN sample1 | import: one XML parse; `multiview_stitcher`: 145 to 182 s, 7.0 GiB peak RSS, loading 58 to 81 s | molecule rows unknown until step 1: about 1 GB per 10⁷ rows, plus two territory images of 50×1496×1496 uint32 (0.45 GB each) when cells are assembled; time measured | reads 7.16 GB of sources once; projection 0.031 GiB; about 254 MiB by the memory model; fusion work about 6 s at the probe's TIFF rate | the whole stack at `scale_yx` 0.5: 68 × 6,167 × 5,442, 2.1 GiB uncompressed in each of the two formats (OME-TIFF, and OME-Zarr in the store), about 440 s of fusion work at the probe's TIFF rate; the store's tables, sized after step 1; under 5 GiB of artifacts |
| cell-culture-3D | 65 to 78 s, 4.69 GiB | as LN, with territory images of 30×1496×1496 uint32 (0.27 GB each) | 4.70 GB read; 0.030 GiB; about 223 MiB | the 2 × 2 block's stack at `scale_yx` 1, about 45 × 2,842 × 2,842 (0.34 GiB uncompressed; the Z extent depends on the placement's z corrections); under 0.5 GiB of artifacts |
| tissue-2D | 44 to 49 s, 0.83 GiB | as LN, with territory images of 3072×3072 uint32 (38 MB each) | 0.53 GB read; 0.104 GiB; about 532 MiB | none (one plane) |

By the assembly estimate a sample fits in its 16 GiB limit up to about 10⁸ molecule rows;
the inventory of step 1 replaces the estimate with the row counts. The fusion times are
upper estimates extrapolated from W-324's multiview-stitcher probe; the issue measures
them. At most ten PNGs are written, eight are planned, and temporary files go to local
scratch; nothing is written into a data directory.

## Limitations

* The tolerances for `multiview_stitcher` come from two cut fixtures with integer offsets
  and one host; MV3 extends them to fractional offsets by an argument about quantization,
  not a measurement, and the non-square case of MV4 was not measured by W-324.
* PL12's review state of the 2024 LN placement under V1 is not known before the import
  exists: `dataset.xml` does not mark the two links BigStitcher's optimization removed.
* The scene fixtures exercise the rules on generated geometry; duplicates and losses on
  real samples are only inspected, not measured against truth.
* The molecule and cell tables of the real samples are not inventoried; their assembly
  figures stay estimates until step 1 of the real examples.
* Viewer qualification stays deferred; no check opens a file in a viewer.
