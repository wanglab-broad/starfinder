# Stitching algorithm specification

Status: Proposed

This page specifies the numerical behavior of each element of {doc}`stitching-contract`:
the three converters into the tile position table, the neighbour derivation, the
`multiview_stitcher` method, the import arithmetic of BigStitcher's `dataset.xml`, and the
placement-quality checks. Each has its parameters, units, defaults, failure behavior and
a resource estimate. The current behavior is in {doc}`stitching-baseline`. Nothing here is
implemented, and no default here is accepted.

## Evidence

Every measured number is cited from W-323
(`/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/runs/W-323/`) and W-324
(`/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/runs/W-324/20261006T063540Z-5cb2f421`,
`notes.md`, `tables/runs.csv`, `tables/comparison_summary.csv`,
`tables/fixtures_summary.csv`, `tables/pairs_between_runs.csv`). W-324 measured
multiview-stitcher 0.1.62 on GP099 (Python 3.12.12) with the raw-acquisition calibration
(0.194398 µm in x and y for LN and cell-culture-3D, 0.094635 µm for tissue-2D, 0.346255 µm
in z), translation only, on the CPU. W-324 ranks no backend and makes no accuracy claim;
neither does this page.

**How to read the resource figures.** Wall and CPU time and peak RSS are from
`/usr/bin/time -v` per process; loading the tiles from the CIFS share is reported apart
from registration where W-324 measured it. Three whole samples, one host, one Python
version: the figures are an order of magnitude for planning, not a bound.

## Converters

All three return a `TilePositionTable` in µm ({doc}`stitching-contract`, "Tile position
table") and read no image. Their cost is a parse of a small text file or a loop over the
FOVs; it was not measured and is negligible beside registration.

### Leica stage file

`positions_from_leica(path, *, stage_axes, fov_ids, snap_to_grid=None)`

1. Read every `XYZStagePointDefinition` of the LAS X file: `PositionID` (int),
   `StageXPos`, `StageYPos` (metres, float64). Other attributes (`Sections`, `StackBegin`,
   `StageZPos`) are kept in the record, not used.
2. Convert to µm (× 1e6) and apply `stage_axes`: stage +X goes to the image axis and sign
   it names, likewise stage +Y. For LN, `x="+x"`, `y="+y"` (W-323 `positions.md`; fit
   residual RMS 0.29 µm in x and 0.34 µm in y).
3. Shift so that the smallest y and x are 0; z is 0 for every FOV.
4. Map `PositionID` to `fov_id` through `fov_ids`.

With `snap_to_grid=GridSnap(overlap=…, tile_shape_yx=…, spacing_yx=…)`, the positions are
replaced by a regular grid, which reproduces the start of today's BigStitcher macro and of
W-324's main configuration. The converter reads no image: the tile size and spacing are
`GridSnap` fields, given as values. `Dataset.stitch` fills them, when the configuration
does not give them, from the FOVs' `ReferenceGrid` (`shape_zyx[1:]` and
`metadata.spacing_zyx[1:]`) and raises `IncompatibleGeometryError` if the FOVs of the
sample do not share one tile size or spacing ({doc}`stitching-contract`, "Coordination on
the dataset").

* **Step for the cells.** Differences of consecutive positions along the scan's fast axis
  (y for a column scan), rounded to 0.1 µm; the step is the most frequent absolute
  difference; a tie goes to the smaller value. (The legacy rule is the most frequent exact
  float difference, signed, with the first value in the counts' order winning a tie,
  {doc}`stitching-baseline` G2; the rounding removes the last-bit ties W-324 recorded in
  `maf_2x3_jitter`. On LN both give 262.0 µm.)
* **Cells.** `column = floor((x − min x)/step + 0.5)`, `row = floor((y − min y)/step +
  0.5)`, the legacy rounding written for every sign.
* **Nominal positions.** Those of `positions_from_grid` for the cells, with step
  `(1 − overlap) × tile_shape_yx × spacing_yx` per axis, in µm (LN: 0.9 × 1496 px =
  1346.4 px, times 0.194398 µm), and `source` `grid`.

| Parameter | Unit | Default | Rule |
| --- | --- | --- | --- |
| `stage_axes` | | none (required) | `StageAxisMapping`; the two entries name different image axes |
| `fov_ids` | | none (required) | every `PositionID` of the file must be a key |
| `snap_to_grid` | | `None` | a `GridSnap` with the three fields below, or `None` to keep the stage positions |
| `GridSnap.overlap` | fraction | none (required) | `0 ≤ overlap < 1`; one value or `(y, x)` |
| `GridSnap.tile_shape_yx` | px | none (required) | integers ≥ 1; the reference grids' Y and X sizes, as for `positions_from_grid` |
| `GridSnap.spacing_yx` | µm per px | none (required) | finite, > 0; the reference grids' Y and X spacing, as for `positions_from_grid` |

Failure: a repeated `PositionID`, a `PositionID` without a FOV id, a non-finite position,
or (with snapping) two positions in one cell raise `ValueError` naming them. The legacy
script summed two such ids into one cell.

### Regular grid

`positions_from_grid(*, columns, rows, order, tile_shape_yx, spacing_yx, overlap, fov_ids)`

* FOV `k` of `fov_ids` goes to a cell by `order`: `column_major_down_right` (column
  `k // rows`, row `k % rows`; the scan of tissue-2D and cell-culture-3D, W-323
  `positions.md`), `row_major_right_down`, and their serpentine forms.
* Nominal `y = row × (1 − overlap_y) × tile_y × spacing_y`, `x = column × (1 − overlap_x)
  × tile_x × spacing_x`, `z = 0`.

| Parameter | Unit | Default | Rule |
| --- | --- | --- | --- |
| `columns`, `rows` | | none | `columns × rows ≥ len(fov_ids)` |
| `order` | | none | one of the four orders |
| `tile_shape_yx` | px | none | integers ≥ 1; the reference grids' Y and X sizes (`Dataset.stitch` takes `ReferenceGrid.shape_zyx[1:]` when not given) |
| `spacing_yx` | µm per px | none | finite, > 0; the reference grids' spacing (`Dataset.stitch` takes `ReferenceGrid.metadata.spacing_zyx[1:]` when not given) |
| `overlap` | fraction | none | `0 ≤ overlap < 1` |

Failure: more FOVs than cells, or an unknown order, raise `ValueError`. The y step is
nominal: W-323 found the registered y step 0.3 % (culture) and 0.5 % (tissue) larger, up
to 46 and 153 px over a column, which the method must correct.

### Supplied table

`positions_from_table(path_or_frame, *, unit="µm")` reads `fov_id`, `nominal_z`,
`nominal_y`, `nominal_x` and converts them to µm (`µm`, `um`, `nm`, `mm` accepted). A
repeated or empty `fov_id`, a missing column or a non-finite value raises `ValueError`.

## Neighbour derivation

`neighbor_pairs(table, grids, *, tolerance_um=0.0)`

* Box of FOV `i` per axis: `[nominal_i, nominal_i + shape_i × spacing_i]`, grown by
  `tolerance_um` on each side.
* `(a, b)` with `a` before `b` is a pair when the boxes intersect in y and x with positive
  area and, for Z>1, in z with positive length. The nominal overlap box is the
  intersection without the tolerance.
* A sweep over x after sorting costs `O(n log n + pairs)`; LN has 64 FOVs.

| Parameter | Unit | Default | Rule |
| --- | --- | --- | --- |
| `tolerance_um` | µm | 0.0 | ≥ 0; covers stage error when the nominal boxes only touch |

multiview-stitcher's own overlap graph found 213 pairs on LN from the grid start, 181 on
tissue-2D and 231 on cell-culture-3D, diagonals included (W-324 notes section 2.1). The
rule here is the same box intersection with zero tolerance; the implementation checks that
it gives the same pairs on its fixtures. No pair is derived from row or column indices.

## `multiview_stitcher`

One call per sample. Steps, as W-324 ran them (notes sections 3 and 5):

1. **Views.** Each tile becomes a one-level image with `spacing` from its reference grid
   and its nominal origin as the array translation (`get_sim_from_array(…,
   translation=…, scale=…)`); a Z=1 tile is 2D. Spacing and translation are in µm.
2. **Pairs.** The library's overlap graph, pruned by `pruning` (`alternating_pattern`
   registers 111 of LN's 213 pairs). The registered pairs must be a subset of the
   contract's neighbour pairs; the others are recorded as `pruned`.
3. **Pairwise registration.** Phase correlation on the overlap, mean-binned by
   `binning_zyx` voxels. Each shift is quantized to `binning / upsample_factor` voxels
   (library default `upsample_factor` 2 in 3D, 10 in 2D): at binning 4 the shifts cluster
   on multiples of 2 voxels.
4. **Link quality.** The Spearman rank correlation of the overlaps after the shift; NaN
   for a constant overlap (the link then gets the identity).
5. **Link filter (L1).** The library's filter at `quality_threshold`, plus dropping NaN
   links, which the library keeps.
6. **Global optimization.** Per component, one reference view fixed; translations solved
   from virtual bead pairs at the overlap corners; while the largest residual exceeds
   `abs_tol`, the link with the largest `(1 − q)² · √residual · log10(degree)` whose
   removal keeps the component connected is removed.
7. **Result.** Per view, the translation of its affine under the new transform key (µm,
   view to common frame, fractional), its component, the links with quality, the kept
   flag, and the links the optimizer removed.

| Field of `MultiviewStitcherConfig` | Unit | Default | Rule |
| --- | --- | --- | --- |
| `method` | | `"multiview_stitcher"` | discriminator, equals the spec name |
| `binning_zyx` | voxels | `(4, 4, 4)` | integers ≥ 1; ignored for z on Z=1 |
| `pairwise` | | `"phase_correlation"` | the only value in §2.10 |
| `upsample_factor` | | `None` (library default) | integer ≥ 1 |
| `quality_threshold` | | `0.7` | in [−1, 1] |
| `drop_nan_quality` | | `True` | |
| `pruning` | | `"alternating_pattern"` | or `None` (register every pair) |
| `overlap_tolerance_um` | µm | `0.0` | ≥ 0 |
| `groupwise` | | `"global_optimization"` | the only value in §2.10 |
| `abs_tol_um`, `rel_tol`, `max_iter` | µm, –, – | `None` (library defaults; `abs_tol` is the voxel diagonal, 0.44 µm for LN) | |

**Behavior W-324 recorded.**

* *Determinism:* identical translations, bit for bit, for repetitions and between one and
  four threads on every sample, and for the 24 fixture reruns (`pairs_between_runs.csv`).
* *Known-answer cut fixtures:* at binning 4, max error 1.09 voxels (synthetic 3 × 3) and
  0.75 (real 2 × 2); at binning 2, below 1e-12 on both (`fixtures_summary.csv`). Integer
  offsets and identical signal in every overlap make these cases easier than real data.
* *LN against BigStitcher,* grid start, this configuration with the filter at 0.7: |d| max
  8.04 px in x, 1.77 px in y, 2.77 slices in z, mostly a trend of 1.69 px per column in x
  (`comparison_summary.csv`). This is agreement, not accuracy.
* *LN from the stage positions:* one component, but 19 FOVs about 147 px off in x, with 21
  links removed by the optimizer and no error raised. The contract therefore starts from
  the snapped grid and bounds corrections ("Placement-quality checks").

**Failure behavior.** No link left at all: the library raises `NotEnoughOverlapError`,
and every FOV is `failed`. A FOV without a kept link to the reference component: its own
component with the identity, so it would stay at its nominal position silently; the
wrapper makes it `failed` (or `metadata_only` with the fallback). A non-finite
translation: `failed`. A constant tile: its links are NaN and dropped, so it is
disconnected. The library's 3D adjacency helper raises on one-plane 3D views
(W-324 notes section 2.6); the wrapper passes Z=1 tiles as 2D.

**Resources** (W-324 notes section 2.2; ranges over the runs at one and four threads, without the `upsample_factor` 10 run, which took 191.5 s on LN):

| Sample | FOVs, shape | Load wall (s) | Register wall (s) | Total wall (s) | Peak RSS (GiB) |
| --- | --- | --- | --- | --- | --- |
| LN sample1, binning 4 | 64 × 50 × 1496 × 1496, uint8 | 57.7 to 81.2 | 75.6 to 100.0 | 145.0 to 181.6 | 7.00 |
| LN sample1, binning 2 (four threads, one run) | the same | 59.6 | 658.6 | 729.8 | 7.37 |
| cell-culture-3D, binning 4 | 70 × 30 × 1496 × 1496 | 11.8 to 20.3 | 41.5 to 48.2 | 65.5 to 77.5 | 4.69 |
| tissue-2D, binning 4 | 56 × 3072 × 3072 | 6.5 to 9.3 | 26.4 to 29.9 | 44.5 to 49.0 | 0.83 |

Memory is the loaded tiles plus 0.1 to 0.5 GiB. Four threads barely shorten a run,
because 3D pairs are registered one at a time. A sample of `n` tiles of `V` bytes each
needs about `n × V` plus 0.5 GiB with eager loading; lazy loading through dask is
possible but was not measured.

## Import arithmetic

`import_bigstitcher(dataset_xml, *, table, view_map, grids, config, review, fallback)`

For each `ViewRegistration` (setup `v`, timepoint 0):

1. **Transforms.** Read every `ViewTransform` by its `Name`. Each `affine` is twelve
   numbers, the row-major 3 × 4 matrix in XYZ. Compose all transforms except
   `calibration`, in file order (newest first), as 4 × 4 matrices. If the composed linear
   part differs from the identity by more than 1e-9, raise `ValueError` naming the view
   and transforms: §2.10 admits translation only. A view without a `Stitching Transform`
   fails the engine task before the import (Fiji's exit status does not report it).
2. **World translation** `w_v = (w_x, w_y, w_z)`: the composed translation. For
   translations this is the sum of `Stitching Transform` and `Translation to Regular Grid`
   that `create_tile_config.py` computes (lines 39-41), in float64, with no cast.
3. **Voxel offset.** The `calibration` transform is `diag(c_x, c_y, c_z)`; BigStitcher
   scales to the smallest voxel size, so `c_x = c_y = 1` and `c_z = voxel_size_z /
   voxel_size_xy` of the macro (1.7894736842105263 for 0.34 / 0.19). The offset in the
   tile's voxels is `u_v = w_v ⊘ (c_x, c_y, c_z)`; z is now in slices.
4. **µm.** `origin_raw = (u_z s_z, u_y s_y, u_x s_x)` with the FOV's reference grid
   spacing `s`. The macro's voxel size only fixes BigStitcher's world units, so rounding in
   the YAML values (0.19 against 0.194398 µm) does not enter the origin.
5. **Gauge** (G1): `c_v = origin_raw_v − nominal_v`; `origin_v = nominal_v + c_v −
   mean(c)` over the `image_supported` FOVs.
6. **Links.** Each `PairwiseResult` (`view_setup_a`, `view_setup_b`, `shift`,
   `correlation`, `overlap_boundingbox`) becomes a link: the shift's translation converted
   as in steps 3 and 4, quality = correlation, kept (the file holds the links that passed
   the macro's filter). The `hash` and the bounding box are recorded. `dataset.xml` does not
   record which links the global optimization removed; the residual check finds them.
7. **Views.** A view mapped to an acquired FOV gives that FOV's row; a padding view keeps
   only its raw `w_v` and grid cell in the method record; a placeholder view gives its FOV
   the status `placeholder`. A FOV of the table without a view is `missing`.

**Status from links.** An acquired view with at least one link into the reference
component is `image_supported`. One without (a view BigStitcher's second round placed from
metadata) follows the fallback rule of `stitch`: with `fallback=None`, the default, its FOV
is `failed`, its origin NaN and its `w_v` kept in `method_translation_*`; with
`fallback="nominal"`, it is `metadata_only` at `origin = nominal` (correction 0), flagged
`fallback`, with `w_v` kept the same way. Review is `required` in both cases. The gauge of
step 5 uses only `image_supported` FOVs, so the choice changes no other origin.

**The legacy view** applies T3 to T7 of {doc}`stitching-baseline` to `w_v` of every view
(FOVs and padding, in view order) with the view map as the grid. It reads no status, so
its output is the same for `fallback=None` and `fallback="nominal"`. On LN it must reproduce
`tile_config_sample1.csv` byte for byte ({doc}`stitching-contract`, "What the import of
the LN `dataset.xml` must reproduce"); on the golden fixture it must reproduce
`TILE_CONFIG` of `test_workflow_stitching_golden.py`.

| Parameter | Unit | Default | Rule |
| --- | --- | --- | --- |
| linear-part tolerance | – | 1e-9 | absolute, per matrix entry |
| `calibration` name | | `"calibration"` | excluded from the composition and used in step 3 |

Failure: a missing or unreadable file, a view without a view-map entry, a non-translation,
or a calibration with `c_x ≠ c_y` raise `ValueError` naming the view. Cost: a parse of one
XML file (141 KB for LN's 72 views and 145 links); not measured, negligible beside the
Fiji run (W-323: 14 min 33 s, almost all of it the HDF5 resave, and 18 s for shifts,
filter and optimization).

## Placement-quality checks

`review_placement(result, policy)` applies these after the gauge and returns the flags per
FOV and link and the review state ({doc}`stitching-contract`, option V1). They read only
the record and cost nothing measurable.

| Check | Rule | Parameter (unit, default) | Outcome |
| --- | --- | --- | --- |
| Components | The reference component holds the most FOVs; ties go to the one with the first FOV in table order | – | FOVs outside it, in `stitch` and in the BigStitcher import alike: `failed` with `fallback=None`, `metadata_only` (flagged `fallback`) with `fallback="nominal"`; review required |
| Unplaced FOVs | any status other than `image_supported` | – | review required |
| Correction bound | `|correction_y| > f × w_y` or `|correction_x| > f × w_x`, with `w` the median nominal overlap width of the FOV's pairs in that axis | `max_correction_fraction` (–, 0.5) | flag `correction_above_bound`; review required |
| Link residual | `‖(origin_b − origin_a) − (nominal_b − nominal_a) − shift_ab‖ > r` for a kept link | `max_residual_um` (µm, 1.0) | flag `residual_above_tolerance` on both FOVs; review required |
| NaN-quality links | quality NaN | – | dropped under L1; flag `nan_quality_link` |
| Optimizer-removed links | reported by the method | – | flag `optimizer_removed_link`; review required |

For scale (W-324 notes section 5): LN corrections from the grid start reach 33.9 px in x
and 44.5 px in y, below half the 149.6 px overlap (74.8 px); from the stage start they
reach 149.9 px in x, above it. The defaults are proposals for W-331's validation design;
W-331 sets the validation tolerances.

## Limitations

* Three samples, one host, Python 3.12 only; two of the references are Fiji
  Grid/Collection results with unrecorded settings (W-324 notes section 6).
* No placement truth on real data; the known-answer fixtures have integer offsets and
  identical signal in every overlap.
* The converters, the neighbour rule, the import and the checks are specified, not run:
  the import's agreement with the LN parity outputs is a requirement on the
  implementation, not a measurement.
* The stage-start failure of multiview-stitcher was observed with library defaults only.
