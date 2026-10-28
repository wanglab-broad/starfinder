# Processing-report algorithms, layout and validation design

% Status: Accepted (W-387, 2026-10-12, at c0aa8b0; the grid-edge mark of the reads along Z written on at 06466ec)

This page fixes how every value and figure of the processing report is computed and
drawn, how the page is laid out and built, how the delivery is validated, and the plan of
the real example. The entries, inputs, tables, reasons and records are in
{doc}`report-contract`; the source of every value and every gap, as they were before the
report, are recorded as history in {doc}`report-baseline`. This page redefines nothing those pages state: every table, column,
reason, part and record is used as {doc}`report-contract` gives it, and where this page
needs a rule that page leaves open ("Open points for the algorithms page") it states it
here. The algorithms, layout and validation design are implemented in
`starfinder.report` ({doc}`api/report`) and the tests that "Implemented checks"
lists.

% §2.11, W-459

## What this design is and is not

* **It describes; it does not judge.** Every definition below reads saved records and
  computes a count, a fraction, a median, a difference or an area. No definition filters,
  corrects, normalizes or rewrites anything; no value is a verdict, a pass or fail state or
  a composite score. The table "Problematic FOVs" lists where a value lies outside a stated
  limit; the limit is a number Jiahao sets, and the row is a pointer, not a grade.
* **The validation establishes engineering correctness, not biological validity.** Every
  check of "Validation design" compares the report's output with a value computed by hand
  or by an independent formula in the test, on synthetic or hand-written records. None
  compares methods, sweeps a parameter or claims accuracy, and none shows that a sample is
  well processed: a report that passes every check shows that the records were read and
  combined as this page states.
* **The contamination diagnostics stay with the separate evidence work on contamination.** The report's MECR is a descriptive count
  of the marker pairs the user supplies (numerator, denominator, ratio). The interpretation
  of MECR as contamination, the neighbour-associated incompatible-marker burden, paired
  retention and every comparison of correction methods stay with that evidence work, as the scope
  of 2026-10-07 and the inputs survey place them.
* **No new dependency.** Everything below uses the standard library, NumPy, pandas,
  SciPy, scikit-image, h5py, tifffile and matplotlib, which the package already requires
  (`src/python/pyproject.toml`, `dependencies`).

% E08, W-152, W-383

## Evidence

Cited by file in run directories kept with the run records outside the repository:

% /home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/runs/

* The source-inventory run of 2026-10-12: `results/distributions/candidate_limits.csv`
  (the candidate limits of every rule on the earlier pipeline's logs of three samples, with
  the FOVs each marks), the per-sample tables beside it, `w376_placement_status.csv`,
  `results/figures/figure_sizes.csv` (the bytes of a whole-sample figure at three widths),
  `results/tables/real_fovs.csv` and `sizing_64.csv` (a complete `FOV.run` of two LN FOVs and
  the estimate for 64), `results/replay/replay.csv` (the replay bit-identical in every round),
  `results/frame/frame_dihedral.csv` (the frame of a Python run of LN sample1) and
  `worker-notes.md` (sections 7 to 9 and 11).
* The agreed layout mockup `processing-report-layout.html` of the report scope of 2026-10-07
  (SHA-256 `b07a0efd…5462`). Its values are placeholders; its sections, blocks and
  interactions are the agreed layout.
* The inputs survey of 2026-10-08: `inputs.md` (the inputs of the real example),
  `markers.yaml`, and `survey-spatial-omics-post-processing.md` (SHA-256 `c8de6ff6…5db`),
  whose section "Practical metrics for contamination and signal preservation" defines MECR.

% W-384/20261012T065105Z-18876aac/, W-384, W-152/20261007T1939Z-s211-report-scope/processing-report-layout.html, W-383/20261008T0240Z-inputs/, W-383

## What Jiahao decides

The open choices of this page, each with its options in the section named and one
recommendation. Statements not in this table follow the settled decisions, the accepted
pages or {doc}`report-contract`. The choices of {doc}`report-contract` ("What Jiahao
decides", 1 to 21) are not repeated; choice 14 there (the provisional limits) is applied
in "Problematic FOVs".

| # | Choice | Recommendation | Section |
| --- | --- | --- | --- |
| A1 | Which FOVs are "near the median read count" | M1: the `max(1, ceil(n / 5))` FOVs nearest the median | "The example FOVs" |
| A2 | The shift of a round whose global steps include an affine step | S1: the displacement of the reference grid's centre under the composed global steps | "Shift" |
| A3 | How the area a FOV owns is computed | OA1: the FOV's own pixels whose centre it owns, times the pixel area | "The area a FOV owns" |
| A4 | The sides of the fraction rules | SD1: the low side only | "Problematic FOVs" |
| A5 | The rounds the examples show | RD1: the reference round in Preprocessing, the first moving round in Registration | "The replay of the examples" |
| A6 | The slab of the marker zoom-ins and the point size | SL1: a slab of 3 µm; points of 1.5 µm with a floor of 2 px | "Sample measurements" |
| A7 | The pixel sizes of the figures | PX1: the table "Figure sizes" | "Figure sizes" |
| A8 | The colour range of a FOV map | CR1: the minimum and maximum of the values drawn | "Style rules and how they are applied" |
| A9 | The size budget of the page | SB1: `6 MiB + 16 KiB × FOVs + 3 MiB × marker figures` | "Size budget" |
| A10 | How `sample.h5ad` is read | H1: with h5py, which the package requires | "Inputs read as arrays" |
| A11 | The subsystem marker of the report's tests | TM1: register `report` in the first issue that adds the module | "Validation design" |
| A13 | The `outcome` of a round that did not finish (last entry `applying`, `estimating`, or a `succeeded` estimation) | OU1: the contract's outcome of the part that did not finish | "Outcome" |

The bounds of the real example are not a choice of this page: Jiahao decided them on
2026-10-12 (three FOVs at a time, 32 GiB per FOV process; "The real example").

The numbers of A6, A7 and A9 are proposals with their source; they stay provisional until
Jiahao sets them.

## Conventions used by every definition

* **A value of a FOV** is a cell of a report table, null when unavailable or not applicable
  ({doc}`report-contract`, "Where a reason is shown and recorded"). Every definition below
  states its result for an empty or undefined input; a null input gives a null result with
  the reason of {doc}`report-contract` (the first reason, in the order of the closed list,
  of the inputs it is computed from), unless the definition says otherwise.
* **Median.** `median(x)` of the non-null values in float64: the middle value for an odd
  count, the mean of the two middle values for an even count (NumPy's `median`). The median
  of no value is null with reason `undefined`.
* **Robust spread.** `MAD(x) = median(abs(x_i − median(x)))` over the same values, and the
  spread `s(x) = 1.4826 × MAD(x)`. The spread of one value is 0.
* **Which FOVs enter a sample statistic.** Only the FOVs whose value is not null. A median
  over FOVs is null only when no FOV has the value (`undefined` when they all have a null
  denominator, otherwise the first reason of their cells).
* **Fractions** are float64 in [0, 1]; a fraction whose denominator is 0 is null with
  reason `undefined`. Counts are int64.
* **Order.** FOVs in list order (`fov_ids`); rounds in the order of the run records'
  `registration` keys (the union over the run records, first-seen order, as `registration.csv`
  orders them); axes `z`, `y`, `x`.
* **Units.** Voxels of the FOV's reference grid for shifts; µm of the sample frame for
  footprints, corrections and areas; reads and molecules count decoded reads.
* **Equality** of a float with a limit uses the strict comparison stated; no tolerance is
  added.

## Values of the tables

Each column of each table of {doc}`report-contract` ("The tables") is defined here as a
formula over named fields of the records, with its unit and its result for an empty or
undefined input. A column that the contract defines as a copy of one field (`status`,
`started_at`, `placement_status`, …) is listed with that field; it needs no formula. The
worker notes of the algorithms specification map every column to its definition here and to the implementation
issue that computes it.

% W-386

### `fovs.csv`

Rows and key, as {doc}`report-contract` states them: one row per FOV of the list, in list order; key `fov_id`.

Per FOV `f` of the list, with `R_f` its run record (`fovs/<f>/run.json`, absent for
a FOV without one), `P` the placement's `stitching/<sample>/fovs.csv`, and `E` the export's
`tables/fovs.csv`:

| Column | Definition | Unit | Empty or undefined input |
| --- | --- | --- | --- |
| `fov_id`, `list_index` | `fov_ids[i]` and `i`. | — | never null |
| `example` | The roles of "The example FOVs" that fall on `f`, `;`-joined in role order. | — | n/a when none |
| `run_record` | `R_f` exists at its place. | — | never null |
| `status` | `R_f.status`. | — | `no_run_record` |
| `failed_step`, `failed_round` | `R_f.error.step`, `R_f.error.round`. | — | n/a when `R_f.error` is null (the run did not fail), or `error.round` is null |
| `started_at`, `ended_at` | `R_f.started_at`, `R_f.ended_at` (ISO 8601, UTC, as written). | — | `no_run_record`; `ended_at` n/a while `status` is `running` |
| `run_seconds` | `ended_at − started_at` in seconds (both parsed as UTC; float64). | s | from its inputs; n/a while running |
| `placement_status` | `P.status`, else `E.placement_status`. | — | `no_placement` when neither exists |
| `correction_z/y/x` | `P.correction_z/y/x` (origin minus the stage position). | µm | `no_placement`; `fov_not_placed` for `missing` or `failed` |
| `footprint_y0/y1/x0/x1`, `centre_y/x` | `E.footprint_*`, `E.centre_*`; without an export, `assembly/_ownership.py`: `footprint_yx(origin_yx, shape_yx, spacing_yx)` of `P`'s origin and the grid `placement.json` `inputs.<f>.grid` records: per axis `[o − s/2, o + (n − ½)·s)` and the centre `o + (n − 1)/2 · s`. | µm | `no_placement` when neither exists; `fov_not_placed` |
| `spots` | `R_f.counts.spots`. | reads | the contract's reasons (`no_run_record`, `run_not_finished`, `step_failed`, `not_configured`) |
| `correct_form` | `R_f.counts.filtering.endpoint_valid` (R3). | reads | as `spots`, and `not_recorded` for a run record without R3 |
| `matched` | `R_f.counts.call_status.assigned`; 0 when `call_status` exists without the key. | reads | as `spots` |
| `merged_by_deduplication` | `R_f.counts.deduplication.merged_reads`. | reads | as `spots`; `not_configured` when `config.pipeline.deduplication` is null |
| `passed_filter` | `R_f.counts.filtering.accepted`. | reads | as `spots`; `not_configured` without a filter |
| `correct_form_fraction`, `matched_fraction`, `merged_fraction`, `passed_filter_fraction` | `count / spots` for the count of the same name (`merged_by_deduplication` for `merged_fraction`). | fraction of `spots` | null with the reason of its count or of `spots`; `undefined` when `spots = 0` |
| `n_cells`, `n_cells_retained`, `n_molecules`, `n_molecules_retained` | `E.n_cells`, `E.n_cells_retained`, `E.n_molecules`, `E.n_molecules_retained`, with the route's meaning ({doc}`report-contract`, "The two routes"). | cells; molecules | `no_export`, `part_not_written`; `no_segmentation` for the cells of an export without cells |
| `reads_in_cells` | "Reads assigned to cells", below, for the FOV. | molecules | as `n_molecules`; `no_segmentation` without cells |
| `reads_in_cells_fraction` | `reads_in_cells / n_molecules_retained`. | fraction | from its inputs; `undefined` when `n_molecules_retained = 0` |
| `owned_area_um2` | "The area a FOV owns", below. | µm² | `no_placement`, `fov_not_placed`, `no_export`; n/a for a FOV that owns no point (`owns_points` false: a `placeholder`) |
| `cell_density_per_mm2` | `n_cells_retained / owned_area_um2 × 10⁶`. | cells per mm² | from its inputs; `undefined` when the area is 0; n/a as the area |

The read levels are not nested ({doc}`report-baseline`, discrepancy 1): `matched` can
exceed `correct_form`, and each fraction has `spots` as its denominator, not the level
before it.

### `registration.csv`

Rows and key, as {doc}`report-contract` states them: one row per FOV of the list and moving round, in list order, then the rounds in the order of the run records' `registration` keys (the union over the run records, in first-seen order); key (`fov_id`, `round`).

The moving rounds are the union of the keys of `R_f.registration` over the run records, in
first-seen order (the reference round is not a key). For FOV `f` of the list and moving round
`r`, with `A = R_f.registration[r]`, the round's list of entries in record order, when the run
record exists and holds the key `r`. A FOV without a run record, or whose record does not hold
`r`, still has its row: every value of the row is null with the contract's reason
(`no_run_record`; `run_not_finished` or `step_failed` when the run did not get to the round;
`not_configured` without a registration), each named by its own row of `unavailable.csv`.

| Column | Definition | Unit | Empty or undefined input |
| --- | --- | --- | --- |
| `outcome` | "Outcome", below: one of `succeeded`, `rejected`, `failed`, `application_failed`, from the `record` and `outcome` of the last entry of `A`. | — | the contract's reasons; `run_not_finished` as "Outcome" states |
| `fallback` | Some entry `e` of `A` with `e.record == "estimation"`, `e.outcome == "succeeded"` and `e.fallback` true. | — | as `outcome` |
| `methods` | The `actual_method` of the succeeded estimation entries in `step` order, `;`-joined. | — | n/a when none succeeded |
| `transform_source` | `run.json` when `R_f.transforms[r]` exists (R1), else `transforms.json` when `registered/transforms.json` holds `r` with `format_version` 2 or 3. | — | as the contract states |
| `shift_z/y/x` | "Shift", below. | voxels | as the contract states; n/a when the chain has no global step |
| `residual_z/y/x` | "Shift residual", below. | voxels | the reason of the shift; `undefined` when no FOV's round succeeded |
| `alignment_score` | "Alignment score", below. | dimensionless | as the contract states; `undefined` when the value is null |

#### Outcome

`FOV._register_round` (`dataset/fov.py`) writes six values into the `outcome` of a
registration entry, and `run.json` is written only when the run starts, with `registration`
empty, and when it ends (`dataset/_run_record.py`: `_RunRecord.finish`). A saved round whose
last entry is `applying`, `estimating` or a `succeeded` estimation did not finish: its run
ended `failed` or `interrupted` inside that round. The report's `outcome` is read from the
last entry `e` of `A`:

| `e.record` | `e.outcome` | When the source leaves it last | Report's `outcome` |
| --- | --- | --- | --- |
| `application` | `succeeded` | The round's transforms were applied and the application QC recorded | `succeeded` |
| `application` | `application_failed` | Applying the transforms raised | `application_failed` |
| `application` | `applying` | The run was interrupted while it applied the transforms, or the registered signal or the application QC raised after them (`status` `failed` or `interrupted`) | `application_failed` |
| `estimation` | `rejected` | A QC criterion refused the last attempt and no recovery followed | `rejected` |
| `estimation` | `failed` | The estimation of the last attempt raised and no recovery followed | `failed` |
| `estimation` | `estimating` | The run was interrupted during an estimation (`status` `interrupted`) | `failed` |
| `estimation` | `succeeded` | The run ended after an estimation succeeded and before the application entry was added: the resampling for the next step, or the chain or warp of the application, raised or was interrupted (`status` `failed` or `interrupted`) | `failed` |

The source writes no other value. A record with `status` `running` holds an empty
`registration` at this revision (the record of the run's start); should a later writer save
entries during a run, a round whose last entry is not a finished one (`applying`,
`estimating`, or a `succeeded` estimation) is null with `run_not_finished` while `status` is
`running`. The run's own end (`failed` or `interrupted`, its step and round) is in
`fovs.csv` `status`, `failed_step` and `failed_round`.

The table is choice A13. **OU1 (recommended)**, the table: an unfinished last entry is read as
the contract's outcome of the part that did not finish, so the maps, the heatmap, the medians
and the rules treat the round as one that did not succeed. OU2: the row is null with
`step_failed`, which {doc}`report-contract` reserves in the null rule of `outcome` for a run
that did not get to the round. A round whose `outcome` is not
`succeeded` has no transform record (`dataset/fov.py`: `FOV._record_transforms` runs after the
application), so its transform, shift and score are null as the contract states for a
registration that failed.

#### Shift

The round's results are rebuilt as `registration.csv` `transform_source` names them
(`_checkpoint.py`: `_restore_result`). The **global steps** are the results whose
`transform.kind` is `translation` or `affine` (`registration/_chain.py`: `transform_kind`);
a local step (`dense`, `bspline`) is always last in a chain ({doc}`registration-algorithms`,
"Transforms"), so the global steps are a prefix `T₁ … T_k` of the chain. Their
composed pull map is `Φ_g(p) = T₁(T₂(…T_k(p)…))`, the composition order of
{doc}`registration-algorithms`. With `c = ((n_z − 1)/2, (n_y − 1)/2, (n_x − 1)/2)` the
centre of the reference grid of shape `reference_shape_zyx`:

```text
shift = Φ_g(c) − c          (z, y, x), voxels of the reference grid, float64
```

* A chain of translations: `shift = Σ_i displacement_zyx(T_i)`, exactly (`TransformChain.translation`),
  whatever `c` is.
* A global step that is affine (`T(p) = A p + b`, `matrix_zyx = [[A, b], [0, 1]]`): its
  contribution depends on the point, and the shift is the displacement of the grid centre
  (choice A2). Options: **S1 (recommended)**, the centre's displacement, which equals the
  translation for a pure translation and is the value a reader compares with the
  neighbouring FOVs; S2, the translation part `b` alone, which for a rotation about the
  grid origin is far from what the image shows; S3, null for a chain with an affine step,
  which leaves the maps empty for an affine recipe.
* A chain without a global step (a recipe of one local step): the shift is n/a.
* The sign is the record's: `displacement_zyx` is the pull displacement, so a feature at
  reference index `p` lies at `p + shift` in the moving round. The earlier pipeline's logs printed
  another sign and axis order ({doc}`report-baseline`, discrepancy 6); the report reads only
  the Python records.

#### Shift residual

For round `r` and axis `a`, with `S_r` the FOVs whose round `r` has `outcome` `succeeded`
and a non-null shift:

```text
median_r,a  = median over f in S_r of shift_a(f, r)
residual_a(f, r) = shift_a(f, r) − median_r,a          for every f with a non-null shift
```

* `S_r` empty: every residual of the round is null, `undefined`.
* One succeeded FOV: its residual is exactly 0.
* Two succeeded FOVs: each residual is half their difference, with opposite signs.
* Fewer than three succeeded FOVs: the residual is defined as above, the map legend says
  "median of n FOVs", and the relative rule with `k ≥ 1` cannot mark any FOV (the deviation
  of each of one or two values from their median is at most their MAD; "Problematic FOVs").
* A FOV whose round did not succeed (`rejected` after its recovery, `failed`,
  `application_failed`) has no shift and therefore no residual (the contract's reasons);
  it never enters `S_r`.

#### Alignment score

`alignment_score(f, r) = e.qc.values.ncc_after` of the entry `e` of `A` with
`e.record == "application"` (the masked NCC of the reference and the registered moving
round under the whole chain; `evaluation/registration.py`: `registration_qc`). It is null
with `step_failed` when the round's `outcome` ("Outcome") is not `succeeded`: the round has
no application entry because the estimation failed, or its application entry ended
`application_failed` or was left `applying`. Otherwise it is null with `undefined` when
`e.qc` or `ncc_after` is null, the QC record's `reasons.ncc_after` in the detail. The score
of an estimation entry is not read: it scores one step, not the chain
(the estimation entries and the application entry hold equal values when the chain has one
step).

### `steps.csv`

Rows and key, as {doc}`report-contract` states them: one row per step; key (`scope`, `fov_id`, `index`). Order: the FOVs in list order, each with its run record's `steps` in record order, then its segmentation and assignment (route per FOV); then the sample steps (`stitch`, then the export's parts in the order of `export.json` `parts`).

| Column | Definition | Unit | Empty or undefined input |
| --- | --- | --- | --- |
| `scope`, `fov_id`, `index`, `name`, `round`, `status` | As {doc}`report-contract` lists them: per FOV, `R_f.steps[i]` in record order (`index = i`), then, on the route per FOV, one row `segment` (method of the cell run: `segmentation.json` `methods`, or `imported` for an import, baseline G13) and one row `assign_molecules`; then the sample rows `stitch` (`placement.json` `method.method`) and one per export part in the order of `export.json` `parts`. | — | as the contract states |
| `method` | `preprocess:<step>` → `<step>`; `register` → `registration.csv` `methods` of the round; `find_spots` → the distinct `method` values of `R_f.steps[i].methods` (the step's provenance entries, one per method that ran, written by `_RunRecord.add_methods`), `;`-joined in record order; `decode_reads` → `R_f.config.pipeline.decoding.method` with `R_f.config.readout_mode`; `filter_reads` → `filter`; `segment`, `assign_molecules`, `stitch` and the export parts as above; every other step n/a. | — | n/a for a step without a method |
| `seconds` | `R_f.steps[i].seconds`; an export part's `wall_s` (`fusion`, `labels`, `spatialdata`, `segmentation` sections). | s | `not_recorded` for stitching, assembly and the per-FOV segmentation and assignment; n/a for a part `off` or `not_run` |

### `readout_totals.csv`

Rows and key, as {doc}`report-contract` states them: one row per level, in the order `spots`, `correct_form`, `matched`, `merged_by_deduplication`, `passed_filter`, `reads_in_cells`; key `level`.

For level `L` in the order `spots`, `correct_form`, `matched`, `merged_by_deduplication`,
`passed_filter`, `reads_in_cells`, with `F_L` the FOVs whose `fovs.csv` cell of `L` is not
null:

| Column | Definition | Unit | Empty or undefined input |
| --- | --- | --- | --- |
| `count` | `Σ_{f in F_L} L(f)`. For `reads_in_cells`, the retained molecules of the sample whose cell is counted, which the report checks against `Σ X` of `sample.h5ad` ("Reads assigned to cells"). | reads | null when `F_L` is empty, with the first reason of the FOVs' cells of `L` |
| `fraction_of_spots` | `count / Σ_{f in F_L ∩ F_spots} spots(f)`: the spots of the same FOVs. For `reads_in_cells` the denominator is the spots of the FOVs whose `spots` and `reads_in_cells` are both not null. | fraction | from its inputs; `undefined` when that sum is 0 |
| `fovs` | `abs(F_L)`. | FOVs | never null (0 when none) |
| `overlaps_counted_per_fov` | `L != "reads_in_cells"`. | — | never null |

The levels above `reads_in_cells` are sums over FOVs and count a read in an overlap once per
FOV that saw it (settled decision 5); `reads_in_cells` is after the overlaps are removed. The figure shows
this ("Readout").

% D5

### Reads assigned to cells

With `M` the export's `tables/molecules.csv` and `C` its `tables/cells.csv`, the counted
cells are `K = {(cell_namespace, cell_id) of rows of C with assembly_status == "retained"
and status == "kept"}` ([assembly specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/assembly.md), "Statuses and reasons"). For FOV `f`:

```text
reads_in_cells(f) = #{ m in M : m.fov_id = f, m.assembly_status = "retained",
                       m.assignment_status = "assigned", (m.cell_namespace, m.cell_id) in K }
reads_in_cells    = Σ_f reads_in_cells(f)
h5ad_whole_count_sum = Σ_ij X_ij of sample.h5ad     (rounded to int64; X holds integer counts)
reads_in_cells_equal = (reads_in_cells == h5ad_whole_count_sum)
```

On the route after fusion a retained molecule's `fov_id` is the owner of its position, so
the same formula counts it once. By identity 4 of [assembly specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/assembly.md) ("Count
accounting") the two sums are equal; the report records both in `report.json` `summary` and
shows the comparison. A difference is shown, with both numbers, and the report is still
written: the report describes, and the difference points at the records. Without cells
(`no_segmentation`) all four values are null.

### `population.csv`

Rows and key, as {doc}`report-contract` states them: one row per population, in this order: `input`, `retained`, `excluded_not_owner`, `excluded_fov_not_placed`, `excluded_duplicate_observation`, `excluded_cell_not_owner`, `excluded_cell_duplicate_observation`, `excluded_outside_grid`, `counted`; key `population`.

`cells(p)` and `molecules(p)` copy `export.json` `assembly.counts` for each population `p`
of the contract's order: `input` is `counts.cells.input`, `retained` is
`counts.cells.retained`, `excluded_<reason>` is `counts.cells.excluded.<reason>` (likewise
for molecules); `counted` is `n_obs` of `sample.h5ad` for cells and `reads_in_cells` for
molecules. A reason that the record's `excluded` object does not hold is n/a, not 0 (the
route after fusion and the duplicate rule, {doc}`report-contract`, "The two routes").

### `problematic_fovs.csv`

Rows and key, as {doc}`report-contract` states them: one row per FOV, rule, round and axis that the rule marks, in list order, then the rule order of `PROBLEM_RULES`, then rounds and axes; key (`fov_id`, `rule`, `round`, `axis`).

Its rows are made by the rules of "Problematic FOVs". For a rule evaluated on values:
`measured` is the FOV's value, `sample_median` the median of the rule's values over the
FOVs that enter it (n/a for an absolute limit of a rule other than the shift residual),
`limit` the limit applied, `unit` the rule's unit. For the shift-residual rule, under the
relative form and under an absolute limit alike, `measured` is the shift and
`sample_median` the round's median shift, so that `measured − sample_median` is the
residual the limit is compared with, as {doc}`report-contract` has it. `reason` is the text
of the rule ("The reason texts").

### `cells.csv`

Rows and key, as {doc}`report-contract` states them: one row per counted cell, in `sample.h5ad` `obs` order; key `obs_name`.

One row per counted cell, in `sample.h5ad` `obs` order:

| Column | Definition | Unit | Empty or undefined input |
| --- | --- | --- | --- |
| `obs_name`, `fov_id` | `obs_names[i]`, `obs.fov_id[i]`. | — | never null |
| `reads` | `Σ_j X_ij` (int64). | molecules | never null; 0 for a cell without counts |
| `genes` | `#{j : X_ij > 0}`. | genes | never null; 0 for a cell without counts |
| `size`, `size_unit` | "Cell size", below. | `size_unit` | never null |

#### Cell size

```text
size(i) = obs.size_physical[i]            when it is finite
        = obs.size_voxels[i]              otherwise
size_unit(i) = "µm³" when finite and the cell is a volume, "µm²" when finite and a plane,
             = "voxels" otherwise
```

The volume or plane is read from the size unit of the record that wrote the cell: on the
route per FOV the FOV's `assignment.json` `size_unit` (`µm^3` or `µm^2`, as `assignment/_persist.py` writes it),
on the route after fusion the export's `segmentation` section (a plane). `size_physical` is
NaN exactly when that record's `calibration` is `unknown` ([assignment specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/assignment.md#cell-table)), so
a cell has `voxels` exactly when its calibration was unknown. A sample whose cells have more
than one unit gets one histogram per unit, in the order µm³, µm², voxels.

### `mecr.csv`

Rows and key, as {doc}`report-contract` states them: one row per pair of `marker_pairs`, in configuration order; key (`gene_a`, `gene_b`).

For a pair `(A, B)` of `marker_pairs` and the threshold `t = positive_threshold`, over the
counted cells (the rows of `X`), with `a_i = X_{i,A}` and `b_i = X_{i,B}` (the whole-cell
counts; a gene listed twice in the codebook is one column of `X`):

```text
cells_a = #{i : a_i ≥ t}      cells_b = #{i : b_i ≥ t}
both    = #{i : a_i ≥ t and b_i ≥ t}
either  = #{i : a_i ≥ t or  b_i ≥ t}  = cells_a + cells_b − both
mecr    = both / either
```

`either = 0` gives `mecr` null with reason `undefined`, for that pair only; the four counts
are written (0). No cell: every count 0 and `mecr` undefined. This is the definition of the
inputs survey (`n(A+ and B+) / n(A+ or B+)`, an observed count of at least one positive by
default), with the threshold configurable (settled decision 5). It is a description of the supplied pairs;
the report states no expected value and draws no conclusion from it.

% W-383, D5

### `reads_along_z.csv`

Rows and key, as {doc}`report-contract` states them: one row per FOV of the export and plane, in list order, then `z` ascending; key (`fov_id`, `z`).

The FOVs of the export are the entries of `export.json` `assembly.fovs`, in list order;
`n_z = assembly.fovs[f].grid.shape_zyx[0]` is the number of planes of FOV `f`'s grid. A
molecule's plane is `floor(z + 0.5)` of its `tables/molecules.csv` `z` (float64; the assignment
sampling rule of the [assignment algorithms specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/assignment-algorithms.md#sampling), "Sampling"). Every molecule of the table,
retained or not, is counted on its own plane, whether or not that plane lies in the FOV's
grid, so that a projected sample whose spots were found in 3D shows where its reads lie:

% §2.9

```text
P(f)        = {plane(m) : m.fov_id = f}
k_lo(f)     = min(0, min P(f))            (0 when f has no molecule)
k_hi(f)     = max(n_z − 1, max P(f))      (n_z − 1 when f has no molecule)
reads(f, k) = #{m : m.fov_id = f, plane(m) = k}      one row for each k = k_lo(f) … k_hi(f)
```

The rows of a FOV run without a gap, with 0 where no molecule lies; a FOV whose molecules all
lie in `[0, n_z − 1]`, or that has none, has the `n_z` rows 0 to `n_z − 1`. `Σ_k reads(f, k)`
equals the FOV's molecules of the table. A row outside `[0, n_z − 1]` is not marked in the
table; the figure of the reads along Z marks the edges of each FOV's grid (decided at the gate
of 2026-10-12: a mark in the figure, no new column).

% W-387

A molecule whose plane lies outside its FOV's grid is not for that reason excluded
`outside_grid`: in `assembly/_assemble.py`: `assemble`, a FOV whose grid has one plane gives
every molecule the index z 0 before any test (`positions[:, 0] = 0.0`; the column `z` keeps
the recorded value), so on a one-plane grid no route excludes a molecule by its `z`. On a grid
of several planes, a `molecules_only` export excludes `outside_grid` the molecule whose
`floor(z + 0.5)` lies outside `[0, n_z − 1]` (`assemble` tests every axis against the FOV's
grid); on the route per FOV a molecule is `outside_grid` when its assignment status is
(`assignment/_assign.py`: `assign_molecules`, which tests `z` against a grid of several
planes for volume and for plane labels); the route after fusion starts from the
`molecules_only` assembly, with its test, and `assembly/_fused.py`:
`assemble_after_fusion` adds the retained molecules that `assign_molecules` places outside
the segmentation grid, a grid of one plane, so by Y or X only. The tissue-2D export of the saved-files run,
a `molecules_only` export of 56 FOVs of grid 1 × 3072 × 3072 whose molecule tables span 30
planes ([stitching and assembly validation design](https://github.com/wanglab-broad/starfinder/blob/dev/design/validation/stitching-assembly.md), "Results of the real examples"), has 5,353,714
molecules in its `export.json` `assembly.counts`, 4,703,832 retained, 649,882 `not_owner` and
none `outside_grid`.

% §2.9, W-391

### `intensity.csv`

Rows and key, as {doc}`report-contract` states them: the intensity distribution of the main example after each step of the recipe, in the order `FOV.run` runs them ("The replay"); one row per round, step, channel and bin, in that order, for each step whose output the replay rebuilds; key (`round`, `step_index`, `channel`, `bin`).

For each round the replay rebuilds ("The replay of the examples"), each recorded entry of
the round in the order `FOV.run` runs them (`step_index` 0 for the loaded and rotated input)
and each channel `c` of the round's output after that entry:

* **Bins.** Let `lo` and `hi` be the minimum and maximum over all channels of that output.
  An integer dtype has bins of integer width `w = max(1, ceil((hi − lo + 1) / 256))`
  starting at `lo`: bin `k` is `[lo + k·w − 0.5, lo + (k + 1)·w − 0.5)` (so a uint8 image
  with `hi − lo < 256` has one bin per grey level). A float dtype has 256 bins of equal width
  over `[lo, hi]`, the last closed; `lo = hi` gives one bin `[lo, hi]`. A boolean output has
  the two bins 0 and 1.
* **Count.** The voxels of channel `c` in each bin; the counts of a channel sum to its
  voxels. Every bin is written, also with a count of 0, so the channels of one entry share
  their edges.

### `unavailable.csv`

Rows and key, as {doc}`report-contract` states them: one row per unavailable value ("Where a reason is shown and recorded"): each unavailable cell of a report table, each unavailable key of `report.json` `summary`, and each part, or part of a part, with no cell for what it lacks. Order: `REPORT_PARTS` order, then the tables in the order of this section with `summary` last and rows without a table first, then the column order of the table, then its row order; key (`part`, `table`, `column`, `row_key`).

Defined by {doc}`report-contract` ("Where a reason is shown and recorded"); this page adds
no rule. The `detail` of a row is the file, the key or the step named there, and for
`undefined` the denominator that is 0 (`spots`, `n_molecules_retained`, `either`, the area,
or "no succeeded FOV").

### `report.json` `summary`

| Key | Definition | Unit | Empty or undefined input |
| --- | --- | --- | --- |
| `fovs_expected` | `abs(fov_ids)`. | FOVs | never null |
| `fovs_with_run_record` | `#{f : run_record(f)}`. | FOVs | never null |
| `fovs_succeeded` | `#{f : status(f) == "succeeded"}`. | FOVs | null with `no_run_record` when no FOV has a run record |
| `spots_median_per_fov` | `median(spots)` over the FOVs with a value. | reads | as "Which FOVs enter a sample statistic" |
| `matched_fraction` | `Σ matched / Σ spots` over the FOVs whose both values are not null. | fraction | `undefined` when the spots sum to 0 |
| `counted_cells` | `n_obs` of `sample.h5ad`. | cells | `no_export`, `part_not_written`, `no_segmentation` |
| `reads_per_cell_median`, `genes_per_cell_median` | `median(cells.csv reads)`, `median(cells.csv genes)`. | molecules; genes | as `counted_cells`; `undefined` with no counted cell |
| `cells_median_per_fov` | `median(n_cells_retained)` over the owning FOVs with a value. | cells | as `n_cells_retained` |
| `reads_in_cells`, `h5ad_whole_count_sum`, `reads_in_cells_equal` | "Reads assigned to cells". | molecules | as `counted_cells` |

### `samples.csv`

Rows and key, as {doc}`report-contract` states them: one row per sample of `samples`, in that order; key `sample`.

Each column copies the key of the present sample's `report.json` the contract names
(`generated_at`, `route`, `summary.fovs_expected`, …); `report_sha256` is the SHA-256 of
that file's bytes. A sample whose `report.json` is missing has `report` `missing` and every
other column n/a.

### Inputs read as arrays

* **`sample.h5ad`** is read with h5py, which the package requires: `X` as written by
  `assignment.write_h5ad` (dense float64, [stitching and assembly validation design](https://github.com/wanglab-broad/starfinder/blob/dev/design/validation/stitching-assembly.md), AS11), or a
  CSR matrix by its `encoding-type` attribute; `obs_names` and the `obs` columns by name; no
  AnnData object is built. Options (choice A10): **H1 (recommended)**, h5py, so that the
  report reads a sample in the plain environment; H2, `anndata.read_h5ad`, which needs the
  `anndata` extra for every part that reads the file; H3, the export's `tables/cells.csv`
  and `counts.csv` instead, which the contract does not name as the input of these values.
* **Tables** (`molecules.csv`, `cells.csv`, `fovs.csv`) with the reader of
  {doc}`checkpoints` ("Table formats") and only the columns a definition names.
* **Images** with tifffile: a fused stack plane by plane (its pages), a label image or a
  reference image whole, one at a time.

### Other values of the page

* **Configuration check** (Summary). Over the run records, the canonical JSON of `R_f.config`
  (keys sorted, no whitespace). One distinct value: "identical for all FOVs"; otherwise
  "differs: n groups", with the top-level keys of `config` that differ between the groups
  and, in the tooltip, the FOVs of each group. No run record: unavailable (`no_run_record`).
* **Acquisition** (Summary). `R` rounds × `C` channels from the union of `R_f.channels`
  (`C` the largest number of channels of a round, "C1–C2 channels" when they differ), and
  "3D" when the Z size is greater than 1, else "2D (Z = 1)"; the Z size is the fused grid's
  (`fusion.fovs[*].grid.shape_zyx[0]`), else the reference image's.
* **Software** (Summary). `R_f.code.version` and `R_f.code.git_commit` (with `R_f.code.git_dirty`); several
  distinct values are listed with their FOV counts. Without a run record, the export's
  `assembly.software`.
* **Run time** (Summary). From the earliest `started_at` to the latest `ended_at` of the run
  records.
* **Time per step** (Workflow overview). Per FOV and step name, `Σ seconds` over the rows of
  `steps.csv` of that FOV with that name (the rounds of a per-round step add up); one box per
  name across FOVs, names in first-seen order. Sample steps are single values.
* **FOV status** (Workflow overview). `status` of `fovs.csv`, and "no run record" for a FOV
  without one; the tooltip adds `run_seconds` and the failed step and round.

### The area a FOV owns

The area is computed by the report from the placement, the FOVs' grids and the export's
ownership rule, on both routes ({doc}`report-baseline`, G6; {doc}`report-contract`, choice
12). With `O = assembly.ownership_map(placement, grids, config=AssemblyConfig(**export.json
assembly.config))` (the same call `assemble` makes; `grids` the reference grids of the
export's FOVs) and FOV `f` of grid `(n_z, n_y, n_x)`, spacing `(s_z, s_y, s_x)` and origin
`(o_z, o_y, o_x)`:

```text
pixel centres  q_ij = (o_z, o_y + i·s_y, o_x + j·s_x),  0 ≤ i < n_y, 0 ≤ j < n_x
owned_area_um2(f) = #{(i, j) : O.owner(q_ij) == f} × s_y × s_x
```

* The region a FOV owns lies inside its footprint ([assembly specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/assembly.md), "Ownership
  regions"), and the footprint is the union of the FOV's pixel supports, so the count covers
  the whole region; a pixel is counted whole when its centre is owned. The count differs
  from the exact area of the region by at most the area of the pixels the region's
  boundary crosses; the validation (DF5) checks the count of pixels itself, which the test
  counts by hand.
* Z is not used: ownership is decided in Y and X ([assembly specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/assembly.md), "The Z axis").
* A FOV that owns no point (`owns_points` false: a `placeholder`) is n/a; a `missing` or
  `failed` FOV is `fov_not_placed`; an owning FOV whose region is empty has area 0, and its
  density is `undefined`.
* Cost: one `owner()` call per FOV on `n_y × n_x` points, 2.2 million points for an LN FOV,
  about 54 MB of float64 coordinates, done one FOV at a time.

Options (choice A3): **OA1 (recommended)**, the pixel count above, which uses the package's
own ownership test and needs no new geometry code; OA2, the exact polygon of the region
(the footprint cut by the bisectors of the neighbouring centres, or the box of
`midpoint_box`), which is exact but is a second implementation of the ownership rule that
can drift from `ownership_map`; OA3, the footprint's area, which counts an overlap twice.

## Problematic FOVs

**The numbers below are provisional.** They come from the source-inventory run's candidate limits on the logs
of the earlier pipeline, whose quantities are defined differently from the Python records
({doc}`report-baseline`, discrepancy 6), and they stay provisional until Jiahao sets them at
the report's gate ({doc}`report-contract`, choice 14).

% W-384, §2.11

### The one form of a rule on values

A rule on values has a `RuleLimit(k, floor, absolute)` and a side. Over the FOVs whose value
`v_f` is not null (and, for the shift residual, per round and axis, the FOVs whose round
succeeded):

```text
m = median(v)         s = 1.4826 × MAD(v)
d_f = m − v_f (low side),  v_f − m (high side)

absolute set:         marks f when  v_f < absolute (low side),  v_f > absolute (high side)
otherwise:            marks f when  d_f > max(k' × s, floor')
                      with k' = k or 0 when k is None, floor' = floor or 0 when floor is None

shift residual:       relative form: v_f = shift_a(f, r), m = its round's median shift,
                      d_f = abs(residual_a(f, r)) (both sides)
                      absolute set: v_f = residual_a(f, r), marks f when abs(v_f) > absolute
```

`k`, `floor` and `absolute` all `None`: the rule is not evaluated, `report.json` `rules`
says so, and one row of `unavailable.csv` (`part` `rule:<rule>`, `table` null) names it
with `not_configured`. The comparison is strict, so a value equal to the limit is not
marked. The limit written to `problematic_fovs.csv` is `max(k' × s, floor')` or `absolute`,
in the rule's unit. With an absolute limit `sample_median` is n/a for every rule but the
shift residual ({doc}`report-contract`, `problematic_fovs.csv`); under both forms the
shift-residual row shows the shift as `measured` and the round's median shift as
`sample_median` (the mockup's "dx +38.0 px" beside "+2.1 px"), so that the residual
`residual_a(f, r)` the formula above compares with the limit is `measured − sample_median`.
Which FOVs a rule marks does not depend on these columns.
`report.json` `rules.<rule>` records `k`, `floor`, `absolute`, the unit, whether it was
evaluated, the FOVs that entered (per round and axis for the residual), and the number of
FOVs it marks.

What the form gives in the edge cases, as `results/check_rule_form.py` of the algorithms specification's run
directory evaluates it (output `check_rule_form.txt`; low and high side; `k = 3`, floor 0.05;
`k = 0`, floor 0; absolute 0.6):

% W-386

* **One FOV.** `m = v` and `s = 0`, so `d = 0`: no relative form marks it, whatever `k` and
  the floor. An absolute limit marks it when its value is beyond the limit on the rule's side
  (0.5 under a low-side absolute 0.6 is marked; under a high-side one it is not).
* **Two FOVs.** Each deviates from the median by the MAD, on opposite sides, and
  `k' × 1.4826 × MAD ≥ MAD` for `k' ≥ 1 / 1.4826 ≈ 0.675`, so the relative form with `k ≥ 1` marks neither
  (0.2 and 0.8: limit 1.33 with `k = 3`). With `k = 0` and floor 0 it marks the one FOV on the
  rule's side (0.2 low, 0.8 high). An absolute limit judges each value on its own.
* **Equal values.** `s = 0` and every `d = 0`: the relative form marks no FOV, whatever `k`
  and the floor (strict comparison). An absolute limit marks every FOV when the common value
  is beyond it on the rule's side and none otherwise: four values of 0.5 under a low-side
  absolute 0.6 are all marked, four of 0.7 none; under a high-side absolute 0.6 the reverse.
* **A spread of zero and one outlier.** `s = 0`, so the limit of the relative form is the
  floor: the outlier is marked when its deviation on the rule's side exceeds the floor (0.5
  among three 0.8 on the low side, 0.95 on the high side), and the others are not. An
  absolute limit judges each value on its own.
* **Undefined values.** A FOV with a null value does not enter `m` or `s` and is not
  evaluated (0.8, 0.8, null, 0.5: median 0.8 of the three values, the 0.5 marked on the low
  side); the row of its null value explains it, and the rule needs no row of its own
  ({doc}`report-contract`). When no FOV has a value, the rule is not evaluated for the
  sample, with one `rule:<rule>` row and the first reason of the values.

### The rules of the first delivery

`PROBLEM_RULES` in order. The counts are the FOVs each rule marks on the logs of the three
samples (the source-inventory run's `results/distributions/candidate_limits.csv`, the row whose `rule` and
`setting` are named; "edge" and "interior" as the source-inventory run places them). LN sample1 has 64 FOVs,
cell-culture-3D 70, tissue-2D 56 (the Snakemake run of 2026-01-22; its 2023 logs are a
second configuration of the same sample and are given in brackets where they differ).

% W-384

| Rule | Section | Value and side | Provisional limit (default) | Unit | Source-inventory row (`rule`; `setting`) | LN sample1 | cell-culture-3D | tissue-2D |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `fov_failed` | Workflow overview | `status` in (`failed`, `interrupted`); `running` is not marked | categorical | — | "workflow: FOV failed or expected and not run (log incomplete or missing)"; "log without the read counts" (the logs hold no status; an incomplete log is the nearest) | 0 | 0 | 0 [2023: 1, tile_1] |
| `fov_not_run` | Workflow overview | No run record, while at least one FOV of the list has one | categorical | — | the same row (every FOV of the three samples has a log, the inputs survey's `inputs.md`) | 0 | 0 | 0 |
| `shift_residual` | Registration | The residual per round and axis, both sides; every moving round of `registration.csv` | `k = 5`, floor 5 | voxels | "registration: shift residual above limit on any axis (sequencing rounds)"; "\|residual\| > max(5 x 1.4826 MAD, 5 vox)" | 1 (edge: Position001, Z residuals 17–19 in every moving round) | 0 | 0 [2023: 0] |
| `alignment_score` | Registration | `alignment_score`, low side | none: not evaluated by default | dimensionless | no row: no log holds a registration quality value (the source-inventory run's notes, section 7) | not evaluated | not evaluated | not evaluated |
| `registration_rejected` | Registration | Some estimation entry of the round has `outcome` `rejected` (a configured QC criterion refused an attempt, whether or not a recovery then succeeded) | categorical | — | no row: no log holds a registration outcome (`<sample>_reads_stats.csv`, row "registration outcome or fallback") | not countable from the logs | not countable | not countable |
| `registration_fallback` | Registration | `fallback` true on a succeeded round | categorical | — | as `registration_rejected` | not countable from the logs | not countable | not countable |
| `correct_form_fraction` | Readout | `correct_form_fraction`, low side | `k = 3`, floor 0.05 | fraction | "readout: correct-form fraction deviating (low)"; "median - fraction > max(3 x 1.4826 MAD, 0.05)" | 1 (edge: Position002) | 0 | 2 (edge tile_1, interior tile_29) [2023: 2, interior tile_29, tile_30] |
| `matched_fraction` | Readout | `matched_fraction`, low side | `k = 3`, floor 0.05 | fraction | "readout: matched fraction deviating (low)"; "median - fraction > max(3 x 1.4826 MAD, 0.05)" | 1 (edge: Position002) | 0 | 2 (edge tile_1, interior tile_29) [2023: 2, interior tile_29, tile_30] |
| `spots_near_zero` | Readout | `spots < spots_near_zero × median(spots)` | 0.05 | reads | "readout: spot count close to zero"; "spots < 0.05 x sample median" | 0 (lowest FOV at 0.07 × median; 0.1 × median marks 2 edge FOVs, Position006 and Position059) | 0 | 0 |
| `not_placed` | Segmentation and assembly | `placement_status` in (`missing`, `failed`) | categorical | — | `w376_placement_status.csv` (every LN and culture FOV `image_supported`; no tissue-2D placement in the first stitching run) | 0 | 0 | not countable from the source-inventory run (the accepted [stitching and assembly validation design](https://github.com/wanglab-broad/starfinder/blob/dev/design/validation/stitching-assembly.md), "Results of the real examples", places all 56 `image_supported` in the tissue-2D run: 0) |
| `stage_position_fallback` | Segmentation and assembly | The placement's `fovs.csv` `flags` holds `fallback` (`metadata_only` at the nominal position; a `supplied` placement is not a fallback) | categorical | — | as `not_placed` | 0 | 0 | as `not_placed` (0) |
| `reads_no_cell` | Segmentation and assembly | An owning FOV with `n_molecules_retained > 0` and `n_cells_retained = 0` | categorical | — | no row: no log holds cells, and the own-folder run's exports are `molecules_only` | not countable from the logs | not countable | not countable |

% W-384, W-383, W-376, W-388, W-389, W-398

Notes on the rules:

* **Counts are judged only near zero** (settled decision 5): `spots_near_zero` is the one rule on a count,
  and its limit is a fraction of the sample's median, so a FOV at the edge of the tissue
  with fewer spots is not marked unless it is close to zero. With `median(spots) = 0` the
  limit is 0 and nothing is marked.
* **Sides (choice A4).** No high-side deviation of either fraction is marked at any setting
  on the three samples (the source-inventory run's `candidate_limits.csv`, the rows "(high)"). Options: **SD1
  (recommended)**, the low side only, as `ReportLimits` states it; SD2, both sides, which
  marks a FOV that decodes better than its neighbours. The shift residual is judged on both
  sides because a shift has no good direction.
* **Rounds and axes of the residual rule.** Every moving round that `FOV.run` registered,
  every axis; one row per (FOV, round, axis) marked. The earlier pipeline also registered the
  morphology rounds (`flamingo`, `organelle`, `protein`), which the source-inventory run tabled apart; a Python
  run records only the rounds `FOV.run` registers, so no such round enters.
* **Alignment score.** Without a default (the source-inventory run's notes, section 11): a user who sets a limit gets
  the low side, `median − score > max(k' × s, floor')` or `score < absolute`. The first
  Python run of a real sample (the real example) gives the distribution of `ncc_after` from
  which Jiahao can set it; the two LN FOVs of the source-inventory run have 0.695 to 0.754.
* **The sample of amendment 2.** A sample with no run record does not evaluate `fov_failed`,
  `fov_not_run`, the registration rules or the readout rules: one `rule:<rule>` row each,
  `no_run_record` ({doc}`report-contract`, "A FOV without a run record").
* **Section.** The section column is the rule's section in this table; the Summary table
  lists the rows in the contract's order (FOVs in list order, then the rule order, then
  rounds and axes).

% D5, W-384, A2

### The reason texts

| Rule | Text in `problematic_fovs.csv` `reason` |
| --- | --- |
| `fov_failed` | "run failed at `<step>`" or "run interrupted at `<step>`" (with ", `<round>`" when the error names one) |
| `fov_not_run` | "expected and not run" |
| `shift_residual` | "shift residual above the limit (`<axis>`, `<round>`)" |
| `alignment_score` | "alignment score below the limit (`<round>`)" |
| `registration_rejected` | "registration rejected (`<round>`)" |
| `registration_fallback` | "registration succeeded with a fallback method (`<round>`: `<methods>`)" |
| `correct_form_fraction` | "correct-form fraction below the limit" |
| `matched_fraction` | "matched fraction below the limit" |
| `spots_near_zero` | "spot count close to zero" |
| `not_placed` | "not placed (`<placement_status>`)" |
| `stage_position_fallback` | "fell back to the stage position" |
| `reads_no_cell` | "reads but no cell" |

## The example FOVs

The automatic choice ({doc}`report-contract`, "The example FOVs"), with `r_f =
passed_filter(f)` (the read count) and `V` the FOVs whose `r_f` is not null, in list order:

1. **`example_1`, the main example.** `m = median(r)` over `V`. Sort `V` by
   `(abs(r_f − m), list_index)` and keep the first `q = max(1, ceil(abs(V) / 5))`: the set
   `N` "near the median" (choice A1). Order `N` by `list_index` and take
   `N[numpy.random.default_rng(seed).integers(len(N))]` with `seed = 0`.
2. **`example_2`, the fewest.** When some FOV has a non-null `n_cells_retained`, the owning
   FOV with the smallest `n_cells_retained` among those not yet chosen (rule
   `fewest_cells`); before a segmentation exists (no FOV has the value), the FOV of `V` with
   the smallest `r_f` among those not yet chosen (`fewest_reads`). Ties: the smallest
   `list_index`.
3. **`example_3`, the most.** As `example_2` with the largest value (`most_cells`,
   `most_reads`).
4. **Two roles on one FOV.** When no FOV is left for a role (a sample of one or two FOVs
   with values), the role takes the FOV its rule gives over all FOVs with a value, also one
   already chosen, so that `fovs.csv` `example` holds several roles. `V` empty: the
   automatic choice is unavailable with the reason of {doc}`report-contract`.

`report.json` `examples` records, per role, `fov_id`, `chosen_by`, `rule`, `value` (`r_f`
or `n_cells_retained`) and `seed`, and for `example_1` the size of `N` and the median.

| Option (choice A1) | "Near the median" | Note |
| --- | --- | --- |
| **M1 (recommended)** | The `max(1, ceil(n/5))` FOVs nearest the median | A fifth of the sample: 13 of 64 LN FOVs; one FOV for a sample of up to five. |
| M2 | The FOVs between the 40th and 60th percentiles | Empty for small samples without a rule for that case. |
| M3 | The one FOV nearest the median | No draw, so the seed has no effect; the main example is always the same FOV for a given sample. |

## The replay of the examples

The steps are those of {doc}`report-contract` ("The replay"); this page fixes the rounds and
what is drawn.

* **Rounds (choice A5).** The replay rebuilds two rounds of the main example: the
  **reference round** (`registration.<round>[*].reference`, the round `FOV.run` processes
  first; without a registration, the first key of `preprocessing.rounds`) and the **first
  moving round** in the order of `registration.csv`. The Preprocessing example and its
  intensity figure show the reference round; the Registration example shows the moving round
  against the reference round. `intensity.csv` holds the rows of both rounds. Options:
  **RD1 (recommended)**, these two rounds, which show every step from the raw input without
  needing a transform for the Preprocessing example, and the cost of two rounds; RD2, the
  moving round with the lowest `alignment_score` for Registration, which shows the worst
  round but changes with the values; RD3, every round, which costs four times the reading
  and memory of RD1 on LN (16 files, one round at a time, 447 MB per round as uint8).
* **Cost.** The replay holds one round of the main example at a time in the dtype of its
  step outputs; for an LN FOV 447 MB per round as uint8 (the source-inventory run's `real_fovs.csv`), and the
  rebuilt chain's resampling as `FOV._register_round` does it. Each file read is checked
  against its SHA-256 ({doc}`report-contract`, "The replay", step 1).
* **The displayed signal.** Each image of an example is the maximum over Z of one channel of
  the round's ZYXC image (the plane itself for `Z = 1`), at full resolution, then reduced to
  the figure's pixel size by the mean over blocks of `b × b` pixels, `b = ceil(n_x / w)` for a
  tile of `w` pixels (partial blocks at the edge are averaged over their pixels).
* **Grey levels of an example tile.** Linear from the 0.1th to the 99.9th percentile of that
  tile's displayed values (`lo`, `hi`), clipped; `lo = hi` draws the tile at mid-grey. Each
  tile has its own range, written beside its colour bar, because a step can change the range
  by orders of magnitude.

% W-384

## How the two routes differ

The route is read as {doc}`report-contract` ("The two routes") states. Every table has the
same columns on both routes; only these elements differ:

| Element | Route per FOV (`per_fov`) | Route after fusion (`after_fusion`) |
| --- | --- | --- |
| Segmentation preview of an example FOV | Labels: the assignment folder's `cell_labels.tif` and `nucleus_labels.tif` (or the linked `segmentation/<run>/labels.tif`), on the FOV's own grid. Background: the reference image `fovs/<fov>/reference.tif` (or the `reference_image` or `reference_image_import` pattern). | Labels: `segmentation_<run>.ome.tif` of the cell run (and of the nucleus run when the export has one), cut to the FOV's footprint (`fovs.csv` `footprint_*` mapped to the label grid by `floor((q − O) / S + ½)`, inclusive of the start, exclusive of the end). Background: the fused image of the marker patterns, cut to the same footprint. |
| Plane of a 3D preview | The plane `z*` of the label image with the most nonzero voxels; the background is the maximum over the planes of the slab of 3 µm centred on `z*` ("Sample measurements"). | A plane (the route after fusion needs a projection sample). |
| Cells attributed to a FOV (`n_cells`, `n_cells_retained`) | The FOV's own cells, and those it retains | The sample's cells whose ownership centroid the FOV owns (both columns equal) |
| Population accounting | Every reason of the route per FOV, with `duplicate_observation` when the rule is on | No duplicate rows (n/a); the cells' input are the run's labels |
| Workflow lanes | Segmentation and assignment in the per-FOV lane | `segment` in the per-sample lane |
| Time of the segmentation | `not_recorded` | `export.json` `segmentation.wall_s` |

Cell density, reads assigned to cells and the rule "reads but no cell" are the same formulas
on both routes (amendment 1).

% A1

## A FOV without a run record

In the sections up to the readout (Workflow overview, Preprocessing, Registration, Readout)
a FOV without a run record is a row of every per-FOV table with `run_record` false and its
values null, each named by its own row of `unavailable.csv` with `no_run_record`
({doc}`report-contract`). On the page:

* **Workflow overview.** The FOV status map and table show it in the category "no run
  record" (hatched, with the text in the tooltip); the time box plots leave it out and say
  so in the axis label ("n of N FOVs").
* **Preprocessing and Registration.** It is never the main example. Its map cells are drawn
  hatched with "no run record" in the tooltip; its heatmap rows are hatched.
* **Readout.** Its map cell is hatched; its row of the per-FOV table shows "—" with the
  reason in the cell's tooltip; the sample totals sum the other FOVs, and the bar's label
  says "n of N FOVs".
* **Later sections.** Made from the export, which holds the FOV when it was exported.

A sample with no run record at all (amendment 2) shows, in each of these four sections, the blocks
of the section each replaced by the unavailable box with the reason `no_run_record`, except
the FOV status map, which shows every FOV as "no run record".

% A2

## Layout of the page

The page follows the agreed mockup section by section. Where the mockup and the settled
decisions differ, the decisions win: the Files section has no marker file (settled decision 4), the
page has a light theme only (settled decision 3; the mockup's dark-scheme rules are left out), the
Summary names the route ({doc}`report-contract`, "The two routes"), and the cell values
and the segmentation preview follow the route (amendment 1). The page header is the heading
"Processing report" alone (Jiahao's review of 2026-10-14): the Summary's facts name the
dataset and the sample, and the page's title stays "Processing report · `<sample>`".

% D4, D3, A1

### How the page is built

No new dependency and no lock change (settled decision 3):

% D3

* **Template.** `starfinder/report/templates/report.html`, shipped in the package (the wheel
  target `packages = ["starfinder"]` of `src/python/pyproject.toml` already includes every
  file under the package, so `pyproject.toml` does not change) and read with
  `importlib.resources`. It holds the page skeleton, the CSS and the script inline: the
  mockup's CSS without the dark-scheme rules, and the mockup's script functions for the FOV
  maps, the tooltip, the legends, the paged tables and the side rail.
* **Filling.** The standard library only. The template holds named slots
  `<!--slot:<name>-->`, each replaced once, in a fixed order, by a string built in Python:
  text through `html.escape`; the data block as below; each static figure as an
  `<img src="data:image/png;base64,…">` or `data:image/jpeg` with an `alt` text naming the
  figure. A slot left unfilled, or filled twice, raises before anything is written.
* **The data block.** One `<script type="application/json" id="report-data">` holding the
  tables of {doc}`report-contract` ("The data embedded in the page"), written with
  `json.dumps(…, ensure_ascii=False, allow_nan=False, separators=(",", ":"))`, a missing
  value as `null`, and every `</` written as `<\/` so that the block cannot close the
  script element. It holds exactly the keys the contract lists, and the maps, tooltips and
  paged tables read only this block, as the contract says. What the script needs beyond the
  tables is part of the shipped template, not data: the reason texts ("How an unavailable
  part is shown") and the labels of the controls are constants of the inline script, and the
  map geometry ("The FOV maps") and the colour ranges are computed by the script from the
  rows of the block. Everything else on the page (the facts, the workflow chart, the
  placeholders of unavailable parts, the static figures) is written by the filler as HTML.
* **FOV maps.** Plain SVG built by the inline script from the data block: one `rect` per FOV,
  a `circle` and a `line` for an arrow; highlight and tooltip on pointer move; a selector or
  button group for the value shown. No library.
* **Static figures.** matplotlib with the `Agg` backend, the bundled DejaVu Sans font, 100
  dpi, saved to memory: plots as PNG, figures that draw image pixels as JPEG at
  `config.jpeg_quality` ({doc}`report-contract`, choice 15). PNG metadata holds no time, so
  the bytes of a figure depend only on its data.
* **No external resource.** The page loads nothing: no `http:` or `https:` address, no
  `//` address, no `<link>`, no `<iframe>`, no `<object>`, no `@import`, no `url(…)` other
  than a `data:` address, and every `src` is a `data:` address; fonts are the system fonts
  of the CSS stack. The comparison page links each sample's `report.html` by a relative
  address, which opens a local file.

### Style rules and how they are applied

From Jiahao's review of the mockup (the outline's style rules for the report):

% §2.11

| Rule | How the page applies it |
| --- | --- |
| No explanatory text under a heading | A block is its heading, an optional control in the title row, the figure or table, and its legend. Unit and coverage notes go in axis labels, legends and tooltips. |
| Every coloured figure has a colour bar with its range | Every colour scale is drawn with a bar labelled with its minimum and maximum (choice A8: **CR1 (recommended)**, the minimum and maximum of the values drawn, so no value lies off the scale; CR2, the 2nd to 98th percentile with values beyond drawn at the end colour, which keeps one outlier from flattening the scale but hides how far it lies; CR3, fixed ranges per quantity, which fit no sample in particular). A range of one value draws every cell at the middle colour with the one value on the bar. Example image tiles have their own grey bar with their range. |
| In a legend the categories are on the left and the scales on the right, on one line | The mockup's `legend(cats, scales)`: categories (failed, outside limits, no run record, not placed, …) left, colour bars and arrow scales right. |
| A FOV map highlights the FOV under the pointer and shows its values in a tooltip | Every SVG map; the tooltip's first line is the FOV, then the values of the selected quantity and its reason when unavailable. |
| A table with more than 20 rows is paged, 20 rows per page | The per-FOV table, the problematic FOVs, the FOV status table of a sample without positions, and the population and MECR tables when they exceed 20 rows. |
| The same information appears once | A value has one figure; the per-FOV table holds counts and the matched fraction, the map shows the fractions, the distributions the spread. The Problematic FOVs table is in the Summary only; the sections mark those FOVs with the "outside limits" outline. |
| Figures side by side have equal width, start on one line and have a divider between them, with any control in the title row | The mockup's `.g2.split` and `.bh` classes, unchanged. |
| A figure fills its column | Static figures are drawn at the pixel width of their column ("Figure sizes") and shown at 100 % of it. |

### The FOV maps

Each FOV is drawn as a rectangle centred at (`centre_x`, `centre_y`) of `fovs.csv`, in the
sample frame (X to the right, Y down, as the fused image). Its half-widths are half the
median distance between the centres of neighbouring placed FOVs along X and along Y (the
nearest other centre within 45° of the axis), times 0.94, so that the rectangles do not
overlap where the footprints do; a sample of one FOV uses its footprint. FOVs without a
centre (`fov_not_placed`, or no positions) are not drawn and the legend counts them ("n not
placed"). The categories, in legend order: failed (the run failed or was interrupted),
running, no run record, outside limits (a FOV with a row in `problematic_fovs.csv` for the
quantity shown: an outline in the warning colour), unavailable (hatched). Values are drawn on
a sequential scale from the page background to the accent colour (the mockup's `mix`);
the residual map uses arrows and colour as below.

### Summary

1. **Facts panel.** Dataset (`dataset_id`); Sample; Route (`per_fov`, `after_fusion`, or
   "no export"); Acquisition ("Other values of the page"); Configuration (the configuration
   check); Version (software); Run time (the earliest `started_at` to the latest `ended_at` of
   the run records); Generated (`generated_at`). Run time and Generated are each one line:
   the date, the start and the end of the run and the time zone (`2026-10-13, 14:19:02 –
   14:42:12 UTC`; an end on another date shows its date too, `2026-10-13, 23:50:00 –
   2026-10-14, 00:10:00 UTC`, and a run without an end reads "not ended"), and the date, the
   time and the time zone (`2026-10-13, 18:18:57 UTC`). The filler writes both in UTC; the
   page's inline script writes them again in the viewer's time zone with its abbreviation
   (`2026-10-13, 14:19:02 – 14:42:12 EDT`). Their tooltips keep the full ISO times and the
   run's seconds; an unavailable run time shows its reason. The panel's columns are at least
   340 px wide, so that both values fit on their line.
2. **Key numbers** (`report.json` `summary`): FOVs succeeded (`fovs_succeeded / fovs_expected`),
   median spots per FOV, spots matched to the codebook (`matched_fraction`, %), counted cells,
   median reads per cell, median genes per cell. An unavailable number shows "—" with the
   reason in its tooltip.
3. **Problematic FOVs.** A table from `problematic_fovs.csv`: FOV, Section, Measured (the
   value with its unit, round and axis, as the mockup writes "dx +38.0 px, round3"),
   Sample median, Reason (the reason text with the limit). Paged. No row: the table shows
   "No FOV outside the limits", and below it, in one line, the rules not evaluated with
   their reasons.

### Workflow overview

1. **Workflow chart** (static HTML cards written by the filler from `steps.csv`, the run
   records' configuration, `placement.json` and `export.json`). Two lanes, "Per FOV" and "Per sample", cards numbered in run order, four per row. Per-FOV cards and the steps they
   stand for: Preprocessing (`preprocess:*`; method: the recipe's steps joined by " → ");
   Registration (`register`; the succeeded methods); Spot finding (`find_spots`); Decoding
   (`decode_reads`; decoder and readout mode); Read QC filter (`filter_reads`);
   Deduplication; and on the route per FOV Segmentation and Assignment. Per-sample cards:
   Stitching (`placement.json` `method.method`), Segmentation (route after fusion),
   Assembly (the ownership rule of `export.json` `assembly.config`), Export (the export's
   parts written, with `dimensionality`). A step that the configuration leaves out is a
   dashed card "not run" (the mockup's `.st.off`); a step with no record yet is a dashed
   card "not yet run". The cards use the first run record in list order with status
   `succeeded`, else the first run record; FOVs whose step list differs are counted in the
   card's tooltip.
2. **FOV status** (SVG map), beside **Time per step** (static PNG): the FOV status
   categories; the time is a horizontal box plot per step name (log scale in seconds,
   whiskers at 1.5 × the interquartile range, every FOV beyond drawn as a point), the sample
   steps as single points on the same axis below a divider, steps without a time listed in
   the axis label as "not recorded". Time per step is drawn for the half column it sits in
   ("Figure sizes"), 26 px as shown per step name; over less than two decades its axis is
   labelled at 1, 2 and 5 × 10ⁿ s. Its axes are aligned with the FOV status map on their left
   (Jiahao's review of 2026-10-14): as shown, they span the map's FOV rectangles from their
   top line to their bottom line, the map's geometry computed from the rows of `fovs` as the
   inline script computes it ("The FOV maps") for a map as wide as the half column (at most
   480 px); when 26 px per step name need more height than the rectangles give, the axes
   start on the rectangles' top line and are 26 px per step name high. A sample without a
   placed FOV has no map: the figure is then at least 280 px high and its axes fill it.

### Preprocessing

1. **Example FOV · `<fov>`, `<round>`** (one static JPEG). A grid: one row per channel of the
   reference round (the channel labels of `run.json` `channels`), one column per state
   (raw input, then each recorded entry in the order `FOV.run` runs them, headed by the
   step's name). Each tile is the maximum over Z of that channel ("The replay of the
   examples"), with its own grey range and colour bar. Two adjacent tiles, side by side or one
   above the other, are separated by a white gap of at least 8 px as shown ("Figure sizes").
   The grid is drawn at least 1,398 px wide, so that it is shown at no more than two thirds of
   its drawn width (Jiahao's review of 2026-10-14): its tiles are 256 px, or the smallest whole
   size above at which the grid with its gaps is 1,398 px wide (693 px for two states, 458 px
   for three); its text is sized for that scale.
2. **Intensity after each step** (one static PNG). One panel per column of the grid, side by
   side and sharing nothing but the figure; in each, one curve per channel, the counts of
   `intensity.csv` against the bin centres, the count axis logarithmic (a bin of count 0 is
   left out of the line), channels in the colours of the legend. Each panel keeps its own
   count axis, with room for its labels between it and the panel before it; the legend is
   one row below the panels and their axis labels, at the figure's left margin.

### Registration

1. **Maps per moving round** (SVG, one per round, three per row), with the button group
   "Shift, residual" (default), "Shift, absolute", "Alignment score" in the title row:
   * residual: an arrow from the FOV centre of `(residual_x, residual_y)` scaled so that the
     largest arrow of the round is 0.45 of a rectangle's width, and the fill colour
     `abs(residual_z)`; the legend shows an arrow of a round number of voxels and the colour
     bar of `abs(residual_z)`;
   * absolute: the same with the shift;
   * alignment score: the fill colour `alignment_score`, no arrow.
   The caption of each map is the round, with "median of n FOVs" when fewer than three
   FOVs' rounds succeeded. Outline: the FOV has a `shift_residual` (residual and absolute) or
   `alignment_score` (score) row for that round.
2. **All FOVs and rounds** (static PNG). A heatmap: one row per FOV in list order, one column
   per (round, axis) of the residual (the axes under each round, the round's name below
   them), a diverging scale symmetric about 0 with range `[−a, a]`, `a` the largest absolute
   residual of the FOVs drawn (at least 1 voxel); null cells hatched grey. Every row drawn is
   labelled with its FOV and is at least 14 px high as shown (15 px). At most 64 rows are
   drawn: a sample of `N ≤ 64` FOVs draws every FOV; a sample of `N > 64` draws 64 of them,
   in list order: first the FOVs that have a row of the `shift_residual` rule in
   `problematic_fovs.csv` (the first 64 in list order if there are more), then, for the `k`
   rows left, the FOVs at the positions `floor(i × u / k)`, `i = 0 … k − 1`, of the `u` other
   FOVs in list order. The axis of the rows then reads "64 of N FOVs shown; every FOV in
   `registration.csv`". The tables are not sampled.
3. **Example FOV · `<fov>`, `<round>`** (one static JPEG of three tiles). "before
   registration", "after registration", "after registration, centre × 4": the reference
   round after its `steps` phase in green and the moving round in magenta, each the maximum
   over channels and over Z, each scaled to [0, 1] by its own 0.1th to 99.9th percentile;
   the RGB pixel is `(m, g, m)`, so the overlap is white. Before: the moving round after its
   `steps` phase; after: after its transforms ("The replay" of {doc}`report-contract`, the
   table of states). The third tile is the central quarter (in Y and X) of the "after" tile,
   enlarged four times by nearest-neighbour repetition; on the "before" and "after" tiles a
   yellow (`#ffd21f`) rectangle at least 2 px wide as shown bounds that quarter, its inner
   edge on the quarter's edge. Two adjacent tiles are separated by a white gap of at least
   8 px as shown. The legend: reference round, moving round, overlap, on a row below the
   tiles and their ranges, at the figure's left margin.

### Readout

1. **Sample totals** (HTML bars from `readout_totals`), beside **Per FOV** (SVG map). The
   bars, one per level in the table's order, share one scale from 0 to the spots; each is
   labelled with its count and fraction of the spots and, when fewer FOVs have the level
   than the spots, "n of N FOVs". The label is written over the whole row and is never
   clipped to its bar: the bar's width states the fraction, the label is readable in full
   whatever that width. A level that was not configured is a dashed bar with
   "not run"; an unavailable level, a dashed bar with "—" and the reason in its tooltip.
   Legend categories: "counted once per FOV that saw it" (every level above) and "after the
   overlaps are removed" (`reads_in_cells`, drawn in another fill); beside the last bar,
   "= sum of sample.h5ad" or "differs from sample.h5ad (sum n)". The map's "Colour by"
   selector: Spots detected, Fraction with correct form, Fraction matched to the codebook,
   Fraction merged by deduplication (when configured), Fraction passing the QC filter.
2. **Distribution across FOVs** (three static PNGs): spots per FOV, correct-form fraction,
   matched fraction. Histograms of `min(30, max(5, ceil(sqrt(n))))` equal bins from the
   smallest to the largest value (one bin of width 1 read or 0.01 when they are equal), a
   vertical line at the median with its value in the axis label.
3. **Per-FOV table** (paged, from `fovs`): FOV, Spots, Correct form, Matched, Merged (when
   configured), Passed QC, Matched %. A FOV with a readout rule's row is marked in the FOV
   cell.

### Segmentation and assembly

1. **Cells per FOV** (SVG map), with the selector Cells (`n_cells`), Cells per mm²
   (`cell_density_per_mm2`), Reads assigned to cells, % (`reads_in_cells_fraction`), Cells
   retained after assembly (`n_cells_retained`), beside **Placement** (SVG map): an arrow of
   `(correction_x, correction_y)` from each FOV centre, scaled so that the largest is 0.45 of
   a rectangle's width, with a scale arrow of a round number of µm; a uniform fill;
   categories "not placed", "fell back to the stage position" (dashed outline) and
   "placeholder" (hatched). The tooltip gives the placement status and `correction_z/y/x`.
2. **Example FOVs** (one static JPEG of three tiles, `example_1` to `example_3`, each titled
   with its FOV and role). The route's preview ("How the two routes differ"): the
   background in grey with its own range, the cell outlines in cyan (`#22d3ee`) and the
   nucleus outlines in yellow (`#ffd21f`), as the mockup draws them. An outline is a pixel
   of the label image whose label differs from one of its four neighbours, computed at full
   resolution and reduced to the tile by the maximum over blocks, so that no outline is
   lost. Two adjacent tiles are separated by a white gap of at least 8 px as shown. The
   legend: nucleus outline, cell outline, on a row below the tiles and their grey bars, at
   the figure's left margin.
3. **Population accounting** (table from `population`): Population, Cells, Molecules, the
   rows of `population.csv` whose two cells are not both n/a, labelled "Input, all FOVs",
   "Retained", "Excluded: not owner", "Excluded: FOV not placed", "Excluded: duplicate
   observation", "Excluded: cell not owner", "Excluded: cell duplicate observation",
   "Excluded: outside the grid", "Counted". An n/a cell is blank; an unavailable cell "—".

### Sample measurements

1. **Per-cell distributions** (three static PNGs): reads per cell, genes per cell, cell
   size (from `cells.csv`). Equal bins over `[0, p99]` of the values, `p99` the 99th
   percentile, at most 30 bins and for integer values bins of integer width
   `max(1, ceil((p99 + 1) / 30))`; the cells above `p99` in one last bar labelled
   "> p99: n cells" at the top right of the plot; the median as a vertical line with its value
   in the axis label; the cell size's unit in its axis label (one histogram per unit, "Cell
   size"). The figures stand in rows of three, each in a third of the column; a row of two
   figures has two columns and a single figure takes a half column, and each figure is drawn
   for the slot it gets ("Figure sizes"). The distributions across FOVs (Readout) are three,
   one row of thirds.
2. **Marker patterns** (static JPEGs), one figure per marker group, with a group selector in
   the title row when there is more than one group; without markers, one figure of the three
   genes with the most retained molecules, titled "Most abundant genes (no markers
   supplied): A, B, C". Each figure has:
   * **The whole-sample view.** The fused image of the export: the first image of
     `export.json` `fusion.outputs` (the configuration's first `images` entry,
     `reference_stain` by default), its projection file when it exists, else the maximum
     over Z of its stack file, read plane by plane (the stack of LN sample1 at scale 0.5 is
     66 × 6,167 × 5,435 uint8, the own-folder run's `export.json`; one plane is 34 MB). The image is
     reduced to `config.figure_width_px` by the mean over blocks; grey from its 0.5th to
     99.8th percentile of the nonzero pixels, with its colour bar. Over it, one point per
     retained molecule (`tables/molecules.csv` with `assembly_status == "retained"`) of the
     group's genes at `(sample_x, sample_y)` mapped to the image's pixels by the fused
     grid (`(q − O) / S` per axis, `O` and `S` the origin and spacing `fusion.outputs`
     records), one colour per gene in the group's order (Okabe–Ito orange `#E69F00`, sky
     blue `#56B4E9`, bluish green `#009E73`, yellow `#F0E442`), drawn in group order so the
     last gene is on top, opacity 0.8. The footprints of the three example FOVs (the zoom-in
     regions) are drawn as yellow (`#ffd21f`) rectangles at least 2 px wide as shown, their
     inner edge on the footprint, numbered 1 to 3: each number in text of 12 px as shown,
     white on a dark backing, inside its own rectangle at the first corner (top left, top
     right, bottom left, bottom right) where it covers no number already placed and lies in no
     other region's rectangle (else the first where it covers no number). A scale bar of
     1, 2 or 5 × 10ⁿ µm, at most a fifth of the width, in the lower right, its label on a dark
     backing.
   * **The zoom-in regions** (three columns, regions 1 to 3 = `example_1` to `example_3`).
     The XY view of each region is the example FOV's footprint, cut from the fused image at
     its full resolution and drawn at the zoom tile's width. **The slab** (choice A6): a slab
     of `slab = 3 µm` in Z centred on `z_c`, the median `sample_z` of the region's retained
     molecules (all genes); the image is the maximum over the planes of the stack whose
     centre lies in `[z_c − slab/2, z_c + slab/2)`, and the points are the molecules of the
     group with `sample_z` in the same interval, so the image and the molecules share the
     slab. A projection export has no planes: the image is its projection and the points
     every retained molecule of the region, and the panel's title says "projection". The XZ
     view of each region: the stack's maximum over the rows whose centre lies in a band of
     `slab` in Y centred on the footprint's centre, over the footprint's X range; the points
     are the molecules of the group in the same band, at `(sample_x, sample_z)`; Z upward,
     drawn at the physical scale with Z stretched by `e = max(1, (W / 6) / H)` for a panel of
     width `W` and Z extent `H` in µm, so that the panel is at most 6:1, with `e` printed in
     its title. Without a stack the XZ view is unavailable with `no_stack`.
   * **The point size.** A point's diameter in pixels of the drawn image is
     `max(2, d_mol / p)`, `d_mol = 1.5 µm` and `p` the µm per pixel of that image; the
     points are rasterized into the image (`rasterized=True`), never a vector element. For
     the LN whole-sample view at 1,600 px (about 1.5 µm per pixel) the floor of 2 px
     applies; for an LN zoom tile of 512 px over a 291 µm footprint, about 2.6 px.
   * **`Z = 1`.** The slab is the one plane: the XY views show every retained molecule of
     the region over the plane, and the XZ views are unavailable with `no_stack`.
   * The legend: the genes with their colours, then the image name, on a row below the
     image and its grey bar, at the figure's left margin; the scale bars.
   * Options (choice A6): **SL1 (recommended)**, a slab of 3 µm, thinner than a nucleus,
     and points of 1.5 µm with a floor of 2 px, which stay visible in the whole-sample view
     (about 1.5 µm per pixel at 1,600 px for LN); SL2, a slab of 6 µm and points of 1 µm with a floor of 1 px, which
     shows whole nuclei but lets points vanish in the whole-sample view; SL3, no slab (the
     projection and every molecule of the region), which draws molecules above and below a
     nucleus over it.
3. **Marker co-expression** (table from `mecr`), beside **Reads along Z** (static PNG). The
   table: Pair (`A / B`), Cells with A, Cells with B, Both, Either, MECR (three decimals;
   "undefined" when `either` is 0). Without `marker_pairs` the block is unavailable with
   `no_markers`. The heatmap (drawn for its half column): one row per FOV, each labelled with
   its FOV and at least 14 px high as shown, at most 64 rows by the rule of "All FOVs and
   rounds" without marked FOVs (the axis of the rows then reads "64 of N FOVs shown; every
   FOV in `reads_along_z.csv`"), one column per plane, labelled every 1, 2, 5, 10, … planes
   so that the labels do not meet, the colour the FOV's
   fraction of its molecules on that plane (`reads / Σ_k reads` of the FOV), on a sequential
   scale from 0 to the largest fraction of the FOVs drawn, with its colour bar; the columns run from the
   lowest to the highest `z` of the table's rows, over all FOVs (from the lower of plane 0
   and the lowest plane a molecule has, to the higher of the largest `n_z − 1` of the FOVs
   and the highest plane a molecule has), and a plane outside a FOV's own rows is hatched in
   its row. A row outside a FOV's grid is drawn like the others, and each FOV's row carries a
   thin line at the two edges of its grid, between the columns of `z = −1` and 0 and between
   those of `n_z − 1` and `n_z`, wherever the figure has a column on both sides (decided at the
   gate of 2026-10-12; the lines need only the grid shape the part already reads). With `Z = 1` and every molecule on plane 0, one
   column; a projected sample whose molecules lie on several planes has a column per plane.

% W-398, W-387

### Files

Three tables of two columns, as the mockup, without the marker row (settled decision 4):

% D4

* **Input.** Images: the common folder of `run.json` `inputs` paths and the number of
  files; Codebook: `codebook.path` (R2) with its SHA-256 (the first 12 characters, the whole
  in the tooltip); Tile positions: `placement.json` `table`.
* **Configuration.** Workflow: `workflow_config.path` (R4); Hash: its SHA-256; Report
  settings: `report.json` `config`, shown as its JSON in a collapsed element.
* **Output.** FOV folders `<output_root>/fovs/<fov_id>/`; Placement
  `stitching/<sample>/`; Sample export `export/<sample>/`; Report
  `report/<sample>/report.html`; Report tables `report/<sample>/tables/`. Every place
  exists or is marked "not written".

### How an unavailable part is shown

* **A part unavailable for the whole sample:** in place of its figure or table, the
  mockup's hatched placeholder (`.ph`) with "Unavailable:" and the reason's text, then the
  detail in smaller type. The box keeps the figure's aspect ratio, so the layout does not
  move.
* **A value unavailable for one FOV:** "—" in a table cell and a hatched rectangle on a map,
  each with "Unavailable: <reason text>" and the detail in the tooltip. Never a zero and
  never a colour of the scale.
* **A rule not evaluated:** listed under the Problematic FOVs table, one line, rule and
  reason.
* **Reason texts** (one sentence each, constants of the template's inline script):

| Reason | Text |
| --- | --- |
| `no_run_record` | no run record for this FOV |
| `run_not_finished` | the run has not finished |
| `step_failed` | the run failed before this step |
| `not_configured` | not part of the run's configuration |
| `not_recorded` | the run did not record this value |
| `no_placement` | no placement for this sample |
| `fov_not_placed` | this FOV was not placed |
| `no_export` | no export for this sample |
| `part_not_written` | the export part that writes this was not written |
| `no_segmentation` | the sample has no cells |
| `no_markers` | no marker pairs supplied |
| `no_stack` | the export has no 3D stack |
| `input_missing` | an input file is missing |
| `input_changed` | an input file differs from its record |
| `field_not_saved` | the registration field was not saved |
| `replay_differs` | the replayed preprocessing differs from the record |
| `undefined` | undefined: the denominator is zero |

(report-algorithms-figure-sizes)=
### Figure sizes

Every static figure of `report.html` and `index.html` is drawn for its slot, the width at
which the page shows it in a window 1,200 px wide (choice A7, PX1, with Jiahao's review of
2026-10-13). The slots follow from the page's CSS: the content column is 1,160 px (the
`.wrap`'s 1,200 px less its padding), the full column of `report.html` 932 px (less the side
rail of 200 px and its gap of 28 px), a half column (`.g2`, gap 48 px) 442 px and a third
(`.g3`, gap 16 px) 300 px; `index.html` has no side rail. The widths are the constants
`FULL_COLUMN`, `HALF_COLUMN`, `THIRD_COLUMN` and `COMPARISON_COLUMN` of
`starfinder/report/_figures.py`:

| Slot | Width (px) | Figures |
| --- | --- | --- |
| Full column | 932 | Intensity panels, All FOVs and rounds, Preprocessing grid, Registration and segmentation examples, whole-sample marker view |
| Half column | 442 | Time per step, Reads along Z; a figure of the three-figure grid in a row of one or two |
| Third of the column | 300 | A figure of the three-figure grid in a row of three (distributions across FOVs and per cell); marker zoom-in tiles |
| Comparison page column | 1,160 | The box plots by sample of `index.html` |

The rules:

* **Size.** A plot (PNG) is drawn at twice the width of its slot, so that it is sharp on a
  dense screen. An image figure (JPEG) keeps its tiles and is shown scaled to its slot. No
  figure is shown wider than two thirds of its drawn width (`s / w ≤ 2/3`, Jiahao's review
  of 2026-10-14): the browser never enlarges a figure's text or tiles; the Preprocessing grid
  is drawn larger to meet it.
* **Text.** No text of a figure, plot or image, is smaller than 11 px as shown: a figure
  `w` px wide in a slot of `s` px is shown at the scale `s / w`, and its texts are written at
  12 px as shown (titles 13 px), that is `12 · w / s` px in the figure, at 100 dpi with the
  bundled DejaVu Sans. Texts that would not fit their place are broken into lines.
* **No overlap.** The layout of each plot is measured on the drawn figure: no text overlaps
  another axes, another text or the edge of the figure.
* **Tiles.** Two adjacent tiles, side by side or one above the other, are separated by a
  white gap of at least 8 px as shown: `n` tiles of `t` px side by side have the smallest
  whole gap `g` with `g · s / (n · t + (n − 1) · g) ≥ 8`.
* **Heatmap rows.** A row is at least 14 px high as shown (15 px), every row drawn has its
  FOV label, and at most 64 rows are drawn ("Registration").
* **Marks.** A region's mark is a yellow (`#ffd21f`) rectangle at least 2 px wide as shown.
* **Legends.** A figure's legend is a row below its plots or tiles (and their axis labels,
  ranges and grey bars), its left edge at the figure's left margin, over nothing, as the
  legends of the page's maps sit below the maps (Jiahao's review of 2026-10-14). A colour or
  grey bar that belongs to one tile stays with its tile.

| Figure | Slot | Shown width (px) | Drawn width × height (px) | Format |
| --- | --- | --- | --- | --- |
| Time per step | half | 442 | 884 × (beside the FOV status map: the axes span its FOV rectangles, or 52 per step name from their top when that is more, + its labels; without a map: `max(560, 52 per step name + its labels)`) (26 px per step name as shown) | PNG |
| Intensity after each step | full | 932 | 1,864 × 720 (360 px as shown) | PNG |
| All FOVs and rounds | full | 932 | 1,864 × (30 px per row drawn, at least the colour bar's label, + its labels) | PNG |
| Reads along Z | half | 442 | 884 × (30 px per row drawn, at least the colour bar's label, + its labels) | PNG |
| Distributions across FOVs and per cell | third or half | 300 or 442 | 600 or 884 wide: 600 × 340 in a third, 884 × 501 in a half (the proportion 600 : 340) | PNG |
| Box plots by sample (`index.html`) | comparison | 1,160 | 2,320 × 773 | PNG |
| Preprocessing example grid | full | 932 | at least 1,398 wide: tiles of 256 px, or the smallest whole size above whose grid with its gaps is 1,398 px wide; the headers and strips sized to their text | JPEG |
| Registration and segmentation examples | full | 932 | 1,564 wide (512 px per tile with its gaps); the header, strip and legend sized to their text | JPEG |
| Whole-sample marker view | full | 932 | `figure_width_px` wide (1,600; contract choice 15) | JPEG |
| Marker zoom-in tile, XY and XZ | third | 300 | 512 wide | JPEG |

PX1 (recommended) is this table; PX2 halves every width (half the bytes, less detail in the
example tiles); PX3 renders every figure at its column width (1×), which is soft on a dense
screen.

### Size budget

From the source-inventory run's `figure_sizes.csv` (a whole-sample projection at the LN pixel size: 1.1 MB as
JPEG at 1,600 px, 1.4 MB with three marker genes; base64 adds a third), the parts of a page
of a 64-FOV sample with two marker groups, before base64:

% W-384

| Part | Bytes, estimate |
| --- | --- |
| Template (CSS and script) | ≤ 60 KiB |
| Data block (tables of 64 FOVs: `fovs` 64 rows, `steps` about 1,800 rows, `registration` 192 rows, `unavailable`) | ≤ 1 MiB |
| About 14 plots (PNG) | ≤ 120 KiB each, 1.7 MiB |
| Preprocessing, registration and segmentation examples (JPEG) | ≤ 600 + 400 + 400 KiB |
| Per marker group: whole-sample view and six zoom-in tiles (JPEG) | ≤ 1.5 MiB + 0.5 MiB |

With base64 on every figure, about 6.5 MiB for the sample and 2.7 MiB per marker group:
about 12 MiB for LN sample1 with the two groups of the inputs survey's `markers.yaml`. The budget
(choice A9) is **SB1 (recommended)**: `report.html ≤ 6 MiB + 16 KiB × FOVs + 3 MiB × marker
figures` (one marker figure without markers), 13 MiB for LN sample1; SB2, a fixed 20 MiB;
SB3, no budget. A page above the budget is still written, and `report.json` records its
bytes and the budget; the validation checks the budget on the synthetic pages and the real
example reports it.

% W-383

## The comparison page

`Dataset.compare_samples` writes `report/index.html` and `report/samples.csv` from the
finished sample reports only ({doc}`report-contract`, "The comparison entry"), with the same
template machinery (`starfinder/report/templates/index.html`), the same CSS and rules, and
one data block holding `samples` and, per present sample, the three per-FOV columns of its
`tables/fovs.csv`.

1. **Samples** (table from `samples.csv`, paged): Sample (a relative link to
   `<sample>/report.html` when present), Report, Generated, Route, FOVs expected, FOVs
   succeeded, Median spots per FOV, Matched fraction, Counted cells, Median cells per FOV. A
   missing sample's row says "missing" and its other cells are blank (n/a).
2. **Three box plots by sample** (static PNGs, one per row of the page): spots per FOV
   (`spots`), fraction matched to the codebook (`matched_fraction`), cells per FOV
   (`n_cells_retained`); one box per present sample in the order of `samples`, whiskers at
   1.5 × the interquartile range, each FOV beyond drawn as a point, the number of FOVs with a
   value under each box; a sample whose column has no value is an empty slot labelled with
   the reason ("no run record", "no cells", "missing").

## Validation design

Engineering validation only (settled decision 7): known answers with pass or fail conditions fixed before
the run, in default-tier pytest modules, plus `slow` modules for the synthetic chains. No
check compares methods, sweeps a parameter or claims accuracy; the checks establish that
the records are read and combined as this page and {doc}`report-contract` state, not that a
sample is biologically sound. `Z = 1` and the comparison page are validated on synthetic
data only (settled decision 7).

% D7

Rules:

* **Fixtures are hand-written records or the synthetic sample scene**, built in the test or
  in a test helper module; no test reads the network share, a run directory, real images or
  the network, and no generated image is committed.
* **Expected values** are written in the fixture by hand, or computed in the test by the
  formula this page states from the fixture's own numbers (for example `count / spots` in
  float64), or come from a package function that is not part of the report and that the row
  names (`ownership_map`, `dataset.load_checkpoint`). Never from the report's own output.
* **Seeds.** The synthetic scenes use the preset seed 42 of {doc}`synthetic-sample-scene`;
  the example draw uses its own seed 0, which is part of the algorithm ("The example FOVs").
  No other random number is used. The hand-written fixtures use none.
* **Tolerances.** Exact equality for every count, string, boolean, reason, row order, dtype,
  hash and image; floats exact where the test computes them by the same float64 formula from
  the same inputs (the fixtures' values are integers or dyadic fractions where a sum is
  involved); the replay bit-identical. A check that says two outputs are equal compares them
  whole, unless it names the fields it leaves out. A tolerance is not adjusted during the
  run; a correct implementation that cannot meet one goes to Jiahao.
* **Markers** (choice A11). Every test file carries a subsystem marker and the kind marker
  `validation` (or `contract` for the entry checks). **TM1 (recommended)**: the first issue
  that adds `starfinder.report` registers a subsystem marker `report` in
  `src/python/pyproject.toml` (a marker line, no dependency or lock change), so that the
  report's tests can be selected; TM2: the report's tests carry `dataset` (the entry is on
  `Dataset`), and `integration` for the chains, with no change to `pyproject.toml`. A test
  that takes 5 s or more is `slow`.
* **Extras.** A test that writes or reads an export needs the `anndata` extra, and one that
  fuses needs `multiview-stitcher`; each skips with a stated reason where its extra is not
  importable, so the CI environment (`dev`, `local-registration`, `checkpoint`) passes or
  skips every test and fails none.

### Fixtures

| Fixture | Construction | Seeds | Needs |
| --- | --- | --- | --- |
| `report_records` | A hand-written output root of one sample, `s1`, with four FOVs `F1` to `F4` on a 2 × 2 grid. Each FOV's grid is 4 × 16 × 16 voxels at spacing (0.5, 0.25, 0.25) µm (footprint 4 × 4 µm in Y and X), nominal step 3 µm. Run records written as JSON in the form `dataset/_run_record.py` writes them, with R1 to R4 ({doc}`report-contract`, "Records added to run.json"), status `succeeded`, the steps of loading, preprocessing, registration and detection to filtering with stated `seconds`, the recipe `min_max_normalization` then `white_tophat`, one translation step, deduplication and a filter with `end_bases`. Reference round `round1`, moving rounds `round2`, `round3`; translations (z, y, x) in round2 F1 (0, 1, 2), F2 (0, 1, 3), F3 (0, 2, 2), F4 (0, 1, 2), in round3 F1 (0, −1, 0), F2 (0, −1, 0), F3 (1, −1, 0), F4 (0, −1, 8); `ncc_after` of the application entries round2 0.9, 0.9, 0.85, 0.9, round3 0.88, 0.88, 0.87, 0.3. Counts: `spots` 1000, 1200, 800, 1000; `endpoint_valid` 800, 960, 400, 800; `assigned` 700, 840, 560, 700; `merged_reads` 10, 20, 0, 10; `accepted` 650, 800, 500, 640. A placement saved with `save_placement`: every FOV `image_supported` at origins (z, y, x) F1 (0, 0, 0), F2 (0, 0, 3.25), F3 (0, 3, 0), F4 (0, 3, 3) µm (F2 corrected by +0.25 µm in X), review `not_needed`. | none | hand-built |
| `report_records` variants | `failed`: F3 `failed` with `error` `{step: register, round: round3}`, no round3 entries and no counts. `not_run`: F4's run record removed. `stage_off`: deduplication null in every `config` and no `merged_reads`; and a copy with `config.pipeline.registration` null and no `registration`, `transforms` or `register` steps. `no_placement`: the `stitching/` folder removed. `one`: F1 alone. `equal`: four FOVs with F1's counts, translations and scores. `no_spot`: F2's counts all 0. `older`: R1 to R4 removed from every run record, once without and once with F1's `registered/transforms.json` (`format_version` 3 since 2026-10-24, 2 before; the entries of `_checkpoint.py`: `_transform_json`). `placement_states`: F3 `metadata_only` flagged `fallback`, F4 `failed`; and a copy whose four FOVs are `metadata_only` flagged `supplied`. | none | hand-built |
| `report_cells` | `report_records` with an export folder written in the test: `export/s1/tables/fovs.csv`, `cells.csv`, `molecules.csv`, `counts.csv` and `sample.h5ad` (through `assignment.write_h5ad`), with `export.json` written through `export/_record.py` so that `verify_section_hashes` passes; route `per_fov`. Five genes `G1` to `G5`; `n_cells_retained` F1 21, F2 0, F3 7, F4 14, with F2 holding retained molecules (reads but no cell); one retained cell `excluded_no_nucleus` (not counted) and one `not_owner` cell, each with molecules; molecule `z` values that include −0.6, −0.5, 0.4999, 0.5 and 3.5 on a grid of 4 planes. Variants: `nocal`, whose cells have `size_physical` NaN and `calibration` `unknown`; `mismatch`, whose `sample.h5ad` holds one count more than its molecules (written with its own hash, so the file agrees with its record); `no_markers`, with `ReportConfig()`. | none | hand-built; `anndata` |
| `hand_tables` | DataFrames and dicts built in each test for the definitions: read counts; chains of results (translation `(0, 1, 2)`; affine with `A = diag(1, 1, 1.1)`, `b = (0, 0.5, −0.75)` on a 4 × 16 × 16 grid, centre (1.5, 7.5, 7.5); a dense last step); count matrices for MECR; molecule `z` values; step outputs of the dtypes uint8, uint16, float32 and bool for the bins. | none | hand-built |
| `scene_chain` | `sample_scene_preset("sample-tiny-v1")` (2 × 2 FOVs of 5 × 36 × 36, four rounds of four channels): the FOV round images written as OME-TIFFs in the dataset layout, `FOV.run` with the source-inventory run's synthetic recipe (the source-inventory run's `scripts/synthetic_run.py`: min–max normalization to uint8 and a white top-hat of radius 3, one translation step on the channel maximum, local maxima, extraction, WTA decoding, scoring, deduplication and a filter with `end_bases`) and checkpoint stages `candidates` and `pre_qc`; the rule outputs; the truth labels imported with `import_labels(…, relabel=False)` and `FOV.assign_molecules`; the truth placement (`placement_from_table` of `placement_truth.csv`, approved) saved; `Dataset.export` on the route per FOV with `dimensionality: both`. `scene_chain_registered`: the same with the stage `registered` too. | 42 | generator addition; `anndata`; `multiview-stitcher`; `slow` |
| `scene_chain_plane` | The same chain on `scene_plane` ([stitching and assembly validation design](https://github.com/wanglab-broad/starfinder/blob/dev/design/validation/stitching-assembly.md), "Fixtures": one-plane FOVs of 1 × 36 × 36), exported twice: on the route per FOV, and on the route after fusion with the truth nuclei projection imported and cells by `seeded_watershed` on the fused image, as `per_fov_routes` and `seeded_runs` of {doc}`sample-segmentation-validation` build them. | 42 | as `scene_chain` |
| `scene_no_run_json` | `scene_chain`'s output root with every `fovs/<fov>/run.json` removed, and a second root exported with `assignment=None` from the rule outputs `fovs/<fov>/molecules.csv` alone (no checkpoint folder), as the own-folder run exported LN sample1. | 42 | as `scene_chain` |
| `scene_two_samples` | `scene_chain` as sample `s1` with its report, and `scene_chain_plane`'s route per FOV saved as sample `s2` under the same output root, without a report. | 42 | as `scene_chain` |

% W-565, W-384, W-398

### Checks

(report-algorithms-checks)=

| # | Check | Fixture (seed) | Metric | Pass/fail tolerance |
| --- | --- | --- | --- | --- |
| RC1 | Collection, a normal sample | `report_records` (none) | Every cell, dtype and row order of `fovs.csv`, `registration.csv`, `steps.csv`, `readout_totals.csv` and `problematic_fovs.csv` | Equal to the expected tables written in the test, compared whole (nothing left out). Among them: residuals F2 round2 `x` +1, F3 round2 `y` +1, F3 round3 `z` +1, F4 round3 `x` +8, every other residual 0; `readout_totals` spots 4000, correct form 2960 (0.74), matched 2800 (0.7), merged 40 (0.01), passed 2590 (0.6475), each with `fovs` 4; `problematic_fovs.csv` exactly two rows, (F3, `correct_form_fraction`, measured 0.5, median 0.8, limit 0.05, `fraction`) and (F4, `shift_residual`, round3, `x`, measured 8, median 0, limit 5, `voxels`); `unavailable.csv` exactly the rows the test lists: `rule:alignment_score` (`not_configured`), the time of `stitch` (`not_recorded`), the replayed examples (`input_missing`: the fixture writes no image) and every part, column and rule that needs an export (`no_export`). Exact. |
| RC2 | Collection, a failed FOV | `report_records` `failed` (none) | F3's cells; residual medians; rule rows | F3 `status` `failed`, `failed_step` `register`, `failed_round` `round3`; its round2 shift kept, its round3 cells and its counts null, each named by one row with `step_failed`; round3 medians over F1, F2 and F4 only (residual F4 `x` 8); one `fov_failed` row "run failed at register, round3". Exact. |
| RC3 | Collection, a FOV expected and not run | `report_records` `not_run`; and a copy with every run record removed but F1's | F4's cells; rule rows | F4 `run_record` false and every column up to the readout null, each named by one row with `no_run_record`; one `fov_not_run` row for F4; medians and totals over F1 to F3 (`fovs` 3; `spots` 3000, `matched` 2100, median spots 1000). In the copy with F1's run record alone: F2, F3 and F4 each named by `no_run_record` rows and marked by `fov_not_run`; medians and totals over F1 alone (`fovs` 1; `spots` 1000, `matched` 700, median spots 1000). Exact. |
| RC4 | Collection, a stage that was not run | `report_records` `stage_off` (both copies) | Deduplication and registration cells; chart cards | `merged_by_deduplication` and `merged_fraction` null with `not_configured` for every FOV; the `readout_totals` row `merged_by_deduplication` with `count` null (`not_configured`) and `fovs` 0; without a registration, `registration.csv` has no row and the registration parts are named by rows with `not_configured`; the chart's Deduplication (or Registration) card is "not run". Exact. |
| RC5 | Collection, no placement | `report_records` `no_placement` (none) | Placement columns; map parts | `placement_status`, `correction_*`, `footprint_*`, `centre_*` null with `no_placement` (one row each); `registration.maps` and the readout map named by rows with `no_placement`; the heatmap and the tables present; `not_placed` and `stage_position_fallback` not evaluated, one `rule:` row each with `no_placement`. Exact. |
| RC6 | Collection, no marker | `report_cells` `no_markers` (none) | `mecr.csv`; the marker figure; `report.json` | `mecr.csv` has its header and no row, `sample.mecr` named by one row with `no_markers`; the marker figure's title names the three genes with the most retained molecules, in decreasing order of molecules (ties by the gene order of `assembly.genes`), as the test counts them from `molecules.csv`. Exact. |
| RC7 | Collection, an unknown calibration | `report_cells` `nocal` (none) | `cells.csv` `size`, `size_unit` | `size` equals `size_voxels` and `size_unit` is `voxels` on every row; the histogram's axis says voxels. Exact. |
| RC8 | Collection, a FOV with no spot | `report_records` `no_spot` (none) | F2's fractions; rules | F2's four fractions null with `undefined` (detail `spots`); F2 enters no median of a fraction rule; F2 is marked by `spots_near_zero` (0 < 0.05 × 900, the median of 1000, 0, 800, 1000); its counts are written as 0. Exact. |
| RC9 | Collection, a sample of one FOV | `report_records` `one` (none) | Residuals, rules, examples | Every residual exactly 0; no row in `problematic_fovs.csv`; `fovs.csv` `example` of F1 is `example_1;example_2;example_3`; maps and heatmap drawn with one FOV. Exact. |
| RC10 | Collection, FOVs with equal values | `report_records` `equal` (none) | Residuals, rules, the example draw | Every residual 0; no FOV marked by any rule on values; `example_1` is F1 (`N` holds the `ceil(4/5) = 1` FOV first in the order `(distance, list_index)`). Exact. |
| RC11 | Collection, an older run record | `report_records` `older` (both copies) | Columns of R1 to R4 | Without `transforms.json`: every shift and residual null with `not_recorded`, `transform_source` null, the registration example unavailable with `not_recorded`; with F1's `transforms.json`: F1's shifts equal those of `report_records` (nothing left out) and `transform_source` is `transforms.json`; `correct_form` null with `not_recorded`; the Files codebook and workflow entries `not_recorded`. Exact. |
| RC12 | Null cells and the closed lists | every `report_records` and `report_cells` case of RC1 to RC11 | Each null cell of each table and each null `summary` key | Each is either n/a by its column's null rule or named by exactly one row of `unavailable.csv` (table, column, row key), with a reason its part can have ({doc}`report-contract`, "Parts, their inputs and their reasons"); no row names two values; every reason is in `UNAVAILABLE_REASONS` and every part in `REPORT_PARTS` or `rule:<rule>`. Exact. |
| RC13 | Collection, a round that did not finish ("Outcome", choice A13) | `report_records` `failed`, three copies, in each of which the test writes F3's `registration.round3` as `FOV._register_round` leaves it and no `transforms.round3`: `applying`, a `succeeded` estimation entry with its QC (`ncc_after` 0.8) and then the application entry left `applying` with `qc` null, F3 `status` `failed`; `estimating`, one estimation entry left `estimating` with `qc` null, F3 `status` `interrupted`; `succeeded`, one `succeeded` estimation entry with its QC (`ncc_after` 0.8) as the last entry, F3 `status` `failed`; `error` `{step: register, round: round3}` in each (none) | F3's round3 `outcome` and `alignment_score` and their rows of `unavailable.csv`; the round3 residual medians | `outcome` `application_failed`, `failed` and `failed`, each non-null and named by no row of `unavailable.csv`; `alignment_score` null with `step_failed` in each copy, named by exactly one row (`registration.csv`, `alignment_score`, (F3, `round3`)): the round's `outcome` is not `succeeded` ("Alignment score"), so neither the leftover `applying` entry's null QC (not `undefined`) nor the estimation entry's 0.8 is read; round3 medians over F1, F2 and F4 only, F3 in no median, F4's residual `x` 8. The strings and reasons exact; the residual within 1e-9 absolute. |
| DF1 | Read levels | `hand_tables` (none) | Counts and fractions | Each fraction equals `count / spots`; `matched > correct_form` is kept (not nested); `spots = 0` gives four nulls with `undefined`; a null count gives a null fraction with the count's reason. Exact. |
| DF2 | Shift | `hand_tables` (none) | `shift_z/y/x` | Translations `(0, 1, 2)` and `(0, −1, 0)` give `(0, 0, 2)`; the affine alone gives `(0, 0.5, 0.0)`; the translation as `T₁` and the affine as `T₂` give `(0, 1.5, 2.0)`, the reverse order `(0, 1.5, 2.2)`; a chain whose last step is dense gives the global steps' shift; a dense step alone gives n/a. The translations exact; every value that involves the affine within 1e-12 absolute, because its float64 bits depend on the order of the operations (in the order of `results/check_expectations.py` of the algorithms specification's run directory the reverse order gives x = 2.200000000000001). |
| DF3 | Shift residual | `hand_tables` (none) | `residual_*` | Medians of an odd and an even count as stated; one succeeded FOV gives 0; two give `±d/2`; a FOV whose round is `rejected`, `failed` or `application_failed` has no residual and is not in the median; no succeeded FOV gives every residual null with `undefined`. Exact. |
| DF4 | Alignment score | `hand_tables` (none) | `alignment_score` | Equals the application entry's `qc.values.ncc_after`, not the estimation entry's (they differ in the fixture); a null QC gives null with `undefined` and the QC's reason in the detail; a round without an application entry gives `step_failed`. Exact. |
| DF5 | Area a FOV owns | `report_records` placement and grids; `placement_states` (none) | `owned_area_um2` | F1 210, F2 200, F3 196, F4 194 pixels of 0.0625 µm² (13.125, 12.5, 12.25 and 12.125 µm²): the counts `ownership_map(…).owner` gives over each FOV's pixel centres for this placement (centres at (y, x) = (1.875, 1.875), (1.875, 5.125), (4.875, 1.875) and (4.875, 4.875) µm, ties to the first FOV in table order, T1), computed with the package's function by `results/check_expectations.py` of the algorithms specification's run directory; the test counts them with the same function and the report's `owned_area_um2` equals count × 0.0625; F4 `failed` gives null with `fov_not_placed`; a `placeholder` gives n/a. Exact. |
| DF6 | Cell density | `report_cells` (none) | `cell_density_per_mm2` | `n_cells_retained / owned_area_um2 × 10⁶` for each FOV (F1: 21 / 13.125 × 10⁶ = 1.6 × 10⁶); an area of 0 gives `undefined`. Exact in float64. |
| DF7 | Reads assigned to cells | `report_cells`, `report_cells` `mismatch` (none) | `reads_in_cells` per FOV and in total; the `summary` check | Per FOV, the retained `assigned` molecules of counted cells as the test counts them from `molecules.csv` and `cells.csv` (the molecules of the retained `excluded_no_nucleus` cell and of the `not_owner` cell left out); the total equals `Σ X` (two integers); with `mismatch` the two sums differ by 1, `reads_in_cells_equal` is false, and the report is written without an error. Exact. |
| DF8 | MECR | `hand_tables`: a count matrix of six cells and genes A, B, C, D, with A = (2, 1, 0, 0, 0, 0), B = (0, 3, 1, 1, 0, 0) and C = D = 0 (none) | `mecr.csv` | With threshold 1, pair (A, B): `cells_a` 2, `cells_b` 3, `both` 1, `either` 4, `mecr` 0.25; with threshold 2, pair (A, B): `cells_a` 1, `cells_b` 1, `both` 0, `either` 2, `mecr` 0.0; with threshold 1, pair (A, C): `cells_a` 2, `cells_b` 0, `both` 0, `either` 2, `mecr` 0.0; pair (C, D): four zeros and `mecr` null with `undefined`, the other pairs unaffected. Exact. |
| DF9 | Cell size and unit | `report_cells`; `report_cells` `nocal`; a sample with a volume and a plane FOV (none) | `cells.csv`; histograms | `size` equals `obs.size_physical` with `µm³` or `µm²` by the record's `size_unit`, `size_voxels` with `voxels` when NaN; a sample with two units gets two histograms in the order µm³, µm², voxels. Exact. |
| DF10 | Reads along Z | `report_cells` molecule `z` values, the test writing −0.6, −0.5, 0.4999, 0.5 and 3.5 on molecules of F1 and every other molecule's `z` in [−0.5, 3.5); `hand_tables` molecule `z` values on three FOVs whose grids have one plane: A (−1.2, 0.0, 0.3, 2.4, 2.6), B (2.0, 3.0), C none (none) | `reads_along_z.csv` | Planes `floor(z + 0.5)`: −0.6 → −1, −0.5 → 0, 0.4999 → 0, 0.5 → 1, 3.5 → 4, each counted on its plane, inside the grid of 4 planes or not; F1 has the 6 rows `z` −1 to 4, the rows −1 and 4 with 1 molecule each; F2, F3 and F4 each have the 4 rows `z` 0 to 3, with 0 where no molecule lies; on the one-plane grids, A has the 5 rows `z` −1 to 3 with 1, 2, 0, 1, 1 molecules, B the 4 rows 0 to 3 with 0, 0, 1, 1, and C the 1 row 0 with 0; each FOV's rows sum to its molecules of the table (A 5, B 2, C 0; the `report_cells` FOVs as the test counts them from `molecules.csv`); no cell is null. Exact. |
| DF11 | Intensity bins | `hand_tables` step outputs (none) | `intensity.csv` rows | uint8 with values 3 to 250: 248 bins of width 1 from 3; uint16 with values 0 to 1000: width `ceil(1001 / 256) = 4`; float32 over [0.25, 0.75]: 256 equal bins, the last closed; a constant float output: one bin; bool: two bins; each channel's counts sum to its voxels and the channels of one entry share their edges. Exact. |
| DF12 | Summary keys and other values | `report_records`, `report_cells` (none) | `report.json` `summary`; configuration check; acquisition; time per step | `spots_median_per_fov` 1000; `matched_fraction` 2800 / 4000 = 0.7; `fovs_succeeded` 4; the medians of `cells.csv` as the test computes them; the configuration check "identical for all FOVs", and "differs: 2 groups" naming `pipeline` when F2's `config.pipeline.spot_finding` is changed; acquisition "3 rounds × 4 channels, 3D"; each FOV's time per step name is the sum of its rows of that name. Exact. |
| PR1 | The rule form | `hand_tables` values (0.80, 0.80, 0.50, 0.80) (none) | Rows and limits | Low side, `k = 3`, floor 0.05: the third FOV only, limit 0.05, median 0.8; with `absolute = 0.6`: the third FOV only, `sample_median` n/a, limit 0.6; with `absolute = 0.5`: none (strict); with `k`, `floor`, `absolute` all `None`: not evaluated, one row `rule:<rule>` with `not_configured`, `report.json` `rules.<rule>.evaluated` false. Exact. |
| PR2 | One FOV, two FOVs, equal and undefined values | `hand_tables` (none) | Rows | One value: no mark with `k = 0.1` and floor 0; two values (0.2, 0.8) with `k = 1` and floor 0: no mark; median 0.5; the two absolute deviations from it, as the exact float64 values, `abs(0.2 − 0.5)` = 0.3 and `abs(0.8 − 0.5)` = 0.30000000000000004, and their median, the MAD, 0.30000000000000004 (the mean of the two); limit 0.44478 within 1e-12 absolute, since the float64 product `1 × 1.4826 × MAD` is 0.44478000000000006; equal values: no mark with `k = 0` and floor 0, and under an absolute limit 0.6 on the low side all four of four values 0.5 marked and none of four values 0.7; null values: not entered and not evaluated, no rule row; all null: one `rule:` row with the values' first reason. Exact, except the limit as stated. |
| PR3 | Shift residual per round and axis | `report_records` (none) | `problematic_fovs.csv` rows and their `measured`, `sample_median` and `limit` | Default limit: only (F4, round3, `x`); with `k = 0`, floor 0.5: (F2, round2, `x`), (F3, round2, `y`), (F3, round3, `z`), (F4, round3, `x`), in list order, then rounds, then axes; under both forms `measured` is the shift and `sample_median` the round's median shift. With the absolute limit `absolute = 0.5` (`k` and `floor` `None`): the same four rows, with `measured`, `sample_median` and `limit` (3, 2, 0.5), (2, 1, 0.5), (1, 0, 0.5) and (8, 0, 0.5), whose differences 1, 1, 1 and 8 are the residuals the limit is compared with; with `absolute = 8`: no row (strict). The rows exact; `measured`, `sample_median` and `limit` within 1e-9 absolute. |
| PR4 | Categorical rules | `report_records` with F2 `interrupted` and F3 `running`; `not_run`; `placement_states`; `report_cells`; registration entries with a rejected attempt followed by a succeeded fallback (none) | Rows | `fov_failed` marks F2 (interrupted) and not F3 (running); `fov_not_run` marks F4 when F1 to F3 have run records, and is not evaluated (`no_run_record`) when none has; the rejected-then-fallback round gives one `registration_rejected` and one `registration_fallback` row; `not_placed` marks F4 (`failed`); `stage_position_fallback` marks F3 (`fallback`) and no FOV of the `supplied` copy; `reads_no_cell` marks F2. Exact. |
| PR5 | Spots close to zero | `hand_tables` spots (1000, 1200, 40, 1000); (0, 0, 0) (none) | Rows | The third FOV (40 < 0.05 × 1000 = 50); with every count 0 the limit is 0 and nothing is marked. Exact. |
| EF1 | Automatic example choice | `report_records`; `report_cells`; a 10-FOV table with `passed_filter` 100, 200, …, 1000 (none) | `report.json` `examples`; `fovs.csv` `example` | Before a segmentation: `example_1` F1 (median 645, distances 5, 155, 145, 5, the tie of F1 and F4 broken by list order, `q = 1`), `example_2` F3 (`fewest_reads`, 500), `example_3` F2 (`most_reads`, 800); with cells: `example_2` F2 (`fewest_cells`, 0), `example_3` F4 (`most_cells`, 14); for the 10-FOV table `N` holds the FOVs of 500 and 600 (`q = 2`) and `example_1` is `N[default_rng(0).integers(2)]`, recorded with `seed` 0. Exact. |
| EF2 | Coinciding, named and unavailable examples | `report_records` `one`; a two-FOV copy; `example_fovs` given; `scene_no_run_json` (none; 42 for the scene) | Roles and rows | One FOV: three roles on F1; two FOVs: the third role on the FOV its rule gives over both; named: the roles in the given order with `chosen_by` `configuration`; no read count: the choice unavailable with `no_run_record`, the example parts each named by a row. Exact. |
| RP1 | The replay against the saved registered image | `scene_chain_registered`, FOV `FOV_001` (42) | Largest absolute difference per round between the replayed image and `dataset.load_checkpoint(…, "registered")` | 0 in every round (bit-identical, every voxel compared, as the source-inventory run's `replay.csv`): the two rounds the report replays, and for the test all four; the replayed fitted values equal the recorded ones. Exact. |
| RP2 | Replay failures | `scene_chain_registered` copies: one recorded `fitted` maximum raised by 1 in `run.json`; one input byte changed; one input removed; a recipe with a `demons` step and a `post_registration` entry run without the stage `registered` (42) | `report.json` `replay`; rows | `replay_differs`; `input_changed`; `input_missing`; `field_not_saved` for the Registration example and, for the moving round, the two rows of the `post_registration` outputs with `field_not_saved`, the round's `input` and `steps` rows of `intensity.csv` written. Exact. |
| RP3 | The two sources of the transforms | `scene_chain_registered` and its copy with R1 removed (42) | Replayed images | The replayed images equal bit for bit, nothing left out; `transform_source` is `run.json` for the first and `transforms.json` for the copy (a field the comparison of images does not read). Exact. |
| PG1 | Embedded data equal the saved tables | every page the RC, CH, RT, Z1 and A2 checks write | The data block parsed with `json`, read back with the dtypes `report.json` lists | Each embedded table equals its CSV cell by cell, every column and row compared, null for a missing value; `cells` and `intensity` are not embedded. Exact. |
| PG2 | Every section and block present | as PG1 | Element ids of `report.html` | The eight sections in the order of "Layout of the page" and, in each, the blocks listed there in order (each id written in the test from this page). Exact. |
| PG3 | Every unavailable part carries its reason | as PG1 | Placeholders and tooltips | For each row of `unavailable.csv` with `table` null, the page has, in the part's place, a placeholder with "Unavailable:" and the reason's text of "How an unavailable part is shown"; each null cell named by a row has "—" and its reason in the cell's tooltip data; the reason texts of the template's script equal that table, and the page holds no `application/json` block other than `report-data`. Exact. |
| PG4 | No external resource | as PG1, and `index.html` of CP1 | The text of each page | No match of `https?:`, of `(src\|href)="//`, of `<link`, `<iframe`, `<object` or `@import`; every `src` attribute starts with `data:`; every `url(` is followed by `data:`; every `href` is `#…` or, on `index.html`, a relative `<sample>/report.html`; no `prefers-color-scheme` rule (light theme only). Exact. |
| PG5 | Size budget | as PG1 | Bytes of `report.html` | At most `6 MiB + 16 KiB × FOVs + 3 MiB × marker figures`, and equal to the bytes `report.json` records for it (both taken from the same written file); every static figure decodes and has the pixel size of "Figure sizes". The rules of "Figure sizes" (Jiahao's first review of the real block report, 2026-10-14), on the figures as drawn, of these pages, the comparison page, a hand-written sample of 64 FOVs and hand-written examples: no text below 11 px as shown, each PNG twice the width of its slot and the slots those of the page's CSS and table, no text over another, another axes or the edge, heatmaps of at most 64 labelled rows of at least 14 px as shown chosen by the rule of "Registration" with the tables unsampled, white gaps of at least 8 px as shown between tiles, the yellow marks of the zoomed quarter and of the regions with their numbers, readout labels never clipped to their bar, and the Summary's Version, Run time and Generated in UTC and in the viewer's time zone. After the second review (2026-10-14): the page header is the heading "Processing report" alone, with the dataset and the sample in the Summary's facts and the title unchanged; the axes of Time per step, as shown, start within 4 px of the top of the FOV rectangles of the map the page's script draws (FOVs on grids of 2 × 2, 8 × 8 and 1 × 4 with the step names of a workflow run record) and end within 4 px of their bottom where 26 px per step name fit, else are at least 26 px per step name high, and a sample without placement draws the figure as before; no figure of `report.html` or `index.html` is shown wider than two thirds of its drawn width; every legend is a row below every axes and tile of its figure, at its left margin, over nothing; the table of figure sizes equals the code's constants. Exact; a width as shown compared as a float, an alignment within 4 px. |
| PG6 | No verdict and no score | as PG1 | Column names of every table and keys of `report.json` | Equal to the lists of {doc}`report-contract` exactly, as sets and in order: no added column or key, so none holds a verdict or a score; `alignment_score` is the quantity of settled decision 5. Exact. |
| PG7 | Output folder and entry checks | `report_records` (none) | Raised errors; the folder | Checks 1 to 7 of {doc}`report-contract` each raise their error, in that order, with the folder unchanged (a marker gene not in the codebook among them); a second report replaces the first; a file the first `report.json` does not list stops the entry with `FileExistsError` before anything is written. Exact. |
| CH1 | The chain after the per-FOV processing | `scene_chain` with the export and assignment folders removed, once also without the placement (42) | Parts present and unavailable | As the first two rows of {doc}`report-contract` ("The report at each stage of a sample's processing"): without the placement the maps `no_placement`; with it the maps and the placement block; Segmentation and assembly and Sample measurements `no_export`; RC12's condition holds. Exact. |
| CH2 | The chain after the assignment | `scene_chain` without the export (42) | Parts | The third row of that table: the segmentation examples and the per-FOV lane's segmentation and assignment present; cells, accounting and per-cell values `no_export`. Exact. |
| CH3 | The chain after the export | `scene_chain` (42), with two marker groups and two pairs of the scene's genes | Every part; the reads check | Every part present except those the configuration lacks; `reads_in_cells` equals `Σ X` of `sample.h5ad` and the molecule count of identity 4 of [assembly specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/assembly.md); the present parts of CH1 ⊆ CH2 ⊆ CH3; each later report replaced the earlier in the same folder. Exact. |
| RT1 | Segmentation and assembly on both routes | `scene_chain_plane`, both exports (42) | `fovs.csv`, `population.csv`, the example tiles | The same columns on both routes; after fusion `n_cells == n_cells_retained` and each equals the count of `cells.csv` rows whose `sample_ownership_centroid_*` the FOV owns under `ownership_map`; the duplicate rows n/a; each example tile's labels equal the fused label image's values over the footprint's pixel range; density by the same formula on both. Exact. |
| RT2 | Sample measurements on both routes | `scene_chain_plane`, both exports (42) | `cells.csv`, `mecr.csv`, `reads_along_z.csv`, the marker figures | `cells.csv` equals the rows of `X` on each route, every row compared; MECR as DF8 recomputed from `X` by the test; the marker figures drawn (decoded, sized) on both. Exact. |
| Z1 | `Z = 1` | `scene_chain_plane`, route per FOV (42) | Registration, reads along Z, marker zooms, acquisition | `shift_z` and `residual_z` 0 on every row; `reads_along_z.csv` one row per FOV, plane 0; three rows `view=xz` with `no_stack` for the XZ zooms; the XY zooms drawn; acquisition "2D (Z = 1)"; PG1 to PG5 hold. Exact. |
| A2 | An export and no `run.json` | `scene_no_run_json`, both roots, with and without `example_fovs` (42) | The report; parts; rows | The report is written; every column up to the readout null with `no_run_record` (one row each); Segmentation and assembly and Sample measurements present with the export's values (cells on the first root, `no_segmentation` on the `molecules_only` root); without `example_fovs` the example parts name `no_run_record`, with them the zoom regions are the named FOVs. Exact. |
| CP1 | The comparison page | `scene_two_samples` (42) | `samples.csv`, `index.html`, errors | Two rows: `s1` present with the values of its `report.json` (every column of the row compared), `s2` `missing` with every other column null; the box plots' embedded values equal `s1`'s `fovs.csv` columns; a table of `s1` changed after its report raises `ValueError` naming the file; nothing under `s1/` is written. Exact. |
| RR1 | The records added to `run.json` | the golden test's synthetic FOV ({doc}`report-contract`, "The tests that pin `run.json`") with stages `candidates` and `pre_qc`, and with `registered` too; recipes with an affine step and with a `demons` step (the golden FOV's own seeded background) | `run.json` keys | `transforms.<round>.results[*]` equal, whole and with nothing left out, the entries `_transform_json(result, i, field_name)` of the run's results with the field name R1 writes (null without `registered`, `<round>_field.npz` with it); `codebook` path and SHA-256 of the file; `counts.filtering.endpoint_valid` equals the filter's `endpoint_valid_count`; `workflow_config` null outside the workflow and the file's path and SHA-256 from a workflow run; a workflow run writes `run.json` and a rerun replaces it (R5). Exact. |
| RR2 | An older record still loads | the golden test's pinned `run.json` form (none) | The report on it | Loads; the values of R1 to R4 null with `not_recorded` (as RC11); `test_report_records_golden.py` passes unchanged. Exact. |
| WF1 | The workflow block and rules | the repository's test configurations with and without a `report` block, through a stub `snakemake` object (none) | Schema, adapter, rules | The block's keys equal the fields of `ReportConfig` (`MarkerGroup`, `RuleLimit`, `ReportLimits`); an unknown key raises naming it; `report: {}` gives `ReportConfig()`; the rules' declared inputs and outputs are those of {doc}`report-contract` (WR1); a configuration without the block has no new rule. Exact. |
| DT1 | Determinism | `report_cells` and `scene_chain`, each built once; the report made by two single-thread processes, one after the other, into the same default output folder, the first report's files copied aside before the second runs (none; 42) | Bytes of every table; `report.json`; `report.html` | Every table byte-identical. `report.json` identical after leaving out `generated_at`, `wall_s`, `cpu_s`, `peak_rss_kib` and the `files` entry of `report.html` (its `sha256` and `bytes`, which depend on the generation time the page shows). `report.html` identical after leaving out the text of the Summary's Generated fact (`generated_at`). No field left in depends on one left out: the tables, the other `files` entries, `inputs`, `summary`, `rules`, `examples`, `replay` and `unavailable` are computed from the records alone, and the page's data blocks hold no time. Exact. |

% W-386, W-384, W-475, W-478, D5

### What the default tier covers and what it does not

RC1 to RC13, DF1 to DF12, PR1 to PR5, EF1, EF2 (without its `scene_no_run_json` part),
PG1 to PG7 on the hand-written pages, RR1, RR2 and WF1 run in the routine tier on
hand-written records, except PG7's entry on `report_cells` (5.4 s) and DT1 on
`report_cells` (6.4 s, two processes), which are `slow` by the 5-s rule (the test-tier rule of 2026-10-20). RP1 to
RP3, CH1 to CH3, RT1, RT2, Z1, A2, CP1 and DT1 on `scene_chain` run the synthetic chain and
are `slow`. The cases with an export need
the `anndata` extra; the chains also need `multiview-stitcher` for the fused image; RP2's
`demons` step needs `local-registration`. Nothing here measures how well a sample was
processed, and nothing shows that a marker pattern is biologically right.

% W-530

### Resource plan

The hand-written fixtures are a few kilobytes each and every routine check takes well under a
second. The synthetic chain of `sample-tiny-v1` took 48 to 84 s wall and 22 to 32 s CPU at one
thread per run, export with the SpatialData store included, within 4 GiB (the source-inventory run's
`results/tables/resources.csv`, rows `synthetic_cp` and `synthetic_reg`); the replay of one
synthetic FOV took 0.8 to 1.2 s. Each chain fixture is built once per module (a module-scoped
fixture) and every `slow` test stays within one CPU, 4 GiB and 1800 s. Each run records wall
time and peak RSS with `/usr/bin/time -v`.

% W-384

### Implemented checks

% Implemented checks (W-459)

The checks are in the modules of `src/python/test/` listed below. The implementation issues that delivered
the report added most of them with their code; `test_report_chain.py` adds
CH1 to CH3, RT1 and RT2 through `Dataset.report`, Z1, A2, CP1 on the chain, DT1 on
`scene_chain`, PG1 to PG6 with RC12 on every page it writes, DF12's Z size from a reference
image when no record holds a grid (at the place of the `Dataset`'s `reference_image`
pattern), and the test that keeps this table true, `test_every_check_names_collected_tests`.
That test reads the rows of the table "Checks" from this page and checks that this table has exactly those rows, in that order;
that every row names at least one test of its module that pytest collects (a name ending in
`*` stands for every test whose name starts so); that none of them is marked as an expected
failure; and that a row whose tier is `default` names no `slow` test, a row whose tier is
`slow` names only `slow` tests, and a row whose tier is `default and slow` names both. A test
of a module that is not named here is an additional check of the code it tests.

% W-451 to W-458

Every tolerance is the one of the table "Checks", unchanged. Where a row says "Exact" for a
number that is not an integer, its tests compare it within an absolute 1e-9 (the project's
default float tolerance), or within the bound the row itself states; the column "Floats" says
which. Counts, strings, reasons, row orders, dtypes, hashes, images and JSON structure are
compared exactly. The tests that write or read an export skip without the `anndata` extra, and
the chains also without `multiview-stitcher`, naming the extra; RP2's `demons` step needs
`local-registration`. The `slow` tests of the report carry the subsystem marker `report` and
run in `pytest-slow-other` (`slow and not learned and not (benchmark or synthetic or
registration or spot_finding)`); the default tier runs the others.

The chain checks read the table "The report at each stage of a sample's processing" of
{doc}`report-contract` part by part, with the reasons of "Parts, their inputs and their
reasons": the same `Dataset.report` call (the default `assignment`) reports each stage into
the sample's default folder, so before the assignment exists the segmentation examples, which
read the assignment folder on the route per FOV, are `input_missing`. A part counts as
present at a stage when no row of `unavailable.csv` names it with a reason other than
`not_configured` and `not_recorded`.

(report-algorithms-implemented-checks)=

| # | Module and tests | Tier | Floats |
| --- | --- | --- | --- |
| RC1 | `test_report_collection.py`, `test_rc1_normal_sample`; `test_report_values.py`, `test_df1_rc1_fractions_and_totals`, `test_rc1_residuals`, `test_df4_rc1_scores_and_a_null_qc`; `test_report_rules.py`, `test_rc1_rows`; `test_report_page.py`, `test_pg1_embedded_data_equal_the_tables` | default | within 1e-9 (the fractions); the dyadic tables exact |
| RC2 | `test_report_collection.py`, `test_rc2_failed_fov`; `test_report_values.py`, `test_rc2_rc13_round3_medians_over_the_succeeded_fovs`; `test_report_rules.py`, `test_pr4_rc2_failed_reason` | default | within 1e-9 (the medians) |
| RC3 | `test_report_collection.py`, `test_rc3_fov_not_run`; `test_report_rules.py`, `test_pr4_fov_not_run`, `test_pr4_no_run_record_at_all` | default | exact |
| RC4 | `test_report_collection.py`, `test_rc4_stage_not_run`; `test_report_page.py`, `test_pg3_every_unavailable_part_carries_its_reason` | default | none compared |
| RC5 | `test_report_collection.py`, `test_rc5_no_placement`; `test_report_rules.py`, `test_rc5_placement_rules_not_evaluated` | default | none compared |
| RC6 | `test_report_measurements.py`, `test_rc6_no_markers` | default | none compared |
| RC7 | `test_report_measurements.py`, `test_df9_rc7_cells_and_the_unknown_calibration` | default | exact |
| RC8 | `test_report_values.py`, `test_rc8_fov_with_no_spot`; `test_report_rules.py`, `test_rc8_rc9_rc10` | default | within 1e-9; the limit 45 of `spots_near_zero` with `pytest.approx`'s default (relative 1e-6) in `test_rc8_rc9_rc10` |
| RC9 | `test_report_values.py`, `test_ef2_rc9_one_fov`; `test_report_rules.py`, `test_rc8_rc9_rc10` | default | exact |
| RC10 | `test_report_values.py`, `test_rc10_equal_values`; `test_report_rules.py`, `test_rc8_rc9_rc10` | default | exact |
| RC11 | `test_report_collection.py`, `test_rc11_older_record` | default | exact |
| RC12 | `test_report_collection.py`, `test_rc12_null_cells_and_closed_lists`, `test_closed_lists_equal_the_contract`; `test_report_chain.py`, `test_ch1_ch2_ch3_the_report_at_each_stage`, `test_rt1_segmentation_and_assembly_on_both_routes`, `test_a2_an_export_and_no_run_json`, `test_cp1_the_comparison_page` | default and slow | none compared |
| RC13 | `test_report_collection.py`, `test_rc13_round_that_did_not_finish`; `test_report_values.py`, `test_rc2_rc13_round3_medians_over_the_succeeded_fovs` | default | within 1e-9 (as stated) |
| DF1 | `test_report_values.py`, `test_df1_fraction_definition`, `test_df1_rc1_fractions_and_totals`, `test_df1_null_count_gives_its_reason` | default | within 1e-9 |
| DF2 | `test_report_values.py`, `test_df2_shift_of_the_global_steps`, `test_df2_affine_and_dense_through_the_collection` | default | within 1e-12 (as stated) |
| DF3 | `test_report_values.py`, `test_df3_residual_definition`, `test_rc1_residuals` | default | exact |
| DF4 | `test_report_values.py`, `test_df4_alignment_score`, `test_df4_rc1_scores_and_a_null_qc` | default | exact |
| DF5 | `test_report_values.py`, `test_df5_df6_area_and_density`, `test_df5_df6_unplaced_placeholder_and_zero_area` | default | within 1e-9 |
| DF6 | `test_report_values.py`, `test_df5_df6_area_and_density`, `test_df5_df6_unplaced_placeholder_and_zero_area` | default | within 1e-9 |
| DF7 | `test_report_measurements.py`, `test_df7_reads_assigned_to_cells`, `test_reads_in_cells_without_the_inputs` | default | within 1e-9 (the fraction) |
| DF8 | `test_report_measurements.py`, `test_df8_mecr_hand_tables`, `test_df8_mecr_report_cells` | default | exact |
| DF9 | `test_report_measurements.py`, `test_df9_rc7_cells_and_the_unknown_calibration`, `test_df9_a_volume_and_a_plane_fov`, `test_size_unit_of_the_records` | default | exact |
| DF10 | `test_report_measurements.py`, `test_df10_reads_along_z_report_cells`, `test_df10_reads_along_z_hand_tables` | default | none compared |
| DF11 | `test_report_replay.py`, `test_df11_intensity_bins` | default | within 1e-9 (the bin edges) |
| DF12 | `test_report_values.py`, `test_df12_summary_keys`, `test_df12_cells_median_per_fov`, `test_df12_configuration_acquisition_software_and_time`, `test_df12_acquisition_from_the_export`; `test_report_chain.py`, `test_z_size_*` | default | within 1e-9 |
| PR1 | `test_report_rules.py`, `test_pr1_rule_form`, `test_pr1_rows_and_records` | default | within 1e-9 |
| PR2 | `test_report_rules.py`, `test_pr2_edge_cases`, `test_pr2_null_values` | default | the limit within 1e-12 (as stated) |
| PR3 | `test_report_rules.py`, `test_pr3_shift_residual` | default | within 1e-9 (as stated) |
| PR4 | `test_report_rules.py`, `test_pr4_failed_and_running`, `test_pr4_fov_not_run`, `test_pr4_no_run_record_at_all`, `test_pr4_rejected_then_fallback`, `test_pr4_placement_rules`, `test_pr4_reads_no_cell` | default | none compared |
| PR5 | `test_report_rules.py`, `test_pr5_spots_near_zero` | default | exact |
| EF1 | `test_report_values.py`, `test_ef1_automatic_choice`, `test_ef1_ten_fovs` | default | exact |
| EF2 | `test_report_values.py`, `test_ef2_rc9_one_fov`, `test_ef2_two_fovs`, `test_ef2_named_examples`, `test_ef2_no_read_count`; `test_report_chain.py`, `test_a2_an_export_and_no_run_json` | default and slow | none compared |
| RP1 | `test_report_replay.py`, `test_rp1_replay_equals_the_saved_registered_image` | slow | exact (bit-identical) |
| RP2 | `test_report_replay.py`, `test_rp2_replay_failures`, `test_rp2_field_not_saved` | slow | none compared |
| RP3 | `test_report_replay.py`, `test_rp3_two_sources_of_the_transforms` | slow | exact (bit-identical) |
| PG1 | `test_report_page.py`, `test_pg1_embedded_data_equal_the_tables`, `test_page_with_every_figure`; `test_report_comparison.py`, `test_page_loads_nothing_and_embeds_samples_csv`; `test_report_chain.py`, `test_ch1_ch2_ch3_the_report_at_each_stage`, `test_rt1_segmentation_and_assembly_on_both_routes`, `test_a2_an_export_and_no_run_json`, `test_cp1_the_comparison_page` | default and slow | within 1e-9 |
| PG2 | `test_report_page.py`, `test_pg2_sections_and_blocks`, `test_page_with_every_figure`; `test_report_chain.py`, `test_ch1_ch2_ch3_the_report_at_each_stage`, `test_rt1_segmentation_and_assembly_on_both_routes`, `test_a2_an_export_and_no_run_json`, `test_cp1_the_comparison_page` | default and slow | none compared |
| PG3 | `test_report_page.py`, `test_pg3_every_unavailable_part_carries_its_reason`, `test_pg3_cells_drawn_by_the_script`, `test_page_with_every_figure`; `test_report_chain.py`, `test_ch1_ch2_ch3_the_report_at_each_stage`, `test_rt1_segmentation_and_assembly_on_both_routes`, `test_a2_an_export_and_no_run_json`, `test_cp1_the_comparison_page` | default and slow | none compared |
| PG4 | `test_report_page.py`, `test_pg4_no_external_resource`, `test_page_with_every_figure`; `test_report_comparison.py`, `test_page_loads_nothing_and_embeds_samples_csv`; `test_report_chain.py`, `test_ch1_ch2_ch3_the_report_at_each_stage`, `test_rt1_segmentation_and_assembly_on_both_routes`, `test_a2_an_export_and_no_run_json`, `test_cp1_the_comparison_page` | default and slow | none compared |
| PG5 | `test_report_page.py`, `test_pg5_size_budget_and_figure_sizes`, `test_page_with_every_figure`; `test_report_chain.py`, `test_ch1_ch2_ch3_the_report_at_each_stage`, `test_rt1_segmentation_and_assembly_on_both_routes`, `test_rt2_sample_measurements_on_both_routes`, `test_a2_an_export_and_no_run_json`, `test_cp1_the_comparison_page`; `test_report_figure_rules.py`, `test_text_size_and_slots_on_every_page`, `test_no_overlap_on_the_pages`, `test_slots_are_the_page_columns_and_the_table`, `test_a_row_of_two_and_one_figure`, `test_comparison_page_figures`, `test_heatmap_rows`, `test_heatmap_tables_are_not_sampled`, `test_a_hand_written_sample_of_64_fovs`, `test_intensity_panels_do_not_overlap`, `test_time_per_step_with_the_steps_of_a_run_record`, `test_preprocessing_grid_gaps`, `test_registration_example_gaps_and_zoom_mark`, `test_segmentation_examples_gaps`, `test_whole_view_regions`, `test_readout_bar_labels_are_not_clipped`, `test_summary_rows`, `test_summary_text_forms`, `test_report_json_does_not_depend_on_the_page`; `test_report_second_review.py`, `test_header`, `test_time_per_step_aligned_with_a_hand_written_map`, `test_time_per_step_aligned_on_the_page`, `test_time_per_step_without_placement`, `test_no_figure_shown_enlarged_on_every_page`, `test_preprocessing_grid_is_drawn_larger`, `test_comparison_page_not_enlarged`, `test_legends_of_the_hand_written_figures`, `test_legend_of_the_whole_sample_view`, `test_figures_table_equals_the_constants`, `test_second_review_on_the_chain_page` | default and slow | within 4 px (the alignment of Time per step), 1e-3 px (the map the script draws against `map_extent`); the ratios as floats |
| PG6 | `test_report_page.py`, `test_pg6_no_verdict_and_no_score`, `test_page_with_every_figure`; `test_report_collection.py`, `test_pg6_schema`; `test_report_chain.py`, `test_ch1_ch2_ch3_the_report_at_each_stage`, `test_rt1_segmentation_and_assembly_on_both_routes`, `test_a2_an_export_and_no_run_json`, `test_cp1_the_comparison_page` | default and slow | none compared |
| PG7 | `test_report_collection.py`, `test_pg7_*`, `test_read_report_checks_the_listed_files`; `test_report_page.py`, `test_pg7_checks_through_dataset_report`, `test_pg7_entry_on_the_dataset` | default and slow | none compared |
| CH1 | `test_report_chain.py`, `test_ch1_ch2_ch3_the_report_at_each_stage` | slow | none compared |
| CH2 | `test_report_chain.py`, `test_ch1_ch2_ch3_the_report_at_each_stage` | slow | none compared |
| CH3 | `test_report_chain.py`, `test_ch1_ch2_ch3_the_report_at_each_stage`, `test_ch3_the_reads_assigned_to_cells`; `test_report_measurements.py`, `test_markers_on_the_stack` | slow | exact (the tables); the marker figures' geometry within 1e-9 |
| RT1 | `test_report_chain.py`, `test_rt1_segmentation_and_assembly_on_both_routes`; `test_report_replay.py`, `test_rt1_segmentation_previews` | slow | within 1e-9 (the owned area and the density) |
| RT2 | `test_report_chain.py`, `test_rt2_sample_measurements_on_both_routes`; `test_report_measurements.py`, `test_rt2_z1_sample_measurements_on_both_routes` | slow | exact (the tables); the marker figures' geometry within 1e-9 |
| Z1 | `test_report_chain.py`, `test_z1_one_plane`, `test_rt1_segmentation_and_assembly_on_both_routes`; `test_report_measurements.py`, `test_rt2_z1_sample_measurements_on_both_routes` | slow | within 1e-9 (`shift_z`, `residual_z`) |
| A2 | `test_report_chain.py`, `test_a2_an_export_and_no_run_json` | slow | exact |
| CP1 | `test_report_chain.py`, `test_cp1_the_comparison_page`; `test_report_comparison.py`, `test_samples_csv_copies_each_report`, `test_box_plots`, `test_a_changed_report_raises_naming_the_file`, `test_reads_the_reports_only` | default and slow | within 1e-9 |
| RR1 | `test_report_run_record_additions.py`, `test_r1_*`, `test_r2_*`, `test_r4_r5_the_rule_writes_run_json_alone_and_no_log`, `test_r5_*` | default | none compared |
| RR2 | `test_report_records_golden.py`, `test_run_record_holds_every_pinned_key_and_value`, `test_checkpoint_folder_holds_every_pinned_file`, `test_the_comparison_accepts_additions_and_finds_removals_and_changes`; `test_report_run_record_additions.py`, `test_an_older_record_and_the_format_version`; `test_report_collection.py`, `test_rc11_older_record` | default | exact |
| WF1 | `test_report_workflow.py`, `test_wf1_*` | default | none compared |
| DT1 | `test_report_collection.py`, `test_dt1_two_single_thread_processes`; `test_report_chain.py`, `test_dt1_the_chain_in_two_single_thread_processes` | slow | exact (bytes) |

### Tour notebook

% Tour notebook (W-460)

The tour notebook's issue of 2026-10-13 added an executed introductory notebook on the report, on `sample-tiny-v1` and
`scene_plane`. It was retired on 2026-10-26 with the other six tours; Git history keeps it. The
[multi-FOV / sample demo](get-started/multi-fov-demo.md) now shows `Dataset.report` after the
per-FOV runs, the stitching and the export of `sample-tiny-v1`, with the report's key numbers
and its FOV table. Like the tour, it is an introduction, not a check of this design.

% W-460, W-587

## The real example

The real example runs as a batch of its own after the implementation (settled decision 7), under its own
authorization for real data and resources, with records as the project guidance defines
them. It has two parts. Neither compares methods or claims accuracy: the reports describe
the runs, and Jiahao inspects them.

% D7

### Part 1: LN sample1 through the Python per-FOV pipeline

**Inputs** (read only):

* Raw images: `<data root>/Processed/sample-dataset/LN/`, the 64 FOVs
  Position001 to Position064 of `round1` to `round4` (sequencing `ch00` to `ch03`; DAPI is
  `round4` `ch04`), each `(50, 1496, 1496)` uint8, 1,024 sequencing files of the 1,280
  (the inputs survey's `inputs.md`, section 4).
* Codebook: `genes.csv` of the same folder (SHA-256 `2131fd29…5e786`), 61 genes.
* Configuration: the source-inventory run's `results/real/ln_config.yaml`, the earlier pipeline's settings that
  Jiahao chose on 2026-10-08 (min–max normalization only; global registration only with
  reference `round4`; local maxima at 0.2 of the channel maximum, on `round4`, the nearest
  setting to the earlier `round1`; extraction 1 × 1 × 1; one segment `A…C`; channel order
  `ch00, ch02, ch01, ch03`), with `rotate_angle: -90` (below), the checkpoint stages
  `candidates` and `pre_qc` through `rules.rsf_single_fov.parameters.checkpoints` (R5 of
  {doc}`report-contract`), and a `report` block holding the inputs survey's `markers.yaml` (two groups,
  four pairs).
* Placement: the LN BigStitcher import of the 2024 `dataset.xml`, as the first stitching run made it and
  the saved-files run and the own-folder run reused it ([stitching and assembly validation design](https://github.com/wanglab-broad/starfinder/blob/dev/design/validation/stitching-assembly.md), "Results of the real
  examples"), saved under the run's `stitching/ln_sample1/`.
* Cells: the block segmentation's nucleus labels of the block Position048, Position049, Position057 and
  Position058 (`results/checkpoints/ln_sample1/<fov>/segmentation/nucleus_dapi/labels.tif` in its run directory),
  imported with the segmentation mask import and assigned with the settings of the block segmentation's
  `expanded_nuclei` assignment (its `assignment.json`). No new segmentation and no model.
* Fused DAPI: `dapi=file:images/flamingo/DAPI/{fov_id}.tif` of the earlier analysis root
  `D/Analyzed/2024-03-08-Hongyu-Covid_LN/`, as the own-folder run fused it.

% /home/unix/jiahao/wanglab/Data/Processed/sample-dataset/LN/, W-383, W-384, W-376, W-391, W-398, W-382, W-382/20261008T023258Z-e2618af4/results/checkpoints/ln_sample1/<fov>/segmentation/nucleus_dapi/labels.tif, §2.9

**The frame.** The source-inventory run found that a Python run with `rotate_angle: 0` lies in the raw TIFF
frame, and that the block segmentation's labels, the earlier molecule tables and the 2024 placement lie in
that frame rotated by −90° in YX: under `np.rot90(k=-1)`, 80.7 % and 80.8 % of the Python
reads of Position048 and Position049 lie within 3 voxels of an earlier read
(`results/frame/frame_dihedral.csv`; section 9 of its notes). The plan therefore runs with
`rotate_angle: -90` and confirms it on the first two FOVs before the others.

% W-384, W-382

**Steps**, each in its own process under `/usr/bin/time -v` and a watchdog:

| # | Step | What it writes | Stop rule |
| --- | --- | --- | --- |
| 1 | Inventory of the inputs of both parts, before step 2: path, size and SHA-256 of every file a later step reads (the 1,024 raw images, the codebook, the configuration and `markers.yaml`, the placement import, the block segmentation's labels and assignment record, the 64 DAPI source images of step 6, and the files of Part 2's exports that a report reads: `export.json`, `sample.h5ad`, `tables/` and the fused image); a file found later to have been read is added with the time of the addition. The SHA-256 that `hash_inputs` writes into `run.json` are a second record, not a replacement | `results/inventory/` | — |
| 2 | Position048 and Position049 through the workflow's processing rule, side by side; then the frame check of the source-inventory run on their reads | two FOVs' records (kept: they are two of the 64) | Fewer than 50 % of a FOV's reads within 3 voxels of an earlier read, or a share of reads inside a nucleus label of the block segmentation more than 0.02 from the earlier table's (0.129, 0.125; the source-inventory run's `frame_check.csv`): stop before step 3 and report |
| 3 | The other 62 FOVs, three at a time | 64 run records and the rule outputs | A FOV that reaches a limit is `resource_stop`, not retried larger, and is a failed FOV of the report |
| 4 | **Report after the per-FOV processing**: `Dataset.report("ln_sample1", <the 64 FOVs>, assignment=None, config=<the report block>)`, with the run records and the placement | `report/ln_sample1/` | — |
| 5 | The block: the four FOVs' labels imported and assigned | `fovs/<fov>/segmentation/`, `assignment/expanded_nuclei/` | — |
| 6 | The DAPI images copied uncompressed with axes ZYX to local scratch (below); then the exports: `ln_sample1`, `molecules_only` (`assignment=None`), `dimensionality: stack`, `scale_yx` 0.5, the DAPI image read from the copies; `ln_sample1_block`, the block's four placement rows, `assignment="expanded_nuclei"`, route per FOV, the same image; then the copies deleted | `export/ln_sample1/`, `export/ln_sample1_block/` | Less than 10 GB free on the scratch disk before the copy: no copy is made and the fusion reads the images from the share (route B below, 5,903 s, the own-folder run); this is not a stop |
| 7 | **Reports after the export**: `ln_sample1` (replacing the report of step 4) and `ln_sample1_block`; then `Dataset.compare_samples(["ln_sample1", "ln_sample1_block"])` | `report/` | — |
| 8 | The manifest, last | `results/manifest.json` | — |

% W-382, W-384, W-398

The report of step 4 is kept as a copy (`report_after_fovs/`) before step 7 replaces it, so
Jiahao sees both stages.

**What Jiahao inspects:**

* After the per-FOV processing (step 4): the FOV status and the time per step; the shift maps
  and the heatmap, against the earlier logs' Z residual of 17 to 19 voxels at Position001
  (the source-inventory run's notes, section 7); the distribution of `ncc_after` over the 64 FOVs and three moving
  rounds, from which the alignment-score limit can be set; the readout totals and the
  per-FOV fractions beside the source-inventory run's `LN_sample1_reads_per_fov.csv` of the earlier logs (two
  definitions, read side by side, not compared as one quantity); the Problematic FOVs table
  under the provisional limits; the replayed preprocessing and registration examples.
* After the export (step 7): the placement block; the marker patterns of the two groups on
  the fused DAPI stack, the zoom-in regions of the three example FOVs and their XZ views,
  with the point size and slab of choice A6; the reads along Z; the population accounting;
  on the block, the segmentation outlines over the reference image in the Python frame,
  cells and density per FOV, the per-cell distributions, MECR of the four pairs of
  `markers.yaml`, and the reads assigned to cells against `sample.h5ad`.
* The page: opens offline in a browser of Jiahao's choice; its bytes against the budget; the
  time and memory each report took.

% W-384

**Resources and the bounds.** Jiahao decided the bounds on 2026-10-12, with the source-inventory run's
measurement before him: the 64 FOVs run **three at a time, each FOV's process under a limit
of 32 GiB RSS**, about 68 GiB on the host together, and 20 GiB of new artifacts. The source-inventory run
measured a complete `FOV.run` of an LN FOV at 618 to 634 s wall and 22.5 GiB peak RSS with
the registration QC as it is (`results/tables/real_fovs.csv`); three at once are 67.5 GiB,
and `sizing_64.csv` gives 3.82 h for the 64 FOVs three at a time. Reducing the memory of the
registration QC (the source-inventory run: 6.4 GiB per FOV without it) was not chosen; it stays possible as
later work outside the report. **The time bound** is **9 hours** for the whole batch, Part 1 and
Part 2 together, as Jiahao decided on 2026-10-12; both routes below are within it, the slower
in 7.9 h.

% W-384, §2.11

**Two routes for the fused DAPI stack.** Route A fuses from uncompressed local copies, as
the first stitching run did in 1,468 s (the own-folder run's notes, the table of fusion runs); route B fuses from the share,
as the own-folder run did in 5,903 s, and is taken when step 6 finds less than 10 GB free on the scratch
disk (not a stop). What route A copies: the pixels of the 64 files
`D/Analyzed/2024-03-08-Hongyu-Covid_LN/images/flamingo/DAPI/Position<NNN>.tif` (4,721,387,720
bytes as saved, the own-folder run), each written uncompressed with axes ZYX by the method of the first stitching run's
`scripts/zyx_copies.py`, with each copy's pixel SHA-256 checked equal to the original's; in
the first stitching run the 64 copies took 7,162,193,280 bytes (6.7 GiB) and 129.5 s. Where: the batch's local
scratch folder `/tmp/linear-runner/<batch>/zyx/`, never the artifact mount. They are scratch,
not new artifacts: listed with their SHA-256 in the manifest and deleted when the export step
ends. They fit the disk: the root filesystem had 44 GB free on 2026-10-12 (batch guidance of
the report batch), and step 6 checks for at least 10 GB free before it copies.

% W-376, W-398, §2.11

| Step | Parallel | Limit per process | Estimate and its basis | New artifacts |
| --- | --- | --- | --- | --- |
| 1: inventory of the inputs | 1 | 1 CPU, 4 GiB, 3600 s | 122,412,640,023 bytes: the 1,024 raw images of 112,014,708 bytes (the size of each of the 32 files that the source-inventory run hashed, the source-inventory run's `results/inventory/inputs.csv`; one shape for all, the inputs survey's `inputs.md`), the DAPI source images 4,721,387,720 bytes (the own-folder run), the files of Part 2's exports 2,961,471,379 bytes (the source-inventory run's `inputs.csv`, the saved-files run's `results/manifest.json`), the labels of four FOVs and the small files about 27 MB (Position057 and 058 at the size of the larger label file, and 10 MB for the small files, both assumed); at the rate of the source-inventory run's inventory step, 5,486,212,411 bytes from the share in 71.82 s at one CPU (76.39 MB/s; `results/inventory/inputs.json`): 1,603 s, with 10 % for the listing and the share's variation 1,770 s | below 1 MB |
| 2, 3: the 64 FOVs | 2 in step 2, then 3 | 1 CPU, 32 GiB RSS, 3600 s | one wave of two and 21 waves of three (62 FOVs), about 634 s each (the source-inventory run's `real_fovs.csv`, the slower FOV): 22 × 634 s = 13,948 s; about 68 GiB on the host (the source-inventory run's `sizing_64.csv`, three in parallel: 67.5 GiB, 3.82 h) | records with `candidates` and `pre_qc` 4.25 GiB; rule outputs 6.8 GiB (the source-inventory run's `sizing_64.csv`) |
| 4: report after the FOVs | 1 | 1 CPU, 16 GiB, 3600 s | 64 run records of about 43 kB; the replay of two rounds of one FOV, 447 MB per round as uint8: about 600 s, below 4 GiB (code-derived) | below 50 MB |
| 5: block import and assignment | 1 | 1 CPU, 16 GiB, 3600 s per FOV | a 50 × 1496 × 1496 uint32 label image is 448 MB ([assignment algorithms specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/assignment-algorithms.md#evidence), "Evidence"): about 120 s per FOV, 480 s | labels and territories, about 3.6 GB |
| 6: copies and exports | 1 | 4 CPU threads, 32 GiB, 10,800 s | route A: the ZYX copies 130 s (the first stitching run) and the fusion from the copies 1,468 s (the first stitching run); route B: no copy and the fusion from the share 5,903 s (the own-folder run); both: the rest of the export run 239 s (the own-folder run: run 6,142 s less fusion 5,903 s) and the block about 180 s | stack 0.66 GB, `molecules.csv` 0.74 GB (the own-folder run), block below 0.2 GB; route A's copies (6.7 GiB) are scratch, deleted |
| 7: reports after the export | 1 | 1 CPU, 16 GiB, 3600 s | `molecules.csv` of 3.8 million rows in memory (about 2 GiB with pandas) and the stack read plane by plane (34 MB per plane): about 900 s each, two reports and the comparison page (about 120 s) | below 50 MB each |
| Part 2: five reports and their comparison page | 1 | 1 CPU, 16 GiB, 3600 s each | the rate of step 7, 900 s for LN's 3,782,580 molecules with its stack, scaled by each export's molecules and at least 120 s (code-derived): `ln_sample1` 900 s, `tissue_2d` 1,274 s (5,353,714 molecules), `cell_culture_3d` 585 s (2,458,865), the two blocks 120 s each, the comparison page 120 s; molecule counts of [stitching and assembly validation design](https://github.com/wanglab-broad/starfinder/blob/dev/design/validation/stitching-assembly.md), "Results of the real examples" | below 0.3 GiB in all |
| 8: manifest, last | 1 | 1 CPU, 4 GiB, 1800 s | about 120 s (the source-inventory run's manifest step: 120.0 s) | below 10 MB |

% W-384, W-383, W-398, W-391, W-376

The total of the wall time, step by step, for each route:

| Step | Route A (local copies), s | Route B (share), s |
| --- | --- | --- |
| 1: inventory of the inputs, 122.4 GB at 76.39 MB/s and 10 % | 1,770 | 1,770 |
| 2, 3: 22 waves of 634 s | 13,948 | 13,948 |
| 4: report after the FOVs | 600 | 600 |
| 5: block import and assignment, 4 × 120 s | 480 | 480 |
| 6: ZYX copies | 130 | 0 |
| 6: export `ln_sample1` (fusion and the rest 239 s) | 1,707 | 6,142 |
| 6: export `ln_sample1_block` | 180 | 180 |
| 7: two reports of 900 s and the comparison page | 1,920 | 1,920 |
| Part 2: `ln_sample1` report | 900 | 900 |
| Part 2: `ln_sample1_block` report | 120 | 120 |
| Part 2: `cell_culture_3d` report | 585 | 585 |
| Part 2: `cell_culture_3d_block` report | 120 | 120 |
| Part 2: `tissue_2d` report | 1,274 | 1,274 |
| Part 2: comparison page | 120 | 120 |
| 8: manifest | 120 | 120 |
| **Total** | **23,974 (6.7 h)** | **28,279 (7.9 h)** |

Both routes are within the 9 hours Jiahao set on 2026-10-12 (the slower takes 7.9 h), and the
new artifacts, about 16 GiB, within 20 GiB. The order puts the report that
needs only the run records first, so a stop at a later step still leaves it.

### Part 2: the three real samples from their exports, with no run record

As soon as the report is implemented, without processing an image (amendment 2): a report of each
existing whole-sample export and of the two cellular blocks, read where they are saved and
written to the new run directory (`output_dir`), never into the run directory of the export.

% A2

| Sample | Export (read only, in the run directory of the run named) | Cells | Fused image | `ReportConfig` |
| --- | --- | --- | --- | --- |
| `ln_sample1` | the own-folder run's `results/export/ln_sample1/` | `molecules_only` | `dapi_stack.ome.tif` | `markers.yaml` groups; `example_fovs` named (below) |
| `ln_sample1_block` | the saved-files run's `results/export/ln_sample1_block/` | the block segmentation's `expanded_nuclei` | none (marker patterns `part_not_written`) | `markers.yaml` groups and pairs; `example_fovs` named |
| `cell_culture_3d` | the saved-files run's `results/export/cell_culture_3d/` | `molecules_only` | `reference_merged_projection.ome.tif` | no markers; `example_fovs` named |
| `cell_culture_3d_block` | the saved-files run's `results/export/cell_culture_3d_block/` | the block segmentation's `cellpose` | none | no markers; `example_fovs` named |
| `tissue_2d` | the saved-files run's `results/export/tissue_2d/` | `molecules_only` | its projection | no markers; `example_fovs` named |

% W-398/20261009T045906Z-2d4054c3/results/export/ln_sample1/, W-391/20261008T203138Z-905169ae/results/export/ln_sample1_block/, W-382, W-391/20261008T203138Z-905169ae/results/export/cell_culture_3d/, W-391/…/results/export/cell_culture_3d_block/, W-391/…/results/export/tissue_2d/

* Every section up to the readout names `no_run_record`; the correction from the stage
  position is `no_placement` where the export's root has no `stitching/<sample>/`; the
  later sections are made from the export.
* The automatic example choice needs the read count of a run record ({doc}`report-contract`,
  "The example FOVs"), so the batch names the three FOVs: it applies the rule of "The
  example FOVs" to the export's `tables/fovs.csv` `n_molecules` in place of `passed_filter`
  (and `n_cells_retained` for the blocks), records the choice and passes it as
  `example_fovs`.
* Every existing export is of the route per FOV (they record no `assembly.route`, read as
  `per_fov`, choice 20 of {doc}`report-contract`); no real export after fusion exists, and
  making one needs a sample segmentation, which this plan does not run. On real data the
  reports show the route per FOV with and without cells; the route after fusion is shown on
  synthetic data (RT1, RT2).
* Then `Dataset.compare_samples` over the five reports.
* Resources: each report one CPU, 16 GiB, 3600 s; the estimates per report are rows of the
  total of Part 1 ("Resources and the bounds"), since both parts run in one batch; new
  artifacts below 0.3 GiB in all.
* Jiahao inspects: the pages of a sample without run records; the marker patterns of LN
  and the three most abundant genes of the others; the reads along Z of tissue-2D, whose
  molecules span 30 planes over a one-plane grid ([stitching and assembly validation design](https://github.com/wanglab-broad/starfinder/blob/dev/design/validation/stitching-assembly.md),
  "Results of the real examples"), so that its rows run over the planes its molecules have,
  beyond the grid's plane 0 (the rule of `reads_along_z.csv`, as amended on 2026-10-12); cells, MECR and the per-cell distributions of the blocks;
  the comparison page.

% W-449

## Limitations

* The provisional limits come from the earlier pipeline's logs, whose quantities differ
  from the Python records (shift axes and sign, spots found on another round, correct form
  as a subset of good reads); four of the twelve rules cannot be counted from those logs.
* The alignment-score rule has no default until a Python run of a real sample gives the
  distribution of `ncc_after`.
* The owned area is a count of pixel centres (OA1); its error is bounded by the pixels the
  region's boundary crosses, not zero.
* The sizes, times and memory of the report itself are estimates from code and from
  the source-inventory run's figure sizes; the first implementation measures them.
* The layout was reviewed as a static mockup; no browser check of the implemented page is
  part of the validation (the real example's inspection is Jiahao's).

% W-384

## Exclusions

No implementation, template, test or change of `run.json`, of a checkpoint, of the package
or of the workflow in this issue; no change of the accepted pages or of
{doc}`report-baseline` and {doc}`report-contract`; no new dependency and no lock change; no
real data. Not in the report: the comparison of the two routes (`compare_routes`), the
earlier pipeline's logs, the registered images, any filtering, correction, normalization,
composite score or pass and fail verdict. Not in the first delivery: the items listed as
excluded in {doc}`report-contract` ("Settled decisions this page follows"). Nothing here
marks a page as accepted.
