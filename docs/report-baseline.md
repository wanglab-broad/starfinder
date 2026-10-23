# Processing-report baseline: the legacy QC notebook, today's diagnostics, the records the report reads and their gaps

Status: Accepted (W-387, 2026-10-12, at c0aa8b0)

This page records what exists for the §2.11 processing report before {doc}`report-contract`
specifies it: the legacy QC notebook and the logs it reads, the diagnostics and figures each
stage already writes, the summary writer of the reference branch, the printing helpers of
the benchmark, and, value by value, the record and field each value of the report comes
from today. For each gap and each discrepancy it gives the options and one recommended
resolution. The recommendations are choices for Jiahao at the gate (W-387); nothing here is
implemented, copied or renamed.

**The report describes a run and judges nothing.** It shows what the records of a sample's
processing hold, lists the FOVs whose values lie outside a stated limit with the measured
value, the sample median and the limit, and says where a value is unavailable and why. It
gives no pass or fail verdict and no composite score, and it filters, corrects or changes
nothing: no table, image or record of the sample is rewritten by it.

Sources, read on 2026-10-12:

* `dev` at `dc5fc3c` (pull request 35, the §2.10 work merged), on the batch branch
  `runner/s211-spec-20261012`. Paths are relative to `src/python/starfinder/` unless they
  start with `src/`, `workflow/` or `docs/`.
* The reference branch `codex/chapter-ii-m1-m2` at `17e8dda` (2026-09-23), read with
  `git show 17e8dda:<path>`; it is not checked out or merged.
* The run directory of W-384 under the benchmark root,
  `runs/W-384/20261012T065105Z-18876aac/`: `worker-notes.md`;
  `results/inventory/report_values.csv` (the 61 values of the report, block by block, from
  the eight sections of the 2026-10-07 scope, the agreed mockup and the inputs of the D4 and
  D5 rules); `results/inventory/source_inventory.csv` (161 rows, one per value and route,
  with record, field, writer and setting); `results/inventory/lookup.csv` and
  `lookup.json` (314 lookups of those fields in the files of two synthetic runs and of two
  W-398 exports, none off its prediction); `results/replay/replay.csv`;
  `results/distributions/candidate_limits.csv`; `results/figures/figure_sizes.csv`;
  `results/tables/real_fovs.csv` and `sizing_64.csv`. Nothing was measured again for this
  page.
* The agreed layout mockup
  `runs/W-152/20261007T1939Z-s211-report-scope/processing-report-layout.html` (its sections,
  blocks and controls; its numbers are placeholders) and the W-383 inputs
  `runs/W-383/20261008T0240Z-inputs/` (`inputs.md`, `markers.yaml`, and the survey that
  defines MECR).

## What exists today

### The legacy notebook `workflow/notebook/create_rsf_qc_report.py.ipynb`

Fifteen cells, last changed in `01896c1` (2025-11-17), run by no rule:

| Cells | What they do |
| --- | --- |
| 0–1 | Imports (pandas, seaborn, matplotlib) and a hard-coded output folder, `…/Analyzed/2024-09-05-Mingrui-PFC-rep1`. |
| 2–6 | Read `log/gr_shifts.csv` (`fov_id`, `round`, `row`, `col`, `z`); draw a heatmap of the three shift numbers of every FOV and round with a fixed colour range of ±0.1 × 2048; list the rows whose `row` lies outside ±200; draw a swarm plot of `z`. |
| 7–9 | Read `log/sf_scores.csv` and `documents/sample-annotation.csv` (`sample_id`, `fov_start`, `fov_end`) and map each FOV to its sample. |
| 10 | The text `ssssss`: the notebook does not run past this cell. |
| 11–13 | Box plots by sample of `spots_in_codebook`, `total_spots` and `good_spots`. |

The two files it reads are concatenations of per-FOV logs that no rule or script at
`dc5fc3c` writes (searched in `workflow/` and `src/matlab/`); the per-FOV logs are
`log/gr_shifts/<fov>.txt` and `log/sf_scores/<fov>.txt`. The `sf_scores` columns it reads
are those of the MATLAB backend (`src/matlab/STARMapDataset.m`, lines 926 to 940:
`total_spots`, `no_color`, `multi_color`, `spots_in_codebook`, `spots_in_correctform` or
`correctform_1` and `correctform_2`, `good_spots`); the Python backend writes JSON to the
same file name (below). The notebook is the only per-sample QC view today. Its shift heatmap
and its per-sample box plots are the ancestors of the report's registration heatmap and of
the comparison page (D2).

### Logs and records each stage writes

| Stage | Record (under the output root) | Written by, at `dc5fc3c` | Exists when | What it holds for the report |
| --- | --- | --- | --- | --- |
| Per-FOV run | `checkpoints/<fov>/run.json` | `dataset/_run_record.py`: `_RunRecord.write` and `_RunRecord.finish`; the record is created by `FOV._start_run_record` in `FOV.run` (`dataset/fov.py`) | `FOV.run` with a `CheckpointConfig`, with any stages (also none). The workflow's `_run_workflow` (`dataset/workflow.py`) calls `fov.run(...)` without one, so a workflow run writes none. | Identity, status and error, code and environment, the configurations, channels, inputs with SHA-256, steps with seconds and methods, the preprocessing record (recipe, per-step `fitted`, transforms per round), the registration attempts with their QC, counts, the checkpoint files ({doc}`checkpoints`, "run.json"). |
| Registered images | `checkpoints/<fov>/registered/transforms.json`, `<round>.ome.tif`, `<round>_field.npz` | `_checkpoint.py`: `write_registered_header`, through `FOV._write_checkpoint` | stage `registered` | Every round's step results with their transforms (translation `displacement_zyx`, affine `matrix_zyx`; B-spline coefficients and dense fields in the `.npz`) and the round's `WarpConfig`. |
| Candidates | `checkpoints/<fov>/candidates.csv` and `.json` | `_checkpoint.py`: `write_candidates` | stage `candidates` | The spot table with the extracted values; the detection and extraction configurations; the image metadata. |
| Reads before the filter | `checkpoints/<fov>/pre_qc.csv` and `.json` | `_checkpoint.py`: `write_pre_qc` | stage `pre_qc` | The reads after decoding, scoring and deduplication; the decoder; the codebook's layout and encoding. |
| Segmentation per FOV | `checkpoints/<fov>/segmentation/<run>/segmentation.json`, `labels.tif` | `segmentation/_persist.py`: `write_run`, through `FOV.segment(checkpoints=…)` | a computed run saved with checkpoints | The run record of {doc}`segmentation-contract`. No time: the result's `wall_seconds` is a diagnostic that is not saved. |
| Assignment per FOV | `checkpoints/<fov>/assignment/<name>/assignment.json`, tables, `cell_labels.tif`, `nucleus_labels.tif` | `assignment/_persist.py`: `write_assignment`, through `FOV.assign_molecules(checkpoints=…)` | `FOV.assign_molecules` with checkpoints (the workflow's `reads_assignment` with an `assembly` block) | The assignment record, whose `inputs.cells` names the segmentation run or the imported mask it read; its tables and label images. |
| Workflow logs | `log/<fov>_rsf.txt` | `dataset/fov.py`: `FOV.save_processing_log` | rule `rsf_single_fov` (`_run_workflow`), or a caller of the method | The registration attempts with their QC; `detected`; the filter counts. |
| | `log/gr_shifts/<fov>.txt` | `dataset/fov.py`: `FOV._save_shift_log` | the same; translation results only | `fov_id`, `round`, `row`, `col`, `z`, which are dy, dx, dz of each translation. |
| | `log/sf_scores/<fov>.txt` | `dataset/fov.py`: `FOV.save_diagnostics` | the same | `detected`; the filter `counts` and `fractions` (with `endpoint_valid`); the read summary of `summarize_reads`; the registration attempts. |
| | `signal/<fov>_goodSpots.csv`, `images/ref_merged/<fov>.tif` | `dataset/fov.py`: `FOV.export_spots`, `FOV.save_reference_image` | the same | The accepted reads; the reference image whose grid stitching reads. |
| Placement | `stitching/<sample>/placement.json`, `fovs.csv`, `links.csv` | `stitching/_persist.py`: `save_placement`, through `Dataset.stitch(output_dir=…)` | rule `stitch_sample` | The method; the positions source with its SHA-256; per FOV the status, origin, nominal position and correction ({doc}`stitching-contract`). |
| Export | `export/<sample>/export.json` | `export/_record.py`: `write_export` and `update_section`, through `dataset/_export.py`: `export_sample` (`Dataset.export`) | rule `export_sample` | The parts and their status; the `assembly` section (route, ownership rule, population counts); the `fusion` section (grid, outputs, `wall_s`); `segmentation`; `labels`; `spatialdata` ({doc}`sample-output-contract`, {doc}`sample-segmentation-contract`). |
| | `export/<sample>/tables/fovs.csv`, `cells.csv`, `molecules.csv`, `counts.csv`, `sample.h5ad` | `assembly/_persist.py`: `save_assembly` (the AnnData by `assembly/_anndata.py`: `write_sample_h5ad`) | the assemble part written or reused | Per FOV the placement status, footprint, cells and molecules with their retained counts; per cell and molecule the assembly status; the counted cells' counts ({doc}`assembly-contract`, "Sample tables"). |
| | `export/<sample>/<image>_projection.ome.tif`, `<image>_stack.ome.tif` | `export/_fuse.py`: `fuse_images` | the fuse part | The fused image the marker patterns are drawn on. |
| | `export/<sample>/segmentation_<run>.ome.tif` | `export/_segment.py`: `segment_fused` | route after fusion, part `segment` | The sample's label image. |

Figures the stages already draw: the assignment figures `expr/<fov>/assignment.png` and
`assignment_single_layer.png` (`dataset/workflow.py`: `_write_assignment_figures`), drawn by
`reads_assignment` and again by `export_sample` with the FOV's ownership region on the route
per FOV; the benchmark's inspection images (below). None is a per-sample view of the
processing.

Diagnostics that exist but that `FOV.run` does not save: the read filter's
`diagnostics["endpoint_valid_count"]` and `endpoint_valid_segment_counts`
(`readout/filtering.py`: `filter_reads`), which reach only `log/sf_scores/<fov>.txt`, as a
fraction, through `FOV.save_diagnostics`; and the segmentation result's `wall_seconds`.

### `src/python/starfinder/reporting.py` on the reference branch (blob `c2e527b`, 161 lines)

Not on `dev`. It writes a standalone HTML summary of a saved run:

| Name | What it does |
| --- | --- |
| `render_table(rows)` | An escaped HTML table of records or of a DataFrame, without the index. |
| `embed_figure(fig, label, svg=False)` | Closes a matplotlib figure and embeds it as a base64 PNG `<img>` (or inline SVG) with an alt label, so the page opens offline. |
| `write_run_summary(path, destination, *, sha256=None, title=…)` | Reads a run record through `starfinder.provenance.read_run` (the reference branch's provenance module, which verifies component checksums) and writes the outcome, the stage states and links to the checkpoints; refuses an existing destination; "missing metrics remain unavailable". |
| `_display`, `_details`, `_population`, `_link` | Private formatting helpers. |

`starfinder.provenance` does not exist on `dev`, and the run record it reads is not
`run.json`.

### The printing helpers of `src/python/starfinder/benchmark/_reporting.py` on `dev`

W-546 removed `_e2e_summary` and `generate_overview_grid` because neither has an in-tree
caller; their definitions are preserved at the audited source revision
[`c30ed2e`](https://github.com/wanglab-broad/starfinder/blob/c30ed2e/src/python/starfinder/benchmark/_reporting.py).
`_print_quality_report` remains as a test helper used by `test_evaluation.py`; it prints the
values of an `EvaluationResult` with their units and reasons, writes "undefined" for a missing
value and adds no threshold. `generate_inspection_image` remains in use by the `synthetic`
command and writes a green–magenta composite of a reference and a moving volume.

### What to bring over

| Part | Proposal | Why |
| --- | --- | --- |
| The notebook's shift heatmap and per-sample box plots | **Keep the views, leave the notebook.** The registration heatmap (all FOVs and rounds) and the comparison page's three box plots (D2) take their place, read from the Python records. | The notebook reads files nothing writes, MATLAB quantities and a hard-coded path, and stops at cell 10. |
| `embed_figure` and `render_table` | **Keep the rules, rewrite the code**: figures embedded as base64 images, every text escaped, no external resource. | D3 asks for one offline page filled with the standard library; the functions are small, and their module needs `starfinder.provenance`, which `dev` lacks. |
| `write_run_summary` | **Leave.** | Its record is not `run.json`. Its two rules, an existing destination is never overwritten and a missing metric stays unavailable, are kept by {doc}`report-contract` ("The output folder", "Unavailable parts"). |
| `_e2e_summary`, `generate_overview_grid` | **Delete in W-546.** | Neither has an in-tree caller; the historical definitions are linked at [`c30ed2e`](https://github.com/wanglab-broad/starfinder/blob/c30ed2e/src/python/starfinder/benchmark/_reporting.py). |
| `_print_quality_report`'s "undefined" with a reason | **Keep the test helper.** | The report's unavailable rule is the same idea with a closed list of reasons. |
| `generate_inspection_image` (green–magenta) | **Keep the helper.** | The `synthetic` command uses it; its input is a benchmark volume pair, and the report's example is replayed from the raw input ({doc}`report-contract`, "The replay"). |

## The source inventory

The value list is W-384's `report_values.csv`: every value of the eight sections of the
2026-10-07 scope (the outline's §2.11 content), block by block as the mockup draws them,
and the inputs of the D4 example choice and of the D5 rules: 61 values. Each row below is a
row of `source_inventory.csv`, one per value and route. Route "every" holds on both routes
to a sample's cells; "per FOV" and "after fusion" are the two routes
({doc}`workflow-downstream`, "Two routes to a sample's cells"); "export only" is a sample
with an export folder and no `run.json` (the W-398 exports). W-384 kept its second export
in a folder `export_after_fusion/` of its own; a sample has one `export/<sample>/`, written
on its route, and the rows below name that folder. Status: **source** (the field holds the
value), **partial** (it holds part of the value, or holds it only under the setting the
note names), **derived** (the report computes the value from the field), **gap** (no record
holds it). The writer and the setting of each record are in "Logs and records each stage
writes" above; a field is a JSON key path (`*` for every key or item), a column of a table,
or an image. A gap or a partial row is resolved in "Gaps" or "Discrepancies" below.

Every row was looked up in the files of W-384's synthetic runs (stages `candidates` and
`pre_qc`, and with `registered` as well; both routes) and in the two W-398 exports
(`lookup.csv`); every lookup matched its prediction.

### Summary

| ID | Value | Route | Record | Field | Status | Note |
| --- | --- | --- | --- | --- | --- | --- |
| S01 | Dataset (dataset ID) | every | `checkpoints/{fov}/run.json` | `dataset_id` | source |  |
|  |  | export only | `export/{sample}/export.json` | `assembly.fovs.*.grid.metadata.frame_id` | partial | export.json records no dataset ID; it is readable only inside the frame_id of a declared grid |
| S02 | Sample (sample ID) | every | `checkpoints/{fov}/run.json` | `sample_id` | source |  |
|  |  | per FOV | `export/{sample}/export.json` | `sample` | source |  |
|  |  | export only | `export/{sample}/export.json` | `sample` | source |  |
| S03 | Acquisition: rounds x channels, 3D or 2D | every | `checkpoints/{fov}/run.json` | `channels` | source | rounds x channels: the rounds and their channel entries |
|  |  | every | `checkpoints/{fov}/candidates.json` | `metadata` | partial | spacing of the detection grid; the Z size is in the export or the reference image |
|  |  | per FOV | `export/{sample}/export.json` | `fusion.fovs.*.grid.shape_zyx` | source | 3D or 2D from the Z size of the FOV grids |
|  |  | export only | `export/{sample}/export.json` | `fusion.fovs.*.grid.shape_zyx` | partial | the Z size only; rounds and channels are not in an export |
| S04 | Configuration check (identical for all FOVs) | every | `checkpoints/{fov}/run.json` | `config.pipeline` | derived | equal config.pipeline (and execution) over the FOVs; a FOV without run.json is unknown |
| S05 | Version (starfinder version and commit) | every | `checkpoints/{fov}/run.json` | `code.git_commit` | source | code.version, code.git_commit, code.git_dirty |
|  |  | per FOV | `export/{sample}/export.json` | `assembly.software.code.git_commit` | source |  |
|  |  | export only | `export/{sample}/export.json` | `assembly.software.code.git_commit` | source |  |
| S06 | Time (generated; time of the processing) | every | `checkpoints/{fov}/run.json` | `started_at` | source | started_at, ended_at per FOV |
|  |  | every | — | — | gap | no record holds it today: it is the report's own value, to be written in its record report.json (D1), which does not exist yet |
| S07 | FOVs succeeded of FOVs expected | every | `checkpoints/{fov}/run.json` | `status` | source | succeeded per FOV; the FOVs expected are the entry's explicit FOV list (D1) |
| S08 | Median spots per FOV | every | `checkpoints/{fov}/run.json` | `counts.spots` | derived | median over FOVs of counts.spots |
| S09 | Spots matched to the codebook (sample fraction) | every | `checkpoints/{fov}/run.json` | `counts.call_status.assigned` | derived | sum of counts.call_status.assigned / sum of counts.spots |
| S10 | Counted cells | per FOV | `export/{sample}/sample.h5ad` | `X` | source | the rows of X (counted cells) |
|  |  | after fusion | `export/{sample}/sample.h5ad` | `X` | source | the rows of X (counted cells) |
|  |  | export only | `export/{sample}/sample.h5ad` | `X` | partial | X has no row: the W-398 exports are molecules_only (no segmentation) |
| S11 | Median reads per cell | per FOV | `export/{sample}/sample.h5ad` | `X` | derived | median of the row sums |
|  |  | after fusion | `export/{sample}/sample.h5ad` | `X` | derived | median of the row sums |
|  |  | export only | `export/{sample}/sample.h5ad` | `X` | partial | no cell |
| S12 | Median genes per cell | per FOV | `export/{sample}/sample.h5ad` | `X` | derived | median of the nonzero entries per row |
|  |  | after fusion | `export/{sample}/sample.h5ad` | `X` | derived | median of the nonzero entries per row |
|  |  | export only | `export/{sample}/sample.h5ad` | `X` | partial | no cell |
| S13 | Table: FOV, section, measured, sample median, reason (collected from every section) | every | — | — | gap | no record holds the collected table today: the report collects it (D1: report.json and tables/, which do not exist yet) from the rule rows S14, R01, R03, R07, D02, D03, D01, G06, G09 and their limits |
| S14 | Rule input: FOV that failed or was expected and not run | every | `checkpoints/{fov}/run.json` | `error` | source | status failed or interrupted with error.step; a FOV of the list without run.json is not run or not recorded (A2: shown as having none) |
| X01 | Choice of the three example FOVs: read count per FOV near the median, fewest and most cells | every | `checkpoints/{fov}/run.json` | `counts.filtering.accepted` | derived | read count per FOV near the median (D4); cells per FOV from tables/fovs.csv n_cells_retained |

### Workflow overview

| ID | Value | Route | Record | Field | Status | Note |
| --- | --- | --- | --- | --- | --- | --- |
| W01 | Per-FOV lane: each step in run order with its method (preprocessing steps, registration, spot finding, decoding, read QC filter, deduplication, segmentation, assignment) | every | `checkpoints/{fov}/run.json` | `preprocessing.recipe.steps` | source | the preprocessing steps in order |
|  |  | every | `checkpoints/{fov}/run.json` | `registration.*.*.actual_method` | source | the registration method per round and step |
|  |  | every | `checkpoints/{fov}/run.json` | `steps.*.methods.*.method` | source | the spot-finding method (provenance entry of find_spots) |
|  |  | every | `checkpoints/{fov}/run.json` | `config.pipeline.decoding.method` | source |  |
|  |  | every | `checkpoints/{fov}/run.json` | `config.pipeline.filtering` | source |  |
|  |  | every | `checkpoints/{fov}/run.json` | `config.pipeline.deduplication` | source | null: not run |
|  |  | per FOV | `checkpoints/{fov}/assignment/{run}/assignment.json` | `inputs.cells.run` | source | the segmentation run the assignment read (here the imported mask 'truth'); its method is in segmentation/<run>/segmentation.json when FOV.segment saved it |
|  |  | per FOV | `checkpoints/{fov}/segmentation/{run}/segmentation.json` | `method` | partial | absent here: the synthetic runs import masks with import_labels; for an imported mask the method is the import, recorded in assignment.json inputs.cells |
|  |  | per FOV | `checkpoints/{fov}/assignment/{run}/assignment.json` | `config` | source | the assignment settings (expansion, sampling) |
| W02 | Per-sample lane: stitching, assembly, export with their methods | every | `stitching/{sample}/placement.json` | `method.method` | source |  |
|  |  | per FOV | `export/{sample}/export.json` | `assembly.ownership.rule` | source |  |
|  |  | per FOV | `export/{sample}/export.json` | `parts` | source | each part written, reused, off or not_run |
|  |  | after fusion | `export/{sample}/export.json` | `segmentation.runs.*.method.method` | source |  |
|  |  | export only | `export/{sample}/export.json` | `fusion.placement.method` | source |  |
| W03 | A step that was not run shown as such | every | `checkpoints/{fov}/run.json` | `config.pipeline` | derived | an operation whose config is null was not run |
|  |  | per FOV | `export/{sample}/export.json` | `parts` | source |  |
| W04 | Status per FOV (succeeded, failed, with the failing step) | every | `checkpoints/{fov}/run.json` | `status` | source |  |
| W05 | Seconds per step across FOVs | every | `checkpoints/{fov}/run.json` | `steps.*.seconds` | source |  |
|  |  | per FOV | `export/{sample}/export.json` | `fusion.wall_s` | source | the only sample step with a recorded time |
|  |  | per FOV | `stitching/{sample}/placement.json` | `wall_s` | gap | no time of stitch, assemble or segment_fused is recorded |

### Preprocessing

| ID | Value | Route | Record | Field | Status | Note |
| --- | --- | --- | --- | --- | --- | --- |
| P01 | Example FOV after each step of the recipe, from the raw input, one row per channel (replayed) | every | `checkpoints/{fov}/run.json` | `inputs.*.path` | source | the raw input (with sha256 when hash_inputs) |
|  |  | every | `checkpoints/{fov}/run.json` | `preprocessing.rounds.*.*.config` | source | the recorded recipe, replayed (results/replay) |
| P02 | Intensity distribution after each step, one curve per channel | every | `checkpoints/{fov}/run.json` | `preprocessing.rounds.*.*.fitted` | derived | the intensity distribution after each step is made by the replay; no histogram is recorded |

### Registration

| ID | Value | Route | Record | Field | Status | Note |
| --- | --- | --- | --- | --- | --- | --- |
| R01 | Shift residual from the round's sample median (per FOV and axis) | every | `checkpoints/{fov}/run.json` | `registration.*.*.qc.details.transform.displacement_zyx` | source | the estimation attempt's QC record holds the step's displacement; refutes gap 1 for global steps |
|  |  | every | `checkpoints/{fov}/registered/transforms.json` | `transforms.*.*.transform.displacement_zyx` | source |  |
|  |  | every | `log/gr_shifts/{fov}.txt` | column `row` | source | row, col, z = dy, dx, dz of each translation result |
| R02 | Absolute shift (per FOV and axis) | every | `checkpoints/{fov}/run.json` | `registration.*.*.qc.details.transform.displacement_zyx` | source | the estimation attempt's QC record holds the step's displacement; refutes gap 1 for global steps |
|  |  | every | `checkpoints/{fov}/registered/transforms.json` | `transforms.*.*.transform.displacement_zyx` | source |  |
|  |  | every | `log/gr_shifts/{fov}.txt` | column `row` | source | row, col, z = dy, dx, dz of each translation result |
| R03 | Alignment score (masked NCC after registration) | every | `checkpoints/{fov}/run.json` | `registration.*.*.qc.values.ncc_after` | source | on the estimation entries (per step) and on the application entry (the whole chain) |
|  |  | every | `log/{fov}_rsf.txt` | `registration_attempts.*.*.qc.values.ncc_after` | source |  |
| R04 | FOVs at their stitched positions (map geometry) | every | `stitching/{sample}/fovs.csv` | column `origin_y` | source |  |
|  |  | per FOV | `export/{sample}/tables/fovs.csv` | column `footprint_y0` | source |  |
|  |  | export only | `export/{sample}/tables/fovs.csv` | column `footprint_y0` | source |  |
| R05 | Heatmap: FOV x round x (dz, dy, dx) | every | `checkpoints/{fov}/run.json` | `registration.*.*.qc.details.transform.displacement_zyx` | source | the estimation attempt's QC record holds the step's displacement; refutes gap 1 for global steps |
|  |  | every | `checkpoints/{fov}/registered/transforms.json` | `transforms.*.*.transform.displacement_zyx` | source |  |
|  |  | every | `log/gr_shifts/{fov}.txt` | column `row` | source | row, col, z = dy, dx, dz of each translation result |
| R06 | Example FOV before and after registration, reference green, moving magenta (replayed) | every | `checkpoints/{fov}/registered/transforms.json` | `transforms` | source | the saved chain, applied to the replayed preprocessing |
|  |  | every | `checkpoints/{fov}/run.json` | `registration.*.*.qc.details.transform` | partial | without the registered stage: the global steps can be rebuilt from the QC records; a dense or B-spline step cannot (statistics only) |
| R07 | Registration rejected or succeeded only with a fallback method | every | `checkpoints/{fov}/run.json` | `registration.*.*.outcome` | source |  |
|  |  | every | `checkpoints/{fov}/run.json` | `registration.*.*.fallback` | source |  |

### Readout

| ID | Value | Route | Record | Field | Status | Note |
| --- | --- | --- | --- | --- | --- | --- |
| D01 | Spots detected | every | `checkpoints/{fov}/run.json` | `counts.spots` | source |  |
|  |  | every | `log/sf_scores/{fov}.txt` | `detected` | source |  |
|  |  | export only | `export/{sample}/export.json` | `counts.spots` | gap | an export holds the accepted reads only |
| D02 | Correct barcode form | every | `checkpoints/{fov}/run.json` | none (`counts.filtering.endpoint_valid_count` absent) | gap | gap 3 confirmed: counts.filtering holds total, accepted, rejected; the count is in the filter result's diagnostics (endpoint_valid_count) |
|  |  | every | `log/sf_scores/{fov}.txt` | `fractions.endpoint_valid` | partial | the fraction over all reads, written by the rule's save_diagnostics, not by FOV.run |
|  |  | every | `checkpoints/{fov}/pre_qc.csv` | column `observed_color_sequence` | partial | recomputable from the color sequences and the filter config; D6: not recomputed by the report |
|  |  | export only | `export/{sample}/tables/molecules.csv` | column `endpoint_valid` | gap | an export holds the accepted reads only, without their form |
| D03 | Matched to the codebook | every | `checkpoints/{fov}/run.json` | `counts.call_status.assigned` | source |  |
|  |  | export only | `export/{sample}/tables/molecules.csv` | column `call_status` | gap | every molecule of an export is matched (it has a gene); the reads that were not matched are not in it, so no fraction |
| D04 | Passed the read QC filter | every | `checkpoints/{fov}/run.json` | `counts.filtering.accepted` | source |  |
|  |  | export only | `export/{sample}/tables/fovs.csv` | column `n_molecules` | partial | the rows of each FOV's molecule table (goodSpots), the reads that passed the filter of the earlier run |
| D05 | Deduplication (reads merged away) | every | `checkpoints/{fov}/run.json` | `counts.deduplication.merged_reads` | source | absent here: the synthetic runs do not deduplicate |
|  |  | export only | `export/{sample}/export.json` | `counts.deduplication` | gap | not in an export |
| D06 | Reads assigned to cells after overlaps are removed (equal to the sum of sample.h5ad counts) | per FOV | `export/{sample}/sample.h5ad` | `X` | source | sum of X = retained reads assigned to a counted cell; equal to the molecules retained with a cell |
|  |  | per FOV | `export/{sample}/tables/molecules.csv` | column `cell_id` | source | retained molecules with a counted cell, the equality the report checks |
|  |  | after fusion | `export/{sample}/sample.h5ad` | `X` | source | sum of X = retained reads assigned to a counted cell; equal to the molecules retained with a cell |
|  |  | after fusion | `export/{sample}/tables/molecules.csv` | column `cell_id` | source | retained molecules with a counted cell, the equality the report checks |
|  |  | export only | `export/{sample}/tables/molecules.csv` | column `cell_id` | gap | molecules_only: no cell |
| D07 | Each read level as a fraction | every | `checkpoints/{fov}/run.json` | `counts` | derived | fractions and tables from D01 to D05 (D02 from the gap's eventual record) |
| D08 | Distributions: spots per FOV, fraction with correct form, fraction matched | every | `checkpoints/{fov}/run.json` | `counts` | derived | fractions and tables from D01 to D05 (D02 from the gap's eventual record) |
| D09 | Per-FOV table: FOV, spots, correct form, in codebook, passed QC, in codebook % | every | `checkpoints/{fov}/run.json` | `counts` | derived | fractions and tables from D01 to D05 (D02 from the gap's eventual record) |

### Segmentation and assembly

| ID | Value | Route | Record | Field | Status | Note |
| --- | --- | --- | --- | --- | --- | --- |
| G01 | Cells per FOV | per FOV | `export/{sample}/tables/fovs.csv` | column `n_cells` | source | per FOV: its own cells (per_fov); the sample cells attributed to it by ownership centroid (after_fusion) |
|  |  | after fusion | `export/{sample}/tables/fovs.csv` | column `n_cells` | source | per FOV: its own cells (per_fov); the sample cells attributed to it by ownership centroid (after_fusion) |
|  |  | export only | `export/{sample}/tables/fovs.csv` | column `n_cells` | partial | 0: no segmentation |
| G02 | Cell density (retained cells attributed to the FOV / area of the region it owns) | per FOV | `export/{sample}/tables/fovs.csv` | column `n_cells_retained` | partial | numerator of the density; the denominator is gap G02 (area owned) |
|  |  | per FOV | `export/{sample}/export.json` | none (`assembly.ownership.area_um2` absent) | gap | no record of the area each FOV owns; computable with assembly.ownership_map from the placement, the grids and assembly.config (rule, tie) |
|  |  | after fusion | `export/{sample}/tables/fovs.csv` | column `n_cells_retained` | partial | numerator of the density; the denominator is gap G02 (area owned) |
|  |  | after fusion | `export/{sample}/export.json` | none (`assembly.ownership.area_um2` absent) | gap | no record of the area each FOV owns; computable with assembly.ownership_map from the placement, the grids and assembly.config (rule, tie) |
|  |  | export only | `export/{sample}/tables/fovs.csv` | column `n_cells_retained` | partial | 0 |
| G03 | Reads assigned to cells, % per FOV | per FOV | `export/{sample}/tables/molecules.csv` | column `assembly_status` | derived | retained molecules with a counted cell / retained molecules, per FOV |
|  |  | after fusion | `export/{sample}/tables/molecules.csv` | column `assembly_status` | derived | retained molecules with a counted cell / retained molecules, per FOV |
|  |  | export only | `export/{sample}/tables/molecules.csv` | column `cell_id` | gap | molecules_only: no cell |
| G04 | Cells retained after assembly per FOV | per FOV | `export/{sample}/tables/fovs.csv` | column `n_cells_retained` | source |  |
|  |  | after fusion | `export/{sample}/tables/fovs.csv` | column `n_cells_retained` | source |  |
|  |  | export only | `export/{sample}/tables/fovs.csv` | column `n_cells_retained` | partial | 0 |
| G05 | Placement of each FOV: correction from the stage position | per FOV | `stitching/{sample}/fovs.csv` | column `correction_y` | source | correction_z/y/x = origin - nominal (stage position) |
|  |  | after fusion | `stitching/{sample}/fovs.csv` | column `correction_y` | source | correction_z/y/x = origin - nominal (stage position) |
|  |  | export only | `export/{sample}/tables/fovs.csv` | column `correction_y` | gap | the export holds origins, not corrections; the placement folder (stitching/<sample>/fovs.csv) has them when it is at hand |
| G06 | Placement status (placed, not placed, fell back to the stage position) | per FOV | `export/{sample}/tables/fovs.csv` | column `placement_status` | source | image_supported, metadata_only (left at the stage position), failed, placeholder |
|  |  | after fusion | `export/{sample}/tables/fovs.csv` | column `placement_status` | source | image_supported, metadata_only (left at the stage position), failed, placeholder |
|  |  | export only | `export/{sample}/tables/fovs.csv` | column `placement_status` | source |  |
| G07 | Segmentation outlines on three example FOVs (nucleus and cell outlines) | per FOV | `checkpoints/{fov}/assignment/{run}/cell_labels.tif` | the label image | source | or ../../segmentation/<run>/labels.tif (linked) when FOV.segment saved the run |
|  |  | per FOV | `checkpoints/{fov}/assignment/{run}/nucleus_labels.tif` | the label image | source |  |
|  |  | after fusion | `export/{sample}/segmentation_cell.ome.tif` | the label image | source | cut to the example FOV's footprint (A1) |
|  |  | after fusion | `export/{sample}/segmentation_nucleus.ome.tif` | the label image | source |  |
|  |  | export only | `export/{sample}/segmentation_cell.ome.tif` | the label image | gap | no segmentation |
| G08 | Population accounting: input, retained, excluded not owner, excluded FOV not placed, counted cells; cells and molecules | per FOV | `export/{sample}/export.json` | `assembly.counts.cells.excluded.not_owner` | source | cells and molecules: input, retained, excluded by reason |
|  |  | per FOV | `export/{sample}/export.json` | `assembly.counts.molecules.excluded.fov_not_placed` | source |  |
|  |  | after fusion | `export/{sample}/export.json` | `assembly.counts.cells.excluded.not_owner` | source | cells and molecules: input, retained, excluded by reason |
|  |  | after fusion | `export/{sample}/export.json` | `assembly.counts.molecules.excluded.fov_not_placed` | source |  |
|  |  | export only | `export/{sample}/export.json` | `assembly.counts.molecules.excluded.not_owner` | source | molecules only; the cell rows are 0 |
| G09 | Reads but no cell (per FOV) | per FOV | `export/{sample}/tables/fovs.csv` | column `n_molecules_retained` | derived | reads but no cell: n_molecules_retained > 0 and n_cells_retained = 0 |
|  |  | after fusion | `export/{sample}/tables/fovs.csv` | column `n_molecules_retained` | derived | reads but no cell: n_molecules_retained > 0 and n_cells_retained = 0 |
|  |  | export only | `export/{sample}/tables/fovs.csv` | column `n_molecules_retained` | partial | every FOV has reads and no cell: the rule does not apply without segmentation |

### Sample measurements

| ID | Value | Route | Record | Field | Status | Note |
| --- | --- | --- | --- | --- | --- | --- |
| M01 | Reads per cell | per FOV | `export/{sample}/sample.h5ad` | `X` | derived | row sums |
|  |  | after fusion | `export/{sample}/sample.h5ad` | `X` | derived | row sums |
|  |  | export only | `export/{sample}/sample.h5ad` | `X` | partial | no cell |
| M02 | Genes per cell | per FOV | `export/{sample}/sample.h5ad` | `X` | derived | nonzero per row |
|  |  | after fusion | `export/{sample}/sample.h5ad` | `X` | derived | nonzero per row |
|  |  | export only | `export/{sample}/sample.h5ad` | `X` | partial | no cell |
| M03 | Cell size (µm³, µm² for a plane, voxels when uncalibrated) | per FOV | `export/{sample}/sample.h5ad` | `obs.size_physical` | source | unit: assignment.json size_unit (per_fov) or the sample assignment record (after_fusion) |
|  |  | after fusion | `export/{sample}/sample.h5ad` | `obs.size_physical` | source | unit: assignment.json size_unit (per_fov) or the sample assignment record (after_fusion) |
|  |  | export only | `export/{sample}/tables/cells.csv` | column `size_physical` | partial | no cell row |
| M04 | Whole-sample view: one point per molecule of the supplied markers on the fused DAPI image | per FOV | `export/{sample}/tables/molecules.csv` | column `sample_x` | source | with gene_id, one point per retained molecule |
|  |  | per FOV | `export/{sample}/export.json` | `fusion.outputs.*.path` | source | the fused DAPI (reference_stain) image |
|  |  | after fusion | `export/{sample}/tables/molecules.csv` | column `sample_x` | source | with gene_id, one point per retained molecule |
|  |  | after fusion | `export/{sample}/export.json` | `fusion.outputs.*.path` | source | the fused DAPI (reference_stain) image |
|  |  | export only | `export/{sample}/tables/molecules.csv` | column `sample_x` | source |  |
|  |  | export only | `export/{sample}/export.json` | `fusion.outputs.*.path` | source |  |
| M05 | Zoom-in regions (footprints of the three example FOVs), XY and XZ | per FOV | `export/{sample}/tables/fovs.csv` | column `footprint_x0` | source | zoom regions: the example FOVs' footprints |
|  |  | per FOV | `export/{sample}/export.json` | `fusion.config.dimensionality` | partial | an XZ view of the image needs a fused stack (dimensionality stack or both); molecules have sample_z |
|  |  | after fusion | `export/{sample}/tables/fovs.csv` | column `footprint_x0` | source | zoom regions: the example FOVs' footprints |
|  |  | after fusion | `export/{sample}/export.json` | `fusion.config.dimensionality` | partial | an XZ view of the image needs a fused stack (dimensionality stack or both); molecules have sample_z |
|  |  | export only | `export/{sample}/tables/fovs.csv` | column `footprint_x0` | source |  |
| M06 | MECR per supplied pair: cells with A, cells with B, both, MECR | per FOV | `export/{sample}/sample.h5ad` | `X` | derived | positive cells per gene of a pair (count >= threshold) |
|  |  | after fusion | `export/{sample}/sample.h5ad` | `X` | derived | positive cells per gene of a pair (count >= threshold) |
|  |  | export only | `export/{sample}/sample.h5ad` | `X` | partial | no cell: MECR undefined |
| M07 | Heatmap: reads per FOV along Z | per FOV | `export/{sample}/tables/molecules.csv` | column `z` | derived | reads per FOV and voxel plane z |
|  |  | after fusion | `export/{sample}/tables/molecules.csv` | column `z` | derived | reads per FOV and voxel plane z |
|  |  | export only | `export/{sample}/tables/molecules.csv` | column `z` | source |  |

### Files

| ID | Value | Route | Record | Field | Status | Note |
| --- | --- | --- | --- | --- | --- | --- |
| F01 | Images (input root and layout) | every | `checkpoints/{fov}/run.json` | `inputs.*.path` | source |  |
|  |  | export only | `export/{sample}/export.json` | `fusion.fovs.*.images` | partial | the images the fusion read (reference images), not the raw input |
| F02 | Codebook path | every | `checkpoints/{fov}/run.json` | none (`codebook` absent) | gap | gap 2 confirmed: no codebook path in run.json (nor in pre_qc.json, which holds the encoding and layout) |
|  |  | every | `checkpoints/{fov}/pre_qc.json` | `encoding` | partial | the encoding, not the file |
| F03 | Tile positions (stage file or positions table) | every | `stitching/{sample}/placement.json` | `table.path` | source | with table.sha256 and table.converter |
|  |  | export only | `export/{sample}/export.json` | `fusion.placement.method` | partial | the method and table hashes only; the positions source is in the placement folder |
| F07 | Markers (the mockup names a marker file; D4: the markers are in the report configuration, no separate file) | every | — | — | gap | no record holds it today: the markers belong to the report configuration (D4, no separate file), which does not exist yet |
| F04 | Workflow configuration path and hash | every | `checkpoints/{fov}/run.json` | none (`config.config_path` absent) | gap | run.json holds the typed configurations, not the workflow file or its hash |
| F05 | Report settings | every | — | — | gap | no record holds it today: the report settings are the report block of the workflow configuration (D4), which does not exist yet |
| F06 | Checkpoints, placement, sample export, report and report tables locations | every | `checkpoints/{fov}/run.json` | `checkpoint_directory` | source | the placement and export folders are the entry's fixed places (D1); the report's own folder is its output |

## Gaps

Each gap is a value, or part of one, that no record holds where the report looks. The
three gaps D6 named were confirmed or refuted from W-384's run with stages `candidates` and
`pre_qc` only (`worker-notes.md`, section 4); the others were found by the inventory.

| # | Gap | Rows | Options | Recommended resolution |
| --- | --- | --- | --- | --- |
| G1 | **The workflow writes no `run.json`.** `_run_workflow` calls `fov.run(adapted.pipeline, execution=adapted.execution)` without a `CheckpointConfig`, so a sample processed by the rule `rsf_single_fov` has no run record: every row whose record is `run.json` is absent for it. What it has are the three logs of `FOV.save_processing_log` and `FOV.save_diagnostics`. | every `run.json` row | (a) The report also reads `log/<fov>_rsf.txt`, `log/gr_shifts/<fov>.txt` and `log/sf_scores/<fov>.txt` as a second source. (b) The workflow writes `run.json`: `_run_workflow` passes `CheckpointConfig(stages=<the workflow's stages, default none>, overwrite=True)`. (c) Both. | **(b)**, in the implementation issue of D6 ({doc}`report-contract`, "Records added to run.json", R5). One record holds every value with one meaning; the logs hold a subset (no steps, no times, no preprocessing record, translations only) and the MATLAB backend writes other columns to the same file names. `stages=()` writes `run.json` alone (43 kB for an LN FOV with the registration attempts, W-384 `real_fovs.csv`); `overwrite=True` because a rerun of the rule replaces its record, and it leaves `other_rounds/`, `segmentation/` and `assignment/` as they are ({doc}`checkpoints`, "Layout"). The logs stay as they are. |
| G2 | **The global transforms outside the stage `registered`.** Refuted for global steps, confirmed for dense and B-spline steps: `run.json` holds no `transforms` key, but each estimation attempt's QC record holds its step's transform summary, `registration.<round>[i].qc.details.transform` (a translation's `displacement_zyx`; an affine's 3 × 3 `matrix`, `offset` and `det`), written by `evaluation/registration.py`: `_transform_summary` through `dataset/fov.py`: `_qc_record`. A dense or B-spline step holds displacement statistics only. The application entry summarises a chain of one translation as `kind: dense` with statistics. | R01, R02, R05, R06 | (a) The report reads the QC records' transform summaries. (b) `run.json` gains the transforms of every round in the form of `transforms.json`, the global ones with their parameters, a dense or B-spline step by its kind and the file that holds it, if any. (c) The stage `registered` writes `transforms.json` always, without the images. | **(b)** (R1 of {doc}`report-contract`). The QC summary is a QC value whose form the registration contract does not fix as a record, it is null for an attempt that failed before its QC, and an affine's offset there is not the 4 × 4 index matrix the chain is rebuilt from. (c) changes what a stage writes. For an older `run.json` the report reads `registered/transforms.json` when it exists and shows the shifts as unavailable otherwise. |
| G3 | **The codebook file.** No key of `run.json`, `pre_qc.json`, `candidates.json` or the three logs names it; `config.encoding` holds the encoding, not the file. {doc}`checkpoints` says so: "`Dataset.load_codebook` does not keep its path". | F02 | (a) `run.json` records the file's path and SHA-256 (`Dataset.load_codebook` keeps them). (b) The report takes the codebook path from its configuration. (c) The report takes the genes from the export or `pre_qc`. | **(a)** (R2). The Files section shows the codebook a run used, and the marker check (D4) needs the gene list of that codebook. (b) would show what the user typed, not what the run read. |
| G4 | **The correct-form count per FOV.** `counts.filtering` holds `total`, `accepted`, `rejected`; the count is the filter result's `diagnostics["endpoint_valid_count"]`, which `_RunRecord.write` does not copy. It reaches `log/sf_scores/<fov>.txt` only as a fraction. | D02 | (a) `run.json` copies the count (and the per-segment counts) into `counts.filtering`. (b) The report recomputes it from `pre_qc` and the filter configuration. (c) The report reads the log's fraction. | **(a)** (R3). D6 settles that missing records are added, not recomputed; (b) needs the `pre_qc` stage and the codebook; (c) needs G1's logs. |
| G5 | **The workflow configuration file and its hash** (Files, Configuration). `run.json` holds the typed configurations, not the file they came from. | F04 | (a) `run.json` records the workflow configuration file's path and SHA-256 when the run comes from the workflow (the top-level `config_path` key names the file). (b) The report takes it from its own configuration. (c) Unavailable. | **(a)** (R4), written by `_run_workflow`; null for a run outside the workflow. |
| G6 | **The area each FOV owns** (cell density on both routes, A1). `export.json` records the ownership rule and tie (`assembly.ownership`), not the regions. | G02 | (a) The report computes the area from the placement, the FOVs' grids and the assembly configuration, with `assembly.ownership_map`. (b) The export records the area per FOV (a new column of `tables/fovs.csv`). | **(a)**: the report computes it, as it computes the shift residual. (b) changes an accepted §2.10 table. The geometry of the area is specified on the algorithms page (W-386, `docs/report-algorithms.md`). |
| G7 | **Time of the sample steps and of the per-FOV segmentation and assignment.** Only the fuse part records its time (`export.json` `fusion.wall_s`; also `labels.wall_s` and `spatialdata.wall_s`); `placement.json`, the `assembly` and `segmentation` sections and the per-FOV `segmentation.json` and `assignment.json` record none (`segmentation.wall_s` exists only on the route after fusion). | W05 | (a) Show these times as unavailable. (b) Add `wall_s` to `placement.json`, to the `assembly` section and to the per-FOV records, as additive keys. | **(a)** now, with reason `not_recorded`; (b) changes records of accepted §2.9 and §2.10 pages and is left to Jiahao at the gate as a later addition. |
| G8 | **The report's own values**: the time the report is made (S06), the collected table of problematic FOVs (S13), the report settings (F05) and the markers (F07). No record holds them because the report and its configuration do not exist yet. | S06, S13, F05, F07 | (a) The report's record `report.json` and its configuration (D1, D4). (b) A separate settings file beside the report. | **(a)** ({doc}`report-contract`, "report.json", "Configuration"). The mockup's marker file is dropped (D4). |
| G9 | **A sample with an export and no `run.json`** (A2; the W-398 exports, `molecules_only`). It lacks every value up to the readout (steps, methods, times, status, inputs, preprocessing, registration and its QC, spots detected, correct form, matched, deduplication), the correction from the stage position, and every cell value. | D01, D02, D03, D05, D06, G03, G05, G07 and the "export only" partial rows | (a) Unavailable with a reason (`no_run_record` per FOV; `no_segmentation` for cells), the later sections made from the export. (b) Read the earlier pipeline's logs where they exist. | **(a)**. The earlier logs define their quantities differently (W-384 `worker-notes.md` section 7: shift axes and sign, spots found on another round, good reads, correct form as a subset of good reads), so one page would mix two definitions under one name. The correction from the stage position is read from `stitching/<sample>/fovs.csv` when the placement folder exists. |
| G10 | **The intensity distribution after each preprocessing step** (P02). No histogram is recorded; `fitted` holds each step's fitted values only. | P02 | (a) The replay computes it on the example FOV (D1). (b) `run.json` records a histogram per round, channel and step. | **(a)**. It is the distribution of the example FOV the page shows, which the replay already makes; (b) would add a record for every FOV that the page does not draw. |
| G11 | **The XZ view of the marker patterns** (M05) needs a fused stack; a projection export (`dimensionality: projection`) has only the molecules' `sample_z`. | M05 | (a) The XZ view of the molecules only (no image) for a projection export. (b) Unavailable without a stack. | **(b)**, with reason `no_stack`: the scope draws molecules on the fused image, and a view without it would be another figure. |
| G12 | **The dataset ID and the acquisition of a sample without `run.json`** (S01, S03): an export records no dataset ID (only inside a declared grid's `frame_id`) and no rounds or channels. | S01, S03 | (a) The dataset ID from the `Dataset` the entry is called on; the rounds and channels unavailable. (b) Both from the `Dataset`'s configuration. (c) Parse the `frame_id`. | **(a)**. The entry is a method of `Dataset`, whose `dataset_id` names the dataset; a `run.json` whose `dataset_id` differs raises ({doc}`report-contract`, "Checks before anything is written"). The rounds and channels the run used are recorded only in `run.json`; the configuration states what is configured now, not what was run. |
| G13 | **The segmentation method of an imported mask** (W01): `segmentation/<run>/segmentation.json` exists only for a run `FOV.segment` saved; an imported mask is recorded in `assignment.json` `inputs.cells`. | W01 | (a) The method from `segmentation.json` when it exists, else from `assignment.json` `inputs.cells` (named "imported"). (b) Unavailable for an imported mask. | **(a)**. |

## Discrepancies

Differences between what the report needs, as the scope, the decisions and the mockup put
it, and what the records hold or how they define it.

| # | Discrepancy | Options | Recommended resolution |
| --- | --- | --- | --- |
| 1 | The levels of the readout are not nested in the Python records. "Correct barcode form" (`endpoint_valid`) is over all reads and independent of the codebook match (`call_status == "assigned"`), so a read can be matched without the correct form and the reverse; in the earlier logs correct form is a subset of the good reads. | (a) Each level a count and a fraction of the spots detected, drawn as bars side by side, not as a funnel. (b) A nested level "correct form and matched", which needs one more count in `run.json`. | **(a)**. The page says the levels are not nested; "passed the read QC filter" is a subset of "matched" with the default `call_statuses=("assigned",)`, and the page states the filter's configuration. |
| 2 | D5 lists "Passed the read QC filter" before "Deduplication", but the pipeline deduplicates before it filters, and the filter rejects the duplicates (`exclude_duplicates`, default true). | (a) Pipeline order: spots, correct form, matched, merged away by deduplication, passed the filter, assigned to cells. (b) D5's order. | **(a)**, so the sample totals follow the steps the workflow overview shows. |
| 3 | `run.json` exists only with a `CheckpointConfig` (G1), while the scope assumes a run record per FOV. | See G1. | See G1. |
| 4 | The application entry's QC record summarises a chain of one translation as `kind: dense` with displacement statistics (W-384 finding 7). | (a) The report reads no transform from QC records. (b) Read the estimation entries' summaries. | **(a)**: the shifts come from the transforms record (G2); the alignment score is the application entry's `qc.values.ncc_after`, which is the score D5 names ("for the whole chain of a round"). |
| 5 | The mockup's Files section names a marker file; D4 puts the markers in the report configuration. | (a) The Files section lists the report configuration (the `report` block), with its markers, and no marker file. (b) A separate marker file beside the report, as the mockup draws it; D4 set this aside. | **(a)**, settled by D4; recorded here, not reopened. |
| 6 | The earlier pipeline's quantities differ from the Python records: its three shift numbers are (−dx, +dy, −dz) of the Python displacement (W-384 `tables/shift_compare.csv`), its spots are found on another round, its good reads are what is left after two drops. | (a) The report reads the Python records only; the earlier logs serve the provisional limits of W-384 only. (b) The report also shows the earlier logs. | **(a)**. A sample analysed before enters through its export (A2). |
| 7 | The scope's per-FOV lane names segmentation and assignment, which are not steps of `FOV.run`: they are `FOV.segment` and `FOV.assign_molecules` with their own records (`segmentation.json`, `assignment.json`), on the route per FOV only; on the route after fusion segmentation is a part of the export. | (a) The per-FOV lane shows them from their records on the route per FOV, and the per-sample lane shows `segment` on the route after fusion. (b) Leave them out. | **(a)**, as A1 requires. |
| 8 | The sample totals end with the reads assigned to cells, which equal the sum of `sample.h5ad`'s whole counts; the levels above are sums over FOVs that count a read in an overlap once per FOV that saw it (D5). | (a) The levels above as sums over the FOVs, each read in an overlap counted once per FOV that saw it, the figure saying so, and the last level checked against `sample.h5ad`. (b) Remove the reads of overlaps from the levels above before summing, which needs a cross-FOV duplicate rule for reads that no record holds; D5 set this aside. (c) No sample total for the levels above, per-FOV values only; D5 set this aside. | **(a)**, settled by D5; recorded here, not reopened. The report records both numbers of the check. |
| 9 | The route after fusion's population accounting has no `duplicate_observation` reasons and null flags (`export.json` `assembly.counts`; W-384 synthetic run: 9 cells in, 9 retained), and its `fovs.csv` `n_cells` is the sample cells attributed to the FOV by its ownership centroid ({doc}`sample-segmentation-contract`, "The FOVs"). | (a) Follow the assembly record of the route the export records: the reasons that do not exist on the route after fusion are absent, not zero. (b) One accounting form for both routes, with zero for the reasons the route does not have; A1 set this aside, and D1 forbids showing a missing value as zero. | **(a)**, settled by A1; recorded here, not reopened ({doc}`report-contract`, "The two routes"). |
| 10 | The mockup's colour control "Cells per mm²" shows a density; A1 defines it as the retained cells attributed to a FOV divided by the area of the region it owns, which no record holds (G6). | See G6 for where the area comes from. For the unit: (a) cells per mm², as the mockup labels it; (b) cells per µm², the unit of the sample frame. | **(a)**: cells per mm², computed from the area of G6 (the numbers per µm² are below 0.01 for tissue). |
| 11 | The comparison of the two routes (`compare_routes`, {doc}`sample-segmentation-contract`) writes a record of its own. | (a) The report does not read it. (b) The report reads and shows its record when one exists; A3 set this aside. | **(a)**, settled by A3; recorded here, not reopened. |
| 12 | The legacy notebook stays in `workflow/notebook/` and reads files nothing writes. | (a) Leave it, and say in `docs/migration.md` that the report replaces it. (b) Remove it with the implementation. | **(a)**; its removal is a later choice. |
