# Benchmark infrastructure

**Status: Proposed — W-771; approval belongs to W-775 at an exact revision.**

This contract replaces the proposed direction of the old Benchmark lifecycle;
it does not remove that API. The implementation baseline is `7cd1e7d712210b684f488a525d214667ffd766b0`
(W-769 plus W-770). File/line citations below refer to that revision. The
[baseline and retirement table](../baselines/benchmark.md) distinguishes present
behavior from outstanding work; the [validation design](../validation/benchmark.md)
gives known answers. No scientific acceptance follows from these records.

Authority for scope is W-124's 2026-11-02 decisions R1–R12, D5 and D9, copied by
the controller into `decisions.md` and `short-set.md` for W-771. W-775 reviews this
plan before any retirement. No new executor, per-experiment script, workflow
task kind, workflow schema, default, or MATLAB convention is introduced here.

## Execution and experiment boundaries

The §2.13 workflow is the executor for FOV and downstream work: use its plan,
task records, verification, engine profiles and failure handling. Stack trials
are a narrow method entry invoked once per process, not another scheduler.

| Experiment | Route and interpretation |
| --- | --- |
| E01a | Single-stack registration through the private trial entry; use held-out detected landmarks for TPS/CPD evaluation. |
| E01b | One FOV and downstream through workflow outputs; compare direct/tiled molecules separately from truth accuracy. |
| E02 | Single-stack detection through the trial entry once a detection adapter exists; real-data counts are plausibility checks. |
| E03 (includes E09) | Evaluate saved read scores; cut-off curves are descriptive, never an optimum/default selector. |
| E04 | Single-stack segmentation through the trial entry once a segmentation adapter exists. |
| E05 | Workflow completion, engine equality, failures/retries and billed cost; freeze cross-host tolerances before execution. |
| E07 | Workflow placement and overlapping molecules in a common physical frame. |
| E12 | Conditional comparison; no automatic authorization to run it. |
| E13a | Synthetic preprocessing through stack trials; execution on UGER via the workflow `sge` profile remains to be connected without a new task kind. |
| E13b | Real FOV outputs through the workflow, unchanged schema; stratify edge/dense FOVs. |
| E14 | Python/MATLAB outputs with explicit boundary conversion; W-212 remains G1's coordinate/channel check; UGER through `sge`. |
| E15 | Fit resource models from task records only; hold out an entire dataset. |

Timing/memory experiments run on GP099-29C; the E13a/E14 UGER route is the stated
exception. Results preceding W-93 are pilots. Neither these routes nor command
proposals authorize datasets, cluster work or scientific sweeps.

## Implemented private trial input and record

Entry: `python -m starfinder.benchmark._trial trial.json` (JSON or YAML mapping).
Source: `src/python/starfinder/benchmark/_trial.py:29–111,124–194`.

| Input field | Implemented rule |
| --- | --- |
| `method` | `module: registration` plus registered `method` and optional `config`, or `module: preprocessing` plus nonempty ordered `steps` of `{method, config}`. Registry dataclasses resolve defaults and canonical type/method names; unknown keys/types fail. No spot-finding or segmentation adapter yet. |
| `inputs` | Registration: exactly `reference`, `moving`; preprocessing: `image`, optionally `reference`. Each value is a path or `{path, sha256}`. Paths resolve relative to the trial file, including absolute paths; they are not confined to an input root. Hashes are computed during resolution and checked again during loading. |
| `metadata` | One common `ImageMetadata` mapping; default frame `trial/reference`. It is used for both registration images, so separate grids are not expressible here. Unknown calibration stays unknown. |
| `seed` | Required integer in `[0, 2**32)`, excluding Boolean. Seeds Python and NumPy; supplies `random_seed` only if a method has that field and it was not supplied explicitly. |
| `condition` | Optional mapping/context retained in the resolved configuration, default `{}`; the standalone resolver does not validate a condition schema. |
| `root_output_path` | Required by the executable entry, relative to the trial file or absolute. Directory must be empty, but may already exist. `_resolve` itself permits omission. No resume or concurrent-writer protocol. |

NPY input uses `allow_pickle=False`; other images use `io.load_volume`. Arrays
remain ZYX/ZYXC. Stage APIs do their own validity checks; `_load` does not itself
assert all arrays finite. Preprocessing executes `run_step` in order and supplies
a reference only to `needs_reference` steps; it is not the sample-fitting/FOV
pipeline. Registration estimates and applies a transform through the package.

`record.json` has format `starfinder.benchmark.trial/1` and these fields:

| Field | Implemented content |
| --- | --- |
| `configuration`, `configuration_sha256` | Resolved configuration and workflow canonical hash after successful resolution. On resolution failure configuration stays the original mapping and the hash is absent. |
| `seed`, `inputs` | Requested seed; resolved absolute paths and SHA-256 values after resolution, otherwise empty inputs. Inputs are not copied into the output. |
| `outputs` | Relative `path`, SHA-256 and `bytes` for `image.npy` and `transform.json` (plus `transform.npz` for dense/B-spline arrays), or `preprocessing.json` containing configs, fitted records and diagnostics. Does not include `record.json` itself. |
| `status`, `error` | `succeeded` with null error, or `failed` with exception type/message. Declared partial outputs are removed after a caught failure; outputs list stays empty. |
| `wall_s`, `cpu_s`, `peak_rss_kib` | Shared process-resource fields, described below. |

Successful entry exits 0, caught method/input failure exits 1. YAML parsing,
missing output-path keys and reservation errors occur before the catch and may
produce no record. This is not a guarantee to record every crash, signal or OOM.
No evaluation, fallback selection or automatic retry occurs here.

Resource source is `src/python/starfinder/_resources.py:6–12`, also used by
`workflow/_task.py:245–254`: wall time starts after output reservation and ends
before writing the final record; CPU is cumulative user/system time for self
plus reaped children; RSS is the larger of self/child high-water marks on Linux,
not their sum and not a simultaneous process-tree peak. Import/startup affects
CPU/RSS even when excluded from the wall interval. These are measurements, not
an enforced memory limit. They differ from the old lifecycle's tracemalloc
allocation count and null RSS; do not combine those quantities under one label.

## Implemented expansion manifest

`starfinder.benchmark._experiment.expand_experiment(path, output_dir)` is private
and has no CLI (`_experiment.py:18–66`). An experiment mapping accepts only
`methods`, `conditions`, `seeds`, `inputs`, and optional common `metadata`.
The four axes must be nonempty lists. Each condition accepts `name` and `method`;
`name` is not required by code. Method overrides merge dictionaries recursively;
lists/scalars replace. Conditions do not generate or mutate data: an input axis
must already identify the files for each desired case.

The Cartesian product order is method, condition, seed, input, with input varying
fastest. Each trial resolves defaults and input hashes before *any* file is
written. Output is `trial-000000.json`, etc., and `manifest.json`:

| Manifest field | Implemented content |
| --- | --- |
| `format` | `starfinder.benchmark.experiment/1` |
| `source` | Absolute experiment path and SHA-256 of source bytes. |
| `trials` | Ordered entries `{path, sha256, configuration_sha256}`. File hash is of canonical JSON trial bytes; configuration hash uses workflow normalization. |

Each trial's output is `output_dir/trial-NNNNNN`. Workflow configuration hashing
excludes the top-level `root_output_path` (`workflow/_shape.py`, `configuration_sha256`),
so relocation changes file hashes but not configuration hashes for otherwise
identical resolved inputs. Absolute input paths remain part of that identity.
Re-expansion to the same directory is byte-stable for unchanged source/inputs;
it overwrites those configuration/manifest files and does not prune stale files
or run outputs. Expansion does no processing and imports no engine.

## Proposed evaluate-only contract

This section specifies new behavior, not the existing `evaluate_benchmark` API.
A saved evaluation has a new output directory and explicitly selected metrics,
source manifests/records and their hashes, population filters, frames, units,
codebook/channel order, truth identities, matching policy/boundary/radius and
metric parameters. It records code/environment identity and per-source failures;
JSON missing values are null with a reason, never a silent zero. Evaluation
must neither invoke processing nor mutate original task/checkpoint/trial records.
Reports consume that saved evaluation, with no new metrics or processing.

Three sources are admitted:

1. **Workflow outputs.** Read declared task outputs and records; enforce the
   completion/fingerprint rule and hash every consumed file. The current command
   `starfinder workflow verify --manifest PATH --hash --final` performs the
   workflow check, but writes `workflow/verify.json`; perform it as a separate
   preflight, or use an equivalent read-only check inside evaluation. Missing
   required files fail explicitly; failed tasks remain visible as skipped metric
   populations, not successful zero rows. Truth is supplied separately, never
   inferred from processing output. Temporary workflow outputs are exempt from
   workflow completion, but cannot serve as missing evaluation inputs.
2. **Checkpoints.** Call public `starfinder.dataset.load_checkpoint(path, stage)`
   for `registered`, `candidates` or `pre_qc`; use restored typed results and
   metadata (`src/python/starfinder/_checkpoint.py:611–700`). Version 3 checks
   payload hashes; versions 1/2 lack them and require an explicit external source
   inventory with that limitation recorded. Missing snapshots or truth needed by
   a metric make that metric unavailable. This permits comparisons of stored
   stages; it is not a changed-config verified resume. Workflow resume rejects
   changed `blocks_sha256`; do not weaken that guard or call `FOV.run` to evaluate.
3. **Private trial outputs.** Verify `record.json` identity and each listed
   output's size/hash, and separately preserve/hash any original inputs needed
   by a metric. The trial record has no truth inventory, so supply one. Unlike
   the old copied-input run directory, a trial directory alone is not a portable
   evaluate-only dataset. Report that difference rather than infer missing truth.

M70/M71 use MATLAB good spots as reference and Python molecules as observed.
Use `assignment.import_legacy_molecules_csv`: one-based XYZ → zero-based ZYX,
explicit gene/population/channel mapping. M69 uses detected displacement ZYX;
MATLAB DFT correction `(row,column,Z)` converts to `-(Z,row,column)`, with no
one-based shift offset. G1 remains `src/python/scripts/matlab_stage_comparison.py`
and `workflow/scripts/compare_matlab_stages.m`; do not replace its mapping check
with a molecule agreement score. Freeze M70 radius before running. Agreement is
not molecular truth.

## Short-set ownership and semantics

All numerical metrics belong in `starfinder.evaluation`; task measurements and
external evidence retain their workflow/billing/manual provenance. The following
paths are relative to `src/python/starfinder/`. “Proposed” means absent as a
public pure metric at this base, not implemented by this design change. The
validation record covers the 38 distinct selected IDs plus M12/M13/M73/M74,
which the short set requests as accompanying measurements.

| IDs | Current home and contract / proposed gap |
| --- | --- |
| M01, M02 | `evaluation/matching.py:10` `match_points`: greedy Euclidean pairs sorted by distance then original indices, one-to-one; recall over eligible truth, precision over eligible detections; empty denominator null. KD-tree candidates replace dense allocation while preserving arithmetic/ties. |
| M03, M05 | `evaluation/spot_finding.py:14,41,80` `f1_score`, `auprc`, `scored_detection_auprc`. F1 zero for no detections with nonempty truth; empty truth null. AUPRC sums positive recall increments times current precision in descending threshold order; tied scores enter together. No threshold optimization. |
| M07, M08, M09 | `evaluation/preprocessing.py:44` `direct_metrics`: reference-round background RMSE/bias; median puncta contrast; population std/mean of nonempty round/channel medians of truth-neighborhood sums. See stage/unit and shell rules below. |
| M10 | `evaluation/readout.py:259` `correct_fraction`: spatial matching uses all detections first; count accepted correct/wrong-gene/unmatched detections afterward, divide correct by all truth. Rejected matches still consume a spatial pair. |
| M15, M16 | `evaluation/preprocessing.py:152,180` `detected_spot_intensity_spread`, `fov_variation`: M09 formula on observed sequences/detected coordinates, and population std/mean across FOV values. No groups/nonpositive mean or any missing FOV value gives null; negative/infinite FOV values fail. Caller separates edge/dense. |
| M17, M18 | `readout` decoded table `call_status`: unmatched/all reads; assigned count. Proposed pure evaluation reducer, including rejected reads in the denominator; do not equate assigned with accepted or correct. |
| M21 | `evaluation/registration.py:663` `evaluate_displacement_field`: median, linear p95 and max Euclidean error on explicit valid-overlap mask, voxel or component-scaled physical units. Empty mask null. |
| M24 | `evaluation/registration.py:276` `evaluate_landmark_alignment`: supplied-landmark recall using an explicit matching-policy parameter frozen by each experiment, called separately before and after. E01a freezes `nearest_candidate` within 2 voxels; fitted methods use held-out landmarks. |
| M26, M29 | `evaluation/registration.py:551` `registration_qc`: Pearson NCC before/after/gain on common valid overlap; fold fraction of det(I+gradient u) ≤ 0 over the transform grid, not just overlap. Central differences with existing edge handling; not independent evidence for image-similarity optimizers. |
| M34 | `evaluation/matching.py:106` `evaluate_molecules`, `any_neighbor`, inclusive Chebyshev `1 + 1e-9`; same-gene fractions in each direction, count ratio and absolute relative count difference. Many-to-one permitted; direct is reference. |
| M36 | `evaluation/spot_finding.py:219` `classify_detections`: unmatched detections near a matched truth are duplicates, others spurious; explicit radius/boundary and optional same/other channel groups. |
| M37, M39, M40 | `evaluation/readout.py:14,106` `evaluate_decoding`, `ranking_quality`: matched-pair gene accuracy/confusion; AUROC with half-credit ties and Hanley–McNeil SE; error at 50/80/90/100% retention with fractional boundary ties. |
| M41 | Proposed `evaluation.readout` cut-off curve over saved scores: retention, unmatched rate, per-gene counts/change and truth error where available. Only M43 qc_score, M46 WTA, M47 codebook-aware, M52 direct; explicit score orientation, no selected optimum. |
| M54, M55, M56, M58 | Proposed `evaluation.segmentation`: instance F1 at IoU 0.5–0.9, split/merge counts and nonzero label-size median/IQR. Existing segmentation validation helpers are not a public implementation of all four definitions. |
| M60 | `assignment/_correspondence.py:77` `match_nuclei` supplies share_in_cell/background and cell association; proposed evaluation adapter summarizes nuclei per cell without recomputing assignment. |
| M62, M68 | Proposed `evaluation.stitching`: gauge-aligned origin error (median/max µm); same-gene greedy overlap molecule distances and directional match rates in a common frame. |
| M63 | `stitching/_stitch.py:333–337` and `_bigstitcher.py:791–795` already save kept-link residual norm in µm. Proposed evaluation reader retains missing/unkept status; does not silently recompute with another gauge. |
| M69 | `evaluation/registration.py:722` `shift_difference`: per-round signed differences, max axis difference and mean L2 difference; missing rounds retain partial results with `missing` status. Per-round L2 can be derived for the planned presentation. |
| M70, M71 | `evaluation/matching.py:106` `evaluate_molecules`: Euclidean greedy same-gene pairs at declared d; both directional fractions and Python/MATLAB count ratio. Missing genes do not match. |
| M75, M76 | `workflow/_task.py:263`, `_verify.py:24,60`: task failure records and final completion/hash verification. Proposed retry aggregation from engine logs and pure cross-engine output comparison; current verifier is not that comparison. |
| M77, M78 | Proposed saved-evidence readers for actual attributed GCP billing and counted human steps. No simulated bill or automatically inferred human count. |
| M79 | Proposed `evaluation` resource-prediction reducer: median/max absolute relative error and underprediction count on held-out dataset records; zero measured resources undefined with explicit count. |

% Amended 2026-11-02 for W-775, amendment A1: M24 freezes matching policy per experiment, with E01a nearest_candidate within 2 voxels.

`direct_metrics` keeps W-770's exact stage semantics. No-background `none`/`pct`
uses zero background estimate; background arms use raw minus `bg_corrected`
snapshot, falling back to final image; `r1`/`r1_recon` map truth through the
observed raw→normalized value map and report normalized units. Contrast uses
center ±(1,1,1) peak and shell ±(2,6,6) minus ±(1,3,3) median, clipped at image
edges; noise is per-channel std where truth signal is below 1% of the global
reference signal maximum. Zero-noise molecules are omitted with a fraction
reported. M09/M15 round coordinates using Python rounding and sum clipped
neighborhoods. M12 is positive input becoming zero/all voxels across rounds;
M13 is unsigned output equal to dtype maximum/all voxels, zero for signed/float
output. These are accompanying metrics, not new primary endpoints.

The preprocessing condition registry now lives in
`synthetic/_preprocessing.py:1–201`, exported by `synthetic/__init__.py:19–21,48–53`:
`PREPROCESSING_CONDITIONS`, `EXTRA_CALIBRATED_CONDITIONS`,
`preprocessing_scene_config`, `calibrated_preprocessing_scene_config`.
D9's metric/condition extraction is satisfied; R4's report and caller migration
still needs the gate. The old script and its threshold selection remain present.

## Proposed public names and migration

| Proposed command | Responsibility; current state |
| --- | --- |
| `starfinder experiment expand --config PATH --output DIR` | Expose deterministic expansion with validated record format; private function exists, CLI absent. |
| `starfinder trial run --config PATH` | Expose one method/process with the trial record; private entry exists, CLI absent. No scheduler/retry logic. |
| `starfinder experiment evaluate --sources PATH --metrics PATH --output DIR` | The evaluate-only contract above; new implementation needed for workflow/checkpoint/trial sources. |
| `starfinder experiment report --evaluation DIR --output DIR` | W-237-based saved-evaluation report; new renderer needed. |

Keep `starfinder synthetic generate` and existing `starfinder workflow` commands.
There is no proposed `experiment run` engine. Remove old `benchmark run/evaluate/report`
and Python lifecycle objects only after W-775 and the replacement tests in the
baseline pass. Do not alias an old schema to the new one. Old run archives stay
readable through preserved source/revision and retained files; converting them
requires explicit inputs/provenance, not an implicit compatibility shim.

The three implementation proposals are [retirement and commands](../proposals/benchmark-drafts/retirement-and-commands.md),
[report template](../proposals/benchmark-drafts/report-template.md) and
[site section](../proposals/benchmark-drafts/site-section.md). Missing metrics and
adapter extensions are named prerequisites, not authorization to implement them
inside W-771 or silently widen the retirement issue.
