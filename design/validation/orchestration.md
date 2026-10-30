# Validation design: workflow orchestration

% Status: Proposed (W-645, 2026-10-30, at 56c35b8)

Status: Proposed. This record is reviewed at the specification gate of workflow orchestration
(W-646) before any implementation; approval publishes nothing.

This page designs the validation of the §2.13 workflow by the layers of decision D10: contract
tests in the quick selection Q; a Nextflow stub-run under its own marker, in a CI job that installs
Nextflow; the golden baseline of today's workflow (W-642, W-643); the `extended` engineering
validation (both engines end to end on a synthetic multi-FOV sample, exact equality between engines
in one environment, the original against the refactored per-FOV core, tiled against direct, and
recovery); and a checklist that a person runs on Broad UGER. It validates the contract of
[orchestration](../specs/orchestration.md), the tiled execution of
[orchestration-tiled](../specs/orchestration-tiled.md) and the engines of
[orchestration-engines](../specs/orchestration-engines.md). Tolerances across infrastructures
(GP099 against UGER, one environment against another) stay with experiment E05 of the benchmark
register; this page states none.

% W-645 (S4 of §2.13); the batch runner/s213-spec-20261029 starts from dev at 56c35b8

## What this design is and is not

It checks software: that the plan writes the manifest the contract defines, that each engine runs
exactly the manifest's tasks in an order its `needs` allow, that a task succeeds only under the
completion rule, that the two engines and the refactored workflow give the results of today's
workflow, and that recovery and verified resume behave as D8 says. It makes no scientific claim:
the tolerances below measure software precision (`docs/contributing.md`, "Expected values and
tolerances"), not accuracy. Previews are off by default (P5); every check on real data turns
`assignment.previews` and `export.previews` on, so that Jiahao can look at them.

## Fixtures

Every fixture is generated in the test session or by the check's script; no image is committed.

| Fixture | What | Size | Within |
| --- | --- | --- | --- |
| `tiny_sample` | `sample_scene_preset("sample-tiny-v1")`, seed 42, written as W-642's golden baseline writes it (`src/python/scripts/workflow_golden.py`, `fixture`): one sample of 2 × 2 FOVs, four rounds of four channels, the reference stain `ch04` and the additional round `flamingo`, `genes.csv`, `fovs.csv`, `positions.csv`, the truth labels as legacy label files | 5 × 36 × 36 voxels per FOV and channel | The default array bound (32 × 64 × 64, four channels, four rounds) per sequencing round; the batch guidance's named example |
| `worked_example` | The contract's eight worked-example configurations on `tiny_sample` (`orchestration.md`, "Worked example": `per-fov`, `after-fusion`, `molecules-only`, `supplied-xml`, `morphology-only`, `reference-only`, `refused-assignment`, `refused-matlab`), with a hand-written `dataset.xml` and view map for `supplied-xml` | Configurations only | — |
| `exactness_fov` | W-644's exactness fixture: the benchmark preset `small` (its twelve-gene codebook, four rounds, four channels, `uint16`, seed 42) at 16 × 64 × 64 with 60 amplicons (`runs/W-644/…/scripts/fixture_synthetic.py`) | 16 × 64 × 64 | The default bound |
| `exactness_fov_deformed` | `exactness_fov` with the deformation preset `gaussian_small` on the moving rounds | 16 × 64 × 64 | The default bound |
| `session_dataset` | The pytest session's small synthetic test dataset | 16 × 256 × 256 | AGENTS.md's existing exception |
| `golden_values` | The committed expected values of W-642 (`src/python/test/data/expected/test_workflow_orchestration_golden/`) and of W-643's invocation D (`…_golden_export/`) | Tables and records | — |
| `core_values` | The expected values of the per-FOV core on `session_dataset`, recorded at the base revision `56c35b8` by the batch-preparation issue before any refactor lands, with the recording script beside them | Tables and records | — |
| `aging_fov` | aging Position400, read-only (W-641's file list with SHA-256) | 36 × 2048 × 2048, nine rounds | Real data: needs the owner's grant per run (W-644's grant was for W-644) |

## Checks

Each row names its fixture, marker (the file's `pytestmark` plus the test's own), selection
(`docs/contributing.md`, "Test selections"), threshold and resource bound. An exact check cites the
identity requirement of `docs/contributing.md` ("Expected values and tolerances") that justifies it:
**I-hash** "file, record, export and run hashes"; **I-repeat** "same-environment repeatability";
**I-discrete** "integer and discrete outputs" and the table's row "integers, counts, labels, integer
images, strings, statuses, booleans"; **I-round** "checkpoint and old-format round trips". A float
compared with a tolerance uses the table's row for its dtype: float64 `rtol=1e-9`, `atol=1e-12`;
float32 `rtol=1e-6`, `atol=1e-6`.

Default resource bound, unless a row says otherwise: one CPU, one numerical thread, no GPU, peak RSS
at most 4 GiB, at most 1800 s per test (AGENTS.md); a test of 5 s or more is `slow`.

(orchestration-validation-checks)=

### Contract tests in Q (C)

| # | Check | Fixture | Marker | Selection | Threshold | Resource bound | Identity |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C1 | The plan writes `resolved.json` and `manifest.json` with the contract's fields; each task's id, kind, identity, `needs`, `inputs`, `outputs`, `record`, `log`, `command`, `execution` and group; the tasks in an order that respects `needs`; the task counts per kind of each variant equal the contract's table (29, 34, 8, 8, 8, 16) | `worked_example` | `workflow`, `contract` | Q | Every field, id, path and count exactly equal to the expected manifest of each variant (committed JSON, generated by the plan prototype's rules and reviewed) | default; under 5 s | I-discrete (strings, counts) |
| C2 | The manifest's `configuration.sha256` equals the SHA-256 of `resolved.json`'s bytes, and `configuration_sha256` the hash of its canonical result-affecting part; a change of `execution` or of the roots keeps it, a change of one scientific key changes it; a second plan on unchanged inputs rewrites neither file (modification times equal) | `worked_example` (`per-fov`) | `workflow`, `contract` | Q | Exact | default; under 5 s | I-hash |
| C3 | Declared outputs of every route: for each variant and task kind, the declared outputs equal the contract's table (`orchestration.md`, "Declared outputs") resolved through `_layout`; every task's outputs equal its group's templates filled with its identity | `worked_example` | `workflow`, `contract` | Q | Exact path sets | default; under 5 s | I-discrete (strings) |
| C4 | A real Snakemake dry run of `workflow/Snakefile` on each variant's manifest: the jobs per rule group equal the tasks per group; no job for a `refused-*` variant (the plan's exit 2 and message); after a completed tiled run a second dry run schedules nothing (open point E-1) | `worked_example`; `exactness_fov` run tiled for the last part | `workflow`, `contract`, `slow` | Slow (`other`); skipped with the reason "Snakemake not installed" where the `workflow` group is absent | Exact job counts | 1800 s; 4 GiB | I-discrete (counts) |
| C5 | Negative checks of the plan: `refused-assignment`, `refused-matlab`, a missing round folder, a channel label matching two files, a run name with `/`, each key that left the Python dialect, `backend: matlab` | `worked_example` and variants built in the test | `workflow`, `contract` | Q | Exit 2; the message names the key, FOV, round or label the contract's message names; nothing written | default; under 5 s | I-discrete (statuses, strings) |
| C6 | Negative checks of a task command: an unknown task, a resolved configuration whose hash differs, a `needs` task without a succeeded record, a missing declared input (each exit 2, nothing written); a resume whose `run.json` differs in configuration hash, input hash or package version (exit 3, every differing field named) | `tiny_sample`, one FOV, configuration `per-fov` without the downstream blocks | `workflow`, `contract` | Q | Exact exit codes and named fields | default; under 5 s per case, else `slow` | I-discrete (statuses) |
| C7 | The completion rule: an API call that raises (monkeypatched) exits 1, removes the attempt's declared outputs, writes `<task>.failed.json` and no record; a declared output missing after the call exits 1; a success writes the record last, listing every declared output but itself with its SHA-256 and the closed log | `tiny_sample`, one FOV | `workflow`, `contract` | Q | Exact | default; under 5 s | I-hash; I-discrete |
| C8 | The boundary: no `.smk` or `.nf` file of the Python workflow imports `starfinder`; the Python branch of `workflow/Snakefile` and `rules/tasks.smk` read no configuration key but `backend`; no package module of the Python workflow imports the legacy module (import graph) | the repository's files | `workflow`, `contract` | Q | No match | default; under 5 s | I-discrete (strings) |
| C9 | The frozen MATLAB workflow: `test_assignment_workflow.py` and `test_workflow_assembly_golden.py` pass unchanged except the import line from the legacy module; a MATLAB Snakemake dry run of `tests/tissue_2D_test.yaml` schedules W-641's three jobs; `tests/minimal_config.yaml` passes the MATLAB schema unchanged | their fixtures; W-641's MATLAB configuration | `workflow`, `contract` (the dry run also `slow`) | Q (the dry run: Slow `other`, skipped without Snakemake) | Exact (their own thresholds; 3 jobs) | default; the dry run 1800 s | I-discrete |

### The Nextflow stub-run (N)

| # | Check | Fixture | Marker | Selection | Threshold | Resource bound | Identity |
| --- | --- | --- | --- | --- | --- | --- | --- |
| N1 | `nextflow run nextflow/main.nf -stub-run -profile local` on the manifest of each Python variant the plan writes (`per-fov`, `molecules-only`, `supplied-xml`, `morphology-only`, `reference-only`, the tiled `after-fusion` with a supplied `dataset.xml`): every task of the manifest runs exactly once (the trace's task names against the manifest's ids), each after every task in its `needs` (start and completion times of the trace), and emits its stub record | `worked_example` manifests | `workflow`, `contract`, `nextflow`, `slow` | `nextflow` (its own CI job with Java 17 and Nextflow 26.10.4); skipped with the reason "Nextflow not installed" elsewhere | Exact task set; no ordering violation | 1800 s; 4 GiB (the JVM included) | I-discrete (strings, counts) |
| N2 | The refusals: the `after-fusion` manifest with Fiji (`bigstitcher_engine`) is refused before any task with the D3 message; a seed chain deeper than three is refused naming it | `worked_example` (`after-fusion`); a variant with a four-run seed chain | `workflow`, `contract`, `nextflow`, `slow` | `nextflow` | Non-zero exit; no task in the trace; the message | As N1 | I-discrete |
| N3 | Exit-status mapping: a stub that exits 1 on its first attempt is retried and succeeds; one that exits 2 is not retried and the run ends | `per-fov` manifest with a fault-injecting stub | `workflow`, `contract`, `nextflow`, `slow` | `nextflow` | Attempts in the trace: 2 and 1 | As N1 | I-discrete (counts) |

### The golden baseline (G)

The refactored Snakemake workflow on the migrated configurations against the values of today's
workflow, with the exceptions the contract lists (`orchestration.md`, "The exception of the golden
comparison": OC3, OC4, D6, D8) and the comparison across environments of W-643 (`STAGE_RECORD_DIGEST`).
P2: the comparison runs in the `slow` tier.

| # | Check | Fixture | Marker | Selection | Threshold | Resource bound | Identity |
| --- | --- | --- | --- | --- | --- | --- | --- |
| G1 | Today's workflow reproduces W-642's values (invocations A, A2, B, C), landed with the lock change before any refactor | `tiny_sample`; `golden_values` | `workflow`, `golden`, `slow` | Slow `other`; skipped without Snakemake | Integers, strings, booleans, integer images exactly; float64 values `rtol=1e-9`, `atol=1e-12`; the file digests only in the recording environment | 1800 s per Snakemake run; 4 GiB | I-discrete; I-hash for the same-environment digests |
| G2 | Today's workflow reproduces W-643's invocation D (stitching, export on three routes, report) | `tiny_sample`; `golden_values` | `workflow`, `golden`, `slow` | Slow `other`; skipped without Snakemake or the `multiview-stitcher` and `anndata` extras | As G1 | 1800 s per run; 4 GiB | As G1 |
| G3 | The refactored workflow (the migrated A to D configurations, the `fov`, `morphology`, `segment` import, `assign`, `stitch`, `export`, `report`, `compare` tasks) reproduces the values of G1 and G2 except the contract's exceptions | `tiny_sample`; `golden_values` | `workflow`, `golden`, `slow` | Slow `other` | As G1, on every value outside the exceptions; each exception names its decision | 1800 s; 4 GiB | As G1 |

### Engineering validation, `extended` (X)

| # | Check | Fixture | Marker | Selection | Threshold | Resource bound | Identity |
| --- | --- | --- | --- | --- | --- | --- | --- |
| X1 | Snakemake end to end: the `per-fov` and `molecules-only` variants with `--cores 1`; every task record says `succeeded`; every declared output exists and hashes as its record lists; `report/index.html` lists the sample | `tiny_sample` | `workflow`, `integration`, `extended` | Extended | Exact (every record, every hash) | 1800 s per variant; 4 GiB | I-hash |
| X2 | Nextflow end to end: the same variants with `-profile local`, and `supplied-xml`; the same checks | `tiny_sample` | `workflow`, `integration`, `extended`, `nextflow` | Extended, in the Nextflow job | As X1 | 1800 s per variant; 4 GiB beside the JVM's 1 GiB | I-hash |
| X3 | Exact equality between engines in one environment: X1's and X2's output trees, each declared output byte-equal by SHA-256, except the task records' `workflow.execution_context` and `wall_s`, `cpu_s`, `peak_rss_kib`, the logs, and the fields of where, when and with which software a record was made (W-642's normalization) | X1's and X2's trees | `workflow`, `integration`, `extended`, `nextflow` | Extended, in the Nextflow job | Exact on every other byte | As X2 | I-repeat; I-hash |
| X4 | Original against refactored on the per-FOV core: the `fov` task on `session_dataset` against `core_values` (the molecule table, `run.json`'s counts and stage records, the reference image, the checkpoint stages) | `session_dataset`; `core_values` | `workflow`, `dataset`, `golden`, `extended` | Extended | Integers, strings, labels and the reference image exactly; float64 `rtol=1e-9`, `atol=1e-12`; float32 `rtol=1e-6`, `atol=1e-6`; the contract's exceptions (OC3 frame ids, OC4 paths) left out | 1800 s; 4 GiB | I-discrete; I-round (the stages) |
| X5 | Tiled against direct, without local registration: tile sizes 32 (a 2 × 2 grid) and 22 (3 × 3), halos 6 and 12, against direct: candidate tables equal (IDs, positions, channels, peak values), molecule tables equal in namespace, ID, z, y, x, gene and entry; scores within tolerance; `run.json`'s counts equal | `exactness_fov` | `workflow`, `dataset`, `extended` | Extended | Exact for the tables' integer and string columns; scores float64 `rtol=1e-9`, `atol=1e-12` (W-644 measured at most 2.2e-16 on this fixture) | 1800 s; 4 GiB | I-discrete |
| X6 | Tiled against direct, with local registration (one demons step, halo 12): the tiled record's proposed tolerance, at least 95 % of direct's molecules have a tiled molecule of the same gene within one voxel (Chebyshev) and at least 95 % of the tiled molecules a direct one, counts within 5 % | `exactness_fov_deformed` | `workflow`, `registration`, `validation`, `extended` | Extended | The three proportions (not exact) | 1800 s; 4 GiB | — |
| X7 | Recovery: a task killed with SIGKILL after it wrote part of its outputs, then rerun by the engine (Snakemake `--rerun-incomplete`, Nextflow `-resume`): the final tree equals an uninterrupted run's; the killed attempt left no record; only the killed task and its downstream ran again (counted in the records' attempts and Nextflow's trace) | `tiny_sample`, `per-fov` | `workflow`, `integration`, `extended` | Extended (the Nextflow half in the Nextflow job) | Exact (hashes, counts) | 1800 s; 4 GiB | I-repeat; I-hash |
| X8 | Verified resume: `fov` with `execution.resume_from_checkpoint` after a stop at `pre_qc` resumes and equals a full run; with a changed scientific key, input or package version it refuses with exit 3 naming every differing field; a version-1 `run.json` is never resumed | `tiny_sample`, one FOV | `workflow`, `dataset`, `extended` | Extended | Exact | 1800 s; 4 GiB | I-round; I-repeat |
| X9 | The local-registration tolerance on a full-size synthetic FOV (OT1): X6 on `thick_medium` with `gaussian_small` at tiles of 512 and 256 | `thick_medium` (100 × 1024 × 1024) with `gaussian_small` | `workflow`, `registration`, `validation`, `extended` | Not in a selection: the validation issue's script, run once under a resource grant (about 6 GiB and 25 min per run, the tiled record's measurements) | Confirms or replaces X6's proportions | Needs an owner's grant: 8 GiB, 3 h, 1 CPU | — |

### The human UGER checklist (U)

Run by a person on Broad UGER (task group 4), each item recorded with its job ids, `qacct` records,
the commands and the outputs' locations. Real data is aging Position400 and the FOVs the owner names;
thresholds across infrastructures (UGER against GP099) are E05's.

| # | Check | Fixture | Marker | Selection | Threshold | Resource bound | Identity |
| --- | --- | --- | --- | --- | --- | --- | --- |
| U1 | The image on UGER: the SIF built from the OCI archive on a compute node; `starfinder --help` and `snakemake --version` inside it; its digest and SIF SHA-256 recorded | the image | none (a person) | owner-run | Both commands exit 0; versions as the lock | one job, 8 GiB, 1 h | — |
| U2 | Snakemake on UGER with the repaired profile: the head as a job, the `per-fov` variant of `tiny_sample`, every record `succeeded` | `tiny_sample` | none | owner-run | Every record `succeeded`; X1's checks | head `h_vmem=8G`; tasks as the manifest | — |
| U3 | Nextflow on UGER with `-profile sge`, the head as a job (`use Java-17`, `NXF_HOME` on `/stanley`) on the same variant; its outputs against U2's | `tiny_sample` | none | owner-run | X3's equality where U2 and U3 ran in the same image | head `h_vmem=8G`; tasks as the manifest | I-repeat (same image) |
| U4 | Both engines on full-size real FOVs, direct and tiled (2 × 2), previews on: task wall time and peak RSS per kind against the tiled record's resource model | `aging_fov` and the FOVs the owner names | none | owner-run | Each task within its request; peaks within the model's prediction or the deviation reported; equality between engines as X3, across the routes as E05 says | the owner's grant | — |
| U5 | GPU: `apptainer exec --nv` on `uger-gpu-d001` or `d002` with `-l gpu=TRUE,operating_system=RedHat8`; the GPU choice of the job script with two jobs on one node; a StarDist `segment` task with `device: cuda` (E-2) | `tiny_sample` and one real FOV | none | owner-run | `nvidia-smi` in the container; two jobs on different devices; the task's record names the device | one GPU job each, 16 GiB, 1 h | — |
| U6 | Fiji: the `bigstitcher_engine` task on UGER with the lab's Fiji inside the image (E-5); a macro whose Fiji run fails is caught by `bigstitcher-check` and retried | `tiny_sample` (`after-fusion` with Fiji) | none | owner-run | `dataset.xml` with a Stitching Transform per view; the failed case exits 1 | 8 GiB, 1 h | — |
| U7 | Retries: a task given too little memory is killed by `h_vmem` and succeeds on its second attempt with twice the request | one real FOV | none | owner-run | Two attempts; the second succeeds | as the task | — |
| U8 | The MATLAB legacy workflow (W-208's Broad MATLAB run): `backend: matlab`, `matlab_launcher: broad`, the frozen profile and environment, `tests/tissue_2D_test.yaml` adapted to one real FOV | one real FOV | none | owner-run | The MATLAB jobs succeed; outputs in today's layout | MATLAB jobs 8 GiB | — |
| U9 | W-212: the Python and MATLAB stage outputs on current dev compared on the same FOV, with the committed MATLAB reference outputs | U8's FOV | none | owner-run | W-212's own criteria | W-212's | — |
| U10 | The shared environment from the lock on a RHEL 8 node (E-3), with its checks | — | none | owner-run | `uv pip check`; `starfinder --help`; Snakemake 9.15.0 | one job, 8 GiB | — |

## Implemented checks

The tests that implement each row, planned; the issue that lands each test fills the module names in.
A test of the validation issue keeps this table and the check tables in step, read with
`docs_pages.design("validation/orchestration.md")`.

(orchestration-validation-implemented-checks)=

| # | Module and tests | Selection | Issue |
| --- | --- | --- | --- |
| C1 | `test_workflow_plan.py` (planned) | Q | TG2-1 |
| C2 | `test_workflow_plan.py` (planned) | Q | TG2-1 |
| C3 | `test_workflow_declared_outputs.py` (planned) | Q | TG2-1, TG2-2, TG2-3 |
| C4 | `test_workflow_snakemake_dryrun.py` (planned) | Slow `other` | TG2-4 |
| C5 | `test_workflow_plan_refusals.py` (planned) | Q | TG2-1 |
| C6 | `test_workflow_commands.py` (planned) | Q | TG2-1 |
| C7 | `test_workflow_completion.py` (planned) | Q | TG2-1 |
| C8 | `test_workflow_boundary.py` (planned) | Q | TG2-4, TG5 |
| C9 | `test_assignment_workflow.py`, `test_workflow_assembly_golden.py` (existing, import line changed); `test_workflow_snakemake_dryrun.py` (planned) | Q; Slow `other` | TG2-4 |
| N1 | `test_workflow_nextflow_stub.py` (planned) | `nextflow` | TG5 |
| N2 | `test_workflow_nextflow_stub.py` (planned) | `nextflow` | TG5 |
| N3 | `test_workflow_nextflow_stub.py` (planned) | `nextflow` | TG5 |
| G1 | `test_workflow_orchestration_golden.py` (W-642's draft) | Slow `other` | PREP |
| G2 | `test_workflow_orchestration_golden.py` (W-643's draft) | Slow `other` | PREP |
| G3 | `test_workflow_orchestration_golden.py` (planned: the refactored workflow) | Slow `other` | TG2-4 |
| X1 | `test_workflow_engines_e2e.py` (planned) | Extended | V |
| X2 | `test_workflow_engines_e2e.py` (planned) | Extended, Nextflow job | V |
| X3 | `test_workflow_engines_e2e.py` (planned) | Extended, Nextflow job | V |
| X4 | `test_workflow_core_golden.py` (planned) | Extended | PREP (values), TG2-1 (test) |
| X5 | `test_workflow_tiled.py` (planned) | Extended | TG2-2 |
| X6 | `test_workflow_tiled.py` (planned) | Extended | TG2-2 |
| X7 | `test_workflow_recovery.py` (planned) | Extended | V |
| X8 | `test_workflow_recovery.py` (planned) | Extended | TG2-1 |
| X9 | the validation issue's script (run once) | — | V |
| U1 to U10 | the UGER checklist's records | owner-run | TG4 |

## Marker and selection changes

* **`nextflow`**: a new kind marker for the tests that run Nextflow (N1 to N3, X2, X3), registered in
  `src/python/pyproject.toml` by task group 5. The CI gets a third job that installs Java 17 and
  Nextflow 26.10.4 and runs `-m nextflow`; elsewhere those tests skip with a stated reason, so they
  pass or skip in `pytest-ci-environment`.
* **The `workflow` group in CI**: both CI jobs add `--group workflow` with the lock change, so C4, C9's
  dry run and G1 to G3 run there instead of skipping.
* `scripts/selection_coverage.py` keeps covering every collected test once: the `nextflow` tests are
  also `slow` or `extended`, so they fall in the Slow or Extended selections, where they skip without
  Nextflow.

## Resource plan

| Layer | Per test | Where |
| --- | --- | --- |
| C | under 5 s, 4 GiB (C4 and C9's dry run `slow`, up to 1800 s) | every change (Q); Slow `other` |
| N | up to 1800 s, 4 GiB with the JVM | the Nextflow CI job |
| G | up to 1800 s per Snakemake run, 4 GiB; G1 about 60 s, G2 about 5 min, measured by W-642 and W-643 | Slow `other` |
| X | up to 1800 s, 4 GiB each; X9 under a grant | batch end and before merge |
| U | the owner's requests on UGER | task group 4 |

## Limitations

* No check here runs MATLAB, Fiji, a GPU or UGER except the human checklist.
* X3's equality holds within one environment; across environments and hosts the comparison is E05's.
* X6's proportions rest on a fixture of about 70 molecules until X9 confirms or replaces them.
