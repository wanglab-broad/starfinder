# Test-suite slimming: an audit of the suite at v0.2.1 and a proposal

% Status: Applied (W-893 phase 2: W-897, W-898, W-899, W-900; 2026-11-21). Proposed by W-893's phase 1, the audit
% (2026-11-19); Jiahao decided by category on 2026-11-19 (W-893 comment 924b6549). The decisions are below.

This page proposes which tests of the suite to keep, merge, move to an evidence workload or retire now that Chapter II
development has tripled the suite, and how to split the shared test helpers. It changes no test and no package code
(W-893, decision D2). Jiahao decides by category, not test by test, on the review page of W-893's run; the decisions
are recorded under each category when they are made. **Move** means moving a test into an evidence workload that runs
on request, as W-536 and W-603 did, not deleting it. **Retire** is used only for a test whose assertion a faster test
of the suite makes on the same run or the same input.

## Decisions and phase 2

Jiahao's decisions of 2026-11-19 (W-893 comment 924b6549, the review page's `decisions`), and the issue of phase 2 that
applied each:

| Category | Decision | Applied by |
| --- | --- | --- |
| H1, H2: the helper splits | agreed | W-897 |
| K1, K2, K3, K7, K10 and option O2 | agreed | W-898 |
| K8 | only if the dry-run cases of `test_workflow_snakemake_dryrun.py` move onto K1's shared run too; they are `slow` contract tests on another configuration, so K8 is left unchanged and its saving dropped | W-898 |
| K4 (with the six S16 cases at seeds 101 and 102), K5, options O1 and O3 (the suite keeps Piscis `20251212`) | agreed | W-899 |
| K6 | rejected, superseded by O1 | W-899 (O1) |
| K9, S1, S2 and H3 | agreed: the rules in `docs/contributing.md` ("Test selections"), H3's `scripts/check_inputs.py` and `test/check_inputs.json`; the runner's configuration is the coordinating session's (W-901) | W-900 |
| `phase2-conditions` | each evidence script runs at the end of a batch whose changes touch its subsystem, and all of them before a tag or a release, on UGER through the coordinating session | W-898, W-899, W-900 |

The sections below are the proposal as phase 1 wrote it; its figures are projections. The measured figures of a full
validation after phase 2 are W-901's.

## What was measured

The suite at the batch base `76cd058` (`dev`, tag `v0.2.1`) collects 6,924 tests in 268 files
(`pytest --collect-only -m ""`, so that `extended` is collected too). The inventory has one row per node id, in
`inventory.csv` and `inventory.json` of W-893's run folder
(`runs/W-893/20261119T053344Z-e7eddfdf/`), with these columns: file, node id, function and parameter id, markers and
tier, the checks of a full validation that select it, run time with its source, the outcome, what it protects with the
reference, the commit that first added the function, the faster test that covers the same property and how that is
known, the category, the proposal with one line of reason, and the seconds a full validation saves.

**Run times.** In the order the batch guidance gives, with the source per row:

1. W-880's full validation of 2026-11-19 (`runs/W-880/20261118T211023Z-d079bfb1/validation-20261119T011126Z-bd320c57/`):
   every test of the extended and Nextflow checks, which ran with `--durations=0`, and the 50 slowest of the fast tier,
   slow-synthetic and slow-other, which ran with `--durations=50`;
2. W-874's validation (`runs/W-874/20261119T031917Z-4f2cd07d/validation-20261119T050227Z-2d3af7ac/`): the 50 slowest
   of its slow-other check;
3. W-893's own runs of the fast tier, the five slow groups and the six learned checks, with
   `--durations=0 --durations-min=0`, one process each, on hyper-threads beside busy cores and under the record
   wrapper's `strace`: they read high and rank the tests. Each own time is put on W-880's scale by the ratio of W-880's
   pytest session time of the same selection to the own run's (0.63 for the fast tier, 0.84 to 0.97 for the slow
   groups, 0.95 to 1.02 for the learned checks; W-874's 0.99). The step of the selection `learned and slow and not
   integration` stopped at the batch's 4 GiB bound, so its 26 tests were measured one by one afterwards, each in its own
   step, and taken as they are. Alone, a test loads its models and fixtures itself, so these times are an upper bound
   on what it costs inside a session.

A test with no measured time has the source `unmeasured`, with no number, and is left out of every saving and of the
projection; no selection's time is spread over its tests. One test is unmeasured:
`test_l12_cpu_and_gpu_labels_agree_within_the_w306_bounds`, which every source skipped (it needs a GPU Python).

A test's time is its setup, call and teardown. A module or session fixture's cost falls on the first test that uses it.

**What a test protects.** Each row names one of five kinds, with its reference:

| Kind | Rule | Tests |
| --- | --- | ---: |
| Specification row | The function is named in a row of a design record at the base (the validation tables map rows to tests), or its name carries a row id that a record's table holds | 3,007 |
| Contract | The `contract` or `golden` marker (an interface, a workflow key, pinned outputs), an identity contract kept by decision (GN1, GA1), or a `validation` test that no design record names and that checks an issue's criteria on the public API | 1,977 |
| Defect | The function was first added by a `fix` commit (the commit is named), or its docstring cites a W- issue as a regression | 559 |
| Development scaffolding | A pin of a base revision, parity with a moved implementation, or a repeated seed with the same assertion | 176 |
| Unit behavior | None of these: an ordinary unit test of its module, or a check of a test's own fixture | 1,205 |

Each row's reference names what is protected, read from the test itself. Where no design row names the test (by
name, or by a pattern such as `test_r1_*` in a row that names its file), the reference is what the test's code names
or reaches: the records and pages its docstring, its body, and the helpers and fixtures of its module it uses name or
read (`docs_pages.read`, `docs_pages.design`), else the package functions it reaches the same way, directly or
through the suite's shared helper modules (a public one with its page under `docs/api/`, an internal one marked as
such). Where the code names neither, a per-test citation names the page that documents what the test asserts (the
workflow guides and reference, the migration guide, `docs/contributing.md`, the docs builder code), or says that the
test checks only its own fixture ("none found", 8 rows). A citation that covers every test of a file is used only
where every test of that file asserts the cited thing (checked test by test; the run's README lists the check). No
row has a bare marker.

**Overlap** is named where it is known from the test code: the same assertion on another parameter (a seed, a
profile), the same assertion on stub tasks, or the same real run made by another test that asserts more. No coverage
run was made to find overlaps (see "Limitations").

## Where the time goes

W-880's full validation ran these checks (the runner's project configuration). The CPU minutes of a one-CPU check are
its wall minutes; Q ran on four CPUs with `-n 4`, and its CPU minutes are its user and system time.

| Check | Wall (min) | CPU (min) |
| --- | ---: | ---: |
| `pytest-quick` (`-n 4`) | 3.1 | 12.0 |
| `pytest-fast-tier` | 27.9 | 27.9 |
| `pytest-learned-slow-piscis-20230905` | 21.3 | 21.3 |
| `pytest-learned-slow-piscis-20251212` | 20.7 | 20.7 |
| `pytest-slow-synthetic` | 18.9 | 18.9 |
| `pytest-slow-other` | 16.1 | 16.1 |
| `pytest-learned-slow-other` | 5.6 | 5.6 |
| `pytest-learned-slow-chain-culture` | 5.3 | 5.3 |
| `pytest-learned-slow-chain-other` | 5.1 | 5.1 |
| `pytest-learned-fast` | 3.7 | 3.7 |
| `pytest-extended-recovery` | 33.6 | 33.6 |
| `pytest-extended-other` | 8.4 | 8.4 |
| `pytest-nextflow-recovery` | 43.4 | 43.4 |
| `pytest-nextflow-engines` | 25.2 | 25.2 |
| `pytest-nextflow-other` | 15.8 | 15.8 |
| `strict-docs` | 4.9 | 4.9 |
| `pytest-slow-benchmark`, `-registration`, `-spot-finding` | 2.8 | 2.8 |
| `reference-audit` | 0.0 | 0.0 |
| **Total** | 261.8 | **270.7 (4.51 CPU-h)** |

The issue's 4.35 CPU-hours count Q's 3.1 wall minutes once; its four CPUs make 12.0 CPU minutes. A `nextflow` test
runs only in the Nextflow checks: the other checks that select it skip it, because only the Nextflow checks put
Nextflow on `PATH`.

The 137 tests of 20 s or more take 190 of the 262 CPU minutes the tests themselves take in a full
validation. They are almost all real-engine workflow runs, the learned detectors' validation and the slow presets of
the synthetic scenes; the 6,200 tests of the fast tier take 28 minutes together.

## The workflow recovery and cache tests

These come first: they are 118 of the 271 CPU minutes (44 %) of a full validation, and a change to
`test/workflow_examples.py`, which they import, reruns all of them (W-880 changed three lines of it).

### What each test runs today

* **X7, X10 (Nextflow) and X11** (`test_workflow_recovery.py`) each run the whole `tiny_sample` per-fov workflow on a
  real engine before damaging or changing it, then rerun the affected tasks and compare with an uninterrupted or a
  fresh run. One complete run costs 61.2 s on Snakemake and 79.5 s on Nextflow (X1 and X2 per-fov in W-880's logs,
  which run exactly that). X11 `restored-twice` changes the processing block four times (A, B, A, B), each with a rerun
  and a fresh comparison: 540 s on Snakemake and 727 s on Nextflow.
* **X10 (Snakemake) and the export-record tests** already start every case from one complete run per engine, kept by
  a module fixture and restored between cases (W-760 and W-758).
* **The Nextflow cache test** (`test_workflow_nextflow_cache.py`) runs the same sequence (A, B, A, B of two blocks, two
  raw replacements and a memory-only change) on stub tasks (266 s) and on real tasks (720 s), with the same assertions:
  which tasks run, their tokens, and the unchanged records.
* **X1, X2 and X3** (`test_workflow_engines_e2e.py`): X3 runs a variant on Snakemake, then on Nextflow, and compares
  the two output trees byte for byte; X1 and X2 run per-fov and molecules-only on one engine each through the same
  `EngineRun.invoke` that X3 calls, with the same assertions. `test_real_per_fov_commands`
  (`test_workflow_nextflow_commands.py`) runs the Nextflow per-fov workflow too, and asserts two things X3 does not:
  the exact set of task kinds, and workflow tokens that start with `rerun:` (X3's comparison leaves the token out).
* **N1** (`test_workflow_nextflow_stub.py`) runs each of eight variants on stub tasks under the `local` profile and
  again under `google-batch,local`.

### Proposal

| Category | Tests | Proposal | Saved (CPU-min) |
| --- | --- | --- | ---: |
| K1: the case matrix of X7, X10 (Nextflow) and X11 | 18 in `test_workflow_recovery.py` | **Merge**: one complete run per engine, shared by the module and restored between cases (the W-758 and W-760 pattern), so each case runs only its damage or change, the rerun and the comparison. **Move** X11 `restored-twice`, `raw-size` and `raw-older` on Nextflow to the engine evidence workload: the stub cache test makes the same sequence on Nextflow (A, B, A, B of two blocks, the two raw replacements). Their Snakemake cases stay, since no stub test covers Snakemake's reruns, and keep the fresh-output comparison; Snakemake's `restored-twice` reuses the fresh runs of configurations A (the shared run) and B (the `processing` case's fresh run), which the module keeps at the same paths | 39.3 |
| K2: the cache test's `[real]` variant | 1 | **Move** to the engine evidence workload; `[stub]` stays | 12.0 |
| K3: X1 and X2 on per-fov and molecules-only, the uncalibrated Nextflow run | 5 | **Retire** the first four (X3 makes the same `EngineRun.invoke` calls with the same assertions, and compares the runs); **move** the uncalibrated Nextflow run (the Snakemake one asserts the same grids on the same commands). **Keep** `test_real_per_fov_commands`: its task-kind set and `rerun:` token checks have no retained replacement, and moving them into X3 would change X3, which compares without the token, for 1.4 CPU-min | 4.9 |
| K8: the Snakemake dry run after a complete per-fov run | 1 | **Merge** its dry-run assertion into K1's shared Snakemake run | 0.9 |
| K10: N1 under `google-batch,local` for seven of its eight variants | 7 | **Retire**; the per-fov run under that profile stays, and `test_workflow_nextflow_google_batch.py` checks the profile offline | 2.4 |

The shared run adds one complete run per engine back to the module (61.2 s and 79.5 s), counted in K1's figure.

### The real and the stub runs per engine and route that remain

| Route | Snakemake, real tasks | Nextflow, real tasks | Stub tasks or dry runs |
| --- | --- | --- | --- |
| per-fov | X3 (with Nextflow); K1's shared run with X7, X10, X11 and the dry run; the export-record run; the held export (`test_workflow_engines_review.py`); the uncalibrated direct run | X3; `test_real_per_fov_commands`; K1's shared run with X7, X10, X11; the export-record run; the held export | N1, N2, N3 and the cache test on Nextflow stubs; the Snakemake dry runs |
| molecules-only | X3 | X3 | N1 |
| supplied-xml | none (as today) | X2 `supplied-xml`; the supplied BigStitcher import command | N1 |
| tiled processing | the uncalibrated tiled run; the tiled dry run after a complete tiled run | `test_real_tiled_commands` (with its `-resume`) | N1 (`after-fusion`) |
| after-fusion, morphology-only, reference-only | none (as today: Fiji and the learned segmenter are not run by the suite) | none (as today) | N1 |

Every engine and route that has a real run today keeps at least one.

## The other categories

| Category | Tests | Proposal | Saved (CPU-min) |
| --- | --- | --- | ---: |
| K4: spot-finding validation, seeds 101 and 102 of S1 to S8 and S10 | 164 in `test_spot_finding_validation.py` (learned and classical) | **Move** to a spot-finding evidence workload; seed 100 stays, and so does S10's Piscis seed 102, the only seed that passes there (100 and 101 are expected failures). `design/specs/spot-finding-algorithms.md` parametrizes each check by seeds 100 to 102 with the same assertion, so its rows change with this (a decision on the design, not only on the suite). The other expected failures recorded per seed (S4, S10) move with their seeds | 24.8 |
| K5: GN5 and GN7 on the three slow presets | 6 in `test_synthetic_cellular_presets.py` | **Move** to the scene's evidence workload: the same checks run on the three routine presets in the fast tier, and the slow presets' generation is most of the time | 6.2 |
| K6: GA1 digests of the three slow context presets | 3 in `test_synthetic_appearance_golden.py` | **Keep** (an identity contract kept by decision); moving them would save 6.9 CPU-min (option O1) and needs Jiahao's decision | 0 |
| K7: G3, the refactored workflow against the values of the workflow before the dispatcher | 28 in `test_workflow_orchestration_golden.py`: G3's two runs and the 26 tests of its own comparison (`compare_migrated` of `scripts/workflow_golden.py`) | **Move** to an orchestration evidence workload with its expected folders: a pin of a finished refactor (W-647). The task commands' values are pinned by `test_workflow_core_golden.py`, `test_workflow_cell_golden.py` and `test_workflow_stitching_golden.py`, and every default change breaks G3 (both failed in W-880's validation) | 2.8 |
| K9: Q at a full validation | the `pytest-quick` check | **Runner**: skip `pytest-quick` when `pytest-fast-tier` runs in the same validation (R = Q ∪ V); Q stays the gate of every other change | 12.0 |
| K0: everything else | 6,691 | **Keep**: contracts, specification rows, defects and unit tests, each under a few seconds on average | 0 |

Development scaffolding that stays: ten tests are pins of a base revision or parity checks of a finished
move (for example `test_paths_keep_the_attributes_of_the_base_revision`, the two `*_the_base_revisions` option checks
of `test_workflow_nextflow_sge.py`), each well under a second; they are listed in the inventory and can be retired
when their module is next changed, without a saving worth a decision now. `test_the_grid_equals_the_base_revisions`
(92 tests) is named after the base revision but holds explicit expected values since W-574, so it is a contract.

## Shared helpers

`test/` has 40 helper modules. The runner's check inputs name them by glob (`src/python/test/workflow_*.py`,
`src/python/test/test_workflow_*`), so a change to any of them reruns every check whose inputs match. From the import
graph of the test modules at the base:

| Helper | Test files that import it, directly or through another module | Their tests | Their CPU in a full validation (min) |
| --- | ---: | ---: | ---: |
| `workflow_tasks.py` | 62 | 2,361 | 184.2 |
| `workflow_examples.py` | 41 | 1,266 | 131.6 |
| `workflow_nextflow.py` | 11 | 188 | 122.1 |
| `tiled_fixtures.py` | 9 | 206 | 96.6 |
| `test_workflow_snakemake_dryrun.py` (a test module that helpers import) | 6 | 141 | 93.0 |
| `workflow_engine_validation.py` | 4 | 115 | 90.3 |
| `coordination_helpers.py` | 86 | 3,371 | 92.2 |
| `expected_values.py` | 34 | 1,438 | 57.4 |

The full table, with the commits since 2026-10-01 that changed each helper, is in the run's
`results/analysis/summary.json` (`helpers`), and each check's import closure under `check_inputs`.

**H1: split `test/workflow_examples.py`** (imported, directly or through another module, by 41 test files with
1,266 tests). Its importers use three groups of names:

* the fixture and its plumbing: `golden`, `input_root`, `output_root`, `write_supplied`, `write_config`, `fix_code`,
  `CLEAN_CODE`, `run_task` and the constants `ROOT`, `SAMPLE`, `FOVS`, `DATASET_ID`: to `workflow_fixture.py`;
* the configurations: `PROCESSING`, `dataset`, `multiview`, `grid`, `variants`, `single_fov`, `KINDS`, `VARIANTS`,
  `REFUSED`, `FIJI`: to `workflow_variants.py`;
* the writer of C1's expected manifests (`normalized`, `write_expected`, `EXPECTED`, `PLACEHOLDERS`), which only
  `test_workflow_plan.py` uses: into that file.

**H2: break the import chain of the engine checks.** `workflow_engine_validation.py` imports `TRIGGERS`, `_snakemake`,
`downstream` and `snakemake` from the test module `test_workflow_snakemake_dryrun.py`, which imports
`test_fov_membership.py` and `test_workflow_plan.py`, which import `tiled_fixtures.py` and `workflow_tasks.py`. The
recovery and engine checks therefore depend on four test modules and two helpers they never call. The four Snakemake
helpers move to `workflow_snakemake.py`.

**H3: name each check's inputs by its import closure.** Each heavy check's runner inputs list the test files it runs
and the modules they import (a script computes the closure from the import graph, as this audit's analysis does, and a
test keeps the configuration equal to it), instead of the globs. With H1 and H2, a change to `workflow_tasks.py`
(imported by 62 test files, none of them a recovery or engine test) no longer reruns the recovery and engine
checks, and a change to the C1 writer or a plan-only constant reruns only the fast tier.

**S1: split the fast tier into two checks** of balanced files (13.8 and 13.8 minutes), so that no check exceeds
20 minutes. The tests and their total time do not change.

**S2: split `pytest-extended-recovery` into two checks** by test: after K1 it still takes about 23.5 minutes
(Snakemake's X10 and X11 and the export-record tests); its largest remaining test, X11 `restored-twice` on Snakemake,
takes about 4.

What the splits do not do: a change to the processing values that the real-engine tests run (as W-880's three lines)
still reruns the real-engine checks, which is right, since those runs depend on them; K1, K2 and K3 make those checks
61 minutes instead of 118.

## Projection against the targets

D3's targets: one full validation in about 2.5 CPU-hours or less, at most 45 minutes of wall time on four check CPUs,
no single check above 20 minutes, Q unchanged or faster.

| Check | W-880 (min) | Saved (min) | Projected (min) |
| --- | ---: | ---: | ---: |
| `pytest-quick` | 12.0 (CPU) | all (K9) | 0.0 |
| `pytest-fast-tier` (S1: two parts) | 27.9 | 0.9 | 27.0 |
| `pytest-learned-slow-piscis-20230905` | 21.3 | 10.4 | 10.8 |
| `pytest-learned-slow-piscis-20251212` | 20.7 | 10.2 | 10.5 |
| `pytest-slow-synthetic` | 18.9 | 6.2 | 12.8 |
| `pytest-slow-other` | 16.1 | 3.3 | 12.7 |
| `pytest-learned-slow-other` | 5.6 | 2.4 | 3.1 |
| `pytest-learned-slow-chain-culture`, `-chain-other` | 10.4 | 0.0 | 10.4 |
| `pytest-learned-fast` | 3.7 | 1.2 | 2.5 |
| `pytest-extended-recovery` (S2: two parts) | 33.6 | 10.2 | 23.5 |
| `pytest-extended-other` | 8.4 | 1.5 | 6.9 |
| `pytest-nextflow-recovery` | 43.4 | 29.1 | 14.3 |
| `pytest-nextflow-engines` | 25.2 | 15.4 | 9.8 |
| `pytest-nextflow-other` | 15.8 | 2.4 | 13.5 |
| `strict-docs`, slow benchmark, registration and spot finding, reference audit | 7.7 | 0.0 | 7.7 |
| **Total** | **270.7 (4.51 CPU-h)** | | **166 (2.76 CPU-h)** |

| | CPU | Wall on four CPUs | Largest check | Q |
| --- | ---: | ---: | --- | --- |
| D3's target | about 2.5 CPU-h or less | 45 min or less | 20 min or less | unchanged or faster |
| W-880, measured | 4.51 CPU-h | 69.1 min placed as below (78 min as the controller ran it) | `pytest-nextflow-recovery` 43.4 min | 3.1 min |
| The categories K1 to K10 with S1 and S2 | **2.76 CPU-h** | **41.8 min** | `pytest-nextflow-recovery` 14.3 min | faster (K7 takes 26 tests out of Q) |
| … and option O1 | 2.64 CPU-h | 40.0 min | 14.3 min | |
| … and option O3 | 2.59 CPU-h | 39.0 min | 14.3 min | |
| … and options O1 and O2 | 2.58 CPU-h | 39.0 min | 14.3 min | |
| … and options O1 and O3 | about 2.48 CPU-h | | 14.3 min | |
| … and options O1, O2 and O3 | **2.41 CPU-h** | 36.4 min | 14.3 min | |

The options, each Jiahao's decision on its own:

* **O1** moves K6's three GA1 slow-preset digests (6.9 CPU-min): it weakens the suite's hold on an identity contract
  kept by decision; the routine presets' digests stay in Q.
* **O2** moves X11 `restored-twice` on Snakemake too (3.9 CPU-min after K1): the suite would then hold no A, B, A, B
  rerun check on Snakemake; X11 `processing` still checks one change and its rerun on both engines.
* **O3** keeps the spot-finding validation rows (S1 to S15 of `test_spot_finding_validation.py`) on one Piscis model,
  `20251212`, and moves those of `20230905` (26 tests, 10.0 CPU-min after K4): the case of "several models where one
  would do" of D1. Both models' weight identity and recorded settings (`test_piscis.py`) stay.

**Method.** Per check: W-880's measured minutes, minus the seconds each proposed change saves among the tests that
check runs (on W-880's scale; measured times only, so an unmeasured test saves nothing), plus the shared runs K1 adds
back; Q is not run at a full validation (K9); the fast tier and extended-recovery are split into two checks each (S1,
S2). Wall time: the checks placed longest first on four CPUs. The measured wall time placed the same way is 69.1
minutes (W-880's controller ran it in 78). Q loses the 26 tests of K7 and is otherwise unchanged.

Of the savings, 2.4 CPU-min come from the 26 tests measured one by one (K4's Spotiflow seeds in
`pytest-learned-slow-other`); as those times are upper bounds, that check's projected 3.1 minutes may be low by up to
that much.

**Result.** D3's targets are not all met by the categories alone. They meet three of the four: 41.8 minutes of wall
time, no check above 20 minutes (with S1 and S2), and a faster Q. They bring the CPU time from 4.51 to 2.76 CPU-hours
(−39 %), not to 2.5. The CPU target is met only with further options, which are Jiahao's decisions: O1 and O3
together (about 2.48 CPU-hours, the options' savings being additive) or all three (2.41). O2 and O3 without O1 give
about 2.53, and no single option is enough.

## Phase 2

Phase 2 applies the approved categories, updates the runner's check inputs and `docs/contributing.md` ("Test
selections"), and follows the safety rules of W-893:

* no identity contract weakened without Jiahao's decision (K6 stays unless he decides otherwise);
* no tolerance widened and no expected value changed;
* `scripts/selection_coverage.py` still holds;
* the line coverage of `src/python/starfinder` reported before and after, per subpackage;
* every retired test listed with its reason (the inventory's rows of the retired categories);
* any AGENTS.md change through an administrative issue.

The design records that name moved or retired tests change with them: the validation tables that map rows to tests
(`design/validation/orchestration.md` for X1 to X3, X7, X10, X11 and G3; `design/specs/spot-finding-algorithms.md` for
the seeds), and the tests that check those tables name collected tests. The evidence workloads are new scripts beside
W-536's and W-603's: the workflow engines (K1's moved cases, K2, the uncalibrated Nextflow run), the spot-finding seeds
(K4), the synthetic scene's slow presets (K5), and the orchestration golden G3 (K7). Where they run follows W-889's rule
for heavy evidence.

## Appendix: tests that unmerged branches add or change

From `git diff --stat 76cd058 <branch> -- src/python/test`, read only (the branches were not checked out or run):

| Branch (revision) | Files | Tests it adds or changes |
| --- | --- | --- |
| `runner/s214-prep-20261118` (`c0d5277`: W-880 and W-874) | 16 files, +603 −93 | new `test_spot_finding_default.py` (+231); `test_stitching_multiview.py` (+125 −3) and `test_stitching_placement.py` (+96 −8) for W-874; `test_spot_finding_noise_estimation.py` (+43 −38) and eight spot-finding, readout and registration files that name the noise threshold the default no longer gives; `test_report_page.py` (+39); `workflow_examples.py` (+2 −1) and `readout_scenes.py` (+5 −2), helpers |
| `runner/s214-demons-20261118` (`a1aa699`) | 1 file, +137 | `test_demons.py` (+137) |
| `runner/s214-w886-20261119` (`f758177`) | 1 file, +137 | `test_demons.py` (+137), the same lines as the demons branch |

None of them changes a test this proposal moves or retires. The new `test_spot_finding_default.py` (W-880, 14 tests) is
a contract of the new default and stays; its times are not in this inventory.

## Limitations

* Run times from W-893's own runs were measured on hyper-threads beside cores that other batches' checks used, and
  under `strace`; they read high. They rank the tests and are put on W-880's scale by one ratio per selection; a time
  from W-880's logs wins wherever it exists.
* W-880's validation ran on `runner/s214-prep-20261118` (W-880's change on the base); its tests that the base does not
  have are not in the inventory, and a test the change modified keeps its W-880 time.
* Two of W-893's own runs differ from the controller's. The steps' `PATH` lacked nvm's Node 24, so 37 tests that run
  the report page's script failed on the system's Node 10; they were rerun by node id with Node 24 and all passed, and
  their times are from that rerun. The step of the selection `learned and slow and not integration` stopped at the
  batch's 4 GiB bound (summed over its process tree) during its third test; it was not retried with a higher limit,
  and its 26 tests were measured one by one (upper bounds, above).
* One test is unmeasured (skipped in every source) and saves nothing in the projection.
* The savings of K1 and K8 are estimates from the cost of one complete run per engine (X1 and X2 per-fov); phase 2
  measures them.
* Overlap is named from the test code (the same assertion, the same run); no coverage comparison of pairs of tests was
  run. Phase 2 reports line coverage before and after.
* "What it protects" is assigned by rules on the markers, the design records at the base, the commit that first added
  the function and its docstring; a row's kind is the first rule that applies, so a regression test that is also a
  specification row is listed as a defect, with the row named.
