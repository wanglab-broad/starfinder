# Retire the approved Benchmark lifecycle and expose experiment commands

**Status: Proposed — draft issue from W-771; create only after W-775.**

## Purpose

Implement the retirement slice approved by [W-775](https://linear.app/jiahaoh/issue/W-775)
from the [benchmark contract](../../specs/benchmark.md), replacing the old public
lifecycle with explicit expansion, one-process trials and evaluate-only access to
workflow outputs/checkpoints. The §2.13 workflow remains the executor.

Shared contract: `starfinder/development.md` as pinned by the batch configuration.
The baseline is `7cd1e7d`; refresh all R1–R12 searches at dispatch. Do not assume
this draft itself authorizes every row to retire.

## Deliverables

* Public `starfinder experiment expand`, `starfinder trial run`, and
  `starfinder experiment evaluate` commands with the approved flags and records.
  The report command belongs to the report-template draft and must land before
  its old provider is removed.
* Read-only evaluation adapters for workflow outputs, trial outputs and
  `dataset.load_checkpoint` results, calling pure package metrics.
* Gate-approved R1/R2/R3 removals and R7 synthetic helper relocation, with maintained
  CLI/examples/imports updated and original source archived by immutable revision
  and checksum. Preserve `synthetic/_storage.py`'s existing responsibilities.
* R4/R5 removal only once their explicit dependencies below are supplied; keep
  R6/R8/R9/R10 and the default R11 retention. Extract R8 comparison to the package
  without retiring the golden script or its historical fixtures.
* Complete [baseline test dispositions](../../baselines/benchmark.md), including
  synthetic generation, MIP axes, resource scopes, undefined values, saved-only
  evaluation and historical-source preservation.

## Acceptance criteria

- [ ] A saved W-775 approval names the exact accepted spec revision, permitted retirement rows and resolutions of open metric-role/M55 questions; the refreshed repository-wide inventory names every remaining caller, classifying the unrelated CI recipe hit and including both notebooks. Evidence: approval and saved searches compared with the baseline table.
- [ ] Public commands implement the approved expansion/trial/evaluation records with help and explicit failure statuses, and no second scheduler or new workflow task kind. Evidence: `test_experiment_cli.py`, existing `test_benchmark_trial.py` plus the named additions, and diff.
- [ ] Evaluation of workflow outputs and all three checkpoint stages invokes no processing and leaves source bytes unchanged; consumed files are hash verified, v1/v2 checkpoint limitations explicit, and failed/undefined metrics distinct. Evidence: `test_experiment_evaluate.py` with processing functions patched to raise, source hashes and corrupt-input fixtures.
- [ ] R1/R2/R3 providers are removed only after every baseline test has its named replacement or reviewed drop reason; no compatibility shim or rewritten golden values is introduced. Evidence: repeated searches, test-disposition ledger and focused test logs; old-layout assertions may change only as individually named in the baseline table.
- [ ] R7 preserves synthetic command outputs, manifests, uint8/uint16 layouts, no-overwrite behavior and dataset/truth identity after relocation. Evidence: base-versus-result bounded fixture hashes and retained `test_benchmark_synthetic.py`, `test_synthetic_amplicon_dataset.py`, `test_synthetic_contracts.py`; only imports and the old lifecycle tail may migrate as named.
- [ ] R4 waits for the new report and package fixture/metric callers; R5 waits for an approved useful-metric/diagnostic extraction inventory and saved-stack/montage callers to migrate. Evidence: dependency completion and baseline replacements; no remaining imports of deleted providers.
- [ ] R8 comparison lives in package code with identity and per-quantity numerical policies kept distinct; R9/R10/X9 and G1 remain usable. Evidence: comparator known answers and retained workflow golden, image-statistics, X9 argument and MATLAB stage-comparison Python tests; no claim of unexecuted MATLAB coverage.
- [ ] Source archives and broken-notebook limitations are explicit, site migration is coordinated with the site draft, and no deployment guard changes. Evidence: archive checksums, diff and explicit limitations for historical notebook cells.

## Exclusions

* No new E02/E04 adapter or missing short-set metric unless separately scoped at
  dispatch. These remain prerequisites to the corresponding experiment, not a
  reason to claim that the whole short set is executable after CLI retirement.
* No R9/R10 deletion, automatic R11 deletion, scientific runs, MATLAB changes,
  real-data threshold sweeps, default changes or new dependencies.
* No notebook execution/output rewrite. The two historical QC notebooks use old
  lifecycle cells; current policy permits import edits only. Archive them with
  their original environment/revision and report non-runnable old cells until a
  separately authorized notebook migration. No shim is justified by them.
* No `.github/workflows/docs.yml` change: its broad recipe-name hit invokes
  `docs/examples/recipes.py`, which does not import the retiring benchmark recipe
  builder. Deployment guards remain unchanged.

## Context

* W-769 and W-770 supply private trials/expansion and moved metrics; no public
  replacement evaluate-only command exists at this baseline.
* [Report-template draft](report-template.md) is a prerequisite for R1/R3/R4
  reporting retirement; [site-section draft](site-section.md) follows landed
  behavior. Stage the change so providers survive until their consumers migrate.
* R5 extraction does not promote every old research diagnostic to a selected
  endpoint; retain only explicitly approved reusable behavior.
