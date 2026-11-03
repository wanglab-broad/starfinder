# Add the saved-evaluation benchmark report template

**Status: Proposed — draft issue from W-771; create only after W-775.**

## Purpose

Implement a benchmark report from saved evaluation records using W-237's human
review format, so an owner can understand the experiment before inspecting its
provenance. This is a benchmark evaluation report, distinct from the existing
`starfinder.report` sample-processing report.

Shared contract: `starfinder/development.md` as pinned by the batch configuration.

## Deliverables

* `starfinder experiment report --evaluation DIR --output DIR`, a package renderer
  over the approved evaluation schema, and a bounded saved-record example.
* A human summary in this exact order: setup; one card per method or mode;
  cross-cutting findings/anomalies; reading order with links to figures/cards.
  Then method details/figures and an appendix with full evidence tables.
* Setup states input/truth provenance, task/trial/checkpoint route, method config,
  conditions, metric meanings/populations, seeds, dtype/mode, fixed cut-offs,
  missing prerequisites, resources and the pilot/scientific qualification.
* Each method card states its purpose, targeted conditions, key before/after
  selected metrics with seed/FOV ranges, supplied precondition/harm results,
  failures, null reasons and limitations. A verdict is shown only when the
  approved protocol supplied one, with its failing clause.
* Findings name their source table and are labelled “finding” or “hypothesis”;
  appendix retains identity/revision/checksums, per-seed/FOV values, full tables,
  diagnostics, resource scope, omissions and reductions. Generated HTML, figures,
  JSON and CSV live outside Git.

## Acceptance criteria

- [ ] The report follows the W-237 order and every card/figure has a reading-order link. Evidence: bounded HTML and screenshots plus `test_experiment_report.py::test_calibrated_cards_follow_the_page_order_and_cover_every_method` migrated from the baseline.
- [ ] Every selected metric retains its units, denominator, condition/seed/FOV, null reason and source table; complete tables include failed and unpaired rows. Evidence: synthetic saved evaluation with known values and report/CSV checks, including the baseline missing/resource-scope replacement.
- [ ] Processing and evaluation entry points patched to raise do not affect rendering; all displayed images/detections are already saved and hash verified, and original files stay unchanged. Evidence: source hashes and `test_experiment_report.py` saved-only/corrupt-input tests.
- [ ] The report never chooses an optimum threshold, silently pools incompatible methods, hides missing FOVs or equates assigned reads with accuracy. Evidence: fixture with two fixed modes, missing values, differing resource scopes and a gene-error population; assertions and inspected cards.
- [ ] W-237 setup, method purpose, before/after ranges, caveats, findings versus hypotheses and appendix survive the transition; the dropped development-seed sweep/flag protocol is not reinstated. Evidence: content review against `design/specs/preprocessing-algorithms.md:668–704` and the approved short set.
- [ ] Every affected `test_preprocessing_report.py` test and old saved-report test has the named baseline replacement or reviewed reason; no tolerance/golden change makes it pass. Evidence: test-disposition ledger and focused checks; old renderer imports and saved layout assertions may change only to the approved evaluation schema.
- [ ] The bounded example writes its report and capture artifacts outside Git and demonstrates visible undefined/error states and readable full tables. Evidence: exact command, measured time/RSS, HTML, captures and output checksums supplied to the owner.

## Exclusions

* No data processing or detection rerun in rendering, no synthetic regeneration
  from a seed as a substitute for saved image identity, no scientific verdict
  invented by a renderer.
* No report-site publication or deletion of old reporting providers until this
  replacement and their consumers pass review.
* No changes to the sample report's schemas or algorithms.

## Context

* W-237's format is preserved in `design/specs/preprocessing-algorithms.md:668–704`
  and `design/notes/benchmark-recipes.md:218–229`; the existing renderer labels
  it explicitly in `benchmarks/preprocessing_report.py:2921–2924`.
* The historical format contains threshold-selected values and precondition
  flags. The new short set drops development-seed threshold selection and
  real-data sweeps; preserve the review structure while presenting frozen
  parameters and supplied evidence, without importing the old selection logic.
* Depends on the evaluate-only schema and [W-775](https://linear.app/jiahaoh/issue/W-775).
  Can land before deletion in [retirement-and-commands](retirement-and-commands.md).
