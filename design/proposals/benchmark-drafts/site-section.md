# Rewrite the Benchmark site section from the approved implementation

**Status: Proposed — draft issue from W-771; create only after W-775 and implementation.**

## Purpose

Replace the site's old Benchmark lifecycle guidance with concise instructions
for the landed workflow/trial/evaluate-only/report implementation and complete
R12's stale-note retirement. The site describes what works at the implementation
commit; Proposed design records stay under `design/`.

Shared contract: `starfinder/development.md` as pinned by the batch configuration.

## Deliverables

* Rewritten `docs/benchmark/index.md`, `data.md`, `experiments.md`, `results.md`,
  plus coherent `docs/benchmark.md`, `docs/benchmark-recipes.md`, authored API
  lists, migration text and bounded examples as needed by the landed commands.
* A short guide using the site's maintained page model, with setup, command
  examples, saved workflow/checkpoint sources, metric ownership, failure/null
  interpretation, resource scopes and links to immutable design records.
* A reference derived from implemented schemas/configs and commands; no
  duplicate manually maintained metric/config table where code can supply it.
* An immutable archive identity for `design/notes/benchmark-recipes.md` and all
  six `design/notes/benchmark-navigation/` pages before their authorized removal;
  updated incoming links and preserved historical scientific evidence links.
* R3 profile-directive cleanup coordinated with implementation, including
  `docs/_ext/benchmark_profiles.py` and its `docs/conf.py` registration where
  obsolete. The CI recipe-name hit is unrelated and requires no CI edit.

## Acceptance criteria

- [ ] Every documented public command and record field exists at the named implementation commit; missing adapters/metrics are explicitly unavailable. Evidence: file/line audit and bounded command logs, not specification text copied into docs.
- [ ] All four Benchmark section pages, both old standalone guides, API lists and incoming navigation links are coherent. Evidence: refreshed R12 repository-wide search, reference audit and strict documentation build; preserve `docs/check_reference.py`'s Benchmark navigation requirement.
- [ ] Every executable example uses the public landed API and writes new artifacts outside Git within project limits. Evidence: exact command/log/resource records and saved outputs; no real dataset or cluster run required.
- [ ] `design/notes/benchmark-recipes.md` and all six navigation notes have verified immutable archive paths/checksums before deletion, with live incoming links migrated. Evidence: archive manifest, before/after searches and diff; keep the W-771 spec/baseline/validation records.
- [ ] The guide distinguishes truth accuracy, output agreement and real-data proxies, and explains unavailable truth, missing scores, null denominators, frame/channel conversion and timing/RSS scopes. Evidence: guide review against implementation and bounded example results.
- [ ] The new report is linked as a saved-evaluation product, and W-212 remains the E14 coordinate/channel-mapping check. Evidence: guide/reference links and report example, not a claim that MATLAB execution or scientific acceptance occurred.
- [ ] No publication guard or dependency is altered; record commit/push/publication state honestly. Evidence: diff and controller checks; the current CI recipe example is not the retiring benchmark recipe builder.

## Exclusions

* No specification published under `docs/`, invented experiment results,
  rewritten historical baseline, new default or auto-selected threshold.
* No push, site deployment or publication; these require separate authorization.
* No notebook execution/output rewrite or MATLAB change.

## Context

* [Retirement baseline](../../baselines/benchmark.md), especially R3/R12 and the
  complete inventory of callers; [contract](../../specs/benchmark.md).
* Depends on the landed [command/retirement](retirement-and-commands.md) and
  [report-template](report-template.md) work, not only on W-775 approval.
* All six navigation notes are in scope even if they do not contain their own
  filename. Preserve archived experiment/data/result provenance before removing
  stale navigation.
