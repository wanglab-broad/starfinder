# Design records

This folder holds STARfinder's design records: the detailed specifications that the
code and its tests follow, the validation designs, and the baselines of earlier
revisions. It is part of the repository but not of the documentation site, which
keeps to short guides and references for users (`docs/`).

Read the site first. Come here when you change a stage or a synthetic scene and need
the full rules behind it.

| Folder | Holds | Read by tests |
| --- | --- | --- |
| `specs/` | The specification of each stage, method family and synthetic scene: the rules the code follows, the options considered and why one was chosen. | `test_report_docs_contract.py` reads the report layout/check tables; `test_segmentation_docs_contract.py` reads the assignment check table; `test_sample_segmentation_workflow.py` reads workflow configuration |
| `specs/synthetic/` | Amplicon, sample, cellular and real-appearance scene specifications. | No direct page reader; their validation tables are read from `validation/` |
| `validation/` | The validation designs: the checks of a scene or a chain, their thresholds, and the table that maps each check to its tests. | `test_synthetic_cellular_checks.py`, `test_synthetic_appearance_checks.py` and `test_stitching_docs_contract.py` read labelled check tables |
| `baselines/` | What a stage did at a fixed revision before it was rewritten, with file and line pins. History; not maintained. | No |
| `proposals/` | Designs proposed and not, or not yet, implemented. | No |
| `notes/` | Longer notes behind a guide: calibration targets, API behaviour, research scripts, the full migration notes. | No |
| `notes/benchmark-navigation/` | Preserved benchmark section, data, experiment and result navigation records. | No |

## How this folder relates to the site

* A guide on the site says what a stage or scene does and how to use it. Its
  specification here says exactly how, and why.
* A table that users look up (presets, truth fields, configuration keys) lives once,
  on the site's reference pages or in the code. A specification here links to it
  rather than copying it.
* Decisions, reviews and measurements are recorded in Linear and in each run's
  records. A specification keeps the rule that a decision settled, not the
  discussion.

## Writing a new specification

A section's specification batch writes its records here, not under `docs/`:

| Record | Path |
| --- | --- |
| The baseline: what the code and the legacy pipeline do today, every discrepancy, a proposed resolution | `baselines/<section>.md` |
| The contract: entries, inputs, outputs, records, configuration | `specs/<section>.md` |
| The algorithms: each method's behaviour, resources and pinning tests | `specs/<section>-algorithms.md` |
| The validation design: fixtures, checks, tolerances, and the table of implemented checks | `validation/<section>.md` |
| A design proposed and not yet adopted | `proposals/<name>.md` |

* Mark records Proposed until the specification gate approves them at an exact commit.
  Approval publishes nothing.
* A table that a test parses carries a MyST label right before it, `(label)=`, and the
  test reads it with `docs_pages.design("<path under design/>")`. Keep the label and the
  table's row shape when you edit it.
* Keep decisions, options considered and evidence in Linear and the run records. A
  record states the rule that was settled and links where it was decided.

When the implementation lands, the batch's last issue writes the site pages from the
implemented code: a short guide and a reference under `docs/`. It does not copy the
specification. The guide is short and follows a fixed set of sections. The reference
holds tables, generated from the code where it can be (`docs/_ext/`). The guide links
its records here by their GitHub URL, and `docs/contributing.md` describes the page
model. When behaviour changes later, update the record and the guide together.

## Editing

The files are MyST Markdown, like the site's pages, but Sphinx does not build them:
a `{doc}` role here does not resolve. Link to a site page with its path, for example
`docs/synthetic-cellular-scene.md`, and to another record with a relative link.
Lines that start with `%` are source comments.

A record's history before it moved here is under its old path in `docs/`:
`git log --follow docs/<page>.md` shows it.
