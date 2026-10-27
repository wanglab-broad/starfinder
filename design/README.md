# Design records

This folder holds STARfinder's design records: the detailed specifications that the
code and its tests follow, the validation designs, and the baselines of earlier
revisions. It is part of the repository but not of the documentation site, which
keeps to short guides and references for users (`docs/`).

Read the site first. Come here when you change a stage or a synthetic scene and need
the full rules behind it.

| Folder | Holds | Read by tests |
| --- | --- | --- |
| `specs/` | The specification of each stage, method family and synthetic scene: the rules the code follows, the options considered and why one was chosen. | No |
| `validation/` | The validation designs: the checks of a scene or a chain, their thresholds, and the table that maps each check to its tests. | Yes: the "Implemented checks" tables, found by their MyST labels |
| `baselines/` | What a stage did at a fixed revision before it was rewritten, with file and line pins. History; not maintained. | No |
| `proposals/` | Designs proposed and not, or not yet, implemented. | No |
| `notes/` | Longer notes behind a guide: calibration targets, research scripts, the full migration notes. | No |

## How this folder relates to the site

* A guide on the site says what a stage or scene does and how to use it. Its
  specification here says exactly how, and why.
* A table that users look up (presets, truth fields, configuration keys) lives once,
  on the site's reference pages or in the code. A specification here links to it
  rather than copying it.
* Decisions, reviews and measurements are recorded in Linear and in each run's
  records. A specification keeps the rule that a decision settled, not the
  discussion.

## Editing

The files are MyST Markdown, like the site's pages, but Sphinx does not build them:
a `{doc}` role here does not resolve. Link to a site page with its path, for example
`docs/synthetic-cellular-scene.md`, and to another record with a relative link.
Lines that start with `%` are source comments.

A record's history before it moved here is under its old path in `docs/`:
`git log --follow docs/<page>.md` shows it.
