# Results

An evaluation of saved sources is kept in its own folder: `evaluation.json`
(sources, consumed files and their SHA-256, metrics with their outcomes, values,
units and reasons), `values.csv` (one row per metric value) and `metrics/` (each
computed metric's details). The {doc}`experiments guide <experiments>` shows how
to create it, and the {doc}`commands reference <commands>` lists its fields.

The older benchmark lifecycle saves its report under its evaluation directory
as `trials.json` and `summary.csv`. The {doc}`lifecycle guide <../benchmark>`
shows how to create it, and the {doc}`recipe reference <../benchmark-recipes>`
lists its files. Results describe the supplied cases and metrics; they do not
establish accuracy on other inputs.
