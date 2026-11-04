# Results

An evaluation of saved sources is kept in its own folder: `evaluation.json`
(sources, consumed files and their SHA-256, metrics with their outcomes, values,
units and reasons), `values.csv` (one row per metric value) and `metrics/` (each
computed metric's details). The {doc}`experiments guide <experiments>` shows how
to create it, and the {doc}`commands reference <commands>` lists its fields.
`starfinder experiment report` renders it in a folder of its own, outside Git:
`report.html`, `report.json`, `tables/` and `figures/`; the
{doc}`report guide <report>` describes the page. Results describe the supplied
sources and metrics; they do not establish accuracy on other inputs.
