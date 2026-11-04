# Experiment commands reference

Look up the experiment and trial commands, the evaluation's and the report's
files and the saved records here. The {doc}`experiments guide <experiments>` and
the {doc}`report guide <report>` show how to use them.
The command, source and argument tables are generated from the package when
the site is built.

## Commands

```{experiment-commands}
```

## Records

| File | Format | Contents |
| --- | --- | --- |
| `manifest.json` of `experiment expand` | `starfinder.benchmark.experiment/1` | `source` (the experiment's absolute path and SHA-256) and `trials`, each `{path, sha256, configuration_sha256}` in expansion order |
| `trial-NNNNNN.json` | — | One resolved trial configuration, the input of `trial run` |
| `record.json` of `trial run` | `starfinder.benchmark.trial/1` | `configuration` and `configuration_sha256`, `seed`, `inputs` (resolved paths and SHA-256), `outputs` (relative `path`, `sha256`, `bytes`), `execution` (spot finding and segmentation: device and model), `status`, `error`, `wall_s`, `cpu_s`, `peak_rss_kib` |
| `evaluation.json` of `experiment evaluate` | `starfinder.experiment.evaluation/1` | Below |
| `values.csv` | — | One row per metric value: `metric`, `metric_id`, `function`, `outcome`, `status`, `value_name`, `value`, `unit`, `reason`; a metric that was not computed has one row with its reason |
| `metrics/NNN-<name>.json` | — | A computed metric's details (pairs, per-item comparisons) |
| `report.html` of `experiment report` | — | The review page: setup, cards, findings, reading order, method details and figures, appendix |
| `report.json` | `starfinder.experiment.report/1` | `evaluation` (path, SHA-256 of `evaluation.json` and `values.csv`, status), `revision`, `protocol` (path, SHA-256), `anchors` in page order, `cards`, `findings` (label, text, source table), `figures` (anchor, path, facts), `omitted`, `verified` (every file read, its SHA-256 and what it was checked against: a recorded SHA-256, or read only for `evaluation.json`, the details files and the protocol, which have none) and `outputs` (every other written file with its SHA-256 and bytes) |
| `tables/*.csv` of `experiment report` | — | `values.csv` (every row of the evaluation's `values.csv` with its method, mode, condition, seed, FOV, side, trial method, setup, counts and parameters), `pairs.csv` (before and after with each side's setup, unpaired rows kept), `resources.csv` (one row per source or task, with its scope), `sources.csv`, `consumed.csv`, `metrics.csv` and, for saved curves, `curves.csv` |
| `figures/*.png` of `experiment report` | — | Each card's key-metric dot plot (`<card>-metrics.png`) and each image figure the protocol names (`<card>-<n>.png`), `<card>` being the card's slug and 8-character key hash |

## Report protocol

The optional `--protocol` of `experiment report`; the
{doc}`report guide <report>` explains each key.

```{report-protocol}
```

## Sources

A sources file is a mapping with one list, `sources`. Every source has a unique
`name` and its `kind`; paths are relative to the sources file.
A SHA-256 is 64 lowercase hexadecimal characters, bare or prefixed `sha256:`.
A required hash or byte count that is missing or malformed fails its source
before any metric reads it.

```{evaluation-sources}
workflow: The plan's manifest and optionally the tasks to read (default every task); each task is checked read-only with the completion rule and SHA-256, and a metric reads a listed output of a complete task, checked against the task record's size and SHA-256
checkpoint: One stage of a per-FOV checkpoint folder through load_checkpoint; version 3 requires the header's nonempty files inventory, naming the stage's files besides the header (every file in registered/ for registered; <stage>.json and each present <stage>.csv or <stage>.parquet table for candidates and pre_qc, whose shared folder's other files are not that stage's), and checks its payload hashes; versions 1 and 2 need an inventory of every stage file with its SHA-256
trial: A trial's output folder: record.json, with its configuration hash recomputed, and every listed output by its size and SHA-256, both required; a recorded file input must carry its SHA-256, which is checked when a metric reads it; optional sha256 of record.json itself
files: A files mapping of key to {path, sha256}, the SHA-256 required, such as truth tables, codebooks or original inputs
```

## Metrics

A metrics file is a mapping with one list, `metrics`. Each metric has a unique
`name`, an optional short-set ID `metric`, a `function` of
`starfinder.evaluation` written `<module>.<name>` (modules `matching`,
`preprocessing`, `readout`, `registration`, `resources`, `segmentation`,
`spot_finding` and `stitching`), literal `parameters` and `arguments`. Both are
keyword arguments of the function and are checked against its signature before
anything is read. Each argument is a mapping:

```{evaluation-arguments}
source: The name of the source it reads
output: A trial's listed output, or a workflow task's declared output
input: A trial's recorded input role, read from its recorded path
task: The workflow task id whose output it reads
file: The key of a files source, or the file inside a workflow folder output
read: table, points, column, array, json or volume; by default from the suffix (.csv, .npy or .npz, .json, .tif)
key: A dotted path into the value read (mapping keys, list indices, attributes), such as spot_result.spots of a candidates checkpoint
where: A table filter, {column: value or [values]}; the rows before and after are recorded
columns: The table columns to pass as a float64 array, such as [z, y, x]
column: One table column to pass as an array
metadata: An ImageMetadata mapping, or true with a trial source for the trial's metadata
mapping: A mapping of labels to arguments, such as {moving: {...}} for evaluate_translation
```

## The saved evaluation

| Field of `evaluation.json` | Contents |
| --- | --- |
| `format`, `generated_at`, `software` | Record format, creation time, and the code (version, Git commit, dirty state) and environment identity |
| `inputs` | The sources and metrics files' paths and SHA-256 |
| `sources` | Per source: `name`, `kind`, `status` (`verified` or `failed`), `error`, `limitations`, `consumed` (each file's `path`, `sha256`, `bytes` and what it was checked against) and its kind's fields: a workflow's `manifest`, `roots`, `tasks` (`complete`, `problems`, record hash) and `incomplete`; a checkpoint's `stage`, `format_version`, `header`, `payload_check`, `channel_labels`, `round_labels`, `readout_mode` and `frames`; a trial's `record`, `trial_status`, `trial_error`, `configuration_sha256`, `frame_id`, `method` and `seed`; a files source's `files` |
| `metrics` | Per metric: `name`, `metric`, `function`, `parameters`, `arguments` (what each read), `sources`, `outcome`; when computed, `status`, `values`, `units`, `counts`, `reasons`, `config` and `details_file`; otherwise `reason`, or `error` for a metric that raised |
| `counts`, `status` | Sources, failed sources, metrics and the metrics per outcome; `complete` when every source is verified and every metric computed, else `incomplete` |
