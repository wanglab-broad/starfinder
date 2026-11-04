# Recipe / Experiments

An experiment expands methods, conditions, seeds and inputs into trial
configurations; each trial runs one method in one process; an evaluation reads
the saved trial, workflow and checkpoint outputs and computes the metrics you
select from `starfinder.evaluation`, without processing anything again. The
{doc}`commands reference <commands>` lists every argument and file field.

```{toctree}
:maxdepth: 1

Experiment commands reference <commands>
Benchmark lifecycle <../benchmark>
Maintained benchmark recipes <../benchmark-recipes>
Codebook-aware research scripts <../research-scripts>
```

## Quick start

From `src/python`, write the bounded example's inputs into a new folder outside
Git, then expand, run both trials and evaluate them:

```bash
uv run python ../../docs/examples/experiment_inputs.py /tmp/starfinder-experiment
uv run starfinder experiment expand --config /tmp/starfinder-experiment/experiment.yaml --output /tmp/starfinder-experiment/trials
uv run starfinder trial run --config /tmp/starfinder-experiment/trials/trial-000000.json
uv run starfinder trial run --config /tmp/starfinder-experiment/trials/trial-000001.json
uv run starfinder experiment evaluate --sources /tmp/starfinder-experiment/sources.yaml --metrics /tmp/starfinder-experiment/metrics.yaml --output /tmp/starfinder-experiment/evaluation
```

The example is an 8 × 16 × 16 impulse pair shifted by (1, −2, 1) voxels, one
translation method under two conditions (the scipy and skimage backends) and
the truth shift. `values.csv` in the evaluation folder then holds, for each
trial, `max_error` 0 voxels and `passed` True. This is a software example, not
an accuracy experiment.

## What it does

| Command | What it does |
| --- | --- |
| `starfinder experiment expand` | Resolves every trial (registry defaults, input SHA-256) before writing, then writes `trial-NNNNNN.json` per trial and `manifest.json`. No processing runs and no engine is imported. |
| `starfinder trial run` | Runs one trial configuration in this process and writes its outputs and `record.json`. There is no scheduler and no retry. |
| `starfinder experiment evaluate` | Reads the listed sources, checks every file it consumes against its SHA-256, calls the selected metrics and writes `evaluation.json`, `values.csv` and `metrics/`. |

The §2.13 workflow stays the executor for FOV and downstream work: run it with
`starfinder workflow`, then evaluate its outputs. To run many trials, give each
`trial-NNNNNN.json` to its own process (a cluster job or a shell loop).

## Experiments

An experiment mapping (YAML or JSON) has the lists `methods`, `conditions`,
`seeds` and `inputs`, and optional common `metadata` and `root_output_path`.
Trials are ordered by method, condition, seed and input, with the input varying
fastest. A condition has an optional `name` and recursive `method` and `inputs`
overrides; a scene descriptor without a condition or seed receives the current
condition name and seed. Each trial's output defaults to
`<output>/trial-NNNNNN`; a `root_output_path` places the outputs elsewhere, for
example on local scratch. Configuration hashes exclude the output root, so a
job can relocate it. `expand` refuses an output folder that is not new or
empty, so no stale trial file stays beside a new manifest.

## Trials

A trial names `method`, `inputs`, `seed` and `root_output_path`, and optional
`metadata` (one common grid, default frame `trial/reference`) and `condition`.
Input paths are relative to the trial file; each may be `{path, sha256}`. The
method's `module` is one of:

* `registration`: a registered `method` and its `config`; inputs `reference`
  and `moving`. Outputs `image.npy` and `transform.json` (plus `transform.npz`
  for a dense field or B-spline coefficients).
* `preprocessing`: ordered `steps` of `{method, config}` and optional
  `reference_steps`, which run on the reference first and cannot require a
  reference themselves; input `image`, optionally `reference`. Outputs
  `image.npy` and `preprocessing.json`.
* `spot_finding`: a method of `SPOT_FINDING_METHODS` with its `config`,
  `device` (`cpu`) and `spot_namespace` (`trial`); input `image` (ZYX or ZYXC).
  Outputs `spots.csv` (the `find_spots` table) and `spot_finding.json`.
* `segmentation`: a method of `SEGMENTATION_METHODS` with its `config`,
  `target`, one `roles` entry per image channel, `device` and
  `label_namespace`; input `image`, and `seeds` labels for a seeded method.
  Outputs `labels.npy` and `segmentation.json`.

A preprocessing or spot-finding input may instead be a regenerative scene
descriptor: `scene` (preset `uncalibrated_preprocessing`, version
`UNCALIBRATED_PREPROCESSING_VERSION`, condition, seed and optional saturation
multiplier `k`), `round` and optional `channel`. A segmentation `image` may name
a `CELLULAR_PRESETS` scene (preset, `CELLULAR_PRESET_VERSION`, the trial seed),
`fov`, `round` (`reference` or `morphology`) and the ordered `stains`. The
installed version and configuration hash are checked before generation, the
scene is generated once inside the trial and its image hashes go to
`scenes.json`; no input image is stored. The uncalibrated preprocessing scene
is a pilot with the documented SNR and gain-spread misses. Learned weights must
already be in the weights cache: they are verified before any input is loaded,
and missing weights fail with `MissingModelError`.

## Evaluation

The sources file names each source and its kind:

* `workflow`: a plan's `workflow/manifest.json`, optionally with `tasks`. Each
  task is checked read-only with the workflow's completion rule and SHA-256
  (as `starfinder workflow verify --hash` does, without writing `verify.json`).
  A metric reads a listed output of a complete task.
* `checkpoint`: a per-FOV checkpoint folder and one `stage` (`registered`,
  `candidates` or `pre_qc`), restored with `starfinder.dataset.load_checkpoint`.
* `trial`: a trial's output folder, its `record.json` and its listed outputs.
* `files`: explicit files, each with its required SHA-256, such as truth tables.

A required SHA-256 or byte count that is missing or malformed fails its source
before any metric reads it.

The metrics file names each metric, an optional short-set ID in `metric`, the
`function` (`<module>.<name>` of a module of `starfinder.evaluation`), its
literal `parameters` and its `arguments`, each read from a source. For example,
the Quick start's metric reads the trial's saved displacement:

```yaml
detected: {mapping: {moving: {source: scipy, output: transform.json, key: transform.displacement_zyx}}}
reference_metadata: {source: scipy, metadata: true}
```

Truth is never inferred from a processing output; supply it as a `files`
source.

## What you get

`evaluation.json` records the code and environment identity, the sources and
metrics files with their SHA-256, each source with its status, limitations and
every consumed file's SHA-256, and each metric with its parameters, the
arguments it read, its outcome, values, units, counts and reasons. `values.csv`
is the long table, one row per metric value; `metrics/` holds each computed
metric's details. The {doc}`commands reference <commands>` lists the fields.

## Main options

* `--config`: the experiment (`expand`) or trial (`trial run`) mapping.
* `--sources`, `--metrics`: the evaluation's two mappings.
* `--output`: a new or empty folder; `evaluate` also refuses a folder inside a
  source.

Exit status 0 means done; 1 means a trial's method or input failed (its
`record.json` says `failed`), or an evaluation's source failed or one of its
metrics was not computed (the evaluation is still written); 2 means a usage or
input error, with nothing written.

## Limits

* A metric's outcome is `computed`, `skipped` (a source it needs failed, a
  trial failed, or a workflow task is not complete), `unavailable` (a file,
  snapshot or value it needs is absent) or `failed` (the metric raised). A
  computed metric whose value is undefined, such as F1 without truth, keeps a
  null value with a reason; no missing value becomes zero.
* A checkpoint header of version 3 must hold its `files` inventory, naming
  the stage's files besides the header with a well-formed SHA-256. For
  `registered` these are every file in its own `registered/` folder. For
  `candidates` and `pre_qc`, which share the FOV's checkpoint folder, they are
  `<stage>.json` and each `<stage>.csv` or `<stage>.parquet` table present; the
  other stage's files and other records in that folder are not that stage's.
  Every file `load_checkpoint` reads is a stage file. A header without the
  inventory, or with an entry missing or malformed, fails its source before any
  payload file or metric is read.
* Checkpoint headers of versions 1 and 2 hold no payload hashes. Such a source
  needs an `inventory` of every stage file with its SHA-256, and the evaluation
  records that its files were checked only against that inventory.
* A trial record holds no truth inventory, and a trial folder alone is not a
  portable evaluate-only dataset: its inputs are referenced by path and
  SHA-256, not copied.
* `record.json`'s `wall_s` starts after the output folder is reserved; `cpu_s`
  and `peak_rss_kib` cover the whole process and its reaped children, RSS as the
  larger of the two high-water marks. They are measurements, not limits.
* The old `starfinder benchmark` commands remain beside these until their
  retirement; the two do not share records.

## How it is checked

The tests run each command's help and exit statuses, expand and run trials
through the commands, and evaluate one workflow FOV with its three checkpoint
stages and a trial with every processing entry point patched to raise, the
source files hashed before and after. Corrupt, missing and version-2 sources,
failed trials, incomplete tasks, metrics that raise and undefined metrics keep
distinct outcomes.

[Benchmark specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/benchmark.md).
