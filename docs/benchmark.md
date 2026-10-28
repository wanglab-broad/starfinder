# Benchmark lifecycle

The benchmark tools run explicit registration or Dataset/FOV pipeline cases,
evaluate their saved outputs, and write JSON and CSV reports. Use the
{doc}`recipe reference <benchmark-recipes>` to build cases and the
{doc}`dataset catalog <datasets>` to identify inputs. The
{doc}`benchmark API <api/benchmark>` provides the Python entry points.

## Quick start

From `src/python`, run this smoke example with a new output directory:

```bash
uv run python ../../docs/examples/benchmark_lifecycle.py /tmp/starfinder-benchmark-smoke
```

It creates an 8 × 16 × 16 impulse pair, runs the processing, evaluation and
report commands in separate processes, and prints the report directory.
`commands.json` keeps their arguments and output. This is a software example,
not an accuracy experiment.

## What it does

| Command | Purpose |
| --- | --- |
| `starfinder synthetic generate` | Generate inputs from a named synthetic preset; see the {doc}`synthetic guide <synthetic-specification>` |
| `starfinder benchmark run` | Process the configured cases and save trial records |
| `starfinder benchmark evaluate` | Compute the selected metrics from saved artifacts |
| `starfinder benchmark report` | Write a report from a saved evaluation |

Run the commands through `uv run` from `src/python`; append `--help` to inspect
arguments. Evaluation and reporting use saved records rather than repeating
processing.

For your own inputs, the following is an **excerpt**: paths and the case
configuration refer to files you supply. The smoke example above creates its
own complete inputs instead.

```bash
uv run starfinder synthetic generate --mode e2e --preset tiny --seed 42 --owner Jiahao --output /external/new-synthetic
uv run starfinder benchmark run --config cases.json --input-root /external/inputs --output-root /external/runs --owner Jiahao
uv run starfinder benchmark evaluate --run-dir /external/runs/RUN_ID
uv run starfinder benchmark report --evaluation-dir /external/runs/RUN_ID/evaluations/EVALUATION_ID
```

## What you get

`run_benchmark` returns a run directory; `evaluate_benchmark` and
`report_benchmark` return new evaluation and report directories. A run contains
its manifest, per-case trial records and processing artifacts. Reports contain
`trials.json` and `summary.csv`, including trial statuses and metric units.
The {doc}`recipe reference <benchmark-recipes>` lists the main files.

## Main options

* `--config`: the JSON case configuration, including repetitions and provenance.
* `--input-root`: where the case's input paths are resolved.
* `--output-root`: where runs are stored.
* `--owner`: who owns the run.
* `--run-id` and `--resume`: reopen a matching run and skip completed trials.

A changed input or configuration calls for a new run. For an interrupted trial,
start a new run to retry. The
[resume specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/benchmark.md#persistence-and-resume)
gives the identity checks and interruption rules.

## Limits

The benchmark does not select a scientific protocol or establish calibration,
accuracy, or biological truth. Pipeline evaluation reports counts. Trial timing
covers loading, processing and artifact persistence; it excludes evaluation.
The memory measurement is traced Python allocation, not process RSS. Use an
external process monitor when you need RSS; the benchmark does not enforce a
memory cap.

## How it is checked

The smoke example checks the known translation, image alignment and preservation
of the processing record through evaluation and reporting. It uses deterministic
impulses, so it checks the lifecycle without real images.

[Full lifecycle specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/benchmark.md).
