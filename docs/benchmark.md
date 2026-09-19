# Benchmark lifecycle

```{toctree}
:hidden:

benchmark-recipes
```

The supported interface is `BenchmarkCase`, `BenchmarkTrialResult`,
`run_benchmark`, `evaluate_benchmark`, and `report_benchmark`. The built-in task
adapters are registration (translation, demons, TPS or CPD) and the shared
Dataset/FOV pipeline. Unsupported tasks fail explicitly. See the
[maintained recipes](benchmark-recipes.md) for source-derived profiles, pipeline
input/output contracts, saved reports and optional backend prerequisites.
No adapter silently chooses a research protocol.

## Four commands

```bash
uv run starfinder synthetic generate --mode e2e --preset tiny --seed 42 --owner Jiahao --output /external/new-synthetic
uv run starfinder benchmark run --config cases.json --input-root /external/inputs --output-root /external/runs --owner Jiahao
uv run starfinder benchmark evaluate --run-dir /external/runs/RUN_ID
uv run starfinder benchmark report --evaluation-dir /external/runs/RUN_ID/evaluations/EVALUATION_ID
```

Each command supports `--help`. `python -m starfinder` is equivalent. The old
`starfinder-generate` alias and `python -m starfinder.benchmark` entry point are
removed. Success exits 0; invalid arguments/configuration, missing artifacts or
checksum failures exit 2 with an error on stderr. A run with processing failures,
or an evaluation with failed/skipped trials, exits 1 and prints the saved result
path. Undefined metrics are valid results, not execution failures. Reporting
failed trials succeeds: their statuses/errors remain in the report.

Synthetic generation requires a new output directory. `--mode registration`
writes reference/moving pairs; it currently supports uint8 only. The E2E mode
supports `--dtype uint8|uint16`. No existing output is overwritten. Generation
is backed by `starfinder.synthetic` functions; manifests record command, seed,
owner and checksums. Existing process-dependent synthetic randomness remains
unqualified; a seed is not a claim of cross-process reproducibility or molecular
truth. See [synthetic API](api/synthetic.rst).

## Explicit configuration

A registration case uses NPY or TIFF single-channel ZYX inputs. Paths are relative
to `--input-root`, must stay within it, and are checksummed. Input and output
roots must be disjoint. Metadata describes the actual grids; unknown geometry
stays unknown. No image conversion, implicit detection, or mask threshold is
selected by the benchmark. For example:

```json
{
  "schema_version": 1,
  "repetitions": 1,
  "provenance": {"seed": 42, "dataset_version": "user-supplied"},
  "cases": [{
    "case_id": "example",
    "task": "registration",
    "inputs": {"reference": "reference.npy", "moving": "moving.npy"},
    "truth": {"correction": "correction.json"},
    "artifacts": {},
    "config": {
      "registration": {"method": "translation", "backend": "scipy_fft"},
      "reference_metadata": {"frame_id": "reference"},
      "moving_metadata": {"frame_id": "moving"},
      "evaluation": {
        "ncc": true,
        "ssim": {"data_range": 255, "policy": "mip"},
        "translation": {"tolerance": 0.5}
      }
    }
  }]
}
```

`correction.json` contains a floating-point ZYX **correction** triple, for example
`[-1.0, 2.0, -1.0]`; there is no inferred sign conversion. Omit the `translation`
evaluation and truth entry when no correction truth is available. Evaluation
must explicitly select at least one metric. SSIM requires `data_range` and a
spatial `policy`. The pure [evaluation APIs](api/evaluation.registration.rst)
retain units, eligible counts, undefined values and reasons.

Fallback is opt-in through `config.fallback`, for example
`{"on_errors": ["InsufficientLandmarksError"], "configs": [{"method": "translation"}]}`.
Only named estimation error categories permit fallback. Input/config errors,
missing dependencies and application failures never substitute an algorithm.
Requested/actual methods, ordered attempts and effective configurations are
saved separately. Successful fallback is labeled `fallback_success`.

## Persistence and resume

Every run has a unique directory (timestamp plus UUID by default), an immutable
schema-versioned manifest, and one record per case/repetition. Inputs and supplied
truth/artifacts are copied into the run. Registered arrays, float transforms,
dense fields where applicable, metadata, diagnostics, and checksums are retained.
JSON records round-trip through `to_dict`/`from_dict` without converting missing
values into zeros or strings. `BenchmarkTrialResult` is distinct from the
algorithm's `registration.RegistrationResult`.

`--resume --run-id ID` requires identical schema, case configuration, owner,
roots, repetitions, provenance and input checksums. Completed trials, including
recorded failures, are verified and skipped. An incomplete trial directory from
an interrupted process is preserved and blocks resumption; use a new run ID to
retry. There is no overwrite flag. Resume is single-writer; concurrent access is
not supported. New input/configuration means a new run.

Evaluation reads saved, checksum-verified inputs/outputs and writes a new
`evaluations/ID` directory. It never calls registration or regenerates images.
Processing failures are skipped; metric errors are recorded as evaluation
failures; zero-denominator metrics remain undefined. Missing/corrupt required
artifacts fail clearly before evaluation starts. Reevaluation does not modify
processing records and no longer needs the original input root.

Reporting reads an evaluation's saved JSON and writes a new `reports/ID` directory
with complete JSON records and a long-form CSV. It does not run processing or
evaluation. Reports retain failures and fallback identity; no automatic ranking
or pooling across incompatible methods is performed.

## Resource and provenance limits

Wall time (`time.perf_counter`, seconds), process CPU time (`time.process_time`,
seconds), and peak traced allocations (`tracemalloc`, bytes) cover one trial's
input loading, estimator attempts, application and artifact persistence. They
exclude evaluation and report generation. Traced allocations are **not process
RSS** and do not establish native/backend peak memory. Process RSS is explicitly
unmeasured (`null`); use an external monitor such as `/usr/bin/time -v`, whose
maximum RSS on Linux is in KiB and covers the whole invoked process.

Manifests record code revision/dirty status/diff identity, runtime, host, numerical
thread settings, CPU affinity, caller provenance, owner and retention. A dirty
worktree or private input path is not a public reproducibility archive. Supply
data accession/version, seed, hardware and limitations in `provenance`; unknown
facts must stay unverified. Backup coverage is unverified. No hard memory cap is
implied. MATLAB execution and scientific benchmark qualification are excluded.

Before: wrapper classes/decorators selected benchmarks and wrote results into
shared default trees. After: construct explicit cases and call
`run_benchmark(cases, input_root=..., output_root=..., owner=...)`, then
`evaluate_benchmark(run_dir)` and `report_benchmark(evaluation_dir)` independently.

## Executable smoke example

From `src/python`, run `uv run python ../../docs/examples/benchmark_lifecycle.py
/external/new-smoke-directory` (on one line). This creates a deterministic
8×16×16 impulse pair and explicit configuration, executes run/evaluate/report
in separate CLI processes, verifies alignment and unchanged processing records,
and retains exact command logs with the report. No random or scientific dataset
is used. Current correctness coverage lives in `test/test_benchmark.py`;
notebook outputs remain historical until explicitly rerun.
