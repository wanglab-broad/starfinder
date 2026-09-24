# Maintained benchmark recipes

The [common lifecycle](benchmark.md) replaces per-dataset Python workers and
scheduler loops. `benchmarks/recipes.py` builds cases; `starfinder benchmark
run/evaluate/report` executes them. `benchmarks/configs/registration.json` and
`pipeline.json` retain selected source parameters. They select **no inputs or
experiments automatically**. A profile named `large` can be tested with a tiny
fixture; that does not validate the large dataset or historical performance.

## Source boundary and provenance

The maintained {download}`source map <../benchmarks/configs/source-map.json>`
assigns every one of the 56 external scripts/notebooks a disposition, original
relative path, SHA-256 and replacement target. Its bounded scope is file depth
≤6 under `starfinder_benchmark`, extensions `.py/.m/.sh/.ipynb`, excluding runs,
records, environments, Git/cache/hidden directories and symlinks. Files deeper
than this boundary are not inventoried. Original files/results/notebook outputs
are retained unchanged. The private byte archive and timestamp/Git-state
manifest are stored outside this repository.
Preservation does not validate historical outputs, dates, scientific claims or
licensing. Backup coverage remains unverified; Jiahao owns retention.

The maintained adapters/report helpers are newly authored STARfinder code under
the Python package's declared MIT terms. Only configuration facts and provenance
locators were extracted; no external implementation or notebook output was
vendored. No new license is asserted for historical files. In particular, the
eight `registration/results/imregdemons_ref/*.m` snapshots are private reference
material, excluded from redistribution and implementation targets. MATLAB API
and original source files remain unchanged.

## Registration profiles

| Profile | Preserved selection |
| --- | --- |
| `translation` | scipy FFT; one worker; explicit reference/moving pair |
| `local-default`, `py_demons` | demons, iterations 100/50/25, sigma 1, antialias |
| `py_diffeo` | diffeomorphic variant with the same pyramid/iterations |
| `tps-default` | noise 3, distance 10, minimum 50 matches, 1000 controls, smoothing 1, grid 32 |
| `tps-small` | noise 2, distance 15, minimum 5 matches, smoothing 2, grid 16 |
| `cpd-default` | noise 5, 1000 controls, automatic kernel width, weight 2, outliers .15, affine first, grid 16 |
| `cpd-small` | same CPD settings, grid 8 |
| `antialias-old` | symmetric, 25 iterations, sigma .5, sitk pyramid |
| `antialias-two-level` | symmetric, 50/25 iterations, sigma .5, antialias |

Select `tps-small` explicitly; dataset names no longer change numerical settings.
The old `run_local_comparison.py` and v2 are distinct recipes, not interchangeable
measurements: the old script refers to symmetric/25/.5 results, while v2 selects
`py_demons` and `py_diffeo`. Old scheduler timeouts (600/900 seconds), implicit
warmups, existence-only resume and inferred dataset sweeps are preserved as
historical behavior. Current execution uses explicit cases, externally bounded
commands and checksum/config-aware resume. No large sweep is dispatched.

A request for `benchmarks/recipes.py` contains explicit builder arguments:

```json
{
  "task": "registration", "profile": "translation", "case_id": "fixture",
  "reference": "reference.npy", "moving": "moving.npy",
  "reference_metadata": {"frame_id": "reference"},
  "moving_metadata": {"frame_id": "moving"},
  "evaluation": {"ncc": true, "ssim": {"data_range": 255, "policy": "mip"}}
}
```

From `src/python`:

```bash
uv run python ../../benchmarks/recipes.py --request /external/request.json --output /external/new-cases.json
uv run starfinder benchmark run --config /external/new-cases.json --input-root /external/inputs --output-root /external/runs --owner Jiahao
```

Case generation validates configuration without reading image data. These paths
are placeholders; supplying them does not authorize scientific execution.

## Pipeline and output adapter

`pipeline_case` builds `task="pipeline"` cases using the same
`from_workflow_config` → `Dataset` → `FOV.run(PipelineConfig, ExecutionConfig)`
path as the workflow. Both batch and streaming use this sequence. Per-channel
TIFF inputs have an explicit `{round: {channel: path}}` source mapping and a
codebook path. Layout (`round/FOV` or `FOV/round`) is resolved by that mapping,
not inferred from a dataset label. All inputs are checksummed and copied to the
unique run. Disposable layout links live in local `TMPDIR`, never on the SMB
artifact mount. No image conversion occurs on loading; normalization to uint8
is an explicit selected stage and appears in effective configuration.

For example, a pipeline request has `task: "pipeline"`, `profile:
"large-streaming"`, `case_id`, `fov_id`, `codebook`, `n_rounds: 4`, and `sources`
with `round1` through `round4`, each naming all four channels. Synthetic profiles
require the explicit round count formerly read from `ground_truth.json`. Real
profiles preserve their source round count, channel order, rotation, reference,
threshold, extraction radius and codebook split. Geometry remains as stored in
TIFF or explicitly unknown; extraction radii are not physical voxel spacing.

The 20 pipeline profiles distinguish batch, streaming and global-plus-local
settings. Source aging profiles use encoded `split_index=4` and `end_bases="CC"`;
the LR source has radius `(2,2,1)`. These are historical script choices, not a
correction to the separately owned nine-round, segmented scientific protocol.
Likewise tissue-2D's historical radius `(1,2,2)` is preserved. Do not replace
these values using dataset overview tables. Qualifying discrepancies belongs to
the scientific owners, not this migration.

Saved `spots.csv`, complete accepted/rejected `reads.csv`, `pipeline.json`,
metadata, float transforms/dense fields and registration attempts support repeated count evaluation and report
creation. Count evaluation reads saved tables only; it does not detect spots,
extract intensities or rerun registration. Counts are not accuracy or molecular
truth. To process already registered stacks, construct a pipeline case with
explicit single-channel TIFF sources and registration disabled; this is a new
**processing run**, not saved-only evaluation. Supplied stacks need declared
compatible frames for extraction. Hybrid exported layouts must be normalized
explicitly; 4D stacks are not guessed or silently split by the adapter.

## Saved report recipes

Current trial/evaluation records use `starfinder benchmark report`. Selected
legacy comparison, scaling, backend-summary and notebook table behavior uses
`benchmarks/report_saved.py` on explicit saved CSVs:

```bash
uv run python ../../benchmarks/report_saved.py --quality /external/quality.csv --timing /external/timing.csv --keys dataset pair_type backend --variant local-v2 --timing-scope 'GNU time whole subprocess, seconds' --memory-scope 'GNU time maximum RSS, KiB' --output /external/new-report
```

The helper requires unique non-null join keys, retains unmatched timing failures
and quality rows with an outer join, and records input/output hashes. Normalize
legacy column names explicitly before joining. There is no algorithm execution,
implicit metric computation, zero-fill, ranking or pooling. Distinct variants
must retain distinct labels. Legacy plots/notebook outputs remain historical;
only reusable table assembly has been selected for migration.

Historical estimator-only `internal_time`, FOV-stage sums and whole-subprocess
wall time are different scopes. Historical tracemalloc `memory_mb` is not peak
RSS; Linux GNU time maximum RSS is KiB. Process-lifetime `ru_maxrss` must not be
reported as an isolated FOV peak. New lifecycle timings include loading,
processing and persistence and exclude evaluation. No speedup/equivalence claim
is made by placing these records in a common table.

## Optional recipes

| Recipe | Prerequisites and retained distinctions |
| --- | --- |
| Postcode aging/synthetic | Separate Postcode dependency/environment, model/config, codebook and exported intensity identities; original `common.py` and both workers remain external. No core dependency or model execution. |
| MATLAB global/local | Licensed MATLAB, original unchanged functions, explicit data/coordinate/channel metadata, separate resource authorization; no vendored reference snapshots. |
| Hybrid MATLAB global → Python LR | Qualified MATLAB global exports, Python local config and explicit channel/frame mapping; preserve distinct backend label. |
| Hybrid Python global → MATLAB LR | Qualified Python exports and separately run licensed MATLAB local stage; never substitute for the reverse hybrid. |
| Backend diagnostics | External backend manifest plus matching saved spots/stacks/metrics; `diagnose_lr_effects.py` pairs retain their original labels. New saved tables can use the common report helper. |
| Aging pad balance | Qualified pad/gene annotations and segmented codebook decisions; original diagnostics remain historical, not a new validated metric. |
| LR synthetic generation | `starfinder synthetic generate` is the common entry point; original recipe has base seed 200, linear deformation magnitude 10, XY shift 50, Z shifts 5/7/10 for large/tissue/thick_medium. Process-dependent hash-derived seeds and truth remain unqualified. Do not run/generate these large presets as routine validation. |
| Antialias/large comparison | Explicit method profile, qualified image pairs and separately authorized memory/time; old symmetric and later demons/diffeomorphic recipes remain distinct. |

No optional recipe is claimed as a runnable scientific experiment merely because
its configuration parses. Qualification and scientific acceptance remain
separate from software validation. Bounded adapter tests use deterministic
8×16×16 TIFFs, four channels/rounds and saved-table fixtures; no real-image
processing, historical notebook execution or MATLAB run is part of validation.

From `src/python`, `uv run python ../../docs/examples/benchmark_recipes.py
/external/new-recipe-smoke` (one line) exercises eight source-derived profiles:
translation, demons, diffeomorphic, small TPS/CPD and pipeline batch, streaming
and global-plus-local. It saves the exact case config, inputs, trials,
evaluations and reports. All images are deterministic and at most 16×32×32;
the pipeline images are 8×16×16. The original scientific presets are not generated.
