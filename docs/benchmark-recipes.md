# Maintained benchmark recipes

Look up the case builders, profiles and saved files here. The
{doc}`lifecycle guide <benchmark>` shows how to run a case; the
{doc}`benchmark API <api/benchmark>` documents the records and functions.

## Case builders

`benchmarks/recipes.py` builds cases from the JSON profiles in
`benchmarks/configs/`. It writes a configuration for `starfinder benchmark run`.

| Builder | What you supply |
| --- | --- |
| `registration_case` | Profile, case ID, reference and moving paths, their metadata, and evaluation settings |
| `pipeline_case` | Profile, case ID, FOV ID, round/channel source mapping, codebook path, and round count where needed |

The registration profiles below are read from `benchmarks/configs/registration.json`
when the site is built. For method options, see the
{doc}`registration reference <registration-algorithms>`.

```{benchmark-profiles} registration
```

The pipeline profiles below are read from `benchmarks/configs/pipeline.json`.
For the processing configuration, see the
{doc}`workflow reference <workflow-configuration>`.

```{benchmark-profiles} pipeline
```

### Case fields

Case options come from {py:class}`~starfinder.benchmark.BenchmarkCase`.

```{config-options} starfinder.benchmark.BenchmarkCase
```

## Saved files

Paths in this table are relative to the run directory unless stated otherwise.

| Path | Contents |
| --- | --- |
| `manifest.json` | Run identity, case configurations, input checksums, host and environment |
| `<case>/<trial>/trial.json` | Statuses, errors, effective configurations, resources and artifact references |
| `<case>/<trial>/trial.checksum.json` | Trial-record checksum |
| `<case>/<trial>/registered.npy`, `<case>/<trial>/transform.json` | Registration output and transform record |
| `<case>/<trial>/spots.csv`, `<case>/<trial>/reads.csv`, `<case>/<trial>/pipeline.json` | Pipeline detections, decoded reads and processing summary |
| `evaluations/<id>/results.json`, `evaluations/<id>/manifest.json` | Evaluation records and their provenance |
| `evaluations/<id>/reports/<id>/trials.json` | Complete report records |
| `evaluations/<id>/reports/<id>/summary.csv` | Long-form metrics: case, trial, requested/actual method, processing/evaluation status, metric, value, unit, metric status and reason |

## Saved-table helper

`benchmarks/report_saved.py` assembles explicitly supplied CSVs without running
image processing. Its command-line help describes the arguments.

| Output | Contents |
| --- | --- |
| `comparison.csv` | Quality values and optional timing values joined on supplied keys, with variant and resource-scope labels |
| `manifest.json` | Source and output checksums, keys and scope labels |

## Optional recipes

These pointers describe prerequisites and preserved recipes; they do not claim
calibration or accuracy on a dataset.

| Recipe | Where to look |
| --- | --- |
| Demons, TPS, CPD | Optional dependencies in the {doc}`registration reference <registration-algorithms>` |
| Postcode, MATLAB and hybrid comparisons | [Recipe notes](https://github.com/wanglab-broad/starfinder/blob/dev/design/notes/benchmark-recipes.md#optional-recipes) |
| Synthetic preprocessing comparisons and report rendering | [Comparison design and commands](https://github.com/wanglab-broad/starfinder/blob/dev/design/notes/benchmark-recipes.md#preprocessing-comparison-on-synthetic-presets) |
| Historical source mapping and preservation | [Source notes](https://github.com/wanglab-broad/starfinder/blob/dev/design/notes/benchmark-recipes.md#source-boundary-and-provenance) |
