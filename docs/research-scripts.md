# Codebook-aware research scripts

These scripts inspect saved codebook-aware decoding results and create diagnostic
tables and montages. Use the {doc}`readout guide <readout-contract>` for the
processing API and the {doc}`benchmark lifecycle <benchmark>` for explicit trials.

## Quick start

From `src/python`, inspect the table-analysis tool's arguments:

```bash
uv run python scripts/diagnose_codebook_aware_rescues.py --help
```

Actual analysis needs saved calls and spot coordinates. The
[saved-result layout](https://github.com/wanglab-broad/starfinder/blob/dev/design/notes/research-scripts.md#saved-result-layout)
describes those inputs; the scripts' help lists their options.

## What it does

| Script under `src/python/scripts/` | Purpose |
| --- | --- |
| `diagnose_codebook_aware_rescues.py` | Compare saved calls with a baseline and summarize rescue patterns |
| `generate_decoding_example_montages.py` | Show selected decoding outcomes across rounds and channels |
| `qc_codebook_aware_rescues.py` | Measure tensor evidence and make rescued-read montages |
| `run_codebook_aware_benchmark.py` | Run named decoder configurations and save comparison tables |

## What you get

The diagnosis tool writes tables and `summary.json` under the dataset's
`diagnostics/` directory. The montage and QC tools write figures and tables under
the chosen output directory. The decoder comparison writes decoded calls and
comparison tables. The
[script notes](https://github.com/wanglab-broad/starfinder/blob/dev/design/notes/research-scripts.md)
list their files.

## Main options

`--result-dir` selects saved results for diagnosis, montage and QC;
`--dataset` selects their dataset. Diagnosis also takes `--baseline-config`;
montage and QC take `--output-dir` for their figures and tables.

## Limits

Montages need registered images as well as tables and intensity tensors.
Preparing missing caches for real inputs can process those images. Saved calls
and diagnostic plots do not establish decoding accuracy or biological truth.

## How it is checked

The help command checks the command-line entry without loading data. Reading help
does not exercise table analysis, image preparation or montage generation.

[Script inputs and outputs](https://github.com/wanglab-broad/starfinder/blob/dev/design/notes/research-scripts.md).
