# Codebook-aware research scripts

These developer tools inspect or benchmark saved results from the codebook-aware
decoder. Run them from `src/python`. `<result-dir>` and `<output-dir>` stand for
their matching command-line arguments; benchmark input paths are rooted at
`--postcode-root` or the configured benchmark data directory.

`_decoding_inputs.py` is a shared helper imported by the saved-input tools. It
declares positional round and channel labels for historical intensity tensors
and adapts decoding, extraction and report tables; it is a Python module, not a
command-line script.

## Saved result layout

The QC and montage scripts use the following per-dataset files:

| Path | Contents |
| --- | --- |
| `<result-dir>/<dataset>/codebook_aware/<config>/<fov>.csv` | Saved decoded calls. |
| `<result-dir>/<dataset>/spots/<fov>.csv` | Spot IDs and zero-based `z`, `y`, `x` coordinates. |
| `<result-dir>/<dataset>/intensity_tensors/<fov>.npy` | Saved tensor with axes `(spot, channel, round)`. |
| `<result-dir>/<dataset>/input_metadata/<fov>.json` | Optional input and codebook metadata. |

Image evidence uses registered per-round TIFFs with `ZYXC` axes. Existing backend
results are read from `registered_final/roundN.tif` or
`registered_global/roundN.tif`. When those are absent for a supported raw real
dataset, the QC helper writes
`<result-dir>/<dataset>/registered_stacks/<fov>/roundN.tif` with
`starfinder.io.save_volume`, then reads it with `starfinder.io.load_volume_zyxc`.
That cache also contains `metadata.json`. The TIFF volume is `ZYXC`, separate
from the saved `(spot, channel, round)` NumPy tensor.

## `diagnose_codebook_aware_rescues.py`

Compares codebook-aware calls with a baseline configuration and summarizes
rescue patterns, confidence, correction rounds and spatial distributions. It
can also measure codebook-neighbor bias when a codebook mapping is available.

It requires the decoded CSV at
`<result-dir>/<dataset>/codebook_aware/<config>/<fov>.csv`, the baseline CSV at
`<result-dir>/<dataset>/codebook_aware/<baseline-config>/<fov>.csv`, and
`<result-dir>/<dataset>/spots/<fov>.csv`. A codebook path can be supplied or
inferred from saved metadata or the dataset's known codebook location. This
table analysis needs no image cache; its saved tables may come from real or
synthetic datasets.

For each FOV it writes `gene_rescue_enrichment.csv`, optionally
`codebook_neighbor_bias.csv`, `top10_shift.csv`, `round_corrections.csv`,
`correction_transitions.csv`, `rescue_sequence_patterns.csv`,
`confidence_by_call_type.csv`, `gate_sensitivity.csv`,
`rescued_confidence_deciles.csv`, `spatial_bins.csv`, `z_slices.csv`,
`edge_summary.csv`, `rescued_examples.csv` and `summary.json` under
`<result-dir>/<dataset>/diagnostics/<fov>_<config>/`. It also writes
`<result-dir>/<dataset>/diagnostics/<config>_summary.json` across processed FOVs.

Options: `--result-dir`, `--dataset`, `--config`, `--baseline-config`, `--fovs`,
`--grid-size`, `--edge-margin-px`, `--min-bin-reads`, `--top-n`,
`--codebook-path`, `--split-index`.

## `generate_decoding_example_montages.py`

Selects representative decoding outcomes and renders each selected spot across
all rounds and channels. The images show the saved registered evidence
alongside the saved decoding and intensity measurements.

It requires the decoded CSV, spot-coordinate CSV and
`(spot, channel, round)` tensor in the saved result layout above, a codebook
mapping, and registered `ZYXC` round images. It reads existing backend
`registered_final` or `registered_global` TIFFs when available; for supported
real datasets without those directories, the shared QC helper creates the
`registered_stacks/<fov>/roundN.tif` cache through `starfinder.io.save_volume`.
It works from saved synthetic or real results; cache creation for raw real
datasets requires their real input images.

It writes `all_round_montages/*.png`, `decoding_examples.csv`,
`decoding_example_summary.csv` and `decoding_example_summary.json` under
`--output-dir`, defaulting to
`<result-dir>/<dataset>/qc/<fov>_<config>_decoding_examples/`.

Options: `--result-dir`, `--dataset`, `--fov`, `--config`,
`--examples-per-category`, `--patch-radius`, `--seed`, `--output-dir`,
`--codebook-path`, `--split-index`.

## `qc_codebook_aware_rescues.py`

Measures saved intensity evidence for codebook-aware rescue calls and creates
image montages for selected rescued reads. It reports tensor-level metrics even
when montage generation is disabled.

It requires the decoded and spot CSVs, the `(spot, channel, round)` tensor and
a codebook mapping in the saved result layout above. Full image QC also
requires registered `ZYXC` round TIFFs: existing backend `registered_final` or
`registered_global` stacks are preferred, otherwise supported raw real datasets
can be processed to the `registered_stacks/<fov>/roundN.tif` cache using
`starfinder.io.save_volume` and loaded using
`starfinder.io.load_volume_zyxc`. Saved synthetic results need their registered
images available; producing a cache from raw inputs requires real data.

It always writes `all_rescued_h1_tensor_metrics.csv` and
`all_rescued_h1_tensor_summary.csv` under `--output-dir`, defaulting to
`<result-dir>/<dataset>/qc/<fov>_<config>_gene_evidence/`. Unless
`--tensor-metrics-only` is set, it also writes `round_montages/*.png`, up to
`--all-round-limit` files in `all_round_montages/*.png`, `qc_examples.csv`,
`qc_group_metrics.csv` and `qc_summary.json` there.

Options: `--result-dir`, `--dataset`, `--fov`, `--config`, `--genes`,
`--control-genes`, `--examples-per-group`, `--patch-radius`,
`--all-round-limit`, `--seed`, `--output-dir`, `--codebook-path`,
`--split-index`, `--tensor-metrics-only`.

## `run_codebook_aware_benchmark.py`

Runs the lightweight codebook-aware decoder under named threshold
configurations, compares its calls and records timing, memory and data metrics.
Synthetic selections are evaluated against saved ground truth; real selections
are summarized from their saved calls.

Required inputs depend on `--datasets`. `synthetic_medium` uses saved Postcode
codebook, ground truth, intensity tensors and spot CSVs under
`<postcode-root>/data/synthetic_medium/` and
`<postcode-root>/results/synthetic_medium/`. `synthetic_large` reads saved
formed-scene codebook, ground truth and per-round channel images under the
benchmark `e2e/data/large/` tree, then prepares or reuses tensor and spot
caches. `LN` and `aging` read backend `run_metadata.json`, zero-based spot CSVs
and registered `ZYXC` TIFFs under
`e2e_backend_comparison/results/<dataset>/<fov>/python_global_only/`; the TIFFs
are loaded with `starfinder.io.load_volume_zyxc` from `registered_final` or
`registered_global`. `cell_culture_3D` and `tissue_2D` use raw real images and
codebooks unless matching cached tensor and spot inputs are reused. These
selections therefore require real data when their caches are absent.

Under `--output-dir` the script writes per-FOV decoded CSVs in
`<dataset>/codebook_aware/<config>/<fov>.csv`, cached
`<dataset>/intensity_tensors/<fov>.npy` and `<dataset>/spots/<fov>.csv` inputs
when prepared, per-dataset `comparison.csv` and `comparison_summary.json`, and
a top-level `comparison_all.csv`. Raw real input preparation also records
`<dataset>/input_metadata/<fov>.json`.

Options: `--datasets`, `--output-dir`, `--postcode-root`, `--aging-codebook`,
`--aging-fovs`, `--ln-fovs`, `--cell-culture-fovs`, `--tissue-2d-fovs`,
`--synthetic-fovs`, `--reuse-inputs`, `--no-reuse-inputs`, `--configs`.
