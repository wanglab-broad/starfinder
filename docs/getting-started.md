# Minimal Python image-to-molecule quickstart

Generate a small synthetic dataset, register its sequencing rounds, detect spots,
extract their color sequences, and filter them against a codebook. The endpoint
is a CSV of **candidate molecules with gene labels**, one file per field of view
(FOV). Segmentation, cell assignment, cell expression matrices, and biological
validation are outside this example. The [single FOV demo](get-started/single-fov-demo.md)
shows the same route as an executed notebook, and the
[multi-FOV / sample demo](get-started/multi-fov-demo.md) continues to cells and a whole sample.

## Prerequisites and installation

Install the package as [Installation](get-started/installation.md#the-python-package)
describes: a STARfinder checkout, `uv` and Python 3.12, with no optional extra. No
downloaded microscopy data, MATLAB, Snakemake, GPU, SimpleITK, or SpatialData is
needed.

## Run the complete example

Still in `src/python`, replace the path below with a **new directory outside the
checkout**. The script refuses to reuse an existing output directory. Use a new
path for each rerun; retain failed outputs for diagnosis.

```bash
export QUICKSTART_OUTPUT=/absolute/path/outside/checkout/starfinder-quickstart
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLCONFIGDIR="${QUICKSTART_OUTPUT}-matplotlib"
uv run python ../../docs/examples/quickstart.py "$QUICKSTART_OUTPUT"
```

Allow one local CPU, 1 GiB RAM, and 50 MiB of output space, in addition to the
installed environment and package cache. FOVs run sequentially. The recorded
run took about 8 seconds and peaked at 294 MiB resident memory with these thread
settings. Runtime and memory depend on the host; these are observations, not
scheduler-enforced limits. A two-minute timeout was used during validation.

The executable source is included below so the displayed recipe and the script
being run remain the same. It uses the public
{py:class}`~starfinder.dataset.Dataset` and
{py:class}`~starfinder.dataset.FOV` APIs.

## Input and processing configuration

The source is the repository's
{py:func}`~starfinder.synthetic.benchmark_scene_preset` **tiny** preset,
generated locally with a fixed seed of **42** by
{py:func}`~starfinder.synthetic.generate_amplicon_dataset`, the data of
`starfinder synthetic generate --mode e2e --preset tiny --seed 42`, and written
by {py:func}`~starfinder.synthetic.save_amplicon_dataset`. The writer saves the
requested configuration and provenance to `input/generation.json` and generated
truth to `input/ground_truth.json` (historical truth keys, version 2.0).

| Setting | Value |
| --- | --- |
| Input size | 2 FOVs, 4 rounds, 4 channels; each channel TIFF is `(8, 128, 128)` in `(Z, Y, X)` |
| Loaded round | `(8, 128, 128, 4)` in `(Z, Y, X, C)`, `uint16`; 32 TIFFs total, 8 MiB pixel payload |
| Spots and codebook | 10 generated amplicons per FOV; 12 synthetic genes, GeneA–GeneL; 5-base barcodes encode 4 colors |
| Signal and noise | `benchmark-presets-v1`: lognormal peak amplitude (median 1500), Gaussian widths about 1.5 (Z) and 1.3 (Y/X) voxels, camera offset 100, spatially varying background, Poisson plus read noise, 5% crosstalk and a 0.95 per-round trend |
| Motion | Continuous translations up to ±2 Z and ±5 Y/X voxels; no local deformation |
| Reference and registration | `round1`; global phase correlation on the sum of channels in reference and moving rounds |
| Loading | Default loading preserves the generated `uint16` values |
| Channel order | `ch00`, `ch01`, `ch02`, `ch03` maps to colors `1`, `2`, `3`, `4` |
| Detection | `intensity_estimation="noise"`, `intensity_threshold=5.0`, `min_distance=1` |
| Extraction | `voxel_size=(1, 2, 2)`: half-widths `(dz, dy, dx)`, a 3×5×5 voxel neighborhood |
| Codebook/filtering | `input/codebook.csv` holds each entry's color sequence, from the generator's reversed-barcode encoding (`EncodingConfig(reverse_bases=True)`); retain exact color-sequence matches |

The writer saves each channel as `input/round1/FOV_001/ch00.tif`, the round/FOV
layout that the dataset loader reads by default (`image_dir_template="{round}/{fov}"`; a
FOV-first tree `input/FOV_001/round1/` is read with `"{fov}/{round}"`), beside the
codebook and the truth files. The direct
constructor takes resolved sample input/output roots; this example does not use
a Snakemake YAML or `from_workflow_config` path expansion. `fov_pattern="FOV_{i:03d}"` is a
`{i}` pattern: `dataset.fov_ids(2, start=1)` names `FOV_001` and `FOV_002`, which the
quickstart also writes to `input/fovs.csv`, the FOV membership the Python workflow reads.

The synthetic channel order above differs from the wavelength-sorted order
`ch00, ch02, ch01, ch03` used for the real datasets. Do not transfer it to real
data without checking acquisition metadata. Noise detection uses a threshold of
median + 5 × MAD × 1.4826 per channel. Extraction half-widths and registration
shifts are in voxel indices, not micrometres or physical voxel spacing.

## Inspect outputs and check completion

A successful run exits with status 0 and prints `Quickstart checks passed`.
The recorded Python 3.12 run printed:

```text
FOV_001: 12 detected, 12 retained
FOV_002: 12 detected, 12 retained
```

These counts describe this software example. Retained codebook matches are not
proof of true molecules: in each FOV, 12 reads were retained from 10 generated
amplicons. This tutorial does not measure precision/recall or validate biological
accuracy. Noise, peak detection, image boundaries, and codebook membership can
affect recovery; do not require one output row per generated spot or tune real
data to these counts.

All paths below are relative to `QUICKSTART_OUTPUT`:

| Output | Schema and meaning |
| --- | --- |
| `input/<round>/FOV_*/ch*.tif` | Generated channel images, one ZYX `uint16` TIFF per round, FOV and channel |
| `input/generation.json` | Generator and preset versions, seed, dtype, requested/effective configuration and per-FOV provenance |
| `input/codebook.csv` | `entry_id,gene_id,color_sequence`; the entry ID is the nucleotide barcode, and the color sequence its four colors |
| `input/fovs.csv` | `sample,fov_id`: the two FOVs as one sample `synthetic`, the membership file of the Python workflow |
| `input/ground_truth.json` | Shape, seed, rounds, genes and FOV records; each spot has gene/barcode/color sequence and 0-based continuous `(z,y,x)` reference position; shifts are `(dz,dy,dx)` |
| `input/formed.csv`, `input/round_truth.csv`, `input/scene_truth.csv` | Formed amplicons (`gene_id`, `color_sequence`), their per-round positions and visibility, and the per-round view of the channel each color sequence selects (MATLAB-compatible `gene` and `color_seq` columns) |
| `results/fovs/FOV_*/reads.csv` | Every detected candidate with its read, as the quickstart requests it: `spot_namespace,spot_id,z,y,x,gene_id,observed_color_sequence,call_status` (`gene_id` is `<NA>` for an unassigned read) |
| `results/fovs/FOV_*/molecules.csv` | Filtered molecule candidates: `spot_namespace,spot_id,z,y,x,gene_id,entry_id,qc_score`; one row per retained read, not per cell (`qc_score` is `<NA>`: the quickstart does not score reads) |
| `results/fovs/FOV_*/run.json` | The run record of `FOV.run` with `CheckpointConfig(stages=())` (no stage files): round roles, steps, registration attempts, transforms (detected `(dz,dy,dx)` displacements as `displacement_zyx`), counts with the filter's fractions, the read summary and `outputs.molecules` ({doc}`checkpoints`, "run.json") |
| `summary.json` | Per-FOV generated/detected/retained counts, detected shifts, gene counts and molecule CSV paths |

In memory, image axes are `(Z,Y,X,C)` and spot coordinates are **0-based**.
Both exported tables keep those **0-based** `z,y,x` voxel coordinates and each
read's identity `(spot_namespace, spot_id)`; select coordinates by name. They are
typed CSVs: a missing value is written `<NA>`. There is no physical calibration,
cell ID, or cross-FOV stitching in these files.

`observed_color_sequence` concatenates the winning color of each round in
`round1`–`round4` order: a string `1`–`4` for the winning channel, `M` for a tie, or
`N` for a NaN maximum. Read it as a string. Filtration here uses exact codebook
membership; scores are diagnostics, not an additional filtering threshold.

The run record's `transforms` hold the **detected displacement**, the
translation transform's `displacement_zyx`: a moving round displaced by
`(dz,dy,dx)=(-2,3,2)` is recorded as `(-2,3,2)` and registered by
sampling it at `p + (-2,3,2)`. See the
[API contracts](api/contracts.md) for details.

The script asserts input shape/dtype, nonempty filtered outputs, valid codebook
labels, four-color sequence format, agreement of detected versus generated
shifts within one voxel per axis, the run record's status, counts, fractions and
transforms, CSV row counts, coordinate bounds and that the
molecule table read back with `load_molecules` holds the in-memory identities and
coordinates. To inspect the molecule tables yourself:

```bash
uv run python - <<'PY'
import os
from pathlib import Path
import pandas as pd

root = Path(os.environ["QUICKSTART_OUTPUT"])
for path in sorted((root / "results" / "fovs").glob("*/molecules.csv")):
    molecules = pd.read_csv(path, na_values=["<NA>"], keep_default_na=False)
    print(path.parent.name, molecules.shape)
    print(molecules.head().to_string(index=False))
    print(molecules.groupby("gene_id").size().to_string())
PY
```

If a check fails, inspect the traceback, `summary.json` if present, the CSVs
already written and the FOV's `run.json` (its `status` and `error`). Check the interpreter, installed lockfile environment,
channel order and input layout before changing processing parameters. The
[small API examples](api/examples.md) isolate individual contracts. Real-image,
MATLAB, local-registration, and cell-level workflows require separate validation.

## Complete source

```{literalinclude} examples/quickstart.py
:language: python
```

`generate_amplicon_dataset` returns a `SyntheticDataset` with the images and the truth tables, and `save_amplicon_dataset` writes it; `starfinder synthetic generate` streams each round to disk instead. See [synthetic contracts](api/synthetic.rst) for scene eligibility and truth/reproducibility limitations.
