# Minimal Python image-to-molecule quickstart

Generate a small synthetic dataset, register its sequencing rounds, detect spots,
extract their color sequences, and filter them against a codebook. The endpoint
is a CSV of **candidate molecules with gene labels**, one file per field of view
(FOV). Segmentation, cell assignment, cell expression matrices, and biological
validation are outside this example.

## Prerequisites and installation

Use a STARfinder checkout, `uv`, and Python 3.12 for the validated route below.
The package supports Python 3.10 or newer; other interpreters have not been
validated for this quickstart. A first installation needs package-registry access
or an existing uv cache. No downloaded microscopy data, MATLAB, Snakemake, GPU,
SimpleITK, or SpatialData is needed.

From the repository root:

```bash
cd src/python
export UV_PYTHON=python3.12
export PYTHONDONTWRITEBYTECODE=1
uv sync --locked --no-default-groups
uv run python --version
```

`UV_PYTHON` can also be the absolute path to your Python 3.12 interpreter.
Validation used conda-forge Python **3.12.12** on Linux host GP099-29C, selected
with `/home/unix/jiahao/miniforge3/bin/python3.12`, in an existing recorded uv
environment. This host's earlier Python 3.14 environment encountered a system
MKL linkage failure; the 3.12 selection changes no dependency pin or algorithm.
This is a host-specific observation, not a general incompatibility claim.

The documentation build additionally uses the `docs` dependency group; see
[contributing](contributing.md) for installation, strict build, and preview.

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
run took about 10 seconds and peaked at 277 MiB resident memory with these thread
settings. Runtime and memory depend on the host; these are observations, not
scheduler-enforced limits. A two-minute timeout was used during validation.

The executable source is included below so the displayed recipe and the script
being run remain the same. It uses the public
{py:class}`~starfinder.dataset.STARMapDataset` and
{py:class}`~starfinder.dataset.FOV` APIs.

## Input and processing configuration

The source is the repository's
{py:func}`~starfinder.benchmark.synthetic.get_preset_config` **tiny** preset,
generated locally with a fixed seed of **42**. The script saves the full generator
configuration to `synthetic_config.json` and generated truth to
`synthetic/ground_truth.json` (generator truth format version 2.0).

| Setting | Value |
| --- | --- |
| Input size | 2 FOVs, 4 rounds, 4 channels; each channel TIFF is `(8, 128, 128)` in `(Z, Y, X)` |
| Loaded round | `(8, 128, 128, 4)` in `(Z, Y, X, C)`, `uint8`; 32 TIFFs total, 4 MiB pixel payload |
| Spots and codebook | 10 generated spots per FOV; 8 synthetic genes, GeneA–GeneH; 5-base barcodes encode 4 colors |
| Signal and noise | Gaussian spot sigma 1.5 voxels; sampled peak intensity 200–255 before per-round jitter; background 20; Gaussian noise sigma 10 |
| Motion | Integer translations up to ±2 Z and ±5 Y/X voxels; no local deformation |
| Reference and registration | `round1`; global phase correlation on the sum of channels in reference and moving rounds |
| Loading | Default loading preserves the generated `uint8` values |
| Channel order | `ch00`, `ch01`, `ch02`, `ch03` maps to colors `1`, `2`, `3`, `4` |
| Detection | `intensity_estimation="noise"`, `intensity_threshold=5.0`, `min_distance=1` |
| Extraction | `voxel_size=(1, 2, 2)`: half-widths `(dz, dy, dx)`, a 3×5×5 voxel neighborhood |
| Codebook/filtering | `do_reverse=True` matches the generator's reversed-barcode encoding; retain exact color-sequence matches |

The generator writes `synthetic/FOV_001/round1/ch00.tif`; the dataset loader
expects `input/round1/FOV_001/ch00.tif`. The script copies the small TIFFs into
that second layout. Both layouts remain available for inspection. The direct
constructor takes resolved sample input/output roots; this example does not use
a Snakemake YAML or `from_config` path expansion.

The synthetic channel order above differs from the wavelength-sorted order
`ch00, ch02, ch01, ch03` used for the real datasets. Do not transfer it to real
data without checking acquisition metadata. Noise detection uses a threshold of
median + 5 × MAD × 1.4826 per channel. Extraction half-widths and registration
shifts are in voxel indices, not micrometres or physical voxel spacing.

## Inspect outputs and check completion

A successful run exits with status 0 and prints `Quickstart checks passed`.
The recorded Python 3.12 run printed:

```text
FOV_001: 10 detected, 7 retained
FOV_002: 13 detected, 13 retained
```

These counts describe this software example. Retained codebook matches are not
proof of true molecules: in FOV_002, 13 reads were retained from 10 generated
spots. This tutorial does not measure precision/recall or validate biological
accuracy. Noise, peak detection, image boundaries, and codebook membership can
affect recovery; do not require one output row per generated spot or tune real
data to these counts.

All paths below are relative to `QUICKSTART_OUTPUT`:

| Output | Schema and meaning |
| --- | --- |
| `synthetic_config.json` | Full synthetic configuration; `codebook: null` selects the built-in 8-gene test codebook; `background_std` is a compatibility field unused by generation |
| `synthetic/codebook.csv` | `gene,barcode`, where barcode is a nucleotide string |
| `synthetic/ground_truth.json` | Shape, seed, rounds, genes and FOV records; each spot has gene/barcode/color sequence and 0-based `(z,y,x)` position; shifts are `(dz,dy,dx)` |
| `synthetic/ground_truth_annotation_FOV_*.png` | Generated reference projections with truth annotations, not detected results |
| `results/signal/FOV_*_allSpots.csv` | All detected candidates, including coordinates, reference `intensity`, 0-based `channel`, per-round colors/scores, and `color_seq` |
| `results/signal/FOV_*_goodSpots.csv` | Filtered molecule candidates: exactly `x,y,z,gene`; one row per retained read, not per cell |
| `results/log/gr_shifts/FOV_*.txt` | CSV despite `.txt` suffix: `fov_id,round,row,col,z`; row/col/z are detected `(dy,dx,dz)` displacements |
| `results/log/FOV_*_rsf.txt` | Processing summary: backend, timestamp, rounds/reference, global shifts, local-registration state and spot counts |
| `summary.json` | Per-FOV generated/detected/retained counts, detected shifts, gene counts and molecule CSV paths |

In memory, image axes are `(Z,Y,X,C)` and spot coordinates are **0-based**.
Both exported signal CSVs use **1-based** `x,y,z` voxel coordinates for MATLAB
compatibility. Column order in `allSpots` starts with `z,y,x`; select coordinates
by name. Saving does not change the in-memory coordinates. There is no physical
calibration, cell ID, or cross-FOV stitching in these files.

A `{round}_color` is a string `1`–`4` for the winning channel, `M` for a tie, or
`N` for a NaN maximum. A `{round}_score` is `-log(maximum L2-normalized channel
intensity)`, with infinity for ties/NaNs. `color_seq` concatenates colors in
`round1`–`round4` order. Read it as a string. Filtration here uses exact codebook
membership; scores are diagnostics, not an additional filtering threshold.

Shift logs contain **detected displacement**, not the correction translation;
the registration API applies its negative internally. For example, a detected
`(dz,dy,dx)=(-2,3,2)` requires correction `(2,-3,-2)`. See the
[API contracts](api/contracts.md) for details.

The script asserts input shape/dtype, nonempty filtered outputs, valid codebook
labels, four-color sequence format, agreement of detected versus generated
shifts within one voxel per axis, CSV row counts, coordinate bounds and the
1-based export conversion. To inspect both output tables yourself:

```bash
uv run python - <<'PY'
import os
from pathlib import Path
import pandas as pd

root = Path(os.environ["QUICKSTART_OUTPUT"])
for path in sorted((root / "results" / "signal").glob("*_goodSpots.csv")):
    molecules = pd.read_csv(path)
    print(path.name, molecules.shape)
    print(molecules.head().to_string(index=False))
    print(molecules.groupby("gene").size().to_string())
PY
```

If a check fails, inspect the traceback, `summary.json` if present, and the CSVs
and logs already written. Check the interpreter, installed lockfile environment,
channel order and input layout before changing processing parameters. The
[small API examples](api/examples.md) isolate individual contracts. Real-image,
MATLAB, local-registration, and cell-level workflows require separate validation.

## Complete source

```{literalinclude} examples/quickstart.py
:language: python
```
