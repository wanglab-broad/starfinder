# Installation

STARfinder is a Python package in `src/python` of the repository, with optional extras
for backends, a separate Snakemake environment for the workflow and a MATLAB backend.
The [single-FOV demo](single-fov-demo.md) and the [quickstart](../getting-started.md) need
only the package itself; the [multi-FOV / sample demo](multi-fov-demo.md) also needs the
`multiview-stitcher` and `anndata` extras.

## The Python package

Use a STARfinder checkout, `uv`, and Python 3.12 for the validated route below.
The package requires Python 3.11 or newer; other interpreters have not been
validated for the demo. A first installation needs package-registry access
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
Validation used conda-forge Python **3.12.12** on Linux. On one host an earlier
Python 3.14 environment met a system MKL linkage failure; the 3.12 selection changes
no dependency pin or algorithm. This is a host-specific observation, not a general
incompatibility claim. `src/python/uv.lock` records the exact resolved versions;
`--locked` stops when the lock does not match `pyproject.toml`.

## Optional extras

Each extra of `src/python/pyproject.toml` installs the packages of one optional
backend. `uv sync` installs only the extras it is given: name every extra you need on
the same command, and again on every later `uv sync`, for example

```bash
uv sync --locked --no-default-groups --extra local-registration --extra checkpoint
```

| Extra | Installs | Needed for |
| --- | --- | --- |
| `ome` | bioio-ome-tiff ≥1.0 | Installed on request; no current public function switches to it ({doc}`../api/backends`) |
| `local-registration` | SimpleITK ≥2.3 | Demons estimation and application; the B-spline grid of the elastix route ({doc}`../api/backends`) |
| `registration-elastix` | itk-elastix 0.25.4, itk 5.4.7 | Rigid, affine and B-spline registration through elastix ({doc}`../api/backends`) |
| `spotiflow` | spotiflow 0.6.5, torch 2.7.1, torchvision 0.22.1 | The learned spot detector Spotiflow (below) |
| `piscis` | piscis 1.1.0, torch 2.7.1, torchvision 0.22.1 | The learned spot detector Piscis (below) |
| `stardist` | stardist 0.9.2, csbdeep 0.8.2, tensorflow 2.20.0 | StarDist segmentation (below) |
| `cellpose` | cellpose 4.2.1.1, torch 2.7.1, torchvision 0.22.1 | Cellpose segmentation (below) |
| `multiview-stitcher` | multiview-stitcher 0.1.62 | The multiview-stitcher stitching method, and the fused image and label export of `Dataset.export` |
| `checkpoint` | pyarrow ≥15 | Parquet tables (`table_format: "parquet"`) in checkpoints and records ({doc}`../checkpoints`) |
| `spatialdata` | spatialdata ≥0.1, spatialdata-io ≥0.1 | The SpatialData store of `Dataset.export`; without it the export records that part as not run |
| `anndata` | anndata 0.12.6 | The per-FOV `raw.h5ad` of the workflow's assignment rule and the sample-level AnnData |
| `visualization` | napari ≥0.4 | Installed on request; no current public function switches to it ({doc}`../api/backends`) |
| `dev` | pytest ≥7.0, pytest-cov ≥4.0, ruff ≥0.1 | Running the tests and the linter |

The repository's CI installs the `dev`, `local-registration` and `checkpoint` extras.
Besides the extras, `pyproject.toml` has two dependency groups: `dev` (pytest, ruff and
Jupyter for the example notebooks) and `docs`, the documentation build
([contributing](../contributing.md)). `--no-default-groups` leaves out both; add
`--group docs` or `--group dev` to install one.

## Learned extras

The learned spot detectors (`spotiflow`, `piscis`) and the learned segmenters
(`stardist`, `cellpose`) carry the marker `python_version < '3.14'`: torch 2.7.1 and
TensorFlow 2.20.0 have no wheel for Python 3.14, so they need Python 3.11 to 3.13 and
install nothing on Python 3.14 and later. On Linux, `[tool.uv.sources]` takes torch and
torchvision from the CPU index of PyTorch, so the environment holds the CPU build.
Importing `starfinder.spot_finding` or `starfinder.segmentation` and constructing a config
never need the extras; running a learned method without its extra raises an
`ImportError`, `SpotFindingBackendUnavailableError` or
`SegmentationBackendUnavailableError`.

Processing never downloads a model: the weights are fetched explicitly beforehand.
Installing the spot detectors, fetching their pretrained weights and configuring a run
are in
{doc}`../learned-detectors`; the segmentation models and their environment are in the
[segmentation contract](../segmentation-contract.md#models).

## Snakemake

The workflow runs under Snakemake 9, in an environment separate from the package's uv
environment. `config/environment-v9.yaml` describes a conda environment with Snakemake 9
(Python ≥3.11, <3.13) and the cluster-generic executor:

```bash
conda env create -f config/environment-v9.yaml
conda activate starfinder-v9
```

When the workflow uses the Python backend, or runs the segmentation and assignment rules
on either backend, install the package and the extras those rules need into the
environment that runs Snakemake; StarDist runs there with the `stardist` extra
({doc}`../workflow-downstream`). To check a configuration and the job graph without
creating that environment, the [dry run](../workflows.md#small-reproducible-dry-run)
adds Snakemake to a uv run with `uv run --with 'snakemake>=9,<10'`. Snakemake is not part
of the `docs` group. The {doc}`../workflow/index` section describes the workflow and its
configuration.

## MATLAB

The MATLAB backend (`backend: matlab`, the workflow's default) targets **MATLAB R2023b or
newer** with the Image Processing Toolbox; the full list of requirements is in
[the MATLAB API](../api/matlab.md#runtime-requirements-and-validation-limits). The
workflow starts MATLAB through `matlab_launcher`: the default `path` runs the executable
named by `STARFINDER_MATLAB_EXECUTABLE`, or `matlab` on `PATH`
([MATLAB launcher](../workflow-configuration.md#matlab-launcher)). The Python package,
the single-FOV demo and the documentation build need no MATLAB or MATLAB license.
