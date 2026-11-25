# STARfinder

[![Tests](https://github.com/wanglab-broad/starfinder/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/wanglab-broad/starfinder/actions/workflows/tests.yml)
[![Documentation](https://github.com/wanglab-broad/starfinder/actions/workflows/docs.yml/badge.svg)](https://wanglab-broad.github.io/starfinder/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE.txt)

STARfinder turns raw microscopy images from STARmap and related spatial transcriptomics
methods into molecules, cells and gene counts. It processes each field of view (FOV),
stitches the FOVs into a sample, and exports each sample with a processing report. Use
it from Python, run whole datasets with Snakemake or Nextflow, or use the frozen MATLAB
backend.

**Documentation:** <https://wanglab-broad.github.io/starfinder/>

## Get started

The Python package lives in `src/python`. With [uv](https://docs.astral.sh/uv/) and
Python 3.12:

```bash
git clone https://github.com/wanglab-broad/starfinder.git
cd starfinder/src/python
UV_PYTHON=python3.12 uv sync --locked --no-default-groups
```

Then run the two demos, which need no microscopy data, GPU or MATLAB:

1. [Single FOV demo](https://wanglab-broad.github.io/starfinder/get-started/single-fov-demo.html):
   one synthetic FOV to a table of molecules
   ([notebook](example/introduction/single_fov_demo.ipynb)).
2. [Multi-FOV / sample demo](https://wanglab-broad.github.io/starfinder/get-started/multi-fov-demo.html):
   a small synthetic sample to cells, exports and a report
   ([notebook](example/introduction/sample_demo.ipynb)).

[Installation](docs/get-started/installation.md) lists the optional extras: learned spot
detectors and segmenters, elastix registration, multiview-stitcher, AnnData, SpatialData
and deconvolution.

## What it does

* **Per FOV:** preprocessing, registration (global, demons, B-spline, TPS), spot finding,
  barcode extraction and decoding with QC, segmentation and assignment of reads to cells.
* **Per sample:** stitching (multiview-stitcher or BigStitcher), export to AnnData and a
  fused OME-TIFF (optionally SpatialData), and an HTML processing report.
* **Whole datasets:** one command plans a run from a configuration file; Snakemake or
  Nextflow runs it on a laptop or a cluster. See the
  [workflow guide](https://wanglab-broad.github.io/starfinder/workflow/index.html).

## Datasets

Test datasets:

1. HeLa cell culture dataset (cell-culture-3D) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11176779.svg)](https://doi.org/10.5281/zenodo.11176779)
2. Mouse brain tissue section dataset (tissue-2D) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11176779.svg)](https://doi.org/10.5281/zenodo.11176779)

Additional sources: [Single Cell Portal](https://singlecell.broadinstitute.org/single_cell/study/SCP2637).

## Broad / Wang Lab users

The workflow runs on the UGER cluster with the provided profiles and container image; see
[Broad UGER](https://wanglab-broad.github.io/starfinder/workflow/broad-uger.html).
The frozen MATLAB workflow keeps its own environment,
`conda env create -f workflow/envs/matlab-legacy.yaml`, and its
[guide](https://wanglab-broad.github.io/starfinder/workflow/matlab-legacy.html).

## Versions and changes

Releases follow semantic versioning from 1.0.0. See the
[release notes](docs/release-notes.md) for what each version changed and what counts as
the public interface, and the [migration guide](docs/migration.md) for updating callers of
earlier Python interfaces.

## Contributing

[Contributing](docs/contributing.md) describes the environment, the test selections and
the documentation build; [CONTEXT.md](CONTEXT.md) holds the vocabulary and package
responsibilities, and [AGENTS.md](AGENTS.md) the operating instructions for agents.

## Citation

TBA

## Contact

For questions and help requests, please use the
[issue tracker](https://github.com/wanglab-broad/starfinder/issues).

## License

[MIT](LICENSE.txt)
