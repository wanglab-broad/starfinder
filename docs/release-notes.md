# Release notes

:::{container} sf-lead
What each tagged version of STARfinder changed for users, and what a version number
promises. Breaking Python changes and how to update callers are in the
[migration guide](migration.md).
:::

## Versioning

From 1.0.0, STARfinder's version numbers follow [semantic versioning](https://semver.org/).
A **major** version may break the public interface below; a **minor** version adds to it
without breaking it; a **patch** version fixes behaviour without changing the interface.
Every breaking change has a row in the [migration guide](migration.md).

The public interface is:

* the Python objects on the [Python API](api/python.rst) pages, with their
  documented parameters and results;
* the `starfinder` command and its documented subcommands and options;
* the keys of the [workflow configuration](workflow/reference.md);
* the documented files of a run: checkpoints, records, sample exports and reports, each
  with its own format version ({doc}`checkpoints`, {doc}`sample-output-contract`).

Not part of it: names that start with `_`, the scripts under `benchmarks/` and
`src/python/scripts/`, test helpers, the design records in `design/`, and the exact
bytes or digests of outputs, which can change with a codec, a dependency or a new
version. A documented file format that changes gets a new format version. The MATLAB
API and its workflow are frozen and keep their own conventions.

**Provisional in 1.0.** The benchmark commands `starfinder experiment` and
`starfinder trial`, and the layout of the saved-evaluation report, serve the
[Benchmark](benchmark/index.md) section and may change in a minor version; each change
still gets a migration row.

The version of an installed package comes from the Git tag (`v1.0.0`); a checkout
between tags reports the next development version, for example `1.0.1.dev3+g…`.

## 1.0.0

The first stable release of the Python package. The last release on `main` was the
MATLAB-era `v0.1.0` (2024); everything below is new since then. The `v0.2.0` to
`v0.4.2` tags were development tags on `dev`; their changes are summarised here.

**Processing, per FOV and per sample.** A typed Python package covers the whole route
from raw images to cells and gene counts:

* image I/O and preprocessing, including the ITK background reconstruction;
* registration: global translation, demons, and B-spline or TPS local registration;
* spot finding: classical detectors (adaptive threshold 0.2 by default) and the learned
  detectors Spotiflow and Piscis as optional extras;
* readout: extraction, decoding, read scores and decoder QC;
* segmentation and assignment: classical methods and StarDist or Cellpose as optional
  extras, with a dataset-tunable segmentation input recipe and
  `starfinder segment preview`;
* stitching with multiview-stitcher or BigStitcher; for single-plane tiles the
  multiview-stitcher link-quality threshold is estimated from the data (`"auto"`);
* sample export to AnnData and a fused 2D OME-TIFF, optionally SpatialData, and an HTML
  processing report for each sample.

**Workflow.** One command plans a run from a configuration file; Snakemake or Nextflow
runs the planned tasks on a laptop or a cluster, including Broad UGER with a container
image built from the lock. A run records the code version, inputs and outputs of each
task, and reruns only what a changed configuration or input affects.

**Deconvolution.** An optional `deconvolution` stage runs deconwolf v0.4.5 on each FOV
and round before the FOV tasks, with a PSF task per run, a read-only FFTW wisdom store
and `starfinder deconvolve wisdom --psf FILE`, which plans with the run's own PSF and
applies only at border quality 2, untiled. The container image builds deconwolf
reproducibly and stops when the binary differs from its pinned SHA-256. A wisdom store
is keyed by that binary: make it again with an image built from this release.

**Storage.** Registered rounds, prepared morphology images and the reference image are
written as TIFF + zlib by default (`execution.image_compression: none` writes them
uncompressed).

**Breaking changes since the development tags.** The named recipe `recipe: deep` is
removed in favour of explicit example configurations; segmentation `scale` and
`diameter` are replaced by `input_recipe`; FOV-level task fingerprints changed;
`starfinder deconvolve wisdom --psf FILE` replaces `--psf-shape`, and the wisdom store
format is `/2`, so earlier stores are refused. The [migration guide](migration.md) has
each row.

**Not covered by this release.** A software release checks software behaviour. It does
not qualify real data, molecular truth or the scientific comparisons of the
[Benchmark](benchmark/index.md) section, which have their own evidence.
