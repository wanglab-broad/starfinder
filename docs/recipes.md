# Python recipes

Use these recipes to isolate image I/O, registration, detection and readout.
The [single-FOV demo](get-started/single-fov-demo.md) introduces the full route;
the [sample demo](get-started/multi-fov-demo.md) continues to cells and a sample.
The displayed functions come from the downloadable
{download}`recipes.py <examples/recipes.py>`;
{download}`quickstart.py <examples/quickstart.py>` prepares their input.

## Quick start

After [installation](get-started/installation.md), run from `src/python` with
new directories outside the checkout:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export QUICKSTART_OUTPUT=/tmp/starfinder-quickstart
export RECIPES_OUTPUT=/tmp/starfinder-recipes
uv run python ../../docs/examples/quickstart.py "$QUICKSTART_OUTPUT"
uv run python ../../docs/examples/recipes.py "$QUICKSTART_OUTPUT" "$RECIPES_OUTPUT"
```

Change the paths for another run. The first script prepares synthetic inputs
and processes two FOVs; the second reuses FOV_001. The tiny preset has four
rounds and four channels at 8 × 128 × 128 voxels, with seed 42.
Allow one CPU, 1 GiB RAM and 50 MiB of output space for these scripts.

## Read and write image stacks

Save a volume, load it back, then stack two channels in the requested order:

```{literalinclude} examples/recipes.py
:language: python
:pyobject: image_io
```

This example checks a uint16 TIFF round trip and a `(12, 24, 24, 2)` round.
Use {py:class}`~starfinder.io.ImageLoadConfig` for loading choices and
{py:class}`~starfinder.io.ImageConversionConfig` for intensity conversion.
For a ZYXC TIFF in Fiji, use Bio-Formats and its Hyperstack view.
The {doc}`I/O API <api/io>` covers other input formats.

## Register two volumes

Estimate a translation and apply the returned transform and application config:

```{literalinclude} examples/recipes.py
:language: python
:pyobject: register_volumes
```

The single-spot example detects `(dz, dy, dx) = (1, -2, 3)` and restores the
reference image. Follow the {doc}`registration guide <registration-contract>`
for image data; the transform's sign is already suitable for application.

## Detect spots in a volume

Add a channel axis to a known ZYX volume and compare two threshold modes:

```{literalinclude} examples/recipes.py
:language: python
:pyobject: detect_spots
```

Both calls find the spot at `(z, y, x) = (5, 10, 10)` in channel 0.
The {doc}`spot-finding reference <spot-finding-algorithms>` links the config
classes. Choose a threshold for its mode rather than copying a value between modes.

## Choose the 3D background radius

Use a radius larger than the puncta, then inspect a line profile through a dim
punctum before and after correction. `radius_voxels_zyx` chooses the footprint
in voxel coordinates; `radius_um_zyx` uses calibrated spacing. The
{py:class}`~starfinder.preprocessing.Background3DConfig` API gives the size rules.

```python
import numpy as np
from starfinder.preprocessing import Background3DConfig, subtract_background_3d

volume = np.full((12, 24, 24), 100, dtype=np.uint16)
volume[5, 10, 10] = 1000
corrected = subtract_background_3d(
    volume, config=Background3DConfig(radius_voxels_zyx=(4, 4, 4)))
```

## Choose a preprocessing recipe

For a constant background, try `ScalarBackgroundConfig`; for a spatially varying
background, try `Background3DConfig`. A recipe can follow background correction
with `PercentileNormalizationConfig`. Compare the dim signal before and after
processing rather than treating a synthetic example as a recommendation for real data.
The {doc}`preprocessing guide <preprocessing-contract>` shows a recipe, and
its {doc}`reference <preprocessing-algorithms>` lists methods and config classes.

## Decode one FOV

Build a dataset from the prepared input root and run a typed pipeline:

```{literalinclude} examples/recipes.py
:language: python
:pyobject: decode_fov
```

The constructor receives resolved input and output roots. The pipeline loads,
registers, detects, extracts, decodes and filters. `CheckpointConfig(stages=())`
writes `run.json` without stage payloads. Readout's
{doc}`guide <readout-contract>` explains the separate result objects.

## Inspect molecule outputs

Export molecules and reads, load the molecule table, and index the reference
image with the saved zero-based coordinates:

```{literalinclude} examples/recipes.py
:language: python
:pyobject: inspect_outputs
```

`results/fovs/FOV_001/` holds `molecules.csv`, `reads.csv` and `run.json`;
`summary.json` holds counts by gene and the molecule CSV path. The
{doc}`coordinate reference <api/contracts>` distinguishes Python ZYX coordinates
from the one-based XYZ in legacy molecule files.

## Limits

These examples check software behavior on synthetic inputs. They do not measure
biological accuracy or join molecules across FOVs. Use the sample demo for the
sample-level route.

## How it is checked

The scripts assert TIFF round trips, the detected translation, spot coordinates,
nonempty codebook-matched reads and exported identities. A successful recipe run
prints `All development recipes passed.`

The [longer recipe notes](https://github.com/wanglab-broad/starfinder/blob/dev/design/notes/recipes.md)
preserve background on preprocessing comparisons and report rendering.
