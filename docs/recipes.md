# Development recipes

Use these small Python recipes to isolate I/O, registration, detection and
barcode processing before configuring a full workflow. Each displayed function
comes from the executable {download}`recipes.py <examples/recipes.py>`; its
assertions check software behavior on small synthetic inputs.

## Run the recipes

Install the [quickstart environment](getting-started.md#prerequisites-and-installation)
first. From `src/python`, choose two **new directories outside the checkout**:

```bash
export UV_PYTHON=python3.12
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export QUICKSTART_OUTPUT=/absolute/external/quickstart-inputs
export RECIPES_OUTPUT=/absolute/external/development-recipes
export MPLCONFIGDIR="$RECIPES_OUTPUT-matplotlib"
uv run python ../../docs/examples/quickstart.py "$QUICKSTART_OUTPUT"
uv run python ../../docs/examples/recipes.py "$QUICKSTART_OUTPUT" "$RECIPES_OUTPUT"
```

If you already completed the quickstart, set `QUICKSTART_OUTPUT` to that directory
and run only the second command. Both scripts refuse to reuse their output
directory. The recipes reuse the tiny preset's seed **42**, TIFFs and codebook;
they process only **FOV_001** (4 rounds of `(8,128,128,4)` uint8). The independent
I/O, registration and detection examples use a deterministic `(12,24,24)` uint16
array with one bright voxel, without randomness. Allow one CPU, 1 GiB RAM and
50 MiB disk for the combined commands; these are planning bounds, not enforced
allocations. No optional backend, MATLAB, Snakemake or downloaded data is needed.

The Python examples were executed with conda-forge **Python 3.12.12** on Linux
GP099-29C using the existing uv environment. Other environments are unverified.
The recipe run took about 5 seconds and peaked at 190 MiB resident memory;
the quickstart preparation took about 10 seconds. Success ends with
`All development recipes passed.` Expected outcomes appear below.
MATLAB and UGER links describe source-checked interfaces; they are
**runtime-unverified instructions**, not executed portions of these recipes.
For fuller input details and limitations, see the [quickstart](getting-started.md).

## Read and write image stacks

{py:func}`~starfinder.io.save_stack` writes TIFF;
{py:func}`~starfinder.io.load_multipage_tiff` reads one ZYX channel;
{py:func}`~starfinder.io.load_image_stacks` stacks channels in the supplied order.
This example preserves uint16 values and checks for unexpected cropping.

```{literalinclude} examples/recipes.py
:language: python
:pyobject: image_io
```

Expected: exact intensity round-trip, shape `(12,24,24,2)`, uint16, no cropping.
`convert_uint8=True` instead min-max scales non-uint8 input; it is not a unit
conversion. Plain TIFFs must already have ZYX axes. OME/ImageJ loading selects
`T=0,C=0`; a generic multi-channel TIFF is not automatically split into channels.
`save_stack` overwrites an existing file, so use a fresh destination.

For workflows, check `seq_channel_order`, path identifiers and `n_rounds` in
[top-level configuration](workflow-configuration.md#top-level-fields).
Python expects `input_root/round/FOV/*ch*.tif`. MATLAB's corresponding entry is
{mat:func}`LoadImageStacks`; see [channel conventions](conventions.md#channel-order).

## Register two volumes

{py:func}`~starfinder.registration.phase_correlate` measures displacement;
{py:func}`~starfinder.registration.apply_shift` applies a translation. Here
`fixed` is the volume returned by `image_io`.

```{literalinclude} examples/recipes.py
:language: python
:pyobject: register_volumes
```

Expected: detected `(1,-2,3)`, correction `(-1,2,-3)`, exact equality after
correction because the single spot stays away from edges. Real data may lose
edge content; equality is not a general quality criterion.
{py:func}`~starfinder.registration.register_volume` performs the negation
internally. Supply equal-shaped 3D volumes; select or merge channels explicitly
before calling a single-channel function.

Workflow settings: `ref_round` and `rules.<rule>.parameters.global_registration`
(`run`, `ref_img`, `mov_img`) in the
[parameter reference](workflow-configuration.md#rule-resources-and-parameters).
The direct FOV API uses `merged`; workflow YAML uses `merged-image`.
For {mat:func}`DFTRegister3D` / {mat:func}`DFTApply3D`, use the
[MATLAB sign and axis convention](conventions.md#registration-signs).

## Detect spots in a volume

{py:func}`~starfinder.spotfinding.find_spots_3d` requires a channel axis, even
for one channel. This example uses the same uint16 volume and exercises two
explicit mode/threshold pairs.

```{literalinclude} examples/recipes.py
:language: python
:pyobject: detect_spots
```

Expected: one spot at internal `(z,y,x)=(5,10,10)`, channel `0`, in both modes.
This zero-background illustration is not a noise-model validation. Noise mode
uses a k-sigma multiplier; adaptive mode uses a fraction of channel maximum.
Do not carry a threshold of `5.0` from noise mode into adaptive mode.

Workflow settings: `rules.<rule>.parameters.spot_finding.intensity_estimation`
and `.intensity_threshold` in the
[parameter reference](workflow-configuration.md#rule-resources-and-parameters).
See [threshold conventions](conventions.md#spot-finding-thresholds) for MATLAB
{mat:func}`SpotFindingMax3D` and wrapper limitations.

## Decode one FOV

Reuse the quickstart's prepared round/FOV layout. The direct
{py:class}`~starfinder.dataset.STARMapDataset` constructor takes resolved roots;
it does not append dataset/sample IDs. Configure rounds with
{py:class}`~starfinder.dataset.LayerState`, load a codebook with
{py:meth}`~starfinder.dataset.STARMapDataset.load_codebook`, then call
{py:meth}`~starfinder.dataset.FOV.load_raw_images`,
{py:meth}`~starfinder.dataset.FOV.global_registration`,
{py:meth}`~starfinder.dataset.FOV.spot_finding`,
{py:meth}`~starfinder.dataset.FOV.reads_extraction` and
{py:meth}`~starfinder.dataset.FOV.reads_filtration` in order.

```{literalinclude} examples/recipes.py
:language: python
:pyobject: decode_fov
```

Recorded outcome: **10 candidates detected, 7 retained** for FOV_001. The script
checks nonempty codebook-matched output, not fixed counts as a biological target.
`do_reverse=True` matches this generator's encoding; real codebooks need their
own orientation check. `voxel_size=(1,2,2)` is a 3×5×5 extraction neighborhood
in ZYX, not physical spacing. No enhancement, local registration or suffix
filtering is enabled in this example.

Workflow settings: `seq_channel_order`, `n_rounds`, `ref_round`, plus the
`load_codebook`, `reads_extraction.voxel_size` and `reads_filtration` blocks in
[configuration](workflow-configuration.md#rule-resources-and-parameters).
The YAML wrapper is not identical to this direct API: its filtering field is
`end_base`, while the FOV keyword is `end_bases`. For base/color conversions
alone, see {py:func}`~starfinder.barcode.encode_bases` and
{py:func}`~starfinder.barcode.decode_color_seq`.

## Inspect molecule outputs

{py:meth}`~starfinder.dataset.FOV.save_signal` adds one to coordinate columns
without mutating the in-memory table. Save all diagnostic columns explicitly
for candidates; the default filtered CSV contains `x,y,z,gene`.
{py:meth}`~starfinder.dataset.FOV.save_log` writes a processing summary.

```{literalinclude} examples/recipes.py
:language: python
:pyobject: inspect_outputs
```

Expected: 10 candidate rows, 7 molecule-candidate rows; converting exported
ZYX coordinates back to zero-based indices reproduces the in-memory table.
The script checks bounds before indexing the reference image. `summary.json`
contains counts by gene and the molecule CSV path. `results/signal/` contains
both CSVs, and `results/log/` contains the FOV summary and detected shift log.

These are per-FOV voxel coordinates and gene labels, with no cell IDs or
cross-FOV stitching. Codebook membership alone does not establish true molecules.
Read `color_seq` as a string to preserve sequence semantics; `M`/`N` mark
ambiguous/NaN extraction. See the [quickstart output schema](getting-started.md#inspect-outputs-and-check-completion),
[workflow output map](workflows.md) and
[downstream input requirements](workflow-downstream.md) before cell assignment.
