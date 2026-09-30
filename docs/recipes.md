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

{py:func}`~starfinder.io.save_volume` writes TIFF;
{py:func}`~starfinder.io.load_volume` reads one ZYX channel;
{py:func}`~starfinder.io.load_round` stacks channels in the supplied order.
This example preserves uint16 values and checks for unexpected cropping.

```{literalinclude} examples/recipes.py
:language: python
:pyobject: image_io
```

Expected: exact intensity round-trip, shape `(12,24,24,2)`, uint16, no cropping.
Use `ImageLoadConfig(conversion=ImageConversionConfig(...))` for explicit
cast/clip/rescale; it is not physical-unit conversion. Plain TIFFs are ZYX by
default. OME/ImageJ loading requires explicit selection for ambiguous T/C axes.
`save_volume` overwrites an existing file, so use a fresh destination.

`save_volume` writes a ZYX array as a plain tifffile TIFF and a ZYXC array as
OME-TIFF: every page is one YX plane, the OME-XML declares SizeZ, SizeC, the
pixel type and the dimension order `XYCZT`, and the
{py:class}`~starfinder.image.ImageMetadata` is kept in an OME-XML comment
annotation. Name ZYXC files `*.ome.tif`, which Bio-Formats uses to pick its OME
reader. {py:func}`~starfinder.io.load_volume_zyxc` reads the whole array back,
and it also reads ZYXC files written in the earlier tifffile layout. To open a
ZYXC volume in Fiji, use **File › Import › Bio-Formats** and choose
**Hyperstack** under *View stack with*; Bio-Formats opens it as a Z×C hyperstack
in its stored dtype.

For workflows, check `seq_channel_order`, path identifiers and `n_rounds` in
[top-level configuration](workflow-configuration.md#top-level-fields).
Python expects `input_root/round/FOV/*ch*.tif`. MATLAB's corresponding entry is
{mat:func}`LoadImageStacks`; see [channel conventions](conventions.md#channel-order).

## Register two volumes

{py:func}`~starfinder.registration.estimate_transform` returns a pull
transform and matching application config. Apply that result with
{py:func}`~starfinder.registration.apply_transform`:

```{literalinclude} examples/recipes.py
:language: python
:pyobject: register_volumes
```

Expected: detected displacement `(1,-2,3)`, exact equality after registration
because the single spot stays away from edges. Real data may lose
edge content; equality is not a general quality criterion.
{py:func}`~starfinder.registration.apply_transform` uses the displacement as it is. Supply equal-shaped 3D volumes; select or merge channels explicitly
before calling a single-channel function.

Workflow settings: `ref_round` and `rules.<rule>.parameters.global_registration`
(`run`, `ref_img`, `mov_img`) in the
[parameter reference](workflow-configuration.md#rule-resources-and-parameters).
The direct FOV API uses `merged`; workflow YAML uses `merged-image`.
For {mat:func}`DFTRegister3D` / {mat:func}`DFTApply3D`, use the
[MATLAB sign and axis convention](conventions.md#registration-signs).

## Detect spots in a volume

{py:func}`~starfinder.spot_finding.find_spots` accepts an explicit singleton channel axis, including
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

## Choose the 3D background radius

{py:class}`~starfinder.preprocessing.Background3DConfig` subtracts a grey
opening with an ellipsoidal footprint of semi-axes `(r_z, r_y, r_x)`. The
opening removes every bright structure that the footprint cannot fit inside,
so the radius sets the scale that separates puncta from background:

* **Larger than the puncta along each axis.** A punctum that fits inside the
  footprint is removed from the background estimate and kept, at full height,
  in the output. With a radius at or below the punctum's half-width, part of
  the punctum is counted as background and its peak is reduced.
* **Not much larger than needed.** Background that varies over distances shorter
  than the footprint is also kept in the output as if it were signal, and the
  cost grows with the footprint volume (about `4/3 π r_z r_y r_x` voxels).
* **Per axis.** Z sampling is usually coarser than XY, so `r_z` in voxels is
  usually smaller. Give `radius_um_zyx` to convert from micrometres with the
  image's `spacing_zyx` (rounded to whole voxels), or `radius_voxels_zyx`
  when spacing is unknown. `2r + 1` may not exceed the volume along any axis,
  which limits `r_z` for thin stacks.

For Gaussian puncta of width σ voxels per axis, the §2.5 synthetic evaluation
uses `r = ceil(3σ) + 1`; for example σ = (0.7, 1.0, 1.0) gives
`radius_voxels_zyx=(4, 4, 4)`, which needs at least 9 Z planes. On real data,
measure the puncta width first, and check a before/after line profile through
a dim punctum. The method, its cost and the evaluation design are in
{doc}`preprocessing-algorithms`.

```python
from starfinder.preprocessing import Background3DConfig, subtract_background_3d

corrected = subtract_background_3d(volume, config=Background3DConfig(radius_voxels_zyx=(4, 4, 4)))
```

## Choose a preprocessing recipe

**Defaults remain unchanged pending E13.** The pipeline default is still recipe 1,
and no recipe is recommended. The §2.5 synthetic comparison (task group 5) and
its inspection report (task group 6) are development evidence on uncalibrated
presets. E13 decides on real data. Until then, choose by the problem your
images show, and check a before/after line profile through a dim punctum.

| Recipe | Steps | Choose it when | Watch for |
| --- | --- | --- | --- |
| Recipe 1 (default, legacy) | `min_max_normalization` → `histogram_matching`, optionally `reconstruction` or `white_tophat` | You need the legacy outputs, which the golden test pins, or you are comparing with MATLAB results | The output is always uint8, and one bright voxel sets the min–max range. Histogram matching assumes balanced bases across channels. Reconstruction can drive MAD, and therefore the noise cutoff, to 0 |
| Recipe 2, scalar background | `scalar_background` → `percentile_normalization` | The background is mainly a constant offset per channel and round, and you want the input dtype preserved | `percentile=10` and `p_high=99.9` are provisional. About 0.1 % of voxels saturate by design |
| Recipe 2, 3D background | `background_3d` → `percentile_normalization` | The background varies in space and along Z, for example autofluorescence in thick tissue | The radius must exceed the puncta ([above](#choose-the-3d-background-radius)), and the cost grows with the footprint volume |

Two recipe modes apply to recipe 2:

* **Extraction source.** `extraction_source="bg_corrected"` reads intensities from
  the background-corrected image before normalization, which keeps linear
  channel ratios. Whether this helps depends on how the decoder handles
  channel scale.
* **Sample-level fitting.** `fit="supplied"` fits one set of statistics over the
  FOVs of a sample instead of per FOV. Consider it when FOVs differ strongly in
  content, such as near-empty fields at tissue edges; see
  {doc}`api/preprocessing` for the two-pass procedure.

The workflow key for an explicit recipe is described in
[Explicit preprocessing recipe](workflow-configuration.md#explicit-preprocessing-recipe).
The methods and the evaluation design are in {doc}`preprocessing-algorithms`.

### Reproduce one comparison

{download}`preprocessing_comparison.py <examples/preprocessing_comparison.py>`
reproduces the scalar-background comparison of the synthetic evaluation on a
tiny preset. The preset uses the `baseline` condition, 10×32×32 voxels, 12
amplicons, uint8, and seeds 0 and 100. The comparisons are isolated
(`none` → `scalar`) and ablation (`pct` → `r2_scalar`). The script reuses the
evaluation harness `benchmarks/preprocessing_synthetic.py`, so scenes, recipes,
detection, decoding and matching are those of the saved evaluation. It writes
no files. From `src/python`:

```bash
uv run python ../../docs/examples/preprocessing_comparison.py
```

```{literalinclude} examples/preprocessing_comparison.py
:language: python
:pyobject: main
```

Recorded outcome (Python 3.12.12, one CPU, about 1 second). On held-out seed
100, the isolated comparison leaves max-F1 unchanged at 0.909. It raises the
correct-decode fraction at `threshold_value=5` from 0.083 to 0.833, a change of
+0.750. The ablation changes neither endpoint. One seed on a tiny scene
illustrates the method. It is not an evaluation result, and the 2-point
low-benefit threshold is provisional.

### Render the inspection report

`benchmarks/preprocessing_report.py` renders a standalone HTML report from a
saved evaluation directory. The report is organized by method and does not
rerun the evaluation:

```bash
uv run python ../../benchmarks/preprocessing_report.py --evaluation /external/w233/evaluation --output /external/w234/report.html
```

Before it writes anything, the renderer checks the following:

* every file listed in the manifest has its recorded checksum;
* the manifest's revision matches the checkout;
* each regenerated panel image equals its manifest checksum.

Keep the report outside the checkout.

## Decode one FOV

Reuse the quickstart's prepared round/FOV layout. The direct
{py:class}`~starfinder.dataset.Dataset` constructor takes resolved roots;
it does not append dataset/sample IDs. Configure rounds with
{py:class}`~starfinder.dataset.RoundState`, load a codebook with
{py:meth}`~starfinder.dataset.Dataset.load_codebook`, then call
{py:meth}`~starfinder.dataset.FOV.load_images`,
{py:meth}`~starfinder.dataset.FOV.register`,
{py:meth}`~starfinder.dataset.FOV.find_spots`,
{py:meth}`~starfinder.dataset.FOV.extract_intensities`,
{py:meth}`~starfinder.dataset.FOV.decode_barcodes` and
{py:meth}`~starfinder.dataset.FOV.filter_reads` in order.

```{literalinclude} examples/recipes.py
:language: python
:pyobject: decode_fov
```

Recorded outcome: **10 candidates detected, 7 retained** for FOV_001. The script
checks nonempty codebook-matched output, not fixed counts as a biological target.
`EncodingConfig(reverse_bases=True)` matches this generator's encoding; real codebooks need their
own orientation check. `voxel_size=(1,2,2)` is a 3×5×5 extraction neighborhood
in ZYX, not physical spacing. No enhancement, local registration or suffix
filtering is enabled in this example.

Workflow settings: `seq_channel_order`, `n_rounds`, `ref_round`, plus the
`load_codebook`, `reads_extraction.voxel_size` and `reads_filtration` blocks in
[configuration](workflow-configuration.md#rule-resources-and-parameters).
The YAML wrapper is not identical to this direct API: its filtering field is
`end_base`, while the typed `ReadFilterConfig` field is `end_bases`. For base/color conversions
alone, see {py:func}`~starfinder.barcode.encode_bases` and
{py:func}`~starfinder.barcode.decode_color_sequence`.

## Inspect molecule outputs

{py:meth}`~starfinder.dataset.FOV.save_spots` adds one to coordinate columns
without mutating the in-memory table. Save all diagnostic columns explicitly
for candidates; the default filtered CSV contains `x,y,z,gene`.
{py:meth}`~starfinder.dataset.FOV.save_processing_log` writes a processing summary.

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
