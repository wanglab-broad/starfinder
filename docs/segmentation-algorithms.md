# Segmentation algorithm specification

Status: Proposed

This page specifies, for each §2.9 segmentation method and each reusable function, the
problem it addresses, its parameters with units and defaults, its failure behavior
(including an image with no foreground) and its resource estimate. The registry,
configs, label contract and records they plug into are in {doc}`segmentation-contract`;
the current behavior is in {doc}`segmentation-baseline`. Nothing here is implemented,
and nothing here ranks the methods, sets a default model or makes an accuracy claim.
The validation design for the whole section is W-308's.

## Evidence

Every measured number comes from W-306, run directory
`/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/runs/W-306/20261004T194036Z-74e12949`
(found with `timeout 30 find …/runs/W-306 -maxdepth 2 -name segmentation-manifest.json`;
`notes.md` sections 3 to 7; tables `segmentation-calls.csv`, `cost-summary.csv`,
`comparisons.csv`, `fov-memory-estimate.json`, `label-dtype-probe.json`), or from W-305
(`runs/W-305/20261004T0630Z-summary/models.md` and `examples.md`). W-307 repeats no
measurement. The measurements ran on GP099-29C (one RTX A5000, driver 580) with StarDist
0.9.2, TensorFlow 2.20.0 and Cellpose 4.2.1.1 with torch 2.7.1; one CPU thread unless
stated. Where W-306 did not measure a function, the estimate below is derived from the code
and labelled as such. The W-306 and W-305 limitations apply throughout; the full list is
in {doc}`segmentation-contract` ("Limitations of the evidence").

Each statement of failure or edge behavior below names what supports it: a W-306 row or
section (measured with the library), the W-307 golden test (`test_segmentation_golden.py`,
run on the unchanged scripts), a probe in the locked environment recorded in the W-307
worker notes, or the code (read, not run). Statements about functions that do not exist
yet are marked as proposed rules.

### How to read the resource figures

W-306 measured calls inside job processes (notes section 4). Every figure below says which
of these kinds it is:

* **Wall** and **CPU**: per-call wall time, and the whole process's CPU time over the call
  (all threads), both measured. Process start-up, imports and the model load are outside
  them; the load has its own row. The learned main and repeat jobs ran a warm-up call after
  the load; the watershed job did not, so its first call includes first-call costs.
* **Peak RSS**: the process's resident-memory high-water mark during the call, reset before
  it. It includes what the model load and earlier calls in the same job left resident; the
  RSS before the call is given in brackets where it matters. It is not the memory one call
  needs on its own.
* **Framework GPU peak**: TensorFlow's or torch's peak allocation during the call. It
  excludes the CUDA context.
* **Process GPU memory**: nvidia-smi's figure for the job process, read after the call. It
  includes the allocator caches and the memory that earlier calls grew (TensorFlow keeps
  what it grows to), so it is per job, not per call, and not interchangeable with the
  framework peak.
* **Projected**: a CPU cost W-306 did not measure, with its basis.
* **Code-derived estimate**: W-307's reading of the code, for functions W-306 did not run.

The one W-306 call without a CPU time (the stopped pilot Cellpose CPU call at diameter
None, whose wall time and peak RSS come from the orchestrator and the watchdog) is not
used on this page.

## What each element addresses

| Element | Kind | Problem addressed |
| --- | --- | --- |
| `stardist` | method | Star-convex objects, above all nuclei, in 2D (tissue-2D PI with `2D_versatile_fluo`) or 3D (LN DAPI with `3D_spleen`) |
| `cellpose` | method | Nuclei or whole cells of irregular shape, from one or two stains, without a trained in-house model (culture with `cpsam_v2`) |
| `seeded_watershed` | method | Cell territories grown from nuclei on a stain (amplicon, cytoplasm or composite), with each cell carrying its nucleus's label |
| `composite_nuclei_amplicon` | input function | One image in which both nuclei and amplicon-filled cytoplasm are bright, the legacy `overlay` input |
| `enhance_with_flamingo` | input function | DAPI with the Flamingo (cytoplasm) stain subtracted, which separates touching nuclei |
| `normalize_percentiles` | input function | The float intensity range StarDist models expect |
| `rescale_input`, `labels_to_grid` | input and label functions | Nucleus detection on a shrunk grid by a method without its own scale, with the labels returned to the exact full-resolution grid |
| `expand_labels` | label function | Territories around nuclei when no cell stain exists, applied once |
| `extend_labels_through_z` | label function | A 3D territory for single-layer culture from 2D labels and a stain |
| `import_labels` | import | Masks made elsewhere (CellProfiler, earlier workflow runs) in the same contract |

## Methods

### `stardist`

**Algorithm.** Load the model folder (`StarDist2D` or `StarDist3D` by the model's
`n_dim`); normalize the input channel with `normalize_percentiles` (csbdeep's
`normalize`, float32, unclipped); call `predict_instances(image, scale=…,
n_tiles=…, prob_thresh=…, nms_thresh=…)`. StarDist resamples the input by `scale`
(`ndi.zoom`, order 1), predicts object probabilities and ray distances, runs non-maximum
suppression on the CPU and renders the surviving polygons or polyhedra on the original
grid, so the labels are on the input grid at every scale (W-306 notes section 6). A Z=1
input to a 2D model is squeezed to YX and the axis restored.

**Parameters** (`StarDistConfig`):

| Parameter | Unit | Default | Notes |
| --- | --- | --- | --- |
| `model` or `model_path` | | required (one) | known `2D_versatile_fluo`; user-trained `3D_spleen` by path |
| `scale` | factor (Y, X, or ZYX) | required | W-305 observations, not settings: 1.0 for `3D_spleen` on LN, 0.25 for `2D_versatile_fluo` on tissue-2D PI. `3D_spleen` stores anisotropy (2.286, 1, 1) but has no spacing parameter; a per-axis scale is the only route (spacing (1, 2, 2) fixture: nothing at scale 1, 2 labels at (1/2.286, 2, 2)). |
| `prob_thresh` | probability | `None` = stored | `3D_spleen` 0.6429, `2D_versatile_fluo` 0.4791 |
| `nms_thresh` | overlap | `None` = stored | `3D_spleen` 0.5, `2D_versatile_fluo` 0.3 |
| `normalize_percentiles` | percent | (1.0, 99.8) | the current script's values |
| `n_tiles` | tiles per axis | `None` = 1×4×4 (volume), 2×2 (plane) | csbdeep may use fewer tiles than requested (W-306 `seam-tiles.json`) |

**How outputs move with the parameters** (W-306 GPU rows; boundaries, not tuning):
`3D_spleen` on LN `dapi_round4` gives 3, 253 and 426 labels at scale 0.5, 0.75 and 1.0,
and 535, 426 and 60 at prob 0.5, the stored 0.643 and 0.8; `2D_versatile_fluo` on the
tissue PI crop gives 39, 62 and 56 labels at scale 0.25, 0.5 and 1.0, and 39, 39 and 15
at prob 0.3, the stored 0.479 and 0.8.

**Failure behavior.**

* No foreground: an all-zero image gives an empty label image without error, on CPU and
  GPU (W-306 `f2:empty`, `f3:empty` rows); the result has `outcome` `empty` (proposed
  rule). `3D_spleen` returns uint16 when it finds nothing and int32 otherwise (W-306
  section 14, all rows); the wrapper converts both to `uint32` (proposed rule).
* Background noise: `2D_versatile_fluo` returns one spurious object along the image
  border, `3D_spleen` none (W-306 `f2:noise`, `f3:noise` rows). No filter is applied; the
  worker notes keep object filtering open.
* The current script on an image with no Otsu component raises at `areas.max()` (W-306
  parity probe `probe_zeros_64x64_2d`; W-307 golden test); the method has no gate.
* Wrong dimensionality and Z=1: rejected by the stage wrapper before the model loads
  (proposed rule). Unwrapped, StarDist 2D on 1×64×64 raises a bare `ValueError`
  (W-306 `f2:z1`) and `3D_spleen` on one plane returns nothing where it finds the same
  blobs in a volume (W-306 `f3:z1`), which is why Z=1 is refused for 3D models.
* Small inputs: no minimum is enforced by the library; inputs from 8×8 run and may be
  empty (W-306 `probe:2d_8x8` and the other `probe:` minimum rows). `min_shape_zyx` stays
  (1, 1, 1).
* Missing or changed model files: `MissingModelError` or `ModelHashMismatchError` before
  loading (proposed rule). Loading a model with the wrong class raises the library's
  `ValueError` about the grid (W-306 `loading-probes.json`); the dimensionality check runs
  first, so it is not reached.
* Low thresholds slow the CPU non-maximum suppression: on the GPU, prob 0.5 took 72 s of
  per-call wall time against 54.6 s at the stored value, and the un-normalized 16×64×64
  fixture took 176–180 s on either device (W-306 section 4; the cause was not measured).
  The method sets no time limit; the cost is the caller's.
* Tiling: on the CPU, tiled and untiled labels are identical; on the GPU, seam-crossing
  labels keep IoU ≥ 0.9996 on the tissue crop, and all 135 seam-crossing LN labels are
  identical to the untiled GPU run (W-306 seam tables). The 3D seam fixture found no
  object, so the 3D statement rests on one LN crop.

**Resources** (measured per call, `cost-summary.csv` and `segmentation-calls.csv`; the
kinds are defined under "How to read the resource figures"):

| Case | CPU, 1 thread | CPU, 4 threads | GPU |
| --- | --- | --- | --- |
| Model load (`load` rows) | `3D_spleen` 0.3 s wall; `2D_versatile_fluo` 0.2 s | | `3D_spleen` 1.0 s, `2D_versatile_fluo` 0.9–1.2 s wall |
| `3D_spleen`, LN 50×512×512, scale 1, 1×4×4 tiles | 226 s wall, 197 s CPU; peak RSS 2,245 MiB (887 MiB before the call) | 58 s wall, 207 s CPU; peak RSS 2,256 MiB | 54.6 s wall, 38.7 s CPU; framework GPU peak 2,707 MiB; process GPU memory 8,922 MiB (job, after the call); peak RSS 1,999 MiB (1,329 MiB before) |
| `3D_spleen`, the other LN crops | 165–190 s wall; peak RSS up to 2,338 MiB | | 3.7–29.5 s wall; framework GPU peak 1,183 MiB |
| `3D_spleen`, LN untiled / 1×2×2 tiles | not run (projected 220 s / 200 s) | | 62.4 s wall, framework GPU peak 8,245 MiB, process GPU memory 17,114 MiB / 42.5 s, 3,074 MiB framework peak |
| `2D_versatile_fluo`, 1024² at scale 0.25 | 1.2 s wall; peak RSS 868 MiB (839 MiB before) | 1.0 s wall | 3.0 s wall (the job's first call at this size); framework GPU peak 231 MiB |
| `2D_versatile_fluo`, 1024² at scale 1.0 | 1.7–1.8 s wall; peak RSS up to 1,041 MiB | | 0.8–1.8 s wall; framework GPU peak up to 454 MiB |

A whole LN field of view (50×1496×1496) was not run. The configuration gives 3,442 MiB
for four dense arrays (leaving out the process baseline, tile activations, workspace and
copies, and assuming dense distances), and a linear extrapolation of the 50×512×512 CPU
call's peak RSS gives about 12,500 MiB and 1,930 s; both are estimates, not bounds
(`fov-memory-estimate.json`). A whole FOV therefore needs a measurement, StarDist's
block-wise prediction or the §2.10 tiling (W-306 choice 7).

### `cellpose`

**Algorithm.** Load `CellposeModel(pretrained_model=<checked absolute path>)`; pass the
channels in role order (`cytoplasm` then `nuclear`, or `nuclear` alone) with
`channel_axis`; call `eval` with the recorded `diameter`, thresholds, `tile_overlap`,
`min_size`, a new normalization mapping and, in 3D mode, `do_3D=True`, `z_axis=0` and
`anisotropy`. Cellpose rescales the image by 30/`diameter`, pads it by 8 pixels a side,
cuts it into 256-pixel tiles whose count per axis is ceil((1 + 2 × `tile_overlap`) ×
size / 256), blends the network outputs over the overlaps, resizes the flows back and
computes masks at the input size (W-306 section 6). 3D mode runs the 2D network on the
YX, ZY and ZX planes.

**Parameters** (`CellposeConfig`):

| Parameter | Unit | Default | Notes |
| --- | --- | --- | --- |
| `model` or `model_path` | | required (one) | known `cpsam_v2` |
| `diameter` | pixels | required | W-305 observation, not a setting: 240 for culture cells; W-306 also ran 60 for nuclei. `None` (explicit) means no rescale and is refused in 3D mode, where diameter None on 14-pixel objects gave 85 fragments (W-306). |
| `do_3d`, `anisotropy` | -, Z/YX spacing ratio | `False`, `None` | 3D mode needs both; `anisotropy=0.5` recovered the 4 objects of the spacing (1, 2, 2) fixture |
| `flow_threshold`, `cellprob_threshold` | -, logit | 0.4, 0.0 | library defaults; W-306 sensitivity rows at diameter 240: cellprob −1 gives 31 labels, the defaults 34, flow 0.8 gives 41 |
| `tile_overlap` | fraction | 0.1 | clipped to [0.05, 0.5] by Cellpose; 0.1 against 0.5 changed outlines (median matched IoU 0.954, 131,716 pixels) |
| `bfloat16` | | `True` | library default on CPU and GPU |
| `min_size` | pixels | 15 | library default |
| `normalize_percentiles` | percent | (1.0, 99.0) | per channel, over the whole stack in 3D mode |

**Failure behavior.**

* No foreground: an all-zero image gives an empty label image without error; background
  noise gives none (W-306 `f2:empty`, `f2:noise` rows).
* A missing model path: Cellpose itself silently loads the default `cpsam_v2` (and would
  download it) (W-306 `loading-probes.json` and code reading of `cellpose/models.py`); the
  wrapper's model check raises first (proposed rule).
* More than 65,535 masks: Cellpose switches from uint16 to uint32 with a warning (W-306
  code reading of `cellpose/dynamics.py` and `label-dtype-probe.json`); the wrapper
  converts to `uint32` either way (proposed rule).
* One plane: the Z axis is dropped by Cellpose (W-306 `f2:z1`, `f3:z1` rows) and restored by
  the wrapper (proposed rule).
* `normalize=False` gives no labels (W-306 `iso_unnormalized` row); Starfinder always passes
  its normalization.
* Seams on dense images: isolated objects on tile edges stay whole (49 of 49, W-306
  `seam-summary.csv`), but moving the tile grid by half a tile changed 3 of 22 interior
  culture cells beyond IoU 0.5, and every interior cell lay on a seam in both layouts
  (W-306 `seam-shift.csv`, choice 10, open).

**Resources:**

| Case | CPU, 1 thread | CPU, 4 threads | GPU |
| --- | --- | --- | --- |
| Model load (`load` rows) | 2.7 s wall; peak RSS 2,435 MiB (679 MiB before) | 2.3 s wall | 2.7 s wall; framework GPU peak 1,760 MiB; process GPU memory 2,020 MiB |
| culture cells 1024² × 2 channels, diameter 240 | 203 s wall, 191 s CPU; peak RSS 1,718 MiB (1,342 MiB before) | 69 s wall, 267 s CPU; peak RSS 1,740 MiB | 1.5 s wall; framework GPU peak 860 MiB; process GPU memory 2,376 MiB (job, after the call) |
| culture nuclei 1024², diameter 60 | 229 s wall; peak RSS 2,361 MiB (1,465 MiB before) | | 0.6 s wall; framework GPU peak 1,503 MiB |
| any input within one 256-pixel tile | 22–25 s wall | | 0.04–0.08 s wall |
| 3D mode, 42×512×512 × 2, diameter 240 | not run; projected 3,015–3,833 s per call (129–164 tiles at the 23.4 s median single-tile CPU call, 3D flow dynamics not included) | | 6.2 s wall; framework GPU peak 1,504 MiB; peak RSS 2,430 MiB (1,855 MiB before) |

On the CPU, flow dynamics take 200/rescale iterations (1,600 at diameter 240), which
W-306 estimates at about 180 s of the 203 s call (the call minus one tile's 23 s, not a
separate measurement). Cellpose 3D mode is GPU-only in practice.

### `seeded_watershed`

**Algorithm** (the W-306 prototype `scripts/seeded_watershed.py`). Convert the stain with
`img_as_float`; smooth it with a Gaussian of `sigma_um / spacing` pixels per axis (an axis
of length 1 is not smoothed); threshold it (Otsu or a number); run
`skimage.segmentation.watershed(-smooth, markers=seeds, mask=(smooth > threshold) or
(seeds > 0), connectivity, compactness)`. The output carries the seed labels, so cell k
contains nucleus k, and it runs on the full-resolution grid with seeds and stain on that
grid (the W-305 rule).

**Parameters** (`SeededWatershedConfig`): `sigma_um` [µm], default 1.5 (provisional; the
W-305 tissue-2D run used 16 pixels × 0.094635 µm = 1.514 µm); `threshold`, default
`"otsu"`, or a number on the `img_as_float` scale ([0, 1] for unsigned integers);
`compactness` 0.0; `connectivity` 1. The spacing comes from the input metadata and is
required.

**Failure behavior** (W-306 section 7, measured on the prototype):

* No seeds: an empty label image, no error (W-306 `no_seeds_2d` rows).
* No foreground (an all-zero stain): the threshold is 0, the mask holds only the seeds,
  and every cell equals its seed (W-306 `empty_stain_2d`).
* A seed outside the foreground is always kept and becomes a cell equal to the seed
  (W-306 `seed_outside_foreground_2d`); on tissue-2D, 3 of 39 nuclei lie entirely outside
  the Otsu foreground and 5 cells equal their nucleus (`tissue2d_amplicon`).
* Different shapes, a spacing of the wrong length or non-integer seeds raise (W-306
  prototype code, `scripts/seeded_watershed.py`); missing spacing raises (proposed rule).
* Repeats are identical (11 cases, each twice); the function has no random state. Giving
  the spacing (1, 2, 2) instead of ignoring it raised the matched IoU against the
  construction cells from 0.964 to 0.980, a geometry check, not accuracy.

**Resources.** Measured per call on CPU, one thread: 0.22–0.25 s wall and 157–161 MiB
peak RSS (121–141 MiB before the call) on a 1024² tissue image; at most 5 ms on the
fixtures. The watershed job had no warm-up call and no watchdog, and its `/usr/bin/time`
record covers only the part after the last reset. A 3D field of view was not measured.

## Input functions

### `composite_nuclei_amplicon`

**Algorithm** (the current `create_nuclei_amplicon_overlay.py`, kept bit for bit). Stretch
the amplicon image between its 0.001 and 0.999 quantiles and the nuclear image between
its 0.005 and 0.995 quantiles (`np.quantile`, linear, over the whole volume) with
`rescale_intensity`; convert both with `img_as_float`; take their voxel-wise maximum;
convert with `img_as_ubyte`. The implementation takes the maximum without stacking the
two images (the result is identical), and an optional projection is the run's
`projection`, not part of the function.

**Parameters** (`CompositeConfig`): `nuclear_quantile` 0.005 and `amplicon_quantile`
0.001, fractions; inputs ZYX on one grid with equal metadata (Z=1 for a plane). Output
uint8 ZYX.

**Failure and edge behavior.**

* A constant image is not set to zero. Its two quantiles are equal, and
  `rescale_intensity` then clips the image to the output range (the uint8 range) instead
  of stretching it (code: `skimage/exposure/exposure.py`, scikit-image 0.26), so its grey
  level passes through unchanged: a constant nuclear image of 50 and a constant amplicon
  image of 80 give an all-80 composite, and a constant nuclear image of 50 with the fixture
  amplicon gives the amplicon composite floored at 50 (W-307 golden test
  `test_constant_and_zero_inputs`, `CONSTANT_DIGESTS`).
* An all-zero nuclear image is the constant 0: the composite is the stretched amplicon
  alone; two all-zero images give an all-zero image; neither raises (golden test).
* Mismatched shapes raise a broadcasting `ValueError` today (W-307 probe, worker notes);
  the function raises `IncompatibleGeometryError` instead, before any computation, and
  `FOV.segment` checks the metadata (proposed rule).
* YX inputs raise `AxisError` today (golden test `test_composite_needs_3d_inputs`); the
  function takes Z=1 for a plane (proposed rule).

**Resources.** Not measured by W-306. Code-derived estimate: two float64 copies and their
maximum, about 24 bytes per voxel beyond the uint8 inputs, so about 2.7 GB for a
50×1496×1496 field of view (the current script's stacked copy adds 16 bytes per voxel,
about 4.5 GB in total, above the 4 GiB target). Measured on the golden fixture (16×64×64,
CPU, one thread): about 10 ms per call (pytest durations).

### `enhance_with_flamingo`

**Algorithm** (the current `enhance_dapi_with_flamingo.py`, kept bit for bit). Stretch the
Flamingo image between its 0.005 and 0.995 quantiles, median-filter each Z plane with a
disk of radius 1 pixel, stretch the nuclear image between its 0.001 and 0.999 quantiles,
convert both to float and return `img_as_ubyte(nuclear × (1 − flamingo))`.

**Parameters** (`FlamingoEnhancementConfig`): `flamingo_quantile` 0.005,
`nuclear_quantile` 0.001 (fractions), `median_radius_px` 1 (pixels, per plane). Inputs
ZYX on one grid; a plane is Z=1 (today a YX input raises).

**Failure and edge behavior.**

* Constant images pass through the stretch unchanged, as for the composite: a constant
  nuclear image of 50 with a constant Flamingo image of 80 gives 50 × (1 − 80/255), an
  all-34 image (golden test `test_constant_and_zero_inputs`).
* An all-zero Flamingo image leaves the stretched nuclear image, and an all-zero nuclear
  image gives zeros; neither raises (golden test, `CONSTANT_DIGESTS`).
* YX inputs raise `RuntimeError` in the per-plane median today (W-307 probe, worker notes);
  the function takes Z=1 for a plane (proposed rule).
* Mismatched shapes raise `IncompatibleGeometryError` (proposed rule).

**Resources.** Not measured by W-306. Code-derived estimate: about three float64 arrays of
the volume, about 2.7 GB for 50×1496×1496; the per-plane median adds one plane. Measured
on the golden fixture (CPU, one thread): about 20 ms per call.

### `normalize_percentiles`

**Algorithm.** `(x − p_low) / (p_high − p_low + 1e-20)` in float32 with the two
percentiles (NumPy, linear) over `axes`, unclipped: csbdeep's `normalize`, which the
`stardist` method calls for the W-306 parity.

**Parameters.** `p_low` 1.0 and `p_high` 99.8 [percent of the intensity distribution];
`axes` all spatial axes of one channel.

**Failure behavior.** A constant image maps to zeros: x − p_low is 0 everywhere (code;
csbdeep 0.8.2 `normalize` on a constant image returns float32 zeros, W-307 probe in the
locked environment). Without normalization StarDist gives different objects (2D: 3 labels,
none matching; 3D: 1 label) and Cellpose none; a scaled image (×10) gives identical
results after normalization (W-306 `iso_unnormalized` and `iso_x10*` rows, section 6).

**Resources.** One float32 copy: 427 MiB for 50×1496×1496, a code-derived estimate from
the model configuration (W-306 `fov-memory-estimate.json`).

### `rescale_input` and `labels_to_grid`

**Algorithm.** `rescale_input` resamples the image by per-axis factors (linear, with
anti-aliasing when shrinking) and returns metadata whose spacing is divided by the
factors (when it has a spacing) and whose `frame_id` records the rescale. `labels_to_grid`
maps a label image back by nearest neighbour by pixel centres (source index ⌊(i + 0.5) ×
n_source / n_target⌋ per axis) onto exactly the target shape, so an odd size keeps its grid
(the current round trip turns 61×67 into 60×68: W-306 P3; and 61×63 into 60×64: golden
test `test_rescale_round_trip_changes_an_odd_grid`).

**Parameters.** `scale_zyx`, dimensionless factors > 0; the target shape.

**Rules and failure behavior** (proposed rules). For direct callers detecting nuclei; a
seeded run rejects seeds whose grid differs from its input's, so a cell run never uses a
shrunk grid. Factors ≤ 0 raise. An axis that would shrink below one pixel keeps one
(scikit-image's `rescale` already does this, W-307 probe).

**Resources.** Not measured. Code-derived estimate: one float copy of the rescaled image
and one label array of the target grid.

## Label functions

### `expand_labels`

**Algorithm.** `planar`: scikit-image `expand_labels` on each Z plane (the current
script and `reads_assignment.py`), distance in pixels or converted from µm with the Y, X
spacing. `volumetric`: the same scikit-image function on the whole volume with its
`spacing` argument set to the ZYX spacing, so each background voxel within the physical
Euclidean distance takes the label of its nearest labelled voxel; ties resolve as the
distance transform resolves them.

**Parameters** (`ExpandLabelsConfig`): `distance` (no default), `unit` `pixel` or `um`,
`mode` `planar` or `volumetric`. The legacy translation is `planar`, `pixel`, the YAML
`distance` (W-306 parity used 4).

**Failure behavior.** An empty label image stays empty, and distance 0 changes nothing
(scikit-image 0.26 `expand_labels`, W-307 probe in the locked environment); expansion is
per plane in 3D today (golden test `test_expansion_is_per_slice`). `unit="um"` without
spacing raises (proposed rule). Applied once per run; the assignment never expands again
(W-308).

**Resources.** Not measured. Code-derived estimate from scikit-image 0.26's implementation: float64 distances,
int32 indices per axis, two masks and the output, about 22 bytes per pixel of one plane in
`planar` mode (about 50 MB for a 1496² plane) and about 26 bytes per voxel in `volumetric`
mode (about 2.9 GB for 50×1496×1496, close to the 4 GiB target together with the input
labels, so `volumetric` on a whole FOV needs a measurement, blocks or the §2.10
tiling).

### `extend_labels_through_z`

**Algorithm** (the Python form of `create_3d_segmentation.m`, per FOV). Median-filter each
plane of the stain; one threshold over the filtered stack (Otsu or a number); per plane:
fill holes, remove objects below `min_area_um2`, dilate by `dilation_um`, fill holes again
(cells), and multiply the mask by the 2D labels. The result has geometry `extended`.

**Parameters** (`ZExtensionConfig`, physical units): `median_um`, `threshold` (`"otsu"`),
`min_area_um2`, `dilation_um`, `fill_holes` (`once` or `twice`). The MATLAB values, in
pixels, are a 10×10 median, 200-pixel (cells) or 10-pixel (nuclei) minimum areas and disks
of radius 10 (cells) or 5 (nuclei); the workflow adapter translates them with the
configured spacing. No default is proposed until a hand-built fixture pins the
translation; MATLAB was not run in this batch, so there is no parity with the example.

**Failure behavior** (proposed rules; the function does not exist yet and MATLAB was not
run). Labels with Z>1 raise (the function takes a plane label image, shape (1, Y, X), and a
ZYX stain with the same Y, X); an empty mask gives an empty label image with `outcome`
`empty`. The MATLAB `Cyto = Cell − Nuclei` is not reproduced (W-308 defines compartments).

**Resources.** Not measured. Code-derived estimate: per-plane filters on one plane at a
time; the output is one `uint32` volume (448 MB for 50×1496×1496, or 4 bytes per voxel).

## Import

`import_labels` reads a YX or ZYX integer TIFF, converts the byte order to native,
validates dtype, sign, shape against the given `ReferenceGrid` and the metadata, and
records the file and array hashes and the grid ({doc}`segmentation-contract`, "External-mask
import"). Failure behavior (proposed rules): a float or boolean mask raises `TypeError`; a
negative value, a shape different from `grid.shape_zyx` or a metadata mismatch raises; an
all-zero mask is a valid `empty` result. The W-305 culture references are big-endian
(W-306 section 6), which the byte-order conversion handles. Resources (code-derived
estimate): one copy of the file's array and its `uint32` conversion.

## Limitations

* Every number is from one host, one field of view per context and few images; the
  synthetic fixtures are outside the models' training data (`3D_spleen` finds nothing in
  the touching fixture).
* There is no annotation, so nothing here measures accuracy; agreement with the legacy
  culture labels only shows two results agree (W-306 section 12).
* The CPU costs of GPU-only calls, Cellpose 3D mode on the CPU and whole fields of view are
  projections or estimates; the composite, Flamingo, expansion, extension and rescale
  estimates above come from the code, not from measurements.
* Peak RSS and process GPU figures include what the model load and earlier calls left in
  the job process, and framework GPU peaks exclude the CUDA context; none of them is the
  memory one call needs on its own ("How to read the resource figures").
* The Cellpose seam behavior on dense images and the 3D StarDist seam behavior beyond one
  LN crop are open.
