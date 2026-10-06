# Segmentation baseline: morphology inputs, StarDist and label files

Status: Accepted (W-309, 2026-10-05, at 3550723)

This page records how segmentation and its inputs behave at revision `6b384cd`
(branch `runner/s29-spec-20261004`, on `dev` after the §2.8 work), before the
Chapter II §2.9 work changes them. It is the reference that the golden test
`src/python/test/test_segmentation_golden.py` pins. The proposed replacement is
described in {doc}`segmentation-contract` and {doc}`segmentation-algorithms`; neither
is accepted. Paths are relative to the repository root, and line numbers are at
`6b384cd`. The assignment half of §2.9 (`reads_assignment.py`, compartments, counts)
is recorded by W-308; this page names it only where segmentation reaches into it.

Measured behavior of the current StarDist script comes from W-306, run directory
`/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/runs/W-306/20261004T194036Z-74e12949`
(found with `find … -maxdepth 2 -name segmentation-manifest.json`; notes `notes.md`,
sections 3 and 6; tables `tables/parity.csv`, `tables/label-dtype-probe.json`; parity
outputs `parity/`).

## Where segmentation lives today

The Python package has no segmentation code. Everything below is a Snakemake script,
an inline rule body or a MATLAB example:

| Piece | Where | Engine |
| --- | --- | --- |
| StarDist 2D/3D with a foreground gate, normalization, rescale, tiling, expansion and a `uint16` cast | `workflow/scripts/stardist_segmentation.py` (rule `stardist_segmentation`, `workflow/rules/segmentation.smk:21-35`) | the conda environment `{envs_path}/stardist` |
| DAPI–amplicon composite | `workflow/scripts/create_nuclei_amplicon_overlay.py` (rule `create_nuclei_amplicon_overlay`, `segmentation.smk:39-48`) | workflow environment |
| Flamingo enhancement of DAPI | `workflow/scripts/enhance_dapi_with_flamingo.py` (rule `enhance_dapi_with_flamingo`, `segmentation.smk:8-17`) | workflow environment |
| DAPI rotation and projection | inline `run:` of rule `rotate_nuclei` (`workflow/rules/registration-py.smk:88-111`; the same lines in `registration.smk:80-103`) | workflow environment |
| Morphology rounds into the sequencing frame | rule `nuclei_registration` (`registration-py.smk:74-84`, `registration.smk:62-76`): `workflow/scripts/nuclei_registration.py` → `starfinder.dataset.workflow._run_nuclei_registration` → `FOV.register_rounds`, or `nuclei_registration.m` | Python package or MATLAB |
| Label preview | `workflow/scripts/create_segmentation_preview.py`; no rule uses it | none |
| Culture extension of 2D labels through z | `example/sequential_workflow/create_3d_segmentation.m` | MATLAB, by hand |
| Second label expansion | `workflow/scripts/reads_assignment.py:52-57`, `:64-67` | workflow environment |

## `stardist_segmentation.py` step by step

| Step | Lines | Behavior at `6b384cd` |
| --- | --- | --- |
| Parameters | 13-14, 26-27, 31, 40, 42, 46-47, 51, 60-61 | Every parameter is read inside the script from `snakemake.config['rules']['stardist_segmentation']['parameters']`: `prob_thresh`, `nms_thresh`, `stardist_model_name`, `stardist_base_path`, `rescale`, `expand_labels`, `distance`. There are no defaults; a missing key raises `KeyError`. The thresholds are always taken from YAML, never from the model's `thresholds.json`. |
| Input | 16 | `imread(snakemake.input[0])`: one image of any dtype; its rank decides everything below. No metadata is read. |
| Foreground gate | 17-23 | Otsu threshold of the whole image, strict `>`; connected components (full connectivity); `areas.max() > 100`, a count of pixels or voxels with no physical unit. If the largest component is not larger, the model is never loaded and an all-zero `uint16` image of the input shape is written (64-66). An image with no component makes `areas.max()` raise `ValueError: zero-size array to reduction operation maximum which has no identity` before that fallback, and no file is written (W-306 parity probe `probe_zeros_64x64_2d`). On background noise the gate passes (W-306 `probe_noise_64x64_2d`, where `2D_versatile_fluo` then returns one object). |
| Dimensionality | 24 | `len(current_img.shape) == 3` chooses `StarDist3D`, anything else `StarDist2D`. A single plane stored as 1×Y×X goes to the 3D model; a 2D image goes to the 2D model; nothing checks that the model's dimensionality matches. |
| Model | 25-27, 45-47 | `StarDist3D(None, name=<stardist_model_name>, basedir=<stardist_base_path>)` reads `config.json`, `thresholds.json` and the weights from that folder; it does not download (W-306 `tables/loading-probes.json`). Nothing records the model's name, path or file hashes. |
| Rescale (optional) | 31-32, 51-52 | With `rescale: true`, `skimage.transform.rescale(img, [1, .5, .5])` (2D `[.5, .5]`): a fixed factor 0.5 in Y and X, with scikit-image's default Gaussian anti-aliasing, returning float64. |
| Normalization | 29, 33, 37, 49, 53, 57 | `csbdeep.utils.normalize(img, 1, 99.8, axis=(0,1,2))` (2D `(0,1)`): one percentile pair over the whole image, unclipped, not recorded. |
| Prediction | 34, 38, 54, 58 | `model.predict_instances(img, n_tiles=[1, 4, 4], …)` in 3D, `n_tiles=[2, 2]` in 2D, with the YAML thresholds; `details` is discarded. |
| Inverse rescale | 35, 55 | `rescale(labels, [1, 2, 2], order=0, preserve_range=True)`: nearest neighbour, so outlines were computed on the shrunk grid. The labels come back on `round(round(n × 0.5) × 2)` pixels per axis, which differs from the input whenever a Y or X size is odd (W-306 P3: 61×67 in, 60×68 out). The file is then written on that grid. |
| Expansion (optional) | 40-42, 60-61 | `skimage.segmentation.expand_labels(…, distance=<distance>)`, in pixels; in 3D one call per Z plane, so labels grow in Y and X only. |
| Cast and write | 63 | `imsave(output, labels.astype('uint16'), compression='zlib')`. Labels above 65,535 wrap: 65,536 becomes 0 (background) and 65,537 becomes 1 (W-306 `tables/label-dtype-probe.json`). `tifffile.imsave` no longer exists in the locked tifffile 2026.1.28, so the script runs only in the legacy environment (tifffile 2024.2.12). The TIFF has no axes or geometry description. |

The rule (`segmentation.smk:21-35`) reads `images/{segmentation_input_folder}/{fovID}.tif`,
where `segmentation_input_folder` comes from `common.smk:199` with the default
`overlay`, and writes `images/stardist_segmentation/{fovID}.tif`. It runs in the conda
environment `{envs_path}/stardist` with 2 threads and records a Snakemake benchmark
file. W-306 ran the unchanged script (sha256 `5a2347f8…`) in the legacy environment on
CPU and a step-by-step prototype in the spike environment; on all 12 parity runs they
are equal pixel for pixel, and both raise on the all-zero probe (W-306 notes section 3,
`tables/parity.csv`).

### Configuration keys: `stardist_segmentation_config`

`workflow/schemas/config.schema.yaml:1382-1416`, referenced from
`rules.stardist_segmentation` (`:254-255`):

| Key | Schema | Script |
| --- | --- | --- |
| `run` | boolean, required | rule enabled (`common.smk`, `ALWAYS_AVAILABLE_RULES`) |
| `resources` | `#/$defs/resources` | `mem_mb`, `runtime` of the rule |
| `parameters.stardist_base_path` | string | `basedir` of the model |
| `parameters.stardist_model_name` | string | `name` of the model |
| `parameters.segmentation_input_folder` | string | folder under `images/` read by the rule; default `overlay` in `common.smk:199` |
| `parameters.prob_thresh`, `parameters.nms_thresh` | numbers in [0, 1] | passed to `predict_instances`; required by the script |
| `parameters.rescale` | boolean | the fixed 0.5 shrink and its inverse |
| `parameters.expand_labels` | boolean | per-plane expansion |
| `parameters.distance` | integer ≥ 0 | expansion distance in pixels |

The block and its `parameters` allow additional properties, and no key has a schema
default. {doc}`workflow-configuration` lists the same keys.

## The composite: `create_nuclei_amplicon_overlay.py`

| Step | Lines | Behavior |
| --- | --- | --- |
| Inputs | 10-11 | `dapi_img` = `images/DAPI/{fovID}.tif` (from `rotate_nuclei`) and `amplicon_img` = `images/ref_merged/{fovID}.tif` (the reference round's channel maximum, written by the sequencing rules; see {doc}`workflows`). No shape, frame or dtype check. |
| Amplicon contrast | 14-18 | `np.quantile` (linear) at 0.001 and 0.999 over the whole volume, then `rescale_intensity(img, in_range)` onto the dtype range. |
| DAPI contrast | 21-25 | The same at 0.005 and 0.995. |
| Combination | 28-33 | Both to float with `img_as_float`; stacked on a new last axis; the maximum over `axis=3`. The axis is hard-coded, so the inputs must be 3D: two YX images raise `AxisError` (golden test). Inputs of different shapes raise a broadcasting `ValueError`. |
| Projection | 35-36 | With the rule parameter `maximum_projection` (read directly; a missing `parameters` block raises `KeyError`), the Z maximum. |
| Output | 38-40 | `img_as_ubyte`, written with `imwrite` and no metadata, to `images/overlay/{fovID}.tif`. |

A constant image is not stretched: its two quantiles are equal, and `rescale_intensity`
then clips it to the output range, which leaves its grey level unchanged (scikit-image
0.26). So an all-zero DAPI image stays zero and the composite is the stretched amplicon
alone, and a constant DAPI image of 50 floors the composite at 50; neither raises. The
golden test pins the output with and without projection (the projected output equals the
Z maximum of the 3D one) and on constant and all-zero inputs.

## Flamingo enhancement: `enhance_dapi_with_flamingo.py`

| Step | Lines | Behavior |
| --- | --- | --- |
| Inputs | 13-14 | `images/flamingo/DAPI/{fovID}.tif` and `images/flamingo/Flamingo/{fovID}.tif` |
| Flamingo | 17-23 | quantile stretch at 0.005/0.995, then a median filter with `disk(1)` on each Z plane. The plane loop assumes 3D: YX inputs raise `RuntimeError: footprint.ndim (2) must match len(axes) (1)`. |
| DAPI | 26-30 | quantile stretch at 0.001/0.999 |
| Combination | 33-40 | `dapi × (1 − flamingo)` in float, `img_as_ubyte`, written with no metadata to `images/flamingo/enhanced_DAPI/{fovID}.tif` |

Constant inputs pass through the stretch unchanged, as in the composite: a constant DAPI
image of 50 and a constant Flamingo image of 80 give an all-34 image, an all-zero Flamingo
image leaves the stretched DAPI, and an all-zero DAPI image gives zeros (golden test).

The two inputs are written by the Python `nuclei_registration` adapter
(`src/python/starfinder/dataset/workflow.py:784-795`) for an additional round named
`flamingo` with channels named `DAPI` and `Flamingo`, but the rule declares only its
two log files as outputs (`registration-py.smk:77-80`), so Snakemake cannot connect the
two rules; {doc}`workflow-downstream` says the folders "must be supplied separately".
The golden test pins the enhancement on its fixture.

## `rotate_nuclei` and the DAPI image

`registration-py.smk:88-111` (identical in `registration.smk:80-103`):

* `get_dapi_input` globs `{INPUT_DIR}/{dapi_round}/{fovID}/*ch04.tif` and returns the
  list unchecked. With no match the rule body fails at `input[0]` (`IndexError`); with
  several, the first in unsorted directory order is read.
* The image is read from the raw input of `dapi_round`. No registration is applied, so
  it is in the reference frame only when `dapi_round` is the reference round. The
  morphology registration of `nuclei_registration` uses the reference round's `ch04` as
  its reference, which is the same image only in that case.
* `scipy.ndimage.rotate(img, rotate_angle, axes=(1, 2))` with SciPy's defaults
  (`reshape=True`, cubic spline). For multiples of 90° this equals `np.rot90` with the
  same sign convention as `FOV.run`'s rotation (checked on a 3×8×10 array). For other
  angles the grid grows (30° turns 8×10 into 12×13), while `FOV.run`'s rotation
  (`_rotated`, `src/python/starfinder/dataset/fov.py:56-75`) keeps the shape with
  bilinear interpolation, so `images/DAPI` and `images/ref_merged` are then on
  different grids.
* With top-level `maximum_projection`, the Z maximum; written by `skimage.io.imsave`
  with no metadata to `images/DAPI/{fovID}.tif`.

## Morphology rounds into the sequencing frame

With `backend: python`, `nuclei_registration.py` calls `_run_nuclei_registration`
(`workflow.py:748-795`). It loads each `additional_round` with its `channel_order`,
rotates it by `rotate_angle`, reads the reference round's `ch04` (rotated the same way)
as an `ExternalReference`, and calls `FOV.register_rounds` (`fov.py:572-676`) with one
translation step on the round's shared stain, the one channel whose name contains the
top-level `ref_channel`. Every channel of the round is resampled through that
transform, and the round takes the reference metadata; `registration_record["rounds"]`
keeps the reference label and its SHA-256 ({doc}`coordination`, "Other rounds and
external references"). It writes `log/{fovID}_nr.txt`, `log/gr_shifts/{fovID}_nr.txt`
and `images/<round>/<channel name>/{fovID}.tif` (ZYX, or the Z maximum with
`maximum_projection`, dtype kept). With the MATLAB backend `nuclei_registration.m`
does the same and also stretches the other rounds. The registered sequencing
reference, `images/ref_merged/{fovID}.tif`, is the reference round's channel maximum
written by `FOV.save_reference_image` (`fov.py:1534-1580`) or by MATLAB.

## Files each rule reads and writes

Paths are relative to the output directory unless marked INPUT.

| Rule or script | Reads | Writes |
| --- | --- | --- |
| `nuclei_registration` | config JSON; INPUT additional rounds (declared); INPUT reference round `ch04` (not declared) | `log/{fovID}_nr.txt`, `log/gr_shifts/{fovID}_nr.txt` (declared); `images/<round>/<channel name>/{fovID}.tif` (not declared) |
| `rotate_nuclei` | INPUT `{dapi_round}/{fovID}/*ch04.tif` (glob) | `images/DAPI/{fovID}.tif` |
| `create_nuclei_amplicon_overlay` | `images/DAPI/{fovID}.tif`, `images/ref_merged/{fovID}.tif` | `images/overlay/{fovID}.tif` (uint8) |
| `enhance_dapi_with_flamingo` | `images/flamingo/DAPI/{fovID}.tif`, `images/flamingo/Flamingo/{fovID}.tif` | `images/flamingo/enhanced_DAPI/{fovID}.tif` (uint8) |
| `stardist_segmentation` | `images/{segmentation_input_folder}/{fovID}.tif` | `images/stardist_segmentation/{fovID}.tif` (uint16, zlib); `log/benchmark/stardist_segmentation/{fovID}.txt` |
| `reads_assignment` | `images/DAPI/{fovID}.tif`, `images/stardist_segmentation/{fovID}.tif` and the tables listed in {doc}`workflow-downstream` | `expr/{fovID}/raw.h5ad`, `expr/{fovID}/reads_assignment.csv` and diagnostics |
| `create_segmentation_preview.py` (no rule) | `snakemake.input['overlay_img']`, `snakemake.input['segmentation']` | an RGB TIFF: the overlay in red and the dilated label boundaries in green |

`create_segmentation_preview.py` is not wired into any rule. It cannot run in the locked
environment: `imwrite(…, compress=6)` raises `TypeError` with tifffile 2026.1.28. It
calls `find_boundaries(…, mode='otter')`, which is not one of scikit-image's documented
modes.

## The culture path: `create_3d_segmentation.m`

`example/sequential_workflow/create_3d_segmentation.m` extends 2D labels through z for
the cell-culture example. It works on the stitched sample in `images/fused/`, not per
FOV, and is run by hand:

1. Read `Cell_label.tif` (2D cell labels from an earlier step, such as
   `create_reference_segmentation.cpproj`) and the 3D `overlay.tif` (lines 16-18).
2. Median filter each plane with a 10×10 window (22-25); one Otsu threshold
   (`graythresh`) over the filtered stack (27-28).
3. Per plane: fill holes, remove objects under 200 pixels, dilate with a disk of radius
   10, fill holes again, and multiply the mask by the 2D labels (31-39). Written as
   `Cell.tif`, uint16 (42-46).
4. The same for nuclei with `DAPI_label.tif` and `DAPI.tif`: 10-pixel minimum, a disk of
   radius 5, no second hole filling (50-76); written as `Nuclei.tif`.
5. `Cyto = Cell − Nuclei` on the label values (88-94). With uint16 saturation this is 0
   where a nucleus carries its cell's label, but where the two labels differ it is their
   difference or 0, not a cytoplasm mask.

Every size is in pixels; no spacing is read. W-305 holds earlier results of this path
(`reference_Cell_3d_42x512x512.tif` and the 2D labels) as references, not annotations.

## Discrepancies with the agreed §2.9 scope

| # | Discrepancy | Proposed resolution |
| --- | --- | --- |
| 1 | The dimensionality is chosen implicitly from the array rank (`stardist_segmentation.py:24`); a 1×Y×X plane goes to the 3D model and nothing checks the model. | The method input is always ZYXC with Z=1 for a plane. Each model declares its dimensionality (StarDist's `config.json` `n_dim`, Cellpose's mode), and the stage wrapper checks it against the input before the model loads: Z=1 runs 2D models and is rejected by 3D models; Z>1 runs 3D models and is rejected by 2D models unless the method declares a per-plane mode ({doc}`segmentation-contract`, "Method registry"). |
| 2 | `areas.max()` fails on an image with no foreground (`:23`), and the 100-voxel gate decides whether the model runs at all. | No foreground gate. An image with no foreground gives an all-zero label image with outcome `empty`, which all three libraries already return (W-306 notes section 6). |
| 3 | Labels are cast to `uint16` (`:63`) and wrap above 65,535. | The label dtype rule of the contract (recommended: `uint32` always), with an explicit, checked conversion of the library's int32, uint16 or uint32 result ({doc}`segmentation-contract`, "Label image"). |
| 4 | Parameters are read from the Snakemake configuration inside the script, with no types or defaults. | Frozen config dataclasses registered in `SEGMENTATION_METHODS`; the workflow adapter translates the legacy `rules.stardist_segmentation.parameters` keys ({doc}`segmentation-contract`, "Workflow configuration"). |
| 5 | The fixed 0.5 rescale (`:32`, `:52`) computes outlines on the shrunk grid and changes the grid for odd sizes. | An explicit, recorded input scale per model call, applied inside the library (StarDist `scale=`, Cellpose `diameter`), which returns labels on the input grid (W-306 section 6); no default is carried between models. The script's round trip is pinned only by the golden test and the W-306 parity outputs. |
| 6 | Label expansion is applied per slice (`:40-42`), and again in `reads_assignment.py:55-57` and `:66-67` with its own `dilation_distance`. | One label operation, `expand_labels`, with an explicit mode (`planar`, the legacy per-plane behavior, or `volumetric` in physical units), applied once in the segmentation entry and recorded in the label provenance; the assignment entry reads the record and never expands again (W-308). |
| 7 | The composite assumes 3D inputs (`create_nuclei_amplicon_overlay.py:33`) and checks no grid. | The composite function takes ZYX images (Z=1 for a plane) and requires both on one grid with equal `ImageMetadata` ({doc}`segmentation-contract`, "Input preparation and label functions"). |
| 8 | The DAPI file is found by an unchecked glob (`registration-py.smk:88-90`). | The nuclear stain is a declared channel of a loaded round (`Dataset.channel_labels`); the workflow adapter requires exactly one matching file and raises naming the matches. |
| 9 | No record of the model's identity. | Pretrained models from a known-models table with file SHA-256 values, user-trained models by explicit path with each file's SHA-256 recorded; both in the provenance entry's `artifacts` ({doc}`segmentation-contract`, "Models"). |
| 10 | The thresholds come only from YAML; the model's stored `thresholds.json` is ignored. | The stored thresholds are the default; any other value is recorded as an override (the W-305 rule). |
| 11 | Normalization (1–99.8 percent over the whole image) and tiling (`n_tiles` 1×4×4 or 2×2) are fixed and not recorded. | Explicit StarDist parameters with these values as defaults, recorded in the provenance entry. |
| 12 | `rotate_nuclei` reads the raw `dapi_round` image and rotates with `reshape=True` and cubic interpolation, so for angles other than multiples of 90°, and when `dapi_round` is not the reference round, the DAPI image is not on the reference grid. | The nuclear stain comes from the reference round or from a round registered by `FOV.register_rounds`, both on the reference grid after `FOV.run`'s rotation. |
| 13 | The registered morphology images of `nuclei_registration` are not declared rule outputs. | A §2.13 rule change ({doc}`segmentation-contract`, "Workflow configuration"). |
| 14 | The Flamingo enhancement assumes 3D inputs (plane loop) and its inputs are not connected to a producing rule. | The function takes ZYX (Z=1 for a plane); the rule wiring is §2.13. |
| 15 | The culture extension through z exists only as a MATLAB example on the stitched sample, in pixels, and its `Cyto` subtraction operates on label values. | A Python label function `extend_labels_through_z` per FOV with physical parameters and geometry `extended`; compartments belong to W-308. |
| 16 | `create_segmentation_preview.py` has no rule and cannot run (`compress=`), and its boundary mode is misspelled. | Replaced by an on-demand label overlay in the segmentation diagnostics; the script's removal is §2.13. |
| 17 | The label TIFF carries no axes, geometry, target or provenance. | The saved format of the contract: ZYX with `ImageMetadata` in the description and a JSON record beside it. |
| 18 | The rule runs in a separate conda environment that lacks Starfinder. | The project environment with the `stardist` and `cellpose` extras ({doc}`segmentation-contract`, "Device and environments"). |
| 19 | No Cellpose backend, no seeded watershed and no external-mask import. | Registered `cellpose` and `seeded_watershed` methods and an `import_labels` function. |
| 20 | The model's `details` (probabilities, polygons) are discarded. | Recorded counts and parameters only; the per-object probabilities are an open choice of the worker notes. |

## Golden test

`src/python/test/test_segmentation_golden.py` (markers `workflow`, `golden`) pins the
current deterministic behavior on its own seeded fixture (seed 20261005): 16×64×64
uint8 images of DAPI (five ellipsoidal nuclei, two of which lie 3 voxels apart, and two
blobs under 100 voxels), the merged amplicon signal (80 bright voxels, four saturated)
and a Flamingo stain (larger ellipsoids around the nuclei), plus a DAPI image holding
only the small blobs. It pins with exact SHA-256 digests:

* (a) the composite, run unchanged through a stub `snakemake` object, with and without
  `maximum_projection`, and the Flamingo enhancement, run the same way; both also on
  constant and all-zero inputs, where the constant passes through the quantile stretch
  unchanged (added in the W-307 repair);
* (b) the foreground gate decision (Otsu threshold, component count, largest area,
  decision and the digest of the area list) on the 3D DAPI image, its Z maximum and the
  small-blob image, which closes the gate and gives an all-zero `uint16` image;
* (c) the nearest-neighbour rescale of labels back to the input grid after the 0.5
  shrink: the shrunk image, the labels on the shrunk grid, the restored int32 labels
  and their equality with a 2×2 block repetition;
* (d) the per-slice expansion (distance 4, the W-306 parity value) and the `uint16`
  cast, for every combination of `rescale` and `expand_labels`, in 3D and in 2D.

The StarDist script cannot be imported in the locked environment (StarDist is absent
and `tifffile.imsave` was removed), so (b) to (d) run one helper,
`legacy_stardist_steps`, whose lines are cited against the script; a test checks that
the 16 cited lines are still in the script. The model call is replaced by a stand-in
(Otsu threshold and connected components, int32). A digest-change test shows that
changing the expansion distance, the gate's minimum area, the rescale factor or the
composite projection changes a digest. Four tests document legacy behavior that §2.9
changes: the gate raises on an all-zero image, the composite raises on 2D inputs, the
round trip turns 61×63 into 60×64, and the cast maps 65,536 to 0.

Three separate single-thread processes (`taskset -c 0`, every thread variable 1)
recomputed every pinned value with byte-identical output
(`scripts/w307_compute_pins.py` and `scripts/pins-run{1,2,3}.json` in the W-307 run
directory; 1.34–1.37 s and 122–125 MB peak RSS each), so the test uses exact equality.
After the repair added the constant and all-zero cases, three more such runs
(`scripts/pins-repair-run{1,2,3}.json`) were again byte-identical and left every earlier
value unchanged. The 28 test cases take about 0.3 s together.

**What the default tier cannot run, and how W-306 pins it.** Model inference
(normalization with csbdeep, `predict_instances` with a real model, tiling) needs
StarDist, TensorFlow and model files, which the default tier does not have. W-306 pins
it instead: `parity/parity_expected.npz` in its run directory (357 KB, sha256
`4a5df997…`) holds the three parity inputs (P1 a 16×128×128 LN DAPI window with
`3D_spleen`, P2 a 512×512 tissue PI window and P3 a 61×67 hand-built image with
`2D_versatile_fluo`) and the script's label image for every combination of `rescale`
and `expand_labels`, written by the unchanged script in the legacy environment on CPU,
with the SHA-256 of each array in `parity/legacy-runs.json`. A prototype that follows
the script with the same library calls reproduced all 12 exactly in the spike
environment (StarDist 0.9.2, TensorFlow 2.20.0) on CPU. The contract names the test
that keeps this pin once the StarDist method exists ({doc}`segmentation-contract`,
"Tests the implementation changes").
