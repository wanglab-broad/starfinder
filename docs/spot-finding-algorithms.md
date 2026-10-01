# Spot-finding algorithm specification

Status: Accepted (W-268, 2026-10-01, at b5fcb7f)

This page specifies the four pipeline spot-finding methods of §2.7: Starfinder
local maxima with the W-218 option, the native Starfish LoG, Spotiflow and Piscis.
It also gives the engineering validation design of task group 6. The registry
entries, configs, weights handling, rounds and records they plug into are in
{doc}`spot-finding-contract`; the current behavior is in
{doc}`spot-finding-baseline`. Nothing here is implemented yet, and nothing here
selects a default method, model or operating point: the parameter values below are
each method's native defaults or the values a check uses.

## Evidence

Versions, weights, behavior and costs come from W-266, run directory
`/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/runs/W-266/20260930T225252Z-967e52bd`
(handoff `worker-notes.md`). It is dependency selection, not a comparison, and it
ranks no detector. The tables cited below are:

* `detectors.csv`: one row per call (283 rows), with outcomes, detections,
  isolated-scene recall, precision and biases, wall and CPU time and peak RSS;
* `localization-per-axis.csv`: per-axis absolute localization errors on the
  isolated-spot scenes;
* `known-weights.csv`: the six models, hashes, sizes, minimum shapes, training
  pixel sizes and loading routes;
* `piscis-seams.csv`: the Piscis seam probe;
* `parity/`: the Starfish `BlobDetector` parity fixtures, prototype and tables,
  copied unchanged into `src/python/test/data/starfish_blob_parity/`;
* `scripts/w266_common.py`: the isolated-spot and seam scene generators
  (`isolated_positions`, `make_case`, `seam_layout`) and the per-axis localization
  (`localization`).

All W-266 runs were on CPU (`CUDA_VISIBLE_DEVICES=""`), mostly at one thread, on
synthetic scenes only.

## What each method addresses

The §2.12 image-formation model ({doc}`synthetic-specification`) renders each
amplicon as an anisotropic Gaussian with lognormal brightness (median 1500) and
widths (axial σ 1.5, lateral σ 1.3 voxels), a small elongation and a random
orientation. It adds a camera baseline, a smooth gradient, broad regional and
tissue background, a per-round signal trend, 5 % crosstalk from each channel into
the next, and Poisson and read noise, and writes uint16.

| Method | Problem addressed | Cause in the image-formation model |
| --- | --- | --- |
| Local maxima | Fast integer-voxel candidates for bright, separated amplicons, one channel at a time, with thresholds tied to the channel's noise floor or maximum. | The amplicon PSF peak is a regional maximum above Poisson and read noise and the baseline. Crosstalk makes a weaker copy of each amplicon in the next channel, which local maxima detects there correctly; quantization and saturation flatten tops into plateaus of tied maxima (W-218 re-measurement). |
| Starfish LoG | Blob-shaped spots on a slowly varying background, across a range of spot sizes, as in starfish pipelines (parity with starfish results). | The scale-normalized Laplacian of Gaussian responds to a Gaussian of matching σ and cancels constant and linear background (baseline, gradient, broad regions); the range of σ covers the lognormal widths. |
| Spotiflow | Sub-pixel localization and detection at low signal-to-noise and for close spots, from a learned heatmap and stereographic flow. | Poisson and read noise and PSF blur hide the sub-pixel truth position; a model trained on spots predicts the position directly. |
| Piscis | Detection that holds across varying intensities and backgrounds, from a learned label map and sub-pixel displacement field, in 2D planes linked through Z. | Background heterogeneity (tissue, regions) and per-spot brightness variation defeat a single global threshold; the network sees each plane standardized (`standardize`). |

## Local maxima

| Field | Specification |
| --- | --- |
| Library and version | scikit-image 0.26.0 `peak_local_max` (locked), as at `42f652d` ({doc}`spot-finding-baseline`). No weights. |
| Parameters (units, defaults) | Unchanged: `threshold_mode` (`noise`), `threshold_value` (5.0; robust σ for `noise`, otherwise a fraction in [0, 1]), `min_distance_voxels` (1), `exclude_border` (True), `measure_peak_intensity` (True). New: `merge_radius_zyx` (None). |
| W-218 option `merge_radius_zyx` | Opt-in within-channel merge of maxima of one channel. After border exclusion, a channel's maxima are sorted by decreasing pixel value, then by increasing z, y, x. A maximum is kept unless an earlier kept maximum of the same channel lies within the ellipsoid Σ (Δᵢ / rᵢ)² ≤ 1 with radii `merge_radius_zyx` in voxels (Δz is 0 for Z=1). Kept maxima keep their coordinates and `peak_intensity`; nothing is averaged. Identities are assigned after the merge. `diagnostics["merged"]` records the number removed per channel. With `None` the result is exactly today's. |
| What the option resolves | Tied maxima of a plateau and split maxima of one amplicon in one channel. The W-218 re-measurement found such ties after legacy uint8 normalization: 2 exact ties at neighbouring voxels, which a radius of (2, 2, 2) removed in a post-hoc check without removing any matched candidate. A radius of at least √3 ≈ 1.73 voxels covers every 26-connected tie. |
| What it does not resolve | The 22 W-218 duplicates on the `small` scene are crosstalk copies in the next channel. Removing them needs a decision across channels: deduplication by position and channel ratio, merging decoded reads with the same color sequence within a radius, or crosstalk compensation before detection. This issue excludes cross-channel deduplication and assigns it to §2.8. The open choice for Jiahao is whether W-218 closes with this within-channel option plus `exclude_border` in YAML, or stays open for §2.8. |
| Border misses | The 5 W-218 misses are border exclusion; `exclude_border` becomes settable in YAML. Its default stays True. |
| Default | `merge_radius_zyx=None` (legacy) until W-268 decides; the golden test pins the legacy behavior. Whether the option becomes the default, and with which radius, is for Jiahao. The radius should then follow the spot size, about 2 σ: (3.0, 2.6, 2.6) voxels for the §2.12 widths. |
| Z handling | Z=1 is a YX plane (z=0). For 1 < Z ≤ 2 × `min_distance_voxels` with border exclusion the result is empty, as today, and a `SpotFindingWarning` says so. |
| Tiling | None. |
| Thresholds and density | The noise threshold uses the whole channel, so at high spot density it rises above the peaks. W-266 found an empty result on the 1×64×64 isolated-spot plane (0.024 spots per voxel); recall was 1.0 at the 3D density (7.6e-4). This is flagged, not changed (W-266 choice 8); the `noise` diagnostics make it visible. |
| Failure | Invalid configs raise at construction; `global` mode on non-integer data raises `ValueError`. There is no iteration and nothing to converge. |
| Resources (W-266, one thread) | Inference 0.6 s and 0.37 GB peak RSS on `medium` (32×512×512), 0.01 s and 0.16 GB on 1×512×512, 6.1 s and 0.95 GB on `large` (30×1024×1024). The merge adds a KD-tree query per channel. |

## Starfish LoG

| Field | Specification |
| --- | --- |
| Semantics (D1) | Native reimplementation of starfish `BlobDetector` at `1fb00cbc`. W-266 found `starfish/core/spots/FindSpots/blob.py` identical at 0.4.0 (`d9a305f`) and `1fb00cbc` (git blob `1e65b215`), with the whole `FindSpots/` and `types/` trees unchanged. starfish is not a dependency (its `docutils<0.20` pin conflicts with the lock). |
| Mode reproduced | `is_volume=True`, `detector_method="blob_log"`, no reference image, one round and channel at a time, `measurement_type="max"`. |
| Modes excluded | The reference-image mode, which detects on a reference image and measures spots on every round and channel; the per-slice 2D mode (`is_volume=False`); and the `blob_dog` and `blob_doh` detector methods. |
| Library and version | scikit-image 0.26.0 `skimage.feature.blob_log` (locked). No weights, no extra. |
| Algorithm | `blob_log(image, min_sigma, max_sigma, num_sigma, threshold, overlap, exclude_border)` on the scaled image, then starfish's four post-processing steps: (1) a single Z plane is squeezed to 2D and gets z=0; (2) coordinates are truncated to integers (`astype(int)`), not rounded; (3) `radius = round(σ × √ndim)`, using the mean of the per-axis σ when they differ; (4) the intensity is read at the truncated position. Starfinder's table holds `z, y, x` (the truncated integers as float64), `channel`, `peak_intensity` (the original pixel value at that position) and `radius`, in `blob_log`'s row order per channel. A parity view maps them back to starfish's columns (`intensity` = the scaled pixel value) for the parity test. |
| Intensity scaling | Integer images are divided by their dtype maximum into float32 [0, 1] (`img_as_float32`, as a starfish `ImageStack` holds them); float images must already lie in [0, 1] (`ValueError` otherwise). |
| Parameters (units) | `min_sigma`, `max_sigma`: σ in voxels, a number or a ZYX 3-tuple; `num_sigma`: number of scales; `threshold`: absolute; `overlap` (0.5): blobs overlapping by more than this fraction are pruned to the larger; `exclude_border` (False). starfish has no defaults for the first four, so they are required. The starfish ISS tutorial values used by W-266 are `min_sigma=1, max_sigma=10, num_sigma=30, threshold=0.01`; they are an example, not a default. |
| Threshold units | The threshold applies to the scale-normalized LoG response of the [0, 1]-scaled image. For a Gaussian spot of amplitude A at matched σ this response is about 0.5 A in 2D and 0.53 A in 3D. A brightness-1500 spot in uint16 (A = 0.0229) responds at about 0.011 to 0.012, just above 0.01, so on the 1×64×64 isolated-spot plane about half of the spots fell below 0.01 (W-266 recall 0.45 to 0.51), while the 3D scene kept all of them. The value is not transferable to other methods or other scalings. |
| σ and spacing | σ is in voxels. W-266 suggested physical units converted with the spacing, and a floor of about one voxel per axis: a 0.5-voxel lateral σ gave hundreds to thousands of noise detections. This contract keeps voxels, matching starfish and D5's rule of no automatic scaling from voxel size; physical units are an open choice. |
| Z handling | Z=1: the squeezed plane, z=0; a 3-tuple σ with Z=1 raises `IncompatibleGeometryError`. Z>1: 3D. |
| Tiling | None. The whole channel is processed, so results equal starfish's. |
| Scale-space memory | W-266 measured a peak minus pre-inference RSS of 10.0 to 10.4 bytes × `num_sigma` × voxels (float32 input; about 2.5 copies of the float32 scale-space cube). D1's planning note says float64; the measurement governs. `diagnostics["geometry"]` records the estimate 10.4 × `num_sigma` × voxels before the call. On `large` (31.5 M voxels) `num_sigma=30` stopped at the 4 GiB target (projected about 10 GB), and `num_sigma=10` used 3.41 GB. Time grows with `num_sigma` and `max_sigma`: 198 s at 30 and 91 s at 10 on `medium`. `max_sigma=10` far exceeds the §2.12 spot σ of 1.3 to 1.5. |
| Parity outputs | W-266 `parity/`: fixtures `volume-2ch` (1 round × 2 channels × 16×64×64, seed 266001, with a close pair 2.5 voxels apart), `plane` (1×64×64, seed 266002) and `empty` (8×32×32 zeros); cases with the ISS settings, with anisotropic σ (2, 1, 1) to (6, 3, 3) and `num_sigma=10`, on the plane and on the empty image; six tables of 9, 8, 9, 8, 11 and 0 rows. The prototype equals every table with `check_exact=True`, and the plane with a 3-tuple σ raises in both implementations. These are the expectations of check S12. |
| Failure | Invalid configs raise at construction. `blob_log` does not iterate. A channel that is constant yields no candidates (outcome `constant`). |
| Resources (W-266, one thread) | `num_sigma=30`: 197 s and 2.66 GB on `medium`, 1.0 s and 0.23 GB on 1×512×512, resource stop on `large`. `num_sigma=10`: 90 s and 1.02 GB on `medium`, 340 s and 3.41 GB on `large`. Four threads did not parallelize it (CPU time equal to wall time). On the validation fixtures (at most 32×64×64) a call needs about 40 MB. |

## Spotiflow

| Field | Specification |
| --- | --- |
| Library and version | spotiflow 0.6.5 (`b4f4645`) with torch 2.7.1+cpu and torchvision 0.22.1+cpu; extra `spotiflow` ({doc}`spot-finding-contract`, "Dependency plan"). |
| Weights | `spotiflow-models` release 0.6.0: `synth_3d` and `smfish_3d` (3D), `general` and `hybiss` (2D), with the SHA-256 values of W-266 `known-weights.csv` ({doc}`spot-finding-contract`, "Known-weights table"); loaded with `Spotiflow.from_folder(<folder>, map_location="cpu")`. Training pixel sizes (operator-retrieved): `synth_3d` 0.2 µm voxels (synthetic), `smfish_3d` 0.13 µm YX and 0.48 µm Z, `general` 0.04 to 0.34 µm (mixed), `hybiss` 0.15, 0.32 and 0.34 µm. |
| Algorithm | `model.predict(image, prob_thresh, n_tiles, min_distance, exclude_border, scale=1, subpix, peak_mode="fast", normalizer="auto", verbose=False, device="cpu")`: the image is normalized by its 1st and 99.8th percentiles, the network predicts a probability heatmap and a stereographic flow, peaks of the heatmap above `prob_thresh` are local maxima (`min_distance`), and the flow refines them to sub-pixel positions. Output: points (ZYX or YX) and `details.prob`. |
| Parameters (units, native defaults) | `prob_thresh`: probability on the sigmoid heatmap in [0, 1]; `None` uses the value stored with the weights: `synth_3d` 0.3, `smfish_3d` 0.4, `general` 0.5, `hybiss` 0.532. `min_distance` 1 (pixels or voxels); `exclude_border` False; `subpix` from the model configuration; `n_tiles` from `max_tile_size`; `scale` 1 (fixed). |
| Threshold range | On the isolated-spot scenes, heatmap maxima at truth were 0.57 to 0.995 and the background maximum 0.07 to 0.16, so the useful range is roughly 0.1 to 0.55 (3D) and 0.16 to 0.78 (2D); every stored value lies inside it. |
| Z handling | Z>1 needs a 3D model; Z=1 needs a 2D model on the squeezed plane; z=0. A 3D model on Z=1 or on Z < 7 returns nothing without error, so the wrapper raises `IncompatibleGeometryError`; the minimum shapes are 3D models Z ≥ 7 with Y, X ≥ 8, and 2D models Y, X ≥ 6 (W-266 probes). A 2D model on a volume raises a channel-count error in torch, which the wrapper preempts with `IncompatibleGeometryError`. |
| Tiling | `n_tiles` (CPU fallback tiles 2048×2048 in 2D and 128×256×256 in 3D; `medium` ran as (1, 2, 2)); tile overlap 4 blocks of 16 px (2D) or 2 blocks of 32 voxels (3D); peaks are kept only in each tile's destination slice. Forced (4, 4) on 1×512×512 and (1, 4, 4) on `medium` gave the same detections as the native run: none lost or repeated, matched shift at most 0.018 px (2D) and below 5e-4 voxels (3D). On 64-pixel images forcing tiles forms no seam, because the overlap covers the tile. |
| Scale | Fixed at 1. W-266: `scale` ≠ 1 raises for 3D models and with sub-pixel refinement; 2D without sub-pixel refinement finds 32 to 79 % of the spots at scale 2 and none at 0.5. |
| Convergence and failure | A feed-forward network: nothing iterates. Library exceptions are re-raised with their message; MKL needed no workaround (82 calls at one and four threads). |
| Repeatability | Two one-thread runs bit-identical; one against four threads: same counts, coordinates within 3.1e-6 voxels (largest nearest-neighbour distance 4.2e-6). |
| Resources (W-266, one thread) | Inference 19 s and 1.3 GB on `medium`, 1.5 s and 0.9 GB on 1×512×512 (2D models), 78 s and 1.76 GB on `large` (`smfish_3d`); imports 3.3 to 5.5 s, model load 0.19 to 0.52 s per process; four threads gave 6.2 s on `medium` (about 3.1× on a shared host). |

## Piscis

| Field | Specification |
| --- | --- |
| Library and version | piscis 1.1.0 (`0f70419`) with torch 2.7.1+cpu; extra `piscis`. The `Piscis` class (max-pooled label map) is used; `PiscisLegacy` (sum-pooled, native threshold 1.0, "the minimum number of fully confident pixels") is not. W-265's empty result came from threshold 1.0 on the `Piscis` class, whose labels never exceed 1. |
| Weights | Hugging Face `wniu/Piscis` at revision `9bdefc72cb`: `20230905` and `20251212` (one 2D network each, used in both modes), SHA-256 equal to the Hugging Face LFS values; loaded through the hash-checked absolute-path wrapper. Training pixel size: not published by the authors for either model. The library default is `20251212`; E02 proposed `20230905`; every run names one, and this page does not choose. |
| Algorithm | `Piscis(model_name=<absolute path without .pt>, input_size=...)` then `predict(image, stack, scale=1, threshold, min_distance)`: each plane is standardized (metadata `adjustment: standardize`) and tiled; the network predicts a label map and a sub-pixel displacement field; the label map is max-pooled over a 3×3 deformable window; spots are positions where labels exceed `threshold`. In plane mode they are `peak_local_max` peaks with `min_distance` pixels; in stack mode they are connected components of `labels > threshold` across planes, at their `regionprops` centroid. |
| Parameters (units, native defaults) | `threshold` 0.5 (label value in [0, 1], kept when strictly above); `min_distance` 1 pixel (plane mode); `input_size` the model's 256 (tile side in pixels, a multiple of 8); `scale` 1 (fixed). |
| Threshold range | Labels at truth were 1.0; the background more than 3 pixels away was at most 1.8e-5 (`20251212`) and up to 0.9994 (`20230905`, one out-of-focus column) in 3D, so the useful range is about 0.001 to 0.99. |
| Z handling | Z=1: plane mode on the squeezed plane, z=0. Z ≥ 2: stack mode. Stack-mode Z is an integer: the component centroid is cast to `int`, and only Y and X receive the sub-pixel displacement. This gives a Z bias of −0.32 to −0.43 and absolute Z errors up to 1.76 voxels on the isolated-spot scene (3 to 14 matches per run above 1 voxel). Spots aligned vertically 8 planes apart merge into one component (recall 0.35 to 0.38 on the aligned probe). Both are specified as Piscis's behavior (W-266 choice 4, accepted); Z linking by Starfinder is not part of §2.7. |
| Tiling | Tile side T = `round(input_size / scale)`; overlap `rint(0.1 × T)` on axes longer than T; tile k starts at k(T − O); each overlap is cut at one keep-boundary b − 0.5, with no merging across tiles (`piscis/core.py:139-188`, `deeptile`). A spot whose prediction lies within a fraction of a pixel of a keep-boundary may be lost or repeated once per adjoining tile: in the seam probe, edge spots had 0 to 2 candidates and corner spots 1 to 4. Spots 0.5 px or more from a boundary had exactly one candidate, and the lateral shift against the untiled run was at most 0.147 px. With the native 256 on a 512-pixel axis the boundaries are 242.5 and 472.5. The keep-boundaries are recorded; seam repeats are not merged (open choice). |
| Scale | Fixed at 1. W-266: `scale` 0.5 shifts Y and X by −0.40 to −0.51 px; at scale 2 `20230905` found nothing in 2D and `20251212` shifted by up to +0.19 px. |
| Convergence and failure | Feed-forward: nothing iterates. Stack mode on Z=1 would raise `AttributeError` inside piscis and plane mode on a volume would treat Z as a batch; the dispatch by Z prevents both. Library exceptions are re-raised with their message. |
| Repeatability | Two one-thread runs bit-identical; one against four threads within 1.8e-7 voxels. |
| Resources (W-266, one thread) | Inference 231 s (`20230905`) and 226 s (`20251212`) with about 1.0 GB on `medium`; 7.1 to 7.4 s and 0.85 GB on 1×512×512; 629 s and 1.19 GB on `large` (`20230905`); four threads 74 to 84 s on `medium` (2.7 to 3.1× on a shared host). |

## Engineering validation design (task group 6)

Task group 6 is engineering validation only (W-152 §2.14 decision, 2026-09-29):
known-answer synthetic fixtures with pass/fail tolerances fixed before the run, in
default-tier pytest modules for local maxima and LoG and extended-tier modules
(`-m extended`, CPU only, `CUDA_VISIBLE_DEVICES=""`, one thread) for Spotiflow and
Piscis. It has no method comparison matrix, no threshold or parameter sweep, no
benefit flag, no default selection from comparative data, and no run framework or
report engine; comparisons belong to E02. The checks follow the §2.6 V-table
({doc}`registration-algorithms`).

Rules:

* Fixtures are hand-built known answers, with their density stated, except S16,
  which uses the §2.12 generator. Every fixture is at most 32×64×64 voxels, four
  channels and four rounds, generated in session; seeds are 100, 101 and 102.
* Each tolerance is of one of two kinds only. Either it cites the W-266
  measurement it is derived from (an exact bound may cite a W-266 bit-identity or
  parity result), or it is marked **provisional** with a one-line reason. Evidence
  from this issue (the golden test, the W-218 re-measurement) is named where it
  exists, but it does not replace the provisional mark. Tolerances are fixed before
  the run; a correct implementation that cannot meet one goes to Jiahao, and the
  bound is not adjusted in the run.
* Metrics come from `evaluate_spots` and `match_points` with the policy stated.
  Unless noted: policy `greedy`, threshold 3.0 voxels, boundary `inclusive`, units
  `voxel`, truth = the rendered centres (W-266's policy).
* Methods: LM = `local_maxima`, LoG = `starfish_log` with the W-266 settings
  (`min_sigma=1, max_sigma=10, num_sigma=30, threshold=0.01`), SF3 = Spotiflow
  `synth_3d` and `smfish_3d`, SF2 = Spotiflow `general` and `hybiss`, PI = Piscis
  `20230905` and `20251212`, each at its native defaults. Each check runs every
  method it names; the results are compared with the check's tolerance only, never
  with each other.

Fixtures:

| Fixture | Construction | Density |
| --- | --- | --- |
| `iso3d` | W-266 isolated-spot scene: 32×64×64, 100 spots, one per column of a 10×10 YX grid (step 6), Z layers 8, 16 and 24 alternating, every axis jittered in [−0.5, 0.5) from the seed (minimum 3D separation 7.7 voxels); brightness 1500, σ Z 1.5 and YX 1.3, baseline 100, Poisson (α 1) plus read noise 3, uint16 (`w266_common.isolated_positions` and `make_case`) | 7.6e-4 spots per voxel |
| `iso_z1` | The same 100 YX positions on 1×64×64 | 0.024 per voxel |
| `iso_z1_sparse` | 25 spots on a 5×5 grid of step 12 on 1×64×64, otherwise as `iso_z1` | 0.0061 per voxel |
| `pairs` | 32×64×64, `iso3d` appearance: 12 lateral pairs 6 px apart in the same plane, and 8 axial pairs 8 planes apart in the same column, all pairs at least 12 voxels from each other | 3.1e-4 |
| `channels` | 16×64×64 × 4 channels: 25 spots (`iso3d` appearance) in channel 0; channels 1 and 2 equal channel 0 plus constant offsets 300 and 1500 (the same noise realization); channel 3 is an independent noise draw with the same spots | 3.8e-4 per channel |
| `coincident` | 16×64×64 × 2 channels: 20 spots at the same positions and amplitude in both channels | 3.1e-4 per channel |
| `empty` | All zeros (8×32×32×2); a constant 100 (8×32×32); a channel zero in 60 % of its voxels (MAD 0) and one zero in 40 % (MAD > 0), with 10 spots each | — |
| `borders` | 16×64×64 and 1×64×64: spots with integer centres at 0, 1, 2 and 3 voxels from each face (Y and X faces only for Z=1), brightness 1500 | 24 spots |
| `seams` | W-266 seam scenes `seam_z1` (1×64×64) and `seam3d` (32×64×64, spots at Z 16 ± 0.5), keeping only the spots 0.5, 1 and 2 px from the keep-boundaries 30.5 and 59.5 of 32-pixel tiles and the controls (`w266_common.seam_layout`) | 18 spots |
| `multiround` | 3 rounds × 16×64×64 × 2 channels on one grid: 15 spots per round and channel, 5 of them at the same positions in every round | 2.3e-4 per round and channel |
| `parity` | `src/python/test/data/starfish_blob_parity/fixtures.py` (W-266) | as W-266 |
| `formed16` | §2.12 `formed_scene_preset("small")` appearance and codebook at 16×64×64 with 24 amplicons, four rounds; once as generated and once after the legacy recipe-1 uint8 min–max normalization | 3.7e-4 |

Checks:

| # | Check | Fixture and methods | Metric (source) | Pass/fail tolerance |
| --- | --- | --- | --- | --- |
| S1 | Isolated-spot recall and precision | `iso3d`: LM, LoG, SF3, PI; `iso_z1`: LoG, SF2, PI; `iso_z1_sparse`: LM. Seeds 100–102 | `evaluate_spots`, default policy | Recall 1.0 and precision ≥ 0.98 per seed (W-266: recall 1.0 for all; precision 0.98 to 1.0 in 3D and 1.0 on Z=1). LoG on `iso_z1`: recall in [0.40, 0.60] and precision ≥ 0.98 (W-266: 0.45, 0.51, 0.48 and 1.0; threshold near the median-spot response). LM on `iso_z1_sparse`: recall 1.0, precision ≥ 0.98, **provisional** (W-266 has no row; on `iso_z1` LM is empty at that density, which is flagged, not gated). |
| S2 | Localization | The matched pairs of S1 | `localization_errors` (new): per-axis absolute error, lateral and 3D distance, maxima | SF3: 3D distance ≤ 0.5 voxels (W-266 max 0.432). SF2: lateral ≤ 0.5 px (max 0.404). PI plane: lateral ≤ 0.15 px (max 0.098). PI stack: lateral ≤ 0.15 px (max 0.119) and absolute Z ≤ 2.0 voxels, **provisional** (max 1.763; W-266 withdrew 1.0). LM and LoG 3D: distance ≤ 0.9 voxels (max 0.818). LoG Z=1: lateral ≤ 0.9 px (max 0.546). LM Z=1: lateral ≤ 0.9 px, **provisional**. |
| S3 | Resolvable pairs (no near-limit gates) | `pairs`: LM, LoG, SF3, PI (lateral pairs only for PI); `pairs` plane z of the lateral pairs as 1×64×64: SF2, PI | `evaluate_spots`; each pair member matched to its own detection | Every pair member matched (pair recall 1.0). Z=1 lateral pairs for SF2 and PI derive from W-266 `iso_z1` (all 100 spots at 6-px spacing resolved); axial pairs for SF3 from the W-266 aligned probe (`smfish_3d` found all 147 spots stacked 8 planes apart). The other 3D cases are **provisional**: W-266 has no pair rows for LM, LoG or PI in 3D, nor for lateral pairs in 3D. PI axial pairs are not gated: stack mode merges them by specification (W-266 recall 0.35 to 0.38). |
| S4 | Per-channel backgrounds and overrides | `channels`: every method; one plan with an override for channel 3 (LM `threshold_value` 6.0, LoG `threshold` 0.02, SF `prob_thresh` 0.5, PI `threshold` 0.6) | Table equality; `diagnostics["effective_settings"]`; thresholds | Channels 1 and 2 give the detections of channel 0. For LM the coordinates are identical (`peak_intensity` differs by the offset), and its noise thresholds exceed channel 0's by the offset within 1e-9. For LoG, SF and PI the count is the same and the coordinates agree within 1e-5 voxels. The effective settings equal the override config for channel 3 and the plan config elsewhere, and channel 3's table equals a single-channel run with the override config. All **provisional**: W-266 did not measure background offsets. The LM bound is the expected cancellation of an offset in the median, the MAD and the maximum filter. The 1e-5 bound for the scaled or normalized methods is set at about twice the size of W-266's one- against four-thread differences (at most 4.2e-6), which are a different perturbation and do not establish it. The override checks are new bookkeeping. |
| S5 | Coincident cross-channel candidates | `coincident`: every method | Rows per channel | Each channel's table equals its single-channel run: every coincident spot keeps one row per channel, and nothing is merged across channels. **Provisional**: W-266 measured no coincident channels; the bound is the contract rule that channels are detected independently (D6). |
| S6 | Empty input and zero channels, with the MAD diagnostics | `empty`: every method | Rows, columns and dtypes; `outcomes`; `noise`; warnings | Zero rows with exactly the declared `output_columns` and dtypes; outcome `constant` for the zero and constant channels without calling the backend; for the 60 %-zero channel (constructed with exactly 60 % zeros) `zero_fraction` 0.6 and `mad` 0 recorded and one `SpotFindingWarning`; for the 40 %-zero channel no warning. **Provisional**: these are new diagnostics and outcomes without a W-266 reference; W-266's empty parity case supports only LoG's typed empty table, and the golden test of this issue pins today's MAD-0 threshold of 0. |
| S7 | Borders | `borders`: LM with `exclude_border` True and False; LoG, SF, PI | Presence of each planted spot | LM True: exactly the spots at distance 0 are absent (`min_distance_voxels=1`); LM False: all present. LoG, SF, PI: every spot at 2 or more voxels from every face detected; spots on a face are reported, not gated. All **provisional**: W-266 placed no truth spot near a face. The LM bound follows the documented border rule, which the golden test pins on its own fixture; the bound for the other methods is an expectation. |
| S8 | Tiling seams | `seams`: PI with `input_size=32` (keep-boundaries 30.5 and 59.5) against the untiled run; `iso_z1` and `iso3d`: SF with forced `n_tiles` (2, 2) and (1, 2, 2) | Candidates within 3 voxels of each truth spot; lateral shift against the untiled run | PI: exactly one candidate per spot and lateral shift ≤ 0.2 px (W-266 `piscis-seams.csv`: all 84 off-boundary and 24 control spots single; maximum shift 0.147 px). SF: same detections as the untiled run within 0.05 px (W-266 medium and 1×512×512: at most 0.018 px). This only shows that `n_tiles` is passed and recorded, since 64-pixel images form no seam; Spotiflow's seam evidence is W-266's realistic-tier runs. LM and LoG do not tile. |
| S9 | Explicit scaling | SF and PI configs; LoG anisotropic σ | Raised error; effective settings | `scale` ≠ 1 raises `ValueError` at construction, and `scale` 1 appears in the effective settings: **provisional**, a contract rule that W-266's `scale` probes motivate (offsets and losses at 0.5 and 2) but do not measure. LoG per-axis σ is honored: exact equality in S12's anisotropic case, derived from the W-266 parity tables. |
| S10 | Multi-round identities | `multiround`: LM and LoG (default tier), SF3 and PI (extended); plan `rounds` = all three, and the default | `round` column; identities; joins | Every row has its round; `spot_id` unique, `"0"`…`"N-1"`; a `(spot_namespace, spot_id)` merge is one-to-one; each of the 5 shared positions gives one row per round (never merged); the reference-round rows without `round` equal the default run's table exactly; decoding the set raises `ValueError`. **Provisional**: a new interface (option A of the contract) without a W-266 reference. |
| S11 | Checkpoint round trip | S10's multi-round sets and S1's `iso3d` tables of LoG, SF3 and PI; CSV and Parquet | `pd.testing.assert_frame_equal(check_exact=True)`; config, plan and header keys | Reloaded table, config and plan equal the originals; a version-2 checkpoint from `42f652d` (written by the golden test's helper) loads unchanged. **Provisional**: new checkpoint content without a W-266 reference; the golden test of this issue shows today's local-maxima candidates round-trip exactly through CSV. |
| S12 | Starfish parity | `parity`: LoG | Starfish tables, column by column after the parity mapping | All six tables equal with `check_exact=True`; the plane with a 3-tuple σ raises `IncompatibleGeometryError`. Exact (W-266 parity). |
| S13 | Determinism | S1 `iso3d` and `iso_z1`, seed 100: every method, run twice in one process and once in a second process, one thread | SHA-256 of the table | Identical. SF and PI: derived from W-266 (two one-thread runs bit-identical on `medium`). LM and LoG: **provisional**, because W-266 repeated only the learned methods; the golden test of this issue gave bit-identical LM digests in three processes. |
| S14 | Dependency and weights errors | Monkeypatched imports; `resolve_weights` on an empty directory and on a tiny fixture entry inserted into `KNOWN_WEIGHTS` with one byte changed; an unknown model name; one SF and one PI detection with network access patched to raise (extended) | Raised error and its message | `SpotFindingBackendUnavailableError` naming the module and extra; `MissingWeightsError` naming the path and the fetch command; `WeightsHashMismatchError` with both hashes; `ValueError` listing the known models; the detections complete without any network call. **Provisional**: contract rules without a W-266 reference; W-266 showed only that both models load from explicit local paths (`piscis_explicit_path` probe, `Spotiflow.from_folder`). |
| S15 | Dimensionality rules | Tiny inputs (1×32×32, 2×32×32, 6×32×32, 7×8×8, 1×5×5, 1×8×8) for every pipeline method | Raised error type; z of the result | LM, LoG and PI succeed on Z=1 with z=0; SF 3D models raise `IncompatibleGeometryError` on Z=1 and Z=6 and run on 7×8×8; SF 2D models raise on Z>1 and on 1×5×5; LoG with a 3-tuple σ raises on Z=1. Derived from W-266: its `z1` development rows (LM, LoG, PI plane mode), its minimum-shape and dimensionality probes (the shapes where Spotiflow returns nothing or raises, Piscis stack mode needing Z ≥ 2) and its parity error case (a 3-tuple σ on a plane raises). The error type `IncompatibleGeometryError` is the contract's. |
| S16 | The W-218 resolution on the §2.12 formed-amplicon scene | `formed16`, seeds 100–102: LM on the reference round with `merge_radius_zyx` None and (2, 2, 2), and with `exclude_border` True and False; the uint8 variant for the merge | `evaluate_spots` with policy `greedy`, 5.0 voxels, `exclusive`, eligible truth = `center_in_bounds` (the W-218 notebook policy); `classify_detections` (new) with channels as groups | With the merge: same-channel duplicates 0, and matched count ≥ the count without the merge. With `exclude_border=False`: no eligible amplicon within 1 voxel of a face missed. Cross-channel duplicates are reported, not gated (§2.8). Derived from the W-218 re-measurement of this issue (`small`, uint8: 2 same-channel ties removed and no matched candidate lost at radius (2, 2, 2); border misses 5 → 0), and **provisional**, because `formed16` is smaller and denser than `small`. |

Metrics that must be added to `starfinder.evaluation.spot_finding`. The task-group-2
foundation issue adds them, because the task-group-3 and task-group-4 issues use them
already (S2, S16); task group 6 reuses them:

* `localization_errors(match_result, detected, truth)`: from the matched pairs, the
  maximum and 95th percentile of the absolute Z, Y and X errors, the lateral and 3D
  distances, and the number of matches with absolute Z error above 1, in the
  matching units (W-266 `localization-per-axis.csv` columns);
* `classify_detections(match_result, detected, truth, *, radius, groups=None)`:
  counts of matched, duplicate (an unmatched detection within `radius` of a matched
  truth point) and spurious detections, with duplicates split into the same group
  as the truth point's matched detection or another group when `groups` (for
  example the channel) is given (the W-218 classification by position and
  channel).

`evaluate_spots`, `match_points` and the W-266 scene generators (ported into the
test helpers) are used as they are.

Resource plan: every fixture is at most 32×64×64 voxels. Scaling W-266's
per-voxel inference cost from `medium` gives about 0.3 s per Spotiflow call and
about 4 s per Piscis stack-mode call on `iso3d`, plus one import of 3 to 6 s per
process. LM and LoG calls take well under 1 s each (LoG with `num_sigma=30` needs
about 40 MB). So the default-tier checks add seconds and the extended-tier checks a
few minutes at one thread. Each run records wall time and maximum RSS with
`/usr/bin/time -v` against the 4 GiB stop target.

### Implemented checks (W-274)

The checks are in `src/python/test/test_spot_finding_validation.py`, one test
function per check (parametrized by method, model, case and seed 100 to 102).
Local maxima and the Starfish LoG run in the default tier; every Spotiflow and
Piscis case is in the extended tier (`-m extended`, CPU, `CUDA_VISIBLE_DEVICES=""`,
one thread, weights from `STARFINDER_WEIGHTS_DIR`). The W-266 isolated-spot and
seam scenes are the ports in `spot_finding_scenes.py`, `learned_detectors.py` and
`test_spot_finding_metrics.py`; the hand-built fixtures are in
`spot_finding_fixtures.py`, and `multiround` is the W-273 fixture of
`test_spot_finding_rounds.py`. Every fixture is generated in session and is at
most 32×64×64 voxels, four channels and three rounds
(`test_fixtures_stay_within_the_resource_bounds_and_their_stated_densities`). The
tolerances are module constants equal to the table above, unchanged; matching is
`greedy`, 3.0 voxels, `inclusive`, except S16.

| # | Test | Route and fixture | Metric | Tolerance |
| --- | --- | --- | --- | --- |
| S1 | `test_s1_isolated_spot_recall_and_precision` | `find_spots` at native defaults (LoG with the W-266 settings) on `iso3d` (LM, LoG, SF3, PI), `iso_z1` (LoG, SF2, PI) and `iso_z1_sparse` (LM; 5×5 grid of step 12 from 8 to 56) | `evaluate_spots` recall and precision | recall 1.0 and precision ≥ 0.98; LoG `iso_z1` recall in [0.40, 0.60] |
| S2 | `test_s2_localization_of_the_s1_matches` | the S1 matches | `localization_errors` maxima | SF3 3D ≤ 0.5; SF2 lateral ≤ 0.5; PI plane lateral ≤ 0.15; PI stack lateral ≤ 0.15 and abs Z ≤ 2.0; LM and LoG 3D ≤ 0.9; LoG and LM Z=1 lateral ≤ 0.9 |
| S3 | `test_s3_every_member_of_a_resolvable_pair_is_matched` | `pairs` (lateral pairs in plane z=26, axial pairs at z=4 and 12; each pair jittered as a whole) for LM, LoG, SF3, PI; its lateral pairs on 1×64×64 for SF2, PI | matched truth indices of `evaluate_spots` | pair recall 1.0 (PI lateral pairs only) |
| S4 | `test_s4_offset_channels_and_a_channel_override` | `channels` with a `SpotFindingPlan` overriding `ch03`, and a single-channel run with the override | rows per channel; `thresholds`; `effective_settings` | LM coordinates identical, `peak_intensity` + offset, thresholds + offset within 1e-9; others same count and coordinates within 1e-5; `ch03` table equal to the single run |
| S5 | `test_s5_coincident_spots_keep_one_row_per_channel` | `coincident`, every method | each channel's rows against its single-channel run | equal with `check_exact=True`; every spot matched in both channels |
| S6 | `test_s6_empty_input_zero_channels_and_the_mad_diagnostics` | zeros 8×32×32×2 and a constant 8×32×32 with a spy on the registry entry; 10×32×32×2 with exactly 60 % and 40 % zeros | rows, columns, dtypes; `outcomes`; `noise`; warnings | typed empty table, outcome `constant`, backend not called; LM `zero_fraction` 0.6 and `mad` 0 with one `SpotFindingWarning`, none for 40 % |
| S7 | `test_s7_spots_near_the_faces` | `borders` (16×64×64, 24 spots; 1×64×64, 16 spots) | matched truth indices | LM `exclude_border` True: exactly the spots at distance 0 absent; False: all present; LoG, SF, PI: every spot at ≥ 2 voxels present |
| S8 | `test_s8_tiling_seams` | PI with `input_size=32` against the native run on `seam_z1` and `seam3d`; SF2 on `iso_z1` with `n_tiles` (2, 2) and SF3 on `iso3d` with (1, 2, 2) against the S1 run | candidates within 3 voxels; lateral shift; one-to-one nearest shift | PI exactly one candidate per spot, shift ≤ 0.2 px; SF same count, shift ≤ 0.05 |
| S9 | `test_s9_explicit_scaling` | SF and PI configs; the S1 runs; the anisotropic parity case | `ValueError`; `effective_settings` | `scale` ≠ 1 raises, `scale` 1 recorded; LoG per-axis σ equal to starfish exactly |
| S10 | `test_s10_multi_round_identities` | `multiround` with `rounds` = all three through `FOV.find_spots`, and the default plan | `round`, `spot_id`, joins, `decode_barcodes` | as the table above; a shared position's rows lie within 3.0 voxels |
| S11 | `test_s11_candidates_checkpoint_round_trip`, `test_s11_a_version_2_checkpoint_from_before_the_plan_keys_loads_unchanged` | the S10 sets and the S1 `iso3d` tables of LoG, SF3, PI, CSV and Parquet; the W-273 saved version-2 checkpoint | `assert_frame_equal(check_exact=True)`; config, plan, header keys | equal; the saved checkpoint's digest equals the golden pin |
| S12 | `test_s12_starfish_parity` | the six W-266 parity tables and the 3-tuple σ plane | column by column after `starfish_view` | exact; `IncompatibleGeometryError` |
| S13 | `test_s13_tables_are_identical_twice_in_one_process_and_in_a_second` | S1 `iso3d` and `iso_z1`, seed 100, every method | SHA-256 of the table | identical |
| S14 | `test_s14_dependency_and_weights_errors` | patched imports; `resolve_weights` on an empty cache and on a fixture entry with one byte changed; unknown models; one SF and one PI detection with the network patched to raise (extended) | raised error and message | as the table above |
| S15 | `test_s15_dimensionality_rules` | W-266's single-spot probe as uint16 on the six tiny shapes, every pipeline method and LoG with a 3-tuple σ | raised error type; z | as the table above |
| S16 | `test_s16_the_w218_option_and_exclude_border_on_formed16` | `formed16`, LM on the reference round: the merge on the uint8 variant; `exclude_border` on both variants | `evaluate_spots` (greedy, 5.0, exclusive, `center_in_bounds`); `classify_detections` by channel | same-channel duplicates 0 and matched ≥ legacy; no eligible amplicon within 1 voxel of a face missed |

Four groups of cases miss their bound with a correct implementation. They are
strict expected failures with the bound unchanged, reported to Jiahao in the W-274
worker notes ([Known limits](#known-limits) gives their bounds, values and evidence):

* S4, both Piscis models, every seed: Piscis standardizes each tile after padding
  the 64-pixel plane to 256 pixels, so a constant offset changes its input. The
  offset channels have the same counts but lie 1.001 to 1.009 voxels from
  channel 0 (the integer stack-mode z changes by 1; lateral up to 0.135 px).
* S5, `smfish_3d`, seed 102: one channel-0 spot has probability 0.396, below the
  stored `prob_thresh` 0.4 (recall 0.95); the table equals its single-channel run.
* S7, `synth_3d` in 3D, every seed: no candidate for the spots 1 and 2 planes from
  the low Z face, while the spots at 0 and 3 planes are found.
* S10, both Piscis models, seeds 100 and 101: stack mode merges the `multiround`
  fixture's two spots of one column (z 4 and 11) into one component, as specified,
  so a shared position in such a column has no row in that round.

Resources at one thread (`/usr/bin/time -v`): the default-tier cases take under a
minute (48 s, peak RSS 0.3 GB). The extended-tier cases take about 37 minutes with a
peak RSS of 1.7 GB: about 1064 s for Piscis `20230905`, 985 s for Piscis
`20251212` and 165 s for Spotiflow. Piscis's native 256-pixel tile costs about
0.73 s per 64×64 plane (24 s per `iso3d` call, not the 4 s of the resource plan
above). Every design case is kept (W-274 decision, option A): the project checks
run the Piscis cases of each model as a separate extended check, selected by the
model name in the test id (`-k "test_spot_finding_validation and 20230905"`, and
likewise `20251212`), and the rest of the extended tier excludes them.

## Known limits

This section collects what the W-274 validation left open: the strict expected
failures, the provisional tolerances and the limits of each method. It states
recorded values only; nothing here was measured again. The sources are:

* **W-274 notes**: `worker-notes.md` of the W-274 run directory
  `/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/runs/W-274/20261001T102622Z-fbb76669`,
  section "Strict expected failures";
* **W-274 values**: `logs/w274-extended-values.json` in the same directory, the
  values each extended case recorded, and `logs/w274-xfail.log`, the rerun of the
  fourteen cases (14 xfailed);
* **tour notebook**: the executed `example/introduction/starfinder_spot_finding_tour.ipynb`,
  section 9, which lists the expected failures from the marks of the test module
  and recomputes one case of each kind;
* **the test module**: the tolerance constants and the `strict_xfail` marks of
  `src/python/test/test_spot_finding_validation.py`;
* this page's method tables, whose values come from W-266 ([Evidence](#evidence)).

### Expected failures

Fourteen cases of `test_spot_finding_validation.py` miss their bound with a correct
implementation. Each is a strict expected failure with the bound unchanged: pytest
reports it as `xfailed`, and a case that starts to pass fails the run. They are
reported to Jiahao in the W-274 notes and are not decided here.

| Check | Method and model | Cases and seeds | Bound missed | Measured value | Cause | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| S4 | Piscis `20230905` and `20251212` | 6: both models, seeds 100, 101 and 102 | Channels 1 and 2 (offsets 300 and 1500) within `S4_COORDINATES` = 1e-5 voxels of channel 0 | Same counts as channel 0, but a largest paired shift of 1.001 to 1.009 voxels: the integer stack-mode z changes by 1, and lateral positions move by up to 0.135 px | Piscis pads each 64-pixel plane into its 256-pixel tile and standardizes the padded tile, so a constant offset changes the network input. | W-274 notes, item 1; W-274 values (`shifts`); tour notebook (`20251212`, seed 100: 1.0035 and 1.0091) |
| S5 | Spotiflow `smfish_3d` | 1: seed 102 | Recall 1.0 in each channel (the assertion `recalls == [1.0, 1.0]`; no named constant) | Channel 0 recall 0.95 (19 of 20 spots); channel 1 finds all 20, and each channel's table equals its single-channel run | The missed spot (z 11.69) has a heatmap probability of 0.396, below the `prob_thresh` 0.4 stored with the weights. | W-274 notes, item 2; W-274 values (`recalls`); tour notebook (probability 0.3963) |
| S7 | Spotiflow `synth_3d`, in 3D | 3: seeds 100, 101 and 102 | Every spot at least `S7_GATED_DISTANCE` = 2 voxels from every face matched | Present by distance 0, 1, 2 and 3 voxels: 6, 5, 5 and 6 of 6 for each seed; the missed spots are 1 and 2 planes from the low Z face, and the one at 2 planes is gated | Not established by the recorded evidence: the model returns no candidate within 16 voxels of these two spots, so it is the model's output near the low Z face, not the matching. | W-274 notes, item 3; W-274 values (`present_by_distance`); tour notebook (seed 100) |
| S10 | Piscis `20230905` and `20251212` | 4: both models, seeds 100 and 101 | Every shared position has a row in every round within `S10_RADIUS` = 3.0 voxels | A shared position has no row in one round, for example (4, 42, 28) of channel 1 in round 1 at seed 100 and (11, 21, 56) of channel 1 in round 2 at seed 101; seed 102 passes | The `multiround` fixture puts sites at z 4 and 11 in the same columns, and Piscis stack mode merges the two into one component near z 8, as specified ([Piscis](#piscis), Z handling). | W-274 notes, item 4; tour notebook (`20251212`, seed 100: two such positions in round 1) |

The counts add up to 6 + 1 + 3 + 4 = 14, the number of `strict_xfail` marks the tour
notebook reads from the module.

### Provisional tolerances

These checks have bounds marked **provisional** in the
[design table](#engineering-validation-design-task-group-6) and in the comments of the
test module: no W-266 measurement stands behind them. Each bound is the module constant
named, or the test's own assertion where the module has no constant.

| Check | Cases | Provisional bound | Why there is no W-266 reference | Evidence |
| --- | --- | --- | --- | --- |
| S1 | LM on `iso_z1_sparse` | `S1_LM_SPARSE_RECALL` = 1.0, `S1_LM_SPARSE_PRECISION` = 0.98 | W-266 has no local-maxima row at this density; on `iso_z1` local maxima is empty, which is flagged, not gated. | Design table, S1; module constants |
| S2 | PI in stack mode, absolute Z error | `S2_PI_STACK_ABS_Z` = 2.0 voxels | W-266 measured a maximum of 1.763 voxels on one scene and withdrew its own bound of 1.0. | Design table, S2; module constant |
| S2 | LM on `iso_z1_sparse`, lateral error | `S2_LM_Z1_LATERAL` = 0.9 px | W-266 has no local-maxima row on a plane. | Design table, S2; module constant |
| S3 | LM, LoG and PI on `pairs`, and the lateral pairs of SF3 in 3D | `S3_PAIR_RECALL` = 1.0 | W-266 has no pair rows for LM, LoG or PI in 3D, nor for lateral pairs in 3D (the Z=1 lateral pairs of SF2 and PI and the axial pairs of SF3 are derived). | Design table, S3; module constant |
| S4 | Every method on `channels` | `S4_LM_THRESHOLD_OFFSET` = 1e-9 (LM thresholds); `S4_COORDINATES` = 1e-5 voxels (the other methods) | W-266 did not measure background offsets; the 1e-5 bound is about twice W-266's one- against four-thread differences (at most 4.2e-6), a different perturbation. | Design table, S4; module constants |
| S5 | Every method on `coincident` | Each channel's table equals its single-channel run, and recall 1.0 in each channel | W-266 measured no coincident channels; the bound is the contract rule that channels are detected independently (D6). | Design table, S5; `test_s5_coincident_spots_keep_one_row_per_channel` |
| S6 | Every method on `empty` | Typed empty tables, outcome `constant`, and for local maxima `zero_fraction` 0.6, `mad` 0 and one warning | These diagnostics and outcomes are new; W-266's empty parity case supports only LoG's typed empty table. | Design table, S6; `test_s6_empty_input_zero_channels_and_the_mad_diagnostics` |
| S7 | LM, LoG, SF and PI on `borders` | `S7_GATED_DISTANCE` = 2 voxels; LM: exactly the spots at distance 0 absent with border exclusion | W-266 placed no truth spot near a face. | Design table, S7; module constant |
| S9 | SF and PI configs | `scale` other than 1 raises `ValueError`, and `scale` 1 is recorded | A contract rule that W-266's `scale` probes motivate but do not measure. | Design table, S9; `test_s9_explicit_scaling` |
| S10 | Every method on `multiround` | `S10_RADIUS` = 3.0 voxels, and the identity and join rules | A new interface (option A of the contract) without a W-266 reference. | Design table, S10; module constant |
| S11 | Every checkpoint round trip | Exact equality (`check_exact=True`) | New checkpoint content without a W-266 reference. | Design table, S11; `test_s11_candidates_checkpoint_round_trip` |
| S13 | LM and LoG | Identical table digests | W-266 repeated only the learned methods. | Design table, S13; `test_s13_tables_are_identical_twice_in_one_process_and_in_a_second` |
| S14 | Every case | The error types and messages | Contract rules; W-266 showed only that both learned methods load from explicit local paths. | Design table, S14; `test_s14_dependency_and_weights_errors` |
| S16 | LM on `formed16` | `S16_SAME_CHANNEL_DUPLICATES` = 0, `S16_FACE_DISTANCE` = 1 voxel | Derived from the W-218 re-measurement on `small`, not from W-266, and provisional because `formed16` is smaller and denser than `small`. | Design table, S16; module constants |

### Limits by method

* **Local maxima.** Crosstalk copies of an amplicon in the next channel are correct
  detections in that channel and are kept; removing them across channels is left to
  the readout stage (§2.8). The within-channel merge does not remove them: on
  `formed16` with the merge, 5, 8 and 9 cross-channel duplicates remained at seeds
  100, 101 and 102, reported and not gated (W-274 notes, S16), and 23 remained on the
  uint8 `small` formed scene of the tour notebook (section 4, printed in section 9). See
  [Local maxima](#local-maxima), "What it does not resolve".
* **Starfish LoG.** `blob_log` holds the whole scale space: W-266 measured 10.0 to
  10.4 bytes × `num_sigma` × voxels of peak memory. The method does not tile, so the
  cost grows with the whole channel: on `large` (31.5 M voxels) `num_sigma=30`
  stopped at the 4 GiB target (projected about 10 GB), and `num_sigma=10` used
  3.41 GB. See [Starfish LoG](#starfish-log), "Scale-space memory" and "Tiling".
* **Spotiflow.** Each model has a minimum shape: 3D models need Z ≥ 7 with Y and X ≥ 8,
  and 2D models need Y and X ≥ 6 on a single plane. Below it the library returns
  nothing without an error, so the wrapper raises `IncompatibleGeometryError`.
  These are W-266 probes ([Spotiflow](#spotiflow), "Z handling"), checked by S15,
  which passes (W-274 notes, "Other checks").
* **Piscis.**
  * *Integer Z in stack mode.* The component centroid is cast to an integer, and only
    Y and X receive the sub-pixel displacement. W-274 recorded absolute Z errors of
    1.218 to 1.763 voxels on `iso3d` under the provisional bound of 2.0 (W-274 notes,
    S1 and S2). Spots aligned in Z merge into one component: S3 axial-pair recall
    was 0 to 0.06 and is not gated (W-274 notes, "Other checks"), and S10 fails for
    this reason (above). See [Piscis](#piscis), "Z handling".
  * *Candidates on a tile boundary.* Each tile overlap is cut at one keep-boundary
    with no merging across tiles, so a spot whose prediction lies within a fraction
    of a pixel of a keep-boundary may be lost or repeated: W-266's seam probe gave
    edge spots 0 to 2 candidates and corner spots 1 to 4. S8 gates only spots 0.5 px
    or more from a boundary. See [Piscis](#piscis), "Tiling".
  * *The native 256-pixel tile on small images.* A 64×64 plane is padded into one
    256-pixel tile, which costs about 0.73 s per plane, 24 s per 32-plane `iso3d`
    call against the 4 s of the resource plan above; Piscis took 2048 s of the
    2217 s extended tier (W-274 notes, "Extended-tier time"). The padding is also
    the cause of the S4 failures (above).
  * *Training pixel sizes.* The authors have not published the training pixel size
    of either model (W-266 `known-weights.csv`; [Piscis](#piscis), "Weights"), so
    the pixel size an image should have for these models is unknown.

### What this does not say

This section ranks no method or model and recommends no default method, model or
operating point. Each value is compared only with its own check's bound, never with
another method's value. The expected failures and provisional bounds are open items
for Jiahao, not accepted limits, and method comparisons belong to E02.
