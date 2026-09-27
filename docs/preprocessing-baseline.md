# Preprocessing baseline

This page records how the existing preprocessing operations behave at revision
`21eda7f` (`dev`, 2026-09-27), before the Chapter II §2.5 step/recipe work
changes them. It is the reference that the golden test
`src/python/test/test_preprocessing_golden.py` pins. The proposed replacement is
described in {doc}`preprocessing-contract` and {doc}`preprocessing-algorithms`;
neither is accepted yet.

## Operations

All five operations take finite, nonempty ZYX or ZYXC arrays and never mutate
their input. The public functions live in `starfinder.preprocessing`
({doc}`api/preprocessing`); `FOV` methods apply them to selected rounds.

| Operation | Input → output dtype and range | Rounding and clipping | Axes and scope | Defaults |
| --- | --- | --- | --- | --- |
| Min–max normalization `normalize_intensity` | Any real dtype → the configured `output_dtype`; linear map of the data range onto `output_range` | Float64 work; integer output uses `rounding` (`truncate` by default, or `nearest_even`); result clipped to `output_range`. A constant group maps to the lower endpoint. With `snr_threshold`, a group whose max/mean is below the threshold is clipped to `output_range` instead of rescaled | Per channel (`scope="per_channel"`) over the whole ZYX volume of one round, or global | No config default for dtype or range; `FOV.normalize_intensity` defaults to `MinMaxNormalizationConfig("uint8", (0, 255))` |
| Reference histogram matching `match_histogram` | Input dtype retained by default (`output_dtype=None`) | Exact CDF mapping by scikit-image `match_histograms`; integer output uses `rounding` (`truncate` by default); out-of-range values raise | Each channel's whole ZYX volume is matched to one ZYX reference volume; no bins | `HistogramMatchingConfig()`; in `FOV.run` the reference is channel `histogram_reference_channel` (0) of the reference round, captured after that round's normalization |
| Morphological reconstruction `reconstruct_background` | Source dtype retained (integers up to 32 bits, or float) | Float64 per slice: subtract the dilation reconstruction of an eroded marker, then add the white top-hat and subtract the black top-hat; saturate to the source dtype's representable range; the integer cast truncates | Each XY slice of each channel independently; disk footprint; reflect boundaries; no Z coupling | `ReconstructionConfig(radius_yx=3)`, radius in XY pixels |
| White top-hat `filter_tophat` | Source dtype retained | Float64 per slice; saturate to the source dtype; integer cast truncates | Each XY slice of each channel; disk footprint; no Z coupling | `TophatConfig(radius_yx=3)`, radius in XY pixels |
| Z projection `project_image` | `max` keeps dtype; `sum` accumulates into uint64/int64/float64; optional explicit conversion | No implicit rescaling; 64-bit integer sums are rejected | Reduces Z to a singleton, keeping (1, Y, X[, C]) | `ProjectionConfig(method="max")` |

### Configuration keys and MATLAB counterparts

| Operation | Workflow key → Python translation (`dataset/workflow.py:127-161`) | MATLAB counterpart |
| --- | --- | --- |
| Min–max | `enhance_contrast: {run, snr_threshold}` → `MinMaxNormalizationConfig("uint8", (0, 255), snr_threshold=...)` | `STARMapDataset.EnhanceContrast("min-max")` → `MinMaxNorm`: `stretchlim(..., 0)` then `imadjustn`, assigned back into the existing array class |
| Histogram matching | `hist_equalize: {run, reference_channel}` → `HistogramMatchingConfig()` and `histogram_reference_channel` | `STARMapDataset.HistEqualize`: `imhistmatchn` with 64 bins against `reference_layer="round1"`, one-based `reference_channel=1` |
| Reconstruction | `morph_recon: {run, radius}` → `ReconstructionConfig(radius_yx=radius)`; `reconstruction_after_registration` is set for resident subtile rules | `STARMapDataset.MorphRecon` → `MorphologicalReconstruction`: the same slice-wise operations, then `uint8(...)` before assignment |
| White top-hat | `tophat: {run, radius}` → `TophatConfig(radius_yx=radius)` | `STARMapDataset.Tophat`: `imtophat` per plane, then `uint8(...)` |
| Projection | Top-level `maximum_projection` → `ProjectionConfig()` for the saved reference image only | `STARMapDataset.MakeProjection` → `MakeProjections` (max by default) |

Python and MATLAB are not claimed to be numerically equivalent. The MATLAB
branches were not executed for this page.

## Execution order in `FOV.run`

`PipelineConfig` has one fixed slot per operation, and `FOV.run`
(`dataset/fov.py:474-600`) applies them in a hard-coded order:

1. The reference round is processed first, then moving rounds in declared order.
   In streaming mode each round is loaded when reached.
2. Per round: rotation → min–max → histogram matching → reconstruction (unless
   `reconstruction_after_registration`) → white top-hat → projection.
3. The histogram reference is the reference round's channel
   `histogram_reference_channel`, copied after that round's own normalization
   and before its later operations (`dataset/fov.py:535-538`).
4. For a moving round, each `RegistrationStep` estimates a transform from the
   processed images (`_registration_image`, `dataset/fov.py:335-342`) and
   immediately resamples and **overwrites** that round's single image
   (`dataset/fov.py:354-378`). With several steps the image is resampled once
   per step.
5. For resident subtile jobs (`lrsf_single_fov_subtile`, `deep_rsf_subtile`),
   reconstruction runs after registration. A copy of the reference taken before
   its reconstruction is used as the registration reference for moving rounds
   (`dataset/fov.py:546-558`).
6. Detection runs on the reference round's processed image; extraction reads each
   round's processed image (`dataset/fov.py:417-425`).

## Intensity statistics after each operation

The noise-mode local-maxima threshold is
`median + threshold_value × 1.4826 × MAD` over all voxels of a channel, with
default `threshold_value=5.0`. The table reports it for the golden fixture:
round 2 of two seeded uint16 rounds, 4×32×32 voxels and four channels, with
per-channel offsets of 200–1600, Gaussian noise of standard deviation 20, twelve
puncta of amplitude 1000–4000 and unequal channel gains. Values are
(zero fraction, median, MAD, threshold) for channels 0–3, and the test pins them
exactly.

| Output | Channel 0 | Channel 1 | Channel 2 | Channel 3 |
| --- | --- | --- | --- | --- |
| Raw uint16 input | 0, 174, 18, 307.4 | 0, 584, 36, 850.9 | 0, 850, 29, 1065.0 | 0, 2638, 44, 2964.2 |
| Min–max to uint8 | 0.001, 13, 3, 35.2 | 0.000, 8, 2, 22.8 | 0.004, 6, 2, 20.8 | 0.000, 11, 3, 33.2 |
| Histogram matching after min–max | 0.006, 6, 2, 20.8 | 0.006, 6, 2, 20.8 | 0.004, 6, 2, 20.8 | 0.003, 6, 2, 20.8 |
| Reconstruction after histogram matching (`FOV.run` output) | **0.790, 0, 0, 0** | **0.758, 0, 0, 0** | **0.754, 0, 0, 0** | **0.794, 0, 0, 0** |
| White top-hat after min–max | 0.166, 3, 2, 17.8 | 0.208, 2, 1, 9.4 | 0.215, 2, 1, 9.4 | 0.167, 3, 2, 17.8 |
| Reconstruction on raw uint16 | **0.699, 0, 0, 0** | **0.728, 0, 0, 0** | **0.730, 0, 0, 0** | **0.724, 0, 0, 0** |
| White top-hat on raw uint16 | 0.126, 19, 10, 93.1 | 0.131, 33, 20, 181.3 | 0.140, 26, 16, 144.6 | 0.123, 41, 24, 218.9 |

On this fixture, **reconstruction reaches MAD = 0**. More than half of the voxels
are zero, so the median, MAD and noise threshold are all zero, and every positive
local maximum would pass detection. Min–max to uint8 also leaves the background
in only a few grey levels (MAD of 2–3), because the brightest puncta set the
range. These are properties of one synthetic fixture, not measurements of
laboratory data.

## Discrepancies with the agreed §2.5 scope

| Observed behavior | Agreed §2.5 scope | Proposed resolution |
| --- | --- | --- |
| Min–max output is hard-coded to `uint8`, `(0, 255)` in `FOV.normalize_intensity` (`dataset/fov.py:291`) and the workflow adapter (`dataset/workflow.py:157`) | Preserve the input dtype by default | Recipe steps default to the input dtype and its full range. The legacy YAML mapping keeps `uint8`, `(0, 255)` so recipe 1 reproduces the golden output |
| `rounding="truncate"` is the default in `MinMaxNormalizationConfig` and `HistogramMatchingConfig` | One shared numerical policy | New steps use round-half-to-even. Legacy steps keep `truncate` through the legacy mapping; changing it is a separate, reviewed behavior change |
| `workflow.py` accepts `tophat`, which `workflow/schemas/config.schema.yaml` does not declare | Configuration keys are validated consistently | Declare `tophat_params` (`run`, `radius`) in the schema for the Python rules, without changing MATLAB-shared keys |
| Morphology radii are XY pixels and filtering is slice-wise | A 3D option with scales in physical units | Keep the XY operations as baselines; add the 3D method with radii in physical units ({doc}`preprocessing-algorithms`) |
| Each registration step resamples and overwrites the single processed image, and extraction reads that image | Every declared measurement snapshot receives the same transform, resampled once from its pre-registration version | Carry named snapshots through registration ({doc}`preprocessing-contract`); §2.6 composes stages |
| Operation order is fixed by `PipelineConfig` slots | Independently selectable operations with a recorded order | Replace the slots with an ordered recipe of steps; map legacy keys to recipe 1 |
| Histogram reference and min–max ranges are always fitted on the current FOV | `fov` and `supplied` fitting modes | Add the modes to the new steps; min–max stays per FOV |
| Reconstruction output can make the noise threshold zero without notice | Record-and-warn diagnostic in §2.7 | No change here; the §2.7 detector records the zero fraction, MAD and threshold and warns |
