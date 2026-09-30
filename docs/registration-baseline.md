# Registration baseline

Status: Accepted (W-246, 2026-09-30, at 47057aa)

This page records how registration behaves at revision `2e48d3f` (branch
`runner/s26-spec-20260929`, on `dev` at `1b4db5a`; the registration source is
unchanged since `1b4db5a`), before the Chapter II §2.6 work changes it. It is
the reference that the golden test
`src/python/test/test_registration_golden.py` pins. The proposed replacement is
described in {doc}`registration-contract` and {doc}`registration-algorithms`;
neither is accepted yet. Paths are relative to `src/python/starfinder/` unless
they start with `src/matlab/`, `workflow/` or `docs/`, and line numbers are at
`2e48d3f`.

Since W-262, `TranslationTransform` stores `displacement_zyx = −correction_zyx`; see "Translation displacement" in {doc}`migration`.

## Methods

Registration estimates a transform from one reference signal and one moving
signal per moving round, then resamples that round. The public functions are
`estimate_transform` and `apply_transform` in `starfinder.registration`
({doc}`api/registration`). `FOV.register` runs one `RegistrationStep` for the
moving rounds, and `FOV.run` runs `PipelineConfig.registration`, an ordered
tuple of steps.

### Common inputs and signal construction

| Field | Behavior at `2e48d3f` |
| --- | --- |
| Inputs | `estimate_transform(reference, moving, *, config, reference_metadata, moving_metadata)` takes two finite 3D ZYX arrays of equal shape (`registration/_api.py:24-54`). The two `ImageMetadata` values must agree in spacing, origin, direction and unit; frame identifiers may differ. There is no unequal-grid conversion. |
| Signal | `FOV._registration_image` (`dataset/fov.py:388-397`) turns a ZYXC round into one ZYX signal. `merged` sums all channels in float64 (`image.sum(axis=-1, dtype=np.float64)`). `single-channel` takes channel `reference_channel`, a zero-based index used for both rounds, and keeps the input dtype. |
| Source of the signal | During `FOV.run` with a preprocessing recipe, the signal of both rounds is built from the recipe's `registration_source` snapshot; with `None` it is the detection image (`dataset/fov.py:622`, `:674`). A step estimates from the current image, so the second step of a recipe sees the moving round already resampled by the first. |
| Reference | The reference round is `RoundState.reference_round`. It is never transformed. Its signal is rebuilt for every step and every moving round. With `post_registration` reconstruction, a copy of the reference taken before its reconstruction is used (`dataset/fov.py:669-681`). In streaming mode only the reference's `registration_source` snapshot is kept across rounds (`dataset/fov.py:692-702`). |
| Z | Z is kept in the signal; nothing projects. Translation accepts singleton axes. Local estimators reject small axes before running: demons needs every axis ≥ 4, TPS and CPD every axis ≥ 2 (`registration/_api.py:50-53`), so a Z=1 round always fails local registration with `IncompatibleGeometryError("local estimator requires 3D; demons axes must be >=4")`. |

### Translation

| Field | Behavior |
| --- | --- |
| Inputs, signal and reference | The common rules above. The FFT casts both signals to float32. |
| Estimator | `TranslationConfig(backend="scipy_fft", fft_workers=1)`. `scipy_fft` is FFT cross-correlation in float32 (`registration/_translation.py:11-65`); `skimage` calls `phase_cross_correlation`. The peak is integer: no subpixel refinement. Singleton axes return 0. An even-length half-period peak is reported as +n/2. |
| Z rules | Any shape, including Z=1 (the Z component is then 0). |
| Transform | `TranslationTransform(correction_zyx)`, voxel index units, direction `moving_to_reference`. Application pulls from `reference index - correction`, so a moving round whose content is displaced by `d` gets `correction = -d`. |
| Defaults | `TranslationConfig()`: `scipy_fft`, one FFT worker. |
| YAML and translation | `global_registration: {run, method: translation, backend, fft_workers, ref_img, mov_img, ref_channel, boundary_mode, recovery}`. `dataset/workflow.py:38-94` maps `method` (default `translation` for the global block), `ref_img`/`mov_img` (`merged-image` or `merged` → `merged`, `single-channel`; default `merged-image`), `ref_channel` (default 0) and `boundary_mode` (→ a `WarpConfig` with backend `translation`). |
| MATLAB counterpart | `STARMapDataset.GlobalRegistration` → `RegisterImagesGlobal` → `DFTRegister3D` then `DFTApply3D` per channel (`src/matlab/STARMapDataset.m:495-626`, `src/matlab/RegisterImagesGlobal.m`). `merged-image` is the channel maximum (`max(..., [], 4)`, lines 547 and 569); `single-channel` matches the channel name containing `ref_channel` (default `"DAPI"`); `input_image` passes an external image. `scale` resizes both signals and divides the shifts; Python has no `scale`. |
| Recovery | `RecoveryConfig(allowed_errors, alternatives)` on the step (`dataset/config.py:17-35`). Only `RegistrationEstimationError` and `InsufficientLandmarksError` may be allowed. Alternatives are estimator configs only (no signal or warp of their own). Translation itself does not raise an estimation error on valid input. |
| Records | `FOV.registration_attempts[round]` gets one entry per attempt: `requested_method`, `actual_method`, `config`, `outcome` (`estimating`, `failed`, `application_failed` or `succeeded`), `failure` (`type`, `message`) and, on success, `application_config` (`dataset/fov.py:418-446`). The backend is not recorded in the attempt; it is in `RegistrationDiagnostics.backend` of the result. |
| Persistence | `transforms.json` entry `kind: "translation"` with `correction_zyx` inline; `log/gr_shifts/<fov>.txt` rows (`fov_id, round, row, col, z` = the detected displacement, `-correction`), written by `save_processing_log` for translation results only (`dataset/fov.py:463-474`). |

### Demons

| Field | Behavior |
| --- | --- |
| Inputs, signal and reference | The common rules above. Both signals are cast to float32; demons compares intensities directly, so the field depends on the signal mode and on channel gains. |
| Estimator | `DemonsConfig(variant="demons", iterations=(100, 50, 25), smoothing_sigma=1, pyramid_mode="antialias")` → SimpleITK `DemonsRegistrationFilter` (or the `diffeomorphic`, `symmetric`, `fast_symmetric` filters) on float32 images of unit spacing (`registration/_demons.py`). `antialias` pads for pyramiding and downsamples with a Butterworth filter by 2 per level; `sitk` uses `sitk.Shrink`. The field is upsampled and doubled between levels. The filter's own stopping rule (maximum RMS change 0.02, SimpleITK default) can end a level early; the elapsed iterations and RMS change are not recorded. |
| Z rules | 3D only, every axis ≥ 4. Z=1 is rejected. |
| Transform | `DenseDisplacementTransform(displacement_zyx)`: a float pull field on the reference grid, `moving index = reference index + displacement[index]`, direction `reference_to_moving`, voxel units, read-only. SimpleITK is used with unit spacing, so the field ignores physical spacing. |
| Defaults | As above. The workflow default iterations are the same; MATLAB defaults differ (next row). |
| YAML and translation | `local_registration: {run, method, iterations, smoothing_sigma, pyramid_mode, ref_channel, boundary_mode, recovery}`. `method` defaults to `demons` for the local block; `diffeomorphic`, `symmetric` and `fast_symmetric` become `DemonsConfig(variant=...)`. The adapter defaults the local block's `ref_img`/`mov_img` to `single-channel` (`dataset/workflow.py:43-44`); the schema does not declare them (`workflow/schemas/config.schema.yaml:569-586`, `additionalProperties: true`), and its `method` enum lists only `demons`, `tps`, `cpd`. `boundary_mode` → `WarpConfig(backend="simpleitk")`. |
| MATLAB counterpart | `STARMapDataset.LocalRegistration` → `RegisterImagesLocal` → `imregdemons(mov, ref, Iterations, 'PyramidLevels', floor(log2(Z)) (≥1), 'AccumulatedFieldSmoothing', afs)` then `imwarp` per channel (`src/matlab/STARMapDataset.m:630-705`, `src/matlab/RegisterImagesLocal.m`). Defaults: `Iterations=10`, `AccumulatedFieldSmoothing=1`, `merged-image` (maximum) for both signals. `rsf_single_fov.m:63` passes only `ref_layer`, so the MATLAB workflow always uses the maximum. The field is discarded. |
| Recovery | As for translation. SimpleITK errors become `RegistrationEstimationError`; a missing SimpleITK raises `RegistrationBackendUnavailableError`, which never recovers. |
| Records | As for translation. `RegistrationDiagnostics(method="demons", backend="simpleitk")`; `converged` and `iterations_completed` stay `None`. |
| Persistence | `transforms.json` entry `kind: "dense"` naming `<round>_field.npz`, which holds one array `result_<i>` per dense result of the round. No `gr_shifts` row. |

### TPS

| Field | Behavior |
| --- | --- |
| Inputs, signal and reference | The common rules above. Landmarks are detected on each signal with its own noise floor, so the signal mode changes the landmark sets. |
| Estimator | `TpsConfig`: noise-floor landmarks in both signals (`median + detection_noise_sigma × 1.4826 × MAD`), nearest-neighbour matching within `match_distance_voxels`, farthest-point subsampling to `max_control_points`, SciPy `RBFInterpolator` (thin-plate kernel, `smoothing`) evaluated on a coarse grid of stride `grid_spacing_voxels` in YX, zoomed (`interpolation_order`), optionally smoothed and clamped (`registration/_tps.py`, `_landmarks.py`, `_fields.py`). Fewer than `min_matches` matches raises `InsufficientLandmarksError`. |
| Z rules | 3D only, every axis ≥ 2. |
| Transform | Dense pull field, as for demons. |
| Defaults | `detection_noise_sigma=3`, `match_distance_voxels=10`, `min_matches=50`, `max_control_points=1000`, `smoothing=1`, `grid_spacing_voxels=32`, `interpolation_order=3`, `field_smoothing_sigma=None`, `clamp_sampling_coordinates=True`. |
| YAML and translation | `local_registration.method: tps`, with MATLAB-style names mapped by the adapter: `detection_threshold` → `detection_noise_sigma`, `match_distance` → `match_distance_voxels`, `tps_smoothing` → `smoothing`, `grid_spacing` → `grid_spacing_voxels`. `boundary_mode` → `WarpConfig(backend="scipy")`. |
| MATLAB counterpart | None. |
| Recovery | As for demons; `InsufficientLandmarksError` is the usual allowed category, for example with `alternatives` naming CPD or demons. |
| Records | As for demons; `RegistrationDiagnostics(method="tps", backend="scipy")`. The landmark counts in the diagnostics stay `None`. |
| Persistence | As for demons: `kind: "dense"` and `<round>_field.npz`. |

### CPD

| Field | Behavior |
| --- | --- |
| Inputs, signal and reference | As for TPS. |
| Estimator | `CpdConfig`: the same landmark detection, then coherent point drift (optional affine CPD first, then non-rigid CPD with a Gaussian kernel of width `kernel_width_voxels`, regularization `regularization_weight`, outlier weight `outlier_fraction`) on anchors and their `neighbors_per_anchor` neighbours within `candidate_radius_voxels`, evaluated to a coarse grid and zoomed (`registration/_cpd.py`). |
| Z rules | 3D only, every axis ≥ 2. |
| Transform | Dense pull field. |
| Defaults | Direct: `detection_noise_sigma=5`, `grid_spacing_voxels=16`, `kernel_width_voxels=None`, `regularization_weight=2`, `outlier_fraction=0.15`, `affine_first=True`, `candidate_radius_voxels=15`, `neighbors_per_anchor=3`. The workflow adapter overrides two of them: `detection_noise_sigma=3.0` and `grid_spacing_voxels=32` (`dataset/workflow.py:64-67`). |
| YAML and translation | `local_registration.method: cpd`, with `beta` → `kernel_width_voxels`, `lmbda` → `regularization_weight`, `cpd_w` → `outlier_fraction`, `candidate_radius` → `candidate_radius_voxels`, `k_neighbors` → `neighbors_per_anchor`, plus the TPS names. |
| MATLAB counterpart | None. |
| Recovery | As for TPS. |
| Records | As for TPS, with `method="cpd"`. |
| Persistence | As for TPS. |

### Transform application

| Field | Behavior |
| --- | --- |
| Inputs | The moving round's full ZYXC image (not the signal) and each of its snapshots, on the transform's moving grid. The reference round is not resampled. |
| Function | `apply_transform(moving, transform, *, config: WarpConfig)` (`registration/_api.py:129-192`) resamples ZYX or ZYXC, reusing one transform for every channel, onto the transform's reference grid. |
| Policy | `WarpConfig(backend, boundary_mode="constant", fill_value=0, output_dtype="input", integer_rounding="nearest_even", clip_to_dtype=True, fft_workers=1)`. The backend must match the transform: `translation` for `TranslationTransform`, `scipy` or `simpleitk` for dense fields. |
| Interpolation | Translation: an exact `np.roll` with zero-filled wrapped bands for integer corrections; a Fourier shift for fractional ones. `scipy`: `map_coordinates` order 1, slice by slice, in float64. `simpleitk`: `DisplacementFieldTransform` and linear `ResampleImageFilter` in float64. Integer outputs are rounded nearest-even, clipped and cast once per application. |
| Boundary | Translation: zero fill only. Dense: constant `fill_value` or nearest extrapolation. |
| Default per method | The estimator returns its own `application_config`: translation → `WarpConfig(backend="translation")`, demons → `simpleitk`, TPS and CPD → `scipy`. A step's `warp` replaces it; the YAML sets only `boundary_mode`. |
| Z rules | Any shape the transform accepts. |
| YAML and translation | Only `boundary_mode` in either block, translated to `WarpConfig(backend=<method's backend>, boundary_mode=...)` (`dataset/workflow.py:89-92`). |
| MATLAB counterpart | `DFTApply3D` for translations; `imwarp` (linear, zero fill) for demons fields. |
| Recovery | Application errors never recover: the attempt is marked `application_failed` and the error propagates (`dataset/fov.py:431-438`). |
| Records | The `WarpConfig` used is stored in the successful attempt (`application_config`) and in the `RegistrationResult`. |
| Persistence | `application_config` is saved with each result in `transforms.json`; the registered checkpoint stores the resampled image and snapshots as TIFFs. The transforms are reloaded with `load_checkpoint`; nothing re-applies them on reload. |

## How registration steps resample the image and the §2.5 snapshots

`FOV.register` (`dataset/fov.py:399-447`) estimates one transform per moving
round and step, then applies it at once:

1. `apply_transform` resamples the round's detection image and, with the same
   `WarpConfig`, every snapshot the recipe kept for that round
   (`dataset/fov.py:432-435`). The results replace the images; the transform
   is appended to `registration_results[round]`.
2. With two steps, the image and every snapshot are resampled twice, and
   integer outputs are rounded to the dtype after each step. The W-232 test
   `test_two_registration_steps_resample_every_snapshot_in_the_same_sequence`
   (`test/test_recipe_sources.py:182-202`) pins exactly this: each snapshot
   equals its pre-registration image with the stored results re-applied in
   order.
3. The round's metadata becomes the transform's reference metadata.
4. `preprocessing_record["transforms"][round][image]` lists, for `detection`
   and each snapshot, the results applied in order, with `result` (the index
   in `registration_results[round]`), `method` and `kind` (`translation` or
   `dense`) (`dataset/fov.py:449-461`). The reference round's lists are empty.

## Morphology registration runs only through MATLAB

The Snakemake rule `nuclei_registration` (`workflow/rules/registration-py.smk:74-89`,
also in `registration.smk`) runs the MATLAB script
`workflow/scripts/nuclei_registration.m` even when the Python backend is
selected. The script loads the `additional_round` rounds into the `other`
layer, applies min–max contrast, reads the reference round's `*ch04.tif`
(DAPI), rotates it by `rotate_angle`, and calls `GlobalRegistration` with
`ref_img="input_image"` (that DAPI volume) and `mov_img="single-channel"`
(the channel whose name contains the top-level `ref_channel`). It is
translation only, writes `log/gr_shifts/<fov>_nr.txt` and saves the other
rounds' registered images. Python has no counterpart: `RoundState.other_rounds`
are loaded but never registered to a shared stain, and there is no external
reference image input.

## Registration metrics in `evaluation/registration.py`

| Function | What it measures | Limits |
| --- | --- | --- |
| `evaluate_translation` | Per-round absolute and L2 error between supplied ZYX displacements, with a strict per-axis tolerance gate | Translations only; no dense-field or affine error; supplied convention, no sign conversion. |
| `normalized_cross_correlation` | Centered Pearson correlation of two equal-shaped arrays in float64; constant input → `None` | Always every element: no valid-overlap mask, so zero-filled borders after a shift count as data and after-registration values are biased against large corrections. |
| `structural_similarity` | scikit-image SSIM with a required positive `data_range` and a policy `volume`, `mip`, `slice` or `plane`; window the largest odd size ≤ 7 that fits | No mask either; the caller chooses `data_range`, so before/after values are comparable only if both use the same range; domains smaller than the window are `None`. |
| `evaluate_mask_overlap` | IoU and Dice of Boolean masks | Caller-supplied masks. |
| `evaluate_landmark_alignment`, `evaluate_registration` | Point matching and a before/after bundle of the above | Never detects or registers; no coverage, fold or transform summary; no optimizer diagnostics. |

The registration benchmark task (`benchmark/_adapters.py:25-120`) evaluates
`ncc`, `ssim` and `translation` from these functions on full arrays.

## Discrepancies with the agreed §2.6 scope

| # | Discrepancy | Proposed resolution |
| --- | --- | --- |
| 1 | The Python `merged` signal sums channels (float64), while MATLAB `merged-image` takes the channel maximum. `FOV.save_reference_image` already writes the maximum for `ref_merged`, so Python registration and its saved reference image disagree. | `RegistrationSignalConfig(mode="max")` is the default and matches MATLAB; `sum` stays selectable. The legacy value `merged-image` (and `merged`) means maximum. Recorded in `docs/migration.md` as an intentional change ({doc}`registration-contract`, "Registration signal"). |
| 2 | The workflow adapter defaults local registration to `single-channel` (channel 0), while MATLAB's workflow uses `merged-image`; the schema does not expose local `ref_img`/`mov_img`. | The legacy local default becomes `merged-image` (maximum), as in MATLAB; the schema declares `ref_img` and `mov_img` for the local block as Python-only keys. |
| 3 | Local estimators reject Z=1 (`registration/_api.py:50-53`). | Z=1 is genuine 2D estimation behind the singleton-Z interface for demons and the new rigid, affine and B-spline methods; TPS and CPD declare 3D only ({doc}`registration-contract`, "Z=1"). |
| 4 | Each registration step resamples the image and every snapshot again, rounding integer data after each step. | Steps compose into one pull field; every snapshot is resampled once, from its pre-registration image ({doc}`registration-contract`, "One final resampling"). |
| 5 | There is no rigid, affine or B-spline method. | Add `rigid`, `affine` and `bspline`, estimated in physical space with elastix ({doc}`registration-algorithms`). |
| 6 | Morphology registration runs only through MATLAB (`nuclei_registration`). | Python other-round and external-reference registration from a selected shared stain, with transfer to the round's other channels ({doc}`registration-contract`, "Other-round and external-reference registration"). The MATLAB rule stays. |
| 7 | Method sets are hard-coded in several modules: `registration/_api.py`, `dataset/config.py:14`, `dataset/workflow.py:38-94`, `:204`, `io/_checkpoint.py:346`, `benchmark/_adapters.py:8-9` and the schema enum. | `REGISTRATION_METHODS` per {doc}`method-registry`; each place derives from it. |
| 8 | `reference_channel` is one zero-based index for both rounds; MATLAB matches channel names (`contains`), and its local reference branch indexes by number while the moving branch matches names. | The signal selects a channel by index or by label, per round when needed (shared stain). |
| 9 | Each method picks its own application backend (`translation`, `simpleitk`, `scipy`); only `boundary_mode` reaches YAML. The SimpleITK and SciPy linear resamplers differ at boundaries. | One recipe-level `WarpConfig` for the final resampling ({doc}`registration-contract`). |
| 10 | Estimation ignores physical spacing; SimpleITK demons runs with unit spacing. | New global and B-spline methods estimate in physical space; demons, TPS and CPD keep index space (declared per method). |
| 11 | Persistence knows only `translation` and `dense`; `FORMAT_VERSION` is 1; `gr_shifts` holds translations only. | New `transforms.json` kinds and `FORMAT_VERSION = 2` with a version-1 reader ({doc}`registration-contract`, "Persistence"). `gr_shifts` keeps its columns and translation rows. |
| 12 | Metrics have no valid-overlap mask, no coverage and no displacement-field truth metric; attempts do not record the backend or optimizer diagnostics. | A routine QC helper and `evaluate_displacement_field` in `starfinder.evaluation`, and richer attempt records ({doc}`registration-contract`, "Routine QC"). |
| 13 | The schema's `local_registration.method` enum (`demons`, `tps`, `cpd`) is narrower than the adapter (also `translation` and the three demons variants). | Widen the schema to the registered names plus the legacy aliases, kept equal by a test (W-240 choice 4, accepted). |
| 14 | MATLAB demons defaults (`Iterations=10`, `floor(log2(Z))` levels) differ from Python's (`(100, 50, 25)`, three levels). | Recorded only; no default changes (defaults beyond translation-only are E01's, W-94). |

## Golden test

`src/python/test/test_registration_golden.py` pins the current behavior on a
seeded fixture of two uint16 rounds, 8×32×32 voxels and four channels. The
moving round shows 60 puncta displaced by (1, 3, −2) voxels plus a smooth YX
bump of up to (0, 1.5, −1.0) voxels, with other channel gains. Through
`FOV.run` it pins:

* translation only: correction (−1, −4, 2) (the bump adds to the Y shift near
  the centre) and the digest of the registered moving round;
* translation → demons: the same correction, the digest of the demons field
  and the digest of the moving round after both resamplings.

The reference round and the fixture inputs are pinned too. Every configuration
is built by one helper, `registration_config`, which is the only part of the
test that refers to `RegistrationStep` or `PipelineConfig.registration`. The
pinned runs use each config's defaults, so a changed `DemonsConfig` default
changes a digest. The §2.6 refactor replaces that helper; its one other named
edit is the translation → demons image digest, which one final resampling
changes ({doc}`registration-contract`, "Tests the implementation changes").
Two further tests show that a changed demons iteration count and the
`single-channel` signal change the pinned translation → demons digests. The translation-only digests do not depend on the
signal on this fixture, because every signal gives the same integer peak.

Three single-thread runs (ITK, OpenMP, BLAS threads 1) gave bit-identical
digests, so the test uses exact equality.
