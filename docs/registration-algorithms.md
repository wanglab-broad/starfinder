# Registration reference

Lookup tables for registration methods, options, transforms, records and files.
The {doc}`guide <registration-contract>` shows a complete call; the
{doc}`registration API <api/registration>` documents the classes and precise rules.
Option descriptions below come from the config class docstrings.

## Methods

```{method-table} starfinder.registration.REGISTRATION_METHODS
```

## Method options

### Translation

Estimate a shift by phase correlation.

See {py:class}`~starfinder.registration.TranslationConfig` for the precise rules.

```{config-options} starfinder.registration.TranslationConfig
:exclude: method
```

### Rigid

Estimate rotation and translation.

See {py:class}`~starfinder.registration.RigidConfig` for the precise rules.

```{config-options} starfinder.registration.RigidConfig
:exclude: method
```

### Affine

Estimate rotation, translation, scale and shear.

See {py:class}`~starfinder.registration.AffineConfig` for the precise rules.

```{config-options} starfinder.registration.AffineConfig
:exclude: method
```

### B-spline

Estimate a smooth deformation on a control grid.

After earlier steps, elastix gets a fixed-image mask by default: the reference voxels
whose pull through the earlier steps lands inside the moving grid, eroded by 2 voxels
from its edges inside the grid, with elastix's own mask erosion off. The step's
diagnostics record it as `support_mask`. An empty mask, or one too small for elastix,
raises `RegistrationEstimationError` with its voxel count.

The erosion does not remove the reference grid's own faces. They hold data; the
zero-filled border lies only where the pull leaves the moving grid. Eroding the faces
too, as the 2026-11-06 talk pilot's mask did, leaves 3 of 8 planes after a 1-plane Z
shift, too few for elastix's samplers on a small stack.

See {py:class}`~starfinder.registration.BSplineConfig` for the precise rules.

```{config-options} starfinder.registration.BSplineConfig
:exclude: method
```

### Demons

Estimate a dense deformation with image intensities.

The default variant, `demons`, is classic Thirion demons with the reference image's
gradient, run level by level on the package's pyramid. Each iteration, in the order
of MATLAB's `imregdemons` and with SimpleITK's boundary rules:

1. samples the moving image at each voxel plus the field, by linear interpolation. A
   sample counts as inside the moving grid up to half a voxel beyond its first and
   last voxels, and there takes the edge voxel's value;
2. computes the update `(F − M) ∇F / (|∇F|² + (F − M)²)`, with `∇F` from central
   differences inside the grid and 0 along each axis on the two faces normal to it.
   The update is 0 where `|F − M| < 0.001`, where the denominator is below 1e-9, or
   where the sample lies outside the moving grid;
3. adds the update to the field, then smooths the field with a sampled Gaussian of
   standard deviation `smoothing_sigma` voxels and radius ⌈3σ⌉ (7 taps for σ = 1),
   normalised to sum 1, on each axis with replicated edges.

So the returned field is smoothed after the last update too. Every level runs all of
its iterations. The diagnostics record, per level, `elapsed_iterations` and
`final_rms_change`, the RMS of the last update over the voxels sampled inside the
grid. This variant used to run SimpleITK's `DemonsRegistrationFilter`, which smooths
before each update and so returned the last update unsmoothed (see the
{doc}`migration guide <migration>`). The variants `diffeomorphic`, `symmetric` and
`fast_symmetric` still run SimpleITK's filters.

The pyramid is not MATLAB's: it filters with a separable rather than a radial
Butterworth filter and resamples on a corner-aligned grid.

See {py:class}`~starfinder.registration.DemonsConfig` for the precise rules.

```{config-options} starfinder.registration.DemonsConfig
:exclude: method
```

### TPS

Match landmarks with a thin-plate spline.

The landmarks of each signal are its regional maxima above the `noise` threshold of
`local_maxima` with `threshold_value` set to `detection_noise_sigma`, the border excluded
(see {py:class}`~starfinder.spot_finding.NoiseLandmarkConfig`). The median and MAD are
taken over the voxels above the minimum when more than half of the signal's voxels sit at
it, and over every voxel otherwise; on an integer signal the MAD has a floor of one grey
level, and a floating-point signal with a MAD of 0 raises `RegistrationEstimationError`
naming the image (reference or moving) and the channel
([truncated channels](spot-finding-contract.md#truncated-channels)). Each reference
landmark is matched to the nearest moving landmark within `match_distance_voxels`; at
least `min_matches` pairs are needed, and farthest-point sampling keeps at most
`max_control_points` of them for the fit.

The step's diagnostics record the reference and moving landmark counts, the matched
pairs (`matched_landmark_count`) and `landmarks`: for `reference` and `moving`, each
channel's noise record (`threshold`, `median`, `mad`, `mad_floored`, `estimation`,
`n_estimation`) and the image's `candidates`, and `control_points`, the pairs the fit
uses. The landmarks' notes (an estimate above the minimum, a floored MAD) are listed in
`warnings`, prefixed with the image.

See {py:class}`~starfinder.registration.TpsConfig` for the precise rules.

```{config-options} starfinder.registration.TpsConfig
:exclude: method
```

### CPD

Align landmark sets with coherent point drift.

The landmarks are TPS's, with `detection_noise_sigma` as the multiplier. CPD has soft
correspondences, not pairs: it subsamples the reference landmarks by farthest-point
anchors and their neighbours, gathers the moving landmarks within
`candidate_radius_voxels` of them and drops the reference points without one. The
diagnostics record the same `landmarks`, where `control_points` is the number of
reference points CPD fits, and `matched_landmark_count` is the number of moving
landmarks it gathered. The other methods leave the landmark counts and `landmarks`
`None`, and a saved record leaves `landmarks` out.

See {py:class}`~starfinder.registration.CpdConfig` for the precise rules.

```{config-options} starfinder.registration.CpdConfig
:exclude: method
```

## Recipe options

### Recipe

See {py:class}`~starfinder.registration.RegistrationRecipe` for the precise rules.

```{config-options} starfinder.registration.RegistrationRecipe
```

### Step

See {py:class}`~starfinder.registration.RegistrationStep` for the precise rules.

```{config-options} starfinder.registration.RegistrationStep
```

### Recovery

See {py:class}`~starfinder.registration.RecoveryConfig` for the precise rules.

```{config-options} starfinder.registration.RecoveryConfig
```

### Signal

See {py:class}`~starfinder.registration.RegistrationSignalConfig` for the precise rules.

```{config-options} starfinder.registration.RegistrationSignalConfig
```

### Final resampling

See {py:class}`~starfinder.registration.WarpConfig` for the precise rules.

```{config-options} starfinder.registration.WarpConfig
```

### QC criteria

See {py:class}`~starfinder.registration.RegistrationQcConfig` for the precise rules.

```{config-options} starfinder.registration.RegistrationQcConfig
```

## Transforms

The {doc}`API <api/registration>` describes transform application and composition.

| Type | Contents |
| --- | --- |
| `TranslationTransform` | Displacement in ZYX voxels |
| `AffineTransform` | Affine matrix and optional physical parameters |
| `BSplineTransform` | Control grid, coefficients and voxel spacing |
| `DenseDisplacementTransform` | Displacement vectors on a ZYX grid |
| `TransformChain` | Composed step transforms |

## Records

`fov.registration_attempts[round]` records estimation and application attempts.
The following keys describe completed attempts.

| Key | Estimation entry | Application entry |
| --- | --- | --- |
| `record` | `"estimation"` | `"application"` |
| `step`, `attempt` | Index of the step, and 0 for its own config or 1, 2, … for recovery alternatives | — |
| `requested_method`, `actual_method`, `fallback` | The step's method, the one that ran, and whether it was an alternative | — |
| `backend`, `backend_versions` | Backend identity and package versions | — |
| `reference`, `reference_sha256` | The reference round's name, or the external reference's label and image hash | — |
| `config` | The method config that ran | — |
| `support_mask` | B-spline only: `used`, `reason` when unused, the mask's `voxels` and its `fraction` of the grid, as in the step's diagnostics | — |
| `application_config` | — | The `WarpConfig` of the final resampling |
| `outcome` | `"succeeded"`, `"failed"` or `"rejected"` | `"succeeded"` or `"application_failed"` |
| `failure` | Error type and message, and the criterion of a rejection | Error type and message |
| `qc` | The step's QC, when available | The chain's QC, when available |

QC records describe overlap and image similarity.

| Value | Meaning |
| --- | --- |
| `coverage` | Valid overlap fraction |
| `ncc_before`, `ncc_after`, `ncc_gain` | Correlation before and after registration, and its gain |
| `ssim_before`, `ssim_after` | SSIM before and after registration |

## Files

| File | Written by | Contents |
| --- | --- | --- |
| `checkpoints/registered/<round>.ome.tif` | `FOV.run` with the `registered` stage | The registered round; snapshots under `registered/<snapshot>/` |
| `checkpoints/registered/transforms.json` | the same | Each round's step transforms, diagnostics and attempts, its resampling, and the recipe |
| `checkpoints/registered/<round>_field.npz` | the same | The dense fields and B-spline coefficients of that round |
| `other_rounds/<round>/image.ome.tif`, `registration.json`, `field.npz` | `FOV.register` on other rounds and `FOV.prepare_morphology`, with `checkpoints` | A registered other round, its chain, attempts and reference; `field.npz` only for dense or B-spline transforms |
| `nuclei_registration.json` | `FOV.save_registration_record` | The registration record of a FOV's other rounds |

Paths are inside the FOV's folder, `<output_root>/fovs/<fov_id>/` by default.
`FOV.load_checkpoint("registered")` and `FOV.load_registered_round(name)` read them
back.

## Workflow keys

The {doc}`workflow reference <workflow/reference>` generates the accepted keys
from the package. The [workflow specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/workflow-configuration.md#explicit-registration-recipe)
keeps route and backend rules.
