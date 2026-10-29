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

```{config-options} starfinder.registration.TranslationConfig
:exclude: method
```

### Rigid

Estimate rotation and translation.

```{config-options} starfinder.registration.RigidConfig
:exclude: method
```

### Affine

Estimate rotation, translation, scale and shear.

```{config-options} starfinder.registration.AffineConfig
:exclude: method
```

### B-spline

Estimate a smooth deformation on a control grid.

```{config-options} starfinder.registration.BSplineConfig
:exclude: method
```

### Demons

Estimate a dense deformation with image intensities.

```{config-options} starfinder.registration.DemonsConfig
:exclude: method
```

### TPS

Match landmarks with a thin-plate spline.

```{config-options} starfinder.registration.TpsConfig
:exclude: method
```

### CPD

Align landmark sets with coherent point drift.

```{config-options} starfinder.registration.CpdConfig
:exclude: method
```

## Recipe options

### Recipe

```{config-options} starfinder.registration.RegistrationRecipe
```

### Step

```{config-options} starfinder.registration.RegistrationStep
```

### Recovery

```{config-options} starfinder.registration.RecoveryConfig
```

### Signal

```{config-options} starfinder.registration.RegistrationSignalConfig
```

### Final resampling

```{config-options} starfinder.registration.WarpConfig
```

### QC criteria

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

The {doc}`workflow reference <workflow-configuration>` generates the accepted keys
from the schema. The [workflow specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/workflow-configuration.md#explicit-registration-recipe)
keeps route and backend rules.
