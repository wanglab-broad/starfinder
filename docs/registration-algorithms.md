# Registration reference

The methods, options, transforms, records and files of registration. The method and
option tables are read from the code when the site is built. The
{doc}`registration guide <registration-contract>` explains how to use them; the
{doc}`registration API <api/registration>` documents every class.

## Methods

```{method-table} starfinder.registration.REGISTRATION_METHODS
```

Each method's name is its `method` value, used by the workflow key. Its registry entry
also declares its step kind (`global` or `local`), the kind of transform it returns
and its estimation space. A global step returns a translation or an affine transform;
a local step returns a B-spline or a dense displacement field. Rigid, affine and
B-spline estimate in physical units from `spacing_zyx` (space `physical`); the others
work on voxel indices (`index`).

## Method options

### Translation

Phase correlation on the whole volume, with the peak taken at whole voxels. `backend`
is `"scipy_fft"` or `"skimage"`; `skimage` needs `fft_workers=1`.

```{config-options} starfinder.registration.TranslationConfig
:exclude: method
```

### Rigid, affine and B-spline

These run with elastix, from the `registration-elastix` extra, imported only when one
of them runs; B-spline also needs `local-registration`, whose SimpleITK evaluates its
grid. `metric` is `"mattes"` (Mattes mutual information, with `histogram_bins` bins for
rigid and 32 bins for affine and B-spline) or `"ncc"` (normalized correlation). `iterations` are per
pyramid level; `samples` are the random samples per
iteration, drawn with `random_seed`.

* **Pyramid.** `levels=None` chooses up to 4 levels from the image size; the
  [algorithms specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/registration-algorithms.md#shared-numerical-rules-for-the-elastix-methods) gives the schedule.
* **Start.** Rigid and affine start from the identity about the grid centre; a
  preceding `translation` step handles large shifts. B-spline starts from zero
  displacement.
* **B-spline grid.** `grid_spacing_physical` is the control-point spacing, in the unit
  of `spacing_zyx` and equal on every axis; `None` uses the X extent divided by 8.
* **Unknown spacing.** When both rounds have no `spacing_zyx`, unit spacing is used and
  the diagnostics record `spacing_source="unknown_unit"` and a warning in `warnings`.

```{config-options} starfinder.registration.RigidConfig
:exclude: method
```

```{config-options} starfinder.registration.AffineConfig
:exclude: method
```

```{config-options} starfinder.registration.BSplineConfig
:exclude: method
```

### Demons

SimpleITK demons, from the `local-registration` extra. `variant` is `"demons"`,
`"diffeomorphic"`, `"symmetric"` or `"fast_symmetric"`. `iterations` gives one entry
per pyramid level, coarsest first. `smoothing_sigma` smooths the displacement field, in
voxels. `pyramid_mode` is `"antialias"` (Butterworth-filtered downsampling) or `"sitk"` (SimpleITK shrinking). On Z=1 the pyramid acts on Y and X only and the field
has no Z component.

```{config-options} starfinder.registration.DemonsConfig
:exclude: method
```

### TPS and CPD

Both detect spots in the two signals above a noise threshold that
`detection_noise_sigma` sets. They evaluate the displacements on a coarse grid of every
Z plane, with Y and X steps of at most `grid_spacing_voxels`, then interpolate it to
every voxel. TPS matches spots within `match_distance_voxels` and needs `min_matches` pairs,
otherwise it raises `InsufficientLandmarksError`. CPD aligns the two spot sets by
coherent point drift; it raises `InsufficientLandmarksError` when it finds too few spots
or too few candidates within `candidate_radius_voxels`. A configuration translated from the MATLAB keys sets CPD's
`detection_noise_sigma` to 3.0 and `grid_spacing_voxels` to 32 unless it gives them.

```{config-options} starfinder.registration.TpsConfig
:exclude: method
```

```{config-options} starfinder.registration.CpdConfig
:exclude: method
```

## Recipe options

```{config-options} starfinder.registration.RegistrationRecipe
```

`steps` are at least one step: zero or more global steps followed by at most one local
step. `warp=None` derives the final resampling from the chain of the steps that
succeeded: `WarpConfig()` (the translation backend) for a chain of translations, and
`WarpConfig(backend="scipy")` otherwise. So a translation step whose recovery
alternative is `affine` resamples with the translation backend when the translation
succeeds.

```{config-options} starfinder.registration.RegistrationStep
```

```{config-options} starfinder.registration.RecoveryConfig
```

`allowed_errors` may name `RegistrationEstimationError`, `InsufficientLandmarksError`
and `RegistrationRejectedError`; the alternatives must have the step's kind and run in
order.

### Signal

```{config-options} starfinder.registration.RegistrationSignalConfig
```

| `mode` | Signal |
| --- | --- |
| `"max"` | The maximum over the channels |
| `"sum"` | The float64 sum over the channels |
| `"channel"` | `reference_channel` of the reference round and `moving_channel` (default: the same) of the moving round, each a zero-based index or a channel label; an `ExternalReference` is used as given, whatever the mode |

### Final resampling

```{config-options} starfinder.registration.WarpConfig
```

| `backend` | Resampling |
| --- | --- |
| `"translation"` | Exact and Fourier shifts, for translations and chains of translations. When `warp` is set, a recipe whose steps and recovery alternatives are all translations must use it, and any other recipe must not; with `warp=None` it is chosen from the chain, as above |
| `"scipy"` | Linear interpolation with SciPy |
| `"simpleitk"` | Linear interpolation with SimpleITK |

`boundary_mode`, `fill_value` and `output_dtype` set what a pull point outside the moving
image gets and the type of the output. How pull points at and beyond the edge are
treated differs between the backends: the
[specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/registration.md#one-final-resampling) ("One final resampling") describes the resampling, and
{py:class}`~starfinder.registration.WarpConfig` documents the options.

### QC criteria

```{config-options} starfinder.registration.RegistrationQcConfig
```

Every criterion is off when `None`. They apply to each step, not to the whole chain;
`max_translation_voxels` applies to translation steps only and `max_fold_fraction` to
B-spline and dense steps only, and an undefined value rejects nothing.
`projections=True` makes `registration_qc` return the Z maximum projections for
overlays; `FOV` does not keep them.

## Transforms

Every transform maps reference voxel indices `p` (ZYX) to moving voxel indices. The
four step types and `TransformChain`, which composes them, record the direction
`"reference_to_moving"` and the units `"voxel_index"`.

| Type | Main fields | Pull map |
| --- | --- | --- |
| `TranslationTransform` | `displacement_zyx` | `p + d` |
| `AffineTransform` | `matrix_zyx` (4 × 4, voxel indices), `physical` (for an elastix estimate, the backend's matrix, centre, translation and parameters) | `A p + b` |
| `BSplineTransform` | the control grid (`grid_size_xyz`, `grid_origin_xyz`, `grid_spacing_xyz`, `grid_direction_xyz`), `coefficients`, `spacing_zyx`, `order` 3 | `p` plus the B-spline displacement; `dense()` evaluates it with SimpleITK |
| `DenseDisplacementTransform` | `displacement_zyx`, shape (Z, Y, X, 3) | `p + u(p)` |
| `TransformChain` | the step transforms in order | `T₁(T₂(…Tₙ(p)))`; `pull_field()` gives the composed field |

The last step is applied to the reference point first. A chain of translations reduces
to one translation whose displacement is the sum.

## Errors

| Error | Raised when |
| --- | --- |
| `InvalidRegistrationConfigError` (a `ValueError`) | A field of a method, signal, warp or QC config is invalid; `RegistrationRecipe`, `RegistrationStep` and `RecoveryConfig` raise `ValueError` or `TypeError` |
| `IncompatibleGeometryError` (`starfinder.image`) | The grids differ, or the shape is below the method's minimum or has a Z it does not support |
| `RegistrationBackendUnavailableError` (an `ImportError`) | The method's optional extra is not installed; never recovered |
| `RegistrationEstimationError` | The estimator fails or returns non-finite values |
| `InsufficientLandmarksError` | TPS or CPD finds too few spots or matches |
| `RegistrationRejectedError` | A step fails a configured QC criterion; the message names the criterion, value and bound |
| `UnsupportedTransformOperationError` | A transform or chain has an unsupported direction, units, type or order, a chain is empty, or a transform is applied with a backend that does not support it |

A recovery may allow only `RegistrationEstimationError`, `InsufficientLandmarksError` and
`RegistrationRejectedError`; the last two are `RegistrationEstimationError`s.
`UnsupportedTransformOperationError` is a `NotImplementedError` and is never recovered,
nor are the configuration, geometry and backend errors.

## Records

`fov.registration_attempts[round]` lists one entry per estimation attempt, then, once
every step has succeeded, one application entry for the round.

| Key | Estimation entry | Application entry |
| --- | --- | --- |
| `record` | `"estimation"` | `"application"` |
| `step`, `attempt` | Index of the step, and 0 for its own config or 1, 2, … for recovery alternatives | — |
| `requested_method`, `actual_method`, `fallback` | The step's method, the one that ran, and whether it was an alternative | — |
| `backend`, `backend_versions` | The backend that ran (`None` when the estimator failed) and the versions of the method's packages | — |
| `reference`, `reference_sha256` | The reference round's name, or the external reference's label and image hash | — |
| `config` | The method config that ran | — |
| `application_config` | — | The `WarpConfig` of the final resampling |
| `outcome` | `"succeeded"`, `"failed"` or `"rejected"` | `"succeeded"` or `"application_failed"` |
| `failure` | Error type and message, and the criterion of a rejection | Error type and message |
| `qc` | The step's QC; `None` for an attempt that failed before its QC | The QC of the whole chain, when the resampling succeeded |

Each QC record holds these values. Coverage is always defined; the NCC and SSIM values
may be `None`, with a reason.

| Value | Meaning |
| --- | --- |
| `coverage` | Fraction of reference voxels whose pull point lies inside the moving grid |
| `ncc_before`, `ncc_after`, `ncc_gain` | Pearson correlation with the reference over that overlap, before and after, and the gain |
| `ssim_before`, `ssim_after` | SSIM of the Z maximum projections, 7 × 7 window, over the valid columns eroded by 3 pixels |

Its `details` hold the transform summary and, for a step, the optimizer's
diagnostics. The summary of an affine or rigid step has `rotation_angle` only for a
rigid step; it is `None` for an affine one.

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

The configuration schema accepts these keys only with `backend: python`.

| Key, under `rules.<rule>.parameters` | Meaning |
| --- | --- |
| `registration.steps` | A list of steps: `method` (a name in the method table), that config's fields, and optional `recovery` and `signal` |
| `registration.signal`, `.warp`, `.qc` | The fields of `RegistrationSignalConfig`, `WarpConfig` and `RegistrationQcConfig` |
| `registration.reference_round` | When given, must equal the dataset's reference round |
| `reference_image_view` | `{mode: max}` (default) or `{mode: channel, channel: <index or label>}`: the reference image the rule writes; the resident subtile rules refuse it |

On the MATLAB backend, the `global_registration` and `local_registration` blocks are
translated to one recipe of one or two steps, or to no registration when neither is
enabled; {doc}`migration` describes how.
