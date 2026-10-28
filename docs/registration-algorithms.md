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
of them runs. `metric` is `"mattes"` (Mattes mutual information with
`histogram_bins` bins) or `"ncc"` (normalized correlation). `iterations` are per
pyramid level, with no convergence test; `samples` are the random samples per
iteration, drawn with `random_seed`.

* **Pyramid.** `levels=None` takes the largest number of levels, up to 4, at which
  the smaller of Y and X is still at least 16 at the coarsest level. Level `l` of `L`
  shrinks Y and X by `2^(L−1−l)`, and Z by the same factor capped at `max(1, Z // 4)`.
* **Start.** Rigid and affine start from the identity about the grid centre; a
  preceding `translation` step handles large shifts. B-spline starts from zero
  displacement.
* **B-spline grid.** `grid_spacing_physical` is the control-point spacing, in the unit
  of `spacing_zyx` and equal on every axis; `None` uses the X extent divided by 8.
* **Unknown spacing.** When both rounds have no `spacing_zyx`, unit spacing is used and
  the diagnostics record `spacing_source="unknown_unit"` with a warning.
* **Failures.** A constant signal raises `RegistrationEstimationError("constant
  registration signal")`; any elastix error becomes a `RegistrationEstimationError`.

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
voxels. `pyramid_mode` is `"antialias"` (low-pass filtering, then downsampling by 2) or
`"sitk"` (SimpleITK shrinking). On Z=1 the pyramid acts on Y and X only and the field
has no Z component.

```{config-options} starfinder.registration.DemonsConfig
:exclude: method
```

### TPS and CPD

Both detect spots in the two signals, as voxels above `median + k · 1.4826 · MAD`
with `k = detection_noise_sigma`, and turn their displacements into a dense field on a
coarse grid of `grid_spacing_voxels`, which is then interpolated to every voxel. TPS
matches spots within `match_distance_voxels` and needs `min_matches` pairs, otherwise
it raises `InsufficientLandmarksError`. CPD aligns the two spot sets by coherent point
drift. A configuration translated from the MATLAB keys sets CPD's
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

`steps` are zero or more global steps followed by at most one local step.
`warp=None` uses `WarpConfig()` (the translation backend) for a chain of translations
and `WarpConfig(backend="scipy")` otherwise.

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
| `"channel"` | `reference_channel` of the reference round and `moving_channel` (default: the same) of the moving round, each a zero-based index or a channel label |

### Final resampling

```{config-options} starfinder.registration.WarpConfig
```

| `backend` | Resampling |
| --- | --- |
| `"translation"` | Chains of translations only: an exact shift for whole voxels, a Fourier shift otherwise; constant zero fill only |
| `"scipy"` | Linear interpolation in float64, one Z plane at a time |
| `"simpleitk"` | Linear interpolation with SimpleITK |

A pull point outside the moving grid gets `fill_value` with `boundary_mode="constant"`,
or the nearest edge value with `"nearest"`. With `output_dtype="input"`, integer
images are rounded to the nearest even integer, clipped to the dtype's range and cast
once.

### QC criteria

```{config-options} starfinder.registration.RegistrationQcConfig
```

Every criterion is off when `None`. `projections=True` makes `registration_qc` return
the Z maximum projections for overlays; `FOV` does not keep them.

## Transforms

Every transform maps reference voxel indices `p` (ZYX) to moving voxel indices; the
direction is `"reference_to_moving"` and the units `"voxel_index"`.

| Type | Main fields | Pull map |
| --- | --- | --- |
| `TranslationTransform` | `displacement_zyx` | `p + d` |
| `AffineTransform` | `matrix_zyx` (4 × 4, voxel indices), `physical` (the backend's matrix, centre, translation and parameters) | `A p + b` |
| `BSplineTransform` | the control grid (`grid_size_xyz`, `grid_origin_xyz`, `grid_spacing_xyz`, `grid_direction_xyz`), `coefficients`, `spacing_zyx`, `order` 3 | `p` plus the B-spline displacement; `dense()` evaluates it |
| `DenseDisplacementTransform` | `displacement_zyx`, shape (Z, Y, X, 3) | `p + u(p)` |
| `TransformChain` | the step transforms in order | `T₁(T₂(…Tₙ(p)))`; `pull_field()` gives the composed field |

The last step is applied to the reference point first. A chain of translations reduces
to one translation whose displacement is the sum.

## Errors

| Error | Raised when |
| --- | --- |
| `InvalidRegistrationConfigError` (a `ValueError`) | A config field is invalid |
| `IncompatibleGeometryError` (`starfinder.image`) | The grids differ, or the shape is below the method's minimum or has a Z it does not support |
| `RegistrationBackendUnavailableError` (an `ImportError`) | The method's optional extra is not installed; never recovered |
| `RegistrationEstimationError` | The estimator fails or returns non-finite values |
| `InsufficientLandmarksError` | TPS or CPD finds too few spots or matches |
| `RegistrationRejectedError` | A step fails a configured QC criterion; the message names the criterion, value and bound |
| `UnsupportedTransformOperationError` | A transform is applied in an unsupported direction or way |

The last three are `RegistrationEstimationError`s, so a recovery may allow them.

## Records

`fov.registration_attempts[round]` lists one entry per estimation attempt, then one
application entry for the round.

| Key | Estimation entry | Application entry |
| --- | --- | --- |
| `record` | `"estimation"` | `"application"` |
| `step`, `attempt` | Index of the step, and 0 for its own config or 1, 2, … for recovery alternatives | — |
| `requested_method`, `actual_method`, `fallback` | The step's method, the one that ran, and whether it was an alternative | — |
| `backend`, `backend_versions` | The backend that ran and the versions of its packages | — |
| `reference`, `reference_sha256` | The reference round's name, or the external reference's label and image hash | — |
| `config` | The method config that ran | — |
| `application_config` | — | The `WarpConfig` of the final resampling |
| `outcome` | `"succeeded"`, `"failed"` or `"rejected"` | `"succeeded"` or `"application_failed"` |
| `failure` | Error type and message, and the criterion of a rejection | Error type and message |
| `qc` | The step's QC | The QC of the whole chain |

Each QC record holds these values; any of them may be `None` with a reason.

| Value | Meaning |
| --- | --- |
| `coverage` | Fraction of reference voxels whose pull point lies inside the moving grid |
| `ncc_before`, `ncc_after`, `ncc_gain` | Pearson correlation with the reference over that overlap, before and after, and the gain |
| `ssim_before`, `ssim_after` | SSIM of the Z maximum projections, 7 × 7 window, over the valid columns eroded by 3 pixels |

Its `details` hold the transform summary and the optimizer's diagnostics.

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

| Key, under `rules.<rule>.parameters` | Meaning |
| --- | --- |
| `registration.steps` | A list of steps: `method` (a name in the method table), that config's fields, and optional `recovery` and `signal` |
| `registration.signal`, `.warp`, `.qc` | The fields of `RegistrationSignalConfig`, `WarpConfig` and `RegistrationQcConfig` |
| `registration.reference_round` | Must equal the dataset's reference round |
| `reference_image_view` | `{mode: max}` (default) or `{mode: channel, channel: <index or label>}`: the reference image the rule writes; the resident subtile rules refuse it |

On the MATLAB backend, the `global_registration` and `local_registration` blocks are
translated to one recipe of at most two steps; {doc}`migration` describes how.
