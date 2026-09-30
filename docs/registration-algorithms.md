# Registration algorithm specification

Status: Proposed

This page specifies the numerical methods that §2.6 adds: rigid, affine and
B-spline registration, and demons on Z=1 data. It also gives the engineering
validation design of registration task group 6. The recipe, transforms,
composition and records they plug into are in {doc}`registration-contract`;
the current behavior is in {doc}`registration-baseline`. Nothing here is
implemented yet, and nothing here sets a default beyond the agreed
translation-only default: the parameter values below are what each method
uses when a recipe selects it.

## Evidence for the backends

The backend choices follow the W-244 comparison of SimpleITK and elastix
(development spike, not E01). Its saved table is
`/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/runs/W-244/20260929T232115Z-a2d4ce97/comparison.csv`
(826 rows; manifest `comparison-manifest.json`, summary tables
`comparison-summary.md`, in the same directory). W-244 ran rigid, affine and
B-spline with both backends on the §2.12 development sizes, on `medium`
(32×512×512), on a 1×512×512 Z=1 case and on `large` (30×1024×1024), with
held-out seeds 100 to 102. Its recommendation is provisional: the detailed
comparison belongs to E01 (W-94). The rows cited below are held-out, matched
settings, medians over cases and seeds, errors in voxels.

| Method | Backend | W-244 rows cited | Reason |
| --- | --- | --- | --- |
| B-spline | elastix (`itk-elastix` 0.25.4, `itk` 5.4.7) | `bspline` × {`elastix`, `sitk`} × `matched` on `medium`, `large` and `z1-512` | Median error 0.14 against 1.13 (medium), 0.16 against 1.12 (large), 0.12 against 0.40 (Z=1 512); 95th percentile 0.80 against 3.71 (medium); registration time 8.8 s against 40.2 s (medium). SimpleITK's own default B-spline took about 340 s on medium and one of five calls failed. The grids were not strictly matched (mainly in Z), so the size of the gap is not attributed to the backend alone. |
| Affine | elastix | `affine` × backends × `matched` on `medium`, `large`, `z1-512`, `small`, `z1`, `small-sp122` | Median 0.14 against 1.36 (medium), 1.35 against 2.56 (large); similar on Z=1 512 (0.17 against 0.18) and on the development sizes (0.03 against 0.03 to 0.04); SimpleITK better on the spacing case (0.07 against 0.17). elastix is faster from medium upward (2.3 s against 10.7 s) and shares one integration and persistence with B-spline. |
| Rigid | elastix (weak preference) | `rigid` × backends × `matched` on all scenes | Equivalent on the development sizes (0.03 to 0.06). elastix closer on medium local-deformation cases (`gaussian_small` 0.09 against 0.75); neither recovers the realistic rigid case with shifts near 50 voxels without pre-alignment. Chosen for one integration with affine and B-spline and for speed from medium upward; SimpleITK remains an adequate locked alternative (worker notes, open choice). |
| Demons on Z=1 | SimpleITK 2.5.3 (locked, `local-registration` extra) | `z1_3d_probe` rows (all six method × backend pairs failed on a 3D Z=1 volume) and the Z=1 rows with `dimension_estimated = 2` (all completed) | W-244 did not run demons. Its Z=1 rows show that both backends estimate genuine 2D transforms from 2D arrays and fail on 3D volumes with Z=1; the existing demons implementation already uses SimpleITK, whose demons filters are dimension-generic, so no new backend is needed. |

elastix is a new optional dependency. Using it needs a dependency and lock
change, which the task group 3 issue must have authorized before it starts;
W-244 measured about 11 s of import time and 0.5 to 0.8 GB of extra baseline
RSS per process, so it is imported lazily and only when a recipe selects an
elastix method.

## What each method corrects

The §2.12 image-formation model ({doc}`synthetic-specification`, step 6) maps
reference points of round r by `F_r(q) = q + t_r + A_r(q − c) + P_r m(u) +
Σ_k v_rk exp(−‖q − c_k‖² / (2 l_k²))`: a translation, a linear term about the
grid centre, a smooth quadratic term and local Gaussian displacements. The
ideal registration pull map of round r is `F_r`, and the truth pull field is
`forward_displacement`. Translation (the current global method) models only
`t_r`.

| Method | Problem addressed | Cause in the image-formation model |
| --- | --- | --- |
| Rigid | Rounds rotated as well as shifted relative to the reference, which translation alone leaves as a position-dependent residual that grows with distance from the rotation centre. | `A_r` restricted to a rotation (`A_r = R − I`), as when a sample is remounted between rounds; physical-space estimation keeps a rotation rigid for anisotropic voxels. |
| Affine | Global scaling and shear between rounds, for example uniform expansion or shrinkage of the sample, which rigid cannot absorb. | The full linear term `A_r(q − c)` with `t_r`. |
| B-spline | Smooth, spatially varying displacement that no global map explains, with a regularized parameterization on a physical control grid. | The quadratic term `P_r m(u)` and broad local terms `v_rk`, at scales larger than the control spacing. |
| Demons on Z=1 | Local registration is impossible for single-plane data today: every local estimator rejects Z=1. | The same local terms `v_rk` and `P_r m(u)`, restricted to YX (the model requires zero Z motion for Z=1). |

## Shared numerical rules for the elastix methods

* **Images.** The reference and moving signals of the step
  ({doc}`registration-contract`, "Registration signal") are passed as float32
  ITK images with spacing `spacing_zyx` reversed to XYZ, origin 0 and identity
  direction; Z=1 inputs are passed as 2D images. Unknown spacing uses 1 and is
  recorded ({doc}`registration-contract`, "Physical-space estimation").
* **Constant signals.** A reference or moving signal with zero variance raises
  `RegistrationEstimationError("constant registration signal")` before the
  backend runs.
* **Pyramid.** `levels` = the largest L ≤ 4 with `min(Y, X) / 2^(L−1) ≥ 16`
  (at least 1); shrink factor `2^(L−1−l)` at level l on every axis, with
  Gaussian smoothing sigma `factor / 2` voxels at levels with factor > 1 and
  none at factor 1 (elastix `FixedRecursiveImagePyramid` and
  `MovingRecursiveImagePyramid` with an explicit schedule). This is the W-244
  matched rule. It shrinks Z as well; W-244 ran it on Z=9 and Z=32. At
  `min_shape_zyx` (Z=4) with a large YX extent it would shrink Z below one
  plane (factor 8 at four levels), so capping the Z factor at
  `max(1, Z // 4)` is an open choice (worker notes).
* **Sampling and seed.** 4096 random samples per iteration
  (`NumberOfSpatialSamples`), redrawn every iteration, with
  `RandomSeed = 1` so repeated single-thread calls are bit-identical (W-244
  determinism rows: identical field hashes for all six method and backend
  pairs at one thread).
* **Optimizer.** elastix `AdaptiveStochasticGradientDescent` with automatic
  parameter estimation, the default of elastix's default parameter maps. It
  runs a fixed number of iterations per level; there is no convergence test.
* **Interpolation during estimation.** The interpolator of elastix's default
  parameter map for the method, recorded in the diagnostics. The final
  resampling is always Starfinder's ({doc}`registration-contract`, "One final
  resampling"), never elastix's (`WriteResultImage=false`).
* **Threads.** The ITK global thread count follows the caller's setting; the
  validation runs use one thread. W-244 found one-thread and four-thread
  results bit-different for both backends.
* **Diagnostics.** `converged=None` (no convergence test), `iterations_completed`
  per level, the final metric value per level and the stop condition from the
  elastix log, the backend versions, and the index-space matrix or grid.
* **Failure.** Any elastix or ITK exception (for example "too many samples map
  outside moving image buffer") and any non-finite parameter becomes
  `RegistrationEstimationError` with the backend message. A missing
  `itk-elastix` raises `RegistrationBackendUnavailableError` naming the
  `registration-elastix` extra; it never recovers. An input below
  `min_shape_zyx` raises `IncompatibleGeometryError` before the backend runs.
  No method is substituted unless the recipe's recovery names it.

## Rigid

| Field | Specification |
| --- | --- |
| Backend and version | elastix through `itk-elastix` 0.25.4 (`itk` 5.4.7); W-244 `rigid` rows. |
| Transform | `EulerTransform` (3D: three angles and a translation; 2D: one angle and a translation), stored as `AffineTransform` with its physical parameters ({doc}`registration-contract`). |
| Metric | `AdvancedMattesMutualInformation`, 32 histogram bins (W-244 tuning candidate k2, the lowest mean median error for rigid on the development seed). |
| Optimizer | Adaptive stochastic gradient descent, 200 iterations per level, 4096 samples. |
| Pyramid | Shared rule above. |
| Initialization | Identity, with the rotation centre at the geometric centre of the reference grid (`AutomaticTransformInitialization` with `GeometricalCenter`). Large shifts are handled by a preceding `translation` step in the recipe, which the composition carries; W-244 found that neither backend recovers shifts near 50 voxels from identity. |
| Parameters (units) | `RigidConfig(metric="mattes", histogram_bins=32, iterations=200` (per level), `samples=4096, levels=None` (derived), `random_seed=1)`. Angles are in radians and translations in the spatial unit of `spacing_zyx` in the stored physical parameters; the index-space matrix is in voxels. |
| Defaults | As listed: W-244 development-tuned starting values, not operating points. |
| Convergence | Fixed iterations; the final metric value and the per-level values are recorded. |
| Failure | Shared rules above. |
| Resources (W-244, one thread, matched) | 0.9 s (1×32×32), 1.0 s (9×32×32), 1.3 s (1×512×512), 2.8 s (32×512×512) of registration; peak RSS 615 to 909 MiB including 539 to 602 MiB of process baseline. Plus about 11 s of `itk-elastix` import once per process. |

## Affine

| Field | Specification |
| --- | --- |
| Backend and version | elastix, as for rigid; W-244 `affine` rows. |
| Transform | `AffineTransform` of dimension 3 or 2 (12 or 6 parameters), stored as `AffineTransform` with its physical parameters. |
| Metric | `AdvancedNormalizedCorrelation` (W-244 candidate k1). |
| Optimizer | Adaptive stochastic gradient descent, 200 iterations per level, 4096 samples. |
| Pyramid | Shared rule above. |
| Initialization | Identity about the geometric centre, as for rigid; a preceding `translation` or `rigid` step provides pre-alignment. |
| Parameters (units) | `AffineConfig(metric="ncc", iterations=200, samples=4096, levels=None, random_seed=1)`. |
| Defaults | As listed; not operating points. |
| Convergence | As for rigid. |
| Failure | Shared rules above. A matrix with `det A ≤ 0` (a reflection or collapse) is returned but flagged in the QC transform summary; it is rejected only if the recipe's QC configures it. |
| Resources (W-244) | 0.7 s (1×32×32), 0.8 s (9×32×32), 1.0 s (1×512×512), 2.3 s (32×512×512), 5.1 s (30×1024×1024); peak RSS up to 1598 MiB on `large` (baseline 780 MiB). |

## B-spline

| Field | Specification |
| --- | --- |
| Backend and version | elastix for estimation (W-244 `bspline` rows); SimpleITK 2.5.3 to evaluate the stored grid into a dense field (W-244 round-trip rows: the converted field differs from the backend's own by at most 1.2e-7 voxels). |
| Transform | Cubic `BSplineTransform` on a physical control grid, dimension 3 or 2, stored with its fixed parameters and coefficients ({doc}`registration-contract`). |
| Metric | `AdvancedNormalizedCorrelation` (W-244 candidate k1), with no bending-energy penalty, as in the W-244 matched setting. |
| Optimizer | Adaptive stochastic gradient descent, 200 iterations per level, 4096 samples. |
| Pyramid | Shared rule above; `GridSpacingSchedule` 1 at every level, so the control spacing is constant over levels. |
| Initialization | Zero coefficients (identity). A preceding global step provides the global alignment. |
| Parameters (units) | `BSplineConfig(metric="ncc", grid_spacing_physical=None, iterations=200, samples=4096, levels=None, random_seed=1)`. `grid_spacing_physical` is the final control-point spacing in the spatial unit of `spacing_zyx`, the same on every axis; `None` means the physical YX extent `X · spacing_x` divided by 8 (the W-244 matched rule: 64 units on `medium`, 128 on `large`). |
| Defaults | As listed; not operating points. |
| Convergence | As for rigid. |
| Failure | Shared rules above. |
| Resources (W-244) | 1.1 s (1×32×32), 4.0 s (9×32×32), 1.7 s (1×512×512), 8.8 s (32×512×512), 11.6 s (30×1024×1024); peak RSS up to 1598 MiB on `large`. Converting the grid to a dense float64 field adds 24 bytes per voxel if a caller requests the whole field; the final resampling evaluates it plane by plane. |

## Demons on Z=1

| Field | Specification |
| --- | --- |
| Backend and version | SimpleITK 2.5.3, the existing demons filters (`DemonsRegistrationFilter` and the three variants), run on 2D images. |
| Transform | A 2D displacement field, embedded as `DenseDisplacementTransform` of shape (1, Y, X, 3) with the Z component exactly 0. |
| Metric | The demons force of each variant (intensity difference driven), as in 3D. |
| Optimizer | Demons iterations per level; the update field and the accumulated field are smoothed with `smoothing_sigma` (voxels), as in 3D. |
| Pyramid | The existing `antialias` pyramid applied in YX only (Butterworth low-pass and downsampling by 2 per level of Y and X; Z untouched), or `sitk` shrinking of Y and X only. The number of levels is `len(iterations)`. |
| Initialization | Zero field at the coarsest level; each finer level starts from the upsampled, doubled field, as in 3D. |
| Parameters (units) | The existing `DemonsConfig` fields and defaults: `variant="demons"`, `iterations=(100, 50, 25)` per level, `smoothing_sigma=1` (voxels), `pyramid_mode="antialias"`. No field is added. |
| Defaults | Unchanged from 3D. |
| Convergence | A level ends after its iterations or when the RMS change of the field falls below SimpleITK's `MaximumRMSError` (0.02); the elapsed iterations and the final RMS change per level are recorded (today they are not). |
| Failure | Y or X below 4 raises `IncompatibleGeometryError`; 1 < Z < 4 stays rejected. SimpleITK errors become `RegistrationEstimationError`; a missing SimpleITK raises `RegistrationBackendUnavailableError`. |
| Resources | Measured for this page with SimpleITK 2.5.3 at one thread and iterations (100, 50, 25): 0.04 s for 1×64×64 and 0.78 s for 32×64×64, about 10 and 6 µs per voxel. That projects to about 3 s for 1×512×512 and 40 to 45 s for 1×2048×2048, with the images (float32) and field (two float64 components) at about 25 bytes per pixel, or roughly 0.1 GB at 2048×2048. |

## Engineering validation design (task group 6)

Task group 6 is engineering validation only (W-152 §2.14 decision,
2026-09-29): known-answer synthetic scenes with pass/fail tolerances fixed
before the run, in default-tier (and, where noted, extended-tier) pytest
modules. It has no arm × condition matrix, no threshold or parameter sweep, no
benefit flag, no default selection from comparative data, and no separate run
framework or HTML report. Method comparisons belong to E01 (W-94).

The tolerances are engineering bounds that detect a broken implementation,
fixed before the run; they are not accuracy claims. Where the answer is exact
(V1, V6 to V11) the tolerance is analytic. For V2 and V3 the W-244 held-out
elastix errors (matched settings, per case, median over seeds) on the same
development scenes were at most 0.03 voxels median and 0.06 p95 (rigid on the
rigid case, `small`, `z1` and the spacing case; affine on `linear_small`,
`small` and `z1`), so 0.25 and 0.5 voxels leave a wide margin. V4 and V5 have
no matching W-244 row: the nearest are elastix B-spline on `polynomial_small`
at `medium` (0.14 median, 0.74 p95) and at Z=1 512 (0.12, 0.61), and W-244 did
not run demons. Their bounds are therefore provisional. They stay fixed during task
group 6; if a correct implementation cannot meet one, the change goes to
Jiahao rather than being adjusted in the run (worker notes, open choice).

All scenes stay within the project bound of 32×64×64 voxels, four channels and
four rounds, are generated in session with `generate_registration_pair`,
`GeometryConfig` or a direct construction, and use seeds 100, 101 and 102.

| # | Check | Fixture | Metric (source) | Pass/fail tolerance |
| --- | --- | --- | --- | --- |
| V1 | Known translation | `DEVELOPMENT_SIZES` `small` (9×32×32) and `z1` (1×32×32), `GeometryConfig` translations (2, −3, 4) and (0, −3, 4), benchmark appearance | `evaluate_translation` (`starfinder.evaluation`); and the existing registration benchmark task with `evaluation: {translation: {tolerance: 0.5}}` | Exact integer correction: every axis error < 0.5 (strict gate), so any wrong sign or axis fails. |
| V2 | Known rigid map | `small`, `z1` and `small` with spacing (1, 2, 2): rotation up to ±3° about the grid centre plus a shift of at most 3 voxels (the W-244 rigid case) | `evaluate_displacement_field` against `forward_displacement`, over the valid overlap | Median ≤ 0.25 and p95 ≤ 0.5 voxels, per seed. |
| V3 | Known affine map | `small` and `z1`, `DEFORMATION_PRESETS["linear_small"]` | As V2 | Median ≤ 0.25 and p95 ≤ 0.5 voxels. |
| V4 | Known deformation, B-spline | 16×64×64 and 1×64×64 with a supplied `GeometryConfig` polynomial term of peak 3 voxels (identity median error above 1 voxel by construction) | As V2 | Median ≤ 0.5 and p95 ≤ 1.0 voxels, and p95 below the identity p95. |
| V5 | Known deformation, demons on Z=1 | 1×64×64 with one local Gaussian control of magnitude 2 voxels and scale 8 | As V2; plus the field's Z component | Median ≤ 0.5 and p95 ≤ 1.0 voxels; Z component exactly 0. |
| V6 | Composition | The four analytic examples of {doc}`registration-contract`; and a chain (translation, affine, dense) on 8×32×32 | Named-point values; `TransformChain.pull_field()` compared with sequential evaluation of the step maps | 1e-12 voxels. |
| V7 | Persistence round trip | Every transform kind on 8×32×32 and 1×32×32, saved as a version-2 registered checkpoint and reloaded | Pull field and re-applied images compared with the uninterrupted run | Bit-identical (`array_equal`). A version-1 checkpoint written by the start revision loads as `sequential`. |
| V8 | Z=1 and small-Z rules | 1×32×32, 2×32×32, 3×32×32 for every registered method | Raised error type; Z component of the result | Methods with `2` in `dimensions` succeed on Z=1 with no Z motion; TPS and CPD raise `IncompatibleGeometryError` on Z=1; elastix methods and demons raise it on Z=2 and 3. |
| V9 | Boundary and coverage | 1×32×32 ramp image, translation correction (0, −5, 0) | `registration_qc` coverage; fill band values | Coverage exactly 27/32; the 5 filled rows equal `fill_value` with constant boundary and the edge row with `nearest`. |
| V10 | Empty and constant input | Constant moving signal; moving round shifted beyond the grid | Raised error; `registration_qc` NCC and SSIM | Rigid, affine and B-spline raise `RegistrationEstimationError("constant registration signal")`; NCC and SSIM are `None` with a reason; coverage 0. |
| V11 | Determinism | V2 to V5 fixtures, seed 100, run twice at one thread | Transform and image digests | Bit-identical. |

Metrics that must be added to `starfinder.evaluation` for these checks, as
specified in {doc}`registration-contract` ("Routine QC"):

* `evaluate_displacement_field(estimated, truth, *, mask, spacing_zyx=None)`:
  median, 95th percentile and maximum of the displacement error, in voxels and
  optionally physical units, over a mask;
* `registration_qc(...)`: valid overlap, coverage, masked NCC, Z-projection
  SSIM, transform summary and optimizer diagnostics;
* an optional `mask` keyword for `normalized_cross_correlation` and
  `structural_similarity`, with unchanged results when it is omitted.

`evaluate_translation`, `forward_displacement`, `generate_registration_pair`,
`GeometryConfig`, `DEFORMATION_PRESETS`, `DEVELOPMENT_SIZES` and the
registration benchmark task are used as they are.

Resource plan: every fixture is at most 16×64×64 voxels; elastix calls take
about 1 to 4 s each at these sizes (W-244) plus one 11 s import per process,
so the task group 6 modules are expected to run in a few minutes at one
thread. Each run records wall time and maximum RSS with `/usr/bin/time -v`
against the 1800 s and 4 GiB stop targets.
