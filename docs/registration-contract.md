# Registration recipe contract

Status: Accepted (W-246, 2026-09-30, at 47057aa)

This page proposes the §2.6 registration contract: a `RegistrationRecipe` of
ordered `RegistrationStep`s, a configurable registration signal, typed
transforms that compose into one pull field, one final resampling of every
image of a moving round, reloadable transforms, and native Python registration
of other rounds from a shared stain. It builds on the {doc}`method-registry`
design and on the §2.5 snapshots of the {doc}`preprocessing-contract`. The
current behavior is recorded in {doc}`registration-baseline`; the numerical
methods are specified in {doc}`registration-algorithms`. Nothing here is
implemented yet.

## Terms

The page uses the terms decided on 2026-09-29 (W-152):

* a **stage** is a pipeline stage: preprocessing, registration, spot finding;
* a **method** is a registrable algorithm of a stage, such as `affine`;
* a **recipe** is a stage's ordered configuration, such as `RegistrationRecipe`;
* a **step** is one position in a recipe and names a method. Recipe positions
  are never called stages.

## Settled decisions this page follows

Decided by Jiahao on 2026-09-28 and 2026-09-29 (W-152):

1. One public module, `starfinder.registration`, with private method modules.
2. `RegistrationRecipe` whose ordered entries keep the existing name
   `RegistrationStep`, in a field named `steps`.
3. Method sets derived from `REGISTRATION_METHODS` ({doc}`method-registry`).
4. The maximum signal is the default, a documented change from the Python sum
   that matches MATLAB.
5. Rigid, affine and B-spline are estimated in physical space.
6. "Other-round and external-reference registration" replaces a morphology
   namespace.
7. The backend is chosen from the W-244 evidence: SimpleITK is already locked;
   elastix is added only where the evidence justifies it
   ({doc}`registration-algorithms`).
8. YAML keys and saved formats keep `steps` and `step`.

## RegistrationRecipe

```python
@dataclass(frozen=True)
class RegistrationRecipe:
    """Ordered registration steps that compose into one pull transform per moving round."""
    steps: tuple[RegistrationStep, ...]                 # at least one; allowed sequences below
    signal: RegistrationSignalConfig = RegistrationSignalConfig()   # maximum
    warp: WarpConfig | None = None                      # final resampling; None: derived (below)
    reference_round: str | None = None                  # None: RoundState.reference_round
    qc: RegistrationQcConfig = RegistrationQcConfig()   # routine QC; rejects nothing by default


@dataclass(frozen=True)
class RegistrationStep:
    """One position of a recipe: a registered method config, optional recovery and signal."""
    config: object                                      # an exact key of REGISTRATION_METHODS
    recovery: RecoveryConfig | None = None
    signal: RegistrationSignalConfig | None = None      # None: the recipe's signal
```

`PipelineConfig.registration` becomes `RegistrationRecipe | None` (default
`None`, no registration), replacing `tuple[RegistrationStep, ...]`. Both classes
live in `starfinder.dataset`, next to `PipelineConfig` and `RecoveryConfig`, as
`RegistrationStep` does today; `RegistrationSignalConfig`,
`RegistrationQcConfig`, the transform types and `REGISTRATION_METHODS` live in
`starfinder.registration`.

`RegistrationStep` loses `reference_image`, `moving_image` and
`reference_channel` (replaced by `signal`) and `warp` (replaced by the recipe's
single `warp`). `RecoveryConfig` keeps its fields; each alternative must be a
`REGISTRATION_METHODS` key with the same step kind as the step's config.

Validation (all in `__post_init__`, before any image is read):

* `steps` is a nonempty tuple of `RegistrationStep`; each config's exact type
  is a `REGISTRATION_METHODS` key (`TypeError('unsupported registration
  config')`, as today);
* the sequence is allowed (next section);
* `reference_round`, when set, must equal `RoundState.reference_round` in
  `FOV.run`; `FOV.register_rounds` (below) accepts any loaded round.

### Allowed step sequences

A recipe is zero or more **global** steps followed by at most one **local**
step, with at least one step:

```text
steps := global* local?        (length >= 1)
global := translation | rigid | affine
local  := demons | bspline | tps | cpd
```

The step kind is the `step_kind` declared in the method's registry entry.
Examples: `(translation)`, `(translation, affine)`, `(translation, demons)`,
`(rigid, bspline)`, `(bspline)`. Rejected: `(demons, translation)`,
`(demons, bspline)`, `()`. A local step is always the innermost map of the
composition (next sections), so its field is evaluated only at reference grid
points and the composition needs no interpolation of a dense field. A second
local step would need one; it is left out until a use needs it (worker notes,
open choice).

A recovery alternative keeps its step's kind, so recovery never changes an
allowed sequence into a rejected one.

## Registration signal

```python
@dataclass(frozen=True)
class RegistrationSignalConfig:
    """How one ZYX registration signal is built from a round's ZYXC source image."""
    mode: str = "max"                          # "max", "sum" or "channel"
    reference_channel: int | str | None = None # "channel": zero-based index or channel label
    moving_channel: int | str | None = None    # "channel": None means reference_channel
```

* **Source.** The signal of each round is built from the §2.5
  `registration_source` snapshot of the preprocessing recipe, or from the
  detection image when it is `None` or there is no preprocessing recipe, as
  today ({doc}`preprocessing-contract`).
* **Reduction.** `max` takes the channel maximum (MATLAB `merged-image`);
  `sum` the float64 channel sum (the current Python `merged`); `channel`
  selects one channel per round. A label is looked up in the round's channel
  labels (`Dataset.channel_order` for sequencing rounds, the round's own
  labels for other rounds); an unknown label or an index outside the image is
  a `ValueError` before estimation. `max` and `channel` are exact in float64
  for every integer dtype up to 32 bits.
* **Output.** A float64 ZYX array. Z is kept: nothing projects, and Z=1 stays
  a singleton.
* **Moving signal of step k.** It is built once from the moving round's source
  and resampled in float64 through the composition of steps 1 to k−1 with the
  recipe's interpolation and boundary policy, without integer rounding. The
  reference signal is not resampled.
* **Default.** `mode="max"`. This is an intentional change for Python callers
  that relied on the sum; `sum` remains selectable.
* **Per-step override.** `RegistrationStep.signal` replaces the recipe's signal
  for that step only; it exists for the legacy mapping, in which the global and
  local blocks may name different representations.

## Registry entries

Registration methods are registered in `REGISTRATION_METHODS`, a public
module-level `dict[type, RegistrationSpec]` in `starfinder.registration`. The
shared fields (stable `name`, exact config type as the key, `run`, `requires`,
`min_shape_zyx`) and the shared lookup, dependency and provenance rules are
those of {doc}`method-registry` ("Fields of the shared spec" and "Shared and
stage-specific capabilities"). `RegistrationSpec` adds:

| Field | Meaning |
| --- | --- |
| `step_kind` | `global` or `local`; drives the allowed sequences. |
| `dimensions` | `frozenset` of `2` and/or `3`: `2` means a Z=1 input is estimated as genuine 2D; `3` means Z>1 is estimated in 3D. |
| `transform_kind` | Kind of the returned transform: `translation`, `affine`, `bspline` or `dense`. |
| `space` | `index` (estimation on voxel indices) or `physical` (estimation on physical coordinates from `spacing_zyx`). |

`min_shape_zyx` applies to 3D inputs; a Z=1 input is checked against its last
two entries. An input with 1 < Z < `min_shape_zyx[0]`, or with Z=1 for a method
without `2` in `dimensions`, raises `IncompatibleGeometryError` before the
estimator runs. The W-240 field `application_backend` is not used: the recipe's
single `warp` replaces per-method application backends.

| Name | Config | `step_kind` | `dimensions` | `min_shape_zyx` | `transform_kind` | `space` | `requires` (extra) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `translation` | `TranslationConfig` (exists) | global | 2, 3 | (1, 1, 1) | translation | index | none |
| `rigid` | `RigidConfig` (new) | global | 2, 3 | (4, 16, 16) | affine | physical | `itk-elastix` (`registration-elastix`) |
| `affine` | `AffineConfig` (new) | global | 2, 3 | (4, 16, 16) | affine | physical | `itk-elastix` (`registration-elastix`) |
| `bspline` | `BSplineConfig` (new) | local | 2, 3 | (4, 16, 16) | bspline | physical | `itk-elastix` (`registration-elastix`); `SimpleITK` (`local-registration`) to evaluate the grid |
| `demons` | `DemonsConfig` (exists) | local | 2, 3 | (4, 4, 4) | dense | index | `SimpleITK` (`local-registration`) |
| `tps` | `TpsConfig` (exists) | local | 3 | (2, 2, 2) | dense | index | none |
| `cpd` | `CpdConfig` (exists) | local | 3 | (2, 2, 2) | dense | index | none |

The table shows the registry after task group 3. Registry move 2 (task group
2) registers only the four existing methods with their current capabilities,
so it changes no behavior: `demons` then declares `dimensions={3}` and keeps
rejecting Z=1. Task group 3 adds `2` to demons and registers `rigid`, `affine`
and `bspline`.

The names, fields and defaults of the four existing configs do not change. The
demons variants stay one method, `demons`, with `variant` (W-240 choice 6).
The `registration-elastix` extra is a dependency change that the
implementation issue must have authorized; see {doc}`registration-algorithms`.

## Transform types

All transforms are **pull** maps from reference to moving coordinates:
the registered image at reference index `p` is the moving image sampled at
`T(p)`. Indices are ZYX voxel indices of equal reference and moving grids
(no unequal-grid conversion).

| Kind | Type | Stored parameters | Pull map `T(p)` | Direction label |
| --- | --- | --- | --- | --- |
| `translation` | `TranslationTransform` (exists) | `correction_zyx` | `p − c` | `moving_to_reference` (kept) |
| `affine` | `AffineTransform` (new) | `matrix_zyx`, a 4×4 float64 index-space matrix `[[A, b], [0, 1]]`; `physical`, the backend's physical parameters (below) | `A p + b` | `reference_to_moving` |
| `bspline` | `BSplineTransform` (new) | the ITK B-spline fixed parameters (dimension, grid size, origin, spacing, direction, in XYZ physical order), the coefficients (`dimension × grid` float64), `order=3`, and the `spacing_zyx` used for estimation | `p + S⁻¹ P v(P S p)` (below) | `reference_to_moving` |
| `dense` | `DenseDisplacementTransform` (exists) | `displacement_zyx` (Z, Y, X, 3) on the reference grid | `p + u(p)` | `reference_to_moving` |
| chain | `TransformChain` (new) | the ordered tuple of the step transforms of one round | `T₁(T₂(…Tₙ(p)))` | `reference_to_moving` |

Here `S = diag(spacing_zyx)` and `P` reverses ZYX to the XYZ order of ITK; `v`
is the B-spline displacement in physical XYZ units. All types keep
`reference_shape_zyx`, `moving_shape_zyx`, both `ImageMetadata` values and
`units="voxel_index"` (the B-spline additionally records its physical frame),
and are frozen with read-only arrays, as the existing types are.

### Composition order

Step k is estimated on the moving signal already resampled through steps
1 to k−1, so its transform maps reference coordinates into that intermediate
image. The composite pull map of a round with steps `T₁ … Tₙ` is therefore

```text
Φ(p) = T₁(T₂(…Tₙ(p)…))        u(p) = Φ(p) − p
```

The last step is applied first to the reference point and the first step last.
`TransformChain.pull_field()` returns `u` as a `DenseDisplacementTransform`
(float64). It evaluates `Tₙ` at grid points (exact for a dense field) and every
earlier step analytically at the resulting points: a translation subtracts its
correction, an affine applies `A p + b`, and a B-spline is evaluated from its
coefficients. Because a local step is always last, no dense field is ever
interpolated during composition. A chain of translations only reduces to one
`TranslationTransform` whose correction is the sum of the corrections.

### Analytic composition examples

Each example gives the expected pull field `u(p) = Φ(p) − p` at named points,
and the value that the wrong order `Tₙ(…T₁(p))` would give, so an order error
fails. The future test file is `test/test_registration_composition.py`
(registration task group 4); each example is one test, with tolerance 1e-12
voxels because every value is exact in float64 up to rounding of the inputs.

**Example 1: translation then affine**
(`test_translation_then_affine_pull_field`). Grid 8×32×32.

* Step 1: `TranslationTransform(correction_zyx=(−1, −4, 2))`, so
  `T₁(q) = q + (1, 4, −2)`.
* Step 2: `AffineTransform` with `A = [[1, 0, 0], [0, 1, 0.1], [0, 0, 1]]`
  (Y sheared by X) and `b = (0, 0.5, 0)`.
* `Φ(p) = A p + b + (1, 4, −2)`, so `u(p) = (1, 4.5 + 0.1 x, −2)`.

| Point `p` (z, y, x) | Expected `u(p)` | Wrong order gives |
| --- | --- | --- |
| (0, 0, 0) | (1, 4.5, −2) | (1, 4.3, −2) |
| (2, 10, 20) | (1, 6.5, −2) | (1, 6.3, −2) |

**Example 2: affine then dense field**
(`test_affine_then_dense_pull_field`). Grid 8×32×32.

* Step 1: `AffineTransform` with `A₁ = diag(1, 1.1, 0.9)` and
  `b₁ = (0, −2, 3)`.
* Step 2: `DenseDisplacementTransform` with `u₂(p) = (0, 0, 0.05 y)`.
* `Φ(p) = A₁ (p + u₂(p)) + b₁`, so
  `u(p) = (0, 0.1 y − 2, −0.1 x + 0.045 y + 3)`.

| Point `p` (z, y, x) | Expected `u(p)` | Wrong order gives |
| --- | --- | --- |
| (0, 0, 0) | (0, −2, 3) | (0, −2, 2.9) |
| (4, 20, 10) | (0, 0, 2.9) | (0, 0, 3) |
| (2, 30, 31) | (0, 1, 1.25) | (0, 1, 1.45) |

**Example 3: Z=1, translation then affine**
(`test_z1_translation_then_affine_pull_field`). Grid 1×32×32; the affine is a
2D estimate embedded with an identity Z row.

* Step 1: `TranslationTransform(correction_zyx=(0, −3, 1))`, so
  `T₁(q) = q + (0, 3, −1)`.
* Step 2: `AffineTransform` with
  `A = [[1, 0, 0], [0, 0.98, 0.05], [0, −0.05, 0.98]]` and `b = (0, 0.2, −0.4)`.
* `u(p) = (0, −0.02 y + 0.05 x + 3.2, −0.05 y − 0.02 x − 1.4)`. The Z
  component is exactly 0 at every point, and the field has shape (1, 32, 32, 3).

| Point `p` (z, y, x) | Expected `u(p)` | Wrong order gives |
| --- | --- | --- |
| (0, 0, 0) | (0, 3.2, −1.4) | (0, 3.09, −1.53) |
| (0, 10, 20) | (0, 4.0, −2.3) | (0, 3.89, −2.43) |
| (0, 31, 0) | (0, 2.58, −2.95) | (0, 2.47, −3.08) |

**Example 4: physical to index space**
(`test_physical_rigid_to_index_matrix`). With `spacing_zyx = (2, 0.5, 0.5)`,
origin 0 and identity direction, a 2° physical rotation in the Z–X plane,
`R = [[cos θ, 0, −sin θ], [0, 1, 0], [sin θ, 0, cos θ]]` in ZYX order, becomes
the index matrix `S⁻¹ R S = [[0.999391, 0, −0.008725], [0, 1, 0],
[0.139598, 0, 0.999391]]` (six decimals). The index matrix of a physical rigid
transform is not orthogonal when spacing is anisotropic; this is why the
affine matrix is stored in index space and the physical parameters are kept
beside it.

## One final resampling

After every step of a moving round has been estimated, the round is resampled
**once**:

* **What is resampled.** The moving round's detection image and every snapshot
  that §2.5 keeps for it, each from its own pre-registration array, with the
  same composite transform. The reference round is not resampled. Nothing is
  resampled per step: step signals are resampled in float64 only for
  estimation and are discarded.
* **Interpolation.** A chain that reduces to one translation uses the current
  translation path: an exact integer shift (`np.roll` with zero-filled wrapped
  bands), or a Fourier shift for a fractional correction. Every other chain is
  sampled with linear interpolation (`map_coordinates`, order 1, in float64),
  plane by plane: the composite pull points of one Z plane are computed, all
  channels of all images are sampled at them, and the plane is discarded, so
  the full composite field is never held for the whole volume. `simpleitk`
  (linear, float64) remains selectable.
* **Default `warp`.** `None` means `WarpConfig(backend="translation")` for a
  translation-only chain and `WarpConfig(backend="scipy")` otherwise; an
  explicit `WarpConfig` must name a backend valid for the chain (`translation`
  only for translation-only chains).
* **Dtype.** The output keeps the input dtype by default (`output_dtype="input"`);
  `float32` and `float64` are selectable. Interpolation is in float64; integer
  outputs are rounded to the nearest even integer, clipped to the dtype range
  and cast once, as `WarpConfig` does today.
* **Boundary.** With `boundary_mode="constant"`, a pull point outside the
  closed box `[0, n−1]` on any axis receives `fill_value` (default 0).
  `nearest` extends the edge values. A translation-only chain requires constant
  zero fill, as today. For Z=1 the Z coordinate of every pull point is exactly
  0, so sampling is bilinear in YX.
* **No quantization across steps.** Because the transforms compose as
  continuous maps and the only rounding is the final cast, an integer
  translation step does not round or quantize the transforms of later steps,
  and no step's output is rounded to the dtype before the next step.

The records keep, per image, the ordered results that were composed
(`preprocessing_record["transforms"][round][image]`, as in §2.5, with the
new kinds), and the round's single application policy.

## Physical-space estimation

Rigid, affine and B-spline are estimated in physical space and stored in index
space (the B-spline additionally in its physical grid):

1. Both signals are handed to the backend as images with spacing
   `P spacing_zyx`, origin 0 and identity direction. The two `ImageMetadata`
   values must already agree in spacing, origin, direction and unit (the
   equal-grid rule); origin and direction do not change the index-space
   result, so they are not passed.
2. The backend returns a physical pull transform `x ↦ M (x − c) + c + t`
   (XYZ) for rigid and affine, or B-spline coefficients on a physical control
   grid.
3. For rigid and affine the index-space matrix is
   `A = S⁻¹ P M P S` and `b = S⁻¹ P (t + c − M c)`, with `S = diag(spacing_zyx)`.
   `AffineTransform.physical` keeps `matrix_xyz`, `center_xyz`,
   `translation_xyz`, the backend's own parameter vector and names (for
   example Euler angles), and `spacing_zyx`.
4. For the B-spline the index-space pull at `p` is `p + S⁻¹ P v(P S p)`; the
   transform keeps the control grid and coefficients, and
   `BSplineTransform.dense()` evaluates it on the reference grid with SimpleITK
   (`BSplineTransform` and `TransformToDisplacementField`), so reloading and
   applying a stored B-spline never needs elastix. W-244 did not validate this
   route: it converted elastix B-splines with ITK's own
   `TransformToDisplacementFieldFilter` on the elastix transform and compared
   the result with the transformix deformation field (at most 1.2e-7 voxels
   apart), and it used SimpleITK only for SimpleITK's own B-splines. Task
   group 3 must show that the SimpleITK evaluation of stored elastix
   coefficients matches the transformix deformation field within 1e-6 voxels
   before relying on it.
5. **Unknown spacing.** When `spacing_zyx` is `None` on both metadata values,
   estimation uses unit spacing and records `spacing_source="unknown_unit"`
   with a warning in `RegistrationDiagnostics.warnings`; the index-space result
   is then the physical one. This is an open choice for Jiahao (worker notes).

Demons, TPS and CPD keep estimating in index space (`space="index"`); their
behavior does not change.

## Z=1

A Z=1 round is registered by genuine 2D estimation behind the singleton-Z
interface:

* callers pass ZYX(C) arrays with Z=1, as for every other shape; no public
  function takes 2D arrays;
* for a method with `2` in `dimensions`, the estimator receives the YX plane
  and runs the backend's 2D form: `Euler2DTransform` or 2D affine, a 2D
  B-spline, SimpleITK 2D demons (pyramid in YX only). A 3D estimate on a Z=1
  volume is never attempted; W-244 found that both backends fail on it;
* the result is embedded in 3D with no Z motion: a translation with zero Z
  correction, an affine with Z row and column `(1, 0, 0)` and `b_z = 0`, a
  B-spline of dimension 2, or a dense field of shape (1, Y, X, 3) whose Z
  component is exactly 0;
* TPS and CPD declare `dimensions={3}` and keep rejecting Z=1 with
  `IncompatibleGeometryError`;
* example 3 above is the composition check for Z=1.

## Persistence

The registered checkpoint keeps its files (`registered/transforms.json` and
`registered/<round>_field.npz`) and gains transform kinds. Its
`FORMAT_VERSION` (`io/_checkpoint.py`, shared by the three checkpoint headers)
changes from 1 to 2:

| Kind | `transforms.json` entry (in `transform`) | Arrays in `<round>_field.npz` |
| --- | --- | --- |
| `translation` | `correction_zyx` inline (unchanged) | none |
| `affine` (new) | `matrix_zyx` (4×4, inline) and `physical` (inline) | none |
| `bspline` (new) | `bspline` (dimension, grid size, origin, spacing, direction, order, `spacing_zyx`) inline | `result_<i>`: the coefficients |
| `dense` | `field` names the file (unchanged) | `result_<i>`: the field |

Each round's list keeps one entry per step result, in step order, with
`transform`, `diagnostics` and the step index. The round gets one
`application` entry with the recipe's `WarpConfig`, instead of a per-result
`application_config`. The header records the recipe (step method names,
signal, warp, QC config).

* **Reload.** `read_checkpoint(..., "registered")` rebuilds the transforms,
  a `TransformChain` per round and the application policy. JSON floats
  round-trip exactly and the arrays are stored as float64, so the reloaded
  chain's `pull_field()` is bit-identical to the one computed in the
  uninterrupted run.
* **Equivalence.** Applying the reloaded chain to the pre-registration images
  with the stored policy gives arrays bit-identical to the registered images of
  the uninterrupted run (the same code path on the same float64 values). The
  test is `test_registered_checkpoint_reapplies_identically` (task group 4).
* **Version 1.** The reader keeps accepting version 1 for the `registered`,
  `candidates` and `pre_qc` checkpoints. A
  version-1 registered checkpoint loads its `translation` and `dense` results
  with their per-result `application_config` and is marked `sequential`, the
  pre-§2.6 semantics; nothing converts it silently.
* **`gr_shifts`.** `log/gr_shifts/<fov>.txt` keeps its columns
  (`fov_id, round, row, col, z`) and its rows: the detected displacement of
  each translation result. Other kinds are not written there.
* **`run.json`.** Its `format_version` stays 1 (W-240 choice 3); the
  registration records gain the fields below and `config.pipeline.registration`
  becomes the recipe's fields.

## Other-round and external-reference registration

This replaces the MATLAB-only morphology path (`nuclei_registration`) with a
native Python path. The MATLAB rule and script stay unchanged.

```python
@dataclass(frozen=True)
class ExternalReference:
    """A reference signal that is not a round of the dataset."""
    image: np.ndarray            # ZYX, on the same grid as the rounds it registers
    metadata: ImageMetadata
    label: str                   # recorded in the attempts, e.g. "ref_round:ch04"

FOV.register_rounds(recipe: RegistrationRecipe, *, rounds: Sequence[str],
                    reference: str | ExternalReference | None = None) -> FOV
```

* **Shared stain.** The recipe's signal is `mode="channel"` with
  `reference_channel` and `moving_channel` naming the shared stain in each
  round, by label (for example `"ch04"` in the reference round and `"ch00"` in
  a morphology round) or by index. Any `REGISTRATION_METHODS` sequence is
  allowed; the MATLAB path corresponds to `(translation,)`.
* **Reference.** `None` uses the recipe's `reference_round` (default the
  dataset reference round). A round name may be any loaded round, including an
  `other_rounds` member. An `ExternalReference` supplies the signal directly;
  its SHA-256 is recorded. It must have the grid of the moving rounds.
* **Transfer to associated channels.** The chain estimated on the shared stain
  is applied, by the one final resampling above, to every channel of the round
  and to every snapshot kept for it, so the stain and the other channels of
  that round share one transform.
* **Records.** Results, attempts and QC are stored as for sequencing rounds,
  with the reference (round name or external label and hash) in each attempt.
* **Workflow.** The Python counterpart of `nuclei_registration` reads the
  reference stain as the MATLAB script does (the reference round's `ch04`
  image, rotated by `rotate_angle`) and the `additional_round` entries with
  their `channel_order`, and writes `log/gr_shifts/<fov>_nr.txt` and the
  registered images under the MATLAB file names. It is added as a Python rule
  of the Python backend; the MATLAB backend is unchanged.
* **Out of scope.** Different grids (`scale`, plane-to-volume), stitching and
  segmentation.

## Routine QC

`starfinder.evaluation.registration` gains `registration_qc` and
`evaluate_displacement_field`; both return the existing evaluation result type
with values, units, counts, config and reasons. `FOV` runs `registration_qc`
for every successful step and for the round's whole chain, on signals only.

| Element | Definition |
| --- | --- |
| Valid overlap | The reference voxels whose pull point under the evaluated transform lies inside the closed box `[0, n−1]` of the moving grid on every axis. |
| Coverage | The valid-overlap fraction of all reference voxels. |
| NCC | Pearson correlation in float64 over the valid overlap. **Matched domains:** "before" compares the reference signal with the moving signal at the start of the step and "after" with the resampled signal, both over the same valid overlap. Undefined (`None`, with a reason) when fewer than two voxels are valid or either side is constant there. `normalized_cross_correlation` gains an optional Boolean `mask` keyword; without it the result is unchanged. |
| Z-maximum-projection SSIM | scikit-image SSIM with a uniform 7×7 window on the Z maximum projections of the signals, averaged over the valid columns (YX positions valid in every plane) eroded by 3 pixels, the same columns before and after. `data_range` is the maximum minus the minimum of the reference projection over those columns, recorded. Undefined when the range is 0, the eroded domain is empty, or Y or X is smaller than 7; for Z=1 the projection is the plane. `structural_similarity` gains the same optional `mask`. |
| Overlays | Not computed in `FOV`: `registration_qc` returns the three maximum projections (reference, before, after) as arrays only when `RegistrationQcConfig.projections` is `True`, for callers that write overlays. |
| Transform summary | Translation: the correction. Affine: `A`, `b`, `det A`, the largest singular value of `A − I`, and the rotation angle for rigid. B-spline and dense: median, 95th percentile and maximum of `|u|` in voxels (and in physical units when spacing is known), and the fraction of voxels with `det(I + ∇u) ≤ 0` (folds), by central differences. |
| Optimizer diagnostics | From `RegistrationDiagnostics`: `converged`, `iterations_completed` (per level), final metric value, stop condition, and for demons the elapsed iterations and final RMS change. Unknown values stay `None`. |
| Truth metrics | Only when the caller supplies a truth pull field (for example `forward_displacement` of a synthetic pair): `evaluate_displacement_field(estimated, truth, *, mask, spacing_zyx=None)` gives the median, 95th percentile and maximum of `|u − u*|` over the valid overlap, in voxels and, with spacing, in physical units. Never computed from images alone. |
| Rejection | `RegistrationQcConfig(min_coverage=None, min_ncc_gain=None, max_fold_fraction=None, max_translation_voxels=None, projections=False)`. Every criterion is `None` by default, so nothing is rejected unless the recipe sets it. A step that fails a configured criterion raises `RegistrationRejectedError`, a subclass of `RegistrationEstimationError`, naming the criterion, the value and the bound; recovery may allow it. |

## Failure and recovery records

`FOV.registration_attempts[round]` stays an ordered list of dicts and keeps its
current keys. Each **estimation** entry has:

* `record: "estimation"`, `step` (index in `recipe.steps`), `attempt` (0 for
  the step's config, 1… for recovery alternatives);
* `requested_method` (the step's config) and `actual_method` (the config that
  ran), as today, plus `fallback` (`true` when `attempt > 0`);
* `backend` and `backend_versions` (for example
  `{"itk-elastix": "0.25.4", "itk": "5.4.7"}`), the backend that actually ran;
* `config`, `outcome` (`estimating`, `failed`, `rejected` or `succeeded`),
  `failure` (`type`, `message`, and for `rejected` the criterion) and, on
  success, `qc`;
* the reference (round name, or external label and SHA-256).

After the last step the round gets one **application** entry:
`record: "application"`, `outcome` (`succeeded` or `application_failed`),
`application_config` and `failure`. Application failures never recover, as
today. E01 (W-94) reads the executed backend per round from the succeeded
estimation entries and sees every fallback through `fallback` and
`actual_method`.

## Workflow configuration

### Legacy mapping

The shared MATLAB keys (`global_registration`, `local_registration` and their
existing fields) keep their names and MATLAB meaning. The adapter
(`dataset/workflow.py`) translates them to one recipe:

| Legacy key | Python meaning after §2.6 |
| --- | --- |
| `global_registration.run` / `local_registration.run` | One global step, then one local step, when enabled; neither enabled → `registration=None`. |
| `method` | Global default `translation`, local default `demons` (unchanged). Names are `REGISTRATION_METHODS` names plus the adapter aliases `diffeomorphic`, `symmetric` and `fast_symmetric` (→ `demons` with `variant`). A method whose step kind does not match its block is rejected. |
| `ref_img`, `mov_img` = `merged-image` (or `merged`) | `RegistrationSignalConfig(mode="max")`. **Changed:** Python `merged` meant the sum; it now means the maximum, as in MATLAB. |
| `ref_img`, `mov_img` = `single-channel` | `mode="channel"` with `reference_channel = ref_channel` (zero-based, Python only). |
| `ref_img` ≠ `mov_img` within a block | Rejected with `ValueError` (open choice in the worker notes). |
| Local `ref_img`, `mov_img` | **Changed default:** `merged-image` (maximum), as the MATLAB workflow uses; was `single-channel`. The schema declares both keys for the local block as Python-only keys. |
| Different signals in the two blocks | The first block's signal becomes the recipe signal and the other step gets a per-step `signal`. |
| `ref_channel` | Unchanged (Python zero-based index). |
| `boundary_mode` | The recipe's `warp.boundary_mode`; different values in the two blocks are rejected, since there is one final resampling. |
| `recovery` | `RecoveryConfig` of that step; alternatives must have the block's step kind. |
| `ref_round` | Must equal the dataset reference round (unchanged). |
| Method-specific MATLAB names (`detection_threshold`, `match_distance`, `tps_smoothing`, `grid_spacing`, `beta`, `lmbda`, `cpd_w`, `candidate_radius`, `k_neighbors`) and the CPD adapter defaults | Unchanged. |
| `local_registration.method` enum in the schema | Widened to the registered names plus the legacy aliases, kept equal by a default-tier test (W-240 choice 4). |

### Python-only key

A Python-only `registration` key in a Python rule's parameters declares a
recipe explicitly. It is rejected together with an enabled
`global_registration` or `local_registration`. Its `steps` entries name a
method and give that config's init fields (YAML lists become tuples), as the
preprocessing key does:

```yaml
registration:
  signal: {mode: max}                # max | sum | channel (+ reference_channel, moving_channel)
  warp: {boundary_mode: constant}    # WarpConfig fields; omitted: derived default
  qc: {min_coverage: null}           # RegistrationQcConfig fields
  steps:
    - method: translation
    - method: affine
    - method: bspline
      recovery: {allowed_errors: [RegistrationEstimationError], alternatives: [{method: demons}]}
```

The schema gains one definition per registered method, kept equal to
`REGISTRATION_METHODS` by a default-tier test, as the preprocessing definitions
are kept equal to the preprocessing registry.

## `docs/migration.md` entries

The implementation adds these entries:

1. **Registration recipe.** `PipelineConfig.registration: tuple[RegistrationStep, ...]`
   → `RegistrationRecipe(steps=(...))`; `RegistrationStep(config, reference_image,
   moving_image, reference_channel, recovery, warp)` → `RegistrationStep(config,
   recovery, signal)`, with the signal on `RegistrationSignalConfig` and the
   warp on the recipe. No aliases.
2. **Intentional change: signal.** `merged` / `merged-image` is the channel
   maximum, not the sum; the workflow's local default is `merged-image`, not
   `single-channel`. Use `mode="sum"` for the previous Python behavior.
3. **Intentional change: one resampling.** A multi-step recipe resamples each
   image once from its pre-registration array; results differ from the
   per-step resampling at the boundary and by integer rounding.
4. **Registered checkpoints version 2.** New kinds `affine` and `bspline`;
   per-round `application`; version 1 still loads, as `sequential`.
5. **New methods and Z=1.** `rigid`, `affine`, `bspline` (optional extra
   `registration-elastix`); demons accepts Z=1 as 2D; `REGISTRATION_METHODS`
   and exact-type lookup; `RegistrationRejectedError`.
6. **Evaluation.** `registration_qc`, `evaluate_displacement_field` and the
   optional `mask` of `normalized_cross_correlation` and `structural_similarity`.
7. **Other rounds.** `FOV.register_rounds` and `ExternalReference`; the Python
   workflow rule for morphology rounds.
8. **Preprocessing names** (registry move 1): `STEPS` → `PREPROCESSING_METHODS`,
   `StepSpec` → `PreprocessingSpec`, `RecipeStep` → `PreprocessingStep`.

## Tests the implementation changes

* `test/test_registration_golden.py`: task group 4 makes two named edits and
  no other. It replaces the body of `registration_config` with a builder of
  `RegistrationRecipe(steps=..., signal=RegistrationSignalConfig(mode="sum" if
  signal == "merged" else "channel", ...))`. It also replaces the one value
  `PINNED_RUNS[("translation", "demons")]["images"]["round2"]`, with a comment
  that names Jiahao's approval. The input and translation-only digests stay.
  The translation → demons field digest stays, because step 2 still sees the
  integer-shifted sum signal. The translation → demons image digest changes
  with the one final resampling. The reviewed change must show that the new
  image equals the pinned one within one intensity unit on every voxel whose
  pull point lies at least one voxel inside the moving grid.
* `test_declared_local_2d_rejection` (`test/test_registration_contract.py:123-126`)
  asserts that demons, TPS and CPD reject Z=1. Task group 3 makes one named
  edit: it removes `DemonsConfig()` from that parametrization and adds a test
  that demons accepts Z=1 as 2D. TPS and CPD keep the rejection unchanged.
* `test_two_registration_steps_resample_every_snapshot_in_the_same_sequence`
  (`test/test_recipe_sources.py:182`, W-232) is replaced by a test that each
  snapshot equals its pre-registration array resampled once by the round's
  chain; its record assertion keeps the per-image lists of results.
* The tests that construct `PipelineConfig(registration=...)` or read the field
  change their construction only: `test_checkpoints.py`, `test_e2e.py`,
  `test_recipe_sources.py`, `test_summaries.py`, `test_projection_views.py`,
  `test_benchmark_recipes.py` and `test_coordination_contract.py`
  ({doc}`method-registry`, move 3), plus `conftest.py`, `test_fov.py` and
  `test_registration_contract.py`, which construct `RegistrationStep` with the
  removed fields.

## Exclusions

No unequal-grid conversion, plane-to-volume registration, subpixel translation
estimator, stitching (§2.10), TPS or CPD performance study, or new default
beyond the agreed translation-only default. Method comparisons belong to E01
(W-94).
