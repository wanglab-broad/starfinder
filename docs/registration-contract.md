# Registration

Registration aligns each moving round of a FOV with its reference round. It builds a
registration signal for each round, estimates a transform in one or more steps, and then
resamples the round once onto the reference grid for spot finding and readout. The
reference round is never changed.

It runs inside `FOV.run`, after the preprocessing steps and before the preprocessing
recipe's `post_registration` steps and spot finding. A
`RegistrationRecipe` in `PipelineConfig.registration` says what to do; `None`, the
default, means no registration. A recipe also registers other rounds, such as a
morphology round, to a reference round or an external reference image. The
{doc}`registration reference <registration-algorithms>` lists every method, option,
transform and record; the {doc}`registration API <api/registration>` documents the
classes.

## Quick start

```python
from dataclasses import replace

from starfinder.dataset import CheckpointConfig, Dataset, PipelineConfig, RoundState
from starfinder.io import ImageLoadConfig
from starfinder.registration import RegistrationRecipe, RegistrationStep, TranslationConfig
from starfinder.synthetic import benchmark_scene_preset, generate_amplicon_dataset, save_amplicon_dataset

# A synthetic FOV of four rounds, the moving rounds shifted by known amounts, written as a raw dataset.
codebook, scene = benchmark_scene_preset("tiny")
generated = generate_amplicon_dataset(codebook, replace(scene, seed=42), fov_ids=("FOV_001",), preset="tiny")
save_amplicon_dataset(generated, "input")
rounds, channels = list(generated.codebook.round_labels), list(generated.channel_labels)

dataset = Dataset(input_root="input", output_root="results", dataset_id="tiny", sample_id="synthetic",
                  output_id="demo", rounds=RoundState(sequencing_rounds=rounds, reference_round="round1"),
                  channel_order=channels)
pipeline = PipelineConfig(
    load=ImageLoadConfig(channel_labels=tuple(channels)),
    registration=RegistrationRecipe((RegistrationStep(TranslationConfig()),)),
)
fov = dataset.fov("FOV_001")
fov.run(pipeline, checkpoints=CheckpointConfig(stages=()))

for name, results in fov.registration_results.items():
    print(name, results[0].transform.displacement_zyx)   # the detected shift, ZYX voxels
```

A recipe can add more steps after the translation and set QC criteria. Affine, like
rigid and B-spline, needs the `registration-elastix` extra when it runs:

```python
from starfinder.registration import AffineConfig, RegistrationQcConfig

recipe = RegistrationRecipe(
    (RegistrationStep(TranslationConfig()), RegistrationStep(AffineConfig())),
    qc=RegistrationQcConfig(min_coverage=0.8),
)
```

## What it does

1. **Signal.** Each round gives a float64 ZYX signal: by default the maximum over its
   channels, or their sum, or one chosen channel; a step may set its own. In `FOV.run`
   it is built from the preprocessing recipe's `registration_source` snapshot, or from
   the detection image when there is none; on the other-round route, from the round's
   image. A Z=1 round stays one plane.
2. **Steps.** A recipe has at least one step: zero or more global steps (`translation`,
   `rigid`, `affine`) followed by at most one local step (`demons`, `bspline`, `tps`,
   `cpd`). Step k is estimated on the moving signal resampled through steps 1 to k−1.
3. **QC.** Every step estimate, and then the round's whole chain once it is resampled,
   gets a routine QC record. Nothing is rejected unless the recipe sets a criterion.
4. **Recovery.** A step may name alternatives of the same kind to try when it fails
   with an error it allows. Nothing else falls back.
5. **One resampling.** The step transforms compose into one `TransformChain`. The
   moving round's image, every channel, and every snapshot preprocessing keeps for it
   are resampled once through that chain, from their pre-registration arrays.

Every transform is a pull map from reference to moving voxel coordinates: the
registered image at reference index `p` is the moving image sampled at `T(p)`.

## Methods

```{method-table} starfinder.registration.REGISTRATION_METHODS
```

Start with `translation`, which needs no extra and corrects shifts only. Add `rigid`
after it when rounds are also rotated, or `affine` when they are also scaled or
sheared, and a local step when the misalignment varies across the field. Rigid, affine
and B-spline are estimated in physical units from the voxel spacing; the Dimensions
column shows which methods take a Z=1 round.

## What you get

* **In memory**, per moving round of the FOV:
  * `fov.images[round]`: the registered image, in the input dtype by default;
  * `fov.registration_results[round]`: one `RegistrationResult` per step, with its
    transform and diagnostics;
  * `fov.registration_chains[round]`: the composed `TransformChain`; its
    `pull_field()` is the whole displacement field;
  * `fov.registration_attempts[round]`: one estimation entry per attempt (method,
    backend, outcome, failure and QC) and, once every step has succeeded, one
    application entry, with the chain's QC when the resampling succeeds;
  * `fov.registration_record`: the recipe summary and the resampling of each round.
* **On disk.** With the `registered` checkpoint stage, `FOV.run` writes the registered
  images, `transforms.json` and, for dense and B-spline transforms,
  `<round>_field.npz` under the FOV's `checkpoints/registered/` folder
  ({doc}`checkpoints`). With any `checkpoints`, `run.json` records the transforms of
  the rounds `FOV.run` registered, and the attempts.

## Main options

| Option | Default | What it sets |
| --- | --- | --- |
| `RegistrationRecipe.steps` | required | The ordered `RegistrationStep`s, each with a method config, optional `recovery` and optional per-step `signal` |
| `RegistrationRecipe.signal` | `RegistrationSignalConfig()`: `mode="max"` | `"max"`, `"sum"`, or `"channel"` with `reference_channel` and `moving_channel` (index or label) |
| `RegistrationRecipe.warp` | `None`: translation for a chain of translations, linear SciPy otherwise | The final resampling: `backend`, `boundary_mode` (`"constant"`, or `"nearest"` with `scipy` or `simpleitk`), `fill_value` (0 with `translation`), `output_dtype` |
| `RegistrationRecipe.reference_round` | `None`: the dataset's reference round | Sequencing rounds: must equal the dataset's reference round when set. Other rounds: must equal the call's `reference` round, and must be `None` with an `ExternalReference` |
| `RegistrationRecipe.qc` | `RegistrationQcConfig()`: rejects nothing | `min_coverage`, `min_ncc_gain`, `max_fold_fraction`, `max_translation_voxels` |
| `RecoveryConfig` | no recovery | `allowed_errors` and ordered `alternatives` |

The reference gives each method's options.

## Routine QC

`registration_qc` (`starfinder.evaluation.registration`) measures, on the signals only:

* **coverage**: the fraction of reference voxels whose pull point falls inside the
  moving grid (the valid overlap);
* **NCC** before and after the step (or the whole chain), over the same valid overlap,
  and their gain;
* **SSIM** before and after, on the Z maximum projections with a 7 × 7 window;
* a **transform summary**: the displacement of a translation step; the matrix, offset
  and determinant of an affine or rigid step, with the largest singular value of
  `A − I`, and for a rigid step only, the rotation angle; displacement statistics and
  the fraction of folded voxels of a B-spline or dense step, and of a whole chain;
* for a step, the optimizer's diagnostics, where the backend reports them.

An NCC or SSIM value that cannot be computed is `None`, with a reason. A step that
fails a criterion set in `RegistrationQcConfig` raises `RegistrationRejectedError`,
which its recovery can allow; the reference says which criteria apply to which steps.

## Other rounds and external references

`FOV.register(recipe, rounds=[...])` registers named rounds. When they are sequencing
rounds, they are registered to the reference round as in `FOV.run`, from their images
or the snapshot `source` names. When none of them
is, it takes the other-round route: each round is registered to `reference`, any
loaded round's name (default the recipe's `reference_round`, else the dataset's
reference round) or an `ExternalReference` holding a ZYX image on the same grid with a
label. Mixing both kinds raises `ValueError`.

The recipe usually registers a shared stain, with
`RegistrationSignalConfig("channel", reference_channel=..., moving_channel=...)`
naming it in each round; the chain found on the stain is applied to every channel. With `checkpoints`, each registered round is saved under
`other_rounds/<round>/` in the FOV folder, and `FOV.load_registered_round(name)`
restores it.

`FOV.prepare_morphology(MorphologyConfig(rotation_degrees, recipe, rounds))` prepares
the morphology images: when the dataset names reference stains, it loads them
from the reference round's folder and rotates them by `rotation_degrees`; then it
loads, rotates and registers each round of `rounds` (default: every other round).
Without a recipe it uses one translation step on the dataset's single reference stain
and the round's one channel of that name, as the MATLAB `nuclei_registration` does.

## Workflow keys

On the Python backend, the `registration` parameter of the five Python processing rules
holds the recipe. Each step names a method and gives that config's fields:

```yaml
backend: python
rules:
  rsf_single_fov:
    parameters:
      registration:
        signal: {mode: max}
        qc: {min_coverage: 0.8}
        steps:
          - method: translation
          - method: demons
            iterations: [100, 50, 25]
            recovery: {allowed_errors: [RegistrationEstimationError], alternatives: [{method: tps}]}
```

`signal`, `warp` and `qc` take the fields of their classes, and `reference_round`, when
given, must equal the dataset's reference round; YAML lists become tuples. The MATLAB keys
`global_registration` and `local_registration` raise under `backend: python`. The Python-only rule parameter `reference_image_view` (`{mode: max}`, or
`{mode: channel, channel: <index or label>}`) chooses the reference image the rule
writes; the resident subtile rules write none and refuse it.
{doc}`workflow-configuration` describes the rules.

## Limits

* The reference and moving rounds must share one grid: the same shape, spacing,
  origin, direction and unit; nothing converts between grids.
* The translation estimator finds whole voxels.
* At most one local step per recipe.
* Rigid, affine and B-spline need at least 16 × 16 in Y and X; demons needs at least
  4 × 4. In 3D, all four need at least 4 planes. TPS and
  CPD reject Z=1 and need at least 2 in every axis.
* Rigid, affine and B-spline sample randomly with `random_seed` (1 by default); the
  tests check exact repeats at one thread only.

## How it is checked

Known-answer synthetic tests check that translation, rigid, affine, B-spline and
demons (on one plane) recover a known shift, rotation, affine map or deformation within
fixed bounds. All seven methods are checked against the Z=1 and small-Z rules, the
composition against exact examples, and checkpoints against re-application. They are listed in the
[registration algorithms record](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/registration-algorithms.md).

The full specification is in
[`design/specs/registration.md`](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/registration.md).

```{toctree}
:hidden:

Registration reference <registration-algorithms>
```
