# Registration

Registration aligns each moving round of a FOV with its reference round. It builds one
registration signal per round, estimates a transform in one or more steps, and then
resamples the round once onto the reference grid, so that spot finding and readout see
every round in the same coordinates. The reference round is never changed.

It runs inside `FOV.run`, after the preprocessing steps and before the preprocessing
recipe's `post_registration` steps and spot finding. A
`RegistrationRecipe` in `PipelineConfig.registration` says what to do; `None`, the
default, means no registration. The same recipe also registers other rounds, such as a
morphology round, to the reference round or to an external reference image. The
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

# A synthetic FOV of four rounds, each shifted by a known amount, written as a raw dataset.
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

A recipe of several steps adds a global or local method after the translation, and can
set QC criteria. Rigid, affine and B-spline need the `registration-elastix` extra when
they run:

```python
from starfinder.registration import AffineConfig, RegistrationQcConfig

recipe = RegistrationRecipe(
    (RegistrationStep(TranslationConfig()), RegistrationStep(AffineConfig())),
    qc=RegistrationQcConfig(min_coverage=0.8),
)
```

## What it does

1. **Signal.** Each round gives one float64 ZYX signal: by default the maximum over its
   channels, or their sum, or one named channel. It is built from the preprocessing
   recipe's `registration_source` snapshot, or from the detection image when there is
   none. Nothing is projected; a Z=1 round stays one plane.
2. **Steps.** A recipe is zero or more global steps (`translation`, `rigid`, `affine`)
   followed by at most one local step (`demons`, `bspline`, `tps`, `cpd`). Step k is
   estimated on the moving signal resampled through steps 1 to k−1.
3. **QC.** Every successful step, and then the round's whole chain, gets a routine QC
   record. Nothing is rejected unless the recipe sets a criterion.
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

Start with `translation`, which needs no extra and suits most data. Add `rigid` or
`affine` after it when rounds are also rotated, scaled or sheared, and a local step
when the misalignment varies across the field. Rigid, affine and B-spline are estimated
in physical units from the voxel spacing; on a Z=1 round, rigid, affine, B-spline and
demons estimate in 2D, while TPS and CPD need 3D.

## What you get

* **In memory**, per moving round of the FOV:
  * `fov.images[round]`: the registered image, in the input dtype by default;
  * `fov.registration_results[round]`: one `RegistrationResult` per step, with its
    transform and diagnostics;
  * `fov.registration_chains[round]`: the composed `TransformChain`; its
    `pull_field()` is the whole displacement field;
  * `fov.registration_attempts[round]`: one estimation entry per attempt (method,
    backend, outcome, failure and QC) and one application entry with the chain's QC;
  * `fov.registration_record`: the recipe summary and the resampling of each round.
* **On disk.** With the `registered` checkpoint stage, `FOV.run` writes the registered
  images, `transforms.json` and, for dense and B-spline transforms,
  `<round>_field.npz` under the FOV's `checkpoints/registered/` folder
  ({doc}`checkpoints`). `run.json` records each round's transforms and attempts.

## Main options

| Option | Default | What it sets |
| --- | --- | --- |
| `RegistrationRecipe.steps` | required | The ordered `RegistrationStep`s, each with a method config, optional `recovery` and optional per-step `signal` |
| `RegistrationRecipe.signal` | `RegistrationSignalConfig()`: `mode="max"` | `"max"`, `"sum"`, or `"channel"` with `reference_channel` and `moving_channel` (index or label) |
| `RegistrationRecipe.warp` | `None`: translation for a chain of translations, linear SciPy otherwise | The final resampling: `backend`, `boundary_mode` (`"constant"` or `"nearest"`), `fill_value`, `output_dtype` |
| `RegistrationRecipe.reference_round` | `None`: the dataset's reference round | Sequencing rounds: must equal the dataset's reference round when set. Other rounds: must equal the call's `reference` round, and must be `None` with an `ExternalReference` |
| `RegistrationRecipe.qc` | `RegistrationQcConfig()`: rejects nothing | `min_coverage`, `min_ncc_gain`, `max_fold_fraction`, `max_translation_voxels` |
| `RecoveryConfig` | no recovery | `allowed_errors` and ordered `alternatives` |

Each method's own options and defaults are in the reference.

## Routine QC

`registration_qc` (`starfinder.evaluation.registration`) measures, on the signals only:

* **coverage**: the fraction of reference voxels whose pull point falls inside the
  moving grid (the valid overlap);
* **NCC** before and after the step, over the same valid overlap, and their gain;
* **SSIM** before and after, on the Z maximum projections with a 7 × 7 window;
* a **transform summary**: the displacement of a translation; the matrix, its
  determinant and the rotation of an affine or rigid step; displacement statistics and
  the fraction of folded voxels of a B-spline or dense field;
* the optimizer's diagnostics, where the backend reports them.

A value that cannot be computed is `None`, with a reason. When a step fails a
criterion set in `RegistrationQcConfig`, it raises `RegistrationRejectedError`, which a
step's recovery can allow.

## Other rounds and external references

`FOV.register(recipe, rounds=[...])` registers named rounds. When they are sequencing
rounds, they are registered to the reference round as in `FOV.run`. When none of them
is, it takes the other-round route: each round is registered to `reference`, any
loaded round's name (default the recipe's `reference_round`, else the dataset's
reference round) or an `ExternalReference` holding a ZYX image on the same grid with a
label. Mixing both kinds raises `ValueError`.

On this route the recipe usually registers a shared stain, with
`RegistrationSignalConfig("channel", reference_channel=..., moving_channel=...)`
naming that stain in each round. The chain found on the stain is applied to every
channel of the round. With `checkpoints`, each registered round is saved under
`other_rounds/<round>/` in the FOV folder, and `FOV.load_registered_round(name)`
restores it.

`FOV.prepare_morphology(MorphologyConfig(rotation_degrees, recipe, rounds))` does the
whole morphology preparation: it loads and rotates the reference stain from the
reference round's folder, then loads, rotates and registers each other round. Without a
recipe it uses one translation step on the shared stain, as the MATLAB
`nuclei_registration` does.

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

The keys `signal`, `warp` and `qc` take the fields of their classes, and
`reference_round` must equal the dataset's reference round; YAML lists become tuples. The MATLAB keys `global_registration` and
`local_registration` raise under `backend: python`; on the MATLAB backend they keep
their meaning. The rule parameter `reference_image_view` (`{mode: max}`, or
`{mode: channel, channel: <index or label>}`) chooses the reference image the rule
writes. {doc}`workflow-configuration` describes the rules.

## Limits

* The reference and moving rounds must share one grid: the same shape, spacing,
  origin, direction and unit. There is no conversion between grids and no plane-to-volume
  registration.
* The translation estimator finds whole voxels only.
* At most one local step per recipe.
* Rigid, affine, B-spline and demons need at least 4 planes in 3D; on Z=1 they need
  16 × 16 (demons 4 × 4) in Y and X. TPS and CPD reject Z=1.
* Rigid, affine and B-spline draw random samples with a fixed seed; the tests check
  that they repeat exactly at one thread, not across thread counts.

## How it is checked

Known-answer synthetic tests check each method: a known shift, rotation, affine map
and deformation must be recovered within fixed bounds, the composition of steps is
checked against exact examples, and a saved checkpoint reloads and re-applies to the
same arrays. The checks are listed in the
[registration algorithms record](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/registration-algorithms.md).

The full specification is in
[`design/specs/registration.md`](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/registration.md).

```{toctree}
:hidden:

Registration reference <registration-algorithms>
```
