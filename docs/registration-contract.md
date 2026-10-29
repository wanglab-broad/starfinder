# Registration

Registration aligns moving rounds of a FOV with a reference image for spot finding
and readout. A `RegistrationRecipe` builds registration signals, estimates transforms
in steps and resamples the moving images through the resulting transform chain.

In `FOV.run`, registration sits between preprocessing and the recipe's
`post_registration` steps. Set `PipelineConfig.registration` to a recipe to enable
it. Use {py:meth}`~starfinder.dataset.FOV.register` for a separate registration call,
including other rounds and external references; its API documents round selection
and reference handling. The {doc}`registration reference <registration-algorithms>`
lists methods, options and saved files.

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

This recipe adds affine refinement and a QC criterion after translation:

```python
from starfinder.registration import AffineConfig, RegistrationQcConfig

recipe = RegistrationRecipe(
    (RegistrationStep(TranslationConfig()), RegistrationStep(AffineConfig())),
    qc=RegistrationQcConfig(min_coverage=0.8),
)
```

## What it does

1. **Signal.** Build a ZYX registration image from the round's channels.
2. **Steps.** Estimate transforms in the recipe's order, refining the alignment.
3. **QC.** Measure alignment quality and apply configured rejection criteria.
4. **Recovery.** Try configured alternatives after an allowed estimation failure.
5. **Resampling.** Apply the composed transform to the moving image and its saved
   preprocessing snapshots.

The [application specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/registration.md)
gives the signal, composition, recovery and resampling rules.

## Methods

```{method-table} starfinder.registration.REGISTRATION_METHODS
```

Start with `translation` for shifts. Consider `rigid` for rotation, `affine` for
scale or shear, and a local method for misalignment that varies across the field.
The generated table lists dimensions and optional extras; the
{doc}`reference <registration-algorithms>` links each method's API documentation.

## What you get

* `fov.images[round]` holds the registered image.
* `fov.registration_results[round]` holds step results with transforms and diagnostics.
* `fov.registration_chains[round]` holds the composed transform.
* `fov.registration_attempts[round]` records estimation and application attempts.
* `fov.registration_record` summarizes the recipe and application settings.

The `registered` checkpoint stage of `FOV.run` saves images and transform records;
`run.json` records the run when checkpoints are enabled. Other-round registration
can save prepared images for reuse. The {doc}`reference <registration-algorithms>`
lists the files, and {doc}`checkpoints` shows how to reload them.

## Main options

* `steps`: the methods to run.
* `signal`: the channels used to estimate alignment.
* `warp`: the final image resampling.
* `qc`: rejection criteria for alignment quality.
* A step's `recovery`: alternatives after an estimation failure.

{py:class}`~starfinder.registration.RegistrationRecipe` documents the recipe.
For other rounds, external references and morphology preparation, see
{py:meth}`~starfinder.dataset.FOV.register`,
{py:class}`~starfinder.registration.ExternalReference` and
{py:meth}`~starfinder.dataset.FOV.prepare_morphology`.
The {doc}`workflow reference <workflow-configuration>` lists workflow keys.

## Limits

Registration does not convert between image grids or establish biological accuracy.
Consult the [algorithms specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/registration-algorithms.md)
for geometry requirements and method limitations, and
{py:class}`~starfinder.registration.WarpConfig` for resampling choices.

### Troubleshooting

| Error | When you may see it | What to do |
| --- | --- | --- |
| `IncompatibleGeometryError` | Image grids or dimensions do not fit the method | Check image metadata and the method's geometry requirements |
| `RegistrationBackendUnavailableError` | An optional backend is missing | Install the extra shown in the methods table |
| `InsufficientLandmarksError` | Landmark estimation lacks usable matches | Inspect the registration signal and consider another method |
| `RegistrationRejectedError` | Alignment fails a configured quality criterion | Inspect the QC record before adjusting the recipe |

## How it is checked

Known-answer synthetic tests exercise shift, rotation and deformation recovery.
Separate checks cover transform composition, geometry handling, recovery and
checkpoint re-application. The
[algorithms record](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/registration-algorithms.md#engineering-validation-design-task-group-6)
describes the validation design.

The full specification is in
[`design/specs/registration.md`](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/registration.md).

```{toctree}
:hidden:

Registration reference <registration-algorithms>
```
