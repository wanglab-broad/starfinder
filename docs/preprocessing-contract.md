# Preprocessing

Preprocessing corrects each round's image before the later stages: it removes
background and normalizes intensities, so that rounds and channels can be compared. A
`PreprocessingRecipe` in `PipelineConfig.preprocessing` lists the steps to run, in
order; `None`, the default, means no preprocessing. Inside `FOV.run`, each round goes
through the recipe's steps, then {doc}`registration <registration-contract>` when
`PipelineConfig.registration` holds a recipe and the round is a moving round, then the
recipe's `post_registration` steps. The stages after them, such as spot finding, run
when they are set.

The {doc}`preprocessing reference <preprocessing-algorithms>` lists every method,
option and record; the {doc}`preprocessing API <api/preprocessing>` documents the
classes.

## Quick start

```python
from dataclasses import replace

from starfinder.dataset import CheckpointConfig, Dataset, PipelineConfig, RoundState
from starfinder.io import ImageLoadConfig
from starfinder.preprocessing import (PercentileNormalizationConfig, PreprocessingRecipe, PreprocessingStep,
    ScalarBackgroundConfig)
from starfinder.synthetic import benchmark_scene_preset, generate_amplicon_dataset, save_amplicon_dataset

# A synthetic FOV of four rounds, written as a raw dataset.
codebook, scene = benchmark_scene_preset("tiny")
generated = generate_amplicon_dataset(codebook, replace(scene, seed=42), fov_ids=("FOV_001",), preset="tiny")
save_amplicon_dataset(generated, "input")
rounds, channels = list(generated.codebook.round_labels), list(generated.channel_labels)

dataset = Dataset(input_root="input", output_root="results", dataset_id="tiny", sample_id="synthetic",
                  output_id="demo", rounds=RoundState(sequencing_rounds=rounds, reference_round="round1"),
                  channel_order=channels)
recipe = PreprocessingRecipe(
    steps=(PreprocessingStep(ScalarBackgroundConfig(), save_as="bg_corrected"),
           PreprocessingStep(PercentileNormalizationConfig())),
    extraction_source="bg_corrected",
)
pipeline = PipelineConfig(load=ImageLoadConfig(channel_labels=tuple(channels)), preprocessing=recipe)
fov = dataset.fov("FOV_001")
fov.run(pipeline, checkpoints=CheckpointConfig(stages=()))

print(fov.images["round1"].dtype, list(fov.snapshots["round1"]))   # the detection image keeps the input dtype
for record in fov.preprocessing_record["rounds"]["round1"]:
    print(record["step"], record["fitted"])                          # the background, then low and high
```

## What it does

1. **Steps.** The steps run in the declared order on each round, one round at a time.
   The output of the last step is the *detection image*. With no steps, it is the
   loaded image.
2. **Snapshots.** A step with `save_as` keeps a named copy of its output. The recipe's
   `extraction_source` names the snapshot that intensity extraction reads, and
   `registration_source` the snapshot that registration signals are built from; both
   default to the detection image.
3. **Registration.** When `PipelineConfig.registration` holds a recipe, each moving
   round is registered to the reference round. The transform is applied once to the
   detection image and to every snapshot of the round, so they stay aligned. The
   reference round is not transformed.
4. **After registration.** The `post_registration` steps run on each round after
   registration, or after the steps when no registration is set.
   {py:class}`~starfinder.preprocessing.PreprocessingRecipe` says which methods may run
   there.

Every step's output is checked; the
[specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/preprocessing.md#enforcement-wrapper) ("Enforcement wrapper") gives the checks.

## Methods

```{method-table} starfinder.preprocessing.PREPROCESSING_METHODS
```

Two recipes are in use. Recipe 1 is `min_max_normalization` then `histogram_matching`,
optionally followed by `reconstruction` or `white_tophat`; the MATLAB keys translate
to its steps. Recipe 2, `scalar_background` or `background_3d` then
`percentile_normalization`, keeps the input dtype. "Choose a preprocessing recipe" in
{doc}`recipes` helps you choose between them.

## What you get

* **In memory**, per round of the FOV that is kept (every round in batch mode):
  * `fov.images[round]`: the detection image, after registration when it is set and
    after the `post_registration` steps;
  * `fov.snapshots[round][name]`: each snapshot, registered with the round when
    registration is set;
  * `fov.preprocessing_record`: the recipe, one record per round and step (its
    config, fitted values, diagnostics, and input and output dtypes), the transforms
    applied to each image of a round, and the path and SHA-256 of the
    supplied-statistics file.
* **On disk.** With any `checkpoints`, `run.json` records it under `preprocessing`.
  With the `registered` checkpoint stage, `FOV.run` writes each round after its
  preprocessing steps, any registration and its `post_registration` steps and, when the
  recipe names an `extraction_source`, that snapshot under
  `checkpoints/registered/<snapshot>/` in the FOV folder ({doc}`checkpoints`).

## Main options

* `PreprocessingRecipe.steps`: the ordered `PreprocessingStep`s, each holding the
  config of one method.
* `PreprocessingStep.save_as`: keeps the step's output as a named snapshot.
* `PreprocessingRecipe.extraction_source` and `registration_source`: the snapshots
  that extraction and registration read.
* `PreprocessingRecipe.post_registration`: the steps run after registration, or after
  the other steps when no registration is set.
* `PreprocessingRecipe.supplied_statistics` and the `fit` option of the methods that
  have one: sample-level statistics.

The reference gives each method's options and their defaults.

## Fitting modes

`scalar_background`, `percentile_normalization` and `histogram_matching` have a `fit`
option. With `fit="fov"`, `scalar_background` and `percentile_normalization` fit
their values on each round of each FOV, and
`histogram_matching` matches every round to one channel of the reference round of
the same FOV. With `fit="supplied"`, they are read from the recipe's
`supplied_statistics` file, which holds statistics merged over a set of FOVs of the
sample. Use it when FOVs differ strongly in content, such as near-empty fields at a
tissue edge.

The file is made by summary passes over the FOVs before the recipe is applied. A
summary pass runs the steps before a supplied step, records histograms of their
output on each FOV, merges them and fits that step's statistics; the specification
says how many passes a recipe with several supplied steps needs. The Snakemake
workflow does not run summary passes. This example, with one supplied step, runs the
summary pass and the application pass by hand on three synthetic FOVs; from
`src/python`, run
`uv run python ../../docs/examples/percentile_two_pass.py <new output directory>`:

```{literalinclude} examples/percentile_two_pass.py
:language: python
```

## Streaming

`FOV.run` processes the reference round first, then the other sequencing rounds in
their declared order. With `ExecutionConfig(mode="streaming")`, rounds are loaded one
at a time, and a moving round's image and snapshots are released once the round is
done unless `retain_images=True`. The [specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/preprocessing.md#streaming) ("Streaming")
says what is kept from the reference round.

## Workflow keys

On the Python backend, the `preprocessing` parameter of the five Python processing
rules holds the recipe. Each step names a method and gives that config's fields:

```yaml
backend: python
rules:
  rsf_single_fov:
    parameters:
      preprocessing:
        steps:
          - method: scalar_background
            percentile: 10.0
            save_as: bg_corrected
          - method: percentile_normalization
        extraction_source: bg_corrected
```

`post_registration` lists steps in the same form; `extraction_source`,
`registration_source` and `supplied_statistics` are the recipe's fields. Without the
key, no preprocessing runs. Under `backend: python`, the MATLAB preprocessing keys
raise an error; {doc}`migration` gives the step that replaces each.
The [workflow specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/workflow-configuration.md#explicit-preprocessing-recipe) describes the rules.

## Limits

* One linear sequence of steps: no branches. Snapshots are copies taken along it.
* Projection is not a step. The pipeline processes ZYX(C) volumes; a projected view
  is an output, such as the reference image.
* The Snakemake workflow does not make the supplied-statistics file.
* The XY methods, `reconstruction` and `white_tophat`, work on each Z plane
  separately.

## How it is checked

A golden test pins the outputs of recipe 1 (min–max, histogram matching,
reconstruction) and of each recipe 1 method on a fixed fixture. Tests check scalar
background, 3D background and percentile normalization against their formulas on
small arrays, and sample-level percentiles and histogram matching against the same
computation on the concatenated volumes. The synthetic evaluation design is in the
[algorithms specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/preprocessing-algorithms.md).

The full specification is in
[`design/specs/preprocessing.md`](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/preprocessing.md).

```{toctree}
:hidden:

Preprocessing reference <preprocessing-algorithms>
```
