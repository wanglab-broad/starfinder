# Checkpoints and run records

Checkpoints let you rerun later processing stages from saved images, signals or reads.
`FOV.run` writes them when you pass `CheckpointConfig`, together with `run.json`,
a record of the processing. The {doc}`checkpoint reference <checkpoints-reference>`
lists the files, columns, record fields and generated configuration defaults.

## Quick start

### Turn checkpoints on

Run this example in a fresh working directory. It creates a small synthetic dataset
and saves the three checkpoint stages for one FOV.

```python
from pathlib import Path
from dataclasses import replace
from starfinder.synthetic import (
    benchmark_scene_preset, generate_amplicon_dataset, save_amplicon_dataset,
)
from starfinder.dataset import Dataset, RoundState, PipelineConfig, ExecutionConfig, CheckpointConfig
from starfinder.io import ImageLoadConfig
from starfinder.spot_finding import LocalMaximaConfig
from starfinder.registration import RegistrationRecipe, RegistrationStep, TranslationConfig
from starfinder.readout import NeighborhoodSumConfig, WtaDecoderConfig, ReadFilterConfig

codebook, scene_config = benchmark_scene_preset("tiny")
scene_config = replace(scene_config, shape_zyx=(4, 24, 24), count=3, seed=42)
scene = generate_amplicon_dataset(codebook, scene_config)
save_amplicon_dataset(scene, "input")
dataset = Dataset(
    Path("input"), Path("results"), "example", "sample", "run",
    RoundState(list(codebook.round_labels), reference_round=codebook.round_labels[0]),
    channel_order=codebook.channel_labels,
)
dataset.load_codebook("input/codebook.csv")
pipeline = PipelineConfig(
    load=ImageLoadConfig(channel_labels=dataset.channel_order),
    registration=RegistrationRecipe((RegistrationStep(TranslationConfig()),)),
    spot_finding=LocalMaximaConfig("adaptive", 0.2),
    extraction=NeighborhoodSumConfig(),
    decoding=WtaDecoderConfig(),
    filtering=ReadFilterConfig(),
)
fov = dataset.fov("FOV_001").run(
    pipeline, execution=ExecutionConfig("streaming"), checkpoints=CheckpointConfig(),
)
print(fov)
```

## What it does

1. Record the run's configuration, input identities and processing steps.
2. Save the configured stages as processing reaches them.
3. Reload a stage into a fresh FOV and run the later operations you choose.

## What you get

### Layout

The default FOV folder is `<output_root>/fovs/<fov_id>/`. `run.json` is in that
folder and the three stage records are in `checkpoints/`.
See the {doc}`layout table <checkpoints-reference>` for payload files.

#### registered

Processed images let you repeat detection and extraction. The
[registered reference](checkpoints-reference.md#registered) lists the saved transforms
and image files.

#### candidates

Spots and extracted signals let you repeat decoding. The
[candidates reference](checkpoints-reference.md#candidates) lists signal and background
columns.

#### pre_qc

Decoded reads let you repeat filtering. The
{ref}`pre-QC reference <checkpoint-pre-qc>` lists the table and header.

### Prepared morphology images

`FOV.prepare_morphology(..., checkpoints=...)` saves prepared images separately in
`other_rounds/`. `FOV.load_registered_round(name)` loads a saved image for later
segmentation. See {doc}`coordination` for round roles.

### The reference image

`FOV.save_reference_image()` saves the reference image; `FOV.load_reference_image()`
loads it for later stages. The dataset's `reference_image` or
`reference_image_import` setting names the file.

### Segmentation runs

`FOV.segment(..., checkpoints=...)` saves segmentation runs in `segmentation/`.
`FOV.load_segmentation(name)` loads one; the {doc}`segmentation reference <segmentation-algorithms>`
lists its records.

### Assignments

`FOV.assign_molecules(..., checkpoints=...)` saves assignments in `assignment/`.
`FOV.load_assignment(name)` loads one; the {doc}`assignment reference <assignment-algorithms>`
lists the tables.

### Table formats

CSV is the default; Parquet uses the `checkpoint` extra. See the
[format reference](checkpoints-reference.md#table-formats).

### run.json

Read this record to inspect the run's status, settings, steps and counts. Writers
such as `FOV.export_spots` add their output paths and hashes after processing.
The {ref}`field reference <checkpoint-run-record>` lists the keys.

## Main options

* `stages`: choose the checkpoint stages to save.
* `directory`: choose another parent for FOV folders.
* `table_format`: choose the table format.
* `hash_inputs`: record input image hashes.
* `overwrite`: replace the run's earlier checkpoint files.

### Reload and continue

After the quick start, run this in the same process. It loads candidates into a fresh
FOV and decodes and filters again, without loading the images.

```python
later = dataset.fov("FOV_001").load_checkpoint("candidates")
later.run(PipelineConfig(decoding=WtaDecoderConfig(), filtering=ReadFilterConfig()))
print(later.results["filtering"])
```

`FOV.save_checkpoint(stage)` saves a stage from resident results. The standalone
`dataset.load_checkpoint(path, stage)` reads a checkpoint folder into a dictionary
of restored attributes. See the {doc}`dataset API <api/dataset>` for signatures.

## Limits

Reloading is an explicit call; these APIs do not schedule work. Prepared images,
segmentation runs and assignments are saved by their own entries rather than
`FOV.run`. See the
[checkpoint specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/checkpoints.md)
for integrity checks, overwrite rules and older record versions.

## How it is checked

Small synthetic cases save and reload images, signals and reads, then compare later
processing with the original results. File-integrity checks also exercise changed
payloads and portable run records.

[Full checkpoint specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/checkpoints.md).
