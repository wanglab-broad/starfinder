# Dataset and FOV coordination

`Dataset` describes a sample's paths, rounds, channels and readout reference.
`FOV` holds the images and results for one field of view. Its processing methods
call the public numerical functions used by direct callers.

Use `PipelineConfig` to choose processing and `ExecutionConfig` to choose image
residency. The {doc}`coordination reference <coordination-reference>` lists their
options; {doc}`checkpoints` shows how to save and continue work.

## Quick start

This complete example makes one small synthetic FOV and runs detection, extraction,
decoding and filtering. Run it in a fresh working directory.

```python
from pathlib import Path
from dataclasses import replace
from starfinder.synthetic import (
    benchmark_scene_preset, generate_amplicon_dataset, save_amplicon_dataset,
)
from starfinder.dataset import Dataset, RoundState, PipelineConfig, ExecutionConfig
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
fov = dataset.fov("FOV_001").run(pipeline, execution=ExecutionConfig("streaming"))
print(fov)
```

## What it does

1. Load and rotate the configured images, then apply the preprocessing recipe.
2. Run the registration recipe and post-registration preprocessing steps.
3. Find spots, extract intensities, decode, score, deduplicate and filter as configured.
4. Keep the stage results on the FOV for inspection and export.

### Channels

`channel_order` describes sequencing colors. Use `other_channel_order` for a
morphology round's own channels and `reference_stains` for stains acquired beside
the reference round's sequencing colors. `ChannelInfo` adds a content name and
wavelength to a channel pattern. `dataset.channel_info(round)` inspects this
information; `dataset.channel_index(round, key)` finds a channel by index, pattern
or name.

### Sequencing rounds and other rounds

`FOV.run` processes the reference round and sequencing rounds.
`FOV.prepare_morphology` prepares reference stains and other rounds separately,
using `MorphologyConfig`. Use it before a segmentation plan that reads these images.
Prepared images can be saved and reloaded; see {doc}`checkpoints`.

### Registration recipe

Set `PipelineConfig.registration` to a `RegistrationRecipe`. Choose the signal and
steps with the {doc}`registration guide <registration-contract>` and its
{doc}`reference <registration-algorithms>`.
`FOV.register` also provides a separate call for loaded morphology rounds, with
`rounds` and an optional `ExternalReference`.

## What you get

(inspecting-results)=
### Summaries and results by stage

`print(dataset)` and `print(fov)` summarize labels and available results.
`fov.results` is a read-only mapping of completed processing stages;
`fov.results["decoding"]` gives `fov.decoding_result` after decoding.
Images are in `fov.images`, metadata in `fov.metadata`, and saved preprocessing
snapshots in `fov.snapshots`.

`fov.export_spots()` writes `fovs/<fov_id>/molecules.csv` under the output root;
`fov.export_spots("reads")` writes the read table. See the
{doc}`array and coordinate reference <api/contracts>` for columns.
With `checkpoints`, `FOV.run` also writes stage files and `run.json`.

## Main options

* `load`: choose image loading.
* `preprocessing`: choose the preprocessing recipe and snapshots.
* `registration`: choose the registration recipe.
* `spot_finding`, `extraction`, `decoding`, `scoring`, `deduplication`, `filtering`:
  choose the readout stages.
* `ExecutionConfig.mode`: choose batch or streaming image residency.
* `ExecutionConfig.retain_images`: keep processed moving images in streaming mode.

## Limits

`FOV.run` does not segment cells, assign molecules to cells or process sample-level
stages. Use the separate {doc}`segmentation <segmentation-contract>`,
{doc}`assignment <assignment-contract>` and {doc}`sample-stage guides <wiki/pipeline-stages>`.
Streaming releases moving images after use; retained results still consume memory.

### Recovery and output boundaries

For an expected registration estimation failure, a step can carry an explicit
`RecoveryConfig`. Inspect `fov.registration_attempts` to see what ran and why an
attempt failed. The
[recovery specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/registration.md#failure-and-recovery-records)
gives the supported failures and alternatives.

## How it is checked

Small synthetic cases compare batch and streaming results and exercise round roles,
channel lookup, registration recipes and saved results. These checks establish
software behavior; accuracy on experimental images needs separate evidence.

[Full coordination specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/coordination.md).
