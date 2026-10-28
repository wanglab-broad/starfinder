# Segmentation

Segmentation makes label images of nuclei or cells for a FOV: ZYX images in which 0 is
background and each positive value is one object. A FOV's label images lie on its
reference grid, the grid of the reference round's image after `FOV.run`, or on that
grid's Z projection, so that {doc}`assignment <assignment-contract>` can place each
molecule in a cell. A label image comes from a method (StarDist, Cellpose or a seeded
watershed) or from a mask made with another tool.

Segmentation is its own call, `FOV.segment`, made after `FOV.run`. The
{doc}`segmentation reference <segmentation-algorithms>` lists the methods, options,
records and files; the {doc}`segmentation API <api/segmentation>` documents the classes.

## Quick start

```python
from dataclasses import replace

import numpy as np

from starfinder.dataset import CheckpointConfig, Dataset, PipelineConfig, RoundState
from starfinder.image import ImageMetadata
from starfinder.io import ImageLoadConfig, save_volume
from starfinder.segmentation import (InputChannel, LabelImportConfig, SeededWatershedConfig, SegmentationPlan,
                                     SegmentationRun)
from starfinder.synthetic import benchmark_scene_preset, generate_amplicon_dataset, save_amplicon_dataset

# A synthetic FOV, loaded with a voxel spacing in µm, which the watershed uses.
codebook, scene = benchmark_scene_preset("tiny")
generated = generate_amplicon_dataset(codebook, replace(scene, seed=42), fov_ids=("FOV_001",), preset="tiny")
save_amplicon_dataset(generated, "input")
rounds, channels = list(generated.codebook.round_labels), list(generated.channel_labels)
dataset = Dataset(input_root="input", output_root="results", dataset_id="tiny", sample_id="synthetic",
                  output_id="demo", rounds=RoundState(sequencing_rounds=rounds, reference_round="round1"),
                  channel_order=channels)
load = ImageLoadConfig(channel_labels=tuple(channels),
                       metadata=ImageMetadata("tiny", spacing_zyx=(0.5, 0.1, 0.1)))
fov = dataset.fov("FOV_001")
fov.run(PipelineConfig(load=load))

# A nucleus mask made by another tool, here two boxes, saved on the FOV's reference grid.
grid = fov.reference_grid()
nuclei = np.zeros(grid.shape_zyx, dtype=np.uint16)
nuclei[:, 30:45, 30:45], nuclei[:, 80:95, 70:85] = 1, 2
save_volume(nuclei, "nuclei.tif", metadata=grid.metadata)

# Import the nuclei, then grow one cell from each nucleus over the amplicon signal.
plan = SegmentationPlan((
    SegmentationRun("nucleus", "nucleus", (), LabelImportConfig("nuclei.tif", "nucleus")),
    SegmentationRun("cell", "cell", (InputChannel("amplicon", reference_merged=True),),
                    SeededWatershedConfig(), seeds="nucleus"),
))
fov.segment(plan, checkpoints=CheckpointConfig())

cells = fov.segmentation_results["cell"]
print(cells.labels.shape, cells.labels.dtype, cells.n_labels)   # ZYX uint32 labels, 0 is background
print(cells.record["outcome"], cells.record["seeds"]["run"])    # "ok", and the seed run
```

The run writes one folder per run, `results/fovs/FOV_001/segmentation/nucleus/` and
`.../cell/`. With a nucleus model installed, the nucleus run names a method such as
`StarDistConfig` with a `nuclear` input instead ([Models](#models)).

## What it does

`FOV.segment` takes the runs of the plan in order. For each run it:

1. **Grid.** Takes the FOV's reference grid (`FOV.reference_grid()`), or its Z
   projection when the run sets `projection`.
2. **Input.** For a method, builds the input image from the FOV's images: one channel
   per `InputChannel`, each with its role.
3. **Labels.** Runs the method, with the labels of the run that `seeds` names as seeds;
   or, for an import, reads the mask file onto the grid.
4. **Operations.** Applies the run's label operations, such as `expand_labels`, in order.
5. **Results.** Keys each result by its run name in `fov.segmentation_results`, and,
   with `checkpoints`, writes each run's files.

## Methods

```{method-table} starfinder.segmentation.SEGMENTATION_METHODS
```

Choose by your stains ([Method input](#method-input)). `stardist` and `cellpose` are
learned methods ([Models](#models)); `seeded_watershed` grows cells from the nuclei of an
earlier run. This page does not recommend a method or a model.

## What you get

* **In memory:** `fov.segmentation_results[name]`, a
  {py:class}`~starfinder.segmentation.SegmentationResult`:
  * `labels`: the `uint32` ZYX label image; a plane is 1×Y×X;
  * `grid`: the {py:class}`~starfinder.segmentation.ReferenceGrid` it lies on;
  * `target` (`nucleus` or `cell`) and `geometry` (`volume`, `plane` or `extended`);
  * `label_namespace`: the identity of its labels;
  * `record`: the run record, with the input, the seeds, the method or the import, the
    label operations and the outcome (`ok`, or `empty` without objects).
* **On disk:** with `checkpoints`, each run's folder
  `<FOV folder>/segmentation/<run>/` holds `labels.tif`, `segmentation.json` and, for a
  method, `input.ome.tif`. `fov.load_segmentation(name)` reads a run back and checks its
  hashes, so assignment can run in another process.

## Main options

* `SegmentationRun.method`: the method config with its settings, or a
  `LabelImportConfig` for an import.
* `target`: what the run segments, `nucleus` or `cell`.
* `inputs`: the channels the method reads.
* `seeds`: the earlier run whose nuclei a method grows from.
* `operations`: label operations after the method, such as expanding the labels.
* `projection`: segments a Z projection of the input as one plane.
* `model` or `model_path`: the model of a learned method.
* `StarDistConfig.scale` and `CellposeConfig.diameter`: how the input is scaled to the
  model's training data.

The reference gives each config's options and their defaults.

## Nucleus and cell targets

Each run makes one label image with one target, so nuclei and cells come from two runs.
A cell run of the seeded watershed names the nucleus run as `seeds`, and each
cell keeps the value of the nucleus it grew from. Assignment pairs nuclei with cells
({doc}`assignment-contract`).

## Method input

Each `InputChannel` gives a channel a role and a source:

* a channel of a loaded round, by its name, file pattern or index:
  `InputChannel("nuclear", round="reference_stain", channel="DAPI")`;
* the reference round's channel maximum: `InputChannel("amplicon", reference_merged=True)`.

The roles are `nuclear`, `cytoplasm`, `membrane`, `amplicon` and `composite`. The
methods read:

* `stardist`: one `nuclear` or `composite` channel;
* `cellpose`: a `cytoplasm` or a `nuclear` channel, or both;
* `seeded_watershed`: one `cytoplasm`, `membrane`, `amplicon` or `composite` channel,
  and the seeds.

The reference round is resident after `FOV.run`. A stain from the reference round's own
stain files (`Dataset.reference_stains`, the round `reference_stain`) or from a
morphology round is brought into the reference frame with
`fov.prepare_morphology(...)` ({doc}`coordination`).

`prepare` makes the channel with an input function: `CompositeConfig` combines a nuclear
stain with the reference merged image (`composite_nuclei_amplicon`), and
`FlamingoEnhancementConfig` dims the DAPI signal where the Flamingo stain of
`prepare_channel` is bright (`enhance_with_flamingo`).

A later process can segment from saved images: `fov.load_reference_image()` restores
the reference merged image, and `fov.load_registered_round(name)` each prepared round. The
[specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/segmentation.md#coordination-per-fov)
says which inputs they serve.

## Importing masks

A mask made elsewhere, by CellProfiler for example, enters as a run whose method is
`LabelImportConfig(path, target)`, which names a YX or ZYX integer TIFF. The result is a
`SegmentationResult` like any other. `relabel=True` numbers the labels 1 to
n; otherwise the file's values are kept.

Without a FOV, `import_labels(path, grid=…, target=…, label_namespace=…)` imports onto a
grid you give: `fov.reference_grid()`, the grid of a saved reference image from
`reference_grid_from_file`, or a `ReferenceGrid` you declare.

## Models

StarDist and Cellpose run in the project environment with their extras. From
`src/python`, on Python 3.11 to 3.13:

```bash
uv sync --locked --extra stardist --extra cellpose
uv run starfinder weights fetch stardist 2D_versatile_fluo
uv run starfinder weights fetch cellpose cpsam_v2
```

* **Known models.** `model` names a model of `KNOWN_MODELS`. They share the weights
  cache and the `starfinder weights` commands of the
  {doc}`learned detectors <learned-detectors>`.
* **Your own model.** `model_path` names a StarDist model folder (`config.json`,
  `thresholds.json`, `weights_best.h5`) or a Cellpose model file.
* **Downloads.** `starfinder weights fetch` downloads the known models; segmentation
  reads the files and checks their SHA-256 before it loads them.

## Devices

`device="cuda"` runs StarDist and Cellpose on a GPU; the default is `"cpu"`. The seeded
watershed runs on the CPU.

## Troubleshooting

| Error | When | What to do |
| --- | --- | --- |
| `SegmentationBackendUnavailableError` | The method's extra is not installed | Install the extra (above) |
| `MissingModelError` | A model file or folder is not there | Fetch the known model, or check `model_path` |
| `ModelHashMismatchError` | A model file differs from its expected SHA-256 | Fetch the model again, or check `model_sha256` |
| `TypeError` naming `scale` or `diameter` | A StarDist or Cellpose config without it | Set the input scale |
| `ValueError` naming `'cuda'` | A plan with a watershed run uses `device="cuda"` | Use `"cpu"` |
| `IncompatibleGeometryError` | An input, a mask or the seeds have another ZYX shape than the run's grid, or the model does not fit the input's Z | Check the grid, or choose a model for 2D or 3D input |

## Workflow configuration

The shared rule `stardist_segmentation`, under both backends, runs StarDist on one input
file per FOV from `rules.stardist_segmentation.parameters`, and writes
`images/stardist_segmentation/{fovID}.tif` with its run record. It runs in Snakemake's
environment, with the package and its `stardist` extra. The reference lists the legacy keys.

With `backend: python`, the top-level `segmentation` block describes a plan:

```yaml
backend: python
segmentation:
  device: cpu
  runs:
    - name: nucleus
      target: nucleus
      inputs:
        - {role: nuclear, round: reference_stain, channel: DAPI}
      method: stardist
      model_path: /absolute/stardist_models/3D_spleen
      scale: 1.0
    - name: cell
      target: cell
      seeds: nucleus
      inputs:
        - {role: amplicon, reference_merged: true}
      method: seeded_watershed
```

A run's other keys are the fields of its method's config, or with `method: import` those
of `LabelImportConfig`. No rule runs the block yet.

## Limits

* No training or fine-tuning, and no segmentation from the molecules.
* No default model, no recommended scale and no accuracy claim.
* Segmentation does not assign molecules or pair nuclei with cells; assignment does.

## How it is checked

Golden tests pin the composite, the Flamingo enhancement, planar expansion and label
resampling to the legacy scripts' outputs. Tests check the stage checks, the seeded
watershed on hand-built fixtures, the saved files and the workflow translation; a
`learned` parity test compares StarDist on the CPU with saved legacy results. The
validation design for segmentation and assignment is in {doc}`assignment-algorithms`.

The full specification is in
[`design/specs/segmentation.md`](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/segmentation.md).

```{toctree}
:hidden:

Segmentation reference <segmentation-algorithms>
```
