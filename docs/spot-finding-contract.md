# Spot finding

Spot finding detects candidate spots in each channel of a round's detection image: the
image after {doc}`preprocessing <preprocessing-contract>` and, for a moving round,
{doc}`registration <registration-contract>`. Each candidate gets zero-based voxel
coordinates and an identity, `spot_id`, which {doc}`readout <readout-contract>` uses to
extract, decode and filter reads. In `FOV.run`, detection runs when
`PipelineConfig.spot_finding` is set.

The {doc}`spot-finding reference <spot-finding-algorithms>` lists every method, option
and record; the {doc}`spot-finding API <api/spot_finding>` documents the classes. The
learned detectors, Spotiflow and Piscis, need an extra and pretrained weights;
{doc}`learned-detectors` explains how to install and run them.

## Quick start

```python
from dataclasses import replace

from starfinder.dataset import CheckpointConfig, Dataset, PipelineConfig, RoundState
from starfinder.io import ImageLoadConfig
from starfinder.spot_finding import LocalMaximaConfig
from starfinder.synthetic import benchmark_scene_preset, generate_amplicon_dataset, save_amplicon_dataset

# A synthetic FOV of four rounds, written as a raw dataset.
codebook, scene = benchmark_scene_preset("tiny")
generated = generate_amplicon_dataset(codebook, replace(scene, seed=42), fov_ids=("FOV_001",), preset="tiny")
save_amplicon_dataset(generated, "input")
rounds, channels = list(generated.codebook.round_labels), list(generated.channel_labels)

dataset = Dataset(input_root="input", output_root="results", dataset_id="tiny", sample_id="synthetic",
                  output_id="demo", rounds=RoundState(sequencing_rounds=rounds, reference_round="round1"),
                  channel_order=channels)
pipeline = PipelineConfig(load=ImageLoadConfig(channel_labels=tuple(channels)),
                          spot_finding=LocalMaximaConfig())
fov = dataset.fov("FOV_001")
fov.run(pipeline, checkpoints=CheckpointConfig(stages=("candidates",)))

result = fov.spot_result
print(result)                                   # the spot count and the first columns
print(result.spots.head())                      # spot_id, z, y, x, channel, peak_intensity
print(result.diagnostics["counts"])             # candidates per channel
```

The run writes `results/fovs/FOV_001/checkpoints/candidates.csv` with
`candidates.json`, and `run.json` in the FOV folder.

To detect one image without a dataset, call `find_spots` with the image, the config, the
image's metadata and a namespace for its identities:

```python
import numpy as np

from starfinder.image import ImageMetadata
from starfinder.spot_finding import LocalMaximaConfig, find_spots

image = np.random.default_rng(0).poisson(100, size=(5, 32, 32, 2)).astype(np.uint16)   # ZYXC
image[2, 10, 12, 0] = image[3, 20, 8, 1] = 1000
result = find_spots(image, config=LocalMaximaConfig(threshold_mode="adaptive", threshold_value=0.5),
                    metadata=ImageMetadata("fov001"), spot_namespace="example/fov001")
print(result.spots[["spot_id", "z", "y", "x", "channel"]])
```

## What it does

1. **Channels.** A pipeline method detects each channel on its own.
2. **Threshold.** Local maxima computes each channel's threshold in the mode that
   `threshold_mode` names (below); the other pipeline methods apply their threshold
   option.
3. **Candidates.** The candidates of all channels form one table. `spot_id` numbers
   its rows `"0"`, `"1"`, … within the result's namespace. In a FOV, the namespace
   names the dataset, sample, FOV and subtile.
4. **Records.** The diagnostics record each channel's threshold, count and effective
   settings, and the software and execution in effect.

## Methods

```{method-table} starfinder.spot_finding.SPOT_FINDING_METHODS
```

`FOV.find_spots`, `FOV.run` and the workflow accept the four pipeline methods:
`local_maxima`, `starfish_log`, `spotiflow` and `piscis`. Registration and the
benchmarks call `noise_landmark` and `percentile_centroid` through `find_spots`.
`local_maxima` is the default of `FOV.find_spots` and of the workflow key, and needs no
extra; `starfish_log` reproduces starfish's `BlobDetector`. This page does not
recommend a method or a model.

## What you get

* **In memory:** `fov.spot_result`, a
  {py:class}`~starfinder.spot_finding.SpotFindingResult`:
  * `spots`: the table, with `spot_id`, `z`, `y`, `x` and the method's columns, such
    as `channel` and `peak_intensity`;
  * `diagnostics`: per channel the threshold, `counts`, `outcomes` (`ok`, `empty` or
    `constant`) and `effective_settings`; the software versions; the execution entry;
    for the learned methods, the loaded weights;
  * `config` and `plan`: the config, and the plan it was detected with.
* **On disk:** with the `candidates` checkpoint stage, `FOV.run` writes the table and
  `candidates.json` under `checkpoints/` in the FOV folder, and
  `dataset.fov(...).load_checkpoint("candidates")` reloads them ({doc}`checkpoints`).
  `run.json` records the config or plan, the device and the method's provenance.

The reference lists every column and diagnostic.

## Main options

* The method config: which method runs, with its settings. For `local_maxima`, the
  options most often changed are `threshold_mode` and `threshold_value` (below),
  `min_distance_voxels` (the smallest distance between two maxima),
  `exclude_border` and `merge_radius_zyx` (merges maxima of one channel that lie close
  together).
* `model`: the pretrained weights of `spotiflow` and `piscis`. Every run names one;
  there is no default model.
* `SpotFindingPlan.channel_overrides`: other settings for named channels.
* `SpotFindingPlan.rounds`: the rounds to detect in.

The reference gives each config's options and their defaults.

## Threshold modes

`LocalMaximaConfig.threshold_mode` says how `threshold_value` becomes a channel's
threshold:

* `noise`, the default: `median + threshold_value × 1.4826 × MAD` of the channel,
  where MAD is the median absolute deviation, so the value counts robust standard
  deviations above the median;
* `adaptive`: `threshold_value` times the channel's maximum;
* `adaptive_round`: `threshold_value` times the maximum of the whole image, all
  channels together;
* `global`: `threshold_value` times the maximum of the image's integer dtype.

Local maxima records each channel's median, MAD and zero fraction in
`diagnostics["noise"]`, in every mode, and a `SpotFindingWarning` names a channel
whose median and MAD do not measure its noise; its threshold is unchanged. The other
pipeline methods have their own thresholds: `threshold` for `starfish_log` and `piscis`,
`prob_thresh` for `spotiflow`.

## Detection plan

A `SpotFindingPlan` runs one method on every channel and replaces the whole config of
the channels its overrides name. A bare config means a plan without overrides.

```python
from starfinder.spot_finding import ChannelOverride, LocalMaximaConfig, SpotFindingPlan

plan = SpotFindingPlan(
    LocalMaximaConfig(),
    channel_overrides=(ChannelOverride("ch01", LocalMaximaConfig(threshold_value=8.0)),),
)
```

An override names its channel by label: `FOV.find_spots` and `FOV.run` take the
labels from `Dataset.channel_order`, and direct `find_spots` from
`config.channel_labels`. {py:class}`~starfinder.spot_finding.ChannelOverride` says
which configs and settings an override takes.

## Detection in several rounds

By default, detection runs in the reference round only. `SpotFindingPlan.rounds`
names the rounds to detect in, for example
`SpotFindingPlan(LocalMaximaConfig(), rounds=("round1", "round2"))`. `FOV.run` detects
each listed round on its registered image, so all coordinates are on the reference
round's grid. The result is one table with a `round` column, and `spot_id` runs over
the whole table. Candidates of different rounds stay separate rows. Per-round
diagnostics are under `diagnostics["rounds"]`.

`FOV.find_spots` and `FOV.run` detect the rounds of a plan; `find_spots` detects one
image. {doc}`readout-contract` describes how direct readout uses such a table.

## Z=1

The methods detect an image with one Z plane as a YX plane, and its candidates have
`z=0`; the
[specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/spot-finding.md#z1-dimensionality-scale-and-tiling)
gives each method's rules for such images. For Spotiflow, choose a model of the
image's dimensionality ({doc}`learned-detectors` gives each model's). Piscis detects
single planes and volumes with the same weights.

## Execution device

Spot finding runs on the CPU only. `ExecutionConfig.device`, the `device` keyword of
`find_spots` and `FOV.find_spots`, and the workflow's `device` key accept only
`"cpu"`, the default; any other value raises
`ValueError("device must be 'cpu'; spot finding runs on CPU only")`. Starfinder does
not change thread settings: set `OMP_NUM_THREADS` and the other thread variables
before you start Python, and `diagnostics["execution"]` records what was in effect.

## Workflow keys

On the Python backend, the `spot_finding` block of a Python processing rule holds
`run`, `ref_round`, `method` (default `local_maxima`) and the config's fields, plus
`channel_overrides` and `rounds`:

```yaml
backend: python
rules:
  rsf_single_fov:
    parameters:
      device: cpu
      spot_finding:
        run: true
        ref_round: round1
        method: local_maxima
        threshold_mode: noise
        threshold_value: 5.0
        channel_overrides:
          ch01: {threshold_value: 8.0}
```

Under `backend: python`, local maxima takes the field names `threshold_mode`,
`threshold_value` and `min_distance_voxels`, in place of the MATLAB keys
`intensity_estimation`, `intensity_threshold` and `min_distance`. "Spot-finding
method key" in {doc}`workflow-configuration` describes the block.

## Limits

* One method per run. Overrides change settings, not the method.
* No deduplication across channels: a crosstalk copy of a spot in the next channel is
  a separate candidate.
* No training or fine-tuning of the learned detectors, and no automatic resampling.
* No GPU.
* Coordinates are voxel indices, not physical units.

## How it is checked

A golden test pins the local-maxima results of the four threshold modes on a small
seeded fixture, in 3D and at Z=1, and the candidates checkpoint that `FOV.run` writes.
Tests compare the Starfish LoG with saved starfish results exactly, and check the four
pipeline methods on small synthetic scenes against their truth positions; the
learned-detector tests need the extras and the weights. The validation design is in
the
[algorithms specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/spot-finding-algorithms.md).

The full specification is in
[`design/specs/spot-finding.md`](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/spot-finding.md).

```{toctree}
:hidden:

Spot-finding reference <spot-finding-algorithms>
```
