# Spot-finding reference

The methods, options, columns, diagnostics and files of spot finding. The method and
option tables are read from the code when the site is built. The
{doc}`spot-finding guide <spot-finding-contract>` explains how to use them; the
{doc}`spot-finding API <api/spot_finding>` documents every class.

## Methods

```{method-table} starfinder.spot_finding.SPOT_FINDING_METHODS
```

Each method's name is its `method` value, used by the workflow key. `FOV.find_spots`,
`FOV.run` and the workflow accept `local_maxima`, `starfish_log`, `spotiflow` and
`piscis`; registration and the benchmarks call `noise_landmark` and
`percentile_centroid` through `find_spots`.

## Method options

### Local maxima

Finds the local maxima of each channel above the channel's threshold with
scikit-image's `peak_local_max`. The guide's
[threshold modes](spot-finding-contract.md#threshold-modes) say how the threshold is
computed. See {py:class}`~starfinder.spot_finding.LocalMaximaConfig`.

```{config-options} starfinder.spot_finding.LocalMaximaConfig
```

### Starfish LoG

A reimplementation of starfish's `BlobDetector` with `blob_log`: Laplacian-of-Gaussian
blobs over a range of sizes, one channel at a time, with the same results as starfish.
See {py:class}`~starfinder.spot_finding.StarfishLogConfig`.

```{config-options} starfinder.spot_finding.StarfishLogConfig
```

### Spotiflow

A learned detector that predicts a probability heatmap and finds spots at its peaks,
optionally refined to sub-voxel positions. It needs the `spotiflow` extra and
verified pretrained weights ({doc}`learned-detectors`). See
{py:class}`~starfinder.spot_finding.SpotiflowConfig`.

```{config-options} starfinder.spot_finding.SpotiflowConfig
```

### Piscis

A learned detector that predicts a label map on each Z plane, in tiles, and links the
planes of a volume. It needs the `piscis` extra and verified pretrained weights
({doc}`learned-detectors`). See
{py:class}`~starfinder.spot_finding.PiscisConfig`.

```{config-options} starfinder.spot_finding.PiscisConfig
```

### Registration methods

See {py:class}`~starfinder.spot_finding.NoiseLandmarkConfig` and
{py:class}`~starfinder.spot_finding.PercentileCentroidConfig`.

```{config-options} starfinder.spot_finding.NoiseLandmarkConfig
```

```{config-options} starfinder.spot_finding.PercentileCentroidConfig
```

## Plan options

```{config-options} starfinder.spot_finding.SpotFindingPlan
```

```{config-options} starfinder.spot_finding.ChannelOverride
```

See {py:class}`~starfinder.spot_finding.SpotFindingPlan` and
{py:class}`~starfinder.spot_finding.ChannelOverride`. The execution device is
`ExecutionConfig.device`, or the `device` keyword of `find_spots` and
`FOV.find_spots`; `"cpu"` is the only value.

## Columns

`SpotFindingResult.spots` holds these columns, in the method's order:

| Column | Type | Methods | Contents |
| --- | --- | --- | --- |
| `spot_id` | string | all | The identity, unique within the result's namespace |
| `z`, `y`, `x` | float64 | all | Zero-based voxel coordinates, which may be fractional |
| `channel` | int64 | the four pipeline methods | The index of the candidate's channel on the image's channel axis |
| `peak_intensity` | float64 | the four pipeline methods (`local_maxima` with `measure_peak_intensity`) | The original pixel value at the candidate's voxel |
| `radius` | float64 | `starfish_log` | The blob radius in voxels |
| `probability` | float64 | `spotiflow` | The heatmap probability of the spot |
| `round` | string | a plan with `rounds` | The round the candidate was detected in |

## Diagnostics

`SpotFindingResult.diagnostics` can hold these keys:

| Key | Contents |
| --- | --- |
| `method`, `channel_labels` | The method name and the channel labels |
| `thresholds` | The threshold of each channel, in the method's units (one value for `percentile_centroid`, which sums the channels) |
| `counts` | Pipeline methods: the candidates per channel |
| `outcomes` | Pipeline methods: per channel, `ok`, `empty` (the method ran and found nothing) or `constant` (a constant channel) |
| `effective_settings` | Per channel, the channel's config |
| `noise` | `local_maxima`: per channel, `zero_fraction`, `median`, `mad` and `threshold` |
| `merged` | `local_maxima` with `merge_radius_zyx`: the maxima removed per channel |
| `warnings` | The messages of the `SpotFindingWarning`s emitted |
| `native` | `starfish_log` and `spotiflow`: per channel, the minimum, median and maximum of `radius` or `probability`, or `None` without candidates |
| `geometry` | `starfish_log`: the scale-space memory estimate; `spotiflow`: the tiling; `piscis`: the mode, tile size, overlap and tile boundaries |
| `model` | `spotiflow` and `piscis`: the model, the loaded files with their SHA-256, and the training pixel size |
| `software` | The versions of starfinder, NumPy, SciPy, scikit-image and the method's libraries |
| `execution` | The device, the torch build for the learned methods, and the thread settings |
| `coordinate_units`, `singleton_z_policy`, `measurements` | `voxel_index`, how a Z=1 image is detected, and what each measured column means |
| `rounds` | A plan with `rounds`: per round, `thresholds`, `counts`, `outcomes`, `noise` and `merged`, which are then not at the top level |

## Pretrained weights

`spotiflow` and `piscis` load the weights their `model` names from Starfinder's
weights cache. `KNOWN_MODELS` lists the models; {doc}`learned-detectors` gives each
model and the commands that fetch and verify them. Detection never downloads.

## Files

| File | Written by | Contents |
| --- | --- | --- |
| `checkpoints/candidates.csv` or `.parquet` | `FOV.run` with the `candidates` stage | The spot table with its namespace, and the extracted signals when extraction ran |
| `checkpoints/candidates.json` | the same | The config, the plan's overrides and rounds, the execution entry, the weights, the JSON-representable diagnostics and the column dtypes |
| `run.json` | `FOV.run` with `checkpoints` | `config.pipeline.spot_finding` (the config or plan), `config.execution` (the device), and the detection's provenance entry |

Paths are inside the FOV's folder, `<output_root>/fovs/<fov_id>/` by default;
{doc}`checkpoint reference <checkpoints-reference>` describes them.

## Workflow keys

With `backend: python`, the `spot_finding` block of a Python processing rule holds:

| Key, under `rules.<rule>.parameters.spot_finding` | Meaning |
| --- | --- |
| `run` | Whether the rule detects spots |
| `ref_round` | The dataset's reference round |
| `method` | A pipeline method of the table above; default `local_maxima` |
| The config's fields | Fields of that method's config, such as `threshold_mode` or `model`; the option tables above mark the required ones |
| `channel_overrides` | A mapping from channel label to the fields that change for that channel |
| `rounds` | A list of the round labels to detect in |

The rule-level key `device` sets `ExecutionConfig.device`. The [workflow specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/workflow-configuration.md#spot-finding-method-key) gives the block's other rules.
