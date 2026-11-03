# Recipe / Experiments

Start with the lifecycle guide to run explicit cases, then look up case builders
and profiles in the recipe reference. The research-script guide describes tools
for inspecting saved codebook-aware decoding results.

```{toctree}
:maxdepth: 1

Benchmark lifecycle <../benchmark>
Maintained benchmark recipes <../benchmark-recipes>
Codebook-aware research scripts <../research-scripts>
```

## Private stack trials

The private entry `python -m starfinder.benchmark._trial trial.json` runs one
registration, preprocessing, spot-finding or segmentation method in one process. Its `method`, `inputs`,
`seed` and `root_output_path` identify the operation and fresh output directory.
It records timing, CPU time, peak RSS, resolved configuration and output hashes
in `record.json`; it does not compute evaluation metrics.

For preprocessing, an input can be a file path, a path with its SHA-256, or a
regenerative descriptor. The descriptor has `scene` (preset, version, condition,
seed and optional saturation multiplier `k`), `round` and optional `channel`.
The supported preset is `uncalibrated_preprocessing`, with version
`UNCALIBRATED_PREPROCESSING_VERSION` from the {doc}`synthetic reference <../synthetic-reference>`.
The installed version and configuration hash are checked before generation.
The trial regenerates a shared scene once, selects its ZYXC image or ZYX channel,
and records the generated image hashes in `scenes.json`. No input image file is
stored. This is an explicitly uncalibrated pilot, with the documented SNR and
gain-spread misses; absolute performance is optimistic.

A `spot_finding` trial names one method of `SPOT_FINDING_METHODS` with its
`config`, and its single `image` input is a file or the same amplicon scene
descriptor as preprocessing. The trial calls `find_spots` and writes the spot
table as `spots.csv` and its diagnostics as `spot_finding.json`. A
`segmentation` trial names one method of `SEGMENTATION_METHODS` with its
`config`, `target` and one channel `roles` entry per image channel. Its `image`
is a file or a cellular scene descriptor: `scene` (a `CELLULAR_PRESETS` name as
preset, `CELLULAR_PRESET_VERSION`, the trial seed), `fov`, `round` (`reference`
or `morphology`) and the ordered `stains` to stack. A seeded method also takes a
`seeds` label file. The trial calls `segment` and writes `labels.npy` and
`segmentation.json`. Learned weights must already be in the Starfinder weights
cache: they are verified before any input is loaded, and missing weights fail
with `MissingModelError`. These two modules record the device and model identity
under `execution` in `record.json`. E02 and E04 evaluate the saved outputs with
`starfinder.evaluation`.

A preprocessing method's optional `reference_steps` run on its reference before
the main steps and are recorded in `reference_preprocessing.json`. These steps
cannot themselves require a reference. For `r1`, this makes min–max normalization
precede histogram matching for both the image and its single-channel reference.

The private `starfinder.benchmark._experiment.expand_experiment` expands methods,
conditions, seeds and input sets in that order, without generating images.
Each condition can override `method` and `inputs` recursively. A scene descriptor
without a condition or seed receives the current condition name and seed.
An optional experiment `root_output_path` places trial outputs on local scratch;
otherwise they default to the expansion directory. Configuration hashes exclude
that output root, so jobs can relocate it to their temporary directory. The
expansion manifest lists each resolved configuration and its two hashes.
