# Amplicon scene

The amplicon scene renders one field of view (FOV) of processed images: amplicons read
out round by round in four channels, with their positions, genes and per-round motion
known. Use it to test or demonstrate registration, spot finding and decoding against
known answers.

It is the base of the other scenes. The {doc}`sample scene <synthetic-sample-scene>`
images several overlapping FOVs of one sample, the
{doc}`cellular scene <synthetic-cellular-scene>` adds nuclei, cells and stains, and the
{doc}`real-appearance scene <synthetic-real-appearance>` takes their appearance from
real images.

## Quick start

```python
from dataclasses import replace

from starfinder.synthetic import (
    benchmark_scene_preset,
    generate_amplicon_dataset,
    save_amplicon_dataset,
)

codebook, config = benchmark_scene_preset("tiny")
data = generate_amplicon_dataset(
    codebook, replace(config, seed=42), fov_ids=("FOV_001",), preset="tiny"
)
image = data.rounds["FOV_001"]["round1"]  # ZYXC, uint16, shape (8, 128, 128, 4)
truth = data.spot_truth                   # one row per amplicon and round
save_amplicon_dataset(data, "synthetic_tiny")  # the layout that Dataset reads
```

`generate_amplicon_scene(codebook, config=config)` returns a single scene in memory
instead. The command line writes a dataset directly:

```bash
starfinder synthetic generate --mode e2e --preset tiny --seed 42 --owner NAME --output NEW_DIR
```

The {doc}`single FOV demo <get-started/single-fov-demo>` runs this scene through the
pipeline and compares the result with the truth.

## What it generates

The scene is built in this order. Steps 4 to 6 and 8 to 10 are optional and off in a
plain `AmpliconSceneConfig`; the benchmark presets switch on motion, crosstalk, a
per-round trend, background, an offset and noise.

1. **Population.** Amplicons are placed uniformly, by a weight map, in clusters, or at
   given coordinates. Each gets a gene, drawn by the codebook's abundance weights.
2. **Properties.** Each amplicon keeps one brightness, axial and lateral width,
   elongation and angle in every round.
3. **Signal.** In each round, an amplicon lights the channel that its colour sequence
   names.
4. **Readout effects.** Persistent loss from a given round on, temporary dropout and
   weakening, a per-round trend and per-round channel gains.
5. **Crosstalk.** A per-round matrix mixes signal between channels.
6. **Motion.** Each round can be translated, warped smoothly, and moved by affine and
   polynomial terms. The reference round stays fixed.
7. **Rendering.** Each amplicon is drawn as an elongated, rotated Gaussian at its moved
   position. No image is interpolated.
8. **Background.** A gradient, broad regions and texture that move with the sample.
9. **Baseline.** A constant offset per round and channel.
10. **Noise.** Signal-dependent noise (Gaussian or Poisson), read noise and spatially
    correlated noise.
11. **Output.** One cast to the output type. Integer output warns when more than 1 % of
    a round's voxels clip.

## Presets

`benchmark_scene_preset(name)` returns a codebook and a configuration for these
presets. They all have four rounds and four channels, and write uint16 unless you ask
for `uint8`.

```{synthetic-presets} amplicon
```

The small presets serve examples and the test suite; `large`, `tissue` and
`thick_medium` measure runtime and memory at FOV and tissue scale. Generating a whole
large preset in memory holds every round of every FOV (about 9 GiB per FOV for
`tissue`): use the command line, or pass `on_round` to stream rounds out. The other
amplicon presets, for model tests, preprocessing evaluation and registration pairs,
are listed in the {doc}`synthetic reference <synthetic-reference>`.

## What you get

* **In memory.** An `AmpliconScene` holds the images of each round (`rounds`), the
  formed amplicons (`formed`), what happened to each amplicon in each round
  (`round_truth`) and a `provenance` record with the configuration, the seed and the
  image hashes. `generate_amplicon_dataset` returns a `SyntheticDataset` that adds
  `spot_truth`, one row per amplicon and round with its gene, channel, position and
  whether it was rendered.
* **On disk.** `save_amplicon_dataset` writes one TIFF per round, FOV and channel in
  the layout that {doc}`Dataset <coordination>` reads, with the codebook and the truth
  tables beside them. `save_amplicon_scene` writes one scene. The
  {doc}`synthetic reference <synthetic-reference>` lists the files and their columns.

## Main options

Change a preset with `dataclasses.replace`, for example
`replace(config, noise=NoiseConfig())` to switch noise off.

| Option of `AmpliconSceneConfig` | Default | Meaning |
| --- | --- | --- |
| `seed` | 42 | Root seed of every random draw |
| `shape_zyx` | (8, 32, 32) | Image shape; Z of 1 gives one plane |
| `dtype` | `"float32"` | `float32`, `float64`, `uint8` or `uint16` |
| `count`, `density`, `coordinates` | 8 amplicons | Set at most one |
| `placement` | `"uniform"` | Or `"weighted"` (with `spatial_weights`) or `"clustered"` |
| `brightness`, `axial_width`, `lateral_width`, `elongation`, `angle` | constants | Each a `ScalarDistribution`: constant, uniform, lognormal or supplied |
| `readout` | off | `ReadoutEffectsConfig`: loss, dropout, weakening, trend, gains, crosstalk |
| `geometry` | off | `GeometryConfig`: translation, local warps, affine, polynomial |
| `background`, `noise` | off | `BackgroundConfig` and `NoiseConfig` |

The background's gradient, regions and texture stay invisible until
`BackgroundConfig.tissue_weights` is set. A changed `shape_zyx` keeps a preset's
background regions in voxels of the original shape; for one plane, also set the Z
translation and the regions' Z values to zero.

## Limits

* The scene starts once amplicons have formed: no RNA, no amplification, no new
  molecules between rounds.
* Spot widths describe spots as they look in processed images, not an optical point
  spread function. Noise is a residual model, not camera physics.
* The background is smooth shapes, not tissue; there are no cells, stains or
  overlapping FOVs.
* Motion moves spots but does not change their shape, and crosstalk does not act on
  the background.
* The appearance of the presets is chosen to be plausible, not fitted to real data.
* The codebook must have four channels and one entry per gene.
* Images are identical for the same seed, NumPy build and CPU.

The full model, with its formulas, is specified in
[`design/specs/synthetic/amplicon-scene.md`](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/synthetic/amplicon-scene.md).
