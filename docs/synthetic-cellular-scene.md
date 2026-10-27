# Cellular scene

The cellular scene generates a synthetic sample with cell structure: lobed and touching
nuclei, cell territories, four cell types, molecules placed in the nucleus or the
cytoplasm, 8-bit stain images and, for culture, an extra morphology round. Every part
comes with its truth, so segmentation, assignment, assembly and the report can be run
on it and compared with known answers.

It extends the {doc}`sample scene <synthetic-sample-scene>`: with every addition left
at its default, it produces exactly the sample scene. The
{doc}`real-appearance scene <synthetic-real-appearance>` starts from the same presets
and replaces the drawn nuclei and stains with ones taken from real images.

Its presets cover three kinds of data:

* **dense 3D tissue**, like lymph node: crowded nuclei in 3D, a DAPI stain;
* **2D tissue**, a one-layer section imaged for projection: a PI stain;
* **sparse culture**: spread-out cells, a DAPI stain, and a morphology round with DAPI
  and Flamingo.

## Quick start

```python
from starfinder.synthetic import (
    cellular_scene_preset,
    generate_cellular_scene,
    save_cellular_scene,
)

codebook, config = cellular_scene_preset("context-tissue2d-v1")
scene = generate_cellular_scene(codebook, config=config)
len(scene.sample.cells), len(scene.sample.molecules)  # cells and molecules
scene.stains["FOV_001"]["PI"]                          # 8-bit ZYX stain of one FOV
save_cellular_scene(scene, "tissue2d")                 # a new or empty directory
```

`docs/examples/cellular_scene.py` generates two presets this way, from `src/python`:
`uv run python ../../docs/examples/cellular_scene.py OUTPUT_DIR`.

## What it generates

* **Nuclei.** Each nucleus is an ellipsoid or, for a set share, a union of two or three
  ellipsoids. Lobes stay joined, so a nucleus is always one piece.
* **Packing.** Nuclei are added one at a time until they fill a target fraction of the
  volume or area. Neighbours may touch along flat faces. If no more nuclei fit, packing
  stops and records the density it reached.
* **Cells.** Each cell is the region around its nucleus, limited by growing the nucleus
  outwards by a set distance or by a reach from its centre. A nucleus always lies inside
  its own cell.
* **Cell types.** Four types fill four regions of the sample, each 90 % pure. Each type
  has two marker genes, and eight genes are shared. Two named marker pairs have known
  answers: one pair never shares a cell, the other is expressed together.
* **Molecules.** Each cell gets a random number of molecules; each molecule lies in the
  nucleus with a set probability, otherwise in the cytoplasm. Molecules between cells
  carry shared genes only.
* **Stains.** One image per stain and FOV. A stain lights nuclei or cytoplasm with soft
  edges, can add bright spots and a smooth unevenness that continues across FOVs, adds
  background and noise, and is scaled to 8 bits.
* **Morphology round.** An extra round of stain images only, shifted per FOV by a known
  small translation, as a morphology round would be.

A scene of one plane works too: set the shapes to Z of 1 and the morphology round's Z
translation to 0.

## Presets

`cellular_scene_preset(name)` returns the 16-gene development codebook and a
configuration. Each kind of data has a routine preset, for examples and fast tests, and a
slow one at a realistic scale.

```{synthetic-presets} cellular
```

On one CPU, a routine preset takes 2 to 4 seconds and about 0.25 GiB; a slow preset 1.5
to 2.5 minutes and under 2 GiB. Nucleus and molecule counts vary with the seed; the
table gives typical values. To draw another sample, change the seed of the FOV
template:

```python
from dataclasses import replace

fov = replace(config.sample.fov, seed=43)
config = replace(config, sample=replace(config.sample, fov=fov))
```

## What you get

* **In memory.** A `CellularScene` holds the `SampleScene` (`sample`), the stain images
  (`stains`), the morphology round (`morphology`), the truth tables of the additions
  and a `record` of the configuration and outcome.
* **On disk.** `save_cellular_scene` writes everything the sample scene writes, plus:
  * the nucleus lobes, each cell's type and region, gene counts per cell, and the
    marker pairs with their expected answers;
  * one table per stain, its objects and its bright regions, and the morphology round's
    true translation per FOV;
  * the stain images per FOV, and `context.json` with the configuration and file
    hashes.

  The {doc}`synthetic reference <synthetic-reference>` lists every file.
  `save_cellular_dataset` writes the scene as a raw dataset, one TIFF per round, FOV and
  channel with a `dataset.json` that describes it, for the pipeline to read.

## Main options

`CellularSceneConfig` adds these parts to the sample scene; leaving one as `None` leaves
it out.

| Part | Main options (defaults) |
| --- | --- |
| `sample` | The `SampleSceneConfig` it extends; its `stain` must be `None` when `stains` is set |
| `nuclei` | `lobed_share` (0): share of nuclei with extra lobes |
| `packing` | `density`: target volume or area fraction; `contact` (0): how far neighbours may touch; `monolayer` (false); `on_crowded` (`"raise"`, or `"stop"`) |
| `territories` | `rule`: `"reach"`, `"grown"` (nucleus grown by `growth`) or `"power"` |
| `types` | Four types, their proportions, regions, purity and marker pairs |
| `molecules` | `per_cell_median` (8), `per_cell_spread`, `nuclear_fraction`, `ambient_density` |
| `stains` | One `StainChannelConfig` per stain: `name`, `source` (`"nucleus"` or `"territory"`), `round`, `channel`, brightness, edge, bright regions, unevenness, output scale (`uint8` or `uint16`) |
| `morphology` | `MorphologyRoundConfig`: round `name` (`"organelle"`) and the largest translation |

## Limits

* It is not a calibrated tissue model. Its parameters are development targets read from
  method labels (StarDist, Cellpose, watershed) on real images; it says nothing about
  how accurate a segmenter is.
* There are no optics, no depth attenuation, no structured tissue background and no
  illumination difference between FOVs.
* Routine scenes hold under ten cells: enough to exercise a pipeline, too few for
  statistics. With 16 planes, many dense 3D nuclei are cut at the top or bottom.

## How it is checked

The generator's tests check that every label and stain image matches the truth tables,
that the identities between the tables hold, and that the slow presets' statistics lie
in ranges measured on real images. The full checks are in
[`design/validation/synthetic-cellular-scene.md`](https://github.com/wanglab-broad/starfinder/blob/dev/design/validation/synthetic-cellular-scene.md),
and the full specification in
[`design/specs/synthetic/cellular-scene.md`](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/synthetic/cellular-scene.md).
