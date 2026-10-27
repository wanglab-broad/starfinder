# Real-appearance scene

The {doc}`cellular scene <synthetic-cellular-scene>` draws its stains from formulas, so
its nuclei look smooth and regular. The real-appearance scene keeps the cellular
scene's sample, cell types and molecules, but takes the look of each nucleus from an
**appearance bank**: nuclei, or for culture whole cells with their nucleus, cut out of
real images. It places them on a synthetic layout and keeps the placed masks as the
truth.

Use it when a segmentation or assignment chain needs images that look like lymph-node,
2D tissue or culture data and still has known labels. A bank can also be drawn from
the cellular scene in memory, so the scene runs without any real data.

## Quick start

```python
from starfinder.synthetic import (
    appearance_scene_preset,
    generate_appearance_scene,
    save_appearance_scene,
)

codebook, config = appearance_scene_preset("appearance-tissue2d-v1")  # needs no real data
scene = generate_appearance_scene(codebook, config=config)
scene.placements                       # one row per placed nucleus or cell
scene.context.stains["FOV_001"]["PI"]  # 8-bit ZYX stain of one FOV
save_appearance_scene(scene, "appearance_tissue2d")  # a new or empty directory
```

`scene.context` is a `CellularScene`, so everything the cellular scene offers, from
truth tables to `save_cellular_dataset`, works on it.

## How it works

* **The bank.** Each entry is one nucleus, or one culture cell with its nucleus, with
  its mask and its stain intensities relative to the background of its source image.
  Building a bank leaves out objects on the image border, very small or very large
  ones, and squashed ones. Real banks also store background patches.
* **A bank without real data.** For the routine presets the bank is drawn from the
  matching cellular preset at a few seeds, without noise. Nothing is read from disk.
* **Variation.** Each placed entry is rotated, may be flipped, and is scaled and
  brightened by up to 10 %.
* **Layout.** Entries are proposed one at a time, mostly against an object already
  placed. A proposal is refused if it overlaps another object by more than 1 %, falls
  mostly outside the sample, or (culture) puts two nuclei in contact. Placement stops
  at a target count or visible stain share. Cells grow from the placed nuclei in dense
  tissue; culture cells keep their own masks.
* **Images.** Each stain is background plus out-of-focus light plus the placed
  entries, scaled to 8 bits. No noise is added; the texture comes from the bank.
* **Morphology round.** For culture, the morphology round's DAPI and Flamingo come from
  the same entries, shifted by each FOV's known translation.

## Using a real bank

The slow presets name a real bank by name and SHA-256: `ln-sample1-v1`,
`tissue2d-v1` and `culture3d-v1`. These are lab files and are not distributed. To use
one, put `<name>.npz` and its `<name>.record.json` in a folder and either set
`STARFINDER_APPEARANCE_BANKS` to that folder or pass
`BankReference(name, sha256, directory=...)`. Loading fails with a clear error if the
folder is not set, the file is missing, or its hash does not match; it never falls back
to another bank.

To build your own bank from labelled images, call
`build_appearance_bank(config, "<folder>/<bank>.npz")` and use the SHA-256 in the
record it returns. With a stored bank and a target visible share, set
`LayoutConfig.target` and `threshold` yourself.

## Presets

`appearance_scene_preset(name)` returns the 16-gene development codebook and a
configuration. Each preset starts from the cellular preset of the same kind and tier.

```{synthetic-presets} appearance
```

The routine presets take a few seconds each. The slow ones need their real bank;
the culture one takes about 11 minutes and 13 GiB.

## What you get

* **In memory.** An `AppearanceScene` holds the cellular scene it built (`context`),
  the placed entries (`placements`), the background patches used (`background_patches`)
  and a `record`.
* **On disk.** `save_appearance_scene` writes the cellular scene's files, plus
  `placements.csv` (each object's bank entry, source, position, rotation, flips, scale
  and gain), `background_patches.csv` and `appearance.json` (configuration, bank
  identity, how the layout ended, and file hashes). Shape columns that only the
  formula-drawn scene has are left empty. The
  {doc}`synthetic reference <synthetic-reference>` lists every file.

## Main options

| Configuration | Main options (defaults) |
| --- | --- |
| `BankReference` | `name`, `sha256`, `directory`; or `preset` with `n` (4) for a bank drawn from a cellular preset |
| `AugmentationConfig` | `scale` (0.9–1.1), `gain` (0.9–1.1), `flip_y` (0.5), `flip_z` (0.5) |
| `LayoutConfig` | `density`: `"visible"` (with `target`, `threshold`) or `"count"` (with `count`); `contact` (0.75); `max_proposals` (40,000); `max_refusals` (4,000); `floor` (0.9) |
| `CompositionConfig` | `light` (`"objects"` or `"none"`), `light_sigma_um`, `background`, `plateau` |
| `AppearanceSceneConfig` | `bank`, the layout, augmentation and composition, and `context`: the cellular configuration whose sample, cell types and molecules are used |

## Limits

* The truth masks are a method's labels (StarDist, Cellpose) on the source images, not
  manual annotation. Nuclei the method missed are absent from the bank.
* Shapes are more regular than real ones, because squashed sources are left out, and
  edges are narrower.
* The 2D tissue stack repeats one plane; rotation is about Z only; there are no
  binucleate or empty cells and no membrane stain.
* The real banks are lab-only, and the parameters are development targets, not a
  calibration.

## How it is checked

The generator's tests rebuild labels, stains and the 8-bit images from
`placements.csv` and the bank, and check that the routine presets read no real data.
Scenes from real banks are compared with targets measured on real images. The full
checks are in
[`design/validation/synthetic-real-appearance-scene.md`](https://github.com/wanglab-broad/starfinder/blob/dev/design/validation/synthetic-real-appearance-scene.md),
and the full specification in
[`design/specs/synthetic/real-appearance-scene.md`](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/synthetic/real-appearance-scene.md).
