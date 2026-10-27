# Sample scene

The sample scene generates one synthetic sample and images it as a grid of overlapping
FOVs. Each FOV sits at a known true position: its nominal place on the grid plus a small
random offset. The sample holds simple cells with nuclei, molecules inside and between
the cells, and a nuclear stain. The truth records where each FOV really is, which cell
each molecule belongs to, and which FOVs see each cell and molecule.

Use it to test stitching, assignment and assembly against answers that real data cannot
give. Each FOV is rendered as an {doc}`amplicon scene <synthetic-specification>`; the
{doc}`cellular scene <synthetic-cellular-scene>` builds on this scene with realistic
nuclei, cell types and several stains.

## Quick start

```python
from starfinder.synthetic import generate_sample_scene, sample_scene_preset, save_sample_scene

codebook, config = sample_scene_preset("sample-tiny-v1")
scene = generate_sample_scene(codebook, config=config)
scene.placement[["fov_id", "true_origin_y", "true_origin_x"]]  # where each FOV really is
scene.fovs["FOV_001"].rounds["round1"]                        # ZYXC image of one FOV
save_sample_scene(scene, "sample_tiny")                       # a new or empty directory
```

The {doc}`multi-FOV demo <get-started/multi-fov-demo>` stitches, assigns and assembles
this scene and compares each step with the truth.

## What it generates

* **The grid.** FOVs are numbered `FOV_001`, `FOV_002`, … down each column, then
  across. Neighbours overlap by a set fraction, all in one layer.
* **True positions.** Each FOV's true origin is its nominal origin plus a random offset
  of up to a few voxels, usually not a whole number of voxels. You can also supply the
  offsets.
* **Rendering.** Every FOV is drawn from the same sample coordinates at its true
  position, with its own round-to-round motion and its own noise. No image is
  interpolated, so overlapping FOVs agree on content but not on noise.
* **Cells.** Ordinary, binucleate and empty cells are placed with a gap between them.
  Nuclei are rotated ellipsoids; each cell is the region around its nucleus, cut off at
  a set reach, so every nucleus lies inside its cell. Nucleus and cell labels are
  written for the whole sample and for each FOV.
* **Molecules.** Each cell gets a random number of molecules inside it; a few more lie
  between the cells.
* **Stain.** One nuclear stain image per FOV, with soft nucleus edges, an offset and
  noise.

The sample is a little larger than the area the FOVs always cover, so that no edge
effect depends on the random offsets. Rows marked `in_sample` lie in the covered area.

## Presets

`sample_scene_preset(name)` returns a codebook of 16 genes in four rounds and a
configuration.

```{synthetic-presets} sample
```

`sample-tiny-v1` is small enough for examples and routine tests; `sample-small-v1` uses
the voxel size of the lymph-node data and takes a few seconds and about 0.4 GiB.

## What you get

* **In memory.** A `SampleScene` holds each FOV's `AmpliconScene` (`fovs`), the true
  placement (`placement`), the nuclei, cells and molecules with their truth, and the
  label volumes. Pass `on_fov` to `generate_sample_scene` to hand each FOV to a function
  instead of keeping it.
* **On disk.** `save_sample_scene` writes `sample.json`, the FOV positions and their
  truth, tables of nuclei, cells and molecules, which FOVs see each of them, the label
  volumes, and one folder per FOV with its images, stain and labels. The
  {doc}`synthetic reference <synthetic-reference>` lists every file.

## Checking a pipeline against it

* **Stitching.** Compare the stitched FOV origins with `true_origin_g1_*` in
  `placement_truth.csv`, or the corrections with `true_correction_*` minus their mean.
* **Assignment.** Compare each molecule's assigned cell with `true_cell_id`. Where
  `label_agrees` is false, the voxel label differs from the cell at the molecule's exact
  position: that is sampling error, not a method error.
* **Assembly.** Every molecule and cell in the sample is seen by at least one FOV;
  `molecule_fovs.csv` and `cell_fovs.csv` say which.
* **Without a segmenter.** Import the saved label images with `import_labels` and the
  per-FOV `molecules.csv` with `import_legacy_molecules_csv`, assign, then place the
  FOVs at their true origins with `placement_from_table` and assemble.

Score rows with `in_sample` true; report the rest separately.

## Main options

| Configuration | Main options (defaults) |
| --- | --- |
| `AcquisitionConfig` | `columns`, `rows` (2, 2); `fov_shape_zyx` (5, 36, 36); `spacing_zyx` in µm (1.0, 0.5, 0.5); `overlap` (1/3); `jitter` (`"uniform"` or `"supplied"`) and `jitter_max_zyx` in voxels (0.4, 1.5, 1.5) |
| `CellsConfig` | `nuclei` (7), `binucleate` (1), `empty` (0); nucleus radii, elongation and angle as distributions; `reach` of a cell beyond its nucleus; `gap` between cells |
| `StainConfig` | `channel_label` (`"DAPI"`), brightness, `edge_width`, offset and noise |
| `SampleSceneConfig` | `fov`, the amplicon-scene template of every FOV; `cells` and `stain` (`None` leaves them out); `molecules_per_cell` (8); `ambient_density` of molecules between cells (0.002 per µm³) |

* The overlap must exceed twice the largest random offset plus one voxel.
* The template's `shape_zyx` must equal `fov_shape_zyx`. Molecule placement belongs to
  the scene, so the template sets no count or placement, and loss, dropout and weakening
  stay off.
* For one plane, set both shapes to Z of 1 and the Z offset to 0.

## Limits

* FOVs move by translation only, in one layer. There is no stage rotation, no
  illumination difference between FOVs and no bleaching in overlaps.
* Cells are smooth ellipsoids with simple territories, and molecules are spread
  uniformly inside them. The scene is not fitted to tissue and says nothing about how
  well a segmenter would do on real data.
* There is no limit on size; large grids need matching memory and time.

The full specification is in
[`design/specs/synthetic/sample-scene.md`](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/synthetic/sample-scene.md).
