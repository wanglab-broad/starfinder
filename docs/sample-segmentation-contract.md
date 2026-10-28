# Sample segmentation

Sample segmentation finds cells on a fused projection, then assigns the sample's
molecules to those cells. Use it for projection samples where cells cross FOV seams.
It sits between {doc}`image fusion <sample-output-contract>` and the sample tables
written by {doc}`assembly <assembly-contract>`.

`Dataset.export` also supports segmentation per FOV followed by assembly. Choose the
route with `ExportConfig.route`; the {doc}`workflow guide <workflow-downstream>` shows
how to make that choice in a file-based run.

## Quick start

Build the settings for a nuclear segmentation of the fused reference stain:

```python
from starfinder.export import (
    ExportConfig, FusionConfig, SampleSegmentationConfig, SampleSegmentationRun,
)
from starfinder.segmentation import StarDistConfig

config = ExportConfig(
    route="after_fusion",
    fusion=FusionConfig(images=("reference_stain",)),
    segmentation=SampleSegmentationConfig(
        runs=(SampleSegmentationRun(
            name="nuclei",
            target="nucleus",
            inputs=(("nuclear", "reference_stain"),),
            method=StarDistConfig(model="2D_versatile_fluo", scale=1.0),
        ),),
        cells="nuclei",
        nuclei="nuclei",
    ),
)
```

Here the nucleus labels serve as the cell territories. The scale illustrates the API;
choose it for your images using the {doc}`segmentation guide <segmentation-contract>`.

The following is an excerpt: it needs a prepared dataset, saved placement, molecule
tables and locally available model weights.

```python
record = dataset.export(
    sample, placement=placement_folder, config=config, output_dir=export_folder,
)
```

For workflow runs, put these settings in `export.segmentation`, with the route in
`export.route`. The top-level `assignment` block supplies the assignment settings.
The [workflow configuration record](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/workflow-configuration.md#stitching-assembly-and-export) has complete examples.

## What it does

1. **Fuse the stains.** The export makes the fused projection from the images selected
   by `FusionConfig.images`.
2. **Segment.** Named runs compute labels from the fused stains, or import a prepared
   mask. A cell run can use a nucleus run as seeds.
3. **Assign.** Assembly collects the sample's molecules and assigns them to the sample
   segmentation. The sample molecule table keeps the source spot identity.
4. **Write the sample.** The export saves the assembly tables and `sample.h5ad`, and
   can also write an inspection label image and a SpatialData store.

The {doc}`segmentation methods <segmentation-algorithms>` describe StarDist, Cellpose
and seeded watershed. Choose a method for the stains you have; the
{doc}`sample-segmentation reference <sample-segmentation-reference>` lists the settings.

## What you get

`Dataset.export` returns an {py:class}`~starfinder.export.ExportRecord`. Its folder
holds `export.json`, sample tables and images. The segmentation adds
`segmentation_<run>.ome.tif` label images; assignment expansion can add
`territories.ome.tif`. Label arrays use ZYX order and `uint32`.

`load_sample_segmentation(export_folder)` returns a
{py:class}`~starfinder.segmentation.SampleSegmentation` with its grid, named runs and
record. The `segmentation` section of `export.json` records input hashes, the grid,
method settings, label files and resource use. The reference lists the files and the
main record fields.

## Main options

* `route`: choose segmentation per FOV or after fusion.
* `images`: select the stains to fuse; a run's `inputs` names those images and their
  roles.
* `scale_yx`: adjust the fused image size. Method pixel lengths are given for the
  acquired grid; the record shows the values given and used.
* `runs`, `cells` and `nuclei`: name the segmentations and the labels used for assignment.
* `assignment`: configure molecule assignment and cell expansion.
* `segment`: rerun segmentation or reuse saved labels.
* `device`: select the segmentation device.

The {py:class}`~starfinder.export.ExportConfig` API documents route defaults; the
[specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/sample-segmentation.md#the-route-setting)
gives the route rules.

## Limits

This stage segments a projection, rather than a fused 3D stack. Plan device memory
for the fused projection and the segmentation method; changing `scale_yx` changes the
image size. The {py:class}`~starfinder.segmentation.StarDistConfig` API describes its
block prediction option. Learned methods need their optional dependencies and local
model weights.

`compare_routes` compares two saved exports of the same sample, one from each route.
It reports cell overlaps, matched-cell counts and molecule assignments; agreement is
not a measure of accuracy. Its `output` argument saves comparison tables separately
from the exports. See {py:func}`~starfinder.export.compare_routes` for usage.

## How it is checked

Small synthetic fixtures exercise the route setting, segmentation, assignment and
output readers. Hand-built cases check the route comparison, including cells at FOV
seams. The [validation design](https://github.com/wanglab-broad/starfinder/blob/dev/design/validation/sample-segmentation.md)
describes those checks.

The full [specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/sample-segmentation.md)
contains the detailed rules.

```{toctree}
:hidden:

Sample-segmentation reference <sample-segmentation-reference>
```
