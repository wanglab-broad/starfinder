# Sample segmentation

Sample segmentation finds cells on a fused projection, then assigns the sample's
molecules to those cells. Use it for projection samples where cells cross FOV seams.
It sits between {doc}`image fusion <sample-output-contract>` and the sample tables
written by {doc}`assembly <assembly-contract>`.

`Dataset.export` also supports segmentation per FOV followed by assembly. Choose the
route with `ExportConfig.route`; the {doc}`workflow reference <workflow/reference>` lists
the `export` keys that make that choice in a file-based run.

## Quick start

Build the settings for a nuclear segmentation of the fused reference stain:

```python
from starfinder.export import (
    ExportConfig, FusionConfig, SampleSegmentationConfig, SampleSegmentationRun,
)
from starfinder.preprocessing import ProjectionConfig
from starfinder.segmentation import InputRecipe, ModelInputConfig, StarDistConfig

recipe = InputRecipe(projection=ProjectionConfig(method="max"), model_input=ModelInputConfig(pixel_size_um=0.4))
config = ExportConfig(
    route="after_fusion",
    fusion=FusionConfig(images=("reference_stain",)),
    segmentation=SampleSegmentationConfig(
        runs=(SampleSegmentationRun(
            name="nuclei",
            target="nucleus",
            inputs=(("nuclear", "reference_stain"),),
            method=StarDistConfig(model="2D_versatile_fluo"),
            input_recipe=recipe,
        ),),
        cells="nuclei",
        nuclei="nuclei",
    ),
)
```

Here the nucleus labels serve as the cell territories. The pixel size illustrates the API;
choose it for your images using the {doc}`segmentation guide <segmentation-contract>`. The
same recipe gives a run per FOV and a run after fusion the same pixel size for the model.

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
   mask. A run's input recipe runs on the fused projections, with the included FOVs'
   footprints as its fit region, so a percentile or a contrast mapping is fitted once over
   the sample. A cell run can use a nucleus run as seeds.
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
`territories.ome.tif`. Label arrays use ZYX order and `uint32`. A run whose recipe changes
the fused image also writes its model input, `model_inputs/<run>.ome.tif`, in a subfolder
that only model inputs use; a run whose recipe leaves the image as it is names the fused
file instead.

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
  acquired grid; the record shows the values given and used. A recipe's lengths are in
  micrometres and its `model_input` gives the pixel size the model sees, so neither needs
  changing with `scale_yx`.
* `input_recipe`: a run's input recipe, a name of `segmentation.input_recipes` in a
  workflow configuration or a recipe inline.
* `runs`, `cells` and `nuclei`: name the segmentations and the labels used for assignment.
* `assignment`: configure molecule assignment and cell expansion.
* `segment`: rerun segmentation or reuse saved labels.
* `device`: select the segmentation device.

The {py:class}`~starfinder.export.ExportConfig` API documents route defaults; the
[specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/sample-segmentation.md#the-route-setting)
gives the route rules.

## Limits

This stage segments a projection, rather than a fused 3D stack. With tiles of more than
one plane, a recipe's `channels` steps and a `sum` or `mean` projection are refused: they
cannot be recovered from the Z maximum; give 2D steps in `after_projection`. Plan device memory
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
