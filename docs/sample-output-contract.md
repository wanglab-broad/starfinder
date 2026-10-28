# Sample output

{py:meth}`~starfinder.dataset.Dataset.export` writes a sample's assembled tables,
fused images and optional label image and SpatialData store into one export folder.
It reads the saved FOV outputs and a placement from {doc}`stitching <stitching-contract>`;
{doc}`assembly <assembly-contract>` supplies the sample tables.
The {doc}`processing report <report-contract>` can then describe the saved export.

## Quick start

With a `dataset`, a saved placement in `placement_dir`, and saved per-FOV
assignments named `default`, export the reference images and cell labels:

```python
from starfinder.export import ExportConfig, FusionConfig, LabelConfig, check_export

export_dir = dataset.output_root / "export" / "sample1"
result = dataset.export(
    "sample1", placement=placement_dir, output_dir=export_dir,
    config=ExportConfig(
        route="per_fov", fuse_labels=True,
        fusion=FusionConfig(images=("reference_merged",), scale_yx=1.0),
        labels=LabelConfig(scale_yx=1.0),
    ),
)
check_export(export_dir)
print(result.parts, result.complete)
```

The {doc}`multi-FOV demo <get-started/multi-fov-demo>` builds the inputs, exports
them and plots the fused image with its labels and molecules.

For the Python workflow, the corresponding export settings are:

```yaml
export:
  route: per_fov
  images: [reference_merged]
  scale_yx: 1.0
  fuse_labels: true
  labels_scale_yx: 1.0
```

This block goes into a workflow with stitching and assembly configured;
{doc}`workflow configuration <workflow-configuration>` gives the complete setup.

## What it does

On the route per FOV:

1. **Assemble.** Combine the saved observations into sample tables with sample
   coordinates, ownership diagnostics and counts.
2. **Fuse.** Use multiview-stitcher to combine placed source images into a
   projection, a stack or both.
3. **Export labels.** When requested, write a sample label image for inspection.
4. **Export SpatialData.** When requested, package the sample tables and available
   images into a SpatialData store.

`route="per_fov"` in the example uses the saved per-FOV assignments. The
{doc}`sample-segmentation guide <sample-segmentation-contract>` describes
`route="after_fusion"`, which segments the fused projection before assembly.

## What you get

In memory, an {py:class}`~starfinder.export.ExportRecord` gives `parts`,
`complete` and the full `record`. On disk, `export.json` describes the folder;
the output profile is the combination of parts and image products you request.

| Product | Where to find it |
| --- | --- |
| Sample tables and counts | `tables/` and `sample.h5ad` |
| Fused projection | `<image>_projection.ome.tif` |
| Fused stack | `<image>_stack.ome.tif` |
| Cell labels for inspection | `cell_labels.ome.tif` |
| SpatialData export | `sample.zarr/` |
| Standalone OME-Zarr fusion output | `<image>_<dimensionality>.ome.zarr/` |

The {doc}`sample-output reference <sample-output-reference>` lists layouts,
record fields and coordinate conventions. Use the saved spacing and origin to
convert a molecule's sample position to an image index.

## Main options

* `fuse`, `fuse_labels` and `export_spatialdata` select image and store products.
* `images` selects the saved reference stain, merged reference image or named files.
* `dimensionality` chooses a projection, a stack for viewing depth, or both.
* `projection` chooses maximum or summed intensity along Z.
* `scale_yx` changes the fused image's Y and X resolution; `labels.scale_yx`
  changes the label image's resolution independently.
* `ome_zarr` requests standalone OME-Zarr output; `pyramid_levels` adds OME-TIFF
  resolution levels.

See the reference's generated options and {py:class}`~starfinder.export.ExportConfig`
for defaults and details.

## Limits

Export reads saved processing results; it does not run the FOV pipeline or estimate
the placement. The label image is an inspection product, while the sample tables
remain the source for counts. Image fusion and label fusion use the
`multiview-stitcher` extra; SpatialData uses the `spatialdata` extra, and saving
the sample AnnData uses `anndata`.

A part that cannot run can leave an incomplete export. Read `parts`, `complete`
and the recorded reason alongside any `ExportIncompleteWarning`.

## How it is checked

`check_export` reads the saved products back against `export.json`, checking file
hashes, image geometry and table relationships. Synthetic checks exercise the
export entry, fusion, labels and SpatialData in the
[validation design](https://github.com/wanglab-broad/starfinder/blob/dev/design/validation/stitching-assembly.md).

[Full specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/sample-output.md).

```{toctree}
:hidden:

Sample-output reference <sample-output-reference>
```
