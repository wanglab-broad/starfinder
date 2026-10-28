# Sample-output reference

Files, fields and coordinates of sample export. The
{doc}`guide <sample-output-contract>` explains how to export;
the {doc}`API <api/export>` documents the functions and classes.

## Output profile options

The profile combines export parts with fusion, label and store settings.
See {py:class}`~starfinder.export.ExportConfig` for details.

```{config-options} starfinder.export.ExportConfig
```

### Fusion

See {py:class}`~starfinder.export.FusionConfig`.

```{config-options} starfinder.export.FusionConfig
```

### Labels

See {py:class}`~starfinder.export.LabelConfig`.

```{config-options} starfinder.export.LabelConfig
```

### SpatialData

See {py:class}`~starfinder.export.SpatialDataExportConfig`.

```{config-options} starfinder.export.SpatialDataExportConfig
```

## Export folder

The workflow writes `export/<sample>/` under its output root;
`Dataset.export` takes an explicit `output_dir`.

| Path | Contents |
| --- | --- |
| `export.json` | Export record |
| `tables/fovs.<fmt>`, `tables/cells.<fmt>`, `tables/molecules.<fmt>`, `tables/counts.<fmt>` | Sample tables, CSV or Parquet; columns in the {doc}`assembly reference <assembly-algorithms>` |
| `sample.h5ad` | Sample AnnData; fields in the assembly reference |
| `<image>_projection.ome.tif` | Fused 2D projection |
| `<image>_stack.ome.tif` | Fused stack |
| `<image>_<dimensionality>.ome.zarr/` | Optional standalone OME-Zarr fusion output |
| `cell_labels.ome.tif` | Optional uint32 instance labels |
| `segmentation_<run>.ome.tif` | Segmentation after fusion; see {doc}`sample-segmentation reference <sample-segmentation-reference>` |
| `territories.ome.tif` | Expanded sample territories after fusion, when present |
| `sample.zarr/` | Optional SpatialData store |

### Image layouts

The record describes the geometry and layout of written images.

| Output | Layout and attributes |
| --- | --- |
| OME-TIFF | BigTIFF, one YX or ZYX series, 512 × 512 zlib tiles; optional SubIFD resolution levels |
| OME-TIFF geometry | `PhysicalSizeX/Y/Z`, `Plane` positions, units µm; `starfinder.image.ImageMetadata` comment annotation restored by `load_volume` |
| OME-TIFF export annotation | Namespace `starfinder.sample_output`; fusion fields `sample`, `image`, `dimensionality`, `projection`, `factor_yx`, `placement_fovs_sha256`, `placement_links_sha256` |
| Standalone OME-Zarr | NGFF multiscales with axes, datasets and coordinate transformations; per-level shape, dtype, chunks, compression and geometry recorded in `fusion.outputs` |
| Stack and maximum projection | uint8, uint16 or float32, matching the input dtype |
| Sum projection | float32 |
| Label export | uint32; `instance_id` identifies a row of the sample cell table |

### Export record

`ExportRecord.record` is the parsed record; `parts` and `complete` expose its
summary. Part statuses are `written`, `reused`, `off`, `empty`, `not_run` and `failed`.

| Field or section | Contents |
| --- | --- |
| `format_version`, `stage`, `sample`, `frame_id` | Record version, export stage and sample identity |
| `parts`, `complete` | Status by part and completion summary |
| `viewer_qualification` | Viewer qualification status |
| `assembly` | Sample-table record and file hashes |
| `fusion` | Config, placement, FOV sources, library, fusion grid, outputs, software and resources |
| `segmentation` | Segmentation after fusion and its input and label-file hashes |
| `labels` | Label config, sources, grid, output, input/output IDs, absent IDs and conflicts |
| `spatialdata` | Elements, omissions, formats, counts, source hashes and store-file hashes |

Fusion output entries describe the products listed below.

| Fields | Contents |
| --- | --- |
| `path`, `sha256`, `format`, `image`, `dimensionality` | File identity and product |
| `projection`, `factor_yx`, `shape_zyx`, `dtype` | Projection choice, scale factor and array description |
| `spacing_zyx`, `origin_zyx` | Output geometry |
| `slab_z` | TIFF output's Z slab, or null for a multi-plane stack |
| `levels`, `compression` | Resolution levels and storage layout |
| `tile_yx` (TIFF), `chunks` (OME-Zarr) | Tile or chunk shape |
| `files` | Standalone OME-Zarr's relative file paths and hashes |

## Fused-image coordinates

Use the output level's geometry, available from `load_volume` or the fusion
output entry. {py:class}`~starfinder.export.FusedGrid` describes the grid and
its levels.

| Quantity | Convention |
| --- | --- |
| Array and geometry components | ZYX; a projection loaded with `load_volume` retains singleton Z |
| Frame and unit | `sample:<sample>`, µm |
| `origin_zyx`, O | Sample position of voxel centre (0, 0, 0) |
| `spacing_zyx`, S | Distance between voxel centres |
| Voxel centre at index v | Sample position O + v ⊙ S |
| Sample position q | Continuous image index (q − O) ⊘ S |
| Projection Z | One slab with its centre and thickness recorded as Z origin and spacing |

The [specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/sample-output.md#sample-position-to-image-position)
gives voxel support and scale-conversion rules.

## SpatialData elements

`sample.zarr` uses the sample name as its coordinate system, in µm. Store
attributes include `frame_id` and `coordinate_system`. The record lists written
and omitted elements.

| Element | Columns or attributes |
| --- | --- |
| Points `molecules` | Molecules with finite sample coordinates; sample X/Y/Z become `x/y/z`, local indices become `fov_index_x/y/z`; `cell_instance_id` adds the cell link |
| Shapes `cell_centroids` | Circles at finite ownership-centroid Y/X, `radius` from original territory size, index `instance_id` |
| Table `cells` | Sample cell rows with finite ownership-centroid Y/X as `obs`, with `region` and `instance_id` |
| Table `expression` | Sample AnnData with added `obs.region` and `obs.instance_id` |
| Tables `cells_unplaced`, `molecules_unplaced` | Rows without finite ownership-centroid Y/X (cells) or sample Z/Y/X (molecules); added `instance_id` or `cell_instance_id`, respectively |
| Images `<image>_projection`, `<image>_stack` | Fused output level with a singleton channel axis; 2D axes CYX, 3D axes CZYX |
| Labels `cell_labels` | Exported label output level, 2D YX or 3D ZYX |
| Annotating-table metadata | `uns["spatialdata_attrs"]`: `region="cell_centroids"`, `region_key="region"`, `instance_key="instance_id"` |
| Raster transform | Scale S, then translation O − S/2, on the element's spatial axes |

## Workflow keys

The Python workflow's `export` block maps to these public configurations.
The {doc}`workflow guide <workflow-configuration>` shows complete configurations.

| Keys | API counterpart |
| --- | --- |
| `assemble`, `fuse`, `fuse_labels`, `export_spatialdata`, `route`, `segment`, `segmentation` | `ExportConfig` fields |
| `images`, `dimensionality`, `projection`, `scale_yx`, `ome_zarr`, `ome_zarr_version`, `pyramid_levels` | `FusionConfig` fields |
| `labels_scale_yx` | `LabelConfig.scale_yx` |
| `spatialdata_zarr_format` | `SpatialDataExportConfig.zarr_format` |
| `assembly` block | Assembly settings, assignment name and table format; see the assembly reference |
