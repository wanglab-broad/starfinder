# Stitching reference

Methods, configurations and placement columns for the {doc}`stitching guide
<stitching-contract>`. The {doc}`stitching API <api/stitching>` documents the
functions and classes.

## Methods and configurations

The method registry supplies this table at build time.

```{method-table} starfinder.stitching.STITCHING_METHODS
```

### Placement in Python

`multiview_stitcher` registers overlapping images by phase correlation and resolves
translations by global optimization. See
{py:class}`~starfinder.stitching.MultiviewStitcherConfig`.

```{config-options} starfinder.stitching.MultiviewStitcherConfig
```

### BigStitcher

`export_bigstitcher` prepares a Fiji run; `import_bigstitcher` reads its
`dataset.xml`. See {py:class}`~starfinder.stitching.BigStitcherConfig`.

```{config-options} starfinder.stitching.BigStitcherConfig
```

### Review and stage conversion

See {py:class}`~starfinder.stitching.ReviewPolicy`,
{py:class}`~starfinder.stitching.StageAxisMapping` and
{py:class}`~starfinder.stitching.GridSnap` for their parameter details.

```{config-options} starfinder.stitching.ReviewPolicy
```

```{config-options} starfinder.stitching.StageAxisMapping
```

```{config-options} starfinder.stitching.GridSnap
```

## Position table

`TilePositionTable.positions` contains the nominal positions in µm.

| Column | Contents |
| --- | --- |
| `fov_id` | FOV identity |
| `position_id` | Acquisition position identifier, when available |
| `nominal_z`, `nominal_y`, `nominal_x` | Nominal origin in the sample frame |
| `source` | `stage`, `grid` or `supplied` |
| `grid_column`, `grid_row` | Grid cell, when available |

## Placement tables

`PlacementResult.fovs` describes the placement of the FOVs.

| Column | Contents |
| --- | --- |
| `fov_id` | FOV identity |
| `status` | `image_supported`, `metadata_only`, `placeholder`, `missing` or `failed` |
| `origin_z`, `origin_y`, `origin_x` | Sample-frame position of reference-grid index `(0, 0, 0)`, µm |
| `nominal_z`, `nominal_y`, `nominal_x` | Nominal position, µm |
| `correction_z`, `correction_y`, `correction_x` | Origin minus nominal position, µm |
| `method_translation_z`, `method_translation_y`, `method_translation_x` | Method result before gauge adjustment |
| `component` | Component of the kept-link graph |
| `n_links_kept` | Number of kept links for the FOV |
| `flags` | Flags separated by semicolons |

`PlacementResult.links` describes the neighbour pairs.

| Column | Contents |
| --- | --- |
| `fov_a`, `fov_b` | FOV identities |
| `overlap_z0`, `overlap_z1`, `overlap_y0`, `overlap_y1`, `overlap_x0`, `overlap_x1` | Nominal overlap box, µm |
| `shift_z`, `shift_y`, `shift_x` | Pairwise shift of `fov_b` against `fov_a`, µm |
| `quality` | Link quality |
| `kept` | Whether the link was kept |
| `reason` | Why a link was dropped |
| `residual` | Link residual, µm |

## Saved files

`save_placement` writes these files inside the placement folder, conventionally
`stitching/<sample>/`.

| File | Contents |
| --- | --- |
| `placement.json` | Frame, method provenance, inputs, review and table hashes |
| `fovs.csv` | The per-FOV placement table |
| `links.csv` | The neighbour-link table |
| `view_map.csv` | BigStitcher import: `view`, `grid_column`, `grid_row`, `fov_id`, `kind` |

## Workflow keys

The Python workflow's `stitching` block exposes these keys; see
{doc}`workflow configuration <workflow-configuration>` for its surrounding setup.

| Key | Purpose |
| --- | --- |
| `method` | Placement backend |
| `config` | Method configuration, from the generated option tables above |
| `input` | Saved stitching image |
| `positions` | Nominal positions |
| `review` | Review policy |
| `fallback` | Nominal-position fallback |
| `bigstitcher_save_dir` | Local BigStitcher working folder |

The `positions` mapping exposes the following keys.

| Key | Purpose |
| --- | --- |
| `source` | Position source |
| `stage_file`, `stage_axes`, `snap_to_grid` | Stage file, axis mapping and optional grid snapping |
| `regular_grid` | Grid layout |
| `tile_shape_yx`, `spacing_yx` | Geometry for grid construction |
| `table`, `unit` | Supplied position table and its unit |

The [stitching specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/stitching.md)
and [algorithm specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/stitching-algorithms.md)
give the detailed placement rules.
