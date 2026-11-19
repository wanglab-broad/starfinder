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

The method builds the overlap graph from the tiles' grids, without pixels. It then
reads each tile of a registered pair once, in table order, bins it by `binning_zyx`
into the window means the library's binning gives, keeps the binned tile and
releases the full-resolution one. For an integer tile the means come from exact
integer sums, which is faster and gives the same values. A `TileLoader` input, which `Dataset.stitch` passes, is called
at that point, so at most one full-resolution tile is held at a time; an array input
stays held by its caller. The pairs are registered on the binned tiles. Each link's
overlap box, whose corners the global optimization uses, is recomputed from the
full-resolution geometry with the library's private `registration._get_overlap_bboxes`.
A test pins that function. The links and translations are therefore those of the
library binning each pair itself, bit for bit.

(stitching-link-threshold)=
#### The link-quality threshold

The default, `quality_threshold=None`, depends on the tiles: `"auto"` when every tile is
single-plane (Z = 1), and 0.7 when tiles have Z > 1. The wrong offsets below move the overlap
in Y and X only, so they cannot see a z error, and a weak 3D link can pass a threshold estimated
this way (W-895 tracks a z-aware null). An explicit `"auto"`, or a number, applies to either kind.

With `"auto"`, the threshold comes from the sample. For each
registered pair the method also measures the quality at two deliberately wrong offsets: the
registered shift moved by half the overlap's width, in binned voxels rounded down, both ways
along the overlap axis (the narrower of Y and X). There no tissue lines up, so these qualities
are a null distribution. The threshold is the larger of 0.2 and the null's 99th percentile, or
0.2 when there is no null value. A link is kept when its quality is at least the threshold and
not NaN. A number replaces the rule and no null is measured. The global optimization then
removes links that disagree with the rest, as before.

`placement.json` records the rule that set the threshold. A given number is the method
config's `quality_threshold`, and `method_diagnostics` has no `link_quality` entry. The default for
tiles with Z > 1 is recorded as `method_diagnostics.link_quality` =
`{"rule": "fixed_default_z_above_1", "threshold": 0.7}`. With `"auto"`, `link_quality` holds:

| Key | Contents |
| --- | --- |
| `rule` | `auto` |
| `threshold` | The threshold used |
| `floor`, `quantile` | 0.2 and 0.99 |
| `offsets` | How the wrong offsets were chosen |
| `null` | `pairs` and `values` counted, and `min`, `median`, `quantile` (the 99th percentile) and `max` when there are values |
| `null_pairs` | Per registered pair: `fov_a`, `fov_b` and its two null qualities (`null` where there is none) |

```{config-options} starfinder.stitching.MultiviewStitcherConfig
```

### BigStitcher

`export_bigstitcher` prepares a Fiji run; `import_bigstitcher` reads its
`dataset.xml`. See {py:class}`~starfinder.stitching.BigStitcherConfig`.

```{config-options} starfinder.stitching.BigStitcherConfig
```

### Review and stage conversion

Review and position-conversion settings. See
{py:class}`~starfinder.stitching.ReviewPolicy`,
{py:class}`~starfinder.stitching.StageAxisMapping` and
{py:class}`~starfinder.stitching.GridSnap` for the API reference.

```{config-options} starfinder.stitching.ReviewPolicy
```

Links removed by the optimizer need review only when a condition fails: the components
before the removals equal those after, every FOV keeps 2 or more kept links (or all it
had), and the residual and correction checks pass. When there are removals,
`record["review"]["optimizer_removed"]` holds `links` (the removed pairs),
`review_required` and `checks`, each of `components_unchanged`, `fovs_keep_links`,
`residuals_within_tolerance` and `corrections_within_bound` with whether it holds.

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
| `placement.json` | Frame, method provenance, inputs, review and table hashes; `review.approvals` stays empty |
| `fovs.csv` | The per-FOV placement table |
| `links.csv` | The neighbour-link table |
| `view_map.csv` | BigStitcher import: `view`, `grid_column`, `grid_row`, `fov_id`, `kind` |
| `approvals.json` | The approvals (`approve_placement`), each with the SHA-256 of the `fovs.csv` and `links.csv` it approves; written only once there is one |

Saving an approved placement whose files are already in the folder writes
`approvals.json` alone. Any other save replaces the folder but keeps
`approvals.json`, so an approval counts again whenever the tables are the ones
it approved. `load_placement` puts the approvals of the saved tables into
`record["review"]["approvals"]`.

## Workflow keys

The Python workflow's `stitching` block exposes these keys; see the
{doc}`workflow reference <workflow/reference>` for its surrounding setup.

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
| `regular_grid` | Grid layout; the workflow fills its cells in the FOV list's row order |
| `tile_shape_yx`, `spacing_yx` | Geometry for grid construction |
| `table`, `unit` | Supplied position table and its unit |

The [stitching specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/stitching.md)
and [algorithm specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/stitching-algorithms.md)
give the detailed placement rules.
