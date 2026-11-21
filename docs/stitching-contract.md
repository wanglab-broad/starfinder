# Stitching

Stitching places the FOVs of one sample in a shared physical frame. It estimates
translations from overlapping images, imports a BigStitcher result, or records a
placement you already know. {doc}`Assembly <assembly-contract>` uses the placement
to put molecules and cells into sample coordinates; {doc}`sample outputs
<sample-output-contract>` provide fusion and export.

## Quick start

This example places the reference-round images of a small synthetic sample. It
uses the `multiview-stitcher` extra.

```python
from starfinder.segmentation import ReferenceGrid
from starfinder.stitching import (
    MultiviewStitcherConfig, StitchingInput, positions_from_table, stitch,
)
from starfinder.synthetic import generate_sample_scene, sample_scene_preset

codebook, config = sample_scene_preset("sample-tiny-v1")
scene = generate_sample_scene(codebook, config=config)
table = positions_from_table(scene.positions)
reference = config.fov.geometry.reference_round or codebook.round_labels[0]
inputs = [
    StitchingInput(
        fov, scene.fovs[fov].rounds[reference].max(axis=-1),
        ReferenceGrid(config.acquisition.fov_shape_zyx,
                      scene.fov_metadata[fov], "declared"),
        "reference_merged",
    )
    for fov in table.fov_ids
]
placement = stitch(
    inputs, table, sample=scene.sample,
    config=MultiviewStitcherConfig(binning_zyx=(1, 1, 1), quality_threshold=-1.0),
)
print(placement.fovs[["fov_id", "status", "origin_z", "origin_y", "origin_x"]])
print(placement.record["review"])
```

The {doc}`multi-FOV demo <get-started/multi-fov-demo>` shows
{py:meth}`~starfinder.dataset.Dataset.stitch` reading saved FOV images, saving the
placement and continuing through assembly and export.

## What it does

1. **Read nominal positions.** Build a `TilePositionTable` from a Leica LAS X stage
   file with `positions_from_leica`, a regular grid with `positions_from_grid`, or
   a CSV or DataFrame with `positions_from_table`.
2. **Prepare images.** `StitchingInput` pairs a ZYX image with its `ReferenceGrid`.
   The image is an array or a `TileLoader`, a function that reads the tile when
   the method needs it, with its dtype declared. `Dataset.stitch` reads saved
   images: `reference_merged`, `reference_stain`, or a `file:<pattern>` path with
   `{fovID}` substitution. It checks each file against its grid from the file's
   header and passes loaders, so no tile is held before placement starts.
3. **Place FOVs.** `neighbor_pairs` describes overlaps of the nominal FOV boxes.
   `stitch` runs the configured placement method and records its links and FOV
   positions.
4. **Inspect the result.** Read the per-FOV status and flags, the links, and
   `record["review"]`. After inspecting a placement, `approve_placement` records
   your review for assembly.

For a known placement, use {py:func}`~starfinder.stitching.placement_from_table`
with the table and reference grids; `origins` supplies positions separately from
the nominal ones. This produces a `PlacementResult` for the same assembly entry.

## Methods

Use `multiview_stitcher` for placement in Python; use the BigStitcher workflow
below to run placement in Fiji.

```{method-table} starfinder.stitching.STITCHING_METHODS
```

### BigStitcher

{py:func}`~starfinder.stitching.export_bigstitcher` writes the tile TIFFs and
returns macro text and a view map. Save the macro and view map, run the macro in
Fiji, then pass its `dataset.xml`, view map, table, reference grids and configuration to
{py:func}`~starfinder.stitching.import_bigstitcher`. The import returns a
`PlacementResult`. The Python-only workflow's `stitching` block coordinates the
export, Fiji engine task and import with `method: bigstitcher`;
`bigstitcher_save_dir` names the local working folder and `execution.fiji` the Fiji
executable; only Snakemake runs the Fiji task. See the {doc}`workflow reference <workflow/reference>`.

BigStitcher treats tiles of one plane (Z = 1) as 2D. For them the export replaces the default
HDF5 levels and pairwise downsample by levels `(1, 1, 1)`, `(2, 2, 1)`, `(4, 4, 1)` (chunks
16 × 16 × 1) and a downsample of `(1, 1, 1)`; fields you set yourself are kept, and tiles of
more than one plane keep today's macro. Before writing any file the export raises
`ValueError` when no resaved level serves the requested downsample, naming the fix, since Fiji
would otherwise fail with `Matrix is singular` and still exit 0.

## What you get

`PlacementResult.fovs` holds FOV positions, statuses and flags. `links` holds
pairwise shifts, quality and residuals; `record` holds the sample frame, method,
inputs and review. Positions are in µm, ZYX, in `sample:<sample>`.

`save_placement` writes `placement.json`, `fovs.csv` and `links.csv` into a
placement folder, conventionally `stitching/<sample>/`; a BigStitcher import also
saves `view_map.csv`. `load_placement` reads the folder. The
{doc}`reference <stitching-algorithms>` lists the columns and workflow keys.

### Reading `links.csv`

Each row is one pair of overlapping FOVs. `kept` is true when the link was used for
placement; otherwise `reason` says why it was not:

* `pruned`: the pair was not registered (only some pairs are, to save time); its shift and
  quality are empty.
* `quality_below_threshold`: the pair was registered, but its `quality`, the rank correlation
  of the two overlaps after the shift, is below the threshold.
* `nan_quality`: one overlap is constant, so the quality is undefined.
* `quality_higher_at_z_offset`: with `quality_threshold` `"auto"` on tiles with more than one
  plane, the link passed the threshold, but its quality is higher a few planes up or down than
  at its registered shift, so the link is probably off in z.
* `optimizer_removed`: the link passed the threshold, but the global optimization dropped it
  because it disagreed with the other links.

`residual` (µm) is the disagreement between a kept link's shift and the final positions. With
`"auto"`, `placement.json` holds the threshold, the qualities at wrong offsets and, for tiles
with more than one plane, each link's z check under `method_diagnostics.link_quality`. Links the optimizer removed stay in the table, and
their FOVs carry the flag `optimizer_removed_link`. They make the review `required` only when
the removals leave the placement less well supported: a component splits, a FOV keeps fewer
than 2 links (and fewer than it had), or a residual or correction check fails.
`record["review"]["optimizer_removed"]` lists the removed links and each condition.

## Main options

* `input` on `Dataset.stitch`: choose the saved image used for placement.
* `stage_axes` on `positions_from_leica`: map the acquisition axes to image axes;
  `snap_to_grid` requests a regular layout.
* `binning_zyx` on `MultiviewStitcherConfig`: choose the image scale used to register
  overlaps. From loaders, the method keeps only the binned tiles and holds at most one
  full-resolution tile at a time; with `binning_zyx=(1, 1, 1)` it keeps every tile at
  full resolution.
* `quality_threshold` on `MultiviewStitcherConfig`: the lowest link quality kept. The
  default estimates it from the sample (`"auto"`) when every tile is single-plane and is 0.7
  for tiles with more than one plane (the {ref}`reference <stitching-link-threshold>` gives
  the rule). An explicit `"auto"`, or a number from −1 to 1, applies to either kind.
* `review`: choose the `ReviewPolicy` used to flag a placement for inspection.
* `fallback`: request nominal positions when image placement cannot connect a FOV.

### Choosing the link-quality threshold

Without `quality_threshold`, single-plane tiles (Z = 1) get `"auto"` and tiles with more than
one plane get 0.7. On stacks whose tiles share one z range, such as tissue-2D (Z = 30), name
`quality_threshold: auto` in the configuration: on tissue-2D the fixed 0.7 rejected real links
and left most FOVs unplaced, while `auto` reproduced the approved placement. The threshold of
`auto` comes from the quality at wrong offsets in Y and X only, which cannot catch a link that
is off in z. So on tiles with more than one plane, `auto` also measures each link about 1 µm up
and down in z and drops a link that scores higher there than at its own shift. On the full LN
sample (W-904) that check passed every link, but `auto`'s in-plane threshold (0.91 there)
dropped 62 real links that 0.7 keeps, so the default for these tiles stays 0.7.

The reference generates configuration defaults from the code. For the
[link-quality rule](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/stitching.md#link-quality-rule)
and [failure and review policy](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/stitching.md#failure-policy),
see the specification and {py:func}`~starfinder.stitching.stitch`.

## Limits

Stitching estimates translations, without rotation or deformation. It does not
run per-FOV processing or fuse images. `Dataset.stitch` starts from saved
per-FOV artifacts; placement quality still needs inspection on your sample.

## How it is checked

Engineering checks cover position conversion, placement records, BigStitcher
export and import, and comparison with known synthetic positions. The shared
[validation design](https://github.com/wanglab-broad/starfinder/blob/dev/design/validation/stitching-assembly.md) describes the checks of
stitching, assembly and sample outputs.

The full [stitching specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/stitching.md)
gives the detailed rules.

```{toctree}
:hidden:

Stitching reference <stitching-algorithms>
```
