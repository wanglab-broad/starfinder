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
   `Dataset.stitch` reads saved images: `reference_merged`, `reference_stain`, or
   a `file:<pattern>` path with `{fovID}` substitution.
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
`bigstitcher_save_dir` names the local working folder and `fiji_path` the Fiji
executable. See {doc}`workflow configuration <workflow-configuration>`.

## What you get

`PlacementResult.fovs` holds FOV positions, statuses and flags. `links` holds
pairwise shifts, quality and residuals; `record` holds the sample frame, method,
inputs and review. Positions are in µm, ZYX, in `sample:<sample>`.

`save_placement` writes `placement.json`, `fovs.csv` and `links.csv` into a
placement folder, conventionally `stitching/<sample>/`; a BigStitcher import also
saves `view_map.csv`. `load_placement` reads the folder. The
{doc}`reference <stitching-algorithms>` lists the columns and workflow keys.

## Main options

* `input` on `Dataset.stitch`: choose the saved image used for placement.
* `stage_axes` on `positions_from_leica`: map the acquisition axes to image axes;
  `snap_to_grid` requests a regular layout.
* `binning_zyx` on `MultiviewStitcherConfig`: choose the image scale used to register
  overlaps; `quality_threshold` controls link filtering.
* `review`: choose the `ReviewPolicy` used to flag a placement for inspection.
* `fallback`: request nominal positions when image placement cannot connect a FOV.

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
{doc}`validation design <stitching-assembly-validation>` describes the checks of
stitching, assembly and sample outputs.

The full [stitching specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/stitching.md)
gives the detailed rules.

```{toctree}
:hidden:

Stitching reference <stitching-algorithms>
```
