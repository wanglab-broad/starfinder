# Stitching guide: tile positions, placement, methods and BigStitcher

% Status: Accepted (W-329, 2026-10-07, at 9033d46)

This page is the maintained guide to the stitching stage and its placement
contract.

% the stitching stage (§2.10)

* **What stitching does.** It places the FOVs of one sample in one physical frame,
  `sample:<sample>` in µm: from each FOV's nominal position and saved stain image it
  registers the neighbour pairs, gives every FOV an origin and exactly one status, and
  flags what needs review before sample assembly; it runs on the CPU, translation only,
  and reads only saved per-FOV files ([Terms](#terms), [Neighbour pairs](#neighbour-pairs),
  [Placement record](#placement-record), [Link-quality rule](#link-quality-rule),
  [Failure policy](#failure-policy), [Not the multi-FOV runner](#not-the-multi-fov-runner)).
* **The entry and its configs.** `stitch` takes one `StitchingInput` per FOV, a
  `TilePositionTable` from one of three converters (a stage file with its
  `StageAxisMapping`, a regular grid or a supplied table) and a config registered in
  `STITCHING_METHODS` (`MultiviewStitcherConfig`), with a `ReviewPolicy` and an opt-in
  fallback; `Dataset.stitch` runs it for one sample from the saved files. BigStitcher, the
  initial default, runs through Fiji as an engine task between `export_bigstitcher` and
  `import_bigstitcher` (`BigStitcherConfig`), and a known placement enters through
  `placement_from_table` ([Names](#names), [The stitch entry](#the-stitch-entry),
  [Tile position table](#tile-position-table), [Input image](#input-image),
  [Method registry](#method-registry), [BigStitcher](#bigstitcher),
  [Supplied placement](#supplied-placement)).
* **What it produces.** One `PlacementResult` for every backend: the per-FOV table
  (status, origin, nominal position, correction, flags), the links with their quality and
  residual, and the record with the frame, gauge, method provenance and review;
  `save_placement` writes it to `<OUTPUT>/stitching/<sample>/` (`placement.json`,
  `fovs.csv`, `links.csv`), which `load_placement` reads back, and assembly refuses it
  while its review is required and not approved
  ([Placement record](#placement-record), [Review criterion](#review-criterion),
  [Persistence](#persistence)).
* **Where the configuration goes.** The method config, `ReviewPolicy` and `fallback`
  passed to `stitch` or `Dataset.stitch`, a sample-level entry outside `PipelineConfig`
  (a settled decision); in the workflow, the Python-only `stitching` block
  ([Workflow configuration](#workflow-configuration)). The `multiview-stitcher` extra and
  the package's Python bound are in [Dependency plan](#dependency-plan).

% a sample-level entry outside `PipelineConfig` (D3)

Beneath this guide, [stitching-algorithms](stitching-algorithms.md) gives the numerical behavior of each
converter, the neighbour derivation, the `multiview_stitcher` method, the import
arithmetic, the supplied placement and the placement-quality checks, with their
resources; [stitching-assembly-validation](../validation/stitching-assembly.md) is the engineering validation design that
stitching shares with assembly and the sample outputs, with its implemented checks and
the results of the real examples; [stitching-baseline](../baselines/stitching.md) is history (`8e7d256`, before
the stitching stage).

% before §2.10

This page specifies the placement contract: a sample-level stitch entry separate
from `PipelineConfig` and `FOV.run`, one tile position table with three converters, an
explicit stage-axis mapping, neighbour pairs from nominal overlap, the stitching input
image, one placement record for every backend, a stitching registry on the shared
mechanism of [method-registry](../../docs/method-registry.md), the BigStitcher macro and import, the failure
policy, persistence and the workflow keys. Its registry fields, lookup, dependency and
provenance rules are those of [method-registry](../../docs/method-registry.md), used unchanged. It uses the
`ReferenceGrid` of the [segmentation specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/segmentation.md#coordination-per-fov) ("Coordination per FOV") and
`ImageMetadata` (`src/python/starfinder/image.py`) as they are, and redefines neither.
It is implemented in `starfinder.stitching`, and nothing is renamed. Sample coordinates of molecules and
cells, overlap ownership, retained and excluded populations and the sample tables belong
to sample assembly ([assembly-contract](assembly.md)), which builds on the placement record below and does not redefine it; the fused
image and the exports belong to the sample outputs ([sample-output-contract](../../docs/sample-output-contract.md)).

% The §2.10 placement contract; sample assembly: W-326; the sample outputs: W-328

Evidence (cited by path; this page repeats only the numbers it needs), kept with the run
records outside the repository:

* The stitching input inventory:
  `20261005T1930Z-fiji-baseline/fiji-baseline.md` (BigStitcher baseline),
  `20261005T2110Z-input-inventory/inputs.md` (calibration),
  `20261005T2050Z-stage-axis-mapping/positions.md` (stage axes, scan order),
  `20261005T2025Z-spike-environment/` (`environment.json`, `lock-diff.txt`,
  `spike-added-packages.txt`, `resolve-python312.txt`).
* The stitching parity runs, whose run directory holds a `stitching-manifest.json`:
  `notes.md` (sections 1 to 5 and 7), `tables/runs.csv`, `tables/comparison_summary.csv`,
  `tables/fixtures_summary.csv`, and the parity outputs under `parity/`.

% The stitching input inventory: W-323, `/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/runs/W-323/`.
% The stitching parity runs: W-324, run directory
% `/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/runs/W-324/20261006T063540Z-5cb2f421`
% (found with `timeout 30 find …/runs/W-324 -maxdepth 2 -name stitching-manifest.json`).
% On this page "the input inventory" is W-323 and "the parity runs" (and their notes) are W-324.

The parity runs are dependency selection: they rank no backend, set no default and make no accuracy
claim. Real data has no placement truth, so agreement between two placements shows only
that they agree. This page makes no accuracy claim either.

% W-324 is dependency selection

## Settled decisions this page follows

Decided by Jiahao in the stitching and assembly planning session (the planning decisions
of 2026-10-05 and their amendment on the sample outputs):

* **Both backends are kept** and BigStitcher stays the initial default; no backend is
  ranked and no default changes here.
* **BigStitcher** runs as an engine task through Fiji. The package generates the macro and
  imports `dataset.xml` into the same placement record; there is no PyImageJ adapter.
* **multiview-stitcher** is an in-process method in `STITCHING_METHODS`; a method takes
  the tiles and nominal positions and returns per-FOV translations and link diagnostics in
  one call.
* **Two sample-level entries** outside `PipelineConfig` and `FOV.run`: `Dataset.stitch`
  and `Dataset.export` (as amended at the review gate of 2026-10-06);
  `Dataset.export` calls `assemble`, `fuse_images`, `fuse_labels` and `export_spatialdata` in
  order ([sample-output-contract](../../docs/sample-output-contract.md), "The export entry"). Ownership regions are
  computed in assemble from the placement.
* **One tile position table**, fed by three converters (Leica stage file, regular grid,
  supplied table), with an explicit stage-axis mapping; placeholder tiles stay inside the
  BigStitcher path.
* **Placement runs on the CPU only.**
* The failure policy agreed on 2026-09-27 (in the planning comments on stitching): successful transforms
  and diagnostics are saved; unreliable or disconnected placement needs review before
  sample assembly; stage-position fallback is opt-in and flagged.
* Amended at the review gate (2026-10-06): a known placement enters through a public
  `placement_from_table` that the caller must approve ("Supplied placement"), and the
  package requires Python 3.11 or later ("Dependency plan").

% Decided in the §2.10 planning session (W-152 comment "§2.10 planning decisions (2026-10-05)" and its reply
% "Amendment on the sample outputs"). Both backends kept, BigStitcher the initial default: D1. BigStitcher through
% Fiji, no PyImageJ adapter: D2. multiview-stitcher in `STITCHING_METHODS`, one call: D3. Two sample-level
% entries: D3, as amended at the W-329 review of 2026-10-06. Placeholder tiles inside the BigStitcher path: D4.
% The failure policy: W-152 comments on §2.10. Amended at the W-329 review (2026-10-06).

Excluded: rotation, affine or deformable stitching, the GPU, a backend ranking, and any
default change.

## Terms

* A **sample** is a set of FOVs placed into one frame: for the Python workflow the rows of
  `fovs.csv` in the input root with that `sample` (for the MATLAB backend a row of
  `documents/sample-annotation.csv`, a FOV range). It is not `Dataset.sample_id`, which
  names an input folder.
* The **sample frame** is the physical ZYX frame, in micrometres, in which every FOV of a
  sample is placed. Its `frame_id` is `sample:<sample>`, its direction is the identity and
  its unit `µm`.
* A FOV's **reference grid** is its `ReferenceGrid` ([segmentation-contract](../../docs/segmentation-contract.md)): the
  shape and `ImageMetadata` of its reference round after `FOV.run`, on which its molecules
  and masks are.
* The **nominal position** of a FOV is where the acquisition puts it in the sample frame,
  before any image is compared: a stage position, a regular-grid position or a supplied
  one.
* A **placement** gives each FOV an **origin**: the sample-frame position of index
  `(0, 0, 0)` of its reference grid. Its **correction** is origin minus nominal position.
* A **link** is one registered neighbour pair, with its measured shift and quality.
* A **component** is a set of FOVs connected by kept links.
* A **padding view** is a blank image that the BigStitcher path adds to complete its
  rectangular grid. It is not a FOV and has no row in the placement record.

## Names

| Option | Names | Effect on the golden values | Effect on existing files and outputs | Effect on the engines |
| --- | --- | --- | --- | --- |
| **N1. A `starfinder.stitching` module (recommended)** | Module `starfinder.stitching`; entry `stitch`; result `PlacementResult`; input `StitchingInput`; registry `STITCHING_METHODS` with `StitchingSpec`; table `TilePositionTable`; dataset coordination `Dataset.stitch` | None: the golden test pins the legacy scripts, which no option edits. | New files only, plus one method on `Dataset` (`dataset/dataset.py`); the exports go into `__all__` and the autosummary list of `docs/api/stitching.rst`, which `docs/check_reference.py` requires. `tile_config_{sample}.csv`, `grid.csv` and `dataset.xml` stay as they are. | The workflow adapter gains `_run_stitching`; the four legacy rules stay. |
| N2. One `starfinder.sample` module for the three sample-level entries | Module `starfinder.sample`; entries `place_fovs`, `assemble`, `fuse_images`; result `SamplePlacement` | None. | The same files; sample assembly and the sample outputs then add to one module, which grows a second responsibility per entry. | The same adapter function under another name. |

% N2: W-326 and W-328 then add to one module

**Recommendation: N1.** The stage is called stitching in the chapter, registries are named
after stages (`SEGMENTATION_METHODS`, `DECODING_METHODS`), and `stitch` reads as the verb
next to `segment` and `assign`. Assemble and fuse keep their own modules or names, which
[assembly-contract](assembly.md) and [sample-output-contract](../../docs/sample-output-contract.md) propose.

% which W-326 and W-328 propose

New names under N1:

| Concept | Name | Kind |
| --- | --- | --- |
| Entry | `stitch(inputs, table, *, config, review=ReviewPolicy(), fallback=None, sample="sample")` | function |
| Result | `PlacementResult` | frozen dataclass |
| Input | `StitchingInput` | frozen dataclass |
| Tile positions | `TilePositionTable`, `GridSnap`; converters `positions_from_leica`, `positions_from_grid`, `positions_from_table` | frozen dataclasses, functions |
| Stage axes | `StageAxisMapping` | frozen dataclass |
| Neighbours | `neighbor_pairs`, `NeighborPair` | function, frozen dataclass |
| Registry | `STITCHING_METHODS: dict[type, StitchingSpec]`; a method returns `MethodPlacement` | registry, frozen dataclass |
| Supplied placement | `placement_from_table(table, grids, *, sample, origins=None)` ("Supplied placement") | function |
| Sample-frame metadata | `placed_metadata(result, fov_id)` | function |
| Method | `MultiviewStitcherConfig` (`multiview_stitcher`) | config |
| BigStitcher | `BigStitcherConfig`, `export_bigstitcher`, `import_bigstitcher` | config, functions |
| Review | `ReviewPolicy`, `review_placement`, `approve_placement` | config, functions |
| Persistence | `save_placement`, `load_placement` | functions |
| Errors | `StitchingBackendUnavailableError`, `PlacementReviewRequiredError` | exceptions |
| Coordination | `Dataset.stitch(sample, fov_ids, *, table, config, input, review, fallback, output_dir)` | method |

## The stitch entry

### Function

```python
def stitch(inputs: Sequence[StitchingInput], table: TilePositionTable, *, config,
           review: ReviewPolicy = ReviewPolicy(), fallback: str | None = None,
           sample: str = "sample") -> PlacementResult: ...
```

`config` is a frozen config registered in `STITCHING_METHODS`. `sample` names the sample of
the result and of its frame `sample:<sample>`; `Dataset.stitch` passes its own. `stitch` checks the inputs
against the table ("Input image"), derives the neighbour pairs, calls the method's `run`
once, converts its translations into origins, assigns one status per FOV, applies the
review policy and returns a `PlacementResult` whose record holds the uniform provenance
entry ([method-registry](../../docs/method-registry.md), "Provenance in run.json") with `stage` `stitching`. It never
reads or writes files and never runs Fiji. It raises for invalid input (unknown config
type, a missing optional dependency, inputs that do not match the table, grids that differ
in spacing, direction or unit); it does not raise for a placement that fails or needs
review: those are statuses and review flags in the result, which is saved like any other.

BigStitcher is not called through `stitch`: its placement enters the same
`PlacementResult` through `import_bigstitcher` ("BigStitcher").

### Coordination on the dataset

```python
Dataset.stitch(sample: str, fov_ids: Sequence[str], *, table: TilePositionTable | Mapping,
               config, input: str = "reference_merged", review=ReviewPolicy(), fallback=None,
               output_dir=None) -> PlacementResult
```

reads, for each FOV, the saved stitching input ("Input image") and its reference grid
with `reference_grid_from_file`, builds the `StitchingInput` list in table order, calls
`stitch`, and with `output_dir` saves the result ("Persistence").

`table` is a built `TilePositionTable`, or the `positions` mapping of the workflow
configuration ("Workflow configuration"), from which `Dataset.stitch` builds the table
with the named converter after reading the reference grids. A converter that needs the
tile size and spacing (`positions_from_grid`, or `positions_from_leica` with a
`GridSnap`) gets them from the mapping's `tile_shape_yx` and `spacing_yx` when it gives
them; otherwise from the FOVs' `ReferenceGrid` ([segmentation-contract](../../docs/segmentation-contract.md)):
`tile_shape_yx = grid.shape_zyx[1:]` and `spacing_yx = grid.metadata.spacing_zyx[1:]`
(the `ImageMetadata` spacing, µm). Taken from the grids, both must be the same for every
FOV of the sample; if the FOVs do not share one tile size or one spacing,
`Dataset.stitch` raises `IncompatibleGeometryError` naming the FOVs and their values, and
the caller must give both values explicitly (spacing must be shared in any case, "Input
image"). The values used are recorded in the table's record. It loads one FOV image at
a time into the list the method needs and holds nothing else: the method's memory is the
loaded tiles (the parity runs: LN 6.67 GiB of uint8 loaded, 7.0 GiB peak).

% W-324: LN 6.67 GiB

### Not the multi-FOV runner

The workflow integration's boundary excludes a multi-FOV runner: a package loop that runs `FOV.run`,
`FOV.segment` or `FOV.assign_molecules` over many FOVs, with scheduling, parallelism and retries.
Snakemake stays that runner. A sample-level computation is different in kind: its input
is inherently a set of FOVs (images that overlap and the positions that relate them), and
its output is one sample-level record that no single FOV owns. `stitch` and
`Dataset.stitch` call no per-FOV stage, schedule no per-FOV work and read only saved
per-FOV artifacts, as `FOV.assign_molecules` reads one FOV's saved artifacts. Snakemake runs one
stitch task per sample after the per-FOV tasks that write those artifacts.

% The §2.13 boundary excludes a multi-FOV runner

## Tile position table

```python
@dataclass(frozen=True)
class TilePositionTable:
    positions: pd.DataFrame      # one row per acquired FOV, in acquisition order
    unit: str = "µm"
    source: str                  # "leica:<path>", "grid", "table:<path>"
    record: dict                 # converter, its parameters, the stage-axis mapping, input SHA-256
```

| Column | Type | Meaning |
| --- | --- | --- |
| `fov_id` | str | The FOV identity, as the workflow's `fovs.csv` lists it or `Dataset.fov_ids` names it (`fov_pattern`, a `{i}` pattern); unique |
| `position_id` | int or missing | The acquisition's own identifier (Leica `PositionID`), when there is one |
| `nominal_z`, `nominal_y`, `nominal_x` | float64 | The nominal origin in the sample frame, µm |
| `source` | str | `stage`, `grid` or `supplied` |
| `grid_column`, `grid_row` | int or missing | The grid cell, when the converter knows one |

Rules: one row per acquired FOV, never a padding row; `fov_id` unique; every nominal value
finite. The rows' order is the acquisition order and is the order of every per-FOV table
of the placement record. Only the converters build tables; each is a function, not a
method, and none reads an image: the tile size and spacing a converter needs are
arguments, as values.

```python
@dataclass(frozen=True)
class GridSnap:
    overlap: float | tuple[float, float]   # fraction, 0 ≤ overlap < 1; one value or (y, x)
    tile_shape_yx: tuple[int, int]         # px; integers ≥ 1; the reference grids' Y and X sizes
    spacing_yx: tuple[float, float]        # µm per px; finite and > 0; the reference grids' Y and X spacing

def positions_from_leica(path, *, stage_axes: StageAxisMapping, fov_ids: Mapping[int, str],
                         snap_to_grid: GridSnap | None = None,
                         position_ids: Sequence[int] | None = None) -> TilePositionTable: ...
def positions_from_grid(*, columns: int, rows: int, order: str, tile_shape_yx: tuple[int, int],
                        spacing_yx: tuple[float, float], overlap: float | tuple[float, float],
                        fov_ids: Sequence[str]) -> TilePositionTable: ...
def positions_from_table(path_or_frame, *, unit: str = "µm") -> TilePositionTable: ...
```

* **Leica stage file** (`.maf`, LAS X): `StageXPos`, `StageYPos` in metres, mapped through
  `stage_axes` into image x and y, in µm, shifted so that the smallest y and x are 0; z is
  0 for every FOV, because the stage file gives no z offset between tiles (the input inventory's
  `positions.md`). `fov_ids` maps `PositionID` to the FOV id. `position_ids` selects the
  positions to read (each in the file once; None reads them all) before the axis mapping,
  the shift, the snap step, the cells and the clash check, so that a file that holds every
  sample's positions gives the table of a file that holds only the selected ones
  (`Dataset.stitch` selects the FOVs of `fov_ids`, since 2026-10-24). `snap_to_grid` optionally
  replaces the positions by a regular grid derived from them, as today's grid derivation
  does ([stitching-algorithms](stitching-algorithms.md), "Leica stage file"): the stage positions decide each
  FOV's cell, and `GridSnap`'s `overlap`, `tile_shape_yx` and `spacing_yx` give the step,
  with the same names, types, units and rules as the arguments of `positions_from_grid`.
  None of the three fields has a default.
* **Regular grid**: FOV `k` in `fov_ids` order goes to a cell by `order`;
  `column_major_down_right`, the scan of the two sample datasets (the input inventory's `positions.md`),
  puts it at column `k // rows`, row `k % rows`. The step is `(1 − overlap) ×
  tile_shape_yx × spacing_yx` per axis. `order`, `overlap`, `tile_shape_yx` and
  `spacing_yx` have no defaults.
* **Supplied table**: a CSV or frame with `fov_id`, `nominal_z`, `nominal_y`, `nominal_x`
  in `unit` (converted to µm), for example from another microscope's metadata.

% W-323 `positions.md` (twice); `Dataset.stitch` selects the FOVs of `fov_ids`, W-566

### Stage-axis mapping

```python
@dataclass(frozen=True)
class StageAxisMapping:
    x: str     # where stage +X goes in the image: "+x", "-x", "+y" or "-y"
    y: str     # where stage +Y goes; the two must name different image axes
    source: str = "declared"
```

The mapping is an explicit input of `positions_from_leica` with no default, and it is
recorded in the table's record. For LN sample1 it is `StageAxisMapping(x="+x", y="+y")`,
for the images as they are stored in the 2024 stitching input (the input inventory's `positions.md`).

% W-323 `positions.md`

**`rotate_angle`.** The legacy key stays what it is: an image rotation applied before
stitching (`rotate_nuclei` and the Python pipeline's `rotation_degrees`), so the stitching
input is already rotated. It says nothing by itself about the stage axes. Today's grid
derivation hard-codes stage X → column (+x) and stage Y → row (+y) for the rotated images
([stitching-baseline](../baselines/stitching.md), G3). The workflow adapter therefore translates a legacy
configuration (one without a `stitching` block) into `StageAxisMapping(x="+x", y="+y",
source="legacy")`, which reproduces the legacy grid, and records the `rotate_angle` it ran
with. A configuration with a `stitching` block and a stage file must name `stage_axes`;
the adapter never infers it from the angle. An angle that is not a multiple of 90° changes
the tile shape (`reshape=True`) and cannot be represented by translation-only placement;
`Dataset.stitch` refuses such inputs when their grids differ in shape.

## Neighbour pairs

Two FOVs are neighbours when their nominal boxes overlap: the box of FOV `i` spans
`nominal_i` to `nominal_i + shape_i × spacing` per axis, from its reference grid. A pair
`(a, b)`, `a` before `b` in table order, is a neighbour pair when the boxes, each grown by
`tolerance_um` per side, intersect in Y and X with an area above zero; for Z>1 grids the z
ranges must intersect as well. Diagonal neighbours are pairs too. Row and column indices
are never used, so a stage file, a regular grid and a supplied table give pairs by the
same rule. `neighbor_pairs(table, grids, *, tolerance_um=0.0)` returns them with the
nominal overlap box of each; the method may register a subset (multiview-stitcher's
`alternating_pattern` registered 111 of LN's 213 pairs, the parity runs' notes, section 2.1) and
reports which.

% W-324 notes section 2.1

## Input image

Each FOV contributes one specified, saved stain image on that FOV's reference grid, the
grid its molecules and masks use:

```python
@dataclass(frozen=True)
class StitchingInput:
    fov_id: str
    image: np.ndarray | None          # ZYX on grid; None for a FOV whose image is absent
    grid: ReferenceGrid               # image.shape == grid.shape_zyx
    source: str                       # "reference_merged", "reference_stain" or "file:<path>"
    file_sha256: str | None = None
```

* `Dataset.stitch(input="reference_merged")` reads the FOV's reference image, the file the
  dataset's `reference_image` names (a file pattern with `{fov_id}`, absolute or relative
  to the output root; default `fovs/{fov_id}/reference.tif`, which
  `FOV.save_reference_image(reference_image="merged")` writes with the reference metadata,
  the image the input inventory names for the two sample datasets); `input="reference_stain"` reads the
  saved `reference_stain` of `FOV.prepare_morphology`
  (`fovs/<fov_id>/other_rounds/reference_stain/image.ome.tif`), the nuclear
  stain LN is stitched on. Its grid is the reference grid by the grid rule of
  the [assignment specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/assignment.md#the-grid-rule): the reference image's stored metadata, or, for a reference
  image that stores none, the grid the dataset's `voxel_size_xy` and `voxel_size_z`
  declare (without them `ValueError` naming them), as `Dataset.export` reads it; the
  workflow's stitching rules read it the same way. A missing reference image raises
  `FileNotFoundError` naming `reference_image` and the path.
* A YX file gives a Z=1 grid; every FOV of a sample must have the same Z size and
  spacing, the unit `µm` and the identity direction, else `IncompatibleGeometryError`;
  `µm`, `um` and `micrometer` are one unit, and a grid without a direction has the
  identity direction. Tile shapes may differ in Y and X
  only if the method's spec allows it (`multiview_stitcher` does).
* The legacy `images/DAPI/{fovID}.tif` (rotated, optionally projected DAPI) is accepted as
  `file:<path>` only when its shape equals the reference grid's; it carries no metadata, so
  its grid is the reference grid read from the FOV's reference image.
* The spacing is the reference grid's: the YAML `voxel_size_xy` and `voxel_size_z` give it
  only for a reference image that stores no metadata, since the stitching input files
  carry no calibration (the input inventory's `inputs.md`).

% the image W-323 names; W-323 `inputs.md`

## Placement record

```python
@dataclass(frozen=True)
class PlacementResult:
    sample: str
    fovs: pd.DataFrame        # one row per row of the tile position table, same order
    links: pd.DataFrame       # one row per registered neighbour pair
    record: dict              # frame, gauge, method provenance, inputs, table, review
```

### Per-FOV table

| Column | Type | Rule |
| --- | --- | --- |
| `fov_id` | str | Equals the table's `fov_id`, same order; unique |
| `status` | str | Exactly one of `image_supported`, `metadata_only`, `placeholder`, `missing`, `failed` |
| `origin_z`, `origin_y`, `origin_x` | float64, µm | The sample-frame position of index `(0, 0, 0)` of the FOV's reference grid. Fractional values are preserved: never rounded or cast. Finite for `image_supported`, `metadata_only` and `placeholder`; NaN for `missing` and `failed` |
| `nominal_z`, `nominal_y`, `nominal_x` | float64, µm | Copied from the tile position table |
| `correction_z`, `correction_y`, `correction_x` | float64, µm | `origin − nominal`; NaN where the origin is NaN |
| `method_translation_z/y/x` | float64 | The method's own result for this FOV before the gauge, in the method's units (µm for `multiview_stitcher`; BigStitcher world units, x and y in pixels and z in xy-pixel units, for the import) |
| `component` | int | Index of the FOV's component of kept links; −1 for `missing`. A kept link to a FOV whose method translation is not finite joins nothing, so such a FOV is its own component |
| `n_links_kept` | int | Kept links of this FOV |
| `flags` | str | `;`-separated: `fallback`, `supplied`, `correction_above_bound`, `residual_above_tolerance`, `nan_quality_link`, `optimizer_removed_link` |

The **statuses**:

* `image_supported`: placed from its image; at least one kept link joins it to the
  reference component (the component with the most FOVs; ties go to the one holding the
  first FOV in table order).
* `metadata_only`: in a measured placement only with `fallback="nominal"` (opt-in): its
  image is present but no kept link joins it to the reference component (no overlap,
  every link dropped, or another component). Its origin is its nominal position, `origin = nominal` (correction
  0), which under gauge G1 is its nominal position in the gauge of the reference
  component; it is flagged `fallback`. The BigStitcher import applies the same rule to a
  view without a link into the reference component (one that BigStitcher's second round
  placed from metadata); BigStitcher's own position for it stays in
  `method_translation_*` ("BigStitcher"). Every FOV of a supplied placement is
  `metadata_only` too, flagged `supplied` instead of `fallback`, at its supplied origin
  when origins are given and at `origin = nominal` otherwise ("Supplied placement").
* `placeholder`: an acquired FOV whose image is absent and was replaced by a blank of the
  tile's shape so that the BigStitcher grid stays complete (a settled decision; only in the BigStitcher
  path, with `BigStitcherConfig.pad_missing=True`); its origin is the one BigStitcher gives
  the blank view, which is its regular-grid position. `pad_missing=True`, off by default, is
  the caller's explicit opt-in to this metadata position; review is `required`.
* `missing`: an acquired FOV whose image is absent and not replaced; no origin.
* `failed`: the method or a check rejected the FOV: disconnected from the reference
  component without the fallback (with `fallback=None`, the default, in `stitch` and in
  the BigStitcher import alike), a non-finite translation, or a method error that
  affects every FOV (multiview-stitcher's `NotEnoughOverlapError` makes every FOV
  `failed`). Its method translation, if any, stays in `method_translation_*`.

% `placeholder`: so that the BigStitcher grid stays complete (D4)

**Index to sample coordinates.** For a FOV with a finite origin, the zero-based index
position `p = (z, y, x)` of its reference grid, the convention of molecules and labels
(the centre of voxel `(z, y, x)` is at integer indices), maps to sample coordinates by

```
sample_zyx = origin_zyx + D · (p ⊙ spacing_zyx)
```

with `spacing_zyx` and `D` (`direction_zyx`) of the FOV's `ReferenceGrid.metadata`.
Equivalently, `placed_metadata(result, fov_id).index_to_world(p)`, where `placed_metadata`
returns `dataclasses.replace(grid.metadata, frame_id="sample:<sample>",
origin_zyx=origin)` read as the grid checks read it (`um` and `micrometer` as `µm`, no
direction as the identity); the recorded metadata is not changed. Example: origin `(0.0, 250.5, 100.25)` µm, spacing `(0.346255,
0.194398, 0.194398)` µm and identity direction map `p = (2, 10, 0)` to `(0.69251,
252.44398, 100.25)` µm. Fractional indices (sub-voxel molecule positions) map by the same
formula. The inverse, for resampling a FOV into a fused image, is
`p = Dᵀ · (sample_zyx − origin_zyx) ⊘ spacing_zyx`.

**Relation to `ImageMetadata.origin`.** A FOV's own `ImageMetadata` stays in the FOV frame
and is not changed by stitching; its `origin_zyx` (usually `(0, 0, 0)` or unknown) is not
read. The placement origin replaces it only in the derived sample-frame metadata above.
Every FOV of a sample must have identity `direction_zyx` (the sample frame's; a grid
without a direction has the identity direction), known spacing and the unit `µm` (spelled
`µm`, `um` or `micrometer`), else `stitch` raises `IncompatibleGeometryError`. A cropped grid
(`ImageMetadata.cropped(start, …)`) maps its own index `q` to the FOV index `q + start`
first.

**Relation to the registration transform direction.** Registration transforms are pull maps from
reference to moving index coordinates within one FOV ([registration-contract](../../docs/registration-contract.md)). A
placement is the opposite sense: a push map from FOV coordinates into the sample frame,
the direction in which both backends store their result ("view to common frame", the parity
runs' notes, section 3). Molecules, already in reference-grid indices after registration, are
pushed into the sample frame with the rule above; a fused image pulls each output voxel
from the FOV through the inverse.

% Relation to the §2.6 transform direction. §2.6 transforms are pull maps; "view to common frame", W-324 notes section 3

**Gauge.** A placement fixes relative positions only; both backends anchor an arbitrary
view. The record removes that freedom with a stated gauge:

| Option | Rule | Effect on the golden values | Effect on existing outputs |
| --- | --- | --- | --- |
| **G1. Zero mean correction (recommended)** | `origin_i = nominal_i + c_i − mean(c_j)` over `image_supported` FOVs, where `c_i` is the method's correction in µm. The sample frame is then the nominal frame, and two placements of the same table are comparable without removing a mean. | None. | `tile_config` is not in this gauge; its legacy view uses the raw method translations ("Where the record differs from `tile_config`"). |
| G2. The method's reference view keeps its nominal position | `origin_i = nominal_i + c_i − c_ref` | None. | The frame then depends on which view the backend fixed (BigStitcher: tile 0, which is a blank for LN). |
| G3. Smallest origin at 0 | as `tile_config`'s T4, without the cast | None. | Close to `tile_config`, but depends on the extreme FOV. |

### Links

| Column | Type | Rule |
| --- | --- | --- |
| `fov_a`, `fov_b` | str | The pair, `fov_a` before `fov_b` in table order |
| `overlap_z0 … overlap_x1` | float64, µm | The nominal overlap box |
| `shift_z`, `shift_y`, `shift_x` | float64, µm | The measured relative shift of `fov_b` against `fov_a` |
| `quality` | float64 | The method's link quality: Spearman correlation for `multiview_stitcher`, BigStitcher's correlation `r` for the import; NaN when undefined |
| `kept` | bool | Whether the link entered the global optimization |
| `reason` | str | Why a link was not kept: `quality_below_threshold`, `nan_quality`, `optimizer_removed`, `pruned` (pair not registered) |
| `residual` | float64, µm | `‖(origin_b − origin_a) − (nominal_b − nominal_a) − shift‖` after placement; NaN for an unkept link |

### Method record

`record["method"]` is the uniform provenance entry of [method-registry](../../docs/method-registry.md) (`stage`
`stitching`, `method`, `config_type`, `implementation`, `config`, `requires` with the
installed versions, `artifacts` empty). For the import it names `bigstitcher` with the
`BigStitcherConfig` that rendered the macro, the macro text's SHA-256, the `dataset.xml`
SHA-256, the Fiji and BigStitcher versions when the engine task recorded them, and the
padding views with their grid cells and raw translations. `record` also holds `frame`
(`frame_id`, unit, direction), `gauge`, `table` (its record), `inputs` (per FOV: source,
file SHA-256, grid shape, metadata, grid SHA-256), `review` and `format_version`.

A test can check the record with these assertions: the `fov_id` column equals the
table's, row for row; `status` takes one of the five values in every row; origins are
float64 and equal the unrounded arithmetic of [stitching-algorithms](stitching-algorithms.md) (a fixture with
fractional translations keeps its fractions); origins are finite exactly for the three
placed statuses; under G1 the mean correction over `image_supported` FOVs is 0 within
1e-9 µm; and `placed_metadata(…).index_to_world(p)` equals the rule above.

## Method registry

`STITCHING_METHODS` is the stage's registry on the shared registry mechanism
([method-registry](../../docs/method-registry.md), "The shared mechanism"): a plain module-level `dict` from an exact
frozen config type to its spec, with lookup, names, lazy dependencies and provenance
through `starfinder._registry`.

% the stage's registry on the W-240 mechanism

```python
@dataclass(frozen=True)
class StitchingSpec:
    name: str                                    # shared
    run: Callable                                # shared; signature below
    requires: tuple[Dependency, ...] = ()        # shared
    min_shape_zyx: tuple[int, int, int] = (1, 1, 1)   # shared
    dimensions: frozenset[int] = frozenset({2, 3})    # Z=1 placed as a plane, Z>1 as a volume
    devices: tuple[str, ...] = ("cpu",)          # placement is CPU only
    transform_kind: str = "translation"          # the only kind the accepted design admits

def run(tiles: Sequence[np.ndarray], grids: Sequence[ReferenceGrid], nominal_zyx: np.ndarray,
        pairs: Sequence[NeighborPair], *, config) -> MethodPlacement: ...

@dataclass(frozen=True)
class MethodPlacement:
    translations_zyx: np.ndarray   # (n, 3) float64, µm, view to common frame; NaN where not placed
    components: np.ndarray         # (n,) int
    links: pd.DataFrame            # fov_a, fov_b, shift_*, quality, kept, reason
    diagnostics: dict              # library-specific: removed links, iterations, reference view
```

The method takes the tiles and nominal positions and returns per-FOV translations and
link diagnostics in one call (a settled decision); `stitch` applies everything else (statuses, gauge,
review). `tiles[i]` is `None` for an absent image, which the method must leave unplaced.

% the code block above: `transform_kind`, the only kind §2.10 admits; one call (D3)

**The first method, `multiview_stitcher`.** `MultiviewStitcherConfig` (discriminator
`method="multiview_stitcher"`), with `requires=(Dependency("multiview_stitcher",
"multiview-stitcher", extra="multiview-stitcher"),)`, `dimensions={2, 3}`,
`devices=("cpu",)` and `min_shape_zyx=(1, 16, 16)`. Its fields, units and defaults are in
[stitching-algorithms](stitching-algorithms.md) ("multiview_stitcher"); the starting configuration is phase
correlation, translation, binning 4 in every axis, links below 0.7 or with NaN quality
dropped, and global optimization. A missing package raises
`StitchingBackendUnavailableError` naming the module and the extra; importing
`starfinder.stitching` never imports it. Its extra and Python bound are in "Dependency
plan".

## BigStitcher

BigStitcher stays the initial default and runs as an engine task through Fiji, as settled.
The package owns the two ends:

% the initial default (D1); an engine task through Fiji (D2)

```python
def export_bigstitcher(table: TilePositionTable, inputs: Sequence[StitchingInput], *,
                       config: BigStitcherConfig, tile_dir: str, save_dir: str
                       ) -> tuple[str, pd.DataFrame]: ...     # macro text, view map
def import_bigstitcher(dataset_xml, *, table: TilePositionTable, view_map: pd.DataFrame,
                       grids: Mapping[str, ReferenceGrid], config: BigStitcherConfig,
                       review: ReviewPolicy = ReviewPolicy(), fallback=None,
                       sample: str = "sample", macro: str | None = None) -> PlacementResult: ...
```

The import's `sample` is that of `stitch`; `macro` is the text of the macro that ran, whose
SHA-256 the record keeps (null when it is not given).

**`BigStitcherConfig`** records today's parameters as defaults ([stitching-baseline](../baselines/stitching.md),
"The macro"): Bioformats loader; `Move Tile to Grid`, `Down & Right`, overlap 10 % in x,
y and z; re-save as multiresolution HDF5 with `subsampling_factors=[{2,2,2}, {4,4,4}]`,
chunks 16³ and deflate compression; phase correlation at 4× downsampling in x, y and z;
link filter `min_r=0.7`, `max_r=1`, no shift or displacement limits; global optimization
`relative=2.5`, `absolute=3.5` px, the two-round strategy with metadata for unconnected
tiles and iterative dropping of bad links, the first tile fixed; and `pad_missing=False`.
The voxel size written into the macro is the reference grids' spacing; the legacy rule
keeps writing `voxel_size_xy` and `voxel_size_z`. The record keeps the config and the
macro's SHA-256.

**The macro function** places the tile files in the grid order BigStitcher expects
(`tile_{j}.tif`, column-major), writes a blank of the tile shape for each empty cell
(padding views) and, with `pad_missing=True`, for each acquired FOV whose image is absent
(status `placeholder`), and returns the view map (`view`, `grid_column`, `grid_row`,
`fov_id` or empty for padding, `kind` `fov`/`padding`/`placeholder`). Placeholder and
padding handling stays inside this path (a settled decision).

% padding handling inside this path (D4)

**The Fiji run stays an engine task**: the rule runs Fiji on the macro with the save folder
on local disk (the input inventory: BigStitcher cannot write its HDF5 on the CIFS mount), and it fails
unless `dataset.xml` exists and holds a `Stitching Transform` for every view, because Fiji
exits with status 0 when the macro fails.

% W-323: BigStitcher cannot write its HDF5 on the CIFS mount

**The import** reads `dataset.xml`, maps each view to a FOV through the view map and the
`ViewSetup` tile attribute (not through the order of the transforms), composes the view's
transforms except `calibration`, refuses a `dataset.xml` with a non-finite number in a
view's affine (`ValueError` naming the setup, the view and the transform), refuses a
transform with a non-identity linear part,
converts world units to µm through the FOV's reference grid (no integer cast), reads the
`PairwiseResult` links with their correlation, and builds the `PlacementResult` with the
gauge, statuses and review of `stitch`. Padding views have no row; their raw translations
stay in the method record. The arithmetic is in [stitching-algorithms](stitching-algorithms.md) ("Import
arithmetic").

The import follows the same fallback rule as `stitch`. An acquired view without a link
into the reference component is a view BigStitcher's second round placed from metadata:

* with `fallback=None`, the default, its FOV is `failed`: the origin is NaN, the
  translation BigStitcher gave it stays as a diagnostic in `method_translation_*` (world
  units, unrounded), and review is `required`;
* with `fallback="nominal"`, its FOV is `metadata_only` at `origin = nominal` (correction
  0) and flagged `fallback`, and review stays `required`. This is not BigStitcher's
  second-round position: the second round moves the view by its metadata relative to tiles
  already placed, so the two differ by the corrections of the tiles it was aligned to and,
  for stage positions, by the difference between the stage positions and BigStitcher's
  regular grid. BigStitcher's position stays in `method_translation_*`, from which it can
  be converted and gauged as for an `image_supported` FOV.

Either way the raw `dataset.xml` content of every view stays available, and the legacy
view does not read statuses: the cast, shift and rectangles (T3 to T7) run on the raw
translations of every view, so `tile_config_{sample}.csv` is the same for any `fallback`
value. No `fallback` value is needed to reproduce today's output: the workflow's default
configuration (no `stitching` block) runs the four legacy rules unchanged and does not
call the import, and with a `stitching` block the legacy writer reads the raw
translations, not the statuses.

**A macro variant from the tile position table (option).** Instead of moving tiles to a
padded regular grid, the macro can define the dataset from the table's nominal positions
(BigStitcher's tile configuration import), with acquired FOVs only and no padding views.
Effect on existing results: the start positions change from the 10 % grid to the stage
positions (LN: about 4 px per step, 1342.6 against 1346.4 px, the parity runs' notes, section 2.4),
which changes the overlap boxes and so the link set; the 2024 `dataset.xml` and
`tile_config_sample1.csv` would not be reproduced, and the parity runs saw multiview-stitcher's
placement from the stage start diverge (19 FOVs about 147 px off). It is therefore an
option for later work, not the default; the golden values are unaffected.

% W-324 notes section 2.4; W-324 saw multiview-stitcher's placement diverge

### What the import of the LN `dataset.xml` must reproduce

From the parity outputs (`parity/` in the parity runs' directory) and the 2024 file
(the input inventory's `20261005T1930Z-fiji-baseline/inputs/dataset_2024.xml`):

% From the W-324 parity outputs (`parity/` in the W-324 run directory) and the 2024 file
% (W-323 `20261005T1930Z-fiji-baseline/inputs/dataset_2024.xml`)

1. **Raw translations.** For each of the 72 setups, the raw translation the import keeps
   (`method_translation_*` of the 64 FOVs and the padding views' entries) equals the sums
   of `prototype/tile_config_ln/T2_sum.csv` exactly: the float64 sum of the two parsed
   translations, before any cast.
2. **View identity.** Setup `i` is `tile_{i}.tif` and row `i` of `script_grid_ln.csv`; the
   64 acquired setups map to the `id` column there and the 8 blank setups are padding
   views.
3. **The legacy view.** Applying the legacy cast, shift and rectangles (T3 to T7) to the
   import's raw translations, with `img_col = img_row = 1496`, reproduces
   `script_tile_config_ln.csv`, which equals the 2024 `tile_config_sample1.csv`, in all 72
   rows and 16 columns, with the minima −5460, −6135 and −69 before the shift.
4. **Links and statuses.** The 145 `PairwiseResult` entries (correlation 0.7001 to 0.9636)
   become 145 links, all kept under the recommended link rule; every acquired view has at
   least one, none involves a blank view, and the 64 acquired views form one component, so
   all 64 FOVs are `image_supported` (counted from the 2024 file for this page).
5. **Units.** z in slices is the world z divided by the calibration's z scale
   1.7894736842105263, as the parity runs' comparison (`compare_xml.py`) converts it; origins in µm
   use the reference grid's spacing. Relative origins agree with the input inventory's Fiji rerun within
   1.3e-06 px, the run-to-run variation of BigStitcher's optimization.

% W-324's comparison (`compare_xml.py`); W-323's rerun

The golden test holds item 3 on its own fixture.
`test_workflow_stitching_golden.py` passes unchanged while the legacy scripts exist; when
the implementation replaces a script, `legacy_tile_config` stays as the frozen reference,
and the import's legacy view must reproduce its pinned `TILE_CONFIG` from the fixture's
`dataset.xml` (item 3 above on the fixture).

### Where the record differs from `tile_config` on purpose

| `tile_config_{sample}.csv` | Placement record | Why |
| --- | --- | --- |
| Integer offsets (truncated) | float64 origins | The fraction is part of the result of both backends |
| Pixels, XYZ, z in xy-pixel units | µm, ZYX, z through the slice spacing | One physical sample frame; the z unit of `dataset.xml` is not a slice |
| Shifted so each axis' minimum is 0 (a positive minimum moves further) | Gauge G1 | A stated frame that does not depend on the extreme FOV |
| One row per grid cell, blanks included with offsets | One row per acquired FOV; padding views only in the method record | Placeholders stay inside the BigStitcher path |
| FOV identity = the grid's `id` (Leica `PositionID`) | `fov_id` from the tile position table | `fovs.csv` names FOVs (`fov_id_pattern` for the MATLAB backend) |
| Non-overlap rectangles from `img_col` and `img_row` | none | Ownership is assemble's ([assembly-contract](assembly.md)), from each grid's shape |
| Calibration from `voxel_size_xy`, `voxel_size_z` | From each FOV's reference grid | The files carry no calibration; the YAML values are rounded for LN |
| No status, no links, no method record | Statuses, links with quality, provenance, review | The failure policy needs them |

% Ownership is assemble's (W-326)

The legacy writer of `tile_config_{sample}.csv` stays for the rule chain that reads it
(`reads_assignment` through the workflow adapter) until assemble replaces it; it takes
its values from the record's raw translations and the view map, so its output is
unchanged. (Since 2026-10-24, with an `assembly` block, where assemble has replaced that chain, the
BigStitcher `stitch_sample` neither writes nor declares the file.)
`test_assignment_workflow.py` and the other readers of `tile_config_{sample}.csv` pass
unchanged, because the legacy writer's output is unchanged.

% W-568: with an `assembly` block

## Supplied placement

A placement that is known rather than measured, such as the placement truth of a
synthetic scene ([synthetic-sample-scene](../../docs/synthetic-sample-scene.md), "Placement truth") or another program's
registration, enters through one public function. It defines no new placement type: the
result is a `PlacementResult` like those of `stitch` and `import_bigstitcher`, and assemble,
`placed_metadata`, the review functions and persistence treat it as they treat any
placement.

```python
def placement_from_table(table: TilePositionTable, grids: Mapping[str, ReferenceGrid], *,
                         sample: str,
                         origins: Mapping[str, tuple[float, float, float]] | None = None,
                         ) -> PlacementResult: ...
```

* **Arguments.** `table` is a tile position table from any converter; its nominal
  positions are the FOVs' nominal positions, in µm, usually read by `positions_from_table`
  (`source` `supplied`). `grids` maps each `fov_id` of the table to that FOV's
  `ReferenceGrid`, the grid its molecules and masks use. `sample` names the sample, whose
  frame is `sample:<sample>`. `origins`, optional, maps each `fov_id` of the table to that
  FOV's supplied origin: three numbers in the order z, y, x, in µm in the sample frame,
  the position of index `(0, 0, 0)` of the FOV's reference grid ("Terms"). The origins are
  an argument of their own, beside the table, so that a supplied placement keeps the
  acquisition's nominal positions (for example one common `nominal_z`) and carries the
  known origins separately. They are read as float64 and used as given: never rounded,
  cast or shifted. µm is the only unit; a caller with origins in another unit converts
  them first. The function reads no image or file, calls no method, derives no neighbour
  pair and needs no extra.
* **Origin and correction of each FOV.** Without `origins` (`None`, the default), each
  FOV's origin is its nominal position (`origin = nominal`, correction 0). With `origins`,
  each FOV's `origin_*` is its supplied value and its `correction_*` is `origin − nominal`,
  the float64 difference, which may be any finite value. In both cases `nominal_*` is
  copied from the table. Origins equal to the nominal positions give the same per-FOV
  table as `origins=None`.
* **Per-FOV table.** In both cases one row per table row, in table order. Every FOV has
  the status `metadata_only` and the flag `supplied`, and its `method_translation_*` is
  NaN, because no method ran. `n_links_kept` is 0, and `component` is the FOV's row index:
  without links each FOV is its own component.
* **Links.** In both cases none: `links` has the columns of "Links" and no row.
* **Method record.** `record["method"]` is the uniform provenance entry with `stage`
  `stitching` and `method` `supplied`, and with `implementation`
  `starfinder.stitching.placement_from_table`. `config_type`, `config`, `requires` and
  `artifacts` are empty. `supplied` is not a key of `STITCHING_METHODS`, so `stitch` cannot
  select it and the workflow's `stitching.method` does not name it. `record["origins"]` is
  `nominal` without the argument and `supplied` with it; the supplied values are the
  `origin_*` columns of `fovs`. `record["table"]` holds the table's record, with the
  SHA-256 of a supplied file. `record["gauge"]` is `none` in both cases: the origins are
  the nominal or the supplied positions as given, and no gauge shift is applied (G1
  averages over `image_supported` FOVs, and there are none). `record["inputs"]` holds, per
  FOV, the source `supplied`, a null file SHA-256 and the grid shape, metadata and SHA-256
  that assemble's grid check reads.
* **Review.** In both cases `record["review"]` is `required` with the reason
  `supplied_placement`, whatever the positions, origins and corrections, so a supplied
  placement reaches assembly only after the caller approves it explicitly with
  `approve_placement(result, *, reviewer, note)` ("Failure policy"). Under V1 the review
  would be required anyway, because no FOV is `image_supported`; the reason names the
  cause. The approval changes no origin and no correction.
* **Persistence.** In both cases the same saved form as any placement: `save_placement`
  writes `stitching/<sample>/placement.json` (with `origins`), `fovs.csv` (the nominal,
  origin and correction columns) and `links.csv` (header only), and `load_placement`
  checks both tables' SHA-256 as for any placement. No view map, macro or `dataset.xml`
  copy exists for it.

**What the review functions report.** `review_placement(result, policy)` keeps, on a
placement whose method is `supplied`, the review `required` with the reasons exactly
`[supplied_placement]`; it adds no flag and changes no status, whatever the corrections.
A correction that is not zero, of any size, including one above `max_residual_um`
(default 1.0 µm, "Review criterion") or above `max_correction_fraction` of a nominal
overlap, sets neither `residual_above_tolerance` nor `correction_above_bound`: the
residual check reads kept links, and the correction bound reads the overlap widths of the
FOV's registered pairs, and a supplied placement has neither. Nor does the components
check make a FOV `failed`, although each FOV is its own component. The corrections are
part of what the caller approves: they are columns of `fovs`, whose SHA-256 the approval
records. `approve_placement` is required before assemble accepts the placement, with or
without `origins` and whatever the corrections.

**In assembly.** Assemble reads a supplied placement as any other ([assembly-contract](assembly.md)):
sample coordinates and ownership come from `origin_*`, and check 8 ("One layer") reads
`nominal_z`. A table with one common `nominal_z` therefore passes check 8 whatever the z
of the supplied origins, and a table whose `nominal_z` differ is refused with or without
`origins`.

**Failure.** `ValueError` for a table with no row, for a `fov_id` of the table without a
grid in `grids` (naming those FOVs), and for a key of `grids` that is not a `fov_id` of the
table. With `origins`, `ValueError` naming the FOVs for a `fov_id` of the table without an
origin in `origins`, for a key of `origins` that is not a `fov_id` of the table, and for an
origin that is not three finite numbers (NaN or ±inf on any axis, or another number of
values); an empty mapping is therefore refused for every FOV, and only `None` omits the
origins. `IncompatibleGeometryError`, as in `stitch`, for grids with unknown spacing, a
unit other than µm (`µm`, `um` and `micrometer` are one unit) or a non-identity direction
(a grid without a direction has the identity direction), and for grids that differ in Z
size or spacing. Tile shapes may differ in Y and X, because no method registers
the tiles. The table's own rules (unique `fov_id`, finite nominal values) are checked when
the table is built.

A test can check it with these assertions: the result is a `PlacementResult`; its `fovs`
rows equal the table's `fov_id`s in order, each `metadata_only` with the flag `supplied`;
without `origins`, origins equal to the nominal values exactly and corrections 0; with
`origins`, origins equal to the supplied values exactly and corrections equal to
`origin − nominal` exactly, also above 1.0 µm; `links` has no row;
`record["method"]["method"]` is `supplied`; `record["origins"]` is `nominal` or
`supplied`; review is `required` with the reason `supplied_placement`, and
`review_placement` adds no flag; assemble refuses the result with
`PlacementReviewRequiredError` until `approve_placement` has run; a save and load round
trip returns equal tables and record; a table FOV without a grid raises `ValueError` naming
it; and a table FOV without an origin, an origin for a FOV the table lacks and an origin
with a NaN or infinite value each raise `ValueError` naming the FOV. The arithmetic is in
[stitching-algorithms](stitching-algorithms.md) ("Supplied placement").

## Link-quality rule

| Option | Rule | Effect on the golden values | Effect on existing outputs |
| --- | --- | --- | --- |
| **L1. Threshold 0.7, NaN dropped (recommended)** | A link is kept when its quality is finite and ≥ 0.7 (the macro's `min_r`). For `multiview_stitcher` the wrapper turns the library's filter on at 0.7 and also drops NaN-quality links, which the library keeps (the parity runs' notes, section 3); for the import, the links in `dataset.xml` are those that passed BigStitcher's filter. | None. | BigStitcher: none (same threshold). multiview-stitcher on LN: 14 of 111 links dropped, placement moved by at most 0.622 px against no filter, still one component (the parity runs' notes, section 2.3). |
| L2. The library default | No quality filter; the global optimization removes links by residual; NaN links kept. | None. | multiview-stitcher's main configuration in the parity runs; a constant tile's NaN links pull it to its nominal position, and a noise tile was placed 22 px off (the parity case `no_signal_noise`). |
| L3. A per-sample threshold | Keep links above a quantile of the sample's qualities | None. | Not measured; the threshold would vary between samples. |

% L1: W-324 notes section 3; W-324 notes section 2.3. L2: multiview-stitcher's main W-324 configuration; W-324 `no_signal_noise`.

**Recommendation: L1.** It applies one rule to both backends, matches the macro's current
behavior, and treats a constant overlap as carrying no information about the offset.

## Failure policy

As agreed on 2026-09-27:

1. **Successful transforms and diagnostics are saved.** Every placement is saved with all
   its FOVs, links and flags, also when some FOVs failed or review is needed. `stitch`
   never discards a result.
2. **Unreliable or disconnected placement needs review before sample assembly.**
   `review_placement(result, policy)` sets `record["review"]` to `not_needed` or
   `required` with the reasons ("Review criterion"). Assemble ([assembly-contract](assembly.md)) refuses a placement
   whose review is `required` and that has no approval, raising
   `PlacementReviewRequiredError` with the reasons. `approve_placement(result, *, reviewer,
   note)` records an approval (reviewer, UTC time, note, the SHA-256 of the per-FOV and
   link tables it approves) and returns a new result; it changes no origin.
3. **Stage-position fallback is opt-in and flagged.** With `fallback=None`, the default, a
   FOV that is not joined to the reference component is `failed`. With
   `fallback="nominal"` it is `metadata_only` at its nominal position in the gauge of the
   reference component (`origin = nominal`) and flagged `fallback`; review stays
   `required`. The BigStitcher import follows the same rule: a view BigStitcher's second
   round placed from metadata is `failed` with `fallback=None`, with BigStitcher's
   translation kept as a diagnostic, and `metadata_only` only with `fallback="nominal"`
   ("BigStitcher"). BigStitcher's two-round strategy is today's built-in fallback inside
   Fiji; the import does not adopt its result as a placement unless the caller opts in.
   The macro keeps the two-round strategy as its default (decided at the review gate,
   2026-10-06); `BigStitcherConfig` records it ("BigStitcher").

% Assemble (W-326) refuses a placement; decided at the W-329 review, 2026-10-06

### Review criterion

| Option | Review is required when | Effect on the golden values | Effect on existing outputs |
| --- | --- | --- | --- |
| **V1. Disconnection, unplaced FOVs, corrections, residuals (recommended)** | any FOV is not `image_supported`; or a correction exceeds `max_correction_fraction` (default 0.5) of the nominal overlap width in Y or X; or a kept link's residual exceeds `max_residual_um` (default 1.0 µm, about 5 px at 0.1944 µm); or the optimizer removed a link (where the backend reports it) | None. | LN BigStitcher import: all 64 FOVs `image_supported`; the residual check decides on the two links the 2024 optimization removed (the input inventory's log), which `dataset.xml` does not mark. multiview-stitcher on LN from the grid start: corrections up to 33.9 px in x and 44.5 px in y, under half the 149.6 px overlap; from the stage start up to 149.9 px, flagged (the parity runs' notes, section 5). |
| V2. Disconnection and failed FOVs only | more than one component, or any `failed` or `missing` FOV | None. | The LN stage-start placement, with 19 FOVs about 147 px off in one component, passes unflagged. |
| V3. Always | every placement | None. | Every sample needs a sign-off before assembly, also the 2024 LN placement. |

% V1: W-323 log; W-324 notes section 5

**Recommendation: V1.** The parity runs' stage-start case shows that a wrong placement can stay
connected, so connectivity alone is not enough; the correction bound catches a shift of
about one overlap width, and the residual catches links the optimization could not
reconcile. The two defaults are proposals for the validation design
([stitching-assembly-validation](../validation/stitching-assembly.md)); neither is
accepted. Whatever the criterion, a supplied placement is always `required`, with the
reason `supplied_placement` ("Supplied placement").

% The W-324 stage-start case; W-331's validation design

## The form of the placement

| Option | Form | Effect on the golden values | Effect on existing outputs |
| --- | --- | --- | --- |
| **F1. An origin per FOV (recommended)** | `origin_z/y/x` per FOV in the sample frame; the FOV's sample-frame `ImageMetadata` is derived by `placed_metadata`. | None. | `tile_config` is a view of the raw translations; `ImageMetadata` needs no change. |
| F2. A separate transform per FOV | A push transform object per FOV (a translation, or a 4 × 4 affine) from FOV index or physical coordinates to the sample frame, stored like the registration transforms. | None. | The same views; a transform type and its persistence to define, only to hold a translation, since rotation and affine stitching are excluded. |

**Recommendation: F1.** Translation is the only kind the accepted design admits, an origin composes with
`ImageMetadata` without a new type, and the index rule above is one line. F2 becomes
worth its cost only if later work admits rotation.

% the only kind §2.10 admits

## Persistence

The placement is saved at sample level, beside the FOV checkpoints:

| Option | Layout | Effect on the golden values | Effect on existing outputs |
| --- | --- | --- | --- |
| **S1. A folder with JSON and CSV (recommended)** | `<OUTPUT>/stitching/<sample>/placement.json` (the record: frame, gauge, method provenance, table record, inputs with SHA-256, review and approvals, the SHA-256 of the two tables, `format_version` 1), `fovs.csv` and `links.csv`; for BigStitcher also `view_map.csv`, the macro and the `dataset.xml` copy's SHA-256 | None. | `output/tile_config_{sample}.csv` is still written by the legacy writer; `images/fused/{sample}/` is unchanged for the legacy rules. pandas and MATLAB read CSV. |
| S2. One JSON file | The record with both tables inline | None. | The same; larger, harder to read in a table tool. |
| S3. Parquet tables | `fovs.parquet`, `links.parquet` | None. | Needs `pyarrow` (the `checkpoint` extra) for a core result. |

**Recommendation: S1.** It follows the run-record pattern of the FOV checkpoints, records
every identity and hash next to the tables, and needs no extra. `save_placement` writes
the three files atomically; `load_placement` checks both tables' SHA-256 against the
record and raises `ValueError` on a mismatch. A table's SHA-256 is that of the bytes
`save_placement` writes for it: the CSV table format of [checkpoints](../../docs/checkpoints.md) ("Table
formats": a header row and no index, floats as `%.17g`, a missing value as `<NA>`, a
string equal to `<NA>` or starting with a backslash written with one more leading
backslash), every line ending in `\n` on every platform, encoded as UTF-8. An approval
and every record that names a placement's tables hold the SHA-256 of these bytes,
computed from the tables in memory. A placement from `placement_from_table` is
saved and loaded the same way ("Supplied placement").

## Dependency plan

Amended at the review gate of 2026-10-06, from the input inventory's spike environment
(`20261005T2025Z-spike-environment/`), the parity runs (notes sections 1 and 5) and the
trial resolution of 2026-10-06, recorded with the review gate's preparation (its
`lock-py311-trial/README.md`):

% Amended at the W-329 review of 2026-10-06, from W-323 (`20261005T2025Z-spike-environment/`), W-324 (notes sections 1 and 5)
% and the trial resolution of 2026-10-06,
% `/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/runs/W-329/20261006T1630Z-gate-review-preparation/lock-py311-trial/README.md`

* **Python bound of the package:** `requires-python >= 3.11`, written
  `requires-python = ">=3.11"` in `pyproject.toml`. multiview-stitcher 0.1.62 declares
  `Requires-Python >=3.11`, so under this bound the extra needs no lower bound of its own.
* **Extra:** `multiview-stitcher = ["multiview-stitcher==0.1.62; python_version < '3.14'"]`,
  named after the backend like the other backend extras (`stardist`, `cellpose`,
  `spotiflow`, `piscis`). The upper bound follows the library's classifiers (3.11 to 3.13)
  and the convention of the existing backend extras. The package stays an optional extra:
  on Python 3.14 and later the extra is empty, and wherever the package is not installed a
  `multiview_stitcher` call raises `StitchingBackendUnavailableError` naming the extra.
* **How they are applied.** The two changes are applied together as one commit, by the
  operator, during the preparation of an implementation batch. No issue of a batch applies
  them: no issue edits `pyproject.toml` or `uv.lock` or runs `uv lock`, and this page
  applies neither.
* **Trial resolution** (uv 0.9.28, `uv lock` on copies of `pyproject.toml` and `uv.lock`
  of `169c410`, with the existing lock as preferences; resolution only, nothing installed):
  * the bound alone removes 27 lock entries that served only Python versions below 3.11
    (four package names leave the lock) and adds none, and no remaining package changes
    version;
  * the bound and the extra remove the same 27 entries and add nine packages:
    `multiview-stitcher` 0.1.62, `ngff-zarr` 0.38.0, `itkwasm` 1.0b201,
    `itkwasm-downsample` 2.0.2, `itkwasm-downsample-wasi` 2.0.2,
    `itkwasm-downsample-emscripten` 2.0.2, `wasmtime` 49.0.0, `importlib-resources` 7.1.0
    and `rich-argparse` 1.8.0. The emscripten package is in the universal lock for that
    platform; the Linux spike environment of the input inventory installed the other eight (71.8 MB on
    disk).
* **Locked packages that change version: none.** The input inventory also resolved the extra against the
  complete locked set on Python 3.12, Linux x86_64 (`resolve-python312.txt`, every other
  package at its locked version), and installed it (`lock-diff.txt`: none differs). The
  pins the library declares are met by the lock's versions: `dask!=2025.11.0` (2024.11.2),
  `zarr>=3` (3.1.5), `ngff-zarr<0.44`, `xarray>=2024.10.0` (2026.1.0).
* **Not verified:** installation or any test on Python 3.11 or 3.13 (only 3.12 has an
  environment), and platforms other than Linux x86_64.

% the Linux spike environment of W-323; W-323 also resolved the extra

**The `pyproject.toml` change** (proposed; applied by the batch-preparation commit, not by
this page):

```toml
[project]
requires-python = ">=3.11"

[project.optional-dependencies]
# Stitching backend (§2.10). multiview-stitcher 0.1.62 requires Python 3.11, as the
# package does, and lists 3.11 to 3.13, so the extra is empty on 3.14 and later.
multiview-stitcher = [
    "multiview-stitcher==0.1.62; python_version < '3.14'",
]
```

The `requires-python` line replaces the current one and the extra is added after
`cellpose`; nothing else in the file changes. Markers that already name Python 3.11 (for
example in the `anndata` and `stardist` extras) stay as they are; under the new bound their
branches for older versions can no longer be selected.

**The `uv.lock` change**, written by `uv lock` in the same commit:

* `requires-python = ">=3.11"`, and the 27 entries of the trial removed;
* the `starfinder` entry gains the extra under `[package.optional-dependencies]`, its
  requirement (specifier `==0.1.62`, marker `python_full_version < '3.14'` with
  `extra == 'multiview-stitcher'`) under `[package.metadata] requires-dist`, and
  `"multiview-stitcher"` in `provides-extras`, as `uv lock` writes them;
* the nine new `[[package]]` entries, at the versions above, with their sdist and wheel
  URLs and hashes;
* no remaining `[[package]]` entry changes version.

Verification for whoever applies it: `uv lock`; `uv export --locked --all-extras
--all-groups --no-hashes` before and after, whose difference must be exactly that of the
trial (the 27 entries removed and the nine packages added); `uv sync --extra
multiview-stitcher` in a fresh environment; the routine test gate. A difference beyond the
trial returns to Jiahao, as the assignment stage's lock change did.

% as the §2.9 lock change did

## Workflow configuration

A Python-only `stitching` block (`backend: python`), read by `_run_stitching` in the
workflow adapter and by the workflow rule that runs it:

% the §2.13 rule that runs it

```yaml
stitching:
  method: bigstitcher            # or multiview_stitcher; bigstitcher is the initial default
  input: reference_merged        # or reference_stain, or a file pattern
  positions:
    source: stage_file           # stage_file | regular_grid | table
    stage_axes: {x: "+x", y: "+y"}   # required with stage_file; no default
    snap_to_grid: {overlap: 0.1} # stage_file only: GridSnap with tile_shape_yx and spacing_yx below; null keeps stage positions
    regular_grid: {columns: 7, rows: 10, order: column_major_down_right, overlap: 0.1}
    tile_shape_yx: null          # px; null: from the FOVs' ReferenceGrid (shape_zyx[1:])
    spacing_yx: null             # µm; null: from the FOVs' ReferenceGrid (metadata.spacing_zyx[1:])
    table: null                  # a CSV path with source: table
    stage_file: /data/acquisition/positions.maf   # required with source: stage_file; no default
  config: {}                     # fields of MultiviewStitcherConfig or BigStitcherConfig
  review: {max_correction_fraction: 0.5, max_residual_um: 1.0}
  fallback: null                 # or nominal; for multiview_stitcher and the BigStitcher import alike
  bigstitcher_save_dir: null     # a local folder; required with method: bigstitcher
```

% `method`: bigstitcher is the initial default (D1)

* Without the block the four legacy rules run unchanged with `backend: matlab`, and the
  adapter's legacy translation (`StageAxisMapping(x="+x", y="+y", source="legacy")`, the
  YAML voxel sizes) applies only to the legacy writer. They run with the MATLAB backend
  only: a Python configuration that enables one without the block raises when the rule
  files are parsed, naming the block.
* The sample's FOVs are its rows of `fovs.csv` in the input root, restricted by
  `selected_fovs`. With `source: stage_file`, `stage_file` is required (no default: the
  Python route writes no `documents/maf/<sample>.maf` copy) and `fovs.csv` must have the
  `position_id` column; the file may hold every sample's positions, and the sample's are
  selected by `position_id` before the conversion. A supplied table (`source: table`) may
  also hold other FOVs: its rows of the sample's selected FOVs are kept. A regular grid is
  laid out over the sample's FOVs of `fovs.csv` and its selected rows kept, so that a
  selection keeps each FOV's cell.
* `fiji_path` is read by the BigStitcher engine rule when it runs, not when the rule file
  is parsed; the key stays required for that rule only.
* With `method: bigstitcher`, `stitch_sample` writes and declares
  `output/tile_config_{sample}.csv` only without an `assembly` block (since 2026-10-24): with one,
  nothing reads it.
* The static schema lists the method names and is kept equal to `STITCHING_METHODS` plus
  `bigstitcher` by a default-tier test, as for the other registries.

% only without an `assembly` block (W-568)

## Exclusions

No implementation, rename, dependency or lock change; no overlap ownership, sample
coordinates of molecules or cells, or export (sample assembly and the sample outputs); no rotation, affine or
deformable stitching, no GPU, no backend ranking or default change; no real data (every
measured number comes from the input inventory and the parity runs by reference).

% (W-326, W-328); from W-323 and W-324 by reference


% toctree: Stitching and assembly validation (§2.10)
