# Sample-output contract: the fused image, the SpatialData export and the output profiles

Status: Proposed

This page proposes what a sample has on disk after spatial assembly: the output profiles
and the workflow keys that select them; the fused image as a 2D projection (the default)
or a 3D stack, at an adjustable scale factor, as OME-TIFF (the default) or OME-Zarr; the
SpatialData export of the sample tables; the optional assembled label export; the
read-back checks; and the resources each output needs. It builds on three contracts and
redefines none of them:

* {doc}`stitching-contract`: `PlacementResult` (per-FOV `status`, float64 `origin_z/y/x`
  in µm), the sample frame (`frame_id` `sample:<sample>`, ZYX, µm, identity direction),
  the index-to-sample rule `sample_zyx = origin_zyx + D · (p ⊙ spacing_zyx)` and its
  inverse, `placed_metadata`, the failure policy and `PlacementReviewRequiredError`;
* {doc}`assembly-contract`: `AssemblyResult` and its sample tables (`fovs`, `cells`,
  `molecules`, `counts`), `sample.h5ad`, the record `assembly.json`, the persistence
  folder `<OUTPUT>/assembly/<sample>/`, footprints and centres, `ownership_map` with
  options O1, T1 and Z1, the sample-level identities and the count accounting;
* the `ReferenceGrid` of {doc}`segmentation-contract` and `ImageMetadata`
  (`src/python/starfinder/image.py`).

Where this page names a column, a function or a rule of those pages, it uses it unchanged.
The current exporter work and today's fused files are recorded in
{doc}`sample-output-baseline`. Nothing here is implemented, nothing is renamed and no
dependency changes. The validation design and the implementation issues are W-331's.

Evidence, cited by path; this page repeats only the numbers it needs:

* W-323, `/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/runs/W-323/`:
  `20261005T2110Z-input-inventory/inputs.md` (tile shapes, dtypes, calibration),
  `20261005T1930Z-fiji-baseline/fiji-baseline.md` (BigStitcher's stored levels, CIFS).
* W-324, run directory
  `/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/runs/W-324/20261006T063540Z-5cb2f421`
  (found with `timeout 30 find …/runs/W-324 -maxdepth 2 -name stitching-manifest.json`):
  `notes.md` sections 2.2 and 3, `tables/runs.csv` (load times), `fusion/fusion_probe.json`
  (fusion of one LN region and the label-fusion cases) and the per-FOV placements
  `runs/ln-grid-b4-t4-rep2/per_fov.csv`, `runs/culture3d-grid-b4-t4/per_fov.csv`,
  `runs/tissue2d-grid-b4-t4/per_fov.csv`.
* W-328, run directory
  `/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/runs/W-328/20261006T111902Z-ad45aa68`:
  `scripts/resource_estimate.py` and its output `tables/resource_estimate.csv`, which
  derive the sizes of "Resources" from the W-323 and W-324 records without reading an
  image.

## Settled decisions this page follows

Decided by Jiahao in the §2.10 planning session (W-152 comment "§2.10 planning decisions
(2026-10-05)" and its reply "Amendment on the sample outputs"), and in the W-152 comments
of 2026-09-27 on §2.10:

* **Three sample-level entries** outside `PipelineConfig` and `FOV.run`: stitch, assemble,
  and fuse and export (D3). The fuse and export entry is separate from stitch and
  assemble; the workflow runs it by default and a key turns fusion off.
* **By default** every assembled sample has a sample-level AnnData (written by assemble,
  {doc}`assembly-contract`) and a fused image as OME-TIFF, a **2D projection** at an
  **adjustable scale factor**, because the image at acquired size is rarely needed.
* **On request**: a SpatialData export of the sample, a 3D stack, OME-Zarr and further
  channels. In the SpatialData store the image is **written once as OME-Zarr** and not
  duplicated.
* **Viewer qualification** in Fiji and napari is **deferred**; opening the files in a viewer
  is not a requirement now.
* 2026-09-27: both image formats are supported; assembled labels are inspection products;
  mask conflicts are flagged, not resolved.

Excluded: viewer sessions, reassignment of molecules or recomputation of counts from a
fused image or a label raster, cross-FOV cell matching, and any change to the placement,
assembly or §2.9 contracts.

## Terms

* The **fused image** is one image of a sample in the sample frame, made from one saved
  image per FOV (a **source image**) and the placement.
* The **fusion grid** (level 0) is the voxel grid of the fused image at the acquired
  spacing ("The fusion grid"). A file's **output level** is the fusion grid reduced by an
  integer factor `k` per axis; its **scale factor** is `1/k`. A file's further
  **resolution levels** halve its output level in Y and X.
* The **3D stack** is the fused image with a Z axis; the **2D projection** collapses Z.
* An **output profile** is the set of files a finished sample has for a given
  configuration ("Output profiles").
* The **store** is the SpatialData Zarr store of one sample.
* A FOV is **included** in fusion when it has a finite origin and a source image: status
  `image_supported` or `metadata_only` ({doc}`stitching-contract`). `placeholder` FOVs (a
  blank image), `missing` and `failed` FOVs contribute nothing.

## Names

| Option | Names | Effect on the golden values | Effect on existing files and outputs |
| --- | --- | --- | --- |
| **N1. One `starfinder.sample_output` module (recommended)** | Entry `write_sample_outputs` and coordination `Dataset.sample_outputs`; fusion `fuse`, `fused_grid`, `FusionInput`, `FusionConfig`, `FusedGrid`, `FusionRecord`; export `export_spatialdata`, `SpatialDataExportConfig`; labels `fuse_labels`; checks `check_fused`, `check_spatialdata_export`; lookup `lookup_source` | None: the golden tests pin the legacy scripts and are untouched. | New files only, plus one method on `Dataset`; `docs/api/sample_output.rst`, `docs/api/inventory.rst` and `docs/api/python-index.rst` gain the exports, which `docs/check_reference.py` requires. The legacy `images/fused/{sample}/` folder is not written. |
| N2. Two modules, `starfinder.fusion` and `starfinder.sample_export` | The same functions split by output; `sample_export` is the reference-branch name | None. | The same files; one entry of D3 is then spread over two modules, and `sample_export` reuses a name whose reference-branch API (`export_sample` with affine source mappings) this page does not adopt. |
| N3. Inside `starfinder.assembly` | `starfinder.assembly.fuse`, `export_spatialdata` | None. | The same files; the assembly module grows image I/O and optional zarr and spatialdata imports, which D3 keeps apart from assemble. |

**Recommendation: N1.** It mirrors D3 (one module per sample-level entry: `stitching`,
`assembly`, `sample_output`), keeps the optional zarr and spatialdata imports out of the
assembly module, and does not reuse a reference-branch name for a different API.

New names under N1:

| Concept | Name | Kind |
| --- | --- | --- |
| Entry | `write_sample_outputs(placement, inputs, *, assembly=None, config=SampleOutputConfig(), output_dir)` | function |
| Coordination | `Dataset.sample_outputs(sample, *, placement, assembly, config=SampleOutputConfig(), output_dir)` | method |
| Config | `SampleOutputConfig(fuse=True, fusion=FusionConfig(), spatialdata=False, labels=False)` | frozen dataclass |
| Fusion | `fuse(placement, inputs, *, ownership, config, output_dir) -> FusionRecord` | function |
| Fusion input | `FusionInput(fov_id, grid, images)` | frozen dataclass |
| Fusion config | `FusionConfig` ("Fusion config") | frozen dataclass |
| Geometry | `fused_grid(placement, grids, *, config) -> FusedGrid` | function, frozen dataclass |
| Record | `FusionRecord`: the content of `fused.json` and the written paths | frozen dataclass |
| Labels | `fuse_labels(placement, assembly, territories, *, config, output_dir)` | function |
| Export | `export_spatialdata(assembly, *, fused=None, labels=None, config=SpatialDataExportConfig(), output_dir) -> Path` | function |
| Checks | `check_fused(folder)`, `check_spatialdata_export(folder)` | functions; raise `ValueError` naming the first failed condition |
| Lookup | `lookup_source(folder, *, spot=None, cell=None)` | function |

## The fuse and export entry

```python
def write_sample_outputs(placement: PlacementResult, inputs: Sequence[FusionInput], *,
                         assembly: AssemblyResult | None = None,
                         config: SampleOutputConfig = SampleOutputConfig(),
                         output_dir) -> dict: ...
```

calls, in this order, `fuse` (unless `config.fuse` is false), `fuse_labels` (with
`config.labels`) and `export_spatialdata` (with `config.spatialdata`), and returns the
paths of the records it wrote. It is the D3 entry; the three functions are its parts and
can be called alone. Like `stitch` and `assemble` it is a sample-level computation, not a
multi-FOV runner ({doc}`stitching-contract`, "Not the multi-FOV runner").

Unlike `stitch` and `assemble`, `fuse` and `fuse_labels` write as they compute and return
only their record, because a fused image does not fit in memory at every size ("Bounded
memory").

`Dataset.sample_outputs(sample, *, placement, assembly, config, output_dir)` loads the
saved placement (`load_placement`) and assembly (`load_assembly`), builds one
`FusionInput` per included FOV from the saved source images and reference grids
(`reference_grid_from_file`), and calls `write_sample_outputs`.

**Checks**, in this order, before anything is written; each raises and names the FOVs:

1. **Review.** A placement whose review is `required` without an approval raises
   `PlacementReviewRequiredError`, exactly as `assemble` does ({doc}`assembly-contract`,
   "Checks the entry applies", 2).
2. **Assembly.** With `assembly`, its record's placement SHA-256 values equal the
   placement's (`ValueError`), so the image and the tables come from one placement.
3. **Inputs.** Every included FOV has exactly one `FusionInput`; no input names another
   FOV (`ValueError`).
4. **Grid.** Each input's grid equals the grid the placement recorded for that FOV, by the
   grid rule of {doc}`assignment-contract`; known spacing, unit `µm`, identity direction
   (`IncompatibleGeometryError`). All included FOVs share one spacing, as `stitch`
   requires.
5. **Images.** Each named source image exists for every included FOV, has the grid's shape
   and one dtype across FOVs (`ValueError` naming the dtypes; no implicit conversion), and
   that dtype is one of uint8, uint16, uint32, int8, int16, int32, float32 and float64, the
   real dtypes the pinned tifffile writes to OME-TIFF (`tifffile.py` lines 14946 to 14963
   of 2026.1.28); any other dtype (uint64, int64, float16, bool, complex) raises
   `ValueError` naming it and the allowed list ("Dtypes written").
6. **Config.** `FusionConfig.__post_init__` validates every field ("Fusion config").

## Output profiles

A finished sample has its outputs in three folders beside its placement
(`stitching/<sample>/`): `assembly/<sample>/`, `fused/<sample>/` and `export/<sample>/`.
`<image>` is the source-image name of "Fused image inputs" (`reference_stain` by default);
`<sample>` the sample name.

| Profile | Selected by | Files that exist | Format | What each contains |
| --- | --- | --- | --- | --- |
| Always (assembly) | an `assembly` block ({doc}`assembly-contract`, "Workflow configuration") | `assembly/<sample>/assembly.json`, `fovs.<fmt>`, `cells.<fmt>`, `molecules.<fmt>`, `counts.<fmt>` | JSON; CSV by default or Parquet | The record and the four sample tables of {doc}`assembly-contract`, "Sample tables" |
| | | `assembly/<sample>/sample.h5ad` | H5AD | The sample-level AnnData of {doc}`assembly-contract`: one row per counted cell; written by assemble, so it is part of every profile, the default included |
| Default | nothing (the `sample_outputs` defaults) | the above, plus `fused/<sample>/fused.json` | JSON | The fusion record ("Run record") |
| | | `fused/<sample>/<image>_projection.ome.tif` | OME-TIFF, BigTIFF, tiled | The maximum projection of `<image>`, at scale 0.5 in Y and X, in the input dtype ("Dtypes written"), with calibration and origin ("OME-TIFF layout"); not written when no FOV is included ("The fusion grid") |
| No fused image | `fuse: false` | the "Always" files only | | No `fused/<sample>/` folder |
| 3D stack | `dimensionality: stack` (or `both`) | `fused/<sample>/<image>_stack.ome.tif` instead of (or beside) the projection | OME-TIFF | The 3D fused image at `scale_yx` in Y and X and `scale_z` in Z |
| OME-Zarr, without SpatialData | `formats: [ome_zarr]` (or both formats), `spatialdata: false` | `fused/<sample>/<image>_projection.ome.zarr` (and `_stack.ome.zarr`) | OME-NGFF 0.4 on Zarr v2 ("OME-Zarr layout") | The same image values as the OME-TIFF, with resolution levels |
| Further channels | `images: [reference_stain, reference_merged, …]` | one file per image name and dimensionality, as above | as above | Each selected source image fused on the same grid with the same placement and ownership |
| SpatialData | `spatialdata: true` | `export/<sample>/export.json`, `export/<sample>/sample.zarr/` | JSON; SpatialData store on Zarr v2, container format 0.1, raster groups in spatialdata's raster format 0.2 (NGFF 0.4 layout, version string `0.4-dev-spatialdata`) | Points `molecules`, shapes `cell_centroids`, tables `cells`, `expression`, `cells_unplaced`, `molecules_unplaced`, and one image element `<image>_projection` (and `<image>_stack`) per fused image, each written when not empty ("SpatialData export", "Empty populations"). The OME-Zarr image exists only here: no `fused/<sample>/*.ome.zarr` is written. The OME-TIFF files of `formats` are still written. |
| SpatialData without a fused image | `spatialdata: true`, `fuse: false` | `export/<sample>/` as above without image elements | | `export.json` records each image element as `omitted`, reason `not_requested` |
| Pyramid in OME-TIFF | `pyramid_levels: L` | the same files | SubIFD levels | `L` further levels, each halving Y and X ("OME-TIFF layout") |
| Assembled labels (optional, proposed for a later delivery) | `labels: true` | `fused/<sample>/labels.json`; with SpatialData the labels element `sample.zarr/labels/cell_labels`, without it `fused/<sample>/cell_labels.ome.tif` | uint32 | Sample-level cell instance IDs placed without blending ("Assembled label export") |

A reader checks a finished sample against the table with the record: `fused.json` lists
every fused file with its SHA-256, shape, dtype, dimensionality, scale factor and format,
and `export.json` every element of the store; a file that exists but is not listed, or is
listed but missing, fails `check_fused` or `check_spatialdata_export`.

The new folders `<OUTPUT>/fused/<sample>/` and `<OUTPUT>/export/<sample>/` are not the
legacy `<OUTPUT>/images/fused/<sample>/`, which keeps BigStitcher's inputs and
`dataset.xml` ({doc}`stitching-baseline`); no existing file is rewritten.

## The fused image

### Fused image inputs

* **A placement** (`PlacementResult`, saved as `stitching/<sample>/`), from either backend:
  the BigStitcher import and `multiview_stitcher` give one type, and both are fused by the
  same code ("Fusion implementation").
* **Per included FOV, the selected source images on its reference grid**, named as the
  stitching inputs are ({doc}`stitching-contract`, "Input image"): `reference_stain` (the
  saved `reference_stain` of `FOV.prepare_morphology`) by default; `reference_merged`
  (`images/ref_merged/{fovID}.tif`); or `<name>=file:<pattern>` with `{fov_id}`, accepted
  only when its shape equals the reference grid's, where `<name>` names its files and store
  element and may use letters, digits, `_`, `-` and `.` only. Further channels are further
  names in `images`. A source image is read from its saved file; `fuse` holds no FOV image
  it is not using ("Bounded memory").
* **The ownership settings**: `ownership` and `tie` of the assembly record when an assembly
  is given, else the defaults of `AssemblyConfig` (O1, T1); the record says which.

### The fusion grid

The fusion grid is computed from the placement and the included FOVs' grids only, by
`fused_grid`:

* **Spacing** `s = (s_z, s_y, s_x)`: the included FOVs' shared spacing, µm.
* **Origin** `o`: per axis, the smallest origin of an included FOV,
  `o_a = min_f origin_f,a`. The FOV with the smallest origin on an axis therefore has its
  voxel centres on grid centres along that axis.
* **Shape** `n0`: per axis `ceil((max_f (origin_f,a + (n_f,a − 1)·s_a) − o_a) / s_a − 1e-9) + 1`,
  so the last voxel centre of every included FOV lies within the grid (the tolerance keeps
  a whole number of voxels from rounding up, as the synthetic sample grid does,
  {doc}`synthetic-sample-scene`).
* **Metadata**: `ImageMetadata(frame_id="sample:<sample>", spacing_zyx=s, origin_zyx=o,
  direction_zyx=identity, spatial_unit="µm")`. Index `v` of the grid is at sample position
  `o + v ⊙ s`.
* **Z = 1 FOVs.** The grid has one plane when every included FOV has Z = 1 and the same
  `origin_z`. Neither earlier contract requires that: {doc}`stitching-contract` places a
  Z = 1 grid as a plane but fixes only relative positions, and the one-layer check of
  {doc}`assembly-contract` ("Checks the entry applies", 8) compares `nominal_z`, not
  `origin_z`. When Z = 1 FOVs have different `origin_z`, the shape formula gives `n0_z > 1`
  (unit spacing and Z origins 0 and 1.4 give `n0_z = 3`), each FOV covers the planes its
  one-plane footprint contains, and the other planes are uncovered there. `fuse` does not
  refuse this; the record lists `n0_z` and the per-FOV `origin_z`, and the projection
  collapses the planes as for any stack.

Excluded FOVs (`placeholder`, `missing`, `failed`) do not extend the grid; the area only
they would cover is uncovered.

**No included FOV.** When no FOV is included (every FOV excluded, or a placement whose
placed FOVs are all placeholders), there is no grid: `fuse` writes no image file and writes
`fused.json` with `outputs: []` and `reason: "no_included_fov"`; `check_fused` accepts that
record only when the placement indeed has no included FOV. One included FOV gives a grid
equal to that FOV's grid (`o` its origin, `n0` its shape), with no overlap and no
uncovered voxel.

### The value of a voxel

For an output voxel with centre `c` (sample frame, µm) at level 0:

1. **Candidates**: the included FOVs whose footprint contains `c`. The footprint is the
   assembly footprint `[origin − s/2, origin + (n − ½)·s)` per axis
   ({doc}`assembly-algorithms`, "Footprints and centres"), in Y and X for the projection
   and in Z, Y and X for the stack.
2. **Overlap rule**: one candidate is chosen ("Overlap rule"; recommended V1: the
   ownership rule).
3. **Sampling**: the chosen FOV's voxel that contains `c`: index `floor(p + ½)` per axis,
   with `p = (c − origin_f) ⊘ s` the inverse rule of {doc}`stitching-contract`. This is the
   §2.9 sampling rule ({doc}`assignment-contract`, "Coordinate sampling"); no
   interpolation.
4. **No candidate**: the voxel is **uncovered** and holds 0. Uncovered voxels are counted
   in the record and excluded from the block means of "The scale factor".

Level 0 of the stack therefore holds input values only: **the dtype is preserved** and no
new value is created. The projection holds per-FOV projections, in the dtypes of "Dtypes
written". The scale factor reduces level 0 by block means, rounded into the level-0 dtype.

### Dtypes written

For every allowed input dtype `d` (check 5 of "The fuse and export entry"), every
configuration writes one definite dtype, the same in OME-TIFF and OME-Zarr:

| Output | Input dtype | Dtype written (OME-TIFF and OME-Zarr) | Rule |
| --- | --- | --- | --- |
| Stack, any scale factor and level | any allowed `d` | `d` | Level 0 holds input values; block means are accumulated in float64 and rounded half to even into `d` for integer `d` (a mean lies between the block's extremes, so it never overflows `d`), kept as `d` for float `d` (a mean of finite float32 values is finite in float32). |
| Projection, `projection: max` | any allowed `d` | `d` | `project_image` keeps the dtype for `max`; then as the stack. |
| Projection, `projection: sum` | uint8, uint16, uint32, int8, int16, int32 | float64 | `project_image` sums integers into uint64 or int64 (`src/python/starfinder/preprocessing/projection.py` line 60), which OME-TIFF cannot store. The sum is computed with `ProjectionConfig(method="sum", output_dtype="float64")`, a checked cast (`src/python/starfinder/io/conversion.py`, `_cast`, lines 23 to 39). It is exact while the number of planes times the dtype's largest magnitude is at most 2^53 (for 32-bit inputs, up to 2^21 = 2,097,152 planes); `fuse` checks that bound before computing and raises `ValueError` naming it otherwise. Block means of the float64 projection stay float64. |
| Projection, `projection: sum` | float32, float64 | float64 | `project_image` sums floats in float64 and raises `ValueError("sum projection overflow")` on a non-finite result (lines 62 to 63). |
| P3 (one plane) and P4 (maximum of the stack), options not offered as keys | any allowed `d` | `d` | Input values, as the stack. |
| Label export, any place, any scale factor and level | uint32 labels | uint32 | Block mode, no arithmetic; more than 2³² − 1 cell rows is an error ("SpatialData export"). |

OME-TIFF accepts every dtype of the third column (`tifffile.py` lines 14946 to 14963:
`uint8`, `uint16`, `uint32`, `int8`, `int16`, `int32`, `float32` as `float`, `float64` as
`double`). Zarr stores any numpy dtype; the rule above keeps the two formats equal. The
dtype is recorded per file in `fused.json`, and OME-TIFF and OME-Zarr check 2 compare it.

Converting the sum to float64 or refusing `projection: sum` with OME-TIFF is a design
choice; this page recommends the conversion (worker notes, choice 11).

### Overlap rule

| Option | Rule | Cost | Consequence |
| --- | --- | --- | --- |
| **V1. Ownership (recommended)** | The candidate that owns `c` by the assembly's ownership rule (O1 nearest footprint centre in Y and X, T1 table order on exact ties), restricted to the candidates. For the projection the candidates are decided in Y and X, so where the owner has a source image the chosen FOV is exactly `ownership_map(placement, grids, config).owner(c)`; where the owner has none (a `placeholder`), the nearest-centre candidate is chosen. For the stack a candidate must also cover `c` in Z; where the Y and X owner covers `c` in Z it is chosen, otherwise the nearest-centre candidate that does. | One `ownership_map` evaluation per output chunk (a nearest-centre test against the few FOVs that overlap the chunk); no weights. | The image at a point comes from the FOV whose molecules and cells assembly retains there, so image and tables agree. Values are input values; seams are visible where neighbouring FOVs differ in brightness. The same rule places labels without blending. |
| V2. Linear blending | Weighted mean of all candidates, weight falling linearly to 0 at each footprint edge (Fiji Grid/Collection "Linear Blending", today's fused files; multiview-stitcher's default weighted average) | One weight per candidate per voxel and a float accumulator per chunk; W-324's region probe with multiview-stitcher's weighted average and linear interpolation: 19.9 s wall, 70 s CPU for 413 million voxels written as TIFF | Smooth seams; the overlap shows a mix of two FOVs, possibly misaligned by the placement error (W-324: up to about 10 px between backends on LN); values not present in any input; cannot place labels (W-324 `fusion_probe.json`: weighted fusion created label IDs 7 to 10 from inputs 1 to 6 and 11 to 16). |
| V3. Maximum | The largest candidate value | One comparison per candidate | No seams for sparse amplicon signal; doubles a molecule seen twice at slightly different places; biased bright in overlaps. |
| V4. First in table order | The candidate first in the tile position table order | None beyond the candidate test | Deterministic, but the overlap shows the FOV whose molecules assembly may have excluded there. |

**Recommendation: V1.** It is the only rule under which the fused image, the sample tables
and an assembled label image say the same thing about a point, it keeps input values and
dtype exactly, and it reuses the assembly's rule instead of adding one. The seams it shows
are the honest state of the data; a blended display is a viewer matter and is deferred.

### Bounded memory

No whole-sample array is held in memory, at any size:

* The output is computed in **bands** of whole output tiles: `512·k_y` rows of level 0
  (one row of 512-pixel tiles at the output level), full width, one plane at a time for
  the stack. For each band, only the FOVs whose footprint meets the band are read.
* **Projection**: each FOV's projection is computed once from its own volume ("The 2D
  projection") and kept (one Y × X plane per FOV); at most one FOV volume is in memory at
  a time.
* **Stack**: for each output plane, at most one plane of each overlapping FOV is in
  memory, read from its file page by page.
* **Resolution levels** of an OME-TIFF are accumulated from the output-level bands as
  they are computed (together at most a third of the output level's bytes) into a
  temporary file on local scratch, not in memory, and written from it after the output
  level: tifffile writes each SubIFD level as one series after the whole main series
  (`tifffile.py` lines 1980 to 1989 of the locked 2026.1.28). OME-Zarr levels are computed
  region by region from the written output level.
* **Inside the SpatialData store** spatialdata computes every level in one `dask`
  computation (`spatialdata/_io/io_raster.py` lines 349 to 350 of 0.6.1). Whether that
  computation stays within the bound with levels given as one lazy graph is not
  established here; it is an assumption for the implementation issue to measure (record
  the peak RSS), with a temporary output-level array on local scratch as the fallback.

The peak is about the FOV projections plus one FOV volume plus one band for the
projection, and one plane per overlapping FOV plus one band for the stack ("Resources"
gives the figures per sample). The record states the band size and the measured peak
RSS.

### Run record

`fused/<sample>/fused.json`, written last and atomically:

* `format_version` 1, `stage` `fusion`, `sample`, `frame_id`;
* the placement: method, gauge, review state and approval, the SHA-256 of `fovs.csv` and
  `links.csv` as `placement.json` records them; the assembly record's SHA-256 and the
  ownership settings used (or `defaults`);
* per FOV: `fov_id`, placement status, included or the reason it is not, origin, grid
  (shape, metadata, SHA-256), and per source image its path and SHA-256;
* the config (`FusionConfig` fields), the fusion grid (`n0`, `s`, `o`), and per output
  file: path, SHA-256, format, image name, dimensionality, projection rule, factors `k`,
  shape, dtype, the level's spacing and origin, the projected z slab for a projection, the
  resolution levels with their shapes, spacings and origins, tile or chunk shape and
  compression;
* counts: voxels at level 0, uncovered voxels, voxels per chosen FOV;
* `reason: "no_included_fov"` with `outputs: []` when no FOV is included ("The fusion
  grid");
* `software` as in `run.json` ({doc}`checkpoints`), the band size, wall and CPU time and
  peak RSS.

```json
{
  "format_version": 1, "stage": "fusion", "sample": "sample1", "frame_id": "sample:sample1",
  "placement": {"method": "bigstitcher", "gauge": "G1", "review": "not_needed",
                "fovs_sha256": "...", "links_sha256": "..."},
  "ownership": {"source": "assembly", "assembly_sha256": "...", "ownership": "nearest_center", "tie": "table_order"},
  "grid": {"shape_zyx": [68, 12333, 10883], "spacing_zyx": [0.346255, 0.194398, 0.194398],
           "origin_zyx": [-2.6205, -8.6413, 1.0461]},
  "outputs": [{"path": "reference_stain_projection.ome.tif", "sha256": "...", "format": "ome_tiff",
               "image": "reference_stain", "dimensionality": "projection", "projection": "max",
               "factors_zyx": [1, 2, 2], "shape_zyx": [1, 6167, 5442], "dtype": "uint8",
               "spacing_zyx": [23.5453, 0.388796, 0.388796], "origin_zyx": [8.9790, -8.5441, 1.1433],
               "levels": [], "tile_yx": [512, 512], "compression": "zlib"}],
  "counts": {"level0_voxels": 0, "uncovered": 0, "per_fov": {}},
  "software": {"starfinder": "...", "git_commit": "...", "packages": {}}
}
```

The numbers are an illustration of the layout, not results.

## The 2D projection

The default fused image. Its Y and X grid is the stack's at the same scale factor; its
one plane represents the whole Z extent.

**Computation, without building the 3D volume.** For each included FOV, its source image
is projected once along Z with `project_image` and `ProjectionConfig(method=…, axis="z")`
(`src/python/starfinder/preprocessing/projection.py`, unchanged). The projections are then
fused in Y and X by "The value of a voxel" with the candidates decided in Y and X. A FOV's
projection uses all its own planes; no Z resampling happens.

**Relation to the 3D stack when FOVs have different Z offsets.** The projection at a point
is the projection of the owning FOV's own column. The stack at the same point takes, plane
by plane, the owner where it covers the plane and another candidate where it does not, and
it resamples Z to the fusion grid. Therefore:

* **Equal Z** (every included FOV has the same `origin_z` and Z size, so every FOV covers
  every plane of the fusion grid, whose planes coincide with the FOVs'): at scale factor 1,
  `projection[0, y, x] == stack[:, y, x].max()` for every covered `(y, x)` under the
  maximum rule, exactly. At other scale factors both are block means of their level-0
  images, and the maximum of block means is not the block mean of maxima, so the identity
  is stated at level 0 only.
* **Different Z offsets** (W-324: LN z corrections from −7.6 to 10.4 slices,
  cell-culture-3D from −4.1 to 10.7): the projection can differ from the maximum of the
  stack where a non-owner FOV covers planes the owner does not, or where the Z resampling
  of the stack moves a plane by up to half a slice. The projection is then the maximum of
  the owner's own planes, which is what assembly's ownership means for a column.

**Geometry.** The projection's one plane is the slab `[z_lo, z_hi)`, the Z extent of the
fusion grid: `z_lo = o_z − s_z/2`, `z_hi = o_z + (n0_z − ½)·s_z`. Its metadata has the
fusion grid's Y and X at the output level, and in Z spacing `S_z = z_hi − z_lo = n0_z·s_z`
and origin `O_z = (z_lo + z_hi)/2 = o_z + s_z·(n0_z − 1)/2`: the stack's formula of "The
scale factor" with `k_z = n0_z`, one block over the whole Z extent. `world_to_index` then
maps every z of the slab to index 0 (by the `floor(j + ½)` rule) and every Y and X like
the stack's. Example: `n0_z = 4`, `s_z = 1`, `o_z = 0` give the slab `[−0.5, 3.5)`,
`S_z = 4` and `O_z = 1.5`. This differs on purpose from
`ImageMetadata.projected`, which leaves the physical fields of a per-FOV projection
unknown; the per-FOV rule is not changed (worker notes, choice 5).

| Option | Projection rule | Cost | Consequence |
| --- | --- | --- | --- |
| **P1. Maximum (recommended)** | `max` over Z | Keeps the dtype; one pass over each FOV | Today's practice (`MAX_*.tif`); spots and nuclei at any depth stay visible; noise floor rises with the number of planes. |
| P2. Sum | `sum` over Z (`ProjectionConfig` `sum`), written as float64 ("Dtypes written") | 8 bytes per pixel, 8 times the uint8 default | Proportional to total signal; not the input dtype, so "dtype preserved" no longer holds. |
| P3. One plane | The owner's plane nearest to a given sample z | Reads one plane per FOV | Cheapest; loses everything out of that plane. |
| P4. Project the stack | `max` over Z of the 3D stack, computed band by band | The cost of the 3D stack ("Resources": LN 8.5 GiB of level 0 fused) | Equals the maximum of the stack by construction, also with different Z offsets. |

**Recommendation: P1, computed projection-first.** It is today's practice, keeps the
dtype, and needs one read of each FOV and no 3D fusion; P4 is the option if exact equality
with the stack matters more than the cost.

## The scale factor

**Meaning per axis.** `scale_yx = 1/k_yx` with `k_yx` a whole number ≥ 1, applied to Y and
X alike; for the stack `scale_z = 1/k_z` (default 1, Z kept as acquired, as today's
`*_05` stacks keep it). The projection has no Z factor. A value whose reciprocal is not a
whole number within 1e-9 is refused by `FusionConfig`.

**Resampling rule.** The output level is level 0 reduced by **block means** over
`k_z × k_y × k_x` blocks, accumulated in float64 over the covered voxels of the block only
(a block without a covered voxel is 0, uncovered), and rounded half to even into the input
dtype for integer dtypes (float dtypes keep their dtype). The level-0 grid is padded at its
far end with uncovered voxels to a whole number of blocks. This is the reduction geometry
of the reference-branch `raster.py` ({doc}`sample-output-baseline`), rewritten per chunk.

**Achieved spacing and origin**, recorded in the file and in the run record, per file
and per axis `a`, with `s`, `o`, `n0` the fusion grid's:

| File | Axis | Spacing `S_a` | Origin `O_a` | Size `N_a` before padding |
| --- | --- | --- | --- | --- |
| Stack | Z, Y, X | `k_a·s_a` | `o_a + s_a·(k_a − 1)/2` | `ceil(n0_a / k_a)` |
| Projection | Y, X | `k_a·s_a` | `o_a + s_a·(k_a − 1)/2` | `ceil(n0_a / k_a)` |
| Projection | Z | `n0_z·s_z` (the slab) | `o_z + s_z·(n0_z − 1)/2` | 1 |

The projection's Z row is the stack's formula with `k_z = n0_z`. So the centre of output
voxel `w` is the centre of its block (for the projection's Z, of the slab), and sample
coordinates map onto the image by the same metadata rule in every file.

**Resolution levels inside a file.** A file with `L` further resolution levels (OME-TIFF
`pyramid_levels`, OME-Zarr automatic) has its output level padded at the far end of Y and
X with uncovered voxels (value 0) to a multiple of `2^L`, so every level is exactly half
the one before. spatialdata infers a level's scale from the ratio of shapes
(`spatialdata/transformations/_utils.py` lines 104 to 109 of 0.6.1), which is then exactly
`2^ℓ`; an OME-TIFF SubIFD level carries no geometry of its own besides its shape (how
viewers derive it is part of the deferred viewer qualification). The padded size is recorded beside `N_a`. Level `ℓ` holds the mean of
each `2^ℓ × 2^ℓ` block of the output level in Y and X (Z kept), over all its voxels,
uncovered and padding voxels counting as 0, rounded half to even into the dtype; it can be
recomputed from the file alone. Its geometry is `S_a^(ℓ) = 2^ℓ·S_a` and
`O_a^(ℓ) = O_a + S_a·(2^ℓ − 1)/2` in Y and X. With `L = 0` (the default OME-TIFF) nothing
is padded.

| Option | Default scale factor | Cost (default output, LN / culture / tissue-2D, uint8) | Consequence |
| --- | --- | --- | --- |
| S1. 1 (acquired spacing) | `scale_yx: 1` | 0.125 / 0.122 / 0.414 GiB | Full detail; the size of today's full fused projections (tissue-2D `PI.tif` 444,335,200 bytes). |
| **S2. 0.5 (recommended)** | `scale_yx: 0.5` | 0.031 / 0.030 / 0.104 GiB | Today's `*_05` practice; 0.39 µm pixels for LN and culture, 0.19 µm for tissue-2D, well below a nucleus diameter. |
| S3. 0.25 | `scale_yx: 0.25` | 0.008 / 0.008 / 0.026 GiB | An overview at 0.78 µm (LN, culture) or 0.38 µm (tissue-2D) pixels. |

**Recommendation: S2.** Existing files use it, the default output stays small for every
sample, and the acquired size remains one key away.

| Option | Resampling | Cost | Consequence |
| --- | --- | --- | --- |
| **R1. Block mean, `1/k` only (recommended)** | as above | One pass, exact geometry | No interpolation choice; the output grid nests in the fusion grid. |
| R2. Any factor, area-weighted | each output voxel averages the level-0 area it covers, with fractional weights | Weights per output voxel | Arbitrary scale factors; voxel centres no longer nest, and the result depends on an interpolation rule. |
| R3. Subsampling | every `k`-th voxel | Cheapest | Aliasing; small spots disappear or not depending on their position. |

## The 3D stack

On request (`dimensionality: stack` or `both`). The same grid rule in three dimensions:
the fusion grid with Z, the value rule with candidates decided in Z, Y and X, the scale
factors `k_z`, `k_y`, `k_x`, the same `S`, `O`, `N`. When every included FOV has Z = 1 and
the same `origin_z` (`n0_z = 1`), the stack is one plane and its values equal the
projection's at the same `scale_yx` under `projection: max`. Its Z geometry equals the
projection's only at `scale_z = 1`: by the table of "The scale factor" a Z factor `k_z > 1`
gives the one plane `S_z = k_z·s_z` and `O_z = o_z + s_z·(k_z − 1)/2` (a block padded with
uncovered planes), while the projection has `S_z = s_z` and `O_z = o_z`; both map every z of
the plane's footprint to index 0. Example: `s_z = 1`, `o_z = 0`, `scale_z = 0.5` give the
stack `S_z = 2`, `O_z = 0.5` and the projection `S_z = 1`, `O_z = 0`.

Its cost is the fusion of every level-0 voxel ("Resources"): LN 68 × 12,333 × 10,883 voxels
(8.5 GiB of uint8 before compression), cell-culture-3D 45 × 13,636 × 9,576 (5.5 GiB). A
`scale_yx` of 0.5 divides the written size by four but not the fusion work, which reads
every FOV voxel once.

## Sample position to image position

This is the condition that ties every fused file to the sample tables.

For a file with recorded spacing `S` and origin `O` (from the file or its record; the
table of "The scale factor" gives them per axis, the projection's Z included), a sample
position `q = (q_z, q_y, q_x)` in µm has the continuous index

```
j = (q − O) ⊘ S
```

and lies in voxel `v = floor(j + ½)` per axis (the voxel whose half-open support
`[O + (v − ½)·S, O + (v + ½)·S)` contains `q`). Below, `j⁰` and `v⁰` are the indices on
the fusion grid (`S = s`, `O = o`), and `k` is a file's factor per axis (`k_z = n0_z` for
the projection's Z).

A test can evaluate, for a set of sample positions inside the fusion grid's extent (the
FOV origins, fractional positions inside each FOV, the molecule positions `sample_z/y/x`
of the sample molecule table, and positions next to voxel boundaries and to ownership
boundaries):

1. **Recorded geometry, per file and per axis.** The `S` and `O` read back from each file
   ("Read-back checks") equal those of `fused.json` within 1e-9 µm, and equal the formulas
   of "The scale factor": for the stack on Z, Y and X and for the projection on Y and X,
   `S_a = k_a·s_a` and `O_a = o_a + s_a·(k_a − 1)/2`; for the projection on Z,
   `S_z = n0_z·s_z` and `O_z = o_z + s_z·(n0_z − 1)/2`. For resolution level `ℓ` of a
   file, `S^(ℓ) = 2^ℓ·S` and `O^(ℓ) = O + S·(2^ℓ − 1)/2` in Y and X, and Z as the output
   level.
2. **Projection and stack agree.** At the same `k_yx`, the Y and X components of `j` and
   `v` are equal in the projection and in the stack. In Z the projection's `v_z` is 0 for
   every `q_z` in `[z_lo, z_hi)`, the stack's `v_z` is `floor(v⁰_z / k_z)`.
3. **Any scale factor.** Per axis, `j = (j⁰ − (k − 1)/2) / k` within 1e-9, and
   `v = floor(v⁰ / k)` exactly, also when `k` does not divide `n0` (the last block is
   partial) and for the projection's Z (`k = n0_z`, so `v_z = 0`). For resolution level `ℓ`
   the same holds with `k·2^ℓ` in Y and X. The floor identities are evaluated at positions
   at least 1e-6 voxel from a voxel boundary of the fusion grid, where float64 rounding of
   `j` cannot move `floor`.
4. **The value follows the fusion rule.** On a fixture whose source images are zero except
   one voxel of value `A > 0` at index `p*` of FOV `g`, at sample position
   `q* = origin_g + p* ⊙ s`, and whose origins are not offset from the grid by exactly half
   a voxel on any axis:
   * Because the fusion grid has the FOVs' spacing, exactly one fusion-grid voxel samples
     FOV `g` at `p*` ("The value of a voxel", step 3): `v* = floor((q* − o) ⊘ s + ½)`, the
     voxel that contains `q*`, whose centre `c* = o + v* ⊙ s` is within half a voxel of
     `q*` on each axis.
   * The stack at the fusion grid holds `A` at `v*` if the overlap rule chooses FOV `g` at
     `c*` (for V1: `g` owns `c*`, among the candidates covering `c*` in Z, Y and X), and 0
     at every other voxel; if the rule chooses another FOV at `c*`, the stack is 0
     everywhere. The choice is made at `c*`, not at `q*`: the two can lie on opposite sides
     of an ownership boundary. A sufficient condition for `A` to appear is that every point
     within half a voxel of `q*` on each axis is owned by `g`.
   * The projection holds `A` at the Y and X of `v*` under the same condition with the
     candidates decided in Y and X, and 0 elsewhere.
   * At factor `k`, the voxel `floor(v* / k)` holds `A` divided by the number of covered
     fusion-grid voxels of its block, rounded half to even (the fixture chooses `A` large
     enough that this is not 0), when `A` appears at `v*`; every other voxel is 0.
   * A test evaluates this oracle for the configured fixture: it computes the owner of `c*`
     with the overlap rule and expects `A` or an all-zero image accordingly.
5. **SpatialData.** In the store, the image element's transform to the sample coordinate
   system maps the intrinsic coordinate `w + ½` of voxel `w` (spatialdata's voxel `w`
   covers `[w, w + 1)`, `spatialdata/_core/data_extent.py` line 90 of 0.6.1) to
   `O + w ⊙ S` on the element's axes (Y and X for a projection; Z, Y and X for a stack),
   so a point at sample position `q` lands in intrinsic voxel `floor(j + ½)`, the `v` of
   condition 1. On the fixture of condition 4, the intrinsic voxel of a point at `q*` holds
   `A` exactly when condition 4 says the file holds `A` there.

Worked counterexample to a rule stated at `q*` instead of `c*`: two FOVs of shape
`(1, 4, 4)` at X origins 0 and 1.4, unit spacing, Y and Z origins 0, only FOV 2's voxel
`(0, 1, 1)` bright. `q*_x = 2.4`, owned by FOV 2 (centres at X 1.5 and 2.9). The fusion
grid has `o_x = 0` and six voxels in X; `v*_x = floor(2.4 + ½) = 2`, `c*_x = 2`, owned by
FOV 1 (distance 0.5 against 0.9), which samples its own index 2, value 0. The stack is 0
everywhere, as condition 4 states.

## File layouts

### OME-TIFF layout

* **Container**: BigTIFF always (LN's stack is 8.5 GiB, above the 4 GiB classic limit),
  written with `tifffile` (a base dependency).
* **Series and axes**: one image per file, one channel; axes `YX` stored with SizeZ 1 for
  the projection and `ZYX` for the stack; the OME-XML declares SizeZ, SizeC 1, SizeT 1 and
  the pixel type of the input dtype.
* **Tiling**: 512 × 512 tiles (written band by band, "Bounded memory"); compression `zlib`
  (deflate, lossless; tifffile encodes and decodes it with its own fallback when
  `imagecodecs` is absent, `tifffile.py` lines 15660 to 15667 of 2026.1.28).
* **Resolution levels**: `pyramid_levels` (default 0) further levels as SubIFDs, with the
  padding, values and geometry of "Resolution levels inside a file", written as SubIFD
  series after the main series ("Bounded memory").
* **Calibration and frame**: `PhysicalSizeX`, `PhysicalSizeY`, `PhysicalSizeZ` = `S` of the
  output level with unit `µm` (for the projection `PhysicalSizeZ` is the slab thickness
  `n0_z·s_z`); for plane `i` the OME `Plane` positions `PositionZ = O_z + i·S_z`,
  `PositionY = O_y`, `PositionX = O_x` in µm (the centre of voxel `(i, 0, 0)`; for the
  projection the slab's middle); and the `ImageMetadata` of the output level as JSON in a `CommentAnnotation` of namespace `starfinder.image.ImageMetadata`,
  the convention of `save_volume`, so `load_volume` restores the metadata. A second
  annotation, namespace `starfinder.sample_output`, holds the sample, image name,
  dimensionality, projection rule and factors, and the SHA-256 of the placement tables.

| Option | Layout | Cost | Consequence |
| --- | --- | --- | --- |
| **T1. BigTIFF, 512² tiles, zlib, SubIFD levels on request (recommended)** | as above | Tiles add an index per plane; zlib costs CPU on write (W-324's TIFF write of 413 MB took 19.9 s with fusion) | Region reads touch only their tiles; `load_volume` reads it; small default files need no pyramid. Reading by Bio-Formats and other viewers belongs to the deferred viewer qualification. |
| T2. Strips, no compression, no levels | One strip per row band, as `save_volume` writes today | Fastest write | The 8.5 GiB stack is read whole for any region; the file is the raw size. |
| T3. One multichannel file per sample | All `images` as C of one OME-TIFF | One file | A channel added later rewrites the file; differs from today's one file per channel. |

**Recommendation: T1.** It reads by region, keeps the default output small, and stays
readable by the existing loader.

### OME-Zarr layout

* **Format**: OME-NGFF 0.4 on Zarr v2, written with ome-zarr's `FormatV04` (version
  `0.4`, Zarr format 2). ome-zarr 0.12.2 defaults to `FormatV05` (0.5, Zarr v3) and uses
  `FormatV04` for a Zarr v2 group or when it is passed (`ome_zarr/format.py` lines 232 to
  250 and 361 to 380, `ome_zarr/writer.py` lines 180 to 197). `multiscales` with axes `c` (type `channel`), `z`
  (stack only), `y`, `x` (type `space`, unit `micrometer`, the NGFF name of µm), one
  channel with its image name in `omero` channel metadata.
* **Chunks**: `(1, 1024, 1024)` for a projection (`c, y, x`), `(1, 16, 512, 512)` for a
  stack; Blosc `lz4`, level 5, byte shuffle; C order; fill value 0.
* **Resolution levels**: the output level, then levels halving Y and X until both are at
  most 1024, with the padding, values and geometry of "Resolution levels inside a file",
  computed region by region from the written output level. Each dataset's
  `coordinateTransformations` are `scale` `[1, S^(ℓ)]` then `translation` `[0, O^(ℓ)]`,
  the NGFF voxel-centre convention, with the spacing and origin of that level.
* **Frame**: group attributes `starfinder` with the output level's `ImageMetadata`, the sample,
  `frame_id` and the SHA-256 of the placement tables.
* **Writer**: standalone files through `ome-zarr` and `zarr` of the `spatialdata` extra.
  Inside the store, spatialdata's own raster writer writes the image with the same
  chunks, codec, padding and levels, the levels passed as an explicit multiscale so it
  computes no second pyramid. Its raster format 0.2 (`RasterFormatV02`, the default
  raster format of container format 0.1) writes the NGFF 0.4 layout on Zarr v2 with the
  `multiscales` version string `0.4-dev-spatialdata` (`spatialdata/_io/format.py` lines
  170 to 179 and 345 to 350 of 0.6.1). The per-level scales there are spatialdata's, from
  the shape ratios, exact because of the padding, and the placement in the sample frame is
  the element transform of "SpatialData export". The
  store's metadata is not edited after writing (the half-voxel problem of W-168,
  {doc}`sample-output-baseline`); how a reader without spatialdata interprets the store's
  image is part of the deferred viewer qualification.

| Option | Layout | Cost | Consequence |
| --- | --- | --- | --- |
| **Z1. NGFF 0.4 on Zarr v2 (recommended)** | as above | spatialdata 0.6.1 writes NGFF 0.5 on Zarr v3 by default, so the export passes `SpatialDataContainerFormatV01`, whose element defaults are `RasterFormatV02`, `ShapesFormatV02`, `PointsFormatV01` and `TablesFormatV01` (`spatialdata/_io/format.py` lines 281 to 285, 329 to 350 and 421 to 425 of 0.6.1); ome-zarr needs `FormatV04` for the standalone files | On Python 3.11 and later the locked zarr 3.1.5 writes Zarr v2 when asked (`zarr_format` of `create_group`, `zarr/api/synchronous.py` line 572; spatialdata passes it, `spatialdata/_core/spatialdata.py` lines 1171 to 1172). On Python 3.10 the extra resolves zarr 2.18.3 and spatialdata 0.3.0; that zarr 2 writes Zarr v2 by default and Zarr v3 only behind an experimental switch (read in zarr 2.18.7, `zarr/_storage/store.py` lines 24 to 26; 2.18.3 and spatialdata 0.3.0 were not read and are an assumption for the implementation issue). It is the version the W-168 store used and W-324's fusion probe wrote. |
| Z2. NGFF 0.5 on Zarr v3 | spatialdata 0.6.1's default | No format argument on Python 3.11 and later | Not writable on Python 3.10 with the locked zarr 2.18.3 (same assumption); sharding is available for very large stores; fewer readers. |
| Z3. NGFF 0.4, no resolution levels | the output level only | About 1/3 less written | Region reads at full detail only. |

**Recommendation: Z1.** It works on every supported Python with the locked versions and
matches the format of the earlier exporter.

## SpatialData export

On request (`spatialdata: true`). It is an export of the sample tables and the fused
image, **not a second source**: every value comes from `assembly/<sample>/` and
`fused/<sample>/`, nothing is filtered, reassigned or recounted, and the sample tables
stay the reference.

**Inputs**: the `AssemblyResult` (from `load_assembly`, which checks every SHA-256), the
`FusionRecord` and, with `labels`, the label record.

**Coordinate system**: one, named after the sample; its axes are the sample frame's, in
µm. spatialdata does not check coordinate-system names, but the export applies its element
naming rule to it (letters, digits, `_`, `-` and `.`; `spatialdata/_core/validation.py`
line 109 of 0.6.1) and refuses other sample names. `frame_id` `sample:<sample>` is recorded
in `export.json` and in the store's attributes.

**Element names.** spatialdata requires every element name to satisfy that rule and to be
unique across all element types, tables included, ignoring case: images, labels, points,
shapes and tables share one set of names (`spatialdata/_core/spatialdata.py` lines 134 to
139, `spatialdata/_core/_elements.py` lines 41 to 48 and 122 to 123,
`check_all_keys_case_insensitively_unique` and `check_key_is_case_insensitively_unique`, `spatialdata/_core/validation.py` lines 113 and 156 of 0.6.1). The store's names are
`molecules`, `cell_centroids`, `cells`, `expression`, `cells_unplaced`,
`molecules_unplaced`, `<image>_projection`, `<image>_stack` and `cell_labels`; no two are
equal ignoring case, and an image name cannot produce a fixed name because only image
elements end in `_projection` or `_stack`. `FusionConfig` refuses image names equal
ignoring case. The obs and var column names and the uns, obsm and layers keys of every
table must satisfy the same rule and be unique ignoring case
(`spatialdata/_core/validation.py` lines 218 to 252); the column names of the sample tables
and the keys of `sample.h5ad` do ({doc}`assembly-contract`), and the export refuses a table
that does not, naming the key.

**Elements** (`export/<sample>/sample.zarr`), each written only when its population has a
row or its image exists ("Empty populations"):

| Element | Content | Transform to the sample coordinate system |
| --- | --- | --- |
| points `molecules` | Every row of the sample molecule table whose `sample_z/y/x` are finite (every molecule of a placed FOV, retained or excluded), with every other column; the §2.9 index columns `z`, `y`, `x` are first renamed `fov_index_z/y/x`; `PointsModel.parse(…, coordinates={"x": "sample_x", "y": "sample_y", "z": "sample_z"})` then turns `sample_x/y/z` into the coordinate columns `x`, `y`, `z` (float64, µm) and does not keep them under their old names (`spatialdata/models/models.py` lines 831 to 847 of 0.6.1); `cell_instance_id` (pandas `UInt32`, null without a cell) is added. No feature or instance key is declared, so the model adds only its `transform` attribute (lines 865 to 895). | Identity |
| shapes `cell_centroids` | One circle per row of the sample cell table with a finite ownership centroid: centre `(sample_ownership_centroid_x, sample_ownership_centroid_y)`, column `radius` the equivalent radius of the original territory (`size_voxels` × voxel volume as a sphere for `volume`, × pixel area as a disc for `plane`), index `instance_id`. Radii must be positive (`ShapesModel`, lines 458 to 463); a cell with `size_voxels` 0 makes the export raise. | Identity |
| table `cells` | The sample cell table for the same rows, every column as `obs`, no `X` (zero variables); added `obs` columns `region` (categorical, every value `cell_centroids`) and `instance_id` (numpy `uint32`, an instance-key dtype the model accepts, lines 1033 to 1047); parsed with `TableModel.parse(region="cell_centroids", region_key="region", instance_key="instance_id")` | annotates `cell_centroids` |
| table `expression` | `sample.h5ad` with every part kept: `X`, `layers`, `var`, `obs`, `obsm` and every key of `uns`; added `obs` columns `region` and `instance_id` as for `cells`; parsed as `cells` | annotates `cell_centroids` |
| tables `cells_unplaced`, `molecules_unplaced` | The rows of the cell and molecule tables without finite sample coordinates (status `fov_not_placed`), every column as `obs`, zero variables, plus `instance_id` (cells) or `cell_instance_id` (molecules) as plain columns; parsed without region, region key or instance key, so the model adds nothing (lines 1131 to 1135) | none |
| image `<image>_projection` (and `<image>_stack`) | The fused image, one element per image name and dimensionality, written once in the store as OME-Zarr ("OME-Zarr layout") with the fused levels | `Sequence([Scale(S), Translation(O − S/2)])` on its axes (Y and X for the projection; Z, Y and X for the stack): spatialdata's voxel `w` covers `[w, w + 1)` (`spatialdata/_core/data_extent.py` line 90), so its centre `w + ½` goes to `O + w ⊙ S`. Further levels get spatialdata's shape-ratio scale before it, exactly `2^ℓ` because of the padding ("Resolution levels inside a file"). |
| labels `cell_labels` (optional) | The assembled label export ("Assembled label export") | as for the image, at the labels' grid |

**Metadata spatialdata adds.** `TableModel.parse` of an annotating table writes
`uns["spatialdata_attrs"] = {"region": "cell_centroids", "region_key": "region",
"instance_key": "instance_id"}` (`spatialdata/models/models.py` lines 1146 to 1164 of
0.6.1) and refuses a table that already has the key (lines 1138 to 1142); the region column must be
categorical (lines 1083 to 1086). A table parsed without these arguments and without the
key is returned unchanged (lines 1131 to 1135). The points and raster models add their
`transform` attributes. This metadata is the export's, not the source's; the round-trip
check tests it separately from the source content ("Read-back checks").

**Empty populations.** The pinned library rejects an empty shapes element (`ShapesModel`,
`spatialdata/models/models.py` lines 449 to 450 of 0.6.1) and an annotating table whose
declared region does not occur among its region values, which is every annotating table
without rows (lines 1048 to 1051). The rule is therefore: **an element whose population has
no row, or whose image does not exist, is left out of the store**, and `export.json` lists
every element the configuration asks for, each with `status` `written` or `omitted`, its
row count or shape, and for an omitted element a `reason`: `empty` (no row), `no_cells`
(a `molecules_only` sample has no territories), `no_included_fov` or `not_requested`
(`fuse: false`, `labels: false`). A name that the store lacks and `export.json` does not
list as omitted is missing, and the read-back check fails on it. The export relies on no
library behavior for empty elements, because it writes none. When every element is
omitted, no `sample.zarr` is written and `export.json` records `store: null` with the
reason; the export does not rely on spatialdata writing an empty store.

| Case | Written | Left out (reason) |
| --- | --- | --- |
| Usual sample | every element with rows; the images | `cells_unplaced`, `molecules_unplaced` when every FOV is placed (`empty`, the common case) |
| No cell with a finite ownership centroid (no cell, or only cells of FOVs without an origin) | `molecules` if any row; `cells_unplaced` if any row; images | `cell_centroids`, `cells`, `expression` (`empty`) |
| Cells, but no counted cell | `cell_centroids`, `cells`, the rest as usual | `expression` (`empty`); `sample.h5ad` has no row ({doc}`assembly-contract`, "Sample-level AnnData") |
| `molecules_only` sample | `molecules` if any row, `molecules_unplaced` if any row, images | `cell_centroids`, `cells`, `expression`, `cells_unplaced` (`empty`); `cell_labels` (`no_cells`) |
| No molecule with finite sample coordinates | the rest as usual | `molecules` (`empty`) |
| No included FOV | each table, points or shapes element that has rows (usually only `cells_unplaced` and `molecules_unplaced`) | image elements and `cell_labels` (`no_included_fov`); every table, points or shapes element without rows (`empty`) |
| `fuse: false` or `labels: false` | the tables, points and shapes as above | image elements or `cell_labels` (`not_requested`) |

In the "No included FOV" row, a placed FOV without a source image (a `placeholder`) still
has sample coordinates, so its observations, if it has any, go to `molecules`, `cells` and
`cell_centroids` as usual; only rows without sample coordinates go to the `_unplaced`
tables.

**Image dimensionality in the store.**

| Option | Store image | Cost | Consequence |
| --- | --- | --- | --- |
| **D1. What the fuse profile writes (recommended)** | The projection by default; the stack when `dimensionality` is `stack`; both with `both` | Nothing beyond the fused images already requested | The default store is small (LN 0.031 GiB of image); points keep their z, the projection's transform maps Y and X; a user who wants 3D in the store asks for the stack once. |
| D2. Always the 3D stack | The stack, whatever the fuse profile | The stack's fusion and size for every export (LN 8.5 GiB at scale 1, 2.1 GiB at 0.5) | 3D points and image agree in Z; every export pays for 3D. |
| D3. Always the projection | The projection only | Smallest | A requested stack never reaches the store. |

**Recommendation: D1.** One setting decides the image everywhere, and the default export
stays small.

**Written once, not duplicated.** With `spatialdata: true` the OME-Zarr image exists only
as the store's image element: `fuse` writes no `fused/<sample>/*.ome.zarr`, and the store
holds one image element per image name and dimensionality. Its values are those of the
fused computation (the same as the OME-TIFF's at the same level); the implementation may
recompute them or read the written OME-TIFF lazily, and the read-back check requires
equality on the extent the two share (padding voxels are 0). The record keeps the image
element's path and the SHA-256 of its first-level array metadata.

**Identities and source lookup.** Molecules keep `(spot_namespace, spot_id)` and `fov_id`;
cells keep `(cell_namespace, cell_id)` and `obs_name` ({doc}`assembly-contract`,
"Sample-level identities"). `instance_id` is `1..N` over the rows of the sample cell table
in its order (uint32; more than 2³² − 1 rows is an error); `cell_instance_id` of a molecule
is the `instance_id` of its `(cell_namespace, cell_id)`. `lookup_source(folder, spot=…)` or
`(cell=…)` reads `export.json`, verifies its hashes, finds the row and returns its FOV, the
FOV's assignment folder, its `assignment.json` SHA-256 and the row's position in the §2.9
table, from the assembly record.

**What it takes from the reference-branch exporter** ({doc}`sample-output-baseline`): the
store-once rule; writing into a sibling staging folder and renaming it into place, so a
failed export publishes nothing; the SHA-256 manifest of every component and its
verification on read; key validation (non-null, unique, string namespaces); the explicit
missingness of W-168 (a molecule without a cell keeps its §2.9 `assignment_status`, never
a guessed cell). It does not take `prepare_rasters`, the affine source mappings, the
`MoleculeIndex` input, the rewrite of NGFF transforms inside the store, the pinned Zarr 2
environment or the napari adapter.

**`export.json`**: `format_version` 1, `stage` `export`, sample, `frame_id`, coordinate
system name, the SHA-256 of `assembly.json`, `fused.json` and (labels) `labels.json`, the
spatialdata version and the formats written, `store` (the store path, or `null` when every
element is omitted), per element its name, kind, `status` (`written` or `omitted`) and for an
omitted element its `reason` ("Empty populations"), and for a written element its path, row
count or shape, transform and, for tables, the `spatialdata_attrs` written and the dtype of every
column after the round trip, and the counts of "Read-back checks" (molecules by `assignment_status`
and by `assembly_status` and reason, cells by `assembly_status`, counted cells), the column
renames, and the SHA-256 of every file of the store.

## Assembled label export

An optional part (`labels: true`), from the 2026-09-27 scope: assembled labels are an
inspection product.

* **Inputs**: the placement, the assembly result and, per FOV, the territory label image
  that assign sampled (the expanded one when assign expanded, as the cell table's
  `ownership_centroid` records), whose SHA-256 the assembly record holds. A `molecules_only`
  sample has no territories: the label export writes no label image, `labels.json` records
  `reason: "no_cells"`, and the store records `cell_labels` as omitted. With no included FOV
  it records `reason: "no_included_fov"` likewise.
* **Unique sample-level labels**: the value of a voxel is the `instance_id` of the cell
  ("SpatialData export", "Identities") whose territory the chosen FOV has there, whatever
  that cell's assembly status; 0 is background, and an uncovered voxel is 0. uint32.
* **Categorical placement without blending**: "The value of a voxel" with V1, nearest
  sampling and no interpolation, on the territory images instead of the source images; at
  a scale factor below 1, the most frequent value of each block, background and uncovered
  voxels counting as 0, the smallest ID on a tie (the reference-branch rule), with every ID
  lost at that level listed in the record (no fallback to full size). Resolution level `ℓ`
  is the mode of each `2^ℓ × 2^ℓ` block of the output level, with the padding of
  "Resolution levels inside a file".
* **Grid**: the stack's grid for `volume` and `extended` territories; for `plane`
  territories the projection's Y and X grid with one plane, the candidates decided in Y and
  X. There is no label projection of volumes, because a maximum of IDs means nothing.
* **Flagged conflicts**: a fusion-grid voxel is a conflict when the chosen FOV's territory
  holds a cell `a` with `assembly_status == "retained"` and another candidate FOV's
  territory holds a retained cell `b` (a cell of another FOV, so `b ≠ a`; no cell is
  matched across FOVs). `labels.json` lists each pair `(a, b)` with its voxel count, and
  separately, per retained cell `b`, the voxels where the chosen FOV holds background but
  `b`'s FOV holds `b`; both beside the cell table's `conflict` flags
  ({doc}`assembly-contract`, "Flags"). No conflict is resolved and no cell is merged.
* **Place**: the labels element `cell_labels` of the store. Its values are the
  `instance_id` of the tables `cells` and `expression`, which annotate `cell_centroids`;
  the labels element is linked to them through that ID, not through a second table
  annotation.
* **Checks a test can evaluate**: every nonzero value is an `instance_id` of the cell
  table; at the fusion grid, every voxel equals the territory value of the FOV that "The
  value of a voxel" chooses there, mapped to its `instance_id` (an independent oracle as for
  the image); the output level equals the block mode of the fusion grid over `k` blocks, and
  resolution level `ℓ` the block mode of the output level over `2^ℓ × 2^ℓ` blocks, and every
  ID missing from a level is listed; the conflict counts recomputed from the territories equal
  `labels.json`.

| Option | Place | Cost | Consequence |
| --- | --- | --- | --- |
| **LP1. Labels element of the store (recommended)** | `sample.zarr/labels/cell_labels`, NGFF 0.4 layout as the image | uint32: 4 bytes per voxel (LN stack 34.0 GiB at scale 1, 8.5 GiB at 0.5) | Image, labels, points and tables in one frame and one store; a label is linked to its table row by `instance_id`. |
| LP2. A separate OME-TIFF | `fused/<sample>/cell_labels.ome.tif` | The same bytes, no zarr needed | Readable without the extra; the link to the tables is only the ID. |
| LP3. Both | both | Twice the bytes | Duplicates the largest output. |

**Recommendation: LP1**, with LP2 when `spatialdata` is false.

| Option | Delivery | Cost | Consequence |
| --- | --- | --- | --- |
| LD1. In the first delivery | with the fused image and the export | The conflict accounting, the territories of every FOV and the uint32 volumes above in the first implementation issue | Labels available at once; the largest and least-qualified output gates the rest. |
| **LD2. In a later delivery (recommended)** | after the fused image and the SpatialData export pass their read-back checks | The rule and the key are specified here; nothing is built yet | The first delivery is the default output and the export; the label export reuses their grid, writer and transforms unchanged. |

**Recommendation: LD2**, for Jiahao to decide at the gate (W-329). The decision is open;
the key `labels` stays false until the part exists.

## Read-back checks

`check_fused(folder)` and `check_spatialdata_export(folder)` read the written files back
through the libraries and evaluate these conditions; no viewer session is involved.

**OME-TIFF** (each file of `fused.json`), with `tifffile` and `load_volume`:

1. The file opens as BigTIFF and OME (`is_bigtiff`, `is_ome`), with one series whose axes
   are `YX` (projection, or a one-plane stack) or `ZYX` (stack) and one channel.
2. Shape (the padded size when the file has levels) and dtype equal the record's; the tile
   shape is 512 × 512 and the compression `zlib`; the number and shapes of SubIFD levels
   equal the record's, each exactly half the one before in Y and X.
3. **Geometry, per axis.** `PhysicalSizeX`, `PhysicalSizeY` and `PhysicalSizeZ` equal the
   record's `S` within 1e-9 µm with unit `µm`, where `S` follows the table of "The scale
   factor" for this file (for a projection, `PhysicalSizeZ` is the slab thickness
   `n0_z·s_z`); the `Plane` positions of plane `i` equal `(O_z + i·S_z, O_y, O_x)`; the
   `ImageMetadata` annotation, as restored by `load_volume`, has that spacing and origin
   and equals the record's metadata; conditions 1 to 3 of "Sample position to image
   position" hold with these values.
4. **Voxel values.** A region read from the file equals an independent oracle computed
   from the source images, the placement and the rules of "The value of a voxel" and "The
   scale factor": exactly at the output level (block means over covered voxels, rounded
   half to even) and at every SubIFD level (block means of the output level, uncovered and
   padding voxels counting as 0). When the sample has two or more included FOVs the region
   spans an ownership boundary; it also includes uncovered voxels when the fusion grid has
   any, and, where the factor does not divide `n0`, the partial last block. On the fixture
   of condition 4 of "Sample position to image position", the whole file equals that
   condition's oracle. The uncovered count equals the record's.
5. The SHA-256 of the file equals the record's.

**OME-Zarr** (standalone files), with `zarr`:

1. The group is Zarr format 2 with `multiscales` version `0.4`; axes names, types and units
   as specified; one dataset per recorded level; each dataset's `scale` and `translation`
   equal the record's `S^(ℓ)` and `O^(ℓ)` per axis within 1e-9 (the projection's Z as in
   OME-TIFF check 3).
2. Each level's shape, dtype, chunks and compressor equal the record's; each level is
   exactly half the one before in Y and X.
3. A region of the output level equals the oracle exactly; every further level equals the
   block mean of the output level as in OME-TIFF check 4.
4. When the output level has more than one chunk, a region read inside one chunk touches
   that chunk only (chunk keys counted).

**SpatialData** round trip, with `spatialdata.read_zarr` of the written store. Each check
below applies to the elements `export.json` records as `written`; an element recorded as
`omitted` is tested by check 1 only.

1. **Elements**: the element names and kinds in the store equal those `export.json`
   records as `written`, and no two names are equal ignoring case; every element the
   configuration asks for is recorded as `written` or `omitted`; for each `omitted`
   element the reason holds when recomputed from the sample tables, the fusion record and
   the configuration (`empty`: its population has no row; `no_cells`: the sample is
   `molecules_only`; `no_included_fov`: `fused.json` has that reason; `not_requested`: the
   key is off), and the element is absent from the store; when `export.json` records
   `store: null`, no `sample.zarr` exists and every element is `omitted`. One image element
   per recorded fused image; no other OME-Zarr group
   of the same image exists under `export/<sample>/` or `fused/<sample>/`; the store is Zarr
   format 2 and each raster group's `multiscales` version string is the one its recorded
   raster format writes (`0.4-dev-spatialdata` for `RasterFormatV02`).
2. **Points**: the `molecules` rows, joined on `(spot_namespace, spot_id)`, equal the
   sample molecule table's rows with finite sample coordinates, column by column: `x`,
   `y`, `z` equal `sample_x`, `sample_y`, `sample_z` exactly, `fov_index_z/y/x` equal the
   §2.9 `z`, `y`, `x`, every other source column equals its source (NaN equals NaN, null
   equals null), `cell_instance_id` follows check 5, and no `sample_x/y/z` column remains; the dtypes equal those `export.json`
   lists. Together with `molecules_unplaced` (each where written), every row of the sample
   molecule table appears exactly once; a sample molecule table without rows has both
   omitted.
3. **Source content of the tables is preserved.** The `cells` and `cells_unplaced` rows,
   joined on `(cell_namespace, cell_id)`, equal the sample cell table's rows in every
   source column, and together contain each of them exactly once; `molecules_unplaced` likewise for its molecule rows; `expression` equals
   `sample.h5ad` in `X`, `layers`, `var` and `obsm` exactly, in every source column of
   `obs`, and in every key of the source `uns`, each with the same value. Values are
   compared, and each column's dtype after the round trip equals the one `export.json`
   records (how anndata 0.12.6 under pandas 3 returns string columns is recorded rather
   than assumed). When `expression` is omitted, `sample.h5ad` has no row.
4. **Export metadata is present, separately.** `cells` and `expression` have the added
   `obs` columns `region` (categorical, every value `cell_centroids`) and `instance_id`
   (`uint32`), and `uns["spatialdata_attrs"]` equal to `{"region": "cell_centroids",
   "region_key": "region", "instance_key": "instance_id"}`; they have no other key or
   column that their source lacks. `cells_unplaced` and `molecules_unplaced` have no
   `spatialdata_attrs` key and no added column besides `instance_id` and
   `cell_instance_id` respectively. Points, shapes and raster elements carry
   the transforms of check 6 and no feature or instance key.
5. **Links and accounting**: `instance_id` of `cells` and `cells_unplaced` together is
   `1..N` in the sample cell table's order (empty when both are omitted); `cells` and
   `cell_centroids` are written or omitted together, and the index of `cell_centroids`
   equals the `instance_id` of `cells`; every molecule of `molecules` and `molecules_unplaced` with a
   non-null cell key has a
   `cell_instance_id` equal to that cell's `instance_id`, and every molecule without one a
   null `cell_instance_id`; every `expression` row is a `cells` row with
   `assembly_status == "retained"` and `status == "kept"`. Counts of molecules by
   `assignment_status` (`assigned`, `unassigned`, `excluded_cell`, `outside_grid`) and by
   `assembly_status` and reason, recounted from `molecules` and `molecules_unplaced`, equal
   `export.json` and the assembly record's counts.
6. **Calibration and alignment**: the transform of each written image (and labels) element to the
   sample coordinate system equals `Scale(S)` then `Translation(O − S/2)` on its axes within
   1e-9, with `S` and `O` of that file's output level (Y and X for a projection); its extent
   there (`get_extent`) is `[O − S/2, O + (N_padded − ½)·S)` per axis; level `ℓ`'s composed
   transform equals `Scale(2^ℓ·S)` then `Translation(O − S/2)` in Y and X; points and shapes
   have the identity transform. For the positions of "Sample position to image position",
   a point at `q`, transformed into the image's intrinsic coordinates, falls in voxel
   `floor(j + ½)`; on the fixture of its condition 4, that voxel holds `A` exactly when the
   condition's oracle says so.
7. **Values**: each written image element's output level equals the corresponding OME-TIFF output
   level exactly on the extent they share, with 0 in the padding (or the oracle when no
   OME-TIFF is written); its further levels follow OME-Zarr check 3.
8. The SHA-256 of every store file equals `export.json` (no file when `store: null`).

**Deferred**: viewer qualification in Fiji and napari (opening the OME-TIFF and the store,
display of levels, channels, labels and points). These checks do not stand in for it; the
records say `viewer_qualification: "deferred"`.

## Fusion implementation

| Option | Implementation | Cost | Effect on the BigStitcher path |
| --- | --- | --- | --- |
| **FI1. A writer on shared raster functions (recommended)** | `fuse` in `starfinder.sample_output`: the fusion grid, the ownership rule (`ownership_map` of assembly), nearest sampling, block means and the band writer, on numpy and `tifffile` (base) and `zarr` (extra) | New code: a band loop and two writers. Not measured; its work per voxel is a candidate test and a gather, against interpolation and weighting in FI2 | The imported BigStitcher placement is a `PlacementResult` like any other and is fused by the same code; no Fiji run and no extra are needed for the default output. |
| FI2. multiview-stitcher's `fuse` | Convert the placement into multiview-stitcher's per-view affine transforms and call its fusion with `fusion_func` and interpolation order set | Needs the `multiview-stitcher` extra (proposed by {doc}`stitching-contract`, Python 3.11 to 3.13 only) for the default output of every sample. W-324 probe (`fusion/fusion_probe.json`): LN 2 × 2 tiles, 51 × 2854 × 2840 uint8, 95.6 s to OME-Zarr, 19.9 s to TIFF (70 s CPU), 1.2 to 1.8 GiB peak RSS | BigStitcher placements must be converted into its transforms, and BigStitcher users need the extra too. Ownership would be a custom `fusion_func`; labels need order 0 and a non-blending function (W-324 notes section 3). |
| FI3. BigStitcher's fusion in Fiji | An engine task that fuses from BigStitcher's HDF5 | A Fiji run per sample and image | Only for BigStitcher placements; the current macro stores no full-resolution level and cannot write its HDF5 on the CIFS mount (W-323 `fiji-baseline.md`); multiview-stitcher placements would need a second path. |

**Recommendation: FI1**, decided from the W-324 measurements: the probe shows that
multiview-stitcher's fusion works on the CPU at a cost dominated by interpolation and
writing, but it ties the default output of every sample, including BigStitcher samples, to
an extra that Python 3.10 cannot install, and its blending is not the recommended overlap
rule. FI1 fuses both backends' placements with one function on the base dependencies.

## Fusion config

```python
@dataclass(frozen=True)
class FusionConfig:
    images: tuple[str, ...] = ("reference_stain",)  # source-image names; further channels
    dimensionality: str = "projection"               # "projection" | "stack" | "both"
    projection: str = "max"                          # "max" | "sum"
    scale_yx: float = 0.5                            # 1/k, k whole ≥ 1
    scale_z: float = 1.0                             # stack only; 1/k
    formats: tuple[str, ...] = ("ome_tiff",)         # "ome_tiff", "ome_zarr"
    pyramid_levels: int = 0                          # OME-TIFF SubIFD levels; OME-Zarr levels are automatic
    band_tiles: int = 1                              # rows of 512-pixel output tiles per band
```

`__post_init__` refuses unknown values, a scale whose reciprocal is not whole, a
`scale_z` other than 1 when `dimensionality` is `projection` (it applies to the stack
only), a negative `pyramid_levels`, an empty `images` or `formats`, and image names that
are equal when case is ignored (spatialdata treats element names case-insensitively,
"SpatialData export"). Once the grid is known, `fuse` refuses a `pyramid_levels` larger
than `floor(log2(max(N_y, N_x)))`, whose last level would be a single padded voxel row.

## Workflow configuration

A Python-only `sample_outputs` block, read by `_run_sample_outputs` in the workflow
adapter and run by one per-sample rule `sample_outputs` after `assemble_sample`
({doc}`assembly-contract`, "Workflow configuration"):

```yaml
sample_outputs:
  fuse: true                  # false: no fused image (the assembly files and sample.h5ad stay)
  images: [reference_stain]   # source images; further channels are further names
  dimensionality: projection  # projection | stack | both
  projection: max             # max | sum
  scale_yx: 0.5               # 1/k
  scale_z: 1.0                # stack only
  formats: [ome_tiff]         # ome_tiff and/or ome_zarr
  pyramid_levels: 0
  spatialdata: false          # true: export/<sample>/sample.zarr
  labels: false               # the assembled label export (proposed for a later delivery)
```

* The rule runs by default for every sample of a configuration with `stitching` and
  `assembly` blocks: without a `sample_outputs` block the defaults above apply. Without an
  `assembly` block the legacy rules run unchanged and nothing here runs.
* It reads `stitching/{sample}/placement.json`, `assembly/{sample}/assembly.json` and the
  source images of the sample's FOVs, and writes the files of "Output profiles". Its
  declared outputs follow the keys.
* A default-tier test keeps the static schema equal to the fields of `FusionConfig` and
  `SampleOutputConfig`, as for the other blocks.

## Resources

From `tables/resource_estimate.csv` of the W-328 run directory, derived by
`scripts/resource_estimate.py` from the W-324 per-FOV placements (main multiview-stitcher
run of each sample), the tile shapes of W-323 `inputs.md`, the load times of W-324
`tables/runs.csv` and the fusion probe of W-324 `fusion/fusion_probe.json`. No image was
read. Sizes are uint8, one channel, before compression. The fusion grid of LN from the
BigStitcher placement (`parity/prototype/tile_config_ln/T2_sum.csv`) has the same extent
within a few voxels (X span 9,391 px against 9,387; Z offsets spanning 16.6 slices against
18).

| Sample | Fusion grid (Z × Y × X) | Default output: projection at 0.5 | 3D stack at scale 1 | Source read once | W-324 load time of the sources |
| --- | --- | --- | --- | --- | --- |
| LN sample1 (64 FOVs, 50 × 1496², DAPI) | 68 × 12,333 × 10,883 | 1 × 6,167 × 5,442: 33.6 million voxels, 0.031 GiB | 9.13 billion voxels, 8.50 GiB | 7.16 GB (6.67 GiB) | 57.7 to 81.2 s |
| cell-culture-3D (70 FOVs, 30 × 1496²) | 45 × 13,636 × 9,576 | 1 × 6,818 × 4,788: 32.6 million, 0.030 GiB | 5.88 billion, 5.47 GiB | 4.70 GB | 11.8 to 20.3 s |
| tissue-2D (56 FOVs, 3072², one plane) | 1 × 22,582 × 19,691 | 1 × 11,291 × 9,846: 111 million, 0.104 GiB | the one plane: 445 million, 0.414 GiB | 0.53 GB | 6.5 to 9.3 s |

The estimates match today's files: tissue-2D `PI.tif` is 444,335,200 bytes against
444,662,162 estimated voxels, and cell-culture-3D `MAX_DAPI.tif` 130,794,660 bytes against
130,578,336 ({doc}`sample-output-baseline`).

**Time.** The default output reads every FOV once (the load times above, from the CIFS
share) and fuses one plane. Extrapolated linearly in voxels from the one W-324 probe
(multiview-stitcher, weighted average, linear interpolation, four threads; not FI1, and not
measured for these samples): fusing and writing the level-0 plane takes about 6 s (LN,
culture) to 21 s (tissue-2D) at the probe's TIFF rate (20.8 million voxels per second) and
31 to 103 s at its OME-Zarr rate (4.3 million per second). The 3D stack at scale 1 takes
about 440 s (LN), 283 s (culture) at the TIFF rate, and 2,111 s and 1,359 s at the OME-Zarr
rate. A `scale_yx` below 1 does not reduce the fusion work, which visits every level-0
voxel; it divides the written bytes by `k²`. FI1 does less work per voxel than the probe
(no interpolation, no weights), so these are upper estimates of the computation, not of
the CIFS reads.

**Memory** (the model of "Bounded memory", without interpreter overhead): the projection
holds every FOV's projection plus one FOV volume plus one band: about 254 MiB for LN, 223
MiB for culture and 532 MiB for tissue-2D (its 56 projections of 3072² are 504 MiB). The
stack holds one plane per FOV plus one band: about 142 MiB (LN) and 154 MiB (culture), plus
at most a third of one output plane for resolution levels. Both stay far below the 7.0 GiB
that multiview-stitcher's registration needed to hold LN in memory (W-324 section 2.2).

**Further outputs**: OME-TIFF SubIFD or OME-Zarr levels add at most a third of the output
level;
each further channel repeats its image's cost; the label export is uint32, four times the
image bytes (LN stack 34.0 GiB at scale 1, 8.5 GiB at 0.5; tissue-2D plane 1.66 GiB and
0.41 GiB); the SpatialData export adds the sample tables (size not estimated here: no real
assembly exists yet) and, with D1, the default image of 0.03 to 0.10 GiB.

### Dependencies per output

| Output or path | Packages | Extra | Locked versions (Python ≥ 3.11 / 3.10) |
| --- | --- | --- | --- |
| Sample tables as CSV (`table_format: csv`, the default) | numpy, pandas | none (base) | as locked |
| Sample tables as Parquet (`table_format: parquet`) | pyarrow, through the checkpoint table writer ({doc}`checkpoints`, "Table formats"; `_require_parquet` in `src/python/starfinder/io/_checkpoint.py` raises without it) | `checkpoint` (`pyarrow>=15`) | pyarrow 23.0.0 / 23.0.0 |
| `sample.h5ad` | anndata | `anndata` | 0.12.6 / 0.11.4 |
| Fused OME-TIFF, projection or stack, with or without SubIFD levels (levels spooled through a numpy memory map on local scratch) | tifffile, numpy; zlib through tifffile's own codec when `imagecodecs` is absent | none (base) | tifffile 2026.1.28 / 2025.5.10 |
| Reading the fused OME-TIFF back (`load_volume`, `check_fused`) | tifffile | none (base) | as above |
| Fused OME-Zarr, standalone (`formats: [ome_zarr]`) and its read-back | zarr, numcodecs (Blosc), ome-zarr, dask | `spatialdata` (which brings them; no extra of their own) | zarr 3.1.5 / 2.18.3; numcodecs 0.15.1 / 0.13.1; ome-zarr 0.12.2 / 0.10.2 or 0.11.1; dask 2024.11.2 |
| SpatialData export and its round trip | spatialdata with anndata, zarr, ome-zarr, dask, xarray, multiscale-spatial-image, geopandas and shapely (shapes), pyarrow (points) | `spatialdata` | spatialdata 0.6.1 / 0.3.0; geopandas 1.1.2; shapely 2.1.2; pyarrow 23.0.0 |
| Assembled labels as OME-TIFF (LP2) | tifffile, numpy | none (base) | as the fused OME-TIFF |
| Assembled labels in the store (LP1) | as the SpatialData export | `spatialdata` | as above |
| Fusion through multiview-stitcher (FI2, not recommended) | multiview-stitcher | `multiview-stitcher`, proposed by {doc}`stitching-contract`, not in the lock | — |

The versions are those of `src/python/uv.lock` (sha256 `bc994b55…`); the package lists of
the `spatialdata` row are the locked dependencies of spatialdata 0.6.1 and 0.3.0.

The default output needs no extra beyond `anndata` for `sample.h5ad` (and `checkpoint`
when the sample tables are Parquet). A missing extra
raises `ModuleNotFoundError` naming it, as `sample_anndata` does; importing
`starfinder.sample_output` imports neither zarr nor spatialdata. `spatialdata-io` in the
`spatialdata` extra is not used by the export; the extra is not changed. No dependency or
lock change.

## Tests the implementation keeps

`test_workflow_stitching_golden.py`, `test_workflow_assembly_golden.py` and every existing
test pass unchanged; nothing here changes a legacy script or an existing output. The new
tests follow W-331's validation design from the conditions of "Sample position to image
position" and "Read-back checks".

## `docs/migration.md` entries

The implementation adds the `starfinder.sample_output` module and its names, the
`sample_outputs` YAML block and the `sample_outputs` rule, and the folders
`<OUTPUT>/fused/<sample>/` and `<OUTPUT>/export/<sample>/` for configurations with
`stitching` and `assembly` blocks. Legacy configurations and `images/fused/{sample}/` are
unchanged.

## Exclusions

No implementation, rename, dependency or lock change; no code brought over from the
reference branch; no viewer session (deferred); no change to the placement, assembly or
§2.9 contracts; no reassignment of molecules or recomputation of counts from a fused image
or a label raster; no validation design or draft issues (W-331); no real data (every
measured number comes from the W-323 and W-324 records by path, and the estimates are
derived from them without reading an image).
