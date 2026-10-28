# Sample-output guide: the export entry, the fused image, the label export, the SpatialData export and the output profiles

% Status: Accepted (W-329, 2026-10-07, at 9033d46)

This page is the maintained guide to the sample outputs, the export stage, and its
contract.

% the sample outputs (§2.10)

* **What the export does.** It writes what a sample has on disk after spatial assembly:
  the sample tables and `sample.h5ad` of the assembly; the fused image, made by
  multiview-stitcher's `fuse` from the placement of either backend, by default a 2D
  maximum projection at scale 0.5 as OME-TIFF and on request a 3D stack, OME-Zarr or
  further channels; the assembled label image, drawn by retained cells; and the
  SpatialData store; and it reads each file back against its record
  ([Settled decisions this page follows](#settled-decisions-this-page-follows),
  [Terms](#terms), [The fused image](#the-fused-image),
  [Assembled label export](#assembled-label-export),
  [SpatialData export](#spatialdata-export), [Read-back checks](#read-back-checks)).
* **The entry and its configs.** `Dataset.export` applies its checks before anything is
  written and calls the parts `assemble`, `fuse` (`fuse_images`), `fuse_labels` and
  `export_spatialdata` in order, with the part `segment` on the route after fusion; each
  part is turned off by its field of `ExportConfig` and can be called alone, and
  `ExportConfig` holds the `AssemblyConfig`, `FusionConfig`, `LabelConfig` and
  `SpatialDataExportConfig` of the parts ([Names](#names),
  [The export entry](#the-export-entry), [Configuration](#configuration)).
* **What it produces.** The export folder `<OUTPUT>/export/<sample>/`: the record
  `export.json` with a section per part, `sample.h5ad`, the four sample tables in
  `tables/`, the fused images `<image>_projection.ome.tif` or `<image>_stack.ome.tif`,
  `cell_labels.ome.tif` and `sample.zarr/`, as the output profile selects; a part that
  cannot run is recorded and warned about with `ExportIncompleteWarning`, and
  `check_export` checks the folder against its record
  ([The export folder and the output profiles](#the-export-folder-and-the-output-profiles),
  [File layouts](#file-layouts), [Resources](#resources)).
* **Where the configuration goes.** The `ExportConfig` passed to `Dataset.export`, a
  sample-level entry outside `PipelineConfig`; in the workflow, the Python-only `export`
  block, which the per-sample rule `export_sample` reads together with the `assembly`
  block ([Workflow configuration](#workflow-configuration)).

Beneath this guide, [stitching and assembly validation design](https://github.com/wanglab-broad/starfinder/blob/dev/design/validation/stitching-assembly.md) is the engineering validation
design that the sample outputs share with stitching and assembly, with the checks of the
export entry, the fused image, the label export, the SpatialData export and the workflow
(rows EX, FI, LB, SD and WF), their implemented checks and the results of the real
examples; [sample-output-baseline](../baselines/sample-output.md) is history (the reference-branch exporter at
`17e8dda` and `dev` at `8e7d256`, before the stitching and assembly work). The stage has no separate algorithm page:
its numerical rules, measured resources and dependencies are on this page.

% §2.10

## Amendments after acceptance

* 2026-10-10: L6 ("Read-back checks") points to the rule of the route after fusion, where
  the number of cells is first known after the part `segment`, so a fresh export of that
  route cannot refuse before anything is written. Found while implementing the readers and the label export of
  the route after fusion (2026-10-10), and in that review. Operator
  decision, pending Jiahao's confirmation.

% W-428

This page specifies what a sample has on disk after spatial assembly: the one sample-level
export entry `Dataset.export` and its parts; the folder `export/<sample>/` and its output
profiles; the fused image, made by multiview-stitcher's `fuse` from the placement of
either backend, as a 2D projection (the default) or a 3D stack, at an adjustable scale
factor, written as OME-TIFF (the default) or OME-Zarr; the assembled label export; the
SpatialData export of the sample tables; the read-back checks; and the resources and
dependencies each output needs. It builds on three contracts and redefines none of them:

* [stitching specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/stitching.md): `PlacementResult` (per-FOV `status`, float64 `origin_z/y/x`
  in µm), the sample frame (`frame_id` `sample:<sample>`, ZYX, µm, identity direction),
  the index-to-sample rule `sample_zyx = origin_zyx + D · (p ⊙ spacing_zyx)` and its
  inverse, `placed_metadata`, the failure policy, `PlacementReviewRequiredError`, and the
  dependency plan (Python 3.11 or later, the `multiview-stitcher` extra);
* [assembly specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/assembly.md): `assemble`, `AssemblyResult` and its sample tables (`fovs`,
  `cells`, `molecules`, `counts`), `sample.h5ad`, the assembly record, `save_assembly` and
  `load_assembly`, footprints and centres, `ownership_map` with options O1, T1 and Z1 and
  its `candidates` argument, the sample-level identities and the count accounting;
* the `ReferenceGrid` of [segmentation-contract](../../docs/segmentation-contract.md) and `ImageMetadata`
  (`src/python/starfinder/image.py`).

Where this page names a column, a function or a rule of those pages, it uses it unchanged.
It is implemented in `starfinder.export`, nothing is renamed and no dependency changes.

This page was revised on 2026-10-06 after the review gate, from the fusion measurement:
the fused image is now made by multiview-stitcher's `fuse`, the export has one
entry and one folder, and the label export is part of the first delivery. It was revised
again on 2026-10-06 after the second round of that review, from the label measurement:
which fused image is written is a setting of each dataset's configuration, and a
dataset of dense 3D tissue writes the stack alone ("The fused image by dataset"); the
label image is drawn by retained cells ("Assembled label export"); and the choices the
page had left open are decided ("Choices decided at the gate").

% the review gate: W-329; the fusion measurement: W-354; the label measurement: W-359

Evidence, cited by path; this page repeats only the numbers it needs. The run directories
are kept with the run records outside the repository:

* **The fusion measurement**. This page cites its files as "the fusion measurement's
  `<path>`", relative to its run directory: `worker-notes.md`
  (sections 1 to 12), `tables/position_summary.csv`, `tables/overlap_dtype.csv`,
  `tables/overlap_line.csv`, `tables/reduced.csv`, `tables/projection_fixture.csv`,
  `tables/projection_ln_region.csv`, `tables/labels_summary.csv`,
  `tables/labels_absent_ids.csv`, `tables/labels_new_values.csv`,
  `tables/ln_two_placements_grids.csv`, `tables/runs.csv`, `readback/readback.csv`,
  `readback/readback.json`, `readback/spatialdata_read.json`,
  `readback/spatialdata_write_ln_stack_*.json`, `fixtures/writers_lazy.json` and the
  scripts `scripts/common.py` and `scripts/fixtures.py`. The library version measured is
  multiview-stitcher 0.1.62 on Python 3.12.12.
* **The label measurement** of the retained-cell rule. This page cites its files as "the
  label measurement's `<path>`", relative to its run directory: `worker-notes.md`,
  `tables/w354_summary.csv`, `tables/w354_cells.csv`, `tables/w354_incomplete_edges.csv`,
  `tables/w354_checks.json`, `tables/two_fov_summary.csv`, `tables/two_fov_cells.csv`,
  `tables/cases.csv`, `tables/runs.csv` and the script `scripts/labels_retained.py`. It
  ran the same library version in the same environment, on two synthetic fixtures only.
* The stitching input inventory: `inputs.md` (tile shapes, dtypes, calibration).
* The stitching parity runs: `notes.md` section 3 and `fusion/fusion_probe.json` (the first
  label-fusion probe).
* The resource estimate of the sample-output specification: `tables/resource_estimate.csv`
  (the Z extent of cell-culture-3D, derived without reading an image).
* The trial resolution for Python 3.11, recorded in the review gate's preparation
  (`lock-py311-trial/README.md`).

% The fusion measurement: W-354, run directory `/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/runs/W-354/20261006T175525Z-174cf769` (found with `timeout 30 find …/runs/W-354 -maxdepth 2 -name fusion-manifest.json`); cited as "W-354 `<path>`".
% The label measurement: W-359, run directory `/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/runs/W-359/20261007T020704Z-40025a5f` (found with `timeout 30 find …/runs/W-359 -maxdepth 2 -name worker-notes.md`); cited as "W-359 `<path>`".
% The stitching input inventory: W-323, `/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/runs/W-323/` and `20261005T2110Z-input-inventory/inputs.md`.
% The stitching parity runs: W-324, run directory `/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/runs/W-324/20261006T063540Z-5cb2f421`.
% The resource estimate of the sample-output specification: W-328, run directory `/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/runs/W-328/20261006T111902Z-ad45aa68`.
% The trial resolution for Python 3.11: `/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/runs/W-329/20261006T1630Z-gate-review-preparation/lock-py311-trial/README.md`.

## Settled decisions this page follows

Decided by Jiahao in the stitching and assembly planning session (the planning decisions
of 2026-10-05 and their reply "Amendment on the sample outputs"), and in the planning
comments of 2026-09-27 on stitching and assembly:

% the §2.10 planning session (W-152 comment "§2.10 planning decisions (2026-10-05)" and its reply "Amendment on the sample outputs"), and the W-152 comments of 2026-09-27 on §2.10

* **By default** every assembled sample has a sample-level AnnData and a fused image as
  OME-TIFF, a **2D projection** at an **adjustable scale factor**, because the image at
  acquired size is rarely needed.
* **On request**: a SpatialData export of the sample, a 3D stack, OME-Zarr and further
  channels. In the SpatialData store the image is **written once as OME-Zarr** and not
  duplicated.
* **Viewer qualification** in Fiji and napari is **deferred**; opening the files in a viewer
  is not a requirement now.
* 2026-09-27: both image formats are supported; assembled labels are inspection products;
  mask conflicts are flagged, not resolved.

Amended at the review gate (comment "Approval record — 2026-10-06"):

% the W-329 review

* **Fusion.** Starfinder writes no fusion of its own. multiview-stitcher's `fuse` fuses
  the placements of both backends: a placement is one type whichever backend produced it,
  so there is one fusion path. For images the library's own blending is used; the earlier
  rule that an image voxel takes the value of the owning FOV is dropped for image values.
  Assembly's ownership still decides which cells and molecules are retained. BigStitcher's
  own fusion in Fiji is not part of this delivery.
* **Labels** are never blended. Interpolation order 0 with a function that takes the
  first valid view places labels without new values (the stitching parity runs). A small function passed to
  the library is allowed; a second fusion implementation is not.
* **The label export** is part of the first delivery, with a default scale factor of 0.25.
* **The default fused image** is a 2D maximum projection at scale 0.5 in Y and X, as
  OME-TIFF, beside the sample-level AnnData. The 3D stack, OME-Zarr and the SpatialData
  export stay options. The default source image is the reference stain, chosen per
  dataset by configuration.
* **One export entry and one folder**, amending the settled stage split of 2026-10-05: `Dataset.stitch` and
  `stitching/<sample>/` stay as they are; `Dataset.assemble` and `Dataset.sample_outputs`
  become one `Dataset.export`; the package functions `assemble`, `fuse_images`, `fuse_labels` and
  `export_spatialdata` stay separate parts that the method calls in order, each can be
  turned off by configuration and called alone, and when a part cannot run the files of
  the earlier parts are still written and the record says which parts exist. Everything
  the entry writes goes to `export/<sample>/`: one record `export.json` with a section per
  part, `sample.h5ad`, the fused images, the label image, `sample.zarr/` and the four
  sample tables in `tables/`. The assembly statuses, reasons, flags and column names keep
  the word "assembly".
* **Python 3.11 or later** (`requires-python = ">=3.11"`), with the optional extra
  `multiview-stitcher`, which the fuse part names when it is missing; both are applied by
  the operator ([stitching specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/stitching.md), "Dependency plan").

% the stitching parity runs: W-324; the settled stage split of 2026-10-05: D3

Amended at the second round of the review gate (comments "Second round, review
decisions, continued" and "Second-round outcome — 2026-10-06"):

% the W-329 review

* **Dense 3D tissue stores the 3D fused image only.** For dense 3D tissue such as
  LN sample1 a 2D image has little use: the export writes the stack, reduced in scale to
  save storage, and no projection; during development and quality control one plane is
  taken from the stack for a figure. Which image is written stays a setting of each
  dataset's configuration, and the default stays the 2D maximum projection at scale 0.5
  ("The fused image by dataset"; the operator's reading, which Jiahao confirms at the
  gate).
* **Figures of LN sample1 show the middle Z plane**, never a maximum projection, because
  the tissue is dense and a projection blurs it, except its seam sections in XZ and YZ,
  which show every plane of one row or one column of voxels and no projection either
  ("The fused image by dataset"). tissue-2D and cell-culture-3D keep their projection.
* **The label image is drawn by retained cells.** Each FOV contributes only the cells it
  retains under the assembly rule, drawn whole, in place of each FOV's labels cut to its
  ownership region; where two retained cells of different FOVs overlap, the first view in
  placement table order is shown ("Assembled label export").
* **The choices this page had left open are decided as it proposed them** ("Choices
  decided at the gate").

At the checkpoint after the fusion measurement the operator decided how this page uses the measurement: it
wrote the option the fusion measurement proposed for each choice as its text, and Jiahao decided those
choices at the second round as written ("Choices decided at the gate"); the
default OME-TIFF is written by Starfinder with tifffile from the image the library fused,
which is a file writer and not a fusion, and the library's own TIFF is specified as the
alternative ("File layouts"); the measured memory peaks are stated and no bounded memory
is promised for a whole-sample stack; and what the fusion measurement did not measure is marked as an
assumption ("Assumptions").

% W-354

Excluded: viewer sessions, BigStitcher's fusion in Fiji, reassignment of molecules or
recomputation of counts from a fused image or a label raster on the route per FOV (the
route after fusion assigns the sample's molecules to the cells of a segmentation of the
fused image: [sample-segmentation-contract](../../docs/sample-segmentation-contract.md)), merging of cells across
FOVs (the assembly's duplicate rule, decided on 2026-10-09, excludes one of two
observations of a cell before these parts read the cell table: [assembly specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/assembly.md),
"Retention", rule 8), and any change to the placement, segmentation or assignment contracts or to the assembly contract beyond its
`Dataset` method, saved layout and workflow configuration, which move to the export
entry.

% §2.9

## Assumptions

The fusion measurement did not measure the following, and this page relies on them. Each is an assumption
for the implementation issue to verify; the sections that rely on one name it.

% W-354

| | Assumption | What the fusion measurement measured instead |
| --- | --- | --- |
| A1 | `fuse` with lazily loaded tiles (a dask array per tile, read from its file on demand) keeps the peak memory of a whole-sample stack well below the bytes of the loaded tiles. | Every run loaded every tile into memory first; the LN stack peaked at 12.35 GiB (`tables/runs.csv`, `ln-mvs-stack-s05-a2`; `worker-notes.md` section 8). |
| A2 | Label fusion behaves on real label images, and on 3D label volumes, as on the 2D fixtures. | One synthetic 2D scene of discs (`tables/labels_summary.csv`); the label measurement tested the retained-cell rule on that scene and on a hand-built 2D fixture of two FOVs, both synthetic (the label measurement's `tables/w354_summary.csv`, `tables/two_fov_summary.csv`). |
| A3 | tifffile writes and reads back a BigTIFF above 4 GiB with the layout of "OME-TIFF layout"; the library's own TIFF writer (the alternative) writes a valid file above 4 GiB. | The largest file was the 2,247,808,158-byte LN stack TIFF (`readback/readback.csv`). |
| A4 | A whole-sample stack of cell-culture-3D costs about 160 s and 8 GiB at scale 0.5. | An estimate from voxel counts, not a run (`worker-notes.md` section 10). |
| A5 | Input dtypes other than uint8, uint16 and float32 pass through `fuse` as they do. | uint8, uint16 and float32 only (`tables/overlap_dtype.csv`). |
| A6 | Two `fuse` calls with the same inputs and arguments return identical arrays. | Each configuration was fused once. |
| A7 | The library's multiscale of a fused image (`msi_utils.get_msim_from_sim`) gives SubIFD levels whose shapes and geometry follow "Resolution levels inside a file" also for odd sizes. On an axis with one voxel the library reports a spacing of 1.0, and the record carries the formula's value. | The library's block mean per tile (`tables/reduced.csv`, route `level1`) and its OME-Zarr levels (`readback/readback.json`). |
| A8 | Starfinder's tiled, zlib-compressed OME-TIFF writer, writing band by band from the lazy fused array, costs about as much as the library's uncompressed writer plus compression. | The library's TIFF writer only (`tables/runs.csv`). |
| A9 | A labels element parsed from the label OME-TIFF (read with tifffile's `aszarr`) writes and reads back like the image element. | Image elements only (`readback/spatialdata_read.json`). |
| A10 | A SpatialData store in container format 0.1 on Zarr format 2 (option Z1) writes and reads back under spatialdata 0.6.1 like the default format. | The default format only (`readback/spatialdata_write_ln_stack_*.json`). |
| A11 | Withdrawn with the owner-masked label rule, an option not taken ("Assembled label export"). It stated that where the owner-masked image does not show the owner's label, the point lies within one input voxel of an O1 territory boundary. | The counts of such points on the fixture (`worker-notes.md` section 6). |
| A12 | A fused image rechunked with dask before the library's OME-Zarr writer is written with those chunks. | Chunks follow the dask chunks passed (`fixtures/writers_lazy.json`). |
| A13 | Every output voxel outside the voxel-centre extent of every included FOV, in the axes fused, holds 0, and a FOV's `level1` image covers no output voxel outside the FOV's level-0 extent. | 0 at the two uncovered corners of the overlap fixture (`tables/overlap_dtype.csv`). |

% W-354; W-359; W-359 `tables/w354_summary.csv`

## Terms

* The **fused image** is one image of a sample in the sample frame, made by
  multiview-stitcher's `fuse` from one saved image per FOV (a **source image**) and the
  placement.
* The **fusion grid** is the library's union grid of the included FOVs at the acquired
  spacing ("The fusion grid"). A file's **output level** is the fusion grid reduced by an
  integer factor `k` in Y and X; its **scale factor** is `1/k`. A file's further
  **resolution levels** reduce its output level by a factor of 2 on the axes that each
  level halves, which depend on the writer ("Resolution levels inside a file").
* The **3D stack** is the fused image with a Z axis; the **2D projection** collapses Z.
  The **middle plane** of a stack of `n_z` planes is the plane of zero-based index
  `n_z // 2` ("The fused image by dataset").
* A **part** is one of the four functions `Dataset.export` calls, and is named after it:
  `assemble`, `fuse` (the function `fuse_images`), `fuse_labels`, `export_spatialdata`.
* The **export folder** is `export/<sample>/`; an **output profile** is the set of files it
  holds for a given configuration ("The export folder and the output profiles").
* The **store** is the SpatialData Zarr store `export/<sample>/sample.zarr/`.
* A FOV is **included** in fusion when it has a finite origin and a source image: status
  `image_supported` or `metadata_only` ([stitching specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/stitching.md)). `placeholder` FOVs (a
  blank image), `missing` and `failed` FOVs contribute nothing.

## Names

| Option | Names | Effect on the golden values | Effect on existing files and outputs |
| --- | --- | --- | --- |
| **N1. One `starfinder.export` module (recommended)** | Coordination `Dataset.export`; config `ExportConfig`; fusion `fuse_images`, `fused_grid`, `FusionInput`, `FusionConfig`, `FusedGrid`, `FusionRecord`; labels `fuse_labels`, `LabelConfig`, `LabelRecord`; export `export_spatialdata`, `SpatialDataExportConfig`; record `ExportRecord`, `ExportIncompleteWarning`; check `check_export`; lookup `lookup_source`. `assemble`, `save_assembly`, `load_assembly` and `write_sample_h5ad` stay in `starfinder.assembly`. | None: the golden tests pin the legacy scripts and are untouched. | New files only, plus one method on `Dataset`; the exports go into `__all__` and the autosummary list of `docs/api/export.rst`, which `docs/check_reference.py` requires. The legacy `images/fused/{sample}/` folder is not written. |
| N2. Two modules, `starfinder.fusion` and `starfinder.sample_export` | The same functions split by output; `sample_export` is the reference-branch name | None. | The same files; the parts of one entry are spread over two modules, and `sample_export` reuses a name whose reference-branch API (`export_sample` with affine source mappings) this page does not adopt. |
| N3. Inside `starfinder.assembly` | `starfinder.assembly.fuse_images`, `export_spatialdata` | None. | The same files; the assembly module grows image I/O and the optional multiview-stitcher, zarr and spatialdata imports. |

**Recommendation: N1.** It keeps the optional imports out of the assembly module, which
`assemble` needs without any extra, and does not reuse a reference-branch name for a
different API.

New names under N1:

| Concept | Name | Kind |
| --- | --- | --- |
| Coordination | `Dataset.export(sample, *, placement, assignment="default", config=ExportConfig(), output_dir)` | method |
| Config | `ExportConfig` ("Configuration") | frozen dataclass |
| Fusion | `fuse_images(placement, inputs, *, assembly=None, config=FusionConfig(), output_dir) -> FusionRecord` | function |
| Fusion input | `FusionInput(fov_id, grid, images)` | frozen dataclass |
| Fusion config | `FusionConfig` ("Configuration") | frozen dataclass |
| Geometry | `fused_grid(placement, grids, *, config) -> FusedGrid` | function, frozen dataclass |
| Labels | `fuse_labels(placement, assembly, *, config=LabelConfig(), output_dir) -> LabelRecord` | function |
| Label config | `LabelConfig` ("Configuration") | frozen dataclass |
| Export | `export_spatialdata(assembly, *, fused=None, labels=None, config=SpatialDataExportConfig(), output_dir) -> Path` | function |
| Records | `FusionRecord`, `LabelRecord`, `ExportRecord`: the content of their sections of `export.json` and the written paths | frozen dataclasses |
| Warning | `ExportIncompleteWarning`, issued when a part could not run | `UserWarning` subclass |
| Check | `check_export(folder)` | function; raises `ValueError` naming the first failed condition |
| Lookup | `lookup_source(folder, *, spot=None, cell=None)` | function |
| Route values | `ROUTES = ("per_fov", "after_fusion")` | tuple |
| Settings of the route | `SampleSegmentationConfig(runs, cells, nuclei=None, assignment=AssignmentConfig(), device="cpu")` | frozen dataclass |
| One run | `SampleSegmentationRun(name, target, inputs, method, seeds=None)` | frozen dataclass |
| The segment part | `segment_fused(output_dir, *, config, sample, identity=None)` returning `SampleSegmentation` | function |
| Its result | `SampleSegmentation(grid, runs, record)` (`starfinder.segmentation`) | frozen dataclass |
| Its loader | `load_sample_segmentation(output_dir)` | function |
| The assembly of this route | the keyword `segmentation: SampleSegmentation | None = None` of `assemble` (`starfinder.assembly`) | keyword |
| The consistency check | `compare_routes(per_fov, after_fusion, *, threshold=0.5, output=None)` returning `RouteComparison` | function, frozen dataclass |
| Export settings | `ExportConfig.route`, `ExportConfig.segment`, `ExportConfig.segmentation` | fields |
| Record | the `segmentation` section of `export.json`; the key `route` in the `assembly` section | JSON |

## The export entry

```python
Dataset.export(sample: str, *, placement: PlacementResult | str | Path,
               assignment: str | None = "default",
               config: ExportConfig = ExportConfig(),
               output_dir: str | Path) -> ExportRecord
```

`placement` is a `PlacementResult` or its saved folder `stitching/<sample>/`, read with
`load_placement`. `assignment` is the assignment name read per FOV, or `None` for a
sample without segmentation (molecules only). `output_dir` is the sample's export folder;
the workflow passes `<OUTPUT>/export/<sample>/`. Like `stitch`, it is a sample-level
computation, not a multi-FOV runner ([stitching specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/stitching.md), "Not the multi-FOV
runner").

% §2.9

### The parts, in order

| Order | Part | Turned off by | What the method does | Writes in `output_dir` |
| --- | --- | --- | --- | --- |
| 1 | `assemble` ([assembly specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/assembly.md)) | `assemble: false` | Builds one `AssemblyInput` per FOV of the placement from the saved assignment folders named `assignment` (or, with `assignment=None`, from each FOV's saved molecule table `fovs/<fov_id>/molecules.csv` read through `load_molecules`, with the identities it holds; since 2026-10-23, and on the workflow's legacy route the one-based `signal/<fov_id>_goodSpots.csv` read through `import_legacy_molecules_csv`, with the identities `csv:<row>`) and the reference grids (the stored metadata of each FOV's reference image, the file the dataset's `reference_image` names, default `fovs/{fov_id}/reference.tif`, read as `reference_grid_from_file` reads it, or, for a file that stores none, the grid the dataset's `voxel_size_xy` and `voxel_size_z` declare; without them such a file raises `ValueError` naming them, and a missing file raises `FileNotFoundError` naming `reference_image` and the path), calls `assemble`, and saves the result with `save_assembly(result, output_dir, table_format=config.table_format)`. It keeps the tables of every FOV and at most two label images in memory at a time ([assembly algorithms specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/assembly-algorithms.md), "Flags"). Turned off, it loads the saved assembly with `load_assembly(output_dir)`, which checks every SHA-256, and records the part as `reused`; when the folder holds no saved assembly and no later part needs one (check 4), it records the part as `off` and `fuse` runs without an assembly. | `tables/fovs.<fmt>`, `tables/cells.<fmt>`, `tables/molecules.<fmt>`, `tables/counts.<fmt>`, `sample.h5ad`; section `assembly` |
| 2 | `fuse` | `fuse: false` | Builds one `FusionInput` per included FOV from the saved source images and reference grids, and calls `fuse_images` with the assembly of part 1. | The fused images; section `fusion` |
| — | `segment` ([sample-segmentation-contract](../../docs/sample-segmentation-contract.md)) | `segment: false` (reuses the saved sample segmentation) | On the route per FOV, recorded `off` (reason `route_per_fov`, or `no_cells` with `assignment=None`). On the route after fusion, calls `segment_fused` on the fused projection, or reuses the saved sample segmentation. | `segmentation_<run>.ome.tif`; section `segmentation` |
| 3 | `fuse_labels` | `fuse_labels: false` (the default; "Choices decided at the gate", 8) | Calls `fuse_labels` with the placement and the assembly. | `cell_labels.ome.tif`; section `labels` |
| 4 | `export_spatialdata` | `export_spatialdata: false` (the default) | Calls `export_spatialdata` with the assembly and the records of parts 2 and 3 that were written. | `sample.zarr/`; section `spatialdata` |

% W-562

The order above is that of the route per FOV. On the route after fusion with an
`assignment` name the parts run in the order `fuse`, `segment`, `assemble` (the
`molecules_only` assembly of the molecule tables, assigned to the cells of the sample
segmentation), `fuse_labels`, `export_spatialdata`
([sample-segmentation-contract](../../docs/sample-segmentation-contract.md), "The parts on each route").

Each part can also be called alone with the saved files of the parts before it:
`assemble` and `save_assembly` as [assembly specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/assembly.md) specifies;
`fuse_images(placement, inputs, *, assembly=None, config, output_dir)`;
`fuse_labels(placement, load_assembly(output_dir), *, config, output_dir)`; and
`export_spatialdata(load_assembly(output_dir), fused=…, labels=…, config, output_dir)`,
which reads the fused and label files from the folder through their records. A part
called alone writes its files and its section of `export.json` (creating the record when
it does not exist) and leaves the other sections unchanged; each section records the
SHA-256 of the sections it was computed from, so `check_export` finds a section made from
an earlier assembly ("Read-back checks", R1). On the route after fusion, where a section
holds times, files take the place of sections: the `segmentation` section records the
SHA-256 of each fused file it read, and the `assembly` section the SHA-256 of each label
file it read (`segmentation_files`) ([sample-segmentation-contract](../../docs/sample-segmentation-contract.md), "What the
segmentation was made from").

### Checks before anything is written

In this order; each raises, names the FOVs or keys, and leaves the folder unchanged:

1. **Config.** `ExportConfig`, `FusionConfig`, `LabelConfig` and `AssemblyConfig`
   validate their fields in `__post_init__` ("Configuration").
2. **Review.** A placement whose review is `required` without an approval raises
   `PlacementReviewRequiredError`, exactly as `assemble` does ([assembly specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/assembly.md),
   "Checks the entry applies", 2).
3. **Required extras.** The assemble part runs only with the `anndata` extra
   (`sample.h5ad`) and, for `table_format: parquet`, the `checkpoint` extra; without one
   the method raises `ModuleNotFoundError` naming it. These are the first part's own
   outputs, so nothing earlier exists to keep.
4. **Assembly.** With `assemble: true`, the checks of [assembly specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/assembly.md) ("Checks
   the entry applies") run inside `assemble`, which computes in memory before
   `save_assembly` writes. With `assemble: false`, when `fuse_labels` or
   `export_spatialdata` is on, `output_dir` must hold a saved assembly that `load_assembly`
   accepts and whose placement SHA-256 values equal the placement's (`ValueError`).
5. **Fusion inputs**, when `fuse` is on: every included FOV has exactly one `FusionInput`
   and no input names another FOV (`ValueError`); each input's grid equals the grid the
   placement recorded for that FOV, by the grid rule of the [assignment specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/assignment.md#the-grid-rule)
   reading `µm`, `um` and `micrometer` as one unit and a grid without a direction as
   having the identity direction, with known spacing, unit `µm` and identity direction
   (`IncompatibleGeometryError`); all
   included FOVs share one spacing, as `stitch` requires, and either all have one plane
   or none has (`ValueError`, "The fusion grid").
6. **Label IDs**, when `fuse_labels` is on: the sample cell table has at most
   16,777,216 rows (2^24), so every `instance_id` fits in float32 exactly ("Assembled
   label export", "IDs and float32"; `ValueError` naming the count).
7. **Route** ([sample-segmentation-contract](../../docs/sample-segmentation-contract.md), "The entry", check 7): the route is a
   value of `ROUTES` or null; `route: after_fusion` with `dimensionality: stack` or with
   `projection: sum` is refused, for a set route and for the default alike; on the route
   after fusion with an `assignment` name, `segmentation` is given, each run's input
   images are names of `images`, `dimensionality` writes the projection, `fuse` is on
   unless `segment` is off, `assemble` is on unless `segment` is off, the reuse conditions
   hold when `segment` is off, the runs' configs validate, their seeds name earlier runs,
   `cells` and `nuclei` name runs of the targets `cell` (or `nucleus` as territories) and
   `nucleus`, and every model resolves with its SHA-256 (`resolve_model`, which imports no
   segmentation library). Each failure raises `ValueError` (or the model errors of
   [segmentation-contract](../../docs/segmentation-contract.md)) naming the key.

### When a part cannot run

A part **cannot run** when an optional extra it needs is not installed or a file it reads
is missing. That is not an error of the entry: the method records the part as `not_run`
with its reason, issues `ExportIncompleteWarning` naming the part and the reason, runs the
later parts that do not need the missing output, and returns. The files of the earlier
parts are written and listed. `export.json` has `complete: false` and lists every part
with its status.

| Condition | Part | Status, reason | The later parts |
| --- | --- | --- | --- |
| `multiview_stitcher` not importable (the extra not installed, or Python 3.14 and later, where the extra is empty) | `fuse` | `not_run`, `extra_missing`, `extra: "multiview-stitcher"` | `fuse_labels` likewise `not_run`, `extra_missing`; `export_spatialdata` runs and records the image and labels elements as omitted, reason `part_not_run` |
| A selected source image missing for an included FOV | `fuse` | `not_run`, `source_image_missing`, with each FOV and image name | No image is fused, for any image name, so the part is all or nothing. `fuse_labels` runs (it reads no source image); `export_spatialdata` omits the image elements (`part_not_run`) |
| A territory file of a FOV with a retained cell missing | `fuse_labels` | `not_run`, `territory_missing`, with each FOV and path | `export_spatialdata` omits `cell_labels` (`part_not_run`) |
| `spatialdata` not importable | `export_spatialdata` | `not_run`, `extra_missing`, `extra: "spatialdata"` | none |
| On the route after fusion, `fuse` did not write the fused images | `segment` | `not_run`, `part_not_run` | `assemble` writes the tables of the `molecules_only` assembly and `sample.h5ad` without a row, its section with `mode` `molecules_only` and the reason ([sample-segmentation-contract](../../docs/sample-segmentation-contract.md), "The entry") |
| On the route after fusion, the backend of a segmentation method not importable | `segment` | `not_run`, `extra_missing`, naming the method's extra (for example `cellpose`) | as above |

Two other outcomes are not failures: `fuse_images` with no included FOV writes no image and
records `status: "empty"`, reason `no_included_fov` (and `segment` on the route after
fusion likewise); `fuse_labels` on a `molecules_only`
sample writes no label image and records `empty`, reason `no_cells`.

Any other error inside a part (a library error, a write error, a territory file whose
SHA-256 differs from the assembly record) records the part as `failed` with the
exception's type and message and the files it completed, writes `export.json`, and is
raised again. The files of the earlier parts stay.

The statuses of a part in `export.json` are therefore `written`, `reused` (assemble
and segment), `off`, `empty`, `not_run` and `failed`.

### How the folder is kept equal to its record

* Before the parts run, the method removes the files that the previous `export.json`
  lists for each part that will run or is off; files of a `reused` assembly stay. It
  never removes a file the record does not list.
* Each part writes each file atomically (a temporary name, then a rename); the store is
  written into a sibling staging folder and renamed into place, so a failed export
  publishes no partial store.
* Before the first part, `export.json` records each part that will run as `not_run` with
  the reason `export_not_finished`, and each part that is off as `off`; each part replaces
  its entry with its own section when it ends, whatever its status.
* `export.json` is written atomically after each part, so after an interruption it lists
  the parts that finished, shows each part that had not finished as `not_run`,
  `export_not_finished`, and has `complete: false`.

## The export folder and the output profiles

`<OUTPUT>/export/<sample>/` holds everything the entry writes. It sits beside the
placement folder `<OUTPUT>/stitching/<sample>/`, which is unchanged
([stitching specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/stitching.md), "Persistence"), and is not the legacy
`<OUTPUT>/images/fused/<sample>/`, which keeps BigStitcher's inputs and `dataset.xml`
([stitching baseline](https://github.com/wanglab-broad/starfinder/blob/dev/design/baselines/stitching.md)); no existing file is rewritten. `<image>` is a source-image
name of "Fused image inputs" (`reference_stain` by default); `<fmt>` is `csv` by default
or `parquet`.

| Profile | Selected by | Files that exist under `export/<sample>/` | Format | What each contains |
| --- | --- | --- | --- | --- |
| **Default** | nothing (the defaults of "Configuration") | `export.json` | JSON | The record: one section per part ("The record"); `labels` and `spatialdata` have status `off` |
| | | `tables/fovs.csv`, `tables/cells.csv`, `tables/molecules.csv`, `tables/counts.csv` | CSV | The four sample tables of [assembly specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/assembly.md), "Sample tables" |
| | | `sample.h5ad` | H5AD | The sample-level AnnData of [assembly specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/assembly.md): one row per counted cell |
| | | `<image>_projection.ome.tif` | OME-TIFF: BigTIFF, 512 × 512 tiles, zlib ("OME-TIFF layout") | The maximum projection of `<image>` fused by the library at scale 0.5 in Y and X, in the input dtype, with calibration and origin; not written when no FOV is included, nor for a dataset that sets `dimensionality: stack` |
| Parquet tables | `table_format: parquet` (the `assembly` block) | `tables/*.parquet` instead of `tables/*.csv` | Parquet | The same tables |
| No fused image | `fuse: false` | the default files without `<image>_projection.ome.tif` | | Section `fusion` has status `off` |
| 3D stack alone (dense 3D tissue, such as LN sample1) | `dimensionality: stack` in the dataset's configuration | `<image>_stack.ome.tif` instead of the projection; no projection file | OME-TIFF | The 3D fused image at `scale_yx` in Y and X, Z as acquired ("The fused image by dataset"); refused on the route after fusion ([sample-segmentation-contract](../../docs/sample-segmentation-contract.md), "The route setting") |
| Stack and projection | `dimensionality: both` | `<image>_stack.ome.tif` beside the projection | OME-TIFF | Both images, on equal Y and X grids |
| Sum projection | `projection: sum` | the projection file | OME-TIFF, float32 | The per-FOV sums over Z, fused ("Dtypes written"); refused on the route after fusion ([sample-segmentation-contract](../../docs/sample-segmentation-contract.md), "The route setting") |
| Another scale factor | `scale_yx: 1/k` | the same names | | The output level of factor `k` |
| Standalone OME-Zarr | `ome_zarr: true` with `export_spatialdata: false` | `<image>_projection.ome.zarr/` (and `_stack.ome.zarr/`) beside each OME-TIFF | OME-NGFF 0.5 on Zarr format 3 ("OME-Zarr layout") | The same array as the OME-TIFF, with the library's resolution levels |
| Further channels | `images: [reference_stain, reference_merged, …]` | one file per image name and dimensionality, as above | as above | Each selected source image fused on the same grid with the same placement |
| Pyramid in OME-TIFF | `pyramid_levels: L` | the same files | SubIFD levels | `L` further levels, each halving Y and X ("Resolution levels inside a file") |
| Label export | `fuse_labels: true` | `cell_labels.ome.tif` | OME-TIFF, uint32 | The `instance_id`s of the retained cells, each drawn whole by its FOV and placed without blending, at `labels_scale_yx` (default 0.25) ("Assembled label export"); not written for a `molecules_only` sample or without an included FOV |
| SpatialData | `export_spatialdata: true` | `sample.zarr/` | SpatialData store, spatialdata's default format (container format 0.2 on Zarr format 3) | Points `molecules`, shapes `cell_centroids`, tables `cells`, `expression`, `cells_unplaced`, `molecules_unplaced`, one image element per fused image and, with the label export, the labels element `cell_labels`; each written when not empty ("SpatialData export"). No standalone `.ome.zarr` is written; the OME-TIFF files stay. |
| Route after fusion | `route: after_fusion`, or by default for `dimensionality: projection` ([sample-segmentation-contract](../../docs/sample-segmentation-contract.md)) | the default files, the fused images of the segmentation's stains among them, and `segmentation_<run>.ome.tif` per segmentation run; with an expansion of the assignment, `territories.ome.tif` | OME-TIFF, uint32 | Each run's labels on the segmentation grid (the fused projection's), and the territories the sample assignment sampled (values `cell_id`); section `segmentation` |
| Assembly reused | `assemble: false` | the tables and `sample.h5ad` of the earlier run, unchanged | | Section `assembly` has status `reused` |
| A part that cannot run | "When a part cannot run" | the files of the parts that ran | | The part's section has status `not_run` and its reason; `complete: false` |

A reader checks a finished folder against its record: `export.json` lists every file of
every part with its SHA-256, and `check_export` fails on a file that exists but is not
listed, or is listed but missing.

## The fused image

### The fused image by dataset

Which of the projection and the stack is written is a setting of each dataset's
configuration: `dimensionality` in the `export` block of the dataset's workflow
configuration ("Workflow configuration"), passed as `FusionConfig.dimensionality`. It
applies to every image name of `images`.

| `dimensionality` | Files written per image name | Used by |
| --- | --- | --- |
| `projection` (the default) | `<image>_projection.ome.tif`: the 2D maximum projection, at scale 0.5 unless `scale_yx` says otherwise ("The 2D projection") | every dataset that does not set the key: tissue-2D and cell-culture-3D |
| `stack` | `<image>_stack.ome.tif` only ("The 3D stack"); no projection file is written | a dataset of dense 3D tissue, such as LN sample1, for which a 2D image has little use |
| `both` | both files, on equal Y and X grids | on request |

With `stack` the `fusion` section lists the stack files only, the `projection` key has no
effect, and the store holds the stack element ("SpatialData export", D1).

**How the stack is reduced.** In Y and X by `scale_yx = 1/k`, the same key and the same
`level1` route as the projection ("The scale factor"); Z stays as acquired, with the
fusion grid's `s_z`, `o_z` and `n0_z`. No reduction in Z is offered, because none was
measured. The LN stack at scale 0.5 was measured at 67 × 6,166 × 5,441 uint8, that is
2,247,796,802 bytes of voxels, written by the library's uncompressed TIFF writer as a
2,247,808,158-byte file (the fusion measurement's `tables/runs.csv`, row `ln-mvs-stack-s05-a2`;
`readback/readback.csv`). Derived here from the fusion grid of 67 × 12,332 × 10,882
(the fusion measurement's `tables/ln_two_placements_grids.csv`) with `ceil(n0/k)` in Y and X, not measured:
8,991,187,208 bytes of voxels at scale 1 (8.37 GiB, rounded) and 562,052,481 bytes at
scale 0.25 (67 × 3,083 × 2,721; 0.52 GiB, rounded). What zlib saves in Starfinder's
OME-TIFF was not measured (A8).

% W-354 `tables/runs.csv`; W-354 `tables/ln_two_placements_grids.csv`

**One plane for a figure.** No single-plane file is written and no configuration key for
one exists: a reader takes one plane from the stack file. In the OME-TIFF layout each Z
plane of a stack is one page of the file's series, so `tifffile.imread(path, key=i)` reads
plane `i` alone, and that plane lies at sample position `PositionZ = O_z + i·S_z` with the
file's Y and X spacing and origin ("OME-TIFF layout"). From the store, the plane is
`[0, i]` of the element `<image>_stack` (axes `c, z, y, x`); the stack of one-plane FOVs
is a 2D file and a 2D element, which is its only plane. The **middle plane** of a stack of
`n_z` planes is the zero-based index `i = n_z // 2`:

* odd `n_z = 2m + 1`: `i = m`, the exact middle, with `m` planes on each side (the LN stack
  of 67 planes: plane 33);
* even `n_z = 2m`: `i = m`, the second of the two middle planes in index order, with `m`
  planes before it and `m − 1` after it (4 planes: plane 2);
* one plane (`n_z = 1`, the stack of one-plane FOVs): `i = 0`, the only plane.

Figures of LN sample1 (placement overviews, ownership regions, seam views and any view of
its fused image) show its middle plane, never a maximum projection, except its seam
sections in XZ and YZ through the stack, which show every plane of one row or one column
of voxels and no projection either ([stitching and assembly validation design](https://github.com/wanglab-broad/starfinder/blob/dev/design/validation/stitching-assembly.md), "Outputs to
inspect").

### Fused image inputs

* **A placement** (`PlacementResult`, saved as `stitching/<sample>/`), from either backend:
  the BigStitcher import and `multiview_stitcher` give one type, and both are fused by the
  same call ("Fusion by multiview-stitcher").
* **Per included FOV, the selected source images on its reference grid**, named as the
  stitching inputs are ([stitching specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/stitching.md), "Input image"): `reference_stain` (the
  saved `reference_stain` of `FOV.prepare_morphology`, whose stain each dataset's
  configuration chooses) by default; `reference_merged` (the FOV's reference image, the file
  the dataset's `reference_image` names, default `fovs/{fov_id}/reference.tif`);
  or `<name>=file:<pattern>` with `{fov_id}`, accepted only when its shape equals the
  reference grid's, where `<name>` names its files and store element and may use letters,
  digits, `_`, `-` and `.` only. Further channels are further names in `images`. A source
  file is one channel, ZYX or YX once its axes of length 1 other than Y and X are set
  aside, so a channel axis or any further axis of length more than 1 is refused; a TIFF
  series that carries no axis metadata (not OME, not ImageJ, no description naming axes)
  and whose axes tifffile reports as exactly one page axis over Y and X (`IYX` or `QYX`)
  is read as ZYX, lazily like the others.
* **Optionally the assembly**, whose record's placement SHA-256 values must equal the
  placement's (`ValueError`), so that the image and the tables come from one placement.
  Fusion uses no ownership: the assembly is only recorded.

### Fusion by multiview-stitcher

`fuse_images` makes every fused image with one call of multiview-stitcher's
`multiview_stitcher.fusion.fuse`, for the placement of either backend. Starfinder chooses
no candidate, interpolates nothing and blends nothing. Around the call it builds the
library's inputs from the placement, computes the output grid it passes, projects each
FOV along Z for the projection route, requests the library's reduced level of each FOV
for a scale factor, and writes the files.

**From a `PlacementResult` to the library's input** (the fusion measurement's `worker-notes.md` section 1,
`scripts/common.py`, functions `mvs_placement` and `bigstitcher_placement`): each
included FOV and source image becomes one image

% W-354 `worker-notes.md`

```python
spatial_image_utils.get_sim_from_array(
    array,                                   # the source image, or its projection
    dims=["z", "y", "x"],                    # ["y", "x"] for a projection or a one-plane FOV
    scale={axis: spacing_um},                # the grid's ImageMetadata spacing
    translation={axis: origin_um},           # the placement's origin_z/y/x of the FOV
    transform_key="placed")
```

whose affine under the key `placed` is the identity: the placement's origin is the sample
position of voxel (0, 0, 0), so the image's own coordinates are sample coordinates. The
images of all included FOVs, in placement table order, are passed to
`fusion.fuse(images, transform_key="placed", fusion_func=weighted_average_fusion,
interpolation_order=1, output_stack_properties=…)`, with the output grid of "The fusion
grid" or "The scale factor". The fusion measurement fused LN sample1 from both placements this way
(`tables/ln_two_placements_grids.csv`).

% W-354

**Fusion function and interpolation order for images.** `weighted_average_fusion`, the
library's default, at interpolation order 1 (linear), with the library's default
`blending_widths` of 3 µm in Z and 10 µm in Y and X (the fusion measurement's `worker-notes.md` section 3).
They are fixed, not configuration keys. `multi_view_deconvolution` is not offered: it
creates values outside the input range, changes single-coverage values (100 became 99 in
uint8) and leaves 9.9999997e-05 (about 1e-4) in uncovered float32 voxels (the fusion measurement's
`tables/overlap_dtype.csv`, `uncovered_corner_*` of the `multi_view_deconvolution` float32
row).

% W-354 `worker-notes.md`; W-354

**What a fused image holds.** Each statement rests on the fusion-measurement record named beside it:

% W-354

| Property | Statement | Fusion-measurement record |
| --- | --- | --- |
| Values where one FOV covers | Where the FOV's voxel centres coincide with output voxel centres, the input values unchanged. At fractional offsets, linear interpolation of the FOV's neighbouring voxels: a single bright voxel spreads over its neighbours, and its intensity centroid stays at most 0.0008183085523003642 µm from its true position, below the bound 0.00082 µm (`max_distance_centroid_um` of the order 1, scale 1, route `none` row). | `tables/overlap_dtype.csv` (`single_a_value`, `single_b_value`); `tables/position_summary.csv` (order 1, scale 1) |
| Values in an overlap | The weighted average of the covering FOVs, with a cosine ramp over the blending width from each FOV's edge. Values lie between the inputs' values; integer averages are truncated, not rounded, so two constant FOVs of 100 and 203 give a ramp from 100 to 202 and an equal-weight mean of 151, not 152. | `tables/overlap_dtype.csv`, `tables/overlap_line.csv` |
| Dtype | The input dtype, in memory, in the library's TIFF and OME-Zarr, and in Starfinder's OME-TIFF (which writes the array it is given). `fuse` converts every input to float32, resamples, and casts the result back with `astype`. | `tables/overlap_dtype.csv` (dtype columns); `worker-notes.md` section 1 |
| Output grid | The grid passed as `output_stack_properties`; the library's recorded origin, spacing and shape equal it ("The fusion grid", "The scale factor"). | `tables/reduced.csv` (`geometry_matches_formula`) |
| Uncovered voxels | 0, also 0.0 (not NaN) for float32: the library fills NaN outside the views and converts it with `nan_to_num` before the cast. | `tables/overlap_dtype.csv` (`uncovered_*`, `uncovered_is_nan`) |
| Position | At scale 1, a single bright voxel's maximum lies on the output voxel nearest its true position. | `tables/position_summary.csv` |

% W-354

**Allowed input dtypes.** uint8, uint16 and float32, the dtypes the fusion measurement fused; one dtype
across FOVs, with no implicit conversion. `fuse_images` refuses any other dtype with
`ValueError` naming it and the allowed list. Integers of 8 and 16 bits are exact in
float32. Other dtypes are an assumption (A5): above 2^24, 32-bit integers are not exact
in float32, and float64 inputs would pass through float32 ("Choices decided at the gate", 9).

% W-354

### The fusion grid

The fusion grid is the library's union grid of the included FOVs, computed by
`fused_grid` from the placement and the FOVs' grids only, with the library's rule
(`fusion.process_output_stack_properties`; the fusion measurement's `worker-notes.md` section 4,
`tables/reduced.csv`):

% W-354 `worker-notes.md`

* **Spacing** `s = (s_z, s_y, s_x)`: the included FOVs' shared spacing, µm.
* **Origin** `o`: per axis the smallest origin of an included FOV, `o_a = min_f origin_f,a`.
* **Shape** `n0`: per axis `floor((max_f (origin_f,a + (n_f,a − 1)·s_a) − o_a) / s_a + 1e-9) + 1`,
  the number of grid voxel centres up to the last voxel centre of any included FOV. A FOV's
  last fractional part beyond the last grid centre is not covered: the grid can be one
  voxel shorter than a grid that contains every FOV's last voxel centre. On LN sample1 the
  grid is 67 × 12,332 × 10,882 (the fusion measurement's `tables/ln_two_placements_grids.csv`, columns
  `scale1_shape_y` and `scale1_shape_x`, and `tables/runs.csv`, `ln-mvs-stack-s05-a2`).
* **Metadata**: `ImageMetadata(frame_id="sample:<sample>", spacing_zyx=s, origin_zyx=o,
  direction_zyx=identity, spatial_unit="µm")`. Index `v` of the grid is at sample position
  `o + v ⊙ s`.

% W-354 `tables/ln_two_placements_grids.csv`

`fuse_images` passes the output level's grid ("The scale factor") explicitly as
`output_stack_properties`, so the file's geometry is the recorded grid by construction;
the fusion measurement found the library's recorded origin, spacing and shape equal to these formulas in
every case but one, the one-plane 3D input below.

% W-354

**One-plane FOVs.** A one-plane 3D input gets a Z spacing of 1.0 in the library instead of
its own, because the library derives spacing from coordinates (the fusion measurement's `tables/reduced.csv`,
case `z1_3d`: `z_spacing_lib` 1.0 against 0.5). When every included FOV has Z = 1, `fuse_images`
therefore passes them as 2D images (dims `["y", "x"]`) and keeps Z in its own record: the
grid has one plane, `n0_z = 1`, and its Z extent is the slab
`[z_lo, z_hi) = [min_f origin_f,z − s_z/2, max_f origin_f,z + s_z/2)`. The stack and the
projection then both have the slab geometry in Z: spacing `S_z = z_hi − z_lo` and origin
`O_z = (z_lo + z_hi)/2`, which are `s_z` and `o_z` when every `origin_z` is equal. FOVs
whose `origin_z` differ are fused into that one plane and the record lists each FOV's
`origin_z`. A placement whose
included FOVs mix one-plane and multi-plane grids is refused (`ValueError`), because the
library fuses one dimensionality at a time.

% W-354 `tables/reduced.csv`

Excluded FOVs (`placeholder`, `missing`, `failed`) are not passed to the library and do
not extend the grid; the area only they would cover is uncovered.

**No included FOV.** When no FOV is included (every FOV excluded, or a placement whose
placed FOVs are all placeholders), there is no grid: `fuse_images` writes no image file and
records the part as `empty` with `outputs: []` and `reason: "no_included_fov"`. Such a
placement has no owning FOV either, so through `Dataset.export` with the assemble part on
it is refused by `assemble` before anything is written ([assembly algorithms specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/assembly-algorithms.md),
"Ownership"); the `empty` record arises when `fuse_images` is called alone.
**One included FOV** gives a grid equal to that FOV's grid (`o` its origin, `n0` its
shape); its voxel centres are the grid's, so at scale 1 the fused image equals the source
image.

### Dtypes written

| Output | Input dtype | Dtype written | Rule |
| --- | --- | --- | --- |
| Stack and `max` projection, any scale factor and level | uint8, uint16, float32 | the input dtype | The library casts its float32 result back to the input dtype by truncation (the fusion measurement's `tables/overlap_dtype.csv`; `worker-notes.md` section 1). The reduced level of each FOV is the library's block mean truncated to the dtype (`msi_utils._mean_dtype`; `worker-notes.md` section 4). |
| Projection, `projection: sum` | uint8, uint16, float32 | float32 | Each FOV's sum is computed by `project_image` with `ProjectionConfig(method="sum", output_dtype="float32")` (`src/python/starfinder/preprocessing/projection.py`, the checked cast of `src/python/starfinder/io/conversion.py`), then fused as float32, which the library keeps (the fusion measurement's `tables/overlap_dtype.csv`, float32 rows). For integer inputs `fuse_images` first checks that the number of planes times the dtype's largest value is at most 2^24, so every sum is exact in float32, and raises `ValueError` naming the bound otherwise (uint8: up to 65,793 planes; uint16: up to 256). |
| Label export, any place and level | uint32 instance IDs | uint32 | No arithmetic on labels; IDs pass through float32 exactly because they are at most 2^24 ("Assembled label export"). |

% W-354 `tables/overlap_dtype.csv`

OME-TIFF stores each of these dtypes (uint8, uint16, float32 as `float`, uint32); the
library's OME-Zarr writer keeps the dtype (the fusion measurement's `tables/overlap_dtype.csv`,
`ome_zarr_dtype`). The dtype is recorded per file in `export.json`.

% W-354 `tables/overlap_dtype.csv`

### Memory

The fusion measurement recorded these peaks with every tile loaded into memory first (`tables/runs.csv`,
`worker-notes.md` section 8; four threads):

% W-354

* the default output of a whole sample: 629,552 to 1,492,944 KiB of peak RSS (0.60 to
  1.42 GiB, rounded; `peak_rss_kib` of the four `*-proj-s05-a2` rows);
* the LN 3D stack at scale 0.5: 12,946,284 KiB (12.35 GiB, rounded), the 7.2 GB of
  loaded tiles plus their reduced copies (`worker-notes.md` section 8).

This page does not promise bounded memory for a whole-sample stack. `fuse_images` reads each
source image as a lazy dask array from its file; that this keeps the peak of a stack well
below the loaded tiles is assumption A1. In one fusion each page of a source file is read
and decoded at most once: Starfinder's reading of the file gives the library one task per
page, as one chunk per page whatever strips the page is stored in (a Z plane; a file whose
pages are not its Z planes is one task, in the chunks of tifffile's Zarr view), and for the
projection one task per file that reads its pages and projects them; a task's result is
kept from the first band of the OME-TIFF writer that needs it until the last, then
released, so the stack holds a few pages of each FOV rather than whole tiles. No source
file is read before the fused array is computed, apart from its header; nothing is staged
or copied on disk; a source given as an array is used as given; the fusion and its
blending stay the library's. A later pass over the fused array (`pyramid_levels`, the
standalone OME-Zarr) reads the sources again. The OME-TIFF writer computes the lazy
fused array band by band (a row of 512-pixel tiles at a time) and writes the tiles of
that band (A8). The record states the measured peak RSS of every part.

### The fusion section of `export.json`

* `status`, and for `not_run`, `empty` and `failed` the reason;
* the placement: method, gauge, review state and approval, the SHA-256 of `fovs.csv` and
  `links.csv` as `placement.json` records them; the SHA-256 of the `assembly` section when
  an assembly was given;
* per FOV: `fov_id`, placement status, included or the reason it is not, origin, grid
  (shape, metadata, SHA-256), and per source image its path and SHA-256;
* the library: `multiview-stitcher` version, `fusion_func` (`weighted_average_fusion`),
  `interpolation_order` (1), `blending_widths`, the reduction route (`level1` or `none`),
  and whether the FOVs were passed as 2D;
* the config (`FusionConfig` fields), the fusion grid (`n0`, `s`, `o`, and the Z slab for
  one-plane FOVs), and per output file: path, SHA-256, format, image name,
  dimensionality, projection rule, factor `k`, shape, dtype, the output level's spacing
  and origin, the projection's Z slab, the resolution levels with their shapes, spacings
  and origins, tile or chunk shape and compression; a standalone OME-Zarr output is a folder,
  so its entry also lists `files`, the map from each file's path relative to the folder
  (with `/` separators) to that file's SHA-256, and its `sha256` is the SHA-256 of that
  map as the canonical JSON of a section ("Read-back checks", R1);
* `software` as in `run.json` ([checkpoints](../../docs/checkpoints.md)), wall and CPU time and peak RSS.

```json
{
  "format_version": 1, "stage": "export", "sample": "sample1", "frame_id": "sample:sample1",
  "complete": true,
  "parts": {"assemble": "written", "fuse": "written", "fuse_labels": "off", "export_spatialdata": "off"},
  "assembly": {"status": "written", "...": "the record of the assembly contract"},
  "fusion": {
    "status": "written", "assembly_sha256": "...",
    "placement": {"method": "multiview_stitcher", "gauge": "G1", "review": "not_needed",
                  "fovs_sha256": "...", "links_sha256": "..."},
    "library": {"multiview_stitcher": "0.1.62", "fusion_func": "weighted_average_fusion",
                "interpolation_order": 1, "blending_widths_um": {"z": 3, "y": 10, "x": 10},
                "route": "level1", "inputs_2d": false},
    "grid": {"shape_zyx": [67, 12332, 10882], "spacing_zyx": [0.346255, 0.194398, 0.194398],
             "origin_zyx": [0.0, -8.641221, 1.046148]},
    "outputs": [{"path": "reference_stain_projection.ome.tif", "sha256": "...", "format": "ome_tiff",
                 "image": "reference_stain", "dimensionality": "projection", "projection": "max",
                 "factor_yx": 2, "shape_zyx": [1, 6166, 5441], "dtype": "uint8",
                 "spacing_zyx": [23.199085, 0.388796, 0.388796],
                 "origin_zyx": [11.426415, -8.544022, 1.143347],
                 "levels": [], "tile_yx": [512, 512], "compression": "zlib"}],
    "software": {"starfinder": "...", "git_commit": "...", "packages": {}},
    "peak_rss_kib": 0, "wall_s": 0, "cpu_s": 0},
  "labels": {"status": "off"},
  "spatialdata": {"status": "off"}
}
```

The Y and X numbers are those of the fusion measurement's LN projection from the multiview-stitcher
placement, rounded to six decimals (`tables/ln_two_placements_grids.csv`, row `mvs`); the
Z numbers illustrate the layout and are not results.

% W-354

## The 2D projection

The default fused image, not written for a dataset that sets `dimensionality: stack`
("The fused image by dataset"). Its Y and X grid is the stack's at the same scale factor;
its one plane represents the whole Z extent.

**Route: projection first.** For each included FOV, its source image is projected along Z
with `project_image` and `ProjectionConfig(method=…, axis="z")`
(`src/python/starfinder/preprocessing/projection.py`, unchanged; for `sum` with
`output_dtype="float32"`), and the per-FOV projections are fused in 2D by the library,
with the FOVs' Y and X spacing and origins. A FOV's projection uses all its own planes; no
Z resampling happens. This is the fusion measurement's projection route, which cost 22.92 to 71.96 s of
wall time and at most 1,492,944 KiB (1.42 GiB, rounded) of peak RSS per whole sample,
against 241.65 s and 12,946,284 KiB (12.35 GiB, rounded) for the LN stack (the fusion measurement's
`tables/runs.csv`, `wall_s` and `peak_rss_kib` of the `*-s05-a2` rows).

% W-354

**Relation to the 3D stack.** The projection is not the maximum of the fused stack, and
no check compares their values. Their Y and X grids are equal at the same scale factor
(the fusion measurement's `tables/projection_fixture.csv` and `tables/projection_ln_region.csv`, column
`grid_yx_equal`). The fusion measurement recorded the differences:

% W-354 `tables/projection_fixture.csv`; W-354

| Input (fusion-measurement table) | Scale | Pixels that differ | Largest difference |
| --- | --- | --- | --- |
| Fixture, integer Z offsets, weighted average (`projection_fixture.csv`) | 1 | 2,134 of 57,600, all in overlaps | 94 |
| Fixture, integer Z offsets, `max_fusion` | 1 | 0 of 57,600 | 0 |
| Fixture, fractional Z offsets, weighted average | 1 | 41,563 of 57,600 | 94 |
| Fixture, any configuration | 0.5 | 13,297 to 14,129 of 14,400 | 18 to 75 |
| LN tiles 30, 31, 39, 40, weighted average (`projection_ln_region.csv`) | 1 | 5,984,543 of 8,105,360 (73.8 %, rounded) | 69 |
| LN tiles 30, 31, 39, 40, weighted average | 0.5 | 1,895,657 of 2,026,340 | 50 |

% W-354

They agree only with integer Z offsets, `max_fusion` and scale 1. Linear interpolation in
Z averages neighbouring planes at fractional Z offsets, the Z blending ramp weights
planes differently in 3D, and the maximum of block means is not the block mean of maxima
(the fusion measurement's `worker-notes.md` section 5).

% W-354 `worker-notes.md`

**Geometry.** The projection's one plane is the slab `[z_lo, z_hi)`, the Z extent of the
stack's fusion grid: `z_lo = o_z − s_z/2`, `z_hi = o_z + (n0_z − ½)·s_z`, with `o_z` and
`n0_z` computed by `fused_grid` with the library's rule of "The fusion grid" (the 2D
fusion itself records no Z). Its metadata has the fusion grid's Y and X at the output
level, and in Z spacing `S_z = z_hi − z_lo = n0_z·s_z` and origin
`O_z = (z_lo + z_hi)/2 = o_z + s_z·(n0_z − 1)/2`, one block over the whole Z extent.
`ImageMetadata.world_to_index` (`src/python/starfinder/image.py`, line 72) returns the
continuous index `j_z = (z − O_z)/S_z`, which lies in `[−½, ½)` for every z of the slab;
the voxel rule `v = floor(j + ½)` of "Sample position to image position" then gives voxel 0
for each. With `O_z = 1` and `S_z = 3`, z = 0 gives `j_z = −1/3` and voxel 0. For one-plane
FOVs `[z_lo, z_hi)` is
the slab of "The fusion grid", and `S_z = z_hi − z_lo`, `O_z = (z_lo + z_hi)/2` as there. This differs on purpose from
`ImageMetadata.projected`, which leaves the physical fields of a per-FOV projection
unknown; the per-FOV rule is not changed.

| Option | Projection rule | Cost | Consequence |
| --- | --- | --- | --- |
| **P1. Maximum, projection first (decided)** | `max` over Z of each FOV, fused in 2D | One read of each FOV, a 2D fusion (the fusion measurement: 22.92 to 71.96 s, at most 1,492,944 KiB) | Today's practice (`MAX_*.tif`); spots and nuclei at any depth stay visible; differs from the maximum of the stack as measured above. |
| P2. Sum | `sum` over Z of each FOV, written as float32 ("Dtypes written") | 4 bytes per pixel | Proportional to total signal; not the input dtype. |
| P3. One plane | Each FOV's plane nearest to a given sample z, fused in 2D | Reads one plane per FOV | Cheapest; loses everything out of that plane. Not offered as a key. |
| P4. Project the stack | `max` over Z of the fused 3D stack | The stack's cost (LN at scale 0.5: 241.65 s, 12.35 GiB rounded) | Equals the maximum of the stack by construction. Not offered as a key. |

% W-354

**Decided: P1.** It is the default decided in the first round and needs no 3D fusion;
no equality between the projection and the maximum of the stack is required, and their
difference is stated rather than hidden ("Choices decided at the gate", 4).

## The scale factor

**Meaning.** `scale_yx = 1/k` with `k` a whole number ≥ 1, applied to Y and X alike; Z is
kept as acquired, as today's `*_05` stacks keep it. The default is 0.5. A value whose
reciprocal is not a whole number within 1e-9 is refused by `FusionConfig`.

**How it is requested from the library: the `level1` route** (the fusion measurement's `worker-notes.md`
section 4, `tables/reduced.csv`, rows `level1`). For `k > 1`, each FOV image (or its
projection) becomes the library's two-level multiscale image
`msi_utils.get_msim_from_sim(sim, scale_factors=[{"y": k, "x": k}])`, and its level 1,
`msi_utils.get_sim_from_msim(msim, scale="scale1")`, is the input of `fuse`. That level is
the block mean of `k × k` voxels on the FOV's own grid, truncated to the dtype. The output
grid is passed as `output_stack_properties`:

% W-354 `worker-notes.md`

| File | Axis | Spacing `S_a` | Origin `O_a` | Size `N_a` |
| --- | --- | --- | --- | --- |
| Stack | Y, X | `k·s_a` | `o_a + s_a·(k − 1)/2` | `ceil(n0_a / k)` |
| Stack | Z | `s_z` | `o_z` | `n0_z` |
| Projection | Y, X | `k·s_a` | `o_a + s_a·(k − 1)/2` | `ceil(n0_a / k)` |
| Projection | Z | `n0_z·s_z` (the slab) | `o_z + s_z·(n0_z − 1)/2` | 1 |

For one-plane FOVs the Z rows of both files are the slab of "The fusion grid"
(`S_z = z_hi − z_lo`, `O_z = (z_lo + z_hi)/2`, one plane). The centre
of output voxel `w` is the centre of its `k × k` block of the fusion grid. On
LN sample1 at `k = 2` this gives 6,166 × 5,441 at 0.388796 µm with origin (−8.544022,
1.143347) µm, rounded to six decimals (the fusion measurement's `tables/ln_two_placements_grids.csv`, row
`mvs`: `origin_y_um` −8.544022432956568, `origin_x_um` 1.1433469582247893). In the fusion measurement's fixtures the
library's recorded geometry equals this table (`geometry_matches_formula` and
`geometry_matches_page` true), a checkerboard of 0 and 200 became 100 in both tiles
(`a_only_values`, `b_only_values`), and every single bright voxel stayed visible: 9 of 9
at 0.5 and at 0.25, with its maximum at most 0.29043 µm (`k = 2`) and 0.58486 µm
(`k = 4`) from the true position (`max_distance_max_um` 0.2904285101109904 and
0.5848581960387487, a distance in Z, Y and X), within `((k − 1)/(2k) + ½)·S` per axis
(`tables/position_summary.csv`, route `level1`).

% W-354 `tables/ln_two_placements_grids.csv`; W-354

This is not the block mean of a fused level 0: the mean is taken per FOV on its own grid,
truncated, and then resampled onto the output grid by the interpolation order, which
matters for FOVs at fractional positions. At `k = 1` the FOV images are the inputs and the
output grid is the fusion grid.

| Option | Route | Cost | Consequence |
| --- | --- | --- | --- |
| **R1. `level1` (decided)** | The library's block mean per FOV, then fused onto the output grid above | One reduced copy per FOV; LN default output 72 s, 0.78 GiB (the fusion measurement's `tables/runs.csv`) | Keeps every bright voxel (9 of 9); geometry nested in the fusion grid; values are means of per-FOV blocks, resampled. |
| R2. Fuse at scale 1, then a block mean in Starfinder | `fuse` at the fusion grid, then the mean of each `k × k` block of the fused image | The fusion of every level-0 voxel (not measured on whole samples) | The earlier page's exact semantics; a resampling step of Starfinder's own, not a fusion. |
| R3. `output_spacing` alone (sampling) | `fuse(..., output_spacing=k·s)` | Cheapest | Samples instead of averaging: at 0.5 only 3 (order 0) or 7 (order 1) of 9 single bright voxels remain; origin `o`, not nested (the fusion measurement's `tables/position_summary.csv`, `tables/reduced.csv`, rows `spacing`). The route of the label export ("Assembled label export"). |
| R4. Multiscale inputs | `fuse` of two-level images with `output_spacing` | As R1 | The library chooses the level per FOV from spacings derived from coordinates, and float error sends some FOVs to level 0 (LN: 8 of 64 FOVs at level 1); averaged and sampled FOVs mix (the fusion measurement's `worker-notes.md` section 4). Not offered. |

% W-354 `tables/runs.csv`; W-354 `tables/position_summary.csv`; W-354 `worker-notes.md`

**Decided at the gate: R1** ("Choices decided at the gate", 1).

| Option | Default scale factor | Default output, measured (the fusion measurement's `tables/runs.csv`) | Consequence |
| --- | --- | --- | --- |
| S1. 1 | `scale_yx: 1` | not measured on whole samples | Full detail; four times the bytes of S2. |
| **S2. 0.5 (decided)** | `scale_yx: 0.5` | LN 6,166 × 5,441, tissue-2D 11,291 × 9,845, cell-culture-3D 6,818 × 4,788, uint8 | Today's `*_05` practice; 0.388796 µm pixels for LN and culture, 0.18927 µm for tissue-2D (twice the spacing of the stitching input inventory). |
| S3. 0.25 | `scale_yx: 0.25` | not measured on whole samples | An overview; the label export's default. |

% W-354 `tables/runs.csv`; W-323

### Resolution levels inside a file

Each level `ℓ` reduces each axis `a` of the output level by its own factor `f_a^(ℓ)`, a
power of 2 that depends on the writer. On every axis, with `S_a`, `O_a` and `N_a` of the
output level:

* spacing `S_a^(ℓ) = f_a^(ℓ)·S_a` and origin `O_a^(ℓ) = O_a + S_a·(f_a^(ℓ) − 1)/2`, so an
  axis with `f_a^(ℓ) = 1` keeps the output level's spacing and origin;
* the shape is recorded per level.

The writers:

* **OME-TIFF** (`pyramid_levels: L`, default 0): `L` SubIFD levels, each from the fused
  output level by the library's own multiscale, `msi_utils.get_msim_from_sim(fused,
  scale_factors=[{"y": 2, "x": 2}] * L)`, the same block mean truncated to the dtype as
  the `level1` route. So `f_y^(ℓ) = f_x^(ℓ) = 2^ℓ` and `f_z^(ℓ) = 1` for a stack (a
  projection has no Z level axis). That the library's multiscale of a fused image has
  these shapes and this geometry, also for odd sizes, is assumption A7. Starfinder
  computes no level of its own.
* **OME-Zarr** (the library's writer): the writer decides per axis. Going from level `ℓ` to
  `ℓ + 1`, an axis of size `n` is halved to `floor(n/2)`, doubling its factor, when
  `floor(n/2) > 100`, and kept otherwise; a further level exists while at least one axis
  is halved (the fusion measurement's `worker-notes.md` section 7: `msi_utils.calc_resolution_levels`,
  `min_shape=100`). Z is included for a stack. The fusion measurement's `readback/readback.json`:
  * `fixture_stack_ngff04` (21 × 240 × 240 at (0.346255, 0.194398, 0.194398) µm, origin
    (1.3, 10.0, −5.0) µm): level 1 is 21 × 120 × 120 with Z kept (`f_z^(1) = 1`: spacing
    0.346255, origin 1.3) and Y and X halved (spacing 0.388796, origins 10.097199 and
    −4.902801 = 10.0 and −5.0 plus 0.194398/2); no level 2, because 120 and 21 cannot be
    halved above 100;
  * `fixture_projection_ngff04`: the same in Y and X;
  * `ln_mvs_projection_ngff04` (6,166 × 5,441 at 0.388796 µm, origin (−8.544022,
    1.143347) µm): levels 1 to 5 halve Y and X, 3,083 × 2,720 down to 192 × 170, with
    spacing `2^ℓ·0.388796` and origin `O + 0.388796·(2^ℓ − 1)/2` (level 1: −8.349624,
    1.337745; level 5: −2.517684, 7.169685, rounded to six decimals); an odd size loses its
    last voxel (5,441 → 2,720).
* **SpatialData store**: one level by default ("SpatialData export"); spatialdata's own
  levels (choice 10) have per-level scales from shape ratios, and they are not covered by
  the formula above.

% W-354 `worker-notes.md`; W-354 `readback/readback.json`

## The 3D stack

Written when the dataset's configuration sets `dimensionality: stack` (the stack alone,
for dense 3D tissue such as LN sample1) or `both` ("The fused image by dataset"). The
same conversion with dims
`["z", "y", "x"]`, the fusion grid in Z, Y and X, and the `level1` route in Y and X only;
Z keeps the fusion grid's `s_z`, `o_z` and `n0_z`. FOVs at fractional Z offsets are
interpolated in Z at order 1 (the fusion measurement's `worker-notes.md` section 5), and voxels no FOV covers,
whole planes included, hold 0 (`tables/overlap_dtype.csv`). When every included FOV has
Z = 1 the stack is the one plane of "The fusion grid", fused from the same 2D inputs as the
projection, so it equals the projection's values under `projection: max`.

% W-354 `worker-notes.md`

Its cost (the fusion measurement's `tables/runs.csv`, `ln-mvs-stack-s05-a2`): LN sample1 at scale 0.5, 67 ×
6,166 × 5,441 uint8, 241.65 s wall, 532.74 s CPU, 12,946,284 KiB (12.35 GiB, rounded) peak
RSS, a 2,247,808,158-byte
TIFF from the library's uncompressed writer. A cell-culture-3D stack was not run (A4).

% W-354 `tables/runs.csv`

## Sample position to image position

This is the condition that ties every fused file to the sample tables.

For a file with recorded spacing `S` and origin `O` (from the file or its record; the
table of "The scale factor" gives them per axis), a sample position `q = (q_z, q_y, q_x)`
in µm has the continuous index

```
j = (q − O) ⊘ S
```

and lies in voxel `v = floor(j + ½)` per axis (the voxel whose half-open support
`[O + (v − ½)·S, O + (v + ½)·S)` contains `q`). Below, `j⁰` and `v⁰` are the indices on
the fusion grid (`S = s`, `O = o`), and `k` is a file's factor per axis (`k_z = n0_z` for
the projection's Z and `k_z = 1` for the stack's, both for multi-plane FOVs; for one-plane
FOVs Z is the slab of "The fusion grid" in both files, and conditions 1 to 3 treat it as
stated there).

A test can evaluate, for a set of sample positions inside the fusion grid's extent (the
FOV origins, fractional positions inside each FOV, the molecule positions `sample_z/y/x`
of the sample molecule table, and positions next to voxel boundaries):

1. **Recorded geometry, per file and per axis.** The `S` and `O` read back from each file
   equal those of `export.json` and the formulas of "The scale factor": within 1e-9 µm for
   Starfinder's OME-TIFF, which writes the recorded float64 values; within 1e-6 µm for the
   library's OME-Zarr, the agreement the fusion measurement found (`readback/readback.csv`,
   `origin_matches_grid`); for resolution level `ℓ`, on every axis `a`,
   `S_a^(ℓ) = f_a^(ℓ)·S_a` and `O_a^(ℓ) = O_a + S_a·(f_a^(ℓ) − 1)/2`, with the per-axis
   factor `f_a^(ℓ)` of "Resolution levels inside a file" (1 on an axis the level keeps).
   For one-plane FOVs, `S_z = z_hi − z_lo` and `O_z = (z_lo + z_hi)/2` in both files.
2. **Projection and stack agree in Y and X.** At the same `k`, the Y and X components of
   `j` and `v` are equal in the projection and in the stack (the fusion measurement's `grid_yx_equal`). In Z
   the projection's `v_z` is 0 for every `q_z` in `[z_lo, z_hi)`, and so is the stack's for
   one-plane FOVs.
3. **Any scale factor.** In Y and X, and in Z for the projection of multi-plane FOVs
   (`k = n0_z`, so `v_z = 0`), `j = (j⁰ − (k − 1)/2) / k` within 1e-9 and
   `v = floor(v⁰ / k)` exactly, also when `k` does not divide `n0` (the last block is
   partial); in Z for the stack of multi-plane FOVs `k_z = 1`, so `j_z = j⁰_z`. For
   resolution level `ℓ` the same holds per axis with `k_a·f_a^(ℓ)` in place of `k_a`. For
   one-plane FOVs Z is condition 2's slab, where no fusion-grid index `j⁰_z` exists. The
   floor identities are evaluated at positions at least 1e-6 voxel from a voxel boundary
   of the fusion grid.
4. **A bright voxel lands where it belongs.** On the fixture of the fusion measurement's position
   measurement (`tables/position_summary.csv`, `worker-notes.md` section 2): 3 × 3 FOVs of
   16 × 128 × 128 uint16 at spacing (0.346255, 0.194398, 0.194398) µm, origins jittered by
   up to 3 px in Y and X around a step of 115.2 px and by up to 2 slices in Z (seed 354),
   and one voxel of 1000 per FOV in the part only that FOV covers. The contract accepts it
   (one dtype, one spacing, finite origins). With `q*` the bright voxel's sample position:
   * at scale 1, the fused maximum near `q*` lies on the output voxel nearest `q*`, and the
     intensity centroid of its 3 × 3 × 3 neighbourhood lies within 0.001 µm of `q*` (the fusion measurement:
     at most 0.0008183085523003642 µm, `max_distance_centroid_um` of the order 1, scale 1
     row);
   * at `k = 2` and `k = 4` (`level1`), each of the nine bright voxels gives a local maximum
     above 0 and at most `floor(1000 / k²)`, within `((k − 1)/(2k) + ½)·S_a` of `q*` on each
     axis `a` of Y and X. The fusion measurement: 9 of 9 found at both factors; the largest distance in Z, Y
     and X, which bounds each axis's, was 0.2904285101109904 µm at `k = 2`, against the
     bound 0.291597 µm (0.75 × 0.388796), and 0.5848581960387487 µm at `k = 4`, against
     0.680393 µm (0.875 × 0.777592); the peaks at order 1 were 78 to 166 and 24 to 48.
5. **SpatialData.** In the store, the image element's transform to the sample coordinate
   system maps the intrinsic coordinate `w + ½` of voxel `w` (spatialdata's voxel `w`
   covers `[w, w + 1)`, `spatialdata/_core/data_extent.py` line 90 of 0.6.1) to
   `O + w ⊙ S` on the element's axes (Y and X for a projection and for the stack of
   one-plane FOVs; Z, Y and X for the stack of multi-plane FOVs), so
   a point at sample position `q` lands in intrinsic voxel `floor(j + ½)`, the `v` of
   condition 1 (the fusion measurement's `readback/spatialdata_read.json`: scale equals the spacing, and the
   translation plus `S/2` equals the library's origin). On the fixture of condition 4, the
   intrinsic voxel of a point at `q*` holds the file's value there.

% W-354; W-354 `grid_yx_equal`; W-354 `readback/spatialdata_read.json`

## File layouts

### What the library writes

From the fusion measurement's `worker-notes.md` section 7, `readback/readback.csv` and
`readback/readback.json`:

% W-354 `worker-notes.md`

* **TIFF**, `multiview_stitcher.io.save_sim_as_tif` (it needs a dask-backed image): classic
  TIFF, not BigTIFF; an ImageJ hyperstack without OME-XML; uncompressed, contiguous, one
  strip per plane, not tiled; one resolution level, no SubIFDs; X and Y calibration as
  `XResolution` / `YResolution` with ImageJ `unit=um`, equal to the fused spacing within
  1e-10; no Z spacing (also in the 67-plane LN stack) and no origin.
* **OME-Zarr**, `ngff_utils.write_sim_to_ome_zarr`: NGFF 0.4 on Zarr format 2 by default,
  NGFF 0.5 on Zarr format 3 with `ngff_version="0.5"`; axes `t, c, (z), y, x`, the space
  axes with unit `micrometer`; per level `scale` and `translation` (the voxel-centre
  origin) equal to the fused grid within 1e-6 µm; levels as in "Resolution levels inside
  a file"; Zstd level 0; chunks equal to the dask chunks of the image passed; `omero`
  channel metadata; no `coordinateTransformations` on the multiscale itself.
* **spatialdata 0.6.1 cannot read the library's OME-Zarr as an image element**: NGFF 0.4
  fails on the missing multiscale `coordinateTransformations`, then on the `t` axis without
  a unit, and NGFF 0.5 fails with `KeyError: 'multiscales'` (`readback/spatialdata_read.json`).

### The earlier layouts against the library's files

Each requirement of the earlier OME-TIFF and OME-Zarr layouts (this page at `6b622e9`),
with whether the library's file meets it:

| Requirement of the earlier layout | Library's file | Outcome |
| --- | --- | --- |
| OME-TIFF: BigTIFF | classic TIFF | Needs Starfinder's step: the OME-TIFF writer below. |
| OME-TIFF: OME-XML with SizeZ, SizeC, SizeT and the pixel type | ImageJ description only | Needs the writer. |
| OME-TIFF: 512 × 512 tiles, zlib | strips, uncompressed | Needs the writer. |
| OME-TIFF: SubIFD levels on request | one level | Needs the writer; the level values come from the library's multiscale (A7). |
| OME-TIFF: `PhysicalSizeX/Y` | `XResolution` / `YResolution` with unit um | Needs the writer (the library's values are correct, in another tag). |
| OME-TIFF: `PhysicalSizeZ`, plane positions (the origin) | absent | Needs the writer. |
| OME-TIFF: the `ImageMetadata` annotation that `load_volume` restores, and the `starfinder.sample_output` annotation | absent | Needs the writer. |
| OME-Zarr: NGFF 0.4 on Zarr format 2 | the default | Holds; reconsidered below for Python 3.11 or later. |
| OME-Zarr: axes `c, (z), y, x` | `t, c, (z), y, x` | Dropped: the `t` axis is accepted. |
| OME-Zarr: per-level `scale` and voxel-centre `translation` | yes, within 1e-6 µm | Holds. |
| OME-Zarr: levels halving Y and X until both are at most 1024, padded to exact halves | halving every axis while above 100, unpadded | Dropped: the library's levels are accepted and recorded. |
| OME-Zarr: Blosc `lz4` | Zstd level 0 | Dropped: Zstd is lossless and accepted. |
| OME-Zarr: chunks `(1, 1024, 1024)` and `(1, 16, 512, 512)` | the dask chunks passed | Starfinder rechunks the lazy array before the writer (a dask call, not a writer; A12). |
| OME-Zarr: group attributes `starfinder` with the `ImageMetadata` | absent | Dropped from the file: recorded in `export.json` instead, so no file of the library is edited. |
| OME-Zarr readable by spatialdata as an image element | no | Dropped: the store is written by spatialdata itself ("SpatialData export"). |

### OME-TIFF layout

**Decided at the gate (T1): Starfinder writes the OME-TIFF with tifffile from the image
the library fused.** This is a file writer, not a fusion: it writes the array `fuse`
returned, value for value, and computes no voxel.

* **Container**: BigTIFF always, written with `tifffile` (a base dependency). A file above
  4 GiB is assumption A3.
* **Series and axes**: one image per file, one channel; axes `YX` stored with SizeZ 1 for
  the projection and for the stack of one-plane FOVs, and `ZYX` for the stack of
  multi-plane FOVs, each Z plane of a stack one page of the series
  (so one plane is read alone, "The fused image by dataset"); the OME-XML declares SizeZ,
  SizeC 1, SizeT 1 and the pixel type of the file's dtype ("Dtypes written").
* **Tiling**: 512 × 512 tiles, written band by band from the lazy fused array (A8);
  compression `zlib` (lossless; tifffile encodes and decodes it with its own fallback when
  `imagecodecs` is absent, `tifffile.py` lines 15660 to 15667 of 2026.1.28).
* **Resolution levels**: `pyramid_levels` (default 0) SubIFD levels of "Resolution levels
  inside a file", written after the main series.
* **Calibration and frame**: `PhysicalSizeX`, `PhysicalSizeY`, `PhysicalSizeZ` = `S` of the
  output level with unit `µm` (for the projection, and for every file of one-plane FOVs,
  `PhysicalSizeZ` is the slab thickness `z_hi − z_lo`); for plane `i` the OME `Plane` positions `PositionZ = O_z + i·S_z`,
  `PositionY = O_y`, `PositionX = O_x` in µm; the `ImageMetadata` of the output level as
  JSON in a `CommentAnnotation` of namespace `starfinder.image.ImageMetadata`, the
  convention of `save_volume`, so `load_volume` restores the metadata; a second
  annotation, namespace `starfinder.sample_output`, with the sample, image name,
  dimensionality, projection rule, factor and the SHA-256 of the placement tables.
* **UUID**: the OME-XML `UUID` of a written file is derived from its content (the shapes,
  dtype, layout, metadata and tiles written), not drawn at random or from the time, so
  that equal content gives equal bytes.
* **File names**: `<image>_projection.ome.tif`, `<image>_stack.ome.tif`,
  `cell_labels.ome.tif`.

**Layout checks** (`check_export`, with `tifffile` and `load_volume`), for each OME-TIFF:

1. The file opens as BigTIFF and OME (`is_bigtiff`, `is_ome`), with one series whose axes
   are `YX` (projection, or a one-plane stack) or `ZYX` (stack) and one channel.
2. Shape and dtype equal the record's; the OME pixel type is the dtype's; the tile shape is
   512 × 512 and the compression `zlib`; the number and shapes of SubIFD levels equal the
   record's.
3. `PhysicalSizeX`, `PhysicalSizeY` and `PhysicalSizeZ` equal the record's `S` within
   1e-9 µm with unit `µm`; the `Plane` positions of plane `i` equal
   `(O_z + i·S_z, O_y, O_x)`; the `ImageMetadata` restored by `load_volume` has that
   spacing and origin.

**The alternative, an option not taken: the library's own TIFF as the default file**
(choice 5 of the fusion measurement, option b). Choosing it would have changed this section only:

% W-354 choice 5

* `fuse_images` writes each fused image with `multiview_stitcher.io.save_sim_as_tif` from the
  dask-backed fused image, as `<image>_projection.tif`, `<image>_stack.tif` and
  `cell_labels.tif`.
* The file is what "What the library writes" lists: a classic ImageJ TIFF, uncompressed
  strips, one level, X and Y calibration only. It lacks BigTIFF (a file above 4 GiB is
  assumption A3), OME-XML, Z spacing, the origin, tiles, compression, resolution levels
  and the `ImageMetadata` annotation, so `load_volume` cannot restore the sample frame
  from it and `pyramid_levels` must be 0.
* The missing calibration and origin are recorded in `export.json` only: per file the
  spacing and origin of the output level, the projection's Z slab and the `ImageMetadata`
  as JSON. A reader takes the frame from there; the file alone places the image nowhere.
* Layout checks: the file opens as an ImageJ TIFF with the series axes `YX` or `ZYX`;
  shape and dtype equal the record's; `1/XResolution` and `1/YResolution` equal the
  record's `S_x` and `S_y` within 1e-10 µm with ImageJ unit `um` (the fusion measurement's
  `readback/readback.csv`); the record holds the origin and Z spacing.

% W-354

| Option | Layout | Cost | Consequence |
| --- | --- | --- | --- |
| **T1. Starfinder's tifffile writer of the fused array (decided)** | as above | A writer of BigTIFF, tiles, zlib, OME-XML and SubIFDs on tifffile (A8) | The default file carries its own calibration and origin, region reads touch only their tiles, and `load_volume` reads it. |
| T2. The library's TIFF | the alternative above | No writer of Starfinder's own | Frame only in `export.json`; no Z spacing, origin or levels; classic TIFF. |
| T3. One multichannel file per sample | All `images` as C of one OME-TIFF | One file | A channel added later rewrites the file. |

**Decided at the gate: T1** ("Choices decided at the gate", 5): the decision names
OME-TIFF as the default format, and the library's file has no Z spacing, no origin and no
OME-XML.

### OME-Zarr layout

The standalone OME-Zarr (`ome_zarr: true` without the SpatialData export) is written by
the library's `ngff_utils.write_sim_to_ome_zarr` from the fused image, rechunked to
`(1, 1, 1024, 1024)` for a projection and `(1, 1, 16, 512, 512)` for a stack (`t, c, z,
y, x`; A12). Starfinder writes no OME-Zarr of its own. Its axes, units, levels, codec and
per-level geometry are the library's ("What the library writes"); its dtype is the fused
image's.

**Python 3.11 or later.** The earlier choice of NGFF 0.4 on Zarr format 2 rested on Python
3.10, where the extra resolved zarr 2.18.3. With `requires-python >= 3.11` the locked
zarr 3.1.5 writes both formats, and the fusion measurement wrote and read back both with the library's
writer (`readback/readback.csv`, `fixture_*_ngff04` and `fixture_*_ngff05`). The store is
written by spatialdata 0.6.1, whose default is Zarr format 3 with raster groups of version
string `0.5-dev-spatialdata`; the fusion measurement wrote and read back stores in that default
(`readback/spatialdata_write_ln_stack_*.json`).

% W-354

| Option | Standalone file | Store | Consequence |
| --- | --- | --- | --- |
| **Z2. NGFF 0.5 on Zarr format 3 (decided)** | the library's writer with `ngff_version="0.5"` | spatialdata's default, no format argument | Both the file and the store in the form the fusion measurement wrote and read back; one Zarr format in the folder; NGFF 0.5 has fewer readers today (viewer qualification is deferred). |
| Z1. NGFF 0.4 on Zarr format 2 | the library's default | `SpatialDataContainerFormatV01` (raster format 0.2, version string `0.4-dev-spatialdata`) | The earlier proposal and the reference-branch format; the store in this format was not written by the fusion measurement (A10). |
| Z3. No resolution levels | the output level only | the store's one level | About a third less written; region reads at full detail only. |

% W-354

**Decided at the gate: Z2** ("Choices decided at the gate", 7).

**Layout checks** (`check_export`, with `zarr`), for each standalone OME-Zarr:

1. The group's Zarr format and NGFF version are the recorded ones (format 3 and 0.5 under
   Z2); the axes are `t, c, (z), y, x` with unit `micrometer` on the space axes; one
   dataset per recorded level; each dataset's `scale` and `translation` equal the record's
   `S^(ℓ)` and `O^(ℓ)` per axis within 1e-6 µm.
2. Each level's shape, dtype, chunks and codec equal the record's.
3. When the output level has more than one chunk, a region read inside one chunk touches
   that chunk only (chunk keys counted).

## SpatialData export

On request (`export_spatialdata: true`). It is an export of the sample tables and the
fused and label images, **not a second source**: every value comes from `tables/`,
`sample.h5ad` and the image files of `export/<sample>/`; nothing is filtered, reassigned
or recounted, and the sample tables stay the reference.

**Inputs**: the `AssemblyResult` (from `load_assembly`, which checks every SHA-256), the
`FusionRecord` and the `LabelRecord` of the parts that were written.

**Coordinate system**: one, named after the sample; its axes are the sample frame's, in
µm. spatialdata does not check coordinate-system names, but the export applies its element
naming rule to it (letters, digits, `_`, `-` and `.`; `spatialdata/_core/validation.py`
line 109 of 0.6.1) and refuses other sample names. `frame_id` `sample:<sample>` is recorded
in the `spatialdata` section and in the store's attributes.

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
and the keys of `sample.h5ad` do ([assembly specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/assembly.md)), and the export refuses a table
that does not, naming the key.

**Elements** (`export/<sample>/sample.zarr`), each written only when its population has a
row or its image exists ("Empty populations"):

| Element | Content | Transform to the sample coordinate system |
| --- | --- | --- |
| points `molecules` | Every row of the sample molecule table whose `sample_z/y/x` are finite (every molecule of a placed FOV, retained or excluded), with every other column; the per-FOV index columns `z`, `y`, `x` are first renamed `fov_index_z/y/x`; `PointsModel.parse(…, coordinates={"x": "sample_x", "y": "sample_y", "z": "sample_z"})` then turns `sample_x/y/z` into the coordinate columns `x`, `y`, `z` (float64, µm) and does not keep them under their old names (`spatialdata/models/models.py` lines 831 to 847 of 0.6.1); `cell_instance_id` (pandas `UInt32`, null without a cell) is added. No feature or instance key is declared, so the model adds only its `transform` attribute (lines 865 to 895). | Identity |
| shapes `cell_centroids` | One circle per row of the sample cell table with a finite ownership centroid: centre `(sample_ownership_centroid_x, sample_ownership_centroid_y)`, column `radius` the equivalent radius of the original territory (`size_voxels` × voxel volume as a sphere for `volume`, × pixel area as a disc for `plane`), index `instance_id`. Radii must be positive (`ShapesModel`, lines 458 to 463); a cell with `size_voxels` 0 makes the export raise. | Identity |
| table `cells` | The sample cell table for the same rows, every column as `obs`, no `X` (zero variables); added `obs` columns `region` (categorical, every value `cell_centroids`) and `instance_id` (numpy `uint32`, an instance-key dtype the model accepts, lines 1033 to 1047); parsed with `TableModel.parse(region="cell_centroids", region_key="region", instance_key="instance_id")` | annotates `cell_centroids` |
| table `expression` | `sample.h5ad` with every part kept: `X`, `layers`, `var`, `obs`, `obsm` and every key of `uns`; added `obs` columns `region` and `instance_id` as for `cells`; parsed as `cells` | annotates `cell_centroids` |
| tables `cells_unplaced`, `molecules_unplaced` | The rows of the cell and molecule tables without finite sample coordinates (status `fov_not_placed`), every column as `obs`, zero variables, plus `instance_id` (cells) or `cell_instance_id` (molecules) as plain columns; parsed without region, region key or instance key, so the model adds nothing (lines 1131 to 1135) | none |
| image `<image>_projection` (and `<image>_stack`) | The fused image, one element per image name and dimensionality: the OME-TIFF's output level read lazily with tifffile's `aszarr` (`dask.array.from_zarr`), given a channel axis of length 1, parsed as `Image2DModel` (`c, y, x`) for the projection and for the stack of one-plane FOVs (axes `YX` in its file), or `Image3DModel` (`c, z, y, x`) for the stack of multi-plane FOVs, one level, in the file's dtype | `Sequence([Scale(S), Translation(O − S/2)])` on its axes (Y and X for a 2D element; Z, Y and X for a 3D one): spatialdata's voxel `w` covers `[w, w + 1)`, so its centre `w + ½` goes to `O + w ⊙ S` (the fusion measurement's `readback/spatialdata_read.json`, the route that works) |
| labels `cell_labels` | The label export ("Assembled label export"), read from `cell_labels.ome.tif` likewise and parsed as `Labels2DModel` or `Labels3DModel`, uint32 (A9) | as for the image, with the label file's `S` and `O` |

% §2.9; W-354 `readback/spatialdata_read.json`

**Store format and levels.** spatialdata 0.6.1's default format (container format 0.2,
Zarr format 3, raster version string `0.5-dev-spatialdata`) under Z2, or
`SpatialDataContainerFormatV01` under Z1 ("OME-Zarr layout"). The image and labels
elements have one level: the fusion measurement wrote the LN stack (67 × 6,166 × 5,441 uint8,
2,247,796,802 bytes, 2.09 GiB rounded) as one level read through `aszarr` at a peak RSS of
1,662,724 KiB (1.59 GiB, rounded) in 156.48 s (`tables/runs.csv`,
`sd-write-ln-stack-single-aszarr`), while `scale_factors=[2, 2]` peaked at 3,360,240 KiB
(3.20 GiB, rounded), also reduced Z (67 → 33 → 16) and gave shape-ratio scales of 2.0303
in Z and 2.0004 in X (67/33 and 5,441/2,720, rounded; `worker-notes.md` section 7)
(`readback/spatialdata_write_ln_stack_single_aszarr.json`,
`readback/spatialdata_write_ln_stack_multiscale_aszarr.json`). One level in the store is
decided ("Choices decided at the gate", 10).

% W-354

**Metadata spatialdata adds.** `TableModel.parse` of an annotating table writes
`uns["spatialdata_attrs"] = {"region": "cell_centroids", "region_key": "region",
"instance_key": "instance_id"}` (`spatialdata/models/models.py` lines 1146 to 1164 of
0.6.1) and refuses a table that already has the key (lines 1138 to 1142); the region column must be
categorical (lines 1083 to 1086). A table parsed without these arguments and without the
key is returned unchanged (lines 1131 to 1135). The points and raster models add their
`transform` attributes. This metadata is the export's, not the source's; the round-trip
check tests it separately from the source content ("Read-back checks").

**A table that already has `uns["spatialdata_attrs"]`** (the table `expression` keeps every
`uns` key of `sample.h5ad`, so it is one when that file holds the key and the element is
written, "Empty populations") is refused by `TableModel.parse` as above, and the export
does not work around it: `export_spatialdata` raises the library's error, neither
overwrites nor drops the source key, and writes no store, leaving no `sample.zarr` and no
staging folder, as check SD4 of
[stitching and assembly validation design](https://github.com/wanglab-broad/starfinder/blob/dev/design/validation/stitching-assembly.md) asserts. Inside `Dataset.export` the same error is an
error inside the part `export_spatialdata`, and the rule of "When a part cannot run"
applies to it.

**Empty populations.** The pinned library rejects an empty shapes element (`ShapesModel`,
`spatialdata/models/models.py` lines 449 to 450 of 0.6.1) and an annotating table whose
declared region does not occur among its region values, which is every annotating table
without rows (lines 1048 to 1051). The rule is therefore: **an element whose population has
no row, or whose image does not exist, is left out of the store**, and the `spatialdata`
section lists every element the configuration asks for, each with `status` `written` or
`omitted`, its row count or shape, and for an omitted element a `reason`: `empty` (no row),
`no_cells` (a `molecules_only` sample has no territories), `no_included_fov`,
`not_requested` (`fuse: false`, `fuse_labels: false`) or `part_not_run` (the part that
makes it could not run, "When a part cannot run"). A name that the store lacks and the
section does not list as omitted is missing, and the read-back check fails on it. The
export relies on no library behavior for empty elements, because it writes none. When
every element is omitted, no `sample.zarr` is written and the section records
`store: null` with the reason.

| Case | Written | Left out (reason) |
| --- | --- | --- |
| Usual sample | every element with rows; the images | `cells_unplaced`, `molecules_unplaced` when every FOV is placed (`empty`, the common case) |
| No cell with a finite ownership centroid (no cell, or only cells of FOVs without an origin) | `molecules` if any row; `cells_unplaced` if any row; images | `cell_centroids`, `cells`, `expression` (`empty`); `cell_labels` (`empty` when the label part wrote no ID) |
| Cells, but no counted cell | `cell_centroids`, `cells`, the rest as usual | `expression` (`empty`); `sample.h5ad` has no row ([assembly specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/assembly.md), "Sample-level AnnData") |
| `molecules_only` sample | `molecules` if any row, `molecules_unplaced` if any row, images | `cell_centroids`, `cells`, `expression`, `cells_unplaced` (`empty`); `cell_labels` (`no_cells`) |
| No molecule with finite sample coordinates | the rest as usual | `molecules` (`empty`) |
| No included FOV | each table, points or shapes element that has rows (usually only `cells_unplaced` and `molecules_unplaced`) | image elements and `cell_labels` (`no_included_fov`); every table, points or shapes element without rows (`empty`) |
| `fuse: false` or `fuse_labels: false` | the tables, points and shapes as above | image elements or `cell_labels` (`not_requested`) |
| `fuse` or `fuse_labels` not run | the tables, points and shapes as above | image elements or `cell_labels` (`part_not_run`) |

In the "No included FOV" row, a placed FOV without a source image (a `placeholder`) still
has sample coordinates, so its observations, if it has any, go to `molecules`, `cells` and
`cell_centroids` as usual; only rows without sample coordinates go to the `_unplaced`
tables.

**Image dimensionality in the store.**

| Option | Store image | Cost | Consequence |
| --- | --- | --- | --- |
| **D1. What the fuse part writes (recommended)** | The projection by default; the stack when `dimensionality` is `stack`; both with `both` | Nothing beyond the fused images already requested | The default store is small; points keep their z, the projection's transform maps Y and X; a user who wants 3D in the store asks for the stack once. |
| D2. Always the 3D stack | The stack, whatever the fuse part writes | The stack's fusion for every export (LN at 0.5: 241.65 s, 12.35 GiB rounded) | 3D points and image agree in Z; every export pays for 3D. |
| D3. Always the projection | The projection only | Smallest | A requested stack never reaches the store. |

**Recommendation: D1.** One setting decides the image everywhere, and the default export
stays small.

**Written once, not duplicated.** With `export_spatialdata: true` the OME-Zarr image
exists only as the store's image element: `fuse_images` writes no standalone `.ome.zarr`, and the
store holds one image element per image name and dimensionality. Its values are read from
the OME-TIFF, so the two are equal. The section keeps the image element's path and the
SHA-256 of its array metadata.

**Identities and source lookup.** Molecules keep `(spot_namespace, spot_id)` and `fov_id`;
cells keep `(cell_namespace, cell_id)` and `obs_name` ([assembly specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/assembly.md),
"Sample-level identities"). `instance_id` is `1..N` over the rows of the sample cell table
in its order (uint32; more than 2³² − 1 rows is an error); `cell_instance_id` of a molecule
is the `instance_id` of its `(cell_namespace, cell_id)`. `lookup_source(folder, spot=…)` or
`(cell=…)` reads `export.json`, verifies each section hash that `export.json` records and
the files of the assembly, finds the row and returns its FOV, the FOV's assignment folder,
its `assignment.json` SHA-256 and the row's position in the FOV's own table, from the `assembly`
section; for a cell of the route after fusion, its FOV, no assignment folder, the row's
position in the sample cell table, and `label_file` (the cell run's label file) and
`label_value` (its `cell_id`) ([sample-segmentation-contract](../../docs/sample-segmentation-contract.md), "Readers that change").
It does not verify the files of the store, which `check_export` verifies.

% §2.9

**What it takes from the reference-branch exporter** ([sample-output-baseline](../baselines/sample-output.md)): the
store-once rule; writing into a sibling staging folder and renaming it into place, so a
failed export publishes nothing; the SHA-256 manifest of every component and its
verification on read; key validation (non-null, unique, string namespaces); the explicit
missingness of the reference-branch specification (a molecule without a cell keeps its `assignment_status`, never
a guessed cell). It does not take `prepare_rasters`, the affine source mappings, the
`MoleculeIndex` input, the rewrite of NGFF transforms inside the store, the pinned Zarr 2
environment or the napari adapter.

% W-168; §2.9

**The `spatialdata` section of `export.json`**: status and reason; coordinate system name;
the SHA-256 of the `assembly`, `fusion` and `labels` sections it read; the spatialdata
version and the formats written; `store` (the store path, or `null` when every element is
omitted); per element its name, kind, `status` (`written` or `omitted`) and for an omitted
element its `reason`, and for a written element its path, row count or shape, transform
and, for tables, the `spatialdata_attrs` written and the dtype of every column after the
round trip; the counts of "Read-back checks" (molecules by `assignment_status` and by
`assembly_status` and reason, cells by `assembly_status`, counted cells); the column
renames; and the SHA-256 of every file of the store (the store's entry in `outputs` lists
`files` and `sha256` by the rule of a standalone OME-Zarr in "The fusion section of
`export.json`").

## Assembled label export

Part of the first delivery (`fuse_labels`), at a default scale factor of 0.25 in Y and X.
The label image is an inspection product: it shows the retained cells of the sample, each
drawn whole as its FOV segmented it (on the route after fusion, each drawn as the sample
segmentation holds it: [sample-segmentation-contract](../../docs/sample-segmentation-contract.md), "The label images"), placed
without blending.

**Inputs**: the placement; the assembly's sample cell table, whose `assembly_status` says
which cells are retained and whose row order gives each cell's `instance_id`; and, per FOV
with at least one retained cell, the territory label image that assign sampled (the
expanded one when assign expanded, as the cell table's `ownership_centroid` records), at
the path the assembly record holds beside its SHA-256 ([assembly specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/assembly.md), "What
assemble records"); when that path is null (`conflicts=False` or an assignment given in
memory), the file the assignment loader resolves from the recorded assignment folder, with the
SHA-256 the assignment record holds. A FOV without a retained cell is not read; a `placeholder`
is one, because it owns no point and its cells are excluded. A `molecules_only` sample has
no territories: the part writes no label image and records `empty`, reason `no_cells`. A
`cells` sample without a retained cell gets a label image of zeros. Without an included
FOV it records `empty`, reason `no_included_fov`. On the route after fusion the input is
the sample's territories in place of each FOV's: the cell run's label image, or the
expanded territories `territories.ome.tif` when assign expanded, with each `cell_id`
mapped to its `instance_id` as the one view of the same library call; the `labels`
section records the rule `segmentation` and `segmentation_files`
([sample-segmentation-contract](../../docs/sample-segmentation-contract.md), "The label images").

% §2.9

**Values.** The value of a voxel is the `instance_id` ("SpatialData export", "Identities")
of the retained cell drawn there; 0 is background, an uncovered voxel, or a voxel where no
FOV draws a retained cell. A cell is retained when the cell table says
`assembly_status == "retained"`, whatever its assignment `status`; the part reads that decision
and does not recompute ownership. uint32.

% §2.9

**Placement by retained cells: the first positive view, order 0** (the label measurement's
`scripts/labels_retained.py`, functions `retained_inputs` and `fuse_labels`, with the fusion measurement's
`scripts/fixtures.py`, function `first_positive`):

% W-359; W-354

1. **Retained-cell images.** Each FOV's territory image is mapped voxel by voxel: a value
   that is the `cell_id` of a retained cell of that FOV becomes that cell's `instance_id`,
   and every other value (background, an excluded cell) becomes 0. A retained cell keeps
   every voxel of its territory, also the voxels in another FOV's ownership region.
2. **Fusion.** The retained-cell images, converted as in "Fusion by multiview-stitcher"
   (as 2D for `plane` territories and one-plane FOVs), in placement table order, are fused
   by `fusion.fuse(images, transform_key="placed", interpolation_order=0,
   fusion_func=_first_positive, output_spacing=…)`, where `_first_positive` is a small
   function passed to the library: for each output voxel, the value of the first view in
   placement table order whose value is valid (not NaN) and greater than 0, and NaN where
   there is none, which the library writes as 0. It is a fusion function the library
   calls, not a second fusion.
3. The output, which the library casts back to uint32, is the label image. No owner mask
   and no shift of the IDs is used, so nothing is decoded.

**One ID domain.** The retained-cell images, the fused output, the file and the `labels`
section all hold `instance_id`s of the sample cell table, with 0 for no cell. The +1 shift
of the owner-masked rule (an option not taken, below) is not needed: no view holds a
positive value except at a retained cell, so 0 means "no retained cell here" and the next
view in table order is taken. The label measurement fused both encodings, unshifted and +1 on the retained
cells, on both fixtures at both scales and found equal images after decoding
(`tables/cases.csv`).

% W-359

Labels are never blended: the fusion measurement found that blending creates label values that occur in no
input, at order 1 and at order 0 alike (4,602 and 2,354 new values on the fixture), and
that the `level1` route's block mean creates them too (1,621 to 2,532), so neither is
used for labels (`tables/labels_summary.csv`, `tables/labels_new_values.csv`).

% W-354

**What the label image shows.** At each output voxel, the cell of the first FOV in
placement table order whose retained-cell image holds a cell at its voxel nearest to the
output voxel's centre. Order 0 takes that nearest voxel only inside the FOV's voxel-centre
extent (index 0 to `n − 1` on each axis), which is the library's coverage. Case by case,
with the label-measurement record of each:

% W-359

| Case | What the label image shows | Label-measurement record |
| --- | --- | --- |
| A retained cell | Its `instance_id` on every output voxel whose nearest voxel of its FOV holds the cell, except where a row below applies. | Label fixture of the fusion measurement at scale 1: 370 of 391 cells drawn whole with one ID (`tables/w354_summary.csv`) |
| An excluded cell | Not drawn: its voxels show a retained cell of another FOV where one lies there, and 0 elsewhere. | Both fixtures, both scales: `ids_output` holds only `instance_id`s of retained cells (`tables/w354_summary.csv`, `tables/two_fov_summary.csv`) |
| Two retained cells of different FOVs whose territories overlap | On the shared voxels, the cell of the FOV first in placement table order; the other cell everywhere else, so it is drawn incomplete. | Two-FOV fixture, cell 9: 1 of its 81 voxels shows cell 8 of the other FOV; label fixture of the fusion measurement at scale 1: 3 cells (`tables/two_fov_cells.csv`, `tables/w354_incomplete_edges.csv`) |
| A cell retained by two FOVs | Both observations, each with its own `instance_id`: the first FOV's whole, the second's where the first does not cover it. One biological cell then shows two IDs. Its rows may carry no flag: `conflict` marks only some of these cells ("Conflicts" below). | Two-FOV fixture, cell 6: two IDs; label fixture of the fusion measurement: no such cell (`tables/two_fov_cells.csv`, `tables/w354_summary.csv`) |
| A cell retained by no FOV | Absent: its area shows 0, or a retained cell of another FOV where one lies there. It may have no flagged row: `unclaimed` marks only some of these cells ("Cells retained by no FOV" below). | Two-FOV fixture, cell 7: absent (`tables/two_fov_cells.csv`) |
| A cell whose ownership centroid lies exactly on an ownership boundary | Retained by the FOV first in table order (the tie rule of O1), and drawn once, whole. | Two-FOV fixture, cell 5: one ID, whole (`tables/two_fov_cells.csv`) |
| A retained cell cut by the border of its FOV's image | Drawn as its FOV holds it, cut at that border; the rest is not drawn from another FOV, which excluded the cell. | Two-FOV fixture, cell 10: 35 of its 81 pixels imaged and drawn (`tables/two_fov_cells.csv`) |
| A retained cell that reaches the first or last voxel of its FOV, at a fractional offset | Up to one row or column of output voxels just beyond its FOV's voxel-centre extent, within half a voxel of its FOV's footprint, stays without the cell, because no other FOV draws it; at the sample's outer border no other FOV covers that row at all. | Label fixture of the fusion measurement at scale 1: 18 cells, 17 at the sample's outer border and 1 at an interior FOV edge (`tables/w354_incomplete_edges.csv`) |
| A placeholder FOV | Nothing: it has no retained cell and is not read. The part of its footprint that no owning FOV covers is 0, and a cell only it imaged is absent; a neighbour's retained cell that reaches into its footprint is drawn as the neighbour holds it. | Label fixture of the fusion measurement with its centre FOV as a placeholder, scale 1: 13 cells absent, each retained by no FOV; no value in no input (`tables/cases.csv`, `tables/case_placeholder_summary.csv`) |
| Background | 0, as is every voxel outside every included FOV and every voxel where no FOV draws a retained cell. | Two FOVs without a cell: an image of zeros at both scales (`tables/cases.csv`) |
| An assembly made with `conflicts=False` | The same image: retention does not depend on the `conflict` and `unclaimed` flags, which are then not computed; the territory files are those of "Inputs", and the `labels` section records the conflict count as null. | Not run: the rule reads `assembly_status` only, which `conflicts` does not change ([assembly specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/assembly.md), "Flags") |

% W-359; W-354

**Counts on the two fixtures.** The label measurement tested the rule on two synthetic fixtures: the
label fixture of the fusion measurement, rebuilt from its recipe and seeds (3 × 3 uint32 FOVs of 112 × 112
cut from one 2D scene of discs, seed 354, at steps of 96 px with a jitter of up to 0.4 px,
seed 355; 391 cells, 538 observations), and a hand-built fixture of two FOVs of 64 × 96 px
side by side, overlapping over 32 px, whose segmentations of the overlap differ so that
one cell is retained by both FOVs and one by neither (13 cells, 21 observations). A cell
is drawn **whole** when exactly one ID shows it on every output voxel whose nearest voxel
of its FOV's footprint holds it and its FOV's image does not cut it; **incomplete** when
one ID shows it otherwise; **absent** when no ID shows it (the label measurement's `worker-notes.md`,
section 3).

% W-359; W-354; W-359 `worker-notes.md`

| Fixture, scale | Cells | Whole, one ID | Incomplete | Two or more IDs | Absent | Values in no input | Label-measurement table |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Label fixture of the fusion measurement, 1 | 391 | 370 | 21: 3 in part under a retained cell of another FOV, 18 beyond their FOV's voxel-centre extent | 0 | 0 | 0 | `tables/w354_summary.csv` |
| Label fixture of the fusion measurement, 0.25 | 391 | 366 | 11: 1 in part under a retained cell of another FOV, 10 beyond their FOV's voxel-centre extent | 0 | 14, too small for any output voxel centre | 0 | `tables/w354_summary.csv` |
| Two-FOV fixture, 1 | 13 | 9 | 2: cell 9 under cell 8, cell 10 cut by its FOV's border | 1 (cell 6) | 1 (cell 7) | 0 | `tables/two_fov_summary.csv`, `tables/two_fov_cells.csv` |
| Two-FOV fixture, 0.25 | 13 | 10 | 1: cell 10 | 1 (cell 6) | 1 (cell 7) | 0 | `tables/two_fov_summary.csv`, `tables/two_fov_cells.csv` |

% W-359; W-354

These are synthetic 2D fixtures; that label fusion behaves the same on real label images
and on 3D label volumes stays an assumption (A2).

**Scale factor 0.25.** `labels_scale_yx` (default 0.25, `1/k` with `k` whole) is requested
with `output_spacing` (the sampling route R3 of "The scale factor"), on the FOVs' own
grids, at order 0: no label is averaged. The label grid then has, per axis of Y and X,
spacing `k·s_a`, origin `o_a` and size `floor((max_f (origin_f,a + (n_f,a − 1)·s_a) − o_a) /
(k·s_a) + 1e-9) + 1` (the fusion measurement's `tables/reduced.csv`, rows `spacing`); Z keeps the fusion
grid's. This grid does not nest with the fused image's at the same `k` (their origins
differ by `s·(k − 1)/2`); each file records its own geometry and the store places each
with its own transform. The part records the grid the library returned and checks it
against these formulas (L5 of "Read-back checks").

% W-354 `tables/reduced.csv`

**IDs absent at the output scale.** Sampling drops small or thin territories: at scale
0.25, 14 of the 391 retained cells of the label fixture of the fusion measurement had no output voxel, and at
scale 1 none (the label measurement's `tables/w354_summary.csv`, columns `absent_ids` and
`absent_ids_list`). The `labels` section therefore records, all as `instance_id`s:

% W-354; W-359 `tables/w354_summary.csv`

* `ids_input`: the `instance_id`s of the retained cells whose territory holds at least one
  voxel in their FOV's image, that is the positive values of the retained-cell images of
  step 1;
* `ids_output`: the nonzero values of the label image;
* `absent_ids = ids_input − ids_output`, in full, with its count.

A retained cell is absent when none of its voxels is shown: at scale 1 only when retained
cells of earlier FOVs cover all of them or they all lie beyond its FOV's voxel-centre
extent, and at a reduced scale also when no output voxel centre falls in it. A cell
retained by no FOV is not in `ids_input`, so `absent_ids` does not hold it, and no count
of the `labels` section covers it; the flag `unclaimed` marks only some of these cells
("Cells retained by no FOV" below). There is no fallback to a finer scale. Example: one
owning FOV whose territory is all one retained cell of `instance_id` 1. Its retained-cell
image holds 1 everywhere, and so does the label image where the FOV covers;
`ids_input = {1}`, `ids_output = {1}` and `absent_ids` is empty.

**Grid and dimensionality.** For `volume` and `extended` territories of multi-plane FOVs
the label image has the stack's Z; for `plane` territories, and for one-plane FOVs, whose
territories are converted as 2D like their images ("The fusion grid"), it is 2D, and its
Z geometry in the file and the record is the projection's slab (`S_z = z_hi − z_lo`,
`O_z = (z_lo + z_hi)/2`), as for a projection. There is no label projection of volumes,
because a maximum of IDs means nothing. Label fusion on real and on 3D label images is
assumption A2.

**IDs and float32.** `fuse` passes values through float32, which holds every integer up to
2^24 = 16,777,216 exactly. The rule shifts no ID, so the largest value is `N` for `N` cell
rows, and the entry refuses a sample with more than 16,777,216 cell rows before writing
("Checks before anything is written", 6; "Choices decided at the gate", 3). The fusion measurement found
that IDs above 2^24 (shifted by 2^24) produce 187 values found in no input even with a
non-blending function (`tables/labels_summary.csv`, `id_offset` 16777216).

% W-354

**Conflicts.** The label image resolves nothing: a cell retained by two FOVs shows two IDs
(above). The flag `conflict` is the cell table's, by the rule of [assembly specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/assembly.md),
"Flags": a retained cell is flagged when its ownership centroid lies inside the territory
of a retained cell of another FOV, or the ownership centroid of a retained cell of another
FOV lies inside its territory. A cell retained by two FOVs may therefore carry no flag:
when neither observation's territory holds the other's ownership centroid, the pair
raises no `conflict`. Cell 6 of the two-FOV fixture is one: A's view and B's are discs of
radius 4 px (49 px) with ownership centroids at x 77 and x 82, 5 px apart, so neither
holds the other's centroid and neither row is flagged (the label measurement's `scripts/labels_retained.py`,
`tables/two_fov_observations.csv`; the flag rule evaluated on the label measurement's recipe in the flag check's
`tables/label_pair_flags.txt`). The conflict count of the `labels` section counts flagged
cells only: the cell rows flagged `conflict` whose `instance_id` is in the label image,
not the cells retained by two FOVs, or null for an assembly made with `conflicts=False`.
On the two-FOV fixture it is 0, although cell 6 shows two IDs (its two views share 0.24 of
the smaller territory, below the duplicate rule's default 0.5). No cell is merged across
FOVs. The duplicate rule of the assembly ([assembly specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/assembly.md), "Retention", rule 8)
excludes the second observation of a cell whose two territories share at least its
threshold of the smaller one, so that observation is not retained and is not drawn; the
label export applies no rule of its own. The rule removes a second observation; it does
not complete a cell cut by a FOV edge.

% W-359 `scripts/labels_retained.py`; W-359; W-360

**Cells retained by no FOV.** The label image does not show such a cell (above). The flag
`unclaimed` is the cell table's, by the rule of [assembly specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/assembly.md), "Flags": a cell
excluded `not_owner` is flagged when no FOV owns its ownership centroid (only possible for
a cell of a `placeholder`), or when the owner FOV has no retained cell there, that is when
the owner's territory at that centroid, sampled as for `conflict`, is 0 or a cell the
owner excluded. A cell retained by no FOV may therefore have no flagged row: when the
owner's territory at the ownership centroid of each of its observations holds a different
cell that the owner retained, for instance because the owner FOV segmented the cell
together with a neighbour as one cell and retained that cell, no FOV retains the cell and
no row is flagged `unclaimed`; its area then shows the owner's merged cell. Cell 7 of the
two-FOV fixture is flagged: A's view is a disc of radius 4 px at x 83 and B's at x 76, so
each ownership centroid is owned by the other FOV, whose territory there is 0, and both
rows are flagged (the label measurement's `scripts/labels_retained.py`; the flag rule evaluated on the label measurement's
recipe in the flag check's `tables/label_pair_flags.txt`). The `labels` section records no count of
these cells.

% W-359 `scripts/labels_retained.py`; W-359; W-360 `tables/label_pair_flags.txt`

**File and store.** `cell_labels.ome.tif` in the folder, in the OME-TIFF layout, uint32,
with the label grid's geometry; with the SpatialData export also the labels element
`cell_labels` of the store, read from that file ("SpatialData export"). Its values are the
`instance_id` of the tables `cells` and `expression`, which annotate `cell_centroids`; the
labels element is linked to them through that ID, not through a second table annotation.

**The `labels` section of `export.json`**: status and reason; the SHA-256 of the
`assembly` section; the rule `retained_cells`; per FOV with a retained cell the territory
path and SHA-256 read and its number of retained cells; the library version, `fusion_func`
`_first_positive`, `interpolation_order` 0 and the route `output_spacing`;
`labels_scale_yx`, the grid returned by the library and the formula grid; the file's path,
SHA-256, shape and dtype; the counts of `ids_input` and `ids_output`, and `absent_ids`;
the conflict count; wall and CPU time and peak RSS.

| Option | Placement rule | On the label fixture of the fusion measurement at scale 1 | Consequence |
| --- | --- | --- | --- |
| **LR4. Retained cells drawn whole, first positive view, order 0 (decided)** | as above | 370 of 391 cells whole with one ID, 21 incomplete, none with two or more IDs, none absent (the label measurement's `tables/w354_summary.csv`) | Each retained cell appears once and whole except where neighbouring segmentations disagree; the image shows assembly's retention, a cell retained by two FOVs as two IDs and a cell retained by no FOV as absent. |
| LR1. Owner-masked FOVs, first positive view, order 0 (option not taken) | each FOV's labels, shifted by 1, cut to its O1 ownership region, with background there encoded as 1 | The fusion measurement: 8,016 of 8,203 overlap points where the covering FOVs disagree show the owner's label, 97.7 % rounded (101 points 0, 86 a neighbour's) (the fusion measurement's `tables/labels_summary.csv`, `tables/labels_overlap_points.csv`). The label measurement's rerun equals the array the fusion measurement saved and draws 322 cells whole, 16 incomplete, 53 with two or more IDs and none absent (the label measurement's `tables/w354_checks.json`, `tables/w354_summary.csv`) | A cell that crosses an ownership boundary is drawn with two IDs. |
| LR2. First valid view in table order, order 0 (option not taken) | no mask | The fusion measurement: 3,991 of 8,209, 48.6 % rounded | Table order decides the overlap; the shown cell may be one assembly excluded there. |
| LR3. `max_fusion`, order 0 (option not taken) | no mask | The fusion measurement: 3,929 of 8,232, 47.7 % rounded | The larger ID wins. |

% W-354; W-359 `tables/w354_summary.csv`; W-354 `tables/labels_summary.csv`; W-359; W-359 `tables/w354_checks.json`

**Decided at the gate: LR4** ("Choices decided at the gate", 2); LR1, LR2 and LR3 are
options not taken.

| Option | Place | Cost | Consequence |
| --- | --- | --- | --- |
| **LP1. The label OME-TIFF always, and the labels element when the store is written (decided)** | `cell_labels.ome.tif`; `sample.zarr/labels/cell_labels` | The store's copy repeats the bytes when the store is requested | Each part writes its own file, so `fuse_labels` and `export_spatialdata` can run alone; the same rule as for the fused image. |
| LP2. The store only when it is written | as the earlier page | One copy | `export_spatialdata` would have to fuse labels itself, or `fuse_labels` depend on the store. |

**Decided at the gate: LP1** ("Choices decided at the gate", 8).

**Size, derived here from the fusion measurement's grids** (not measured; uint32, four bytes per voxel): at
0.25 the LN label grid is 3,083 × 2,721 in Y and X; as a volume with 67 planes that is
2.25 GB, as a plane 34 MB. tissue-2D's plane is about 5,646 × 4,923, 0.11 GB.
cell-culture-3D is about 3,409 × 2,394, 1.47 GB as a volume of 45 planes (the Z extent of
the resource estimate's `tables/resource_estimate.csv`).

% W-354; W-328 `tables/resource_estimate.csv`

## Read-back checks

`check_export(folder)` reads the record and the written files back through the libraries
and evaluates these conditions; no viewer session is involved. Each condition with a
tolerance names the fusion-measurement record it is taken from, and each fixture is one the contract
accepts.

% W-354

**The record and the folder**

* R1. `export.json` parses; every part has one of the statuses of "When a part cannot run";
  no part is still the entry `not_run`, `export_not_finished` of an export that stopped
  before the part ended (the condition names the part);
  `complete` is true exactly when no requested part is `not_run` or `failed`; each section
  that records the SHA-256 of another section equals that section's current SHA-256 (the
  SHA-256 of its canonical JSON encoded as UTF-8: keys sorted, the separators `,` and `:`
  with no spaces, characters outside ASCII written as themselves rather than escaped, and
  a NaN or infinite value refused with `ValueError` rather than written; in Python,
  `json.dumps(section, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
  allow_nan=False)`). The parts include
  `segment` of the route after fusion ([sample-segmentation-contract](../../docs/sample-segmentation-contract.md), "The entry").
* R2. Every file the record lists exists with its SHA-256; every file under
  `export/<sample>/` (the store's files included) is listed, but those under `previews/`,
  the workflow's opt-in ownership previews (since 2026-10-24). A part with status `off`, `not_run` or
  `empty` lists no file.
* R3. When the `assembly` section is `written` or `reused`, it and the files of `tables/`
  and `sample.h5ad` pass `load_assembly` ([assembly specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/assembly.md), "Persistence"); when
  it is `off`, no such file exists.

% W-568

**Fused images**, each file of the `fusion` section:

* F1. The layout checks of its layout ("OME-TIFF layout", "OME-Zarr layout"), and
  conditions 1 to 3 of "Sample position to image position".
* F2. **Dtype.** The file's dtype is the input dtype of "Dtypes written" (float32 for a
  `sum` projection) and equals the record's (the fusion measurement's `tables/overlap_dtype.csv`).
* F3. **Uncovered voxels.** Coverage is decided in the axes in which `fuse_images` fused the
  file: Y and X for a projection and for every file of one-plane FOVs (fused as 2D, "The
  fusion grid"), and Z, Y and X for the stack of multi-plane FOVs. A voxel is uncovered
  when, for every included FOV, its centre lies outside that FOV's voxel-centre extent
  `[origin_f,a, origin_f,a + (n_f,a − 1)·s_a]` on at least one of those axes, by more than
  1e-6 of a fusion-grid voxel. Every uncovered voxel holds exactly 0 (the fusion measurement's
  `tables/overlap_dtype.csv`, `uncovered_corner_top_right` and
  `uncovered_corner_bottom_left` 0 for every function but `multi_view_deconvolution`, and
  `uncovered_is_nan` false). The slab of one-plane FOVs is not a coverage axis: two
  one-plane FOVs with constant positive values, unit Z spacing and Z origins 0 and 2 give
  one plane with slab origin 1 and spacing 3, whose voxels inside either FOV in Y and X
  hold that FOV's value, and only voxels outside both in Y and X hold 0. That the library
  gives 0 at every uncovered voxel, not only at the measured corners, and that a FOV's
  `level1` image covers no output voxel outside the FOV's level-0 extent, is assumption
  A13.
* F4. **Values of the library's blending**, on the fusion measurement's overlap fixture
  (`worker-notes.md` section 3): two constant 2D FOVs of 32 × 256 px at 0.194398 µm, the
  second at (8, 200) px, overlapping over 56 px in X, with values 100 and 203 (uint8),
  1000 and 3001 (uint16), and 0.25 and 0.8 (float32). At scale 1, where one FOV covers,
  the value equals that FOV's exactly; in the overlap every value lies between the two
  inputs, inclusive (the fusion measurement's `overlap_min` and `overlap_max`: 100 to 202, 1000 to 3000, and
  0.25000232458114624 to 0.7999976873397827); and an
  integer value never exceeds the truncation of the float mean, so 203 does not occur in
  the uint8 overlap.
* F5. **Position.** Condition 4 of "Sample position to image position" on its fixture, for
  the stack, which that 3D fixture of multi-plane FOVs gives (the condition is stated in Z,
  Y and X); a projection file is tied to the library by F6.
* F6. **Library consistency.** On a fixture of 2 × 2 FOVs, the output level read from the
  file equals the array a direct call of `fusion.fuse` returns for the inputs and
  arguments the `fusion` section records, exactly (assumption A6). This is what ties the
  file to the library's fusion; no oracle of Starfinder's own recomputes blended values.
* F7. **Projection and stack.** With `dimensionality: both`, the two files have equal Y and
  X grids (condition 2); their values are not compared (the fusion measurement's
  `tables/projection_fixture.csv`).
* F8. **No included FOV.** The section has `outputs: []` and `reason: "no_included_fov"`
  exactly when the placement has no included FOV.
* F9. A standalone OME-Zarr's output level equals the OME-TIFF's exactly (both are
  lossless writes of one fused array).

% W-354 `tables/overlap_dtype.csv`; W-354; W-354 `overlap_min`

**Labels**, the file of the `labels` section:

* L1. The layout checks of the OME-TIFF layout; dtype uint32.
* L2. **No new value.** Every nonzero value of the label image is the `instance_id` of a
  retained cell of the sample cell table and belongs to `ids_input` (the label measurement: no value found
  in no input on either fixture at either scale, column `values_in_no_input` of
  `tables/w354_summary.csv` and `tables/two_fov_summary.csv`). In the example of "IDs
  absent at the output scale" the label image holds 1, which is in `ids_input = {1}`.
* L3. **Drawn by retained cells.** At every voxel of value `v`, the territory of `v`'s FOV,
  sampled at the voxel's centre by `floor(c + ½)`, holds `v`'s cell (the label measurement checked this
  for every drawn voxel of both fixtures at both scales, the assertions of
  `scripts/labels_retained.py`, function `per_cell`). On the label measurement's two fixtures, built from
  their recipes, the counts of "Counts on the two fixtures" (whole, incomplete, two or
  more IDs, absent, values in no input) are reproduced exactly at scale 1 and 0.25, and
  the cells of the two-FOV fixture show the outcomes of `tables/two_fov_cells.csv`: cell 6
  two IDs, cell 7 absent, cell 5 one ID and whole, cell 9 one voxel under cell 8 at scale
  1, cell 10 cut. By the rule `segmentation` of the route after fusion, L3 reads the
  sample's territories instead of each FOV's: at every voxel of value `v`, the sample's
  territories, sampled at the voxel's centre by `floor(c + ½)`, hold `v`'s cell.
* L4. **Absent IDs.** The record's `ids_input`, `ids_output` and `absent_ids` equal their
  recomputation: `ids_input` from the territory files, the `instance_id` map and the cell
  table's `assembly_status`, `ids_output` from the label file, and
  `absent_ids = ids_input − ids_output` (empty in the example).
* L5. **Grid.** The label file's spacing, origin and size equal the formulas of "Scale
  factor 0.25" within 1e-9 µm (the fusion measurement's `tables/reduced.csv`, `geometry_matches_formula` of
  the `spacing` rows). By the rule `segmentation`, the grid is that of "The label images"
  of [sample-segmentation-contract](../../docs/sample-segmentation-contract.md): the label scale's spacing, the territories'
  origin, and `floor((N − 1)·S / (k·s) + 1e-9) + 1` pixels for territories of `N` pixels
  of spacing `S`, Z the segmentation grid's slab.
* L6. **ID bound.** A sample with more than 16,777,216 cell rows (2^24) is refused before
  anything is written (on the route after fusion, when the number of cells is first known:
  [sample-segmentation-contract](../../docs/sample-segmentation-contract.md), "The label images").

% W-359; W-354 `tables/reduced.csv`

**The sample segmentation** of the route after fusion, when the part `segment` is
`written` or `reused`: the conditions T1 to T5 of [sample-segmentation-contract](../../docs/sample-segmentation-contract.md)
("The record").

* T1. The `segmentation` section's label files exist with their SHA-256, are uint32 and
  have the recorded grid's shape and metadata; when the `fusion` section is `written`,
  for each input image it lists the fused projection, whose SHA-256 equals the recorded
  input SHA-256 and whose grid equals the recorded grid within 1e-9 µm; when it is `off`,
  `not_run` or `empty`, the label files alone.
* T2. The positive values of the cell run's label image are exactly the `cell_id`s of the
  sample cell table, one row each, all with the run's namespace.
* T3. Every retained molecule's `cell_id` is the territory value at the pixel
  `floor(j + ½)` of `j = (sample_yx − O_yx) ⊘ S_yx` (0 for an `unassigned` molecule), and
  a molecule outside the grid is excluded `outside_grid`.
* T4. Every cell's `fov_id` and `owner_fov_id` follow the rule of "The cells", recomputed
  with the ownership regions.
* T5. The `assembly` and `labels` sections' `segmentation_files` equal the label files and
  SHA-256 values of the `segmentation` section (and of `territories.ome.tif`).

**SpatialData** round trip, with `spatialdata.read_zarr` of the written store. Each check
below applies to the elements the `spatialdata` section records as `written`; an element
recorded as `omitted` is tested by check S1 only.

* S1. **Elements**: the element names and kinds in the store equal those recorded as
  `written`, and no two names are equal ignoring case; every element the configuration
  asks for is recorded as `written` or `omitted`; for each `omitted` element the reason
  holds when recomputed from the sample tables, the `fusion` and `labels` sections and the
  configuration (`empty`: its population has no row; `no_cells`: the sample is
  `molecules_only`; `no_included_fov`: the `fusion` section has that reason;
  `not_requested`: the key is off; `part_not_run`: the part's status is `not_run`), and the
  element is absent from the store; when the section records `store: null`, no
  `sample.zarr` exists and every element is `omitted`. One image element per recorded
  fused image; no standalone OME-Zarr of the same image exists under `export/<sample>/`;
  the store's Zarr format and each raster group's version string are those of the
  recorded format.
* S2. **Points**: the `molecules` rows, joined on `(spot_namespace, spot_id)`, equal the
  sample molecule table's rows with finite sample coordinates, column by column: `x`,
  `y`, `z` equal `sample_x`, `sample_y`, `sample_z` exactly, `fov_index_z/y/x` equal the
  per-FOV index columns `z`, `y`, `x`, every other source column equals its source (NaN equals NaN, null
  equals null), the column `cell_namespace` of the molecule's cell key among them (a
  table saved before that column existed has none, and neither has its store),
  `cell_instance_id` follows S5, and no `sample_x/y/z` column remains; the
  dtypes equal those the section lists. Together with `molecules_unplaced` (each where
  written), every row of the sample molecule table appears exactly once; a sample
  molecule table without rows has both omitted.
* S3. **Source content of the tables is preserved.** The `cells` and `cells_unplaced`
  rows, joined on `(cell_namespace, cell_id)`, equal the sample cell table's rows in every
  source column, and together contain each of them exactly once; `molecules_unplaced`
  likewise for its molecule rows; `expression` equals `sample.h5ad` in `X`, `layers`,
  `var` and `obsm` exactly, in every source column of `obs`, and in every key of the
  source `uns`, each with the same value. Values are compared, and each column's dtype
  after the round trip equals the one the section records (how anndata 0.12.6 under
  pandas 3 returns string columns is recorded rather than assumed). When `expression` is
  omitted, `sample.h5ad` has no row.
* S4. **Export metadata is present, separately.** `cells` and `expression` have the added
  `obs` columns `region` (categorical, every value `cell_centroids`) and `instance_id`
  (`uint32`), and `uns["spatialdata_attrs"]` equal to `{"region": "cell_centroids",
  "region_key": "region", "instance_key": "instance_id"}`; they have no other key or
  column that their source lacks. `cells_unplaced` and `molecules_unplaced` have no
  `spatialdata_attrs` key and no added column besides `instance_id` and
  `cell_instance_id` respectively. Points, shapes and raster elements carry the transforms
  of S6 and no feature or instance key.
* S5. **Links and accounting**: `instance_id` of `cells` and `cells_unplaced` together is
  `1..N` in the sample cell table's order (empty when both are omitted); `cells` and
  `cell_centroids` are written or omitted together, and the index of `cell_centroids`
  equals the `instance_id` of `cells`; every molecule of `molecules` and
  `molecules_unplaced` with a non-null cell key has a `cell_instance_id` equal to that
  cell's `instance_id`, and every molecule without one a null `cell_instance_id`; every
  `expression` row is a `cells` row with `assembly_status == "retained"` and
  `status == "kept"`. Counts of molecules by `assignment_status` (`assigned`,
  `unassigned`, `excluded_cell`, `outside_grid`) and by `assembly_status` and reason,
  recounted from `molecules` and `molecules_unplaced`, equal the section's and the
  assembly record's counts.
* S6. **Calibration and alignment**: the transform of each written image and labels
  element to the sample coordinate system equals `Scale(S)` then `Translation(O − S/2)` on
  its axes within 1e-9, with `S` and `O` of its file's output level (the fusion measurement's
  `readback/spatialdata_read.json`); its extent there (`get_extent`) is
  `[O − S/2, O + (N − ½)·S)` per axis; points and shapes have the identity transform. A
  point at `q`, transformed into the image's intrinsic coordinates, falls in voxel
  `floor(j + ½)` (condition 5 of "Sample position to image position").
* S7. **Values**: each written image and labels element equals its OME-TIFF's output level
  exactly, in value and dtype (the element is read from that file).
* S8. The SHA-256 of every store file equals the section's (no file when `store: null`).

% §2.9; W-354

**Deferred**: viewer qualification in Fiji and napari (opening the OME-TIFF and the store,
display of levels, channels, labels and points). These checks do not stand in for it; the
record says `viewer_qualification: "deferred"`.

## Configuration

```python
@dataclass(frozen=True)
class FusionConfig:
    images: tuple[str, ...] = ("reference_stain",)  # source-image names; further channels
    dimensionality: str = "projection"               # "projection" | "stack" | "both"; per dataset
    projection: str = "max"                          # "max" | "sum"
    scale_yx: float = 0.5                            # 1/k, k whole ≥ 1
    ome_zarr: bool = False                           # standalone OME-Zarr beside each OME-TIFF
    ome_zarr_version: str = "0.5"                    # "0.5" (Z2) | "0.4" (Z1)
    pyramid_levels: int = 0                          # OME-TIFF SubIFD levels

@dataclass(frozen=True)
class LabelConfig:
    scale_yx: float = 0.25                           # 1/k, k whole ≥ 1

@dataclass(frozen=True)
class ExportConfig:
    assemble: bool = True
    fuse: bool = True
    fuse_labels: bool = False
    export_spatialdata: bool = False
    assembly: AssemblyConfig = AssemblyConfig()      # [assembly specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/assembly.md)
    table_format: str = "csv"                        # "csv" | "parquet"
    fusion: FusionConfig = FusionConfig()
    labels: LabelConfig = LabelConfig()
    spatialdata: SpatialDataExportConfig = SpatialDataExportConfig()
    route: str | None = None                         # "per_fov" | "after_fusion"; None: by dimensionality
    segment: bool = True                             # the part segment of the route after fusion
    segmentation: SampleSegmentationConfig | None = None  # its settings (sample-segmentation-contract)
```

`__post_init__` refuses unknown values, a scale whose reciprocal is not whole, a negative
`pyramid_levels`, an empty `images`, image names that are equal when case is ignored
(spatialdata treats element names case-insensitively, "SpatialData export"), and
`ome_zarr` together with `export_spatialdata` (the store holds the one OME-Zarr copy).
Once the grid is known, `fuse_images` refuses a `pyramid_levels` larger than
`floor(log2(max(N_y, N_x)))`. The fusion function, the interpolation orders and the
reduction routes are fixed by this page, not configuration keys.

The OME-Zarr format choice of "OME-Zarr layout" is the one configuration of the store and
of the standalone file:

```python
@dataclass(frozen=True)
class SpatialDataExportConfig:
    zarr_format: int = 3            # 3: spatialdata's default (Z2); 2: SpatialDataContainerFormatV01 (Z1, A10)
```

and `FusionConfig.ome_zarr_version` (`"0.5"` under Z2, `"0.4"` under Z1), passed as the
library writer's `ngff_version`. `__post_init__` refuses other values. Both follow
the gate's choice 7; the defaults are Z2.

## Resources

Recorded by the fusion measurement (`tables/runs.csv`, `worker-notes.md` sections 8 and 10; wall, CPU and
peak RSS rounded from `wall_s`, `cpu_s` and `peak_rss_kib`), four threads,
`level1` route, weighted average at order 1, every tile loaded first, the library's
uncompressed TIFF writer:

% W-354

| Sample and output | Grid (Z ×) Y × X | Wall | CPU | Peak RSS | Bytes |
| --- | --- | --- | --- | --- | --- |
| tissue-2D, projection at 0.5 | 11,291 × 9,845 | 24.2 s | 47.2 s | 1.42 GiB | 111,160,279 |
| LN sample1, projection at 0.5, multiview-stitcher placement, TIFF and OME-Zarr | 6,166 × 5,441 | 72.0 s | 40.8 s | 0.78 GiB | 60,416,083 |
| LN sample1, projection at 0.5, BigStitcher placement | 6,167 × 5,435 | 67.9 s | 38.1 s | 0.60 GiB | 33,518,029 |
| cell-culture-3D, projection at 0.5 | 6,818 × 4,788 | 22.9 s | 28.7 s | 0.62 GiB | 32,644,968 |
| LN sample1, 3D stack at 0.5 | 67 × 6,166 × 5,441 | 241.7 s | 532.7 s | 12.35 GiB | 2,247,808,158 |

* The LN projections are dominated by loading the 64 tiles from the share (about 59 s);
  fusing and writing took under 10 s (`in_process_steps_wall_s`: `fuse_compute` plus the
  writes, 4.72 + 0.6 + 2.81 = 8.13 s from the multiview-stitcher placement and
  3.73 + 0.61 = 4.34 s from BigStitcher's).
* The stack held the 7.2 GB of loaded tiles plus their reduced copies; fusing and writing
  it took 173.82 s (`in_process_steps_wall_s`). No bounded memory is promised for a
  whole-sample stack (A1).
* A cell-culture-3D stack was not run: about 160 s and 8 GiB by voxel count, an estimate
  (A4).
* A dataset that sets `dimensionality: stack` pays the stack's cost instead of the
  projection's: for LN sample1 at scale 0.5, the stack row above in place of its
  projection row ("The fused image by dataset").
* Starfinder's OME-TIFF writer adds tiling and zlib to these costs (A8).
* The SpatialData store with one image level: 1,662,724 KiB (1.59 GiB, rounded) peak and
  156.48 s for the LN stack
  through `aszarr` (the fusion measurement's `readback/spatialdata_write_ln_stack_single_aszarr.json`); the
  default projection's image is 32,644,968 to 111,160,279 bytes (0.030 to 0.104 GiB,
  rounded; `output_bytes` of the `*-proj-s05-a2` rows of `tables/runs.csv`, where the LN
  multiview-stitcher row's 60,416,083 bytes include its OME-Zarr).
* The label export at 0.25 was not measured on real labels (A2); its sizes are in
  "Assembled label export".

% W-354 `readback/spatialdata_write_ln_stack_single_aszarr.json`

### Dependencies per output

On Python 3.11 or later, the supported versions once the dependency plan of
[stitching specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/stitching.md) is applied. The versions are those of `src/python/uv.lock`
(sha256 `bc994b55…`) for its Python 3.11-and-later entries, and of the trial resolution of
2026-10-06 for the `multiview-stitcher` extra, which is not yet in the lock.

| Output or path | Packages | Extra | Versions |
| --- | --- | --- | --- |
| Sample tables as CSV (`table_format: csv`, the default) | numpy, pandas | none (base) | numpy 2.2.6, pandas 3.0.0 |
| Sample tables as Parquet | pyarrow, through the checkpoint table writer ([checkpoints](../../docs/checkpoints.md), "Table formats") | `checkpoint` | pyarrow 23.0.0 |
| `sample.h5ad` | anndata | `anndata` | anndata 0.12.6 |
| Any fused image, the default projection included, and the label export (`fuse_images`, `fuse_labels`) | multiview-stitcher with ngff-zarr, itkwasm, itkwasm-downsample, wasmtime, dask, xarray, zarr | `multiview-stitcher` (Python 3.11 to 3.13; empty on 3.14 and later, where the parts record `extra_missing`) | multiview-stitcher 0.1.62, ngff-zarr 0.38.0, itkwasm 1.0b201, itkwasm-downsample 2.0.2, wasmtime 49.0.0; dask 2024.11.2, xarray 2026.1.0, zarr 3.1.5 as locked |
| Fused and label OME-TIFF (T1) and its read-back | tifffile, numpy; zlib through tifffile's own codec when `imagecodecs` is absent | none (base) | tifffile 2026.1.28 |
| Standalone OME-Zarr (`ome_zarr: true`) and its read-back | the library's writer, zarr | `multiview-stitcher` | as above |
| SpatialData export and its round trip | spatialdata with anndata, zarr, dask, xarray, multiscale-spatial-image, geopandas and shapely (shapes), pyarrow (points); tifffile `aszarr` to read the image files | `spatialdata` | spatialdata 0.6.1, geopandas 1.1.2, shapely 2.1.2, pyarrow 23.0.0, zarr 3.1.5 |

The default output therefore needs the `anndata` and `multiview-stitcher` extras (and
`checkpoint` for Parquet tables). Without `multiview-stitcher` the tables and
`sample.h5ad` are still written and the record says the fuse part did not run. A missing
extra is named in the record and the warning; importing `starfinder.export`
imports neither multiview-stitcher, zarr nor spatialdata. `spatialdata-io` in the
`spatialdata` extra is not used. No dependency or lock change is made by this page.

## Choices decided at the gate

Jiahao decided at the second round of the review gate (2026-10-06) each of the eleven
choices this page had left open, as the page proposed it. Choices 1 to 6 are the fusion measurement's
(`worker-notes.md` section 12). The sections named hold the decided text; the other
options stay there as options not taken.

% W-329; W-354

1. **Meaning of a reduced scale**: R1, the library's `level1` route. Not taken: R2 fuse at
   scale 1 and take a block mean in Starfinder, R3 sampling ("The scale factor").
2. **Label placement rule**: LR4, by retained cells, each drawn whole, with the first
   positive view at order 0. Not taken: LR1 owner-masked FOVs, LR2 first valid view, LR3
   `max_fusion` ("Assembled label export").
3. **Label IDs above the bound of the float32 computation**: refused before anything is
   written, at more than 16,777,216 cell rows. Not taken: remap to a compact range before
   fusion and back, or accept the float32 rounding ("Assembled label export", "IDs and
   float32").
4. **Projection and stack**: no equality between the projection and the maximum of the
   stack; the measured difference is stated. Not taken: P4, projecting the stack ("The 2D
   projection").
5. **The default TIFF**: T1, Starfinder writes the OME-TIFF with tifffile from the array the
   library fused. Not taken: T2 the library's TIFF ("OME-TIFF layout").
6. **Fusion grid size**: the library's union grid is the fusion grid. Not taken: the earlier
   grid that contains every FOV's last voxel centre, passed as `output_stack_properties`
   ("The fusion grid").
7. **OME-Zarr format under Python 3.11 or later**: Z2, OME-NGFF 0.5 on Zarr format 3. Not
   taken: Z1 NGFF 0.4 on Zarr format 2 ("OME-Zarr layout").
8. **The label export's default and place**: off by default; LP1, the label OME-TIFF is
   always written by its part, and the store element when the store is written. Not taken:
   on by default, and LP2 ("Assembled label export").
9. **Input dtypes**: uint8, uint16 and float32, the dtypes measured. Not taken: also int8,
   int16, uint32, int32 and float64 after the implementation verifies them (A5).
10. **Levels in the store**: one resolution level. Not taken: spatialdata's
    `scale_factors`, which reduce Z and cost more memory ("SpatialData export").
11. **A part that cannot run**: recorded and warned about. Not taken: raise after writing
    the record, or refuse before writing ("When a part cannot run").

## Golden tests and the validation checks

% Golden tests and the §2.10 checks

`test_workflow_stitching_golden.py`, `test_workflow_assembly_golden.py` and every existing
test pass unchanged; nothing here changes a legacy script or an existing output. The checks
of the export entry, the fused image, the label export, the SpatialData export and the
workflow blocks are the rows EX, FI, LB, SD and WF of [stitching and assembly validation design](https://github.com/wanglab-broad/starfinder/blob/dev/design/validation/stitching-assembly.md),
"Checks", from the conditions of "Sample position to image position" and "Read-back
checks", with their tests in "Implemented checks". Tests that call
multiview-stitcher need its extra and are skipped without it, naming the extra.

% "Implemented checks (W-373)" of [stitching and assembly validation design](https://github.com/wanglab-broad/starfinder/blob/dev/design/validation/stitching-assembly.md)

## Workflow configuration

The Python-only `export` block, read by `_run_export` in the workflow adapter and run by
one per-sample rule `export_sample`, which runs `Dataset.export` after the placement
exists and, on the route per FOV, after `reads_assignment` of every FOV of the sample; on
the route after fusion the rule needs the molecule tables and the source images
([sample-segmentation-contract](../../docs/sample-segmentation-contract.md), "The route setting"). The assemble part takes its
keys from the `assembly` block ([assembly specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/assembly.md), "Workflow configuration"):

```yaml
export:
  assemble: true              # false: reuse the saved assembly of export/<sample>/
  fuse: true                  # false: no fused image (the tables and sample.h5ad stay)
  fuse_labels: false          # true: cell_labels.ome.tif at labels_scale_yx
  export_spatialdata: false   # true: export/<sample>/sample.zarr
  images: [reference_stain]   # source images; further channels are further names
  dimensionality: projection  # projection | stack | both; dense 3D tissue (LN sample1): stack
  projection: max             # max | sum
  scale_yx: 0.5               # 1/k
  ome_zarr: false             # standalone OME-Zarr (not with export_spatialdata)
  ome_zarr_version: "0.5"     # "0.5" | "0.4"
  pyramid_levels: 0
  labels_scale_yx: 0.25       # 1/k
  spatialdata_zarr_format: 3  # 3 | 2
  route: null                 # per_fov | after_fusion; null: by dimensionality
  segment: true               # false: reuse the saved sample segmentation
  segmentation: null          # the settings of the route after fusion
```

* The rule runs for every sample of a configuration with `stitching` and `assembly`
  blocks; without an `export` block the defaults above apply. Without an `assembly` block
  the legacy rules run unchanged and nothing here runs.
* The block is part of each dataset's configuration, so `dimensionality` is set per
  dataset: a dataset of dense 3D tissue, such as LN sample1, sets `dimensionality: stack`
  and gets no projection file; the others keep the default projection ("The fused image
  by dataset").
* It reads `stitching/{sample}/placement.json`, the saved assignment folders (or molecule
  tables) and reference grids of the sample's FOVs (from the files the top-level
  `reference_image` names), and the source images, and writes the
  files of "The export folder and the output profiles". Its declared output is
  `export/{sample}/export.json`, which lists every other file; a part that cannot run
  therefore does not fail the rule, and the warning and the record say so ("When a part
  cannot run").
* After `Dataset.export`, with `assembly.assignment` set, `assembly.previews: true` (since
  2026-10-24; default false) and the assembly written or reused, the rule also draws each FOV's
  assignment figures `export/{sample}/previews/{fovID}/assignment.png` and
  `assignment_single_layer.png`, with the FOV's ownership region
  ([assembly specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/assembly.md), "Workflow configuration"; added on 2026-10-08). The folder `previews/` is a
  declared directory output of the rule on the route per FOV with `assemble` on, the
  conditions the configuration decides; the rule creates it whenever it declares it (empty
  when the assemble part was neither written nor reused). It is not listed in
  `export.json`; `Dataset.export` itself writes no figure. On the route after fusion, or
  with `assemble: false`, no assignment figure is drawn and the folder is not declared.
* A default-tier test keeps the static schema equal to the fields of `ExportConfig`,
  `FusionConfig`, `LabelConfig` and `SpatialDataExportConfig`, as for the other blocks.

% W-568; W-330

## Exclusions

No implementation, rename, dependency or lock change; no code brought over from the
reference branch; no fusion of Starfinder's own and no BigStitcher fusion in Fiji; no
viewer session (deferred); no change to the placement, segmentation or assignment contracts, and none to the
assembly contract beyond its `Dataset` method, saved layout and workflow configuration;
no reassignment of molecules or recomputation of counts from a fused image or a label
raster on the route per FOV (the route after fusion assigns the sample's molecules to the
cells of a segmentation of the fused image: [sample-segmentation-contract](../../docs/sample-segmentation-contract.md)); no real data (every measured number comes from the records of the evidence list by path, and the label and stack sizes are derived from them without reading
an image).

% §2.9; W-323; W-324; W-328; W-354; W-359



% toctree title: Stitching and assembly validation (§2.10)
