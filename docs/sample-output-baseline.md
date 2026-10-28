# Sample-output baseline: the reference-branch exporter, `dev` and today's fused files

% Status: Accepted (W-329, 2026-10-07, at 9033d46)

Note: {doc}`sample-output-contract` was accepted at its review gate (2026-10-07) and is implemented in `starfinder.export` and `Dataset.export`; this page keeps the text of its revision.

% Note (W-583); accepted at W-329 (2026-10-07)

This page records what exists for the sample outputs of stitching and assembly before {doc}`sample-output-contract`
specifies them: the shared raster preparation and the partial exporter on
the reference branch, the reference branch's sample-export specification and its documentation there, what `dev`
has, and the fused files of the two sample datasets as today's practice. For each part it
proposes to bring it over, rewrite it or leave it. Nothing here is implemented, nothing is
copied from the reference branch and nothing is renamed.

% the §2.10 sample outputs; the shared raster preparation (W-169); the partial exporter (W-170); the W-168 specification

Sources, read on 2026-10-06:

* the reference branch at `17e8dda` (2026-09-23), read with `git show
  17e8dda:<path>` and `git ls-tree`; it is not checked out or merged;
* `dev` at `8e7d256` (`origin/dev`, the batch's base), and the batch branch at `d9f62ad`;
* the file listings (names and sizes only, no pixel read) of the `images/fused/` folders of
  the cell-culture-3D and tissue-2D sample-dataset outputs, kept outside the repository;
* `workflow/scripts/test/grid_stitching_backup.txt` on `dev`, the Fiji macro that wrote the
  tissue-2D fused files.

% the reference branch `origin/codex/chapter-ii-m1-m2` at `17e8dda` ("feat(w-172): implement validated issue deliverables", 2026-09-23)
% the file listings of `~/wanglab/Data/Analyzed/sample-dataset-output/cell-culture-3D/images/fused/` and `…/tissue-2D/images/fused/`

## The reference branch at `17e8dda`

### `src/python/starfinder/raster.py` (blob `3c5952a`, 299 lines): the shared raster preparation

% heading: `src/python/starfinder/raster.py` (blob `3c5952a`, 299 lines): W-169

Aligned-raster preparation for a sample export. It takes images that are already on one
common grid and prepares them; it assembles nothing and resamples no source grid.

| Name | What it does |
| --- | --- |
| `RasterConfig` | Frozen config: absolute power-of-two reduction factors `factors_zyx`, `coarser_levels` (extra pyramid levels), `coordinate_space` (`physical` or `index`, the latter only with every physical field unknown) and an optional `max_output_bytes` budget on the emitted arrays (not RSS). |
| `RasterLevel` | One prepared level: the images (`ImageLoadResult`, ZYXC), the uint labels (ZYX), the level's `ImageMetadata`, its factors and `index_to_source_zyx`, the 4 × 4 map from output voxel centres to native voxel centres (`i_source = f·i_output + (f − 1)/2`), and the trial diagnostics. |
| `RasterResult` | The accepted levels, the unchanged source metadata, the config and every trial, accepted or rejected. |
| `RasterPreparationError` | Invalid input or a resource failure, with the rejected trials in `diagnostics`. |
| `prepare_rasters(images, labels, *, metadata, declared_ids, config)` | Checks that every image shares the labels' grid, shape and metadata and has unique ordered channel labels, and that the declared label IDs equal the native positive labels; then tries the requested factors. Each trial needs exact divisibility; labels take the most frequent value of each block, background included, smallest ID on a tie; images take the block mean in float64, stored as float32 (float64 stays float64). A trial is rejected if a label is lost or new, if a cell's centroid moves more than `norm(f − 1)/2` source voxels or its Hausdorff distance exceeds `norm(f − 1)`, or if the extent changes; a rejected trial halves every factor above one and retries from the native arrays, down to `(1, 1, 1)`, where failure is an error. Coarser levels double the factors until one fails. The output geometry is spacing × f and origin + spacing·(f − 1)/2. Z = 1 requires `fz = 1`; there is no projection. |
| `remap_labels(masks, *, cell_keys)` | Combines disjoint, already aligned label images from several namespaces into one uint32 image with IDs `1..N` (sorted by namespace, then local label) and a reversible row map to `(cell_namespace, cell_id)`; overlap or a reused cell key is an error, never a merge. |
| `_blocks`, `_trial`, `_labels` | Private: block view of an array, one reduction trial with its fidelity checks, label validation. |

### `src/python/starfinder/sample_export.py` (blob `dca9fda`, 469 lines): the partial exporter

% heading: `src/python/starfinder/sample_export.py` (blob `dca9fda`, 469 lines): W-170, partial

One SpatialData Zarr store per sample, written from a `RasterResult` and saved per-FOV
artifacts.

| Name | What it does |
| --- | --- |
| `SavedFile` | A source file pinned by SHA-256; `verify()` rehashes it and returns a descriptor. |
| `SampleExportConfig` | `sample_id`, `target_frame_id`, a per-artifact `source_mappings` dict (each a `source_to_target` 4 × 4 affine with a provenance file and its SHA-256), `section_id`, `expression_meaning`, `assignment_provenance`, `complete_cell_relation`, `molecules_absent_reason`. |
| `export_sample(directory, rasters, *, config, raster_sources, cell_map, molecules=None, assignments=None, expression=None)` | Refuses an existing destination; verifies every source hash; loads each FOV's final molecule checkpoint through `MoleculeIndex` and `load_final_checkpoint`, maps the molecules through the supplied affine and checks its inverse; joins supplied assignments (missing rows become `unavailable`, explicit `unassigned` stays distinct) and a supplied H5AD by cell key (mask-only and table-only cells kept apart); builds `Image3DModel` and `Labels3DModel` elements (CZYX and ZYX, chunks `(1, 8, 32, 32)` and `(8, 32, 32)`), a `PointsModel` in the target frame and a `TableModel` bound to the labels by `region` and `instance_id`; writes the store with spatialdata into a sibling staging folder; rewrites each NGFF dataset's scale and translation to the absolute level geometry, checks codec, chunks and values of every `.zarray`; writes `molecules.parquet`, `cell-map.parquet`, `unmatched-cells.parquet`, `open-in-fiji.json`, `README.html` and `export-manifest.json` (every output's SHA-256); rechecks the sources and renames the staging folder into place. It imports `anndata`, `dask`, `datatree`, `spatialdata` and `zarr` on use. |
| `lookup_export_trace(directory, *, spot_namespace, spot_id)` | Verifies the manifest hashes and resolves a molecule through its final checkpoint to the original trace (`source_trace` of the reference branch's molecule outputs). |
| `open_sample_viewer(directory, *, level=0)` | A napari adapter that adds each stored level with its own scale and translation (the pinned napari multiscale reader lost the half-voxel offsets) and docks a cell and molecule inspector. |
| `_hash`, `_json`, `_keys`, `_matrix`, `_molecules`, `_geometry`, `_read_manifest` | Private helpers: hashing, JSON, key validation, affine validation, molecule loading, level geometry, manifest verification. |

% W-164 `source_trace`

It depends on `starfinder.io.molecules` (`MoleculeIndex`, `load_final_checkpoint`,
`ArtifactReference`), which exists on the reference branch only.

### Tests

`src/python/test/test_raster.py` (blob `82702d0`, 196 lines, 11 tests: the literal cases,
ties, geometry and fidelity, pyramid stop, dtype and mean oracle, channels, unknown
calibration, namespaces, budgets, unsupported frames) and `test_sample_export.py` (blob
`32b2820`, 262 lines, 8 tests: geometry, populations and links, the raster cases, missing
links, rejections, mask-only and table-only cells, regional chunk reads, stain role and
tampering, the three-molecule link oracle).

### Documentation on the reference branch

| File | Content |
| --- | --- |
| `docs/sample-export-contract.md` (blob `3260d7f`, 470 lines) | The reference branch's sample-export specification, `starfinder.sample_export/1`: payloads and populations (final molecules only, pre-QC counts kept, `unassigned` against `unavailable`, assignment links by key, mask-only and table-only cells, no recount of expression); the shared geometry and fallback of the shared raster preparation with its frozen numerical checks; the literal cases; one SpatialData Zarr v2 store with NGFF 0.4 raster groups for both viewers and no second raster copy; a pinned environment (spatialdata 0.2.5, ome-zarr 0.10.3, zarr 2.18.7, napari 0.5.6, napari-spatialdata 0.5.3, PyQt5) separate from `uv.lock`; the Fiji and napari reopening checklist; the evidence and delivery schema. Its last section records that the pinned Fiji N5 Viewer could not parse the written raster groups, so viewer qualification stayed open. |
| `docs/api/raster.rst`, `docs/api/sample_export.rst` (18 lines each) | Autosummary pages of the two modules. |
| `docs/examples/sample_export_cases.json` (28 lines) | Literal cases E1 to E9 (valid 2D and 3D reduction, small-cell and collision fallback, namespaces, pyramid stop, a missing declared label, empty and uncalibrated input), with independent expected labels, means and origins. |
| `docs/export-viewer-requirements.txt` (24 lines) | The isolated export and viewer pins listed above, to be locked separately. |
| `docs/foundation-qualification.md` (125 lines) | The foundation qualification: the bounded MATLAB/Python comparison protocol and the storage and reload protocol, which reruns the raster and export tests; Fiji acceptance is deferred to the partial exporter. |

% The W-168 specification; the shared geometry and fallback of W-169; W-171; deferred to W-170

The premise of the reference branch's sample-export specification differs from the accepted stitching and assembly design now: it took the assembly as a caller-supplied
affine per source and left "which overlap observation survives" open. That design now has a
placement record ([stitching specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/stitching.md)) and sample tables with ownership and
retention ([assembly specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/assembly.md)), so the export reads those instead.

% The W-168 premise differs from §2.10 now; §2.10 now has a placement record

## What `dev` has

Checked with `git ls-tree -r origin/dev` at `8e7d256`: no `raster.py`, no
`sample_export.py`, no `starfinder/io/molecules.py`, no sample-export page and no
`export-viewer-requirements.txt` (the same was checked at `2e945a8` on 2026-10-05).
What `dev` does have and the contract uses:

| Part | Where | Relevance |
| --- | --- | --- |
| OME-TIFF writer and readers | `src/python/starfinder/io/tiff.py`: `save_volume` (ZYX TIFF with a JSON description, or ZYXC OME-TIFF with the `ImageMetadata` as JSON in a `CommentAnnotation` of namespace `starfinder.image.ImageMetadata`), `load_volume`, `load_volume_zyxc` | Whole-array writes, no BigTIFF flag, no tiling, no resolution levels and no OME `PhysicalSize`; the metadata convention is reused by the fused files. |
| Geometry | `src/python/starfinder/image.py`: `ImageMetadata` with `index_to_world`, `world_to_index`, `cropped`, `rotated`, `projected` | `projected(method)` returns a frame with unknown physical fields; the fused projection needs its Y and X calibration, which the contract records differently ("The 2D projection"). |
| Projection | `src/python/starfinder/preprocessing/projection.py`: `ProjectionConfig` (`max` or `sum`, axis `z` or `channel`), `project_image` | The per-FOV projection rule the fused projection reuses for one FOV. |
| Placement and assembly | [stitching specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/stitching.md), [assembly specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/assembly.md) (Proposed, this batch) | The inputs of fusion and export. |
| Extras | `pyproject.toml`: `spatialdata = ["spatialdata>=0.1", "spatialdata-io>=0.1"]`; `anndata`; base `tifffile>=2023.0` | Locked: spatialdata 0.6.1, ome-zarr 0.12.2, zarr 3.1.5, numcodecs 0.15.1, anndata 0.12.6, tifffile 2026.1.28 on Python 3.11 and later; spatialdata 0.3.0, ome-zarr 0.10.2 or 0.11.1, zarr 2.18.3, numcodecs 0.13.1, anndata 0.11.4, tifffile 2025.5.10 on 3.10. The locked environment of this batch has zarr, numcodecs, tifffile and anndata installed, not spatialdata or ome-zarr. |
| Workflow | `workflow/rules/stitching.smk`, `common.smk` (`FUSE_DIR = <OUTPUT>/images/fused`) | `images/fused/{sample}/` holds only BigStitcher's inputs and `dataset.xml`; no rule fuses an image. |

spatialdata 0.6.1 (source read from the uv cache, `spatialdata/_io/format.py`, not
imported) writes by default its container format 0.2 on OME-NGFF 0.5 and Zarr v3
(`CurrentRasterFormat = RasterFormatV03`, a `FormatV05`), and can still write the NGFF 0.4 layout on
Zarr v2 (`SpatialDataContainerFormatV01` with `RasterFormatV01`, whose `multiscales`
version string is `0.4`, or `RasterFormatV02`, its default there, whose string is
`0.4-dev-spatialdata`; lines 156 to 179 and 345 to 350). Its raster
extent in intrinsic coordinates is `[0, n)` per axis (`_get_extent_of_data_array` in
`spatialdata/_core/data_extent.py`): voxel `i` covers `[i, i + 1)`, so its centre is at
`i + ½`, unlike the voxel-centre convention of `ImageMetadata`.

## Today's fused files

Fiji Grid/Collection stitching wrote the tissue-2D files
(`workflow/scripts/test/grid_stitching_backup.txt`: `fusion_method=[Linear Blending]`; the
first channel computed `TileConfiguration.registered.txt`, the other channels reused it).
The tool that wrote the cell-culture-3D files is not recorded in Git; the stitching input inventory names a Fiji
Grid/Collection 3D run as that dataset's reference placement. The `*_05.tif` files were
written between December 2023 and August 2024 by a tool that is not recorded. Sizes from
the listing:

% the stitching input inventory: W-323

| Dataset | Full-size files | `*_05` files | What the sizes show |
| --- | --- | --- | --- |
| tissue-2D (7 × 8 tiles of 3072², one plane) | `PI.tif`, `Gfap.tif`, `plaque.tif`, `tau.tif`, `ref_merged.tif`, `overlay.tif`: 444,335,200 bytes each; `PI_label.tif` 888,671,808; `labeled_cells.tif` 888,670,336 | `PI_05.tif`, `Gfap_05.tif`, `plaque_05.tif`, `tau_05.tif`: 111,083,920; `PI_label_05.tif` 222,169,294; `labeled_cells_05.tif` 222,167,935 | uint8 about 22,580 × 19,690 px (the contract's estimate from the placement of the stitching parity runs is 22,582 × 19,691); `_05` is a quarter: Y and X halved. Label images at 2 bytes per pixel. |
| cell-culture-3D (7 × 10 tiles of 30 × 1496², volumes) | `DAPI.tif`, `ER.tif`, `Flamingo.tif`, `ref_merged.tif`, `overlay.tif`: 5,493,369,191 to 5,493,370,046; `MAX_DAPI.tif` 130,794,660; `Cell.tif`, `Cyto.tif`, `Nuclei.tif`, `Cell_label.tif`, `DAPI_label.tif` (smaller, Dec 2023) | `DAPI_05.tif`, `ER_05.tif`, `Flamingo_05.tif`, `ref_merged_05.tif`: 1,373,491,604 to 1,373,492,384; `MAX_*_05.tif` 32,702,200; `Cell_05.tif`, `Nuclei_05.tif` 2,746,980,098 and 2,746,980,274 | Full stack about 42 planes of a 130.8 million-pixel uint8 plane (estimate 13,636 × 9,576); `_05` stacks are a quarter: Y and X halved, Z kept. `MAX_` is Fiji's name for a Z maximum projection, projected from the fused stack. |

% the placement of the stitching parity runs: W-324

So today's practice is: a fused stack per channel at full size, a maximum projection, the
same at half size in Y and X, fused label images, plain TIFF without calibration, linear
blending in overlaps, and positions from a Grid/Collection run rather than from the
placement the molecules use. LN sample1 has the 2024 BigStitcher placement but no fused
image is listed in the stitching input inventory.

% the W-323 inventory

## Proposal for each part

| Part | Proposal | Why |
| --- | --- | --- |
| `prepare_rasters` and its fidelity fallback | **Leave.** | It prepares rasters that are already aligned; fusion has to place FOVs, so its input does not exist. The halving fallback keeps native resolution whenever one small cell would vanish, which for a whole sample means a full-size image; the fused image instead has a fixed, requested scale factor. |
| The block-reduction geometry of `raster.py` (`i_source = f·i_output + (f − 1)/2`, spacing × f, origin + spacing·(f − 1)/2, float64 accumulation) | **Keep the geometry, leave the code**: the output grid of the fused image ({doc}`sample-output-contract`, "The scale factor") has this spacing and origin, and the block mean is multiview-stitcher's own reduction of each FOV. | The geometry is right and tested by literal cases; since the review gate (2026-10-06) the fused image is made by multiview-stitcher's `fuse`, so no block-mean code is rewritten. |
| The categorical block rule for labels (most frequent value, smallest ID on a tie) | **Leave.** The label export samples at interpolation order 0 without blending and lists the IDs lost at its scale factor instead of a fallback ({doc}`sample-output-contract`, "Assembled label export"). | Labels are placed by multiview-stitcher's `fuse` with a small non-blending function; a block mode would be a second label fusion. |
| `remap_labels` (unique uint32 IDs with a reversible map) | **Rewrite** as the instance-ID rule of the label export and of the SpatialData tables. | The accepted stitching and assembly design has sample-level cell keys from assembly; the ID is a row of the sample cell table, not a sort of label files. |
| `export_sample`: store-once rule, staging folder and rename, SHA-256 manifest and its verification, key validation, explicit missingness | **Bring over the rules, rewrite the code** against the sample tables. | The rules match the settled decisions; the inputs (`MoleculeIndex`, affine source mappings, a supplied H5AD) are replaced by the assembly result, which already holds coordinates, links and counts. |
| The rewrite of NGFF dataset transforms inside a spatialdata store | **Leave**; specify the transform once through spatialdata's own API and check it on read-back. | Editing spatialdata's metadata after writing made two conventions disagree (NGFF voxel centres against spatialdata's `[i, i + 1)`), which is the half-voxel problem the reference branch's specification recorded. |
| `lookup_export_trace` | **Rewrite** as a lookup from a molecule or cell key to its FOV's assignment folder through the assembly record. | The final-checkpoint trace of the reference branch's molecule outputs is not on `dev`; the assembly record holds every per-FOV source and hash. |
| `open_sample_viewer`, `open-in-fiji.json`, `README.html` | **Leave.** | Viewer qualification is deferred (Jiahao, 2026-10-05). |
| `export-viewer-requirements.txt` and the separate pinned environment | **Leave.** | The locked `spatialdata` extra replaces it; no dependency or lock change. |
| `test_raster.py`, `test_sample_export.py`, `sample_export_cases.json` | **Leave the files; reuse the ideas** (literal expected values, a regional read that touches fewer chunks, tampering) in the validation design ([stitching and assembly validation design](https://github.com/wanglab-broad/starfinder/blob/dev/design/validation/stitching-assembly.md)). | Their subjects are left or rewritten. |
| The population rules of the reference branch's specification (no new filtering, `unassigned` against `unavailable`, join by key, no recount) | **Bring over** as rules of the SpatialData export. | They are what "an export of the sample tables, not a second source" means. |
| `save_volume`, `load_volume` on `dev` | **Keep**; the fused OME-TIFF uses the same metadata annotation and must be readable by `load_volume`. | One convention for every saved image. |
| Today's fused files and Fiji Grid/Collection | **Leave** as legacy outputs; the new files go to `<OUTPUT>/export/<sample>/`. | Different placement, blending and calibration; no existing file is rewritten. |

% the W-329 review (2026-10-06); §2.10 has sample-level cell keys; the half-voxel problem W-168 recorded; its FOV's §2.9 assignment folder; the final-checkpoint trace (W-164); W-331's validation design; W-168 population rules
