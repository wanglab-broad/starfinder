# Compressed image outputs: TIFF + zlib by default for FOV checkpoints and saved sets

% Status: Approved (Jiahao, 2026-11-21, W-909): every recommendation ("Jiahao's answers" below). Written against `70edc72` (v0.3.0); the branch changes design records only, so every code reference on this page holds at `70edc72`.

This page specifies how the package compresses the image files it keeps. Jiahao decided on
2026-11-20 (W-881) that TIFF + zlib is the recommended way to store image data
("把TIFF + zlib 记成推荐数据存储方式"), and asked for a specification wherever the package can use it.

* **What changes.** The three uncompressed image outputs that the package keeps are written
  as TIFF + zlib (deflate, level 6, no predictor) by default:
  * the stage `registered` (`checkpoints/registered/<round>.ome.tif` and its snapshots), which
    is also what a saved spot-finding set holds;
  * the reference image (`fovs/<fov>/reference.tif`);
  * the prepared morphology images (`other_rounds/<name>/image.ome.tif`).
* **What does not change.** The outputs that are already deflate (segmentation and assignment
  labels, the segmentation input, the export's OME-TIFFs). The files MATLAB reads or writes. The
  short-lived files. The chunked HDF5 checkpoint of tiled execution. The synthetic writers.
  `save_volume`'s own default.
* **One key to opt out.** `execution.image_compression: zlib | none` in the workflow, and
  `CheckpointConfig.image_compression` in Python. The key is outside every task hash, so the
  codec is a storage detail: switching it reruns no task and refuses no resume.
* **Identity.** Records keep the file's SHA-256 as their integrity check. A comparison of
  content between two writes uses the decoded array. Written sets stay readable: an uncompressed
  set, such as W-903's, still loads and is still read by `processing.spot_finding_input`.

The validation design is [compressed-image-outputs validation](../validation/compressed-image-outputs.md).
The pages this one builds on are [checkpoints](checkpoints.md) (the layout, the stage records,
the reference image, the prepared images), [orchestration](orchestration.md) (the execution
section, the hashes, verified resume, the saved spot-finding input of W-889),
[orchestration-tiled](orchestration-tiled.md) (the chunked checkpoint) and
[sample-output](sample-output.md) (the export's OME-TIFF writer).

This is a software design. It makes no data claim. The sizes and times it quotes are the
measurements of W-881, W-908, W-910 and W-903, named where they are used.

## Jiahao's answers (2026-11-21)

Jiahao answered on 2026-11-21 (W-909 comment a38e3ab1; the review page's `decisions`): question 1 "agree", and the others "其他按推荐". The questions stay as they were asked, each with its answer.

1. **Accept the recommendations** of the eleven decisions in the table below?
   **Answer:** accepted.
2. **The scope of the default (decision 2).** The recommendation covers the stage `registered`,
   the reference image and the prepared morphology images. The reference image is included
   although it is small beside the stage (about 6 % of a tissue-2D FOV's stage): it is a
   declared output of every `fov` task and is kept. Include it, or keep the default to the
   checkpoint stages and saved sets only?
   **Answer:** include it (CO1).
3. **The key (decision 6).** `execution.image_compression` puts the codec beside the other
   settings that say how a run runs and not what it computes. The alternative,
   `processing.checkpoints` as a mapping, would put it inside the `fov` task's hash, so that
   switching the codec reruns every `fov` task. Is the name and the place right?
   **Answer:** accepted (K1).
4. **The kept chunked checkpoint (decision 9).** The recommendation keeps the HDF5 checkpoint
   of tiled execution uncompressed, also when the stage `registered` keeps it. A kept tiled
   checkpoint is then the one large image output left uncompressed. Accept, or map
   `image_compression: zlib` to its gzip option when it is kept?
   **Answer:** accepted: uncompressed (HC1).
5. **The BigStitcher tiles (decision 3).** `bigstitcher_save_dir/<sample>/tiles/tile_<j>.tif`
   are a declared output of the `bigstitcher_macro` task and are kept, although Fiji re-saves
   them as deflate HDF5 at once. The recommendation leaves them uncompressed, because
   BigStitcher's loader has not been shown to read deflate TIFF. Should a follow-up issue make
   them temporary, or check the loader and compress them?
   **Answer:** a follow-up issue, as recommended. Its direction was not chosen, so the follow-up carries both.
6. **The dense fields (decision 5).** LN's dataset configuration is E14's demons arm: a
   translation, then demons (`sample-dataset/LN/starfinder.yaml`; `docs/workflow/preparing-a-dataset.md`,
   "A configuration per dataset"). A stage `registered` written with it holds one float64 field
   per moving round, about 2.7 GB each by the shape (computed). That is more than the round images, 1.8 GB per FOV. The recommendation leaves the
   `.npz` fields as they are and reports their sizes in the validation (RD1). Should a follow-up
   issue measure `np.savez_compressed` for them?
   **Answer:** yes, a follow-up issue measures it; the fields stay as they are here.
7. **The real-data validation.** Its runs (sizes on six FOVs, paired times on two, the run from
   W-903's set) are larger than a batch may run (W-889 D3; W-903's tasks reached about 16 GB
   maxvmem each). The recommendation is a separate evidence issue on UGER, run by the
   coordinating session before the implementation merges. Accept?
   **Answer:** accepted.

## Decisions

Jiahao approved every recommendation on 2026-11-21. The column "Decision" holds what was approved. The
options and the evidence of each choice are in its section.

| # | Choice | Decision | Alternatives | Evidence | Section |
| --- | --- | --- | --- | --- | --- |
| 1 | The inventory | The table of every image file the package writes, with dtype, size per FOV, lifetime, MATLAB reader and codec | — | The recorded searches of the package's writers and their classification (repair run); W-903's set; shapes of W-910's inventories | [The inventory](#the-inventory) |
| 2 | Which outputs compress by default | CO1: every kept image output the package writes uncompressed today, except the files MATLAB reads or writes, the chunked checkpoint and the BigStitcher tiles: the stage `registered`, the reference image, the prepared morphology images. The Zarr outputs (I18, I19) keep their own lossless codec (ZA1) | CO2: the checkpoint stages only; CO3: every image file, short-lived ones included; ZA2: Zarr outputs to zlib | W-903's set: the stage is 18.1 of its 19.9 GB; W-908: the stage of tile_27 compresses to 0.063; I18 and I19 are Zstd-compressed already | [Which outputs compress](#which-outputs-compress), [The Zarr outputs](#the-zarr-outputs) |
| 3 | Short-lived local intermediates | SL1: stay uncompressed | SL2: compress them too | W-881: a warm read on UGER is faster uncompressed (tissue-2D 3.85 s against 6.60 s; LN 1.66 s against 2.29 s) | [Short-lived files](#short-lived-files) |
| 4 | The codec and its options | CD1: zlib (deflate) level 6, no predictor, through tifffile, as `save_volume(compress=True)` writes today | CD2: zlib with predictor 2; CD3: zstd level 3; CD4: another zlib level | W-881 (Tzlib against Tzstd and HDF5 gzip); W-908's verdict: predictor 2 does not join | [The codec](#the-codec) |
| 5 | Float data | FL1: float images take the same codec, without predictor 3; dense fields (`.npz`) are excluded and stated | FL2: predictor 3 for float images; FL3: float images uncompressed | No float output among the measured workloads; the float writers that compress today use zlib without a predictor | [Float data](#float-data) |
| 6 | Configuration and records | K1: `execution.image_compression` (`zlib`, the default, or `none`) and `CheckpointConfig.image_compression`; the codec recorded in `run.json`, in the stage record and in each image's record entry | K2: a key in `processing.checkpoints`; K3: no key | The hashes of `workflow/_shape.py` and `_plan.py`; the resume and saved-set checks of `workflow/_resume.py` | [Configuration](#configuration), [Records](#records) |
| 7 | Identity contracts | ID1: a record's hash is the file's SHA-256 (integrity); a comparison of content between two writes is by the decoded array; byte equality stays where both sides come from one writer, one setting and one environment | ID2: records also carry the decoded array's SHA-256 for every image; ID3: decoded identity everywhere | `_prepared.py` and the segmentation records already hold both; GN1 and GA1 already decode TIFFs | [Identity](#identity) |
| 8 | Compatibility | CP1: every reader takes either codec; the verified resume and `spot_finding_input` treat the codec as a storage detail; written sets are read as they are | CP2: refuse a set or a resume whose codec differs from the setting | `_resume.check` and `check_input` compare configuration and the set's own hashes, never a codec | [Compatibility](#compatibility) |
| 9 | The tiled chunked checkpoint | HC1: unchanged, uncompressed HDF5 by default, gzip level 1 an option of the Python writer | HC2: gzip 1 when `image_compression` is `zlib`; HC3: gzip 1 always | W-881: HDF5 gzip reads slower than uncompressed on UGER's NFS (tissue-2D 17.87 s against 15.11 s); orchestration-tiled: gzip 1 costs about 9 % of the global task | [The chunked checkpoint](#the-chunked-checkpoint) |
| 10 | MATLAB | MT1: the files MATLAB reads or writes keep their format | MT2: compress them as W-910 showed MATLAB reads deflate | AGENTS.md; W-910's readers check | [MATLAB](#matlab) |
| 11 | Docs and guidance | DG1: the implementation writes the guide and reference changes listed under "Pages that change"; "Preparing a dataset" says that inputs may be compressed TIFF; the shared development guidance changes at a batch boundary after the implementation | — | W-910: 2,456 deflate copies decode equal to their originals | [Docs and guidance](#docs-and-guidance) |

## Why

W-881 compared TIFF + zlib with compressed HDF5 and the uncompressed files on one FOV of each
dataset, on the host's share and on UGER's NFS:

* the bytes fall to 0.083 (LN Position002) and 0.138 (tissue-2D tile_27) of the uncompressed
  files;
* a cold read is as fast as compressed HDF5 on the host's share (0.99 of its reduction) and
  faster on UGER's NFS (LN 4.78 s against 6.84 s; tissue-2D 10.13 s against 17.87 s);
* the codec is standard TIFF, so the readers need no change.

W-908 then measured the stage `registered` itself: the four rounds of W-903's tile_27, written
by `save_volume`'s own ZYXC call, fall from 4,529,933,524 to 285,778,070 bytes (0.0631), at
22.03 CPU-s on one UGER core.

What the package compresses at `70edc72`:

* **Already deflate.** The export's OME-TIFFs: `io/_ome.py` writes 512 × 512 zlib tiles
  (`COMPRESSION = "zlib"`, `src/python/starfinder/io/_ome.py:27`), and `export/_check.py:196`
  requires it. The segmentation labels, the segmentation input and the assignment labels:
  `save_volume(..., compress=True)` (`src/python/starfinder/segmentation/_persist.py:57`, `:96`).
* **Uncompressed.** The stage `registered`, the reference image and the prepared morphology
  images: `save_volume` without `compress` (`src/python/starfinder/_checkpoint.py:328`,
  `src/python/starfinder/dataset/fov.py:2776`, `src/python/starfinder/dataset/_prepared.py:70`).
* **Optional.** The chunked HDF5 checkpoint takes gzip level 1, off by default
  (`src/python/starfinder/io/chunked.py:119`).
* `save_volume` has a `compress` flag that writes zlib (`src/python/starfinder/io/tiff.py:505`).

Since W-889, a run's stage `registered` is kept and read again by later runs
(`processing.spot_finding_input`). A stage is now stored and moved, not written once. W-903's
saved set of four tissue-2D tiles is 19.9 GB, of which the stage is 18.1 GB.

% Evidence: runs/W-881/20261120T1900Z-ctiff-h5 (README.md, results/tables.md); runs/W-908/20261120T2111Z-predictor (README.md, results/tables.md, records/inputs-registered.json); runs/W-903/20261120T1826Z-saved-set (README.md, set-manifest.json); runs/W-910/20261120T2302Z-zlib-copies (README.md, results/ratio-by-position.tsv)

## Terms

* A **kept output** is a file a run leaves in its output root for later steps or later runs: a
  checkpoint stage, a declared output of a task, a saved set.
* A **short-lived file** is written and read within one run and is not meant to outlive it: a
  temporary sibling before its rename, the chunked checkpoint when it is not kept.
* The **codec setting** is `zlib` or `none`. `zlib` means deflate at level 6 without a
  predictor, TIFF Compression tag 8 (Adobe deflate, as tifffile writes `"zlib"`).
* The **decoded identity** of an image file is the dtype, the shape and the SHA-256 of the
  decoded array's C-order bytes. `starfinder.io.chunked.array_sha256` and
  `dataset._prepared.image_sha256` compute that hash, and GN1 and GA1 hash the three together for
  a TIFF.
* The **file identity** is the SHA-256 of the file's bytes (`_records.file_sha256`).

## The inventory

Every image file the package writes at `70edc72`. The list comes from a recorded search of
`src/python/starfinder` for TIFF, Zarr, HDF5 and other array-to-file writes. Every hit of that search is in a table
of the repair run, either as a row below or as a write that is not an image, with the reason
(`runs/W-909/20261121T053747Z-71136cee/results/searches/` and `results/classification/package.tsv`). The writes that
are not images are tables (CSV, Parquet, H5AD), records (JSON), `.npz` archives, model files and rendered figures
(PNG, JPEG). A rendered figure is a picture of results, written in a format that is compressed already, and not
image data. Sizes are per FOV, uncompressed:

* "measured" sizes are files of W-903's set;
* "computed" sizes are the shape times the itemsize, without the TIFF header (W-910's
  inventories: tissue-2D ZYX 30 × 3,072 × 3,072 uint8; LN ZYX 50 × 1,496 × 1,496 uint8). Both
  are taken as four sequencing rounds of four channels: tissue-2D as W-903's set holds it, LN as
  its raw folders hold `ch00`–`ch03` in every round (W-881's LN workload of 17 TIFFs adds round 4's
  `ch04`).

"MATLAB" says whether MATLAB code reads or writes the file. For I6 to I9 the size is given by its formula,
because the shape depends on the run. Z is the run grid's planes, 30 (tissue-2D) or 50 (LN), or 1 after a Z projection,
as W-870's tissue-2D runs project. C is the model channels (1 to 3). A run's grid is its FOV's reference grid, 3,072 ×
3,072 or 1,496 × 1,496 pixels. The values are computed and were never measured, compressed or not.

(compressed-image-outputs-inventory)=

| # | Output | Path | Writer | Dtype | tissue-2D per FOV | LN per FOV | Lifetime | MATLAB | Codec at `70edc72` | Proposed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| I1 | Registered round | `fovs/<fov>/checkpoints/registered/<round>.ome.tif` | `_checkpoint.write_registered_round` → `save_volume` (`_checkpoint.py:328`) | as produced (uint8 on both datasets) | 1,132,483,381 B per round, 4,529,933,524 B for tile_27's four rounds (measured) | 447,603,200 B per round, 1,790,412,800 B (computed) | kept: a stage, a saved set | no | none | **zlib** |
| I2 | Registered snapshot | `…/registered/<snapshot>/<round>.ome.tif` | same | same | one round's size per round, with an extraction source; none in W-903's set | same | kept | no | none | **zlib** |
| I3 | Reference image | `fovs/<fov>/reference.tif` (`reference_image`) | `FOV.save_reference_image` (`dataset/fov.py:2776` ZYX, `:2780` YX with a projection) | the reference round's (uint8) | 283,120,910 B (measured) | 111,900,800 B (computed) | kept: a declared output of `fov` and `tile_global` | no (MATLAB writes `images/ref_merged/`) | none | **zlib** |
| I4 | Prepared morphology image | `fovs/<fov>/other_rounds/<name>/image.ome.tif` | `_prepared.write_image` (`dataset/_prepared.py:70`) | as produced | 1,132,462,080 B for the protein round of four channels (computed) | 335,702,400 B for the flamingo round of three channels (computed) | kept | no | none | **zlib** |
| I5 | Dense field | `…/registered/<round>_field.npz`, `other_rounds/<name>/field.npz` | `np.savez` (`_checkpoint.py:390`, `_prepared.py:83`) | float64, ZYX × 3 | 6,794,772,480 B per moving round with a dense step (computed); none with translation, as W-903 | 2,685,619,200 B per moving round (computed) | kept | no | none (`.npz`) | unchanged ([Float data](#float-data)) |
| I6 | Segmentation labels | `segmentation/<run>/labels.tif` | `segmentation._persist.write_labels` (`:57`) | uint32 | Z × 3,072 × 3,072 × 4 B: 37,748,736 B projected, 1,132,462,080 B in 3D (computed) | Z × 1,496 × 1,496 × 4 B: 8,952,064 B projected, 447,603,200 B in 3D (computed) | kept | no | zlib | unchanged |
| I7 | Segmentation input | `segmentation/<run>/input.ome.tif` | `segmentation/_persist.py:96` | the model input's: float32 (4 B) once a recipe step has run, else the source's (uint8, 1 B, on both datasets) | Z × 3,072 × 3,072 × C × itemsize: 37,748,736 B × C projected float32, 9,437,184 B × C projected uint8 (computed) | Z × 1,496 × 1,496 × C × itemsize: 8,952,064 B × C projected float32, 2,238,016 B × C projected uint8 (computed) | kept | no | zlib | unchanged |
| I8 | Assignment labels | `assignment/<name>/{territories,cell_labels,nucleus_labels}.tif` | `write_labels` (`assignment/_persist.py:82`, `:98`) | uint32 | per file as I6: `territories.tif` with an expansion; `cell_labels.tif` and `nucleus_labels.tif` only when the run is not saved under its own folder (computed) | as I6 (computed) | kept | no | zlib | unchanged |
| I9 | Preview model inputs | `<preview output>/inputs/<fov>/<recipe>.ome.tif` | `segmentation/_preview.py:631` | the model input's | as I7, per previewed FOV and recipe (the whole model input; `crop_um` only crops the figures) (computed) | as I7 (computed) | kept (opt-in, `save_inputs`) | no | zlib | unchanged |
| I10 | Morphology channel images | `images/<round>/<label>/<fov>.tif` (`morphology.channel_images`, off by default) | `workflow/_commands.py:378`, `:382` | as produced | one ZYX channel 283,115,520 B (computed) | 111,900,800 B (computed) | kept | yes: the MATLAB layout | none | unchanged ([MATLAB](#matlab)) |
| I11 | Export OME-TIFFs | `export/<sample>/<route>/…ome.tif`: fused images, `segmentation_<run>`, `cell_labels`, `territories`, `model_inputs/<run>` | `io/_ome.write_ome_tiff` (`export/_fuse.py:750`, `export/_segment.py:411`, `:596`, `export/_labels.py:549`, `:645`, `assembly/_persist.py:103`) | the image's | per sample on the fused grid, not per FOV | per sample on the fused grid, not per FOV | kept | no | zlib, 512 × 512 tiles | unchanged |
| I12 | Chunked checkpoint | `fovs/<fov>/tiled/registered.h5` | `io.chunked.ChunkedCheckpointWriter` (`io/chunked.py:119`; `dataset/tiled.py:373`) | as produced | about the stage's size | about the stage's size | until the merge; kept when `processing.checkpoints` holds `registered` | no | none; gzip 1 by option | unchanged ([The chunked checkpoint](#the-chunked-checkpoint)) |
| I13 | BigStitcher tiles | `<bigstitcher_save_dir>/<sample>/tiles/tile_<j>.tif` | `stitching/_bigstitcher.py:473` | the reference image's | one reference image per view | one reference image per view | kept: a declared output of `bigstitcher_macro`; Fiji re-saves them as deflate HDF5 | no (Fiji reads them) | none | unchanged ([Short-lived files](#short-lived-files)) |
| I14 | Synthetic scenes | `synthetic/_storage.py:60`, `:128`, `:361`, `:334` (a raw float32 field), `synthetic/_context.py:1701`, `:1703`, `:1777`, `:1786`, `synthetic/_sample.py:1249`, `:1256`, `:1259` | `save_volume`; `tifffile.imwrite` (ImageJ); raw bytes | as generated | not a dataset output: test-sized scenes | not a dataset output: test-sized scenes | kept: inputs and fixtures | no | none | unchanged |
| I15 | Legacy workflow outputs | `_legacy_workflow.py:50`, `:60`, `:110` | `tifffile.imwrite` | the legacy route's | not sized: the legacy route's outputs, which this page leaves unchanged | not sized, as tissue-2D | kept | the legacy route's layout | none; zlib for labels | unchanged |
| I16 | Benchmark trial arrays | `image.npy`, `labels.npy` (`benchmark/_trial.py:317`, `:336`) | `np.save` | the trial's | not sized: a benchmark trial's crop, not a FOV | not sized, as tissue-2D | kept | no | none (`.npy`) | unchanged: not TIFF, benchmark records |
| I17 | Temporary siblings | `.<uuid>.tmp.<name>` and `atomic_file` names | every writer above | the final file's | the final file's size | the final file's size | renamed into place | — | the final file's | the final file's |
| I18 | Standalone OME-Zarr of a fused image | `export/<sample>/<route>/<image>_{projection,stack}.ome.zarr` (`FusionConfig.ome_zarr`, off by default; refused with the SpatialData export) | `export/_fuse.py:755` → `multiview_stitcher.ngff_utils.write_sim_to_ome_zarr` (`:765`) | the fused image's | the fused grid's, per sample | the fused grid's, per sample | kept (opt-in) | no | Zarr: Zstd level 0 (sample-output, W-354), chunks t, c, (z), y, x of (1, 1, 1024, 1024) or (1, 1, 16, 512, 512) (`export/_fuse.py:65`); the codec is recorded in the `fusion` section of `export.json`, in the output entry of format `ome_zarr` (`export/_fuse.py:919`): `levels[].codec` per level and `compression` (`:777`–`:779`, `:789`), and in the store's own `zarr.json` of each level | unchanged ([The Zarr outputs](#the-zarr-outputs)) |
| I19 | SpatialData store | `export/<sample>/<route>/sample.zarr` (`export_spatialdata`) | `export/_spatialdata.py:710` → `SpatialData.write` (spatialdata 0.6.1) → ome-zarr 0.12.2's writer | the fused images' and labels' | the fused grid's, per sample | the fused grid's, per sample | kept | no | Zarr format 3 (the default `zarr_format`): zarr-python 3.1.5's default ZstdCodec, level 0 (`zarr/core/array.py:5086`); format 2: Blosc zstd level 5 with shuffle (`ome_zarr/writer.py:175`, `:277`). `export.json` records no codec for it: its `spatialdata` section gives each raster element `levels` as the integer 1 (`export/_spatialdata.py:437`), `array_metadata`, the path of the level's `zarr.json` or `.zarray`, and `array_metadata_sha256` (`:523`–`:525`), and the store's file map and its SHA-256 (`:721`). The codec is in that array metadata | unchanged ([The Zarr outputs](#the-zarr-outputs)) |

* I12's HDF5 is the only HDF5 the package writes with images (search p3); `sample.h5ad` (AnnData) holds tables.
* W-903's set holds, per tile, four I1 files (18.12 GB for the four tiles), one I3 file and the
  tables. Its other kept files (`candidates.csv`, `pre_qc.csv`, `molecules.csv`, the records)
  are tables and JSON, outside this page.
* The ratio of one FOV does not carry to a dataset. W-910 copied every raw TIFF of both datasets
  with zlib: LN came to 37.5 % (W-881's Position002 at 8.3 % is the third most compressible of 64
  positions), tissue-2D to 14.8 % (tile_27 at 14.6 % is 25th of 56). The stage `registered` was
  measured on tile_27 only (0.0631). Any projection for a dataset uses a sample of positions
  (validation RD1).

% W-910 results/ratio-by-position.tsv: tissue-2D tile_49 0.1098 (rank 1 of 56), tile_27 0.1456 (25), tile_23 0.1973 (56); LN Position006 0.0544 (1 of 64), Position010 0.4116 (32), Position046 0.5346 (64)

## Which outputs compress

| Option | Rule | Consequence |
| --- | --- | --- |
| **CO1. Every kept image output the package writes uncompressed, with three exceptions (recommended)** | I1–I4 compress by default: I1, I2, I3 and I4 when the image is an integer or float TIFF (I5 is `.npz`, see "Float data"). The exceptions: the files MATLAB reads or writes (I10, I15), the chunked checkpoint (I12) and the BigStitcher tiles (I13). The Zarr outputs (I18, I19) are compressed already and keep their codec ("The Zarr outputs"). | The stage shrinks about 16-fold on tissue-2D tile_27 (W-908). W-903's set of 19.9 GB would come to at most about 3 GB if every tile's stage compressed like tile_27's, counting the reference images uncompressed (an inference, not a measure). Every kept image the Python route writes is then deflate except I12 and I13, each with its reason. |
| CO2. The checkpoint stages only | I1, I2 compress; I3 and I4 stay uncompressed | The reference image (about 6 % of the stage on tissue-2D) and the morphology images stay large. The rule is narrower than "every persisted image output". |
| CO3. Every image file | CO1 plus I10, I12, I13 and the short-lived files | Changes MATLAB's files (decision 10), the chunked checkpoint's window reads (decision 9) and the BigStitcher input; adds decode time to files read once. |

**Recommendation: CO1.**

* The synthetic writers (I14) are not outputs of a run: they write inputs and test fixtures.
  Two tests flip a byte inside the first strip of a synthetic raw image and assert that it is
  uncompressed (`test_workflow_nextflow_cache.py:94`, `test_workflow_recovery.py:537`). Their
  files stay as they are.
* `save_volume`'s default stays `compress=False`. It is a public function that users and the
  synthetic writers call. The package's writers I1–I4 pass the codec setting explicitly.
  `compress=True` writes zlib with `compressionargs={"level": 6}` given explicitly. tifffile's
  default level is 6 with the locked imagecodecs (the repair run's `codec-check`), so the files that writers I6–I9 write today keep
  their bytes (validation CW4).

### The Zarr outputs

The export writes two Zarr outputs (I18, I19). Both are chunked stores whose chunks are compressed by the Zarr codec
the writing library chooses, and both are read by Zarr readers (napari, spatialdata), not as TIFFs.

| Option | Rule | Consequence |
| --- | --- | --- |
| **ZA1. This specification does not govern them; each keeps its writer's codec (recommended)** | I18: Zstd level 0, as `multiview_stitcher` writes it, recorded per level in `export.json` (`levels[].codec`, and `compression`) and checked by the export's read-back (F1, `export/_check.py:315`, `:349`). I19: zarr-python's default ZstdCodec (level 0) at Zarr format 3, the default, or Blosc zstd level 5 with shuffle at format 2, as spatialdata and ome-zarr write it. `export.json` holds no codec for I19, only `array_metadata` and its SHA-256 per raster element (`export/_spatialdata.py:523`–`:525`); the codec is in that store's own array metadata, and the read-back checks the store's files and their hashes (S8, `export/_check.py:1116`). | Both are lossless and compressed already. `image_compression` does not apply to them. |
| ZA2. Write them with a zlib (gzip) codec too | Pass a compressor to both writers | One codec name for every output, but a change to libraries' own writers that sample-output decided not to edit ("no file of the library is edited"), no evidence that zlib beats Zstd on Zarr chunks, and a new read-back rule. |

**Recommendation: ZA1.** "TIFF + zlib" is a decision about TIFF files. A Zarr store's chunks are already compressed
with Zstd, a codec that W-881 and W-908 found 5 to 9 % larger and 2.5 to 3.6 times faster to write than zlib on TIFF. The store is
written once per sample by the export and is not a checkpoint or a saved set. I18 is off by default.

## Short-lived files

| Option | Rule | Consequence |
| --- | --- | --- |
| **SL1. Short-lived files stay uncompressed (recommended)** | Temporary siblings take the codec of their final file (they are that file before its rename). The chunked checkpoint stays as decision 9 says. Nothing else short-lived is written as an image. | A file read once, usually from a warm page cache, is not decoded. |
| SL2. Compress them too | As CO3 | Write CPU and decode time for files that are deleted. |

**Recommendation: SL1.** W-881 measured a warm read on UGER, where the page cache serves it:
uncompressed is faster (tissue-2D tile_27 3.85 s against 6.60 s with zlib; LN Position002
1.66 s against 2.29 s). A compressed file pays off when it is stored or moved, which a
short-lived file is not.

The BigStitcher tiles (I13) are not short-lived: the plan declares `tiles/` an output of
`bigstitcher_macro` and nothing removes it (`workflow/_plan.py:756`–`758`). They are left
uncompressed for another reason. Fiji reads them through BigStitcher's "Automatic Loader
(Bioformats based)" and re-saves them as deflate HDF5. W-910 showed that Bio-Formats 8.0.0 opens
deflate TIFFs with `BF.openImagePlus`, but BigStitcher's dataset definition was not run on
them. Question 5 asks whether to make them temporary or to check the loader.

## The codec

| Option | Bytes (tissue-2D tile_27, × source) | Write CPU-s | Cold read on UGER's NFS | Consequence |
| --- | --- | --- | --- | --- |
| **CD1. zlib level 6, no predictor (recommended)** | registered 0.0631; raw 0.1383 | 22.03 (registered); 40.35 (raw) | registered 12.29 s; raw 9.81 s | The standard TIFF codec: MATLAB (`Tiff`, `imread`), ImageJ and Bio-Formats read it (W-910). The export already writes it. |
| CD2. zlib with predictor 2 | registered 0.0688 (larger); raw 0.1372 | 25.54; 45.91 | 14.53 s; 12.05 s | W-908's rule: a cut of at least 10 % on two of three workloads, a read rise of at most 5 %. Both conditions fail. |
| CD3. zstd level 3 | registered 0.0687; raw 0.1480 | 8.91; 11.63 | 13.41 s; 11.18 s | Writes 2.5–3.6 times faster; files 5–9 % larger and reads slower. The readers check of W-910 covers deflate only. |
| CD4. Another zlib level | not measured | — | — | No evidence; level 6 is zlib's default and W-881's arm. |

**Recommendation: CD1.** W-908's verdict decides the predictor: "Predictor 2 does not join the
zlib recommendation. Both conditions of the rule fail." On these uint8 images, horizontal
differencing breaks the long zero runs that deflate already compresses well.

* The level is fixed at 6 and not a key: the evidence covers level 6 only.
* The writer is tifffile with the imagecodecs of the lock (tifffile 2026.1.28, imagecodecs
  2026.1.14 in W-881 and W-908). The strips are about 256 KiB (tifffile's `262144 // rowsize`
  rows; 85 rows of a 3,072-pixel tissue-2D plane, 175 of a 1,496-pixel LN plane), as in W-910's copies.
* The repair run's step `codec-check` checked these facts in the locked environment (tifffile 2026.1.28, imagecodecs
  2026.1.14 with zlib 1.3.1):
  * imagecodecs' default zlib level and tifffile's `compression="zlib"` both write the bytes of level 6;
  * the strips hold 85 and 175 rows;
  * the files carry Compression 8 and no Predictor tag.

  The results are in `runs/W-909/20261121T053747Z-71136cee/results/codec/codec_check.json`.

## Float data

| Option | Rule | Consequence |
| --- | --- | --- |
| **FL1. Float images take the same codec without a predictor; dense fields are excluded (recommended)** | I1–I4 compress float32 and float64 images with zlib level 6 and no predictor, as for integers. The `.npz` dense fields (I5) are not changed. | One codec for every dtype, the rule the float writers I7 and I11 (`model_inputs` after fusion) already follow. How well float images compress here is not measured. |
| FL2. Predictor 3 (floating point) for float images | TIFF predictor 3 on float32 and float64 | Usually smaller float files; not measured on this package's data. |
| FL3. Float images uncompressed | `none` for a float dtype | A second rule by dtype, without evidence for it. |

**Recommendation: FL1.**

* None of the measured workloads is float. W-908: "Predictor 3 (floating point) does not apply to
  uint8."
* I1–I4 are float only when the pipeline produces float images. Both datasets produce uint8.
* Validation RD5 measures predictor 3 on float32 model inputs, by W-908's rule. If it meets the
  rule, a later issue can adopt FL2 for float dtypes without changing this page's key.
* The dense fields are float64 `.npz` archives, not TIFFs. Only a dense or B-spline registration
  step writes them. By the shape, a tissue-2D FOV's stage with such a step would hold 6.8 GB per
  moving round in them, and LN's, whose dataset configuration is a demons arm, 2.7 GB per moving
  round. `np.savez_compressed` is the natural analogue. It is left to a later issue with its own
  measurement (question 6), because no field was measured and float displacement fields are not
  expected to compress like the sparse uint8 images measured here. RD1 reports their sizes.

## Configuration

| Option | Where | Hashes | Consequence |
| --- | --- | --- | --- |
| **K1. `execution.image_compression` and `CheckpointConfig.image_compression` (recommended)** | The workflow's `execution` section; the Python `CheckpointConfig` | Outside `blocks_sha256`, `configuration_sha256` and the task fingerprint (`workflow/_shape.py:197`, `:205`; `workflow/_plan.py:1032` leaves `execution` out) | Switching the codec reruns no finished task and refuses no resume. Old and new files mix within one output root, each recorded with its codec. |
| K2. `processing.checkpoints` as a mapping (`stages`, `image_compression`) | The processing block | Inside the `fov` task's `blocks_sha256` | Switching the codec reruns every `fov` task and refuses every resume (`_resume.check` compares `blocks_sha256`). A storage detail becomes part of the result's identity. |
| K3. No key | — | — | No opt-out. W-881: uncompressed reads faster from a warm local cache, so a run on local scratch may prefer it. |

**Recommendation: K1.**

```yaml
execution:
  image_compression: zlib     # zlib (default) or none
```

```python
@dataclass(frozen=True)
class CheckpointConfig:
    stages: tuple[str, ...] = ('registered', 'candidates', 'pre_qc')
    directory: Path | str | None = None
    table_format: str = 'csv'
    hash_inputs: bool = True
    overwrite: bool = False
    image_compression: str = 'zlib'   # 'zlib' or 'none'
```

* **Values.** `zlib` and `none`. Anything else is refused: `PlanRefused` (exit 2) at plan time,
  naming `execution.image_compression` and the two values; `ValueError` in
  `CheckpointConfig.__post_init__`.
* **Default.** `zlib`, in the workflow and in Python. The plan writes the default into the
  resolved configuration's `execution` section, as it writes the other execution defaults.
* **Who reads it.**
  * `FOV.run` writes I1 and I2 with `checkpoints.image_compression`.
  * `FOV.prepare_morphology` and `FOV.register` write I4 with the `checkpoints` they are given.
  * `FOV.save_reference_image` gains the keyword `image_compression: str = 'zlib'`.
  * The `fov`, `tile_global`, `tile_merge` and `morphology` commands pass
    `execution.image_compression` to all three. Direct and tiled therefore write the same
    reference image bytes (`test_workflow_tiled_tasks.py:97` keeps checking it).
* **Placement.** `execution` is the section every task kind reads besides its blocks
  (`ConfigurationView`, `workflow/_shape.py`); `EXECUTION_KEYS` gains `image_compression`. The
  commands read it from the resolved configuration. It is not copied into a task's `execution`
  entry of the manifest, which holds the task's resources.
* **The precedent.** orchestration chose `execution.resume_from_checkpoint` for verified resume
  with the same reason: "Resuming never changes a result, so it stays outside the configuration
  hash" (`design/specs/orchestration.md`, decision D8).
* **Not a processing setting.** The key never changes an array, and its records show which codec
  each file has. A run that switches it keeps its finished tasks, and a resume keeps the stage it
  resumes from with that stage's codec ([Compatibility](#compatibility)).

## Records

Every record that names an image file says which codec that file has. Each addition is a new
key; no record's format version changes.

| Record | Key added | Value |
| --- | --- | --- |
| `run.json` `config.checkpoints` | `image_compression` | The `CheckpointConfig` field as given |
| `checkpoints/registered/transforms.json` (the stage record) | `image_compression` (top level) | `zlib` or `none`: the codec of every round image and snapshot it lists. A stage record without the key was written before this page and is uncompressed. |
| `run.json` `outputs.reference_image` | `compression` | `zlib` or `none` |
| `other_rounds/<name>/registration.json` `image` | `compression` | `zlib` or `none`; beside its `sha256` (decoded) and `file_sha256` |

* The stage record's `files` entries stay `{path, sha256}`. `payload_entries` at `70edc72`
  refuses an entry with any other key (`_checkpoint.py:239`), so a new key there would make
  v0.3.0 unable to read a new set. No v0.3.0 reader checks the stage record's top-level keys as a
  set; each reads the keys it needs. Validation CR3 checks that v0.3.0's reader loads a new stage.
* The stage record's format version stays 3, and the prepared image record's stays 1. A reader
  takes a record without the key as uncompressed. A reader never needs the key: tifffile reads
  the codec from the file.
* The plan's resolved configuration holds `execution.image_compression`. The task record's
  `workflow` section is unchanged.

## Identity

| Option | Rule | Consequence |
| --- | --- | --- |
| **ID1. File identity for records, decoded identity for comparisons of content (recommended)** | Every record keeps the file's SHA-256 as its integrity check: it detects a damaged or replaced file, and the record is written with the file. Content is compared between two writes (two runs, two routes, two revisions, two environments, two codecs) by decoded identity. Byte equality stays where both sides come from one writer, one setting and one environment, AGENTS.md's same-environment repeatability. | No record changes its rule; values change with the codec. The prepared image and segmentation records already hold both hashes, and GN1 and GA1 already decode TIFFs. Compressed bytes depend on the zlib build, so a file hash of a compressed image is an identity of one environment only. |
| ID2. Records also carry each image's decoded SHA-256 | The stage record gains a decoded hash per image; so does `outputs.reference_image` | A cross-codec comparison without a separate decode. Hashing about 4.5 GB more per tissue-2D FOV; a `files` key that v0.3.0 refuses unless placed elsewhere. |
| ID3. Decoded identity everywhere | Records switch to decoded hashes | A changed header, tag or metadata goes undetected, and every record and reader changes. |

**Recommendation: ID1.**

**Where each contract stands.**

(compressed-image-outputs-identity)=

| Contract | What it hashes | Under ID1 | Change |
| --- | --- | --- | --- |
| Stage records' `files` (`_checkpoint.payload`, checked before a read) | File | File | Values change with the codec; rule unchanged |
| `run.json` `checkpoints.<stage>.sha256` | The stage record's bytes (JSON) | File | The `registered` record's bytes change (its new key) |
| `run.json` `inputs[].sha256` | Raw inputs (file) | File | None: the package does not write its inputs |
| `run.json` `outputs.reference_image.sha256` | File | File | Value changes; `compression` added |
| Prepared image record (`_prepared.py:88`) | `sha256` decoded and `file_sha256` | Both, as today | `file_sha256` value changes; `compression` added |
| Segmentation and assignment label records | Decoded `sha256` and `file_sha256` | Both, as today | None (already zlib) |
| Stitching inputs (`placement.json` `inputs.<fov>.file_sha256`) | The reference image file | File | Value changes |
| Export records (`export.json` fusion sources) | Source files | File | Value changes for reference images |
| Verified resume (`workflow/_resume.py:45`) | `blocks_sha256`, code, inputs, and the earlier attempt's own stage hashes | Unchanged | None: the codec is outside `blocks_sha256` |
| Saved set (`check_input`, `workflow/_resume.py:261`) | The set's stage record against its `run.json`; `INPUT_FIELDS` | Unchanged | None: the codec is not compared |
| Chunked checkpoint (`array_sha256`, `prepare_global`) | Decoded | Unchanged | None |
| W-889 set manifest (W-903's `set-manifest.json`, a run-folder script) | Files | Files | A set's own manifest lists its files; a set and a re-encoded copy are compared by decoded identity, as W-910 compared its copies |
| A19 parity (`HASH_SCRIPT`, `test_assignment.py:808`) | Assignment tables | Unchanged | None: no image |
| GN1, GA1 (`test_synthetic_sample_golden.py`, `test_synthetic_appearance_golden.py`) | Decoded TIFF arrays; CSV; JSON without `files` | Unchanged | None: the synthetic writers do not change |
| Array goldens (registration, preprocessing, segmentation, readout, spot finding, assignment) | In-memory arrays and tables | Unchanged | None |
| X4 core golden (`test_workflow_core_golden.py`) | Records across environments, image statistics decoded; file digests skipped across environments | Unchanged rule | Its records gain keys: migration M2 |
| G3 orchestration golden (`test_workflow_orchestration_golden.py`) | As X4; OC3 explains reference image digests | Unchanged rule | Records gain keys; OC3's reasons name the codec: migrations M3, M4 |

**Tests and records that change.** No expected value changes silently. Each change below is
made by the implementation, with the reason in the test.

(compressed-image-outputs-migration)=

| # | Test or record | Today | After | Why |
| --- | --- | --- | --- | --- |
| M1 | `test_report_records_golden.py`: the expected `run.json` `config.checkpoints` | `stages`, `directory`, `table_format`, `hash_inputs`, `overwrite` | the same and `image_compression: "zlib"` | The new `CheckpointConfig` field (decision 6) |
| M2 | `test_workflow_core_golden.py` (X4): the comparison of `records/fovs/<fov>/run.json` | No difference | `config.checkpoints.image_compression` and `outputs.reference_image.compression` only in actual, explained by a named exception "W-909" beside OC3, OC4 and D8 | The expected folder was recorded by W-734 and is updated only by its named decisions. The new keys are named, not written into it. |
| M3 | `test_workflow_orchestration_golden.py` (G3), `scripts/workflow_golden.py` `EXCEPTIONS` | — | A "W-909" exception: the same two keys only in actual | As M2 |
| M4 | `scripts/workflow_golden.py` `EXCEPTIONS`, the three OC3 rows on reference image digests (`run.json` `outputs.reference_image.sha256`, `placement.json` `file_sha256`, `export.json` `reference_merged.sha256`) | Reason: "its stored metadata holds the frame id" | Reason: the frame id and, since W-909, the codec | The pattern already explains the difference; its reason gains the second cause |

**The tests and records, by a recorded search.** Three searches of `src/python/test` were run and kept in the
repair run (`runs/W-909/20261121T053747Z-71136cee/results/searches/`):

* t1: every line with `hashlib`, `sha256` or `digest`, 1,221 lines;
* t2: every byte comparison of files (`read_bytes() ==`, `filecmp`), 58 lines;
* t3: every file under `test/data` that holds a SHA-256, 49 files.

Every line and every hash value is classified in `results/classification/` (`tests.tsv` and
`expected-records.tsv`). It is the hash of an image file, the hash of a record or a non-image file, the digest of a
decoded or in-memory array, or no hash value (imports, helper definitions, docstrings, messages, key names,
placeholders). Lines inside a string that a test runs as code in a second process count as code (the
classification's second attempt, `scripts/classify_v2.py`). For an image file, the table gives three more things. First, whether this page changes the codec of
its writer, I1–I4 or another writer. Second, how the test compares. Third, the effect.

| Class (t1 and t2, 1,279 hits) | Hits |
| --- | --- |
| Hash or bytes of an image file | 179 (54 of a file whose writer this page changes, 13 of folders holding both kinds, 112 of other writers' files) |
| Hash of a record or a non-image file | 446 |
| Digest of a decoded or in-memory array or table | 220 |
| No hash value | 434 |

**The tests that hash a file this page changes (I1–I4), and why each passes unchanged.**

(compressed-image-outputs-tests)=

| How the test compares | Tests (file: lines) | Why it holds |
| --- | --- | --- |
| A record against the file it names, written together (`self`) | `test_workflow_fov_tasks.py:248`; `test_workflow_completion.py:100`; `test_run_record_version_2.py:116`, `:129`; `test_report_chain.py:406`, `:407`, `:822`, `:824`; `test_report_collection.py:660`; `test_dataset_fov_layout.py:192`; `test_morphology_rounds.py:397`, `:424`, `:425`, `:427`, `:581`; `test_segmentation_saved_images.py:206`, `:247`, `:375`, `:378`, `:397` | The test hashes the file it reads and the record holds the hash the writer took of that file. Both values change with the codec, together. |
| One file before and after an operation that must not rewrite it (`same-file`) | `test_dataset_fov_layout.py:50` (with `:126`–`:136`); `test_report_page.py:823`; `test_workflow_commands.py:49`; `test_workflow_declared_grids.py:139`, `:280`, `:296`; `test_sample_segmentation_compare.py:90`, `:295`, `:304`, `:320`, `:429`, `:435` | The same file on both sides |
| Two writes by one writer and one setting in one environment (`two-writes`) | `test_workflow_recovery.py:63`, `:97`, `:105`, `:114`, `:120`, `:129`, `:196`, `:202`, `:221`, `:228`, `:239`, `:257`, `:265`, `:279` (`:221` and `:228` include the registered rounds, a full run against a stop and a resume from the stage registered); `test_workflow_spot_finding_input.py:73`, `:91`, `:122`, `:132`; `test_workflow_tiled_tasks.py:97`; `test_morphology_rounds.py:148`; `test_workflow_declared_grids.py:159`; `workflow_engine_validation.py:212`, `:441`, `:442` (Snakemake against Nextflow) | Both sides take the default setting, and zlib at level 6 writes equal bytes for equal input in one environment (validation CW4, CW6). These are the checks that every route passes the setting (decision 6). |
| A value fixed before the run (`pinned`) | `test_workflow_orchestration_golden.py:207`, `:209`, `:210` (the cases "reference image digest" and "report input digest of a reference image"); `:314` (a fixture edit of a reference image's digest, which tests the report's verification and compares no file) | G3 compares across environments, where OC3 already explains a difference of these digests. Migration M4 adds the codec to OC3's reason. |

The review's examples are in the classification:

* `test_workflow_fov_tasks.py:248` compares run.json's `outputs` with their files. `reference.tif` (I3) is one of them,
  so the row is `self` and the effect `unchanged`.
* `test_sample_segmentation_recipe.py:141` compares a model input OME-TIFF of the export (I11, writer unchanged) with
  `export.json` and the run record (`self`).
* `test_segmentation_persistence.py:81` and `:115` compare `labels.tif` (I6) and `input.ome.tif` (I7) with their
  record. Both writers compress already and do not change.

**The expected records (t3).** The walk of every JSON file under `test/data` found 1,010 hash values:

* 547 hash an image file;
* 395 hash records or tables;
* 66 are decoded `sha256` values of grids and records;
* 2 are not hash values.

Of the image-file hashes, 24 name a file this page changes. Each is the reference image (I3) in G3's expected folders:

* `values.json` `file:fovs/<fov>/reference.tif`, 4, `same_environment`: skipped across environments
  (`evaluation/workflow.py:114`);
* `run.json` `outputs.reference_image.sha256`, 4;
* `export.json` `fusion.fovs[].images.reference_merged.sha256`, 12, three routes of four FOVs;
* `report.json` inputs of kind `reference_image`, 4.

OC3 explains the last three groups across environments (M4). X4's expected folder holds no file hash of an I1–I4 file.

Of the other 523, 522 are hash values of files whose writer does not change:

* 444 are the fixture's raw inputs, written by the synthetic writers (I14). G3 and X4 compare them exactly in one
  environment and skip them across environments. This is one more reason why the synthetic writers stay as they are
  (decision 2).
* 78 are export and label TIFFs (I6, I8, I11).

The last one is X4's recorder script (`record_core_values.py`), which writes both kinds.

**Other tests that stay.**

* Uncompressed raw inputs, which the synthetic writers keep: `test_workflow_nextflow_cache.py:94`,
  `test_workflow_recovery.py:537`. These two assert Compression 1 and are not hash comparisons, so the searches
  did not find them.
* `test_io.py::test_save_with_compression`: `save_volume`'s default is unchanged.
* `test_ome_tiff.py:211`, `:261`–`:301` and `test_stitching_ome_uuid.py`: process determinism and the content UUID of
  `save_volume`'s and `write_ome_tiff`'s files. Their writers' defaults do not change.
* The C1 plan manifests (`data/expected/test_workflow_plan/`): a task's `execution` entry holds its resources only,
  and the resolved configuration's hash is masked there. Validation WF1 confirms that they do not change.

## Compatibility

| Option | Rule | Consequence |
| --- | --- | --- |
| **CP1. The codec is a storage detail for every reader and check (recommended)** | Every reader takes zlib and uncompressed files: tifffile decodes either, and the header readers (`_volume_header`, `_zyxc_header`) read tags only. The verified resume and `spot_finding_input` compare configuration and the set's own hashes, never a codec. | W-903's uncompressed set loads and is read by `processing.spot_finding_input` unchanged. A resume after switching the codec continues. v0.3.0 reads a new compressed stage (CR3). |
| CP2. A set or a resume must have the current codec | Refuse, or rewrite, a stage of another codec | W-903's set would be refused or rewritten, for no change in its content. |

**Recommendation: CP1.**

* **A resume after a switch.** A `fov` task interrupted under `none` and resumed under `zlib` passes
  `_resume.check`: the setting is outside `blocks_sha256`. The kept stage stays uncompressed, and
  its stage record says so (`image_compression: none` or no key). The files written after the
  resume take the new setting, and `run.json` records the setting of the attempt that wrote it.
* **A saved set.** `check_input` compares the set's `run.json` with the task's `INPUT_FIELDS`
  (`load`, `rotation_degrees`, `preprocessing`, `registration`) and the set's stage record with
  its own hash. A set of either codec, or one written by v0.3.0 or by W-889's `9640410`, is read
  as it is. The run that reads it writes its own outputs with its own setting.
* **No conversion in place.** Recompressing an existing set would change its file hashes and
  therefore its stage record and `run.json`, which `check_input` checks. This page provides no
  tool for it. A set is made compressed by making it again. A converter that rewrites the
  records with the decoded identity checked first would be a separate issue.
* **The fusion and the stitching** read the reference image page by page or whole
  (`export/_fuse.py:451`, the `TileLoader`), which a zlib file serves like an uncompressed one,
  one decode per page.

## The chunked checkpoint

| Option | Rule | Consequence |
| --- | --- | --- |
| **HC1. Unchanged: uncompressed by default, gzip level 1 an option of the Python writer (recommended)** | `ChunkedCheckpointWriter(compression=None)` as today; `image_compression` does not apply to it | The tiles keep reading their windows without decoding; a kept tiled checkpoint is uncompressed. |
| HC2. gzip 1 when `image_compression` is `zlib` | The tiled tasks pass gzip to the writer | One rule for every kept image, at the cost below on every tiled run. |
| HC3. gzip 1 always | Change the writer's default | As HC2, also for runs that delete it after the merge. |

**Recommendation: HC1.**

* W-881 found h5py's gzip slow on UGER's NFS. The compressed HDF5 cold read was slower than the
  uncompressed files on tissue-2D (17.87 s against 15.11 s) and barely faster on LN (6.84 s
  against 7.29 s), while TIFF + zlib was 33 % and 34 % faster.
* orchestration-tiled measured gzip 1 at about 9 % of the global task on a real FOV (93 s against
  983 s, nine rounds).
* The checkpoint lives until the merge unless `registered` is kept, and its windows are read once
  per tile.
* Question 4 asks whether a kept checkpoint should still compress (HC2).

## MATLAB

| Option | Rule | Consequence |
| --- | --- | --- |
| **MT1. The files MATLAB reads or writes keep their format (recommended)** | I10 (`images/<round>/<label>/<fov>.tif`, the MATLAB-era layout that `morphology.channel_images` writes) and I15 (the legacy route) are unchanged. | AGENTS.md: the files, keys and conventions MATLAB reads or writes stay unchanged. |
| MT2. Compress them too | As I1–I4 | W-910 read one deflate TIFF per dataset in MATLAB R2023b (`Tiff`, `imread`) equal to tifffile. That shows that MATLAB loads deflate, but these files are off by default and few, so the change gains little. |

**Recommendation: MT1.** None of I1–I4 is read or written by MATLAB. The MATLAB route writes its
own reference image, `images/ref_merged/<fov>.tif`, and reads no `fovs/<fov>/` file.

## Docs and guidance

* **Preparing a dataset** (`docs/workflow/preparing-a-dataset.md`): the inputs may be compressed
  TIFF. `load_round` reads any codec tifffile decodes, and W-910's 2,456 deflate copies decode
  equal to their originals. The implementation issue writes the sentence; this issue writes no
  `docs/` page.
* **The shared development guidance.** After the implementation lands, it recommends TIFF + zlib
  (level 6, no predictor) for large image evidence in run folders. It is a private configuration
  file, changed at a batch boundary by the coordinating session.

## Pages that change

When the implementation lands:

* `design/specs/checkpoints.md`: "Layout" (the codec of `registered/<round>.ome.tif`,
  `reference.tif` and `other_rounds/<name>/image.ome.tif`), "registered" (the record's
  `image_compression`), "Prepared morphology images" and "The reference image" (their codec and
  records), "run.json" (`config.checkpoints`, `outputs.reference_image`).
* `design/specs/orchestration.md`: the `execution` row of the configuration shape (line 682) and
  the decisions table beside "Verified resume (D8)".
* `design/specs/orchestration-tiled.md`: "Today's subtile and deep routes", which describes the
  stage `registered` as "one uncompressed ZYXC OME-TIFF per round".
* `docs/checkpoints-reference.md`, `docs/workflow-configuration.md` (the execution table),
  `docs/workflow/preparing-a-dataset.md`, `docs/workflow/index.md` ("Spot finding from a saved
  set": a set of either codec is read), `docs/migration.md` (the new default of `FOV.run`'s
  checkpoints).
* `src/python/starfinder/workflow/schema.json` (the key).

## Exclusions

* No conversion of existing sets or run folders, and no change to W-903's set.
* No change to the export's OME-TIFF writer, to its 512 × 512 tiles or to `export/_check.py`.
* No zstd, no predictor and no level other than 6 (decision 4).
* No compression of the `.npz` dense fields (decision 5), of the BigStitcher tiles (question 5) or
  of the chunked checkpoint (decision 9).
* `imagecodecs` is not added to `run.json`'s `environment.packages`. A compressed file's hash is
  an identity of one environment, which the code identity and the container image already name.
