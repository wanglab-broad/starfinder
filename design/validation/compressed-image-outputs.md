# Validation design: compressed image outputs

% Status: Approved with the specification (Jiahao, 2026-11-21, W-909). Written against `70edc72` (v0.3.0).

This is the validation plan for implementing
[compressed-image-outputs](../specs/compressed-image-outputs.md). It has two parts:

* **Package checks** on synthetic fixtures, in the test suite: the codec of each writer, round
  trips, the opt-out, the records, the workflow's hashes, resume and saved sets, and the
  migrated goldens.
* **Real-data checks**, run as a separate evidence issue on UGER (W-889 D3):
  * sizes before and after on sampled tissue-2D and LN FOVs;
  * the `fov` task's wall time and peak RSS before and after;
  * identity by decoded content;
  * a run from W-903's uncompressed saved set.

The real-data checks are development checks. They make no data claim.

The expected values below come from the specification, from the files' TIFF tags and from the
evidence of W-881, W-908, W-910 and W-903, named in each row. They do not come from running the
new code.

**Tolerances** follow `docs/contributing.md` ("Expected values and tolerances"):

* Every identity on this page is exact: hashes, decoded arrays (integer and float, compared by
  their decoded identity), TIFF tags (integers), dtype and codec strings, record keys, exit
  statuses and counts. Compression is lossless, so a decoded array equals its source bit for
  bit, also for floats.
* A measured size, time or memory figure is compared with its stated bound. The figure itself is
  reported, not asserted.

## Fixtures

No check of this page needs a plane of a dataset's real size. The strip layout at 3,072 and 1,496 pixels is evidence
outside the suite (the repair run's `codec-check`). Every routine fixture is at most 32 × 64 × 64 voxels with at most four channels and four rounds,
as the project guidance bounds them.

| Fixture | Construction | Needs |
| --- | --- | --- |
| `volumes` | ZYXC arrays of 8 × 64 × 64 × 2 in uint8, uint16, uint32, float32 and float64, a sparse pattern (zeros with Gaussian spots) so that zlib shortens them, and the same as ZYX; an `ImageMetadata` with spacing (1.0, 0.1, 0.1) µm. | hand-built |
| `session_dataset` | The pytest session's small synthetic dataset (16 × 256 × 256, the existing exception), read by `FOV.run` with checkpoints. | existing |
| `workflow_examples` | The worked-example configurations of `test/workflow_examples.py` (`X`), with `execution.image_compression` set per row. | existing |
| `base_reader` | A scratch clone of `70edc72` (`git worktree add` or `git archive`) under the test's temporary root, whose `src/python` a subprocess puts first on `PYTHONPATH`, as the A19 `HASH_SCRIPT` runs code in its own process. | hand-built; skips with a stated reason when Git or the revision is absent |
| `old_set` | A saved set written with `image_compression: none` by the new code, and one written by `base_reader` at `70edc72` (uncompressed, no `image_compression` key). | hand-built from `workflow_examples` |

(compressed-image-outputs-checks)=

## Checks

"Needs": **H** hand-built or existing fixtures only; **WF** the workflow fixtures of
`workflow_examples`; **B** `base_reader`; **S** `slow`. Every row passes or skips in the CI
environment (extras `dev`, `local-registration`, `checkpoint`).

### Writers and codecs (CW)

| # | Check | Fixture | Expected | Tolerance | Needs |
| --- | --- | --- | --- | --- | --- |
| CW1 | Each writer of the inventory writes its codec | `volumes`; `session_dataset` | With the default, I1, I2, I3 (ZYX and the YX projection) and I4 have, on every page, TIFF Compression 8 and Predictor 1 (or no Predictor tag). With `none`, Compression 1. I6–I9 and I11 are Compression 8 in both settings. I10, I12 (HDF5 filters: none) and I13 are uncompressed in both settings. I14 stays uncompressed. I18 and I19 keep their Zarr codec in both settings, each compared with `70edc72` for the same input. I18: the `fusion` output entry of format `ome_zarr` in `export.json`, its `levels[].codec` and `compression` (`export/_fuse.py:777`–`:779`, `:789`), and the codecs of each level's `zarr.json` in the store are equal. I19: `export.json` records no codec. In its `spatialdata` section, each raster element's `levels` (the integer 1, `export/_spatialdata.py:437`), `array_metadata` (the path of the level's `zarr.json` or `.zarray`) and `array_metadata_sha256` (`:523`–`:525`) are equal, and so are the codecs that this array metadata of the store names. | exact (integers, strings) | H; for I18 and I19 the `multiview-stitcher` and `spatialdata` extras, skipped with the extra named |
| CW2 | Round trip | `volumes` through `write_registered_round`, `save_reference_image`, `_prepared.write_image` and their readers | Each read returns the decoded identity of the array written (dtype, shape, C-order SHA-256), every dtype of the fixture, and the `ImageMetadata` written. | exact (hashes; dtype strings; metadata equality) | H |
| CW3 | The opt-out reproduces `70edc72` | `volumes`; `base_reader` | With `image_compression: none`, the bytes of I1, I3 and I4 for the same array and metadata equal the base revision's files. The hashes are taken from the base revision in a subprocess, in the same environment. | exact (hashes) | H, B |
| CW4 | The writers that compress today keep their bytes | `volumes`; `base_reader` | `save_volume(compress=True)` with its explicit level 6 writes the same bytes as at `70edc72` (tifffile's default level is 6 with the locked imagecodecs), for ZYX and ZYXC. So do `write_labels` and the segmentation input. | exact (hashes) | H, B |
| CW5 | Values are checked | — | `CheckpointConfig(image_compression="zstd")` raises `ValueError` naming the field and `zlib`, `none`. A workflow configuration with `execution.image_compression: gzip` is refused at plan time, exit 2, naming the key and both values. | exact (exception types, exit status, strings) | H, WF |
| CW6 | The level and predictor are fixed | `volumes` | A file written with the default decodes equal, and its bytes equal `tifffile.imwrite` of the same array and metadata with `compression="zlib"`, `compressionargs={"level": 6}`, no predictor. | exact (hashes) | H |

### Records (CR)

| # | Check | Fixture | Expected | Tolerance | Needs |
| --- | --- | --- | --- | --- | --- |
| CR1 | The codec is recorded | `session_dataset`, both settings | `run.json` `config.checkpoints.image_compression`, the stage record's top-level `image_compression`, `run.json` `outputs.reference_image.compression` and `registration.json` `image.compression` each equal the setting, and each equals the codec read from its file's tags. The stage record's `files` entries are exactly `{path, sha256}`, and its format version is 3. | exact (strings, integers) | H |
| CR2 | Older records load | A stage record and a `registration.json` without the new key (written by `base_reader`, or the key removed) | They load; the reader takes them as uncompressed; no warning. | exact (arrays by hash; strings) | H, B |
| CR3 | v0.3.0 reads a new stage | A compressed stage written by the new code; `base_reader` | In a subprocess at `70edc72`, `FOV.load_checkpoint("registered")` restores the new stage (its checkpoint reader, `payload_entries` included), and `FOV.load_registered_round` reads a new `other_rounds` image. The arrays equal the written ones. | exact (hashes) | H, B, S if over 5 s |
| CR4 | Integrity is still the file | `session_dataset` | A byte flipped in a compressed round image makes the stage's read refuse with the file's SHA-256 mismatch, as for an uncompressed one. | exact (exception strings) | H |

### Workflow (WF)

| # | Check | Fixture | Expected | Tolerance | Needs |
| --- | --- | --- | --- | --- | --- |
| WF1 | The key is outside every hash | `workflow_examples`, planned with `zlib` and with `none` | Every task's `blocks_sha256` and `fingerprint` and the `configuration_sha256` are equal between the two plans. The resolved configuration holds `execution.image_compression` with the default written out. The committed C1 manifests (`data/expected/test_workflow_plan/`) are unchanged. | exact (hashes; files) | WF |
| WF2 | A resume after a switch | `workflow_examples`, a `fov` task with `registered` stopped after its stage under `none`, resumed under `zlib` | The resume is not refused, and `resumed_from` is `registered`. The kept round images stay Compression 1, and their stage record says `none`. `reference.tif` is Compression 8. `molecules.csv` equals that of a run without the stop, in the same environment. | exact (statuses, integers, strings, hashes) | WF, S if over 5 s |
| WF3 | A saved set of either codec is read | `old_set` (both); a task with `processing.spot_finding_input` and the default | `check_input` returns no difference for either set. `molecules.csv`, `candidates.csv` and `pre_qc.csv` equal those of a run from raw images in the same environment; `reference.tif` is Compression 8 and decodes equal. | exact (hashes; integers) | WF, B, S if over 5 s |
| WF4 | Direct and tiled agree | `workflow_examples`, direct and tiled | `reference.tif` is byte-equal between the routes (`test_workflow_tiled_tasks.py:97`, kept); `tiled/registered.h5` has no filter; a kept chunked checkpoint is unchanged in format. | exact (hashes; strings) | WF |
| WF5 | Engines agree | `test_workflow_engine_comparison.py` | Snakemake and Nextflow produce byte-equal outputs, compressed ones included. | exact (hashes) | WF |

### Identity and migrated expectations (ID)

| # | Check | Expected | Needs |
| --- | --- | --- | --- |
| ID1 | M1 | `test_report_records_golden.py` passes with `image_compression: "zlib"` added to its expected `config.checkpoints`, and no other change. | H |
| ID2 | M2 | X4 (`test_workflow_core_golden.py`) reports only the two new keys, each explained by the "W-909" exception; nothing unexplained. | WF |
| ID3 | M3, M4 | G3 (`test_workflow_orchestration_golden.py`) reports only the new keys under "W-909" and the reference image digests under OC3 with its amended reason. | WF |
| ID4 | Unchanged contracts | GN1, GA1, the array goldens, A19's in-process repeatability and every test the specification lists under "Other tests that stay" pass unchanged. The controller's `a19-parity` is not triggered: no assignment or segmentation code changes. | H, WF |
| ID5 | The tests that hash an I1–I4 file | Every test of the specification's table "The tests that hash a file this page changes" (`compressed-image-outputs-tests`) passes unchanged with the default `zlib`, and again with `image_compression: none`. The table comes from the recorded classification of the repair run (`results/classification/tests.tsv`, rows of class `image-file` with codec change `yes` or `both`). The pinned G3 cases pass with M4's amended reason. | H, WF |

### MATLAB (MT)

| # | Check | Expected | Needs |
| --- | --- | --- | --- |
| MT1 | The MATLAB-era files are unchanged | With `morphology.channel_images` on, `images/<round>/<label>/<fov>.tif` has Compression 1 under both settings, and its bytes equal the base revision's for the same input. No MATLAB run is needed: no file MATLAB reads changes. | WF, B |

### What the default tier covers

CW1–CW6, CR1, CR2, CR4, WF1, WF4 and ID1 run in the routine tier. CR3, WF2 and WF3 are `slow`
if they take 5 s or more on one CPU. ID2, ID3, ID5 and WF5 run in their existing selections (the
golden and workflow tests). The implementation reports the measured time of every new test that
takes 1 s or more.

## Real-data checks (separate evidence issue)

These checks process real FOVs and need more than a batch may run (W-889 D3: at most 1 CPU-hour
of real-data evidence inside a batch, on saved sets or crops). They are one evidence issue that
the coordinating session runs on UGER, on the implementation branch's commit, before the merge.
RD4 also belongs there although it reads a saved set: W-903's runs from the set reached about 16 GB
maxvmem per task, above a batch's 4 GiB bound.

* **Records** as the project guidance defines them: an input inventory first, one wrapper per
  step, the resource table, the manifest and the verifier.
* **One UGER node** for every timed step, named in the records. Timing as a result stays on one
  host.

### The FOVs

The ratio of one FOV does not carry to a dataset (W-910). The sizes are therefore taken on three
positions per dataset that span W-910's per-position ratios of the raw images. The times are
taken on the median one.

| Dataset | Position | W-910 raw ratio (rank) | Role |
| --- | --- | --- | --- |
| tissue-2D | tile_49 | 0.1098 (1 of 56) | sizes |
| tissue-2D | tile_27 | 0.1456 (25 of 56) | sizes, times, W-908's registered measure, W-903's set |
| tissue-2D | tile_23 | 0.1973 (56 of 56) | sizes |
| LN | Position006 | 0.0544 (1 of 64) | sizes |
| LN | Position010 | 0.4116 (32 of 64) | sizes, times |
| LN | Position046 | 0.5346 (64 of 64) | sizes |

The configurations:

* tissue-2D: W-903's `configs/set.yaml` (checkpoints `registered`, `candidates`, `pre_qc`;
  translation registration);
* LN: the dataset's `starfinder.yaml` of W-873 (`sample-dataset/LN`, E14's demons arm), with the
  same checkpoint stages. Its stage `registered` also holds a dense field per moving round (I5).

Each run has a `morphology` block, so that I4 is written. Where a dataset's configuration has
none, the evidence issue adds its dataset's reference stain round and records the addition.

### Checks

| # | Check | Data | Expected and bound |
| --- | --- | --- | --- |
| RD1 | Sizes per output | The six FOVs with `zlib` (the default) | For I1, I3 and I4 of each FOV: the bytes, and the uncompressed bytes computed from the shape and dtype (equal to a `none` file's bytes less its header, which RD2's `none` arm gives for two FOVs). The bytes of LN's dense fields (I5), unchanged by this page, for question 6. The ratio per output and per FOV is reported, not bounded. One bound comes from W-908: when tile_27's four registered rounds have the decoded identity of W-903's set, their total bytes are within 1 % of W-908's Tzlib arm, 285,778,070 bytes. |
| RD2 | Time and memory | tissue-2D tile_27 and LN Position010; arms `none` and `zlib`; two repeats each in the order none, zlib, zlib, none; one node | Per arm, the median of the `fov` task's `wall_s` and `peak_rss_kib` from its task record. Bounds: the wall-time rise is at most 5 % of the `none` arm, and the peak RSS rise at most 5 %. The expected rise is about 1–2 %. W-908 wrote tile_27's four registered rounds in 22.03 CPU-s, against 1,888–1,945 s per tissue-2D `fov` task in W-903. The issue's estimate from W-881's raw-image writes is about 40 s. |
| RD3 | Identity by decoded content | RD2's arms; RD1's runs | Between the `none` and `zlib` arms of one FOV: every I1, I3 and I4 file has the same decoded identity, and `molecules.csv`, `candidates.csv` and `pre_qc.csv` are byte-identical (one environment). For tile_27, the decoded identity of each registered round is also compared with W-903's set and reported. Equal is expected when the processing has not changed since `9640410` (W-889's stale rule). |
| RD4 | A run from W-903's uncompressed set | W-903's `configs/from-set.yaml` with the new code, on its four tiles | The set is read unchanged: no difference from `check_input`. Per tile, the genes' molecule counts equal W-880's noise counts (64 of 64 genes), and `molecules.csv` is byte-identical to W-903's run 2. `reference.tif` is Compression 8 and decodes equal to W-903's run 2 file. W-903's set is opened read-only, and its manifest verifies after the run (`scripts/make_set_manifest.py verify`). |
| RD5 | Predictor 3 on float data (decision 5, not gating) | The float32 model inputs (`segmentation/<run>/input.ome.tif`) of tissue-2D tiles 49, 27 and 23: one `segment` task per tile after RD1's runs, with a recipe that has a float step; each file re-encoded with zlib and with zlib + predictor 3 | W-908's rule: predictor 3 joins for float dtypes only if it cuts the bytes by at least 10 % on two of three files and raises the cold and warm reads by at most 5 % on every file. Otherwise FL1 stands. |

**Resources.** RD1 and RD2 are 12 `fov` tasks: RD2's eight (two FOVs, two arms, two repeats),
whose `zlib` runs serve RD1 too, and RD1's four other FOVs. Six are tissue-2D (1,888–1,945 s
each in W-903) and six LN (not timed before). RD4 is four `fov` tasks from the set (326–364 s
each in W-903, at about 16 GB maxvmem). RD5 adds three `segment` tasks. The evidence issue states its
slot-hour grant from these figures and checks the pilot's projection against it before the rest
is submitted (W-910's process note).

## Implemented checks

The implementation issue adds the table that maps each row above to its test, with the measured
time of each test that takes 1 s or more.
