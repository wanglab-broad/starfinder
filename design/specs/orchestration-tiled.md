# Native tiled execution: the global task, the tile grid, the tile task, the merge and the deep-tissue recipe

% Status: Proposed (W-644, 2026-10-29, at 56c35b8)

Status: Proposed. This record is reviewed at the specification gate of workflow orchestration
(W-646) before any implementation; approval publishes nothing.

This page specifies how one FOV that does not fit one job runs as tasks (decision D5): a global
task per FOV that loads, preprocesses and registers the rounds and writes them once as a chunked
checkpoint; one task per FOV and tile that reads its window and halo, optionally registers locally
and detects, extracts, decodes, scores and filters; and a merge per FOV that keeps each read in the
core of its own tile and writes the FOV's molecule table and `run.json` under the same contract as
direct execution. It also expresses today's deep route as a recipe, states when tiled equals
direct, gives a resource model checked against measurements on `thick_medium` and on a full-size
real 3D FOV, and lists what the Python backend retires. It builds on the contract
[orchestration](orchestration.md) (W-643), whose task kinds `tile_global`, `tile` and `tile_merge`
it fills in, and on the baseline [orchestration](../baselines/orchestration.md) (W-642). Statements
about today's behaviour cite the file and function at `56c35b8`, the base of the batch; paths are
relative to the repository root, and `starfinder/` stands for `src/python/starfinder/`.

% W-644 (S3 of §2.13); the batch runner/s213-spec-20261029 starts from dev at 56c35b8

The decisions this page implements (W-152, "§2.13 planning decisions (2026-10-27)"):

| Decision | What this page specifies for it |
| --- | --- |
| D5 | The global task, the chunked checkpoint, the tile grid and halo, the tile task, the merge, the deep recipe, what is retired |
| D2 | The three task kinds' commands, identities, declared inputs and outputs, and the tile list in the manifest |
| D8 | The chunked checkpoint's relation to the stage `registered`, resume of the global task, the records of the three kinds |

Evidence, kept with the run records outside the repository and cited by path: the scratch
prototype (`scripts/tiled.py`, `scripts/datasets.py`), the exactness series on a synthetic fixture
(`results/tiled/synthetic/`), the measurements on `thick_medium` and on aging Position400
(`results/tiled/thick_medium/`, `results/tiled/aging-9a0ec77/`, measured with the package at W-722's commit
`9a0ec77`, [measurements](#measurements); the earlier, stopped attempt at `56c35b8` in `results/tiled/aging/`), the
measurement table
(`results/tables/measurements.csv` and `.md`), the resource table of every step
(`results/tables/resources.csv`), the worker notes (`notes.md`) and the run's manifest
(`results/manifest.json`).

% Run directory: /home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/runs/W-644/20261029T200039Z-9454d4c8

## Terms

* **Direct**: one `fov` task runs the whole per-FOV core with `FOV.run` (`processing.mode: direct`).
* **Tiled**: the same core as three task kinds: `tile_global`, one `tile` per tile, `tile_merge`
  (`processing.mode: tiled`).
* **Tile grid**: a partition of the FOV's Y × X plane into rectangles, the **cores**; Z is never
  split. A **tile** is one core; its **window** is the core grown by the **halo** on every side and
  clipped to the FOV.
* **Footprint** of an operation: the largest lateral distance, in voxels, from a voxel or a
  candidate to an input voxel its result depends on.
* **FOV-level statistic**: a value an operation computes over the whole FOV (a detection threshold
  from the image maximum or noise, a min-max range, a histogram), which a window cannot compute.
* **Chunked checkpoint**: the registered rounds of one FOV, written once by the global task in a
  chunked format that a tile reads by window.

## Today's subtile and deep routes

Today's Python backend has two tiled routes, both built on the package's subtile API
(`design/baselines/orchestration.md`, the rule inventory, rows `gr_single_fov_subtile` to
`stitch_subtile`):

* **Subtile.** `gr_single_fov_subtile` (`workflow/rules/registration-py.smk:35-52`) runs
  `_run_workflow` (`starfinder/_snakemake.py:28`) on the whole FOV: `from_workflow_config`
  (`starfinder/dataset/workflow.py:479`) translates the rule with `retain_images=True` for the two
  creation rules (`workflow.py:603-605`) and a `SubtileConfig(sqrt_pieces, overlap_ratio)` whose
  windows `SubtileConfig.compute_windows` (`starfinder/dataset/types.py:237`) computes from
  `img_row` and `img_col` (`workflow.py:589-599`, `_image_shape`, `workflow.py:757`); `FOV.run`
  (`starfinder/dataset/fov.py:1154`) keeps every processed round; `FOV.save_reference_image`
  (`fov.py:2350`) writes the reference image and `FOV.save_subtiles` (`fov.py:2474`) writes one
  compressed NPZ per window, `output/subtile/<fov>/subtile_data_<n>.npz`, with every round's crop,
  and `subtile_coords.csv` with one-based corners. `lrsf_single_fov_subtile`
  (`workflow/rules/spot-finding-py.smk:14-28`) loads one NPZ with `FOV.from_subtile`
  (`fov.py:2568`), whose `subtile_id` enters the detection namespace (`FOV._detect_round`,
  `fov.py:920`) and the record folder `fovs/<fov>/subtile_<n>/` (`_layout.fov_dir`,
  `starfinder/_layout.py:95`), runs `FOV.run` without load and rotation, and writes the one-based
  `subtile_goodSpots_<n>.csv` without identities (`_export_legacy_spots`,
  `starfinder/readout/spots.py:154`). `stitch_subtile` (`spot-finding-py.smk:50-62`) runs
  `workflow/scripts/stitch_subtile.py`, which shifts each table by its one-based corner and keeps a
  read by comparing it with the next subtile's corner (`stitch_subtile.py:19-44`), and draws a
  preview on the reference image (`stitch_subtile.py:46-62`).
* **Deep.** `deep_create_subtile` (`registration-py.smk:56-72`) runs the same adapter with the
  parameters of its own rule (typically load and rotation only) and writes the same subtile files;
  `deep_rsf_subtile` (`spot-finding-py.smk:32-46`) runs each subtile's whole core, its registration
  included, with the parameters of its own rule. For a configuration in the MATLAB dialect the
  resident rules put reconstruction after registration (`_legacy_recipe`,
  `starfinder/dataset/_legacy_config.py:105`, line 117).

The subtile count is `sqrt_pieces` squared, restated in `workflow/rules/common.smk:259-264`
(`N_SUBTILE`, default 4); `workflow/Snakefile:63-74` orders the rules by `workflow_mode`. The
overlap is a fraction of the subtile (`overlap_ratio`, default 0.1, half on each side), not a
distance derived from what the operations read, and the merge keeps a read by comparing its
one-based position with corners rather than by ownership of a half-open core.

Streaming execution already bounds the memory of direct: in `FOV.run`'s streaming mode each round is
loaded when its turn comes (`fov.py:1337`), and a moving round's image is dropped after its
detection and extraction (`fov.py:1375-1383`), so that one moving round and the reference round are
resident at once. The `registered` checkpoint stage writes each round as it is produced
(`fov.py:1366`) with `_checkpoint.write_registered_round` (`starfinder/_checkpoint.py:254`): one
uncompressed ZYXC OME-TIFF per round (`save_volume`, `starfinder/io/tiff.py:339`), strips of whole
rows, no tiles.

## The flow

1. **`tile_global`** (one per FOV) runs `FOV.run` with the operations up to and including global
   registration and the post-registration steps, in streaming mode; it writes each registered round
   into the chunked checkpoint as it is produced, the reference image as direct writes it, and the
   FOV-level statistics the tiles need.
2. **`tile`** (one per FOV and tile) reads its window of every round from the chunked checkpoint,
   optionally registers each moving round locally inside the window, runs detection, extraction,
   decoding, scoring and filtering with `FOV.run`, and keeps the candidates and reads whose position
   lies in its core.
3. **`tile_merge`** (one per FOV) concatenates the tiles' core candidates and reads, gives them the
   FOV's identities in the detector's order, and writes `fovs/<fov>/run.json`, `molecules.csv` and
   the checkpoint stages as direct writes them.

The plan computes the tile list and the halo from the resolved configuration and the image shape
(read from the TIFF headers without the pixels, as `_volume_header`, `starfinder/io/tiff.py:215`,
does), and writes one `tile` task per tile into the manifest, so an engine fans out over the
manifest's tiles and never computes a tile itself (D2).

## The global task

**What it runs.** `FOV.run` of the resolved processing block with detection and readout removed
(`spot_finding`, `extraction`, `decoding`, `scoring`, `deduplication` and `filtering` set to `None`,
which `PipelineConfig`, `starfinder/dataset/config.py:124`, supports), in streaming mode: load,
rotation, the preprocessing recipe's steps, global registration and the recipe's
`post_registration` steps. Every operation that needs the whole FOV runs here: min-max
normalization and histogram matching with `fit="fov"` compute per-channel statistics of the whole
round (`normalize_min_max`, `starfinder/preprocessing/normalization.py:40`;
`HistogramMatchingConfig`, `normalization.py:79`), reconstruction by dilation propagates without a
distance bound (`_morph`, `starfinder/preprocessing/morphology.py:39-50`), and global registration
estimates one transform from the whole rounds. The prototype ran exactly this call, the rule's
`PipelineConfig` with those six fields set to `None` (`tiled.py`, `cmd_global`).

**The FOV-level statistics.** After the run the reference round's detection image is still resident
(streaming keeps it, `fov.py:1375-1385`). The task computes from it what a tile cannot:

| Statistic | Why a tile cannot compute it | Where it goes |
| --- | --- | --- |
| The detection threshold of each channel (`local_maxima` modes `noise`, `adaptive`, `adaptive_round`: the channel's median and MAD, the channel maximum, the image maximum; `_noise_record`, `starfinder/spot_finding/_methods.py:106`) | A window's median, MAD or maximum differ from the FOV's | The global record; each tile detects with these absolute thresholds |
| The extraction's image background and noise per round and channel (`_image_background`, `starfinder/readout/extraction.py:219`) | Whole-round median and MAD | The global record; the merge writes them into the `candidates` stage header, where direct writes them (`_checkpoint.write_candidates`, `_checkpoint.py:438`); no read uses them |

The mode `global` of `local_maxima` (a fraction of the dtype maximum) and every other value the tiles use are
FOV-independent already. The prototype computed the thresholds with the package's own formula and passed each
to its tile as `threshold_mode="global"` with the threshold divided by the dtype maximum, after checking that
the product compares equal on every integer pixel value; the thresholds equalled direct's on the synthetic
fixture, `thick_medium` and the real FOV (`comparison.json`, `thresholds_equal`).

**What it writes.** The reference image, exactly as direct (`FOV.save_reference_image`); the chunked
checkpoint; its record `global.json`; and its `run.json` (the identity, inputs and preprocessing and
registration records `FOV.run` writes, without counts), from which the merge copies those sections.

**The chunked checkpoint.** One file per FOV holding, per pipeline round (the reference round and the
sequencing rounds, `FOV._pipeline_rounds`), the detection image after the post-registration steps, and the
recipe's extraction-source snapshot when one is set (as the `registered` stage does,
`FOV._checkpoint_snapshots`, `fov.py:1720`), in the dtype the preprocessing produced (`uint8` after min-max to
`uint8`). Each array is ZYXC with chunks of (Z, c, c, C), all Z planes and channels in one chunk, and carries
its `ImageMetadata`; the file carries the identity and the round and channel labels. Its size is the size of
the registered rounds: 5,436,024,832 bytes for aging Position400 (nine rounds of 36 × 2048 × 2048 × 4,
`uint8`) and 3,355,465,384 for `thick_medium` (four rounds of 100 × 1024 × 1024 × 4, `uint16`), the size of
the OME-TIFF `registered` stage to within 0.01 %. A tile window touches every chunk that overlaps it; with `c`
dividing the tile size, a window of `T + 2h` touches at most `T/c + 2` chunks per axis.

### Choice: the format of the chunked checkpoint

The package writes OME-TIFF only (`save_volume`, `io/tiff.py:339`, strips; `write_ome_tiff`,
`starfinder/io/_ome.py:73`, 512 × 512 zlib tiles of a ZYX or YX array, for the sample export). HDF5
(`h5py`) is a core dependency, used today to read AnnData (`read_h5ad`,
`starfinder/report/_measurements.py:142`); OME-Zarr is written only by the fusion, through
`multiview_stitcher` (`starfinder/export/_fuse.py:697`), with the `multiview-stitcher` extra.

| Option | Cost | Consequence |
| --- | --- | --- |
| A. HDF5, one file per FOV, one dataset per round, chunks (Z, c, c, C), uncompressed (recommended) | A new writer and reader in `starfinder.io` (a few dozen lines over `h5py`) | Core dependency, no lock change; one file per FOV; a window reads only its chunks; measured in the page cache, the 16 windows of a 4 × 4 grid of `thick_medium` (3.35 GiB) in 5.5 s, 628 MiB/s, against 117 s of the tiles' own work, and of the real FOV (5.24 GiB) in 12.9 s, 416 MiB/s, against 1,408 s |
| B. OME-Zarr (zarr), one array per round | The `zarr` package becomes a core dependency (a lock change) or tiled needs an extra | Many small files per FOV (one per chunk), costly on the SMB mount and on cluster file systems; readable by napari and Fiji |
| C. The existing `registered` stage (uncompressed strip OME-TIFF), read by memory map | None: it exists | A window of width `T + 2h` reads whole rows (`X` wide), and memory-mapped pages count toward the task's resident memory; measured the same windows in 2.3 s, 1,498 MiB/s (`thick_medium`) and 5.6 s, 955 MiB/s (the real FOV), in the page cache, where reading whole rows costs nothing |
| D. Tiled OME-TIFF (`write_ome_tiff`), one ZYX file per round and channel | Its 512 tiles fix `c` at 512; four to five files per round | Every tile read decompresses zlib tiles; the export's annotations do not fit a checkpoint |

Recommendation: A, uncompressed. C read the same windows 2.3 to 2.4 times faster, in the page cache, where the whole rows
it reads cost little; from disk a window reads whole rows, `X / (T + 2h)` times its own bytes (3.8 for a 4 × 4
tile of `thick_medium`, 7.6 for an 8 × 8 tile of a 2048 FOV), its mapped pages count toward the tile's memory
while it reads, and it cannot be compressed; both reads are small beside the tiles' own work (4.7 % and 2 % of it
on `thick_medium`, 0.9 % and 0.4 % on the real FOV). Compression: on `thick_medium`'s round 1 (`uint16`), lzf
gives 1.28 × in 7.2 s and gzip level 1 1.97 × in 11.9 s on one CPU, so gzip would add about a quarter to the
global task (four rounds, 48 s against 182 s); on the real FOV's round 1 (`uint8`) lzf gives 1.42 × in 5.3 s and
gzip level 1 1.81 × in 10.3 s, about 9 % of its global task for nine rounds (93 s against 983 s); the checkpoint
lives until the merge, so disk is rarely the limit. Compression is an option of the writer (gzip level 1) for a
run that is short of scratch.

### Choice: the chunk edge `c`

| Option | Cost | Consequence |
| --- | --- | --- |
| A. `c = 128` (recommended) | 256 chunks per round of a 2048 × 2048 FOV | Divides every tile size that is a multiple of 128; a window touches at most `T/128 + 2` chunks per axis; one chunk of the real FOV is 2.4 MB (36 × 128 × 128 × 4, `uint8`; 13 MB for `thick_medium`'s 100 planes in `uint16`) |
| B. `c = T`, the tile size | One chunk per tile | A window with its halo touches nine chunks, most of them for a halo of a few voxels |
| C. `c = 64` | Four times as many chunks | Less over-read, more per-chunk overhead |

Recommendation: A, fixed by the format (not a configuration key), recorded in `global.json`.

### Choice: the chunked checkpoint and the stage `registered`

The contract leaves this to this page (`orchestration.md`, "Checkpoints and resume").

| Option | Cost | Consequence |
| --- | --- | --- |
| A. The chunked checkpoint is the tiled form of the stage `registered`: written by every tiled run (the tiles need it), kept when `processing.checkpoints` holds `registered`, otherwise removed by the merge after it succeeds (recommended) | The disk of the registered rounds until the merge | One stage, two forms; `FOV.load_registered_round` reads either (an addition for the HDF5 form); a segmentation input that names a sequencing round keeps it, as the plan adds `registered` for direct |
| B. A separate temporary file, always removed after the merge; `registered` in `processing.checkpoints` also writes the OME-TIFF stage | Two writes of the same rounds when `registered` is on | Simple rule, double disk and time |
| C. Always kept | The disk of every FOV's registered rounds | Resume of the tiles from it at any time |

Recommendation: A.

**Resume of the global task.** With `execution.resume_from_checkpoint`, the global task resumes when
its `run.json` matches (the table of the contract) and the chunked checkpoint's record lists every
round with the SHA-256 of its array bytes; it then has nothing to rerun and exits 0 after checking
the hashes. Otherwise it reruns from scratch (the rounds are produced in order, so a partly written
checkpoint is discarded). Tile and merge tasks rerun from scratch; they are short.

## The tile grid

**The grid.** For tile size `T` (Y and X), the grid has `n_y = ceil(Y / T)` rows and
`n_x = ceil(X / T)` columns, and the cores have the integer proportional boundaries
`y_k = k · Y // n_y`, `x_k = k · X // n_x` (the rule `SubtileConfig.compute_windows` uses for its
bases, `types.py:237`), so every core is at most `T` and the cores differ by at most one voxel.

**Half-open cores.** Core `(row, col)` is `[y_row, y_row+1) × [x_col, x_col+1)`; the cores partition
the plane. A candidate or read at `(z, y, x)` (the detector's coordinates, integers for
`local_maxima`, floating for learned detectors) belongs to the one tile whose core contains `(y, x)`;
no rounding is applied, so the rule holds for floating coordinates too.

**The window.** `[max(y_row − h, 0), min(y_row+1 + h, Y))` and likewise in X; Z is whole.

**Tile identity.** A tile is `r<row>c<col>`, zero-based, row-major; it is `identity.tile` and the
last part of the task id (`tile/<sample>/<fov>/r0c1`, the id form of `orchestration.md`, "The
manifest"). Its records live under `fovs/<fov>/tiles/r<row>c<col>/`.

**Spot identity.** A tile detects in the namespace `[dataset, sample, fov, "r<row>c<col>"]` (the
fourth element is today's `subtile_id`, `fov.py:920`) with the detector's own IDs; these are
provisional and appear only in the tile's files. The merge gives every candidate its FOV ID in the
FOV's namespace `[dataset, sample, fov, null]`, the namespace direct writes: the rank of the
candidate in the detector's order. For `local_maxima` the order is channel, then decreasing pixel
value, then z, y, x: `_maxima` (`_methods.py:123`) detects channel by channel, and
`peak_local_max` returns peaks by decreasing value, ties in C order (scikit-image 0.26,
`_get_high_intensity_peaks`, a stable sort of `np.nonzero`'s output); `_detect`
(`starfinder/spot_finding/__init__.py:202`) numbers the rows in that order (line 271). The
merged IDs are therefore unique (one owner per candidate), deterministic (they depend on values
and positions only, not on the grid) and equal to direct's. Measured: equal on every grid of the
synthetic fixture, of `thick_medium` and of the real FOV (2 × 2 and 4 × 4, 766,104 candidates;
`candidates_equal`).

### Choice: the identities of a tiled FOV

| Option | Cost | Consequence |
| --- | --- | --- |
| A. FOV IDs by the detector's order, assigned by the merge (recommended) | The merge must know the order of each detector: each `SPOT_FINDING_METHODS` entry declares its ordering key, or the merge refuses the detector in tiled mode | Tiled and direct molecule tables are equal, IDs included; downstream joins (assignment, export) cannot tell the modes apart |
| B. Tile-qualified IDs (`r0c1:17`) in the FOV namespace | None | Unique and deterministic, but different from direct's; a comparison must match on positions |
| C. IDs from coordinates (`z_y_x_c`) | A new ID form | Equal across grids, but not direct's today, and two candidates at one voxel in two channels need the channel in the ID |

Recommendation: A. For the learned detectors (`spotiflow`, `piscis`) and `starfish_log` the merge
sorts by channel, then z, y, x unless the method declares otherwise; direct does not reorder, so
their tiled IDs differ from direct's until direct adopts the same order (a change for their issue).

### Choice: how the tile size is chosen

| Option | Cost | Consequence |
| --- | --- | --- |
| A. A fixed count per axis (`sqrt_pieces`, today's key) | None | The tile's memory depends on the FOV's size, so one count does not fit two datasets |
| B. A tile size in voxels, `processing.tiled.tile_size_yx` (recommended) | The user picks a number | The tile's memory is bounded by the size, whatever the FOV; with local registration the size changes results, so it belongs in the configuration hash, as a scientific setting |
| C. A memory target: the plan picks the largest tile whose predicted peak fits `execution.tasks.tile.memory_mb` | A resource model in the plan | Results with local registration would then depend on an execution setting outside the hash, which the boundary of 2026-09-28 forbids |

Recommendation: B, with a default of 512, and the plan's check of the predicted peak
([resource model](#the-resource-model)) against `execution.tasks.tile.memory_mb`: a tile that would
exceed it fails at plan time, naming the size that would fit. The size rests on the measurements: at 512 a
tile of `thick_medium` peaks at 1.57 GiB (all four tiles of the 2 × 2 grid), a quarter of its global task
(6.06 GiB), and on the real FOV at 512 (4 × 4) a tile peaks at 0.78 to 0.82 GiB against its global task's
7.67 GiB and at 1024 (2 × 2) at 2.62 to 2.64 GiB; halving it to 256 saves another 1 GiB on `thick_medium` (0.57
GiB) for 20 % more CPU time in tiles (118.6 s against 99.1 s, each task paying the interpreter and the
configuration again; on the real FOV the twelve more tasks of 512 against 1024 cost 30 s each, its codebook's
load: 1,408 s against 1,051 s), and doubling it to 1024 makes a `thick_medium` tile as large as its global task
(5.64 GiB).

## The halo rule

The halo is the largest footprint of the operations the tile runs, so that every result for a voxel
or candidate of the core reads only voxels of the window:

```text
h = max(detection footprint, local registration margin, extraction neighbourhood)
```

| Operation | Footprint (lateral, voxels) | Source |
| --- | --- | --- |
| `local_maxima` | `min_distance_voxels` (`d`): the maximum filter of size `2d + 1` (scikit-image's `_get_peak_mask`, `mode="nearest"`) and the border exclusion of width `d` (`exclude_border`, `_peaks`, `_methods.py:97`). With `d > 1` the spacing step (`ensure_spacing`) and with `merge_radius_zyx` the within-channel merge (`_merge`, `_methods.py:147`) are greedy over the whole channel: not bounded (below) | `LocalMaximaConfig`, `starfinder/spot_finding/_config.py:59` |
| `starfish_log` | `ceil(3 · max_sigma)` per axis, the Laplacian of Gaussian's support | `StarfishLogConfig` |
| `spotiflow`, `piscis` | The model's receptive field, which the method must declare; the tile also needs at least the method's minimum shape (`min_shape_zyx`) | `SPOT_FINDING_METHODS`, `_methods.py` |
| Extraction (`neighborhood_sum`) | `max(neighborhood_radius_zyx[1:], background.outer_radius_zyx[1:])`; defaults (2, 2) and (6, 6): 6 | `NeighborhoodSumConfig`, `LocalBackgroundConfig`, `extraction.py:27-82`; boxes are clipped at the image's faces (`extract_intensities`, `extraction.py:293-294`), so a box clipped at a window face differs from the FOV's |
| Decoding, scoring, filtering | 0: per read | `decode_reads`, `decoding.py:306`; `score_reads`, `scoring.py:107`; `filter_reads`, `filtering.py:124` |
| Deduplication | Not bounded: links reads within `distance_voxels` transitively | `DeduplicationConfig`, `starfinder/readout/deduplication.py:36`; runs in the merge (below) |
| Local registration | The margin of the next section | — |

With the defaults of the measured configurations (`min_distance_voxels` 1, extraction box (1, 2, 2),
ring (1, 6, 6)) the rule gives `h = 6`. The synthetic series shows that 6 is the tight value: with
`h = 6` or 12 tiled equals direct on 2 × 2 and 3 × 3 grids; with `h` from 1 to 5 every candidate
and every identity, position and gene is equal but scores differ (up to 0.18, the ring is clipped);
with `h = 0` 11 of 71 candidates are lost at the tile faces (`results/tiled/synthetic/g*/comparison.json`).

**The plan computes the halo.** The plan computes the minimum from the resolved configuration
with this table and writes it into the resolved configuration (`processing.tiled.halo`); a
configured halo below the minimum is refused at plan time, naming the operation that sets it, and a
larger one is accepted. A detector without a declared footprint, `local_maxima` with
`min_distance_voxels > 1` or a `merge_radius_zyx`, and deduplication in the tiles have no finite
footprint: the plan refuses them in tiled mode unless `processing.tiled.exact: false` accepts a
result that may differ from direct (see [equivalence](#equivalence-with-direct)).

**The local registration margin.** A local registration step estimates a field from the window; a
voxel of the core is moved by the field near it, whose estimate depends on image content within
the method's support. Proposed margin: the largest displacement the step may find plus its
smoothing support, `m = max_displacement + 3 · smoothing_sigma · 2^(levels − 1)` voxels, with
`max_displacement` a configuration value (default 30 voxels, the largest cap of the synthetic
deformation presets, `DEFORMATION_PRESETS`, `starfinder/synthetic/_presets.py`), so 42 for the
default `DemonsConfig` (three levels, `smoothing_sigma` 1). The margin is a proposal: no measurement
of this batch sets it, and the validation issue calibrates it (W-645).

### Choice: the halo

| Option | Cost | Consequence |
| --- | --- | --- |
| A. Computed by the plan from the footprints, a configured value accepted only above it (recommended) | A footprint per method in the registries | Exact by construction where the footprints are finite; a user cannot choose an inexact halo by accident |
| B. A configured fraction of the tile (today's `overlap_ratio`) | None | The overlap grows with the tile, not with what the operations read; 0.1 of a 512 tile is 25 voxels each side, of a 64 tile 3, which is less than the ring (6) |
| C. A fixed number (for example 16) | None | Covers today's defaults, silently wrong for a wider ring or a local registration |

Recommendation: A.

## The tile task

**What it reads.** Its window of every pipeline round from the chunked checkpoint; the window's
geometry is the round's `ImageMetadata` cropped at the window's start (`ImageMetadata.cropped`,
`starfinder/image.py:88`, as `FOV.save_subtiles` crops today, `fov.py:2523-2524`); the FOV-level
statistics from `global.json`; the codebook.

**What it runs.** `FOV.run` of the processing block with load, rotation and the preprocessing recipe
removed, in batch mode (the window is small), with the detector's thresholds replaced by the
FOV-level ones and, when `processing.tiled.local_registration` is set, that registration recipe
(its steps run on the window of each moving round against the reference round's window, through
`FOV.register` as `FOV.run` calls it). Detection, extraction, decoding, scoring and filtering are
the existing stage functions with the block's settings. Deduplication is not run in the tile.

**What it keeps.** The candidates and reads whose position lies in its core, in FOV coordinates (the
window's start added), with their provisional IDs.

**What it writes**, under `fovs/<fov>/tiles/r<row>c<col>/`: `candidates.csv` (the core's
candidates with their intensities, background and the columns of the `candidates` stage table,
`candidates_frame`, `_checkpoint.py:76`), `reads.csv` (the core's reads after decoding and scoring,
the columns of the `pre_qc` stage table) and `tile.json` (identity, window, core, halo, thresholds
used, counts in window and core, the attempt). Filtering runs in the merge on the merged reads, so
that the tile writes no molecule table: filtering is per read (`filter_reads`), so the result is the
same, and deduplication, which must run before filtering, can run on the merged reads.

### Choice: what a tile writes

| Option | Cost | Consequence |
| --- | --- | --- |
| A. The core's candidates and reads (the two stage tables' columns); the merge deduplicates, filters and writes everything (recommended) | Larger tile files (every candidate's intensities) | The merge writes the FOV's `candidates` and `pre_qc` stages and molecules exactly as direct; deduplication is exact across tile faces |
| B. The core's molecules only (what the prototype wrote) | Small files | The FOV's `candidates` and `pre_qc` stages (on by default, D8) cannot be written; no deduplication |

Recommendation: A. The prototype wrote B plus the core candidates, which was enough for the
comparison of identities, positions, genes and scores.

**The threshold option.** Passing an absolute threshold through `threshold_mode="global"` is exact
for integer images but a workaround. Proposed: a `threshold_mode` `"absolute"` of `LocalMaximaConfig`
(`threshold_value` in grey levels), which the global record's thresholds fill. The tile records
both the configured mode and the absolute value it used.

## The merge

**Ownership.** Each candidate and read is owned by the one tile whose core contains it; the merge
checks that no candidate appears twice (`(z, y, x, channel)` unique) and that every tile reported.

**The deterministic order.** The merge concatenates the tiles' candidates, sorts them by the
detector's order and numbers them from 0 (the FOV IDs); the reads take their candidate's FOV ID.
The order depends on values and positions only, so the result does not depend on the grid, the
halo or the order in which tiles finished.

**What it runs.** On the merged reads: deduplication when configured (`deduplicate_reads`,
`deduplication.py:134`), then filtering (`filter_reads`), as `FOV.run` orders them after scoring.

**What it writes**, under the same contract as direct (D5): `fovs/<fov>/molecules.csv` with the
columns, dtypes and writer of `FOV.export_spots` (`fov.py:2409`; the prototype used the same
writer, `_write_spot_table`); the `candidates` and `pre_qc` stages that `processing.checkpoints`
names, written by `_checkpoint.write_candidates` and `write_pre_qc` with the FOV's identity (no
`subtile_id`) and the global task's image background; and `fovs/<fov>/run.json` version 2 with the
fields direct writes (the global task's inputs and preprocessing and registration records, the
merge's counts, `outputs` with `molecules_sha256`) plus a section `tiled`:

| Field of `run.json.tiled` | Content |
| --- | --- |
| `grid` | `tile_size_yx`, `n_yx`, `halo`, `chunk_zyxc`, `local_registration` (the recipe or null) |
| `checkpoint` | The chunked checkpoint's path relative to the FOV folder, its SHA-256 per round, and whether it was kept |
| `tiles` | One entry per tile: `tile`, `core_yx`, `window_yx`, `candidates`, `reads`, the SHA-256 of its files and the task record's attempt |
| `statistics` | The FOV-level thresholds and image background the tiles used |

The section is additive within version 2, like the contract's `workflow` key; a reader that does not
know it loads the record as direct's.

## The deep-tissue recipe

D5 makes `deep` a recipe usable with direct or tiled. Today's Python deep route is the operations
below; each is mapped to a step of the recipe or of tiled execution, or listed as dropped with the
reason.

(orchestration-tiled-deep-map)=

| Operation of today's deep route | Where | In the recipe or tiled execution | Reason |
| --- | --- | --- | --- |
| Load the raw rounds with the channel patterns | `deep_create_subtile` → `_run_workflow` (`_snakemake.py:28`); `ImageLoadConfig` (`workflow.py:581`) | Global task (or `fov`), streaming | As direct |
| Rotation by `rotate_angle` | `PipelineConfig.rotation_degrees` (`workflow.py:582`) | Global task (or `fov`) | As direct |
| Preprocessing of the creation rule (native `preprocessing.steps`; MATLAB dialect `enhance_contrast`, `hist_equalize` → min-max, histogram matching, `_legacy_recipe`) | `from_workflow_config` (`workflow.py:545-556`), `_legacy_recipe` (`_legacy_config.py:105`) | Recipe `preprocessing.steps`: min-max to `uint8` and histogram matching (`fit="fov"`) | FOV-level statistics: only the global task (or `fov`) computes them exactly |
| Global registration of the creation rule, if configured (MATLAB's deep script registers at scale 0.25) | `registration` of `deep_create_subtile` | Recipe `registration.steps`: translation | The scale 0.25 is MATLAB's (`deep_create_subtile.m`); the Python translation has no scale option and estimates at full resolution |
| Keep every processed round (`retain_images=True`) | `from_workflow_config` (`workflow.py:603-605`), `ExecutionConfig` (`config.py:19`) | Dropped | The global task writes each round into the chunked checkpoint as it is produced (streaming); nothing needs all rounds resident |
| Subtile windows: `sqrt_pieces` × `sqrt_pieces` with `overlap_ratio`, from `img_row`, `img_col` | `SubtileConfig.compute_windows` (`types.py:237`), `_image_shape` (`workflow.py:757`) | Tile grid (`tile_size_yx`) and halo (the footprint rule) | The overlap becomes the computed halo; the shape is read from the images, so `img_row` and `img_col` are not needed |
| The reference image | `FOV.save_reference_image` (`fov.py:2350`) in the creation rule | Global task (or `fov`) | As direct |
| One compressed NPZ per subtile with every round's crop, and one-based `subtile_coords.csv` | `FOV.save_subtiles` (`fov.py:2474`), `output/subtile/<fov>/` (`_FovPaths.subtile_dir`, `_layout.py:232`) | Chunked checkpoint (written once, read by window) and the manifest's tile list | A crop copy per subtile repeats the overlap and holds every round of a subtile in one file; one-based corners are MATLAB's convention |
| The creation rule's `run.json` with `outputs.subtile_data`, `subtile_coords` | `FOV.save_subtiles` (`fov.py:2559-2565`) | `global.json` and the global task's `run.json` | Same role, chunked checkpoint instead of NPZ files |
| Load one subtile as a FOV with `subtile_id` | `FOV.from_subtile` (`fov.py:2568`), resident rules (`_RESIDENT_RULES`, `workflow.py:25`) | Tile task: the window read from the chunked checkpoint | The namespace takes the tile's id (provisional); the merge gives FOV IDs |
| Preprocessing of the resident rule (native `preprocessing`; MATLAB dialect `morph_recon` → reconstruction after registration) | `_legacy_recipe` (`_legacy_config.py:117`) | Recipe `preprocessing.post_registration`: reconstruction (radius 3), run in the global task (or `fov`) | Reconstruction by dilation is not bounded in distance (`_morph`, `morphology.py:50`), so in a tile it is not exact; after a tile's local registration it runs in the tile (`processing.tiled.post_local_registration`), inexactly by nature |
| Registration of each subtile (the resident rule's `registration`; MATLAB dialect `global_registration`, a translation per subtile) | `deep_rsf_subtile` → `_run_workflow` | `processing.tiled.local_registration`: one translation step per tile (piecewise rigid); with direct, a local step over the FOV | Per-subtile registration is a local registration whose model is a translation per tile |
| Detection (MATLAB dialect `intensity_estimation: global`, forwarded only by the deep script) | `spot_finding` of `deep_rsf_subtile` | Tile task; `threshold_mode` as configured, FOV-level thresholds from the global task | Mode `global` is FOV-independent; the other modes need the FOV's statistics |
| Extraction (`voxel_size`), codebook (`load_codebook.split_index`), filtering (`end_base`, `n_barcode_segments`) | Resident rule parameters | Tile task (extraction, decoding, scoring) and merge (filtering), with the processing block's settings | As direct |
| The one-based `subtile_goodSpots_<n>.csv` without identities | `_export_legacy_spots` (`spots.py:154`) | Dropped | The tile writes zero-based candidates and reads with identities |
| The subtile's `run.json` under `fovs/<fov>/subtile_<n>/` | `_layout.fov_dir` (`_layout.py:95`) | `fovs/<fov>/tiles/r<row>c<col>/tile.json` and the merge's `run.json.tiled.tiles` | One FOV record, as direct |
| The merge by one-based corners | `stitch_subtile.py:19-44` | Merge by half-open core ownership | The corner rule takes each overlap from the later subtile from the overlap's first voxel on, so the reads it keeps there lie at that subtile's face, where its detection and extraction see the face; a core owner reads every kept read at least `h` from its window's faces |
| The preview of the merged reads on the reference image | `stitch_subtile.py:46-62` | Dropped from the core | Previews are options of the assign and export commands (P5, `orchestration.md`) |
| `temp()` of the subtile files; `cleanup_subtile.py` (no rule calls it, O21) | `registration-py.smk:42-44`, `:62-64`, `spot-finding-py.smk:20`, `:38` | The merge removes the chunked checkpoint unless `registered` is kept | The engine's temporary-file mechanisms differ; the package removes what it owns |
| The rule order by `workflow_mode` and `N_SUBTILE` | `Snakefile:63-74`, `common.smk:259-264` | The plan's task graph and tile list (W-643) | D2, D4 |

**The recipe.** `processing.recipe: deep` is expanded by the plan into the resolved configuration
(so the hash covers the steps, not the name), and a key the user also sets wins over the recipe's:

```yaml
processing:
  mode: tiled                 # or direct
  recipe: deep
# expands to
  preprocessing:
    steps:
      - {method: min_max_normalization, output_dtype: uint8, output_range: [0, 255], rounding: truncate}
      - {method: histogram_matching, reference_channel: 0}
    post_registration:
      - {method: reconstruction, radius_yx: 3}
  registration: {steps: [{method: translation}], signal: {mode: max}}
  tiled:                      # mode tiled only
    local_registration: {steps: [{method: translation}], signal: {mode: max}}
```

With `mode: direct` the recipe is the same without `tiled`: the per-subtile translation has no
counterpart in one FOV task, and a user who wants deep's piecewise correction there adds a local
step (`demons`) to `registration.steps`. The detection and readout settings stay in their keys
(deep's `threshold_mode: global` and its codebook are dataset choices, not the recipe's).

### Choice: how the recipe is expressed

| Option | Cost | Consequence |
| --- | --- | --- |
| A. A named recipe the plan expands into the resolved configuration (recommended) | A recipe table in the package, documented in the reference | One key to ask for deep; the resolved configuration shows every step; the hash does not depend on the name |
| B. A documented example configuration, no name | None in the package | Nothing to maintain; users copy steps and drift |
| C. A recipe file next to the configuration | A file format | Shareable recipes; one more input for the plan to hash |

Recommendation: A.

## Equivalence with direct

**Without local registration, tiled equals direct exactly** when the halo meets the rule and every
operation of the tile has a finite footprint. The argument:

1. The global task runs the same calls on the same FOV as direct's `FOV.run` up to the
   post-registration steps (streaming, the same recipe), so its registered rounds are direct's,
   value for value; the chunked checkpoint stores them losslessly (uncompressed, the same dtype).
2. A tile's window holds, for every voxel of its core, every voxel within `h` of it, so an
   operation whose footprint is at most `h` gives at a core voxel the result it gives in the FOV:
   the maximum filter and the threshold comparison of `local_maxima` (with the FOV-level threshold),
   the border exclusion (a window face at distance `h ≥ d` from the core is not a FOV face; where
   the window face is the FOV face, direct excludes the same border), and the extraction box and
   ring (clipped only at FOV faces).
3. Decoding, scoring and filtering act on one read at a time, so a read gets the same call, score
   and decision.
4. The merge's IDs are the detector's order, which is direct's numbering.

The prototype confirmed it: on the synthetic fixture (2 × 2 and 3 × 3 grids, halos 6 and 12), on `thick_medium`
(every tile of the 1 × 1, 2 × 2 and 4 × 4 grids, 8,018 candidates and 5,488 molecules) and on the real FOV (every
tile of the 2 × 2 and 4 × 4 grids, 766,104 candidates and 207,686 molecules, at `9a0ec77`), the candidate tables
are equal (IDs, positions, channels, peak values) and the molecule tables are equal in namespace, ID, z, y, x,
gene and entry; the scores differ by at most 2.2e-16 (synthetic), 3.6e-15 (`thick_medium`) and 1.4e-14 (the real
FOV), the last bit of float64 arithmetic that numpy vectorizes differently over arrays of different lengths; the
files are therefore not byte-equal, and a comparison uses the tolerance of `docs/contributing.md` for scores.

**Where exactness does not hold, and why:**

* `local_maxima` with `min_distance_voxels > 1`: `ensure_spacing` keeps peaks greedily by
  decreasing value over the whole channel, so a chain of peaks can carry a decision across any
  distance; the same holds for `merge_radius_zyx` (`_merge`).
* A channel constant over a whole window: `peak_local_max` returns no peak for a trivial image
  (`_get_peak_mask`), where the FOV, not constant, returns the window's voxels above the threshold.
  Only a window entirely saturated or empty in one channel can meet it.
* Learned detectors: their output near a window face depends on the receptive field and the model's
  own tiling; exactness holds only with a halo at least the declared receptive field.
* Deduplication in a tile (chains across faces); the merge runs it instead.

**With local registration, tiled differs from direct**: the field is estimated from the window, with
the window's faces as boundary and a pyramid over the window, so the registered moving rounds
differ near and away from the faces, and with them extraction, decoding and the filter. Molecules
that both find stay at the same voxel and gene; some appear in one only. Measured on the synthetic
fixture (one demons step, halo 12): 2 × 2: direct 68, tiled 70, every direct molecule matched by a
tiled one of the same gene within one voxel (Chebyshev), 2 tiled only; 3 × 3: direct 68, tiled 69,
67 matched both ways. **Proposed tolerance** for the validation (D10, `extended`): at least 95 % of
direct's molecules have a tiled molecule of the same gene within one voxel and at least 95 % of the
tiled molecules have a direct one, and the counts differ by at most 5 %. The numbers rest on a
fixture of 70 molecules; the validation issue confirms or replaces them on `thick_medium` with a
deformation preset (W-645).

## The resource model

Per task, on one CPU with one thread, from the measurements below. `V = Z · Y · X` is the FOV's voxel count
per channel, `R` the pipeline rounds, `C` the channels, `s` the bytes per voxel of the chunked checkpoint, `T`
the tile edge and `h` the halo.

| Task | Peak memory | Checked against |
| --- | --- | --- |
| `tile_global` | Set by global registration, not by the rounds held: about 55 to 62 bytes per voxel of `V`, the transient arrays of a phase correlation of two whole volumes; between steps the task holds the reference round and one round | aging 8,236,257,280 bytes at `9a0ec77` (54.5 per voxel; 8,237,187,072 at `56c35b8`; the trace: 4.67 GiB after the first round's preprocessing, 7.63 GiB after the first `register`, 7.67 GiB at the end, 1.48 GiB held between steps), `thick_medium` 6,506,643,456 (62.1 per voxel) |
| `fov` (direct, streaming) | The larger of the global task's and the whole FOV's detection and readout | `thick_medium` 6,499,131,392 (equal to its global task within 0.2 %), aging 9,482,248,192 at `9a0ec77` (the global task's plus 1.25 GB) |
| `tile` | `a + b · W`, with `W = R · Z · (T + 2h)² · C · s` the bytes of the window of every round: `a` = 0.207 GB, `b` = 1.74 fitted on `thick_medium`; `a` = 0.181 GB, `b` = 1.93 fitted on the real FOV | `thick_medium`: 21 tiles of three sizes (1 × 1, 2 × 2, 4 × 4), largest residual 1.1 %. The real FOV at `9a0ec77`: 20 tiles of two sizes, largest residual of its own fit 1.5 %; `thick_medium`'s fit predicts them 8.9 % low on average at 1024 and 4.9 % low at 512 (at most 9.3 %), so the plan uses the larger slope, `a` = 0.21 GB and `b` = 1.93, which bounds every measured tile (`thick_medium`'s 1 × 1 tile 9 % below it). An earlier 8 × 8 tile at `56c35b8`, stopped in its readout: at least 0.388 GB |
| `tile_merge` | Small: the candidate and read tables | `thick_medium`: 0.16 GiB, 1.3 s for 8,018 candidates; the real FOV: 0.50 to 0.54 GiB, 35 s for 766,104 |

| Task | Wall time | Checked against |
| --- | --- | --- |
| `tile_global` | Per round: load, rotation, the preprocessing steps and, for a moving round, registration; registration dominates | aging 983 s (at `56c35b8` and at `9a0ec77` alike): per round 6.8 s load, 3 s rotation, 23 s preprocessing, 73 s registration (eight moving rounds: 588 of 925 s), 4.4 s writing the round; 14 s copying into HDF5; `thick_medium` 182 s |
| `tile` | A fixed part per task (the interpreter, the translation and the codebook) plus detection over the window and readout per candidate | the codebook: 0.1 s for 64 genes, 27 s for aging's 14,242 entries (a trial); `thick_medium` tiles of 512: 24.8 s, of 256: 7.3 s; their sum per grid: 103 s (1 × 1), 99 s (2 × 2), 119 s (4 × 4); the real FOV at `9a0ec77`, tiles of 1024: 254 to 271 s, of 512: 83 to 94 s; their sums 1,051 s (2 × 2) and 1,408 s (4 × 4) |
| `fov` (direct) | The global task's time plus the readout of every candidate | `thick_medium` 234 s (global 182 s); aging 1,841 s at `9a0ec77` (global 983 s, readout 858 s); at `56c35b8` more than 2 h 4 min (stopped in its decoding, below) |

**What tiling bounds.** Tiling bounds the memory of the tile tasks, not of the global task: on both datasets
the global task's peak is direct's or close to it, because both are set by registering whole rounds before any
detection. A FOV whose direct run does not fit a job does not fit as a global task either, unless global
registration needs less memory (an open point, OT5). What tiling buys today is the split of detection and
readout into tasks that run in parallel, and a place for local registration, which costs memory and time per
window. On one CPU tiled costs more than direct (the real FOV: global, tiles and merge 2,069 s for 2 × 2 and 2,426 s
for 4 × 4, against direct's 1,841 s); with one CPU per tile the FOV's wall time is the global task's plus the longest
tile's plus the merge, 1,289 s and 1,112 s.

**The tile-size check.** The plan predicts a tile's peak with `a + b · W` for `processing.tiled.tile_size_yx`
and refuses a size whose prediction exceeds `execution.tasks.tile.memory_mb`, naming the largest multiple of
128 that fits. `a` and `b` are the package's constants, measured as here and kept with their tests.

**Decoding with a large codebook.** At `56c35b8`, the base this page describes, WTA decoding builds the
codebook's sequence-to-gene table once per read (`_run_wta`, `starfinder/readout/decoding.py:215`, evaluates
`Codebook.seq_to_gene`, a property that builds a dictionary of the whole codebook, `codebook.py:459-461`), so it
costs about 53 ms per read with aging's 14,242 entries (a trial: 500 reads in 27 s, 2,000 in 106 s) and hours for a
FOV's hundreds of thousands of reads, in direct and in tiles alike: the first real-FOV attempt was stopped for it
(below). W-722 (commit `9a0ec77`, not yet merged into this base) builds the lookups once per call: 0.21 ms per read
on its 14,000-entry fixture, the decoded tables unchanged (OT4). With it the real FOV's whole readout, extraction
included, took 858 s.

## Measurements

Every task on CPU 10, one thread, its own process under `/usr/bin/time -v` (peak RSS: "Maximum resident set
size"; wall: "Elapsed"); halo 6; chunks (Z, 128, 128, C); the run's `results/tables/measurements.csv` holds
every attempt, `resources.csv` every step.

The real-FOV rows were measured with the package at W-722's commit `9a0ec77` (tree `b9ffde69`, `src/python`
exported outside every worktree, the same locked environment), which differs from `56c35b8` only in the decoders'
and two synthetic generators' lookups, with equal decoded tables, diagnostics and summary in W-722's 27 digest
comparisons; every other row was measured at `56c35b8`.

(orchestration-tiled-measurements)=

| Dataset | Task | Tile edge (grid) | Tasks | Peak RSS (GiB) | Wall (s) | Result |
| --- | --- | --- | --- | --- | --- | --- |
| `thick_medium` (100 × 1024 × 1024, 4 rounds, `uint16`) | direct, streaming | — | 1 | 6.05 | 234 | 8,018 candidates, 5,488 molecules |
| | global | — | 1 | 6.06 | 182 | checkpoint 3.36 GB, copied in 5.6 s |
| | tile | 1024 (1 × 1) | 1 | 5.64 | 103 | equal to direct |
| | tile | 512 (2 × 2) | 4 | 1.57 (each) | 24.3 to 25.1 | equal to direct |
| | tile | 256 (4 × 4) | 16 | 0.56 to 0.57 | 6.8 to 8.5 | equal to direct |
| | merge | — | 1 per grid | 0.16 | 1.2 to 1.3 | — |
| aging Position400 (36 × 2048 × 2048, 9 rounds, `uint8`; at `9a0ec77`) | direct, streaming | — | 1 | 8.83 | 1,841 | 766,104 candidates, 207,686 molecules |
| | global | — | 1 | 7.67 | 983 | checkpoint 5.44 GB, copied in 14 s |
| | tile | 1024 (2 × 2) | 4 | 2.62 to 2.64 | 254 to 271 | equal to direct |
| | tile | 512 (4 × 4) | 16 | 0.78 to 0.82 | 83 to 94 | equal to direct |
| | merge | — | 1 per grid | 0.50 to 0.54 | 35 | — |

The first attempt of `thick_medium`'s direct run (18 min 51 s, 6.57 GiB) was stopped: with the min-max to
`uint8` of its benchmark profile, one channel's noise threshold is 2 grey levels and nearly every background
maximum is a candidate; the measurement runs on the `uint16` images. The first real-FOV attempt, at `56c35b8`,
spent its 3 h before the tiles: direct was stopped after 2 h 4 min in its readout, by the trial's timing in
decoding (OT4), at 8.83 GiB, and one 8 × 8 tile was stopped at the bound with at least 0.36 GiB; its global task
took 983 s at 7.67 GiB, as at `9a0ec77`. These earlier figures are superseded by the rows above.

## What is retired

For the Python backend (D5; MATLAB's subtile and deep modes are unchanged, D7):

| Retired | Where |
| --- | --- |
| The rules `gr_single_fov_subtile`, `lrsf_single_fov_subtile`, `deep_create_subtile`, `deep_rsf_subtile` and `stitch_subtile` for `backend: python`, their adapter scripts and the resident branch of `_run_workflow` | `registration-py.smk:35-72`, `spot-finding-py.smk:14-62`, `workflow/scripts/{gr_single_fov_subtile,lrsf_single_fov_subtile,deep_create_subtile,deep_rsf_subtile}.py`, `_snakemake.py:50-70` |
| `output/subtile/<fov>/` (`subtile_coords.csv`, `subtile_data_<n>.npz`, `subtile_goodSpots_<n>.csv`) and `fovs/<fov>/subtile_<n>/run.json` | `FOV.save_subtiles`, `_export_legacy_spots`, `_layout.fov_dir` with a `subtile_id` |
| The one-based subtile CSVs and their writer | `_export_legacy_spots` (`spots.py:154`) |
| The shared `stitch_subtile.py` for Python: it stays as it is for MATLAB, which still reads its one-based files; the Python branch for the reference image's path (`stitch_subtile.py:48-49`) goes | `workflow/scripts/stitch_subtile.py` |
| The translation of `create_subtiles` (`sqrt_pieces`, `overlap_ratio`), `img_row` and `img_col` for subtile windows, `retain_images` for the creation rules, `_RESIDENT_RULES` and the resident load and rotation rules | `workflow.py:23-25`, `:580-605`, `:757` |
| `N_SUBTILE`, `SUBTILE_RUN_RECORD` and the subtile `ruleorder` for Python | `common.smk:259-264`, `:436`; `Snakefile:63-74` |

### Choice: the package's subtile API

`FOV.from_subtile`, `FOV.save_subtiles`, `SubtileConfig` with `compute_windows`, `CropWindow`,
`Dataset.subtile` (`starfinder/dataset/dataset.py:124`), `_FovPaths.subtile_dir`, and `FOV.subtile_id`
with its place in the namespace and the record folder. Today only `_snakemake.py` (`:54`, `:70`) and
`workflow.py` (`:597-598`) call them in the package; eight test files use them (`test_types.py` 15
uses, `test_coordination_contract.py` 6, `test_dataset.py`, `test_e2e.py`, `test_fov.py`,
`test_image_contracts.py` 5 each, `test_run_record_version_2.py` 4, `test_sample_method_names.py`
1), and `docs/api/dataset.rst`, `docs/migration.md:44` and `design/specs/checkpoints.md:83-84`
document them.

| Option | Cost | Consequence |
| --- | --- | --- |
| A. Remove them with the legacy route; `FOV.subtile_id` becomes `FOV.tile` (the tile id or None) in the namespace and the record identity; the tiled entries replace the API (recommended) | The eight test files change; a migration row | One way to split a FOV; the NPZ format, one-based corners and the overlap ratio leave the package; `run.json`'s `subtile_id` field is kept for reading version-2 records and written null |
| B. Deprecate for one release, then A | A warning path and its tests | Notebooks that crop by `SubtileConfig` keep working for a while; no user outside the lab is known |
| C. Keep them as a general crop API | Two windowing rules in the package (`compute_windows` with an overlap ratio and the tile grid with a halo) | Users may build tiles that are not exact without noticing |

Recommendation: A. The tiled entries (`tile_global`, `tile`, `tile_merge`) are the package's API for
a FOV in tiles, and the reader of the chunked checkpoint by window (a function of `starfinder.io`)
is the crop a notebook needs.

## Declared files of the tiled task kinds

The contract's open point OP1 (`orchestration.md`, "Open points"), with `<fov folder>` =
`_layout.fov_dir(output_root, fov)`:

(orchestration-tiled-outputs)=

| Task kind | Reads | Declared outputs |
| --- | --- | --- |
| `tile_global` | The FOV's raw images, the codebook (for the record), the supplied file | The reference image; `<fov folder>/tiled/registered.h5`; `<fov folder>/tiled/global.json`; `<fov folder>/tiled/run.json` |
| `tile` | `registered.h5` (its window), `global.json`, the codebook | `<fov folder>/tiles/r<row>c<col>/candidates.csv`, `reads.csv`, `tile.json` |
| `tile_merge` | Every tile's three files, `global.json`, `tiled/run.json` | `run_record` (`<fov folder>/run.json`), `spot_table` (`<fov folder>/molecules.csv`), the `candidates` and `pre_qc` stages of `processing.checkpoints`; it removes `tiled/registered.h5` after success unless `registered` is kept |

Every kind also declares its task record and log (D8). The configuration keys are
`processing.mode` (`direct` or `tiled`), `processing.recipe` (null or `deep`) and
`processing.tiled`: `tile_size_yx`, `halo` (null: the plan's minimum), `local_registration` (null
or a registration recipe), `post_local_registration` (null or preprocessing steps run in the tile
after its local registration), `exact` (true: refuse operations without a finite footprint).

## Choices in brief

| Choice | Recommendation |
| --- | --- |
| The format of the chunked checkpoint | HDF5, one file per FOV, uncompressed |
| The chunk edge | 128, fixed by the format |
| The chunked checkpoint and `registered` | The tiled form of `registered`; kept only when `registered` is configured |
| The identities of a tiled FOV | FOV IDs by the detector's order, assigned by the merge |
| How the tile size is chosen | `processing.tiled.tile_size_yx`, checked against the task memory at plan time |
| The halo | Computed by the plan from the footprints; a larger configured value accepted |
| What a tile writes | The core's candidates and reads; the merge deduplicates, filters and writes |
| The threshold of a tile | A new `threshold_mode: absolute`, filled from the global task |
| How the deep recipe is expressed | A named recipe expanded by the plan |
| The package's subtile API | Removed with the legacy route; the tiled entries replace it |

## Open points

* **OT1 (W-645).** The local registration margin and the tolerance of tiled against direct with
  local registration are proposals; the validation design calibrates them on `thick_medium` with a
  deformation preset.
* **OT2.** Each detector's ordering key and footprint as registry fields (`SPOT_FINDING_METHODS`);
  until they exist the plan accepts `local_maxima` (with `min_distance_voxels` 1 and no merge
  radius) and refuses the others in tiled mode with `exact: true`.
* **OT3.** `FOV.load_registered_round` reading the HDF5 form of `registered`, for a segmentation run
  on a sequencing round of a tiled FOV.
* **OT4 (fixed by W-722).** WTA decoding's per-read rebuild of `Codebook.seq_to_gene` (`decoding.py:215`), which
  made the decoding of a large codebook take hours at `56c35b8`. W-722 fixed it (commit `9a0ec77`, branch
  `runner/ot4-decode-20261029`); the fix is not yet merged into the base this page describes, and the real-FOV
  measurements above were made with it.
* **OT5.** The global task's memory, set by whole-volume registration. The gate (W-646, follow-up F1) splits it:
  memory cuts that keep every result (a smaller transform type, in-place transforms, release between rounds) are a
  draft issue of §2.13, checked against identical shifts; registering a projection or a reduced scale, as MATLAB's
  deep script registers at 0.25, changes results and stays a registration follow-up with its own design and benchmark.
