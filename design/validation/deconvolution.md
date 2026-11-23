# Validation design: the deconvolution stage

% Status: Approved with the specification (Jiahao, 2026-11-21, W-918); the scaling checks follow decision 3 as changed. Written against `70edc72` (v0.3.0).

This is the validation plan for implementing [deconvolution](../specs/deconvolution.md). It has
three parts:

* **Package checks** on synthetic fixtures, in the test suite: the plan and the engines, the
  scaling, the PSF, the wisdom store, the records and reuse. They run deconwolf through a stand-in
  executable, so they need no binary.
* **deconwolf and image checks** that need the real binary: a few small tests that skip without
  it, and the image's own checks on the host and on UGER.
* **Real-data checks**, run as a separate evidence issue (W-889 D3): the stage on 9735seqA1
  Position022 reproduces W-915's deconwolf output, and the CPU mode, the wisdom store and reuse
  work on a UGER RedHat 8 node.

The real-data checks are development checks. They make no data claim: agreement with W-915 is
repeatability of one program, not accuracy, and similarity to Huygens is not accuracy (W-855).

The expected values below come from the specification, from deconwolf's documented behaviour as
W-915 and W-916 found it, and from their measurements, named in each row. They do not come from
running the new code.

**Tolerances** follow `docs/contributing.md` ("Expected values and tolerances"):

* Hashes, integer arrays, counts, strings, exit statuses, record keys and TIFF tags are exact. A
  stored copy made by the stand-in, uint8 or uint16, is an integer array and is compared exactly.
* The output of the real deconwolf is compared with its stated bound, built from the quantization
  steps and from W-916's agreement between modes (1.6 × 10⁻⁵ of the maximum, rounded up to
  2 × 10⁻⁵). No bound is invented beyond those.
* Times, memory and sizes are reported against their stated resource bounds; they are not
  results.

## Fixtures

Every routine fixture is at most 32 × 64 × 64 voxels with at most four channels and four rounds,
as the project guidance bounds them.

| Fixture | Construction | Needs |
| --- | --- | --- |
| `stand_in_dw` | A Python script installed as `dw` in a temporary folder and named by `execution.deconwolf`. It accepts v0.4.5's options that the stage passes (`--iter`, `--threads`, `--gpu`, `--tilesize`, `--noplan`, `--float`, `--offset`, `--out`, the image and the PSF). It writes the float32 output y = x + 5 untiled and y = x tiled, so that the expected copy is known exactly, and a log with `scaling:` and `Iteration` lines. Unless `--noplan` or `--gpu` is given, it writes `fftw_wisdom_float_inplace_threads_<n>.dat` where v0.4.5 would: under `$HOME/.config/deconwolf/` when `$HOME/.config` exists, else in its working directory. It appends its arguments, `HOME`, `OMP_NUM_THREADS`, its working directory and the wisdom files it found to a call log. `--version` prints a v0.4.5 text. Options make it fail on a given channel, or write an output whose minimum breaks the offset rule. | hand-built |
| `input_root` | A synthetic input root of 2 FOVs × 2 rounds × 2 labels, 8 × 32 × 32 uint8, sparse Gaussian spots, in the default `{round}/{fov}` layout, with `fovs.csv`, a codebook and one reference stain label, `dapi`, in the reference round's folder beside its sequencing labels `ch00` and `ch01`. Raw names hold the FOV id, as real names do (`Position001_ch00.tif`). `voxel_size_xy` 0.1 and `voxel_size_z` 0.3. Every channel is a plain TIFF written by `tifffile.imwrite` without `starfinder_metadata`, the case of raw microscope files: `load_volume` reports `metadata_source` `unknown`, and each channel's frame is built from its own path (`io/tiff.py:260`, `:268`). | hand-built |
| `input_root_stored` | `input_root` with every channel written by `save_volume` with one `ImageMetadata` per round folder: frame id `raw:<round>/<fov>`, spacing (0.3, 0.1, 0.1) µm, a non-zero origin. All channels of a folder share it, so the raw round loads. | hand-built |
| `workflow_examples` | The worked-example configurations of `test/workflow_examples.py` (X), with a `deconvolution` block added per row. | existing |
| `wisdom_store` | A store folder with a manifest and one wisdom file for the key the stand-in's task computes, and one for another key. | hand-built |
| `psf_small` | A 7 × 9 × 9 float32 Gaussian with full support: G(z, y, x) = exp(−(z − 3)²/2 − ((y − 4)² + (x − 4)²)/(2 × 1.5²)), computed in float64, divided by its sum, then cast to float32. The peak is at the centre (3, 4, 4). Also the same array rolled by +2 planes and +1 row (`np.roll`, shift (2, 1, 0)). | hand-built |
| `psf_core` | A 7 × 9 × 9 float32 array with zero support near the edges: G of `psf_small` on the core z ∈ {2, 3, 4}, y ∈ {2, …, 6}, x ∈ {2, …, 6}, exactly 0 elsewhere, divided by its sum in float64, then cast to float32. Also the same array rolled by (2, 1, 0). The roll moves the core to z ∈ {4, 5, 6}, y ∈ {3, …, 7}, inside the array, and the slices it wraps (planes 0–1 and row 0 of the rolled array) are all 0. | hand-built |
| `real_dw` | The deconwolf binary, when `dw` is on `PATH` or `STARFINDER_TEST_DECONWOLF` names it. Tests that need it skip with that reason otherwise. They do not run in the CI environment. | the image, or a host build |

(deconvolution-checks)=

## Checks

"Needs": **H** hand-built or existing fixtures only; **WF** the workflow fixtures; **D** `real_dw`;
**P** the `deconvolution` extra (psfmodels), skipped with the extra named; **S** `slow`. Every row
without D or P passes or skips in the CI environment (extras `dev`, `local-registration`,
`checkpoint`).

### The plan (PL)

| # | Check | Fixture | Expected | Tolerance | Needs |
| --- | --- | --- | --- | --- | --- |
| PL1 | No block, no change | `workflow_examples` without the block | The committed C1 manifests (`data/expected/test_workflow_plan/`) are unchanged, and so is every task's `blocks_sha256` and `fingerprint`; X4 and G3 report no difference. | exact (files, hashes) | WF |
| PL2 | The graph | `input_root` with the block, `rounds: null` | One `psf` task with no needs. One `deconvolve` task per (FOV, sequencing round), ids `deconvolve/<sample>/<fov>/<round>`, each needing `psf`, in one group per round name (`deconvolve.<round name>`). Each declares its record folder `<copy>/records/<round name>/<fov>/`, and no two tasks declare one output. Every `fov` and `summarize` task needs the `deconvolve` tasks of its FOV. Its declared inputs name `<copy>/<round>/<fov>/<label>.tif` and no raw sequencing file, and its entry holds `deconvolved`. The `morphology` task still reads the raw stain. With `rounds` naming `reference_stain` too, it reads the copy's stain file and needs that task. | exact (ids, lists) | H |
| PL2b | The reference round and its stains | `input_root`, `rounds` naming both sequencing rounds and `reference_stain` | For each FOV, two tasks write into the reference round's image folder: `deconvolve/<sample>/<fov>/<ref round>`, declaring `ch00.tif`, `ch01.tif` and `records/<ref round>/<fov>/`, and `deconvolve/<sample>/<fov>/reference_stain`, declaring `dapi.tif` and `records/reference_stain/<fov>/`. The two tasks' declared outputs are disjoint, and they are in different groups whose templates the plan accepts. | exact (paths, sets) | H |
| PL3 | Refusals | the block varied | Each case of the specification's "Plan refusals" exits 2 with `PlanRefused` naming the key: `backend: matlab`; `iterations` absent, 0 or 2.5; an unknown round in `rounds`; a round no consumer reads; a missing `voxel_size_xy` or `voxel_size_z`, with `generate` and with `file`; a label's wavelengths missing with `generate`; a missing or absent file, or no `psf.voxel_size_um`, with `file`; a declared spacing that differs from the dataset's by 2 parts in 10⁶ in Y and X, and again in Z (one part in 10⁷ is accepted); a stain label equal to a sequencing label of the reference round, and a stain label `ch0` beside the sequencing label `ch00`, with both rounds deconvolved; a `tile_size` below the PSF's lateral size plus 40; `dtype: float32`; a scale of 0 and a scale mapping missing a (round, label), both under `dtype: uint16`; a relative `copy_root` other than `output` and `beside`. | exact (exit status, exception type, key named) | H |
| PL4 | What is hashed | `input_root`, planned twice with one change each | Changing `iterations`, `tile_size`, `dtype`, `scale` (under `dtype: uint16`), `border_quality`, `rounds` or a PSF parameter changes the copy id, the `deconvolve` and `psf` fingerprints and every consumer's fingerprint. So does changing `voxel_size_xy` alone, and `voxel_size_z` alone. With `psf.source: file`, rewriting a PSF file with other values at the same path changes the copy id. Under `dtype: uint16`, `scale: null` and `scale: 64` give one copy id. Changing `execution.tasks.deconvolve.threads` or `gpu`, `execution.deconwolf` or `execution.deconwolf_wisdom` changes no `blocks_sha256`, fingerprint or copy id. | exact (hashes) | H |
| PL5 | Selection growth | `input_root`, `selected_fovs` of one FOV, then both | The first FOV's `deconvolve` tasks and the `psf` task keep their fingerprints (W-877's rule). | exact (hashes) | H |
| PL6 | The copy's place | `copy_root` `output`, `beside` and an absolute path | The copy folder is `<output root>/deconvolved/<sample_id>/dw-<id>`, `<root_input_path>/<dataset_id>-dw/<sample_id>/dw-<id>` and `<P>/<dataset_id>/<sample_id>/dw-<id>`. `<id>` is the first 12 hex digits of the SHA-256 of the canonical JSON of the specification's seven fields (`dataset_id`, `sample_id`, `image_dir_template`, `voxel_size_xy`, `voxel_size_z`, the block less `copy_root`, `psf_files`), computed independently in the test. Paths outside the output root are absolute in the manifest. | exact (strings) | H |
| PL7 | A saved set needs no sequencing copy | `input_root` with `processing.spot_finding_input` and no `summarize` | No `deconvolve` task for a sequencing round; the plan does not refuse. | exact (ids) | H |
| PL8 | The fingerprint's inputs | `copy_root` absolute, the copy not yet made | The plan succeeds. The consumers' fingerprints do not stat the copy's files; they change when a `deconvolve` fingerprint changes. | exact (hashes) | H |

### The engines (EN)

| # | Check | Fixture | Expected | Tolerance | Needs |
| --- | --- | --- | --- | --- | --- |
| EN1 | Snakemake | `input_root`, `snakemake -n`, with `reference_stain` deconvolved | One rule per group: each `deconvolve.<round name>` group's rules have the wildcard `fov`, and `round` for a sequencing round; Snakemake finds no two rules for one output. A task with `gpu: true` asks `resources: gpu=1`. The order is `psf`, `deconvolve`, then the consumers. | exact (rule names, wildcards) | WF |
| EN2 | Nextflow | `input_root`, `-stub-run` | `checkedTasks` accepts both kinds. `PSF` and `DECONVOLVE` run before `SUMMARIZE_0`, and a `gpu` task runs as `DECONVOLVE_GPU`. | exact (process names, order) | WF |
| EN3 | Engines agree | `test_workflow_engine_comparison.py` with the block and `stand_in_dw` | Snakemake and Nextflow write byte-equal copies, records less their time and host fields, and consumer outputs. | exact (hashes) | WF, S |

### The scaling and the copy (SC)

These codes are this page's own. SC1 checks the specification's default form, SC3 (uint8 with a
factor per round), and SC1b its option, SC1 (uint16 with a scale).

| # | Check | Fixture | Expected | Tolerance | Needs |
| --- | --- | --- | --- | --- | --- |
| SC1 | Offset and stretch factor, untiled (the default uint8) | `stand_in_dw`, untiled, the default `dtype` | The stand-in returns x + 5. With M the maximum of x over the round's channels and f = 255 / M, each copy file equals `clip(rint(x × f), 0, 255)` of its raw channel as uint8, with one f for all channels of the round. The record says kept offset 5 and holds f and M under `storage`. A round whose x is all 0 stores zeros, with f = 1. | exact (integer arrays; f and M as the float64 the record holds) | H |
| SC1b | Offset and scale, untiled (uint16) | as SC1 with `dtype: uint16`, scale 64 | Each copy file equals `clip(rint(x × 64), 0, 65535)` of its raw channel as uint16. | exact (integer arrays) | H |
| SC2 | Offset and scale, tiled | as SC1 and SC1b with `tile_size` | Kept offset 0; the same expected array. | exact | H |
| SC3 | The offset rule is checked | `stand_in_dw` writing an untiled output with minimum 0; `stand_in_dw` whose `--version` names 0.4.6 | Each task fails, naming the rule or the version; no copy file or record is left; the failed record lists what was removed. A tiled output with minimum 5 does not fail, and its record holds the minimum. | exact (exit status, strings) | H |
| SC4 | Saturation | `dtype: uint16`, raw values that exceed 65,535 / s with a large scale | Those voxels store 65,535, the record counts them exactly, and the log warns once. Negative values after the offset store 0 and are counted, at uint16 and at uint8 (a stand-in output below the offset). | exact (counts) | H |
| SC5 | Scale per round and label | `dtype: uint16` with a mapping scale; a plan with `scale` and the default `dtype` | Each file uses its (round, label) scale; the record holds it. The plan with `scale` and uint8 is refused, naming the key. | exact | H |
| SC6 | The file and its metadata | the copies of SC1 from `input_root` (plain TIFFs) and from `input_root_stored` | ZYX uint8 (SC1) and uint16 (SC1b), TIFF Compression 8 and no Predictor tag (W-909's codec). Only `<label>.tif` files lie in the image folder. From `input_root`: no copy file holds `starfinder_metadata`, and `load_volume` reports `metadata_source` `unknown` for each, as for its raw file. From `input_root_stored`: each copy file holds `starfinder_metadata` equal, field for field, to its raw file's, and `load_volume` reports `stored`. | exact (tags, strings, dataclass equality) | H |
| SC7 | The copy loads as the raw round loads | the copies of SC6; `load_round` with the configured labels on the raw folder and on `<copy>/<image dir>`, for every round with two or more channels, the reference round with its stains included | Both loads succeed through the same loader, neither raising "channel frames differ" or "channel geometry differs". Labels are in configured order, the dtype is the copy's (uint8 by default), and the arrays are SC1's. The geometry (`spacing_zyx`, `origin_zyx`, `direction_zyx`, `spatial_unit`) of the copy's round equals the raw round's. From `input_root`, the copy round's frame id is the copy folder's resolved path, as the raw round's is its folder's (`io/tiff.py:449`). From `input_root_stored`, the two rounds' `ImageMetadata` are equal, the frame id included. A negative control stores the raw file's path-built `ImageMetadata` in each copy file of a plain round: `load_round` then raises "channel frames differ", which is the defect this rule avoids. A `fov` task of `workflow_examples` with the block runs to `succeeded` on the copy of each fixture. | exact (strings, dataclass equality, integer arrays) | H, WF |
| SC8 | The warning for intensities across FOVs | a plan with the block at the default `dtype` and a preprocessing step with `fit: supplied`; one with a `global` spot threshold; one with an `absolute` threshold; the three again with `dtype: uint16` | The first three plans warn once each among the plan's warnings, naming `dtype: uint16`; the uint16 plans, and a uint8 plan that feeds none of the three, do not warn; no plan is refused. | exact (strings) | H |

### The PSF (PS)

| # | Check | Fixture | Expected | Tolerance | Needs |
| --- | --- | --- | --- | --- | --- |
| PS1 | Centring | `psf_core` and `psf_small`, unrolled and rolled | **(a) Zero-supported, rolled.** The peak moves to the centre (3, 4, 4), and the record's shift is (−2, −1, 0). The centring rolls back by (−2, −1, 0) and zeroes the slices it wraps (result planes 5–6 and row 8), which hold only zeros. So it loses nothing: the record's sum dropped is 0, and the result equals the unrolled `psf_core`. **(b) Full support, rolled: an expectation by formula.** The centring zeroes the same slices, which now hold G's tails. The expected array is T / ΣT, where T is the unrolled `psf_small` with planes z ∈ {5, 6} and row y = 8 set to 0, and ΣT is its sum in float64. The test evaluates that formula; it is not equal to `psf_small`. The record's sum dropped equals 1 − ΣT. The result's peak is at the centre and its sum is 1. **(c) Unrolled.** For both fixtures the shift is (0, 0, 0), nothing is zeroed, and the result equals the fixture. | float32 arrays: `rtol=1e-6`, `atol=1e-6` (`docs/contributing.md`, "Expected values and tolerances": float32 computed values, and the dtype's tolerance for an analytic expectation); the sum dropped and the sums: the same float32 tolerance; indices and shifts: exact | H |
| PS2 | A supplied file | `psf_small` as `psf.files`, declared spacing equal to `input_root`'s | `psf.json` holds the path, file and decoded SHA-256, and the declared spacing. A file not of sum 1 is renormalised in memory, and the record says so. A file with a negative value, or not ZYX, fails the `psf` task. | exact; 1e-9 (sum) | H |
| PS3 | Generation | W-855's parameters for 9735seqA1 ch00 at depth 0 (NA 1.30, ni 1.518, ns 1.338, 499/529 nm, pinhole 307 nm, voxels 0.094635 and 0.346255 µm, 21 × 33 × 33) | Shape 21 × 33 × 33, peak index (10, 16, 16), no centring shift, sum 1. Axial and lateral FWHM through the peak are 0.4933 and 0.2362 µm, as W-855's `psf_params.json` records them (from its 4-decimal record). `psf.json` records `dxy` 0.094635 and `dz` 0.346255. | 1e-4 µm (FWHM, the record's rounding); exact (indices, recorded parameters); 1e-9 (sum) | P |
| PS4 | The spacing reaches the PSF | PS3's parameters, then the same with `voxel_size_xy` 0.0473175 (half) | The second PSF's file SHA-256 differs from the first, its `psf.json` records `dxy` 0.0473175, and its lateral FWHM in voxels is about twice the first's (the FWHM in µm is the same within one voxel of the finer grid). | exact (hashes, recorded parameters); 1 voxel (FWHM) | P |

### FFTW wisdom (WI)

| # | Check | Fixture | Expected | Tolerance | Needs |
| --- | --- | --- | --- | --- | --- |
| WI1 | The key | hand-built field values | The key is the SHA-256 of the canonical JSON of the specification's fields. Each field changed alone changes it. Two PSFs of one shape and different values give one key. | exact (hashes) | H |
| WI2 | A matching key | `wisdom_store`, a CPU task | The stand-in's call log shows `HOME` under the task's temporary root, its working directory there too, and the store's file found under `$HOME/.config/deconwolf/` with the name for the task's threads; no `--noplan`. The record says `measure`, the key and the file's SHA-256. The store's files and manifest have the same SHA-256, size and modification time after the task. | exact | H |
| WI3 | No match, or no store | `wisdom_store` with another key; `deconwolf_wisdom: null` | `--noplan`, an empty wisdom folder, and the record's reason (`no matching key`, `no store`). | exact | H |
| WI4 | A damaged store | `wisdom_store` with a file whose SHA-256 differs from its manifest | The task fails, naming the file. | exact | H |
| WI5 | Nothing written outside the task | a CPU task with `stand_in_dw` writing wisdom back | After the task, no file is new or changed outside the task's temporary root and its declared outputs: not the store, not the engine's working directory, not the real `HOME`. | exact (file lists, hashes) | H |
| WI6 | One generating job per key | `starfinder deconvolve wisdom` with `stand_in_dw` | It adds the key's file and its manifest entry. A second call for the key exits non-zero without writing. Two concurrent calls leave one entry and one file, and the manifest parses. | exact | H |
| WI7 | GPU tasks | a task with `gpu: true` | `--gpu --threads 1`, an empty wisdom folder, and the record's planning `gpu`. | exact | H |

### Records (RC)

| # | Check | Fixture | Expected | Tolerance | Needs |
| --- | --- | --- | --- | --- | --- |
| RC1 | The round record | SC1's run | Every key of the specification's table. Each input and output `file_sha256` and decoded `sha256` equals a hash computed independently in the test. The PSF hash equals `psf.json`'s; `iterations_done` equals `iterations`; the binary's SHA-256 is the stand-in's. | exact | H |
| RC2 | `copy.json` | SC1's run | The copy id and its seven fields: the dataset and sample, the template, `voxel_size_xy` and `voxel_size_z`, the block less `copy_root`, `psf_files`; and `psf.json`'s SHA-256. Recomputing the id from the recorded fields gives the recorded id. | exact | H |
| RC3 | `run.json` | a `fov` task and a `morphology` task on the copy, `reference_stain` deconvolved | `deconvolution` holds the copy's absolute path, its id and, per round name, the record it read with its SHA-256: the `fov` task names the reference round's record, the `morphology` task the `reference_stain` record. `inputs[]` name the copy's files, absolute when outside the input root, with SHA-256 equal to that round record's output `file_sha256`. | exact | H, WF |
| RC4 | Completion and failure | `stand_in_dw` failing on the second channel | The task record of a success lists every declared output with its SHA-256. A failure leaves no copy file, round record or task record; it leaves the failed record, with the removed paths, and the log. | exact | H |
| RC5 | A shared image folder keeps two records | `input_root`, `rounds` naming the reference round and `reference_stain`, run once in each order, and once with both tasks started together | Both tasks succeed. `records/<ref round>/<fov>/deconvolution.json` names `ch00` and `ch01` and their logs; `records/reference_stain/<fov>/deconvolution.json` names `dapi` and its log; each log exists with the SHA-256 its record holds. Then the stain's raw file is rewritten and the run repeated: the stain task redoes its round, and the reference round's task reuses its own. The reference round's record, logs and label files keep their bytes and modification times, and `starfinder workflow verify` calls its task complete. The same holds the other way round, with a sequencing raw file rewritten. A failure of the stain task (`stand_in_dw` failing on `dapi`) removes only `dapi.tif` and the stain's record folder. | exact (hashes, file lists, statuses) | H, WF |

### Reuse and staleness (RU)

| # | Check | Fixture | Expected | Tolerance | Needs |
| --- | --- | --- | --- | --- | --- |
| RU1 | Verified reuse | `copy_root` absolute; two output roots planned and run one after the other | In the second run the `psf` task writes nothing, and every `deconvolve` task succeeds with `reused: true` and no stand-in call. Every file of the copy, `copy.json` and the PSFs included, keeps its bytes and modification time. | exact | H, WF |
| RU2 | A changed raw file | RU1, one raw channel rewritten with other values | That round is redone (one stand-in call per label of the round), its record is replaced, and the other rounds are reused. | exact | H, WF |
| RU3 | A changed setting | RU1 with `iterations` changed | A new copy id and folder; the first copy is unchanged (hashes). | exact | H, WF |
| RU4 | A consumer checks | a copy file replaced after its record | The `fov` task fails, naming the file and both hashes. | exact | H, WF |
| RU5 | Resume and saved sets | a `fov` task stopped after `registered`, then the copy redone with other values; a saved set made without the block, read by a run with it | The resume is refused on the input hashes. The saved set is refused, naming both copy ids (one absent). A set and a run with the same copy id are accepted. | exact | WF, S if over 5 s |
| RU6 | One writer per round | two `deconvolve` tasks of one (round name, FOV) started together | One writes, the other waits on the lock in `records/<round name>/<fov>/` and then reuses the round. One stand-in call per label in total. | exact (counts) | H |
| RU7 | `code_version: rerun` | RU1 with another stand-in (another SHA-256) | Under `keep` the round is reused; under `rerun` it is redone. | exact | H, WF |
| RU8 | A change of spacing alone | RU1, then a third run with `voxel_size_xy` changed and nothing else; with `psf.source: file` the declared `psf.voxel_size_um` is changed to match | A new copy id and folder. The `psf` task writes the new copy's PSFs and `psf.json`, which records the new spacing, and every `deconvolve` task deconvolves again into the new folder (stand-in calls for every label). The first copy keeps every file's bytes and modification time. The same with `voxel_size_z`. A fourth run with RU1's configuration again reuses the first copy's PSF and every round with no stand-in call. | exact (hashes, counts) | H, WF |
| RU9 | A changed supplied PSF | RU1 with `psf.source: file`, then the PSF file rewritten with other values at the same path | A new copy id and folder, holding the new PSF; every round is deconvolved again there. The first copy is unchanged. | exact | H, WF |
| RU10 | A damaged copy's PSF | RU1's copy with a PSF file rewritten after its `psf.json`, and again with a recorded spacing in `psf.json` edited | The `psf` task of the next run fails, naming the file or the field, and writes nothing. | exact (exit status, strings) | H |
| RU11 | The spacing reaches a generated PSF in a reused copy | RU8 with `psf.source: generate` | As RU8, and the two copies' PSF files differ (PS4). | exact | P, WF |

### deconwolf itself (DW)

| # | Check | Fixture | Expected | Tolerance | Needs |
| --- | --- | --- | --- | --- | --- |
| DW1 | The options exist | `real_dw` | `dw --version` names 0.4.5. `dw --help` lists every option the stage passes: `--iter`, `--threads`, `--gpu`, `--tilesize`, `--noplan`, `--offset`, `--out` and `--float`. If `--float` is missing, the specification's fallback (deconwolf's uint16 and its logged scale) is the implemented path, and RD1's and RD1b's bounds gain one step of that scale. | exact (strings) | D |
| DW2 | The offset rule and determinism | `real_dw`, 16 × 64 × 64 point sources blurred by `psf_small`, CPU, `--noplan`, 3 iterations; the peak in its own untiled run at 6 iterations | The untiled float output's minimum is within 0.5 of 5; a tiled run (`--tilesize 32`) has minimum below 0.5. Two untiled runs at 1 and 2 threads are byte-identical (W-916: thread counts change no byte). In its own untiled run with `--iter 6`, the brightest point's peak less the offset 5 rises above its input value. v0.4.5's `--iter 3` makes two Richardson–Lucy updates, after which the output at that point, offset included, is 25.28 against the input's 28.35; at `--iter 6` the peak less 5 measured 54.355 on the CPU, the A5000 and the A30 (amended 2026-11-22, Jiahao's decision D2, W-929). | exact (hashes); the 0.5 bands of W-916 | D |
| DW3 | The stage on the real binary | `input_root` with `real_dw`, CPU | The tasks succeed; SC6 and RC1 hold; the record's planning is `estimate`. | exact | D, S if over 5 s |

### MATLAB (MT)

| # | Check | Expected | Needs |
| --- | --- | --- | --- |
| MT1 | MATLAB is untouched | With `backend: matlab`, a configuration with the block is refused (PL3). The MATLAB rule set and its schema are byte-identical to `70edc72`. No MATLAB run is needed: no file MATLAB reads changes. | H |

### What the default tier covers

PL1–PL8 with PL2b, EN1, SC1–SC8 with SC1b, PS1, PS2, WI1–WI7, RC1–RC5, RU1–RU4, RU6–RU10 and MT1 run in the
routine tier, each under 5 s on one CPU or marked `slow`. EN2 and EN3 run in the engine
selections; PS3, PS4 and RU11 where the extra is installed; DW1–DW3 where the binary is. The implementation reports the measured
time of every new test that takes 1 s or more.

## Image checks

These run in the image's own check (`src/python/scripts/image_check.py`) and on the UGER nodes,
after the image is built, as W-878 checked v0.3.0's image. They are part of the implementation's
image record, not of the suite.

| # | Check | Where | Expected and bound |
| --- | --- | --- | --- |
| IM1 | deconwolf is in every target | `cpu`, `learned`, `learned-cuda`, no network | `dw --version` names 0.4.5 with OpenMP, OpenCL and VkFFT; `ldd dw` finds every library; the binary's SHA-256 equals the image's label. psfmodels imports in each target. |
| IM2 | RedHat 8 starts it | one UGER CPU node with `operating_system=RedHat8`, the SIF through `apptainer exec` | DW1 and DW2 pass inside the SIF: the case where W-916's host binary failed (glibc 2.29 against 2.28; libraries missing). One job, under 0.25 slot-h. |
| IM3 | The GPU in the image | one UGER GPU node (A30), `apptainer exec --nv`, `CUDA_VISIBLE_DEVICES` set to a GPU without a compute process | DW2 with `--gpu` passes. deconwolf's log names an A30. While it runs, `nvidia-smi --query-compute-apps` shows its process on the chosen GPU only. One job, under 0.25 slot-h. |
| IM4 | The host's GPU | GP099-29C, `docker run --gpus` or `apptainer --nv`, the RTX A5000 | As IM3, on the card of W-915. |

## Real-data checks (separate evidence issue)

These checks deconvolve a whole position from raw images: more than a batch may run (W-889 D3,
at most 1 CPU-hour of real-data evidence inside a batch, on saved sets or crops). They are one
evidence issue that the coordinating session runs on the implementation branch's commit, with
the built image, before the merge.

* **Records** as the project guidance defines them: an input inventory first, one wrapper per
  step, the resource table, the manifest and the verifier.
* **Inputs.** 9735seqA1 Position022 (`data/raw/9735seqA1/9735seqA1_Position022.ome.tiff`, read
  only), W-855's centred PSFs at depth 0 and W-915's kept outputs and spot tables, each checked
  against its run's manifest first.
* **The staged input root.** The package reads one file per channel, so a step writes the
  position's four channels as ZYX uint8 TIFFs into a one-round input root on local scratch
  (`round1/Position022/ch0<c>.tif`). Each decoded array equals W-855's `load_raw_channel` of that
  channel (exact, by decoded SHA-256). The histogram mode is 0 on every channel (W-855), so W-915's
  preparation (minus the mode, clipped at 0, float32) is the identity on these arrays, and the
  stage needs none.
* **Only the stage runs.** The plan holds the stage and a consumer, as the stage requires. The
  engine is given the `deconvolve` tasks' records as its targets, so no `fov` task runs.
* **One host per timed step**, named in the records. Times are reported, not compared.

### Checks

`v_w` is W-915's kept value, `stored / scaling_w − 5`, clipped at 0 (its untiled rule).
`v_s` is the copy's value at `dtype: uint16`, `stored / s`, with s = 64. `v_8` is the default copy's value,
`stored / f`, with f the round's stretch factor from its record. RD1–RD5 run at `dtype: uint16`, scale 64, so
that their bounds test the stage's arithmetic; RD1b checks the default.

| # | Check | Data | Expected and bound |
| --- | --- | --- | --- |
| RD0 | The generated PSF | The `psf` task with W-855's parameters for the four channels at depth 0 | Each decoded array equals W-855's centred PSF of that channel (`results/psf/centred/psf_ch0<c>_pz0.0.tif`). With psfmodels 0.3.3, W-855's version, equality is expected; the bound is 1 × 10⁻⁷ of the PSF's peak, for float32 rounding after a float64 normalisation. The centring shift is 0 on every channel, as W-855 found at depth 0. |
| RD1 | The stage reproduces W-915 | Position022, four channels; `psf.source: file` with W-855's centred PSFs (so that RD0 and RD1 test one thing each); `iterations: 5`; untiled; GPU on the host's RTX A5000, the card of W-915; the image's deconwolf | Per channel: max \|v_s − v_w\| ≤ 1 / scaling_w + 0.5 / s + 2 × 10⁻⁵ × max(v_w). The first term is W-915's quantization step, a full step because deconwolf's rounding rule was not recorded. The second is the copy's half step. The third covers a build or device difference at W-916's agreement between modes. For ch00 (scaling_w 101.886, max 638.2) the bound is 0.0098 + 0.0078 + 0.0128 = 0.0304 grey levels. Also per channel: NCC(v_s, v_w) ≥ 1 − 10⁻⁶ (W-916's untiled arms: 1 − 10⁻¹⁰). Spot F1 against W-915's spots at adaptive 0.2: ≥ 0.999. At noise 5.0, two conditions: the spot F1 against W-915's spots is within 0.0005 of the quantization control's spot F1 against W-915's spots, the control being W-915's values requantized to rint(v × 64) / 64; and the spot F1 against the control's spots is ≥ 0.999. The bound first written for noise 5.0, F1 against W-915's spots ≥ 0.999, rested on W-916's CPU arms, a perturbation of about this size, which gave 1.0000 and 0.999999. That premise fails for the uint16 copy: the copy's rounding is absolute, ±0.5 / 64 grey level, while W-916's mode differences were relative. The control alone gives 0.998862, 0.998598 and 0.998283 against W-915's spots for ch01–ch03 (W-925). Scale 64 stays (amended 2026-11-22, Jiahao's decision D3, option A, W-929). The records name the copy, the PSF hashes, `iterations_done` 5 and the kept offset 5. Reported: the fraction of voxels within the first two terms alone, which is 1 if the float output reproduces W-915's exactly. |
| RD1b | The default uint8 copy | As RD1, at the default `dtype` | The record's f equals 255 / M with M its recorded round maximum, and M agrees with the maximum of v_w over the four channels within RD1's bound for that channel. f is one value for the four channels. Per channel: max \|v_8 − v_w\| ≤ 1 / scaling_w + 0.5 / f + 2 × 10⁻⁵ × max(v_w) + max(v_w) × δ_M / M: RD1's bound with the uint8 half step, and a last term for f, whose M is known within δ_M, RD1's bound for the channel that holds M. Reported, not bounded: spot F1 against W-915's spots at adaptive 0.2 and at noise 5.0; the coordinating session's measurement, with a factor per channel volume, gave 0.982–0.983 and 0.83–0.91. |
| RD2 | The CPU mode on RedHat 8 | Position022 ch00, CPU, 4 threads, `--noplan`, `dtype: uint16`, scale 64, one UGER RedHat 8 node, the SIF | max \|Δv\| against RD1's ch00 ≤ 2 × 10⁻⁵ × max(v) + 1 / s, which is W-916's CPU–GPU agreement plus one step of each copy's rounding. Reported: wall time, CPU time, VmPeak and RSS. The task's `h_vmem` of 3,500 MB per slot × 4 slots holds, with no kill. |
| RD3 | The wisdom store | `starfinder deconvolve wisdom` once on RD2's node class (6 threads, ch00's shape and PSF shape), then RD2 again at 6 threads with the store | The second run's record says `measure` and names the key. Its output agrees with RD2 within RD2's bound. The store's files have the same SHA-256 before and after. Reported: the planning time against W-916's 1,617 s and the warm time against W-916's 74 s. |
| RD4 | Reuse across runs | RD1's plan again, into a second output root, with `copy_root` the same absolute path | Every `deconvolve` task says `reused: true` and ran no deconwolf. The copy's files keep their SHA-256 and modification times. |
| RD5 | The GPU on UGER | Position022 ch00, GPU, one UGER A30 node, the SIF with `--nv` | Agreement with RD1's ch00 within RD2's bound (another card). Reported: the wall time and the GPU memory peak against W-916's 35 s and 14.0 GiB. |
| RD6 | Sizes | RD1's and RD1b's copies | Per channel: each copy's bytes, and the ratio to the raw channel's 283,115,520 bytes. Reported, not bounded. W-915's zlib uint16 files, at a per-volume scale, were 310–425 MB per channel; the coordinating session's measurements are in the specification's scaling section. |

**Resources.** RD1 and RD1b are about 2.5 min of GPU each (W-916: 35 s per channel). RD2 is about 3 min at
4 threads (166 s at 4 threads in W-916), at 10 GiB. RD3's planning is about 27 min at 6
threads, about 2 CPU-hours (W-916: 7,030 CPU-s), and its warm run about 1.3 min. RD5 is about
1 min of GPU. The staging and the comparisons read about 1 GB of raw image and 1.6 GB of W-915's
outputs. In total that is about 3 CPU-hours, three UGER jobs (one GPU) and the host's GPU. The
evidence issue states its grant from these figures.

## Implemented checks

% Added 2026-11-22 by W-923, the implementation.

The table maps each check above to its tests. Files are under `src/python/test/`. "Time" is the
measured call time in seconds of the row's new tests together, on one CPU and one numerical thread,
all their parameters, where it is 1 s or more; "—" is below 1 s (a test written before W-923, such as
PL1's C1 and X4, is not counted). The times were measured under the records' tracer, which inflates
them somewhat (W-923's run folder, `results/tables/new-tests-1s.csv`). W-928, the repair of W-923's review,
measured the RU1 and RU4 rows again with the tests it added, and its repair the RC4 and RU6 rows (W-928's run folder,
the same table). W-929 measured the WI6 row again with its new test, 6.8 s and 0.4 s (W-929's run folder,
`results/tables/new-tests.csv`). No test but EN2, EN3,
WI6 and RU1's two engine tests takes 5 s or more in one parameter, and those are marked `slow`.
`test_workflow_deconvolution_docs_contract.py` checks that every row names tests that exist.

* **Fixtures.** `deconvolution_cases.py` holds `stand_in_dw` (`stand_in`), `input_root` and
  `input_root_stored` (`input_root`, with the additional round `other` that a morphology block
  registers by its shared stain), `psf_small`, `psf_core` and the PSF files. `wisdom_store` is
  `store_for` of `test_workflow_deconvolution_tasks.py`. `workflow_examples` are the worked
  examples of `workflow_variants.py`, whose `fov` task PL1 and SC7 run; the tests of the tasks plan
  `input_root` with a processing block that registers and finds spots.
* **Where they run.** EN1 runs in the routine tier (Snakemake's dry run, under 5 s). EN2 and EN3 are
  `nextflow` and `slow`, in the engine selections. RU1's engine tests (W-928) run two output roots one after the
  other with an absolute `copy_root`: the Snakemake one is `slow` and runs in the slow selection, the Nextflow one is
  `nextflow` and `slow`. `test_ru1_ru2_ru3_verified_reuse` runs the same reuse by the task commands. RU5 runs in the
  routine tier (3.7 s). WI6's command test is `slow` (6.5 s under the tracer, 6.8 s in W-929's run: two concurrent `starfinder deconvolve wisdom` processes and a stand-in
  that waits 1 s). PS3, PS4 and RU11 skip without
  psfmodels, naming the extra; DW1 to DW3 skip without `dw` on PATH or `STARFINDER_TEST_DECONWOLF`, and so do
  WI6's run of `starfinder deconvolve wisdom` on the real binary and DW2's peak run at 6 iterations (W-929), which
  the image's evidence issue runs.
  X4 (PL1) is `extended`; G3 runs in `benchmarks/orchestration_golden_evidence.py`, which the
  coordinating session runs.
* **What differs from the rows above.** PL3's stain label equal to a sequencing label is refused
  before the stage, by the translation's own rule ("reference_stains share the patterns"), which
  names `reference_stains`; the stage's own check refuses the label `ch0` beside `ch00`. RU10's
  damaged copy is refused with exit 2 before any work (the `psf` command's preparation), so that it
  writes and removes nothing. SC6 also checks that the copy follows `execution.image_compression`
  (`none` writes Compression 1, W-923's reading of the specification). RU4 also checks a round record whose
  `channels` list is empty or lacks a channel: the `fov` task fails, naming the round and the file the record does
  not name (W-928). RC4 also checks that a `psf` or `deconvolve` task that reused the copy and then failed (its task
  record's write made to fail) removes no copy file, and that a redo failing at its round record removes exactly
  the label files and logs it wrote; RU6 checks that a task whose round lock is never released fails, naming the
  lock, and removes no copy file (its wait shortened to 0.5 s in the test, as `runtime_min` is at least a minute)
  (W-928's repair). `stand_in_dw` also refuses an input whose maximum is below 1, as deconwolf v0.4.5 does
  (`src/dw.c`: "ERROR: The image has too low intensity, can not continue!", exit 1), writing nothing. WI6 also checks
  that it refuses the volume uniform on [0, 1) that `starfinder deconvolve wisdom` wrote before W-929, and that the
  command's volume, with values from 1 to 255, makes the key's file; on the real binary it checks that the store
  holds the key. SC1's round of zeros turns the refusal off: it checks the stage's rule for a zero round, which
  v0.4.5 itself would refuse (W-929).

(deconvolution-implemented-checks)=

| Check | File | Tests | Time (s, one CPU) |
| --- | --- | --- | --- |
| PL1 | `test_workflow_deconvolution_plan.py`; `test_workflow_plan.py`; `test_workflow_core_golden.py` | `test_pl1_a_plan_without_the_block_has_no_stage`; `test_the_plan_writes_the_expected_manifest_of_each_variant`; `test_the_fov_command_reproduces_the_core_values` | — |
| PL2 | `test_workflow_deconvolution_plan.py` | `test_pl2_the_graph` | — |
| PL2b | `test_workflow_deconvolution_plan.py` | `test_pl2b_the_reference_round_and_its_stains` | — |
| PL3 | `test_workflow_deconvolution_plan.py` | `test_pl3_refusals`, `test_pl3_refusals_of_supplied_files`, `test_pl3_labels_confused_in_one_image_folder` | — |
| PL4 | `test_workflow_deconvolution_plan.py` | `test_pl4_what_is_hashed`, `test_pl4_a_rewritten_psf_file_changes_the_copy_id` | — |
| PL5 | `test_workflow_deconvolution_plan.py` | `test_pl5_selection_growth_keeps_the_first_fovs_tasks` | — |
| PL6 | `test_workflow_deconvolution_plan.py` | `test_pl6_the_copys_place` | — |
| PL7 | `test_workflow_deconvolution_plan.py` | `test_pl7_a_saved_set_needs_no_sequencing_copy` | — |
| PL8 | `test_workflow_deconvolution_plan.py` | `test_pl8_the_fingerprints_inputs` | — |
| EN1 | `test_workflow_deconvolution_plan.py` | `test_en1_snakemake` | 4.4 |
| EN2 | `test_workflow_deconvolution_engines.py` | `test_en2_nextflow` | 42.8 |
| EN3 | `test_workflow_deconvolution_engines.py` | `test_en3_the_engines_agree` | 60.0 |
| SC1 | `test_workflow_deconvolution_tasks.py` | `test_sc1_sc2_offset_and_stretch_factor`, `test_sc1_a_round_of_zeros_stores_zeros` | 1.1 |
| SC1b | `test_workflow_deconvolution_tasks.py` | `test_sc1b_sc2_offset_and_scale_uint16` | — |
| SC2 | `test_workflow_deconvolution_tasks.py` | `test_sc1_sc2_offset_and_stretch_factor`, `test_sc1b_sc2_offset_and_scale_uint16` | 1.5 |
| SC3 | `test_workflow_deconvolution_tasks.py` | `test_sc3_the_offset_rule_is_checked`, `test_sc3_a_tiled_minimum_of_5_does_not_fail` | — |
| SC4 | `test_workflow_deconvolution_tasks.py` | `test_sc4_saturation_and_negatives` | 1.1 |
| SC5 | `test_workflow_deconvolution_tasks.py`; `test_workflow_deconvolution_plan.py` | `test_sc5_a_scale_per_round_and_label`; `test_pl3_refusals` | — |
| SC6 | `test_workflow_deconvolution_tasks.py` | `test_sc6_the_file_and_its_metadata`, `test_sc6_the_copy_follows_image_compression` | 2.7 |
| SC7 | `test_workflow_deconvolution_tasks.py` | `test_sc7_the_copy_loads_as_the_raw_round_loads` | 2.0 |
| SC8 | `test_workflow_deconvolution_plan.py` | `test_sc8_the_warning_for_intensities_across_fovs`, `test_sc8_no_warning_without_a_feed` | — |
| PS1 | `test_workflow_deconvolution_tasks.py` | `test_ps1_centring` | — |
| PS2 | `test_workflow_deconvolution_tasks.py` | `test_ps2_a_supplied_file` | — |
| PS3 | `test_workflow_deconvolution_tasks.py` | `test_ps3_generation` | skips here (P) |
| PS4 | `test_workflow_deconvolution_tasks.py` | `test_ps4_the_spacing_reaches_the_psf` | skips here (P) |
| WI1 | `test_workflow_deconvolution_tasks.py` | `test_wi1_the_key` | — |
| WI2 | `test_workflow_deconvolution_tasks.py` | `test_wi2_a_matching_key` | — |
| WI3 | `test_workflow_deconvolution_tasks.py` | `test_wi3_no_match_or_no_store` | — |
| WI4 | `test_workflow_deconvolution_tasks.py` | `test_wi4_a_damaged_store` | — |
| WI5 | `test_workflow_deconvolution_tasks.py` | `test_wi5_nothing_written_outside_the_task` | — |
| WI6 | `test_workflow_deconvolution_tasks.py` | `test_wi6_one_generating_job_per_key`, `test_wi6_the_planning_volume_is_bright_enough`, `test_wi6_the_command_on_the_real_binary` | 7.2 |
| WI7 | `test_workflow_deconvolution_tasks.py` | `test_wi7_gpu_tasks` | — |
| RC1 | `test_workflow_deconvolution_tasks.py` | `test_rc1_rc2_the_round_record_and_copy_json` | — |
| RC2 | `test_workflow_deconvolution_tasks.py` | `test_rc1_rc2_the_round_record_and_copy_json` | — |
| RC3 | `test_workflow_deconvolution_tasks.py` | `test_rc3_run_json_names_the_copy` | 1.1 |
| RC4 | `test_workflow_deconvolution_tasks.py` | `test_rc4_completion_and_failure`, `test_rc4_a_failure_after_reuse_removes_no_copy_file`, `test_rc4_a_failed_redo_removes_what_it_wrote` | 2.6 |
| RC5 | `test_workflow_deconvolution_tasks.py` | `test_rc5_a_shared_image_folder_keeps_two_records`, `test_rc5_a_rewritten_raw_file_redoes_its_round_only` | 5.9 |
| RU1 | `test_workflow_deconvolution_tasks.py`; `test_workflow_deconvolution_engines.py` | `test_ru1_ru2_ru3_verified_reuse`; `test_ru1_snakemake_reuses_a_shared_copy`, `test_ru1_nextflow_reuses_a_shared_copy` | 67.4 |
| RU2 | `test_workflow_deconvolution_tasks.py` | `test_ru1_ru2_ru3_verified_reuse` | 1.8 |
| RU3 | `test_workflow_deconvolution_tasks.py` | `test_ru1_ru2_ru3_verified_reuse` | 1.8 |
| RU4 | `test_workflow_deconvolution_tasks.py` | `test_ru4_a_consumer_checks`, `test_ru4_a_record_that_names_too_few_files` | 2.1 |
| RU5 | `test_workflow_deconvolution_tasks.py` | `test_ru5_resume_and_saved_sets` | 3.7 |
| RU6 | `test_workflow_deconvolution_tasks.py` | `test_ru6_one_writer_per_round`, `test_ru6_a_lock_that_is_not_released_removes_nothing` | 4.6 |
| RU7 | `test_workflow_deconvolution_tasks.py` | `test_ru7_code_version_rerun` | 1.4 |
| RU8 | `test_workflow_deconvolution_tasks.py` | `test_ru8_a_change_of_spacing_alone` | 3.0 |
| RU9 | `test_workflow_deconvolution_tasks.py` | `test_ru9_a_changed_supplied_psf` | 1.4 |
| RU10 | `test_workflow_deconvolution_tasks.py` | `test_ru10_a_damaged_copys_psf` | 1.5 |
| RU11 | `test_workflow_deconvolution_tasks.py` | `test_ru11_the_spacing_reaches_a_generated_psf` | skips here (P) |
| DW1 | `test_workflow_deconvolution_tasks.py` | `test_dw1_the_options_exist` | skips here (D) |
| DW2 | `test_workflow_deconvolution_tasks.py` | `test_dw2_the_offset_rule_and_determinism`, `test_dw2_the_peak_rises_at_6_iterations` | skips here (D) |
| DW3 | `test_workflow_deconvolution_tasks.py` | `test_dw3_the_stage_on_the_real_binary` | skips here (D) |
| MT1 | `test_workflow_deconvolution_plan.py` | `test_mt1_matlab_is_untouched` | — |
