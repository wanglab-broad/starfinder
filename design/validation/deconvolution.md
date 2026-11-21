# Validation design: the deconvolution stage

% Status: Draft for Jiahao's review, written against `70edc72` (v0.3.0). W-918, 2026-11-21.

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
  stored uint16 copy made by the stand-in is an integer array and is compared exactly.
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
| `input_root` | A synthetic input root of 2 FOVs × 2 rounds × 2 labels, 8 × 32 × 32 uint8, sparse Gaussian spots, in the default `{round}/{fov}` layout, with `fovs.csv`, a codebook and one reference stain label, `dapi`, in the reference round's folder beside its sequencing labels `ch00` and `ch01`. Raw names hold the FOV id, as real names do (`Position001_ch00.tif`). `voxel_size_xy` 0.1 and `voxel_size_z` 0.3. | hand-built |
| `workflow_examples` | The worked-example configurations of `test/workflow_examples.py` (X), with a `deconvolution` block added per row. | existing |
| `wisdom_store` | A store folder with a manifest and one wisdom file for the key the stand-in's task computes, and one for another key. | hand-built |
| `psf_small` | A 7 × 9 × 9 float32 Gaussian, sum 1, peak at the centre, and the same rolled by +2 planes and +1 row. | hand-built |
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
| PL3 | Refusals | the block varied | Each case of the specification's "Plan refusals" exits 2 with `PlanRefused` naming the key: `backend: matlab`; `iterations` absent, 0 or 2.5; an unknown round in `rounds`; a round no consumer reads; a missing `voxel_size_xy` or `voxel_size_z`, with `generate` and with `file`; a label's wavelengths missing with `generate`; a missing or absent file, or no `psf.voxel_size_um`, with `file`; a declared spacing that differs from the dataset's by 2 parts in 10⁶ in Y and X, and again in Z (one part in 10⁷ is accepted); a stain label equal to a sequencing label of the reference round, and a stain label `ch0` beside the sequencing label `ch00`, with both rounds deconvolved; a `tile_size` below the PSF's lateral size plus 40; a scale of 0; a scale mapping missing a (round, label); a relative `copy_root` other than `output` and `beside`. | exact (exit status, exception type, key named) | H |
| PL4 | What is hashed | `input_root`, planned twice with one change each | Changing `iterations`, `tile_size`, `scale`, `border_quality`, `rounds` or a PSF parameter changes the copy id, the `deconvolve` and `psf` fingerprints and every consumer's fingerprint. So does changing `voxel_size_xy` alone, and `voxel_size_z` alone. With `psf.source: file`, rewriting a PSF file with other values at the same path changes the copy id. Changing `execution.tasks.deconvolve.threads` or `gpu`, `execution.deconwolf` or `execution.deconwolf_wisdom` changes no `blocks_sha256`, fingerprint or copy id. | exact (hashes) | H |
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

| # | Check | Fixture | Expected | Tolerance | Needs |
| --- | --- | --- | --- | --- | --- |
| SC1 | Offset and scale, untiled | `stand_in_dw`, untiled, scale 64 | The stand-in returns x + 5, so each copy file equals `clip(rint(x × 64), 0, 65535)` of its raw channel as uint16. The record says kept offset 5. | exact (integer arrays) | H |
| SC2 | Offset and scale, tiled | as SC1 with `tile_size` | Kept offset 0; the same expected array. | exact | H |
| SC3 | The offset rule is checked | `stand_in_dw` writing an untiled output with minimum 0; `stand_in_dw` whose `--version` names 0.4.6 | Each task fails, naming the rule or the version; no copy file or record is left; the failed record lists what was removed. A tiled output with minimum 5 does not fail, and its record holds the minimum. | exact (exit status, strings) | H |
| SC4 | Saturation | raw values that exceed 65,535 / s with a large scale | Those voxels store 65,535, the record counts them exactly, and the log warns once. Negative values after the offset store 0 and are counted. | exact (counts) | H |
| SC5 | Scale per round and label | a mapping scale | Each file uses its (round, label) scale; the record holds it. | exact | H |
| SC6 | The file | the copy of SC1 | ZYX uint16, TIFF Compression 8 and no Predictor tag (W-909's codec), and the raw file's `ImageMetadata`. Only `<label>.tif` files lie in the image folder. | exact (tags, strings) | H |
| SC7 | Consumers read it | the copy of SC1; `load_round` on `<copy>/<round>/<fov>` | The labels in configured order, uint16, the arrays of SC1. A `fov` task of `workflow_examples` with the block runs to `succeeded` on the copy. | exact | H, WF |

### The PSF (PS)

| # | Check | Fixture | Expected | Tolerance | Needs |
| --- | --- | --- | --- | --- | --- |
| PS1 | Centring | `psf_small`, rolled | The peak moves to the centre index; the wrapped planes and rows are 0; the sum is 1. The record's shift is (−2, −1, 0), and the record also holds the sum dropped. The centred copy of the unrolled fixture equals it. | 1e-9 absolute (float sums); exact (indices) | H |
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
| DW1 | The options exist | `real_dw` | `dw --version` names 0.4.5. `dw --help` lists every option the stage passes: `--iter`, `--threads`, `--gpu`, `--tilesize`, `--noplan`, `--offset`, `--out` and `--float`. If `--float` is missing, the specification's fallback (deconwolf's uint16 and its logged scale) is the implemented path, and RD1's bound gains one step of that scale. | exact (strings) | D |
| DW2 | The offset rule and determinism | `real_dw`, 16 × 64 × 64 point sources blurred by `psf_small`, CPU, `--noplan`, 3 iterations | The untiled float output's minimum is within 0.5 of 5; a tiled run (`--tilesize 32`) has minimum below 0.5. Two untiled runs at 1 and 2 threads are byte-identical (W-916: thread counts change no byte). The brightest point's peak rises above its input value. | exact (hashes); the 0.5 bands of W-916 | D |
| DW3 | The stage on the real binary | `input_root` with `real_dw`, CPU | The tasks succeed; SC6 and RC1 hold; the record's planning is `estimate`. | exact | D, S if over 5 s |

### MATLAB (MT)

| # | Check | Expected | Needs |
| --- | --- | --- | --- |
| MT1 | MATLAB is untouched | With `backend: matlab`, a configuration with the block is refused (PL3). The MATLAB rule set and its schema are byte-identical to `70edc72`. No MATLAB run is needed: no file MATLAB reads changes. | H |

### What the default tier covers

PL1–PL8 with PL2b, EN1, SC1–SC7, PS1, PS2, WI1–WI7, RC1–RC5, RU1–RU4, RU6–RU10 and MT1 run in the
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
`v_s` is the copy's value, `stored / s`, with s = 64.

| # | Check | Data | Expected and bound |
| --- | --- | --- | --- |
| RD0 | The generated PSF | The `psf` task with W-855's parameters for the four channels at depth 0 | Each decoded array equals W-855's centred PSF of that channel (`results/psf/centred/psf_ch0<c>_pz0.0.tif`). With psfmodels 0.3.3, W-855's version, equality is expected; the bound is 1 × 10⁻⁷ of the PSF's peak, for float32 rounding after a float64 normalisation. The centring shift is 0 on every channel, as W-855 found at depth 0. |
| RD1 | The stage reproduces W-915 | Position022, four channels; `psf.source: file` with W-855's centred PSFs (so that RD0 and RD1 test one thing each); `iterations: 5`; untiled; GPU on the host's RTX A5000, the card of W-915; the image's deconwolf | Per channel: max \|v_s − v_w\| ≤ 1 / scaling_w + 0.5 / s + 2 × 10⁻⁵ × max(v_w). The first term is W-915's quantization step, a full step because deconwolf's rounding rule was not recorded. The second is the copy's half step. The third covers a build or device difference at W-916's agreement between modes. For ch00 (scaling_w 101.886, max 638.2) the bound is 0.0098 + 0.0078 + 0.0128 = 0.0304 grey levels. Also per channel: NCC(v_s, v_w) ≥ 1 − 10⁻⁶ (W-916's untiled arms: 1 − 10⁻¹⁰). Spot F1 against W-915's spots: ≥ 0.999 at adaptive 0.2 and at noise 5.0. W-916's CPU arms, a perturbation of the same size, gave 1.0000 and 0.999999. The records name the copy, the PSF hashes, `iterations_done` 5 and the kept offset 5. Reported: the fraction of voxels within the first two terms alone, which is 1 if the float output reproduces W-915's exactly. |
| RD2 | The CPU mode on RedHat 8 | Position022 ch00, CPU, 4 threads, `--noplan`, one UGER RedHat 8 node, the SIF | max \|Δv\| against RD1's ch00 ≤ 2 × 10⁻⁵ × max(v) + 1 / s, which is W-916's CPU–GPU agreement plus one step of each copy's rounding. Reported: wall time, CPU time, VmPeak and RSS. The task's `h_vmem` of 3,500 MB per slot × 4 slots holds, with no kill. |
| RD3 | The wisdom store | `starfinder deconvolve wisdom` once on RD2's node class (6 threads, ch00's shape and PSF shape), then RD2 again at 6 threads with the store | The second run's record says `measure` and names the key. Its output agrees with RD2 within RD2's bound. The store's files have the same SHA-256 before and after. Reported: the planning time against W-916's 1,617 s and the warm time against W-916's 74 s. |
| RD4 | Reuse across runs | RD1's plan again, into a second output root, with `copy_root` the same absolute path | Every `deconvolve` task says `reused: true` and ran no deconwolf. The copy's files keep their SHA-256 and modification times. |
| RD5 | The GPU on UGER | Position022 ch00, GPU, one UGER A30 node, the SIF with `--nv` | Agreement with RD1's ch00 within RD2's bound (another card). Reported: the wall time and the GPU memory peak against W-916's 35 s and 14.0 GiB. |
| RD6 | Sizes | RD1's copy | Per channel: the copy's bytes, and the ratio to the raw channel's 283,115,520 bytes. Reported, not bounded. W-915's zlib uint16 files, at a per-volume scale, were 310–425 MB per channel. |

**Resources.** RD1 is about 2.5 min of GPU (W-916: 35 s per channel). RD2 is about 3 min at
4 threads (166 s at 4 threads in W-916), at 10 GiB. RD3's planning is about 27 min at 6
threads, about 2 CPU-hours (W-916: 7,030 CPU-s), and its warm run about 1.3 min. RD5 is about
1 min of GPU. The staging and the comparisons read about 1 GB of raw image and 1.6 GB of W-915's
outputs. In total that is about 3 CPU-hours, three UGER jobs (one GPU) and the host's GPU. The
evidence issue states its grant from these figures.

## Implemented checks

The implementation issue adds the table that maps each row above to its test, with the measured
time of each test that takes 1 s or more.
