# Deconvolution stage: a deconvolved TIFF + zlib copy of the images, made by deconwolf before the FOV tasks

% Status: Draft for Jiahao's review, written against `70edc72` (v0.3.0). W-918, 2026-11-21. The branch head `2a301f3` changes design records only, so every code reference on this page holds at `70edc72`.

This page specifies deconvolution as a workflow stage of its own. Jiahao decided on 2026-11-20
(W-855, comment 70988672) to make deconwolf the primary deconvolution and to arrange a rule for
reusing FFTW wisdom ("把 deconwolf 当做 primary deconvolution 选择，并且安排wisdom 复用规则"), and
then asked for this specification ("File the spec issue"), in answer to the recommendation that
deconvolution be a stage and not a step inside the FOV tasks.

* **What the stage does.** A new `deconvolution` block makes the plan add one `deconvolve` task per
  (FOV, round) and one `psf` task per run. Each `deconvolve` task runs deconwolf on every channel
  of its round and writes a deconvolved copy of the round's folder. The copy is TIFF + zlib, uint16 with one fixed scale,
  in the input's folder layout. The FOV tasks (`summarize`, `fov`, `tile_global`, `morphology`) then
  read the copy in place of the raw images.
* **What does not change.** A run without a `deconvolution` block plans exactly the manifest it
  plans today: no hash, fingerprint, task or record changes. Nothing MATLAB reads or writes changes.
  The raw images are never written.
* **Keys.** The `deconvolution` block holds what changes the result: the rounds, iterations,
  method, border, tiling, PSF and scale. `execution` holds what does not: the device, the threads,
  the binary and the wisdom store. That line follows W-916: deconwolf's CPU and GPU outputs agree
  within 1.6 × 10⁻⁵ of the maximum, and so do its two FFT planning modes, while tiling changes
  values by up to 1.3 % of the maximum.
* **Identity and reuse.** The copy's folder is named by a hash of its settings, the voxel spacing
  and any supplied PSF's content. Each (FOV, deconvolved round) has a record of its own, keyed by
  the round name, with the input and output SHA-256 (file and decoded array), the PSF's SHA-256, the
  deconwolf version and binary hash, the options and the wisdom key used. A later run with the
  same settings reuses the copy after checking those hashes. Each FOV task records which copy it
  read.

The validation design is [deconvolution validation](../validation/deconvolution.md). The pages this
one builds on are [orchestration](orchestration.md) (task kinds, the plan and manifest, the blocks
of a task kind, completion, resume), [orchestration-engines](orchestration-engines.md) (the
fingerprint, GPU tasks, the image), [checkpoints](checkpoints.md) (`run.json`) and
[compressed-image-outputs](compressed-image-outputs.md) (the TIFF + zlib codec, W-909).

This is a software design. It makes no data claim. The times, sizes and agreements it quotes are
the measurements of W-855, W-915 and W-916 on 9735seqA1 (Leica SP8, three positions), named where
they are used. Similarity to Huygens is not accuracy (W-855).

## Questions for Jiahao

1. **Accept the recommendations** of the twelve decisions in the table below?
2. **psfmodels as a dependency (decision 4).** Generating the PSF from configuration needs
   psfmodels 0.3.3 (W-855's version) in the lock, as a new optional extra `deconvolution`. W-855
   installed it by building its source distribution, so the image's build stage may need a C++
   compiler. The alternative is to accept only a supplied PSF file and keep the lock unchanged.
   Add the extra, or start with supplied files only?
3. **The scale (decision 3).** The recommendation stores the deconvolved values as uint16 with
   one fixed scale, 64 by default, so that a stored value is 64 × the deconvolved grey level.
   On W-915's twelve channel volumes, the largest deconvolved value is 521–638 raw grey levels,
   inside the range of 0–1,023.98 that scale 64 can hold. The alternative closest to Huygens' exports is the raw
   dtype (uint8) with one recorded scale, as Huygens writes them (W-855 records its stretch factors 0.284–0.296).
   Is uint16 at 64 right?
4. **Iterations (decision 5).** The recommendation makes `iterations` a required key with no
   default: 5 was frozen on one SP8 dataset (W-915), at the low edge of its grid. Each dataset
   chooses its count by W-855's development-position rule against a comparator, recorded in
   "Preparing a dataset". Should 5 be the default instead?
5. **The device and the planning mode outside the hash (decisions 6 and 7).** Switching between
   CPU and GPU, or between planned and `--noplan` FFTs, reruns nothing, although each changes
   values by up to 1.6 × 10⁻⁵ of the maximum (W-916). Tiling, which changes values by up to 1.3 %,
   is in the hash. Is that the right line?
6. **deconwolf in every image target (decision 6).** The recommendation builds deconwolf v0.4.5
   from source in a build stage of its own and copies it into the runtime stage that `cpu`,
   `learned` and `learned-cuda` all start from. No new target is needed, because a run has one
   image for all its tasks (`execution.container`). Accept?
7. **The real-data check.** Running the stage on Position022 is more than a batch may run on raw
   images (W-889 D3). The recommendation is a separate evidence issue on the implementation
   branch's commit, run by the coordinating session before the merge: on the host's RTX A5000,
   the card of W-915, and once on a UGER RedHat 8 node in the image. Accept?
8. **The plan reads supplied PSF files (decision 9).** With `psf.source: file`, the copy id holds
   each file's SHA-256, so the plan reads those files. That is an exception to the rule that the
   plan reads only the sizes and times of its inputs. The files are small: W-855's PSFs are
   21 × 33 × 33 float32, about 92 KB each. The recommendation is to read them, so that a changed
   PSF gives a new copy and never rewrites an existing one. The alternative leaves the content
   out of the id, finds a changed file at task time, and rewrites that copy's PSF and rounds in
   place, under records that other runs name. Accept the exception?
9. **A supplied PSF's declared spacing (decision 4).** The recommendation refuses at plan time a
   `psf.voxel_size_um` that differs from `voxel_size_xy` or `voxel_size_z` by more than 1 part in
   10⁶. The alternative is to warn and run. A PSF sampled on another grid than the images
   deconvolves them wrongly, so refuse?

## Decisions

The options and the evidence of each choice are in its section.

(deconvolution-decisions)=

| # | Choice | Recommendation | Alternatives | Evidence | Section |
| --- | --- | --- | --- | --- | --- |
| 1 | The stage and its task kinds | TK1: a `deconvolve` task per (FOV, round), and one `psf` task per run, before every task that reads a deconvolved round; a `deconvolution` block enables them | TK2: per (FOV, round, channel); TK3: per FOV; TK4: a step inside `fov` | The unit of `load_round` is a round folder (`io/tiff.py:394`); the plan's image discovery is per (FOV, round, label) (`workflow/_plan.py:325`–`:341`); W-916: about 35 s per channel volume on the GPU, 74–575 s on the CPU | [The stage](#the-stage-and-its-task-kinds) |
| 2 | The output and where it lives | OU1: a copy of each deconvolved round folder, TIFF + zlib, one `<label>.tif` per channel in the input's `image_dir_template` layout, under `<copy base>/dw-<copy id>/`; `copy_root: output` (default, under the output root), `beside` (`<dataset_id>-dw/` beside the dataset's folder, as W-910's `-zlib/` copies) or an absolute path | OU2: under the output root only; OU3: beside the data only; OU4: the raw file names | W-909 and W-881 (TIFF + zlib, no predictor); W-910's sibling `-zlib/` copies; the plan's group templates (`workflow/_plan.py:552`–`:561`) | [The output](#the-output) |
| 3 | The scaling | SC1: deconwolf's float32 output, its kept offset removed (5 untiled, 0 tiled, v0.4.5), clipped at 0, stored as uint16 = rint(value × scale) with one fixed scale (default 64, a number or per round and label), saturation counted | SC2: deconwolf's own uint16 with a per-volume scale; SC3: the raw dtype with a recorded scale; SC4: float32 | W-915: per-volume scalings 101.9–124.5 differ by channel and position; maxima 521–638 after the offset; `src/dw.c` 2661–2665 and 2012–2024 (W-915, W-916) | [The scaling](#the-scaling) |
| 4 | The PSF | PS1: generated by the `psf` task with psfmodels `confocal_psf(model="vectorial")` from the block's optics and the dataset's voxel size, centred on its peak; or PS2, a supplied file per label with its declared spacing, refused when that differs from the dataset's; both recorded with their SHA-256 and spacing, and the spacing in the copy id | PS2 only (no dependency); a PSF per task | W-855 amendments 1 and 2; its PSF records | [The PSF](#the-psf) |
| 5 | deconwolf's settings | DS1: SHB (`--gpu` runs it as SHBCL2), `iterations` required, border quality 2, background automatic, offset 5, untiled by default with `tile_size` optional | DS2: a default of 5 iterations; DS3: deconwolf's automatic stopping | W-915's frozen choice (5 on 022, at the grid's edge; auto 0.9675 outside the tolerance); W-916's tiling | [deconwolf's settings](#deconwolfs-settings) |
| 6 | Execution | EX1: CPU with threads by default, the GPU by `execution.tasks.deconvolve.gpu`; resources per task from W-916; deconwolf v0.4.5 built from source in the image's shared runtime stage | EX2: GPU by default; EX3: a new image target; EX4: a host binary by path | W-916: the mode table, the projection, the RedHat 8 failure; `Containerfile:111`–`:133`; one image per run (`workflow/_plan.py:522`) | [Execution](#execution) |
| 7 | FFTW wisdom | WS1: a store named by `execution.deconwolf_wisdom`, one file per key (job shape, PSF shape, border, threads, CPU model, FFTW and deconwolf versions) with a manifest; one generating command per new key, outside the plan; tasks copy a matching file into a private `$HOME/.config/deconwolf/` and never write the store; `--noplan` when none matches | WS2: a `wisdom` task in the plan; WS3: no wisdom, always `--noplan`; WS4: wisdom left to deconwolf's defaults | W-855's reuse rule; W-916: cold 1,692 s, warm 74 s, `--noplan` 153 s at 6 threads; where deconwolf reads its wisdom | [FFTW wisdom](#fftw-wisdom) |
| 8 | Records | RC1: a record per (FOV, deconvolved round) beside the copy, keyed by the round name (never by the image folder, which `reference_stain` shares with the reference round), and the copy's own `copy.json`; `run.json` of a FOV task gains a `deconvolution` key | RC2: everything in the task record; RC3: the deconwolf log only | W-909's identity rule; `run.json`'s `spot_finding_input` precedent | [Records](#records) |
| 9 | Reuse and staleness | RU1: within a run, the engines' fingerprints; across runs, the copy id names the settings, the voxel spacing and any supplied PSF's content, and a task verifies its own round's record before it reuses that round; a FOV task checks and records the copy it read | RU2: one copy per root, refused on a mismatch; RU3: one copy, overwritten | The fingerprint (orchestration-engines); verified resume (`workflow/_resume.py:45`) | [Reuse and staleness](#reuse-and-staleness) |
| 10 | What the stage does not do | NS1: no Huygens, no measured PSF, no decoding-based tuning, no preprocessing; a dataset without the block does not run the stage | — | The issue's scope | [What the stage does not do](#what-the-stage-does-not-do) |
| 11 | MATLAB | MT1: MATLAB's files and keys unchanged; `backend: matlab` refuses the block; the copy is not offered to MATLAB | MT2: write a MATLAB-readable copy | AGENTS.md; W-910's MATLAB read of deflate TIFFs | [MATLAB](#matlab) |
| 12 | Docs | DG1: the implementation writes the guide, "Preparing a dataset", the configuration reference and the UGER page; no migration row | — | `docs/contributing.md`, "Documentation pages" | [Docs](#docs) |

## Why

**The evidence.** W-855, W-915 and W-916 (all done) compared, on 9735seqA1's Positions 022, 028
and 037 (30 × 3,072 × 3,072 uint8, four channels, one round):

* deconwolf v0.4.5 (SHB, 5 iterations, W-855's centred confocal PSF at depth 0) is slightly
  closer to Huygens than plain Richardson–Lucy with the same PSF. Its NCC to Huygens is
  0.977–0.984 (022), 0.980–0.986 (028) and 0.967–0.973 (037), against RL's 0.976–0.982,
  0.979–0.985 and 0.962–0.970. Its spots match 97 % of Huygens' with F1 0.78–0.83 (RL 0.76–0.80).
* Its CPU output equals its GPU output within 1.6 × 10⁻⁵ of the maximum (NCC 0.9999999999), with
  the same spots. Thread counts change no byte.
* Tiled at 2,048 or 1,024 pixels, it runs in 4.0 or 2.5 GiB of GPU memory, against 14.0 GiB
  untiled. Tiled outputs differ from the untiled one near the seams, by up to 1.1–1.3 % of the
  maximum (spot F1 0.9998 adaptive, 0.9975 noise).
* On the CPU, one channel volume takes 74 s at 6 threads with reused FFTW wisdom, 153 s with
  `--noplan`, and 1,692 s when it plans cold. The GPU takes 35 s untiled.

**How it fits.**

* The lab already deconvolves as a separate step: Huygens writes deconvolved images, and the
  pipeline reads them. The catalog datasets, tissue-2D and LN, are Huygens' uint8 exports
  (`design/notes/image-statistics.md`, "Limitations"). A stage that writes a deconvolved copy keeps
  that shape, and puts its records, configuration and provenance in starfinder.
* It follows this week's decisions: TIFF + zlib is the recommended storage (W-881, W-909), and
  intermediate sets are kept and reused (W-889, W-910).

**Why not inside the FOV tasks.** A `fov` task is single-core registration and spot finding, about
30 minutes on a tissue-2D FOV (W-903). Deconvolution wants a GPU or many cores and 10 GiB per
process (W-916). Inside `fov`, every rerun of a FOV would deconvolve again, a resume from a
checkpoint could not skip it, and the GPU would sit idle for most of the task.

% Evidence: runs/W-855/20261120T1907Z-e06-pilot (README.md); runs/W-915/20261121T0003Z-deconwolf (README.md, results/dw/dw_<p>.json, results/choice/frozen_choice.json); runs/W-916/20261121T0219Z-dw-modes (README.md, results/tables/projection.json, results/build/dw_build.json); runs/W-878/20261121T0157Z-image-v0.3.0 (README.md)

## Terms

* A **deconvolved round** is a (round, image) pair that `deconvolution.rounds` names: a
  sequencing round's channels, an additional round's channels, or `reference_stain`, the
  reference stains in the reference round's folder. Its **round name** is the sequencing or
  additional round's name, or `reference_stain`. The round name, not the image folder, identifies
  a deconvolved round: `reference_stain` and the reference round share one image folder
  (`workflow/_plan.py:333`–`:334`).
* The **copy** is the folder of one dataset sample's deconvolved rounds made with one set of
  settings: `<copy base>/dw-<copy id>/`.
* The **copy id** is the first 12 hexadecimal digits of the SHA-256 of the canonical JSON
  (`workflow/_shape.py:124`) of `{dataset_id, sample_id, image_dir_template, voxel_size_xy,
  voxel_size_z, deconvolution, psf_files}`:
  * `deconvolution` is the resolved block less `copy_root`, which holds every optical parameter
    of a generated PSF ([The PSF](#the-psf), "The PSF's inputs");
  * `voxel_size_xy` and `voxel_size_z` are the two top-level keys (`DATASET_KEYS`,
    `workflow/_shape.py:28`–`:31`). They are the only inputs of a PSF that are read from outside
    the block, and they are in the id whatever `psf.source` is;
  * `psf_files` is `{<label>: <SHA-256 of the file>}` with `psf.source: file`, and null otherwise.

  So a change of the spacing, of any optical parameter, or of a supplied PSF file's content gives
  another copy id, and so another folder and a new PSF.
* A **consumer** is a planned task that reads a deconvolved round from the input root today:
  `summarize`, `fov`, `tile_global` and `morphology`.
* The **kept offset** is the part of deconwolf's `--offset` (5 by default) still in its saved
  output: 5 untiled and 0 tiled in v0.4.5 (`src/dw.c` 2651–2665 and 2012–2024; W-915, W-916).
* **Planning** is how deconwolf's FFTW plans are made on the CPU: `measure` with wisdom from the
  store, or `estimate` (`--noplan`, FFTW_ESTIMATE).

## The stage and its task kinds

| Option | Tasks for tissue-2D (56 FOVs × 4 rounds × 4 channels) | Consequence |
| --- | --- | --- |
| **TK1. `deconvolve` per (FOV, round), `psf` per run (recommended)** | 224 + 1 | A task makes one round folder, the unit `load_round` reads and a consumer needs. Its four channels run one after another in one process tree: on the GPU about 2.5 min of deconwolf, on the CPU 5–38 min by threads and planning (W-916). One record per folder. |
| TK2. Per (FOV, round, channel) | 896 + 1 | Finer packing on GPUs. Four times the jobs, a `label` identity field, and a round folder complete only when four tasks are. |
| TK3. Per FOV | 56 + 1 | 16 channel volumes per task, 20 min to 2.6 h on the CPU; less parallel; a rerun of one round reruns all four. |
| TK4. A step inside `fov` | 0 | Mixes GPU or many-core work with single-core work; every `fov` rerun and every resume deconvolves again ([Why](#why)). |

**Recommendation: TK1.**

**The task kinds.** Two new kinds join `TASK_BLOCKS` (`workflow/_shape.py:98`):

| Kind | Identity | Blocks it reads | Command | Declared inputs | Declared outputs |
| --- | --- | --- | --- | --- | --- |
| `psf` | the run (no fields, like `compare`) | `deconvolution` | `starfinder workflow psf` | a supplied PSF file per label, when `psf.source` is `file` | `<copy>/psf/<label>.tif` for every label of a deconvolved round, `<copy>/psf/psf.json`, `<copy>/copy.json` |
| `deconvolve` | `sample`, `fov`, `round` (the round name) | `deconvolution` | `starfinder workflow deconvolve` | the round's raw channel files (`input:` paths, as `Inputs` finds them, `workflow/_plan.py:338`–`:341`); the PSF files of its labels | `<copy>/<image dir>/<label>.tif` per label; `<copy>/records/<round name>/<fov>/deconvolution.json`; `<copy>/records/<round name>/<fov>/<label>.dw.log` per label |

`<image dir>` is `image_dir_template` filled with the round and FOV (`_layout.image_dir`,
`_layout.py:309`). For `reference_stain` it is the reference round's folder, and the copy holds the
stain labels there. A round's sequencing labels and its stain labels are separate deconvolved
rounds; both may share an image folder, with different label files.

* **Records are keyed by the round name, never by the image folder.** When `rounds` names both the
  reference round and `reference_stain`, two `deconvolve` tasks write into one image folder
  (`workflow/_plan.py:333`–`:334`). A record path built from the image folder would be written by
  both, and each would overwrite the other's provenance, completion and reuse. So each task's
  record, logs and lock lie in `<copy>/records/<round name>/<fov>/`, a folder that only that task
  writes. The path follows the task's identity (round name, FOV) rather than `image_dir_template`
  for the same reason: two layouts of one identity cannot collide, and the record of a task is
  found from its id alone. The logs are named by label inside that folder. A round's labels are
  unique, so no two logs collide.
* **Label files in a shared image folder.** The images of the two rounds do share the folder,
  with one `<label>.tif` per label. The plan refuses a configuration in which two labels written
  into one image folder of the copy could be confused: two equal labels, or a label whose pattern
  `*<label>*.tif` (`io/tiff.py:425`) matches another label's file name, as `ch0` would match
  `ch00.tif`. Equal labels would also select the same raw file in the reference round's folder for
  both rounds (`_channel_file`, `workflow/_plan.py:276`–`:286`).
* **One group per deconvolved round.** Each round name has its own group, `deconvolve.<round
  name>`, as each supplied step has `summarize.<step>` (`workflow/_plan.py:603`). The rounds' label
  sets differ, and the stains' image folder holds the reference round's name as a literal. The
  plan refuses a group whose tasks declare different output templates (`workflow/_plan.py:558`–`:561`),
  so one group for all rounds would be refused.

* **The identity field `round`.** `ID_FIELDS` (`workflow/_shape.py:119`) gains `round` after
  `fov`, and the record and log templates write it as a wildcard, as `fov` is
  (`workflow/_plan.py:537`). No existing task has the field, so no existing id changes. The value is the round
  name, or `reference_stain`.
* **The hash.** `blocks_sha256` of both kinds covers the common part and the `deconvolution`
  block. Both leave `selected_fovs` and `output_id` out, as the FOV-level kinds do (W-877,
  `workflow/_shape.py:110`, `:114`): `deconvolve` joins `FOV_KINDS`, and `psf` joins the same rule
  although it is run-level, so that growing the selection keeps the PSF and every finished
  `deconvolve` task.
* **Execution.** `KIND_THREADS` gains `deconvolve: 4`. With `execution.tasks.deconvolve.gpu: true`
  the defaults become 1 thread, as W-915 and W-916 ran the GPU. The resources are in
  [Execution](#execution).

**The graph.** The plan adds the stage only when the block is present:

1. One `psf` task, with no needs.
2. One `deconvolve` task for each (FOV, deconvolved round) that some consumer reads, needing `psf`.
   A FOV whose `fov` task reads a saved spot-finding set (`processing.spot_finding_input`) reads no
   sequencing image, so it gets no `deconvolve` task for its sequencing rounds unless `summarize`
   or `tile_global` read them.
3. Each consumer's declared inputs name the copy's files of its deconvolved rounds in place of
   the raw files, and its `needs` gain the `deconvolve` tasks that make them. Its task entry gains
   `deconvolved`: `{<round name>: <copy folder of the round, relative to the output root or
   absolute>}`, which its command reads to load those rounds from the copy. A `morphology` task
   that reads the stains has a `reference_stain` entry, whose folder is the reference round's. Its blocks do not change, so
   its `blocks_sha256` does not; its fingerprint changes through `entry` and `needs`
   (orchestration-engines, "The task fingerprint").
4. The fingerprint's `inputs` component stats only the run's own inputs
   (`workflow/_plan.py:1041`). A copy file outside the output root is an absolute path and a
   declared output of a `deconvolve` task, which does not exist when the plan runs. The
   implementation therefore leaves the declared outputs of other tasks out of that component;
   they reach the fingerprint through `needs`.

**Plan refusals** (exit 2, `PlanRefused`, naming the key):

* `backend: matlab` with a `deconvolution` block;
* `iterations` absent, or not a positive integer;
* a `rounds` entry that names no round of the configuration, or that no consumer reads;
* a block without `voxel_size_xy` and `voxel_size_z`, whatever `psf.source` is: the copy id and
  the PSF check need them;
* `psf.source: generate` without the excitation and emission of a label that a deconvolved round
  reads;
* `psf.source: file` without a file for such a label, a file that does not exist, or without
  `psf.voxel_size_um`, the files' declared spacing;
* a declared spacing that differs from `voxel_size_xy` or `voxel_size_z` by more than 1 part in
  10⁶ of the dataset's value. A PSF sampled on another grid than the images deconvolves them
  wrongly, so the plan refuses rather than warns;
* two labels that could be confused in one image folder of the copy (above);
* `tile_size` smaller than the PSF's lateral size plus deconwolf's padding of 20 pixels on each
  side;
* `scale` that is not positive, or a mapping that misses a deconvolved (round, label);
* `copy_root` that is relative but not `output` or `beside`.

**Completion and staleness.** Both kinds follow the success rule of D8 unchanged (orchestration,
"Completion"): the task record lists every declared output with its SHA-256, and a failed
attempt removes the outputs it wrote. A `deconvolve` task that reuses a verified copy writes no
image and so removes none when it fails ([Reuse and staleness](#reuse-and-staleness)). A task's
declared outputs are its own label files and its own record folder, so a failure or a rerun of the
`reference_stain` task removes or rewrites nothing of the reference round's task, and the reverse.
Each task's record is the completion of its own deconvolved round only.

**Snakemake.** `rules/tasks.smk` needs no change: it makes one rule per group of the manifest. A
`deconvolve.<round name>` group's outputs have the wildcard `{fov}`, and `{round}` too for a
sequencing or additional round, and lack `{sample}`, so the group is split by the fields its
outputs lack, as `fov`'s is split by sample. A GPU task gets `resources: gpu=1` and, on UGER,
`-l gpu=TRUE,operating_system=RedHat8` (orchestration-engines, "GPU tasks").

**Nextflow.** `nextflow/modules/deconvolve.nf` holds `DECONVOLVE` (label `cpu`) and
`DECONVOLVE_GPU` (label `gpu`), and `nextflow/modules/psf.nf` holds `PSF` (label `cpu`). `main.nf`
adds both kinds to `checkedTasks`' allowed list (`nextflow/main.nf:79`) and routes `deconvolve` by
`execution.gpu`, as it routes `fov` (`:171`–`:178`), before `SUMMARIZE_0`. On `google-batch`, the
GPU label's machine and image already apply (`nextflow/conf/google_batch.config:31`–`:35`), and
`withName: DECONVOLVE` takes `n2-highmem-4` (32 GB for a 10 GiB process).

## The configuration

```yaml
deconvolution:
  rounds: null              # null: the sequencing rounds; or a list of round names and reference_stain
  copy_root: output         # output, beside, or an absolute path
  iterations: 5             # required, no default (decision 5)
  method: shb               # the one method offered
  border_quality: 2
  tile_size: null           # null: untiled; or deconwolf's --tilesize in pixels (padding 20)
  scale: 64                 # a number, or {<round>: {<label>: number}}
  psf:
    source: generate        # generate or file
    files: null             # with file: {<label>: path}, relative to the input root or absolute
    voxel_size_um: null     # with file: the files' declared spacing {xy: …, z: …}
    na: 1.30
    immersion_ri: 1.518     # ni = ni0
    sample_ri: 1.338        # ns
    pinhole_radius_nm: 307  # back-projected into the sample
    depth_um: 0.0           # pz
    shape_zyx: [21, 33, 33]
    channels:               # per label
      ch00: {excitation_nm: 499, emission_nm: 529}
      ch01: {excitation_nm: 598, emission_nm: 634}
      ch02: {excitation_nm: 557, emission_nm: 577}
      ch03: {excitation_nm: 653, emission_nm: 692}

execution:
  deconwolf: null           # the dw executable; null: dw on PATH
  deconwolf_wisdom: null    # the wisdom store's folder; null: CPU tasks plan with --noplan
  tasks:
    deconvolve: {threads: 4, memory_mb: 14000, runtime_min: 60, gpu: false}
```

The example values are 9735seqA1's, from Huygens' settings (W-855's `psf.py`); they are not
defaults. Only `rounds`, `copy_root`, `method`, `border_quality`, `tile_size`, `scale`,
`psf.source`, `psf.depth_um` and `psf.shape_zyx` have defaults. The optics have none, because
they belong to the microscope. The voxel spacing is not in the block: it is the dataset's
`voxel_size_xy` and `voxel_size_z`, which the block requires ([The PSF](#the-psf), "The PSF's
inputs").

* `deconvolution` joins `BLOCKS` (`workflow/_shape.py:26`) first, before `processing`, because
  it runs first. A present block enables the stage (D4).
* `deconwolf` and `deconwolf_wisdom` join `EXECUTION_KEYS` (`workflow/_shape.py:92`), like
  `fiji`. Both are outside every hash: the binary's identity and the wisdom key are recorded per
  task, not hashed.
* `schema.json` gains the block and the two keys.

## The output

| Option | Where | Consequence |
| --- | --- | --- |
| **OU1. One folder layout, three places by `copy_root` (recommended)** | `output` (default): `<output root>/deconvolved/<sample_id>/dw-<copy id>/`. `beside`: `<root_input_path>/<dataset_id>-dw/<sample_id>/dw-<copy id>/`, in a sibling of the dataset's folder, which itself is not written. An absolute path `P`: `P/<dataset_id>/<sample_id>/dw-<copy id>/` | The default writes nothing outside the run's output root. `beside` keeps the copy with the data, like Huygens' outputs and W-910's `-zlib/` copies, for reuse by every later output root. A shared path serves several datasets. |
| OU2. Under the output root only | The `output` place | Another output id deconvolves again. |
| OU3. Beside the data only | The `beside` place | Every run writes into the data share; a run with no write access there cannot deconvolve. |
| OU4. The raw files' names inside the copy | `<copy>/<image dir>/<raw file name>` | A group's output templates must be equal for all its tasks (`workflow/_plan.py:558`–`:561`). Raw names that hold a FOV id or vary by FOV would refuse the plan. |

**Recommendation: OU1, with files named `<label>.tif`.**

The copy:

```
dw-<copy id>/
  copy.json                         the settings, the copy id, the PSF records (psf task)
  psf/<label>.tif, psf/psf.json     the PSFs (psf task)
  <image dir>/<label>.tif           one deconvolved channel (deconvolve task)
  records/<round name>/<fov>/deconvolution.json, <label>.dw.log, the task's lock
```

* A folder that `load_round` reads holds only channel images. `load_round` finds `<label>.tif` by
  its `*<label>*.tif` pattern (`io/tiff.py:425`), as it finds a raw file.
* Each image is ZYX uint16, written by `save_volume` with W-909's codec (zlib level 6, no
  predictor; `compressed-image-outputs.md`, decision 4). Its metadata is described in the next bullet.
* **Metadata: the copy stores exactly what the raw file stores.** The `deconvolve` command reads
  each raw channel with `load_volume` and the default configuration. It passes the result's
  `ImageMetadata` to `save_volume` only when `diagnostics["metadata_source"]` is `stored`, that
  is, when the raw file holds STARfinder metadata (`io/tiff.py:260`, `:268`). Otherwise it passes
  `metadata=None`, and `save_volume` writes no `starfinder_metadata` (`io/tiff.py:545`–`:546`).
  * **A raw file without stored geometry.** This is the plain TIFF or a microscope's OME-TIFF.
    `load_volume` gives it a frame built from its own path and marks the source `unknown`
    (`:260`, `:268`). Storing that `ImageMetadata` in the copy would turn every channel's source
    into `stored`, each with a different frame id. `load_round` would then refuse the copy with
    "channel frames differ" (`:439`–`:441`), although it accepts the raw round. With nothing
    stored, the copy's channels stay `unknown`. `load_round` combines them as it combines the raw
    ones: it checks that the geometry agrees (`:438`) and gives the round the frame of its folder
    (`:449`). The geometry (spacing, origin, direction, unit) is the unknown default in both. The
    frame is the folder's path in both, which is the copy's folder for the copy. Raw rounds
    already have one such frame per folder.
  * **A raw file with stored metadata.** The copy stores the same `ImageMetadata`, field for
    field. A raw round loads only when its channels' frames agree (`:439`–`:441`), so the copy's
    frames agree too. `load_round` returns the same metadata for both (`:449`).
  * **The rule.** A raw round that `load_round` accepts gives a copy that it accepts, with the
    same `metadata_source`, the same geometry and, where metadata is stored, the same frame. The
    alternative, one consistent round frame stored in every channel of the copy, was not chosen.
    It would store metadata that the raw images never declared. The channels' source would
    change from `unknown` to `stored`, so `load_round` would take the round's metadata from the
    files and not from the folder (`:449`). The copy would then no longer load exactly as the raw
    round loads.
* The codec is not optional here: the copy is a kept image output from the start, so
  `execution.image_compression` (W-909) also applies to it when it lands. Until then it is zlib.
* **Size.** W-915 kept deconwolf's uint16 outputs with zlib: 310–425 MB per channel volume,
  1.33–1.64 GB per position, against 1.13 GB for the raw position (uint8, uncompressed). So a copy
  is about 1.2–1.5 times its raw images. This is an inference: W-915's values used a per-volume
  scale of 101.9–124.5, not 64.
* `beside` writes into the data share. The plan checks nothing about access; the first
  `deconvolve` task fails if it cannot write.

## The scaling

deconwolf v0.4.5 adds `--offset` (5) to its input. Untiled, it subtracts the offset from its input
buffer, not from the output, so the saved output keeps it (`src/dw.c` 2661–2665). Tiled, it
subtracts it from each deconvolved tile and clips at 0 (`src/dw.c` 2012–2024). By default it saves
uint16 with a scale of 65,535 / the volume's maximum, written in its log (W-915, W-916).

| Option | Stored value | Consequence |
| --- | --- | --- |
| **SC1. Float output, offset removed, one fixed scale, uint16 (recommended)** | rint((y − o) × s), clipped to 0–65,535, where y is deconwolf's float32 output, o the kept offset of the mode and s the scale of the (round, label) | Every FOV of a round and channel has one intensity scale, so thresholds and the two-pass statistics compare FOVs. One quantization, of at most 0.5 / s grey levels. |
| SC2. deconwolf's own uint16 output | stored / (65,535 / max) | Each volume has its own scale: on W-915 101.9–124.5 across channels and positions. Intensities are not comparable across FOVs without the logged scale, and the offset stays in. |
| SC3. The raw dtype with a recorded scale | uint8, value × s with s ≈ 0.4 | Huygens' form (one stretch factor for all channels). 256 levels for values that reach 638: steps of about 2.5 raw grey levels. Half SC1's bytes. |
| SC4. float32 | y − o | No quantization; four bytes per voxel and poor zlib compression. A float raw image is a new case for every consumer. |

**Recommendation: SC1.**

* **The output dtype is uint16.** A deconvolved value exceeds the raw range: W-915's maxima after
  the offset are 521–638 grey levels on all twelve channel volumes of a uint8 dataset.
* **The default scale is 64.** It holds 0–1,023.98 grey levels, 1.6 times the largest value seen.
  Its step is 1/64 of a grey level. A dataset whose raw images are not uint8 sets its own scale.
  "Preparing a dataset" states the rule: s × (largest expected value) ≤ 65,535.
* **The scale is one number by default.** A mapping `{<round>: {<label>: s}}` sets it per round and channel, as the issue asks.
  A plan refuses a mapping that misses a deconvolved (round, label).
* **The kept offset is removed by mode.** The command subtracts 5 untiled and 0 tiled, the rule
  of v0.4.5, and clips at 0. The rule is known for v0.4.5 only, so a binary whose `--version`
  names another version fails the task, naming it: a new version must re-establish the rule
  first. The command also checks the untiled output's minimum. Below 4.5, the offset was not
  kept, and the task fails. This is the half of W-916's check (`scripts/offsets.py`) that holds on
  any data. The other half, a tiled minimum below 0.5, needs voxels that deconvolve to about 0,
  as 9735seqA1's background does. A dataset need not have such voxels, so the record holds the
  minimum without a check.
* **Float output.** The command asks deconwolf for float32 output (`--float`), so that the only
  quantization is the stage's own. Validation check DW1 confirms that v0.4.5's `--help` lists the
  option. If it does not, the command reads deconwolf's uint16 output and its logged scale, and
  the stage's error bound gains one step of that scale (validation RD1).
* **Saturation** is counted, not refused. The record holds the number of voxels clipped at 65,535
  and at 0 and the float maximum, and the task log warns when the first count is not 0.

## The PSF

| Option | Rule | Consequence |
| --- | --- | --- |
| **PS1. Generated by the `psf` task from configuration, or PS2 by file (recommended)** | `psfmodels.confocal_psf(model="vectorial")` per label: NA, ni = ni0, ns, the label's excitation and emission, pinhole in Airy units = radius / (0.61 × emission / NA), the dataset's `voxel_size_xy` (Y and X) and `voxel_size_z`, depth `pz`, shape `shape_zyx`; psfmodels' defaults for the coverslip, the working distance and oversampling, recorded. Normalised to sum 1, centred on its peak, written float32 | One PSF file per label for the whole run; every `deconvolve` task records its SHA-256. The lock gains psfmodels (question 2). |
| PS2 alone. A supplied file per label | `psf.files` and its declared spacing `psf.voxel_size_um` | No dependency. The PSF's optical parameters are whatever made the file; the record holds its hash and its declared spacing. |
| A PSF generated inside each `deconvolve` task | As PS1, per task | 224 generations of the same arrays; the same hash in one environment, but no one file to point to. |

**Recommendation: PS1 with PS2.**

* **The PSF's inputs.** A generated PSF depends on these values, and on nothing else:

  | Input | Read from | Recorded in `psf.json` | In the copy id |
  | --- | --- | --- | --- |
  | NA, ni = ni0, ns, the pinhole radius, the depth, the shape, each label's excitation and emission | the `deconvolution` block (`psf.*`) | yes | yes, as part of the block |
  | The spacing in Y and X | the top-level `voxel_size_xy` (`DATASET_KEYS`, `workflow/_shape.py:28`–`:31`; the same value `Dataset.voxel_size_xy` takes, `dataset/workflow.py:945`) | yes, as psfmodels' `dxy` | yes, explicitly |
  | The spacing in Z | the top-level `voxel_size_z` (as above) | yes, as `dz` | yes, explicitly |
  | psfmodels' defaults for the coverslip, the working distance and oversampling (tg = tg0 = 170 µm, ng = ng0 = 1.515, ti0 = 150 µm, oversample 3, W-855) | psfmodels' own version | yes, with psfmodels' version | no: they are not settings; a copy's PSF is written once ([Reuse and staleness](#reuse-and-staleness)) |

  The spacing is the one input from outside the block, so it is named in the copy id on its own,
  whatever `psf.source` is. The spacing the raw files may store is not used: the plan reads no
  image, and the copy id must be known when the plan runs. For a supplied file, the identity is
  the file's SHA-256 and its declared spacing, `psf.voxel_size_um`. The plan refuses a declared
  spacing that differs from the dataset's ("Plan refusals"), so the declared spacing in the id
  always equals `voxel_size_xy` and `voxel_size_z`.
* **The function.** W-855 amendment 1: in psfmodels 0.3.3 `tot_psf` is the light-sheet total PSF;
  `confocal_psf(model="vectorial")` is the confocal one. Not represented: the excitation beam
  fill and Huygens' own PSF, which Huygens Essential does not export.
* **Centring** (W-855 amendment 2). An off-centre PSF shifts the result along z. The `psf` task
  moves the PSF by whole voxels so that its peak is at the centre index: `np.roll`, then the
  slices the roll wrapped around (planes, rows or columns, in each axis it shifts) set to 0, then
  normalised to sum 1 again. A PSF with non-zero values in those slices therefore loses them: the
  result is not the shifted PSF, but the shifted PSF truncated and renormalised. The record holds the shift, the share
  of the sum dropped and the peak's offset before the shift. At depth 0 W-855's PSFs needed no
  shift; at 5.2 µm they moved −2 planes and lost 0.23–0.24 % of their sum.
* **The size.** The default 21 × 33 × 33 is W-855's: it holds 96.6–96.8 % of the sum of a
  41 × 65 × 65 computation at depth 0. The PSF's size sets deconwolf's job shape (border quality 2
  pads the image by the PSF's size less one: 30 × 3,072 × 3,072 becomes 50 × 3,104 × 3,104, W-916),
  so it enters the wisdom key.
* **Records.** `psf/psf.json` holds, per label: the function and psfmodels' version, every
  parameter passed (the spacing `dxy` and `dz` among them) and every default used, the shape, the peak index before and after centring,
  the shift, the sum dropped, the axial and lateral FWHM through the peak, the file's SHA-256 and
  the decoded array's. With `source: file`, it holds the file's path, both hashes and the declared
  spacing, and the task
  checks that the file's dtype is a real type, its axes are ZYX, it has no negative value, and it is
  normalised to sum 1 (it renormalises in memory otherwise and records that).
* **The dependency.** psfmodels joins the lock as an optional extra, `deconvolution`, and the
  image's CPU extras, so that every target has it. W-855's install built psfmodels 0.3.3 from its
  source distribution (its `env-install` records list the build environment). The implementation
  checks whether the lock resolves a wheel for Linux x86-64 and Python 3.12. If not, the image's
  build stage gains a C++ compiler. The CI environment (`dev`, `local-registration`, `checkpoint`)
  lacks the extra, so the tests that generate a PSF skip there with the reason.

## deconwolf's settings

| Setting | Recommendation | Alternatives and evidence |
| --- | --- | --- |
| Input | The command reads each raw channel with the package's reader (`load_volume`), hashes the decoded array, and writes it as an uncompressed float32 ZYX TIFF in the task's temporary folder, which deconwolf reads, as W-915 did | Passing the raw file itself: deconwolf would read OME-TIFF, ImageJ or deflate files with its own reader, and the array it deconvolves would not be the one the record hashes. |
| Method | SHB, deconwolf's default; `--gpu` runs it as SHBCL2 (W-915). The key `method` has the one value `shb`. | RL (`--method rl`) and the others are not offered: none was measured against Huygens in deconwolf. W-855's RL was clij2-fft's, not deconwolf's. |
| Iterations | `iterations`, required, no default | DS2: a default of 5, W-915's frozen count for 9735seqA1. It was chosen on one position, at the low edge of the grid 5–80, by a 0.005 tie-break. DS3: deconwolf's automatic stopping (relative I-divergence change below 0.02, at most 250). On 022 it stopped after 6–7 iterations, at a mean NCC of 0.9675 against 0.9814 for 5, outside the 0.005 tolerance, and it stops per volume, so FOVs of one dataset would differ. |
| Border | Quality 2, deconwolf's default (W-915, W-916) | Other qualities change the job shape and the result; not measured. |
| Background | deconwolf's automatic estimate (default) | W-855's preparation subtracted the histogram mode and clipped at 0 before deconwolf. The mode was 0 on all twelve channel volumes, so the evidence does not tell the two apart; the stage does not prepare ([What the stage does not do](#what-the-stage-does-not-do)). |
| Offset | 5, the default, removed by the stage (decision 3) | `--offset 0` changes the algorithm's positivity margin; not measured. |
| Tiling | Untiled by default; `tile_size` sets `--tilesize` (padding 20, the default) | Tiling is in the hash: it changes values by up to 1.3 % of the maximum (W-916). Untiled needs 14 GiB of GPU memory on a 30 × 3,072 × 3,072 volume, which fits UGER's A30 (24 GB) and Google's L4 (24 GB); tiles of 2,048 fit 4 GiB. |

**How a dataset chooses its iterations.** "Preparing a dataset" gives W-855's development-position
rule, as W-915 applied it: on one development position with a comparator (a Huygens result of
the same position), run the grid 5, 10, 20, 40 and 80; take the count with the highest
whole-volume NCC to the comparator averaged over the channels; among counts within 0.005 of the
best, take the fewest. Freeze the choice before the held-out positions run. A dataset without a
comparator states its count and its reason in its configuration's comment; the stage does not
choose for it.

## Execution

**Modes.**

| Mode | How it is chosen | deconwolf call | Evidence (one 30 × 3,072 × 3,072 channel, W-916) |
| --- | --- | --- | --- |
| CPU (default) | `execution.tasks.deconvolve.gpu: false` | `--threads <task threads>`, `OMP_NUM_THREADS` the same; wisdom from the store or `--noplan` | 74 s at 6 threads with wisdom; `--noplan` 153 s at 6 threads, 166 s at 4, 575 s at 1; 10 GiB RSS, 10.4 GiB VmPeak |
| GPU | `gpu: true` | `--gpu --threads 1`; no wisdom; the device is the first OpenCL device the job sees | 35 s untiled, 14.0 GiB GPU, 4.8 GiB RSS, 21.4 GiB VmPeak; tiled 2,048: 53.5 s, 4.0 GiB GPU, 9.4 GiB VmPeak |

| Option | Consequence |
| --- | --- |
| **EX1. CPU by default, the GPU by the task's `gpu` (recommended)** | Every host, UGER node and cloud machine runs the default. The GPU is the fast path where one is free. The device is execution, like the threads: switching it reruns nothing ([Reuse and staleness](#reuse-and-staleness)). |
| EX2. GPU by default | 2.2 GPU-h for 224 volumes against about 30 slot-h on the CPU (W-916's projection), but UGER has two usable GPU nodes (W-641), and a host without a GPU fails every task. |

**Recommendation: EX1.** The GPU uses the orchestration's device choice unchanged: the job script
picks the GPU without a compute process and exports `CUDA_VISIBLE_DEVICES`
(orchestration-engines, "Choice: which GPU a job uses"). deconwolf takes the first OpenCL platform
(W-915); in the image the only platform is NVIDIA's, whose OpenCL honours `CUDA_VISIBLE_DEVICES`.
Validation IM3 checks that on a UGER GPU node.

**Resources per task** (four channels of 30 × 3,072 × 3,072; LN's 50 × 1,496 × 1,496 volumes
have about a third of the job's voxels):

| Place | CPU task | GPU task |
| --- | --- | --- |
| Defaults | 4 threads, `memory_mb` 14,000, `runtime_min` 60 | 1 thread, `memory_mb` 26,000, `runtime_min` 30 |
| Host | as the defaults; four concurrent tasks hold about 40 GiB | one task per GPU; untiled needs 14 GiB of the card |
| UGER | `h_vmem` is per slot: 14,000 / 4 = 3,500 MB × 4 slots covers the 10.4 GiB VmPeak (W-916); 4 × 166 s with `--noplan`, about 11 min, or 5 min with wisdom at 6 threads | `-l gpu=TRUE,operating_system=RedHat8`; `h_vmem` 26,000 MB covers the 21.4 GiB VmPeak, which UGER counts |
| Google Batch | `n2-highmem-4` (32 GB) | `g2-standard-4` with one L4, the GPU label's machine; not measured on Google |

The runtimes double on a retry (orchestration-engines, "Resources of a retried task"). These
figures are W-916's, on GP099-29C; UGER's and Google's CPUs were not measured.

**deconwolf in the image.**

| Option | Consequence |
| --- | --- |
| **The runtime stage of every target (recommended)** | A build stage `build-deconwolf` builds v0.4.5 (commit `ca062c49`) from source on the same Debian bookworm base: CMake 3.25, `-DENABLE_GPU=ON -DCMAKE_BUILD_TYPE=Release`, and `-include inttypes.h`, which W-915 needed with gcc 9.4 (its `src/dw_imshift.c` uses `PRId64`). The `runtime` stage (`Containerfile:111`) installs its libraries (FFTW single and double precision, libtiff, GSL, libpng, the OpenCL ICD loader, libgomp), copies `dw` and `libtrafo.so`, and writes `/etc/OpenCL/vendors/nvidia.icd` naming `libnvidia-opencl.so.1`, which `apptainer --nv` and `docker --gpus` bind from the host. `cpu`, `learned` and `learned-cuda` all have it, at a few tens of MB. |
| A new target `deconwolf` | A run has one image for all its tasks (`execution.container`, `workflow/_plan.py:522`), so the FOV tasks would also run in it. A per-kind image is a new execution feature. |
| The learned image | Couples deconvolution to the learned extras; the CPU image would lack it. |
| A host binary by path only | W-916: the host-built binary does not start on UGER's RedHat 8 (glibc 2.29 needed, 2.28 there; libtrafo, GSL, OpenCL and FFTW not found). In the image, deconwolf uses the image's glibc 2.36 and libraries. |

* The source is fetched with a pinned checksum (`ADD --checksum=sha256:…` of the tag's archive),
  as the base images are pinned by digest. Its labels gain `org.starfinder.deconwolf=0.4.5` and the
  binary's SHA-256. `image_check.py` gains a `deconwolf` check: `dw --version` names 0.4.5 with
  OpenMP, OpenCL and VkFFT, and `ldd` finds every library.
* `execution.deconwolf` names another binary for a run without the image (the host's own build).
  The record holds its SHA-256 either way.

## FFTW wisdom

W-855's reuse rule, as the issue restates it: a key, a store with a manifest, one generating job per
new key, read-only use, and `--noplan` when no wisdom matches.

**How deconwolf finds wisdom** (W-916, `src/fft.c`, `get_swf_file_name`). It reads and writes
`fftw_wisdom_float_inplace_threads_<n>.dat` in `$HOME/.config/deconwolf/` when `$HOME/.config`
exists, and otherwise under that bare name in its working directory. With `--gpu` it imports wisdom
but exports none. A CPU run that plans writes its wisdom file there (W-916's first CPU arm wrote
it into the steps' shared working directory).

| Option | Consequence |
| --- | --- |
| **WS1. A read-only store, a generating command, `--noplan` otherwise (recommended)** | A cold plan costs about 27 min at 6 threads once per key, not per task. A task never writes the store. A task without a matching key still finishes, in about twice the warm time. |
| WS2. A `wisdom` task in the plan | The plan cannot know which CPU model a cluster job will land on, and a wisdom file made on one model is not the key of another. |
| WS3. Always `--noplan` | No store to keep; every CPU volume costs about twice the warm time (153 s against 74 s at 6 threads). |
| WS4. deconwolf's defaults (plan cold, keep the file) | Every fresh job plans cold: 1,692 s for a volume that then takes 74 s. Concurrent jobs write one shared file. |

**Recommendation: WS1.**

* **The key.** The SHA-256 of the canonical JSON of:

  | Field | Value |
  | --- | --- |
  | `job_shape_zyx` | The padded shape deconwolf transforms: the image shape plus the PSF shape less one in each axis (border quality 2), for example 50 × 3,104 × 3,104 |
  | `psf_shape_zyx` | The PSF's shape. The PSF's values do not affect FFTW's plans, so its hash is not in the key |
  | `border_quality`, `tile_size` | The two settings that change the transform sizes |
  | `threads` | The task's threads; deconwolf's file name holds it too |
  | `cpu` | `model name` and the sorted `flags` of `/proc/cpuinfo`, hashed |
  | `fftw`, `deconwolf` | The FFT backend line of `dw --version` (for example `fftw-3.3.8-sse2-avx`) and deconwolf's version and binary SHA-256 |

* **The store.** `execution.deconwolf_wisdom` names a folder: `<store>/<key>/<deconwolf's file
  name>` and `<store>/manifest.json`, which lists each key with its fields, the file's bytes and
  SHA-256, and when, where (host, job id) and by which command it was made. The manifest is
  rewritten atomically, under a lock file made with `O_EXCL`.
* **One generating job per new key.** `starfinder deconvolve wisdom --store <dir> --shape <z y x>
  --psf-shape <z y x> --threads <n>` plans for that key on the node it runs on, with deconwolf's
  FFTW_MEASURE on a synthetic volume of the image shape. The plans depend on the sizes only. It
  takes the store's lock for the key, refuses if the key exists, and adds the file and its
  manifest entry. It is run by hand or as a job, once per (CPU model, shape, threads) of a
  deployment, outside the plan (WS2's reason). W-916's planning took 1,617 s at 6 threads.
* **Read-only use.** A CPU `deconvolve` task computes its key, and when the store's manifest lists
  it, it checks the file's SHA-256 against the manifest and copies the file into its own
  `$HOME/.config/deconwolf/`. A file whose SHA-256 differs from the manifest fails the task,
  naming the file: a damaged store is reported, not worked around. Whatever deconwolf writes back
  into that private folder is deleted with the task's temporary folder. The store is never opened for writing
  by a task.
* **`--noplan` when no key matches**, and when the store is unset. The record says so.
* **Finding the store.** The command runs deconwolf with `HOME` set to a private folder under the
  task's temporary root, with `.config/deconwolf/` created first (holding the copy or nothing), and
  with the same folder as its working directory. So no wisdom is read or written outside the task,
  which W-916's first CPU arm did through the bare-name fallback. `HOME` is set for the deconwolf
  process only.
* **GPU tasks** run with an empty wisdom folder, the condition in which W-915 and W-916 measured
  them (VkFFT plans on the card).
* **The values.** Planning changes the output by up to 1.6 × 10⁻⁵ of the maximum (W-916: ESTIMATE
  against MEASURE). Threads and reused wisdom change no byte within one planning mode. So the mode
  is recorded per task and stays outside the hash (question 5).

## Records

| Option | Consequence |
| --- | --- |
| **RC1. A record per (FOV, deconvolved round) beside the copy, `copy.json`, and a `deconvolution` key in `run.json` (recommended)** | The copy carries its own provenance wherever it is reused, independent of the run that made it. A FOV's `run.json` says which copy it read. |
| RC2. Only the task record | The task record lies in the output root that made the copy; a copy reused from another root would have no provenance beside it. |
| RC3. deconwolf's log only | No hashes, no scale, no wisdom key. |

**Recommendation: RC1.**

**`records/<round name>/<fov>/deconvolution.json`** (format `starfinder.deconvolution/1`), one per
`deconvolve` task, written by that task only:

| Key | Value |
| --- | --- |
| `copy_id`, `fov`, `round`, `labels`, `image_dir` | The task's identity: `round` is the round name (`reference_stain` for the stains); `image_dir` is the image folder its labels were written to, which another round may share |
| `deconwolf` | `version`, `dw --version` text, the binary's path and SHA-256, the FFT backend |
| `device` | `cpu` with the CPU model and threads, or `gpu` with the OpenCL device name and `CUDA_VISIBLE_DEVICES` |
| `options` | Iterations, method, border quality, tile size and padding, offset, background, and the argument list passed |
| `planning` | `{mode: measure, wisdom_key, wisdom_sha256}`, or `{mode: estimate}` with the reason (`no store` or `no matching key`), or `{mode: gpu}` |
| `channels[]` | Per label: the input `path`, `file_sha256`, and decoded `dtype`, `shape` and `sha256`; the PSF `path` and `file_sha256`; deconwolf's `iterations_done`; the float output's minimum and maximum; the kept offset removed; the scale; the counts clipped at 0 and at 65,535; the output `path`, `file_sha256` and decoded `sha256`; the log's path and SHA-256; the wall and CPU seconds |
| `code` | The package's code identity (`orchestration.md`, "The code identity") |
| `started_at`, `ended_at`, `host`, `job_id` | Where and when |

**`copy.json`**: the copy id and every field it hashes (`dataset_id`, `sample_id`,
`image_dir_template`, `voxel_size_xy`, `voxel_size_z`, the resolved block less `copy_root`,
`psf_files`), and the `psf/psf.json` record with its SHA-256.

**`run.json` of a FOV task** that read a deconvolved round gains `deconvolution`, an additive key
like `spot_finding_input` (`checkpoints.md`, "run.json"): `{copy: <absolute path>, copy_id,
rounds: {<round name>: {record, record_sha256}}}`, one entry per deconvolved round it read, so that
a `morphology` task that reads the reference round's stains names the `reference_stain` record. Its `inputs[]` list the copy's files, with absolute
paths when the copy lies outside the input root (the manifest's rule, OC4). The record's format
version does not change.

Hashes follow W-909's identity rule (ID1): every file is recorded by its file SHA-256 for
integrity; a comparison of content between two writes uses the decoded identity. The record holds
both for every input and output image, as `_prepared.py`'s record does.

## Reuse and staleness

| Option | Consequence |
| --- | --- |
| **RU1. Settings-named copies, verified reuse, consumers check (recommended)** | Two settings never write one folder. A round is redone only when its inputs or PSF changed. |
| RU2. One copy per root, refused on a mismatch | A second setting needs a new root by hand. |
| RU3. One copy per root, overwritten | A run with other settings silently changes the images another run's records name. |

**Recommendation: RU1.**

* **Within a run.** The engines decide by the fingerprint (orchestration-engines, "The task
  fingerprint"): a `deconvolve` or `psf` task reruns when the block or the common part changes
  (its `blocks_sha256`, whose common part holds `voxel_size_xy` and `voxel_size_z`), when a
  raw file's bytes or modification time change (`inputs`), or when the PSF task reruns (`needs`).
  Its consumers follow through `needs`. Switching the device, the threads or the wisdom store
  changes no hash and reruns nothing.
* **Across runs: the copy id.** A change of any setting in the block other than `copy_root`, of
  `voxel_size_xy` or `voxel_size_z`, or of a supplied PSF file's content gives another copy id
  ([Terms](#terms)) and so another folder, where the `psf` task writes a new PSF and every round is
  deconvolved again. The old copy stays for the runs that read it.
* **Across runs: the PSF.** The `psf` task of a later run finds `copy.json` with its copy id. It
  writes nothing and lists the existing files when all of these hold:
  * the fields `copy.json` records equal its own, the spacing included;
  * the parameters `psf/psf.json` records for each label (`dxy`, `dz` and the optics, or a
    supplied file's SHA-256 and declared spacing) equal the ones it would use;
  * every PSF file has the SHA-256 that `psf/psf.json` records.

  So a copy's PSF is written once, and a psfmodels or NumPy build that would write other bytes
  does not invalidate the copy's rounds. A difference in the first two cannot happen for a
  correct copy, because the copy id hashes those fields. A difference in any of the three means a
  damaged or hand-edited copy, and the task fails, naming the file and the field.
* **Across runs: verified reuse.** A `deconvolve` task whose record folder
  `records/<round name>/<fov>/` already holds a record hashes its raw inputs and the PSF files. It
  reads only that record, never the record of another round that shares its image folder. It
  reuses the round when:
  * the record's `round`, input `file_sha256`, PSF `file_sha256` and `copy_id` equal its own (the
    copy id covers the spacing and the PSF's parameters);
  * every output file named in the record exists with its recorded SHA-256;
  * with `execution.code_version: rerun` (orchestration-engines, "The code version in the fingerprint"),
    also the record's code identity and deconwolf binary hash equal its own.

  It then writes its task record with `reused: true` and runs no deconwolf. Otherwise it redoes
  the round: it removes its own old record first, writes its own label images atomically, and its
  record last. The other round's label files and record in a shared image folder are not touched,
  so it stays complete.
  A lock file in the record folder, `records/<round name>/<fov>/.lock`, made with `O_EXCL`, keeps
  two tasks of one (round name, FOV) from writing at once; the second waits up to its runtime and
  then fails. The task removes its lock when it ends, whether it succeeded or failed. The tasks of the reference round and of `reference_stain` hold different locks and
  may run at once, since they write different files.
* **A consumer checks the copy.** A FOV task that reads a deconvolved round checks that the record
  of that round name exists and names its files, and that each file's SHA-256 equals the record's. It takes
  that hash with `hash_inputs`, and for these files also when `hash_inputs` is off. A mismatch fails the task, naming the file: the copy is stale
  or damaged. It then records the copy in `run.json` ([Records](#records)).
* **Verified resume and saved sets.** `_resume.check` compares `run.json`'s input hashes, which
  are the copy's files, so a resume against a changed copy is refused as for changed raw images. A
  saved spot-finding set (`check_input`, `INPUT_FIELDS` at `workflow/_resume.py:218`) must also
  match the run's deconvolution. The set's `run.json` `deconvolution.copy_id` must equal the
  run's copy id. Both may also be absent, for a set and a run without the stage. Otherwise the task
  fails, naming both.

## What the stage does not do

* No Huygens: the stage neither runs it nor reads its outputs. A Huygens-deconvolved dataset is an
  input like any other and has no `deconvolution` block.
* No measured PSF and no PSF fitting. A measured PSF may be supplied as a file (`psf.source:
  file`); the stage does not make one.
* No tuning by decoding or spot results; the iterations come from configuration.
* No preprocessing: no background subtraction, no hot-pixel removal, no bleaching correction
  (which Huygens does, W-855). deconwolf's own background estimate is its default.
* No new rounds: the copy holds exactly the deconvolved rounds' channels.
* A dataset without the block does not run the stage, and its plan is unchanged.

## MATLAB

| Option | Consequence |
| --- | --- |
| **MT1. MATLAB unchanged; the block is refused with `backend: matlab` (recommended)** | AGENTS.md: the files, keys and conventions MATLAB reads or writes stay unchanged. |
| MT2. Write a copy for MATLAB | A copy in MATLAB's naming and dtype; MATLAB's pipeline was never checked on deconvolved uint16. |

**Recommendation: MT1.** The copy is in the input's folder layout and its files are TIFF + zlib,
which MATLAB R2023b reads (W-910). A user could point a MATLAB run's input root at a copy by hand,
but the file names (`<label>.tif`) and the dtype differ from what MATLAB's datasets use, so the
stage does not offer it, and no MATLAB file or key changes.

## Docs

When the implementation lands, it writes:

* `docs/workflow/index.md`: a section on the deconvolution stage: when to use it, the copy,
  reuse, the GPU and the wisdom store.
* `docs/workflow/preparing-a-dataset.md`: when a dataset needs the block, which optics to take
  from the microscope's records, how to choose the iterations (W-855's rule), and how to choose
  the scale.
* `docs/workflow-configuration.md`: the `deconvolution` block and the two `execution` keys, in its
  tables.
* `docs/workflow/broad-uger.md`: the GPU and CPU resources, and how to make the wisdom store.
* `docs/checkpoints-reference.md`: the `run.json` key.
* No row in `docs/migration.md`: no existing key changes.

## Pages that change

When the implementation lands:

* `design/specs/orchestration.md`: "Terms" (the kinds `psf` and `deconvolve`), "The blocks of a
  task kind", "The task graph from data dependencies", "The Python configuration shape" (the block
  and the execution keys) and "What an engine needs, per task kind".
* `design/specs/orchestration-engines.md`: "Coverage", "GPU tasks (D6)", "The directory and the
  modules" (the two modules), "The image" (the build stage) and "The task fingerprint" (the
  `inputs` component, graph step 4).
* `design/specs/checkpoints.md`: "run.json" (the `deconvolution` key).
* `design/specs/compressed-image-outputs.md`: the inventory gains the copy's images as a kept
  output with zlib.
* `src/python/starfinder/workflow/schema.json`, `Containerfile`, `src/python/scripts/image_check.py`,
  `nextflow/`, and the lock (question 2).

## Exclusions

* No implementation, test or configuration in this issue: it writes the design records only.
* No conversion of a Huygens-deconvolved dataset, and no comparison with Huygens inside the stage.
* No other deconvolution program and no other deconwolf method.
* No timing claims beyond W-916's host: UGER's and Google's machines were not measured.
* The wisdom reuse rule is taken as the issue restates W-855's comment 70988672; the comment
  itself was not read by this issue's session.
