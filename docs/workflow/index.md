# Workflow

:::{container} sf-lead
Run STARfinder over every FOV and sample of a dataset. One command plans the run
from a configuration file; Snakemake or Nextflow then runs each planned task as a
job, on a laptop or on a cluster.
:::

```{toctree}
:hidden:
:maxdepth: 2

Preparing a dataset <preparing-a-dataset>
Reference <reference>
Broad UGER <broad-uger>
MATLAB legacy workflow <matlab-legacy>
```

## Quick start

Run these commands from `src/python` in the repository, with [uv](https://docs.astral.sh/uv/).
They write a small synthetic dataset (four FOVs of 5 × 36 × 36 voxels, four
rounds of four channels, a nuclear stain and cell labels), then plan and run the
{download}`full configuration <../examples/workflow-full.yaml>` on it: every
block, from the molecules of each FOV to the sample's export and report. `WORK`
must be a new folder outside the checkout.

```bash
WORK=/tmp/starfinder-workflow
uv run python scripts/workflow_golden.py fixture "$WORK/fixture"
cp -r "$WORK/fixture/labels" "$WORK/fixture/input/sample-tiny-v1/sample/"
uv run python ../../docs/examples/prepare_workflow.py full "$WORK/snakemake" --input-root "$WORK/fixture/input"
uv run starfinder workflow plan --config "$WORK/snakemake/config.yaml"
```

The labels go into the input folder, where the example's two imported
segmentation runs read them. The {download}`preparation script <../examples/prepare_workflow.py>`
copies the example and fills in its two roots. The plan prints the manifest's path
and its 29 tasks by kind.

**Snakemake.** A dry run lists the jobs; the second command runs them on one core,
with the extras that stitching and the export need, and `verify` checks that every
task is complete:

```bash
uv run --group workflow snakemake -s ../../workflow/Snakefile \
  --configfile "$WORK/snakemake/config.yaml" --directory "$WORK/snakemake" --cores 1 -n
uv run --extra multiview-stitcher --extra anndata --group workflow snakemake -s ../../workflow/Snakefile \
  --configfile "$WORK/snakemake/config.yaml" --directory "$WORK/snakemake" --cores 1
uv run starfinder workflow verify --manifest "$WORK/snakemake/output/sample-tiny-v1/full/workflow/manifest.json" --final
```

**Nextflow.** With Nextflow 26.10.4 or newer and Java 17 or newer on `PATH`
({doc}`../get-started/installation`), a stub run checks the graph and writes
placeholder outputs in its own folder; the second run processes the images.
Nextflow runs from each run's folder, which keeps its `work/` and `.nextflow/`
folders out of the checkout:

```bash
PROJECT=$PWD
uv run python ../../docs/examples/prepare_workflow.py full "$WORK/nextflow-stub" --input-root "$WORK/fixture/input"
uv run python ../../docs/examples/prepare_workflow.py full "$WORK/nextflow" --input-root "$WORK/fixture/input"
cd "$WORK/nextflow-stub"
uv run --project "$PROJECT" nextflow run "$PROJECT/../../nextflow/main.nf" -profile local \
  --config "$WORK/nextflow-stub/config.yaml" -stub-run
cd "$WORK/nextflow"
uv run --project "$PROJECT" --extra multiview-stitcher --extra anndata nextflow run "$PROJECT/../../nextflow/main.nf" \
  -profile local --config "$WORK/nextflow/config.yaml"
cd "$PROJECT"
uv run starfinder workflow verify --manifest "$WORK/nextflow/output/sample-tiny-v1/full/workflow/manifest.json" --final
```

For your own data, {doc}`prepare the dataset <preparing-a-dataset>` and edit a
copy of the full configuration or of the
{download}`minimal configuration <../examples/workflow-minimal.yaml>`, which runs
the per-FOV processing alone; the
{download}`recipe 2 example <../examples/workflow-recipe-2.yaml>` shows an explicit
preprocessing recipe. Four examples of global processing run the per-FOV
processing with explicit steps: each channel min-max normalized to `uint8` with
truncation, then histogram-matched to channel 0, and no reconstruction after
registration. Their registration differs:

* {download}`global <../examples/workflow-global.yaml>`: one translation on the
  channels' maximum;
* {download}`global with B-spline <../examples/workflow-global-bspline.yaml>`: a
  translation, then `bspline` with its defaults;
* {download}`global with demons <../examples/workflow-global-demons.yaml>`: a
  translation, then `demons` with its defaults, on the whole FOV;
* {download}`global with demons in tiles <../examples/workflow-global-demons-tiled.yaml>`:
  `mode: tiled`, a translation of the whole FOV, then `demons` with its defaults
  in each tile (`processing.tiled.local_registration`).

`prepare_workflow.py` takes each example's name without `workflow-` and `.yaml`,
for example `global-demons`.

## What it does

**The plan.** `starfinder workflow plan --config <file>` reads one configuration
file, checks it and refuses, before any task runs, a configuration it cannot run,
naming the key. It reads the FOV list, the file `fov_list` names or else
`fovs.csv` under the input folder (`<root_input_path>/<dataset_id>/<sample_id>/`),
looks up every input image in place and
writes two files under the output root (`<root_output_path>/<dataset_id>/<output_id>/`):
`workflow/resolved.json`, the configuration with every default written out, and
`workflow/manifest.json`, the tasks.

**The tasks.** Each present block of the configuration adds its tasks: `fov` per
FOV for `processing` (or `tile_global`, `tile` and `tile_merge` in tiled mode, and
`summarize` and `merge` for supplied statistics), `morphology` per FOV, `segment`
per FOV and segmentation run, `assign` per FOV, `stitch` (or the BigStitcher
tasks), `export` and `report` per sample, and with a report one `compare`. The manifest gives each task its inputs, its declared outputs, the
tasks it needs, its resources and its command,
`starfinder workflow <command> --manifest <manifest> --task <id>`. The
{doc}`reference <reference>` lists the task kinds and their outputs.

**The engines.** Snakemake (`workflow/Snakefile`) and Nextflow
(`nextflow/main.nf`) both run the plan first, make one job per task and run the
task's command. Neither reads the configuration's blocks: what a task does is
its command's. A task checks its inputs and the configuration's hash before any work,
writes its log, runs one API call, checks its declared outputs and writes its
task record last.

**Records.** Exit status 0 means the task succeeded; 1, that it failed, after it
removed its declared outputs; 2, that it refused before any work; 3, that a
verified resume was refused; 4, that an export was held for review (below). A task is complete when its record says `succeeded`
and lists its declared outputs and its log, each still there with its recorded
size (a temporary output may since have been removed), and its fingerprint (its
configuration, inputs and needs) has not changed. `starfinder workflow verify`
applies that rule to every task.

**Reruns.** The engines decide differently. Nextflow runs `verify` before its
tasks and again at the end: with `-resume` it keeps the complete tasks and reruns
the others and the tasks downstream of them, and the run fails if a task is still
not complete. Snakemake does not read the records. It reruns a task when the
task's record or a declared non-temporary output is missing (including a whole
FOV folder or a terminal output), its fingerprint has changed or an input is newer
than its outputs, and then the tasks downstream. Removed temporary outputs do not
schedule jobs after a completed tiled run. Snakemake uses its own rerun triggers:
it does not rerun a task for a deleted log or for an output of another size;
when a changed output has a newer
modification time, the tasks that read it rerun on the changed file, and the run
still exits 0; with its original modification time, no task reruns. So end a
Snakemake run with `verify --final`, as the Quick start
does. To rerun a task it names, remove the task's record,
`workflow/tasks/<task>.json`, and run Snakemake again.

(workflow-placement-review)=
**A placement that needs review.** After `stitch`, the review criterion of the
{doc}`stitching contract <../stitching-contract>` marks a placement `required`
when a FOV is not `image_supported`, a correction exceeds half the nominal
overlap width, a kept link's residual exceeds 1 µm (both bounds are
`stitching.review`), the optimizer removed a link, or the placement is supplied.
The `export` task then stops before it reads any origin and writes the held
page `workflow/held/<sample>/<route>.html`: the review's reasons, the flagged FOVs, a
summary of the placement (its FOVs by status, its links, the tables' SHA-256)
and how to approve it. It makes no report: the `report` task makes the full
report once, after the export. The task exits 4, the run fails, and
`verify --final` names the export held. Nextflow does not retry it; Snakemake
retries it only with `--retries`, and it holds again each time.

To review the placement, open it with
`load_placement("<output root>/stitching/<sample>")`. Inspect each FOV's status
and flags in `result.fovs`, the links in `result.links`, and
`result.record["review"]`. Record the review with
`approve_placement(result, reviewer=..., note=...)` and save it into the same
folder with `save_placement`. The approval goes to `approvals.json` in that
folder, keyed by the SHA-256 of the two tables it approves; `placement.json`,
`fovs.csv` and `links.csv` keep their bytes and modification times. Then run
the workflow again, the same way on both engines: Nextflow with `-resume`, or
Snakemake. Neither reruns `stitch`, whose outputs are unchanged; the export,
the report and the comparison run, and the held page is removed.
`verify --final` then passes. `verify` reads `approvals.json`: it names a held
export whose placement is now approved, and finds a finished export not
complete when its placement needs review and `approvals.json` holds no approval
of its tables.

An approval holds only for the tables it approved. A later rerun of `stitch`
keeps `approvals.json`; if it places the FOVs as before the approval still
counts, and if it places them differently the export is held again. Remove
`approvals.json` to withdraw every approval. The engine tests run this route
with a `stitch` task.

## What you get

Under the output root, each task writes the files the
{doc}`reference <reference>` lists for its kind, for example
`fovs/<fov>/molecules.csv` and `fovs/<fov>/run.json` for a `fov` task, and
`export/<sample>/<route>/` for an `export` task, whose route is `per_fov`,
`after_fusion` or `molecules_only`. The report of a sample is in
`report/<sample>/<route>/` (`report/<sample>/no_export/` without an export), and
the comparison page `report/index.html` lists every report the output root holds,
one row per sample and route.

Each route has its own export and report folders and task records, so one output
root holds several routes. Run the configuration of the first route, then plan and
run the configuration of the second into the same output root: its FOV-level and
placement tasks are complete already, so only its export, report and comparison
run, and no file of the first route changes. Copying a route's outputs aside is not
needed. The second plan's manifest names both routes, the first as a route found in
the output root, which no engine runs, and `verify --final` lists both: it checks the
first route's export and report records too (their outputs at their recorded size,
and with `--hash` their SHA-256), and fails, naming the file, when one of them is
missing or changed.

The workflow's own files are in `workflow/`:

* `resolved.json` and `manifest.json`, written by the plan;
* `tasks/<task>.json`, the record of each task that succeeded, with its outputs
  and their SHA-256, the execution context and the time and memory it took;
  `tasks/<task>.failed.json` keeps the record of a failed attempt;
* `logs/<task>.log`, the log of each task;
* `verify.json`, written by `verify`;
* `exports/<sample>/<route>.json`, a copy of each sample's export record, which lets a
  rerun of `export` remove the files the earlier export wrote;
* `held/<sample>/<route>.html`, the held page of an export held for review, until the
  export runs again.

## Main options

* The dataset keys at the top level: the roots and identifiers, `n_rounds`,
  `ref_round`, `seq_channel_order`, `reference_stains`, `additional_round`,
  the FOV list (`fov_list`), `selected_fovs`, the codebook and the image layout.
* `processing`: the per-FOV pipeline, with `mode` `direct` (one task per FOV) or
  `tiled` (tiles of a FOV), the preprocessing and registration steps and the
  checkpoint stages.
* `deconvolution`: a deconvolved copy of the raw images, made by deconwolf before
  the FOV tasks ({ref}`below <workflow-deconvolution>`).
* `morphology`, `segmentation` and `assignment`: the images for segmentation,
  the segmentation runs and the assignment of molecules to cells.
* `stitching`, `assembly`, `export` and `report`: the sample outputs.
* `execution`: each task kind's threads, memory, runtime and GPU, the image, the
  Fiji path, a verified resume of `fov` tasks (`resume_from_checkpoint`),
  whether a new package version reruns finished tasks (`code_version`) and the
  codec of the kept images (`image_compression`: `zlib`, the default, or
  `none`), which reruns no task and refuses no resume.

The {doc}`reference <reference>` lists every key with its default, and the
{doc}`migration guide <../migration>` maps the keys of earlier configurations.

(workflow-spot-finding-input)=
### Spot finding from a saved set

The stage `registered` holds each round after its preprocessing, registration
and post-registration steps (reconstruction, for example): the stack that spot
finding reads. To try other spot-finding, readout or decoding settings on the
same FOVs without running preprocessing and registration again, save that stage
once and point later runs at it:

1. Run the workflow with `registered` among the checkpoint stages, for example
   `checkpoints: [registered, candidates, pre_qc]`. Its output root is the
   saved set; each FOV's stage is under
   `fovs/<fov>/checkpoints/registered/`, and `fovs/<fov>/run.json` records the
   package version and the steps that made it.
2. In a later configuration, with its own `output_id`, keep the processing
   block's `load_raw_images`, `preprocessing` and `registration` as they were,
   change spot finding or the steps after it, and name the set:

   ```yaml
   processing:
     spot_finding_input: /path/to/<dataset_id>/<output_id of the saved set>
     checkpoints: [candidates, pre_qc]
   ```

Each `fov` task then declares the set's `run.json` and registered stage record
of its FOV in place of the raw images, loads the stage (checking every image
against the SHA-256 the stage record lists) and runs spot finding, extraction,
decoding, scoring, deduplication and filtering. Its `run.json` records this
run's whole processing configuration, the set's inputs and steps, and a
`spot_finding_input` section naming the set's `run.json`, its registered stage
record with their SHA-256 and the package version that made the set.

The task refuses a set (exit 2, naming every field) whose `run.json` does not
fit: another dataset, sample or FOV, a run that did not succeed, no registered
stage, a package version that cannot be established or a tree with uncommitted
changes, other round roles or channels, or other `load_raw_images`,
`rotate_angle`, `preprocessing` or `registration` settings, and a supplied
statistics file with another SHA-256. It reads a set that another package
version made: a change that leaves preprocessing and registration output
bit-identical keeps the set valid. A change to that output makes the set stale,
which the task cannot see from the version, so make the set again at the new
version. A set is a run's output root: keep it as long as runs read it, and
copy it with its `run.json` files.

A set of either codec is read as it is: one written with
`execution.image_compression: zlib` (the default), one written
uncompressed, with `none` or by an earlier package version. The codec is not
compared. The run that reads a set writes its own files with its own setting.
Do not recompress a set in place: its stage record and `run.json` hold the
SHA-256 of every file, and the task refuses a set whose files differ from them.
To get a compressed set, make the set again.
`backend: matlab` runs the {doc}`MATLAB legacy workflow <matlab-legacy>` instead.

(workflow-deconvolution)=
### Deconvolution

Raw confocal images can be deconvolved by the workflow itself, as a stage before
the FOV tasks, in place of a deconvolution by another program before the run. A
`deconvolution` block makes the plan add one `psf` task per run and one
`deconvolve` task per FOV and deconvolved round. Each runs
[deconwolf](https://github.com/elgw/deconwolf) v0.4.5 on every channel of its round
and writes a deconvolved copy of the round's folder. The `summarize`, `fov`,
`tile_global` and `morphology` tasks then read the copy in place of the raw images.
A dataset without the block runs no stage, and its plan does not change.

```yaml
deconvolution:
  rounds: null                # the sequencing rounds; or names, and reference_stain
  iterations: 5               # required: chosen per dataset ("Preparing a dataset")
  psf:
    na: 1.30
    immersion_ri: 1.518
    sample_ri: 1.338
    pinhole_radius_nm: 307
    channels:
      ch00: {excitation_nm: 499, emission_nm: 529}
      ch01: {excitation_nm: 598, emission_nm: 634}
```

The optics above are an example (9735seqA1's Leica SP8); they have no default.
The block needs `voxel_size_xy` and `voxel_size_z`. A generated PSF needs the
`deconvolution` extra (psfmodels, `uv sync --extra deconvolution`); with
`psf.source: file` the PSFs are supplied files and the extra is not needed.

* **The copy.** By default it lies under the output root, in
  `deconvolved/<sample_id>/dw-<copy id>/`: the input's folder layout with one
  `<label>.tif` per channel, its PSFs in `psf/`, `copy.json`, and one record per
  FOV and round in `records/<round>/<fov>/deconvolution.json`. `copy_root: beside`
  puts it beside the dataset's folder (`<dataset_id>-dw/`), and an absolute path
  under that path, where later output roots reuse it. The copy id hashes every
  setting of the block but `copy_root`, the voxel size and any supplied PSF's
  content, so another setting writes another copy and never changes this one.
* **The values.** By default each file is uint8, as Huygens' exports are: one
  stretch factor per FOV and round, f = 255 / the round's maximum after
  deconvolution, which the record keeps. The FOV tasks read the values as stored;
  each FOV has its own factor, so the plan warns when such a copy feeds a two-pass
  statistic (`fit: supplied`) or a `global` or `absolute` spot threshold.
  `dtype: uint16` stores rint(value × scale) instead, with one `scale` (64 by
  default, or one per round and label) for every FOV. The files are TIFF with the
  codec of `execution.image_compression` (zlib by default).
* **Reuse.** A `deconvolve` task whose round is already in the copy reuses it when
  the raw files, the PSFs and the outputs have the SHA-256 that its record holds,
  and runs no deconwolf, under Snakemake and Nextflow alike; the copy's files keep
  their bytes and modification times. A changed raw file redoes its round only. A
  FOV task checks that the record names every copy file it reads, with that
  file's SHA-256, and names the copy in its `run.json` (`deconvolution`).
* **CPU or GPU.** A task runs on the CPU with 4 threads by default.
  `execution.tasks.deconvolve.gpu: true` runs deconwolf with `--gpu` on one GPU;
  the device changes no hash and reruns nothing. `deconvolution.tile_size` splits a
  volume into tiles for a smaller GPU; it is in the hash, because tiles change
  values near their seams.
* **FFTW wisdom.** A CPU task plans its FFTs with `--noplan` unless
  `execution.deconwolf_wisdom` names a store that holds a plan for its key (image
  and PSF shape, threads, CPU model, FFTW and deconwolf). Make one with
  `starfinder deconvolve wisdom --store <dir> --shape <z y x> --psf-shape <z y x>
  --threads <n>` once per node type, shape and thread count; tasks only read the
  store. The {doc}`UGER guide <broad-uger>` gives the resources.

## Limits

* Only Snakemake runs Fiji, so the BigStitcher placement's Fiji task needs
  Snakemake; Nextflow refuses it unless `stitching.dataset_xml` supplies
  BigStitcher's result.
* A segmentation, the export's segmentation, or a Spotiflow or Piscis spot finding
  (`processing.spot_finding.device`) with `device: cuda` makes a GPU task, which
  needs a GPU where it runs. The {doc}`UGER guide <broad-uger>`
  describes how the cluster profiles place GPU jobs.
* Nextflow's local executor refuses a task that asks for more CPUs than the
  machine has; lower `execution.tasks.<kind>.threads` (the examples set one
  thread per `fov` task).
* With retries set, Snakemake retries a refused task (exit 2 or 3) or a held
  export (exit 4) like a failed one; Nextflow retries none of them.
* Under Snakemake, `--cores N` limits the threads the running tasks declare
  (`execution.tasks.<kind>.threads`), not their number. From Snakemake 9.27.1,
  `--cores 1` runs one single-thread task at a time and `--cores 2` runs two; Snakemake
  9.15.0 and earlier could start two tasks together under `--cores 1` after a lone
  ready job finished. Memory is not limited: give `--resources mem_mb=<n>` where a
  task's memory needs a bound.
* Under Snakemake, an output whose size changed is not rerun. Its consumers
  rerun on the changed file when its modification time is newer, and the run
  exits 0. Only `workflow verify --final` reports it.
* The MATLAB workflow is frozen: the plan refuses `backend: matlab`, and the
  Snakefile runs the legacy rules for it. It refuses a `deconvolution` block too.
* A deconvolved copy outside the output root (`copy_root: beside` or a path) is
  not one of Snakemake's outputs, so that Snakemake neither removes nor touches
  the files that another output root reuses. Snakemake does not rerun a task for a
  file of that copy removed or changed after the task ran: the FOV task that reads
  it fails, naming the file. Remove the output root's record of that round's
  `deconvolve` task (`workflow/tasks/deconvolve.<sample>.<fov>.<round>.json`) to
  redo the round.
* `processing.spot_finding_input` works in mode `direct` only, and a run that
  reads a set writes no `registered` stage of its own, so its segmentation
  cannot read a registered sequencing round. The package version of a set is
  recorded, not compared: whether a newer version changed preprocessing or
  registration output is for its user to know.

## How it is checked

Contract tests check the plan's refusals, the manifest and the task records;
engine tests run the same plans under Snakemake and Nextflow, including resumes,
failures and reruns. A golden comparison runs the workflow on the synthetic
fixture of the Quick start under Snakemake and compares its outputs with those of
the earlier Snakemake workflow, allowing only the recorded exceptions; an engine
comparison checks that Nextflow writes the same outputs. The Quick start above
runs on that fixture as written.

The workflow follows the
[workflow contract](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/orchestration.md),
the [engine record](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/orchestration-engines.md),
the [tiled execution record](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/orchestration-tiled.md)
and the [validation design](https://github.com/wanglab-broad/starfinder/blob/dev/design/validation/orchestration.md).
