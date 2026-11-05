# Workflow

:::{container} sf-lead
Run STARfinder over every FOV and sample of a dataset. One command plans the run
from a configuration file; Snakemake or Nextflow then runs each planned task as a
job, on a laptop or on a cluster.
:::

```{toctree}
:hidden:
:maxdepth: 2

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

For your own data, edit a copy of the full configuration or of the
{download}`minimal configuration <../examples/workflow-minimal.yaml>`, which runs
the per-FOV processing alone; the
{download}`recipe 2 example <../examples/workflow-recipe-2.yaml>` shows an explicit
preprocessing recipe.

## What it does

**The plan.** `starfinder workflow plan --config <file>` reads one configuration
file, checks it and refuses, before any task runs, a configuration it cannot run,
naming the key. It reads `fovs.csv` under the input folder
(`<root_input_path>/<dataset_id>/<sample_id>/`), looks up every input image and
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
The `export` task then stops. Once every other task the report needs has
succeeded, it makes the sample report from the FOVs' records and
`placement.json`, and the report's Placement section shows the review's reasons,
the flagged FOVs and any approvals. The task exits 4, the run fails, and
`verify --final` names the export held. Nextflow does not retry it; Snakemake
retries it only with `--retries`, and it holds again each time.

With an `assignment` block and an `export.route` other than `per_fov`, the
report also needs the `assign` tasks, which the export does not. If one of them
has not succeeded when the export is held, no report is made, and the export's
log names the tasks the report waits for. The review below does not need the
report, and the run after the approval makes it.

To review the placement, open it with
`load_placement("<output root>/stitching/<sample>")`. Inspect each FOV's status
and flags in `result.fovs`, the links in `result.links`, and
`result.record["review"]`. Record the review with
`approve_placement(result, reviewer=..., note=...)` and save it into the same
folder with `save_placement`. Then run the workflow again.

* **Nextflow** with `-resume` reruns `stitch`, because `placement.json` changed.
  The rerun keeps the approval, because its tables are unchanged. Then it runs
  the export, the report and the comparison.
* **Snakemake** removes a task's outputs before it reruns it, which would remove
  the approval. So first run the task by its command, where the tasks run (on
  the cluster, in the image, as `verify` is):
  `starfinder workflow stitch --manifest <output root>/workflow/manifest.json --task stitch/<sample>`,
  or for a BigStitcher placement `starfinder workflow bigstitcher-import` with
  `--task bigstitcher_import/<sample>`.
  It keeps the approval and records the task again. Then run Snakemake.

In both, `verify --final` then passes. An approval holds only for the tables it
approved: if the rerun places the FOVs differently, the export is held again.
The engine tests run this route with a `stitch` task.

## What you get

Under the output root, each task writes the files the
{doc}`reference <reference>` lists for its kind, for example
`fovs/<fov>/molecules.csv` and `fovs/<fov>/run.json` for a `fov` task, and
`export/<sample>/` for an `export` task. The workflow's own files are in
`workflow/`:

* `resolved.json` and `manifest.json`, written by the plan;
* `tasks/<task>.json`, the record of each task that succeeded, with its outputs
  and their SHA-256, the execution context and the time and memory it took;
  `tasks/<task>.failed.json` keeps the record of a failed attempt;
* `logs/<task>.log`, the log of each task;
* `verify.json`, written by `verify`;
* `exports/<sample>.json`, a copy of each sample's export record, which lets a
  rerun of `export` remove the files the earlier export wrote.

## Main options

* The dataset keys at the top level: the roots and identifiers, `n_rounds`,
  `ref_round`, `seq_channel_order`, `reference_stains`, `additional_round`,
  `selected_fovs`, the codebook and the image layout.
* `processing`: the per-FOV pipeline, with `mode` `direct` (one task per FOV) or
  `tiled` (tiles of a FOV), `recipe: deep` and the checkpoint stages.
* `morphology`, `segmentation` and `assignment`: the images for segmentation,
  the segmentation runs and the assignment of molecules to cells.
* `stitching`, `assembly`, `export` and `report`: the sample outputs.
* `execution`: each task kind's threads, memory, runtime and GPU, the image, the
  Fiji path, a verified resume of `fov` tasks (`resume_from_checkpoint`) and
  whether a new package version reruns finished tasks (`code_version`).

The {doc}`reference <reference>` lists every key with its default, and the
{doc}`migration guide <../migration>` maps the keys of earlier configurations.
`backend: matlab` runs the {doc}`MATLAB legacy workflow <matlab-legacy>` instead.

## Limits

* Only Snakemake runs Fiji, so the BigStitcher placement's Fiji task needs
  Snakemake; Nextflow refuses it unless `stitching.dataset_xml` supplies
  BigStitcher's result.
* A segmentation, or the export's segmentation, with `device: cuda` makes a GPU
  task, which needs a GPU where it runs. The {doc}`UGER guide <broad-uger>`
  describes how the cluster profiles place GPU jobs.
* Nextflow's local executor refuses a task that asks for more CPUs than the
  machine has; lower `execution.tasks.<kind>.threads` (the examples set one
  thread per `fov` task).
* With retries set, Snakemake retries a refused task (exit 2 or 3) or a held
  export (exit 4) like a failed one; Nextflow retries none of them.
* Under Snakemake, an output whose size changed is not rerun. Its consumers
  rerun on the changed file when its modification time is newer, and the run
  exits 0. Only `workflow verify --final` reports it.
* The MATLAB workflow is frozen: the plan refuses `backend: matlab`, and the
  Snakefile runs the legacy rules for it.

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
