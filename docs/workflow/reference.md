# Workflow reference

Look up the workflow's commands, task kinds, configuration keys and profiles here.
The {doc}`workflow guide <index>` shows how to run them. The tables are generated
from the package when the site is built.

(workflow-commands)=
## Commands

`starfinder workflow` has the plan, `verify` and one command per task kind. An
engine runs a task's command as the manifest gives it,
`starfinder workflow <command> --manifest <manifest> --task <id>`, with the
execution context appended; the context is recorded in the task record and never
changes what a task does.

```{workflow-commands}
--manifest: The plan's workflow/manifest.json
--engine: The engine that runs the task, recorded in the task record
--engine-version: The engine's version, recorded in the task record
--job-id: The scheduler's job id, else the engine's (Snakemake's job number, Nextflow's task hash), recorded in the task record
--attempt: The engine's attempt number, recorded in the task record
```

(workflow-exit-status)=
## Exit status

| Status | Meaning | Snakemake | Nextflow |
| --- | --- | --- | --- |
| 0 | The task succeeded; its record was written last | Done | Done |
| 1 | The task failed; its declared outputs were removed and `<task>.failed.json` written | Retried up to `--retries` | Retried up to `--max_retries` (default 2); `--max_retries 0` runs the task once and finishes the run |
| 2 | Refused before any work; nothing was written | Retried like 1 | Not retried; the run finishes |
| 3 | A verified resume was refused | Retried like 1 | Not retried; the run finishes |
| 4 | The export was held for review: its placement needs review and holds no approval of its tables; it failed as 1 does, its failed record says `held`, and it made the sample report when every other task the report needs had succeeded ({ref}`the route <workflow-placement-review>`) | Retried like 1 | Not retried; the run finishes |

`plan` exits 2 when it refuses a configuration and `verify` exits 1 with `--final`
when a task is not complete. Each retry multiplies a task's memory and runtime by
the attempt.

(workflow-task-kinds)=
## Task kinds

Each kind's command, what one task covers, the configuration blocks its command
reads, its default threads and its declared outputs, relative to the output
root. The outputs are those that the plan declares for the
{download}`full example <../examples/workflow-full.yaml>` and for the same
configuration in tiled mode, with a BigStitcher placement and segmentation runs by
a method; a configuration that writes less declares less (no checkpoint stages,
no previews, no nuclei, an imported segmentation without `input.ome.tif`).
`{fov}`, `{sample}`, `{step}` and `{tile}` stand for the task's identity, `<run>`,
`<name>` and `<round>` for a segmentation run, an assignment name and another
round. A temporary output is removed once the tasks that read it have succeeded.

```{workflow-tasks}
```

Besides its declared outputs, every task writes `workflow/logs/<task>.log` and,
when it succeeds, `workflow/tasks/<task>.json`, where `<task>` is the task id
with `/` written `.` (`fov.sample.FOV_001`). A failed attempt writes
`workflow/tasks/<task>.failed.json`; `verify` writes `workflow/verify.json`; the
`export` command keeps a copy of each sample's `export.json` in
`workflow/exports/<sample>.json`, so that a rerun can remove the files of the
earlier export after an engine removed the declared record.

(workflow-configuration-keys)=
## Configuration keys

A configuration holds the dataset keys at the top level, one block per step and an
`execution` section; a present block enables its step. "Plan default" is the value
that the plan writes into `workflow/resolved.json` when the key is absent; a dash
means that the plan writes none and the API object named in the description
applies its own default, which its API page lists.

(workflow-keys-dataset)=
### Dataset keys

```{workflow-keys} dataset
backend: python; matlab selects the frozen MATLAB workflow, which the Snakefile runs and the plan refuses.
root_input_path: The folder that holds <dataset_id>/<sample_id>/, the input folder.
dataset_id: The dataset's folder under both roots.
sample_id: The input folder's name under <root_input_path>/<dataset_id>/.
root_output_path: The folder that holds <dataset_id>/<output_id>/, the output root.
output_id: The output root's name under <root_output_path>/<dataset_id>/.
selected_fovs: The FOV ids of the input folder's fovs.csv to run; null runs every FOV.
n_rounds: The number of sequencing rounds, named round1 to round<n_rounds>.
ref_round: The sequencing round the others are registered to.
ref_channel: The shared stain's name in the other rounds, and with several reference_stains the one of the reference round, for the morphology task.
reference_stains: The reference round's stain files besides the sequencing channels, a list of {channel, name}.
seq_channel_order: The channel labels of a sequencing round in colour order; each label matches one file *<label>*.tif of the round's folder.
additional_round: The other rounds, each a round_name and its channel_order, which the morphology block registers.
rotate_angle: The rotation of each FOV in degrees; 0 is no rotation.
voxel_size_xy: The voxel size in Y and X in micrometres; with voxel_size_z, fills unknown raw-image spacing at load in direct and tiled modes, preserving all other metadata and any known spacing, and declares the grid of saved reference files without stored metadata.
voxel_size_z: The voxel size in Z in micrometres; both voxel sizes must be set to calibrate raw images, giving spacing (voxel_size_z, voxel_size_xy, voxel_size_xy) in micrometer before rotation.
reference_image: The file pattern, with {fov_id}, of each FOV's saved reference image, absolute or relative to the output root.
reference_image_import: The file pattern, with {fov_id}, of a reference image another tool wrote; the processing tasks then write none. It excludes reference_image.
image_dir_template: The folder of one round of one FOV under the input folder.
readout_mode: multiplexed reads a codebook; direct reads a direct panel.
codebook: The codebook of readout_mode multiplexed, absolute or relative to the input folder; null is genes.csv.
direct_panel: The direct panel of readout_mode direct, absolute or relative to the input folder; null is genes.csv.
```

(workflow-keys-processing)=
### `processing`

The per-FOV pipeline of the `fov` tasks, or in tiled mode of the `tile_global`,
`tile` and `tile_merge` tasks. The {doc}`preprocessing <../preprocessing-algorithms>`,
{doc}`registration <../registration-algorithms>`,
{doc}`spot finding <../spot-finding-algorithms>` and
{doc}`readout <../readout-algorithms>` references list the methods and their fields.

```{workflow-keys} processing
mode: direct runs one fov task per FOV; tiled runs a FOV in tiles (processing.tiled).
tiled: The tiled settings below, read in mode tiled only.
checkpoints: The checkpoint stages the fov task writes; the plan adds registered when a segmentation input reads a sequencing round.
streaming: true runs FOV.run in streaming execution, false in batch.
device: cpu, the device of registration and readout; spot_finding.device sets the device of spot finding.
load_codebook: The codebook settings: split_index and encoding.
load_raw_images: The image-loading settings: subdir.
preprocessing: The preprocessing recipe: steps, post_registration, extraction_source, registration_source and supplied_statistics.
registration: The registration recipe: steps, signal, warp, qc and reference_round.
reference_image_view: The saved reference image: mode max (the channel maximum) or channel with channel, and projection.
spot_finding: Present, spots are found: method and its config's fields, channel_overrides, rounds and device, cpu (default) or cuda, which spotiflow and piscis accept and which makes each direct fov task a GPU task.
reads_extraction: Present, reads are extracted: voxel_size, the Z, Y and X half-widths in voxels, and background.
reads_filtration: Present, reads are filtered and decoded: end_base, start_base, n_barcode_segments and split_index.
decoding: The decoder: method wta, codebook_aware or direct, and its config's fields.
scoring: The read score's settings; false turns scoring off.
deduplication: Present, duplicate reads are removed: distance_voxels and compatibility.
```

```{workflow-keys} processing.tiled
```

For a full-size Position400 FOV in tiled mode, the whole `tile_merge` task measured 6.37–7.08 GiB RSS, which its table-processing code does not explain; the 8,000 MB default left about 397 MB above U4's maximum observed peak.

(workflow-keys-morphology)=
### `morphology`

```{workflow-keys} morphology
rounds: The other rounds to register on the reference stain.
recipe: A registration recipe; null is one translation step on the shared stain.
channel_images: null, or {projection: null or max} to write images/<round>/<label>/<fov>.tif too.
```

(workflow-keys-segmentation)=
### `segmentation`

```{workflow-keys} segmentation
device: cpu, or cuda, which makes each segment task a GPU task.
runs: The segmentation runs, each with name, target, inputs, seeds, projection, operations and a method with its config's fields.
```

The {doc}`segmentation reference <../segmentation-algorithms>` lists the methods,
the run's fields and the preparation functions.

(workflow-keys-assignment)=
### `assignment`

```{workflow-keys} assignment
previews: true draws the assignment figures into fovs/<fov>/previews/.
```

(workflow-keys-stitching)=
### `stitching`

```{workflow-keys} stitching
method: A stitching method, run by the stitch task, or bigstitcher, run by the BigStitcher tasks.
input: The images placed: reference_merged, reference_stain or file:<pattern>.
positions: The nominal positions: source stage_file, regular_grid or table, with that source's settings.
config: The fields of the method's config, MultiviewStitcherConfig or BigStitcherConfig.
review: The fields of ReviewPolicy.
fallback: null, or nominal to place a FOV that stitching cannot link at its nominal position, flagged fallback.
bigstitcher_save_dir: A local folder for BigStitcher's tiles, macro and result; required with bigstitcher.
dataset_xml: A supplied BigStitcher dataset.xml per sample, a pattern with {sample}; the macro and Fiji tasks are then skipped.
view_map: The view map of the supplied dataset.xml, a pattern with {sample}.
```

The {doc}`stitching reference <../stitching-algorithms>` lists the methods and
their config fields.

(workflow-keys-assembly)=
### `assembly`

```{workflow-keys} assembly
assignment: The name of each FOV's saved assignment to assemble; null assembles molecules only.
table_format: The format of the sample tables: csv or parquet.
```

(workflow-keys-export)=
### `export`

```{workflow-keys} export
previews: true draws the ownership figures into export/<sample>/previews/ on the route per FOV.
```

(workflow-keys-report)=
### `report`

The `report` task reports the assignment that `assembly.assignment` names. A configuration
without an `assembly` block reports the `name` of its `assignment` block, which the
`assign` tasks saved for every FOV, so a run with `segmentation`, `assignment` and
`report` blocks shows its cells; without either block the report is made without cells.
The task reads the `report`, `assembly` and `assignment` blocks.

```{workflow-keys} report
```

(workflow-keys-execution)=
### `execution`

```{workflow-keys} execution
defaults: The resources of every task, below.
tasks: Resources per task kind, over the defaults and each kind's threads.
container: The image of every task, where the engine runs tasks in an image (one value for the run, none per task kind): under Snakemake's UGER profile an OCI manifest digest, under Nextflow the value of its container directive (on sge the SIF's path),; with `--container` it is the only way to set the tasks' image under Nextflow, which refuses to start with a container engine enabled and neither set; null uses the engine's default.
fiji: The Fiji executable of the bigstitcher_engine task.
resume_from_checkpoint: true lets a fov task resume from its checkpoints after it checks run.json; a mismatch exits 3.
code_version: keep, or rerun to rerun finished tasks after the package version changes.
log_level: The level of the task logs.
```

```{workflow-keys} execution.defaults
threads: The threads of a task; the fov, tile_global, tile and bigstitcher_engine kinds default to 4.
memory_mb: The memory of a task in MB.
runtime_min: The runtime of a task in minutes.
gpu: true requests a GPU.
```

(workflow-left-keys)=
## Keys that left the Python configuration

The plan refuses these keys, naming where each went. It also refuses the `run`
flags of the processing sub-blocks (a present block enables its step),
`assembly.previews` and `assignment.checkpoints`. The
{doc}`migration guide <../migration>` maps earlier configurations.

```{workflow-left-keys}
```

(workflow-profiles)=
## Profiles

**Snakemake on Broad UGER** (`profile/broad-uger/config.yaml`), passed with
`--profile`; the {doc}`UGER guide <broad-uger>` describes its scripts:

```{workflow-profile} profile/broad-uger
```

**Nextflow** (`nextflow/nextflow.config`), selected with `-profile`:

| Profile | Settings |
| --- | --- |
| `local` | The local executor, one task at a time |
| `sge` | The SGE executor with per-slot `h_vmem`, and the `smp` parallel environment for a task of more than one CPU; Apptainer with `/stanley` bound; `--container` required, `--weights_dir` optional; GPU tasks request RedHat8 GPU nodes and add `--nv` |
| `google-batch` | Google Batch with spot machines and a work folder in a bucket; the project, region, bucket, run and image digests are placeholders to fill in |

Every Nextflow profile takes `--config` (the configuration file), `--container`
(the image of every task and of the plan and verification processes; with `execution.container`, the only way to set an image, see
{ref}`the Nextflow guide <workflow-nextflow-images>`) and `--max_retries` (default 2); exit statuses 2, 3 and 4 finish
the run without a retry, and so does every failure under `--max_retries 0`. `--weights_dir` (unset by default) is read by the `sge`
profile only: when given, it adds `--env STARFINDER_WEIGHTS_DIR=<folder>` to the
`apptainer` command of every task, CPU and GPU, and the folder must be on `/stanley`.

The {doc}`MATLAB legacy workflow <matlab-legacy>` has its own keys and profile.
