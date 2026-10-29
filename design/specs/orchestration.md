# Workflow orchestration contract: the CLI, the plan and manifest, the configuration, completion, resume and logging

% Status: Proposed (W-643, 2026-10-29, at 56c35b8)

Status: Proposed. This record is reviewed at the specification gate of workflow orchestration
(W-646) before any implementation; approval publishes nothing.

This page specifies the engine-neutral contract that replaces the code that talks to Snakemake
today: a command group of the package, a plan command that writes a resolved configuration and a
manifest of every task, the Python configuration in its own shape, the rule by which a task has
succeeded, checkpoints and verified resume, and the per-task log and records. Snakemake and
Nextflow are both designed against it by the engine record (W-645); native tiled execution is the
tiled record (W-644). It builds on the baseline [orchestration](../baselines/orchestration.md)
(W-642), whose classification, configuration inventory and discrepancies it cites by their labels.
Statements about today's behaviour cite the file and function at `56c35b8`, the base of the batch;
paths are relative to the repository root.

% W-643 (S2 of §2.13); the batch runner/s213-spec-20261029 starts from dev at 56c35b8

The decisions this page implements (W-152, "§2.13 planning decisions (2026-10-27)", on top of the
boundary of 2026-09-28: logic that can change a result belongs in the Python API; scheduling,
placement, order and retries belong to the engine; `.smk` and `.nf` files carry no defaults,
parameter translation or result-affecting computation):

| Decision | What this page specifies for it |
| --- | --- |
| D1 | The task kinds of the whole Python workflow (per-FOV processing, the two-pass statistics, morphology, segment, assign, stitch, export, report) |
| D2 | The CLI, one command per API entry; the plan command, the resolved configuration and its hash, the manifest; engines read only the manifest and call the CLI |
| D3 | The BigStitcher macro and import commands and the Fiji engine task of the manifest |
| D4 | The configuration shape, one block per step and an execution section; the task graph from data dependencies; the migration table |
| D5 | The tiled task kinds (their algorithm is the tiled record's) |
| D6 | The morphology, segment and assign task kinds |
| D7 | The default backend `python`; the refusal of a configuration without `backend` that holds MATLAB-dialect keys; the MATLAB legacy boundary |
| D8 | Checkpoint defaults, verified resume, the success rule, atomic publication, one log per task and the `run.json` additions |

The decisions taken at the W-642 checkpoint (Jiahao, 2026-10-29) that this page carries:

| Point | Decision |
| --- | --- |
| OC2 | `tests/minimal_config.yaml` stays as the schema fixture of the frozen MATLAB dialect; its description as a starting template is corrected ([changed statements](#changed-statements-of-the-existing-records)) |
| OC3 | In the new shape `rotate_angle: 0` means no rotation and the step is skipped; frame ids lose `/rotate:0` ([the golden comparison's exception](#the-exception-of-the-golden-comparison)) |
| OC4 | The plan and the loaders name frames and inputs relative to the input root, as `run.json` version 2 records them |
| OC5 | `codebook`, `direct_panel` and `image_dir_template` move to the schema's Python-only clause, and the `common.smk` refusal goes |
| P1 | The rest of the golden baseline is recorded first ([below](#the-rest-of-the-golden-baseline)) |
| P5 | Previews are options of the assign and export commands, off by default |

Evidence, kept with the run records outside the repository and cited by path: the two golden runs
of invocation D and the comparison across environments (`results/golden/`), the prototype of the
plan command and the worked example (`scripts/plan_prototype.py`, `results/example/`), the
comparison with today's planners (`results/example/planner-comparison.json`), the checks of the
migration and traceability tables (`results/migration/`, `results/traceability/`), the worker
notes (`notes.md`) and the run's manifest (`results/manifest.json`).

% Run directory: /home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/runs/W-643/20261029T175804Z-9e040431

## The rest of the golden baseline

W-642 pinned the per-FOV core, morphology, assignment from imported labels and the BigStitcher
macro; the stitching with `multiview_stitcher`, the export and the report after the export could
not run in the Snakemake environment of W-641 (P1). They are recorded here, at `56c35b8`, with
`src/python/scripts/workflow_golden.py` and its new invocation D, in the golden scratch
environment `w642-golden-20261029T1750Z` (the project lock with the extras dev, ome,
local-registration, checkpoint, anndata, multiview-stitcher and spatialdata, and Snakemake 9.15.0
outside the lock; identity `runs/W-642/20261029T1750Z-golden-env/environment.json`). The
package of the worktree equals `56c35b8` (`git diff --quiet 56c35b8 -- src/python/starfinder`
exits 0).

% results/environment.json (package_equals_base, golden_environment.identity_sha256)

**Invocation D.** On the fixture of W-642 (`sample-tiny-v1`, seed 42), three Snakemake
invocations with target `all`, one per export route, each in its own output root:

* `golden-per-fov`: configuration B with the report block and `export: {route: per_fov,
  images: [reference_merged], fuse_labels: true, labels_scale_yx: 0.5}`; the truth labels are
  copied first as the legacy label files, as in B;
* `golden-molecules-only`: configuration A with the stitching block, `assembly: {assignment:
  null}` and `export: {images: [reference_merged]}`;
* `golden-after-fusion`: configuration A with the stitching block, `assembly: {assignment:
  default}` and an export block whose segmentation imports one cell run: the Z maximum of the
  per-FOV route's fused cell labels, written on the fused projection grid of that export.

The stitching block is B's (`multiview_stitcher` on the positions table) with `config:
{binning_zyx: [1, 2, 2], quality_threshold: -1.0}`, the setting of the package's stitching tests
(`test/test_stitching_reference_image.py`): the default binning of 4 in Z leaves one plane of the
five-plane FOVs, on which `multiview_stitcher`'s overlap is undefined (`linprog` refuses its
NaN; the first trial's Snakemake log, `results/trials/trial1-snakemake-D-per-fov.log`). The export fuses the reference image (`reference_merged`, a
declared output of `rsf_single_fov`): the default `reference_stain` reads the saved morphology
form, which no rule declares (`_export_sample_files`, `_snakemake.py:834`), so its order after
`nuclei_registration` would not be guaranteed. Every route writes its export, report and
comparison page; the parts written are `assemble`, `fuse` and, by route, `fuse_labels` (per FOV)
or `segment` (after fusion).

**The values.** The records, tables and images of `stitching/`, `export/` and `report/` of each
route, extracted as W-642 extracted A to C (`extract-export`); the per-FOV outputs come from the
configurations of A and B and are pinned by their values. Two trials in different folders showed
three more files whose bytes hold absolute paths: `placement.json` (the positions table's path and
each FOV's reference image as `file:<absolute path>`), `export.json` and `sample.h5ad` (its `uns`
holds the export's records). Their digests, and the digests that the export record computes over
them (`assembly_sha256`), are left out like W-642's varying files; their content is compared
through the normalized records. The placement record's absolute paths are a discrepancy of the
kind O12 describes: under OC4 the stitch task records them relative to the input and output roots.
The expected values are committed beside W-642's, in
`src/python/test/data/expected/test_workflow_orchestration_golden_export/`; W-642's folder is
unchanged.

**Two runs, and A to C across environments.** Each run generates the fixture, runs A, A2, B, C and
D in the golden environment and extracts both sets of values (about 7 minutes and a peak RSS of
about 360 MiB per run on one CPU). The comparison:

* **D, run 2 against run 1** (same environment, every class, the file digests included): no
  difference, over 218 values, 54 tables, the normalized records of the three routes and the
  integer images. A negative control that changes four values, each by more than its tolerance (a
  cell centroid of the per-FOV route's table by a relative 1e-6, the after-fusion route's retained
  molecule count by 1, one pixel of a fused projection by 1, one file digest), is reported four
  times and nothing else.
* **A to C, run 2 against run 1** (same environment): no difference, over 335 values and 29
  tables.
* **A to C across environments**: W-642's committed values (Snakemake environment `starfinder-v9`,
  NumPy 2.3.5, pandas 2.3.3) against run 1 (the golden environment, NumPy 2.2.6, pandas 3.0.0),
  with the tolerances of `docs/contributing.md` and without the same-environment file digests:
  every value agrees within its tolerance. Every value that differs at all is listed (8): the file digest of each FOV's
  `checkpoints/candidates.json` and the digest of that file that `run.json` records under
  `checkpoints.candidates.sha256`. A run of W-642's invocations in `starfinder-v9` shows why: the
  two `candidates.json` files differ in one leaf only, `detection_diagnostics.software.numpy`
  (`2.3.5`, `2.2.6`); `pre_qc.json` and the stage tables are identical. That digest is therefore an
  identity of the recording environment, like the file digests, and the comparison across
  environments skips it (`STAGE_RECORD_DIGEST` of `workflow_golden.py`); W-642's expected values
  are unchanged. Every float of the records and tables is bit-identical across the two
  environments on this fixture.

% results/golden/comparison.json (compare.rerun-2), results/golden/candidates-diagnosis.json (diagnose_candidates,
% after workflow_v9 in ~/miniforge3/envs/starfinder-v9), results/golden/run1, run2 (commands, configurations, logs,
% outputs.csv, jobs.json, expected/, expected-d/), results/golden/installed.json

## Terms

* **Task**: one call of one command of the CLI on one identity (a FOV, a FOV and a tile, a FOV and
  a segmentation run, a sample, a sample and a statistics step, or the run), or one engine-side tool
  run (Fiji). A task reads only its declared inputs and writes only its declared outputs.
* **Task kind**: the command a task calls: `summarize`, `merge`, `fov`, `tile_global`, `tile`,
  `tile_merge`, `morphology`, `segment`, `assign`, `bigstitcher_macro`, `bigstitcher_engine`,
  `bigstitcher_import`, `stitch`, `export`, `report`, `compare`.
* **Plan**: the command that runs once before fan-out and writes the resolved configuration and the
  manifest.
* **Manifest**: the file that lists every task of a run with its identity, dependencies, declared
  inputs and outputs, record, log, command and execution settings.
* **Resolved configuration**: the configuration with every default written out, in its canonical
  serialization; its **hash** identifies the result-affecting settings of a run.
* **Engine**: Snakemake or Nextflow; it reads the manifest, schedules its tasks, runs their commands
  and retries a failed task from scratch. It imports nothing from the package.

## The flow

1. The engine runs `starfinder workflow plan --config <file.yaml>` (how it runs before the
   Snakemake graph is built is the engine record's).
2. The plan validates every block, discovers the inputs, resolves the configuration, derives the
   task graph from data dependencies and writes `<output root>/workflow/resolved.json` and
   `<output root>/workflow/manifest.json`. A refused configuration exits 2 and writes nothing.
3. The engine reads the manifest, turns each task into a job with the task's `needs`, `inputs`,
   `outputs` and `execution`, and runs each task's `command` with its execution context.
4. Each command reads the manifest and the resolved configuration, runs one API entry, writes its
   declared outputs, its log and, last, its task record, and exits 0; or it exits non-zero and leaves
   no declared output behind.

The rule that bounds the engines: **no command reads an engine object** (no `snakemake` object,
no Nextflow variable), and no `.smk` or `.nf` file imports the package; an engine passes nothing to
a command but the manifest's path, the task id and its execution context. Today every Python rule
script hands its `snakemake` object to a runner of `starfinder._snakemake` (for example
`_run_workflow`, `_snakemake.py:28`, which reads `snakemake.config`, `.input`, `.output` and
`.wildcards`), and `common.smk` imports the package when the rule files are parsed
(`common.smk:212-214`, `:414-434`); both go.

## The CLI

The command group `starfinder workflow` is added to the package's console script `starfinder`
(`starfinder/__main__.py`, `main`), beside `synthetic`, `benchmark` and `weights`. It has one
command per API entry (D2). The commands live in a package module (working name
`starfinder.workflow`); each is a thin adapter that reads the manifest and the resolved
configuration, translates the block with the kept translators of `starfinder.dataset.workflow`
and calls the API entry. The API itself does not change: the seam kept is `WorkflowConfig`,
`from_workflow_config`, the block translators, `_rule_checkpoints`, `_layout`, `PipelineConfig`
(an operation set to `None` disables it) and the entries behind them.

**Common arguments.** Every task command takes the same arguments:

| Argument | Meaning |
| --- | --- |
| `--manifest PATH` | The manifest, which names the resolved configuration (path and hash), the roots and the task |
| `--task ID` | The task id in the manifest; its identity (FOV, tile, run, sample, step) comes from there |
| `--engine NAME` | `snakemake`, `nextflow` or `manual`: recorded in the execution context |
| `--job-id ID`, `--attempt N`, `--image REF` | The engine's job id, the attempt number (1 for the first) and the image reference by digest; recorded, never interpreted |

The command checks, before any work, that the task exists, that the resolved configuration's file
hashes to the manifest's `configuration.sha256`, that every task it `needs` has a task record with
status `succeeded`, and that every declared input exists. A failure of these checks exits 2 and
writes nothing.

(orchestration-commands)=

| Command | Task kind | API entry | Reads | Writes |
| --- | --- | --- | --- | --- |
| `plan` | none | input discovery, the block translators (validation) | the configuration file, `fovs.csv`, the input folders (names, sizes, modification times), the codebook, the supplied files | `workflow/resolved.json`, `workflow/manifest.json` |
| `summarize` | `summarize` | `summary_recipe`, `FOV.run` of the summary recipe, `summarize_histograms`, `save_histograms` | the FOV's raw images; the earlier supplied file when a later step is summarized | `statistics/<sample>/<step>/histograms/<fov>.npz` |
| `merge` | `merge` | `load_histograms`, `merge_histograms`, `fit_supplied_section`, `supplied_statistics`, `save_supplied_statistics` | the sample's histogram files; the earlier supplied file | `statistics/<sample>/<step>/supplied.json` |
| `fov` | `fov` | `FOV.run`, `FOV.save_reference_image`, `FOV.export_spots` | the FOV's raw images, the codebook, the supplied file | `fovs/<fov>/run.json`, the reference image, `fovs/<fov>/molecules.csv`, the checkpoint stages |
| `tile-global` | `tile_global` | the tiled record's global entry | the FOV's raw images, the codebook, the supplied file | the reference image, the chunked registered checkpoint and its record (W-644) |
| `tile` | `tile` | the tiled record's tile entry | the window and halo of the chunked checkpoint | the tile's record and molecules (W-644) |
| `tile-merge` | `tile_merge` | the tiled record's merge entry | the tiles' molecules and records | `fovs/<fov>/run.json`, `fovs/<fov>/molecules.csv` (as direct, D5) |
| `morphology` | `morphology` | `FOV.prepare_morphology` (`MorphologyConfig` with `rotate_angle`), `FOV.save_registration_record` | the reference round's stain files and the other rounds' raw images | `fovs/<fov>/nuclei_registration.json`, `fovs/<fov>/other_rounds/<name>/` per prepared image, optionally the channel images |
| `segment` | `segment` | `FOV.load_reference_image`, `FOV.load_registered_round`, `FOV.segment` with checkpoints, for one run of the plan | the reference image, the saved forms its inputs name, the seeds' run, an imported label file | `fovs/<fov>/segmentation/<run>/` |
| `assign` | `assign` | `load_molecules`, the saved segmentation runs, `assign_molecules`, `summarize_assignment`, `write_assignment` (as `_run_reads_assignment` of an assembly block does today) | `fovs/<fov>/molecules.csv`, the reference image, the codebook, the saved cell and nucleus runs | `fovs/<fov>/assignment/<name>/`, previews with `assignment.previews` |
| `bigstitcher-macro` | `bigstitcher_macro` | `export_bigstitcher` on the sample's stitching images and position table | the stitching images, the reference grids, the position file | `<save dir>/<sample>/BigStitcher_macro.ijm`, `view_map.csv`, `tiles/` |
| none (engine tool) | `bigstitcher_engine` | Fiji runs the macro (D3); the manifest gives the tool's command line | the macro, the tiles | `<save dir>/<sample>/dataset.xml` |
| `bigstitcher-import` | `bigstitcher_import` | `_check_bigstitcher_engine`, `import_bigstitcher`, `save_placement` | `dataset.xml` (the engine task's, or a supplied one), the view map, the macro, the reference grids, the position file | `stitching/<sample>/placement.json`, `fovs.csv`, `links.csv` |
| `stitch` | `stitch` | `Dataset.stitch` | the stitching images, the reference grids, the position file | `stitching/<sample>/placement.json`, `fovs.csv`, `links.csv` |
| `export` | `export` | `Dataset.export`; with `export.previews`, the ownership previews | the placement and, by route, the saved assignments or the molecule tables and reference images, the fusion's source images | `export/<sample>/export.json` (which lists the folder's other files), `export/<sample>/previews/` with `export.previews` |
| `report` | `report` | `Dataset.report` | the FOVs' `run.json` and checkpoint stages, the placement, `export.json`, the saved assignments | `report/<sample>/report.json` (which lists the folder's other files) |
| `compare` | `compare` | `Dataset.compare_samples` | every sample's `report.json` | `report/samples.csv`, `report/index.html` |

Every command also writes its log, `workflow/logs/<task>.log`, and on success its task record,
`workflow/tasks/<task>.json` ([logging and records](#logging-and-records)); `<task>` is the task
id with `/` written `.`.

**Exit codes.** The package's console script exits 2 for a usage or input error and 1 for a stage
failure (`starfinder/__main__.py`, `main`). The workflow commands keep that and add one code:

| Code | Meaning | Engine |
| --- | --- | --- |
| 0 | The task succeeded: the API call returned, every declared output exists and is recorded in the task record, which says `succeeded` | Done |
| 1 | The task failed: the API call raised, or the completion check found a declared output missing; the declared outputs this attempt wrote are removed | May retry from scratch (D8) |
| 2 | Refused before any work: usage, an unknown task, a resolved configuration whose hash differs from the manifest's, a `needs` task without a succeeded record, a missing declared input, or a refused configuration (the plan) | Does not retry: the same call refuses again |
| 3 | A resume refused: `resume_from_checkpoint` is on and the saved record does not match ([checkpoints and resume](#checkpoints-and-resume)) | Does not retry; the user decides |

### Choice: the name of the command group

| Option | Cost | Consequence |
| --- | --- | --- |
| A. `starfinder workflow <command>` (recommended) | One more group in `starfinder/__main__.py` | One console script; `starfinder --help` lists it beside `synthetic`, `benchmark` and `weights` |
| B. A second console script, `starfinder-workflow` | One more entry point in `pyproject.toml` | Two commands to install and find |
| C. `starfinder task <command>` | As A | A name that says less about what the group is for |

Recommendation: A.

### Choice: the arguments of a task command

| Option | Cost | Consequence |
| --- | --- | --- |
| A. `--manifest` and `--task` only (recommended) | The command reads the manifest each time (one JSON file) | One command line form for every kind; the engine copies the manifest's `command`; nothing about a task's identity, inputs or settings is on the command line, so an engine cannot pass a translated parameter |
| B. Explicit arguments per command (`--config`, `--output-root`, `--fov`, `--run`, ...) | Each command defines its own arguments; engines build each command line | The engine files restate the identity and paths of each task, which the boundary forbids them to compute |
| C. One dispatcher `starfinder workflow run-task` | One entry instead of one per API entry | Contradicts D2 (one command per API entry) |

Recommendation: A. A person runs one task by hand with the same two arguments.

### Choice: how the execution context reaches the command

| Option | Cost | Consequence |
| --- | --- | --- |
| A. Flags `--engine`, `--job-id`, `--attempt`, `--image` (recommended) | Each engine adds four arguments to the manifest's command | Visible in the engine's job log and in the task log; testable without an environment |
| B. Environment variables (`STARFINDER_TASK_*`) | None in the command line | Invisible in the command line; a missing variable is silent |
| C. A context file written by the engine | One more file per task | Another declared input the engine must write |

Recommendation: A. How each engine obtains its job id and attempt (Nextflow has `task.attempt`
and `task.hash`; Snakemake has the attempt of a job and its executor's job id) is the engine
record's.

### Choice: exit codes

| Option | Cost | Consequence |
| --- | --- | --- |
| A. 0, 1, 2 and 3 as above (recommended) | Engines map exit codes to retry or stop | A refused call and a refused resume are not retried; a failed task is |
| B. 0 and 1 only | Simplest | An engine retries a refusal that cannot succeed, three times on UGER today (`profile/broad-uger/config.yaml`) |

Recommendation: A.

## The plan command

`starfinder workflow plan --config <file.yaml> [--output-root PATH]` runs once before fan-out. It
replaces the parse-time validation of `common.smk`, its input enumeration and its target list
(`get_overall_output`, `common.smk:283`).

### Validation

The plan validates every block before it writes anything, with the translators that today run
when the rule files are parsed; a configuration error stops the run before any task, as it does
today (`common.smk:413-447`), and the error names the key.

| Check | Today | In the plan |
| --- | --- | --- |
| V1 | The backend is `python` or `matlab` (`common.smk:21-23`) | `backend` absent or `python`; `matlab` is refused, naming the legacy workflow ([the MATLAB legacy boundary](#the-matlab-legacy-boundary)) |
| V2 | The legacy sample rules need their blocks (`_check_sample_rules`, `workflow.py:1578`); the presets need their rule sections (`validate_workflow_mode_dependencies`, `common.smk:57-92`) | Without `backend`, a MATLAB-dialect key (`rules`, `workflow_mode`, `dapi_round`, `fov_id_pattern`, `n_fovs`, the subset keys, `matlab_launcher`, `matlab_single_thread`) is refused, asking for `backend: matlab` (D7); with `backend: python`, each key that left the dialect is refused, naming where it went ([migration](#migration-from-the-current-python-keys)) |
| V3 | A rule enabled without its input producer fails when Snakemake builds the graph (`MissingInputException`, O8) | A step whose input no task produces is refused, naming the missing block ([the task graph](#the-task-graph-from-data-dependencies)) |
| V4 | `codebook`, `direct_panel`, `image_dir_template` refused for MATLAB in `common.smk:231-235` | The schema's Python-only clause refuses them (OC5) |
| V5 | `_check_python_dialect` and `from_workflow_config` of each enabled processing rule (`common.smk:417-422`) | The same translator on the `processing` block, once |
| V6 | `_stitching_block`, `_stage_position_ids` (`common.smk:440-442`) | The same, on the `stitching` block |
| V7 | `_export_config`, `_check_export_route` (`common.smk:443-445`) | The same; the models are still checked by `Dataset.export` |
| V8 | `_report_block` (`common.smk:446-447`) | The same |
| V9 | The schema (`Snakefile:19`, `snakemake.utils.validate`) | The plan validates the configuration against the Python schema with `jsonschema`, without inserting defaults (the resolved configuration writes them) |
| V10 | `_membership` (`common.smk:213-216`) | The same: `fovs.csv`, `selected_fovs` |

### Input discovery

The plan lists, without reading any image: `fovs.csv`; for each selected FOV and each round a task
reads (the sequencing rounds for `processing`, the other rounds of the morphology block), the round's
folder (`_layout.image_dir`, today `_image_dir_inputs`,
`_snakemake.py:597`) and in it the one file of each channel label, found as `io.load_round` finds
it (`*<label>*.tif` or `.tiff`, `io/tiff.py`, `load_round`); the reference stain files of the
reference round, which the morphology task reads and `FOV.run` does not (the `fov` task's `run.json`
lists the sequencing channels only); the codebook or panel (`_reference_file`, `workflow.py:908`); and each supplied
file (the position table or stage file, a supplied `dataset.xml`, a supplied statistics file, an
imported label file). For each file it records the path relative to the input root (OC4; a file
outside it by its absolute path, as `run.json` version 2 does), its size in bytes and its
modification time in nanoseconds. It checks completeness: every selected FOV has a folder for
every round it needs, and each channel label matches exactly one file; a missing or ambiguous file
is refused, naming the FOV, the round and the label. Full content hashes are computed inside the
tasks, which read the files anyway (`run.json` `inputs`, `CheckpointConfig.hash_inputs`).

`selected_fovs` restricts every task, per-sample tasks included, as `_membership` does today
(`workflow.py:1481`); a regular grid is still laid out over the sample's FOVs of `fovs.csv`
(`_sample_grid_fovs`, `workflow.py:1614`), which the stitch tasks read from the manifest's
membership.

### The resolved configuration

The resolved configuration, `workflow/resolved.json`, holds:

* `format`: `starfinder.workflow.resolved/1`;
* `source`: the configuration file the plan read, its absolute path and the SHA-256 of its bytes;
* `configuration`: the configuration in the [Python shape](#the-python-configuration-shape) with
  every default the contract defines written out (the backend, the processing mode and checkpoint
  stages, the morphology rounds, the assignment name, the previews, the execution defaults, the
  supplied statistics file the two-pass tasks write) and `api`, the API configs the block
  translators build from it, each dataclass written field by field with its type name
  (`PipelineConfig`, `ExecutionConfig`, the `SegmentationPlan`, the `AssignmentConfig`, the stitching
  method config, `ExportConfig`, `ReportConfig`).

A task translates its block again from `configuration` and checks that the result equals `api`, so
a package that translates differently from the one that planned is refused (exit 2) rather than
run.

**Canonical serialization.** JSON, UTF-8, object keys sorted, no insignificant whitespace
(`separators=(",", ":")`), floats in Python's shortest round-trip form, tuples as lists, paths as
strings, `NaN` and infinities refused.

**The hash.** `configuration_sha256` is the SHA-256 of the canonical serialization of
`configuration` without `execution`, `root_input_path` and `root_output_path`: the settings that
can change a result. A change of threads, memory, container or retries, or a move of the trees,
keeps it; a change of any scientific setting changes it. The manifest also records the SHA-256 of
the whole file, which a task checks.

### Choice: the serialization and the scope of the hash

| Option | Cost | Consequence |
| --- | --- | --- |
| A. Canonical JSON; the hash over the result-affecting part (recommended) | The plan writes one more file; the hash leaves out a named list of keys | Resume and reports compare scientific settings across hosts and resources; moving a tree or raising memory does not invalidate a checkpoint |
| B. Canonical JSON; the hash over everything | Simplest | A change of threads invalidates every resume |
| C. Hash the source YAML | No resolved file | Two files with the same settings in another order differ; defaults are not part of the hash, so a default changed by a package update goes unnoticed |
| D. RFC 8785 (JCS) exactly | A JCS implementation (numbers in ECMAScript form) | Byte-identical with other JCS tools; the package has no such dependency today |

Recommendation: A; JSON values of the configuration are strings, integers, floats, booleans,
lists and objects, for which A's form and JCS differ only in the spelling of some floats.

### Choice: what the resolved configuration holds

| Option | Cost | Consequence |
| --- | --- | --- |
| A. The configuration with its defaults and the translated API configs (`api`) (recommended) | A larger file; the dataclasses are written field by field | The record says exactly what each task runs; a translator change between the plan and a task is detected |
| B. The configuration with its defaults only | Smaller | A task re-translates silently |

Recommendation: A.

### The manifest

`workflow/manifest.json`, one JSON file, format `starfinder.workflow.manifest/1`:

(orchestration-manifest-fields)=

| Field | Content |
| --- | --- |
| `format` | `starfinder.workflow.manifest/1` |
| `backend` | `python` |
| `configuration` | `path` (`workflow/resolved.json`), `sha256` (of the file), `configuration_sha256` (the resolved-configuration hash) |
| `roots` | `input` and `output`, absolute |
| `membership` | `fovs` (in the order of `fovs.csv`), `samples` (each sample's FOVs), `selected_fovs` |
| `inputs` | `files` (`fovs.csv`, the codebook, the supplied files) and `images` (per FOV, round and channel label), each with `path`, `bytes`, `mtime_ns` |
| `export_route` | `per_fov`, `after_fusion`, `molecules_only` or null |
| `tasks` | The tasks, in an order that respects `needs` |
| `tasks[].id` | `<kind>/<sample>[/<step>][/<fov>][/<tile>][/<run> or /<name>]`; `compare` has the id `compare` |
| `tasks[].kind` | The task kind |
| `tasks[].identity` | `sample`, and as the kind has them `fov`, `tile`, `run`, `name`, `step` |
| `tasks[].needs` | The ids of the tasks whose outputs it reads |
| `tasks[].inputs` | Declared inputs: paths relative to the output root, or `input:<path>` relative to the input root, or absolute for a file outside both |
| `tasks[].outputs` | Declared outputs, relative to the output root (a folder ends in `/`) |
| `tasks[].record`, `tasks[].log` | `workflow/tasks/<task>.json`, `workflow/logs/<task>.log` |
| `tasks[].command` | The command line, `["starfinder", "workflow", <command>, "--manifest", <the manifest's absolute path>, "--task", <id>]`, to which the engine adds its context; for `bigstitcher_engine`, the tool's command line with the placeholder `{fiji}`, which the engine replaces by `execution.tool.path` |
| `tasks[].execution` | `threads`, `memory_mb`, `runtime_min`, `gpu`, `container` and, for the Fiji task, `tool`; passed through from the execution section, never read by the command |

The manifest is regenerated by every run of the plan command; the plan writes the file only when its
content changes, so that an engine that compares modification times reruns nothing on an unchanged
plan. A change of the configuration file, of `fovs.csv`, of an input's path, size or modification
time, of a supplied file or of the package version changes the manifest. Which tasks then rerun is
the engine's decision, on the declared inputs and outputs (and, for Snakemake, its own records).

### Choice: the manifest format

| Option | Cost | Consequence |
| --- | --- | --- |
| A. One JSON file (recommended) | Read whole by each command (a few hundred kB for a thousand FOVs) | Snakemake reads it with `json`, Nextflow with its JSON parser; one file to hash and compare |
| B. One JSON file per task | Many small files | No task reads the others; the plan writes thousands of files |
| C. A CSV per task kind | Lists in cells (inputs, outputs) | Nextflow's `splitCsv` reads it directly; lists need a separator rule |
| D. JSON Lines | One line per task | Streams; D8 excludes a JSONL event stream, and the manifest is not one, but it invites one |

Recommendation: A; if a run of many thousands of FOVs makes the file slow, an index per kind can
be added without changing the fields.

### Choice: task ids and paths in the manifest

| Option | Cost | Consequence |
| --- | --- | --- |
| A. Ids that spell the identity (`fov/sample/FOV_001`); paths relative to the output root, `input:` relative to the input root, absolute outside both (recommended) | Ids are as long as their identities | A person reads which task failed from its id and its log's name; a copied or moved tree keeps its manifest valid with new `roots`, as `run.json` version 2 keeps its paths (OC4) |
| B. Hashed ids, absolute paths | None | Ids say nothing; a moved tree needs a new plan |

Recommendation: A. A FOV id, a run name or an assignment name becomes part of a file name, as it does
today in `fovs/<fov>/`; the plan refuses one with a `/`.

### Choice: when the manifest is written

| Option | Cost | Consequence |
| --- | --- | --- |
| A. Every engine invocation runs the plan, which rewrites the files only when their content changes (recommended) | The plan's input scan (sizes and times, no hash) at every invocation | The manifest always describes the inputs as they are; an unchanged plan touches nothing, so no task reruns for it |
| B. The user runs the plan when something changes | None | A stale manifest runs old inputs |

Recommendation: A.

### Choice: where the workflow files live

| Option | Cost | Consequence |
| --- | --- | --- |
| A. `<output root>/workflow/` (`resolved.json`, `manifest.json`, `tasks/`, `logs/`), with `_layout` helpers (recommended) | Four new helpers in `_layout` | One folder that the run owns; the stage folders stay as they are |
| B. The MATLAB-era `log/` | None | Mixes with MATLAB's console logs and Snakemake's benchmark files |
| C. Beside each output (`fovs/<fov>/logs/`) | A rule per kind | The per-sample tasks have no FOV folder |

Recommendation: A, through `_layout.workflow_dir`, `manifest_file`, `task_record` and
`task_log`.

## The task graph from data dependencies

D4: a present block enables its step, and the plan derives the tasks from the data each step needs.
Each data product has one producer; each consumer `needs` the producer's task:

(orchestration-graph)=

| Data product | Produced by | Read by |
| --- | --- | --- |
| The supplied statistics of a sample and step | `merge` (after the `summarize` of each FOV of the sample) | the next `summarize` of a later supplied step; `fov` or `tile_global` |
| A FOV's reference image (`reference_image`) | `fov` or `tile_global`; none with `reference_image_import` (an input) | `segment` (the grid; `reference_merged` and composite inputs), `assign` (the grid), `stitch`, `bigstitcher_macro`, `bigstitcher_import` (the grids), `export` (the grids on the route after fusion, and the fusion of `reference_merged`) |
| A FOV's molecules (`fovs/<fov>/molecules.csv`) | `fov` or `tile_merge`, with `spot_finding`, `reads_extraction` and `reads_filtration` | `assign`; `export` on the routes after fusion and molecules only |
| A FOV's run record and checkpoint stages | `fov` or `tile_merge` | `report` |
| A registered sequencing round (`checkpoints/registered/<round>.ome.tif`) | `fov`, when a segmentation input names a sequencing round: the plan adds the stage `registered` | `segment` |
| A prepared image (`other_rounds/<name>/`: `reference_stain` and each other round) | `morphology` | `segment` (inputs of other rounds, and the grid without a processing block), `stitch` with `input: reference_stain`, `export` fusing `reference_stain` |
| A segmentation run (`segmentation/<run>/`) | `segment` (after the run its `seeds` name) | `assign` (`cells`, `nuclei`) |
| A saved assignment (`assignment/<name>/`) | `assign` | `export` on the route per FOV; `report` |
| A sample's placement (`stitching/<sample>/`) | `stitch` or `bigstitcher_import` | `export`, `report` |
| A sample's export (`export/<sample>/`) | `export` | `report` |
| A sample's report | `report` | `compare` |

The cases D4 names:

* **Processing that stops before spot finding.** A `processing` block without `spot_finding`
  (`PipelineConfig.spot_finding` and the operations after it `None`) produces the run record and
  the reference image only; a step that reads molecules is then refused.
* **Morphology alone for nuclear stains.** `morphology` and a segmentation run on the reference
  stain need no `processing` block: the run's grid is the reference stain's saved form, which shares
  the reference round's acquisition (an API addition, [open point OP4](#open-points)).
* **The reference image for composites.** A run whose input is the composite of the nuclear stain
  and the reference merged image (`prepare: {function: composite_nuclei_amplicon}`) needs the
  `processing` block up to the reference image, and `morphology` for the stain.
* **Molecules for assignment.** `assignment` needs `processing` with the three operations that make
  molecules, and the segmentation runs it names.
* **The two-pass barrier.** A preprocessing step with `fit: supplied` and no `supplied_statistics`
  makes the plan add, per sample and per supplied step in recipe order, one `summarize` per FOV and
  one `merge`; every `fov` (or `tile_global`) task of the sample needs the last `merge`
  ([the two-pass statistics tasks](#the-two-pass-statistics-tasks)).
* **The export routes.** `per_fov` needs every FOV's saved assignment; `after_fusion` needs the
  molecules and the reference images (the export segments the fused projection itself);
  `molecules_only` (`assembly.assignment: null`) needs the molecules. The route is resolved by
  `resolve_route` (`export/_segment.py`) as today.

**The plan-time error.** A step whose input no task produces is refused before any task runs,
naming the missing block, for example (the prototype's message for an assignment block without a
processing block):

```text
plan refused: the assignment block assigns each FOV's molecules, which only the processing block produces (with
spot_finding, reads_extraction and reads_filtration): add a processing block
```

Today the same configuration passes the parse and fails in the graph or in a job: the
`ruleorder` and `_legacy_molecules` choose a molecule producer twice (O18), and
`enhance_dapi_with_flamingo` reads files no rule declares (O8).

## The Python configuration shape

The Python dialect gets its own shape (D4): the dataset keys at the top level as today, one block
per step, and an execution section. A present block enables its step.

```yaml
backend: python                 # the default (D7)
root_input_path: /data/aging
dataset_id: aging
sample_id: sample1
root_output_path: /results
output_id: run1
n_rounds: 4
ref_round: round1
seq_channel_order: [ch00, ch01, ch02, ch03]
reference_stains: [{channel: ch04, name: DAPI}]
ref_channel: DAPI
additional_round: [{round_name: flamingo, channel_order: [{channel: ch04, name: DAPI}, {channel: ch05, name: Flamingo}]}]
rotate_angle: 0                 # 0: no rotation (OC3)
voxel_size_xy: 0.5
voxel_size_z: 1.0
processing:                     # the per-FOV core: direct or tiled
  mode: direct                  # or tiled (W-644)
  preprocessing: {steps: [...]}
  registration: {steps: [{method: translation}]}
  spot_finding: {method: local_maxima}
  reads_extraction: {voxel_size: [1, 2, 2]}
  reads_filtration: {end_base: CC}
  decoding: {method: wta}
  checkpoints: [candidates, pre_qc]   # the default (D8)
morphology: {}                  # the reference stain and the other rounds, prepared
segmentation: {runs: [...]}     # one segment task per (FOV, run)
assignment: {cells: cell}       # one assign task per FOV; previews: false
stitching: {method: multiview_stitcher, positions: {...}}
assembly: {assignment: default}
export: {route: per_fov}        # previews: false
report: {}
execution:
  defaults: {threads: 1, memory_mb: 8000, runtime_min: 30, gpu: false}
  tasks: {fov: {threads: 4, memory_mb: 16000}}
  container: null               # the image by digest (D9)
  fiji: null                    # the BigStitcher engine task's tool (D3)
  resume_from_checkpoint: false
```

**The blocks.**

| Block | Its keys | Its tasks |
| --- | --- | --- |
| `processing` | Today's processing rule parameters without the `run` flags: `preprocessing`, `registration`, `reference_image_view` (with `projection`), `spot_finding`, `reads_extraction`, `reads_filtration`, `decoding`, `scoring` (`false` turns it off), `deduplication`, `load_codebook`, `load_raw_images`, `streaming`, `device`; and `mode` (`direct` or `tiled`), `tiled` (W-644) and `checkpoints` | `fov`, or `tile_global`, `tile`, `tile_merge`; `summarize` and `merge` for supplied steps |
| `morphology` | `rounds` (default every other round), `recipe` (a registration recipe; default one translation step on the shared stain, `MorphologyConfig`), `channel_images` (null; or `{projection: null or max}` to write `images/<round>/<label>/<fov>.tif` as today) | `morphology` |
| `segmentation` | As today: `device`, `runs` | `segment` per (FOV, run) |
| `assignment` | As today: the `AssignmentConfig` fields, `cells`, `nuclei`, `name` (default `default`), `population`; and `previews` (P5) | `assign` per FOV |
| `stitching` | As today; and `dataset_xml` (a supplied file with `{sample}`, which skips the macro and the Fiji task, D3) | `stitch`, or `bigstitcher_macro`, `bigstitcher_engine`, `bigstitcher_import` |
| `assembly`, `export` | As today; `export.previews` (P5) | `export` per sample |
| `report` | As today | `report` per sample, `compare` |
| `execution` | `defaults` and `tasks.<kind>` (`threads`, `memory_mb`, `runtime_min`, `gpu`), `container`, `fiji`, `resume_from_checkpoint`, `log_level` | none; passed through to each task's `execution` |

No scientific default changes: scoring stays on whenever the block decodes, deduplication stays
off without its block, and every translator default (for example the WTA decoder's `diagnostics:
true`) stays as W-642 recorded it. The defaults that do change are the ones a decision changed:
the checkpoint stages (D8), the backend (D7) and the meaning of `rotate_angle: 0` (OC3).

**The default backend and the MATLAB dialect (D7).** `backend` defaults to `python`. A
configuration without `backend` that holds a key of the MATLAB dialect is refused:

```text
plan refused: this configuration has no backend and holds keys of the MATLAB dialect (workflow_mode, rules); the
default backend is python: add backend: matlab to run the MATLAB workflow, or migrate it to the Python
configuration (docs/migration.md, "The Python configuration shape")
```

The schema keeps the MATLAB dialect for `backend: matlab` and gives the Python shape its own
branch; whether the two are one schema with two branches or two schema files is the engine record's.

### Choice: the shape of the top level and of the processing block

| Option | Cost | Consequence |
| --- | --- | --- |
| A. Dataset keys at the top level as today; the processing block keeps today's sub-block names without `run` (recommended) | The migration is a move and a deletion per key | Every key below `processing` keeps its name, its translator and its tests; the migration table is mechanical |
| B. A `dataset` block for the top-level keys and renamed sub-blocks (`codebook` for `load_codebook`, `detection` for `spot_finding`) | Every key moves and many are renamed | A cleaner file; every guide, example and test changes |

Recommendation: A.

### Choice: the execution section

| Option | Cost | Consequence |
| --- | --- | --- |
| A. `defaults` and `tasks.<kind>` (recommended) | Kind names in the configuration | Resources follow the task, which is what an engine schedules (O15) |
| B. Resources inside each block | Each block grows engine keys | The boundary between scientific and execution keys blurs, and the hash must skip keys inside blocks |

Recommendation: A; `execution` is outside the configuration hash.

### Choice: the names of the new keys

| Key | Option | Cost | Consequence |
| --- | --- | --- | --- |
| The previews (P5) | A. `assignment.previews` and `export.previews` (recommended) | Two keys | Each command reads its own block; one can be on without the other |
| | B. One top-level `previews` | One key | A key outside the blocks that two commands read |
| The supplied BigStitcher result (D3) | A. `stitching.dataset_xml` with `{sample}` (recommended) | One key | The method stays `bigstitcher`, whose config the import reads; the plan drops the macro and Fiji tasks |
| | B. A method `bigstitcher_import` | A method name outside `STITCHING_METHODS` | Two names for one method config |
| Verified resume (D8) | A. `execution.resume_from_checkpoint` (recommended) | None | Resuming never changes a result, so it stays outside the configuration hash |
| | B. `processing.resume_from_checkpoint` | None | Turning resume on would change the hash that resume compares |
| The morphology channel images | A. `morphology.channel_images`, off by default (recommended) | One key | D6 retires their readers; the golden comparison turns it on to compare them |
| | B. Always written | Disk per FOV and round | Files nothing reads |
| | C. Removed | None | The golden baseline's morphology images (W-642, A) have no counterpart |

Recommendation: A for each.

### Migration from the current Python keys

Every key of W-642's [configuration inventory](../baselines/orchestration.md#the-python-configuration-as-read-today)
appears once below, with its place in the new shape or its removal and the reason. The migration
guide's table (written when the implementation lands) is generated from this one; there is no shim.

% Generated and checked by scripts/migration_table.py (results/migration/migration.md, check.json)

(orchestration-migration)=

| Key (W-642's inventory) | Proposed key | Reason |
| --- | --- | --- |
| `config_path` | Removed | The plan records the file it read (`source` of the resolved configuration: path and SHA-256), and `run.json` records the resolved configuration's hash (D2, D8) |
| `starfinder_path` | Removed | The engines run the `starfinder` command of the locked environment or image; no rule script imports a checkout (D2, D9) |
| `root_input_path` | Unchanged | The plan records the input files relative to the input root (OC4) |
| `dataset_id`, `sample_id` | Unchanged |  |
| `root_output_path`, `output_id` | Unchanged |  |
| `fiji_path` | `execution.fiji` | The tool of the BigStitcher engine task, which only Snakemake runs (D3) |
| `envs_path` | Removed | No reader (O7; D6, D9) |
| `selected_fovs` | Unchanged | Applied by the plan to every task (D2) |
| `n_rounds` | Unchanged |  |
| `ref_round` | Unchanged |  |
| `ref_channel` | Unchanged | Read by the morphology block (the shared stain) |
| `reference_stains` | Unchanged |  |
| `seq_channel_order` | Unchanged |  |
| `seq_channel_order[]`, `reference_stains[]`, `additional_round[].channel_order[]`: a pattern string or a mapping `{channel, name, wavelength}` | Unchanged |  |
| `additional_round` | Unchanged | No longer read when a workflow file is parsed, so `[]` is not needed (D2) |
| `additional_round[].round_name` | Unchanged |  |
| `additional_round[].channel_order` | Unchanged |  |
| `additional_round[].channel_order[].channel`, `.name` (`nuclei_registration`) | Unchanged | Read by the morphology block |
| `rotate_angle` | Unchanged | Required as today; `0` means no rotation and the step is skipped, so frame ids lose `/rotate:0` (OC3) |
| `img_row`, `img_col` | Removed | Their readers are retired: the subtile windows (D5) and the legacy tile configuration writer (D6) |
| `img_z` | Removed | Its reader, a YX label file on the legacy route of `reads_assignment`, is retired (D6) |
| `voxel_size_xy`, `voxel_size_z` | Unchanged |  |
| `reference_image` | Unchanged |  |
| `reference_image_import` | Unchanged |  |
| `image_dir_template` | Unchanged | In the schema's Python-only clause; the `common.smk` refusal goes (OC5) |
| `readout_mode` | Unchanged |  |
| `codebook` (`readout_mode: multiplexed`), `direct_panel` (`direct`) | Unchanged | In the schema's Python-only clause; the `common.smk` refusal goes (OC5) |
| `maximum_projection` | `processing.reference_image_view.projection` (`max`) and `morphology.channel_images.projection` (`max`) | Its two remaining readers; the third, the legacy StarDist translation, is retired (D6) |
| `backend` | Unchanged | Default `python`; without it, MATLAB-dialect keys are refused, naming `backend: matlab` (D7) |
| `workflow_mode` | Removed | A present block enables its step (D4); direct or tiled is `processing.mode` (D5) |
| `rules`, `rules.<rule>.parameters` | Removed | Each rule's parameters move to the block of its step (the rows below) (D4) |
| `rules.<rule>.run` | Removed | A present block enables its step (D4) |
| `rules.<rule>.resources.mem_mb`, `.runtime` | `execution.tasks.<kind>.memory_mb`, `.runtime_min` | By task kind, with `execution.defaults` (8000 and 30, today's defaults) (D4; O15) |
| `matlab_launcher`, `matlab_single_thread` (MATLAB rules only) | MATLAB dialect only | Without `backend`, a configuration that holds them is refused, naming `backend: matlab` (D7) |
| `rules.<processing rule>.parameters`: the keys it accepts | `processing` | The same keys, unknown keys raise; the five processing rules become one block with `mode` (D4, D5) |
| `rules.<processing rule>.parameters.streaming` | `processing.streaming` |  |
| `rules.<processing rule>.parameters.device` | `processing.device` |  |
| `rules.<processing rule>.parameters.<block>.run` (`spot_finding`, `reads_extraction`, `reads_filtration`, `load_codebook`, `load_raw_images`, `create_subtiles`) | Removed | A present sub-block enables its operation (`spot_finding`, `reads_extraction`, `reads_filtration`); the codebook is loaded whenever the block decodes and the raw images always; `create_subtiles` is retired (D4, D5) |
| `rules.<processing rule>.parameters.load_codebook`, `.load_codebook.run` | `processing.load_codebook` | Without `run` (D4) |
| `rules.<processing rule>.parameters.load_codebook.split_index` | `processing.load_codebook.split_index` |  |
| `rules.<processing rule>.parameters.load_codebook.encoding` | `processing.load_codebook.encoding` |  |
| `rules.<processing rule>.parameters.load_codebook.encoding.method` | `processing.load_codebook.encoding.method` |  |
| `rules.<processing rule>.parameters.load_codebook.encoding.base_to_color`, `.pair_to_color` | `processing.load_codebook.encoding.base_to_color`, `.pair_to_color` |  |
| `rules.<processing rule>.parameters.load_raw_images`, `.subdir` | `processing.load_raw_images.subdir` | Without `run`: the processing task always loads the raw images (D4) |
| `rules.<processing rule>.parameters.preprocessing` | `processing.preprocessing` |  |
| `….preprocessing.steps[]`, `….preprocessing.post_registration[]` | `processing.preprocessing.steps[]`, `.post_registration[]` |  |
| `….preprocessing.extraction_source`, `.registration_source`, `.supplied_statistics` | `processing.preprocessing.extraction_source`, `.registration_source`, `.supplied_statistics` | Without `supplied_statistics`, a `fit: supplied` step makes the plan add the two-pass tasks, whose merge writes the file (§2.5, W-230) |
| `….preprocessing.steps[].method`, `….preprocessing.post_registration[].method` | `processing.preprocessing.steps[].method`, `.post_registration[].method` |  |
| `….preprocessing.steps[].save_as`, `….preprocessing.post_registration[].save_as` | `processing.preprocessing.steps[].save_as`, `.post_registration[].save_as` |  |
| `rules.<processing rule>.parameters.registration` | `processing.registration` |  |
| `….registration.steps[]` | `processing.registration.steps[]` |  |
| `….registration.signal`, `.warp`, `.qc` | `processing.registration.signal`, `.warp`, `.qc` |  |
| `….registration.reference_round` | `processing.registration.reference_round` |  |
| `….registration.steps[].method` | `processing.registration.steps[].method` |  |
| `….registration.steps[].signal` | `processing.registration.steps[].signal` |  |
| `….registration.steps[].recovery` | `processing.registration.steps[].recovery` |  |
| `….registration.steps[].recovery.allowed_errors` | `processing.registration.steps[].recovery.allowed_errors` |  |
| `….registration.steps[].recovery.alternatives[]` | `processing.registration.steps[].recovery.alternatives[]` |  |
| `rules.<processing rule>.parameters.reference_image_view`, `.mode`, `.channel` | `processing.reference_image_view`, `.mode`, `.channel` | With `projection` (from `maximum_projection`) |
| `rules.<processing rule>.parameters.spot_finding` | `processing.spot_finding` | Present: detection on (D4) |
| `rules.<processing rule>.parameters.spot_finding.ref_round` | `processing.spot_finding.ref_round` |  |
| `rules.<processing rule>.parameters.spot_finding.method` | `processing.spot_finding.method` |  |
| `rules.<processing rule>.parameters.spot_finding.channel_overrides.<label>` | `processing.spot_finding.channel_overrides.<label>` |  |
| `rules.<processing rule>.parameters.spot_finding.rounds` | `processing.spot_finding.rounds` |  |
| `rules.<processing rule>.parameters.reads_extraction`, `.voxel_size` | `processing.reads_extraction`, `.voxel_size` | Present: extraction on (D4) |
| `rules.<processing rule>.parameters.reads_extraction.background` | `processing.reads_extraction.background` |  |
| `rules.<processing rule>.parameters.reads_filtration` | `processing.reads_filtration` | Present: decoding and filtering on (D4) |
| `rules.<processing rule>.parameters.reads_filtration.end_base` | `processing.reads_filtration.end_base` |  |
| `rules.<processing rule>.parameters.reads_filtration.n_barcode_segments`, `.split_index` | `processing.reads_filtration.n_barcode_segments`, `.split_index` |  |
| `rules.<processing rule>.parameters.reads_filtration.start_base`, `.exclude_invalid_endpoints`, `.score_bounds` | `processing.reads_filtration.start_base`, `.exclude_invalid_endpoints`, `.score_bounds` |  |
| `rules.<processing rule>.parameters.decoding` | `processing.decoding` |  |
| `rules.<processing rule>.parameters.decoding.method` | `processing.decoding.method` |  |
| `rules.<processing rule>.parameters.decoding` of method `wta` | `processing.decoding` of method `wta` | The translator's defaults unchanged |
| `rules.<processing rule>.parameters.decoding` of method `codebook_aware` | `processing.decoding` of method `codebook_aware` | The translator's defaults unchanged |
| `rules.<processing rule>.parameters.decoding` of method `direct` | `processing.decoding` of method `direct` |  |
| `rules.<processing rule>.parameters.scoring`, `.run`, `.method` | `processing.scoring`, `.method` | On whenever the block decodes, as today; `scoring: false` replaces `run: false` (no scientific default changes) |
| `rules.<processing rule>.parameters.deduplication`, `.run` | `processing.deduplication` | Present: on (its `run` goes); absent: off, as today (D4) |
| `rules.<processing rule>.parameters.create_subtiles`, `.sqrt_pieces`, `.overlap_ratio` (of `gr_single_fov_subtile` or `deep_create_subtile`) | Removed | The legacy subtile route is retired for Python; the tiled settings are `processing.tiled`, which W-644 specifies (D5) |
| `rules.<processing rule>.parameters.checkpoints`, `.stages` | `processing.checkpoints` | A list of stages, default `[candidates, pre_qc]`; the plan adds `registered` when a data dependency needs it (D8; O5) |
| Refused in `rules.<processing rule>.parameters`: `enhance_contrast`, `hist_equalize`, `morph_recon`, `tophat`, `snr_threshold`, `global_registration`, `local_registration` | Refused in `processing` | As today |
| Refused in `rules.<processing rule>.parameters.spot_finding` and its `channel_overrides` (method `local_maxima`): `intensity_estimation`, `intensity_threshold`, `min_distance` | Refused in `processing.spot_finding` | As today |
| Refused in `….registration` steps and alternatives: the MATLAB parameter names and `ref_img`, `mov_img`, `ref_channel`, `boundary_mode`, `ref_round`; the demons variant names as `method`; `merged-image`, `merged`, `single-channel` as a signal `mode` | Refused in `processing.registration` | As today |
| Refused in `….preprocessing` steps: `radius` of a `reconstruction` or `white_tophat` step | Refused in `processing.preprocessing` | As today |
| Refused at the top level: `dapi_round`, `channel_order`, `fov_pattern` | Refused | As today; `dapi_round` is also a MATLAB-dialect key of D7's check |
| `readout_mode: direct` refuses `rules.<processing rule>.parameters.load_codebook.split_index`, `.encoding` and `reads_filtration.end_base`, `.split_index`, `.n_barcode_segments`, `.exclude_invalid_endpoints` | Refused in `processing` with `readout_mode: direct` | As today |
| `rules.nuclei_registration.parameters.checkpoints` | Removed | The morphology task always saves the prepared images (`other_rounds/`), which the segment, stitch and export tasks read (D4, D6) |
| `rules.create_nuclei_amplicon_overlay.parameters.maximum_projection` | Removed | The composite is a segmentation input (`prepare: {function: composite_nuclei_amplicon}`) and its projection the run's `projection` (D6) |
| `rules.stardist_segmentation.parameters`: the keys it accepts | `segmentation.runs[]` of method `stardist` | The legacy segmentation chain is retired for Python (D6); the rows below map each key |
| `rules.stardist_segmentation.parameters.stardist_base_path`, `.stardist_model_name` | `segmentation.runs[].model_path` or `.model` | `<stardist_base_path>/<stardist_model_name>`, or the known model's name (D6) |
| `rules.stardist_segmentation.parameters.segmentation_input_folder` | `segmentation.runs[].inputs[]` | `overlay`: the reference stain with `prepare: {function: composite_nuclei_amplicon}` (role `composite`); `DAPI`: the reference stain (role `nuclear`); `flamingo/enhanced_DAPI`: the `flamingo` round's DAPI channel with `prepare: {function: enhance_with_flamingo}` and `prepare_channel` (D6) |
| `rules.stardist_segmentation.parameters.prob_thresh`, `.nms_thresh` | `segmentation.runs[].prob_thresh`, `.nms_thresh` |  |
| `rules.stardist_segmentation.parameters.rescale` | `segmentation.runs[].scale` | `true`: 0.5, `false`: 1.0 (D6) |
| `rules.stardist_segmentation.parameters.expand_labels`, `.distance` | `segmentation.runs[].operations[]` | `{operation: expand_labels, distance, unit: pixel, mode: planar}` (D6) |
| `rules.stardist_segmentation.parameters.target`, `.device` | `segmentation.runs[].target`; `segmentation.device` | (D6) |
| `rules.reads_assignment.parameters.expand_labels`, `.dilation_distance` | `assignment.expansion` with `assignment.legacy_pixel_expansion: true` | `{distance: <dilation_distance>, unit: pixel, mode: planar}` (D6) |
| `segmentation` | Unchanged | Present: the segment tasks per (FOV, run) (D6) |
| `segmentation.device`, `.runs` | Unchanged | `device` gives each segment task its GPU (D6) |
| `segmentation.runs[].name`, `.method` | Unchanged |  |
| `segmentation.runs[].target`, `.seeds` | Unchanged | `seeds` orders the segment tasks (D6) |
| `segmentation.runs[].projection` | Unchanged |  |
| `segmentation.runs[].operations[]`, `.operation` | Unchanged |  |
| `segmentation.runs[].inputs[]` | Unchanged | Each input's round names the task that produces it (D4) |
| `segmentation.runs[].inputs[].prepare`, `.prepare.function` | Unchanged |  |
| `assignment` | Unchanged | Present: the assign task per FOV (D6); with `previews` (P5) |
| `assignment.cells`, `.nuclei`, `.name`, `.population` | Unchanged | `name` defaults to `default` |
| `assignment.expansion`, `.correspondence` | Unchanged |  |
| `assignment.legacy_pixel_expansion`, `.exclude_cells_without_nucleus` | Unchanged |  |
| `assignment.checkpoints` | Removed | The assign task always saves the assignment folder (D6, D8) |
| `stitching` | Unchanged |  |
| `stitching.method` | Unchanged | Default `bigstitcher` (§2.10); Nextflow refuses it without a supplied `dataset.xml` (D3) |
| `stitching.input` | Unchanged |  |
| `stitching.config` | Unchanged |  |
| `stitching.review`, `.fallback` | Unchanged |  |
| `stitching.bigstitcher_save_dir` | Unchanged |  |
| `stitching.positions` | Unchanged |  |
| `stitching.positions.source` | Unchanged |  |
| `stitching.positions.stage_file`, `.stage_axes`, `.stage_axes.x`, `.stage_axes.y` | Unchanged |  |
| `stitching.positions.snap_to_grid`, `.snap_to_grid.overlap` | Unchanged |  |
| `stitching.positions.regular_grid`, `.columns`, `.rows`, `.order`, `.overlap` | Unchanged |  |
| `stitching.positions.table`, `.unit` | Unchanged |  |
| `stitching.positions.tile_shape_yx`, `.spacing_yx` | Unchanged |  |
| `assembly` | Unchanged |  |
| `assembly.assignment` | Unchanged | On the route per FOV it must be the assignment block's `name` |
| `assembly.table_format`, `.previews` | `assembly.table_format`; `assignment.previews` and `export.previews` | The previews are options of the assign and export commands, off by default (P5) |
| `export` | Unchanged | With `previews` (P5) |
| `export.segmentation` | Unchanged |  |
| `export.segmentation.device`, `.cells`, `.nuclei`, `.runs` | Unchanged | `device` gives the export task its GPU (D6) |
| `export.segmentation.runs[].name`, `.method`, `.target`, `.seeds`, `.inputs` | Unchanged |  |
| `export.segmentation.runs[].inputs[].role`, `.image` | Unchanged |  |
| `report` | Unchanged |  |
| `report.markers[]`, `report.limits`, `report.limits.spots_near_zero` | Unchanged |  |

(orchestration-migration-fields)=

| Where (W-642's appendix) | Dataclass | Proposed place | Reason |
| --- | --- | --- | --- |
| `rules.<processing rule>.parameters.preprocessing.steps[] and .post_registration[] (method min_max_normalization)` | `MinMaxNormalizationConfig` | `processing.preprocessing.steps[] and .post_registration[] (method min_max_normalization)` | The fields keep their names (D4) |
| `rules.<processing rule>.parameters.preprocessing.steps[] and .post_registration[] (method histogram_matching)` | `HistogramMatchingConfig` | `processing.preprocessing.steps[] and .post_registration[] (method histogram_matching)` | The fields keep their names (D4) |
| `rules.<processing rule>.parameters.preprocessing.steps[] and .post_registration[] (method reconstruction)` | `ReconstructionConfig` | `processing.preprocessing.steps[] and .post_registration[] (method reconstruction)` | The fields keep their names (D4) |
| `rules.<processing rule>.parameters.preprocessing.steps[] and .post_registration[] (method white_tophat)` | `TophatConfig` | `processing.preprocessing.steps[] and .post_registration[] (method white_tophat)` | The fields keep their names (D4) |
| `rules.<processing rule>.parameters.preprocessing.steps[] and .post_registration[] (method scalar_background)` | `ScalarBackgroundConfig` | `processing.preprocessing.steps[] and .post_registration[] (method scalar_background)` | The fields keep their names (D4) |
| `rules.<processing rule>.parameters.preprocessing.steps[] and .post_registration[] (method background_3d)` | `Background3DConfig` | `processing.preprocessing.steps[] and .post_registration[] (method background_3d)` | The fields keep their names (D4) |
| `rules.<processing rule>.parameters.preprocessing.steps[] and .post_registration[] (method percentile_normalization)` | `PercentileNormalizationConfig` | `processing.preprocessing.steps[] and .post_registration[] (method percentile_normalization)` | The fields keep their names (D4) |
| `rules.<processing rule>.parameters.registration.steps[] and recovery.alternatives[] (method translation)` | `TranslationConfig` | `processing.registration.steps[] and recovery.alternatives[] (method translation)` | The fields keep their names (D4) |
| `rules.<processing rule>.parameters.registration.steps[] and recovery.alternatives[] (method rigid)` | `RigidConfig` | `processing.registration.steps[] and recovery.alternatives[] (method rigid)` | The fields keep their names (D4) |
| `rules.<processing rule>.parameters.registration.steps[] and recovery.alternatives[] (method affine)` | `AffineConfig` | `processing.registration.steps[] and recovery.alternatives[] (method affine)` | The fields keep their names (D4) |
| `rules.<processing rule>.parameters.registration.steps[] and recovery.alternatives[] (method bspline)` | `BSplineConfig` | `processing.registration.steps[] and recovery.alternatives[] (method bspline)` | The fields keep their names (D4) |
| `rules.<processing rule>.parameters.registration.steps[] and recovery.alternatives[] (method demons)` | `DemonsConfig` | `processing.registration.steps[] and recovery.alternatives[] (method demons)` | The fields keep their names (D4) |
| `rules.<processing rule>.parameters.registration.steps[] and recovery.alternatives[] (method tps)` | `TpsConfig` | `processing.registration.steps[] and recovery.alternatives[] (method tps)` | The fields keep their names (D4) |
| `rules.<processing rule>.parameters.registration.steps[] and recovery.alternatives[] (method cpd)` | `CpdConfig` | `processing.registration.steps[] and recovery.alternatives[] (method cpd)` | The fields keep their names (D4) |
| `rules.<processing rule>.parameters.registration.signal (recipe and step)` | `RegistrationSignalConfig` | `processing.registration.signal (recipe and step)` | The fields keep their names (D4) |
| `rules.<processing rule>.parameters.registration.warp` | `WarpConfig` | `processing.registration.warp` | The fields keep their names (D4) |
| `rules.<processing rule>.parameters.registration.qc` | `RegistrationQcConfig` | `processing.registration.qc` | The fields keep their names (D4) |
| `rules.<processing rule>.parameters.spot_finding and channel_overrides.<label> (method local_maxima)` | `LocalMaximaConfig` | `processing.spot_finding and channel_overrides.<label> (method local_maxima)` | The fields keep their names (D4) |
| `rules.<processing rule>.parameters.spot_finding and channel_overrides.<label> (method starfish_log)` | `StarfishLogConfig` | `processing.spot_finding and channel_overrides.<label> (method starfish_log)` | The fields keep their names (D4) |
| `rules.<processing rule>.parameters.spot_finding and channel_overrides.<label> (method spotiflow)` | `SpotiflowConfig` | `processing.spot_finding and channel_overrides.<label> (method spotiflow)` | The fields keep their names (D4) |
| `rules.<processing rule>.parameters.spot_finding and channel_overrides.<label> (method piscis)` | `PiscisConfig` | `processing.spot_finding and channel_overrides.<label> (method piscis)` | The fields keep their names (D4) |
| `rules.<processing rule>.parameters.reads_extraction.background` | `LocalBackgroundConfig` | `processing.reads_extraction.background` | The fields keep their names (D4) |
| `rules.<processing rule>.parameters.load_codebook.encoding (method two_base)` | `EncodingConfig` | `processing.load_codebook.encoding (method two_base)` | The fields keep their names (D4) |
| `rules.<processing rule>.parameters.load_codebook.encoding (method one_base)` | `OneBaseEncodingConfig` | `processing.load_codebook.encoding (method one_base)` | The fields keep their names (D4) |
| `rules.<processing rule>.parameters.decoding (method wta)` | `WtaDecoderConfig` | `processing.decoding (method wta)` | The fields keep their names (D4) |
| `rules.<processing rule>.parameters.decoding (method codebook_aware)` | `CodebookAwareDecoderConfig` | `processing.decoding (method codebook_aware)` | The fields keep their names (D4) |
| `rules.<processing rule>.parameters.decoding (method direct)` | `DirectDecoderConfig` | `processing.decoding (method direct)` | The fields keep their names (D4) |
| `rules.<processing rule>.parameters.deduplication (with run)` | `DeduplicationConfig` | `processing.deduplication` | The fields keep their names (D4) |
| `rules.nuclei_registration.parameters.checkpoints` | `CheckpointConfig` | Removed | The morphology command fixes its `CheckpointConfig` (the saved form is always written) (D6) |
| `assignment.checkpoints` | `CheckpointConfig` | Removed | The assign command fixes its `CheckpointConfig` (the folder is always written) (D6, D8) |
| `segmentation.runs[] and export.segmentation.runs[] (method stardist)` | `StarDistConfig` | Unchanged |  |
| `segmentation.runs[] and export.segmentation.runs[] (method cellpose)` | `CellposeConfig` | Unchanged |  |
| `segmentation.runs[] and export.segmentation.runs[] (method seeded_watershed)` | `SeededWatershedConfig` | Unchanged |  |
| `segmentation.runs[] and export.segmentation.runs[] (method import)` | `LabelImportConfig` | Unchanged |  |
| `segmentation.runs[].inputs[]` | `InputChannel` | Unchanged |  |
| `segmentation.runs[].inputs[].prepare (function composite_nuclei_amplicon)` | `CompositeConfig` | Unchanged |  |
| `segmentation.runs[].inputs[].prepare (function enhance_with_flamingo)` | `FlamingoEnhancementConfig` | Unchanged |  |
| `segmentation.runs[].projection (a mapping)` | `ProjectionConfig` | Unchanged |  |
| `segmentation.runs[].operations[] (operation expand_labels)` | `ExpandLabelsConfig` | Unchanged |  |
| `segmentation.runs[].operations[] (operation extend_labels_through_z)` | `ZExtensionConfig` | Unchanged |  |
| `assignment (besides cells, nuclei, name, population, checkpoints)` | `AssignmentConfig` | Unchanged |  |
| `assignment.expansion` | `ExpandLabelsConfig` | Unchanged |  |
| `assignment.correspondence` | `CorrespondenceConfig` | Unchanged |  |
| `stitching.config (method multiview_stitcher)` | `MultiviewStitcherConfig` | Unchanged |  |
| `stitching.config (method bigstitcher)` | `BigStitcherConfig` | Unchanged |  |
| `stitching.review` | `ReviewPolicy` | Unchanged |  |
| `assembly (besides assignment, table_format, previews)` | `AssemblyConfig` | Unchanged |  |
| `export (the flags; route and segment included)` | `ExportConfig` | Unchanged |  |
| `export (the fusion fields)` | `FusionConfig` | Unchanged |  |
| `export (labels_<field>)` | `LabelConfig` | Unchanged |  |
| `export (spatialdata_<field>)` | `SpatialDataExportConfig` | Unchanged |  |
| `report` | `ReportConfig` | Unchanged |  |
| `report.markers[]` | `MarkerGroup` | Unchanged |  |
| `report.limits` | `ReportLimits` | Unchanged |  |
| `report.limits.<rule>` | `RuleLimit` | Unchanged |  |

New keys, which no current key maps to: `processing.mode`, `processing.tiled` (W-644),
`morphology` and its `rounds`, `recipe`, `channel_images`, `assignment.previews`,
`export.previews`, `stitching.dataset_xml`, and `execution` with `defaults`, `tasks`, `container`,
`fiji`, `resume_from_checkpoint` and `log_level`.

## Declared outputs

The files each task kind declares, resolved through `_layout`, by route and option. A folder ends in
`/`. `<fov folder>` is `_layout.fov_dir(output_root, fov)`.

(orchestration-declared-outputs)=

| Task kind | Declared outputs | By option |
| --- | --- | --- |
| `summarize` | `statistics/<sample>/<step>/histograms/<fov>.npz` | One per supplied step |
| `merge` | `statistics/<sample>/<step>/supplied.json` | The last one is the file the processing tasks read |
| `fov` | `run_record` (`<fov folder>/run.json`), `reference_image_path` (none with `reference_image_import`), `spot_table` (`<fov folder>/molecules.csv`, with the operations that make molecules); the stage record of each stage of `processing.checkpoints` that its operations reach (`checkpoints/candidates.json` with detection, `checkpoints/pre_qc.json` with decoding, `checkpoints/registered/transforms.json` with registration), whose payload files the record lists | `checkpoints/registered/<round>.ome.tif` of a round a segmentation input names |
| `tile_global` | The reference image; the chunked registered checkpoint and its record | W-644 names the files |
| `tile` | The tile's record and molecules | W-644 |
| `tile_merge` | `run_record`, `spot_table` | As `fov` (D5) |
| `morphology` | `nuclei_registration_record`; per prepared image (`reference_stain` with `reference_stains`, each round of `morphology.rounds`) `other_round_dir/<name>/image.ome.tif` and `registration.json` (`field.npz` with a dense recipe) | `image_file(<round>, <label>)` per channel with `morphology.channel_images` |
| `segment` | `segmentation_dir/<run>/labels.tif`, `segmentation.json`, and `input.ome.tif` except for an import | — |
| `assign` | `assignment_dir/<name>/molecules.csv`, `cells.csv`, `counts.csv`, `assignment.json` (Parquet with `table_format: parquet`) | `nuclei.csv` with `nuclei`; `territories.tif` with `expansion`; `fov_previews_dir` with `assignment.previews` |
| `bigstitcher_macro` | `<save dir>/<sample>/BigStitcher_macro.ijm`, `view_map.csv`, `tiles/` | Not with `stitching.dataset_xml` |
| `bigstitcher_engine` | `<save dir>/<sample>/dataset.xml` | Not with `stitching.dataset_xml` |
| `bigstitcher_import`, `stitch` | `stitching_dir/placement.json`, `fovs.csv`, `links.csv` | — |
| `export` | `export_dir/export.json` (it lists the folder's files: the tables, `sample.h5ad`, the fused images, the label images and, after fusion, `segmentation_<run>.ome.tif`) | `export_previews_dir` with `export.previews` on the route per FOV with `assemble` |
| `report` | `report_dir/report.json` (it lists the folder's files) | — |
| `compare` | `report_root/samples.csv`, `report_root/index.html` | — |
| every kind | `workflow/tasks/<task>.json`, `workflow/logs/<task>.log` | — |

`export.json` and `report.json` stay the one declared file of their folders, as today
(`_export_sample_files`, `_snakemake.py:834`; `_report_sample_files`, `:959`): which parts an
export writes depends on its extras and inputs (a part that cannot run is recorded `not_run`,
`_run_export`, `:888`), so the record that lists them is the declaration. Today's undeclared
outputs become declared: the morphology task's prepared images and channel images (O8), the stage
records the report reads (today the report rule declares only `export.json`, `placement.json` or
`run.json` as its input, `_report_sample_files`, and reads the stages without declaring them), the
placement's `fovs.csv` and `links.csv`, the assignment folder's tables, the macro's tiles and the
comparison page.

**The worked example's comparison.** For the three variants of the [worked example](#worked-example)
that today's workflow can express, the declared outputs of each task kind equal what today's
planners and the Snakemake parse return, except where a decision changes them; each exception names
its decision:

| Task kind (today's rule) | Variants | Equal to today | Only in the contract | Only today | Decision |
| --- | --- | --- | --- | --- | --- |
| `fov` (`rsf_single_fov`) | per-fov, molecules-only | `run.json`, `reference.tif`, `molecules.csv` of every FOV (12 each) | `checkpoints/candidates.json`, `pre_qc.json` (the report reads them) | — | D8 |
| `fov` (`rsf_single_fov`) | after-fusion | — | — | The 12 files, which `tile_merge` and `tile_global` declare instead | D5 |
| `tile_global`, `tile`, `tile_merge` | after-fusion | — | Every file (W-644 names them) | — | D5 |
| `morphology` (`nuclei_registration`) | per-fov, after-fusion | `nuclei_registration.json` (4) | `other_rounds/{reference_stain,flamingo}/image.ome.tif`, `registration.json` | — | D6 (O8), D8 |
| `segment` | per-fov | — | Every file (no planner today) | — | D6 |
| `assign` (`reads_assignment`) | per-fov | `assignment/default/assignment.json` (4) | `molecules.csv`, `cells.csv`, `counts.csv`; `nuclei.csv` (the nucleus run); `previews/` | `previews` (written without the trailing `/`) | D8; D6; P5 |
| `stitch` (`stitch_sample`) | per-fov, molecules-only | `placement.json` | `fovs.csv`, `links.csv` | — | D8 |
| `bigstitcher_macro`, `bigstitcher_engine`, `bigstitcher_import` (`bigstitcher_macro_sample`, `bigstitcher_engine`, `stitch_sample`) | after-fusion | The macro, the view map, `dataset.xml`, `placement.json` | `tiles/`; `fovs.csv`, `links.csv` | — | D3, D8 |
| `export` (`export_sample`) | all three | `export.json` | `previews/` (per-fov) | `previews` | D8; P5 |
| `report` (`report_sample`) | all three | `report.json` | — | — | D8 |
| `compare` (`report_dataset`) | all three | `samples.csv` | `index.html` | — | D8 |
| `summarize`, `merge` | per-fov | — | Every file (no planner today) | — | §2.5 through the W-230 API |

Every kind also declares its task record and log (D8). The comparison finds no difference that
no decision explains; every path the planner functions return is also in the Snakemake parse.
Today's planners also show two declared inputs the contract adds: the per-FOV route's export reads
the fusion's source images without declaring them (`_export_sample_files` declares them on the
route after fusion only), and the reference stain's saved form is read and not declared by the
export and the stitching with `input: reference_stain`.

% results/example/planner-comparison.json and .md (step compare_planners.rerun-6), with each variant's
% snakemake-detailed-summary.tsv (Snakemake 9.15.0 of the golden scratch environment, --detailed-summary --forceall all)

## Completion

**The success rule (D8).** A task succeeded when its task record says `status: succeeded` and lists
every declared output in `outputs`, each with its path and SHA-256; for the `fov` and `tile_merge`
kinds, `run.json` also says `succeeded` and lists the outputs written after the processing, as
version 2 already records them (`checkpoints.md`, "run.json"; `FOV._record_output`,
`dataset/fov.py`). Otherwise the command exits non-zero.

**On failure.** The command catches the exception, writes the traceback to its log, removes the
declared outputs this attempt wrote, writes `workflow/tasks/<task>.failed.json` (the failed record,
not a declared output, kept for diagnosis), and exits 1. Today a rule that raises leaves its files,
and Snakemake removes the declared ones and keeps the undeclared (`stitching/<sample>/fovs.csv`,
the morphology images).

**No partial output (atomic writes).** Each file is written to a temporary sibling and renamed into
place (`os.replace`), as most writers of the package already do: `atomic_file` and `write_json`
(`_records.py`), which the records and the checkpoint tables use, `save_placement`
(`stitching/_persist.py`, which replaces the placement folder as a whole) and `write_ome_tiff`
(`io/_ome.py`). The implementation checks every writer of a declared output and makes any other
atomic (OP6). The task record is written last: it is the commit point, and a downstream command
reads nothing whose producer has no record that says `succeeded`.

### Choice: how a task publishes its outputs

| Option | Cost | Consequence |
| --- | --- | --- |
| A. In place, each file atomic, the task record last, cleanup on failure (recommended) | Every writer of a declared output writes atomically | Outputs land where `_layout` puts them; a reader never sees a half-written file; the record decides completion |
| B. A staging folder per task, then a rename of each file | Each API entry would write to another root, which its records then name | The records of `run.json` and `export.json` name paths relative to their own folders, which a move breaks; the rename is per file anyway |
| C. Rely on the engine's cleanup | None | Nextflow does not remove files written outside its work folder; undeclared files stay |

Recommendation: A.

### Choice: the task record

| Option | Cost | Consequence |
| --- | --- | --- |
| A. One task record per task, `workflow/tasks/<task>.json`, plus the `run.json` additions for the FOV kinds (recommended) | One more file per task | One completion rule for every kind; the stage records keep their formats; resume and the report read `run.json` |
| B. The additions in each stage's own record (`placement.json`, `export.json`, `segmentation.json`, ...) | Every record format changes | The records' hashes, which later stages record (`record_sha256`), would change with the execution context |

Recommendation: A.

## Checkpoints and resume

**Defaults (D8).** The `fov` task runs `FOV.run` with `CheckpointConfig(stages=<processing.checkpoints>,
overwrite=True)`, as `_rule_checkpoints` does today (`workflow.py:767`), with the default stages
`candidates` and `pre_qc` (today `{stages: []}`, O5). The plan adds `registered` when a
segmentation input names a sequencing round (the `segment` task reads that round's registered
image); the tiled record decides how its global task's chunked checkpoint relates to `registered`.
A stage that the configured operations never reach (no spot finding: no `candidates`) is simply
not written, as `FOV.run` does today.

**A failed task reruns from scratch.** The engine retries a task that exited 1; the command
removes its earlier declared outputs first and `FOV.run` with `overwrite=True` removes its earlier
stage files (`checkpoints.md`, "Turn checkpoints on").

**Verified resume.** With `execution.resume_from_checkpoint: true` the `fov` command resumes from
the latest stage the earlier attempt wrote, only when the earlier `run.json` matches. It compares:

| Field of `run.json` (version 2) | Must equal |
| --- | --- |
| `format_version` | 2 (a version-1 record is never resumed, as `checkpoints.md` says) |
| `dataset_id`, `sample_id`, `fov_id`, `subtile_id` | The task's identity |
| `workflow.configuration_sha256` (new) | The manifest's resolved-configuration hash |
| `code.version`, `code.git_commit` (when recorded) | The running package's |
| `inputs[].path`, `inputs[].sha256` | The task's inputs, hashed again (the full hashes are computed inside the task) |
| `codebook.path`, `codebook.sha256` | The codebook the task loads |
| `preprocessing.supplied_statistics.sha256` | The supplied file the task reads |
| `checkpoints.<stage>.sha256` | The SHA-256 of the stage record on disk; and each payload file the stage record lists (`files`) has its recorded SHA-256 |

It then loads the stage (`FOV.load_checkpoint`, which checks the round labels, channel order,
readout mode and encoding itself) and runs `FOV.run` with the part of the `PipelineConfig` after
that stage. A mismatch refuses, with exit 3 and a message that names every field that differs, for
example:

```text
resume_from_checkpoint refused for fov/sample/FOV_001: fovs/FOV_001/run.json does not match this task:
workflow.configuration_sha256 is sha256:3f1c… (recorded), sha256:9a0e… (now); inputs round2/FOV_001/ch01.tif
sha256 differs; rerun without resume_from_checkpoint, or restore the configuration and inputs it was made with
```

Nothing verifies a resume against `run.json` today (`checkpoints.md`, "Reload and continue").
Resume applies to the `fov` kind (and to the tiled global task, by the tiled record); every other
kind reruns from scratch.

### Choice: what a resume must match

| Option | Cost | Consequence |
| --- | --- | --- |
| A. The table above, the package version included (recommended) | A package update forces a full rerun | A resumed result is the result of one package version and one configuration |
| B. Without the package version | Faster recovery after an update | A stage written by another version is continued by this one; a changed method would mix two versions in one result |

Recommendation: A.

## Logging and records

**The per-task log.** `workflow/logs/<task>.log` (`_layout.task_log`), one per task, rewritten by
each attempt with the attempt number in its first line. The command attaches a file handler to the
`starfinder` logger, which has none today (`dataset/_logging.py`; `design/notes/workflow-history.md`,
"Logs, time and memory"), at level `INFO` (`execution.log_level` sets `DEBUG`), and captures Python warnings.
One line per record:

```text
2026-10-29T18:17:51.123Z INFO starfinder.dataset fov/sample/FOV_001 [FOV_001] Completed register in 1.92s
```

The first lines give the task id, the command line, the execution context, the package version and
the two hashes; on failure the last lines hold the traceback. The engine's own job log (its stdout
and stderr) stays the engine's.

**The task record**, `workflow/tasks/<task>.json`: `format` (`starfinder.workflow.task/1`), `id`,
`kind`, `identity`, `status`, `started_at`, `ended_at`, `outputs` (path and SHA-256 of each declared
output), and the section `workflow` below.

**The `run.json` additions (D8)**, a new key `workflow` of the `fov` and `tile_merge` records, also
written into every task record:

| Field | Content |
| --- | --- |
| `task` | The task id and kind |
| `backend` | `python`, and the device the API ran on (`cpu` or `cuda`) |
| `fallback` | null, or `{requested, used, reason}` when the task ran on another device than the configuration requested (no entry of the package falls back today, so null) |
| `configuration_sha256` | The resolved-configuration hash |
| `manifest_sha256` | The manifest's hash |
| `execution_context` | `engine`, `engine_version`, `job_id`, `attempt`, `image` (by digest), `host` |
| `wall_s`, `cpu_s`, `peak_rss_kib` | The task's wall time, user and system CPU time of the process and its children, and peak resident memory (`resource.getrusage`), as `export.json` records them for its parts |

`workflow_config` (today the path and SHA-256 of `config_path`, `checkpoints.md`) records the
resolved configuration (`workflow/resolved.json`, relative to the output root, and its SHA-256).

**Additive within version 2.** The key is optional: a record without it still loads, as
`channels`, `workflow_config` and `outputs` were added (`checkpoints.md`, "run.json"). The readers
follow without a change: `dataset/_run_record.py` (`READ_VERSIONS = (1, 2)`), the report's
collector (`report/_collect.py`, `_check_run_records`, which checks the version and identity only)
and its replay (`report/_replay.py`, `replay_fov`). The report may later add the task's `wall_s`,
`cpu_s` and `peak_rss_kib` to its step table (`report/_collect.py` reads `wall_s` of the export's
parts today); that is a report change for its own issue.

### Choice: the log format

| Option | Cost | Consequence |
| --- | --- | --- |
| A. Plain text, one line per record, UTC times (recommended) | None | Readable in a terminal and in a UGER log viewer; D8 excludes a JSONL event stream |
| B. logfmt (`key=value`) | A formatter | Machine-parsable without being a JSON stream |

Recommendation: A; the machine-readable facts are in the task record.

### Choice: the version of the `run.json` additions

| Option | Cost | Consequence |
| --- | --- | --- |
| A. Additive within version 2 (recommended) | None for readers | A version-2 record with or without `workflow` loads; verified resume needs `workflow.configuration_sha256`, so a record without it is not resumed |
| B. Version 3 | Every reader accepts 3; the replay and the report change their version lists | The version says whether the workflow section exists; old records stay readable |

Recommendation: A.

### Choice: what `backend` and `fallback` record

| Option | Cost | Consequence |
| --- | --- | --- |
| A. The workflow backend and the device, and a device fallback (recommended) | None today: no entry falls back | Ready for a learned method that falls back from `cuda` to `cpu` |
| B. Also each method's own fallbacks (the registration recovery alternatives, the stitching `fallback: nominal`) | A summary of other records | Duplicates `run.json` `registration` (each attempt records `fallback`, `dataset/fov.py`) and `placement.json` |

Recommendation: A; the decision's wording leaves the meaning to this record, so the worker notes
list it for the gate.

## The two-pass statistics tasks

A recipe step with `fit: supplied` reads its statistics from a supplied file
(`PreprocessingRecipe.supplied_statistics`, `preprocessing/steps.py`). When the configuration names
no file, the plan adds the passes `preprocessing-algorithms.md` ("Recipe stage of a summary")
requires, through the summary and merge API of W-230 (`preprocessing/histograms.py`,
`preprocessing/supplied.py`):

1. **Summarize, per FOV and supplied step.** `summary_recipe(recipe, step)` gives the recipe of the
   steps before it; the command runs `FOV.run` with that recipe (load and preprocess, no
   registration or detection), then `summarize_histograms` of each round with its
   `summarized_after`, and `save_histograms` to `statistics/<sample>/<step>/histograms/<fov>.npz`.
2. **Merge, per sample and step (the barrier).** After every FOV of the sample: `load_histograms`,
   `merge_histograms` (with the FOVs a configuration excludes), `fit_supplied_section` for the step,
   `supplied_statistics` with the sections of the earlier steps, and `save_supplied_statistics` to
   `statistics/<sample>/<step>/supplied.json`.
3. **A later supplied step** is summarized with the earlier step's file as its recipe's
   `supplied_statistics`, so its `summarize` tasks need the earlier `merge`.
4. **Processing.** Each `fov` (or `tile_global`) task of the sample passes the last file as
   `supplied_statistics`; `run.json` records it with its SHA-256 (`preprocessing.supplied_statistics`).

The scalar-background shortcut (a summary derived from the earlier one without another pass) is an
optimization of the merge command; the graph does not depend on it.

### Choice: the scope of a statistics file

| Option | Cost | Consequence |
| --- | --- | --- |
| A. Per sample (the `sample` of `fovs.csv`) (recommended) | One barrier per sample | "Sample-level statistics" (§2.5) mean the sample's FOVs; samples run independently |
| B. Per run (every selected FOV) | One barrier for the run | Samples share statistics; one sample's edge FOVs change another's normalization |

Recommendation: A; a configuration that wants one file for the run supplies it.

## The MATLAB legacy boundary

`backend: matlab` runs the frozen MATLAB workflow (D7): its rules, scripts, keys and files are
unchanged ([the frozen surface](../baselines/orchestration.md#the-matlab-legacy-surface-frozen-by-d7)),
it keeps its own enumeration (the MATLAB branch of today's `common.smk:236-279`: the sample
annotation, `fov_id_pattern`, the subset keys) and its parse-time definitions, and the adapters of
its shared rules move unchanged into one private legacy module that the Python workflow never
imports, pinned by `test_assignment_workflow.py` and `test_workflow_assembly_golden.py`. The plan
command refuses `backend: matlab` (exit 2) with a message that names the legacy workflow's
entry; it writes no manifest for it, and Nextflow does not run it (D1).

### Choice: the plan for `backend: matlab`

| Option | Cost | Consequence |
| --- | --- | --- |
| A. Refuse, naming the legacy entry (recommended) | None | The legacy workflow stays as it is; the contract covers one dialect |
| B. Write a manifest of the MATLAB rules | A planner of the frozen rules | The MATLAB enumeration would be reimplemented, which D7 freezes |

Recommendation: A.

## What an engine needs, per task kind

Every input of a task comes from the manifest or from the command line; no engine file imports the
package. The engine reads, for every task: `id`, `kind`, `needs`, `inputs`, `outputs`, `record`,
`log`, `command`, `execution`; it passes `--engine`, `--job-id`, `--attempt`, `--image`.

(orchestration-engine-inputs)=

| Task kind | What the command reads | From where | What the engine adds |
| --- | --- | --- | --- |
| `summarize` | the FOV's raw images, the earlier supplied file, the step and its recipe | `inputs`; the resolved configuration through the manifest | nothing |
| `merge` | the histogram files of the sample, the earlier supplied file | `inputs` | nothing |
| `fov` | the raw images, the codebook, the supplied file, the processing block, the stages | `inputs`; the resolved configuration | nothing |
| `tile_global`, `tile`, `tile_merge` | the raw images, the chunked checkpoint, the tile's window and halo, the tiles' outputs | `inputs`, `identity.tile`; the window and halo by W-644 | nothing: the tiles are tasks of the manifest |
| `morphology` | the stain and other rounds' files, the morphology block, `rotate_angle` | `inputs`; the resolved configuration | nothing |
| `segment` | the reference image or saved forms, the seeds' run, the import file, the run | `inputs`, `identity.run`; the resolved configuration | a GPU when `execution.gpu` says so |
| `assign` | the molecules, the reference image, the codebook, the saved runs, the block | `inputs`; the resolved configuration | nothing |
| `bigstitcher_macro` | the stitching images, the grids, the position file, the block | `inputs`; the resolved configuration | nothing |
| `bigstitcher_engine` | the macro, the tiles | `inputs`; the command line in `command` with `execution.tool` | `{fiji}` replaced by `execution.tool.path`; Snakemake only (D3) |
| `bigstitcher_import`, `stitch` | `dataset.xml`, the view map, the macro, the grids, the position file, the block | `inputs`; the resolved configuration | nothing |
| `export` | the placement, the route's per-FOV files, the fusion's sources, the blocks | `inputs`; the resolved configuration | a GPU when `execution.gpu` says so (the route after fusion on `cuda`) |
| `report` | the run records and stages, the placement, `export.json`, the assignments, the block | `inputs`; the resolved configuration | nothing |
| `compare` | every `report.json` | `inputs` | nothing |

The checklist per task kind. Each column is a question an engine design answers "from the
manifest" for every kind; a cell that is not "manifest" names the exception:

| Task kind | Identity | Paths | Parameters | Resources | Command line | Package import in the engine file |
| --- | --- | --- | --- | --- | --- | --- |
| `summarize`, `merge` | manifest | manifest | resolved configuration | manifest | manifest | none |
| `fov` | manifest | manifest | resolved configuration | manifest | manifest | none |
| `tile_global`, `tile`, `tile_merge` | manifest (the tile list is the plan's, W-644) | manifest | resolved configuration | manifest | manifest | none |
| `morphology` | manifest | manifest | resolved configuration | manifest | manifest | none |
| `segment` | manifest | manifest | resolved configuration | manifest (`gpu` from the run's `device`) | manifest | none |
| `assign` | manifest | manifest | resolved configuration | manifest | manifest | none |
| `bigstitcher_macro`, `bigstitcher_import`, `stitch` | manifest | manifest | resolved configuration | manifest | manifest | none |
| `bigstitcher_engine` | manifest | manifest | the macro the `bigstitcher_macro` task wrote | manifest | manifest, with `{fiji}` from `execution.tool.path` | none |
| `export`, `report`, `compare` | manifest | manifest | resolved configuration | manifest | manifest | none |

The engine's context is passed as four arguments and only recorded. The prototype manifests of the
worked example hold every one of these for every task.

## Traceability

Every item of W-642's [classification](../baselines/orchestration.md#the-snakemake-adapters-and-the-shared-definitions)
(runners, planners, writers, validation, input enumeration) and the contract element that replaces
it, or the MATLAB legacy module or frozen MATLAB workflow (D7):

% Generated and checked by scripts/traceability.py (results/traceability/traceability.md, check.json)

(orchestration-traceability)=

| Definition | Where | Class | Replaced by |
| --- | --- | --- | --- |
| `_run_workflow` | `_snakemake.py:28` | runner | The `fov` command (direct); `tile-global`, `tile`, `tile-merge` (tiled, D5); the resident-subtile branch is retired (D5) |
| `_run_nuclei_registration` | `_snakemake.py:73` | runner | The `morphology` command |
| `_read_stack` | `_snakemake.py:141` | runner | The private MATLAB legacy module, unchanged (D7) |
| `_run_nuclei_amplicon_overlay` | `_snakemake.py:148` | runner | The private MATLAB legacy module, unchanged (D7); for Python, a segmentation input with `prepare: composite_nuclei_amplicon` of the `segment` command (D6) |
| `_run_enhance_dapi_with_flamingo` | `_snakemake.py:168` | runner | The private MATLAB legacy module, unchanged (D7); for Python, `prepare: enhance_with_flamingo` of the `segment` command (D6) |
| `_run_stardist_segmentation` | `_snakemake.py:178` | runner | The `segment` command (method `stardist`); The private MATLAB legacy module, unchanged (D7) for MATLAB's shared rule |
| `_fov_sample` | `_snakemake.py:227` | input enumeration | The manifest's `membership` and each task's identity `sample`; The private MATLAB legacy module, unchanged (D7) for the annotation branch |
| `_tile_record` | `_snakemake.py:254` | writer | The private MATLAB legacy module, unchanged (D7) |
| `_legacy_cells` | `_snakemake.py:264` | runner | The private MATLAB legacy module, unchanged (D7) |
| `_assembly_cells` | `_snakemake.py:287` | runner | The `assign` command (the saved cell run on the reference grid) |
| `_in_box` | `_snakemake.py:318` | writer | The private MATLAB legacy module, unchanged (D7) |
| `_cell_obs` | `_snakemake.py:324` | writer | The private MATLAB legacy module, unchanged (D7) |
| `_write_raw_h5ad` | `_snakemake.py:359` | writer | The private MATLAB legacy module, unchanged (D7) |
| `_write_reads_assignment` | `_snakemake.py:390` | writer | The private MATLAB legacy module, unchanged (D7) |
| `_assignment_ratio` | `_snakemake.py:447` | writer | The private MATLAB legacy module, unchanged (D7) |
| `_write_assignment_figures` | `_snakemake.py:455` | writer | The previews of the `assign` and `export` commands (P5); The private MATLAB legacy module, unchanged (D7) |
| `_run_reads_assignment` | `_snakemake.py:483` | runner | The `assign` command; The private MATLAB legacy module, unchanged (D7) for its legacy route |
| `_reads_assignment_outputs` | `_snakemake.py:560` | planner | The manifest: the declared outputs of `assign` |
| `_stitching_position_inputs` | `_snakemake.py:580` | planner | The manifest: the declared inputs of `stitch`, `bigstitcher_macro`, `bigstitcher_import` |
| `_image_dir_inputs` | `_snakemake.py:597` | planner | The plan's input discovery (`inputs.images`) and the declared inputs of `fov`, `tile_global`, `summarize`, `morphology` |
| `_reference_image_inputs` | `_snakemake.py:609` | planner | The manifest: the declared inputs of `segment`, `assign`, `stitch`, `bigstitcher_macro`, `bigstitcher_import`, `export` |
| `_reference_image_outputs` | `_snakemake.py:625` | planner | The manifest: the declared outputs of `fov`, `tile_global` |
| `_run_record_outputs` | `_snakemake.py:637` | planner | The manifest: the declared outputs of `fov`, `tile_merge` |
| `_assignment_record_outputs` | `_snakemake.py:648` | planner | The manifest: the declared outputs of `assign`; the declared inputs of `export` on the route per FOV |
| `_shared_image_inputs` | `_snakemake.py:654` | planner | The manifest: the declared inputs of `segment` (the saved forms of `morphology`); The private MATLAB legacy module, unchanged (D7) |
| `_bigstitcher_folder` | `_snakemake.py:679` | planner | The manifest: the declared outputs of `bigstitcher_macro`, `bigstitcher_engine` |
| `_write_tile_config` | `_snakemake.py:686` | writer | The `bigstitcher-import` command, which writes the placement only: the tile configuration's one reader, the legacy route of `reads_assignment`, is retired (D6) |
| `_read_view_map` | `_snakemake.py:702` | runner | The `bigstitcher-import` command |
| `_run_stitching` | `_snakemake.py:707` | runner | The `stitch` command; The `bigstitcher-import` command (D3) |
| `_run_bigstitcher_macro` | `_snakemake.py:756` | runner | The `bigstitcher-macro` command (D3) |
| `_check_bigstitcher_engine` | `_snakemake.py:781` | validation | The `bigstitcher-import` command: it checks `dataset.xml` first (D3) |
| `_run_bigstitcher_engine` | `_snakemake.py:814` | runner | The task kind `bigstitcher_engine` of the manifest: the engine runs `execution.fiji` (D3); the check moves to `bigstitcher-import` |
| `_export_sample_files` | `_snakemake.py:834` | planner | The manifest: the declared inputs and outputs of `export`, by route |
| `_export_preview_outputs` | `_snakemake.py:873` | planner | The manifest: the declared outputs of `export` (`export.previews`, P5) |
| `_run_export` | `_snakemake.py:888` | runner | The `export` command |
| `_write_ownership_figures` | `_snakemake.py:931` | writer | The `export` command (`export.previews`, P5) |
| `_report_sample_files` | `_snakemake.py:959` | planner | The manifest: the declared inputs and outputs of `report` |
| `_report_dataset_files` | `_snakemake.py:979` | planner | The manifest: the declared inputs and outputs of `compare` |
| `_run_report_sample` | `_snakemake.py:986` | runner | The `report` command |
| `_run_report_dataset` | `_snakemake.py:1001` | runner | The `compare` command |
| `sys.path.insert` (the rules folder) | `common.smk:16` | runner | The frozen MATLAB workflow, unchanged (D7) |
| `BACKEND` | `common.smk:21` | planner | The resolved configuration's `backend` (default `python`); The frozen MATLAB workflow, unchanged (D7) keeps the switch to its rules |
| `raise` (unknown backend) | `common.smk:23` | validation | The plan command's validation (V1) |
| `STITCHING_BLOCK` | `common.smk:27` | planner | The plan's task graph: a present block enables its step (D4) |
| `ASSEMBLY_BLOCK` | `common.smk:28` | planner | The plan's task graph (D4) |
| `LEGACY_STITCHING_RULES` | `common.smk:29` | planner | The plan command's validation (V2: MATLAB-dialect keys, D7); The frozen MATLAB workflow, unchanged (D7) |
| `LEGACY_SAMPLE_RULES` | `common.smk:31` | planner | The plan command's validation (V2); The frozen MATLAB workflow, unchanged (D7) |
| `REPORT_BLOCK` | `common.smk:33` | planner | The plan's task graph (D4) |
| `WORKFLOW_MODE` | `common.smk:38` | planner | The plan's task graph and `processing.mode` (D4, D5); The frozen MATLAB workflow, unchanged (D7) |
| `WORKFLOW_PRESETS` | `common.smk:41` | planner | The plan's task graph (D4); The frozen MATLAB workflow, unchanged (D7) |
| `validate_workflow_mode_dependencies` | `common.smk:57` | validation | The plan command's validation (V2, V3); The frozen MATLAB workflow, unchanged (D7) |
| `validate_workflow_mode_dependencies()` | `common.smk:92` | validation | The plan command's validation (V2, V3); The frozen MATLAB workflow, unchanged (D7) |
| `ALWAYS_AVAILABLE_RULES` | `common.smk:96` | planner | The plan's task graph (D4); The frozen MATLAB workflow, unchanged (D7) |
| `is_rule_enabled` | `common.smk:111` | planner | The plan's task graph (D4); The frozen MATLAB workflow, unchanged (D7) |
| `get_rule_config` | `common.smk:129` | planner | The resolved configuration's `execution` and each task's `execution` in the manifest (D4) |
| `DEFAULT_RESOURCES` | `common.smk:156` | planner | The resolved configuration's `execution.defaults` (D4) |
| `yaml_to_json` | `common.smk:163` | writer | The frozen MATLAB workflow, unchanged (D7) |
| `run_matlab_scripts` | `common.smk:175` | runner | The frozen MATLAB workflow, unchanged (D7) |
| `get_additional_round_names` | `common.smk:184` | input enumeration | The plan's input discovery (other rounds); The frozen MATLAB workflow, unchanged (D7) |
| `make_get_runtime` | `common.smk:187` | planner | Each task's `execution.runtime_min` in the manifest; retries and their resources are the engine's (D8) |
| `INPUT_DIR` | `common.smk:199` | input enumeration | The manifest's `roots.input` (`_layout`) |
| `OUTPUT_DIR` | `common.smk:200` | planner | The manifest's `roots.output` (`_layout`) |
| `DAPI_DIR` | `common.smk:201` | planner | The frozen MATLAB workflow, unchanged (D7); for Python `rotate_nuclei` is retired (D6) |
| `FUSE_DIR` | `common.smk:202` | planner | The frozen MATLAB workflow, unchanged (D7) (never read, O20) |
| `SEG_INPUT_DIR` | `common.smk:203` | planner | The inputs of a segmentation run (`segment`, D6); The frozen MATLAB workflow, unchanged (D7) |
| `DOC_DIR` | `common.smk:204` | planner | The frozen MATLAB workflow, unchanged (D7) |
| `ROUND` | `common.smk:206` | input enumeration | The plan's input discovery (sequencing rounds) |
| `sys.path.insert` (the package) | `common.smk:212` | runner | Each task's `command` in the manifest: the installed `starfinder` command (D2, D9) |
| `MEMBERSHIP` | `common.smk:215` | input enumeration | The plan's input discovery (`fovs.csv`) |
| `_MEMBERS` | `common.smk:216` | input enumeration | The manifest's `membership` |
| `SAMPLE_TO_FOV` | `common.smk:217` | input enumeration | The manifest's `membership.samples` |
| `FOV_TO_SAMPLE` | `common.smk:218` | input enumeration | Each task's identity `sample` |
| `SAMPLE` | `common.smk:219` | input enumeration | The manifest's `membership.samples` |
| `FOVS` | `common.smk:220` | input enumeration | The manifest's `membership.fovs` |
| `FOV_CONSTRAINT` | `common.smk:224` | input enumeration | The manifest's task identities: an engine runs the tasks it lists |
| `CODEBOOK` | `common.smk:227` | input enumeration | The plan's input discovery (`inputs.files.codebook`) |
| `_PYTHON_INPUT_KEYS` | `common.smk:231` | validation | The schema's Python-only clause (OC5); The plan command's validation (V4) |
| `raise` (Python-only input keys) | `common.smk:234` | validation | The schema's Python-only clause (OC5); The plan command's validation (V4) |
| `CODEBOOK` | `common.smk:236` | input enumeration | The frozen MATLAB workflow, unchanged (D7) |
| `MEMBERSHIP` | `common.smk:237` | input enumeration | The frozen MATLAB workflow, unchanged (D7) |
| `SAMPLE_ANNOT` | `common.smk:238` | input enumeration | The frozen MATLAB workflow, unchanged (D7) |
| `SAMPLE_TO_FOV` | `common.smk:239` | input enumeration | The frozen MATLAB workflow, unchanged (D7) |
| `FOV_TO_SAMPLE` | `common.smk:240` | input enumeration | The frozen MATLAB workflow, unchanged (D7) |
| `for current_sample` | `common.smk:241` | input enumeration | The frozen MATLAB workflow, unchanged (D7) |
| `SAMPLE` | `common.smk:254` | input enumeration | The frozen MATLAB workflow, unchanged (D7) |
| `FOVS` | `common.smk:255` | input enumeration | The frozen MATLAB workflow, unchanged (D7) |
| `N_SUBTILE` | `common.smk:260` | planner | The tile tasks of the manifest (D5, W-644); The frozen MATLAB workflow, unchanged (D7) |
| `N_SUBTILE` | `common.smk:262` | planner | The tile tasks of the manifest (D5, W-644); The frozen MATLAB workflow, unchanged (D7) |
| `N_SUBTILE` | `common.smk:264` | planner | The tile tasks of the manifest (D5, W-644); The frozen MATLAB workflow, unchanged (D7) |
| `config.setdefault` (`subset_range`) | `common.smk:271` | validation | The frozen MATLAB workflow, unchanged (D7) |
| `config.setdefault` (`subset_random`) | `common.smk:272` | validation | The frozen MATLAB workflow, unchanged (D7) |
| `FOVS` (`subset_list`) | `common.smk:274` | input enumeration | The frozen MATLAB workflow, unchanged (D7) |
| `SAMPLE` (`subset_list`) | `common.smk:275` | input enumeration | The frozen MATLAB workflow, unchanged (D7) |
| `FOVS` (`subset_range`) | `common.smk:277` | input enumeration | The frozen MATLAB workflow, unchanged (D7) |
| `FOVS` (`subset_random`) | `common.smk:279` | input enumeration | The frozen MATLAB workflow, unchanged (D7) |
| `get_overall_output` | `common.smk:283` | planner | The manifest's task list: every task with its declared outputs |
| `rsf_preparation` | `common.smk:392` | writer | The frozen MATLAB workflow, unchanged (D7) |
| `MOLECULE_TABLE` | `common.smk:404` | planner | The frozen MATLAB workflow, unchanged (D7) |
| `SHARED_IMAGE_INPUTS` | `common.smk:408` | planner | The frozen MATLAB workflow, unchanged (D7) |
| `_check_python_dialect(config)` | `common.smk:417` | validation | The plan command's validation (V5) |
| `for python_rule` | `common.smk:418` | validation | The plan command's validation (V5: the processing block translated) |
| `_check_sample_rules(config)` | `common.smk:427` | validation | The plan command's validation (V2, V3) |
| `MOLECULE_TABLE` | `common.smk:428` | planner | The manifest: the declared outputs of `fov`, `tile_merge` (`molecules.csv`); the declared inputs of `assign` and `export` |
| `RUN_RECORD` | `common.smk:435` | planner | The manifest: the declared outputs of `fov`, `tile_merge` |
| `SUBTILE_RUN_RECORD` | `common.smk:436` | planner | The manifest: the declared outputs of `tile` (D5, W-644) |
| `REFERENCE_IMAGE_OUTPUT` | `common.smk:437` | planner | The manifest: the declared outputs of `fov`, `tile_global` |
| `REFERENCE_IMAGE_INPUT` | `common.smk:438` | planner | The manifest: the declared inputs of `segment`, `assign`, `stitch`, `export` |
| `SHARED_IMAGE_INPUTS` | `common.smk:439` | planner | The manifest: the declared inputs of `segment` (the saved forms of `morphology`) |
| `_stitching_block(config)` | `common.smk:441` | validation | The plan command's validation (V6) |
| `_stage_position_ids(...)` | `common.smk:442` | validation | The plan command's validation (V6) |
| `_export_config(config)` | `common.smk:444` | validation | The plan command's validation (V7) |
| `_check_export_route(config)` | `common.smk:445` | validation | The plan command's validation (V7) |
| `_report_block(config)` | `common.smk:447` | validation | The plan command's validation (V8) |

## Worked example

The manifest of the synthetic fixture W-642 used (`sample-tiny-v1`, seed 42: one sample of four
FOVs, four rounds of four channels, the reference stain `ch04` and the additional round
`flamingo`), produced by the prototype of the plan command (`scripts/plan_prototype.py`, scratch
code of the run directory that calls `_layout`, `_membership` and the block translators). Five
variants cover every task kind and each export route; two more show the refusals:

| Variant | The blocks | Tasks | Kinds |
| --- | --- | --- | --- |
| `per-fov` | `processing` (direct, one `percentile_normalization` step with `fit: supplied`), `morphology`, `segmentation` (two imported runs, `cell` and `nucleus`), `assignment` (with nuclei and previews), `stitching` (`multiview_stitcher`), `assembly`, `export` (`route: per_fov`, fusing `reference_merged` and `reference_stain`, previews), `report` | 29 | `summarize` 4, `merge` 1, `fov` 4, `morphology` 4, `segment` 8, `assign` 4, `stitch` 1, `export` 1, `report` 1, `compare` 1 |
| `after-fusion` | `processing` (`mode: tiled`, four tiles as a placeholder for W-644), `morphology`, `stitching` (`bigstitcher` on a regular grid), `assembly`, `export` (the default route of a projection, one imported cell run), `report` | 34 | `tile_global` 4, `tile` 16, `tile_merge` 4, `morphology` 4, `bigstitcher_macro` 1, `bigstitcher_engine` 1, `bigstitcher_import` 1, `export` 1, `report` 1, `compare` 1 |
| `molecules-only` | `processing`, `stitching`, `assembly` (`assignment: null`), `export`, `report` | 8 | `fov` 4, `stitch` 1, `export` 1, `report` 1, `compare` 1 |
| `morphology-only` (D4) | `morphology`, `segmentation` (a StarDist nucleus run on the reference stain, `device: cuda`) | 8 | `morphology` 4, `segment` 4 (each `gpu: true`, needing only its FOV's `morphology`) |
| `reference-only` (D4) | `processing` without `spot_finding` (the reference image), `morphology`, `segmentation` (an imported nucleus run and a seeded watershed on the composite of the reference stain and the reference merged image) | 16 | `fov` 4, `morphology` 4, `segment` 8 |
| `refused-assignment` | `morphology`, `segmentation`, `assignment`, no `processing` | — | Refused, exit 2: "the assignment block assigns each FOV's molecules, which only the processing block produces (with spot_finding, reads_extraction and reads_filtration): add a processing block" |
| `refused-matlab` | No `backend`, `workflow_mode` and `rules` | — | Refused, exit 2, with the D7 message above |

Between them the variants hold every task kind and the three export routes. The `per-fov`
manifest's tasks run in this order of `needs`: the four `summarize` tasks, their `merge` (the
barrier), the four `fov` tasks, the `morphology` and `segment` tasks, the four `assign` tasks, the
`stitch` task (after the four `fov` tasks), the `export` (after the `assign`, `morphology` and
`stitch` tasks), the `report` and the `compare`.

% results/example/configs/ (each variant in the proposed shape, and current-<variant>.yaml in today's dialect for
% the three variants it can express), results/example/<variant>/resolved.json, resolved.sha256 and manifest.json
% (steps plan_<variant>.rerun-5)

Four tasks from the variants' manifests (`results/example/<variant>/manifest.json`), shortened where
marked `…`; each manifest holds every field for every task:

```text
{"id": "merge/sample/percentile_normalization", "kind": "merge",
 "identity": {"sample": "sample", "step": "percentile_normalization"},
 "needs": ["summarize/sample/percentile_normalization/FOV_001", "summarize/sample/percentile_normalization/FOV_002",
           "summarize/sample/percentile_normalization/FOV_003", "summarize/sample/percentile_normalization/FOV_004"],
 "inputs": ["statistics/sample/percentile_normalization/histograms/FOV_001.npz", "…/FOV_002.npz", "…/FOV_003.npz",
            "…/FOV_004.npz"],
 "outputs": ["statistics/sample/percentile_normalization/supplied.json"],
 "record": "workflow/tasks/merge.sample.percentile_normalization.json",
 "log": "workflow/logs/merge.sample.percentile_normalization.log",
 "command": ["starfinder", "workflow", "merge", "--manifest", "<output root>/workflow/manifest.json",
             "--task", "merge/sample/percentile_normalization"],
 "execution": {"threads": 1, "memory_mb": 8000, "runtime_min": 30, "gpu": false, "container": null}}
{"id": "assign/sample/FOV_001/default", "kind": "assign",
 "identity": {"sample": "sample", "fov": "FOV_001", "name": "default"},
 "needs": ["fov/sample/FOV_001", "segment/sample/FOV_001/cell", "segment/sample/FOV_001/nucleus"],
 "inputs": ["fovs/FOV_001/molecules.csv", "fovs/FOV_001/reference.tif",
            "fovs/FOV_001/segmentation/cell/labels.tif", "fovs/FOV_001/segmentation/cell/segmentation.json",
            "fovs/FOV_001/segmentation/nucleus/labels.tif", "fovs/FOV_001/segmentation/nucleus/segmentation.json",
            "input:genes.csv"],
 "outputs": ["fovs/FOV_001/assignment/default/assignment.json", "fovs/FOV_001/assignment/default/cells.csv",
             "fovs/FOV_001/assignment/default/counts.csv", "fovs/FOV_001/assignment/default/molecules.csv",
             "fovs/FOV_001/assignment/default/nuclei.csv", "fovs/FOV_001/previews/"], "…": "…"}
{"id": "segment/sample/FOV_001/nucleus", "kind": "segment", "…": "the morphology-only variant",
 "needs": ["morphology/sample/FOV_001"],
 "inputs": ["fovs/FOV_001/other_rounds/reference_stain/image.ome.tif",
            "fovs/FOV_001/other_rounds/reference_stain/registration.json"],
 "outputs": ["fovs/FOV_001/segmentation/nucleus/input.ome.tif", "fovs/FOV_001/segmentation/nucleus/labels.tif",
             "fovs/FOV_001/segmentation/nucleus/segmentation.json"],
 "execution": {"threads": 1, "memory_mb": 8000, "runtime_min": 30, "gpu": true, "container": null}}
{"id": "bigstitcher_engine/sample", "kind": "bigstitcher_engine", "…": "the after-fusion variant",
 "needs": ["bigstitcher_macro/sample"],
 "inputs": ["<save dir>/sample/BigStitcher_macro.ijm", "<save dir>/sample/tiles/", "<save dir>/sample/view_map.csv"],
 "outputs": ["<save dir>/sample/dataset.xml"],
 "command": ["{fiji}", "-Djava.io.tmpdir=<save dir>/sample", "--ij2", "--headless", "--run",
             "<save dir>/sample/BigStitcher_macro.ijm"],
 "execution": {"threads": 4, "memory_mb": 8000, "runtime_min": 30, "gpu": false, "container": null,
               "tool": {"name": "fiji", "path": "/stanley/WangLab/Tools/Fiji.app/ImageJ-linux64"}}}
```

The manifest's header of the `per-fov` variant: `format` `starfinder.workflow.manifest/1`,
`backend` `python`, `configuration` (`workflow/resolved.json`, its SHA-256 and the
resolved-configuration hash), `roots`, `membership` (`FOV_001` to `FOV_004` in sample `sample`,
`selected_fovs` null), `inputs` (for example `round1/FOV_001/ch00.ome.tif`, 27,931 bytes, with its
modification time; the stain `round1/FOV_001/ch04.tif` under the image `reference_stain`, which the
`morphology` task reads and the `fov` task does not) and `export_route` `per_fov`. The `<save dir>`
is `stitching.bigstitcher_save_dir`, outside the output root, so its paths are absolute; the
`{fiji}` placeholder is the one substitution an engine makes, from `execution.tool.path`
(`_run_bigstitcher_engine`, `_snakemake.py:814`, builds the same command line today).

## The exception of the golden comparison

The refactored workflow is compared with the golden baseline (W-642 and invocation D above) on the
migrated configurations, with the tolerances of `docs/contributing.md`. The comparison leaves out:

* **OC3.** The frame ids and every value derived from them: with `rotate_angle: 0` the rotation
  step is skipped, so `run.json` records no `rotate_round` step and every frame id loses
  `/rotate:0` (today `["sample-tiny-v1","sample","FOV_001","formed"]/reference/rotate:0`); the
  stored image metadata that holds a frame id (the reference image, the morphology images and
  saved forms, the label images, the fused images) and those files' hashes, and the records that
  hash them (`grid_sha256`, `record_sha256`, `assembly_sha256`); and the `rotate_round` rows of
  `run.json` `steps` and of the report's step table.
* **OC4.** The paths recorded relative to the input and output roots instead of absolute (O12, and
  `placement.json` above): the morphology images' frame ids and the placement's sources, and the
  digests of the files that hold them.
* **D6.** The outputs of retired rules: `images/DAPI/`, `images/overlay/`,
  `images/flamingo/enhanced_DAPI/` and, unless `morphology.channel_images` is on, the channel images.
* **D8.** The checkpoint stages of the default (the golden configurations state theirs, so the
  migrated ones state the same), and the new task records and logs.

Everything else is compared as W-642 and invocation D recorded it: the molecule tables, the counts
and statuses, the registration results, the assignment tables, the placement, the export tables and
the report's tables.

## Changed statements of the existing records

Each statement of `design/specs/workflows.md`, `workflow-configuration.md`,
`workflow-downstream.md`, `broad-uger.md` and `checkpoints.md` that this contract would change,
with the old and the proposed wording. The records are not edited by this batch; the
implementation issue that lands each part edits them.

(orchestration-changed-statements)=

| Record and place | Old wording | Proposed wording | Why |
| --- | --- | --- | --- |
| `workflows.md:6-8` | "Snakemake schedules whole FOVs or subtiles; load, rotate, enhance, register, detect, extract and filter are operations **inside** those jobs" | "An engine (Snakemake or Nextflow) runs the tasks of the plan's manifest: whole FOVs, or tiles of a FOV; load, rotate, enhance, register, detect, extract and filter are operations inside those tasks" | D2, D5 |
| `workflows.md:39-41` | "The membership file must exist **before parsing the workflow**, even for a dry run or sequencing-only job; the workflow reads it during startup" | "The membership file must exist before the plan command runs; the plan reads it" (Python); unchanged for MATLAB | D2 |
| `workflows.md:76-77` | "The MATLAB backend reads `INPUT/genes.csv` and the round-first layout only, and refuses the three keys." | "The MATLAB backend reads `INPUT/genes.csv` and the round-first layout only; the schema refuses the three keys without `backend: python`." | OC5 |
| `workflows.md:89` | "`FOV.rotate` if angle is nonzero" | "`FOV.run` rotates by `rotate_angle`; `0` is no rotation and the step is skipped" (today the adapter passes 0 as a rotation, O11) | OC3 |
| `workflows.md:109-111` | "The Python script writes them using `FOV.run` with checkpoints, `save_reference_image` and `export_spots`" | "The `fov` command writes them using `FOV.run` with checkpoints, `save_reference_image` and `export_spots`" | D2 |
| `workflows.md:117-121` | "Its file follows from the configuration, the backend and the producer the `ruleorder` of the `Snakefile` selects" | "Python: the plan names the producer, the `fov` or `tile_merge` task, and its file is `fovs/{fovID}/molecules.csv`; MATLAB: the `ruleorder` selects it as before" | D4, D5 (O18) |
| `workflows.md:127-131` | The `stitch_subtile` row for "either" backend and "The subtile and deep route stays legacy" | The row for MATLAB only; "For the Python backend the subtile route is retired; tiled execution writes the same files as direct" | D5 |
| `workflows.md:171-175` | "The Python rules take no JSON copy: their inputs are named (`codebook`, `images`, `subtile`), the adapters read them by name and read the configuration from `snakemake.config`" | "The Python tasks read the resolved configuration and their inputs from the manifest" | D2 |
| `workflows.md:179` | "`backend: matlab` is the default." | "`backend: python` is the default; a configuration without `backend` that holds MATLAB-dialect keys is refused, asking for `backend: matlab`." | D7 (O6) |
| `workflows.md:186-223` | "Nuclei registration": the Python rule `nuclei_registration` writes `images/<round>/<channel label>/<fov>.tif`; with `rules.nuclei_registration.parameters.checkpoints` it also writes the saved form; `enhance_dapi_with_flamingo` reads the label folders | "The `morphology` task writes `nuclei_registration.json` and the saved form of each prepared image (always), and the channel images with `morphology.channel_images`; nothing reads the channel images" | D6 (O8) |
| `workflows.md:225-230` | The modes table (`free`, `direct`, `subtile`, `deep`) | MATLAB only; "Python: `processing.mode` is `direct` or `tiled`; `deep` is a recipe of either (W-644)" | D4, D5 |
| `workflows.md:232-252` | Preset modes, `run` flags, the explicit target `all` and the `ruleorder` | MATLAB only; "Python: a present block enables its step; the plan derives the tasks" | D4 |
| `workflows.md:312-314` | "a direct one-FOV example schedules config preparation, `rsf_single_fov`, and `all`; a 2×2 subtile grid schedules one creation job, four processing jobs, a stitch job, preparation and `all`" | "On the MATLAB backend a direct one-FOV example schedules config preparation, `rsf_single_fov` and `all` (3 jobs) ...; on the Python backend the plan's manifest lists the tasks" | D7 (O13) |
| `workflows.md:327-330` | "The older `tests/minimal_config.yaml` and `tests/tissue_2D_test.yaml` remain historical starting templates, not ready-to-run portable examples. The former holds placeholder paths and needs a sample annotation at startup" | "`tests/minimal_config.yaml` is the schema fixture of the frozen MATLAB dialect, not a template: it does not parse (it has no `additional_round` and no sample annotation). `tests/tissue_2D_test.yaml` remains a historical MATLAB template" | OC2 (O4) |
| `workflow-configuration.md:15` | `config_path`: "absolute path to the same `.yaml` passed to `--configfile`" | "MATLAB backend only; the Python plan records the file it read" | D2 |
| `workflow-configuration.md:16` | `starfinder_path`: "repository root used to locate Python/MATLAB code" | "MATLAB backend only; the Python tasks run the installed `starfinder` command" | D2, D9 |
| `workflow-configuration.md:18` | `fiji_path`: "the Python BigStitcher engine rule reads it when it runs" | "Python: `execution.fiji`, the tool of the BigStitcher engine task" | D3 |
| `workflow-configuration.md:19` | `envs_path`: "kept in the schema until workflow orchestration" | Removed from the Python dialect | D6, D9 (O7) |
| `workflow-configuration.md:39` | `additional_round`: "supply `[]` when unused; accessed at parse time" | "optional for Python; read by the plan" | D2 |
| `workflow-configuration.md:40` | `backend`: default `matlab` | default `python` | D7 |
| `workflow-configuration.md:46` | `workflow_mode` | MATLAB backend only | D4 |
| `workflow-configuration.md:52` | `rules`, required by the schema: "each provided section requires boolean `run`" | MATLAB backend only; the Python dialect has blocks and `execution` | D4 |
| `workflow-configuration.md:95` | "raises `ValueError` when the rule files are parsed" (`fovs.csv`) | "the plan refuses it" | D2 |
| `workflow-configuration.md:213-225` | "`resources` is optional; if present, it accepts only `mem_mb` (integer >=1000, default 8000) and `runtime`" (each `rules.<name>` with its `run` flag) | Python: "`execution.defaults` and `execution.tasks.<kind>` give `threads`, `memory_mb`, `runtime_min` and `gpu`"; MATLAB unchanged | D4 (O15) |
| `workflow-configuration.md:270` | `checkpoints`: "absent or null is `{stages: []}`, `run.json` alone" | `processing.checkpoints`: "a list of stages, default `[candidates, pre_qc]`; the plan adds `registered` when a segmentation input reads a sequencing round" | D8 (O5) |
| `workflow-configuration.md:296-301` | "Each Python processing rule runs its FOV (or subtile) with `CheckpointConfig(stages=<stages>, overwrite=True)`"; "absent or null it is `{stages: []}`, which writes `run.json` alone" | "Each `fov` task runs its FOV with `CheckpointConfig(stages=<processing.checkpoints>, overwrite=True)`; the default stages are `candidates` and `pre_qc`" | D8 |
| `workflow-configuration.md:325` | "the schema refuses it with `backend: matlab` or without `backend` (whose default is `matlab`)" | "the schema refuses it with `backend: matlab`" | D7 |
| `workflow-configuration.md:616-617` | "`fiji_path` is read by `bigstitcher_engine` when it runs, not when the rules are parsed" | "`execution.fiji` is the tool of the `bigstitcher_engine` task, which only Snakemake runs; Nextflow refuses `method: bigstitcher` without `stitching.dataset_xml`" | D3 |
| `workflow-configuration.md:624-628` | A Python configuration that enables a legacy sample rule without its block "raises when the rule files are parsed, naming the blocks to add" | "a Python configuration has no rule names: the plan refuses `rules`, naming the blocks" | D4, D7 |
| `workflow-configuration.md:711-713` | "The blocks raise naming the key when the rule files are parsed; so does a route the configuration alone refuses" | "The plan refuses them, naming the key; so does a route the configuration alone refuses" | D2 |
| `workflow-configuration.md:797-798` | "An unknown key, at any level, raises `ValueError` naming it when the rule files are parsed." | "... when the plan runs." | D2 |
| `workflow-configuration.md:864-866` | "Snakemake combines `--configfile` and CLI `--config` values for Python scripts and DAG selection; the Python rules read that combined `snakemake.config`" | "The plan reads one configuration file; overrides are edits of that file (the resolved configuration records it and its hash)" | D2 |
| `workflow-downstream.md:8-12` | "The rules below are included for both backends. The segmentation and assignment scripts call the Starfinder package under both backends, so the environment that runs Snakemake needs the package" | "For the Python backend the morphology, segment and assign tasks replace these rules and run in the locked environment or image; the rules below stay for the MATLAB backend" | D6, D7, D9 |
| `workflow-downstream.md:30` | "both: registered `images/<round>/<channel name>/{fovID}.tif`" | "MATLAB: `images/<round>/<channel name>/`; Python (`morphology`): the saved forms, and `images/<round>/<channel label>/` with `morphology.channel_images`" | D6 (O10) |
| `workflow-downstream.md:31-34` | The rows of `rotate_nuclei`, `create_nuclei_amplicon_overlay`, `enhance_dapi_with_flamingo` ("those input folders must be supplied separately") and `stardist_segmentation` | MATLAB backend only; Python: the `segment` task with the inputs and `prepare` functions of its run | D6 (O9) |
| `workflow-downstream.md:48-51` | "The Python-only `segmentation` block is not wired to a rule yet (that is left to the workflow integration), and `rotate_nuclei` and the legacy segmentation rules are unchanged." | "The `segmentation` block runs as one `segment` task per (FOV, run); for the Python backend `rotate_nuclei` and the legacy segmentation rules are retired." | D6 |
| `workflow-downstream.md:92` | `reads_assignment` with "sample tile config" | MATLAB backend only; Python: the `assign` task, which reads no tile configuration | D6 |
| `workflow-downstream.md:143-150` | "the figures only with `previews: true`, into `fovs/{fovID}/previews/`" (of the assembly block) | "with `assignment.previews: true`, into `fovs/{fovID}/previews/`; the export's with `export.previews: true`" | P5 |
| `workflow-downstream.md:183-185` | "until it does, the workflow refuses it when the rules are parsed, naming both" | "until it does, the plan refuses it, naming both" | D2 |
| `broad-uger.md:12-17` | "The repository's `config/environment-v9.yaml` describes a Snakemake 9 environment"; "install the package/dependencies into the actual compute environment when using the Python backend" | The locked environment and its image (the engine record proposes the wording) | D9 |
| `broad-uger.md:46-48` | "conda deployment"; "three retries" | "the image of the lock; retries are the engine's, of exit status 1 only" | D8, D9 (O22) |
| `broad-uger.md:56-57` | "Logs use a rule log, `params.uger_log`, or `~/snakemake-logs/{rule}/`." | "Each task writes `workflow/logs/<task>.log`; the job's standard output and error stay the profile's" | D8 (O22) |
| `checkpoints.md:524-526` | "Resuming is always an explicit call; there is no scheduler. Nothing verifies a resume against `run.json` today: verified resume, scheduling and job completion belong to workflow orchestration." | "In the package, resuming is an explicit call. The workflow's `fov` task verifies a resume against `run.json` version 2 when `execution.resume_from_checkpoint` is on, and refuses on a mismatch (exit 3)." | D8 |
| `checkpoints.md:556-561` | "Two meanings, kept apart": "`status` is the outcome of `FOV.run`'s processing" and `outputs` the files written after it | Adds: "A workflow task succeeded only when its record says `succeeded` and every declared output is recorded" | D8 |
| `checkpoints.md:573` | `workflow_config`: "`{path, sha256}` of the workflow configuration file a workflow rule ran with" (the top-level `config_path`) | "... of the resolved configuration a workflow task ran with (`workflow/resolved.json`)" | D2, D8 |
| `checkpoints.md:585` | "The declared outputs of the Python processing rules are all recorded here" | "The declared outputs of the `fov` and `tile_merge` tasks are all recorded here" | D2, D5 |
| `checkpoints.md:563-586` | The field table | Adds the row `workflow` ([logging and records](#logging-and-records)) | D8 |
| `checkpoints.md:602-604` | "The workflow's processing rules write `run.json`" | "The workflow's `fov` and `tile_merge` tasks write `run.json`" | D2 |

## Choices in the commands

### Choice: what the assign command reads

| Option | Cost | Consequence |
| --- | --- | --- |
| A. `fovs/<fov>/molecules.csv` (`load_molecules`), the saved runs and `assign_molecules`, as `_run_reads_assignment` does today for an assembly block (recommended) | The command composes three public calls | It reads the `fov` task's declared output; the golden assignments of W-642's invocation B are produced this way |
| B. The FOV restored from its checkpoints (`load_checkpoint("candidates")`, `"pre_qc"`) and filtered again, then `FOV.assign_molecules` | Filtering runs again; the stages must exist | The assign task depends on checkpoint stages instead of the molecule table |

Recommendation: A.

### Choice: an imported label file per FOV (OP3)

| Option | Cost | Consequence |
| --- | --- | --- |
| A. `segmentation.runs[].path` may hold `{fov_id}`, which the plan resolves into each segment task's input (recommended) | The plan substitutes; `LabelImportConfig.path` stays one path per call | One run imports a folder of label files; the task's declared input is the FOV's file |
| B. A folder and a naming rule (`path_dir`, `<fov_id>.tif`) | A new key | A second way to name a file |

Recommendation: A.

### Choice: the grid of a segmentation run without a processing block (OP4)

| Option | Cost | Consequence |
| --- | --- | --- |
| A. The reference stain's saved form gives the reference grid: an addition to `FOV.reference_grid` (recommended) | An API change in `dataset/fov.py`, with its tests | D4's "nucleus segmentation from DAPI needs only morphology" holds; the stain shares the reference round's acquisition, so the grid is the same |
| B. Require the processing block up to the reference image | No API change | A nuclear-stain-only workflow loads and preprocesses every sequencing round |
| C. Require `reference_image_import` | No API change | The user supplies a file the workflow could derive |

Recommendation: A.

### Choice: where the morphology command's grouping of rounds goes (OP5)

| Option | Cost | Consequence |
| --- | --- | --- |
| A. Into the package's command module, unchanged (recommended) | The logic moves from `_snakemake.py` to the workflow module | The command reproduces the golden morphology records; the API stays as it is |
| B. Into `FOV.prepare_morphology` (rounds of several stain patterns in one call) | An API change | One call per FOV prepares every round, by hand as in the workflow |

Recommendation: A for the refactor; B is a later API improvement.

## Open points

What this page leaves to the other records of the batch, and the additions it needs (the choices
among them are above, with options and one recommendation):

* **OP1 (W-644).** The tiled task kinds' files, the tile list and the halo; the relation of the
  chunked checkpoint to the stage `registered`; resume of the global task.
* **OP2 (W-645).** How the plan runs before the Snakemake graph is built, the engines' use of the
  manifest, how Nextflow tracks outputs written outside its work folder, the schema's two branches,
  and the image.
* **OP3.** An imported label file per FOV: `segmentation.runs[].path` with `{fov_id}`, which the plan
  resolves per task (today `LabelImportConfig.path` is one path).
* **OP4.** The grid of a segmentation run without a processing block: the reference stain's saved
  form as the reference grid (an addition to `FOV.reference_grid`, `dataset/fov.py`).
* **OP5.** The morphology command's grouping of rounds by stain pattern, which today's adapter does
  (`_run_nuclei_registration`, `_snakemake.py:73-125`), moves into the package with the command.
* **OP6.** Atomic writes for the writers of declared outputs that do not write atomically yet.
