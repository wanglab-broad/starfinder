# Workflow engines: Snakemake and Nextflow over the contract, the Fiji and GPU tasks, the environment, the profiles and the MATLAB legacy separation

% Status: Proposed (W-645, 2026-10-30, at 56c35b8)

Status: Proposed. This record is reviewed at the specification gate of workflow orchestration
(W-646) before any implementation; approval publishes nothing.

This page designs the two engines that run the engine-neutral contract of
[orchestration](orchestration.md) (W-643) and the tiled task kinds of
[orchestration-tiled](orchestration-tiled.md) (W-644): a refactored Snakemake workflow and a new
Nextflow workflow, both reading the plan's manifest and calling the package's `starfinder workflow`
commands. It also specifies the BigStitcher (Fiji) task, the GPU tasks, the environment (the lock
change, the image, Apptainer, the retired conda files, the Nextflow and Java pins), the profiles
(Broad UGER for both engines and Google Batch for Nextflow), the separation of the frozen MATLAB
workflow, and the site pages that the last implementation issue writes. Its validation design is
[../validation/orchestration.md](../validation/orchestration.md). Statements about today's behaviour
cite the file and function at `56c35b8`, the base of the batch; paths are relative to the repository
root, and `starfinder/` stands for `src/python/starfinder/`.

% W-645 (S4 of §2.13); the batch runner/s213-spec-20261029 starts from dev at 56c35b8

The decisions this page implements (W-152, "§2.13 planning decisions (2026-10-27)", on top of the
boundary of 2026-09-28: logic that can change a result belongs in the Python API; scheduling,
placement, order and retries belong to the engine; `.smk` and `.nf` files carry no defaults,
parameter translation or result-affecting computation):

| Decision | What this page specifies for it |
| --- | --- |
| D1 | Both engines cover every task kind of the Python workflow ([coverage](#coverage)); MATLAB stays a legacy workflow in Snakemake only |
| D3 | The BigStitcher engine task in Snakemake only; Nextflow's refusal of `method: bigstitcher` without a supplied `dataset.xml` |
| D6 | The GPU request of the `segment` (and `export`) tasks per engine, one backend per process, the device |
| D7 | The dispatch to the frozen MATLAB rule set, the directory layout, the private legacy module and its pinning tests |
| D8 | How each engine retries, resumes and logs under the contract's completion rule |
| D9 | The lock change, the image from `uv.lock`, Apptainer on UGER, the retired conda files, the Broad shared environment, the Nextflow and Java pins |

The decisions taken at the W-642 checkpoint (Jiahao, 2026-10-29) that this page carries:

| Point | Decision | Where |
| --- | --- | --- |
| OC1 | `workflow/rules/gcloud-backup.smk` leaves `workflow/`; a draft issue removes it | [the MATLAB legacy separation](#the-matlab-legacy-separation) |
| OC2 | `tests/minimal_config.yaml` stays unchanged, the schema fixture of the frozen MATLAB dialect | [the MATLAB legacy separation](#the-matlab-legacy-separation) |
| OC6 | `cleanup_subtile.py`, `create_BigStitcher_macro.py`, `workflow/scripts/test/` and `workflow/notebook/` stay beside the frozen MATLAB workflow; `create_segmentation_preview.py` is retired | [the MATLAB legacy separation](#the-matlab-legacy-separation) |
| P2 | The original-against-refactored golden comparison runs in the `slow` tier | the validation design |
| P5 | Previews are options of the assign and export commands, off by default; every real-data check turns them on | the validation design |
| D9 | No release of `snakemake-interface-common` accepts the lock's `packaging==26.0`; the plan resolves it explicitly | [the lock change](#the-lock-change) |

Evidence, kept with the run records outside the repository and cited by path: the scratch
resolutions of the `workflow` group (`results/lock/<variant>/`: the edited `pyproject.toml`, the
resolved `uv.lock`, `uv-lock.log`, `export.txt`, `diff.json`), the image preflight
(`results/image/preflight.json`), the size estimate (`results/image/size-estimate.json`), the glibc
check of the locked wheels (`results/image/glibc.json`), the draft implementation issues
(`drafts/`), the worker notes (`notes.md`) and the run's manifest (`results/manifest.json`).

% Run directory: /home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/runs/W-645/20261030T050515Z-6722871a

## Terms

* **Engine**: Snakemake or Nextflow. It reads the manifest, turns each task into a job, runs the
  task's `command` (and, for the Fiji task, its `tool_command` first) and retries a failed task from
  scratch. It imports nothing from the package (the contract, "The flow").
* **Head**: the engine's own process, which schedules and submits jobs; on UGER it runs as a job.
* **Job**: one run of one task by an engine, on the head's host or as a cluster job.
* **Profile**: an engine's execution settings for one place (local, Broad UGER, Google Batch).
  A profile never holds a scientific setting.
* **Image**: the CPU container image built from `uv.lock` (D9), pinned by digest.

## What both engines share

Each engine does the same four things, in its own syntax:

1. **It runs the plan once before fan-out**, `starfinder workflow plan --config <file.yaml>`, and
   reads `<output root>/workflow/manifest.json` (the contract, "The plan command").
2. **It makes one job per task** with the task's `needs` as dependencies, its declared files, and
   its `execution` as the job's resources.
3. **It runs the task's command** with its execution context appended:
   `--engine <name> --engine-version <version> --job-id <id> --attempt <n> --image <digest or none>`.
4. **It maps the exit status** ([below](#exit-status-and-retries)): 0 done; 1 retried from scratch
   up to the profile's limit; 2 and 3 not retried.

**The execution context per engine.** The contract records `engine`, `engine_version`, `job_id`,
`attempt`, `image` and `host`; the command reads the host itself.

| Field | Snakemake | Nextflow |
| --- | --- | --- |
| `engine` | `snakemake` | `nextflow` |
| `engine_version` | `snakemake.__version__`, read by the Snakefile when it is parsed | `nextflow.version` |
| `job_id` | `$JOB_ID` on UGER (set by the scheduler in the job), else Snakemake's job number (`{jobid}` of the shell command) | `$JOB_ID` on UGER, else `task.hash` |
| `attempt` | the job's attempt, through a resource `attempt=lambda wildcards, attempt: attempt` | `task.attempt` |
| `image` | `$STARFINDER_IMAGE`, which the profile's job script sets to the image digest, else `none` | `task.container` (by digest), else `none` |

### Exit status and retries

| Exit | Contract meaning | Snakemake | Nextflow |
| --- | --- | --- | --- |
| 0 | Succeeded; the task record is written last | Done | Done |
| 1 | Failed; the attempt's declared outputs are removed | Retried up to the profile's `retries` | `errorStrategy` `retry` up to `maxRetries` |
| 2 | Refused before any work | Retried like 1: Snakemake does not distinguish exit statuses (each retry refuses again in seconds) | `finish` (no retry; running tasks end) |
| 3 | A resume refused | As 2 | `finish` |
| 137, 140, other | Killed by the scheduler (memory or time) | Retried like 1, with the attempt's larger resources | `retry` |

A retry may raise memory and runtime (an execution setting); it never changes a scientific
parameter (D8). Today's rules multiply only the runtime of the subtile rules by the attempt
(`make_get_runtime`, `workflow/rules/common.smk:187-195`).

#### Choice: resources of a retried task

| Option | Cost | Consequence |
| --- | --- | --- |
| A. Memory and runtime × attempt for every kind (recommended) | A larger request on each retry | A task killed for memory or time succeeds on its second attempt if twice the request suffices; the resource model of the tiled record is the first attempt's request |
| B. Runtime × attempt only, as today | None | A task killed for memory fails three times with the same request |
| C. No scaling | None | Retries help only transient failures |

Recommendation: A.

#### Choice: Snakemake and the refusals (exit 2 and 3)

| Option | Cost | Consequence |
| --- | --- | --- |
| A. Accept the retries of a refusal (recommended) | A refused task is submitted `retries` more times; each refuses in seconds, after its queue wait | No engine-side logic; the log of every attempt names the refusal |
| B. A wrapper in the rule's shell command that turns 2 and 3 into a marker file and exit 0 | A second completion rule | Breaks the contract's success rule: the job "succeeds" without a record |
| C. A Snakemake executor plugin that reads the exit status | A plugin to maintain | Out of proportion for a refusal that the plan, not a task, normally reports |

Recommendation: A. The plan refuses configuration errors before any job (exit 2 of the plan stops
the engine before fan-out), so task refusals are rare: a changed resolved configuration under a
running engine, or a missing input.

### The task fingerprint

A task's cache key must change when its result could change. In the contract's manifest a task entry
holds paths, its `needs` and its command (`--manifest <path> --task <id>`); the resolved-configuration
hash is in the top-level `configuration`, and the inputs' `bytes` and `mtime_ns` are in the top-level
`inputs` (`orchestration.md`, "The manifest"). A changed scientific setting, or a raw image or codebook
replaced at the same path, therefore changes the manifest but no task entry. The plan adds one
fingerprint per task, the same for both engines ([additions to the contract](#additions-to-the-contract)):

`tasks[].fingerprint` is the SHA-256 of the canonical JSON (the contract's serialization: keys sorted,
no insignificant whitespace) of an object with:

| Component | What it makes a task rerun on |
| --- | --- |
| `entry`: the task entry without `fingerprint`, `execution` and `command` | A change of its kind, identity, `needs`, declared inputs or outputs, record, log, group or Fiji command line |
| `configuration_sha256`: the resolved-configuration hash | A change of any scientific setting (the hash's scope, `orchestration.md`, "The resolved configuration") |
| `inputs`: for every declared input outside the output root (each `input:` path and each absolute path: the raw images, the codebook, the supplied files), its `path`, `bytes` and `mtime_ns` from `manifest.inputs` | An input replaced at the same path with another size or another modification time, older included |
| `package`: the package's version, and its git commit when recorded | A new package version, as verified resume requires (`orchestration.md`, "Choice: what a resume must match", A) |
| `needs`: the fingerprints of the tasks in `needs`, in their order | Any of the above upstream: staleness is transitive, so `verify` and a person find the consumers of a stale task as well as the engines do |

Left out, with the reason: `execution` (threads, memory, runtime, GPU, container, retries), which the
contract keeps outside the configuration hash because it never changes a result, so a larger memory
request does not rerun a finished task; `command`, whose only content beyond the kind and id is the
manifest's absolute path, so moving the trees with new `roots` keeps the fingerprints, as the
contract keeps paths relative (OC4); and the full content hashes of the inputs, which the plan does not
read (it lists sizes and times only; the tasks hash their inputs and `run.json` records them). An input
replaced by another of the same size whose modification time is set back to the old one exactly is
therefore not seen; verified resume, which compares `run.json`'s input hashes, still refuses it for the
`fov` task.

The plan computes the fingerprints in the manifest's task order, which respects `needs`. Each task
record keeps the fingerprint it ran with, `workflow.fingerprint`, beside `configuration_sha256` and
`manifest_sha256` (the contract, "Logging and records"). **The staleness rule, one for both engines and
for a person:** a task is complete only when its record says `succeeded`, every output it lists is
present with its recorded size (or SHA-256 with `--hash`), and its `workflow.fingerprint` equals the
current manifest's `tasks[].fingerprint`; the `verify` command applies this rule
([below](#-resume-and-the-completion-rule-d8)). Snakemake carries the fingerprint in the rule's
`params`; Nextflow carries it in the task entry `t` and in `VERIFY_BEFORE`'s token, both part of the
process's cache hash.

#### Choice: what the fingerprint covers

| Option | Cost | Consequence |
| --- | --- | --- |
| A. The five components above, with the run-wide resolved-configuration hash (recommended) | One SHA-256 per task at plan time | Any scientific change reruns every task, as verified resume already refuses every stage after one; an input reruns its tasks and their downstream only |
| B. As A, with a hash of only the configuration blocks the task's kind reads instead of the run-wide hash | A map from kind to blocks in the plan, kept in step with the commands | A change of the report block reruns only `report` and `compare`; a wrong map silently keeps stale results |
| C. As A, with the inputs' content hashes | The plan reads every raw image at every invocation (5.2 GiB per aging FOV) | Catches a same-size replacement with the old time kept, at the cost of the plan's input scan, which the contract keeps to sizes and times |

Recommendation: A.

## Snakemake

### The Snakefile over the manifest

`workflow/Snakefile` keeps its name and becomes a dispatcher. For `backend: matlab` it includes the
frozen MATLAB rule set ([the MATLAB legacy separation](#the-matlab-legacy-separation)); otherwise it
runs the plan, reads the manifest and defines the task rules of `workflow/rules/tasks.smk`. A
sketch, not code to land as written:

```python
import json, subprocess
import snakemake as _snakemake

if config.get("backend") == "matlab":
    include: "rules/legacy.smk"            # today's MATLAB includes, unchanged
else:
    CONFIGFILE = only_one(workflow.configfiles)   # the plan reads one file (contract, "Changed statements")
    if workflow.is_main_process:
        subprocess.run(["starfinder", "workflow", "plan", "--config", CONFIGFILE], check=True)
    MANIFEST = json.load(open(manifest_path(CONFIGFILE)))
    ENGINE = ["--engine", "snakemake", "--engine-version", _snakemake.__version__]
    include: "rules/tasks.smk"             # one rule per rule group of the manifest
```

Today the Snakefile validates the configuration against the schema (`workflow/Snakefile:19`,
`snakemake.utils.validate`), includes `common.smk`, whose parse imports the package
(`common.smk:212-214`, `:414-447`), and selects rule files by backend and block
(`workflow/Snakefile:27-55`). Under this design the Python branch validates nothing itself: the
plan validates every block (the contract, V1 to V10), and a refusal of the plan (exit 2) stops
Snakemake before the graph is built, with the plan's message. Nothing in the Python branch reads a
configuration key other than `backend`; the plan finds the output root from the file it is given.

**Only the main process plans.** With the cluster-generic executor every job runs Snakemake again on
its node, which parses the Snakefile (Snakemake 9.15.0: `Workflow.is_main_process` is false in a
remote job, `snakemake/workflow.py:298`). A plan in every job would rescan the inputs and could
rewrite the manifest while other jobs read it. The remote parse reads the manifest that the main
process wrote.

**One rule per rule group.** Snakemake requires output patterns, not functions, so a rule cannot take
its outputs from a task's manifest entry alone. The plan therefore adds a `groups` section to the
manifest ([additions to the contract](#additions-to-the-contract)): the tasks of one kind whose
declared outputs follow the same templates form one group (a kind, or a kind and a segmentation run,
because an imported run declares no `input.ome.tif`), with the templates written over the identity's
fields as wildcards:

```text
{"name": "segment.cell", "kind": "segment", "wildcards": ["sample", "fov"],
 "outputs": ["fovs/{fov}/segmentation/cell/labels.tif", "fovs/{fov}/segmentation/cell/segmentation.json",
             "fovs/{fov}/segmentation/cell/input.ome.tif"],
 "record": "workflow/tasks/segment.{sample}.{fov}.cell.json", "log": "workflow/logs/segment.{sample}.{fov}.cell.log",
 "temporary": []}
```

`tasks.smk` defines one rule per group in a loop (Snakemake's `rule:` with `name:`), each with:

| Directive | From the manifest | Note |
| --- | --- | --- |
| `name` | `<kind>_<n>`, the group's index | Rule names stay identifiers whatever the run names |
| `wildcard_constraints` | each wildcard as an alternation of the escaped values of the group's tasks | As `FOV_CONSTRAINT` does today for `fovID` (`workflow/Snakefile:27-29`, `common.smk:224`); no pattern of one group matches another group's file |
| `input` | a function of the wildcards: the task's `inputs` (with `input:` resolved against `roots.input`) and the `record` of each task in `needs` | The records make the order the contract's: a consumer starts after its producers' records exist |
| `output` | the group's `outputs` and `record`, each under `roots.output`; `temp()` for the group's `temporary` outputs | The record is an output, so a job that exits 0 without it fails Snakemake's own check |
| `log` | the group's `log` | Snakemake keeps a `log` file of a failed job and removes failed `output` files; the command writes the log itself |
| `params` | `command` and `fingerprint` of the task ([the task fingerprint](#the-task-fingerprint)), which covers the entry, the resolved-configuration hash, the declared inputs outside the output root by size and modification time, the package version and the `needs`' fingerprints | Snakemake's `params` rerun trigger reruns exactly the tasks whose fingerprint changed: a changed scientific setting, an input replaced at the same path (also with an older modification time, which the `mtime` trigger alone misses), a new package version, and every task downstream of one of these. The profile sets `rerun-triggers: [mtime, params, input]`; not `code` (the rules are generated from the manifest, and the fingerprint carries the package version) or `software-env` (no conda) |
| `threads`, `resources` | `execution.threads`; `mem_mb` and `runtime` from `execution.memory_mb` and `runtime_min` × attempt; `gpu=1` when `execution.gpu`; `attempt` | No default lives in the rule: a task without a value takes the plan's resolved `execution.defaults` |
| `shell` | `{params.command} --engine snakemake --engine-version <v> --job-id "${{JOB_ID:-{jobid}}}" --attempt {resources.attempt} --image "${{STARFINDER_IMAGE:-none}}"` | The `bigstitcher_engine` group runs `tool_command` first ([Fiji](#fiji-the-bigstitcher-task-in-snakemake-only)) |

`rule all` takes every task's record. Targets are task records, so `snakemake <record path>` runs one
task and its upstream.

#### Choice: how Snakemake knows each task's files

| Option | Cost | Consequence |
| --- | --- | --- |
| A. Rule groups in the manifest, one rule per group, outputs as templates over the identity (recommended) | One more manifest section, which the plan checks: every task's outputs equal its group's templates filled with its identity | Snakemake knows every declared output: a deleted output reruns its task, a failed job's outputs are removed by Snakemake as well as by the command, and `temp()` works |
| B. One rule per kind whose only outputs are the task record and the log, the record's inputs being the `needs`' records | None in the manifest | Snakemake sees no data file: a deleted `molecules.csv` is not rerun (its consumer refuses, exit 2, naming the file), and `temp()` is unavailable |
| C. One rule per task, defined in a loop | None in the manifest | Thousands of rules for a full dataset (one per FOV and kind): a slow parse, a rule list nobody reads |

Recommendation: A; B is the fallback if a group cannot be formed for a later kind.

#### Choice: when the plan runs before the Snakemake graph is built

| Option | Cost | Consequence |
| --- | --- | --- |
| A. At parse time, in the main process only (recommended) | The plan's input scan (sizes and times, no hash) at every invocation, dry runs included | A dry run (`-n`) lists every task of the run; the rules exist when Snakemake parses them, which option A of the previous choice needs. The plan writes its files only when their content changes, so a dry run leaves an unchanged plan untouched; on a changed input it rewrites `workflow/manifest.json` and `resolved.json` before showing the jobs |
| B. A checkpoint rule `plan` whose output the task rules read through `checkpoints.plan.get()` | Every task's input function waits on the checkpoint | A dry run shows the plan job only, and the rules cannot be defined from the manifest (Snakemake re-evaluates inputs after a checkpoint, never outputs), so only the previous choice's B works |
| C. The user runs the plan, and the Snakefile reads an existing manifest | One more command | A stale manifest runs old inputs (the contract chose "every engine invocation runs the plan") |

Recommendation: A.

**Local rules and temporary files.** `merge` and `compare` are local rules (`localrules:`): each is
a small barrier task (a histogram merge, a comparison page) whose queue wait on UGER would exceed its
run time; on UGER the head runs as a job, so local rules run inside that job, not on the login node.
The tiled record's chunked checkpoint `tiled/registered.h5` is the one temporary output today: the
manifest marks it `temporary` unless the stage `registered` is kept, and Snakemake declares it
`temp()`, so a later run does not rerun `tile_global` because the file is gone. The tiled record
already has the merge remove it after success (`orchestration-tiled.md`, "Declared files of the tiled
task kinds"); under Snakemake the removal is Snakemake's, and the merge's own removal applies to the
other engines and to a run by hand ([open check E-1](#open-points)). The UGER profile's `notemp: true`
(`profile/broad-uger/config.yaml:20`) goes ([profiles](#the-snakemake-uger-profile)).

**Benchmark and log directives.** No rule has a `log:` directive today; heavy Python rules have a
`benchmark:` file (`log/benchmark/<rule>/{fovID}.txt`, for example `rsf_single_fov`,
`workflow/rules/registration-py.smk:16-31`).

#### Choice: Snakemake's benchmark directive

| Option | Cost | Consequence |
| --- | --- | --- |
| A. No `benchmark:`; the task record's `wall_s`, `cpu_s` and `peak_rss_kib` are the measurement (recommended) | None | One measurement per task, the same under both engines and by hand; Nextflow's trace file stays its own engine log |
| B. `benchmark: workflow/benchmarks/<task>.tsv` for every rule | One more file per task and a monitoring thread per job | Snakemake's own figures beside the record's, measured around a different process tree |

Recommendation: A.

The job's standard output and error stay the profile's (on UGER, `qsub -o`, [below](#the-snakemake-uger-profile)).

### The dispatch to the MATLAB rule set and the directory layout (D7)

The MATLAB launcher builds the MATLAB path from `starfinder_path` as
`<starfinder_path>/workflow/scripts` (`matlab_command`, `workflow/rules/matlab_launcher.py:33-74`,
the path at `:49`), and every MATLAB rule names its script relative to its rule file
(`script: "../scripts/…"`). Moving the MATLAB files would change the launcher, which D7 freezes.

#### Choice: the directory layout

| Option | Cost | Consequence |
| --- | --- | --- |
| A. The MATLAB files stay where they are; the Python rules leave (recommended) | `common.smk` loses its Python branches and becomes the MATLAB rule set's own; `rules/legacy.smk` holds today's MATLAB includes from `workflow/Snakefile:23-80` | No MATLAB path or launcher changes. `workflow/` then holds `Snakefile` (the dispatcher), `rules/tasks.smk` (Python), `rules/legacy.smk`, `rules/common.smk`, the MATLAB rule files and the shared `segmentation.smk` and `reads-assignment.smk` (MATLAB only), `scripts/` (MATLAB scripts and the shared rules' adapters), `schemas/` |
| B. Move the MATLAB workflow to `workflow/legacy/` | The launcher's path, `starfinder_path` users and every guide change | A clean split, paid for with a change D7 freezes |
| C. Two entries, `workflow/Snakefile` (Python) and `workflow/legacy.smk` (MATLAB), chosen with `-s` | Users choose the file | No dispatch by `backend`, which the issue asks for |

Recommendation: A. The Python files that leave `workflow/`: `registration-py.smk`,
`spot-finding-py.smk`, `stitching-py.smk`, `export.smk`, `report.smk` and the Python adapters
`rsf_single_fov.py`, `gr_single_fov_subtile.py`, `lrsf_single_fov_subtile.py`,
`deep_create_subtile.py`, `deep_rsf_subtile.py`, `nuclei_registration.py`, `bigstitcher_macro_sample.py`,
`bigstitcher_engine.py`, `stitch_sample.py`, `export_sample.py`, `report_sample.py`,
`report_dataset.py` (the baseline's rule inventory, backend `python`). The Python branch of
`rotate_nuclei` (`registration-py.smk:90-122`) goes with D6.

**The schema.** Today one schema holds both dialects with backend guards
(`workflow/schemas/config.schema.yaml`, `allOf`, `:346-576`; the contract left "one schema with two
branches or two schema files" to this record).

#### Choice: the schema of the two dialects

| Option | Cost | Consequence |
| --- | --- | --- |
| A. Two schemas: the MATLAB dialect stays in `workflow/schemas/config.schema.yaml`, validated by the legacy Snakefile as today; the Python shape is a package resource read by the plan (recommended) | The MATLAB schema's Python guards become dead; they are removed only with a MATLAB-dialect test of each guard's refusal kept | The plan validates with the package's own schema wherever it runs (Nextflow, by hand); the engine files hold no schema |
| B. One schema with two branches, read by the plan from the package and by the legacy Snakefile from `workflow/` | Two copies or a path from the package into `workflow/` | Every change of either dialect touches the shared file |

Recommendation: A. `tests/minimal_config.yaml` stays the MATLAB dialect's schema fixture unchanged
(OC2); `test_workflow_config_validation.py` keeps reading it.

**What the MATLAB branch keeps.** Its rules, scripts, keys and files are unchanged (D7; the baseline,
"The MATLAB legacy surface frozen by D7"); its parse-time definitions stay in `common.smk` (its
enumeration `:236-279`, `rsf_preparation` `:391-398`, the presets `:38-110`); its `ruleorder`
(`workflow/Snakefile:63-74`) and local rules (`:79-80`) move into `rules/legacy.smk` unchanged.

### Coverage

Every task kind of the contract's manifest (`orchestration.md`, "Terms"), with its Snakemake rule
group and its Nextflow process, or the reason it is not covered:

(orchestration-engines-coverage)=

| Task kind | Command | Snakemake | Nextflow | Reason when not covered |
| --- | --- | --- | --- | --- |
| `summarize` | `summarize` | group `summarize.<step>` (wildcards `sample`, `step`, `fov`) | `SUMMARIZE_<p>`, one alias per supplied-step position up to 3 ([below](#choice-repeated-summarize-and-merge-stages)) | — |
| `merge` | `merge` | group `merge.<step>` (`sample`, `step`), local rule | `MERGE_<p>`, as `SUMMARIZE_<p>` | — |
| `fov` | `fov` | group `fov` (`sample`, `fov`) | `FOV` | — |
| `tile_global` | `tile-global` | group `tile_global` (`sample`, `fov`), `registered.h5` `temp()` unless kept | `TILE_GLOBAL` | — |
| `tile` | `tile` | group `tile` (`sample`, `fov`, `tile`) | `TILE` | — |
| `tile_merge` | `tile-merge` | group `tile_merge` (`sample`, `fov`) | `TILE_MERGE` | — |
| `morphology` | `morphology` | group `morphology` (`sample`, `fov`) | `MORPHOLOGY` | — |
| `segment` | `segment` | group `segment.<run>` (`sample`, `fov`), one per run | `SEGMENT` (one alias per seed depth, each also with the label `gpu`, [below](#channels-from-the-manifest)) | — |
| `assign` | `assign` | group `assign.<name>` (`sample`, `fov`) | `ASSIGN` | — |
| `bigstitcher_macro` | `bigstitcher-macro` | group `bigstitcher_macro` (`sample`) | none | D3: Nextflow runs no Fiji, so nothing reads the macro; a supplied `dataset.xml` with its view map replaces it |
| `bigstitcher_engine` | Fiji, then `bigstitcher-check` | group `bigstitcher_engine` (`sample`) | none: a manifest that holds it is refused before any task | D3: Fiji only in Snakemake |
| `bigstitcher_import` | `bigstitcher-import` | group `bigstitcher_import` (`sample`) | `BIGSTITCHER_IMPORT`, on the supplied route only (`stitching.dataset_xml`) | — |
| `stitch` | `stitch` | group `stitch` (`sample`) | `STITCH` | — |
| `export` | `export` | group `export` (`sample`) | `EXPORT` (and its `gpu` alias) | — |
| `report` | `report` | group `report` (`sample`) | `REPORT` | — |
| `compare` | `compare` | group `compare` (no wildcard), local rule | `COMPARE` | — |

The plan itself is not a task kind: Snakemake runs it at parse time, Nextflow as its first step
(`PLAN`, [below](#the-plan-in-nextflow)); neither are Nextflow's two `VERIFY` jobs, which check
completion ([below](#-resume-and-the-completion-rule-d8)). The MATLAB legacy workflow is Snakemake only (D1).

### Fiji: the BigStitcher task in Snakemake only

D3 keeps Fiji as an engine task. Today `bigstitcher_engine` (`workflow/rules/stitching-py.smk:54-67`)
runs a script whose adapter reads `fiji_path` when the rule runs, removes the old `dataset.xml`,
starts Fiji with `-Djava.io.tmpdir=<BigStitcher folder> --ij2 --headless --run <macro>` and then
checks the result, because Fiji exits 0 when its macro fails (`_run_bigstitcher_engine`,
`starfinder/_snakemake.py:814-831`; the check `_check_bigstitcher_engine`, `:781-811`). The macro
comes from `bigstitcher_macro_sample` (`_run_bigstitcher_macro`, `:756-778`) and the import from
`stitch_sample` (`_run_stitching`, `:707-753`).

Under the contract the three tasks are:

| Task | Who writes what | Snakemake rule's shell |
| --- | --- | --- |
| `bigstitcher_macro` | The package: `starfinder workflow bigstitcher-macro` writes the macro, `view_map.csv` and `tiles/` | the task's `command` |
| `bigstitcher_engine` | The engine runs Fiji from `tool_command` with `{fiji}` replaced by `execution.tool.path` (`execution.fiji` of the configuration, today `fiji_path`, `workflow/schemas/config.schema.yaml:35-37`), then `starfinder workflow bigstitcher-check` checks `dataset.xml` against the view map and writes the task record | `rm -f <dataset.xml>; <tool_command with {fiji} replaced>; <command>`: the check runs whatever Fiji's exit status, and decides (the contract, "Choice: the record of the Fiji task") |
| `bigstitcher_import` | The package: `starfinder workflow bigstitcher-import` checks `dataset.xml` again, imports it and writes the placement | the task's `command` |

The Fiji job's resources: `execution.tasks.bigstitcher_engine` (threads 4 as today; memory at least
8 GB on UGER, where `h_vmem` is a consumable with a 1G default and the JVM needs about 8G, W-641).
Fiji runs from the lab's install (`/stanley/WangLab/Tools/Fiji.app/ImageJ-linux64`, BigStitcher
1.2.14, headless on UGER compute nodes, W-641), with its own bundled Java (Zulu JDK 8). The Fiji job
runs, like every job, inside the image with `/stanley` bound ([profiles](#the-snakemake-uger-profile)),
so Fiji runs from the bound install and `bigstitcher-check` runs in the image after it; whether Fiji's
launcher runs inside the image is open point E-5.

#### Choice: where Fiji comes from

| Option | Cost | Consequence |
| --- | --- | --- |
| A. The lab install named by `execution.fiji` (recommended) | The path is site-specific | Today's route (`fiji_path`); verified headless on UGER (W-641) |
| B. The `fiji-bigstitcher:latest` image on GP099 (888 MB, Fiji, OpenJDK 8 and the BigStitcher update site, W-641) | An image of unknown provenance to pin and publish | One more image to own; its BigStitcher version is not recorded |

Recommendation: A; B is revisited if Fiji must run on Google Batch, which D3 excludes.

**Nextflow's refusal.** Nextflow implements no `bigstitcher_macro` or `bigstitcher_engine`
process. A manifest that holds a task of a kind Nextflow does not implement is refused by `main.nf`
before any task, with one message for D3:

```text
Nextflow runs no Fiji (D3): this plan has the task bigstitcher_engine/sample. Supply the BigStitcher
result as stitching.dataset_xml and stitching.view_map, use stitching.method multiview_stitcher, or run
the plan with Snakemake
```

#### Choice: where the refusal of `method: bigstitcher` under Nextflow lives

| Option | Cost | Consequence |
| --- | --- | --- |
| A. `main.nf` refuses any manifest that holds a task kind it does not implement, naming it (recommended) | A list of implemented kinds in `main.nf` | An engine capability check, not a configuration rule: `main.nf` reads no configuration key; the same check catches any future kind Nextflow lacks |
| B. `starfinder workflow plan --engine nextflow` refuses the configuration | An engine argument to the plan | The plan would know engines; the manifest of one configuration would depend on the engine |

Recommendation: A.

### GPU tasks (D6)

The `segment` task of a run whose `device` is `cuda` has `execution.gpu: true`; so has the `export`
task on the route after fusion with `export.segmentation.device: cuda` (the contract, "What an engine
needs"). Today the device is one key of the segmentation block (`segmentation.device`,
`_segmentation_block`, `starfinder/dataset/workflow.py:1262-1287`) and of the export's segmentation
(`_sample_segmentation_block`, `:1687-1717`); the learned backends use the first visible GPU (`_tensorflow_device`, `_torch_device`,
`starfinder/segmentation/_learned.py:274`, `:296`: `"cuda"` is device 0 of the visible devices, and
TensorFlow's memory growth is turned on).

| | Snakemake | Nextflow |
| --- | --- | --- |
| The request | `resources: gpu=1`; locally `--resources gpu=<n>` caps concurrent GPU jobs; on UGER the submit script adds `-l gpu=TRUE,operating_system=RedHat8` for a job with `gpu` | the module's alias with the label `gpu`, to which `main.nf` routes a task with `execution.gpu`; `accelerator 1`; on `sge` the label's `clusterOptions` add `-l gpu=TRUE,operating_system=RedHat8`; on `google-batch` `accelerator 1, type: <GPU type>` and a machine type with a GPU |
| The device | the job script chooses one GPU ([below](#choice-which-gpu-a-job-uses)) and exports `CUDA_VISIBLE_DEVICES` | the same, in the process's `beforeScript` |
| The image | `apptainer exec --nv` | `apptainer.runOptions` with `--nv` for the label `gpu` |

**One backend per process.** A `segment` task runs one run, so one method and its framework
(TensorFlow for StarDist, torch for Cellpose); no process loads both. This holds by construction of
the manifest (one task per (FOV, run), D6), so the engines add nothing. Today `stardist_segmentation`
runs one method per job as well (`workflow/rules/segmentation.smk:22-34`).

**The UGER request.** W-641 found two usable GPU nodes (`uger-gpu-d001`, `d002`, 4 × A30, driver
575.57.08, RHEL 8); a job submitted from the RHEL 7 login node gets `operating_system=RedHat7` added
and never matches a GPU node unless it asks for `RedHat8`; `gpu` is a host boolean, not a count, and
UGER does not isolate devices: a job sees all four GPUs, and up to three jobs share a node.

#### Choice: which GPU a job uses

| Option | Cost | Consequence |
| --- | --- | --- |
| A. The job script picks the GPU with no compute process (`nvidia-smi --query-compute-apps`), exports `CUDA_VISIBLE_DEVICES`, and the record keeps the variable (recommended) | A few lines of shell in the profile; a race between two jobs starting at once on one node | Placement stays the engine's (the boundary); the package uses device 0 of what it sees, as today |
| B. A device index in the configuration (`cuda:1`) | A new key that would enter the configuration hash | Placement in a scientific setting; a fixed index collides with other users |
| C. The package picks the least-loaded GPU | Logic in the API | Placement in the API, which the boundary gives to the engine |

Recommendation: A. The race is bounded by the queues' one slot per GPU node each (W-641); the
UGER checklist of the validation design checks it.

## Nextflow

### The directory and the modules

#### Choice: where the Nextflow workflow lives

| Option | Cost | Consequence |
| --- | --- | --- |
| A. `nextflow/` at the repository root: `main.nf`, `nextflow.config`, `modules/<kind>.nf`, `conf/{local,sge,google_batch}.config` (recommended) | A second top-level folder beside `workflow/` | Each engine's files in one folder; `nextflow run nextflow/main.nf` from the root; the Snakemake folder keeps its name, which guides and the MATLAB launcher use |
| B. `workflow/nextflow/` | None | Mixes with the frozen MATLAB files of option A of the layout choice |
| C. Rename `workflow/` to `snakemake/` and add `nextflow/` | Every path in guides, profiles and the launcher | Symmetric names; a change D7 freezes |

Recommendation: A.

DSL2, one module per task kind of the [coverage table](#coverage), each a process with the same
shape:

```groovy
process FOV {
    tag "${task_id}"
    label 'cpu'
    cpus   { t.execution.threads }
    memory { "${t.execution.memory_mb * task.attempt} MB" }
    time   { "${t.execution.runtime_min * task.attempt} min" }
    container { t.execution.container ?: null }
    input:  tuple val(task_id), val(t), path(needs_records)
    output: tuple val(task_id), path("${task_id.replace('/', '.')}.json"), emit: done
    script:
    """
    ${t.command.join(' ')} --engine nextflow --engine-version ${nextflow.version} \\
        --job-id \${JOB_ID:-${task.hash}} --attempt ${task.attempt} --image ${task.container ?: 'none'}
    cp ${params.output_root}/${t.record} .
    """
    stub:
    """
    stub_task.sh ${params.manifest} ${task_id}
    """
}
```

A process label is static, so the kinds that may request a GPU (`segment`, `export`) are included
twice, as `SEGMENT` (label `cpu`) and `SEGMENT_GPU` (label `gpu`), and `main.nf` routes each task by
its `execution.gpu` with the `branch` operator.

The command writes the task's declared outputs in place under `roots.output`, as the contract
chose ("Choice: how a task publishes its outputs"); the process copies the task record into its work
folder and emits it, which is the one file Nextflow tracks per task.

### Channels from the manifest

`main.nf` reads the manifest with `groovy.json.JsonSlurper` and builds, per kind, a channel of
`(task_id, task)` from `manifest.tasks`. A task starts when every task it `needs` has emitted its
record. One producer often feeds several tasks (a `tile_global` feeds every `tile` of its FOV, a
`merge` every `fov` of its sample, a `fov` its `assign` and the sample's `stitch`), so the channel of edges `(need, task)` holds the same key many times. `join` does not
support duplicate keys on tuples (Nextflow's operator reference, `join`); `combine` with `by` is the
operator that pairs every matching item, one output per edge. The edges are therefore combined with the
emitted `(id, record)` of every upstream process and grouped per task with a `groupKey` that carries the
number of its needs, so that `groupTuple` emits the task as soon as its last need is done and tasks of
one FOV proceed without waiting for the other FOVs:

```groovy
// done: (id, record) emitted once by each upstream process, mixed into one channel
ready = tasks_of_kind
    .filter { id, t -> t.needs }
    .flatMap { id, t -> t.needs.collect { need -> tuple(need, id, t) } }  // one item per edge
    .combine(done, by: 0)                                                   // (need, id, t, record), one per edge
    .map { need, id, t, record -> tuple(groupKey(id, t.needs.size()), t, record) }
    .groupTuple()                                                           // emitted when all needs are done
    .map { key, ts, records -> tuple(key.toString(), ts[0], records) }
    .mix(tasks_of_kind.filter { id, t -> !t.needs }.map { id, t -> tuple(id, t, []) })
```

This uses tuples and the operators of Nextflow 26.10.4 as they are, with no preview feature; the
record form of `join` with static typing, which emits every matching combination, would need the
typed-workflow preview and is not used. Kinds feed each other in the fixed order of the contract's
graph (`orchestration.md`, "The task graph from data dependencies"), with one alias per position
where a kind feeds itself or a kind that feeds it back (below).

A `segment` task may need another `segment` task (the seeds' run, ordered by seeds, D6), and a DSL2
process cannot feed itself. `main.nf` computes each segment task's seed depth from its `needs` (0
without seeds) and includes the module once per depth (`SEGMENT_0`, `SEGMENT_1`, `SEGMENT_2`, and
each again with the label `gpu`), each fed by the depth before it; a deeper chain is refused before any task, naming it. Computing the
depth is ordering, the engine's (the boundary).

#### Choice: segmentation runs that seed each other

| Option | Cost | Consequence |
| --- | --- | --- |
| A. One alias of `SEGMENT` per seed depth, up to 3, deeper chains refused (recommended) | Three includes | Covers nucleus → cell and one more level; the refusal names the chain |
| B. One process per segmentation run name | Run names are configuration values; includes are static | Not expressible without generating `main.nf` |
| C. Nextflow's experimental process recursion | An experimental feature | May change between Nextflow releases |

Recommendation: A.

A preprocessing recipe may hold more than one step with `fit: supplied`. The `summarize` tasks of a
later supplied step then need the earlier step's `merge` (the contract, "The two-pass statistics
tasks", item 3), so `SUMMARIZE` feeds `MERGE`, which feeds `SUMMARIZE` again: a cycle between two
processes, which DSL2 cannot express, as for seeded segmentation runs. `main.nf` computes each
statistics task's position `p` (0 for the first supplied step, 1 for the next) from its `needs`, includes
the pair once per position (`SUMMARIZE_0` and `MERGE_0`, `SUMMARIZE_1` and `MERGE_1`, `SUMMARIZE_2` and
`MERGE_2`), each pair fed by the `MERGE` of the position before it, and refuses before any task a recipe
with more than three supplied steps, naming the step. Snakemake needs no change: its graph follows the
groups `summarize.<step>` and `merge.<step>` through their files.

#### Choice: repeated summarize and merge stages

| Option | Cost | Consequence |
| --- | --- | --- |
| A. One alias pair per supplied-step position, up to 3, deeper recipes refused naming the step (recommended) | Six includes | Covers every recipe in use (one supplied step; two with a later step summarized on the earlier one's statistics) and one more; each sample's FOVs still run their `summarize` tasks in parallel |
| B. One `STATISTICS` process per sample that runs every pass of every supplied step in one job | A task the manifest does not have | Breaks the contract's one command per task; the passes of one sample run one after another in one job, the summaries of its FOVs no longer in parallel |
| C. Nextflow's experimental process recursion | An experimental feature | May change between Nextflow releases |
| D. Refuse more than one supplied step under Nextflow | None | A D1 exclusion for a configuration Snakemake runs, without an engine reason |

Recommendation: A.

### The plan in Nextflow

`PLAN` is the first process (`cache false`, so it runs at every invocation, as the contract's "every
engine invocation runs the plan" requires): it runs `starfinder workflow plan --config <file>` (in
the image) and emits the manifest's path; `main.nf` reads the manifest after `PLAN` completes, in a
`workflow` block that builds the channels from it. The plan's refusal (exit 2) ends the run with the
plan's message.

#### Choice: when Nextflow reads the manifest

| Option | Cost | Consequence |
| --- | --- | --- |
| A. `PLAN` as a process, the channels built from its emitted manifest (recommended) | `splitJson` or a `map` over the emitted file | The plan runs in the image like every task; it is never cached, and an unchanged plan rewrites nothing, so `-resume` still reuses the tasks |
| B. The plan run in the head before `workflow {}` (`"starfinder workflow plan".execute()`) | None | The head needs the package installed outside the image; the plan's environment differs from the tasks' |
| C. The user runs the plan; `--manifest` is a parameter | One more command | As Snakemake's C: a stale manifest |

Recommendation: A.

### `publishDir` and the declared outputs

#### Choice: how Nextflow publishes the declared outputs

| Option | Cost | Consequence |
| --- | --- | --- |
| A. No `publishDir` for data: the command writes in place under `roots.output` (the contract's option A), the process emits its task record (recommended) | Nextflow does not track the data files | One output tree for both engines; a record says which files a task wrote, with their hashes; `publishDir` is used only for the run's `trace.txt`, `report.html` and `timeline.html` under `workflow/nextflow/` |
| B. The command writes into the work folder and `publishDir` copies the declared outputs into `roots.output` | Every command gets a per-task output root; the inputs of other tasks must be staged into it | The records name paths relative to an output root that each task would have its own copy of; the contract rejected staging (its choice B) |

Recommendation: A.

### Resources and retries

Per process, from the task's `execution`: `cpus`, `memory` and `time` × `task.attempt` (the
[retry choice](#choice-resources-of-a-retried-task)), `accelerator` for the label `gpu`, `container`
from `execution.container`. The error strategy maps the exit status:

```groovy
errorStrategy { task.exitStatus in [2, 3] ? 'finish' : (task.attempt <= params.max_retries ? 'retry' : 'finish') }
maxRetries params.max_retries
```

`params.max_retries` is the profile's (2 for `sge`, as Snakemake's `retries` below).

### `-resume` and the completion rule (D8)

**What `-resume` detects and what it does not.** Nextflow's `-resume` reuses a task when the hash of
its process script and inputs matches a cached task whose work folder still holds its outputs. Here a
task's inputs are its `needs`' emitted records, staged from the producers' work folders and hashed by
path, size and modification time (the default `cache` mode), and the one output Nextflow tracks is the
copy of the task record in the work folder ([above](#the-directory-and-the-modules)). So `-resume`
reruns a task when its script, its task entry or a producer's emitted record changes, or when its own
work folder is gone. Without the [fingerprint](#the-task-fingerprint) it would not see a changed
scientific setting or an input replaced at the same path either, because the configuration hash and
the inputs' sizes and times live outside the task entries; with it in the entry and in the token below,
both change the cache key. It does not see the output root: a deleted or truncated `molecules.csv`, a deleted
`report/index.html` or a deleted task record under `roots.output` changes no input and no work folder,
so the task is reused and emits its cached record copy, and so is every consumer, whose inputs (those
copies) are unchanged too. A consumer's command, which would check its `needs`' records and declared
inputs (the contract, "Common arguments"), never runs; a terminal task (`compare`, or the last task of
a selected run) has no consumer at all. A cache that outlived its outputs therefore passes silently
under `-resume` alone.

What still holds: a task that runs removes its earlier record first and writes it last (the contract,
"Completion"), so a consumer that does run never reads a half-made predecessor; and `-resume` is not
verified resume (`execution.resume_from_checkpoint`, the `fov` task's resume from its stages against
`run.json`, is a setting of the command and works the same under both engines). Snakemake needs no
counterpart: its rule groups declare every output, and Snakemake reruns a task whose output is missing.

**The design (option D below).** One more command, `starfinder workflow verify --manifest M
[--hash] [--final]`, applies D8's completion rule to every task of the manifest from the output root
alone, by the [staleness rule](#the-task-fingerprint): the task record says `succeeded`, every output
it lists exists with its recorded size (with `--hash`, its SHA-256), and the record's
`workflow.fingerprint` equals the current manifest's. Outputs the manifest marks `temporary` and the
merge removed are exempt. Two
uncached processes run it (`cache false`, one local job each):

1. **`VERIFY_BEFORE`, after `PLAN` and before fan-out**, writes `workflow/verify.json` and emits one token
   per task: the task's current fingerprint when the task is complete, shows no trace of a run (no
   record, no failed record, no declared output) or is stale (its record carries another fingerprint);
   and `rerun:<SHA-256 of verify.json>` when it is damaged with the current fingerprint (a record whose
   outputs are missing or differ in size or hash, outputs without a record, a failed record, or a task
   the previous run's final check listed as missing). Every process takes its task's token as a `val`
   input, so the token is part of the cache hash. A complete task keeps its token and its cache. A stale
   task's token is a fingerprint it never ran with, so it misses the cache and reruns; using the new
   fingerprint itself, not a `rerun:` token, means that the next invocation finds its cache under the
   same key and does not run it a second time. A damaged task misses the cache and reruns. The consumers
   of a rerun task rerun because its emitted record changes, and a stale task's consumers are stale
   too (their fingerprints include its). A task that ran once and is intact gets the token it ran with,
   so a second invocation reruns nothing.
2. **`VERIFY_AFTER`, after the last task**, runs with `--final`: every task of the manifest must be
   complete, or the run fails (exit 1) naming each task that is not and recording them in
   `workflow/verify.json`. This catches the one case `VERIFY_BEFORE` cannot tell from a task never run:
   a task whose record and every output were deleted while its work folder remains, which `-resume`
   reuses; the next invocation's `VERIFY_BEFORE` reads the list and gives those tasks a `rerun:` token.

D8's rule stays the authority: completion is decided by the `verify` command on the records and files
in the output root, the same rule the commands apply; Nextflow's cache only saves rerunning a task that
the rule already finds complete. Both engines and the manual remedy share one staleness rule: a task is
complete only when its record says `succeeded`, lists its outputs intact, and carries the current
manifest's fingerprint. `verify` is a contract addition
([additions to the contract](#additions-to-the-contract)).

#### Choice: Nextflow's cache, deleted outputs and stale results

The cost of a check is what it reads: the measured direct outputs of one full-size FOV (aging
Position400 at `9a0ec77`, `runs/W-644/20261029T200039Z-9454d4c8/results/scratch/cleanup_aging9a0ec77.files.csv`)
are `reference.tif` 151,001,250 bytes, `molecules.csv` 25,017,203 bytes and the candidates table
24,581,676 bytes, about 200 MB per FOV with the records and stages; the chunked checkpoint (5.4 GB per
FOV) is temporary and exempt.

| Option | Cost | Consequence |
| --- | --- | --- |
| A. `-resume` alone, the gap documented, with the manual remedy: rerun without `-resume`, or delete the work folders of the affected tasks (`nextflow log <run> -f workdir,tag`) and resume | None | With the fingerprint in each task entry, a changed scientific setting or an input replaced at the same path (another size or time, older included) changes the cache key, so those tasks and their consumers rerun; a deleted output under `roots.output` goes unnoticed until someone reads it, a terminal output never; a record stale for any other reason is not checked; D8 holds only for tasks that run |
| B. `cache 'deep'` on the `needs` records | Nextflow reads each staged record (a few kB) | The records are hashed by content instead of path, size and time, so a producer that reran with a new record reruns its consumers even if the file's time was kept; for a changed setting or a replaced input it adds nothing to the fingerprint; it sees nothing under `roots.output`, so no deleted output is caught: the gap of A stays |
| C. A status check inside each process: every process `cache false`, its script first runs `starfinder workflow verify --task <id>` and exits 0 when the task is complete | One scheduler job per task at every invocation (on UGER thousands of queued jobs for a resumed full dataset, each waiting in the queue to stat or hash about 200 MB per FOV) | Catches every deletion, terminal tasks included, and every stale record (its fingerprint against the manifest's), and reruns the task; Nextflow's cache is off, and a consumer of a rerun producer reruns only because its own fingerprint changed with its producer's |
| D. Two uncached `VERIFY` jobs, before fan-out and at the end, with a token per task in the cache hash (recommended) | Two local jobs per invocation; a stat of every listed output (a few thousand for a dataset; seconds), or with `--hash` a read of about 200 MB per FOV (on `/stanley` about 2 s per FOV at 100 MB/s, about 35 min for 1,000 FOVs) | Catches a changed scientific setting, an input replaced at the same path (another size or an older or newer time), a deleted, truncated or resized output, a deleted record and a failed task before fan-out, and reruns exactly the stale or damaged tasks and their consumers; a record whose fingerprint differs fails `verify`; the end check makes a task deleted whole fail the run loudly, and the next invocation reruns it; a same-size corruption is caught only with `--hash` |

Recommendation: D, with sizes by default and `--hash` before results are delivered.

### The stub-run

Each process's `stub:` block runs `nextflow/bin/stub_task.sh`, which reads the task from the
manifest and writes empty files at its declared outputs and a task record with `status: succeeded`
and `"stub": true`, without the package. `nextflow run nextflow/main.nf -stub-run -profile local
--manifest <manifest>` then exercises the channels (including a producer that feeds several
tasks), the ordering, the seed depths, the supplied-step positions and the refusals on every manifest
the plan writes, in seconds and without data; it is the validation design's N
layer, in its own CI job with Java and Nextflow.

### The profiles

| Profile | Settings |
| --- | --- |
| `local` | `executor 'local'`; `apptainer.enabled` or `docker.enabled` with the image, or neither (the locked environment on `PATH`); `executor.cpus` the host's allowance |
| `sge` | `executor 'sge'`; `penv 'smp'`; `clusterOptions` `-P broad -S /bin/bash -l h_vmem=<memory / cpus>M -l h_rt=<time>` (as W-641's run used: `-P broad -l h_vmem=1G -l h_rt=00:10:00 -S /bin/bash`, two tasks on UGER); the label `gpu` adds `-l gpu=TRUE,operating_system=RedHat8`; `apptainer.enabled`, `apptainer.cacheDir` and `NXF_HOME` on `/stanley`, `apptainer.runOptions '--bind /stanley,/local/scratch'`; `executor.queueSize 200`, `pollInterval '30 sec'`; `params.max_retries 2` |
| `google-batch` | [Google Batch readiness](#google-batch-readiness) |

The Nextflow head on UGER is a job itself (W-641's `nextflow_head.sh`: `use UGER`, `use Java-17`,
`NXF_HOME` on `/stanley`, `NXF_OPTS="-Xms256m -Xmx1g"`), requested with `h_vmem=8G` (the JVM, W-641).

## The environment (D9)

### The lock change

Snakemake 9 and its cluster-generic executor join the lock in a `workflow` dependency group (D9).
The lock pins `packaging==26.0`, which no release of `snakemake-interface-common` accepts:
releases up to 1.23.0 need `packaging` below 26, and 1.23.1 and later need 26.1 or later (W-642's
golden scratch environment had to use `packaging` 25.0 with 1.22.0,
`runs/W-642/20261029T1750Z-golden-env/environment.json`). Five scratch resolutions with `uv lock`
0.9.28 on a copy of `src/python/pyproject.toml` and `uv.lock` (the project's files unchanged before
and after, each checked against `56c35b8`):

(orchestration-engines-lock)=

| Variant | `workflow` group | Other change | `uv lock` | Changed locked packages | Added |
| --- | --- | --- | --- | --- | --- |
| `snakemake-9.15.0` | `snakemake==9.15.0`, `snakemake-executor-plugin-cluster-generic==1.0.9` | — | resolves | `packaging` 26.0 → 26.3 (`snakemake-interface-common` 1.24.0) | 25 |
| `snakemake-9.15.0-packaging-lt-26` | as above | constraint `packaging<26` | resolves | `packaging` 26.0 → 25.0 (`snakemake-interface-common` 1.23.0) | 25 |
| `snakemake-9.15.0-packaging-ge-26.1` | as above | constraint `packaging>=26.1` | resolves | `packaging` 26.0 → 26.3 (1.24.0) | 25 |
| `snakemake-9.27.1` | `snakemake==9.27.1`, cluster-generic 1.0.9 | — | resolves | `packaging` 26.0 → 25.0 (1.23.0) | 27 (`greenlet`, `sqlalchemy`, `sqlmodel` more, `reretry` less) |
| `recommended` | as `snakemake-9.15.0` | `jsonschema>=4.18` and `pyyaml>=6.0` added to the package's dependencies | resolves | `packaging` 26.0 → 26.3 (1.24.0) | 25 |

In every variant no locked package other than `packaging` changes version and none is removed; the
added packages are Snakemake's (in the recommended variant: `snakemake` 9.15.0,
`snakemake-executor-plugin-cluster-generic` 1.0.9, `snakemake-interface-common` 1.24.0,
`-executor-plugins` 9.4.0, `-logger-plugins` 2.1.0, `-report-plugins` 1.3.0, `-scheduler-plugins`
2.0.2, `-storage-plugins` 4.4.1, and `argparse-dataclass`, `conda-inject`, `connection-pool`, `dpath`,
`gitdb`, `gitpython`, `humanfriendly`, `immutables`, `pulp` 3.3.2, `pyreadline3`, `reretry`,
`smart-open`, `smmap`, `tabulate`, `tenacity`, `throttler`, `yte`). `jsonschema` 4.26.0 and `pyyaml`
6.0.3 are already in the lock; the plan imports both (it validates the configuration with
`jsonschema` and reads YAML, the contract's V9), and neither is a direct dependency of the package
today (`src/python/pyproject.toml`, `[project] dependencies`), so the recommended variant declares them.

% results/lock/<variant>/diff.json, uv-lock.log, export.txt (steps lock_<variant>; the first attempt of
% snakemake-9.15.0 ran with UV_LOCKED set by the shell and only checked the lock: it is kept as failed)

#### Choice: the lock change

| Option | Cost | Consequence |
| --- | --- | --- |
| A. The recommended variant: Snakemake 9.15.0 and cluster-generic 1.0.9 in `workflow`, `packaging` 26.3 as `uv lock` chooses it, `jsonschema` and `pyyaml` declared (recommended) | `packaging` moves forward by three patch releases | The Snakemake that recorded the golden baseline and W-641's dry runs; no constraint to maintain; the plan's imports declared |
| B. As A with `packaging<26` (25.0) | A constraint in `[tool.uv]` that holds every package below `packaging` 26 | The `packaging` of `starfinder-v9` and of the golden scratch environment; a later release of anything that needs `packaging>=26` cannot be locked |
| C. Snakemake 9.27.1 (the newest 9.x on 2026-10-30) | Two more packages than A | `packaging` 25.0 by the resolver; the golden baseline was recorded with 9.15.0 and would be compared across a Snakemake change |

Recommendation: A. The batch-preparation draft lands it with the golden comparison tests and the CI
change, under the complete suite.

**The CI change.** The CI installs `--extra dev --extra local-registration --extra checkpoint`
(`.github/workflows/tests.yml:48-49`, `:96-97`). With the lock change both jobs add `--group
workflow`, so the contract tests run a real Snakemake dry run (D10); the `pytest-ci-environment` check
follows. A third job installs Java 17 and Nextflow 26.10.4 for the stub-run (TG5).

### The image

**A Containerfile from `uv.lock`.** A sketch, to land with task group 3 (no Containerfile is added by
this batch):

```dockerfile
FROM ghcr.io/astral-sh/uv:0.9.28@sha256:59240a65d6b57e6c507429b45f01b8f2c7c0bbeee0fb697c41a39c6a8e3a4cfb AS uv
FROM python:3.12-slim-bookworm@sha256:e697c81897302a4c4333bade000d98038fbb21eb064f8ee1fd42d02ebcd9d337 AS build
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_LOCKED=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never UV_PROJECT_ENVIRONMENT=/opt/starfinder
WORKDIR /src/python
COPY src/python/pyproject.toml src/python/uv.lock ./
RUN uv sync --no-install-project --no-default-groups --group workflow \
      --extra ome --extra local-registration --extra checkpoint --extra anndata \
      --extra multiview-stitcher --extra spatialdata
COPY src/python/ ./
RUN uv sync --no-default-groups --group workflow --no-editable \
      --extra ome --extra local-registration --extra checkpoint --extra anndata \
      --extra multiview-stitcher --extra spatialdata
FROM python:3.12-slim-bookworm@sha256:e697c81897302a4c4333bade000d98038fbb21eb064f8ee1fd42d02ebcd9d337
COPY --from=build /opt/starfinder /opt/starfinder
ENV PATH=/opt/starfinder/bin:$PATH OMP_NUM_THREADS=1 MPLBACKEND=Agg
LABEL org.opencontainers.image.revision=<commit> org.starfinder.uv-lock-sha256=<sha256 of uv.lock>
```

The digests are the index digests of the two public images read from their registries on
2026-10-30 (`results/image/preflight.json`; W-641's trivial build used `python:3.12-slim` at an
earlier digest). The image holds the package, its locked dependencies and Snakemake (the `workflow`
group), so a Snakemake job on a UGER node runs inside it ([below](#apptainer-on-uger)); it holds no
Fiji, MATLAB, Java or Nextflow. The weights of learned methods stay outside it
(`STARFINDER_WEIGHTS_DIR`, bound at run time).

**The digest.** The build exports an OCI archive (`docker buildx build --output
type=oci,dest=<file>`), never a tag in a registry: the image's identity is the manifest digest of
that archive, which `--image` passes to every task and the task record keeps. Publishing to a
registry needs its own authorization (D9).

**Size.** Estimated from the recommended lock (compressed wheel sizes of each set for CPython 3.12
on Linux x86_64, scaled by the ratio of installed to compressed bytes of W-642's golden scratch
environment, 3.24; `results/image/size-estimate.json`), plus about 120 MB of base (W-641's trivial
build):

| Content | Packages | Wheels | Installed (estimate) |
| --- | --- | --- | --- |
| Core and `workflow` | 104 | 0.18 GB | 0.57 GB |
| CPU extras (`ome`, `local-registration`, `checkpoint`, `anndata`, `multiview-stitcher`, `spatialdata`) and `workflow` | 178 | 0.47 GB | 1.51 GB |
| As above and the learned extras (`stardist`, `cellpose`, `spotiflow`, `piscis`; torch from the CPU index) | 242 | 1.32 GB | 4.27 GB |
| As above and `registration-elastix` | 250 | 1.63 GB | 5.26 GB |

#### Choice: the image's extras

| Option | Cost | Consequence |
| --- | --- | --- |
| A. The CPU extras and `workflow`, about 1.6 GB (recommended) | Learned methods do not run in the image | Every deterministic task kind runs; a learned run (StarDist, Cellpose, learned detectors) needs another environment, which the first implementation does not provide on UGER |
| B. A and the learned extras, about 4.4 GB | Three times the size; TensorFlow and torch in one image | Learned methods run on CPU in the image (the lock takes torch from the CPU index on Linux, `[tool.uv.sources]` of `pyproject.toml`) |
| C. Two images, A and B | Two builds and digests | A small default image and a learned one chosen per task by `execution.tasks.segment.container` |

Recommendation: C if learned segmentation is run on UGER in the first implementation, A otherwise;
the gate chooses.

**A GPU variant.** W-641 shows UGER's GPUs usable with conditions (RHEL 8 nodes; Apptainer `--nv`
there not verified). A GPU image also cannot come from today's lock: torch is locked from the CPU
index on Linux (`[tool.uv.sources]`, `src/python/pyproject.toml`), and TensorFlow's CUDA libraries
are not locked. The plan defers it: the first implementation runs `segment` with `device: cpu`, and
task group 4 checks `--nv` on a GPU node; a GPU variant then needs its own lock decision (a CUDA
index for torch with uv's conflicting extras, and TensorFlow's `and-cuda` extra).

**The scratch image of this batch was not built.** The batch builds it only if `df -h /` shows at
least 40 GB free before the build; it showed 31 GB on 2026-10-30 (`results/image/preflight.json`).
Docker builds on GP099 (W-641), and BuildKit there is `buildx` 0.9.1 with the `docker` driver, which
cannot export an OCI archive: the build needs a `docker-container` builder (`docker buildx create
--driver docker-container`), removed with its cache after the build (`docker buildx rm`).

### Apptainer on UGER

Apptainer 1.3.2 runs on UGER compute nodes, not on the login node, and runs Docker images (W-641).
The OCI archive is copied to `/stanley/WangLab/jiahao/test/starfinder_benchmark/containers/` and
turned into a SIF on a compute node (`apptainer build starfinder-<digest>.sif
oci-archive://<archive>`), without a registry; the SIF's SHA-256 is recorded beside the archive's
digest. Jobs run `apptainer exec --cleanenv --bind /stanley,/local/scratch <sif> …`;
`APPTAINER_CACHEDIR` is on `/stanley` (the Broad home has 933 MB free, W-641) and `APPTAINER_TMPDIR`
is the job's `/local/scratch/<job>`.

**Why the image, and not an environment, on UGER.** The UGER CPU nodes run RHEL 7.9 (glibc 2.17;
W-641's `uger-d003`). Of the 178 packages of the recommended image's set, ten have no wheel for glibc
2.17 in the lock: `scipy` 1.17.0, `scikit-image` 0.26.0, `scikit-learn` 1.8.0, `h5py` 3.15.1,
`pandas` 3.0.0, `contourpy` 1.3.3, `imagecodecs` 2026.1.14, `pyarrow` 23.0.0, `pyproj` 3.7.2 and
`ujson` 5.11.0 need glibc 2.24 to 2.28; the learned extras add nine more, torch 2.7.1 among them
(`results/image/glibc.json`). Three pure-Python packages have only an sdist (`connection-pool`,
`flowio`, `pims`) and build anywhere. A uv environment of the lock therefore cannot be installed on the RHEL 7
CPU nodes; the image brings its own glibc (Debian bookworm, 2.36). The RHEL 8 GPU nodes (glibc 2.28)
could hold one.

### The conda environment files

No rule declares `conda:`; the UGER profile deploys with conda and its job script activates
`/stanley/WangLab/envs/starfinder-v9` (`profile/broad-uger/config.yaml:11`,
`broad-jobscript.sh:7-11`); `envs_path` is read by no rule (`config.schema.yaml:27-29`; the
baseline's O7).

| File | What it describes | Plan |
| --- | --- | --- |
| `config/environment-v9.yaml` | The conda environment `starfinder-v9` (Snakemake 9, cluster-generic, the GCS storage plugin, a scientific stack unpinned) | Kept as the frozen MATLAB workflow's environment, moved beside it ([choice](#choice-the-environment-of-the-matlab-legacy-workflow)) |
| `config/environment.yaml` | An older full conda export (384 lines) | Retired |
| `config/environment.yml` | An older full conda export (343 lines) | Retired |
| `config/stardist.yml` | A StarDist conda environment | Retired: StarDist is the lock's `stardist` extra |
| `workflow/envs/starfinder.yml` | A conda export no rule names | Retired |
| `workflow/envs/stardist.yml` | A StarDist conda export no rule names | Retired |

`envs_path` leaves the Python dialect (the contract's migration table) and stays in the MATLAB
schema, unread, frozen.

#### Choice: the environment of the MATLAB legacy workflow

The MATLAB jobs load MATLAB from the host (`use Matlab`, `broad-jobscript.sh:13`) and run the shared
rules' Python adapters, which import the private legacy module from the checkout that
`starfinder_path` names.

| Option | Cost | Consequence |
| --- | --- | --- |
| A. Keep `starfinder-v9` as the legacy workflow's frozen environment: `config/environment-v9.yaml` moves to `workflow/envs/matlab-legacy.yaml` unchanged, and the legacy profile activates it (recommended) | One conda file stays | The legacy workflow keeps the environment it runs in today; the Python workflow never uses it |
| B. The legacy workflow in the image, with MATLAB bound in from the host | MATLAB's libraries and licence inside a container on RHEL 7 | Untested; a change D7 freezes |
| C. A uv environment of the lock | glibc 2.17 on the RHEL 7 nodes, where MATLAB runs | Not installable there (above) |

Recommendation: A.

### The Broad shared environment from the lock

The Python workflow on UGER runs in the image; the shared environment is for interactive use on
Broad (notebooks, a task by hand) on RHEL 8 hosts, built from the lock without conda:

1. On a RHEL 8 node (`qrsh -l operating_system=RedHat8 -l h_vmem=8G`), with `UV_CACHE_DIR` and
   `UV_PYTHON_INSTALL_DIR` under `/stanley/WangLab/envs/.uv/`: install uv 0.9.28 (the CI's version)
   into `/stanley/WangLab/envs/.uv/bin`.
2. Check out the commit to deploy and note it; `UV_PROJECT_ENVIRONMENT=/stanley/WangLab/envs/starfinder-<short commit>`.
3. `uv sync --locked --no-default-groups --group workflow --extra <the image's extras>` from
   `src/python` (Python 3.12 from uv's own build).
4. `uv pip check`; `starfinder --help`; `snakemake --version` equals 9.15.0; write
   `environment.json` beside it (commit, `uv.lock` SHA-256, extras, `uv pip freeze`).
5. Point `/stanley/WangLab/envs/starfinder-current` at it only after the checks; keep the previous one
   until the next is accepted.

### Nextflow and Java

| Item | Pin | Where |
| --- | --- | --- |
| Nextflow | `NXF_VER=26.10.4` (build 14435); `nextflow.config` holds `manifest.nextflowVersion = '!>=26.10.4'` | Broad: the self-contained release `starfinder_benchmark/tools/nextflow/26.10.4/nextflow` (SHA-256 `2cbda6e7fdbbb1e7a413973eff549653f566ba6bb7bdfbb58eb78d0f831fdec4`, W-641), `NXF_HOME` on `/stanley`; GP099: `~/.local/share/w641-nextflow`; CI: the release downloaded by version |
| Java | 17 or later (Nextflow 26 needs 17) | Broad: `use Java-17` (OpenJDK 17.0.8); GP099: OpenJDK 23 in the conda prefix; CI: Temurin 17 |

Neither enters `uv.lock` (D9). `NXF_DISABLE_CHECK_LATEST=true` on UGER.

## The profiles

### The Snakemake UGER profile

Today's profile (`profile/broad-uger/`) has four defects for Snakemake 9.15.0:

| Defect | Where | Repair |
| --- | --- | --- |
| The submit and status commands are paths relative to the working directory (`profile/broad-uger/broad-submit.py {dependencies}`) | `config.yaml:7-8` | Bare file names, `broad-submit.py` and `broad-status.py`: Snakemake resolves a string value that names a file in the profile folder against that folder (Snakemake 9.15.0, `snakemake/profiles.py:69-72`); `{dependencies}`, which the cluster-generic executor fills only with `immediate-submit` (off, `config.yaml:29`), is dropped |
| The job script is a `default-resources` string, which selects no template (`design/specs/broad-uger.md`, "Run") | `config.yaml:14-15` | `jobscript: broad-jobscript.sh`, a profile key resolved against the profile folder the same way |
| `default-resources` holds no resource | `config.yaml:14-15` | `default-resources: [mem_mb=8000, runtime=30]`, the values of `DEFAULT_RESOURCES` (`common.smk:156-159`), for the MATLAB rule set's rules without their own; the Python rules always carry the manifest's resources |
| conda deployment, `notemp: true`, three retries; the job script activates a conda environment and loads MATLAB for every job; logs under `~/snakemake-logs/` (`broad-submit.py:69-74`) on a home with 933 MB free | `config.yaml:11`, `:20`, `:26`; `broad-jobscript.sh:5-13`; `broad-submit.py:56-74` | No software deployment; `notemp` removed; `retries: 2`; the job script runs the job in the image; job logs under `<output root>/workflow/logs/engine/` |

#### Choice: one profile or two

| Option | Cost | Consequence |
| --- | --- | --- |
| A. `profile/broad-uger/` for the Python workflow (the image), and `profile/broad-uger-matlab/` frozen for the legacy workflow (today's files with the three path repairs only) (recommended) | Two profile folders | The Python job script has no conda and no MATLAB; the legacy one keeps `use Matlab` and the frozen environment |
| B. One profile whose job script branches on the rule | Shell logic keyed on rule names | A Python job could load MATLAB, as every job does today |

Recommendation: A.

The Python job script, in outline:

```bash
#!/usr/bin/env bash
# properties = {properties}
source /broad/software/scripts/useuse; use UGER
export STARFINDER_IMAGE=<digest> APPTAINER_CACHEDIR=/stanley/WangLab/.apptainer APPTAINER_TMPDIR=/local/scratch/$JOB_ID
export OMP_NUM_THREADS=<threads> CUDA_VISIBLE_DEVICES=<the free GPU, for a gpu job>
apptainer exec --cleanenv --bind /stanley,/local/scratch [--nv] <sif> bash -c '{exec_job}'
```

`{exec_job}` runs Snakemake inside the image, which has the `workflow` group, so the node needs no
Python environment (glibc 2.17, above). The submit script keeps its logic
(`broad-submit.py`: `-pe smp <threads>`, `h_vmem` = `mem_mb / threads`, `h_rt` from `runtime`,
reservation above 15 GiB or four threads) and adds `-l gpu=TRUE,operating_system=RedHat8` for a job
with the resource `gpu`, and `-l operating_system=RedHat8` for nothing else. The Snakemake head runs
as a UGER job with the same image (`h_vmem=8G`), so the profile works from any host that can submit.

### Nextflow's `sge` configuration

As in [the profiles](#the-profiles) above: `penv 'smp'`; `clusterOptions` built from `task.cpus`,
`task.memory` and `task.time` into `-P broad -S /bin/bash -l h_vmem=<memory / cpus>M -l h_rt=<time>`,
because `h_vmem` is a per-slot consumable on UGER (W-641); the label `gpu` adds `-l
gpu=TRUE,operating_system=RedHat8` and `apptainer.runOptions` `--nv`; `apptainer.enabled true` with
the SIF of the image; job logs are Nextflow's `.command.log` in the work folder, which sits under
`/stanley` (`-work-dir`), and the task logs are the commands' own under the output root.

### Google Batch readiness

No cloud resource is used by this batch or by the first implementation; this section makes the
`google-batch` profile ready to try once GCP is authorized.

**The profile.** `process.executor 'google-batch'`; `google.project`, `google.location`
(`us-central1`); `workDir 'gs://<bucket>/work'`; `google.batch.spot true` for retryable tasks with
`maxRetries` covering preemption; `process.container` the image in Artifact Registry by digest
(`us-central1-docker.pkg.dev/<project>/<repository>/starfinder@sha256:…`); `machineType` per label
(`n2-highmem-*` for `fov` and `tile_global`, whose peaks reach about 9 GiB on a full-size FOV, the
tiled record's resource model); the label `gpu`: `accelerator 1, type: 'nvidia-l4'` (or the GPU the
project's quota allows) with a G2 machine.

**GCS path conventions.**

| What | Path |
| --- | --- |
| Inputs | `gs://<bucket>/data/<dataset_id>/<sample_id>/`, the layout of `root_input_path/dataset_id/sample_id` |
| Outputs | `gs://<bucket>/results/<dataset_id>/<output_id>/`, the output root's layout (`_layout`) |
| Nextflow work | `gs://<bucket>/work/<run>/`, deleted after a run is accepted |
| Image | Artifact Registry, by digest |

The commands write in place under `roots.output` (the contract), which on Google Batch is a GCS
bucket mounted with Cloud Storage FUSE. The chunked checkpoint (HDF5, written by window) and the
atomic renames of the contract need a local disk: the readiness issue decides between writing the
output root to a local or persistent disk and copying it to GCS at the end of each task, and running
the tiled route only with a persistent disk.

**The checklist** (each item recorded by the owner before the first cloud run):

| Area | Item |
| --- | --- |
| Access | The GCP project and its billing account; a service account for Batch with Batch Job Editor, Service Account User, Logs Writer, Storage Object Admin on the bucket and Artifact Registry Reader; the Batch, Compute, Storage and Artifact Registry APIs enabled |
| Authorization | Publishing the image to Artifact Registry (D9: its own authorization); the data that may leave Broad (real data needs the lab's data-use approval) |
| Budget | A budget with alerts on the project; the quotas of CPUs, memory and GPUs in the region; spot or standard VMs; an estimate per FOV from the task records' `wall_s` and `peak_rss_kib` (the tiled record's resource model) |
| Data transfer | The transfer from `/stanley` to GCS (`gcloud storage rsync`), its size (aging: 5.2 GiB per FOV, W-641) and time; egress of results back to Broad; the bucket's location in the Batch region (no cross-region charges); the lifecycle rule of `work/` |
| Records | The task records' `image` (the Artifact Registry digest) and `host` on Batch VMs; `wall_s` and cost per task kind |

## The MATLAB legacy separation

**The private legacy module.** The adapters that the frozen MATLAB workflow's shared rules call move
unchanged from `starfinder/_snakemake.py` into one private module, working name
`starfinder/_legacy_workflow.py`, which the Python workflow and the `starfinder workflow` commands
never import. Its contents, from the baseline's classification (`orchestration.md` of
`design/baselines/`, "The Snakemake adapters and the shared definitions") and the contract's
traceability table (the rows "The private MATLAB legacy module, unchanged (D7)"):

| Function | Today at | Called by |
| --- | --- | --- |
| `_read_stack` | `_snakemake.py:141` | `_run_nuclei_amplicon_overlay` |
| `_run_nuclei_amplicon_overlay` | `_snakemake.py:148` | `scripts/create_nuclei_amplicon_overlay.py` |
| `_run_enhance_dapi_with_flamingo` | `_snakemake.py:168` | `scripts/enhance_dapi_with_flamingo.py` |
| `_run_stardist_segmentation` | `_snakemake.py:178` | `scripts/stardist_segmentation.py` |
| `_fov_sample` | `_snakemake.py:227` | `_run_reads_assignment` (the annotation branch) |
| `_tile_record` | `_snakemake.py:254` | `_run_reads_assignment` |
| `_legacy_cells` | `_snakemake.py:264` | `_run_reads_assignment` |
| `_in_box`, `_cell_obs`, `_write_raw_h5ad` | `_snakemake.py:318`, `:324`, `:359` | `_run_reads_assignment` |
| `_write_reads_assignment`, `_assignment_ratio`, `_write_assignment_figures` | `_snakemake.py:390`, `:447`, `:455` | `_run_reads_assignment` |
| `_run_reads_assignment` | `_snakemake.py:483` | `scripts/reads_assignment.py` |
| `_shared_image_inputs` | `_snakemake.py:654` | `segmentation.smk` (`SHARED_IMAGE_INPUTS`, `common.smk:408-410`) |

With them go the helpers they call that the Python commands do not keep (the implementation lists
them by the import graph), and the MATLAB-backend translation they read
(`starfinder/dataset/_legacy_config.py`, `legacy_processing`, `matlab_reference_stains`), which
stays where it is because `from_workflow_config`'s MATLAB branch reads it. The four adapter scripts
change one import line each (`from starfinder._snakemake import …` to `from
starfinder._legacy_workflow import …`); their behaviour does not change.

**The pinning tests.** `test_assignment_workflow.py` (`from starfinder._snakemake import
_run_reads_assignment, _write_raw_h5ad`, line 22) and `test_workflow_assembly_golden.py` (`from
starfinder._snakemake import _run_reads_assignment`, line 49) import from the legacy module instead
and are otherwise unchanged; they pin the legacy outputs (`test_a18_the_legacy_outputs_equal_the_frozen_helper`,
`test_the_adapter_writes_the_legacy_values`). Their Python-route tests
(`test_the_python_route_reads_molecules_csv_by_key`) move to the `assign` command's tests.

#### Choice: the name of the legacy module

| Option | Cost | Consequence |
| --- | --- | --- |
| A. `starfinder/_legacy_workflow.py` (recommended) | One module | Its name says what it is for; private |
| B. Keep `starfinder/_snakemake.py` with only the legacy functions | None | The name suggests the Snakemake interface of the package, which the Python workflow no longer has |

Recommendation: A.

**The files beside the frozen workflow (OC6).** `cleanup_subtile.py`, `create_BigStitcher_macro.py`,
`workflow/scripts/test/` and `workflow/notebook/` stay unchanged beside the MATLAB files; the
legacy-module draft lists them for Jiahao's decision on their fate. `create_segmentation_preview.py`
is retired (no rule calls it, the baseline's O21). `workflow/rules/gcloud-backup.smk` leaves
`workflow/` (OC1): it belongs to neither workflow and opens a Google Cloud Storage client when parsed
(`gcloud-backup.smk:18-22`); its own draft removes it, or moves it to a lab data-tools location if
Jiahao says the backup is still used.

## Additions to the contract

The engines need six additions to the contract of W-643 (Proposed). Each is listed for the gate;
the implementation of task group 2 lands them with the contract.

| Addition | Why | Options | Recommendation |
| --- | --- | --- | --- |
| A fifth context argument, `--engine-version` | The contract records `engine_version` in the execution context but passes four arguments | A. `--engine-version V`; B. `--engine NAME@V`; C. the command reads it from the environment | A |
| The manifest's `groups`: per group the kind, wildcards, output templates, record and log templates, and the temporary outputs; each task names its group | Snakemake needs output patterns ([the choice](#choice-how-snakemake-knows-each-tasks-files)) | A. In the manifest, checked by the plan; B. derived by `tasks.smk` from the tasks (logic in an engine file) | A |
| A `temporary` flag on declared outputs | Snakemake's `temp()` for the chunked checkpoint | A. Per group in `groups`; B. per task | A |
| The plan reads one configuration file | `workflow.configfiles` may hold several, and `--config` overrides change nothing in the file the plan reads | A. The Snakefile refuses several files and `--config` for the Python backend, naming the edit; B. the plan merges them | A (the contract's changed statement for `workflow-configuration.md:864-866`) |
| A command `verify` (`--manifest`, `--task`, `--hash`, `--final`) that applies D8's completion rule and the staleness rule (the record's fingerprint against the manifest's) from the output root and, for Nextflow, writes the per-task tokens | Nextflow's `-resume` does not see the output root ([the choice](#choice-nextflows-cache-deleted-outputs-and-stale-results)) | A. A package command; B. a check written in `main.nf` (completion logic in an engine file); C. none (the gap documented) | A |
| `tasks[].fingerprint` in the manifest and `workflow.fingerprint` in the task record | The configuration hash and the inputs' sizes and times are outside the task entries, so neither engine's cache nor `verify` would see a changed setting or a replaced input ([the task fingerprint](#the-task-fingerprint)) | A. One fingerprint per task, written by the plan and kept in the record; B. each engine derives its own key from the manifest's top-level fields (logic in the engine files, two rules); C. none | A |

## The site pages for task group 6

`docs/contributing.md` ("Documentation pages") and `design/README.md` ("How this folder relates to the
site") set the page model: a short guide (about 1,500 words, the fixed sections) and a reference of
tables generated from the code where possible (`docs/_ext/`), written from the implemented code by the
batch's last issue, linking this record by its GitHub URL.

| New page | Content | Generated from the code |
| --- | --- | --- |
| `docs/workflow/index.md`, the guide (rewritten) | Lead; Quick start (plan, a Snakemake dry run, a local run of the per-fov example; the same with Nextflow); What it does (plan, manifest, tasks, records); What you get (`workflow/`, the task records and logs); Main options (the blocks and `execution`); Limits (Fiji only in Snakemake, MATLAB legacy, GPU); How it is checked; links to this record, the contract and the validation design | — |
| `docs/workflow/reference.md`, the reference | The commands (from the CLI's parser), the task kinds with their declared outputs, the configuration keys of each block and of `execution` (`config-options`), the exit codes, the profiles' settings, the migration table from the old keys | The commands and the configuration keys |

| Current page | Fate |
| --- | --- |
| `docs/workflows.md` (the Snakemake guide) | Replaced by the guide |
| `docs/workflow-configuration.md` | Its Python keys move to the reference; its MATLAB keys stay on a legacy page |
| `docs/workflow-downstream.md` | Replaced by the guide's segment and assign tasks and the reference; the legacy rules move to the legacy page |
| `docs/workflow/snakemake.md`, `docs/workflow/nextflow.md` | Replaced by the guide's two Quick start routes |
| `docs/workflow/broad-uger.md` | Kept as the UGER guide, rewritten for both engines and the image |
| new `docs/workflow/matlab-legacy.md` | The frozen MATLAB workflow: its configuration, launcher and profile |
| `docs/get-started/installation.md`, "Snakemake" | The `workflow` group and the image instead of `config/environment-v9.yaml` |
| `docs/migration.md` | Gains the migration table of the contract (generated from it) |

The records of today's workflow (`design/specs/workflows.md`, `workflow-configuration.md`,
`workflow-downstream.md`, `broad-uger.md`, `checkpoints.md`) take the changed statements of the
contract (`orchestration.md`, "Changed statements of the existing records"), each with the issue that
lands its part; `broad-uger.md:12-17` takes this wording: "The workflow runs in the image of the lock
through Apptainer on UGER compute nodes; the head of either engine runs as a UGER job. The MATLAB
legacy workflow keeps its conda environment and profile."

## Task groups and draft issues

The implementation follows the gate W-646 (D11) in task groups 2 to 6, a validation issue and a
results gate; the drafts, one file each in the run directory (`drafts/`), replace the placeholder
issues W-647 to W-653 after approval. Their order, dependencies and the map from each deliverable of
D1 to D10 to a draft are in `drafts/README.md`.

## Open points

* **E-1.** Whether Snakemake 9.15.0 treats a missing `temp()` output of a completed producer as
  up to date when the profile keeps `notemp` off and the merge has removed the file first; the
  contract test of task group 2 checks it with a dry run after a completed tiled run (it must
  schedule nothing).
* **E-2.** Apptainer `--nv` on the RHEL 8 GPU nodes (W-641 did not try it); task group 4 checks it
  before any GPU task runs in the image.
* **E-3.** Whether UGER has RHEL 8 CPU nodes for the shared environment; task group 4 asks.
* **E-4.** The scratch image of this batch was not built (less than 40 GB free); task group 3
  builds it, records its size and digest, and replaces the estimate above.
* **E-5.** Whether the lab's Fiji runs headless inside the image with `/stanley` bound. If it does
  not (a system library its Java needs missing from the slim base), the image adds that library, or the
  Fiji task takes the `fiji-bigstitcher` image (the Fiji choice's B); task group 4 checks it.
