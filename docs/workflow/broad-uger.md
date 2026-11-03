# Broad UGER

:::{container} sf-lead
Run the workflow on Broad's UGER cluster with either engine. The head of the
engine runs as a UGER job, and every task runs in the workflow image built from
the package's lock, through Apptainer on the compute nodes.
:::

This page says how the profiles work on UGER. Both engines, the image's SIF, the
memory limits, the GPU route and Fiji in the image were run on the cluster in
November 2026; where a run found a limit, the page states it.

## Quick start

1. **The image.** Build the SIF from the CPU image's OCI archive on a Broad
   compute node, under `/stanley`, and record its checksum. The CPU image holds
   the CPU extras and the `workflow` group; a second image, `learned`, adds the
   learned extras on CPU, and a third, `learned-cuda`, the same backends built for
   CUDA, for GPU tasks (below).
2. **Snakemake.** Run the head as a UGER job with 8 GiB of memory, in the same
   image, with the checkout and the data under `/stanley`. Set `STARFINDER_IMAGE`
   to the image's OCI manifest digest and `STARFINDER_SIF` to its SIF in the
   head's environment, with `STARFINDER_WEIGHTS_DIR`, a folder under `/stanley`
   that holds the weights, when a task uses a learned method; give `snakemake`
   absolute paths:
   `-s <checkout>/workflow/Snakefile --configfile <configuration> --profile <checkout>/profile/broad-uger`.
   The head submits the tasks from inside the image, so the image needs the
   cluster's scheduler commands: after `use UGER`, run it with
   `apptainer exec --cleanenv --bind /stanley,/broad/uge,/local/scratch`, put the
   folder of `qsub` first on the image's `PATH` and pass every `SGE_*` variable
   with `--env`. Point `XDG_CACHE_HOME`, `MPLCONFIGDIR` and `NUMBA_CACHE_DIR` at a
   folder under `/stanley`, not the home folder.
   End the run with `starfinder workflow verify --manifest <manifest> --final`, in
   the image: Snakemake reruns tasks with missing declared non-temporary outputs
   and their downstream, but does not read the task records or detect a deleted
   log or an output of another size (the {doc}`workflow guide <index>`, "Reruns").
   `--hash` also compares each output's SHA-256, not only its size.
3. **Nextflow.** Run the head as a UGER job, outside the image, with Java 17 or
   newer and Nextflow 26.10.4 or newer (`use UGER` and `use Java-17` give the
   scheduler commands and Java):
   `nextflow run <checkout>/nextflow/main.nf -profile sge --config <configuration> --container <SIF> -work-dir <folder under /stanley>`.
   The `sge` profile refuses to start without `--container`, which the plan and
   `verify` jobs use as well as the tasks. `-with-trace`, `-with-report` and
   `-with-timeline` need `ps` in the task's image: the images built from the
   current `Containerfile` hold it (procps), those built before November 3, 2026
   do not.

Without `STARFINDER_IMAGE`, the Snakemake jobs use the profile's default CPU image
digest; without `STARFINDER_SIF`, they look for
`/stanley/WangLab/jiahao/test/starfinder_benchmark/containers/starfinder-<digest hex>.sif`.
`execution.container`, when the configuration sets it, replaces the image of every
task; there is no image per task kind. Each engine reads it its own way. Under
Snakemake it is an OCI manifest digest; the submit script refuses anything else,
and `STARFINDER_SIF`, when set, is used for every job whatever the digest. Under
Nextflow the value goes to the `container` directive as it is, so under `sge` it
is the SIF's path on the shared drive. A configuration's value therefore suits one
engine.

## What it does

**Snakemake** (`profile/broad-uger/`, with the cluster-generic executor). The
submit script reads the job's threads and resources from Snakemake and its image
from the task's manifest entry, and calls
`qsub -P broad -S /bin/bash`, with `-pe smp <threads>` and linear CPU binding,
`h_vmem` per slot (the task's memory divided by its threads), `h_rt` from its
runtime, and a reservation above 15 GiB or at four threads or more. The status
script asks `qstat`, then `qacct`, and counts a job without accounting as still
running. The job script runs the task's command with `apptainer exec --cleanenv
--bind /stanley,/local/scratch`, passes the image, the job id, the thread
variables and `STARFINDER_WEIGHTS_DIR` explicitly, and exits with Apptainer's
status. `APPTAINER_CACHEDIR` defaults to `/stanley/WangLab/<user>/.apptainer`, in
the user's own lab folder, because `/stanley/WangLab` itself is not writable; the job
stops if that folder does not exist, so set `APPTAINER_CACHEDIR` to another
folder on `/stanley` then. Temporary files use the job's local scratch.

**Nextflow** (the `sge` profile of `nextflow/nextflow.config`). The SGE executor
submits each task with `-P broad`, `h_vmem` per slot and `h_rt`, and a task of more
than one CPU with the `smp` parallel environment, as the Snakemake profile does, and runs it with Apptainer, `/stanley` bound, in the task's
`execution.container` or the `--container` image.

**Memory.** UGER reserves `h_vmem` per slot and kills a job that uses more than it
is allowed. In the `smp` parallel environment the allowance includes one slot more,
for the job's master task: a task of one thread is held to its memory, and a task of
n threads to (n + 1)/n of it, under both engines.

**GPU tasks.** A GPU task requests `gpu=TRUE,operating_system=RedHat8` under both
engines and runs Apptainer with `--nv`. UGER assigns no device: the Snakemake job
script picks the first GPU with no memory in use, sets `CUDA_VISIBLE_DEVICES`, and
fails when there is none; under Nextflow the task sees every GPU of the node. A
Broad GPU node runs one job at a time, whatever its queue, so a GPU task has the
node and its four GPUs to itself and uses one of them.

A GPU task needs the `learned-cuda` image. `--nv` brings the driver but not the
CUDA runtime, and the `cpu` and `learned` images hold the CPU build of PyTorch and
TensorFlow without its CUDA libraries, so with them a task with
`segmentation.device: cuda` stops with "TensorFlow sees no GPU". The
`learned-cuda` image holds the lock's `cuda` extra (the
{doc}`installation guide <../get-started/installation>`, "Learned extras"): torch
built for CUDA 12.6 and TensorFlow with NVIDIA's CUDA libraries, besides the CPU
extras and the `workflow` group, so it runs every task of a run. Give it to a run
with GPU tasks as `STARFINDER_IMAGE` under Snakemake or `--container` under
Nextflow. StarDist and Cellpose run on the GPU; spot finding runs on the CPU. The
image is 4.4 GB as an OCI archive and 9.5 GB unpacked, against 1.5 and 4.8 GB for
`learned`.

## What you get

The outputs and records are those of a local run (the
{doc}`workflow guide <index>`, "What you get"). Each task writes
`workflow/logs/<task>.log`; under Snakemake, the jobs' standard output and error
go to `<output root>/workflow/logs/engine/`, with UGER's job id in their names, so
that retries keep their logs. Nextflow keeps each job's output in its work
folder.

## Main options

* The Snakemake profile's settings, listed in the
  {doc}`reference <reference>` ("Profiles"): default resources of 8000 MB and 30
  minutes, two retries, keep-going, rerun-incomplete, 1000 jobs, a 60-second
  latency wait and `rerun-triggers: [mtime, params, input]`. Give `--jobs` for
  the account's submission limit.
* Nextflow's `--max_retries` (default 2) and `--container`.
* `execution.defaults` and `execution.tasks.<kind>` of the configuration: each
  task's threads, memory, runtime and GPU. A retry multiplies memory and runtime
  by the attempt.

## Limits

* The lock's environment needs a newer glibc than the RHEL 7 CPU nodes have,
  which is why the tasks run in the image.
* The `learned-cuda` image was checked on a workstation GPU, an RTX A5000; it has
  not yet run on UGER's A30 nodes. Nextflow's trace, report and timeline have not
  yet run on UGER with the images that hold `ps`.
* Fiji runs only under Snakemake; under Nextflow, supply BigStitcher's
  `dataset.xml` (`stitching.dataset_xml`). BigStitcher's defaults
  (`stitching.config`) suit full-size tiles: its loader skips files under
  `exclude` KB (10), and its pairwise shifts run at a 4× downsampling. Fiji exits
  0 when its macro fails; the engine task then fails on its own check of
  `dataset.xml`.
* Under Nextflow, tasks of learned methods get no `STARFINDER_WEIGHTS_DIR` and so
  find no weights; run them under Snakemake, whose head passes the variable on.
* Under Snakemake the tasks see the home folder, and libraries write their caches
  under `~/.cache` there; the Broad home is small.
* The {doc}`MATLAB legacy workflow <matlab-legacy>` keeps its own profile,
  `profile/broad-uger-matlab/`, and its conda environment.

## How it is checked

Focused tests resolve the profile outside the checkout, check the scheduler
arguments the submit script builds, run the job script with a stand-in
Apptainer. They submit no UGER job and
run no GPU, SIF, MATLAB or Fiji job.

The cluster runs of November 2026 ran both engines on UGER, on a small synthetic
sample and on four full-size FOVs, direct and tiled, with equal outputs; a task
killed by `h_vmem` that succeeded on its retry with twice the memory; the GPU
route on an A30 node; the Fiji task inside the image; and the MATLAB workflow
under its frozen profile. On the small sample's 36 × 36 tiles BigStitcher found
no pairwise shift, so the Fiji task's successful stitch on UGER is not shown;
its failure path is.

[UGER design record](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/broad-uger.md).
