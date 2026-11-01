# Broad UGER

:::{container} sf-lead
Run the workflow on Broad's UGER cluster with either engine. The head of the
engine runs as a UGER job, and every task runs in the workflow image built from
the package's lock, through Apptainer on the compute nodes.
:::

This page says how the profiles are meant to work. The Snakemake profile is
checked locally with stand-in scheduler commands; a UGER submission under either
engine, a GPU job, the SIF and Fiji in the image have not been checked on the
cluster yet.

## Quick start

1. **The image.** Build the SIF from the CPU image's OCI archive on a Broad
   compute node, under `/stanley`, and record its checksum. The CPU image holds
   the CPU extras and the `workflow` group; a second image adds the learned
   extras, on CPU.
2. **Snakemake.** Run the head as a UGER job with 8 GiB of memory, in the same
   image, with the checkout and the data under `/stanley`. Set `STARFINDER_IMAGE`
   to the image's OCI manifest digest and `STARFINDER_SIF` to its SIF in the
   head's environment, and give `snakemake` absolute paths:
   `-s <checkout>/workflow/Snakefile --configfile <configuration> --profile <checkout>/profile/broad-uger`.
3. **Nextflow.** Run the head as a UGER job with Java 17 or newer and Nextflow
   26.10.4 or newer:
   `nextflow run <checkout>/nextflow/main.nf -profile sge --config <configuration> --container <SIF> -work-dir <folder under /stanley>`.
   The `sge` profile refuses to start without `--container`, which the plan and
   `verify` jobs use as well as the tasks.

Without `STARFINDER_IMAGE`, the Snakemake jobs use the profile's default CPU image
digest; without `STARFINDER_SIF`, they look for
`/stanley/WangLab/jiahao/test/starfinder_benchmark/containers/starfinder-<digest hex>.sif`.
A task's `execution.container`, when the configuration sets it, selects the image
digest under both engines; keep the head's image and the SIF consistent with it.

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
status. `APPTAINER_CACHEDIR` defaults to `/stanley/WangLab/.apptainer`; temporary
files use the job's local scratch.

**Nextflow** (the `sge` profile of `nextflow/nextflow.config`). The SGE executor
submits each task with the `smp` parallel environment, `-P broad`, `h_vmem` per
slot and `h_rt`, and runs it with Apptainer, `/stanley` bound, in the task's
`execution.container` or the `--container` image.

**GPU tasks.** A GPU task requests `gpu=TRUE,operating_system=RedHat8` under both
engines and runs Apptainer with `--nv`. UGER assigns no device: the Snakemake job
script picks the first GPU with no memory in use, sets `CUDA_VISIBLE_DEVICES`, and
fails when there is none. This choice does not reserve the device against
another job. The CPU images hold no GPU build of PyTorch.

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
* The GPU choice above is not a reservation.
* Fiji runs only under Snakemake; under Nextflow, supply BigStitcher's
  `dataset.xml` (`stitching.dataset_xml`).
* The {doc}`MATLAB legacy workflow <matlab-legacy>` keeps its own profile,
  `profile/broad-uger-matlab/`, and its conda environment.

## How it is checked

Focused tests resolve the profile outside the checkout, check the scheduler
arguments the submit script builds, run the job script with a stand-in
Apptainer. They submit no UGER job and
run no GPU, SIF, MATLAB or Fiji job.

[UGER design record](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/broad-uger.md).
