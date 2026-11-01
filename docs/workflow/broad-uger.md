# Broad UGER

The Python workflow uses `profile/broad-uger/` with the cluster-generic executor.
The Snakemake head and its jobs run in the same locked image through Apptainer;
compute nodes need no conda environment. Scheduler submission and real-image
validation belong to the Broad execution gate.

## Quick start

Before submission, build the SIF from the chosen OCI archive on a Broad compute
node and record its checksum. Set `STARFINDER_IMAGE` to the archive's OCI manifest
digest and `STARFINDER_SIF` to its SIF path in the head job's environment.
The default is W-648's CPU image; without a SIF override, jobs look for
`/stanley/WangLab/jiahao/test/starfinder_benchmark/containers/starfinder-<digest hex>.sif`.
The manifest's `execution.container`, when set, selects the image digest.
Keep the head's image and the SIF consistent with that setting.

Run the head as a UGER job with 8 GiB memory, in the same image and with the
checkout and data under `/stanley`. Give Snakemake the absolute Snakefile,
configuration and profile paths when working outside the repository.
The profile resolves its submit, status and job-script filenames relative to
its own directory.

## What it does

The job script runs Apptainer with `--cleanenv` and binds `/stanley,/local/scratch`.
The cache defaults to `/stanley/WangLab/.apptainer`; temporary files use the job's
local scratch. `STARFINDER_WEIGHTS_DIR`, when set, passes through to the container.
The submit script translates total memory to per-slot `h_vmem`, runtime minutes
to `h_rt`, and multiple threads to the SMP parallel environment and CPU binding.
GPU jobs request `gpu=TRUE,operating_system=RedHat8`, select the first GPU with
zero used memory and run Apptainer with `--nv`; they fail if no such device exists.
This selection does not reserve a GPU against concurrent jobs.

## What you get

(submission-status-and-logs)=
### Submission, status and logs

Scheduler stdout and stderr go under `<output root>/workflow/logs/engine/`,
with UGER job IDs in the filenames. Task logs stay under `workflow/logs/`.
The status script queries `qstat` then `qacct`, treating missing accounting as
still running. The Apptainer exit status is the job's exit status.

## Main options

(profile-settings)=
### Profile settings

The profile sets `jobscript: broad-jobscript.sh`, default resources of 8000 MB
and 30 minutes, two retries, keep-going, rerun-incomplete and
`rerun-triggers: [mtime, params, input]`. Temporary outputs can be removed.
Choose `--jobs` for the account's submission limit; the profile defaults to 1000.

(environment)=
## Limits

The frozen MATLAB workflow uses `profile/broad-uger-matlab/` and
`workflow/envs/matlab-legacy.yaml`. Its job script still activates
`/stanley/WangLab/envs/starfinder-v9` and loads MATLAB through Broad's modules.
Use `backend: matlab` and `matlab_launcher: broad` for that workflow.
The Python profile has no MATLAB or conda activation.

(run)=
## How it is checked

Focused tests check profile resolution outside the repository, mocked scheduler
arguments and the job script's execution with a stand-in Apptainer. They submit
no UGER jobs and establish no actual GPU, SIF, MATLAB or Fiji execution coverage.
The Broad gate checks those tools, scheduler access, licenses and resources.

[UGER design record](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/broad-uger.md).
