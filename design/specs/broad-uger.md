# Broad UGER

The Python profile `profile/broad-uger/` implements the accepted
[engine record](orchestration-engines.md#the-snakemake-uger-profile).
The frozen MATLAB profile is `profile/broad-uger-matlab/`.
This record does not certify a UGER submission; task group 4 owns that execution gate.

## Environment

The workflow runs in the image of the lock (`uv.lock`) through Apptainer 1.3.2 on
UGER compute nodes; the head of either engine runs as a UGER job. The MATLAB
legacy workflow keeps its conda environment and profile. RHEL 7 CPU nodes cannot host the lock's uv environment because
several wheels require a newer glibc. The Snakemake head runs as a UGER job in the
same image with 8 GiB memory, so its generated job command's absolute Python path
exists inside each job's image. `/stanley` and `/local/scratch` are bound in.

W-648's OCI archives and manifest digests are recorded in
`runs/W-648/20261031T224549Z-23bc2f92/results/manifest.json` under the external
benchmark root. Task group 4 builds the SIFs and records their checksums; a local
profile check establishes neither SIF availability nor its execution on Broad.
The default image is W-648's CPU manifest digest. A task's manifest
`execution.container`, or otherwise `STARFINDER_IMAGE` from the head, selects a
digest; `STARFINDER_SIF` supplies its SIF path, or the default is
`/stanley/WangLab/jiahao/test/starfinder_benchmark/containers/starfinder-<digest hex>.sif`.
The caller keeps the head, manifest digest and SIF consistent.

Apptainer runs with `--cleanenv`; the job script passes the image identity, job ID,
thread counts, selected CUDA device, scratch directory and optional
`STARFINDER_WEIGHTS_DIR` explicitly. `APPTAINER_CACHEDIR` defaults to
`/stanley/WangLab/.apptainer`, and `APPTAINER_TMPDIR` uses UGER's local `TMPDIR`
or `/local/scratch/$JOB_ID`. The interactive shared uv environment is a separate
[RHEL 8 procedure](../notes/broad-shared-environment.md).

## Profile settings

Submit and status commands are bare filenames, `broad-submit.py` and
`broad-status.py`, and `jobscript: broad-jobscript.sh` selects the template.
Snakemake resolves all three relative to the profile directory, including when
invoked outside the checkout with an absolute profile path. `default-resources`
sets `mem_mb=8000` and `runtime=30`; Python rules carry their manifest resources.
The Python profile has no software deployment and no `notemp`; it uses two
retries, keep-going, rerun-incomplete, 1000 jobs, one local core and 60-second
latency wait. `rerun-triggers: [mtime, params, input]` follows the engine record's
fingerprint rule, without `code` or `software-env` triggers.

## Submission, status and logs

The submit script reads Snakemake job properties and the task's manifest, and
submits with `qsub -P broad -S /bin/bash`. It retains `-pe smp <threads>`, linear
binding, per-slot `h_vmem` (total `mem_mb` divided by threads), `h_rt` from runtime
minutes, and reservation above 15 GiB or at four or more threads. GPU jobs alone
add `-l gpu=TRUE,operating_system=RedHat8`; CPU jobs request no OS override.
Each task writes `workflow/logs/<task>.log`; the job's standard output and error
stay the profile's, under `<output root>/workflow/logs/engine/`; UGER's job IDs
distinguish attempts. The status script consults `qstat` then `qacct` and treats absent
accounting as still running. The job script returns Apptainer's exit status.

UGER assigns no GPU device. For a GPU job the job script queries `nvidia-smi`,
chooses the first device with zero used memory, passes `CUDA_VISIBLE_DEVICES`
inside the clean environment and adds `--nv`; if no device is free it fails.
This is a device choice, not an atomic reservation against another job. The
CPU images hold no GPU torch build; task group 4 verifies `--nv` before any GPU
image or actual GPU processing is claimed.

## Frozen MATLAB profile

`profile/broad-uger-matlab/` keeps the prior files with only the three recorded
path repairs: bare submit/status names, the `jobscript` key, and actual default
resources. It retains conda deployment, `notemp`, three retries, the original
submit/status behavior and module loading. The template activates
`/stanley/WangLab/envs/starfinder-v9` before loading MATLAB. Select
`backend: matlab` and `matlab_launcher: broad`.
The environment description moved unchanged to `workflow/envs/matlab-legacy.yaml`;
its historical header still names its original basename.

No MATLAB rule, script, launcher or dialect key changes. The unused conda
exports were retired; Python jobs use the locked image. Scheduler allocation,
SIFs, tools, weights, licenses and full-size resources require the Broad gate.
