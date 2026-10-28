# Broad UGER

The `profile/broad-uger/` profile submits Snakemake jobs to the Broad Institute's
UGER cluster. It includes executor settings, a job script, a submit script and a
status script. Use the {doc}`local dry run <../workflows>` to check the job graph
before submitting.

## Quick start

On a Broad submission host, prepare the environment described by
`config/environment-v9.yaml` and install the package and processing dependencies
there. Edit your configuration for real inputs.

Submission excerpt — these commands require Broad scheduler access and a locally
reviewed job script, so the documentation's local dry run does not execute them:

```bash
snakemake -s workflow/Snakefile --configfile /absolute/config.yaml --cores 1 -n all
snakemake -s workflow/Snakefile --configfile /absolute/config.yaml \
  --profile profile/broad-uger --workflow-profile profile/broad-uger \
  --jobscript profile/broad-uger/broad-jobscript.sh all
```

Run from the repository root. The profile's submit and status paths are relative
to that directory. The job script activates a site-specific environment and loads
UGER, Anaconda3 and MATLAB through Broad's module setup. Review those paths on the
submission host. `matlab_launcher: broad` selects the corresponding MATLAB setup.

## What it does

The cluster-generic executor calls `broad-submit.py` to submit with `qsub` and
`broad-status.py` to query `qstat` and `qacct`. The job script runs the job and
prints UGER usage afterwards. The
[profile source](https://github.com/wanglab-broad/starfinder/tree/dev/profile/broad-uger)
gives the scheduler resource conversion.

## What you get

(submission-status-and-logs)=
### Submission, status and logs

The submit script routes standard output and error to a rule log,
`params.uger_log`, or the user's `snakemake-logs/<rule>/` directory. Processing
outputs use the paths in the {doc}`Snakemake guide <../workflows>`.

## Main options

(profile-settings)=
### Profile settings

Use `--jobs` to choose a submission limit for your account. The checked-in
profile enables retries, keep-going and retention of temporary files. Pass
`--jobscript` explicitly: the profile's `default-resources` entry does not select
a job script template.

(environment)=
## Limits

The environment file describes Snakemake 9 with the cluster-generic executor;
the package's uv environment is separate. Scheduler allocation, modules, models
and MATLAB licenses need checking on the execution host. A local dry run cannot
verify cluster execution or resource sufficiency.

(run)=
## How it is checked

Dry-run the edited configuration locally, then check the scheduled jobs and
resource requests on your submission host. Inspect scheduler logs alongside
STARfinder's processing records.

[UGER profile details](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/broad-uger.md).
