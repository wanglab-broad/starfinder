# Broad UGER

The profile `profile/broad-uger/` runs the Snakemake workflow on the Broad Institute's
UGER cluster. It holds the profile settings (`config.yaml`), a job script template
(`broad-jobscript.sh`) and the submit and status scripts (`broad-submit.py`,
`broad-status.py`). This page describes them as they are in the checkout, moved here
from {doc}`../workflows` ("Broad UGER execution"); it does not certify submission on
other clusters.

## Environment

The repository's `config/environment-v9.yaml` describes a Snakemake 9 environment
(Python >=3.11,<3.13) including the cluster-generic executor. It is separate from
the Python package's uv environment; install the package/dependencies into the
actual compute environment when using the Python backend. StarDist runs in the
environment that runs Snakemake, with the package's `stardist` extra; no rule reads `envs_path`
({doc}`../get-started/installation`, "Snakemake").

## Run

On a suitably configured Broad host, after editing paths and preparing real data:

```bash
conda env create -f config/environment-v9.yaml
conda activate starfinder-v9
# From the repository root, with a real, fully edited config:
snakemake -s workflow/Snakefile --configfile /absolute/config.yaml --cores 1 -n all
snakemake -s workflow/Snakefile --configfile /absolute/config.yaml \
  --profile profile/broad-uger --workflow-profile profile/broad-uger \
  --jobscript profile/broad-uger/broad-jobscript.sh all
```

The explicit `--jobscript` is intentional: the checked-in profile puts a
`jobscript` string under `default-resources`, which does not select Snakemake's
jobscript template. Review a site-local copy of the template before submission:
it activates `/stanley/WangLab/envs/starfinder-v9` and loads UGER, Anaconda3 and
MATLAB through Broad `useuse`. Set `matlab_launcher: broad` in configurations
submitted this way ([MATLAB launcher](../workflow-configuration.md#matlab-launcher)).
Profile submit/status paths are relative to the
working directory; running from another directory requires absolute paths in
a local profile copy.

## Profile settings

The [profile](https://github.com/wanglab-broad/starfinder/tree/dev/profile/broad-uger)
sets `executor: cluster-generic`, conda deployment, `jobs: 1000`, one local core,
60-second latency wait, three retries, keep-going, rerun-incomplete and retention
of temporary files. Choose a smaller `--jobs` limit appropriate to your account.
For the MATLAB backend, `rsf_preparation` and `create_sample_maf` always run locally.

## Submission, status and logs

`broad-submit.py` reads job properties, submits via `qsub -P broad`, uses an SMP
parallel environment for multithreaded jobs, divides total `mem_mb` by threads
for `h_vmem`, and converts runtime minutes to `h_rt`. It requests reservation
above 15 GiB or at four or more threads. Logs use a rule log, `params.uger_log`,
or `~/snakemake-logs/{rule}/`. `broad-status.py` consults `qstat` then `qacct`;
missing accounting information is treated as still running. After the job, the job script prints
its UGER usage (`qstat -j`), then exits 0 if the `jobfinished` file that the submit
script passes it exists and with the job's exit status otherwise.

Scheduler access,
account/project allocation, tool modules, environment paths, model files and
licenses must be checked on the submission host. A local dry run does not test
them or validate cluster resource sufficiency.
