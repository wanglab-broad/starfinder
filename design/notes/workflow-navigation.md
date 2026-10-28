# Snakemake

The Snakemake workflow is `workflow/Snakefile` with its rules in `workflow/rules/`, its
scripts in `workflow/scripts/` and its configuration schema in
`workflow/schemas/config.schema.yaml`. One YAML file configures a run, and the config
scheme describes its keys and blocks. A profile describes how
the jobs are submitted on a cluster; the checkout carries the profile of the Broad UGER
cluster.


Install Snakemake as [Installation](../../docs/get-started/installation.md#snakemake) describes.


# Nextflow

A Nextflow workflow is planned and not implemented: the repository has no Nextflow
pipeline, configuration or profile, and nothing on this site describes one. The
workflow that exists is the Snakemake workflow ([snakemake](../../docs/workflow/snakemake.md)).
