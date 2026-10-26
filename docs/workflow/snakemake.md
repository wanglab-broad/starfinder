# Snakemake

The Snakemake workflow is `workflow/Snakefile` with its rules in `workflow/rules/`, its
scripts in `workflow/scripts/` and its configuration schema in
`workflow/schemas/config.schema.yaml`. One YAML file configures a run, and the config
scheme describes its keys and blocks. A profile describes how
the jobs are submitted on a cluster; the checkout carries the profile of the Broad UGER
cluster.

```{toctree}
:maxdepth: 2

Workflows and configuration <../workflows>
Config scheme <../workflow-configuration>
Profile: Broad UGER <broad-uger>
Downstream segmentation and assignment <../workflow-downstream>
```

Install Snakemake as [Installation](../get-started/installation.md#snakemake) describes.
