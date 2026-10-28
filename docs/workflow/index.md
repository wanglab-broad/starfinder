# Workflow

:::{container} sf-lead
Run STARfinder over a dataset with Snakemake: process FOVs, prepare cells, place
FOVs into samples and write sample outputs and reports.
:::

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Snakemake
:link: snakemake
:link-type: doc
:class-card: sf-card
Run a workflow, look up configuration keys and use the Broad UGER profile.
:::

:::{grid-item-card} Nextflow
:link: nextflow
:link-type: doc
:class-card: sf-card
Workflow availability.
:::

::::

```{toctree}
:hidden:
:maxdepth: 2

Snakemake <snakemake>
Nextflow <nextflow>
```

## How the workflow relates to the Python API

Snakemake schedules file-based jobs. The Python backend's processing jobs call
{py:meth}`FOV.run <starfinder.dataset.FOV.run>`; segmentation and assignment run
separately. Sample jobs call the dataset's stitch, export and report entries.
The {doc}`coordination guide <../coordination>` shows direct API use.

Start with the {doc}`Snakemake guide <../workflows>`, then use the
{doc}`configuration reference <../workflow-configuration>` to look up keys.
The {doc}`MATLAB API <../api/matlab>` describes the MATLAB processing backend.
