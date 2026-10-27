# Workflow

:::{container} sf-lead
The workflow runs STARfinder over every FOV and sample of a dataset, from round/FOV
TIFF directories and a codebook to per-FOV molecule tables and cells, and to each
sample's export and processing report.
:::

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Snakemake
:link: snakemake
:link-type: doc
:class-card: sf-card
The workflow, its configuration and the Broad UGER profile.
:::

:::{grid-item-card} Nextflow
:link: nextflow
:link-type: doc
:class-card: sf-card sf-card-planned
Planned; not implemented yet.
:::

::::

```{toctree}
:hidden:
:maxdepth: 2

Snakemake <snakemake>
Nextflow <nextflow>
```

## How the workflow relates to the Python API

The workflow schedules and the Python API processes. A Snakemake rule decides which
FOV, subtile or sample a job covers and which files it reads and writes; inside the job,
the processing is the same Dataset/FOV API that a direct Python caller uses
({doc}`../coordination`, {doc}`../architecture`):

* **Dataset/FOV.** A per-FOV job runs {py:meth}`FOV.run <starfinder.dataset.FOV.run>`
  with its stages, load to filtering, as operations inside the one job, not as separate
  rules. The per-sample jobs call {py:meth}`Dataset.stitch <starfinder.dataset.Dataset.stitch>`,
  {py:meth}`Dataset.export <starfinder.dataset.Dataset.export>` and
  {py:meth}`Dataset.report <starfinder.dataset.Dataset.report>`; the dataset's comparison
  page is {py:meth}`Dataset.compare_samples <starfinder.dataset.Dataset.compare_samples>`.
  The segment and assign entries run as their own jobs after the molecule tables
  ({doc}`../workflow-downstream`).
* **The adapters.** Two layers connect a rule to that API. The public translation,
  {py:func}`~starfinder.dataset.from_workflow_config` with its
  {py:class}`~starfinder.dataset.WorkflowConfig` and the block translators of
  `starfinder.dataset.workflow`, turns the workflow's YAML into a `Dataset` and typed
  configurations, and rejects an unknown or misplaced key when Snakemake parses the rule
  files, before any job starts. The private runners of `starfinder._snakemake` take a
  rule script's `snakemake` object, read its configuration, inputs, outputs and
  wildcards, call the API and write the rule's declared files, including the
  MATLAB-shaped files that later rules read.

So a configuration that the workflow accepts describes calls of the API; the
[configuration reference](../workflow-configuration.md) gives each key and block and how
it is read. `backend: python` selects the Python rules for the
sequencing stages; `backend: matlab`, the default, runs the MATLAB scripts of the
{doc}`../api/matlab` instead, and the segmentation and assignment rules call the package
under both backends. The adapters are the part a planned rewrite of the workflow replaces; the API they call
is kept ([what the rewrite replaces](../workflows.md#what-213-replaces)).
