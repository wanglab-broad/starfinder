# Workflow

The workflow runs STARfinder over every FOV and sample of a dataset: it turns round/FOV
TIFF directories and a codebook into per-FOV molecule tables, then segments cells and
assigns molecules to them, stitches each sample's FOVs, assembles and exports the
sample and writes its processing report. The workflow that exists is the Snakemake
workflow in `workflow/`; a Nextflow workflow is planned.

```{toctree}
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
under both backends. The adapters are the part §2.13 replaces; the API they call is kept
([What §2.13 replaces](../workflows.md#what-213-replaces)).
