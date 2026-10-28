# What workflow orchestration replaces

% What §2.13 replaces

Workflow orchestration, a later part of the package's development, replaces the code that
talks to Snakemake and keeps what that code calls. This section records today's state as
input for that work; it proposes no design.

% §2.13

**Replaced.** The private module `starfinder._snakemake`, which only `workflow/` and
the tests import, holds:

- the runners that take a rule script's `snakemake` object: `_run_workflow` for the
  five processing rules and one `_run_*` function for each other Python rule;
- the rule input and output lists that the rule files call (`_stitching_position_inputs`,
  `_reference_image_inputs`, `_export_sample_files`, `_export_preview_outputs`,
  `_reads_assignment_outputs`, `_report_sample_files` and `_report_dataset_files`), which
  plan the files the engine schedules;
- the writers of the MATLAB-shaped files of `reads_assignment` and `stitch_sample`
  (`raw.h5ad`, `reads_assignment.csv`, `log.txt`, the assignment figures and
  `tile_config_<sample>.csv`). Since 2026-10-24 the per-FOV files are written without an
  `assembly` block or with molecules only, and the tile configuration without the block;
  the route of a saved assignment writes the assignment folder alone, and the figures
  only with `assembly.previews`.

% W-568

Workflow orchestration also replaces the input enumeration spread over the rule files (`SAMPLE_TO_FOV`
and the `get_*_input` functions, from `INPUT/fovs.csv` for the Python backend, read by
`_membership`, and `documents/sample-annotation.csv` for the MATLAB backend) and the
start-up validation in `workflow/rules/common.smk`: the backend and workflow-mode
checks and the translation of each enabled Python rule and block when Snakemake
parses the rule files.

% §2.13

**Kept.** The seam is `starfinder.dataset.workflow`: `WorkflowConfig`,
`from_workflow_config`, the block translators (such as `_stitching_block`,
`_export_config` and `_report_block`) and `_rule_checkpoints`
([Configuration reference](../specs/workflow-configuration.md)). Behind it are the single-FOV,
stage and sample operations they configure: `FOV.run` and the `FOV` stage methods,
`Dataset.stitch`, `Dataset.export`, `Dataset.report` and `Dataset.compare_samples`.
The runner and planner signatures do not change before workflow orchestration.

% §2.13

### Logs, time and memory

- **Snakemake** records the wall time and peak RSS of a job through the rule's
  `benchmark:` directive, a tab-separated file under `log/benchmark/<rule>/`. Only
  these rules have one: `rsf_single_fov`, `gr_single_fov_subtile`,
  `deep_create_subtile`, `lrsf_single_fov_subtile`, `deep_rsf_subtile`,
  `stardist_segmentation`, `bigstitcher_engine` and the legacy
  `run_BigStitcher_macro`. No rule has a `log:` directive. On the cluster, a job's
  standard output and error go to the file the UGER profile names
  ([workflow/broad-uger](../specs/broad-uger.md), "Submission, status and logs").
- **STARfinder** records the seconds of each FOV step in `fovs/<fov>/run.json`
  (`steps`, appended by the `_log_step` decorator of the `FOV` methods). The
  processing rules always write it: `_run_workflow` passes `FOV.run` the
  `CheckpointConfig` of `_rule_checkpoints`
  ([Run records of the processing rules](../specs/workflow-configuration.md#run-records-of-the-processing-rules)).
  STARfinder records no memory figure of a FOV.
- **Completion.** `run.json` `status` `succeeded` means that `FOV.run`'s processing
  succeeded; `_run_workflow` then writes the rule's other declared outputs, and each
  writer adds its file to `outputs` (path relative to the FOV folder, SHA-256). A
  declared output missing from `outputs` was not written. Version 2 of the record
  (since 2026-10-24) provides what verified resume and job completion need; that logic,
  with scheduling, invalidation and retry, is left to workflow orchestration.
- At the sample level, the fusion, labels, segmentation and SpatialData parts of
  `Dataset.export`, and `Dataset.report`, record `wall_s`, `cpu_s` and
  `peak_rss_kib` (the process's peak RSS) in `export.json` and `report.json`; the
  assembly part records no time. The report reads the parts' `wall_s` and the
  `run.json` steps into its step table.
- No handler is attached to the `"starfinder"` logger, so its INFO lines (each
  step's start and its seconds) are not printed.
- The Python backend writes no processing log beside `run.json`; human log files
  are left to workflow orchestration. The MATLAB backend writes its console logs under `log/`.

% W-565, §2.13

The two work together in two places today. Each rule script calls a runner of
`starfinder._snakemake` with its `snakemake` object, from which the runner reads the
configuration, inputs, outputs and wildcards. `common.smk` calls
`from_workflow_config` and the block translators when the rule files are parsed, so
a configuration error stops the run before any job starts.
