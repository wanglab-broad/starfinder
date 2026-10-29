# Workflow orchestration baseline: rules, adapters, boundary, configuration and the golden baseline

% Status: Proposed (W-642, 2026-10-29, at 56c35b8)

Status: Proposed. This record is reviewed at the specification gate of workflow orchestration
before any implementation; approval publishes nothing.

This page records what the Snakemake workflow does at revision `56c35b8` (Starfinder `dev`, the
base of the workflow orchestration specification batch), before workflow orchestration replaces the
code that talks to Snakemake. It inventories every rule of both backends, every function of
`starfinder._snakemake` and every parse-time definition of `workflow/rules/common.smk`, the places
where logic that can change a result sits outside the Python API, the Python configuration as it is
read today, the MATLAB legacy surface that stays frozen, and every discrepancy found between the
documents, the schema and the code. It also pins the outputs of the current workflow on a synthetic
fixture as the golden baseline that the refactor must reproduce. Paths are relative to the
repository root, and line numbers are at `56c35b8`.

% W-642 (S1 of §2.13); the batch runner/s213-spec-20261029 starts from dev at 56c35b8 (contains W-620)

What workflow orchestration replaces and keeps is summarized in
[workflow-history](../notes/workflow-history.md) ("What workflow orchestration replaces"); the full
records of today's workflow are [workflows](../specs/workflows.md),
[workflow-configuration](../specs/workflow-configuration.md),
[workflow-downstream](../specs/workflow-downstream.md), [broad-uger](../specs/broad-uger.md),
[checkpoints](../specs/checkpoints.md) and [coordination](../specs/coordination.md). This page
does not edit them; a statement of theirs that the new design would change is listed by the
contract record of workflow orchestration with its old and proposed wording.

The decisions this page refers to were agreed on 2026-10-27 on top of the boundary of 2026-09-28
(the boundary: logic that can change a result belongs in the Python API, scheduling, placement,
order and retries belong to the engine; `.smk` and `.nf` files carry no defaults, parameter
translation or result-affecting computation; the API has no multi-FOV runner):

% W-152 comment "§2.13 planning decisions (2026-10-27)" and the 2026-09-28 scope and boundary comments

| Decision | In short |
| --- | --- |
| D1 | Snakemake and Nextflow both cover the whole Python workflow; MATLAB stays a legacy workflow in Snakemake only. |
| D2 | A package command group, one command per API entry; a plan command validates every block, writes the resolved configuration with its hash and a manifest of every task; the engines read only the manifest and call the CLI. |
| D3 | Fiji only in Snakemake, as the BigStitcher engine task through the CLI; Nextflow runs no Fiji. |
| D4 | The Python dialect gets its own shape: one block per step and an execution section; a present block enables its step; the plan derives the task graph from data dependencies; `workflow_mode`, the `rules.<rule>.run` flags and the MATLAB rule names leave the Python dialect; migration by a table. |
| D5 | Native tiled execution; for the Python backend the legacy subtile route is retired. |
| D6 | Downstream tasks: morphology per FOV, segment per (FOV, run), assign per FOV; the legacy segmentation chain of the Python backend is retired. |
| D7 | The MATLAB workflow is separated and frozen; its adapters move unchanged into one private legacy module; the default backend becomes `python`. |
| D8 | Checkpoints `candidates` and `pre_qc` on by default; verified resume; a task succeeded only when every declared output is recorded; one log per task. |
| D9 | One lock, two forms: `uv.lock` and a CPU image pinned by digest; Snakemake 9 in a `workflow` dependency group; stale environment files retired. |
| D10 | Validation by layer, including this golden baseline of the current workflow. |
| D11 | This specification, then the human gate, then implementation from the drafts. |

Evidence, cited by path and not repeated, kept with the run records outside the repository: the
environment record, the listings this page is checked against (`results/listings/`), the two golden
runs and their comparison (`results/golden/`), and the run's manifest (`results/manifest.json`).

% Run directory: /home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/runs/W-642/20261029T150607Z-e8793b9c
% (results/environment.json, results/listings/, results/golden/run1, results/golden/run2, results/golden/comparison.json,
% results/manifest.json; the worker notes notes.md). The dry runs and the Snakemake environment of W-641:
% runs/W-641/20261028T0256Z-inputs/ (inputs.md, gp099/dryrun/, gp099/starfinder-v9-pip-freeze.txt).

## Where the workflow lives today

| Piece | Where | What it holds |
| --- | --- | --- |
| Entry | `workflow/Snakefile` | schema validation (`:19`), `include` of `common.smk` (`:23`), the Python FOV wildcard constraint (`:27-29`), the backend's rule files (`:32-37`), `segmentation.smk` (`:39`), the stitching rules by block or backend (`:44-47`), `reads-assignment.smk` (`:48`), `export.smk` with an `assembly` block (`:49-50`), `report.smk` with a `report` block (`:52-53`), `utils.smk` for MATLAB (`:54-55`), the `ruleorder` by `workflow_mode` (`:63-74`), the MATLAB local rules (`:79-80`), `rule all` (`:84-86`) |
| Shared definitions | `workflow/rules/common.smk` | backend and block switches, `workflow_mode` presets and flags, resource defaults, paths, FOV membership, the target list `get_overall_output`, the MATLAB `rsf_preparation` rule, the parse-time validation of the Python blocks |
| Python rules | `registration-py.smk`, `spot-finding-py.smk`, `stitching-py.smk`, `export.smk`, `report.smk` | the processing, morphology, stitching, export and report rules of `backend: python` |
| MATLAB rules | `registration.smk`, `spot-finding.smk`, `stitching.smk`, `utils.smk` | the processing, sample stitching and stage-file rules of `backend: matlab` |
| Shared rules | `segmentation.smk`, `reads-assignment.smk` | the legacy segmentation chain and `reads_assignment` of both backends; the MATLAB-only sample aggregation rules |
| MATLAB launcher | `workflow/rules/matlab_launcher.py` | the MATLAB command (`matlab_launcher`, `matlab_single_thread`) |
| Scripts | `workflow/scripts/*.py`, `*.m` | one adapter per Python rule (each calls one runner of `starfinder._snakemake`), the MATLAB entry points, and the legacy Python scripts with their own logic |
| Adapters | `src/python/starfinder/_snakemake.py` | the runners, planners and MATLAB-shaped writers ([classification](#the-snakemake-adapters-and-the-shared-definitions)) |
| Translation | `src/python/starfinder/dataset/workflow.py`, `dataset/_legacy_config.py` | `WorkflowConfig`, `from_workflow_config`, the block translators, `_rule_checkpoints`, the MATLAB-backend translation; kept by workflow orchestration |
| Schema | `workflow/schemas/config.schema.yaml` | the configuration's JSON Schema, with the backend guards (`allOf`, `:346-576`) |
| Profile | `profile/broad-uger/` | the UGER profile ([broad-uger](../specs/broad-uger.md)) |
| Environments | `config/environment-v9.yaml`, `config/environment.yaml`, `config/environment.yml`, `config/stardist.yml`, `workflow/envs/starfinder.yml`, `workflow/envs/stardist.yml` | conda environment files; no rule declares `conda:` |
| Not included | `workflow/rules/gcloud-backup.smk` | a separate backup workflow that the Snakefile does not include ([discrepancies](#discrepancies), O1) |

## Rule inventory

Every rule that the Snakefile defines, per backend, as `snakemake --list-rules` lists it: 19 rules
for `backend: python` (the union of three configurations: the golden configurations A and B and a
configuration with a stitching block of method `bigstitcher`, which defines the other form of
`stitch_sample` and the two BigStitcher rules) and 22 for `backend: matlab` (W-641's configuration
from `tests/tissue_2D_test.yaml`). A rule defined in a shared file is one row with backend `both`;
a rule name that each backend defines in its own file has a row per backend, and `stitch_sample`
has a row for each of its two definitions. Paths in the rule columns are relative to
`OUTPUT = root_output_path/dataset_id/output_id` unless marked INPUT
(`root_input_path/dataset_id/sample_id`). "Undeclared" names a file the rule writes or reads
without declaring it. Every rule has the default of one thread unless stated; `mem_mb` and
`runtime` come from `rules.<section>.resources` through `get_rule_config` (`common.smk:129-153`)
with the defaults 8000 MB and 30 minutes (`DEFAULT_RESOURCES`, `common.smk:156-159`); "runtime ×
attempt" is `make_get_runtime` (`common.smk:187-195`). No rule has a `log:` directive.

% Listings: results/listings/rules-python.txt, rules-matlab.txt and rules-<configuration>.txt (Snakemake 9.15.0 of
% ~/miniforge3/envs/starfinder-v9, outside the project lock, as W-641 recorded it); configurations under
% results/listings/configs/. The check of this table against them: results/listings/record_check.json.

(orchestration-rule-inventory)=

| Rule | Backend | Defined at | Inputs | Outputs | Script or body | Threads and resources | Benchmark | Keys that select and parameterize it |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `all` | both | `Snakefile:84-86` | `get_overall_output` (`common.smk:283-385`) | none | none | none | none | every flag and block (the target list) |
| `rsf_single_fov` | python | `registration-py.smk:16-31` | `codebook=CODEBOOK` (the reference file, `_reference_file`), `images=_image_dir_inputs(config, ROUND)` | `fovs/{fovID}/run.json`, the reference image (`reference_image`; none with `reference_image_import`), `fovs/{fovID}/molecules.csv`; undeclared: `fovs/{fovID}/checkpoints/` | `scripts/rsf_single_fov.py` → `_run_workflow` | 4 threads; `mem_mb`, `runtime` | `log/benchmark/rsf_single_fov/{fovID}.txt` | `workflow_mode: direct` or `rules.rsf_single_fov.run`; `rules.rsf_single_fov.parameters.*`; `n_rounds`, `ref_round`, `seq_channel_order`, `rotate_angle`, `readout_mode`, `codebook`/`direct_panel`, `image_dir_template`, `reference_image`/`reference_image_import`, `maximum_projection`, `voxel_size_*`, `config_path` |
| `gr_single_fov_subtile` | python | `registration-py.smk:35-52` | `codebook`, `images` | `run.json`, the reference image, temp `output/subtile/{fovID}/subtile_coords.csv` and `subtile_data_{n}.npz` for `N_SUBTILE` | `scripts/gr_single_fov_subtile.py` → `_run_workflow` | 4 threads; `mem_mb`, runtime × attempt | `log/benchmark/gr_single_fov_subtile/{fovID}.txt` | `workflow_mode: subtile` or `rules.gr_single_fov_subtile.run`; its `parameters` (`create_subtiles.sqrt_pieces`, default 4 at `common.smk:259-260`); `img_row`, `img_col`; the keys of `rsf_single_fov` |
| `deep_create_subtile` | python | `registration-py.smk:56-72` | `images` | as `gr_single_fov_subtile` | `scripts/deep_create_subtile.py` → `_run_workflow` | 4 threads; `mem_mb`, runtime × attempt | `log/benchmark/deep_create_subtile/{fovID}.txt` | `workflow_mode: deep` or `rules.deep_create_subtile.run`; its `parameters`; `img_row`, `img_col` |
| `nuclei_registration` | python | `registration-py.smk:76-86` | `images=_image_dir_inputs(config, additional_round_names)` | `fovs/{fovID}/nuclei_registration.json`; undeclared: `images/<round>/<channel label>/{fovID}.tif`, and with `parameters.checkpoints` `fovs/{fovID}/other_rounds/` | `scripts/nuclei_registration.py` → `_run_nuclei_registration` | `mem_mb` | none | `rules.nuclei_registration.run` (always available); `parameters.checkpoints`; `additional_round`, `ref_channel`, `reference_stains`, `ref_round`, `rotate_angle`, `maximum_projection` |
| `rotate_nuclei` | python | `registration-py.smk:90-122` | `get_dapi_input`: the one file `INPUT/<ref_round>/<fov>/*<stain>.tif` of the reference stain (`:90-101`) | `images/DAPI/{fovID}.tif` | inline `run:` (`:110-122`) | `mem_mb` | none | `rules.rotate_nuclei.run` (always available); `reference_stains`, `ref_channel`, `ref_round`, `image_dir_template`, `rotate_angle`, `maximum_projection` |
| `lrsf_single_fov_subtile` | python | `spot-finding-py.smk:14-28` | `codebook`, `subtile=output/subtile/{fovID}/subtile_data_{n_subtile}.npz` | `fovs/{fovID}/subtile_{n_subtile}/run.json`, temp `output/subtile/{fovID}/subtile_goodSpots_{n_subtile}.csv` | `scripts/lrsf_single_fov_subtile.py` → `_run_workflow` | 4 threads; `mem_mb`, runtime × attempt | `log/benchmark/lrsf_single_fov_subtile/{fovID}_{n_subtile}.txt` | `workflow_mode: subtile` or its flag; its `parameters` |
| `deep_rsf_subtile` | python | `spot-finding-py.smk:32-46` | as `lrsf_single_fov_subtile` | as `lrsf_single_fov_subtile` | `scripts/deep_rsf_subtile.py` → `_run_workflow` | 4 threads; `mem_mb`, runtime × attempt | `log/benchmark/deep_rsf_subtile/{fovID}_{n_subtile}.txt` | `workflow_mode: deep` or its flag; its `parameters` |
| `stitch_subtile` | python | `spot-finding-py.smk:50-62` | `subtile_coords.csv`, every `subtile_goodSpots_{n}.csv`, `reference=REFERENCE_IMAGE_INPUT` | `signal/{fovID}_goodSpots.csv`, `signal/{fovID}_goodSpots.png` | `scripts/stitch_subtile.py` (its own logic) | `mem_mb`, `runtime` | none | `workflow_mode: subtile`/`deep` or its flag; the `ruleorder` (`Snakefile:63-74`) |
| `bigstitcher_macro_sample` | python | `stitching-py.smk:37-48` | `MEMBERSHIP`, the sample's reference images (`get_stitching_images`), the position file (`get_stitching_positions`) | `<bigstitcher_save_dir>/{sample}/BigStitcher_macro.ijm`, `view_map.csv`; undeclared: `tiles/` | `scripts/bigstitcher_macro_sample.py` → `_run_bigstitcher_macro` | `mem_mb` of `rules.create_BigStitcher_macro` | none | a `stitching` block with method `bigstitcher` (the default, `stitching-py.smk:12`): `input`, `positions`, `config`, `bigstitcher_save_dir` |
| `bigstitcher_engine` | python | `stitching-py.smk:54-67` | the macro, `view_map.csv` | `<bigstitcher_save_dir>/{sample}/dataset.xml`; undeclared: BigStitcher's other files | `scripts/bigstitcher_engine.py` → `_run_bigstitcher_engine` (Fiji) | 4 threads; `mem_mb`, `runtime` of `rules.run_BigStitcher_macro` | `log/benchmark/bigstitcher_engine/{sample}.txt` | method `bigstitcher`; `fiji_path` (read when the rule runs) |
| `stitch_sample` | python | `stitching-py.smk:71-85` (method `bigstitcher`) | `MEMBERSHIP`, `dataset.xml`, `view_map.csv`, the macro, the reference images, the position file | `stitching/{sample}/placement.json`; without an `assembly` block `output/tile_config_{sample}.csv`; undeclared: `stitching/{sample}/fovs.csv`, `links.csv` | `scripts/stitch_sample.py` → `_run_stitching` | `mem_mb` of `rules.create_tile_config` | none | method `bigstitcher`; `review`, `fallback`; `img_row`, `img_col` without an `assembly` block |
| `stitch_sample` | python | `stitching-py.smk:91-101` (an in-process method) | `MEMBERSHIP`, the reference images, the position file | `stitching/{sample}/placement.json`; undeclared: `fovs.csv`, `links.csv` | `scripts/stitch_sample.py` → `_run_stitching` (`Dataset.stitch`) | `mem_mb` of `rules.stitching_preparation` | none | method `multiview_stitcher`: `config`, `input`, `positions`, `review`, `fallback` |
| `export_sample` | python | `export.smk:29-38` | `get_export_input` (`_export_sample_files`) | `export/{sample}/export.json`; with `assembly.previews` the folder `export/{sample}/previews/`; undeclared: every other file of `export/{sample}/` (listed in `export.json`) | `scripts/export_sample.py` → `_run_export` | `mem_mb` of `rules.create_sample_h5ad` | none | an `assembly` block (and a `stitching` block); the `export` block |
| `report_sample` | python | `report.smk:24-32` | `get_report_input` (`_report_sample_files`: `export.json`, else `placement.json`, else the FOVs' `run.json`) | `report/{sample}/report.json`; undeclared: the other files of `report/{sample}/` | `scripts/report_sample.py` → `_run_report_sample` | `mem_mb` 8000, fixed | none | a `report` block |
| `report_dataset` | python | `report.smk:36-44` | every sample's `report/{sample}/report.json` | `report/samples.csv`; undeclared: `report/index.html` | `scripts/report_dataset.py` → `_run_report_dataset` | `mem_mb` 8000, fixed | none | a `report` block |
| `enhance_dapi_with_flamingo` | both | `segmentation.smk:9-18` | `dapi_img`, `flamingo_img` = `SHARED_IMAGE_INPUTS` (Python: input functions naming `images/flamingo/<label>/{fovID}.tif` of the `flamingo` round's channels named `DAPI` and `Flamingo`, `_shared_image_inputs`; MATLAB: `images/flamingo/DAPI/`, `images/flamingo/Flamingo/`, `common.smk:408-410`) | `images/flamingo/enhanced_DAPI/{fovID}.tif` | `scripts/enhance_dapi_with_flamingo.py` → `_run_enhance_dapi_with_flamingo` | `mem_mb` | none | `rules.enhance_dapi_with_flamingo.run` (free mode only); `additional_round` |
| `stardist_segmentation` | both | `segmentation.smk:22-34` | `images/<segmentation_input_folder>/{fovID}.tif` (`SEG_INPUT_DIR`, default `overlay`, `common.smk:203`) | `images/stardist_segmentation/{fovID}.tif`; undeclared: `{fovID}.json` beside it | `scripts/stardist_segmentation.py` → `_run_stardist_segmentation` | 2 threads; `mem_mb`, `runtime` | `log/benchmark/stardist_segmentation/{fovID}.txt` | `rules.stardist_segmentation.run` and `parameters`; `voxel_size_*`, `maximum_projection`, `rules.create_nuclei_amplicon_overlay.parameters.maximum_projection` |
| `create_nuclei_amplicon_overlay` | both | `segmentation.smk:38-47` | `images/DAPI/{fovID}.tif`, `amplicon_img` (Python: the reference image; MATLAB: `images/ref_merged/{fovID}.tif`) | `images/overlay/{fovID}.tif` | `scripts/create_nuclei_amplicon_overlay.py` → `_run_nuclei_amplicon_overlay` | `mem_mb` | none | `rules.create_nuclei_amplicon_overlay.run`, `parameters.maximum_projection` |
| `reads_assignment` | both | `reads-assignment.smk:21-48` | `MEMBERSHIP`, `images/DAPI/{fovID}.tif`, `images/stardist_segmentation/{fovID}.tif`, `MOLECULE_TABLE`, `CODEBOOK`, and either every sample's `output/tile_config_{sample}.csv` or, with an `assembly` block, the FOV's reference image | without an `assembly` block `expr/{fovID}/raw.h5ad`, `reads_assignment.csv` (undeclared: `assignment.png`, `assignment_single_layer.png`, `log.txt`); with an assignment name `fovs/{fovID}/assignment/<name>/assignment.json` (and `fovs/{fovID}/previews/` with `previews`); with `assignment: null` the `expr/` files | `scripts/reads_assignment.py` → `_run_reads_assignment` | `mem_mb` | none | `rules.reads_assignment.run` and `parameters`; `rules.stardist_segmentation.parameters` (`target`, `segmentation_input_folder`, `expand_labels`); `img_z`, `voxel_size_*`; the `assembly` block; the codebook keys |
| `rsf_preparation` | matlab | `common.smk:391-398` | `config_path` | the configuration's `.json` copy | inline `run:` (`yaml_to_json`) | local rule (`Snakefile:79-80`) | none | dependency only |
| `rsf_single_fov` | matlab | `registration.smk:8-29` | the JSON copy, `INPUT/genes.csv`, `INPUT/<round>/{fovID}` | `log/{fovID}_rsf.txt`, `log/sf_scores/{fovID}.txt`, `images/ref_merged/{fovID}.tif`, `signal/{fovID}_goodSpots.csv` | `run_matlab_scripts` → `rsf_single_fov.m` | 4 threads; `mem_mb`, `runtime` | `log/benchmark/rsf_single_fov/{fovID}.txt` | `workflow_mode: direct` or the flag; `rules.rsf_single_fov.parameters.*` (MATLAB keys) |
| `gr_single_fov_subtile` | matlab | `registration.smk:33-56` | as `rsf_single_fov` | `log/{fovID}_gr.txt`, `images/ref_merged/{fovID}.tif`, temp `subtile_coords.csv`, `subtile_data_{n}.mat` | `run_matlab_scripts` → `gr_single_fov_subtile.m` | 4 threads; `mem_mb`, runtime × attempt; `params.uger_log` | `log/benchmark/gr_single_fov_subtile/{fovID}.txt` | `workflow_mode: subtile` or the flag; its `parameters` |
| `nuclei_registration` | matlab | `registration.smk:60-76` | the JSON copy, `INPUT/<additional round>/{fovID}` | `log/{fovID}_nr.txt`, `log/gr_shifts/{fovID}_nr.txt`; undeclared: `images/<round>/<channel name>/{fovID}.tif` | `run_matlab_scripts` → `nuclei_registration.m` | `mem_mb` | none | the flag; `additional_round`, `dapi_round`, `ref_channel`, `ref_round`, `rotate_angle`, `maximum_projection` |
| `rotate_nuclei` | matlab | `registration.smk:80-106` | `INPUT/<dapi_round>/{fovID}/*ch04.tif` | `images/DAPI/{fovID}.tif` | inline `run:` (`:94-106`) | `mem_mb` | none | the flag; `dapi_round`, `rotate_angle`, `maximum_projection` |
| `lrsf_single_fov_subtile` | matlab | `spot-finding.smk:8-27` | the JSON copy, `INPUT/genes.csv`, `subtile_data_{n_subtile}.mat` | `log/sf_scores/{fovID}_{n_subtile}.txt`, temp `subtile_goodSpots_{n_subtile}.csv` | `run_matlab_scripts` → `lrsf_single_fov_subtile.m` | 4 threads; `mem_mb`, runtime × attempt | `log/benchmark/lrsf_single_fov_subtile/{fovID}_{n_subtile}.txt` | `workflow_mode: subtile` or the flag; its `parameters` |
| `deep_create_subtile` | matlab | `spot-finding.smk:31-50` | the JSON copy, `INPUT/<round>/{fovID}` | `images/ref_merged/{fovID}.tif`, temp `subtile_coords.csv`, `subtile_data_{n}.mat` | `run_matlab_scripts` → `deep_create_subtile.m` | 4 threads; `mem_mb`, runtime × attempt | `log/benchmark/deep_create_subtile/{fovID}.txt` | `workflow_mode: deep` or the flag; its `parameters` |
| `deep_rsf_subtile` | matlab | `spot-finding.smk:54-73` | as `lrsf_single_fov_subtile` | as `lrsf_single_fov_subtile` | `run_matlab_scripts` → `deep_rsf_subtile.m` | 4 threads; `mem_mb`, runtime × attempt | `log/benchmark/deep_rsf_subtile/{fovID}_{n_subtile}.txt` | `workflow_mode: deep` or the flag; its `parameters` |
| `rsf_single_fov_seq` | matlab | `spot-finding.smk:77-92` | the JSON copy, `INPUT/<round>/{fovID}` | `signal/{fovID}_allSpots.csv` | `run_matlab_scripts` → `rsf_single_fov_seq.m` | 2 threads; `mem_mb`, `runtime` | none | the flag (free mode only); its `parameters` |
| `stitch_subtile` | matlab | `spot-finding.smk:96-107` | `subtile_coords.csv`, every `subtile_goodSpots_{n}.csv`; undeclared: `images/ref_merged/{fovID}.tif` | `signal/{fovID}_goodSpots.csv`, `.png` | `scripts/stitch_subtile.py` | `mem_mb`, `runtime` | none | `workflow_mode: subtile`/`deep` or the flag |
| `stitching_preparation` | matlab | `stitching.smk:13-25` | `documents/sample-annotation.csv`, `documents/maf/{sample}.maf`, `images/DAPI/{fovID}.tif` of the sample | `images/fused/{sample}/blank.tif`, `grid.csv`, `grid.png`; undeclared: `images/fused/{sample}/DAPI/tile_{j}.tif` | `scripts/stitching_preparation.py` | `mem_mb` | none | the flag; `img_col`, `img_row` |
| `create_BigStitcher_macro` | matlab | `stitching.smk:29-37` | `images/fused/{sample}/grid.csv` | `images/fused/{sample}/BigStitcher_macro.ijm` | `scripts/create_BigStitcher_macro_1.py` | `mem_mb` | none | the flag; `voxel_size_xy`, `voxel_size_z` |
| `run_BigStitcher_macro` | matlab | `stitching.smk:41-56` | the macro, `grid.csv` | `images/fused/{sample}/DAPI/dataset.xml` | `shell:` with `fiji_path` read when the file is parsed | 4 threads; `mem_mb`, `runtime` | `log/benchmark/run_BigStitcher_macro/{sample}.txt` | the flag; `fiji_path` |
| `create_tile_config` | matlab | `stitching.smk:60-70` | every sample's `dataset.xml` and `grid.csv` | `output/tile_config_{sample}.csv`, `.html` | `scripts/create_tile_config.py` | `mem_mb` | none | the flag; `img_col`, `img_row` |
| `create_sample_h5ad` | matlab | `reads-assignment.smk:54-66` | `documents/sample-annotation.csv`, the sample's `expr/{fovID}/raw.h5ad` | `expr/{sample}_raw.h5ad` | `scripts/create_sample_h5ad.py` | `mem_mb` | none | the flag; `fov_id_pattern` |
| `create_sample_reads_assignment` | matlab | `reads-assignment.smk:70-82` | the annotation, the sample's `expr/{fovID}/reads_assignment.csv` | `expr/{sample}_reads_assignment.csv` | `scripts/create_sample_reads_assignment.py` | `mem_mb` | none | the flag; `fov_id_pattern` |
| `create_sample_maf` | matlab | `utils.smk:9-19` | `documents/sample-annotation.csv`, `documents/raw.maf` | `documents/maf/{sample}.maf` | `scripts/create_sample_maf.py` | `mem_mb` 8000, `runtime` 2, fixed; local rule | none | dependency only |

`workflow/rules/gcloud-backup.smk` defines two more rules in a file that the Snakefile does not
include: `all` (`:60-61`, no input) and `upload_to_gcs` (`:63-74`, whose input and output are the
same files). It was not parsed for the listing: its module level opens a Google Cloud Storage client
and lists a bucket (`:18-22`).

% results/listings/rules-gcloud-backup.source.txt (read from the source)

## The Snakemake adapters and the shared definitions

Every top-level function of `src/python/starfinder/_snakemake.py` and every parse-time definition
of `workflow/rules/common.smk`, each classified once:

* **runner**: takes a rule's `snakemake` object, or is a helper only a runner calls;
* **planner**: declares files, targets, resources or rule selections;
* **writer**: writes a MATLAB-shaped file (the per-FOV assignment files, the tile configuration,
  the figures and log beside them, the JSON copy of the configuration);
* **validation**: refuses a configuration;
* **input enumeration**: lists the FOVs, samples, rounds or input files.

A parse-time definition of `common.smk` is a statement at parse level (at module level, or inside
`if`/`elif`/`else` blocks only) that defines a function or a rule, binds a name, calls a function
or raises, and each `for` loop at parse level (one entry); its imports are listed below the table.
Callers are the files that name the definition at `56c35b8` (rule files, scripts, the package's
other functions), tests apart. The replacement names the decision that replaces it: D2 (the plan,
the manifest and the CLI), D3 (the BigStitcher engine task), D5 (tiled execution), D6 (the
downstream tasks) or D7 (the frozen MATLAB legacy and its private legacy module).

% Checked against results/listings/snakemake_defs.txt (`grep -n "^def "`), common_smk_definitions.tsv and callers.json.

(orchestration-classification)=

| Definition | Where | Class | Callers | Replacement |
| --- | --- | --- | --- | --- |
| `_run_workflow` | `_snakemake.py:28` | runner | `scripts/rsf_single_fov.py`, `gr_single_fov_subtile.py`, `lrsf_single_fov_subtile.py`, `deep_create_subtile.py`, `deep_rsf_subtile.py` | D2: the processing command of a FOV (`FOV.run`, `save_reference_image`, `export_spots`); D5: the tiled tasks; its resident-subtile branch is retired (D5) |
| `_run_nuclei_registration` | `_snakemake.py:73` | runner | `scripts/nuclei_registration.py` | D6: the morphology command (`FOV.prepare_morphology`), with declared outputs |
| `_read_stack` | `_snakemake.py:141` | runner | `_run_nuclei_amplicon_overlay`, `_run_enhance_dapi_with_flamingo` | D6: retired with the legacy segmentation chain for Python; D7: the legacy module |
| `_run_nuclei_amplicon_overlay` | `_snakemake.py:148` | runner | `scripts/create_nuclei_amplicon_overlay.py` | D6: retired for Python (the segmentation run's input `prepare: composite_nuclei_amplicon`); D7: moves unchanged to the legacy module for MATLAB's shared rule |
| `_run_enhance_dapi_with_flamingo` | `_snakemake.py:168` | runner | `scripts/enhance_dapi_with_flamingo.py` | D6: retired for Python (`prepare: enhance_with_flamingo`); D7: the legacy module |
| `_run_stardist_segmentation` | `_snakemake.py:178` | runner | `scripts/stardist_segmentation.py` | D6: the segment command per (FOV, run) (`FOV.segment`); D7: the legacy module |
| `_fov_sample` | `_snakemake.py:227` | input enumeration | `_run_reads_assignment` | D2: the plan's membership (`_membership`); D7: the annotation branch to the legacy module |
| `_tile_record` | `_snakemake.py:254` | writer | `_run_reads_assignment` | D6: retired with the legacy route of `reads_assignment`; D7: the legacy module |
| `_legacy_cells` | `_snakemake.py:264` | runner | `_run_reads_assignment` | D6: retired; D7: the legacy module |
| `_assembly_cells` | `_snakemake.py:287` | runner | `_run_reads_assignment` | D6: the assign command (`FOV.assign_molecules` on the reference grid) |
| `_in_box` | `_snakemake.py:318` | writer | `_write_raw_h5ad`, `_write_reads_assignment` | D6: retired; D7: the legacy module |
| `_cell_obs` | `_snakemake.py:324` | writer | `_write_raw_h5ad` | D6: retired; D7: the legacy module |
| `_write_raw_h5ad` | `_snakemake.py:359` | writer | `_run_reads_assignment` | D6: retired (MATLAB-shaped per-FOV files); D7: the legacy module |
| `_write_reads_assignment` | `_snakemake.py:390` | writer | `_run_reads_assignment` | D6: retired; D7: the legacy module |
| `_assignment_ratio` | `_snakemake.py:447` | writer | `_run_reads_assignment` | D6: retired; D7: the legacy module |
| `_write_assignment_figures` | `_snakemake.py:455` | writer | `_run_reads_assignment`, `_write_ownership_figures` | D6: the assign command's previews (open point P5); D7: the legacy module |
| `_run_reads_assignment` | `_snakemake.py:483` | runner | `scripts/reads_assignment.py` | D6: the assign command (`FOV.assign_molecules`); D7: its legacy route to the legacy module |
| `_reads_assignment_outputs` | `_snakemake.py:560` | planner | `common.smk:370-375`, `reads-assignment.smk:22-24` | D2: the manifest's declared outputs of the assign task |
| `_stitching_position_inputs` | `_snakemake.py:580` | planner | `stitching-py.smk:10,24` | D2: the manifest's declared inputs of the stitch task |
| `_image_dir_inputs` | `_snakemake.py:597` | planner | `registration-py.smk:19,38,58,80` | D2: the plan's input discovery |
| `_reference_image_inputs` | `_snakemake.py:609` | planner | `common.smk:438`, `reads-assignment.smk:23`, `stitching-py.smk:19`, `_export_sample_files`, `_shared_image_inputs` | D2: the manifest |
| `_reference_image_outputs` | `_snakemake.py:625` | planner | `common.smk:289,309,322,437` | D2: the manifest |
| `_run_record_outputs` | `_snakemake.py:637` | planner | `common.smk:289,308,321,331,340,435,436` | D2: the manifest |
| `_assignment_record_outputs` | `_snakemake.py:648` | planner | `_reads_assignment_outputs`, `_export_sample_files` | D2: the manifest |
| `_shared_image_inputs` | `_snakemake.py:654` | planner | `common.smk:439` | D6: retired with `enhance_dapi_with_flamingo` and the overlay rule for Python |
| `_bigstitcher_folder` | `_snakemake.py:679` | planner | `_run_bigstitcher_macro` | D3: the BigStitcher engine task's folder in the plan |
| `_write_tile_config` | `_snakemake.py:686` | writer | `_run_stitching` | D6: the tile-configuration writer is retired |
| `_read_view_map` | `_snakemake.py:702` | runner | `_run_stitching`, `_run_bigstitcher_engine` | D3: the import command |
| `_run_stitching` | `_snakemake.py:707` | runner | `scripts/stitch_sample.py` | D2: the stitch command (`Dataset.stitch`); D3: the import of `dataset.xml` |
| `_run_bigstitcher_macro` | `_snakemake.py:756` | runner | `scripts/bigstitcher_macro_sample.py` | D3: the macro command (Python writes the macro and the tiles) |
| `_check_bigstitcher_engine` | `_snakemake.py:781` | validation | `_run_bigstitcher_engine` | D3: the import command checks `dataset.xml` |
| `_run_bigstitcher_engine` | `_snakemake.py:814` | runner | `scripts/bigstitcher_engine.py` | D3: the engine task runs `fiji_path`; the package spawns no Fiji |
| `_export_sample_files` | `_snakemake.py:834` | planner | `export.smk:14,21,33` | D2: the manifest |
| `_export_preview_outputs` | `_snakemake.py:873` | planner | `export.smk:14,34`, `_run_export` | D2: the manifest |
| `_run_export` | `_snakemake.py:888` | runner | `scripts/export_sample.py` | D2: the export command (`Dataset.export`) |
| `_write_ownership_figures` | `_snakemake.py:931` | writer | `_run_export` | D2: the export command's previews (open point P5) |
| `_report_sample_files` | `_snakemake.py:959` | planner | `report.smk:9,19,28` | D2: the manifest |
| `_report_dataset_files` | `_snakemake.py:979` | planner | `report.smk:9,38,40` | D2: the manifest |
| `_run_report_sample` | `_snakemake.py:986` | runner | `scripts/report_sample.py` | D2: the report command (`Dataset.report`) |
| `_run_report_dataset` | `_snakemake.py:1001` | runner | `scripts/report_dataset.py` | D2: the comparison command (`Dataset.compare_samples`) |
| `sys.path.insert` (the rules folder) | `common.smk:16` | runner | `import matlab_launcher` (`:17`) | D7: the MATLAB launcher stays with the legacy workflow |
| `BACKEND` | `common.smk:21` | planner | `Snakefile:27-79`, `common.smk`, `reads-assignment.smk:50` | D7: default `python`; the Snakefile keeps the switch to the frozen MATLAB rules |
| `raise` (unknown backend) | `common.smk:23` | validation | parse | D2: the plan refuses it (Python); D7: kept for the legacy switch |
| `STITCHING_BLOCK` | `common.smk:27` | planner | `Snakefile:44`, `common.smk:290,298,440` | D4: a present block enables its step, derived by the plan (D2) |
| `ASSEMBLY_BLOCK` | `common.smk:28` | planner | `Snakefile:49`, `common.smk`, `reads-assignment.smk:21`, `stitching-py.smk:26,33` | D4, D2 |
| `LEGACY_STITCHING_RULES` | `common.smk:29` | planner | `common.smk:298` | D4: MATLAB rule names leave the Python dialect; D7 |
| `LEGACY_SAMPLE_RULES` | `common.smk:31` | planner | `common.smk:300` | D4; D7 |
| `REPORT_BLOCK` | `common.smk:33` | planner | `Snakefile:52`, `common.smk:294,446` | D4, D2 |
| `WORKFLOW_MODE` | `common.smk:38` | planner | `Snakefile:63-69`, `common.smk:64,123,127` | D4: `workflow_mode` leaves the Python dialect; D5; D7 keeps it for MATLAB |
| `WORKFLOW_PRESETS` | `common.smk:41` | planner | `common.smk:70,77,127` | D4, D5; D7 |
| `validate_workflow_mode_dependencies` | `common.smk:57` | validation | `common.smk:92` | D2 (the plan); D7 |
| `validate_workflow_mode_dependencies()` | `common.smk:92` | validation | parse | D2; D7 |
| `ALWAYS_AVAILABLE_RULES` | `common.smk:96` | planner | `common.smk:116,119` | D4; D7 |
| `is_rule_enabled` | `common.smk:111` | planner | `Snakefile:63-69`, `common.smk:259,261,302,420`, `stitching-py.smk:26` | D4: the plan derives the graph from data dependencies; D7 |
| `get_rule_config` | `common.smk:129` | planner | every rule file's `resources`, `common.smk:194,203,260,262` | D4: the execution section; D2: the manifest carries each task's resources |
| `DEFAULT_RESOURCES` | `common.smk:156` | planner | every rule file's `resources` | D4: the execution section's defaults, in Python |
| `yaml_to_json` | `common.smk:163` | writer | `rsf_preparation` (`:398`) | D7 |
| `run_matlab_scripts` | `common.smk:175` | runner | `registration.smk:29,56,76`, `spot-finding.smk:27,50,73,92` | D7 |
| `get_additional_round_names` | `common.smk:184` | input enumeration | `registration-py.smk:76`, `registration.smk:60` | D2: the plan's rounds; D7 |
| `make_get_runtime` | `common.smk:187` | planner | the subtile rules' `runtime` | D8: retries and their resources are the engine's; D4: the base runtime |
| `INPUT_DIR` | `common.smk:199` | input enumeration | `common.smk:215,226,236`, `registration-py.smk:97`, `registration.smk`, `spot-finding.smk` | D2: `_layout` and the plan's input discovery |
| `OUTPUT_DIR` | `common.smk:200` | planner | every rule file | D2: `_layout` in the plan |
| `DAPI_DIR` | `common.smk:201` | planner | `rotate_nuclei`, `reads_assignment`, `stitching.smk:9` | D6: retired with `rotate_nuclei` for Python; D7 |
| `FUSE_DIR` | `common.smk:202` | planner | none | no reader ([discrepancies](#discrepancies), O20) |
| `SEG_INPUT_DIR` | `common.smk:203` | planner | `segmentation.smk:24` | D6: retired with `stardist_segmentation`; D7 |
| `DOC_DIR` | `common.smk:204` | planner | `utils.smk:11-14` | D7 |
| `ROUND` | `common.smk:206` | input enumeration | `registration-py.smk:19,38,58`, `registration.smk`, `spot-finding.smk` | D2 |
| `sys.path.insert` (the package) | `common.smk:212` | runner | the imports of `:213-214,414-433` | D2: the engines import nothing from the package |
| `MEMBERSHIP` | `common.smk:215` | input enumeration | `reads-assignment.smk:36`, `stitching-py.smk:39,73,93`, `common.smk:442` | D2: the plan reads `fovs.csv` |
| `_MEMBERS` | `common.smk:216` | input enumeration | `common.smk:217,220` | D2 |
| `SAMPLE_TO_FOV` | `common.smk:217` | input enumeration | `export.smk:21`, `report.smk:19`, `stitching-py.smk:19` | D2 |
| `FOV_TO_SAMPLE` | `common.smk:218` | input enumeration | none at parse in the Python branch | D2 |
| `SAMPLE` | `common.smk:219` | input enumeration | `common.smk:291-295`, `report.smk:12` | D2 |
| `FOVS` | `common.smk:220` | input enumeration | `common.smk:224,308-379` | D2 |
| `FOV_CONSTRAINT` | `common.smk:224` | input enumeration | `Snakefile:29` | D2: the manifest's task identities |
| `CODEBOOK` | `common.smk:227` | input enumeration | `registration-py.smk:18,37`, `spot-finding-py.smk:16,34`, `reads-assignment.smk:40` | D2 |
| `_PYTHON_INPUT_KEYS` | `common.smk:231` | validation | `common.smk:233-234` | D7; open choice OC5 |
| `raise` (Python-only input keys) | `common.smk:234` | validation | parse | D7; open choice OC5 |
| `CODEBOOK` | `common.smk:236` | input enumeration | `reads-assignment.smk:40` (MATLAB) | D7 |
| `MEMBERSHIP` | `common.smk:237` | input enumeration | `common.smk:238`, `reads-assignment.smk:36` | D7 |
| `SAMPLE_ANNOT` | `common.smk:238` | input enumeration | `common.smk:241-245` | D7 |
| `SAMPLE_TO_FOV` | `common.smk:239` | input enumeration | `reads-assignment.smk:55,71`, `stitching.smk:9` | D7 |
| `FOV_TO_SAMPLE` | `common.smk:240` | input enumeration | `common.smk:250,275` | D7 |
| `for current_sample` | `common.smk:241` | input enumeration | parse | D7 |
| `SAMPLE` | `common.smk:254` | input enumeration | `common.smk:291-383`, `reads-assignment.smk`, `stitching.smk`, `utils.smk` | D7 |
| `FOVS` | `common.smk:255` | input enumeration | `common.smk:308-379` | D7 |
| `N_SUBTILE` | `common.smk:260` | planner | `registration-py.smk:44,64`, `spot-finding-py.smk:53`, `registration.smk:42`, `spot-finding.smk:38,99` | D5: the tiles are tasks of the manifest; D7 |
| `N_SUBTILE` | `common.smk:262` | planner | as above | D5; D7 |
| `N_SUBTILE` | `common.smk:264` | planner | as above | D5; D7 |
| `config.setdefault` (`subset_range`) | `common.smk:271` | validation | parse | D7 |
| `config.setdefault` (`subset_random`) | `common.smk:272` | validation | parse | D7 |
| `FOVS` (`subset_list`) | `common.smk:274` | input enumeration | `common.smk:308-379` | D7 |
| `SAMPLE` (`subset_list`) | `common.smk:275` | input enumeration | `common.smk:291-383` | D7 |
| `FOVS` (`subset_range`) | `common.smk:277` | input enumeration | `common.smk:308-379` | D7 |
| `FOVS` (`subset_random`) | `common.smk:279` | input enumeration | `common.smk:308-379` | D7 (unseeded, O3) |
| `get_overall_output` | `common.smk:283` | planner | `Snakefile:86` | D2: the manifest lists every task and its outputs |
| `rsf_preparation` | `common.smk:392` | writer | `Snakefile:80`, the MATLAB rules' JSON input | D7 |
| `MOLECULE_TABLE` | `common.smk:404` | planner | `reads-assignment.smk:39` (MATLAB) | D7 |
| `SHARED_IMAGE_INPUTS` | `common.smk:408` | planner | `segmentation.smk:11,12,41` (MATLAB) | D7 |
| `_check_python_dialect(config)` | `common.smk:417` | validation | parse | D2: the plan validates every block |
| `for python_rule` | `common.smk:418` | validation | parse (`from_workflow_config` of each enabled processing rule) | D2 |
| `_check_sample_rules(config)` | `common.smk:427` | validation | parse | D2; D4 (the MATLAB rule names leave the Python dialect) |
| `MOLECULE_TABLE` | `common.smk:428` | planner | `reads-assignment.smk:39` | D2; D5 retires the legacy subtile producer |
| `RUN_RECORD` | `common.smk:435` | planner | `registration-py.smk:21,40,60` | D2 |
| `SUBTILE_RUN_RECORD` | `common.smk:436` | planner | `spot-finding-py.smk:19,37` | D5: retired with the resident subtile rules |
| `REFERENCE_IMAGE_OUTPUT` | `common.smk:437` | planner | `registration-py.smk:22,41,61` | D2 |
| `REFERENCE_IMAGE_INPUT` | `common.smk:438` | planner | `spot-finding-py.smk:54` | D2; D5 |
| `SHARED_IMAGE_INPUTS` | `common.smk:439` | planner | `segmentation.smk:11,12,41` | D6: retired for Python |
| `_stitching_block(config)` | `common.smk:441` | validation | parse | D2 |
| `_stage_position_ids(...)` | `common.smk:442` | validation | parse | D2 |
| `_export_config(config)` | `common.smk:444` | validation | parse | D2 |
| `_check_export_route(config)` | `common.smk:445` | validation | parse | D2 |
| `_report_block(config)` | `common.smk:447` | validation | parse | D2 |

The imports of `common.smk` at parse time (not definitions): the standard library and pandas
(`:8-13`), `matlab_launcher` (`:17`), and for the Python backend `starfinder._layout`,
`_membership`, `_reference_file` (`:213-214`), `re` (`:223`), `from_workflow_config`,
`_check_python_dialect` (`:414-415`), the block translators (`:424-425`) and the planners of
`starfinder._snakemake` (`:433-434`; `get_overall_output` imports two more at `:289` and `:373`).
The rule files import planners themselves: `stitching-py.smk:10`, `reads-assignment.smk:22`,
`export.smk:14`, `report.smk:9`, and `registration-py.smk:91-92` inside `get_dapi_input`.

% results/listings/common_smk_imports.tsv

## The boundary inventory

Every place where logic that can change a result sits in a `.smk` file, a `workflow/scripts/` file
or package code that is coupled to the engine, and where it goes. "Python" is the Python API and
the plan; "frozen" is the MATLAB legacy workflow (D7).

(orchestration-boundary)=

| Where | What can change a result | Goes to |
| --- | --- | --- |
| `registration-py.smk:110-122` (`rotate_nuclei`, Python) | The DAPI image is rotated inline with `scipy.ndimage.rotate` at its default spline order 3 with `reshape`, then projected by `maximum_projection`; the package rotates the sequencing rounds with `np.rot90` for multiples of 90° and bilinear interpolation without `reshape` otherwise (`preprocessing/_rotation.py:12-23`) | D6: retired for Python; the morphology task rotates with the package (`FOV.prepare_morphology`) |
| `registration.smk:94-106` (`rotate_nuclei`, MATLAB) | The same inline rotation | Frozen (D7) |
| `registration-py.smk:90-101` (`get_dapi_input`) | The stain file is found by a glob; the `image_dir_template` default is restated (`:93`) | D2: input discovery in Python |
| `common.smk:259-264` (`N_SUBTILE`) | The subtile count and its default `sqrt_pieces` 4, restated beside `from_workflow_config` (`workflow.py:589-599`) | D5 (Python); frozen (D7) |
| `common.smk:203` (`SEG_INPUT_DIR`) | The segmentation input folder and its default `overlay`, restated beside `_legacy_target` and `_segmentation` (`workflow.py:1066,1182`) | D6: retired; frozen for MATLAB |
| `common.smk:156-159`, `get_rule_config`, `make_get_runtime` | Resource defaults and the runtime per attempt | D4: the execution section (defaults in Python); D8: retries are the engine's |
| `common.smk:259-279` (MATLAB subsets) | The FOV selection, including the unseeded `random.sample` (`:279`) | Frozen (D7) |
| `common.smk:283-385` (`get_overall_output`), `Snakefile:63-74` (`ruleorder`) | Which rules run and which producer writes a FOV's molecule table; the same choice is made again by `_legacy_molecules` (`workflow.py:1070-1087`) | D2: the manifest; D4: the plan derives the graph; D5 retires the second producer for Python |
| `stitching-py.smk:12`, `:26-28`, `:33` | The default method `bigstitcher` restated, a parse-time refusal, and whether the tile configuration is declared | D2: the plan |
| `workflow/scripts/stitch_subtile.py:19-44` | The merge of the subtile molecule tables (one-based offsets, the keep rule by subtile corner) | D5: retired for Python; frozen for MATLAB |
| `workflow/scripts/stitch_subtile.py:46-62` | The preview of the merged molecules on the reference image | D5; frozen |
| `workflow/scripts/stitching_preparation.py`, `create_BigStitcher_macro_1.py`, `create_tile_config.py`, `create_sample_maf.py`, `create_sample_h5ad.py`, `create_sample_reads_assignment.py` | The MATLAB sample route: grid from the stage file, blank padding and tile copies (`Position{id:03}` hard-coded, `stitching_preparation.py:104`), the macro, the non-overlap rectangles, the per-sample stage file and the sample aggregation | Frozen (D7) |
| `_snakemake.py:814-831` (`_run_bigstitcher_engine`) | The package runs Fiji through `subprocess` with its Java folder and checks `dataset.xml` | D3: the engine task runs `fiji_path`; Python writes the macro and imports and checks `dataset.xml` |
| `_snakemake.py:686-699` (`_write_tile_config`), `:318-452` (the legacy writers and the overlap filter) | The tile configuration, the global offsets and the overlap filter applied outside the API, and the MATLAB-shaped per-FOV files | D6: retired for Python; D7: the legacy module |
| `_snakemake.py:73-138` (`_run_nuclei_registration`) | The rounds registered in groups by stain pattern, and the image files written per channel label (with the projection) | D6: the morphology command |
| `_snakemake.py:148-224` | The overlay, the Flamingo enhancement and the StarDist run on legacy files | D6 (Python); D7 (MATLAB's shared rules) |
| `_snakemake.py:483-557` (`_run_reads_assignment`) | The molecule reader chosen by producer, the grid of the label file, the record name | D6: the assign command; D7 |
| `_snakemake.py:707-778` (`_run_stitching`, `_run_bigstitcher_macro`) | The regular grid laid out over the sample's unselected FOVs too (`_sample_table`), the reference grids | D2: the stitch and macro commands |
| `_snakemake.py:888-956` (`_run_export`, `_write_ownership_figures`) | The previews folder created when declared, the ownership figures | D2: the export command (P5) |
| `common.smk:212`, every `workflow/scripts/*.py:3-4` | The package is imported from the checkout that `starfinder_path` names, not from the environment | D2: the engines call the CLI; D9: the locked environment and image |
| `workflow/rules/matlab_launcher.py` | The MATLAB command line | Frozen (D7) |

## The Python configuration as read today

The Python dialect is every key that a configuration with `backend: python` may hold, read when
the rule files are parsed (`common.smk:413-447`) and again by the rule's runner. The table gives
each key, where it is read (the function, or the module-level key table, with its file and line;
unless named otherwise the file is `src/python/starfinder/dataset/workflow.py`), and the default the
translator applies. Where the translator applies another default than the config dataclass it
builds, both are given (the decoders' `diagnostics` and `allow_rescue`, the processing rules'
checkpoints, the extraction's background ring, the subtile count, the StarDist scale, the legacy
expansions). "schema" marks a default that `snakemake.utils.validate` inserts (`Snakefile:19`).
Where a mapping's keys are the init fields of a config dataclass, the appendix lists every field
with the dataclass default and, where it differs, the translator's. This is the input of the
migration table of decision D4.

The table is generated by a sweep of the translators, which checks that each cited line holds the
key and that every string the translators read as a key is cited here or is no key of the dialect
(a value, a message, a column of `fovs.csv`, a key of a written record), each exclusion with its
reason; the appendix covers the dataclass fields, which the sweep by construction does not.

% Sweep: run directory results/listings/translator_keys.{csv,json,md} (step translator_keys.rerun-1, script
% scripts/translator_keys.py); the record check compares this table with translator_keys.md (record_check.json).

(orchestration-python-keys)=

| Key | Read by | Default (the translator's; the dataclass's where it differs) |
| --- | --- | --- |
| `config_path` | `from_workflow_config` (`workflow.py:542`) | required by the schema |
| `starfinder_path` | `common.smk:212` | required by the schema |
| `root_input_path` | `_workflow_dataset` (`workflow.py:879`), `_input_root` (`workflow.py:1611`) | required by the schema |
| `dataset_id`, `sample_id` | `_workflow_dataset` (`workflow.py:879`), `_run_reads_assignment` (`_snakemake.py:524`), `_run_export` (`_snakemake.py:922`) | required by the schema |
| `root_output_path`, `output_id` | `_workflow_dataset` (`workflow.py:880`), `_run_reads_assignment` (`_snakemake.py:524`), `_run_export` (`_snakemake.py:922`) | required by the schema |
| `fiji_path` | `_run_bigstitcher_engine` (`_snakemake.py:823`) | none; read when `bigstitcher_engine` runs |
| `envs_path` | no reader | none; no reader |
| `selected_fovs` | `_membership` (`workflow.py:1533`) | null: every FOV of `fovs.csv` |
| `n_rounds` | `_workflow_dataset` (`workflow.py:867`) | required by the schema |
| `ref_round` | `_workflow_dataset` (`workflow.py:872`) | required by the schema |
| `ref_channel` | `_nuclear_stain` (`workflow.py:846`), `_nuclei_registration` (`workflow.py:979`) | none |
| `reference_stains` | `_reference_stains` (`workflow.py:820`), `_reference_stains` (`workflow.py:825`) | null: none |
| `seq_channel_order` | `_workflow_dataset` (`workflow.py:874`) | none; an empty list loads no raw image (`from_workflow_config`, `:581`) |
| `seq_channel_order[]`, `reference_stains[]`, `additional_round[].channel_order[]`: a pattern string or a mapping `{channel, name, wavelength}` | `from_value` (`types.py:100`), `from_value` (`types.py:105`), `from_value` (`types.py:106`) | `name` and `wavelength` none (`wavelength: unavailable` reads as none) |
| `additional_round` | `_additional_rounds` (`workflow.py:795`) | none; required when the rule files are parsed (`common.smk:184-185`) |
| `additional_round[].round_name` | `_additional_rounds` (`workflow.py:800`) | required |
| `additional_round[].channel_order` | `_additional_rounds` (`workflow.py:804`) | none: the round takes the sequencing channels |
| `additional_round[].channel_order[].channel`, `.name` (`nuclei_registration`) | `_nuclei_registration` (`workflow.py:988`) | required by that rule |
| `rotate_angle` | `from_workflow_config` (`workflow.py:582`), `_run_nuclei_registration` (`_snakemake.py:108`), `registration-py.smk:115` | required by the schema; 0 is a rotation step (O11); none for the resident subtile rules |
| `img_row`, `img_col` | `_image_shape` (`workflow.py:760`), `_image_shape` (`workflow.py:764`) | none; required by the subtile windows and the legacy tile configuration writer |
| `img_z` | `_legacy_cells` (`_snakemake.py:276`) | none; required for a YX label file of `reads_assignment` |
| `voxel_size_xy`, `voxel_size_z` | `_workflow_dataset` (`workflow.py:883`), `_declared_metadata` (`workflow.py:1109`), `reference_settings` (`_declared.py:39`) | none |
| `reference_image` | `reference_settings` (`_declared.py:33`), `_layout.py:53`, `_layout.py:55` | `fovs/{fov_id}/reference.tif` |
| `reference_image_import` | `reference_settings` (`_declared.py:33`), `_layout.py:54` | null: none |
| `image_dir_template` | `_image_dir_template` (`workflow.py:894`), `registration-py.smk:93` | `"{round}/{fov}"` |
| `readout_mode` | `_readout_mode` (`workflow.py:387`), `_reference_file` (`workflow.py:916`) | `multiplexed` (schema) |
| `codebook` (`readout_mode: multiplexed`), `direct_panel` (`direct`) | `_REFERENCE_KEYS` (`workflow.py:902`) | `<input root>/genes.csv` |
| `maximum_projection` | `from_workflow_config` (`workflow.py:606`), `_segmentation` (`workflow.py:1184`), `_run_nuclei_registration` (`_snakemake.py:126`), `registration-py.smk:119` | `false` (schema) |
| `backend` | `from_workflow_config` (`workflow.py:533`), `_python_only` (`workflow.py:1045`), `_reference_stains` (`workflow.py:819`), `_legacy_molecules` (`workflow.py:1080`), `_fov_sample` (`_snakemake.py:238`) | `matlab` (schema) |
| `workflow_mode` | `_legacy_molecules` (`workflow.py:1082`), `common.smk:38` | `free` (schema) |
| `rules`, `rules.<rule>.parameters` | `from_workflow_config` (`workflow.py:532`), `_rule_parameters` (`workflow.py:1037`), `from_workflow_config` (`workflow.py:594`), `_first_rule` (`workflow.py:944`), `_legacy_molecules` (`workflow.py:1085`), `_check_sample_rules` (`workflow.py:1588`) | required by the schema; `parameters` absent: `{}` |
| `rules.<rule>.run` | `_legacy_molecules` (`workflow.py:1086`), `_check_sample_rules` (`workflow.py:1589`), `common.smk:120` | `false`; required by the schema in each section |
| `rules.<rule>.resources.mem_mb`, `.runtime` | `registration-py.smk:26`, `registration-py.smk:27` | 8000, 30 (`common.smk:156-159`) |
| `matlab_launcher`, `matlab_single_thread` (MATLAB rules only) | `matlab_command` (`matlab_launcher.py:42`), `matlab_command` (`matlab_launcher.py:45`) | `path`, `false` (schema) |
| `rules.<processing rule>.parameters`: the keys it accepts | `from_workflow_config` (`workflow.py:537`), `from_workflow_config` (`workflow.py:538`), `from_workflow_config` (`workflow.py:539`) | unknown keys raise |
| `rules.<processing rule>.parameters.streaming` | `from_workflow_config` (`workflow.py:600`) | `false`: `ExecutionConfig.mode` `batch` |
| `rules.<processing rule>.parameters.device` | `from_workflow_config` (`workflow.py:605`) | `cpu` |
| `rules.<processing rule>.parameters.<block>.run` (`spot_finding`, `reads_extraction`, `reads_filtration`, `load_codebook`, `load_raw_images`, `create_subtiles`) | `_operation` (`workflow.py:72`), `_operation` (`workflow.py:75`) | `false`; must be Boolean |
| `rules.<processing rule>.parameters.load_codebook`, `.load_codebook.run` | `from_workflow_config` (`workflow.py:562`) | loaded whenever the rule decodes; `run` is not read (`_snakemake.py:51-53`) |
| `rules.<processing rule>.parameters.load_codebook.split_index` | `from_workflow_config` (`workflow.py:562`), `_codebook_layout` (`workflow.py:422`) | none: one segment |
| `rules.<processing rule>.parameters.load_codebook.encoding` | `from_workflow_config` (`workflow.py:562`), `_codebook_layout` (`workflow.py:421`) | `two_base` with the default table |
| `rules.<processing rule>.parameters.load_codebook.encoding.method` | `_encoding` (`workflow.py:277`) | `two_base` |
| `rules.<processing rule>.parameters.load_codebook.encoding.base_to_color`, `.pair_to_color` | `_encoding` (`workflow.py:280`) | the dataclass's (appendix); integer colours read as strings |
| `rules.<processing rule>.parameters.load_raw_images`, `.subdir` | `from_workflow_config` (`workflow.py:574`), `from_workflow_config` (`workflow.py:581`) | load the raw images (not in the resident subtile rules); `subdir` the `ImageLoadConfig` default `''` |
| `rules.<processing rule>.parameters.preprocessing` | `from_workflow_config` (`workflow.py:547`), `_explicit_recipe` (`workflow.py:231`) | none: no preprocessing |
| `….preprocessing.steps[]`, `….preprocessing.post_registration[]` | `_explicit_recipe` (`workflow.py:233`), `_explicit_recipe` (`workflow.py:234`) | `steps` required; `post_registration` `[]` |
| `….preprocessing.extraction_source`, `.registration_source`, `.supplied_statistics` | `_explicit_recipe` (`workflow.py:236`), `_explicit_recipe` (`workflow.py:237`), `_explicit_recipe` (`workflow.py:235`) | none |
| `….preprocessing.steps[].method`, `….preprocessing.post_registration[].method` | `_preprocessing_steps` (`workflow.py:211`), `_preprocessing_steps` (`workflow.py:214`) | required: a `PREPROCESSING_METHODS` name |
| `….preprocessing.steps[].save_as`, `….preprocessing.post_registration[].save_as` | `_preprocessing_steps` (`workflow.py:215`) | none: the step's output is not saved |
| `rules.<processing rule>.parameters.registration` | `from_workflow_config` (`workflow.py:548`) | none: no registration |
| `….registration.steps[]` | `_explicit_registration` (`workflow.py:127`), `_explicit_registration` (`workflow.py:133`) | required, a list |
| `….registration.signal`, `.warp`, `.qc` | `_explicit_registration` (`workflow.py:130`) | none: the recipe's defaults |
| `….registration.reference_round` | `_explicit_registration` (`workflow.py:134`) | none: the dataset's reference round |
| `….registration.steps[].method` | `_registration_step` (`workflow.py:97`), `_registration_step` (`workflow.py:100`) | required: a `REGISTRATION_METHODS` name |
| `….registration.steps[].signal` | `_registration_step` (`workflow.py:101`) | none: the recipe's signal |
| `….registration.steps[].recovery` | `_registration_step` (`workflow.py:101`) | none: no recovery |
| `….registration.steps[].recovery.allowed_errors` | `_recovery` (`workflow.py:87`), `_recovery` (`workflow.py:89`), `_RECOVERY_ERRORS` (`workflow.py:78`), `_RECOVERY_ERRORS` (`workflow.py:79`), `_RECOVERY_ERRORS` (`workflow.py:80`) | required: a list of `InsufficientLandmarksError`, `RegistrationEstimationError`, `RegistrationRejectedError` (`:78-80`); another name raises `ValueError` |
| `….registration.steps[].recovery.alternatives[]` | `_recovery` (`workflow.py:87`), `_recovery` (`workflow.py:92`), `alternative` (`workflow.py:106`) | required: a list of estimators (a `method` and its fields); `recovery` or `signal` in an alternative raises (`:106-107`) |
| `rules.<processing rule>.parameters.reference_image_view`, `.mode`, `.channel` | `_reference_image_view` (`workflow.py:731`), `_reference_image_view` (`workflow.py:736`), `_reference_image_view` (`workflow.py:739`), `_reference_image_view` (`workflow.py:741`), `_reference_image_view` (`workflow.py:746`) | `{mode: max}`; `channel` required with `mode: channel` |
| `rules.<processing rule>.parameters.spot_finding` | `from_workflow_config` (`workflow.py:558`) | none: no detection |
| `rules.<processing rule>.parameters.spot_finding.ref_round` | `from_workflow_config` (`workflow.py:558`), `from_workflow_config` (`workflow.py:577`), `_detection` (`workflow.py:172`) | the reference round |
| `rules.<processing rule>.parameters.spot_finding.method` | `from_workflow_config` (`workflow.py:558`), `_detection` (`workflow.py:173`) | `local_maxima` |
| `rules.<processing rule>.parameters.spot_finding.channel_overrides.<label>` | `from_workflow_config` (`workflow.py:558`), `_detection` (`workflow.py:174`) | none; each entry holds the method's fields |
| `rules.<processing rule>.parameters.spot_finding.rounds` | `from_workflow_config` (`workflow.py:558`), `_detection` (`workflow.py:175`) | none: the reference round only |
| `rules.<processing rule>.parameters.reads_extraction`, `.voxel_size` | `from_workflow_config` (`workflow.py:560`), `_extraction` (`workflow.py:315`) | none: no extraction; `voxel_size` `(1, 2, 2)` (the dataclass's) |
| `rules.<processing rule>.parameters.reads_extraction.background` | `from_workflow_config` (`workflow.py:560`), `_extraction` (`workflow.py:316`) | absent: the ring of `LocalBackgroundConfig` (inner `(1, 3, 3)`, outer `(1, 6, 6)`, 16 voxels) grown along every axis where `voxel_size` exceeds the inner box (`:319-325`); `false`: none; a mapping: its fields |
| `rules.<processing rule>.parameters.reads_filtration` | `from_workflow_config` (`workflow.py:561`) | none: no decoding or filtering |
| `rules.<processing rule>.parameters.reads_filtration.end_base` | `from_workflow_config` (`workflow.py:561`), `_codebook_layout` (`workflow.py:432`) | none |
| `rules.<processing rule>.parameters.reads_filtration.n_barcode_segments`, `.split_index` | `from_workflow_config` (`workflow.py:561`), `_codebook_layout` (`workflow.py:426`), `_codebook_layout` (`workflow.py:430`) | none: checked against the codebook layout |
| `rules.<processing rule>.parameters.reads_filtration.start_base`, `.exclude_invalid_endpoints`, `.score_bounds` | `from_workflow_config` (`workflow.py:588`), `from_workflow_config` (`workflow.py:561`) | `C`, `false`, `{}` (the `ReadFilterConfig` defaults) |
| `rules.<processing rule>.parameters.decoding` | `from_workflow_config` (`workflow.py:567`), `from_workflow_config` (`workflow.py:569`), `from_workflow_config` (`workflow.py:570`) | absent: `WtaDecoderConfig(negative_policy='reject', diagnostics=True)`, or with `readout_mode: direct` `DirectDecoderConfig()`; needs `reads_filtration.run` |
| `rules.<processing rule>.parameters.decoding.method` | `_decoding` (`workflow.py:296`) | `wta`, or `direct` with `readout_mode: direct` |
| `rules.<processing rule>.parameters.decoding` of method `wta` | `_decoding` (`workflow.py:302`) | `negative_policy` `reject`, `diagnostics` `true` (the dataclass: `false`) |
| `rules.<processing rule>.parameters.decoding` of method `codebook_aware` | `_decoding` (`workflow.py:302`) | the dataclass's fields, with `diagnostics` `true` and `allow_rescue` `false` (the dataclass: `false` and `true`) |
| `rules.<processing rule>.parameters.decoding` of method `direct` | `_decoding` (`workflow.py:296`) | `DirectDecoderConfig()` (no field) |
| `rules.<processing rule>.parameters.scoring`, `.run`, `.method` | `_scoring` (`workflow.py:336`), `_scoring` (`workflow.py:339`), `_scoring` (`workflow.py:340`), `_scoring` (`workflow.py:343`) | on whenever the rule decodes (`ReadScoreConfig()`); `method` `bgcorr_probability` |
| `rules.<processing rule>.parameters.deduplication`, `.run` | `_deduplication` (`workflow.py:361`), `_deduplication` (`workflow.py:367`), `_deduplication` (`workflow.py:368`), `_deduplication` (`workflow.py:363`) | off; with `run: true` the `DeduplicationConfig` fields |
| `rules.<processing rule>.parameters.create_subtiles`, `.sqrt_pieces`, `.overlap_ratio` (of `gr_single_fov_subtile` or `deep_create_subtile`) | `from_workflow_config` (`workflow.py:595`), `from_workflow_config` (`workflow.py:597`) | `sqrt_pieces` 4 (`SubtileConfig` has no default), `overlap_ratio` 0.1 |
| `rules.<processing rule>.parameters.checkpoints`, `.stages` | `_rule_checkpoints` (`workflow.py:777`), `_rule_checkpoints` (`workflow.py:784`), `_rule_checkpoints` (`workflow.py:785`) | `{stages: []}`: `run.json` alone, with `overwrite: true` (the `CheckpointConfig` defaults: all three stages, `overwrite: false`) |
| Refused in `rules.<processing rule>.parameters`: `enhance_contrast`, `hist_equalize`, `morph_recon`, `tophat`, `snr_threshold`, `global_registration`, `local_registration` | `_NATIVE_FORMS` (`workflow.py:30`), `_NATIVE_FORMS` (`workflow.py:32`), `_NATIVE_FORMS` (`workflow.py:33`), `_NATIVE_FORMS` (`workflow.py:35`), `_NATIVE_FORMS` (`workflow.py:36`), `_NATIVE_FORMS` (`workflow.py:37`), `_NATIVE_FORMS` (`workflow.py:39`), `_check_python_dialect` (`workflow.py:634`) | raise `ValueError` naming the native form |
| Refused in `rules.<processing rule>.parameters.spot_finding` and its `channel_overrides` (method `local_maxima`): `intensity_estimation`, `intensity_threshold`, `min_distance` | `_NATIVE_SPOT_FIELDS` (`workflow.py:43`), `_NATIVE_SPOT_FIELDS` (`workflow.py:44`), `_check_spot_keys` (`workflow.py:662`) | raise `ValueError` naming the `LocalMaximaConfig` fields |
| Refused in `….registration` steps and alternatives: the MATLAB parameter names and `ref_img`, `mov_img`, `ref_channel`, `boundary_mode`, `ref_round`; the demons variant names as `method`; `merged-image`, `merged`, `single-channel` as a signal `mode` | `_NATIVE_PARAMETERS` (`workflow.py:46`), `_NATIVE_PARAMETERS` (`workflow.py:47`), `_NATIVE_PARAMETERS` (`workflow.py:48`), `_NATIVE_PARAMETERS` (`workflow.py:49`), `_NATIVE_BLOCK_KEYS` (`workflow.py:52`), `_NATIVE_BLOCK_KEYS` (`workflow.py:53`), `_NATIVE_BLOCK_KEYS` (`workflow.py:54`), `_NATIVE_BLOCK_KEYS` (`workflow.py:55`), `_NATIVE_BLOCK_KEYS` (`workflow.py:56`), `_VARIANT_METHODS` (`workflow.py:58`), `_MATLAB_SIGNALS` (`workflow.py:59`) | raise `ValueError` naming the native form |
| Refused in `….preprocessing` steps: `radius` of a `reconstruction` or `white_tophat` step | `_check_preprocessing_names` (`workflow.py:714`) | raise `ValueError` naming `radius_yx` |
| Refused at the top level: `dapi_round`, `channel_order`, `fov_pattern` | `_check_python_dialect` (`workflow.py:629`), `_reference_stains` (`workflow.py:823`), `from_workflow_config` (`workflow.py:529`) | raise `ValueError` |
| `readout_mode: direct` refuses `rules.<processing rule>.parameters.load_codebook.split_index`, `.encoding` and `reads_filtration.end_base`, `.split_index`, `.n_barcode_segments`, `.exclude_invalid_endpoints` | `_BARCODE_KEYS` (`workflow.py:381`), `_BARCODE_KEYS` (`workflow.py:382`) | raise `ValueError` |
| `rules.nuclei_registration.parameters.checkpoints` | `_nuclei_checkpoints` (`workflow.py:1011`) | null: none; a mapping is a `CheckpointConfig` with its own defaults (all three stages, `overwrite: false`) |
| `rules.create_nuclei_amplicon_overlay.parameters.maximum_projection` | `_run_nuclei_amplicon_overlay` (`_snakemake.py:159`), `_segmentation` (`workflow.py:1185`) | `false` |
| `rules.stardist_segmentation.parameters`: the keys it accepts | `_STARDIST_KEYS` (`workflow.py:1023`), `_STARDIST_KEYS` (`workflow.py:1024`) | unknown keys raise |
| `rules.stardist_segmentation.parameters.stardist_base_path`, `.stardist_model_name` | `_segmentation` (`workflow.py:1164`) | required |
| `rules.stardist_segmentation.parameters.segmentation_input_folder` | `_segmentation` (`workflow.py:1182`), `_legacy_target` (`workflow.py:1066`), `_STARDIST_KEYS` (`workflow.py:1023`) | `overlay` |
| `rules.stardist_segmentation.parameters.prob_thresh`, `.nms_thresh` | `_segmentation` (`workflow.py:1172`), `_STARDIST_KEYS` (`workflow.py:1023`) | the model's stored thresholds (`StarDistConfig` `None`) |
| `rules.stardist_segmentation.parameters.rescale` | `_segmentation` (`workflow.py:1175`), `_STARDIST_KEYS` (`workflow.py:1024`) | `false`: `StarDistConfig.scale` 1.0 (`true`: 0.5; the dataclass has no default) |
| `rules.stardist_segmentation.parameters.expand_labels`, `.distance` | `_segmentation` (`workflow.py:1175`), `_segmentation` (`workflow.py:1179`), `_segmentation` (`workflow.py:1181`), `_STARDIST_KEYS` (`workflow.py:1024`), `_reads_assignment` (`workflow.py:1351`) | `false`; `distance` required with `true` (`ExpandLabelsConfig(distance, 'pixel', 'planar')`) |
| `rules.stardist_segmentation.parameters.target`, `.device` | `_legacy_target` (`workflow.py:1060`), `_legacy_target` (`workflow.py:1062`), `_segmentation` (`workflow.py:1163`), `_segmentation` (`workflow.py:1193`), `_STARDIST_KEYS` (`workflow.py:1024`) | `cell` for `overlay`, else `nucleus`; `cpu` |
| `rules.reads_assignment.parameters.expand_labels`, `.dilation_distance` | `_READS_ASSIGNMENT_KEYS` (`workflow.py:1025`), `_reads_assignment` (`workflow.py:1350`), `_reads_assignment` (`workflow.py:1361`), `_reads_assignment` (`workflow.py:1363`) | `false`; `dilation_distance` required with `true` (`ExpandLabelsConfig(d, 'pixel', 'planar')` and `legacy_pixel_expansion: true`; the `AssignmentConfig` default `false`) |
| `segmentation` | `_segmentation_block` (`workflow.py:1277`), `_segmentation` (`workflow.py:1155`) | none; no rule runs it (`_segmentation`, `:1155-1160`, refuses it) |
| `segmentation.device`, `.runs` | `_segmentation_block` (`workflow.py:1280`), `_segmentation_block` (`workflow.py:1284`), `_segmentation_block` (`workflow.py:1281`) | `cpu`; `runs` required, nonempty |
| `segmentation.runs[].name`, `.method` | `_segmentation_run` (`workflow.py:1234`), `_SEGMENTATION_RUN_KEYS` (`workflow.py:1029`) | required (`method` `import` or a `SEGMENTATION_METHODS` name) |
| `segmentation.runs[].target`, `.seeds` | `_SEGMENTATION_RUN_KEYS` (`workflow.py:1029`), `_segmentation_run` (`workflow.py:1258`), `_segmentation_run` (`workflow.py:1239`) | none |
| `segmentation.runs[].projection` | `_SEGMENTATION_RUN_KEYS` (`workflow.py:1029`), `_segmentation_run` (`workflow.py:1240`) | none; `true`: `ProjectionConfig()`; `false`: none; a mapping: its fields |
| `segmentation.runs[].operations[]`, `.operation` | `_SEGMENTATION_RUN_KEYS` (`workflow.py:1029`), `_segmentation_run` (`workflow.py:1248`), `_segmentation_run` (`workflow.py:1250`), `_segmentation_run` (`workflow.py:1249`) | `[]`; `operation` required, `expand_labels` or `extend_labels_through_z` |
| `segmentation.runs[].inputs[]` | `_SEGMENTATION_RUN_KEYS` (`workflow.py:1029`), `_segmentation_run` (`workflow.py:1255`) | `[]` |
| `segmentation.runs[].inputs[].prepare`, `.prepare.function` | `_input_channel` (`workflow.py:1202`), `_input_channel` (`workflow.py:1205`), `_input_channel` (`workflow.py:1208`), `_input_channel` (`workflow.py:1204`) | none; `function` required, `composite_nuclei_amplicon` or `enhance_with_flamingo` |
| `assignment` | `_assignment_block` (`workflow.py:1306`), `_reads_assignment` (`workflow.py:1342`), `_sample_segmentation_block` (`workflow.py:1715`) | none; no rule runs it (`_reads_assignment`, `:1342-1347`, refuses it); read by `export.segmentation` (`:1715`) |
| `assignment.cells`, `.nuclei`, `.name`, `.population` | `_assignment_block` (`workflow.py:1320`), `_ASSIGNMENT_CALL_KEYS` (`workflow.py:1033`) | none (the `FOV.assign_molecules` arguments) |
| `assignment.expansion`, `.correspondence` | `_assignment_block` (`workflow.py:1312`), `_assignment_block` (`workflow.py:1314`), `_ASSIGNMENT_CALL_KEYS` (`workflow.py:1033`) | none: the `AssignmentConfig` defaults |
| `assignment.legacy_pixel_expansion`, `.exclude_cells_without_nucleus` | `_assignment_block` (`workflow.py:1317`) | the `AssignmentConfig` defaults |
| `assignment.checkpoints` | `_assignment_block` (`workflow.py:1321`), `_ASSIGNMENT_CALL_KEYS` (`workflow.py:1033`) | none; a mapping is a `CheckpointConfig` |
| `stitching` | `_stitching_block` (`workflow.py:1427`), `_stitching_position_inputs` (`_snakemake.py:588`), `_report_sample_files` (`_snakemake.py:972`) | none |
| `stitching.method` | `_stitching_block` (`workflow.py:1431`), `_STITCHING_KEYS` (`workflow.py:1374`) | `bigstitcher` |
| `stitching.input` | `_stitching_block` (`workflow.py:1437`), `_STITCHING_KEYS` (`workflow.py:1374`) | `reference_merged` |
| `stitching.config` | `_stitching_block` (`workflow.py:1436`), `_STITCHING_KEYS` (`workflow.py:1374`) | `{}`: the method config's defaults |
| `stitching.review`, `.fallback` | `_stitching_block` (`workflow.py:1451`), `_stitching_block` (`workflow.py:1452`), `_STITCHING_KEYS` (`workflow.py:1374`) | `ReviewPolicy()`; `fallback` null |
| `stitching.bigstitcher_save_dir` | `_stitching_block` (`workflow.py:1455`), `_STITCHING_KEYS` (`workflow.py:1374`) | required with method `bigstitcher`, refused otherwise |
| `stitching.positions` | `_stitching_block` (`workflow.py:1439`), `_STITCHING_KEYS` (`workflow.py:1374`), `_stage_position_ids` (`workflow.py:1569`), `_stitching_position_inputs` (`_snakemake.py:588`) | required |
| `stitching.positions.source` | `_stitching_block` (`workflow.py:1443`), `table_from_positions` (`_stitching.py:174`), `_stage_position_ids` (`workflow.py:1570`), `_stitching_position_inputs` (`_snakemake.py:589`), `_run_stitching` (`_snakemake.py:740`) | required: `stage_file`, `regular_grid` or `table` |
| `stitching.positions.stage_file`, `.stage_axes`, `.stage_axes.x`, `.stage_axes.y` | `_stitching_block` (`workflow.py:1446`), `_stitching_block` (`workflow.py:1443`), `table_from_positions` (`_stitching.py:183`), `table_from_positions` (`_stitching.py:186`), `table_from_positions` (`_stitching.py:196`), `_stitching_position_inputs` (`_snakemake.py:590`) | required with `stage_file` |
| `stitching.positions.snap_to_grid`, `.snap_to_grid.overlap` | `table_from_positions` (`_stitching.py:191`), `table_from_positions` (`_stitching.py:192`), `table_from_positions` (`_stitching.py:194`) | none: no snap |
| `stitching.positions.regular_grid`, `.columns`, `.rows`, `.order`, `.overlap` | `table_from_positions` (`_stitching.py:177`), `table_from_positions` (`_stitching.py:179`) | required with `regular_grid` |
| `stitching.positions.table`, `.unit` | `_stitching_block` (`workflow.py:1449`), `table_from_positions` (`_stitching.py:199`), `table_from_positions` (`_stitching.py:201`), `_stitching_position_inputs` (`_snakemake.py:592`) | required with `table`; `unit` `µm` |
| `stitching.positions.tile_shape_yx`, `.spacing_yx` | `_geometry` (`_stitching.py:99`) | the reference grids' |
| `assembly` | `_assembly_block` (`workflow.py:1658`), `_assembly_block` (`workflow.py:1656`), `_stitching_block` (`workflow.py:1461`), `_run_reads_assignment` (`_snakemake.py:522`), `_run_stitching` (`_snakemake.py:752`), `_report_sample_files` (`_snakemake.py:970`) | none; needs a `stitching` block; without it the BigStitcher import writes the legacy tile configuration |
| `assembly.assignment` | `_assembly_block` (`workflow.py:1665`), `_ASSEMBLY_CALL_KEYS` (`workflow.py:1379`) | `default`; null: molecules only |
| `assembly.table_format`, `.previews` | `_assembly_block` (`workflow.py:1668`), `_assembly_block` (`workflow.py:1671`), `_ASSEMBLY_CALL_KEYS` (`workflow.py:1379`) | `csv`, `false` |
| `export` | `_export_config` (`workflow.py:1737`), `_export_config` (`workflow.py:1733`) | absent: the defaults; needs an `assembly` block |
| `export.segmentation` | `_export_config` (`workflow.py:1748`) | none |
| `export.segmentation.device`, `.cells`, `.nuclei`, `.runs` | `_sample_segmentation_block` (`workflow.py:1696`), `_sample_segmentation_block` (`workflow.py:1717`), `_sample_segmentation_block` (`workflow.py:1716`), `_sample_segmentation_block` (`workflow.py:1697`) | `cpu`, none, none; `runs` required |
| `export.segmentation.runs[].name`, `.method`, `.target`, `.seeds`, `.inputs` | `_SAMPLE_SEGMENTATION_RUN_KEYS` (`workflow.py:1031`), `_sample_segmentation_block` (`workflow.py:1702`), `_sample_segmentation_block` (`workflow.py:1714`), `_sample_segmentation_block` (`workflow.py:1711`), `_sample_segmentation_block` (`workflow.py:1706`) | `name` and `method` required |
| `export.segmentation.runs[].inputs[].role`, `.image` | `_sample_segmentation_block` (`workflow.py:1710`), `_sample_segmentation_block` (`workflow.py:1713`) | none |
| `report` | `_report_block` (`workflow.py:1815`) | `{}` or null: `ReportConfig()` |
| `report.markers[]`, `report.limits`, `report.limits.spots_near_zero` | `_report_block` (`workflow.py:1822`), `_report_block` (`workflow.py:1826`), `_report_block` (`workflow.py:1832`) | the `ReportConfig` defaults |

### Appendix: the dataclass fields of the Python dialect

Generated from the package at `56c35b8`: every init field and its declared default
("required" without one), and in the last column the default the translator applies instead, where
it applies one ("none": the dataclass is built from the given fields alone).

% results/listings/python_dialect_fields.csv and .md (step python_keys.rerun-1)

| Where | Translator | Dataclass | Fields and the dataclass defaults | The translator's default where it differs |
| --- | --- | --- | --- | --- |
| `rules.<processing rule>.parameters.preprocessing.steps[] and .post_registration[] (method min_max_normalization)` | `_preprocessing_steps` | `MinMaxNormalizationConfig` | `output_dtype` (required), `output_range` (required), `scope` "per_channel", `snr_threshold` null, `rounding` "truncate" | none |
| `rules.<processing rule>.parameters.preprocessing.steps[] and .post_registration[] (method histogram_matching)` | `_preprocessing_steps` | `HistogramMatchingConfig` | `output_dtype` null, `rounding` "truncate", `reference_channel` 0, `fit` "fov" | none |
| `rules.<processing rule>.parameters.preprocessing.steps[] and .post_registration[] (method reconstruction)` | `_preprocessing_steps` | `ReconstructionConfig` | `radius_yx` 3 | none |
| `rules.<processing rule>.parameters.preprocessing.steps[] and .post_registration[] (method white_tophat)` | `_preprocessing_steps` | `TophatConfig` | `radius_yx` 3 | none |
| `rules.<processing rule>.parameters.preprocessing.steps[] and .post_registration[] (method scalar_background)` | `_preprocessing_steps` | `ScalarBackgroundConfig` | `percentile` 10.0, `fit` "fov" | none |
| `rules.<processing rule>.parameters.preprocessing.steps[] and .post_registration[] (method background_3d)` | `_preprocessing_steps` | `Background3DConfig` | `radius_um_zyx` null, `radius_voxels_zyx` null | none |
| `rules.<processing rule>.parameters.preprocessing.steps[] and .post_registration[] (method percentile_normalization)` | `_preprocessing_steps` | `PercentileNormalizationConfig` | `p_low` 1.0, `p_high` 99.9, `fit` "fov" | none |
| `rules.<processing rule>.parameters.registration.steps[] and recovery.alternatives[] (method translation)` | `_registration_step` | `TranslationConfig` | `backend` "scipy_fft", `fft_workers` 1 | none |
| `rules.<processing rule>.parameters.registration.steps[] and recovery.alternatives[] (method rigid)` | `_registration_step` | `RigidConfig` | `metric` "mattes", `histogram_bins` 32, `iterations` 200, `samples` 4096, `levels` null, `random_seed` 1 | none |
| `rules.<processing rule>.parameters.registration.steps[] and recovery.alternatives[] (method affine)` | `_registration_step` | `AffineConfig` | `metric` "ncc", `iterations` 200, `samples` 4096, `levels` null, `random_seed` 1 | none |
| `rules.<processing rule>.parameters.registration.steps[] and recovery.alternatives[] (method bspline)` | `_registration_step` | `BSplineConfig` | `metric` "ncc", `grid_spacing_physical` null, `iterations` 200, `samples` 4096, `levels` null, `random_seed` 1 | none |
| `rules.<processing rule>.parameters.registration.steps[] and recovery.alternatives[] (method demons)` | `_registration_step` | `DemonsConfig` | `variant` "demons", `iterations` [100, 50, 25], `smoothing_sigma` 1, `pyramid_mode` "antialias" | none |
| `rules.<processing rule>.parameters.registration.steps[] and recovery.alternatives[] (method tps)` | `_registration_step` | `TpsConfig` | `detection_noise_sigma` 3, `match_distance_voxels` 10, `min_matches` 50, `max_control_points` 1000, `smoothing` 1, `grid_spacing_voxels` 32, `interpolation_order` 3, `field_smoothing_sigma` null, `clamp_sampling_coordinates` true | none |
| `rules.<processing rule>.parameters.registration.steps[] and recovery.alternatives[] (method cpd)` | `_registration_step` | `CpdConfig` | `detection_noise_sigma` 5, `max_control_points` 1000, `kernel_width_voxels` null, `regularization_weight` 2, `outlier_fraction` 0.15, `affine_first` true, `grid_spacing_voxels` 16, `candidate_radius_voxels` 15, `neighbors_per_anchor` 3, `interpolation_order` 3, `field_smoothing_sigma` null, `clamp_sampling_coordinates` true | none |
| `rules.<processing rule>.parameters.registration.signal (recipe and step)` | `_explicit_registration / _typed` | `RegistrationSignalConfig` | `mode` "max", `reference_channel` null, `moving_channel` null | none |
| `rules.<processing rule>.parameters.registration.warp` | `_explicit_registration / _typed` | `WarpConfig` | `backend` "translation", `boundary_mode` "constant", `fill_value` 0, `output_dtype` "input", `integer_rounding` "nearest_even", `clip_to_dtype` true, `fft_workers` 1 | none |
| `rules.<processing rule>.parameters.registration.qc` | `_explicit_registration / _typed` | `RegistrationQcConfig` | `min_coverage` null, `min_ncc_gain` null, `max_fold_fraction` null, `max_translation_voxels` null, `projections` false | none |
| `rules.<processing rule>.parameters.spot_finding and channel_overrides.<label> (method local_maxima)` | `_detection` | `LocalMaximaConfig` | `threshold_mode` "noise", `threshold_value` 5.0, `min_distance_voxels` 1, `exclude_border` true, `channel_labels` null, `measure_peak_intensity` true, `merge_radius_zyx` null | none |
| `rules.<processing rule>.parameters.spot_finding and channel_overrides.<label> (method starfish_log)` | `_detection` | `StarfishLogConfig` | `min_sigma` (required), `max_sigma` (required), `num_sigma` (required), `threshold` (required), `overlap` 0.5, `exclude_border` false, `channel_labels` null | none |
| `rules.<processing rule>.parameters.spot_finding and channel_overrides.<label> (method spotiflow)` | `_detection` | `SpotiflowConfig` | `model` (required), `prob_thresh` null, `min_distance` 1, `exclude_border` false, `subpix` null, `n_tiles` null, `scale` 1.0, `channel_labels` null | none |
| `rules.<processing rule>.parameters.spot_finding and channel_overrides.<label> (method piscis)` | `_detection` | `PiscisConfig` | `model` (required), `threshold` 0.5, `min_distance` 1, `input_size` null, `scale` 1.0, `channel_labels` null | none |
| `rules.<processing rule>.parameters.reads_extraction.background` | `_extraction` | `LocalBackgroundConfig` | `inner_radius_zyx` [1, 3, 3], `outer_radius_zyx` [1, 6, 6], `min_voxels` 16 | the key absent: `inner_radius_zyx` and `outer_radius_zyx` grown by the same voxels along every axis where `voxel_size` exceeds `(1, 3, 3)` (`_extraction`, `:319-325`); a mapping: none |
| `rules.<processing rule>.parameters.load_codebook.encoding (method two_base)` | `_encoding` | `EncodingConfig` | `reverse_bases` true, `pair_to_color` {"AA": "1", "CC": "1", "GG": "1", "TT": "1", "AC": "2", "CA": "2", "GT": "2", "TG": "2", "AG": "3", "CT": "3", "GA": "3", "TC": "3", "AT": "4", "CG": "4", "GC": "4", "TA": "4"} | none; integer colours of `pair_to_color` read as strings (`_encoding`, `:280-282`) |
| `rules.<processing rule>.parameters.load_codebook.encoding (method one_base)` | `_encoding` | `OneBaseEncodingConfig` | `base_to_color` (required), `reverse_bases` false | none; integer colours of `base_to_color` read as strings (`_encoding`, `:280-282`) |
| `rules.<processing rule>.parameters.decoding (method wta)` | `_decoding` | `WtaDecoderConfig` | `negative_policy` "reject", `diagnostics` false | `diagnostics` `true` (`_decoding`, `:302`; without a `decoding` block `WtaDecoderConfig(diagnostics=True)`, `:570`) |
| `rules.<processing rule>.parameters.decoding (method codebook_aware)` | `_decoding` | `CodebookAwareDecoderConfig` | `max_hamming` 1, `max_corrected_round_margin` 0.2, `min_score_delta` 0.25, `min_geomean_probability` 0.45, `max_correction_penalty` 1.5, `allow_exact` true, `allow_rescue` true, `negative_policy` "reject", `diagnostics` false | `diagnostics` `true`, `allow_rescue` `false` (`_decoding`, `:302`) |
| `rules.<processing rule>.parameters.decoding (method direct)` | `_decoding` | `DirectDecoderConfig` | none | none |
| `rules.<processing rule>.parameters.deduplication (with run)` | `_deduplication` | `DeduplicationConfig` | `distance_voxels` 1.0, `compatibility` "same_sequence" | none |
| `rules.nuclei_registration.parameters.checkpoints` | `_nuclei_checkpoints` | `CheckpointConfig` | `stages` ["registered", "candidates", "pre_qc"], `directory` null, `table_format` "csv", `hash_inputs` true, `overwrite` false | none |
| `assignment.checkpoints` | `_assignment_block` | `CheckpointConfig` | `stages` ["registered", "candidates", "pre_qc"], `directory` null, `table_format` "csv", `hash_inputs` true, `overwrite` false | none |
| `segmentation.runs[] and export.segmentation.runs[] (method stardist)` | `_run_method` | `StarDistConfig` | `scale` (required), `model` null, `model_path` null, `model_sha256` null, `prob_thresh` null, `nms_thresh` null, `normalize_percentiles` [1.0, 99.8], `n_tiles` null, `block_size` null, `min_overlap` null, `context` null | none |
| `segmentation.runs[] and export.segmentation.runs[] (method cellpose)` | `_run_method` | `CellposeConfig` | `diameter` (required), `model` null, `model_path` null, `model_sha256` null, `do_3d` false, `anisotropy` null, `flow_threshold` 0.4, `cellprob_threshold` 0.0, `tile_overlap` 0.1, `bfloat16` true, `min_size` 15, `normalize_percentiles` [1.0, 99.0] | none |
| `segmentation.runs[] and export.segmentation.runs[] (method seeded_watershed)` | `_run_method` | `SeededWatershedConfig` | `sigma_um` 1.5, `threshold` "otsu", `compactness` 0.0, `connectivity` 1 | none |
| `segmentation.runs[] and export.segmentation.runs[] (method import)` | `_run_method` | `LabelImportConfig` | `path` (required), `geometry` null, `relabel` false | `target` the run's `target` (`_run_method`, `:1220`) |
| `segmentation.runs[].inputs[]` | `_input_channel` | `InputChannel` | `role` (required), `round` null, `channel` null, `reference_merged` false, `prepare_channel` null | none |
| `segmentation.runs[].inputs[].prepare (function composite_nuclei_amplicon)` | `_input_channel` | `CompositeConfig` | `nuclear_quantile` 0.005, `amplicon_quantile` 0.001 | none |
| `segmentation.runs[].inputs[].prepare (function enhance_with_flamingo)` | `_input_channel` | `FlamingoEnhancementConfig` | `flamingo_quantile` 0.005, `nuclear_quantile` 0.001, `median_radius_px` 1 | none |
| `segmentation.runs[].projection (a mapping)` | `_segmentation_run` | `ProjectionConfig` | `method` "max", `output_dtype` null, `conversion` null, `axis` "z" | none |
| `segmentation.runs[].operations[] (operation expand_labels)` | `_segmentation_run` | `ExpandLabelsConfig` | `distance` (required), `unit` (required), `mode` (required) | none |
| `segmentation.runs[].operations[] (operation extend_labels_through_z)` | `_segmentation_run` | `ZExtensionConfig` | `median_um` (required), `threshold` "otsu", `min_area_um2` (required), `dilation_um` (required), `fill_holes` (required) | none |
| `assignment (besides cells, nuclei, name, population, checkpoints)` | `_assignment_block` | `AssignmentConfig` | `expansion` null, `legacy_pixel_expansion` false, `correspondence` "CorrespondenceConfig(match_fraction=0.5, outside_tolerance=0.1)", `exclude_cells_without_nucleus` null | none |
| `assignment.expansion` | `_assignment_block` | `ExpandLabelsConfig` | `distance` (required), `unit` (required), `mode` (required) | none |
| `assignment.correspondence` | `_assignment_block` | `CorrespondenceConfig` | `match_fraction` 0.5, `outside_tolerance` 0.1 | none |
| `stitching.config (method multiview_stitcher)` | `_stitching_block` | `MultiviewStitcherConfig` | `binning_zyx` [4, 4, 4], `pairwise` "phase_correlation", `upsample_factor` null, `quality_threshold` 0.7, `drop_nan_quality` true, `pruning` "alternating_pattern", `overlap_tolerance_um` 0.0, `groupwise` "global_optimization", `abs_tol_um` null, `rel_tol` null, `max_iter` null | none |
| `stitching.config (method bigstitcher)` | `_stitching_block` | `BigStitcherConfig` | `overlap_percent_xyz` [10, 10, 10], `exclude` 10, `subsampling_factors` [[2, 2, 2], [4, 4, 4]], `hdf5_chunk_sizes` [[16, 16, 16], [16, 16, 16]], `timepoints_per_partition` 1, `setups_per_partition` 0, `deflate_compression` true, `downsample_xyz` [4, 4, 4], `min_r` 0.7, `max_r` 1, `max_shift_xyz` [0, 0, 0], `max_displacement` 0, `relative` 2.5, `absolute` 3.5, `pad_missing` false | none |
| `stitching.review` | `_stitching_block` | `ReviewPolicy` | `max_correction_fraction` 0.5, `max_residual_um` 1.0 | none |
| `assembly (besides assignment, table_format, previews)` | `_assembly_block` | `AssemblyConfig` | `ownership` "nearest_center", `centroid` "territory", `tie` "table_order", `conflicts` true, `duplicate_threshold` 0.5 | none |
| `export (the flags; route and segment included)` | `_export_config` | `ExportConfig` | `assemble` true, `fuse` true, `fuse_labels` false, `export_spatialdata` false, `route` null, `segment` true, `segmentation` null | none |
| `export (the fusion fields)` | `_export_config` | `FusionConfig` | `images` ["reference_stain"], `dimensionality` "projection", `projection` "max", `scale_yx` 0.5, `ome_zarr` false, `ome_zarr_version` "0.5", `pyramid_levels` 0 | none |
| `export (labels_<field>)` | `_export_config` | `LabelConfig` | `scale_yx` 0.25 | none |
| `export (spatialdata_<field>)` | `_export_config` | `SpatialDataExportConfig` | `zarr_format` 3 | none |
| `report` | `_report_block` | `ReportConfig` | `markers` [], `marker_pairs` [], `positive_threshold` 1, `example_fovs` null, `limits` "ReportLimits(shift_residual=RuleLimit(k=5.0, floor=5.0, absolute=None), alignment_score=RuleLimit(k=None, floor=None, absolute=None), correct_form_fraction=RuleLimit(k=3.0, floor=0.05, absolute=None), matched_fraction=RuleLimit(k=3.0, floor=0.05, absolute=None), spots_near_zero=0.05)", `figure_width_px` 1600, `jpeg_quality` 85 | none |
| `report.markers[]` | `_report_block` | `MarkerGroup` | `name` (required), `genes` (required) | none |
| `report.limits` | `_report_block` | `ReportLimits` | `shift_residual` "RuleLimit(k=5.0, floor=5.0, absolute=None)", `alignment_score` "RuleLimit(k=None, floor=None, absolute=None)", `correct_form_fraction` "RuleLimit(k=3.0, floor=0.05, absolute=None)", `matched_fraction` "RuleLimit(k=3.0, floor=0.05, absolute=None)", `spots_near_zero` 0.05 | none |
| `report.limits.<rule>` | `_report_block` | `RuleLimit` | `k` null, `floor` null, `absolute` null | none |

## The MATLAB legacy surface frozen by D7

Decision D7 keeps the MATLAB workflow's rules, scripts, keys and files unchanged and moves the
adapters it needs, unchanged, into one private legacy module that the Python workflow never
imports, pinned by `test_assignment_workflow.py` and `test_workflow_assembly_golden.py`.

**Rules.** The 17 rows of backend `matlab` of the [rule inventory](#rule-inventory) and the five
rows of backend `both` (`all`, `enhance_dapi_with_flamingo`, `stardist_segmentation`,
`create_nuclei_amplicon_overlay`, `reads_assignment`), as `rules-matlab.txt` lists them.

**Scripts.** `rsf_single_fov.m`, `gr_single_fov_subtile.m`, `lrsf_single_fov_subtile.m`,
`deep_create_subtile.m`, `deep_rsf_subtile.m`, `rsf_single_fov_seq.m`, `nuclei_registration.m`;
the Python scripts of the MATLAB rules: `stitch_subtile.py` (shared with the Python subtile route),
`stitching_preparation.py`, `create_BigStitcher_macro_1.py`, `create_tile_config.py`,
`create_sample_maf.py`, `create_sample_h5ad.py`, `create_sample_reads_assignment.py`; the adapters
of the shared rules (`create_nuclei_amplicon_overlay.py`, `enhance_dapi_with_flamingo.py`,
`stardist_segmentation.py`, `reads_assignment.py`); `workflow/rules/matlab_launcher.py`. No rule
calls `create_BigStitcher_macro.py` (an N5 variant), `cleanup_subtile.py`,
`create_segmentation_preview.py`, the files of `workflow/scripts/test/` or
`workflow/notebook/create_rsf_qc_report.py.ipynb`.

**Keys.** The MATLAB scripts read, from the JSON copy: `starfinder_path`, `root_input_path`,
`dataset_id`, `sample_id`, `root_output_path`, `output_id`, `ref_round`, `ref_channel`,
`rotate_angle`, `maximum_projection`, `seq_channel_order` (the `channel`/`name` struct form),
`additional_round` (`round_name`, `channel_order` by name) and, under `rules.<rule>.parameters`,
the blocks `enhance_contrast`, `hist_equalize` (`reference_channel`, one-based), `morph_recon`
(`radius`), `global_registration` (`run`, `ref_round`, `ref_img`, `mov_img`), `local_registration`
(`run`, `ref_round`), `create_subtiles` (`run`, `sqrt_pieces`), `spot_finding` (`run`, `ref_round`,
`intensity_threshold`, and in the deep script `intensity_estimation`), `load_codebook` (`run`,
`split_index`, one-based), `reads_extraction` (`run`, `voxel_size` as row, column, Z),
`reads_filtration` (`run`, `end_base`, `n_barcode_segments`, `split_index`) and, for
`rsf_single_fov_seq`, `nuclei_registration` (`run`, `layers_to_register`, `ref_dapi_path`,
`ref_dapi_pattern`, `mov_dapi_path`, `mov_dapi_pattern`). The rule files and `common.smk` read
`config_path`, `fiji_path` (when parsed, `stitching.smk:56`), `dapi_round`, `fov_id_pattern`,
`n_fovs`, `subset_list`, `subset_range`, `subset_start`, `subset_end`, `subset_random`,
`n_random_tests`, `workflow_mode`, `rules.<rule>.run` and `.resources`, `img_col`, `img_row`,
`voxel_size_xy`, `voxel_size_z`, `matlab_launcher` and `matlab_single_thread`. The shared rules'
adapters read a MATLAB-backend configuration through `dataset/_legacy_config.py`
(`legacy_processing`, `:125`; `matlab_reference_stains`, `:176`).

% results/listings: the MATLAB keys from `grep -o "config\.[A-Za-z_.]*" workflow/scripts/*.m`

**Files and conventions.** MATLAB reads the JSON copy of the configuration (`jsondecode`), the
round-first images `INPUT/<round>/<fov>/*ch0N.tif` (default channel order `ch00, ch02, ch01, ch03`),
the headerless `INPUT/genes.csv` (`gene,barcode`), `documents/sample-annotation.csv`
(`sample_id, fov_start, fov_end`, one-based inclusive FOV indices named by `fov_id_pattern`) and
`documents/raw.maf` (Leica stage points by `PositionID`). It writes the console logs
`log/<fov>_rsf.txt`, `_gr.txt`, `_nr.txt`, the score logs `log/sf_scores/<fov>[_<n>].txt`, the
shifts `log/gr_shifts/<fov>_nr.txt`, `images/ref_merged/<fov>.tif`, the molecule tables
`signal/<fov>_goodSpots.csv` and `signal/<fov>_allSpots.csv` (one-based `x` column, `y` row, `z`,
and `gene`), the subtile files `output/subtile/<fov>/subtile_coords.csv` (one-based subtile
corners), `subtile_data_<n>.mat` and `subtile_goodSpots_<n>.csv` (one-based within the subtile), and
`images/<round>/<channel name>/<fov>.tif` (min–max stretched). Its sample route writes
`documents/maf/<sample>.maf`, `images/fused/<sample>/` (`blank.tif`, `grid.csv` with `col, row, id,
grid`, `grid.png`, `DAPI/tile_<j>.tif`, `BigStitcher_macro.ijm`, `DAPI/dataset.xml`),
`output/tile_config_<sample>.csv` (integer offsets `x, y, z`, `fov_index`, the non-overlap box
`start/end_x/y` and its `_norm` form) and `.html`; the shared `reads_assignment` writes, without an
`assembly` block, `expr/<fov>/raw.h5ad` (`obs` `sample`, `fov_id`, `volume`, `fov_x/y/z`,
`seg_label`, `global_x/y/z` and the cell-table columns), `expr/<fov>/reads_assignment.csv` (the
legacy columns zero-based after subtracting one), `assignment.png`,
`assignment_single_layer.png` and `log.txt`, and the aggregation rules
`expr/<sample>_raw.h5ad` and `expr/<sample>_reads_assignment.csv`.

**The package code its shared rules call.** The segmentation chain: `_run_nuclei_amplicon_overlay`
(with `_read_stack`), `_run_enhance_dapi_with_flamingo`, `_run_stardist_segmentation`, and through
them `starfinder.segmentation` (`composite_nuclei_amplicon`, `enhance_with_flamingo`, `segment`) and
`dataset.workflow._segmentation`, `_declared_metadata`, `_label_namespace`. `reads_assignment`:
`_run_reads_assignment` with `_fov_sample` (its annotation branch), `_tile_record`, `_legacy_cells`,
`_write_raw_h5ad`, `_cell_obs`, `_in_box`, `_write_reads_assignment`, `_assignment_ratio`,
`_write_assignment_figures`, and `dataset.workflow._reads_assignment`, `_legacy_target`,
`_legacy_molecules` (true for MATLAB), `_reference_genes` (through `from_workflow_config`'s MATLAB
branch, `dataset/_legacy_config.py`), and `starfinder.assignment` (`assign_molecules`,
`import_legacy_molecules_csv`, `write_h5ad`, `plot_assignment`). `stitch_subtile.py` imports no
package code.

## Discrepancies

Every discrepancy found between the documents, the schema and the code at `56c35b8`, with a
proposed resolution and the decision it follows from or the open choice it raises (OC1 to OC6,
proposed in the [open points](#open-points-for-the-specification)). O1 to O7 are the known defects
listed on 2026-09-28, as they stand at this revision.

(orchestration-discrepancies)=

| # | Discrepancy | Evidence | Proposed resolution | Follows from |
| --- | --- | --- | --- | --- |
| O1 | `gcloud-backup.smk` is a separate backup workflow inside `workflow/rules/`: not included by the Snakefile, it opens a Google Cloud Storage client and lists a bucket when parsed, hard-codes `PROCESSED_DIR` and overwrites its own list of files (`:14-25`), and its `upload_to_gcs` declares the same files as input and output (`:63-67`) | `workflow/rules/gcloud-backup.smk`; not in either listing | Remove it from `workflow/`; if the lab needs it, it lives with the lab's data tools, outside both workflows | OC1 (no decision covers it; D1 defines the workflows' coverage) |
| O2 | `stitch_subtile` reads the FOV's reference image for its preview: declared on the Python backend since the reference-image work (`spot-finding-py.smk:54`, `stitch_subtile.py:48-49`), undeclared on the MATLAB backend (`spot-finding.smk:96-107`, `stitch_subtile.py:50-51`) | the rule inventory | Python: the route is retired; MATLAB: kept as it is and stated in the legacy record | D5; D7 |
| O3 | `subset_random` draws the FOVs with an unseeded `random.sample` | `common.smk:278-279` | Kept for MATLAB and documented; the Python dialect has no random selection (`selected_fovs`) | D7 |
| O4 | `tests/minimal_config.yaml` (MATLAB dialect, no `backend`, placeholder paths) passes the schema but cannot be parsed: it has no `additional_round`, which `get_additional_round_names` indexes (`common.smk:184-185`, from `registration.smk:60`); it also needs a sample annotation. `design/specs/workflows.md:327-330` calls it a historical template and does not say it fails | `rules-matlab-tests-minimal_config.txt` (`KeyError: 'additional_round'`); `test_workflow_config_validation.py:137-150` uses it as a schema input only | Keep it unchanged as the schema fixture of the frozen MATLAB dialect and say in the records that it does not parse; the Python templates are `docs/examples/` | D7; OC2 |
| O5 | The processing rules write `run.json` alone by default: `_rule_checkpoints` gives `{stages: []}` | `dataset/workflow.py:767-785` | `candidates` and `pre_qc` on by default, `registered` off unless a data dependency needs it; the golden configuration states its stages explicitly, so the refactored default reproduces it | D8 |
| O6 | The default backend is `matlab` | `common.smk:21`, `config.schema.yaml:219-223`, `workflow.py:533,819,1045,1080`, `_snakemake.py:238`, `stitch_subtile.py:48`; `design/specs/workflows.md:179` | Default `python`; a configuration without `backend` that holds MATLAB-dialect keys fails, asking for `backend: matlab` | D7 |
| O7 | `envs_path` is a schema key that no rule reads; conda environment files remain | `config.schema.yaml:27-29`; `config/stardist.yml`, `workflow/envs/stardist.yml`, `workflow/envs/starfinder.yml`, `config/environment*.y*ml`; the profile's `software-deployment-method: conda` | Remove `envs_path` from the Python dialect (the migration table says so); retire the stale environment files | D6; D9 |
| O8 | The Python `nuclei_registration` writes its channel images without declaring them, and `enhance_dapi_with_flamingo` reads exactly those files, so `all` with that rule enabled fails before any job (`MissingInputException`) unless an earlier run wrote them | `registration-py.smk:78-86`, `_snakemake.py:127-138`, `_snakemake.py:654-676`; the dry run of the wrapped step `behaviour_evidence.rerun-1` (`results/evidence/o8-dryrun.log`); `segmentation.md:1303` assigns it to workflow orchestration | The morphology task declares every output; `enhance_dapi_with_flamingo` is retired for Python (`prepare: enhance_with_flamingo`) | D6 |
| O9 | `workflow-downstream.md:33` gives `enhance_dapi_with_flamingo` the inputs `images/flamingo/DAPI/` and `images/flamingo/Flamingo/`, "supplied separately"; on the Python backend they are the channel-label folders that `nuclei_registration` writes | `_snakemake.py:654-676` | The contract record's changed statements correct it; the rule is retired for Python | D6 |
| O10 | `workflow-downstream.md:30` says both backends write `images/<round>/<channel name>/`; the Python rule names the folder by channel label, as `workflows.md` ("Nuclei registration") says | `_snakemake.py:127-131` | The contract record's changed statements correct it | D6; D7 |
| O11 | `workflows.md:89` says the Python rule rotates "if angle is nonzero"; the adapter passes `rotate_angle: 0` as a rotation (`workflow.py:582`), and `FOV.run` rotates whenever the angle is not None (`dataset/fov.py:371-372`): `run.json` records a `rotate_round` step and every frame id gains `/rotate:0`, so a label file that stores the scene's own frame is refused by `import_labels` | the golden `run.json` records (`results/golden/frame_ids.json`); the run of the wrapped step `behaviour_evidence.rerun-1` with the scene's label file (`results/evidence/o11-run.log`) | The refactor keeps the behaviour, which the golden baseline pins; the statement is corrected; whether 0 means no rotation in the new shape is OC3 | D4; OC3 |
| O12 | An image loaded from TIFFs without stored metadata takes an absolute input path as frame id (for the Python `nuclei_registration` images, the reference round's FOV folder, with `/rotate:0`), and its grid stores no spacing although `voxel_size_xy` and `voxel_size_z` are given; the bytes of those images therefore depend on where the input lies | the stored descriptions of `images/flamingo/ch04/FOV_001.tif` and `ch05` of both golden runs (`results/golden/frame_ids.json`) | The plan records input paths relative to the input root, as `run.json` version 2 does; OC4 | D2; D8; OC4 |
| O13 | `workflows.md:312-315` says a one-FOV direct dry run schedules config preparation, `rsf_single_fov` and `all`, and a 2 × 2 subtile run preparation too; the Python backend has no `rsf_preparation` (`common.smk:391-398`), so it schedules 2 and 7 jobs; `workflows.md:247-249` gives the same MATLAB reason for the target `all` | W-641's dry runs (`gp099/dryrun/`); `dryrun-python-golden-a.log` | The contract record's changed statements make them backend-specific | D7 |
| O14 | `codebook`, `direct_panel` and `image_dir_template` are Python-only in the code (`common.smk:231-235`, `workflow.py:897,919`) but not in the schema's Python-only clause, which accepts them with `backend: matlab` | `config.schema.yaml:524-540` | The schema refuses them without `backend: python`, and the parse-time refusal leaves `common.smk`; OC5 | D2; D7; OC5 |
| O15 | Python rules take their resources from the sections of MATLAB rule names that the Python dialect refuses to enable: `bigstitcher_macro_sample` from `create_BigStitcher_macro`, `bigstitcher_engine` from `run_BigStitcher_macro`, `stitch_sample` from `create_tile_config` or `stitching_preparation`, `export_sample` from `create_sample_h5ad`; `report_sample` and `report_dataset` have fixed resources | `stitching-py.smk:46,62-63,83,99`, `export.smk:36`, `report.smk:30,42`; `_check_sample_rules` (`workflow.py:1578-1602`) | Each task's resources come from the execution section | D4 |
| O16 | Defaults restated in `.smk` files: `stitching.method` (`stitching-py.smk:12`), `image_dir_template` (`registration-py.smk:93`), `segmentation_input_folder` (`common.smk:203`), `create_subtiles.sqrt_pieces` (`common.smk:259-262`), the resources (`common.smk:156-159`) | the rule files beside `workflow.py:1431,896,1066,597` | Defaults live in Python only; the plan resolves them into the resolved configuration | D2; D4 |
| O17 | Validation in `.smk` files at parse time: the backend (`common.smk:22-23`), the Python-only input keys (`:231-235`), the presets (`:57-92`), `stitching-py.smk:26-28` | the classification table | The plan validates every block before fan-out | D2 |
| O18 | The producer of a FOV's molecule table is chosen twice: by the `ruleorder` (`Snakefile:63-74`) and by `_legacy_molecules` (`workflow.py:1070-1087`) | the two definitions | One derivation, in the plan; the second producer is retired for Python | D2; D4; D5 |
| O19 | `reads_assignment` without an `assembly` block declares every sample's tile configuration for each FOV (`reads-assignment.smk:26-27`); `create_tile_config` declares every sample's inputs and outputs (`stitching.smk:62-66`) | the rule inventory | Python: the legacy route is retired; MATLAB: kept | D6; D7 |
| O20 | `FUSE_DIR` is defined and never read | `common.smk:202`; `callers.json` | Removed with the planners of `common.smk` | D2 |
| O21 | Scripts that no rule calls stay in `workflow/scripts/`: `create_segmentation_preview.py`, `cleanup_subtile.py`, `create_BigStitcher_macro.py`, `workflow/scripts/test/`, and `workflow/notebook/` | the rule inventory | `create_segmentation_preview.py` is retired; the others are kept unchanged beside the frozen MATLAB workflow until the legacy module issue decides; OC6 | D6; D7; OC6 |
| O22 | The UGER profile deploys conda environments while no rule declares one, retries every job three times with the runtime multiplied by the attempt for the subtile rules, and names a job log only for MATLAB's `gr_single_fov_subtile` (`params.uger_log`) | `profile/broad-uger/config.yaml`, `broad-submit.py:62-66`, `registration.smk:43-44` | The profile of the new workflow runs the image of the lock, keeps retries as the engine's and names one log per task | D8; D9 |
| O23 | The inline `rotate_nuclei` rotates the stain with another method than the package rotates the sequencing rounds (spline order 3 with `reshape`, against `np.rot90` or bilinear without `reshape`); for an angle that is not a multiple of 90° the DAPI image and the reference image differ in shape | `registration-py.smk:110-122`, `preprocessing/_rotation.py:12-23` | The morphology task rotates with the package; MATLAB's rule is frozen | D6; D7 |

## The golden baseline

The current workflow with `backend: python`, run by Snakemake on a synthetic fixture, and the
explicit expected values of its outputs, which the refactored workflow must reproduce (D10).

**Where it lands.** The batch's locked environment has no Snakemake, so no pytest can run the
workflow there. The expected values and the recording script are committed:
`src/python/test/data/expected/test_workflow_orchestration_golden/` (with its `README.md`) and
`src/python/scripts/workflow_golden.py` (`fixture`, `run`, `extract`, `compare`). The comparison
test is a draft for task group 2, to land with the lock change that puts Snakemake 9 in a `workflow`
dependency group; the worker notes hold the draft. No test reads the expected values yet.

**Environment.** Host GP099-29C (Intel Xeon Gold 6234), one CPU (`taskset -c 9`), one numerical
thread, no GPU. The fixture is generated and the outputs are read in the locked environment
`dev-s210-export-20261007T050852Z` (Python 3.12.12, NumPy 2.2.6, pandas 3.0.0, tifffile 2026.1.28).
The workflow runs with Snakemake 9.15.0 of `~/miniforge3/envs/starfinder-v9`, outside the lock and
as installed (Python 3.12.12, NumPy 2.3.5, pandas 2.3.3, SciPy 1.17.0, scikit-image 0.26.0,
tifffile 2026.1.14, anndata 0.12.7, `snakemake-executor-plugin-cluster-generic` 1.0.9; its installed
distributions equal W-641's freeze), whose rule scripts import Starfinder from the checkout at
`56c35b8` that `starfinder_path` names. That environment has neither `multiview-stitcher`,
`spatialdata`, `pyarrow` nor `SimpleITK`, nor a learned backend.

% results/environment.json; results/environment/starfinder-v9-distributions.txt

**Fixture and seeds.** `sample_scene_preset("sample-tiny-v1")` with its seed 42: 2 × 2 FOVs
(`FOV_001` to `FOV_004`) of 5 × 36 × 36 voxels at 1.0 × 0.5 × 0.5 µm, four rounds of four channels
(`ch00` to `ch03`) written as one OME-TIFF per round, FOV and channel; the nuclear stain as the
reference round's `ch04.tif` and as `ch04.tif` of an additional round `flamingo`, whose `ch05.tif`
is the reference round's channel maximum (these three as uint16 TIFFs without stored metadata, as a
microscope writes them); `genes.csv` (the preset's 16 genes as `two_base` barcodes, checked to load
as the preset's codebook), `fovs.csv` (one sample) and `positions.csv`; the truth cell labels as
plain ZYX TIFFs without stored metadata, the form of a legacy label file. The workflow itself draws
no random number: translation registration by FFT on one worker, local maxima, and Snakemake with
`--cores 1`.

**The run.** Four invocations in one output root (the commands are in
`results/golden/run1/commands.json`):

* A, `config-a.yaml`, target `all`: `rsf_single_fov` (min–max normalization to uint8 and a white
  top-hat, one translation step on the channel maximum, local maxima, extraction with the local
  background, WTA decoding, scoring, deduplication and a filter with end bases `CC`, checkpoint
  stages `candidates` and `pre_qc`), `rotate_nuclei`, `nuclei_registration`,
  `create_nuclei_amplicon_overlay`, `report_sample` and `report_dataset` (the `report` block; the
  report reads the FOVs' `run.json`): 19 jobs;
* A2, the same configuration, target the files of `enhance_dapi_with_flamingo` (4 jobs), whose
  inputs exist only once A has run (O8);
* B, `config-b.yaml` (A with a `stitching` block of method `multiview_stitcher`, an `assembly`
  block with the assignment `default` and `export.route: per_fov`, without which the rule files
  refuse the configuration when parsed, naming `export.segmentation`), the truth labels copied first as
  `images/stardist_segmentation/<fov>.tif`; target each FOV's
  `fovs/<fov>/assignment/default/assignment.json` with `--allowed-rules` that leave out
  `stardist_segmentation`: `reads_assignment` imports the labels and assigns (4 jobs);
* C, `config-c.yaml` (A with a `stitching` block of method `bigstitcher` on a 2 × 2 regular grid),
  target the macro and view map with `bigstitcher_macro_sample` the one rule allowed (1 job).

**Not covered, with the reason.** `stitch_sample` with `multiview_stitcher`: the Snakemake
environment does not have the `multiview-stitcher` extra, and nothing may be installed into it;
`bigstitcher_engine` and `stitch_sample` of method `bigstitcher`: they run Fiji, which this batch
does not run; `export_sample`, and `report_sample` and `report_dataset` with an `assembly` or
`stitching` block: they read the placement; `stardist_segmentation`: a learned method; the
legacy route of `reads_assignment` (without an `assembly` block): it reads the tile configuration
of the BigStitcher import; the `subtile` and `deep` modes of the Python backend: decision D5
retires them, so the refactor has no output of theirs to reproduce; every MATLAB rule: MATLAB is
not executed. The stitching, export and report-after-export part of the golden baseline can be
recorded with the same script once Snakemake runs in the locked environment (open point P1).

**The values.** Following `docs/contributing.md` ("Expected values and tolerances"): the
`run.json`, `nuclei_registration.json`, `assignment.json` and `report.json` records without the
fields of where, when and by which software they were made and their measured times and memory
(`started_at`, `ended_at`, `generated_at`, `code`, `environment`, `software`, `input_root`,
`output_root`, `workflow_config`, `seconds`, `wall_s`, `cpu_s`, `peak_rss_kib`, the digests of
records and files that hold them), floats within `rtol=1e-9`, `atol=1e-12` and every other leaf
exactly; the molecule tables, the assignment tables (`cells.csv`, `counts.csv`, `molecules.csv`),
the report's tables (its time columns left out, its intensity histograms as one row per
histogram) and the view map, integer, string and boolean columns exactly and computed float64
columns within the same tolerance; each FOV's reference image exactly (an integer image); the
shape, dtype, minimum, maximum and sum of the morphology images and the BigStitcher tiles exactly
(all integer images); the macro's text exactly with the work folder written `<work>`; the fixture's
file digests exactly; and the SHA-256 of every output file that holds no time or absolute path, an
identity that holds in the recording environment only.

**Two runs.** The whole recording (fixture, the four invocations, the extraction) ran twice in the
same environment, each in a new folder: 101 output files each, about 58 s of wall time and a peak
RSS of about 200 MiB per run. The second run reproduces every value of the first, the
same-environment digests included; a negative control that changes four values, each by more than
its tolerance, is reported four times and nothing else.

% results/golden/run1/, run2/ (fixture.json, configs, commands.json, logs, outputs.csv, jobs.json, expected/),
% the route refusal and O8, O11: results/evidence/ (step behaviour_evidence.rerun-1);
% results/golden/comparison.json; the resource table results/tables/resources.csv

## Open points for the specification

What this page needs from the contract, tiled and engine records, and the choices it raises, each
proposed in the worker notes with options and one recommendation:

* **P1.** The golden values of stitching with `multiview_stitcher`, export and the report after the
  export, recorded at this revision once Snakemake runs in the locked environment (the lock change
  of task group 2).
* **P2.** The comparison test of the golden baseline: its marker tier and where it runs (the
  engine contract's validation design).
* **P3.** How `rotate_angle: 0` and the frame ids it changes map to the new configuration shape
  (OC3).
* **P4.** How the manifest names input files and frames: relative to the input root, or absolute
  (OC4).
* **P5.** Where the assignment and ownership previews go: an option of the assign and export
  commands, or a separate preview task.
* **OC1** to **OC6**, as the [discrepancies](#discrepancies) raise them: the backup workflow, the
  MATLAB minimal template, rotation by 0, input paths in frames, the schema's Python-only keys, the
  unused scripts.
