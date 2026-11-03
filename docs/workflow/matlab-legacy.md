# MATLAB legacy workflow

:::{container} sf-lead
The MATLAB workflow is frozen: its rules, scripts, launcher and configuration keys
stay as they were, and it runs under Snakemake only. New runs use the
{doc}`Python workflow <index>`.
:::

## Running it

A configuration with `backend: matlab` selects it. `workflow/Snakefile` then
includes the MATLAB rule set (`workflow/rules/legacy.smk`), which validates the
configuration against `workflow/schemas/config.schema.yaml` and runs the rules
that its `workflow_mode` and `rules.<rule>.run` flags select. `starfinder workflow
plan` refuses `backend: matlab`, and so does Nextflow, which runs the plan first.
A configuration without `backend` that holds a key of this dialect (`rules`,
`workflow_mode`, the subset keys and others) is refused by the plan, asking for
`backend: matlab`: the default backend is `python`.

The FOVs of a sample come from `<OUTPUT>/documents/sample-annotation.csv`, with
the columns `sample_id,fov_start,fov_end` (inclusive FOV indices, named by
`fov_id_pattern`), and the subset keys select among them. The membership file
must exist before Snakemake parses the workflow. The modes are `free` (each
rule's `run` flag), `direct` (`rsf_single_fov`), `subtile` (global registration,
then subtiles and their stitch) and `deep` (the deep-tissue subtile path).

## MATLAB launcher

`matlab_launcher: path`, the default, runs the executable named by
`STARFINDER_MATLAB_EXECUTABLE`, or else `matlab` on `PATH`; `broad` sources the
Broad `useuse` script and loads `use Matlab` before running `matlab`. MATLAB runs
in `-batch` mode, so an error in a script fails its job;
`matlab_single_thread: true` adds `-singleCompThread`. MATLAB R2023b or newer with
the Image Processing Toolbox is needed (see {doc}`../get-started/installation`).

## Environment and profile

The MATLAB rules keep the conda environment of `workflow/envs/matlab-legacy.yaml`
(named `starfinder-v9`); the segmentation and assignment rules call the
Starfinder package, so that environment needs the package and the extras those
rules use. On Broad UGER, `profile/broad-uger-matlab/` keeps conda deployment,
`notemp`, three retries and Broad's module loading, and its job script activates
`/stanley/WangLab/envs/starfinder-v9` and then runs `use Matlab`. Under that
profile keep `matlab_launcher: path`, the default: `matlab` is then on `PATH`.
`broad` repeats `use Matlab` in a shell that inherits the job script's, where it
returns 1, so the rule exits 1 before MATLAB starts; `broad` is for a job that
has not loaded MATLAB.

## Downstream rules

The MATLAB rule set keeps its downstream rules: `nuclei_registration`,
`rotate_nuclei`, `create_nuclei_amplicon_overlay`, `enhance_dapi_with_flamingo`,
`stardist_segmentation` and `reads_assignment`, the legacy stitching rules
(`stitching_preparation`, `create_BigStitcher_macro`, `run_BigStitcher_macro`,
`create_tile_config`) and `create_sample_h5ad` and
`create_sample_reads_assignment`. MATLAB direct jobs and the subtile stitch write
`signal/<fov>_goodSpots.csv`, with one-based `x,y,z,gene`; `reads_assignment`
writes `expr/<fov>/raw.h5ad` and `reads_assignment.csv`.

## Configuration keys

The tables are generated from `workflow/schemas/config.schema.yaml`, which the
MATLAB rule set validates against. A dash in "Schema default" means that the
schema declares none; the rules may apply one. Named types refer to the
definitions below the first table. The schema also lists the keys that only the
Python backend accepted before the Python configuration shape; it refuses them
with `backend: matlab`.

```{workflow-config}
```

(matlab-schema-definitions)=
### Schema definitions

```{workflow-config} #/$defs
```

## Records

The [workflow record](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/workflows.md),
the [configuration record](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/workflow-configuration.md)
and the [downstream record](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/workflow-downstream.md)
describe the MATLAB rules, their files and their keys; the
[engine record](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/orchestration-engines.md#the-matlab-legacy-separation)
gives the separation.
