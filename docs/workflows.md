# Snakemake workflow

(workflows-and-configuration)=

The workflow schedules STARfinder over FOVs and samples. A direct-mode job can
load images, preprocess, register, find spots and read molecules; later jobs can
segment, assign, stitch, export and report. Use the {doc}`Python API <coordination>`
for an interactive pipeline and Snakemake for a file-based run.

## Quick start

(small-reproducible-dry-run)=
### Small, reproducible dry run

Install Snakemake as {doc}`get-started/installation` describes. From `src/python`,
use a new absolute directory outside the checkout for `RUN`:

```bash
REPO=$(cd ../.. && pwd)
RUN=/absolute/external/new-workflow-check
uv run python ../../docs/examples/prepare_workflow.py "$RUN"
export XDG_CACHE_HOME="$RUN/cache"
uv run python -m snakemake \
  -s "$REPO/workflow/Snakefile" --configfile "$RUN/config.yaml" \
  --directory "$RUN" --cores 1 -n all
```

The helper prepares the bundled minimal configuration, a codebook row, `fovs.csv`
and empty round/FOV directories. It creates no TIFFs. Keep `-n`: this example
checks the job graph, not image processing. The
{doc}`single-FOV demo <get-started/single-fov-demo>` processes small images.

For a real run, edit a copy of the
{download}`minimal configuration <examples/workflow-minimal.yaml>`, provide your
images and membership file, and inspect a dry run of `all` before execution.

(minimal-configuration)=
The minimal YAML and the
{download}`preparation helper <examples/prepare_workflow.py>` are downloadable.
The {doc}`configuration reference <workflow-configuration>` lists the keys.

## What it does

(paths-and-input-contract)=
### Paths and inputs

`INPUT` is `root_input_path/dataset_id/sample_id` and `OUTPUT` is
`root_output_path/dataset_id/output_id`.

The Python backend reads membership from `INPUT/fovs.csv`, with `sample,fov_id`
columns; `position_id` associates FOVs with stage-file positions. Supply this file
before running Snakemake. The MATLAB backend uses
`OUTPUT/documents/sample-annotation.csv`, with `sample_id,fov_start,fov_end`.

Sequencing images normally live under `INPUT/<round>/<fov>/`. Set
`image_dir_template` to use a FOV-first layout with the Python backend. The
codebook is normally `INPUT/genes.csv`; `codebook` selects another multiplexed
codebook and `direct_panel` selects a direct-readout panel with that backend.

(stages-and-files)=
### Stages and files

1. A processing job reads a FOV's sequencing rounds and writes its molecules.
2. Morphology jobs prepare images for segmentation; {doc}`workflow-downstream`
   explains segmentation and molecule assignment.
3. Sample jobs can place FOVs, assemble tables, export images and write reports.

(modes-and-backend-selection)=
### Modes and backends

Choose `direct` for a job per FOV, `subtile` for global registration followed by
subtile processing, or `deep` for the deep-tissue subtile path. `free` uses the
individual rule flags. The preparation helper accepts `--template full --mode
subtile` and `--template full --mode deep` for dry runs in fresh directories.

`backend` selects the Python or MATLAB sequencing rules. The
[mode specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/workflows.md#modes-and-backend-selection)
gives the rule-selection details.

(broad-uger-execution)=
The {doc}`UGER guide <workflow/broad-uger>`
describes cluster submission.

## What you get

(molecule-tables)=
### Molecule tables

The Python direct job writes `fovs/<fov>/molecules.csv` under OUTPUT, with spot
identity, zero-based ZYX position, gene, entry and quality columns. MATLAB direct
jobs and the subtile stitch job write `signal/<fov>_goodSpots.csv`, with one-based
`x,y,z,gene`. The downstream adapters read the corresponding table.

(projection-views-and-the-reference-merged-image)=
### Reference images

The Python processing record is `fovs/<fov>/run.json`. Its saved reference image
is configured by `reference_image`; an imported image is configured by
`reference_image_import`. MATLAB writes `images/ref_merged/<fov>.tif` and console
logs under `log/`. See the
[reference-image specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/workflows.md#projection-views-and-the-reference-merged-image)
for the image's content and projection settings.

(nuclei-registration)=
### Nuclei registration

`nuclei_registration` prepares additional morphology rounds. With the Python
backend it calls `FOV.prepare_morphology` and writes
`fovs/<fov>/nuclei_registration.json` and registered channel images. The
[configuration reference](workflow-configuration.md#segmentation-and-assignment)
links the downstream setup.

## Main options

* `selected_fovs`: choose the Python run's FOVs.
* `rules`: choose jobs and their processing parameters.
* `stitching`: place a sample's FOVs.
* `assembly` and `export`: choose the sample outputs.
* `report`: request processing reports.

## Limits

A dry run checks scheduling without processing images. It does not check model
availability, image content or scientific performance. MATLAB execution needs
MATLAB and its toolboxes; learned segmentation needs its method's dependencies.

## How it is checked

Configuration validation and a local dry run help find setup problems before
execution. Inspect the scheduled inputs and outputs, then use a small real run
to check your data and compute environment.

The [workflow history](https://github.com/wanglab-broad/starfinder/blob/dev/design/notes/workflow-history.md) records the adapter boundary.

[Workflow specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/workflows.md).
