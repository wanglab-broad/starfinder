# Preparing a dataset

:::{container} sf-lead
Get a new dataset ready for `starfinder workflow plan`: the input folder, the FOV
list, the codebook and the configuration. The workflow reads a raw dataset folder
where it is; nothing is copied, linked or written into it.
:::

## Quick start

tissue-2D's configuration is `starfinder.yaml` in its folder, beside the FOV
list `fovs.csv` it names. Append the run's roots and output id to a copy, and
plan:

```bash
cat /stanley/WangLab/Data/Processed/sample-dataset/tissue-2D/starfinder.yaml - > tissue2d.yaml <<'EOF'
root_input_path: /stanley/WangLab/Data/Processed
root_output_path: /stanley/WangLab/<user>/tissue-2D/output
output_id: full
EOF
starfinder workflow plan --config tissue2d.yaml
```

The FOV list holds `tile_1` to `tile_56` in the order of their numbers, which
is the scan order of this dataset. The plan prints the manifest's path and its
tasks by kind: 228 tasks for 56 FOVs.

## What it does

**The input folder.** The input folder is
`<root_input_path>/<dataset_id>/<sample_id>/`. It holds one folder per round and
FOV, `<round>/<fov>/` (or `<fov>/<round>/` with `image_dir_template: "{fov}/{round}"`).
The sequencing rounds are `round1` to `round<n_rounds>`; other rounds are named
by `additional_round`.

In a round's folder, each configured channel label matches one file,
`*<label>*.tif` or `*<label>*.tiff`. For tissue-2D the label `ch00` matches
`9919seqA1_TileScan_005_Tile 1_cmle_ch00.tif`. A file that matches no label is
never read: the microscope's `.hgsb` and `_history.txt` files can stay where they
are. The images may be uncompressed or compressed TIFF, for example TIFF with zlib
(deflate): the reader decodes any codec that tifffile reads, so a copy of a raw folder
compressed with zlib to save space is read as the original. The plan refuses a label that matches no file or two, naming the FOV, the
round, the label and the files it found, before any task runs. The image reader
of the tasks (`starfinder.io.load_round`) finds the files with the same pattern
and raises `ValueError` with the count when a label does not match exactly one
file. The plan records each image's path, size and modification time; it reads
no image content and writes nothing under the input folder.

**The FOV list.** The FOV list is a CSV file with the columns `sample` and
`fov_id`, and optionally `position_id`, the stage file's PositionID of each FOV.
`sample` groups the FOVs that are stitched, exported and reported together.
`fov_list` names the file, absolute or relative to the input folder. Without
`fov_list`, the plan reads `fovs.csv` in the input folder. Keep the list next to
the configuration: beside the data with a dataset's own configuration (below),
or outside a raw folder you do not curate. Either way the workflow reads the
folder in place and writes nothing into it. `selected_fovs` then picks FOVs of
the list to run. To process a dataset in parts, for example the first tiles
while the rest are imaged, plan each larger selection into the same output
root: the finished FOVs' tasks are kept, and the new FOVs and the sample-level
tasks (stitching, export, report) run
({ref}`the task kinds <workflow-task-kinds>`).

**The row order.** The rows are read in order: a sample's FOVs keep the order of
their rows in the manifest, the sample tables and the report. With
`stitching.positions.source: regular_grid`, the grid places the FOVs in that
order, the first row in the first cell of `regular_grid.order`. Write the rows in
scan order. Sorted as text, `tile_1, tile_10, tile_11, ...` puts numbered FOVs in
the wrong cells without an error, so the plan warns when a sample's rows are in
text order and the numbers in the FOV ids give another order.

**The codebook.** The codebook is `genes.csv` in the input folder, or the file
`codebook` names (`direct_panel` in `readout_mode: direct`), absolute or relative
to the input folder. tissue-2D's `genes.csv` is in its raw folder.

(preparing-a-dataset-configuration)=
## A configuration per dataset

Each dataset the workflow has run on keeps its configuration beside its data:
`starfinder.yaml`, and the FOV list `fovs.csv` it names, in the dataset's folder
under `/stanley/WangLab/Data/Processed/`.

| Dataset | Folder | The run it holds |
| --- | --- | --- |
| tissue-2D | `sample-dataset/tissue-2D` | W-870, all 56 FOVs |
| cell-culture-3D | `sample-dataset/cell-culture-3D` | E14's held-out demons arm (W-773) |
| Covid_LN | `sample-dataset/LN` | E14's held-out demons arm |
| mPFC-aging | `2025-04-26-Jiakun-mPFC-aging` | E14's held-out demons arm, tiled |

A dataset's file holds the whole configuration of its last full run, in the
Python form:

* **The dataset's facts:** the dataset and sample ids, the rounds and colour
  order, the rotation and voxel size, the reference stains and additional
  rounds with their channel names, the FOV list, and the stitching positions
  where the dataset records them. The codebook is the folder's `genes.csv`,
  the default of `codebook`.
* **The run's own choices:** its steps and settings as that run set them. A
  key the run left unset is unset here too, so a later run takes the current
  default.

Each value carries a comment naming its source. The file holds no root, no
`output_id` and no `selected_fovs`: give them at run time by appending them, as
in the Quick start. Since the file holds none of these keys, each appears once.

**A new dataset.** Write its `starfinder.yaml` directly in this form, with
tissue-2D's as the template. There is no converter from `dataset-info.json`.
That file is MATLAB-era and the workflow does not read it, and the plan refuses
two of its keys, `fov_id_pattern` and `maximum_projection`. Check each value
against the files: the voxel size from the TIFF tags, the channel names from
the round folders. Then write the FOV list beside it. tissue-2D's list was
written this way:

```python
import csv
from pathlib import Path

raw = Path("/stanley/WangLab/Data/Processed/sample-dataset/tissue-2D")
number = lambda fov: int(fov.rsplit("_", 1)[1])
fovs = sorted((p.name for p in (raw / "round1").iterdir() if p.is_dir()), key=number)
with open("fovs.csv", "w", newline="") as handle:
    writer = csv.writer(handle, lineterminator="\n")
    writer.writerow(["sample", "fov_id"])
    writer.writerows(["tissue-2D", fov] for fov in fovs)
```

This is tissue-2D's `starfinder.yaml`:

```yaml
# tissue-2D: the Python workflow's configuration of /stanley/WangLab/Data/Processed/sample-dataset/tissue-2D (W-873).
# The whole configuration of the dataset's last full run, W-870's stage 2 on all 56 FOVs, export route per_fov
# (runs/W-870/20261118T0309Z-tissue2d/configs/stage2-per_fov.yaml, after the placement approval; its base is
# configs/tissue2d.yaml, whose comments give the decisions D1-D4, Jiahao, 2026-11-17). Its processing.recipe deep is
# written out as the steps it expanded to (W-887). A key that run left unset stays unset here.
# Give at run time: root_input_path (the folder that holds sample-dataset/, /stanley/WangLab/Data/Processed),
# root_output_path and output_id; selected_fovs only to run some FOVs.
backend: python
# W-870: the dataset's ids; <root_input_path>/sample-dataset/tissue-2D is this folder
dataset_id: sample-dataset
sample_id: tissue-2D
# W-872's key: the FOV list beside this file, tile_1 .. tile_56 in scan order (the order of their numbers; the grid
# below fills in that order), the same rows as W-870's link-tree fovs.csv
fov_list: fovs.csv
# W-870 (E14 datasets.json, tissue2d): four sequencing rounds (codes of 5 bases in genes.csv), reference round1
n_rounds: 4
ref_round: round1
# W-870 (E14 datasets.json): the colour order ch00, ch02, ch01, ch03 (488, 546, 594, 647 nm)
seq_channel_order: [ch00, ch02, ch01, ch03]
# W-870 (E14 datasets.json; dataset-info.json rotate_angle)
rotate_angle: -90
# W-870 (E14 datasets.json): XY from the TIFF resolution tag (2^32-1 / 406454246 per um), Z the ImageJ spacing tag
voxel_size_xy: 0.094635
voxel_size_z: 0.346255
# W-870 (D3): PI is the shared stain; dataset-info.json ref_channel PI. round1's fifth file, ch04, is PI.
ref_channel: PI
reference_stains: [{channel: ch04, name: PI}]
# W-870 (dataset-info.json additional_round and channel_order): the protein round's four channels
additional_round:
  - round_name: protein
    channel_order:
      - {channel: ch00, name: plaque}
      - {channel: ch01, name: tau}
      - {channel: ch02, name: PI}
      - {channel: ch03, name: Gfap}
# The codebook is genes.csv in this folder, the default of the key codebook, which W-870 left unset.
processing:
  # W-870 (D1: E14's global arm)
  mode: direct
  # W-870's recipe deep, written out as the steps it expanded to (W-887's migration guide; W-870's fovs/<fov>/run.json
  # record them): min-max to uint8, histogram matching to channel 0, and a reconstruction of radius 3 after
  # registration; one global translation on the channels' maximum
  preprocessing:
    steps:
      - {method: min_max_normalization, output_dtype: uint8, output_range: [0, 255], rounding: truncate}
      - {method: histogram_matching, reference_channel: 0}
    post_registration:
      - {method: reconstruction, radius_yx: 3}
  registration:
    steps:
      - {method: translation}
    signal: {mode: max}
  # W-870
  streaming: true
  # W-870: the package's spot finding and extraction (empty blocks: the current defaults)
  spot_finding: {}
  reads_extraction: {}
  # W-870 (E14 datasets.json): codes of 5 bases start C and end C
  reads_filtration: {end_base: CC, start_base: C}
# W-870 (D3): the protein round, registered by one translation on the shared stain (recipe null), and one maximum
# projection per channel
morphology:
  rounds: [protein]
  recipe: null
  channel_images: {projection: max}
# W-870 (D2): StarDist's 2D_versatile_fluo on the PI stain's maximum projection, scaled by 0.25 (a 6-10 um nucleus is
# about 16-26 px after scaling). W-891: the input recipe pi_max; the model sees pixels of 0.37854 um (0.094635 / 0.25),
# the factor 0.25 per FOV and 0.5 after fusion at k = 2
segmentation:
  device: cpu
  input_recipes:
    pi_max:
      projection: {method: max}
      model_input: {pixel_size_um: 0.37854}
  runs:
    - name: nucleus
      target: nucleus
      method: stardist
      model: 2D_versatile_fluo
      inputs: [{role: nuclear, round: reference_stain, channel: PI}]
      input_recipe: pi_max
# W-870 (D2): the cells are the nuclei expanded by 0.7776 um in each plane (W-858's primary territory)
assignment:
  cells: nucleus
  nuclei: nucleus
  name: default
  expansion: {distance: 0.7776, unit: um, mode: planar}
  previews: true
# W-870: multiview_stitcher (bigstitcher fails on single-plane input, W-856); quality_threshold 0.3 (the coordinator's
# decision of 2026-11-18: 0.7 rejected every real link, qualities 0.49-0.68). dataset-info.json and grid.csv: 7 columns
# x 8 rows, column by column, top down then right, overlap 0.1
stitching:
  method: multiview_stitcher
  config: {quality_threshold: 0.3}
  positions:
    source: regular_grid
    regular_grid: {columns: 7, rows: 8, order: column_major_down_right, overlap: 0.1}
# W-870
assembly: {assignment: default}
# W-870 (D4, route 1 of 2): per_fov; its route 2 replaced this block by route after_fusion with export.segmentation
export:
  route: per_fov
  images: [reference_merged, reference_stain]
  previews: true
# W-870
report: {}
# W-870's stage-2 requests on UGER (scripts/stage2_resources.py from stage 1's measures; stitch, report, export and
# compare after the placement approval, scripts/stage2_a2.py), one thread per task
execution:
  defaults: {threads: 1, memory_mb: 8000, runtime_min: 30, gpu: false}
  tasks:
    fov: {threads: 1, memory_mb: 32000, runtime_min: 105}
    morphology: {threads: 1, memory_mb: 28000, runtime_min: 15}
    segment: {threads: 1, memory_mb: 12000, runtime_min: 15}
    assign: {threads: 1, memory_mb: 12000, runtime_min: 15}
    report: {threads: 1, memory_mb: 24000, runtime_min: 45}
    stitch: {threads: 1, memory_mb: 16000, runtime_min: 15}
    export: {threads: 1, memory_mb: 64000, runtime_min: 120}
    compare: {threads: 1, memory_mb: 8000, runtime_min: 30}
  resume_from_checkpoint: false
  code_version: keep
  log_level: INFO
```

Its `processing` block writes out the steps of the retired `recipe: deep`
({ref}`migration guide <migration-named-recipe>`). The
{doc}`reference <reference>` lists every key.

(preparing-a-dataset-deconvolution)=
## Deconvolved images

A dataset whose images are raw confocal stacks can be deconvolved by the workflow
({ref}`the deconvolution stage <workflow-deconvolution>`). A dataset that another
program already deconvolved, such as the Huygens exports of the catalog datasets,
is an input like any other and has no `deconvolution` block.

* **The optics** come from the microscope's records of the acquisition (a Leica
  `.lif` or Huygens' `.hgsb` settings): the objective's numerical aperture, the
  immersion medium's and the sample's refractive indices, the pinhole radius
  back-projected into the sample, and each channel's excitation and emission
  wavelengths. The block holds them under `psf`; none has a default. The voxel size
  is the dataset's `voxel_size_xy` and `voxel_size_z`. A measured PSF can be given
  as files instead (`psf.source: file`, with their declared spacing
  `psf.voxel_size_um`, which must equal the dataset's).
* **The iterations** have no default. Choose them on one development position that
  has a comparator, such as a Huygens result of the same position: run 5, 10, 20, 40
  and 80 iterations, take the count with the highest whole-volume normalised
  cross-correlation to the comparator averaged over the channels, and among the
  counts within 0.005 of the best take the fewest. Freeze the count before the other
  positions run. 9735seqA1 chose 5 this way (W-915), at the low edge of the grid. A
  dataset without a comparator states its count and its reason in a comment of its
  configuration.
* **The storage.** The default copy is uint8 with one stretch factor per FOV and
  round, as Huygens' exports are; it is about 0.4 times the size of an 8-bit raw
  position with zlib. A dataset whose raw images are not uint8, or that compares
  intensities across FOVs (supplied statistics, or a `global` or `absolute` spot
  threshold), or that relies on spot finding's `noise` threshold, sets
  `dtype: uint16`: one fixed `scale` for every FOV, 64 by default, so that a stored
  value is 64 × the deconvolved grey level. Choose the scale so that
  scale × the largest expected value stays within 65,535; 64 holds grey levels up to
  1,023.98. The record of each round counts the voxels that saturate.

## Learned methods and the weights folder

Four methods run a trained model: the segmentation methods `stardist` and
`cellpose`, and the spot-finding methods `spotiflow` and `piscis`. A task that
runs one needs:

* **the learned image**, `learned` on CPU or `learned-cuda` with `device: cuda`.
  The CPU image does not hold the learned backends. The
  {doc}`UGER guide <broad-uger>` describes the three images.
* **the weights folder**, which holds the models as `<method>/<model>/`. Give it
  as `STARFINDER_WEIGHTS_DIR` in the head's environment under Snakemake, or as
  `--weights_dir` under Nextflow, which passes it to every task as
  `STARFINDER_WEIGHTS_DIR`. Without it a task reads
  `~/.cache/starfinder/weights`. `starfinder weights fetch <method> <model>`
  downloads a model into it, for example `stardist 2D_versatile_fluo`.

The package has no classical nucleus segmentation: `seeded_watershed` grows
cells from the nuclei of a seed run, so a run that segments nuclei from a stain
needs `stardist` or `cellpose`. A run with `method: import` reads label images
another tool wrote and needs neither.

## On UGER

The tasks run in the image with only `/stanley` bound (the Snakemake head also
binds `/broad/uge`), so the input folder, the FOV list, the codebook, the output
root and the weights folder must all be under `/stanley`. Give their `/stanley`
paths in the configuration, not those of another machine's mount of the same
share. The plan runs on the cluster too, in the Snakemake head's container or as
a Nextflow task; it records the paths it read, and the tasks open them there.
The raw folder is read in place there as well. The {doc}`UGER guide <broad-uger>` gives the profiles.

## Limits

* `fov_list` is part of the configuration, like `codebook`: a plan of the same
  configuration with another path to the list gives other task fingerprints, and
  the tasks rerun. A path relative to the input folder stays the same where
  both lie on one share under different mount points.
* The FOV list is a file; the configuration does not hold the FOVs inline.
* The order warning looks at FOV ids that share one numeric pattern (the same
  text around their last number). It does not know the scan order: a list in
  numeric order that is not the scan order passes without a warning.

## How it is checked

Contract tests plan small input folders written in the test: the FOV list named
by `fov_list` (absolute and relative) and without it, files of no configured
channel left unread, a label that matches two files refused with both names, and
the order warning. W-872 planned the raw tissue-2D folder in place with
`fov_list` and compared the task set and the image list with W-870's plan on
its link tree. W-873 planned each dataset's configuration on its folder, read
in place, with the run's keys appended. tissue-2D's resolved configuration
equals W-870's, apart from the roots, the output id, the FOV list and two
defaults changed since: the spot-finding threshold (W-880) and the retired
recipe key (W-887).
