# Preparing a dataset

:::{container} sf-lead
Get a new dataset ready for `starfinder workflow plan`: the input folder, the FOV
list, the codebook and the configuration. The workflow reads a raw dataset folder
where it is; nothing is copied, linked or written into it.
:::

## Quick start

The tissue-2D sample dataset is a raw folder in the layout the workflow reads.
Write its FOV list outside the folder, name the list in the configuration with
`fov_list`, and plan:

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

```bash
starfinder workflow plan --config tissue2d.yaml
```

The list holds `tile_1` to `tile_56` in the order of their numbers, which is
the scan order of this dataset. The plan prints the manifest's path and its tasks
by kind; with the configuration below, 228 tasks for 56 FOVs.

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
are. The plan refuses a label that matches no file or two, naming the FOV, the
round, the label and the files it found, before any task runs. The image reader
of the tasks (`starfinder.io.load_round`) finds the files with the same pattern
and raises `ValueError` with the count when a label does not match exactly one
file. The plan records each image's path, size and modification time; it reads
no image content and writes nothing under the input folder.

**The FOV list.** The FOV list is a CSV file with the columns `sample` and
`fov_id`, and optionally `position_id`, the stage file's PositionID of each FOV.
`sample` groups the FOVs that are stitched, exported and reported together.
`fov_list` names the file, absolute or relative to the input folder. Without
`fov_list`, the plan reads `fovs.csv` in the input folder. Keep the list outside
a raw folder, next to the configuration, so that the folder is read in place.
`selected_fovs` then picks FOVs of the list to run.

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

## The configuration

This is the configuration W-870 ran tissue-2D with, end to end, with three
changes: `fov_list` and the raw folder replace the link tree it built, and the
package defaults replace its previews, its stitching quality threshold and most
of its resources. The FOV list is the file of the Quick start.

```yaml
backend: python
root_input_path: /stanley/WangLab/Data/Processed
root_output_path: /stanley/WangLab/<user>/tissue-2D/output
dataset_id: sample-dataset
sample_id: tissue-2D
output_id: full
fov_list: /stanley/WangLab/<user>/tissue-2D/fovs.csv
n_rounds: 4
ref_round: round1
seq_channel_order: [ch00, ch02, ch01, ch03]
rotate_angle: -90
voxel_size_xy: 0.094635
voxel_size_z: 0.346255
ref_channel: PI
reference_stains: [{channel: ch04, name: PI}]
additional_round:
  - round_name: protein
    channel_order:
      - {channel: ch00, name: plaque}
      - {channel: ch01, name: tau}
      - {channel: ch02, name: PI}
      - {channel: ch03, name: Gfap}
processing:
  mode: direct
  recipe: deep
  streaming: true
  spot_finding: {}
  reads_extraction: {}
  reads_filtration: {end_base: CC, start_base: C}
morphology:
  rounds: [protein]
  recipe: null
  channel_images: {projection: max}
segmentation:
  device: cpu
  runs:
    - name: nucleus
      target: nucleus
      method: stardist
      model: 2D_versatile_fluo
      scale: 0.25
      inputs: [{role: nuclear, round: reference_stain, channel: PI}]
      projection: {method: max}
assignment:
  cells: nucleus
  nuclei: nucleus
  expansion: {distance: 0.7776, unit: um, mode: planar}
stitching:
  method: multiview_stitcher
  positions:
    source: regular_grid
    regular_grid: {columns: 7, rows: 8, order: column_major_down_right, overlap: 0.1}
assembly: {assignment: default}
export:
  route: per_fov
  images: [reference_merged, reference_stain]
report: {}
execution:
  tasks:
    fov: {memory_mb: 64000, runtime_min: 180}
```

Each value comes from the dataset: the rounds and colour order, the rotation and
voxel size, `ch04` of `round1` as the PI stain, the protein round's four channels,
the code's start and end bases, and the 7 × 8 grid filled column by column,
top down. The {doc}`reference <reference>` lists every key.

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
its link tree.
