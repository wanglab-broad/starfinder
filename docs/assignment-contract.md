# Assignment

Assignment places each molecule of a FOV in the cell whose territory holds it and counts
the molecules of each gene per cell. With nuclei, it also pairs nuclei with cells and
splits a cell's counts into nuclear and cytoplasmic parts where the pairing is clear.

It is the third call per FOV, `FOV.assign_molecules`, after `FOV.run`
({doc}`readout <readout-contract>`) and `FOV.segment`
({doc}`segmentation <segmentation-contract>`). The
{doc}`assignment reference <assignment-algorithms>` lists the options, statuses, columns
and files; the {doc}`assignment API <api/assignment>` documents the classes.

## Quick start

```python
from dataclasses import replace

import numpy as np

from starfinder.assignment import AssignmentConfig
from starfinder.dataset import CheckpointConfig, Dataset, PipelineConfig, RoundState
from starfinder.image import ImageMetadata
from starfinder.io import ImageLoadConfig, save_volume
from starfinder.readout import NeighborhoodSumConfig, ReadFilterConfig, ReadScoreConfig, WtaDecoderConfig
from starfinder.segmentation import (ExpandLabelsConfig, InputChannel, LabelImportConfig, SeededWatershedConfig,
                                     SegmentationPlan, SegmentationRun)
from starfinder.spot_finding import LocalMaximaConfig
from starfinder.synthetic import benchmark_scene_preset, generate_amplicon_dataset, save_amplicon_dataset

# Molecules: a synthetic FOV with a voxel spacing in µm, decoded by FOV.run.
codebook, scene = benchmark_scene_preset("tiny")
generated = generate_amplicon_dataset(codebook, replace(scene, seed=42), fov_ids=("FOV_001",), preset="tiny")
save_amplicon_dataset(generated, "input")
rounds, channels = list(generated.codebook.round_labels), list(generated.channel_labels)
dataset = Dataset(input_root="input", output_root="results", dataset_id="tiny", sample_id="synthetic",
                  output_id="demo", rounds=RoundState(sequencing_rounds=rounds, reference_round="round1"),
                  channel_order=channels)
dataset.load_codebook("input/codebook.csv")
load = ImageLoadConfig(channel_labels=tuple(channels),
                       metadata=ImageMetadata("tiny", spacing_zyx=(0.5, 0.1, 0.1), spatial_unit="micrometer"))
fov = dataset.fov("FOV_001")
fov.run(PipelineConfig(load=load, spot_finding=LocalMaximaConfig(), extraction=NeighborhoodSumConfig(),
                       decoding=WtaDecoderConfig(), scoring=ReadScoreConfig(), filtering=ReadFilterConfig()))

# Label images: imported nuclei, and one cell grown from each nucleus.
grid = fov.reference_grid()
nuclei = np.zeros(grid.shape_zyx, dtype=np.uint16)
nuclei[:, 30:45, 30:45], nuclei[:, 80:95, 70:85] = 1, 2
save_volume(nuclei, "nuclei.tif", metadata=grid.metadata)
fov.segment(SegmentationPlan((
    SegmentationRun("nucleus", "nucleus", (), LabelImportConfig("nuclei.tif", "nucleus")),
    SegmentationRun("cell", "cell", (InputChannel("amplicon", reference_merged=True),),
                    SeededWatershedConfig(), seeds="nucleus"),
)), checkpoints=CheckpointConfig())

# Assignment, with the nuclei and a 0.5 µm expansion of the cells.
config = AssignmentConfig(expansion=ExpandLabelsConfig(distance=0.5, unit="um", mode="planar"))
fov.assign_molecules(config, cells="cell", nuclei="nucleus", checkpoints=CheckpointConfig())

result = fov.assignment_results["default"]
print(result.molecules.assignment_status.value_counts().to_dict())    # molecules per status
print(result.cells[["cell_id", "status", "compartments", "n_molecules"]])
matrix, cells = result.matrix("whole")                                 # kept cells × genes
print(matrix.shape)
```

The call writes `results/fovs/FOV_001/assignment/default/`. Without `nuclei` you get
whole-cell counts only.

## What it does

1. **Molecules.** Builds the molecule table from the FOV's reads.
2. **Expansion**, if asked: grows the cell territories once, keeping the original labels.
3. **Sampling.** Reads the territory, and the nucleus, at each molecule's voxel.
4. **Nuclei and cells**, with nuclei: pairs nuclei with cells by overlap and flags
   doubtful pairs.
5. **Compartments.** Splits the molecules of clearly paired cells into nuclear and
   cytoplasmic.
6. **Exclusion.** Applies the configured cell exclusion.
7. **Counts.** Counts the assigned molecules per cell and gene.
8. **Saving**, with `checkpoints`.

## What you get

* **In memory:** `fov.assignment_results[name]`, an
  {py:class}`~starfinder.assignment.AssignmentResult`:
  * `molecules`: one row per input molecule, with its `assignment_status`, `cell_id`,
    `nucleus_id` and `compartment`;
  * `cells`: one row per cell ([Cell table](#cell-table));
  * `counts`: the nonzero counts of the kept cells: `cell_id`, `gene_id`, `compartment`
    (`whole`, `nucleus` or `cytoplasm`) and `count`;
  * `nuclei`: one row per nucleus, or `None` without nuclei;
  * `cell_labels`, `territories` and `nucleus_labels`: the original cell labels, the
    expanded ones (or `None`) and the nucleus labels (or `None`);
  * `record`: the run record: inputs, settings and totals.

  `result.matrix(compartment)` gives a dense cells × genes matrix; `plot_assignment`
  draws a result.
* **On disk:** with `checkpoints`, `<FOV folder>/assignment/<name>/` holds the tables,
  `assignment.json` and the label images; label images already saved
  under their segmentation run are linked, not copied. `fov.load_assignment(name)` reads
  the folder back and checks its hashes.

## Main options

* `cells` and `nuclei`: the segmentation runs, or `SegmentationResult` objects, used as
  cell territories and nuclei.
* `AssignmentConfig.expansion`: grows the cell territories before sampling.
* `CorrespondenceConfig.match_fraction`: controls nucleus–cell pairing by overlap.
* `CorrespondenceConfig.outside_tolerance`: controls flags for nuclei outside their cells.
* `AssignmentConfig.exclude_cells_without_nucleus`: keeps or excludes the `no_nucleus`
  cells.
* `population`: the accepted reads (`final`) or every read called to a gene (`called`).

The reference gives each config's options and their defaults.

## The assign entry

To assign without a FOV, call `assign_molecules` with a
{py:class}`~starfinder.assignment.MoleculeTable`, the cell `SegmentationResult`, the
molecules' reference grid and, optionally, the nuclei; it reads and writes no file. A
`MoleculeTable` comes from `molecule_table` (a spot result and its reads),
`load_molecules` (the `molecules.csv` of `FOV.export_spots`) or
`import_legacy_molecules_csv` (a legacy one-based `x, y, z, gene` CSV).

## Coordinate sampling

Assignment samples the label images at each molecule's position on the reference grid;
see the [sampling specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/assignment.md#coordinate-sampling)
for voxel boundaries and plane images.

## Label expansion

`AssignmentConfig.expansion` grows cell territories into the background before sampling,
while preserving the original labels in `cell_labels` and the grown labels in `territories`.
The molecule table records `in_expansion` and `original_cell_id`; see
{py:class}`~starfinder.segmentation.ExpandLabelsConfig` for expansion settings.

## Nucleus–cell correspondence

With `nuclei`, assignment pairs nuclei with cells by overlap and records correspondence
statuses and flags to help you inspect doubtful pairs; see the
[correspondence specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/assignment.md#nucleuscell-correspondence)
for the matching rules.

To pair them yourself, pass `correspondence`, a table of `nucleus_id` and `cell_id`.

## Compartments

Assignment can split whole-cell counts into nuclear and cytoplasmic counts when the
nucleus–cell pairing is clear; see the
[compartment specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/assignment.md#compartments)
for eligibility and count representation.

## Statuses

Every input molecule keeps one row with one status:

* `assigned`: in the territory of a kept cell;
* `unassigned`: in the grid, outside every territory;
* `excluded_cell`: in the territory of an excluded cell;
* `outside_grid`: outside the grid.

The cell table also records exclusions, controlled by
`AssignmentConfig.exclude_cells_without_nucleus`; see
{py:class}`~starfinder.assignment.AssignmentConfig` for details.

## Cell table

`result.cells` has one row per cell with its status, sizes,
centroids, numbers of molecules and matched nuclei, correspondence, flags and
compartments. The reference lists the columns.

## Count accounting

The result checks that its molecule totals and cell counts agree; see the
[count specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/assignment.md#count-accounting)
for the identities checked.
`FOV.assign_molecules` stores `summarize_assignment(result)` in `record["counts"]`,
with the totals before and after exclusion.

## Troubleshooting

| Error | When | What to do |
| --- | --- | --- |
| `IncompatibleGeometryError` | The label images are not on the molecules' grid, or the nuclei not on the cell grid | Segment on `fov.reference_grid()`, or on its projection |
| `ValueError` naming "expanded by segmentation" | The cell run expanded its labels in `FOV.segment` | Segment without the expansion and set `AssignmentConfig.expansion` |
| `ValueError` naming "um" or "pixel" | The expansion unit does not suit the grid | Use `um` on a grid with a voxel spacing and unit, `pixel` on one without |
| `ValueError` naming "the assignment block" | The `reads_assignment` rule runs while an `assignment` block is set | Remove the block or the rule |

## Workflow configuration

The `reads_assignment` rule assigns each FOV's molecules to the label image of the
`stardist_segmentation` rule. In `rules.reads_assignment.parameters`, `expand_labels:
true` with `dilation_distance` grows the cells once, in each plane, by that many pixels.
Without an `assembly` block the rule writes `expr/{fovID}/raw.h5ad` and
`expr/{fovID}/reads_assignment.csv`; {doc}`workflow-downstream` describes its outputs
with one.

With `backend: python`, the top-level `assignment` block holds the settings of
`FOV.assign_molecules`:

```yaml
backend: python
assignment:
  cells: cell
  nuclei: nucleus
  expansion: {distance: 0.5, unit: um, mode: planar}
  correspondence: {match_fraction: 0.5, outside_tolerance: 0.1}
```

Its keys are the `AssignmentConfig` fields and `cells`, `nuclei`, `name`, `population`
and `checkpoints`. The `segmentation` of the [export block](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/sample-segmentation.md#where-the-settings-live)
takes its `AssignmentConfig` fields from it.

## Limits

* No segmentation, and no probabilistic or nearest-cell assignment.
* No filter by cell size or count, and no repair of doubtful pairs.
* No stitching: {doc}`assembly <assembly-contract>` joins the FOVs of a sample.
* No accuracy claim.

## How it is checked

Hand-built label images and molecules with known answers check the statuses, the
sampling, the pairing, the compartments, the exclusion, the counts, the saved files and
the workflow translation. A golden test checks that the package reproduces the
legacy script's cells and counts. The validation design is in
[`design/specs/assignment-algorithms.md`](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/assignment-algorithms.md).

The full specification is in
[`design/specs/assignment.md`](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/assignment.md).

```{toctree}
:hidden:

Assignment reference <assignment-algorithms>
```
