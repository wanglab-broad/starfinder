# Assignment reference

The options, statuses, columns, files and workflow keys of assignment. The option tables
are read from the code when the site is built. The
{doc}`assignment guide <assignment-contract>` explains how to use them; the
{doc}`assignment API <api/assignment>` documents every class.

## Options

### Assignment

The expansion, the correspondence thresholds and the exclusion. See
{py:class}`~starfinder.assignment.AssignmentConfig`.

```{config-options} starfinder.assignment.AssignmentConfig
```

### Correspondence

The overlap rule that pairs nuclei with cells. See
{py:class}`~starfinder.assignment.CorrespondenceConfig`.

```{config-options} starfinder.assignment.CorrespondenceConfig
```

### Expansion

The expansion is segmentation's `expand_labels`, applied by assignment to the cell
territories. See {py:class}`~starfinder.segmentation.ExpandLabelsConfig`.

```{config-options} starfinder.segmentation.ExpandLabelsConfig
```

## Functions

| Function | What it does |
| --- | --- |
| `assign_molecules` | Assigns a `MoleculeTable` to a cell label image, with optional nuclei, and returns an `AssignmentResult` |
| `molecule_table` | Builds a `MoleculeTable` from a spot result and its reads |
| `load_molecules` | Reads the `molecules.csv` that `FOV.export_spots` writes |
| `import_legacy_molecules_csv` | Reads a legacy one-based `x, y, z, gene` CSV |
| `labels_at_positions` | Reads a label image at molecule positions |
| `match_nuclei` | Pairs nuclei with cells by overlap |
| `summarize_assignment` | Counts statuses, flags and states, with quantiles and the totals before and after the exclusion |
| `plot_assignment` | Draws the territories, the cell centres and the molecules, with two histograms |
| `write_h5ad` | Writes long counts as an AnnData file |

`FOV.assign_molecules` runs `assign_molecules` on a FOV's results, and
`FOV.load_assignment` reads a saved assignment back.

## Statuses

| Molecule status (`ASSIGNMENT_STATUSES`) | Meaning |
| --- | --- |
| `assigned` | In the territory of a kept cell |
| `unassigned` | In the grid, outside every territory |
| `excluded_cell` | In the territory of an excluded cell |
| `outside_grid` | Outside the grid |

| Cell status (`CELL_STATUSES`) | Meaning |
| --- | --- |
| `kept` | In the counts |
| `excluded_no_nucleus` | Excluded: no nucleus belongs to it |

| Cell correspondence (`CELL_CORRESPONDENCE`) | Meaning |
| --- | --- |
| `matched` | At least one nucleus is matched to it |
| `ambiguous` | No matched nucleus, and an ambiguous nucleus overlaps it |
| `no_nucleus` | No matched nucleus, and no ambiguous nucleus overlaps it |
| `unavailable` | No nuclei were given |

| Nucleus status (`NUCLEUS_STATUSES`) | Meaning |
| --- | --- |
| `matched` | Paired with one cell |
| `ambiguous` | Overlaps cells, none paired with it |
| `no_cell` | Outside every cell |

| Compartments (`COMPARTMENT_STATES`) | Meaning |
| --- | --- |
| `available` | Nuclear and cytoplasmic counts exist |
| `withheld` | Not split: a doubtful pairing |
| `no_nucleus` | Not split: no nucleus belongs to the cell |
| `unavailable` | Not split: no nuclei were given |

| Cell flag | Meaning |
| --- | --- |
| `several_nuclei` | Two or more matched nuclei |
| `ambiguous_nucleus` | An ambiguous nucleus overlaps the cell |
| `nucleus_outside_cell` | A nucleus matched to the cell is flagged `outside` |
| `foreign_nucleus` | The cell holds part of a nucleus matched to another cell |

## Tables

### Molecules

`AssignmentResult.molecules`, one row per input molecule, in input order:

| Column | Type | Contents |
| --- | --- | --- |
| `spot_namespace`, `spot_id` | string | The molecule's identity |
| `z`, `y`, `x` | float64 | Zero-based position on the reference grid |
| `gene_id` | string | The gene |
| `voxel_z`, `voxel_y`, `voxel_x` | Int64 | The sampled voxel |
| `assignment_status` | string | The molecule status |
| `cell_id` | UInt32 | The cell, or null |
| `in_expansion` | boolean | Whether the cell's value comes from the expansion only |
| `original_cell_id` | UInt32 | The original cell label at the voxel |
| `nucleus_id` | UInt32 | The nucleus label at the voxel, 0 outside every nucleus |
| `compartment` | string | `nucleus` or `cytoplasm`, or the cell's compartments state; null unless `assigned` |

### Cells

`AssignmentResult.cells`, one row per cell, in increasing `cell_id`:

| Column | Type | Contents |
| --- | --- | --- |
| `cell_id` | UInt32 | The label value |
| `status` | string | The cell status |
| `exclusion_reason` | string | `no_matched_nucleus` for an excluded cell |
| `size_voxels` | int64 | Voxels of the original territory |
| `expanded_size_voxels` | Int64 | Voxels of the expanded territory |
| `size_physical`, `expanded_size_physical` | float64 | The same in the unit of `record["size_unit"]` |
| `centroid_z`, `centroid_y`, `centroid_x` | float64 | Mean voxel index of the original territory |
| `expanded_centroid_z`, `expanded_centroid_y`, `expanded_centroid_x` | float64 | The same for the expanded territory |
| `n_molecules` | int64 | Molecules in the territory, `assigned` or `excluded_cell` |
| `n_nuclei` | Int64 | Matched nuclei |
| `correspondence` | string | The cell correspondence |
| `correspondence_flags` | string | The cell flags, joined by `;` |
| `compartments` | string | The compartments state |

### Nuclei

`AssignmentResult.nuclei`, one row per nucleus, in increasing `nucleus_id`:

| Column | Type | Contents |
| --- | --- | --- |
| `nucleus_id` | UInt32 | The label value |
| `size_voxels` | int64 | Voxels of the nucleus |
| `status` | string | The nucleus status |
| `cell_id` | UInt32 | The matched cell, or null |
| `share_in_cell` | float64 | Share of the nucleus in the matched cell, or in the cell holding most of it; 0 for `no_cell` |
| `share_background` | float64 | Share of the nucleus outside every cell |
| `n_cells_overlapped` | int64 | Cells the nucleus overlaps |
| `outside` | boolean | A matched nucleus lies partly outside its cell, beyond `outside_tolerance` |
| `seed_value_agrees` | boolean | For cells grown from these nuclei as seeds: whether the nucleus is matched to the cell of the same value; null otherwise |

### Counts

`AssignmentResult.counts`, the nonzero counts of the kept cells:

| Column | Type | Contents |
| --- | --- | --- |
| `cell_id` | UInt32 | The cell |
| `gene_id` | string | The gene |
| `compartment` | string | `whole`, `nucleus` or `cytoplasm` |
| `count` | int64 | Molecules |

## Files

`FOV.assign_molecules` with `checkpoints` writes `<FOV folder>/assignment/<name>/`:

| File | Contents |
| --- | --- |
| `molecules`, `cells`, `counts`, `nuclei` | The tables, as `.csv` or `.parquet` (`CheckpointConfig.table_format`); `nuclei` with nuclei only |
| `assignment.json` | The run record |
| `territories.tif` | The expanded territories, with an expansion |
| `cell_labels.tif`, `nucleus_labels.tif` | Copies of the label images that are not saved under their segmentation run |

The label images are ZYX `uint32` with the grid's metadata.

## Records

`AssignmentResult.record`, written as `assignment.json`, holds:

| Key | Contents |
| --- | --- |
| `format_version`, `stage` | `1` and `"assignment"` |
| `dataset_id`, `sample_id`, `fov_id`, `subtile_id`, `name` | The identities and the assignment name |
| `cell_namespace` | The cell run's label namespace |
| `territory_source` | The cell run, its target, whether it was segmented or imported, and whether assignment expanded it |
| `grid` | The molecules' grid: shape, metadata, source, SHA-256 and the grid check |
| `inputs` | The molecule table, the cell and nucleus runs with their records and label files, and the correspondence source |
| `config` | The settings, with where the exclusion setting came from |
| `sampling` | The sampling rule and the sampled axes |
| `expansion` | The expansion, with the SHA-256 of both masks and, when saved, their files; or `null` |
| `calibration`, `calibration_source`, `size_unit` | Whether the grid has a voxel spacing and unit, where it came from, and the unit of the physical sizes |
| `counts` | The totals; after `FOV.assign_molecules`, those of `summarize_assignment` |
| `outcome` | `ok`, or `empty` without cells |
| `files` | Each written file with its SHA-256 |
| `software` | The software versions |

## Workflow keys

### The assignment block

With `backend: python`, the top-level `assignment` block:

| Key | Meaning |
| --- | --- |
| `cells`, `nuclei`, `name`, `population` | As in `FOV.assign_molecules` |
| `expansion` | The `ExpandLabelsConfig` fields, or `null` |
| `correspondence` | The `CorrespondenceConfig` fields |
| `legacy_pixel_expansion`, `exclude_cells_without_nucleus` | As in `AssignmentConfig` |
| `checkpoints` | The `CheckpointConfig` fields |

### The reads_assignment rule

| Legacy key | Becomes |
| --- | --- |
| `expand_labels: true`, `dilation_distance` | One planar expansion by `dilation_distance` pixels |
| `expand_labels: false` | No expansion |

Without an `assembly` block the rule writes:

| File | Contents |
| --- | --- |
| `expr/{fovID}/raw.h5ad` | AnnData: whole-cell counts in `X` and the cells in `obs`; with nuclei, `nucleus` and `cytoplasm` layers |
| `expr/{fovID}/reads_assignment.csv` | The molecule rows with their assignment columns |
| `expr/{fovID}/assignment.png`, `assignment_single_layer.png` | The `plot_assignment` figures |
| `expr/{fovID}/log.txt` | The molecule totals |

The run record keeps the legacy keys under `workflow.legacy`. {doc}`workflow-downstream`
describes the rule's outputs with an `assembly` block.
