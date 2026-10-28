# Sample assembly

Assembly puts the molecules and cells of one sample into the coordinates of its
{doc}`placement <stitching-contract>`. It records which observations are retained,
which are excluded and why, and builds the sample tables and cell-by-gene counts.
{doc}`Sample output <sample-output-contract>` coordinates assembly with image
fusion and export through `Dataset.export`.

## Quick start

This excerpt starts with a reviewed `placement`, a `grids` mapping of FOV IDs to
`ReferenceGrid` objects, and an `assignments` mapping of FOV IDs to completed
`AssignmentResult` objects or saved assignment folders. These inputs come from
the preceding stitching and assignment stages. Saving uses the `anndata` extra.

```python
from starfinder.assembly import (
    AssemblyInput, assemble, load_assembly, save_assembly,
)

inputs = [
    AssemblyInput(fov, grids[fov], assignment=assignment)
    for fov, assignment in assignments.items()
]
result = assemble(placement, inputs)
print(result.cells[["obs_name", "assembly_status", "assembly_reason"]])
print(result.molecules[["spot_id", "sample_y", "sample_x", "assembly_status"]])
saved = save_assembly(result, "export/sample1")
restored = load_assembly("export/sample1")
```

For molecules without segmentation, pass `molecules=` with a `MoleculeTable` in
`AssemblyInput`. The {doc}`multi-FOV demo <get-started/multi-fov-demo>` shows the
saved-data workflow through placement, assembly and export.

## What it does

1. **Read the inputs.** `assemble` takes a `PlacementResult` and per-FOV inputs.
   An input pairs its reference grid with an assignment result, a saved
   assignment folder, or a molecule table.
2. **Map coordinates.** Assembly adds sample-coordinate columns in µm while
   keeping the local coordinates. The placement and reference grids supply the
   spatial information.
3. **Choose observations.** On the route with per-FOV assignments, ownership
   regions select cell observations and unassigned molecules. The duplicate rule
   can exclude overlapping cell observations; molecules assigned to a cell
   follow that cell's retention.
4. **Account for the sample.** The result keeps retained and excluded observations
   with statuses, reasons and flags, and gathers counts for counted cells.
   Construction checks the count accounting.

`ownership_map` provides a point query for overlap ownership. `fov_ownership`
provides the region and excluded observations that `plot_assignment` can draw.
The [specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/assembly.md#ownership-regions)
gives the ownership and retention rules. The
{doc}`sample segmentation guide <sample-segmentation-contract>` describes assembly
with cells segmented after fusion.

## What you get

`AssemblyResult` holds `fovs`, `cells`, `molecules`, `counts`, the ordered `genes`
and a `record`. `assembly_status` distinguishes `retained` from `excluded`;
`assembly_reason` explains an exclusion. `assembly_flags` provides diagnostics.
A cell is identified by `(cell_namespace, cell_id)`, a molecule by
`(spot_namespace, spot_id)`, and `obs_name` names a cell in the sample AnnData.

`result.counted_cells()` selects retained cells whose assignment status is
`kept`. `result.matrix()` gives their dense whole-cell count matrix and its cell
rows. The {doc}`reference <assembly-algorithms>` lists the table columns, reasons
and flags.

`save_assembly` writes `tables/fovs`, `tables/cells`, `tables/molecules` and
`tables/counts` as CSV or Parquet, `sample.h5ad`, and the `assembly` section of
`export.json`. `load_assembly` reads the tables and verifies recorded file hashes.
`write_sample_h5ad` can write the AnnData separately: its observations are the
counted cells and `obsm["spatial"]` holds their ownership centroids in X, Y order
in µm. A molecules-only result has no cell observations in its AnnData; retained
molecule totals per gene are in `var["n_molecules_retained"]`.

## Main options

* `ownership`: choose how overlap ownership regions are formed.
* `centroid`: choose the cell centroid used for ownership.
* `duplicate_threshold`: control the duplicate-observation rule.
* `conflicts`: request the conflict and unclaimed diagnostics.
* `table_format` on `save_assembly`: choose the saved table format.

The reference generates the configuration defaults from the code;
{py:class}`~starfinder.assembly.AssemblyConfig` documents the options.

In a Python workflow, `assembly.assignment` names the saved per-FOV assignment;
`assembly.table_format` selects the table format and `assembly.previews` requests
ownership figures. The configuration fields also live in `assembly`.
The `export` block controls which parts of `Dataset.export` run. See the
{doc}`workflow configuration <workflow-configuration>` and the reference's
[workflow keys](assembly-algorithms.md#workflow-keys).

## Limits

Assembly does not estimate FOV positions or fuse images. On the route with
per-FOV assignments it does not merge cells across FOVs, complete cut cells, or
move molecules to another cell. Retention is a choice among observations, not a
reconstruction of a complete cell across a seam.

For input geometry, placement review and identity requirements, see
{py:func}`~starfinder.assembly.assemble`. The specification gives the
[count accounting](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/assembly.md#count-accounting).

## How it is checked

Small hand-built samples check ownership, retention, duplicate observations and
accounting. Synthetic sample scenes exercise the path from per-FOV assignments
to sample tables, and persistence checks read saved tables and AnnData back.
See the [stitching and assembly validation design](https://github.com/wanglab-broad/starfinder/blob/dev/design/validation/stitching-assembly.md).

[Full assembly specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/assembly.md).

```{toctree}
:hidden:

Assembly reference <assembly-algorithms>
```
