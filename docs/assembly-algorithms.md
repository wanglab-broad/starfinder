# Assembly reference

The configuration, statuses, flags, table columns and AnnData fields of sample
assembly. The {doc}`guide <assembly-contract>` explains how to use the stage;
the {doc}`API <api/assembly>` documents its functions and classes.

## Configuration

See {py:class}`~starfinder.assembly.AssemblyConfig` for option details.

```{config-options} starfinder.assembly.AssemblyConfig
```

## Statuses and reasons

`assembly_status` and `assembly_reason` describe retention in the sample tables.

| Status | Meaning |
| --- | --- |
| `retained` | Observation kept in the sample population |
| `excluded` | Observation kept in the tables but excluded from the sample population |

| Reason | Table | Meaning |
| --- | --- | --- |
| `not_owner` | Cells, molecules | Observation not retained by ownership |
| `fov_not_placed` | Cells, molecules | FOV has no placed origin |
| `duplicate_observation` | Cells | Another cell observation retained by the duplicate rule |
| `cell_not_owner` | Molecules | Assigned cell excluded by ownership |
| `cell_duplicate_observation` | Molecules | Assigned cell excluded by the duplicate rule |
| `outside_grid` | Molecules | Molecule outside the input grid |

## Flags

These diagnostics are recorded in `assembly_flags`. The record's `flags` section
lists which were computed.

| Flag | Table | Meaning |
| --- | --- | --- |
| `fov_edge` | Cells | Territory touches a Y or X edge of its FOV |
| `z_edge` | Cells | Territory touches a Z edge |
| `conflict` | Cells | Retained observations conflict across FOVs |
| `unclaimed` | Cells | Ownership-excluded cell has no retained territory at its ownership centroid in the owner FOV, or no owner |
| `outside_owner_region` | Molecules | Retained molecule lies in another FOV's ownership region |
| `cell_unclaimed` | Molecules | Excluded molecule belongs to a cell flagged `unclaimed` |

The [specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/assembly.md#flags)
gives the flag definitions.

## Sample tables

The four DataFrames belong to {py:class}`~starfinder.assembly.AssemblyResult`.
Coordinates prefixed with `sample_`, and FOV origins, footprints and centres,
are in µm. Local coordinates remain in voxel indices.

### FOVs

| Columns | Contents |
| --- | --- |
| `fov_id`, `placement_status` | FOV identity and placement status |
| `origin_z`, `origin_y`, `origin_x` | Placed origin |
| `kind` | Assembly input kind |
| `footprint_y0`, `footprint_y1`, `footprint_x0`, `footprint_x1` | FOV footprint |
| `centre_y`, `centre_x`, `owns_points` | Footprint centre and ownership participation |
| `n_cells`, `n_cells_retained` | Cell observations and retained cell observations |
| `n_molecules`, `n_molecules_retained` | Molecule observations and retained molecule observations |

### Cells

| Columns | Contents |
| --- | --- |
| `fov_id`, `cell_namespace`, `cell_id`, `obs_name` | FOV and sample cell identities |
| `status`, `exclusion_reason` | Assignment status and reason |
| `size_voxels`, `expanded_size_voxels`, `size_physical`, `expanded_size_physical` | Original and expanded territory sizes |
| `centroid_z`, `centroid_y`, `centroid_x`, `expanded_centroid_z`, `expanded_centroid_y`, `expanded_centroid_x` | Local centroids |
| `n_molecules`, `n_nuclei`, `correspondence`, `correspondence_flags`, `compartments` | Assignment measurements |
| `sample_centroid_z`, `sample_centroid_y`, `sample_centroid_x` | Original centroid in sample coordinates |
| `sample_expanded_centroid_z`, `sample_expanded_centroid_y`, `sample_expanded_centroid_x` | Expanded centroid in sample coordinates |
| `ownership_centroid`, `sample_ownership_centroid_z`, `sample_ownership_centroid_y`, `sample_ownership_centroid_x` | Selected centroid and its sample coordinates |
| `owner_fov_id`, `assembly_status`, `assembly_reason`, `assembly_flags` | Ownership and retention diagnostics |
| `conflict_with`, `duplicate_of` | Related cell observations, named by `obs_name` |

### Molecules

| Columns | Contents |
| --- | --- |
| `fov_id`, `spot_namespace`, `spot_id`, `gene_id` | FOV, molecule and gene identities |
| `z`, `y`, `x` | Local molecule coordinates |
| `assignment_status`, `cell_namespace` | Assignment status and cell namespace |
| `voxel_z`, `voxel_y`, `voxel_x`, `cell_id`, `in_expansion`, `original_cell_id`, `nucleus_id`, `compartment` | Assignment columns in a result with cells |
| `sample_z`, `sample_y`, `sample_x` | Sample molecule coordinates |
| `owner_fov_id`, `assembly_status`, `assembly_reason`, `assembly_flags` | Ownership and retention diagnostics |

### Counts

| Columns | Contents |
| --- | --- |
| `fov_id`, `cell_namespace`, `cell_id` | Counted cell identity |
| `gene_id`, `compartment`, `count` | Gene, compartment and integer count |

## Saved files

`save_assembly` writes the sample's export folder; `load_assembly` reads it.

| File | Contents |
| --- | --- |
| `tables/fovs.<fmt>`, `tables/cells.<fmt>`, `tables/molecules.<fmt>`, `tables/counts.<fmt>` | Sample tables; CSV or Parquet |
| `sample.h5ad` | Sample AnnData |
| `export.json`, section `assembly` | Assembly record and file hashes |
| `territories.ome.tif` | Expanded territories from an assignment after fusion, when present |

## AnnData

`write_sample_h5ad` writes the counted cells and their counts. It uses the
`anndata` extra.

| Part | Contents |
| --- | --- |
| `obs_names` | Cell `obs_name` |
| `obs` identities | `sample`, `fov_id`, `cell_namespace`, `cell_id` |
| `obs` measurements | `size_voxels`, `expanded_size_voxels`, `size_physical`, `centroid_z`, `centroid_y`, `centroid_x`, `n_molecules`, `n_nuclei`, `correspondence`, `correspondence_flags`, `compartments` |
| `obs` coordinates and diagnostics | The nine `sample_*centroid_*` columns, `ownership_centroid`, `assembly_flags`, `conflict_with` |
| `obs` annotations | Columns supplied through `write_sample_h5ad(annotation=...)` |
| `var_names` | Ordered sample genes |
| `X` | Whole-cell counts, float64 |
| `layers` | `nucleus` and `cytoplasm` when the assignment inputs include nuclei |
| `obsm["spatial"]` | Ownership centroid X, Y, in µm |
| `uns` | `assembly`, `spatial_unit`, `frame_id`, `spatial_centroid` |
| `var["n_molecules_retained"]` | Retained molecules per gene in molecules-only mode, whose AnnData has no cell rows |

## Workflow keys

The Python workflow's `assembly` block combines configuration fields with the
coordination keys below. See {doc}`workflow configuration <workflow-configuration>`
for a complete workflow.

| Key | API counterpart |
| --- | --- |
| `ownership`, `centroid`, `tie`, `conflicts`, `duplicate_threshold` | `AssemblyConfig` fields, listed above |
| `assignment` | `Dataset.export(assignment=...)` |
| `table_format` | `ExportConfig.table_format` |
| `previews` | Assignment figures requested by the workflow |
