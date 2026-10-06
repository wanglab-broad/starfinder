# Assembly algorithm specification

Status: Proposed

This page specifies the numerical behavior of each element of {doc}`assembly-contract`:
the coordinate rule, the ownership regions, retention, the flags and the count accounting,
and the sample-level AnnData built from them. Each has its parameters, units, defaults,
failure behavior and a resource estimate. The current behavior is in
{doc}`assembly-baseline`; the placement it starts from is specified in
{doc}`stitching-contract` and {doc}`stitching-algorithms`, and the per-FOV tables in
{doc}`assignment-contract` and {doc}`assignment-algorithms`. Nothing here is implemented,
and no default here is accepted.

## Evidence

* The golden fixture of `src/python/test/test_workflow_assembly_golden.py` (three FOVs of
  8 × 48 × 64 voxels, nine cells, 24 molecules) and the effect of each ownership option
  on it, computed by `scripts/option_effects.py` into `tables/option_effects.json` in the
  W-326 run directory
  `/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/runs/W-326/20261006T092110Z-48452974`.
* The LN sample1 placement of 2024 as the W-324 parity records give it
  (`/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/runs/W-324/20261006T063540Z-5cb2f421/parity/`:
  `script_tile_config_ln.csv`, `prototype/tile_config_ln/T2_sum.csv`), rasterized at 4 px
  by `scripts/ln_ownership.py` into `tables/ln_ownership.json` (W-326 run directory). That
  derivation reads records only, took 14 s and 1.6 GB peak RSS on one CPU, and measures
  areas, not molecules or cells: no image or molecule table of LN was read.

No resource figure below was measured on an implementation; each is an order of
magnitude from the array sizes, for planning.

## What each element addresses

| Element | Discrepancies of {doc}`assembly-baseline` |
| --- | --- |
| Coordinate rule | 1 integer offsets, 2 truncated centroids, 11 pixel XYZ units |
| Ownership | 3 XY only, 4 `img_col` for Y, 8 grid indices and blanks, 12 rectangles past the tile, 18 order-dependent MATLAB merge |
| Retention | 5 empty branch, 6 molecules of dropped cells, 16 `outside_grid` molecules, 19 unplaced FOVs |
| Flags | 14 invisible duplicates and losses |
| Accounting | 9 no status or reason, 7 diagnostics before the filter |
| AnnData | 13 shared keys, 15 `create_sample_h5ad.py` |

## Coordinate rule

`sample_zyx = origin_zyx + D · (p ⊙ spacing_zyx)` per observation, with `origin` the FOV's
row of `placement.fovs` and `spacing`, `D` from its `ReferenceGrid.metadata`
({doc}`stitching-contract`, "Index to sample coordinates"). With identity direction (check
6 of the entry) this is `origin + p * spacing` per axis, vectorized over the FOV's table.

| Input | `p` | Output columns |
| --- | --- | --- |
| molecule | `z, y, x` (float64, zero-based; sub-voxel kept) | `sample_z, sample_y, sample_x` |
| cell | `centroid_z/y/x`; `expanded_centroid_z/y/x` | `sample_centroid_*`; `sample_expanded_centroid_*` (NaN where the §2.9 value is NaN) |
| plane cell | as above, but `centroid_z` is not a position | `sample_centroid_z` = NaN |

| Parameter | Unit | Default | Rule |
| --- | --- | --- | --- |
| none | µm out, index in | – | float64 throughout; no rounding, no cast |

Failure: a FOV without a finite origin is not mapped (its rows get NaN and the reason
`fov_not_placed`); unknown spacing, a unit other than µm or a non-identity direction
raise `IncompatibleGeometryError` before any mapping. Example: origin `(1.3, 37.4, 0.7)`
µm with spacing 1 µm maps `(3.0, 8.0, 20.0)` to `(4.3, 45.4, 20.7)`; with the LN spacing
`(0.346255, 0.194398, 0.194398)` µm, `p = (2, 10, 0)` from origin `(0, 250.5, 100.25)`
maps to `(0.69251, 252.44398, 100.25)` µm, the example of {doc}`stitching-contract`.

Cost: three multiply-adds per row; memory, three float64 columns per table (24 bytes per
row).

## Ownership

### Footprints and centres

For each placed FOV `f` with grid sizes `(n_y, n_x)` and spacing `(s_y, s_x)`:

* footprint `[o_y − s_y/2, o_y + (n_y − ½)·s_y) × [o_x − s_x/2, o_x + (n_x − ½)·s_x)`,
  half-open, the union of its voxel supports;
* centre `(o_y + (n_y − 1)/2 · s_y, o_x + (n_x − 1)/2 · s_x)`, the centre of the middle
  voxel position.

Both are written to the `fovs` table and the record, so a reader can recompute every
owner.

### O1, nearest centre (recommended default)

For a point `q` (Y and X of a sample position):

1. Candidates: the placed FOVs whose footprint contains `q`.
2. No candidate: no owner.
3. Otherwise the candidate with the smallest `(q_y − c_y)² + (q_x − c_x)²`, in float64.
4. Exact ties (equal float64 squared distances): the candidate first in the placement's
   table order (T1).

For two FOVs of one shape and spacing the boundary is the perpendicular bisector of their
centres, which for a regular grid is the midline of the overlap; three or more FOVs meet
at points, and every covered point has one owner.

**On the golden fixture** (fractional origins `(y, x)`: FOV1 `(0, 0)`, FOV2 `(37.4, 0.7)`,
FOV3 `(−0.3, 55.4)`, in the fixture's unit voxels): the FOV1/FOV3 boundary runs from
x 59.12 at y 8 to x 59.26 at y 34, the FOV1/FOV2 boundary near y 42.2. The retained scene
cells (`tables/option_effects.json`) are, in 3D, FOV1: 1, 2, 5, 9; FOV2: 4, 6; FOV3: 3, 7,
8, every cell once; in 2D FOV3 also retains 9 (its projected centroid lies at x 59.5,
FOV1's at 54.6), so cell 9 is retained twice and both copies get `conflict`. In both, cell
9's molecule at x 64 (outside FOV1's tile) is excluded `cell_not_owner` in 3D, where FOV3's
copy of the cell is not the owner, and retained in 2D with FOV3's copy. The legacy pins
keep cell 4 twice and lose that molecule in both.

**On LN** (`tables/ln_ownership.json`): every covered point has exactly one owner, and no
raster point was an exact tie; the owner differs from the 2024 rectangles on 159,040 px²
(0.13 %) of 119.4 million px², a band along each boundary as wide as the fractional
offset the cast removed (up to 0.998 px in x and 0.926 px in y) plus the areas the
rectangles owned twice or not at all.

### O2, midpoint boxes (option)

Per FOV an axis-aligned box: its footprint, with the edge towards each neighbour moved
to the middle of their overlap: towards a left neighbour `l`,
`x0 = (o_x(l) + n_x(l)·s_x + o_x(f))/2 − s_x/2`, and likewise for the other three sides,
each with its own axis's size. Neighbours are the four grid neighbours (`grid_row`,
`grid_column` of the tile position table), so O2 needs a grid. Ties on an edge go by T2
(half-open). On the fixture: box edges x 59.2 and y 42.2, the same retained sets as O1.
On LN: 33,760 px² owned twice and 2,576 px² unowned, as the legacy rule has at blank
cells and corners.

| Parameter | Unit | Default | Rule |
| --- | --- | --- | --- |
| `ownership` | – | `nearest_center` | `nearest_center` (O1) or `midpoint_box` (O2); `midpoint_box` raises `ValueError` when the tile position table has no grid cells |
| `tie` | – | `table_order` | T1; `half_open` (T2) only with `midpoint_box` |

Failure: a placement with no placed FOV raises `ValueError`; a sample with FOVs of
different `nominal_z` raises (check 8 of the entry).

**Cost.** Candidates are found by binning the footprints on a grid of cells one tile wide
(each point then tests the few FOVs of its bin; four or fewer on a grid with overlaps
under half a tile). Per point a few float64 comparisons: `O(points × candidates)` time,
`O(points)` memory, vectorized per FOV. The tables of a whole sample are the memory that
matters: about 100 bytes per molecule row with the §2.9 columns, so 10 million molecules
take about 1 GB. Not measured.

## Retention

Per FOV, in order:

1. Placement status: no finite origin → every row `excluded`, `fov_not_placed`.
2. Cells: owner of the ownership centroid (`centroid="territory"`: the expanded centroid
   when assign expanded, else the original; `"original"`: always the original) → retained
   when it is the FOV, else `not_owner`.
3. Molecules by `assignment_status`: `assigned`, `excluded_cell` → the cell's status
   (reason `cell_not_owner` for `not_owner`); `unassigned` → owner of the molecule's own
   position; `outside_grid` → `outside_grid`. In `molecules_only` mode every molecule is
   treated as `unassigned`, except `outside_grid` positions (`z, y, x` outside the grid by
   the §2.9 bounds), which are excluded.
4. Counts: the FOV's §2.9 `counts` rows of the retained cells whose §2.9 status is `kept`,
   with `fov_id` and `cell_namespace` added.

| Parameter | Unit | Default | Rule |
| --- | --- | --- | --- |
| `centroid` | – | `territory` | `territory` or `original` |

`territory` follows the legacy filter, which tested the expanded centroid
(`_cell_obs`), and the counts, which are taken over the territory assign sampled. The
two differ only for a cell whose expansion is asymmetric, at a FOV edge or against a
neighbour.

Failure: none beyond the entry's checks; every row gets a status. Cost: one owner query
per cell and per unassigned molecule, one join of molecules to cells by `cell_id`.

## Flags

| Flag | Computation | Cost |
| --- | --- | --- |
| `fov_edge` | The set of label values on the first and last Y row and X column of the territory image (all planes); a cell is flagged when its `cell_id` is in it. | Reads `2 × (n_y + n_x) × n_z` voxels per FOV. |
| `z_edge` | The same on the first and last Z plane, for `volume` and `extended` territories. | `2 × n_y × n_x` voxels per FOV. |
| `conflict` | For each retained cell of FOV `a` and each other placed FOV `b` whose footprint contains its sample centroid: map the centroid into `b`'s index coordinates (`p = (q − origin_b) ⊘ spacing_b`), sample `b`'s territory at `floor(p + 0.5)` (Y and X for a `plane`; Z, Y and X otherwise; outside the grid is no territory). A value that is a retained cell of `b` flags both cells and records each in the other's `conflict_with`. | One voxel read per (cell, overlapping FOV) pair. |
| `unclaimed` | For each cell excluded `not_owner`: the same sampling in the owner FOV; flagged when the value is 0 or a cell the owner did not retain. | One voxel read per excluded cell. |
| `outside_owner_region` | A retained molecule of a retained cell whose own position's owner is another FOV. | One owner query per such molecule. |
| `cell_unclaimed` | A molecule of a cell flagged `unclaimed`. | A join. |

The territories are those of the §2.9 result (`territories` when assign expanded, else
`cell_labels`), read from the saved assignment folder with their SHA-256 checked. Pairs of
FOVs are processed so that at most two territory images are in memory: 4 bytes per voxel
(`uint32`), 8.95 MB per plane of a 1496 × 1496 grid.

| Parameter | Unit | Default | Rule |
| --- | --- | --- | --- |
| `conflicts` | – | `True` | `False` skips `conflict` and `unclaimed` (no label image is read); the columns are then null and the record says so |

Failure: a territory file missing or with another SHA-256 raises `ValueError` naming the
path and both hashes (the §2.9 loader's rule).

On the fixture under O1: in 2D FOV1's cell 6 and FOV3's cell 5 (both scene cell 9) are
flagged `conflict`; in 3D FOV3's cell 5 is excluded `not_owner` and not `unclaimed`
(FOV1 retains a cell at its centroid), and FOV1's cell 6 is flagged `fov_edge` (it touches
x 63).

## Count accounting

The identities of {doc}`assembly-contract` ("Count accounting") are checked on every
result:

1. one row per input key in the sample cell and molecule tables (anti-join of the inputs'
   keys against the sample tables is empty both ways; no duplicated key);
2. per FOV and in total, `inputs = retained + Σ excluded by reason`, for cells and for
   molecules;
3. per molecule, the status rule of "Retention" recomputed from the molecule's
   `assignment_status`, its cell's status and its owner;
4. the sample `counts` equal the concatenation of the per-FOV §2.9 `counts` restricted to
   retained kept cells (a sorted comparison of `(fov_id, cell_id, gene_id, compartment,
   count)`), and `Σ whole` equals the retained `assigned` molecules and the sum of
   `n_molecules` over retained kept cells;
5. the §2.9 compartment identities over the retained cells.

Parameters: none; the quantities are counts of rows and molecules (integers), compared
exactly. A violation raises `ValueError` naming the identity and the first differing key.
Cost: sorts and joins over the tables, `O(n log n)`; no copy beyond the key columns.

## Sample-level AnnData

From the cell table and `counts` ({doc}`assembly-contract`, "Sample-level AnnData"),
through `counts_anndata`:

1. Select the rows with `assembly_status == "retained"` and `status == "kept"`, in table
   order.
2. Build `X` (float64, rows × genes) from the `whole` counts by `(cell_namespace,
   cell_id)` and `gene_id`, 0 where there is no row; the `nucleus` and `cytoplasm` layers
   likewise, NaN rows for cells whose `compartments` is not `available`, only when some
   FOV gave nuclei.
3. `obs` from the selected rows (strings written as anndata expects, as `raw.h5ad` is),
   `obs_names` = `obs_name`, `obsm["spatial"]` = `[sample_centroid_x, sample_centroid_y]`
   of the ownership centroid, `var_names` = `genes`, `uns` = the record as JSON, the unit
   and the frame.
4. `molecules_only`: zero rows, `var["n_molecules_retained"]`.

| Parameter | Unit | Default | Rule |
| --- | --- | --- | --- |
| `annotation` | – | `None` | a mapping of extra `obs` columns for the sample (the workflow passes the annotation row) |

Failure: without `anndata`, `ModuleNotFoundError` naming the `anndata` extra; a
non-unique `obs_name` cannot occur after the entry's checks.

Cost: `X` and each layer are dense float64, as in `raw.h5ad`: 8 bytes × cells × genes
(100,000 cells × 1,000 genes is 0.8 GB per matrix). A sparse `X` would change the
`raw.h5ad` convention and is not proposed here.

## Limitations

* The ownership comparison on LN is an area comparison of two rules on one placement;
  the number of cells or molecules whose owner changes depends on their density there
  and was not measured.
* The fixture is synthetic, one layer, with tiles of one shape; unequal tile shapes and
  stage positions are covered by the rule, not by a pinned case.
* Nearest-centre ownership is a geometric rule. It does not decide which FOV saw a cell
  better (focus, completeness), and no quality-weighted ownership is proposed.
* Duplicates and losses are flagged, not counted as biological cells; their frequency on
  real samples is not known and is a question for the validation design (W-331).
* The resource figures are estimates from array sizes, not measurements.
