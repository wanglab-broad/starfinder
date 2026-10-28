# Assembly algorithm specification

% Status: Accepted (W-329, 2026-10-07, at 9033d46)

This page specifies the numerical behavior of each element of [Assembly guide: sample coordinates, ownership, retained and excluded populations and the sample tables](assembly.md):
the coordinate rule, the ownership regions, retention, the duplicate rule, the flags and
the count accounting, and the sample-level AnnData built from them. Each has its
parameters, units, defaults, failure behavior and a resource estimate. The behavior before sample assembly is history, in
[Assembly baseline: tile rectangles, global coordinates, the overlap filter and the sample tables](../baselines/assembly.md); the placement it starts from is specified in
[stitching specification](stitching.md) and [stitching algorithm specification](stitching-algorithms.md), and the per-FOV tables in
the [assignment specification](assignment.md) and the [assignment algorithms specification](assignment-algorithms.md). Each element is implemented in
`starfinder.assembly`, and no default here is accepted.

% The behavior before §2.10

## Evidence

* The golden fixture of `src/python/test/test_workflow_assembly_golden.py` (three FOVs of
  8 × 48 × 64 voxels, nine cells, 24 molecules) and the effect of each ownership option
  on it, computed by `scripts/option_effects.py` into `tables/option_effects.json` in the
  run directory of the assembly measurements, kept with the run records outside the repository.
* The LN sample1 placement of 2024 as the stitching parity records give it
  (their `parity/` folder:
  `script_tile_config_ln.csv`, `prototype/tile_config_ln/T2_sum.csv`), rasterized at 4 px
  by `scripts/ln_ownership.py` into `tables/ln_ownership.json` (the assembly measurements' run directory). That
  derivation reads records only, took 14 s and 1.6 GB peak RSS on one CPU, and measures
  areas, not molecules or cells: no image or molecule table of LN was read.

% The assembly measurements: the W-326 run directory
% `/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/runs/W-326/20261006T092110Z-48452974`.
% The stitching parity records: W-324,
% `/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/runs/W-324/20261006T063540Z-5cb2f421/parity/`.

No resource figure below was measured on an implementation; each is an order of
magnitude from the array sizes, for planning.

## What each element addresses

| Element | Discrepancies of [Assembly baseline: tile rectangles, global coordinates, the overlap filter and the sample tables](../baselines/assembly.md) |
| --- | --- |
| Coordinate rule | 1 integer offsets, 2 truncated centroids, 11 pixel XYZ units |
| Ownership | 3 XY only, 4 `img_col` for Y, 8 grid indices and blanks, 12 rectangles past the tile, 18 order-dependent MATLAB merge |
| Retention | 5 empty branch, 6 molecules of dropped cells, 16 `outside_grid` molecules, 19 unplaced FOVs |
| Duplicate rule | 14 duplicates (decided on 2026-10-09) |
| Flags | 14 invisible duplicates and losses |
| Accounting | 9 no status or reason, 7 diagnostics before the filter |
| AnnData | 13 shared keys, 15 `create_sample_h5ad.py` |

## Coordinate rule

`sample_zyx = origin_zyx + D · (p ⊙ spacing_zyx)` per observation, with `origin` the FOV's
row of `placement.fovs` and `spacing`, `D` from its `ReferenceGrid.metadata`
([stitching specification](stitching.md), "Index to sample coordinates"). With identity direction (check
6 of the entry) this is `origin + p * spacing` per axis, vectorized over the FOV's table.

| Input | `p` | Output columns |
| --- | --- | --- |
| molecule | `z, y, x` (float64, zero-based; sub-voxel kept); on a one-plane grid `z` is read as 0, the plane | `sample_z, sample_y, sample_x` |
| cell | `centroid_z/y/x`; `expanded_centroid_z/y/x` | `sample_centroid_*`; `sample_expanded_centroid_*` (NaN where the assignment value is NaN) |
| plane cell | as above, but `centroid_z` is not a position | `sample_centroid_z` = NaN |
| ownership centroid | the expanded centroid when `centroid="territory"` and assign expanded, else the original | `ownership_centroid` (`expanded` or `original`) and `sample_ownership_centroid_*`, a copy of the pair it names |

| Parameter | Unit | Default | Rule |
| --- | --- | --- | --- |
| none | µm out, index in | – | float64 throughout; no rounding, no cast |

Failure: a FOV without a finite origin is not mapped (its rows get NaN and the reason
`fov_not_placed`); unknown spacing, a unit other than µm or a non-identity direction
raise `IncompatibleGeometryError` before any mapping. Example: origin `(1.3, 37.4, 0.7)`
µm with spacing 1 µm maps `(3.0, 8.0, 20.0)` to `(4.3, 45.4, 20.7)`; with the LN spacing
`(0.346255, 0.194398, 0.194398)` µm, `p = (2, 10, 0)` from origin `(0, 250.5, 100.25)`
maps to `(0.69251, 252.44398, 100.25)` µm, the example of [stitching specification](stitching.md).

Cost: three multiply-adds per row; memory, three float64 columns per table (24 bytes per
row).

## Ownership

### Footprints and centres

For each placed FOV `f` with grid sizes `(n_y, n_x)` and spacing `(s_y, s_x)`:

* footprint `[o_y − s_y/2, o_y + (n_y − ½)·s_y) × [o_x − s_x/2, o_x + (n_x − ½)·s_x)`,
  half-open, the union of its voxel supports;
* centre `(o_y + (n_y − 1)/2 · s_y, o_x + (n_x − 1)/2 · s_x)`, the centre of the middle
  voxel position.

Both are written to the `fovs` table and the record, for every placed FOV, placeholders
included, with `owns_points`, so a reader can recompute every owner. Only owning FOVs
([Assembly guide: sample coordinates, ownership, retained and excluded populations and the sample tables](assembly.md), "Terms": placed, with an input, `image_supported` or
`metadata_only`) take part in the rules below; a `placeholder`'s footprint is kept for the
record and never decides an owner.

### O1, nearest centre (recommended default)

For a point `q` (Y and X of a sample position):

1. Candidates: the owning FOVs whose footprint contains `q`, restricted to the
   `candidates` argument when it is given.
2. No candidate: no owner.
3. Otherwise the candidate with the smallest `(q_y − c_y)² + (q_x − c_x)²`, in float64.
4. Exact ties (equal float64 squared distances): the candidate first in the placement's
   table order (T1).

For two FOVs of one shape and spacing the boundary is the perpendicular bisector of their
centres, which for a regular grid is the midline of the overlap; three or more FOVs meet
at points, and every point covered by an owning FOV has one owner.

**A placeholder in the grid.** Its footprint is not a candidate, so a point there goes to
the nearest owning FOV that covers it, or to none. A placeholder at a corner of the grid
has at most three neighbours (two sides and one diagonal), one at an edge at most five,
one inside the grid at most eight. Each part of its footprint that neighbours cover goes
to the nearest of them; where two neighbours cover a point, as in a corner overlap, the
nearest centre decides as anywhere else. The rest of its footprint, which no owning FOV
covers (its middle, and at the grid's border the strip along the outer edge), has no
owner. With overlaps below half a tile, that middle is never empty. Neither the golden
fixture nor the LN records have a placeholder (LN's blank cells are padding views), so no
figure below changes; `test_workflow_assembly_golden.py` still pins the legacy behavior,
whose rectangles include every grid cell.

**On the golden fixture** (fractional origins `(y, x)`: FOV1 `(0, 0)`, FOV2 `(37.4, 0.7)`,
FOV3 `(−0.3, 55.4)`, in the fixture's unit voxels): the FOV1/FOV3 boundary runs from
x 59.12 at y 8 to x 59.26 at y 34, the FOV1/FOV2 boundary near y 42.2. The retained scene
cells (`tables/option_effects.json`) are, in 3D, FOV1: 1, 2, 5, 9; FOV2: 4, 6; FOV3: 3, 7,
8, every cell once; in 2D FOV3 also retains 9 (its projected centroid lies at x 59.5,
FOV1's at 54.6), so cell 9 is retained twice by ownership and both copies get `conflict`
with the duplicate rule off; with the default rule FOV3's copy (42 voxels, sharing 0.90 of
them with FOV1's 87) is excluded `duplicate_observation` and neither is flagged. In both,
cell 9's molecule at x 64 (outside FOV1's tile) is excluded `cell_not_owner` in 3D, where
FOV3's copy of the cell is not the owner, and retained in 2D with FOV3's copy (with the
rule off; excluded `cell_duplicate_observation` with the default rule). The legacy pins
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
`grid_column` of the tile position table) that are owning FOVs (and candidates, when
`candidates` is given), so O2 needs a grid; towards a placeholder, the edge stays at the
FOV's own footprint edge, and only owning FOVs have boxes. Ties on an edge go by T2
(half-open). On the fixture: box edges x 59.2 and y 42.2, the same retained sets as O1.
On LN: 33,760 px² owned twice and 2,576 px² unowned, as the legacy rule has at blank
cells and corners.

| Parameter | Unit | Default | Rule |
| --- | --- | --- | --- |
| `ownership` | – | `nearest_center` | `nearest_center` (O1) or `midpoint_box` (O2); `midpoint_box` raises `ValueError` when the tile position table has no grid cells |
| `tie` | – | `table_order` | T1; `half_open` (T2) only with `midpoint_box` |
| `candidates` (argument of `ownership_map`) | – | `None` (every owning FOV) | a non-empty collection of `fov_id`s, each an owning FOV of the placement; restricts the FOVs that may own a point |

Failure: a placement with no owning FOV raises `ValueError` (also when every FOV is a
`placeholder`); `candidates` that is empty or names a FOV that is not an owning FOV (not in
the placement, not placed, a `placeholder` or without a grid) raises `ValueError` naming
it; a sample with FOVs of different `nominal_z` raises (check 8 of the entry).

**Cost.** Candidates are found by binning the footprints on a grid of cells one tile wide
(each point then tests the few FOVs of its bin; four or fewer on a grid with overlaps
under half a tile). Per point a few float64 comparisons: `O(points × candidates)` time,
`O(points)` memory, vectorized per FOV. The tables of a whole sample are the memory that
matters: about 100 bytes per molecule row with the assignment columns, so 10 million molecules
take about 1 GB. Not measured.

% with the §2.9 columns

## Retention

Per FOV, in order:

1. Placement status: no finite origin → every row `excluded`, `fov_not_placed`.
2. Cells: owner of the ownership centroid, `sample_ownership_centroid_y/x`
   (`centroid="territory"`: the expanded centroid when assign expanded, else the original;
   `"original"`: always the original) → retained when it is the FOV, else `not_owner`
   (for a `placeholder`, always `not_owner`, with a null owner where no owning FOV covers
   the centroid).
3. Molecules by `assignment_status`: `assigned`, `excluded_cell` → the cell's status
   (reason `cell_not_owner` for `not_owner`); `unassigned` → owner of the molecule's own
   position; `outside_grid` → `outside_grid`. In `molecules_only` mode every molecule is
   treated as `unassigned`, except `outside_grid` positions (`z, y, x` outside the grid by
   the assignment bounds; `y, x` only on a one-plane grid), which are excluded.
4. Counts: the FOV's assignment `counts` rows of its counted cells (retained, assignment status
   `kept`; [Assembly guide: sample coordinates, ownership, retained and excluded populations and the sample tables](assembly.md), "Statuses and reasons"), with `fov_id` and
   `cell_namespace` added. A retained `excluded_no_nucleus` cell has no assignment count rows and
   gets none.

% Step 3: the §2.9 bounds. Step 4: §2.9 `counts`, §2.9 status, no §2.9 count rows.

| Parameter | Unit | Default | Rule |
| --- | --- | --- | --- |
| `centroid` | – | `territory` | `territory` or `original` |

`territory` follows the legacy filter, which tested the expanded centroid
(`_cell_obs`), and the counts, which are taken over the territory assign sampled. The
two differ only for a cell whose expansion is asymmetric, at a FOV edge or against a
neighbour.

Failure: none beyond the entry's checks; every row gets a status. Cost: one owner query
per cell and per unassigned molecule, one join of molecules to cells by `cell_id`.

## Duplicate rule

Decided by Jiahao on 2026-10-09 ([Cells seen by two FOVs: measurements and a proposal](../proposals/cells-across-fovs.md), question 3, option b)
and implemented on that date: rule 8 of [Assembly guide: sample coordinates, ownership, retained and excluded populations and the sample tables](assembly.md), "Retention". **Order of the
steps inside `assemble`:** retention by the ownership centroid (steps 1 and 2 above); then
the duplicate rule among the cells retained so far; then the flags ("Flags"), among the
cells retained after the rule, by their definitions; then the molecules, which follow
their cell (step 3), so that the molecules of an observation the rule excludes are
excluded `cell_duplicate_observation`.

% implemented by W-400

**The measure.** Each FOV's retained-cell territory (the territory assign sampled, with
every value that is not one of its retained cells set to 0) is placed on the sample grid
at scale 1 as the label export places it ([sample-output-contract](sample-output.md), "Assembled label
export"): in Y and X the label grid at factor 1 (spacing `s`, origin the smallest origin
of the included FOVs), in Z the fusion grid's Z (spacing `s_z`, origin the smallest
`origin_z`; Y and X only when the territories are `plane` or the FOVs have one plane), each
FOV at its origin, a grid voxel `v` showing the FOV voxel `floor(c + ½)`, `c = (o_grid +
v·s − origin)/s`, where `0 ≤ c ≤ n − 1` (a margin of 1e-9 voxel). A cell's size is the
number of grid voxels it shows; two cells of two FOVs share the grid voxels that show both.
The share of a pair is its shared voxels divided by the size of the smaller cell (0 when
that size is 0), computed in float64 and compared with the threshold as `share ≥ t`.

**The decision.** The retained cells are taken in the order of decreasing size, ties by
placement table order and then `cell_id`. A cell stays unless it is paired, at a share of
at least `t`, with a cell of another FOV that already stays; then it is excluded
`duplicate_observation` and `duplicate_of` names the first such cell in that order. A
cell paired with several cells (one cell of FOV A over two of FOV B, a cell seen by three
or four FOVs at a corner, a chain in which A's cell pairs with B's and B's with C's) is
decided by the same order: in a chain whose first cell is the largest, the middle cell is
excluded and the third stays, because its only partner does not stay. Neither the order of
the inputs nor the order of the FOV pairs changes the result.

| Parameter | Unit | Default | Rule |
| --- | --- | --- | --- |
| `duplicate_threshold` | share of the smaller territory's voxels | `0.5` | a number above 0 and at most 1, or null (the rule off); a bool, a string or any other value raises `ValueError` naming the field |

On the two cellular blocks measured for that proposal, 0.5 picks exactly the cells that `conflict` marked
([Cells seen by two FOVs: measurements and a proposal](../proposals/cells-across-fovs.md), measurement 1 and question 3). The kept observation is
the larger: a cut observation is smaller than an uncut one. The rule removes a second
observation; it does not complete a cell cut by a FOV edge.

% the two cellular blocks of W-399

Failure: with the rule on and retained cells in two or more FOVs, FOVs of different
spacings or of mixed `plane` and `volume` territories among them raise `ValueError` (the
label export refuses the same inputs); a territory file missing or with another SHA-256
raises as for the flags. Cost: the territories are read two FOVs at a time (each FOV
once, and once more for each earlier FOV in table order whose extent overlaps it), the
shared voxels counted plane by plane on the box where both extents overlap, so at most two
territory images and one plane of each box are in memory; time `O(voxels of the overlap
boxes)`. The rule reads the territories also when `conflicts=False`. Measured by its implementation on
the implementation, unlike the estimates elsewhere on this page, on a synthetic 3 × 3 grid
of 16×256×256 tiles with cells along every seam (1,158 cell observations, 45,775 molecule
observations, saved assignment folders, one CPU): `assemble` took 1.58 s without the rule
and 2.08 s with it, with a peak RSS of 254 MB to 264 MB in both.

% Measured by W-400

## Flags

| Flag | Computation | Cost |
| --- | --- | --- |
| `fov_edge` | The set of label values on the first and last Y row and X column of the territory image (all planes); a cell is flagged when its `cell_id` is in it. | Reads `2 × (n_y + n_x) × n_z` voxels per FOV. |
| `z_edge` | The same on the first and last Z plane, for `volume` and `extended` territories. | `2 × n_y × n_x` voxels per FOV. |
| `conflict` | For each retained cell of FOV `a` and each other placed FOV `b` whose footprint contains its ownership centroid (`sample_ownership_centroid_*`): map that centroid into `b`'s index coordinates (`p = (q − origin_b) ⊘ spacing_b`), sample `b`'s territory at `floor(p + 0.5)` (Y and X for a `plane`; Z, Y and X otherwise; outside the grid is no territory). A value that is a retained cell of `b` flags both cells and records each in the other's `conflict_with`. | One voxel read per (cell, overlapping FOV) pair. |
| `unclaimed` | For each cell excluded `not_owner`: flagged when its ownership centroid has no owner (a cell of a `placeholder` outside every owning footprint); otherwise the same sampling in the owner FOV, flagged when the value is 0 or a cell the owner did not retain. | One voxel read per excluded cell with an owner. |
| `outside_owner_region` | A retained molecule of a retained cell whose own position's owner is another FOV. | One owner query per such molecule. |
| `cell_unclaimed` | A molecule of a cell flagged `unclaimed`. | A join. |

The territories are those of the assignment result (`territories` when assign expanded, else
`cell_labels`), read from the saved assignment folder with their SHA-256 checked. The
record (the `assembly` section of `export.json`) holds, per FOV, the path of the file
read, as the assignment loader resolved it, beside that SHA-256 ([Assembly guide: sample coordinates, ownership, retained and excluded populations and the sample tables](assembly.md), "What assemble records"), so a
later reader, such as the label export, opens the file directly. Pairs of
FOVs are processed so that at most two territory images are in memory: 4 bytes per voxel
(`uint32`), 8.95 MB per plane of a 1496 × 1496 grid. A retained cell is one the cell
table retains after the duplicate rule: a cell the rule excluded is neither flagged
`conflict` nor tested as a retained cell of its FOV.

% the §2.9 result; the §2.9 loader

| Parameter | Unit | Default | Rule |
| --- | --- | --- | --- |
| `conflicts` | – | `True` | `False` skips `conflict` and `unclaimed` (no label image is read for them; the duplicate rule, when on, still reads the territories); the columns are then null and the record says so |

Failure: a territory file missing or with another SHA-256 raises `ValueError` naming the
path and both hashes (the assignment loader's rule).

% the §2.9 loader's rule

On the fixture under O1: in 2D FOV1's cell 6 and FOV3's cell 5 (both scene cell 9) are
flagged `conflict` with the duplicate rule off (with the default rule FOV3's cell 5 is
excluded `duplicate_observation` and neither is flagged); in 3D FOV3's cell 5 is excluded `not_owner` and not `unclaimed`
(FOV1 retains a cell at its ownership centroid), and FOV1's cell 6 is flagged `fov_edge` (it touches
x 63).

## Count accounting

The identities of [Assembly guide: sample coordinates, ownership, retained and excluded populations and the sample tables](assembly.md) ("Count accounting") are checked on every
result:

1. one row per input key in the sample cell and molecule tables (anti-join of the inputs'
   keys against the sample tables is empty both ways; no duplicated key);
2. per FOV and in total, `inputs = retained + Σ excluded by reason`, for cells and for
   molecules;
3. per molecule, the status rule of "Retention" recomputed from the molecule's
   `assignment_status`, its cell's status and its owner;
4. the sample `counts` equal the concatenation of the per-FOV assignment `counts` restricted to
   the counted cells (a sorted comparison of `(fov_id, cell_id, gene_id, compartment,
   count)`), and `Σ whole` equals the retained `assigned` molecules and the sum of
   `n_molecules` over the counted cells;
5. the assignment compartment identities over the counted cells.

% 4: the per-FOV §2.9 `counts`; 5: the §2.9 compartment identities

Placeholders change none of these identities. A placeholder's observations, when it has an
input, all get excluded reasons (`not_owner`, `cell_not_owner`, `outside_grid`), and
identity 3 recomputes them with the placeholder never being the owner.

Parameters: none; the quantities are counts of rows and molecules (integers), compared
exactly. A violation raises `ValueError` naming the identity and the first differing key.
Cost: sorts and joins over the tables, `O(n log n)`; no copy beyond the key columns.

## Sample-level AnnData

From the cell table and `counts` ([Assembly guide: sample coordinates, ownership, retained and excluded populations and the sample tables](assembly.md), "Sample-level AnnData"):
`write_sample_h5ad(result, path, *, annotation=None)` prepares the sample's `obs` and the
other parts (steps 1 and 3 to 4) and passes them to `write_h5ad(path, obs, keys, counts,
genes, *, layers, uns, var, obsm)` of `starfinder.assignment`, which builds the AnnData
(step 2) and writes it at `path`:

1. Select the counted cells: the cell-table rows with `assembly_status == "retained"` and
   assignment `status == "kept"`, in table order. A retained cell whose assignment status is
   `excluded_no_nucleus` is not selected; it stays in the cell table with its assembly
   status. Check that the number selected equals `adata.n_obs` at the end.
2. Build `X` (float64, rows × genes) from the `whole` counts by `(cell_namespace,
   cell_id)` and `gene_id`, 0 where there is no row; the `nucleus` and `cytoplasm` layers
   likewise, NaN rows for cells whose `compartments` is not `available`, only when some
   FOV gave nuclei.
3. `obs` from the selected rows (strings written as anndata expects, as `raw.h5ad` is),
   with every sample-coordinate column copied under its own name; `obs_names` =
   `obs_name`; `obsm["spatial"]` = the columns `[sample_ownership_centroid_x,
   sample_ownership_centroid_y]` of the selected rows, which equal
   `sample_expanded_centroid_x/y` where `ownership_centroid` is `expanded` and
   `sample_centroid_x/y` where it is `original`; `var_names` = `genes`; `uns` = the record
   as JSON, the unit, the frame and `spatial_centroid` (the two columns `obsm["spatial"]`
   copies, the `centroid` setting and the row count per `ownership_centroid` value).
4. `molecules_only`: zero rows, `var["n_molecules_retained"]`, passed to `write_h5ad` as
   `var`; with no row in `obs`, `write_h5ad` writes `X` of shape `(0, n_genes)`.

% Step 1: §2.9 `status == "kept"`; a retained cell whose §2.9 status is

| Parameter | Unit | Default | Rule |
| --- | --- | --- | --- |
| `annotation` | – | `None` | a mapping of extra `obs` columns for the sample (the workflow passes the annotation row) |

Failure: without `anndata`, `ModuleNotFoundError` naming the `anndata` extra, from
`write_h5ad` and `write_sample_h5ad` alike; a non-unique `obs_name` cannot occur after the
entry's checks; a `keys` row count that differs from `obs`, or a key column missing from
`counts`, raises `ValueError` in `write_h5ad`. The workflow adapter calls the same
`write_h5ad` for `raw.h5ad`, with the `obs` that `_cell_obs` prepares after its tile-box
filter, and the file's content is as before.

Cost: `X` and each layer are dense float64, as in `raw.h5ad`: 8 bytes × cells × genes
(100,000 cells × 1,000 genes is 0.8 GB per matrix). A sparse `X` would change the
`raw.h5ad` convention and is not proposed here.

## Golden tests and the validation checks

% Golden tests and the §2.10 checks

The golden and legacy tests hold the behavior of the rules this page replaces.
`test_workflow_assembly_golden.py`, `test_workflow_stitching_golden.py`,
`test_assignment_golden.py` and `test_assignment_workflow.py` pass unchanged; their
helpers stay as the frozen legacy reference.

The checks of the elements above are rows AS1 to AS17 and WF1 of
[Validation design: placement, assembly and sample outputs](../validation/stitching-assembly.md) ("Checks"), each named with its tests in
"Implemented checks".

% "Implemented checks (W-373)"

## Limitations

* The ownership comparison on LN is an area comparison of two rules on one placement;
  the number of cells or molecules whose owner changes depends on their density there
  and was not measured.
* The fixture is synthetic, one layer, with tiles of one shape; unequal tile shapes,
  stage positions and placeholder FOVs are covered by the rule, not by a pinned case.
* Nearest-centre ownership is a geometric rule. It does not decide which FOV saw a cell
  better (focus, completeness), and no quality-weighted ownership is proposed.
* Losses are flagged, and duplicates are flagged or, at or above the duplicate threshold,
  resolved by the duplicate rule; neither is counted as biological cells, and their
  frequency on real samples is not known and is a question for the validation design
  ([Validation design: placement, assembly and sample outputs](../validation/stitching-assembly.md)).
* The resource figures are estimates from array sizes, not measurements.

% the validation design (W-331)
