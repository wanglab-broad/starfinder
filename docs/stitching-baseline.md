# Stitching baseline: stage grid, BigStitcher macro, Fiji run and tile configuration

% Status: Accepted (W-329, 2026-10-07, at 9033d46)

Note: {doc}`stitching-contract` and {doc}`stitching-algorithms` were accepted at their review gate (2026-10-07) and are implemented in `starfinder.stitching`; this page keeps the text of its revision.

% Note (W-583); accepted at W-329 (2026-10-07)

This page records how stitching behaves at revision `8e7d256` (branch
`runner/s210-spec-20261006`, on `dev` after the assignment work), before the stitching
and assembly work changes it. It is the reference that the golden test
`src/python/test/test_workflow_stitching_golden.py` pins. The proposed replacement is
described in {doc}`stitching-contract` and {doc}`stitching-algorithms`; neither is
accepted. Paths are relative to the repository root, and line numbers are at `8e7d256`.
The assembly half of the stitching and assembly work (sample coordinates of molecules and cells, overlap ownership,
the sample tables) is recorded by {doc}`assembly-baseline`; this page names it only where the stitching
scripts reach into it, and marks which parts of `create_tile_config.py` belong to it.

% on `dev` after the §2.9 work, before the Chapter II §2.10 work changes it; the assembly half of §2.10 is recorded by W-326

Evidence, cited by path and not repeated, kept with the run records outside the repository:

* The stitching input inventory:
  the Fiji baseline `20261005T1930Z-fiji-baseline/fiji-baseline.md`, the input
  inventory `20261005T2110Z-input-inventory/inputs.md` and the stage-axis mapping
  `20261005T2050Z-stage-axis-mapping/positions.md`.
* The stitching parity runs, whose run directory holds a `stitching-manifest.json`:
  `notes.md` section 4 and the parity outputs under `parity/` (`parity_report.json`, the
  script outputs `script_*.csv`, and the step-by-step prototype `prototype/<case>/`).

% The stitching input inventory: W-323 (`/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/runs/W-323/`).
% The stitching parity runs: W-324, run directory
% `/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/runs/W-324/20261006T063540Z-5cb2f421`
% (found with `timeout 30 find …/runs/W-324 -maxdepth 2 -name stitching-manifest.json`).
% On this page "the input inventory" is W-323 and "the parity runs" (their notes, outputs and prototype) are W-324.

## Where stitching lives today

The Python package has no stitching code. Everything is a Snakemake rule, a workflow
script, Fiji, or a MATLAB example:

| Piece | Where | Engine |
| --- | --- | --- |
| Per-sample stage file | rule `create_sample_maf` (`workflow/rules/utils.smk:8-18`), `workflow/scripts/create_sample_maf.py` | workflow environment (BeautifulSoup) |
| Stage file to a row and column grid, blank padding, tile copies | rule `stitching_preparation` (`workflow/rules/stitching.smk:13-25`), `workflow/scripts/stitching_preparation.py` | workflow environment |
| BigStitcher macro | rule `create_BigStitcher_macro` (`stitching.smk:29-37`), `workflow/scripts/create_BigStitcher_macro_1.py` | workflow environment |
| Fiji run | rule `run_BigStitcher_macro` (`stitching.smk:41-56`), a shell command | Fiji with BigStitcher |
| `dataset.xml` to the tile configuration and the non-overlap rectangles | rule `create_tile_config` (`stitching.smk:60-70`), `workflow/scripts/create_tile_config.py` | workflow environment (plotly) |
| The stitching input image | rule `rotate_nuclei` (`workflow/rules/registration-py.smk:95-114`; the same lines in `registration.smk:87-106`): `images/DAPI/{fovID}.tif` | workflow environment |
| Consumers of the tile configuration | `reads_assignment` through the workflow adapter (global coordinates and the overlap filter; {doc}`assignment-baseline`) | Python package |
| Legacy Grid/Collection path | `example/sequential_workflow/reads_stitching.m`, `src/matlab/ParseFijiTileConfiguration.m`, `example/sequential_workflow/stitch_clustermap.py`; Fiji Grid/Collection run by hand | MATLAB, Python, Fiji |
| Unused | `workflow/scripts/create_BigStitcher_macro.py` (an N5 variant no rule calls) and `run_fiji_macros` (`workflow/rules/common.smk:174-178`, no caller) | none |

## Files each rule reads and writes

Note: these rules, `create_sample_maf` included, now run with `backend: matlab` only; the Python stitching block reads an explicit `stitching.positions.stage_file` and `INPUT/fovs.csv`, and writes no `documents/maf/` copy.

% Note (W-566)

All paths are relative to `OUTPUT = {root_output_path}/{dataset_id}/{output_id}`.

| Rule | Reads | Writes (declared) | Writes besides |
| --- | --- | --- | --- |
| `create_sample_maf` | `documents/sample-annotation.csv`, `documents/raw.maf` | `documents/maf/{sample}.maf` per sample (the stage points of the sample's FOV range) | none |
| `stitching_preparation` | `documents/sample-annotation.csv`, `documents/maf/{sample}.maf`, `images/DAPI/{fovID}.tif` of the sample | `images/fused/{sample}/blank.tif`, `grid.csv`, `grid.png` | `images/fused/{sample}/DAPI/tile_{j}.tif`, one copy per grid cell |
| `create_BigStitcher_macro` | `images/fused/{sample}/grid.csv` | `images/fused/{sample}/BigStitcher_macro.ijm` | none |
| `run_BigStitcher_macro` | the macro and `grid.csv`; the macro itself reads `images/fused/{sample}/DAPI/*.tif` | `images/fused/{sample}/DAPI/dataset.xml` | `dataset.h5` and BigStitcher's `dataset.xml~*` backups in the same folder; Java's temporary files there (`-Djava.io.tmpdir`) |
| `create_tile_config` | `dataset.xml`, `grid.csv` | `output/tile_config_{sample}.csv`, `output/tile_config_{sample}.html` | none |

What Snakemake tracks follows from these declarations:

* **Changed inputs rerun the rule.** `stitching_preparation` declares the annotation, the
  sample's stage file and the sample's DAPI images (`get_stitching_input`,
  `{DAPI_DIR}/{fovID}.tif` for every FOV of the sample, `stitching.smk:8-9`, `:14-17`), so a changed
  DAPI image makes the rule run again, and the script then copies every tile again
  (`stitching_preparation.py:101-116`). The new `grid.csv` in turn reruns
  `create_BigStitcher_macro`, `run_BigStitcher_macro` and `create_tile_config`, which
  declare it as an input. The declared names follow `fov_id_pattern`, while the script
  reads `images/DAPI/Position{id:03}.tif` (line 113); the two coincide only for the pattern
  `Position{i:03}`.
* **Undeclared outputs are not tracked.** The tile copies under
  `images/fused/{sample}/DAPI/`, `dataset.h5` and BigStitcher's backups are not declared
  outputs (`stitching.smk:18-21`, `:44-45`), so Snakemake does not notice when one of
  them is removed or altered and does not regenerate it. `run_BigStitcher_macro` does not
  declare the tile copies as inputs either (`stitching.smk:42-43`): a tile copy changed by
  hand does not rerun Fiji.

## `stitching_preparation.py` step by step

The steps are named G1 to G5 as in the parity runs' prototype (`scripts/parity_prototype.py`).

% the W-324 prototype

| Step | Lines | Behavior at `8e7d256` |
| --- | --- | --- |
| G1 Read | 15-17 | `pd.read_xml` of the stage file; keeps `StageXPos`, `StageYPos` (metres), `PositionID`, `Sections`. `n_layers` is the largest `Sections`. Every other attribute (`StageZPos`, `StackBegin`, `PositionIdentifier`) is discarded. |
| G2 Step | 20-21 | The step is the most frequent value of `StageYPos[i+1] − StageYPos[i]` over consecutive rows of the file: `pd.value_counts(temp).index.values[0]`. Only Y is used; X is assumed to have the same step. The differences are exact floats, so two steps that differ in the last bits count separately, and a tie goes to the first value in the counts' order (the parity case `maf_2x3_jitter`, two ties). A scan whose most frequent difference is negative gives a negative step. **pandas 3.0.0, which the lock has for Python 3.11 to 3.13, has no `pd.value_counts`**: the script raises `AttributeError` there (the parity runs' notes, section 4). |
| G3 Cells | 24-30 | Relative position per axis `int((pos − min)/step + 0.5)`: rounding half up, since the argument is never negative. The grid shape is `int((max − min)/step + 1.5)` per axis. Stage X becomes the column and stage Y the row, with no sign change or swap: the axis mapping is fixed here. |
| G4 Matrix | 32-38 | `coo_matrix((PositionID, (row, col)))`, then `toarray()`: an empty cell is 0, and two positions that round into one cell are summed into one id (none in the parity cases). Lines 40-42 keep a commented-out rotation and flip. |
| G5 Grid | 44-49 | `DataFrame(matrix).unstack(1)` lists the cells column by column (all rows of column 0, then column 1). `grid.csv` has the index `j`, `col`, `row`, `id` (the PositionID, 0 for an empty cell) and `grid` = `tile_{row}_{col}`. `j` is the tile number every later step uses. |
| Plot | 51-74 | `grid.png`: occupied cells, ids and names; `plt.show()` under a non-interactive backend only warns. |
| Blank | 91-93 | `blank.tif` = zeros of shape `(n_layers, img_col, img_row)`, uint8: X before Y, which is the transpose of a ZYX tile when `img_col ≠ img_row`. |
| Tile copies | 95-117 | For each `j`, copy `images/DAPI/Position{id:03}.tif` (or `blank.tif` for id 0) to `images/fused/{sample}/DAPI/tile_{j}.tif`. The folder `images/DAPI` and the name `Position%03d` are fixed in the script; `fov_id_pattern` is not read. The id is the stage file's `PositionID`, which is assumed to equal the FOV number. |

On the LN sample1 stage file the unchanged script, with `pd.value_counts` restored,
writes the 2024 `grid.csv` byte for byte: 8 columns by 9 rows, 64 acquired positions
and 8 blank cells, step 262.0 µm (the parity runs' `parity_report.json`, `grid_ln`).

% The table above: G2 W-324 `maf_2x3_jitter`, W-324 notes section 4; G4 the W-324 cases. Here: W-324 `parity_report.json`.

## The macro: `create_BigStitcher_macro_1.py`

The rule writes one Fiji macro per sample. Every value except the paths, the voxel size
and the grid size is a literal in the script (lines 10-37):

| Command | Parameter | Value | From |
| --- | --- | --- | --- |
| Define dataset | `define_dataset` | `[Automatic Loader (Bioformats based)]`, `project_filename=dataset.xml` | literal |
| | `path` | `{OUTPUT}/images/fused/{sample}/DAPI/*.tif`, `exclude=10`, `pattern_0=Tiles` | config paths |
| | voxel size | `voxel_size_x = voxel_size_y = voxel_size_xy`, `voxel_size_z = voxel_size_z`, `voxel_size_unit=µm` | config |
| | grid | `[Move Tile to Grid (Macro-scriptable)]`, `grid_type=[Down & Right             ]`, `tiles_x = max(col)+1`, `tiles_y = max(row)+1`, `tiles_z=1` | `grid.csv` |
| | overlap | `overlap_x_(%)=10 overlap_y_(%)=10 overlap_z_(%)=10` | literal |
| | loading | `keep_metadata_rotation`, `how_to_load_images=[Re-save as multiresolution HDF5]`, `load_raw_data_virtually`, `dataset_save_path={OUTPUT}/images/fused/{sample}/DAPI` | literal, config paths |
| | pyramid | `manual_mipmap_setup`, `subsampling_factors=[{2,2,2}, {4,4,4}]`, `hdf5_chunk_sizes=[{16,16,16}, {16,16,16}]`, `timepoints_per_partition=1`, `setups_per_partition=0`, `use_deflate_compression` | literal |
| Calculate pairwise shifts | selection | all angles, channels, illuminations, tiles and timepoints | literal |
| | method | `[Phase Correlation]`, `downsample_in_x=4 downsample_in_y=4 downsample_in_z=4` | literal |
| Filter pairwise shifts | link filter | `filter_by_link_quality min_r=0.7 max_r=1`; `max_shift_in_x/y/z=0` and `max_displacement=0` (off) | literal |
| Optimize globally and apply shifts | tolerances | `relative=2.5 absolute=3.5` (pixels) | literal |
| | strategy | `[Two-Round using Metadata to align unconnected Tiles and iterative dropping of bad links]` | literal |
| | grouping | `show_expert_grouping_options`, timepoints and angles individually, channels and illuminations grouped, `how_to_treat_tiles=compare`, `fix_group_0-0` (the first tile is fixed) | literal |
| End | | `eval("script", "System.exit(0);")`, `run("Quit")` | literal |

The whole text then has every `//` replaced by `/`. The tile size and the tile order are
not in the macro: BigStitcher reads them from the files, and "Down & Right" places tile
`k` at column `k // tiles_y`, row `k % tiles_y`, which is the order of `grid.csv`.

Recorded behavior of this macro (the input inventory's `fiji-baseline.md`): Fiji ImageJ 2.14.0 with
BigStitcher 1.2.14 reproduces the 2024 placement of LN sample1 (145 of 145 links
identical, tiles within 1.3e-06 px); the current `subsampling_factors` store no
full-resolution level, while the 2024 run used BigStitcher's automatic mipmap setup
(the placement is unaffected, because shifts are computed at 4× downsampling); and
the z translations in `dataset.xml` are in units of the xy pixel size.

% W-323 `fiji-baseline.md`

The unused `create_BigStitcher_macro.py` differs in four places: the `path` is the folder
without `*.tif`, it re-saves as N5 with Gzip, its `subsampling_factors` are
`[{1,1,1}, {4,4,4}]` with `n5_block_sizes`, and it sets `number_of_threads=8`.

## The Fiji rule

`run_BigStitcher_macro` runs
`{fiji_path} -Djava.io.tmpdir={OUTPUT}/images/fused/{sample}/DAPI --ij2 --headless --run <macro>`
with 4 threads and a Snakemake benchmark file. `config['fiji_path']` is read inside an
f-string when the rule file is parsed, so every workflow needs the key even when the
rule is disabled ({doc}`workflow-configuration`). Two facts from the input inventory decide how the
rule may be trusted: Fiji exits with status 0 when the macro fails, so only a written
`dataset.xml` shows success; and BigStitcher cannot write its HDF5 on the CIFS artifact
mount from GP099, so the save folder must be on local disk.

% Two facts from W-323

## Parsing `dataset.xml`: `create_tile_config.py` step by step

The steps are named T1 to T7 as in the parity runs' prototype.

% the W-324 prototype

| Step | Lines | Behavior at `8e7d256` |
| --- | --- | --- |
| T1 Transforms | 19-30 | `pd.read_xml(…, xpath=".//ViewRegistration")` gives the largest setup id; `xpath=".//ViewTransform"` gives every transform in file order, and the setup of row `r` is set to `r // 3`. Each view is assumed to hold exactly three transforms, in setup order; the pivot keeps two by name, `Stitching Transform` and `Translation to Regular Grid`. A view with a fourth transform (for example a second stitching round) shifts every later assignment; a missing name raises `KeyError`. |
| T2 Composition | 31-41 | From each of the two affines, entries 3, 7 and 11 (the translations in x, y and z) are summed. The linear parts are ignored, which is exact only when both are pure translations; the `calibration` transform (z scaled by `voxel_size_z / voxel_size_xy`, 1.789 for 0.34 / 0.19) is not applied, so the sum is the view's origin in BigStitcher's world units: x and y in pixels, **z in units of the xy pixel size**. |
| T3 Cast | 43-45 | `astype(int)`: truncation towards zero (−0.4 becomes 0, while floor would give −1). |
| T4 Shift | 47-49 | Each axis gets `abs(min)` added. A minimum ≤ 0 moves to 0; a positive minimum is moved further from 0 (the parity case `xml_3x2_positive`: 10 becomes 20). |
| T5 Join | 50-58 | `fov_index` = setup id; joined with `grid.csv` by row position, so setup `i` is grid row `i` and `tile_{i}.tif`. The FOV identity is the grid's `id`; a blank cell keeps its offsets in `x, y, z`. |
| T6 Rectangles | 60-120 | For each acquired tile, the non-overlap box: towards a non-blank left or upper neighbour the edge is `int((neighbour + size − self)/2 + 0.5) + self`, the middle of the overlap; towards a right or lower neighbour `int((self + size − neighbour)/2 + 0.5 + neighbour)`; without a neighbour (or next to a blank) the tile's own edge. `size` is `img_col` for x and for the start in y uses `img_row`, but **the end in y and the lower edge use `img_col`** (lines 106 and 108). Blank tiles get the box 0. Only the four row and column neighbours are considered; z has no box. |
| T7 Output | 122-140 | `start/end_x/y_norm` = box − tile offset, 0 for blanks; the CSV holds `x, y, z, fov_index, col, row, id, grid`, the four box columns and the four normalized ones. `plotly.express` writes a 3D scatter as HTML. plotly is not in the lock, so the script cannot be imported in the locked environment. |

% T4: W-324 `xml_3x2_positive`

On LN sample1 the unchanged script (with a stand-in plotly module, in the parity runs) writes the 2024
`tile_config_sample1.csv` byte for byte. The minima before T4 are −5460, −6135 and −69
(the parity runs' `parity_report.json`, `tile_config_ln`).

% a stand-in plotly module, W-324; W-324 `parity_report.json`

### Which parts belong to assembly

| Part | Lines | Belongs to |
| --- | --- | --- |
| T1, T2, T5: which view is which FOV, and its fractional origin | 19-41, 50-58 | Placement: the import of `dataset.xml` ({doc}`stitching-contract`, "BigStitcher") |
| T3 integer cast | 43-45 | Neither: dropped; kept only in a legacy view that reproduces `tile_config` |
| T4 shift to a zero minimum | 47-49 | The sample frame: its origin is a placement choice ({doc}`stitching-contract`, "Placement record"); integer sample coordinates of molecules and cells are assembly's ({doc}`assembly-baseline`) |
| T6, T7 non-overlap rectangles | 60-130 | Assembly ({doc}`assembly-baseline`): overlap ownership, computed in assemble from the placement |
| HTML plot | 134-140 | Output diagnostics |

% T4: integer sample coordinates of molecules and cells are W-326's. T6, T7: Assembly (W-326).

## Workflow keys that reach stitching

| Key | Where it reaches stitching | Effect |
| --- | --- | --- |
| `rotate_angle` | `rotate_nuclei` (`registration-py.smk:107-109`) rotates the DAPI image by `scipy.ndimage.rotate(…, angle, axes=(1, 2))` (Y–X plane, `reshape=True`, cubic) before it becomes `images/DAPI/{fovID}.tif`; the Python sequencing pipeline applies the same angle (`rotation_degrees`) | The stage-to-image axis mapping is implicit: G3 maps stage X to columns and stage Y to rows with no sign change, so the angle must be chosen so that this holds for the rotated images. For LN, stage X maps to image +x and stage Y to image +y (the input inventory's `positions.md`). An angle that is not a multiple of 90° changes the tile shape. |
| `voxel_size_xy`, `voxel_size_z` | the macro's voxel size | Calibration of BigStitcher's world units; for LN 0.19 and 0.34 µm, about 2 % smaller than the raw acquisition's 0.194398 and 0.346255 µm (the input inventory's `inputs.md`). The tile files carry no calibration. |
| `img_col` | `blank.tif` second axis; T6 x extents and **the y end** | width X in pixels |
| `img_row` | `blank.tif` third axis; T6 y start only | height Y in pixels |
| `fiji_path` | `run_BigStitcher_macro` shell command | needed at parse time |
| `root_output_path`, `dataset_id`, `output_id` | every path, also inside the macro | |
| `maximum_projection` | `rotate_nuclei` projects DAPI to YX before stitching | BigStitcher then reads 2D tiles; `tiles_z=1` and the z overlap are unchanged |
| `documents/sample-annotation.csv` | sample → FOV range (`common.smk:204-220`) | which FOVs belong to a sample; `fov_id_pattern` names them for the rule inputs but not inside `stitching_preparation.py` |

% `rotate_angle`: W-323 `positions.md`. `voxel_size_xy`, `voxel_size_z`: W-323 `inputs.md`.

## The legacy Grid/Collection path

The two sample datasets (tissue-2D, cell-culture-3D) were stitched with Fiji's
Grid/Collection plugin, run by hand; its settings are not recorded (the input inventory's `inputs.md`).
Its result, `TileConfiguration.registered.txt`, has fractional offsets and is read by:

* `src/matlab/ParseFijiTileConfiguration.m`: lines from 5, the FOV name before the first
  `.`, `abs(min)` added per axis, then `int32` (MATLAB rounds to nearest);
* `example/sequential_workflow/reads_stitching.m`: adds the offsets to the integer spot
  coordinates of each FOV, clips them to the fused image, and keeps a spot only if no
  earlier FOV's spot bounding box covers it (first come, first owned), writing
  `signal/fused_goodSpots.csv`;
* `example/sequential_workflow/stitch_clustermap.py`: parses both tile configurations with
  a regular expression, rounds with `int(v + 0.5)`, and joins them with
  `documents/grid.csv`.

% its settings are not recorded (W-323 `inputs.md`)

None of these is called by a rule. They are recorded because the parity runs' references of
the two sample datasets come from this path.

% the W-324 references

## Discrepancies with the agreed scope

% Discrepancies with the agreed §2.10 scope

Each line names the current behavior and the proposed resolution in
{doc}`stitching-contract` or {doc}`stitching-algorithms`.

| # | Current behavior | Proposed resolution |
| --- | --- | --- |
| 1 | **Translations cast to integers** (T3) before anything uses them; the fractional result of both backends is lost. | The placement record keeps float64 origins in micrometres ("Placement record"); the import has no cast. A legacy view reproduces the cast only where `tile_config_{sample}.csv` is still written. |
| 2 | **Stage positions reduced to a row and column grid with a fixed 10 % overlap** (G2-G5 and the macro); the stage positions are discarded. | One tile position table with the nominal position of each FOV and its source, filled by three converters (Leica stage file, regular grid, supplied table); the regular-grid overlap is a parameter. Neighbours come from the overlap of nominal positions. |
| 3 | **Blank tiles added to complete a rectangle** (G4, blank copies) and given offsets in `tile_config`. | Placeholders stay inside the BigStitcher path (a settled decision): the macro function pads, the import drops padding views. The placement record has one row per acquired FOV. |
| 4 | **The axis mapping is implicit in `rotate_angle`** (G3 fixes stage X → column, stage Y → row). | An explicit stage-axis mapping with no default; `rotate_angle` stays an image rotation, and the workflow adapter translates the legacy configuration into the mapping G3 hard-codes, recorded as such ("Stage-axis mapping"). |
| 5 | **Macro parameters fixed in the script.** | A package function renders the macro from a `BigStitcherConfig` whose defaults are today's values, recorded in the placement record. |
| 6 | **The transform names of `dataset.xml` hard-coded**, three transforms per view assigned by position (T1). | The import maps views to FOVs by setup id and the tile attribute, composes every transform of the view except `calibration`, refuses a non-translation, and raises naming the view when a transform is missing ("BigStitcher"). |
| 7 | **Overlap rectangles computed together with placement** (T6-T7). | Placement stops at the per-FOV origins; ownership is computed in assemble ({doc}`assembly-contract`) from the placement. |
| 8 | **`img_col` used for the Y extent** (lines 106, 108). | Assemble takes each FOV's extent from its `ReferenceGrid` shape; the legacy view keeps the bug only to reproduce existing files. |
| 9 | **`fiji_path` needed at parse time.** | The BigStitcher engine task reads `fiji_path` only when it runs; the key becomes optional for workflows that do not stitch with BigStitcher ("Workflow configuration"). |
| 10 | **No package API.** | A `starfinder.stitching` module with the stitch entry, the tile position table, `STITCHING_METHODS`, the macro function and the import ("Names"). |
| 11 | z translations of `dataset.xml` are in xy-pixel units and are cast and used as slice offsets (T2, T3; the input inventory's finding 3). | The import divides z by the calibration's z scale before converting to micrometres. |
| 12 | Fiji exits with status 0 when the macro fails. | The engine task checks that `dataset.xml` exists and holds a stitching transform for every view, else fails. |
| 13 | `stitching_preparation.py` fails with pandas 3.0.0 (`pd.value_counts`), the version the lock has for Python 3.11 to 3.13. | The package converter does not use `pd.value_counts`; its step rule and tie rule are explicit ({doc}`stitching-algorithms`, "Leica stage file"). The legacy script is not repaired by this issue. |
| 14 | `create_tile_config.py` imports plotly, which is not in the lock. | The plot leaves the placement path; a diagnostic plot uses matplotlib. |
| 15 | The step comes from Y alone, as the most frequent exact float difference, with a tie broken by order; two positions in one cell are summed into one id. | The stage converter keeps positions, so no step is needed; the optional snapping to a grid ({doc}`stitching-algorithms`) states its step and tie rule and refuses two positions in one cell. |
| 16 | The input folder `images/DAPI` and the name `Position%03d` are fixed; `fov_id_pattern` is ignored; the stage file's `PositionID` is taken as the FOV number. | The input is a specified, saved stain image on each FOV's reference grid; FOV identities come from the tile position table ("Input image"). |
| 17 | T4 moves a positive minimum away from 0. | The sample frame's origin is defined by a stated gauge ("Placement record"). |
| 18 | The tile copies and `dataset.h5` are undeclared outputs; BigStitcher's save folder must be on local disk. | The engine task declares its outputs and takes a local save folder as a parameter. |
| 19 | The calibration is taken from `voxel_size_xy` and `voxel_size_z`, rounded for LN. | The import converts through each FOV's `ReferenceGrid` spacing ({doc}`stitching-algorithms`, "Import arithmetic"). |
| 20 | `create_tile_config.py` converts one-element arrays to scalars with `int()` (lines 88, 94, 100, 106), deprecated since NumPy 1.25. | Not repaired; recorded. |
| 21 | The current macro stores no full-resolution pyramid level (the input inventory's finding 2). | The macro function records its pyramid setting; a fusion from BigStitcher's HDF5 is not part of the placement path. |

% 3: inside the BigStitcher path (D4). 7: computed in assemble (W-326). 11: W-323 finding 3. 21: W-323 finding 2.

## Golden test

`src/python/test/test_workflow_stitching_golden.py` (markers `workflow`, `golden`, default
tier) pins the current behavior on hand-built fixtures generated in the test: a LAS X
stage file of 8 positions on a 3 × 3 grid (column-major scan, one cell not acquired,
step 2⁻¹² m so equal steps give equal float differences, sub-micrometre X jitter) and the
transforms of a BigStitcher `dataset.xml` for that grid (tiles of 100 × 80 px, 10 %
regular grid, stitching translations of both signs). No image is read, Fiji never runs,
and nothing outside the test is read.

* **Grid derivation** (`stitching_preparation.py`, unchanged, through a stub
  `snakemake`): `grid.csv` text exactly; the blank cell at `j = 4`; the blank of shape
  `(3, img_col, img_row)`; which file each `tile_{j}.tif` copies. pandas 3.0.0 has no
  `pd.value_counts`, so the test restores it for the run as the pandas 2 function
  (`pd.Series(values).value_counts()`); nothing else is changed.
* **Macro** (`create_BigStitcher_macro_1.py`, unchanged, through a stub): every parameter
  of the table above and the SHA-256 of the text.
* **Translation arithmetic** (`create_tile_config.py`): its import of plotly fails in the
  locked environment, so the test pins the arithmetic through `legacy_tile_config`, which
  follows lines 19-130 with the line numbers cited; a test checks the cited lines against
  the file. The fractional sums before the cast, every `tile_config` column, and the
  truncation of −0.4 and −0.6 to 0 are pinned. The unchanged script, run once with a
  stand-in `plotly.express` that only replaces the plot, writes the same CSV as the
  helper.
* **Sensitivity**: moving one stage position into the empty cell changes the pinned
  grid, and changing one stitching translation changes a pinned `tile_config` value.

The values are identical over three separate single-thread runs in the locked
environment (Python 3.12.12, pandas 3.0.0, NumPy 2.2.6, SciPy 1.17.0, tifffile
2026.1.28). The nine tests take under 1 s in total. The parity runs' outputs cover the
same scripts on the real LN sample1 files.

% The W-324 parity outputs
