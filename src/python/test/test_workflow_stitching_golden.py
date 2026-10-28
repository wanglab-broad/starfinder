"""Golden baseline for the current stitching scripts (W-325, §2.10).

Pins, on small hand-built fixtures generated in the test (a Leica stage file and a
BigStitcher ``dataset.xml``; no image is read and Fiji never runs):

(a) the grid derivation of ``workflow/scripts/stitching_preparation.py``: the step taken
    as the most frequent difference of consecutive stage Y positions, the rounding of each
    position to a column and a row, the blank padding of the empty cell, the column-major
    ``grid.csv``, the blank tile's shape and which file each ``tile_<j>.tif`` copies;
(b) the macro text of ``workflow/scripts/create_BigStitcher_macro_1.py``, with every
    parameter it fixes;
(c) the translation arithmetic of ``workflow/scripts/create_tile_config.py``: the sum of
    the ``Stitching Transform`` and ``Translation to Regular Grid`` translations, the
    integer cast (truncation towards zero), the shift by the absolute minimum, and the
    non-overlap rectangles, including ``img_col`` used for the Y extent.

(a) and (b) run the unchanged scripts through a stub ``snakemake`` object. The locked
environment has their imports, but pandas 3.0.0 removed ``pd.value_counts``, which
``stitching_preparation.py:21`` calls; the test restores it for the run as the pandas 2
function (``pd.Series(values).value_counts()``) and changes nothing else.

(c): ``create_tile_config.py`` imports ``plotly.express`` (line 7), which is not in the
lock. The arithmetic is therefore held by ``legacy_tile_config``, which follows the
script at 8e7d256 line by line, with the cited lines (``SCRIPT_LINES``). The unchanged
script is run once with a stand-in ``plotly.express`` module, which only replaces the
plot written after the CSV, to show that its CSV equals the helper's. The tests that ran
only the helper are gone (W-535), as is the check of the stage step on the fixture alone
(W-576); ``TILE_CONFIG`` and ``SUMS`` are asserted on the package
in test_stitching_bigstitcher.py and test_stitching_workflow.py.

The values were produced in the locked project environment (Python 3.12.12, pandas 3.0.0,
NumPy 2.2.6, SciPy 1.17.0, tifffile 2026.1.28) and are identical over repeated
single-thread runs; see design/baselines/stitching.md. When the §2.10 implementation
replaces these scripts, ``legacy_tile_config`` stays as the frozen legacy reference;
design/specs/stitching.md names which pinned values the package functions must
reproduce.
"""
import hashlib
import importlib.util
import runpy
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import tifffile

pytestmark = [pytest.mark.workflow, pytest.mark.golden]

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "workflow" / "scripts"

# ---------------------------------------------------------------------------------------
# Fixture A: a Leica stage file of 8 positions on a 3 × 3 grid, scanned down each column
# and then to the right; the centre cell (column 1, row 1) was not acquired. Stage units
# are metres, as in LAS X. The step is 2**-12 m (244.140625 µm), exactly representable,
# so equal steps give equal float differences; X carries sub-micrometre jitter.
STEP_M = 2.0 ** -12
X0_M, Y0_M = 0.0625, 0.046875
# PositionID -> (column, row, x jitter in metres)
POSITIONS = {1: (0, 0, 0.0), 2: (0, 1, 3e-7), 3: (0, 2, -2e-7), 4: (1, 0, 4e-7),
             5: (1, 2, -1e-7), 6: (2, 0, 2e-7), 7: (2, 1, -4e-7), 8: (2, 2, 1e-7)}
SECTIONS = 3
IMG_COL, IMG_ROW = 6, 4      # tiny stand-in tiles for the copies: width X 6, height Y 4

# grid.csv as stitching_preparation.py writes it for fixture A: column-major, id 0 = blank
GRID_CSV = (",col,row,id,grid\n"
            "0,0,0,1,tile_0_0\n1,0,1,2,tile_1_0\n2,0,2,3,tile_2_0\n"
            "3,1,0,4,tile_0_1\n4,1,1,0,tile_1_1\n5,1,2,5,tile_2_1\n"
            "6,2,0,6,tile_0_2\n7,2,1,7,tile_1_2\n8,2,2,8,tile_2_2\n")

# ---------------------------------------------------------------------------------------
# Fixture B: the transforms of a BigStitcher dataset.xml for the grid of fixture A, with
# tiles of 100 × 80 px (img_col 100, img_row 80); no image, only the transforms matter.
# Translation to Regular Grid: 10 % overlap (90 px in x, 72 px in y), z centred, in units
# of the xy pixel size (the calibration transform scales z by 0.34 / 0.19).
TILE_COL, TILE_ROW = 100, 80
Z_REGULAR = -2.6842105263157894
Z_CALIBRATION = 1.7894736842105263
# setup (= tile index j = grid row) -> Stitching Transform translation (x, y, z); setup 0 is
# BigStitcher's fixed tile and setup 4 the blank. Values of both signs, so that the integer
# cast truncates towards zero where floor would not (setup 5 x, setup 1 y, setup 2 z).
STITCHING = {0: (0.0, 0.0, 0.0), 1: (1.25, -0.6, -0.6), 2: (-1.6, 3.2, 1.9),
             3: (2.7, -1.6, 0.94), 4: (0.0, 0.0, 0.0), 5: (-0.4, 2.45, -1.3),
             6: (4.6, 0.75, 0.3), 7: (3.1, -2.2, -0.45), 8: (5.9, 1.5, 2.6)}

VOXEL_SIZE_XY, VOXEL_SIZE_Z = 0.19, 0.34


def stage_file(positions=POSITIONS):
    """LAS X stage-point list (the attributes stitching_preparation.py reads), in PositionID order."""
    lines = ['<?xml version="1.0" encoding="utf-8"?>',
             '<XYZStagePointDefinitionList StageOrderNumber="0">']
    for pid, (col, row, jitter) in positions.items():
        x, y = X0_M + col * STEP_M + jitter, Y0_M + row * STEP_M
        lines.append(f' <XYZStagePointDefinition PositionID="{pid}" PositionIdentifier="Position{pid}" '
                     f'Sections="{SECTIONS}" StageXPos="{x!r}" StageYPos="{y!r}" />')
    lines.append('</XYZStagePointDefinitionList>')
    return "\n".join(lines) + "\n"


def _affine(x, y, z, z_scale=1.0):
    return f"1.0 0.0 0.0 {x!r} 0.0 1.0 0.0 {y!r} 0.0 0.0 {z_scale!r} {z!r}"


def dataset_xml(stitching=STITCHING):
    """ViewRegistrations of a BigStitcher dataset.xml: three transforms per setup, newest first."""
    parts = ['<?xml version="1.0" encoding="UTF-8"?>', '<SpimData version="0.2">', '  <ViewRegistrations>']
    for setup, (sx, sy, sz) in stitching.items():
        col, row = divmod(setup, 3)
        regular = (90.0 * col - 90.0, 72.0 * row - 72.0, Z_REGULAR)
        parts.append(f'    <ViewRegistration timepoint="0" setup="{setup}">')
        for name, affine in (("Stitching Transform", _affine(sx, sy, sz)),
                             ("Translation to Regular Grid", _affine(*regular)),
                             ("calibration", _affine(0.0, 0.0, 0.0, Z_CALIBRATION))):
            parts += ['      <ViewTransform type="affine">', f'        <Name>{name}</Name>',
                      f'        <affine>{affine}</affine>', '      </ViewTransform>']
        parts.append('    </ViewRegistration>')
    parts += ['  </ViewRegistrations>', '</SpimData>']
    return "\n".join(parts) + "\n"


class _Wildcards(dict):
    """Snakemake wildcards: the scripts read them both as attributes and by key."""
    __getattr__ = dict.__getitem__


def _config(root):
    return {"root_output_path": str(root), "dataset_id": "dataset", "output_id": "output",
            "img_col": IMG_COL, "img_row": IMG_ROW,
            "voxel_size_xy": VOXEL_SIZE_XY, "voxel_size_z": VOXEL_SIZE_Z}


def run_stitching_preparation(tmp_path, monkeypatch, maf_text):
    """stitching_preparation.py through a stub snakemake object; returns the output folder."""
    base = tmp_path / "dataset" / "output"
    (base / "images" / "fused").mkdir(parents=True)
    dapi = base / "images" / "DAPI"
    dapi.mkdir()
    for pid in range(1, 9):  # one tiny, distinct ZYX image per acquired position
        tifffile.imwrite(dapi / f"Position{pid:03}.tif", np.full((SECTIONS, IMG_ROW, IMG_COL), pid, np.uint8),
                         photometric="minisblack")
    annotation = tmp_path / "sample-annotation.csv"
    annotation.write_text("sample_id,fov_start,fov_end\nsample1,1,8\n")
    maf = tmp_path / "sample1.maf"
    maf.write_text(maf_text)
    # pandas 3 removed pd.value_counts (stitching_preparation.py:21); restore the pandas 2 function.
    monkeypatch.setattr(pd, "value_counts", lambda values: pd.Series(values).value_counts(), raising=False)
    # The script imports tqdm (stitching_preparation.py:9) and never calls it; where tqdm is not installed,
    # the import gets a stand-in.
    if importlib.util.find_spec("tqdm") is None:
        stand_in = types.ModuleType("tqdm")
        stand_in.tqdm = lambda iterable=None, *args, **kwargs: iterable
        monkeypatch.setitem(sys.modules, "tqdm", stand_in)
    snakemake = SimpleNamespace(input=[str(annotation), str(maf)], config=_config(tmp_path),
                                wildcards=_Wildcards(sample="sample1"))
    try:
        runpy.run_path(str(SCRIPTS / "stitching_preparation.py"), init_globals={"snakemake": snakemake})
    finally:
        plt.close("all")
    return base / "images" / "fused" / "sample1"


def render_macro(tmp_path, grid_text=GRID_CSV):
    """create_BigStitcher_macro_1.py through a stub snakemake object; returns the macro text."""
    grid, macro = tmp_path / "grid.csv", tmp_path / "BigStitcher_macro.ijm"
    grid.write_text(grid_text)
    config = _config("/data/root")  # a fixed path, so the text does not depend on tmp_path
    snakemake = SimpleNamespace(input=[str(grid)], output=[str(macro)], config=config,
                                wildcards=_Wildcards(sample="sample1"))
    runpy.run_path(str(SCRIPTS / "create_BigStitcher_macro_1.py"), init_globals={"snakemake": snakemake})
    return macro.read_text()


def legacy_tile_config(dataset_xml_path, grid_csv_path, img_col, img_row):
    """workflow/scripts/create_tile_config.py at 8e7d256, lines 19-130, line by line, without the plot.

    Returns the frame the script writes to output/tile_config_{sample}.csv (line 132) and the
    per-setup fractional sums before the integer cast (lines 39-41).
    """
    setup_df = pd.read_xml(dataset_xml_path, xpath=".//ViewRegistration")                    # :19
    setup_max = setup_df['setup'].max()                                                      # :20
    setup_list = np.array([[i] * 3 for i in range(setup_max + 1)]).flatten()                 # :22-25
    transform_df = pd.read_xml(dataset_xml_path, xpath=".//ViewTransform")                   # :27
    transform_df['setup'] = setup_list                                                       # :28
    transform_df = transform_df.pivot(index='setup', columns='Name', values='affine')        # :29
    transform_df = transform_df.loc[:, ["Stitching Transform", "Translation to Regular Grid"]]  # :30
    for axis, position in (('x', 3), ('y', 7), ('z', 11)):                                   # :31-37
        for suffix, name in (('st', "Stitching Transform"), ('trg', "Translation to Regular Grid")):
            transform_df[f'{axis}_{suffix}'] = (transform_df[name].str.split(' ')
                                                .apply(lambda x: x[position]).astype(float))
    for axis in 'xyz':
        transform_df[axis] = transform_df[f'{axis}_st'] + transform_df[f'{axis}_trg']        # :39-41
    sums = transform_df.loc[:, ['x', 'y', 'z']].copy()
    for axis in 'xyz':
        transform_df[axis] = transform_df[axis].astype(int)                                  # :43-45
    for axis in 'xyz':
        transform_df[axis] = transform_df[axis] + np.abs(transform_df[axis].min())           # :47-49
    transform_df['fov_index'] = transform_df.index                                           # :50
    transform_df = transform_df.loc[:, ['x', 'y', 'z', 'fov_index']]                         # :52
    grid_df = pd.read_csv(grid_csv_path, index_col=0)                                        # :55
    tile_config_df = pd.concat([transform_df, grid_df], axis=1)                              # :58

    def neighbor(name, column):
        """The neighbour's column value if the neighbour exists and is not blank (:86, :92, :98, :104)."""
        rows = tile_config_df.grid == name
        if name in tile_config_df.grid.values and tile_config_df.loc[rows, 'id'].values != 0:
            return tile_config_df.loc[rows, column].values[0]
        return None

    boxes = []                                       # start_x, start_y, end_x, end_y per tile
    for i in range(tile_config_df.shape[0]):                                       # :65
        record = tile_config_df.iloc[i]
        if record['id'] == 0:                                                      # :69-73
            boxes.append((0, 0, 0, 0))
            continue
        row, col, x, y = record['row'], record['col'], record['x'], record['y']    # :76-79
        left = neighbor(f"tile_{row}_{col - 1}", 'x')                              # :81, :86-87
        up = neighbor(f"tile_{row - 1}_{col}", 'y')                                # :83, :92-93
        right = neighbor(f"tile_{row}_{col + 1}", 'x')                             # :82, :98-99
        down = neighbor(f"tile_{row + 1}_{col}", 'y')                              # :84, :104-105
        start_x = x if left is None else int((left + img_col - x) / 2 + 0.5) + x   # :88, :90
        start_y = y if up is None else int((up + img_row - y) / 2 + 0.5) + y       # :94, :96
        end_x = x + img_col if right is None else int((x + img_col - right) / 2 + 0.5 + right)  # :100, :102
        # img_col, not img_row, for the Y extent, as the script has it
        end_y = y + img_col if down is None else int((y + img_col - down) / 2 + 0.5 + down)  # :106, :108
        boxes.append((start_x, start_y, end_x, end_y))

    for k, edge in enumerate(('start_x', 'start_y', 'end_x', 'end_y')):           # :117-120
        tile_config_df[edge] = [box[k] for box in boxes]
    for edge, axis in (('start_x', 'x'), ('start_y', 'y'), ('end_x', 'x'), ('end_y', 'y')):  # :122-125
        tile_config_df[f'{edge}_norm'] = tile_config_df[edge] - tile_config_df[axis]
    for edge in ('start_x', 'start_y', 'end_x', 'end_y'):                                    # :127-130
        tile_config_df.loc[tile_config_df.id == 0, f'{edge}_norm'] = 0
    return tile_config_df, sums


# The lines of create_tile_config.py at 8e7d256 that legacy_tile_config follows (blob 55ee5c1).
SCRIPT_LINES = {
    19: "setup_df = pd.read_xml(current_dataset_xml, xpath=\".//ViewRegistration\")",
    27: "transform_df = pd.read_xml(current_dataset_xml, xpath=\".//ViewTransform\")",
    29: "transform_df = transform_df.pivot(index='setup', columns='Name', values='affine')",
    31: ("transform_df['x_st'] = transform_df[\"Stitching Transform\"].str.split(' ')"
         ".apply(lambda x: x[3]).astype(float)"),
    37: ("transform_df['z_trg'] = transform_df[\"Translation to Regular Grid\"].str.split(' ')"
         ".apply(lambda x: x[11]).astype(float)"),
    39: "transform_df['x'] = transform_df['x_st'] + transform_df['x_trg']",
    43: "transform_df['x'] = transform_df['x'].astype(int)",
    45: "transform_df['z'] = transform_df['z'].astype(int)",
    47: "transform_df['x'] = transform_df['x'] + np.abs(transform_df['x'].min())",
    58: "tile_config_df = pd.concat([transform_df, grid_df], axis=1)",
    88: "me_start_x = int((left_x + snakemake.config['img_col'] - current_x)/2 + 0.5) + current_x",
    94: "me_start_y = int((left_y + snakemake.config['img_row'] - current_y)/2 + 0.5) + current_y",
    100: "me_end_x = int((current_x + snakemake.config['img_col'] - right_x)/2 + 0.5 + right_x)",
    102: "me_end_x =  current_x + snakemake.config['img_col']",
    106: "me_end_y = int((current_y + snakemake.config['img_col'] - right_y)/2 + 0.5 + right_y)",
    108: "me_end_y = current_y + snakemake.config['img_col']",
    127: "tile_config_df.loc[tile_config_df.id == 0, 'start_x_norm'] = 0",
}
# The lines of stitching_preparation.py at 8e7d256 (blob 0c4338d) that the docs cite.
PREPARATION_LINES = {
    21: "step = pd.value_counts(temp).index.values[0]",
    27: "coo_shape = (coo_shape / step + 0.5 + 1).astype(int)",
    28: "df['relative_x'] = np.array((df['StageXPos'] - point_relative_min[0])/step + 0.5, dtype=int)",
    45: "ordering_list = ordering_df.unstack(1)",
    92: "blank_3d = np.zeros((z, x, y), dtype=np.uint8)",
}

# ---------------------------------------------------------------------------------------
# Pinned values.
# Step (m) and the stage-file grid: (col, row, id) per grid row j, and tile_<j>.tif's source
GRID_A = [(0, 0, 1), (0, 1, 2), (0, 2, 3), (1, 0, 4), (1, 1, 0), (1, 2, 5), (2, 0, 6), (2, 1, 7), (2, 2, 8)]
BLANK_SHAPE = (SECTIONS, IMG_COL, IMG_ROW)   # (z, img_col, img_row): X before Y
MACRO_SHA256 = "0b5fe52481747be9d6d8c06ae39c31f51aa5ab3a6e6c20c296b55f26c9f9d08c"
# Per setup: fractional sums (x, y, z) before the cast, in pixels (z in units of the xy pixel)
SUMS = {0: (-90.0, -72.0, -2.6842105263157894), 1: (-88.75, -0.6, -3.2842105263157895),
        2: (-91.6, 75.2, -0.7842105263157895), 3: (2.7, -73.6, -1.7442105263157894),
        4: (0.0, 0.0, -2.6842105263157894), 5: (-0.4, 74.45, -3.984210526315789),
        6: (94.6, -71.25, -2.3842105263157896), 7: (93.1, -2.2, -3.1342105263157896),
        8: (95.9, 73.5, -0.08421052631578929)}
# tile_config columns per grid row j:
# x, y, z, start_x, start_y, end_x, end_y, start_x_norm, start_y_norm, end_x_norm, end_y_norm
TILE_CONFIG = [
    (1, 1, 1, 1, 1, 97, 87, 0, 0, 96, 86),
    (3, 73, 0, 3, 77, 103, 161, 0, 4, 100, 88),
    (0, 148, 3, 0, 151, 96, 248, 0, 3, 96, 100),
    (93, 0, 2, 97, 0, 189, 100, 4, 0, 96, 100),
    (91, 73, 1, 0, 0, 0, 0, 0, 0, 0, 0),          # the blank: offsets kept, rectangle zero
    (91, 147, 0, 96, 147, 189, 247, 5, 0, 98, 100),
    (185, 2, 1, 189, 2, 285, 87, 4, 0, 100, 85),
    (184, 71, 0, 184, 77, 284, 159, 0, 6, 100, 88),
    (186, 146, 3, 189, 149, 286, 246, 3, 3, 100, 100),
]


def test_stage_file_grid(tmp_path, monkeypatch):
    out = run_stitching_preparation(tmp_path, monkeypatch, stage_file())
    assert (out / "grid.csv").read_text() == GRID_CSV
    grid = pd.read_csv(out / "grid.csv", index_col=0)
    assert list(grid[["col", "row", "id"]].itertuples(index=False, name=None)) == GRID_A
    assert (out / "grid.png").is_file()
    blank = tifffile.imread(out / "blank.tif")
    assert blank.shape == BLANK_SHAPE and blank.dtype == np.uint8 and not blank.any()
    for j, (_, _, pid) in enumerate(GRID_A):
        tile = tifffile.imread(out / "DAPI" / f"tile_{j}.tif")
        if pid == 0:
            assert tile.shape == BLANK_SHAPE and not tile.any()
        else:
            assert tile.shape == (SECTIONS, IMG_ROW, IMG_COL) and (tile == pid).all()


def test_bigstitcher_macro(tmp_path):
    text = render_macro(tmp_path)
    for parameter in (
            "path=/data/root/dataset/output/images/fused/sample1/DAPI/*.tif exclude=10 pattern_0=Tiles",
            "voxel_size_x=0.19 voxel_size_y=0.19 voxel_size_z=0.34 voxel_size_unit=µm",
            "grid_type=[Down & Right             ] tiles_x=3 tiles_y=3 tiles_z=1",
            "overlap_x_(%)=10 overlap_y_(%)=10 overlap_z_(%)=10",
            "how_to_load_images=[Re-save as multiresolution HDF5]",
            "subsampling_factors=[{2,2,2}, {4,4,4}] hdf5_chunk_sizes=[{16,16,16}, {16,16,16}]",
            "method=[Phase Correlation] downsample_in_x=4 downsample_in_y=4 downsample_in_z=4",
            "filter_by_link_quality min_r=0.7 max_r=1 max_shift_in_x=0",
            "relative=2.5 absolute=3.5",
            "global_optimization_strategy=[Two-Round using Metadata to align unconnected Tiles "
            "and iterative dropping of bad links]",
            "fix_group_0-0"):
        assert parameter in text
    assert hashlib.sha256(text.encode()).hexdigest() == MACRO_SHA256


def test_script_writes_the_helper_csv(tmp_path, monkeypatch):
    """The unchanged script, with a stand-in plotly.express for its plot, writes the helper's CSV."""
    xml, grid = tmp_path / "dataset.xml", tmp_path / "grid.csv"
    xml.write_text(dataset_xml())
    grid.write_text(GRID_CSV)
    csv, html = tmp_path / "tile_config_sample1.csv", tmp_path / "tile_config_sample1.html"

    class Figure:
        def update_traces(self, **kwargs): pass
        def update_scenes(self, **kwargs): pass
        def write_html(self, path): Path(path).write_text("")

    express = types.ModuleType("plotly.express")
    express.scatter_3d = lambda *args, **kwargs: Figure()
    plotly = types.ModuleType("plotly")
    plotly.express = express
    monkeypatch.setitem(sys.modules, "plotly", plotly)
    monkeypatch.setitem(sys.modules, "plotly.express", express)
    config = _config(tmp_path) | {"img_col": TILE_COL, "img_row": TILE_ROW}
    snakemake = SimpleNamespace(input=[str(xml), str(grid)], output=[str(csv), str(html)], config=config,
                                wildcards=_Wildcards(sample="sample1"))
    runpy.run_path(str(SCRIPTS / "create_tile_config.py"), init_globals={"snakemake": snakemake})
    frame, _ = legacy_tile_config(xml, grid, TILE_COL, TILE_ROW)
    assert csv.read_text() == frame.to_csv()


def test_changing_one_stage_position_changes_the_grid(tmp_path, monkeypatch):
    """Position 5 moved one step up, into the empty centre cell."""
    moved = dict(POSITIONS)
    moved[5] = (1, 1, -1e-7)
    out = run_stitching_preparation(tmp_path, monkeypatch, stage_file(moved))
    grid = pd.read_csv(out / "grid.csv", index_col=0)
    ids = list(grid["id"])
    assert ids != [pid for _, _, pid in GRID_A]
    assert (ids[4], ids[5]) == (5, 0)
