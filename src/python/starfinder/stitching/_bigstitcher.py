"""The two package ends of the BigStitcher path: the macro and the import of ``dataset.xml``.

design/specs/stitching.md ("BigStitcher", "Where the record differs from ``tile_config`` on
purpose") and design/specs/stitching-algorithms.md ("Import arithmetic"). Fiji itself is an engine
task: nothing here starts it. The legacy view applies T3 to T7 of design/baselines/stitching.md
to the import's raw translations and the view map, so that ``tile_config_{sample}.csv`` can
still be written from the record.
"""
from __future__ import annotations

import hashlib
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Integral, Real
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pandas as pd

from starfinder._records import file_sha256
from starfinder._registry import provenance
from starfinder.image import IncompatibleGeometryError
from starfinder.segmentation._labels import ReferenceGrid

from ._methods import LINEAR_TOLERANCE, StitchingSpec
from ._neighbors import boxes
from ._placement import (AXES, FORMAT_VERSION, FOV_COLUMNS, LINK_COLUMNS, STAGE, PlacementResult, _key_sets,
                         check_grids, empty_links, frame_record, input_record)
from ._positions import TilePositionTable, _names
from ._review import ReviewPolicy, review_placement
from ._stitch import FALLBACKS, StitchingInput, _check_inputs, _components

VIEW_MAP_COLUMNS = ("view", "grid_column", "grid_row", "fov_id", "kind")
VIEW_MAP_DTYPES = {"view": "int64", "grid_column": "int64", "grid_row": "int64", "fov_id": "str", "kind": "str"}
VIEW_KINDS = ("fov", "padding", "placeholder")
CALIBRATION = "calibration"
STITCHING_TRANSFORM = "Stitching Transform"
TWO_ROUND = "Two-Round using Metadata to align unconnected Tiles and iterative dropping of bad links"
# BigStitcher's own label of the "Down & Right" grid, padded as its menu writes it.
GRID_TYPE = "Down & Right             "
LEGACY_COLUMNS = ("x", "y", "z", "fov_index", "col", "row", "id", "grid", "start_x", "start_y", "end_x", "end_y",
                  "start_x_norm", "start_y_norm", "end_x_norm", "end_y_norm")
_TILE = re.compile(r"tile_(\d+)\.tif")


def _number(value) -> str:
    """A macro number: an integral value without a decimal point, any other as Python writes the float."""
    value = float(value)
    return str(int(value)) if value.is_integer() else repr(value)


def _check_number(name, value, *, low=None, integer=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral if integer else Real) \
            or not math.isfinite(value) or (low is not None and value < low):
        kind = "an integer" if integer else "a finite number"
        raise ValueError(f"BigStitcherConfig.{name} must be {kind}" + (f" ≥ {low}" if low is not None else "")
                         + f"; got {value!r}")


def _check_triple(name, value, *, low=None, integer=False):
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 3:
        raise ValueError(f"BigStitcherConfig.{name} must be three values (x, y, z); got {value!r}")
    for item in value:
        _check_number(name, item, low=low, integer=integer)
    return tuple(value)


@dataclass(frozen=True)
class BigStitcherConfig:
    """The parameters of the BigStitcher macro; the defaults are today's macro (create_BigStitcher_macro_1.py).

    Parameters
    ----------
    overlap_percent_xyz : tuple
        Nominal tile overlap percentages in XYZ order.
    exclude : int
        Nonnegative Bioformats loader exclusion setting in the dataset-definition macro.
    subsampling_factors : tuple
        Positive integer XYZ subsampling factors for the HDF5 resolution levels.
    hdf5_chunk_sizes : tuple
        Positive integer XYZ chunk dimensions, in voxels, for the HDF5 resolution levels.
    timepoints_per_partition : int
        Nonnegative timepoint count per HDF5 partition in the macro.
    setups_per_partition : int
        Nonnegative setup count per HDF5 partition in the macro.
    deflate_compression : bool
        Enable deflate compression for the HDF5 export.
    downsample_xyz : tuple
        Positive integer XYZ downsampling factors for pairwise phase correlation.
    min_r : float
        Lower link-quality threshold for the BigStitcher pairwise-shift filter.
    max_r : float
        Upper link-quality threshold for the BigStitcher pairwise-shift filter.
    max_shift_xyz : tuple
        Nonnegative XYZ shift-limit settings passed to the BigStitcher filter.
        Zero requests no shift limit on that axis.
    max_displacement : float
        Nonnegative displacement-limit setting passed to the BigStitcher filter.
        Zero requests no displacement limit.
    relative : float
        Nonnegative relative-error setting for BigStitcher global optimization.
    absolute : float
        Nonnegative absolute-error setting for BigStitcher global optimization, in pixels.
    pad_missing : bool
        Represent acquired FOVs with absent images as blank placeholder views.
    global_optimization_strategy : str
        Macro strategy, ``Two-Round using Metadata to align unconnected Tiles and iterative dropping of bad links``.
    method : str
        Method identifier, ``bigstitcher``.

    Notes
    -----
    Dataset definition: the Bioformats loader on ``tile_{j}.tif`` (files below
    ``exclude`` are skipped, as today), ``Move Tile to Grid`` with ``Down &
    Right`` and ``overlap_percent_xyz`` (10 % in x, y and z), re-saved as
    multiresolution HDF5 with ``subsampling_factors`` ``[{2,2,2}, {4,4,4}]``,
    ``hdf5_chunk_sizes`` 16³ and deflate compression. Pairwise shifts: phase
    correlation at ``downsample_xyz`` (4× in x, y and z). Link filter:
    ``min_r`` 0.7, ``max_r`` 1, no shift limits (``max_shift_xyz`` 0) and no
    displacement limit (``max_displacement`` 0). Global optimization:
    ``relative`` 2.5 and ``absolute`` 3.5 px, the two-round strategy with
    metadata for unconnected tiles and iterative dropping of bad links
    (``global_optimization_strategy``, fixed), the first tile fixed.
    ``pad_missing`` (default False) replaces an acquired FOV whose image is
    absent by a blank, a ``placeholder`` view, instead of a padding view; it is
    the caller's opt-in to that FOV's regular-grid position. The voxel size is
    not a field: the macro writes the reference grids' spacing.

    Raises
    ------
    ValueError
        A field outside its rule, naming it.
    """

    overlap_percent_xyz: tuple = (10, 10, 10)
    exclude: int = 10
    subsampling_factors: tuple = ((2, 2, 2), (4, 4, 4))
    hdf5_chunk_sizes: tuple = ((16, 16, 16), (16, 16, 16))
    timepoints_per_partition: int = 1
    setups_per_partition: int = 0
    deflate_compression: bool = True
    downsample_xyz: tuple = (4, 4, 4)
    min_r: float = 0.7
    max_r: float = 1
    max_shift_xyz: tuple = (0, 0, 0)
    max_displacement: float = 0
    relative: float = 2.5
    absolute: float = 3.5
    pad_missing: bool = False
    global_optimization_strategy: str = field(default=TWO_ROUND, init=False)
    method: str = field(default="bigstitcher", init=False)

    def __post_init__(self):
        overlap = _check_triple("overlap_percent_xyz", self.overlap_percent_xyz, low=0)
        if any(v >= 100 for v in overlap):
            raise ValueError(f"BigStitcherConfig.overlap_percent_xyz must be below 100; got {overlap!r}")
        _check_number("exclude", self.exclude, low=0, integer=True)
        for name in ("subsampling_factors", "hdf5_chunk_sizes"):
            levels = getattr(self, name)
            if isinstance(levels, (str, bytes)) or not isinstance(levels, Sequence) or not levels:
                raise ValueError(f"BigStitcherConfig.{name} must be a nonempty sequence of (x, y, z) levels")
            object.__setattr__(self, name, tuple(_check_triple(name, level, low=1, integer=True) for level in levels))
        if len(self.subsampling_factors) != len(self.hdf5_chunk_sizes):
            raise ValueError("BigStitcherConfig.subsampling_factors and hdf5_chunk_sizes need one entry per level")
        _check_number("timepoints_per_partition", self.timepoints_per_partition, low=0, integer=True)
        _check_number("setups_per_partition", self.setups_per_partition, low=0, integer=True)
        for name in ("deflate_compression", "pad_missing"):
            if not isinstance(getattr(self, name), (bool, np.bool_)):
                raise ValueError(f"BigStitcherConfig.{name} must be True or False")
        object.__setattr__(self, "overlap_percent_xyz", overlap)
        object.__setattr__(self, "downsample_xyz", _check_triple("downsample_xyz", self.downsample_xyz, low=1,
                                                                 integer=True))
        object.__setattr__(self, "max_shift_xyz", _check_triple("max_shift_xyz", self.max_shift_xyz, low=0))
        for name in ("min_r", "max_r", "max_displacement", "relative", "absolute"):
            _check_number(name, getattr(self, name), low=0)
        if not self.min_r <= self.max_r <= 1:
            raise ValueError(f"BigStitcherConfig needs 0 ≤ min_r ≤ max_r ≤ 1; got {self.min_r!r}, {self.max_r!r}")


def _macro_path(value, name):
    """A folder as the macro writes it; a path with whitespace is bracketed, as Fiji's option strings need."""
    if not isinstance(value, (str, os.PathLike)):
        raise TypeError(f"{name} must be a path")
    text = os.fspath(value)
    if not text:
        raise ValueError(f"{name} must not be empty")
    text = text.rstrip("/") or "/"
    return text


def _option(path):
    return f"[{path}]" if re.search(r"\s", path) else path


def _macro_text(config, voxel_xyz, tiles_x, tiles_y, tile_dir, save_dir):
    """The macro, line for line the text of create_BigStitcher_macro_1.py with these values."""
    c = config
    xy, z = _number(voxel_xyz[0]), _number(voxel_xyz[2])
    ox, oy, oz = (_number(v) for v in c.overlap_percent_xyz)
    dx, dy, dz = (_number(v) for v in c.downsample_xyz)
    sx, sy, sz = (_number(v) for v in c.max_shift_xyz)
    factors = ", ".join("{" + ",".join(str(v) for v in level) + "}" for level in c.subsampling_factors)
    chunks = ", ".join("{" + ",".join(str(v) for v in level) + "}" for level in c.hdf5_chunk_sizes)
    deflate = " use_deflate_compression" if c.deflate_compression else ""
    select = _option(f"{save_dir}/dataset.xml")
    every = ("process_angle=[All angles] process_channel=[All channels] process_illumination=[All illuminations] "
             "process_tile=[All tiles] process_timepoint=[All Timepoints] ")
    lines = [
        "",
        'run("Define dataset ...", ',
        '"define_dataset=[Automatic Loader (Bioformats based)] project_filename=dataset.xml " +',
        f'"path={_option(tile_dir + "/*.tif")} exclude={c.exclude} pattern_0=Tiles " +',
        f'"modify_voxel_size? voxel_size_x={xy} voxel_size_y={_number(voxel_xyz[1])} voxel_size_z={z} '
        'voxel_size_unit=µm " +',
        f'"move_tiles_to_grid_(per_angle)?=[Move Tile to Grid (Macro-scriptable)] grid_type=[{GRID_TYPE}] '
        f'tiles_x={tiles_x} tiles_y={tiles_y} tiles_z=1 overlap_x_(%)={ox} overlap_y_(%)={oy} overlap_z_(%)={oz} " +',
        '"keep_metadata_rotation how_to_load_images=[Re-save as multiresolution HDF5] load_raw_data_virtually " +',
        f'"dataset_save_path={_option(save_dir)} manual_mipmap_setup " +',
        f'"subsampling_factors=[{factors}] hdf5_chunk_sizes=[{chunks}] '
        f'timepoints_per_partition={c.timepoints_per_partition} setups_per_partition={c.setups_per_partition}'
        f'{deflate}");',
        "",
        'run("Calculate pairwise shifts ...", ',
        f'"select={select} " + ',
        f'"{every}" +',
        f'"method=[Phase Correlation] downsample_in_x={dx} downsample_in_y={dy} downsample_in_z={dz}");',
        "",
        'run("Filter pairwise shifts ...", ',
        f'"select={select} " +',
        f'"filter_by_link_quality min_r={_number(c.min_r)} max_r={_number(c.max_r)} max_shift_in_x={sx} '
        f'max_shift_in_y={sy} max_shift_in_z={sz} max_displacement={_number(c.max_displacement)}");',
        "",
        'run("Optimize globally and apply shifts ...", ',
        f'"select={select} " +',
        f'"{every}" + ',
        f'"relative={_number(c.relative)} absolute={_number(c.absolute)} '
        f'global_optimization_strategy=[{c.global_optimization_strategy}] show_expert_grouping_options " +',
        '"how_to_treat_timepoints=[treat individually] how_to_treat_channels=group how_to_treat_illuminations=group '
        'how_to_treat_angles=[treat individually] how_to_treat_tiles=compare fix_group_0-0");',
        "",
        'eval("script", "System.exit(0);");',
        'run("Quit");',
        "",
    ]
    # The legacy script replaces every "//" of its text by "/" (doubled separators in its paths).
    return "\n".join(lines).replace("//", "/")


def _cells(table):
    frame = table.positions
    missing = [fov for fov, c, r in zip(table.fov_ids, frame["grid_column"], frame["grid_row"])
               if pd.isna(c) or pd.isna(r)]
    if missing:
        raise ValueError("the BigStitcher macro places FOVs on a regular grid and needs the table's grid_column and "
                         f"grid_row (positions_from_grid, or positions_from_leica with snap_to_grid); FOVs "
                         f"{_names(missing)} have no cell")
    cells = [(int(c), int(r)) for c, r in zip(frame["grid_column"], frame["grid_row"])]
    negative = [fov for fov, (c, r) in zip(table.fov_ids, cells) if c < 0 or r < 0]
    if negative:
        raise ValueError(f"grid cells must be ≥ 0; FOVs {_names(negative)} are not")
    return cells


def _prepare_tile_dir(tile_dir, n_views):
    """The tile folder: created if absent; it may hold only the ``tile_{j}.tif`` files this macro writes."""
    folder = Path(tile_dir)
    if folder.exists() and not folder.is_dir():
        raise FileExistsError(f"tile_dir {folder} is a file")
    folder.mkdir(parents=True, exist_ok=True)
    foreign = sorted(p.name for p in folder.iterdir() if p.name.lower().endswith((".tif", ".tiff"))
                     and not (_TILE.fullmatch(p.name) and int(_TILE.fullmatch(p.name).group(1)) < n_views))
    if foreign:
        raise FileExistsError(f"tile_dir {folder} holds TIFF files the macro would also read ({', '.join(foreign)}); "
                              "BigStitcher loads every *.tif of the folder")
    return folder


def export_bigstitcher(table: TilePositionTable, inputs: Sequence[StitchingInput], *, config: BigStitcherConfig,
                       tile_dir, save_dir) -> tuple[str, pd.DataFrame]:
    """Write the tiles BigStitcher reads and return the Fiji macro text and the view map.

    The grid is the table's cells: ``tiles_x`` the largest ``grid_column`` + 1,
    ``tiles_y`` the largest ``grid_row`` + 1, and view ``j`` is the cell at
    column ``j // tiles_y``, row ``j % tiles_y`` (column-major, BigStitcher's
    ``Down & Right``), read from ``tile_dir/tile_{j}.tif``. Each view is one of:

    * ``fov``: an acquired FOV with an image; ``tile_{j}.tif`` is its image;
    * ``placeholder``: with ``config.pad_missing=True``, an acquired FOV whose
      image is absent; ``tile_{j}.tif`` is a blank of the tile shape;
    * ``padding``: an empty cell, or with ``pad_missing=False`` the cell of an
      acquired FOV whose image is absent (that FOV is then ``missing`` in the
      import); a blank, and no ``fov_id``.

    The blanks are zeros of the tiles' shape and dtype. The macro text is that
    of today's ``create_BigStitcher_macro_1.py`` with the config's values, the
    tile folder (``path=<tile_dir>/*.tif``), the save folder
    (``dataset_save_path`` and ``select=<save_dir>/dataset.xml``, a local folder
    for the engine task) and the voxel size from the reference grids' spacing
    (µm, x and y the grids' X and Y spacing, z their Z spacing). Fiji is not run.

    Parameters
    ----------
    table : TilePositionTable
        Every row with ``grid_column`` and ``grid_row``.
    inputs : Sequence[StitchingInput]
        One per row of ``table``, in table order; ``image`` None for an absent
        image. All grids share one shape, spacing, unit µm and identity
        direction.
    config : BigStitcherConfig
    tile_dir : str or Path
        The folder of ``tile_{j}.tif``; created if absent, and it may hold no
        other TIFF file.
    save_dir : str or Path
        BigStitcher's save folder, written into the macro (on local disk).

    Returns
    -------
    tuple[str, pandas.DataFrame]
        The macro text and the view map: ``view`` (``j``), ``grid_column``,
        ``grid_row``, ``fov_id`` (empty for padding) and ``kind`` (``fov``,
        ``padding`` or ``placeholder``), one row per view in view order.

    Raises
    ------
    TypeError
        A table that is not a TilePositionTable, a config that is not a
        BigStitcherConfig or inputs that are not StitchingInput.
    ValueError
        A table without cells, two FOVs in one cell, inputs that do not match
        the table, or images of more than one dtype.
    IncompatibleGeometryError
        Grids outside the sample-frame geometry or of more than one shape.
    FileExistsError
        tile_dir is a file or holds other TIFF files.
    """
    import tifffile
    if not isinstance(table, TilePositionTable):
        raise TypeError("table must be a TilePositionTable")
    if not isinstance(config, BigStitcherConfig):
        raise TypeError("config must be a BigStitcherConfig")
    tile_text, save_text = _macro_path(tile_dir, "tile_dir"), _macro_path(save_dir, "save_dir")
    if not table.fov_ids:
        raise ValueError("the tile position table has no row")
    inputs = _check_inputs(inputs, table)
    grids = {item.fov_id: item.grid for item in inputs}
    check_grids(grids)
    shapes = {}
    for fov, grid in grids.items():
        shapes.setdefault(grid.shape_zyx, []).append(fov)
    if len(shapes) > 1:
        described = "; ".join(f"{_names(fovs)}: {shape}" for shape, fovs in shapes.items())
        raise IncompatibleGeometryError(f"the BigStitcher grid needs one tile shape; {described}")
    shape = next(iter(shapes))
    dtypes = {np.asarray(item.image).dtype for item in inputs if item.image is not None}
    if len(dtypes) > 1:
        raise ValueError(f"the tiles must share one dtype; got {sorted(str(d) for d in dtypes)}")
    dtype = dtypes.pop() if dtypes else np.dtype(np.uint8)
    cells = _cells(table)
    owners = {}
    for item, cell in zip(inputs, cells):
        owners.setdefault(cell, []).append(item.fov_id)
    clashes = {cell: fovs for cell, fovs in owners.items() if len(fovs) > 1}
    if clashes:
        raise ValueError("two FOVs in one grid cell: " + "; ".join(f"{_names(f)} in column {c}, row {r}"
                                                                   for (c, r), f in clashes.items()))
    tiles_x = max(c for c, _ in cells) + 1
    tiles_y = max(r for _, r in cells) + 1
    by_cell = {cell: item for item, cell in zip(inputs, cells)}
    rows = []
    for j in range(tiles_x * tiles_y):
        cell = divmod(j, tiles_y)
        item = by_cell.get(cell)
        if item is not None and item.image is not None:
            kind, fov = "fov", item.fov_id
        elif item is not None and config.pad_missing:
            kind, fov = "placeholder", item.fov_id
        else:
            kind, fov = "padding", ""
        rows.append((j, cell[0], cell[1], fov, kind))
    view_map = pd.DataFrame(rows, columns=list(VIEW_MAP_COLUMNS)).astype(VIEW_MAP_DTYPES)
    spacing = grids[inputs[0].fov_id].metadata.spacing_zyx
    text = _macro_text(config, (spacing[2], spacing[1], spacing[0]), tiles_x, tiles_y, tile_text, save_text)

    folder = _prepare_tile_dir(tile_dir, len(rows))
    blank = None
    for j, cell, kind in zip(view_map["view"], zip(view_map["grid_column"], view_map["grid_row"]), view_map["kind"]):
        if kind == "fov":
            image = np.asarray(by_cell[cell].image)
        else:
            blank = np.zeros(shape, dtype) if blank is None else blank
            image = blank
        tifffile.imwrite(folder / f"tile_{j}.tif", image, photometric="minisblack")
    return text, view_map


def _check_view_map(view_map, table):
    if not isinstance(view_map, pd.DataFrame):
        raise TypeError("view_map must be a pandas DataFrame")
    missing = [c for c in VIEW_MAP_COLUMNS if c not in view_map.columns]
    if missing:
        raise ValueError(f"the view map lacks the columns {missing}")
    frame = view_map[list(VIEW_MAP_COLUMNS)].copy()
    frame["fov_id"] = ["" if pd.isna(v) else v for v in frame["fov_id"]]
    for column in ("view", "grid_column", "grid_row"):
        values = list(frame[column])
        if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) or v < 0 for v in values):
            raise ValueError(f"view map column {column!r} must hold integers ≥ 0")
    frame = frame.astype(VIEW_MAP_DTYPES).sort_values("view", kind="stable").reset_index(drop=True)
    repeated = sorted({int(v) for v in frame["view"][frame["view"].duplicated()]})
    if repeated:
        raise ValueError(f"the view map repeats views {_names(repeated)}")
    unknown = sorted(set(frame["kind"]) - set(VIEW_KINDS))
    if unknown:
        raise ValueError(f"view map kind must be one of {list(VIEW_KINDS)}; got {_names(unknown)}")
    padding = frame["kind"] == "padding"
    has_fov = frame["fov_id"] != ""
    bad = list(frame["view"][padding & has_fov]) + list(frame["view"][~padding & ~has_fov])
    if bad:
        raise ValueError(f"views {_names(sorted(bad))} of the view map: a padding view has no fov_id, and a fov or "
                         "placeholder view names one")
    cells = frame[["grid_column", "grid_row"]]
    shared = sorted({int(v) for v in frame["view"][cells.duplicated(keep=False)]})
    if shared:
        raise ValueError(f"views {_names(shared)} of the view map share a grid cell")
    named = frame[~padding]
    twice = sorted(set(named["fov_id"][named["fov_id"].duplicated()]))
    if twice:
        raise ValueError(f"the view map names FOVs {_names(twice)} in more than one view")
    outside = [fov for fov in named["fov_id"] if fov not in set(table.fov_ids)]
    if outside:
        raise ValueError(f"the view map names FOVs the table lacks: {_names(outside)}")
    return frame


def _float(text, what):
    try:
        return float(text)
    except (TypeError, ValueError):
        raise ValueError(f"{what}: {text!r} is not a number") from None


def _affine(text, what):
    """The 4 × 4 matrix of a BigStitcher affine: twelve numbers, the row-major 3 × 4 matrix in XYZ."""
    values = (text or "").split()
    if len(values) != 12:
        raise ValueError(f"{what} must hold twelve numbers; got {len(values)}")
    numbers = [_float(v, what) for v in values]
    return [numbers[0:4], numbers[4:8], numbers[8:12], [0.0, 0.0, 0.0, 1.0]]


def _product(a, b):
    """a · b of two 4 × 4 matrices, each entry summed in index order."""
    out = []
    for i in range(4):
        row = []
        for j in range(4):
            total = a[i][0] * b[0][j]
            for k in range(1, 4):
                total = total + a[i][k] * b[k][j]
            row.append(total)
        out.append(row)
    return out


def _linear_off(matrix):
    return max(abs(matrix[i][j] - (1.0 if i == j else 0.0)) for i in range(3) for j in range(3))


def _child(element, name):
    for child in element:
        if child.tag.rsplit("}", 1)[-1] == name:
            return child
    return None


def _children(element, name):
    return [child for child in element.iter() if child.tag.rsplit("}", 1)[-1] == name]


def _text(element, name):
    child = _child(element, name)
    return None if child is None else (child.text or "").strip()


def _setup_tiles(root, path):
    """Setup id → view: the ``tile`` attribute of each ``ViewSetup``, read through the tile's name if numeric."""
    names = {}
    for attributes in _children(root, "Attributes"):
        if attributes.get("name") == "tile":
            for tile in attributes:
                tile_id, tile_name = _text(tile, "id"), _text(tile, "name")
                if tile_id is not None and tile_name is not None and tile_name.isdigit():
                    names[tile_id] = int(tile_name)
    tiles = {}
    for setup in _children(root, "ViewSetup"):
        setup_id = _text(setup, "id")
        attributes = _child(setup, "attributes")
        tile = None if attributes is None else _text(attributes, "tile")
        if setup_id is None or tile is None or not setup_id.isdigit() or not tile.isdigit():
            raise ValueError(f"{path}: a ViewSetup lacks its id or its tile attribute")
        tiles[int(setup_id)] = names.get(tile, int(tile))
    return tiles


def _views(root, path, tiles):
    """view → (setup, raw world translation (x, y, z), calibration (c_x, c_y, c_z), transform names)."""
    views = {}
    for registration in _children(root, "ViewRegistration"):
        setup_text, timepoint = registration.get("setup"), registration.get("timepoint", "0")
        if setup_text is None or not setup_text.isdigit():
            raise ValueError(f"{path}: a ViewRegistration lacks its setup")
        setup = int(setup_text)
        if timepoint != "0":
            raise ValueError(f"{path}: setup {setup} is registered at timepoint {timepoint}; the import reads "
                             "timepoint 0 only")
        if setup not in tiles:
            raise ValueError(f"{path}: setup {setup} has no ViewSetup with a tile attribute, which maps it to its view")
        view = tiles[setup]
        if view in views:
            raise ValueError(f"{path}: view {view} (tile attribute) is registered by setups {views[view][0]} and "
                             f"{setup}")
        names, composed, calibration = [], None, None
        for transform in _children(registration, "ViewTransform"):
            name = _text(transform, "Name") or ""
            matrix = _affine(_text(transform, "affine"), f"{path}: view {view}, transform {name!r}")
            if not all(math.isfinite(v) for row in matrix for v in row):
                raise ValueError(f"{path}: setup {setup} (view {view}), transform {name!r}: the affine holds a "
                                 f"non-finite number ({_text(transform, 'affine')!r})")
            if name == CALIBRATION:
                if calibration is not None:
                    raise ValueError(f"{path}: view {view} has two calibration transforms")
                calibration = matrix
                continue
            names.append(name)
            composed = matrix if composed is None else _product(composed, matrix)
        if calibration is None:
            raise ValueError(f"{path}: view {view} has no calibration transform")
        if STITCHING_TRANSFORM not in names:
            raise ValueError(f"{path}: view {view} has no {STITCHING_TRANSFORM!r}; the engine task fails such a run")
        off = _linear_off(composed)
        if not off <= LINEAR_TOLERANCE:
            # §2.10: stitching admits translation only.
            raise ValueError(f"{path}: view {view} composes {names} to a linear part {off:.3g} off the identity; "
                             f"stitching admits translation only (tolerance {LINEAR_TOLERANCE})")
        c_x, c_y, c_z = calibration[0][0], calibration[1][1], calibration[2][2]
        if c_x != c_y:
            raise ValueError(f"{path}: view {view} has a calibration with c_x {c_x!r} ≠ c_y {c_y!r}")
        if not all(math.isfinite(c) and c > 0 for c in (c_x, c_y, c_z)):
            raise ValueError(f"{path}: view {view} has a calibration scale that is not finite and > 0")
        world = (composed[0][3], composed[1][3], composed[2][3])
        views[view] = (setup, world, (c_x, c_y, c_z), names)
    return views


def _pairwise(root, path, setups):
    """Every PairwiseResult: (view a, view b, translation (x, y, z), correlation, hash, bounding box)."""
    results = []
    for element in _children(root, "PairwiseResult"):
        ends = []
        for end in ("view_setup_a", "view_setup_b"):
            value = element.get(end)
            value = _text(element, end) if value is None else value.strip()
            parts = (value or "").split()
            if len(parts) != 1 or not parts[0].isdigit():
                raise ValueError(f"{path}: a PairwiseResult's {end} must be one setup id; got {value!r}")
            setup = int(parts[0])
            if setup not in setups:
                raise ValueError(f"{path}: a PairwiseResult names setup {setup}, which has no ViewRegistration")
            ends.append(setups[setup])
        what = f"{path}: PairwiseResult of views {ends[0]} and {ends[1]}"
        shift = _affine(_text(element, "shift"), f"{what}, shift")
        off = _linear_off(shift)
        if not off <= LINEAR_TOLERANCE:
            # §2.10: stitching admits translation only.
            raise ValueError(f"{what}: the shift's linear part is {off:.3g} off the identity; stitching admits "
                             "translation only")
        correlation = _float(_text(element, "correlation"), f"{what}, correlation")
        results.append((ends[0], ends[1], (shift[0][3], shift[1][3], shift[2][3]), correlation,
                        _text(element, "hash"), _text(element, "overlap_boundingbox")))
    return results


def _to_um(world_xyz, calibration_xyz, spacing_zyx):
    """World units to µm, ZYX: ``u = w ⊘ c`` (z now in slices), then ``u ⊙ s`` with the reference spacing."""
    voxel = [w / c for w, c in zip(world_xyz, calibration_xyz)]
    return np.array([voxel[2] * spacing_zyx[0], voxel[1] * spacing_zyx[1], voxel[0] * spacing_zyx[2]])


def import_bigstitcher(dataset_xml, *, table: TilePositionTable, view_map: pd.DataFrame,
                       grids: Mapping[str, ReferenceGrid], config: BigStitcherConfig,
                       review: ReviewPolicy = ReviewPolicy(), fallback: str | None = None,
                       sample: str = "sample", macro: str | None = None) -> PlacementResult:
    """Read BigStitcher's ``dataset.xml`` into a :class:`PlacementResult`, as ``stitch`` would build it.

    For each ``ViewRegistration`` (timepoint 0) its setup maps to a view through
    the ``ViewSetup`` tile attribute (not through the order of the transforms),
    and the view to a FOV through ``view_map``. Every transform except
    ``calibration`` is composed in file order (newest first) as 4 × 4 matrices;
    a composed linear part off the identity by more than 1e-9 per entry is
    refused. The composed translation is the raw world translation ``w`` (x and
    y in pixels, z in xy-pixel units); ``u = w ⊘ (c_x, c_y, c_z)`` with the
    calibration's diagonal is the offset in the tile's voxels, and
    ``origin_raw = u ⊙ s`` (ZYX) with the FOV's reference spacing ``s``, never
    the macro's voxel size, and no cast. Then:

    * ``method_translation_z/y/x`` is ``w`` (world units, unrounded) for every
      FOV with a view; a padding view has no row and keeps ``w`` and its cell
      in ``record["method"]["views"]``;
    * each ``PairwiseResult`` between two FOV views is a kept link: its shift's
      translation converted as above (``shift_*``, µm, of ``fov_b`` against
      ``fov_a``), its correlation as ``quality``, the nominal overlap box from
      the table (NaN when the nominal boxes do not overlap); hash and bounding
      box are recorded;
    * statuses: ``missing`` for a FOV of the table without a view;
      ``placeholder`` for a placeholder view; ``image_supported`` for a FOV view
      with a kept link into the reference component (the component of kept
      links between FOV views with the most FOVs, a tie going to the first in
      table order); any other FOV view (one BigStitcher's second round placed
      from metadata) is ``failed`` with ``fallback=None`` and ``metadata_only``
      at ``origin = nominal``, flagged ``fallback``, with ``fallback="nominal"``;
    * gauge G1: ``c = origin_raw − nominal``; ``origin = nominal + c − mean(c)``
      with the mean over the ``image_supported`` FOVs, applied to placeholders
      too (no shift when no FOV is ``image_supported``);
    * :func:`review_placement` applies ``review``.

    Parameters
    ----------
    dataset_xml : str or Path
        BigStitcher's ``dataset.xml``.
    table : TilePositionTable
        The nominal positions; its rows are the result's rows.
    view_map : pandas.DataFrame
        ``view``, ``grid_column``, ``grid_row``, ``fov_id`` (empty or missing for
        padding) and ``kind``, as :func:`export_bigstitcher` returns it.
    grids : Mapping[str, ReferenceGrid]
        One reference grid per ``fov_id`` of the table, and no other key.
    config : BigStitcherConfig
        The config that rendered the macro; recorded.
    review : ReviewPolicy
    fallback : str or None
        ``None`` (default) or ``"nominal"`` (opt-in, flagged).
    sample : str
        The sample; the frame is ``sample:<sample>``.
    macro : str or None
        The macro text that ran, whose SHA-256 is recorded (null when not given).

    Returns
    -------
    PlacementResult
        ``record["method"]`` is the uniform provenance entry (method
        ``bigstitcher``) with ``macro_sha256``, ``dataset_xml_sha256``, the
        engine's Fiji and BigStitcher versions (null: the engine task records
        them), the ``views`` (view, setup, cell, kind, fov_id, raw translation,
        calibration, legacy id) and the ``pairwise`` entries.

    Raises
    ------
    TypeError
        A table, config, review or view map of another type.
    ValueError
        A missing or unreadable file (naming the path); a view without a view-map
        entry or a view-map entry without a view (naming the view); a view
        without ``Stitching Transform`` or calibration, a non-translation, a
        calibration with ``c_x ≠ c_y`` (naming the view); a ``ViewTransform``
        whose affine holds a non-finite number (naming the setup, the transform
        and the file), whatever ``fallback``; a view map that breaks
        its rules; an unknown fallback; grids that do not match the table.
    IncompatibleGeometryError
        Grids outside the sample-frame geometry.
    """
    if not isinstance(table, TilePositionTable):
        raise TypeError("table must be a TilePositionTable")
    if not isinstance(config, BigStitcherConfig):
        raise TypeError("config must be a BigStitcherConfig")
    if not isinstance(review, ReviewPolicy):
        raise TypeError("review must be a ReviewPolicy")
    if fallback not in FALLBACKS:
        raise ValueError(f"fallback must be None or 'nominal'; got {fallback!r}")
    if not isinstance(sample, str) or not sample:
        raise ValueError("sample must be a nonempty string")
    if macro is not None and not isinstance(macro, str):
        raise TypeError("macro must be the macro text or None")
    ids = table.fov_ids
    if not ids:
        raise ValueError("the tile position table has no row")
    if not isinstance(grids, Mapping):
        raise TypeError("grids must map each fov_id of the table to its ReferenceGrid")
    _key_sets(ids, grids, "a grid")
    check_grids({fov: grids[fov] for fov in ids})
    views_map = _check_view_map(view_map, table)
    path = Path(dataset_xml)
    if not path.is_file():
        raise ValueError(f"dataset.xml not found: {path}")
    try:
        root = ElementTree.parse(path).getroot()
    except ElementTree.ParseError as exc:
        raise ValueError(f"{path} is not readable XML: {exc}") from None
    tiles = _setup_tiles(root, path)
    views = _views(root, path, tiles)
    entries = {int(row.view): row for row in views_map.itertuples(index=False)}
    unmapped = sorted(set(views) - set(entries))
    if unmapped:
        raise ValueError(f"{path}: views {_names(unmapped)} have no entry in the view map")
    absent = sorted(set(entries) - set(views))
    if absent:
        raise ValueError(f"{path}: views {_names(absent)} of the view map have no ViewRegistration")
    pairwise = _pairwise(root, path, {setup: view for view, (setup, *_) in views.items()})

    n = len(ids)
    index = {fov: i for i, fov in enumerate(ids)}
    spacing = {fov: grids[fov].metadata.spacing_zyx for fov in ids}
    kind = np.array(["missing"] * n, dtype=object)
    view_of = {}
    world = np.full((n, 3), np.nan)
    origin_raw = np.full((n, 3), np.nan)
    for view, entry in entries.items():
        if entry.kind == "padding":
            continue
        i = index[entry.fov_id]
        kind[i], view_of[i] = entry.kind, view
        _, w, calibration, _ = views[view]
        world[i] = (w[2], w[1], w[0])
        origin_raw[i] = _to_um(w, calibration, spacing[entry.fov_id])
    nominal = table.nominal_zyx()
    present = kind != "missing"
    placeable = (kind == "fov") & np.isfinite(origin_raw).all(axis=1)

    lo, hi = boxes(table, {fov: grids[fov] for fov in ids})
    found, padding_links = {}, 0
    for view_a, view_b, shift_xyz, correlation, digest, box in pairwise:
        ea, eb = entries[view_a], entries[view_b]
        if ea.kind == "padding" or eb.kind == "padding":
            padding_links += 1
            continue
        a, b, sign = index[ea.fov_id], index[eb.fov_id], 1.0
        if a == b:
            raise ValueError(f"{path}: a PairwiseResult links view {view_a} to itself")
        if a > b:
            a, b, sign = b, a, -1.0
        if (a, b) in found:
            raise ValueError(f"{path}: FOVs {ids[a]!r} and {ids[b]!r} have more than one PairwiseResult")
        shift = sign * _to_um(shift_xyz, views[view_a][2], spacing[ids[a]])
        found[(a, b)] = (shift, correlation)
    edges = sorted(found)
    component = _components(n, present, placeable, edges)
    joined = np.zeros(n, dtype=bool)
    for a, b in edges:
        if placeable[a] and placeable[b]:
            joined[a] = joined[b] = True
    candidates = np.unique(component[placeable])
    sizes = np.bincount(component[present]) if present.any() else np.zeros(0, dtype=np.int64)
    reference = int(candidates[np.argmax(sizes[candidates])]) if candidates.size else -2

    status = np.empty(n, dtype=object)
    flags = [[] for _ in range(n)]
    for i in range(n):
        if kind[i] == "missing":
            status[i] = "missing"
        elif kind[i] == "placeholder":
            status[i] = "placeholder"
        elif placeable[i] and component[i] == reference and joined[i]:
            status[i] = "image_supported"
        elif fallback == "nominal":
            status[i], flags[i] = "metadata_only", ["fallback"]
        else:
            status[i] = "failed"
    supported = status == "image_supported"
    correction_raw = origin_raw - nominal
    mean = correction_raw[supported].mean(axis=0) if supported.any() else None
    origin = np.full((n, 3), np.nan)
    shift_gauge = np.zeros(3) if mean is None else mean
    for mask in (supported, status == "placeholder"):
        origin[mask] = nominal[mask] + correction_raw[mask] - shift_gauge
    fell_back = status == "metadata_only"
    origin[fell_back] = nominal[fell_back]
    correction = origin - nominal

    rows = []
    for (a, b), (shift, correlation) in found.items():
        start, end = np.maximum(lo[a], lo[b]), np.minimum(hi[a], hi[b])
        overlaps = all(end[k] > start[k] for k in (1, 2))
        box = [v for s, e in zip(start, end) for v in ((float(s), float(e)) if overlaps else (np.nan, np.nan))]
        residual = np.nan
        if np.isfinite(origin[[a, b]]).all():
            difference = (origin[b] - origin[a]) - (nominal[b] - nominal[a]) - shift
            residual = float(np.sqrt(np.sum(difference * difference)))
        rows.append((ids[a], ids[b], *box, *(float(v) for v in shift), float(correlation), True, "", residual))
    rows.sort(key=lambda r: (index[r[0]], index[r[1]]))
    links = pd.DataFrame(rows, columns=list(LINK_COLUMNS)) if rows else empty_links()
    n_kept = np.zeros(n, dtype=np.int64)
    for a, b in edges:
        n_kept[a] += 1
        n_kept[b] += 1

    fovs = pd.DataFrame({"fov_id": ids, "status": list(status),
                         **{f"origin_{a}": origin[:, i] for i, a in enumerate(AXES)},
                         **{f"nominal_{a}": nominal[:, i] for i, a in enumerate(AXES)},
                         **{f"correction_{a}": correction[:, i] for i, a in enumerate(AXES)},
                         **{f"method_translation_{a}": world[:, i] for i, a in enumerate(AXES)},
                         "component": component, "n_links_kept": n_kept,
                         "flags": [";".join(f) for f in flags]}, columns=list(FOV_COLUMNS))
    position_ids = table.positions["position_id"]
    view_records = []
    for view, entry in sorted(entries.items()):
        setup, w, calibration, names = views[view]
        if entry.kind == "padding":
            legacy_id = 0
        else:
            i = index[entry.fov_id]
            legacy_id = i + 1 if pd.isna(position_ids[i]) else int(position_ids[i])
        view_records.append({"view": view, "setup": setup, "grid_column": int(entry.grid_column),
                             "grid_row": int(entry.grid_row), "kind": entry.kind, "fov_id": entry.fov_id,
                             "raw_translation_xyz": [float(v) for v in w],
                             "calibration_xyz": [float(v) for v in calibration], "transforms": names,
                             "legacy_id": legacy_id})
    method = provenance(StitchingSpec("bigstitcher", import_bigstitcher), config, STAGE)
    method.update({"macro_sha256": None if macro is None else hashlib.sha256(macro.encode("utf-8")).hexdigest(),
                   "dataset_xml_sha256": file_sha256(path), "dataset_xml_path": str(path.resolve()),
                   "fiji_version": None, "bigstitcher_version": None, "views": view_records,
                   "pairwise": [{"view_a": va, "view_b": vb, "translation_xyz": [float(v) for v in t],
                                 "correlation": float(r), "hash": h, "overlap_boundingbox": bb}
                                for va, vb, t, r, h, bb in pairwise],
                   "pairwise_with_padding": padding_links})
    record = {"format_version": FORMAT_VERSION, "stage": STAGE, "sample": sample, "frame": frame_record(sample),
              "gauge": "G1", "gauge_mean_correction_zyx": None if mean is None else [float(v) for v in mean],
              "method": method,
              "method_diagnostics": {"components": [int(c) for c in component], "reference_component": reference},
              "table": table.record,
              "inputs": {fov: input_record("bigstitcher" if i in view_of else "none", None, grids[fov])
                         for i, fov in enumerate(ids)},
              "fallback": fallback, "review": {"approvals": []}}
    return review_placement(PlacementResult(sample, fovs, links, record), review)


def view_map_of(result: PlacementResult) -> pd.DataFrame | None:
    """The view map an import recorded (``record["method"]["views"]``), or None for another placement."""
    method = result.record.get("method") or {}
    if method.get("method") != "bigstitcher" or "views" not in method:
        return None
    rows = [(v["view"], v["grid_column"], v["grid_row"], v["fov_id"], v["kind"]) for v in method["views"]]
    return pd.DataFrame(rows, columns=list(VIEW_MAP_COLUMNS)).astype(VIEW_MAP_DTYPES)


def legacy_tile_config_view(result: PlacementResult, *, img_col: int, img_row: int) -> pd.DataFrame:
    """The legacy view: ``tile_config_{sample}.csv`` from an import's raw translations and view map.

    Applies T3 to T7 of design/baselines/stitching.md to the raw world translation
    ``w`` of every view (FOVs, placeholders and padding, in view order, view
    ``j`` being grid row ``j``): the cast ``int`` (truncation towards zero), the
    shift by ``|min|`` per axis, the join with the grid (``col``, ``row``,
    ``id``, ``grid`` ``tile_{row}_{col}``), the non-overlap rectangles from the
    four row and column neighbours with ``img_col`` for the X extent and for the
    end in Y (as the script has it) and ``img_row`` for the start in Y, and the
    normalized boxes. ``id`` is the table's ``position_id`` of the view's FOV
    (the 1-based table row when the table has none) and 0 for padding. It reads
    no status, so it is the same for every ``fallback``. ``frame.to_csv()`` is
    the text ``create_tile_config.py`` writes.

    Parameters
    ----------
    result : PlacementResult
        A result of :func:`import_bigstitcher`.
    img_col, img_row : int
        The legacy tile width (X) and height (Y), px.

    Returns
    -------
    pandas.DataFrame
        The columns ``x``, ``y``, ``z``, ``fov_index``, ``col``, ``row``, ``id``,
        ``grid``, ``start_x``, ``start_y``, ``end_x``, ``end_y`` and the four
        ``_norm`` columns, one row per view, indexed by view.

    Raises
    ------
    ValueError
        A result that is not an import, views that are not 0 to n − 1, a
        non-finite raw translation, or an img_col or img_row that is not an
        integer ≥ 1.
    """
    if not isinstance(result, PlacementResult):
        raise TypeError("result must be a PlacementResult")
    for name, value in (("img_col", img_col), ("img_row", img_row)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be an integer ≥ 1; got {value!r}")
    method = result.record.get("method") or {}
    if method.get("method") != "bigstitcher" or "views" not in method:
        raise ValueError("the legacy tile configuration is a view of a BigStitcher import")
    views = sorted(method["views"], key=lambda v: v["view"])
    if [v["view"] for v in views] != list(range(len(views))):
        raise ValueError("the legacy tile configuration needs the views 0 to n − 1 (tile_{j}.tif)")
    raw = np.array([v["raw_translation_xyz"] for v in views], dtype=np.float64).reshape(len(views), 3)
    bad = [v["view"] for v, w in zip(views, raw) if not np.isfinite(w).all()]
    if bad:
        raise ValueError(f"views {_names(bad)} have a non-finite raw translation")
    cast = raw.astype(np.int64)                                     # T3: truncation towards zero
    cast = cast + np.abs(cast.min(axis=0))                          # T4: shift by |min| per axis
    x, y, z = (cast[:, k].tolist() for k in range(3))
    col = [int(v["grid_column"]) for v in views]
    row = [int(v["grid_row"]) for v in views]
    ids = [int(v["legacy_id"]) for v in views]
    names = [f"tile_{r}_{c}" for c, r in zip(col, row)]
    lookup = {name: k for k, name in enumerate(names)}              # T5: the first view of a name, as .values[0]

    def neighbor(name, values):
        k = lookup.get(name)
        return None if k is None or ids[k] == 0 else values[k]

    boxes_ = []                                                     # T6
    for k in range(len(views)):
        if ids[k] == 0:
            boxes_.append((0, 0, 0, 0))
            continue
        r, c, xk, yk = row[k], col[k], x[k], y[k]
        left = neighbor(f"tile_{r}_{c - 1}", x)
        up = neighbor(f"tile_{r - 1}_{c}", y)
        right = neighbor(f"tile_{r}_{c + 1}", x)
        down = neighbor(f"tile_{r + 1}_{c}", y)
        start_x = xk if left is None else int((left + img_col - xk) / 2 + 0.5) + xk
        start_y = yk if up is None else int((up + img_row - yk) / 2 + 0.5) + yk
        end_x = xk + img_col if right is None else int((xk + img_col - right) / 2 + 0.5 + right)
        end_y = yk + img_col if down is None else int((yk + img_col - down) / 2 + 0.5 + down)   # img_col, as T6
        boxes_.append((start_x, start_y, end_x, end_y))
    frame = pd.DataFrame({"x": x, "y": y, "z": z, "fov_index": list(range(len(views))), "col": col, "row": row,
                          "id": ids, "grid": names}, columns=list(LEGACY_COLUMNS[:8]))
    frame = frame.astype({c: "int64" for c in LEGACY_COLUMNS[:7]})
    for k, edge in enumerate(("start_x", "start_y", "end_x", "end_y")):
        frame[edge] = np.array([box[k] for box in boxes_], dtype=np.int64)
    for edge, axis in (("start_x", "x"), ("start_y", "y"), ("end_x", "x"), ("end_y", "y")):   # T7
        frame[f"{edge}_norm"] = frame[edge] - frame[axis]
        frame.loc[frame["id"] == 0, f"{edge}_norm"] = 0
    return frame
