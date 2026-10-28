"""The tile position table, its three converters and the stage-axis mapping.

design/specs/stitching.md ("Tile position table", "Stage-axis mapping") and
design/specs/stitching-algorithms.md ("Converters"). Every converter returns a table in µm and
reads no image: the tile size and spacing it needs are arguments, as values.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Integral, Real
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pandas as pd

from starfinder._records import file_sha256, json_record

COLUMNS = ("fov_id", "position_id", "nominal_z", "nominal_y", "nominal_x", "source", "grid_column", "grid_row")
DTYPES = {"fov_id": "str", "position_id": "Int64", "nominal_z": "float64", "nominal_y": "float64",
          "nominal_x": "float64", "source": "str", "grid_column": "Int64", "grid_row": "Int64"}
NOMINAL = ("nominal_z", "nominal_y", "nominal_x")
SOURCES = ("stage", "grid", "supplied")
ORDERS = ("column_major_down_right", "column_major_down_right_serpentine", "row_major_right_down",
          "row_major_right_down_serpentine")
# Supplied-table units and their conversion to µm: (operation, factor).
UNITS = {"µm": None, "um": None, "nm": ("divide", 1000.0), "mm": ("multiply", 1000.0)}
_AXES = ("+x", "-x", "+y", "-y")
_TABLE_SOURCE = re.compile(r"leica:.+|grid|table:.+")


def _names(values, limit=10):
    values = [str(v) for v in values]
    return ", ".join(values[:limit]) + (f" and {len(values) - limit} more" if len(values) > limit else "")


@dataclass(frozen=True, eq=False)
class TilePositionTable:
    """One row per acquired FOV, in acquisition order, with its nominal origin in µm.

    ``positions`` has the columns ``fov_id`` (unique), ``position_id`` (the
    acquisition's own identifier, ``Int64``, missing when there is none),
    ``nominal_z``, ``nominal_y``, ``nominal_x`` (float64, µm, finite), ``source``
    (``stage``, ``grid`` or ``supplied``) and ``grid_column``, ``grid_row``
    (``Int64``, missing when the converter knows no cell). The rows' order is the
    order of every per-FOV table of a placement. ``source`` names the converter's
    input (``"leica:<path>"``, ``"grid"`` or ``"table:<path>"``); ``record`` holds
    the converter, its parameters, the stage-axis mapping and the input's SHA-256,
    as JSON values. ``unit`` is always ``µm``. Only the converters
    (:func:`positions_from_leica`, :func:`positions_from_grid`,
    :func:`positions_from_table`) are meant to build tables; a table may have no
    row, which :func:`placement_from_table` refuses.

    Raises
    ------
    TypeError
        positions is not a DataFrame or record not a mapping.
    ValueError
        Other columns, a missing, empty or repeated ``fov_id``, a non-finite
        nominal value, an unknown row source, a unit other than ``µm`` or an
        unknown table source.
    """

    positions: pd.DataFrame
    source: str
    record: dict = field(default_factory=dict)
    unit: str = "µm"

    def __post_init__(self):
        if self.unit != "µm":
            raise ValueError(f"a tile position table is in µm; got unit {self.unit!r}")
        if not isinstance(self.source, str) or not _TABLE_SOURCE.fullmatch(self.source):
            raise ValueError(f"source must be 'leica:<path>', 'grid' or 'table:<path>'; got {self.source!r}")
        if not isinstance(self.record, Mapping):
            raise TypeError("record must be a mapping")
        frame = self.positions
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("positions must be a pandas DataFrame")
        if tuple(frame.columns) != COLUMNS:
            raise ValueError(f"positions must have the columns {list(COLUMNS)}; got {list(frame.columns)}")
        ids = list(frame["fov_id"])
        bad = [i for i, v in enumerate(ids) if not isinstance(v, str) or not v]
        if bad:
            raise ValueError(f"fov_id must be a nonempty string in every row; rows {_names(bad)} are not")
        repeated = sorted(k for k, n in Counter(ids).items() if n > 1)
        if repeated:
            raise ValueError(f"fov_id must be unique; repeated: {_names(repeated)}")
        values = frame[list(NOMINAL)].to_numpy(dtype=np.float64)
        bad = [ids[i] for i in np.flatnonzero(~np.isfinite(values).all(axis=1))]
        if bad:
            raise ValueError(f"every nominal value must be finite; FOVs {_names(bad)} are not")
        unknown = sorted({v for v in frame["source"] if v not in SOURCES}, key=str)
        if unknown:
            raise ValueError(f"row source must be one of {list(SOURCES)}; got {_names(unknown)}")
        object.__setattr__(self, "positions", frame.astype(DTYPES).reset_index(drop=True))
        object.__setattr__(self, "record", json_record(dict(self.record)))

    @property
    def fov_ids(self) -> list[str]:
        """The ``fov_id`` column, in table order."""
        return list(self.positions["fov_id"])

    def nominal_zyx(self) -> np.ndarray:
        """The ``(n, 3)`` float64 array of nominal origins (z, y, x), µm, a copy."""
        return self.positions[list(NOMINAL)].to_numpy(dtype=np.float64, copy=True)


@dataclass(frozen=True)
class StageAxisMapping:
    """Where the stage axes go in the image: each of ``x`` and ``y`` is ``"+x"``, ``"-x"``, ``"+y"`` or ``"-y"``.

    ``x`` says where stage +X goes, ``y`` where stage +Y goes; the two name
    different image axes. ``source`` says where the mapping comes from
    (``"declared"`` by the caller, ``"legacy"`` for the workflow's translation of
    a legacy configuration). For LN sample1 it is ``StageAxisMapping(x="+x",
    y="+y")`` (design/specs/stitching.md, "Stage-axis mapping").

    Raises
    ------
    ValueError
        An entry that is not one of the four, both naming one image axis, or an
        empty source.
    """

    x: str
    y: str
    source: str = "declared"

    def __post_init__(self):
        for name in ("x", "y"):
            if getattr(self, name) not in _AXES:
                raise ValueError(f"StageAxisMapping.{name} must be one of {list(_AXES)}; got {getattr(self, name)!r}")
        if self.x[1] == self.y[1]:
            raise ValueError(f"StageAxisMapping names image axis {self.x[1]!r} twice (x={self.x!r}, y={self.y!r}); "
                             "stage X and Y must go to different image axes")
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("StageAxisMapping.source must be a nonempty string")


def _real(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number; got {value!r}")
    return float(value)


def _pair(value, name):
    if (isinstance(value, (str, bytes)) or not isinstance(value, (Sequence, np.ndarray))
            or len(value) != 2):
        raise ValueError(f"{name} must be a (y, x) pair; got {value!r}")
    return tuple(value)


def _grid_geometry(overlap, tile_shape_yx, spacing_yx, what):
    """Checked (overlap_yx, tile_shape_yx, spacing_yx) of a regular grid; ValueError names the field."""
    if isinstance(overlap, (Sequence, np.ndarray)) and not isinstance(overlap, (str, bytes)):
        values = _pair(overlap, f"{what} overlap")
    else:
        values = (overlap, overlap)
    overlap_yx = []
    for axis, value in zip("yx", values):
        value = _real(value, f"{what} overlap ({axis})")
        if not (math.isfinite(value) and 0 <= value < 1):
            raise ValueError(f"{what} overlap ({axis}) must satisfy 0 ≤ overlap < 1; got {value}")
        overlap_yx.append(value)
    tile = []
    for axis, value in zip("yx", _pair(tile_shape_yx, f"{what} tile_shape_yx")):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{what} tile_shape_yx ({axis}) must be an integer ≥ 1; got {value!r}")
        tile.append(int(value))
    spacing = []
    for axis, value in zip("yx", _pair(spacing_yx, f"{what} spacing_yx")):
        value = _real(value, f"{what} spacing_yx ({axis})")
        if not (math.isfinite(value) and value > 0):
            raise ValueError(f"{what} spacing_yx ({axis}) must be finite and > 0 µm; got {value}")
        spacing.append(value)
    return tuple(overlap_yx), tuple(tile), tuple(spacing)


def _steps(overlap_yx, tile_yx, spacing_yx):
    """The grid step per axis, ``(1 − overlap) × tile × spacing`` µm, evaluated left to right."""
    return tuple((1 - o) * n * s for o, n, s in zip(overlap_yx, tile_yx, spacing_yx))


@dataclass(frozen=True)
class GridSnap:
    """Snap stage positions to a regular grid: the step is ``(1 − overlap) × tile_shape_yx × spacing_yx``.

    ``overlap`` is a fraction, ``0 ≤ overlap < 1``, one value or ``(y, x)``;
    ``tile_shape_yx`` the reference grids' Y and X sizes (integers ≥ 1, px);
    ``spacing_yx`` their Y and X spacing (finite, > 0, µm per px). None of the
    three has a default; the names, units and rules are those of
    :func:`positions_from_grid`.

    Raises
    ------
    ValueError
        A field outside its rule, naming it.
    """

    overlap: float | tuple[float, float]
    tile_shape_yx: tuple[int, int]
    spacing_yx: tuple[float, float]

    def __post_init__(self):
        _grid_geometry(self.overlap, self.tile_shape_yx, self.spacing_yx, "GridSnap")


def _cell(k, columns, rows, order):
    """(column, row) of the k-th FOV in a grid scanned by order."""
    if order.startswith("column_major"):
        column, row = divmod(k, rows)
        if order.endswith("_serpentine") and column % 2:
            row = rows - 1 - row
    else:
        row, column = divmod(k, columns)
        if order.endswith("_serpentine") and row % 2:
            column = columns - 1 - column
    return column, row


def _fov_ids(fov_ids):
    if isinstance(fov_ids, (str, bytes)) or not isinstance(fov_ids, Sequence):
        raise ValueError("fov_ids must be a sequence of FOV IDs")
    ids = list(fov_ids)
    if not ids:
        raise ValueError("fov_ids must not be empty")
    return ids


def _frame(fov_ids, nominal, source, position_ids=None, cells=None):
    n = len(fov_ids)
    return pd.DataFrame({
        "fov_id": list(fov_ids),
        "position_id": pd.array(position_ids if position_ids is not None else [None] * n, dtype="Int64"),
        "nominal_z": np.asarray([p[0] for p in nominal], dtype=np.float64),
        "nominal_y": np.asarray([p[1] for p in nominal], dtype=np.float64),
        "nominal_x": np.asarray([p[2] for p in nominal], dtype=np.float64),
        "source": [source] * n,
        "grid_column": pd.array([c for c, _ in cells] if cells is not None else [None] * n, dtype="Int64"),
        "grid_row": pd.array([r for _, r in cells] if cells is not None else [None] * n, dtype="Int64"),
    }, columns=list(COLUMNS))


def _grid_nominal(cells, step_yx):
    return [(0.0, row * step_yx[0], column * step_yx[1]) for column, row in cells]


def positions_from_grid(*, columns: int, rows: int, order: str, tile_shape_yx: tuple[int, int],
                        spacing_yx: tuple[float, float], overlap: float | tuple[float, float],
                        fov_ids: Sequence[str]) -> TilePositionTable:
    """Tile position table of a regular grid (design/specs/stitching-algorithms.md, "Regular grid").

    FOV ``k`` of ``fov_ids`` goes to a cell by ``order``:
    ``column_major_down_right`` (column ``k // rows``, row ``k % rows``),
    ``row_major_right_down`` (row ``k // columns``, column ``k % columns``) or
    their ``_serpentine`` forms (every second column scanned up, every second row
    scanned left). Nominal ``y = row × step_y``, ``x = column × step_x`` and
    ``z = 0``, with ``step = (1 − overlap) × tile_shape_yx × spacing_yx`` per axis
    (µm). No argument has a default.

    Parameters
    ----------
    columns, rows : int
        Grid size; ``columns × rows`` at least the number of FOVs.
    order : str
        One of the four orders above.
    tile_shape_yx : tuple[int, int]
        The reference grids' Y and X sizes, px (integers ≥ 1).
    spacing_yx : tuple[float, float]
        The reference grids' Y and X spacing, µm per px (finite, > 0).
    overlap : float or tuple[float, float]
        Fraction, ``0 ≤ overlap < 1``; one value or ``(y, x)``.
    fov_ids : Sequence[str]
        FOV IDs in scan order.

    Returns
    -------
    TilePositionTable
        ``source`` ``grid``; every row ``source`` ``grid`` with its cell and no
        ``position_id``; the record holds the parameters and the steps.

    Raises
    ------
    ValueError
        An argument outside its rule (naming it), an unknown order, or more
        FOVs than cells.
    """
    for name, value in (("columns", columns), ("rows", rows)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"positions_from_grid {name} must be an integer ≥ 1; got {value!r}")
    if order not in ORDERS:
        raise ValueError(f"positions_from_grid order must be one of {list(ORDERS)}; got {order!r}")
    overlap_yx, tile, spacing = _grid_geometry(overlap, tile_shape_yx, spacing_yx, "positions_from_grid")
    ids = _fov_ids(fov_ids)
    if len(ids) > columns * rows:
        raise ValueError(f"positions_from_grid has {len(ids)} FOVs for {columns} × {rows} = {columns * rows} cells")
    cells = [_cell(k, int(columns), int(rows), order) for k in range(len(ids))]
    step = _steps(overlap_yx, tile, spacing)
    record = {"converter": "positions_from_grid", "source": "grid", "columns": int(columns), "rows": int(rows),
              "order": order, "tile_shape_yx": list(tile), "spacing_yx": list(spacing), "overlap_yx": list(overlap_yx),
              "step_yx_um": list(step)}
    return TilePositionTable(_frame(ids, _grid_nominal(cells, step), "grid", cells=cells), "grid", record)


def _local(tag):
    return tag.rsplit("}", 1)[-1]


def _stage_points(path):
    """(PositionID, StageXPos, StageYPos, other attributes) of every XYZStagePointDefinition, in file order."""
    try:
        root = ElementTree.parse(path).getroot()
    except ElementTree.ParseError as exc:
        raise ValueError(f"stage file {path} is not readable XML: {exc}") from None
    points = []
    for element in root.iter():
        if _local(element.tag) != "XYZStagePointDefinition":
            continue
        attributes = dict(element.attrib)
        missing = [k for k in ("PositionID", "StageXPos", "StageYPos") if k not in attributes]
        if missing:
            raise ValueError(f"stage file {path}: a stage point lacks {', '.join(missing)}")
        try:
            pid = int(attributes.pop("PositionID"))
        except ValueError:
            raise ValueError(f"stage file {path}: PositionID must be an integer") from None
        try:
            x, y = float(attributes.pop("StageXPos")), float(attributes.pop("StageYPos"))
        except ValueError:
            raise ValueError(f"stage file {path}: PositionID {pid} has a non-numeric stage position") from None
        points.append((pid, x, y, attributes))
    if not points:
        raise ValueError(f"stage file {path} holds no XYZStagePointDefinition")
    return points


def _snap_step(values):
    """Most frequent nonzero absolute consecutive difference, rounded to 0.1 µm; a tie goes to the smaller."""
    differences = np.round(np.abs(np.diff(np.asarray(values, dtype=np.float64))), 1)
    differences = differences[differences > 0]
    if differences.size == 0:
        return None
    steps, counts = np.unique(differences, return_counts=True)
    return float(steps[np.argmax(counts)])


def _select_points(points, position_ids, path):
    """The points of position_ids, in file order: each an integer PositionID the file holds, none repeated."""
    wanted = list(position_ids)
    if not wanted or any(isinstance(p, (bool, np.bool_)) or not isinstance(p, Integral) for p in wanted):
        raise ValueError(f"position_ids must be a nonempty list of integer PositionIDs; got {wanted!r}")
    repeated = sorted(k for k, n in Counter(wanted).items() if n > 1)
    if repeated:
        raise ValueError(f"position_ids repeats PositionID {_names(repeated)}")
    held = {p[0] for p in points}
    absent = [p for p in wanted if p not in held]
    if absent:
        raise ValueError(f"PositionID {_names(absent)} is not in stage file {path}")
    keep = set(wanted)
    return [p for p in points if p[0] in keep]


def positions_from_leica(path, *, stage_axes: StageAxisMapping, fov_ids: Mapping[int, str],
                         snap_to_grid: GridSnap | None = None, position_ids=None) -> TilePositionTable:
    """Tile position table of a Leica LAS X stage file (design/specs/stitching-algorithms.md, "Leica stage file").

    Reads every ``XYZStagePointDefinition`` (``PositionID``, ``StageXPos`` and
    ``StageYPos`` in metres), converts to µm (× 1e6), sends stage +X and +Y where
    ``stage_axes`` names, shifts so that the smallest y and x are 0 and sets z to
    0 (the stage file gives no z offset between tiles). Rows keep the file's
    order. Other attributes (``Sections``, ``StackBegin``, ``StageZPos``, ...) are
    kept in the record, not used.

    With ``position_ids`` only those positions are read (a whole-acquisition
    file and a sample's): they are selected first, and the axis mapping, the
    shift, the snap step, the cells and the clash check see only them, so a
    file that holds other samples' positions gives the table of a file that
    holds only these.

    With ``snap_to_grid`` the positions are replaced by a regular grid: the step
    for the cells is the most frequent absolute difference of consecutive y
    positions (the scan's fast axis), rounded to 0.1 µm, a tie going to the
    smaller value (x when every y difference is 0); ``column = floor(x / step +
    0.5)``, ``row = floor(y / step + 0.5)``; the nominal positions are those of
    :func:`positions_from_grid` for the cells, with the snap's step, and the rows'
    ``source`` is ``grid``.

    Parameters
    ----------
    path : str or Path
        The stage file (``.maf``).
    stage_axes : StageAxisMapping
        Required; no default.
    fov_ids : Mapping[int, str]
        ``PositionID`` → FOV ID; every ``PositionID`` read (of the file, or of
        ``position_ids``) must be a key.
    snap_to_grid : GridSnap or None
        The grid to snap to, or None (default) to keep the stage positions.
    position_ids : Sequence[int] or None
        The ``PositionID`` values to read, each in the file once; None
        (default) reads every position of the file.

    Returns
    -------
    TilePositionTable
        ``source`` ``leica:<path>``; ``position_id`` holds the ``PositionID``;
        rows ``stage`` (cells missing) or, snapped, ``grid`` with their cells.
        The record holds the path, the file's SHA-256, the mapping, the FOV IDs,
        the stage values and attributes of the positions read, and the snap
        with its step.

    Raises
    ------
    TypeError
        stage_axes is not a StageAxisMapping or snap_to_grid not a GridSnap.
    ValueError
        An unreadable file, a repeated ``PositionID``, an empty or repeated
        ``position_ids`` or one the file does not hold, a ``PositionID`` without
        a FOV ID, a non-finite position, or (snapped) two positions in one cell,
        each naming them.
    FileNotFoundError
        The file does not exist.
    """
    if not isinstance(stage_axes, StageAxisMapping):
        raise TypeError("stage_axes must be a StageAxisMapping; it has no default")
    if snap_to_grid is not None and not isinstance(snap_to_grid, GridSnap):
        raise TypeError("snap_to_grid must be a GridSnap or None")
    if not isinstance(fov_ids, Mapping):
        raise TypeError("fov_ids must map each PositionID to a FOV ID")
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"stage file not found: {path}")
    points = _stage_points(path)
    repeated = sorted(k for k, n in Counter(p[0] for p in points).items() if n > 1)
    if repeated:
        raise ValueError(f"stage file {path} repeats PositionID {_names(repeated)}")
    if position_ids is not None:
        points = _select_points(points, position_ids, path)
    unmapped = [p[0] for p in points if p[0] not in fov_ids]
    if unmapped:
        raise ValueError(f"PositionID {_names(unmapped)} of {path} has no FOV ID in fov_ids")
    bad = [p[0] for p in points if not (math.isfinite(p[1]) and math.isfinite(p[2]))]
    if bad:
        raise ValueError(f"stage file {path}: PositionID {_names(bad)} has a non-finite stage position")

    image = {"x": np.zeros(len(points)), "y": np.zeros(len(points))}
    for k, (_, x_m, y_m, _) in enumerate(points):
        for value, target in ((x_m * 1e6, stage_axes.x), (y_m * 1e6, stage_axes.y)):
            image[target[1]][k] += value if target[0] == "+" else -value
    y = image["y"] - image["y"].min()
    x = image["x"] - image["x"].min()
    pids = [p[0] for p in points]
    ids = [fov_ids[pid] for pid in pids]
    record = {"converter": "positions_from_leica", "source": f"leica:{path.resolve()}", "path": str(path.resolve()),
              "sha256": file_sha256(path),
              "stage_axes": {"x": stage_axes.x, "y": stage_axes.y, "source": stage_axes.source},
              "fov_ids": {str(pid): fov for pid, fov in zip(pids, ids)},
              "stage_positions": [{"position_id": pid, "stage_x_m": x_m, "stage_y_m": y_m, "attributes": attributes}
                                  for pid, x_m, y_m, attributes in points],
              "snap_to_grid": None}
    if snap_to_grid is None:
        nominal = [(0.0, float(b), float(a)) for b, a in zip(y, x)]
        return TilePositionTable(_frame(ids, nominal, "stage", position_ids=pids), record["source"], record)

    overlap_yx, tile, spacing = _grid_geometry(snap_to_grid.overlap, snap_to_grid.tile_shape_yx,
                                               snap_to_grid.spacing_yx, "GridSnap")
    axis, step = "y", _snap_step(y)
    if step is None:
        axis, step = "x", _snap_step(x)
    if step is None:
        cells = [(0, 0)] * len(points)
    else:
        cells = [(int(math.floor(a / step + 0.5)), int(math.floor(b / step + 0.5))) for a, b in zip(x, y)]
    shared = {}
    for pid, cell in zip(pids, cells):
        shared.setdefault(cell, []).append(pid)
    clashes = {cell: group for cell, group in shared.items() if len(group) > 1}
    if clashes:
        described = "; ".join(f"PositionID {_names(group)} in column {c}, row {r}" for (c, r), group in clashes.items())
        raise ValueError(f"snapping {path} puts two positions in one cell: {described}")
    grid_step = _steps(overlap_yx, tile, spacing)
    record["snap_to_grid"] = {"overlap_yx": list(overlap_yx), "tile_shape_yx": list(tile), "spacing_yx": list(spacing),
                              "cell_step_um": step, "cell_step_axis": axis, "step_yx_um": list(grid_step)}
    return TilePositionTable(_frame(ids, _grid_nominal(cells, grid_step), "grid", position_ids=pids, cells=cells),
                             record["source"], record)


def _to_um(values, unit):
    conversion = UNITS[unit]
    if conversion is None:
        return values
    operation, factor = conversion
    return values / factor if operation == "divide" else values * factor


def positions_from_table(path_or_frame, *, unit: str = "µm") -> TilePositionTable:
    """Tile position table of a supplied CSV or frame (design/specs/stitching-algorithms.md, "Supplied table").

    Reads ``fov_id``, ``nominal_z``, ``nominal_y`` and ``nominal_x`` (other
    columns are ignored) and converts the nominal values from ``unit`` to µm:
    ``µm`` and ``um`` are copied as read, ``nm`` divided by 1000 and ``mm``
    multiplied by 1000. A CSV is read as text and each value parsed as a float64
    (``float``), so a value written with round-trip precision is read back
    bitwise.

    Parameters
    ----------
    path_or_frame : str, Path or pandas.DataFrame
        The CSV file or the frame.
    unit : str
        ``µm`` (default), ``um``, ``nm`` or ``mm``.

    Returns
    -------
    TilePositionTable
        ``source`` ``table:<path>`` (``table:frame`` for a frame); every row
        ``supplied``, without ``position_id`` or cell. The record holds the
        unit, the path and the file's SHA-256 (null for a frame).

    Raises
    ------
    ValueError
        An unknown unit, a missing column, a repeated or empty ``fov_id``, or a
        value that is not a finite number (naming the FOVs).
    FileNotFoundError
        The file does not exist.
    """
    if unit not in UNITS:
        raise ValueError(f"unit must be one of {list(UNITS)}; got {unit!r}")
    if isinstance(path_or_frame, pd.DataFrame):
        frame, path = path_or_frame, None
    else:
        path = Path(path_or_frame)
        if not path.is_file():
            raise FileNotFoundError(f"position table not found: {path}")
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in ("fov_id", *NOMINAL) if c not in frame.columns]
    if missing:
        raise ValueError(f"the position table lacks the columns {missing}")
    ids = list(frame["fov_id"])
    empty = [i for i, v in enumerate(ids) if not isinstance(v, str) or not v]
    if empty:
        raise ValueError(f"fov_id must be a nonempty string; rows {_names(empty)} are not")
    repeated = sorted(k for k, n in Counter(ids).items() if n > 1)
    if repeated:
        raise ValueError(f"fov_id must be unique; repeated: {_names(repeated)}")
    columns = []
    for column in NOMINAL:
        values, bad = [], []
        for fov, value in zip(ids, frame[column]):
            try:
                if isinstance(value, (bool, np.bool_)):
                    raise ValueError
                number = float(value)
            except (TypeError, ValueError):
                bad.append(fov)
                number = math.nan
            if not math.isfinite(number):
                bad.append(fov)
            values.append(number)
        if bad:
            raise ValueError(f"{column} must be a finite number; FOVs {_names(sorted(set(bad), key=ids.index))} "
                             "are not")
        columns.append(_to_um(np.asarray(values, dtype=np.float64), unit))
    nominal = list(zip(*columns))
    source = "table:frame" if path is None else f"table:{path.resolve()}"
    record = {"converter": "positions_from_table", "source": source, "unit": unit,
              "path": None if path is None else str(path.resolve()),
              "sha256": None if path is None else file_sha256(path)}
    return TilePositionTable(_frame(ids, nominal, "supplied"), source, record)
