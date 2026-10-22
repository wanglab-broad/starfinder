"""The saved placement, option S1 (docs/stitching-contract.md, "Persistence").

A placement is a folder, by convention ``<OUTPUT>/stitching/<sample>/``, holding
``placement.json`` (the record with the SHA-256 and dtypes of both tables),
``fovs.csv`` and ``links.csv``; a BigStitcher import also has ``view_map.csv``, and its
record holds the SHA-256 of the macro and of ``dataset.xml``. The files are written into a temporary sibling folder
that replaces the target only when all of them are written, so an interrupted save
leaves no partial folder.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path

from ._placement import FORMAT_VERSION, STAGE, PlacementResult, table_csv

RECORD_FILE = "placement.json"
TABLE_FILES = {"fovs": "fovs.csv", "links": "links.csv"}
VIEW_MAP_FILE = "view_map.csv"
# Files a placement folder may hold; view_map.csv belongs to the BigStitcher import.
PLACEMENT_FILES = (RECORD_FILE, *TABLE_FILES.values(), VIEW_MAP_FILE)


def _tables(result):
    """name → (file name, frame) of the tables a placement saves: fovs, links and an import's view map."""
    from ._bigstitcher import view_map_of
    tables = {name: (file_name, getattr(result, name)) for name, file_name in TABLE_FILES.items()}
    view_map = view_map_of(result)
    if view_map is not None:
        tables["view_map"] = (VIEW_MAP_FILE, view_map)
    return tables


def _write_file(path, data: bytes) -> None:
    """Write one file of the placement folder."""
    Path(path).write_bytes(data)


def save_placement(result: PlacementResult, directory) -> Path:
    """Write a placement folder: ``placement.json``, ``fovs.csv`` and ``links.csv``; return the folder.

    The tables are the checkpoint CSV form (float64 as ``%.17g``, so origins
    read back bit for bit); ``placement.json`` is the record with ``tables``,
    the path, SHA-256 and dtypes of each table. A BigStitcher import also gets
    ``view_map.csv`` (``view``, ``grid_column``, ``grid_row``, ``fov_id``,
    ``kind``, from ``record["method"]["views"]``), listed in ``tables`` with
    its SHA-256; the record keeps the macro's and ``dataset.xml``'s SHA-256. An
    existing placement folder is replaced as a whole.

    Parameters
    ----------
    result : PlacementResult
        Any placement: from ``stitch``, the BigStitcher import or
        ``placement_from_table``.
    directory : str or Path
        The folder, by convention ``<OUTPUT>/stitching/<sample>/``.

    Raises
    ------
    TypeError
        result is not a PlacementResult.
    FileExistsError
        directory is a file, or a folder holding files that are not placement
        files.
    """
    from starfinder._records import _dtype_map, jsonable
    if not isinstance(result, PlacementResult):
        raise TypeError("result must be a PlacementResult")
    directory = Path(directory)
    if directory.exists():
        if not directory.is_dir():
            raise FileExistsError(f"{directory} is a file, not a placement folder")
        foreign = sorted(p.name for p in directory.iterdir() if p.name not in PLACEMENT_FILES)
        if foreign:
            raise FileExistsError(f"{directory} holds files that are not placement files ({', '.join(foreign)}); "
                                  "save the placement in its own folder")
    if "tables" in result.record:
        raise ValueError("the placement record must not hold a 'tables' key; save_placement writes it")
    directory.parent.mkdir(parents=True, exist_ok=True)
    tmp = directory.parent / f".{directory.name}.{uuid.uuid4().hex}.tmp"
    old = directory.parent / f".{directory.name}.{uuid.uuid4().hex}.old"
    tmp.mkdir()
    moved = False
    try:
        tables = {}
        for name, (file_name, frame) in _tables(result).items():
            data = table_csv(frame)
            _write_file(tmp / file_name, data)
            tables[name] = {"path": file_name, "sha256": hashlib.sha256(data).hexdigest(),
                            "dtypes": _dtype_map(frame)}
        record = dict(result.record, tables=tables)
        text = json.dumps(jsonable(record), indent=2, allow_nan=False) + "\n"
        _write_file(tmp / RECORD_FILE, text.encode("utf-8"))
        if directory.exists():
            os.replace(directory, old)
            moved = True
        os.replace(tmp, directory)
    except BaseException:
        if moved and not directory.exists():
            os.replace(old, directory)
        raise
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(old, ignore_errors=True)
    return directory


def load_placement(directory) -> PlacementResult:
    """Read a placement folder written by :func:`save_placement`, checking both tables' SHA-256.

    Returns a :class:`PlacementResult` whose tables equal the saved ones
    (float64 bit for bit) and whose record equals the saved record without
    ``tables``. A BigStitcher import's ``view_map.csv`` is checked too: against
    its recorded SHA-256 and against the views of the record.

    Raises
    ------
    FileNotFoundError
        The folder has no ``placement.json``.
    ValueError
        A record of another version or stage, or a table that is missing or
        whose SHA-256 differs from the record (naming the file and both hashes).
    """
    from starfinder._records import _read_table, check_file
    directory = Path(directory)
    path = directory / RECORD_FILE
    if not path.is_file():
        raise FileNotFoundError(f"no saved placement at {path}")
    record = json.loads(path.read_text())
    if record.get("format_version") != FORMAT_VERSION or record.get("stage") != STAGE:
        raise ValueError(f"{path} is not a version {FORMAT_VERSION} placement record")
    tables = record.pop("tables", None)
    if not isinstance(tables, dict) or set(tables) - {"view_map"} != set(TABLE_FILES):
        raise ValueError(f"{path} does not record the tables {sorted(TABLE_FILES)}")
    frames = {}
    for name in tables:
        entry = tables[name]
        table_path = directory / entry["path"]
        check_file(table_path, entry["sha256"], f"placement table {name}")
        frames[name] = _read_table(table_path, entry["dtypes"])
    result = PlacementResult(record["sample"], frames["fovs"], frames["links"], record)
    expected = _tables(result).get("view_map")
    if ("view_map" in frames) != (expected is not None):
        raise ValueError(f"{path}: the view map must be saved exactly for a BigStitcher import")
    if expected is not None and table_csv(frames["view_map"]) != table_csv(expected[1]):
        raise ValueError(f"placement table view_map {directory / VIEW_MAP_FILE} differs from the views of the record")
    return result
