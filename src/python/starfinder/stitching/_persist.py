"""The saved placement, option S1 (design/specs/stitching.md, "Persistence").

A placement is a folder, by convention ``<OUTPUT>/stitching/<sample>/``, holding
``placement.json`` (the record with the SHA-256 and dtypes of both tables),
``fovs.csv`` and ``links.csv``; a BigStitcher import also has ``view_map.csv``, and its
record holds the SHA-256 of the macro and of ``dataset.xml``. The files are written into a temporary sibling folder
that replaces the target only when all of them are written, so an interrupted save
leaves no partial folder. The approvals are kept beside them in ``approvals.json``, each keyed by the SHA-256 of
the two tables it approves, so that recording an approval changes none of the placement's files (W-875).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from dataclasses import replace
from pathlib import Path

from ._placement import FORMAT_VERSION, STAGE, PlacementResult, table_csv

RECORD_FILE = "placement.json"
TABLE_FILES = {"fovs": "fovs.csv", "links": "links.csv"}
VIEW_MAP_FILE = "view_map.csv"
APPROVALS_FILE = "approvals.json"
APPROVALS_STAGE = "placement_approvals"
APPROVAL_KEYS = ("reviewer", "time_utc", "note", "fovs_sha256", "links_sha256")
# Files a placement folder may hold; view_map.csv belongs to the BigStitcher import, approvals.json to the review.
PLACEMENT_FILES = (RECORD_FILE, *TABLE_FILES.values(), VIEW_MAP_FILE, APPROVALS_FILE)


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


def _approval_list(value, where):
    """The approvals of a record or of ``approvals.json``, each a dict with the keys of APPROVAL_KEYS as strings;
    ValueError naming ``where`` otherwise."""
    if not isinstance(value, list) or not all(isinstance(a, dict) for a in value):
        raise ValueError(f"{where}: the approvals must be a list of objects")
    for approval in value:
        wrong = [k for k in APPROVAL_KEYS if not isinstance(approval.get(k), str)]
        if wrong:
            raise ValueError(f"{where}: an approval lacks the string keys {', '.join(wrong)}")
    return [dict(a) for a in value]


def _joined(*groups):
    """The approvals of the groups in their order, each once."""
    out = []
    for group in groups:
        for approval in group:
            if approval not in out:
                out.append(approval)
    return out


def _read_approvals(directory, sample):
    """The approvals of ``<directory>/approvals.json`` ([] when there is none); ValueError when it is not an
    approvals file of the sample."""
    path = Path(directory) / APPROVALS_FILE
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text())
    except ValueError as error:
        raise ValueError(f"{path} is not a JSON file: {error}") from None
    if not isinstance(data, dict) or data.get("format_version") != FORMAT_VERSION \
            or data.get("stage") != APPROVALS_STAGE:
        raise ValueError(f"{path} is not a version {FORMAT_VERSION} approvals file")
    if data.get("sample") != sample:
        raise ValueError(f"{path} holds the approvals of sample {data.get('sample')!r}, not {sample!r}")
    return _approval_list(data.get("approvals"), str(path))


def _approvals_text(sample, approvals) -> bytes:
    return (json.dumps({"format_version": FORMAT_VERSION, "stage": APPROVALS_STAGE, "sample": sample,
                        "approvals": approvals}, indent=2, allow_nan=False) + "\n").encode("utf-8")


def saved_approvals(directory, sample, fovs_sha256, links_sha256) -> list:
    """The approvals of a saved placement's tables: those of ``approvals.json`` and, as written before W-875, of
    ``placement.json`` ``review.approvals``, whose ``fovs_sha256`` and ``links_sha256`` equal the given SHA-256.

    ValueError when ``approvals.json`` is not an approvals file of the sample or an approval is malformed.
    """
    directory = Path(directory)
    record = {}
    if (directory / RECORD_FILE).is_file():
        try:
            record = json.loads((directory / RECORD_FILE).read_text())
        except ValueError:
            record = {}
    review = record.get("review") if isinstance(record, dict) else None
    earlier = review.get("approvals") or [] if isinstance(review, dict) else []
    approvals = _joined(_approval_list(earlier, str(directory / RECORD_FILE)), _read_approvals(directory, sample))
    return [a for a in approvals if (a["fovs_sha256"], a["links_sha256"]) == (fovs_sha256, links_sha256)]


def save_placement(result: PlacementResult, directory) -> Path:
    """Write a placement folder: ``placement.json``, ``fovs.csv``, ``links.csv`` and the approvals; return the folder.

    Parameters
    ----------
    result : PlacementResult
        Any placement: from ``stitch``, the BigStitcher import or
        ``placement_from_table``.
    directory : str or Path
        The folder, by convention ``<OUTPUT>/stitching/<sample>/``.

    Returns
    -------
    pathlib.Path
        Placement folder containing the saved record and tables.

    Notes
    -----
    The tables are the checkpoint CSV form (float64 as ``%.17g``, so origins
    read back bit for bit); ``placement.json`` is the record with ``tables``,
    the path, SHA-256 and dtypes of each table. A BigStitcher import also gets
    ``view_map.csv`` (``view``, ``grid_column``, ``grid_row``, ``fov_id``,
    ``kind``, from ``record["method"]["views"]``), listed in ``tables`` with
    its SHA-256; the record keeps the macro's and ``dataset.xml``'s SHA-256.

    The approvals of ``record["review"]["approvals"]`` are written to
    ``approvals.json``, joined to those it already holds, and
    ``placement.json`` keeps ``review.approvals`` empty. Each approval holds
    the SHA-256 of the two tables it approves, which key it: it counts only
    for a placement with those tables. So saving an approved placement
    whose files are already in the folder, byte for byte, writes
    ``approvals.json`` alone and leaves the other files untouched; in a
    workflow, the placement task's outputs do not change and the task does
    not run again. Otherwise the folder is replaced as a whole, but for
    ``approvals.json``, which is kept. Remove ``approvals.json`` to withdraw
    every approval.

    Raises
    ------
    TypeError
        result is not a PlacementResult.
    FileExistsError
        directory is a file, or a folder holding files that are not placement
        files.
    ValueError
        The record holds a ``tables`` key or a malformed approval, or the
        folder's ``approvals.json`` is not the sample's approvals file.
    """
    from starfinder._records import _dtype_map, atomic_file, jsonable
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
    review = result.record.get("review")
    given = _approval_list(review.get("approvals") or [], "record['review']['approvals']") \
        if isinstance(review, dict) else []
    files, tables = {}, {}
    for name, (file_name, frame) in _tables(result).items():
        files[file_name] = table_csv(frame)
        tables[name] = {"path": file_name, "sha256": hashlib.sha256(files[file_name]).hexdigest(),
                        "dtypes": _dtype_map(frame)}
    record = dict(result.record, tables=tables)
    if isinstance(review, dict) and "approvals" in review:
        record["review"] = dict(review, approvals=[])
    files[RECORD_FILE] = (json.dumps(jsonable(record), indent=2, allow_nan=False) + "\n").encode("utf-8")
    kept = _read_approvals(directory, result.sample) if directory.is_dir() else []
    approvals = _joined(kept, given)
    if directory.is_dir() and _holds(directory, files):
        if approvals != kept:
            with atomic_file(directory / APPROVALS_FILE) as tmp:
                tmp.write_bytes(_approvals_text(result.sample, approvals))
        return directory
    directory.parent.mkdir(parents=True, exist_ok=True)
    tmp = directory.parent / f".{directory.name}.{uuid.uuid4().hex}.tmp"
    old = directory.parent / f".{directory.name}.{uuid.uuid4().hex}.old"
    tmp.mkdir()
    moved = False
    try:
        for file_name, data in files.items():
            _write_file(tmp / file_name, data)
        if approvals:
            _write_file(tmp / APPROVALS_FILE, _approvals_text(result.sample, approvals))
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


def _holds(directory, files) -> bool:
    """Whether the folder holds exactly these placement files with these bytes (``approvals.json`` aside)."""
    present = {p.name for p in directory.iterdir() if p.name != APPROVALS_FILE}
    return present == set(files) and all((directory / name).read_bytes() == data for name, data in files.items())


def load_placement(directory) -> PlacementResult:
    """Read a placement folder written by :func:`save_placement`, checking both tables' SHA-256.

    Parameters
    ----------
    directory : str or pathlib.Path
        Saved placement folder containing ``placement.json`` and its tables.

    Returns
    -------
    PlacementResult
        Placement reconstructed from the saved tables and record.

    Notes
    -----
    Returns a :class:`PlacementResult` whose tables equal the saved ones
    (float64 bit for bit) and whose record equals the saved record without
    ``tables``, but for ``review.approvals``: the approvals of
    ``approvals.json`` (and those an earlier ``placement.json`` held) whose
    table SHA-256 equal the saved tables'. A BigStitcher import's
    ``view_map.csv`` is checked too: against its recorded SHA-256 and against
    the views of the record.

    Raises
    ------
    FileNotFoundError
        The folder has no ``placement.json``.
    ValueError
        A record of another version or stage, or a table that is missing or
        whose SHA-256 differs from the record (naming the file and both hashes),
        or an ``approvals.json`` that is not the sample's approvals file.
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
    if not isinstance(record.get("review"), dict):
        return result
    approvals = saved_approvals(directory, record["sample"], tables["fovs"]["sha256"], tables["links"]["sha256"])
    return replace(result, record=dict(record, review=dict(record["review"], approvals=approvals)))
