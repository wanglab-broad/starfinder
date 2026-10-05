"""The saved layout of an assignment (docs/assignment-contract.md, "Persistence", option L1).

``<checkpoint dir>/<fov_id>/assignment/<name>/`` holds ``molecules``, ``cells``,
``counts`` and (with nuclei) ``nuclei`` as CSV or Parquet through the checkpoint table
writer, ``assignment.json`` and the label images of the table "Label images of a
checkpointed assignment": ``territories.tif`` with an expansion, and ``cell_labels.tif``
and ``nucleus_labels.tif`` only for inputs that are not saved under their run. A saved
input is linked by its relative path, never copied.
"""
from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

from starfinder.segmentation._labels import array_sha256
from starfinder.segmentation._persist import check_file, file_sha256, json_record, read_labels, write_labels

from ._assign import FORMAT_VERSION, AssignmentResult

RECORD_FILE = "assignment.json"
TABLES = ("molecules", "cells", "counts", "nuclei")
TERRITORIES_FILE = "territories.tif"
CELL_LABELS_FILE = "cell_labels.tif"
NUCLEUS_LABELS_FILE = "nucleus_labels.tif"


def assignment_directory(fov_directory, name) -> Path:
    """``<fov checkpoint dir>/assignment/<name>``."""
    return Path(fov_directory) / "assignment" / name


def _files(directory):
    from starfinder.io._checkpoint import TABLE_FORMATS
    directory = Path(directory)
    return ([directory / RECORD_FILE] + [directory / f"{t}.{f}" for t in TABLES for f in TABLE_FORMATS]
            + [directory / n for n in (TERRITORIES_FILE, CELL_LABELS_FILE, NUCLEUS_LABELS_FILE)])


def check_writable(directory, overwrite) -> None:
    """FileExistsError when the assignment folder holds saved files and overwrite is False."""
    existing = [path.name for path in _files(directory) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"assignment folder {directory} exists ({', '.join(existing)}); "
                              "pass overwrite=True to replace it")


def _link(directory, path):
    return Path(os.path.relpath(Path(path).resolve(), Path(directory).resolve())).as_posix()


def write_assignment(directory, result: AssignmentResult, *, metadata, cells_saved=None, nuclei_saved=None,
                     same_run=False, table_format="csv") -> AssignmentResult:
    """Write an assignment folder; return the result with the saved record.

    ``metadata`` is the label grid's ``ImageMetadata``. ``cells_saved`` and
    ``nuclei_saved`` are the paths of the inputs' ``segmentation/<run>/labels.tif``
    when they are saved under their run (linked), else None (written here).
    ``same_run`` says the nuclei are the cell run itself (one file serves both).
    The returned record is the JSON form written to ``assignment.json``.
    """
    from starfinder.io._checkpoint import _write_table, write_json
    directory = Path(directory)
    for path in _files(directory):
        path.unlink(missing_ok=True)
    directory.mkdir(parents=True, exist_ok=True)
    record = json_record(result.record)
    files = {}
    for name in TABLES:
        frame = getattr(result, name)
        if frame is None:
            continue
        file_name, dtypes = _write_table(frame, directory, name, table_format)
        files[name] = {"path": file_name, "sha256": file_sha256(directory / file_name), "dtypes": dtypes}

    def label_file(labels, saved, file_name, key):
        if saved is not None:
            return {"path": _link(directory, saved), "sha256": file_sha256(saved)}, True
        entry = {"path": file_name, "sha256": write_labels(directory / file_name, labels, metadata)}
        files[key] = dict(entry)
        return entry, False

    inputs = record["inputs"]
    cell_file, saved = label_file(result.cell_labels, cells_saved, CELL_LABELS_FILE, "cell_labels")
    inputs["cells"] = dict(inputs["cells"], saved_under_run=saved, file=cell_file)
    if result.nucleus_labels is not None:
        if same_run:
            nucleus_file, saved = dict(cell_file), inputs["cells"]["saved_under_run"]
        else:
            nucleus_file, saved = label_file(result.nucleus_labels, nuclei_saved, NUCLEUS_LABELS_FILE,
                                             "nucleus_labels")
        inputs["nuclei"] = dict(inputs["nuclei"], saved_under_run=saved, file=nucleus_file)
    if result.territories is not None:
        expanded = {"path": TERRITORIES_FILE,
                    "sha256": write_labels(directory / TERRITORIES_FILE, result.territories, metadata)}
        files["territories"] = dict(expanded)
        expansion = record["expansion"]
        expansion["original"] = dict(expansion["original"], path=cell_file["path"], file_sha256=cell_file["sha256"])
        expansion["expanded"] = dict(expansion["expanded"], path=TERRITORIES_FILE, file_sha256=expanded["sha256"])
    record["files"] = files
    record = json_record(record)
    write_json(record, directory / RECORD_FILE)
    return replace(result, record=record)


def _path(directory, entry):
    """A recorded path inside or linked from the folder, normalized (``../../segmentation/…`` resolved lexically)."""
    return Path(os.path.normpath(Path(directory) / entry["path"]))


def _labels(directory, entry, expected, what):
    path = _path(directory, entry)
    labels, _ = read_labels(path)
    if array_sha256(labels) != expected:
        raise ValueError(f"{what} {path} holds labels with SHA-256 {array_sha256(labels)}, the record {expected}")
    return labels


def read_assignment(directory, name) -> AssignmentResult:
    """Read an assignment folder and its linked label files, checking every recorded SHA-256."""
    from starfinder.io._checkpoint import _read_table
    directory = Path(directory)
    path = directory / RECORD_FILE
    if not path.is_file():
        raise FileNotFoundError(f"no saved assignment {name!r} at {path}")
    record = json.loads(path.read_text())
    if record.get("format_version") != FORMAT_VERSION or record.get("stage") != "assignment":
        raise ValueError(f"{path} is not a version {FORMAT_VERSION} assignment record")
    if record.get("name") != name:
        raise ValueError(f"{path} records assignment {record.get('name')!r}, not {name!r}")
    files, inputs = record["files"], record["inputs"]
    for key, entry in files.items():
        check_file(_path(directory, entry), entry["sha256"], f"assignment file {key}")
    for key in ("cells", "nuclei"):
        if inputs.get(key) is not None:
            entry = inputs[key]["file"]
            check_file(_path(directory, entry), entry["sha256"], f"{key} label file")
    tables = {key: _read_table(directory / files[key]["path"], files[key]["dtypes"]) if key in files else None
              for key in TABLES}
    cell_labels = _labels(directory, inputs["cells"]["file"], inputs["cells"]["labels_sha256"], "cell label file")
    territories = None
    if "territories" in files:
        territories = _labels(directory, files["territories"], record["expansion"]["expanded"]["sha256"],
                              "territory file")
    nucleus_labels = None
    if inputs.get("nuclei") is not None:
        nucleus_labels = _labels(directory, inputs["nuclei"]["file"], inputs["nuclei"]["labels_sha256"],
                                 "nucleus label file")
    return AssignmentResult(tables["molecules"], tables["cells"], tables["counts"], tables["nuclei"], cell_labels,
                            territories, nucleus_labels, tuple(inputs["molecules"]["genes"]),
                            record["cell_namespace"], record)
