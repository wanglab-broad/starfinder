"""The saved format of a segmentation run (docs/segmentation-contract.md, "Saved format", option F1).

Each run is a folder ``<checkpoint dir>/<fov_id>/segmentation/<run>/`` holding
``labels.tif`` (ZYX ``uint32``, zlib, ``ImageMetadata`` in its description),
``input.ome.tif`` (the segmentation input, ZYXC OME-TIFF; computed runs only) and
``segmentation.json`` (the run record). ``FOV.segment`` writes it and
``FOV.load_segmentation`` reads it; neither touches ``run.json`` or the ``FOV.run``
stages.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from starfinder.image import ImageMetadata

from ._labels import FORMAT_VERSION, LABEL_DTYPE, ReferenceGrid, SegmentationResult, array_sha256

LABELS_FILE = "labels.tif"
INPUT_FILE = "input.ome.tif"
RECORD_FILE = "segmentation.json"
FILES = (RECORD_FILE, LABELS_FILE, INPUT_FILE)   # removal order: the record first


def run_directory(fov_directory, name) -> Path:
    """``<fov checkpoint dir>/segmentation/<run>``."""
    return Path(fov_directory) / "segmentation" / name


def file_sha256(path) -> str:
    """SHA-256 of a file's bytes."""
    from starfinder.dataset._run_record import _sha256
    return _sha256(path)


def json_record(record):
    """The record as it reads back from its JSON file (tuples become lists, non-finite floats null)."""
    from starfinder.io._checkpoint import _jsonable
    return json.loads(json.dumps(_jsonable(record), allow_nan=False))


def check_file(path, recorded, what="file") -> None:
    """A recorded file must exist and have the recorded SHA-256; ValueError names the path and both hashes."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"{what} {path} is missing (recorded SHA-256 {recorded})")
    actual = file_sha256(path)
    if actual != recorded:
        raise ValueError(f"{what} {path} has SHA-256 {actual}, the record {recorded}")


def read_labels(path) -> tuple[np.ndarray, ImageMetadata | None]:
    """A saved ``uint32`` ZYX label file in native byte order, and its stored metadata (None without)."""
    from starfinder.io import load_volume
    loaded = load_volume(path)
    labels = loaded.image
    if labels.dtype.kind != "u" or labels.dtype.itemsize != LABEL_DTYPE.itemsize:
        raise ValueError(f"label file {path} holds {labels.dtype}, not uint32")
    labels = np.ascontiguousarray(labels, dtype=LABEL_DTYPE)
    stored = loaded.metadata if loaded.diagnostics["metadata_source"] == "stored" else None
    return labels, stored


def write_labels(path, labels, metadata) -> str:
    """Write a ``uint32`` ZYX label image (zlib, with its metadata) atomically; return the file's SHA-256."""
    from starfinder.io._checkpoint import _atomic
    from starfinder.io.tiff import save_volume
    with _atomic(path) as tmp:
        save_volume(np.ascontiguousarray(labels, dtype=LABEL_DTYPE), tmp, compress=True, metadata=metadata)
    return file_sha256(path)


def check_writable(directory, overwrite) -> None:
    """FileExistsError when the run folder holds saved files and overwrite is False."""
    existing = [name for name in FILES if (Path(directory) / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(f"segmentation run folder {directory} exists ({', '.join(existing)}); "
                              "pass overwrite=True to replace it")


def clear_run(directory) -> None:
    """Remove the files this module writes in a run folder, the record first."""
    for name in FILES:
        (Path(directory) / name).unlink(missing_ok=True)


def write_run(directory, result: SegmentationResult, segmentation_input=None) -> SegmentationResult:
    """Write one run's files; return the result with the saved record (paths and file hashes added).

    ``segmentation_input`` is the run's :class:`SegmentationInput` (None for an
    import, which has no segmentation input). The returned record is the JSON
    form written to ``segmentation.json``, so a reload compares equal.
    """
    from starfinder.io._checkpoint import write_json
    from starfinder.io.tiff import save_volume
    directory = Path(directory)
    clear_run(directory)
    directory.mkdir(parents=True, exist_ok=True)
    record = dict(result.record)
    labels = dict(record["labels"])
    labels_sha = write_labels(directory / LABELS_FILE, result.labels, result.grid.metadata)
    record["labels"] = {"path": LABELS_FILE, "dtype": labels["dtype"], "sha256": labels["sha256"],
                        "file_sha256": labels_sha, "n_labels": labels["n_labels"], "max_label": labels["max_label"]}
    if segmentation_input is not None:
        from starfinder.io._checkpoint import _atomic
        path = directory / INPUT_FILE
        with _atomic(path) as tmp:
            save_volume(np.asarray(segmentation_input.image), tmp, compress=True,
                        metadata=segmentation_input.grid.metadata)
        record["input"] = dict(record["input"], path=INPUT_FILE, file_sha256=file_sha256(path))
    record = json_record(record)
    write_json(record, directory / RECORD_FILE)
    return replace(result, record=record)


def read_run(directory, name) -> SegmentationResult:
    """Read one saved run, checking the recorded SHA-256 values; see ``FOV.load_segmentation``."""
    directory = Path(directory)
    path = directory / RECORD_FILE
    if not path.is_file():
        raise FileNotFoundError(f"no saved segmentation run {name!r} at {path}")
    record = json.loads(path.read_text())
    if record.get("format_version") != FORMAT_VERSION or record.get("stage") != "segmentation":
        raise ValueError(f"{path} is not a version {FORMAT_VERSION} segmentation record")
    if record.get("run") != name:
        raise ValueError(f"{path} records run {record.get('run')!r}, not {name!r}")
    entry = record["labels"]
    labels_path = directory / entry["path"]
    check_file(labels_path, entry["file_sha256"], "label file")
    labels, stored = read_labels(labels_path)
    if array_sha256(labels) != entry["sha256"]:
        raise ValueError(f"label file {labels_path} holds labels with SHA-256 {array_sha256(labels)}, "
                         f"the record {entry['sha256']}")
    source = record.get("input") or {}
    if source.get("path"):
        check_file(directory / source["path"], source["file_sha256"], "segmentation input")
    grid = record["grid"]
    grid = ReferenceGrid(tuple(grid["shape_zyx"]), ImageMetadata(**grid["metadata"]), grid["source"], grid["sha256"])
    if stored is not None and stored != grid.metadata:
        raise ValueError(f"label file {labels_path} stores metadata {stored!r}, the record {grid.metadata!r}")
    return SegmentationResult(labels, grid, record["target"], record["geometry"], record["label_namespace"], record)
