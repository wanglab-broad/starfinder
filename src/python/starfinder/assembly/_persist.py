"""The assembly in the sample's export folder, option S1 (docs/assembly-contract.md, "Persistence").

``<OUTPUT>/export/<sample>/`` holds the ``assembly`` section of ``export.json`` (the
record, with the path and SHA-256 of every file), ``tables/fovs.<fmt>``,
``tables/cells.<fmt>``, ``tables/molecules.<fmt>``, ``tables/counts.<fmt>`` and
``sample.h5ad``. Every file is written to a temporary sibling and renamed into place, so
an interrupted save leaves each file complete or absent; the other sections of
``export.json`` are kept as they are.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from ._anndata import sample_anndata
from ._assemble import AssemblyResult, fill_cell_namespace
from ._fused import TERRITORIES_FILE, territories_of, with_territories

EXPORT_FILE = "export.json"
SECTION = "assembly"
TABLES = ("fovs", "cells", "molecules", "counts")
TABLE_DIR = "tables"
ANNDATA_FILE = "sample.h5ad"


def _write_table(frame, directory, name, table_format):
    """Write one sample table atomically through the checkpoint table writer; return (file name, dtypes)."""
    from starfinder.io._checkpoint import _write_table as write
    return write(frame, directory, name, table_format)


def _write_anndata(result, path):
    """Write sample.h5ad atomically."""
    from starfinder.io._checkpoint import _atomic
    with _atomic(path) as tmp:
        sample_anndata(result, tmp)


def save_assembly(result: AssemblyResult, folder, *, table_format: str = "csv") -> AssemblyResult:
    """Write an assembled sample into its export folder; return the result with the saved record.

    Writes ``tables/fovs``, ``tables/cells``, ``tables/molecules`` and
    ``tables/counts`` in ``table_format`` (``csv``, float64 as ``%.17g``, or
    ``parquet``) through the checkpoint table writer, and ``sample.h5ad``
    (:func:`sample_anndata`), each atomically; a table of the other format
    left by an earlier save is removed. Then it replaces only the ``assembly``
    section of ``folder/export.json`` (created when absent) with the record,
    whose ``files`` lists each file's path, SHA-256 and, for a table, its
    dtypes. ``sample.h5ad`` holds the record without its ``files``. A result
    of the route after fusion whose assignment expanded the cells also writes
    the territories as ``territories.ome.tif`` (uint32, values ``cell_id``, on
    the segmentation grid), listed in ``files`` and in the record's
    ``segmentation_files``.

    Parameters
    ----------
    result : AssemblyResult
        The assembled sample.
    folder : str or Path
        The sample's export folder, by convention ``<OUTPUT>/export/<sample>/``.
    table_format : str
        ``csv`` (default) or ``parquet``.

    Returns
    -------
    AssemblyResult
        The same tables with the record as saved.

    Raises
    ------
    TypeError
        result is not an AssemblyResult.
    ValueError
        An unknown table format, or an ``export.json`` that is not a JSON
        object.
    ModuleNotFoundError
        anndata is not installed (the ``anndata`` extra).
    """
    from starfinder.io._checkpoint import TABLE_FORMATS, write_json
    from starfinder.segmentation._persist import file_sha256, json_record
    if not isinstance(result, AssemblyResult):
        raise TypeError("result must be an AssemblyResult")
    if table_format not in TABLE_FORMATS:
        raise ValueError(f"table_format must be one of {list(TABLE_FORMATS)}; got {table_format!r}")
    folder = Path(folder)
    export = folder / EXPORT_FILE
    sections = {}
    if export.exists():
        sections = json.loads(export.read_text())
        if not isinstance(sections, dict):
            raise ValueError(f"{export} is not a JSON object")
    tables = folder / TABLE_DIR
    files = {}
    territories = territories_of(result)
    if territories is not None:
        # The route after fusion with an expansion: the territories assign sampled, on the segmentation grid.
        from starfinder.sample_output._tiff import write_ome_tiff
        labels, metadata = territories
        folder.mkdir(parents=True, exist_ok=True)
        write_ome_tiff(folder / TERRITORIES_FILE, labels[0], metadata=metadata,
                       annotation={"sample": result.sample, "image": TERRITORIES_FILE[:-len(".ome.tif")],
                                   "values": "cell_id"})
        digest = file_sha256(folder / TERRITORIES_FILE)
        files["territories"] = {"path": TERRITORIES_FILE, "sha256": digest}
        entries = [e for e in result.record.get("segmentation_files") or [] if e.get("run") is not None]
        result = with_territories(replace(result, record=dict(
            result.record, segmentation_files=entries + [{"run": None, "path": TERRITORIES_FILE, "sha256": digest}])),
            territories)
    for name in TABLES:
        file_name, dtypes = _write_table(getattr(result, name), tables, name, table_format)
        files[name] = {"path": f"{TABLE_DIR}/{file_name}", "sha256": file_sha256(tables / file_name),
                       "dtypes": dtypes}
        for other in TABLE_FORMATS:
            if other != table_format:
                (tables / f"{name}.{other}").unlink(missing_ok=True)
    record = json_record(dict(result.record, files={}, table_format=table_format))
    _write_anndata(replace(result, record=record), folder / ANNDATA_FILE)
    files["anndata"] = {"path": ANNDATA_FILE, "sha256": file_sha256(folder / ANNDATA_FILE)}
    record = json_record(dict(record, files=files))
    sections[SECTION] = record
    write_json(sections, export)
    return with_territories(replace(result, record=record), territories)


def load_assembly(folder) -> AssemblyResult:
    """Read the assembly of an export folder, checking every recorded SHA-256.

    Every file of the section's ``files`` and every territory label file a
    FOV's entry records beside its SHA-256 (``conflicts=True`` with a saved
    assignment folder; the inputs of the assembly, read again by the label
    export) must exist with that SHA-256. A molecule table saved without the
    column ``cell_namespace`` (before it existed) gets it from the cell table:
    a molecule with a ``cell_id`` takes the namespace of the cell row of its
    ``fov_id`` and ``cell_id``, null where no single row holds them. The
    ``territories.ome.tif`` of the route after fusion is read with the result,
    so that it can be saved again.

    Raises
    ------
    FileNotFoundError
        No ``export.json``, or one without an ``assembly`` section.
    ValueError
        A record of another format version or stage, or a file that is
        missing or whose SHA-256 differs from the record (naming the path and
        both hashes).
    """
    from starfinder.io._checkpoint import _read_table
    from starfinder.segmentation._persist import check_file
    folder = Path(folder)
    export = folder / EXPORT_FILE
    if not export.is_file():
        raise FileNotFoundError(f"no export record at {export}")
    record = json.loads(export.read_text()).get(SECTION)
    if not isinstance(record, dict):
        raise FileNotFoundError(f"{export} has no {SECTION!r} section")
    if record.get("format_version") != 1 or record.get("stage") != SECTION:
        raise ValueError(f"{export} does not hold a version 1 assembly record")
    files = record.get("files") or {}
    for key, entry in files.items():
        check_file(folder / entry["path"], entry["sha256"], f"assembly file {key}")
    for entry in record.get("fovs") or []:
        territories = entry.get("territories") or {}
        if territories.get("path"):
            check_file(territories["path"], territories["sha256"], f"territory file (FOV {entry.get('fov_id')})")
    frames = {name: _read_table(folder / files[name]["path"], files[name]["dtypes"]) for name in TABLES}
    # A molecule table saved before the column cell_namespace existed gets it from the cell table.
    frames["molecules"] = fill_cell_namespace(frames["molecules"], frames["cells"])
    result = AssemblyResult(record["sample"], frames["fovs"], frames["cells"], frames["molecules"], frames["counts"],
                            tuple(record["genes"]), record)
    if "territories" in files:
        # The territories of the route after fusion, kept with the result so that it can be saved again.
        import tifffile

        from starfinder.image import ImageMetadata
        from starfinder.io.tiff import METADATA_NAMESPACE
        from starfinder.sample_output._tiff import read_ome_layout
        path = folder / files["territories"]["path"]
        labels = np.ascontiguousarray(tifffile.imread(path), dtype=np.uint32)
        stored = read_ome_layout(path)["annotations"][METADATA_NAMESPACE]
        metadata = ImageMetadata(stored["frame_id"], tuple(stored["spacing_zyx"]), tuple(stored["origin_zyx"]),
                                 tuple(tuple(row) for row in stored["direction_zyx"]), stored["spatial_unit"])
        result = with_territories(result, (labels[np.newaxis] if labels.ndim == 2 else labels, metadata))
    return result
