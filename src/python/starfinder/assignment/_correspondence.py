"""Nucleus–cell correspondence by overlap (docs/assignment-algorithms.md, "Overlap correspondence")."""
from __future__ import annotations

import numpy as np
import pandas as pd

from starfinder.image import IncompatibleGeometryError
from starfinder.segmentation import SegmentationResult, to_label_dtype

from ._config import CORRESPONDENCE_FLAGS, WITHHOLDING_FLAGS, CorrespondenceConfig


def _labels(value, name):
    array = value.labels if isinstance(value, SegmentationResult) else to_label_dtype(value)
    if array.ndim != 3:
        raise IncompatibleGeometryError(f"{name} must be a ZYX label image (a plane is 1×Y×X)")
    return array


def overlap_counts(nuclei: np.ndarray, cells: np.ndarray) -> pd.DataFrame:
    """The voxel count of each (nucleus, cell) pair over the nuclear voxels, cell 0 the background.

    Accumulated per Z plane with one uint64 key ``n × (max_cell + 1) + c``; rows are
    sorted by nucleus, then cell.
    """
    width = np.uint64(int(cells.max(initial=0)) + 1)
    keys, counts = [], []
    for z in range(nuclei.shape[0]):
        mask = nuclei[z] > 0
        if mask.any():
            key = nuclei[z][mask].astype(np.uint64) * width + cells[z][mask].astype(np.uint64)
            plane_keys, plane_counts = np.unique(key, return_counts=True)
            keys.append(plane_keys)
            counts.append(plane_counts.astype(np.int64))
    if keys:
        unique, inverse = np.unique(np.concatenate(keys), return_inverse=True)
        totals = np.zeros(len(unique), np.int64)
        np.add.at(totals, inverse, np.concatenate(counts))
    else:
        unique, totals = np.zeros(0, np.uint64), np.zeros(0, np.int64)
    return pd.DataFrame({"nucleus_id": (unique // width).astype(np.int64), "cell_id": (unique % width).astype(np.int64),
                         "count": totals})


def check_supplied(supplied, nucleus_ids, cell_ids) -> dict[int, int]:
    """A supplied correspondence table as {nucleus_id: cell_id}, after its checks.

    Raises
    ------
    TypeError
        supplied is not a DataFrame.
    ValueError
        Missing columns, values that are not nonnegative integers, a nucleus that
        appears twice, or a value absent from its image (naming the missing values).
    """
    if not isinstance(supplied, pd.DataFrame):
        raise TypeError("a supplied correspondence must be a DataFrame with the columns nucleus_id and cell_id")
    if not {"nucleus_id", "cell_id"} <= set(supplied.columns):
        raise ValueError("a supplied correspondence needs the columns nucleus_id and cell_id")
    pairs = {}
    for column in ("nucleus_id", "cell_id"):
        values = supplied[column]
        if values.isna().any() or values.dtype.kind not in "iu" or (values < 0).any():
            raise ValueError(f"supplied {column} must be nonnegative integers")
        pairs[column] = values.to_numpy(np.int64)
    nuclei, cells = pairs["nucleus_id"], pairs["cell_id"]
    repeated = sorted({int(n) for n in nuclei[pd.Series(nuclei).duplicated().to_numpy()]})
    if repeated:
        raise ValueError(f"a nucleus appears more than once in the supplied correspondence: {repeated}")
    for name, values, known in (("nucleus", nuclei, nucleus_ids), ("cell", cells, cell_ids)):
        missing = sorted({int(v) for v in values} - {int(v) for v in known})
        if missing:
            raise ValueError(f"supplied {name} values absent from the {name} image: {missing}")
    return {int(n): int(c) for n, c in zip(nuclei, cells)}


def match_nuclei(nuclei, cells, *, config: CorrespondenceConfig = CorrespondenceConfig(),
                 supplied: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Relate the nuclei to the cells by overlap, or by a supplied table, and flag every doubtful case.

    For nucleus ``n`` with ``s_n`` voxels, ``f(n, c) = o(n, c) / s_n`` is the share
    inside cell ``c`` (the territories as given, so after any expansion). ``n`` is
    ``matched`` to the cell ``c*`` with the largest share (ties to the smaller
    ``cell_id``) when ``f(n, c*) > match_fraction``; else ``ambiguous`` when it
    overlaps a cell; else ``no_cell``. A matched nucleus whose share outside its
    cell exceeds ``outside_tolerance`` is ``outside``. A ``supplied`` table
    (``nucleus_id``, ``cell_id``) replaces the majority rule only: a nucleus in it is
    matched to its row's cell, one absent from it is ``ambiguous`` or ``no_cell`` by
    overlap, and shares and flags follow from overlap as above. Nothing is repaired.

    Parameters
    ----------
    nuclei, cells : numpy.ndarray or SegmentationResult
        The nucleus labels and the territories, ZYX on one grid.
    config : CorrespondenceConfig
        The thresholds.
    supplied : pandas.DataFrame, optional
        A correspondence to use instead of the majority rule.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        The nucleus table (``nucleus_id``, ``size_voxels``, ``status``,
        ``cell_id``, ``share_in_cell``, ``share_background``,
        ``n_cells_overlapped``, ``outside``, ``seed_value_agrees`` null), one row
        per nucleus in increasing ``nucleus_id``; and per cell, in increasing
        ``cell_id``: ``n_nuclei``, ``correspondence``, ``correspondence_flags``
        (``;``-joined) and ``compartments``.

    Raises
    ------
    IncompatibleGeometryError
        The two images differ in shape or are not ZYX.
    TypeError, ValueError
        A config of the wrong type, or the errors of a supplied table.
    """
    if not isinstance(config, CorrespondenceConfig):
        raise TypeError("config must be a CorrespondenceConfig")
    nucleus_labels, cell_labels = _labels(nuclei, "nuclei"), _labels(cells, "cells")
    if nucleus_labels.shape != cell_labels.shape:
        raise IncompatibleGeometryError(f"nuclei {nucleus_labels.shape} and cells {cell_labels.shape} are on "
                                        "different grids")
    cell_ids = np.unique(cell_labels)
    cell_ids = cell_ids[cell_ids > 0].astype(np.int64)
    overlaps = overlap_counts(nucleus_labels, cell_labels)
    nucleus_ids = np.unique(overlaps.nucleus_id.to_numpy())
    chosen = None if supplied is None else check_supplied(supplied, nucleus_ids, cell_ids)

    rows, matched_to, ambiguous, outside = [], {}, set(), set()
    for n, group in overlaps.groupby("nucleus_id", sort=True):
        n = int(n)
        size = int(group["count"].sum())
        inside = group[group.cell_id > 0].sort_values(["count", "cell_id"], ascending=[False, True])
        background = int(group.loc[group.cell_id == 0, "count"].sum())
        share = dict(zip(inside.cell_id.astype(int), inside["count"].astype(int)))
        best = int(inside.cell_id.iloc[0]) if len(inside) else None
        if chosen is None:
            cell = best if best is not None and share[best] / size > config.match_fraction else None
        else:
            cell = chosen.get(n)
        if cell is not None:
            status, overlap = "matched", share.get(cell, 0)
            is_outside = (size - overlap) / size > config.outside_tolerance
            matched_to[n] = cell
            if is_outside:
                outside.add(n)
        else:
            status = "ambiguous" if best is not None else "no_cell"
            overlap, is_outside = (share[best] if best is not None else 0), False
            if best is not None:
                ambiguous.add(n)
        rows.append((n, size, status, cell, overlap / size, background / size, len(inside), is_outside))
    columns = list(zip(*rows)) if rows else [()] * 8
    nucleus_table = pd.DataFrame({
        "nucleus_id": pd.array(list(columns[0]), dtype="UInt32"),
        "size_voxels": np.array(columns[1], np.int64),
        "status": pd.array(list(columns[2]), dtype="string"),
        "cell_id": pd.array(list(columns[3]), dtype="UInt32"),
        "share_in_cell": np.array(columns[4], np.float64),
        "share_background": np.array(columns[5], np.float64),
        "n_cells_overlapped": np.array(columns[6], np.int64),
        "outside": np.array(columns[7], bool),
        "seed_value_agrees": pd.array([None] * len(rows), dtype="boolean"),
    })

    own, over = {}, {}
    for n, c in matched_to.items():
        own.setdefault(c, []).append(n)
    touching = overlaps[overlaps.cell_id > 0]
    for n, c in zip(touching.nucleus_id.astype(int), touching.cell_id.astype(int)):
        over.setdefault(c, set()).add(n)
    per_cell = {name: [] for name in ("n_nuclei", "correspondence", "correspondence_flags", "compartments")}
    for c in cell_ids.tolist():
        mine, near = own.get(c, []), over.get(c, set())
        flags = {"several_nuclei": len(mine) >= 2,
                 "ambiguous_nucleus": bool(near & ambiguous),
                 "nucleus_outside_cell": any(n in outside for n in mine),
                 "foreign_nucleus": any(n in matched_to and matched_to[n] != c for n in near)}
        if mine:
            correspondence = "matched"
        elif flags["ambiguous_nucleus"]:
            correspondence = "ambiguous"
        else:
            correspondence = "no_nucleus"
        if correspondence == "no_nucleus":
            compartments = "no_nucleus"
        elif any(flags[f] for f in WITHHOLDING_FLAGS):
            compartments = "withheld"
        else:
            compartments = "available"
        per_cell["n_nuclei"].append(len(mine))
        per_cell["correspondence"].append(correspondence)
        per_cell["correspondence_flags"].append(";".join(f for f in CORRESPONDENCE_FLAGS if flags[f]))
        per_cell["compartments"].append(compartments)
    cell_table = pd.DataFrame({
        "cell_id": pd.array(cell_ids.tolist(), dtype="UInt32"),
        "n_nuclei": pd.array(per_cell["n_nuclei"], dtype="Int64"),
        "correspondence": pd.array(per_cell["correspondence"], dtype="string"),
        "correspondence_flags": pd.array(per_cell["correspondence_flags"], dtype="string"),
        "compartments": pd.array(per_cell["compartments"], dtype="string"),
    })
    return nucleus_table, cell_table
