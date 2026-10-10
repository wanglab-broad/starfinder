"""The sample-level AnnData, ``sample.h5ad`` (docs/assembly-contract.md, "Sample-level AnnData")."""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from starfinder.assignment._anndata import _anndata, write_h5ad

from ._assemble import AssemblyResult

# The §2.9 cell columns raw.h5ad carries, copied into obs.
CELL_OBS = ("size_voxels", "expanded_size_voxels", "size_physical", "centroid_z", "centroid_y", "centroid_x",
            "n_molecules", "n_nuclei", "correspondence", "correspondence_flags", "compartments")
COORDINATE_OBS = tuple(f"sample_{name}_{a}" for name in ("centroid", "expanded_centroid") for a in "zyx")
OWNERSHIP_OBS = tuple(f"sample_ownership_centroid_{a}" for a in "zyx")
SPATIAL_COLUMNS = ("sample_ownership_centroid_x", "sample_ownership_centroid_y")


def _column(values):
    """An obs column as anndata stores it: object strings (Categorical where one is missing), numbers as they are."""
    if isinstance(values.dtype, pd.StringDtype) or values.dtype == object:
        if values.isna().any():
            return pd.Categorical(np.array([None if pd.isna(v) else str(v) for v in values], dtype=object))
        return np.array([str(v) for v in values], dtype=object)
    if isinstance(values.dtype, pd.api.extensions.ExtensionDtype):
        return values.array
    return values.to_numpy()


def sample_anndata(result: AssemblyResult, path, *, annotation: Mapping | pd.Series | None = None) -> Path:
    """Write ``sample.h5ad`` of an assembled sample from its sample tables; return the path.

    One row per counted cell (``assembly_status == "retained"`` and §2.9
    ``status == "kept"``), in the cell table's order, named by ``obs_name``.
    ``obs`` holds ``sample``, ``fov_id``, ``cell_namespace``, ``cell_id``, the
    §2.9 cell columns ``raw.h5ad`` carries, every sample-coordinate column,
    ``ownership_centroid``, ``assembly_flags``, ``conflict_with`` and, with
    ``annotation``, the columns of the sample's annotation row. ``X`` is the
    float64 whole-cell counts in ``genes`` order; when some FOV gave nuclei (on
    the route after fusion: when the sample segmentation's nucleus run entered
    the sample assignment, as its record ``assignment`` says), the
    ``nucleus`` and ``cytoplasm`` layers (NaN rows where ``compartments`` is
    not ``available``). ``obsm["spatial"]`` is ``[sample_ownership_centroid_x,
    sample_ownership_centroid_y]`` (µm); ``uns`` holds the record as JSON
    (``assembly``), ``spatial_unit`` ``"µm"``, ``frame_id`` and
    ``spatial_centroid`` (JSON: the two columns, the ``centroid`` setting and
    the rows per ``ownership_centroid`` value). In ``molecules_only`` mode it
    has no row and ``var["n_molecules_retained"]`` holds the retained molecules
    per gene. Built and written by :func:`starfinder.assignment.write_h5ad`.

    Parameters
    ----------
    result : AssemblyResult
        The assembled sample.
    path : str or Path
        The file to write; its folder is created.
    annotation : Mapping or pandas.Series, optional
        Extra ``obs`` columns for the sample, one value each (the workflow
        passes the sample's annotation row).

    Returns
    -------
    Path

    Raises
    ------
    ModuleNotFoundError
        anndata is not installed (the ``anndata`` extra).
    TypeError
        result is not an AssemblyResult or annotation not a mapping.
    ValueError
        An annotation column that is already an ``obs`` column.
    """
    _anndata("sample_anndata")
    if not isinstance(result, AssemblyResult):
        raise TypeError("result must be an AssemblyResult")
    if annotation is not None and not isinstance(annotation, (Mapping, pd.Series)):
        raise TypeError("annotation must be a mapping of column to value, or a pandas Series")
    rows = result.counted_cells()
    n = len(rows)
    with pd.option_context("future.infer_string", False):
        columns = {"sample": np.array([result.sample] * n, dtype=object)}
        for name in ("fov_id", "cell_namespace"):
            columns[name] = _column(rows[name])
        columns["cell_id"] = rows["cell_id"].to_numpy(np.int64)
        for name in (*CELL_OBS, *COORDINATE_OBS, "ownership_centroid", *OWNERSHIP_OBS, "assembly_flags",
                     "conflict_with"):
            columns[name] = _column(rows[name])
        for name, value in ({} if annotation is None else dict(annotation)).items():
            name = str(name)
            if name in columns:
                raise ValueError(f"annotation column {name!r} is already an obs column")
            columns[name] = np.array([None if pd.isna(value) else str(value)] * n, dtype=object) \
                if isinstance(value, str) or value is None else np.full(n, value)
        obs = pd.DataFrame(columns, index=pd.Index([str(v) for v in rows["obs_name"]], dtype=object))
    keys = rows[["cell_namespace", "cell_id"]]
    if result.record.get("route") == "after_fusion" and result.mode == "cells":
        # The route after fusion: from the sample segmentation, through the sample assignment's record.
        layers = ((result.record.get("assignment") or {}).get("inputs") or {}).get("nuclei") is not None
    else:
        layers = any(entry.get("nuclei") for entry in result.record.get("fovs", []))
    var = None
    if result.mode == "molecules_only":
        retained = result.molecules[result.molecules["assembly_status"].eq("retained").to_numpy(dtype=bool)]
        per_gene = retained["gene_id"].astype(object).value_counts()
        var = pd.DataFrame({"n_molecules_retained": np.array([int(per_gene.get(g, 0)) for g in result.genes],
                                                             dtype=np.int64)})
    expanded = int(rows["ownership_centroid"].eq("expanded").sum())
    spatial = {"columns": list(SPATIAL_COLUMNS), "centroid": result.record.get("config", {}).get("centroid"),
               "rows": {"expanded": expanded, "original": n - expanded}}
    from starfinder.io._checkpoint import _jsonable
    uns = {"assembly": json.dumps(_jsonable(result.record), default=str), "spatial_unit": "µm",
           "frame_id": (result.record.get("frame") or {}).get("frame_id", f"sample:{result.sample}"),
           "spatial_centroid": json.dumps(spatial)}
    obsm = {"spatial": rows[list(SPATIAL_COLUMNS)].to_numpy(np.float64).reshape(n, 2)}
    return write_h5ad(path, obs, keys, result.counts, result.genes, layers=layers, uns=uns, var=var, obsm=obsm)
