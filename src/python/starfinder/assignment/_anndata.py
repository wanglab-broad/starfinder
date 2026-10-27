"""The one AnnData writer of the per-FOV assignment and the sample assembly: ``write_h5ad``
(docs/assembly-contract.md, "Sample-level AnnData")."""
# §2.9, §2.10
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

LAYERS = ("nucleus", "cytoplasm")


def _anndata(caller):
    """Import anndata lazily; without it, ModuleNotFoundError names the extra."""
    try:
        import anndata
    except ImportError:
        raise ModuleNotFoundError(f"{caller} needs anndata; install the 'anndata' extra "
                                  "(pip install 'starfinder[anndata]')", name="anndata") from None
    return anndata


def _plain_strings(frame):
    """A copy whose pandas string columns are object strings (Categorical where a value is missing).

    anndata 0.12 writes no pandas Arrow string arrays, which pandas 3 infers;
    object and Categorical columns are kept as they are.
    """
    frame = frame.copy()
    for column in frame.columns:
        values = frame[column]
        if isinstance(values.dtype, pd.StringDtype):
            if values.isna().any():
                frame[column] = pd.Categorical(np.array([None if pd.isna(v) else str(v) for v in values],
                                                        dtype=object))
            else:
                frame[column] = np.array([str(v) for v in values], dtype=object)
    frame.index = pd.Index([str(i) for i in frame.index], dtype=object)
    return frame


def write_h5ad(path, obs: pd.DataFrame, keys: pd.DataFrame, counts: pd.DataFrame, genes: Sequence[str], *,
               layers: bool, uns: Mapping[str, str], var: pd.DataFrame | None = None,
               obsm: Mapping[str, np.ndarray] | None = None) -> Path:
    """Build an AnnData from long counts and write it at ``path``; return the path.

    ``obs`` holds one row per cell to write, in order, with the ``obs_names`` as
    its index; ``keys`` the same rows' key columns, which ``counts`` also has
    (``cell_id`` within one FOV; ``cell_namespace`` and ``cell_id`` for a
    sample). ``X`` is the float64 ``whole`` counts in ``genes`` order, 0 where a
    cell has no row. With ``layers=True`` the ``nucleus`` and ``cytoplasm``
    layers are float64, 0.0 for a measured zero and NaN for every gene of a row
    whose ``obs["compartments"]`` is not ``available``. ``var_names`` are
    ``genes``, with the columns of ``var`` when given; ``uns`` and ``obsm`` are
    stored as given. Pandas string columns of ``obs`` and ``var`` are written as
    object strings (Categorical where a value is missing), as anndata expects.
    With no row, ``X`` (and each layer) has shape ``(0, len(genes))``. The folder
    of ``path`` is created.

    Parameters
    ----------
    path : str or Path
        The ``.h5ad`` file.
    obs : pandas.DataFrame
        One row per cell, indexed by the obs names.
    keys : pandas.DataFrame
        The rows' key columns, row for row with ``obs``.
    counts : pandas.DataFrame
        Long counts with the key columns, ``gene_id``, ``compartment`` and
        ``count`` (docs/assignment-contract.md).
    genes : Sequence[str]
        The gene order of ``X``.
    layers : bool
        Whether to write the ``nucleus`` and ``cytoplasm`` layers.
    uns : Mapping[str, str]
        Stored as ``uns``.
    var : pandas.DataFrame, optional
        Columns of ``var``, one row per gene in ``genes`` order.
    obsm : Mapping[str, numpy.ndarray], optional
        Stored as ``obsm``.

    Returns
    -------
    Path

    Raises
    ------
    ModuleNotFoundError
        anndata is not installed (the ``anndata`` extra).
    ValueError
        ``keys`` has another number of rows than ``obs``, a key column is
        missing from ``counts``, ``layers=True`` without an ``obs``
        ``compartments`` column, or ``var`` has another number of rows than
        ``genes``.
    """
    anndata = _anndata("write_h5ad")
    genes = [str(g) for g in genes]
    if len(keys) != len(obs):
        raise ValueError(f"keys has {len(keys)} rows and obs {len(obs)}; give one key row per obs row")
    missing = [c for c in keys.columns if c not in counts.columns]
    if missing:
        raise ValueError(f"counts lacks the key columns {missing}")
    if layers and "compartments" not in obs.columns:
        raise ValueError("layers=True needs the obs column 'compartments'")
    if var is not None and len(var) != len(genes):
        raise ValueError(f"var has {len(var)} rows for {len(genes)} genes")
    key_columns = list(keys.columns)
    row_index = keys[key_columns].reset_index(drop=True).assign(_row=np.arange(len(keys), dtype=np.int64))
    gene_of = {g: j for j, g in enumerate(genes)}

    def dense(compartment):
        matrix = np.zeros((len(obs), len(genes)), np.float64)
        rows = counts[counts["compartment"].astype(object).eq(compartment).to_numpy(dtype=bool)]
        rows = rows[key_columns + ["gene_id", "count"]].merge(row_index, on=key_columns, how="inner")
        if len(rows):
            j = rows["gene_id"].astype(object).map(gene_of)
            if j.isna().any():
                raise ValueError(f"counts hold genes outside genes: {sorted(set(rows.gene_id[j.isna()]))}")
            np.add.at(matrix, (rows["_row"].to_numpy(np.int64), j.to_numpy(np.int64)),
                      rows["count"].to_numpy(np.float64))
        return matrix

    written_layers = {}
    if layers:
        unavailable = ~obs["compartments"].astype(object).eq("available").to_numpy(dtype=bool)
        for name in LAYERS:
            layer = dense(name)
            layer[unavailable, :] = np.nan
            written_layers[name] = layer
    with pd.option_context("future.infer_string", False):
        var_frame = (pd.DataFrame(index=pd.Index(genes, dtype=object)) if var is None else
                     _plain_strings(var.set_axis(pd.Index(genes, dtype=object), axis=0)))
        var_frame.index = pd.Index(genes, dtype=object)
        adata = anndata.AnnData(X=dense("whole"), obs=_plain_strings(obs), var=var_frame, layers=written_layers)
        for key, value in (obsm or {}).items():
            adata.obsm[key] = np.asarray(value)
        for key, value in uns.items():
            adata.uns[key] = value
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        adata.write(path)
    return path
