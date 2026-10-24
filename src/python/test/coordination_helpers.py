"""Test-only projections of structured results for existing truth comparisons, the golden digests, and the
writer of hand-built ``molecules.csv`` fixtures.

``digest`` and ``table_digest`` are the one definition of the array and table digests the
golden tests pin (W-535); they hash with ``hashlib`` here, never with package code.
"""
import hashlib
import json

import numpy as np
import pandas as pd

from starfinder.readout.spots import _join_spots


def spot_table(fov, accepted=False):
    reads = fov.filtering_result if accepted else fov.decoding_result
    return _join_spots(fov.spot_result, reads, accepted_only=accepted)


def detected_shifts(fov):
    return {name: results[0].transform.displacement_zyx
            for name, results in fov.registration_results.items()}


def digest(array):
    """SHA-256 over dtype, shape and C-order bytes."""
    array = np.ascontiguousarray(array)
    h = hashlib.sha256(f"{array.dtype.str}|{array.shape}|".encode())
    h.update(array.tobytes())
    return h.hexdigest()


def table_digest(table):
    """SHA-256 over the column names and dtypes and the CSV text (floats in %.17g)."""
    header = json.dumps([[str(c), str(t)] for c, t in table.dtypes.items()])
    text = table.to_csv(index=False, float_format="%.17g", na_rep="<NA>")
    return hashlib.sha256((header + "\n" + text).encode()).hexdigest()


def write_molecules_csv(path, spot_namespace, positions_zyx, genes, spot_ids=None):
    """A hand-built ``molecules.csv``: the typed, zero-based table ``FOV.export_spots`` writes.

    ``positions_zyx`` are zero-based (z, y, x); ``spot_ids`` default to ``m<row>`` (not a number, so never one of
    the row-position keys of the route after fusion, nor a legacy ``csv:<row>``). ``entry_id`` and
    ``qc_score`` are missing (``<NA>``). Returns the path.
    """
    from starfinder.readout.spots import MOLECULE_DTYPES, _write_spot_table
    positions = np.asarray(list(positions_zyx), np.float64).reshape(-1, 3)
    n = len(positions)
    ids = [f"m{i}" for i in range(n)] if spot_ids is None else list(spot_ids)
    frame = pd.DataFrame({"spot_namespace": [spot_namespace] * n, "spot_id": ids, "z": positions[:, 0],
                          "y": positions[:, 1], "x": positions[:, 2], "gene_id": list(genes),
                          "entry_id": [None] * n, "qc_score": np.full(n, np.nan)}).astype(MOLECULE_DTYPES)
    _write_spot_table(frame, path)
    return path
