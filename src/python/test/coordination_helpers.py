"""Test-only projections of structured results for existing truth comparisons, and the golden digests.

``digest`` and ``table_digest`` are the one definition of the array and table digests the
golden tests pin (W-535); they hash with ``hashlib`` here, never with package code.
"""
import hashlib
import json

import numpy as np

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
