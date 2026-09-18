"""Shared final conversion for native registration resamplers."""

import numpy as np


def _output_dtype(input_dtype, output_dtype):
    if output_dtype == "input":
        return np.dtype(input_dtype)
    if output_dtype not in ("float32", "float64"):
        raise ValueError("output_dtype must be input, float32 or float64")
    return np.dtype(output_dtype)


def _cast_warp_output(values, dtype):
    """Round once (nearest even), saturate, then cast integer outputs.

    Endpoint masks also avoid float representations of int64/uint64 maxima
    overflowing on cast. Floating interpolation still has finite precision.
    """
    dtype = np.dtype(dtype)
    if dtype.kind not in "iu":
        return values.astype(dtype)
    rounded = np.rint(values)
    limits = np.iinfo(dtype)
    low = rounded <= limits.min
    high = rounded >= limits.max
    # Keep out-of-range values out of the cast, including rounded 2**64.
    np.copyto(rounded, 0, where=low | high)
    result = rounded.astype(dtype)
    result[low] = limits.min
    result[high] = limits.max
    return result
