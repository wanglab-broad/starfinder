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

from scipy.ndimage import map_coordinates

def apply_tps_deformation(
    volume: np.ndarray,
    displacement_field: np.ndarray,
    boundary_mode: str = "constant",
    *,
    output_dtype: str = "input",
    fill_value: float = 0,
) -> np.ndarray:
    """Warp a volume using a displacement field, slice-by-slice.

    Processes coordinates and floating samples one Z-slice at a time rather
    than allocating a full-volume coordinate grid or floating input copy.

    Parameters
    ----------
    volume : np.ndarray
        Input volume, shape (Z, Y, X).
    displacement_field : np.ndarray
        Displacement field, shape (Z, Y, X, 3) with (dz, dy, dx).
    boundary_mode : str
        How to handle out-of-bounds source coordinates:

        - ``"constant"`` (default): Fill with 0 (black bands at edges).

        - ``"nearest"``: Extend edge pixels (no black bands).

    output_dtype : str, optional
        "input" (default), "float32" or "float64". Interpolate in floating
        point, then round nearest-even, clip and cast once for integer output.

    Returns
    -------
    np.ndarray
        Warped volume, same shape as input; source dtype by default.
    """
    nz, ny, nx = volume.shape
    dtype = _output_dtype(volume.dtype, output_dtype)
    result = np.empty_like(volume, dtype=dtype)

    # Pre-compute YX coordinate grids (reused across slices)
    yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float32)

    for z in range(nz):
        dz = displacement_field[z, :, :, 0]
        dy = displacement_field[z, :, :, 1]
        dx = displacement_field[z, :, :, 2]

        # Source coordinates = current position + displacement
        src_z = np.full_like(yy, z, dtype=np.float32) + dz
        src_y = yy + dy
        src_x = xx + dx

        coords = np.array([src_z, src_y, src_x])
        # Request floating interpolation directly: SciPy otherwise rounds
        # into the source integer dtype before our final conversion.
        sampled = map_coordinates(
            volume, coords, order=1, mode=boundary_mode, cval=fill_value, output=np.float64
        )
        result[z] = _cast_warp_output(sampled, dtype)

    return result
