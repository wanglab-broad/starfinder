"""Fused numba kernels of the classic demons iteration (_demons._thirion_iterations).

One pass over the voxels samples the moving image, computes the Thirion update,
adds it to the field and accumulates the RMS sum, in place, with no temporary
array. The kernels compile on import (with numba's cache when a cache location
is writable: NUMBA_CACHE_DIR when it is set) and run on one thread, without
fastmath, so each voxel's arithmetic is that of the NumPy and SciPy loop they
replace (W-882). This module imports numba; _demons imports it lazily.
"""

from __future__ import annotations

import math

import numba
from numba import types

# Inputs are read-only C-contiguous float64 arrays (a writable array converts to one); the field is writable.
_IMAGE_3D = types.Array(types.float64, 3, "C", readonly=True)
_GRADIENT_3D = types.Array(types.float64, 4, "C", readonly=True)
_FIELD_3D = types.float64[:, :, :, ::1]
_IMAGE_2D = types.Array(types.float64, 2, "C", readonly=True)
_GRADIENT_2D = types.Array(types.float64, 3, "C", readonly=True)
_FIELD_2D = types.float64[:, :, ::1]
_RESULT = types.Tuple((types.int64, types.float64))


def _jit(signature, inline=False):
    """numba.njit with numba's cache, or without it where no cache location is writable.

    numba raises RuntimeError when it decorates a cached function and finds no
    writable cache location (a read-only install with NUMBA_CACHE_DIR unset);
    the kernel then compiles in each process instead.
    """
    options = dict(error_model="numpy", inline="always" if inline else "never")

    def decorate(function):
        try:
            return numba.njit(signature, cache=True, **options)(function)
        except RuntimeError:
            return numba.njit(signature, **options)(function)

    return decorate


@_jit(types.UniTuple(types.int64, 2)(types.float64, types.int64), inline=True)
def _taps(coordinate, size):
    """The two voxel indices of linear interpolation at coordinate on an axis of size voxels, clamped to the grid.

    As scipy.ndimage.map_coordinates(order=1, mode="nearest"): the taps are
    floor(coordinate) and the next voxel, each moved to the nearest voxel of the
    grid, and the weights stay those of the unclamped coordinate.
    """
    first = int(math.floor(coordinate))
    second = first + 1
    first = min(max(first, 0), size - 1)
    second = min(max(second, 0), size - 1)
    return first, second


@_jit(_RESULT(_IMAGE_3D, _IMAGE_3D, _GRADIENT_3D, _IMAGE_3D, _FIELD_3D))
def thirion_update_3d(fixed, moving, gradient, gradient_sq, field):
    """Add one Thirion update to a ZYX field in place; return the inside count and the sum of squared updates.

    Parameters
    ----------
    fixed : numpy.ndarray
        Fixed image, float64, C-contiguous, shape (Z, Y, X).
    moving : numpy.ndarray
        Moving image, float64, C-contiguous, on the fixed image's grid.
    gradient : numpy.ndarray
        The fixed image's gradient (_demons._face_gradient), float64,
        C-contiguous, shape (3, Z, Y, X), components ZYX.
    gradient_sq : numpy.ndarray
        Its squared norm, (gz * gz + gy * gy) + gx * gx, float64, C-contiguous,
        shape (Z, Y, X).
    field : numpy.ndarray
        Pulling field in voxels, float64, C-contiguous, shape (3, Z, Y, X),
        components ZYX; updated in place.

    Returns
    -------
    count : int
        Number of voxels whose sample p + field(p) lies inside the moving grid,
        [-0.5, N - 0.5) on every axis.
    total : float
        Sum over those voxels of |update|^2, scale^2 * gradient_sq, by
        compensated (Neumaier) summation.

    Notes
    -----
    Per voxel p, as the NumPy loop: the sample at p + field(p) by
    linear interpolation, with map_coordinates' taps, products
    ((v * wz) * wy) * wx and corner order (X fastest); diff = F - M;
    denominator = gradient_sq + diff * diff; scale = diff / denominator, or 0
    where p is outside, |diff| < 0.001 or the denominator < 1e-9; then
    field(p) += scale * gradient(p) on every axis. Only the sum of the RMS
    differs from NumPy's pairwise summation, in its rounding.
    """
    nz, ny, nx = fixed.shape
    count = 0
    total = 0.0
    compensation = 0.0
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                cz = field[0, z, y, x] + z
                cy = field[1, z, y, x] + y
                cx = field[2, z, y, x] + x
                scale = 0.0
                inside = (cz >= -0.5) and (cz < nz - 0.5) and (cy >= -0.5) and (cy < ny - 0.5) \
                    and (cx >= -0.5) and (cx < nx - 0.5)
                if inside:
                    z0, z1 = _taps(cz, nz)
                    y0, y1 = _taps(cy, ny)
                    x0, x1 = _taps(cx, nx)
                    tz = cz - math.floor(cz)
                    ty = cy - math.floor(cy)
                    tx = cx - math.floor(cx)
                    az = 1.0 - tz
                    ay = 1.0 - ty
                    ax = 1.0 - tx
                    bz = 1.0 - az
                    by = 1.0 - ay
                    bx = 1.0 - ax
                    warped = 0.0
                    warped += moving[z0, y0, x0] * az * ay * ax
                    warped += moving[z0, y0, x1] * az * ay * bx
                    warped += moving[z0, y1, x0] * az * by * ax
                    warped += moving[z0, y1, x1] * az * by * bx
                    warped += moving[z1, y0, x0] * bz * ay * ax
                    warped += moving[z1, y0, x1] * bz * ay * bx
                    warped += moving[z1, y1, x0] * bz * by * ax
                    warped += moving[z1, y1, x1] * bz * by * bx
                    diff = fixed[z, y, x] - warped
                    denominator = gradient_sq[z, y, x] + diff * diff
                    if abs(diff) >= 0.001 and denominator >= 1e-9:
                        scale = diff / denominator
                field[0, z, y, x] += scale * gradient[0, z, y, x]
                field[1, z, y, x] += scale * gradient[1, z, y, x]
                field[2, z, y, x] += scale * gradient[2, z, y, x]
                if inside:
                    count += 1
                    term = scale * scale * gradient_sq[z, y, x]
                    running = total + term
                    if abs(total) >= abs(term):
                        compensation += (total - running) + term
                    else:
                        compensation += (term - running) + total
                    total = running
    return count, total + compensation


@_jit(_RESULT(_IMAGE_2D, _IMAGE_2D, _GRADIENT_2D, _IMAGE_2D, _FIELD_2D))
def thirion_update_2d(fixed, moving, gradient, gradient_sq, field):
    """Add one Thirion update to a YX field in place; return the inside count and the sum of squared updates.

    Parameters
    ----------
    fixed : numpy.ndarray
        Fixed image, float64, C-contiguous, shape (Y, X).
    moving : numpy.ndarray
        Moving image, float64, C-contiguous, on the fixed image's grid.
    gradient : numpy.ndarray
        The fixed image's gradient (_demons._face_gradient), float64,
        C-contiguous, shape (2, Y, X), components YX.
    gradient_sq : numpy.ndarray
        Its squared norm, gy * gy + gx * gx, float64, C-contiguous, shape (Y, X).
    field : numpy.ndarray
        Pulling field in voxels, float64, C-contiguous, shape (2, Y, X),
        components YX; updated in place.

    Returns
    -------
    count : int
        Number of pixels whose sample lies inside the moving grid,
        [-0.5, N - 0.5) on both axes.
    total : float
        Sum over those pixels of |update|^2, by compensated (Neumaier)
        summation.

    Notes
    -----
    As thirion_update_3d on two axes: products (v * wy) * wx, corner order
    X fastest.
    """
    ny, nx = fixed.shape
    count = 0
    total = 0.0
    compensation = 0.0
    for y in range(ny):
        for x in range(nx):
            cy = field[0, y, x] + y
            cx = field[1, y, x] + x
            scale = 0.0
            inside = (cy >= -0.5) and (cy < ny - 0.5) and (cx >= -0.5) and (cx < nx - 0.5)
            if inside:
                y0, y1 = _taps(cy, ny)
                x0, x1 = _taps(cx, nx)
                ty = cy - math.floor(cy)
                tx = cx - math.floor(cx)
                ay = 1.0 - ty
                ax = 1.0 - tx
                by = 1.0 - ay
                bx = 1.0 - ax
                warped = 0.0
                warped += moving[y0, x0] * ay * ax
                warped += moving[y0, x1] * ay * bx
                warped += moving[y1, x0] * by * ax
                warped += moving[y1, x1] * by * bx
                diff = fixed[y, x] - warped
                denominator = gradient_sq[y, x] + diff * diff
                if abs(diff) >= 0.001 and denominator >= 1e-9:
                    scale = diff / denominator
            field[0, y, x] += scale * gradient[0, y, x]
            field[1, y, x] += scale * gradient[1, y, x]
            if inside:
                count += 1
                term = scale * scale * gradient_sq[y, x]
                running = total + term
                if abs(total) >= abs(term):
                    compensation += (total - running) + term
                else:
                    compensation += (term - running) + total
                total = running
    return count, total + compensation
