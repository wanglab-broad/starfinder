"""Historical integer coordinate perturbations; no inverse-field interpretation."""
import numpy as np

def _apply_shift_to_spots(
    spots: list[tuple],
    shift: tuple[int, int, int],
    shape: tuple[int, int, int],
) -> list[tuple]:
    """Shift spot coordinates. Drop spots that move out of bounds.

    Parameters
    ----------
    spots : list[tuple[int, int, int, int, float]]
        List of (z, y, x, intensity, sigma) spot tuples.
    shift : tuple[int, int, int]
        Shift as (dz, dy, dx).
    shape : tuple[int, int, int]
        Volume bounds (Z, Y, X).

    Returns
    -------
    list[tuple[int, int, int, int, float]]
        Shifted spots with out-of-bounds spots removed.
    """
    dz, dy, dx = shift
    z_size, y_size, x_size = shape
    result = []
    for z, y, x, intensity, sigma in spots:
        nz, ny, nx = z + dz, y + dy, x + dx
        if 0 <= nz < z_size and 0 <= ny < y_size and 0 <= nx < x_size:
            result.append((nz, ny, nx, intensity, sigma))
    return result

def _apply_deformation_to_spots(
    spots: list[tuple],
    field: np.ndarray,
    shape: tuple[int, int, int],
) -> list[tuple]:
    """Move spots by sampling displacement field at their positions.

    Parameters
    ----------
    spots : list[tuple[int, int, int, int, float]]
        List of (z, y, x, intensity, sigma) spot tuples.
    field : np.ndarray
        Displacement field with shape (Z, Y, X, 3), last axis is (dz, dy, dx).
    shape : tuple[int, int, int]
        Volume bounds (Z, Y, X).

    Returns
    -------
    list[tuple[int, int, int, int, float]]
        Deformed spots with out-of-bounds spots removed.
    """
    z_size, y_size, x_size = shape
    result = []
    for z, y, x, intensity, sigma in spots:
        # Clamp to valid field indices (spot should already be in bounds)
        fz = min(max(z, 0), z_size - 1)
        fy = min(max(y, 0), y_size - 1)
        fx = min(max(x, 0), x_size - 1)
        dz, dy, dx = field[fz, fy, fx]
        nz = int(round(z + dz))
        ny = int(round(y + dy))
        nx = int(round(x + dx))
        if 0 <= nz < z_size and 0 <= ny < y_size and 0 <= nx < x_size:
            result.append((nz, ny, nx, intensity, sigma))
    return result
