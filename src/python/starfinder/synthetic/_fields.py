"""Synthetic forward displacement fields, in ZYX voxel-index units."""
from typing import Literal
import numpy as np

def generate_displacement_field(
    shape: tuple[int, int, int],
    deform_type: Literal["polynomial", "gaussian", "multi_point", "linear"],
    max_displacement: float,
    seed: int = 42,
    **kwargs,
) -> np.ndarray:
    """Create a displacement field for local deformation.

    Parameters
    ----------
    shape : tuple[int, int, int]
        Volume shape as (Z, Y, X).
    deform_type : str
        Type of deformation: "polynomial", "gaussian", "multi_point", or "linear".
    max_displacement : float
        Maximum displacement in pixels.
    seed : int
        Random seed.
    **kwargs
        Additional parameters (radius for gaussian, n_points for multi_point).

    Returns
    -------
    np.ndarray
        Displacement field with shape (Z, Y, X, 3), where last axis is (dz, dy, dx).
    """
    rng = np.random.default_rng(seed)
    z_size, y_size, x_size = shape

    # Create coordinate grids normalized to [-1, 1]
    z_coords = np.linspace(-1, 1, z_size)
    y_coords = np.linspace(-1, 1, y_size)
    x_coords = np.linspace(-1, 1, x_size)
    zz, yy, xx = np.meshgrid(z_coords, y_coords, x_coords, indexing="ij")

    field = np.zeros((*shape, 3), dtype=np.float32)

    if deform_type == "polynomial":
        coeffs = rng.uniform(-1, 1, size=(3, 6))
        for axis in range(3):
            c = coeffs[axis]
            displacement = (
                c[0]
                + c[1] * xx
                + c[2] * yy
                + c[3] * zz
                + c[4] * xx * yy
                + c[5] * yy * zz
            )
            displacement = displacement / np.abs(displacement).max() * max_displacement
            field[..., axis] = displacement

    elif deform_type == "gaussian":
        radius = kwargs.get("radius", 30)
        margin = 0.3
        center_z = rng.uniform(-1 + margin, 1 - margin)
        center_y = rng.uniform(-1 + margin, 1 - margin)
        center_x = rng.uniform(-1 + margin, 1 - margin)
        direction = rng.normal(size=3)
        direction = direction / np.linalg.norm(direction)
        dist_sq = (
            ((zz - center_z) * z_size / 2) ** 2
            + ((yy - center_y) * y_size / 2) ** 2
            + ((xx - center_x) * x_size / 2) ** 2
        )
        gaussian = np.exp(-dist_sq / (2 * radius**2))
        for axis in range(3):
            field[..., axis] = gaussian * direction[axis] * max_displacement

    elif deform_type == "multi_point":
        n_points = kwargs.get("n_points", 4)
        radius = kwargs.get("radius", 25)
        for _ in range(n_points):
            margin = 0.3
            center_z = rng.uniform(-1 + margin, 1 - margin)
            center_y = rng.uniform(-1 + margin, 1 - margin)
            center_x = rng.uniform(-1 + margin, 1 - margin)
            direction = rng.normal(size=3)
            direction = direction / np.linalg.norm(direction)
            dist_sq = (
                ((zz - center_z) * z_size / 2) ** 2
                + ((yy - center_y) * y_size / 2) ** 2
                + ((xx - center_x) * x_size / 2) ** 2
            )
            gaussian = np.exp(-dist_sq / (2 * radius**2))
            for axis in range(3):
                field[..., axis] += gaussian * direction[axis] * max_displacement

    elif deform_type == "linear":
        coeffs = rng.uniform(-1, 1, size=(3, 5))
        for axis in range(3):
            c = coeffs[axis]
            displacement = c[0] + c[1] * xx + c[2] * yy + c[3] * zz + c[4] * xx * yy
            displacement = displacement / np.abs(displacement).max() * max_displacement
            field[..., axis] = displacement

    return field
