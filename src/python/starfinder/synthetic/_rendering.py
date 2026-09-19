"""Shared analytical Gaussian renderer."""
from typing import Literal
import numpy as np
import pandas as pd
from ._truth import _scene_table, _render_rows

def render_spots(
    shape: tuple[int, int, int],
    spots: pd.DataFrame,
    background: int = 20,
    noise_std: int = 10,
    seed: int | None = None,
    add_noise: bool = True,
    dtype: Literal["uint8", "uint16"] = "uint8",
) -> np.ndarray:
    """Create a single 3D image stack with spots at specified locations.

    Each spot carries its own sigma, enabling per-round PSF variation.

    Parameters
    ----------
    shape : tuple[int, int, int]
        Image shape as (z, y, x).
    spots : pandas.DataFrame
        Scene table with unique spot_id and finite z/y/x/intensity/sigma
        columns. Coordinates are zero-based voxel indices; sigma is positive.
    background : int
        Mean background intensity.
    noise_std : int
        Standard deviation of additive noise.
    seed : int, optional
        Random seed for reproducibility.
    add_noise : bool
        Whether to add Gaussian noise (default True).
    dtype : {"uint8", "uint16"}
        Output data type (default "uint8").

    Returns
    -------
    np.ndarray
        3D image stack with specified dtype.
    """
    if dtype not in ('uint8', 'uint16'):
        raise ValueError('dtype must be uint8 or uint16')
    rng = np.random.default_rng(seed)

    # Create background with slight variation
    image = rng.normal(background, background / 4, shape).astype(np.float32)

    # Add spots using localized Gaussian kernels
    for z, y, x, intensity, sigma in _render_rows(spots):
        if 0 <= z < shape[0] and 0 <= y < shape[1] and 0 <= x < shape[2]:
            kernel_radius = int(np.ceil(sigma * 4))
            z0, z1 = max(0, z - kernel_radius), min(shape[0], z + kernel_radius + 1)
            y0, y1 = max(0, y - kernel_radius), min(shape[1], y + kernel_radius + 1)
            x0, x1 = max(0, x - kernel_radius), min(shape[2], x + kernel_radius + 1)
            z0, z1 = int(np.ceil(z0)), int(np.ceil(z1))
            y0, y1 = int(np.ceil(y0)), int(np.ceil(y1))
            x0, x1 = int(np.ceil(x0)), int(np.ceil(x1))

            zz, yy, xx = np.ogrid[z0 - z : z1 - z, y0 - y : y1 - y, x0 - x : x1 - x]
            dist_sq = zz**2 + yy**2 + xx**2
            gaussian_spot = intensity * np.exp(-dist_sq / (2 * sigma**2))
            image[z0:z1, y0:y1, x0:x1] += gaussian_spot

    # Add noise (optional)
    if add_noise and noise_std > 0:
        image += rng.normal(0, noise_std, shape)

    # Clip and convert to output dtype
    if dtype == "uint8":
        image = np.clip(image, 0, 255).astype(np.uint8)
    else:
        image = np.clip(image, 0, 65535).astype(np.uint16)

    return image

def generate_volume(
    shape: tuple[int, int, int],
    n_spots: int = 20,
    spot_intensity: int = 200,
    background: int = 20,
    noise_std: int = 5,
    seed: int | None = None,
) -> np.ndarray:
    """Create a single 3D volume with Gaussian spots for registration testing.

    Parameters
    ----------
    shape : tuple[int, int, int]
        Volume shape as (Z, Y, X).
    n_spots : int
        Number of spots to place (default 20).
    spot_intensity : int
        Peak intensity of spots (default 200).
    background : int
        Background intensity level (default 20).
    noise_std : int
        Standard deviation of Gaussian noise (default 5).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Volume with shape (Z, Y, X) containing spots, dtype uint8.
    """
    rng = np.random.default_rng(seed)
    spot_sigma = 1.5

    z_size, y_size, x_size = shape
    margin_z = max(1, z_size // 8)
    margin_xy = max(5, min(y_size, x_size) // 16)

    spots = []
    for _ in range(n_spots):
        z = int(rng.integers(margin_z, max(margin_z + 1, z_size - margin_z)))
        y = int(rng.integers(margin_xy, max(margin_xy + 1, y_size - margin_xy)))
        x = int(rng.integers(margin_xy, max(margin_xy + 1, x_size - margin_xy)))
        intensity = int(spot_intensity + rng.integers(-20, 21))
        spots.append((z, y, x, intensity, spot_sigma))

    return render_spots(
        shape=shape,
        spots=_scene_table(spots),
        background=background,
        noise_std=noise_std,
        seed=seed,
        add_noise=True,
        dtype="uint8",
    )
