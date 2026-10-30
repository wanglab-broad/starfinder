"""XY slice-wise morphology with safe intermediate precision."""
from dataclasses import dataclass

import numpy as np
from skimage.morphology import black_tophat, disk, erosion, reconstruction, white_tophat

from starfinder.image import _validate_image


def _radius(radius):
    if isinstance(radius, bool) or not isinstance(radius, int) or radius < 0:
        raise ValueError("radius_yx must be a nonnegative integer")


@dataclass(frozen=True)
class ReconstructionConfig:
    """Background removal by morphological reconstruction in XY, on each Z plane and channel.

    Parameters
    ----------
    radius_yx : int
        Nonnegative disk radius in XY pixels.

    Notes
    -----
    Disk radius in XY pixels, independently applied to each Z/channel slice.
    """
    radius_yx: int = 3

    def __post_init__(self):
        _radius(self.radius_yx)


@dataclass(frozen=True)
class TophatConfig:
    """White top-hat background removal in XY, on each Z plane and channel.

    Parameters
    ----------
    radius_yx : int
        Nonnegative disk radius in XY pixels.

    Notes
    -----
    Disk radius in XY pixels; no filtering along Z or channels.
    """
    radius_yx: int = 3

    def __post_init__(self):
        _radius(self.radius_yx)


def _morph(volume, radius, reconstruct):
    volume = _validate_image(volume)
    if volume.dtype.kind in "ui" and volume.dtype.itemsize > 4:
        raise ValueError("morphology supports integers up to 32 bits and floating images")
    channels = volume[..., None] if volume.ndim == 3 else volume
    result = np.empty_like(channels)
    se = disk(radius)
    for c in range(channels.shape[-1]):
        for z in range(channels.shape[0]):
            image = channels[z, :, :, c].astype(np.float64)
            if reconstruct:
                background = reconstruction(erosion(image, se), image, method="dilation")
                image = image - background
                image = image + white_tophat(image, se) - black_tophat(image, se)
            else:
                image = white_tophat(image, se)
            if not np.isfinite(image).all():
                raise ValueError("morphology produced nonfinite values")
            # Explicit saturation to representable source dtype, never wrap.
            limits = np.iinfo(volume.dtype) if volume.dtype.kind in "ui" else np.finfo(volume.dtype)
            image = np.clip(image, limits.min, limits.max)
            if not np.isfinite(image).all():
                raise ValueError("morphology produced nonfinite values")
            result[z, :, :, c] = image.astype(volume.dtype)
    return result[..., 0] if volume.ndim == 3 else result


def filter_tophat(volume: np.ndarray, *, config: TophatConfig = TophatConfig()) -> np.ndarray:
    """White tophat per XY slice/channel, preserving source dtype and shape.

    Allocates an output and float64 slice work. Reflect boundaries (skimage
    default), disk footprint. Constants give zero; empty/nonfinite inputs error.
    Inputs are not mutated. Results saturate to source dtype representability.
    """
    return _morph(volume, config.radius_yx, False)


def correct_background(volume: np.ndarray, *, config: ReconstructionConfig = ReconstructionConfig()) -> np.ndarray:
    """Correct the background by morphological reconstruction; return the corrected image.

    Per XY slice and channel: subtract the background reconstructed by dilation
    from the slice eroded with a disk of ``config.radius_yx``, then add the white
    top-hat and subtract the black top-hat of the result. The step's registry ID
    stays ``reconstruction``.

    Float64 slice intermediates avoid uint16 signed overflow. Output retains
    source dtype, explicitly saturating to its representable range (float
    outputs may be negative). Allocates output and slice work, not a full float
    volume. Disk/reflect XY boundaries; no Z/channel mixing. Constants give
    zero. Empty/nonfinite inputs error. Input is never mutated.
    """
    return _morph(volume, config.radius_yx, True)
