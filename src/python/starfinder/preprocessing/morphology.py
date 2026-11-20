"""XY slice-wise morphology with safe intermediate precision."""
from dataclasses import dataclass, field, replace

import numpy as np
from skimage.morphology import black_tophat, disk, erosion, white_tophat

from starfinder.image import _validate_image
from starfinder.preprocessing._lengths import OMIT_NONE, check_length, yx_pixels


def _radius(radius):
    if isinstance(radius, bool) or not isinstance(radius, int) or radius < 0:
        raise ValueError("radius_yx must be a nonnegative integer")


def _check_radius(config):
    """Exactly one of radius_yx and radius_um; neither given is radius_yx 3, the default in pixels."""
    name = type(config).__name__
    if config.radius_um is None:
        if config.radius_yx is None:
            object.__setattr__(config, "radius_yx", 3)
        _radius(config.radius_yx)
        return
    if config.radius_yx is not None:
        raise ValueError(f"{name} takes exactly one of radius_yx (pixels) and radius_um (micrometres); got both")
    object.__setattr__(config, "radius_um", check_length(config.radius_um, f"{name}.radius_um"))


def _pixel_config(config, metadata):
    """The config with radius_um converted to radius_yx = floor(radius_um / spacing_yx + ½) with ``metadata``."""
    if config.radius_um is None:
        return config
    return replace(config, radius_yx=yx_pixels(config.radius_um, metadata, f"{type(config).__name__}.radius_um"),
                   radius_um=None)


def _pixels(config, what):
    if config.radius_yx is None:
        raise ValueError(f"{what} with radius_um needs the image's spacing: run the step through run_step (FOV.run or "
                         "an input recipe), which converts it, or give radius_yx")
    return config.radius_yx


@dataclass(frozen=True)
class ReconstructionConfig:
    """Background removal by morphological reconstruction in XY, on each Z plane and channel.

    Parameters
    ----------
    radius_yx : int or None
        Nonnegative disk radius in XY pixels; ``None`` with ``radius_um`` also ``None`` is 3 pixels.
    radius_um : float or None
        Nonnegative disk radius in micrometres, converted to
        ``floor(radius_um / spacing_yx + ½)`` pixels with the image's spacing;
        ``None`` gives the radius in pixels.

    Notes
    -----
    Disk radius in XY pixels, independently applied to each Z/channel slice.
    The radius sets the erosion that seeds the reconstruction and the two
    top-hats; the reconstruction itself uses the 3 x 3 neighbourhood.
    Exactly one of ``radius_yx`` and ``radius_um`` is given (neither: 3
    pixels). ``radius_um`` needs equal Y and X spacing (relative difference at
    most 1e-9) and is converted by the step (``run_step``) with
    ``StepContext.metadata``; an input recipe of segmentation takes
    ``radius_um`` only (design/specs/segmentation-input-recipe.md).
    """
    radius_yx: int | None = None
    radius_um: float | None = field(default=None, metadata=OMIT_NONE)

    def __post_init__(self):
        _check_radius(self)


@dataclass(frozen=True)
class TophatConfig:
    """White top-hat background removal in XY, on each Z plane and channel.

    Parameters
    ----------
    radius_yx : int or None
        Nonnegative disk radius in XY pixels; ``None`` with ``radius_um`` also ``None`` is 3 pixels.
    radius_um : float or None
        Nonnegative disk radius in micrometres, converted to
        ``floor(radius_um / spacing_yx + ½)`` pixels with the image's spacing;
        ``None`` gives the radius in pixels.

    Notes
    -----
    Disk radius in XY pixels; no filtering along Z or channels. Exactly one
    of ``radius_yx`` and ``radius_um`` is given (neither: 3 pixels).
    ``radius_um`` needs equal Y and X spacing (relative difference at most
    1e-9) and is converted by the step (``run_step``) with
    ``StepContext.metadata``; an input recipe of segmentation takes
    ``radius_um`` only (design/specs/segmentation-input-recipe.md).
    """
    radius_yx: int | None = None
    radius_um: float | None = field(default=None, metadata=OMIT_NONE)

    def __post_init__(self):
        _check_radius(self)


def _reconstruct_by_dilation(marker, mask, dtype):
    """Grayscale reconstruction by dilation of marker under mask, in dtype, with 8-connectivity.

    Parameters
    ----------
    marker : numpy.ndarray
        Float64 YX slice, at most mask everywhere and integer-valued when
        dtype is an integer type.
    mask : numpy.ndarray
        Float64 YX slice of the same shape.
    dtype : numpy.dtype
        Work type: the source integer type of the slice, or float64.

    Returns
    -------
    numpy.ndarray
        The float64 reconstruction, with the shape of mask.

    Notes
    -----
    Runs SimpleITK's ReconstructionByDilationImageFilter (Robinson and Whelan's
    downhill filter, no global sort) with FullyConnected, the 3 x 3 neighbourhood
    of skimage.morphology.reconstruction's default footprint; its result equals
    reconstruction(marker, mask, method="dilation") exactly. The reconstruction
    takes its values from marker and mask, so an integer slice gives the same
    result in its own type as in float64, and the casts to dtype and back are
    exact.
    """
    import SimpleITK as sitk

    reconstruct = sitk.ReconstructionByDilationImageFilter()
    reconstruct.SetFullyConnected(True)
    background = reconstruct.Execute(sitk.GetImageFromArray(np.ascontiguousarray(marker, dtype=dtype)),
                                     sitk.GetImageFromArray(np.ascontiguousarray(mask, dtype=dtype)))
    return sitk.GetArrayFromImage(background).astype(np.float64)


def _morph(volume, radius, reconstruct):
    volume = _validate_image(volume)
    if volume.dtype.kind in "ui" and volume.dtype.itemsize > 4:
        raise ValueError("morphology supports integers up to 32 bits and floating images")
    channels = volume[..., None] if volume.ndim == 3 else volume
    result = np.empty_like(channels)
    se = disk(radius)
    # The reconstruction's work type: the source integer type in native byte order, otherwise float64.
    work = volume.dtype.newbyteorder("=") if volume.dtype.kind in "ui" else np.dtype(np.float64)
    for c in range(channels.shape[-1]):
        for z in range(channels.shape[0]):
            image = channels[z, :, :, c].astype(np.float64)
            if reconstruct:
                background = _reconstruct_by_dilation(erosion(image, se), image, work)
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

    Parameters
    ----------
    volume : numpy.ndarray
        Finite intensity image in ZYX or ZYXC axis order.
    config : TophatConfig
        XY disk radius for the white top-hat; the default is 3 pixels.

    Returns
    -------
    numpy.ndarray
        White top-hat image with the input shape and dtype.

    Notes
    -----
    Allocates an output and float64 slice work. Reflect boundaries (skimage
    default), disk footprint. Constants give zero; empty/nonfinite inputs error.
    Inputs are not mutated. Results saturate to source dtype representability.
    A config with ``radius_um`` is converted by the step (``run_step``), which
    knows the spacing; called directly it raises ValueError.
    """
    return _morph(volume, _pixels(config, "filter_tophat"), False)


def correct_background(volume: np.ndarray, *, config: ReconstructionConfig = ReconstructionConfig()) -> np.ndarray:
    """Correct the background by morphological reconstruction; return the corrected image.

    Parameters
    ----------
    volume : numpy.ndarray
        Finite intensity image in ZYX or ZYXC axis order.
    config : ReconstructionConfig
        XY disk radius for reconstruction; the default is 3 pixels.

    Returns
    -------
    numpy.ndarray
        Background-corrected image with the input shape and dtype.

    Notes
    -----
    Per XY slice and channel: subtract the background reconstructed by dilation
    from the slice eroded with a disk of ``config.radius_yx``, then add the white
    top-hat and subtract the black top-hat of the result. The step's registry ID
    stays ``reconstruction``.

    The reconstruction runs SimpleITK's ReconstructionByDilationImageFilter
    (Robinson and Whelan's downhill filter) with 8-connectivity, in the
    slice's own integer type, or in float64 for floating inputs. Its result
    equals ``skimage.morphology.reconstruction(..., method="dilation")``
    exactly, without that function's sort of every seed and mask value.

    Float64 slice intermediates avoid uint16 signed overflow. Output retains
    source dtype, explicitly saturating to its representable range (float
    outputs may be negative). Allocates output and slice work, not a full float
    volume. Disk/reflect XY boundaries; no Z/channel mixing. Constants give
    zero. Empty/nonfinite inputs error. Input is never mutated. A config with
    ``radius_um`` is converted by the step (``run_step``), which knows the
    spacing; called directly it raises ValueError.
    """
    return _morph(volume, _pixels(config, "correct_background"), True)
