"""Smoothing and contrast steps with lengths in micrometres: ``gaussian`` and ``clahe``.

Both are per-channel steps of ``PREPROCESSING_METHODS`` whose lengths are converted with
the image's spacing (``StepContext.metadata``), for an input recipe of segmentation and for
``FOV.run`` alike (design/specs/segmentation-input-recipe.md, "The steps").
"""
from dataclasses import dataclass
from numbers import Integral, Real
import math

import numpy as np

from starfinder.preprocessing._lengths import check_lengths_zyx, kernel_pixels, sigma_pixels


@dataclass(frozen=True)
class GaussianConfig:
    """Gaussian smoothing with a standard deviation in micrometres.

    Parameters
    ----------
    sigma_um : float or tuple[float, float, float]
        Positive finite standard deviation in micrometres: one number for every
        axis, or a ZYX 3-tuple.

    Notes
    -----
    The step smooths each channel with ``skimage.filters.gaussian`` and its
    defaults (mode ``nearest``, truncate 4.0) at ``sigma_px_zyx = sigma_um /
    spacing`` per axis, 0 on an axis of length 1, which is the conversion and
    call of the ``seeded_watershed`` method. It needs the image's spacing. A
    float image keeps its dtype; an integer image is smoothed with
    ``preserve_range=True`` and rounded with ``np.rint`` back to its dtype.
    """
    sigma_um: float | tuple[float, float, float]

    def __post_init__(self):
        object.__setattr__(self, "sigma_um", check_lengths_zyx(self.sigma_um, "GaussianConfig.sigma_um"))


@dataclass(frozen=True)
class ClaheConfig:
    """Contrast-limited adaptive histogram equalization with a kernel in micrometres.

    Parameters
    ----------
    kernel_um : float or tuple[float, float, float]
        Positive finite kernel size in micrometres: one number for every axis,
        or a ZYX 3-tuple. It has no default.
    clip_limit : float
        Clip limit of the histograms, in (0, 1].
    nbins : int
        Number of histogram bins, at least 2.

    Notes
    -----
    Per channel: the fit region's minimum and maximum are mapped linearly to
    [0, 1] and the image is clipped there (``fitted.in_range``; a constant fit
    region gives zeros and records ``degenerate_range`` true), then
    ``skimage.exposure.equalize_adapthist(image, kernel_size=kernel_px_zyx,
    clip_limit, nbins)`` runs with ``kernel_px_zyx = max(1, floor(kernel_um /
    spacing + ½))`` per axis, 1 on an axis of length 1. The fit region is
    ``StepContext.fit_mask`` (None: the whole image). A float image keeps its
    dtype; on an integer image the [0, 1] result is scaled to the dtype's full
    range and rounded with ``np.rint``.
    """
    kernel_um: float | tuple[float, float, float]
    clip_limit: float = 0.01
    nbins: int = 256

    def __post_init__(self):
        if self.kernel_um is None:
            raise ValueError("ClaheConfig.kernel_um is required: give the kernel size in micrometres")
        object.__setattr__(self, "kernel_um", check_lengths_zyx(self.kernel_um, "ClaheConfig.kernel_um"))
        limit = self.clip_limit
        if isinstance(limit, bool) or not isinstance(limit, Real) or not math.isfinite(limit) or not 0 < limit <= 1:
            raise ValueError(f"ClaheConfig.clip_limit must be in (0, 1]; got {limit!r}")
        if isinstance(self.nbins, bool) or not isinstance(self.nbins, Integral) or self.nbins < 2:
            raise ValueError(f"ClaheConfig.nbins must be an integer >= 2; got {self.nbins!r}")
        object.__setattr__(self, "nbins", int(self.nbins))


def _channels(volume):
    return volume[..., None] if volume.ndim == 3 else volume


def _to_dtype(values, dtype, scale):
    """A float result in ``dtype``: kept for a float dtype, else ``np.rint(values × scale)`` (``scale`` 1 or the
    dtype's maximum)."""
    if dtype.kind == "f":
        return values.astype(dtype, copy=False)
    info = np.iinfo(dtype)
    return np.clip(np.rint(values * scale), info.min, info.max).astype(dtype)


def gaussian_used(config, metadata, shape_zyx) -> dict:
    """The value of a ``gaussian`` step on a grid: ``{"sigma_px_zyx": [...]}``."""
    return {"sigma_px_zyx": list(sigma_pixels(config.sigma_um, metadata, shape_zyx, "GaussianConfig.sigma_um"))}


def clahe_used(config, metadata, shape_zyx) -> dict:
    """The value of a ``clahe`` step on a grid: ``{"kernel_px_zyx": [...]}``."""
    return {"kernel_px_zyx": list(kernel_pixels(config.kernel_um, metadata, shape_zyx, "ClaheConfig.kernel_um"))}


def _gaussian(volume, config, context):
    from skimage.filters import gaussian
    sigma = tuple(gaussian_used(config, context.metadata, volume.shape[:3])["sigma_px_zyx"])
    channels = _channels(volume)
    result = np.empty_like(channels)
    for c in range(channels.shape[-1]):
        channel = channels[..., c]
        if channel.dtype.kind == "f":
            result[..., c] = gaussian(channel, sigma=sigma).astype(channel.dtype, copy=False)
        else:
            result[..., c] = _to_dtype(gaussian(channel, sigma=sigma, preserve_range=True), channel.dtype, 1)
    return result[..., 0] if volume.ndim == 3 else result, {}, {}


def _clahe(volume, config, context):
    from skimage.exposure import equalize_adapthist
    kernel = tuple(clahe_used(config, context.metadata, volume.shape[:3])["kernel_px_zyx"])
    mask = context.fit_mask
    channels = _channels(volume)
    result = np.empty_like(channels)
    ranges, degenerate = [], []
    for c in range(channels.shape[-1]):
        channel = channels[..., c].astype(np.float64)
        region = channel if mask is None else channel[mask]
        low, high = (float(region.min()), float(region.max())) if region.size else (0.0, 0.0)
        ranges.append([low, high])
        degenerate.append(bool(high == low))
        if high == low:
            equalized = np.zeros(channel.shape, np.float64)
        else:
            stretched = np.clip((channel - low) / (high - low), 0.0, 1.0)
            equalized = equalize_adapthist(stretched, kernel_size=kernel, clip_limit=config.clip_limit,
                                           nbins=config.nbins)
        dtype = channels.dtype
        result[..., c] = _to_dtype(equalized, dtype, 1 if dtype.kind == "f" else np.iinfo(dtype).max)
    one = volume.ndim == 3
    fitted = {"in_range": ranges[0] if one else ranges}
    diagnostics = {"degenerate_range": degenerate[0] if one else degenerate}
    return result[..., 0] if one else result, fitted, diagnostics
