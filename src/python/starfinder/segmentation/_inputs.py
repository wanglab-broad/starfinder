"""Input functions that are not methods (D4): the composite, the Flamingo enhancement, normalization, rescaling.

Each takes arrays, never changes them, and returns its result with a record mapping
``{"function", "config", "inputs", "output", ...}`` whose hashes are array SHA-256
values (docs/segmentation-contract.md, "Input preparation and label functions").
None is registered in ``PREPROCESSING_METHODS``: the composite and the enhancement
combine two images, normalization returns float32, and rescaling changes the grid.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from numbers import Integral, Real

import numpy as np

from starfinder.image import ImageMetadata, IncompatibleGeometryError, _validate_image

from ._labels import _json, array_sha256


def _quantile(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not 0 <= value < 0.5:
        raise ValueError(f"{name} must be a fraction in [0, 0.5); got {value!r}")
    return float(value)


@dataclass(frozen=True, kw_only=True)
class CompositeConfig:
    """Settings of :func:`composite_nuclei_amplicon`.

    Each image is stretched between its q and 1 − q quantiles over the whole
    volume (fractions): ``nuclear_quantile`` for the nuclear (DAPI) image and
    ``amplicon_quantile`` for the amplicon (reference merged) image. The defaults
    are the values of ``create_nuclei_amplicon_overlay.py``.
    """

    nuclear_quantile: float = 0.005
    amplicon_quantile: float = 0.001

    def __post_init__(self):
        for name in ("nuclear_quantile", "amplicon_quantile"):
            object.__setattr__(self, name, _quantile(getattr(self, name), name))


@dataclass(frozen=True, kw_only=True)
class FlamingoEnhancementConfig:
    """Settings of :func:`enhance_with_flamingo`.

    ``flamingo_quantile`` and ``nuclear_quantile`` are the stretch fractions (q and
    1 − q quantiles over the whole volume); ``median_radius_px`` is the radius in
    pixels of the disk the stretched Flamingo image is median-filtered with in each
    Z plane (0: one pixel, no filtering). The defaults are the values of
    ``enhance_dapi_with_flamingo.py``.
    """

    flamingo_quantile: float = 0.005
    nuclear_quantile: float = 0.001
    median_radius_px: int = 1

    def __post_init__(self):
        for name in ("flamingo_quantile", "nuclear_quantile"):
            object.__setattr__(self, name, _quantile(getattr(self, name), name))
        radius = self.median_radius_px
        if isinstance(radius, bool) or not isinstance(radius, Integral) or radius < 0:
            raise ValueError(f"median_radius_px must be an integer >= 0; got {radius!r}")
        object.__setattr__(self, "median_radius_px", int(radius))


def _pair(first, second, names):
    """Two finite real ZYX arrays of one shape, checked before any computation."""
    first, second = np.asarray(first), np.asarray(second)
    for name, array in zip(names, (first, second)):
        if array.ndim != 3:
            raise IncompatibleGeometryError(f"{name} must be ZYX (a plane is 1×Y×X); got shape {array.shape}")
    if first.shape != second.shape:
        raise IncompatibleGeometryError(f"{names[0]} {first.shape} and {names[1]} {second.shape} must have one shape")
    return _validate_image(first, ndim=(3,)), _validate_image(second, ndim=(3,))


def _stretch(image, q):
    """The scripts' contrast stretch: rescale_intensity between the q and 1 − q quantiles (linear)."""
    from skimage.exposure import rescale_intensity

    vmin = np.quantile(image, q)
    vmax = np.quantile(image, 1 - q)
    return rescale_intensity(image, (vmin, vmax)), [float(vmin), float(vmax)]


def composite_nuclei_amplicon(nuclear, amplicon, *,
                              config: CompositeConfig = CompositeConfig()) -> tuple[np.ndarray, dict]:
    """The DAPI–amplicon composite: nuclei and amplicon-filled cytoplasm bright in one image.

    ``create_nuclei_amplicon_overlay.py`` bit for bit: stretch each image between its
    q and 1 − q quantiles (``np.quantile``, linear, whole volume) with
    ``rescale_intensity``, convert both with ``img_as_float``, take their voxel-wise
    maximum and convert with ``img_as_ubyte``. A constant image passes through the
    stretch unchanged. The script's optional maximum projection is not part of the
    function: it is a run's ``projection``. The arrays carry no metadata, so the
    caller (``FOV.segment``) checks that both are on the reference grid.

    Parameters
    ----------
    nuclear : numpy.ndarray
        Finite real ZYX nuclear (DAPI) image; a plane is 1×Y×X.
    amplicon : numpy.ndarray
        Finite real ZYX amplicon image of the same shape, such as the reference
        round's channel maximum.
    config : CompositeConfig
        The two stretch fractions.

    Returns
    -------
    tuple[numpy.ndarray, dict]
        The ``uint8`` ZYX composite, and the record ``{"function", "config",
        "inputs", "output", "quantiles"}``: ``inputs`` holds the nuclear and
        amplicon SHA-256, ``quantiles`` the low and high values reached for
        ``nuclear`` and ``amplicon``.

    Raises
    ------
    IncompatibleGeometryError
        An input is not three-dimensional, or the shapes differ.
    ValueError
        An input is empty or not finite and real (InvalidImageError).
    TypeError
        config is not a CompositeConfig.
    """
    from skimage.util import img_as_float, img_as_ubyte

    if not isinstance(config, CompositeConfig):
        raise TypeError("config must be a CompositeConfig")
    nuclear, amplicon = _pair(nuclear, amplicon, ("nuclear", "amplicon"))
    amplicon_stretched, amplicon_values = _stretch(amplicon, config.amplicon_quantile)
    nuclear_stretched, nuclear_values = _stretch(nuclear, config.nuclear_quantile)
    output = img_as_ubyte(np.maximum(img_as_float(nuclear_stretched), img_as_float(amplicon_stretched)))
    record = {"function": "composite_nuclei_amplicon", "config": _json(asdict(config)),
              "inputs": [array_sha256(nuclear), array_sha256(amplicon)], "output": array_sha256(output),
              "quantiles": {"nuclear": nuclear_values, "amplicon": amplicon_values}}
    return output, record


def enhance_with_flamingo(nuclear, flamingo, *,
                          config: FlamingoEnhancementConfig = FlamingoEnhancementConfig()) -> tuple[np.ndarray, dict]:
    """DAPI with the Flamingo (cytoplasm) stain subtracted, which separates touching nuclei.

    ``enhance_dapi_with_flamingo.py`` bit for bit: stretch the Flamingo image between
    its q and 1 − q quantiles, median-filter each Z plane with a disk of
    ``median_radius_px``, stretch the nuclear image likewise, convert both to float
    and return ``img_as_ubyte(nuclear × (1 − flamingo))``. A constant image passes
    through the stretch unchanged.

    Parameters
    ----------
    nuclear : numpy.ndarray
        Finite real ZYX nuclear (DAPI) image; a plane is 1×Y×X.
    flamingo : numpy.ndarray
        Finite real ZYX Flamingo image of the same shape.
    config : FlamingoEnhancementConfig
        The two stretch fractions and the median radius.

    Returns
    -------
    tuple[numpy.ndarray, dict]
        The ``uint8`` ZYX enhanced image, and the record ``{"function", "config",
        "inputs", "output", "quantiles"}``: ``inputs`` holds the nuclear and
        Flamingo SHA-256, ``quantiles`` the low and high values reached for
        ``nuclear`` and ``flamingo``.

    Raises
    ------
    IncompatibleGeometryError
        An input is not three-dimensional, or the shapes differ.
    ValueError
        An input is empty or not finite and real (InvalidImageError).
    TypeError
        config is not a FlamingoEnhancementConfig.
    """
    from skimage.filters import median
    from skimage.morphology import disk
    from skimage.util import img_as_float, img_as_ubyte, invert

    if not isinstance(config, FlamingoEnhancementConfig):
        raise TypeError("config must be a FlamingoEnhancementConfig")
    nuclear, flamingo = _pair(nuclear, flamingo, ("nuclear", "flamingo"))
    stretched, flamingo_values = _stretch(flamingo, config.flamingo_quantile)
    footprint = disk(config.median_radius_px)
    filtered = np.empty_like(stretched)  # the script assigns each filtered plane back into its array
    for z in range(stretched.shape[0]):
        filtered[z] = median(stretched[z], footprint)
    nuclear_stretched, nuclear_values = _stretch(nuclear, config.nuclear_quantile)
    output = img_as_ubyte(img_as_float(nuclear_stretched) * invert(img_as_float(filtered)))
    record = {"function": "enhance_with_flamingo", "config": _json(asdict(config)),
              "inputs": [array_sha256(nuclear), array_sha256(flamingo)], "output": array_sha256(output),
              "quantiles": {"nuclear": nuclear_values, "flamingo": flamingo_values}}
    return output, record


def _percent(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not 0 <= value <= 100:
        raise ValueError(f"{name} must be a percent in [0, 100]; got {value!r}")
    return float(value)


def normalize_percentiles(image, *, p_low: float = 1.0, p_high: float = 99.8,
                          axes=None) -> tuple[np.ndarray, dict]:
    """Percentile normalization to float32, unclipped: csbdeep's ``normalize``.

    ``(x − p_low) / (p_high − p_low + 1e-20)`` in float32, with the two percentiles
    computed by ``np.percentile`` (linear) over ``axes`` and cast to float32, as
    csbdeep 0.8.2 computes it for the ``stardist`` method. A constant image maps to
    zeros. Unlike ``percentile_normalization`` (a preprocessing method), the dtype
    becomes float32 and nothing is clipped.

    Parameters
    ----------
    image : numpy.ndarray
        Finite real ZYX or ZYXC image.
    p_low, p_high : float
        Percents of the intensity distribution, 0 ≤ p_low < p_high ≤ 100.
    axes : tuple[int, ...] | None
        Axes the percentiles are taken over; None: the spatial axes (0, 1, 2), so
        each channel of a ZYXC image is normalized on its own.

    Returns
    -------
    tuple[numpy.ndarray, dict]
        The float32 image of the input's shape, and the record ``{"function",
        "config", "inputs", "output", "percentiles"}``: ``config`` holds ``p_low``,
        ``p_high`` and ``axes``, ``percentiles`` the ``low`` and ``high`` values
        reached (a number, or a nested list with one value per position of the
        axes not reduced).

    Raises
    ------
    ValueError
        The image is not finite, real and ZYX or ZYXC (InvalidImageError), a
        percent is outside [0, 100] or p_low ≥ p_high, or an axis is repeated or
        outside the image.
    """
    image = _validate_image(image)
    p_low, p_high = _percent(p_low, "p_low"), _percent(p_high, "p_high")
    if not p_low < p_high:
        raise ValueError(f"p_low must be below p_high; got {p_low} and {p_high}")
    if axes is None:
        axes = (0, 1, 2)
    elif isinstance(axes, (bool, np.bool_)) or not isinstance(axes, (tuple, list)) or any(
            isinstance(a, (bool, np.bool_)) or not isinstance(a, (int, np.integer)) for a in axes):
        raise ValueError(f"axes must be a tuple of axis indices or None; got {axes!r}")
    axes = tuple(int(a) for a in axes)
    if not axes or len(set(axes)) != len(axes) or any(not 0 <= a < image.ndim for a in axes):
        raise ValueError(f"axes must be distinct axes of the {image.ndim}-dimensional image; got {axes}")
    low = np.percentile(image, p_low, axis=axes, keepdims=True)
    high = np.percentile(image, p_high, axis=axes, keepdims=True)
    x = image.astype(np.float32, copy=False)
    low32, high32 = low.astype(np.float32, copy=False), high.astype(np.float32, copy=False)
    output = (x - low32) / (high32 - low32 + np.float32(1e-20))

    def reached(values):
        values = np.squeeze(values, axis=axes)
        return values.item() if values.ndim == 0 else values.tolist()

    record = {"function": "normalize_percentiles", "config": {"p_low": p_low, "p_high": p_high, "axes": list(axes)},
              "inputs": [array_sha256(image)], "output": array_sha256(output),
              "percentiles": {"low": reached(low), "high": reached(high)}}
    return output, record


def rescale_input(image, metadata: ImageMetadata, *, scale_zyx) -> tuple[np.ndarray, ImageMetadata, dict]:
    """Resample a ZYX image by per-axis factors, with the matching metadata.

    For a direct caller that detects nuclei on a shrunk input with a method that
    has no scale of its own; :func:`labels_to_grid` maps the labels back onto the
    exact input grid. ``skimage.transform.rescale`` with linear interpolation,
    anti-aliasing on the shrinking axes (its default) and its default range
    conversion, so the output is float64 and an integer image is on the [0, 1]
    scale of its dtype, as the legacy ``rescale(image, [1, .5, .5])`` gives it.
    Each output size is ``round(n × factor)``, at least one pixel.

    The metadata's ``frame_id`` gains ``/rescale:<z>,<y>,<x>``; its spacing, when it
    has one, is divided by the factors; its origin moves to the centre of the first
    output voxel (``+ direction @ ((0.5 / factor − 0.5) × spacing)``) when spacing,
    origin and direction are known, and is otherwise unknown; direction and unit
    are kept.

    Parameters
    ----------
    image : numpy.ndarray
        Finite real ZYX image.
    metadata : ImageMetadata
        The image's geometry.
    scale_zyx : tuple[float, float, float]
        Dimensionless factors > 0 (0.5 halves an axis).

    Returns
    -------
    tuple[numpy.ndarray, ImageMetadata, dict]
        The resampled float64 image, its metadata, and the record ``{"function",
        "config", "inputs", "output", "input_shape", "output_shape",
        "frame_id"}``, ``config`` holding ``scale_zyx``.

    Raises
    ------
    ValueError
        A factor is not a finite number > 0, or the image is not finite, real
        and ZYX (InvalidImageError).
    TypeError
        metadata is not an ImageMetadata.
    """
    from skimage.transform import rescale

    if not isinstance(metadata, ImageMetadata):
        raise TypeError("metadata must be an ImageMetadata")
    factors = tuple(scale_zyx) if isinstance(scale_zyx, (tuple, list)) else None
    if factors is None or len(factors) != 3 or any(
            isinstance(f, bool) or not isinstance(f, Real) or not math.isfinite(f) or f <= 0 for f in factors):
        raise ValueError(f"scale_zyx must be three finite factors > 0; got {scale_zyx!r}")
    factors = tuple(float(f) for f in factors)
    image = _validate_image(image, ndim=(3,))
    output = rescale(image, factors, order=1)
    frame_id = f"{metadata.frame_id}/rescale:{','.join(repr(f) for f in factors)}"
    spacing = origin = None
    if metadata.spacing_zyx is not None:
        spacing = tuple(s / f for s, f in zip(metadata.spacing_zyx, factors))
        if metadata.origin_zyx is not None and metadata.direction_zyx is not None:
            shift = np.array([(0.5 / f - 0.5) * s for s, f in zip(metadata.spacing_zyx, factors)])
            origin = tuple(np.asarray(metadata.origin_zyx) + np.asarray(metadata.direction_zyx) @ shift)
    rescaled = replace(metadata, frame_id=frame_id, spacing_zyx=spacing, origin_zyx=origin)
    record = {"function": "rescale_input", "config": {"scale_zyx": list(factors)},
              "inputs": [array_sha256(image)], "output": array_sha256(output),
              "input_shape": list(image.shape), "output_shape": list(output.shape), "frame_id": frame_id}
    return output, rescaled, record
