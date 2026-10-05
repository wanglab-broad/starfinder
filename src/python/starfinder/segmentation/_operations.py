"""Label functions that are not methods (D4): expand_labels, labels_to_grid and the culture extension through z."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from numbers import Real

import numpy as np
from scipy import ndimage

from starfinder.image import ImageMetadata, IncompatibleGeometryError, _validate_image

from ._labels import _json, array_sha256, to_label_dtype


def _target_shape(target_shape):
    shape = tuple(target_shape) if isinstance(target_shape, (tuple, list)) else None
    if (shape is None or len(shape) != 3
            or any(isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 1 for n in shape)):
        raise ValueError(f"target_shape must be three positive integers (ZYX); got {target_shape!r}")
    return tuple(int(n) for n in shape)


def labels_to_grid(labels, *, target_shape) -> tuple[np.ndarray, dict]:
    """Map a label image onto exactly ``target_shape`` by nearest neighbour at pixel centres.

    Per axis the source index of output index i is ⌊(i + 0.5) × n_source / n_target⌋,
    computed in integers, so an odd size keeps its grid (unlike the legacy rescale
    round trip, which turns 61×63 into 60×64). For a direct caller that detected
    nuclei on a shrunk input with a method without its own scale; the caller
    attaches the target grid. Values are kept and the output is ``uint32``.

    Returns
    -------
    tuple[numpy.ndarray, dict]
        The C-contiguous ``uint32`` ZYX labels, and the record ``{"function",
        "config", "inputs", "output", "source_shape", "target_shape"}`` whose
        ``inputs`` and ``output`` are array SHA-256 values.

    Raises
    ------
    IncompatibleGeometryError
        labels is not three-dimensional (a plane is 1×Y×X).
    ValueError
        target_shape is not three positive integers, or a label is negative or
        above 2**32 - 1.
    TypeError
        labels does not have an integer dtype.
    """
    array = np.asarray(labels)
    if array.ndim != 3:
        raise IncompatibleGeometryError(f"labels must be ZYX (a plane is 1×Y×X); got {array.ndim} dimensions")
    target = _target_shape(target_shape)
    source = to_label_dtype(array)
    index = [((2 * np.arange(n_target) + 1) * n_source) // (2 * n_target)
             for n_source, n_target in zip(source.shape, target)]
    output = np.ascontiguousarray(source[np.ix_(*index)])
    record = {"function": "labels_to_grid", "config": {"target_shape": list(target)},
              "inputs": [array_sha256(array)], "output": array_sha256(output),
              "source_shape": list(array.shape), "target_shape": list(target)}
    return output, record


def _nonnegative(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0 \
            or (positive and value == 0):
        raise ValueError(f"{name} must be a finite number {'> 0' if positive else '>= 0'}; got {value!r}")
    return float(value)


EXPANSION_UNITS = ("pixel", "um")
EXPANSION_MODES = ("planar", "volumetric")


@dataclass(frozen=True)
class ExpandLabelsConfig:
    """Settings of :func:`expand_labels`, the one label expansion; no field has a default.

    ``distance`` is how far each label grows into the background, in ``unit``:
    ``"pixel"`` (voxel steps) or ``"um"`` (micrometres, converted with the
    metadata's spacing). ``mode`` is ``"planar"`` (each Z plane on its own, in Y
    and X: the legacy expansion of ``stardist_segmentation.py`` and
    ``reads_assignment.py``) or ``"volumetric"`` (3D Euclidean distance).

    Raises
    ------
    ValueError
        distance is not a finite number >= 0, or unit or mode is unknown.
    """

    distance: float
    unit: str
    mode: str

    def __post_init__(self):
        object.__setattr__(self, "distance", _nonnegative(self.distance, "distance"))
        if self.unit not in EXPANSION_UNITS:
            raise ValueError(f"unit must be pixel or um; got {self.unit!r}")
        if self.mode not in EXPANSION_MODES:
            raise ValueError(f"mode must be planar or volumetric; got {self.mode!r}")


def expand_labels(labels, metadata: ImageMetadata | None, *, config: ExpandLabelsConfig) -> tuple[np.ndarray, dict]:
    """Grow every label into the background by a distance, without overwriting or renumbering.

    scikit-image's ``expand_labels``: each background voxel within ``distance`` of
    a label takes the value of its nearest labelled voxel. ``planar`` applies it to
    each Z plane (Y and X only); ``volumetric`` to the whole volume. With
    ``unit="pixel"`` the distance is in voxel steps (spacing 1 on every axis); with
    ``unit="um"`` it is physical, and the Y, X spacing (``planar``) or the Z, Y, X
    spacing (``volumetric``) of ``metadata`` is passed to the distance transform.
    Every label keeps its voxels and its value, so each original label is contained
    in its expansion; an empty image stays empty and distance 0 changes nothing.

    Parameters
    ----------
    labels : numpy.ndarray
        Integer ZYX label image (a plane is 1×Y×X).
    metadata : ImageMetadata or None
        The labels' geometry; its ``spacing_zyx`` is required for ``unit="um"``.
    config : ExpandLabelsConfig
        Distance, unit and mode.

    Returns
    -------
    tuple[numpy.ndarray, dict]
        The C-contiguous ``uint32`` ZYX labels and the record: ``function``,
        ``config``, ``inputs`` and ``output`` (array SHA-256 values), ``mode``,
        ``unit``, ``distance``, ``spacing`` (the spacing given to the distance
        transform, Y, X for ``planar`` and Z, Y, X for ``volumetric``),
        ``physical_distance_yx`` (the distance in µm along Y and X, None when
        the metadata has no spacing) and ``voxels_added``.

    Raises
    ------
    IncompatibleGeometryError
        labels is not three-dimensional.
    ValueError
        ``unit="um"`` and the metadata has no spacing, or a label is negative or
        above 2**32 - 1.
    TypeError
        config or metadata has the wrong type, or labels is not integer.
    """
    from skimage.segmentation import expand_labels as _expand

    if not isinstance(config, ExpandLabelsConfig):
        raise TypeError("config must be an ExpandLabelsConfig")
    if metadata is not None and not isinstance(metadata, ImageMetadata):
        raise TypeError("metadata must be an ImageMetadata or None")
    array = np.asarray(labels)
    if array.ndim != 3:
        raise IncompatibleGeometryError(f"labels must be ZYX (a plane is 1×Y×X); got {array.ndim} dimensions")
    source = to_label_dtype(array)
    spacing_zyx = None if metadata is None else metadata.spacing_zyx
    if config.unit == "um" and spacing_zyx is None:
        raise ValueError("expand_labels with unit 'um' needs metadata.spacing_zyx; give the distance in pixels "
                         "on an uncalibrated grid")
    axes = slice(1, 3) if config.mode == "planar" else slice(0, 3)
    spacing = tuple(spacing_zyx[axes]) if config.unit == "um" else (1.0,) * (axes.stop - axes.start)
    if config.mode == "planar":
        output = np.stack([_expand(plane, distance=config.distance, spacing=spacing) for plane in source])
    else:
        output = _expand(source, distance=config.distance, spacing=spacing)
    output = np.ascontiguousarray(output, dtype=np.uint32)
    if spacing_zyx is None:
        physical = None
    elif config.unit == "um":
        physical = [config.distance, config.distance]
    else:
        physical = [config.distance * spacing_zyx[1], config.distance * spacing_zyx[2]]
    record = {"function": "expand_labels", "config": _json(asdict(config)), "inputs": [array_sha256(array)],
              "output": array_sha256(output), "mode": config.mode, "unit": config.unit,
              "distance": config.distance, "spacing": list(spacing), "physical_distance_yx": physical,
              "voxels_added": int(np.count_nonzero(output) - np.count_nonzero(source))}
    return output, record


@dataclass(frozen=True, kw_only=True)
class ZExtensionConfig:
    """Settings of :func:`extend_labels_through_z`, in physical units.

    ``median_um`` is the side of the square median window applied to each plane
    of the stain [µm]; ``threshold`` is ``"otsu"`` (Otsu's threshold of the
    filtered stack) or a number on the [0, 1] scale of the stain's dtype range, as
    ``skimage.util.img_as_float`` and MATLAB's ``graythresh`` give it (a uint8 grey
    level g is g / 255); a voxel is foreground when its filtered value is strictly
    greater, as MATLAB's ``imbinarize``. ``min_area_um2`` removes the objects of a
    plane's mask smaller than this area [µm²]; ``dilation_um`` is the radius of the
    disk the mask is dilated with [µm] (0: no dilation); ``fill_holes`` is
    ``"once"`` (before the area filter, the nuclei of ``create_3d_segmentation.m``)
    or ``"twice"`` (again after the dilation, its cells). Lengths are converted to
    pixels with the Y and X spacing of the metadata. No default is set for the
    sizes: the MATLAB values are in pixels and the workflow adapter translates them
    with the configured spacing.
    """

    median_um: float
    threshold: str | float = "otsu"
    min_area_um2: float
    dilation_um: float
    fill_holes: str

    def __post_init__(self):
        object.__setattr__(self, "median_um", _nonnegative(self.median_um, "median_um", positive=True))
        object.__setattr__(self, "min_area_um2", _nonnegative(self.min_area_um2, "min_area_um2"))
        object.__setattr__(self, "dilation_um", _nonnegative(self.dilation_um, "dilation_um"))
        if self.threshold != "otsu":
            if (isinstance(self.threshold, (bool, str)) or not isinstance(self.threshold, Real)
                    or not 0 <= self.threshold <= 1):
                raise ValueError("threshold must be 'otsu' or a number on the [0, 1] scale of the stain's "
                                 f"dtype range; got {self.threshold!r}")
            object.__setattr__(self, "threshold", float(self.threshold))
        if self.fill_holes not in ("once", "twice"):
            raise ValueError(f"fill_holes must be once or twice; got {self.fill_holes!r}")


def _pixels(length_um, spacing_um):
    """A length in pixels, rounded half up, at least 1."""
    return max(1, math.floor(length_um / spacing_um + 0.5))


def _disk(radius_y, radius_x):
    """Boolean ellipse footprint {(dy, dx): (dy / ry)² + (dx / rx)² ≤ 1}; radii in pixels."""
    ny, nx = math.floor(radius_y), math.floor(radius_x)
    dy, dx = np.ogrid[-ny:ny + 1, -nx:nx + 1]
    return (dy / radius_y) ** 2 + (dx / radius_x) ** 2 <= 1


def _remove_small(mask, min_pixels):
    """Drop 8-connected objects with fewer than min_pixels pixels (MATLAB's bwareaopen)."""
    if min_pixels <= 1:
        return mask
    components, _ = ndimage.label(mask, structure=np.ones((3, 3), bool))
    keep = np.bincount(components.ravel()) >= min_pixels
    keep[0] = False
    return keep[components]


def extend_labels_through_z(labels_2d, stain, metadata: ImageMetadata, *,
                            config: ZExtensionConfig) -> tuple[np.ndarray, dict]:
    """Extend plane labels through z where a stain is foreground (single-layer culture).

    The Python form of ``example/sequential_workflow/create_3d_segmentation.m`` per
    FOV, without its ``Cyto`` subtraction: median-filter each plane of the stain
    (zero padding, as ``medfilt2``); one threshold over the filtered stack on the
    [0, 1] scale of the stain's dtype range, foreground strictly greater; per plane
    fill holes, remove objects below ``min_area_um2`` (8-connected), dilate by a
    disk of ``dilation_um`` (an ellipse in pixels for anisotropic Y, X spacing),
    fill holes again for ``fill_holes="twice"``, and keep the plane labels inside
    the mask. ``FOV.segment`` records the source run and sets the geometry to
    ``extended``; there is no MATLAB parity (MATLAB's ``strel('disk')`` is an
    approximation and its median window placement for even sizes differs).

    Parameters
    ----------
    labels_2d : numpy.ndarray
        Plane label image of shape (1, Y, X), integer.
    stain : numpy.ndarray
        Finite real ZYX stain with the same Y and X.
    metadata : ImageMetadata
        The stain's geometry; ``spacing_zyx`` is required.
    config : ZExtensionConfig
        Sizes in µm and the threshold.

    Returns
    -------
    tuple[numpy.ndarray, dict]
        The ``uint32`` ZYX labels (all zero when no voxel is foreground) and the
        record: ``function``, ``config``, ``inputs`` (the plane labels' and the
        stain's SHA-256), ``output``, ``threshold`` (the value reached on the
        [0, 1] scale) and ``threshold_source`` (``otsu`` or ``config``), ``pixels``
        (median window, minimum area and dilation radii in pixels), ``geometry``
        ``extended`` and ``outcome`` (``ok`` or ``empty``).

    Raises
    ------
    IncompatibleGeometryError
        labels_2d is not (1, Y, X), or the stain is not ZYX with the same Y, X.
    ValueError
        metadata has no spacing, a label is negative, or the stain is not finite
        and real (InvalidImageError).
    TypeError
        config or metadata has the wrong type, or labels_2d is not integer.
    """
    from skimage.filters import threshold_otsu
    from skimage.util import img_as_float

    if not isinstance(config, ZExtensionConfig):
        raise TypeError("config must be a ZExtensionConfig")
    if not isinstance(metadata, ImageMetadata):
        raise TypeError("metadata must be an ImageMetadata")
    labels = np.asarray(labels_2d)
    if labels.ndim != 3 or labels.shape[0] != 1:
        raise IncompatibleGeometryError(f"labels_2d must be a plane label image of shape (1, Y, X); got {labels.shape}")
    plane = to_label_dtype(labels)[0]
    stain = np.asarray(stain)
    if stain.ndim != 3 or stain.shape[1:] != plane.shape:
        raise IncompatibleGeometryError(f"stain must be ZYX with Y, X {plane.shape}; got {stain.shape}")
    stain = _validate_image(stain, ndim=(3,))
    if metadata.spacing_zyx is None:
        raise ValueError("extend_labels_through_z needs metadata.spacing_zyx to convert µm to pixels")
    _, sy, sx = metadata.spacing_zyx
    window = (_pixels(config.median_um, sy), _pixels(config.median_um, sx))
    min_pixels = max(0, math.ceil(config.min_area_um2 / (sy * sx) - 1e-9))
    radii = (config.dilation_um / sy, config.dilation_um / sx)
    footprint = _disk(*radii) if config.dilation_um > 0 else None

    filtered = np.stack([ndimage.median_filter(stain[z], size=window, mode="constant", cval=0)
                         for z in range(stain.shape[0])])
    scaled = img_as_float(filtered)
    if config.threshold == "otsu":
        threshold, source = float(threshold_otsu(scaled)), "otsu"
    else:
        threshold, source = config.threshold, "config"
    foreground = scaled > threshold

    output = np.zeros(stain.shape, np.uint32)
    for z in range(stain.shape[0]):
        mask = ndimage.binary_fill_holes(foreground[z])
        mask = _remove_small(mask, min_pixels)
        if footprint is not None:
            mask = ndimage.binary_dilation(mask, structure=footprint)
        if config.fill_holes == "twice":
            mask = ndimage.binary_fill_holes(mask)
        output[z] = np.where(mask, plane, 0)
    record = {"function": "extend_labels_through_z", "config": _json(asdict(config)),
              "inputs": [array_sha256(labels), array_sha256(stain)], "output": array_sha256(output),
              "threshold": threshold, "threshold_source": source,
              "pixels": {"median_window_yx": list(window), "min_area": min_pixels,
                         "dilation_radius_yx": list(radii)},
              "geometry": "extended", "outcome": "ok" if output.any() else "empty"}
    return output, record
