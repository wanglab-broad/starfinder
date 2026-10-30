"""TIFF persistence with explicit selection, conversion and geometry."""
from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
from pathlib import Path
import uuid
import warnings
from xml.etree import ElementTree

import numpy as np
import tifffile

from starfinder.image import ImageMetadata, _validate_image
from starfinder.io.conversion import ImageConversionConfig, convert_image

# Namespace of the OME-XML CommentAnnotation holding ImageMetadata as JSON.
METADATA_NAMESPACE = "starfinder.image.ImageMetadata"
# Namespace of the name-based OME UUIDs: the UUID of a file is uuid5 of the SHA-256 of what is written.
_UUID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "starfinder.io.ome_uuid")


def _ome_digest(header: dict):
    """A SHA-256 of an OME-TIFF's content, started with its header (shape, dtype, layout, metadata) as JSON.

    The caller adds the pixel data in the order it writes them.
    """
    return hashlib.sha256(json.dumps(header, sort_keys=True).encode())


def _ome_uuid(digest) -> str:
    """The ``urn:uuid:`` of the content ``digest``, so that equal content gives equal bytes (not time-based)."""
    return f"urn:uuid:{uuid.uuid5(_UUID_NAMESPACE, digest.hexdigest())}"


def _content_uuid(array: np.ndarray, header: dict) -> str:
    """The OME UUID of an in-memory array written with ``header``."""
    digest = _ome_digest(dict(header, dtype=array.dtype.str, shape=list(array.shape)))
    digest.update(np.ascontiguousarray(array))
    return _ome_uuid(digest)


def _stamp_uuid(path, placeholder: str, value: str) -> None:
    """Replace the ``placeholder`` UUID of an OME-TIFF's OME-XML by ``value``, in place (same length)."""
    old, new = placeholder.encode("ascii"), value.encode("ascii")
    with tifffile.TiffFile(path) as tif:
        tag = tif.pages.first.tags["ImageDescription"]
        offset, count = tag.valueoffset, tag.count
    with open(path, "r+b") as handle:
        handle.seek(offset)
        description = handle.read(count)
        if description.count(old) != 1 or len(old) != len(new):
            raise RuntimeError(f"{path}: the OME-XML does not hold the placeholder UUID once")
        handle.seek(offset + description.index(old))
        handle.write(new)


def _writes_ome(path: Path) -> bool:
    """Whether tifffile writes OME-XML by default to ``path``: a ``*.ome.<ext>`` name."""
    name, ext = os.path.splitext(path.name.lower())
    return bool(ext) and name.endswith(".ome")


@dataclass(frozen=True)
class ImageLoadConfig:
    """Source selection and optional intensity conversion.

    Parameters
    ----------
    channel_labels : tuple of str
        Ordered channel labels. Round loading also uses them as filename patterns when ``source_paths`` is ``None``.
    source_paths : tuple of str or None
        Explicit round-source paths in channel-label order, relative to the round directory's ``subdir``; ``None`` uses filename patterns.
    subdir : str
        Subdirectory of the round directory containing the round's source files.
    source_axes : str or None
        Source axis order: ``YX``, ``ZYX``, ``ZYXC``, ``CZYX`` or ``TZCYX``; ``None`` uses the TIFF axis interpretation.
    series_index : int or None
        Nonnegative, zero-based TIFF series index; ``None`` selects an unambiguous series.
    time_index : int or None
        Nonnegative, zero-based time index; ``None`` selects an unambiguous time point.
    channel_index : int or None
        Nonnegative, zero-based source-channel index; ``None`` selects an unambiguous channel.
    crop_policy : str
        Round channel-shape policy: ``error`` or ``minimum`` cropping.
    conversion : ImageConversionConfig or None
        Intensity-conversion policy; ``None`` preserves the source dtype and intensities.
    metadata : ImageMetadata or None
        Explicit image geometry; ``None`` uses stored STARfinder metadata when present, otherwise unknown geometry.

    Notes
    -----
    Multiple series, times or channels require explicit indices. Round loading
    uses unique filename patterns in channel_labels, or explicit source_paths
    relative to the round directory's subdir. crop_policy='minimum' explicitly crops
    unequal shapes at their low-index corner and records the mapping.
    source_axes can explicitly describe legacy arrays without axis metadata.
    Geometry supplied here overrides stored metadata; absent geometry remains
    unknown. TIFF physical calibration is not inferred from resolution tags.
    """

    channel_labels: tuple[str, ...] = ("channel0",)
    source_paths: tuple[str, ...] | None = None
    subdir: str = ""
    source_axes: str | None = None
    series_index: int | None = None
    time_index: int | None = None
    channel_index: int | None = None
    crop_policy: str = "error"
    conversion: ImageConversionConfig | None = None
    metadata: ImageMetadata | None = None

    def __post_init__(self):
        labels = tuple(self.channel_labels)
        if not labels or any(not isinstance(x, str) or not x for x in labels) or len(set(labels)) != len(labels):
            raise ValueError("channel_labels must be nonempty and unique")
        object.__setattr__(self, "channel_labels", labels)
        if self.source_paths is not None:
            object.__setattr__(self, "source_paths", tuple(self.source_paths))
            if len(self.source_paths) != len(labels):
                raise ValueError("source_paths must match channel_labels")
        if self.crop_policy not in ("error", "minimum"):
            raise ValueError("crop_policy must be error or minimum")
        if self.source_axes not in (None, "YX", "ZYX", "ZYXC", "CZYX", "TZCYX"):
            raise ValueError("unsupported source_axes")
        for name in ("series_index", "time_index", "channel_index"):
            v = getattr(self, name)
            if v is not None and (isinstance(v, bool) or not isinstance(v, int) or v < 0):
                raise ValueError(f"{name} must be a nonnegative integer or None")


@dataclass(frozen=True)
class ImageLoadResult:
    """Loaded array, geometry, ordered channel/source identities and diagnostics.

    Parameters
    ----------
    image : numpy.ndarray
        Loaded finite real intensity array in ZYX or ZYXC order.
    metadata : ImageMetadata
        Spatial geometry of the loaded image.
    channel_labels : tuple of str
        Channel identities in image channel order.
    source_paths : tuple of pathlib.Path
        TIFF paths contributing to the loaded image, in channel order for round
        loading.
    diagnostics : dict
        Loading and conversion details; an empty dictionary is the default.
    """

    image: np.ndarray
    metadata: ImageMetadata
    channel_labels: tuple[str, ...]
    source_paths: tuple[Path, ...]
    diagnostics: dict = field(default_factory=dict)


def _conversion_diagnostics(image, config):
    if config is None:
        return None
    result = asdict(config)
    if config.mode == "rescale" and config.range_policy == "data":
        groups = [image[..., c] for c in range(image.shape[-1])] if image.ndim == 4 and config.scope == "per_channel" else [image]
        result["effective_input_ranges"] = [(float(g.min()), float(g.max())) for g in groups]
    return result


def _select(index, count, name):
    if index is None:
        if count != 1:
            raise ValueError(f"ambiguous {name}; select an explicit index")
        return 0
    if index >= count:
        raise ValueError(f"{name} index {index} outside size {count}")
    return index


def _local(tag):
    return tag.rsplit("}", 1)[-1]


def _stored(tif, index):
    """Saved axes/metadata: the OME-XML annotation, or tifffile's JSON description."""
    if not tif.is_ome:
        return (tif.shaped_metadata[index] if tif.shaped_metadata else None) or {}
    for element in ElementTree.fromstring(tif.ome_metadata).iter():
        if _local(element.tag) == "CommentAnnotation" and element.get("Namespace") == METADATA_NAMESPACE:
            value = next((child.text for child in element if _local(child.tag) == "Value"), None)
            if value:
                return {"starfinder_metadata": json.loads(value)}
    return {}


def _ome_zyxc(tif, path):
    """Read a one-image OME-TIFF as a contiguous ZYXC array; other axes must be singleton."""
    if len(tif.series) != 1:
        raise ValueError(f"{path} holds {len(tif.series)} OME images; expected one")
    series = tif.series[0]
    axes, shape = series.get_axes(False), series.get_shape(False)
    if any(a not in "ZCYX" and n != 1 for a, n in zip(axes, shape)) or "Y" not in axes or "X" not in axes:
        raise ValueError(f"{path} has OME axes {series.axes}; expected Z, C, Y and X only")
    data = series.asarray().reshape(tuple(n for a, n in zip(axes, shape) if a in "ZCYX"))
    axes = "".join(a for a in axes if a in "ZCYX")
    missing = "".join(a for a in "ZC" if a not in axes)
    data, axes = data.reshape(data.shape + (1,) * len(missing)), axes + missing
    return np.ascontiguousarray(data.transpose([axes.index(a) for a in "ZYXC"]))


def load_volume(path: Path | str, *, config: ImageLoadConfig = ImageLoadConfig()) -> ImageLoadResult:
    """Load one finite ZYX channel, preserving dtype by default.

    Parameters
    ----------
    path : pathlib.Path or str
        Path to the source TIFF.
    config : ImageLoadConfig
        Source selection, channel identity, geometry and intensity-conversion
        settings.

    Returns
    -------
    ImageLoadResult
        Loaded ZYX channel with geometry, channel identity, source path and
        diagnostics.

    Notes
    -----
    OME/ImageJ axes and explicitly saved axes select series/T/C, with ambiguity
    errors. Plain TIFF arrays are ZYX (YX gains singleton Z). Allocates one
    loaded volume; optional conversion adds output and float64 work buffers.
    Missing paths raise FileNotFoundError; invalid shape/selection raises
    ValueError. Returns stored STARfinder metadata or explicitly unknown fields.
    """
    if len(config.channel_labels) != 1:
        raise ValueError("load_volume requires one channel label")
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"TIFF file not found: {path}")
    with tifffile.TiffFile(path) as tif:
        index = _select(config.series_index, len(tif.series), "series")
        series = tif.series[index]
        stored = _stored(tif, index)
        data = series.asarray()
        axes = config.source_axes or series.axes
        if config.source_axes is not None and len(axes) != data.ndim:
            raise ValueError("source_axes does not match array dimensions")
        if config.source_axes is not None or tif.is_ome or tif.is_imagej or "axes" in stored:
            for axis, selected in (("T", config.time_index), ("C", config.channel_index)):
                if axis in axes:
                    pos = axes.index(axis)
                    data = np.take(data, _select(selected, data.shape[pos], axis), axis=pos)
                    axes = axes[:pos] + axes[pos + 1:]
                elif selected not in (None, 0):
                    raise ValueError(f"{axis} axis is absent")
            if axes not in ("YX", "ZYX"):
                raise ValueError(f"unsupported TIFF axes {axes}; expected ZYX")
        elif config.channel_index not in (None, 0) or config.time_index not in (None, 0):
            raise ValueError("plain ZYX TIFF has no channel/time selection")
        metadata = config.metadata or (ImageMetadata(**stored["starfinder_metadata"]) if "starfinder_metadata" in stored else ImageMetadata(str(path.resolve())))
    if data.ndim == 2:
        data = data[None, ...]
    data = _validate_image(data, ndim=(3,))
    diagnostics = {"source_dtype": str(data.dtype), "original_shape": data.shape, "series_index": index, "time_index": config.time_index, "channel_index": config.channel_index, "conversion": _conversion_diagnostics(data, config.conversion)}
    if config.conversion is not None:
        data = convert_image(data, config=config.conversion)
    diagnostics["output_dtype"] = str(data.dtype)
    diagnostics["metadata_source"] = "config" if config.metadata else "stored" if "starfinder_metadata" in stored else "unknown"
    return ImageLoadResult(data, metadata, config.channel_labels, (path,), diagnostics)


def _volume_header(path: Path | str, *, floating: bool = False) -> tuple[tuple[int, int, int] | None, bool] | None:
    """(ZYX shape, whether metadata is stored) that load_volume with the default config gives, from the header alone.

    No pixel is read. The shape follows load_volume's rules (the series'
    axes, singleton T and C dropped for an OME, ImageJ or axis-declaring file,
    any other file read as ZYX, YX gaining a singleton Z). It is None where
    the header does not settle what load_volume gives: a floating-point file,
    whose finiteness only its pixels show, and every file load_volume refuses
    (a T or C axis longer than one, other axes, not two or three dimensions,
    an empty dimension, a dtype that is not an integer). The whole result is
    None for a file of several series (load_volume asks for an index).
    With ``floating`` true a floating-point file gets its shape by the same
    rules, for a caller that needs the shape and not the pixels (the report's
    Z size); the default keeps the rule above for every other caller.
    Missing paths raise FileNotFoundError as load_volume does. The file is
    read without a buffer, so that only the bytes tifffile asks for are read:
    a buffered file reads a whole block (st_blksize, 1 MiB on an SMB share)
    at each seek to a page's header.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"TIFF file not found: {path}")
    with open(path, "rb", buffering=0) as handle, tifffile.TiffFile(handle) as tif:
        if len(tif.series) != 1:
            return None
        series = tif.series[0]
        stored = _stored(tif, 0)
        shape, axes = tuple(int(n) for n in series.shape), series.axes
        if tif.is_ome or tif.is_imagej or "axes" in stored:
            for axis in "TC":
                if axis in axes:
                    pos = axes.index(axis)
                    if shape[pos] != 1:
                        return None, "starfinder_metadata" in stored
                    shape, axes = shape[:pos] + shape[pos + 1:], axes[:pos] + axes[pos + 1:]
            if axes not in ("YX", "ZYX"):
                shape = None
        if shape is not None and len(shape) == 2:
            shape = (1, *shape)
        if shape is None or len(shape) != 3 or 0 in shape or \
                np.dtype(series.dtype).kind not in ("uif" if floating else "ui"):
            shape = None
        return shape, "starfinder_metadata" in stored


def load_round(round_dir: Path | str, *, config: ImageLoadConfig) -> ImageLoadResult:
    """Load unique ordered channel TIFFs into ZYXC.

    Parameters
    ----------
    round_dir : pathlib.Path or str
        Round directory containing the configured source subdirectory.
    config : ImageLoadConfig
        Ordered channel selection, source geometry, cropping and
        intensity-conversion settings.

    Returns
    -------
    ImageLoadResult
        Loaded ZYXC round with geometry, ordered channel labels, source paths and
        diagnostics.

    Notes
    -----
    Rejects ambiguous/missing matches, mixed source dtypes, unequal dimensions
    without explicit crop policy, and inconsistent stored geometry. No silent
    channel reordering or type promotion. Stacking allocates a new array.
    """
    directory = Path(round_dir) / config.subdir
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")
    paths = []
    for i, label in enumerate(config.channel_labels):
        if config.source_paths is not None:
            matches = [directory / config.source_paths[i]]
        else:
            matches = sorted(set(directory.glob(f"*{label}*.tif")) | set(directory.glob(f"*{label}*.tiff")))
        if len(matches) != 1:
            raise ValueError(f"expected one TIFF matching channel {label!r}; found {len(matches)}")
        paths.append(matches[0])
    if len(set(p.resolve() for p in paths)) != len(paths):
        raise ValueError("channel patterns select the same source file")
    loaded = [load_volume(path, config=ImageLoadConfig(channel_labels=(label,), source_axes=config.source_axes, series_index=config.series_index, time_index=config.time_index, channel_index=config.channel_index, metadata=config.metadata)) for path, label in zip(paths, config.channel_labels)]
    shapes = [r.image.shape for r in loaded]
    if len({r.image.dtype for r in loaded}) != 1:
        raise ValueError("channel dtypes differ; convert sources explicitly before stacking")
    # Unknown file-specific frames can be combined into the declared round frame.
    geometries = [(r.metadata.spacing_zyx, r.metadata.origin_zyx, r.metadata.direction_zyx, r.metadata.spatial_unit) for r in loaded]
    if any(g != geometries[0] for g in geometries):
        raise ValueError("channel geometry differs")
    known = any(r.diagnostics["metadata_source"] != "unknown" for r in loaded)
    if known and any(r.metadata.frame_id != loaded[0].metadata.frame_id for r in loaded):
        raise ValueError("channel frames differ")
    shape = tuple(min(s[i] for s in shapes) for i in range(3))
    cropped = len(set(shapes)) != 1
    if cropped and config.crop_policy == "error":
        raise ValueError("channel size mismatch; request crop_policy='minimum'")
    if cropped:
        warnings.warn("channel size mismatch: explicit minimum crop at origin", UserWarning)
    data = np.stack([r.image[:shape[0], :shape[1], :shape[2]] for r in loaded], axis=-1)
    metadata = config.metadata or (loaded[0].metadata if known else ImageMetadata(str(directory.resolve())))
    if cropped:
        metadata = metadata.cropped((0, 0, 0), frame_id=f"{metadata.frame_id}/crop:{shape}")
    diagnostics = {"original_shapes": shapes, "cropped": cropped, "crop_start_zyx": (0, 0, 0) if cropped else None, "source_dtype": str(data.dtype), "conversion": _conversion_diagnostics(data, config.conversion)}
    if config.conversion is not None:
        data = convert_image(data, config=config.conversion)
    diagnostics["output_dtype"] = str(data.dtype)
    return ImageLoadResult(data, metadata, config.channel_labels, tuple(paths), diagnostics)


def load_volume_zyxc(path: Path | str, *, channel_labels: tuple[str, ...] | None = None) -> ImageLoadResult:
    """Load a whole ZYXC TIFF written by save_volume in one read, preserving dtype.

    Parameters
    ----------
    path : pathlib.Path or str
        Path to the source multi-channel TIFF.
    channel_labels : tuple of str or None
        Unique labels in C-axis order; ``None`` generates ``channel0``,
        ``channel1``, and so on.

    Returns
    -------
    ImageLoadResult
        Loaded ZYXC image with geometry, ordered channel labels, source path and
        diagnostics.

    Notes
    -----
    Reads OME-TIFF (one image; axes other than Z, C, Y and X must be singleton)
    and the earlier tifffile layout that declares ZYXC axes in its JSON
    description. Singleton Z and C are kept. channel_labels must match C; None
    labels channels channel0, channel1, ... Stored STARfinder metadata is
    restored, otherwise the frame is the resolved path.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"TIFF file not found: {path}")
    with tifffile.TiffFile(path) as tif:
        stored = _stored(tif, 0)
        if tif.is_ome:
            data = _ome_zyxc(tif, path)
        elif stored.get("axes") != "ZYXC":
            raise ValueError(f"{path} does not declare ZYXC axes")
        else:
            data = tif.series[0].asarray()
    data = _validate_image(data, ndim=(4,))
    labels = tuple(f"channel{i}" for i in range(data.shape[3])) if channel_labels is None else tuple(channel_labels)
    if len(labels) != data.shape[3] or len(set(labels)) != len(labels):
        raise ValueError("channel_labels must be unique and match the channel axis")
    metadata = ImageMetadata(**stored["starfinder_metadata"]) if "starfinder_metadata" in stored else ImageMetadata(str(path.resolve()))
    diagnostics = {"source_dtype": str(data.dtype), "output_dtype": str(data.dtype),
                   "metadata_source": "stored" if "starfinder_metadata" in stored else "unknown"}
    return ImageLoadResult(data, metadata, labels, (path,), diagnostics)


def save_volume(image: np.ndarray, path: Path | str, compress: bool = False, *, metadata: ImageMetadata | None = None, conversion: ImageConversionConfig | None = None) -> None:
    """Write finite ZYX TIFF or ZYXC OME-TIFF with explicit axes and optional geometry.

    Parameters
    ----------
    image : numpy.ndarray
        Finite real intensity array in ZYX or ZYXC order.
    path : pathlib.Path or str
        Destination TIFF path.
    compress : bool
        Whether to apply zlib compression.
    metadata : ImageMetadata or None
        Geometry to store with the image; ``None`` omits STARfinder geometry
        metadata.
    conversion : ImageConversionConfig or None
        Intensity-conversion settings; ``None`` preserves the input dtype and
        intensities.

    Notes
    -----
    ZYX is a tifffile TIFF with a JSON description. ZYXC is OME-TIFF: every
    page is one YX plane (DimensionOrder XYCZT), the OME-XML declares SizeZ,
    SizeC and the pixel type, and metadata is a JSON CommentAnnotation. Name
    ZYXC files ``*.ome.tif`` so Bio-Formats (Fiji) opens them as a Z×C
    hyperstack. Preserves dtype, including float64, unless conversion is
    supplied. Overwrites an existing file; creates parents. No input mutation.
    ZYXC can be read one channel at a time using an explicit channel_index in
    load_volume, or whole with load_volume_zyxc. The UUID of the OME-XML is a
    name-based UUID of the pixel data and the metadata written, so the same
    array and metadata give the same bytes (also for ZYX on a ``*.ome.*`` name,
    which tifffile writes as OME-TIFF).
    """
    image = _validate_image(image)
    if conversion is not None:
        image = convert_image(image, config=conversion)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    compression = "zlib" if compress else None
    if image.ndim == 3:
        info = {"axes": "ZYX"}
        if metadata is not None:
            info["starfinder_metadata"] = asdict(metadata)
        if _writes_ome(path):
            info["UUID"] = _content_uuid(image, dict(info, compression=compression))
        tifffile.imwrite(path, image, compression=compression, photometric="minisblack", metadata=info)
        return
    info = {"axes": "ZCYX"}
    if metadata is not None:
        info["CommentAnnotation"] = {"Namespace": METADATA_NAMESPACE, "Value": json.dumps(asdict(metadata))}
    info["UUID"] = _content_uuid(image, dict(info, compression=compression))
    tifffile.imwrite(path, np.moveaxis(image, 3, 1), ome=True, compression=compression,
                     photometric="minisblack", metadata=info)
