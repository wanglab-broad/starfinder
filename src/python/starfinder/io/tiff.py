"""TIFF persistence with explicit selection, conversion and geometry."""
from dataclasses import asdict, dataclass, field
from pathlib import Path
import warnings

import numpy as np
import tifffile

from starfinder.image import ImageMetadata, _validate_image
from starfinder.io.conversion import ImageConversionConfig, convert_image


@dataclass(frozen=True)
class ImageLoadConfig:
    """Source selection and optional intensity conversion.

    Multiple series, times or channels require explicit indices. Round loading
    uses unique filename patterns in channel_labels, or explicit source_paths
    relative to the round directory. crop_policy='minimum' explicitly crops
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
    """Loaded array, geometry, ordered channel/source identities and diagnostics."""

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


def load_volume(path: Path | str, *, config: ImageLoadConfig = ImageLoadConfig()) -> ImageLoadResult:
    """Load one finite ZYX channel, preserving dtype by default.

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
        stored = tif.shaped_metadata[index] if tif.shaped_metadata else {}
        stored = stored or {}
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


def load_round(round_dir: Path | str, *, config: ImageLoadConfig) -> ImageLoadResult:
    """Load unique ordered channel TIFFs into ZYXC.

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


def save_volume(image: np.ndarray, path: Path | str, compress: bool = False, *, metadata: ImageMetadata | None = None, conversion: ImageConversionConfig | None = None) -> None:
    """Write finite ZYX/ZYXC TIFF with explicit axes and optional geometry.

    Preserves dtype unless conversion is supplied. Overwrites an existing file;
    creates parents. No input mutation. ZYXC can be read one channel at a time
    using an explicit channel_index in load_volume.
    """
    image = _validate_image(image)
    if conversion is not None:
        image = convert_image(image, config=conversion)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    info = {"axes": "ZYX" if image.ndim == 3 else "ZYXC"}
    if metadata is not None:
        info["starfinder_metadata"] = asdict(metadata)
    tifffile.imwrite(path, image, compression="zlib" if compress else None, photometric="minisblack", metadata=info)
