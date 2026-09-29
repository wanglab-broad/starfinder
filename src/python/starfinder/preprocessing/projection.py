"""Z or channel projection with no implicit display rescaling."""
from dataclasses import dataclass
import numpy as np

from starfinder.image import _validate_image
from starfinder.io.conversion import ImageConversionConfig, convert_image, _dtype


@dataclass(frozen=True)
class ProjectionConfig:
    """Reduction axis and method, with optional explicit output conversion.

    Projection is an output view, not a preprocessing step. axis ``"z"`` is
    the visualization view: it reduces Z and keeps a singleton Z.
    axis ``"channel"`` is the inspection view: it merges the channels of a
    ZYXC image into ZYX, as in each FOV's reference merged image.

    Max retains dtype. Sum uses uint64/int64 for integer inputs up to 32 bits,
    float64 for floats. Integer 64-bit sums are rejected rather than overflow.
    Requested output_dtype uses checked casting; conversion is required for
    clipping or rescaling. Both cannot be specified together.
    """
    method: str = "max"
    output_dtype: str | None = None
    conversion: ImageConversionConfig | None = None
    axis: str = "z"

    def __post_init__(self):
        if self.method not in ("max", "sum"):
            raise ValueError("projection method must be max or sum")
        if self.axis not in ("z", "channel"):
            raise ValueError("projection axis must be z or channel")
        if self.output_dtype is not None:
            _dtype(self.output_dtype)
        if self.output_dtype is not None and self.conversion is not None:
            raise ValueError("choose output_dtype or conversion")


def project_image(volume: np.ndarray, *, config: ProjectionConfig = ProjectionConfig()) -> np.ndarray:
    """Reduce Z to (1,Y,X[,C]) or channels to (Z,Y,X); input remains unchanged.

    Axis ``"channel"`` requires a ZYXC image. Allocates a reduced array with
    an adequate accumulator for sum. No spatial padding. Empty/nonfinite images
    error; constants retain their max or sum. No implicit normalization or
    integer overflow. Geometry of a Z projection is separately derived with
    ImageMetadata.projected, which records collapsed-source frame; a channel
    projection keeps the spatial frame.
    """
    if config.axis == "channel":
        volume = _validate_image(volume, ndim=(4,))
        axis, keepdims = -1, False
    else:
        volume = _validate_image(volume)
        axis, keepdims = 0, True
    if config.method == "max":
        result = np.max(volume, axis=axis, keepdims=keepdims)
    else:
        if volume.dtype.kind in "ui" and volume.dtype.itemsize > 4:
            raise ValueError("64-bit integer sum projection is unsupported; convert explicitly")
        dtype = np.uint64 if volume.dtype.kind == "u" else np.int64 if volume.dtype.kind == "i" else np.float64
        result = np.sum(volume, axis=axis, keepdims=keepdims, dtype=dtype)
        if not np.isfinite(result).all():
            raise ValueError("sum projection overflow")
    conversion = config.conversion
    if config.output_dtype is not None:
        conversion = ImageConversionConfig(config.output_dtype, "cast")
    return convert_image(result, config=conversion) if conversion else result
