"""Singleton-Z projection with no implicit display rescaling."""
from dataclasses import dataclass
import numpy as np

from starfinder.image import _validate_image
from starfinder.io.conversion import ImageConversionConfig, convert_image, _dtype


@dataclass(frozen=True)
class ProjectionConfig:
    """Z reduction and optional explicit output conversion.

    Max retains dtype. Sum uses uint64/int64 for integer inputs up to 32 bits,
    float64 for floats. Integer 64-bit sums are rejected rather than overflow.
    Requested output_dtype uses checked casting; conversion is required for
    clipping or rescaling. Both cannot be specified together.
    """
    method: str = "max"
    output_dtype: str | None = None
    conversion: ImageConversionConfig | None = None

    def __post_init__(self):
        if self.method not in ("max", "sum"):
            raise ValueError("projection method must be max or sum")
        if self.output_dtype is not None:
            _dtype(self.output_dtype)
        if self.output_dtype is not None and self.conversion is not None:
            raise ValueError("choose output_dtype or conversion")


def project_image(volume: np.ndarray, *, config: ProjectionConfig = ProjectionConfig()) -> np.ndarray:
    """Reduce Z and retain shape (1,Y,X[,C]); input remains unchanged.

    Allocates a reduced array with an adequate accumulator for sum. No spatial
    padding. Empty/nonfinite images error; constants retain their max or sum.
    No implicit normalization or integer overflow. Geometry is separately
    derived with ImageMetadata.projected, which records collapsed-source frame.
    """
    volume = _validate_image(volume)
    if config.method == "max":
        result = np.max(volume, axis=0, keepdims=True)
    else:
        if volume.dtype.kind in "ui" and volume.dtype.itemsize > 4:
            raise ValueError("64-bit integer sum projection is unsupported; convert explicitly")
        dtype = np.uint64 if volume.dtype.kind == "u" else np.int64 if volume.dtype.kind == "i" else np.float64
        result = np.sum(volume, axis=0, keepdims=True, dtype=dtype)
        if not np.isfinite(result).all():
            raise ValueError("sum projection overflow")
    conversion = config.conversion
    if config.output_dtype is not None:
        conversion = ImageConversionConfig(config.output_dtype, "cast")
    return convert_image(result, config=conversion) if conversion else result
