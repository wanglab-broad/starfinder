"""Explicit intensity conversion, without implicit display normalization."""
from dataclasses import dataclass

import numpy as np

from starfinder.image import _validate_image


def _range(value, name):
    if value is None:
        return
    if len(value) != 2 or not np.isfinite(value).all() or value[0] >= value[1]:
        raise ValueError(f"{name} must be a finite increasing pair")


def _dtype(dtype):
    dtype = np.dtype(dtype)
    if dtype.kind not in "uif":
        raise ValueError("output_dtype must be a real numeric dtype")
    return dtype


def _cast(values, dtype, rounding):
    dtype = _dtype(dtype)
    original_low, original_high = values.min().item(), values.max().item()
    if dtype.kind in "ui":
        if values.dtype.kind == "f":
            values = np.rint(values) if rounding == "nearest_even" else np.trunc(values)
        limits = np.iinfo(dtype)
    else:
        limits = np.finfo(dtype)
    # Python integer extrema avoid uint64 -> float64 promotion at 2**64.
    low, high = values.min().item(), values.max().item()
    if not np.isfinite(values).all() or low < limits.min or high > limits.max or original_low < limits.min or original_high > limits.max:
        raise ValueError("output range is not representable; request explicit clip or rescale")
    result = values.astype(dtype)
    if not np.isfinite(result).all():
        raise ValueError("conversion produced nonfinite values")
    return result


@dataclass(frozen=True)
class ImageConversionConfig:
    """Explicit cast, clip or linear rescale policy.

    Rescale requires input_range or range_policy='data', and output_range.
    Constant data-derived ranges map to output_range's lower endpoint. Clip
    requires output_range. Cast rejects range loss. Integer rounding is
    nearest_even or truncate. Scope is global or per_channel (ZYXC only).
    Conversion allocates an output plus float64 work per active channel/group.
    """

    output_dtype: str
    mode: str
    input_range: tuple[float, float] | None = None
    output_range: tuple[float, float] | None = None
    range_policy: str = "declared"
    scope: str = "global"
    rounding: str = "nearest_even"

    def __post_init__(self):
        dtype = _dtype(self.output_dtype)
        object.__setattr__(self, "output_dtype", dtype.name)
        if self.mode not in ("cast", "clip", "rescale") or self.scope not in ("global", "per_channel") or self.rounding not in ("nearest_even", "truncate"):
            raise ValueError("invalid conversion mode, scope or rounding")
        if self.range_policy not in ("declared", "data"):
            raise ValueError("range_policy must be declared or data")
        _range(self.input_range, "input_range")
        _range(self.output_range, "output_range")
        if self.mode == "rescale":
            if self.output_range is None or (self.input_range is None and self.range_policy != "data"):
                raise ValueError("rescale requires input range policy and output_range")
            if self.range_policy == "data" and self.input_range is not None:
                raise ValueError("choose input_range or data range, not both")
        elif self.range_policy != "declared" or self.input_range is not None:
            raise ValueError("input range policy is only used for rescale")
        if self.mode == "clip" and self.output_range is None:
            raise ValueError("clip requires output_range")
        if self.mode == "cast" and self.output_range is not None:
            raise ValueError("cast does not use output_range")
        if self.output_range is not None:
            _cast(np.asarray(self.output_range), dtype, self.rounding)


def convert_image(image, *, config: ImageConversionConfig):
    """Convert finite ZYX/ZYXC data without mutating the input.

    Declared rescale rejects values outside input_range; clip is explicit.
    Floating calculations use float64; integer outputs use configured rounding.
    """
    image = _validate_image(image)
    if config.mode == "cast":
        return _cast(image, config.output_dtype, config.rounding)
    result = np.empty(image.shape, dtype=config.output_dtype)
    groups = range(image.shape[-1]) if image.ndim == 4 and config.scope == "per_channel" else [None]
    for c in groups:
        key = (..., c) if c is not None else (...,)
        work = image[key].astype(np.float64)
        low, high = config.output_range
        if config.mode == "clip":
            work = np.clip(work, low, high)
        else:
            a, b = config.input_range if config.range_policy == "declared" else (work.min(), work.max())
            if np.any(work < a) or np.any(work > b):
                raise ValueError("image outside declared input_range")
            work = np.full_like(work, low) if a == b else np.clip((work - a) / (b - a) * (high - low) + low, low, high)
        result[key] = _cast(work, config.output_dtype, config.rounding)
    return result
