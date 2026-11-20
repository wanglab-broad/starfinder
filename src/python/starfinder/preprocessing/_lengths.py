"""Lengths in micrometres converted to pixels with an image's spacing, and the record of a config.

Every integer length uses the rounding ``floor(x + ½)``, so 2.5 pixels give 3 (Python's
``round`` would give 2); see design/specs/segmentation-input-recipe.md ("The steps").
"""
from __future__ import annotations

import math
from dataclasses import asdict, fields
from numbers import Real

#: Field metadata of a length in micrometres that a config records only when it is set, so that a record written
#: before the field existed keeps its form (``TophatConfig.radius_um`` and its kin).
OMIT_NONE = {"record": "omit_none"}
#: The keys that set a dataset's spacing, named by every refusal of a length in µm on a grid without one.
SPACING_KEYS = "voxel_size_xy and voxel_size_z"


def config_record(config) -> dict:
    """``asdict(config)`` without the fields marked :data:`OMIT_NONE` whose value is None."""
    record = asdict(config)
    for item in fields(config):
        if item.metadata.get("record") == OMIT_NONE["record"] and record.get(item.name) is None:
            record.pop(item.name, None)
    return record


def half_up(value: float) -> int:
    """``floor(value + ½)``, the rounding of every integer length."""
    return int(math.floor(value + 0.5))


def check_length(value, name, *, positive=False):
    """A finite number (≥ 0, or > 0 with ``positive``), returned as float; ValueError naming ``name`` otherwise."""
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0 \
            or (positive and value == 0):
        raise ValueError(f"{name} must be a {'positive' if positive else 'nonnegative'} finite number of "
                         f"micrometres; got {value!r}")
    return float(value)


def check_lengths_zyx(value, name, *, positive=True):
    """One number for every axis or a ZYX 3-tuple of them; returns the number or the tuple of floats."""
    if isinstance(value, (tuple, list)):
        if len(value) != 3:
            raise ValueError(f"{name} must be a number (every axis) or a ZYX 3-tuple; got {value!r}")
        return tuple(check_length(v, name, positive=positive) for v in value)
    return check_length(value, name, positive=positive)


def spacing_of(metadata, name) -> tuple[float, float, float]:
    """The ZYX spacing of ``metadata``; ValueError naming the field ``name`` and the spacing keys without one."""
    spacing = None if metadata is None else metadata.spacing_zyx
    if spacing is None:
        raise ValueError(f"{name} is a length in micrometres and needs the image's spacing, which the grid does not "
                         f"have: set {SPACING_KEYS} (micrometres) before loading raw images, or give the length in "
                         "pixels outside a recipe")
    return tuple(float(s) for s in spacing)


def yx_pixels(value_um, metadata, name) -> int:
    """``floor(value_um / spacing_yx + ½)`` pixels in Y and X; the Y and X spacing must be equal (relative 1e-9)."""
    _, sy, sx = spacing_of(metadata, name)
    if abs(sy - sx) > 1e-9 * max(abs(sy), abs(sx)):
        raise ValueError(f"{name} is one length in Y and X and needs equal Y and X spacing; the spacing is "
                         f"{sy} (Y) and {sx} (X)")
    return half_up(float(value_um) / sy)


def _per_axis(value):
    return tuple(value) if isinstance(value, (tuple, list)) else (value,) * 3


def sigma_pixels(sigma_um, metadata, shape_zyx, name="sigma_um") -> tuple[float, float, float]:
    """``sigma_um / spacing`` per axis, 0 on an axis of length 1."""
    spacing = spacing_of(metadata, name)
    return tuple(0.0 if n == 1 else float(v) / s for v, s, n in zip(_per_axis(sigma_um), spacing, shape_zyx))


def kernel_pixels(kernel_um, metadata, shape_zyx, name="kernel_um") -> tuple[int, int, int]:
    """``max(1, floor(kernel_um / spacing + ½))`` per axis, 1 on an axis of length 1."""
    spacing = spacing_of(metadata, name)
    return tuple(1 if n == 1 else max(1, half_up(float(v) / s))
                 for v, s, n in zip(_per_axis(kernel_um), spacing, shape_zyx))
