"""Aligned sample rasters with conservative, label-preserving resolution.

This module prepares arrays only. It neither assembles source grids nor assigns
molecules/expression to cells. See ``starfinder.sample_export/1``.
"""
from copy import deepcopy
from dataclasses import dataclass, replace
from itertools import product
from typing import Mapping

import numpy as np
from scipy.spatial import cKDTree

from starfinder.image import ImageMetadata, IncompatibleGeometryError, _validate_image
from starfinder.io import ImageLoadResult

__all__ = [
    "RasterConfig", "RasterLevel", "RasterPreparationError", "RasterResult",
    "prepare_rasters", "remap_labels",
]


class RasterPreparationError(ValueError):
    """Invalid native input or resource failure; no complete result is returned.

    ``diagnostics`` retains the rejected grid trials when available.
    """

    def __init__(self, message, *, diagnostics=None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}


@dataclass(frozen=True)
class RasterConfig:
    """Absolute power-of-two factors and optional additional pyramid levels.

    ``coordinate_space`` is explicitly ``physical`` or ``index``. Index inputs
    must have all physical fields unknown. ``max_output_bytes`` bounds the sum
    of emitted image/mask array bytes, including native fallback, not process RSS.
    Images/channels are selected explicitly in the input mapping before this call.
    """

    factors_zyx: tuple[int, int, int] = (1, 1, 1)
    coarser_levels: int = 0
    coordinate_space: str = "physical"
    max_output_bytes: int | None = None

    def __post_init__(self):
        f = tuple(self.factors_zyx)
        if len(f) != 3 or any(type(v) is not int or v < 1 or v & (v - 1) for v in f):
            raise ValueError("factors_zyx must be three positive integer powers of two")
        object.__setattr__(self, "factors_zyx", f)
        if type(self.coarser_levels) is not int or self.coarser_levels < 0:
            raise ValueError("coarser_levels must be a nonnegative integer")
        if self.coordinate_space not in ("physical", "index"):
            raise ValueError("coordinate_space must be physical or index")
        if self.max_output_bytes is not None and (
            type(self.max_output_bytes) is not int or self.max_output_bytes < 1
        ):
            raise ValueError("max_output_bytes must be a positive integer or None")


@dataclass(frozen=True)
class RasterLevel:
    """One common image/mask grid; arrays remain ZYXC/ZYX.

    ``index_to_source_zyx`` maps output centers to native centers, not world
    coordinates. ``metadata`` already includes that offset in physical space;
    consumers must not apply it twice. Diagnostics use source-index units.
    """

    images: dict[str, ImageLoadResult]
    labels: np.ndarray
    metadata: ImageMetadata
    factors_zyx: tuple[int, int, int]
    index_to_source_zyx: np.ndarray
    diagnostics: dict


@dataclass(frozen=True)
class RasterResult:
    """Accepted levels, every rejected trial, and unchanged source geometry."""

    levels: tuple[RasterLevel, ...]
    source_metadata: ImageMetadata
    config: RasterConfig
    diagnostics: dict


def _labels(value):
    a = np.asarray(value)
    if a.ndim != 3 or any(n == 0 for n in a.shape) or a.dtype.kind not in "ui" or np.any(a < 0):
        raise RasterPreparationError("labels must be nonempty nonnegative integer ZYX, not Boolean")
    return a


def remap_labels(masks: Mapping[str, np.ndarray], *, cell_keys: Mapping[tuple, tuple]):
    """Remap disjoint, already aligned masks to uint32 and a reversible row map.

    Keys are ``(namespace, local_label)``; values are explicit
    ``(cell_namespace, cell_id)`` pairs of nonempty strings. Sort source keys by
    namespace then integer label. Reject overlaps, missing/extra keys and reused
    global cell keys. No biological equivalence or grid assembly is inferred.
    Returns ``(labels, rows)``; each row records both keys and ``instance_id``.
    """
    if not masks or any(not isinstance(k, str) or not k for k in masks):
        raise RasterPreparationError("nonempty named masks required")
    arrays = {k: _labels(v) for k, v in masks.items()}
    shape = next(iter(arrays.values())).shape
    if any(a.shape != shape for a in arrays.values()):
        raise RasterPreparationError("remapping requires already aligned identical shapes")
    keys = sorted((ns, int(v)) for ns, a in arrays.items() for v in np.unique(a) if v)
    if set(keys) != set(cell_keys):
        raise RasterPreparationError("cell map must exactly cover native positive labels")
    cells = [cell_keys[k] for k in keys]
    if any(not isinstance(c, tuple) or len(c) != 2 or any(not isinstance(v, str) or not v for v in c) for c in cells):
        raise RasterPreparationError("cell keys must be pairs of nonempty strings")
    if len(set(cells)) != len(cells):
        raise RasterPreparationError("reused global cell key requires upstream biological reconciliation")
    if len(keys) > np.iinfo(np.uint32).max:
        raise RasterPreparationError("too many instances for uint32")
    out = np.zeros(shape, dtype=np.uint32)
    rows = []
    for instance_id, (ns, local) in enumerate(keys, 1):
        support = arrays[ns] == local
        if np.any(out[support]):
            raise RasterPreparationError("overlapping positive masks require upstream reconciliation")
        out[support] = instance_id
        rows.append(dict(mask_namespace=ns, local_label=local,
                         cell_namespace=cell_keys[(ns, local)][0],
                         cell_id=cell_keys[(ns, local)][1], instance_id=instance_id))
    return out, rows


def _blocks(a, factors):
    z, y, x = (n // f for n, f in zip(a.shape[:3], factors))
    fz, fy, fx = factors
    tail = a.shape[3:]
    return a.reshape(z, fz, y, fy, x, fx, *tail).transpose(
        0, 2, 4, 1, 3, 5, *range(6, 6 + len(tail))
    ).reshape(z, y, x, fz * fy * fx, *tail)


def _trial(native, factors, native_points):
    shape = native.shape
    record = dict(factors_zyx=list(factors), failed_checks=[], accepted=False)
    if any(n % f for n, f in zip(shape, factors)):
        record["failed_checks"].append("indivisible_grid")
        return None, record
    blocks = _blocks(native, factors)
    reduced = np.empty(blocks.shape[:3], dtype=native.dtype)
    mixed = tied = 0
    for index in np.ndindex(reduced.shape):
        ids, counts = np.unique(blocks[index], return_counts=True)
        reduced[index] = ids[np.argmax(counts)]  # sorted IDs: smallest wins ties
        mixed += int(np.count_nonzero(ids) > 1)
        tied += int(np.count_nonzero(counts == counts.max()) > 1)
    expected = set(native_points)
    observed = {int(v) for v in np.unique(reduced) if v}
    missing = sorted(expected - observed)
    record.update(shape_zyx=list(reduced.shape), mixed_positive_blocks=mixed,
                  tied_blocks=tied, missing_labels=missing, new_labels=sorted(observed - expected),
                  native_label_count=len(expected), output_label_count=len(observed))
    if missing or observed - expected:
        record["failed_checks"].append("label_set")
    f = np.asarray(factors)
    offset = (f - 1) / 2
    limit = float(np.linalg.norm(f - 1))
    record["centroid_bound"] = limit / 2
    record["hausdorff_bound"] = limit
    record["cells"] = []
    for label, points in native_points.items():
        out_indices = np.argwhere(reduced == label)
        row = dict(label=label, native_voxels=len(points), output_voxels=len(out_indices),
                   centroid_displacement=None, hausdorff_distance=None, min_support_fraction=None)
        if len(out_indices):
            mapped = out_indices * f + offset
            centroid = float(np.linalg.norm(points.mean(axis=0) - mapped.mean(axis=0)))
            hausdorff = max(float(cKDTree(points).query(mapped)[0].max()),
                            float(cKDTree(mapped).query(points)[0].max()))
            support = float(np.min(np.mean(blocks[tuple(out_indices.T)] == label, axis=-1)))
            row.update(centroid_displacement=centroid, hausdorff_distance=hausdorff,
                       min_support_fraction=support)
            for failed, ok in (("support", support > 0), ("centroid", centroid <= limit / 2 + 1e-9),
                               ("hausdorff", hausdorff <= limit + 1e-9)):
                if not ok and failed not in record["failed_checks"]:
                    record["failed_checks"].append(failed)
        record["cells"].append(row)
    # Compare all support corners in native index units, not fabricated units.
    corners = np.asarray(list(product(*[(-.5, n - .5) for n in reduced.shape])))
    expected_corners = np.asarray(list(product(*[(-.5, n - .5) for n in shape])))
    error = float(np.max(np.abs(corners * f + offset - expected_corners)))
    record["extent_max_error"] = error
    if error > 1e-9:
        record["failed_checks"].append("extent")
    record["accepted"] = not record["failed_checks"]
    return reduced, record


def prepare_rasters(images: Mapping[str, ImageLoadResult], labels: np.ndarray, *,
                    metadata: ImageMetadata, declared_ids, config: RasterConfig) -> RasterResult:
    """Prepare selected aligned images and labels without changing source arrays.

    Every image must have identical native geometry/extent and explicit ordered
    channel labels. Source paths and diagnostics (including upstream transform
    history) are copied to every output. ``declared_ids`` must exactly match the
    native positive labels; absence is a hard error. Use :func:`remap_labels`
    separately when local namespaces require export IDs. No spatial transform,
    cropping, projection, expression reassignment or mask merging is performed.

    Failed base trials halve every factor above one. Optional coarser trials
    double non-singleton native axes and stop at their first failure. All levels
    are computed directly from native inputs, including intensity block means.
    """
    labels = _labels(labels)
    ids = tuple(declared_ids)
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or v < 1 for v in ids) or len(set(ids)) != len(ids):
        raise RasterPreparationError("declared_ids must be unique positive integers")
    actual = {int(v) for v in np.unique(labels) if v}
    if actual != set(ids):
        raise RasterPreparationError("declared/native label mismatch", diagnostics={
            "missing_native_labels": sorted(set(ids) - actual),
            "undeclared_native_labels": sorted(actual - set(ids)), "native_shape_zyx": list(labels.shape)})
    physical = (metadata.spacing_zyx, metadata.origin_zyx, metadata.direction_zyx, metadata.spatial_unit)
    if config.coordinate_space == "physical":
        metadata._require_physical()
        if not np.array_equal(metadata.direction_zyx, np.eye(3)):
            raise IncompatibleGeometryError("rotated target frames require upstream regridding")
    elif any(v is not None for v in physical):
        raise IncompatibleGeometryError("explicit index frame requires unknown physical fields")
    if labels.shape[0] == 1 and config.factors_zyx[0] != 1:
        raise RasterPreparationError("singleton Z requires fz=1; projection is separate")
    if not images or any(not isinstance(name, str) or not name for name in images):
        raise RasterPreparationError("select at least one named image")
    for name, source in images.items():
        a = _validate_image(source.image, ndim=(4,))
        if a.shape[:3] != labels.shape or source.metadata != metadata:
            raise IncompatibleGeometryError(f"{name}: source grids must already be aligned")
        channels = source.channel_labels
        if len(channels) != a.shape[3] or len(set(channels)) != len(channels) or any(not isinstance(c, str) or not c for c in channels):
            raise RasterPreparationError(f"{name}: unique ordered channel labels required")
    points = {v: np.argwhere(labels == v) for v in sorted(actual)}
    diagnostics = dict(requested_factors_zyx=list(config.factors_zyx), trials=[],
                       coordinate_space=config.coordinate_space,
                       calibration="known" if config.coordinate_space == "physical" else "unknown",
                       source_dtypes={k: str(v.image.dtype) for k, v in images.items()},
                       label_dtype=str(labels.dtype), output_bytes=0, pyramid_stop_reason=None)
    levels = []
    factors = config.factors_zyx
    while True:
        mask, trial = _trial(labels, factors, points)
        trial["role"] = "base" if not levels else "coarser"
        diagnostics["trials"].append(trial)
        if not trial["accepted"]:
            if levels:
                diagnostics["pyramid_stop_reason"] = "coarser_level_not_preservable"
                break
            if factors == (1, 1, 1):
                raise RasterPreparationError("native fidelity failure", diagnostics=diagnostics)
            factors = tuple(max(1, f // 2) for f in factors)
            continue
        native = factors == (1, 1, 1)
        dtypes = {k: v.image.dtype if native else np.dtype(
            "float64" if v.image.dtype.kind == "f" and v.image.dtype.itemsize == 8 else "float32"
        ) for k, v in images.items()}
        required = mask.nbytes + sum(mask.size * images[k].image.shape[3] * d.itemsize for k, d in dtypes.items())
        if config.max_output_bytes is not None and diagnostics["output_bytes"] + required > config.max_output_bytes:
            diagnostics.update(required_level_bytes=required, required_shape_zyx=list(mask.shape))
            raise RasterPreparationError("output byte budget exceeded, including required fallback", diagnostics=diagnostics)
        f = np.asarray(factors)
        offset = (f - 1) / 2
        grid = metadata if config.coordinate_space == "index" else replace(
            metadata, spacing_zyx=tuple(np.asarray(metadata.spacing_zyx) * f),
            origin_zyx=tuple(np.asarray(metadata.origin_zyx) + np.asarray(metadata.spacing_zyx) * offset))
        transform = np.eye(4)
        transform[:3, :3] = np.diag(f)
        transform[:3, 3] = offset
        prepared = {}
        for name, source in images.items():
            a = source.image.copy() if native else _blocks(source.image, factors).mean(axis=3, dtype=np.float64).astype(dtypes[name])
            if not np.isfinite(a).all():
                raise RasterPreparationError("nonfinite intensity after block mean", diagnostics=diagnostics)
            prepared[name] = replace(source, image=a, metadata=grid, diagnostics=deepcopy(source.diagnostics))
        trial["output_dtypes"] = {k: str(v.image.dtype) for k, v in prepared.items()}
        trial["output_bytes"] = required
        levels.append(RasterLevel(prepared, mask, grid, factors, transform, trial))
        diagnostics["output_bytes"] += required
        if len(levels) == 1:
            diagnostics["achieved_factors_zyx"] = list(factors)
            diagnostics["fallback"] = factors != config.factors_zyx
        if len(levels) > config.coarser_levels:
            break
        next_factors = tuple(f * 2 if n > 1 else f for f, n in zip(factors, labels.shape))
        if next_factors == factors:
            diagnostics["pyramid_stop_reason"] = "no_coarser_grid"
            break
        factors = next_factors
    return RasterResult(tuple(levels), metadata, config, diagnostics)
