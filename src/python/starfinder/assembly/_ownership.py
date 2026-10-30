"""Ownership regions computed from the placement (design/specs/assembly.md, "Ownership regions").

O1 (``nearest_center``): a point is owned by the owning FOV whose footprint contains it and
whose footprint centre is nearest in Y and X; an exact tie goes to the FOV first in the
placement's table order (T1). O2 (``midpoint_box``): per owning FOV an axis-aligned box whose
edge towards each grid neighbour is the middle of their overlap. Only owning FOVs (placed,
with an input or a grid, ``image_supported`` or ``metadata_only``) own points; a
``placeholder`` keeps its footprint for the record and owns nothing. Z never decides an
owner (Z1).
"""
from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from starfinder.image import ImageMetadata
from starfinder.segmentation import ReferenceGrid
from starfinder.stitching import PlacementResult
from starfinder.stitching._placement import OWNING, PLACED

from ._config import AssemblyConfig

GEOMETRY_COLUMNS = ("fov_id", "placement_status", "footprint_y0", "footprint_y1", "footprint_x0", "footprint_x1",
                    "centre_y", "centre_x", "owns_points", "candidate", "box_y0", "box_y1", "box_x0", "box_x1",
                    "box_y1_closed", "box_x1_closed")


def recorded_grid(placement: PlacementResult, fov_id: str) -> tuple[tuple[int, int, int], ImageMetadata] | None:
    """The grid shape and metadata ``record["inputs"]`` holds for a FOV, or None."""
    entry = (placement.record.get("inputs") or {}).get(fov_id)
    if not isinstance(entry, Mapping) or not isinstance(entry.get("grid"), Mapping):
        return None
    grid = entry["grid"]
    return tuple(int(n) for n in grid["shape_zyx"]), ImageMetadata(**grid["metadata"])


def footprint_yx(origin_yx, shape_yx, spacing_yx):
    """Half-open footprint ``(y0, y1, x0, x1)`` and centre ``(y, x)`` of a placed grid, µm.

    Per axis ``[o − s/2, o + (n − ½)·s)``, the union of the voxel supports, and
    the centre ``o + (n − 1)/2 · s`` of the middle voxel position.
    """
    (oy, ox), (ny, nx), (sy, sx) = origin_yx, shape_yx, spacing_yx
    box = (oy - sy / 2, oy + (ny - 0.5) * sy, ox - sx / 2, ox + (nx - 0.5) * sx)
    return box, (oy + (ny - 1) / 2 * sy, ox + (nx - 1) / 2 * sx)


def grid_cells(placement: PlacementResult) -> dict[str, tuple[int, int]] | None:
    """FOV → ``(grid_row, grid_column)`` of the tile position table the placement records, or None.

    Read from a BigStitcher import's view map (``record["method"]["views"]``),
    or recomputed from the nominal positions and the grid step a table of
    ``positions_from_grid`` or a snapped stage file records (``step_yx_um``):
    the nominal positions of those tables are ``row × step_y`` and
    ``column × step_x``. A supplied table and an unsnapped stage file have no
    cells.
    """
    method = placement.record.get("method") or {}
    views = method.get("views") if method.get("method") == "bigstitcher" else None
    if views:
        return {v["fov_id"]: (int(v["grid_row"]), int(v["grid_column"])) for v in views if v.get("fov_id")}
    table = placement.record.get("table") or {}
    step = None
    if table.get("converter") == "positions_from_grid":
        step = table.get("step_yx_um")
    elif isinstance(table.get("snap_to_grid"), Mapping):
        step = table["snap_to_grid"].get("step_yx_um")
    if not step:
        return None
    fovs = placement.fovs
    rows = np.floor(fovs["nominal_y"].to_numpy(np.float64) / float(step[0]) + 0.5).astype(np.int64)
    columns = np.floor(fovs["nominal_x"].to_numpy(np.float64) / float(step[1]) + 0.5).astype(np.int64)
    return {fov: (int(r), int(c)) for fov, r, c in zip(fovs["fov_id"], rows, columns)}


@dataclass(frozen=True, eq=False)
class OwnershipMap:
    """The ownership regions of one placement, a point test.

    Parameters
    ----------
    sample : str
        Identifier of the sample whose placement supplies the regions.
    geometry : pandas.DataFrame
        Per-FOV ownership geometry with footprints and centres in micrometres.
    config : AssemblyConfig
        Ownership and tie settings used for point tests.

    Notes
    -----
    ``geometry`` has one row per FOV of the placement, in table order:
    ``fov_id``, ``placement_status``, the half-open footprint
    ``footprint_y0/y1/x0/x1`` and centre ``centre_y/x`` (µm, NaN for a FOV that
    is not placed), ``owns_points`` (an owning FOV: placed, with a grid,
    ``image_supported`` or ``metadata_only``), ``candidate`` (it may own a
    point under ``candidates``) and, under ``midpoint_box``, the box
    ``box_y0/y1/x0/x1`` with ``box_y1_closed`` and ``box_x1_closed`` (an upper
    edge set at a neighbour's midpoint that includes the point on it, under
    ``tie="table_order"``). Build it with :func:`ownership_map`.
    """

    sample: str
    geometry: pd.DataFrame
    config: AssemblyConfig

    @property
    def owning(self) -> tuple[str, ...]:
        """The owning FOVs, in table order."""
        return tuple(self.geometry["fov_id"][self.geometry["owns_points"].to_numpy(dtype=bool)])

    @property
    def candidates(self) -> tuple[str, ...]:
        """The FOVs that may own a point, in table order."""
        return tuple(self.geometry["fov_id"][self.geometry["candidate"].to_numpy(dtype=bool)])

    def owner(self, points_zyx) -> np.ndarray:
        """The owning ``fov_id`` of each sample-frame point, or None where no candidate owns it.

        Parameters
        ----------
        points_zyx : array-like
            Sample-frame points in micrometres, in ZYX order, as shape ``(m, 3)`` or a single
            shape ``(3,)`` point.

        Returns
        -------
        numpy.ndarray
            One-dimensional object array of owning FOV identifiers, with ``None`` for a point
            without an owner.

        Notes
        -----
        ``points_zyx`` is ``(m, 3)`` (µm, ZYX); only Y and X are read. A point
        with a non-finite Y or X has no owner. Returns an object array of
        length ``m``.
        """
        points = np.asarray(points_zyx, dtype=np.float64)
        if points.ndim == 1:
            points = points[None, :]
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"points_zyx must have shape (m, 3); got {points.shape}")
        q_y, q_x = points[:, 1], points[:, 2]
        rows = self.geometry[self.geometry["candidate"].to_numpy(dtype=bool)]
        best = np.full(len(points), -1, dtype=np.int64)
        with np.errstate(invalid="ignore"):
            if self.config.ownership == "nearest_center":
                distance = np.full(len(points), np.inf)
                for k, row in enumerate(rows.itertuples(index=False)):
                    inside = ((q_y >= row.footprint_y0) & (q_y < row.footprint_y1) & (q_x >= row.footprint_x0)
                              & (q_x < row.footprint_x1))
                    squared = (q_y - row.centre_y) ** 2 + (q_x - row.centre_x) ** 2
                    nearer = inside & (squared < distance)   # strict: an exact tie keeps the earlier FOV (T1)
                    distance[nearer] = squared[nearer]
                    best[nearer] = k
            else:
                for k, row in enumerate(rows.itertuples(index=False)):
                    upper_y = (q_y <= row.box_y1) if row.box_y1_closed else (q_y < row.box_y1)
                    upper_x = (q_x <= row.box_x1) if row.box_x1_closed else (q_x < row.box_x1)
                    inside = (q_y >= row.box_y0) & upper_y & (q_x >= row.box_x0) & upper_x
                    first = inside & (best < 0)
                    best[first] = k
        ids = np.array(list(rows["fov_id"]) + [None], dtype=object)
        return ids[np.where(best < 0, len(rows), best)]


def _candidate_set(candidates, placement, owning, grids):
    if isinstance(candidates, (str, bytes)) or not isinstance(candidates, Collection):
        raise ValueError(f"candidates must be a collection of fov_id, or None; got {candidates!r}")
    candidates = list(candidates)
    if not candidates:
        raise ValueError("candidates must name at least one owning FOV; got an empty collection")
    statuses = dict(zip(placement.fovs["fov_id"], placement.fovs["status"]))
    reasons = []
    for fov in candidates:
        if fov in owning:
            continue
        if fov not in statuses:
            reasons.append(f"{fov} (not in the placement)")
        elif statuses[fov] == "placeholder":
            reasons.append(f"{fov} (a placeholder)")
        elif statuses[fov] not in PLACED:
            reasons.append(f"{fov} (not placed: {statuses[fov]})")
        elif fov not in grids:
            reasons.append(f"{fov} (no grid in grids)")
        else:
            reasons.append(f"{fov} (status {statuses[fov]})")
    if reasons:
        raise ValueError(f"candidates must be owning FOVs of the placement of sample {placement.sample!r}; "
                         f"not owning: {', '.join(reasons)}")
    return set(candidates)


def ownership_map(placement: PlacementResult, grids: Mapping[str, ReferenceGrid], *, config: AssemblyConfig,
                  candidates: Collection[str] | None = None) -> OwnershipMap:
    """The ownership regions of a placement, from its origins and each placed FOV's grid shape and spacing.

    Owning FOVs are the FOVs of status ``image_supported`` or
    ``metadata_only`` that have a grid in ``grids`` (:func:`assemble` passes
    the grids of its inputs). Every placed FOV, placeholders included, has a
    footprint and a centre (from ``grids`` or the grid the placement records);
    only owning FOVs own points. ``candidates`` restricts the FOVs that may own
    a point; None (default) means every owning FOV, which is what
    :func:`assemble` uses. Under ``midpoint_box`` the boxes are built among the
    owning candidates from the grid cells the placement records.

    Parameters
    ----------
    placement : PlacementResult
        Any placement.
    grids : Mapping[str, ReferenceGrid]
        The FOVs' reference grids.
    config : AssemblyConfig
        ``ownership`` and ``tie`` are read.
    candidates : Collection[str], optional
        A nonempty collection of owning FOVs.

    Returns
    -------
    OwnershipMap

    Raises
    ------
    TypeError
        placement is not a PlacementResult, grids not a mapping of
        ReferenceGrid or config not an AssemblyConfig.
    ValueError
        No owning FOV (also when every FOV is a ``placeholder``);
        ``candidates`` empty or naming a FOV that is not an owning FOV (not in
        the placement, not placed, a placeholder, or without a grid), naming
        it; ``midpoint_box`` on a placement without grid cells.
    """
    if not isinstance(placement, PlacementResult):
        raise TypeError("placement must be a PlacementResult")
    if not isinstance(config, AssemblyConfig):
        raise TypeError("config must be an AssemblyConfig")
    if not isinstance(grids, Mapping) or not all(isinstance(g, ReferenceGrid) for g in grids.values()):
        raise TypeError("grids must map fov_id to ReferenceGrid")
    fovs = placement.fovs
    ids = list(fovs["fov_id"])
    statuses = list(fovs["status"])
    owning = [fov for fov, status in zip(ids, statuses) if status in OWNING and fov in grids]
    if not owning:
        raise ValueError(f"no FOV of the placement of sample {placement.sample!r} owns a point: an owning FOV is "
                         "image_supported or metadata_only with an input; statuses "
                         f"{', '.join(f'{f}: {s}' for f, s in zip(ids, statuses))}")
    allowed = set(owning) if candidates is None else _candidate_set(candidates, placement, set(owning), grids)

    shape_spacing = {}
    for fov, status in zip(ids, statuses):
        if status not in PLACED:
            continue
        if fov in grids:
            shape_spacing[fov] = (grids[fov].shape_zyx, grids[fov].metadata.spacing_zyx)
        else:
            recorded = recorded_grid(placement, fov)
            if recorded is not None and recorded[1].spacing_zyx is not None:
                shape_spacing[fov] = (recorded[0], recorded[1].spacing_zyx)
    nan = float("nan")
    rows = []
    for row in fovs.itertuples(index=False):
        fov = row.fov_id
        if fov in shape_spacing:
            shape, spacing = shape_spacing[fov]
            box, centre = footprint_yx((row.origin_y, row.origin_x), shape[1:], spacing[1:])
        else:
            box, centre = (nan, nan, nan, nan), (nan, nan)
        rows.append({"fov_id": fov, "placement_status": row.status, "footprint_y0": box[0], "footprint_y1": box[1],
                     "footprint_x0": box[2], "footprint_x1": box[3], "centre_y": centre[0], "centre_x": centre[1],
                     "owns_points": fov in owning, "candidate": fov in allowed,
                     "box_y0": nan, "box_y1": nan, "box_x0": nan, "box_x1": nan,
                     "box_y1_closed": False, "box_x1_closed": False})
    if config.ownership == "midpoint_box":
        cells = grid_cells(placement)
        if cells is None:
            raise ValueError(f"ownership='midpoint_box' needs the grid cells of the tile position table; the "
                             f"placement of sample {placement.sample!r} has none (use nearest_center)")
        origin = {r.fov_id: (r.origin_y, r.origin_x) for r in fovs.itertuples(index=False)}
        at = {cells[fov]: fov for fov in ids if fov in allowed and fov in cells}
        closed = config.tie == "table_order"
        for entry in rows:
            fov = entry["fov_id"]
            if fov not in allowed:
                continue
            if fov not in cells:
                raise ValueError(f"ownership='midpoint_box' needs a grid cell for every owning FOV; {fov} has none")
            (r, c), (oy, ox) = cells[fov], origin[fov]
            (ny, nx), (sy, sx) = shape_spacing[fov][0][1:], shape_spacing[fov][1][1:]
            y0, y1, x0, x1 = entry["footprint_y0"], entry["footprint_y1"], entry["footprint_x0"], entry["footprint_x1"]
            if (r, c - 1) in at:
                left = at[(r, c - 1)]
                x0 = (origin[left][1] + shape_spacing[left][0][2] * sx + ox) / 2 - sx / 2
            if (r, c + 1) in at:
                x1 = (ox + nx * sx + origin[at[(r, c + 1)]][1]) / 2 - sx / 2
                entry["box_x1_closed"] = closed
            if (r - 1, c) in at:
                up = at[(r - 1, c)]
                y0 = (origin[up][0] + shape_spacing[up][0][1] * sy + oy) / 2 - sy / 2
            if (r + 1, c) in at:
                y1 = (oy + ny * sy + origin[at[(r + 1, c)]][0]) / 2 - sy / 2
                entry["box_y1_closed"] = closed
            entry.update(box_y0=y0, box_y1=y1, box_x0=x0, box_x1=x1)
    geometry = pd.DataFrame(rows, columns=list(GEOMETRY_COLUMNS)).astype(
        {"fov_id": "string", "placement_status": "string", "owns_points": bool, "candidate": bool,
         "box_y1_closed": bool, "box_x1_closed": bool})
    return OwnershipMap(placement.sample, geometry, config)

