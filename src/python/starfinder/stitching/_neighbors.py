"""Neighbour pairs from nominal overlap (design/specs/stitching-algorithms.md, "Neighbour derivation")."""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real

import numpy as np

from starfinder.image import IncompatibleGeometryError
from starfinder.segmentation._labels import ReferenceGrid

from ._positions import TilePositionTable, _names


@dataclass(frozen=True)
class NeighborPair:
    """Two FOVs whose nominal boxes overlap, ``fov_a`` before ``fov_b`` in table order.

    ``overlap_zyx`` is the nominal overlap box, ``((z0, z1), (y0, y1), (x0,
    x1))`` in µm: the intersection of the two boxes without the tolerance (for a
    pair found only through a positive tolerance, an end may precede its start).
    ``index_a`` and ``index_b`` are the two FOVs' rows in the table, which are
    their positions in the tiles a method receives.
    """

    fov_a: str
    fov_b: str
    overlap_zyx: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    index_a: int
    index_b: int


def boxes(table: TilePositionTable, grids: Mapping[str, ReferenceGrid]) -> tuple[np.ndarray, np.ndarray]:
    """``(lo, hi)``, each ``(n, 3)``: the nominal box of each FOV, ``nominal`` to ``nominal + shape × spacing``."""
    ids = table.fov_ids
    missing = [fov for fov in ids if fov not in grids]
    if missing:
        raise ValueError(f"FOVs of the table without a grid: {_names(missing)}")
    unknown = [fov for fov in ids if grids[fov].metadata.spacing_zyx is None]
    if unknown:
        raise IncompatibleGeometryError(f"neighbour boxes need known spacing; FOVs {_names(unknown)} have none")
    lo = table.nominal_zyx()
    extent = np.array([[n * s for n, s in zip(grids[fov].shape_zyx, grids[fov].metadata.spacing_zyx)] for fov in ids],
                      dtype=np.float64).reshape(len(ids), 3)
    return lo, lo + extent


def neighbor_pairs(table: TilePositionTable, grids: Mapping[str, ReferenceGrid], *,
                   tolerance_um: float = 0.0) -> list[NeighborPair]:
    """Every pair of FOVs whose nominal boxes overlap, diagonals included.

    The box of FOV ``i`` spans ``nominal_i`` to ``nominal_i + shape_i ×
    spacing_i`` per axis, from its reference grid. ``(a, b)``, ``a`` before
    ``b`` in table order, is a pair when the boxes, each grown by
    ``tolerance_um`` on each side, intersect in Y and X with positive length
    and, when both grids have Z > 1, in Z with positive length. Row and column
    indices are never read, so a stage file, a regular grid and a supplied
    table give pairs by one rule.

    Parameters
    ----------
    table : TilePositionTable
        The nominal positions.
    grids : Mapping[str, ReferenceGrid]
        The reference grid of every FOV of the table.
    tolerance_um : float
        Growth per side, µm (default 0.0); finite and ≥ 0.

    Returns
    -------
    list[NeighborPair]
        Ordered by ``fov_a``, then ``fov_b``, in table order, with the nominal
        overlap box of each.

    Raises
    ------
    ValueError
        A negative or non-finite tolerance, or a FOV without a grid.
    IncompatibleGeometryError
        A grid with unknown spacing.
    """
    if isinstance(tolerance_um, (bool, np.bool_)) or not isinstance(tolerance_um, Real) \
            or not math.isfinite(tolerance_um) or tolerance_um < 0:
        raise ValueError(f"tolerance_um must be a finite number ≥ 0; got {tolerance_um!r}")
    ids = table.fov_ids
    lo, hi = boxes(table, grids)
    volume = np.array([grids[fov].shape_zyx[0] > 1 for fov in ids])
    pairs = []
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            start = np.maximum(lo[a], lo[b])
            end = np.minimum(hi[a], hi[b])
            grown = end - start + 2 * float(tolerance_um)
            axes = (1, 2, 0) if volume[a] and volume[b] else (1, 2)
            if all(grown[k] > 0 for k in axes):
                box = tuple((float(s), float(e)) for s, e in zip(start, end))
                pairs.append(NeighborPair(ids[a], ids[b], box, a, b))
    return pairs
