"""The duplicate rule: one of two observations of a cell stays retained.

design/specs/assembly.md ("Retention", rule 8) and design/specs/assembly-algorithms.md ("Duplicate
rule"). Each FOV's retained-cell territories are placed on the sample grid at scale 1 as the
label export places them (docs/sample-output-contract.md, "Assembled label export"): the grid
of ``label_grid`` at factor 1 in Y and X and the fusion grid's Z, each FOV at its origin, an
output voxel taking the value of the FOV voxel nearest its centre (index ``floor(i + ½)``)
inside the FOV's voxel-centre extent, and ``retained_image`` keeping only the retained cells.
The shared voxels of two cells are counted two FOVs at a time on the box where their extents
overlap, plane by plane, so at most two territory images are in memory.
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np

from starfinder.segmentation._operations import retained_image

#: The margin of the voxel-centre extent, in voxels.
EPS = 1e-9


def _indices(grid_origin, s, v_lo, v_hi, origin, n):
    """Grid indices ``v`` in ``[v_lo, v_hi]`` inside a FOV's voxel-centre extent, and its nearest index for each."""
    v = np.arange(v_lo, v_hi + 1, dtype=np.int64)
    i = (grid_origin + v * s - origin) / s
    inside = (i >= -EPS) & (i <= n - 1 + EPS)
    return v[inside], np.floor(i[inside] + 0.5).astype(np.int64).clip(0, n - 1)


def _span(grid_origin, s, lo, hi):
    """The grid indices of the voxel centres in ``[lo, hi]`` µm."""
    return math.ceil((lo - grid_origin) / s - EPS), math.floor((hi - grid_origin) / s + EPS)


def _index(values):
    """A slice when the indices are consecutive (a view), else the index array."""
    if len(values) and np.array_equal(values, np.arange(values[0], values[0] + len(values))):
        return slice(int(values[0]), int(values[0]) + len(values))
    return values


def _plane(territory, z, y, x):
    """Plane ``z`` of a territory at the Y and X indices given: a view when both are consecutive."""
    sy, sx = _index(y), _index(x)
    if isinstance(sy, slice) and isinstance(sx, slice):
        return territory[z, sy, sx]
    return territory[z][np.ix_(y, x)]


def _extent(fov, grid):
    """Per axis the FOV indices of its voxels on the grid (Z ``[0]`` for a 2D measure)."""
    out = []
    for a in (0, 1, 2):
        if a == 0 and grid["two_d"]:
            out.append(np.zeros(1, np.int64))
            continue
        s, o, n = grid["spacing"][a], fov["origin"][a], fov["shape"][a]
        _, index = _indices(grid["origin"][a], s, *_span(grid["origin"][a], s, o, o + (n - 1) * s), o, n)
        out.append(index)
    return out


def _common(f, g, grid):
    """Per axis the indices of FOVs f and g at the grid voxels both show, or None when they share none."""
    out = []
    for a in (0, 1, 2):
        if a == 0 and grid["two_d"]:
            out.append((np.zeros(1, np.int64), np.zeros(1, np.int64)))
            continue
        s, go = grid["spacing"][a], grid["origin"][a]
        lo = max(f["origin"][a], g["origin"][a])
        hi = min(f["origin"][a] + (f["shape"][a] - 1) * s, g["origin"][a] + (g["shape"][a] - 1) * s)
        if hi < lo - EPS * s:
            return None
        v_lo, v_hi = _span(go, s, lo, hi)
        vf, i_f = _indices(go, s, v_lo, v_hi, f["origin"][a], f["shape"][a])
        vg, i_g = _indices(go, s, v_lo, v_hi, g["origin"][a], g["shape"][a])
        common = np.intersect1d(vf, vg)
        if not len(common):
            return None
        out.append((i_f[np.searchsorted(vf, common)], i_g[np.searchsorted(vg, common)]))
    return out


def measure(fovs, load, grid):
    """The territory sizes and shared voxels of the retained cells on the sample grid.

    ``fovs`` lists, in placement table order, per FOV with a retained cell a
    mapping with ``fov_id``, ``origin`` (z, y, x µm), ``shape`` (its grid's
    ZYX) and ``cells`` (its retained ``cell_id`` values, sorted, int64);
    ``load(fov_id)`` returns its territory image (ZYX, one plane for a
    ``plane`` territory); ``grid`` holds ``origin`` and ``spacing`` (z, y, x)
    and ``two_d``. Returns ``sizes``, ``{(fov_id, cell_id): voxels}`` for every
    retained cell, and ``shared``, ``{((f, a), (g, b)): voxels}`` for every pair
    of cells of two FOVs that show one grid voxel, ``f`` before ``g`` in
    ``fovs``.
    """
    sizes, shared = {}, Counter()
    for k, f in enumerate(fovs):
        ids_f = f["cells"]
        territory = load(f["fov_id"])
        z, y, x = _extent(f, grid)
        counted = Counter()
        for zi in z.tolist():
            placed = retained_image(_plane(territory, zi, y, x), ids_f, ids_f)
            values, n = np.unique(placed[placed > 0], return_counts=True)
            counted.update(dict(zip(values.tolist(), n.tolist())))
        sizes.update({(f["fov_id"], int(c)): int(counted.get(int(c), 0)) for c in ids_f.tolist()})
        for g in fovs[k + 1:]:
            common = _common(f, g, grid)
            if common is None:
                continue
            ids_g = g["cells"]
            other = load(g["fov_id"])
            (zf, zg), (yf, yg), (xf, xg) = common
            for a, b in zip(zf.tolist(), zg.tolist()):
                left = retained_image(_plane(territory, a, yf, xf), ids_f, ids_f)
                right = retained_image(_plane(other, b, yg, xg), ids_g, ids_g)
                both = (left > 0) & (right > 0)
                if not both.any():
                    continue
                # Two uint32 cell IDs in one unsigned 64-bit key, so that every ID up to 2^32 − 1 comes back whole.
                keys = (left[both].astype(np.uint64) << np.uint64(32)) | right[both].astype(np.uint64)
                values, n = np.unique(keys, return_counts=True)
                for key, count in zip(values.tolist(), n.tolist()):
                    shared[((f["fov_id"], key >> 32), (g["fov_id"], key & 0xFFFFFFFF))] += count
            del other
        del territory
    return sizes, dict(shared)


def share(sizes, pair, voxels):
    """The shared voxels of a pair as a share of its smaller territory (0.0 when that territory is empty)."""
    smaller = min(sizes[pair[0]], sizes[pair[1]])
    return voxels / smaller if smaller else 0.0


def decide(sizes, shared, threshold, position):
    """The excluded observations: ``{(fov_id, cell_id): (kept fov_id, kept cell_id)}``.

    The retained cells are taken in the order of decreasing territory size, ties
    by placement table order (``position``) and then ``cell_id``. A cell stays
    unless it is paired, at a share of at least ``threshold`` of the smaller
    territory, with a cell of another FOV that already stays; then it is
    excluded and names the first such cell in that order. The result does not
    depend on the order of ``sizes`` or ``shared``.
    """
    partners = {}
    for pair, voxels in shared.items():
        if share(sizes, pair, voxels) >= threshold:
            a, b = pair
            partners.setdefault(a, set()).add(b)
            partners.setdefault(b, set()).add(a)

    def key(cell):
        return -sizes[cell], position[cell[0]], cell[1]

    kept, excluded = set(), {}
    for cell in sorted(sizes, key=key):
        staying = [p for p in partners.get(cell, ()) if p in kept]
        if staying:
            excluded[cell] = min(staying, key=key)
        else:
            kept.add(cell)
    return excluded
