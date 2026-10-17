"""Augmentation, layout and truth labels of the real-appearance scene (docs/synthetic-real-appearance.md).

Bank entries are placed on the accepted acquisition layout (``_sample._layout``: the FOV grid, the scene domain Ω,
the covered region K and the FOV jitter) by PL2 with its anchor rule, one proposal at a time, until DN3's count or
DN1's visible-stain target is reached, or ``max_proposals`` proposals or ``max_refusals`` refusals in a row
("When placement stops"). Each proposal draws its entry from BS2's queue (a permutation of the bank from
``context.appearance.entry``), its transform ``A_k = S_k R_k F_k`` from ``context.appearance.augment`` (AU1) and its
centre from ``context.appearance.placement``, each at its own entity ``object-<k>-proposal-<j>``, with ``k`` the
object it would become and ``j`` the proposal's number in the scene, both from 1.

A placed object ``k`` holds ``σ`` where its entry's signed distance ``D_e`` ("Terms"), interpolated linearly at
``T_k⁻¹ σ`` in the entry's voxel index, is at most 0 (IN2); outside the entry box (on any axis of more than one voxel)
it holds nothing and its distance is infinite. ``δ_k(σ) = m_k · D_e(T_k⁻¹ σ)``. One placement order rules both labels:
in the dense contexts ``N(σ)`` is the first object that holds ``σ`` and ``C(σ) = N(σ)`` there, elsewhere CT1 (the
object of smallest ``δ_k``, ties to the smaller index, if that ``δ_k`` is at most its growth ``g_k``, else 0); in the
culture ``C(σ)`` is the first cell that holds ``σ`` and ``N(σ)`` that cell where its nucleus ``D_e`` is at most 0
(CU1). A 2D entry is a column through the planes (TZ1). The placement tests (the share of a mask in Ω, the overlap
with earlier masks, a culture nucleus against earlier nuclei) are decided once on the sample grid; the anchor test is
a point test.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from ._appearance_bank import AppearanceBank
from ._common import _draw, _integer
from ._context import ContextSceneConfig, TerritoryConfig, _direction, _unit
from ._formed import _STREAM_NAMESPACE
from ._sample import _layout, _real, _streams

# The stream components of the layout ("Random streams"): entity and draws of each.
_LAYOUT_COMPONENTS = {
    "context.appearance.entry": ("bank", "one permutation of the bank (BS2); the next permutation of the same "
                                         "generator each time the bank is exhausted"),
    "context.appearance.augment": ("object-<k>-proposal-<j>", "five uniforms: θ, the Y flip, the Z flip, the scale, "
                                                              "the gain"),
    "context.appearance.placement": ("object-<k>-proposal-<j>", "the contact uniform; for a contact proposal the "
                                                                "partner index and the direction (two uniforms on "
                                                                "the sphere, one on the circle); else z, y, x"),
    "context.appearance.growth": ("cell-<k>", "one value of CT1's growth"),
}
# PL2's constants (docs/synthetic-real-appearance.md, "Layout"; proto.py::place_all), compared on voxel counts in
# whole percent: refused below 35 % of the mask in Ω, or above 1 % of it on earlier masks.
_OVERLAP_MAX = 1
_DOMAIN_SHARE = 35
_WIDEN = 0.3
_LADDER = tuple(0.5 + 0.04 * m for m in range(23))  # 0.5, 0.54, ..., 1.38: below 1.4, as np.arange(0.5, 1.4, 0.04)
# Bounding boxes are widened by this much (µm) so that a voxel centre on a box corner is never culled by rounding.
_PAD = 1e-6


def _range(value, name):
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError(f"{name} must be a (low, high) tuple")
    low, high = (_real(v, name) for v in value)
    if not 0 < low <= high:
        raise ValueError(f"{name} must satisfy 0 < low ≤ high; got {value!r}")
    return low, high


@dataclass(frozen=True)
class AugmentationConfig:
    """The augmentation of a placed entry (AU1 of docs/synthetic-real-appearance.md).

    Every proposal draws a rotation ``θ`` about Z uniform in ``[0, 2π)``, a flip in Y with
    probability ``flip_y``, in dense 3D a flip in Z with probability ``flip_z`` (none in X), a
    scale uniform in ``scale`` and a gain uniform in ``gain``. The scale is isotropic for dense
    3D entries and acts in Y and X only for culture cells and 2D tissue entries, which keep
    their Z. Ranges are ``(low, high)`` with ``0 < low ≤ high``; ValueError names the field
    otherwise. Instances are not hashable.
    """

    __hash__ = None

    scale: tuple[float, float] = (0.9, 1.1)
    gain: tuple[float, float] = (0.9, 1.1)
    flip_y: float = 0.5
    flip_z: float = 0.5

    def __post_init__(self):
        _range(self.scale, "scale")
        _range(self.gain, "gain")
        _unit(self.flip_y, "flip_y")
        _unit(self.flip_z, "flip_z")


@dataclass(frozen=True)
class LayoutConfig:
    """Placement (PL2), density (DN1 or DN3) and the stop of the appearance scene.

    ``density="visible"`` (DN1) places until the composed stain's visible-stain share, with
    the fixed ``threshold`` relative to the plateau, reaches ``target``, checked after every
    ``check_every`` placements; ``target`` and ``threshold`` None take the medians over the
    bank's source images. ``density="count"`` (DN3) places ``count`` objects. A
    proposal is made against a placed object with probability ``contact``. Placement ends at
    the count or the target, at ``max_proposals`` proposals in total or after
    ``max_refusals`` refused proposals in a row; a reached count or share below ``floor`` of
    its aim raises ValueError naming both ("When placement stops"). Instances are not
    hashable.
    """

    __hash__ = None

    density: str = "visible"
    count: int | None = None
    target: float | None = None
    threshold: float | None = None
    check_every: int = 1
    contact: float = 0.75
    max_proposals: int = 40_000
    max_refusals: int = 4_000
    floor: float = 0.9

    def __post_init__(self):
        if self.density not in ("count", "visible"):
            raise ValueError(f"density must be 'count' (DN3) or 'visible' (DN1); got {self.density!r}")
        if self.density == "count":
            _integer(self.count, "count", 1)
            if self.target is not None or self.threshold is not None:
                raise ValueError("target and threshold belong to density='visible' (DN1); give count only")
        else:
            if self.count is not None:
                raise ValueError("count belongs to density='count' (DN3); give target and threshold, or None")
            if self.target is not None:
                target = _real(self.target, "target", strict=True)
                if not target < 1:
                    raise ValueError(f"target must satisfy 0 < target < 1; got {target}")
            if self.threshold is not None:
                _real(self.threshold, "threshold", strict=True)
        _integer(self.check_every, "check_every", 1)
        _unit(self.contact, "contact")
        _integer(self.max_proposals, "max_proposals", 1)
        _integer(self.max_refusals, "max_refusals", 1)
        floor = _real(self.floor, "floor", strict=True)
        if floor > 1:
            raise ValueError(f"floor must satisfy 0 < floor ≤ 1; got {floor}")


# ------------------------------------------------------------------------------------------------ entries


def _bank_kind(bank):
    """``cell`` (culture cells, CU1), ``column`` (2D entries, TZ1) or ``nucleus`` (dense 3D nuclei, CT1)."""
    if bank.record["entry"] == "cell":
        return "cell"
    flat = bank.entries.shape_z.to_numpy() == 1
    if flat.all():
        return "column"
    if flat.any():
        raise ValueError(f"bank {bank.name}: its nucleus entries mix 2D and 3D boxes")
    return "nucleus"


class _Entries:
    """Float64 copies of the entries' signed distances, made once per entry."""

    def __init__(self, bank):
        self.bank = bank
        self.kind = _bank_kind(bank)
        self._distance, self._nucleus, self._core, self._deepest = {}, {}, {}, {}
        table = bank.entries
        self.shapes = table[["shape_z", "shape_y", "shape_x"]].to_numpy(dtype=np.int64)
        self.spacings = table[["spacing_z", "spacing_y", "spacing_x"]].to_numpy(dtype=np.float64)
        self.origins = table[["box_origin_z", "box_origin_y", "box_origin_x"]].to_numpy(dtype=np.int64)
        self.volumes = table.volume.to_numpy(dtype=np.float64)

    def distance(self, row):
        if row not in self._distance:
            self._distance[row] = np.asarray(self.bank.distance(row), dtype=np.float64)
        return self._distance[row]

    def core(self, row):
        """The index box (lo, hi) of the entry's mask voxels grown by one voxel and clipped to its box: linear
        interpolation of ``D_e`` reaches 0 only within one voxel of a voxel where ``D_e ≤ 0``."""
        if row not in self._core:
            where = np.nonzero(self.distance(row) <= 0)
            shape = self.shapes[row]
            self._core[row] = (np.array([max(int(w.min()) - 1, 0) for w in where], dtype=np.float64),
                               np.array([min(int(w.max()) + 1, n - 1) for w, n in zip(where, shape)],
                                        dtype=np.float64))
        return self._core[row]

    def deepest(self, row):
        """The voxel index of the entry's smallest ``D_e``; among equal values (a culture cell through every plane has
        the same ``D_e`` in each) the one nearest the box centre ``x̄_e``, by the squared distance from its index offset
        (mirror-image voxels tie exactly), then the first in C order."""
        if row not in self._deepest:
            D = self.distance(row)
            where = np.argwhere(D == D.min()).astype(np.float64)
            centre = (self.shapes[row] - 1) / 2
            offset = (where - centre) * self.spacings[row]
            self._deepest[row] = where[int(np.argmin((offset * offset).sum(axis=1)))]
        return self._deepest[row]

    def nucleus(self, row):
        if self.kind != "cell":
            return None
        if row not in self._nucleus:
            self._nucleus[row] = np.asarray(self.bank.nucleus_distance(row), dtype=np.float64)
        return self._nucleus[row]

    def spacing(self, row):
        return self.spacings[row]

    def shape(self, row):
        return self.shapes[row]


def _matrix(theta, flip_z, flip_y, scale_z, scale_yx):
    """``A = S R F`` and its inverse ``F Rᵀ S⁻¹``: F = diag(f_z, f_y, 1), R the rotation by θ in the Y-X plane, S =
    diag(s_z, s, s) ("Augmentation", the transform)."""
    c, s = math.cos(theta), math.sin(theta)
    F = np.diag([float(flip_z), float(flip_y), 1.0])
    R = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    S = np.diag([float(scale_z), float(scale_yx), float(scale_yx)])
    inverse = F @ R.T @ np.diag([1.0 / float(scale_z), 1.0 / float(scale_yx), 1.0 / float(scale_yx)])
    return S @ R @ F, inverse


class _Object:
    """A placed (or proposed) object: its entry, transform ``T(x) = c + A (x − x̄_e)`` and bounding boxes."""

    def __init__(self, k, entries, row, centre, theta, flip_z, flip_y, scale_z, scale_yx, gain):
        self.k, self.row = int(k), int(row)
        self.n = entries.shape(row)
        self.s_e = entries.spacing(row)
        self.xbar = (self.n - 1) / 2 * self.s_e
        self.D = entries.distance(row)
        self.ND = entries.nucleus(row)
        self.deepest = entries.deepest(row)
        self.theta, self.flip_z, self.flip_y = float(theta), int(flip_z), int(flip_y)
        self.scale_z, self.scale_yx, self.gain = float(scale_z), float(scale_yx), float(gain)
        self.m = self.scale_yx
        self.A, self.Ainv = _matrix(self.theta, self.flip_z, self.flip_y, self.scale_z, self.scale_yx)
        self._box = self._terms(np.zeros(3), self.n - 1.0)
        self._core = self._terms(*entries.core(row))
        self.move(centre)

    def _terms(self, first, last):
        """min and max over the corners of the entry index box [first, last] of ``A (x − x̄_e)``."""
        corners = np.array([[a, b, d] for a in (first[0], last[0]) for b in (first[1], last[1])
                            for d in (first[2], last[2])]) * self.s_e - self.xbar
        A = self.A
        terms = np.stack([corners[:, 0] * A[a, 0] + corners[:, 1] * A[a, 1] + corners[:, 2] * A[a, 2]
                          for a in range(3)], axis=1)
        return terms.min(axis=0), terms.max(axis=0)

    def move(self, centre):
        """Set the centre ``c`` and the bounding boxes of the whole entry box and of its mask core (a column in Z for
        a 2D entry, TZ1); returns the object."""
        self.c = np.asarray(centre, dtype=np.float64).copy()
        for name, (low, high) in (("", self._box), ("core_", self._core)):
            lo, hi = self.c + low - _PAD, self.c + high + _PAD
            if self.n[0] == 1:
                lo[0], hi[0] = -np.inf, np.inf
            setattr(self, name + "lo", lo)
            setattr(self, name + "hi", hi)
        return self

    def forward(self, x):
        """``T(x)`` for entry-frame points ``x`` (µm, (P, 3))."""
        d = np.asarray(x, dtype=np.float64) - self.xbar
        A = self.A
        return np.stack([self.c[a] + d[:, 0] * A[a, 0] + d[:, 1] * A[a, 1] + d[:, 2] * A[a, 2] for a in range(3)],
                        axis=1)

    def index(self, points):
        """The entry voxel index ``T⁻¹(σ) / s_e`` of sample positions (P, 3), and whether each lies in the box (an
        axis of one voxel is a column: its index is 0 everywhere)."""
        d = points - self.c
        B = self.Ainv
        q = np.empty_like(d)
        for a in range(3):
            q[:, a] = (self.xbar[a] + d[:, 0] * B[a, 0] + d[:, 1] * B[a, 1] + d[:, 2] * B[a, 2]) / self.s_e[a]
        inside = np.ones(len(points), dtype=bool)
        for a in range(3):
            if self.n[a] == 1:
                q[:, a] = 0.0
            else:
                inside &= (q[:, a] >= 0) & (q[:, a] <= self.n[a] - 1)
        return q, inside

    def values(self, points, array):
        """``array`` (on the entry box) interpolated linearly at ``T⁻¹(σ)`` of (P, 3) points; +inf outside the box."""
        q, inside = self.index(points)
        out = np.full(len(points), np.inf)
        if inside.any():
            out[inside] = ndi.map_coordinates(array, q[inside].T, order=1, mode="nearest", prefilter=False)
        return out

    def grid_values(self, coords, array):
        """``values`` on the outer grid of the 1-D sample coordinates ``coords`` (z, y, x), with the same arithmetic
        per point; +inf outside the box."""
        shape = tuple(len(v) for v in coords)
        t = [coords[0][:, None, None] - self.c[0], coords[1][None, :, None] - self.c[1],
             coords[2][None, None, :] - self.c[2]]
        B = self.Ainv
        q, inside = [], np.ones(shape, dtype=bool)
        for a in range(3):
            if self.n[a] == 1:
                q.append(np.zeros(shape))
                continue
            qa = np.broadcast_to((self.xbar[a] + t[0] * B[a, 0] + t[1] * B[a, 1] + t[2] * B[a, 2]) / self.s_e[a],
                                 shape)
            inside &= (qa >= 0) & (qa <= self.n[a] - 1)
            q.append(qa)
        out = np.full(shape, np.inf)
        if inside.any():
            out[inside] = ndi.map_coordinates(array, [qa[inside] for qa in q], order=1, mode="nearest",
                                              prefilter=False)
        return out

    def select(self, points):
        """The rows of (P, 3) points inside the object's bounding box."""
        return np.flatnonzero(np.all((points >= self.lo) & (points <= self.hi), axis=1))

    def anchor(self):
        """The transformed position of the entry's deepest point, the voxel of smallest ``D_e`` (PL2's anchor)."""
        return self.forward((self.deepest * self.s_e)[None])[0]


def _window(lo, hi, origin, spacing, shape):
    """The index range [first, last] per axis of a grid's voxel centres within [lo, hi] (µm), clipped to the grid;
    None when empty."""
    first, last = [], []
    for a in range(3):
        start = 0 if not np.isfinite(lo[a]) else max(int(math.floor((lo[a] - origin[a]) / spacing[a])), 0)
        stop = (shape[a] - 1 if not np.isfinite(hi[a])
                else min(int(math.ceil((hi[a] - origin[a]) / spacing[a])), shape[a] - 1))
        if stop < start:
            return None
        first.append(start), last.append(stop)
    return first, last


def _prefilter_sigma(target_spacing, scale_zyx, entry_spacing, entry_shape):
    """IN2's anti-aliasing prefilter in source voxels per axis: ``max(0, t_a / (m_a s_e,a) − 1) / 2``, where the target
    spacing ``t_a`` is coarser than the scaled source spacing (``skimage.transform.resize``'s rule); 0 along an axis
    of one voxel. It depends on the entry, its scale and the target spacing only, never on a FOV's offset."""
    sigma = []
    for t, m, s, n in zip(target_spacing, scale_zyx, entry_spacing, entry_shape):
        sigma.append(0.0 if n == 1 else max(0.0, float(t) / (float(m) * float(s)) - 1.0) / 2.0)
    return tuple(sigma)


def _prefiltered(array, sigma):
    """An entry array after IN2's prefilter (unchanged where every σ is 0)."""
    array = np.asarray(array, dtype=np.float64)
    if not any(v > 0 for v in sigma):
        return array
    return ndi.gaussian_filter(array, sigma, mode="nearest")


# ------------------------------------------------------------------------------------------------ labels


class _Labels:
    """The point evaluator of ``N``, ``C`` and ``δ_k`` for placed objects in placement order (one ownership rule)."""

    def __init__(self, kind, objects, growth=None):
        self.kind = kind
        self.objects = list(objects)
        if kind != "cell":
            if growth is None or len(growth) != len(self.objects):
                raise ValueError("CT1 needs one growth value per placed object")
            self.growth = np.asarray(growth, dtype=np.float64)
        else:
            self.growth = None

    def _run(self, shape, parts):
        """(C, N) on positions of ``shape``; ``parts(obj)`` gives (where, values(array)) for the positions in obj's
        box: ``where`` indexes the label arrays and ``values(array)`` interpolates an entry array there."""
        N = np.zeros(shape, dtype=np.uint32)
        if self.kind == "cell":
            C = np.zeros(shape, dtype=np.uint32)
            for obj in self.objects:
                part = parts(obj)
                if part is None:
                    continue
                where, values = part
                new = (values(obj.D) <= 0) & (C[where] == 0)
                if not new.any():
                    continue
                cell, nucleus = C[where], N[where]
                cell[new] = obj.k
                nucleus[new & (values(obj.ND, new) <= 0)] = obj.k
                C[where], N[where] = cell, nucleus
            return C, N
        best = np.full(shape, np.inf)
        owner = np.zeros(shape, dtype=np.int64)
        for i, obj in enumerate(self.objects):
            part = parts(obj)
            if part is None:
                continue
            where, values = part
            D = values(obj.D)
            nucleus = N[where]
            nucleus[(D <= 0) & (nucleus == 0)] = obj.k
            N[where] = nucleus
            delta = obj.m * D
            near, who = best[where], owner[where]
            better = delta < near
            near[better] = delta[better]
            who[better] = i
            best[where], owner[where] = near, who
        C = N.copy()
        ks = np.array([obj.k for obj in self.objects], dtype=np.uint32)
        free = (N == 0) & np.isfinite(best)
        if free.any():
            near = owner[free]
            hit = best[free] <= self.growth[near]
            cell = C[free]
            cell[hit] = ks[near[hit]]
            C[free] = cell
        return C, N

    def labels(self, points):
        """(C, N) at (..., 3) sample positions (µm): the point evaluator."""
        points = np.asarray(points, dtype=np.float64)
        flat = points.reshape(-1, 3)

        def parts(obj):
            rows = obj.select(flat)
            if not len(rows):
                return None
            mine = flat[rows]

            def values(array, chosen=None):
                if chosen is None:
                    return obj.values(mine, array)
                out = np.full(len(rows), np.inf)
                out[chosen] = obj.values(mine[chosen], array)
                return out
            return rows, values
        C, N = self._run(len(flat), parts)
        return C.reshape(points.shape[:-1]), N.reshape(points.shape[:-1])

    def delta(self, points):
        """``δ_k`` (µm) of every object at (..., 3) sample positions: (..., K), +inf outside an object's box."""
        points = np.asarray(points, dtype=np.float64)
        flat = points.reshape(-1, 3)
        out = np.full((len(flat), len(self.objects)), np.inf)
        for i, obj in enumerate(self.objects):
            rows = obj.select(flat)
            if len(rows):
                out[rows, i] = obj.m * obj.values(flat[rows], obj.D)
        return out.reshape(points.shape[:-1] + (len(self.objects),))

    def grid(self, origin, spacing, shape):
        """(C, N) at the voxel centres ``origin + p ⊙ spacing`` of a grid of ``shape`` (ZYX), each object evaluated on
        the part of the grid its box covers, with the point evaluator's arithmetic."""
        origin = np.asarray(origin, dtype=np.float64)
        spacing = np.asarray(spacing, dtype=np.float64)
        shape = tuple(int(v) for v in shape)

        def parts(obj):
            window = _window(obj.lo, obj.hi, origin, spacing, shape)
            if window is None:
                return None
            first, last = window
            where = tuple(slice(start, stop + 1) for start, stop in zip(first, last))
            coords = [origin[a] + np.arange(first[a], last[a] + 1) * spacing[a] for a in range(3)]

            def values(array, chosen=None):
                return obj.grid_values(coords, array)
            return where, values
        return self._run(shape, parts)

    def axes(self, coords):
        """(C, N) on the outer grid of the increasing 1-D sample coordinates ``coords`` (z, y, x), each object
        evaluated at the coordinates within its box, with the point evaluator's arithmetic; equal to ``grid`` on the
        same voxel centres."""
        coords = [np.asarray(v, dtype=np.float64) for v in coords]
        shape = tuple(len(v) for v in coords)

        def parts(obj):
            window = _axes_window(obj.lo, obj.hi, coords)
            if window is None:
                return None
            where = tuple(slice(start, stop) for start, stop in window)
            sub = [v[start:stop] for v, (start, stop) in zip(coords, window)]

            def values(array, chosen=None):
                return obj.grid_values(sub, array)
            return where, values
        return self._run(shape, parts)


def _axes_window(lo, hi, coords):
    """The index range [start, stop) per axis of the increasing coordinates ``coords`` within [lo, hi]; None when
    empty."""
    window = []
    for a in range(3):
        start = int(np.searchsorted(coords[a], lo[a], side="left"))
        stop = int(np.searchsorted(coords[a], hi[a], side="right"))
        if stop <= start:
            return None
        window.append((start, stop))
    return window


# ------------------------------------------------------------------------------------------------ the layout


def _acquisition(context, stream):
    """The accepted acquisition layout of a context configuration: ``_layout`` (Ω grown by a morphology round's
    bound), the FOV true origins with the J1 jitter (``sample.jitter``, as ``_sample._generate`` draws it) and the
    morphology round's translation per FOV in µm (MT1, ``context.morphology.translation``, as
    ``_context._stain_part`` draws it), None without a morphology round."""
    acquisition = context.sample.acquisition
    morphology = context.morphology
    grow = None if morphology is None else np.asarray(morphology.translation_max_zyx, dtype=np.float64)
    layout = _layout(acquisition, grow)
    jitter = np.empty((len(layout.ids), 3), dtype=np.float64)
    for k, fov in enumerate(layout.ids):
        if acquisition.jitter == "uniform":
            jitter[k] = layout.bound * (2 * stream("sample.jitter", fov).random(3) - 1) + 0.0
        else:
            jitter[k] = np.asarray(acquisition.jitter_zyx[fov], dtype=np.float64) + 0.0
    true = layout.nominal + jitter * layout.s
    shifts = None
    if morphology is not None:
        bound = np.asarray(morphology.translation_max_zyx, dtype=np.float64)
        shifts = {}
        for fov in layout.ids:
            if morphology.translation == "uniform":
                vox = bound * (2 * stream("context.morphology.translation", fov).random(3) - 1) + 0.0
            else:
                vox = np.asarray(morphology.translation_zyx[fov], dtype=np.float64) + 0.0
            shifts[fov] = vox * layout.s
    return layout, true, shifts


def _augment(stream, k, j, augmentation, kind):
    """AU1's draws of proposal ``j`` of object ``k``: (θ, f_z, f_y, s_z, s, gain)."""
    u = stream("context.appearance.augment", f"object-{k}-proposal-{j}").random(5)
    theta = 2 * math.pi * float(u[0])
    if theta >= 2 * math.pi:
        theta = 0.0
    flip_y = -1 if u[1] < augmentation.flip_y else 1
    flip_z = -1 if (kind == "nucleus" and u[2] < augmentation.flip_z) else 1
    low, high = augmentation.scale
    scale = low + (high - low) * float(u[3])
    low, high = augmentation.gain
    gain = low + (high - low) * float(u[4])
    return theta, flip_z, flip_y, (scale if kind == "nucleus" else 1.0), scale, gain


def _culture_shift(entries, z_mid):
    """The culture bank's one Z shift ``Δ = z_mid − (the median over the bank's entries of their nucleus centroid z in
    their source image)``, µm, unrounded; with the median."""
    centroids = []
    for row in range(len(entries.bank)):
        z = np.nonzero(entries.nucleus(row) <= 0)[0]
        centroids.append((float(entries.origins[row][0]) + float(z.mean())) * float(entries.spacings[row][0]))
    median = float(np.median(centroids))
    return float(z_mid) - median, median, centroids


def _radius(entries, row, scale_z, scale_yx):
    """The equivalent radius (µm) of an entry's mask after its scale: of a sphere of its volume, or with a 2D entry of
    a disc of its area (proto.py::transform, ``radius_um``)."""
    volume = float(entries.volumes[row])
    if int(entries.shapes[row][0]) == 1:
        return math.sqrt(volume * scale_yx * scale_yx / math.pi)
    return (3 * volume * scale_z * scale_yx * scale_yx / (4 * math.pi)) ** (1 / 3)


def _extent(entries, row, scale_z, scale_yx):
    """The entry box's extent after its scale, rotation-invariant in Y and X:
    (n_z s_z m_z, hypot(n_y s_y, n_x s_x) m, the same)."""
    n, s = entries.shape(row), entries.spacing(row)
    lateral = math.hypot(n[1] * s[1], n[2] * s[2]) * scale_yx
    return np.array([n[0] * s[0] * scale_z, lateral, lateral])


class _Lattice:
    """The placement lattice: the sample grid's voxel centres (one plane in a planar placement), Ω and the canvases of
    placed masks and (culture) nuclei on the sample grid."""

    def __init__(self, layout, planar, z0):
        self.s = layout.s
        self.omega = layout.omega
        self.planar = planar
        self.origin = layout.origin.copy()
        self.shape = layout.shape.copy()
        if planar:
            self.origin[0] = z0
            self.shape[0] = 1
        self.voxel = float(self.s[1] * self.s[2]) if planar else float(np.prod(self.s))
        self.canvas = np.zeros(tuple(self.shape), dtype=bool)
        self.nuclei = np.zeros(tuple(self.shape), dtype=bool)

    def mask(self, obj):
        """The object's mask on the lattice around its mask core (a point outside the core is in no cell with a mask
        voxel), with its counts; None when the core misses the lattice's plane."""
        first, axes = [], []
        for a in range(3):
            if a == 0 and self.planar:
                if not (obj.core_lo[0] <= self.origin[0] <= obj.core_hi[0]):
                    return None
                first.append(0)
                axes.append(np.zeros(1, dtype=np.int64))
                continue
            f = int(math.floor((obj.core_lo[a] - self.origin[a]) / self.s[a]))
            first.append(f)
            axes.append(np.arange(f, int(math.ceil((obj.core_hi[a] - self.origin[a]) / self.s[a])) + 1))
        coords = [self.origin[a] + axes[a] * self.s[a] for a in range(3)]
        shape = tuple(len(v) for v in axes)
        # The entry index is affine in the window index i: q = M i + offset, interpolated in one call, +inf outside
        # the entry box (its rounding differs from the point evaluator's only far below the 1e-9 µm of a boundary
        # voxel).
        B, base = obj.Ainv, self.origin + np.array(first) * self.s - obj.c
        matrix, offset = np.zeros((3, 3)), np.zeros(3)
        for a in range(3):
            if obj.n[a] > 1:
                matrix[a] = B[a] * self.s / obj.s_e[a]
                offset[a] = (obj.xbar[a] + B[a, 0] * base[0] + B[a, 1] * base[1] + B[a, 2] * base[2]) / obj.s_e[a]
        held = ndi.affine_transform(obj.D, matrix, offset=offset, output_shape=shape, order=1, mode="constant",
                                    cval=np.inf, prefilter=False) <= 0
        window = self._window(first, shape)
        within = [(c >= self.omega[0][a]) & (c < self.omega[1][a]) for a, c in enumerate(coords)]
        inside = int((held & within[0][:, None, None] & within[1][None, :, None] & within[2][None, None, :]).sum())
        overlap = 0 if window is None else int((self.canvas[window[0]] & held[window[1]]).sum())
        return SimpleNamespace(first=np.array(first), held=held, window=window, obj=obj, matrix=matrix,
                               offset=offset, shape=shape, counts=(int(held.sum()), inside, overlap), _nucleus=None)

    @staticmethod
    def verdict(cut):
        """PL2's tests on the lattice: ``domain`` (no voxel, or less than 35 % of the mask in Ω), ``overlap`` (more than
        1 % of it on earlier masks of the sample grid) or ``ok``."""
        if cut is None:
            return "domain"
        voxels, inside, overlap = cut.counts
        if voxels == 0 or 100 * inside < _DOMAIN_SHARE * voxels:
            return "domain"
        if 100 * overlap > _OVERLAP_MAX * voxels:
            return "overlap"
        return "ok"

    @staticmethod
    def nucleus(cut):
        """A culture cell's nucleus mask on the lattice window of ``cut``: its nucleus ``D_e ≤ 0`` where the cell holds
        the voxel; computed once, when the cut is taken."""
        if cut._nucleus is None:
            values = ndi.affine_transform(cut.obj.ND, cut.matrix, offset=cut.offset, output_shape=cut.shape, order=1,
                                          mode="constant", cval=np.inf, prefilter=False)
            cut._nucleus = cut.held & (values <= 0)
        return cut._nucleus

    def _window(self, first, shape):
        target, source = [], []
        for a in range(3):
            start, stop = max(first[a], 0), min(first[a] + shape[a], self.shape[a])
            if stop <= start:
                return None
            target.append(slice(start, stop))
            source.append(slice(start - first[a], stop - first[a]))
        return tuple(target), tuple(source)

    def touches(self, cut):
        """Whether a culture nucleus mask shares a voxel of the sample grid with an earlier nucleus mask."""
        nucleus = self.nucleus(cut)
        window = cut.window
        return window is not None and bool((self.nuclei[window[0]] & nucleus[window[1]]).any())

    def add(self, cut):
        window = cut.window
        if window is None:
            return
        self.canvas[window[0]] |= cut.held[window[1]]
        if cut.obj.ND is not None:
            self.nuclei[window[0]] |= self.nucleus(cut)[window[1]]


def _place_appearance(context, bank, augmentation=AugmentationConfig(), layout=LayoutConfig(), *, visible=None,
                      target=None):
    """Place bank entries on the acquisition layout of ``context`` by PL2 and make the truth labels.

    ``context`` is the unchanged ``ContextSceneConfig`` (its acquisition, seed, split, scene key, morphology round and,
    for CT1's growth law, its territories; its nuclei, packing and stains are not used), ``bank`` an
    ``AppearanceBank``. ``visible(objects)`` is DN1's density check, the visible-stain share of the composed stain on
    the sample grid after the objects placed so far, and ``target`` DN1's target where ``layout.target`` is None (the
    median over the bank's source images); ``generate_appearance_scene`` supplies both; DN3 needs neither.
    Returns a namespace with ``placements`` (one row per placed object, the columns of ``placements.csv``),
    ``proposals`` (one row per proposal), ``cells`` (one ordinary cell per object), ``labels`` (the point evaluator),
    the acquisition (``layout``, ``fov_ids``, ``true``, ``shifts``), ``record`` and ``log`` (the stream descriptors
    drawn). Raises ValueError for invalid input and for a stop below ``floor``.
    """
    layout_config = layout
    if not isinstance(context, ContextSceneConfig):
        raise TypeError("context must be ContextSceneConfig")
    if not isinstance(bank, AppearanceBank):
        raise TypeError("bank must be AppearanceBank")
    if not isinstance(augmentation, AugmentationConfig):
        raise TypeError("augmentation must be AugmentationConfig")
    if not isinstance(layout_config, LayoutConfig):
        raise TypeError("layout must be LayoutConfig")
    if not len(bank):
        raise ValueError(f"bank {bank.name}: it holds no entry")
    aim = None
    if layout_config.density == "visible":
        if visible is None:
            raise ValueError("density='visible' (DN1): the density check needs the composed stain (visible), which "
                             "generate_appearance_scene supplies")
        aim = layout_config.target if layout_config.target is not None else target
        if aim is None:
            raise ValueError("density='visible' (DN1): no target given (layout.target or the bank's target)")
        aim = _real(aim, "target", strict=True)
    template = context.sample.fov
    log = {}
    stream = _streams(template.split, template.seed, template.scene_key, log)
    grid, true, shifts = _acquisition(context, stream)
    entries = _Entries(bank)
    kind = entries.kind
    flat = bool(grid.flat)
    z_mid = float((grid.n[0] - 1) / 2 * grid.s[0])
    planar = flat or kind == "column"
    z0 = 0.0 if flat else z_mid
    lattice = _Lattice(grid, planar, z0)
    shift = median = None
    if kind == "cell" and not flat:
        shift, median, _ = _culture_shift(entries, z_mid)
    on_circle = planar or kind == "cell"
    territories = context.territories if context.territories is not None else TerritoryConfig()

    def fix_z(centre, row):
        if flat:
            centre[0] = 0.0
        elif kind == "column":
            centre[0] = z_mid
        elif kind == "cell":
            centre[0] = ((float(entries.origins[row][0]) + (float(entries.shapes[row][0]) - 1) / 2)
                         * float(entries.spacings[row][0]) + shift)
        return centre

    entry_ids = bank.entries.entry_id.astype(str).tolist()
    sources = bank.entries.source_image.astype(str).tolist()
    labels_of = bank.entries.source_label.astype(np.int64).tolist()
    plateaus = bank.entries.plateau.astype(np.float64).tolist()

    # BS2's queue: one permutation of the bank; the same generator gives the next one when the bank is exhausted.
    entry_stream = stream("context.appearance.entry", "bank")
    order = deque(int(v) for v in entry_stream.permutation(len(bank)))
    cycles = 1
    objects, radii, rows, proposals = [], [], [], []
    trace = []
    j, in_a_row, refused = 0, 0, 0
    reasons = dict(domain=0, overlap=0, nucleus=0, anchor=0)
    share = None
    stopped_by = None
    lo_w, hi_w = grid.omega
    while True:
        if layout_config.density == "count" and len(objects) >= layout_config.count:
            stopped_by = "count"
            break
        if layout_config.density == "visible" and share is not None and share >= aim:
            stopped_by = "target"
            break
        if j >= layout_config.max_proposals:
            stopped_by = "max_proposals"
            break
        if in_a_row >= layout_config.max_refusals:
            stopped_by = "max_refusals"
            break
        j += 1
        k = len(objects) + 1
        if not order:
            order.extend(int(v) for v in entry_stream.permutation(len(bank)))
            cycles += 1
        row = order.popleft()
        theta, flip_z, flip_y, scale_z, scale_yx, gain = _augment(stream, k, j, augmentation, kind)
        r_k = _radius(entries, row, scale_z, scale_yx)
        rng = stream("context.appearance.placement", f"object-{k}-proposal-{j}")
        contact = rng.random() < layout_config.contact and bool(objects)
        partner, step, taken, reason = None, None, None, None
        if contact:
            partner = int(rng.integers(len(objects)))
            direction = _direction(rng, on_circle)
            base = objects[partner].c
            reach = radii[partner] + r_k
            reason = "overlap"
            obj = None
            for f in _LADDER:
                centre = fix_z(base + f * reach * direction, row)
                if obj is None:
                    obj = _Object(k, entries, row, centre, theta, flip_z, flip_y, scale_z, scale_yx, gain)
                else:
                    obj.move(centre)
                cut = lattice.mask(obj)
                verdict = lattice.verdict(cut)
                if verdict == "domain":
                    reason = "domain"
                    break
                if verdict == "ok":
                    taken, step, reason = (obj, cut, cut.counts[0], cut.counts[2]), f, None
                    break
        else:
            extent = _extent(entries, row, scale_z, scale_yx)
            u = rng.random(3)
            low, high = lo_w - _WIDEN * extent, hi_w + _WIDEN * extent
            centre = fix_z(low + u * (high - low), row)
            obj = _Object(k, entries, row, centre, theta, flip_z, flip_y, scale_z, scale_yx, gain)
            cut = lattice.mask(obj)
            verdict = lattice.verdict(cut)
            if verdict == "ok":
                taken = (obj, cut, cut.counts[0], cut.counts[2])
            else:
                reason = verdict
        if taken is not None and kind == "cell" and lattice.touches(taken[1]):
            taken, reason = None, "nucleus"
        if taken is not None:
            anchor = taken[0].anchor()
            point = anchor[None]
            if any(len(o.select(point)) and o.values(point, o.D)[0] <= 0 for o in objects):
                taken, reason = None, "anchor"
        entry_id = entry_ids[row]
        proposals.append((j, k, entry_id, bool(contact), None if partner is None else partner + 1, step,
                          "refused" if taken is None else "placed", reason))
        if taken is None:
            order.append(row)
            in_a_row += 1
            refused += 1
            reasons[reason] += 1
            continue
        obj, cut, voxels, overlap = taken
        in_a_row = 0
        lattice.add(cut)
        objects.append(obj)
        radii.append(r_k)
        rows.append((k, "cell" if kind == "cell" else "nucleus", entry_id, sources[row], labels_of[row],
                     plateaus[row], *obj.c.tolist(), *anchor.tolist(), obj.theta, obj.flip_z, obj.flip_y,
                     obj.scale_z, obj.scale_yx, obj.gain, j, overlap / voxels, voxels * lattice.voxel))
        if layout_config.density == "visible" and len(objects) % layout_config.check_every == 0:
            share = float(visible(list(objects)))
            trace.append(dict(placed=len(objects), proposals=j, visible_share=share))
    mode = layout_config.density
    if mode == "visible":
        if not trace or trace[-1]["placed"] != len(objects):
            share = float(visible(list(objects)))
            trace.append(dict(placed=len(objects), proposals=j, visible_share=share))
        reached = share
    else:
        reached, aim = len(objects), layout_config.count
    if stopped_by in ("max_proposals", "max_refusals") and reached < layout_config.floor * aim:
        what = "a visible share of" if mode == "visible" else "a count of"
        name = "target" if mode == "visible" else "count"
        limit = (f"max_proposals={layout_config.max_proposals}" if stopped_by == "max_proposals"
                 else f"max_refusals={layout_config.max_refusals} refused proposals in a row")
        raise ValueError(f"placement stopped at {limit} after {j} proposals with {what} {reached} "
                         f"({len(objects)} objects placed), below floor · {name} = {layout_config.floor} · {aim} = "
                         f"{layout_config.floor * aim}")
    growth = None
    if kind != "cell":
        growth = []
        for obj in objects:
            g = _draw(territories.growth, "growth", f"cell-{obj.k}",
                      lambda _name, entity: stream("context.appearance.growth", entity))
            if not (math.isfinite(g) and g > 0):
                raise ValueError(f"cell-{obj.k}: nonfinite or nonpositive growth")
            growth.append(g)
    placements = pd.DataFrame(rows, columns=[
        "object_id", "kind", "entry_id", "source_image", "source_label", "source_plateau", "centre_z", "centre_y",
        "centre_x", "anchor_z", "anchor_y", "anchor_x", "theta", "flip_z", "flip_y", "scale_z", "scale_yx", "gain",
        "proposal", "overlap_share", "volume"])
    placements = placements.astype(dict(object_id=np.uint32, kind=object, entry_id=object, source_image=object,
                                        source_label=np.int64, flip_z=np.int64, flip_y=np.int64, proposal=np.int64,
                                        **{c: np.float64 for c in ("source_plateau", "centre_z", "centre_y",
                                                                   "centre_x", "anchor_z", "anchor_y", "anchor_x",
                                                                   "theta", "scale_z", "scale_yx", "gain",
                                                                   "overlap_share", "volume")}))
    table = pd.DataFrame(proposals, columns=["proposal", "object", "entry_id", "contact", "partner", "step",
                                             "outcome", "reason"])
    table = table.astype(dict(proposal=np.int64, object=np.int64, entry_id=object, contact=bool, partner="Int64",
                              step=np.float64, outcome=object, reason=object))
    cells = pd.DataFrame(dict(cell_id=placements.object_id.to_numpy(), nucleus_id=placements.object_id.to_numpy(),
                              kind="ordinary", n_nuclei=np.ones(len(objects), dtype=np.int64),
                              anchor_z=placements.anchor_z, anchor_y=placements.anchor_y,
                              anchor_x=placements.anchor_x,
                              growth=np.asarray(growth if growth is not None else [np.nan] * len(objects),
                                                dtype=np.float64)))
    cells = cells.astype(dict(cell_id=np.uint32, nucleus_id=np.uint32, kind=object))
    labels = _Labels(kind, objects, growth)
    components = sorted({descriptor[4] for descriptor in log.values()})
    record = dict(
        kind=kind, territories="CU1" if kind == "cell" else "CT1", planar=planar, flat=flat, z_mid=z_mid,
        placement_plane_z=z0 if planar else None, culture_shift_um=shift, culture_median_centroid_um=median,
        density=dict(mode=mode, count=layout_config.count, target=aim if mode == "visible" else None,
                     threshold=layout_config.threshold, check_every=layout_config.check_every,
                     reached=reached, placed=len(objects), proposals=j, refused=refused, refused_in_a_row=in_a_row,
                     refusals=reasons, stopped_by=stopped_by, floor=layout_config.floor, trace=trace),
        bank=dict(name=bank.name, sha256=bank.sha256, digest=bank.digest, entries=len(bank), cycles=cycles),
        augmentation=asdict(augmentation), layout=asdict(layout_config),
        rules=dict(overlap_max_percent=_OVERLAP_MAX, domain_share_percent=_DOMAIN_SHARE, ladder=list(_LADDER),
                   widen=_WIDEN),
        stream=dict(namespace=_STREAM_NAMESPACE, split=template.split, seed=template.seed,
                    scene_key=template.scene_key,
                    components={k: dict(entity=e, draws=d) for k, (e, d) in _LAYOUT_COMPONENTS.items()},
                    drawn=components))
    return SimpleNamespace(kind=kind, placements=placements, proposals=table, cells=cells, labels=labels,
                           objects=objects, layout=grid, fov_ids=list(grid.ids), true=true, shifts=shifts,
                           record=record, log=log)


def _labels_from_table(bank, placements, growth=None):
    """The point evaluator rebuilt from a placements table (the columns of ``placements.csv``) and the bank alone:
    each row's entry, centre and transform in ``object_id`` order; ``growth`` per row for CT1 (None in the culture)."""
    entries = _Entries(bank)
    objects = []
    for row in placements.sort_values("object_id", kind="stable").itertuples():
        objects.append(_Object(int(row.object_id), entries, bank.entry(row.entry_id),
                               (row.centre_z, row.centre_y, row.centre_x), row.theta, row.flip_z, row.flip_y,
                               row.scale_z, row.scale_yx, row.gain))
    return _Labels(entries.kind, objects, growth)


def _sample_labels(placed):
    """``N`` and ``C`` on the sample grid of a placed layout."""
    grid = placed.layout
    C, N = placed.labels.grid(grid.origin, grid.s, grid.shape)
    return dict(nucleus=N, cell=C)


def _fov_labels(placed):
    """``N`` and ``C`` at every FOV's voxel centres ``true_f + p ⊙ s``, by FOV ID."""
    grid = placed.layout
    out = {}
    for k, fov in enumerate(placed.fov_ids):
        C, N = placed.labels.grid(placed.true[k], grid.s, grid.n)
        out[fov] = dict(nucleus=N, cell=C)
    return out


def _morphology_labels(placed):
    """``N`` and ``C`` on the morphology round's grid ``true_f + τ_f + q ⊙ s`` of every FOV; empty without a
    morphology round."""
    if placed.shifts is None:
        return {}
    grid = placed.layout
    out = {}
    for k, fov in enumerate(placed.fov_ids):
        C, N = placed.labels.grid(placed.true[k] + placed.shifts[fov], grid.s, grid.n)
        out[fov] = dict(nucleus=N, cell=C)
    return out
