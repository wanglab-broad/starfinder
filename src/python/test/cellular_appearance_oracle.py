"""Test oracle of the real-appearance scene (docs/synthetic-real-appearance-validation.md, "Conventions").

The bank's part (W-516): written from the formulas of docs/synthetic-real-appearance.md ("Terms", "The appearance
bank", "What an entry carries", "Composition" for the patch rule) and the bank's record alone. It imports nothing
from ``starfinder.synthetic``; it reads a bank through ``AppearanceBank``'s arrays and record and recomputes, from
the source arrays, which rule leaves out each label, the plateau and the median over the visible stain of each
source image, BR4's guard, and per entry its signed distance, local background, rim level, excess inside the mask and
the four steps of OUT1 outside it, with the library calls the page names (``scipy.ndimage``'s distance transform,
Gaussian filter and median; ``skimage.filters.threshold_otsu``). The settings the page leaves to configuration (the
margins, the fall-off length and blurs, the bins, the patch rule) are read from the bank's record.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math

import numpy as np
from scipy import ndimage as ndi
from scipy.interpolate import RegularGridInterpolator
from skimage.filters import threshold_otsu

# Labels whose rim share or size lies within this distance of a rule's limit are reported (GA5).
EPS = 1e-9
# One float32 rounding, relative (GA6, GA14).
FLOAT32 = 2.0 ** -23


def face_structure(shape):
    """Face neighbours in 3D; in a single plane only the Y and X neighbours."""
    s = np.zeros((3, 3, 3), dtype=bool)
    s[1, 1, :] = s[1, :, 1] = True
    if shape[0] > 1:
        s[:, 1, 1] = True
    return s


def sigmas(sigma_um, spacing, shape):
    """A Gaussian of ``sigma_um`` µm in source voxels per axis, none along a single-plane axis."""
    return [sigma_um / s if n > 1 else 0.0 for s, n in zip(spacing, shape)]


def signed_distance(mask, spacing):
    """D_e = EDT(not M) − EDT(M) ("Terms")."""
    return ndi.distance_transform_edt(~mask, sampling=spacing) - ndi.distance_transform_edt(mask, sampling=spacing)


def plateau(labels, stain):
    """The median over the labels of each label's median stain ("Terms")."""
    ids = [int(v) for v in np.unique(labels) if v > 0]
    medians = [float(np.median(stain[labels == i])) for i in ids]
    return float(np.median(medians)) if medians else math.nan


def visible_median(stain):
    """The median stain where the stain smoothed by one voxel lies above its Otsu threshold ("Terms")."""
    smooth = ndi.gaussian_filter(stain, [1.0 if n > 1 else 0.0 for n in stain.shape])
    return float(np.median(stain[smooth > threshold_otsu(smooth)]))


def guard(plateaus, visible, factor):
    """BR4: the images refused, each with its reason; and the bank's median plateau."""
    median = float(np.median(plateaus))
    refused = []
    for k, (p, v) in enumerate(zip(plateaus, visible)):
        if p / median > factor or median / p > factor:
            refused.append((k, "median", p, median))
        if v / p > factor or p / v > factor:
            refused.append((k, "visible", p, v))
    return refused, median


def box_of(labels, label, margin_um, spacing, keep_z):
    """The label's bounding box grown by the margin (whole voxels, ceil), clipped; every plane with ``keep_z``."""
    where = np.nonzero(labels == label)
    flat = labels.shape[0] == 1
    margin = [int(math.ceil(m / s)) for m, s in zip(margin_um, spacing)]
    if flat or keep_z:
        margin[0] = 0
    lo = [max(int(w.min()) - m, 0) for w, m in zip(where, margin)]
    hi = [min(int(w.max()) + 1 + m, n) for w, m, n in zip(where, margin, labels.shape)]
    if keep_z:
        lo[0], hi[0] = 0, labels.shape[0]
    return tuple(slice(a, b) for a, b in zip(lo, hi))


def selection(labels, nucleus_labels, spacing, configuration):
    """Per label of a source image: the first rule that leaves it out (``border``, ``size``, ``nucleus``,
    ``squashed``) or ``kept``, its voxels, rim share, nucleus label and nucleus voxels, and the labels within EPS of
    a size or rim-share limit ("Selection of entries", in the rules' order)."""
    rules = configuration["selection"]
    keep_z = configuration["keep_z"]
    flat = labels.shape[0] == 1
    rows = {}
    for label in (int(v) for v in np.unique(labels) if v > 0):
        where = np.nonzero(labels == label)
        z_end = (not flat) and (where[0].min() == 0 or where[0].max() == labels.shape[0] - 1)
        yx_edge = any(w.min() == 0 or w.max() == n - 1 for w, n in zip(where[1:], labels.shape[1:]))
        rows[label] = dict(voxels=len(where[0]), border=yx_edge or (z_end and rules["border_z"]), z_end=z_end)
    sizes = [r["voxels"] for r in rows.values() if not r["border"]]
    lo, hi = np.percentile(sizes, rules["size_percentiles"]) if sizes else (math.nan, math.nan)
    structure = face_structure(labels.shape)
    near = []
    for label, row in rows.items():
        row.update(rim_share=math.nan, nucleus_label=0, nucleus_voxels=0)
        if row["border"]:
            row["outcome"] = "border"
            continue
        if min(abs(row["voxels"] - lo), abs(row["voxels"] - hi)) <= EPS:
            near.append(label)
        if not lo <= row["voxels"] <= hi:
            row["outcome"] = "size"
            continue
        box = box_of(labels, label, configuration["margin_um"], spacing, keep_z)
        own = labels[box] == label
        other = (labels[box] > 0) & ~own
        rim = own & ~ndi.binary_erosion(own, structure=structure)
        row["rim_share"] = float((rim & ndi.binary_dilation(other, structure=structure)).sum() / rim.sum())
        if nucleus_labels is not None:
            inside = nucleus_labels[box][own]
            inside = inside[inside > 0]
            if inside.size:
                ids, counts = np.unique(inside, return_counts=True)
                row["nucleus_label"] = int(ids[np.argmax(counts)])
                row["nucleus_voxels"] = int(counts.max())
            if row["nucleus_voxels"] < rules["min_nucleus_voxels"]:
                row["outcome"] = "nucleus"
                continue
        if abs(row["rim_share"] - rules["rim_share_max"]) <= EPS:
            near.append(label)
        row["outcome"] = "squashed" if row["rim_share"] > rules["rim_share_max"] else "kept"
    counts = dict(labels=len(rows), left_out_border=sum(r["outcome"] == "border" for r in rows.values()),
                  left_out_border_z_alone=(sum(r["z_end"] for r in rows.values())
                                           if rules["border_z"] and not flat else 0),
                  left_out_size_after_border=sum(r["outcome"] == "size" for r in rows.values()),
                  left_out_nucleus_after_size=sum(r["outcome"] == "nucleus" for r in rows.values()),
                  left_out_squashed_after_rest=sum(r["outcome"] == "squashed" for r in rows.values()),
                  kept=sum(r["outcome"] == "kept" for r in rows.values()))
    return rows, counts, near


def fall_off(X, own, other, d, r, spacing, settings, monotone=True):
    """OUT1's four steps on an entry's box ("What an entry carries"): (1) the surroundings smoothed and capped by the
    linear fall-off from the rim level; (2) the weight v of the nearest source label, smoothed, 0 on other labels;
    (3) ``own``, the per-distance median of the capped values where v > 0.5 (made non-increasing when ``monotone``);
    (4) ``kept · v + own · (1 − v)`` blurred outside the mask, the excess unchanged inside; zero beyond λ plus the
    blur's four deviations. Returns the field and the parts (kept, v, own)."""
    length, blur, step = settings["length_um"], settings["blur_um"], settings["bin_um"]
    shape = X.shape
    cap = r * np.maximum(0.0, 1.0 - d / length)
    kept = np.minimum(ndi.gaussian_filter(X, sigmas(settings["smooth_um"], spacing, shape)), cap)
    d_other = ndi.distance_transform_edt(~other, sampling=spacing) if other.any() else np.full(shape, np.inf)
    v = ndi.gaussian_filter((d <= d_other).astype(np.float64), sigmas(settings["weight_um"], spacing, shape))
    v[other] = 0.0
    n = int(round(length / step))
    edges = step * np.arange(n + 1)
    centres = (edges[:-1] + edges[1:]) / 2
    side = ~own & (v > 0.5)
    profile = np.array([np.median(kept[side & (d > a) & (d <= b)])
                        if np.count_nonzero(side & (d > a) & (d <= b)) >= settings["min_bin_voxels"] else np.nan
                        for a, b in zip(edges[:-1], edges[1:])])
    ok = np.isfinite(profile)
    if ok.sum() >= settings["min_bins"]:
        profile = np.interp(centres, centres[ok], profile[ok])
        if monotone:
            profile = np.minimum.accumulate(profile)
        own_field = np.interp(d, np.r_[0.0, centres, length], np.r_[min(r, profile[0]), profile, 0.0])
    else:
        own_field = cap
    outside = kept * v + own_field * (1.0 - v)
    field = np.where(own, X, ndi.gaussian_filter(np.where(own, X, outside), sigmas(blur, spacing, shape)))
    field[d > length + 4 * blur] = 0.0
    return field, dict(kept=kept, v=v, own=own_field)


def entry(stains, labels, nucleus_labels, label, nucleus_label, spacing, configuration, monotone=True):
    """An entry recomputed from its source arrays: box, D_e, nucleus D_e, and per channel the local background ``b``
    (the 20th percentile over the box's unlabelled voxels, 0 with 20 or fewer), the rim level ``r`` (the median
    excess over the inner rim), the excess ``max(I − b, 0)`` and the stored field of OUT1, all in source units."""
    box = box_of(labels, label, configuration["margin_um"], spacing, configuration["keep_z"])
    L = labels[box]
    own = L == label
    other = (L > 0) & ~own
    free = ~own & ~other
    rim = own & ~ndi.binary_erosion(own, structure=face_structure(own.shape))
    d = ndi.distance_transform_edt(~own, sampling=spacing)
    out = dict(box=box, own=own, other=other, d=d, distance=signed_distance(own, spacing), b=[], r=[], X=[], field=[],
               parts=[])
    if nucleus_labels is not None:
        out["nucleus"] = (nucleus_labels[box] == nucleus_label) & own
        out["nucleus_distance"] = signed_distance(out["nucleus"], spacing)
    for stain in stains:
        values = stain[box]
        b = float(np.percentile(values[free], 20)) if np.count_nonzero(free) > 20 else 0.0
        X = np.maximum(values - b, 0.0)
        r = float(np.median(X[rim]))
        field, parts = fall_off(X, own, other, d, r, spacing, configuration["fall_off"], monotone)
        out["b"].append(b), out["r"].append(r), out["X"].append(X), out["field"].append(field)
        out["parts"].append(parts)
    return out


def foreground(labels, nucleus_labels, stains, spacing, plateau_value, settings):
    """BG1's patch rule on a source image: the voxels within ``label_distance_um`` of a label, or with a stain above
    ``level`` of the plateau."""
    labelled = labels > 0
    if nucleus_labels is not None:
        labelled = labelled | (nucleus_labels > 0)
    mask = ndi.distance_transform_edt(~labelled, sampling=spacing) < settings["label_distance_um"]
    for stain in stains:
        mask |= stain > settings["level"] * plateau_value
    return mask


def patch_share(mask, origin, size):
    """The share of a patch's voxels that are foreground."""
    return float(mask[tuple(slice(int(o), int(o) + int(n)) for o, n in zip(origin, size))].mean())


def bank_entries(bank):
    """The (source image, source label) of every entry of a bank."""
    return {(str(i), int(label)) for i, label in zip(bank.arrays["source_image"], bank.arrays["source_label"])}


def stored(bank, entry_id):
    """An entry's stored arrays: (D_e, excess per channel, nucleus D_e or None), float64 views."""
    i = bank.entry(entry_id)
    start, shape = int(bank.arrays["offset"][i]), tuple(int(v) for v in bank.arrays["box_shape"][i])
    size = int(np.prod(shape))
    distance = bank.arrays["distance"][start:start + size].reshape(shape).astype(np.float64)
    excess = bank.arrays["excess"][:, start:start + size].reshape((-1, *shape)).astype(np.float64)
    nucleus = (bank.arrays["nucleus_distance"][start:start + size].reshape(shape).astype(np.float64)
               if bank.record["entry"] == "cell" else None)
    return distance, excess, nucleus


# ------------------------------------------------------------------------------------------------ the label part
# (W-517) Written from docs/synthetic-real-appearance.md ("Terms", "Augmentation", "Layout", "Truth", "Random
# streams") and the layout's own record of its draw order; it reads the bank through ``stored`` and recomputes the
# transforms with its own matrices and inverse, the entries' distances with ``scipy.interpolate``'s linear
# interpolator on the entry frame (µm), the streams from their descriptors and the labels, territories, masks and
# placement draws from those.

# A voxel whose interpolated D_e lies within this distance (µm) of 0, or where two objects' δ tie within it under CT1,
# is a boundary voxel ("Identities a test can check").
BOUNDARY = 1e-9
# PL2: the ladder of f, the widening of Ω for a uniform proposal, the 35 % in Ω and the 1 % overlap.
LADDER = [0.5 + 0.04 * m for m in range(23)]
WIDEN = 0.3


def stream(scheme, component, entity):
    """The generator of one descriptor (namespace, split, seed, scene key, component, entity, round, channel): its
    compact UTF-8 JSON array, hashed by SHA-256 and read as a big-endian integer, seeds PCG64."""
    descriptor = [scheme["namespace"], scheme["split"], scheme["seed"], scheme["scene_key"], component, entity, None,
                  None]
    encoded = json.dumps(descriptor, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return np.random.Generator(np.random.PCG64(int(hashlib.sha256(encoded.encode("utf-8")).hexdigest(), 16)))


def kind_of(bank):
    """``cell`` (culture cells, CU1), ``column`` (2D entries, TZ1) or ``nucleus`` (dense 3D, CT1)."""
    if bank.record["entry"] == "cell":
        return "cell"
    return "column" if (bank.arrays["box_shape"][:, 0] == 1).all() else "nucleus"


def matrix(theta, flip_z, flip_y, scale_z, scale_yx):
    """A = S R F ("Augmentation", the transform)."""
    rotation = np.array([[1, 0, 0], [0, math.cos(theta), -math.sin(theta)], [0, math.sin(theta), math.cos(theta)]])
    return np.diag([scale_z, scale_yx, scale_yx]) @ rotation @ np.diag([flip_z, flip_y, 1.0])


class Frame:
    """One placed object of a placements row: T(x) = c + A (x − x̄_e), its entry's D_e (and nucleus D_e) as linear
    interpolators on the entry frame in µm, +inf outside the box; a 2D entry is evaluated in its plane whatever z."""

    def __init__(self, bank, row):
        self.k = int(row.object_id)
        info = bank.entries.iloc[bank.entry(row.entry_id)]
        self.shape = np.array([info.shape_z, info.shape_y, info.shape_x], dtype=np.int64)
        self.spacing = np.array([info.spacing_z, info.spacing_y, info.spacing_x], dtype=np.float64)
        self.xbar = (self.shape - 1) / 2 * self.spacing
        self.c = np.array([row.centre_z, row.centre_y, row.centre_x], dtype=np.float64)
        self.A = matrix(row.theta, row.flip_z, row.flip_y, row.scale_z, row.scale_yx)
        self.Ainv = np.linalg.inv(self.A)
        self.m = float(row.scale_yx)
        self.gain = float(row.gain)
        distance, _, nucleus = stored(bank, row.entry_id)
        self.D = self._interpolator(distance)
        self.ND = None if nucleus is None else self._interpolator(nucleus)
        self.distance = distance
        self.box_origin = np.array([info.box_origin_z, info.box_origin_y, info.box_origin_x], dtype=np.float64)

    def _interpolator(self, values):
        axes = [np.arange(n) * s for n, s in zip(self.shape, self.spacing)]
        if self.shape[0] == 1:
            return RegularGridInterpolator(axes[1:], values[0], bounds_error=False, fill_value=np.inf)
        return RegularGridInterpolator(axes, values, bounds_error=False, fill_value=np.inf)

    def at(self, centre):
        """The same object with its centre moved to ``centre`` (its entry and transform shared)."""
        moved = copy.copy(self)
        moved.c = np.asarray(centre, dtype=np.float64)
        return moved

    def entry_frame(self, points):
        return self.xbar + (np.asarray(points, dtype=np.float64).reshape(-1, 3) - self.c) @ self.Ainv.T

    def forward(self, x):
        return self.c + (np.asarray(x, dtype=np.float64) - self.xbar) @ self.A.T

    def value(self, interpolator, points):
        x = self.entry_frame(points)
        if self.shape[0] == 1:
            x = x[:, 1:]
        extent = ((self.shape - 1) * self.spacing)[1:] if self.shape[0] == 1 else (self.shape - 1) * self.spacing
        box = np.all((x >= 0) & (x <= extent), axis=1)
        out = np.full(len(x), np.inf)
        if box.any():
            out[box] = interpolator(x[box])
        return out

    def d(self, points):
        return self.value(self.D, points)

    def nd(self, points):
        return self.value(self.ND, points)

    def deepest(self):
        """The entry's voxel of smallest D_e, transformed: the anchor PL2 records (of equal values, the one nearest the
        box centre, then the first in C order)."""
        candidates = np.argwhere(self.distance == self.distance.min())
        # The offset from the box centre per axis from the index, so that mirror-image voxels tie exactly.
        squared = (((candidates - (self.shape - 1) / 2) * self.spacing) ** 2).sum(axis=1)
        return self.forward(candidates[int(np.argmin(squared))] * self.spacing)

    def corners(self, core=False):
        """The transformed corners of the entry box, or with ``core`` of the box of its mask voxels grown by one voxel
        (outside it every interpolation cell has corners outside the mask only, so D_e > 0 there)."""
        if core:
            where = np.nonzero(self.distance <= 0)
            lo = [max(int(w.min()) - 1, 0) for w in where]
            hi = [min(int(w.max()) + 1, n - 1) for w, n in zip(where, self.shape)]
        else:
            lo, hi = [0, 0, 0], list(self.shape - 1)
        q = np.array([[a, b, c] for a in (lo[0], hi[0]) for b in (lo[1], hi[1]) for c in (lo[2], hi[2])],
                     dtype=np.float64)
        return self.forward(q * self.spacing)


def frames(bank, placements):
    return [Frame(bank, row) for row in placements.sort_values("object_id").itertuples()]


def labels(points, objects, kind, growth=None):
    """N and C at (P, 3) sample positions by one placement order ("Truth"; "One ownership rule for nuclei and
    cells"), and the boundary voxels: a D_e (or, in the culture, the owner's nucleus D_e) within BOUNDARY of 0, or two
    smallest δ within BOUNDARY of each other under CT1; also the voxels whose smallest δ lies within BOUNDARY of its
    growth (reported)."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    P = len(points)
    ids = np.array([o.k for o in objects], dtype=np.int64)
    D = np.full((P, len(objects)), np.inf)
    for i, o in enumerate(objects):
        corners = o.corners()
        lo, hi = corners.min(axis=0) - 1e-6, corners.max(axis=0) + 1e-6
        if o.shape[0] == 1:
            lo[0], hi[0] = -np.inf, np.inf
        rows = np.flatnonzero(np.all((points >= lo) & (points <= hi), axis=1))
        D[rows, i] = o.d(points[rows])
    held = D <= 0
    anyheld = held.any(axis=1)
    first = np.argmax(held, axis=1)
    boundary = (np.abs(D) <= BOUNDARY).any(axis=1)
    edge = np.zeros(P, dtype=bool)
    N = np.zeros(P, dtype=np.int64)
    C = np.zeros(P, dtype=np.int64)
    if kind == "cell":
        C[anyheld] = ids[first[anyheld]]
        for i, o in enumerate(objects):
            mine = np.flatnonzero(anyheld & (first == i))
            if len(mine):
                nd = o.nd(points[mine])
                N[mine[nd <= 0]] = o.k
                boundary[mine[np.abs(nd) <= BOUNDARY]] = True
        return N, C, boundary, edge
    N[anyheld] = ids[first[anyheld]]
    C[:] = N
    if objects:
        delta = D * np.array([o.m for o in objects])
        free = np.flatnonzero(~anyheld)
        part = delta[free]
        nearest = np.argmin(part, axis=1)
        best = part[np.arange(len(free)), nearest]
        g = np.asarray(growth, dtype=np.float64)
        hit = np.isfinite(best) & (best <= g[nearest])
        C[free[hit]] = ids[nearest[hit]]
        if part.shape[1] > 1:
            two = np.sort(part, axis=1)[:, :2]
            finite = np.flatnonzero(np.isfinite(two).all(axis=1))
            boundary[free[finite[np.abs(two[finite, 1] - two[finite, 0]) <= BOUNDARY]]] = True
        edge[free[np.isfinite(best) & (np.abs(best - g[nearest]) <= BOUNDARY)]] = True
    return N, C, boundary, edge


def grid_points(origin, spacing, shape):
    """The voxel centres origin + p ⊙ spacing of a ZYX grid, (*shape, 3)."""
    return np.asarray(origin, dtype=np.float64) + np.moveaxis(np.indices(tuple(shape), dtype=np.float64), 0, -1) \
        * np.asarray(spacing, dtype=np.float64)


def lattice_mask(o, origin, spacing, plane_z=None):
    """The object's mask on the unbounded lattice origin + p ⊙ spacing (one plane at z = plane_z for a planar
    placement), over the lattice points around its transformed mask core: (first index, mask, nucleus mask or None,
    points)."""
    origin = np.asarray(origin, dtype=np.float64).copy()
    spacing = np.asarray(spacing, dtype=np.float64)
    corners = o.corners(core=True)
    lo = np.floor((corners.min(axis=0) - origin) / spacing).astype(np.int64) - 1
    hi = np.ceil((corners.max(axis=0) - origin) / spacing).astype(np.int64) + 1
    if plane_z is not None:
        origin[0] = plane_z
        lo[0] = hi[0] = 0
    shape = hi - lo + 1
    points = origin + (np.moveaxis(np.indices(tuple(shape), dtype=np.float64), 0, -1) + lo) * spacing
    flat = points.reshape(-1, 3)
    mask = (o.d(flat) <= 0).reshape(tuple(shape))
    nucleus = None
    if o.ND is not None:
        nucleus = np.zeros(len(flat), dtype=bool)
        chosen = mask.ravel()
        nucleus[chosen] = o.nd(flat[chosen]) <= 0
        nucleus = nucleus.reshape(tuple(shape))
    return lo, mask, nucleus, points


def on_grid(lo, mask, shape):
    """The part of a lattice mask on a grid of ``shape``: (grid slices, mask slices) or None."""
    target, source = [], []
    for a in range(3):
        start, stop = max(lo[a], 0), min(lo[a] + mask.shape[a], shape[a])
        if stop <= start:
            return None
        target.append(slice(start, stop))
        source.append(slice(start - lo[a], stop - lo[a]))
    return tuple(target), tuple(source)


def inside(points, omega):
    """Points in Ω, half-open [lo, hi) per axis."""
    return np.all((points >= omega[0]) & (points < omega[1]), axis=-1)


def augment_draws(scheme, k, j, augmentation, kind):
    """AU1's draws of proposal j of object k, in the layout's order: five uniforms θ, Y flip, Z flip, scale, gain."""
    u = stream(scheme, "context.appearance.augment", f"object-{k}-proposal-{j}").random(5)
    theta = 2 * math.pi * u[0]
    scale = augmentation["scale"][0] + (augmentation["scale"][1] - augmentation["scale"][0]) * u[3]
    return dict(theta=theta if theta < 2 * math.pi else 0.0, flip_y=-1 if u[1] < augmentation["flip_y"] else 1,
                flip_z=-1 if kind == "nucleus" and u[2] < augmentation["flip_z"] else 1,
                scale_z=scale if kind == "nucleus" else 1.0, scale_yx=scale,
                gain=augmentation["gain"][0] + (augmentation["gain"][1] - augmentation["gain"][0]) * u[4])


def placement_draws(scheme, k, j, contact, placed, on_circle):
    """PL2's draws of proposal j of object k: the contact uniform; for a contact proposal the partner (0-based among
    the placed objects) and the direction (on the circle, or two uniforms on the sphere); else three uniforms."""
    rng = stream(scheme, "context.appearance.placement", f"object-{k}-proposal-{j}")
    if rng.random() < contact and placed:
        partner = int(rng.integers(placed))
        if on_circle:
            phi = 2 * math.pi * rng.random()
            u = np.array([0.0, math.sin(phi), math.cos(phi)])
        else:
            a, b = rng.random(2)
            z = 2 * a - 1
            s = math.sqrt(max(0.0, 1 - z * z))
            u = np.array([z, s * math.sin(2 * math.pi * b), s * math.cos(2 * math.pi * b)])
        return dict(contact=True, partner=partner, direction=u)
    return dict(contact=False, uniform=rng.random(3))


def replay_queue(scheme, size, placed_at, last):
    """BS2's queue replayed: the permutation of ``context.appearance.entry`` (entity ``bank``; the next permutation of
    the same generator once the bank is exhausted); at each proposal j up to ``last`` the head is drawn, kept when
    ``placed_at`` (proposal → placed) holds j, else moved to the end. Returns the entry row drawn at each proposal."""
    rng = stream(scheme, "context.appearance.entry", "bank")
    queue = list(rng.permutation(size))
    drawn = {}
    for j in range(1, last + 1):
        if not queue:
            queue = list(rng.permutation(size))
        head = int(queue.pop(0))
        drawn[j] = head
        if j not in placed_at:
            queue.append(head)
    return drawn


def equivalent_radius(bank, entry_id, scale_z, scale_yx):
    """The radius (µm) of a sphere of the scaled entry's volume, or of a disc of its area for a 2D entry."""
    info = bank.entries.iloc[bank.entry(entry_id)]
    if int(info.shape_z) == 1:
        return math.sqrt(info.volume * scale_yx ** 2 / math.pi)
    return (3 * info.volume * scale_z * scale_yx ** 2 / (4 * math.pi)) ** (1 / 3)


def widening(bank, entry_id, scale_z, scale_yx):
    """0.3 of the scaled entry box's extent, rotation-invariant in Y and X (n_z s_z m_z; hypot(n_y s_y, n_x s_x) m)."""
    info = bank.entries.iloc[bank.entry(entry_id)]
    lateral = math.hypot(info.shape_y * info.spacing_y, info.shape_x * info.spacing_x) * scale_yx
    return WIDEN * np.array([info.shape_z * info.spacing_z * scale_z, lateral, lateral])
