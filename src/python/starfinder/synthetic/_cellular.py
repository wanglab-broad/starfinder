"""Shape functions of the cellular scene: lobed nuclei, extents, contact distances, labels.

See design/specs/synthetic/cellular-scene.md ("Nucleus shape", "Packing", "Cell territories").
A nucleus is a union of ellipsoids (lobe 1 optionally perturbed by real spherical
harmonics, N2); every function here is a closed form of the sample position and the
drawn values, so labels stay point functions evaluated at voxel centres. This module
imports no other synthetic module.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
from scipy.special import sph_harm_y


def _harmonic_terms(degree, planar):
    """The terms of r in drawing order: (degree, order) by degree then order, or (k, cos/sin) in a plane."""
    if degree < 2:
        return []
    if planar:
        return [(k, kind) for k in range(2, degree + 1) for kind in ("cos", "sin")]
    return [(l, m) for l in range(2, degree + 1) for m in range(-l, l + 1)]


def _harmonic_bound(degree, planar):
    """M_Y, the largest abs(Y_lm) over l ≤ L: 1 for the Fourier form, sqrt((2L + 1) / 4π) on the sphere.

    On the sphere Σ_m Y_lm² = (2l + 1) / 4π at every direction, and Y_l0 reaches the bound at the poles.
    """
    if planar:
        return 1.0
    return math.sqrt((2 * degree + 1) / (4 * math.pi))


def _frame(vectors, cos, sin, semi):
    """Vectors (..., 3) in an ellipsoid's normalized frame (d_z / a_z, u / (a_l e), v / a_l)."""
    u = cos * vectors[..., 1] + sin * vectors[..., 2]
    v = -sin * vectors[..., 1] + cos * vectors[..., 2]
    return np.stack([vectors[..., 0] / semi[..., 0], u / semi[..., 1], v / semi[..., 2]], axis=-1)


def _norm(q):
    return np.sqrt(q[..., 0] ** 2 + q[..., 1] ** 2 + q[..., 2] ** 2)


def _radius(q, coefficients, terms, planar):
    """r(d̂') of N2 at the directions of q (..., 3) in lobe 1's normalized frame (polar axis z).

    In one plane or a monolayer r(φ) = 1 + Σ (a_k cos kφ + b_k sin kφ) with φ the angle of (d'_u, d'_v);
    otherwise r = 1 + Σ a_lm Y_lm with the real harmonics of scipy.special.sph_harm_y.
    """
    phi = np.arctan2(q[..., 2], q[..., 1])
    r = np.ones(q.shape[:-1])
    if planar:
        for j, (k, kind) in enumerate(terms):
            r = r + coefficients[..., j] * (np.cos(k * phi) if kind == "cos" else np.sin(k * phi))
        return r
    norm = _norm(q)
    ratio = np.divide(q[..., 0], norm, out=np.ones_like(norm), where=norm > 0)
    theta = np.arccos(np.clip(ratio, -1.0, 1.0))
    cache = {}
    for j, (l, m) in enumerate(terms):
        if (l, abs(m)) not in cache:
            cache[l, abs(m)] = sph_harm_y(l, abs(m), theta, phi)
        y = cache[l, abs(m)]
        value = y.real if m == 0 else math.sqrt(2) * (-1) ** abs(m) * (y.real if m > 0 else y.imag)
        r = r + coefficients[..., j] * value
    return r


def _shapes(centres, lobes, coefficients, terms, planar):
    """Stacked nucleus shapes: lobes per nucleus as (offset_zyx, axial, lateral, elongation, angle)."""
    count = len(lobes)
    width = max((len(own) for own in lobes), default=1)
    offset = np.zeros((count, width, 3))
    semi = np.ones((count, width, 3))
    cos, sin = np.ones((count, width)), np.zeros((count, width))
    on = np.zeros((count, width), dtype=bool)
    for n, own in enumerate(lobes):
        for l, (o, axial, lateral, elongation, angle) in enumerate(own):
            offset[n, l] = o
            semi[n, l] = (axial, lateral * elongation, lateral)
            cos[n, l], sin[n, l] = math.cos(angle), math.sin(angle)
            on[n, l] = True
    centre = np.asarray(centres, dtype=np.float64).reshape(-1, 3)
    harmonic = None
    if terms:
        harmonic = np.asarray(coefficients, dtype=np.float64).reshape(count, len(terms))
    return SimpleNamespace(centre=centre, offset=offset, lobe_centre=centre[:, None, :] + offset, semi=semi,
                           cos=cos, sin=sin, on=on, coefficients=harmonic, terms=terms, planar=planar)


def _nucleus_rho(points, index, shapes):
    """ρ_n = min over lobes of ρ_{n,l} at points (m, 3) for nucleus indices (m,); lobe 1 under N2 is ‖d'‖ / r."""
    rho = np.full(len(points), np.inf)
    for l in range(shapes.on.shape[1]):
        sel = np.flatnonzero(shapes.on[index, l])
        if not len(sel):
            continue
        i = index[sel]
        q = _frame(points[sel] - shapes.lobe_centre[i, l], shapes.cos[i, l], shapes.sin[i, l], shapes.semi[i, l])
        value = _norm(q)
        if l == 0 and shapes.coefficients is not None:
            value = value / _radius(q, shapes.coefficients[i], shapes.terms, shapes.planar)
        rho[sel] = np.minimum(rho[sel], value)
    return rho


def _grown_rho(points, index, shapes, growth):
    """ρ_n^{+δ}: the minimum over lobes of the ellipsoid with every semi-axis grown by δ (no harmonic factor)."""
    rho = np.full(len(points), np.inf)
    for l in range(shapes.on.shape[1]):
        sel = np.flatnonzero(shapes.on[index, l])
        if not len(sel):
            continue
        i = index[sel]
        q = _frame(points[sel] - shapes.lobe_centre[i, l], shapes.cos[i, l], shapes.sin[i, l],
                   shapes.semi[i, l] + growth[sel, None])
        rho[sel] = np.minimum(rho[sel], _norm(q))
    return rho


def _lobe_one_extent(shapes, index, directions):
    """e_{n,1}(û): the length of the ray c_n + s û, s ≥ 0, inside lobe 1 (times r(d̂'(û)) under N2)."""
    q = _frame(directions, shapes.cos[index, 0], shapes.sin[index, 0], shapes.semi[index, 0])
    out = 1.0 / _norm(q)
    if shapes.coefficients is not None:
        out = out * _radius(q, shapes.coefficients[index], shapes.terms, shapes.planar)
    return out


def _extent(shapes, index, directions):
    """e_n(û) for nuclei ``index`` (m,) along unit directions (m, 3): the end of the union of the lobes'
    intervals of s that contains s = 0 (lobe 1 contains the centre)."""
    end = _lobe_one_extent(shapes, index, directions)
    intervals = []
    for l in range(1, shapes.on.shape[1]):
        args = shapes.cos[index, l], shapes.sin[index, l], shapes.semi[index, l]
        q, p = _frame(directions, *args), _frame(shapes.offset[index, l], *args)
        a = (q * q).sum(axis=-1)
        b = (q * p).sum(axis=-1)
        c = (p * p).sum(axis=-1) - 1
        disc = b * b - a * c
        root = np.sqrt(np.maximum(disc, 0.0))
        intervals.append(((b - root) / a, (b + root) / a, shapes.on[index, l] & (disc >= 0)))
    for _ in intervals:
        for start, stop, valid in intervals:
            end = np.where(valid & (start <= end) & (stop > end), stop, end)
    return end


def _contact(e_i, e_k, delta):
    """t_ik(û) = min(e_i + sqrt(e_i² − Δ), e_k + sqrt(e_k² + Δ)), 0 where a radicand is negative."""
    a, b = e_i * e_i - delta, e_k * e_k + delta
    t = np.minimum(e_i + np.sqrt(np.maximum(a, 0.0)), e_k + np.sqrt(np.maximum(b, 0.0)))
    return np.where((a >= 0) & (b >= 0), t, 0.0)


def _labels(points, sites, omega, block):
    """C and N of the cellular scene at (..., 3) sample positions: the power-nearest site (ties to the smaller
    index), σ in Ω, the territory test of the rule (reach T1, grown T2, power T3) and ρ_n ≤ 1."""
    shape = points.shape[:-1]
    flat = points.reshape(-1, 3)
    cells = np.zeros(len(flat), dtype=np.uint32)
    nuclei = np.zeros(len(flat), dtype=np.uint32)
    if not len(sites.centres):
        return cells.reshape(shape), nuclei.reshape(shape)
    lo, hi = omega
    for start in range(0, len(flat), block):
        part = flat[start:start + block]
        distance = None
        for a in range(3):
            delta = part[:, a, None] - sites.centres[None, :, a]
            distance = delta * delta if distance is None else distance + delta * delta
        nearest = np.argmin(distance - sites.radii * sites.radii, axis=1)
        rows = np.arange(len(part))
        inside = np.all((part >= lo) & (part < hi), axis=-1)
        owner = sites.nucleus[nearest]
        rho = np.full(len(part), np.inf)
        has = np.flatnonzero(inside & (owner > 0))
        if len(has):
            rho[has] = _nucleus_rho(part[has], owner[has].astype(np.int64) - 1, sites.shapes)
        if sites.rule == "reach":
            territory = np.sqrt(distance[rows, nearest]) <= sites.radii[nearest] + sites.reach[nearest]
        elif sites.rule == "grown":
            territory = rho <= 1
            grow = np.flatnonzero(inside & (owner > 0) & ~territory)
            if len(grow):
                territory[grow] = _grown_rho(part[grow], owner[grow].astype(np.int64) - 1, sites.shapes,
                                             sites.growth[nearest[grow]]) <= 1
            empty = np.flatnonzero(inside & (owner == 0))
            territory[empty] = (np.sqrt(distance[empty, nearest[empty]])
                                <= sites.empty_radius + sites.growth[nearest[empty]])
        else:
            territory = np.ones(len(part), dtype=bool)
        hit = inside & territory
        cells[start:start + len(part)] = np.where(hit, sites.cell[nearest], 0)
        nuclei[start:start + len(part)] = np.where(hit & (rho <= 1), owner, 0)
    return cells.reshape(shape), nuclei.reshape(shape)
