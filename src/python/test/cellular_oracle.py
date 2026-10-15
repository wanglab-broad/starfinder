"""Test oracle of the cellular scene (docs/synthetic-cellular-scene-validation.md, "Two test helpers").

Written from the formulas of docs/synthetic-cellular-scene.md and the truth tables alone: it
imports nothing from ``starfinder.synthetic`` and reads a scene through its tables
(``nuclei``, ``lobes``, ``harmonics``, ``cells``, ``cell_context``) and its two records.
It recomputes the nucleus and cell labels ``N`` and ``C`` with its own arithmetic and reports
the positions whose decision lies within 1e-9 µm of a boundary (a power face, a lobe or
harmonic surface, a territory limit, the edge of Ω), the extent of a nucleus along a ray and
the contact distance of two sites. ``StainOracle`` (W-484) adds the three stain signals with
their cut-offs, the bright regions and the field U from ``stain_objects``, ``bright_regions``,
``unevenness`` and the requested stain configurations, with every site in every face term.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
from scipy.special import erfc, sph_harm_y

# Decisions within this distance (µm) of 0 are boundary positions.
EPS = 1e-9


def real_harmonic(l, m, theta, phi):
    """Y_lm = sqrt(2) (−1)^m Re Y_l^m (m > 0), Y_l^0, sqrt(2) (−1)^m Im Y_l^|m| (m < 0); θ polar, φ azimuth."""
    y = sph_harm_y(l, abs(m), theta, phi)
    if m == 0:
        return y.real
    return math.sqrt(2) * (-1) ** abs(m) * (y.real if m > 0 else y.imag)


def harmonic_bound(degree, planar):
    """M_Y: 1 for cos kφ and sin kφ; on the sphere the largest abs(Y_lm) of degree ≤ L, sqrt((2L + 1) / 4π)."""
    return 1.0 if planar else math.sqrt((2 * degree + 1) / (4 * math.pi))


@dataclass
class Lobe:
    centre: np.ndarray   # c_n + o_l
    offset: np.ndarray
    semi: np.ndarray     # a_z, a_l e, a_l
    rotation: np.ndarray  # rows: z, u, v in Z, Y, X


@dataclass
class Nucleus:
    nucleus_id: int
    cell_id: int
    centre: np.ndarray
    site_radius: float
    lobes: list
    coefficients: dict   # (degree, order) -> a


def _rotation(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]])


class Oracle:
    """N, C and the boundary positions of one scene from its truth tables."""

    def __init__(self, scene):
        sample, record = scene.sample, scene.record
        effective = record["effective_config"]
        self.planar = bool(effective["planar"])
        self.rule = effective["territory_rule"]
        requested = sample.record["requested_config"]
        self.empty_radius = float(requested["cells"]["empty_radius"])
        domain = sample.record["domain"]["omega"]
        self.lo, self.hi = np.array(domain["lo_zyx"]), np.array(domain["hi_zyx"])
        harmonics = {}
        for row in scene.harmonics.itertuples():
            order = row.order if isinstance(row.order, str) else int(row.order)
            harmonics.setdefault(int(row.nucleus_id), {})[int(row.degree), order] = float(row.coefficient)
        lobes = {}
        for row in scene.lobes.sort_values(["nucleus_id", "lobe"]).itertuples():
            offset = np.array([row.offset_z, row.offset_y, row.offset_x])
            lobes.setdefault(int(row.nucleus_id), []).append((offset, row.axial_radius, row.lateral_radius,
                                                              row.elongation, row.angle))
        self.nuclei = {}
        for x in sample.nuclei.itertuples():
            centre = np.array([x.centre_z, x.centre_y, x.centre_x])
            own = [Lobe(centre + o, o, np.array([a, b * e, b]), _rotation(t)) for o, a, b, e, t in lobes[x.nucleus_id]]
            self.nuclei[int(x.nucleus_id)] = Nucleus(int(x.nucleus_id), int(x.cell_id), centre, float(x.site_radius),
                                                     own, harmonics.get(int(x.nucleus_id), {}))
        self.amplitude = {n: sum(abs(v) for v in x.coefficients.values()) for n, x in self.nuclei.items()}
        growth = dict(zip(scene.cell_context.cell_id.astype(int), scene.cell_context.growth))
        centres, radii, cells, nuclei, reach, grown = [], [], [], [], [], []
        for cell in sample.cells.itertuples():
            own = sorted((n for n in self.nuclei.values() if n.cell_id == cell.cell_id), key=lambda n: n.nucleus_id)
            sites = [(n.centre, n.site_radius, n.nucleus_id) for n in own] or [
                (np.array([cell.anchor_z, cell.anchor_y, cell.anchor_x]), self.empty_radius, 0)]
            for centre, radius, n in sites:
                centres.append(centre)
                radii.append(radius)
                cells.append(int(cell.cell_id))
                nuclei.append(n)
                reach.append(cell.reach)
                grown.append(growth.get(int(cell.cell_id), np.nan))
        self.centres = np.array(centres).reshape(-1, 3)
        self.radii, self.cells, self.site_nuclei = np.array(radii), np.array(cells), np.array(nuclei)
        self.reach, self.growth = np.array(reach), np.array(grown, dtype=float)

    # ------------------------------------------------------------------ shapes

    def _normalized(self, lobe, d, grow=0.0):
        return (d @ lobe.rotation.T) / (lobe.semi + grow)

    def radius(self, nucleus, q):
        """r(d̂') of lobe 1 at the directions of q (..., 3) in its normalized frame."""
        if not nucleus.coefficients:
            return np.ones(q.shape[:-1])
        phi = np.arctan2(q[..., 2], q[..., 1])
        r = np.ones(q.shape[:-1])
        if self.planar:
            for (k, kind), a in nucleus.coefficients.items():
                r += a * (np.cos(k * phi) if kind == "cos" else np.sin(k * phi))
            return r
        norm = np.linalg.norm(q, axis=-1)
        theta = np.arccos(np.clip(np.where(norm > 0, q[..., 0] / np.where(norm > 0, norm, 1), 1.0), -1, 1))
        for (l, m), a in nucleus.coefficients.items():
            r += a * real_harmonic(l, m, theta, phi)
        return r

    def lobe_rho(self, nucleus, l, points, grow=0.0):
        lobe = nucleus.lobes[l]
        q = self._normalized(lobe, points - lobe.centre, grow)
        value = np.linalg.norm(q, axis=-1)
        if l == 0 and nucleus.coefficients and np.all(grow == 0):
            value = value / self.radius(nucleus, q)
        return value

    def rho(self, nucleus_id, points, grow=0.0):
        nucleus = self.nuclei[nucleus_id]
        return np.min([self.lobe_rho(nucleus, l, points, grow) for l in range(len(nucleus.lobes))], axis=0)

    def _surface_margin(self, nucleus_id, points, grow=0.0):
        """True where a lobe's decision ρ_l = 1 lies within EPS µm (ρ is Lipschitz with 1 / a_min; under N2 a
        factor 10 covers the harmonic's slope)."""
        nucleus = self.nuclei[nucleus_id]
        near = np.zeros(len(points), dtype=bool)
        for l, lobe in enumerate(nucleus.lobes):
            factor = 10.0 if (l == 0 and nucleus.coefficients) else 1.0
            scale = float(np.min(lobe.semi + grow))
            near |= np.abs(self.lobe_rho(nucleus, l, points, grow) - 1) <= factor * EPS / scale
        return near

    # ------------------------------------------------------------------ labels

    def labels(self, points):
        """(C, N, boundary) at (..., 3) sample positions."""
        shape = points.shape[:-1]
        p = points.reshape(-1, 3)
        cell = np.zeros(len(p), dtype=np.int64)
        nucleus = np.zeros(len(p), dtype=np.int64)
        boundary = np.zeros(len(p), dtype=bool)
        if not len(self.centres):
            return cell.reshape(shape), nucleus.reshape(shape), boundary.reshape(shape)
        diff = p[:, None, :] - self.centres[None, :, :]
        squared = np.einsum("pjk,pjk->pj", diff, diff)
        power = squared - self.radii ** 2
        best = np.argmin(power, axis=1)
        rows = np.arange(len(p))
        if len(self.centres) > 1:
            gap = np.linalg.norm(self.centres[None, :, :] - self.centres[best][:, None, :], axis=-1)
            with np.errstate(divide="ignore", invalid="ignore"):
                face = (power - power[rows, best][:, None]) / (2 * gap)
            face[rows, best] = np.inf
            boundary |= np.abs(face).min(axis=1) <= EPS
        inside = np.all((p >= self.lo) & (p < self.hi), axis=1)
        boundary |= np.any(np.abs(p - self.lo) <= EPS, axis=1) | np.any(np.abs(p - self.hi) <= EPS, axis=1)
        owner = self.site_nuclei[best]
        rho = np.full(len(p), np.inf)
        for n in np.unique(owner[owner > 0]):
            mask = owner == n
            rho[mask] = self.rho(n, p[mask])
            boundary[mask] |= self._surface_margin(n, p[mask])
        distance = np.sqrt(squared[rows, best])
        if self.rule == "reach":
            limit = self.radii[best] + self.reach[best]
            territory = distance <= limit
            boundary |= np.abs(distance - limit) <= EPS
        elif self.rule == "grown":
            territory = rho <= 1
            for n in np.unique(owner[owner > 0]):
                mask = owner == n
                delta = self.growth[best[mask]]
                territory[mask] |= self.rho(n, p[mask], delta[:, None]) <= 1
                boundary[mask] |= self._surface_margin(n, p[mask], delta[:, None])
            empty = owner == 0
            limit = self.empty_radius + self.growth[best]
            territory[empty] = distance[empty] <= limit[empty]
            boundary[empty] |= np.abs(distance[empty] - limit[empty]) <= EPS
        else:
            territory = np.ones(len(p), dtype=bool)
        hit = inside & territory
        cell[hit] = self.cells[best][hit]
        nucleus[hit & (rho <= 1)] = owner[hit & (rho <= 1)]
        return cell.reshape(shape), nucleus.reshape(shape), boundary.reshape(shape)

    def label_power(self, point, sites=None):
        """Power distances of one point with the arithmetic K1 names (the label function's): the squares of the Z,
        Y and X differences summed in that order, less R²."""
        centres, radii = (self.centres, self.radii) if sites is None else (self.centres[sites], self.radii[sites])
        distance = (point[0] - centres[:, 0]) ** 2
        distance = distance + (point[1] - centres[:, 1]) ** 2
        distance = distance + (point[2] - centres[:, 2]) ** 2
        return distance - radii * radii

    def power_nearest(self, points):
        """The site index of smallest power distance (ties to the smaller index) at (m, 3) points."""
        diff = points[:, None, :] - self.centres[None, :, :]
        return np.argmin(np.einsum("pjk,pjk->pj", diff, diff) - self.radii ** 2, axis=1)

    # ------------------------------------------------------------------ extents and contact distances

    def lobe_one_extent(self, nucleus_id, u):
        nucleus = self.nuclei[nucleus_id]
        q = self._normalized(nucleus.lobes[0], np.asarray(u, dtype=float))
        return float(self.radius(nucleus, q[None])[0] / np.linalg.norm(q))

    def lobe_extent(self, nucleus_id, l, u):
        """ε_l(û): the length of the ray from lobe l's own centre along û inside lobe l."""
        lobe = self.nuclei[nucleus_id].lobes[l]
        return float(1.0 / np.linalg.norm(self._normalized(lobe, np.asarray(u, dtype=float))))

    def extent(self, nucleus_id, u):
        """e_n(û): the end of the union of the lobes' intervals of s along c_n + s û that contains s = 0."""
        nucleus = self.nuclei[nucleus_id]
        u = np.asarray(u, dtype=float)
        end = self.lobe_one_extent(nucleus_id, u)
        intervals = []
        for lobe in nucleus.lobes[1:]:
            a_vec = self._normalized(lobe, u)
            b_vec = self._normalized(lobe, lobe.offset)
            a, b, c = a_vec @ a_vec, a_vec @ b_vec, b_vec @ b_vec - 1
            disc = b * b - a * c
            if disc >= 0:
                intervals.append(((b - math.sqrt(disc)) / a, (b + math.sqrt(disc)) / a))
        changed = True
        while changed:
            changed = False
            for start, stop in intervals:
                if start <= end < stop:
                    end, changed = stop, True
        return end

    def site_extent(self, site, u):
        n = int(self.site_nuclei[site])
        return self.empty_radius if n == 0 else self.extent(n, u)

    def contact_distance(self, i, k, u):
        """t_ik(û) for sites i and k, with û from c_i towards c_k."""
        delta = self.radii[i] ** 2 - self.radii[k] ** 2
        e_i, e_k = self.site_extent(i, u), self.site_extent(k, -np.asarray(u))
        if e_i ** 2 - delta < 0 or e_k ** 2 + delta < 0:
            return 0.0
        return min(e_i + math.sqrt(e_i ** 2 - delta), e_k + math.sqrt(e_k ** 2 + delta))


# ---------------------------------------------------------------------- grids

def sample_grid_points(scene):
    grid = scene.sample.record["sample_grid"]
    q = np.moveaxis(np.indices(tuple(grid["shape_zyx"]), dtype=float), 0, -1)
    return np.array(grid["origin_zyx"]) + q * np.array(grid["spacing_zyx"])


def fov_grid_points(scene, fov):
    row = scene.sample.placement.set_index("fov_id").loc[fov]
    true = np.array([row.true_origin_z, row.true_origin_y, row.true_origin_x])
    spacing = np.array([row.spacing_z, row.spacing_y, row.spacing_x])
    shape = (int(row.shape_z), int(row.shape_y), int(row.shape_x))
    return true + np.moveaxis(np.indices(shape, dtype=float), 0, -1) * spacing


def grids(scene):
    """(name, points, labels or None) for the sample grid and every FOV grid of a scene."""
    sample = scene.sample
    result = [("sample", sample_grid_points(scene), sample.sample_labels)]
    for fov in sample.positions.fov_id:
        result.append((fov, fov_grid_points(scene, fov), sample.fov_labels.get(fov)))
    return result


# ---------------------------------------------------------------------- identities 1 to 5 of the sample scene

def check_sample_identities(scene, oracle):
    """Identities 1 to 5 of docs/synthetic-sample-scene.md on the scene's tables with the cellular labels.

    Returns the number of boundary positions met (molecules and anchors), which the caller reports.
    """
    sample = scene.sample
    molecules, cells, nuclei = sample.molecules, sample.cells, sample.nuclei
    sigma = molecules[["sample_z", "sample_y", "sample_x"]].to_numpy()
    covered = sample.record["domain"]["covered"]
    k_lo, k_hi = np.array(covered["lo_zyx"]), np.array(covered["hi_zyx"])
    inside = lambda points: np.all((points >= k_lo) & (points < k_hi), axis=-1)  # noqa: E731
    fov_ids = list(sample.positions.fov_id)
    np.testing.assert_array_equal(molecules.in_sample, inside(sigma))
    anchors = cells[["anchor_z", "anchor_y", "anchor_x"]].to_numpy()
    np.testing.assert_array_equal(cells.in_sample, inside(anchors))
    cell_of, nucleus_of, near = oracle.labels(sigma)
    ok = ~near
    np.testing.assert_array_equal(molecules.true_cell_id.fillna(0).astype(int)[ok], cell_of[ok])
    np.testing.assert_array_equal(molecules.true_nucleus_id.fillna(0).astype(int)[ok], nucleus_of[ok])
    expected = np.where(nucleus_of > 0, "nucleus", np.where(cell_of > 0, "cytoplasm", "outside"))
    assert molecules.compartment.to_numpy(dtype=object)[ok].tolist() == expected[ok].tolist()
    # 1. one cell or none; the nucleus of that cell; membership and counts.
    owner = dict(zip(nuclei.nucleus_id, nuclei.cell_id))
    for row in molecules.dropna(subset=["true_nucleus_id"]).itertuples():
        assert owner[row.true_nucleus_id] == row.true_cell_id
    membership = dict(zip(cells.cell_id, cells.in_sample))
    expected_membership = [membership[c] if pd.notna(c) else None for c in molecules.true_cell_id]
    assert [None if pd.isna(v) else bool(v) for v in molecules.true_cell_in_sample] == expected_membership
    counts = molecules.true_cell_id.value_counts()
    assert cells.n_molecules.tolist() == [int(counts.get(k, 0)) for k in cells.cell_id]
    # 2. labels: compared by the caller (GN4) on every grid; voxel counts of the sample cells here.
    sample_voxels = np.bincount(sample.sample_labels["cell"].ravel().astype(np.int64), minlength=len(cells) + 1)
    assert cells.n_voxels.tolist() == [int(sample_voxels[k]) for k in cells.cell_id]
    # 3. coverage and multiplicity from the placement truth.
    seen = np.zeros((len(molecules), len(fov_ids)), dtype=bool)
    for f, row in enumerate(sample.placement.itertuples()):
        spacing = np.array([row.spacing_z, row.spacing_y, row.spacing_x])
        shape = np.array([row.shape_z, row.shape_y, row.shape_x])
        p = (sigma - np.array([row.true_origin_z, row.true_origin_y, row.true_origin_x])) / spacing
        seen[:, f] = np.all((p >= -.5) & (p < shape - .5), axis=1)
    np.testing.assert_array_equal(molecules.n_fovs, seen.sum(axis=1))
    assert molecules.fov_ids.fillna("").tolist() == [";".join(np.array(fov_ids)[s]) for s in seen]
    assert (seen.sum(axis=1)[molecules.in_sample.to_numpy()] >= 1).all()
    # 4. cells by their anchors: one cell_fovs row per FOV whose footprint holds the anchor or a label voxel.
    for row in sample.placement.itertuples():
        spacing = np.array([row.spacing_z, row.spacing_y, row.spacing_x])
        shape = np.array([row.shape_z, row.shape_y, row.shape_x])
        p = (anchors - np.array([row.true_origin_z, row.true_origin_y, row.true_origin_x])) / spacing
        anchored = set(cells.cell_id[np.all((p >= -.5) & (p < shape - .5), axis=1)])
        labels = sample.fov_labels[row.fov_id]["cell"]
        own = sample.cell_fovs[sample.cell_fovs.fov_id == row.fov_id]
        assert set(own.cell_id) == anchored | (set(np.unique(labels)) - {0})
        for r in own.itertuples():
            assert r.n_voxels == int((labels == r.cell_id).sum())
            assert r.anchor_in_footprint == (r.cell_id in anchored)
    # 5. sampled labels against the truth, by the nullable-ID rule.
    truth = dict(zip(molecules.amplicon_id, molecules.true_cell_id.fillna(0).astype(int)))
    rows = sample.molecule_fovs
    for row in sample.placement.itertuples():
        own = rows[rows.fov_id == row.fov_id]
        shape = np.array([row.shape_z, row.shape_y, row.shape_x])
        p = own[["z", "y", "x"]].to_numpy()
        footprint = np.all((p >= -.5) & (p < shape - .5), axis=1)
        np.testing.assert_array_equal(own.in_footprint, footprint)
        voxel = np.floor(p[footprint] + .5).astype(int)
        labels = sample.fov_labels[row.fov_id]
        at = tuple(voxel.T)
        np.testing.assert_array_equal(own.fov_cell_label[footprint].astype(int), labels["cell"][at])
        np.testing.assert_array_equal(own.fov_nucleus_label[footprint].astype(int), labels["nucleus"][at])
        np.testing.assert_array_equal(own.label_agrees[footprint].astype(bool),
                                      [truth[a] == v for a, v in zip(own.amplicon_id[footprint], labels["cell"][at])])
    return int(near.sum())


# ---------------------------------------------------------------------- stains (W-484)

# ½ erfc(2√2): the largest change of an edge whose argument stays at or below −4w (the site-skipping rule, choice V9).
TAIL = 0.5 * math.erfc(2 * math.sqrt(2))


class StainOracle:
    """The signal of one stain of a scene from its tables and the requested stain configuration ("Stains").

    Every face term takes every other site (no site is skipped). ``signal`` returns the pre-noise signal, the
    positions within EPS µm of a cut-off surface (``ρ_n = 1 + 4w / a_min``, ``h_n = 4w``, the territory's
    ``d_k = 4w``, the band's ``β_k = 4b``, the box of ``4 s_b`` of a bright region) and, for the nucleus source, the
    V9 allowance ``½ erfc(2√2) · A_n U (1 + T_n)`` summed over the nuclei whose argument is at most −4w.
    """

    def __init__(self, oracle, scene, name):
        self.o = oracle
        config = next(r for r in scene.record["requested_config"]["stains"] if r["name"] == name)
        self.name, self.source = name, config["source"]
        self.w, self.b = float(config["edge_width"]), float(config["band_width"])
        self.q = float(config["nuclear_ratio"])
        own = scene.stain_objects[scene.stain_objects.stain == name]
        self.plateau = dict(zip(own.object_id.astype(int).tolist(), own.plateau.astype(float).tolist()))
        self.regions = {}
        for row in scene.bright_regions[scene.bright_regions.stain == name].itertuples():
            self.regions.setdefault(int(row.nucleus_id), []).append(
                (np.array([row.centre_z, row.centre_y, row.centre_x]), float(row.size), float(row.height)))
        bumps = scene.unevenness[scene.unevenness.stain == name]
        self.bumps = bumps[["centre_z", "centre_y", "centre_x"]].to_numpy(dtype=float)
        self.z = bumps.z.to_numpy(dtype=float)
        self.amplitude = None if config["unevenness"] is None else float(config["unevenness"]["amplitude"])
        self.length = float(bumps.length.iloc[0]) if len(bumps) else None
        self.site_of = {int(n): j for j, n in enumerate(oracle.site_nuclei) if n}
        self.cell_sites = {}
        for j, k in enumerate(oracle.cells):
            self.cell_sites.setdefault(int(k), []).append(j)

    # ------------------------------------------------------------------ terms

    def powers(self, p):
        diff = p[:, None, :] - self.o.centres[None, :, :]
        return np.einsum("pjk,pjk->pj", diff, diff) - self.o.radii ** 2

    def faces(self, power, i, others):
        """max over others of (pow_i − pow_j) / (2 ‖c_i − c_j‖); −∞ without a site."""
        if not len(others):
            return np.full(len(power), -np.inf)
        gaps = np.linalg.norm(self.o.centres[others] - self.o.centres[i], axis=1)
        return ((power[:, [i]] - power[:, others]) / (2 * gaps)).max(axis=1)

    def _lobe_margin(self, n, p, level, grow=0.0):
        nucleus = self.o.nuclei[n]
        near = np.zeros(len(p), dtype=bool)
        for l, lobe in enumerate(nucleus.lobes):
            factor = 10.0 if (l == 0 and nucleus.coefficients and grow == 0) else 1.0
            near |= np.abs(self.o.lobe_rho(nucleus, l, p, grow) - level) <= factor * EPS * level / float(
                np.min(lobe.semi + grow))
        return near

    def rbar(self, n):
        return float(np.prod(self.o.nuclei[n].lobes[0].semi) ** (1 / 3))

    def nuclei(self, p, power, w):
        """Per nucleus n: (argument, E_n, near) with edge w."""
        result = {}
        sites = np.arange(len(self.o.centres))
        for n, nucleus in self.o.nuclei.items():
            i = self.site_of[n]
            h = self.faces(power, i, sites[sites != i])
            rho = self.o.rho(n, p)
            amin = float(nucleus.lobes[0].semi.min())
            cutoff = 1 + 4 * w / amin
            argument = np.maximum((rho - 1) * self.rbar(n), h)
            edge = np.where((rho > cutoff) | (h > 4 * w), 0.0, .5 * erfc(argument / (math.sqrt(2) * w)))
            near = (np.abs(h - 4 * w) <= EPS) | self._lobe_margin(n, p, cutoff)
            result[n] = (argument, edge, near)
        return result

    def outer(self, p, j):
        o = self.o
        if o.rule == "power":
            return np.full(len(p), -np.inf)
        distance = np.linalg.norm(p - o.centres[j], axis=1)
        if o.rule == "reach":
            return distance - (o.radii[j] + o.reach[j])
        n = int(o.site_nuclei[j])
        if not n:
            return distance - (o.empty_radius + o.growth[j])
        rbar = self.rbar(n)
        return np.minimum((o.rho(n, p) - 1) * rbar, (o.rho(n, p, o.growth[j]) - 1) * rbar)

    def cell(self, p, power, k):
        """(d_k, β_k) of cell k at p: d_k = max(h*_k, o_{j_k}); β_k = min(abs(d_k), g_k)."""
        own = self.cell_sites[k]
        others = np.array([j for j in range(len(self.o.centres)) if j not in own], dtype=int)
        pick = np.argmin(power[:, own], axis=1)
        rows = np.arange(len(p))
        star = np.stack([self.faces(power, j, others) for j in own], axis=1)[rows, pick]
        outer = np.stack([self.outer(p, j) for j in own], axis=1)
        d = np.maximum(star, outer[rows, pick])
        beta = np.abs(d)
        if len(own) == 2:
            j, jj = own
            internal = (power[:, j] - power[:, jj]) / (2 * np.linalg.norm(self.o.centres[j] - self.o.centres[jj]))
            g = np.max([np.abs(internal), star, outer.min(axis=1), -outer.max(axis=1)], axis=0)
            beta = np.minimum(beta, g)
        return d, beta

    def texture(self, n, p):
        """(T_n, near) at p: the bright regions of nucleus n, each 0 outside its box of 4 s_b."""
        total = np.zeros(len(p))
        near = np.zeros(len(p), dtype=bool)
        for centre, size, height in self.regions.get(n, ()):
            chebyshev = np.abs(p - centre).max(axis=1)
            squared = (((p - centre) / size) ** 2).sum(axis=1)
            total += np.where(chebyshev <= 4 * size, height * np.exp(-squared / 2), 0.0)
            near |= np.abs(chebyshev - 4 * size) <= EPS
        return total, near

    def field(self, p):
        """U1 at p, normalized: log U = a Σ z_q w_q / sqrt(Σ w_q²); 1 with no centre."""
        if self.amplitude is None or not len(self.bumps):
            return np.ones(len(p))
        diff = p[:, None, :] - self.bumps[None, :, :]
        squared = np.einsum("pjk,pjk->pj", diff, diff)
        weights = np.exp(-(squared - squared.min(axis=1, keepdims=True)) / (2 * self.length ** 2))
        return np.exp(self.amplitude * (weights @ self.z) / np.sqrt((weights ** 2).sum(axis=1)))

    # ------------------------------------------------------------------ the signal

    def signal(self, points):
        """(signal, near, allowance) at (..., 3) positions."""
        shape = points.shape[:-1]
        p = points.reshape(-1, 3)
        out, near, allowance = np.zeros(len(p)), np.zeros(len(p), dtype=bool), np.zeros(len(p))
        if len(self.o.centres):
            power = self.powers(p)
            if self.source == "nucleus":
                for n, (argument, edge, close) in self.nuclei(p, power, self.w).items():
                    texture, box = self.texture(n, p)
                    out += self.plateau[n] * edge * (1 + texture)
                    near |= close | (box & (edge > 0))
                    allowance += np.where(argument <= -4 * self.w, TAIL * self.plateau[n] * (1 + texture), 0.0)
            elif self.source == "territory":
                terms = self.nuclei(p, power, self.w)
                indicator = np.minimum(1.0, sum(edge for _, edge, _ in terms.values()))
                for _, _, close in terms.values():
                    near |= close
                for k in self.cell_sites:
                    d, _ = self.cell(p, power, k)
                    edge = np.where(d > 4 * self.w, 0.0, .5 * erfc(d / (math.sqrt(2) * self.w)))
                    out += self.plateau[k] * edge * (1 - (1 - self.q) * indicator)
                    near |= np.abs(d - 4 * self.w) <= 20 * EPS
            else:
                for k in self.cell_sites:
                    _, beta = self.cell(p, power, k)
                    out += self.plateau[k] * np.where(beta > 4 * self.b, 0.0, np.exp(-(beta / self.b) ** 2 / 2))
                    near |= np.abs(beta - 4 * self.b) <= 20 * EPS
        field = self.field(p)
        return (out * field).reshape(shape), near.reshape(shape), (allowance * field).reshape(shape)
