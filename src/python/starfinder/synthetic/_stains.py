"""Stains and the morphology round of the cellular scene (design/specs/synthetic/cellular-scene.md, "Stains", "The morphology
round").

Every stain is a point function of the sample position, built from the scene's truth tables: the nucleus, territory
and boundary sources share the power faces and the territory tests of the labels (``_cellular._labels``), the bright
regions (Tx1) and the unevenness field (U1) are closed forms with centres in µm, and a FOV or the morphology round
evaluates them at its own voxel centres, without a label image, a distance transform or resampling. The generator and
the point evaluator call the same functions (``_Model``), so the evaluator reproduces the rendered signal.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
from types import SimpleNamespace

import numpy as np
from scipy.special import erfc

from ._cellular import _grown_rho, _labels, _nucleus_rho
from ._common import ScalarDistribution, _draw, _label
from ._observation import BackgroundConfig, NoiseConfig, _noise
from ._sample import _BLOCK, _checked_distribution, _lognormal, _real, _triple

_SOURCES = ("nucleus", "territory", "boundary")
_ROUNDS = ("reference", "morphology")
_OUTPUTS = ("uint8", "uint16")
# A face term below −6√2 w over an object's box changes no value of its edge: ½ erfc(x) is exactly 1 in float64 for
# x ≤ −5.9, so such a site is skipped from the territory source and the soft nucleus indicator M without any change.
# The nucleus source skips at the page's −4w ("Stains"), which changes E_n by at most ½ erfc(2√2) of the plateau.
_EXACT = 6 * math.sqrt(2)
# Slack (µm) of the boxes and of the skip test; a box only selects the voxels an object can light.
_SLACK = 1e-6
# Rows of positions per block of face terms.
_ROWS = 1 << 15


def _plain_name(value, name):
    try:
        _label(value)
    except ValueError:
        raise ValueError(f"{name} must be a nonempty NFC string") from None
    if value in (".", "..") or "/" in value or "\\" in value:
        raise ValueError(f"{name} must be a plain file name; got {value!r}")
    return value


@dataclass(frozen=True)
class BrightRegionConfig:
    """Bright regions of a nucleus stain (Tx1), µm.

    A nucleus has ``Poisson(per_nucleus)`` regions, each centred uniformly inside the nucleus
    (by rejection on its label), with size ``s_b`` drawn from ``size`` and height ``h_b`` from
    ``height``; the nucleus's plateau is multiplied by ``1 + Σ_b h_b exp(−½ ‖(σ − c_b) / s_b‖²)``,
    each Gaussian set to 0 outside its box of half-width ``4 s_b``. Instances are not hashable.
    """

    __hash__ = None

    per_nucleus: float = 2.0
    size: ScalarDistribution = field(default_factory=lambda: _lognormal(0.3, 0.4))
    height: ScalarDistribution = field(default_factory=lambda: _lognormal(1.0, 0.3))

    def __post_init__(self):
        _real(self.per_nucleus, "per_nucleus")
        for name in ("size", "height"):
            _checked_distribution(getattr(self, name), "width.lateral", (), name)


@dataclass(frozen=True)
class UnevennessConfig:
    """A smooth multiplicative unevenness of a stain across the sample (U1), µm.

    ``K_U ~ Poisson(centres_per_area · A_Ω / A_FOV)`` centres lie uniformly in the Y-X box of
    the scene domain at the z of its centre, each with a standard normal ``z_q``; with one
    length ``L`` (``length``, None: the larger of the FOV's Y and X widths),
    ``log U = amplitude · Σ_q z_q w_q / sqrt(Σ_q w_q²)`` with Gaussian weights ``w_q``, so that
    ``log U`` has standard deviation ``amplitude`` at every point. With no centre ``U = 1``.
    Instances are not hashable.
    """

    __hash__ = None

    centres_per_area: float = 4.0
    amplitude: float = 0.2
    length: float | None = None

    def __post_init__(self):
        _real(self.centres_per_area, "centres_per_area")
        _real(self.amplitude, "amplitude")
        if self.length is not None:
            _real(self.length, "length", strict=True)


@dataclass(frozen=True)
class OutputScaleConfig:
    """The 8-bit output of a stain (B1): scale, then round and clip.

    After the noise the float64 image is multiplied by ``k = plateau / m_A``, ``m_A`` the
    median of the stain's brightness law, then rounded half to even and clipped to the range
    of ``dtype`` (``uint8`` or ``uint16``); ``plateau`` lies strictly between 0 and the
    largest value of ``dtype``. Instances are not hashable.
    """

    __hash__ = None

    dtype: str = "uint8"
    plateau: float = 50.0

    def __post_init__(self):
        if self.dtype not in _OUTPUTS:
            raise ValueError(f"dtype must be one of {list(_OUTPUTS)}; got {self.dtype!r}")
        plateau = _real(self.plateau, "plateau", strict=True)
        top = np.iinfo(self.dtype).max
        if not plateau < top:
            raise ValueError(f"plateau must satisfy 0 < plateau < {top}; got {plateau}")


@dataclass(frozen=True)
class StainChannelConfig:
    """One stain channel of the cellular scene (S1): a geometry source and its appearance.

    ``source`` is ``nucleus`` (nucleus interiors with a soft edge of width ``edge_width`` µm
    across the surface and the faces between touching nuclei, optional bright regions
    ``texture``), ``territory`` (cell territories, dimmer inside the nucleus by
    ``nuclear_ratio``) or ``boundary`` (a band of width ``band_width`` µm on every boundary of
    a cell). ``brightness`` is the plateau per object (nucleus or cell). ``round`` is
    ``reference`` (a stain file of the reference round) or ``morphology`` (a channel of the
    morphology round); ``channel`` is its file pattern. ``unevenness``, ``background`` (in
    the sample grid's index units) and ``noise`` follow the sample scene's stain; ``output``
    None casts to the template dtype as today. Stains that share a name share their draws.
    Instances are not hashable.
    """

    __hash__ = None

    name: str
    source: str = "nucleus"
    round: str = "reference"
    channel: str = "ch04"
    brightness: ScalarDistribution = field(default_factory=lambda: _lognormal(1000, 0.2))
    edge_width: float = 0.5
    band_width: float = 0.5
    nuclear_ratio: float = 0.3
    texture: BrightRegionConfig | None = None
    unevenness: UnevennessConfig | None = None
    background: BackgroundConfig = field(
        default_factory=lambda: BackgroundConfig(baseline_enabled=True, baseline=((100.0,),)))
    noise: NoiseConfig = field(default_factory=lambda: NoiseConfig(
        dependent_enabled=True, model="poisson", alpha=1.0, independent_enabled=True, sigma=3.0))
    output: OutputScaleConfig | None = None

    def __post_init__(self):
        _plain_name(self.name, "name")
        if self.source not in _SOURCES:
            raise ValueError(f"source must be one of {list(_SOURCES)}; got {self.source!r}")
        if self.round not in _ROUNDS:
            raise ValueError(f"round must be one of {list(_ROUNDS)}; got {self.round!r}")
        _plain_name(self.channel, "channel")
        spec = self.brightness
        ids = tuple(spec.values) if isinstance(spec, ScalarDistribution) and isinstance(spec.values, dict) else ()
        _checked_distribution(spec, "brightness", ids, "brightness")
        _real(self.edge_width, "edge_width", strict=True)
        _real(self.band_width, "band_width", strict=True)
        _real(self.nuclear_ratio, "nuclear_ratio")
        if self.texture is not None:
            if not isinstance(self.texture, BrightRegionConfig):
                raise TypeError("texture must be BrightRegionConfig or None")
            if self.source != "nucleus":
                raise ValueError(f"texture: bright regions belong to a nucleus stain; {self.name!r} has source "
                                 f"{self.source!r}")
        if self.unevenness is not None and not isinstance(self.unevenness, UnevennessConfig):
            raise TypeError("unevenness must be UnevennessConfig or None")
        if not isinstance(self.background, BackgroundConfig):
            raise TypeError("background must be BackgroundConfig")
        _noise(self.noise)
        if self.output is not None and not isinstance(self.output, OutputScaleConfig):
            raise TypeError("output must be OutputScaleConfig or None")


@dataclass(frozen=True)
class MorphologyRoundConfig:
    """The morphology round: stain channels acquired with a known translation per FOV.

    The round ``name`` holds the stains whose ``round`` is ``morphology``; its index ``q`` in
    FOV ``f`` images the sample position ``true_f + τ_f + q ⊙ s``. ``translation="uniform"``
    (MT1) draws ``τ_{f,a} = T_a (2u − 1) s_a`` with ``T`` = ``translation_max_zyx`` (voxels, 0 in
    z for one plane); ``"supplied"`` takes ``translation_zyx`` (voxels per FOV ID, within the
    bound). The scene domain grows by ``T ⊙ s`` on each side. Instances are not hashable.
    """

    __hash__ = None

    name: str = "organelle"
    translation: str = "uniform"
    translation_max_zyx: tuple[float, float, float] = (0.5, 6.0, 6.0)
    translation_zyx: Mapping[str, tuple[float, float, float]] | None = None

    def __post_init__(self):
        _plain_name(self.name, "name")
        bound = _triple(self.translation_max_zyx, "translation_max_zyx")
        if (bound < 0).any():
            raise ValueError(f"translation_max_zyx must be finite and ≥ 0; got {tuple(self.translation_max_zyx)}")
        if self.translation == "uniform":
            if self.translation_zyx is not None:
                raise ValueError("translation_zyx requires translation='supplied'")
        elif self.translation == "supplied":
            if not isinstance(self.translation_zyx, Mapping):
                raise ValueError("translation_zyx must map every FOV ID to a ZYX triple with translation='supplied'")
            for fov, value in self.translation_zyx.items():
                values = _triple(value, f"translation_zyx[{fov!r}]")
                for axis, v, limit in zip("zyx", values, bound):
                    if abs(v) > limit:
                        raise ValueError(f"translation_zyx[{fov!r}] exceeds the {axis} bound: |{v}| > {limit}")
        else:
            raise ValueError(f"translation must be 'uniform' or 'supplied'; got {self.translation!r}")


def _median(spec, values):
    """m_A of B1: the median of a brightness law (of the supplied values in supplied mode)."""
    p = spec.parameters
    if spec.mode == "constant":
        return float(p[0])
    if spec.mode == "uniform":
        return float((p[0] + p[1]) / 2)
    if spec.mode == "lognormal":
        return float(math.exp(p[0]))
    if not values:
        raise ValueError("output: the median of a supplied brightness law with no value is undefined")
    return float(np.median(list(values)))


# ---------------------------------------------------------------------------------------------- positions


class _Points:
    """Sample positions of a stain evaluation: a flat (m, 3) array, or the voxel centres ``origin + p ⊙ s`` of a grid
    of ``shape``, of which each object forms only the voxels of its box (with the arithmetic of the whole grid)."""

    def __init__(self, flat=None, *, origin=None, spacing=None, shape=None):
        self.grid = flat is None
        if self.grid:
            self.origin = np.asarray(origin, dtype=np.float64)
            self.spacing = np.asarray(spacing, dtype=np.float64)
            self.shape = tuple(int(v) for v in shape)
        else:
            self.flat = np.asarray(flat, dtype=np.float64).reshape(-1, 3)
            self.shape = (len(self.flat),)

    def zeros(self):
        return np.zeros(self.shape, dtype=np.float64)

    def all(self):
        if not self.grid:
            return self.flat
        index = np.moveaxis(np.indices(self.shape, dtype=np.float64), 0, -1)
        return (self.origin + index * self.spacing).reshape(-1, 3)

    def box(self, lo, hi):
        """(key, points, inside) of the box [lo, hi] (µm): a key into an array of the positions' shape, the candidate
        positions (m, 3) and whether each lies in the box; None when no position does."""
        if not self.grid:
            where = np.flatnonzero(np.all((self.flat >= lo) & (self.flat <= hi), axis=1))
            if not len(where):
                return None
            return where, self.flat[where], np.ones(len(where), dtype=bool)
        with np.errstate(invalid="ignore", over="ignore"):
            a = np.clip(np.floor((lo - self.origin) / self.spacing), 0, self.shape).astype(np.int64)
            b = np.clip(np.ceil((hi - self.origin) / self.spacing) + 1, 0, self.shape).astype(np.int64)
        if (b <= a).any():
            return None
        index = np.moveaxis(np.indices(tuple(b - a), dtype=np.float64), 0, -1) + a
        points = (self.origin + index * self.spacing).reshape(-1, 3)
        inside = np.all((points >= lo) & (points <= hi), axis=1)
        if not inside.any():
            return None
        return tuple(slice(i, j) for i, j in zip(a, b)), points, inside


def _add(out, key, inside, values):
    """out[box] += values at the positions of the box that lie inside it."""
    if isinstance(key, tuple):
        full = np.zeros(inside.shape, dtype=np.float64)
        full[inside] = values
        out[key] += full.reshape(out[key].shape)
    else:
        out[key] += values


def _gather(array, key, inside):
    if isinstance(key, tuple):
        return array[key].reshape(-1)[inside]
    return array[key]


def _squared(points, centres):
    """Squared distances (m, k), summed z, y, x as the label function sums them."""
    distance = None
    for a in range(3):
        delta = points[:, a, None] - centres[None, :, a]
        distance = delta * delta if distance is None else distance + delta * delta
    return distance


# ---------------------------------------------------------------------------------------------- the stain functions


class _Model:
    """The stain functions of one scene: its label sites (``_context._sites``) and, per stain name, the source, the
    widths, the plateau of each object, the bright regions and the unevenness field."""

    def __init__(self, sites, stains):
        self.stains = stains
        self.sites = sites if sites is not None and len(sites.centres) else None
        if self.sites is None:
            return
        centres, radii = sites.centres, sites.radii
        self.gaps = np.sqrt(_squared(centres, centres))
        self.nucleus_row, self.rbar, self.amin, self.radius = {}, {}, {}, {}
        for row, n in enumerate(sites.nucleus.tolist()):
            if n:
                semi = sites.shapes.semi[n - 1, 0]
                self.nucleus_row[n] = row
                self.rbar[n] = float(np.prod(semi) ** (1 / 3))
                self.amin[n] = float(semi.min())
                self.radius[n] = float(radii[row])
        self.cell_rows = {}
        for row, k in enumerate(sites.cell.tolist()):
            self.cell_rows.setdefault(k, []).append(row)

    # ------------------------------------------------------------------ face terms

    def _kept(self, i, candidates, lo, hi, threshold):
        """The candidate sites whose face term (pow_i − pow_j) / (2 ‖c_i − c_j‖) can reach ``threshold`` in the box:
        it is affine with unit slope, so its largest value is at the box centre plus the slope times the half-widths."""
        candidates = np.asarray(candidates, dtype=np.int64)
        if not len(candidates) or not (np.isfinite(lo).all() and np.isfinite(hi).all()):
            return candidates
        centres, radii = self.sites.centres, self.sites.radii
        mid, half = (lo + hi) / 2, (hi - lo) / 2
        rows = np.concatenate([[i], candidates])
        power = _squared(mid[None], centres[rows]) - radii[rows] * radii[rows]
        gap = self.gaps[i, candidates]
        slope = np.abs(centres[candidates] - centres[i]) / gap[:, None]
        top = (power[0, 0] - power[0, 1:]) / (2 * gap) + slope @ half
        return candidates[top >= threshold - _SLACK]

    def _faces(self, points, i, others):
        """max over sites j of others of (pow_i − pow_j) / (2 ‖c_i − c_j‖) at points (m, 3); −∞ without a site."""
        result = np.full(len(points), -np.inf)
        if not len(others):
            return result
        centres, radii = self.sites.centres, self.sites.radii
        rows = np.concatenate([[i], others])
        twice = 2 * self.gaps[i, others]
        for start in range(0, len(points), _ROWS):
            part = points[start:start + _ROWS]
            power = _squared(part, centres[rows]) - radii[rows] * radii[rows]
            result[start:start + _ROWS] = ((power[:, :1] - power[:, 1:]) / twice).max(axis=1)
        return result

    # ------------------------------------------------------------------ nuclei

    def _nucleus(self, points, n, width, threshold):
        """(key, inside, E_n, positions) of nucleus n with edge ``width``, its sites skipped below ``threshold``."""
        row = self.nucleus_row[n]
        centre = self.sites.centres[row]
        cutoff = 1 + 4 * width / self.amin[n]
        half = cutoff * self.radius[n] + _SLACK
        lo, hi = centre - half, centre + half
        box = points.box(lo, hi)
        if box is None:
            return None
        key, candidates, inside = box
        q = candidates[inside]
        others = np.array([j for j in range(len(self.sites.centres)) if j != row], dtype=np.int64)
        h = self._faces(q, row, self._kept(row, others, lo, hi, threshold))
        rho = _nucleus_rho(q, np.full(len(q), n - 1, dtype=np.int64), self.sites.shapes)
        argument = np.maximum((rho - 1) * self.rbar[n], h)
        edge = .5 * erfc(argument / (math.sqrt(2) * width))
        edge[(rho > cutoff) | (h > 4 * width)] = 0.0
        return key, inside, edge, q

    @staticmethod
    def _texture(regions, q):
        """T_n at q: Σ_b h_b exp(−½ ‖(σ − c_b) / s_b‖²), each term 0 outside its box of half-width 4 s_b."""
        total = np.zeros(len(q))
        for centre, size, height in regions:
            delta = (q - centre) / size
            near = np.all(np.abs(q - centre) <= 4 * size, axis=1)
            squared = delta[:, 0] * delta[:, 0] + delta[:, 1] * delta[:, 1] + delta[:, 2] * delta[:, 2]
            total += np.where(near, height * np.exp(-.5 * squared), 0.0)
        return total

    def _indicator(self, points, width):
        """M = min(1, Σ_n E_n) with edge ``width``, sites skipped only where no value changes."""
        total = points.zeros()
        for n in self.nucleus_row:
            result = self._nucleus(points, n, width, -_EXACT * width)
            if result is not None:
                _add(total, result[0], result[1], result[2])
        return np.minimum(total, 1.0)

    # ------------------------------------------------------------------ cells

    def _outer(self, q, j, squared):
        """o_j at q: the outer term of site j's territory test (T1, T2 or T3)."""
        sites = self.sites
        if sites.rule == "power":
            return np.full(len(q), -np.inf)
        if sites.rule == "reach":
            return np.sqrt(squared) - (sites.radii[j] + sites.reach[j])
        n = int(sites.nucleus[j])
        growth = float(sites.growth[j])
        if not n:
            return np.sqrt(squared) - (sites.empty_radius + growth)
        index = np.full(len(q), n - 1, dtype=np.int64)
        rho = _nucleus_rho(q, index, sites.shapes)
        grown = _grown_rho(q, index, sites.shapes, np.full(len(q), growth))
        return np.minimum((rho - 1) * self.rbar[n], (grown - 1) * self.rbar[n])

    def _reach(self, j, width):
        """Half-width (µm) of the ball about site j beyond which o_j > 4 · width."""
        sites = self.sites
        if sites.rule == "power":
            return np.inf
        if sites.rule == "reach":
            return sites.radii[j] + sites.reach[j] + 4 * width
        n = int(sites.nucleus[j])
        if not n:
            return sites.empty_radius + sites.growth[j] + 4 * width
        return (1 + 4 * width / self.rbar[n]) * (self.radius[n] + sites.growth[j])

    def _cell(self, points, k, width, threshold, boundary=False):
        """(key, inside, d_k, β_k) of cell k: d_k = max(h*_k, o_{j_k}) and, for the band, β_k = min(abs(d_k), g_k)."""
        sites = self.sites
        rows = self.cell_rows[k]
        reach = np.array([self._reach(j, width) for j in rows]) + _SLACK
        lo = np.min(sites.centres[rows] - reach[:, None], axis=0)
        hi = np.max(sites.centres[rows] + reach[:, None], axis=0)
        box = points.box(lo, hi)
        if box is None:
            return None
        key, candidates, inside = box
        q = candidates[inside]
        others = np.array([i for i in range(len(sites.centres)) if i not in rows], dtype=np.int64)
        kept = np.unique(np.concatenate([self._kept(j, others, lo, hi, threshold) for j in rows]))
        squared = _squared(q, sites.centres[rows])
        power = squared - sites.radii[rows] * sites.radii[rows]
        nearest = np.argmin(power, axis=1)
        pick = np.arange(len(q))
        faces = np.stack([self._faces(q, j, kept) for j in rows], axis=1)
        outer = np.stack([self._outer(q, j, squared[:, a]) for a, j in enumerate(rows)], axis=1)
        star = faces[pick, nearest]
        distance = np.maximum(star, outer[pick, nearest])
        if not boundary:
            return key, inside, distance, None
        beta = np.abs(distance)
        if len(rows) == 2:
            internal = (power[:, 0] - power[:, 1]) / (2 * self.gaps[rows[0], rows[1]])
            term = np.maximum(np.maximum(np.abs(internal), star),
                              np.maximum(np.minimum(outer[:, 0], outer[:, 1]), -np.maximum(outer[:, 0], outer[:, 1])))
            beta = np.minimum(beta, term)
        return key, inside, distance, beta

    # ------------------------------------------------------------------ the field and the sources

    @staticmethod
    def _field(field, points):
        """U at the positions (U1); 1 with no centre."""
        if field is None or not len(field.centres):
            return None
        flat = points.all()
        out = np.empty(len(flat))
        length2 = field.length * field.length
        for start in range(0, len(flat), _BLOCK >> 2):
            p = flat[start:start + (_BLOCK >> 2)]
            squared = [(p[:, 0] - u[0]) ** 2 + (p[:, 1] - u[1]) ** 2 + (p[:, 2] - u[2]) ** 2 for u in field.centres]
            low = squared[0].copy()
            for value in squared[1:]:
                np.minimum(low, value, out=low)
            numerator, denominator = np.zeros(len(p)), np.zeros(len(p))
            for z, value in zip(field.z, squared):
                weight = np.exp(-.5 * (value - low) / length2)
                numerator += z * weight
                denominator += weight * weight
            out[start:start + len(p)] = np.exp(field.amplitude * numerator / np.sqrt(denominator))
        return out.reshape(points.shape)

    def signal(self, name, points):
        """The pre-noise signal of stain ``name`` (without its background) at the positions (a ``_Points``)."""
        stain = self.stains[name]
        out = points.zeros()
        if self.sites is not None:
            if stain.source == "nucleus":
                w = stain.edge_width
                for n in self.nucleus_row:
                    result = self._nucleus(points, n, w, -4 * w)
                    if result is not None:
                        key, inside, edge, q = result
                        texture = self._texture(stain.regions.get(n, ()), q)
                        _add(out, key, inside, stain.plateaus[n] * edge * (1 + texture))
            elif stain.source == "territory":
                w = stain.edge_width
                indicator = self._indicator(points, w)
                for k in self.cell_rows:
                    result = self._cell(points, k, w, -_EXACT * w)
                    if result is not None:
                        key, inside, distance, _ = result
                        edge = .5 * erfc(distance / (math.sqrt(2) * w))
                        edge[distance > 4 * w] = 0.0
                        nuclear = 1 - (1 - stain.nuclear_ratio) * _gather(indicator, key, inside)
                        _add(out, key, inside, stain.plateaus[k] * edge * nuclear)
            else:
                b = stain.band_width
                for k in self.cell_rows:
                    result = self._cell(points, k, b, -4 * b, boundary=True)
                    if result is not None:
                        key, inside, _, beta = result
                        band = np.where(beta <= 4 * b, np.exp(-.5 * (beta / b) ** 2), 0.0)
                        _add(out, key, inside, stain.plateaus[k] * band)
        field = self._field(stain.field, points)
        if field is not None:
            out *= field
        return out

    def band(self, name, k, points):
        """(d_k, β_k) of the boundary stain ``name`` for cell k at flat positions; NaN beyond the cell's box."""
        stain = self.stains[name]
        d, beta = np.full(points.shape, np.nan), np.full(points.shape, np.nan)
        result = self._cell(points, k, stain.band_width, -4 * stain.band_width, boundary=True)
        if result is not None:
            key, inside, distance, value = result
            d[key], beta[key] = distance, value
        return d, beta

    def field(self, name, points):
        """U of stain ``name`` at the positions (ones without a field)."""
        value = self._field(self.stains[name].field, points)
        return np.ones(points.shape) if value is None else value


def _model(sites, stains, objects, regions, bumps):
    """The ``_Model`` of a scene from its stain configurations (records of ``StainChannelConfig``, one per name) and
    the tables ``stain_objects``, ``bright_regions`` and ``unevenness``."""
    specs = {}
    for record in stains:
        name = record["name"]
        if name in specs:
            continue
        own = objects[objects.stain == name]
        plateaus = dict(zip(own.object_id.astype(np.int64).tolist(), own.plateau.astype(np.float64).tolist()))
        texture = {}
        for row in regions[regions.stain == name].itertuples():
            texture.setdefault(int(row.nucleus_id), []).append(
                (np.array([row.centre_z, row.centre_y, row.centre_x]), float(row.size), float(row.height)))
        rows = bumps[bumps.stain == name]
        field_ = None
        if record.get("unevenness") is not None:
            field_ = SimpleNamespace(centres=rows[["centre_z", "centre_y", "centre_x"]].to_numpy(dtype=np.float64),
                                     z=rows.z.to_numpy(dtype=np.float64),
                                     amplitude=float(record["unevenness"]["amplitude"]),
                                     length=float(rows.length.iloc[0]) if len(rows) else math.nan)
        specs[name] = SimpleNamespace(source=record["source"], edge_width=float(record["edge_width"]),
                                      band_width=float(record["band_width"]),
                                      nuclear_ratio=float(record["nuclear_ratio"]), plateaus=plateaus,
                                      regions=texture, field=field_)
    return _Model(sites, specs)


# ---------------------------------------------------------------------------------------------- the draws


def _plateaus(stain, objects, stream):
    """Plateau per object (``nucleus-<n>`` or ``cell-<k>``) from ``context.stain.brightness``."""
    entities = tuple(objects)
    _checked_distribution(stain.brightness, "brightness", entities, f"stains[{stain.name!r}].brightness")
    values = [_draw(stain.brightness, "brightness", entity,
                    lambda _name, e: stream("context.stain.brightness", e, None, stain.name)) for entity in entities]
    if not np.isfinite(values).all() or (np.asarray(values) < 0).any():
        raise ValueError(f"stains[{stain.name!r}]: nonfinite or negative plateau")
    return values


def _bright_regions(stain, nuclei, sites, omega, flat, limit, stream):
    """Tx1 rows (region, nucleus_id, centre, size, height) of a nucleus stain: a Poisson count per nucleus, each
    centre the first uniform proposal in the nucleus's site box (within Ω) whose label is the nucleus."""
    texture, name = stain.texture, stain.name
    lo, hi = omega
    rows = []
    for n in nuclei:
        entity = f"nucleus-{n}"
        count = int(stream("context.texture.count", entity, None, name).poisson(texture.per_nucleus))
        row = int(np.flatnonzero(sites.nucleus == n)[0])
        centre, radius = sites.centres[row], sites.radii[row]
        box = (np.maximum(centre - radius, lo), np.minimum(centre + radius, hi))
        for b in range(1, count + 1):
            region = f"{entity}-region-{b}"
            rng = stream("context.texture.placement", region, None, name)
            point, drawn, size = None, 0, 4
            while point is None and drawn < limit:
                size = min(size, limit - drawn)
                proposals = rng.uniform(box[0], box[1], size=(size, 3))
                if flat:
                    proposals[:, 0] = 0.0
                hits = np.flatnonzero(_labels(proposals, sites, omega, _BLOCK)[1] == n)
                if len(hits):
                    point = proposals[hits[0]]
                drawn += size
                size = min(2 * size, 256)
            if point is None:
                raise ValueError(f"{region} of stain {name!r}: no proposal inside nucleus {n} after "
                                 f"max_proposals={limit}")
            values = [_draw(spec, label, region,
                            lambda _name, e, component=component: stream(component, e, None, name))
                      for spec, label, component in ((texture.size, "size", "context.texture.size"),
                                                     (texture.height, "height", "context.texture.height"))]
            if not (np.isfinite(values).all() and min(values) > 0):
                raise ValueError(f"{region} of stain {name!r}: nonfinite or nonpositive size or height")
            rows.append((b, n, *point.tolist(), *values))
    return rows


def _bumps(stain, omega, fov_width, flat, stream):
    """U1 rows (bump, centre, z_q) and the length of a stain's field: a Poisson count over the domain's Y-X area,
    centres uniform in its Y-X box at the z of its centre."""
    field_, name = stain.unevenness, stain.name
    lo, hi = omega
    area = float((hi[1] - lo[1]) * (hi[2] - lo[2]))
    count = int(stream("context.unevenness.count", "field", None, name).poisson(
        field_.centres_per_area * area / float(fov_width[0] * fov_width[1])))
    z = 0.0 if flat else float((lo[0] + hi[0]) / 2)
    rows = []
    for q in range(1, count + 1):
        entity = f"bump-{q}"
        centre = stream("context.unevenness.placement", entity, None, name).uniform(lo, hi)
        centre[0] = z
        rows.append((q, *centre.tolist(), float(stream("context.unevenness.amplitude", entity, None, name)
                                                .standard_normal())))
    length = float(field_.length) if field_.length is not None else float(max(fov_width))
    return rows, length
