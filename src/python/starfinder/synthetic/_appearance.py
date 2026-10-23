"""Composition, output, the morphology round and the truth of the real-appearance scene.

See docs/synthetic-real-appearance.md ("Composition", "Truth", "The morphology round", "Random streams").
``generate_appearance_scene`` places bank entries on the acquisition layout of an unchanged ``CellularSceneConfig``
(``_appearance_layout._place_appearance``), runs the sample scene's generation with the placed objects as its cells,
one ordinary cell with one nucleus each (the accepted cell types and molecules of ``_context._Context``, placed by
rejection with the labels of the placed masks), and composes every stain as a point function of the sample position:

    I(σ) = B(σ) + κ · H(σ) + S(σ)

``S`` is the owner's excess inside a placed mask (the first object holding ``σ``) and the sum of the objects' own
fall-offs elsewhere; ``H`` is the sum of each object's own excess blurred by the out-of-focus light (H2), and ``κ``
is computed once per scene on the sample grid; ``B`` is the bank's real background patches quilted with ramp weights on
a lattice fixed in the sample frame (BG1), or for a closed-form bank its preset's background (BG2). No observation
stage is added (OB1); an 8-bit stain is ``clip(rint(g · I), 0, 255)`` with ``g`` the median plateau of the bank's
source images (O8), and a stain without an output is ``I`` cast to the template dtype. The truth is the
``save_cellular_scene`` layout with the ellipsoid columns null, plus ``appearance.json``, ``placements.csv`` and
``background_patches.csv`` (AT2).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu

from starfinder._records import file_sha256
from starfinder.readout import Codebook

from ._appearance_bank import (BANK_DIRECTORY_VARIABLE, BankReference, _struct,
                               load_appearance_bank)
from ._appearance_layout import (_LAYOUT_COMPONENTS, _PAD, AugmentationConfig, LayoutConfig, _axes_window,
                                 _bank_kind, _Labels, _labels_from_table, _place_appearance, _prefilter_sigma,
                                 _prefiltered)
from ._common import _probabilities
from ._context import (_CONTEXT_COMPONENTS, CELLULAR_PRESET_VERSION, CELLULAR_TRUTH_VERSION, CellularScene,
                       CellularSceneConfig, _check_stain_names, _Context, _payload, _stain_table, _tables,
                       save_cellular_scene)
from ._formed import _STREAM_KEY, _STREAM_NAMESPACE, _plain
from ._observation import evaluate_background
from ._sample import _cast, _generate, _layout, _real, _streams

#: Version of the appearance truth files (appearance.json, placements.csv, background_patches.csv); any change to a
#: column, a key or a rule bumps it.
APPEARANCE_TRUTH_VERSION = "1"
# The version of the appearance presets (docs/synthetic-real-appearance.md, "Presets"), recorded in appearance.json.
_APPEARANCE_PRESET_VERSION = "appearance-presets-v1"
_LIGHT = ("objects", "none")
_BACKGROUNDS = ("patches", "closed-form")
# The stream component of the composition ("Random streams").
_COMPOSITION_COMPONENTS = {
    "context.appearance.background": ("patch-<z>-<y>-<x>", "one patch index per lattice cell, shared by the stains"),
}
# Above this many point-object pairs the point evaluator tests every object's box itself instead of culling first.
_CULL = 2_000_000
_PATCH_COLUMNS = ["stain", "lattice_z", "lattice_y", "lattice_x", "patch_id", "source_image"]


@dataclass(frozen=True)
class CompositionConfig:
    """How the stains of the appearance scene are composed (docs/synthetic-real-appearance.md, "Composition").

    ``light="objects"`` adds the out-of-focus light H2: each placed object's own excess
    blurred by a Gaussian of ``light_sigma_um`` (Z, Y, X) µm in the sample frame, scaled once
    per scene and stain by ``κ``, so that the background plus the light, averaged over the
    rims of the placed masks on the sample grid, equals the mean local source background of
    the placed objects; ``"none"`` sets ``κ = 0`` (H3). ``background`` is ``"patches"`` (BG1,
    the bank's real patches on a lattice fixed in the sample frame) or ``"closed-form"`` (BG2,
    the stain background of a closed-form bank's preset); None takes BG1 for a stored bank
    and BG2 for a closed-form bank. ``plateau`` is the grey level ``g`` that a stain with an
    8-bit output is scaled by; None takes the median plateau of the bank's source images
    (O8). No observation stage is added (OB1). Instances are not hashable.
    """

    __hash__ = None

    light: str = "objects"
    light_sigma_um: tuple[float, float, float] = (2.0, 1.0, 1.0)
    background: str | None = None
    plateau: float | None = None

    def __post_init__(self):
        if self.light not in _LIGHT:
            raise ValueError(f"light must be one of {list(_LIGHT)}; got {self.light!r}")
        if not isinstance(self.light_sigma_um, tuple) or len(self.light_sigma_um) != 3:
            raise ValueError("light_sigma_um must be a (z, y, x) tuple of µm")
        for value in self.light_sigma_um:
            _real(value, "light_sigma_um", strict=True)
        if self.background is not None and self.background not in _BACKGROUNDS:
            raise ValueError(f"background must be None or one of {list(_BACKGROUNDS)}; got {self.background!r}")
        if self.plateau is not None:
            _real(self.plateau, "plateau", strict=True)


@dataclass(frozen=True)
class AppearanceSceneConfig:
    """A cellular scene with real appearance (docs/synthetic-real-appearance.md, E1).

    ``context`` is an unchanged ``CellularSceneConfig``: its acquisition, seed, split and scene
    key, cell types, molecules, territory growth law and morphology round are used; its
    nuclei, packing and the stains' geometry, brightness, edge, texture, unevenness,
    background and noise are not. Each of its stains is the bank channel of its name, with
    its round, file pattern and output. ``bank`` names the bank (a stored bank by name and
    SHA-256, or a closed-form bank). ``augmentation`` (AU1), ``layout`` (PL2, DN1 or DN3, the
    stop) and ``composition`` (H2, BG1 or BG2, OB1, O8) are the new parts. The context's
    ``sample.cells`` must be given (the molecule draws read it) and ``sample.stain`` must be
    None. Instances are not hashable.
    """

    __hash__ = None

    context: CellularSceneConfig
    bank: BankReference
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    layout: LayoutConfig = field(default_factory=LayoutConfig)
    composition: CompositionConfig = field(default_factory=CompositionConfig)

    def __post_init__(self):
        for name, kind in (("context", CellularSceneConfig), ("bank", BankReference),
                           ("augmentation", AugmentationConfig), ("layout", LayoutConfig),
                           ("composition", CompositionConfig)):
            if not isinstance(getattr(self, name), kind):
                raise TypeError(f"{name} must be {kind.__name__}")
        if self.context.sample.cells is None:
            raise ValueError("context.sample.cells must be CellsConfig: the placed objects are its cells")
        if self.context.sample.stain is not None:
            raise ValueError("context.sample.stain must be None: the stains of an appearance scene are its stains, "
                             "each a channel of the bank")


@dataclass
class AppearanceScene:
    """A generated appearance scene (docs/synthetic-real-appearance.md, "Truth").

    ``context`` is a ``CellularScene`` whose tables and layout are those of the placed objects
    (one ordinary cell with one nucleus each, the shape columns null, the closed-form stain
    tables empty) and whose stains are the composed ones; ``save_cellular_dataset(scene.context,
    ...)`` writes its dataset. ``placements`` has one row per placed object (the columns of
    ``placements.csv``), ``background_patches`` one row per stain and lattice cell (BG1; empty
    with BG2). ``record`` holds the versions, the requested and effective configuration, the
    bank's name, SHA-256 and digest, the reached density, the placed objects and proposals,
    ``κ`` and ``g`` per stain and the streams.
    """

    context: CellularScene
    placements: pd.DataFrame
    background_patches: pd.DataFrame
    record: dict


# ------------------------------------------------------------------------------------------------ interpolation


def _index_terms(obj, t, a):
    """The entry index of axis ``a`` from the offsets ``t`` (z, y, x) from the object's centre: the point evaluator's
    arithmetic (``_Object.index``)."""
    B = obj.Ainv
    return (obj.xbar[a] + t[0] * B[a, 0] + t[1] * B[a, 1] + t[2] * B[a, 2]) / obj.s_e[a]


def _on_grid(obj, coords, array, pad):
    """``array`` (on the entry box grown by ``pad`` voxels per axis) interpolated linearly at ``T⁻¹σ`` on the outer
    grid of ``coords``; 0 outside the grown box. An axis of one voxel is a column (index 0 everywhere)."""
    shape = tuple(len(v) for v in coords)
    t = [coords[0][:, None, None] - obj.c[0], coords[1][None, :, None] - obj.c[1],
         coords[2][None, None, :] - obj.c[2]]
    q, inside = [], np.ones(shape, dtype=bool)
    for a in range(3):
        if obj.n[a] == 1:
            q.append(np.zeros(shape))
            continue
        qa = np.broadcast_to(_index_terms(obj, t, a), shape)
        if pad[a]:
            qa = qa + pad[a]
        inside &= (qa >= 0) & (qa <= array.shape[a] - 1)
        q.append(qa)
    out = np.zeros(shape)
    if inside.any():
        out[inside] = ndi.map_coordinates(array, [qa[inside] for qa in q], order=1, mode="nearest", prefilter=False)
    return out


def _at_points(obj, points, array, pad):
    """``_on_grid`` at (P, 3) sample positions."""
    d = points - obj.c
    t = [d[:, 0], d[:, 1], d[:, 2]]
    q, inside = [], np.ones(len(points), dtype=bool)
    for a in range(3):
        if obj.n[a] == 1:
            q.append(np.zeros(len(points)))
            continue
        qa = _index_terms(obj, t, a)
        if pad[a]:
            qa = qa + pad[a]
        inside &= (qa >= 0) & (qa <= array.shape[a] - 1)
        q.append(qa)
    out = np.zeros(len(points))
    if inside.any():
        out[inside] = ndi.map_coordinates(array, [qa[inside] for qa in q], order=1, mode="nearest", prefilter=False)
    return out


def _grid_axes(points):
    """The coordinate vectors (z, y, x) when (Z, Y, X, 3) ``points`` are the outer grid of three increasing vectors,
    else None."""
    if points.ndim != 4 or points.shape[-1] != 3:
        return None
    axes = [points[:, 0, 0, 0], points[0, :, 0, 1], points[0, 0, :, 2]]
    for a, v in enumerate(axes):
        if len(v) > 1 and not np.all(np.diff(v) > 0):
            return None
        index = [None, None, None]
        index[a] = slice(None)
        if not np.array_equal(points[..., a], np.broadcast_to(v[tuple(index)], points.shape[:3])):
            return None
    return [np.array(v) for v in axes]


# ------------------------------------------------------------------------------------------------ the composition


class _Composer:
    """The placed objects' arrays of the composition, made once per object and channel: the excess after IN2's
    prefilter for the target spacing, and its copy blurred by H2's Gaussian on the entry box grown by three
    deviations (zero beyond the entry box)."""

    def __init__(self, bank, spacing, light_sigma_um, light):
        self.bank = bank
        self.spacing = np.asarray(spacing, dtype=np.float64)
        self.light_sigma = np.asarray(light_sigma_um, dtype=np.float64)
        self.light = light == "objects"
        self.objects = []
        self._raw, self._excess, self._haze, self._bounds = {}, {}, {}, {}

    def raw(self, obj, c):
        key = (obj.row, c)
        if key not in self._raw:
            self._raw[key] = np.asarray(self.bank.excess(obj.row)[c], dtype=np.float64)
        return self._raw[key]

    def excess(self, i, c):
        """ε of object ``i`` and channel ``c`` after IN2's prefilter of ``max(0, t_a / (m_a s_e,a) − 1) / 2`` source
        voxels per axis."""
        key = (i, c)
        if key not in self._excess:
            obj = self.objects[i]
            sigma = _prefilter_sigma(self.spacing, (obj.scale_z, obj.scale_yx, obj.scale_yx), obj.s_e, obj.n)
            self._excess[key] = _prefiltered(self.raw(obj, c), sigma)
        return self._excess[key]

    def haze_sigma(self, obj):
        """H2's deviations in entry voxels: ``light_sigma_um`` over the object's scale per axis, over the source
        spacing; none along an axis of one voxel (a column)."""
        scale = (obj.scale_z, obj.scale_yx, obj.scale_yx)
        return tuple(0.0 if n == 1 else float(h / m / s) for h, m, s, n in zip(self.light_sigma, scale, obj.s_e, obj.n))

    def haze(self, i, c):
        """(the blurred copy ``G_k * ε`` on the entry box grown by ``pad`` voxels, ``pad``) of object ``i``."""
        key = (i, c)
        if key not in self._haze:
            obj = self.objects[i]
            sigma = self.haze_sigma(obj)
            pad = tuple(int(math.ceil(3 * v)) for v in sigma)
            padded = np.pad(self.raw(obj, c), [(p, p) for p in pad])
            self._haze[key] = (ndi.gaussian_filter(padded, sigma, mode="constant", cval=0.0), pad)
        return self._haze[key]

    def haze_bounds(self, i):
        """The bounding box (lo, hi) in the sample frame of object ``i``'s grown box (a column in Z for a 2D entry)."""
        if i not in self._bounds:
            obj = self.objects[i]
            pad = np.array([int(math.ceil(3 * v)) for v in self.haze_sigma(obj)], dtype=np.float64)
            low, high = obj._terms(-pad, obj.n - 1.0 + pad)
            lo, hi = obj.c + low - _PAD, obj.c + high + _PAD
            if obj.n[0] == 1:
                lo[0], hi[0] = -np.inf, np.inf
            self._bounds[i] = (lo, hi)
        return self._bounds[i]


class _Canvas:
    """The sums of the composition on the outer grid of increasing coordinates (z, y, x), the objects added in
    placement order: the union of the placed masks, the owner's excess, the sum of every object's excess (its own
    fall-off outside its mask) and the sum of the blurred copies, per channel."""

    def __init__(self, composer, coords, channels):
        self.composer = composer
        self.coords = [np.asarray(v, dtype=np.float64) for v in coords]
        self.shape = tuple(len(v) for v in self.coords)
        self.channels = list(channels)
        self.held = np.zeros(self.shape, dtype=bool)
        self.owner = {c: np.zeros(self.shape) for c in self.channels}
        self.total = {c: np.zeros(self.shape) for c in self.channels}
        self.haze = {c: np.zeros(self.shape) for c in self.channels}
        self.added = 0

    def add_until(self, count):
        while self.added < count:
            self._add(self.added)
            self.added += 1
        return self

    def _add(self, i):
        composer = self.composer
        obj = composer.objects[i]
        window = _axes_window(obj.lo, obj.hi, self.coords)
        if window is not None:
            where = tuple(slice(start, stop) for start, stop in window)
            sub = [v[start:stop] for v, (start, stop) in zip(self.coords, window)]
            mask = obj.grid_values(sub, obj.D) <= 0
            held = self.held[where]
            new = mask & ~held
            for c in self.channels:
                e = obj.gain * _on_grid(obj, sub, composer.excess(i, c), (0, 0, 0))
                self.owner[c][where][new] = e[new]
                self.total[c][where] += e
            held |= mask
        if composer.light:
            lo, hi = composer.haze_bounds(i)
            window = _axes_window(lo, hi, self.coords)
            if window is not None:
                where = tuple(slice(start, stop) for start, stop in window)
                sub = [v[start:stop] for v, (start, stop) in zip(self.coords, window)]
                for c in self.channels:
                    array, pad = composer.haze(i, c)
                    self.haze[c][where] += obj.gain * _on_grid(obj, sub, array, pad)

    def signal(self, c):
        """S: the owner's excess inside the placed masks, the sum of the fall-offs elsewhere."""
        return np.where(self.held, self.owner[c], self.total[c])

    def image(self, c, kappa, background):
        return background + kappa * self.haze[c] + self.signal(c)


def _compose_points(composer, c, kappa, background, points):
    """I at (..., 3) sample positions by the same terms as ``_Canvas`` (the point evaluator); ``background`` is B at
    those positions."""
    points = np.asarray(points, dtype=np.float64)
    flat = points.reshape(-1, 3)
    held = np.zeros(len(flat), dtype=bool)
    owner, total, haze = np.zeros(len(flat)), np.zeros(len(flat)), np.zeros(len(flat))
    for i, obj in enumerate(composer.objects):
        rows = obj.select(flat)
        if len(rows):
            sub = flat[rows]
            mask = obj.values(sub, obj.D) <= 0
            e = obj.gain * _at_points(obj, sub, composer.excess(i, c), (0, 0, 0))
            new = mask & ~held[rows]
            owner[rows[new]] = e[new]
            held[rows[mask]] = True
            total[rows] += e
        if composer.light:
            lo, hi = composer.haze_bounds(i)
            rows = np.flatnonzero(np.all((flat >= lo) & (flat <= hi), axis=1))
            if len(rows):
                array, pad = composer.haze(i, c)
                haze[rows] += obj.gain * _at_points(obj, flat[rows], array, pad)
    signal = np.where(held, owner, total)
    return (np.asarray(background).reshape(-1) + kappa * haze + signal).reshape(points.shape[:-1])


def _kappa(canvas, c, background, local, column):
    """κ of H2 on the sample grid (``proto.py::place_all``, ``images``): ``max(b̄ − mean_rim(B), 0) / mean_rim(H)``
    over the rims of the placed masks (their union less its face-connected erosion, in the plane for 2D entries), with
    ``b̄`` the mean local source background of the placed objects; 0 without a rim or without light there."""
    held = canvas.held
    terms = dict(mean_local_background=float(np.mean(local)) if len(local) else None, rim_voxels=0,
                 rim_background=None, rim_light=None)
    if not len(local) or not held.any():
        return 0.0, terms
    structure = _struct((1,) if column else held.shape)
    rim = held & ~ndi.binary_erosion(held, structure=structure)
    if not rim.any():
        return 0.0, terms
    rim_b = float(background[rim].mean())
    rim_h = float(canvas.haze[c][rim].mean())
    terms.update(rim_voxels=int(rim.sum()), rim_background=rim_b, rim_light=rim_h)
    if not rim_h > 0:
        return 0.0, terms
    return max(terms["mean_local_background"] - rim_b, 0.0) / rim_h, terms


def _visible_share(image, threshold, form):
    """The share of ``image`` (or of its Z maximum) whose Gaussian of one voxel lies above ``threshold``."""
    plane = image.max(axis=0) if form == "zmax" else image
    smooth = ndi.gaussian_filter(plane, [1.0 if n > 1 else 0.0 for n in plane.shape])
    return float(np.mean(smooth > threshold))


def _density_form(kind, bank):
    """DN1's stain and form: the culture cells' stain (the bank's last channel; Flamingo) as a Z maximum, a 2D
    entry's stain as a Z maximum (every plane under TZ1), and a dense 3D bank's nuclear stain on the volume."""
    if kind == "cell":
        return len(bank.channels) - 1, "zmax"
    if kind == "column":
        return 0, "zmax"
    return 0, "volume"


def _density_target(bank, extent_um, channel, form):
    """DN1's target and threshold: the medians over the bank's source images of the visible-stain share and of the
    Otsu threshold over the image's plateau, on the middle planes matching the scene's Z extent ``extent_um``. A
    closed-form bank's source images are rendered again in memory; a stored bank holds none."""
    preset = bank.record.get("preset")
    if preset is None:
        raise ValueError(f"bank {bank.name}: density='visible' (DN1) needs layout.target and layout.threshold, "
                         "because a stored bank holds no source image to measure them on")
    from ._appearance_bank import _closed_form_source
    plateaus = {row["image"]: float(row["plateau"]) for row in bank.record["images"]}
    sources = (_closed_form_source(preset, int(seed)) for seed in bank.record["seeds"][:-1])
    rows = _density_rows(sources, plateaus, extent_um, channel, form)
    target = float(np.median([r["visible_share"] for r in rows]))
    threshold = float(np.median([r["threshold_over_plateau"] for r in rows]))
    return target, threshold, rows


def _density_rows(sources, plateaus, extent_um, channel, form):
    """Per source image (a loaded source of the build step): DN1's visible-stain share and Otsu threshold over the
    image's plateau ``plateaus[image]`` on its middle planes matching the Z extent ``extent_um``."""
    rows = []
    for source in sources:
        stain = source["stains"][channel]
        depth = stain.shape[0]
        planes = min(max(int(round(extent_um / float(source["spacing"][0]))), 1), depth)
        start = (depth - planes) // 2
        region = stain[start:start + planes]
        plane = region.max(axis=0) if form == "zmax" else region
        smooth = ndi.gaussian_filter(plane, [1.0 if n > 1 else 0.0 for n in plane.shape])
        threshold = float(threshold_otsu(smooth))
        plateau = plateaus[source["image"]]
        rows.append(dict(image=source["image"], planes=[start, start + planes], visible_share=float(np.mean(
            smooth > threshold)), threshold=threshold, plateau=plateau, threshold_over_plateau=threshold / plateau))
    return rows


# ------------------------------------------------------------------------------------------------ the background


class _PatchBackground:
    """BG1: one bank patch per cell of a lattice fixed in the sample frame, from Ω's lower corner with the period of
    the patch size less its overlap (µm), quilted as ``Σ_p w_p(σ) P_p(σ − o_p) / Σ_p w_p(σ)``. Each patch is
    interpolated linearly on its grid (edge values beyond its voxels); its weight is per axis the prototype's ramp
    (``proto.py::quilt``), ``min(1, (k + 1) / (v + 1), (n − k) / (v + 1))`` at voxel ``k`` of ``n`` overlapping by
    ``v``, interpolated linearly and 0 one voxel beyond each end, so that ``B`` is continuous. An axis of one voxel is
    a column. The patch of a lattice cell is one index drawn from ``context.appearance.background``, shared by the
    stains."""

    def __init__(self, bank, omega, stream, draws=None):
        patches = bank.arrays["patch"]
        if not len(patches):
            raise ValueError(f"bank {bank.name}: it holds no background patch; background='patches' (BG1) needs them")
        configuration = bank.record["configuration"]
        self.bank = bank
        self.spacing = np.asarray(configuration["spacing_zyx"], dtype=np.float64)
        self.n = np.array(patches.shape[2:], dtype=np.int64)
        overlap = np.asarray(configuration["patches"]["overlap_zyx"], dtype=np.int64)
        self.overlap = np.where(self.n > 1, np.clip(overlap, 0, self.n - 1), 0)
        self.period = (self.n - self.overlap) * self.spacing
        self.origin = np.asarray(omega[0], dtype=np.float64).copy()
        extent = np.asarray(omega[1], dtype=np.float64) - self.origin
        self.counts = [1 if self.n[a] == 1 else int(math.ceil(extent[a] / self.period[a])) + 1 for a in range(3)]
        # One patch index per lattice cell; ``draws`` rebuilds the lattice from a scene's background_patches.csv.
        self.draws = draws if draws is not None else {
            cell: int(stream("context.appearance.background", "patch-{}-{}-{}".format(*cell)).integers(len(patches)))
            for cell in np.ndindex(*self.counts)}
        self._values, self._ramps = {}, {}

    def record(self):
        return dict(mode="patches", rule="BG1", lattice_origin_zyx=self.origin.tolist(),
                    period_zyx=self.period.tolist(), patch_shape_zyx=self.n.tolist(),
                    overlap_zyx=self.overlap.tolist(), cells_zyx=list(self.counts),
                    patch_spacing_zyx=self.spacing.tolist(),
                    weights="per axis the ramp min(1, (k + 1) / (v + 1), (n − k) / (v + 1)) at voxel k, interpolated "
                            "linearly, 0 one voxel beyond each end")

    def rows(self, stains):
        images = self.bank.patches.source_image.tolist()
        return [(name, *cell, index, images[index]) for name in stains for cell, index in self.draws.items()]

    def _patch(self, index, c):
        key = (index, c)
        if key not in self._values:
            self._values[key] = np.asarray(self.bank.patch(index)[c], dtype=np.float64)
        return self._values[key]

    def _corner(self, cell):
        return self.origin + np.asarray(cell, dtype=np.float64) * self.period

    def _weight(self, a, u):
        """The ramp of axis ``a`` at indices ``u``: its voxel values interpolated linearly, 0 one voxel beyond each
        end."""
        if a not in self._ramps:
            n, v = int(self.n[a]), int(self.overlap[a])
            k = np.arange(n, dtype=np.float64)
            values = np.minimum(np.minimum((k + 1) / (v + 1), (n - k) / (v + 1)), 1.0)
            self._ramps[a] = (np.concatenate([[-1.0], k, [float(n)]]), np.concatenate([[0.0], values, [0.0]]))
        return np.interp(u, *self._ramps[a], left=0.0, right=0.0)

    def grid(self, coords, c):
        """B of channel ``c`` on the outer grid of increasing ``coords``."""
        coords = [np.asarray(v, dtype=np.float64) for v in coords]
        shape = tuple(len(v) for v in coords)
        num, den = np.zeros(shape), np.zeros(shape)
        for cell, index in self.draws.items():
            o = self._corner(cell)
            window, u, w = [], [], []
            for a in range(3):
                if self.n[a] == 1:
                    window.append(slice(0, shape[a]))
                    u.append(np.zeros(shape[a]))
                    w.append(np.ones(shape[a]))
                    continue
                start = int(np.searchsorted(coords[a], o[a] - self.spacing[a], side="right"))
                stop = int(np.searchsorted(coords[a], o[a] + self.n[a] * self.spacing[a], side="left"))
                if stop <= start:
                    break
                window.append(slice(start, stop))
                ua = (coords[a][start:stop] - o[a]) / self.spacing[a]
                u.append(ua)
                w.append(self._weight(a, ua))
            else:
                where = tuple(window)
                weight = w[0][:, None, None] * w[1][None, :, None] * w[2][None, None, :]
                q = np.meshgrid(*u, indexing="ij")
                values = ndi.map_coordinates(self._patch(index, c), q, order=1, mode="nearest", prefilter=False)
                num[where] += weight * values
                den[where] += weight
        if not (den > 0).all():
            raise RuntimeError("the background lattice does not cover the grid")
        return num / den

    def points(self, points, c):
        """B of channel ``c`` at (..., 3) sample positions."""
        points = np.asarray(points, dtype=np.float64)
        flat = points.reshape(-1, 3)
        num, den = np.zeros(len(flat)), np.zeros(len(flat))
        for cell, index in self.draws.items():
            o = self._corner(cell)
            u = (flat - o) / self.spacing
            keep = np.ones(len(flat), dtype=bool)
            for a in range(3):
                if self.n[a] == 1:
                    u[:, a] = 0.0
                else:
                    keep &= (u[:, a] > -1) & (u[:, a] < self.n[a])
            rows = np.flatnonzero(keep)
            if not len(rows):
                continue
            weight = np.ones(len(rows))
            for a in range(3):
                if self.n[a] > 1:
                    weight = weight * self._weight(a, u[rows, a])
            values = ndi.map_coordinates(self._patch(index, c), u[rows].T, order=1, mode="nearest", prefilter=False)
            num[rows] += weight * values
            den[rows] += weight
        if not (den > 0).all():
            raise RuntimeError("the background lattice does not cover the points")
        return (num / den).reshape(points.shape[:-1])


class _ClosedFormBackground:
    """BG2: the stain background of a closed-form bank's preset (its ``BackgroundConfig``: baseline and regions on the
    preset's sample grid, multiplied by the preset's output scale, as the bank's source images hold it), a point
    function of σ through the index of the preset's sample grid, over the scene's plateau ``g``."""

    def __init__(self, bank, g):
        preset = bank.record.get("preset")
        if preset is None:
            raise ValueError(f"background='closed-form' (BG2) needs a closed-form bank; bank {bank.name} is a stored "
                             "bank, whose background is its patches (BG1)")
        from ._appearance_bank import _closed_form_scene
        scene = _closed_form_scene(preset, int(bank.record["seeds"][0]))
        grid = scene.sample.record["sample_grid"]
        self.preset = preset
        self.origin = np.asarray(grid["origin_zyx"], dtype=np.float64)
        self.spacing = np.asarray(grid["spacing_zyx"], dtype=np.float64)
        self.g = float(g)
        self.terms = []
        for name in bank.channels:
            effective = next(s for s in scene.record["effective_config"]["stains"] if s["name"] == name)
            background = effective["background"]
            self.terms.append(dict(stain=name, scale=float(effective["scale"]),
                                   baseline=float(np.asarray(background["baseline"])[0, 0]),
                                   weight=float(np.asarray(background["tissue_weights"])[0, 0]),
                                   components=background["components"]))

    def record(self):
        return dict(mode="closed-form", rule="BG2", preset=self.preset, grid_origin_zyx=self.origin.tolist(),
                    grid_spacing_zyx=self.spacing.tolist(), plateau=self.g, stains=_plain(self.terms))

    def rows(self, stains):
        return []

    def points(self, points, c):
        points = np.asarray(points, dtype=np.float64)
        term = self.terms[c]
        index = (points - self.origin) / self.spacing
        tissue = evaluate_background(term["components"], index) if term["components"] else 0.0
        return term["scale"] * (term["baseline"] + term["weight"] * tissue) / self.g + np.zeros(points.shape[:-1])

    def grid(self, coords, c):
        return self.points(np.stack(np.meshgrid(*coords, indexing="ij"), axis=-1), c)


# ------------------------------------------------------------------------------------------------ the truth


class _SceneSites:
    """The placed objects as the sample scene's label sites: one site per cell (its centre and a radius holding its
    territory, the points where its ``δ_k`` is at most its growth under CT1 or its own mask under CU1; for a 2D entry
    its column through Ω), each cell with its one nucleus of the same ID. ``appearance`` is the label function
    ``_sample._label_functions`` dispatches to: the placed objects in placement order."""

    def __init__(self, labels, objects, omega):
        self.labels = labels
        self.objects = list(objects)
        self.centres = np.array([o.c for o in objects], dtype=np.float64).reshape(-1, 3)
        levels = (np.zeros(len(objects)) if labels.growth is None
                  else np.asarray(labels.growth, dtype=np.float64) / np.array([o.m for o in objects]))
        self.radii = np.array([_box_radius(o, omega, level) for o, level in zip(objects, levels)], dtype=np.float64)
        self.cell = np.array([o.k for o in objects], dtype=np.uint32)
        self.nucleus = self.cell.copy()
        self._lo = np.array([o.lo for o in objects], dtype=np.float64).reshape(-1, 3)
        self._hi = np.array([o.hi for o in objects], dtype=np.float64).reshape(-1, 3)

    def appearance(self, points):
        """(C, N) at (..., 3) sample positions."""
        points = np.asarray(points, dtype=np.float64)
        axes = _grid_axes(points)
        if axes is not None:
            return self.labels.axes(axes)
        flat = points.reshape(-1, 3)
        count = len(self.objects)
        if count and len(flat) * count <= _CULL:
            near = np.all((flat[:, None, :] >= self._lo[None]) & (flat[:, None, :] <= self._hi[None]), axis=2)
            chosen = np.flatnonzero(near.any(axis=0))
            if len(chosen) < count:
                growth = None if self.labels.growth is None else self.labels.growth[chosen]
                subset = _Labels(self.labels.kind, [self.objects[i] for i in chosen], growth)
                return subset.labels(points)
        return self.labels.labels(points)


def _box_radius(obj, omega, level):
    """The radius of a sphere about the object's centre that holds every point where its interpolated ``D_e`` is at
    most ``level`` (µm): linear interpolation reaches ``level`` only in a cell with a corner at or below it, so the
    entry voxels at or below ``level`` grown by one voxel, transformed, hold them. For a 2D entry (a column) it also
    reaches both Z ends of Ω."""
    where = np.nonzero(obj.D <= level)
    if not len(where[0]):
        return 0.0
    first = np.array([max(int(w.min()) - 1, 0) for w in where], dtype=np.float64)
    last = np.array([min(int(w.max()) + 1, int(n) - 1) for w, n in zip(where, obj.n)], dtype=np.float64)
    low, high = obj._terms(first, last)
    if obj.n[0] == 1:
        lateral = float(np.hypot(max(abs(low[1]), abs(high[1])), max(abs(low[2]), abs(high[2]))))
        return max(lateral, abs(obj.c[0] - omega[0][0]), abs(omega[1][0] - obj.c[0])) + _PAD
    return float(np.linalg.norm(np.maximum(np.abs(low), np.abs(high)))) + _PAD


def _site_radius(obj, coords, nucleus):
    """The largest distance from the object's centre ``c_k`` to a sample-grid voxel its mask holds (its nucleus mask
    for a culture nucleus row); NaN without such a voxel."""
    window = _axes_window(obj.lo, obj.hi, coords)
    if window is None:
        return math.nan
    sub = [v[start:stop] for v, (start, stop) in zip(coords, window)]
    held = obj.grid_values(sub, obj.D) <= 0
    if nucleus and obj.ND is not None:
        held &= obj.grid_values(sub, obj.ND) <= 0
    where = np.nonzero(held)
    if not len(where[0]):
        return math.nan
    squared = sum((sub[a][where[a]] - obj.c[a]) ** 2 for a in range(3))
    return float(np.sqrt(squared.max()))


class _AppearanceContext(_Context):
    """The accepted cell types and molecules (``_context._Context``) on the placed objects: its placement hook returns
    the placed objects as one ordinary cell with one nucleus each; molecules are placed by rejection in each cell's
    site sphere, which holds its entry box, with the labels of the placed masks."""

    def __init__(self, config, codebook, stream, built):
        super().__init__(config, codebook, stream)
        self.built = built
        # T1's form with a reach of 0: a molecule's proposal box is the cell's site sphere.
        self.rule = "reach"

    def _place(self, cells_config, layout, stream):
        expected = self.built.layout
        if not (np.array_equal(layout.origin, expected.origin) and np.array_equal(layout.shape, expected.shape)):
            raise RuntimeError("the sample scene's layout differs from the placement's")
        return self.built.nuclei, self.built.cells, self.built.sites


def _truth(placed, coords, omega):
    """The nucleus and cell records of the placed objects for the sample scene's tables: centre and anchor from the
    placement (at ``Z=1`` the cell's anchor in the plane, z = 0, so that sample membership tests its Y and X), the
    shape columns null, ``site_radius`` of the placed mask, one ordinary cell with one nucleus, CT1's growth."""
    kind, flat = placed.kind, bool(placed.layout.flat)
    anchors = placed.placements[["anchor_z", "anchor_y", "anchor_x"]].to_numpy(dtype=np.float64)
    growth = placed.cells.growth.to_numpy(dtype=np.float64)
    sites = _SceneSites(placed.labels, placed.objects, omega)
    nuclei, cells = [], []
    for i, obj in enumerate(placed.objects):
        anchor = anchors[i].copy()
        if flat:
            anchor[0] = 0.0
        nuclei.append(dict(nucleus_id=obj.k, cell_id=obj.k, centre=obj.c.copy(), axial=math.nan, lateral=math.nan,
                           elongation=math.nan, angle=math.nan, site_radius=_site_radius(obj, coords, kind == "cell"),
                           lobes=None, coefficients=None))
        cells.append(dict(cell_id=obj.k, kind="ordinary", anchor=anchor, n_nuclei=1, reach=0.0,
                          growth=None if kind == "cell" else float(growth[i]),
                          sites=[(obj.c.copy(), float(sites.radii[i]))]))
    return nuclei, cells, sites


def _translations(context, fov_ids, spacing, stream):
    """The morphology round's translation per FOV in voxels and µm, drawn as ``_context._stain_part`` draws it."""
    morphology = context.morphology
    out = {}
    if morphology is None:
        return out
    bound = np.asarray(morphology.translation_max_zyx, dtype=np.float64)
    for fov in fov_ids:
        if morphology.translation == "uniform":
            vox = bound * (2 * stream("context.morphology.translation", fov).random(3) - 1) + 0.0
        else:
            vox = np.asarray(morphology.translation_zyx[fov], dtype=np.float64) + 0.0
        out[fov] = (vox, vox * spacing)
    return out


def _channels(context, bank):
    """The bank channel of each stain, by name; ValueError naming a stain the bank does not hold."""
    out = {}
    for stain in context.stains:
        if stain.name not in bank.channels:
            raise ValueError(f"stains[{stain.name!r}]: the bank {bank.name} holds the channels {list(bank.channels)}; "
                             "each stain of an appearance scene is a bank channel of its name")
        out[stain.name] = bank.channels.index(stain.name)
    return out


def _output(J, stain, g, dtype):
    """The cast stain: ``clip(rint(g · J), 0, max)`` with an output (O8), else ``J`` cast to the template dtype; with
    the dtype and the factor."""
    if not np.isfinite(J).all():
        raise ValueError(f"nonfinite composed stain {stain.name!r}")
    if stain.output is not None:
        return _cast(g * J, stain.output.dtype), stain.output.dtype, g
    return _cast(J.copy(), dtype), dtype, 1.0


def _config_payload(config):
    def part(value):
        return dict(_plain(asdict(value)), type=type(value).__name__)
    return dict(type="AppearanceSceneConfig", context=_payload(config.context), bank=part(config.bank),
                augmentation=part(config.augmentation), layout=part(config.layout),
                composition=part(config.composition))


def _record_path(reference):
    if reference.preset is not None:
        return None
    directory = reference.directory if reference.directory is not None else os.environ.get(BANK_DIRECTORY_VARIABLE)
    return None if directory is None else str(Path(directory) / f"{reference.name}.record.json")


# ------------------------------------------------------------------------------------------------ the entry


def generate_appearance_scene(codebook: Codebook, *, config: AppearanceSceneConfig,
                              on_fov: Callable | None = None) -> AppearanceScene:
    """Generate one cellular scene with real appearance and acquire it as a grid of FOVs.

    The bank ``config.bank`` is loaded (``load_appearance_bank``) and its entries placed on the
    acquisition layout of ``config.context`` by PL2 (AU1's augmentation, BS2's draws, DN1's
    visible-stain target or DN3's count, the stop of "When placement stops"). The placed
    objects are the cells of the sample scene's generation (``generate_sample_scene``'s
    code, ``on_fov`` passed on as there): one ordinary cell with one nucleus each, labelled
    by one placement order (CT1 in the dense contexts, the cells' own masks in the culture),
    with the accepted cell types and molecules. Each stain is the composition ``B + κH + S``
    of the bank channel of its name, evaluated at the FOV's voxel centres (the morphology
    round's at the centres moved by its translation; MO1); no noise is added, and a stain
    with an 8-bit output is scaled by ``g``. DN1 measures the visible-stain share of the
    composed stain on the sample grid after every ``check_every`` placements, against the
    medians over a closed-form bank's source images or ``layout.target`` and
    ``layout.threshold``. Invalid input, a bank refused by its SHA-256, a stain the bank
    does not hold and a stop below ``floor`` raise ValueError; nothing is written.
    """
    if not isinstance(config, AppearanceSceneConfig):
        raise TypeError("config must be AppearanceSceneConfig")
    if not isinstance(codebook, Codebook):
        raise TypeError("codebook must be Codebook")
    if on_fov is not None and not callable(on_fov):
        raise TypeError("on_fov must be callable or None")
    context = config.context
    _check_stain_names(context, codebook)
    bank = load_appearance_bank(config.bank)
    channel_of = _channels(context, bank)
    composition, layout_config = config.composition, config.layout
    mode = composition.background or ("closed-form" if bank.record.get("preset") is not None else "patches")
    g = float(composition.plateau) if composition.plateau is not None else float(bank.record["plateau_median"])
    kind = _bank_kind(bank)
    column = kind == "column"
    template, acquisition = context.sample.fov, context.sample.acquisition
    grow = None if context.morphology is None else np.asarray(context.morphology.translation_max_zyx, dtype=np.float64)
    grid = _layout(acquisition, grow)
    sample_coords = [grid.origin[a] + np.arange(grid.shape[a]) * grid.s[a] for a in range(3)]
    log = {}
    stream = _streams(template.split, template.seed, template.scene_key, log)
    background = (_PatchBackground(bank, grid.omega, stream) if mode == "patches"
                  else _ClosedFormBackground(bank, g))
    composer = _Composer(bank, grid.s, composition.light_sigma_um, composition.light)
    local = {c: bank.entries[f"local_background_{name}"].to_numpy(dtype=np.float64)
             for c, name in enumerate(bank.channels)}

    density = dict(stain=None, form=None, sources=None)
    visible = target = threshold = None
    if layout_config.density == "visible":
        channel, form = _density_form(kind, bank)
        target, threshold = layout_config.target, layout_config.threshold
        if target is None or threshold is None:
            measured_target, measured_threshold, rows = _density_target(bank, float(grid.n[0] * grid.s[0]), channel,
                                                                        form)
            target = measured_target if target is None else target
            threshold = measured_threshold if threshold is None else threshold
            density["sources"] = rows
        density.update(stain=bank.channels[channel], form=form)
        dn1_background = background.grid(sample_coords, channel)
        dn1_canvas = _Canvas(composer, sample_coords, [channel])

        def visible(objects):
            composer.objects = objects
            dn1_canvas.add_until(len(objects))
            kappa, _ = _kappa(dn1_canvas, channel, dn1_background, local[channel][[o.row for o in objects]], column)
            return _visible_share(dn1_canvas.image(channel, kappa, dn1_background), threshold, form)

    placed = _place_appearance(context, bank, config.augmentation, layout_config, visible=visible, target=target)
    if not (np.array_equal(placed.layout.origin, grid.origin) and np.array_equal(placed.layout.shape, grid.shape)):
        raise RuntimeError("the placement's layout differs from the scene's")
    objects = placed.objects
    composer.objects = objects
    rows = [o.row for o in objects]

    # κ per bank channel, once per scene on the sample grid.
    used = sorted(set(channel_of.values()))
    sample_canvas = _Canvas(composer, sample_coords, used).add_until(len(objects))
    kappa, kappa_terms = {}, {}
    for c in used:
        kappa[c], kappa_terms[c] = _kappa(sample_canvas, c, background.grid(sample_coords, c), local[c][rows], column)
    del sample_canvas

    nuclei, cells, sites = _truth(placed, sample_coords, grid.omega)
    state = _AppearanceContext(context, codebook, stream, SimpleNamespace(layout=grid, nuclei=nuclei, cells=cells,
                                                                          sites=sites))
    scene = _generate(codebook, context.sample, on_fov, state.cellular, state.molecules, grow=grow)
    true = scene.placement[["true_origin_z", "true_origin_y", "true_origin_x"]].to_numpy(dtype=np.float64)
    if not np.array_equal(true, placed.true):
        raise RuntimeError("the sample scene's FOV origins differ from the placement's")
    # T1's reach does not apply: the territories are CT1's growth (cell_context) or the cells' own masks (CU1).
    scene.cells["reach"] = np.nan
    tables = _tables(context, codebook, scene, state)
    tables["lobes"] = tables["lobes"].iloc[:0].copy()

    fov_ids = list(scene.placement.fov_id)
    shifts = _translations(context, fov_ids, grid.s, stream)
    for fov in fov_ids:
        if shifts and not np.array_equal(shifts[fov][1], placed.shifts[fov]):
            raise RuntimeError("the morphology round's translation differs from the placement's")
    stains = context.stains
    rows_of = {name: [] for name in ("stain_channels", "morphology_truth")}
    effective_stains, appearance_stains = [], []
    for stain in stains:
        c = channel_of[stain.name]
        dtype = stain.output.dtype if stain.output is not None else template.dtype
        factor = g if stain.output is not None else 1.0
        rows_of["stain_channels"].append((stain.name, stain.source, stain.round, stain.channel, dtype, factor,
                                          np.nan))
        effective_stains.append(dict(name=stain.name, source=stain.source, round=stain.round, channel=stain.channel,
                                     dtype=dtype, scale=factor, m_A=None, background=None, noise=None))
        appearance_stains.append(dict(name=stain.name, round=stain.round, channel=stain.channel,
                                      bank_channel=bank.channels[c], dtype=dtype, scale=factor, kappa=kappa[c],
                                      kappa_terms=kappa_terms[c]))
    morphology = context.morphology
    for fov in fov_ids:
        if fov in shifts:
            vox, um = shifts[fov]
            rows_of["morphology_truth"].append((fov, morphology.name, *um.tolist(), *vox.tolist(),
                                                *(-vox + 0.0).tolist()))
    images, rounds = {}, {}
    n = grid.n
    for k, fov in enumerate(fov_ids):
        images[fov], planes, canvases = {}, [], {}
        for stain in stains:
            c = channel_of[stain.name]
            start = true[k] if stain.round == "reference" else true[k] + shifts[fov][1]
            if stain.round not in canvases:
                coords = [start[a] + np.arange(n[a]) * grid.s[a] for a in range(3)]
                canvases[stain.round] = (coords, _Canvas(composer, coords, used).add_until(len(objects)), {})
            coords, canvas, backgrounds = canvases[stain.round]
            if c not in backgrounds:
                backgrounds[c] = background.grid(coords, c)
            image, _, _ = _output(canvas.image(c, kappa[c], backgrounds[c]), stain, g, template.dtype)
            if stain.round == "reference":
                images[fov][stain.name] = image
            else:
                planes.append(image)
        if planes:
            rounds[fov] = np.stack(planes, axis=-1)
        del canvases

    tables.update(stain_channels=_stain_table("stain_channels", rows_of["stain_channels"]),
                  stain_objects=_stain_table("stain_objects", []), bright_regions=_stain_table("bright_regions", []),
                  unevenness=_stain_table("unevenness", []),
                  morphology_truth=_stain_table("morphology_truth", rows_of["morphology_truth"]))
    names = []
    for stain in stains:
        if stain.name not in names:
            names.append(stain.name)
    patches = pd.DataFrame(background.rows(names), columns=_PATCH_COLUMNS).astype(dict(
        stain=object, lattice_z=np.int64, lattice_y=np.int64, lattice_x=np.int64, patch_id=np.int64,
        source_image=object))

    morphology_effective = None
    if morphology is not None:
        morphology_effective = dict(name=morphology.name, translation=morphology.translation,
                                    translation_max_zyx=[float(v) for v in morphology.translation_max_zyx],
                                    domain_growth_um=(np.asarray(morphology.translation_max_zyx) * grid.s).tolist(),
                                    stains=[s.name for s in stains if s.round == "morphology"])
    territory = "CU1" if kind == "cell" else "CT1"
    effective = dict(
        type="CellularSceneConfig", territory_rule=territory, planar=bool(placed.record["planar"]),
        z_mid=float(placed.record["z_mid"]), harmonic_terms=[], harmonic_bound=None, empty_radius=None, packing=None,
        types=None if context.types is None else dict(
            names=list(context.types.types),
            proportions=_probabilities(context.types.proportions, len(context.types.types), "proportions").tolist()),
        stains=effective_stains, morphology=morphology_effective)
    streams = dict(placed.log)
    streams.update(log)
    context_record = dict(
        generator="starfinder.synthetic.generate_appearance_scene", context_truth_version=CELLULAR_TRUTH_VERSION,
        context_preset_version=CELLULAR_PRESET_VERSION, requested_config=_payload(context),
        effective_config=_plain(effective), seed=template.seed,
        stream_scheme=dict(key=list(_STREAM_KEY), namespace=_STREAM_NAMESPACE, encoding="compact UTF-8 JSON array",
                           digest="SHA-256, big-endian integer", bit_generator="PCG64", numpy_version=np.__version__,
                           components={k: dict(entity=e, draws=d) for k, (e, d) in {
                               **_CONTEXT_COMPONENTS, **_LAYOUT_COMPONENTS, **_COMPOSITION_COMPONENTS}.items()},
                           streams=list(streams.values())))
    context_scene = CellularScene(scene, images, rounds, **tables, record=context_record)

    placement = placed.record
    density_record = dict(placement["density"], threshold=threshold if layout_config.density == "visible" else None,
                          **density)
    components = {**_LAYOUT_COMPONENTS, **_COMPOSITION_COMPONENTS}
    drawn = [d for d in streams.values() if str(d[4]).startswith("context.appearance.")]
    reference = config.bank
    record = dict(
        generator="starfinder.synthetic.generate_appearance_scene",
        appearance_truth_version=APPEARANCE_TRUTH_VERSION, appearance_preset_version=_APPEARANCE_PRESET_VERSION,
        requested_config=_config_payload(config),
        effective_config=dict(
            type="AppearanceSceneConfig", kind=kind, territories=territory, planar=placement["planar"],
            z_mid=placement["z_mid"], placement_plane_z=placement["placement_plane_z"],
            culture_shift_um=placement["culture_shift_um"],
            culture_median_centroid_um=placement["culture_median_centroid_um"], rules=placement["rules"],
            augmentation=placement["augmentation"], layout=placement["layout"],
            composition=dict(light=composition.light, light_sigma_um=list(composition.light_sigma_um),
                             light_rule="H2" if composition.light == "objects" else "H3", background=mode,
                             observation="OB1: no observation stage added", output="O8", plateau=g),
            stains=appearance_stains),
        bank=dict(name=bank.name, sha256=bank.sha256, digest=bank.digest, preset=bank.record.get("preset"),
                  n=bank.record.get("n"), seeds=bank.record.get("seeds"), entries=len(bank),
                  plateau_median=float(bank.record["plateau_median"]), channels=list(bank.channels),
                  record_path=_record_path(reference), cycles=placement["bank"]["cycles"]),
        density=density_record, placed=int(len(objects)), proposals=int(placement["density"]["proposals"]),
        kappa={s["name"]: s["kappa"] for s in appearance_stains}, plateau=g,
        background=background.record(),
        stream_scheme=dict(namespace=_STREAM_NAMESPACE, split=template.split, seed=template.seed,
                           scene_key=template.scene_key,
                           components={k: dict(entity=e, draws=d) for k, (e, d) in components.items()},
                           drawn=sorted({d[4] for d in drawn}), streams=drawn))
    return AppearanceScene(context=context_scene, placements=placed.placements, background_patches=patches,
                           record=_plain(record))


# ------------------------------------------------------------------------------------------------ the evaluator


def _point_evaluator(scene, bank):
    """The stains of an appearance scene rebuilt from its truth (``placements``, ``background_patches``, the sample
    grid, the record's ``κ``, ``g`` and composition) and the bank: ``stain(name, points)`` gives ``I`` at (..., 3)
    sample positions (the point evaluator), ``grid(name, origin, shape)`` on the voxel centres ``origin + p ⊙ s``
    with the generator's grid arithmetic, and ``labels(points)`` gives (C, N)."""
    record, context = scene.record, scene.context
    growth = None
    if _bank_kind(bank) != "cell":
        order = context.cell_context.set_index("cell_id").growth
        growth = [float(order[int(k)]) for k in scene.placements.sort_values("object_id").object_id]
    labels = _labels_from_table(bank, scene.placements, growth)
    spacing = np.asarray(context.sample.record["sample_grid"]["spacing_zyx"], dtype=np.float64)
    composition = record["effective_config"]["composition"]
    composer = _Composer(bank, spacing, composition["light_sigma_um"], composition["light"])
    composer.objects = labels.objects
    if composition["background"] == "patches":
        domain = context.sample.record["domain"]["omega"]
        table = scene.background_patches
        table = table[table.stain == table.stain.iloc[0]] if len(table) else table
        draws = {(int(r.lattice_z), int(r.lattice_y), int(r.lattice_x)): int(r.patch_id) for r in table.itertuples()}
        background = _PatchBackground(bank, (domain["lo_zyx"], domain["hi_zyx"]), None, draws=draws)
    else:
        background = _ClosedFormBackground(bank, record["plateau"])
    kappa = record["kappa"]

    def stain(name, points):
        c = bank.channels.index(name)
        points = np.asarray(points, dtype=np.float64)
        return _compose_points(composer, c, kappa[name], background.points(points, c), points)

    def grid(name, origin, shape):
        c = bank.channels.index(name)
        coords = [float(origin[a]) + np.arange(int(shape[a])) * spacing[a] for a in range(3)]
        canvas = _Canvas(composer, coords, [c]).add_until(len(composer.objects))
        return canvas.image(c, kappa[name], background.grid(coords, c))
    return SimpleNamespace(stain=stain, grid=grid, labels=labels.labels, background=background, composer=composer)


# ------------------------------------------------------------------------------------------------ persistence


def save_appearance_scene(scene: AppearanceScene, directory: Path | str) -> Path:
    """Write an appearance scene to a new or empty directory and return its path.

    Layout (docs/synthetic-real-appearance.md, AT2): the ``save_cellular_scene`` layout of
    ``scene.context`` (``nuclei.csv`` and ``cells.csv`` with their columns, the shape columns
    null; ``lobes.csv``, ``harmonics.csv``, ``stain_objects.csv``, ``bright_regions.csv`` and
    ``unevenness.csv`` empty; ``fovs/<fov_id>/stains/<name>.ome.tif`` the composed stains;
    ``context.json``), plus ``placements.csv``, ``background_patches.csv`` and
    ``appearance.json`` (the record with ``appearance_truth_version`` and the SHA-256 of every
    other file). A scene generated with ``on_fov`` raises ValueError; an existing nonempty
    directory raises FileExistsError, and nothing is overwritten.
    """
    if not isinstance(scene, AppearanceScene):
        raise TypeError("scene must be AppearanceScene")
    directory = save_cellular_scene(scene.context, directory)
    scene.placements.to_csv(directory / "placements.csv", index=False)
    scene.background_patches.to_csv(directory / "background_patches.csv", index=False)
    files = {path.relative_to(directory).as_posix(): file_sha256(path)
             for path in sorted(directory.rglob("*")) if path.is_file()}
    record = dict(scene.record, files=files)
    (directory / "appearance.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return directory
