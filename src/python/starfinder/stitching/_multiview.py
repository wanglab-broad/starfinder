"""The ``multiview_stitcher`` placement method (design/specs/stitching-algorithms.md, "multiview_stitcher").

Importing this module registers ``MultiviewStitcherConfig`` in ``STITCHING_METHODS``. The
library is imported inside the method function only: importing ``starfinder.stitching``
never imports it, and ``stitch`` checks the dependency (the ``multiview-stitcher`` extra)
before the call.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from numbers import Integral, Real

import numpy as np
import pandas as pd

from starfinder._registry import Dependency

from ._methods import LINEAR_TOLERANCE, METHOD_LINK_COLUMNS, STITCHING_METHODS, MethodPlacement, StitchingSpec
from ._stitch import TileLoader

PAIRWISE = ("phase_correlation",)
PRUNING = ("alternating_pattern", None)
GROUPWISE = ("global_optimization",)
_NOMINAL = "nominal"


def _integer(value, name, low=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < low:
        raise ValueError(f"{name} must be an integer ≥ {low}; got {value!r}")


def _number(value, name, *, low=None, strict=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number; got {value!r}")
    if low is not None and (value <= low if strict else value < low):
        raise ValueError(f"{name} must be {'>' if strict else '≥'} {low}; got {value!r}")


@dataclass(frozen=True)
class MultiviewStitcherConfig:
    """Place overlapping FOVs by phase correlation and global translation optimization.

    Parameters
    ----------
    binning_zyx : tuple of int
        Positive mean-binning factors for pairwise registration: ZYX voxels in 3D and YX voxels in 2D.
    pairwise : str
        Pairwise registration method, ``phase_correlation``.
    upsample_factor : int or None
        Positive subpixel upsampling factor for phase correlation.
        ``None`` uses the library default: 10 in 2D and 2 in 3D.
    quality_threshold : float
        Lower Spearman-correlation quality threshold for keeping a registered link.
    drop_nan_quality : bool
        Drop registered links whose quality is NaN.
    pruning : str or None
        Overlap-graph pruning method, ``alternating_pattern``; ``None`` disables pruning.
    overlap_tolerance_um : float
        Nonnegative box-growth tolerance for neighbour derivation and library overlap regions, in µm.
    groupwise : str
        Groupwise resolution method, ``global_optimization``.
    abs_tol_um : float or None
        Positive absolute residual tolerance for global optimization, in µm.
        ``None`` uses the library's maximum voxel diagonal across graph nodes.
    rel_tol : float or None
        Positive relative-change tolerance for global optimization.
        ``None`` uses the library default, 0.0001.
    max_iter : int or None
        Positive iteration budget for global optimization.
        ``None`` uses the library default, 500.
    method : str
        Method identifier, ``multiview_stitcher``.

    Notes
    -----
    Phase correlation on each registered pair's overlap, mean-binned by
    ``binning_zyx`` voxels (z ignored for Z=1 tiles, which are registered as
    planes), each shift quantized to ``binning / upsample_factor`` voxels
    (``None``: the library's default, 2 in 3D and 10 in 2D). The library's
    overlap graph is pruned by ``pruning`` (``"alternating_pattern"`` or
    ``None``, every pair registered). Link rule L1: a link is kept when its
    quality (Spearman correlation of the overlaps after the shift) is at least
    ``quality_threshold``; with ``drop_nan_quality`` a NaN-quality link, which
    the library keeps, is dropped as well. The kept links enter the library's
    global optimization (translation), whose ``abs_tol_um``, ``rel_tol`` and
    ``max_iter`` are the library's defaults when ``None`` (``abs_tol`` is then
    the voxel diagonal in µm). ``overlap_tolerance_um`` grows the boxes of the
    neighbour derivation and of the library's overlap regions. CPU only.

    Memory: each tile of a registered pair is read once (a ``TileLoader``
    input is called then), binned by ``binning_zyx`` and released, so a run
    from loaders, as ``Dataset.stitch`` passes them, holds at most one
    full-resolution tile and the binned tiles (1/64 of the pixels at the
    default binning). An array input stays held by its caller. With
    ``binning_zyx`` (1, 1, 1) nothing is binned and every tile is kept at full
    resolution. The links and translations equal those of the library binning
    each pair itself, bit for bit.

    Raises
    ------
    ValueError
        A field outside the rules of design/specs/stitching-algorithms.md: a binning
        that is not three integers ≥ 1, an ``upsample_factor`` below 1, a
        ``quality_threshold`` outside [−1, 1], a negative
        ``overlap_tolerance_um``, a ``pairwise``, ``pruning`` or ``groupwise``
        value the table does not list, a non-positive tolerance or an
        iteration count below 1.
    TypeError
        ``drop_nan_quality`` is not Boolean.
    """
    # D3

    binning_zyx: tuple[int, int, int] = (4, 4, 4)
    pairwise: str = "phase_correlation"
    upsample_factor: int | None = None
    quality_threshold: float = 0.7
    drop_nan_quality: bool = True
    pruning: str | None = "alternating_pattern"
    overlap_tolerance_um: float = 0.0
    groupwise: str = "global_optimization"
    abs_tol_um: float | None = None
    rel_tol: float | None = None
    max_iter: int | None = None
    method: str = field(default="multiview_stitcher", init=False)

    def __post_init__(self):
        binning = self.binning_zyx
        if isinstance(binning, (str, bytes)) or not hasattr(binning, "__len__") or len(binning) != 3:
            raise ValueError(f"binning_zyx must be three integers ≥ 1; got {binning!r}")
        for value in binning:
            _integer(value, "binning_zyx")
        object.__setattr__(self, "binning_zyx", tuple(int(v) for v in binning))
        if self.pairwise not in PAIRWISE:
            raise ValueError(f"pairwise must be one of {list(PAIRWISE)}; got {self.pairwise!r}")
        if self.upsample_factor is not None:
            _integer(self.upsample_factor, "upsample_factor")
        _number(self.quality_threshold, "quality_threshold")
        if not -1.0 <= self.quality_threshold <= 1.0:
            raise ValueError(f"quality_threshold must be in [-1, 1]; got {self.quality_threshold!r}")
        if not isinstance(self.drop_nan_quality, (bool, np.bool_)):
            raise TypeError("drop_nan_quality must be Boolean")
        if self.pruning not in PRUNING:
            raise ValueError(f"pruning must be one of {list(PRUNING)}; got {self.pruning!r}")
        _number(self.overlap_tolerance_um, "overlap_tolerance_um", low=0.0)
        if self.groupwise not in GROUPWISE:
            raise ValueError(f"groupwise must be one of {list(GROUPWISE)}; got {self.groupwise!r}")
        for name in ("abs_tol_um", "rel_tol"):
            if getattr(self, name) is not None:
                _number(getattr(self, name), name, low=0.0, strict=True)
        if self.max_iter is not None:
            _integer(self.max_iter, "max_iter")


def _scalar(value):
    return float(np.asarray(value).squeeze())


def _translation(affine, ndim):
    """The translation column of a library affine, unchanged; raise if its linear part is not the identity.

    An entry within ``LINEAR_TOLERANCE`` of the identity, the import's, counts as the identity.
    """
    matrix = np.asarray(affine, dtype=np.float64).squeeze()
    if matrix.shape != (ndim + 1, ndim + 1):
        raise ValueError(f"multiview-stitcher returned an affine of shape {matrix.shape}; expected "
                         f"{(ndim + 1, ndim + 1)}")
    if not np.abs(matrix[:ndim, :ndim] - np.eye(ndim)).max() <= LINEAR_TOLERANCE:
        raise ValueError("multiview-stitcher returned a transform that is not a translation")
    return matrix[:ndim, ndim]


def _labels(n, present, edges):
    """Components of ``edges`` over the present tiles, numbered by first tile in table order; −1 for absent."""
    parent = list(range(n))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in edges:
        ra, rb = root(a), root(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    labels, out = {}, np.full(n, -1, dtype=np.int64)
    for i in present:
        out[i] = labels.setdefault(root(i), len(labels))
    return out


def _dtype(tile):
    return tile.dtype if isinstance(tile, TileLoader) else np.asarray(tile).dtype


def _msim(sim):
    """The one-level multiscale image of ``sim``, its channel axis selected as register() does before the graph."""
    from multiview_stitcher import msi_utils
    msim = msi_utils.get_msim_from_sim(sim, scale_factors=[])
    if "c" in msi_utils.get_dims(msim):
        msim = msi_utils.multiscale_sel_coords(msim, {"c": sim.coords["c"].values[0]})
    return msim


def _integer_means(array, factors):
    """The window means of an integer array, trimmed to whole windows, cast back to its dtype; None for another dtype.

    Each window's sum is exact in an integer accumulator and in float64 (below 2**53), so the float64 mean,
    sum / window size, and its cast are those of the library's float64 mean, bit for bit. Sums are taken one axis at
    a time by adding strided slices, which is several times faster than the library's reduction.
    """
    dtype, count = array.dtype, math.prod(factors)
    if dtype.kind not in "ui":
        return None
    info = np.iinfo(dtype)
    bound = max(abs(int(info.min)), int(info.max)) * count
    if bound >= 2 ** 53:
        return None
    candidates = (np.uint16, np.uint32, np.uint64) if dtype.kind == "u" else (np.int16, np.int32, np.int64)
    accumulator = next(t for t in candidates if np.iinfo(t).max >= bound)
    out = array[tuple(slice(0, n - n % f) for n, f in zip(array.shape, factors))]
    for axis in reversed(range(array.ndim)):
        window = [slice(None)] * array.ndim
        window[axis] = slice(0, None, factors[axis])
        total = out[tuple(window)].astype(accumulator)
        for k in range(1, factors[axis]):
            window[axis] = slice(k, None, factors[axis])
            total += out[tuple(window)]
        out = total
    return (out.astype(np.float64) / count).astype(dtype)


def _binned_msim(tile, plane, binning, geometry):
    """The tile's image binned once, as the library bins it for each of its pairs; the full-resolution pixels are
    read once (a TileLoader is called here) and released before the next tile is read.

    The binning is register_pair_of_msims's own, ``coarsen(binning, boundary="trim").mean().astype(dtype)`` on the
    library's image of the tile: its coordinates and attributes, and, for an integer tile, values computed by
    :func:`_integer_means`, which equal the library's; a floating-point tile is computed by the library's own
    reduction. With every factor 1 there is nothing to bin, and the image keeps the full-resolution tile, as the
    library would.
    """
    from multiview_stitcher import spatial_image_utils as si_utils
    array = tile() if isinstance(tile, TileLoader) else np.asarray(tile)
    image = array[0] if plane else array
    sim = si_utils.get_sim_from_array(image, **geometry)
    if max(binning.values()) > 1:
        full = sim
        sim = full.coarsen(binning, boundary="trim").mean().astype(full.dtype)
        sim.attrs.update(copy.deepcopy(full.attrs))
        means = _integer_means(image, [binning[d] for d in geometry["dims"]])
        sim = sim.compute() if means is None else sim.copy(data=means.reshape(sim.shape))
        del full
    del array, image
    return _msim(sim)


def _pairwise_registrations(msims, binned, graph, config):
    """The library's pairwise registrations of ``graph``'s edges on the binned images (binning 1), each link's
    ``bbox`` taken from the full-resolution geometry.

    The library takes a link's ``bbox`` (the overlap box whose corners the global optimization uses as virtual
    beads) from the images it registers: from the binned images it would be off by up to half a bin. It is
    recomputed with the function the library calls for it, ``registration._get_overlap_bboxes``, on the
    full-resolution images ``msims`` (geometry only), with the library's arguments, so that the links, and so the
    placement, are those of the library binning each pair itself. The private function is pinned by a test.
    """
    from multiview_stitcher import msi_utils, registration
    dims = msi_utils.get_spatial_dims(msims[0])
    kwargs = {} if config.upsample_factor is None else {"upsample_factor": config.upsample_factor}
    computed = registration.compute_pairwise_registrations(
        binned, graph, transform_key=_NOMINAL, registration_binning={d: 1 for d in dims},
        overlap_tolerance=config.overlap_tolerance_um, pairwise_reg_func=registration.phase_correlation_registration,
        pairwise_reg_func_kwargs=kwargs)
    # register_pair_of_msims turns a number into one tolerance per spatial axis.
    tolerance = {d: float(config.overlap_tolerance_um) for d in dims}
    for edge in computed.edges:
        sims = [msi_utils.get_sim_from_msim(msims[k], scale="scale0") for k in sorted(edge)]
        box = registration._get_overlap_bboxes(*sims, input_transform_key=_NOMINAL, output_transform_key=_NOMINAL,
                                               overlap_tolerance=tolerance)
        old = computed.edges[edge]["bbox"]
        computed.edges[edge]["bbox"] = old.copy(data=np.array([box["lowers"][0], box["uppers"][0]]).reshape(old.shape))
    return computed


def _multiview_stitcher(tiles, grids, nominal_zyx, pairs, *, config: MultiviewStitcherConfig) -> MethodPlacement:
    """Place the tiles with multiview-stitcher in one call (design/specs/stitching-algorithms.md, steps 1 to 7).

    1. Each present tile becomes a one-level image of its grid's shape (no
       pixels: a zero-stride array) with its grid's spacing and its nominal
       origin (µm) as the array translation; Z=1 tiles are passed as 2D
       (y, x), so their z translation is exactly 0.
    2. The library's overlap graph (``overlap_tolerance_um``) is restricted to
       the contract's neighbour pairs, so the registered pairs are a subset of
       them, and pruned by ``pruning``; every other neighbour pair is left
       unregistered, which ``stitch`` records as ``pruned``.
    3. and 4. Each tile of a registered pair is read once, in table order (a
       :class:`TileLoader` is called then), binned by ``binning_zyx`` as the
       library bins it for a pair, and released, so that at most one
       full-resolution tile read from a loader is held at a time. Phase
       correlation per registered pair on the binned images (binning 1) at
       ``upsample_factor``; each link's overlap box (the global optimization's
       virtual beads) is recomputed from the full-resolution geometry with
       the library's ``registration._get_overlap_bboxes``, so the links equal
       the library's own binning of each pair bit for bit. The link's shift
       (µm, of ``fov_b`` against ``fov_a``, view to common frame) is the
       negated translation of the library's fixed-to-moving transform, and its
       quality the library's.
    5. Link rule L1: quality below ``quality_threshold`` gives ``kept`` false
       with ``quality_below_threshold``; NaN quality with ``drop_nan_quality``
       gives ``nan_quality``.
    6. The library's global optimization (translation) on the kept links; a
       kept link the optimizer removes becomes ``optimizer_removed``.
    7. Per tile, the translation of its resolved affine (µm, fractional); an
       absent tile gets NaN. With no link left, or no overlap at all, the
       library raises ``NotEnoughOverlapError``: every translation is NaN, so
       ``stitch`` makes every FOV ``failed``, and the error is recorded in the
       diagnostics.

    The diagnostics hold the outcome, the registration dimensions, the
    library's overlap pairs (and, as table rows, any outside the neighbour
    pairs, which are dropped), the pruned and the optimizer-removed pairs, and
    the edge residuals (µm).
    """
    from multiview_stitcher import mv_graph, param_resolution
    from multiview_stitcher import spatial_image_utils as si_utils

    n = len(tiles)
    present = [i for i, tile in enumerate(tiles) if tile is not None]
    plane = bool(grids) and grids[0].shape_zyx[0] == 1
    axes = (1, 2) if plane else (0, 1, 2)
    dims = [("z", "y", "x")[k] for k in axes]
    ndim = len(axes)
    translations = np.full((n, 3), np.nan)
    by_index = {(p.index_a, p.index_b): p for p in pairs}
    diagnostics = {"outcome": "completed", "registration_dims": "".join(dims)}
    rows = []

    def link_rows(registered, quality, shift, reasons):
        for e in registered:
            pair = by_index[(present[e[0]], present[e[1]])]
            rows.append((pair.fov_a, pair.fov_b, *shift[e], quality[e], reasons[e] == "", reasons[e]))

    def fov_pairs(edges):
        return [[by_index[(present[a], present[b])].fov_a, by_index[(present[a], present[b])].fov_b]
                for a, b in edges]

    if not present:
        diagnostics["outcome"] = "no_tile"
        return MethodPlacement(translations, np.full(n, -1, dtype=np.int64),
                               pd.DataFrame([], columns=list(METHOD_LINK_COLUMNS)), diagnostics)

    # Each tile's full-resolution geometry, without pixels (a zero-stride array): the overlap graph and each
    # link's bbox come from these images, and the binned images from the pixels.
    msims, geometries = [], []
    for i in present:
        geometries.append({"dims": dims, "transform_key": _NOMINAL,
                           "scale": {dims[j]: float(grids[i].metadata.spacing_zyx[k]) for j, k in enumerate(axes)},
                           "translation": {dims[j]: float(nominal_zyx[i, k]) for j, k in enumerate(axes)}})
        shape = grids[i].shape_zyx[1:] if plane else grids[i].shape_zyx
        msims.append(_msim(si_utils.get_sim_from_array(np.broadcast_to(np.zeros((), _dtype(tiles[i])), shape),
                                                       **geometries[-1])))

    graph = mv_graph.build_view_adjacency_graph_from_msims(msims, transform_key=_NOMINAL,
                                                           overlap_tolerance=config.overlap_tolerance_um)
    outside = sorted(tuple(sorted(e)) for e in graph.edges if (present[min(e)], present[max(e)]) not in by_index)
    graph.remove_edges_from(outside)
    diagnostics["overlap_pairs"] = fov_pairs(sorted(tuple(sorted(e)) for e in graph.edges))
    diagnostics["overlap_rows_outside_neighbor_pairs"] = [[present[a], present[b]] for a, b in outside]
    try:
        pruned_graph = mv_graph.prune_view_adjacency_graph(graph, method=config.pruning)
    except mv_graph.NotEnoughOverlapError as exc:
        diagnostics.update(outcome="not_enough_overlap", error=str(exc), pruned_pairs=[], optimizer_removed_links=[])
        return MethodPlacement(translations, _labels(n, present, []), pd.DataFrame(rows, columns=list(
            METHOD_LINK_COLUMNS)), diagnostics)
    registered = sorted(tuple(sorted(e)) for e in pruned_graph.edges)
    diagnostics["pruned_pairs"] = fov_pairs(sorted(set(tuple(sorted(e)) for e in graph.edges) - set(registered)))

    # The pixels: each tile of a registered pair is read once, in table order, and only its binned image is kept.
    binning = {dims[j]: config.binning_zyx[k] for j, k in enumerate(axes)}
    needed = {k for e in registered for k in e}
    binned = [_binned_msim(tiles[i], plane, binning, geometries[k]) if k in needed else msims[k]
              for k, i in enumerate(present)]
    computed = _pairwise_registrations(msims, binned, pruned_graph, config)

    quality, shift, reasons = {}, {}, {}
    for e in registered:
        quality[e] = _scalar(computed.edges[e]["quality"])
        full = np.zeros(3)
        full[list(axes)] = -_translation(computed.edges[e]["transform"], ndim)
        shift[e] = full
        if math.isnan(quality[e]):
            reasons[e] = "nan_quality" if config.drop_nan_quality else ""
        else:
            reasons[e] = "quality_below_threshold" if quality[e] < config.quality_threshold else ""
    kept = computed.copy()
    kept.remove_edges_from([e for e in registered if reasons[e]])

    options = {"transform": "translation"}
    for name, key in (("abs_tol_um", "abs_tol"), ("rel_tol", "rel_tol"), ("max_iter", "max_iter")):
        if getattr(config, name) is not None:
            options[key] = getattr(config, name)
    try:
        params, info = param_resolution.groupwise_resolution(kept, method=config.groupwise, **options)
    except mv_graph.NotEnoughOverlapError as exc:
        diagnostics.update(outcome="not_enough_overlap", error=str(exc), optimizer_removed_links=[])
        link_rows(registered, quality, shift, reasons)
        return MethodPlacement(translations, _labels(n, present, []),
                               pd.DataFrame(rows, columns=list(METHOD_LINK_COLUMNS)), diagnostics)

    used = {tuple(sorted(e)) for edges in info.get("used_edges", {}).values() for e in edges}
    removed = [e for e in registered if not reasons[e] and e not in used]
    for e in removed:
        reasons[e] = "optimizer_removed"
    for j, i in enumerate(present):
        translations[i] = 0.0   # z of a plane stays exactly 0
        translations[i, list(axes)] = _translation(params[j], ndim)
    link_rows(registered, quality, shift, reasons)
    residuals = {tuple(sorted(e)): float(v) for values in info.get("edge_residuals", {}).values()
                 for e, v in values.items()}
    diagnostics["optimizer_removed_links"] = fov_pairs(removed)
    diagnostics["edge_residuals_um"] = [[*fov_pairs([e])[0], residuals[e]] for e in sorted(residuals)]
    components = _labels(n, present, [(present[a], present[b]) for a, b in registered if not reasons[(a, b)]])
    return MethodPlacement(translations, components, pd.DataFrame(rows, columns=list(METHOD_LINK_COLUMNS)),
                           diagnostics)


STITCHING_METHODS[MultiviewStitcherConfig] = StitchingSpec(
    "multiview_stitcher", _multiview_stitcher,
    requires=(Dependency("multiview_stitcher", "multiview-stitcher", extra="multiview-stitcher"),),
    min_shape_zyx=(1, 16, 16), dimensions=frozenset({2, 3}), devices=("cpu",))
