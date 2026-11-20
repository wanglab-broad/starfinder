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
# The link-quality threshold estimated from the data (quality_threshold "auto"; W-874, Jiahao, 2026-11-18):
# max(QUALITY_FLOOR, the NULL_QUANTILE quantile of the qualities at deliberately wrong offsets).
AUTO = "auto"
QUALITY_FLOOR = 0.2
NULL_QUANTILE = 0.99
# The default of quality_threshold (None) per tile set (decision B, Jiahao, 2026-11-19): "auto" for single-plane tiles,
# FIXED_DEFAULT for tiles with Z > 1, whose z error the in-plane null cannot see. W-895 adds the per-link z check to
# "auto" on tiles with Z > 1; the default stays 0.7 until W-904's LN evidence (Jiahao, 2026-11-19, "Yes, adopt it").
FIXED_DEFAULT = 0.7
FIXED_DEFAULT_RULE = "fixed_default_z_above_1"
# The per-link z check of "auto" on tiles with Z > 1 (W-895): each link's quality at the registered shift ± d planes
# of the binned overlap, d = max(1, round(Z_OFFSET_UM / z step)); a link scoring higher at either offset is dropped.
Z_OFFSET_UM = 1.0
Z_DROP_REASON = "quality_higher_at_z_offset"
Z_CHECK_RULE = ("a link is kept only if its quality at the registered shift is at least its quality at each z offset "
                "that has a value")
Z_OUTCOMES = ("passed", Z_DROP_REASON, "skipped_thin_overlap", "skipped_nan_quality")
# The fraction of the moving overlap that must stay valid after a shift, the library's (phase_correlation_registration).
_MIN_VALID_FRACTION = 0.1


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
    quality_threshold : float, str or None
        Lower Spearman-correlation quality threshold for keeping a registered link: a number in
        [−1, 1], or ``"auto"`` for the threshold estimated from the data, the larger of 0.2 and
        the 99th percentile of the qualities at deliberately wrong offsets, with a per-link z check
        on tiles with Z > 1. ``None`` (the default) uses ``"auto"`` when every tile is single-plane
        (Z = 1) and 0.7 when tiles have Z > 1.
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

    The threshold ``"auto"`` is estimated from the data. For each registered
    pair, the pair's quality is also measured at two deliberately wrong
    offsets: the registered shift moved by half the overlap's width, in binned
    voxels rounded down, both ways along the overlap axis (the narrower of Y
    and X in the binned overlap). Where the moved overlap keeps less than a
    tenth of its voxels, or its quality is NaN, there is no value. These
    qualities are a null distribution, where no tissue lines up, and the
    threshold is ``max(0.2, q99)``, ``q99`` being their 99th percentile
    (NumPy's linear quantile); without any null value it is 0.2. The
    record's ``method_diagnostics["link_quality"]`` holds the rule
    (``"auto"``), the chosen threshold, the floor, the quantile, the null's
    summary and each pair's two null values.

    The null moves the overlap in Y and X and cannot see a z error, so on
    tiles with Z > 1 ``"auto"`` also checks each link in z. The link's
    quality is measured, as the null's, at the registered shift moved by
    ``d`` planes of the binned overlap both ways along z, with
    ``d = max(1, round(1 µm / z step))``, the z step being the z spacing
    times ``binning_zyx[0]`` (Python's ``round``, half to even; 3 planes at
    0.35 µm without binning, 1 at LN's 0.346 µm with binning 4). A link is
    kept only if its quality at the registered shift is at least its quality
    at each z offset that has a value; an offset with too little valid
    overlap (NaN) does not count against it. A link that passes the
    threshold but fails this check gets ``kept`` false and the reason
    ``quality_higher_at_z_offset``. A pair whose binned overlap has fewer
    than ``2 d + 1`` planes skips the check. ``link_quality["z_check"]``
    holds ``offset_planes`` (``d``), ``offset_um``, ``z_step_um``,
    ``min_planes``, the ``rule``, the number ``dropped`` and, per registered
    link, ``[fov_a, fov_b, quality at −d, quality, quality at +d, outcome]``
    (``None`` for a NaN and for the offsets of a skipped pair), the outcome
    being ``passed``, ``quality_higher_at_z_offset``,
    ``skipped_thin_overlap`` or ``skipped_nan_quality``. Single-plane tiles,
    registered in 2D, have no z check and no ``z_check`` entry.

    The default, ``None``, is ``"auto"`` for single-plane tiles only; tiles
    with Z > 1 get 0.7, recorded as ``{"rule": "fixed_default_z_above_1",
    "threshold": 0.7}`` with no null and no z check measured, until the LN
    evidence (W-904) decides whether ``"auto"`` with its z check becomes
    their default. An explicit ``"auto"`` applies to either kind of tile. A
    given number is used as it is: no null and no z check are measured, the
    record's method config holds the number, and ``method_diagnostics``
    gains no ``link_quality`` entry. The global optimization's outlier
    removal (``abs_tol_um``, ``rel_tol``) stays the second line.

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
        ``quality_threshold`` that is neither ``None``, ``"auto"`` nor a number in [−1, 1], a negative
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
    quality_threshold: float | str | None = None
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
        if self.quality_threshold is not None and not (isinstance(self.quality_threshold, str)
                                                       and self.quality_threshold == AUTO):
            if isinstance(self.quality_threshold, str):
                raise ValueError(f"quality_threshold must be None, 'auto' or a number in [-1, 1]; "
                                 f"got {self.quality_threshold!r}")
            _number(self.quality_threshold, "quality_threshold")
            if not -1.0 <= self.quality_threshold <= 1.0:
                raise ValueError(f"quality_threshold must be None, 'auto' or a number in [-1, 1]; "
                                 f"got {self.quality_threshold!r}")
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


def _quality_at(fixed, moving, shift):
    """The link quality of ``moving`` shifted by ``shift`` (voxels) against ``fixed``, as the library measures it.

    The steps of the library's ``phase_correlation_registration`` for one candidate shift: each overlap
    rescaled to [0, 1] over its own finite range, ``moving`` resampled at ``x + shift`` (linear, NaN outside),
    and the Spearman correlation over the voxels valid in both. NaN where less than a tenth of the moving
    overlap's valid voxels stays (the library's −1), or where the correlation is undefined (a constant side).
    """
    from scipy import ndimage, stats
    from skimage.exposure import rescale_intensity
    im0, im1 = (rescale_intensity(np.asarray(image), in_range=(np.nanmin(image), np.nanmax(image)), out_range=(0, 1))
                for image in (fixed, moving))
    ndim = im0.ndim
    matrix = np.eye(ndim + 1)
    matrix[:ndim, ndim] = shift
    moved = ndimage.affine_transform(im1, matrix, order=1, mode="constant", cval=np.nan)
    mask = ~np.isnan(moved) & ~np.isnan(im0)
    if not mask.any() or float(mask.sum()) / np.sum(~np.isnan(im1)) < _MIN_VALID_FRACTION:
        return math.nan
    a, b = im0[mask], moved[mask] - 1
    if np.all(a == a[0]) or np.all(b == b[0]):
        return math.nan
    return float(stats.spearmanr(a, b).correlation)


def _null_offsets(shape, shift):
    """The two deliberately wrong shifts of a registered pair: ``shift`` ± half the overlap width (voxels,
    rounded down, at least 1) along the overlap axis, the narrower of the last two axes (Y and X)."""
    shift = np.asarray(shift, dtype=np.float64)
    axis = len(shape) - 2 + int(np.argmin(shape[-2:]))
    step = np.zeros(len(shape))
    step[axis] = max(1, shape[axis] // 2)
    return shift - step, shift + step


def _null_qualities(fixed, moving, shift):
    """The pair's qualities at :func:`_null_offsets`, in that order (NaN where there is none)."""
    return [_quality_at(fixed, moving, offset) for offset in _null_offsets(np.shape(fixed), shift)]


def _z_offset_planes(spacing_z_um, binning_z):
    """The z offset ``d`` of the z check in planes of the binned overlap: ``max(1, round(1 µm / z step))``, the z step
    being ``spacing_z_um × binning_z`` (Python's ``round``, half to even); 3 at 0.35 µm without binning."""
    return max(1, round(Z_OFFSET_UM / (float(spacing_z_um) * int(binning_z))))


def _z_qualities(fixed, moving, shift, d):
    """The pair's qualities at the registered ``shift`` moved by −``d`` and +``d`` planes along z (axis 0), in that
    order (NaN where there is none); ``None`` when the binned overlap has fewer than ``2 d + 1`` planes."""
    if np.shape(fixed)[0] < 2 * d + 1:
        return None
    step = np.zeros(np.ndim(fixed))
    step[0] = d
    shift = np.asarray(shift, dtype=np.float64)
    return [_quality_at(fixed, moving, shift - step), _quality_at(fixed, moving, shift + step)]


def _z_outcome(quality, z):
    """The z check's outcome for a link of ``quality`` with :func:`_z_qualities` ``z``: ``"skipped_nan_quality"``
    (the library gives a constant overlap NaN quality without calling the pairwise function, so ``z`` is then
    ``None`` too), ``"skipped_thin_overlap"`` (``z`` is ``None``), ``"quality_higher_at_z_offset"`` when a z offset
    with a value (not NaN) scores higher, else ``"passed"``."""
    if math.isnan(quality):
        return "skipped_nan_quality"
    if z is None:
        return "skipped_thin_overlap"
    return Z_DROP_REASON if any(q > quality for q in z if not math.isnan(q)) else "passed"


def _recording(null, z=None, d=None):
    """The library's phase correlation, which also appends the pair's :func:`_null_qualities` to ``null`` and, with
    ``d``, its :func:`_z_qualities` (one entry, ``None`` for a thin overlap) to ``z``."""
    from multiview_stitcher import registration

    def pairwise(fixed_data, moving_data, **kwargs):
        result = registration.phase_correlation_registration(fixed_data, moving_data, **kwargs)
        if isinstance(result, dict):
            ndim = np.ndim(fixed_data)
            shift = np.asarray(result["affine_matrix"], dtype=np.float64)[:ndim, ndim]
            fixed, moving = np.asarray(fixed_data.data), np.asarray(moving_data.data)
            null.extend(_null_qualities(fixed, moving, shift))
            if d is not None:
                z.append(_z_qualities(fixed, moving, shift, d))
        return result

    return pairwise


def _null_executor(nulls, zs=None, d=None):
    """A ``pairwise_executor`` for the library: each edge registered on its own, as the library's default path
    registers it, by a :func:`_recording` function of its own, whose null qualities go to ``nulls[edge]`` and, with
    ``d``, whose z qualities go to ``zs[edge]``."""
    from dask import compute
    from multiview_stitcher import registration

    def execute(msims, edges, register_kwargs):
        params = []
        for edge in edges:
            z = None if d is None else zs.setdefault(edge, [])
            kwargs = dict(register_kwargs, pairwise_reg_func=_recording(nulls.setdefault(edge, []), z, d),
                          pairwise_reg_func_kwargs=dict(register_kwargs.get("pairwise_reg_func_kwargs") or {}))
            params.append(compute(registration.register_pair_of_msims_over_time(
                msims[edge[0]], msims[edge[1]], **kwargs))[0])
        return params

    return execute


def _resolved_threshold(quality_threshold, plane):
    """(rule, threshold) of a configured ``quality_threshold`` for a tile set: ``("auto", None)`` for ``"auto"`` and
    for the default on single-plane tiles (``plane``), ``(FIXED_DEFAULT_RULE, 0.7)`` for the default on tiles with
    Z > 1, and ``("given", value)`` for a number."""
    if quality_threshold is None:
        return (AUTO, None) if plane else (FIXED_DEFAULT_RULE, FIXED_DEFAULT)
    if isinstance(quality_threshold, str):
        return AUTO, None
    return "given", quality_threshold


def _auto_threshold(nulls, registered, fov_pairs):
    """The threshold of ``"auto"`` and the record of its null (``method_diagnostics["link_quality"]``)."""
    per_pair = [[*fov_pairs([e])[0], *[None if math.isnan(q) else q for q in nulls.get(e, [])]] for e in registered]
    values = np.array([q for e in registered for q in nulls.get(e, []) if math.isfinite(q)], dtype=np.float64)
    quantile = float(np.quantile(values, NULL_QUANTILE)) if values.size else None
    threshold = QUALITY_FLOOR if quantile is None else max(QUALITY_FLOOR, quantile)
    summary = {"pairs": sum(any(math.isfinite(q) for q in nulls.get(e, [])) for e in registered),
               "values": int(values.size)}
    if values.size:
        summary.update(min=float(values.min()), median=float(np.median(values)), quantile=quantile,
                       max=float(values.max()))
    return threshold, {"rule": AUTO, "threshold": threshold, "floor": QUALITY_FLOOR, "quantile": NULL_QUANTILE,
                       "offsets": "the registered shift ± half the overlap width along the overlap axis",
                       "null": summary, "null_pairs": per_pair}


def _z_check(d, z_step_um, zs, quality, registered, fov_pairs):
    """Each registered link's :func:`_z_outcome` and the record of the z check (``link_quality["z_check"]``).

    ``zs[edge]`` holds the edge's :func:`_z_qualities` entry and ``quality[edge]`` its quality at the registered
    shift. Each link's row is ``[fov_a, fov_b, quality at −d, quality, quality at +d, outcome]``, ``None`` for a NaN
    and for the two offsets of a thin overlap.
    """
    outcomes, links = {}, []
    for e in registered:
        z = (zs.get(e) or [None])[-1]
        outcomes[e] = _z_outcome(quality[e], z)
        below, above = (None, None) if z is None else z
        links.append([*fov_pairs([e])[0], *[None if v is None or math.isnan(v) else v
                                            for v in (below, quality[e], above)], outcomes[e]])
    return outcomes, {"offset_planes": d, "offset_um": Z_OFFSET_UM, "z_step_um": z_step_um, "min_planes": 2 * d + 1,
                      "rule": Z_CHECK_RULE, "dropped": sum(o == Z_DROP_REASON for o in outcomes.values()),
                      "links": links}


def _pairwise_registrations(msims, binned, graph, config, nulls=None, zs=None, d=None):
    """The library's pairwise registrations of ``graph``'s edges on the binned images (binning 1), each link's
    ``bbox`` taken from the full-resolution geometry.

    The library takes a link's ``bbox`` (the overlap box whose corners the global optimization uses as virtual
    beads) from the images it registers: from the binned images it would be off by up to half a bin. It is
    recomputed with the function the library calls for it, ``registration._get_overlap_bboxes``, on the
    full-resolution images ``msims`` (geometry only), with the library's arguments, so that the links, and so the
    placement, are those of the library binning each pair itself. The private function is pinned by a test.

    With ``nulls`` (a dict), each edge is registered through :func:`_null_executor`, which fills ``nulls[edge]``
    with its null qualities and, with ``d``, ``zs[edge]`` with its z qualities; the registrations are the same.
    """
    from multiview_stitcher import msi_utils, registration
    dims = msi_utils.get_spatial_dims(msims[0])
    kwargs = {} if config.upsample_factor is None else {"upsample_factor": config.upsample_factor}
    executor = {} if nulls is None else {"pairwise_executor": _null_executor(nulls, zs, d)}
    computed = registration.compute_pairwise_registrations(
        binned, graph, transform_key=_NOMINAL, registration_binning={dim: 1 for dim in dims},
        overlap_tolerance=config.overlap_tolerance_um, pairwise_reg_func=registration.phase_correlation_registration,
        pairwise_reg_func_kwargs=kwargs, **executor)
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
       quality the library's. The threshold's rule comes from
       ``quality_threshold`` and the tiles: ``"auto"``, or the default on
       single-plane tiles; 0.7 for the default on tiles with Z > 1; or the
       given number. With the rule ``"auto"`` each pair
       is registered on its own (the library's ``pairwise_executor``), with
       the same result, and its two null qualities are measured on the
       library's own overlap arrays; the threshold is ``max(0.2, q99)`` of
       them (``MultiviewStitcherConfig``), recorded in the diagnostics'
       ``link_quality``. On tiles with Z > 1, ``"auto"`` also measures each
       pair's quality at the registered shift ± ``d`` planes along z, on the
       same arrays (the z check, recorded in ``link_quality["z_check"]``).
    5. Link rule L1: quality below the threshold gives ``kept`` false
       with ``quality_below_threshold``; NaN quality with ``drop_nan_quality``
       gives ``nan_quality``. A link that passes the threshold and fails the
       z check gives ``quality_higher_at_z_offset``.
    6. The library's global optimization (translation) on the kept links; a
       kept link the optimizer removes becomes ``optimizer_removed``.
    7. Per tile, the translation of its resolved affine (µm, fractional); an
       absent tile gets NaN. With no link left, or no overlap at all, the
       library raises ``NotEnoughOverlapError``: every translation is NaN, so
       ``stitch`` makes every FOV ``failed``, and the error is recorded in the
       diagnostics.

    The diagnostics hold the outcome, the registration dimensions, the
    library's overlap pairs (and, as table rows, any outside the neighbour
    pairs, which are dropped), the pruned and the optimizer-removed pairs, the
    edge residuals (µm) and, unless a number was given, ``link_quality``
    (its ``rule``: ``"auto"`` or ``"fixed_default_z_above_1"``).
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
    rule, threshold = _resolved_threshold(config.quality_threshold, plane)
    if rule == FIXED_DEFAULT_RULE:
        diagnostics["link_quality"] = {"rule": rule, "threshold": threshold}
    auto = rule == AUTO
    nulls = {} if auto else None
    # The z check of "auto" on tiles with Z > 1; every tile has one z spacing (stitch checks it).
    spacing_z = float(grids[present[0]].metadata.spacing_zyx[0])
    d = _z_offset_planes(spacing_z, config.binning_zyx[0]) if auto and not plane else None
    zs = {} if d is not None else None
    computed = _pairwise_registrations(msims, binned, pruned_graph, config, nulls, zs, d)
    if auto:
        threshold, diagnostics["link_quality"] = _auto_threshold(nulls, registered, fov_pairs)

    quality, shift, reasons = {}, {}, {}
    for e in registered:
        quality[e] = _scalar(computed.edges[e]["quality"])
        full = np.zeros(3)
        full[list(axes)] = -_translation(computed.edges[e]["transform"], ndim)
        shift[e] = full
    outcomes = {}
    if d is not None:
        outcomes, diagnostics["link_quality"]["z_check"] = _z_check(d, spacing_z * config.binning_zyx[0], zs, quality,
                                                                    registered, fov_pairs)
    for e in registered:
        if math.isnan(quality[e]):
            reasons[e] = "nan_quality" if config.drop_nan_quality else ""
        elif quality[e] < threshold:
            reasons[e] = "quality_below_threshold"
        else:
            reasons[e] = Z_DROP_REASON if outcomes.get(e) == Z_DROP_REASON else ""
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
