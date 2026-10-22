"""The stitch entry and its input (docs/stitching-contract.md, "The stitch entry", "Input image")."""
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from starfinder._registry import provenance, require, spec_for
from starfinder.image import IncompatibleGeometryError
from starfinder._checkpoint import _json_diagnostics
from starfinder.segmentation._labels import ReferenceGrid

from ._errors import StitchingBackendUnavailableError
from ._methods import METHOD_LINK_COLUMNS, STITCHING_METHODS, WHAT, MethodPlacement
from ._neighbors import neighbor_pairs
from ._placement import (AXES, FORMAT_VERSION, FOV_COLUMNS, LINK_COLUMNS, LINK_REASONS, STAGE, PlacementResult,
                         check_grids, empty_links, frame_record, input_record)
from ._positions import TilePositionTable, _names
from ._review import ReviewPolicy, review_placement

_SOURCE = re.compile(r"reference_merged|reference_stain|file:.+")
_SHA256 = re.compile(r"[0-9a-f]{64}")
FALLBACKS = (None, "nominal")


@dataclass(frozen=True, eq=False)
class StitchingInput:
    """One FOV's stitching image on its reference grid.

    ``image`` is a ZYX array with ``image.shape == grid.shape_zyx`` (a YX array is
    read as one plane, Z=1), or ``None`` for a FOV whose image is absent (status
    ``missing``). ``source`` is ``"reference_merged"``, ``"reference_stain"`` or
    ``"file:<path>"``; ``file_sha256`` the SHA-256 of the file it was read from,
    when there is one.

    Raises
    ------
    TypeError
        grid is not a ReferenceGrid, or the image is not a real numeric array.
    ValueError
        An empty fov_id, another source form, or a malformed SHA-256.
    IncompatibleGeometryError
        The image is not ZYX (or YX) of the grid's shape.
    """

    fov_id: str
    image: np.ndarray | None
    grid: ReferenceGrid
    source: str
    file_sha256: str | None = None

    def __post_init__(self):
        if not isinstance(self.fov_id, str) or not self.fov_id:
            raise ValueError("fov_id must be a nonempty string")
        if not isinstance(self.grid, ReferenceGrid):
            raise TypeError("grid must be a ReferenceGrid")
        if not isinstance(self.source, str) or not _SOURCE.fullmatch(self.source):
            raise ValueError(f"source must be 'reference_merged', 'reference_stain' or 'file:<path>'; "
                             f"got {self.source!r}")
        if self.file_sha256 is not None and (not isinstance(self.file_sha256, str)
                                             or not _SHA256.fullmatch(self.file_sha256)):
            raise ValueError("file_sha256 must be None or 64 lowercase hexadecimal digits")
        if self.image is None:
            return
        image = np.asarray(self.image)
        if image.dtype.kind not in "uif":
            raise TypeError(f"the image of FOV {self.fov_id!r} must be a real numeric array; got {image.dtype}")
        if image.ndim == 2:
            image = image[None]
        if image.shape != self.grid.shape_zyx:
            raise IncompatibleGeometryError(f"the image of FOV {self.fov_id!r} has shape {image.shape}; its grid "
                                            f"{self.grid.shape_zyx}")
        object.__setattr__(self, "image", image)


def _check_inputs(inputs, table):
    inputs = list(inputs)
    wrong = [i for i, item in enumerate(inputs) if not isinstance(item, StitchingInput)]
    if wrong:
        raise TypeError(f"inputs must be StitchingInput; items {_names(wrong)} are not")
    given, ids = [item.fov_id for item in inputs], table.fov_ids
    if given != ids:
        missing = [f for f in ids if f not in given]
        extra = [f for f in given if f not in ids]
        detail = (f"missing {_names(missing)}" if missing else "") + ("; " if missing and extra else "") \
            + (f"not in the table {_names(extra)}" if extra else "")
        raise ValueError("inputs must match the tile position table, one per row in table order; "
                         + (detail or f"got the order {_names(given)}"))
    return inputs


def _check_shapes(spec, grids):
    z = next(iter(grids.values())).shape_zyx[0]
    dimension = 2 if z == 1 else 3
    if dimension not in spec.dimensions:
        kind = "Z=1 tiles (placed as planes)" if dimension == 2 else "Z>1 tiles (placed as volumes)"
        raise IncompatibleGeometryError(f"{WHAT} {spec.name!r} does not accept {kind}")
    low = spec.min_shape_zyx if dimension == 3 else (1, *spec.min_shape_zyx[1:])
    small = [f"{fov} {grid.shape_zyx}" for fov, grid in grids.items()
             if any(n < m for n, m in zip(grid.shape_zyx, low))]
    if small:
        raise IncompatibleGeometryError(f"{WHAT} {spec.name!r} needs tiles of at least {low}; "
                                        f"FOVs {_names(small)} are smaller")


def _method_links(links, pairs):
    """The method's links keyed by pair, each oriented ``fov_a`` before ``fov_b`` as the pair is."""
    if not isinstance(links, pd.DataFrame):
        raise TypeError(f"a {WHAT} must return its links as a DataFrame")
    missing = [c for c in METHOD_LINK_COLUMNS if c not in links.columns]
    if missing:
        raise ValueError(f"the {WHAT}'s links lack the columns {missing}")
    order = {(p.fov_a, p.fov_b): p for p in pairs}
    found = {}
    for row in links.itertuples(index=False):
        key, sign = (row.fov_a, row.fov_b), 1.0
        if key not in order and (row.fov_b, row.fov_a) in order:
            key, sign = (row.fov_b, row.fov_a), -1.0
        if key not in order:
            raise ValueError(f"the {WHAT} registered {row.fov_a!r} and {row.fov_b!r}, which are not a neighbour pair")
        if key in found:
            raise ValueError(f"the {WHAT} returned the pair {key[0]!r}, {key[1]!r} more than once")
        kept = row.kept
        if not isinstance(kept, (bool, np.bool_)):
            raise ValueError(f"kept of the link {key[0]!r}, {key[1]!r} must be Boolean")
        reason = "" if kept else row.reason
        if not kept and reason not in LINK_REASONS[1:]:
            raise ValueError(f"the unkept link {key[0]!r}, {key[1]!r} needs a reason among {list(LINK_REASONS[1:])}; "
                             f"got {row.reason!r}")
        shift = tuple(sign * float(getattr(row, f"shift_{a}")) for a in AXES)
        found[key] = (shift, float(row.quality), bool(kept), reason)
    return found


def _components(n, present, placeable, edges):
    """Component per present FOV, numbered by first FOV in table order; −1 only for an absent image.

    Kept links join two FOVs only when both are placeable (present with a finite
    translation), so a present FOV the method did not place is its own component.
    """
    parent = list(range(n))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in edges:
        if placeable[a] and placeable[b]:
            ra, rb = root(a), root(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    labels, component = {}, np.full(n, -1, dtype=np.int64)
    for i in range(n):
        if present[i]:
            component[i] = labels.setdefault(root(i), len(labels))
    return component


def stitch(inputs: Sequence[StitchingInput], table: TilePositionTable, *, config,
           review: ReviewPolicy = ReviewPolicy(), fallback: str | None = None,
           sample: str = "sample") -> PlacementResult:
    """Place a sample's FOVs with a registered method; return the placement record.

    In order: (1) ``config`` is registered in ``STITCHING_METHODS`` by its exact
    type; (2) ``review`` and ``fallback`` are valid; (3) the inputs match the
    table, one per row in table order; (4) the grids share the sample-frame
    geometry (known spacing, unit µm, identity direction, one Z size and
    spacing; Y and X sizes may differ) and fit the method's ``dimensions`` and
    ``min_shape_zyx``; (5) the method's optional dependencies import; then the
    neighbour pairs are derived (with the config's ``overlap_tolerance_um`` when
    it has one), the method runs once, and:

    * the links table has one row per neighbour pair: the method's shift,
      quality, kept flag and reason, or ``pruned`` for a pair it did not
      register; a kept link's residual is ``‖(origin_b − origin_a) − (nominal_b
      − nominal_a) − shift‖``;
    * the components are those of the kept links between FOVs the method placed
      (present, finite translation); every present FOV has one (a FOV with a
      non-finite translation is its own), and ``component`` is −1 only for
      ``missing``; the reference component is chosen among the components of
      placed FOVs: the most FOVs, a tie going to the one with the first FOV in
      table order;
    * each FOV gets one status: ``missing`` (no image), ``image_supported``
      (joined to the reference component by a kept link), ``failed`` for a
      non-finite translation whatever the ``fallback`` (its
      ``method_translation_*`` stays as the method returned it), else
      ``failed`` with ``fallback=None`` or ``metadata_only`` at ``origin =
      nominal`` flagged ``fallback`` with ``fallback="nominal"``;
    * gauge G1: ``origin = nominal + c − mean(c)`` with ``c`` the method's
      translation (µm) and the mean over the ``image_supported`` FOVs;
    * :func:`review_placement` applies ``review``.

    It never reads or writes a file and never raises for a placement that fails
    or needs review: those are statuses, flags and the review state.

    Parameters
    ----------
    inputs : Sequence[StitchingInput]
        One per row of ``table``, in table order.
    table : TilePositionTable
        The nominal positions.
    config
        A frozen config registered in ``STITCHING_METHODS``.
    review : ReviewPolicy
        The review criterion (default ``ReviewPolicy()``).
    fallback : str or None
        ``None`` (default) or ``"nominal"`` (opt-in, flagged).
    sample : str
        The sample; the frame is ``sample:<sample>`` (``Dataset.stitch`` passes
        its own).

    Returns
    -------
    PlacementResult
        The record holds ``format_version`` 1, ``frame``, ``gauge`` ``G1`` with
        the mean removed, ``method`` (the uniform provenance entry, ``stage``
        ``stitching``), ``method_diagnostics``, ``table``, ``inputs``,
        ``fallback`` and ``review``.

    Raises
    ------
    TypeError
        An unregistered config type, inputs that are not StitchingInput, a table
        that is not a TilePositionTable or a review that is not a ReviewPolicy.
    ValueError
        An unknown fallback, an empty sample, inputs that do not match the table,
        or a method result of another form.
    IncompatibleGeometryError
        Grids outside the sample-frame geometry, or tiles the method does not
        accept.
    StitchingBackendUnavailableError
        A dependency of the method is not installed.
    """
    spec = spec_for(STITCHING_METHODS, config, WHAT, TypeError)
    if not isinstance(review, ReviewPolicy):
        raise TypeError("review must be a ReviewPolicy")
    if fallback not in FALLBACKS:
        raise ValueError(f"fallback must be None or 'nominal'; got {fallback!r}")
    if not isinstance(table, TilePositionTable):
        raise TypeError("table must be a TilePositionTable")
    if not isinstance(sample, str) or not sample:
        raise ValueError("sample must be a nonempty string")
    if not table.fov_ids:
        raise ValueError("the tile position table has no row")
    inputs = _check_inputs(inputs, table)
    grids = {item.fov_id: item.grid for item in inputs}
    check_grids(grids)
    _check_shapes(spec, grids)
    require(spec, WHAT, StitchingBackendUnavailableError)

    ids, n = table.fov_ids, len(inputs)
    pairs = neighbor_pairs(table, grids, tolerance_um=getattr(config, "overlap_tolerance_um", 0.0))
    nominal = table.nominal_zyx()
    output = spec.run([item.image for item in inputs], [item.grid for item in inputs], nominal.copy(), pairs,
                      config=config)
    if not isinstance(output, MethodPlacement):
        raise ValueError(f"{WHAT} {spec.name!r} must return a MethodPlacement")
    translation = np.asarray(output.translations_zyx, dtype=np.float64)
    if translation.shape != (n, 3):
        raise ValueError(f"{WHAT} {spec.name!r} returned translations of shape {translation.shape}; expected {(n, 3)}")
    method_components = np.asarray(output.components)
    if method_components.shape != (n,):
        raise ValueError(f"{WHAT} {spec.name!r} returned components of shape {method_components.shape}; "
                         f"expected {(n,)}")
    registered = _method_links(output.links, pairs)

    index = {fov: i for i, fov in enumerate(ids)}
    present = np.array([item.image is not None for item in inputs])
    finite = np.isfinite(translation).all(axis=1)
    placeable = present & finite
    kept_edges = [(index[a], index[b]) for (a, b), value in registered.items() if value[2]]
    component = _components(n, present, placeable, kept_edges)
    joined = np.zeros(n, dtype=bool)
    for a, b in kept_edges:
        if placeable[a] and placeable[b]:
            joined[a] = joined[b] = True
    # A component holding a placeable FOV holds only placeable FOVs. Candidates are numbered by
    # their first FOV and sorted, so argmax breaks a tie by table order.
    candidates = np.unique(component[placeable])
    sizes = np.bincount(component[present])
    reference = int(candidates[np.argmax(sizes[candidates])]) if candidates.size else -2

    status = np.empty(n, dtype=object)
    flags = [[] for _ in range(n)]
    for i in range(n):
        if not present[i]:
            status[i] = "missing"
        elif component[i] == reference and joined[i]:
            status[i] = "image_supported"
        elif not finite[i]:
            status[i] = "failed"   # the method gave no position, so the fallback does not apply
        elif fallback == "nominal":
            status[i], flags[i] = "metadata_only", ["fallback"]
        else:
            status[i] = "failed"
    supported = status == "image_supported"
    origin = np.full((n, 3), np.nan)
    mean = translation[supported].mean(axis=0) if supported.any() else None
    if mean is not None:
        origin[supported] = nominal[supported] + translation[supported] - mean
    fell_back = status == "metadata_only"
    origin[fell_back] = nominal[fell_back]
    correction = origin - nominal

    rows = []
    for pair in pairs:
        (z0, z1), (y0, y1), (x0, x1) = pair.overlap_zyx
        shift, quality, kept, reason = registered.get((pair.fov_a, pair.fov_b),
                                                      ((np.nan,) * 3, np.nan, False, "pruned"))
        a, b = index[pair.fov_a], index[pair.fov_b]
        residual = np.nan
        if kept and np.isfinite(origin[[a, b]]).all():
            difference = (origin[b] - origin[a]) - (nominal[b] - nominal[a]) - np.asarray(shift)
            residual = float(np.sqrt(np.sum(difference * difference)))
        rows.append((pair.fov_a, pair.fov_b, z0, z1, y0, y1, x0, x1, *shift, quality, kept, reason, residual))
    links = pd.DataFrame(rows, columns=list(LINK_COLUMNS)) if rows else empty_links()
    n_kept = np.zeros(n, dtype=np.int64)
    for a, b in kept_edges:
        n_kept[a] += 1
        n_kept[b] += 1

    fovs = pd.DataFrame({"fov_id": ids, "status": list(status),
                         **{f"origin_{a}": origin[:, i] for i, a in enumerate(AXES)},
                         **{f"nominal_{a}": nominal[:, i] for i, a in enumerate(AXES)},
                         **{f"correction_{a}": correction[:, i] for i, a in enumerate(AXES)},
                         **{f"method_translation_{a}": translation[:, i] for i, a in enumerate(AXES)},
                         "component": component, "n_links_kept": n_kept,
                         "flags": [";".join(f) for f in flags]}, columns=list(FOV_COLUMNS))
    diagnostics = _json_diagnostics(dict(output.diagnostics)) if isinstance(output.diagnostics, dict) else {}
    record = {"format_version": FORMAT_VERSION, "stage": STAGE, "sample": sample, "frame": frame_record(sample),
              "gauge": "G1", "gauge_mean_correction_zyx": None if mean is None else [float(v) for v in mean],
              "method": provenance(spec, config, STAGE),
              "method_diagnostics": {"components": [int(c) for c in method_components], **diagnostics},
              "table": table.record,
              "inputs": {item.fov_id: input_record(item.source, item.file_sha256, item.grid) for item in inputs},
              "fallback": fallback, "review": {"approvals": []}}
    return review_placement(PlacementResult(sample, fovs, links, record), review)
