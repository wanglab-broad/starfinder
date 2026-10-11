"""The placement record, the index-to-sample rule and the supplied placement.

docs/stitching-contract.md ("Placement record", "Supplied placement") and
docs/stitching-algorithms.md ("Supplied placement").
"""
from __future__ import annotations

import hashlib
import io
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from numbers import Real

import numpy as np
import pandas as pd

from starfinder.image import ImageMetadata, IncompatibleGeometryError
from starfinder.segmentation._labels import ReferenceGrid, _grid_record

from ._positions import TilePositionTable, _names

FORMAT_VERSION = 1
STAGE = "stitching"
STATUSES = ("image_supported", "metadata_only", "placeholder", "missing", "failed")
PLACED = ("image_supported", "metadata_only", "placeholder")
FLAGS = ("fallback", "supplied", "correction_above_bound", "residual_above_tolerance", "nan_quality_link",
         "optimizer_removed_link")
LINK_REASONS = ("", "quality_below_threshold", "nan_quality", "optimizer_removed", "pruned")
AXES = ("z", "y", "x")
FOV_COLUMNS = ("fov_id", "status", *(f"origin_{a}" for a in AXES), *(f"nominal_{a}" for a in AXES),
               *(f"correction_{a}" for a in AXES), *(f"method_translation_{a}" for a in AXES),
               "component", "n_links_kept", "flags")
FOV_DTYPES = {c: "float64" for c in FOV_COLUMNS} | {"fov_id": "str", "status": "str", "flags": "str",
                                                   "component": "int64", "n_links_kept": "int64"}
LINK_COLUMNS = ("fov_a", "fov_b", *(f"overlap_{a}{e}" for a in AXES for e in "01"), *(f"shift_{a}" for a in AXES),
                "quality", "kept", "reason", "residual")
LINK_DTYPES = {c: "float64" for c in LINK_COLUMNS} | {"fov_a": "str", "fov_b": "str", "kept": "bool",
                                                     "reason": "str"}
IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
#: The spellings of the micrometre a grid may carry; §2.10 reads them as one unit.
MICROMETRE = ("µm", "um", "micrometer")
SUPPLIED_IMPLEMENTATION = "starfinder.stitching.placement_from_table"


def _json(record):
    from starfinder.segmentation._persist import json_record
    return json_record(record)


def frame_record(sample):
    """The record's ``frame``: the sample frame, identity direction, µm."""
    return {"frame_id": f"sample:{sample}", "unit": "µm", "direction": "identity"}


def input_record(source, file_sha256, grid):
    """One FOV's ``record["inputs"]`` entry: input source, file SHA-256 and the grid (shape, metadata, SHA-256)."""
    return {"source": source, "file_sha256": file_sha256, "grid": _grid_record(grid)}


def empty_links() -> pd.DataFrame:
    """A link table with the columns of "Links" and no row."""
    return pd.DataFrame({c: pd.Series([], dtype=d) for c, d in LINK_DTYPES.items()}, columns=list(LINK_COLUMNS))


def table_csv(frame) -> bytes:
    """The CSV bytes ``save_placement`` writes for a table: the checkpoint CSV form, float64 as ``%.17g``.

    Strings are escaped as the checkpoint tables escape them, so every value
    reads back exactly; the SHA-256 of these bytes is the table's SHA-256 in
    ``placement.json`` and in an approval.
    """
    from starfinder.io._checkpoint import _CONTROL, NA_TOKEN, _dtype_map, _escape_strings
    text = frame.copy()
    for column, dtype in _dtype_map(frame).items():
        if dtype in ("string", "str"):
            strings = frame[column].astype("string")
            if strings.str.contains(_CONTROL, regex=True, na=False).any():
                raise ValueError(f"placement column {column!r} contains a control character")
            text[column] = _escape_strings(strings)
    buffer = io.StringIO()
    text.to_csv(buffer, index=False, float_format="%.17g", na_rep=NA_TOKEN, lineterminator="\n")
    return buffer.getvalue().encode("utf-8")


def table_sha256(frame) -> str:
    """SHA-256 of :func:`table_csv`."""
    return hashlib.sha256(table_csv(frame)).hexdigest()


@dataclass(frozen=True, eq=False)
class PlacementResult:
    """One placement of a sample's FOVs, whichever backend made it.

    ``fovs`` has one row per row of the tile position table, in the same order,
    with the columns ``fov_id``, ``status`` (``image_supported``,
    ``metadata_only``, ``placeholder``, ``missing`` or ``failed``),
    ``origin_z/y/x`` (float64, µm: the sample-frame position of index
    ``(0, 0, 0)`` of the FOV's reference grid; finite exactly for the three placed
    statuses), ``nominal_z/y/x``, ``correction_z/y/x`` (``origin − nominal``),
    ``method_translation_z/y/x`` (the method's own result before the gauge),
    ``component`` (−1 for ``missing``), ``n_links_kept`` and ``flags``
    (``;``-separated). ``links`` has one row per neighbour pair: ``fov_a``,
    ``fov_b``, the nominal overlap box ``overlap_z0`` … ``overlap_x1`` (µm),
    ``shift_z/y/x`` (µm, of ``fov_b`` against ``fov_a``), ``quality``, ``kept``,
    ``reason`` (empty for a kept link) and ``residual`` (µm, NaN for an unkept
    link). ``record`` holds ``format_version``, ``stage``, ``sample``,
    ``frame``, ``gauge``, ``method`` (the uniform provenance entry), ``table``
    (the table's record), ``inputs`` (per FOV: source, file SHA-256 and grid) and
    ``review``, as JSON values. See docs/stitching-contract.md ("Placement
    record").

    Raises
    ------
    TypeError
        A table that is not a DataFrame or a record that is not a mapping.
    ValueError
        An empty sample, other columns, a repeated ``fov_id``, an unknown status,
        flag or link reason, an origin that is finite for ``missing`` or
        ``failed`` or not finite for a placed status, or a link naming a FOV the
        table lacks.
    """

    sample: str
    fovs: pd.DataFrame
    links: pd.DataFrame
    record: dict

    def __post_init__(self):
        if not isinstance(self.sample, str) or not self.sample:
            raise ValueError("sample must be a nonempty string")
        for name, columns, dtypes in (("fovs", FOV_COLUMNS, FOV_DTYPES), ("links", LINK_COLUMNS, LINK_DTYPES)):
            frame = getattr(self, name)
            if not isinstance(frame, pd.DataFrame):
                raise TypeError(f"{name} must be a pandas DataFrame")
            if tuple(frame.columns) != columns:
                raise ValueError(f"{name} must have the columns {list(columns)}; got {list(frame.columns)}")
            object.__setattr__(self, name, frame.astype(dtypes).reset_index(drop=True))
        if not isinstance(self.record, Mapping):
            raise TypeError("record must be a mapping")
        fovs, links = self.fovs, self.links
        ids = list(fovs["fov_id"])
        repeated = sorted(k for k, n in Counter(ids).items() if n > 1)
        if repeated:
            raise ValueError(f"fov_id must be unique; repeated: {_names(repeated)}")
        unknown = sorted(set(fovs["status"]) - set(STATUSES))
        if unknown:
            raise ValueError(f"status must be one of {list(STATUSES)}; got {_names(unknown)}")
        flags = {f for value in fovs["flags"] for f in value.split(";") if f}
        if flags - set(FLAGS):
            raise ValueError(f"flags must be among {list(FLAGS)}; got {_names(sorted(flags - set(FLAGS)))}")
        finite = np.isfinite(fovs[[f"origin_{a}" for a in AXES]].to_numpy()).all(axis=1)
        placed = fovs["status"].isin(PLACED).to_numpy()
        wrong = [fov for fov, f, p in zip(ids, finite, placed) if f != p]
        if wrong:
            raise ValueError(f"origins must be finite exactly for {', '.join(PLACED)}; FOVs {_names(wrong)} break it")
        outside = sorted((set(links["fov_a"]) | set(links["fov_b"])) - set(ids))
        if outside:
            raise ValueError(f"links name FOVs the placement lacks: {_names(outside)}")
        reasons = sorted(set(links["reason"]) - set(LINK_REASONS))
        if reasons:
            raise ValueError(f"link reason must be one of {list(LINK_REASONS)}; got {_names(reasons)}")
        object.__setattr__(self, "record", _json(dict(self.record)))


def frame_metadata(metadata: ImageMetadata) -> ImageMetadata:
    """The metadata as §2.10 checks a grid: a spelling of :data:`MICROMETRE` reads as ``µm`` and no direction as
    the identity. Nothing written changes; only the checks compare this form."""
    unit = "µm" if metadata.spatial_unit in MICROMETRE else metadata.spatial_unit
    direction = IDENTITY if metadata.direction_zyx is None else metadata.direction_zyx
    if (unit, direction) == (metadata.spatial_unit, metadata.direction_zyx):
        return metadata
    return replace(metadata, spatial_unit=unit, direction_zyx=direction)


def same_frame_grid(shape_a, metadata_a, shape_b, metadata_b) -> bool:
    """Whether two grids are one grid for §2.10: equal shape and equal :func:`frame_metadata`."""
    return tuple(shape_a) == tuple(shape_b) and frame_metadata(metadata_a) == frame_metadata(metadata_b)


def grid_problems(metadata) -> list[str]:
    """Which of ``unknown spacing``, ``a unit other than µm`` and ``a direction other than the identity`` a grid
    has, by :func:`frame_metadata`."""
    read = frame_metadata(metadata)
    return [what for what, bad in (("unknown spacing", read.spacing_zyx is None),
                                   ("a unit other than µm", read.spatial_unit != "µm"),
                                   ("a direction other than the identity", read.direction_zyx != IDENTITY)) if bad]


def check_grids(grids: Mapping[str, ReferenceGrid]) -> None:
    """The sample-frame geometry rule shared by ``stitch`` and ``placement_from_table``.

    Every grid has known spacing, unit ``µm`` (spelled ``µm``, ``um`` or
    ``micrometer``) and identity direction (or none, which is the identity),
    and all share Z size and spacing; Y and X sizes may differ.
    ``IncompatibleGeometryError`` names the FOVs and their values.
    """
    problems = {"unknown spacing": [], "a unit other than µm": [], "a direction other than the identity": []}
    for fov, grid in grids.items():
        metadata = grid.metadata
        shown = {"a unit other than µm": metadata.spatial_unit,
                 "a direction other than the identity": metadata.direction_zyx}
        for what in grid_problems(metadata):
            problems[what].append(f"{fov} ({shown[what]!r})" if what in shown else fov)
    found = [f"{what}: {_names(fovs)}" for what, fovs in problems.items() if fovs]
    if found:
        raise IncompatibleGeometryError("the sample frame needs known spacing, unit µm and identity direction "
                                        f"on every reference grid; FOVs with {'; '.join(found)}")
    for what, value in (("Z size", lambda g: g.shape_zyx[0]), ("spacing", lambda g: g.metadata.spacing_zyx)):
        groups = {}
        for fov, grid in grids.items():
            groups.setdefault(value(grid), []).append(fov)
        if len(groups) > 1:
            described = "; ".join(f"{_names(fovs)}: {key!r}" for key, fovs in groups.items())
            raise IncompatibleGeometryError(f"the FOVs of a sample must share one {what}; {described}")


def placed_metadata(result: PlacementResult, fov_id: str) -> ImageMetadata:
    """The FOV's reference-grid metadata in the sample frame.

    ``dataclasses.replace(grid.metadata, frame_id="sample:<sample>",
    origin_zyx=origin)``, with the grid recorded in ``record["inputs"]`` and the
    FOV's placement origin, read as §2.10 checks a grid (a spelling of the
    micrometre as ``µm``, no direction as the identity), so that ``index_to_world(p)`` is ``origin + D · (p ⊙ spacing)`` for an index
    position ``p`` (integer or fractional) of the FOV's reference grid, also for
    a grid recorded without a direction. The record is not changed.

    Raises
    ------
    ValueError
        The FOV is not a row of the placement, or it has no origin (``missing``
        or ``failed``).
    """
    rows = result.fovs.index[result.fovs["fov_id"] == fov_id]
    if len(rows) != 1:
        raise ValueError(f"FOV {fov_id!r} is not a row of the placement of sample {result.sample!r}")
    row = result.fovs.loc[rows[0]]
    origin = tuple(float(row[f"origin_{a}"]) for a in AXES)
    if not all(math.isfinite(v) for v in origin):
        raise ValueError(f"FOV {fov_id!r} has no origin (status {row['status']!r})")
    metadata = ImageMetadata(**result.record["inputs"][fov_id]["grid"]["metadata"])
    return frame_metadata(replace(metadata, frame_id=f"sample:{result.sample}", origin_zyx=origin))


def _origin_triple(value):
    """The three float64 values of a supplied origin, or None when it is not three finite real numbers."""
    if isinstance(value, (str, bytes)) or not isinstance(value, (Sequence, np.ndarray)):
        return None
    values = list(np.asarray(value, dtype=object).ravel()) if isinstance(value, np.ndarray) else list(value)
    if len(values) != 3 or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Real) for v in values):
        return None
    values = tuple(float(v) for v in values)
    return values if all(math.isfinite(v) for v in values) else None


def _key_sets(ids, keys, what):
    missing = [fov for fov in ids if fov not in keys]
    if missing:
        raise ValueError(f"FOVs of the table without {what}: {_names(missing)}")
    extra = [key for key in keys if key not in set(ids)]
    if extra:
        raise ValueError(f"{what} given for FOVs the table lacks: {_names(extra)}")


def placement_from_table(table: TilePositionTable, grids: Mapping[str, ReferenceGrid], *, sample: str,
                         origins: Mapping[str, tuple[float, float, float]] | None = None) -> PlacementResult:
    """A placement that is known rather than measured, as a :class:`PlacementResult`.

    Every FOV of ``table`` gets one row, in table order: status
    ``metadata_only``, flag ``supplied``, ``nominal_*`` copied from the table,
    ``method_translation_*`` NaN (no method ran), ``n_links_kept`` 0 and
    ``component`` its row index. Without ``origins`` each origin is its nominal
    position, copied, and each correction 0.0; with ``origins`` each origin is
    the supplied value as float64, never rounded, cast or shifted, and each
    correction ``origin − nominal`` (one subtraction per axis). ``links`` has
    no row. The record's method is ``supplied``
    (``starfinder.stitching.placement_from_table``, empty config type, config,
    requires and artifacts), ``origins`` is ``nominal`` or ``supplied``, the
    gauge ``none``, and each input ``supplied`` with a null file SHA-256 and the
    grid. Review is ``required`` with the reason ``supplied_placement``, whatever
    the positions: the caller approves it with :func:`approve_placement`. It
    reads and writes no file and calls no method.

    Parameters
    ----------
    table : TilePositionTable
        At least one row; its nominal positions are the FOVs' nominal positions.
    grids : Mapping[str, ReferenceGrid]
        One reference grid per ``fov_id`` of the table, and no other key.
    sample : str
        The sample; its frame is ``sample:<sample>``.
    origins : Mapping[str, tuple[float, float, float]] or None
        None (default), or one ``(z, y, x)`` of finite numbers (µm, sample frame)
        per ``fov_id`` of the table, and no other key.

    Returns
    -------
    PlacementResult

    Raises
    ------
    TypeError
        table is not a TilePositionTable, grids or origins not a mapping, or a
        grid not a ReferenceGrid.
    ValueError
        An empty sample or table; a table FOV without a grid, a grid key outside
        the table; with origins, a table FOV without an origin, an origin key
        outside the table, or an origin that is not three finite numbers (each
        naming the FOVs; ``origins={}`` names every FOV).
    IncompatibleGeometryError
        A grid with unknown spacing, a unit other than µm or a non-identity
        direction, or grids that differ in Z size, spacing, direction or unit.
    """
    if not isinstance(table, TilePositionTable):
        raise TypeError("table must be a TilePositionTable")
    if not isinstance(sample, str) or not sample:
        raise ValueError("sample must be a nonempty string")
    ids = table.fov_ids
    if not ids:
        raise ValueError("the tile position table has no row")
    if not isinstance(grids, Mapping):
        raise TypeError("grids must map each fov_id of the table to its ReferenceGrid")
    _key_sets(ids, grids, "a grid")
    wrong = [fov for fov in ids if not isinstance(grids[fov], ReferenceGrid)]
    if wrong:
        raise TypeError(f"grids must be ReferenceGrid; FOVs {_names(wrong)} are not")
    supplied = None
    if origins is not None:
        if not isinstance(origins, Mapping):
            raise TypeError("origins must map each fov_id of the table to a (z, y, x) origin, or be None")
        _key_sets(ids, origins, "an origin")
        supplied = {fov: _origin_triple(origins[fov]) for fov in ids}
        bad = [fov for fov, value in supplied.items() if value is None]
        if bad:
            raise ValueError(f"an origin must be three finite numbers (z, y, x) in µm; FOVs {_names(bad)} are not")
    check_grids({fov: grids[fov] for fov in ids})

    n = len(ids)
    nominal = table.nominal_zyx()
    if supplied is None:
        origin = nominal.copy()
        correction = np.zeros((n, 3), dtype=np.float64)
    else:
        origin = np.array([supplied[fov] for fov in ids], dtype=np.float64)
        correction = origin - nominal
    fovs = pd.DataFrame({"fov_id": ids, "status": ["metadata_only"] * n,
                         **{f"origin_{a}": origin[:, i] for i, a in enumerate(AXES)},
                         **{f"nominal_{a}": nominal[:, i] for i, a in enumerate(AXES)},
                         **{f"correction_{a}": correction[:, i] for i, a in enumerate(AXES)},
                         **{f"method_translation_{a}": np.full(n, np.nan) for a in AXES},
                         "component": np.arange(n, dtype=np.int64), "n_links_kept": np.zeros(n, dtype=np.int64),
                         "flags": ["supplied"] * n}, columns=list(FOV_COLUMNS))
    record = {"format_version": FORMAT_VERSION, "stage": STAGE, "sample": sample, "frame": frame_record(sample),
              "gauge": "none", "origins": "nominal" if supplied is None else "supplied",
              "method": {"stage": STAGE, "method": "supplied", "config_type": "",
                         "implementation": SUPPLIED_IMPLEMENTATION, "config": {}, "requires": {}, "artifacts": []},
              "table": table.record,
              "inputs": {fov: input_record("supplied", None, grids[fov]) for fov in ids},
              "fallback": None,
              "review": {"state": "required", "reasons": ["supplied_placement"], "policy": None, "flagged": {},
                         "approvals": []}}
    return PlacementResult(sample, fovs, empty_links(), record)
