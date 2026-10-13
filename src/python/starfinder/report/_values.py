"""The values the report computes from its records (docs/report-algorithms.md, "Values of the tables").

A value is a number or string, :data:`NA` (it does not exist in its case) or an
:class:`Unavailable` (null, named by one row of ``unavailable.csv``). Each definition here is a
function of record fields or of cells of the report's tables, so that the tests compute the
same definitions on hand-written tables; :mod:`._collect` applies them to a sample's records.
"""
from __future__ import annotations

import json
import math
from datetime import datetime
from typing import NamedTuple

import numpy as np

from ._schema import UNAVAILABLE_REASONS


class Unavailable(NamedTuple):
    """A null value named by one row of ``unavailable.csv``."""

    reason: str
    detail: str | None = None


class _Marker:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name


#: A null value that does not exist in its case; no row names it.
NA = _Marker("NA")

#: The seed of the draw of the main example ("The example FOVs"); part of the algorithm.
EXAMPLE_SEED = 0
#: The factor of the robust spread: s = 1.4826 × MAD.
SPREAD_FACTOR = 1.4826
AXES = ("z", "y", "x")
_GLOBAL_KINDS = ("translation", "affine")


def _rank(reason):
    return UNAVAILABLE_REASONS.index(reason)


def first(*values):
    """The unavailable value whose reason comes first in the closed list, or None when none is unavailable (P1)."""
    found = [v for v in values if isinstance(v, Unavailable)]
    return min(found, key=lambda v: _rank(v.reason)) if found else None


def is_value(value) -> bool:
    """A cell that holds a value (not NA or Unavailable)."""
    return value is not None and not isinstance(value, (Unavailable, _Marker))


# --- Sample statistics ("Conventions used by every definition") ----------------------------------------------

def median(values) -> float:
    """The median of values in float64 (NumPy's ``median``); the caller passes at least one value."""
    return float(np.median(np.asarray(values, dtype=np.float64)))


def mad(values, centre=None) -> float:
    """``median(abs(x_i − median(x)))``, or about ``centre`` when given."""
    centre = median(values) if centre is None else centre
    return median([abs(float(v) - centre) for v in values])


def over_fovs(cells, statistic, empty="no FOV"):
    """A statistic over the FOVs whose cell holds a value; with none, the first reason of the cells, or
    ``undefined`` when no cell has one ("Which FOVs enter a sample statistic")."""
    values = [v for v in cells if is_value(v)]
    if values:
        return statistic(values)
    return first(*cells) or Unavailable("undefined", empty)


# --- fovs.csv -----------------------------------------------------------------------------------------------

def fraction(count, total, detail):
    """``count / total`` in float64; the first unavailable input, ``undefined`` (detail the denominator's name)
    when ``total`` is 0."""
    missing = first(count, total)
    if missing is not None:
        return missing
    if total == 0:
        return Unavailable("undefined", detail)
    return count / total


def run_seconds(started_at, ended_at):
    """``ended_at − started_at`` in seconds (ISO 8601, both read as UTC), float64."""
    missing = first(started_at, ended_at)
    if missing is not None:
        return missing
    return (datetime.fromisoformat(ended_at) - datetime.fromisoformat(started_at)).total_seconds()


def owned_pixels(ownership, fov, origin_zyx, shape_zyx, spacing_zyx, rows=256) -> int:
    """The pixels of a FOV's Y, X grid whose centre ``(o_z, o_y + i·s_y, o_x + j·s_x)`` the FOV owns under
    ``ownership.owner`` ("The area a FOV owns", OA1); tested ``rows`` grid rows at a time."""
    (oz, oy, ox), (ny, nx), (sy, sx) = origin_zyx, shape_zyx[1:], spacing_zyx[1:]
    count = 0
    x = ox + np.arange(nx, dtype=np.float64) * sx
    for y0 in range(0, ny, rows):
        y = oy + np.arange(y0, min(y0 + rows, ny), dtype=np.float64) * sy
        points = np.empty((len(y) * nx, 3), dtype=np.float64)
        points[:, 0], points[:, 1], points[:, 2] = oz, np.repeat(y, nx), np.tile(x, len(y))
        count += int((ownership.owner(points) == fov).sum())
    return count


def density(cells, area):
    """``cells / area × 10⁶`` (cells per mm² from µm²); NA as the area, ``undefined`` for an area of 0."""
    if area is NA:
        return NA
    missing = first(cells, area)
    if missing is not None:
        return missing
    if area == 0:
        return Unavailable("undefined", "the area")
    return cells / area * 1e6


# --- registration.csv ---------------------------------------------------------------------------------------

def chain_shift(entries, application_config=None):
    """The shift of a round ("Shift", S1) from its transform entries (the ``_transform_json`` form).

    The global steps (``translation``, ``affine``) are a prefix of the chain; they are rebuilt with
    ``io/_checkpoint.py``: ``_restore_result`` and composed as ``registration/_chain.py`` composes them,
    ``Φ_g(p) = T₁(T₂(…T_k(p)…))``. A chain of translations gives the sum of the displacements exactly; a
    chain with an affine step gives ``Φ_g(c) − c`` at the centre ``c`` of the reference grid. Returns a
    ``(z, y, x)`` tuple of floats, or :data:`NA` for a chain without a global step.
    """
    from starfinder.io._checkpoint import _registration_methods, _restore_result
    from starfinder.registration import TransformChain, WarpConfig
    from starfinder.registration._chain import _affine
    from starfinder.registration._types import TranslationTransform

    prefix = []
    for entry in entries:
        if (entry.get("transform") or {}).get("kind") not in _GLOBAL_KINDS:
            break
        prefix.append(entry)
    if not prefix:
        return NA

    def no_field(name):  # a global step reads no field file
        raise ValueError(f"a global step names the field {name!r}")

    from starfinder.io._checkpoint import _config
    application = _config(application_config, WarpConfig) if application_config else WarpConfig()
    methods = _registration_methods()
    transforms = [_restore_result(entry, i, no_field, 2, application, methods).transform
                  for i, entry in enumerate(prefix)]
    chain = TransformChain(tuple(transforms))
    translation = chain.translation()
    if translation is not None:
        return tuple(float(v) for v in translation.displacement_zyx)
    centre = np.array([(n - 1) / 2 for n in chain.reference_shape_zyx], dtype=np.float64)
    point = centre
    for transform in reversed(chain.transforms):
        if isinstance(transform, TranslationTransform):
            point = point + np.asarray(transform.displacement_zyx, dtype=np.float64)
        else:
            point = _affine(transform.matrix_zyx, point)
    return tuple(float(v) for v in point - centre)


def residuals(rows):
    """Fill ``residual_<axis>`` of registration rows from their ``shift_<axis>`` ("Shift residual").

    Per round and axis, over the rows whose ``outcome`` is ``succeeded`` and whose shift is a value:
    ``median_r,a`` and ``residual = shift − median_r,a``; a row without a shift value keeps the shift's state
    (its reason, or NA). When no row of the round succeeded, every residual of the round is ``undefined``
    (the column's rule "undefined when no FOV's round succeeded"). Returns ``{(round, axis): median_r,a}``
    for the rounds and axes with a value.
    """
    medians = {}
    for name in dict.fromkeys(row["round"] for row in rows):
        own = [row for row in rows if row["round"] == name]
        succeeded = [row for row in own if row.get("outcome") == "succeeded"]
        for axis in AXES:
            column = f"shift_{axis}"
            members = [row[column] for row in succeeded if is_value(row[column])]
            if members:
                medians[(name, axis)] = median(members)
            for row in own:
                shift = row[column]
                if not succeeded:
                    row[f"residual_{axis}"] = Unavailable("undefined", "no succeeded FOV")
                elif not is_value(shift):
                    row[f"residual_{axis}"] = shift
                elif row.get("outcome") != "succeeded":
                    row[f"residual_{axis}"] = Unavailable("step_failed", f"outcome {row.get('outcome')}")
                else:
                    row[f"residual_{axis}"] = shift - medians[(name, axis)]
    return medians


def alignment_score(entries, outcome):
    """``qc.values.ncc_after`` of the round's application entry ("Alignment score").

    ``step_failed`` when the round's outcome is not ``succeeded``; ``undefined`` when the QC or the value is
    null, with the QC's ``reasons.ncc_after`` in the detail. An estimation entry's score is not read.
    """
    if outcome != "succeeded":
        return Unavailable("step_failed", f"outcome {outcome}")
    application = [e for e in entries if e.get("record") == "application"]
    qc = application[-1].get("qc") if application else None
    value = ((qc or {}).get("values") or {}).get("ncc_after")
    if value is None:
        reason = ((qc or {}).get("reasons") or {}).get("ncc_after")
        return Unavailable("undefined", "application qc is null" if qc is None else
                           f"qc.values.ncc_after is null ({reason})" if reason else "qc.values.ncc_after is null")
    return float(value)


# --- The example FOVs -------------------------------------------------------------------------------------

class Example(NamedTuple):
    """One role of the example FOVs: the FOV and how it was chosen (``report.json`` ``examples``)."""

    role: str
    fov_id: str
    chosen_by: str
    rule: str | None = None
    value: int | None = None
    seed: int | None = None
    near_median: int | None = None
    median: float | None = None

    def record(self) -> dict:
        entry = {"fov_id": self.fov_id, "chosen_by": self.chosen_by}
        if self.chosen_by == "automatic":
            entry.update(rule=self.rule, value=self.value, seed=self.seed)
            if self.role == "example_1":
                entry.update(near_median=self.near_median, median=self.median)
        return entry


ROLES = ("example_1", "example_2", "example_3")


def choose_examples(fov_ids, reads, cells, owning=None, named=None):
    """The three example FOVs ("The example FOVs", M1).

    ``reads`` and ``cells`` map each FOV to its ``passed_filter`` and ``n_cells_retained`` (a value, or any
    null); ``owning`` the FOVs that own points (None: every FOV); ``named`` the configured three FOVs.
    Returns the three :class:`Example` roles in order, or the first unavailable read count (or
    ``undefined``) when no FOV has one and none is named.
    """
    if named is not None:
        return [Example(role, fov, "configuration") for role, fov in zip(ROLES, named)]
    index = {fov: i for i, fov in enumerate(fov_ids)}
    have = [f for f in fov_ids if is_value(reads.get(f))]
    if not have:
        return first(*(reads.get(f) for f in fov_ids)) or Unavailable("undefined", "no FOV has a read count")
    r = {f: int(reads[f]) for f in have}
    centre = median(list(r.values()))
    q = max(1, math.ceil(len(have) / 5))
    near = sorted(sorted(have, key=lambda f: (abs(r[f] - centre), index[f]))[:q], key=index.get)
    main = near[int(np.random.default_rng(EXAMPLE_SEED).integers(len(near)))]
    chosen = [Example("example_1", main, "automatic", "median_reads", r[main], EXAMPLE_SEED, len(near), centre)]
    by_cells = [f for f in fov_ids if is_value(cells.get(f)) and (owning is None or f in owning)]
    values, kind = ({f: int(cells[f]) for f in by_cells}, "cells") if by_cells else (r, "reads")
    for role, rule, sign in (("example_2", f"fewest_{kind}", 1), ("example_3", f"most_{kind}", -1)):
        taken = {e.fov_id for e in chosen}
        pool = [f for f in values if f not in taken] or list(values)
        fov = min(pool, key=lambda f: (sign * values[f], index[f]))
        chosen.append(Example(role, fov, "automatic", rule, values[fov], EXAMPLE_SEED))
    return chosen


# --- Other values of the page -----------------------------------------------------------------------------

def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def configuration_check(records):
    """The Summary's configuration check over the run records (``fov → record or None``).

    Returns ``{"text", "groups", "differing_keys"}``: "identical for all FOVs" for one distinct canonical
    ``config``, else "differs: n groups" with the top-level keys of ``config`` that differ between the
    groups; ``groups`` lists the FOVs of each group in first-seen order. ``no_run_record`` without a record.
    """
    present = {f: r for f, r in records.items() if r is not None}
    if not present:
        return Unavailable("no_run_record", "no FOV of the list has a run record")
    groups = {}
    for fov, record in present.items():
        groups.setdefault(_canonical(record.get("config")), []).append(fov)
    configs = [present[fovs[0]].get("config") or {} for fovs in groups.values()]
    keys = list(dict.fromkeys(k for config in configs for k in config))
    differing = [k for k in keys if len({_canonical(config.get(k)) for config in configs}) > 1]
    text = "identical for all FOVs" if len(groups) == 1 else f"differs: {len(groups)} groups"
    return {"text": text, "groups": list(groups.values()), "differing_keys": differing}


def z_size(export, placement, reference_images=None):
    """The Z size of the acquisition: the fused grid's (``export.json`` ``fusion.fovs[*].grid.shape_zyx[0]``),
    else the FOVs' reference grids as the export's assembly or the placement records them, else the shape of a
    reference image (``reference_images``, a callable called only then: :func:`reference_image_z_size`); None
    when none of these exists."""
    sources = (_get_list(export, "fusion", "fovs"), _get_list(export, "assembly", "fovs"),
               [{"grid": g.get("grid")} for g in ((placement or {}).get("inputs") or {}).values()
                if isinstance(g, dict)])
    for entries in sources:
        sizes = [int(e["grid"]["shape_zyx"][0]) for e in entries
                 if isinstance(e, dict) and isinstance(e.get("grid"), dict) and e["grid"].get("shape_zyx")]
        if sizes:
            return max(sizes)
    return None if reference_images is None else reference_images()


def reference_image_z_size(inputs, root, fov_ids):
    """The Z size from the reference image ``images/ref_merged/<fov>.tif`` of the first FOV in list order whose
    image exists and whose TIFF header settles its shape (``io.tiff._volume_header``: no pixel is decoded); the file
    is hashed in blocks and listed in ``report.json`` ``inputs`` (kind ``reference_image``). None when no FOV has one.
    """
    from pathlib import Path

    from starfinder.io.tiff import _volume_header
    for fov in fov_ids:
        path = Path(root) / "images" / "ref_merged" / f"{fov}.tif"
        if not path.is_file():
            continue
        if inputs.verify(path, "reference_image") is not None:
            continue
        header = _volume_header(path)
        if header is not None and header[0] is not None:
            return int(header[0][0])
    return None


def _get_list(mapping, *keys):
    for key in keys:
        mapping = mapping.get(key) if isinstance(mapping, dict) else None
    return mapping if isinstance(mapping, list) else []


def acquisition(records, planes):
    """The Summary's acquisition: rounds × channels from the union of the run records' ``channels`` and
    "3D" or "2D (Z = 1)" from the Z size ``planes`` (None when no record holds it)."""
    channels = {}
    for record in records.values():
        for name, entries in ((record or {}).get("channels") or {}).items():
            channels[name] = max(channels.get(name, 0), len(entries or []))
    if not channels:
        return Unavailable("no_run_record", "no run record holds channels")
    low, high = min(channels.values()), max(channels.values())
    text = f"{len(channels)} rounds × " + (f"{high} channels" if low == high else f"{low}–{high} channels")
    if planes is not None:
        text += ", 3D" if planes > 1 else ", 2D (Z = 1)"
    return {"text": text, "rounds": len(channels), "channels": [low, high], "z_size": planes}


def software(records, export_software=None):
    """The Summary's software: each distinct ``code`` (version, commit, dirty) of the run records with its FOV
    count, in first-seen order; without a run record the export's ``assembly.software``."""
    present = [r for r in records.values() if r is not None]
    if not present:
        if export_software is None:
            return Unavailable("no_run_record", "no FOV of the list has a run record")
        return [{"software": export_software, "fovs": None}]
    found = {}
    for record in present:
        code = record.get("code") or {}
        key = (code.get("version"), code.get("git_commit"), code.get("git_dirty"))
        found[key] = found.get(key, 0) + 1
    return [{"version": v, "git_commit": c, "git_dirty": d, "fovs": n} for (v, c, d), n in found.items()]


def run_time(records):
    """The Summary's run time: the earliest ``started_at`` to the latest ``ended_at`` of the run records."""
    starts = [r["started_at"] for r in records.values() if r is not None and r.get("started_at")]
    ends = [r["ended_at"] for r in records.values() if r is not None and r.get("ended_at")]
    if not starts:
        return Unavailable("no_run_record", "no FOV of the list has a run record")
    start = min(starts, key=datetime.fromisoformat)
    if not ends:
        return {"started_at": start, "ended_at": None, "seconds": None}
    end = max(ends, key=datetime.fromisoformat)
    return {"started_at": start, "ended_at": end, "seconds": run_seconds(start, end)}


def time_per_step(steps):
    """The Workflow overview's time per step: per step name (first-seen order), ``Σ seconds`` of each FOV's rows
    of that name, and the single value of a sample step. ``steps`` are rows of ``steps.csv`` (dicts)."""
    times = {}
    for row in steps:
        seconds = row.get("seconds")
        if not is_value(seconds) or (isinstance(seconds, float) and math.isnan(seconds)):
            continue
        entry = times.setdefault(row["name"], {})
        owner = row["fov_id"] if row["scope"] == "fov" else "sample"
        entry[owner] = entry.get(owner, 0.0) + float(seconds)
    return times
