"""The rules of the problematic FOVs (docs/report-algorithms.md, "Problematic FOVs").

Each rule of ``PROBLEM_RULES`` lists where a value lies outside a stated limit, or where a
categorical state holds; a row is a pointer, not a grade. :func:`rule_form` is the one form of
a rule on values; :func:`evaluate_rules` applies every rule of the first delivery to the
report's ``fovs`` and ``registration`` rows and gives the rows of ``problematic_fovs.csv``, the
``rules`` entries of ``report.json`` and the ``rule:<rule>`` rows of ``unavailable.csv``.
"""
from __future__ import annotations

from typing import NamedTuple

from ._config import ReportLimits, RuleLimit
from ._schema import PROBLEM_RULES, RULE_UNITS
from ._values import AXES, NA, SPREAD_FACTOR, Unavailable, first, is_value, mad, median

#: The section of the page each rule belongs to ("The rules of the first delivery").
RULE_SECTIONS = {"fov_failed": "Workflow overview", "fov_not_run": "Workflow overview",
                 "shift_residual": "Registration", "alignment_score": "Registration",
                 "registration_rejected": "Registration", "registration_fallback": "Registration",
                 "correct_form_fraction": "Readout", "matched_fraction": "Readout", "spots_near_zero": "Readout",
                 "not_placed": "Segmentation and assembly", "stage_position_fallback": "Segmentation and assembly",
                 "reads_no_cell": "Segmentation and assembly"}
_UNPLACED = ("missing", "failed")


class RuleForm(NamedTuple):
    """What the one form of a rule gives on a set of values: the keys it marks, the median of the values, and
    the limit applied (``max(k' × s, floor')`` or ``absolute``)."""

    marked: list
    median: float
    limit: float
    absolute: bool


def rule_form(values, limit: RuleLimit, side: str, deviations=None) -> RuleForm | None:
    """The one form of a rule on values ("The one form of a rule on values").

    ``values`` is a list of ``(key, value)`` pairs of the values that enter (non-null); ``side`` is ``low``,
    ``high`` or ``both``. With ``limit.absolute`` set, a value beyond it on the rule's side is marked
    (``both``: its deviation from the median beyond it). Otherwise ``m = median(v)``,
    ``s = 1.4826 × MAD(v)`` and a deviation ``d`` (``m − v`` low, ``v − m`` high, ``abs(v − m)`` both) larger
    than ``max(k' × s, floor')`` is marked, ``k'`` and ``floor'`` being 0 when None. The comparison is strict.
    ``deviations`` gives ``v − m`` per key when the caller holds it (the shift residual). None when no value
    enters or no limit is set.
    """
    if not values or all(v is None for v in (limit.k, limit.floor, limit.absolute)):
        return None
    numbers = [float(v) for _, v in values]
    centre = median(numbers)
    signed = {key: (deviations[key] if deviations is not None else float(v) - centre) for key, v in values}
    if limit.absolute is not None:
        bound = limit.absolute
        if side == "both":
            marked = [key for key, _ in values if abs(signed[key]) > bound]
        elif side == "low":
            marked = [key for key, v in values if float(v) < bound]
        else:
            marked = [key for key, v in values if float(v) > bound]
        return RuleForm(marked, centre, bound, True)
    spread = SPREAD_FACTOR * median([abs(signed[key]) for key, _ in values]) if deviations is not None \
        else SPREAD_FACTOR * mad(numbers, centre)
    bound = max((limit.k or 0.0) * spread, limit.floor or 0.0)
    distance = {"low": lambda d: -d, "high": lambda d: d, "both": abs}[side]
    marked = [key for key, _ in values if distance(signed[key]) > bound]
    return RuleForm(marked, centre, bound, False)


def _row(fov, rule, reason, *, round_name=NA, axis=NA, measured=NA, sample_median=NA, limit=NA):
    return {"fov_id": fov, "section": RULE_SECTIONS[rule], "rule": rule, "round": round_name, "axis": axis,
            "measured": measured, "sample_median": sample_median, "limit": limit,
            "unit": RULE_UNITS.get(rule, NA), "reason": reason}


def _not_evaluated(cells, fallback):
    """The reason a rule is not evaluated for the whole sample: the first reason of its values' cells."""
    return first(*cells) or fallback


class RuleInputs(NamedTuple):
    """What the rules read besides the report's ``fovs`` and ``registration`` rows."""

    fov_ids: list
    records: dict             # fov → run record or None
    placement_flags: dict | Unavailable   # fov → the placement's flags ("" for none), or the placement's reason
    rounds_reason: Unavailable | None     # the reason the registration has no round, when it has none
    shift_medians: dict       # (round, axis) → the round's median shift


def evaluate_rules(fovs, registration, limits: ReportLimits, inputs: RuleInputs):
    """Every rule of ``PROBLEM_RULES`` on the report's rows ("The rules of the first delivery").

    Returns ``(rows, rules, part_rows)``: the rows of ``problematic_fovs.csv`` in the contract's order (FOVs
    in list order, then the rule order, then rounds and axes), per rule ``evaluated`` and ``fovs_marked``,
    and the ``rule:<rule>`` rows (part, Unavailable, row key) of the rules not evaluated for the sample.
    A rule without a limit is not evaluated with ``not_configured``.
    """
    marks, state = [], {}
    by_fov = {row["fov_id"]: row for row in fovs}
    rounds = list(dict.fromkeys(row["round"] for row in registration))
    any_record = any(r is not None for r in inputs.records.values())
    none = Unavailable("no_run_record", "no FOV of the list has a run record")

    def add(rule, fov, reason, order=(0, 0), **values):
        marks.append(((inputs.fov_ids.index(fov), PROBLEM_RULES.index(rule), *order), _row(fov, rule, reason,
                                                                                           **values)))

    # Workflow overview.
    if not any_record:
        state["fov_failed"] = state["fov_not_run"] = none
    else:
        for fov in inputs.fov_ids:
            record = inputs.records[fov]
            if record is None:
                add("fov_not_run", fov, "expected and not run")
            elif record.get("status") in ("failed", "interrupted"):
                error = record.get("error") or {}
                where = error.get("step") if error.get("round") is None else f"{error.get('step')}, {error['round']}"
                add("fov_failed", fov, f"run {record['status']} at {where}")

    # Registration.
    registration_reason = None
    if not any_record:
        registration_reason = none
    elif not rounds:
        registration_reason = inputs.rounds_reason or Unavailable("not_configured", "no FOV registered a round")
    if registration_reason is not None:
        for rule in ("shift_residual", "alignment_score", "registration_rejected", "registration_fallback"):
            state[rule] = registration_reason
    else:
        _shift_residual(registration, rounds, limits.shift_residual, inputs.shift_medians, add, state)
        _alignment_score(registration, rounds, limits.alignment_score, add, state)
        _registration_entries(registration, rounds, inputs.records, add)

    # Readout.
    for rule in ("correct_form_fraction", "matched_fraction"):
        cells = [by_fov[f][rule] for f in inputs.fov_ids]
        values = [(f, by_fov[f][rule]) for f in inputs.fov_ids if is_value(by_fov[f][rule])]
        form = rule_form(values, getattr(limits, rule), "low")
        if not values:
            state[rule] = _not_evaluated(cells, Unavailable("undefined", "no FOV has the value"))
        elif form is not None:
            label = rule.replace("_", " ").replace("correct form", "correct-form")
            for fov in form.marked:
                add(rule, fov, f"{label} below the limit", measured=float(by_fov[fov][rule]),
                    sample_median=NA if form.absolute else form.median, limit=form.limit)
    cells = [by_fov[f]["spots"] for f in inputs.fov_ids]
    spots = [(f, by_fov[f]["spots"]) for f in inputs.fov_ids if is_value(by_fov[f]["spots"])]
    if not spots:
        state["spots_near_zero"] = _not_evaluated(cells, Unavailable("undefined", "no FOV has spots"))
    else:
        centre = median([v for _, v in spots])
        bound = limits.spots_near_zero * centre
        for fov, value in spots:
            if value < bound:
                add("spots_near_zero", fov, "spot count close to zero", measured=float(value), sample_median=centre,
                    limit=bound)

    # Segmentation and assembly.
    cells = [by_fov[f]["placement_status"] for f in inputs.fov_ids]
    if not any(is_value(c) for c in cells):
        state["not_placed"] = _not_evaluated(cells, Unavailable("undefined", "no FOV has a placement status"))
    else:
        for fov in inputs.fov_ids:
            status = by_fov[fov]["placement_status"]
            if is_value(status) and status in _UNPLACED:
                add("not_placed", fov, f"not placed ({status})")
    if isinstance(inputs.placement_flags, Unavailable):
        state["stage_position_fallback"] = inputs.placement_flags
    else:
        for fov in inputs.fov_ids:
            if "fallback" in str(inputs.placement_flags.get(fov) or "").split(";"):
                add("stage_position_fallback", fov, "fell back to the stage position")
    owning = [f for f in inputs.fov_ids if by_fov[f]["owned_area_um2"] is not NA]
    cells = [by_fov[f][c] for f in owning for c in ("n_molecules_retained", "n_cells_retained")]
    pairs = [f for f in owning if is_value(by_fov[f]["n_molecules_retained"]) and
             is_value(by_fov[f]["n_cells_retained"])]
    if not pairs:
        state["reads_no_cell"] = _not_evaluated(cells, Unavailable("undefined", "no owning FOV"))
    else:
        for fov in pairs:
            if by_fov[fov]["n_molecules_retained"] > 0 and by_fov[fov]["n_cells_retained"] == 0:
                add("reads_no_cell", fov, "reads but no cell")

    marks.sort(key=lambda m: m[0])
    rows = [row for _, row in marks]
    rules, part_rows = {}, []
    for rule in PROBLEM_RULES:
        configured = rule not in RULE_UNITS or rule == "spots_near_zero" or any(
            v is not None for v in (getattr(limits, rule).k, getattr(limits, rule).floor,
                                    getattr(limits, rule).absolute))
        if not configured:
            rules[rule] = {"evaluated": False, "fovs_marked": NA}
            part_rows.append((f"rule:{rule}", Unavailable("not_configured", f"config.limits.{rule} sets no limit"),
                              ""))
        elif rule in state:
            rules[rule] = {"evaluated": False, "fovs_marked": NA}
            part_rows.append((f"rule:{rule}", state[rule], ""))
        else:
            rules[rule] = {"evaluated": True, "fovs_marked": len({r["fov_id"] for r in rows if r["rule"] == rule})}
    return rows, rules, part_rows


def _shift_residual(registration, rounds, limit, medians, add, state):
    """Per round and axis, over the FOVs whose round succeeded with a shift; both sides. A row holds the shift as
    ``measured`` and the round's median shift as ``sample_median`` under both forms of the limit. A limit with
    ``k``, ``floor`` and ``absolute`` all None is not evaluated (``evaluate_rules`` records ``not_configured``)."""
    if all(v is None for v in (limit.k, limit.floor, limit.absolute)):
        return
    cells = [row[f"shift_{a}"] for row in registration for a in AXES]
    if not any(is_value(c) for c in cells):
        state["shift_residual"] = _not_evaluated(cells, Unavailable("undefined", "no FOV has a shift"))
        return
    for r_index, name in enumerate(rounds):
        rows = [row for row in registration if row["round"] == name and row["outcome"] == "succeeded"]
        for a_index, axis in enumerate(AXES):
            values = [(row["fov_id"], row[f"shift_{axis}"]) for row in rows if is_value(row[f"shift_{axis}"])]
            if not values:
                continue
            deviations = {row["fov_id"]: row[f"residual_{axis}"] for row in rows if is_value(row[f"shift_{axis}"])}
            form = rule_form(values, limit, "both", deviations)
            if form is None:
                continue
            shift = dict(values)
            for fov in form.marked:
                add("shift_residual", fov, f"shift residual above the limit ({axis}, {name})", (r_index, a_index),
                    round_name=name, axis=axis, measured=float(shift[fov]), sample_median=medians[(name, axis)],
                    limit=form.limit)


def _alignment_score(registration, rounds, limit, add, state):
    """Per round, the low side of ``alignment_score`` over the FOVs with a score."""
    if all(v is None for v in (limit.k, limit.floor, limit.absolute)):
        return
    cells = [row["alignment_score"] for row in registration]
    if not any(is_value(c) for c in cells):
        state["alignment_score"] = _not_evaluated(cells, Unavailable("undefined", "no FOV has a score"))
        return
    for r_index, name in enumerate(rounds):
        values = [(row["fov_id"], row["alignment_score"]) for row in registration
                  if row["round"] == name and is_value(row["alignment_score"])]
        form = rule_form(values, limit, "low")
        if form is None:
            continue
        score = dict(values)
        for fov in form.marked:
            add("alignment_score", fov, f"alignment score below the limit ({name})", (r_index, 0), round_name=name,
                measured=float(score[fov]), sample_median=NA if form.absolute else form.median, limit=form.limit)


def _registration_entries(registration, rounds, records, add):
    """``registration_rejected``: some estimation entry of the round was rejected; ``registration_fallback``:
    ``fallback`` true on a succeeded round."""
    for row in registration:
        record = records.get(row["fov_id"])
        if record is None:
            continue
        r_index = rounds.index(row["round"])
        entries = (record.get("registration") or {}).get(row["round"]) or []
        if any(e.get("record") == "estimation" and e.get("outcome") == "rejected" for e in entries):
            add("registration_rejected", row["fov_id"], f"registration rejected ({row['round']})", (r_index, 0),
                round_name=row["round"])
        if row["outcome"] == "succeeded" and row["fallback"] is True:
            methods = row["methods"] if is_value(row["methods"]) else ""
            add("registration_fallback", row["fov_id"],
                f"registration succeeded with a fallback method ({row['round']}: {methods})", (r_index, 0),
                round_name=row["round"])
