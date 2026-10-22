"""The report page ``report.html`` (docs/report-algorithms.md, "Layout of the page", "How the page is built").

The page is the template shipped with the package (``templates/report.html``, read with
``importlib.resources``): the page skeleton, the CSS and the inline script. Its named slots
``<!--slot:<name>-->`` are each replaced once, in the fixed order :data:`SLOTS`, by a string
built here with the standard library: text through ``html.escape``; the tables of the
contract's "The data embedded in the page" as one block ``<script type="application/json"
id="report-data">``; each static figure as an ``<img>`` with a ``data:`` address; the facts,
the workflow chart, the readout bars, the Files tables and the box of each part that is
unavailable. The FOV maps, their tooltips and the paged tables are drawn by the inline script
from the data block alone; the reason texts and the labels of the controls are constants of
that script, and :data:`REASON_TEXTS` is read from it, so that the page has one copy of them.

The page loads nothing (no address other than ``data:`` and ``#…``) and has a light theme only.
Its static figures are made here from the collection (the time per step, the heatmap of the
residuals and the distributions across FOVs) or by the parts that draw them (``_figures``:
the examples of the replay and of the segmentation previews; ``_measurements``: the sample
measurements), never twice.
"""
from __future__ import annotations

import base64
import html
import json
import math
import os
import re
from importlib import resources

import numpy as np
import pandas as pd

from starfinder import _layout

from ._schema import READOUT_LEVELS
from ._values import (Unavailable, acquisition, configuration_check, is_value, reference_image_z_size, run_time,
                      software, time_per_step, z_size)

TEMPLATE = "report.html"
#: The slots of the template, in document order; each is filled exactly once.
SLOTS = ("title", "heading", "facts", "key_numbers", "problematic", "chart", "fov_status", "time", "prep_title",
         "prep_example", "prep_intensity", "reg_maps", "reg_heatmap", "reg_title", "reg_example", "read_totals",
         "read_map", "read_distributions", "read_table", "cells_map", "placement", "seg_examples", "population",
         "per_cell", "markers", "mecr", "reads_z", "files_input", "files_configuration", "files_output", "data")
#: The tables of the data block, in the contract's order ("The data embedded in the page"); ``cells`` and
#: ``intensity`` are saved but not embedded.
DATA_TABLES = ("fovs", "registration", "steps", "readout_totals", "population", "problematic_fovs", "mecr",
               "reads_along_z", "unavailable")
#: The eight sections of the page and, in each, the ids of its blocks, in page order ("Layout of the page").
SECTIONS = (
    ("summary", "Summary", ("b-facts", "b-key-numbers", "b-problematic")),
    ("run", "Workflow overview", ("b-chart", "b-fov-status", "b-time")),
    ("prep", "Preprocessing", ("b-prep-example", "b-prep-intensity")),
    ("reg", "Registration", ("b-reg-maps", "b-reg-heatmap", "b-reg-example")),
    ("reads", "Readout", ("b-read-totals", "b-read-map", "b-read-distributions", "b-read-table")),
    ("cells", "Segmentation and assembly", ("b-cells-map", "b-placement", "b-seg-examples", "b-population")),
    ("sample", "Sample measurements", ("b-per-cell", "b-markers", "b-mecr", "b-reads-z")),
    ("files", "Files", ("b-files-input", "b-files-configuration", "b-files-output")),
)
#: The size budget of the page (SB1): 6 MiB + 16 KiB per FOV + 3 MiB per marker figure.
KIB, MIB = 1024, 1024 * 1024
_SLOT = re.compile(r"<!--slot:([a-z_]+)-->")
_LEVELS = {"spots": "Spots detected", "correct_form": "Correct barcode form", "matched": "Matched to the codebook",
           "merged_by_deduplication": "Merged by deduplication", "passed_filter": "Passed the read QC filter",
           "reads_in_cells": "Reads assigned to cells"}


def template_text(name=TEMPLATE) -> str:
    """A shipped template: ``starfinder/report/templates/report.html``, or the one named."""
    return resources.files("starfinder.report").joinpath("templates", name).read_text(encoding="utf-8")


def _script_constant(template, name):
    match = re.search(rf"var {name} = (\{{.*?\}});\n", template)
    if match is None:
        raise ValueError(f"the template's script has no constant {name}")
    return json.loads(match.group(1))


#: The text of each reason ("How an unavailable part is shown"), the constant ``REASONS`` of the template's script.
REASON_TEXTS = _script_constant(template_text(), "REASONS")


def budget_bytes(n_fovs, marker_figures) -> int:
    """The size budget of ``report.html`` (SB1): ``6 MiB + 16 KiB × FOVs + 3 MiB × marker figures``."""
    return 6 * MIB + 16 * KIB * n_fovs + 3 * MIB * marker_figures


def fill(template, values, slots=SLOTS) -> str:
    """The template with each slot replaced once, in the order ``slots`` (:data:`SLOTS` for ``report.html``); a slot
    the template lacks or holds twice, or a value without its slot, raises ValueError before anything is written."""
    pieces = _SLOT.split(template)
    names = pieces[1::2]
    if names != list(slots):
        raise ValueError(f"the template's slots {names} are not {list(slots)}, each once in this order")
    if set(values) != set(slots):
        raise ValueError(f"slots without a value {sorted(set(slots) - set(values))}, values without a slot "
                         f"{sorted(set(values) - set(slots))}")
    out = [pieces[0]]
    for name, text in zip(names, pieces[2::2]):
        out += [values[name], text]
    return "".join(out)


def _plain(value):
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or value is pd.NA or (isinstance(value, float) and math.isnan(value)):
        return None
    return value


def table_rows(frame) -> list[dict]:
    """A table's rows as objects of its columns, a missing value None."""
    columns = list(frame.columns)
    return [{c: _plain(v) for c, v in zip(columns, values)}
            for values in frame.astype(object).itertuples(index=False, name=None)]


def data_block(frames) -> str:
    """The JSON text of the data block: the tables of :data:`DATA_TABLES` (strict JSON, a missing value null), with
    every ``</`` written ``<\\/`` and every ``<!--`` ``\\u003c!--``, so that the block cannot end its script element."""
    data = {name: table_rows(frames[name]) for name in DATA_TABLES}
    text = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    return text.replace("</", "<\\/").replace("<!--", "\\u003c!--")


def _e(value) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _tip(lines) -> str:
    return _e("|".join(str(line) for line in lines if line is not None and line != ""))


def image(figure, name, alt) -> str:
    """A static figure as an ``<img>`` with a ``data:`` address, its pixel size and the figure's name."""
    mime = "image/png" if figure.format == "png" else "image/jpeg"
    data = base64.b64encode(figure.data).decode("ascii")
    return (f'<img class="fig" src="data:{mime};base64,{data}" alt="{_e(alt)}" width="{figure.width}" '
            f'height="{figure.height}" data-figure="{_e(name)}">')


def grid(images) -> str:
    """The figures of a three-figure grid, in rows of three (``g3 fixed``); a row of two figures has two columns and
    a single figure takes a half column (``g2``), as :func:`._figures.grid_slot` draws them."""
    rows = [images[k:k + 3] for k in range(0, len(images), 3)]
    return "".join(f'<div class="{"g3 fixed" if len(row) == 3 else "g2"}">{"".join(row)}</div>' for row in rows)


def unavailable_box(row, *, ratio="16/9", line=False, inline=False) -> str:
    """The box of a row of ``unavailable.csv`` without a table: "Unavailable:" and the reason's text, then
    the detail in smaller type; a box of the figure's aspect ratio, a line, or inline text (a fact, a file)."""
    attributes = (f'data-part="{_e(row["part"])}" data-key="{_e(row["row_key"])}" '
                  f'data-reason="{_e(row["reason"])}"')
    detail = f'<small>{_e(row["detail"])}</small>' if row.get("detail") else ""
    text = f'Unavailable: {_e(REASON_TEXTS[row["reason"]])}'
    if inline:
        return f'<span class="phi" {attributes}>{text}{detail}</span>'
    style = "" if line else f' style="--ar:{ratio}"'
    return f'<div class="ph{" line" if line else ""}" {attributes}{style}><span>{text}</span>{detail}</div>'


def _derived(reason, what, ratio="1/1") -> str:
    """A box for a figure drawn from table cells that has nothing to draw (its cells' rows explain it)."""
    return (f'<div class="ph" data-cells="{_e(what)}" style="--ar:{ratio}"><span>Unavailable: '
            f'{_e(REASON_TEXTS[reason])}</span></div>')


def _int(value) -> str:
    return f"{int(value):,}"


def _number(value) -> str:
    value = float(value)
    return f"{value:,.0f}" if value.is_integer() or abs(value) >= 1000 else f"{value:,.4g}"


def _utc(iso):
    from datetime import datetime, timezone
    moment = datetime.fromisoformat(iso)
    return (moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)


def moment_text(iso) -> str:
    """A time as the Summary writes it in UTC, on one line: ``2026-10-13, 18:18:57 UTC`` (the page's script writes
    it again in the viewer's time zone)."""
    moment = _utc(iso)
    return f"{moment:%Y-%m-%d, %H:%M:%S} UTC"


def span_text(start, end) -> str:
    """The run time as the Summary writes it in UTC, on one line: the date, the start and the end
    (``2026-10-13, 14:19:02 – 14:42:12 UTC``), the end's date too when it is another day, "not ended" without an
    end."""
    first = _utc(start)
    if not end:
        return f"{first:%Y-%m-%d, %H:%M:%S} UTC – not ended"
    last = _utc(end)
    shown = f"{last:%H:%M:%S}" if last.date() == first.date() else f"{last:%Y-%m-%d, %H:%M:%S}"
    return f"{first:%Y-%m-%d, %H:%M:%S} – {shown} UTC"


def _get(mapping, *keys):
    for key in keys:
        if not isinstance(mapping, dict) or key not in mapping:
            return None
        mapping = mapping[key]
    return mapping


# --- The figures ----------------------------------------------------------------------------------------------------

class PageFigures:
    """The static figures of a collection, each drawn once: per name a :class:`._figures.FigureImage`, None when the
    part draws no figure; ``sample`` the figures of "Sample measurements" (``_measurements.SampleFigures``)."""

    def __init__(self, collection, tables):
        from . import _figures
        c = self.c = collection
        quality = c.config.jpeg_quality
        self.time, self.not_recorded = self._time(tables["steps"], tables["fovs"])
        rounds = list(dict.fromkeys(r["round"] for r in tables["registration"]))
        marked = [r["fov_id"] for r in tables["problematic_fovs"] if r["rule"] == "shift_residual"]
        self.heatmap = (_figures.residual_heatmap_figure(tables["registration"], c.fov_ids, rounds, marked)
                        if tables["registration"] else None)
        self.distributions = self._distributions(tables["fovs"])
        replay = c.replay if c.replay is not None and c.replay.missing is None else None
        self.replay = replay
        ok = replay is not None and replay.reference in replay.rounds
        self.preprocessing = _figures.preprocessing_figure(replay, quality=quality) if ok else None
        self.intensity = _figures.intensity_figure(replay) if ok else None
        moving = replay.rounds.get(replay.moving[0]) if ok and replay.moving else None
        self.registration = (_figures.registration_figure(replay, quality=quality)
                             if moving is not None and moving.after is not None and moving.transform_missing is None
                             else None)
        previews = c.previews if isinstance(c.previews, dict) else {}
        self.segmentation = (_figures.segmentation_figure(previews, quality=quality, reasons=REASON_TEXTS)
                             if any(hasattr(p, "background") for p in previews.values()) else None)
        self.sample = c.measurements.figures()

    @staticmethod
    def _time(steps, fovs):
        """Time per step, its axes spanning the FOV rectangles of the FOV status map beside it (drawn from the same
        rows of ``fovs`` as the page's script draws the map)."""
        from ._figures import time_per_step_figure
        from ._values import NA
        times = time_per_step(steps)
        fov_names, sample_names, not_recorded = [], [], []
        for row in steps:
            names = fov_names if row["scope"] == "fov" else sample_names
            if row["seconds"] is not NA and row["name"] not in names:
                names.append(row["name"])
        for name in fov_names + sample_names:
            if name not in times:
                not_recorded.append(name)
        figure = time_per_step_figure(times, [n for n in fov_names if n in times],
                                      [n for n in sample_names if n in times], not_recorded, fovs)
        return figure, not_recorded

    @staticmethod
    def _distributions(fovs):
        from ._figures import distribution_figure, grid_slot
        from ._values import first
        figures = []
        shown = (("spots", "spots per FOV", True), ("correct_form_fraction", "fraction with correct form", False),
                 ("matched_fraction", "fraction matched to the codebook", False))
        for column, label, integer in shown:
            cells = [row[column] for row in fovs]
            values = [v for v in cells if is_value(v)]
            missing = first(*cells)
            empty = f"no value (Unavailable: {REASON_TEXTS[missing.reason]})" if missing is not None else None
            figures.append((column, label, distribution_figure(values, column, label, integer=integer, empty=empty,
                                                               slot=grid_slot(len(shown)))))
        return figures


# --- The page -------------------------------------------------------------------------------------------------------

class _Page:
    def __init__(self, collection, tables, frames, unavailable, summary, identity, generated_at, config_json, folder,
                 figures):
        self.c, self.tables, self.frames, self.summary = collection, tables, frames, summary
        self.unavailable, self.identity = unavailable, identity
        self.generated_at, self.config_json, self.folder, self.figures = generated_at, config_json, folder, figures
        self.rows = {name: table_rows(frames[name]) for name in DATA_TABLES}
        self.named = {(u["table"], u["column"], u["row_key"]): u for u in unavailable if u["table"] is not None}
        self.placed = any(r["centre_x"] is not None and r["centre_y"] is not None for r in self.rows["fovs"])

    def parts(self, part):
        """The rows of ``unavailable.csv`` without a table of a part, in their order."""
        return [u for u in self.unavailable if u["table"] is None and u["part"] == part]

    def dash(self, table, column, key):
        """The "—" of a named null cell written by the filler, with its reason in the tooltip data."""
        u = self.named[(table, column, key)]
        return (f'<span data-cell="{_e(table)}|{_e(column)}|{_e(key)}" '
                f'data-t="{_tip(["Unavailable: " + REASON_TEXTS[u["reason"]], u["detail"]])}">—</span>')

    def values(self) -> dict:
        c = self.c
        title = f"Processing report · {c.sample}"
        values = {
            "title": _e(title),
            # The heading alone: the Summary's facts name the dataset and the sample.
            "heading": "<h1>Processing report</h1>",
            "facts": self.facts(), "key_numbers": self.key_numbers(), "problematic": self.problematic(),
            "chart": self.chart(), "fov_status": self.fov_status(),
            "time": image(self.figures.time, "time", "Time per step" + (
                f"; not recorded: {', '.join(self.figures.not_recorded)}" if self.figures.not_recorded else "")),
            **self.preprocessing(), **self.registration(), **self.readout(), **self.assembly(), **self.sample(),
            **self.files(),
            "data": data_block(self.frames),
        }
        return values

    # Summary ------------------------------------------------------------------------------------------------------

    def facts(self) -> str:
        c, identity = self.c, self.identity
        rows = {u["row_key"]: u for u in self.parts("summary.identity")}

        def fact(label, value, *, attrs=""):
            return f'<div><dt>{_e(label)}</dt><dd{attrs}>{value}</dd></div>'
        acq = identity["acquisition"]
        if isinstance(acq, Unavailable):
            planes = identity["z_size"]
            text = unavailable_box(rows["entry=acquisition"], inline=True)
            if planes is not None:
                text += _e(", 3D" if planes > 1 else ", 2D (Z = 1)")
        else:
            text = _e(acq["text"])
        if "entry=z_size" in rows:
            text += " " + unavailable_box(rows["entry=z_size"], inline=True)
        check = identity["configuration"]
        if isinstance(check, Unavailable):
            configuration = fact("Configuration", unavailable_box(rows["entry=configuration"], inline=True))
        else:
            groups = [f"group {i + 1}: {', '.join(g)}" for i, g in enumerate(check["groups"])]
            text_check = check["text"] + (f" ({', '.join(check['differing_keys'])})" if check["differing_keys"] else "")
            configuration = fact("Configuration", _e(text_check), attrs=f' data-t="{_tip(["Configuration"] + groups)}"')
        found = identity["software"]
        if isinstance(found, Unavailable):
            version = unavailable_box(rows["entry=software"], inline=True)
        else:
            parts = []
            for entry in found:
                if "software" in entry:
                    parts.append(f"<code>{_e(json.dumps(entry['software'], sort_keys=True))}</code> (export)")
                    continue
                commit = (entry.get("git_commit") or "")[:12]
                dirty = ", modified" if entry.get("git_dirty") else ""
                fovs = f" ({entry['fovs']} FOVs)" if len(found) > 1 else ""
                parts.append(f"<code>{_e(entry.get('version'))} @ {_e(commit)}</code>{_e(dirty + fovs)}")
            version = "; ".join(parts)
        timing = identity["run_time"]
        if isinstance(timing, Unavailable):
            run = fact("Run time", unavailable_box(rows["entry=run_time"], inline=True))
        else:
            tip = _tip(["Run time", f"{timing['started_at']} to {timing['ended_at'] or 'not ended'}",
                        f"{timing['seconds']:.1f} s" if timing["seconds"] is not None else None])
            end = f' data-end="{_e(timing["ended_at"])}"' if timing["ended_at"] else ""
            run = fact("Run time", f'<span id="run-span" data-start="{_e(timing["started_at"])}"{end} data-t="{tip}">'
                                   f'{_e(span_text(timing["started_at"], timing["ended_at"]))}</span>',
                       attrs=' id="run-time" class="one"')
        # The text of Generated, its tooltip and its time are inside the element (DT1 leaves the element's text out).
        generated = fact("Generated", f'<span id="generated-at" data-at="{_e(self.generated_at)}" '
                                      f'data-t="{_tip(["Generated", self.generated_at])}">'
                                      f'{_e(moment_text(self.generated_at))}</span>',
                         attrs=' id="generated" class="one"')
        route = c.route if c.route is not None else "no export"
        return "".join([fact("Dataset", f"<code>{_e(c.dataset_id)}</code>"),
                        fact("Sample", f"<code>{_e(c.sample)}</code>"), fact("Route", _e(route)),
                        fact("Acquisition", text), configuration, fact("Version", version), run, generated])

    def key_numbers(self) -> str:
        summary = self.summary

        def number(key, label, show):
            value = summary[key]
            if value is None and ("summary", key, "") in self.named:
                shown = self.dash("summary", key, "")
            else:
                shown = _e(show(value)) if value is not None else ""
            return f'<div class="kpi"><b>{shown}</b><span>{_e(label)}</span></div>'
        expected = summary["fovs_expected"]
        return "".join([
            number("fovs_succeeded", "FOVs succeeded", lambda v: f"{_int(v)} / {_int(expected)}"),
            number("spots_median_per_fov", "median spots per FOV", _number),
            number("matched_fraction", "spots matched to the codebook", lambda v: f"{100 * v:.1f} %"),
            number("counted_cells", "counted cells", _int),
            number("reads_per_cell_median", "median reads per cell", _number),
            number("genes_per_cell_median", "median genes per cell", _number)])

    def problematic(self) -> str:
        if self.rows["problematic_fovs"]:
            out = '<div id="t-problems"></div>'
        else:
            out = '<div class="panel"><div class="none">No FOV outside the limits</div></div>'
        rules = [u for u in self.unavailable if u["table"] is None and u["part"].startswith("rule:")]
        if rules:
            items = [f'{_e(u["part"][len("rule:"):])}: {unavailable_box(u, inline=True)}' for u in rules]
            out += f'<p class="ne">Not evaluated: {"; ".join(items)}</p>'
        return out

    # Workflow overview -----------------------------------------------------------------------------------------------

    def chart(self) -> str:
        c = self.c
        records = [c.records[f] for f in c.fov_ids if c.records[f] is not None]
        lead = next((r for r in records if r.get("status") == "succeeded"), records[0] if records else None)
        steps = self.rows["steps"]
        names = {}
        for row in steps:
            if row["scope"] == "fov":
                names.setdefault(row["fov_id"], []).append(row["name"])
        lead_fov = lead.get("fov_id") if lead else None
        lead_names = names.get(lead_fov, [])
        differ = sum(1 for r in records if names.get(r.get("fov_id"), []) != lead_names)
        tip = f"{differ} of {len(records)} FOVs ran another list of steps" if differ else None
        pipeline = _get(lead, "config", "pipeline") or {}

        def method(name):
            for row in steps:
                if row["scope"] == "fov" and row["fov_id"] == lead_fov and row["name"] == name and row["method"]:
                    return row["method"]
            return None

        def card(title, detail=None, *, off=None):
            return (title, detail, off)
        fov_cards = []
        if lead is None:
            for title in ("Preprocessing", "Registration", "Spot finding", "Decoding", "Read QC filter",
                          "Deduplication"):
                fov_cards.append(card(title, off="no run record"))
        else:
            recipe = _get(lead, "preprocessing", "recipe", "steps") or []
            fov_cards.append(card("Preprocessing", " → ".join(recipe)) if pipeline.get("preprocessing") and recipe
                             else card("Preprocessing", off="not run"))
            for title, key, step, default in (("Registration", "registration", "register", None),
                                              ("Spot finding", "spot_finding", "find_spots", None),
                                              ("Decoding", "decoding", "decode_barcodes", None),
                                              ("Read QC filter", "filtering", "filter_reads", None)):
                if pipeline.get(key) is None:
                    fov_cards.append(card(title, off="not run"))
                elif step not in lead_names:
                    fov_cards.append(card(title, off="not yet run"))
                else:
                    fov_cards.append(card(title, method(step) or _get(pipeline, key, "method") or ""))
            dedup = pipeline.get("deduplication")
            if dedup is None:
                fov_cards.append(card("Deduplication", off="not run"))
            elif "deduplicate_reads" not in lead_names:
                fov_cards.append(card("Deduplication", off="not yet run"))
            else:
                fov_cards.append(card("Deduplication", f"within {dedup.get('distance_voxels')} voxels"))
        if c.route != "after_fusion":
            if c.assignment is None:
                fov_cards += [card("Segmentation", off="not run"), card("Assignment", off="not run")]
            elif lead_fov is not None and "segment" in lead_names:
                fov_cards += [card("Segmentation", method("segment") or ""), card("Assignment", method("assign") or "")]
            else:
                fov_cards += [card("Segmentation", off="not yet run"), card("Assignment", off="not yet run")]
        sample_cards = [card("Stitching", _get(c.placement, "method", "method") or "") if c.placement is not None
                        else card("Stitching", off="not yet run")]
        export = c.export
        parts = (export or {}).get("parts") or {}
        if c.route == "after_fusion":
            sample_cards.append(self._part_card("Segmentation", "segment", parts,
                                                next((r["method"] for r in steps if r["scope"] == "sample"
                                                      and r["name"] == "segment" and r["method"]), "")))
        sample_cards.append(self._part_card("Assembly", "assemble", parts,
                                            _get(c.assembly, "config", "ownership") or
                                            _get(c.assembly, "ownership", "rule") or ""))
        if export is None:
            sample_cards.append(card("Export", off="not yet run"))
        else:
            written = [p for p, s in parts.items() if s in ("written", "reused")]
            kinds = sorted({o.get("dimensionality") for o in _get(export, "fusion", "outputs") or []
                            if isinstance(o, dict) and o.get("dimensionality")})
            dimensionality = "both" if len(kinds) > 1 else (kinds[0] if kinds else None)
            sample_cards.append(card("Export", ", ".join(written) + (f"; {dimensionality}" if dimensionality else "")))
        number = 0

        def lane(label, cards, tooltip):
            nonlocal number
            html_cards = []
            for title, detail, off in cards:
                number += 1
                attrs = f' data-t="{_tip([title, tooltip])}"' if tooltip else ""
                html_cards.append(f'<div class="st{" off" if off else ""}"{attrs}><b><span class="n">{number}</span>'
                                  f'{_e(title)}</b><i>{_e(off or detail)}</i></div>')
            return (f'<div class="lane"><span class="lab">{_e(label)}</span>'
                    f'<div class="cards">{"".join(html_cards)}</div></div>')
        out = lane("Per FOV", fov_cards, tip) + lane("Per sample", sample_cards, None)
        missing = self.parts("workflow.steps")
        if missing:
            out += '<div class="lanes-ph">' + "".join(unavailable_box(u, line=True) for u in missing) + "</div>"
        return out

    @staticmethod
    def _part_card(title, part, parts, detail):
        status = parts.get(part)
        if status is None:
            return (title, None, "not yet run")
        if status not in ("written", "reused"):
            return (title, None, "not run" if status in ("off", "not_run") else status)
        return (title, detail, None)

    def fov_status(self) -> str:
        if self.placed:
            return '<div id="m-status"></div><div class="lg one" id="lg-status"></div>'
        return '<div id="t-status"></div>'

    # Preprocessing and Registration ----------------------------------------------------------------------------------

    def _example_title(self, round_name):
        replay = self.figures.replay
        return f" · {_e(replay.fov_id)}, {_e(round_name)}" if replay is not None and round_name else ""

    def _figure_or_rows(self, figure, name, alt, rows, ratio):
        if figure is not None:
            return image(figure, name, alt) + "".join(unavailable_box(u, line=True) for u in rows)
        if not rows:
            return ""
        return unavailable_box(rows[0], ratio=ratio) + "".join(unavailable_box(u, line=True) for u in rows[1:])

    def preprocessing(self) -> dict:
        f = self.figures
        reference = f.replay.reference if f.preprocessing is not None else None
        return {"prep_title": self._example_title(reference),
                "prep_example": self._figure_or_rows(f.preprocessing, "preprocessing", "Preprocessing example",
                                                     self.parts("preprocessing.example"), "4/3"),
                "prep_intensity": self._figure_or_rows(f.intensity, "intensity", "Intensity after each step",
                                                       self.parts("preprocessing.intensity"), "3/1")}

    def registration(self) -> dict:
        f = self.figures
        maps = self.parts("registration.maps")
        if maps:
            reg_maps = "".join(unavailable_box(u, ratio="3/1") for u in maps)
        elif not self.placed:
            reg_maps = _derived(self._centre_reason(), "fovs.centre_x", "3/1")
        else:
            reg_maps = '<div class="g3" id="reg-maps"></div><div class="lg" id="reg-lg"></div>'
        moving = f.replay.moving[0] if f.registration is not None else None
        return {"reg_maps": reg_maps,
                "reg_heatmap": self._figure_or_rows(f.heatmap, "heatmap", "All FOVs and rounds",
                                                    self.parts("registration.heatmap"), "3/1"),
                "reg_title": self._example_title(moving),
                "reg_example": self._figure_or_rows(f.registration, "registration", "Registration example",
                                                    self.parts("registration.example"), "3/1")}

    def _centre_reason(self):
        for row in self.rows["fovs"]:
            u = self.named.get(("fovs", "centre_x", f"fov_id={row['fov_id']}"))
            if u is not None:
                return u["reason"]
        return "no_placement"

    def _map(self, ids, part=None):
        rows = self.parts(part) if part else []
        if rows:
            return "".join(unavailable_box(u, ratio="1/1") for u in rows)
        if not self.placed:
            return _derived(self._centre_reason(), "fovs.centre_x")
        return f'<div id="{ids[0]}"></div><div class="lg one" id="{ids[1]}"></div>'

    # Readout -------------------------------------------------------------------------------------------------------

    def readout(self) -> dict:
        totals = {r["level"]: r for r in self.rows["readout_totals"]}
        spots = totals["spots"]
        n_fovs = spots["fovs"]
        bars = []
        for level in READOUT_LEVELS:
            row = totals[level]
            key = f"level={level}"
            label = _LEVELS[level]
            if row["count"] is None:
                u = self.named.get(("readout_totals", "count", key))
                state = "not run" if u is not None and u["reason"] == "not_configured" else None
                dash = self.dash("readout_totals", "count", key) if u is not None else ""
                if ("readout_totals", "fraction_of_spots", key) in self.named:
                    dash += " · " + self.dash("readout_totals", "fraction_of_spots", key)
                after = f" · {state}" if state else ""
                bars.append(f'<div class="lv opt"><span>{_e(label)} · {dash}{after}</span></div>')
                continue
            width = 100.0 if not spots["count"] else min(100.0, 100.0 * row["count"] / spots["count"])
            if row["fraction_of_spots"] is not None:
                share = f"{100 * row['fraction_of_spots']:.1f} %"
            elif ("readout_totals", "fraction_of_spots", key) in self.named:
                share = self.dash("readout_totals", "fraction_of_spots", key)
            else:
                share = ""
            cover = f" · {row['fovs']} of {n_fovs} FOVs" if row["fovs"] < n_fovs else ""
            cls = "lv cell" if level == "reads_in_cells" else "lv"
            text = f"{_e(label)} · {_int(row['count'])} · {share}{_e(cover)}"
            # The bar is the level's fraction of the spots; its label is written over the whole row, not clipped.
            bars.append(f'<div class="{cls}"><i style="width:{width:.2f}%"></i><span>{text}</span></div>')
        summary = self.summary
        if summary["reads_in_cells_equal"] is True:
            check = "= sum of sample.h5ad"
        elif summary["reads_in_cells_equal"] is False:
            check = f"differs from sample.h5ad (sum {_int(summary['h5ad_whole_count_sum'])})"
        else:
            check = ""
        legend = ('<div class="lg"><span class="grp"><span class="it"><i class="sw ok"></i>counted once per FOV that '
                  'saw it</span><span class="it"><i class="sw cell"></i>after the overlaps are removed</span></span>'
                  '</div>')
        totals_html = (f'<div class="bars">{"".join(bars)}</div>'
                       + (f'<div class="bars"><span class="chk">reads assigned to cells {_e(check)}</span></div>'
                          if check else "") + legend)
        distributions = self.parts("readout.distributions")
        if distributions:
            dist = "".join(unavailable_box(u, ratio="3/1") for u in distributions)
        else:
            dist = grid([image(figure, f"distribution:{column}", f"Distribution across FOVs: {label}")
                         for column, label, figure in self.figures.distributions])
        table = self.parts("readout.table")
        return {"read_totals": totals_html, "read_map": self._map(("m-reads", "lg-reads"), "readout.per_fov"),
                "read_distributions": dist,
                "read_table": "".join(unavailable_box(u, ratio="3/1") for u in table) if table else
                '<div id="t-reads"></div>'}

    # Segmentation and assembly ------------------------------------------------------------------------------------

    def assembly(self) -> dict:
        f = self.figures
        rows = self.parts("assembly.examples")
        examples = self._figure_or_rows(f.segmentation, "segmentation", "Segmentation examples", rows, "3/1")
        return {"cells_map": self._map(("m-cells", "lg-cells")), "placement": self._map(("m-place", "lg-place")),
                "seg_examples": examples, "population": '<div id="t-population"></div>'}

    # Sample measurements ----------------------------------------------------------------------------------------------

    def sample(self) -> dict:
        figures = self.figures.sample
        per_cell_rows = self.parts("sample.per_cell")
        if isinstance(figures.per_cell, list):
            per_cell = grid([image(fig, f"per_cell:{fig.tiles[0]['name']}",
                                   f"Per-cell distribution: {fig.tiles[0]['name']}") for fig in figures.per_cell])
            per_cell += "".join(unavailable_box(u, line=True) for u in per_cell_rows)
        else:
            per_cell = "".join(unavailable_box(u, ratio="3/1") for u in per_cell_rows)
        markers = "".join(unavailable_box(u, ratio="16/7") for u in self.parts("sample.markers"))
        zoom_rows = self.parts("sample.marker_zoom")
        if isinstance(figures.markers, list):
            for k, figure in enumerate(figures.markers):
                tiles = []
                zooms = figure.zooms if isinstance(figure.zooms, dict) else {}
                for (role, view), tile in zooms.items():
                    if not isinstance(tile, Unavailable):
                        tiles.append((role, view, tile))
                columns = []
                for role in dict.fromkeys(role for role, _ in zooms):
                    column = "".join(image(tile, f"marker:{k}:{role}:{view}", f"{figure.group.title}: {role} {view}")
                                     for r, view, tile in tiles if r == role)
                    columns.append(f'<div class="figs">{column}</div>')
                hidden = " hidden" if k else ""
                markers += (f'<div class="mg" data-group="{_e(figure.group.name)}"{hidden}><div class="figs">'
                            f'{image(figure.whole, f"marker:{k}:whole", figure.group.title)}</div>'
                            f'<div class="g3 fixed">{"".join(columns)}</div></div>')
            markers += "".join(unavailable_box(u, line=True) for u in zoom_rows)
        else:
            markers += "".join(unavailable_box(u, line=True) for u in zoom_rows)
        mecr_rows = self.parts("sample.mecr")
        mecr = "".join(unavailable_box(u, ratio="16/9") for u in mecr_rows) if mecr_rows else '<div id="t-mecr"></div>'
        along = figures.reads_along_z
        reads = (image(along, "reads_along_z", "Reads along Z") if not isinstance(along, Unavailable) else
                 "".join(unavailable_box(u, ratio="16/9") for u in self.parts("sample.reads_along_z")))
        return {"per_cell": per_cell, "markers": markers, "mecr": mecr, "reads_z": reads}

    # Files ---------------------------------------------------------------------------------------------------------

    @staticmethod
    def _table(rows):
        body = "".join(f"<tr><td>{_e(label)}</td><td>{value}</td></tr>" for label, value in rows)
        return f'<div class="panel tw"><table class="tl"><tbody>{body}</tbody></table></div>'

    def files(self) -> dict:
        c = self.c
        entries = {u["row_key"]: u for u in self.parts("files.input") + self.parts("files.configuration")}
        records = [c.records[f] for f in c.fov_ids if c.records[f] is not None]

        def entry(key, make):
            return unavailable_box(entries[key], inline=True) if key in entries else make()

        def images():
            paths = list(dict.fromkeys(str(i.get("path")) for r in records for i in r.get("inputs") or []
                                       if isinstance(i, dict) and i.get("path")))
            if not paths:
                return '<span class="muted">no input recorded</span>'
            folder = os.path.commonpath(paths) if len(paths) > 1 else os.path.dirname(paths[0])
            return f"<code>{_e(folder)}/</code> ({len(paths)} files)"

        def codebook():
            found = next((r["codebook"] for r in records if isinstance(r.get("codebook"), dict)), None)
            if found is None:
                return '<span class="muted">no codebook recorded</span>'
            digest = found.get("sha256") or ""
            return (f'<code>{_e(found.get("path"))}</code> · SHA-256 '
                    f'<code data-t="{_tip(["SHA-256", digest])}">{_e(digest[:12])}</code>')

        def positions():
            table = (c.placement or {}).get("table")
            if isinstance(table, dict) and table.get("path"):
                return f"<code>{_e(table['path'])}</code>"
            if table:
                return f"<code>{_e(json.dumps(table, sort_keys=True))}</code>"
            return '<span class="muted">no positions file recorded</span>'
        workflow = next((r["workflow_config"] for r in records if isinstance(r.get("workflow_config"), dict)), None)

        def workflow_path():
            if workflow is None:
                return '<span class="muted">not run from the workflow</span>'
            return f"<code>{_e(workflow.get('path'))}</code>"
        settings = (f'<details><summary>report.json config</summary><pre>'
                    f'{_e(json.dumps(self.config_json, indent=2, ensure_ascii=False))}</pre></details>')
        root = c.root

        def place(path, shown):
            exists = path.exists()
            return f"<code>{_e(shown)}</code>" + ("" if exists else ' <span class="muted">not written</span>')
        checkpoints = any(_layout.checkpoint_dir(root, f).is_dir() for f in c.fov_ids)
        report = self.folder
        output = [("Output root", f"<code>{_e(root)}</code>"),
                  ("Checkpoints", "<code>checkpoints/&lt;fov_id&gt;/</code>" +
                   ("" if checkpoints else ' <span class="muted">not written</span>')),
                  ("Placement", place(_layout.stitching_dir(root, c.sample), f"{_layout.STITCHING}/{c.sample}/")),
                  ("Sample export", place(_layout.export_dir(root, c.sample), f"{_layout.EXPORT}/{c.sample}/")),
                  ("Report", f"<code>{_e(self._shown(report / 'report.html'))}</code>"),
                  ("Report tables", f"<code>{_e(self._shown(report / 'tables'))}/</code>")]
        return {"files_input": self._table([("Images", entry("entry=images", images)),
                                            ("Codebook", entry("entry=codebook", codebook)),
                                            ("Tile positions", entry("entry=positions", positions))]),
                "files_configuration": self._table([
                    ("Workflow", entry("entry=workflow", workflow_path)),
                    ("Hash", "" if "entry=workflow" in entries or workflow is None else
                     f"<code>{_e(workflow.get('sha256'))}</code>"),
                    ("Report settings", settings)]),
                "files_output": self._table(output)}

    def _shown(self, path):
        try:
            return path.resolve().relative_to(self.c.root.resolve()).as_posix()
        except ValueError:
            return str(path)


def identity_facts(collection) -> tuple[dict, list]:
    """The facts of the Summary that the run records and the export hold (configuration check, acquisition with the
    Z size, software, run time) and the rows of ``unavailable.csv`` of the part ``summary.identity``: one per fact
    that is unavailable (``entry=<fact>``), and ``entry=z_size``: ``not_recorded`` when no record holds a grid and
    no FOV has a reference image, or the reason no reference image could be read (``input_changed``)."""
    c = collection
    pattern = getattr(c, "reference_image", None)
    planes = z_size(c.export, c.placement, lambda: reference_image_z_size(c.inputs, c.root, c.fov_ids, pattern))
    unread = planes if isinstance(planes, Unavailable) else None
    planes = None if unread is not None else planes
    facts = {"acquisition": acquisition(c.records, planes), "z_size": planes,
             "configuration": configuration_check(c.records),
             "software": software(c.records, _get(c.assembly, "software")),
             "run_time": run_time(c.records)}
    rows = []
    for name in ("acquisition", "z_size", "configuration", "software", "run_time"):
        value = facts[name]
        if name == "z_size":
            if value is None:
                rows.append(("summary.identity", unread or Unavailable(
                    "not_recorded", "no record holds a grid (export.json fusion.fovs or assembly.fovs, "
                                    "placement.json inputs) and no FOV has a reference image "
                                    f"({pattern or _layout.REFERENCE_IMAGE})"), "entry=z_size"))
        elif isinstance(value, Unavailable):
            rows.append(("summary.identity", value, f"entry={name}"))
    return facts, rows


def build_page(collection, tables, frames, unavailable, summary, identity, generated_at, config_json,
               folder) -> tuple:
    """The text of ``report.html`` and its static figures (:class:`PageFigures`)."""
    figures = PageFigures(collection, tables)
    page = _Page(collection, tables, frames, unavailable, summary, identity, generated_at, config_json, folder,
                 figures)
    return fill(template_text(), page.values()), figures
