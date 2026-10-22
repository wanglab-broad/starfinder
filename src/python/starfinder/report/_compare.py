"""The comparison page of a dataset (docs/report-contract.md, "The comparison entry", "``samples.csv``";
docs/report-algorithms.md, "The comparison page").

:func:`compare_samples` reads finished sample reports only: each sample's
``<report_root>/<sample>/report.json``, checked with :func:`._schema.read_report` against
every file it lists, and its ``tables/fovs.csv``. It reads no run record, placement or
export and recomputes nothing a sample report holds: each column of ``samples.csv`` copies a
key of the sample's ``report.json``, and the three box plots draw the columns ``spots``,
``matched_fraction`` and ``n_cells_retained`` of its ``fovs.csv``. A sample whose folder or
``report.json`` is missing is a row with ``report`` ``missing`` and every other column null.
``index.html`` is the template ``templates/index.html`` filled as ``report.html`` is
(``_page.fill``), with the CSS of ``report.html``; it embeds ``samples.csv`` and the per-FOV
values of the box plots as one data block. Everything is read and checked before
``index.html`` and then ``samples.csv`` replace earlier ones under the report root; nothing
is written under a sample's folder.
"""
from __future__ import annotations

import hashlib
import html
import json
import re
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import quote

import pandas as pd

from ._schema import REPORT_FILE, SAMPLES_DTYPES, TABLE_DIR, ComparisonRecord, read_report

INDEX_FILE = "index.html"
SAMPLES_FILE = "samples.csv"
INDEX_TEMPLATE = "index.html"
#: The slots of ``templates/index.html``, in document order; each is filled exactly once.
INDEX_SLOTS = ("title", "style", "heading", "samples", "box_spots", "box_matched", "box_cells", "data")
#: The keys of ``report.json`` ``summary`` that ``samples.csv`` copies, in column order.
SUMMARY_COLUMNS = ("fovs_expected", "fovs_succeeded", "spots_median_per_fov", "matched_fraction", "counted_cells",
                   "cells_median_per_fov")
#: The three box plots: the slot, the ``fovs.csv`` column, the block's title and the axis label.
BOX_PLOTS = (("box_spots", "spots", "Spots per FOV", "spots per FOV"),
             ("box_matched", "matched_fraction", "Fraction matched to the codebook",
              "fraction matched to the codebook"),
             ("box_cells", "n_cells_retained", "Cells per FOV", "cells per FOV"))
#: The per-FOV columns of ``fovs.csv`` the page embeds for each present sample.
FOV_COLUMNS = ("fov_id",) + tuple(column for _, column, _, _ in BOX_PLOTS)
#: The label of an empty slot: a missing report, or the reason the sample's ``unavailable`` gives the column's first
#: unavailable cell (the reasons of a cell count read from the export are those of a sample without cells).
MISSING_LABEL = "missing"
SLOT_LABELS = {"no_run_record": "no run record", "no_export": "no cells", "part_not_written": "no cells",
               "no_segmentation": "no cells"}
_COLUMN_HEADS = ("Sample", "Report", "Generated", "Route", "FOVs expected", "FOVs succeeded", "Median spots per FOV",
                 "Matched fraction", "Counted cells", "Median cells per FOV")
_SEPARATORS = re.compile(r"[/\\]")


def _check_samples(samples) -> list[str]:
    """``samples`` is a nonempty sequence of unique sample names, each a nonempty string naming one folder."""
    if isinstance(samples, (str, bytes)) or not isinstance(samples, Sequence):
        raise ValueError(f"samples must be a sequence of sample names; got {samples!r}")
    samples = list(samples)
    if not samples:
        raise ValueError("samples must name at least one sample")
    for sample in samples:
        if not isinstance(sample, str) or not sample or sample in (".", "..") or _SEPARATORS.search(sample):
            raise ValueError(f"each sample must be a nonempty name of one folder; got {sample!r}")
    repeated = sorted({s for s in samples if samples.count(s) > 1})
    if repeated:
        raise ValueError(f"samples must be unique; repeated: {repeated}")
    return samples


def _read_sample(folder: Path):
    """A present sample's ``report.json`` (checked against every file it lists), its SHA-256 and its ``fovs.csv``."""
    from starfinder._records import _read_table
    path = folder / REPORT_FILE
    data = path.read_bytes()
    try:
        record = read_report(folder)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"{path} is not a report record: {error}") from error
    if record.get("summary") is None or any(key not in record["summary"] for key in SUMMARY_COLUMNS) or \
            any(key not in record for key in ("generated_at", "route", "unavailable")):
        raise ValueError(f"{path} lacks a key of samples.csv: generated_at, route, unavailable or summary "
                         f"{list(SUMMARY_COLUMNS)}")
    entry = next((e for e in record.get("files") or [] if e.get("path") == f"{TABLE_DIR}/fovs.csv"), None)
    if entry is None or "dtypes" not in entry:
        raise ValueError(f"{path} lists no {TABLE_DIR}/fovs.csv with its dtypes")
    fovs = _read_table(folder / entry["path"], entry["dtypes"])
    missing = [c for c in FOV_COLUMNS if c not in fovs.columns]
    if missing:
        raise ValueError(f"{folder / entry['path']} has no columns {missing}")
    return record, hashlib.sha256(data).hexdigest(), fovs[list(FOV_COLUMNS)]


def _slot_label(record, column) -> str:
    """The label of a present sample's empty slot: the reason of the column's first cell its ``unavailable`` names."""
    from ._page import REASON_TEXTS
    reason = next((u.get("reason") for u in record["unavailable"]
                   if u.get("table") == "fovs" and u.get("column") == column), None)
    if reason is None:
        return "no value"
    return SLOT_LABELS.get(reason, REASON_TEXTS.get(reason, reason))


def _e(value) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _cell(value, show, tip=None) -> str:
    if value is None:
        return f'<td title="{_e(tip)}">—</td>' if tip else "<td></td>"
    return f"<td>{_e(show(value))}</td>"


def _samples_table(rows, records) -> str:
    """The samples table, from the rows of ``samples.csv``: a present sample links its ``report.html``; a missing
    sample says "missing" and its other cells are blank; a key the sample's report holds as unavailable is "—" with
    its reason in the tooltip."""
    from ._page import REASON_TEXTS, _int, _number
    shows = {"fovs_expected": _int, "fovs_succeeded": _int, "spots_median_per_fov": _number,
             "matched_fraction": lambda v: f"{100 * v:.1f} %", "counted_cells": _int, "cells_median_per_fov": _number}
    body = []
    for row in rows:
        sample = row["sample"]
        if row["report"] != "present":
            body.append(f'<tr><td>{_e(sample)}</td><td class="miss">missing</td>' + "<td></td>" * 8 + "</tr>")
            continue
        named = {u["column"]: u for u in records[sample]["unavailable"] if u.get("table") == "summary"}
        cells = [f'<td><a href="{_e(quote(sample, safe=""))}/report.html">{_e(sample)}</a></td>', "<td>present</td>",
                 f"<td><code>{_e(row['generated_at'])}</code></td>",
                 f"<td>{_e(row['route'] if row['route'] is not None else 'no export')}</td>"]
        for key in SUMMARY_COLUMNS:
            u = named.get(key)
            tip = None if u is None else f"Unavailable: {REASON_TEXTS.get(u['reason'], u['reason'])}"
            cells.append(_cell(row[key], shows[key], tip))
        body.append("<tr>" + "".join(cells) + "</tr>")
    head = "".join(f"<th>{_e(h)}</th>" for h in _COLUMN_HEADS)
    return (f'<div class="panel" id="t-samples"><div class="tw"><table><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div></div>')


def _data_block(rows, fovs) -> str:
    """The JSON text of the data block: ``samples`` (the rows of ``samples.csv``) and ``fovs`` (per present sample,
    its rows of the per-FOV columns), written as the report's data block is."""
    data = {"samples": rows, "fovs": fovs}
    text = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    return text.replace("</", "<\\/").replace("<!--", "\\u003c!--")


def compare_samples(report_root, samples: Sequence[str], *, dataset_id: str | None = None) -> ComparisonRecord:
    """Write the comparison page ``index.html`` and ``samples.csv`` of finished sample reports.

    Parameters
    ----------
    report_root : str or Path
        The folder of the sample reports, by convention ``<OUTPUT>/report/``: each sample's report is
        ``<report_root>/<sample>/``.
    samples : Sequence[str]
        The samples, nonempty and unique, in the order of the table and the box plots.
    dataset_id : str or None
        The dataset the page names in its heading.

    Returns
    -------
    ComparisonRecord
        The report root and the record of what was written.

    Raises
    ------
    ValueError
        ``samples`` is empty, repeats a sample or holds a name that is not one folder's; or a present report whose
        ``report.json`` is not a version 1 report record, or one of whose listed files is missing or differs from its
        recorded SHA-256 (naming the file). Nothing is written then.
    """
    from starfinder._records import _write_table, atomic_file
    from ._figures import sample_box_figure
    from ._page import fill, image, table_rows, template_text
    samples = _check_samples(samples)
    root = Path(report_root)
    rows, records, fovs = [], {}, {}
    for sample in samples:
        folder = root / sample
        row = {column: None for column in SAMPLES_DTYPES} | {"sample": sample, "report": "missing"}
        if (folder / REPORT_FILE).is_file():
            record, digest, table = _read_sample(folder)
            records[sample], fovs[sample] = record, table_rows(table)
            row |= {"report": "present", "report_sha256": digest, "generated_at": record["generated_at"],
                    "route": record["route"], **{key: record["summary"][key] for key in SUMMARY_COLUMNS}}
        rows.append(row)
    frame = pd.DataFrame({c: pd.Series([row[c] for row in rows], dtype=object).astype(d)
                          for c, d in SAMPLES_DTYPES.items()}, columns=list(SAMPLES_DTYPES))
    rows = table_rows(frame)

    values, box_plots = {}, {}
    for slot, column, title, label in BOX_PLOTS:
        slots = []
        for sample in samples:
            found = [r[column] for r in fovs[sample] if r[column] is not None] if sample in fovs else []
            slots.append({"sample": sample, "values": found,
                          "label": None if found else (_slot_label(records[sample], column) if sample in records
                                                       else MISSING_LABEL)})
        figure = sample_box_figure(column, label, slots)
        box_plots[column] = figure.tiles[0]["slots"]
        alt = f"{title} by sample: " + "; ".join(
            f"{s['sample']}, {s['fovs']} FOVs" if s["fovs"] else f"{s['sample']}, {s['label']}"
            for s in box_plots[column])
        values[slot] = image(figure, f"box:{column}", alt)
    style = re.search(r"<style>.*?</style>", template_text(), re.S).group(0)
    named = f"<code>{_e(dataset_id)}</code> · " if dataset_id else ""
    values |= {"title": _e("Sample comparison" + (f" · {dataset_id}" if dataset_id else "")),
               "heading": f'<h1>Sample comparison</h1><span class="sub">{named}{len(samples)} '
                          f'sample{"" if len(samples) == 1 else "s"}</span>',
               "style": style, "samples": _samples_table(rows, records), "data": _data_block(rows, fovs)}
    page = fill(template_text(INDEX_TEMPLATE), values, INDEX_SLOTS).encode("utf-8")

    # Written last, in this order, so that samples.csv (the workflow's declared output) implies index.html.
    with atomic_file(root / INDEX_FILE) as tmp:
        tmp.write_bytes(page)
    file_name, dtypes = _write_table(frame, root, Path(SAMPLES_FILE).stem, "csv")
    table = (root / file_name).read_bytes()
    files = [{"path": INDEX_FILE, "sha256": hashlib.sha256(page).hexdigest(), "bytes": len(page)},
             {"path": file_name, "sha256": hashlib.sha256(table).hexdigest(), "bytes": len(table), "dtypes": dtypes}]
    return ComparisonRecord(root, {"samples": rows, "files": files, "box_plots": box_plots})
