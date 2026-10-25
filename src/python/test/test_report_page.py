"""The report page and the entry ``Dataset.report`` (W-456).

Checks of docs/report-algorithms.md ("Validation design") on the pages of the hand-written
fixtures ``report_records`` (every variant of RC1 to RC13) and ``report_cells`` (and its
variants): the data block equals the saved tables (PG1), every section and block is present
in page order (PG2), every unavailable part and named null cell carries its reason (PG3), the
page loads nothing and has a light theme only (PG4), its size is within the budget and every
figure has its size (PG5), no table or record key holds a verdict (PG6), and the entry
``Dataset.report`` checks before anything is written, writes its default folder, reads only
the places of "Where each record is looked for" and calls no processing stage (PG7). The
section and block ids, the reason texts and the figure sizes are written here from the page;
expected values never come from the report's own output.

The page is checked by its text, not in a browser. The cells and map tooltips that the inline
script draws from the data block are checked by running that script under ``node`` with a
minimal document stand-in, when ``node`` is installed (the test skips without it).
"""
from __future__ import annotations

import base64
import html
import io
import json
import os
import re
import shutil
import stat
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd
import pytest

from starfinder.report import MarkerGroup, ReportConfig, ReportRecord, load_report
from starfinder.report import _page as page_module

from . import hash_oracles
from .report_fixtures import CASES, DATASET, FOVS, IDS, SAMPLE, SAMPLE_ID, report_cells, report_records
from .report_sizes import (DISTRIBUTION_HALF, DISTRIBUTION_THIRD, EXAMPLE_TILE, HEATMAP_WIDTH, INTENSITY_SIZE,
                           MAX_ROWS, READS_Z_WIDTH, ROW_PX, THIRD, TIME_WIDTH, grid_slot, preprocessing_tile,
                           time_height_at_least, tiles_width)
from .test_report_collection import (CHECKS, CONTRACT_COLUMNS, CONTRACT_RECORD_KEYS, CONTRACT_SUMMARY_KEYS,
                                     anndata_or_skip, snapshot)

pytestmark = [pytest.mark.report, pytest.mark.validation]

# --- Written from docs/report-algorithms.md ("Layout of the page", "How an unavailable part is shown") ----------

SECTIONS = [("summary", "Summary", ["b-facts", "b-key-numbers", "b-problematic"]),
            ("run", "Workflow overview", ["b-chart", "b-fov-status", "b-time"]),
            ("prep", "Preprocessing", ["b-prep-example", "b-prep-intensity"]),
            ("reg", "Registration", ["b-reg-maps", "b-reg-heatmap", "b-reg-example"]),
            ("reads", "Readout", ["b-read-totals", "b-read-map", "b-read-distributions", "b-read-table"]),
            ("cells", "Segmentation and assembly", ["b-cells-map", "b-placement", "b-seg-examples",
                                                    "b-population"]),
            ("sample", "Sample measurements", ["b-per-cell", "b-markers", "b-mecr", "b-reads-z"]),
            ("files", "Files", ["b-files-input", "b-files-configuration", "b-files-output"])]
# The part each block draws (docs/report-contract.md, "Parts, their inputs and their reasons").
BLOCK_PARTS = {"b-facts": "summary.identity", "b-key-numbers": "summary.key_numbers",
               "b-problematic": "summary.problematic_fovs", "b-chart": "workflow.steps",
               "b-fov-status": "workflow.fov_status", "b-time": "workflow.time",
               "b-prep-example": "preprocessing.example", "b-prep-intensity": "preprocessing.intensity",
               "b-reg-maps": "registration.maps", "b-reg-heatmap": "registration.heatmap",
               "b-reg-example": "registration.example", "b-read-totals": "readout.totals",
               "b-read-map": "readout.per_fov", "b-read-distributions": "readout.distributions",
               "b-read-table": "readout.table", "b-cells-map": "assembly.cells_per_fov",
               "b-placement": "assembly.placement", "b-seg-examples": "assembly.examples",
               "b-population": "assembly.accounting", "b-per-cell": "sample.per_cell",
               "b-markers": "sample.markers sample.marker_zoom", "b-mecr": "sample.mecr",
               "b-reads-z": "sample.reads_along_z", "b-files-input": "files.input",
               "b-files-configuration": "files.configuration", "b-files-output": "files.output"}
REASON_TEXTS = {
    "no_run_record": "no run record for this FOV", "run_not_finished": "the run has not finished",
    "step_failed": "the run failed before this step", "not_configured": "not part of the run's configuration",
    "not_recorded": "the run did not record this value", "no_placement": "no placement for this sample",
    "fov_not_placed": "this FOV was not placed", "no_export": "no export for this sample",
    "part_not_written": "the export part that writes this was not written",
    "no_segmentation": "the sample has no cells", "no_markers": "no marker pairs supplied",
    "no_stack": "the export has no 3D stack", "input_missing": "an input file is missing",
    "input_changed": "an input file differs from its record",
    "field_not_saved": "the registration field was not saved",
    "replay_differs": "the replayed preprocessing differs from the record",
    "undefined": "undefined: the denominator is zero"}
# "The data embedded in the page": the tables of the block, in order; cells and intensity are not embedded.
EMBEDDED = ["fovs", "registration", "steps", "readout_totals", "population", "problematic_fovs", "mecr",
            "reads_along_z", "unavailable"]
# The key numbers of the Summary: the summary keys the page shows.
KEY_NUMBERS = ["fovs_succeeded", "spots_median_per_fov", "matched_fraction", "counted_cells",
               "reads_per_cell_median", "genes_per_cell_median"]
KIB, MIB = 1024, 1024 * 1024
TOL = 1e-9



class Page:
    """A collected case: the case, the folder, its report.json, its tables and the page's text."""

    def __init__(self, case, folder, record):
        self.case, self.folder, self.record = case, Path(folder), record
        self.text = (self.folder / "report.html").read_text(encoding="utf-8")
        self.tables = {Path(e["path"]).stem: hash_oracles.read_table(self.folder / e["path"], e["dtypes"])
                       for e in record["files"] if "dtypes" in e}
        self.dtypes = {Path(e["path"]).stem: e["dtypes"] for e in record["files"] if "dtypes" in e}

    @property
    def unavailable(self):
        return [{k: (None if pd.isna(v) else v) for k, v in row.items()}
                for row in self.tables["unavailable"].to_dict("records")]

    def data(self):
        blocks = re.findall(r'<script type="application/json" id="report-data">(.*?)</script>', self.text, re.S)
        assert len(blocks) == 1
        return blocks[0], json.loads(blocks[0])


@pytest.fixture(scope="module")
def pages(tmp_path_factory):
    """The pages of every case, each collected once for the module."""
    made = {}

    def page(maker, variant):
        if (maker, variant) not in made:
            if maker == "report_cells":
                anndata_or_skip()
            root = tmp_path_factory.mktemp(f"{maker}-{variant}") / "root"
            case = (report_cells if maker == "report_cells" else report_records)(root, variant)
            folder, record = case.collect()
            made[(maker, variant)] = Page(case, folder, record)
        return made[(maker, variant)]
    return page


# --- An HTML reader of the elements the checks name -----------------------------------------------------------

class Elements(HTMLParser):
    """The elements with a ``data-reason`` (placeholders), a ``data-cell`` (cells the filler writes), an ``id`` and
    every ``<img>``: their attributes and their text."""

    VOID = {"img", "br", "meta", "input", "hr"}

    def __init__(self, text):
        super().__init__(convert_charrefs=True)
        self.found, self.ids, self.images, self.open = [], [], [], []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.append(attrs["id"])
        if tag == "img":
            self.images.append(attrs)
        if tag in self.VOID:
            return
        for entry in self.open:
            entry["depth"] += 1
        if "data-reason" in attrs or "data-cell" in attrs or tag == "h2":
            entry = {"tag": tag, "attrs": attrs, "text": "", "depth": 1}
            self.found.append(entry)
            self.open.append(entry)

    def handle_endtag(self, tag):
        if tag in self.VOID:
            return
        for entry in list(self.open):
            entry["depth"] -= 1
            if entry["depth"] == 0:
                self.open.remove(entry)

    def handle_data(self, data):
        for entry in self.open:
            entry["text"] += data


# --- PG1: the embedded data equal the saved tables -----------------------------------------------------------

@pytest.mark.parametrize("maker,variant", CASES, ids=IDS)
def test_pg1_embedded_data_equal_the_tables(pages, maker, variant):
    page = pages(maker, variant)
    text, data = page.data()
    assert "</" not in text
    assert list(data) == EMBEDDED
    for name in EMBEDDED:
        dtypes = page.dtypes[name]
        rows = data[name]
        assert all(list(row) == list(dtypes) for row in rows), name
        embedded = pd.DataFrame(rows, columns=list(dtypes), dtype=object).astype(dtypes) if rows else \
            pd.DataFrame({c: pd.Series([], dtype=object) for c in dtypes}).astype(dtypes)
        pd.testing.assert_frame_equal(embedded, page.tables[name], check_exact=False, rtol=0, atol=TOL)
        # Every value is the table's, null exactly where the table's cell is missing.
        for row, saved in zip(rows, page.tables[name].to_dict("records")):
            for column in dtypes:
                assert (row[column] is None) == bool(pd.isna(saved[column])), (name, column)
    assert "cells" not in data and "intensity" not in data


# --- PG2: every section and block, in page order ------------------------------------------------------------

@pytest.mark.parametrize("maker,variant", CASES, ids=IDS)
def test_pg2_sections_and_blocks(pages, maker, variant):
    page = pages(maker, variant)
    elements = Elements(page.text)
    expected = [i for section, _, blocks in SECTIONS for i in [section] + blocks]
    found = [i for i in elements.ids if i in expected]
    assert found == expected
    assert [e["text"] for e in elements.found if e["tag"] == "h2"] == [name for _, name, _ in SECTIONS]
    for block, part in BLOCK_PARTS.items():
        assert re.search(rf'id="{block}"[^>]*data-part="{re.escape(part)}"', page.text), block
    # The nav lists the eight sections, each by its id.
    assert re.findall(r'<a href="#([a-z]+)">', page.text) == [s for s, _, _ in SECTIONS]


# --- PG3: every unavailable part carries its reason --------------------------------------------------------------

def block_span(text, block):
    """The text of a block: from its id to the next block's id of its section, or the end of its section."""
    order = [b for _, _, blocks in SECTIONS for b in blocks]
    start = text.index(f'id="{block}"')
    end = text.index("</section>", start)
    following = order[order.index(block) + 1:]
    for other in following:
        at = text.find(f'id="{other}"', start)
        if 0 <= at < end:
            end = at
            break
    return start, end


def reason_constants(text):
    script = re.search(r"<script>\n(.*?)</script>", text, re.S).group(1)
    return json.loads(re.search(r"var REASONS = (\{.*?\});\n", script).group(1))


@pytest.mark.parametrize("maker,variant", CASES, ids=IDS)
def test_pg3_every_unavailable_part_carries_its_reason(pages, maker, variant):
    page = pages(maker, variant)
    assert reason_constants(page.text) == REASON_TEXTS
    assert page_module.REASON_TEXTS == REASON_TEXTS
    assert len(re.findall(r"application/json", page.text)) == 1
    elements = Elements(page.text)
    placeholders = [e for e in elements.found if "data-reason" in e["attrs"]]
    rows = [u for u in page.unavailable if u["table"] is None]
    assert rows, "every case has a part without a table cell"
    # One placeholder per row without a table, in the part's block, with "Unavailable:" and the reason's text.
    for row in rows:
        mine = [e for e in placeholders if e["attrs"]["data-part"] == row["part"]
                and e["attrs"]["data-key"] == (row["row_key"] or "")]
        assert len(mine) == 1, row
        element = mine[0]
        assert element["attrs"]["data-reason"] == row["reason"]
        assert element["text"].startswith(f"Unavailable: {REASON_TEXTS[row['reason']]}"), element
        if row["detail"]:
            assert element["text"].endswith(row["detail"])
    assert len(placeholders) == len(rows)
    blocks = {part: block for block, parts in BLOCK_PARTS.items() for part in parts.split()}
    for row in rows:
        block = blocks.get(row["part"], "b-problematic")  # rule:<rule> is listed under the Problematic FOVs
        start, end = block_span(page.text, block)
        assert f'data-part="{html.escape(row["part"])}" data-key="{html.escape(row["row_key"] or "")}"' in \
            page.text[start:end], (row, block)
    # The named null cells the filler writes (the key numbers, the readout totals): "—" and the reason's text in the
    # tooltip data. The cells the script draws are checked by test_pg3_cells_drawn_by_the_script.
    cells = {e["attrs"]["data-cell"]: e for e in elements.found if "data-cell" in e["attrs"]}
    shown = [r for r in page.unavailable if r["table"] == "readout_totals" or
             (r["table"] == "summary" and r["column"] in KEY_NUMBERS)]
    for row in shown:
        element = cells[f"{row['table']}|{row['column']}|{row['row_key'] or ''}"]
        assert element["text"] == "—"
        assert element["attrs"]["data-t"].split("|")[0] == f"Unavailable: {REASON_TEXTS[row['reason']]}"
    assert set(cells) == {f"{r['table']}|{r['column']}|{r['row_key'] or ''}" for r in shown}
    # The time per step names in its text ("not recorded: …") each step of a scope that has no time and whose
    # missing times are named by rows; a step that did not run (n/a) is not drawn.
    steps = page.tables["steps"]
    keys = {r["row_key"] for r in page.unavailable if r["table"] == "steps" and r["column"] == "seconds"}
    alt = next(i["alt"] for i in elements.images if i.get("data-figure") == "time")
    listed = alt.split("; not recorded: ")[1].split(", ") if "; not recorded: " in alt else []
    expected = []
    for (scope, name), group in steps.groupby(["scope", "name"], sort=False):
        named_rows = [f"scope={s};fov_id={'' if pd.isna(f) else f};index={i}" in keys
                      for s, f, i in group[["scope", "fov_id", "index"]].itertuples(index=False)]
        if group["seconds"].isna().all() and any(named_rows):
            expected.append(name)
    assert sorted(set(listed)) == sorted(set(expected))


# The script of a page under node, with a minimal document stand-in: each element's HTML as the script writes it,
# after the first drawing and after each button it made is pressed once.
SHIM = r"""
const fs = require('fs');
const page = fs.readFileSync(process.argv[2], 'utf8');
const data = page.match(/<script type="application\/json" id="report-data">([\s\S]*?)<\/script>/)[1];
const code = page.match(/<script>\n([\s\S]*?)<\/script>/)[1];
const written = {}, first = {}, elements = {};
function attrsOf(html, tag) { return [...html.matchAll(new RegExp('<' + tag + '\\b([^>]*)>', 'g'))].map(m => m[1]); }
function El(id) {
  this.id = id; this._html = ''; this.hidden = false; this.style = {}; this.className = ''; this.attributes = {};
  this.textContent = ''; this.offsetWidth = 100; this.parentNode = null; this.classList = {toggle() {}};
}
Object.defineProperty(El.prototype, 'innerHTML', {get() { return this._html; }, set(v) {
  this._html = String(v); if (this.id) { written[this.id] = this._html; if (!(this.id in first)) first[this.id] = ''; }
}});
El.prototype.appendChild = function (c) { this.innerHTML = this._html + '<div class="' + c.className + '">' + c.innerHTML + '</div>'; return c; };
El.prototype.getAttribute = function (n) { return this.attributes[n] == null ? null : this.attributes[n]; };
El.prototype.setAttribute = function (n, v) { this.attributes[n] = String(v); };
El.prototype.querySelectorAll = function (s) {
  if (s === '.pager button' && !/class="pager"/.test(this._html)) return [];
  if (s !== '.pager button' && s !== 'button') throw new Error(s);
  this._buttons = attrsOf(this._html, 'button').map(a => { const b = new El(null); const m = a.match(/data-mode="([^"]*)"/); if (m) b.attributes['data-mode'] = m[1]; return b; });
  return this._buttons;
};
function selectValue(id) {
  for (const h of Object.values(written)) { const m = h.match(new RegExp('<select id="' + id + '">\\s*<option value="([^"]*)"')); if (m) return m[1]; }
  return null;
}
const document = {
  getElementById(id) {
    if (id === 'report-data') return {textContent: data};
    if (!(id in elements)) {
      const p = new RegExp(' id="' + id + '"');
      if (!p.test(page) && !Object.values(written).some(h => p.test(h))) return null;
      const e = new El(id); const v = selectValue(id); if (v !== null) e.value = v; elements[id] = e;
    }
    return elements[id];
  },
  createElement() { return new El(null); }, addEventListener() {},
  querySelectorAll(s) {
    if (s === '.mg') return attrsOf(page, 'div').filter(a => /class="mg"/.test(a)).map(a => { const e = new El(null); e.attributes['data-group'] = (a.match(/data-group="([^"]*)"/) || [])[1]; return e; });
    if (s === '#nav a') return [...page.matchAll(/<a href="(#[^"]+)"/g)].map(m => { const e = new El(null); e.attributes.href = m[1]; return e; });
    if (s === '.sec') return [];
    throw new Error(s);
  },
};
new Function('document', 'window', code)(document, {innerWidth: 1200});
Object.assign(first, written);
for (const e of Object.values(elements)) for (const b of (e._buttons || []).slice()) if (b.onclick) b.onclick();
process.stdout.write(JSON.stringify({first: first, last: written}));
"""


def run_script(page, tmp_path):
    node = shutil.which("node") or shutil.which("nodejs")
    if node is None:
        pytest.skip("node is not installed: the cells the page's script draws are checked by running the script")
    shim = tmp_path / "shim.js"
    shim.write_text(SHIM)
    done = subprocess.run([node, str(shim), str(page.folder / "report.html")], capture_output=True, text=True,
                          check=True, timeout=60)
    return json.loads(done.stdout)


def table_cells(markup):
    """The rows of a table the script wrote: per row, its cells as (attributes, text)."""
    rows = []
    for row in re.findall(r"<tr>(.*?)</tr>", markup.split("<tbody>", 1)[1], re.S):
        rows.append([(dict(re.findall(r'([a-z-]+)="([^"]*)"', attrs)), html.unescape(text))
                     for attrs, text in re.findall(r"<td([^>]*)>(.*?)</td>", row, re.S)])
    return rows


def map_tips(markup):
    """FOV → the tooltip lines of its rectangle in a map the script drew."""
    tips = {}
    for attrs in re.findall(r"<rect ([^>]*)>", markup):
        lines = html.unescape(re.search(r'data-t="([^"]*)"', attrs).group(1)).split("|")
        tips[lines[0]] = lines[1:]
    return tips


@pytest.mark.parametrize("maker,variant", CASES, ids=IDS)
def test_pg3_cells_drawn_by_the_script(pages, tmp_path, maker, variant):
    page = pages(maker, variant)
    drawn = run_script(page, tmp_path)
    named = {(r["table"], r["column"], r["row_key"]): r for r in page.unavailable if r["table"] is not None}

    def check(cell, table, column, key):
        attrs, text = cell
        if (table, column, key) in named:
            reason = named[(table, column, key)]["reason"]
            assert text == "—", (table, column, key)
            assert html.unescape(attrs["data-t"]).split("|")[0] == f"Unavailable: {REASON_TEXTS[reason]}"
        else:
            assert text != "—" and "data-t" not in attrs, (table, column, key)
    fovs = page.tables["fovs"]
    merged = bool(fovs["merged_by_deduplication"].notna().any())
    columns = ["spots", "correct_form", "matched"] + (["merged_by_deduplication"] if merged else []) + [
        "passed_filter", "matched_fraction"]
    if "t-reads" in drawn["first"]:
        rows = table_cells(drawn["first"]["t-reads"])
        assert [r[0][1] for r in rows] == list(fovs["fov_id"])
        for row, fov in zip(rows, fovs["fov_id"]):
            for cell, column in zip(row[1:], columns):
                check(cell, "fovs", column, f"fov_id={fov}")
    rows = table_cells(drawn["first"]["t-population"])
    population = page.tables["population"]
    shown = [p for p, c, m in population[["population", "cells", "molecules"]].itertuples(index=False)
             if not (pd.isna(c) and ("population", "cells", f"population={p}") not in named
                     and pd.isna(m) and ("population", "molecules", f"population={p}") not in named)]
    assert len(rows) == len(shown)
    for row, name in zip(rows, shown):
        for cell, column in zip(row[1:], ("cells", "molecules")):
            check(cell, "population", column, f"population={name}")
    if "t-mecr" in drawn["first"]:
        rows = table_cells(drawn["first"]["t-mecr"])
        for row, (a, b) in zip(rows, page.tables["mecr"][["gene_a", "gene_b"]].itertuples(index=False)):
            for cell, column in zip(row[1:], ("cells_a", "cells_b", "both", "either", "mecr")):
                check(cell, "mecr", column, f"gene_a={a};gene_b={b}")
    # The maps: a FOV whose value is named carries the reason in its tooltip; on a map of a value of the run records
    # a failed, running or not-run FOV is drawn as its category, on a map of the export's values it is not.
    status = dict(zip(fovs["fov_id"], fovs["status"]))
    ran = dict(zip(fovs["fov_id"], fovs["run_record"]))
    for host, column, from_run in (("m-reads", "spots", True), ("m-cells", "n_cells", False)):
        if host not in drawn["first"]:
            continue
        for fov, lines in map_tips(drawn["first"][host]).items():
            key = ("fovs", column, f"fov_id={fov}")
            if from_run and (not ran[fov] or status[fov] in ("failed", "interrupted", "running")):
                continue
            if key in named:
                assert lines[0] == f"Unavailable: {REASON_TEXTS[named[key]['reason']]}", (host, fov)
            else:
                assert not any(line.startswith("Unavailable") for line in lines), (host, fov)
    if "reg-maps" in drawn["first"]:
        registration = page.tables["registration"]
        tips = re.findall(r'<div class="cap">([^<]*)</div><svg[^>]*>(.*?)</svg>', drawn["first"]["reg-maps"], re.S)
        for caption, markup in tips:
            name = caption.split(" · ")[0]
            for fov, lines in map_tips(markup).items():
                if not ran[fov] or status[fov] in ("failed", "interrupted", "running"):
                    continue
                key = f"fov_id={fov};round={name}"
                reasons = [named[("registration", f"residual_{a}", key)]["reason"] for a in "xyz"
                           if ("registration", f"residual_{a}", key) in named]
                if reasons and not registration.empty:
                    assert lines[1] == f"Unavailable: {REASON_TEXTS[reasons[0]]}", (fov, name)


# --- PG4: no external resource and a light theme ----------------------------------------------------------------

@pytest.mark.parametrize("maker,variant", CASES, ids=IDS)
def test_pg4_no_external_resource(pages, maker, variant):
    text = pages(maker, variant).text
    assert not re.search(r"https?:", text, re.I)
    assert not re.search(r'(src|href)="//', text)
    for word in ("<link", "<iframe", "<object", "@import", "prefers-color-scheme"):
        assert word not in text.lower(), word
    sources = re.findall(r'\ssrc="([^"]*)"', text)
    assert sources and all(s.startswith("data:") for s in sources)
    assert all(text[m.end():].startswith("data:") for m in re.finditer(r"url\(", text))
    assert all(h.startswith("#") for h in re.findall(r'\shref="([^"]*)"', text))
    assert len(re.findall(r"<script\b", text)) == 2 and not re.search(r"<script[^>]*\ssrc=", text)


# --- PG5: the size budget and the figure sizes ------------------------------------------------------------------

def heatmap_rows_at_least(image, rows):
    """A heatmap of ``rows`` FOVs (at most 64 drawn) has at least 14 px as shown per row."""
    return image.size[1] / 2 >= ROW_PX * min(rows, MAX_ROWS)


def check_figures(page, extra=None):
    """Every static figure decodes and has the pixel size of "Figure sizes"; returns the names of the figures."""
    from PIL import Image
    n_fovs = len(page.case.fov_ids)
    names = []
    for attrs in Elements(page.text).images:
        name = attrs["data-figure"]
        mime, data = re.fullmatch(r"data:(image/(?:png|jpeg));base64,(.+)", attrs["src"]).groups()
        image = Image.open(io.BytesIO(base64.b64decode(data, validate=True)))
        image.load()
        assert image.format == {"image/png": "PNG", "image/jpeg": "JPEG"}[mime]
        assert (int(attrs["width"]), int(attrs["height"])) == image.size, name
        expected = None
        if name == "time":
            expected = (TIME_WIDTH, max(image.size[1], time_height_at_least(0)))
        if name == "heatmap" and heatmap_rows_at_least(image, n_fovs):
            expected = (HEATMAP_WIDTH, image.size[1])
        if name.startswith("distribution:"):
            expected = DISTRIBUTION_THIRD
        if name.startswith("per_cell:"):
            row = re.findall(r'data-figure="(per_cell:[^"]*)"', page.text)
            k = row.index(name)
            expected = DISTRIBUTION_THIRD if grid_slot(min(3, len(row) - 3 * (k // 3))) == THIRD else \
                DISTRIBUTION_HALF
        if name == "reads_along_z" and heatmap_rows_at_least(image, len(set(page.tables["reads_along_z"]["fov_id"]))):
            expected = (READS_Z_WIDTH, image.size[1])
        if extra is not None and expected is None:
            expected = extra(name, image)
        assert expected is not None, f"no expected size for {name}"
        if expected != "checked":
            assert image.size == expected, name
        assert mime == ("image/png" if name in ("time", "heatmap", "reads_along_z", "intensity") or
                        name.startswith(("distribution:", "per_cell:")) else "image/jpeg"), name
        names.append(name)
    return names


@pytest.mark.parametrize("maker,variant", CASES, ids=IDS)
def test_pg5_size_budget_and_figure_sizes(pages, maker, variant):
    page = pages(maker, variant)
    size = (page.folder / "report.html").stat().st_size
    entry = next(e for e in page.record["files"] if e["path"] == "report.html")
    markers = len(page.case.config.markers) or 1
    budget = 6 * MIB + 16 * KIB * len(page.case.fov_ids) + 3 * MIB * markers
    assert entry["bytes"] == size == len(page.text.encode("utf-8")) and entry["budget_bytes"] == budget
    assert size <= budget
    names = check_figures(page)
    assert "time" in names and ("heatmap" in names) == (not page.tables["registration"].empty)
    assert [n for n in names if n.startswith("distribution:")] == (
        [] if page.tables["fovs"]["run_record"].sum() == 0 else
        ["distribution:spots", "distribution:correct_form_fraction", "distribution:matched_fraction"])
    if maker == "report_cells":
        assert {"per_cell:reads", "per_cell:genes", "reads_along_z"} <= set(names)
    assert load_report(page.folder) == page.record


# --- PG6: no verdict and no score -------------------------------------------------------------------------------

@pytest.mark.parametrize("maker,variant", CASES, ids=IDS)
def test_pg6_no_verdict_and_no_score(pages, maker, variant):
    page = pages(maker, variant)
    assert list(page.record) == CONTRACT_RECORD_KEYS and list(page.record["summary"]) == CONTRACT_SUMMARY_KEYS
    for name, columns in CONTRACT_COLUMNS.items():
        assert list(page.tables[name].columns) == [c for c, _, _ in columns], name
    words = json.dumps(page.record).lower()
    assert not any(w in words for w in ('"verdict"', '"score"', '"pass"', '"grade"'))


# --- The template and its slots ---------------------------------------------------------------------------------

def test_slots_are_filled_once_in_order(tmp_path, monkeypatch):
    template = page_module.template_text()
    assert re.findall(r"<!--slot:([a-z_]+)-->", template) == list(page_module.SLOTS)
    values = {name: f"[{name}]" for name in page_module.SLOTS}
    filled = page_module.fill(template, values)
    assert "<!--slot:" not in filled and all(f"[{name}]" in filled for name in page_module.SLOTS)
    with pytest.raises(ValueError, match="slots without a value"):
        page_module.fill(template, {k: v for k, v in values.items() if k != "facts"})
    with pytest.raises(ValueError, match="each once in this order"):
        page_module.fill(template.replace("<!--slot:time-->", ""), values)
    with pytest.raises(ValueError, match="each once in this order"):
        page_module.fill(template + "<!--slot:time-->", values)
    # A template with a slot left out raises before anything is written: the earlier report stays as it was.
    case = report_records(tmp_path / "root")
    case.collect()
    earlier = snapshot(case.folder)
    monkeypatch.setattr(page_module, "template_text", lambda: template.replace("<!--slot:facts-->", ""))
    with pytest.raises(ValueError, match="each once in this order"):
        case.collect()
    assert snapshot(case.folder) == earlier
    assert [p.name for p in case.folder.parent.iterdir()] == [SAMPLE]


def test_data_block_cannot_end_its_script(tmp_path):
    frames = {name: pd.DataFrame({"x": pd.Series(["</script><!--a", None], dtype="string")})
              for name in page_module.DATA_TABLES}
    text = page_module.data_block(frames)
    assert "</" not in text and "<!--" not in text
    assert json.loads(text)["fovs"] == [{"x": "</script><!--a"}, {"x": None}]


@pytest.mark.contract
@pytest.mark.parametrize("recorded", ["decode_reads", "decode_barcodes"])
def test_decoding_method_and_card_for_either_recorded_step_name(tmp_path, recorded):
    # W-557 (NM-16): a run recorded before the rename names its decoding step after FOV.decode_barcodes; the
    # report reads it as decode_reads (RECORDED_STEP_NAMES), and steps.csv keeps the recorded name.
    from starfinder.report._values import RECORDED_STEP_NAMES
    assert RECORDED_STEP_NAMES == {"decode_barcodes": "decode_reads"}
    case = report_records(tmp_path / "root")
    for fov in case.fov_ids:
        record = case.run_record(fov)
        for step in record["steps"]:
            if step["name"] == "decode_reads":
                step["name"] = recorded
        case.write_run_record(fov, record)
    folder, record = case.collect()
    page = Page(case, folder, record)
    rows = page.tables["steps"]
    decoding = rows[rows["name"] == recorded]
    assert len(decoding) == len(case.fov_ids) and set(decoding["method"]) == {"wta (multiplexed)"}
    cards = re.findall(r'<div class="st( off)?"[^>]*><b><span class="n">\d+</span>Decoding</b><i>([^<]*)</i>',
                       page.text)
    assert cards == [("", "wta (multiplexed)")]


# --- PG7 through Dataset.report ----------------------------------------------------------------------------------

def dataset_of(case):
    from starfinder.dataset import Dataset
    return Dataset(case.root / "input", case.root, DATASET, SAMPLE_ID, "output")


def report_through(case, **changes):
    arguments = dict(fov_ids=list(case.fov_ids), config=case.config) | changes
    fov_ids = arguments.pop("fov_ids")
    return dataset_of(case).report(case.sample, fov_ids, **arguments)


@pytest.fixture
def no_processing(monkeypatch):
    """Every processing stage raises: the entry reads saved records only."""
    from starfinder.dataset import FOV, Dataset

    def stage(name):
        def run(*args, **kwargs):
            raise AssertionError(f"Dataset.report called {name}")
        return run
    for owner, name in ((FOV, "run"), (FOV, "segment"), (FOV, "assign_molecules"), (Dataset, "stitch"),
                        (Dataset, "export")):
        monkeypatch.setattr(owner, name, stage(f"{owner.__name__}.{name}"))


ENTRY_FIXTURES = ("report_records", "report_cells")


def entry_case(root, maker):
    """``report_records`` (run records and a placement, no export) or ``report_cells`` (with the export)."""
    if maker == "report_cells":
        anndata_or_skip()
        return report_cells(root)
    return report_records(root)


def _check4_placement(case):
    path = case.root / "stitching" / SAMPLE / "placement.json"
    record = json.loads(path.read_text())
    del record["inputs"]["F4"]
    path.write_text(json.dumps(record))


def checks_for(maker):
    """CHECKS of test_report_collection.py for a fixture: without an export, check 2 removes no export folder and
    check 4's violation is a placement that does not hold the FOV list."""
    if maker == "report_cells":
        return CHECKS
    checks = []
    for number, error, message, edit, changes in CHECKS:
        if number == 2:
            edit = None
        if number == 4:
            message, edit = r"placement\.json: the placement holds the FOVs \['F1', 'F2', 'F3'\]", _check4_placement
        checks.append((number, error, message, edit, changes))
    return checks


@pytest.mark.parametrize("maker", ENTRY_FIXTURES)
@pytest.mark.parametrize("number", [c[0] for c in CHECKS])
def test_pg7_checks_through_dataset_report(tmp_path, no_processing, number, maker):
    case = entry_case(tmp_path / "root", maker)
    report_through(case)
    arguments = {}
    checks = checks_for(maker)
    for later, _, _, edit, changes in checks[number - 1:]:
        if number <= 2 and later in (3, 4):
            continue
        if edit is not None:
            edit(case)
        arguments = changes | arguments
    _, error, message, _, _ = checks[number - 1]
    earlier = snapshot(case.folder)
    with pytest.raises(error, match=message):
        report_through(case, **arguments)
    assert snapshot(case.folder) == earlier
    assert [p.name for p in case.folder.parent.iterdir()] == [SAMPLE]


def input_hashes(root):
    """The SHA-256 of every file under the output root but the report folder."""
    import hashlib
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file() and (root / "report") not in p.parents}


# The parts that need the export, unavailable for the whole sample of report_records with no_export.
EXPORT_PARTS = ("sample.per_cell", "sample.markers", "sample.marker_zoom", "sample.mecr", "sample.reads_along_z")
EXPORT_COLUMNS = ("n_cells", "n_cells_retained", "n_molecules", "n_molecules_retained", "reads_in_cells")


def _read_only(root, keep):
    """Every folder and file under root read-only, except ``keep`` and what is under it; returns an undo."""
    changed = []
    for path in [root, *root.rglob("*")]:
        if path == keep or keep in path.parents:
            continue
        mode = path.stat().st_mode
        path.chmod(mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
        changed.append((path, mode))

    def undo():
        for path, mode in reversed(changed):
            path.chmod(mode)
    return undo


# The places of "Where each record is looked for" under the output root, and the report's own folder.
PLACES = [r"fovs/[^/]+/run\.json", r"fovs/[^/]+/checkpoints/registered/.+", r"fovs/[^/]+/checkpoints/pre_qc\.json",
          r"fovs/[^/]+/segmentation/[^/]+/.+", r"fovs/[^/]+/assignment/[^/]+/.+",
          r"fovs/[^/]+/reference\.tif", rf"stitching/{SAMPLE}/.+", rf"export/{SAMPLE}/.+",
          rf"report/{SAMPLE}/.+", rf"report/\.{SAMPLE}\.[0-9a-f]+\.tmp(/.+)?"]


@pytest.mark.contract
@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root ignores read-only permissions")
# report_cells takes 5.4 s on one CPU with its imports done beforehand (W-530).
@pytest.mark.parametrize("maker", ["report_records", pytest.param("report_cells", marks=pytest.mark.slow)])
def test_pg7_entry_on_the_dataset(tmp_path, no_processing, maker):
    case = entry_case(tmp_path / "root", maker)
    (case.root / "report").mkdir()
    inputs_before = input_hashes(case.root)
    opened = []
    record_opens = {"on": False}

    def audit(event, arguments):
        if event == "open" and record_opens["on"] and isinstance(arguments[0], (str, bytes, os.PathLike)):
            opened.append(os.fsdecode(arguments[0]))
    sys.addaudithook(audit)
    undo = _read_only(case.root, case.root / "report")
    try:
        record_opens["on"] = True
        result = report_through(case)
    finally:
        record_opens["on"] = False
        undo()
    # The default folder, the record and its files.
    assert isinstance(result, ReportRecord) and result.path == case.root / "report" / SAMPLE
    assert result.record == load_report(result.path)
    assert (result.record["dataset_id"], result.record["sample_id"]) == (DATASET, SAMPLE_ID)
    assert sorted(p.name for p in result.path.iterdir()) == ["report.html", "report.json", "tables"]
    assert sorted(p.name for p in (result.path / "tables").iterdir()) == sorted(
        f"{t}.csv" for t in ("fovs", "registration", "steps", "readout_totals", "population", "problematic_fovs",
                             "cells", "mecr", "reads_along_z", "intensity", "unavailable"))
    # No input file changed: the hashes of every file outside the report folder are those before the entry.
    assert input_hashes(case.root) == inputs_before
    # Every file opened under the output root is at a place of "Where each record is looked for", or a file the run
    # records name (the codebook), or the report's own folder.
    named = {str(Path(case.run_record(f)["codebook"]["path"]).resolve()) for f in FOVS} | {
        str(Path(i["path"]).resolve()) for f in FOVS for i in case.run_record(f)["inputs"]}
    root = case.root.resolve()
    for path in opened:
        path = Path(path)
        if not path.is_absolute():
            continue
        resolved = path.resolve()
        if root not in resolved.parents or str(resolved) in named:
            continue
        relative = resolved.relative_to(root).as_posix()
        assert any(re.fullmatch(p, relative) for p in PLACES), relative
    assert any(p.endswith("run.json") for p in opened)
    # The record lists every file read, at those places.
    for entry in result.record["inputs"]:
        path = Path(entry["path"]) if Path(entry["path"]).is_absolute() else case.root / entry["path"]
        if str(path.resolve()) not in named:
            assert any(re.fullmatch(p, entry["path"]) for p in PLACES), entry
    if maker == "report_records":
        # A sample with run records and no export: the record says so, the parts that need the export are
        # unavailable with no_export (named once each), and the page is still written with their boxes.
        record = result.record
        assert record["route"] is None and record["stages"]["export"] is False
        assert record["stages"]["run_records"] == list(case.fov_ids) and record["stages"]["placement"] is True
        rows = record["unavailable"]
        page = (result.path / "report.html").read_text(encoding="utf-8")
        for part in EXPORT_PARTS:
            assert [r["reason"] for r in rows if r["part"] == part and r["table"] is None] == ["no_export"], part
            assert re.search(rf'data-part="{re.escape(part)}" data-key="" data-reason="no_export"[^>]*>'
                             r"<span>Unavailable: no export for this sample</span>", page), part
        for column in EXPORT_COLUMNS:
            assert [(r["row_key"], r["reason"]) for r in rows if r["table"] == "fovs" and r["column"] == column] == [
                (f"fov_id={f}", "no_export") for f in case.fov_ids], column
        for key in ("counted_cells", "reads_in_cells", "h5ad_whole_count_sum"):
            assert record["summary"][key] is None
            assert [r["reason"] for r in rows if r["table"] == "summary" and r["column"] == key] == ["no_export"]
        assert re.search(r'data-cell="summary\|counted_cells\|" data-t="Unavailable: no export for this sample',
                         page)
        assert all(r["reason"] == "no_export" for r in rows if r["table"] == "population" and r["column"] == "cells")
        # The parts from the run records are present: the readout table and the registration heatmap.
        assert '<div id="t-reads"></div>' in page and 'data-figure="heatmap"' in page
        assert record["summary"]["fovs_succeeded"] == len(case.fov_ids)
    # A second report through the entry replaces the first; an output_dir elsewhere is allowed.
    again = report_through(case)
    assert again.path == result.path and load_report(again.path) == again.record
    other = report_through(case, output_dir=tmp_path / "copy")
    assert other.path == tmp_path / "copy" and (other.path / "report.html").is_file()


# --- Every figure, on the synthetic chain ------------------------------------------------------------------------

from .report_scene import scene_chain_registered  # noqa: E402,F401


class _ChainCase:
    """The arguments of a chain's report, as the checks above read them from a case."""

    def __init__(self, chain, config):
        self.fov_ids, self.config, self.root = list(chain.fov_ids), config, chain.root


@pytest.mark.slow
def test_page_with_every_figure(scene_chain_registered):  # noqa: F811
    """PG1 to PG5 on a page with the replayed examples, the segmentation examples and two marker groups: each figure
    is embedded once, decodes and has the size of "Figure sizes"."""
    chain = scene_chain_registered
    genes = json.loads((chain.root / "export" / chain.sample / "export.json").read_text())["assembly"]["genes"]
    config = ReportConfig(markers=(MarkerGroup("first", tuple(genes[:2])), MarkerGroup("second", tuple(genes[2:5]))),
                          marker_pairs=((genes[0], genes[1]),))
    folder, record = chain.collect(config=config)
    page = Page(_ChainCase(chain, config), folder, record)
    for check in (test_pg1_embedded_data_equal_the_tables, test_pg2_sections_and_blocks,
                  test_pg3_every_unavailable_part_carries_its_reason, test_pg4_no_external_resource,
                  test_pg6_no_verdict_and_no_score):
        check(lambda maker, variant: page, "scene_chain_registered", None)
    reference = record["replay"]["rounds"]["reference"]
    run = chain.run_record(record["replay"]["fov_id"])
    columns = len(run["preprocessing"]["rounds"][reference]) + 1
    rows = len(run["channels"][reference])

    def sizes(name, image):
        if name == "preprocessing":
            assert image.size[1] > rows * preprocessing_tile(columns)
            return (tiles_width(columns, preprocessing_tile(columns)), image.size[1])
        if name in ("registration", "segmentation"):
            assert image.size[1] > EXAMPLE_TILE
            return (tiles_width(3, EXAMPLE_TILE), image.size[1])
        if name == "intensity":
            return INTENSITY_SIZE
        if re.fullmatch(r"marker:\d+:whole", name):
            assert image.size[0] == config.figure_width_px
            return "checked"
        if re.fullmatch(r"marker:\d+:example_\d:(xy|xz)", name):
            assert image.size[0] == 512
            return "checked"
        return None
    names = check_figures(page, sizes)
    assert {"preprocessing", "intensity", "registration", "segmentation", "marker:0:whole", "marker:1:whole"} <= \
        set(names)
    assert len(names) == len(set(names))
    assert sum(n.startswith("marker:1:") for n in names) == 1 + 6  # the whole view and three XY and three XZ tiles
    entry = next(e for e in record["files"] if e["path"] == "report.html")
    assert entry["bytes"] == (folder / "report.html").stat().st_size <= entry["budget_bytes"] == \
        6 * MIB + 16 * KIB * len(chain.fov_ids) + 3 * MIB * 2
    # The second marker group is hidden until the selector shows it.
    assert re.findall(r'<div class="mg" data-group="([^"]*)"( hidden)?>', page.text) == [("first", ""),
                                                                                       ("second", " hidden")]
