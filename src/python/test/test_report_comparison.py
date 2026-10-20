"""The comparison page of a dataset and the entry ``Dataset.compare_samples`` (W-457).

Checks of docs/report-contract.md ("The comparison entry", "``samples.csv``") and
docs/report-algorithms.md ("The comparison page", CP1, PG1, PG4) on finished reports of the
hand-written fixtures: ``report_cells`` reported as sample ``s1``, ``s2`` with no report folder,
``s3`` with a folder and no ``report.json``, and ``report_records`` (no export, so no cells)
reported as ``s4``. The table copies each present report's keys (every column compared with
the record), the box plots embed and draw the ``fovs.csv`` columns of each present sample, a
report whose files differ from its record raises before anything is written, nothing under a
sample's folder is written and no run record, placement or export is opened, and the page
loads nothing. The columns, dtypes, record keys and figure sizes are written here from the
pages; expected values come from the sample reports' own files and the fixtures' numbers,
never from the comparison's output.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import shutil
import sys
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd
import pytest

from starfinder.io._checkpoint import _read_table
from starfinder.report import ComparisonRecord, read_report

from .report_fixtures import CELLS_RETAINED, COUNTS, DATASET, FOVS, SAMPLE_ID, report_cells, report_records
from .test_report_collection import anndata_or_skip, snapshot

pytestmark = [pytest.mark.report, pytest.mark.validation]

# --- Written from docs/report-contract.md ("samples.csv") and docs/report-algorithms.md ("Figure sizes") ---------

# Column, dtype (nullable where the column can be null), and the key of report.json it copies.
SAMPLES_COLUMNS = [("sample", "string", None), ("report", "string", None), ("report_sha256", "string", None),
                   ("generated_at", "string", ("generated_at",)), ("route", "string", ("route",)),
                   ("fovs_expected", "Int64", ("summary", "fovs_expected")),
                   ("fovs_succeeded", "Int64", ("summary", "fovs_succeeded")),
                   ("spots_median_per_fov", "float64", ("summary", "spots_median_per_fov")),
                   ("matched_fraction", "float64", ("summary", "matched_fraction")),
                   ("counted_cells", "Int64", ("summary", "counted_cells")),
                   ("cells_median_per_fov", "float64", ("summary", "cells_median_per_fov"))]
# The three box plots: the figure's name on the page and the fovs.csv column it draws.
BOX_COLUMNS = ["spots", "matched_fraction", "n_cells_retained"]
# A plot of the comparison page's column of 1,160 px ("Figure sizes"), at twice its width: one per row of the page.
BOX_SIZE = (2320, 773)
TOL = 1e-9


def dataset_at(root):
    from starfinder.dataset import Dataset
    return Dataset(root / "input", root, DATASET, SAMPLE_ID, "output")


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """The output root ``a`` with the reports ``s1`` (report_cells), ``s3`` (a folder without report.json) and ``s4``
    (report_records of the root ``b``, written into ``a/report/s4``); ``s2`` has no folder."""
    anndata_or_skip()
    base = tmp_path_factory.mktemp("comparison")
    cells = report_cells(base / "a")
    cells.collect()
    records = report_records(base / "b")
    records.collect(output_dir=base / "a" / "report" / "s4")
    (base / "a" / "report" / "s3" / "tables").mkdir(parents=True)
    (base / "a" / "report" / "s3" / "notes.txt").write_text("not a report\n")
    return base / "a"


@pytest.fixture
def root(built, tmp_path):
    """A copy of the built output root for one test, so that a test may change it."""
    copy = tmp_path / "a"
    shutil.copytree(built, copy, symlinks=True)
    return copy


def sample_folders(root):
    return {name: snapshot(root / "report" / name) for name in ("s1", "s2", "s3", "s4")}


def written(result):
    """samples.csv read back with the dtypes the record lists, and the text of index.html."""
    entry = next(e for e in result.record["files"] if e["path"] == "samples.csv")
    frame = _read_table(result.path / "samples.csv", entry["dtypes"])
    return frame, (result.path / "index.html").read_text(encoding="utf-8"), entry["dtypes"]


def data_block(text):
    blocks = re.findall(r'<script type="application/json" id="report-data">(.*?)</script>', text, re.S)
    assert len(blocks) == 1
    return blocks[0], json.loads(blocks[0])


def key(record, path):
    for name in path:
        record = record[name]
    return record


def missing_cell(value):
    return value is None or bool(pd.isna(value))


def equal(actual, expected):
    """A cell of samples.csv equals the record's key: integers and strings exactly, floats to 1e-9, null as null."""
    if expected is None:
        return missing_cell(actual)
    if isinstance(expected, float):
        return not missing_cell(actual) and abs(float(actual) - expected) <= TOL
    return not missing_cell(actual) and actual == expected


# --- The table ------------------------------------------------------------------------------------------------------

def test_samples_csv_copies_each_report(root):
    before = sample_folders(root)
    result = dataset_at(root).compare_samples(["s1", "s2", "s3"])
    assert isinstance(result, ComparisonRecord) and result.path == root / "report"
    frame, _, dtypes = written(result)
    assert dtypes == {c: d for c, d, _ in SAMPLES_COLUMNS}
    assert list(frame.columns) == [c for c, _, _ in SAMPLES_COLUMNS]
    assert list(frame["sample"]) == ["s1", "s2", "s3"]
    assert list(frame["report"]) == ["present", "missing", "missing"]
    rows = frame.to_dict("records")
    s1 = root / "report" / "s1" / "report.json"
    record = json.loads(s1.read_text(encoding="utf-8"))
    assert rows[0]["report_sha256"] == hashlib.sha256(s1.read_bytes()).hexdigest()
    for column, _, path in SAMPLES_COLUMNS:
        if path is not None:
            assert equal(rows[0][column], key(record, path)), column
    # report_cells: every key of the row has a value (the export holds the cells).
    assert all(not missing_cell(v) for v in rows[0].values())
    assert rows[0]["route"] == "per_fov" and rows[0]["fovs_expected"] == len(FOVS)
    for row in rows[1:]:
        assert all(missing_cell(row[c]) for c, _, _ in SAMPLES_COLUMNS if c not in ("sample", "report")), row
    # The record returns the rows it wrote; nothing under a sample's folder is written.
    assert [r["sample"] for r in result.record["samples"]] == ["s1", "s2", "s3"]
    assert sample_folders(root) == before
    assert sorted(p.name for p in result.path.iterdir()) == ["index.html", "s1", "s3", "s4", "samples.csv"]
    for entry in result.record["files"]:
        data = (result.path / entry["path"]).read_bytes()
        assert (entry["sha256"], entry["bytes"]) == (hashlib.sha256(data).hexdigest(), len(data))


def test_samples_csv_of_a_report_without_cells(root):
    """s4 (no export): route null and the cell keys null, as its report.json holds them."""
    result = dataset_at(root).compare_samples(["s4"])
    frame, _, _ = written(result)
    row = frame.to_dict("records")[0]
    record = read_report(root / "report" / "s4")
    assert row["report"] == "present"
    for column, _, path in SAMPLES_COLUMNS:
        if path is not None:
            assert equal(row[column], key(record, path)), column
    assert missing_cell(row["route"]) and missing_cell(row["counted_cells"]) and \
        missing_cell(row["cells_median_per_fov"])
    assert row["spots_median_per_fov"] == pytest.approx(1000.0, abs=TOL)


# --- The box plots --------------------------------------------------------------------------------------------------

class Images(HTMLParser):
    """Every ``<img>`` of a page, and every ``href``."""

    def __init__(self, text):
        super().__init__(convert_charrefs=True)
        self.images, self.links = [], []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "img":
            self.images.append(attrs)
        if "href" in attrs:
            self.links.append(attrs["href"])


def decode(attrs):
    from PIL import Image
    mime, data = re.fullmatch(r"data:(image/png);base64,(.+)", attrs["src"]).groups()
    image = Image.open(io.BytesIO(base64.b64decode(data, validate=True)))
    image.load()
    assert image.format == "PNG"
    return image.convert("RGB")


def hatched(image, centre_x):
    """Whether the column of pixels 30 px right of a slot's centre crosses the empty slot's fill (#eeeeee)."""
    x = centre_x + 30
    return sum(1 for y in range(image.height) if image.getpixel((x, y)) == (238, 238, 238)) > 100


def test_box_plots(root):
    samples = ["s1", "s2", "s3", "s4"]
    result = dataset_at(root).compare_samples(samples)
    _, text, _ = written(result)
    _, data = data_block(text)
    assert list(data) == ["samples", "fovs"]
    assert list(data["fovs"]) == ["s1", "s4"]
    for sample in ("s1", "s4"):
        record = read_report(root / "report" / sample)
        entry = next(e for e in record["files"] if e["path"] == "tables/fovs.csv")
        fovs = _read_table(root / "report" / sample / "tables" / "fovs.csv", entry["dtypes"])
        rows = data["fovs"][sample]
        assert [r["fov_id"] for r in rows] == list(fovs["fov_id"]) == list(FOVS)
        for row, saved in zip(rows, fovs.to_dict("records")):
            assert list(row) == ["fov_id"] + BOX_COLUMNS
            for column in BOX_COLUMNS:
                assert (row[column] is None) == missing_cell(saved[column]), (sample, column)
                if row[column] is not None:
                    assert row[column] == pytest.approx(float(saved[column]), abs=TOL), (sample, column)
    # The fixtures' own numbers: spots and assigned / spots of every FOV; s1's retained cells, s4 none.
    for sample in ("s1", "s4"):
        rows = data["fovs"][sample]
        assert [r["spots"] for r in rows] == [COUNTS["spots"][f] for f in FOVS]
        assert [r["matched_fraction"] for r in rows] == pytest.approx(
            [COUNTS["assigned"][f] / COUNTS["spots"][f] for f in FOVS], abs=TOL)
    assert [r["n_cells_retained"] for r in data["fovs"]["s1"]] == [CELLS_RETAINED[f] for f in FOVS]
    assert [r["n_cells_retained"] for r in data["fovs"]["s4"]] == [None] * len(FOVS)
    # The slots: one per sample in order, a box with its FOVs, or an empty slot labelled with its reason.
    slots = result.record["box_plots"]
    assert list(slots) == BOX_COLUMNS
    for column in BOX_COLUMNS:
        assert [s["sample"] for s in slots[column]] == samples
        assert [(s["fovs"], s["label"]) for s in slots[column]][1:3] == [(0, "missing"), (0, "missing")]
        assert (slots[column][0]["fovs"], slots[column][0]["label"]) == (4, None)
    assert (slots["spots"][3]["fovs"], slots["matched_fraction"][3]["fovs"]) == (4, 4)
    assert (slots["n_cells_retained"][3]["fovs"], slots["n_cells_retained"][3]["label"]) == (0, "no cells")
    # The decoded figures: three PNGs of a full column, in page order, each slot drawn where it is.
    images = Images(text).images
    assert [a["data-figure"] for a in images] == [f"box:{c}" for c in BOX_COLUMNS]
    for attrs, column in zip(images, BOX_COLUMNS):
        image = decode(attrs)
        assert image.size == BOX_SIZE == (int(attrs["width"]), int(attrs["height"])), column
        # The axes span x = 160 to 2296 px over the slots 0.5 to 4.5.
        centres = [round(160 + (i - 0.5) / len(samples) * (BOX_SIZE[0] - 184)) for i in range(1, len(samples) + 1)]
        empty = [s["fovs"] == 0 for s in slots[column]]
        assert [hatched(image, x) for x in centres] == empty, column
    assert "s4, no cells" in images[2]["alt"] and "s1, 4 FOVs" in images[2]["alt"]


# --- Hash checks and read-only reports ----------------------------------------------------------------------------

def _change_fovs_table(root):
    path = root / "report" / "s1" / "tables" / "fovs.csv"
    data = path.read_bytes()
    assert b"\nF2,1," in data
    path.write_bytes(data.replace(b"\nF2,1,", b"\nF2,9,", 1))
    return "tables/fovs.csv"


def _change_unavailable_table(root):
    path = root / "report" / "s1" / "tables" / "unavailable.csv"
    path.write_bytes(path.read_bytes() + b"\n")
    return "tables/unavailable.csv"


def _change_recorded_hash(root):
    path = root / "report" / "s1" / "report.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    entry = next(e for e in record["files"] if e["path"] == "tables/fovs.csv")
    entry["sha256"] = "0" * 64
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return "tables/fovs.csv"


def _truncate_record(root):
    path = root / "report" / "s1" / "report.json"
    path.write_bytes(path.read_bytes()[:200])
    return "report.json"


def _other_version(root):
    path = root / "report" / "s1" / "report.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["format_version"] = 2
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return "report.json"


CHANGES = {"fovs_table": _change_fovs_table, "unavailable_table": _change_unavailable_table,
           "recorded_hash": _change_recorded_hash, "truncated_record": _truncate_record,
           "other_version": _other_version}


@pytest.mark.parametrize("change", list(CHANGES))
def test_a_changed_report_raises_naming_the_file(root, change):
    dataset = dataset_at(root)
    dataset.compare_samples(["s1", "s2"])
    earlier = {name: (root / "report" / name).read_bytes() for name in ("index.html", "samples.csv")}
    named = CHANGES[change](root)
    before = sample_folders(root)
    with pytest.raises(ValueError, match=re.escape(str(root / "report" / "s1" / named))):
        dataset.compare_samples(["s1", "s2", "s4"])
    # Nothing is written: the sample folders and the earlier page and table are byte for byte as they were.
    assert sample_folders(root) == before
    assert {name: (root / "report" / name).read_bytes() for name in earlier} == earlier
    assert sorted(p.name for p in (root / "report").iterdir()) == ["index.html", "s1", "s3", "s4", "samples.csv"]


def test_reads_the_reports_only(root, monkeypatch):
    """No run record, placement or export is opened (their readers raise, and no file under them is opened), nothing
    under a sample folder is written and no file outside the report root changes."""
    import h5py

    import starfinder.report._collect as collect
    import starfinder.report._measurements as measurements
    import starfinder.sample_output._record as export_record
    import starfinder.stitching as stitching
    import starfinder.stitching._persist as persist

    def refuse(name):
        def read(*args, **kwargs):
            raise AssertionError(f"compare_samples called {name}")
        return read
    for owner, name in ((collect, "collect_report"), (collect, "_check_run_records"), (collect, "_check_export"),
                        (collect, "_check_placement"), (stitching, "load_placement"), (persist, "load_placement"),
                        (export_record, "read_export"), (export_record, "verify_section_hashes"),
                        (measurements, "read_h5ad"), (h5py, "File")):
        monkeypatch.setattr(owner, name, refuse(f"{owner.__name__}.{name}"))
    outside = {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*"))
               if p.is_file() and (root / "report") not in p.parents}
    assert any(p.startswith(("checkpoints/", "stitching/", "export/")) for p in outside)
    before = sample_folders(root)
    opened, recording = [], {"on": False}

    def audit(event, arguments):
        if event == "open" and recording["on"] and isinstance(arguments[0], (str, bytes, os.PathLike)):
            opened.append(os.fsdecode(arguments[0]))
    sys.addaudithook(audit)
    recording["on"] = True
    try:
        dataset_at(root).compare_samples(["s1", "s2", "s3", "s4"])
    finally:
        recording["on"] = False
    under = [Path(p).resolve() for p in opened if Path(p).resolve().is_relative_to(root.resolve())]
    relative = sorted({p.relative_to(root.resolve()).as_posix() for p in under})
    assert relative and all(p.startswith("report/") for p in relative), relative
    assert not any(p.startswith(("report/s2/", "report/s3/")) for p in relative)
    assert sample_folders(root) == before
    assert {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*"))
            if p.is_file() and (root / "report") not in p.parents} == outside


# --- The page -----------------------------------------------------------------------------------------------------

def test_page_loads_nothing_and_embeds_samples_csv(root):
    result = dataset_at(root).compare_samples(["s1", "s2", "s3"])
    frame, text, dtypes = written(result)
    # PG4, with the relative links of index.html: no external resource, a light theme only.
    assert not re.search(r"https?:", text, re.I)
    assert not re.search(r'(src|href)="//', text)
    for word in ("<link", "<iframe", "<object", "@import", "prefers-color-scheme"):
        assert word not in text.lower(), word
    sources = re.findall(r'\ssrc="([^"]*)"', text)
    assert len(sources) == 3 and all(s.startswith("data:") for s in sources)
    assert all(text[m.end():].startswith("data:") for m in re.finditer(r"url\(", text))
    assert re.findall(r'\shref="([^"]*)"', text) == Images(text).links == ["s1/report.html"]
    assert len(re.findall(r"<script\b", text)) == 2 and not re.search(r"<script[^>]*\ssrc=", text)
    # PG1 on samples: the embedded rows equal samples.csv cell by cell, null exactly where the table's cell is missing.
    block, data = data_block(text)
    assert "</" not in block
    rows = data["samples"]
    assert all(list(row) == list(dtypes) for row in rows)
    embedded = pd.DataFrame(rows, columns=list(dtypes), dtype=object).astype(dtypes)
    pd.testing.assert_frame_equal(embedded, frame, check_exact=False, rtol=0, atol=TOL)
    for row, saved in zip(rows, frame.to_dict("records")):
        for column in dtypes:
            assert (row[column] is None) == missing_cell(saved[column]), column
    # The table of the page: a present sample links its report; a missing one says "missing", its other cells blank.
    body = text.split('id="t-samples"', 1)[1].split("</tbody>", 1)[0]
    cells = [re.findall(r"<td[^>]*>(.*?)</td>", r, re.S) for r in re.findall(r"<tr>(.*?)</tr>", body, re.S)][1:]
    assert [c[0] for c in cells] == ['<a href="s1/report.html">s1</a>', "s2", "s3"]
    assert [c[1] for c in cells] == ["present", "missing", "missing"]
    assert all(c[2:] == [""] * 8 for c in cells[1:])
    assert all(c != "" for c in cells[0])


def test_template_slots_are_filled_once_in_order():
    from starfinder.report import _compare
    from starfinder.report._page import fill, template_text
    template = template_text(_compare.INDEX_TEMPLATE)
    assert re.findall(r"<!--slot:([a-z_]+)-->", template) == list(_compare.INDEX_SLOTS)
    values = {name: f"[{name}]" for name in _compare.INDEX_SLOTS}
    assert "<!--slot:" not in fill(template, values, _compare.INDEX_SLOTS)
    with pytest.raises(ValueError, match="each once in this order"):
        fill(template + "<!--slot:data-->", values, _compare.INDEX_SLOTS)


def test_a_second_page_replaces_the_first(root):
    dataset = dataset_at(root)
    dataset.compare_samples(["s1", "s2", "s3", "s4"])
    result = dataset.compare_samples(["s4", "s1"])
    frame, text, _ = written(result)
    assert list(frame["sample"]) == ["s4", "s1"] and list(frame["report"]) == ["present", "present"]
    assert re.findall(r'\shref="([^"]*)"', text) == ["s4/report.html", "s1/report.html"]
    assert sorted(p.name for p in result.path.iterdir()) == ["index.html", "s1", "s3", "s4", "samples.csv"]


# --- Argument checks ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("samples", [[], ["s1", "s1"], ["s1", "s2", "s1"], "s1", [""], ["s1/tables"], [".."], [1]],
                         ids=["empty", "repeated", "repeated_later", "string", "empty_name", "path", "parent",
                              "not_a_string"])
def test_argument_checks(tmp_path, samples):
    """Runs without the anndata extra: s1 is report_records."""
    case = report_records(tmp_path / "root")
    case.collect()
    report = case.root / "report"
    before = snapshot(report)
    with pytest.raises(ValueError):
        dataset_at(case.root).compare_samples(samples)
    assert snapshot(report) == before
    assert sorted(p.name for p in report.iterdir()) == ["s1"]


@pytest.mark.contract
def test_a_report_root_given_or_absent(tmp_path):
    """``report_root`` is read in place of ``<output_root>/report/``; one that does not exist gives missing rows."""
    case = report_records(tmp_path / "root")
    case.collect(output_dir=tmp_path / "elsewhere" / "s1")
    dataset = dataset_at(case.root)
    result = dataset.compare_samples(["s1"], report_root=tmp_path / "elsewhere")
    assert result.path == tmp_path / "elsewhere"
    assert list(written(result)[0]["report"]) == ["present"]
    result = dataset.compare_samples(["s1", "s2"])
    assert result.path == case.root / "report"
    assert list(written(result)[0]["report"]) == ["missing", "missing"]
