"""The figures and the page after Jiahao's first review of the real block report (W-475).

Checks of docs/report-algorithms.md ("Figure sizes" and the passages it names) on the figures
of the pages the page tests build (``report_records`` and ``report_cells`` with their
variants), on the comparison page, on a hand-written sample of 64 FOVs and on hand-written
examples: no text is smaller than 11 px as shown (the text size), each PNG is twice the
width of its slot and the slots are those of the page's CSS and of the page's table (the
slots), no text overlaps another, another axes or the edge (no overlap), a heatmap draws
at most 64 rows of at least 14 px as shown, each labelled, sampled by the page's rule (the
heatmap rows), adjacent tiles have a white gap of at least 8 px as shown and the zoomed
region is bounded by a yellow rectangle (the tiles), the regions of the whole-sample view
are yellow rectangles with readable numbers (the regions), the label of a readout level is
never clipped to its bar (the readout bars), and the Summary writes "Version", "Run time"
and "Generated" on one line each, in UTC and then in the viewer's time zone (the summary).

The figures are checked as drawn, before JPEG or PNG coding (``_figures.KEEP_CANVAS``). The
numbers are written here from the page (``report_sizes``); nothing is read from a run
directory and no image is committed.
"""
from __future__ import annotations

import base64
import html
import io
import json
import math
import re
import shutil
import subprocess
from datetime import datetime, timezone
from itertools import combinations
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from starfinder.report import _figures, _page
from starfinder.report._measurements import GroupFigureData

from . import report_fixtures
from .report_fixtures import CASES, IDS, collect_drawn, report_cells, report_records
from .report_sizes import (BOX_SIZE, COMPARISON, DISTRIBUTION_HALF, DISTRIBUTION_THIRD, EXAMPLE_TILE, FULL, HALF,
                           INTENSITY_SIZE, MARK_PX, MAX_ROWS, MIN_TEXT_PX, ROW_PX, THIRD, TILE_GAP_PX,
                           preprocessing_tile, tile_gap, tiles_width, time_height_at_least)
from .test_report_collection import anndata_or_skip
from .test_report_page import SHIM

pytestmark = [pytest.mark.report, pytest.mark.validation]
# The shared fixtures of report_fixtures.py: the pages drawn once for the session, and KEEP_CANVAS on.
drawn, keep = report_fixtures.drawn, report_fixtures.keep

YELLOW = (0xff, 0xd2, 0x1f)
WHITE = (255, 255, 255)
# The step names of a workflow run record (the per-FOV steps, then the sample steps, of LN sample1's runs).
FOV_STEPS = ("load_images", "rotate_round", "preprocess:min_max_normalization", "find_spots", "extract_round",
             "register", "write_checkpoint:candidates", "decode_reads", "score_reads",
             "write_checkpoint:pre_qc", "filter_reads", "segment", "assign_molecules")
SAMPLE_STEPS = ("stitch", "assemble", "fuse", "segment", "fuse_labels", "export_spatialdata")


# --- The figures of a page, drawn with their canvases --------------------------------------------------------------

def page_figures(figures):
    """[(name, FigureImage)] of a page's PageFigures, named as the page's ``data-figure``."""
    from starfinder.report._values import Unavailable
    named = [("time", figures.time), ("heatmap", figures.heatmap), ("preprocessing", figures.preprocessing),
             ("intensity", figures.intensity), ("registration", figures.registration),
             ("segmentation", figures.segmentation)]
    named += [(f"distribution:{column}", figure) for column, _, figure in figures.distributions]
    sample = figures.sample
    if isinstance(sample.per_cell, list):
        named += [(f"per_cell:{f.tiles[0]['name']}", f) for f in sample.per_cell]
    if not isinstance(sample.reads_along_z, Unavailable):
        named.append(("reads_along_z", sample.reads_along_z))
    if isinstance(sample.markers, list):
        for k, marker in enumerate(sample.markers):
            named.append((f"marker:{k}:whole", marker.whole))
            for (role, view), tile in (marker.zooms.items() if isinstance(marker.zooms, dict) else []):
                if not isinstance(tile, Unavailable):
                    named.append((f"marker:{k}:{role}:{view}", tile))
    return [(name, figure) for name, figure in named if figure is not None]


# --- Checks of one figure ------------------------------------------------------------------------------------------

def shown(figure):
    """The scale of a figure as shown: its slot's width over its pixel width."""
    return figure.slot / figure.width


def glyph_runs(text, nominal_px):
    """The size in figure px of each glyph run of a text drawn at ``nominal_px``: the size itself for plain text; for
    mathtext (an even number of unescaped "$"), the size of each glyph Matplotlib's mathtext parser lays out, so that a
    superscript counts at its own size."""
    from matplotlib import cbook
    from matplotlib.font_manager import FontProperties
    from matplotlib.mathtext import MathTextParser
    if not cbook.is_math_text(text):
        return [nominal_px]
    parsed = MathTextParser("path").parse(text, dpi=72, prop=FontProperties(size=nominal_px * 72 / 100))
    return [glyph[1] * 100 / 72 for glyph in parsed.glyphs]


def smallest_text(figure):
    """The smallest glyph run of a figure's texts in px as shown, measured here from each text and its font size (and
    equal to the size the figure records)."""
    assert figure.texts, "a figure without text"
    sizes = []
    for t in figure.texts:
        runs = glyph_runs(t["text"], t["nominal_px"])
        assert min(runs) == pytest.approx(t["size_px"], abs=1e-9), t["text"]
        sizes += runs
    return min(sizes) * shown(figure)


def intersect(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def overlaps(figure):
    """The intersections of a text's box with another axes' frame, another text's box or the outside of the canvas."""
    found = []
    for t in figure.texts:
        x0, y0, x1, y1 = t["box"]
        if x0 < 0 or y0 < 0 or x1 > figure.width or y1 > figure.height:
            found.append(("edge", t["text"], t["box"]))
        for i, frame in enumerate(figure.frames):
            if i != t["axes"] and intersect(t["box"], frame):
                found.append(("axes", t["text"], i))
    for a, b in combinations(figure.texts, 2):
        if intersect(a["box"], b["box"]):
            found.append(("text", a["text"], b["text"]))
    return found


def check_text_size(name, figure):
    assert smallest_text(figure) >= MIN_TEXT_PX, (name, smallest_text(figure))


def check_slot(name, figure):
    """A PNG is twice the width of its slot; the slot of each figure is the one of its place on the page."""
    if figure.format == "png":
        assert figure.width == 2 * figure.slot, name
    expected = {"time": HALF, "reads_along_z": HALF, "heatmap": FULL, "intensity": FULL, "preprocessing": FULL,
                "registration": FULL, "segmentation": FULL}.get(name)
    if re.fullmatch(r"marker:\d+:whole", name):
        expected = FULL
    if re.fullmatch(r"marker:\d+:example_\d:(xy|xz)", name):
        expected = THIRD
    if expected is not None:
        assert figure.slot == expected, name


# --- The text size and the slots on the pages the page tests build -------------------------------------------------

@pytest.mark.parametrize("maker,variant", CASES, ids=IDS)
def test_text_size_and_slots_on_every_page(drawn, maker, variant):
    page = drawn(maker, variant)
    names = []
    for name, figure in page_figures(page.figures):
        check_text_size(name, figure)
        check_slot(name, figure)
        names.append(name)
        assert f'data-figure="{name}"' in page.text, name
        assert figure.canvas is not None and figure.canvas.shape[:2] == (figure.height, figure.width), name
    assert "time" in names
    check_declared_sizes(page.text)
    # A row of the three-figure grid: three figures in thirds; the distributions across FOVs are three.
    distributions = [f for n, f in page_figures(page.figures) if n.startswith("distribution:")]
    assert all((f.width, f.height) == DISTRIBUTION_THIRD and f.slot == THIRD for f in distributions)
    per_cell = [f for n, f in page_figures(page.figures) if n.startswith("per_cell:")]
    if per_cell:
        assert len(per_cell) == 3 and all((f.width, f.height) == DISTRIBUTION_THIRD for f in per_cell)
        assert re.search(r'<div class="g3 fixed">(<img [^>]*data-figure="per_cell:[^"]*">){3}</div>', page.text)


def test_no_overlap_on_the_pages(drawn):
    for maker, variant in (("report_records", None), ("report_cells", None)):
        for name, figure in page_figures(drawn(maker, variant).figures):
            assert overlaps(figure) == [], (maker, name)


def css_rules(text):
    """{selector: declarations} of the page's CSS (a selector list is split)."""
    style = re.search(r"<style>(.*?)</style>", text, re.S).group(1)
    style = re.sub(r"/\*.*?\*/", "", style, flags=re.S)
    rules = {}
    for selectors, body in re.findall(r"([^{}]+)\{([^{}]*)\}", style):
        for selector in selectors.split(","):
            rules.setdefault(selector.strip(), []).append(body)
    return {k: ";".join(v) for k, v in rules.items()}


def px(declarations, name):
    return [float(v) for v in re.search(rf"(?:^|;){name}:([^;]+)", declarations).group(1).replace("px", "").split()]


def test_slots_are_the_page_columns_and_the_table():
    """The slots follow from the page's CSS in a window 1,200 px wide and equal the code's constants and the numbers
    of the page's table "Figure sizes" as ``report_sizes`` writes them. ``test_report_docs_contract.py`` checks the
    page's table and its rules against ``report_sizes`` (W-602)."""
    rules = css_rules(_page.template_text())
    wrap = rules[".wrap"]
    content = min(1200.0, px(wrap, "max-width")[0]) - 2 * px(wrap, "padding-inline")[0]
    shell = rules[".shell"]
    rail = float(re.search(r"grid-template-columns:(\d+)px", shell).group(1))
    full = content - rail - px(shell, "gap")[0]
    half = (full - px(rules[".g2"], "gap")[-1]) / 2
    third = (full - 2 * px(rules[".g3"], "gap")[0]) / 3
    assert (full, half, third, content) == (FULL, HALF, THIRD, COMPARISON)
    assert (_figures.FULL_COLUMN, _figures.HALF_COLUMN, _figures.THIRD_COLUMN, _figures.COMPARISON_COLUMN) == (
        FULL, HALF, THIRD, COMPARISON)
    assert ".g3.fixed" in rules and "repeat(3," in rules[".g3.fixed"] and "repeat(2," in rules[".g2"]
    assert (_figures.MIN_TEXT_PX, _figures.TILE_GAP_PX, _figures.MARK_PX, _figures.HEATMAP_ROWS) == (
        MIN_TEXT_PX, TILE_GAP_PX, MARK_PX, MAX_ROWS)


def test_a_row_of_two_and_one_figure(tmp_path):
    """A row of two figures of the grid has two columns, a single figure a half column, each drawn for its slot."""
    images = [f'<img data-figure="f{k}">' for k in range(5)]
    assert _page.grid(images[:3]) == f'<div class="g3 fixed">{"".join(images[:3])}</div>'
    assert _page.grid(images[:2]) == f'<div class="g2">{"".join(images[:2])}</div>'
    assert _page.grid(images[:1]) == f'<div class="g2">{images[0]}</div>'
    assert _page.grid(images) == (f'<div class="g3 fixed">{"".join(images[:3])}</div>'
                                  f'<div class="g2">{"".join(images[3:])}</div>')
    assert [_figures.grid_slot(n) for n in (1, 2, 3)] == [HALF, HALF, THIRD]
    # The per-cell distributions of report_cells without a cell size: two figures in a row of two columns.
    from starfinder.report import _measurements
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_measurements, "SIZE_UNITS", ())
        page = collect_drawn("report_cells", None, tmp_path / "root")
    per_cell = [(n, f) for n, f in page_figures(page.figures) if n.startswith("per_cell:")]
    assert [n for n, _ in per_cell] == ["per_cell:reads", "per_cell:genes"]
    for name, figure in per_cell:
        assert (figure.width, figure.height) == DISTRIBUTION_HALF and figure.slot == HALF
        check_text_size(name, figure)
        assert overlaps(figure) == []
    assert re.search(r'<div class="g2">(<img [^>]*data-figure="per_cell:[^"]*">){2}</div>', page.text)
    # "Time per step" and "Reads along Z" sit in the half columns of a two-column row.
    for block in ("b-time", "b-reads-z"):
        start = page.text.index(f'id="{block}"')
        assert page.text.rfind('<div class="g2 split">', 0, start) > page.text.rfind("</section>", 0, start)


# --- The comparison page --------------------------------------------------------------------------------------------

def test_comparison_page_figures(tmp_path, keep):
    from starfinder.report._compare import compare_samples
    case = report_records(tmp_path / "root")
    case.collect()
    made = []
    original = _figures.sample_box_figure

    def box(*args, **kwargs):
        made.append(original(*args, **kwargs))
        return made[-1]
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_figures, "sample_box_figure", box)
        compare_samples(case.root / "report", [case.sample, "missing"])
    assert len(made) == 3
    for figure in made:
        assert (figure.width, figure.height) == BOX_SIZE and figure.slot == COMPARISON
        check_text_size("box", figure)
        assert overlaps(figure) == []
    text = (case.root / "report" / "index.html").read_text(encoding="utf-8")
    assert len(re.findall(rf'width="{BOX_SIZE[0]}" height="{BOX_SIZE[1]}" data-figure="box:', text)) == 3


# --- A hand-written sample of 64 FOVs, and the heatmap rows -------------------------------------------------------

def fov_names(n):
    return [f"Position{k:03d}" for k in range(1, n + 1)]


def registration_rows(fovs, rounds=("round2", "round3")):
    rng = np.random.default_rng(7)
    return [{"fov_id": f, "round": r, **{f"residual_{a}": float(rng.normal(0, 2)) for a in "zyx"}}
            for f in fovs for r in rounds]


def along_rows(fovs, planes=50):
    rng = np.random.default_rng(8)
    return [{"fov_id": f, "z": z, "reads": int(rng.integers(0, 500))} for f in fovs for z in range(planes)]


def expected_rows(fovs, marked=()):
    """The rule of "Registration" written here: every FOV up to 64; above, the marked FOVs (the first 64), then the
    FOVs at floor(i · u / k) of the u others, all in list order."""
    if len(fovs) <= 64:
        return list(fovs)
    first = [f for f in fovs if f in set(marked)][:64]
    others = [f for f in fovs if f not in set(marked)]
    k, u = 64 - len(first), len(others)
    picked = set(first) | {others[math.floor(i * u / k)] for i in range(k)}
    return [f for f in fovs if f in picked]


def row_labels(figure):
    """The FOV labels of a heatmap's rows (the tick labels of its first axes, top to bottom)."""
    labels = [t for t in figure.texts if t["axes"] == 0 and t["text"].startswith("Position")]
    return [t["text"] for t in sorted(labels, key=lambda t: t["box"][1])]


@pytest.mark.parametrize("n", [4, 64, 65, 200])
def test_heatmap_rows(keep, n):
    fovs = fov_names(n)
    # Three FOVs with a shift_residual row at positions the even spacing of 64 of 200 does not pick.
    spaced = {math.floor(i * n / 64) for i in range(64)}
    marked = [fovs[p] for p in (1, 101, 198)] if n == 200 else []
    assert all(fovs.index(f) not in spaced for f in marked)
    rows = registration_rows(fovs)
    rows_before = json.dumps(rows)
    heatmap = _figures.residual_heatmap_figure(rows, fovs, ["round2", "round3"], marked)
    along_input = along_rows(fovs)
    along = _figures.reads_along_z_figure(along_input, {f: 50 for f in fovs})
    assert json.dumps(rows) == rows_before  # the rows given are not changed
    for figure, chosen, table in ((heatmap, expected_rows(fovs, marked), "registration.csv"),
                                  (along, expected_rows(fovs), "reads_along_z.csv")):
        assert figure.marks["rows"] == chosen and len(chosen) == min(n, 64)
        assert row_labels(figure) == chosen
        assert figure.marks["row_px"] * shown(figure) >= ROW_PX
        note = f"64 of {n} FOVs shown; every FOV in {table}"
        labels = [t["text"] for t in figure.texts]
        assert (note in labels) == (n > 64) and figure.marks["note"] == (note if n > 64 else None)
        check_text_size(table, figure)
        assert overlaps(figure) == [], (table, overlaps(figure)[:3])
    assert set(marked) <= set(heatmap.marks["rows"])
    assert [t["fov_id"] for t in along.tiles] == expected_rows(fovs)


def test_heatmap_tables_are_not_sampled(tmp_path):
    """The figure draws a sample, the tables hold every FOV: with the cap at two rows the four FOVs of report_records
    are all in registration.csv, byte for byte the tables of the report without the cap."""
    full = collect_drawn("report_records", None, tmp_path / "a" / "root")
    capped = collect_drawn("report_records", None, tmp_path / "b" / "root", HEATMAP_ROWS=2)
    for path in sorted((full.folder / "tables").iterdir()):
        assert path.read_bytes() == (capped.folder / "tables" / path.name).read_bytes(), path.name
    assert len(capped.figures.heatmap.marks["rows"]) == 2 and capped.figures.heatmap.marks["fovs"] == 4
    assert sorted(set(pd.read_csv(capped.folder / "tables" / "registration.csv")["fov_id"])) == ["F1", "F2", "F3", "F4"]


def test_a_hand_written_sample_of_64_fovs(keep):
    """Every figure drawn from the tables of 64 FOVs: the text size, the slot and no overlap."""
    fovs = fov_names(64)
    rng = np.random.default_rng(9)
    times = {name: {f: float(rng.uniform(1, 600)) for f in fovs} for name in FOV_STEPS}
    times |= {name: {"sample": float(rng.uniform(1, 2000))} for name in SAMPLE_STEPS if name not in times}
    sample = [n for n in SAMPLE_STEPS if n not in FOV_STEPS]
    figures = {
        "time": _figures.time_per_step_figure(times, list(FOV_STEPS), sample, ["segment"]),
        "heatmap": _figures.residual_heatmap_figure(registration_rows(fovs, ("round1", "round2", "round3", "round4")),
                                                    fovs, ["round1", "round2", "round3", "round4"]),
        "reads_along_z": _figures.reads_along_z_figure(along_rows(fovs, 66), {f: 50 for f in fovs}),
        "distribution:spots": _figures.distribution_figure(list(rng.integers(4000, 98000, 64)), "spots",
                                                           "spots per FOV", integer=True),
        "distribution:matched_fraction": _figures.distribution_figure(list(rng.uniform(0.8, 0.95, 64)), "matched",
                                                                      "fraction matched to the codebook",
                                                                      integer=False),
        "per_cell:reads": _figures.histogram_figure(list(rng.poisson(9, 8000)), "reads", "reads per cell",
                                                    "molecules", integer=True, slot=HALF),
        "per_cell:size": _figures.histogram_figure(list(rng.gamma(4, 150, 8000)), "size:µm³", "cell size", "µm³",
                                                   integer=False),
    }
    for name, figure in figures.items():
        check_text_size(name, figure)
        check_slot(name, figure)
        assert overlaps(figure) == [], (name, overlaps(figure)[:3])
    time = figures["time"]
    assert time.width == 2 * HALF and time.height >= time_height_at_least(len(FOV_STEPS) + len(sample))


# --- No overlap: the intensity panels and "Time per step" ---------------------------------------------------------

def hand_replay(states, shape=(80, 64), channels=("ch00", "ch01", "ch02", "ch03")):
    """A replay of one FOV written here: the reference round with ``states`` (step names), each channel a random
    image, its intensity rows, and a moving round shifted by two pixels."""
    rng = np.random.default_rng(3)
    signals = [[rng.gamma(2, 50, shape) for _ in channels] for _ in states]
    rows = []
    for j, signal in enumerate(signals):
        for c, image in zip(channels, signal):
            counts, edges = np.histogram(image, bins=64, range=(0, 65535 if j == 0 else 255))
            rows += [("round1", j, "steps", states[j], c, k, edges[k], edges[k + 1], int(counts[k]))
                     for k in range(64)]
    reference = SimpleNamespace(
        name="round1", channels=list(channels), rows=rows, before=np.max(signals[-1], axis=0),
        states=[SimpleNamespace(step=s, step_index=j, phase="steps", signal=signals[j]) for j, s in enumerate(states)])
    after = reference.before.copy()
    moving = SimpleNamespace(name="round2", before=np.roll(after, 2, axis=1), after=after, transform_missing=None)
    return SimpleNamespace(reference="round1", moving=["round2"], fov_id="F1",
                           rounds={"round1": reference, "round2": moving})


@pytest.mark.parametrize("states", [("input", "min_max_normalization"),
                                    ("input", "min_max_normalization", "white_tophat", "gaussian_smoothing")])
def test_intensity_panels_do_not_overlap(keep, states):
    figure = _figures.intensity_figure(hand_replay(states))
    assert (figure.width, figure.height) == INTENSITY_SIZE and figure.slot == FULL
    check_text_size("intensity", figure)
    assert overlaps(figure) == [], overlaps(figure)[:3]
    # One count axis per panel: each panel has its own count labels, left of its own frame and right of the panel
    # before it.
    panels = figure.frames
    assert len(panels) == len(states)
    for k, frame in enumerate(panels):
        counts = [t for t in figure.texts if t["axes"] == k and t["box"][2] <= frame[0] and t["text"] != "voxels"]
        assert counts, k
        if k:
            assert min(t["box"][0] for t in counts) > panels[k - 1][2]


def test_time_per_step_with_the_steps_of_a_run_record(keep):
    rng = np.random.default_rng(4)
    names = [n for n in FOV_STEPS]
    times = {n: {f: float(rng.uniform(0.5, 300)) for f in fov_names(4)} for n in names}
    figure = _figures.time_per_step_figure(times, names, [], [])
    assert figure.width == 2 * HALF and figure.slot == HALF
    assert figure.height >= time_height_at_least(len(names))
    check_text_size("time", figure)
    assert overlaps(figure) == []
    labels = [t["text"] for t in figure.texts if t["axes"] == 0]
    assert all(n in labels for n in names)


# --- The tiles: gaps and the zoom mark ---------------------------------------------------------------------------

def white(pixels):
    return bool(np.all(pixels == np.array(WHITE, dtype=np.uint8)))


def check_gaps(figure):
    """Between two tiles side by side every pixel over the tiles' height is white, the gap at least 8 px as shown;
    between two tiles one above the other, at least 8 px as shown of white rows over the tile's width."""
    canvas, boxes = figure.canvas, [tuple(int(round(v)) for v in b) for b in figure.marks["tiles"]]
    scale = shown(figure)
    pairs = 0
    for a in boxes:
        for b in boxes:
            if a[1] == b[1] and b[0] > a[2] and not any(c[1] == a[1] and a[2] <= c[0] < b[0] for c in boxes):
                assert white(canvas[a[1]:a[3], a[2]:b[0]]), (a, b)
                assert (b[0] - a[2]) * scale >= TILE_GAP_PX
                pairs += 1
            if a[0] == b[0] and b[1] > a[3] and not any(c[0] == a[0] and a[3] <= c[1] < b[1] for c in boxes):
                rows = [white(canvas[y, a[0]:a[2]]) for y in range(a[3], b[1])]
                run = best = 0
                for r in rows:
                    run = run + 1 if r else 0
                    best = max(best, run)
                assert best * scale >= TILE_GAP_PX, (a, b)
                pairs += 1
    return pairs


def check_declared_sizes(text):
    """Every ``<img>`` of a page declares the width and height its image decodes to."""
    from PIL import Image
    images = re.findall(r"<img [^>]*>", text)
    assert images
    for tag in images:
        attrs = dict(re.findall(r'([a-z-]+)="([^"]*)"', tag))
        data = base64.b64decode(re.fullmatch(r"data:image/(?:png|jpeg);base64,(.+)", attrs["src"]).group(1))
        assert Image.open(io.BytesIO(data)).size == (int(attrs["width"]), int(attrs["height"])), attrs["data-figure"]


def test_declared_size_is_the_drawn_size(keep):
    """A figure's recorded size is the size of its drawn canvas and of its saved image, also for the heights whose
    size in inches (h / 100) is a float just below h, which Agg draws h − 1 px high (the whole-sample view of the
    block report was 1,733 declared and 1,732 drawn)."""
    from PIL import Image
    for height in list(range(1700, 1800)) + [1733]:
        for form in ("jpeg", "png"):
            figure = _figures._figure(160, height)
            figure.text(0.5, 0.5, "x")
            image = _figures._finish(figure, form, 80)
            decoded = Image.open(io.BytesIO(image.data)).size
            assert decoded == (image.width, image.height) == image.canvas.shape[1::-1], (height, form)


def test_glyph_runs_are_measured(keep):
    """The text records measure each glyph run: a mathtext exponent counts at its own size (0.7 of the text's), so
    that a "10^7" label at 12 px as shown measures 8.4 px; the log axes of the figures draw no such label."""
    figure = _figures._figure(200, 100)
    figure.text(0.1, 0.5, r"$\mathdefault{10^{7}}$", fontsize=_figures._pt(24))
    image = _figures._finish(figure, "png", 100)
    assert image.texts[0]["nominal_px"] == pytest.approx(24, abs=1e-9)
    assert image.texts[0]["size_px"] * shown(image) == pytest.approx(0.7 * 12, abs=1e-9)
    assert smallest_text(image) == pytest.approx(8.4, abs=1e-9)
    assert [_figures.count_label(v) for v in (1, 10, 100, 1e3, 1e4, 1e5, 1e6, 1e7, 1e9)] == [
        "1", "10", "100", "1k", "10k", "100k", "1M", "10M", "1G"]
    replay = hand_replay(("input", "min_max_normalization"))
    intensity = _figures.intensity_figure(replay)
    from matplotlib import cbook
    assert not any(cbook.is_math_text(t["text"]) for t in intensity.texts)
    assert {"10", "100", "1k"} <= {t["text"] for t in intensity.texts}
    check_text_size("intensity", intensity)


def check_jpeg(figure):
    from PIL import Image
    image = Image.open(io.BytesIO(figure.data))
    assert image.format == "JPEG" and image.size == (figure.width, figure.height) == figure.canvas.shape[1::-1]


def test_preprocessing_grid_gaps(keep):
    replay = hand_replay(("input", "min_max_normalization", "white_tophat"))
    figure = _figures.preprocessing_figure(replay)
    assert figure.width == tiles_width(3, preprocessing_tile(3)) and figure.slot == FULL
    assert check_gaps(figure) == 2 * 4 + 3 * 3  # two gaps in each of four rows, three in each of three columns
    check_text_size("preprocessing", figure)
    assert overlaps(figure) == []
    check_jpeg(figure)


def yellow_sides(canvas, inner, image=None, line_px=0):
    """Per side of a region (x0, y0, x1, y1, figure pixels), at the middle of its part inside ``image`` (the box of
    the image the region is drawn on): how far from the region's edge the yellow line begins outwards (0 or 1 drawn
    pixel: within 1 px) and how wide it is (drawn pixels); None when no yellow line begins within 1 px of the edge.
    A side within ``line_px`` of the image's edge, or beyond it, is drawn inside the image: its line is looked for
    from the image's edge inwards."""
    mask = np.all(canvas == np.array(YELLOW, dtype=np.uint8), axis=-1)
    x0, y0, x1, y1 = inner
    if image is not None:
        cx, cy = int((max(x0, image[0]) + min(x1, image[2])) / 2), int((max(y0, image[1]) + min(y1, image[3])) / 2)
    else:
        cx, cy = int((x0 + x1) / 2), int((y0 + y1) / 2)
    out = {}
    for side, line, edge, step in (("top", mask[:, cx], y0, -1), ("bottom", mask[:, cx], y1, 1),
                                   ("left", mask[cy, :], x0, -1), ("right", mask[cy, :], x1, 1)):
        bound = None if image is None else image[{"top": 1, "bottom": 3, "left": 0, "right": 2}[side]]
        if bound is not None and (edge - bound) * -step <= line_px:
            k, width = int(round(bound)) + (0 if step < 0 else -1), 0
            while line[k]:
                width += 1
                k -= step
            out[side] = (0, width) if width else None
            continue
        # The first pixel outside the region: row/column edge − 1 going up or left, edge going down or right.
        first = int(round(edge)) - 1 if step < 0 else int(round(edge))
        begin = next((k for k in (first - step, first, first + step) if line[k]), None)
        if begin is None:
            out[side] = None
            continue
        k, width = begin, 0
        while line[k]:
            width += 1
            k += step
        out[side] = (abs(begin - first), width)
    return out


def check_ring(figure, inner, label, image=None):
    line = figure.marks.get("line_px", 0)
    for side, found in yellow_sides(figure.canvas, inner, image, line).items():
        assert found is not None, (label, side)
        distance, width = found
        assert distance <= 1 and width * shown(figure) >= MARK_PX, (label, side, distance, width)


def test_registration_example_gaps_and_zoom_mark(keep):
    replay = hand_replay(("input", "min_max_normalization"))
    figure = _figures.registration_figure(replay)
    assert figure.width == tiles_width(3, EXAMPLE_TILE) and figure.slot == FULL
    assert check_gaps(figure) == 2
    check_text_size("registration", figure)
    assert overlaps(figure) == []
    check_jpeg(figure)
    # The region the third tile enlarges, written here: rows 30 to 50 and columns 24 to 40 of the 80 × 64 tile
    # image (its central quarter), shown at 512 / 80 = 6.4 px per pixel and centred in X in the 512 px tile.
    gap = tile_gap(3, EXAMPLE_TILE)
    top = figure.marks["tiles"][0][1]
    k = EXAMPLE_TILE / 80
    left = (EXAMPLE_TILE - 64 * k) / 2
    for j in range(2):
        x = j * (EXAMPLE_TILE + gap)
        inner = (x + left + 24 * k, top + 30 * k, x + left + 40 * k, top + 50 * k)
        assert figure.marks["zoom"][j] == pytest.approx(inner, abs=1e-6)
        check_ring(figure, inner, j)
    # The third tile has no mark.
    third = [int(v) for v in figure.marks["tiles"][2]]
    assert not np.all(figure.canvas[third[1]:third[3], third[0]:third[2]] == np.array(YELLOW, np.uint8),
                      axis=-1).any()


def hand_previews():
    rng = np.random.default_rng(5)
    previews = {}
    for k, role in enumerate(("example_1", "example_2")):
        labels = np.zeros((64, 64), int)
        labels[10:30, 10:30], labels[35:60, 30:62] = 1, 2
        outline = np.zeros_like(labels, bool)
        outline[1:-1, 1:-1] = labels[1:-1, 1:-1] != labels[:-2, 1:-1]
        previews[role] = SimpleNamespace(background=rng.gamma(2, 40, (64, 64)), cell_outline=outline,
                                         nucleus_outline=outline[::-1], extent=(-0.5, 63.5, 63.5, -0.5),
                                         label_extent=(-0.5, 63.5, 63.5, -0.5), fov_id=f"Position00{k + 1}",
                                         route="per_fov")
    previews["example_3"] = SimpleNamespace(reason="no_segmentation")
    return previews


def test_segmentation_examples_gaps(keep):
    figure = _figures.segmentation_figure(hand_previews(), reasons=_page.REASON_TEXTS)
    assert figure.width == tiles_width(3, EXAMPLE_TILE) and figure.slot == FULL
    assert check_gaps(figure) == 2
    check_text_size("segmentation", figure)
    assert overlaps(figure) == []
    check_jpeg(figure)


# --- The regions of the whole-sample marker view -----------------------------------------------------------------

def test_whole_view_regions(tmp_path, keep):
    """The whole-sample view of report_cells' molecules, footprints and example FOVs over a fused image written here
    (report_cells has no fused image): three yellow rectangles, each number readable on a dark backing inside its own
    rectangle, no two numbers over each other."""
    page = collect_drawn("report_cells", None, tmp_path / "root")
    export = page.case.root / "export" / page.case.sample / "tables"
    molecules = pd.read_csv(export / "molecules.csv")
    fovs = pd.read_csv(export / "fovs.csv").set_index("fov_id")
    roles = ("example_1", "example_2", "example_3")
    regions = [(role, page.record["examples"][role]["fov_id"],
                tuple(float(fovs.loc[page.record["examples"][role]["fov_id"], c])
                      for c in ("footprint_y0", "footprint_y1", "footprint_x0", "footprint_x1"))) for role in roles]
    assert len({r[1] for r in regions}) == 3
    spacing = 0.025
    image = np.random.default_rng(6).gamma(2, 50, (int(7.5 / spacing), int(7.5 / spacing)))
    view = SimpleNamespace(image="reference_stain", whole=image,
                           whole_grid=((0.0, -0.25, -0.25), (1.0, spacing, spacing)))
    group = GroupFigureData("g", "first", ("G1", "G2"), ("#E69F00", "#56B4E9"))
    retained = molecules["assembly_status"].eq("retained").to_numpy(bool)
    figure = _figures._whole_view(group, view, molecules, retained, regions, 1600, 85)
    assert figure.width == 1600 and figure.slot == FULL
    check_text_size("whole", figure)
    assert overlaps(figure) == []
    check_jpeg(figure)
    numbers = figure.marks["regions"]
    assert [n["k"] for n in numbers] == [1, 2, 3]
    canvas = figure.canvas
    # The image spans the pixel edges −0.25 − S/2 to −0.25 + (n − ½)·S µm over the image's box.
    left, top, right, bottom = figure.marks["image"]
    lo, hi = -0.25 - spacing / 2, -0.25 + (image.shape[1] - 0.5) * spacing
    lo_y, hi_y = -0.25 - spacing / 2, -0.25 + (image.shape[0] - 0.5) * spacing
    for n, (_, _, (fy0, fy1, fx0, fx1)) in zip(numbers, regions):
        inner = n["region"]
        assert inner == pytest.approx((left + (fx0 - lo) / (hi - lo) * (right - left),
                                       top + (fy0 - lo_y) / (hi_y - lo_y) * (bottom - top),
                                       left + (fx1 - lo) / (hi - lo) * (right - left),
                                       top + (fy1 - lo_y) / (hi_y - lo_y) * (bottom - top)), abs=1e-6)
        # The rectangle: yellow, at least 2 px wide as shown, its inner edge on the footprint (within 1 px).
        check_ring(figure, inner, n["k"], figure.marks["image"])
        # The number: at least 11 px as shown, on a dark backing, inside its own rectangle.
        text = [t for t in figure.texts if t["text"] == str(n["k"]) and t["axes"] is not None]
        assert len(text) == 1 and text[0]["size_px"] * shown(figure) >= MIN_TEXT_PX
        b = n["backing"]
        assert inner[0] <= b[0] and b[2] <= inner[2] and inner[1] <= b[1] and b[3] <= inner[3], n
        tb = text[0]["box"]
        assert b[0] <= tb[0] and tb[2] <= b[2] and b[1] <= tb[1] and tb[3] <= b[3]
        corner = canvas[int(b[1]) + 1, int(b[0]) + 1].astype(int)
        assert corner.max() <= 40, corner
    for a, b in combinations(numbers, 2):
        assert not intersect(a["backing"], b["backing"]), (a["k"], b["k"])
    # No white rectangle: on each region's edge the pixels just outside are yellow, not white.
    for n in numbers:
        x0, y0, x1, y1 = (int(round(v)) for v in n["region"])
        assert tuple(canvas[y0 - 1, (x0 + x1) // 2]) == YELLOW and tuple(canvas[(y0 + y1) // 2, x0 - 1]) == YELLOW


# --- The readout bars --------------------------------------------------------------------------------------------

def readout_html(spots, cells):
    """The Sample totals of a report whose reads assigned to cells are ``cells`` of ``spots`` spots."""
    page = _page._Page.__new__(_page._Page)
    levels = ("spots", "correct_form", "matched", "merged_by_deduplication", "passed_filter", "reads_in_cells")
    counts = dict(zip(levels, (spots, int(spots * 0.9), int(spots * 0.8), 0, int(spots * 0.7), cells)))
    page.rows = {"readout_totals": [{"level": lv, "count": counts[lv], "fraction_of_spots": counts[lv] / spots,
                                     "fovs": 4} for lv in levels], "fovs": []}
    page.named, page.unavailable, page.placed = {}, [], False
    page.summary = {"reads_in_cells_equal": True, "h5ad_whole_count_sum": cells}
    page.figures = SimpleNamespace(distributions=[])
    return page.readout()["read_totals"]


def check_bars(markup, spots, counts):
    rules = css_rules(_page.template_text())
    for selector, body in rules.items():
        if ".bars" in selector:
            assert "overflow:hidden" not in body and "text-overflow" not in body and "nowrap" not in body, selector
    found = re.findall(r'<div class="(lv[^"]*)"><i style="width:([\d.]+)%"></i><span>(.*?)</span></div>', markup)
    assert len(found) == len(counts)
    for (cls, width, label), (name, count) in zip(found, counts):
        assert float(width) == pytest.approx(100 * count / spots, abs=0.005)
        assert html.unescape(label).startswith(f"{name} · {count:,} · {100 * count / spots:.1f} %")
    return found


def test_readout_bar_labels_are_not_clipped(drawn):
    markup = readout_html(10000, 100)
    found = check_bars(markup, 10000, [("Spots detected", 10000), ("Correct barcode form", 9000),
                                       ("Matched to the codebook", 8000), ("Merged by deduplication", 0),
                                       ("Passed the read QC filter", 7000), ("Reads assigned to cells", 100)])
    assert found[-1][0] == "lv cell" and found[-1][1] == "1.00"
    # report_cells: the same structure, every label whole in the page.
    page = drawn("report_cells", None)
    totals = pd.read_csv(page.folder / "tables" / "readout_totals.csv")
    spots = int(totals.loc[totals["level"].eq("spots"), "count"].iloc[0])
    names = dict(zip(("spots", "correct_form", "matched", "merged_by_deduplication", "passed_filter",
                      "reads_in_cells"), ("Spots detected", "Correct barcode form", "Matched to the codebook",
                                          "Merged by deduplication", "Passed the read QC filter",
                                          "Reads assigned to cells")))
    shown_levels = [(names[r.level], int(r.count)) for r in totals.itertuples() if not pd.isna(r.count)]
    block = page.text[page.text.index('id="b-read-totals"'):page.text.index('id="b-read-map"')]
    check_bars(block, spots, shown_levels)
    assert ' style="width:' not in block.replace('<i style="width:', "")


# --- The summary -------------------------------------------------------------------------------------------------

def facts(text):
    block = re.search(r'id="b-facts"[^>]*>(.*?)</dl>', text, re.S).group(1)
    return {html.unescape(dt): dd for dt, dd in re.findall(r"<dt>(.*?)</dt><dd[^>]*>(.*?)</dd>", block, re.S)}


def utc_text(iso):
    moment = datetime.fromisoformat(iso).astimezone(timezone.utc)
    return f"{moment:%Y-%m-%d, %H:%M:%S}"


def zone_text(iso, zone):
    from zoneinfo import ZoneInfo
    moment = datetime.fromisoformat(iso).astimezone(ZoneInfo(zone))
    return f"{moment:%Y-%m-%d, %H:%M:%S}", moment.tzname()


EXTENDED = (SHIM
            .replace("const written = {}, first = {}, elements = {};",
                     "const written = {}, first = {}, elements = {}, texts = {};")
            .replace("this.textContent = ''; this.offsetWidth = 100;", "this._text = ''; this.offsetWidth = 100;")
            .replace("Object.defineProperty(El.prototype, 'innerHTML',",
                     "Object.defineProperty(El.prototype, 'textContent', {get() { return this._text; }, set(v) {\n"
                     "  this._text = String(v); if (this.id) texts[this.id] = this._text; }});\n"
                     "Object.defineProperty(El.prototype, 'innerHTML',")
            .replace("const e = new El(id); const v = selectValue(id);",
                     "const e = new El(id); const tag = page.match(new RegExp('<[a-z]+\\\\b([^>]*\\\\sid=\"' + id + "
                     "'\"[^>]*)>'));\n      if (tag) for (const m of tag[1].matchAll(/([a-z-]+)=\"([^\"]*)\"/g)) "
                     "e.attributes[m[1]] = m[2].replace(/&quot;/g, '\"').replace(/&#x27;/g, \"'\")"
                     ".replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&amp;/g, '&');\n"
                     "      const v = selectValue(id);")
            .replace("process.stdout.write(JSON.stringify({first: first, last: written}));",
                     "process.stdout.write(JSON.stringify({first: first, last: written, texts: texts}));"))


def run_script(folder, tmp_path, zone):
    import os
    node = shutil.which("node") or shutil.which("nodejs")
    if node is None:
        pytest.skip("node is not installed: the times the page's script writes are checked by running the script")
    for old in ("const written = {}, first = {}, elements = {};", "this.textContent = ''; this.offsetWidth = 100;",
                "Object.defineProperty(El.prototype, 'innerHTML',", "const e = new El(id); const v = selectValue(id);",
                "process.stdout.write(JSON.stringify({first: first, last: written}));"):
        assert SHIM.count(old) == 1, old
    shim = tmp_path / "shim.js"
    shim.write_text(EXTENDED)
    done = subprocess.run([node, str(shim), str(folder / "report.html")], capture_output=True, text=True, check=True,
                          timeout=60, env=dict(os.environ, TZ=zone))
    return json.loads(done.stdout)["texts"]


def test_summary_rows(drawn, tmp_path):
    page = drawn("report_cells", None)
    found = facts(page.text)
    assert list(found) == ["Dataset", "Sample", "Route", "Acquisition", "Configuration", "Version", "Run time",
                           "Generated"]
    assert "starfinder" not in found
    record = page.record
    runs = [json.loads((page.case.root / "fovs" / f / "run.json").read_text()) for f in page.case.fov_ids]
    start = min((r["started_at"] for r in runs), key=datetime.fromisoformat)
    end = max((r["ended_at"] for r in runs), key=datetime.fromisoformat)
    generated = record["generated_at"]
    run, made = (re.sub(r"<[^>]+>", "", found[k]) for k in ("Run time", "Generated"))
    same_day = utc_text(start)[:10] == utc_text(end)[:10]
    assert html.unescape(run) == f"{utc_text(start)} – {utc_text(end)[12:] if same_day else utc_text(end)} UTC"
    assert html.unescape(made) == f"{utc_text(generated)} UTC"
    for value in (found["Run time"], found["Generated"]):
        assert "<br" not in value and "\n" not in html.unescape(re.sub(r"<[^>]+>", "", value))
    assert re.search(r'<dd id="run-time" class="one">', page.text) and re.search(r'<dd id="generated" class="one">',
                                                                                  page.text)
    assert "white-space:nowrap" in css_rules(page.text)[".facts dd.one"]
    # The tooltips keep the ISO times and the seconds of the run.
    tip = html.unescape(re.search(r'id="run-span"[^>]*data-t="([^"]*)"', page.text).group(1)).split("|")
    assert tip[:2] == ["Run time", f"{start} to {end}"] and tip[2].endswith(" s")
    assert html.unescape(re.search(r'id="generated-at"[^>]*data-t="([^"]*)"', page.text).group(1)).split("|") == [
        "Generated", generated]
    # The script writes both again in the viewer's time zone, with its abbreviation, on one line.
    for zone in ("America/New_York", "UTC"):
        texts = run_script(page.folder, tmp_path, zone)
        (s_day, s_time), s_zone = (lambda t: ((t[0][:10], t[0][12:]), t[1]))(zone_text(start, zone))
        (e_day, e_time), e_zone = (lambda t: ((t[0][:10], t[0][12:]), t[1]))(zone_text(end, zone))
        expected = (f"{s_day}, {s_time}" + (f" {s_zone}" if s_zone != e_zone else "") + " – " +
                    (f"{e_day}, " if e_day != s_day else "") + f"{e_time} {e_zone}")
        assert texts["run-span"] == expected, zone
        day, zone_name = zone_text(generated, zone)
        assert texts["generated-at"] == f"{day} {zone_name}", zone
        assert zone_name == ("UTC" if zone == "UTC" else zone_name) and zone_name in ("UTC", "EDT", "EST")
        assert "\n" not in texts["run-span"] + texts["generated-at"]


def test_summary_text_forms():
    assert _page.moment_text("2026-10-13T22:18:57.666071+00:00") == "2026-10-13, 22:18:57 UTC"
    assert _page.span_text("2026-10-13T18:19:02.2+00:00", "2026-10-13T18:42:12.2+00:00") == \
        "2026-10-13, 18:19:02 – 18:42:12 UTC"
    assert _page.span_text("2026-10-13T23:50:00+00:00", "2026-10-14T00:10:00+00:00") == \
        "2026-10-13, 23:50:00 – 2026-10-14, 00:10:00 UTC"
    assert _page.span_text("2026-10-13T14:19:02-04:00", None) == "2026-10-13, 18:19:02 UTC – not ended"


def test_report_json_does_not_depend_on_the_page(tmp_path):
    """report.json is the same whatever the page draws: with the page replaced by a fixed text, every key and value
    is equal but the checksum and size of report.html (and the time and resources of the run)."""
    anndata_or_skip()
    made = []
    case = report_cells(tmp_path / "root")
    for fixed in (None, "<html></html>"):
        with pytest.MonkeyPatch.context() as patch:
            if fixed is not None:
                original = _page.build_page
                patch.setattr(_page, "build_page", lambda *a, **kw: (fixed, original(*a, **kw)[1]))
            folder, _ = case.collect()
        record = json.loads((folder / "report.json").read_text())
        for key in ("generated_at", "wall_s", "cpu_s", "peak_rss_kib"):
            del record[key]
        entry = next(e for e in record["files"] if e["path"] == "report.html")
        del entry["sha256"], entry["bytes"]
        made.append(record)
    assert made[0] == made[1]


# --- Every figure of the synthetic chain's page (slow) --------------------------------------------------------------

from .report_scene import scene_chain_registered  # noqa: E402,F401


@pytest.mark.slow
def test_rules_on_the_chain_page(scene_chain_registered):  # noqa: F811
    """The page of test_page_with_every_figure (replayed examples, segmentation examples, two marker groups): every
    figure's text size, slot and layout, the tiles' gaps, the zoom mark and the regions of the whole-sample views."""
    from starfinder.report import MarkerGroup, ReportConfig
    chain = scene_chain_registered
    genes = json.loads((chain.root / "export" / chain.sample / "export.json").read_text())["assembly"]["genes"]
    config = ReportConfig(markers=(MarkerGroup("first", tuple(genes[:2])), MarkerGroup("second", tuple(genes[2:5]))),
                          marker_pairs=((genes[0], genes[1]),))
    found = {}
    original = _page.build_page

    def build(*args, **kwargs):
        text, figures = original(*args, **kwargs)
        found["figures"] = figures
        return text, figures
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_figures, "KEEP_CANVAS", True)
        patch.setattr(_page, "build_page", build)
        chain.collect(config=config)
    named = dict(page_figures(found["figures"]))
    assert {"preprocessing", "intensity", "registration", "segmentation", "marker:0:whole", "marker:1:whole"} <= \
        set(named)
    for name, figure in named.items():
        check_text_size(name, figure)
        check_slot(name, figure)
        assert overlaps(figure) == [], (name, overlaps(figure)[:3])
        if figure.format == "jpeg":
            check_jpeg(figure)
    assert check_gaps(named["preprocessing"]) > 0
    assert check_gaps(named["registration"]) == check_gaps(named["segmentation"]) == 2
    for j, inner in enumerate(named["registration"].marks["zoom"]):
        check_ring(named["registration"], inner, j)
    for k in (0, 1):
        whole = named[f"marker:{k}:whole"]
        numbers = whole.marks["regions"]
        assert [n["k"] for n in numbers] == [1, 2, 3]
        for n in numbers:
            check_ring(whole, n["region"], n["k"], whole.marks["image"])
        for a, b in combinations(numbers, 2):
            assert not intersect(a["backing"], b["backing"])
