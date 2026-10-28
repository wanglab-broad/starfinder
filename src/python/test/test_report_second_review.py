"""The figures and the page after Jiahao's second review of the real block report (W-478).

Checks of design/specs/report-algorithms.md ("Layout of the page", "Workflow overview", "Preprocessing",
"Registration", "Segmentation and assembly", "Sample measurements", "Figure sizes"): the page
header is the heading "Processing report" alone (the header); the axes of "Time per step" span,
as shown, the FOV rectangles of the FOV status map beside it, which the page's inline script
draws (the alignment); no static figure is shown wider than two thirds of its drawn width (no
figure shown enlarged); and every legend of a static figure is a row below its plots or tiles,
at the figure's left margin, over nothing (the legends).

The map is the one the page's script draws, run under ``node`` with the page tests' document
stand-in (the test skips without node); its geometry is read from the SVG the script writes.
The figures are checked as drawn (``_figures.KEEP_CANVAS``). The numbers are written here from
the page (``report_sizes``); nothing is read from a run directory and no image is committed.
"""
from __future__ import annotations

import html
import json
import re
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from starfinder.report import _figures, _page
from starfinder.report._measurements import GroupFigureData

from . import report_fixtures
from .report_fixtures import CASES, IDS, collect_drawn
from .report_sizes import FIGURES, FULL, HALF, MIN_TEXT_PX, SHOWN_AT_MOST, THIRD, preprocessing_tile, tiles_width
from .test_report_figure_rules import (FOV_STEPS, SAMPLE_STEPS, check_slot, check_text_size, css_rules,
                                       hand_previews, hand_replay, intersect, overlaps, page_figures, shown)
from .test_report_page import SHIM

pytestmark = [pytest.mark.report, pytest.mark.validation]
# The shared fixtures of report_fixtures.py: the pages drawn once for the session, and KEEP_CANVAS on.
drawn, keep = report_fixtures.drawn, report_fixtures.keep

#: Px as shown per step name of "Time per step", and the tolerance of the alignment, px as shown.
STEP_ROW_PX, ALIGN_PX = 26, 4
#: The inset of a figure's texts from its tiles' left edge, px as shown (titles, ranges and legend).
INSET_PX = 4


# --- The header -----------------------------------------------------------------------------------------------------

def test_header(drawn):
    """The page header holds the heading "Processing report" and nothing else; the Summary's facts name the dataset
    and the sample as before; the page's <title> is unchanged."""
    page = drawn("report_cells", None)
    header = re.search(r'<div class="head">(.*?)</div>', page.text, re.S).group(1)
    assert header == "<h1>Processing report</h1>"
    assert re.findall(r"<title>(.*?)</title>", page.text) == [html.escape(f"Processing report · {page.case.sample}")]
    facts = dict(re.findall(r"<dt>(.*?)</dt><dd[^>]*>(.*?)</dd>",
                            re.search(r'id="b-facts"[^>]*>(.*?)</dl>', page.text, re.S).group(1), re.S))
    assert facts["Dataset"] == f"<code>{html.escape(page.record['dataset_id'])}</code>"
    assert facts["Sample"] == f"<code>{html.escape(page.case.sample)}</code>"
    # The comparison page keeps its own heading line (its CSS rule is shared).
    assert ".head .sub" in css_rules(_page.template_text())


# --- "Time per step" aligned with the FOV status map ------------------------------------------------------------

def run_script(path, tmp_path):
    node = shutil.which("node") or shutil.which("nodejs")
    if node is None:
        pytest.skip("node is not installed: the FOV status map is the one the page's script draws")
    shim = tmp_path / "shim.js"
    shim.write_text(SHIM)
    done = subprocess.run([node, str(shim), str(path)], capture_output=True, text=True, check=True, timeout=60)
    return json.loads(done.stdout)["first"]


def map_rectangles(markup):
    """The top and bottom, px as shown in the half column, of the FOV rectangles of the map the script wrote: the
    SVG is as wide as the half column, at most the max-width of ``.map svg``, and as high as its viewBox gives."""
    vx, vy, vw, vh = (float(v) for v in re.search(r'<svg viewBox="([^"]*)"', markup).group(1).split())
    rects = [dict(re.findall(r'([a-z-]+)="([^"]*)"', attrs)) for attrs in re.findall(r"<rect ([^>]*)>", markup)]
    assert rects
    top = min(float(r["y"]) for r in rects)
    bottom = max(float(r["y"]) + float(r["height"]) for r in rects)
    width = min(HALF, float(re.search(r"max-width:(\d+)px", css_rules(_page.template_text())[".map svg"]).group(1)))
    k = width / vw
    return (top - vy) * k, (bottom - vy) * k


def hand_page(fovs, folder):
    """A page of the shipped template holding only the FOV status map of the rows ``fovs`` (the data block's other
    tables empty)."""
    data = {name: [] for name in _page.DATA_TABLES}
    data["fovs"] = fovs
    values = {name: "" for name in _page.SLOTS}
    values.update(title="hand", heading="<h1>Processing report</h1>", data=json.dumps(data),
                  fov_status='<div id="m-status"></div><div class="lg one" id="lg-status"></div>')
    path = Path(folder) / "report.html"
    path.write_text(_page.fill(_page.template_text(), values), encoding="utf-8")
    return path


def grid_fovs(rows, columns, pitch=300.0, half=160.0):
    """FOVs on a grid of ``rows`` × ``columns``, ``pitch`` µm apart, each with a footprint of ``2 · half`` µm."""
    out = []
    for i in range(rows):
        for j in range(columns):
            cx, cy = 100.0 + pitch * j, 50.0 + pitch * i
            out.append({"fov_id": f"Position{len(out) + 1:03d}", "run_record": True, "status": "succeeded",
                        "run_seconds": 10.0, "centre_x": cx, "centre_y": cy, "footprint_x0": cx - half,
                        "footprint_x1": cx + half, "footprint_y0": cy - half, "footprint_y1": cy + half})
    return out


def step_times(fovs, names, sample):
    rng = np.random.default_rng(11)
    times = {name: {f["fov_id"]: float(rng.uniform(0.5, 600)) for f in fovs} for name in names}
    times |= {name: {"sample": float(rng.uniform(1, 2000))} for name in sample}
    return times


def step_labels(figure):
    """The step names of "Time per step": the labels left of its axes' frame, top to bottom."""
    frame = figure.frames[0]
    labels = [t for t in figure.texts if t["axes"] == 0 and t["box"][2] <= frame[0]]
    return [t["text"] for t in sorted(labels, key=lambda t: t["box"][1])]


def axes_shown(figure):
    """The top and bottom of the plot's axes as shown (its frame times the slot over the figure's width)."""
    frame = figure.frames[0]
    return frame[1] * shown(figure), frame[3] * shown(figure)


def check_alignment(figure, rectangles, names):
    """The axes start on the map's top line; they end on its bottom line where 26 px per step name fit, else they
    are 26 px per step name high."""
    (top, bottom), (map_top, map_bottom) = axes_shown(figure), rectangles
    assert abs(top - map_top) <= ALIGN_PX, (top, map_top)
    if STEP_ROW_PX * names <= map_bottom - map_top:
        assert abs(bottom - map_bottom) <= ALIGN_PX, (bottom, map_bottom)
    else:
        assert bottom - top >= STEP_ROW_PX * names - 1e-9, (bottom - top, names)
    check_text_size("time", figure)
    check_slot("time", figure)
    assert overlaps(figure) == [], overlaps(figure)[:3]


SAMPLE_ONLY = tuple(n for n in SAMPLE_STEPS if n not in FOV_STEPS)


@pytest.mark.parametrize("shape", [(8, 8), (1, 4), (2, 2)], ids=["8x8", "1x4", "2x2"])
@pytest.mark.parametrize("sample", [(), SAMPLE_ONLY], ids=["fov-steps", "all-steps"])
def test_time_per_step_aligned_with_a_hand_written_map(tmp_path, keep, shape, sample):
    """FOVs on a grid of 8 × 8, a row of 1 × 4 and a grid of 2 × 2, with the step names of a workflow run record (the
    per-FOV steps alone, 13 names, and with the sample steps, 18): the map the page's script draws from the rows, the
    figure drawn from the same rows."""
    fovs = grid_fovs(*shape)
    rectangles = map_rectangles(run_script(hand_page(fovs, tmp_path), tmp_path)["m-status"])
    assert _figures.map_extent(fovs) == pytest.approx(rectangles, abs=1e-3)
    names = list(FOV_STEPS) + list(sample)
    figure = _figures.time_per_step_figure(step_times(fovs, FOV_STEPS, sample), list(FOV_STEPS), list(sample), [],
                                           fovs)
    check_alignment(figure, rectangles, len(names))
    fits = STEP_ROW_PX * len(names) <= rectangles[1] - rectangles[0]
    assert fits == {(8, 8): not sample, (1, 4): False, (2, 2): not sample}[shape]


def test_time_per_step_aligned_on_the_page(drawn, tmp_path):
    """report_cells (FOVs on a grid of 2 × 2): the figure of the page against the map its script draws."""
    page = drawn("report_cells", None)
    rectangles = map_rectangles(run_script(page.folder / "report.html", tmp_path)["m-status"])
    figure = page.figures.time
    steps = pd.read_csv(page.folder / "tables" / "steps.csv")
    timed = steps[steps["seconds"].notna()]
    labels = step_labels(figure)
    assert set(labels) == set(timed["name"]) and len(labels) > 1
    check_alignment(figure, rectangles, len(labels))
    # The FOV status block and Time per step are the two halves of one row.
    row = re.search(r'<div class="g2 split">\s*<div class="blk" id="b-fov-status".*?id="b-time"', page.text, re.S)
    assert row is not None


def test_time_per_step_without_placement(tmp_path):
    """report_records without positions: no map, and the figure as W-475 drew it (at least 280 px high as shown, its
    axes filling it less the margins): the bytes the figure has when it is drawn without the rows."""
    calls = []
    original = _figures.time_per_step_figure

    def record(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)
    page = collect_drawn("report_records", "no_placement", tmp_path / "root", time_per_step_figure=record)
    assert 'id="m-status"' not in page.text and 'id="t-status"' in page.text
    figure = page.figures.time
    assert len(calls) == 1 and _figures.map_extent(calls[0][4]) is None
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_figures, "KEEP_CANVAS", True)
        alone = original(*calls[0][:4])
    assert figure.data == alone.data and "map" not in figure.marks
    assert figure.height * shown(figure) >= 280 and figure.height >= 2 * STEP_ROW_PX * len(step_labels(figure))


# --- No figure shown enlarged --------------------------------------------------------------------------------------

def check_not_enlarged(name, figure):
    assert figure.slot / figure.width <= SHOWN_AT_MOST + 1e-12, (name, figure.slot, figure.width)
    if figure.format == "png":
        assert figure.width == 2 * figure.slot, name


@pytest.mark.parametrize("maker,variant", CASES, ids=IDS)
def test_no_figure_shown_enlarged_on_every_page(drawn, maker, variant):
    page = drawn(maker, variant)
    for name, figure in page_figures(page.figures):
        check_not_enlarged(name, figure)
        check_text_size(name, figure)
        check_slot(name, figure)


@pytest.mark.parametrize("columns", [1, 2, 3, 4, 5, 6, 8])
def test_preprocessing_grid_is_drawn_larger(keep, columns):
    """The Preprocessing grid of 1 to 8 states: tiles of 256 px or the smallest size whose grid is at least 3/2 of the
    full column wide (1,398 px), so that it is shown at no more than two thirds of its drawn width."""
    states = ("input",) + tuple(f"step_{k}" for k in range(1, columns))
    figure = _figures.preprocessing_figure(hand_replay(states, channels=("ch00", "ch01")))
    tile = preprocessing_tile(columns)
    assert figure.width == tiles_width(columns, tile) and figure.slot == FULL
    assert figure.width >= 1398 and (tile == 256 or tiles_width(columns, tile - 1) < 1398)
    assert {2: 693, 3: 458}.get(columns, tile) == tile
    check_not_enlarged("preprocessing", figure)
    check_text_size("preprocessing", figure)
    assert overlaps(figure) == []


def test_comparison_page_not_enlarged(tmp_path, keep):
    from starfinder.report._compare import compare_samples

    from .report_fixtures import report_records
    case = report_records(tmp_path / "root")
    case.collect()
    made = []
    original = _figures.sample_box_figure

    def box(*args, **kwargs):
        made.append(original(*args, **kwargs))
        return made[-1]
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_figures, "sample_box_figure", box)
        compare_samples(case.root / "report", [case.sample])
    assert len(made) == 3
    for figure in made:
        check_not_enlarged("box", figure)


# --- The legends below ----------------------------------------------------------------------------------------------

def check_legend_below(name, figure, labels):
    """The legend (its recorded box and the texts on its row) is below every axes and tile of the figure, its left edge
    at the figure's left margin (within the inset of the figure's texts), inside the canvas and over nothing."""
    x0, y0, x1, y1 = figure.marks["legend"]
    row = [t for t in figure.texts if y0 - 1 <= (t["box"][1] + t["box"][3]) / 2 <= y1 + 1]
    assert sorted(t["text"] for t in row) == sorted(labels), name
    box = (min([x0] + [t["box"][0] for t in row]), min([y0] + [t["box"][1] for t in row]),
           max([x1] + [t["box"][2] for t in row]), max([y1] + [t["box"][3] for t in row]))
    others = [t for t in figure.texts if t not in row]
    tiles = list(figure.marks.get("tiles", [])) + ([figure.marks["image"]] if "image" in figure.marks else [])
    content = list(figure.frames) + tiles
    assert box[1] >= max(b[3] for b in content), (name, box, max(b[3] for b in content))
    assert not any(intersect(box, b) for b in content + [t["box"] for t in others]), name
    assert box[1] >= max(t["box"][3] for t in others), name
    assert 0 <= box[0] and box[2] <= figure.width and box[3] <= figure.height, name
    left = min([b[0] for b in content] + [t["box"][0] for t in others])
    assert left - 0.5 <= x0 <= left + INSET_PX / shown(figure) + 0.5, (name, x0, left)
    return box


def test_legends_of_the_hand_written_figures(keep):
    replay = hand_replay(("input", "min_max_normalization", "white_tophat", "gaussian_smoothing"))
    figures = {"intensity": (_figures.intensity_figure(replay), ["ch00", "ch01", "ch02", "ch03"]),
               "registration": (_figures.registration_figure(replay),
                                ["reference round", "moving round", "overlap"]),
               "segmentation": (_figures.segmentation_figure(hand_previews(), reasons=_page.REASON_TEXTS),
                                ["nucleus outline", "cell outline"])}
    for name, (figure, labels) in figures.items():
        check_legend_below(name, figure, labels)
        check_text_size(name, figure)
        assert overlaps(figure) == [], (name, overlaps(figure)[:3])
    # The other figures of the examples draw no legend.
    assert "legend" not in _figures.preprocessing_figure(replay).marks


def test_legend_of_the_whole_sample_view(tmp_path, keep):
    """The whole-sample view of report_cells' molecules over a fused image written here: the genes and the image name
    in a row below the image and its grey bar."""
    page = collect_drawn("report_cells", None, tmp_path / "root")
    export = page.case.root / "export" / page.case.sample / "tables"
    molecules = pd.read_csv(export / "molecules.csv")
    image = np.random.default_rng(6).gamma(2, 50, (300, 300))
    view = SimpleNamespace(image="reference_stain", whole=image, whole_grid=((0.0, -0.25, -0.25), (1.0, 0.025, 0.025)))
    group = GroupFigureData("g", "first", ("G1", "G2"), ("#E69F00", "#56B4E9"))
    retained = molecules["assembly_status"].eq("retained").to_numpy(bool)
    figure = _figures._whole_view(group, view, molecules, retained, [], 1600, 85)
    check_legend_below("whole", figure, ["G1", "G2", "reference_stain"])
    check_text_size("whole", figure)
    assert overlaps(figure) == []


def test_figures_table_equals_the_constants():
    """The table of "Figure sizes", as ``report_sizes.FIGURES`` writes it from the page, states each figure's slot,
    shown width and drawn width; they equal the code's constants, and every shown width is at most two thirds of the
    drawn width. ``test_report_docs_contract.py`` checks that the page's table is ``FIGURES`` (W-602)."""
    table = FIGURES
    slots = {"full": [FULL], "half": [HALF], "third": [THIRD], "third or half": [THIRD, HALF], "comparison": [1160]}
    for name, (slot, shown_w, drawn) in table.items():
        assert shown_w == slots[slot], name
    assert table["Time per step"][2][:1] == [2 * _figures.HALF_COLUMN]
    assert table["Intensity after each step"][2][:1] == [2 * _figures.FULL_COLUMN]
    assert table["All FOVs and rounds"][2][:1] == [2 * _figures.FULL_COLUMN]
    assert table["Reads along Z"][2][:1] == [2 * _figures.HALF_COLUMN]
    assert table["Distributions across FOVs and per cell"][2][:2] == [2 * _figures.THIRD_COLUMN,
                                                                      2 * _figures.HALF_COLUMN]
    assert table["Box plots by sample (`index.html`)"][2][:1] == [2 * _figures.COMPARISON_COLUMN]
    assert table["Preprocessing example grid"][2][:2] == [1398, _figures.PREPROCESSING_TILE]
    assert tiles_width(2, preprocessing_tile(2)) == 1398
    assert table["Registration and segmentation examples"][2][:1] == [tiles_width(3, _figures.EXAMPLE_TILE)]
    from starfinder.report import ReportConfig
    assert table["Whole-sample marker view"][2][:1] == [ReportConfig().figure_width_px]
    assert table["Marker zoom-in tile, XY and XZ"][2][:1] == [_figures.EXAMPLE_TILE]
    for name, (slot, shown_w, drawn) in table.items():
        for width, slot_width in zip(drawn[:len(slots[slot])], slots[slot]):
            assert slot_width / width <= SHOWN_AT_MOST + 1e-12, name
    assert len(table) == 10
    assert _figures.SHOWN_AT_MOST == (2, 3)
    assert (_figures.MAP_MAX_WIDTH, MIN_TEXT_PX) == (480, 11)


# --- Every figure of the synthetic chain's page (slow) --------------------------------------------------------------

from .report_scene import scene_chain_registered  # noqa: E402,F401


@pytest.mark.slow
def test_second_review_on_the_chain_page(scene_chain_registered, tmp_path):  # noqa: F811
    """The page with every figure (replayed examples, segmentation examples, two marker groups): no figure shown
    enlarged, every legend below its figure, Time per step aligned with the map the script draws."""
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
        folder, _ = chain.collect(config=config)
    named = dict(page_figures(found["figures"]))
    legends = {"intensity", "registration", "segmentation", "marker:0:whole", "marker:1:whole"}
    assert legends <= set(named)
    for name, figure in named.items():
        check_not_enlarged(name, figure)
        assert ("legend" in figure.marks) == (name in legends), name
    replay = found["figures"].replay
    channels = [t["channel"] for t in named["preprocessing"].tiles if t["step_index"] == 0]
    check_legend_below("intensity", named["intensity"], channels)
    check_legend_below("registration", named["registration"], ["reference round", "moving round", "overlap"])
    check_legend_below("segmentation", named["segmentation"], ["nucleus outline", "cell outline"])
    for k, group in enumerate(config.markers):
        whole = named[f"marker:{k}:whole"]
        check_legend_below(f"marker:{k}:whole", whole, list(group.genes) + [whole.tiles[0]["image"]])
    assert replay is not None
    text = (folder / "report.html").read_text(encoding="utf-8")
    if 'id="m-status"' in text:
        rectangles = map_rectangles(run_script(folder / "report.html", tmp_path)["m-status"])
        check_alignment(named["time"], rectangles, len(step_labels(named["time"])))
