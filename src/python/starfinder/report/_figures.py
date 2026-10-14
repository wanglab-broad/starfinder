"""The static figures of the report page (docs/report-algorithms.md, "Workflow overview", "Preprocessing",
"Registration", "Readout", "Segmentation and assembly", "Sample measurements", "The replay of the examples",
"Figure sizes").

Drawn with matplotlib's ``Agg`` canvas at 100 dpi and saved to memory, with the bundled
DejaVu Sans font: the image figures as JPEG at the configuration's quality, the plots as PNG
(whose metadata holds no time). Each figure is drawn for its slot, the width at which the page
shows it ("Figure sizes"): the full column (932 px), a half column (442 px), a third (300 px)
or the comparison page's column (1,160 px). A plot is drawn at twice the width of its slot;
an image figure keeps its tiles (256 px per tile in the Preprocessing grid, 512 px in the
Registration and segmentation examples and the marker zoom-in tiles, ``figure_width_px`` for
the whole-sample view). Every text is written so that it is at least 11 px as shown (12 px,
titles 13 px), every layout is measured so that no text overlaps another, another axes or
the edge, and two tiles side by side or one above the other are separated by a white gap of
at least 8 px as shown.

The displayed signal of a tile is reduced to the tile's pixel size by the mean over blocks of
``b × b`` pixels, ``b = ceil(n_x / w)`` (partial blocks averaged over their pixels); its grey
levels are linear from the 0.1th to the 99.9th percentile of the tile's displayed values,
clipped, and a tile with ``lo = hi`` is drawn at mid-grey. An outline is reduced by the
maximum over the same blocks, so that no outline is lost. Each builder returns a
:class:`FigureImage` with the bytes, the pixel size, its slot, the round it shows, each tile's
range, and the texts it drew with their sizes and boxes.
"""
from __future__ import annotations

import io
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ._values import median

#: The tile sizes of "Figure sizes" (PX1).
PREPROCESSING_TILE = 256
EXAMPLE_TILE = 512
#: The slots of "Figure sizes": the width in px at which the page shows a figure in a window 1,200 px wide (the
#: full column of report.html, a half column ``.g2``, a third ``.g3``, and the column of index.html).
FULL_COLUMN, HALF_COLUMN, THIRD_COLUMN, COMPARISON_COLUMN = 932, 442, 300, 1160
#: A plot (PNG) is drawn at this many times the width of its slot.
PLOT_SCALE = 2
#: Text as shown, px: no text below MIN_TEXT_PX; labels, ticks and legends at TEXT_PX, titles at TITLE_PX.
MIN_TEXT_PX, TEXT_PX, TITLE_PX = 11, 12, 13
#: The white gap between adjacent tiles, and the width of a yellow region mark, px as shown.
TILE_GAP_PX, MARK_PX = 8, 2
MARK_COLOUR = "#ffd21f"
#: Heights as shown, px: per step name of "Time per step" and its least height; the intensity panels; per heatmap row.
STEP_ROW_PX, TIME_MIN_PX, INTENSITY_PX, HEATMAP_ROW_PX = 26, 280, 360, 15
#: The proportion width : height of a distribution (across FOVs or per cell) and of a box plot by sample.
DISTRIBUTION_RATIO, BOX_RATIO = (600, 340), (3, 1)
#: The most rows of a heatmap; above it, a sample of the FOVs (:func:`heatmap_rows`).
HEATMAP_ROWS = 64
#: The percentiles of a tile's grey range.
GREY_PERCENTILES = (0.1, 99.9)
CELL_OUTLINE = "#22d3ee"
NUCLEUS_OUTLINE = "#ffd21f"
REFERENCE_COLOUR, MOVING_COLOUR = "#00ff00", "#ff00ff"
#: The backing of a region's number on the whole-sample view.
_BACKING = "#05070a"
_DPI = 100
#: Keep each figure's canvas (RGB, as drawn before JPEG or PNG coding) in :attr:`FigureImage.canvas` (the tests).
KEEP_CANVAS = False


def plot_size(slot, ratio):
    """The pixel size of a plot of ``ratio`` (width, height) in ``slot``: twice the slot's width."""
    width = PLOT_SCALE * slot
    return width, int(round(width * ratio[1] / ratio[0]))


#: The distributions across FOVs and per cell in a third of the column (600 × 340).
SMALL_PLOT = plot_size(THIRD_COLUMN, DISTRIBUTION_RATIO)


@dataclass
class FigureImage:
    """A static figure: its bytes, format (``png`` or ``jpeg``), pixel size, the round it shows and its tiles.

    Each tile is a dict with its ``name`` and, for an image tile, its ``round``, ``channel`` and grey range
    ``lo``, ``hi`` (or ``reference``, ``moving`` ranges for a Registration tile). ``slot`` is the width in px at
    which the page shows the figure; ``texts`` every text drawn (``text``, ``size_px`` in figure pixels, ``box``
    ``(x0, y0, x1, y1)`` in figure pixels from the top left, ``axes`` the index of its axes in ``frames`` or None;
    ``size_px`` is the smallest glyph run, a mathtext superscript included, ``nominal_px`` the text's font size);
    ``frames`` the box of each axes; ``marks`` the boxes of the tiles and of the marks drawn on them; ``canvas`` the
    drawn RGB canvas when :data:`KEEP_CANVAS` is set.
    """

    data: bytes = field(repr=False)
    format: str
    width: int
    height: int
    round: str | None = None
    tiles: list = field(default_factory=list)
    slot: int | None = None
    texts: list = field(default_factory=list, repr=False)
    frames: list = field(default_factory=list, repr=False)
    marks: dict = field(default_factory=dict, repr=False)
    canvas: object = field(default=None, repr=False)


def block_mean(image, width):
    """The mean over blocks of ``b × b`` pixels, ``b = ceil(n_x / width)``; partial blocks at the edge are averaged
    over their pixels."""
    return _blocks(image, width, "mean")


def block_max(image, width):
    """The maximum over the blocks of :func:`block_mean`."""
    return _blocks(image, width, "max")


def _blocks(image, width, how):
    image = np.asarray(image, dtype=np.float64)
    b = max(1, math.ceil(image.shape[1] / width))
    if b == 1:
        return image
    ny, nx = (math.ceil(n / b) for n in image.shape)
    padded = np.full((ny * b, nx * b), np.nan if how == "mean" else -np.inf)
    padded[:image.shape[0], :image.shape[1]] = image
    blocks = padded.reshape(ny, b, nx, b)
    return np.nanmean(blocks, axis=(1, 3)) if how == "mean" else blocks.max(axis=(1, 3))


def grey_range(values):
    """The 0.1th and 99.9th percentile of a tile's displayed values (NumPy's default linear percentile)."""
    lo, hi = np.percentile(np.asarray(values, dtype=np.float64), GREY_PERCENTILES)
    return float(lo), float(hi)


def scaled(values, lo, hi):
    """Values mapped linearly from ``[lo, hi]`` to ``[0, 1]`` and clipped; ``lo = hi`` gives mid-grey."""
    values = np.asarray(values, dtype=np.float64)
    if hi <= lo:
        return np.full(values.shape, 0.5)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


# --- The size rule and the layout ---------------------------------------------------------------------------------

def tile_gap(n, tile, slot):
    """The white gap in px between ``n`` tiles of ``tile`` px side by side shown in ``slot``: the smallest whole
    ``g`` with ``g · slot / (n · tile + (n − 1) · g) ≥ 8`` (0 for one tile)."""
    if n < 2:
        return 0
    g = 0
    while g * slot < TILE_GAP_PX * (n * tile + (n - 1) * g):
        g += 1
    return g


def _pt(px):
    """Points of a size in figure pixels at 100 dpi."""
    return px * 72.0 / _DPI


class _Scale:
    """The pixels of a figure ``width`` px wide per pixel as shown in ``slot``."""

    def __init__(self, width, slot):
        self.width, self.slot = width, slot
        self.u = width / slot

    def __call__(self, shown):
        return shown * self.u

    @property
    def text(self):
        return TEXT_PX * self.u

    @property
    def title(self):
        return TITLE_PX * self.u


def _figure(width, height):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    figure = Figure(figsize=(width / _DPI, height / _DPI), dpi=_DPI, facecolor="white")
    FigureCanvasAgg(figure)
    return figure


def _resize(figure, width, height):
    figure.set_size_inches(width / _DPI, height / _DPI)


def _place(axes, x, y, w, h):
    """Axes at pixel position (x, y from the top) and size."""
    width, height = _figure_size(axes.figure)
    axes.set_position((x / width, 1 - (y + h) / height, max(w, 1) / width, max(h, 1) / height))


def _figure_size(figure):
    return figure.bbox.width, figure.bbox.height


def _axes(figure, x, y, w, h):
    """Axes at pixel position (x, y from the top) and size, without frame or ticks."""
    axes = figure.add_axes((0, 0, 1, 1))
    _place(axes, x, y, w, h)
    axes.set_axis_off()
    return axes


def _text(figure, x, y, text, px, **kwargs):
    """A text of ``px`` figure pixels at pixel position (x, y from the top)."""
    width, height = _figure_size(figure)
    return figure.text(x / width, 1 - y / height, text, fontsize=_pt(px), **kwargs)


def _renderer(figure):
    return figure.canvas.get_renderer()


def text_width(figure, text, px):
    """The drawn width in pixels of a text (its widest line) of ``px`` figure pixels."""
    from matplotlib.font_manager import FontProperties
    renderer = _renderer(figure)
    prop = FontProperties(size=_pt(px))
    return max(renderer.get_text_width_height_descent(line or " ", prop, ismath=False)[0]
               for line in str(text).split("\n"))


def _lines(px, n):
    """The height in pixels of ``n`` lines of text of ``px`` figure pixels (line spacing 1.2)."""
    return px * (1.2 * (n - 1) + 1.2)


_BREAK = re.compile(r"(?<=[ ,;·:_/])")


def wrap(figure, text, px, width):
    """``text`` broken into lines no wider than ``width`` pixels where it can be: after a space, a comma, a
    semicolon, a middle dot, a colon, an underscore or a slash."""
    out = []
    for paragraph in str(text).split("\n"):
        line = ""
        for word in _BREAK.split(paragraph):
            if line and text_width(figure, (line + word).rstrip(), px) > width:
                out.append(line.rstrip())
                line = word.lstrip()
            else:
                line += word
        out.append(line.rstrip())
    return "\n".join(out)


def _box(extent, height):
    """A display box as (x0, y0, x1, y1) in pixels from the top left."""
    return (float(extent.x0), float(height - extent.y1), float(extent.x1), float(height - extent.y0))


def _drawn_texts(figure):
    """Every text the figure draws, with the axes it belongs to (None for a text of the figure): the tick labels
    shown, the axis labels of axes with axes on, titles, legends and free texts."""
    from matplotlib.text import Text
    skip, drawn, seen = set(), [], set()
    for axes in figure.axes:
        for axis in (axes.xaxis, axes.yaxis):
            for tick in axis.majorTicks + axis.minorTicks:
                skip.update((id(tick.label1), id(tick.label2)))
            if not axes.axison or not axis.get_visible():
                skip.update((id(axis.label), id(axis.offsetText)))
                continue
            for tick in axis._update_ticks():
                for label in (tick.label1, tick.label2):
                    if label.get_visible() and label.get_text().strip():
                        drawn.append((label, axes))
    for axes in figure.axes:
        for text in axes.findobj(lambda artist: isinstance(artist, Text)):
            seen.add(id(text))
            if id(text) not in skip and text.get_visible() and text.get_text().strip():
                drawn.append((text, axes))
    for text in figure.findobj(lambda artist: isinstance(artist, Text)):
        if id(text) not in seen and id(text) not in skip and text.get_visible() and text.get_text().strip():
            drawn.append((text, None))
    return drawn


def _layout(figure):
    """The texts (with their sizes and boxes) and the axes frames of a drawn figure."""
    renderer = _renderer(figure)
    height = figure.bbox.height
    frames = [_box(a.get_window_extent(renderer), height) for a in figure.axes]
    index = {id(a): i for i, a in enumerate(figure.axes)}
    texts = []
    for text, owner in _drawn_texts(figure):
        texts.append({"text": text.get_text(), "size_px": min(glyph_px(text)),
                      "nominal_px": text.get_fontsize() * _DPI / 72.0,
                      "box": _box(text.get_window_extent(renderer), height),
                      "axes": None if owner is None else index[id(owner)]})
    return texts, frames


def glyph_px(text):
    """The size in figure pixels of each glyph run of a text as drawn: its font size, or, for mathtext, the size of
    each glyph the mathtext parser lays out (a superscript at 0.7 of the text's size)."""
    from matplotlib.mathtext import MathTextParser
    shown, math_text = text._preprocess_math(text.get_text())
    if not math_text:
        return [text.get_fontsize() * _DPI / 72.0]
    parsed = MathTextParser("path").parse(shown, dpi=72, prop=text.get_fontproperties())
    return [glyph[1] * _DPI / 72.0 for glyph in parsed.glyphs] or [text.get_fontsize() * _DPI / 72.0]


def _save(figure, form, quality=85):
    buffer = io.BytesIO()
    if form == "jpeg":
        figure.savefig(buffer, format="jpeg", dpi=_DPI, pil_kwargs={"quality": quality})
    else:
        figure.savefig(buffer, format="png", dpi=_DPI, metadata={"Software": None})
    return buffer.getvalue()


def _finish(figure, form, slot, *, quality=85, shows=None, tiles=None, marks=None) -> FigureImage:
    """The figure saved, with its slot, its texts and frames as drawn, and its canvas when kept."""
    data = _save(figure, form, quality)
    texts, frames = _layout(figure)
    # The size of the canvas as drawn (and saved): a figure of h px whose size in inches h / 100 is a float just
    # below h is drawn h − 1 px high.
    height, width = np.asarray(figure.canvas.buffer_rgba()).shape[:2]
    canvas = np.asarray(figure.canvas.buffer_rgba())[..., :3].copy() if KEEP_CANVAS else None
    return FigureImage(data, form, width, height, shows, list(tiles or []), slot, texts, frames, dict(marks or {}),
                       canvas)


def _overhangs(axes):
    """How far the axes' ticks, labels and titles reach beyond its frame: (left, top, right, bottom) pixels."""
    renderer = _renderer(axes.figure)
    tight, frame = axes.get_tightbbox(renderer), axes.get_window_extent(renderer)
    return (max(0.0, frame.x0 - tight.x0), max(0.0, tight.y1 - frame.y1), max(0.0, tight.x1 - frame.x1),
            max(0.0, frame.y0 - tight.y0))


def _fit(axes, box, pad):
    """Place ``axes`` so that its frame with its ticks and labels fills ``box`` (x0, y0, x1, y1 from the top left)
    less ``pad`` on every side; returns the frame (x, y, w, h)."""
    x0, y0, x1, y1 = box
    left = top = right = bottom = 0.0
    for _ in range(4):
        frame = (x0 + pad + left, y0 + pad + top, x1 - x0 - 2 * pad - left - right,
                 y1 - y0 - 2 * pad - top - bottom)
        _place(axes, *frame)
        found = _overhangs(axes)
        if all(abs(a - b) < 0.5 for a, b in zip(found, (left, top, right, bottom))):
            break
        left, top, right, bottom = found
    return frame


def _tick_boxes(axis):
    renderer = _renderer(axis.axes.figure)
    return [t.label1.get_window_extent(renderer) for t in axis._update_ticks()
            if t.label1.get_visible() and t.label1.get_text().strip()]


def _crowded(boxes, pad):
    boxes = sorted(boxes, key=lambda b: (b.x0, b.y0))
    for a, b in zip(boxes, boxes[1:]):
        if a.x1 + pad > b.x0 and a.y1 + pad > b.y0 and b.y1 + pad > a.y0:
            return True
    return any(a.y1 + pad > b.y0 and a.x1 > b.x0 and b.x1 > a.x0
               for a, b in zip(sorted(boxes, key=lambda b: b.y0), sorted(boxes, key=lambda b: b.y0)[1:]))


def _thin(axis, pad, counts=(6, 5, 4, 3, 2), integer=False):
    """Fewer ticks on a linear ``axis`` until no two of its labels come closer than ``pad`` pixels (whole numbers
    only with ``integer``)."""
    from matplotlib.ticker import MaxNLocator
    for n in counts:
        axis.set_major_locator(MaxNLocator(n, integer=integer))
        if not _crowded(_tick_boxes(axis), pad):
            return


def _style(axes, s, corner=False):
    """Tick labels at the text size, ticks and frame scaled with the figure; with ``corner`` the labels of the X axis
    sit half a line lower, so that they cannot meet the lowest label of the Y axis at the corner."""
    axes.tick_params(labelsize=_pt(s.text), length=_pt(s(3)), width=_pt(s(0.5)), pad=_pt(s(2)))
    if corner:
        axes.tick_params(axis="x", pad=_pt(s(2) + 0.6 * s.text))
    for spine in axes.spines.values():
        spine.set_linewidth(_pt(s(0.5)))


def _grey_bar(figure, x, y, width, lo, hi, s, label=None):
    """A grey colour bar under a tile, its range written below it (wrapped to the tile); returns the strip's
    height in pixels."""
    pad = s(4)
    bar = _axes(figure, x + pad, y + pad, width - 2 * pad, s(5))
    bar.imshow(np.linspace(0, 1, 256)[None, :], cmap="gray", aspect="auto", vmin=0, vmax=1)
    text = wrap(figure, _range_text(lo, hi, label), s.text, width - 2 * pad)
    _text(figure, x + pad, y + pad + s(5) + s(3), text, s.text, ha="left", va="top")
    return _strip_height(text.count("\n") + 1, s)


def _range_text(lo, hi, label=None):
    text = f"{lo:.4g} – {hi:.4g}" if hi > lo else f"{lo:.4g} (one value)"
    return text if label is None else f"{label}   {text}"


def _strip_height(lines, s):
    """The strip under a tile: the bar, its range in ``lines`` lines and a margin."""
    return s(4) + s(5) + s(3) + _lines(s.text, lines) + s(3)


def _strip_lines(figure, width, s, texts):
    """The most lines any of ``texts`` takes under a tile ``width`` pixels wide."""
    return max(wrap(figure, t, s.text, width - 2 * s(4)).count("\n") + 1 for t in texts) if texts else 1


def _tile(figure, x, y, size, values, lo, hi):
    """A grey tile of displayed values (already reduced) with its own range."""
    axes = _axes(figure, x, y, size, size)
    axes.imshow(scaled(values, lo, hi), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    return axes


def _legend_width(figure, items, s):
    return sum(s(10) + s(4) + text_width(figure, label, s.text) + s(12) for label, _ in items) - s(12)


def _legend(figure, items, y, s, *, right=None, after=None):
    """Legend squares and their labels on one row at ``y`` (its top), right-aligned at ``right`` pixels (the
    figure's width less a margin by default); ``after`` a grey text written after the items."""
    from matplotlib.patches import Rectangle
    width, height = _figure_size(figure)
    total = _legend_width(figure, items, s) + (s(12) + text_width(figure, after, s.text) if after else 0)
    x = (width - s(6) if right is None else right) - total
    square = s(10)
    middle = y + _lines(s.text, 1) / 2
    for label, colour in items:
        figure.patches.append(Rectangle((x / width, 1 - (middle + square / 2) / height), square / width,
                                        square / height, facecolor=colour, edgecolor="#555555",
                                        linewidth=_pt(s(0.5)), transform=figure.transFigure, figure=figure))
        _text(figure, x + square + s(4), middle, label, s.text, ha="left", va="center_baseline")
        x += square + s(4) + text_width(figure, label, s.text) + s(12)
    if after:
        _text(figure, x, middle, after, s.text, ha="left", va="center_baseline", color="#555555")


def _title(figure, x, y, text, s):
    """A tile's title, left-aligned above the tile at pixel ``x``, its top at ``y``."""
    _text(figure, x + s(4), y, text, s.title, ha="left", va="top")


def _unavailable_tile(figure, x, y, size, text, s):
    from matplotlib.patches import Rectangle
    axes = _axes(figure, x, y, size, size)
    axes.add_patch(Rectangle((0, 0), 1, 1, facecolor="#eeeeee", edgecolor="#999999", hatch="//"))
    shown = wrap(figure, f"Unavailable: {text}", s.text, size - s(16))
    axes.text(0.5, 0.5, shown, ha="center", va="center", fontsize=_pt(s.text), multialignment="center",
              bbox={"boxstyle": "square,pad=0.3", "facecolor": "#eeeeee", "edgecolor": "none"})


def _header(figure, s, titles, width, legend_items=None):
    """The header of an image figure: its legend on one row (right-aligned), then the tiles' titles, each wrapped
    to its tile; ``titles`` is [(x, text)]. Returns the wrapped titles and the header's height in pixels."""
    wrapped = [(x, wrap(figure, t, s.title, width - s(8))) for x, t in titles]
    lines = max((t.count("\n") + 1 for _, t in wrapped), default=0)
    legend = _lines(s.text, 1) + s(4) if legend_items else 0
    return wrapped, s(4) + legend + _lines(s.title, lines) + s(4)


def _draw_header(figure, s, wrapped, legend_items=None, legend_after=None):
    y = s(4)
    if legend_items:
        _legend(figure, legend_items, y, s, after=legend_after)
        y += _lines(s.text, 1) + s(4)
    for x, text in wrapped:
        _title(figure, x, y, text, s)


# --- Preprocessing ----------------------------------------------------------------------------------------------

def preprocessing_figure(replay, *, quality=85) -> FigureImage:
    """"Example FOV · <fov>, <round>": the reference round's grid, one row per channel, one column per state (the
    raw input, then each recorded entry in the order ``FOV.run`` runs them, headed by the step's name), tiles of
    256 px separated by white gaps of at least 8 px as shown in the full column."""
    name = replay.reference
    replayed = replay.rounds[name]
    rows, columns = len(replayed.channels), len(replayed.states)
    size, slot = PREPROCESSING_TILE, FULL_COLUMN
    gap = tile_gap(columns, size, slot)
    width = size * columns + gap * (columns - 1)
    s = _Scale(width, slot)
    figure = _figure(width, size)
    shown, ranges = {}, {}
    for j, state in enumerate(replayed.states):
        for i in range(rows):
            shown[i, j] = block_mean(state.signal[i], size)
            ranges[i, j] = grey_range(shown[i, j])
    titles, header = _header(figure, s, [(j * (size + gap), state.step) for j, state in enumerate(replayed.states)],
                             size)
    strip = _strip_height(_strip_lines(figure, size, s, [_range_text(*ranges[i, j], channel if j == 0 else None)
                                                           for (i, j) in ranges for channel in
                                                           [replayed.channels[i]]]), s)
    below = math.ceil(s(TILE_GAP_PX)) + 1
    height = int(math.ceil(header + rows * (size + strip) + (rows - 1) * below))
    _resize(figure, width, height)
    _draw_header(figure, s, titles)
    tiles, boxes = [], []
    for j, state in enumerate(replayed.states):
        for i, channel in enumerate(replayed.channels):
            x, y = j * (size + gap), header + i * (size + strip + below)
            lo, hi = ranges[i, j]
            _tile(figure, x, y, size, shown[i, j], lo, hi)
            _grey_bar(figure, x, y + size, size, lo, hi, s, channel if j == 0 else None)
            boxes.append((x, y, x + size, y + size))
            tiles.append({"name": state.step, "step_index": state.step_index, "phase": state.phase,
                          "round": name, "channel": channel, "lo": lo, "hi": hi})
    return _finish(figure, "jpeg", slot, quality=quality, shows=name, tiles=tiles,
                   marks={"tiles": boxes, "grid": (rows, columns)})


def intensity_figure(replay) -> FigureImage:
    """"Intensity after each step": one panel per column of the Preprocessing grid, in each one curve per channel,
    the counts of ``intensity.csv`` against the bin centres, the count axis logarithmic (a bin of count 0 is left
    out of the line); each panel with its own count axis, the legend of the channels above the panels."""
    name = replay.reference
    replayed = replay.rounds[name]
    slot = FULL_COLUMN
    width, height = PLOT_SCALE * slot, int(round(PLOT_SCALE * INTENSITY_PX))
    s = _Scale(width, slot)
    figure = _figure(width, height)
    states = replayed.states
    colours = [f"C{k}" for k in range(len(replayed.channels))]
    tiles, panels = [], []
    for j, state in enumerate(states):
        rows = [r for r in replayed.rows if r[1] == state.step_index]
        axes = figure.add_axes((0, 0, 1, 1))
        for c, channel in enumerate(replayed.channels):
            mine = [r for r in rows if r[4] == channel]
            centres = np.array([(r[6] + r[7]) / 2 for r in mine])
            counts = np.array([r[8] for r in mine], dtype=np.float64)
            counts[counts == 0] = np.nan
            axes.plot(centres, counts, color=colours[c], linewidth=_pt(s(1)), label=channel)
        axes.set_yscale("log")
        _plain_counts(axes.yaxis)
        axes.set_xlabel("grey level", fontsize=_pt(s.text))
        if j == 0:
            axes.set_ylabel("voxels", fontsize=_pt(s.text))
        _style(axes, s, corner=True)
        panels.append(axes)
        tiles.append({"name": state.step, "step_index": state.step_index, "round": name})
    pad, between = s(6), s(10)
    legend = _lines(s.text, 1) + s(6)
    n = len(panels)
    top = pad + legend
    for _ in range(3):
        over = [_overhangs(a) for a in panels]
        inner = width - 2 * pad - sum(o[0] + o[2] for o in over) - between * (n - 1)
        panel = inner / n
        titles = [wrap(figure, state.step, s.title, panel) for state in states]
        title_h = _lines(s.title, max(t.count("\n") + 1 for t in titles)) + s(4)
        bottom = max(o[3] for o in over)
        x = pad
        for axes, o, title in zip(panels, over, titles):
            x += o[0]
            _place(axes, x, top + title_h, panel, height - top - title_h - bottom - pad)
            x += panel + o[2] + between
        for axes in panels:
            _thin(axes.xaxis, s(6), (5, 4, 3, 2))
    for axes, title in zip(panels, titles):
        axes.set_title(title, fontsize=_pt(s.title), pad=_pt(s(4)))
    _legend(figure, list(zip(replayed.channels, colours)), pad, s, right=width - pad)
    return _finish(figure, "png", slot, shows=name, tiles=tiles)


# --- Registration -----------------------------------------------------------------------------------------------

def count_label(value):
    """A count of a logarithmic axis in plain text, without superscripts (1, 10, 100, 1k, 10k, 1M, 10M, 1G):
    every glyph is drawn at the label's own size."""
    for scale, suffix in ((1e9, "G"), (1e6, "M"), (1e3, "k")):
        if value >= scale:
            return f"{value / scale:g}{suffix}"
    return f"{value:g}"


def _plain_counts(axis):
    """The decades of a logarithmic count axis labelled by :func:`count_label`; the minor ticks unlabelled. The
    default labels (``10^7``) are mathtext, whose exponent is drawn at 0.7 of the label's size."""
    from matplotlib.ticker import FuncFormatter, NullFormatter
    axis.set_major_formatter(FuncFormatter(lambda value, _: count_label(value)))
    axis.set_minor_formatter(NullFormatter())


def _overlay(reference, moving, size):
    """The RGB tile ``(m, g, m)`` of the reference (green) and the moving round (magenta), each reduced to the tile
    and scaled by its own range; returns the tile and the two ranges."""
    reference, moving = block_mean(reference, size), block_mean(moving, size)
    g_range, m_range = grey_range(reference), grey_range(moving)
    g, m = scaled(reference, *g_range), scaled(moving, *m_range)
    return np.stack([m, g, m], axis=-1), g_range, m_range


def _registration_text(g_range, m_range):
    return f"reference {g_range[0]:.4g} – {g_range[1]:.4g}   moving {m_range[0]:.4g} – {m_range[1]:.4g}"


def _zoom_mark(axes, region, line):
    """The yellow rectangle around ``region`` (rows y0, y1 and columns x0, x1 of the tile's pixels, end
    excluded), ``line`` pixels wide, its inner edge on the region's edge; returns the region's box in figure
    pixels from the top left."""
    from matplotlib.patches import Rectangle
    y0, y1, x0, x1 = region
    axes.apply_aspect()
    transform = axes.transData
    (ax0, ay0), (ax1, ay1) = transform.transform([(x0 - 0.5, y0 - 0.5), (x1 - 0.5, y1 - 0.5)])
    per_x, per_y = abs(ax1 - ax0) / max(x1 - x0, 1), abs(ay1 - ay0) / max(y1 - y0, 1)
    ox, oy = line / 2 / per_x, line / 2 / per_y
    axes.add_patch(Rectangle((x0 - 0.5 - ox, y0 - 0.5 - oy), x1 - x0 + 2 * ox, y1 - y0 + 2 * oy, fill=False,
                             edgecolor=MARK_COLOUR, linewidth=_pt(line), antialiased=False, joinstyle="miter"))
    height = axes.figure.bbox.height
    return (float(min(ax0, ax1)), float(height - max(ay0, ay1)), float(max(ax0, ax1)), float(height - min(ay0, ay1)))


def registration_figure(replay, *, quality=85) -> FigureImage:
    """"Example FOV · <fov>, <round>": "before registration", "after registration" and "after registration,
    centre × 4" of the first moving round (magenta) against the reference round after its ``steps`` phase
    (green), each the maximum over channels and over Z; the third tile is the central quarter, in Y and X, of
    the "after" tile enlarged four times by nearest-neighbour repetition, and a yellow rectangle bounds that
    quarter on the first two tiles. The tiles are separated by white gaps of at least 8 px as shown."""
    reference = replay.rounds[replay.reference].before
    moving = replay.rounds[replay.moving[0]]
    size, slot = EXAMPLE_TILE, FULL_COLUMN
    gap = tile_gap(3, size, slot)
    width = 3 * size + 2 * gap
    s = _Scale(width, slot)
    figure = _figure(width, size)
    before, g0, m0 = _overlay(reference, moving.before, size)
    after, g1, m1 = _overlay(reference, moving.after, size)
    ny, nx = after.shape[:2]
    y0, x0 = ny * 3 // 8, nx * 3 // 8
    hq, wq = max(1, ny // 4), max(1, nx // 4)
    centre = after[y0:y0 + hq, x0:x0 + wq].repeat(4, axis=0).repeat(4, axis=1)
    shown = (("before registration", before, g0, m0), ("after registration", after, g1, m1),
             ("after registration, centre × 4", centre, g1, m1))
    items = (("reference round", REFERENCE_COLOUR), ("moving round", MOVING_COLOUR), ("overlap", "#ffffff"))
    titles, header = _header(figure, s, [(j * (size + gap), t[0]) for j, t in enumerate(shown)], size, items)
    lines = _strip_lines(figure, size + s(8), s, [_registration_text(g, m) for _, _, g, m in shown])
    strip = s(4) + _lines(s.text, lines) + s(4)
    _resize(figure, width, int(math.ceil(header + size + strip)))
    _draw_header(figure, s, titles, items)
    line = math.ceil(s(MARK_PX))
    tiles, boxes, marks = [], [], []
    for j, (title, image, g_range, m_range) in enumerate(shown):
        x = j * (size + gap)
        axes = _axes(figure, x, header, size, size)
        axes.imshow(image, interpolation="nearest")
        if j < 2:
            marks.append(_zoom_mark(axes, (y0, y0 + hq, x0, x0 + wq), line))
        text = wrap(figure, _registration_text(g_range, m_range), s.text, size - s(4))
        _text(figure, x + s(4), header + size + s(4), text, s.text, ha="left", va="top")
        boxes.append((x, header, x + size, header + size))
        tiles.append({"name": title, "round": moving.name, "reference_round": replay.reference,
                      "reference": g_range, "moving": m_range})
    return _finish(figure, "jpeg", slot, quality=quality, shows=moving.name, tiles=tiles,
                   marks={"tiles": boxes, "zoom": marks, "zoom_line_px": line})


# --- Segmentation and assembly ----------------------------------------------------------------------------------

def _outline_rgba(outline, colour):
    from matplotlib.colors import to_rgba
    rgba = np.zeros((*outline.shape, 4))
    rgba[outline > 0] = to_rgba(colour)
    return rgba


def segmentation_figure(previews, *, quality=85, reasons=None) -> FigureImage:
    """"Example FOVs": one tile per role ``example_1`` to ``example_3``, titled with its FOV and role: the
    background in grey with its own range, the cell outlines in cyan and the nucleus outlines in yellow. A role
    without a preview is drawn as a hatched tile with its reason (``reasons`` maps a reason to its text). The tiles
    are separated by white gaps of at least 8 px as shown."""
    size, slot = EXAMPLE_TILE, FULL_COLUMN
    roles = ("example_1", "example_2", "example_3")
    gap = tile_gap(3, size, slot)
    width = 3 * size + 2 * gap
    s = _Scale(width, slot)
    figure = _figure(width, size)
    items = (("nucleus outline", NUCLEUS_OUTLINE), ("cell outline", CELL_OUTLINE))
    named, ranges = [], {}
    for j, role in enumerate(roles):
        preview = previews.get(role)
        if preview is None or not hasattr(preview, "background"):
            named.append((j * (size + gap), role))
            continue
        background = block_mean(preview.background, size)
        ranges[role] = (background, *grey_range(background))
        named.append((j * (size + gap), f"{preview.fov_id} · {role}"))
    titles, header = _header(figure, s, named, size, items)
    strip = _strip_height(_strip_lines(figure, size, s, [_range_text(lo, hi) for _, lo, hi in ranges.values()]), s)
    _resize(figure, width, int(math.ceil(header + size + strip)))
    _draw_header(figure, s, titles, items)
    tiles, boxes = [], []
    for j, role in enumerate(roles):
        preview = previews.get(role)
        x = j * (size + gap)
        boxes.append((x, header, x + size, header + size))
        if role not in ranges:
            reason = getattr(preview, "reason", "no example")
            _unavailable_tile(figure, x, header, size, (reasons or {}).get(reason, reason), s)
            tiles.append({"name": role, "fov_id": None, "reason": reason})
            continue
        background, lo, hi = ranges[role]
        axes = _axes(figure, x, header, size, size)
        axes.imshow(scaled(background, lo, hi), cmap="gray", vmin=0, vmax=1, interpolation="nearest",
                    extent=preview.extent)
        for labels, colour in ((preview.cell_outline, CELL_OUTLINE), (preview.nucleus_outline, NUCLEUS_OUTLINE)):
            if labels is not None:
                axes.imshow(_outline_rgba(block_max(labels, size), colour), interpolation="nearest",
                            extent=preview.label_extent)
        x0, x1, y1, y0 = preview.extent
        axes.set_xlim(min(x0, preview.label_extent[0]), max(x1, preview.label_extent[1]))
        axes.set_ylim(max(y1, preview.label_extent[2]), min(y0, preview.label_extent[3]))
        _grey_bar(figure, x, header + size, size, lo, hi, s)
        tiles.append({"name": role, "fov_id": preview.fov_id, "route": preview.route, "lo": lo, "hi": hi})
    return _finish(figure, "jpeg", slot, quality=quality, tiles=tiles, marks={"tiles": boxes})


# --- Sample measurements ----------------------------------------------------------------------------------------

#: The most bins of a per-cell distribution.
HISTOGRAM_BINS = 30
#: The percentiles of the grey range of a fused image ("Sample measurements").
FUSED_PERCENTILES = (0.5, 99.8)
_EDGE_COLOUR = "#000000"


def histogram_bins(values, integer):
    """The bins of a per-cell distribution: equal bins over ``[0, p99]``, ``p99`` the 99th percentile (NumPy's
    linear percentile), at most 30; integer values in bins of integer width ``max(1, ceil((p99 + 1) / 30))``, bin
    ``k`` holding ``k·w ≤ v < (k + 1)·w``; other values in 30 bins, the last closed. The cells above ``p99`` are
    counted apart. Returns ``{"edges", "counts", "above", "p99", "width"}``; no value gives no bin."""
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {"edges": [], "counts": [], "above": 0, "p99": None, "width": None}
    p99 = float(np.percentile(values, 99))
    inside = values[values <= p99]
    if integer:
        width = max(1, math.ceil((p99 + 1) / HISTOGRAM_BINS))
        bins = int(math.floor(p99 / width)) + 1
        edges = [float(k * width) for k in range(bins + 1)]
        counts = np.bincount(np.floor(inside / width).astype(np.int64), minlength=bins)[:bins]
    elif p99 <= 0:
        width, edges = 0.0, [0.0, 0.0]
        counts = np.array([len(inside)])
    else:
        counts, edges = np.histogram(inside, bins=HISTOGRAM_BINS, range=(0.0, p99))
        width, edges = p99 / HISTOGRAM_BINS, [float(e) for e in edges]
    return {"edges": edges, "counts": [int(c) for c in counts], "above": int((values > p99).sum()), "p99": p99,
            "width": float(width)}


def grid_slot(n):
    """The slot of each figure of a three-figure grid's row of ``n`` figures: a third for three, a half (two
    columns, or one figure in a half column) for fewer."""
    return THIRD_COLUMN if n >= 3 else HALF_COLUMN


def _plot_axes(figure, s, xlabel, ylabel, integer=False):
    """One axes of a small plot with its labels wrapped to the plot, fitted to the figure."""
    width, height = _figure_size(figure)
    axes = figure.add_axes((0, 0, 1, 1))
    axes.set_ylabel(ylabel, fontsize=_pt(s.text))
    _style(axes, s, corner=True)
    frame = _fit(axes, (0, 0, width, height), s(6))
    for _ in range(2):
        axes.set_xlabel(wrap(figure, xlabel, s.text, frame[2]), fontsize=_pt(s.text))
        _thin(axes.xaxis, s(6), (6, 5, 4, 3), integer)
        _thin(axes.yaxis, s(2), (5, 4, 3), True)
        frame = _fit(axes, (0, 0, width, height), s(6))
    return axes


def histogram_figure(values, name, title, unit, *, integer, slot=THIRD_COLUMN) -> FigureImage:
    """A per-cell distribution (PNG twice its slot's width, at 600 : 340): the bins of :func:`histogram_bins`, the
    cells above ``p99`` in one last bar, counted in the text "> p99: n cells" at the top right of the plot, the
    median as a vertical line with its value in the axis label, and the unit in the axis label."""
    width, height = plot_size(slot, DISTRIBUTION_RATIO)
    s = _Scale(width, slot)
    figure = _figure(width, height)
    bins = histogram_bins(values, integer)
    centre = median(values) if len(values) else None
    label = f"{title} ({unit})" + (f" · median {centre:.4g}" if centre is not None else " · no counted cell")
    axes = _plot_axes(figure, s, label, "cells", integer)
    if bins["counts"]:
        edges, counts = bins["edges"], bins["counts"]
        step = bins["width"] or 1.0
        lefts = [edges[k] for k in range(len(counts))]
        axes.bar(lefts, counts, width=step, align="edge", color="#4c72b0", edgecolor="white", linewidth=_pt(s(0.5)))
        if bins["above"]:
            right = edges[len(counts)] + step * 0.5
            axes.bar([right], [bins["above"]], width=step, align="edge", color="#999999", edgecolor="white",
                     linewidth=_pt(s(0.5)))
            axes.set_ylim(0, max(max(counts), bins["above"]) * 1.25)
            axes.text(0.98, 0.97, f"> p99: {bins['above']} cells", transform=axes.transAxes, ha="right", va="top",
                      fontsize=_pt(s.text), color="#555555")
        axes.axvline(centre, color="#c44e52", linewidth=_pt(s(1.2)))
    _plot_axes_refit(axes, s, label, integer)
    tile = {"name": name, "xlabel": label, "unit": unit, "median": centre, **bins}
    return _finish(figure, "png", slot, tiles=[tile])


def _plot_axes_refit(axes, s, xlabel, integer=False):
    """Fit a small plot again once its data are drawn (the ticks follow the data); the count axis in whole
    numbers, the value axis too with ``integer``."""
    figure = axes.figure
    width, height = _figure_size(figure)
    for _ in range(2):
        frame = _fit(axes, (0, 0, width, height), s(6))
        axes.set_xlabel(wrap(figure, xlabel, s.text, frame[2]), fontsize=_pt(s.text))
        _thin(axes.xaxis, s(6), (6, 5, 4, 3), integer)
        _thin(axes.yaxis, s(2), (5, 4, 3), True)
    _fit(axes, (0, 0, width, height), s(6))


def grid_edges(columns, n_z):
    """The grid-edge lines of a FOV's row of the reads along Z: the pairs of column indices between the columns of
    ``z = −1`` and 0 and between those of ``n_z − 1`` and ``n_z``, each only where the figure has a column on both
    sides (none for a FOV without a grid)."""
    columns = list(columns)
    if n_z is None:
        return []
    return [(columns.index(a), columns.index(b)) for a, b in ((-1, 0), (n_z - 1, n_z))
            if a in columns and b in columns]


def heatmap_rows(fov_ids, marked=()):
    """The FOVs a heatmap draws, in list order: every FOV up to 64; above 64, the FOVs of ``marked`` (the first 64
    in list order if there are more), then, for the ``k`` rows left, the FOVs at the positions
    ``floor(i × u / k)``, ``i = 0 … k − 1``, of the ``u`` other FOVs in list order."""
    fov_ids = list(fov_ids)
    if len(fov_ids) <= HEATMAP_ROWS:
        return fov_ids
    marked = set(marked)
    first = [f for f in fov_ids if f in marked][:HEATMAP_ROWS]
    others = [f for f in fov_ids if f not in marked]
    k, u = HEATMAP_ROWS - len(first), len(others)
    chosen = set(first) | {others[i * u // k] for i in range(k)} if k and u else set(first)
    return [f for f in fov_ids if f in chosen]


def _heatmap_layout(figure, axes, bar, s, rows):
    """Lay out a heatmap and its colour bar: rows of 15 px as shown (at least the colour bar's label high), the
    labels fitted; returns the figure's height."""
    width = figure.bbox.width
    pad, gap, bar_w = s(6), s(10), s(7)
    for _ in range(3):
        left, top, right, bottom = _overhangs(axes)
        bar_over = _overhangs(bar)
        label = bar.yaxis.label
        label_h = text_width(figure, label.get_text(), s.text) + s(8)
        data_h = max(rows * s(HEATMAP_ROW_PX), label_h, s(120))
        height = int(math.ceil(pad + top + data_h + bottom + pad))
        _resize(figure, width, height)
        data_w = width - 2 * pad - left - right - gap - bar_w - bar_over[2]
        _place(axes, pad + left, pad + top, data_w, data_h)
        _place(bar, pad + left + data_w + right + gap, pad + top, bar_w, data_h)
    return height


def _heatmap_note(fovs, drawn, table):
    """The axis label of a heatmap that shows a sample of its FOVs."""
    return f"{len(drawn)} of {len(fovs)} FOVs shown; every FOV in {table}" if len(drawn) < len(fovs) else None


def reads_along_z_figure(rows, planes) -> FigureImage:
    """"Reads along Z" (PNG twice the half column's width, rows of 15 px as shown): one row per FOV (at most 64,
    :func:`heatmap_rows` without marked FOVs), one column per plane from the lowest to the highest ``z`` of the
    table's rows; the colour the FOV's fraction of its molecules on the plane, from 0 to the largest fraction of
    the FOVs drawn, with its colour bar; a plane outside the FOV's own rows hatched, a FOV without molecules light
    grey; a thin line at the two edges of each FOV's grid (:func:`grid_edges`)."""
    from matplotlib.patches import Rectangle
    all_fovs = list(dict.fromkeys(r["fov_id"] for r in rows))
    fovs = heatmap_rows(all_fovs)
    columns = list(range(min(r["z"] for r in rows), max(r["z"] for r in rows) + 1)) if rows else []
    slot = HALF_COLUMN
    width = PLOT_SCALE * slot
    s = _Scale(width, slot)
    figure = _figure(width, 600)
    axes = figure.add_axes((0, 0, 1, 1))
    bar = figure.add_axes((0, 0, 1, 1))
    fractions = np.full((len(fovs), len(columns)), np.nan)
    own, totals = {}, {}
    for i, fov in enumerate(fovs):
        mine = [r for r in rows if r["fov_id"] == fov]
        totals[fov] = sum(r["reads"] for r in mine)
        own[fov] = (mine[0]["z"], mine[-1]["z"])
        if totals[fov]:
            for r in mine:
                fractions[i, columns.index(r["z"])] = r["reads"] / totals[fov]
    high = float(np.nanmax(fractions)) if np.isfinite(fractions).any() else 0.0
    image = axes.imshow(np.ma.masked_invalid(fractions), cmap="viridis", vmin=0.0, vmax=high if high > 0 else 1.0,
                        aspect="auto", interpolation="nearest",
                        extent=(-0.5, len(columns) - 0.5, len(fovs) - 0.5, -0.5))
    tiles = []
    for i, fov in enumerate(fovs):
        low, top_z = own[fov]
        for j, z in enumerate(columns):
            if not low <= z <= top_z:
                axes.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="#eeeeee", edgecolor="#999999",
                                         hatch="//", linewidth=0))
            elif not totals[fov]:
                axes.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="#dddddd", linewidth=0))
        edges = grid_edges(columns, planes.get(fov))
        for _, b in edges:
            axes.plot([b - 0.5, b - 0.5], [i - 0.5, i + 0.5], color=_EDGE_COLOUR, linewidth=_pt(s(1)))
        tiles.append({"name": fov, "fov_id": fov, "n_z": planes.get(fov), "rows": [low, top_z],
                      "columns": columns, "edges": [list(e) for e in edges], "reads": totals[fov]})
    axes.set_xlim(-0.5, len(columns) - 0.5)
    axes.set_ylim(len(fovs) - 0.5, -0.5)
    axes.set_yticks(range(len(fovs)), fovs)
    note = _heatmap_note(all_fovs, fovs, "reads_along_z.csv")
    if note:
        axes.set_ylabel(note, fontsize=_pt(s.text))
    _style(axes, s)
    figure.colorbar(image, cax=bar)
    bar.set_yticks([0.0, high if high > 0 else 1.0], ["0", f"{high:.3g}"])
    _style(bar, s)
    bar.set_ylabel("fraction of the FOV's molecules", fontsize=_pt(s.text))
    xlabel = "plane z (voxel planes of the FOV's grid; lines: the grid's edges)"
    for _ in range(2):
        _heatmap_layout(figure, axes, bar, s, len(fovs))
        data_w = axes.get_window_extent(_renderer(figure)).width
        if columns:
            widest = max(text_width(figure, str(z), s.text) for z in columns)
            per = data_w / len(columns)
            step = next(k for k in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 10 ** 9)
                        if k * per >= widest + s(8) or k >= len(columns))
            ticks = [j for j, z in enumerate(columns) if z % step == 0] or [0]
            axes.set_xticks(ticks, [str(columns[j]) for j in ticks])
        axes.set_xlabel(wrap(figure, xlabel, s.text, data_w), fontsize=_pt(s.text))
    _heatmap_layout(figure, axes, bar, s, len(fovs))
    marks = {"rows": list(fovs), "row_px": axes.get_window_extent(_renderer(figure)).height / max(1, len(fovs)),
             "fovs": len(all_fovs), "note": note}
    return _finish(figure, "png", slot, tiles=tiles, marks=marks)


# --- Workflow overview, Registration, Readout -------------------------------------------------------------------

#: The bins of a distribution across FOVs: ``min(30, max(5, ceil(sqrt(n))))``.
FOV_BINS = (5, 30)


def time_per_step_figure(times, fov_names, sample_names, not_recorded) -> FigureImage:
    """"Time per step" (PNG twice the half column's width, 26 px as shown per step name and at least 280 px): a
    horizontal box plot of the seconds per FOV of each per-FOV step name (log scale, whiskers at 1.5 × the
    interquartile range, every FOV beyond drawn as a point), then, below a divider, each sample step as a single
    point on the same axis; the steps without a time are listed in the axis label as "not recorded". ``times`` is
    :func:`._values.time_per_step`."""
    slot = HALF_COLUMN
    width = PLOT_SCALE * slot
    s = _Scale(width, slot)
    figure = _figure(width, int(PLOT_SCALE * TIME_MIN_PX))
    axes = figure.add_axes((0, 0, 1, 1))
    names = list(fov_names) + list(sample_names)
    positions = {name: len(names) - i for i, name in enumerate(names)}
    tiles = []
    boxes = [name for name in fov_names if times.get(name)]
    line = _pt(s(0.8))
    if boxes:
        axes.boxplot([list(times[n].values()) for n in boxes], positions=[positions[n] for n in boxes],
                     orientation="horizontal", whis=1.5, widths=0.6,
                     flierprops={"marker": "o", "markersize": _pt(s(3))},
                     medianprops={"color": "#1f4fae", "linewidth": line}, boxprops={"linewidth": line},
                     whiskerprops={"linewidth": line}, capprops={"linewidth": line})
    for name in sample_names:
        if times.get(name):
            axes.plot([times[name]["sample"]], [positions[name]], "o", color="#17263a", markersize=_pt(s(4)))
    if fov_names and sample_names:
        axes.axhline(len(sample_names) + 0.5, color="#999999", linewidth=line)
    for name in names:
        tiles.append({"name": name, "scope": "sample" if name in sample_names else "fov",
                      "values": dict(times.get(name, {}))})
    shown = [v for name in names for v in times.get(name, {}).values() if v > 0]
    if shown:
        from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
        axes.set_xscale("log")
        # Plain tick labels (1, 10, 0.1 s): the default labels of a log axis are typeset as mathtext. Over less than
        # two decades the labels are at 1, 2 and 5 × 10ⁿ, so that the axis has more than one.
        if max(shown) < 100 * min(shown):
            axes.xaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
        axes.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        axes.xaxis.set_minor_formatter(NullFormatter())
    axes.set_yticks([positions[n] for n in names], names)
    axes.set_ylim(0.5, max(1, len(names)) + 0.5)
    label = "seconds (log scale)" if shown else "seconds (no step has a time)"
    if not_recorded:
        label += " · not recorded: " + ", ".join(not_recorded)
    _style(axes, s)
    pad = s(6)
    data_h = s(STEP_ROW_PX) * max(1, len(names))
    for _ in range(3):
        frame = _fit(axes, (0, 0, width, figure.bbox.height), pad)
        axes.set_xlabel(wrap(figure, label, s.text, frame[2]), fontsize=_pt(s.text))
        left, top, right, bottom = _overhangs(axes)
        height = int(math.ceil(max(PLOT_SCALE * TIME_MIN_PX, 2 * pad + top + data_h + bottom)))
        _resize(figure, width, height)
    _fit(axes, (0, 0, width, height), pad)
    return _finish(figure, "png", slot, tiles=tiles)


def residual_heatmap_figure(rows, fov_ids, rounds, marked=()) -> FigureImage:
    """"All FOVs and rounds" (PNG twice the full column's width, rows of 15 px as shown): one row per FOV in list
    order (at most 64, :func:`heatmap_rows` with the FOVs ``marked`` by the ``shift_residual`` rule first), each
    labelled with its FOV, one column per (round, axis) of the residual, a diverging scale symmetric about 0 with
    range ``[−a, a]``, ``a`` the largest absolute residual of the FOVs drawn (at least 1 voxel); null cells hatched
    grey."""
    from matplotlib.patches import Rectangle
    from matplotlib.transforms import blended_transform_factory, offset_copy
    drawn = heatmap_rows(fov_ids, marked)
    columns = [(name, axis) for name in rounds for axis in ("z", "y", "x")]
    by_key = {(r["fov_id"], r["round"]): r for r in rows}
    values = np.full((len(drawn), len(columns)), np.nan)
    for i, fov in enumerate(drawn):
        for j, (name, axis) in enumerate(columns):
            value = (by_key.get((fov, name)) or {}).get(f"residual_{axis}")
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                values[i, j] = float(value)
    finite = np.abs(values[np.isfinite(values)])
    a = max(1.0, float(finite.max())) if finite.size else 1.0
    slot = FULL_COLUMN
    width = PLOT_SCALE * slot
    s = _Scale(width, slot)
    figure = _figure(width, 600)
    axes = figure.add_axes((0, 0, 1, 1))
    bar = figure.add_axes((0, 0, 1, 1))
    image = axes.imshow(np.ma.masked_invalid(values), cmap="RdBu_r", vmin=-a, vmax=a, aspect="auto",
                        interpolation="nearest", extent=(-0.5, len(columns) - 0.5, len(drawn) - 0.5, -0.5))
    for i, j in zip(*np.nonzero(~np.isfinite(values))):
        axes.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="#dddddd", edgecolor="#999999", hatch="//",
                                 linewidth=0))
    for j in range(3, len(columns), 3):
        axes.axvline(j - 0.5, color="#ffffff", linewidth=_pt(s(2)))
    axes.set_xlim(-0.5, len(columns) - 0.5)
    axes.set_ylim(len(drawn) - 0.5, -0.5)
    axes.set_xticks(range(len(columns)), [axis for _, axis in columns])
    axes.set_yticks(range(len(drawn)), drawn)
    note = _heatmap_note(fov_ids, drawn, "registration.csv")
    if note:
        axes.set_ylabel(note, fontsize=_pt(s.text))
    _style(axes, s)
    # The rounds: one name under each group of three columns, below the axis letters.
    below = offset_copy(blended_transform_factory(axes.transData, axes.transAxes), figure,
                        y=-(s(3) + s(2) + _lines(s.text, 1) + s(2)), units="dots")
    for k, name in enumerate(rounds):
        axes.text(3 * k + 1, 0, name, transform=below, ha="center", va="top", fontsize=_pt(s.text))
    axes.xaxis.labelpad = _pt(_lines(s.text, 1) + s(6))
    axes.set_xlabel("residual per round and axis (voxels; hatched: unavailable)", fontsize=_pt(s.text))
    figure.colorbar(image, cax=bar)
    bar.set_yticks([-a, a], [f"{-a:.3g}", f"{a:.3g}"])
    _style(bar, s)
    bar.set_ylabel("residual (voxels)", fontsize=_pt(s.text))
    for _ in range(2):
        _heatmap_layout(figure, axes, bar, s, len(drawn))
        data_w = axes.get_window_extent(_renderer(figure)).width
        axes.set_xlabel(wrap(figure, axes.get_xlabel().replace("\n", " "), s.text, data_w), fontsize=_pt(s.text))
    _heatmap_layout(figure, axes, bar, s, len(drawn))
    tile = {"name": "residuals", "columns": [list(c) for c in columns], "rows": list(drawn), "range": a,
            "null_cells": int((~np.isfinite(values)).sum()), "labelled_every": 1, "fovs": len(fov_ids),
            "note": note}
    marks = {"rows": list(drawn), "row_px": axes.get_window_extent(_renderer(figure)).height / max(1, len(drawn)),
             "fovs": len(fov_ids), "note": note}
    return _finish(figure, "png", slot, tiles=[tile], marks=marks)


def fov_bins(values, integer):
    """The bins of a distribution across FOVs: ``min(30, max(5, ceil(sqrt(n))))`` equal bins from the smallest to
    the largest value; when they are equal, one bin of width 1 read (``integer``) or 0.01. Returns the edges."""
    values = np.asarray(values, dtype=np.float64)
    low, high = float(values.min()), float(values.max())
    if low == high:
        return [low, low + (1.0 if integer else 0.01)]
    n = min(FOV_BINS[1], max(FOV_BINS[0], math.ceil(math.sqrt(len(values)))))
    return [float(e) for e in np.linspace(low, high, n + 1)]


def distribution_figure(values, name, label, *, integer, empty=None, slot=THIRD_COLUMN) -> FigureImage:
    """A distribution across FOVs (PNG twice its slot's width, at 600 : 340): a histogram of the bins of
    :func:`fov_bins`, the median as a vertical line with its value in the axis label. ``empty`` is the axis label's
    text when no FOV has a value."""
    width, height = plot_size(slot, DISTRIBUTION_RATIO)
    s = _Scale(width, slot)
    figure = _figure(width, height)
    tile = {"name": name, "n": len(values)}
    if len(values):
        edges = fov_bins(values, integer)
        counts, _ = np.histogram(np.asarray(values, dtype=np.float64), bins=edges)
        centre = median(values)
        text = f"{label} · median {centre:.4g}"
        axes = _plot_axes(figure, s, text, "FOVs")
        axes.stairs(counts, edges, fill=True, color="#4c72b0", edgecolor="white")
        axes.axvline(centre, color="#c44e52", linewidth=_pt(s(1.2)))
        tile.update(edges=edges, counts=[int(c) for c in counts], median=centre)
    else:
        text = f"{label} · {empty or 'no FOV has a value'}"
        axes = _plot_axes(figure, s, text, "FOVs")
    _plot_axes_refit(axes, s, text, integer)
    tile["xlabel"] = text
    return _finish(figure, "png", slot, tiles=[tile])


#: Above this many samples the labels under the boxes of the comparison page are turned upright.
BOX_UPRIGHT_LABELS = 16
#: The plot of a box plot by sample spans from BOX_LEFT px to the width less BOX_RIGHT px.
BOX_LEFT, BOX_RIGHT = 160, 24


def sample_box_figure(name, label, slots) -> FigureImage:
    """A box plot by sample of the comparison page (PNG twice the comparison column's width, at 3 : 1): one slot
    per sample in the order of ``slots``, each a dict with ``sample``, ``values`` (the FOVs' values, nulls left out)
    and ``label`` (the reason a slot without a value shows). A slot with values is a box, whiskers at 1.5 × the
    interquartile range and each FOV beyond drawn as a point, with the number of FOVs with a value under it; a slot
    without one is an empty hatched slot labelled with its reason."""
    from matplotlib.transforms import blended_transform_factory
    slot = COMPARISON_COLUMN
    width, height = plot_size(slot, BOX_RATIO)
    s = _Scale(width, slot)
    figure = _figure(width, height)
    upright = len(slots) > BOX_UPRIGHT_LABELS
    axes = figure.add_axes((0, 0, 1, 1))
    boxes = [(i, slot_) for i, slot_ in enumerate(slots, 1) if slot_["values"]]
    line = _pt(s(0.8))
    if boxes:
        axes.boxplot([slot_["values"] for _, slot_ in boxes], positions=[i for i, _ in boxes], whis=1.5,
                     widths=0.5, flierprops={"marker": "o", "markersize": _pt(s(3))},
                     medianprops={"color": "#1f4fae", "linewidth": line}, boxprops={"linewidth": line},
                     whiskerprops={"linewidth": line}, capprops={"linewidth": line})
    across = blended_transform_factory(axes.transData, axes.transAxes)
    tiles, ticks = [], []
    for i, slot_ in enumerate(slots, 1):
        n = len(slot_["values"])
        if n:
            ticks.append(f"{slot_['sample']}\nn = {n}")
        else:
            axes.axvspan(i - 0.3, i + 0.3, facecolor="#eeeeee", edgecolor="#999999", hatch="//", linewidth=line)
            axes.text(i, 0.5, slot_["label"], transform=across, ha="center", va="center", rotation=90,
                      fontsize=_pt(s.text), color="#5b6878")
            ticks.append(f"{slot_['sample']}\n{slot_['label']}")
        tiles.append({"sample": slot_["sample"], "fovs": n, "label": None if n else slot_["label"]})
    axes.set_xticks(range(1, len(slots) + 1), ticks, rotation=90 if upright else 0)
    axes.set_xlim(0.5, len(slots) + 0.5)
    axes.set_ylabel(label, fontsize=_pt(s.text))
    _style(axes, s)
    _place(axes, BOX_LEFT, s(8), width - BOX_LEFT - BOX_RIGHT, height - s(8) - s(60))
    for _ in range(3):
        bottom = _overhangs(axes)[3]
        _place(axes, BOX_LEFT, s(8), width - BOX_LEFT - BOX_RIGHT, height - s(8) - bottom - s(6))
        _thin(axes.yaxis, s(2), (6, 5, 4, 3))
    return _finish(figure, "png", slot, tiles=[{"name": name, "slots": tiles}])


def fused_range(values):
    """The grey range of a fused image: its 0.5th and 99.8th percentiles over the nonzero pixels (0, 0 without
    one)."""
    values = np.asarray(values, dtype=np.float64)
    nonzero = values[values != 0]
    if not nonzero.size:
        return 0.0, 0.0
    lo, hi = np.percentile(nonzero, FUSED_PERCENTILES)
    return float(lo), float(hi)


def _pages(path):
    """The planes of a fused OME-TIFF, one at a time (its first series, full resolution)."""
    import tifffile
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        level = series.levels[0] if getattr(series, "levels", None) else series
        pages = list(level.pages)
        shape = tuple(level.shape)
        if len(pages) == (shape[0] if len(shape) == 3 else 1):
            for page in pages:
                yield np.asarray(page.asarray())
        else:
            data = np.asarray(level.asarray())
            yield from (data if data.ndim == 3 else data[np.newaxis])


@dataclass
class _Region:
    role: str
    fov_id: str
    footprint: tuple
    rows: tuple = (0, 0)
    cols: tuple = (0, 0)
    molecules: np.ndarray = None
    slab: tuple | None = None
    planes: list = field(default_factory=list)
    xy: np.ndarray = None
    band: tuple | None = None
    band_rows: tuple = (0, 0)
    xz: np.ndarray = None


@dataclass
class FusedView:
    """What the marker figures draw of the fused image: the whole-sample image (its projection, else the maximum
    over Z of its stack), the grid ``(origin_zyx, spacing_zyx)`` of each, and per zoom-in region its XY image (the
    footprint over the slab's planes, or the projection) and XZ image (the stack's maximum over the band's rows);
    the stack is read once, plane by plane. ``plane_grid`` is the grid of the regions' images: the stack's when it
    has more than one plane, else the projection's."""

    image: str
    whole: np.ndarray
    whole_grid: tuple
    stack: bool
    plane_grid: tuple
    regions: dict

    @classmethod
    def read(cls, export_dir, sources, regions, molecules, retained):
        from ._measurements import SLAB_UM, slab
        from ._previews import pixel_range
        projection, stack = sources["projection"], sources["stack"]
        three_d = stack is not None and int(stack["shape_zyx"][0]) > 1

        def grid_of(entry):
            return (tuple(float(v) for v in entry["origin_zyx"]), tuple(float(v) for v in entry["spacing_zyx"]))
        plane_entry = stack if three_d else (projection or stack)
        (oz, oy, ox), (sz, sy, sx) = grid_of(plane_entry)
        ny, nx = (int(n) for n in plane_entry["shape_zyx"][1:])
        sample_y, sample_x = molecules["sample_y"].to_numpy(np.float64), molecules["sample_x"].to_numpy(np.float64)
        sample_z = molecules["sample_z"].to_numpy(np.float64)
        views = {}
        for role, fov, (y0, y1, x0, x1) in regions:
            region = _Region(role, fov, (y0, y1, x0, x1), pixel_range(y0, y1, oy, sy, ny),
                             pixel_range(x0, x1, ox, sx, nx))
            inside = retained & (sample_y >= y0) & (sample_y < y1) & (sample_x >= x0) & (sample_x < x1)
            region.molecules = inside
            if three_d:
                nz = int(stack["shape_zyx"][0])
                centres = oz + np.arange(nz) * sz
                # A region without a retained molecule: the slab centred on the stack's middle plane.
                middle = oz + (nz - 1) / 2 * sz
                region.slab = slab(sample_z[inside]) or (middle, middle - SLAB_UM / 2, middle + SLAB_UM / 2)
                z_c, low, high = region.slab
                planes = [k for k in range(nz) if low <= centres[k] < high]
                region.planes = planes or [int(np.argmin(np.abs(centres - z_c)))]
                cy = (y0 + y1) / 2
                region.band = (cy - SLAB_UM / 2, cy + SLAB_UM / 2)
                row_centres = oy + np.arange(ny) * sy
                band = np.flatnonzero((row_centres >= region.band[0]) & (row_centres < region.band[1]))
                if not len(band):
                    band = np.array([int(np.argmin(np.abs(row_centres - cy)))])
                region.band_rows = (int(band[0]), int(band[-1]) + 1)
                region.xz = np.zeros((nz, region.cols[1] - region.cols[0]))
            views[role] = region
        whole = None
        if projection is not None:
            whole = next(_pages(Path(export_dir) / projection["path"])).astype(np.float64)
            whole_grid = grid_of(projection)
        if three_d:
            for k, plane in enumerate(_pages(Path(export_dir) / stack["path"])):
                plane = plane.astype(np.float64)
                if projection is None:
                    whole = plane if whole is None else np.maximum(whole, plane)
                for region in views.values():
                    (r0, r1), (c0, c1) = region.rows, region.cols
                    if k in region.planes:
                        cut = plane[r0:r1, c0:c1]
                        region.xy = cut if region.xy is None else np.maximum(region.xy, cut)
                    b0, b1 = region.band_rows
                    region.xz[k] = plane[b0:b1, c0:c1].max(axis=0) if c1 > c0 else 0
            if projection is None:
                whole_grid = grid_of(stack)
        else:
            if whole is None:
                whole = next(_pages(Path(export_dir) / stack["path"])).astype(np.float64)
                whole_grid = grid_of(stack)
            for region in views.values():
                (r0, r1), (c0, c1) = region.rows, region.cols
                region.xy = whole[r0:r1, c0:c1]
        return cls(sources["image"], whole, whole_grid, three_d, grid_of(plane_entry), views)


def _point_area(diameter_px):
    """matplotlib's marker area (points²) of a point ``diameter_px`` pixels wide at 100 dpi."""
    return (diameter_px * 72.0 / 100.0) ** 2


def _scale_bar_um(width_um):
    """The largest length of 1, 2 or 5 × 10ⁿ µm that is at most a fifth of ``width_um``."""
    limit = width_um / 5
    if limit <= 0:
        return None
    power = 10.0 ** math.floor(math.log10(limit))
    return max(f * power for f in (1, 2, 5) if f * power <= limit)


def _image_axes(figure, x, y, w, h, image, grid, lo, hi, *, origin="upper"):
    """An image in µm on axes of ``w × h`` pixels at ``(x, y)``: the pixel edges ``O + (i ∓ ½)·S``; returns the
    axes and the µm extent ``(x0, x1, y0, y1)`` of the image."""
    (oy, ox), (sy, sx) = grid
    shown = block_mean(image, w) if image.size else np.zeros((1, 1))
    b = max(1, math.ceil(image.shape[1] / w)) if image.size else 1
    x0, y0 = ox - sx / 2, oy - sy / 2
    x1, y1 = ox + (image.shape[1] - 0.5) * sx, oy + (image.shape[0] - 0.5) * sy
    shown_x1, shown_y1 = ox + (shown.shape[1] * b - 0.5) * sx, oy + (shown.shape[0] * b - 0.5) * sy
    axes = _axes(figure, x, y, w, h)
    if origin == "upper":
        axes.imshow(scaled(shown, lo, hi), cmap="gray", vmin=0, vmax=1, interpolation="nearest", aspect="auto",
                    extent=(x0, shown_x1, shown_y1, y0))
        axes.set_xlim(x0, x1)
        axes.set_ylim(y1, y0)
    else:
        axes.imshow(scaled(shown, lo, hi), cmap="gray", vmin=0, vmax=1, interpolation="nearest", aspect="auto",
                    origin="lower", extent=(x0, shown_x1, y0, shown_y1))
        axes.set_xlim(x0, x1)
        axes.set_ylim(y0, y1)
    return axes, (x0, x1, y0, y1)


def _points(axes, molecules, chosen, group, diameter, ycolumn="sample_y"):
    """The group's molecules among ``chosen``, one colour per gene drawn in group order (the last gene on top),
    opacity 0.8, rasterized; returns per gene the rows drawn (indices of the molecule table)."""
    genes = molecules["gene_id"].astype(object).to_numpy()
    x, y = molecules["sample_x"].to_numpy(np.float64), molecules[ycolumn].to_numpy(np.float64)
    drawn = {}
    for gene, colour in zip(group.genes, group.colours):
        rows = np.flatnonzero(chosen & (genes == gene))
        drawn[gene] = rows
        if len(rows):
            axes.scatter(x[rows], y[rows], s=_point_area(diameter), c=colour, alpha=0.8, linewidths=0,
                         rasterized=True)
    return drawn


def _region_box(axes, footprint, line):
    """A region's yellow rectangle, ``line`` pixels wide, its inner edge on the footprint (y0, y1, x0, x1 in µm), a
    side at the edge of the image drawn just inside it; returns the rectangle's inner box and the part of it inside
    the image and its lines, in pixels from the top left."""
    from matplotlib.patches import Rectangle
    figure = axes.figure
    height = figure.bbox.height
    fy0, fy1, fx0, fx1 = footprint
    (px0, py0), (px1, py1) = axes.transData.transform([(fx0, fy0), (fx1, fy1)])
    per_x, per_y = abs(px1 - px0) / max(fx1 - fx0, 1e-12), abs(py1 - py0) / max(fy1 - fy0, 1e-12)
    ox, oy = line / 2 / per_x, line / 2 / per_y
    (xa, xb), (ya, yb) = sorted(axes.get_xlim()), sorted(axes.get_ylim())
    left, right = max(fx0 - ox, xa + ox), min(fx1 + ox, xb - ox)
    top, bottom = max(fy0 - oy, ya + oy), min(fy1 + oy, yb - oy)
    axes.add_patch(Rectangle((left, top), right - left, bottom - top, fill=False, edgecolor=MARK_COLOUR,
                             linewidth=_pt(line), antialiased=False, joinstyle="miter"))
    inner = (float(min(px0, px1)), float(height - max(py0, py1)), float(max(px0, px1)), float(height - min(py0, py1)))
    frame = _box(axes.get_window_extent(_renderer(figure)), height)
    visible = (max(inner[0], frame[0] + line), max(inner[1], frame[1] + line), min(inner[2], frame[2] - line),
               min(inner[3], frame[3] - line))
    return inner, visible


def _region_number(axes, k, visible, others, placed, s):
    """Region ``k``'s number on a dark backing inside its rectangle (``visible``), at the first corner (top left, top
    right, bottom left, bottom right) where the backing fits, covers no number in ``placed`` and lies in no other
    region's box (``others``); else at the first corner where it fits and covers no number. Returns the backing's
    box and the number's text box in pixels from the top left."""
    from matplotlib.transforms import IdentityTransform
    figure = axes.figure
    height = figure.bbox.height
    pad = 0.25
    number = axes.text(0, 0, str(k), transform=IdentityTransform(), ha="center", va="center", color="white",
                       fontsize=_pt(s.text), bbox={"boxstyle": f"square,pad={pad}", "facecolor": _BACKING,
                                                   "edgecolor": "none"})
    extent = number.get_window_extent(_renderer(figure))
    grow = pad * s.text
    bw, bh = extent.width + 2 * grow, extent.height + 2 * grow
    x0, y0, x1, y1 = visible
    margin = s(2)
    corners = [(x0 + margin, y0 + margin), (x1 - margin - bw, y0 + margin), (x0 + margin, y1 - margin - bh),
               (x1 - margin - bw, y1 - margin - bh)]
    boxes = [(cx, cy, cx + bw, cy + bh) for cx, cy in corners]

    def meets(box, among):
        return any(box[0] < b[2] and b[0] < box[2] and box[1] < b[3] and b[1] < box[3] for b in among)
    fits = bw + 2 * margin <= x1 - x0 and bh + 2 * margin <= y1 - y0
    backing = next((b for b in boxes if fits and not meets(b, placed) and not meets(b, others)),
                   next((b for b in boxes if fits and not meets(b, placed)), boxes[0]))
    number.set_position(((backing[0] + backing[2]) / 2, height - (backing[1] + backing[3]) / 2))
    return backing, _box(number.get_window_extent(_renderer(figure)), height)


def _whole_view(group, view, molecules, retained, regions, width, quality):
    image = view.whole
    (_, oy, ox), (_, sy, sx) = view.whole_grid
    ny, nx = image.shape
    width_um, height_um = nx * sx, ny * sy
    h = max(1, int(round(width * height_um / width_um)))
    slot = FULL_COLUMN
    s = _Scale(width, slot)
    figure = _figure(width, h)
    items = list(zip(group.genes, group.colours))
    titles, header = _header(figure, s, [(0, group.title)], width, items)
    lo, hi = fused_range(block_mean(image, width))
    strip = _strip_height(_strip_lines(figure, width, s, [_range_text(lo, hi, view.image)]), s)
    _resize(figure, width, int(math.ceil(header + h + strip)))
    axes, (x0, x1, y0, y1) = _image_axes(figure, 0, header, width, h, image, ((oy, ox), (sy, sx)), lo, hi)
    um_per_px = width_um / width
    diameter = _point_diameter(um_per_px)
    drawn = _points(axes, molecules, retained, group, diameter)
    line = math.ceil(s(MARK_PX))
    # The rectangles first, then each number at a corner that only its own rectangle holds, where one does.
    boxes = [(k, role, *_region_box(axes, footprint, line))
             for k, (role, fov, footprint) in enumerate(regions, start=1) if footprint is not None]
    placed, numbers = [], []
    for k, role, inner, visible in boxes:
        others = [b[2] for b in boxes if b[0] != k]
        backing, text_box = _region_number(axes, k, visible, others, placed, s)
        placed.append(backing)
        numbers.append({"k": k, "role": role, "region": inner, "backing": backing, "text": text_box})
    length = _scale_bar_um(width_um)
    if length is not None:
        xr = x1 - (x1 - x0) * 0.03
        yr = y1 - (y1 - y0) * 0.04
        axes.plot([xr - length, xr], [yr, yr], color="white", linewidth=_pt(s(2)))
        axes.text(xr - length / 2, yr - (y1 - y0) * 0.015, f"{length:g} µm", color="white", fontsize=_pt(s.text),
                  ha="center", va="bottom", bbox={"boxstyle": "square,pad=0.2", "facecolor": _BACKING,
                                                  "edgecolor": "none", "alpha": 0.6})
    _draw_header(figure, s, titles, items, view.image)
    _grey_bar(figure, 0, header + h, width, lo, hi, s, view.image)
    tile = {"name": "whole", "title": group.title, "image": view.image,
            "genes": {g: int(len(r)) for g, r in drawn.items()},
            "point_diameter_px": diameter, "um_per_px": um_per_px, "lo": lo, "hi": hi, "scale_bar_um": length}
    return _finish(figure, "jpeg", slot, quality=quality, tiles=[tile],
                   marks={"regions": numbers, "line_px": line, "image": (0, header, width, header + h)})


def _point_diameter(um_per_px):
    from ._measurements import point_diameter
    return point_diameter(um_per_px)


def _zoom_tile(title, w, h, image, lo, hi, s):
    """A zoom-in tile's figure: its title wrapped above, the strip of its grey bar below; returns the figure and
    the header's height."""
    figure = _figure(w, h)
    titles, header = _header(figure, s, [(0, title)], w)
    strip = _strip_height(_strip_lines(figure, w, s, [_range_text(lo, hi)]), s)
    _resize(figure, w, int(math.ceil(header + h + strip)))
    _draw_header(figure, s, titles)
    return figure, header


def _zoom_xy(k, group, view, region, molecules, quality):
    (_, oy, ox), (_, sy, sx) = view.plane_grid
    (r0, r1), (c0, c1) = region.rows, region.cols
    image = region.xy if region.xy is not None else np.zeros((0, 0))
    w, slot = EXAMPLE_TILE, THIRD_COLUMN
    s = _Scale(w, slot)
    width_um, height_um = max(c1 - c0, 1) * sx, max(r1 - r0, 1) * sy
    h = max(1, int(round(w * height_um / width_um)))
    lo, hi = fused_range(block_mean(image, w)) if image.size else (0.0, 0.0)
    where = "projection" if region.slab is None else f"z {region.slab[0]:.3g} ± 1.5 µm"
    title = f"Region {k} · {region.fov_id} · {where}"
    figure, header = _zoom_tile(title, w, h, image, lo, hi, s)
    axes, _ = _image_axes(figure, 0, header, w, h, image, ((oy + r0 * sy, ox + c0 * sx), (sy, sx)), lo, hi)
    chosen = region.molecules.copy()
    if region.slab is not None:
        z = molecules["sample_z"].to_numpy(np.float64)
        chosen &= (z >= region.slab[1]) & (z < region.slab[2])
    um_per_px = width_um / w
    diameter = _point_diameter(um_per_px)
    drawn = _points(axes, molecules, chosen, group, diameter)
    _grey_bar(figure, 0, header + h, w, lo, hi, s)
    tile = {"name": f"region_{k}_xy", "title": title, "role": region.role, "fov_id": region.fov_id, "view": "xy",
            "projection": region.slab is None, "z_c": None if region.slab is None else region.slab[0],
            "slab": None if region.slab is None else [region.slab[1], region.slab[2]], "planes": region.planes,
            "rows": sorted(int(i) for r in drawn.values() for i in r),
            "genes": {g: int(len(r)) for g, r in drawn.items()}, "point_diameter_px": diameter,
            "um_per_px": um_per_px, "lo": lo, "hi": hi}
    return _finish(figure, "jpeg", slot, quality=quality, tiles=[tile])


def _zoom_xz(k, group, view, region, molecules, retained, quality):
    (oz, _, ox), (sz, _, sx) = view.plane_grid
    (c0, c1) = region.cols
    image = region.xz
    w, slot = EXAMPLE_TILE, THIRD_COLUMN
    s = _Scale(w, slot)
    width_um, depth_um = max(c1 - c0, 1) * sx, image.shape[0] * sz
    stretch = max(1.0, (width_um / 6) / depth_um)
    h = max(1, int(round(w * depth_um * stretch / width_um)))
    lo, hi = fused_range(block_mean(image, w)) if image.size else (0.0, 0.0)
    title = f"Region {k} · {region.fov_id} · XZ, Z × {stretch:.3g}"
    figure, header = _zoom_tile(title, w, h, image, lo, hi, s)
    axes, _ = _image_axes(figure, 0, header, w, h, image, ((oz, ox + c0 * sx), (sz, sx)), lo, hi, origin="lower")
    y = molecules["sample_y"].to_numpy(np.float64)
    x = molecules["sample_x"].to_numpy(np.float64)
    band = retained & (y >= region.band[0]) & (y < region.band[1]) & (x >= region.footprint[2]) & \
        (x < region.footprint[3])
    um_per_px = width_um / w
    diameter = _point_diameter(um_per_px)
    drawn = _points(axes, molecules, band, group, diameter, "sample_z")
    _grey_bar(figure, 0, header + h, w, lo, hi, s)
    tile = {"name": f"region_{k}_xz", "title": title, "role": region.role, "fov_id": region.fov_id, "view": "xz",
            "stretch": stretch, "band": list(region.band), "rows": sorted(int(i) for r in drawn.values() for i in r),
            "genes": {g: int(len(r)) for g, r in drawn.items()}, "point_diameter_px": diameter,
            "um_per_px": um_per_px, "lo": lo, "hi": hi}
    return _finish(figure, "jpeg", slot, quality=quality, tiles=[tile])


def marker_figure(group, view, molecules, retained, regions, *, width, quality):
    """The figure of one marker group (:class:`._measurements.MarkerFigure`): the whole-sample view
    (``figure_width_px`` wide JPEG) and, per zoom-in region (``example_1`` to ``example_3``), the XY tile and the
    XZ tile (512 px wide JPEGs); an XZ tile without a stack of more than one plane is unavailable (``no_stack``),
    and a region whose footprint or example is unavailable has its reason."""
    from ._measurements import MarkerFigure
    from ._values import Unavailable
    listed = [] if isinstance(regions, Unavailable) else regions
    footprints = [(r, f, None if isinstance(p, Unavailable) else p) for r, f, p in listed]
    whole = _whole_view(group, view, molecules, retained, footprints, width, quality)
    zooms = {}
    if isinstance(regions, Unavailable):
        return MarkerFigure(group, whole, regions)
    for k, (role, fov, footprint) in enumerate(regions, start=1):
        if isinstance(footprint, Unavailable):
            zooms[(role, "xy")] = zooms[(role, "xz")] = footprint
            continue
        region = view.regions[role]
        zooms[(role, "xy")] = _zoom_xy(k, group, view, region, molecules, quality)
        zooms[(role, "xz")] = (_zoom_xz(k, group, view, region, molecules, retained, quality) if view.stack else
                               Unavailable("no_stack", "the export has no fused stack of more than one plane"))
    return MarkerFigure(group, whole, zooms)
