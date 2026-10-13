"""The static figures of the examples (docs/report-algorithms.md, "Preprocessing", "Registration", "Segmentation
and assembly", "The replay of the examples", "Figure sizes").

Drawn with matplotlib's ``Agg`` canvas at 100 dpi and saved to memory, with the bundled
DejaVu Sans font: the image figures as JPEG at the configuration's quality, the plots as PNG
(whose metadata holds no time). Each figure is laid out in pixels, so that its size is the
one of "Figure sizes" (PX1): the Preprocessing grid 256 px per tile, the Registration and
segmentation examples 512 px per tile, the intensity panels 1,200 × 400.

The displayed signal of a tile is reduced to the tile's pixel size by the mean over blocks of
``b × b`` pixels, ``b = ceil(n_x / w)`` (partial blocks averaged over their pixels); its grey
levels are linear from the 0.1th to the 99.9th percentile of the tile's displayed values,
clipped, and a tile with ``lo = hi`` is drawn at mid-grey. An outline is reduced by the
maximum over the same blocks, so that no outline is lost. Each builder returns a
:class:`FigureImage` with the bytes, the pixel size, the round it shows and each tile's range.
"""
from __future__ import annotations

import io
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ._values import median

#: The pixel sizes of "Figure sizes" (PX1).
PREPROCESSING_TILE = 256
EXAMPLE_TILE = 512
PLOT_SIZE = (1200, 400)
#: The percentiles of a tile's grey range.
GREY_PERCENTILES = (0.1, 99.9)
CELL_OUTLINE = "#22d3ee"
NUCLEUS_OUTLINE = "#ffd21f"
REFERENCE_COLOUR, MOVING_COLOUR = "#00ff00", "#ff00ff"
_HEADER, _STRIP = 28, 34


@dataclass
class FigureImage:
    """A static figure: its bytes, format (``png`` or ``jpeg``), pixel size, the round it shows and its tiles.

    Each tile is a dict with its ``name`` and, for an image tile, its ``round``, ``channel`` and grey range
    ``lo``, ``hi`` (or ``reference``, ``moving`` ranges for a Registration tile).
    """

    data: bytes = field(repr=False)
    format: str
    width: int
    height: int
    round: str | None = None
    tiles: list = field(default_factory=list)


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


def _figure(width, height):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    figure = Figure(figsize=(width / 100, height / 100), dpi=100, facecolor="white")
    FigureCanvasAgg(figure)
    return figure


def _axes(figure, x, y, w, h):
    """Axes at pixel position (x, y from the top) and size, without frame or ticks."""
    width, height = figure.get_size_inches() * figure.dpi
    axes = figure.add_axes((x / width, 1 - (y + h) / height, w / width, h / height))
    axes.set_axis_off()
    return axes


def _save(figure, form, quality=85):
    buffer = io.BytesIO()
    if form == "jpeg":
        figure.savefig(buffer, format="jpeg", dpi=100, pil_kwargs={"quality": quality})
    else:
        figure.savefig(buffer, format="png", dpi=100, metadata={"Software": None})
    return buffer.getvalue()


def _grey_bar(figure, x, y, width, lo, hi, label=None):
    """A grey colour bar under a tile with its range written at its ends."""
    bar = _axes(figure, x + 8, y + 4, width - 16, 8)
    bar.imshow(np.linspace(0, 1, 256)[None, :], cmap="gray", aspect="auto", vmin=0, vmax=1)
    text = f"{lo:.4g} – {hi:.4g}" if hi > lo else f"{lo:.4g} (one value)"
    figure.text((x + 8) / figure.bbox.width, 1 - (y + 26) / figure.bbox.height,
                text if label is None else f"{label}   {text}", fontsize=7, ha="left", va="bottom")


def _tile(figure, x, y, size, values):
    """A grey tile of displayed values with its own range; returns the range."""
    values = block_mean(values, size)
    lo, hi = grey_range(values)
    axes = _axes(figure, x, y, size, size)
    axes.imshow(scaled(values, lo, hi), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    return lo, hi


def _legend(figure, items):
    """Legend squares and their labels, right-aligned in the header strip."""
    from matplotlib.patches import Rectangle
    width, height = figure.bbox.width, figure.bbox.height
    x = width - 8 - sum(16 + 7 * len(label) for label, _ in items)
    for label, colour in items:
        figure.patches.append(Rectangle((x / width, 1 - 20 / height), 10 / width, 10 / height, facecolor=colour,
                                        edgecolor="#555555", linewidth=0.5, transform=figure.transFigure,
                                        figure=figure))
        figure.text((x + 14) / width, 1 - 20 / height, label, fontsize=8, ha="left", va="bottom")
        x += 16 + 7 * len(label)


def _title(figure, x, text):
    """A tile's title, left-aligned in the header strip above the tile at pixel ``x``."""
    figure.text((x + 8) / figure.bbox.width, 1 - 6 / figure.bbox.height, text, ha="left", va="top", fontsize=9)


def _unavailable_tile(figure, x, y, size, text):
    from matplotlib.patches import Rectangle
    axes = _axes(figure, x, y, size, size)
    axes.add_patch(Rectangle((0, 0), 1, 1, facecolor="#eeeeee", edgecolor="#999999", hatch="//"))
    axes.text(0.5, 0.5, f"Unavailable: {text}", ha="center", va="center", fontsize=8, wrap=True)


# --- Preprocessing ----------------------------------------------------------------------------------------------

def preprocessing_figure(replay, *, quality=85) -> FigureImage:
    """"Example FOV · <fov>, <round>": the reference round's grid, one row per channel, one column per state (the
    raw input, then each recorded entry in the order ``FOV.run`` runs them, headed by the step's name)."""
    name = replay.reference
    replayed = replay.rounds[name]
    rows, columns = len(replayed.channels), len(replayed.states)
    size = PREPROCESSING_TILE
    width, height = size * columns, _HEADER + rows * (size + _STRIP)
    figure = _figure(width, height)
    tiles = []
    for j, state in enumerate(replayed.states):
        figure.text((j + 0.5) * size / width, 1 - 6 / height, state.step, ha="center", va="top", fontsize=9)
        for i, channel in enumerate(replayed.channels):
            x, y = j * size, _HEADER + i * (size + _STRIP)
            lo, hi = _tile(figure, x, y, size, state.signal[i])
            _grey_bar(figure, x, y + size, size, lo, hi, channel if j == 0 else None)
            tiles.append({"name": state.step, "step_index": state.step_index, "phase": state.phase,
                          "round": name, "channel": channel, "lo": lo, "hi": hi})
    return FigureImage(_save(figure, "jpeg", quality), "jpeg", width, height, name, tiles)


def intensity_figure(replay) -> FigureImage:
    """"Intensity after each step": one panel per column of the Preprocessing grid, in each one curve per channel,
    the counts of ``intensity.csv`` against the bin centres, the count axis logarithmic (a bin of count 0 is left
    out of the line)."""
    name = replay.reference
    replayed = replay.rounds[name]
    width, height = PLOT_SIZE
    figure = _figure(width, height)
    states = replayed.states
    gap, left, bottom, top = 16, 56, 44, 30
    panel = (width - left - gap * (len(states) - 1) - 12) / len(states)
    colours = [f"C{k}" for k in range(len(replayed.channels))]
    tiles = []
    for j, state in enumerate(states):
        rows = [r for r in replayed.rows if r[1] == state.step_index]
        axes = figure.add_axes(((left + j * (panel + gap)) / width, bottom / height, panel / width,
                                (height - bottom - top) / height))
        for c, channel in enumerate(replayed.channels):
            mine = [r for r in rows if r[4] == channel]
            centres = np.array([(r[6] + r[7]) / 2 for r in mine])
            counts = np.array([r[8] for r in mine], dtype=np.float64)
            counts[counts == 0] = np.nan
            axes.plot(centres, counts, color=colours[c], linewidth=1, label=channel)
        axes.set_yscale("log")
        axes.set_title(state.step, fontsize=9)
        axes.set_xlabel("grey level", fontsize=8)
        axes.tick_params(labelsize=7)
        if j == 0:
            axes.set_ylabel("voxels", fontsize=8)
            axes.legend(fontsize=7, frameon=False)
        tiles.append({"name": state.step, "step_index": state.step_index, "round": name})
    return FigureImage(_save(figure, "png"), "png", width, height, name, tiles)


# --- Registration -----------------------------------------------------------------------------------------------

def _overlay(reference, moving, size):
    """The RGB tile ``(m, g, m)`` of the reference (green) and the moving round (magenta), each reduced to the tile
    and scaled by its own range; returns the tile and the two ranges."""
    reference, moving = block_mean(reference, size), block_mean(moving, size)
    g_range, m_range = grey_range(reference), grey_range(moving)
    g, m = scaled(reference, *g_range), scaled(moving, *m_range)
    return np.stack([m, g, m], axis=-1), g_range, m_range


def registration_figure(replay, *, quality=85) -> FigureImage:
    """"Example FOV · <fov>, <round>": "before registration", "after registration" and "after registration,
    centre × 4" of the first moving round (magenta) against the reference round after its ``steps`` phase
    (green), each the maximum over channels and over Z; the third tile is the central quarter, in Y and X, of
    the "after" tile enlarged four times by nearest-neighbour repetition."""
    reference = replay.rounds[replay.reference].before
    moving = replay.rounds[replay.moving[0]]
    size = EXAMPLE_TILE
    width, height = 3 * size, _HEADER + size + _STRIP
    figure = _figure(width, height)
    before, g0, m0 = _overlay(reference, moving.before, size)
    after, g1, m1 = _overlay(reference, moving.after, size)
    ny, nx = after.shape[:2]
    y0, x0 = ny * 3 // 8, nx * 3 // 8
    centre = after[y0:y0 + max(1, ny // 4), x0:x0 + max(1, nx // 4)].repeat(4, axis=0).repeat(4, axis=1)
    tiles = []
    for j, (title, image, g_range, m_range) in enumerate((("before registration", before, g0, m0),
                                                          ("after registration", after, g1, m1),
                                                          ("after registration, centre × 4", centre, g1, m1))):
        axes = _axes(figure, j * size, _HEADER, size, size)
        axes.imshow(image, interpolation="nearest")
        _title(figure, j * size, title)
        figure.text((j * size + 8) / width, 1 - (_HEADER + size + 22) / height,
                    f"reference {g_range[0]:.4g} – {g_range[1]:.4g}   moving {m_range[0]:.4g} – {m_range[1]:.4g}",
                    fontsize=7, ha="left", va="bottom")
        tiles.append({"name": title, "round": moving.name, "reference_round": replay.reference,
                      "reference": g_range, "moving": m_range})
    _legend(figure, (("reference round", REFERENCE_COLOUR), ("moving round", MOVING_COLOUR), ("overlap", "#ffffff")))
    return FigureImage(_save(figure, "jpeg", quality), "jpeg", width, height, moving.name, tiles)


# --- Segmentation and assembly ----------------------------------------------------------------------------------

def _outline_rgba(outline, colour):
    from matplotlib.colors import to_rgba
    rgba = np.zeros((*outline.shape, 4))
    rgba[outline > 0] = to_rgba(colour)
    return rgba


def segmentation_figure(previews, *, quality=85, reasons=None) -> FigureImage:
    """"Example FOVs": one tile per role ``example_1`` to ``example_3``, titled with its FOV and role: the
    background in grey with its own range, the cell outlines in cyan and the nucleus outlines in yellow. A role
    without a preview is drawn as a hatched tile with its reason (``reasons`` maps a reason to its text)."""
    size = EXAMPLE_TILE
    roles = ("example_1", "example_2", "example_3")
    width, height = 3 * size, _HEADER + size + _STRIP
    figure = _figure(width, height)
    tiles = []
    for j, role in enumerate(roles):
        preview = previews.get(role)
        x = j * size
        if preview is None or not hasattr(preview, "background"):
            reason = getattr(preview, "reason", "no example")
            _unavailable_tile(figure, x, _HEADER, size, (reasons or {}).get(reason, reason))
            _title(figure, x, role)
            tiles.append({"name": role, "fov_id": None, "reason": reason})
            continue
        background = block_mean(preview.background, size)
        lo, hi = grey_range(background)
        axes = _axes(figure, x, _HEADER, size, size)
        axes.imshow(scaled(background, lo, hi), cmap="gray", vmin=0, vmax=1, interpolation="nearest",
                    extent=preview.extent)
        for labels, colour in ((preview.cell_outline, CELL_OUTLINE), (preview.nucleus_outline, NUCLEUS_OUTLINE)):
            if labels is not None:
                axes.imshow(_outline_rgba(block_max(labels, size), colour), interpolation="nearest",
                            extent=preview.label_extent)
        x0, x1, y1, y0 = preview.extent
        axes.set_xlim(min(x0, preview.label_extent[0]), max(x1, preview.label_extent[1]))
        axes.set_ylim(max(y1, preview.label_extent[2]), min(y0, preview.label_extent[3]))
        _title(figure, x, f"{preview.fov_id} · {role}")
        _grey_bar(figure, x, _HEADER + size, size, lo, hi)
        tiles.append({"name": role, "fov_id": preview.fov_id, "route": preview.route, "lo": lo, "hi": hi})
    _legend(figure, (("nucleus outline", NUCLEUS_OUTLINE), ("cell outline", CELL_OUTLINE)))
    return FigureImage(_save(figure, "jpeg", quality), "jpeg", width, height, None, tiles)


# --- Sample measurements ----------------------------------------------------------------------------------------

#: The size of a plot of a third or a half column ("Figure sizes"): the per-cell distributions.
SMALL_PLOT = (600, 340)
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


def histogram_figure(values, name, title, unit, *, integer) -> FigureImage:
    """A per-cell distribution (600 × 340 PNG): the bins of :func:`histogram_bins`, the cells above ``p99`` in one
    last bar labelled "> p99 (n cells)", the median as a vertical line with its value in the axis label, and the
    unit in the axis label."""
    width, height = SMALL_PLOT
    figure = _figure(width, height)
    axes = figure.add_axes((64 / width, 52 / height, (width - 80) / width, (height - 80) / height))
    bins = histogram_bins(values, integer)
    centre = median(values) if len(values) else None
    label = f"{title} ({unit})" + (f" · median {centre:.4g}" if centre is not None else " · no counted cell")
    if bins["counts"]:
        edges, counts = bins["edges"], bins["counts"]
        step = bins["width"] or 1.0
        lefts = [edges[k] for k in range(len(counts))]
        axes.bar(lefts, counts, width=step, align="edge", color="#4c72b0", edgecolor="white", linewidth=0.5)
        if bins["above"]:
            right = edges[len(counts)] + step * 0.5
            axes.bar([right], [bins["above"]], width=step, align="edge", color="#999999", edgecolor="white",
                     linewidth=0.5)
            axes.text(right + step, bins["above"], f"> p99 ({bins['above']} cells)", ha="right", va="bottom",
                      fontsize=7)
        axes.axvline(centre, color="#c44e52", linewidth=1.2)
    axes.set_xlabel(label, fontsize=8)
    axes.set_ylabel("cells", fontsize=8)
    axes.tick_params(labelsize=7)
    tile = {"name": name, "xlabel": label, "unit": unit, "median": centre, **bins}
    return FigureImage(_save(figure, "png"), "png", width, height, None, [tile])


def grid_edges(columns, n_z):
    """The grid-edge lines of a FOV's row of the reads along Z: the pairs of column indices between the columns of
    ``z = −1`` and 0 and between those of ``n_z − 1`` and ``n_z``, each only where the figure has a column on both
    sides (none for a FOV without a grid)."""
    columns = list(columns)
    if n_z is None:
        return []
    return [(columns.index(a), columns.index(b)) for a, b in ((-1, 0), (n_z - 1, n_z))
            if a in columns and b in columns]


def reads_along_z_figure(rows, planes) -> FigureImage:
    """"Reads along Z" (PNG, 1,200 × ``min(1,200, max(300, 12 per row))``): one row per FOV, one column per plane
    from the lowest to the highest ``z`` of the table's rows; the colour the FOV's fraction of its molecules on the
    plane, from 0 to the largest fraction, with its colour bar; a plane outside the FOV's own rows hatched, a FOV
    without molecules light grey; a thin line at the two edges of each FOV's grid (:func:`grid_edges`)."""
    from matplotlib.patches import Rectangle
    fovs = list(dict.fromkeys(r["fov_id"] for r in rows))
    columns = list(range(min(r["z"] for r in rows), max(r["z"] for r in rows) + 1)) if rows else []
    width, height = PLOT_SIZE[0], min(1200, max(300, 12 * len(fovs)))
    figure = _figure(width, height)
    left, right, bottom, top = 110, 110, 44, 16
    axes = figure.add_axes((left / width, bottom / height, (width - left - right) / width,
                            (height - bottom - top) / height))
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
            axes.plot([b - 0.5, b - 0.5], [i - 0.5, i + 0.5], color=_EDGE_COLOUR, linewidth=1)
        tiles.append({"name": fov, "fov_id": fov, "n_z": planes.get(fov), "rows": [low, top_z],
                      "columns": columns, "edges": [list(e) for e in edges], "reads": totals[fov]})
    axes.set_xlim(-0.5, len(columns) - 0.5)
    axes.set_ylim(len(fovs) - 0.5, -0.5)
    step = max(1, math.ceil(len(columns) / 30))
    axes.set_xticks(range(0, len(columns), step), [str(z) for z in columns[::step]])
    every = max(1, math.ceil(len(fovs) / 50))
    axes.set_yticks(range(0, len(fovs), every), fovs[::every])
    axes.tick_params(labelsize=7)
    axes.set_xlabel("plane z (voxel planes of the FOV's grid; lines: the grid's edges)", fontsize=8)
    bar = figure.add_axes(((width - right + 24) / width, bottom / height, 12 / width,
                           (height - bottom - top) / height))
    figure.colorbar(image, cax=bar)
    bar.set_yticks([0.0, high if high > 0 else 1.0], ["0", f"{high:.3g}"])
    bar.tick_params(labelsize=7)
    bar.set_ylabel("fraction of the FOV's molecules", fontsize=7)
    return FigureImage(_save(figure, "png"), "png", width, height, None, tiles)


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


def _gene_legend(figure, group, image):
    from matplotlib.patches import Rectangle
    width, height = figure.bbox.width, figure.bbox.height
    items = list(zip(group.genes, group.colours))
    x = width - 8 - sum(16 + 7 * len(g) for g, _ in items) - 7 * len(image) - 8
    for gene, colour in items:
        figure.patches.append(Rectangle((x / width, 1 - 20 / height), 10 / width, 10 / height, facecolor=colour,
                                        edgecolor="#555555", linewidth=0.5, transform=figure.transFigure,
                                        figure=figure))
        figure.text((x + 14) / width, 1 - 20 / height, gene, fontsize=8, ha="left", va="bottom")
        x += 16 + 7 * len(gene)
    figure.text((x + 8) / width, 1 - 20 / height, image, fontsize=8, ha="left", va="bottom", color="#555555")


def _whole_view(group, view, molecules, retained, regions, width, quality):
    from matplotlib.patches import Rectangle
    image = view.whole
    (_, oy, ox), (_, sy, sx) = view.whole_grid
    ny, nx = image.shape
    width_um, height_um = nx * sx, ny * sy
    h = max(1, int(round(width * height_um / width_um)))
    figure = _figure(width, _HEADER + h + _STRIP)
    lo, hi = fused_range(block_mean(image, width))
    axes, (x0, x1, y0, y1) = _image_axes(figure, 0, _HEADER, width, h, image, ((oy, ox), (sy, sx)), lo, hi)
    um_per_px = width_um / width
    diameter = _point_diameter(um_per_px)
    drawn = _points(axes, molecules, retained, group, diameter)
    for k, (role, fov, footprint) in enumerate(regions, start=1):
        if footprint is None:
            continue
        fy0, fy1, fx0, fx1 = footprint
        axes.add_patch(Rectangle((fx0, fy0), fx1 - fx0, fy1 - fy0, fill=False, edgecolor="white", linewidth=1))
        axes.text(fx0, fy0, f" {k}", color="white", fontsize=8, ha="left", va="top")
    length = _scale_bar_um(width_um)
    if length is not None:
        xr = x1 - (x1 - x0) * 0.03
        yr = y1 - (y1 - y0) * 0.04
        axes.plot([xr - length, xr], [yr, yr], color="white", linewidth=2)
        axes.text(xr - length / 2, yr - (y1 - y0) * 0.01, f"{length:g} µm", color="white", fontsize=8,
                  ha="center", va="bottom")
    _title(figure, 0, group.title)
    _gene_legend(figure, group, view.image)
    _grey_bar(figure, 0, _HEADER + h, width, lo, hi, view.image)
    tile = {"name": "whole", "title": group.title, "image": view.image,
            "genes": {g: int(len(r)) for g, r in drawn.items()},
            "point_diameter_px": diameter, "um_per_px": um_per_px, "lo": lo, "hi": hi, "scale_bar_um": length}
    return FigureImage(_save(figure, "jpeg", quality), "jpeg", width, _HEADER + h + _STRIP, None, [tile])


def _point_diameter(um_per_px):
    from ._measurements import point_diameter
    return point_diameter(um_per_px)


def _zoom_xy(k, group, view, region, molecules, quality):
    (_, oy, ox), (_, sy, sx) = view.plane_grid
    (r0, r1), (c0, c1) = region.rows, region.cols
    image = region.xy if region.xy is not None else np.zeros((0, 0))
    w = EXAMPLE_TILE
    width_um, height_um = max(c1 - c0, 1) * sx, max(r1 - r0, 1) * sy
    h = max(1, int(round(w * height_um / width_um)))
    figure = _figure(w, _HEADER + h + _STRIP)
    lo, hi = fused_range(block_mean(image, w)) if image.size else (0.0, 0.0)
    axes, _ = _image_axes(figure, 0, _HEADER, w, h, image, ((oy + r0 * sy, ox + c0 * sx), (sy, sx)), lo, hi)
    chosen = region.molecules.copy()
    if region.slab is not None:
        z = molecules["sample_z"].to_numpy(np.float64)
        chosen &= (z >= region.slab[1]) & (z < region.slab[2])
    um_per_px = width_um / w
    diameter = _point_diameter(um_per_px)
    drawn = _points(axes, molecules, chosen, group, diameter)
    where = "projection" if region.slab is None else f"z {region.slab[0]:.3g} ± 1.5 µm"
    title = f"Region {k} · {region.fov_id} · {where}"
    _title(figure, 0, title)
    _grey_bar(figure, 0, _HEADER + h, w, lo, hi)
    tile = {"name": f"region_{k}_xy", "title": title, "role": region.role, "fov_id": region.fov_id, "view": "xy",
            "projection": region.slab is None, "z_c": None if region.slab is None else region.slab[0],
            "slab": None if region.slab is None else [region.slab[1], region.slab[2]], "planes": region.planes,
            "rows": sorted(int(i) for r in drawn.values() for i in r),
            "genes": {g: int(len(r)) for g, r in drawn.items()}, "point_diameter_px": diameter,
            "um_per_px": um_per_px, "lo": lo, "hi": hi}
    return FigureImage(_save(figure, "jpeg", quality), "jpeg", w, _HEADER + h + _STRIP, None, [tile])


def _zoom_xz(k, group, view, region, molecules, retained, quality):
    (oz, _, ox), (sz, _, sx) = view.plane_grid
    (c0, c1) = region.cols
    image = region.xz
    w = EXAMPLE_TILE
    width_um, depth_um = max(c1 - c0, 1) * sx, image.shape[0] * sz
    stretch = max(1.0, (width_um / 6) / depth_um)
    h = max(1, int(round(w * depth_um * stretch / width_um)))
    figure = _figure(w, _HEADER + h + _STRIP)
    lo, hi = fused_range(block_mean(image, w)) if image.size else (0.0, 0.0)
    axes, _ = _image_axes(figure, 0, _HEADER, w, h, image, ((oz, ox + c0 * sx), (sz, sx)), lo, hi, origin="lower")
    y = molecules["sample_y"].to_numpy(np.float64)
    x = molecules["sample_x"].to_numpy(np.float64)
    band = retained & (y >= region.band[0]) & (y < region.band[1]) & (x >= region.footprint[2]) & \
        (x < region.footprint[3])
    um_per_px = width_um / w
    diameter = _point_diameter(um_per_px)
    drawn = _points(axes, molecules, band, group, diameter, "sample_z")
    title = f"Region {k} · {region.fov_id} · XZ, Z × {stretch:.3g}"
    _title(figure, 0, title)
    _grey_bar(figure, 0, _HEADER + h, w, lo, hi)
    tile = {"name": f"region_{k}_xz", "title": title, "role": region.role, "fov_id": region.fov_id, "view": "xz",
            "stretch": stretch, "band": list(region.band), "rows": sorted(int(i) for r in drawn.values() for i in r),
            "genes": {g: int(len(r)) for g, r in drawn.items()}, "point_diameter_px": diameter,
            "um_per_px": um_per_px, "lo": lo, "hi": hi}
    return FigureImage(_save(figure, "jpeg", quality), "jpeg", w, _HEADER + h + _STRIP, None, [tile])


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
