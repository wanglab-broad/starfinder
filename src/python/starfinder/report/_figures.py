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

import numpy as np

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
