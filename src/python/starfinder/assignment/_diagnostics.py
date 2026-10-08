"""On-demand diagnostics of an assignment: a summary mapping and a figure."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from ._assign import AssignmentResult
from ._config import (ASSIGNMENT_STATUSES, CELL_CORRESPONDENCE, CELL_STATUSES, COMPARTMENT_STATES,
                      CORRESPONDENCE_FLAGS, NUCLEUS_STATUSES)

_QUANTILES = (0.0, 0.25, 0.5, 0.75, 1.0)
# plot_assignment: the two views, the outline colour and width, the area and edge of a cell centre, the
# colour of each drawn molecule status and cell status (a kept cell as an assigned molecule, an excluded
# one as an excluded_cell molecule), and the legend text of each. Outline width and centre area are in
# points, so they do not shrink when a large image is shown small; molecule points still do.
_VIEWS = ("z_max", "single_layer")
_OUTLINE_COLOUR = "lime"
_OUTLINE_WIDTH = 0.8
_CENTRE_AREA = 30.0
_CENTRE_EDGE = ("black", 0.5)
_STATUS_COLOURS = {"assigned": "dodgerblue", "unassigned": "red", "excluded_cell": "orange"}
_STATUS_LEGEND = {"assigned": "assigned", "unassigned": "unassigned", "excluded_cell": "excluded"}
_CENTRE_COLOURS = {"kept": _STATUS_COLOURS["assigned"],
                   "excluded_no_nucleus": _STATUS_COLOURS["excluded_cell"]}
_CENTRE_LEGEND = {"kept": "kept", "excluded_no_nucleus": "excluded"}
# With ownership: the region line (the legacy tile-box colour), width in points, and the colour and legend
# text of the cells and molecules the assembly excluded.
_REGION_COLOUR = "yellow"
_REGION_WIDTH = 1.2
_REGION_LEGEND = "ownership region"
_EXCLUDED_COLOUR = "grey"
_EXCLUDED_LEGEND = "excluded from sample"


@dataclass(frozen=True, eq=False)
class FOVOwnership:
    """What the sample assembly decided for one FOV, for :func:`plot_assignment`.

    ``region`` is a boolean Y, X array on the FOV's grid: True at the pixels
    whose centres the FOV owns in the sample. ``excluded_cells`` holds the
    ``cell_id`` of each of the FOV's cells and ``excluded_molecules`` the
    ``spot_id`` of each of its molecules that the assembly excluded (assembly
    status ``excluded``). ``fov_id`` names the FOV, or None. Plain data: build
    it with :func:`starfinder.assembly.fov_ownership` from an assembled sample.

    Raises
    ------
    ValueError
        region is not a two-dimensional boolean array, or an identity is not
        a nonnegative integer (cells) or a string (molecules).
    """

    region: np.ndarray
    excluded_cells: frozenset[int] = frozenset()
    excluded_molecules: frozenset[str] = frozenset()
    fov_id: str | None = None

    def __post_init__(self):
        region = np.asarray(self.region)
        if region.ndim != 2 or region.dtype != np.bool_:
            raise ValueError(f"region must be a Y, X boolean array; got {region.dtype} of shape {region.shape}")
        region = region.copy()
        region.setflags(write=False)
        object.__setattr__(self, "region", region)
        for name, kind in (("excluded_cells", "cell_id"), ("excluded_molecules", "spot_id")):
            values = getattr(self, name)
            if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
                raise ValueError(f"{name} must be a collection of {kind}; got {values!r}")
            values = list(values)
            if name == "excluded_cells":
                wrong = [v for v in values if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
                         or v < 0]
            else:
                wrong = [v for v in values if not isinstance(v, str)]
            if wrong:
                raise ValueError(f"{name} must hold {kind} values; got {wrong[:5]!r}")
            object.__setattr__(self, name, frozenset(int(v) for v in values) if name == "excluded_cells"
                               else frozenset(values))


def _check_ownership(ownership, result, shape):
    """The ownership record of plot_assignment: its type, its region's Y, X and its identities."""
    if not isinstance(ownership, FOVOwnership):
        raise TypeError(f"ownership must be a FOVOwnership or None; got {type(ownership).__name__}")
    if ownership.region.shape != tuple(shape):
        raise ValueError(f"ownership region Y, X {ownership.region.shape} differ from the territories' {tuple(shape)}")
    for name, given, held in (("cells", ownership.excluded_cells, result.cells.cell_id.astype("int64")),
                              ("molecules", ownership.excluded_molecules, result.molecules.spot_id.astype(object))):
        unknown = sorted(given - set(held.tolist()), key=str)
        if unknown:
            raise ValueError(f"ownership names excluded {name} the result does not have: {unknown[:5]}"
                             + (f" and {len(unknown) - 5} more" if len(unknown) > 5 else ""))


def _region_lines(region):
    """The closed boundary of a Y, X boolean region: one (x, y) polyline per loop along the pixel edges between
    its pixels and the others (the image border included), in pixel index coordinates, vertices at the corners."""
    inside = np.asarray(region, bool)
    padded = np.pad(inside, 1)
    # Corner (r, c) is the point (c - 0.5, r - 0.5). Each region pixel's sides that face another pixel run
    # clockwise on screen (top rightwards, right downwards, bottom leftwards, left upwards), so every corner
    # has as many edges in as out, and following them closes each loop.
    outgoing = {}
    for facing, start, end in ((padded[:-2, 1:-1], (0, 0), (0, 1)), (padded[1:-1, 2:], (0, 1), (1, 1)),
                               (padded[2:, 1:-1], (1, 1), (1, 0)), (padded[1:-1, :-2], (1, 0), (0, 0))):
        for r, c in zip(*(axis.tolist() for axis in np.nonzero(inside & ~facing))):
            outgoing.setdefault((r + start[0], c + start[1]), []).append((r + end[0], c + end[1]))
    lines = []
    while outgoing:
        first = next(iter(outgoing))
        loop, corner = [first], first
        while True:
            ends = outgoing[corner]
            following = ends.pop()
            if not ends:
                del outgoing[corner]
            loop.append(following)
            corner = following
            if corner == first:
                break
        points = np.asarray(loop[:-1], np.float64)
        step = np.roll(points, -1, axis=0) - points
        turns = (step != np.roll(step, 1, axis=0)).any(axis=1)   # keep the corners where the line turns
        points = points[turns]
        points = np.vstack([points, points[:1]])
        lines.append(np.column_stack((points[:, 1] - 0.5, points[:, 0] - 0.5)))
    return lines


def _quantiles(values):
    values = np.asarray(values, np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    return {str(q): float(v) for q, v in zip(_QUANTILES, np.quantile(values, _QUANTILES))}


def _outline_segments(plane):
    """(x, y) contours of every territory of a YX label plane: closed lines halfway between its pixel
    centres and the outside ones, in pixel index coordinates."""
    from scipy import ndimage
    from skimage.measure import find_contours

    segments = []
    for value, box in enumerate(ndimage.find_objects(plane), start=1):
        if box is None:
            continue
        y0, x0 = box[0].start, box[1].start
        # One pixel of background around the box closes the contours of a territory at its border.
        mask = np.pad(plane[box] == value, 1).astype(np.float64)
        segments += [np.column_stack((c[:, 1] + x0 - 1, c[:, 0] + y0 - 1)) for c in find_contours(mask, 0.5)]
    return segments


def summarize_assignment(result: AssignmentResult) -> dict:
    """Counts of an assignment: the record's totals, then flags, nuclei, quantiles and the exclusion.

    Keys: the totals of ``record["counts"]`` (molecules per status, cells, kept
    and excluded cells, nuclei, cells per correspondence status, kept cells per
    compartment state, assigned molecules per compartment value); ``cells_by_status``
    and ``cells_by_flag``; ``nuclei_by_status`` (None without nuclei);
    ``quantiles`` (0, 0.25, 0.5, 0.75 and 1 of ``size_voxels``, ``size_physical``,
    ``n_molecules`` and ``n_nuclei``, None when no finite value exists); and
    ``exclusion``, the totals before (every cell) and after (kept cells) the
    exclusion, each with the same five keys: ``cells``, ``molecules`` (in cells),
    and the ``whole``, ``nucleus`` and ``cytoplasm`` totals. An excluded cell has
    no compartment counts, so ``nucleus`` and ``cytoplasm`` are equal before and
    after; without an excluded cell, ``before`` equals ``after``.

    Raises
    ------
    TypeError
        result is not an AssignmentResult.
    """
    if not isinstance(result, AssignmentResult):
        raise TypeError("result must be an AssignmentResult")
    cells, molecules, counts = result.cells, result.molecules, result.counts
    status = molecules.assignment_status.astype(object)
    flags = cells.correspondence_flags.dropna().astype(object)
    totals = {c: int(counts.loc[counts.compartment.eq(c).to_numpy(dtype=bool), "count"].sum())
              for c in ("whole", "nucleus", "cytoplasm")}
    kept = cells.status.eq("kept").to_numpy(dtype=bool)
    summary = dict(result.record.get("counts", {}))
    summary.update({
        "molecules_by_status": {s: int(status.eq(s).sum()) for s in ASSIGNMENT_STATUSES},
        "cells_by_status": {s: int(cells.status.eq(s).sum()) for s in CELL_STATUSES},
        "cells_by_correspondence": {s: int(cells.correspondence.eq(s).sum()) for s in CELL_CORRESPONDENCE},
        "cells_by_flag": {f: int(sum(f in v.split(";") for v in flags)) for f in CORRESPONDENCE_FLAGS},
        "cells_by_compartments": {s: int(cells.compartments.eq(s).sum()) for s in COMPARTMENT_STATES},
        "nuclei_by_status": (None if result.nuclei is None else
                             {s: int(result.nuclei.status.eq(s).sum()) for s in NUCLEUS_STATUSES}),
        "quantiles": {"size_voxels": _quantiles(cells.size_voxels),
                      "size_physical": _quantiles(cells.size_physical),
                      "n_molecules": _quantiles(cells.n_molecules),
                      "n_nuclei": _quantiles(cells.n_nuclei.astype("Float64").to_numpy(np.float64, na_value=np.nan))},
        "exclusion": {
            # Excluded cells have no compartment counts: the compartment totals are those after.
            "before": {"cells": len(cells), "molecules": int(status.isin(("assigned", "excluded_cell")).sum()),
                       "whole": int(cells.n_molecules.sum()), "nucleus": totals["nucleus"],
                       "cytoplasm": totals["cytoplasm"]},
            "after": {"cells": int(kept.sum()), "molecules": int(status.eq("assigned").sum()), **totals},
        },
    })
    return summary




def plot_assignment(result: AssignmentResult, *, image=None, view=None, z=None,
                    ownership: FOVOwnership | None = None):
    """One row of four panels: cell centres, molecules by status, voxels and molecules per cell.

    Panel 1 shows ``image`` in grey scale (scaled between its 0.5 and 99.5
    percentiles; none by default) with the territory outlines as green lines
    0.8 points wide and one dot of 30 points² with a thin black edge per cell
    centre (both sizes independent of the image size), coloured by the cell's
    status: ``kept`` blue and
    ``excluded_no_nucleus`` orange (drawn, and listed in the legend as
    ``excluded``, only when the result has an excluded cell); panel 2 the same
    image and outlines with the molecules, ``assigned`` blue, ``unassigned`` red
    and ``excluded_cell`` orange (drawn, and listed in the legend as
    ``excluded``, only when the result has such molecules); ``outside_grid``
    molecules are not drawn. Panels 3 and 4 are the histograms of the voxels
    and of the molecules per cell, over every cell. The territories are the
    ones assign used (the expanded ones when it expanded), and a cell's centre
    is the centroid of that territory.

    ``view="z_max"`` (the default without ``z``) draws the Z maximum of the
    territories and of a ZYX ``image``, every molecule with a position on the
    grid and every cell centre. ``view="single_layer"`` (the default with ``z``)
    draws plane ``z`` (default the middle plane ``Z // 2``): the outlines and
    the image of that plane, the molecules whose sampled voxel has that Z index
    and the centres of the cells whose territory occurs in it. It needs
    territories with Z > 1; a plane or Z=1 result has the ``z_max`` view only. A
    YX ``image`` is drawn as is in both views.

    ``ownership`` (a :class:`FOVOwnership`, from
    :func:`starfinder.assembly.fov_ownership`) shows what the sample assembly
    decided for the FOV, in both views. Panels 1 and 2 then draw the closed
    boundary of the FOV's ownership region as a yellow line 1.2 points wide
    (the image border included where the region reaches it, legend entry
    ``ownership region``); a cell or molecule the assembly excluded is drawn
    grey (legend entry ``excluded from sample``, only when the panel draws
    one), every other keeps the colour of its status; and each panel's x-axis
    label states how many of the cells (panel 1) or molecules (panel 2) it
    draws the assembly retained. Panels 3 and 4 do not change.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    TypeError
        result is not an AssignmentResult, or ownership not a FOVOwnership.
    ValueError
        An unknown view, ``z`` with the ``z_max`` view, the ``single_layer``
        view of a Z=1 result, ``z`` outside the territories, an image that
        does not match their Y, X (or, in the ``single_layer`` view, their Z),
        an ownership region of another Y, X, or an ownership record naming a
        cell or molecule the result does not have.
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    if not isinstance(result, AssignmentResult):
        raise TypeError("result must be an AssignmentResult")
    view = ("z_max" if z is None else "single_layer") if view is None else view
    if view not in _VIEWS:
        raise ValueError(f"view must be one of {_VIEWS}; got {view!r}")
    territories = result.territories if result.territories is not None else result.cell_labels
    n_z = territories.shape[0]
    if view == "z_max":
        if z is not None:
            raise ValueError("z selects the plane of the single_layer view; the z_max view takes none")
        plane = territories.max(axis=0)
    else:
        if n_z == 1:
            raise ValueError("the single_layer view needs territories with Z > 1; "
                             "a plane or Z=1 result has the z_max view only")
        z = n_z // 2 if z is None else z
        if isinstance(z, bool) or not isinstance(z, (int, np.integer)) or not 0 <= z < n_z:
            raise ValueError(f"z must be a plane of the territories (0 to {n_z - 1}); got {z!r}")
        plane = territories[z]
    background = None
    if image is not None:
        background = np.asarray(image)
        if background.ndim == 3:
            if view == "z_max" or background.shape[0] == 1:
                background = background.max(axis=0)
            elif background.shape[0] == n_z:
                background = background[z]
            else:
                raise ValueError(f"image Z {background.shape[0]} differs from the territories' {n_z}")
        if background.shape != plane.shape:
            raise ValueError(f"image Y, X {background.shape} differ from the territories' {plane.shape}")
    if ownership is not None:
        _check_ownership(ownership, result, plane.shape)

    molecules = result.molecules
    drawn = ~molecules.assignment_status.eq("outside_grid").to_numpy(dtype=bool)
    if view == "single_layer":
        drawn &= molecules.voxel_z.eq(z).fillna(False).to_numpy(dtype=bool)
    molecules = molecules[drawn]
    statuses = [s for s in _STATUS_COLOURS
                if s != "excluded_cell" or result.molecules.assignment_status.eq(s).any()]
    cells = result.cells
    cell_statuses = [s for s in _CENTRE_COLOURS if s == "kept" or cells.status.eq(s).any()]
    prefix = "expanded_" if result.territories is not None else ""
    centres = cells
    if view == "single_layer":
        centres = cells[np.isin(cells.cell_id.to_numpy(np.int64), np.unique(plane))]
    where = "Z maximum" if view == "z_max" else f"z = {z}"
    # Molecule marker area in points², smaller as the image grows: 12 up to about 250 pixels, 2 from 1500.
    size = float(np.clip(3000.0 / max(plane.shape), 2.0, 12.0))
    legend = dict(loc="upper right", fontsize="small")

    # With ownership, the cells and molecules the assembly excluded leave their status colour for grey.
    gone_centres = np.zeros(len(centres), bool)
    gone_molecules = np.zeros(len(molecules), bool)
    if ownership is not None:
        gone_centres = np.isin(centres.cell_id.to_numpy(np.int64), np.fromiter(ownership.excluded_cells, np.int64))
        gone_molecules = molecules.spot_id.astype(object).isin(ownership.excluded_molecules).to_numpy(dtype=bool)

    def retained(ax, gone, what):
        if ownership is not None:
            ax.set_xlabel(f"{len(gone) - int(gone.sum())} of {len(gone)} {what} retained in the sample")

    segments = _outline_segments(plane)
    if background is not None and background.size:
        low, high = np.percentile(background, (0.5, 99.5))
    figure, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax in axes[:2]:
        if background is not None:
            ax.imshow(background, cmap="gray", vmin=low, vmax=max(high, low + 1e-12), interpolation="nearest",
                      label="image")
        # Unlabelled (no legend entry); the gid names the artist.
        ax.add_collection(LineCollection(segments, colors=_OUTLINE_COLOUR, linewidths=_OUTLINE_WIDTH,
                                         gid="outlines"), autolim=False)
        if ownership is not None:
            # Unclipped and above the axes frame, so that the line stays whole where the region meets the border.
            ax.add_collection(LineCollection(_region_lines(ownership.region), colors=_REGION_COLOUR,
                                             linewidths=_REGION_WIDTH, gid="ownership_region",
                                             label=_REGION_LEGEND, clip_on=False, zorder=2.6), autolim=False)
        ax.set_xlim(-0.5, plane.shape[1] - 0.5)
        ax.set_ylim(plane.shape[0] - 0.5, -0.5)
    ax = axes[0]
    x, y = f"{prefix}centroid_x", f"{prefix}centroid_y"
    for status in cell_statuses:
        chosen = centres[centres.status.eq(status).to_numpy(dtype=bool) & ~gone_centres]
        ax.scatter(chosen[x], chosen[y], s=_CENTRE_AREA,
                   c=_CENTRE_COLOURS[status], edgecolors=_CENTRE_EDGE[0], linewidths=_CENTRE_EDGE[1],
                   label=_CENTRE_LEGEND[status])
    if gone_centres.any():
        chosen = centres[gone_centres]
        ax.scatter(chosen[x], chosen[y], s=_CENTRE_AREA, c=_EXCLUDED_COLOUR, edgecolors=_CENTRE_EDGE[0],
                   linewidths=_CENTRE_EDGE[1], label=_EXCLUDED_LEGEND, gid="excluded_from_sample")
    ax.legend(**legend)
    ax.set_title(f"cell centres, {where}")
    retained(ax, gone_centres, "cells")
    ax = axes[1]
    for status in statuses:
        chosen = molecules[molecules.assignment_status.eq(status).to_numpy(dtype=bool) & ~gone_molecules]
        ax.scatter(chosen.x, chosen.y, s=size, c=_STATUS_COLOURS[status], linewidths=0,
                   label=_STATUS_LEGEND[status])
    if gone_molecules.any():
        chosen = molecules[gone_molecules]
        ax.scatter(chosen.x, chosen.y, s=size, c=_EXCLUDED_COLOUR, linewidths=0, label=_EXCLUDED_LEGEND,
                   gid="excluded_from_sample")
    ax.legend(**legend, markerscale=max(1.0, 12.0 / size))
    ax.set_title(f"molecules, {where}")
    retained(ax, gone_molecules, "molecules")
    for ax, column, label in ((axes[2], f"{prefix}size_voxels", "voxels per cell"),
                              (axes[3], "n_molecules", "molecules per cell")):
        values = cells[column].dropna().to_numpy(np.float64)
        if len(values):
            ax.hist(values, bins=min(30, max(1, len(np.unique(values)))))
        ax.set_title(label)
    figure.tight_layout()
    if ownership is not None:
        # A second pass: the first measures the x-axis labels of the image panels before their aspect shrinks them,
        # and leaves too little room below a square image.
        figure.tight_layout()
    return figure
