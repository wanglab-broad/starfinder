"""On-demand diagnostics of an assignment: a summary mapping and a figure."""
from __future__ import annotations

import numpy as np

from ._assign import AssignmentResult
from ._config import (ASSIGNMENT_STATUSES, CELL_CORRESPONDENCE, CELL_STATUSES, COMPARTMENT_STATES,
                      CORRESPONDENCE_FLAGS, NUCLEUS_STATUSES)

_QUANTILES = (0.0, 0.25, 0.5, 0.75, 1.0)
# plot_assignment: the two views, the outline colour, the colour of each drawn molecule status and cell
# status (a kept cell as an assigned molecule, an excluded one as an excluded_cell molecule), and the
# legend text of each.
_VIEWS = ("z_max", "single_layer")
_OUTLINE_COLOUR = "lime"
_STATUS_COLOURS = {"assigned": "dodgerblue", "unassigned": "red", "excluded_cell": "orange"}
_STATUS_LEGEND = {"assigned": "assigned", "unassigned": "unassigned", "excluded_cell": "excluded"}
_CENTRE_COLOURS = {"kept": _STATUS_COLOURS["assigned"],
                   "excluded_no_nucleus": _STATUS_COLOURS["excluded_cell"]}
_CENTRE_LEGEND = {"kept": "kept", "excluded_no_nucleus": "excluded"}


def _quantiles(values):
    values = np.asarray(values, np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    return {str(q): float(v) for q, v in zip(_QUANTILES, np.quantile(values, _QUANTILES))}


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




def plot_assignment(result: AssignmentResult, *, image=None, view=None, z=None):
    """One row of four panels: cell centres, molecules by status, voxels and molecules per cell.

    Panel 1 shows ``image`` in grey scale (scaled between its 0.5 and 99.5
    percentiles; none by default) with the territory outlines in green and one
    dot per cell centre, coloured by the cell's status: ``kept`` blue and
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

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    TypeError
        result is not an AssignmentResult.
    ValueError
        An unknown view, ``z`` with the ``z_max`` view, the ``single_layer``
        view of a Z=1 result, ``z`` outside the territories, or an image that
        does not match their Y, X (or, in the ``single_layer`` view, their Z).
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from skimage.segmentation import find_boundaries

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
    # Marker area in points², smaller as the image grows: 12 up to about 250 pixels, 2 from 1500.
    size = float(np.clip(3000.0 / max(plane.shape), 2.0, 12.0))
    legend = dict(loc="upper right", fontsize="small", markerscale=max(1.0, 12.0 / size))

    outline = np.ma.masked_where(~find_boundaries(plane, mode="inner"), np.ones(plane.shape))
    if background is not None and background.size:
        low, high = np.percentile(background, (0.5, 99.5))
    figure, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax in axes[:2]:
        if background is not None:
            ax.imshow(background, cmap="gray", vmin=low, vmax=max(high, low + 1e-12), interpolation="nearest",
                      label="image")
        ax.imshow(outline, cmap=ListedColormap([_OUTLINE_COLOUR]), interpolation="nearest", label="outlines")
        ax.set_xlim(-0.5, plane.shape[1] - 0.5)
        ax.set_ylim(plane.shape[0] - 0.5, -0.5)
    ax = axes[0]
    for status in cell_statuses:
        chosen = centres[centres.status.eq(status).to_numpy(dtype=bool)]
        ax.scatter(chosen[f"{prefix}centroid_x"], chosen[f"{prefix}centroid_y"], s=2 * size,
                   c=_CENTRE_COLOURS[status], linewidths=0, label=_CENTRE_LEGEND[status])
    ax.legend(**legend)
    ax.set_title(f"cell centres, {where}")
    ax = axes[1]
    for status in statuses:
        chosen = molecules[molecules.assignment_status.eq(status).to_numpy(dtype=bool)]
        ax.scatter(chosen.x, chosen.y, s=size, c=_STATUS_COLOURS[status], linewidths=0,
                   label=_STATUS_LEGEND[status])
    ax.legend(**legend)
    ax.set_title(f"molecules, {where}")
    for ax, column, label in ((axes[2], f"{prefix}size_voxels", "voxels per cell"),
                              (axes[3], "n_molecules", "molecules per cell")):
        values = cells[column].dropna().to_numpy(np.float64)
        if len(values):
            ax.hist(values, bins=min(30, max(1, len(np.unique(values)))))
        ax.set_title(label)
    figure.tight_layout()
    return figure
