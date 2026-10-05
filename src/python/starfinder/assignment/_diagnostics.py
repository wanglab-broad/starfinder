"""On-demand diagnostics of an assignment: a summary mapping and a figure."""
from __future__ import annotations

import numpy as np

from ._assign import AssignmentResult
from ._config import (ASSIGNMENT_STATUSES, CELL_CORRESPONDENCE, CELL_STATUSES, COMPARTMENT_STATES,
                      CORRESPONDENCE_FLAGS, NUCLEUS_STATUSES)

_QUANTILES = (0.0, 0.25, 0.5, 0.75, 1.0)
_STATUS_COLOURS = {"assigned": "tab:green", "unassigned": "tab:grey", "excluded_cell": "tab:orange",
                   "outside_grid": "tab:red"}


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
    exclusion: cells, molecules in cells, and the whole, nuclear and cytoplasmic
    totals.

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
            "before": {"cells": len(cells), "molecules": int(status.isin(("assigned", "excluded_cell")).sum()),
                       "whole": int(cells.n_molecules.sum())},
            "after": {"cells": int(kept.sum()), "molecules": int(status.eq("assigned").sum()), **totals},
        },
    })
    return summary


def plot_assignment(result: AssignmentResult, *, image=None, z=None):
    """Territory outlines over an image with molecules coloured by status, and three histograms.

    The left panel shows the outlines of the territories (the expanded ones when
    assign expanded) as their Z maximum, or plane ``z``, over ``image`` (a ZYX or
    YX array reduced the same way; none by default), with each molecule as a dot
    coloured by its ``assignment_status`` (``outside_grid`` molecules are not
    drawn). The other panels are the histograms of ``size_voxels``,
    ``n_molecules`` and ``n_nuclei`` (when nuclei were given).

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    TypeError
        result is not an AssignmentResult.
    ValueError
        z is outside the territories, or image does not match their Y, X.
    """
    import matplotlib.pyplot as plt
    from skimage.segmentation import find_boundaries

    if not isinstance(result, AssignmentResult):
        raise TypeError("result must be an AssignmentResult")
    territories = result.territories if result.territories is not None else result.cell_labels
    if z is None:
        plane = territories.max(axis=0)
    else:
        if isinstance(z, bool) or not isinstance(z, (int, np.integer)) or not 0 <= z < territories.shape[0]:
            raise ValueError(f"z must be a plane of the territories (0 to {territories.shape[0] - 1}); got {z!r}")
        plane = territories[z]
    background = None
    if image is not None:
        background = np.asarray(image)
        if background.ndim == 3:
            background = background.max(axis=0) if z is None or background.shape[0] == 1 else background[z]
        if background.shape != plane.shape:
            raise ValueError(f"image Y, X {background.shape} differ from the territories' {plane.shape}")
    figure, axes = plt.subplots(1, 4, figsize=(16, 4))
    ax = axes[0]
    if background is not None:
        ax.imshow(background, cmap="gray")
    outline = np.ma.masked_where(~find_boundaries(plane, mode="inner"), np.ones(plane.shape))
    ax.imshow(outline, cmap="autumn", alpha=0.8, interpolation="nearest")
    molecules = result.molecules
    if z is not None and territories.shape[0] > 1:
        molecules = molecules[molecules.voxel_z.eq(z).fillna(False).to_numpy(dtype=bool)]
    for status, colour in _STATUS_COLOURS.items():
        chosen = molecules[molecules.assignment_status.eq(status).to_numpy(dtype=bool)]
        if status != "outside_grid" and len(chosen):
            ax.scatter(chosen.x, chosen.y, s=4, c=colour, label=status)
    ax.set_title("territories" + ("" if z is None else f", z = {z}"))
    ax.set_xlim(-0.5, plane.shape[1] - 0.5)
    ax.set_ylim(plane.shape[0] - 0.5, -0.5)
    if len(molecules):
        ax.legend(loc="upper right", fontsize="small")
    cells = result.cells
    for ax, column, label in ((axes[1], "size_voxels", "voxels per cell"),
                              (axes[2], "n_molecules", "molecules per cell"),
                              (axes[3], "n_nuclei", "nuclei per cell")):
        values = cells[column].dropna().to_numpy(np.float64)
        if len(values):
            ax.hist(values, bins=min(30, max(1, len(np.unique(values)))))
        ax.set_title(label)
    figure.tight_layout()
    return figure
