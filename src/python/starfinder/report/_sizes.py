"""The size of the segmented objects (design/specs/report-algorithms.md, "Object sizes").

The equivalent diameter of each nucleus and cell, in µm, across the sample on the sample's route, the rows of
``sizes.csv``, the part ``assembly.sizes`` and the values of the rule ``small_nucleus_fraction``. A display rule, not a
qualification of the segmentation.

Route per FOV: each FOV's own label images (the files ``assignment.json`` names, each checked against its recorded
SHA-256) on the FOV's grid; the objects of a FOV are the labels of its images. Route after fusion: the sample's label
images ``segmentation_<run>.ome.tif`` on the segmentation grid; an object belongs to the FOV that owns its centroid
under the export's ownership map.
"""
from __future__ import annotations

import numpy as np

from starfinder import _layout

from ._values import NA, Unavailable, first

#: The percentiles of the diameter columns of ``sizes.csv`` (NumPy's linear percentile).
PERCENTILES = (5, 25, 50, 75, 95)
#: The two kinds of object, in row order: the table's ``object`` value and the label image's key.
OBJECTS = (("nucleus", "nuclei"), ("cell", "cells"))
#: The plural of each kind, for the text of the figures.
PLURAL = {"nucleus": "nuclei", "cell": "cells"}
_DIAMETER_COLUMNS = ("diameter_p5", "diameter_p25", "diameter_median", "diameter_p75", "diameter_p95")


def equivalent_diameters(labels, spacing_zyx):
    """``(ids, diameters)`` of the objects of a ZYX label image: the nonzero labels in increasing order and their
    equivalent diameters in the unit of ``spacing_zyx``.

    A one-plane image gives the diameter of the circle of the object's area, ``2·sqrt(A / π)``; an image with
    several planes the diameter of the sphere of the object's volume, ``(6·V / π)^(1/3)``. An object's area or
    volume is its voxel count times the voxel's own.
    """
    labels = np.asarray(labels)
    ids, counts = np.unique(labels[labels != 0], return_counts=True)
    sz, sy, sx = (float(v) for v in spacing_zyx)
    if labels.shape[0] == 1:
        return ids, 2.0 * np.sqrt(counts * sy * sx / np.pi)
    return ids, np.cbrt(6.0 * counts * sz * sy * sx / np.pi)


def centroids_yx(labels, ids, spacing_zyx, origin_zyx):
    """The centroids ``(y, x)`` in the frame of the grid of the objects ``ids`` of a ZYX label image, in the unit
    of ``spacing_zyx`` (pixel ``i`` centred at ``origin + i × spacing``), one row per id."""
    labels = np.asarray(labels)
    index = np.searchsorted(ids, labels[labels != 0])
    _, yy, xx = np.nonzero(labels)
    count = np.bincount(index, minlength=len(ids)).astype(np.float64)
    out = np.empty((len(ids), 2))
    for column, (coordinate, step, origin) in enumerate(((yy, spacing_zyx[1], origin_zyx[1]),
                                                          (xx, spacing_zyx[2], origin_zyx[2]))):
        out[:, column] = origin + step * np.bincount(index, weights=coordinate, minlength=len(ids)) / count
    return out


def size_row(scope, fov, kind, diameters, limit):
    """One row of ``sizes.csv``: the count, the percentiles of the diameters, and the objects below ``limit``.

    The percentiles and the fraction of a set without an object are ``undefined``."""
    n = len(diameters)
    row = {"scope": scope, "fov_id": fov, "object": kind, "n_objects": n}
    if n == 0:
        none = Unavailable("undefined", f"no {kind}")
        row.update({c: none for c in _DIAMETER_COLUMNS})
        row.update(below_limit=0, below_fraction=none)
        return row
    row.update(zip(_DIAMETER_COLUMNS, (float(v) for v in np.percentile(diameters, PERCENTILES))))
    below = int(np.count_nonzero(diameters < limit))
    row.update(below_limit=below, below_fraction=below / n)
    return row


class _Sizes:
    """The sizes of a collection's objects, computed once; ``collection`` is the report's ``_Collection``."""

    def __init__(self, collection):
        self.c = collection
        self._done = False

    @property
    def limit_um(self):
        return self.c.config.limits.small_nucleus_diameter_um

    def compute(self):
        """Compute the sizes once.

        Sets ``diameters`` (object → sample-wide array in µm), ``by_fov`` ((fov, object) → array),
        ``rows`` (the rows of ``sizes.csv``) and ``part_rows`` (the rows of ``unavailable.csv`` of the part), and
        ``reason`` (an ``Unavailable`` when the sizes of the whole sample cannot be made, else None).
        """
        if self._done:
            return
        self._done = True
        self.diameters, self.by_fov, self.rows, self.part_rows, self.reason = {}, {}, [], [], None
        c = self.c
        source = first(c.no_cells)
        if source is not None:
            self.reason = source
        elif c.route == "after_fusion":
            self._after_fusion()
        else:
            self._per_fov()
        if self.reason is not None:
            self.part_rows.append(("assembly.sizes", self.reason, ""))
            return
        self._rows()

    # The sources ------------------------------------------------------------------------------------------------

    def _per_fov(self):
        from ._previews import fov_labels
        c = self.c
        samples = {kind: [] for kind, _ in OBJECTS}
        found = False
        for fov in c.fov_ids:
            if fov not in c.assignments:
                if fov in c.lane_missing:
                    self.part_rows.append(("assembly.sizes", c.lane_missing[fov], f"fov_id={fov}"))
                continue
            record, _ = c.assignments[fov]
            folder = _layout.assignment_dir(_layout.fov_dir(c.root, fov), c.assignment)
            labels = fov_labels(c, record, folder)
            if isinstance(labels, Unavailable):
                self.part_rows.append(("assembly.sizes", labels, f"fov_id={fov}"))
                continue
            spacing = ((record.get("grid") or {}).get("metadata") or {}).get("spacing_zyx")
            if spacing is None:
                self.part_rows.append(("assembly.sizes", Unavailable(
                    "not_recorded", f"{c.inputs.relative(folder)}/assignment.json: no grid spacing"),
                    f"fov_id={fov}"))
                continue
            found = True
            for kind, key in OBJECTS:
                if labels[key] is not None:
                    d = equivalent_diameters(labels[key], spacing)[1]
                    self.by_fov[(fov, kind)] = d
                    samples[kind].append(d)
        if not found:
            self.reason = first(*(r for _, r, _ in self.part_rows)) or (
                c.e_state if c.export is None else Unavailable("input_missing", "no FOV has an assignment record"))
            if self.reason is None:  # an export whose assemble part holds nothing for the sample
                self.reason = Unavailable("input_missing", "no FOV has an assignment record")
            self.part_rows = []
            return
        self.diameters = {kind: np.concatenate(v) if v else None for kind, v in samples.items()}

    def _after_fusion(self):
        c = self.c
        sources = c.previewer.label_sources()
        if isinstance(sources, Unavailable):
            self.reason = sources
            return
        files, grid = sources
        spacing, origin = grid["spacing_zyx"], grid["origin_zyx"]
        ownership = self._ownership()
        for kind, key in OBJECTS:
            labels = files[key]
            if labels is None:
                continue
            ids, d = equivalent_diameters(labels, spacing)
            self.diameters[kind] = d
            if isinstance(ownership, Unavailable):
                continue
            owner = ownership.owner(np.column_stack([np.full(len(ids), origin[0]),
                                                     centroids_yx(labels, ids, spacing, origin)]))
            for fov in ownership.owning:
                if fov in c.fov_ids:
                    self.by_fov[(fov, kind)] = d[owner == fov]
        if isinstance(ownership, Unavailable):
            self.part_rows.append(("assembly.sizes", ownership, "scope=fov"))
        self.diameters = {kind: self.diameters.get(kind) for kind, _ in OBJECTS}

    def _ownership(self):
        c = self.c
        if c.placement is None or c.assembly is None:
            return Unavailable("no_placement", "the objects of a FOV are those its ownership region holds")
        return c._ownership()

    # The table --------------------------------------------------------------------------------------------------

    def _rows(self):
        limit = self.limit_um
        for kind, _ in OBJECTS:
            if self.diameters.get(kind) is not None:
                self.rows.append(size_row("sample", NA, kind, self.diameters[kind], limit))
        for fov in self.c.fov_ids:
            for kind, _ in OBJECTS:
                if (fov, kind) in self.by_fov:
                    self.rows.append(size_row("fov", fov, kind, self.by_fov[(fov, kind)], limit))
        if not self.rows:
            self.reason = Unavailable("no_segmentation", "the label images hold no nucleus or cell file")
            self.part_rows = [("assembly.sizes", self.reason, "")]

    def nucleus_fractions(self):
        """FOV → the fraction of its nuclei below the limit, for the rule ``small_nucleus_fraction``; or the
        ``Unavailable`` that stops the rule for the sample (the objects of no FOV can be told)."""
        self.compute()
        if self.reason is not None:
            return self.reason
        rows = [r for r in self.rows if r["scope"] == "fov" and r["object"] == "nucleus"]
        if not rows:
            return first(*(r for _, r, _ in self.part_rows)) or Unavailable(
                "no_segmentation", "no nucleus label image")
        return {r["fov_id"]: r["below_fraction"] for r in rows}
