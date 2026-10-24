"""The segmentation previews of the example FOVs on both routes (docs/report-algorithms.md, "How the two routes
differ", "Segmentation and assembly").

Route per FOV (and a sample without an export whose assignment folders exist): the FOV's
own label images, the files ``assignment.json`` ``inputs.cells.file`` and
``inputs.nuclei.file`` name (``cell_labels.tif``, ``nucleus_labels.tif`` or the linked
``segmentation/<run>/labels.tif``), each checked against its recorded SHA-256, on the FOV's
grid, over its reference image (``fovs/<fov>/reference.tif`` by default). A 3D preview shows the plane
``z*`` of the cell label image with the most nonzero voxels, over the maximum of the
reference image over the planes of the slab of 3 µm centred on ``z*``.

Route after fusion: ``segmentation_<run>.ome.tif`` of the cell run (and of the nucleus run
when the export has one) cut to the FOV's footprint (``fovs.csv`` ``footprint_*``), mapped to
the label grid by ``floor((q − O) / S + ½)``, the start included and the end excluded; the
background is the fused image of the marker patterns (the first image of ``export.json``
``fusion.outputs``, its projection, else the maximum over Z of its stack) cut to the same
footprint on its own grid.

An outline is a pixel of a label image whose label differs from one of its four neighbours
(inside the image), computed at full resolution.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from starfinder import _layout

from ._values import Unavailable, first

#: The slab of a 3D preview's background, µm (choice A6, SL1).
SLAB_UM = 3.0
_WRITTEN = ("written", "reused")


@dataclass
class Preview:
    """One example tile: the background and the label images in the same plane, with their extents.

    ``extent`` and ``label_extent`` are ``(x0, x1, y1, y0)`` of the pixel edges (matplotlib's ``extent`` with the
    origin at the top), in the FOV's pixels on the route per FOV and in µm of the sample frame after fusion.
    ``window`` is the label image's pixel range ``(y0, y1, x0, x1)`` the tile shows (the footprint's range after
    fusion; the whole grid per FOV) and ``plane`` its plane.
    """

    fov_id: str
    role: str
    route: str
    background: np.ndarray = field(repr=False)
    cells: np.ndarray = field(repr=False)
    nuclei: np.ndarray | None = field(repr=False)
    extent: tuple
    label_extent: tuple
    window: tuple
    plane: int = 0

    @property
    def cell_outline(self):
        return outline(self.cells)

    @property
    def nucleus_outline(self):
        return None if self.nuclei is None else outline(self.nuclei)


def outline(labels) -> np.ndarray:
    """The pixels of a 2D label image whose label differs from one of their four neighbours inside the image."""
    labels = np.asarray(labels)
    edge = np.zeros(labels.shape, dtype=bool)
    differ = labels[1:, :] != labels[:-1, :]
    edge[1:, :] |= differ
    edge[:-1, :] |= differ
    differ = labels[:, 1:] != labels[:, :-1]
    edge[:, 1:] |= differ
    edge[:, :-1] |= differ
    return edge


def pixel_range(low, high, origin, spacing, size):
    """The pixel indices ``[floor((low − O) / S + ½), floor((high − O) / S + ½))`` of a grid axis, clipped to
    it."""
    start = int(np.floor((low - origin) / spacing + 0.5))
    stop = int(np.floor((high - origin) / spacing + 0.5))
    return max(0, min(size, start)), max(0, min(size, stop))


def _get(mapping, *keys):
    for key in keys:
        if not isinstance(mapping, dict) or key not in mapping:
            return None
        mapping = mapping[key]
    return mapping


def _read_image(inputs, path, kind, recorded=None):
    """A TIFF read whole with tifffile (squeezed to ZYX or YX), or the reason it cannot be read."""
    import tifffile
    data = inputs.read(path, kind, recorded)
    if isinstance(data, Unavailable):
        return data
    return np.asarray(tifffile.imread(path))


def _zyx(image):
    image = np.asarray(image)
    if image.ndim == 2:
        return image[np.newaxis]
    if image.ndim == 4:  # ZYXC with one channel (the saved reference image of a one-channel round)
        return image.max(axis=-1)
    return image


class _Previews:
    """The previews of a collection's example FOVs; ``collection`` is the report's ``_Collection``."""

    def __init__(self, collection):
        self.c = collection

    def _key(self, fov, role):
        return f"fov_id={fov};role={role}"

    def per_fov(self, fov, role):
        c = self.c
        if c.no_cells is not None:
            return c.no_cells
        folder = _layout.assignment_dir(_layout.fov_dir(c.root, fov), c.assignment)
        if fov not in c.assignments:
            if fov in c.lane_missing:
                return c.lane_missing[fov]
            return Unavailable("input_missing", f"{c.inputs.relative(folder)}/{_layout.ASSIGNMENT_RECORD}")
        record, _ = c.assignments[fov]
        labels = {}
        for key in ("cells", "nuclei"):
            entry = _get(record, "inputs", key, "file")
            if not entry or not entry.get("path"):
                if key == "cells":
                    return Unavailable("input_missing", f"{c.inputs.relative(folder)}/assignment.json names no "
                                                        "cell label file")
                labels[key] = None
                continue
            path = Path(os.path.normpath(folder / entry["path"]))
            image = _read_image(c.inputs, path, "label_image", entry.get("sha256"))
            if isinstance(image, Unavailable):
                return image
            labels[key] = _zyx(image)
        pattern = getattr(c, "reference_image", None) or _layout.REFERENCE_IMAGE
        path = _layout.reference_image_path(c.root, fov, pattern, getattr(c, "reference_image_import", None))
        background = _read_image(c.inputs, path, "reference_image")
        if isinstance(background, Unavailable):
            return background
        background = _zyx(background)
        cells, nuclei = labels["cells"], labels["nuclei"]
        plane = int(np.argmax((cells != 0).reshape(cells.shape[0], -1).sum(axis=1)))
        spacing_z = float(_get(record, "grid", "metadata", "spacing_zyx")[0])
        offsets = (np.arange(background.shape[0]) - plane) * spacing_z
        slab = (offsets >= -SLAB_UM / 2) & (offsets < SLAB_UM / 2)
        ny, nx = cells.shape[1:]
        extent = (-0.5, nx - 0.5, ny - 0.5, -0.5)
        return Preview(fov, "", "per_fov", background[slab].max(axis=0), cells[plane],
                       None if nuclei is None else nuclei[plane], extent, extent, (0, ny, 0, nx), plane)

    def _after_fusion_sources(self):
        """The cell and nucleus label files and the fused background of the export, or the reason it lacks one."""
        if hasattr(self, "_sources"):
            return self._sources
        from starfinder.export._record import part_status
        c = self.c
        export, export_dir = c.export, c.export_dir
        missing = first(*(Unavailable("part_not_written", f"{part}: {part_status(export, part)}")
                          for part in ("segment", "fuse") if part_status(export, part) not in _WRITTEN),
                        c.no_cells)
        if missing is not None:
            self._sources = missing
            return missing
        section = export.get("segmentation") or {}
        runs = {r.get("name"): r for r in section.get("runs") or []}
        cell_run = c.segmentation or section.get("cells")
        nucleus_run = None if c.segmentation else section.get("nuclei")
        grid = _get(section, "inputs", "grid") or {}
        files = {}
        for key, name in (("cells", cell_run), ("nuclei", nucleus_run)):
            if name is None:
                files[key] = None
                continue
            run = runs.get(name) or {}
            if not run.get("path"):
                self._sources = Unavailable("input_missing", f"export.json segmentation.runs holds no file of {name}")
                return self._sources
            image = _read_image(c.inputs, export_dir / run["path"], "label_image", run.get("sha256"))
            if isinstance(image, Unavailable):
                self._sources = image
                return image
            files[key] = _zyx(image)
        outputs = _get(export, "fusion", "outputs") or []
        name = outputs[0].get("image") if outputs else None
        entries = [o for o in outputs if o.get("image") == name and o.get("format", "ome_tiff") == "ome_tiff"]
        entry = next((o for o in entries if o.get("dimensionality") == "projection"), entries[0] if entries else None)
        if entry is None:
            self._sources = Unavailable("part_not_written", "fuse: export.json fusion.outputs holds no image")
            return self._sources
        image = _read_image(c.inputs, export_dir / entry["path"], "fused_image", entry.get("sha256"))
        if isinstance(image, Unavailable):
            self._sources = image
            return image
        self._sources = (files, grid, _zyx(image).max(axis=0), entry)
        return self._sources

    def after_fusion(self, fov, role):
        c = self.c
        sources = self._after_fusion_sources()
        if isinstance(sources, Unavailable):
            return sources
        files, grid, fused, entry = sources
        row = next(r for r in c.fovs if r["fov_id"] == fov)
        footprint = [row[k] for k in ("footprint_y0", "footprint_y1", "footprint_x0", "footprint_x1")]
        missing = first(*footprint)
        if missing is not None:
            return missing
        y0, y1, x0, x1 = (float(v) for v in footprint)
        (_, sy, sx), (_, oy, ox) = grid["spacing_zyx"], grid["origin_zyx"]
        cells = files["cells"]
        plane = int(np.argmax((cells != 0).reshape(cells.shape[0], -1).sum(axis=1)))
        iy0, iy1 = pixel_range(y0, y1, oy, sy, cells.shape[1])
        ix0, ix1 = pixel_range(x0, x1, ox, sx, cells.shape[2])
        (_, fy, fx), (_, gy, gx) = entry["spacing_zyx"], entry["origin_zyx"]
        jy0, jy1 = pixel_range(y0, y1, gy, fy, fused.shape[0])
        jx0, jx1 = pixel_range(x0, x1, gx, fx, fused.shape[1])

        def edges(start, stop, origin, spacing):
            return origin + (start - 0.5) * spacing, origin + (stop - 0.5) * spacing
        ly, lx = edges(iy0, iy1, oy, sy), edges(ix0, ix1, ox, sx)
        by, bx = edges(jy0, jy1, gy, fy), edges(jx0, jx1, gx, fx)
        nuclei = files["nuclei"]
        return Preview(fov, "", "after_fusion", fused[jy0:jy1, jx0:jx1],
                       cells[plane, iy0:iy1, ix0:ix1],
                       None if nuclei is None else nuclei[min(plane, nuclei.shape[0] - 1), iy0:iy1, ix0:ix1],
                       (bx[0], bx[1], by[1], by[0]), (lx[0], lx[1], ly[1], ly[0]), (iy0, iy1, ix0, ix1), plane)

    def build(self):
        """``(previews, rows)``: per role its :class:`Preview` or :class:`Unavailable`, and the rows of
        ``unavailable.csv`` (part ``assembly.examples``) of the roles without a preview."""
        c = self.c
        if c.example_reason is not None:
            return {}, [("assembly.examples", c.example_reason, "")]
        previews, rows = {}, []
        for example in c.examples:
            if c.route == "after_fusion":
                preview = self.after_fusion(example.fov_id, example.role)
            else:
                preview = self.per_fov(example.fov_id, example.role)
            if isinstance(preview, Unavailable):
                rows.append(("assembly.examples", preview, self._key(example.fov_id, example.role)))
            else:
                preview.role = example.role
            previews[example.role] = preview
        return previews, rows
