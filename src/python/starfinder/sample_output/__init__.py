"""Sample outputs (§2.10): the export entry, its record and the fused sample image.

``Dataset.export`` is the one sample-level export entry: it calls the parts ``assemble``
(:mod:`starfinder.assembly`), :func:`fuse`, ``fuse_labels`` and ``export_spatialdata`` in
order, each turned on or off by :class:`ExportConfig`, and writes everything to the
sample's export folder ``<OUTPUT>/export/<sample>/`` with one record ``export.json`` (a
section per part, :class:`ExportRecord`). A part that cannot run (an extra not installed,
an input file missing) is recorded ``not_run`` with :class:`ExportIncompleteWarning`, and
the files of the earlier parts are still written. :func:`fuse` makes the fused image with
multiview-stitcher's ``fuse`` from the placement of either backend (:class:`FusionInput`,
:class:`FusionConfig`): a 2D maximum projection at scale 0.5 by default, or the 3D stack
(``dimensionality``), on the library's union grid (:func:`fused_grid`,
:class:`FusedGrid`), written as OME-TIFF by Starfinder and on request as the library's
OME-Zarr (:class:`FusionRecord`). :func:`fuse_labels` writes the assembled label image
``cell_labels.ome.tif`` (:class:`LabelConfig`, :class:`LabelRecord`): the retained cells of
the sample, each drawn whole with its ``instance_id``, placed by the library at order 0
with the first positive view and never blended, at scale 0.25 by default (off by default
in the entry). :func:`check_export` checks a folder against its record. The SpatialData
part comes with its own issue and is recorded ``off``.
Importing this module imports neither multiview-stitcher, zarr nor spatialdata. See
docs/sample-output-contract.md.
"""
from ._check import check_export
from ._config import ExportConfig, FusionConfig, LabelConfig, SpatialDataExportConfig
from ._fuse import FusionInput, FusionRecord, fuse
from ._grid import FusedGrid, fused_grid
from ._labels import LabelRecord, fuse_labels
from ._record import ExportIncompleteWarning, ExportRecord

__all__ = ["ExportConfig", "FusionConfig", "LabelConfig", "SpatialDataExportConfig", "fuse", "fused_grid",
           "FusionInput", "FusedGrid", "FusionRecord", "fuse_labels", "LabelRecord", "ExportRecord",
           "ExportIncompleteWarning", "check_export"]
