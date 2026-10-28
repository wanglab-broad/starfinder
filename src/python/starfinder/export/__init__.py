"""Sample outputs: the export entry, its record and the fused sample image.

``Dataset.export`` is the one sample-level export entry: it calls the parts ``assemble``
(:mod:`starfinder.assembly`), ``fuse`` (:func:`fuse_images`), ``fuse_labels`` and
``export_spatialdata`` in order, each turned on or off by :class:`ExportConfig`, and writes
everything to the sample's export folder ``<OUTPUT>/export/<sample>/`` with one record ``export.json`` (a
section per part, :class:`ExportRecord`). A part that cannot run (an extra not installed,
an input file missing) is recorded ``not_run`` with :class:`ExportIncompleteWarning`, and
the files of the earlier parts are still written. :func:`fuse_images` makes the fused image with
multiview-stitcher's ``fuse`` from the placement of either backend (:class:`FusionInput`,
:class:`FusionConfig`): a 2D maximum projection at scale 0.5 by default, or the 3D stack
(``dimensionality``), on the library's union grid (:func:`fused_grid`,
:class:`FusedGrid`), written as OME-TIFF by Starfinder and on request as the library's
OME-Zarr (:class:`FusionRecord`). :func:`fuse_labels` writes the assembled label image
``cell_labels.ome.tif`` (:class:`LabelConfig`, :class:`LabelRecord`): the retained cells of
the sample, each drawn whole with its ``instance_id``, placed by the library at order 0
with the first positive view and never blended, at scale 0.25 by default (off by default
in the entry). :func:`export_spatialdata` writes the SpatialData store ``sample.zarr``
(:class:`SpatialDataExportConfig`; off by default in the entry): the sample tables as
points, shapes and tables, and each fused and label image once, read from its OME-TIFF,
with one level; :func:`lookup_source` traces a molecule or cell back to its FOV's
assignment. :func:`check_export` checks a folder against its record. :func:`segment_fused`
segments the fused projection of a sample on the route after fusion (:data:`ROUTES`,
:class:`SampleSegmentationConfig`, :class:`SampleSegmentationRun`) and writes one label
image per run and the ``segmentation`` section (:class:`~starfinder.segmentation.SampleSegmentation`,
:func:`load_sample_segmentation`); :func:`compare_routes` reports how the cells and
molecules of an export folder of each route correspond (:class:`RouteComparison`) and
decides nothing; see design/specs/sample-segmentation.md.
Importing this module imports neither multiview-stitcher, zarr nor spatialdata. See
design/specs/sample-output.md.
"""
# §2.10, §2.9
from ._check import check_export
from ._compare import RouteComparison, compare_routes
from ._config import ROUTES as _ROUTES
from ._config import (ExportConfig, FusionConfig, LabelConfig, SampleSegmentationConfig, SampleSegmentationRun,
                      SpatialDataExportConfig)
from ._fuse import FusionInput, FusionRecord, fuse_images
from ._grid import FusedGrid, fused_grid
from ._labels import LabelRecord, fuse_labels
from ._record import ExportIncompleteWarning, ExportRecord
from ._segment import load_sample_segmentation, segment_fused
from ._spatialdata import export_spatialdata, lookup_source

#: The values of ``ExportConfig.route``: ``per_fov`` (segmentation per FOV followed by assembly) and
#: ``after_fusion`` (segmentation of the fused projection); None takes the default by ``dimensionality``.
ROUTES = _ROUTES

__all__ = ["ExportConfig", "FusionConfig", "LabelConfig", "SpatialDataExportConfig", "fuse_images", "fused_grid",
           "FusionInput", "FusedGrid", "FusionRecord", "fuse_labels", "LabelRecord", "ExportRecord",
           "ExportIncompleteWarning", "export_spatialdata", "lookup_source", "check_export", "ROUTES",
           "SampleSegmentationConfig", "SampleSegmentationRun", "segment_fused",
           "load_sample_segmentation", "compare_routes", "RouteComparison"]
