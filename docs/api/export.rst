starfinder.export
=================

Sample export, image and label fusion, and SpatialData output. The
:doc:`sample-output guide <../sample-output-contract>` shows the export workflow; the
:doc:`sample-output reference <../sample-output-reference>` lists options and files.

.. currentmodule:: starfinder.export

Export configuration and record
-------------------------------

.. autosummary::
   :toctree: generated

   ExportConfig
   ExportIncompleteWarning
   ExportRecord
   ROUTES

Image and label fusion
----------------------

.. autosummary::
   :toctree: generated

   fuse_images
   fuse_labels
   fused_grid
   FusedGrid
   FusionConfig
   FusionInput
   FusionRecord
   LabelConfig
   LabelRecord

Sample segmentation and route comparison
----------------------------------------

.. autosummary::
   :toctree: generated

   compare_routes
   load_sample_segmentation
   RouteComparison
   SampleSegmentationConfig
   SampleSegmentationRun
   segment_fused

SpatialData and inspection
--------------------------

.. autosummary::
   :toctree: generated

   check_export
   export_spatialdata
   lookup_source
   SpatialDataExportConfig
