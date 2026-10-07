starfinder.sample_output
========================

Sample outputs (§2.10): the one sample-level export entry, its record and the fused
sample image. See :doc:`../sample-output-contract`.

:py:meth:`~starfinder.dataset.Dataset.export` writes everything to the sample's export
folder ``<OUTPUT>/export/<sample>/``. After the checks before anything is written (the
configs, the placement's review, the ``anndata`` extra, the assembly, the fusion inputs)
it removes the files the previous ``export.json`` lists for each part that runs or is off
and calls the parts in order, writing ``export.json`` after each: ``assemble`` (saved with
:py:func:`~starfinder.assembly.save_assembly`), :py:func:`~starfinder.sample_output.fuse`,
``fuse_labels`` and ``export_spatialdata``. :py:class:`~starfinder.sample_output.ExportConfig`
turns each part on or off and holds the configuration of each; the label and SpatialData
parts are not implemented yet and are recorded ``off``. A part that cannot run (an extra
not installed, a source image missing) is recorded ``not_run`` with its reason and
:py:class:`~starfinder.sample_output.ExportIncompleteWarning`; the files of the earlier
parts are still written and ``complete`` is false. The method returns an
:py:class:`~starfinder.sample_output.ExportRecord`, and
:py:func:`~starfinder.sample_output.check_export` checks a folder against its record.

:py:func:`~starfinder.sample_output.fuse` makes the fused image with
multiview-stitcher's ``fusion.fuse`` from a placement of either backend and one
:py:class:`~starfinder.sample_output.FusionInput` per included FOV
(``image_supported`` or ``metadata_only``). :py:class:`~starfinder.sample_output.FusionConfig`
selects the source images (``reference_stain`` by default), the dimensionality
(``projection``, the default 2D maximum projection; ``stack``, the 3D stack alone, for
dense 3D tissue; or ``both``), the projection rule (``max`` or ``sum``, float32), the
scale ``scale_yx = 1/k`` in Y and X (0.5 by default; Z is never reduced; requested from
the library's level 1 of each FOV), the standalone OME-Zarr and the OME-TIFF SubIFD
levels. The output grid is the library's union grid of the included FOVs,
:py:func:`~starfinder.sample_output.fused_grid` (a
:py:class:`~starfinder.sample_output.FusedGrid`), reduced by ``k``. Each fused array is
written by Starfinder as OME-TIFF (BigTIFF, 512 × 512 zlib tiles, band by band, with
calibration, plane positions and the ``ImageMetadata`` that
:py:func:`~starfinder.io.load_volume` restores); the input dtypes are uint8, uint16 and
float32, written in the input dtype. :py:class:`~starfinder.sample_output.FusionRecord`
holds the ``fusion`` section and the written paths. Importing the module imports neither
multiview-stitcher, zarr nor spatialdata; ``fuse`` needs the ``multiview-stitcher``
extra. :py:class:`~starfinder.sample_output.LabelConfig` and
:py:class:`~starfinder.sample_output.SpatialDataExportConfig` configure the label and
SpatialData parts.

.. currentmodule:: starfinder.sample_output

.. autosummary::
   :toctree: generated

   check_export
   ExportConfig
   ExportIncompleteWarning
   ExportRecord
   fuse
   fused_grid
   FusedGrid
   FusionConfig
   FusionInput
   FusionRecord
   LabelConfig
   SpatialDataExportConfig
