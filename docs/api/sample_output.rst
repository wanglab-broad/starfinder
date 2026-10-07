starfinder.sample_output
========================

Sample outputs (§2.10): the one sample-level export entry, its record, the fused
sample image and the assembled label image. See :doc:`../sample-output-contract`.

:py:meth:`~starfinder.dataset.Dataset.export` writes everything to the sample's export
folder ``<OUTPUT>/export/<sample>/``. After the checks before anything is written (the
configs, the placement's review, the ``anndata`` extra, the assembly, the fusion inputs,
the label IDs)
it removes the files the previous ``export.json`` lists for each part that runs or is off
and calls the parts in order, writing ``export.json`` after each: ``assemble`` (saved with
:py:func:`~starfinder.assembly.save_assembly`), :py:func:`~starfinder.sample_output.fuse`,
:py:func:`~starfinder.sample_output.fuse_labels` (off by default) and
``export_spatialdata``. :py:class:`~starfinder.sample_output.ExportConfig` turns each part
on or off and holds the configuration of each; the SpatialData part is not implemented yet
and is recorded ``off``. A part that cannot run (an extra not installed, a source image or
a territory file missing) is recorded ``not_run`` with its reason and
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
extra. :py:class:`~starfinder.sample_output.SpatialDataExportConfig` configures the
SpatialData part.

:py:func:`~starfinder.sample_output.fuse_labels` writes the assembled label image
``cell_labels.ome.tif`` (uint32), an inspection product drawn by retained cells (rule LR4):
each FOV's territory image, the one assign sampled, keeps only the cells the sample cell
table marks ``retained`` in that FOV, each mapped to its ``instance_id`` (``1..N`` in
cell table order) and kept whole; the library fuses these images in placement table
order at interpolation order 0 with a small first-positive function, so labels are never
blended and no value occurs that is in no input. The scale is
:py:class:`~starfinder.sample_output.LabelConfig` ``scale_yx = 1/k`` (0.25 by default),
requested with ``output_spacing``. The ``labels`` section records the territory files
read, the library's grid against the formula grid, the counts of ``ids_input`` and
``ids_output``, the ``absent_ids`` and the conflict count;
:py:class:`~starfinder.sample_output.LabelRecord` holds it and the written path. A sample
with more than 2^24 = 16,777,216 cell rows is refused before anything is written.

.. currentmodule:: starfinder.sample_output

.. autosummary::
   :toctree: generated

   check_export
   ExportConfig
   ExportIncompleteWarning
   ExportRecord
   fuse
   fuse_labels
   fused_grid
   FusedGrid
   FusionConfig
   FusionInput
   FusionRecord
   LabelConfig
   LabelRecord
   SpatialDataExportConfig
