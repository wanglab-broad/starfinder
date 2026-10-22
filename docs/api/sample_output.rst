starfinder.sample_output
========================

Sample outputs (§2.10): the one sample-level export entry, its record, the fused
sample image, the assembled label image and the SpatialData export. See
:doc:`../sample-output-contract`.

:py:meth:`~starfinder.dataset.Dataset.export` writes everything to the sample's export
folder ``<OUTPUT>/export/<sample>/``. After the checks before anything is written (the
configs, the placement's review, the ``anndata`` extra, the assembly, the fusion inputs,
the label IDs)
it removes the files the previous ``export.json`` lists for each part that runs or is off
and calls the parts in order, writing ``export.json`` after each: ``assemble`` (saved with
:py:func:`~starfinder.assembly.save_assembly`), :py:func:`~starfinder.sample_output.fuse`,
:py:func:`~starfinder.sample_output.fuse_labels` (off by default) and
:py:func:`~starfinder.sample_output.export_spatialdata` (off by default).
:py:class:`~starfinder.sample_output.ExportConfig` turns each part on or off and holds the
configuration of each. A part that cannot run (an extra not installed, a source image or
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
:py:func:`~starfinder.io.load_volume` restores, and an OME UUID derived from the pixel data
and the metadata written, so the same content gives the same bytes); the input dtypes are
uint8, uint16 and float32, written in the input dtype. :py:class:`~starfinder.sample_output.FusionRecord`
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

:py:func:`~starfinder.sample_output.export_spatialdata` writes the SpatialData store
``sample.zarr/``, an export of the sample tables and the fused and label images, not a
second source: nothing is filtered, reassigned or recounted. One coordinate system, named
after the sample (whose name must satisfy spatialdata's element naming rule), in the
sample frame's axes and µm. The points ``molecules`` hold every molecule with finite
sample coordinates (the §2.9 ``z, y, x`` renamed ``fov_index_z/y/x``, ``sample_x/y/z``
the coordinate columns ``x, y, z``, ``cell_instance_id`` added); the shapes
``cell_centroids`` one circle per cell with a finite ownership centroid, at that centroid,
with the equivalent radius of the original territory (a sphere for ``volume``, a disc for
``plane``; a cell of zero voxels is refused), indexed by ``instance_id`` (``1..N`` over the
sample cell table); the table ``cells`` those cell rows and ``expression``
(``sample.h5ad`` with every part kept) annotate ``cell_centroids`` through ``region`` and
``instance_id``; the tables ``cells_unplaced`` and ``molecules_unplaced`` hold the rows
without sample coordinates. Each fused image (``<image>_projection``, ``<image>_stack``)
and the label image (``cell_labels``) is one element, its OME-TIFF's output level read
lazily and written once, with one level, transform ``Scale(S)`` then
``Translation(O − S/2)``; no standalone OME-Zarr is written beside the store. An element
whose population has no row, or whose image does not exist, is left out and recorded
``omitted`` with its reason. :py:class:`~starfinder.sample_output.SpatialDataExportConfig`
selects Zarr format 3 (spatialdata's default format, the default) or 2. The store is
written into a staging folder and renamed into place; the ``spatialdata`` section of
``export.json`` records every element, the formats, the counts and the SHA-256 of every
store file. A table that already has ``uns["spatialdata_attrs"]`` is refused by
spatialdata and no store is written. Without the ``spatialdata`` extra the part is
recorded ``not_run``. :py:func:`~starfinder.sample_output.lookup_source` traces a molecule
or a cell of an export folder back to its FOV, its §2.9 assignment folder and the row's
position in the §2.9 table, after verifying the record's hashes.
:py:func:`~starfinder.sample_output.check_export` reads the store back and evaluates the
SpatialData conditions S1 to S8.

The route to a sample's cells is ``ExportConfig.route``, a value of
:py:data:`~starfinder.sample_output.ROUTES` (``per_fov``: segmentation per FOV followed by
assembly; ``after_fusion``: segmentation of the fused projection) or None for the default by
``dimensionality`` (``projection`` gives ``after_fusion``, ``stack`` and ``both`` give
``per_fov``). Check 7 of :py:meth:`~starfinder.dataset.Dataset.export` refuses before
anything is written the route after fusion with ``dimensionality: stack`` or
``projection: sum``, and, with an assignment, without ``ExportConfig.segmentation`` (a
:py:class:`~starfinder.sample_output.SampleSegmentationConfig` of
:py:class:`~starfinder.sample_output.SampleSegmentationRun`), with an image that is not in
``images``, with ``fuse`` or ``assemble`` off while ``segment`` is on, or with a model that
is missing or whose SHA-256 differs. On the route per FOV the part ``segment`` is recorded
``off`` (reason ``route_per_fov``; ``no_cells`` without an assignment). On the route after
fusion with an assignment the parts run in the order ``fuse``, ``segment``, ``assemble``
(:py:func:`~starfinder.assembly.assemble` with ``segmentation``: the sample's molecules
assigned to the sample segmentation), ``fuse_labels`` and ``export_spatialdata``.
:py:func:`~starfinder.sample_output.segment_fused`, the part ``segment``, also callable alone
on a folder whose ``fuse`` ran, segments the fused projection of each input image on the
segmentation grid (the fused file's grid, frame ``sample:<sample>``), with the method's pixel
lengths given for the acquired grid and converted to the segmentation grid by the table
"Lengths in pixels" of :doc:`../sample-segmentation-contract` (both values recorded),
and writes one ``segmentation_<run>.ome.tif`` per run (uint32) and the ``segmentation``
section of ``export.json``, whose inputs are the fused files' SHA-256 and the grid;
:py:func:`~starfinder.sample_output.load_sample_segmentation` reads them back as a
:py:class:`~starfinder.segmentation.SampleSegmentation`, each label file checked against its
SHA-256. On the route after fusion ``fuse_labels`` passes the sample's territories as its one
view (rule ``segmentation``), ``lookup_source`` returns for a cell its row in the sample cell
table, ``label_file`` and ``label_value`` and no assignment folder, the circles of the
SpatialData store take their radii from the segmentation grid's spacing, and
``check_export`` adds the conditions T1 to T5 and checks the label image against the sample's
territories. :py:func:`~starfinder.sample_output.compare_routes` reads an export folder of each
route for the same sample and reports, as a
:py:class:`~starfinder.sample_output.RouteComparison`, how their cells and molecules
correspond: both routes' counted cells on the segmentation grid, the share of every pair by
the measure of the duplicate rule, the matched pairs, the counts per matched pair and one of
six categories per molecule. It decides nothing, changes no file of either folder and writes
its tables only into a folder ``output`` outside both. See
:doc:`../sample-segmentation-contract`.

.. currentmodule:: starfinder.sample_output

.. autosummary::
   :toctree: generated

   check_export
   compare_routes
   export_spatialdata
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
   load_sample_segmentation
   lookup_source
   RouteComparison
   ROUTES
   SampleSegmentationConfig
   SampleSegmentationRun
   segment_fused
   SpatialDataExportConfig
