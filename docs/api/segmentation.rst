starfinder.segmentation
=======================

Segmentation (§2.9) gives every label image one contract, whether a method computed it
or a mask was imported; see :doc:`../segmentation-contract` and
:doc:`../segmentation-algorithms`.

A :py:class:`~starfinder.segmentation.SegmentationResult` holds a ``uint32`` ZYX label
image (a plane is 1×Y×X; 0 is background and each positive value one object, never
relabelled), the :py:class:`~starfinder.segmentation.ReferenceGrid` it lies on, its
target (``nucleus`` or ``cell``), its geometry (``volume``, ``plane`` or ``extended``),
its ``label_namespace`` and its run record. Construction checks the dtype, the shape
against ``grid.shape_zyx``, the target and that the geometry agrees with Z.
:py:func:`~starfinder.segmentation.to_label_dtype` is the label dtype rule: it converts
int32, uint16 or any other integer labels to ``uint32`` and raises ``ValueError`` on a
negative value or one above 2³²−1 instead of wrapping, and ``TypeError`` on a float or
boolean array.

A :py:class:`~starfinder.segmentation.ReferenceGrid` is a shape, an
:py:class:`~starfinder.image.ImageMetadata`, a source and the SHA-256 of the image it was
read from (its dtype, shape and C-order bytes). ``FOV.reference_grid()`` returns the
grid of the resident reference round after ``FOV.run`` or after
``FOV.load_checkpoint("registered")``;
:py:func:`~starfinder.segmentation.reference_grid_from_file` reads it from a TIFF such as
``images/ref_merged/{fovID}.tif``; a caller without a molecule run declares one with
``source="declared"``. ``ReferenceGrid.projected()`` is the Z=1 grid of a projection.

:py:func:`~starfinder.segmentation.import_labels` brings a mask made elsewhere
(CellProfiler, an earlier workflow run) onto a grid: the byte order is converted to
native, a YX file becomes 1×Y×X, a float or boolean mask raises ``TypeError``, another
shape raises :py:class:`~starfinder.image.IncompatibleGeometryError`, stored metadata
must equal the grid's, and ``relabel=True`` maps the values to 1…n. Its record has no
method entry and an ``import`` entry with the file's and the array's SHA-256.
:py:class:`~starfinder.segmentation.LabelImportConfig` names an import as a run of a
plan; import is not a segmentation method.

:py:func:`~starfinder.segmentation.labels_to_grid` (nearest neighbour at pixel centres
onto exactly the target shape) and
:py:func:`~starfinder.segmentation.extend_labels_through_z` (plane labels kept where a
stain is foreground, with the sizes of
:py:class:`~starfinder.segmentation.ZExtensionConfig` in µm and its threshold on the
[0, 1] scale of the stain's dtype range) are label functions, not methods; each returns
its labels and a record mapping. The model and backend errors
:py:class:`~starfinder.segmentation.SegmentationBackendUnavailableError`,
:py:class:`~starfinder.segmentation.MissingModelError` and
:py:class:`~starfinder.segmentation.ModelHashMismatchError` are those the contract names
for the segmentation methods.

.. currentmodule:: starfinder.segmentation

.. autosummary::
   :toctree: generated

   extend_labels_through_z
   import_labels
   LabelImportConfig
   labels_to_grid
   MissingModelError
   ModelHashMismatchError
   reference_grid_from_file
   ReferenceGrid
   SegmentationBackendUnavailableError
   SegmentationResult
   to_label_dtype
   ZExtensionConfig
