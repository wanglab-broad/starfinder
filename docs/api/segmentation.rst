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
read from (its C-order bytes alone, without a dtype or shape prefix).
``FOV.reference_grid()`` returns the grid of the resident reference round after
``FOV.run`` or after ``FOV.load_checkpoint("registered")``, or, without that round, the
grid of the saved reference image after ``FOV.load_reference_image()``;
:py:func:`~starfinder.segmentation.reference_grid_from_file` reads it from a TIFF such as
``images/ref_merged/{fovID}.tif``; a caller without a molecule run declares one with
``source="declared"``. ``ReferenceGrid.projected()`` is the Z=1 grid of a projection.
Two grids are one grid when their shapes and metadata are equal; the source and hash are
recorded, not compared (:doc:`../assignment-contract`, "The grid rule").

:py:func:`~starfinder.segmentation.import_labels` brings a mask made elsewhere
(CellProfiler, an earlier workflow run) onto a grid: the byte order is converted to
native, a YX file becomes 1×Y×X, a float or boolean mask raises ``TypeError``, another
shape raises :py:class:`~starfinder.image.IncompatibleGeometryError`, stored metadata
must equal the grid's, and ``relabel=True`` maps the values to 1…n. Its record has no
method entry and an ``import`` entry with the file's and the array's SHA-256.
:py:class:`~starfinder.segmentation.LabelImportConfig` names an import as a run of a
plan; import is not a segmentation method.

The input functions prepare a segmentation input from images and leave their inputs
unchanged; each returns its result and a record mapping (``function``, ``config``,
``inputs`` and ``output`` as array SHA-256 values, and the values it reached).
:py:func:`~starfinder.segmentation.composite_nuclei_amplicon` (with
:py:class:`~starfinder.segmentation.CompositeConfig`) and
:py:func:`~starfinder.segmentation.enhance_with_flamingo` (with
:py:class:`~starfinder.segmentation.FlamingoEnhancementConfig`) are the DAPI–amplicon
composite and the Flamingo-assisted DAPI enhancement of the workflow scripts, bit for
bit, on two ZYX arrays of one shape (a plane is 1×Y×X; other shapes raise
:py:class:`~starfinder.image.IncompatibleGeometryError`), with a ``uint8`` output; the
script's maximum projection is a run's projection, not part of the function.
:py:func:`~starfinder.segmentation.normalize_percentiles` is csbdeep's percentile
normalization (float32, unclipped; a constant image gives zeros), and
:py:func:`~starfinder.segmentation.rescale_input` resamples an image by per-axis factors
and returns metadata whose spacing is divided by the factors and whose ``frame_id``
records the rescale. None of them is a preprocessing method: they combine two images,
change the dtype or change the grid.

:py:func:`~starfinder.segmentation.expand_labels` grows every label into the background
by a distance (:py:class:`~starfinder.segmentation.ExpandLabelsConfig`: ``distance``,
``unit`` ``pixel`` or ``um``, ``mode`` ``planar``, each Z plane as the workflow scripts do,
or ``volumetric``), never overwriting or renumbering a label; ``um`` needs the metadata's
spacing. It is the one label expansion: the assignment entry applies it to the cell
territories and keeps both masks (:doc:`assignment`).
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

:py:func:`~starfinder.segmentation.segment` runs one segmentation method on a
:py:class:`~starfinder.segmentation.SegmentationInput` (a ZYXC image on its
:py:class:`~starfinder.segmentation.ReferenceGrid`, with one role per channel:
``nuclear``, ``cytoplasm``, ``membrane``, ``amplicon`` or ``composite``) and returns a
:py:class:`~starfinder.segmentation.SegmentationResult` on the input's grid. It reads and
writes no file. :py:data:`~starfinder.segmentation.SEGMENTATION_METHODS` maps each exact
config type to its :py:class:`~starfinder.segmentation.SegmentationSpec`: the method name
(the config's ``method`` value), its targets, accepted and required roles, seeds,
dimensions, whether it needs a model, its devices and optional dependencies; see
:doc:`../method-registry`. Before the method's ``run`` is called with a
:py:class:`~starfinder.segmentation.MethodContext`, ``segment`` checks, in order, the
config (a :py:class:`~starfinder.segmentation.LabelImportConfig` raises ``TypeError``),
the target, the input and its roles, the device (``"cpu"`` or ``"cuda"``, among the
method's), the dependencies
(:py:class:`~starfinder.segmentation.SegmentationBackendUnavailableError`), the model
files and their SHA-256, the dimensionality and minimum shape, and the seeds (a nucleus
result on the input's grid). Afterwards it checks the labels: an integer array of the
input's ZYX shape (a Z axis dropped for one plane is restored), with no negative value,
converted to ``uint32``. The run record holds the method's provenance entry with its
model artifacts, its execution entry and its effective parameters, the input's SHA-256
and channel sources, and the seed run with its labels' SHA-256. A result without objects
has outcome ``empty``; there is no foreground gate.

The registry holds ``stardist``, ``cellpose`` and ``seeded_watershed``. ``seeded_watershed``
(:py:class:`~starfinder.segmentation.SeededWatershedConfig`): cells grown from the nuclei
of a seed run on one stain channel, smoothed by a Gaussian of ``sigma_um`` converted with
the input's spacing (required), thresholded (Otsu by default), with the mask united with
every seed voxel. Every seed keeps its value, so cell k contains nucleus k, also for a
nucleus outside the stained foreground. It runs on the CPU.

``stardist`` (:py:class:`~starfinder.segmentation.StarDistConfig`, the ``stardist``
extra) segments one ``nuclear`` or ``composite`` channel with a StarDist model: a 2D model
on a plane (Z=1), a 3D model on a volume, decided by the model's ``n_dim``; the other
combination is rejected before the model is built. ``scale`` is required (a number scales
Y and X, a 3-tuple is per axis ZYX), the stored ``thresholds.json`` is used unless a
threshold is given as an override, and the input is normalized by csbdeep's percentiles
(1, 99.8). With ``block_size``, ``min_overlap`` and ``context`` all given (never some of
them, never with a scale other than 1, and with ``min_overlap + 2 × context <
block_size`` on every axis) the method predicts block by block with
``predict_instances_big``; the record keeps the requested values and the effective
values, which the library rounds up to multiples of the model's grid. Objects must be
smaller than ``min_overlap``; there is no default block size. ``cellpose``
(:py:class:`~starfinder.segmentation.CellposeConfig`, the ``cellpose`` extra) segments a
``cytoplasm`` or ``nuclear`` channel, optionally with ``nuclear`` beside ``cytoplasm``;
``diameter`` is required, 3D mode (``do_3d`` with ``anisotropy``) runs on a volume, and the
library defaults it keeps (``tile_overlap``, ``bfloat16``, the thresholds) are recorded. Both
run on ``"cpu"`` or ``"cuda"``; the execution entry records the framework, its CUDA
build and thread counts, and on ``"cuda"`` the GPU and TensorFlow's memory growth. A
library result in int32, uint16 or uint32 becomes ``uint32``; a Z axis Cellpose drops for
one plane is restored.

No segmentation entry downloads a model.
:py:data:`~starfinder.segmentation.KNOWN_MODELS` maps ``(method, model)`` to a
:py:class:`~starfinder.segmentation.KnownModel` (its download, the
:py:class:`~starfinder.segmentation.ModelFile` entries the library loads with their
SHA-256, and the documented properties) for the pretrained models Starfinder can verify,
``("stardist", "2D_versatile_fluo")`` and ``("cellpose", "cpsam_v2")``; it sets no default
model. :py:func:`~starfinder.segmentation.resolve_model` finds a known model in the
weights cache (``<root>/<method>/<model>/``, the §2.7 cache) or a user-trained model by
path (a StarDist folder with ``config.json``, ``thresholds.json`` and ``weights_best.h5``,
such as ``3D_spleen``; a Cellpose model file), recomputes each file's SHA-256 and returns
the path with the provenance artifacts; a missing file raises
:py:class:`~starfinder.segmentation.MissingModelError` naming the fetch command, a changed
one :py:class:`~starfinder.segmentation.ModelHashMismatchError` naming both hashes.
``segment`` calls it before any library call. ``starfinder weights fetch stardist
2D_versatile_fluo`` (or ``cellpose cpsam_v2``) is the explicit download, and ``starfinder
weights list`` and ``verify`` cover the segmentation models with the detector weights.

``FOV.segment(plan, device="cpu")`` runs a
:py:class:`~starfinder.segmentation.SegmentationPlan` of
:py:class:`~starfinder.segmentation.SegmentationRun` entries in order on the FOV's
resident images and stores each result in ``FOV.segmentation_results`` under the run's
name, after every run has finished. A run imports a mask or names a method, its target,
its input channels (:py:class:`~starfinder.segmentation.InputChannel`: a channel of the
reference round, of a sequencing round registered by ``FOV.run`` or of a morphology round
prepared by ``FOV.prepare_morphology`` (or ``FOV.register_rounds``) or reloaded by
``FOV.load_registered_round``, or the reference round's channel maximum (the saved
reference image of ``FOV.load_reference_image`` when the round is not resident), each
optionally through the composite or the Flamingo enhancement), an earlier run as its
seeds, an optional Z projection, and label operations in order: ``expand_labels`` and
``extend_labels_through_z`` after a projection (an import takes none). Every
run is on ``FOV.reference_grid()``, or on its projection. An input round that is not
loaded, a morphology round without an entry in ``registration_record["rounds"]`` or with
other metadata raises ``ValueError``; a round of another shape, or seeds on another grid,
raises :py:class:`~starfinder.image.IncompatibleGeometryError`. With
``checkpoints=CheckpointConfig(…)``, each run is also written to
``<checkpoint dir>/<fov_id>/segmentation/<run>/`` (``labels.tif``, ``input.ome.tif`` for a
computed run, ``segmentation.json``), and ``FOV.load_segmentation(name)`` reads it back,
checking the recorded SHA-256 values (see :doc:`../checkpoints`).

.. currentmodule:: starfinder.segmentation

.. autosummary::
   :toctree: generated

   CellposeConfig
   composite_nuclei_amplicon
   CompositeConfig
   enhance_with_flamingo
   expand_labels
   ExpandLabelsConfig
   extend_labels_through_z
   FlamingoEnhancementConfig
   import_labels
   InputChannel
   KNOWN_MODELS
   KnownModel
   LabelImportConfig
   labels_to_grid
   MethodContext
   MissingModelError
   ModelFile
   ModelHashMismatchError
   normalize_percentiles
   reference_grid_from_file
   ReferenceGrid
   rescale_input
   resolve_model
   SeededWatershedConfig
   segment
   SEGMENTATION_METHODS
   SegmentationBackendUnavailableError
   SegmentationInput
   SegmentationPlan
   SegmentationResult
   SegmentationRun
   SegmentationSpec
   StarDistConfig
   to_label_dtype
   ZExtensionConfig
