starfinder.segmentation
=======================

Label images from segmentation methods or imported masks. The
:doc:`segmentation guide <../segmentation-contract>` shows the workflow; the
:doc:`segmentation reference <../segmentation-algorithms>` lists methods, options and files.

.. currentmodule:: starfinder.segmentation

Grids, inputs and results
-------------------------

.. autosummary::
   :toctree: generated

   reference_grid_from_file
   ReferenceGrid
   SampleSegmentation
   SegmentationInput
   SegmentationResult

Plans and mask import
---------------------

.. autosummary::
   :toctree: generated

   import_labels
   InputChannel
   LabelImportConfig
   SegmentationPlan
   SegmentationRun

Methods and registry
--------------------

.. autosummary::
   :toctree: generated

   CellposeConfig
   MethodContext
   SeededWatershedConfig
   segment
   SEGMENTATION_METHODS
   SegmentationSpec
   StarDistConfig

Input preparation
-----------------

.. autosummary::
   :toctree: generated

   composite_nuclei_amplicon
   CompositeConfig
   enhance_with_flamingo
   FlamingoEnhancementConfig
   normalize_percentile_unclipped
   rescale_input

Label operations
----------------

.. autosummary::
   :toctree: generated

   expand_labels
   ExpandLabelsConfig
   extend_labels_through_z
   resample_labels
   to_label_dtype
   ZExtensionConfig

Models and errors
-----------------

.. autosummary::
   :toctree: generated

   KNOWN_MODELS
   KnownModel
   MissingModelError
   ModelFile
   ModelHashMismatchError
   resolve_model
   SegmentationBackendUnavailableError
