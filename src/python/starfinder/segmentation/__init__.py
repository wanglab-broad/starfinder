"""Segmentation (§2.9): the segment entry and its registry, the label contract, import, input and label functions.

Every segmentation result and every imported mask is a :class:`SegmentationResult`:
a ``uint32`` ZYX label image (a plane is 1×Y×X) on a :class:`ReferenceGrid`, with
its target (``nucleus`` or ``cell``), geometry (``volume``, ``plane`` or
``extended``), identity namespace and run record. Masks made elsewhere enter through
:func:`import_labels` (not a method). :func:`composite_nuclei_amplicon`,
:func:`enhance_with_flamingo`, :func:`normalize_percentiles` and
:func:`rescale_input` are plain input functions, :func:`labels_to_grid` and
:func:`extend_labels_through_z` plain label functions; each returns its result and a
record mapping and leaves its inputs unchanged. :func:`expand_labels` is the one
label expansion; the assignment entry (``starfinder.assignment``) applies it to the
cell territories and keeps both masks. :func:`segment` runs one method of
``SEGMENTATION_METHODS`` on a :class:`SegmentationInput` behind the stage checks;
``FOV.segment`` runs a :class:`SegmentationPlan` of named runs on one FOV. See
docs/segmentation-contract.md and docs/segmentation-algorithms.md.
"""
from ._errors import MissingModelError, ModelHashMismatchError, SegmentationBackendUnavailableError
from ._import import LabelImportConfig, import_labels
from ._inputs import (CompositeConfig, FlamingoEnhancementConfig, composite_nuclei_amplicon, enhance_with_flamingo,
                      normalize_percentiles, rescale_input)
from ._labels import ReferenceGrid, SegmentationResult, reference_grid_from_file, to_label_dtype
from ._methods import SEGMENTATION_METHODS as _SEGMENTATION_METHODS
from ._methods import MethodContext, SeededWatershedConfig, SegmentationSpec
from ._operations import ExpandLabelsConfig, ZExtensionConfig, expand_labels, extend_labels_through_z, labels_to_grid
from ._plan import InputChannel, SegmentationPlan, SegmentationRun
from ._segment import SegmentationInput, segment

#: The segmentation method registry, mapping each exact frozen config type to its
#: SegmentationSpec. segment, SegmentationRun and FOV.segment derive their method sets from it.
SEGMENTATION_METHODS = _SEGMENTATION_METHODS

__all__ = ["ReferenceGrid", "reference_grid_from_file", "SegmentationResult", "to_label_dtype",
           "LabelImportConfig", "import_labels", "labels_to_grid", "ExpandLabelsConfig", "expand_labels",
           "ZExtensionConfig", "extend_labels_through_z",
           "CompositeConfig", "composite_nuclei_amplicon", "FlamingoEnhancementConfig", "enhance_with_flamingo",
           "normalize_percentiles", "rescale_input",
           "segment", "SegmentationInput", "SEGMENTATION_METHODS", "SegmentationSpec", "MethodContext",
           "SeededWatershedConfig", "SegmentationPlan", "SegmentationRun", "InputChannel",
           "SegmentationBackendUnavailableError", "MissingModelError", "ModelHashMismatchError"]
