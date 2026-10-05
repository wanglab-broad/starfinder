"""Segmentation (§2.9): the label contract, external-mask import and label functions.

Every segmentation result and every imported mask is a :class:`SegmentationResult`:
a ``uint32`` ZYX label image (a plane is 1×Y×X) on a :class:`ReferenceGrid`, with
its target (``nucleus`` or ``cell``), geometry (``volume``, ``plane`` or
``extended``), identity namespace and run record. Masks made elsewhere enter through
:func:`import_labels` (not a method). :func:`labels_to_grid` and
:func:`extend_labels_through_z` are plain label functions, each returning its result
and a record mapping. See docs/segmentation-contract.md and
docs/segmentation-algorithms.md.
"""
from ._errors import MissingModelError, ModelHashMismatchError, SegmentationBackendUnavailableError
from ._import import LabelImportConfig, import_labels
from ._labels import ReferenceGrid, SegmentationResult, reference_grid_from_file, to_label_dtype
from ._operations import ZExtensionConfig, extend_labels_through_z, labels_to_grid

__all__ = ["ReferenceGrid", "reference_grid_from_file", "SegmentationResult", "to_label_dtype",
           "LabelImportConfig", "import_labels", "labels_to_grid", "ZExtensionConfig", "extend_labels_through_z",
           "SegmentationBackendUnavailableError", "MissingModelError", "ModelHashMismatchError"]
