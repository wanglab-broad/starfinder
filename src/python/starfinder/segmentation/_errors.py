"""Segmentation errors (docs/segmentation-contract.md, "Names")."""


class SegmentationBackendUnavailableError(ImportError):
    """An optional dependency of a segmentation method (its extra) is not installed."""


class MissingModelError(FileNotFoundError):
    """A model folder or a file it needs is missing; segmentation never downloads, so fetch it explicitly."""


class ModelHashMismatchError(ValueError):
    """A model file's SHA-256 differs from the expected value."""
