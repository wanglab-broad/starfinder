"""Stitching errors (docs/stitching-contract.md, "Names")."""


class StitchingBackendUnavailableError(ImportError):
    """An optional dependency of a stitching method (its extra) is not installed."""


class PlacementReviewRequiredError(ValueError):
    """A placement whose review is required holds no approval of its current tables."""
