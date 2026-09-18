"""Shared, serializable metric results and coordinate validation."""
from dataclasses import dataclass, field
from typing import Any
import numpy as np
from starfinder.image import ImageMetadata, IncompatibleGeometryError


@dataclass(frozen=True)
class EvaluationResult:
    """Metric values (None if undefined), units, populations and provenance.

    Status is ok, undefined (at least one unavailable metric), missing (an
    input is absent), or failed (an explicitly supplied upstream failure).
    Details contain per-item comparisons/pairs; config records effective policy.
    """
    values: dict[str, float | bool | None]
    units: dict[str, str]
    counts: dict[str, int]
    status: str
    reasons: dict[str, str]
    config: dict[str, Any]
    details: dict[str, Any] = field(default_factory=dict)


def _result(values, units, counts, config, *, reasons=None, details=None, status=None):
    reasons = dict(reasons or {})
    for key, value in values.items():
        if value is None:
            reasons.setdefault(key, "zero denominator or no eligible observations")
        elif not np.isfinite(value):
            raise ValueError(f"nonfinite metric {key}")
    return EvaluationResult(values, units, counts,
                            status or ("undefined" if reasons else "ok"),
                            reasons, config, details or {})


def _geometry(reference: ImageMetadata, observed: ImageMetadata, units: str):
    if reference != observed:
        raise IncompatibleGeometryError("evaluation requires the same frame and geometry; convert explicitly first")
    if units != "voxel":
        reference._require_physical()
        if units != reference.spatial_unit:
            raise IncompatibleGeometryError("coordinate units do not match metadata")
    if not isinstance(units, str) or not units:
        raise ValueError("units must be explicit")


def _threshold(value):
    if not np.isfinite(value) or value < 0:
        raise ValueError("threshold must be finite and nonnegative")


def _eligible(mask, n):
    if mask is None:
        return np.ones(n, dtype=bool)
    mask = np.asarray(mask)
    if mask.dtype != bool or mask.shape != (n,):
        raise ValueError("eligibility must be a Boolean mask of population length")
    return mask
