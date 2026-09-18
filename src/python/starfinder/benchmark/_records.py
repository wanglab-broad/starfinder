"""Versioned experiment records; JSON contains no implicit Python objects."""
from dataclasses import dataclass, field, asdict
from typing import Any
import json


def _json(value):
    return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))


@dataclass(frozen=True)
class BenchmarkCase:
    """Explicit inputs relative to input_root (registration NPY/TIFF; pipeline TIFF).

    config requires registration, reference_metadata, moving_metadata and
    evaluation entries. truth and artifacts are named file references, never
    inferred from directory names. Pipeline cases supply workflow, sources, fov_id and count evaluation;
    see the maintained benchmark recipes.
    """
    case_id: str
    task: str
    inputs: dict[str, str]
    config: dict[str, Any]
    truth: dict[str, str] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)

    def to_dict(self):
        """Return a strict JSON-compatible record."""
        return _json(asdict(self))

    @classmethod
    def from_dict(cls, value):
        """Load a record, rejecting unknown fields."""
        return cls(**_json(value))


@dataclass
class BenchmarkTrialResult:
    """One processing trial, with distinct processing and evaluation states.

    Resources carry value/unit/source/scope. Artifact paths are relative to the
    run; metric results retain undefined values as null with reasons.
    """
    run_id: str
    case_id: str
    trial_id: str
    requested_method: str
    actual_method: str | None
    status: dict[str, str]
    errors: dict[str, Any]
    effective_configs: dict[str, Any]
    resources: dict[str, Any]
    artifacts: dict[str, Any]
    attempts: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        """Serialize statuses/configs/metadata without lossy string coercion."""
        return _json(asdict(self))

    @classmethod
    def from_dict(cls, value):
        """Reload the complete saved trial record."""
        return cls(**_json(value))
