"""Versioned experiment records; JSON contains no implicit Python objects."""
from dataclasses import dataclass, field, asdict
from typing import Any
import json


def _json(value):
    return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))


@dataclass(frozen=True)
class BenchmarkCase:
    """Explicit inputs relative to input_root (registration NPY/TIFF; pipeline TIFF).

    Parameters
    ----------
    case_id : str
        Case identifier used in the run's output directory.
    task : str
        Benchmark task: ``registration`` or ``pipeline``.
    inputs : dict of str to str
        Named input-file paths relative to ``input_root``.
    config : dict of str to Any
        Task-specific processing and evaluation settings.
    truth : dict of str to str
        Named ground-truth file paths relative to ``input_root``; an empty mapping supplies no truth files.
    artifacts : dict of str to str
        Named auxiliary input-file paths relative to ``input_root``; an empty mapping supplies no auxiliary files.

    Notes
    -----
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
        """Return a strict JSON-compatible record.

        Returns
        -------
        dict
            Case fields converted to strict JSON-compatible values.
        """
        return _json(asdict(self))

    @classmethod
    def from_dict(cls, value):
        """Load a record, rejecting unknown fields.

        Parameters
        ----------
        value : dict
            JSON-compatible case fields to load.

        Returns
        -------
        BenchmarkCase
            Case constructed from the supplied fields.
        """
        return cls(**_json(value))


@dataclass
class BenchmarkTrialResult:
    """One processing trial, with distinct processing and evaluation states.

    Parameters
    ----------
    run_id : str
        Identifier of the benchmark run.
    case_id : str
        Identifier of the benchmark case.
    trial_id : str
        Identifier of the processing repetition within the case.
    requested_method : str
        Requested registration method name or ``pipeline`` for a pipeline case.
    actual_method : str or None
        Method used for processing; ``None`` when no method has been recorded.
    status : dict of str to str
        Processing and evaluation states.
    errors : dict of str to Any
        Error records for processing and evaluation.
    effective_configs : dict of str to Any
        Processing settings recorded for the trial.
    resources : dict of str to Any
        Resource records with value, unit, source and scope.
    artifacts : dict of str to Any
        Named saved-artifact references with paths relative to the run.
    attempts : list of dict of str to Any
        Processing attempt records; defaults to an empty list.
    metrics : dict of str to Any
        Evaluation metric results; defaults to an empty mapping.
    provenance : dict of str to Any
        Source and evaluation provenance; defaults to an empty mapping.

    Notes
    -----
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
        """Serialize statuses/configs/metadata without lossy string coercion.

        Returns
        -------
        dict
            Trial fields converted to strict JSON-compatible values.
        """
        return _json(asdict(self))

    @classmethod
    def from_dict(cls, value):
        """Reload the complete saved trial record.

        Parameters
        ----------
        value : dict
            JSON-compatible trial fields to load.

        Returns
        -------
        BenchmarkTrialResult
            Trial record constructed from the supplied fields.
        """
        return cls(**_json(value))
