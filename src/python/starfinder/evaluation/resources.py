"""Pure resource-prediction error on explicitly held-out dataset records."""
import numpy as np
import pandas as pd

from ._result import EvaluationResult, _result

__all__ = ["resource_prediction_errors"]


def resource_prediction_errors(records: pd.DataFrame, *, training_dataset_ids,
                               target: str, units: str) -> EvaluationResult:
    """Summarize resource-prediction errors without fitting a model (M79).

    Parameters
    ----------
    records : pandas.DataFrame
        Held-out records with nonempty string ``dataset_id`` and finite,
        nonnegative ``measured`` and ``predicted`` columns in the supplied unit.
        Supply wall-time and peak-RSS predictions in separate calls.
    training_dataset_ids : iterable of str
        Complete training dataset IDs, disjoint from every supplied record's
        dataset ID, including records with zero measured resource.
    target : str
        Resource target: ``wall_time`` or ``peak_rss``.
    units : str
        ``second`` for wall_time; ``byte``, ``KiB``, ``MiB`` or ``GiB`` for peak_rss.
        Measured and predicted values must already share this unit.

    Returns
    -------
    EvaluationResult
        Median and maximum absolute relative error, underprediction count over
        positive measured resources, row errors and excluded zero-resource
        records. With no positive measured resources, errors are undefined.
        Dataset separation and the resource unit are retained as provenance.
    """
    valid_units = {"wall_time": ("second",), "peak_rss": ("byte", "KiB", "MiB", "GiB")}
    if target not in valid_units or units not in valid_units[target]:
        raise ValueError("target and units must name a supported resource and its unit")
    if not {"dataset_id", "measured", "predicted"} <= set(records.columns):
        raise ValueError("records require dataset_id, measured and predicted")
    if isinstance(training_dataset_ids, str):
        raise ValueError("training_dataset_ids must be an iterable of dataset IDs")
    training = list(training_dataset_ids)
    evaluation = records.dataset_id.tolist()
    if any(not isinstance(v, str) or not v for v in training + evaluation):
        raise ValueError("dataset IDs must be nonempty strings")
    if set(training) & set(evaluation):
        raise ValueError("evaluation dataset IDs must be held-out from training")
    measured = records.measured.to_numpy(dtype=float)
    predicted = records.predicted.to_numpy(dtype=float)
    if any(not np.isfinite(v).all() or (v < 0).any() for v in (measured, predicted)):
        raise ValueError("resources must be finite and nonnegative")
    defined = measured > 0
    errors = np.abs(predicted[defined] - measured[defined]) / measured[defined]
    values = {"median_absolute_relative_error": float(np.median(errors)) if len(errors) else None,
              "max_absolute_relative_error": float(np.max(errors)) if len(errors) else None}
    return _result(values, {k: "fraction" for k in values},
                   {"total": len(records), "defined": int(defined.sum()), "measured_zero": int((~defined).sum()),
                    "underpredictions": int((predicted[defined] < measured[defined]).sum())},
                   {"target": target, "resource_units": units,
                    "training_dataset_ids": sorted(set(training)), "evaluation_dataset_ids": sorted(set(evaluation)),
                    "denominator": "positive measured resource"},
                   details={"absolute_relative_errors": errors.tolist(),
                            "defined_positions": np.flatnonzero(defined).tolist(),
                            "excluded_records": [{"position": int(i), "dataset_id": evaluation[i],
                                                  "reason": "zero measured resource"}
                                                 for i in np.flatnonzero(~defined)]})
