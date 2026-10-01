"""Detection evaluation against supplied, explicitly eligible truth."""
import numpy as np
from scipy.spatial.distance import cdist

from ._result import _result
from .matching import match_points

__all__ = ["evaluate_spots", "localization_errors", "classify_detections"]


def evaluate_spots(detected, truth, **matching_config):
    """Evaluate supplied (N,3) detections against truth; see match_points.

    All matching policy, threshold, units and geometry arguments are required
    by match_points. Reference means truth, observed means detected.
    """
    return match_points(truth, detected, **matching_config)


def _points(values, name):
    points = np.asarray(values, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError(f"{name} must be finite (N,3) ZYX points")
    return points


def _pairs(match_result, detected, truth):
    """Matched (truth index, detected index, distance) arrays of an evaluate_spots result."""
    pairs = match_result.details["matched_pairs"]
    counts = match_result.counts
    if counts.get("total_reference") != len(truth) or counts.get("total_observed") != len(detected):
        raise ValueError("detected and truth must be the populations the match result was computed from")
    i = np.array([p[0] for p in pairs], dtype=int)
    j = np.array([p[1] for p in pairs], dtype=int)
    return i, j, np.array([p[2] for p in pairs], dtype=float)


def localization_errors(match_result, detected, truth):
    """Per-axis localization errors of the matched pairs of an evaluate_spots result.

    match_result comes from evaluate_spots(detected, truth, ...) on the same
    (N,3) ZYX points. Over the matched pairs, values hold the maximum and the
    95th percentile (numpy.quantile, linear) of the absolute Z, Y and X
    errors ``|detected - truth|`` (``abs_z_max``, ``abs_z_p95``, ...), of the
    lateral distance hypot(Y, X) (``lateral_max``, ``lateral_p95``) and of
    the 3D matched distance (``dist_max``, ``dist_p95``), in the matching
    units; they are None without matches. counts hold ``matched`` and
    ``n_abs_z_gt_1``, the matches whose absolute Z error is above 1.
    """
    detected, truth = _points(detected, "detected"), _points(truth, "truth")
    i, j, distance = _pairs(match_result, detected, truth)
    names = [f"abs_{axis}_{stat}" for axis in "zyx" for stat in ("max", "p95")]
    names += ["lateral_max", "lateral_p95", "dist_max", "dist_p95"]
    values = dict.fromkeys(names)
    error = np.abs(detected[j] - truth[i]) if len(i) else np.empty((0, 3))
    if len(i):
        for k, axis in enumerate("zyx"):
            values[f"abs_{axis}_max"] = float(error[:, k].max())
            values[f"abs_{axis}_p95"] = float(np.quantile(error[:, k], 0.95))
        lateral = np.hypot(error[:, 1], error[:, 2])
        values.update(lateral_max=float(lateral.max()), lateral_p95=float(np.quantile(lateral, 0.95)),
                      dist_max=float(distance.max()), dist_p95=float(np.quantile(distance, 0.95)))
    units = match_result.config["units"]
    return _result(values, dict.fromkeys(names, units),
                   {"matched": int(len(i)), "n_abs_z_gt_1": int((error[:, 0] > 1).sum())},
                   {**match_result.config, "statistics": "maximum and numpy.quantile(0.95), linear"})


def classify_detections(match_result, detected, truth, *, radius, groups=None, boundary="exclusive"):
    """Count matched, duplicate and spurious detections of an evaluate_spots result.

    A detection in a matched pair is matched. An unmatched detection is a
    duplicate when it lies within radius of a matched truth point (strictly
    for boundary "exclusive", the W-218 classification), and spurious
    otherwise. With groups (one label per detection, for example its
    channel), each duplicate is also counted as same group or other group:
    the group of the duplicate compared with the group of the detection
    matched to its nearest matched truth point within radius. counts hold
    ``matched``, ``duplicate``, ``spurious`` and, with groups,
    ``duplicate_same_group`` and ``duplicate_other_group``; radius is in the
    matching units.
    """
    detected, truth = _points(detected, "detected"), _points(truth, "truth")
    if not np.isfinite(radius) or radius < 0:
        raise ValueError("radius must be finite and nonnegative")
    if boundary not in ("inclusive", "exclusive"):
        raise ValueError("boundary must be inclusive or exclusive")
    if groups is not None:
        groups = np.asarray(groups)
        if groups.shape != (len(detected),):
            raise ValueError("groups must hold one label per detection")
    i, j, _ = _pairs(match_result, detected, truth)
    partner = dict(zip(i.tolist(), j.tolist()))
    unmatched = np.setdiff1d(np.arange(len(detected)), j)
    counts = {"matched": int(len(j)), "duplicate": 0, "spurious": 0}
    if groups is not None:
        counts.update(duplicate_same_group=0, duplicate_other_group=0)
    matched_truth = np.array(sorted(partner), dtype=int)
    distance = (cdist(detected[unmatched], truth[matched_truth]) if len(unmatched) and len(matched_truth)
                else np.empty((len(unmatched), 0)))
    near = distance <= radius if boundary == "inclusive" else distance < radius
    for row, d in enumerate(unmatched):
        if not near[row].any():
            counts["spurious"] += 1
            continue
        counts["duplicate"] += 1
        if groups is not None:
            candidates = np.flatnonzero(near[row])
            nearest = matched_truth[candidates[np.argmin(distance[row, candidates])]]
            same = groups[partner[int(nearest)]] == groups[d]
            counts["duplicate_same_group" if same else "duplicate_other_group"] += 1
    return _result({}, {}, counts, {**match_result.config, "radius": float(radius), "radius_boundary": boundary,
                                    "groups": groups is not None})
