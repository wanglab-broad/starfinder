"""Explicit one-to-one point matching without detection or I/O."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from ._result import _result, _geometry, _threshold, _eligible

__all__ = ["match_points"]


def match_points(reference, observed, *, policy, threshold, units,
                 reference_metadata, observed_metadata, boundary="inclusive",
                 eligible_reference=None, eligible_observed=None):
    """Match finite (N,3) ZYX coordinates already expressed in ``units``.

    ``nearest_candidate`` proposes only the nearest observed point per reference,
    then excludes duplicate observed assignments (registration's historical
    KDTree policy). ``greedy`` sorts all admissible pairs by distance, reference
    index and observed index. It can use second candidates. Neither is optimal
    bipartite matching. Pair indices refer to the original supplied populations.
    Threshold boundary is explicitly inclusive or exclusive. Eligibility masks
    exclude points from both candidate generation and metric denominators.
    """
    _geometry(reference_metadata, observed_metadata, units)
    _threshold(threshold)
    if policy not in ("nearest_candidate", "greedy") or boundary not in ("inclusive", "exclusive"):
        raise ValueError("unsupported matching policy or threshold boundary")
    arrays = [np.asarray(x, dtype=float) for x in (reference, observed)]
    if any(x.ndim != 2 or x.shape[1] != 3 or not np.isfinite(x).all() for x in arrays):
        raise ValueError("points must be finite (N,3) ZYX arrays")
    ref, obs = arrays
    ri = np.flatnonzero(_eligible(eligible_reference, len(ref)))
    oi = np.flatnonzero(_eligible(eligible_observed, len(obs)))
    pairs = []
    if len(ri) and len(oi):
        if policy == "nearest_candidate":
            distances, indices = cKDTree(obs[oi]).query(ref[ri], k=1)
            valid = distances <= threshold if boundary == "inclusive" else distances < threshold
            candidates = [(float(distances[i]), int(ri[i]), int(oi[indices[i]]))
                          for i in np.flatnonzero(valid)[np.argsort(distances[valid])]]
        else:
            distances = cdist(ref[ri], obs[oi])
            valid = distances <= threshold if boundary == "inclusive" else distances < threshold
            candidates = sorted((float(distances[i, j]), int(ri[i]), int(oi[j]))
                                for i, j in zip(*np.nonzero(valid)))
        used_ref, used_obs = set(), set()
        for distance, i, j in candidates:
            if i not in used_ref and j not in used_obs:
                pairs.append((i, j, distance))
                used_ref.add(i)
                used_obs.add(j)
    n = len(pairs)
    return _result(
        {"recall": n / len(ri) if len(ri) else None,
         "precision": n / len(oi) if len(oi) else None,
         "mean_distance": float(np.mean([p[2] for p in pairs])) if pairs else None},
        {"recall": "fraction", "precision": "fraction", "mean_distance": units},
        {"total_reference": len(ref), "total_observed": len(obs),
         "eligible_reference": len(ri), "eligible_observed": len(oi), "matched": n},
        {"policy": policy, "threshold": float(threshold), "units": units,
         "boundary": boundary, "frame_id": reference_metadata.frame_id,
         "denominator": "eligible populations"}, details={"matched_pairs": pairs})
