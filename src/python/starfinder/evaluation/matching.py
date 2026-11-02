"""Explicit one-to-one point matching without detection or I/O."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from ._result import EvaluationResult, _result, _geometry, _threshold, _eligible

__all__ = ["match_points", "evaluate_molecules"]


def match_points(reference, observed, *, policy, threshold, units,
                 reference_metadata, observed_metadata, boundary="inclusive",
                 eligible_reference=None, eligible_observed=None):
    """Match finite (N,3) ZYX coordinates already expressed in ``units``.

    Parameters
    ----------
    reference : array_like
        Finite (N, 3) ZYX reference coordinates in the supplied units.
    observed : array_like
        Finite (N, 3) ZYX observed coordinates in the supplied units.
    policy : str
        Matching policy: "nearest_candidate" or "greedy".
    threshold : float
        Non-negative distance threshold in the supplied units.
    units : str
        Coordinate and distance unit label, "voxel" or the metadata spatial unit.
    reference_metadata : ImageMetadata
        Spatial description of the reference coordinate frame.
    observed_metadata : ImageMetadata
        Spatial description of the observed coordinate frame.
    boundary : str
        Threshold comparison mode: "inclusive" or "exclusive".
    eligible_reference : array_like or None
        Boolean eligibility vector for reference points; None includes the full reference
        population.
    eligible_observed : array_like or None
        Boolean eligibility vector for observed points; None includes the full observed population.

    Returns
    -------
    EvaluationResult
        Recall, precision, mean matched distance, population counts, and original-index matched
        pairs.

    Notes
    -----
    ``nearest_candidate`` proposes only the nearest observed point per reference,
    then excludes duplicate observed assignments (registration's historical
    KDTree policy). ``greedy`` sorts all admissible pairs by distance, reference
    index and observed index, using radius-query KD-tree candidates. It can use second candidates. Neither is optimal
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
            # A slightly enlarged radius avoids losing a boundary candidate to
            # tree arithmetic. cdist recomputes only these neighbours, preserving
            # the old distance arithmetic and tie order without an N x M table.
            radius = threshold + 8 * np.finfo(float).eps * max(1., threshold)
            neighbours = cKDTree(obs[oi]).query_ball_point(ref[ri], radius)
            candidates = []
            for i, indices in enumerate(neighbours):
                if not indices:
                    continue
                distances = cdist(ref[ri[i:i + 1]], obs[oi[indices]])[0]
                valid = distances <= threshold if boundary == "inclusive" else distances < threshold
                candidates.extend((float(distances[k]), int(ri[i]), int(oi[indices[k]]))
                                  for k in np.flatnonzero(valid))
            candidates.sort()
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


def evaluate_molecules(reference, observed, *, policy, threshold, units,
                       reference_metadata, observed_metadata, boundary="inclusive") -> EvaluationResult:
    """Compare same-gene molecule positions in both directions (M34, M70 and M71).

    Parameters
    ----------
    reference : pandas.DataFrame
        Reference molecules with zero-based ``z, y, x`` and ``gene_id`` columns.
        For M70, the table from ``import_legacy_molecules_csv`` on MATLAB output.
    observed : pandas.DataFrame
        Observed molecules with the same columns and coordinate convention.
    policy : str
        ``any_neighbor`` for X6/M34's Chebyshev neighbour rule, or a one-to-one
        ``match_points`` policy (``greedy`` or ``nearest_candidate``).
    threshold : float
        Distance limit in supplied units; M34 uses ``1 + 1e-9``.
    units : str
        Coordinate unit, ``voxel`` or the metadata spatial unit.
    reference_metadata : ImageMetadata
        Reference coordinate frame and geometry.
    observed_metadata : ImageMetadata
        Observed coordinate frame and geometry, equal to the reference.
    boundary : str
        ``inclusive`` or ``exclusive`` threshold comparison.

    Returns
    -------
    EvaluationResult
        Reference and observed matched fractions, observed/reference count ratio
        (M71), absolute relative count difference, counts and matching parameters.
        Empty denominators are undefined. Missing genes never match.

    Notes
    -----
    M34 supplies direct molecules as reference, tiled as observed, and uses
    ``any_neighbor`` with the inclusive Chebyshev limit ``1 + 1e-9``. That
    policy permits many-to-one neighbours independently in each direction.
    M70 supplies converted MATLAB molecules as reference and Python as observed,
    using Euclidean one-to-one matching with the specified threshold.
    """
    _geometry(reference_metadata, observed_metadata, units)
    _threshold(threshold)
    if policy not in ("any_neighbor", "greedy", "nearest_candidate"):
        raise ValueError("unsupported molecule matching policy")
    if boundary not in ("inclusive", "exclusive"):
        raise ValueError("unsupported threshold boundary")
    for table in (reference, observed):
        if not {"z", "y", "x", "gene_id"} <= set(table.columns):
            raise ValueError("molecules require z, y, x and gene_id columns")
        xyz = table[["z", "y", "x"]].to_numpy(dtype=float)
        if not np.isfinite(xyz).all():
            raise ValueError("molecule coordinates must be finite")
    matched_reference = matched_observed = 0
    for gene, part in reference.groupby("gene_id"):
        other = observed[observed.gene_id == gene]
        if not len(other):
            continue
        a = part[["z", "y", "x"]].to_numpy(dtype=float)
        b = other[["z", "y", "x"]].to_numpy(dtype=float)
        if policy == "any_neighbor":
            for source, target, direction in ((a, b, 0), (b, a, 1)):
                distance, _ = cKDTree(target).query(source, p=np.inf)
                valid = distance <= threshold if boundary == "inclusive" else distance < threshold
                if direction == 0:
                    matched_reference += int(valid.sum())
                else:
                    matched_observed += int(valid.sum())
        else:
            result = match_points(a, b, policy=policy, threshold=threshold, units=units,
                                  reference_metadata=reference_metadata, observed_metadata=observed_metadata,
                                  boundary=boundary)
            matched_reference += result.counts["matched"]
            matched_observed += result.counts["matched"]
    nr, no = len(reference), len(observed)
    values = {"reference_fraction": matched_reference / nr if nr else None,
              "observed_fraction": matched_observed / no if no else None,
              "count_ratio": no / nr if nr else None,
              "count_difference": abs(no - nr) / nr if nr else None}
    return _result(values, {key: "fraction" if key.endswith("fraction") else "dimensionless" for key in values},
                   {"total_reference": nr, "total_observed": no,
                    "matched_reference": matched_reference, "matched_observed": matched_observed},
                   {"policy": policy, "threshold": float(threshold), "boundary": boundary, "units": units,
                    "distance": "chebyshev" if policy == "any_neighbor" else "euclidean",
                    "frame_id": reference_metadata.frame_id, "gene": "same nonmissing gene_id",
                    "denominator": "full supplied populations"})
