"""The pre-W-769 dense greedy implementation, retained only as a test oracle."""
import numpy as np
from scipy.spatial.distance import cdist


def dense_greedy_oracle(reference, observed, threshold, boundary, eligible_reference, eligible_observed):
    ri, oi = np.flatnonzero(eligible_reference), np.flatnonzero(eligible_observed)
    distances = cdist(reference[ri], observed[oi])
    valid = distances <= threshold if boundary == "inclusive" else distances < threshold
    candidates = sorted((float(distances[i, j]), int(ri[i]), int(oi[j])) for i, j in zip(*np.nonzero(valid)))
    pairs, used_ref, used_obs = [], set(), set()
    for distance, i, j in candidates:
        if i not in used_ref and j not in used_obs:
            pairs.append((i, j, distance))
            used_ref.add(i)
            used_obs.add(j)
    return pairs


