"""Historical L12/W-393 helpers, moved unchanged for evaluation reuse.

The single-owner half-tie rule remains historical behavior; public M55
adds the other qualifying relation without changing these definitions.
"""
import numpy as np

# The original W-393 assertion tolerance; the helper body is unchanged.
TOL = 1e-9


def l12_agreement(reference, other):
    """(labels in reference, labels in other, matched fraction of the reference labels, median matched IoU).

    Pairs at IoU ≥ 0.5 are matched one to one, the larger IoU first (two pairs can share a label only
    at exactly 0.5).
    """
    a, b = np.asarray(reference).ravel(), np.asarray(other).ravel()
    ids_a, size_a = np.unique(a[a > 0], return_counts=True)
    ids_b, size_b = np.unique(b[b > 0], return_counts=True)
    sizes_a, sizes_b = dict(zip(ids_a.tolist(), size_a.tolist())), dict(zip(ids_b.tolist(), size_b.tolist()))
    both = (a > 0) & (b > 0)
    pairs, overlaps = np.unique(np.stack([a[both], b[both]]), axis=1, return_counts=True)
    candidates = []
    for (i, j), overlap in zip(pairs.T.tolist(), overlaps.tolist()):
        iou = overlap / (sizes_a[i] + sizes_b[j] - overlap)
        if iou >= 0.5:
            candidates.append((-iou, i, j))
    used_a, used_b, matched = set(), set(), []
    for negative, i, j in sorted(candidates):
        if i not in used_a and j not in used_b:
            used_a.add(i)
            used_b.add(j)
            matched.append(-negative)
    fraction = len(matched) / len(sizes_a) if sizes_a else 1.0
    median = float(np.median(matched)) if matched else float("nan")
    return len(sizes_a), len(sizes_b), fraction, median


def match_nuclei(truth, pred):
    """W-393's definitions (notes.md §2) on L12's matching (``l12_agreement``: one to one at IoU ≥ 0.5, the larger
    IoU first): a predicted label is owned by the truth nucleus holding at least half of its voxels; a split is a
    truth nucleus owning at least two predicted labels; a merge is a predicted label holding at least half of each of
    at least two truth nuclei; missed is a truth nucleus not matched and in no split or merge; a false positive is a
    predicted label not matched and owned by no truth nucleus. Each truth nucleus is counted once by its ID with all
    its voxels (V4)."""
    t, p = np.asarray(truth).ravel(), np.asarray(pred).ravel()
    tid, tsz = np.unique(t[t > 0], return_counts=True)
    pid, psz = np.unique(p[p > 0], return_counts=True)
    ts, ps = dict(zip(tid.tolist(), tsz.tolist())), dict(zip(pid.tolist(), psz.tolist()))
    both = (t > 0) & (p > 0)
    pairs, overlaps = (np.unique(np.stack([t[both], p[both]]), axis=1, return_counts=True) if both.any()
                       else (np.zeros((2, 0), np.int64), np.zeros(0, np.int64)))
    candidates, owner, held = [], {}, {}
    for (i, j), o in zip(pairs.T.tolist(), overlaps.tolist()):
        iou = o / (ts[i] + ps[j] - o)
        if iou >= 0.5:
            candidates.append((-iou, i, j))
        if o >= 0.5 * ps[j]:
            owner[j] = i
        if o >= 0.5 * ts[i]:
            held.setdefault(j, []).append(i)
    used_t, used_p, matched = set(), set(), {}
    for negative, i, j in sorted(candidates):
        if i not in used_t and j not in used_p:
            used_t.add(i)
            used_p.add(j)
            matched[j] = (i, -negative)
    owned = {}
    for j, i in owner.items():
        owned.setdefault(i, []).append(j)
    split = {i for i, js in owned.items() if len(js) >= 2}
    merges = {j for j, ii in held.items() if len(ii) >= 2}
    merged = {i for j in merges for i in held[j]}
    missed = [i for i in ts if i not in used_t and i not in split and i not in merged]
    false_positive = [j for j in ps if j not in used_p and j not in owner]
    # The matching is L12's: its matched fraction of the truth labels is this one.
    n_truth, _, fraction, _ = l12_agreement(truth, pred)
    assert n_truth == len(ts) and abs(fraction * n_truth - len(matched)) <= TOL * max(1, n_truth)
    return dict(truth=ts, pred=ps, matched=matched, split=split, merges=merges, merged=merged, missed=missed,
                false_positive=false_positive, held=held, owned=owned)
