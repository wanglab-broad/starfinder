"""Pure instance metrics; zero is background and IDs retain disconnected pieces."""
import numpy as np

from starfinder.assignment._correspondence import overlap_counts

from ._result import EvaluationResult, _result
from ._segmentation_helpers import match_nuclei as _match_nuclei

__all__ = ["instance_f1", "split_merge_counts", "label_size_summary"]


def _label_pair(truth, predicted):
    arrays = tuple(np.asarray(a) for a in (truth, predicted))
    for a in arrays:
        if a.ndim != 3 or 0 in a.shape or a.dtype.kind not in "ui" or np.any(a < 0):
            raise ValueError("labels must be nonempty nonnegative integer ZYX arrays")
    if arrays[0].shape != arrays[1].shape:
        raise ValueError("truth and predicted labels must share the same grid shape")
    # Shared integer dtype prevents NumPy stacking signed/unsigned labels as
    # float64 inside the historical helpers, which could round sparse IDs.
    return tuple(a.astype(np.uint64, copy=False) for a in arrays)


def instance_f1(truth: np.ndarray, predicted: np.ndarray) -> EvaluationResult:
    """Evaluate descending-IoU greedy instance F1 at IoU 0.5–0.9 (M54).

    Parameters
    ----------
    truth : numpy.ndarray
        Nonempty nonnegative integer ZYX truth labels on the comparison grid.
        Zero is background; each distinct nonzero ID denotes one instance.
    predicted : numpy.ndarray
        Predicted labels on the same grid and with the same format. Disconnected
        pieces retain their label identity in both populations.

    Returns
    -------
    EvaluationResult
        F1 at each inclusive IoU threshold 0.5, 0.6, 0.7, 0.8 and 0.9, and a
        curve of TP/FP/FN counts and matched (truth ID, predicted ID, IoU) pairs.
        The historical L12/W-393 greedy matching orders descending IoU, then
        ascending truth and predicted IDs; it is not optimal matching. Higher
        thresholds retain the prefix of those greedy matches. No truth leaves
        F1 undefined with a reason, including when both label images are empty.
    """
    truth, predicted = _label_pair(truth, predicted)
    matched = _match_nuclei(truth, predicted)
    nt, npred = len(matched["truth"]), len(matched["pred"])
    thresholds = [.5, .6, .7, .8, .9]
    curve, values = [], {}
    for threshold in thresholds:
        # Lower-IoU candidates follow every higher-IoU candidate in the shared
        # greedy ordering, so dropping them cannot change the higher matches.
        pairs = [(i, j, iou) for j, (i, iou) in matched["matched"].items() if iou >= threshold]
        tp = len(pairs)
        f1 = 2 * tp / (nt + npred) if nt else None
        row = dict(iou_threshold=threshold, tp=tp, fp=npred - tp, fn=nt - tp,
                   f1=f1, matched_pairs=pairs)
        if f1 is None:
            row["reason"] = "no truth instances"
        curve.append(row)
        values[f"f1_at_{threshold:g}"] = f1
    return _result(values, {k: "fraction" for k in values}, {"truth": nt, "predicted": npred},
                   {"iou_thresholds": thresholds, "boundary": "inclusive", "policy": "descending-IoU greedy",
                    "ties": "truth ID, then predicted ID", "background": 0},
                   reasons={k: "no truth instances" for k in values} if not nt else {},
                   details={"curve": curve})


def split_merge_counts(truth: np.ndarray, predicted: np.ndarray) -> EvaluationResult:
    """Count truth splits and predicted merges using all half-ownership ties (M55/M56).

    Parameters
    ----------
    truth : numpy.ndarray
        Nonempty nonnegative integer ZYX truth labels on the comparison grid.
        Zero is background and disconnected pieces with one ID remain one instance.
    predicted : numpy.ndarray
        Predicted labels on the same grid and in the same format.

    Returns
    -------
    EvaluationResult
        Split count (truth IDs owning at least two predicted labels), merge
        count (predicted IDs covering at least half of each of at least two
        truth instances), and the number of distinct truth IDs in those merges.
        Ownership means at least half the predicted label's voxels lie in a
        truth label; exact half ties count both owners. Historical matching and
        merge computations are reused unchanged. Details list split, merge,
        merged-truth IDs and every owned predicted ID. Empty populations give
        zero counts without treating background as an instance.
    """
    truth, predicted = _label_pair(truth, predicted)
    historical = _match_nuclei(truth, predicted)
    owned = {i: set(js) for i, js in historical["owned"].items()}
    # Restore only the extra exact-half owners omitted by the historical
    # single-owner dictionary. Reuse the package overlap counter; compact IDs
    # first so its uint64 pair keys cannot overflow on sparse integer labels.
    ids, ranks = [], []
    for labels in (truth, predicted):
        unique, inverse = np.unique(labels, return_inverse=True)
        ranked = inverse.reshape(labels.shape) + 1
        ranked[labels == 0] = 0
        ids.append(unique)
        ranks.append(ranked)
    overlaps = overlap_counts(*ranks)
    for i, j, count in overlaps.itertuples(index=False, name=None):
        if j:
            truth_id, pred_id = int(ids[0][i - 1]), int(ids[1][j - 1])
            if 2 * count == historical["pred"][pred_id]:
                owned.setdefault(truth_id, set()).add(pred_id)
    split = sorted(i for i, js in owned.items() if len(js) >= 2)
    merges, merged = sorted(historical["merges"]), sorted(historical["merged"])
    values = {"splits": len(split), "merges": len(merges), "merged_truth": len(merged)}
    return _result(values, {k: "count" for k in values},
                   {"truth": len(historical["truth"]), "predicted": len(historical["pred"]), **values},
                   {"ownership": "all truth labels with at least half the predicted voxels",
                    "merge": "predicted label covering at least half of each of two or more truth labels",
                    "background": 0, "instance": "distinct nonzero ID"},
                   details={"split_truth_ids": split, "merge_predicted_ids": merges, "merged_truth_ids": merged,
                            "owned_predicted_ids": {i: sorted(js) for i, js in sorted(owned.items())}})


def label_size_summary(labels: np.ndarray) -> EvaluationResult:
    """Summarize distinct nonzero label sizes in voxels (M58).

    Parameters
    ----------
    labels : numpy.ndarray
        Nonempty ZYX array of nonnegative integer labels. Background is zero;
        disconnected pieces carrying the same ID form one counted instance.

    Returns
    -------
    EvaluationResult
        Label count, median, linear first and third quartiles, IQR in voxels,
        and the per-ID sizes. Empty foreground has count zero and null sizes.
    """
    labels = np.asarray(labels)
    if labels.ndim != 3 or 0 in labels.shape or labels.dtype.kind not in "ui" or np.any(labels < 0):
        raise ValueError("labels must be a nonempty nonnegative integer ZYX array")
    ids, sizes = np.unique(labels[labels > 0], return_counts=True)
    if len(ids):
        q1, median, q3 = np.quantile(sizes, [.25, .5, .75], method="linear").tolist()
        values = dict(median=median, q1=q1, q3=q3, iqr=q3 - q1)
    else:
        values = dict.fromkeys(("median", "q1", "q3", "iqr"))
    return _result(values, {k: "voxel" for k in values}, {"labels": len(ids)},
                   {"background": 0, "quantile_method": "linear", "instance": "distinct nonzero ID"},
                   details={"label_sizes": dict(zip(ids.tolist(), sizes.tolist()))})
