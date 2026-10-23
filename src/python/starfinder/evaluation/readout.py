"""Pure decoding accuracy over explicitly supplied spatial matches, score ranking and duplicate measures."""
import math

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from ._result import _result

__all__ = ["evaluate_decoding", "evaluate_deduplication", "ranking_quality"]


def evaluate_decoding(decoded, truth, *, matches, sequence_column="observed_color_sequence"):
    """Compare decoded gene IDs and color sequences with truth using match_points results.

    ``decoded`` names them ``gene_id`` and ``sequence_column`` (by default
    ``observed_color_sequence``, as in decoding results); ``truth`` names them
    ``gene_id`` and ``color_sequence``, as in synthetic ``spot_truth`` and
    ``formed``. Values are ``gene_id_accuracy`` and ``color_sequence_accuracy``.
    Missing columns or truth labels are undefined, never fabricated negatives.
    A missing predicted label is an incorrect call. Accuracy denominators are
    matched pairs with available truth for that label; unmatched truth affects
    detection recall, not conditional decoding accuracy. Rows are positional.
    """
    if not isinstance(sequence_column, str) or not sequence_column:
        raise ValueError("sequence_column must be a nonempty column name")
    pairs = matches.details.get("matched_pairs", [])
    if (matches.counts["total_reference"] != len(truth) or
            matches.counts["total_observed"] != len(decoded)):
        raise ValueError("matching populations do not match supplied tables")
    values, counts, reasons, confusion = {}, dict(matches.counts), {}, {}
    for name, predicted in (("gene_id", "gene_id"), ("color_sequence", sequence_column)):
        key = name + "_accuracy"
        eligible = correct = 0
        if predicted not in decoded or name not in truth:
            values[key] = None
            reasons[key] = "missing predicted or truth column"
        else:
            for i, j, _ in pairs:
                target, pred = truth.iloc[i][name], decoded.iloc[j][predicted]
                if pd.isna(target):
                    continue
                eligible += 1
                same = pd.notna(pred) and str(target) == str(pred)
                correct += int(same)
                if name == "gene_id" and not same:
                    label = f"{target}->{pred}"
                    confusion[label] = confusion.get(label, 0) + 1
            values[key] = correct / eligible if eligible else None
        counts["eligible_" + name] = eligible
        counts["correct_" + name] = correct
    return _result(values, {k: "fraction" for k in values}, counts,
                   {**matches.config, "denominator": "matched pairs with nonmissing truth label",
                    "sequence_column": sequence_column},
                   reasons=reasons, details={"gene_confusion": confusion},
                   status=matches.status if matches.status in ("missing", "failed") else None)


def _hanley_mcneil(auroc, n_correct, n_incorrect):
    """Hanley and McNeil (1982) standard error of an AUROC."""
    q1, q2 = auroc / (2 - auroc), 2 * auroc * auroc / (1 + auroc)
    variance = (auroc * (1 - auroc) + (n_correct - 1) * (q1 - auroc * auroc)
                + (n_incorrect - 1) * (q2 - auroc * auroc)) / (n_correct * n_incorrect)
    return math.sqrt(max(variance, 0.0))


def _error_at_retention(score, incorrect, q):
    """Incorrect fraction among the best ceil(q*n) scores; a tied boundary group counts proportionally."""
    n = len(score)
    k = int(np.ceil(q * n - 1e-9))
    order = np.argsort(-score, kind="stable")
    s, w = score[order], incorrect[order].astype(float)
    kept = errors = 0.0
    i = 0
    while i < n and kept < k:
        j = i
        while j < n and s[j] == s[i]:
            j += 1
        take = min(j - i, k - kept)
        errors += w[i:j].sum() * take / (j - i)
        kept += take
        i = j
    return float(errors / k)


def ranking_quality(score, correct, *, orientation, retention=(0.5, 0.8, 0.9, 1.0)):
    """How well a score ranks correct calls above incorrect ones (the W-278 scores.csv measures).

    score holds one value per call and correct whether the call is correct;
    orientation is "lower" or "higher", the direction that ranks a call as more
    reliable. Calls with a NaN score are excluded and counted. Values are
    ``auroc`` (the probability that a correct call ranks above an incorrect one,
    ties counting one half), ``auroc_se`` (Hanley and McNeil 1982) and
    ``error_at_<percent>``, the incorrect fraction among the best-ranked
    ceil(q*n) calls for each retention level q, where a tied group at the
    boundary contributes in proportion. An empty class leaves the AUROC and its
    standard error undefined with a reason; no calls leave every value
    undefined. A ranking measure only: no cutoff and no calibrated probability.
    """
    if orientation not in ("lower", "higher"):
        raise ValueError("orientation must be 'lower' or 'higher'")
    retention = tuple(retention)
    if not retention or any(isinstance(q, bool) or not isinstance(q, (int, float)) or not 0 < q <= 1
                            for q in retention):
        raise ValueError("retention levels must be in (0, 1]")
    keys = [f"error_at_{round(q * 100):g}" for q in retention]
    if len(set(keys)) != len(keys):
        raise ValueError("retention levels must be distinct percentages")
    score = np.asarray(score, dtype=float)
    correct = np.asarray(correct)
    if score.ndim != 1 or correct.shape != score.shape or correct.dtype != bool:
        raise ValueError("score and correct must be one-dimensional, of equal length, correct Boolean")
    defined = ~np.isnan(score)
    oriented = (-score if orientation == "lower" else score)[defined]
    good = correct[defined]
    n, n_correct = len(oriented), int(good.sum())
    n_incorrect = n - n_correct
    values, reasons = {"auroc": None, "auroc_se": None}, {}
    if n_correct and n_incorrect:
        ranks = rankdata(np.concatenate([oriented[good], oriented[~good]]))
        auroc = float((ranks[:n_correct].sum() - n_correct * (n_correct + 1) / 2) / (n_correct * n_incorrect))
        values["auroc"], values["auroc_se"] = auroc, _hanley_mcneil(auroc, n_correct, n_incorrect)
    else:
        reason = "no scored calls" if not n else "no incorrect calls" if n_correct else "no correct calls"
        reasons.update(auroc=reason, auroc_se=reason)
    for q, key in zip(retention, keys):
        values[key] = _error_at_retention(oriented, ~good, q) if n else None
        if not n:
            reasons[key] = "no scored calls"
    units = {key: ("dimensionless" if key.startswith("auroc") else "fraction") for key in values}
    return _result(values, units,
                   {"total": int(len(score)), "scored": n, "correct": n_correct, "incorrect": n_incorrect,
                    "score_undefined": int((~defined).sum())},
                   {"orientation": orientation, "retention": list(retention), "ties": "one half (AUROC), "
                    "proportional at the retention boundary", "standard_error": "Hanley-McNeil 1982"},
                   reasons=reasons)


def _labels(values, name, n=None):
    values = list(values)
    if n is not None and len(values) != n:
        raise ValueError(f"{name} must have one value per read")
    return [None if v is None or (not isinstance(v, str) and pd.isna(v)) else v for v in values]


def evaluate_deduplication(groups, source, *, pairs):
    """Missed duplicates and false merges over a stated pair population (the W-278 duplicates.csv measures).

    groups gives each read's duplicate group (reads with equal labels were
    merged; a missing label is a read that was not grouped), source each read's
    true source (for example the amplicon; missing when unattributed), and
    pairs the (i, j) read index pairs of the population, for example the
    cross-channel pairs within 5 voxels plus every true duplicate pair. A pair
    of one source is a true duplicate and a missed duplicate when not merged; a
    pair of two sources is distinct and a false merge when merged; a pair with
    an unattributed read is counted and left out. Rates are over the true
    duplicate and distinct pairs and undefined, with a reason, when there are none.
    """
    groups = _labels(groups, "groups")
    source = _labels(source, "source", len(groups))
    pairs = [tuple(pair) for pair in pairs]
    n = len(groups)
    if any(len(p) != 2 or any(isinstance(k, bool) or not isinstance(k, (int, np.integer)) or not 0 <= k < n
                              for k in p) or p[0] == p[1] for p in pairs):
        raise ValueError("pairs must be (i, j) pairs of distinct read indices")
    if len({frozenset(p) for p in pairs}) != len(pairs):
        raise ValueError("pairs must be unique")
    missed, false, merged = [], [], 0
    true_pairs = distinct = unattributed = 0
    for i, j in pairs:
        together = groups[i] is not None and groups[i] == groups[j]
        merged += together
        if source[i] is None or source[j] is None:
            unattributed += 1
        elif source[i] == source[j]:
            true_pairs += 1
            if not together:
                missed.append((int(i), int(j)))
        else:
            distinct += 1
            if together:
                false.append((int(i), int(j)))
    values = {"missed_duplicates": len(missed), "false_merges": len(false),
              "missed_duplicate_rate": len(missed) / true_pairs if true_pairs else None,
              "false_merge_rate": len(false) / distinct if distinct else None}
    reasons = {}
    if not true_pairs:
        reasons["missed_duplicate_rate"] = "no true duplicate pairs in the population"
    if not distinct:
        reasons["false_merge_rate"] = "no distinct pairs in the population"
    return _result(values, {"missed_duplicates": "count", "false_merges": "count",
                            "missed_duplicate_rate": "fraction", "false_merge_rate": "fraction"},
                   {"pairs": len(pairs), "true_duplicate_pairs": true_pairs, "distinct_pairs": distinct,
                    "unattributed_pairs": unattributed, "merged_pairs": int(merged)},
                   {"population": "supplied pairs", "merged": "equal nonmissing group labels"},
                   reasons=reasons, details={"missed_pairs": missed, "false_merge_pairs": false})
