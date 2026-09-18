"""Pure decoding accuracy over explicitly supplied spatial matches."""
import pandas as pd
from ._result import _result

__all__ = ["evaluate_decoding"]


def evaluate_decoding(decoded, truth, *, matches):
    """Compare gene/color_seq columns of tables using match_points results.

    Missing columns or truth labels are undefined, never fabricated negatives.
    A missing predicted label is an incorrect call. Accuracy denominators are
    matched pairs with available truth for that label; unmatched truth affects
    detection recall, not conditional decoding accuracy. Rows are positional.
    """
    pairs = matches.details.get("matched_pairs", [])
    if (matches.counts["total_reference"] != len(truth) or
            matches.counts["total_observed"] != len(decoded)):
        raise ValueError("matching populations do not match supplied tables")
    values, counts, reasons, confusion = {}, dict(matches.counts), {}, {}
    for column in ("gene", "color_seq"):
        key = column + "_accuracy"
        eligible = correct = 0
        if column not in decoded or column not in truth:
            values[key] = None
            reasons[key] = "missing predicted or truth column"
        else:
            for i, j, _ in pairs:
                target, pred = truth.iloc[i][column], decoded.iloc[j][column]
                if pd.isna(target):
                    continue
                eligible += 1
                same = pd.notna(pred) and str(target) == str(pred)
                correct += int(same)
                if column == "gene" and not same:
                    label = f"{target}->{pred}"
                    confusion[label] = confusion.get(label, 0) + 1
            values[key] = correct / eligible if eligible else None
        counts["eligible_" + column] = eligible
        counts["correct_" + column] = correct
    return _result(values, {k: "fraction" for k in values}, counts,
                   {**matches.config, "denominator": "matched pairs with nonmissing truth label"},
                   reasons=reasons, details={"gene_confusion": confusion},
                   status=matches.status if matches.status in ("missing", "failed") else None)
