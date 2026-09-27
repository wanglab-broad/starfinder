"""Pure decoding accuracy over explicitly supplied spatial matches."""
import pandas as pd
from ._result import _result

__all__ = ["evaluate_decoding"]


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
