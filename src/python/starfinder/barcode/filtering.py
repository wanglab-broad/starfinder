"""Filter detected reads against codebook and assign gene names.

Reference: src/matlab/FilterReads.m

The actual filtering is by codebook membership only. End-base validation
(if provided) computes diagnostic statistics but does not affect which
reads pass the filter.
"""

import pandas as pd

from starfinder.barcode.encoding import decode_color_seq


def filter_reads(
    spots: pd.DataFrame,
    seq_to_gene: dict[str, str],
    end_bases: str | None = None,
    start_base: str = "C",
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Filter spots by codebook matching and assign gene names.

    Parameters
    ----------
    spots : pd.DataFrame
        Must have a ``color_seq`` column with color sequences (e.g., "4422").
    seq_to_gene : dict[str, str]
        Mapping from color sequence to gene name (from ``load_codebook``).
    end_bases : str or None
        Two-character string (e.g., "CC") specifying expected start and end
        bases of the decoded barcode. Used for diagnostic statistics only —
        does not affect filtering.
    start_base : str
        Known first base for decoding (default "C"). Only used when
        ``end_bases`` is provided.

    Returns
    -------
    good_spots : pd.DataFrame
        Filtered spots with an added ``gene`` column.
    stats : dict[str, float]
        Filtration statistics:
        - ``"n_total"``: total number of input spots
        - ``"n_in_codebook"``: number matching codebook
        - ``"in_codebook"``: fraction matching codebook
        - ``"correct_form"``: fraction with correct start/end bases (if end_bases)
        - ``"validated"``: fraction of correct-form reads in codebook (if end_bases)
    """
    color_seqs = spots["color_seq"]
    n_total = len(spots)

    # Filter by codebook membership
    in_codebook = color_seqs.isin(seq_to_gene)
    n_in_codebook = int(in_codebook.sum())

    stats: dict[str, float] = {
        "n_total": n_total,
        "n_in_codebook": n_in_codebook,
        "in_codebook": n_in_codebook / n_total if n_total > 0 else 0.0,
    }

    # Diagnostic: end-base validation
    if end_bases is not None and len(end_bases) == 2:
        expected_start = end_bases[0]
        expected_end = end_bases[1]

        barcodes = color_seqs.map(
            lambda seq: decode_color_seq(seq, start_base)
            if all(c in "1234" for c in seq)
            else ""
        )
        correct_form = barcodes.map(
            lambda bc: len(bc) > 0
            and bc[0] == expected_start
            and bc[-1] == expected_end
        )
        n_correct = int(correct_form.sum())

        stats["correct_form"] = n_correct / n_total if n_total > 0 else 0.0
        stats["validated"] = (
            n_in_codebook / n_correct if n_correct > 0 else 0.0
        )

    # Build filtered DataFrame with gene assignment
    good_spots = spots.loc[in_codebook].copy()
    good_spots["gene"] = good_spots["color_seq"].map(seq_to_gene)

    return good_spots, stats
