"""Codebook loading and bidirectional gene-sequence lookup.

Reference: src/matlab/LoadCodebook.m

Reads a CSV codebook (gene, barcode columns), encodes barcodes to
color-space sequences, and builds forward/reverse lookup dictionaries.
"""

import csv
from pathlib import Path

from starfinder.barcode.encoding import encode_bases


def load_codebook(
    path: str | Path,
    do_reverse: bool = True,
    split_index: int | None = None,
) -> tuple[dict[str, str], dict[str, str]]:
    """Load codebook CSV and build bidirectional gene<->color_seq lookup.

    Parameters
    ----------
    path : str or Path
        Path to CSV file with columns ``gene,barcode``.
    do_reverse : bool
        If True (default), reverse each barcode before encoding.
        This matches the STARmap convention where sequencing reads
        barcodes in reverse order.
    split_index : int or None
        If set, remove the character at this 0-based index from the
        encoded sequence, then swap front/back halves around that
        position. Used for multi-segment barcodes.

    Returns
    -------
    gene_to_seq : dict[str, str]
        Mapping from gene name to color sequence.
    seq_to_gene : dict[str, str]
        Mapping from color sequence to gene name.
    """
    path = Path(path)

    gene_to_seq = {}
    seq_to_gene = {}

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gene = row["gene"]
            barcode = row["barcode"]

            if do_reverse:
                barcode = barcode[::-1]

            color_seq = encode_bases(barcode)

            if split_index is not None:
                # Remove character at split_index, then swap halves
                color_seq = color_seq[:split_index] + color_seq[split_index + 1 :]
                front = color_seq[:split_index]
                back = color_seq[split_index:]
                color_seq = back + front

            gene_to_seq[gene] = color_seq
            seq_to_gene[color_seq] = gene

    return gene_to_seq, seq_to_gene
