"""Validated, ordered STARmap codebooks. Colors are symbols, not channels."""

from dataclasses import dataclass, field
from numbers import Integral
from pathlib import Path
import csv

import pandas as pd

from ._encoding import encode_bases


def _labels(labels, name):
    if (
        not isinstance(labels, tuple)
        or not labels
        or any(not isinstance(x, str) or not x for x in labels)
        or len(set(labels)) != len(labels)
    ):
        raise ValueError(f"{name} must be a nonempty tuple of unique strings")


@dataclass(frozen=True)
class EncodingConfig:
    """Reverse bases before encoding; optionally remove/swap at split_index.

    The split indexes the encoded sequence, removes that color, then moves the
    trailing segment before the leading segment. Both segments must be nonempty.
    """

    reverse_bases: bool = True
    split_index: int | None = None

    def __post_init__(self):
        if type(self.reverse_bases) is not bool:
            raise ValueError("reverse_bases must be Boolean")
        if self.split_index is not None and (
            isinstance(self.split_index, bool)
            or not isinstance(self.split_index, Integral)
            or self.split_index < 1
        ):
            raise ValueError("split_index must be a positive integer")

    def encode(self, bases: str) -> str:
        """Encode one validated base sequence with this declared policy."""
        if not isinstance(bases, str) or len(bases) < 2 or any(b not in "ACGT" for b in bases):
            raise ValueError("base_sequence must contain at least two uppercase A/C/G/T bases")
        seq = encode_bases(bases[::-1] if self.reverse_bases else bases)
        if self.split_index is not None:
            i = self.split_index
            if i >= len(seq) - 1:
                raise ValueError("split_index must leave two nonempty encoded segments")
            seq = seq[i + 1 :] + seq[:i]
        return seq


@dataclass(frozen=True)
class Codebook:
    """Ordered gene_id/color_sequence table, labels, channel mapping and encoding.

    Optional base_sequence is checked against the declared encoding. Duplicate
    genes or encoded sequences are errors, including duplicate identical rows.
    color_to_channel maps all four STARmap symbols to distinct zero-based indices.
    """

    table: pd.DataFrame
    round_labels: tuple[str, ...]
    channel_labels: tuple[str, ...]
    color_to_channel: dict[str, int] = field(
        default_factory=lambda: {str(i + 1): i for i in range(4)}
    )
    encoding: EncodingConfig = field(default_factory=EncodingConfig)

    def __post_init__(self):
        _labels(self.round_labels, "round_labels")
        _labels(self.channel_labels, "channel_labels")
        if not isinstance(self.encoding, EncodingConfig):
            raise TypeError("encoding must be EncodingConfig")
        mapping = self.color_to_channel
        if (
            set(mapping) != set("1234")
            or len(self.channel_labels) != 4
            or any(isinstance(v, bool) or not isinstance(v, Integral) for v in mapping.values())
            or set(mapping.values()) != set(range(4))
        ):
            raise ValueError(
                "color_to_channel must bijectively map colors 1–4 onto four labeled channels"
            )
        if not self.table.columns.is_unique or not {"gene_id", "color_sequence"}.issubset(
            self.table
        ):
            raise ValueError("codebook requires unique gene_id/color_sequence columns")
        genes, sequences = set(), set()
        for i, row in enumerate(self.table.to_dict("records"), 1):
            gene, seq = row["gene_id"], row["color_sequence"]
            try:
                if not isinstance(gene, str) or not gene.strip():
                    raise ValueError("gene_id must be nonempty")
                if gene in genes:
                    raise ValueError(f"duplicate gene_id {gene!r}")
                if (
                    not isinstance(seq, str)
                    or len(seq) != len(self.round_labels)
                    or any(c not in "1234" for c in seq)
                ):
                    raise ValueError("invalid color_sequence symbols or length")
                if seq in sequences:
                    raise ValueError(f"encoded sequence collision {seq!r}")
                if "base_sequence" in row and self.encoding.encode(row["base_sequence"]) != seq:
                    raise ValueError("base_sequence disagrees with color_sequence/encoding")
                genes.add(gene)
                sequences.add(seq)
            except ValueError as exc:
                raise ValueError(f"codebook row {i}: {exc}") from exc
        object.__setattr__(
            self,
            "table",
            self.table.copy().astype(
                {
                    c: "string"
                    for c in self.table.columns
                    if c in ("gene_id", "color_sequence", "base_sequence")
                }
            ),
        )
        object.__setattr__(self, "color_to_channel", dict(mapping))

    @property
    def gene_to_seq(self):
        """Gene lookup; validated uniqueness prevents silent overwrites."""
        return dict(zip(self.table.gene_id, self.table.color_sequence))

    @property
    def seq_to_gene(self):
        """Color sequence lookup in source order."""
        return dict(zip(self.table.color_sequence, self.table.gene_id))

    @property
    def genes(self):
        """Gene IDs in source order."""
        return self.table.gene_id.tolist()

    @property
    def n_genes(self):
        """Number of validated rows."""
        return len(self.table)


def load_codebook(
    path: str | Path,
    *,
    round_labels: tuple[str, ...],
    channel_labels: tuple[str, ...],
    encoding: EncodingConfig = EncodingConfig(),
    color_to_channel: dict[str, int] | None = None,
) -> Codebook:
    """Read headerless gene,barcode or gene/barcode CSV (UTF-8 BOM accepted).

    Labels are required; no acquisition order is inferred from sequence length.
    Canonical gene_id/color_sequence[/base_sequence] headers are also supported.
    Errors identify source rows; no collisions overwrite earlier records.
    """
    records = []
    seen_genes, seen_sequences = set(), set()
    with open(path, newline="", encoding="utf-8-sig") as handle:
        rows = csv.reader(handle)
        header = None
        for line, values in enumerate(rows, 1):
            if line == 1 and values and values[0].strip() in ("gene", "gene_id"):
                header = [v.strip() for v in values]
                continue
            try:
                if header is None or header == ["gene", "barcode"]:
                    if len(values) != 2:
                        raise ValueError("expected gene,barcode")
                    gene, bases = (v.strip() for v in values)
                    row = dict(
                        gene_id=gene, base_sequence=bases, color_sequence=encoding.encode(bases)
                    )
                else:
                    if len(values) != len(header):
                        raise ValueError("column count mismatch")
                    row = dict(zip(header, (v.strip() for v in values)))
                    if "gene_id" not in row or "color_sequence" not in row:
                        raise ValueError("expected gene_id/color_sequence columns")
                if row["gene_id"] in seen_genes or row["color_sequence"] in seen_sequences:
                    raise ValueError("duplicate gene ID or encoded sequence collision")
                seen_genes.add(row["gene_id"])
                seen_sequences.add(row["color_sequence"])
                records.append(row)
                Codebook(
                    pd.DataFrame([row]),
                    round_labels,
                    channel_labels,
                    color_to_channel
                    if color_to_channel is not None
                    else {str(i + 1): i for i in range(4)},
                    encoding,
                )
            except (ValueError, KeyError) as exc:
                raise ValueError(f"{path}: row {line}: {exc}") from exc
    table = (
        pd.DataFrame(records) if records else pd.DataFrame(columns=["gene_id", "color_sequence"])
    )
    return Codebook(
        table,
        round_labels,
        channel_labels,
        color_to_channel if color_to_channel is not None else {str(i + 1): i for i in range(4)},
        encoding,
    )
