"""Validated, ordered STARmap codebooks. Colors are symbols, not channels.

A codebook row is an entry: one barcode with its color sequence. Several entries
may share a gene (docs/readout-contract.md, "Codebook entries and genes").
Encodings are registered in ENCODINGS by exact config type; the segment layout
is a typed description on the codebook.
"""

from dataclasses import KW_ONLY, dataclass, field
from numbers import Integral
from pathlib import Path
from typing import Any, Callable
import csv

import pandas as pd

from starfinder._registry import Dependency, check_name, spec_for
from ._encoding import decode_color_sequence, encode_bases
from ._layout import (
    BarcodeLayout,
    Segment,
    check_layout,
    colors_have_ends,
    encode_layout,
    read_orientation,
    segment_bases,
    segment_colors,
)

_BASES = "ACGT"


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
    """two_base encoding: reverse bases before encoding; optionally remove/swap at split_index.

    The split indexes the encoded sequence (zero-based), removes that color, then
    moves the trailing segment before the leading segment. Both segments must be
    nonempty. split_index is the legacy two-segment form: a codebook translates it
    into a two-segment BarcodeLayout, and it is mutually exclusive with
    Codebook.layout. MATLAB's split_index s equals split_index s - 1 here.
    """

    reverse_bases: bool = True
    split_index: int | None = None
    method: str = field(default="two_base", init=False)

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
class OneBaseEncodingConfig:
    """one_base encoding: each base is one round's color through base_to_color.

    base_to_color maps A, C, G and T one-to-one onto the colors "1" to "4" and has
    no default. reverse_bases reverses a segment's bases before encoding.
    """

    base_to_color: dict[str, str]
    reverse_bases: bool = False
    method: str = field(default="one_base", init=False)

    def __post_init__(self):
        mapping = self.base_to_color
        if (
            not isinstance(mapping, dict)
            or set(mapping) != set(_BASES)
            or any(not isinstance(c, str) for c in mapping.values())
            or set(mapping.values()) != set("1234")
        ):
            raise ValueError("base_to_color must map A, C, G and T one-to-one onto the colors 1-4")
        if type(self.reverse_bases) is not bool:
            raise ValueError("reverse_bases must be Boolean")
        object.__setattr__(self, "base_to_color", dict(mapping))


@dataclass(frozen=True)
class EncodingSpec:
    """Registered barcode encoding: stable name, the bases/colors maps and declared capabilities.

    encode(bases, config) gives the colors of one segment's bases (barcode order) in
    acquisition order; decode(colors, config, first_base) gives the segment's bases
    in barcode order, where first_base is the segment's first base in read
    orientation, or None when needs_first_base is False. colors_for(n) is the
    number of colors of an n-base segment, junction_colors the colors spanning two
    adjacent segments that are never acquired. alphabet holds the color symbols,
    which map to channels through Codebook.color_to_channel; symbols is the kind a
    decoder declares in DecodingSpec.encodings ("color": one color per round).
    requires lists optional dependencies.
    """

    name: str
    encode: Callable[[str, Any], str]
    decode: Callable[[str, Any, str | None], str]
    colors_for: Callable[[int], int]
    _: KW_ONLY
    needs_first_base: bool
    junction_colors: int
    alphabet: str = "1234"
    symbols: str = "color"
    requires: tuple[Dependency, ...] = ()

    def __post_init__(self):
        check_name(self.name, "encoding")
        for name in ("encode", "decode", "colors_for"):
            if not callable(getattr(self, name)):
                raise TypeError(f"encoding {self.name!r} {name} must be callable")
        if not isinstance(self.needs_first_base, bool):
            raise TypeError(f"needs_first_base of {self.name!r} must be Boolean")
        if (
            isinstance(self.junction_colors, bool)
            or not isinstance(self.junction_colors, int)
            or self.junction_colors < 0
        ):
            raise ValueError(f"junction_colors of {self.name!r} must be a nonnegative integer")
        for name in ("alphabet", "symbols"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} of {self.name!r} must be a nonempty string")
        if not isinstance(self.requires, tuple) or not all(isinstance(d, Dependency) for d in self.requires):
            raise TypeError(f"encoding {self.name!r} requires must be a tuple of Dependency")


def _segment(bases, minimum):
    if not isinstance(bases, str) or len(bases) < minimum or any(b not in _BASES for b in bases):
        raise ValueError(f"a segment needs at least {minimum} uppercase A/C/G/T bases")


def _colors(colors, alphabet="1234"):
    if not isinstance(colors, str) or not colors or any(c not in alphabet for c in colors):
        raise ValueError(f"colors must be a nonempty string of {alphabet}")


def _two_base_encode(bases, config):
    _segment(bases, 2)
    return encode_bases(read_orientation(bases, config))


def _two_base_decode(colors, config, first_base):
    _colors(colors)
    if first_base not in tuple(_BASES):
        raise ValueError("two_base decoding needs the first base (A/C/G/T) in read orientation")
    return read_orientation(decode_color_sequence(colors, first_base), config)


def _one_base_encode(bases, config):
    _segment(bases, 1)
    return "".join(config.base_to_color[b] for b in read_orientation(bases, config))


def _one_base_decode(colors, config, first_base=None):
    _colors(colors)
    inverse = {color: base for base, color in config.base_to_color.items()}
    return read_orientation("".join(inverse[c] for c in colors), config)


#: Barcode encodings by exact config type (docs/readout-contract.md, "Encoding registry").
ENCODINGS: dict[type, EncodingSpec] = {
    EncodingConfig: EncodingSpec(
        "two_base", _two_base_encode, _two_base_decode, lambda n: n - 1,
        needs_first_base=True, junction_colors=1),
    OneBaseEncodingConfig: EncodingSpec(
        "one_base", _one_base_encode, _one_base_decode, lambda n: n,
        needs_first_base=False, junction_colors=0),
}


def encoding_spec(config) -> EncodingSpec:
    """Spec registered for type(config) exactly; TypeError otherwise."""
    return spec_for(ENCODINGS, config, "encoding", TypeError,
                    "encoding must be a config registered in ENCODINGS (exact type)")


def _layout(encoding, spec, layout, n_rounds):
    """Effective layout: the given one, the legacy split's two segments, or one segment."""
    split = getattr(encoding, "split_index", None)
    if split is not None:
        # The legacy split: n bases give n - 2 colors; MATLAB's one-based s is split + 1.
        n, s = n_rounds + 2 * spec.junction_colors, split + 1
        if n - s < 2:
            raise ValueError("split_index must leave two nonempty encoded segments")
        derived = (
            BarcodeLayout((Segment("A", n - s), Segment("B", s)), ("A", "B"))
            if encoding.reverse_bases
            else BarcodeLayout((Segment("A", s), Segment("B", n - s)), ("B", "A"))
        )
        if layout is not None and layout != derived:
            raise ValueError("EncodingConfig.split_index and Codebook.layout are mutually exclusive")
        layout = derived
    elif layout is None:
        # One segment over the whole barcode: n bases give n - junction_colors colors.
        layout = BarcodeLayout((Segment("A", n_rounds + spec.junction_colors),))
    elif not isinstance(layout, BarcodeLayout):
        raise TypeError("layout must be BarcodeLayout")
    layout.__post_init__()
    check_layout(layout, spec, n_rounds)
    return layout


def _check_ends(row, layout, spec, encoding):
    """Raise ValueError naming the entry and segment whose ends are not among the declared ones."""
    if not any(s.ends for s in layout.segments):
        return
    bases = row.get("base_sequence")
    colors = segment_colors(row["color_sequence"], layout, spec)
    parts = segment_bases(bases, layout) if isinstance(bases, str) else None
    for segment in layout.segments:
        if not segment.ends:
            continue
        if parts is not None:
            read = read_orientation(parts[segment.name], encoding)
            valid = (read[0], read[-1]) in segment.ends
        else:
            valid = colors_have_ends(colors[segment.name], segment.ends, spec, encoding)
        if not valid:
            allowed = ", ".join(f + l for f, l in segment.ends)
            raise ValueError(
                f"entry {row['entry_id']!r}: segment {segment.name!r} ends are not among {allowed}")


def _collision(seen, column, value, row):
    """Raise ValueError naming both source rows of a repeated entry_id, base_sequence or color_sequence."""
    if value in seen[column]:
        if column == "color_sequence":
            raise ValueError(f"encoded sequence collision {value!r} with row {seen[column][value]}")
        raise ValueError(f"repeated {column} {value!r} of row {seen[column][value]}")
    seen[column][value] = row


@dataclass(frozen=True)
class Codebook:
    """Ordered entry table, labels, channel mapping, encoding and segment layout.

    The table has entry_id (unique), gene_id (may repeat), color_sequence (unique)
    and an optional base_sequence (unique), which is checked against the encoding
    and layout. Without an entry_id column, entry_id is the color_sequence.
    Repeated entry_id, color_sequence or base_sequence raise naming both rows.
    color_to_channel maps all four STARmap symbols to distinct zero-based indices.
    layout None means one segment over the whole barcode, or the two segments of
    a legacy EncodingConfig.split_index; after validation it holds the effective
    layout. A segment with declared ends rejects an entry whose ends differ.
    The repr is a one-line size and label summary without table rows.
    """

    table: pd.DataFrame
    round_labels: tuple[str, ...]
    channel_labels: tuple[str, ...]
    color_to_channel: dict[str, int] = field(
        default_factory=lambda: {str(i + 1): i for i in range(4)}
    )
    encoding: EncodingConfig | OneBaseEncodingConfig = field(default_factory=EncodingConfig)
    layout: BarcodeLayout | None = None

    def __post_init__(self):
        _labels(self.round_labels, "round_labels")
        _labels(self.channel_labels, "channel_labels")
        spec = encoding_spec(self.encoding)
        self.encoding.__post_init__()
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
        layout = _layout(self.encoding, spec, self.layout, len(self.round_labels))
        table = self.table.copy()
        if "entry_id" not in table:
            table.insert(0, "entry_id", table["color_sequence"])
        seen = {"entry_id": {}, "base_sequence": {}, "color_sequence": {}}
        for i, row in enumerate(table.to_dict("records"), 1):
            entry, gene, seq = row["entry_id"], row["gene_id"], row["color_sequence"]
            try:
                if not isinstance(entry, str) or not entry.strip():
                    raise ValueError("entry_id must be nonempty")
                if not isinstance(gene, str) or not gene.strip():
                    raise ValueError("gene_id must be nonempty")
                if (
                    not isinstance(seq, str)
                    or len(seq) != len(self.round_labels)
                    or any(c not in spec.alphabet for c in seq)
                ):
                    raise ValueError("invalid color_sequence symbols or length")
                if "base_sequence" in row and encode_layout(
                    row["base_sequence"], layout, spec, self.encoding
                ) != seq:
                    raise ValueError("base_sequence disagrees with color_sequence/encoding")
                _check_ends(row, layout, spec, self.encoding)
                for column in seen:
                    if column in row:
                        _collision(seen, column, row[column], i)
            except ValueError as exc:
                raise ValueError(f"codebook row {i}: {exc}") from exc
        object.__setattr__(
            self,
            "table",
            table.astype(
                {
                    c: "string"
                    for c in table.columns
                    if c in ("entry_id", "gene_id", "color_sequence", "base_sequence")
                }
            ),
        )
        object.__setattr__(self, "color_to_channel", dict(mapping))
        object.__setattr__(self, "layout", layout)

    def __repr__(self):
        size = (
            f"{self.n_genes} genes"
            if self.n_entries == self.n_genes
            else f"{self.n_entries} entries of {self.n_genes} genes"
        )
        return (
            f"Codebook: {size} × {len(self.round_labels)} rounds, "
            f"channels {', '.join(self.channel_labels)}"
        )

    @property
    def gene_to_seq(self):
        """Gene lookup for codebooks whose genes are unique; ValueError otherwise (use entry_to_seq)."""
        if self.n_genes != self.n_entries:
            raise ValueError(
                f"gene_to_seq needs one entry per gene; this codebook has {self.n_entries} entries "
                f"of {self.n_genes} genes (use entry_to_seq)"
            )
        return dict(zip(self.table.gene_id, self.table.color_sequence))

    @property
    def seq_to_gene(self):
        """Color sequence to gene lookup in source order."""
        return dict(zip(self.table.color_sequence, self.table.gene_id))

    @property
    def seq_to_entry(self):
        """Color sequence to entry_id lookup in source order."""
        return dict(zip(self.table.color_sequence, self.table.entry_id))

    @property
    def entry_to_seq(self):
        """entry_id to color sequence lookup in source order."""
        return dict(zip(self.table.entry_id, self.table.color_sequence))

    @property
    def genes(self):
        """Distinct gene IDs in first-appearance order."""
        return list(dict.fromkeys(self.table.gene_id.tolist()))

    @property
    def n_genes(self):
        """Number of distinct genes."""
        return len(self.genes)

    @property
    def n_entries(self):
        """Number of entries (rows)."""
        return len(self.table)


def load_codebook(
    path: str | Path,
    *,
    round_labels: tuple[str, ...],
    channel_labels: tuple[str, ...],
    encoding: EncodingConfig | OneBaseEncodingConfig = EncodingConfig(),
    color_to_channel: dict[str, int] | None = None,
    layout: BarcodeLayout | None = None,
) -> Codebook:
    """Read headerless gene,barcode or gene/barcode CSV (UTF-8 BOM accepted).

    Labels are required; no acquisition order is inferred from sequence length.
    A gene,barcode row is one entry: gene_id is the first column and entry_id the
    barcode as written. Canonical entry_id,gene_id,color_sequence[,base_sequence]
    headers are also supported, with entry_id optional (default color_sequence).
    Barcodes are encoded with encoding and layout (None: one segment, or the
    legacy split of EncodingConfig.split_index). Errors identify source rows; a
    repeated entry_id, base_sequence or color_sequence names both rows.
    """
    mapping = color_to_channel if color_to_channel is not None else {str(i + 1): i for i in range(4)}
    spec = encoding_spec(encoding)
    # Validates the encoding, labels and layout once and resolves the effective layout.
    probe = Codebook(pd.DataFrame(columns=["gene_id", "color_sequence"]), round_labels,
                     channel_labels, mapping, encoding, layout)
    records = []
    seen = {"entry_id": {}, "base_sequence": {}, "color_sequence": {}}
    with open(path, newline="", encoding="utf-8-sig") as handle:
        rows = csv.reader(handle)
        header = None
        for line, values in enumerate(rows, 1):
            if line == 1 and values and values[0].strip() in ("gene", "gene_id", "entry_id"):
                header = [v.strip() for v in values]
                continue
            try:
                if header is None or header == ["gene", "barcode"]:
                    if len(values) != 2:
                        raise ValueError("expected gene,barcode")
                    gene, bases = (v.strip() for v in values)
                    colors = (encoding.encode(bases) if layout is None and type(encoding) is EncodingConfig
                              else encode_layout(bases, probe.layout, spec, encoding))
                    row = dict(entry_id=bases, gene_id=gene, base_sequence=bases, color_sequence=colors)
                else:
                    if len(values) != len(header):
                        raise ValueError("column count mismatch")
                    row = dict(zip(header, (v.strip() for v in values)))
                    if "gene_id" not in row or "color_sequence" not in row:
                        raise ValueError("expected gene_id/color_sequence columns")
                    if "entry_id" not in row:
                        row = dict(entry_id=row["color_sequence"], **row)
                for column in seen:
                    if column in row:
                        _collision(seen, column, row[column], line)
                records.append(row)
                Codebook(pd.DataFrame([row]), round_labels, channel_labels, mapping, encoding, layout)
            except (ValueError, KeyError) as exc:
                raise ValueError(f"{path}: row {line}: {exc}") from exc
    table = (
        pd.DataFrame(records)
        if records
        else pd.DataFrame(columns=["entry_id", "gene_id", "color_sequence"])
    )
    return Codebook(table, round_labels, channel_labels, mapping, encoding, layout)
