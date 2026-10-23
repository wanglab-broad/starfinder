"""Typed segment layout of a codebook barcode (docs/readout-contract.md, "Segment layout").

A layout is a description, not a registry: segments in barcode order with their
lengths in bases and allowed (first, last) end bases in read orientation, and the
order in which their colors are acquired. The helpers here take the encoding's
spec and config, so they work for every registered encoding.
"""

from dataclasses import dataclass
from numbers import Integral

_BASES = "ACGT"


@dataclass(frozen=True)
class Segment:
    """One separately read part of a barcode.

    name labels the segment (for example "A"); bases is its length in bases, in
    barcode order; ends holds the allowed (first, last) base pairs in read
    orientation (after reverse_bases), and () leaves the ends unchecked.
    """

    name: str
    bases: int
    ends: tuple[tuple[str, str], ...] = ()

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("segment name must be a nonempty string")
        if isinstance(self.bases, bool) or not isinstance(self.bases, Integral) or self.bases < 1:
            raise ValueError(f"segment {self.name!r} bases must be a positive integer")
        if (
            not isinstance(self.ends, tuple)
            or any(
                not isinstance(pair, tuple)
                or len(pair) != 2
                or any(not isinstance(b, str) or b not in _BASES or len(b) != 1 for b in pair)
                for pair in self.ends
            )
            or len(set(self.ends)) != len(self.ends)
        ):
            raise ValueError(
                f"segment {self.name!r} ends must be a tuple of unique (first, last) pairs of A/C/G/T"
            )


@dataclass(frozen=True)
class BarcodeLayout:
    """Segments of a codebook barcode and the order their colors are acquired.

    segments are in barcode order, as written in the codebook file; their lengths
    sum to every entry's barcode length. acquisition_order lists the segment names
    in round order (None: barcode order). Each segment is encoded on its own bases,
    and the color sequence is the segments' colors joined in acquisition order;
    a color spanning two adjacent segments (two_base) is never acquired.
    """

    segments: tuple[Segment, ...]
    acquisition_order: tuple[str, ...] | None = None

    def __post_init__(self):
        if (
            not isinstance(self.segments, tuple)
            or not self.segments
            or any(not isinstance(s, Segment) for s in self.segments)
        ):
            raise ValueError("layout segments must be a nonempty tuple of Segment")
        names = [s.name for s in self.segments]
        if len(set(names)) != len(names):
            raise ValueError("layout segment names must be unique")
        order = self.acquisition_order
        if order is not None and (
            not isinstance(order, tuple) or sorted(order) != sorted(names)
        ):
            raise ValueError("acquisition_order must list every segment name once")

    @property
    def acquired(self) -> tuple[Segment, ...]:
        """Segments in acquisition (round) order."""
        if self.acquisition_order is None:
            return self.segments
        by_name = {s.name: s for s in self.segments}
        return tuple(by_name[name] for name in self.acquisition_order)

    @property
    def n_bases(self) -> int:
        """Barcode length in bases."""
        return sum(s.bases for s in self.segments)


def check_layout(layout, spec, n_rounds):
    """Raise ValueError unless every segment yields a color and the colors fill the rounds."""
    for segment in layout.segments:
        if spec.colors_for(segment.bases) < 1:
            raise ValueError(f"segment {segment.name!r} has too few bases for the {spec.name} encoding")
    total = sum(spec.colors_for(s.bases) for s in layout.segments)
    if total != n_rounds:
        raise ValueError(f"layout segments give {total} colors; the codebook has {n_rounds} rounds")


def encode_layout(bases, layout, spec, config):
    """Color sequence of one barcode: each segment encoded on its own bases, joined in acquisition order."""
    if not isinstance(bases, str) or len(bases) != layout.n_bases:
        raise ValueError(f"base_sequence must have {layout.n_bases} bases for the segment layout")
    colors, start = {}, 0
    for segment in layout.segments:
        colors[segment.name] = spec.encode(bases[start : start + segment.bases], config)
        start += segment.bases
    return "".join(colors[s.name] for s in layout.acquired)


def segment_colors(colors, layout, spec):
    """Mapping of segment name to its slice of an acquired color sequence."""
    out, start = {}, 0
    for segment in layout.acquired:
        n = spec.colors_for(segment.bases)
        out[segment.name] = colors[start : start + n]
        start += n
    return out


def segment_bases(bases, layout):
    """Mapping of segment name to its bases, in barcode order."""
    out, start = {}, 0
    for segment in layout.segments:
        out[segment.name] = bases[start : start + segment.bases]
        start += segment.bases
    return out


def read_orientation(bases, config):
    """Bases of a segment in read orientation (reversed when the encoding reverses)."""
    return bases[::-1] if getattr(config, "reverse_bases", False) else bases


def colors_have_ends(colors, ends, spec, config):
    """Whether a segment's colors decode, from some allowed first base, to an allowed (first, last) pair.

    Colors outside the encoding's alphabet (M, N or missing values) never pass.
    """
    if not isinstance(colors, str) or not colors or any(c not in spec.alphabet for c in colors):
        return False
    for first, last in ends:
        try:
            read = read_orientation(
                spec.decode(colors, config, first if spec.needs_first_base else None), config
            )
        except ValueError:
            continue
        if read[0] == first and read[-1] == last:
            return True
    return False
