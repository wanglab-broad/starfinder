"""Type definitions for the dataset/FOV layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeAlias

import numpy as np

# Type aliases
Shift3D: TypeAlias = tuple[float, float, float]  # (dz, dy, dx)
ImageArray: TypeAlias = np.ndarray  # Shape: (Z, Y, X, C)
ChannelOrder: TypeAlias = list[str]  # e.g., ["ch00", "ch01", "ch02", "ch03"]


@dataclass
class LayerState:
    """Tracks which rounds belong to sequencing vs other categories.

    Invariants:
    - ``ref`` must be in ``seq`` or ``other`` (if set)
    - A round cannot appear in both ``seq`` and ``other``
    """

    seq: list[str] = field(default_factory=list)
    other: list[str] = field(default_factory=list)
    ref: str | None = None

    @property
    def all_layers(self) -> list[str]:
        """All loaded layers in order (seq first, then other)."""
        return self.seq + self.other

    @property
    def to_register(self) -> list[str]:
        """Layers that need registration (all except ref)."""
        return [r for r in self.all_layers if r != self.ref]

    def validate(self) -> None:
        """Check invariants. Raises ValueError if violated."""
        if self.ref is not None and self.ref not in self.all_layers:
            raise ValueError(f"ref '{self.ref}' not found in seq or other")
        overlap = set(self.seq) & set(self.other)
        if overlap:
            raise ValueError(f"Rounds in both seq and other: {overlap}")


@dataclass
class Codebook:
    """Barcode-to-gene mapping with factory method.

    Wraps the two dicts returned by ``starfinder.barcode.load_codebook()``
    into a single object with named access.
    """

    gene_to_seq: dict[str, str]
    seq_to_gene: dict[str, str]

    @property
    def genes(self) -> list[str]:
        """Ordered gene list."""
        return sorted(self.gene_to_seq.keys())

    @property
    def n_genes(self) -> int:
        return len(self.gene_to_seq)

    @classmethod
    def from_csv(
        cls,
        path: Path | str,
        do_reverse: bool = True,
        split_index: int | None = None,
    ) -> Codebook:
        """Load codebook from CSV file.

        Delegates to ``starfinder.barcode.load_codebook()``.
        """
        from starfinder.barcode import load_codebook

        gene_to_seq, seq_to_gene = load_codebook(
            path, do_reverse=do_reverse, split_index=split_index
        )
        return cls(gene_to_seq=gene_to_seq, seq_to_gene=seq_to_gene)


@dataclass(frozen=True)
class CropWindow:
    """Immutable crop region for subtile extraction (Y/X only; Z kept whole).

    All coordinates are 0-based with exclusive end (Python slice convention).
    """

    y_start: int
    y_end: int  # exclusive
    x_start: int
    x_end: int  # exclusive

    def to_slice(self) -> tuple[slice, slice]:
        """Return (slice_y, slice_x) for array indexing."""
        return (
            slice(self.y_start, self.y_end),
            slice(self.x_start, self.x_end),
        )


@dataclass
class SubtileConfig:
    """Dataset-level subtile partitioning configuration.

    Computes overlapping 2D windows that tile the Y/X plane.
    Matches MATLAB ``MakeSubtileTable`` / ``CreateSubtiles`` tiling logic.
    """

    sqrt_pieces: int
    overlap_ratio: float = 0.1
    windows: list[CropWindow] = field(default_factory=list)

    @property
    def n_subtiles(self) -> int:
        """Total number of subtiles."""
        return len(self.windows)

    def compute_windows(self, height: int, width: int) -> None:
        """Populate self.windows for a given (Y, X) image size.

        Tiles are ``sqrt_pieces x sqrt_pieces`` with overlap. Edge tiles
        are clamped to image boundaries. Outer edges have no overlap
        extension. Uses height for tile size (MATLAB uses dims(1)).
        """
        n = self.sqrt_pieces
        tile_size = height // n
        overlap_half = int(tile_size * self.overlap_ratio) // 2

        self.windows = []
        for row in range(n):
            for col in range(n):
                # Base tile boundaries
                y0 = row * tile_size
                y1 = (row + 1) * tile_size
                x0 = col * tile_size
                x1 = (col + 1) * tile_size

                # Extend by overlap (except at outer edges)
                if row > 0:
                    y0 -= overlap_half
                if row < n - 1:
                    y1 += overlap_half
                if col > 0:
                    x0 -= overlap_half
                if col < n - 1:
                    x1 += overlap_half

                # Clamp to image boundary
                y1 = min(y1, height)
                x1 = min(x1, width)

                self.windows.append(CropWindow(y0, y1, x0, x1))
