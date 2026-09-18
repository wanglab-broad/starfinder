"""Type definitions for the dataset/FOV layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeAlias

import numpy as np

# Type aliases
ImageArray: TypeAlias = np.ndarray  # Shape: (Z, Y, X, C)
ChannelOrder: TypeAlias = list[str]  # e.g., ["ch00", "ch01", "ch02", "ch03"]


@dataclass
class LayerState:
    """Tracks which rounds belong to sequencing vs other categories.

    Invariants:

    - ``ref`` must be in ``seq`` or ``other`` (if set)

    - A round cannot appear in both ``seq`` and ``other``

    Parameters
    ----------
    seq : list[str]
        Sequencing round names, default new empty list.
    other : list[str]
        Non-sequencing round names, default new empty list.
    ref : str | None
        Reference round name or None (default). Call validate explicitly to enforce invariants.

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


@dataclass(frozen=True)
class CropWindow:
    """Immutable crop region for subtile extraction (Y/X only; Z kept whole).

    All coordinates are 0-based with exclusive end (Python slice convention).

    Parameters
    ----------
    y_start : int
        Zero-based inclusive Y start.
    y_end : int
        Zero-based exclusive Y end.
    x_start : int
        Zero-based inclusive X start.
    x_end : int
        Zero-based exclusive X end.

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

    Parameters
    ----------
    sqrt_pieces : int
        Positive number of tiles per axis.
    overlap_ratio : float
        Fractional overlap; default 0.1.
    windows : list[CropWindow]
        Computed CropWindow list; default new empty list. Call compute_windows before extraction.

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

        Parameters
        ----------
        height : int
            Image height Y in pixels; tile size is height // sqrt_pieces for BOTH axes.
        width : int
            Image width X in pixels, used for clipping right edges.

        Returns
        -------
        None
            Replaces windows in row-major order. Non-square sizes and remainders
            are not redistributed; check window coverage before using these cases.
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
