"""Type definitions for the dataset/FOV layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Round categories

@dataclass
class RoundState:
    """Tracks which rounds belong to sequencing vs other_rounds categories.

    Invariants:

    - ``reference_round`` must be in ``sequencing_rounds`` or ``other_rounds`` (if set)

    - A round cannot appear in both ``sequencing_rounds`` and ``other_rounds``

    Parameters
    ----------
    sequencing_rounds : list[str]
        Sequencing round names, default new empty list.
    other_rounds : list[str]
        Non-sequencing round names, default new empty list.
    reference_round : str | None
        Reference round name or None (default). Call validate explicitly to enforce invariants.

    """

    sequencing_rounds: list[str] = field(default_factory=list)
    other_rounds: list[str] = field(default_factory=list)
    reference_round: str | None = None

    @property
    def all_rounds(self) -> list[str]:
        """All loaded layers in order (sequencing_rounds first, then other_rounds)."""
        return self.sequencing_rounds + self.other_rounds

    @property
    def moving_rounds(self) -> list[str]:
        """Layers that need registration (all except reference_round)."""
        return [r for r in self.all_rounds if r != self.reference_round]

    def validate(self) -> None:
        """Check invariants. Raises ValueError if violated."""
        if self.reference_round is not None and self.reference_round not in self.all_rounds:
            raise ValueError(f"reference_round '{self.reference_round}' not found in sequencing_rounds or other_rounds")
        if len(set(self.all_rounds)) != len(self.all_rounds) or any(not isinstance(r, str) or not r for r in self.all_rounds):
            raise ValueError("round names must be nonempty and unique")
        overlap = set(self.sequencing_rounds) & set(self.other_rounds)
        if overlap:
            raise ValueError(f"Rounds in both sequencing_rounds and other_rounds: {overlap}")


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
        extension. Partitions each axis independently and covers remainder pixels.

        Parameters
        ----------
        height : int
            Image height Y in pixels; partitions use integer proportional boundaries.
        width : int
            Image width X in pixels, used for clipping right edges.

        Returns
        -------
        None
            Replaces windows in row-major order. Rectangular images and remainder pixels are fully covered.
        """
        n = self.sqrt_pieces
        if isinstance(n, bool) or not isinstance(n, int) or n < 1 or min(height, width) < n:
            raise ValueError("positive dimensions must accommodate sqrt_pieces")
        if not np.isfinite(self.overlap_ratio) or not 0 <= self.overlap_ratio < 1:
            raise ValueError("overlap_ratio must be in [0, 1)")
        tile_y, tile_x = height // n, width // n
        overlap_y = int(tile_y * self.overlap_ratio) // 2
        overlap_x = int(tile_x * self.overlap_ratio) // 2

        self.windows = []
        for row in range(n):
            for col in range(n):
                # Base tile boundaries
                y0 = row * height // n
                y1 = (row + 1) * height // n
                x0 = col * width // n
                x1 = (col + 1) * width // n

                # Extend by overlap (except at outer edges)
                if row > 0:
                    y0 -= overlap_y
                if row < n - 1:
                    y1 += overlap_y
                if col > 0:
                    x0 -= overlap_x
                if col < n - 1:
                    x1 += overlap_x

                # Clamp to image boundary
                y1 = min(y1, height)
                x1 = min(x1, width)

                self.windows.append(CropWindow(y0, y1, x0, x1))
