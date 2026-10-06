"""Type definitions for the dataset/FOV layer."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
from numbers import Real
from pathlib import Path

import numpy as np

# The written form of a missing wavelength (records and summaries).
UNAVAILABLE = "unavailable"

# The image name of the reference round's stain files (Dataset.reference_stains); no round may take it.
REFERENCE_STAIN = "reference_stain"

_CHANNEL_KEYS = ("channel", "name", "wavelength")


@dataclass(frozen=True)
class ChannelInfo:
    """One channel of a round: its file pattern, content name and wavelength (MATLAB's channel_order_dict entry).

    Parameters
    ----------
    channel : str
        Nonempty file pattern (``*<channel>.tif``), the channel label of
        ``Dataset.channel_labels``.
    name : str or None
        The content, for example ``"DAPI"`` or ``"seq"``; None (default) when
        not given. Names may repeat within a round.
    wavelength : float or None
        Positive finite wavelength in nanometres; None (default) when not
        known, written as ``"unavailable"`` (:meth:`record`).

    Raises
    ------
    TypeError
        channel is not a string, name is not a string or None, or wavelength is
        not a real number or None.
    ValueError
        An empty channel or name, or a wavelength that is not positive and finite.
    """

    channel: str
    name: str | None = None
    wavelength: float | None = None

    def __post_init__(self):
        if not isinstance(self.channel, str):
            raise TypeError(f"a channel pattern must be a string; got {self.channel!r}")
        if not self.channel:
            raise ValueError("a channel pattern must be nonempty")
        if self.name is not None and not isinstance(self.name, str):
            raise TypeError(f"the name of channel {self.channel!r} must be a string or None; got {self.name!r}")
        if self.name == "":
            raise ValueError(f"the name of channel {self.channel!r} must be nonempty or None")
        if self.wavelength is not None:
            if isinstance(self.wavelength, bool) or not isinstance(self.wavelength, Real):
                raise TypeError(f"the wavelength of channel {self.channel!r} must be a number in nm or None; "
                                f"got {self.wavelength!r}")
            if not math.isfinite(self.wavelength) or self.wavelength <= 0:
                raise ValueError(f"the wavelength of channel {self.channel!r} must be positive and finite; "
                                 f"got {self.wavelength!r}")
            object.__setattr__(self, "wavelength", float(self.wavelength))

    @classmethod
    def from_value(cls, value) -> ChannelInfo:
        """A ChannelInfo from one accepted form: a pattern string, a ChannelInfo or a mapping.

        Parameters
        ----------
        value : str, ChannelInfo or Mapping
            A mapping has ``channel`` and optionally ``name`` and ``wavelength``
            (MATLAB's keys); its wavelength ``"unavailable"`` reads as None.

        Returns
        -------
        ChannelInfo
            The value itself for a ChannelInfo.

        Raises
        ------
        TypeError
            Another type.
        ValueError
            A mapping without channel or with another key, or the field errors
            of ChannelInfo.
        """
        if isinstance(value, ChannelInfo):
            return value
        if isinstance(value, str):
            return cls(value)
        if isinstance(value, Mapping):
            unknown = [key for key in value if key not in _CHANNEL_KEYS]
            if unknown or "channel" not in value:
                raise ValueError(f"a channel mapping has the keys channel, name and wavelength (channel required); "
                                 f"got {sorted(map(str, value))}")
            wavelength = value.get("wavelength")
            return cls(value["channel"], value.get("name"), None if wavelength == UNAVAILABLE else wavelength)
        raise TypeError(f"a channel is a pattern string, a ChannelInfo or a mapping; got {value!r}")

    def record(self) -> dict:
        """The written form: channel, name (None as null) and wavelength (None as ``"unavailable"``)."""
        return {"channel": self.channel, "name": self.name,
                "wavelength": UNAVAILABLE if self.wavelength is None else self.wavelength}


def _channel_infos(values, what) -> tuple[ChannelInfo, ...]:
    """Accepted channel forms of one round as ChannelInfo; the patterns must be unique."""
    if isinstance(values, (str, Mapping, ChannelInfo)):
        raise TypeError(f"{what} must be a sequence of channels, not one channel")
    infos = tuple(ChannelInfo.from_value(value) for value in values)
    patterns = [info.channel for info in infos]
    repeated = sorted({p for p in patterns if patterns.count(p) > 1})
    if repeated:
        raise ValueError(f"{what} repeats the channel patterns {repeated}; patterns must be unique")
    return infos


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
