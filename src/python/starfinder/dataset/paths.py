"""Path helpers for FOV output locations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FOVPaths:
    """Immutable path helper for consistent output locations."""

    output_root: Path
    fov_id: str

    @property
    def ref_merged_tif(self) -> Path:
        return self.output_root / "images" / "ref_merged" / f"{self.fov_id}.tif"

    @property
    def subtile_dir(self) -> Path:
        return self.output_root / "output" / "subtile" / self.fov_id

    def signal_csv(self, slot: str) -> Path:
        return self.output_root / "signal" / f"{self.fov_id}_{slot}.csv"

    def signal_png(self, slot: str) -> Path:
        return self.output_root / "signal" / f"{self.fov_id}_{slot}.png"

    def shift_log(self, suffix: str = "") -> Path:
        name = f"{self.fov_id}{suffix}.txt" if suffix else f"{self.fov_id}.txt"
        return self.output_root / "log" / "gr_shifts" / name

    def score_log(self, suffix: str = "") -> Path:
        name = f"{self.fov_id}{suffix}.txt" if suffix else f"{self.fov_id}.txt"
        return self.output_root / "log" / "sf_scores" / name
