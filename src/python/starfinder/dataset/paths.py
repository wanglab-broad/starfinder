"""Path helpers for FOV output locations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FOVPaths:
    """Immutable path helper for consistent output locations.

    Parameters
    ----------
    output_root : Path
        Output root Path.
    fov_id : str
        FOV identifier used in filenames. Helpers return paths without creating files.

    """

    output_root: Path
    fov_id: str

    @property
    def ref_merged_tif(self) -> Path:
        """Path to the reference TIFF under images/ref_merged.

        Returns
        -------
        pathlib.Path
            Path under output_root; no filesystem mutation.
        """
        return self.output_root / "images" / "ref_merged" / f"{self.fov_id}.tif"

    @property
    def subtile_dir(self) -> Path:
        """Path to the subtile directory under output/subtile.

        Returns
        -------
        pathlib.Path
            Path under output_root; no filesystem mutation.
        """
        return self.output_root / "output" / "subtile" / self.fov_id

    def rsf_log(self) -> Path:
        """Return log/<fov_id>_rsf.txt without creating it.

        Returns
        -------
        pathlib.Path
            Path under output_root; no filesystem mutation.
        """
        return self.output_root / "log" / f"{self.fov_id}_rsf.txt"

    def gr_log(self) -> Path:
        """Return log/<fov_id>_gr.txt without creating it.

        Returns
        -------
        pathlib.Path
            Path under output_root; no filesystem mutation.
        """
        return self.output_root / "log" / f"{self.fov_id}_gr.txt"

    def signal_csv(self, slot: str) -> Path:
        """Return signal/<fov_id>_<slot>.csv; slot is the caller-provided label.

        Returns
        -------
        pathlib.Path
            Path under output_root; no filesystem mutation.

        Parameters
        ----------
        slot : str
            Caller-provided signal label, usually allSpots or goodSpots.
        """
        return self.output_root / "signal" / f"{self.fov_id}_{slot}.csv"

    def signal_png(self, slot: str) -> Path:
        """Return signal/<fov_id>_<slot>.png; slot is the caller-provided label.

        Returns
        -------
        pathlib.Path
            Path under output_root; no filesystem mutation.

        Parameters
        ----------
        slot : str
            Caller-provided signal label.
        """
        return self.output_root / "signal" / f"{self.fov_id}_{slot}.png"

    def shift_log(self, suffix: str = "") -> Path:
        """Return log/gr_shifts/<fov_id><suffix>.txt; suffix defaults to empty.

        Returns
        -------
        pathlib.Path
            Path under output_root; no filesystem mutation.

        Parameters
        ----------
        suffix : str
            Optional text appended to fov_id; default empty string.
        """
        name = f"{self.fov_id}{suffix}.txt" if suffix else f"{self.fov_id}.txt"
        return self.output_root / "log" / "gr_shifts" / name

    def score_log(self, suffix: str = "") -> Path:
        """Return log/sf_scores/<fov_id><suffix>.txt; suffix defaults to empty.

        Returns
        -------
        pathlib.Path
            Path under output_root; no filesystem mutation.

        Parameters
        ----------
        suffix : str
            Optional text appended to fov_id; default empty string.
        """
        name = f"{self.fov_id}{suffix}.txt" if suffix else f"{self.fov_id}.txt"
        return self.output_root / "log" / "sf_scores" / name
