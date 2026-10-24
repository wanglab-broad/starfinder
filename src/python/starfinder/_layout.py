"""The layout of a dataset's output root: the folders and record files the package reads and writes.

Private and unexported; it imports nothing from ``starfinder``. Every helper
returns a path without touching the filesystem. The MATLAB-era folders
(``images``, ``signal``, ``log``, ``output``, ``expr``, ``documents``) keep
their names, and the ``.smk`` rules spell the same paths on their own. A FOV's
own folder ``fovs/<fov_id>/`` holds the tables ``FOV.export_spots`` writes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

#: The folders under the output root.
CHECKPOINTS = "checkpoints"
FOVS = "fovs"
STITCHING = "stitching"
EXPORT = "export"
REPORT = "report"
#: The record files: a FOV's run record in its checkpoint folder, a sample's export record in its export folder.
RUN_RECORD = "run.json"
EXPORT_FILE = "export.json"
#: Inside a FOV's checkpoint folder: the registration folder and its header, and an assignment's record.
REGISTERED = "registered"
TRANSFORMS_FILE = "transforms.json"
ASSIGNMENT_RECORD = "assignment.json"
#: The tables FOV.export_spots writes in a FOV's folder, by slot: the accepted molecules and, on request, the reads.
SPOT_TABLES = {"molecules": "molecules.csv", "reads": "reads.csv"}
#: The setting naming a FOV's reference image (workflow YAML key and ``Dataset`` field) and its default: a file
#: pattern with ``{fov_id}``, absolute or relative to the output root.
REFERENCE_IMAGE_KEY = "reference_image"
REFERENCE_IMAGE = "images/ref_merged/{fov_id}.tif"


def check_reference_image(value) -> str:
    """The ``reference_image`` pattern; ValueError naming the setting for anything but a string with ``{fov_id}``."""
    if not isinstance(value, str) or "{fov_id}" not in value:
        raise ValueError(f"{REFERENCE_IMAGE_KEY} must be a file pattern with {{fov_id}}, absolute or relative to the "
                         f"output root (default {REFERENCE_IMAGE!r}); got {value!r}")
    return value


def reference_image_path(output_root, fov_id, pattern=REFERENCE_IMAGE) -> Path:
    """The file of a FOV's reference image: ``pattern`` with ``{fov_id}`` replaced, relative to ``output_root``."""
    path = Path(check_reference_image(pattern).replace("{fov_id}", fov_id))
    return path if path.is_absolute() else Path(output_root) / path


def checkpoint_dir(output_root, fov_id) -> Path:
    """``checkpoints/<fov_id>``, the default checkpoint folder of a FOV."""
    return Path(output_root) / CHECKPOINTS / fov_id


def run_record(output_root, fov_id) -> Path:
    """``checkpoints/<fov_id>/run.json``, the run record of a FOV."""
    return checkpoint_dir(output_root, fov_id) / RUN_RECORD


def fov_dir(output_root, fov_id) -> Path:
    """``fovs/<fov_id>``, the folder of a FOV's results."""
    return Path(output_root) / FOVS / fov_id


def spot_table(output_root, fov_id, slot="molecules") -> Path:
    """``fovs/<fov_id>/molecules.csv`` or ``reads.csv``, the table of a ``FOV.export_spots`` slot."""
    if slot not in SPOT_TABLES:
        raise ValueError(f"slot must be one of {list(SPOT_TABLES)}; got {slot!r}")
    return fov_dir(output_root, fov_id) / SPOT_TABLES[slot]


def stitching_dir(output_root, sample) -> Path:
    """``stitching/<sample>``, the placement folder of a sample."""
    return Path(output_root) / STITCHING / sample


def export_dir(output_root, sample) -> Path:
    """``export/<sample>``, the export folder of a sample."""
    return Path(output_root) / EXPORT / sample


def report_root(output_root) -> Path:
    """``report``, the report root of a dataset (the sample reports and the comparison page)."""
    return Path(output_root) / REPORT


def report_dir(output_root, sample) -> Path:
    """``report/<sample>``, the report folder of a sample."""
    return report_root(output_root) / sample


def image_file(output_root, fov_id, *folders) -> Path:
    """``images/<folders...>/<fov_id>.tif``, a FOV's image under the MATLAB-era images folder."""
    return Path(output_root, "images", *folders, f"{fov_id}.tif")


def expr_dir(output_root, fov_id) -> Path:
    """``expr/<fov_id>``, the folder of the workflow's per-FOV assignment outputs."""
    return Path(output_root) / "expr" / fov_id


def tile_config(output_root, sample) -> Path:
    """``output/tile_config_<sample>.csv``, the MATLAB-era tile configuration of a sample."""
    return Path(output_root) / "output" / f"tile_config_{sample}.csv"


def stage_file(output_root, sample) -> Path:
    """``documents/maf/<sample>.maf``, the default stage file of a sample."""
    return Path(output_root) / "documents" / "maf" / f"{sample}.maf"


def sample_annotation(output_root) -> Path:
    """``documents/sample-annotation.csv``, the FOV list of each sample."""
    return Path(output_root) / "documents" / "sample-annotation.csv"


@dataclass(frozen=True)
class _FovPaths:
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

    @property
    def checkpoint_dir(self) -> Path:
        """Default per-FOV checkpoint directory under checkpoints/.

        Returns
        -------
        pathlib.Path
            Path under output_root; no filesystem mutation.
        """
        return self.output_root / CHECKPOINTS / self.fov_id

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

    def nr_log(self) -> Path:
        """Return log/<fov_id>_nr.txt (nuclei_registration) without creating it.

        Returns
        -------
        pathlib.Path
            Path under output_root; no filesystem mutation.
        """
        return self.output_root / "log" / f"{self.fov_id}_nr.txt"

    def spot_table(self, slot: str = "molecules") -> Path:
        """Return fovs/<fov_id>/<slot>.csv, the table FOV.export_spots writes for slot molecules or reads.

        Returns
        -------
        pathlib.Path
            Path under output_root; no filesystem mutation.

        Parameters
        ----------
        slot : str
            ``molecules`` (default) or ``reads``.
        """
        return spot_table(self.output_root, self.fov_id, slot)

    def signal_csv(self, slot: str) -> Path:
        """Return signal/<fov_id>_<slot>.csv; slot is the caller-provided label.

        Returns
        -------
        pathlib.Path
            Path under output_root; no filesystem mutation.

        Parameters
        ----------
        slot : str
            Caller-provided signal label: goodSpots, the legacy route's one-based molecule table
            (MATLAB's rsf_single_fov and stitch_subtile).
        """
        return self.output_root / "signal" / f"{self.fov_id}_{slot}.csv"

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
