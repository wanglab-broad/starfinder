"""The layout of a dataset's output root: the folders and record files the package reads and writes.

Private and unexported; it imports nothing from ``starfinder``. Every helper
returns a path without touching the filesystem. The MATLAB-era folders
(``images``, ``signal``, ``log``, ``output``, ``expr``, ``documents``) keep
their names, and the ``.smk`` rules of the MATLAB backend spell the same paths
on their own. A FOV's own folder ``fovs/<fov_id>/`` holds its durable results: the run record, the
tables ``FOV.export_spots`` writes, the registration record of its other
rounds, its reference image and the folders ``other_rounds/``,
``segmentation/`` and ``assignment/``; the restart material of ``FOV.run``
goes in its ``checkpoints/`` folder (design/specs/checkpoints.md, "Layout"). The
workflow's opt-in assignment previews (``assembly.previews``) go in the FOV
folder's ``previews/`` and the export folder's ``previews/<fov_id>/``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

#: The folders under the output root: the FOV folders and the sample-stage roots.
FOVS = "fovs"
STITCHING = "stitching"
EXPORT = "export"
REPORT = "report"
#: Inside a FOV's folder: the restart folder of FOV.run's checkpoint stages. It is also the key under which
#: run.json lists the stage files and the configuration names the stages.
CHECKPOINTS = "checkpoints"
#: Inside a FOV's folder: the prepared images of other rounds, the saved segmentation runs and assignments.
OTHER_ROUNDS = "other_rounds"
SEGMENTATION = "segmentation"
ASSIGNMENT = "assignment"
# Tiled execution (design/specs/orchestration-tiled.md, "Declared files of the tiled task kinds"): the global task's
# folder and files, and the tiles' folder of each FOV.
TILED = "tiled"
TILES = "tiles"
CHUNKED_CHECKPOINT = "registered.h5"
GLOBAL_RECORD = "global.json"
TILE_FILES = ("candidates.csv", "reads.csv", "tile.json")
#: The record files: a FOV's run record in its folder, a sample's export record in its export folder.
RUN_RECORD = "run.json"
EXPORT_FILE = "export.json"
#: Inside a FOV's checkpoint folder: the registration folder and its header; an assignment's record.
REGISTERED = "registered"
TRANSFORMS_FILE = "transforms.json"
ASSIGNMENT_RECORD = "assignment.json"
#: The tables FOV.export_spots writes in a FOV's folder, by slot: the accepted molecules and, on request, the reads.
SPOT_TABLES = {"molecules": "molecules.csv", "reads": "reads.csv"}
#: The registration record of a FOV's other rounds, which FOV.save_registration_record writes in the FOV's folder.
NUCLEI_REGISTRATION = "nuclei_registration.json"
#: The workflow's opt-in assignment previews (``assembly.previews``): the folder of a FOV's local figures in its
#: folder, and of the ownership figures in a sample's export folder (``previews/<fov_id>``), which export.json
#: does not list.
PREVIEWS = "previews"
#: The settings naming a FOV's reference image (workflow YAML keys and ``Dataset`` fields), each a file pattern
#: with ``{fov_id}``, absolute or relative to the output root: ``reference_image``, the file the processing
#: writes (default the FOV folder's ``reference.tif``), or ``reference_image_import``, a file another tool or
#: rule produced, which the processing never writes and every reader reads instead. The two exclude each other.
REFERENCE_IMAGE_KEY = "reference_image"
REFERENCE_IMAGE_IMPORT_KEY = "reference_image_import"
REFERENCE_IMAGE = "fovs/{fov_id}/reference.tif"


def check_reference_image(value, key=REFERENCE_IMAGE_KEY) -> str:
    """The pattern of setting ``key``; ValueError naming the setting for anything but a string with ``{fov_id}``."""
    if not isinstance(value, str) or "{fov_id}" not in value:
        default = f" (default {REFERENCE_IMAGE!r})" if key == REFERENCE_IMAGE_KEY else ""
        raise ValueError(f"{key} must be a file pattern with {{fov_id}}, absolute or relative to the output "
                         f"root{default}; got {value!r}")
    return value


def check_reference_import(reference_image, imported):
    """The ``reference_image_import`` pattern, or None; ValueError when it is given beside another reference_image.

    An imported reference image replaces the one the processing writes, so a
    ``reference_image`` other than the default cannot be given with it.
    """
    if imported is None:
        return None
    check_reference_image(imported, REFERENCE_IMAGE_IMPORT_KEY)
    if reference_image != REFERENCE_IMAGE:
        raise ValueError(f"{REFERENCE_IMAGE_IMPORT_KEY} ({imported!r}) excludes {REFERENCE_IMAGE_KEY} "
                         f"({reference_image!r}): an imported reference image is read where it is and never written")
    return imported


def reference_image_path(output_root, fov_id, pattern=REFERENCE_IMAGE, imported=None) -> Path:
    """The file of a FOV's reference image, the one resolver for its writer and every reader.

    ``imported`` (the ``reference_image_import`` pattern) when it is given, else
    ``pattern`` (``reference_image``), with ``{fov_id}`` replaced; a relative
    pattern is joined to ``output_root``.
    """
    if imported is not None:
        check_reference_image(imported, REFERENCE_IMAGE_IMPORT_KEY)
    path = Path((check_reference_image(pattern) if imported is None else imported).replace("{fov_id}", fov_id))
    return path if path.is_absolute() else Path(output_root) / path


def fov_dir(output_root, fov_id, *, parent=None) -> Path:
    """``fovs/<fov_id>``, the folder of a FOV's durable results.

    ``parent`` (``CheckpointConfig.directory``) replaces ``<output_root>/fovs``
    as the parent of the FOV folders when given.
    """
    return (Path(output_root) / FOVS if parent is None else Path(parent)) / fov_id


def checkpoint_dir(fov_directory) -> Path:
    """``<fov folder>/checkpoints``, the restart folder of a FOV's checkpoint stages."""
    return Path(fov_directory) / CHECKPOINTS


def run_record(output_root, fov_id) -> Path:
    """``fovs/<fov_id>/run.json``, the run record of a FOV."""
    return fov_dir(output_root, fov_id) / RUN_RECORD


def tiled_dir(fov_directory) -> Path:
    """``<fov folder>/tiled``: the global task's files of tiled execution (the chunked checkpoint, its records)."""
    return Path(fov_directory) / TILED


def tile_dir(fov_directory, tile) -> Path:
    """``<fov folder>/tiles/<tile>``: one tile task's files of tiled execution (``tile`` is ``r<row>c<col>``)."""
    return Path(fov_directory) / TILES / tile


def other_round_dir(fov_directory, name) -> Path:
    """``<fov folder>/other_rounds/<name>``, the saved form of a prepared image."""
    return Path(fov_directory) / OTHER_ROUNDS / name


def segmentation_dir(fov_directory, run) -> Path:
    """``<fov folder>/segmentation/<run>``, a saved segmentation run."""
    return Path(fov_directory) / SEGMENTATION / run


def assignment_dir(fov_directory, name) -> Path:
    """``<fov folder>/assignment/<name>``, a saved assignment."""
    return Path(fov_directory) / ASSIGNMENT / name


def spot_table(output_root, fov_id, slot="molecules") -> Path:
    """``fovs/<fov_id>/molecules.csv`` or ``reads.csv``, the table of a ``FOV.export_spots`` slot."""
    if slot not in SPOT_TABLES:
        raise ValueError(f"slot must be one of {list(SPOT_TABLES)}; got {slot!r}")
    return fov_dir(output_root, fov_id) / SPOT_TABLES[slot]


def nuclei_registration_record(output_root, fov_id) -> Path:
    """``fovs/<fov_id>/nuclei_registration.json``, the registration record of a FOV's other rounds."""
    return fov_dir(output_root, fov_id) / NUCLEI_REGISTRATION


def stitching_dir(output_root, sample) -> Path:
    """``stitching/<sample>``, the placement folder of a sample."""
    return Path(output_root) / STITCHING / sample


def export_dir(output_root, sample) -> Path:
    """``export/<sample>``, the export folder of a sample."""
    return Path(output_root) / EXPORT / sample


def fov_previews_dir(output_root, fov_id) -> Path:
    """``fovs/<fov_id>/previews``, the folder of a FOV's local assignment previews."""
    return fov_dir(output_root, fov_id) / PREVIEWS


def export_previews_dir(output_root, sample, fov_id=None) -> Path:
    """``export/<sample>/previews`` (a FOV's: ``export/<sample>/previews/<fov_id>``), the ownership previews."""
    folder = export_dir(output_root, sample) / PREVIEWS
    return folder if fov_id is None else folder / fov_id


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


def fovs_file(input_root) -> Path:
    """``fovs.csv`` under the input root, the Python workflow's FOV membership (``sample,fov_id``)."""
    return Path(input_root) / "fovs.csv"


#: The workflow's own folder under the output root (design/specs/orchestration.md, "Where the workflow files live"):
#: the plan's resolved configuration and manifest, one record and one log per task, and verify's record.
WORKFLOW = "workflow"
RESOLVED_FILE = "resolved.json"
MANIFEST_FILE = "manifest.json"
VERIFY_FILE = "verify.json"
TASKS = "tasks"
LOGS = "logs"
#: The two-pass statistics of a sample and a supplied step: one histogram file per FOV and the merged file.
STATISTICS = "statistics"
SUPPLIED_FILE = "supplied.json"


def workflow_dir(output_root) -> Path:
    """``workflow``, the folder of the plan's files, the task records and the task logs."""
    return Path(output_root) / WORKFLOW


def workflow_export_record(output_root, sample) -> Path:
    """``workflow/exports/<sample>.json``, the export record kept across engine output removal."""
    return workflow_dir(output_root) / "exports" / f"{sample}.json"


def resolved_file(output_root) -> Path:
    """``workflow/resolved.json``, the resolved configuration the plan writes."""
    return workflow_dir(output_root) / RESOLVED_FILE


def manifest_file(output_root) -> Path:
    """``workflow/manifest.json``, the manifest of every task of a run."""
    return workflow_dir(output_root) / MANIFEST_FILE


def task_name(task_id) -> str:
    """A task id as a file name: ``/`` written ``.`` (``fov/sample/FOV_001`` is ``fov.sample.FOV_001``)."""
    return str(task_id).replace("/", ".")


def task_record(output_root, task_id) -> Path:
    """``workflow/tasks/<task>.json``, the task record, written last by a task that succeeded."""
    return workflow_dir(output_root) / TASKS / f"{task_name(task_id)}.json"


def failed_record(output_root, task_id) -> Path:
    """``workflow/tasks/<task>.failed.json``, the record of a failed attempt, kept for diagnosis."""
    return workflow_dir(output_root) / TASKS / f"{task_name(task_id)}.failed.json"


def task_log(output_root, task_id) -> Path:
    """``workflow/logs/<task>.log``, the log of a task, rewritten by each attempt."""
    return workflow_dir(output_root) / LOGS / f"{task_name(task_id)}.log"


def statistics_dir(output_root, sample, step) -> Path:
    """``statistics/<sample>/<step>``, the histograms and the supplied file of a sample's supplied step."""
    return Path(output_root) / STATISTICS / sample / step


#: The layouts of the acquisition images under the input root (``image_dir_template``, the workflow key and the
#: ``Dataset`` field): the folder of one round of one FOV, round-first (the default, the layout MATLAB reads) or
#: FOV-first. The channel files in that folder are found as before (``io.load_round``).
IMAGE_DIR_TEMPLATES = ("{round}/{fov}", "{fov}/{round}")
IMAGE_DIR_TEMPLATE = IMAGE_DIR_TEMPLATES[0]
#: The codebook (or direct panel) under the input root when the configuration names no other file.
CODEBOOK_FILE = "genes.csv"


def check_image_dir_template(value) -> str:
    """The template ``value``; ValueError naming the two layouts for anything else."""
    if value not in IMAGE_DIR_TEMPLATES:
        raise ValueError(f"image_dir_template must be one of {', '.join(map(repr, IMAGE_DIR_TEMPLATES))} "
                         f"(default {IMAGE_DIR_TEMPLATE!r}); got {value!r}")
    return value


def image_dir(input_root, round_name, fov_id, template=IMAGE_DIR_TEMPLATE) -> Path:
    """``<round>/<fov_id>`` or ``<fov_id>/<round>`` under the input root, as ``template`` lays out the images."""
    return Path(input_root) / check_image_dir_template(template).format(round=round_name, fov=fov_id)


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
    def fov_dir(self) -> Path:
        """The FOV's folder under fovs/, which holds its durable results.

        Returns
        -------
        pathlib.Path
            Path under output_root; no filesystem mutation.
        """
        return fov_dir(self.output_root, self.fov_id)

    @property
    def checkpoint_dir(self) -> Path:
        """Default checkpoint folder of the FOV: fovs/<fov_id>/checkpoints.

        Returns
        -------
        pathlib.Path
            Path under output_root; no filesystem mutation.
        """
        return checkpoint_dir(self.fov_dir)

    @property
    def run_record(self) -> Path:
        """Default run record of the FOV: fovs/<fov_id>/run.json.

        Returns
        -------
        pathlib.Path
            Path under output_root; no filesystem mutation.
        """
        return run_record(self.output_root, self.fov_id)

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
