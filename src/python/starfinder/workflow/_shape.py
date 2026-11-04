"""The Python configuration shape of the workflow, its canonical serialization and the view a command reads it through.

The constants of design/specs/orchestration.md: the blocks and the dataset keys of the shape, the keys of the MATLAB
dialect (D7) and those that left the Python dialect (the migration table), the defaults the plan writes out, the task
kinds with their commands, and the blocks each kind reads ("The blocks of a task kind"), which is also what its command
may read (``ConfigurationView``).
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import PurePath

#: The resolved configuration's and the manifest's formats; the task record's format.
RESOLVED_FORMAT = "starfinder.workflow.resolved/1"
MANIFEST_FORMAT = "starfinder.workflow.manifest/1"
TASK_FORMAT = "starfinder.workflow.task/1"
FAILED_FORMAT = "starfinder.workflow.task-failed/1"
VERIFY_FORMAT = "starfinder.workflow.verify/1"

#: The step blocks of the Python shape, in the order of the contract; a present block enables its step (D4).
BLOCKS = ("processing", "morphology", "segmentation", "assignment", "stitching", "assembly", "export", "report")
#: The top-level keys outside the blocks and execution (the dataset keys); every kind may read them.
DATASET_KEYS = ("backend", "root_input_path", "dataset_id", "sample_id", "root_output_path", "output_id",
                "selected_fovs", "n_rounds", "ref_round", "ref_channel", "reference_stains", "seq_channel_order",
                "additional_round", "rotate_angle", "voxel_size_xy", "voxel_size_z", "reference_image",
                "reference_image_import", "image_dir_template", "readout_mode", "codebook", "direct_panel")
#: The two roots, which the resolved-configuration hash and the blocks' hash leave out.
ROOT_KEYS = ("root_input_path", "root_output_path")
EXECUTION = "execution"
TOP_KEYS = (*DATASET_KEYS, *BLOCKS, EXECUTION)
#: The keys of the MATLAB dialect: a configuration without backend that holds one is refused (D7).
MATLAB_KEYS = ("rules", "workflow_mode", "dapi_round", "fov_id_pattern", "n_fovs", "subset_list", "subset_range",
               "subset_start", "subset_end", "subset_random", "n_random_tests", "matlab_launcher",
               "matlab_single_thread")
#: The top-level keys that left the Python dialect, with where each went (the contract's migration table).
LEFT_KEYS = {
    "config_path": "nothing: the plan records the file it read (source of the resolved configuration)",
    "starfinder_path": "nothing: the engines run the installed starfinder command",
    "fiji_path": "execution.fiji",
    "envs_path": "nothing: no rule read it",
    "img_row": "nothing: its readers are retired",
    "img_col": "nothing: its readers are retired",
    "img_z": "nothing: its reader is retired",
    "maximum_projection": "processing.reference_image_view.projection and morphology.channel_images.projection",
    "workflow_mode": "a present block enables its step (processing.mode for direct or tiled)",
    "rules": "the step blocks (processing, morphology, segmentation, assignment, stitching, assembly, export, "
             "report) and execution.tasks",
    "matlab_launcher": "nothing: the MATLAB dialect only (backend: matlab)",
    "matlab_single_thread": "nothing: the MATLAB dialect only (backend: matlab)",
    "dapi_round": "reference_stains: [{channel, name}]",
    "channel_order": "seq_channel_order",
    "fov_pattern": "fovs.csv and selected_fovs",
    "fov_id_pattern": "fovs.csv and selected_fovs",
    "n_fovs": "fovs.csv and selected_fovs",
    "subset_list": "selected_fovs",
    "subset_range": "selected_fovs",
    "subset_start": "selected_fovs",
    "subset_end": "selected_fovs",
    "subset_random": "selected_fovs",
    "n_random_tests": "selected_fovs",
}
#: The processing block's keys (today's processing rule parameters without the run flags, D4) and the new keys.
PROCESSING_KEYS = ("mode", "recipe", "tiled", "checkpoints", "streaming", "device", "load_codebook", "load_raw_images",
                   "preprocessing", "registration", "reference_image_view", "spot_finding", "reads_extraction",
                   "reads_filtration", "decoding", "scoring", "deduplication")
#: The keys of the processing block's sub-blocks that left with the run flags or the subtile route.
PROCESSING_LEFT = {"create_subtiles": "processing.mode: tiled and processing.tiled: the legacy subtile route is "
                                      "retired for Python (D5)"}
#: The defaults of processing.tiled (design/specs/orchestration-tiled.md, "Declared files of the tiled task kinds");
#: the plan writes the halo it computes in place of null.
TILED_DEFAULTS = {"tile_size_yx": 512, "halo": None, "local_registration": None, "post_local_registration": None,
                  "exact": True}
#: The named recipes of processing.recipe, which the plan expands into the resolved configuration (a key the user
#: also sets wins); ``tiled`` applies in mode tiled only (design/specs/orchestration-tiled.md, "The deep-tissue
#: recipe").
RECIPES = {"deep": {
    "preprocessing": {"steps": [
        {"method": "min_max_normalization", "output_dtype": "uint8", "output_range": [0, 255], "rounding": "truncate"},
        {"method": "histogram_matching", "reference_channel": 0}],
        "post_registration": [{"method": "reconstruction", "radius_yx": 3}]},
    "registration": {"steps": [{"method": "translation"}], "signal": {"mode": "max"}},
    "tiled": {"local_registration": {"steps": [{"method": "translation"}], "signal": {"mode": "max"}}}}}
#: The default checkpoint stages of the fov task (D8) and every stage in order.
CHECKPOINT_STAGES = ("candidates", "pre_qc")
STAGES = ("registered", "candidates", "pre_qc")
#: The execution defaults (today's DEFAULT_RESOURCES and rule threads, common.smk:156-159 and the rule files).
EXECUTION_DEFAULTS = {"threads": 1, "memory_mb": 8000, "runtime_min": 30, "gpu": False}
KIND_THREADS = {"fov": 4, "tile_global": 4, "tile": 4, "bigstitcher_engine": 4}
EXECUTION_KEYS = ("defaults", "tasks", "container", "fiji", "resume_from_checkpoint", "code_version", "log_level")
RESOURCE_KEYS = ("threads", "memory_mb", "runtime_min", "gpu")
CODE_VERSIONS = ("keep", "rerun")
LOG_LEVELS = ("INFO", "DEBUG")

#: The task kinds (design/specs/orchestration.md, "Terms") and the blocks each reads besides the common part.
TASK_BLOCKS = {
    "summarize": ("processing",), "merge": ("processing",), "fov": ("processing",),
    "tile_global": ("processing",), "tile": ("processing",), "tile_merge": ("processing",),
    "morphology": ("morphology",), "segment": ("segmentation",), "assign": ("assignment",),
    "bigstitcher_macro": ("stitching",), "bigstitcher_engine": ("stitching",), "bigstitcher_import": ("stitching",),
    "stitch": ("stitching",), "export": ("assembly", "export", "assignment", "stitching", "processing"),
    "report": ("report", "assembly", "assignment"), "compare": ("report",),
}
KINDS = tuple(TASK_BLOCKS)
#: The command of each kind: the kind with ``-``; the Fiji task's record is written by bigstitcher-check.
COMMANDS = {kind: kind.replace("_", "-") for kind in KINDS} | {"bigstitcher_engine": "bigstitcher-check"}
#: The identity fields of a task id, in the id's order.
ID_FIELDS = ("sample", "step", "fov", "tile", "run", "name")


# --- the canonical serialization -------------------------------------------------------------------------------

def canonical(value) -> bytes:
    """The contract's canonical serialization: JSON, UTF-8, keys sorted, no insignificant whitespace, floats in
    Python's shortest round-trip form, NaN and infinities refused."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def sha256(data: bytes) -> str:
    """``sha256:<hex>`` of bytes, the form of every hash the plan and the records write."""
    return "sha256:" + hashlib.sha256(data).hexdigest()


def plain(value):
    """A value as the resolved configuration writes it: a dataclass field by field with its type name (``type``),
    tuples as lists, paths as strings, enums by value; NaN and infinities refused."""
    if is_dataclass(value) and not isinstance(value, type):
        return {"type": type(value).__name__, **{f.name: plain(getattr(value, f.name)) for f in fields(value)}}
    if isinstance(value, Mapping):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, PurePath):
        return str(value)
    if isinstance(value, Enum):
        return plain(value.value)
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"the resolved configuration refuses {value!r}")
    if hasattr(value, "item") and callable(value.item) and type(value).__module__ == "numpy":
        return plain(value.item())
    return value


def common_part(configuration) -> dict:
    """The common part of a configuration: every top-level key outside the blocks, execution and the two roots."""
    return {k: v for k, v in configuration.items() if k not in BLOCKS and k not in ROOT_KEYS and k != EXECUTION
            and k != "api"}


def blocks_sha256(configuration, kind) -> str:
    """The hash of the common part and the blocks ``kind`` reads, an absent block written as null."""
    part = {"common": common_part(configuration),
            "blocks": {block: configuration.get(block) for block in TASK_BLOCKS[kind]}}
    return sha256(canonical(part))


def configuration_sha256(configuration) -> str:
    """The resolved-configuration hash: the configuration without execution and the two roots."""
    return sha256(canonical({k: v for k, v in configuration.items() if k != EXECUTION and k not in ROOT_KEYS}))


# --- the view of a command -------------------------------------------------------------------------------------

class BlockAccessError(RuntimeError):
    """A command read a block outside its kind's map (design/specs/orchestration.md, "The blocks of a task kind")."""


class ConfigurationView(Mapping):
    """The resolved configuration (or its ``api``) as a command of ``kind`` may read it: the common part, the
    execution section, the two roots and the blocks of ``TASK_BLOCKS[kind]``.

    Reading another block raises BlockAccessError, also through ``in`` and ``get``, in every run: a block missing
    from the map fails the command that reads it the first time it runs.
    """

    def __init__(self, data, kind, what="configuration"):
        self._data, self._kind, self._what = dict(data), kind, what
        self._blocks = TASK_BLOCKS[kind]

    def _allowed(self, key):
        return key not in BLOCKS or key in self._blocks

    def __getitem__(self, key):
        if not self._allowed(key):
            raise BlockAccessError(f"the {self._kind} command read the {key} block of the {self._what}; a {self._kind} "
                                   f"task reads the common part and {', '.join(self._blocks)} only")
        return self._data[key]

    def __contains__(self, key):
        if not self._allowed(key):
            self[key]
        return key in self._data

    def get(self, key, default=None):
        return self[key] if key in self else default

    def __iter__(self):
        return (key for key in self._data if self._allowed(key))

    def __len__(self):
        return sum(1 for _ in self)

    def plain(self) -> dict:
        """The readable keys as a plain dict, for the translators."""
        return {key: self._data[key] for key in self}
