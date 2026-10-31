"""One task of the workflow run by its command on files the test writes (the destination of W-750's moves).

The tests that drove the old Python rules through their adapters (a stub ``snakemake`` object) now drive the
``starfinder workflow`` commands: a test writes an input root whose raw images are empty files (the plan lists them
by path, size and time and reads no content, design/specs/orchestration.md, "The plan command"), plans its
configuration in the Python shape, writes by hand the declared outputs of the earlier tasks it needs (a FOV's
reference image, its molecule table, its saved segmentation or assignment) with records that say succeeded, and runs
the task by its command (``run``: ``starfinder workflow <command> --manifest M --task T`` in this process, the exit
status returned), or calls the kind's function on the task's ``Run`` (``call``: the API entry the command runs, no
check, log or record) to spy on the arguments it passes, as the adapter tests spied.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import yaml

from starfinder import _layout

SEQUENCING = ("ch00", "ch01", "ch02", "ch03")


def input_root(config):
    return Path(config["root_input_path"]) / config["dataset_id"] / config["sample_id"]


def output_root(config):
    return Path(config["root_output_path"]) / config["dataset_id"] / config["output_id"]


def dataset_keys(work, *, n_rounds=3, output_id="out", **keys):
    """The dataset keys of a configuration in the Python shape whose input root is ``<work>/input/data/sample`` and
    output root ``<work>/output/data/<output_id>``."""
    work = Path(work)
    return {"backend": "python", "root_input_path": str(work / "input"), "dataset_id": "data", "sample_id": "sample",
            "root_output_path": str(work / "output"), "output_id": output_id, "n_rounds": n_rounds,
            "ref_round": "round1", "rotate_angle": 0, "seq_channel_order": list(SEQUENCING), **keys}


def write_inputs(config, fovs_csv, *, codebook="gene,barcode\n", rounds=None):
    """fovs.csv (the given text), the codebook and an empty file of every channel the plan lists, under the
    configuration's input root (round-first): the sequencing rounds' channels, the reference stains in the reference
    round and the additional rounds' channels, for every FOV of fovs.csv. ``rounds`` limits the rounds written."""
    root = input_root(config)
    root.mkdir(parents=True, exist_ok=True)
    _layout.fovs_file(root).write_text(fovs_csv)
    if codebook is not None:
        (root / _layout.CODEBOOK_FILE).write_text(codebook)
    fovs = [line.split(",")[1].strip() for line in fovs_csv.strip().splitlines()[1:]]
    label = lambda entry: entry if isinstance(entry, str) else entry["channel"]  # noqa: E731
    wanted = {f"round{i}": [label(c) for c in config.get("seq_channel_order") or []]
              for i in range(1, config["n_rounds"] + 1)}
    wanted[config["ref_round"]] = wanted.get(config["ref_round"], []) + [label(c) for c in
                                                                          config.get("reference_stains") or []]
    for entry in config.get("additional_round") or []:
        wanted[entry["round_name"]] = [label(c) for c in entry.get("channel_order") or []]
    template = config.get("image_dir_template") or _layout.IMAGE_DIR_TEMPLATE
    for name, labels in wanted.items():
        if rounds is not None and name not in rounds:
            continue
        for fov in fovs:
            folder = _layout.image_dir(root, name, fov, template)
            folder.mkdir(parents=True, exist_ok=True)
            for channel in labels:
                (folder / f"{fov}_{channel}.tif").write_bytes(b"")
    return root


def plan(work, name, config):
    """Write the configuration to ``<work>/configs/<name>.yaml`` and plan it: (manifest path, manifest)."""
    from starfinder.workflow import plan as plan_command
    path = Path(work) / "configs" / f"{name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    manifest = plan_command(path)
    return Path(manifest["roots"]["output"]) / "workflow" / "manifest.json", manifest


def tasks(manifest_path, *kinds):
    """The tasks of the given kinds (every task without a kind), in the manifest's order."""
    listed = json.loads(Path(manifest_path).read_text())["tasks"]
    return [t for t in listed if not kinds or t["kind"] in kinds]


def task(manifest_path, kind, **identity):
    """The one task of ``kind`` whose identity holds ``identity``."""
    (found,) = [t for t in tasks(manifest_path, kind) if all(t["identity"].get(k) == v for k, v in identity.items())]
    return found


def succeed(manifest_path, *kinds):
    """Records that say succeeded for every task of the kinds, whose declared outputs the test writes itself (as a
    task's command would, before the tasks that need them run)."""
    out = Path(json.loads(Path(manifest_path).read_text())["roots"]["output"])
    for t in tasks(manifest_path, *kinds):
        path = _layout.task_record(out, t["id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"id": t["id"], "status": "succeeded"}) + "\n")


def run(manifest_path, task_id, *extra):
    """Run one task by its command in this process; the exit status."""
    from starfinder.__main__ import main
    from starfinder.workflow._shape import COMMANDS
    kind = next(t["kind"] for t in tasks(manifest_path) if t["id"] == task_id)
    return main(["workflow", COMMANDS[kind], "--manifest", str(manifest_path), "--task", task_id, *extra])


def run_kinds(manifest_path, *kinds):
    """Run every task of the kinds by its command, in the manifest's order; each must exit 0."""
    for t in tasks(manifest_path, *kinds):
        status = run(manifest_path, t["id"])
        assert status == 0, (t["id"], status)


def call(manifest_path, task_id):
    """The kind's function of the command on the task's Run (the API entry the command runs, without the command's
    checks, log and records); returns the Run."""
    from starfinder.workflow._commands import FUNCTIONS
    from starfinder.workflow._shape import COMMANDS
    from starfinder.workflow._task import Run
    kind = next(t["kind"] for t in tasks(manifest_path) if t["id"] == task_id)
    current = Run(manifest_path, task_id, COMMANDS[kind])
    FUNCTIONS[kind](current)
    return current


def failure(manifest_path, task_id):
    """The error of a task's failed record: {type, message}."""
    out = Path(json.loads(Path(manifest_path).read_text())["roots"]["output"])
    return json.loads(_layout.failed_record(out, task_id).read_text())["error"]


def validation_error(config):
    """The package's validation of a configuration in the Python shape before it reads any file, as the plan runs it:
    the shape (V1, V2), the Python schema (V9, the package resource ``starfinder/workflow/schema.json``, without
    inserting defaults), the block translators (V5 to V8) in the resolution and the data dependencies (V3). None when
    it passes, else the message
    of the refusal (PlanRefused). The tests that validated a ``backend: python`` configuration against
    ``workflow/schemas/config.schema.yaml`` validate it here (W-750)."""
    from starfinder.workflow._plan import PlanRefused, check_producers, check_shape, resolve, validate_schema
    try:
        check_shape(config)
        validate_schema(config)
        check_producers(resolve(config, {"path": "<test>", "sha256": None})["configuration"])
    except PlanRefused as error:
        return str(error)
    return None


def python_shape(config, rule="rsf_single_fov"):
    """A configuration of the old Python dialect (``rules.<rule>.parameters``) in the Python shape, for the package's
    validation: the dataset keys kept, the rule's parameters as the processing block without the sub-blocks' run flags
    (D4: a present block enables its step), ``maximum_projection`` as processing.reference_image_view.projection,
    a checkpoints mapping's stages as the list processing.checkpoints, the other blocks kept and every other rule left
    out (a Python configuration names no rule). The keys that left the dialect, except those the rule's parameters
    hold, are left out too; a test that asserts their refusal adds them back."""
    from starfinder.workflow._shape import BLOCKS, DATASET_KEYS
    out = {k: copy.deepcopy(v) for k, v in config.items() if k in DATASET_KEYS or k in BLOCKS}
    out["backend"] = "python"
    parameters = copy.deepcopy(((config.get("rules") or {}).get(rule) or {}).get("parameters"))
    if parameters is not None or config.get("maximum_projection"):
        processing = {k: ({n: x for n, x in v.items() if n != "run"} if isinstance(v, dict) else v)
                      for k, v in (parameters or {}).items()}
        checkpoints = processing.get("checkpoints")
        if isinstance(checkpoints, dict):
            processing["checkpoints"] = list(checkpoints.get("stages") or [])
        elif "checkpoints" in processing and checkpoints is None:
            processing["checkpoints"] = []
        if config.get("maximum_projection"):
            view = dict(processing.get("reference_image_view") or {})
            processing["reference_image_view"] = dict(view, projection="max")
        out["processing"] = processing
    return out
