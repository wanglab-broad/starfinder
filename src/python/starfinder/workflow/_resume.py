"""Verified resume of the fov task (D8; design/specs/orchestration.md, "Checkpoints and resume").

With ``execution.resume_from_checkpoint`` the fov command resumes from the latest stage an earlier attempt wrote,
only when the earlier ``run.json`` matches the task: its version, identity, ``workflow.blocks_sha256``, code
identity, inputs, codebook, supplied statistics and stage records. A mismatch refuses with exit 3 and a message that
names every field that differs. The check reads ``run.json``, never the task record or the log.

With ``processing.spot_finding_input`` the fov task reads the registered stage of a saved set instead
(``prepare_input``): the set's ``run.json`` must match the task's preprocessing and registration (``check_input``), not
its package version; a mismatch refuses with exit 2.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from starfinder import _checkpoint, _layout
from starfinder._records import file_sha256
from starfinder.dataset._run_record import resolve_source
from starfinder.workflow._task import ResumeRefused, TaskRefused

#: A version of a build with uncommitted changes (hatch-vcs adds a local part ``.d<date>``).
_DIRTY_VERSION = re.compile(r"\+.*\bd\d{8}\b|\.d\d{8}")


def _code_problems(code, side):
    """Why a code identity cannot be resumed: a dirty tree, a dirty build's version, or no version."""
    problems = []
    if not isinstance(code, dict) or not code.get("version"):
        problems.append(f"code.version cannot be established ({side})")
        return problems
    if code.get("git_dirty"):
        problems.append(f"code.git_dirty is true ({side})")
    if _DIRTY_VERSION.search(str(code["version"])):
        problems.append(f"code.version {code['version']} is a build with uncommitted changes ({side})")
    return problems


def _short(value):
    text = str(value)
    return text if len(text) <= 24 else text[:24] + "…"


def check(run, record, now_code):
    """The fields of the earlier run.json ``record`` that differ from the task ``run``; [] when it may resume."""
    differences = []
    if record.get("format_version") != 2:
        return [f"format_version is {record.get('format_version')!r} (recorded), 2 (now): a version-1 record is "
                "never resumed"]
    configuration = run.configuration
    expected = {"dataset_id": configuration["dataset_id"], "sample_id": configuration["sample_id"],
                "fov_id": run.identity["fov"], "subtile_id": None}
    for key, value in expected.items():
        if record.get(key) != value:
            differences.append(f"{key} is {record.get(key)!r} (recorded), {value!r} (now)")
    workflow = record.get("workflow") or {}
    if workflow.get("blocks_sha256") != run.task["blocks_sha256"]:
        differences.append(f"workflow.blocks_sha256 is {_short(workflow.get('blocks_sha256'))} (recorded), "
                           f"{_short(run.task['blocks_sha256'])} (now)")
    code = record.get("code") or {}
    differences += _code_problems(code, "recorded") + _code_problems(now_code, "now")
    if code.get("version") and now_code.get("version") and code["version"] != now_code["version"]:
        differences.append(f"code.version is {code['version']} (recorded), {now_code['version']} (now)")
    # "(when recorded)": a recorded commit must equal the running package's, whether or not that package has one.
    if code.get("git_commit") and code["git_commit"] != now_code.get("git_commit"):
        differences.append(f"code.git_commit is {_short(code['git_commit'])} (recorded), "
                           f"{_short(now_code.get('git_commit'))} (now)")
    root = run.root
    declared = {i[len("input:"):] for i in run.task["inputs"] if i.startswith("input:")}
    recorded = {entry["path"] for entry in record.get("inputs") or []}
    images = {d for d in declared if any(d == c["path"] for rounds in run.manifest["inputs"]["images"].values()
                                          for labels in rounds.values() for c in labels.values())}
    # The deconvolved copy's files the task reads, as run.json names them (relative to the input root, else absolute).
    images |= copy_inputs(run)
    for path in sorted(images ^ recorded):
        differences.append(f"inputs {path} is {'not ' if path not in recorded else ''}recorded"
                           f"{'' if path in images else ' and no input of this task'}")
    for entry in record.get("inputs") or []:
        path = resolve_source(record, entry["path"], root)
        if entry["path"] in images and (path is None or not path.is_file() or entry.get("sha256") is None
                                        or file_sha256(path) != entry["sha256"]):
            differences.append(f"inputs {entry['path']} sha256 differs")
    # The codebook and the supplied statistics this task reads, from its declared inputs; each row requires equality,
    # so a file present on one side only differs.
    from starfinder.workflow._commands import codebook_input, supplied_input
    codebook = record.get("codebook") or {}
    differences += _file_differences("codebook", codebook.get("path"), codebook.get("sha256"), codebook_input(run),
                                     record, root)
    supplied = (record.get("preprocessing") or {}).get("supplied_statistics") or {}
    differences += _file_differences("preprocessing.supplied_statistics", supplied.get("path"), supplied.get("sha256"),
                                     supplied_input(run), record, root)
    folder = _layout.fov_dir(run.out, run.identity["fov"])
    for stage, entry in (record.get(_layout.CHECKPOINTS) or {}).items():
        path = folder / entry["path"]
        if not path.is_file() or entry.get("sha256") is None or file_sha256(path) != entry["sha256"]:
            differences.append(f"checkpoints.{stage} {entry['path']} sha256 differs")
            continue
        header = json.loads(path.read_text())
        for item in header.get("files") or []:
            payload = path.parent / item["path"]
            if not payload.is_file() or file_sha256(payload) != item.get("sha256"):
                differences.append(f"checkpoints.{stage} file {item['path']} sha256 differs")
    return differences


def copy_inputs(run):
    """The deconvolved copy's image files a task declares, as run.json names its inputs (``source_path``)."""
    from starfinder.dataset._run_record import source_path
    produced = {o for t in run.manifest["tasks"] if t["kind"] == "deconvolve" for o in t["outputs"]}
    return {str(source_path(run.path(i), run.root)) for i in run.task["inputs"] if i in produced and i.endswith(".tif")}


def _file_differences(field, recorded, digest, now, record, root):
    """How a file the record names (``recorded`` with its ``digest``) differs from the file the task reads now."""
    if recorded is None and now is None:
        return []
    if recorded is None:
        return [f"{field} is not recorded (recorded), {now} (now)"]
    if now is None:
        return [f"{field} is {recorded} (recorded), none (now)"]
    path = resolve_source(record, recorded, root)
    if path is None or Path(path).absolute() != Path(now).absolute():
        return [f"{field}.path is {recorded} (recorded), {now} (now)"]
    if not Path(now).is_file() or digest != file_sha256(now):
        return [f"{field} {recorded} sha256 differs"]
    return []


def latest_stage(record):
    """The latest stage the earlier attempt wrote (registered, candidates, pre_qc), or None."""
    written = [stage for stage in _checkpoint.STAGES if stage in (record.get(_layout.CHECKPOINTS) or {})]
    return written[-1] if written else None


def kept_outputs(run, stage, record):
    """The declared outputs a resume after ``stage`` keeps: run.json, the stage records up to it and the payload files
    each lists (the registered images of a segmentation input among them)."""
    folder = _layout.fov_dir(run.out, run.identity["fov"])
    kept = {_layout.run_record(run.out, run.identity["fov"])}
    for name in _checkpoint.STAGES[:_checkpoint.STAGES.index(stage) + 1]:
        entry = (record.get(_layout.CHECKPOINTS) or {}).get(name)
        if entry is None:
            continue
        header = folder / entry["path"]
        kept.add(header)
        kept |= {header.parent / item["path"] for item in json.loads(header.read_text()).get("files") or []}
    return [declared for declared, path in run.outputs() if path in kept]


def prepare(run):
    """The fov command's verified resume: (stage, earlier record) to resume from, or None for a run from scratch.

    Raises ResumeRefused (exit 3) naming every field that differs. Without resume_from_checkpoint, or without an
    earlier run.json, or with a record that holds no stage (an attempt stopped before its first stage), the task runs
    from scratch: the stage is found before the record is checked.
    """
    from starfinder.workflow._plan import package_identity
    if not run.configuration["execution"].get("resume_from_checkpoint"):
        return None
    path = _layout.run_record(run.out, run.identity["fov"])
    if not path.is_file():
        return None
    try:
        record = json.loads(path.read_text())
    except ValueError:
        raise ResumeRefused(f"{path.relative_to(run.out)} is not readable JSON; rerun without "
                            "resume_from_checkpoint") from None
    stage = latest_stage(record) if isinstance(record, dict) else None
    if stage is None:
        return None
    differences = check(run, record, package_identity())
    if differences:
        raise ResumeRefused(f"{path.relative_to(run.out)} does not match this task: " + "; ".join(differences)
                            + "; rerun without resume_from_checkpoint, or restore the configuration and inputs it "
                            "was made with")
    return stage, record


def prepare_global(run):
    """The tile_global command's verified resume (design/specs/orchestration-tiled.md, "Resume of the global task"):
    ("registered", earlier record) when ``tiled/run.json`` says succeeded and matches the task (as the fov task's,
    ``check``) and the chunked checkpoint holds every pipeline round with the SHA-256 global.json records, hashed
    again; None for a run from scratch (no resume asked, no earlier run, or a checkpoint that is incomplete or differs:
    the rounds are produced in order, so a partly written checkpoint is discarded).

    Raises ResumeRefused (exit 3) naming every field of run.json that differs.
    """
    from starfinder.io.chunked import chunked_round_sha256
    from starfinder.workflow._plan import package_identity
    if not run.configuration["execution"].get("resume_from_checkpoint"):
        return None
    folder = _layout.tiled_dir(_layout.fov_dir(run.out, run.identity["fov"]))
    path, record_path = folder / _layout.RUN_RECORD, folder / _layout.GLOBAL_RECORD
    store = folder / _layout.CHUNKED_CHECKPOINT
    if not (path.is_file() and record_path.is_file() and store.is_file()):
        return None
    try:
        record, global_record = json.loads(path.read_text()), json.loads(record_path.read_text())
    except ValueError:
        return None
    if not isinstance(record, dict) or record.get("status") != "succeeded":
        return None
    differences = check(run, record, package_identity())
    if differences:
        raise ResumeRefused(f"{path.relative_to(run.out)} does not match this task: " + "; ".join(differences)
                            + "; rerun without resume_from_checkpoint, or restore the configuration and inputs it "
                            "was made with")
    rounds = global_record.get("checkpoint", {}).get("rounds") or {}
    try:
        for name, entry in rounds.items():
            if chunked_round_sha256(store, name) != entry["sha256"]:
                return None
            for snapshot, digest in (entry.get("snapshots") or {}).items():
                if chunked_round_sha256(store, name, snapshot) != digest:
                    return None
    except (OSError, KeyError, ValueError):
        return None
    if not rounds:
        return None
    return "registered", record


# --- the saved spot-finding input (processing.spot_finding_input) -------------------------------------------------

#: The PipelineConfig fields that make the registered stage: a saved set is read only when the task's equal the set's.
INPUT_FIELDS = ("load", "rotation_degrees", "preprocessing", "registration")


def _json(value):
    """A value as run.json holds it (``_records.jsonable``, then a JSON round trip)."""
    from starfinder._records import jsonable
    return json.loads(json.dumps(jsonable(value)))


def _first_difference(recorded, now, where):
    """``(dotted path, recorded value, value now)`` of the first leaf where two JSON values differ, or None."""
    if isinstance(recorded, dict) and isinstance(now, dict):
        for key in sorted(set(recorded) | set(now)):
            if key not in recorded or key not in now:
                return f"{where}.{key}", recorded.get(key, "absent"), now.get(key, "absent")
            found = _first_difference(recorded[key], now[key], f"{where}.{key}")
            if found:
                return found
        return None
    if isinstance(recorded, list) and isinstance(now, list) and len(recorded) == len(now):
        for i, (a, b) in enumerate(zip(recorded, now)):
            found = _first_difference(a, b, f"{where}.{i}")
            if found:
                return found
        return None
    return None if recorded == now else (where, recorded, now)


def _without_supplied(value):
    """A preprocessing recipe without its supplied statistics path, which is compared by the file's SHA-256."""
    return {k: v for k, v in value.items() if k != "supplied_statistics"} if isinstance(value, dict) else value


def saved_set(run):
    """The saved set the fov task ``run`` reads: ``processing.spot_finding_input``, absolute or relative to the
    manifest's input root; None without the key."""
    value = run.configuration["processing"].get("spot_finding_input")
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else run.root / path


def check_input(run, root, record, pipeline, fov_, supplied):
    """The fields of a saved set's ``run.json`` (``record``) that differ from the fov task ``run`` reading it; [] when
    it may be read.

    The set must be a version-2 record of a run that succeeded, of this dataset, sample and FOV, made by a package
    whose code identity is established and clean, with a registered stage whose record has the SHA-256 run.json
    names; its round roles, channels and the PipelineConfig fields ``INPUT_FIELDS`` (load, rotation, the
    preprocessing recipe and the registration recipe) must equal this task's, and the supplied statistics file, when
    either side has one, the same SHA-256. The package version is not compared: a set made at another revision is
    read, and its code identity is recorded.
    """
    from dataclasses import asdict
    if record.get("format_version") != 2:
        return [f"format_version is {record.get('format_version')!r} (recorded), 2 (now): only a version-2 record is "
                "read"]
    differences = []
    if record.get("status") != "succeeded":
        differences.append(f"status is {record.get('status')!r} (recorded): the set's run did not succeed")
    expected = {"dataset_id": run.configuration["dataset_id"], "sample_id": run.configuration["sample_id"],
                "fov_id": run.identity["fov"], "subtile_id": None}
    for key, value in expected.items():
        if record.get(key) != value:
            differences.append(f"{key} is {record.get(key)!r} (recorded), {value!r} (now)")
    differences += _code_problems(record.get("code") or {}, "recorded")
    entry = (record.get(_layout.CHECKPOINTS) or {}).get("registered")
    if entry is None:
        differences.append("checkpoints.registered is not recorded: the set's run wrote no stage registered")
    else:
        path = _layout.fov_dir(root, run.identity["fov"]) / entry["path"]
        if not path.is_file() or entry.get("sha256") is None or file_sha256(path) != entry["sha256"]:
            differences.append(f"checkpoints.registered {entry['path']} sha256 differs")
    for key, now in (("rounds", _json(asdict(fov_.rounds))), ("channels", _json(fov_.dataset._channel_record()))):
        found = _first_difference(record.get(key), now, key)
        if found:
            differences.append(f"{found[0]} is {_short(found[1])} (recorded), {_short(found[2])} (now)")
    recorded = (record.get("config") or {}).get("pipeline") or {}
    for name in INPUT_FIELDS:
        now = _without_supplied(_json(getattr(pipeline, name)))
        found = _first_difference(_without_supplied(recorded.get(name)), now, f"config.pipeline.{name}")
        if found:
            differences.append(f"{found[0]} is {_short(found[1])} (recorded), {_short(found[2])} (now)")
    digest = ((record.get("preprocessing") or {}).get("supplied_statistics") or {}).get("sha256")
    now = None if supplied is None else file_sha256(supplied)
    if digest != now:
        differences.append(f"preprocessing.supplied_statistics sha256 is {_short(digest)} (recorded), {_short(now)} "
                           "(now)")
    # The deconvolved copy the set's run read must be this run's (design/specs/deconvolution.md, "Verified resume and
    # saved sets"); both may be absent.
    from starfinder.workflow._deconvolve import run_copy_id
    recorded_copy = (record.get("deconvolution") or {}).get("copy_id")
    now_copy = run_copy_id(run.manifest)
    if recorded_copy != now_copy:
        differences.append(f"deconvolution.copy_id is {recorded_copy or 'absent'} (recorded), {now_copy or 'absent'} "
                           "(now)")
    return differences


def prepare_input(run):
    """The fov task's saved spot-finding input: ``{root, record, path}`` (the set's output root, its FOV's run.json and
    that file's path) when the task reads one, else None.

    Raises TaskRefused (exit 2) when the set's ``run.json`` cannot be read or differs from the task (``check_input``),
    naming every field that differs.
    """
    from starfinder.workflow._commands import _processing, supplied_input
    root = saved_set(run)
    if root is None:
        return None
    path = _layout.run_record(root, run.identity["fov"])
    try:
        record = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise TaskRefused(f"processing.spot_finding_input: {path} cannot be read as JSON: {error}") from None
    adapted, pipeline, _ = _processing(run)
    fov_ = adapted.dataset.fov(run.identity["fov"])
    differences = check_input(run, root, record if isinstance(record, dict) else {}, pipeline, fov_,
                              supplied_input(run))
    if differences:
        raise TaskRefused(f"processing.spot_finding_input {path} does not match this task: " + "; ".join(differences)
                          + "; make the set again with this task's preprocessing and registration, or restore the "
                          "settings it was made with")
    return {"root": root, "record": record, "path": path}
