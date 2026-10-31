"""Verified resume of the fov task (D8; design/specs/orchestration.md, "Checkpoints and resume").

With ``execution.resume_from_checkpoint`` the fov command resumes from the latest stage an earlier attempt wrote,
only when the earlier ``run.json`` matches the task: its version, identity, ``workflow.blocks_sha256``, code
identity, inputs, codebook, supplied statistics and stage records. A mismatch refuses with exit 3 and a message that
names every field that differs. The check reads ``run.json``, never the task record or the log.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from starfinder import _checkpoint, _layout
from starfinder._records import file_sha256
from starfinder.dataset._run_record import resolve_source
from starfinder.workflow._task import ResumeRefused

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
