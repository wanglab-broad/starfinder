"""The verify command: D8's completion rule and the staleness rule from the output root, and the per-task tokens.

design/specs/orchestration-engines.md, "-resume and the completion rule (D8)": a task is complete only when its record
says ``succeeded``, lists every declared output of the task and its log, every output it lists is present with its
recorded size (with ``hash``, its SHA-256) and its ``workflow.fingerprint`` equals the current manifest's. A complete
task's token is the record's ``workflow.token``; every other task's is ``rerun:<id>``, ``<id>`` fresh to the
invocation (amendment A1), and so is a complete task's whose record holds no token (Snakemake's, or a person's): that
task stays complete and passes ``final``, because completion is one rule for both engines and a person. An export
task is complete only while the placement it read needs no review or ``approvals.json`` beside it holds an approval of
its tables (W-875). The routes found in the output root (the manifest's ``routes``, W-876: another plan's export and
report of a sample, whose report this plan's compare task reads) are no tasks of this manifest; their records are checked
by the same rule without the fingerprint, which another plan made. The result is written to ``workflow/verify.json``.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from starfinder import _layout
from starfinder._records import atomic_file, file_sha256
from starfinder.workflow._shape import VERIFY_FORMAT
from starfinder.workflow._task import load_manifest, read_record, resolve_path


def _placement(manifest, task):
    """The placement.json a task declares as an input, or None."""
    return next((resolve_path(manifest, i) for i in task.get("inputs") or [] if Path(i).name == "placement.json"),
                None)


def _unapproved(manifest, task):
    """Why the placement an export task reads holds no approval it needs, or None: it reads no placement, its
    placement.json is no placement record whose review is ``required`` (an engine's stub writes placeholder text; the
    placement task's own completion checks that file), or ``approvals.json`` (or, as written before W-875,
    ``placement.json``) holds an approval whose table SHA-256 equal those ``placement.json`` records."""
    from starfinder.stitching._persist import APPROVALS_FILE, saved_approvals
    path = _placement(manifest, task)
    if path is None or not path.is_file():
        return None
    try:
        record = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    review = record.get("review") if isinstance(record, dict) else None
    if not isinstance(review, dict) or review.get("state") != "required":
        return None
    try:
        tables = record["tables"]
        approved = saved_approvals(path.parent, record["sample"], tables["fovs"]["sha256"], tables["links"]["sha256"])
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        return f"the approvals of the placement {path.parent} cannot be read: {error}"
    return None if approved else f"the placement {path} needs review and {APPROVALS_FILE} holds no approval of its tables"


def _problems(manifest, task, record, check_hash, temporary, fingerprint=True):
    """Why a task is not complete under the staleness rule ([] when it is); without ``fingerprint`` (a found route's
    task, which another plan made) the record's fingerprint is not compared."""
    if record is None:
        failed = _layout.failed_record(manifest["roots"]["output"], task["id"])
        if not failed.is_file():
            return ["no task record"]
        try:
            held = json.loads(failed.read_text()).get("status") == "held"
        except (OSError, ValueError, AttributeError):
            held = False
        placement = _placement(manifest, task) if held and task["kind"] == "export" else None
        if placement is not None and placement.is_file() and _unapproved(manifest, task) is None:
            return ["held for review (a failed record that says held) and no task record; the placement now holds "
                    "an approval of its tables or needs no review: run the workflow again"]
        return ["held for review (a failed record that says held) and no task record" if held
                else "a failed record and no task record"]
    if record.get("status") != "succeeded":
        return [f"the record says {record.get('status')!r}"]
    problems = []
    recorded = (record.get("workflow") or {}).get("fingerprint")
    if fingerprint and recorded != task["fingerprint"]:
        problems.append(f"the record's fingerprint {recorded} differs from the manifest's {task['fingerprint']}")
    listed = {entry.get("path"): entry for entry in record.get("outputs") or []}
    unlisted = [o for o in [*task["outputs"], task["log"]] if o not in listed]
    if unlisted:
        problems.append(f"the record does not list {', '.join(unlisted)}")
    for declared, entry in listed.items():
        if declared in temporary:
            continue
        path = resolve_path(manifest, declared)
        if declared.endswith("/"):
            if not path.is_dir():
                problems.append(f"{declared} is missing")
                continue
            files = [(path / item["path"], item, declared + item["path"]) for item in entry.get("files") or []]
        else:
            files = [(path, entry, declared)]
        for file, item, name in files:
            if not file.is_file():
                problems.append(f"{name} is missing")
            elif file.stat().st_size != item.get("bytes"):
                problems.append(f"{name} has {file.stat().st_size} bytes, the record {item.get('bytes')}")
            elif check_hash and "sha256:" + file_sha256(file) != item.get("sha256"):
                problems.append(f"{name} has another SHA-256 than the record's")
    if task["kind"] == "export":
        unapproved = _unapproved(manifest, task)
        if unapproved is not None:
            problems.append(unapproved)
    return problems


def _routes(manifest, results, check_hash):
    """The routes of each sample in the manifest's ``routes``, each with its tasks' state: this plan's route from the
    results of its tasks; a found route's export and report by the record rule without the fingerprint (their record
    says succeeded, lists the outputs the manifest names and the log, and every output it lists exists with its
    recorded size, and with ``check_hash`` its SHA-256)."""
    out = {}
    for sample, entries in (manifest.get("routes") or {}).items():
        listed = []
        for entry in entries:
            if entry["planned"]:
                tasks = {t: {"complete": results[t]["complete"], "problems": results[t]["problems"]}
                         for t in entry["tasks"]}
            else:
                tasks = {}
                for found in entry["tasks"]:
                    problems = _problems(manifest, found, read_record(manifest, found["id"]), check_hash, set(),
                                         fingerprint=False)
                    tasks[found["id"]] = {"complete": not problems, "problems": problems}
            listed.append({"route": entry["route"], "planned": entry["planned"],
                           "complete": all(t["complete"] for t in tasks.values()), "tasks": tasks})
        out[sample] = listed
    return out


def verify(manifest, *, task=None, hash=False, final=False) -> dict:
    """Apply the completion and staleness rule to the tasks of a manifest and give each its token.

    Parameters
    ----------
    manifest : str or pathlib.Path
        The plan's ``workflow/manifest.json``.
    task : str or None
        One task id; None checks every task of the manifest.
    hash : bool
        Compare each listed output's SHA-256 too, not only its size.
    final : bool
        The check at the end of a run: the result's ``passed`` is False when a task is not complete.

    Returns
    -------
    dict
        The record written to ``workflow/verify.json``: ``invocation`` (the id of this invocation), ``tasks`` (per
        task id: ``complete``, ``problems`` and ``token``), ``incomplete`` (the ids of the tasks that are not
        complete), ``routes`` (per sample, every export route the manifest names: ``route``, ``planned``,
        ``complete`` and per task id ``complete`` and ``problems``; empty with ``task``), ``found_incomplete`` (the
        task ids of found routes that are not complete) and ``passed``.

    Notes
    -----
    A task is complete when its record says ``succeeded``, lists every declared output and its log, every output it
    lists exists with its recorded size (and with ``hash`` its SHA-256), and its ``workflow.fingerprint`` equals the
    manifest's (design/specs/orchestration-engines.md, "The task fingerprint"). Outputs the manifest's groups mark
    temporary are exempt. An export task is complete only while the placement it read needs no review or the
    ``approvals.json`` beside ``placement.json`` holds an approval of the tables ``placement.json`` records; an export
    held for review stays incomplete, and its problem says so once the placement holds an approval. A complete
    task's token is its record's ``workflow.token``; every other task's, and a complete task's whose record holds no
    token, is ``rerun:<invocation>``, which differs between invocations.

    A route found in the output root (the manifest's ``routes`` with ``planned`` false: another plan's export and
    report of a sample, whose report this plan's compare task reads) is no task of this manifest and gets no token.
    Its export and report records are checked by the same rule except the fingerprint, which another plan made: the
    record says ``succeeded``, lists the output the manifest names and the log, and every output it lists exists with
    its recorded size (with ``hash``, its SHA-256); a missing or changed file is a problem naming the file. With
    ``final`` a found route that is not complete fails ``passed`` too. An output root with one route lists one route.
    """
    path = Path(manifest).absolute()
    data, digest = load_manifest(path)
    tasks = data["tasks"] if task is None else [t for t in data["tasks"] if t["id"] == task]
    if task is not None and not tasks:
        raise ValueError(f"the manifest {path} has no task {task!r}")
    temporary = {}
    for group in data.get("groups") or []:
        temporary[group["name"]] = set(group.get("temporary") or ())
    invocation = uuid.uuid4().hex
    results = {}
    for entry in tasks:
        record = read_record(data, entry["id"])
        exempt = {template.format(**entry["identity"]) for template in temporary.get(entry.get("group"), ())}
        problems = _problems(data, entry, record, hash, exempt)
        token = (record.get("workflow") or {}).get("token") if not problems else None
        results[entry["id"]] = {"complete": not problems, "problems": problems,
                                "token": token if token is not None else f"rerun:{invocation}"}
    incomplete = [k for k, v in results.items() if not v["complete"]]
    routes, found_incomplete = ({} if task is not None else _routes(data, results, hash)), []
    for sample in routes.values():
        for route in sample:
            if not route["planned"]:
                found_incomplete += [k for k, v in route["tasks"].items() if not v["complete"]]
    out = {"format": VERIFY_FORMAT, "manifest": str(path), "manifest_sha256": digest, "invocation": invocation,
           "generated_at": datetime.now(timezone.utc).isoformat(), "hash": bool(hash), "final": bool(final),
           "tasks": results, "incomplete": incomplete, "routes": routes, "found_incomplete": found_incomplete,
           "passed": not (final and (incomplete or found_incomplete))}
    with atomic_file(_layout.workflow_dir(data["roots"]["output"]) / _layout.VERIFY_FILE) as tmp:
        tmp.write_text(json.dumps(out, indent=1) + "\n")
    return out
