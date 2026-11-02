"""One task of the manifest run by its command: the checks before work, the log, completion and the records.

design/specs/orchestration.md, "The CLI" (the common arguments and the exit codes), "Completion" and "Logging and
records". A command checks, before any work, that the task exists, that the resolved configuration hashes to the
manifest's, that every task it needs has a succeeded record, that every declared input exists and that its blocks
translate as the plan translated them (exit 2, nothing written; a refused verified resume exits 3). It then removes
its earlier record, writes its log, runs one API entry, checks that every declared output exists, closes the log and
writes its task record last (exit 0). A failure writes the traceback to the log, removes every declared output (those
a verified resume kept included), so that no declared output but the log remains, and writes ``<task>.failed.json``
(exit 1).
"""
from __future__ import annotations

import json
import logging
import shutil
import socket
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from starfinder import _layout
from starfinder._records import atomic_file, file_sha256
from starfinder._resources import process_resources
from starfinder.workflow._shape import (COMMANDS, FAILED_FORMAT, MANIFEST_FORMAT, TASK_FORMAT, ConfigurationView,
    sha256)

logger = logging.getLogger("starfinder.workflow")

#: The kinds whose command removes the earlier declared outputs before its work: the others' API entries manage
#: their folders (export and report remove what their earlier record lists; without report.json, the report command
#: first removes only the page and tables, preserving foreign files for the public guard; export first removes its
#: unlisted previews/ whatever export.previews says and restores its record from workflow/exports/ when needed,
#: saving it again even on failure before declared outputs are removed; the Fiji task's dataset.xml is the tool's,
#: written in the same job before the check).
PRE_REMOVE = frozenset({"summarize", "merge", "fov", "tile_global", "tile", "tile_merge", "morphology", "segment",
                        "assign", "bigstitcher_macro", "bigstitcher_import", "stitch"})


class TaskRefused(Exception):
    """A task refused before any work (exit status 2): the message names what the check found."""


class ResumeRefused(Exception):
    """A verified resume refused (exit status 3): the message names every field of run.json that differs."""


class TaskFailed(RuntimeError):
    """A task whose API call returned but whose completion check failed (exit status 1)."""


@dataclass(frozen=True)
class Context:
    """The execution context an engine passes to a command; recorded, never interpreted."""
    engine: str = "manual"
    engine_version: str | None = None
    job_id: str | None = None
    attempt: int = 1
    image: str | None = None
    cache_token: str | None = None


def _now():
    return datetime.now(timezone.utc).isoformat()


class _UtcFormatter(logging.Formatter):
    """``2026-10-29T18:17:51.123Z INFO starfinder.dataset <task> <message>``: one line per record, UTC."""
    converter = time.gmtime

    def __init__(self, task_id):
        super().__init__(f"%(asctime)s %(levelname)s %(name)s {task_id} %(message)s")

    def formatTime(self, record, datefmt=None):
        return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + f".{int(record.msecs):03d}Z"


def load_manifest(path):
    """The manifest at ``path`` and the SHA-256 of its bytes; TaskRefused when it is unreadable or another format."""
    path = Path(path)
    try:
        data = path.read_bytes()
        manifest = json.loads(data)
    except (OSError, ValueError) as error:
        raise TaskRefused(f"the manifest {path} cannot be read: {error}") from None
    if not isinstance(manifest, dict) or manifest.get("format") != MANIFEST_FORMAT:
        raise TaskRefused(f"{path} is not a manifest of format {MANIFEST_FORMAT}")
    return manifest, sha256(data)


def resolve_path(manifest, declared) -> Path:
    """The file a declared path names: ``input:<path>`` under the input root, a relative path under the output root,
    an absolute path as it is."""
    roots = manifest["roots"]
    if declared.startswith("input:"):
        return Path(roots["input"]) / declared[len("input:"):]
    path = Path(declared)
    return path if path.is_absolute() else Path(roots["output"]) / declared


def read_record(manifest, task_id):
    """A task's record, or None when it has none (or an unreadable one)."""
    path = _layout.task_record(manifest["roots"]["output"], task_id)
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


class Run:
    """One task of a manifest, as its command runs it: the manifest, the task, the roots and the views.

    ``configuration`` and ``api`` are the resolved configuration and its API configs as the task's kind may read
    them (ConfigurationView).
    """

    def __init__(self, manifest_path, task_id, command, context=Context()):
        self.manifest_path = Path(manifest_path).absolute()
        self.manifest, self.manifest_sha256 = load_manifest(self.manifest_path)
        tasks = {t["id"]: t for t in self.manifest["tasks"]}
        if task_id not in tasks:
            raise TaskRefused(f"the manifest {self.manifest_path} has no task {task_id!r}")
        self.task = tasks[task_id]
        self.tasks = tasks
        self.kind = self.task["kind"]
        if COMMANDS[self.kind] != command:
            raise TaskRefused(f"task {task_id} is of kind {self.kind}, which the command {COMMANDS[self.kind]} runs, "
                              f"not {command}")
        self.context = context
        self.out = Path(self.manifest["roots"]["output"])
        self.root = Path(self.manifest["roots"]["input"])
        self.identity = self.task["identity"]
        self.device = "cpu"
        self.fallback = None
        self.kept = set()
        # A verified resume of the fov task: (stage, earlier run.json), set by the command's preparation.
        self.resume = None
        self.started = time.perf_counter()
        path = self.out / self.manifest["configuration"]["path"]
        try:
            data = path.read_bytes()
        except OSError as error:
            raise TaskRefused(f"the resolved configuration {path} cannot be read: {error.strerror}") from None
        if sha256(data) != self.manifest["configuration"]["sha256"]:
            raise TaskRefused(f"the resolved configuration {path} has SHA-256 {sha256(data)}; the manifest names "
                              f"{self.manifest['configuration']['sha256']}: plan the run again")
        self.resolved = json.loads(data)
        configuration = dict(self.resolved["configuration"])
        api = configuration.pop("api")
        self.configuration = ConfigurationView(configuration, self.kind)
        self.api = ConfigurationView(api, self.kind, "api")

    @property
    def id(self):
        return self.task["id"]

    def path(self, declared) -> Path:
        return resolve_path(self.manifest, declared)

    def inputs(self, suffix=None):
        """The task's declared inputs as paths, those ending in ``suffix`` when given, in the manifest's order."""
        return [self.path(i) for i in self.task["inputs"] if suffix is None or i.endswith(suffix)]

    def dataset(self, dataset=None):
        """The Dataset of the task (``_workflow_dataset`` of the common part), with the manifest's roots."""
        if dataset is None:
            from starfinder.dataset.workflow import _workflow_dataset
            from starfinder.workflow._translate import _base
            dataset = _workflow_dataset(_base(self.configuration.plain()))
        dataset.input_root, dataset.output_root = self.root, self.out
        return dataset

    # --- the checks before work ----------------------------------------------------------------------------------

    def check(self):
        """The common checks of every command (exit 2): the translation of every block the kind names, the needs'
        records and the declared inputs (the task's existence and the resolved configuration's hash are checked when
        the run is read)."""
        from starfinder.workflow._translate import TranslationMismatch, check_api
        try:
            check_api(self.configuration, self.api, self.task["blocks"], self.kind)
        except TranslationMismatch as error:
            raise TaskRefused(str(error)) from None
        waiting = []
        for need in self.task["needs"]:
            record = read_record(self.manifest, need)
            if record is None or record.get("status") != "succeeded":
                waiting.append(need)
        if waiting:
            raise TaskRefused(f"task {self.id} needs {', '.join(waiting)}, which "
                              f"{'has' if len(waiting) == 1 else 'have'} no record that says succeeded")
        missing = [i for i in self.task["inputs"]
                   if not (self.path(i).is_dir() if i.endswith("/") else self.path(i).is_file())]
        if missing:
            raise TaskRefused(f"task {self.id}: the declared inputs {', '.join(missing)} do not exist")

    # --- the declared outputs ------------------------------------------------------------------------------------

    def outputs(self):
        """The declared outputs (without the record and the log) as (declared, path)."""
        return [(o, self.path(o)) for o in self.task["outputs"]]

    def remove_outputs(self, keep=()):
        """Remove the declared outputs but those in ``keep``; return the declared paths removed."""
        removed = []
        for declared, path in self.outputs():
            if declared in keep:
                continue
            if path.is_dir() and declared.endswith("/"):
                shutil.rmtree(path)
                removed.append(declared)
            elif path.is_file():
                path.unlink()
                removed.append(declared)
        return removed

    def output_entries(self):
        """Path, SHA-256 and bytes of every declared output; a folder lists its files."""
        entries = []
        for declared, path in self.outputs():
            if declared.endswith("/"):
                files = [{"path": p.relative_to(path).as_posix(), "sha256": "sha256:" + file_sha256(p),
                          "bytes": p.stat().st_size} for p in sorted(path.rglob("*")) if p.is_file()]
                entries.append({"path": declared, "sha256": sha256(json.dumps(files, sort_keys=True).encode()),
                                "bytes": sum(f["bytes"] for f in files), "files": files})
            else:
                entries.append({"path": declared, "sha256": "sha256:" + file_sha256(path),
                                "bytes": path.stat().st_size})
        return entries

    # --- the record ----------------------------------------------------------------------------------------------

    def workflow_section(self):
        """The ``workflow`` section of the task record and of the fov kinds' run.json (D8), measured until now."""
        from starfinder.workflow._plan import package_identity
        context = self.context
        return {
            "task": {"id": self.id, "kind": self.kind},
            "backend": {"name": "python", "device": self.device},
            "fallback": self.fallback,
            "configuration_sha256": self.manifest["configuration"]["configuration_sha256"],
            "blocks_sha256": self.task["blocks_sha256"],
            "manifest_sha256": self.manifest_sha256,
            "fingerprint": self.task["fingerprint"],
            "token": context.cache_token,
            "code": package_identity(),
            "execution_context": {"engine": context.engine, "engine_version": context.engine_version,
                                  "job_id": context.job_id, "attempt": context.attempt, "image": context.image,
                                  "host": socket.gethostname()},
            "resumed_from": None if self.resume is None else self.resume[0],
            **process_resources(self.started),
        }


def _write(path, data):
    with atomic_file(path) as tmp:
        tmp.write_text(json.dumps(data, indent=1) + "\n")


def execute(command, manifest_path, task_id, function, context=Context(), argv=None, prepare=None):
    """Run one task: the checks, the log, ``function(run)``, completion and the records; return the exit status.

    ``prepare(run)``, called after the common checks and before anything is written, may raise ResumeRefused (exit
    3) or TaskRefused (exit 2) and returns the declared outputs to keep from an earlier attempt (a verified resume).
    """
    try:
        run = Run(manifest_path, task_id, command, context)
        run.check()
        run.kept = set(prepare(run) or ()) if prepare is not None else set()
    except TaskRefused as error:
        print(f"{command} refused: {error}", file=sys.stderr)
        return 2
    except ResumeRefused as error:
        print(f"resume_from_checkpoint refused for {task_id}: {error}", file=sys.stderr)
        return 3
    out = run.out
    record_path = _layout.task_record(out, task_id)
    failed_path = _layout.failed_record(out, task_id)
    log_path = _layout.task_log(out, task_id)
    run.started, started_at = time.perf_counter(), _now()
    # The record of an earlier success goes first: a failed attempt never leaves a succeeded record behind.
    record_path.unlink(missing_ok=True)
    failed_path.unlink(missing_ok=True)
    if run.kind in PRE_REMOVE:
        run.remove_outputs(keep=run.kept)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    handler.setFormatter(_UtcFormatter(task_id))
    level = logging.DEBUG if run.configuration["execution"].get("log_level") == "DEBUG" else logging.INFO
    handler.setLevel(level)
    package_logger, warnings_logger = logging.getLogger("starfinder"), logging.getLogger("py.warnings")
    saved_level = package_logger.level
    package_logger.addHandler(handler)
    warnings_logger.addHandler(handler)
    package_logger.setLevel(level)
    logging.captureWarnings(True)
    closed = False

    def close():
        nonlocal closed
        if not closed:
            package_logger.removeHandler(handler)
            warnings_logger.removeHandler(handler)
            package_logger.setLevel(saved_level)
            logging.captureWarnings(False)
            handler.close()
            closed = True

    try:
        logger.info(f"task {task_id} ({run.kind}), attempt {context.attempt}")
        logger.info(f"command: {' '.join(argv) if argv else command}")
        logger.info(f"execution context: engine {context.engine} {context.engine_version or ''}, job "
                    f"{context.job_id}, attempt {context.attempt}, image {context.image}, cache token "
                    f"{context.cache_token}")
        from starfinder.workflow._plan import package_identity
        logger.info(f"package: {package_identity()}")
        logger.info(f"configuration_sha256 {run.manifest['configuration']['configuration_sha256']}, blocks_sha256 "
                    f"{run.task['blocks_sha256']}, fingerprint {run.task['fingerprint']}")
        if run.kept:
            logger.info(f"kept from the earlier attempt: {', '.join(sorted(run.kept))}")
        function(run)
        missing = [declared for declared, path in run.outputs()
                   if not (path.is_dir() if declared.endswith("/") else path.is_file())]
        if missing:
            raise TaskFailed(f"the task's call returned without the declared outputs {', '.join(missing)}")
        if run.kind in ("fov", "tile_merge", "tile_global"):
            from starfinder.dataset._run_record import set_workflow
            folder = _layout.fov_dir(out, run.identity["fov"])
            set_workflow((_layout.tiled_dir(folder) if run.kind == "tile_global" else folder) / _layout.RUN_RECORD,
                         run.workflow_section(), {"path": run.manifest["configuration"]["path"],
                                                  "sha256": run.manifest["configuration"]["sha256"]})
        outputs = run.output_entries()
        section = run.workflow_section()
        logger.info(f"succeeded in {section['wall_s']:.2f}s; {len(outputs)} declared outputs")
        close()
        log_entry = {"path": run.task["log"], "sha256": "sha256:" + file_sha256(log_path),
                     "bytes": log_path.stat().st_size}
        _write(record_path, {"format": TASK_FORMAT, "id": task_id, "kind": run.kind, "identity": run.identity,
                             "status": "succeeded", "started_at": started_at, "ended_at": _now(),
                             "outputs": [*outputs, log_entry], "workflow": section})
        return 0
    except Exception as error:
        logger.error(f"failed: {type(error).__name__}: {error}\n{traceback.format_exc()}")
        # A failed attempt leaves no declared output but its log, resumed or not: what a verified resume kept and what
        # an earlier attempt of a kind that does not remove its outputs first left go too, so that the next attempt
        # runs from scratch. An interruption (below) leaves the files as they are.
        removed = run.remove_outputs()
        logger.error(f"removed the declared outputs: {', '.join(removed) or 'none'}")
        section = run.workflow_section()
        close()
        _write(failed_path, {"format": FAILED_FORMAT, "id": task_id, "kind": run.kind, "identity": run.identity,
                             "status": "failed", "started_at": started_at, "ended_at": _now(),
                             "error": {"type": type(error).__name__, "message": str(error)}, "removed": removed,
                             "log": {"path": run.task["log"], "sha256": "sha256:" + file_sha256(log_path)},
                             "workflow": section})
        print(f"{command} failed: {task_id}: {type(error).__name__}: {error} (log {log_path})", file=sys.stderr)
        return 1
    finally:
        # An interruption (a BaseException) leaves the files as they are, as a kill would.
        close()
