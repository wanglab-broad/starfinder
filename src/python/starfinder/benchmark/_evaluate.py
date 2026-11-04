"""Private evaluate-only adapters behind ``starfinder experiment evaluate``: saved sources in, metric records out.

design/specs/benchmark.md, "Proposed evaluate-only contract". The adapters read; they never process. A sources file
lists named sources of four kinds:

``workflow``
    A plan's ``workflow/manifest.json`` and, optionally, the ``tasks`` to read (default every task). Each task is
    checked with the workflow's completion and staleness rule with hashes (``starfinder workflow verify --hash``),
    read-only: no ``verify.json`` is written. A metric reads a listed output of a complete task, verified against the
    task record's size and SHA-256; an incomplete task skips the metric; a temporary output that is gone makes it
    unavailable.
``checkpoint``
    A per-FOV checkpoint folder and one ``stage`` (``registered``, ``candidates`` or ``pre_qc``), restored with
    ``starfinder.dataset.load_checkpoint``. A version-3 header's payload hashes are checked; versions 1 and 2 hold
    none, so they need an explicit ``inventory`` of every stage file with its SHA-256, and the record keeps that
    limitation.
``trial``
    A private trial's output folder: ``record.json`` (format ``starfinder.benchmark.trial/1``, its configuration hash
    recomputed) and every output it lists, by size and SHA-256. A recorded input a metric reads is checked against
    the record's SHA-256. A trial record holds no truth: supply it as a ``files`` source.
``files``
    Explicit files (truth, codebooks, original inputs), each with its SHA-256.

A metrics file lists the metrics to compute: a ``name``, an optional short-set ``metric`` ID, a ``function`` of
``starfinder.evaluation`` (``<module>.<name>``), its literal ``parameters`` and its ``arguments`` read from the
sources. The saved evaluation keeps every source's identity and failures, every consumed file with its SHA-256,
each metric's parameters and arguments, and its values with units, counts and reasons; a missing value is null with
a reason, never a zero. A metric's outcome is ``computed`` (the result's own status then says ok, undefined, missing
or failed), ``skipped`` (a source it needs failed, or a workflow task is not complete), ``unavailable`` (a file,
snapshot or value it needs is absent) or ``failed`` (the metric raised).
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib
import inspect
import io
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import yaml

from starfinder._records import jsonable, software, tuples, write_json

EVALUATION_FORMAT = "starfinder.experiment.evaluation/1"
TRIAL_FORMAT = "starfinder.benchmark.trial/1"
SOURCE_KINDS = ("workflow", "checkpoint", "trial", "files")
OUTCOMES = ("computed", "skipped", "unavailable", "failed")
#: The modules of starfinder.evaluation whose public functions a metric may name.
METRIC_MODULES = ("matching", "preprocessing", "readout", "registration", "resources", "segmentation",
                  "spot_finding", "stitching")
READS = ("table", "points", "column", "array", "json", "volume")
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_DIGEST = re.compile(r"^(?:sha256:)?[0-9a-f]{64}$")
_ARGUMENT_KEYS = {"source", "output", "input", "task", "file", "read", "key", "where", "columns", "column",
                  "metadata", "mapping"}


class SourceFailure(Exception):
    """A source, or a file it lists, fails its identity or integrity check."""


class Skipped(Exception):
    """A metric needs a failed source or an incomplete workflow task."""


class Unavailable(Exception):
    """A file, snapshot or value a metric needs is absent."""


def _mapping(path, key):
    try:
        value = yaml.safe_load(Path(path).read_bytes())
    except yaml.YAMLError as error:
        raise ValueError(f"{path}: not YAML or JSON: {error}") from None
    if not isinstance(value, dict) or set(value) != {key} or not isinstance(value[key], list) or not value[key]:
        raise ValueError(f"{path} must be a mapping with exactly one nonempty list {key!r}")
    return value[key]


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _bare(digest):
    return digest.removeprefix("sha256:") if isinstance(digest, str) else digest


def _digest(value, where):
    """A required SHA-256 (64 lowercase hex characters, bare or ``sha256:``-prefixed), bare; SourceFailure naming
    ``where`` (the source, the entry and the field) when it is missing or malformed."""
    if not isinstance(value, str) or not _DIGEST.match(value):
        raise SourceFailure(f"{where}: a SHA-256 of 64 lowercase hex characters (optionally sha256:-prefixed) is "
                            f"required, not {value!r}")
    return _bare(value)


def _size(value, where):
    """A required byte count (a nonnegative integer); SourceFailure naming ``where`` when missing or malformed."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SourceFailure(f"{where}: an integer byte count is required, not {value!r}")
    return value


def _optional_digest(entry, key, where):
    """The sources file's own SHA-256 of a record that holds no recorded hash elsewhere: None when the key is
    absent, else a required digest."""
    return _digest(entry[key], where) if key in entry else None


def _read_value(data, path, read, spec):
    """A file's bytes as the value ``read`` names (default by suffix)."""
    suffix = Path(path).suffix.lower()
    read = read or {".csv": "table", ".npy": "array", ".npz": "array", ".json": "json", ".tif": "volume",
                    ".tiff": "volume"}.get(suffix)
    if read not in READS:
        raise ValueError(f"cannot read {path}: name read as one of {list(READS)}")
    try:
        if read in ("table", "points", "column"):
            return pd.read_csv(io.BytesIO(data))
        if read == "array":
            if suffix == ".npz":
                with np.load(io.BytesIO(data), allow_pickle=False) as arrays:
                    return {name: arrays[name] for name in arrays.files}
            return np.load(io.BytesIO(data), allow_pickle=False)
        if read == "json":
            return json.loads(data)
        from starfinder.io import load_volume
        return load_volume(path).image
    except (ValueError, OSError, KeyError, EOFError) as error:
        # The bytes are the recorded ones, so the file itself cannot serve as this source's content.
        raise SourceFailure(f"{path} cannot be read as {read}: {error}") from None


class _Source:
    kind = None
    keys = set()

    def __init__(self, name, entry, base):
        unknown = set(entry) - {"name", "kind"} - self.keys
        if unknown:
            raise ValueError(f"source {name!r}: unsupported keys {sorted(unknown)}")
        self.name, self.entry, self.base = name, entry, base
        self.consumed = {}
        self.limitations = []
        self.error = None
        self.record = {"name": name, "kind": self.kind}

    def _path(self, key):
        if key not in self.entry or not isinstance(self.entry[key], str):
            raise ValueError(f"source {self.name!r} needs a path {key!r}")
        return (self.base / self.entry[key]).resolve()

    def consume(self, path, sha256=None, size=None, checked_against=None):
        """A file's bytes, hashed as they are read and checked against the recorded size and SHA-256.

        ``sha256`` and ``size`` are optional only for the files that have no recorded hash or size anywhere: a trial's
        record.json and a workflow manifest when the sources file gives no ``sha256`` for them, a workflow task
        record, and a version-3 checkpoint header without an inventory (their bytes are hashed and listed). Every
        other caller passes a value it has already required with ``_digest`` and ``_size``.
        """
        path = Path(path)
        if sha256 is not None:
            sha256 = _digest(sha256, f"the {checked_against} entry for {path}")
        if size is not None:
            size = _size(size, f"the {checked_against} entry for {path}")
        if not path.is_file():
            raise Unavailable(f"{path} is missing")
        data = path.read_bytes()
        digest = _sha(data)
        if size is not None and len(data) != size:
            raise SourceFailure(f"{path} has {len(data)} bytes, the {checked_against} {size}")
        if sha256 is not None and digest != _bare(sha256):
            raise SourceFailure(f"{path} has SHA-256 {digest}, the {checked_against} {_bare(sha256)}")
        self.consumed[str(path)] = {"path": str(path), "sha256": digest, "bytes": len(data),
                                    "checked_against": checked_against}
        return data

    def roots(self):
        return []

    def summary(self):
        return dict(self.record, status="failed" if self.error else "verified", error=self.error,
                    limitations=self.limitations, consumed=list(self.consumed.values()))


class _Files(_Source):
    kind = "files"
    keys = {"files"}

    def verify(self):
        files = self.entry.get("files")
        if not isinstance(files, dict) or not files:
            raise ValueError(f"source {self.name!r} needs a nonempty mapping files: {{key: {{path, sha256}}}}")
        self.files = {}
        for key, item in files.items():
            if not isinstance(item, dict) or set(item) - {"path", "sha256"} or not isinstance(item.get("path"), str):
                raise ValueError(f"source {self.name!r} file {key!r} needs exactly path and sha256")
            self.files[key] = ((self.base / item["path"]).resolve(), item.get("sha256"))
        self.record["files"] = {key: {"path": str(p), "sha256": _bare(s)} for key, (p, s) in self.files.items()}
        # Every file's SHA-256 is required, and checked before any metric reads the source.
        self.files = {key: (path, _digest(digest, f"source {self.name!r} file {key!r} field sha256"))
                      for key, (path, digest) in self.files.items()}
        for path, digest in self.files.values():
            self.consume(path, digest, checked_against="sources file")

    def open(self, spec):
        if spec.get("file") not in self.files:
            raise Unavailable(f"source {self.name!r} has no file {spec.get('file')!r}")
        path, digest = self.files[spec["file"]]
        return _read_value(self.consume(path, digest, checked_against="sources file"), path, spec.get("read"), spec)


class _Trial(_Source):
    kind = "trial"
    keys = {"path", "sha256"}

    def roots(self):
        return [self._path("path")]

    def verify(self):
        from starfinder.workflow._shape import configuration_sha256
        self.folder = self._path("path")
        self.record["path"] = str(self.folder)
        self.limitations.append("a trial record holds no truth inventory: truth comes from a files source, and the "
                                "trial folder alone is not a portable evaluate-only dataset")
        record_file = self.folder / "record.json"
        if not record_file.is_file():
            raise SourceFailure(f"{record_file} is missing")
        data = self.consume(record_file, _optional_digest(self.entry, "sha256", f"source {self.name!r} field sha256"),
                            checked_against="sources file")
        try:
            record = json.loads(data)
        except ValueError as error:
            raise SourceFailure(f"{record_file} is not JSON: {error}") from None
        if not isinstance(record, dict) or record.get("format") != TRIAL_FORMAT:
            raise SourceFailure(f"{record_file} is not a {TRIAL_FORMAT} record")
        self.trial = record
        self.record.update(record={"path": str(record_file), "sha256": _sha(data)}, trial_status=record.get("status"),
                           trial_error=record.get("error"), configuration_sha256=record.get("configuration_sha256"),
                           frame_id=((record.get("configuration") or {}).get("metadata") or {}).get("frame_id"),
                           method=(record.get("configuration") or {}).get("method"), seed=record.get("seed"))
        if record.get("status") == "succeeded":
            if record.get("configuration_sha256") != configuration_sha256(record["configuration"]):
                raise SourceFailure(f"{record_file}'s configuration_sha256 differs from its configuration's")
            # A succeeded trial lists every output with its SHA-256 and byte count (_trial._outputs), and every
            # file input with its SHA-256; a missing or malformed value fails the source before any metric reads it.
            for role, entry in (record.get("inputs") or {}).items():
                if isinstance(entry, dict) and "path" in entry:
                    _digest(entry.get("sha256"), f"source {self.name!r} input {role!r} field sha256")
            for output in record.get("outputs") or []:
                self._output(output)

    def _output(self, output):
        if not isinstance(output, dict) or not isinstance(output.get("path"), str):
            raise SourceFailure(f"source {self.name!r}: a trial output entry needs a path, not {output!r}")
        name = Path(output["path"])
        if name.is_absolute() or ".." in name.parts:
            raise SourceFailure(f"trial output {output['path']!r} is outside its folder")
        where = f"source {self.name!r} output {output['path']!r} field"
        digest, size = _digest(output.get("sha256"), f"{where} sha256"), _size(output.get("bytes"), f"{where} bytes")
        try:
            return self.consume(self.folder / name, digest, size, checked_against="trial record")
        except Unavailable as error:
            raise SourceFailure(f"listed trial output {error}") from None

    def open(self, spec):
        if self.trial.get("status") != "succeeded":
            raise Skipped(f"trial {self.folder} has status {self.trial.get('status')!r}: {self.trial.get('error')}")
        if spec.get("metadata") is True:
            from starfinder.image import ImageMetadata
            return ImageMetadata(**tuples(self.trial["configuration"]["metadata"]))
        if "output" in spec:
            output = next((o for o in self.trial.get("outputs") or [] if o["path"] == spec["output"]), None)
            if output is None:
                raise Unavailable(f"trial {self.folder} lists no output {spec['output']!r}")
            return _read_value(self._output(output), self.folder / output["path"], spec.get("read"), spec)
        if "input" in spec:
            entry = (self.trial.get("inputs") or {}).get(spec["input"])
            if entry is None or "path" not in entry:
                raise Unavailable(f"trial {self.folder} has no file input {spec['input']!r} (a generated scene "
                                  "input has no file)")
            path = Path(entry["path"])
            digest = _digest(entry.get("sha256"), f"source {self.name!r} input {spec['input']!r} field sha256")
            return _read_value(self.consume(path, digest, checked_against="trial record"), path,
                               spec.get("read"), spec)
        raise ValueError(f"an argument of trial source {self.name!r} names output, input or metadata: true")


class _Workflow(_Source):
    kind = "workflow"
    keys = {"manifest", "tasks", "sha256"}

    def roots(self):
        return [Path(r).resolve() for r in (self.manifest.get("roots") or {}).values() if r] \
            if hasattr(self, "manifest") else []

    def verify(self):
        from starfinder import _layout
        from starfinder.workflow._verify import _problems
        path = self._path("manifest")
        self.record["manifest"] = {"path": str(path)}
        if not path.is_file():
            raise SourceFailure(f"the manifest {path} is missing")
        data = self.consume(path, _optional_digest(self.entry, "sha256", f"source {self.name!r} field sha256"),
                            checked_against="sources file")
        try:
            manifest = json.loads(data)
        except ValueError as error:
            raise SourceFailure(f"the manifest {path} is not JSON: {error}") from None
        from starfinder.workflow._shape import MANIFEST_FORMAT
        if not isinstance(manifest, dict) or manifest.get("format") != MANIFEST_FORMAT:
            raise SourceFailure(f"{path} is not a manifest of format {MANIFEST_FORMAT}")
        self.manifest = manifest
        self.record["manifest"]["sha256"] = _sha(data)
        self.record["roots"] = manifest.get("roots")
        tasks = {t["id"]: t for t in manifest["tasks"]}
        selected = self.entry.get("tasks", list(tasks))
        if not isinstance(selected, list) or not selected or any(t not in tasks for t in selected):
            raise ValueError(f"source {self.name!r}: tasks must be a nonempty list of the manifest's task ids")
        temporary = {g["name"]: set(g.get("temporary") or ()) for g in manifest.get("groups") or []}
        self.tasks, out = {}, {}
        for task_id in selected:
            task = tasks[task_id]
            exempt = {template.format(**task["identity"]) for template in temporary.get(task.get("group"), ())}
            record_file = _layout.task_record(manifest["roots"]["output"], task_id)
            record = None
            if record_file.is_file():
                raw = self.consume(record_file, checked_against=None)
                try:
                    record = json.loads(raw)
                except ValueError:
                    record = None
            problems = _problems(manifest, task, record, True, exempt) if record is not None or \
                not record_file.is_file() else ["the task record is not JSON"]
            self.tasks[task_id] = (task, record, exempt, problems)
            out[task_id] = {"kind": task["kind"], "complete": not problems, "problems": problems,
                            "record": None if record is None else
                            {"path": str(record_file), "sha256": self.consumed[str(record_file)]["sha256"],
                             "status": record.get("status")}}
        self.record["tasks"] = out
        self.record["incomplete"] = [k for k, v in out.items() if not v["complete"]]
        self.record["completion_rule"] = ("the workflow's completion and staleness rule with SHA-256 "
                                          "(starfinder workflow verify --hash), applied read-only")

    def open(self, spec):
        from starfinder.workflow._task import resolve_path
        task_id = spec.get("task")
        if task_id not in self.tasks:
            raise ValueError(f"an argument of workflow source {self.name!r} names a selected task, not {task_id!r}")
        task, record, exempt, problems = self.tasks[task_id]
        if problems:
            raise Skipped(f"workflow task {task_id} is not complete: {'; '.join(problems)}")
        declared = spec.get("output")
        listed = {entry.get("path"): entry for entry in record.get("outputs") or []}
        if declared not in listed:
            raise Unavailable(f"workflow task {task_id} lists no output {declared!r}")
        entry, path = listed[declared], resolve_path(self.manifest, declared)
        if declared.endswith("/"):
            item = next((i for i in entry.get("files") or [] if i.get("path") == spec.get("file")), None)
            if item is None:
                raise Unavailable(f"workflow output {declared} lists no file {spec.get('file')!r}")
            entry, path = item, path / item["path"]
        if declared in exempt and not path.is_file():
            raise Unavailable(f"the temporary workflow output {declared} of {task_id} is gone")
        where = f"source {self.name!r} task {task_id!r} output {declared!r}" + (
            f" file {spec.get('file')!r}" if declared.endswith("/") else "") + " field"
        digest, size = _digest(entry.get("sha256"), f"{where} sha256"), _size(entry.get("bytes"), f"{where} bytes")
        try:
            data = self.consume(path, digest, size, checked_against="task record")
        except Unavailable as error:
            raise SourceFailure(f"listed workflow output {error}") from None
        return _read_value(data, path, spec.get("read"), spec)


def _stage_files(directory, stage):
    """Every file of one checkpoint stage that is present, relative to the stage header's folder."""
    from starfinder._checkpoint import header_path
    from starfinder._records import TABLE_FORMATS
    folder = header_path(directory, stage).parent
    if stage == "registered":
        return folder, sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file())
    names = [f"{stage}.json"] + [f"{stage}.{f}" for f in TABLE_FORMATS if (folder / f"{stage}.{f}").is_file()]
    return folder, names


class _Checkpoint(_Source):
    kind = "checkpoint"
    keys = {"path", "stage", "inventory"}

    def roots(self):
        return [self._path("path")]

    def verify(self):
        from starfinder._checkpoint import STAGES, header_path, read_header
        self.directory = self._path("path")
        stage = self.entry.get("stage")
        if stage not in STAGES:
            raise ValueError(f"source {self.name!r}: stage must be one of {list(STAGES)}")
        self.stage = stage
        self.record.update(path=str(self.directory), stage=stage)
        header_file = header_path(self.directory, stage)
        try:
            header = read_header(self.directory, stage)
        except (OSError, ValueError) as error:
            raise SourceFailure(str(error)) from None
        version = header["format_version"]
        folder, present = _stage_files(self.directory, stage)
        inventory = self.entry.get("inventory")
        if inventory is not None:
            if not isinstance(inventory, list) or any(not isinstance(i, dict) or set(i) != {"path", "sha256"}
                                                      for i in inventory):
                raise ValueError(f"source {self.name!r}: inventory is a list of {{path, sha256}}")
            inventory = {i["path"]: _digest(i["sha256"], f"source {self.name!r} inventory {i['path']!r} field sha256")
                         for i in inventory}
        self.record.update(format_version=version, header={"path": str(header_file)},
                           channel_labels=header.get("channel_labels"), round_labels=header.get("round_labels"),
                           readout_mode=header.get("readout_mode"))
        header_name = header_file.relative_to(folder).as_posix()
        if version >= 3:
            self.record["payload_check"] = "the header's SHA-256 of each payload file"
            self.consume(header_file, inventory.get(header_name) if inventory else None,
                         checked_against="inventory" if inventory else None)
            for entry in header.get("files") or []:
                name = Path(entry["path"])
                if name.is_absolute() or ".." in name.parts:
                    raise SourceFailure(f"{stage} checkpoint names {entry['path']!r}, outside its folder")
                digest = _digest(entry.get("sha256"), f"source {self.name!r} header file {entry['path']!r} field sha256")
                try:
                    self.consume(folder / name, digest, checked_against="checkpoint header")
                except Unavailable as error:
                    raise SourceFailure(f"{stage} checkpoint payload {error}") from None
        else:
            self.limitations.append(f"a version-{version} checkpoint header holds no payload hashes: its files are "
                                    "checked only against the supplied external inventory")
            self.record["payload_check"] = "the supplied external inventory"
            if inventory is None:
                raise SourceFailure(f"the version-{version} {stage} checkpoint in {self.directory} holds no payload "
                                    "hashes; supply an inventory of every stage file with its SHA-256")
            missing = sorted(set(present) - set(inventory))
            if missing:
                raise SourceFailure(f"the inventory does not list the stage files {missing}")
            for name, digest in inventory.items():
                try:
                    self.consume(folder / name, digest, checked_against="inventory")
                except Unavailable as error:
                    raise SourceFailure(f"inventoried checkpoint file {error}") from None
        self.record["header"]["sha256"] = self.consumed[str(header_file)]["sha256"]
        self.loaded = None

    def open(self, spec):
        if self.loaded is None:
            from starfinder.dataset import load_checkpoint
            try:
                self.loaded = load_checkpoint(self.directory, self.stage)
            except (OSError, ValueError) as error:
                raise SourceFailure(f"load_checkpoint: {error}") from None
            frames = {}
            if self.stage == "registered":
                frames = {name: m.frame_id for name, m in self.loaded["metadata"].items()}
            elif self.stage == "candidates":
                frames = {"spots": self.loaded["spot_result"].metadata.frame_id}
            self.record["frames"] = frames
        return self.loaded


_KINDS = {"workflow": _Workflow, "checkpoint": _Checkpoint, "trial": _Trial, "files": _Files}


def _select(value, key):
    for part in key.split("."):
        if isinstance(value, dict):
            if part not in value:
                raise Unavailable(f"no {part!r} in {key!r}")
            value = value[part]
        elif isinstance(value, list) and part.isdigit():
            if int(part) >= len(value):
                raise Unavailable(f"no item {part} in {key!r}")
            value = value[int(part)]
        elif hasattr(value, part):
            value = getattr(value, part)
        else:
            raise Unavailable(f"no {part!r} in {key!r}")
    if value is None:
        raise Unavailable(f"{key!r} is absent (None) in this source")
    return value


def _argument(spec, sources, used):
    """One metric argument read from its source, with a description of what was read."""
    if not isinstance(spec, dict) or set(spec) - _ARGUMENT_KEYS:
        raise ValueError(f"an argument is a mapping of {sorted(_ARGUMENT_KEYS)}")
    if isinstance(spec.get("metadata"), dict):
        from starfinder.image import ImageMetadata
        return ImageMetadata(**tuples(spec["metadata"])), {"metadata": spec["metadata"]}
    if "mapping" in spec:
        # A mapping of labels (rounds, FOVs) to arguments, such as evaluate_translation's {round: ZYX}.
        values, described = {}, {}
        for label, item in spec["mapping"].items():
            values[label], described[label] = _argument(item, sources, used)
        return values, {"mapping": described}
    name = spec.get("source")
    source = sources[name]
    used.add(name)
    if source.error is not None:
        raise Skipped(f"source {name!r} failed: {source.error}")
    try:
        value = source.open(spec)
    except SourceFailure as error:
        source.error = {"type": "SourceFailure", "message": str(error)}
        raise Skipped(f"source {name!r} failed while it was read: {error}") from None
    described = dict(spec)
    if "key" in spec:
        value = _select(value, spec["key"])
    if "where" in spec:
        if not isinstance(value, pd.DataFrame) or not isinstance(spec["where"], dict):
            raise ValueError("where filters a table by {column: value or [values]}")
        before = len(value)
        for column, wanted in spec["where"].items():
            if column not in value:
                raise Unavailable(f"the table has no column {column!r}")
            value = value[value[column].isin(wanted if isinstance(wanted, list) else [wanted])]
        value = value.reset_index(drop=True)
        described["rows"] = {"before": before, "after": len(value)}
    if "columns" in spec or "column" in spec:
        if not isinstance(value, pd.DataFrame):
            raise ValueError("columns and column select from a table")
        wanted = spec.get("columns", [spec.get("column")])
        missing = [c for c in wanted if c not in value]
        if missing:
            raise Unavailable(f"the table has no columns {missing}")
        value = value[wanted].to_numpy(np.float64) if "columns" in spec else value[spec["column"]].to_numpy()
    return value, described


def _function(name):
    module, _, function = str(name).partition(".")
    if module not in METRIC_MODULES or not function:
        raise ValueError(f"function {name!r} must be <module>.<name> with a module of {list(METRIC_MODULES)}")
    target = importlib.import_module(f"starfinder.evaluation.{module}")
    if function not in getattr(target, "__all__", ()):
        raise ValueError(f"starfinder.evaluation.{module} has no public function {function!r}")
    return getattr(target, function)


def _metric_specs(path):
    specs = _mapping(path, "metrics")
    names = set()
    for spec in specs:
        if not isinstance(spec, dict) or set(spec) - {"name", "metric", "function", "parameters", "arguments"}:
            raise ValueError("a metric is a mapping of name, metric, function, parameters and arguments")
        name = spec.get("name")
        if not isinstance(name, str) or not _NAME.match(name) or name in names:
            raise ValueError(f"metric name {name!r} must be unique, of letters, digits, '_', '.' and '-'")
        names.add(name)
        function = _function(spec.get("function"))
        parameters, arguments = spec.get("parameters", {}), spec.get("arguments", {})
        if not isinstance(parameters, dict) or not isinstance(arguments, dict) or set(parameters) & set(arguments):
            raise ValueError(f"metric {name!r}: parameters and arguments are mappings without shared names")
        try:
            inspect.signature(function).bind(**dict.fromkeys(parameters), **dict.fromkeys(arguments))
        except TypeError as error:
            raise ValueError(f"metric {name!r}: {spec['function']}: {error}") from None
    return specs


def _source_specs(path):
    specs, sources = _mapping(path, "sources"), {}
    base = Path(path).resolve().parent
    for entry in specs:
        if not isinstance(entry, dict) or entry.get("kind") not in SOURCE_KINDS:
            raise ValueError(f"a source is a mapping with a kind of {list(SOURCE_KINDS)}")
        name = entry.get("name")
        if not isinstance(name, str) or not _NAME.match(name) or name in sources:
            raise ValueError(f"source name {name!r} must be unique, of letters, digits, '_', '.' and '-'")
        sources[name] = _KINDS[entry["kind"]](name, entry, base)
    return sources


def _check_argument(name, argument, sources):
    if isinstance(argument, dict) and isinstance(argument.get("metadata"), dict):
        return
    if isinstance(argument, dict) and isinstance(argument.get("mapping"), dict) and argument["mapping"]:
        for item in argument["mapping"].values():
            _check_argument(name, item, sources)
        return
    if not isinstance(argument, dict) or argument.get("source") not in sources:
        raise ValueError(f"metric {name!r}: each argument names a source of the sources file, or is "
                         "{metadata: {...}} or {mapping: {label: argument}}")


def _check_arguments(specs, sources):
    for spec in specs:
        for argument in spec.get("arguments", {}).values():
            _check_argument(spec["name"], argument, sources)


def _json(value):
    """Plain JSON of a metric's values and details: arrays and tables as lists, non-finite floats null."""
    if isinstance(value, np.ndarray):
        return _json(value.tolist())
    if isinstance(value, pd.DataFrame):
        return _json(value.to_dict(orient="records"))
    if isinstance(value, pd.Series):
        return _json(value.tolist())
    if isinstance(value, dict):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json(v) for v in value]
    value = jsonable(value)
    if isinstance(value, (dict, list)):
        return _json(value)
    if value is pd.NA or value is pd.NaT:
        return None
    return value if value is None or isinstance(value, (bool, int, float, str)) else str(value)


def _plain_result(value):
    """The JSON fields of a metric's return value: an EvaluationResult, a mapping or one value."""
    from starfinder.evaluation import EvaluationResult
    if isinstance(value, EvaluationResult):
        out = dict(status=value.status, values=dict(value.values), units=dict(value.units),
                   counts=dict(value.counts), reasons=dict(value.reasons), config=value.config,
                   details=value.details)
    else:
        values = dict(value) if isinstance(value, dict) else {"value": value}
        out = dict(status=None, values=values, units={}, counts={}, reasons={}, config={}, details={})
    out = _json(out)
    for key, item in out["values"].items():
        if item is None and key not in out["reasons"]:
            out["reasons"][key] = "the metric returned no value (undefined, or not finite)"
    if out["status"] is None:
        out["status"] = "ok" if all(v is not None for v in out["values"].values()) else "undefined"
    return out


def _within(path, roots):
    path = Path(path).resolve()
    return next((root for root in roots if path == root or path.is_relative_to(root)), None)


def evaluate(sources_path, metrics_path, output_dir):
    """Write a saved evaluation of the listed sources and metrics in ``output_dir``; return (its path, failed).

    ``failed`` is True when a source failed or a metric's outcome is not ``computed``; the evaluation is still
    written. Invalid sources or metrics files, an output folder that is not new or empty, or one inside a source,
    raise ValueError or OSError before anything is written.
    """
    sources_path, metrics_path = Path(sources_path).resolve(), Path(metrics_path).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f"the evaluation output must be a new or empty folder: {output_dir}")
    sources = _source_specs(sources_path)
    specs = _metric_specs(metrics_path)
    _check_arguments(specs, sources)
    inputs = {"sources": {"path": str(sources_path), "sha256": _sha(sources_path.read_bytes())},
              "metrics": {"path": str(metrics_path), "sha256": _sha(metrics_path.read_bytes())}}
    for source in sources.values():
        try:
            source.verify()
        except (SourceFailure, Unavailable, OSError) as error:
            source.error = {"type": type(error).__name__, "message": str(error)}
        root = _within(output_dir, source.roots())
        if root is not None:
            raise ValueError(f"the evaluation output {output_dir} is inside source {source.name!r} ({root})")
    metrics = []
    for spec in specs:
        used = set()
        record = {"name": spec["name"], "metric": spec.get("metric"), "function": spec["function"],
                  "parameters": spec.get("parameters", {}), "arguments": {}, "sources": []}
        try:
            arguments = {}
            for key, argument in spec.get("arguments", {}).items():
                arguments[key], record["arguments"][key] = _argument(argument, sources, used)
            try:
                result = _function(spec["function"])(**tuples(spec.get("parameters", {})), **arguments)
            except Exception as error:
                record.update(outcome="failed", error={"type": type(error).__name__, "message": str(error)})
            else:
                record.update(outcome="computed", **_plain_result(result))
        except Skipped as error:
            record.update(outcome="skipped", reason=str(error))
        except Unavailable as error:
            record.update(outcome="unavailable", reason=str(error))
        record["sources"] = sorted(used)
        metrics.append(record)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, record in enumerate(metrics):
        details = record.pop("details", None)
        if details is not None:
            name = f"metrics/{i:03d}-{record['name']}.json"
            write_json(details, output_dir / name)
            record["details_file"] = name
        for key, value in (record.get("values") or {}).items():
            rows.append({"metric": record["name"], "metric_id": record["metric"], "function": record["function"],
                         "outcome": record["outcome"], "status": record.get("status"), "value_name": key,
                         "value": value if value is None or isinstance(value, (bool, int, float, str))
                         else json.dumps(value), "unit": record["units"].get(key),
                         "reason": record["reasons"].get(key)})
        if record["outcome"] != "computed":
            rows.append({"metric": record["name"], "metric_id": record["metric"], "function": record["function"],
                         "outcome": record["outcome"], "status": None, "value_name": None, "value": None,
                         "unit": None, "reason": record.get("reason") or record["error"]["message"]})
    columns = ["metric", "metric_id", "function", "outcome", "status", "value_name", "value", "unit", "reason"]
    pd.DataFrame(rows, columns=columns).to_csv(output_dir / "values.csv", index=False)
    summaries = [source.summary() for source in sources.values()]
    counts = {outcome: sum(m["outcome"] == outcome for m in metrics) for outcome in OUTCOMES}
    failed = bool(any(s["status"] == "failed" for s in summaries) or counts["computed"] != len(metrics))
    evaluation = {"format": EVALUATION_FORMAT, "generated_at": datetime.now(timezone.utc).isoformat(),
                  "software": software(), "inputs": inputs, "sources": summaries, "metrics": metrics,
                  "counts": {"sources": len(summaries), "failed_sources": sum(s["status"] == "failed"
                                                                              for s in summaries),
                             "metrics": len(metrics), **counts},
                  "status": "incomplete" if failed else "complete"}
    write_json(evaluation, output_dir / "evaluation.json")
    return output_dir, failed
