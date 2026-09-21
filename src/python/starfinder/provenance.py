"""Versioned processing records, independent of benchmark execution manifests.

JSON contains metadata only. Numerical diagnostic arrays (including dense
transforms) are lossless, checksummed NPY components; no pickle is read.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version, PackageNotFoundError
import json
import math
import os
from pathlib import Path
import platform
import uuid
import warnings

import numpy as np

__all__ = ["RunRecorder", "read_run"]

_CONTRACT = "starfinder.artifacts/1"
_STAGES = {"prepared_input", "registered_images", "candidates_signals", "decoded_pre_qc", "final_accepted"}
_STAGE = {"register": "registered_images", "estimate_transform": "registered_images",
          "apply_transform": "registered_images", "find_spots": "candidates_signals",
          "_extract_round": "candidates_signals", "_assemble_intensities": "candidates_signals",
          "extract_intensities": "candidates_signals", "decode_barcodes": "decoded_pre_qc",
          "filter_reads": "final_accepted"}
_THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS", "CUDA_VISIBLE_DEVICES")
_FIELDS = {
    "run": "run_id created_at status dataset_id sample_id code environment config sources artifacts events failures saving_policy owner retention backup_status",
    "event": "event_id run_id sequence timestamp stage FOV round attempt operation outcome requested_method actual_method config_ref input_artifacts output_artifacts diagnostics failure_id",
    "failure": "failure_id run_id event_id stage FOV round category type message traceback_ref requested_method actual_method recovery_action recovery_event_id",
    "artifact": "artifact_id run_id stage status dataset_id sample_id FOV subtile parents config_ref source_refs components payload omission_reason failure_id",
}


class _SerializationError(TypeError):
    """Metadata could not be serialized, distinct from invalid stage input."""


def _header(kind):
    return dict(schema_name=f"starfinder.{kind}", schema_version=1, contract_id=_CONTRACT)


def _now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _unknown(reason):
    return {"value": None, "reason": reason}


def _environment(seed):
    dependencies = {}
    for name in ("numpy", "scipy", "pandas", "scikit-image", "SimpleITK", "tifffile"):
        try:
            dependencies[name] = version(name)
        except PackageNotFoundError:
            dependencies[name] = _unknown("not installed")
    return dict(python=platform.python_version(), dependencies=dependencies,
                backend="python", host=platform.node(), threads_devices={k: os.getenv(k) if k in os.environ else _unknown("not set") for k in _THREADS},
                seed=seed if seed is not None else _unknown("not supplied; no seed inferred"),
                resources=_unknown("allocation not supplied"))


def _require(condition, context):
    if not condition:
        raise ValueError(f"provenance: {context}")


def _string(value):
    return isinstance(value, str) and bool(value)


def _timestamp(value):
    _require(_string(value), "timestamp must be a UTC string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("provenance: invalid timestamp") from exc
    _require(parsed.utcoffset() is not None and parsed.utcoffset().total_seconds() == 0,
             "timestamp must be UTC")


def _record(record, kind):
    _require(isinstance(record, dict), f"{kind} must be an object")
    _require(record.get("schema_name") == f"starfinder.{kind}" and
             type(record.get("schema_version")) is int and record["schema_version"] == 1 and
             record.get("contract_id") == _CONTRACT, f"unsupported {kind} schema/version/contract")
    _require(set(_FIELDS[kind].split()) <= record.keys(), f"{kind} missing required fields")
    allowed = set(_FIELDS[kind].split()) | {"schema_name", "schema_version", "contract_id", "extensions"}
    _require(record.keys() <= allowed, f"{kind} unknown fields (use namespaced extensions)")
    if "extensions" in record:
        _require(isinstance(record["extensions"], dict) and all(_string(k) for k in record["extensions"]),
                 f"{kind} extensions must be namespaced objects")
    for key in ("run_id", f"{kind}_id"):
        if key in record:
            _require(_string(record[key]), f"{kind}.{key} must be nonempty")
    if kind != "run":
        _require(record["stage"] in _STAGES, f"{kind} invalid stage")
        _require(record["FOV"] is None or _string(record["FOV"]), f"{kind}.FOV")
    if kind in ("event", "failure"):
        for key in ("round", "requested_method", "actual_method", "failure_id"):
            if key in record:
                _require(record[key] is None or _string(record[key]), f"{kind}.{key}")
    if kind == "event":
        _timestamp(record["timestamp"])
        _require(type(record["sequence"]) is int and record["sequence"] >= 0, "event.sequence")
        _require(type(record["attempt"]) is int and record["attempt"] > 0, "event.attempt")
        _require(record["outcome"] in {"started", "succeeded", "failed", "skipped", "recovered"}, "event.outcome")
        _require(_string(record["operation"]) and isinstance(record["diagnostics"], dict), "event operation/diagnostics")
        for key in ("input_artifacts", "output_artifacts"):
            _require(isinstance(record[key], list) and all(_string(x) for x in record[key]), f"event.{key}")
        _require(_string(record["config_ref"]), "event.config_ref")
        _require((record["failure_id"] is not None) == (record["outcome"] == "failed"), "event failure/outcome mismatch")
    if kind == "failure":
        _require(record["category"] in {"invalid_input", "missing_input", "estimation", "application", "serialization", "integrity", "interrupted"}, "failure.category")
        for key in ("event_id", "type", "recovery_action"):
            _require(_string(record[key]), f"failure.{key}")
        _require(isinstance(record["message"], str), "failure.message")
        for key in ("traceback_ref", "recovery_event_id"):
            _require(record[key] is None or _string(record[key]), f"failure.{key}")
    if kind == "artifact":
        _require(record["status"] in {"complete", "omitted", "failed"}, "artifact.status")
        for key in ("dataset_id", "sample_id", "FOV", "config_ref"):
            _require(_string(record[key]), f"artifact.{key}")
        _require(record["subtile"] is None or type(record["subtile"]) is int and record["subtile"] > 0, "artifact.subtile")
        for key in ("parents", "source_refs", "components"):
            _require(isinstance(record[key], list), f"artifact.{key}")
        _require(isinstance(record["payload"], dict), "artifact.payload")
        if record["status"] == "complete":
            _require(record["omission_reason"] is None and record["failure_id"] is None, "complete artifact cannot have failure/omission")
            _require(bool(record["components"]), "complete artifact requires saved components (including empty tables)")
        elif record["status"] == "omitted":
            _require(not record["components"] and _string(record["omission_reason"]) and record["failure_id"] is None, "omitted artifact requires reason and no components")
        else:
            _require(_string(record["failure_id"]) and record["omission_reason"] is None, "failed artifact requires failure_id")


def _validate(run):
    _record(run, "run")
    _timestamp(run["created_at"])
    _require(run["status"] in {"running", "succeeded", "failed", "interrupted"}, "run.status")
    for key in ("dataset_id", "sample_id", "owner", "retention", "backup_status"):
        _require(_string(run[key]), f"run.{key}")
    for key in ("code", "environment", "config", "saving_policy"):
        _require(isinstance(run[key], dict), f"run.{key}")
    _require({"commit", "dirty", "patch_sha256", "snapshot_sha256", "package_version", "source_location"} <= run["code"].keys(), "run.code required fields")
    _require(run["code"]["dirty"] is None or type(run["code"]["dirty"]) is bool, "run.code.dirty")
    if run["code"]["commit"] is None or run["code"]["dirty"] is None:
        _require(_string(run["code"].get("unknown_reason")), "unknown code identity requires reason")
    if run["code"]["dirty"] is True:
        _require(_is_hash(run["code"]["patch_sha256"]) or _is_hash(run["code"]["snapshot_sha256"]), "dirty code requires patch/snapshot checksum")
    _require({"python", "dependencies", "backend", "host", "threads_devices", "seed", "resources"} <= run["environment"].keys(), "run.environment required fields")
    _require({"requested", "effective", "operations"} <= run["config"].keys(), "run.config requested/effective/operations")
    _require(isinstance(run["config"]["operations"], dict), "run.config.operations")
    for key in ("sources", "artifacts", "events", "failures"):
        _require(isinstance(run[key], list), f"run.{key}")
    source_ids = set()
    for source in run["sources"]:
        _require(isinstance(source, dict) and {"source_id", "catalog", "uri", "sha256", "unverified_reason", "selection"} <= source.keys(), "source required fields")
        _require(_string(source["source_id"]) and source["source_id"] not in source_ids, "source identity")
        source_ids.add(source["source_id"])
        _require(isinstance(source["selection"], dict), "source.selection")
        _require(source["sha256"] is None and _string(source["unverified_reason"]) or _is_hash(source["sha256"]), "source checksum or unverified reason")
    groups = {}
    all_ids = set()
    for kind, plural in (("artifact", "artifacts"), ("event", "events"), ("failure", "failures")):
        group = {}
        for record in run[plural]:
            _record(record, kind)
            identity = record[f"{kind}_id"]
            _require(identity not in all_ids and record["run_id"] == run["run_id"], f"duplicate/foreign {kind} identity {identity}")
            all_ids.add(identity)
            group[identity] = record
        groups[kind] = group
    previous = -1
    config_refs = {"config/effective"} | {f"config/operations/{key}" for key in run["config"]["operations"]}
    for event in run["events"]:
        _require(event["sequence"] > previous, "event sequence must increase")
        previous = event["sequence"]
        _require(event["config_ref"] in config_refs, "event config_ref unresolved")
        for identity in event["input_artifacts"] + event["output_artifacts"]:
            _require(identity in groups["artifact"], "event artifact reference unresolved")
        if event["failure_id"] is not None:
            _require(event["failure_id"] in groups["failure"] and groups["failure"][event["failure_id"]]["event_id"] == event["event_id"], "event failure reference unresolved")
    for failure in run["failures"]:
        _require(failure["event_id"] in groups["event"] and groups["event"][failure["event_id"]]["failure_id"] == failure["failure_id"], "failure event reference unresolved")
        recovery = failure["recovery_event_id"]
        _require(recovery is None or recovery in groups["event"] and groups["event"][recovery]["outcome"] == "recovered", "failure recovery reference unresolved")
    for artifact in run["artifacts"]:
        _require(artifact["dataset_id"] == run["dataset_id"] and artifact["sample_id"] == run["sample_id"], "artifact dataset/sample mismatch")
        _require(artifact["config_ref"] in config_refs, "artifact config_ref unresolved")
        _require(all(x in source_ids for x in artifact["source_refs"]), "artifact source reference unresolved")
        if artifact["failure_id"] is not None:
            _require(artifact["failure_id"] in groups["failure"], "artifact failure reference unresolved")
        for parent in artifact["parents"]:
            _require(isinstance(parent, dict) and all(_string(parent.get(k)) for k in ("run_id", "artifact_id")) and _is_hash(parent.get("sha256")), "artifact parent identity/checksum")
    if run["status"] == "succeeded":
        _require(not any(f["recovery_event_id"] is None for f in run["failures"]), "successful run has unrecovered failure")
        _require(not any(a["status"] == "failed" for a in run["artifacts"]), "successful run has failed artifact")
    if run["status"] in ("failed", "interrupted"):
        _require(bool(run["failures"]), "failed/interrupted run requires failure record")


def _is_hash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _component(root, descriptor):
    _require(isinstance(descriptor, dict) and {"component_id", "path", "format", "size", "sha256"} <= descriptor.keys(), "component required fields")
    _require(_string(descriptor["component_id"]) and _string(descriptor["path"]), "component identity/path")
    relative = Path(descriptor["path"])
    _require(not relative.is_absolute() and ".." not in relative.parts and relative.parts, "component path escapes root")
    path = (root / relative).resolve()
    _require(path.is_relative_to(root.resolve()), "component symlink escapes root")
    _require(type(descriptor["size"]) is int and descriptor["size"] >= 0 and _is_hash(descriptor["sha256"]), "component size/checksum type")
    if not path.is_file():
        raise FileNotFoundError(f"component {descriptor['component_id']}: {path}")
    _require(path.stat().st_size == descriptor["size"] and _hash(path) == descriptor["sha256"], f"component {descriptor['component_id']} integrity failure")
    return path


def _decode(value, root, components):
    if isinstance(value, list):
        return [_decode(x, root, components) for x in value]
    if not isinstance(value, dict):
        _require(not isinstance(value, float) or math.isfinite(value), "untagged nonfinite number")
        return value
    if "float_special" in value:
        _require(set(value) == {"float_special"} and value["float_special"] in {"nan", "+inf", "-inf"}, "invalid special float")
        return {"nan": float("nan"), "+inf": float("inf"), "-inf": -float("inf")}[value["float_special"]]
    if "array_component" in value:
        _require(set(value) == {"array_component"}, "invalid array descriptor")
        descriptor = value["array_component"]
        path = _component(root, descriptor)
        identity = descriptor["component_id"]
        _require(identity not in components or components[identity] == descriptor, "conflicting component identity")
        components[identity] = descriptor
        _require(descriptor["format"] == "npy" and "shape" in descriptor and "dtype" in descriptor, "unsupported array component")
        try:
            array = np.load(path, allow_pickle=False)
        except Exception as exc:
            raise ValueError(f"component {identity}: invalid NPY") from exc
        _require(isinstance(array, np.ndarray) and list(array.shape) == descriptor["shape"] and array.dtype.str == descriptor["dtype"], f"component {identity} shape/dtype mismatch")
        return array
    decoded = {k: _decode(v, root, components) for k, v in value.items()}
    if set(decoded) == {"type", "fields"}:
        _require(_string(decoded['type']) and isinstance(decoded['fields'], dict), "typed metadata requires type/fields")
        if decoded['type'] == 'ImageMetadata':
            from starfinder.image import ImageMetadata
            expected = {f.name for f in fields(ImageMetadata)}
            _require(set(decoded['fields']) == expected, "ImageMetadata required fields")
            try:
                ImageMetadata(**decoded['fields'])
            except (TypeError, ValueError) as exc:
                raise ValueError("provenance: invalid ImageMetadata geometry") from exc
        if decoded['type'] in ('TranslationTransform', 'DenseDisplacementTransform'):
            from starfinder.image import ImageMetadata
            from starfinder.registration import TranslationTransform, DenseDisplacementTransform
            transform_type = {'TranslationTransform': TranslationTransform,
                              'DenseDisplacementTransform': DenseDisplacementTransform}[decoded['type']]
            _require(set(decoded['fields']) == {f.name for f in fields(transform_type)}, "transform required fields")
            parameters = dict(decoded['fields'])
            try:
                for key in ('reference_metadata', 'moving_metadata'):
                    _require(parameters[key]['type'] == 'ImageMetadata', 'transform metadata type')
                    parameters[key] = ImageMetadata(**parameters[key]['fields'])
                transform_type(**parameters)
            except (TypeError, ValueError, KeyError) as exc:
                raise ValueError('provenance: invalid transform geometry/direction/values') from exc
    return decoded


def read_run(path: str | Path, *, sha256: str | None = None) -> dict:
    """Read and validate a run and every linked diagnostic component.

    Parameters
    ----------
    path : str or pathlib.Path
        Run directory or its run.json. Running/interrupted records are inspectable,
        but are never promoted to success. No processing algorithms or arbitrary
        constructors run; known geometry/transform validators check metadata.
    sha256 : str or None
        Optional expected hash of the exact manifest bytes (external handoff).

    Returns
    -------
    dict
        Validated records. Tagged floats and diagnostic NPY arrays are restored;
        typed config/geometry records remain dictionaries with type names.

    Raises
    ------
    ValueError
        Malformed/unsupported schema, unresolved identity or integrity mismatch.
    FileNotFoundError
        Manifest or referenced component is absent. No regeneration occurs.
    """
    path = Path(path)
    if path.is_dir():
        path = path / "run.json"
    if sha256 is not None:
        _require(_is_hash(sha256) and _hash(path) == sha256, "run manifest integrity failure")
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, f"duplicate JSON key {key}")
            result[key] = value
        return result
    def constant(value):
        raise ValueError(f"provenance: untagged nonfinite JSON number {value}")
    run = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs, parse_constant=constant)
    try:
        _validate(run)
        components = {}
        for artifact in run["artifacts"]:
            seen = set()
            for descriptor in artifact["components"]:
                _component(path.parent, descriptor)
                _require(descriptor["component_id"] not in seen, "duplicate artifact component ID")
                seen.add(descriptor["component_id"])
        return _decode(run, path.parent, components)
    except (TypeError, KeyError) as exc:
        raise ValueError(f"provenance: malformed record field: {exc}") from exc


class RunRecorder:
    """One immutable-identity FOV run, written incrementally to a fresh directory.

    Pass this object to ``FOV.run(..., provenance=recorder)``. It can be used
    once. Updates publish run.json atomically; completed runs are never replaced.
    This records processing state, not image/table checkpoints or a scheduler.

    Parameters
    ----------
    directory : str or pathlib.Path
        Fresh directory; an existing path raises FileExistsError.
    dataset_id, sample_id : str
        Explicit identities, checked against the FOV before processing.
    code : dict or None
        Commit, dirty, patch_sha256, snapshot_sha256, package_version and
        source_location. Unknown supplied fields need an unknown_reason.
        None records unknown revision/dirty state without invoking Git.
    sources : sequence of dict
        Source ID, catalog, URI, SHA-256 or unverified_reason, and selection.
        Original source geometry belongs in selection; paths are never hashed
        or accessed implicitly. In-memory input identities are added at binding.
    seed : object or None
        Supplied seed/stream identity; None remains explicitly unknown.
    owner, retention : str
        Artifact ownership and retention policy; no automatic deletion.
    """

    def __init__(self, directory, *, dataset_id, sample_id, code=None, sources=(),
                 seed=None, owner="unspecified", retention="owner decision required"):
        import starfinder
        self.directory = Path(directory)
        self._used = False
        self._arrays = {}
        self._attempts = {}
        self._failure_exceptions = {}
        self._fov = None
        self._run = dict(_header("run"), run_id=uuid.uuid4().hex, created_at=_now(), status="running",
            dataset_id=dataset_id, sample_id=sample_id,
            code=copy.deepcopy(code) if code is not None else dict(commit=None, dirty=None, patch_sha256=None,
                snapshot_sha256=None, package_version=starfinder.__version__, source_location=starfinder.__file__,
                unknown_reason="revision and dirty state not supplied"),
            environment=_environment(copy.deepcopy(seed)), config=dict(requested=None, effective=None, operations={}),
            sources=copy.deepcopy(list(sources)), artifacts=[], events=[], failures=[],
            saving_policy=dict(provenance=True, diagnostics=True, images=False, candidates_signals=True,
                               decoded_pre_qc=True, final_accepted=True,
                               payload_writer="separate checkpoint APIs; no implicit payload save"),
            owner=owner, retention=retention, backup_status="unverified", extensions={"starfinder.provenance": {}})
        _validate(self._run)
        self.directory.mkdir(parents=True, exist_ok=False)
        self._publish()

    @property
    def path(self) -> Path:
        """Location of the latest atomic run manifest."""
        return self.directory / "run.json"

    @property
    def run_id(self) -> str:
        """Unique immutable identity for references from checkpoint writers."""
        return self._run["run_id"]

    def record_artifact(self, artifact: dict) -> None:
        """Link a v1 artifact record during this run, after its writer validates it.

        Components must already exist under this run directory. Their stored
        bytes, size and checksum are verified before publishing the link.
        This method validates metadata, not format-specific image/table payload
        invariants; the owning checkpoint writer remains responsible for those.
        Records cannot be replaced, and terminal runs cannot be modified.
        """
        _require(self._run["status"] == "running", "terminal run is immutable")
        encoded = self._encode(artifact)
        _record(encoded, "artifact")
        for descriptor in encoded["components"]:
            _component(self.directory, descriptor)
        candidate = dict(self._run, artifacts=self._run["artifacts"] + [encoded])
        _validate(self._encode(candidate))
        self._run = candidate
        self._publish()

    def _loaded_sources(self, name, loaded, config):
        layers = loaded.diagnostics.get('source_layers', [])
        for index, (channel, path) in enumerate(zip(loaded.channel_labels, loaded.source_paths)):
            self._run["sources"].append(dict(source_id=uuid.uuid4().hex, catalog=None,
                uri=str(path.resolve()), sha256=_hash(path), unverified_reason=None,
                selection=self._encode(dict(round=name, channel=channel, config=config,
                    load_diagnostics=loaded.diagnostics, output_geometry=loaded.metadata,
                    original_geometry=layers[index]['metadata'] if layers else _unknown("not exposed separately by round loader")))))
        self._publish()

    def _encode(self, value):
        if is_dataclass(value) and not isinstance(value, type):
            return {"type": type(value).__name__, "fields": {f.name: self._encode(getattr(value, f.name)) for f in fields(value)}}
        if isinstance(value, np.ndarray):
            _require(not value.dtype.hasobject, "object arrays cannot be persisted")
            # Hash includes shape/dtype as well as bytes, without changing values.
            key = hashlib.sha256(str((value.shape, value.dtype.str)).encode() + value.tobytes()).hexdigest()
            if key not in self._arrays:
                name = f"diagnostic-{key}.npy"
                path = self.directory / name
                with path.open("xb") as stream:
                    np.save(stream, value, allow_pickle=False)
                self._arrays[key] = dict(component_id=key, path=name, format="npy", size=path.stat().st_size,
                                         sha256=_hash(path), shape=list(value.shape), dtype=value.dtype.str)
            return {"array_component": self._arrays[key]}
        if isinstance(value, np.generic):
            return self._encode(value.item())
        if isinstance(value, float) and not math.isfinite(value):
            return {"float_special": "nan" if math.isnan(value) else "+inf" if value > 0 else "-inf"}
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, type) and issubclass(value, Exception):
            return {"exception_type": value.__name__}  # never imported/constructed on read
        if isinstance(value, dict):
            _require(all(isinstance(k, str) for k in value), "metadata keys must be strings")
            return {k: self._encode(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [self._encode(v) for v in value]
        if value is None or type(value) in (str, int, float, bool):
            return value
        raise _SerializationError(f"unsupported provenance value: {type(value).__name__}")

    def _publish(self):
        encoded = self._encode(self._run)
        _validate(encoded)
        content = json.dumps(encoded, indent=2, allow_nan=False).encode("utf-8")
        temporary = self.directory / "run.json.tmp"
        with temporary.open("wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(self.path)

    def _event(self, operation, outcome, *, stage=None, round_name=None, attempt=1,
               requested=None, actual=None, config_ref="config/effective", diagnostics=None, failure_id=None):
        event = dict(_header("event"), event_id=uuid.uuid4().hex, run_id=self._run["run_id"],
            sequence=len(self._run["events"]), timestamp=_now(), stage=stage or _STAGE.get(operation, "prepared_input"),
            FOV=self._fov.fov_id if self._fov is not None else None, round=round_name,
            attempt=attempt, operation=operation, outcome=outcome, requested_method=requested,
            actual_method=actual, config_ref=config_ref, input_artifacts=[], output_artifacts=[],
            diagnostics=diagnostics or {}, failure_id=failure_id)
        self._run["events"].append(event)
        return event

    def _failure(self, event, error, category=None):
        from starfinder.registration import RegistrationEstimationError
        previous = self._failure_exceptions.get(id(error))
        category = ("interrupted" if not isinstance(error, Exception) else category) or (previous["category"] if previous else None) or (
            "missing_input" if isinstance(error, FileNotFoundError) else
            "serialization" if isinstance(error, _SerializationError) else
            "estimation" if isinstance(error, RegistrationEstimationError) else "invalid_input")
        failure = dict(_header("failure"), failure_id=uuid.uuid4().hex, run_id=self._run["run_id"],
            event_id=event["event_id"], stage=event["stage"], FOV=event["FOV"], round=event["round"],
            category=category, type=type(error).__name__, message=str(error), traceback_ref=None,
            requested_method=event["requested_method"], actual_method=event["actual_method"],
            recovery_action="none", recovery_event_id=None)
        event["failure_id"] = failure["failure_id"]
        self._run["failures"].append(failure)
        self._failure_exceptions[id(error)] = failure
        return failure

    @contextmanager
    def _operation(self, operation, config, *, round_name=None, requested=None, actual=None):
        stage = _STAGE.get(operation, "prepared_input")
        key = (stage, round_name)
        attempt = self._attempts.get(key, 0) + 1
        self._attempts[key] = attempt
        config_id = uuid.uuid4().hex
        self._run["config"]["operations"][config_id] = self._encode(config)
        arguments = dict(stage=stage, round_name=round_name, attempt=attempt, requested=requested,
                         actual=actual, config_ref=f"config/operations/{config_id}")
        self._event(operation, "started", **arguments)
        self._publish()
        diagnostics = {"method_reason": "not a method operation"} if requested is None else {}
        try:
            with warnings.catch_warnings(record=True) as caught:
                try:
                    yield diagnostics
                finally:
                    diagnostics["warnings"] = [dict(type=w.category.__name__, message=str(w.message)) for w in caught]
            outcome = "succeeded"
        except BaseException as error:
            event = self._event(operation, "failed", diagnostics=diagnostics, **arguments)
            self._failure(event, error, "application" if operation == "apply_transform" else None)
            self._publish()
            raise
        else:
            self._event(operation, outcome, diagnostics=diagnostics, **arguments)
            self._publish()
        finally:
            # Retain normal caller visibility in addition to structured warnings.
            for warning in caught:
                warnings.warn_explicit(warning.message, warning.category, warning.filename, warning.lineno)

    def _recover(self, error, operation, *, round_name, requested, actual):
        failure = self._failure_exceptions[id(error)]
        event = self._event(operation, "recovered", round_name=round_name,
            attempt=self._attempts[("registered_images", round_name)], requested=requested, actual=actual,
            diagnostics={"original_failure_id": failure["failure_id"]})
        failure.update(recovery_action="configured_alternative", recovery_event_id=event["event_id"])
        self._publish()

    def _snapshot(self, fov):
        return dict(rounds=fov.rounds, channel_labels=fov.dataset.channel_order, subtile=fov.subtile_id,
            geometry=fov.metadata, load_diagnostics=fov.load_diagnostics,
            registration_state={name: ("reference_unchanged" if name == fov.rounds.reference_round else
                "failed" if fov.registration_attempts.get(name) and fov.registration_attempts[name][-1]['outcome'] != 'succeeded' else
                "applied" if fov.registration_results.get(name) else "not_registered") for name in fov.rounds.all_rounds},
            registration_attempts=fov.registration_attempts, registration_results=fov.registration_results,
            images={name: dict(shape=list(a.shape), dtype=a.dtype.str) for name, a in fov.images.items()},
            detected=len(fov.spot_result.spots) if fov.spot_result is not None else None,
            detection_config=fov.spot_result.config if fov.spot_result is not None else None,
            detection_diagnostics=fov.spot_result.diagnostics if fov.spot_result is not None else None,
            extraction_config=fov.intensity_result.config if fov.intensity_result is not None else None,
            extraction_diagnostics=fov.intensity_result.diagnostics if fov.intensity_result is not None else None,
            decoding_config=fov.decoding_result.config if fov.decoding_result is not None else None,
            filtering_config=fov.filtering_result.config if fov.filtering_result is not None else None,
            counts=fov.filtering_result.counts if fov.filtering_result is not None else None,
            fractions=fov.filtering_result.fractions if fov.filtering_result is not None else None,
            partial_round_signals=list(fov._round_intensities))

    def _stage_records(self, fov):
        """Explicit absence/failure records, never pretend unsaved payloads exist."""
        extension = self._run["extensions"]["starfinder.provenance"]
        extension["stage_state"] = {}
        for stage in sorted({e["stage"] for e in self._run["events"]}):
            events = [e for e in self._run["events"] if e["stage"] == stage]
            failures = [f for f in self._run["failures"] if f["stage"] == stage and f["recovery_event_id"] is None]
            state = "failed" if failures else "succeeded"
            if not failures and stage == 'candidates_signals' and (
                    fov.intensity_result is None or fov._round_intensities):
                state = 'partial'
            extension["stage_state"][stage] = state
            if any(a["stage"] == stage for a in self._run["artifacts"]):
                continue
            artifact = dict(_header("artifact"), artifact_id=uuid.uuid4().hex, run_id=self.run_id,
                stage=stage, status="failed" if failures else "omitted", dataset_id=self._run["dataset_id"],
                sample_id=self._run["sample_id"], FOV=fov.fov_id, subtile=fov.subtile_id, parents=[],
                config_ref="config/effective", source_refs=[s["source_id"] for s in self._run["sources"]],
                components=[], payload={"operation_event_ids": [e["event_id"] for e in events], "stage_state": state},
                omission_reason=None if failures else "incomplete_stage" if state == 'partial' else "payload_not_saved_by_provenance_recorder",
                failure_id=failures[-1]["failure_id"] if failures else None)
            self._run["artifacts"].append(artifact)
            events[-1]["output_artifacts"].append(artifact["artifact_id"])

    @contextmanager
    def _bind(self, fov, config, execution):
        _require(not self._used, "RunRecorder is single-use; create a fresh run")
        _require((fov.dataset.dataset_id, fov.dataset.sample_id) == (self._run["dataset_id"], self._run["sample_id"]), "FOV dataset/sample differs from run")
        _require(getattr(fov, "_provenance", None) is None, "FOV already has an active recorder")
        self._used = True
        self._fov = fov
        extension = self._run["extensions"]["starfinder.provenance"]
        try:
            self._run["config"]["requested"] = self._encode(dict(pipeline=config, execution=execution))
            self._run["config"]["effective"] = self._run["config"]["requested"]
            extension["initial_state"] = self._encode(self._snapshot(fov))
            if fov.codebook is not None:
                book = fov.codebook
                # A locator/identity, not a replacement for the checkpoint's table.
                extension['codebook'] = self._encode(dict(round_labels=book.round_labels,
                    channel_labels=book.channel_labels, color_to_channel=book.color_to_channel,
                    encoding=book.encoding, columns=list(book.table.columns), rows=len(book.table),
                    table_sha256=hashlib.sha256(book.table.to_csv(index=False).encode('utf-8')).hexdigest(),
                    checksum_scope='UTF-8 pandas CSV with index=False; table payload not saved'))
            for name, array in fov.images.items():
                self._run["sources"].append(dict(source_id=uuid.uuid4().hex, catalog=None,
                    uri=f"memory:{fov.fov_id}/{name}", sha256=hashlib.sha256(array.tobytes()).hexdigest(),
                    unverified_reason=None, selection=self._encode(dict(round=name, axes="ZYXC" if array.ndim == 4 else "ZYX",
                        shape=array.shape, dtype=array.dtype.str, metadata=fov.metadata.get(name),
                        checksum_scope="C-order array bytes; upstream acquisition unverified"))))
            self._publish()
            fov._provenance = self
            yield
        except BaseException as error:
            self._run["status"] = "failed" if isinstance(error, Exception) else "interrupted"
            if id(error) not in self._failure_exceptions:
                event = self._event("run", "failed")
                self._failure(event, error, "serialization" if isinstance(error, (OSError, _SerializationError)) else None)
            raise
        else:
            self._run["status"] = "succeeded"
        finally:
            fov._provenance = None
            try:
                extension["final_state"] = self._encode(self._snapshot(fov))
                self._stage_records(fov)
                self._publish()
            except BaseException:
                # A failed persistence write must never leave a published success.
                self._run["status"] = "failed"
                raise
