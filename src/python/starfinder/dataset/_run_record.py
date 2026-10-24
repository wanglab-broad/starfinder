"""Small per-FOV run record (run.json) in the FOV folder, written while checkpoints are active.

Version 2 (docs/checkpoints.md, "run.json") is portable: the output and stage
paths are relative to the FOV folder, and the files the run reads are relative
to the input root when they lie inside it, else absolute; every file it reads
has its SHA-256. ``input_root`` and ``output_root`` record the roots of the run
as provenance. ``checkpoints.<stage>`` names the stage record with its SHA-256,
and the stage record names and hashes its payload files. Two meanings are kept
apart: ``status`` is the outcome of ``FOV.run``'s processing, written when the
run ends, and ``outputs`` lists the files written after it (molecules, reads,
the reference image, the subtile files), each added as it is written. Version
1 recorded the paths as given and linked the stages by file names; its readers
read it as recorded.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import traceback

from starfinder import _records
from starfinder._layout import CHECKPOINTS, REGISTERED, RUN_RECORD, checkpoint_dir
from starfinder._records import file_sha256, software

logger = logging.getLogger("starfinder")

FORMAT_VERSION = 2
#: The run.json versions the readers accept: version 1 records paths as given and its stages by file names only.
READ_VERSIONS = (1, 2)


def source_path(path, input_root) -> str:
    """A file a run reads, as version 2 records it: relative to ``input_root`` when inside it, else absolute.

    Lexical (``os.path.abspath``, no symbolic link is resolved); POSIX
    separators for a relative path.
    """
    path = os.path.abspath(path)
    if input_root is not None:
        root = os.path.abspath(input_root)
        if os.path.commonpath([path, root]) == root:
            return Path(os.path.relpath(path, root)).as_posix()
    return path


def resolve_source(record, path, input_root=None) -> Path | None:
    """The file a run.json path of a source names (``inputs``, ``codebook``, ``workflow_config``, supplied statistics).

    A version-1 record keeps the path as the run was given it, read as
    recorded. In version 2 an absolute path is read as it is and a relative
    one is joined to ``input_root`` (the root supplied at read time) or, without
    one, to the record's ``input_root``; None when neither is given, so that
    no relative path is opened against the current folder.
    """
    if path is None:
        return None
    if record.get("format_version", 1) == 1 or Path(path).is_absolute():
        return Path(path)
    root = input_root if input_root is not None else record.get("input_root")
    return None if root is None else Path(root) / path


def output_entry(record_path, path, **extra) -> dict:
    """An ``outputs`` entry: the file's path relative to the folder of ``record_path``, its SHA-256, then ``extra``."""
    relative = os.path.relpath(Path(path).resolve(), Path(record_path).parent.resolve())
    return {"path": Path(relative).as_posix(), "sha256": file_sha256(path), **extra}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _codebook_record(dataset):
    """The codebook (multiplexed) or direct panel (direct) of the run: kind, file, SHA-256, entries, genes; or None.

    path and sha256 are those Dataset.load_codebook or load_direct_panel kept for
    the object the dataset holds now (path as source_path records it); null for
    one set without a file.
    """
    kind = "codebook" if dataset.readout_mode == "multiplexed" else "direct_panel"
    reference = dataset.codebook if kind == "codebook" else dataset.direct_panel
    if reference is None:
        return None
    source = dataset._reference_files.get(kind)
    path, sha256 = (source[1], source[2]) if source is not None and source[0] is reference else (None, None)
    if path is not None:
        path = source_path(path, dataset.input_root)
    entries = reference.n_entries if kind == "codebook" else len(reference.table)
    return {"kind": kind, "path": path, "sha256": sha256, "entries": entries, "genes": reference.n_genes}


def _workflow_config_record(dataset):
    """Path and SHA-256 of the workflow configuration file from_workflow_config translated; None outside the workflow."""
    path = dataset._workflow_config
    if path is None:
        return None
    # A configuration whose config_path names no readable file keeps its path with a null hash.
    return {"path": source_path(path, dataset.input_root),
            "sha256": file_sha256(path) if Path(path).is_file() else None}


def _transforms(fov, written):
    """R1: per registered moving round, the entries of transforms.json (_transform_json) and its WarpConfig.

    written lists the files the stage registered wrote, relative to the FOV
    folder; a dense field or B-spline coefficients are named only when that
    stage wrote the round's field file.
    """
    from starfinder._checkpoint import _transform_json
    rounds = set(fov._pipeline_rounds())
    applications = fov.registration_record.get("application", {})
    transforms = {}
    for name, results in fov.registration_results.items():
        # Other rounds (prepare_morphology, register on other rounds) are not processed by FOV.run.
        if name not in rounds or not results:
            continue
        field_name = f"{name}_field.npz" if f"{CHECKPOINTS}/{REGISTERED}/{name}_field.npz" in written else None
        transforms[name] = {"reference": fov.rounds.reference_round,
                            "results": [_transform_json(r, i, field_name) for i, r in enumerate(results)],
                            "application_config": applications.get(name, results[-1].application_config)}
    return transforms


def add_output(path, slot, entry):
    """Set ``outputs.<slot>`` of a written run.json to ``entry`` and rewrite the record atomically.

    The writers of the outputs (``FOV.export_spots``, ``save_reference_image``,
    ``save_subtiles``, the subtile rules' tables) call it after the FOV's latest
    checkpointed run succeeded; the other keys, ``status`` among them, and
    ``format_version`` are kept.
    """
    import json
    data = json.loads(Path(path).read_text())
    data.setdefault("outputs", {})[slot] = entry
    _records.write_json(data, path)


class _RunRecord:
    """Mutable run.json content; rewritten atomically at step boundaries.

    directory is the FOV folder, which holds run.json; the stage files go to its
    checkpoint folder. ``written`` lists each stage's files relative to the FOV
    folder as they are written, and ``registered`` the payload entries of the
    registered rounds written so far (path relative to the stage record's
    folder, SHA-256); run.json names each stage by its record and that record's
    SHA-256.
    """

    def __init__(self, fov, directory, checkpoints, config, execution):
        self.fov = fov
        self.directory = Path(directory)
        self.checkpoint_directory = checkpoint_dir(self.directory)
        self.checkpoints = checkpoints
        self.failure = None
        self._methods = None
        self.written = {}
        self.registered = []
        dataset = fov.dataset
        self.input_root = dataset.input_root
        # The parent of the FOV folders, when the configuration names one, relative to the FOV folder.
        stored = dict(asdict(checkpoints), directory=None if checkpoints.directory is None else
                      Path(os.path.relpath(os.path.abspath(checkpoints.directory),
                                           os.path.abspath(self.directory))).as_posix())
        from starfinder.readout.codebook import _encoding_record
        # The codebook's encoding (null without a codebook or in readout mode direct).
        encoding = (_encoding_record(dataset.codebook.encoding)
                    if dataset.codebook is not None and dataset.readout_mode == "multiplexed" else None)
        self.data = {
            "format_version": FORMAT_VERSION, "dataset_id": dataset.dataset_id,
            "sample_id": dataset.sample_id, "fov_id": fov.fov_id, "subtile_id": fov.subtile_id,
            # The roots of the run, absolute, as provenance; the record's paths do not depend on them.
            "input_root": None if dataset.input_root is None else os.path.abspath(dataset.input_root),
            "output_root": os.path.abspath(dataset.output_root),
            "status": "running", "started_at": _now(), "ended_at": None, "error": None,
            **software(),
            # The dataset's readout mode (docs/readout-contract.md) is recorded with the configs.
            "config": {"pipeline": config, "execution": execution, CHECKPOINTS: stored,
                       "readout_mode": dataset.readout_mode, "encoding": encoding},
            # The workflow configuration file of a workflow rule (null outside the workflow).
            "workflow_config": _workflow_config_record(dataset),
            # Channel pattern, name and wavelength per round (docs/checkpoints.md, "run.json").
            "channels": dataset._channel_record(),
            # The round roles of the run (RoundState).
            "rounds": asdict(fov.rounds),
            "inputs": [], "codebook": _codebook_record(dataset), "steps": [], "preprocessing": None,
            "registration": {}, "transforms": {}, "counts": {},
            # The checkpoint folder and each stage's record with its SHA-256, relative to the FOV folder.
            "checkpoint_directory": CHECKPOINTS, CHECKPOINTS: {},
            # The files written after the processing, by slot (add_output).
            "outputs": {},
        }

    @property
    def path(self):
        return self.directory / RUN_RECORD

    def add_inputs(self, paths):
        """Record loaded source files (relative to the input root), hashing them unless hash_inputs is False."""
        for path in paths:
            self.data["inputs"].append({"path": source_path(path, self.input_root),
                "sha256": file_sha256(path) if self.checkpoints.hash_inputs else None})

    def add_checkpoint(self, stage, files):
        """Record a stage's files, given relative to the checkpoint folder; name a written stage record with its hash.

        ``checkpoints.<stage>`` is ``{path, sha256}`` of the stage record
        (``candidates.json``, ``pre_qc.json``, ``registered/transforms.json``)
        once it is among ``files``.
        """
        from starfinder._checkpoint import header_path
        self.written.setdefault(stage, []).extend(f"{CHECKPOINTS}/{name}" for name in files)
        header = header_path(self.checkpoint_directory, stage)
        name = header.relative_to(self.checkpoint_directory).as_posix()
        if name in files:
            self.data[CHECKPOINTS][stage] = {"path": f"{CHECKPOINTS}/{name}", "sha256": file_sha256(header)}

    def add_methods(self, entries):
        """Provenance entries of the methods the current step ran; the step's record gets them as methods."""
        self._methods = list(entries)

    def add_step(self, name, round_name, seconds, status):
        step = {"name": name, "round": round_name, "seconds": seconds, "status": status}
        if self._methods is not None:
            step["methods"], self._methods = self._methods, None
        self.data["steps"].append(step)

    def fail_step(self, name, round_name):
        # The innermost failing step is reported; enclosing steps keep it.
        if self.failure is None:
            self.failure = (name, round_name)

    def write(self):
        fov = self.fov
        preprocessing = fov.preprocessing_record or None
        supplied = (preprocessing or {}).get("supplied_statistics") or {}
        if supplied.get("path") is not None:
            # The supplied statistics file, as source_path records it (the FOV's record keeps the path as given).
            preprocessing = dict(preprocessing, supplied_statistics=dict(
                supplied, path=source_path(supplied["path"], self.input_root)))
        counts = {}
        if fov.spot_result is not None:
            counts["spots"] = len(fov.spot_result.spots)
        if fov.intensity_result is not None:
            counts["intensities"] = len(fov.intensity_result.spot_ids)
        if fov.decoding_result is not None:
            counts["call_status"] = {str(k): int(v) for k, v in fov.decoding_result.table.call_status.value_counts().items()}
        if fov.scoring_result is not None:
            counts["scoring"] = fov.scoring_result.counts
        if fov.deduplication_result is not None:
            counts["deduplication"] = fov.deduplication_result.counts
        if fov.filtering_result is not None:
            # A copy, which gains the diagnostics' endpoint counts and the fractions.
            counts["filtering"] = dict(fov.filtering_result.counts)
            diagnostics = fov.filtering_result.diagnostics
            # Present exactly when the filter checked endpoints (end_bases or a layout's segment ends).
            if "endpoint_valid_count" in diagnostics:
                counts["filtering"]["endpoint_valid"] = diagnostics["endpoint_valid_count"]
            if "endpoint_valid_segment_counts" in diagnostics:
                counts["filtering"]["endpoint_valid_segments"] = dict(diagnostics["endpoint_valid_segment_counts"])
            # accepted and endpoint_valid over all reads, null when there are none.
            counts["filtering"]["fractions"] = dict(fov.filtering_result.fractions)
        if fov.decoding_result is not None:
            # The §2.8 population summary (docs/readout-contract.md, "Diagnostics") of the latest reads,
            # with or without scoring and deduplication.
            from starfinder.readout import summarize_reads
            counts["summary"] = summarize_reads(fov.results)
        self.data.update(preprocessing=preprocessing,
                         registration=fov.registration_attempts,
                         transforms=_transforms(fov, self.written.get("registered", [])), counts=counts)
        # Through the module, so that a test can make the write fail.
        _records.write_json(self.data, self.path)

    def finish(self, status, error=None, round_name=None):
        """Record the final status of the processing; a failed write never masks ``error``.

        ``succeeded`` means that ``FOV.run``'s processing succeeded; the files
        written after it are recorded under ``outputs`` (add_output).
        """
        self.data.update(status=status, ended_at=_now())
        if error is not None:
            step, failed_round = self.failure or ("run", round_name)
            self.data["error"] = {"step": step, "round": failed_round, "type": type(error).__name__,
                "message": str(error),
                "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__))}
            try:
                self.write()
            except Exception:
                logger.exception(f"[{self.fov.fov_id}] Could not write {self.path}")
        else:
            try:
                self.write()
            except BaseException:
                # The success was never recorded on disk; let the caller record the failure.
                self.data.update(status="running", ended_at=None)
                raise
