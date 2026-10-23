"""Small per-FOV run record (run.json) written while checkpoints are active."""
from __future__ import annotations

from datetime import datetime, timezone
import logging
from pathlib import Path
import traceback

from starfinder import _records
from starfinder._layout import RUN_RECORD
from starfinder._records import file_sha256, software

logger = logging.getLogger("starfinder")

FORMAT_VERSION = 1


def _now():
    return datetime.now(timezone.utc).isoformat()


def _codebook_record(dataset):
    """The codebook (multiplexed) or direct panel (direct) of the run: kind, file, SHA-256, entries, genes; or None.

    path and sha256 are those Dataset.load_codebook or load_direct_panel kept for
    the object the dataset holds now; null for one set without a file.
    """
    kind = "codebook" if dataset.readout_mode == "multiplexed" else "direct_panel"
    reference = dataset.codebook if kind == "codebook" else dataset.direct_panel
    if reference is None:
        return None
    source = dataset._reference_files.get(kind)
    path, sha256 = (source[1], source[2]) if source is not None and source[0] is reference else (None, None)
    entries = reference.n_entries if kind == "codebook" else len(reference.table)
    return {"kind": kind, "path": path, "sha256": sha256, "entries": entries, "genes": reference.n_genes}


def _workflow_config_record(dataset):
    """Path and SHA-256 of the workflow configuration file from_workflow_config translated; None outside the workflow."""
    path = dataset._workflow_config
    if path is None:
        return None
    # A configuration whose config_path names no readable file keeps its path with a null hash.
    return {"path": str(path), "sha256": file_sha256(path) if Path(path).is_file() else None}


def _transforms(fov, written):
    """R1: per registered moving round, the entries of transforms.json (_transform_json) and its WarpConfig.

    written lists the files the stage registered wrote; a dense field or B-spline
    coefficients are named only when that stage wrote the round's field file.
    """
    from starfinder._checkpoint import _transform_json
    rounds = set(fov._pipeline_rounds())
    applications = fov.registration_record.get("application", {})
    transforms = {}
    for name, results in fov.registration_results.items():
        # Other rounds (prepare_morphology, register_rounds) are not processed by FOV.run.
        if name not in rounds or not results:
            continue
        field_name = f"{name}_field.npz" if f"registered/{name}_field.npz" in written else None
        transforms[name] = {"reference": fov.rounds.reference_round,
                            "results": [_transform_json(r, i, field_name) for i, r in enumerate(results)],
                            "application_config": applications.get(name, results[-1].application_config)}
    return transforms


class _RunRecord:
    """Mutable run.json content; rewritten atomically at step boundaries."""

    def __init__(self, fov, directory, checkpoints, config, execution):
        self.fov = fov
        self.directory = Path(directory)
        self.checkpoints = checkpoints
        self.failure = None
        self._methods = None
        dataset = fov.dataset
        from starfinder.readout.codebook import _encoding_record
        # The codebook's encoding (null without a codebook or in readout mode direct).
        encoding = (_encoding_record(dataset.codebook.encoding)
                    if dataset.codebook is not None and dataset.readout_mode == "multiplexed" else None)
        self.data = {
            "format_version": FORMAT_VERSION, "dataset_id": dataset.dataset_id,
            "sample_id": dataset.sample_id, "fov_id": fov.fov_id, "subtile_id": fov.subtile_id,
            "status": "running", "started_at": _now(), "ended_at": None, "error": None,
            **software(),
            # The dataset's readout mode (docs/readout-contract.md) is recorded with the configs.
            "config": {"pipeline": config, "execution": execution, "checkpoints": checkpoints,
                       "readout_mode": dataset.readout_mode, "encoding": encoding},
            # The workflow configuration file of a workflow rule (null outside the workflow).
            "workflow_config": _workflow_config_record(dataset),
            # Channel pattern, name and wavelength per round (docs/checkpoints.md, "run.json").
            "channels": dataset.channel_record(),
            "inputs": [], "codebook": _codebook_record(dataset), "steps": [], "preprocessing": None,
            "registration": {}, "transforms": {}, "counts": {},
            "checkpoint_directory": str(self.directory), "checkpoints": {},
        }

    @property
    def path(self):
        return self.directory / RUN_RECORD

    def add_inputs(self, paths):
        """Record loaded source files, hashing them unless hash_inputs is False."""
        for path in paths:
            self.data["inputs"].append({"path": str(path),
                "sha256": file_sha256(path) if self.checkpoints.hash_inputs else None})

    def add_checkpoint(self, stage, files):
        self.data["checkpoints"].setdefault(stage, []).extend(files)

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
            # A copy: the result's counts are also written to the rule's logs.
            counts["filtering"] = dict(fov.filtering_result.counts)
            diagnostics = fov.filtering_result.diagnostics
            # Present exactly when the filter checked endpoints (end_bases or a layout's segment ends).
            if "endpoint_valid_count" in diagnostics:
                counts["filtering"]["endpoint_valid"] = diagnostics["endpoint_valid_count"]
            if "endpoint_valid_segment_counts" in diagnostics:
                counts["filtering"]["endpoint_valid_segments"] = dict(diagnostics["endpoint_valid_segment_counts"])
        if fov.scoring_result is not None or fov.deduplication_result is not None:
            # The §2.8 population summary (docs/readout-contract.md, "Diagnostics") of the scored or
            # deduplicated reads; runs without those stages keep the earlier counts only.
            from starfinder.readout import summarize_reads
            counts["summary"] = summarize_reads(fov.results)
        self.data.update(preprocessing=fov.preprocessing_record or None,
                         registration=fov.registration_attempts,
                         transforms=_transforms(fov, self.data["checkpoints"].get("registered", [])), counts=counts)
        # Through the module, so that a test can make the write fail.
        _records.write_json(self.data, self.path)

    def finish(self, status, error=None, round_name=None):
        """Record the final status; a failed write never masks ``error``."""
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
