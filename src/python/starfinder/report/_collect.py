"""The collection of one sample's records into the report's tables and record (docs/report-contract.md).

:func:`collect_report` applies the checks before anything is written, reads the
records at their fixed places under the output root (the run records,
``registered/transforms.json`` for a run record without ``transforms``, the
placement, the per-FOV assignment and segmentation records and the export), checks
each file against the SHA-256 a record holds for it, and writes the tables that copy
record fields or compute values from them (``_values``: the read fractions, the shifts
and their residuals, the alignment score, the area a FOV owns and the cell density; the
summary keys; the example FOVs), the problematic FOVs (``_rules``), the replay of the
main example (``_replay``: ``intensity.csv``, ``report.json`` ``replay``), the
segmentation previews of the example FOVs (``_previews``), the sample measurements
(``_measurements``: the reads assigned to cells, ``cells.csv``, ``mecr.csv``,
``reads_along_z.csv`` and the summary keys of the sample's cells), ``unavailable.csv``, the
page ``report.html`` (``_page``) and ``report.json`` into a staging folder that replaces the
report folder when it is complete (RP1). :func:`report_sample` is the entry
``Dataset.report``; :func:`report_examples` and :func:`report_measurements` make the static
figures without writing anything.

A cell is a value, not applicable (null, no row of ``unavailable.csv``) or unavailable
(null, one row with its reason).
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import shutil
import uuid
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from starfinder import _layout

from ._config import ReportConfig, marker_genes
from ._measurements import _Measurements
from ._previews import _Previews
from ._replay import reference_round, replay_example, round_transform
from ._rules import RuleInputs, evaluate_rules
from ._schema import (FORMAT_VERSION, POPULATIONS, PROBLEM_RULES, READOUT_LEVELS, REPORT_FILE, REPORT_PARTS,
                      REPORT_TABLES, RULE_UNITS, STAGE, SUMMARY_KEYS, TABLE_DIR, TABLE_DTYPES, ReportRecord, row_key)
from ._values import (NA, ROLES, Unavailable, _Marker, alignment_score, chain_shift, choose_examples,
                      density, first, fraction, is_value, median, over_fovs, owned_pixels, residuals,
                      run_seconds, step_name)

PAGE_FILE = "report.html"
_UNPLACED = ("missing", "failed")
_WRITTEN = ("written", "reused")
#: The tables this collection writes, in folder order; the others are written by the parts that make them.
COLLECTED_TABLES = ("fovs", "registration", "steps", "readout_totals", "population", "problematic_fovs",
                    "cells", "mecr", "reads_along_z", "intensity", "unavailable")
#: The parts the replay of the main example makes (docs/report-contract.md, "The replay").
REPLAY_PARTS = ("preprocessing.example", "preprocessing.intensity", "registration.example")
#: The parts of the columns of the collected tables (docs/report-contract.md, "Parts, their inputs and their reasons").
COLUMN_PARTS = {
    "fovs": {**{c: "workflow.fov_status" for c in ("example", "status", "failed_step", "failed_round", "started_at",
                                                    "ended_at", "run_seconds")},
             **{c: "assembly.placement" for c in ("placement_status", "correction_z", "correction_y",
                                                   "correction_x", "footprint_y0", "footprint_y1", "footprint_x0",
                                                   "footprint_x1", "centre_y", "centre_x")},
             **{c: "readout.per_fov" for c in ("spots", "correct_form", "matched", "merged_by_deduplication",
                                                "passed_filter", "correct_form_fraction", "matched_fraction",
                                                "merged_fraction", "passed_filter_fraction")},
             **{c: "assembly.cells_per_fov" for c in ("n_cells", "n_cells_retained", "n_molecules",
                                                       "n_molecules_retained", "reads_in_cells",
                                                       "reads_in_cells_fraction", "owned_area_um2",
                                                       "cell_density_per_mm2")}},
    "registration": {c: "registration.heatmap" for c in TABLE_DTYPES["registration"]},
    "steps": {"seconds": "workflow.time"},
    "readout_totals": {"count": "readout.totals", "fraction_of_spots": "readout.totals"},
    "population": {"cells": "assembly.accounting", "molecules": "assembly.accounting"},
    "cells": {c: "sample.per_cell" for c in TABLE_DTYPES["cells"]},
    "mecr": {c: "sample.mecr" for c in TABLE_DTYPES["mecr"]},
    "reads_along_z": {c: "sample.reads_along_z" for c in TABLE_DTYPES["reads_along_z"]},
}
# The read levels of a run record: the count's place in counts and the pipeline config that makes it.
_COUNTS = {"spots": (("spots",), "spot_finding"), "matched": (("call_status", "assigned"), "decoding"),
           "merged_by_deduplication": (("deduplication", "merged_reads"), "deduplication"),
           "passed_filter": (("filtering", "accepted"), "filtering"),
           "correct_form": (("filtering", "endpoint_valid"), "filtering")}
_FRACTIONS = {"correct_form_fraction": "correct_form", "matched_fraction": "matched",
              "merged_fraction": "merged_by_deduplication", "passed_filter_fraction": "passed_filter"}
# The report's outcome of a round from its last entry (docs/report-algorithms.md, "Outcome", choice A13).
_OUTCOMES = {("application", "succeeded"): "succeeded", ("application", "application_failed"): "application_failed",
             ("application", "applying"): "application_failed", ("estimation", "rejected"): "rejected",
             ("estimation", "failed"): "failed", ("estimation", "estimating"): "failed",
             ("estimation", "succeeded"): "failed"}
_UNFINISHED = {("application", "applying"), ("estimation", "estimating"), ("estimation", "succeeded")}
_SECTIONS = {"assemble": "assembly", "fuse": "fusion", "segment": "segmentation", "fuse_labels": "labels",
             "export_spatialdata": "spatialdata"}
# The R1 to R4 keys a run record holds since docs/report-contract.md ("Records added to run.json").
_ADDED_KEYS = ("transforms", "codebook", "workflow_config")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _now():
    return datetime.now(timezone.utc).isoformat()


def _get(mapping, *keys):
    for key in keys:
        if not isinstance(mapping, dict) or key not in mapping:
            return None
        mapping = mapping[key]
    return mapping


class _Inputs:
    """Reads the files of the report, listing each in ``report.json`` ``inputs`` once, in the order first read."""

    def __init__(self, root: Path):
        self.root = root
        self.entries = {}

    def relative(self, path) -> str:
        path = Path(path)
        try:
            return path.resolve().relative_to(self.root.resolve()).as_posix()
        except ValueError:
            return str(path.resolve())

    def read(self, path, kind, recorded=None):
        """The file's bytes, or :class:`Unavailable` (``input_missing``, or ``input_changed`` against ``recorded``)."""
        path = Path(path)
        name = self.relative(path)
        if not path.is_file():
            return Unavailable("input_missing", name if recorded is None else f"{name} (recorded SHA-256 {recorded})")
        data = path.read_bytes()
        digest = _sha256(data)
        self.entries.setdefault(name, {"path": name, "kind": kind, "sha256": digest, "bytes": len(data),
                                       "verified": recorded is not None})
        if recorded is not None and digest != recorded:
            return Unavailable("input_changed", f"{name} has SHA-256 {digest}, the record {recorded}")
        return data

    def verify(self, path, kind, recorded=None):
        """As :meth:`read`, the file hashed in blocks without holding its bytes: None, or :class:`Unavailable`."""
        from starfinder._records import file_sha256
        path = Path(path)
        name = self.relative(path)
        if not path.is_file():
            return Unavailable("input_missing", name if recorded is None else f"{name} (recorded SHA-256 {recorded})")
        digest = file_sha256(path)
        self.entries.setdefault(name, {"path": name, "kind": kind, "sha256": digest, "bytes": path.stat().st_size,
                                       "verified": recorded is not None})
        if recorded is not None and digest != recorded:
            return Unavailable("input_changed", f"{name} has SHA-256 {digest}, the record {recorded}")
        return None

    def json(self, path, kind, *, strict=True):
        """A JSON file's content; one that does not parse raises ValueError, or is None with ``strict`` False.

        The entry checks read their records strictly; a record read after them that does not parse is read as
        absent, so that the parts it feeds are unavailable instead of the entry raising.
        """
        data = self.read(path, kind)
        if isinstance(data, Unavailable):
            return data
        try:
            return json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            if not strict:
                return None
            raise ValueError(f"{path} is not a JSON file: {error}") from error

    def record(self, path, kind, recorded):
        """A JSON record checked against the SHA-256 of its canonical content that another record holds.

        ``assignment.json`` ``inputs.cells.record_sha256`` hashes the segmentation record the assignment read
        (``_records.py``: ``json_sha256``), not the file's bytes. Returns the record, None when it
        does not parse, or :class:`Unavailable` (``input_missing``, ``input_changed``).
        """
        from starfinder._records import json_sha256
        path = Path(path)
        name = self.relative(path)
        if not path.is_file():
            return Unavailable("input_missing", f"{name} (recorded record SHA-256 {recorded})")
        record = self.json(path, kind, strict=False)
        if recorded is not None:
            self.entries[name]["verified"] = True
            digest = json_sha256(record) if record is not None else None
            if digest != recorded:
                return Unavailable("input_changed", f"{name} holds a record of SHA-256 {digest}, the assignment's "
                                                    f"record {recorded}")
        return record

    def table(self, path, kind, recorded, dtypes):
        """A recorded table read with its dtypes, or :class:`Unavailable`."""
        from starfinder._records import _read_table
        data = self.read(path, kind, recorded)
        if isinstance(data, Unavailable):
            return data
        return _read_table(path, dtypes)


# --- The checks before anything is written ------------------------------------------------------------------

def _check_arguments(sample, fov_ids, config):
    if config is None:
        config = ReportConfig()
    if not isinstance(config, ReportConfig):
        raise TypeError(f"config must be a ReportConfig or None; got {type(config).__name__}")
    if isinstance(fov_ids, str) or not isinstance(fov_ids, Sequence):
        raise ValueError(f"fov_ids must be a sequence of FOV IDs; got {fov_ids!r}")
    fov_ids = list(fov_ids)
    if not fov_ids or not all(isinstance(f, str) and f for f in fov_ids) or len(set(fov_ids)) != len(fov_ids):
        raise ValueError(f"fov_ids must be a nonempty sequence of unique nonempty strings; got {fov_ids!r}")
    if not isinstance(sample, str) or not sample:
        raise ValueError(f"sample must be a nonempty string; got {sample!r}")
    return fov_ids, config


def _check_run_records(inputs, root, fov_ids, dataset_id, sample_id):
    records = {}
    for fov in fov_ids:
        path = _layout.run_record(root, fov)
        if not path.is_file():
            records[fov] = None
            continue
        record = inputs.json(path, "run_record")
        if not isinstance(record, dict):
            raise ValueError(f"{path} is not a JSON object")
        for key, expected in (("format_version", 1), ("dataset_id", dataset_id), ("sample_id", sample_id),
                              ("fov_id", fov)):
            if record.get(key) != expected:
                raise ValueError(f"{path}: {key} is {record.get(key)!r}, expected {expected!r}")
        records[fov] = record
    return records


def _check_export(inputs, export_dir, sample, fov_ids, assignment):
    from starfinder.export._record import verify_section_hashes
    path = export_dir / _layout.EXPORT_FILE
    if not path.is_file():
        return None
    record = inputs.json(path, "export_record")
    if not isinstance(record, dict):
        raise ValueError(f"{path} is not a JSON object")
    verify_section_hashes(record, where=str(path))
    if record.get("sample") != sample:
        raise ValueError(f"{path}: sample is {record.get('sample')!r}, expected {sample!r}")
    assembly = record.get("assembly")
    if isinstance(assembly, dict) and isinstance(assembly.get("fovs"), list):
        exported = {entry.get("fov_id") for entry in assembly["fovs"]}
        if exported != set(fov_ids):
            raise ValueError(f"{path}: assembly.fovs holds the FOVs {sorted(map(str, exported))}, the FOV list "
                             f"{sorted(fov_ids)}")
        if (assembly.get("route") or "per_fov") == "per_fov":
            names = {(entry.get("assignment") or {}).get("name") for entry in assembly["fovs"]} - {None}
            if names and names != {assignment}:
                raise ValueError(f"{path}: the route per FOV assembled the assignment {sorted(names)}, "
                                 f"the report reads {assignment!r}")
    return record


def _check_placement(inputs, stitching_dir, fov_ids):
    path = stitching_dir / "placement.json"
    if not path.is_file():
        return None
    record = inputs.json(path, "placement")
    placed = set((record.get("inputs") or {}) if isinstance(record, dict) else ())
    if placed != set(fov_ids):
        raise ValueError(f"{path}: the placement holds the FOVs {sorted(placed)}, the FOV list {sorted(fov_ids)}")
    return record


def _codebook_genes(data: bytes, kind) -> list[str]:
    """The gene names of a codebook or direct panel file (the gene column of the forms the loaders read)."""
    rows = [r for r in csv.reader(io.StringIO(data.decode("utf-8-sig"), newline="")) if r]
    if not rows:
        return []
    header = [v.strip() for v in rows[0]]
    if kind == "direct_panel" or header[0] in ("gene", "gene_id", "entry_id"):
        column = header.index("gene_id") if "gene_id" in header else header.index("gene")
        rows = rows[1:]
    else:
        column = 0
    return [r[column].strip() for r in rows if len(r) > column]


def _check_markers(inputs, config, records, export, export_dir):
    genes = marker_genes(config)
    if not genes:
        return
    source = None
    for record in records.values():
        codebook = (record or {}).get("codebook")
        if isinstance(codebook, dict) and codebook.get("path"):
            data = inputs.read(codebook["path"], "codebook", codebook.get("sha256"))
            if not isinstance(data, Unavailable):
                source = (codebook["path"], set(_codebook_genes(data, codebook.get("kind"))))
            break
    if source is None and isinstance(_get(export, "assembly", "genes"), list):
        source = (str(export_dir / _layout.EXPORT_FILE), set(export["assembly"]["genes"]))
    if source is None:
        raise ValueError("the configuration names marker genes, but there is no gene list to check them against: "
                         "no run record names a readable codebook (run.json codebook) and there is no "
                         f"{export_dir / _layout.EXPORT_FILE} assembly.genes")
    for gene in genes:
        if gene not in source[1]:
            raise ValueError(f"marker gene {gene!r} is not in the gene list of {source[0]}")


def _check_examples(config, fov_ids):
    if config.example_fovs is not None:
        outside = [f for f in config.example_fovs if f not in fov_ids]
        if outside:
            raise ValueError(f"example_fovs {outside} are not FOVs of the list")


def own_files(folder: Path) -> list[str]:
    """The files of an existing report folder, all of them the report's own; FileExistsError naming another one."""
    if not folder.exists():
        return []
    if not folder.is_dir():
        raise FileExistsError(f"the report folder {folder} is a file")
    files = sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*") if not p.is_dir())
    if not files:
        return []
    if REPORT_FILE not in files:
        raise FileExistsError(f"{folder / files[0]} is not a file of a report (the folder has no {REPORT_FILE}); "
                              "the report is written only into its own folder")
    try:
        record = json.loads((folder / REPORT_FILE).read_text(encoding="utf-8"))
        listed = {entry["path"] for entry in record["files"]}
    except (ValueError, KeyError, TypeError) as error:
        raise FileExistsError(f"{folder / REPORT_FILE} is not a report record ({error}); "
                              "the report is written only into its own folder") from error
    foreign = [f for f in files if f != REPORT_FILE and f not in listed]
    if foreign:
        raise FileExistsError(f"{folder / foreign[0]} is not a file of the report {folder / REPORT_FILE} lists; "
                              "the report is written only into its own folder")
    return files


# --- The collection --------------------------------------------------------------------------------------------

class _Collection:
    def __init__(self, inputs, root, sample, fov_ids, records, export, placement, assignment, segmentation,
                 config):
        self.inputs, self.root, self.sample, self.fov_ids = inputs, root, sample, fov_ids
        self.records, self.export, self.placement = records, export, placement
        self.assignment, self.segmentation, self.config = assignment, segmentation, config
        self.export_dir = _layout.export_dir(root, sample)
        self.stitching_dir = _layout.stitching_dir(root, sample)
        self._transform_files = {}
        self.assembly = (export or {}).get("assembly") if isinstance((export or {}).get("assembly"), dict) else None
        self.route = None if export is None else ((self.assembly or {}).get("route") or "per_fov")
        self._placement_frame()
        self._export_frame()
        self.assignments = self._assignments()
        self.measurements = _Measurements(self)

    # The sources -------------------------------------------------------------------------------------------

    def _placement_frame(self):
        self.p_missing = None
        if self.placement is None:
            self.p_missing = Unavailable("no_placement", f"{self.inputs.relative(self.stitching_dir)}/placement.json")
            return
        entry = _get(self.placement, "tables", "fovs") or {}
        frame = self.inputs.table(self.stitching_dir / entry.get("path", "fovs.csv"), "placement_table",
                                  entry.get("sha256"), entry.get("dtypes") or {})
        if isinstance(frame, Unavailable):
            self.p_missing = frame
        else:
            self.p_rows = {row["fov_id"]: row for row in frame.to_dict("records")}

    def _export_frame(self):
        # The export's state (no_export, part_not_written) and its fovs table's (input_missing, input_changed).
        self.e_state = self.e_table = None
        if self.export is None:
            self.e_state = Unavailable("no_export", f"{self.inputs.relative(self.export_dir)}/export.json")
            return
        from starfinder.export._record import part_status
        status = part_status(self.export, "assemble")
        if status not in _WRITTEN or self.assembly is None:
            self.e_state = Unavailable("part_not_written", f"assemble: {status}")
            return
        entry = _get(self.assembly, "files", "fovs") or {}
        frame = self.inputs.table(self.export_dir / entry.get("path", "tables/fovs.csv"), "export_table",
                                  entry.get("sha256"), entry.get("dtypes") or {})
        if isinstance(frame, Unavailable):
            self.e_table = frame
        else:
            self.e_rows = {row["fov_id"]: row for row in frame.to_dict("records")}

    @property
    def no_cells(self):
        """``no_segmentation`` for a sample without cells (an export ``molecules_only``, or no assignment)."""
        if self.assignment is None or (self.assembly or {}).get("mode") == "molecules_only":
            return Unavailable("no_segmentation", "no assignment" if self.assignment is None else
                               "export assembled molecules_only")
        return None

    def _assignments(self):
        """FOV → its assignment record (route per FOV, or no export), with the segmentation record it read.

        ``assignment.json`` is checked against the SHA-256 the export records for it
        (``assembly.fovs[*].assignment.record_sha256`` of a folder source), and the cell run's
        ``segmentation.json`` against the record SHA-256 ``assignment.json`` holds for it; a FOV whose record is
        missing or changed has no segmentation and assignment rows, and ``lane_missing`` holds its reason.
        """
        found, self.lane_missing = {}, {}
        if self.assignment is None or self.route == "after_fusion":
            return found
        recorded = {}
        for entry in (self.assembly or {}).get("fovs") or []:
            saved = entry.get("assignment") or {}
            if saved.get("source") == "folder" and saved.get("name") == self.assignment and saved.get("record_sha256"):
                recorded[entry.get("fov_id")] = saved["record_sha256"]
        for fov in self.fov_ids:
            path = _layout.checkpoint_dir(self.root, fov) / "assignment" / self.assignment / _layout.ASSIGNMENT_RECORD
            if not path.is_file() and fov not in recorded:
                continue
            data = self.inputs.read(path, "assignment_record", recorded.get(fov))
            if isinstance(data, Unavailable):
                self.lane_missing[fov] = data
                continue
            record = self.inputs.json(path, "assignment_record", strict=False)
            if not isinstance(record, dict):
                continue
            cells = _get(record, "inputs", "cells") or {}
            run = cells.get("run")
            seg = None
            if run is not None:
                path = _layout.checkpoint_dir(self.root, fov) / "segmentation" / run / "segmentation.json"
                # The run's record exists when the run was saved under it (its labels linked from there); an
                # import or a run kept in memory has none, and the assignment's own copy of the record is read.
                if path.is_file() or cells.get("saved_under_run"):
                    seg = self.inputs.record(path, "segmentation_record", cells.get("record_sha256"))
            if isinstance(seg, Unavailable):
                self.lane_missing[fov] = seg
                continue
            found[fov] = (record, seg if isinstance(seg, dict) else cells.get("record"))
        return found

    # fovs.csv ----------------------------------------------------------------------------------------------

    def _record_path(self, fov):
        return f"{_layout.CHECKPOINTS}/{fov}/{_layout.RUN_RECORD}"

    def count(self, fov, level):
        """A read level of a FOV's run record, or the reason it has none."""
        record = self.records[fov]
        if record is None:
            return Unavailable("no_run_record", self._record_path(fov))
        keys, step = _COUNTS[level]
        counts = record.get("counts") or {}
        value = _get(counts, *keys)
        if level == "matched" and value is None and isinstance(counts.get("call_status"), dict):
            value = 0
        if value is not None:
            return int(value)
        pipeline = _get(record, "config", "pipeline") or {}
        if level == "correct_form" and isinstance(counts.get("filtering"), dict):
            # The filter ran: no endpoint check, or a run record written before R3.
            # A record with R3 holds the count exactly when the filter checked endpoints; an older one never does,
            # and only its end_bases tell that the filter checked them (a layout's segment ends are not in run.json).
            if any(key in record for key in _ADDED_KEYS) or _get(pipeline, "filtering", "end_bases") is None:
                return Unavailable("not_configured", "the filter checked no endpoints")
            return Unavailable("not_recorded", f"{self._record_path(fov)}: written before "
                                               "counts.filtering.endpoint_valid")
        return self._not_reached(fov, record, pipeline.get(step) is not None, step)

    def _not_reached(self, fov, record, configured, step):
        status = record.get("status")
        if status == "running":
            return Unavailable("run_not_finished", f"{self._record_path(fov)}: status running")
        if status in ("failed", "interrupted"):
            error = record.get("error") or {}
            where = error.get("step") if error.get("round") is None else f"{error.get('step')}, {error.get('round')}"
            return Unavailable("step_failed", f"run {status} at {where}")
        if not configured:
            return Unavailable("not_configured", f"config.pipeline.{step} is null")
        return Unavailable("not_recorded", f"{self._record_path(fov)}: no record of {step}")

    def fovs_rows(self):
        rows = []
        for index, fov in enumerate(self.fov_ids):
            record = self.records[fov]
            row = {"fov_id": fov, "list_index": index, "example": NA, "run_record": record is not None}
            row.update(self._status(fov, record))
            row.update(self._placement_columns(fov))
            for level in _COUNTS:
                row[level] = self.count(fov, level)
            for column, level in _FRACTIONS.items():
                row[column] = fraction(row[level], row["spots"], "spots")
            row.update(self._export_columns(fov, row))
            rows.append({c: row[c] for c in TABLE_DTYPES["fovs"]})
        self.fovs = rows
        self.examples = self._choose_examples(rows)
        if isinstance(self.examples, list):
            for row in rows:
                roles = [e.role for e in self.examples if e.fov_id == row["fov_id"]]
                row["example"] = ";".join(roles) if roles else NA
        return rows

    def examples_record(self):
        """``report.json`` ``examples``: per role its FOV and how it was chosen; each role null when no example
        could be chosen (the reason of the read counts is in ``example_reason``)."""
        if isinstance(self.examples, Unavailable):
            return {role: None for role in ROLES}
        return {e.role: e.record() for e in self.examples}

    @property
    def example_reason(self):
        """The reason no example could be chosen, or None."""
        return self.examples if isinstance(self.examples, Unavailable) else None

    def _choose_examples(self, rows):
        """The example FOVs ("The example FOVs"): the named three, or the automatic choice from the read counts
        and the counted cells of the owning FOVs; the reason of the read counts when none can be chosen."""
        return choose_examples(self.fov_ids, {r["fov_id"]: r["passed_filter"] for r in rows},
                               {r["fov_id"]: r["n_cells_retained"] for r in rows},
                               owning={r["fov_id"] for r in rows if r["owned_area_um2"] is not NA},
                               named=self.config.example_fovs)

    def _status(self, fov, record):
        names = ("status", "failed_step", "failed_round", "started_at", "ended_at", "run_seconds")
        if record is None:
            return {c: Unavailable("no_run_record", self._record_path(fov)) for c in names}
        status, error = record.get("status"), record.get("error")
        running = status == "running"
        not_recorded = Unavailable("not_recorded", f"{self._record_path(fov)}: no value")
        return {"status": status if status is not None else not_recorded,
                "failed_step": error.get("step") if error else NA,
                "failed_round": error.get("round") if error and error.get("round") is not None else NA,
                "started_at": record.get("started_at") or not_recorded,
                "ended_at": NA if running else (record.get("ended_at") or not_recorded),
                "run_seconds": NA if running else run_seconds(record.get("started_at") or not_recorded,
                                                              record.get("ended_at") or not_recorded)}

    def _p_row(self, fov):
        return None if self.p_missing is not None else self.p_rows.get(fov)

    def _e_row(self, fov):
        return None if self.e_state is not None or self.e_table is not None else self.e_rows.get(fov)

    def _placement_columns(self, fov):
        """The placement columns: each from its first source that exists, with that source's reasons.

        ``placement_status`` and ``correction_*`` read the placement, the status the export when there is no
        placement; the footprint and centre read the export's fovs table, else the placement's origin and grid.
        """
        from starfinder.assembly._ownership import footprint_yx
        p, e = self._p_row(fov), self._e_row(fov)
        out = {}
        if self.placement is not None:
            out["placement_status"] = self.p_missing or p["status"]
        elif self.e_state is None:
            out["placement_status"] = self.e_table or e["placement_status"]
        else:
            out["placement_status"] = first(self.p_missing, self.e_state)
        for axis in "zyx":
            out[f"correction_{axis}"] = (self.p_missing if p is None else
                                         Unavailable("fov_not_placed", f"placement status {p['status']}")
                                         if p["status"] in _UNPLACED else float(p[f"correction_{axis}"]))
        columns = ("footprint_y0", "footprint_y1", "footprint_x0", "footprint_x1", "centre_y", "centre_x")
        if self.e_state is None:
            if e is None:
                values = [self.e_table] * len(columns)
            elif e["placement_status"] in _UNPLACED or pd.isna(e["footprint_y0"]):
                values = [Unavailable("fov_not_placed", f"placement status {e['placement_status']}")] * len(columns)
            else:
                values = [float(e[c]) for c in columns]
        elif self.placement is not None:
            if p is None:
                values = [self.p_missing] * len(columns)
            elif p["status"] in _UNPLACED:
                values = [Unavailable("fov_not_placed", f"placement status {p['status']}")] * len(columns)
            else:
                grid = self.placement["inputs"][fov]["grid"]
                (y0, y1, x0, x1), (cy, cx) = footprint_yx((p["origin_y"], p["origin_x"]), grid["shape_zyx"][1:],
                                                          grid["metadata"]["spacing_zyx"][1:])
                values = [float(v) for v in (y0, y1, x0, x1, cy, cx)]
        else:
            values = [first(self.p_missing, self.e_state)] * len(columns)
        out.update(zip(columns, values))
        return out

    def _export_columns(self, fov, row):
        e = self._e_row(fov)
        no_cells = self.no_cells
        out = {}
        for column in ("n_cells", "n_cells_retained", "n_molecules", "n_molecules_retained"):
            cells = column.startswith("n_cells")
            missing = first(self.e_state, no_cells if cells else None, self.e_table)
            out[column] = missing or int(e[column])
        out["reads_in_cells"] = first(self.e_state, no_cells) or self.measurements.reads_in_cells_of(fov)
        out["reads_in_cells_fraction"] = fraction(out["reads_in_cells"], out["n_molecules_retained"],
                                                  "n_molecules_retained")
        # The area a FOV owns: the placement, its status, and the ownership rule of the export.
        p = self._p_row(fov)
        status = p["status"] if p is not None else row["placement_status"]
        owns = (p["status"] != "placeholder") if p is not None else (
            bool(e["owns_points"]) if e is not None else True)
        if not owns:
            out["owned_area_um2"] = out["cell_density_per_mm2"] = NA
            return out
        unplaced = Unavailable("fov_not_placed", f"placement status {status}") if status in _UNPLACED else None
        out["owned_area_um2"] = first(self.p_missing, unplaced, self.e_state) or self._owned_area(fov)
        out["cell_density_per_mm2"] = density(out["n_cells_retained"], out["owned_area_um2"])
        return out

    def _ownership(self):
        """The export's ownership regions: ``assembly.ownership_map`` of the placement, the grids of the export's
        FOVs and ``AssemblyConfig(**export.json assembly.config)``, the call ``assemble`` makes; or the reason
        the placement cannot be loaded."""
        if not hasattr(self, "_ownership_map"):
            from starfinder.assembly import AssemblyConfig, ownership_map
            from starfinder.image import ImageMetadata
            from starfinder.segmentation import ReferenceGrid
            from starfinder.stitching import load_placement
            from starfinder.stitching._persist import TABLE_FILES
            tables = self.placement.get("tables") or {}
            for name, file_name in TABLE_FILES.items():
                entry = tables.get(name) or {}
                data = self.inputs.read(self.stitching_dir / entry.get("path", file_name), "placement_table",
                                        entry.get("sha256"))
                if isinstance(data, Unavailable):
                    self._ownership_map = data
                    return data
            self.grids = {e["fov_id"]: ReferenceGrid(tuple(e["grid"]["shape_zyx"]),
                                                     ImageMetadata(**e["grid"]["metadata"]), "declared")
                          for e in self.assembly.get("fovs") or [] if isinstance(e.get("grid"), dict)}
            self._ownership_map = ownership_map(load_placement(self.stitching_dir), self.grids,
                                                config=AssemblyConfig(**self.assembly.get("config") or {}))
        return self._ownership_map

    def _owned_area(self, fov):
        """"The area a FOV owns" (OA1): the FOV's pixels whose centre it owns, times the pixel area, µm²."""
        ownership = self._ownership()
        if isinstance(ownership, Unavailable):
            return ownership
        if fov not in self.grids:
            return NA  # a FOV the export holds without an input owns no point (ownership_map needs its grid)
        p, grid = self.p_rows[fov], self.grids[fov]
        spacing = grid.metadata.spacing_zyx
        pixels = owned_pixels(ownership, fov, (p["origin_z"], p["origin_y"], p["origin_x"]), grid.shape_zyx, spacing)
        return pixels * spacing[1] * spacing[2]

    # registration.csv --------------------------------------------------------------------------------------

    def moving_rounds(self):
        rounds = []
        for fov in self.fov_ids:
            for name in (self.records[fov] or {}).get("registration") or {}:
                if name not in rounds:
                    rounds.append(name)
        return rounds

    def _transforms_file(self, fov):
        if fov not in self._transform_files:
            path = _layout.checkpoint_dir(self.root, fov) / _layout.REGISTERED / _layout.TRANSFORMS_FILE
            record = self.inputs.json(path, "transforms", strict=False) if path.is_file() else None
            self._transform_files[fov] = record if isinstance(record, dict) else None
        return self._transform_files[fov]

    def _transform(self, fov, record, name):
        """``(source, results, application_config)`` of a succeeded round's transforms (R1, else
        transforms.json), or Unavailable."""
        return round_transform(record, lambda: self._transforms_file(fov), fov, name)

    def registration_rows(self):
        rows = []
        for fov in self.fov_ids:
            for name in self.moving_rounds():
                rows.append(self._registration_row(fov, name))
        self.shift_medians = residuals(rows)
        return rows

    def _registration_row(self, fov, name):
        record = self.records[fov]
        row = {"fov_id": fov, "round": name}
        columns = [c for c in TABLE_DTYPES["registration"] if c not in row]
        if record is None:
            return row | {c: Unavailable("no_run_record", self._record_path(fov)) for c in columns}
        entries = (record.get("registration") or {}).get(name) or []
        if not entries:
            configured = _get(record, "config", "pipeline", "registration") is not None
            missing = self._not_reached(fov, record, configured, "registration")
            return row | {c: missing for c in columns}
        last = entries[-1]
        key = (last.get("record"), last.get("outcome"))
        if record.get("status") == "running" and key in _UNFINISHED:
            missing = Unavailable("run_not_finished", f"{self._record_path(fov)}: status running")
            return row | {c: missing for c in columns}
        outcome = _OUTCOMES.get(key)
        if outcome is None:
            return row | {c: Unavailable("not_recorded", f"registration.{name}: last entry {key}") for c in columns}
        succeeded = [e for e in entries if e.get("record") == "estimation" and e.get("outcome") == "succeeded"]
        row["outcome"] = outcome
        row["fallback"] = any(bool(e.get("fallback")) for e in succeeded)
        methods = [e.get("actual_method") for e in sorted(succeeded, key=lambda e: e.get("step", 0))]
        row["methods"] = ";".join(str(m) for m in methods) if methods else NA
        if outcome != "succeeded":
            missing = Unavailable("step_failed", f"registration.{name}: outcome {outcome}")
            return row | {c: missing for c in columns if c not in row}
        transform = self._transform(fov, record, name)
        if isinstance(transform, Unavailable):
            row["transform_source"] = transform
            shift = (transform,) * 3
        else:
            row["transform_source"] = transform[0]
            shift = chain_shift(transform[1], transform[2])
            shift = (NA,) * 3 if shift is NA else shift
        for axis, value in zip("zyx", shift):
            row[f"shift_{axis}"] = value
            row[f"residual_{axis}"] = value  # replaced by residuals() once every row of the round exists
        row["alignment_score"] = alignment_score(entries, outcome)
        return row

    # steps.csv ---------------------------------------------------------------------------------------------

    def steps_rows(self, registration):
        methods = {(r["fov_id"], r["round"]): r["methods"] for r in registration}
        rows = []
        for fov in self.fov_ids:
            record = self.records[fov]
            steps = (record or {}).get("steps") or []
            for index, step in enumerate(steps):
                name = step.get("name")
                rows.append({"scope": "fov", "fov_id": fov, "index": index, "name": name,
                             "round": step.get("round") if step.get("round") is not None else NA,
                             "method": self._step_method(record, step, methods.get((fov, step.get("round")))),
                             "status": step.get("status"),
                             "seconds": float(step["seconds"]) if step.get("seconds") is not None else
                             Unavailable("not_recorded", f"{self._record_path(fov)}: steps[{index}].seconds")})
            if fov in self.assignments:
                saved, segmentation = self.assignments[fov]
                cells = _get(saved, "inputs", "cells") or {}
                # The methods of the cell run's record; an imported mask has none (baseline G13 (a)).
                methods_of = [m.get("method") for m in (segmentation or {}).get("methods") or []
                              if isinstance(m, dict) and m.get("method")]
                method = ";".join(methods_of) if methods_of else "imported"
                for offset, (name, step_method) in enumerate((("segment", method), ("assign", cells.get("run") or NA))):
                    rows.append({"scope": "fov", "fov_id": fov, "index": len(steps) + offset, "name": name,
                                 "round": NA, "method": step_method, "status": "succeeded",
                                 "seconds": Unavailable("not_recorded", f"no record holds the time of {name}")})
        index = 0
        if self.placement is not None:
            rows.append({"scope": "sample", "fov_id": NA, "index": index, "name": "stitch", "round": NA,
                         "method": _get(self.placement, "method", "method") or NA, "status": "succeeded",
                         "seconds": Unavailable("not_recorded", "no record holds the time of stitch")})
            index += 1
        for part, status in ((self.export or {}).get("parts") or {}).items():
            section = (self.export or {}).get(_SECTIONS.get(part)) or {}
            if status in ("off", "not_run"):
                seconds = NA
            elif isinstance(section.get("wall_s"), (int, float)):
                seconds = float(section["wall_s"])
            else:
                seconds = Unavailable("not_recorded", f"export.json {_SECTIONS.get(part)} records no wall_s")
            rows.append({"scope": "sample", "fov_id": NA, "index": index, "name": part, "round": NA,
                         "method": self._part_method(part, section), "status": status, "seconds": seconds})
            index += 1
        return rows

    @staticmethod
    def _step_method(record, step, registration_methods):
        name = step_name(step.get("name") or "")
        if name.startswith("preprocess:"):
            return name.split(":", 1)[1]
        if name == "register":
            return registration_methods if isinstance(registration_methods, str) else NA
        if name == "find_spots":
            methods = list(dict.fromkeys(m.get("method") for m in step.get("methods") or [] if m.get("method")))
            return ";".join(methods) if methods else NA
        if name == "decode_reads":
            method = _get(record, "config", "pipeline", "decoding", "method")
            mode = _get(record, "config", "readout_mode")
            return NA if method is None else (f"{method} ({mode})" if mode else method)
        if name == "filter_reads":
            return "filter"
        return NA

    @staticmethod
    def _part_method(part, section):
        if part != "segment":
            return NA
        names = []
        for run in section.get("runs") or []:
            method = run.get("method")
            names.append(method.get("method") if isinstance(method, dict) and method.get("method") else
                         "imported" if run.get("import") else None)
        names = [n for n in names if n]
        return ";".join(names) if names else NA

    # readout_totals.csv and population.csv ---------------------------------------------------------------------

    @staticmethod
    def readout_rows(fovs):
        """``readout_totals.csv``: per level the sum over the FOVs with a value, and its fraction of the spots of
        the same FOVs (``count / Σ_{F_L ∩ F_spots} spots``; for ``reads_in_cells`` the FOVs whose ``spots`` and
        ``reads_in_cells`` are both values)."""
        rows = []
        spots = None
        for level in READOUT_LEVELS:
            cells = [row[level] for row in fovs]
            values = [v for v in cells if isinstance(v, int)]
            if values:
                count = sum(values)
            else:
                count = first(*cells)
            if level == "spots":
                spots = count
            if not isinstance(count, int) or not isinstance(spots, int):
                share = first(count, spots)
            else:
                share = fraction(count, sum(row["spots"] for row in fovs
                                            if isinstance(row[level], int) and isinstance(row["spots"], int)),
                                 "spots")
            rows.append({"level": level, "count": count, "fraction_of_spots": share, "fovs": len(values),
                         "overlaps_counted_per_fov": level != "reads_in_cells"})
        return rows

    def population_rows(self):
        counts = (self.assembly or {}).get("counts") if self.e_state is None else None
        no_cells = self.no_cells
        rows = []
        for population in POPULATIONS:
            reason = population[len("excluded_"):] if population.startswith("excluded_") else None
            row = {"population": population}
            for column in ("cells", "molecules"):
                held = _get(counts, column, "excluded") or {}
                if reason in (("cell_not_owner", "cell_duplicate_observation", "outside_grid") if column == "cells"
                              else ("duplicate_observation",)):
                    row[column] = NA  # a population of the other kind only
                elif self.e_state is not None:
                    row[column] = self.e_state
                elif reason is not None and reason not in held:
                    row[column] = NA  # a reason the record does not hold (the route, the duplicate rule)
                elif no_cells is not None and (column == "cells" or population == "counted"):
                    row[column] = no_cells
                elif population == "counted":
                    # n_obs of sample.h5ad, and the retained molecules of counted cells ("Reads assigned to cells").
                    row[column] = (self.measurements.counted_cells() if column == "cells" else
                                   self.measurements.total(self.fovs))
                elif reason is not None:
                    row[column] = int(held[reason])
                else:
                    value = _get(counts, column, population)
                    row[column] = int(value) if value is not None else Unavailable(
                        "not_recorded", f"export.json assembly.counts.{column}.{population}")
            rows.append(row)
        return rows

    # The examples: the replay of the main example and the segmentation previews ------------------------------

    def replay_rounds(self, record):
        """The rounds the replay rebuilds (RD1): the reference round and the first moving round in the order of
        ``registration.csv``; without a registration, the first other round the run processed."""
        reference = reference_round(record)
        moving = self.moving_rounds()
        if not moving:
            moving = [r for r in (_get(record, "preprocessing", "rounds") or {}) if r != reference]
        return [reference, moving[0] if moving else None]

    def example_replay(self):
        """The replay of the main example (None when there is none to replay) and the rows of ``unavailable.csv``
        of what it lacks: the three parts it makes, a moving round it cannot replay (``preprocessing.intensity``),
        the outputs of a moving round's ``post_registration`` entries when its transforms cannot be rebuilt (PR1),
        and the Registration example (RE1)."""
        if self.example_reason is not None:
            return None, [(part, self.example_reason, "") for part in REPLAY_PARTS]
        fov = next(e.fov_id for e in self.examples if e.role == "example_1")
        key = f"fov_id={fov};role=example_1"
        record = self.records[fov]
        if record is None:
            missing = Unavailable("no_run_record", self._record_path(fov))
            return None, [(part, missing, key) for part in REPLAY_PARTS]
        rounds = self.replay_rounds(record)
        replay = replay_example(self.inputs, self.root, fov, record, rounds,
                                lambda name: self._transform(fov, record, name))
        if replay.missing is not None:
            return replay, [(part, replay.missing, key) for part in REPLAY_PARTS]
        rows = [("preprocessing.intensity", missing, f"{key};round={name}")
                for name, missing in replay.round_missing.items()]
        for name, replayed in replay.rounds.items():
            if replayed.post_missing is not None:
                rows += [(part, replayed.post_missing, f"{key};round={name};phase=post_registration")
                         for part in ("preprocessing.example", "preprocessing.intensity")]
        moving = rounds[1]
        if moving is None:
            registration = Unavailable("not_configured", "the run has no moving round")
        else:
            registration = replay.round_missing.get(moving) or replay.rounds[moving].transform_missing
        if registration is not None:
            rows.append(("registration.example", registration, key))
        return replay, rows

    def examples_parts(self):
        """The replay, the segmentation previews and the rows of the parts they make."""
        self.replay, rows = self.example_replay()
        self.previews, preview_rows = _Previews(self).build()
        return rows + preview_rows

    def intensity_rows(self):
        """``intensity.csv``: the rows of the replayed rounds, the reference round first."""
        if self.replay is None or self.replay.missing is not None:
            return []
        names = TABLE_DTYPES["intensity"]
        return [dict(zip(names, row)) for replayed in self.replay.rounds.values() for row in replayed.rows]

    # Parts with no cell ------------------------------------------------------------------------------------

    def no_rounds_reason(self):
        """The reason ``registration.csv`` has no round: no run record, or the first reason of the run records'
        registration (not configured, not reached)."""
        if not any(r is not None for r in self.records.values()):
            return Unavailable("no_run_record", "no FOV of the list has a run record")
        return first(*(self._not_reached(f, r, _get(r, "config", "pipeline", "registration") is not None,
                                         "registration")
                       for f, r in self.records.items() if r is not None))

    def part_rows_for(self, registration):
        rows = [("workflow.steps", reason, f"fov_id={fov}") for fov, reason in self.lane_missing.items()]
        maps_missing = None if self.placement is not None or self.export is not None else self.p_missing
        if not registration:
            reason = self.no_rounds_reason()
            rows.append(("registration.maps", first(reason, maps_missing), ""))
            rows.append(("registration.heatmap", reason, ""))
        elif maps_missing is not None:
            rows.append(("registration.maps", maps_missing, ""))
        if maps_missing is not None:
            rows.append(("readout.per_fov", maps_missing, ""))
        records = [r for r in self.records.values() if r is not None]
        none = Unavailable("no_run_record", "no FOV of the list has a run record")
        rows.append(("files.input", none if not records else None, "entry=images"))
        rows.append(("files.input", none if not records else None if any("codebook" in r for r in records) else
                     Unavailable("not_recorded", "no run record holds codebook"), "entry=codebook"))
        rows.append(("files.input", self.p_missing if self.placement is None else None, "entry=positions"))
        if not records:
            # The distributions and the per-FOV table read the counts of the run records ("Parts, their inputs").
            rows += [("readout.distributions", none, ""), ("readout.table", none, "")]
        rows.append(("files.configuration", none if not records else None if any(
            "workflow_config" in r for r in records) else Unavailable("not_recorded",
                                                                      "no run record holds workflow_config"),
            "entry=workflow"))
        return [(part, value, key) for part, value, key in rows if value is not None]

    # report.json --------------------------------------------------------------------------------------------

    def summary(self, fovs):
        """``report.json`` ``summary``: a median or sum over FOVs is unavailable only when no FOV has the value,
        with the first reason of their cells; the keys of the sample's cells from ``sample.h5ad``, ``cells.csv`` and
        the reads assigned to cells (``_measurements``)."""
        records = [r for r in self.records.values() if r is not None]

        def matched_fraction():
            both = [row for row in fovs if is_value(row["matched"]) and is_value(row["spots"])]
            if not both:
                return first(*(row[c] for row in fovs for c in ("matched", "spots"))) or Unavailable("undefined",
                                                                                                   "no FOV")
            return fraction(sum(row["matched"] for row in both), sum(row["spots"] for row in both), "spots")

        owning = [row for row in fovs if row["owned_area_um2"] is not NA]
        summary = {"fovs_expected": len(self.fov_ids), "fovs_with_run_record": len(records),
                   "fovs_succeeded": sum(r.get("status") == "succeeded" for r in records) if records else
                   Unavailable("no_run_record", "no FOV of the list has a run record"),
                   "spots_median_per_fov": over_fovs([row["spots"] for row in fovs], median),
                   "matched_fraction": matched_fraction(),
                   "cells_median_per_fov": over_fovs([row["n_cells_retained"] for row in owning], median),
                   **self.measurements.summary(fovs)}
        return {key: summary[key] for key in SUMMARY_KEYS}

    def rules(self, fovs, registration, rounds_reason):
        """The problematic FOVs: the rows of ``problematic_fovs.csv``, ``report.json`` ``rules`` (the limit, the
        unit, whether the rule was evaluated, the FOVs it marks) and the ``rule:<rule>`` rows of the rules not
        evaluated for the sample."""
        limits = self.config.limits
        if self.placement is None or self.p_missing is not None:
            flags = self.p_missing
        else:
            flags = {fov: ("" if pd.isna(row["flags"]) else row["flags"]) for fov, row in self.p_rows.items()}
        rows, evaluated, part_rows = evaluate_rules(
            fovs, registration, limits, RuleInputs(self.fov_ids, self.records, flags, rounds_reason,
                                                   self.shift_medians))
        rules = {}
        for rule in PROBLEM_RULES:
            if rule == "spots_near_zero":
                limit = {"fraction_of_median": limits.spots_near_zero}
            elif rule in RULE_UNITS:
                setting = getattr(limits, rule)
                limit = {"k": setting.k, "floor": setting.floor, "absolute": setting.absolute}
            else:
                limit = NA  # a categorical rule
            rules[rule] = {"limit": limit, "unit": RULE_UNITS.get(rule, NA), **evaluated[rule]}
        return rows, rules, part_rows

    def resolved_runs(self):
        assignment = {"name": self.assignment,
                      "folders": {fov: f"{_layout.CHECKPOINTS}/{fov}/assignment/{self.assignment}"
                                  for fov in self.assignments}}
        if self.route == "after_fusion":
            section = (self.export or {}).get("segmentation") or {}
            runs = {"cells": self.segmentation or section.get("cells"),
                    "nuclei": None if self.segmentation else section.get("nuclei")}
        else:
            runs = {}
            for fov, (saved, _) in self.assignments.items():
                inputs = saved.get("inputs") or {}
                runs[fov] = {"cells": self.segmentation or _get(inputs, "cells", "run"),
                             "nuclei": None if self.segmentation else _get(inputs, "nuclei", "run")}
        return assignment, {"name": self.segmentation, "runs": runs}


# --- Tables ------------------------------------------------------------------------------------------------------

def _frame(name, rows):
    dtypes = TABLE_DTYPES[name]
    data = {c: pd.Series([None if isinstance(r[c], (Unavailable, _Marker)) else r[c] for r in rows],
                         dtype=object).astype(dtypes[c]) for c in dtypes}
    return pd.DataFrame(data, columns=list(dtypes))


def _part_rank(part):
    """A part's place in the order of ``unavailable.csv``; ``rule:<rule>`` sorts with the problematic FOVs."""
    if part.startswith("rule:"):
        return REPORT_PARTS.index("summary.problematic_fovs"), PROBLEM_RULES.index(part[len("rule:"):])
    return REPORT_PARTS.index(part), -1


#: The tables in the order of the rows of ``unavailable.csv``: rows without a table first, ``summary`` last.
_TABLE_RANK = {name: i for i, name in enumerate(REPORT_TABLES)} | {"summary": len(REPORT_TABLES)}


def _unavailable_rows(tables, part_rows, summary):
    """The rows of ``unavailable.csv``, in the contract's order (``summary`` keys after the tables)."""
    entries = []
    for k_index, (key, value) in enumerate(summary.items()):
        if isinstance(value, Unavailable):
            entries.append(((*_part_rank("summary.key_numbers"), _TABLE_RANK["summary"], k_index, 0),
                            {"part": "summary.key_numbers", "table": "summary", "column": key, "row_key": "",
                             "reason": value.reason, "detail": value.detail}))
    for position, (part, value, key) in enumerate(part_rows):
        entries.append(((*_part_rank(part), -1, 0, position),
                        {"part": part, "table": None, "column": None, "row_key": key, "reason": value.reason,
                         "detail": value.detail}))
    for table, rows in tables.items():
        parts = COLUMN_PARTS.get(table, {})
        for c_index, column in enumerate(TABLE_DTYPES[table]):
            for r_index, row in enumerate(rows):
                value = row[column]
                if isinstance(value, Unavailable):
                    entries.append(((*_part_rank(parts[column]), _TABLE_RANK[table], c_index, r_index),
                                    {"part": parts[column], "table": table, "column": column,
                                     "row_key": row_key(table, _plain(row)), "reason": value.reason,
                                     "detail": value.detail}))
    entries.sort(key=lambda e: e[0])
    return [row for _, row in entries]


def _plain(row):
    return {k: (None if isinstance(v, (Unavailable, _Marker)) else v) for k, v in row.items()}


# --- The entry -----------------------------------------------------------------------------------------------------

def collect_report(output_root, sample: str, fov_ids: Sequence[str], *, dataset_id: str, sample_id: str,
                   assignment: str | None = "default", segmentation: str | None = None,
                   config: ReportConfig | None = None, output_dir=None,
                   reference_image: str | None = None) -> tuple[Path, dict]:
    """Collect one sample's records into its report folder; return the folder and the ``report.json`` content.

    The checks 1 to 7 of docs/report-contract.md ("Checks before anything is
    written") run first, in order, each raising with the folder unchanged. The
    tables of ``REPORT_TABLES``, the page ``report.html`` and the record are
    written into a staging folder beside the report folder, which then replaces
    the earlier report's own files (RP1).

    Parameters
    ----------
    output_root : str or Path
        The dataset's output root.
    sample : str
        The sample: ``stitching/<sample>/``, ``export/<sample>/`` and ``report/<sample>/``.
    fov_ids : Sequence[str]
        The FOV list, nonempty and unique, in report order.
    dataset_id, sample_id : str
        The ``Dataset``'s IDs, which every run record must hold.
    assignment : str or None
        The §2.9 assignment name of the route per FOV; None for a sample without cells.
    segmentation : str or None
        The segmentation run the examples show; None for the runs the assignment or export read.
    config : ReportConfig or None
        None is ``ReportConfig()``.
    output_dir : str or Path or None
        None is ``<output_root>/report/<sample>/``.
    reference_image : str or None
        The ``Dataset``'s ``reference_image`` pattern, where the Summary's Z size is read when no record holds a grid;
        None is its default ``images/ref_merged/{fov_id}.tif``.

    Raises
    ------
    TypeError
        config is not a ReportConfig.
    ValueError
        Arguments of the wrong form, a run record, export or placement that does
        not belong to the sample or the FOV list, a marker gene not in the gene
        list, example FOVs outside the list.
    FileNotFoundError
        No FOV of the list has a run record and there is no export.
    FileExistsError
        The report folder holds a file that is not the report's own.
    """
    from starfinder._records import _write_table, jsonable, software, write_json
    from starfinder.export._record import Timer
    from ._page import budget_bytes, build_page, identity_facts
    timer = Timer()
    folder, collection = _prepare(output_root, sample, fov_ids, dataset_id=dataset_id, sample_id=sample_id,
                                  assignment=assignment, segmentation=segmentation, config=config,
                                  output_dir=output_dir)
    collection.reference_image = reference_image
    inputs, records, placement, export = collection.inputs, collection.records, collection.placement, collection.export
    fov_ids, config = collection.fov_ids, collection.config
    fovs = collection.fovs_rows()
    registration = collection.registration_rows()
    measurements = collection.measurements
    cells = measurements.cells_rows()
    tables = {"fovs": fovs, "registration": registration, "steps": collection.steps_rows(registration),
              "readout_totals": collection.readout_rows(fovs), "population": collection.population_rows(),
              "problematic_fovs": [], "cells": [] if isinstance(cells, Unavailable) else cells,
              "mecr": measurements.mecr_rows()}
    along = measurements.reads_along_z_rows()
    tables["reads_along_z"] = [] if isinstance(along, Unavailable) else along
    tables["problematic_fovs"], rules, rule_rows = collection.rules(
        fovs, registration, None if registration else collection.no_rounds_reason())
    example_rows = collection.examples_parts()
    tables["intensity"] = collection.intensity_rows()
    summary = collection.summary(fovs)
    identity, identity_rows = identity_facts(collection)
    unavailable = _unavailable_rows(tables, identity_rows + collection.part_rows_for(registration) + example_rows
                                    + rule_rows + measurements.part_rows(), summary)
    tables["unavailable"] = unavailable
    resolved_assignment, resolved_segmentation = collection.resolved_runs()
    frames = {name: _frame(name, tables[name]) for name in COLLECTED_TABLES}
    generated_at = _now()
    config_json = jsonable(config)
    page, _ = build_page(collection, tables, frames, [_plain(row) for row in unavailable], _plain(summary), identity,
                         generated_at, config_json, folder)
    page_bytes = page.encode("utf-8")
    marker_figures = len(config.markers) or 1

    folder.parent.mkdir(parents=True, exist_ok=True)
    staging = folder.parent / f".{folder.name}.{uuid.uuid4().hex}.tmp"
    try:
        (staging / TABLE_DIR).mkdir(parents=True)
        (staging / PAGE_FILE).write_bytes(page_bytes)
        # The page records its bytes and its budget (SB1); a page above the budget is still written.
        files = [{"path": PAGE_FILE, "sha256": _sha256(page_bytes), "bytes": len(page_bytes),
                  "budget_bytes": budget_bytes(len(fov_ids), marker_figures)}]
        for name in COLLECTED_TABLES:
            file_name, dtypes = _write_table(frames[name], staging / TABLE_DIR, name, "csv")
            data = (staging / TABLE_DIR / file_name).read_bytes()
            files.append({"path": f"{TABLE_DIR}/{file_name}", "sha256": _sha256(data), "bytes": len(data),
                          "dtypes": dtypes})
        record = {
            "format_version": FORMAT_VERSION, "stage": STAGE, "sample": sample, "dataset_id": dataset_id,
            "sample_id": sample_id, "fov_ids": fov_ids, "route": collection.route,
            "assignment": resolved_assignment, "segmentation": resolved_segmentation, "generated_at": generated_at,
            "software": software(), "config": config_json,
            "stages": {"run_records": [f for f in fov_ids if records[f] is not None],
                       "placement": placement is not None, "export": export is not None,
                       "parts": dict((export or {}).get("parts") or {})},
            "inputs": list(inputs.entries.values()), "files": files,
            "unavailable": [_plain(row) for row in unavailable],
            "examples": collection.examples_record(),
            "replay": None if collection.replay is None else collection.replay.record,
            "rules": {rule: _plain(entry) for rule, entry in rules.items()}, "summary": _plain(summary),
            **timer.record()}
        write_json(record, staging / REPORT_FILE)
        _replace(folder, staging, [entry["path"] for entry in files])
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return folder, json.loads((folder / REPORT_FILE).read_text(encoding="utf-8"))


def report_sample(dataset, sample: str, fov_ids: Sequence[str], *, assignment: str | None = "default",
                  segmentation: str | None = None, config: ReportConfig | None = None,
                  output_dir=None) -> ReportRecord:
    """``Dataset.report``: :func:`collect_report` on the dataset's output root and IDs, as a :class:`ReportRecord`."""
    folder, record = collect_report(dataset.output_root, sample, fov_ids, dataset_id=dataset.dataset_id,
                                    sample_id=dataset.sample_id, assignment=assignment, segmentation=segmentation,
                                    config=config, output_dir=output_dir, reference_image=dataset.reference_image)
    return ReportRecord(Path(folder), record)


def report_examples(output_root, sample: str, fov_ids: Sequence[str], *, dataset_id: str, sample_id: str,
                    assignment: str | None = "default", segmentation: str | None = None,
                    config: ReportConfig | None = None):
    """The images of the examples as :func:`collect_report` makes them, without writing anything: the replay of
    the main example (:class:`._replay.Replay`, or None when there is none to replay) and the segmentation preview
    of each example role (:class:`._previews.Preview`, or :class:`._values.Unavailable`). The checks before
    anything is written apply, as for the report."""
    _, collection = _prepare(output_root, sample, fov_ids, dataset_id=dataset_id, sample_id=sample_id,
                             assignment=assignment, segmentation=segmentation, config=config, output_dir=None,
                             check_folder=False)
    collection.fovs_rows()
    collection.registration_rows()
    collection.examples_parts()
    return collection.replay, collection.previews


def report_measurements(output_root, sample: str, fov_ids: Sequence[str], *, dataset_id: str, sample_id: str,
                        assignment: str | None = "default", segmentation: str | None = None,
                        config: ReportConfig | None = None):
    """The static figures of "Sample measurements" as the report makes them, without writing anything
    (:class:`._measurements.SampleFigures`): the per-cell distributions, the marker figures with the data of the
    group selector, and the reads along Z; a part that cannot be drawn is its :class:`._values.Unavailable`, the
    reason its row of ``unavailable.csv`` holds. The checks before anything is written apply, as for the report."""
    _, collection = _prepare(output_root, sample, fov_ids, dataset_id=dataset_id, sample_id=sample_id,
                             assignment=assignment, segmentation=segmentation, config=config, output_dir=None,
                             check_folder=False)
    collection.fovs_rows()
    return collection.measurements.figures()


def _prepare(output_root, sample, fov_ids, *, dataset_id, sample_id, assignment, segmentation, config, output_dir,
             check_folder=True):
    """The checks 1 to 7 before anything is written, in order; the report folder and the collection."""
    root = Path(output_root)
    fov_ids, config = _check_arguments(sample, fov_ids, config)
    if assignment is not None and (not isinstance(assignment, str) or not assignment):
        raise ValueError(f"assignment must be a saved assignment name or None; got {assignment!r}")
    if segmentation is not None and (not isinstance(segmentation, str) or not segmentation):
        raise ValueError(f"segmentation must be a run name or None; got {segmentation!r}")
    inputs = _Inputs(root)
    export_dir, stitching_dir = _layout.export_dir(root, sample), _layout.stitching_dir(root, sample)
    if not any(_layout.run_record(root, f).is_file() for f in fov_ids) and \
            not (export_dir / _layout.EXPORT_FILE).is_file():
        raise FileNotFoundError(f"nothing to report for sample {sample!r}: no FOV of the list has a run record "
                                f"({root / _layout.CHECKPOINTS}/<fov>/{_layout.RUN_RECORD}) and there is no "
                                f"{export_dir / _layout.EXPORT_FILE}")
    records = _check_run_records(inputs, root, fov_ids, dataset_id, sample_id)
    export = _check_export(inputs, export_dir, sample, fov_ids, assignment)
    placement = _check_placement(inputs, stitching_dir, fov_ids)
    _check_markers(inputs, config, records, export, export_dir)
    _check_examples(config, fov_ids)
    folder = Path(output_dir) if output_dir is not None else _layout.report_dir(root, sample)
    if check_folder:
        own_files(folder)
    collection = _Collection(inputs, root, sample, fov_ids, records, export, placement, assignment, segmentation,
                             config)
    collection.dataset_id = dataset_id
    return folder, collection


def _replace(folder: Path, staging: Path, paths):
    """Remove the earlier report's own files and move the new ones in, ``report.json`` last (RP1)."""
    if not folder.exists():
        os.replace(staging, folder)
        return
    earlier = own_files(folder)
    for relative in [REPORT_FILE] + [p for p in earlier if p != REPORT_FILE]:
        (folder / relative).unlink(missing_ok=True)
    for directory in sorted({(folder / p).parent for p in earlier}, key=lambda d: len(d.parts), reverse=True):
        if directory != folder and directory.is_dir() and not any(directory.iterdir()):
            directory.rmdir()
    for relative in list(paths) + [REPORT_FILE]:
        (folder / relative).parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging / relative, folder / relative)
