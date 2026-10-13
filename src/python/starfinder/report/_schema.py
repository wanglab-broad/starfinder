"""The closed lists, the table schemas and the reader of the report's record (docs/report-contract.md).

Every column's dtype is the contract's type in the checkpoint table form: a column
that can be null is of a nullable dtype (``Int64``, ``boolean``), a column that is
never null of the plain one (``int64``, ``bool``); strings are ``string`` and floats
``float64`` either way.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

#: The reasons a value or a part can be unavailable, in the order that decides which one a value gets (P1).
UNAVAILABLE_REASONS = (
    "no_run_record", "run_not_finished", "step_failed", "not_configured", "not_recorded", "no_placement",
    "fov_not_placed", "no_export", "part_not_written", "no_segmentation", "no_markers", "no_stack",
    "input_missing", "input_changed", "field_not_saved", "replay_differs", "undefined")

#: The parts of the page, in page order ("Parts, their inputs and their reasons").
REPORT_PARTS = (
    "summary.identity", "summary.key_numbers", "summary.problematic_fovs",
    "workflow.steps", "workflow.fov_status", "workflow.time",
    "preprocessing.example", "preprocessing.intensity",
    "registration.maps", "registration.heatmap", "registration.example",
    "readout.totals", "readout.per_fov", "readout.distributions", "readout.table",
    "assembly.cells_per_fov", "assembly.placement", "assembly.examples", "assembly.accounting",
    "sample.per_cell", "sample.markers", "sample.marker_zoom", "sample.mecr", "sample.reads_along_z",
    "files.input", "files.configuration", "files.output")

#: The tables of ``tables/``, in the order of the output folder ("The output folder").
REPORT_TABLES = ("fovs", "registration", "steps", "readout_totals", "population", "problematic_fovs", "cells",
                 "mecr", "reads_along_z", "intensity", "unavailable")

#: The rules of the problematic FOVs, in the order of their rows (D5).
PROBLEM_RULES = ("fov_failed", "fov_not_run", "shift_residual", "alignment_score", "registration_rejected",
                 "registration_fallback", "correct_form_fraction", "matched_fraction", "spots_near_zero",
                 "not_placed", "stage_position_fallback", "reads_no_cell")

REPORT_FILE = "report.json"
TABLE_DIR = "tables"
FORMAT_VERSION = 1
STAGE = "report"

_AXES = ("z", "y", "x")

#: Each table's columns and dtypes, in column order ("The tables").
TABLE_DTYPES = {
    "fovs": {
        "fov_id": "string", "list_index": "int64", "example": "string", "run_record": "bool", "status": "string",
        "failed_step": "string", "failed_round": "string", "started_at": "string", "ended_at": "string",
        "run_seconds": "float64", "placement_status": "string",
        **{f"correction_{a}": "float64" for a in _AXES},
        **{c: "float64" for c in ("footprint_y0", "footprint_y1", "footprint_x0", "footprint_x1", "centre_y",
                                   "centre_x")},
        **{c: "Int64" for c in ("spots", "correct_form", "matched", "merged_by_deduplication", "passed_filter")},
        **{c: "float64" for c in ("correct_form_fraction", "matched_fraction", "merged_fraction",
                                   "passed_filter_fraction")},
        **{c: "Int64" for c in ("n_cells", "n_cells_retained", "n_molecules", "n_molecules_retained",
                                 "reads_in_cells")},
        "reads_in_cells_fraction": "float64", "owned_area_um2": "float64", "cell_density_per_mm2": "float64"},
    "registration": {
        "fov_id": "string", "round": "string", "outcome": "string", "fallback": "boolean", "methods": "string",
        "transform_source": "string", **{f"shift_{a}": "float64" for a in _AXES},
        **{f"residual_{a}": "float64" for a in _AXES}, "alignment_score": "float64"},
    "steps": {"scope": "string", "fov_id": "string", "index": "int64", "name": "string", "round": "string",
              "method": "string", "status": "string", "seconds": "float64"},
    "readout_totals": {"level": "string", "count": "Int64", "fraction_of_spots": "float64", "fovs": "int64",
                       "overlaps_counted_per_fov": "bool"},
    "population": {"population": "string", "cells": "Int64", "molecules": "Int64"},
    "problematic_fovs": {"fov_id": "string", "section": "string", "rule": "string", "round": "string",
                         "axis": "string", "measured": "float64", "sample_median": "float64", "limit": "float64",
                         "unit": "string", "reason": "string"},
    "cells": {"obs_name": "string", "fov_id": "string", "reads": "int64", "genes": "int64", "size": "float64",
              "size_unit": "string"},
    "mecr": {"gene_a": "string", "gene_b": "string", "threshold": "int64", "cells_a": "Int64", "cells_b": "Int64",
             "both": "Int64", "either": "Int64", "mecr": "float64"},
    "reads_along_z": {"fov_id": "string", "z": "int64", "reads": "int64"},
    "intensity": {"round": "string", "step_index": "int64", "phase": "string", "step": "string",
                  "channel": "string", "bin": "int64", "bin_low": "float64", "bin_high": "float64",
                  "count": "int64"},
    "unavailable": {"part": "string", "table": "string", "column": "string", "row_key": "string",
                    "reason": "string", "detail": "string"},
}

#: Each table's key columns, in key order; a row key is ``name=value`` per key column joined by ``;``.
TABLE_KEYS = {"fovs": ("fov_id",), "registration": ("fov_id", "round"), "steps": ("scope", "fov_id", "index"),
              "readout_totals": ("level",), "population": ("population",),
              "problematic_fovs": ("fov_id", "rule", "round", "axis"), "cells": ("obs_name",),
              "mecr": ("gene_a", "gene_b"), "reads_along_z": ("fov_id", "z"),
              "intensity": ("round", "step_index", "channel", "bin"),
              "unavailable": ("part", "table", "column", "row_key")}

#: The levels of ``readout_totals.csv`` and the populations of ``population.csv``, in row order.
READOUT_LEVELS = ("spots", "correct_form", "matched", "merged_by_deduplication", "passed_filter", "reads_in_cells")
POPULATIONS = ("input", "retained", "excluded_not_owner", "excluded_fov_not_placed",
               "excluded_duplicate_observation", "excluded_cell_not_owner", "excluded_cell_duplicate_observation",
               "excluded_outside_grid", "counted")

#: The keys of ``report.json``, in the contract's order.
RECORD_KEYS = ("format_version", "stage", "sample", "dataset_id", "sample_id", "fov_ids", "route", "assignment",
               "segmentation", "generated_at", "software", "config", "stages", "inputs", "files", "unavailable",
               "examples", "replay", "rules", "summary", "wall_s", "cpu_s", "peak_rss_kib")

#: The keys of ``report.json`` ``summary``, in the contract's order.
SUMMARY_KEYS = ("fovs_expected", "fovs_with_run_record", "fovs_succeeded", "spots_median_per_fov", "matched_fraction",
                "counted_cells", "reads_per_cell_median", "genes_per_cell_median", "cells_median_per_fov",
                "reads_in_cells", "h5ad_whole_count_sum", "reads_in_cells_equal")
#: The keys of each rule's entry of ``report.json`` ``rules``: the limit applied, the unit, whether it was
#: evaluated and the number of FOVs it marks.
RULE_KEYS = ("limit", "unit", "evaluated", "fovs_marked")
#: The unit of each rule on values (docs/report-algorithms.md, "The rules of the first delivery"); the
#: categorical rules have none.
RULE_UNITS = {"shift_residual": "voxels", "alignment_score": "dimensionless", "correct_form_fraction": "fraction",
              "matched_fraction": "fraction", "spots_near_zero": "reads"}


#: The values the staged delivery of the report (W-452 to W-456) left to a later issue: none, since the report is
#: complete. Every null cell is not applicable by its column's rule or named by one row of ``unavailable.csv``.
PENDING_VALUES = ()


#: The kinds of a file of ``report.json`` ``inputs``.
INPUT_KINDS = ("run_record", "transforms", "field", "raw_image", "supplied_statistics", "codebook",
               "segmentation_record", "assignment_record", "label_image", "reference_image", "placement",
               "placement_table", "export_record", "export_table", "anndata", "fused_image")


def row_key(table: str, row) -> str:
    """The row key of a row of a table: ``name=value`` for each key column, in key order, joined by ``;``.

    A null key value is written empty (``scope=sample;fov_id=;index=0``).
    """
    return ";".join(f"{name}={'' if row.get(name) is None else row[name]}" for name in TABLE_KEYS[table])


@dataclass(frozen=True)
class ReportRecord:
    """The result of ``Dataset.report``: the report folder and the content of its ``report.json``."""

    path: Path
    record: dict


def read_report(folder) -> dict:
    """The record ``report.json`` of a report folder, checked against the files it lists.

    Parameters
    ----------
    folder : str or Path
        A sample report's folder, by convention ``<OUTPUT>/report/<sample>/``.

    Returns
    -------
    dict
        The parsed ``report.json``.

    Raises
    ------
    FileNotFoundError
        The folder has no ``report.json``.
    ValueError
        A record of another format version or stage, or a file of its
        ``files`` list that is missing or whose SHA-256 or size differs from
        the record (naming the file).
    """
    from starfinder.segmentation._persist import file_sha256
    folder = Path(folder)
    path = folder / REPORT_FILE
    if not path.is_file():
        raise FileNotFoundError(f"no report record at {path}")
    record = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(record, dict) or record.get("format_version") != FORMAT_VERSION or record.get("stage") != STAGE:
        raise ValueError(f"{path} is not a version {FORMAT_VERSION} report record")
    for entry in record.get("files") or []:
        file = folder / entry["path"]
        if not file.is_file():
            raise ValueError(f"report file {file} is missing (recorded SHA-256 {entry['sha256']})")
        digest = file_sha256(file)
        if digest != entry["sha256"] or file.stat().st_size != entry["bytes"]:
            raise ValueError(f"report file {file} has SHA-256 {digest}, the record {entry['sha256']}")
    return record
