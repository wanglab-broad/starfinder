"""Hand-written records of one sample for the §2.11 report (docs/report-algorithms.md, "Fixtures").

``report_records``: an output root of sample ``s1`` with four FOVs ``F1`` to ``F4`` on a
2 × 2 grid, each of 4 × 16 × 16 voxels at spacing (0.5, 0.25, 0.25) µm, nominal step 3 µm.
Each FOV's ``checkpoints/<fov>/run.json`` is written as JSON in the form
``dataset/_run_record.py`` writes it, with R1 to R4: status ``succeeded``, the steps of
loading, preprocessing (``min_max_normalization`` then ``white_tophat``), registration (one
translation step) and detection to filtering with stated seconds, deduplication and a filter
with ``end_bases``. The transform entries are made by ``_checkpoint.py``:
``_transform_json`` from the package's own result types. The placement is saved with
``save_placement``: every FOV ``image_supported`` at the origins of the page (F2 corrected by
+0.25 µm in X), review ``not_needed``. The run records name raw inputs whose bytes are
fixed here; the files are written only on request (the page's fixture writes no image).

``report_cells``: ``report_records`` with each FOV's saved assignment ``default`` (and its cell run) and an
export folder that records their SHA-256 as the assembly does: ``export/s1/tables/fovs.csv``,
``cells.csv``, ``molecules.csv``, ``counts.csv`` through the checkpoint table writer and
``sample.h5ad`` through ``assignment.write_h5ad``, with ``export.json`` written through
``export/_record.py`` so that ``verify_section_hashes`` passes; route ``per_fov``.

Every variant of the page's table is a ``variant`` name. Nothing is read from a run
directory, no image is written and no random number is used.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from starfinder.image import ImageMetadata
from starfinder.report import MarkerGroup, ReportConfig, _figures, _page
from starfinder.report._collect import collect_report

DATASET, SAMPLE, SAMPLE_ID = "report", "s1", "s1"
FOVS = ("F1", "F2", "F3", "F4")
ROUNDS = ("round1", "round2", "round3")
MOVING = ("round2", "round3")
CHANNELS = ("ch00", "ch01", "ch02", "ch03")
GENES = ("G1", "G2", "G3", "G4", "G5")
SHAPE_ZYX = (4, 16, 16)
SPACING_ZYX = (0.5, 0.25, 0.25)
IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
ORIGINS = {"F1": (0.0, 0.0, 0.0), "F2": (0.0, 0.0, 3.25), "F3": (0.0, 3.0, 0.0), "F4": (0.0, 3.0, 3.0)}
NOMINAL = {"F1": (0.0, 0.0, 0.0), "F2": (0.0, 0.0, 3.0), "F3": (0.0, 3.0, 0.0), "F4": (0.0, 3.0, 3.0)}
TRANSLATIONS = {"round2": {"F1": (0, 1, 2), "F2": (0, 1, 3), "F3": (0, 2, 2), "F4": (0, 1, 2)},
                "round3": {"F1": (0, -1, 0), "F2": (0, -1, 0), "F3": (1, -1, 0), "F4": (0, -1, 8)}}
NCC_AFTER = {"round2": {"F1": 0.9, "F2": 0.9, "F3": 0.85, "F4": 0.9},
             "round3": {"F1": 0.88, "F2": 0.88, "F3": 0.87, "F4": 0.3}}
COUNTS = {"spots": {"F1": 1000, "F2": 1200, "F3": 800, "F4": 1000},
          "endpoint_valid": {"F1": 800, "F2": 960, "F3": 400, "F4": 800},
          "assigned": {"F1": 700, "F2": 840, "F3": 560, "F4": 700},
          "merged_reads": {"F1": 10, "F2": 20, "F3": 0, "F4": 10},
          "accepted": {"F1": 650, "F2": 800, "F3": 500, "F4": 640}}
# The seconds of each step, dyadic so that their sums are exact.
PER_ROUND_SECONDS = {"preprocess:min_max_normalization": 0.25, "preprocess:white_tophat": 0.25, "register": 0.5,
                     "extract_round": 0.125}
STARTED = {"F1": "2026-10-12T10:00:00+00:00", "F2": "2026-10-12T10:01:00+00:00",
           "F3": "2026-10-12T10:02:00+00:00", "F4": "2026-10-12T10:03:00+00:00"}
ENDED = {"F1": "2026-10-12T10:00:08+00:00", "F2": "2026-10-12T10:01:08+00:00",
         "F3": "2026-10-12T10:02:08+00:00", "F4": "2026-10-12T10:03:08+00:00"}
# The variants of docs/report-algorithms.md ("Fixtures"); RC13's three copies of ``failed`` are failed_<last entry>.
RECORD_VARIANTS = ("failed", "failed_applying", "failed_estimating", "failed_succeeded", "not_run", "only_f1",
                   "stage_off", "stage_off_registration", "no_placement", "one", "equal", "no_spot", "older",
                   "older_transforms", "placement_states", "placement_supplied")
CELL_VARIANTS = ("nocal", "mismatch", "no_markers")
#: n_cells_retained of report_cells; F1's include the retained cell that is excluded_no_nucleus.
CELLS_RETAINED = {"F1": 21, "F2": 0, "F3": 7, "F4": 14}


@dataclass
class ReportCase:
    """An output root and the arguments of its report."""

    root: Path
    fov_ids: list = field(default_factory=lambda: list(FOVS))
    config: ReportConfig = field(default_factory=ReportConfig)
    sample: str = SAMPLE
    dataset_id: str = DATASET
    sample_id: str = SAMPLE_ID

    @property
    def folder(self) -> Path:
        return self.root / "report" / self.sample

    def checkpoint(self, fov) -> Path:
        return self.root / "checkpoints" / fov

    def run_record(self, fov) -> dict:
        return json.loads((self.checkpoint(fov) / "run.json").read_text())

    def write_run_record(self, fov, record) -> None:
        (self.checkpoint(fov) / "run.json").write_text(json.dumps(record, indent=2) + "\n")

    def collect(self, **changes):
        arguments = dict(sample=self.sample, fov_ids=self.fov_ids, dataset_id=self.dataset_id,
                         sample_id=self.sample_id, config=self.config) | changes
        return collect_report(self.root, **arguments)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def metadata(frame="fov") -> ImageMetadata:
    return ImageMetadata(frame, SPACING_ZYX, (0.0, 0.0, 0.0), IDENTITY, "µm")


def jsonable(value):
    from starfinder._records import jsonable as _jsonable
    return json.loads(json.dumps(_jsonable(value)))


# --- Run records ---------------------------------------------------------------------------------

def raw_bytes(root, round_name, fov, channel) -> bytes:
    """The bytes of a raw input the run records name (written only by ``write_inputs``)."""
    return f"raw {round_name} {fov} {channel}\n".encode()


def raw_path(root, round_name, fov, channel) -> Path:
    return Path(root) / "in" / round_name / fov / f"{channel}.tif"


def write_inputs(root, fovs=FOVS) -> None:
    for fov in fovs:
        for r in ROUNDS:
            for c in CHANNELS:
                path = raw_path(root, r, fov, c)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw_bytes(root, r, fov, c))


CODEBOOK_TEXT = "".join(f"{g},{b}\n" for g, b in zip(GENES, ("CACG", "CGCC", "CTCA", "CAAC", "CCGC")))


def registration_result(displacement):
    """A RegistrationResult of one translation, the form FOV.run records."""
    from starfinder.registration import TranslationConfig, WarpConfig
    from starfinder.registration._types import RegistrationDiagnostics, RegistrationResult, TranslationTransform
    transform = TranslationTransform(tuple(float(v) for v in displacement), SHAPE_ZYX, SHAPE_ZYX, metadata(),
                                     metadata())
    diagnostics = RegistrationDiagnostics("translation", "scipy_fft", TranslationConfig())
    return RegistrationResult(transform, diagnostics, WarpConfig())


def transform_entry(displacement):
    from starfinder._checkpoint import _transform_json
    return jsonable(_transform_json(registration_result(displacement), 0, None))


def _qc(ncc_after):
    return {"status": "ok", "values": {"coverage": 1.0, "ncc_before": 0.25, "ncc_after": ncc_after,
                                       "ncc_gain": None if ncc_after is None else ncc_after - 0.25},
            "units": {}, "counts": {}, "reasons": {}, "config": {}, "details": {}}


WARP = {"backend": "translation", "boundary_mode": "constant", "fill_value": 0, "output_dtype": "input",
        "integer_rounding": "nearest_even", "clip_to_dtype": True, "fft_workers": 1}


def estimation_entry(ncc_after, outcome="succeeded", *, fallback=False, method="translation", qc=True):
    return {"record": "estimation", "step": 0, "attempt": 0, "requested_method": "translation",
            "actual_method": method, "fallback": fallback, "backend": "scipy_fft", "backend_versions": {},
            "reference": "round1", "reference_sha256": None,
            "config": {"backend": "scipy_fft", "fft_workers": 1, "method": "translation"},
            "outcome": outcome, "failure": None, "qc": _qc(ncc_after) if qc else None}


def application_entry(ncc_after, outcome="succeeded", *, qc=True):
    return {"record": "application", "outcome": outcome, "application_config": dict(WARP), "failure": None,
            "qc": _qc(ncc_after) if qc else None}


def pipeline_config(*, deduplication=True, registration=True):
    return {
        "load": {"channel_labels": list(CHANNELS), "source_paths": None, "subdir": "", "source_axes": None,
                 "series_index": None, "time_index": None, "channel_index": None, "crop_policy": "error",
                 "conversion": None, "metadata": None},
        "rotation_degrees": None,
        "preprocessing": {"steps": [{"config": {"output_dtype": "uint8", "output_range": [0, 255],
                                                "scope": "per_channel", "snr_threshold": None,
                                                "rounding": "truncate", "method": "min_max_normalization"},
                                     "save_as": None},
                                    {"config": {"radius_yx": 2, "method": "white_tophat"}, "save_as": None}],
                          "post_registration": [], "extraction_source": None, "registration_source": None,
                          "supplied_statistics": None},
        "registration": None if not registration else {
            "steps": [{"config": {"backend": "scipy_fft", "fft_workers": 1, "method": "translation"},
                       "recovery": None, "signal": None}],
            "signal": {"mode": "max", "reference_channel": None, "moving_channel": None}, "warp": None,
            "reference_round": None, "qc": {}},
        "spot_finding": {"threshold_mode": "adaptive", "threshold_value": 0.2, "min_distance_voxels": 1,
                         "exclude_border": True, "channel_labels": None, "measure_peak_intensity": True,
                         "merge_radius_zyx": None, "method": "local_maxima"},
        "extraction": {"neighborhood_radius_zyx": [0, 1, 1], "sampling": "nearest", "boundary": "zero",
                       "background": None},
        "decoding": {"negative_policy": "reject", "diagnostics": False, "method": "wta"},
        "filtering": {"call_statuses": ["assigned"], "score_bounds": {}, "end_bases": "CC", "start_base": "C",
                      "exclude_invalid_endpoints": False, "exclude_duplicates": True},
        "scoring": {"method": "bgcorr_probability"},
        "deduplication": {"distance_voxels": 1.0, "compatibility": "same_sequence"} if deduplication else None}


def _steps(*, deduplication=True, registration=True, fail_at=None):
    detection = {"stage": "spot_finding", "method": "local_maxima",
                 "config_type": "starfinder.spot_finding.LocalMaximaConfig",
                 "implementation": "starfinder.spot_finding._methods._local_maxima", "config": {}, "requires": {},
                 "artifacts": [], "execution": {"device": "cpu", "framework": None, "threads": {}}}

    def step(name, round_name=None, seconds=None, **extra):
        seconds = PER_ROUND_SECONDS[name] if seconds is None else seconds
        return {"name": name, "round": round_name, "seconds": seconds, "status": "succeeded", **extra}

    steps = [step("load_images", seconds=0.5), step("preprocess:min_max_normalization", "round1"),
             step("preprocess:white_tophat", "round1"), step("find_spots", seconds=1.0, methods=[detection]),
             step("extract_round", "round1")]
    for name in MOVING:
        steps += [step("preprocess:min_max_normalization", name), step("preprocess:white_tophat", name)]
        if registration:
            steps.append(step("register", name))
        if fail_at == ("register", name):
            steps[-1]["status"] = "failed"
            return steps
        steps.append(step("extract_round", name))
    steps += [step("decode_barcodes", seconds=0.25), step("score_reads", seconds=0.125)]
    if deduplication:
        steps.append(step("deduplicate_reads", seconds=0.125))
    steps.append(step("filter_reads", seconds=0.125))
    return steps


def _preprocessing():
    entries = [{"index": 0, "stage": "steps", "step": "min_max_normalization", "config": {}, "fitted": {},
                "diagnostics": {}, "input_dtype": "uint16", "output_dtype": "uint8", "save_as": None},
               {"index": 1, "stage": "steps", "step": "white_tophat", "config": {"radius_yx": 2}, "fitted": {},
                "diagnostics": {}, "input_dtype": "uint8", "output_dtype": "uint8", "save_as": None}]
    return {"recipe": {"steps": ["min_max_normalization", "white_tophat"], "post_registration": [],
                       "extraction_source": None, "registration_source": None},
            "rounds": {r: copy.deepcopy(entries) for r in ROUNDS}, "transforms": {},
            "supplied_statistics": {"path": None, "sha256": None}}


def counts_record(fov, *, deduplication=True, source=None):
    source = source or fov
    spots = COUNTS["spots"][source]
    assigned = COUNTS["assigned"][source]
    merged = COUNTS["merged_reads"][source] if deduplication else 0
    accepted = COUNTS["accepted"][source]
    counts = {"spots": spots, "intensities": spots,
              "call_status": {"assigned": assigned, "unmatched": spots - assigned},
              "scoring": {"total": spots, "scored": spots, "no_assignment": 0, "background_unavailable": 0}}
    if deduplication:
        counts["deduplication"] = {"total": spots, "groups": merged, "merged_reads": merged, "conflicting_groups": 0}
    counts["filtering"] = {"total": spots - merged, "accepted": accepted, "rejected": spots - merged - accepted,
                           "endpoint_valid": COUNTS["endpoint_valid"][source]}
    return counts


def run_record(root, fov, *, source=None, deduplication=True, registration=True):
    """The run.json of a FOV, in the form _RunRecord writes it (R1 to R4 included)."""
    root = Path(root)
    source = source or fov
    codebook = root / "codebook.csv"
    reg = {name: [estimation_entry(NCC_AFTER[name][source] - 0.01), application_entry(NCC_AFTER[name][source])]
           for name in MOVING} if registration else {}
    transforms = {name: {"reference": "round1", "results": [transform_entry(TRANSLATIONS[name][source])],
                         "application_config": dict(WARP)} for name in MOVING} if registration else {}
    return {
        "format_version": 1, "dataset_id": DATASET, "sample_id": SAMPLE_ID, "fov_id": fov, "subtile_id": None,
        "status": "succeeded", "started_at": STARTED[fov], "ended_at": ENDED[fov], "error": None,
        "code": {"version": "0.1.0", "git_commit": "0" * 40, "git_dirty": False},
        "environment": {"python": "3.12.12", "platform": "test", "packages": {}},
        "config": {"pipeline": pipeline_config(deduplication=deduplication, registration=registration),
                   "execution": {"mode": "batch", "retain_images": False, "device": "cpu"},
                   "checkpoints": {"stages": ["candidates", "pre_qc"], "directory": None, "table_format": "csv",
                                   "hash_inputs": True, "overwrite": False},
                   "readout_mode": "multiplexed", "encoding": {"method": "two_base"}},
        "workflow_config": None,
        "channels": {r: [{"channel": c, "name": None, "wavelength": "unavailable"} for c in CHANNELS]
                     for r in ROUNDS},
        "inputs": [{"path": str(raw_path(root, r, fov, c)), "sha256": sha256(raw_bytes(root, r, fov, c))}
                   for r in ROUNDS for c in CHANNELS],
        "codebook": {"kind": "codebook", "path": str(codebook), "sha256": sha256(CODEBOOK_TEXT.encode()),
                     "entries": len(GENES), "genes": len(GENES)},
        "steps": _steps(deduplication=deduplication, registration=registration),
        "preprocessing": _preprocessing(),
        "registration": reg, "transforms": transforms,
        "counts": counts_record(fov, deduplication=deduplication, source=source),
        "checkpoint_directory": str(root / "checkpoints" / fov), "checkpoints": {},
    }


def placement_result(fov_ids, *, statuses=None, flags=None, supplied=False):
    """The PlacementResult of the 2 × 2 grid, constructed as the assembly fixtures construct theirs."""
    from starfinder.segmentation import ReferenceGrid
    from starfinder.stitching import PlacementResult
    from starfinder.stitching._placement import FOV_COLUMNS, empty_links, frame_record, input_record
    statuses, flags = statuses or {}, flags or {}
    rows = []
    for fov in fov_ids:
        status = statuses.get(fov, "metadata_only" if supplied else "image_supported")
        origin = ORIGINS[fov] if status not in ("missing", "failed") else (np.nan, np.nan, np.nan)
        rows.append({"fov_id": fov, "status": status, **{f"origin_{a}": v for a, v in zip("zyx", origin)},
                     **{f"nominal_{a}": v for a, v in zip("zyx", NOMINAL[fov])},
                     **{f"correction_{a}": o - n for a, o, n in zip("zyx", origin, NOMINAL[fov])},
                     **{f"method_translation_{a}": np.nan for a in "zyx"},
                     "component": -1 if status == "missing" else 0, "n_links_kept": 0,
                     "flags": "supplied" if supplied else flags.get(fov, "")})
    grid = ReferenceGrid(SHAPE_ZYX, metadata(), "declared")
    record = {"format_version": 1, "stage": "stitching", "sample": SAMPLE, "frame": frame_record(SAMPLE),
              "gauge": "G1", "method": {"stage": "stitching", "method": "supplied" if supplied else "constructed",
                                        "config_type": "", "implementation": "test", "config": {}, "requires": {},
                                        "artifacts": []},
              "table": {}, "inputs": {fov: input_record("constructed", None, grid) for fov in fov_ids},
              "fallback": None,
              "review": {"state": "not_needed", "reasons": [], "policy": None, "flagged": {}, "approvals": []}}
    return PlacementResult(SAMPLE, pd.DataFrame(rows, columns=list(FOV_COLUMNS)), empty_links(), record)


def write_transforms_file(root, fov, *, source=None):
    """``checkpoints/<fov>/registered/transforms.json`` (format version 2) through write_registered_header."""
    from starfinder._checkpoint import write_registered_header
    source = source or fov
    results = {name: [registration_result(TRANSLATIONS[name][source])] for name in MOVING}
    record = {"semantics": "recipe", "recipe": None,
              "application": {name: results[name][0].application_config for name in MOVING}}
    write_registered_header(Path(root) / "checkpoints" / fov, {"fov_id": fov}, results, record)


def report_records(root, variant=None, *, inputs=False) -> ReportCase:
    """Write ``report_records`` (or a variant of it) under ``root``; ``inputs`` also writes the raw inputs."""
    from starfinder.stitching import save_placement
    if variant is not None and variant not in RECORD_VARIANTS:
        raise ValueError(f"unknown report_records variant {variant!r}")
    root = Path(root)
    fov_ids = ["F1"] if variant == "one" else list(FOVS)
    case = ReportCase(root, fov_ids)
    (root / "codebook.csv").parent.mkdir(parents=True, exist_ok=True)
    (root / "codebook.csv").write_text(CODEBOOK_TEXT)
    for fov in fov_ids:
        record = run_record(root, fov, source="F1" if variant == "equal" else None,
                            deduplication=variant != "stage_off", registration=variant != "stage_off_registration")
        case.checkpoint(fov).mkdir(parents=True, exist_ok=True)
        case.write_run_record(fov, record)
    if inputs:
        write_inputs(root, fov_ids)
    statuses, flags, supplied = {}, {}, False
    if variant == "placement_states":
        statuses, flags = {"F3": "metadata_only", "F4": "failed"}, {"F3": "fallback"}
    if variant == "placement_supplied":
        supplied = True
    if variant != "no_placement":
        save_placement(placement_result(fov_ids, statuses=statuses, flags=flags, supplied=supplied),
                       root / "stitching" / SAMPLE)
    if variant is not None and variant.startswith("failed"):
        _fail_f3(case, variant)
    if variant == "not_run":
        (case.checkpoint("F4") / "run.json").unlink()
    if variant == "only_f1":
        for fov in ("F2", "F3", "F4"):
            (case.checkpoint(fov) / "run.json").unlink()
    if variant == "no_spot":
        record = case.run_record("F2")
        record["counts"] = {"spots": 0, "intensities": 0, "call_status": {"assigned": 0},
                            "scoring": {"total": 0, "scored": 0, "no_assignment": 0, "background_unavailable": 0},
                            "deduplication": {"total": 0, "groups": 0, "merged_reads": 0, "conflicting_groups": 0},
                            "filtering": {"total": 0, "accepted": 0, "rejected": 0, "endpoint_valid": 0}}
        case.write_run_record("F2", record)
    if variant in ("older", "older_transforms"):
        for fov in fov_ids:
            record = case.run_record(fov)
            for key in ("transforms", "codebook", "workflow_config"):
                del record[key]
            del record["counts"]["filtering"]["endpoint_valid"]
            case.write_run_record(fov, record)
        if variant == "older_transforms":
            write_transforms_file(root, "F1")
    return case


def _fail_f3(case, variant):
    """F3 failed at register, round3 (``failed``), or left inside round3 as FOV._register_round leaves it (RC13)."""
    record = case.run_record("F3")
    status = "interrupted" if variant == "failed_estimating" else "failed"
    record.update(status=status, error={"step": "register", "round": "round3", "type": "RuntimeError",
                                        "message": "registration failed", "traceback": ""})
    record["steps"] = _steps(fail_at=("register", "round3"))
    record["counts"] = {}
    del record["transforms"]["round3"]
    if variant == "failed":
        del record["registration"]["round3"]
    elif variant == "failed_applying":
        record["registration"]["round3"] = [estimation_entry(0.8), application_entry(None, "applying", qc=False)]
    elif variant == "failed_estimating":
        record["registration"]["round3"] = [estimation_entry(None, "estimating", qc=False)]
    else:
        record["registration"]["round3"] = [estimation_entry(0.8)]
    case.write_run_record("F3", record)


# --- The export ----------------------------------------------------------------------------------

def _namespace(*parts):
    return json.dumps(list(parts), separators=(",", ":"))


def _footprint(fov):
    from starfinder.assembly._ownership import footprint_yx
    (y0, y1, x0, x1), (cy, cx) = footprint_yx(ORIGINS[fov][1:], SHAPE_ZYX[1:], SPACING_ZYX[1:])
    return {"footprint_y0": y0, "footprint_y1": y1, "footprint_x0": x0, "footprint_x1": x1, "centre_y": cy,
            "centre_x": cx}


def _cell_rows(nocal=False):
    """Every cell: F1 20 kept and one retained excluded_no_nucleus, F2 one not_owner, F3 7 and F4 14 kept."""
    plan = {"F1": ["kept"] * 20 + ["excluded_no_nucleus"], "F2": ["not_owner"], "F3": ["kept"] * 7,
            "F4": ["kept"] * 14}
    rows = []
    for fov, states in plan.items():
        namespace = _namespace(DATASET, SAMPLE_ID, fov, None, "cell")
        for i, state in enumerate(states, start=1):
            y, x = 1 + (i - 1) // 5 * 3, 1 + (i - 1) % 5 * 3
            oz, oy, ox = ORIGINS[fov]
            centre = (oz + 1.0 * SPACING_ZYX[0], oy + y * SPACING_ZYX[1], ox + x * SPACING_ZYX[2])
            retained = state != "not_owner"
            rows.append({
                "fov_id": fov, "cell_namespace": namespace, "cell_id": i, "obs_name": f"{fov}:cell:{i}",
                "status": "kept" if state != "excluded_no_nucleus" else "excluded_no_nucleus",
                "exclusion_reason": pd.NA if state != "excluded_no_nucleus" else "no_nucleus",
                "size_voxels": 8, "expanded_size_voxels": pd.NA,
                "size_physical": np.nan if nocal else 8 * float(np.prod(SPACING_ZYX)),
                "expanded_size_physical": np.nan, "centroid_z": 1.0, "centroid_y": float(y), "centroid_x": float(x),
                "expanded_centroid_z": np.nan, "expanded_centroid_y": np.nan, "expanded_centroid_x": np.nan,
                "n_molecules": 0, "n_nuclei": pd.NA, "correspondence": "no_nuclei", "correspondence_flags": "",
                "compartments": "unavailable",
                **{f"sample_centroid_{a}": v for a, v in zip("zyx", centre)},
                **{f"sample_expanded_centroid_{a}": np.nan for a in "zyx"}, "ownership_centroid": "territory",
                **{f"sample_ownership_centroid_{a}": v for a, v in zip("zyx", centre)},
                "owner_fov_id": fov if retained else "F1", "assembly_status": "retained" if retained else "excluded",
                "assembly_reason": "" if retained else "not_owner", "assembly_flags": "", "conflict_with": "",
                "duplicate_of": ""})
    return rows


def _molecule_rows(cells):
    """Each cell's molecules (F3's cells two each), F2's three retained unassigned reads and F1's five z values."""
    rows = []

    def add(fov, z, y, x, gene, status, cell=None, assembly="retained", reason=""):
        oz, oy, ox = ORIGINS[fov]
        plane = int(np.floor(z + 0.5))
        inside = 0 <= plane < SHAPE_ZYX[0]
        rows.append({"fov_id": fov, "spot_namespace": _namespace(DATASET, SAMPLE_ID, fov, None),
                     "spot_id": f"{fov}:m{sum(r['fov_id'] == fov for r in rows)}", "z": float(z), "y": float(y),
                     "x": float(x), "gene_id": gene, "voxel_z": plane if inside else pd.NA, "voxel_y": int(y),
                     "voxel_x": int(x), "assignment_status": status,
                     "cell_id": pd.NA if cell is None else cell["cell_id"], "in_expansion": pd.NA,
                     "original_cell_id": pd.NA, "nucleus_id": pd.NA,
                     "compartment": "whole" if cell is not None else pd.NA,
                     "cell_namespace": pd.NA if cell is None else cell["cell_namespace"],
                     "sample_z": oz + z * SPACING_ZYX[0], "sample_y": oy + y * SPACING_ZYX[1],
                     "sample_x": ox + x * SPACING_ZYX[2], "owner_fov_id": fov, "assembly_status": assembly,
                     "assembly_reason": reason, "assembly_flags": ""})

    for k, cell in enumerate(cells):
        per_cell = 2 if cell["fov_id"] == "F3" or cell["status"] != "kept" or cell["fov_id"] == "F2" else 1
        for j in range(per_cell):
            gene = GENES[(k + j) % len(GENES)]
            if cell["assembly_status"] == "retained":
                add(cell["fov_id"], 1.0, cell["centroid_y"], cell["centroid_x"], gene, "assigned", cell)
            else:
                add(cell["fov_id"], 1.0, cell["centroid_y"], cell["centroid_x"], gene, "assigned", cell,
                    "excluded", "cell_not_owner")
    for y in (2.0, 5.0, 8.0):
        add("F2", 2.0, y, 14.0, "G1", "background")
    for z in (-0.6, -0.5, 0.4999, 0.5, 3.5):
        plane = int(np.floor(z + 0.5))
        outside = not 0 <= plane < SHAPE_ZYX[0]
        add("F1", z, 15.0, 15.0, "G2", "outside_grid" if outside else "background",
            assembly="excluded" if outside else "retained", reason="outside_grid" if outside else "")
    return rows


def write_export(case: ReportCase, variant=None) -> None:
    """The export folder of report_cells, route per_fov, written through the package's writers."""
    from starfinder.assembly._assemble import CELL_DTYPES, COUNT_DTYPES, FOV_DTYPES, molecule_dtypes
    from starfinder.assembly import AssemblyConfig, CELL_ASSEMBLY_REASONS, MOLECULE_ASSEMBLY_REASONS
    from starfinder.assignment import write_h5ad
    from starfinder._records import _write_table
    from starfinder.export._record import update_section
    from starfinder._records import file_sha256
    from starfinder.stitching._placement import frame_record
    folder = case.root / "export" / case.sample
    (folder / "tables").mkdir(parents=True, exist_ok=True)
    cells = _cell_rows(nocal=variant == "nocal")
    molecules = _molecule_rows(cells)
    counts = []
    for cell in cells:
        own = [m["gene_id"] for m in molecules
               if m["fov_id"] == cell["fov_id"] and m["cell_id"] is not pd.NA and m["cell_id"] == cell["cell_id"]]
        cell["n_molecules"] = len(own)
        genes = pd.Series(own, dtype=object).value_counts()
        for gene in GENES:
            if gene in genes:
                counts.append({"fov_id": cell["fov_id"], "cell_namespace": cell["cell_namespace"],
                               "cell_id": cell["cell_id"], "gene_id": gene, "compartment": "whole",
                               "count": int(genes[gene])})
    fovs = []
    for fov in FOVS:
        own_cells = [c for c in cells if c["fov_id"] == fov]
        own_molecules = [m for m in molecules if m["fov_id"] == fov]
        fovs.append({"fov_id": fov, "placement_status": "image_supported",
                     **{f"origin_{a}": v for a, v in zip("zyx", ORIGINS[fov])}, "kind": "assignment",
                     **_footprint(fov), "owns_points": True, "n_cells": len(own_cells),
                     "n_cells_retained": sum(c["assembly_status"] == "retained" for c in own_cells),
                     "n_molecules": len(own_molecules),
                     "n_molecules_retained": sum(m["assembly_status"] == "retained" for m in own_molecules)})
    assert {f["fov_id"]: f["n_cells_retained"] for f in fovs} == CELLS_RETAINED
    frames = {"fovs": pd.DataFrame(fovs, columns=list(FOV_DTYPES)).astype(FOV_DTYPES),
              "cells": pd.DataFrame(cells, columns=list(CELL_DTYPES)).astype(CELL_DTYPES),
              "molecules": pd.DataFrame(molecules, columns=list(molecule_dtypes("cells"))).astype(
                  molecule_dtypes("cells")),
              "counts": pd.DataFrame(counts, columns=list(COUNT_DTYPES)).astype(COUNT_DTYPES)}
    files = {}
    for name, frame in frames.items():
        file_name, dtypes = _write_table(frame, folder / "tables", name, "csv")
        files[name] = {"path": f"tables/{file_name}", "sha256": file_sha256(folder / "tables" / file_name),
                       "dtypes": dtypes}
    counted = frames["cells"][frames["cells"]["assembly_status"].eq("retained").to_numpy(bool)
                              & frames["cells"]["status"].eq("kept").to_numpy(bool)].reset_index(drop=True)
    long = frames["counts"]
    if variant == "mismatch":
        extra = long.iloc[[0]].copy()
        extra["count"] = 1
        long = pd.concat([long, extra], ignore_index=True)
    obs = pd.DataFrame({"fov_id": counted["fov_id"].astype(str).to_numpy(),
                        "size_physical": counted["size_physical"].to_numpy(),
                        "size_voxels": counted["size_voxels"].to_numpy()},
                       index=pd.Index(counted["obs_name"].astype(str).to_numpy()))
    keys = counted[["cell_namespace", "cell_id"]]
    write_h5ad(folder / "sample.h5ad", obs, keys, long, GENES, layers=False, uns={"sample": case.sample})
    files["anndata"] = {"path": "sample.h5ad", "sha256": file_sha256(folder / "sample.h5ad")}
    cell_frame, molecule_frame = frames["cells"], frames["molecules"]

    def population(frame, reasons, column):
        status = frame["assembly_status"].astype(object)
        return {"input": int(sum(f[column] for f in fovs)), "retained": int(status.eq("retained").sum()),
                "excluded": {r: int((status.eq("excluded") & frame["assembly_reason"].astype(object).eq(r)).sum())
                             for r in reasons}}
    grid = {"shape_zyx": list(SHAPE_ZYX), "metadata": jsonable(asdict(metadata())), "sha256": None,
            "source": "declared", "calibration_source": "grid"}
    section = jsonable({
        "format_version": 1, "stage": "assembly", "sample": case.sample, "mode": "cells", "route": "per_fov",
        "route_source": "default", "frame": frame_record(case.sample), "genes": list(GENES),
        "placement": {"sample": case.sample, "method": "constructed", "review": "not_needed"},
        "config": asdict(AssemblyConfig()),
        "ownership": {"rule": "nearest_center", "tie": "table_order", "axes": "yx", "owning_fovs": list(FOVS)},
        "flags": {"computed": [], "conflicts": True},
        "fovs": [{"fov_id": f["fov_id"], "placement_status": f["placement_status"],
                  "origin_zyx": list(ORIGINS[f["fov_id"]]), "kind": "assignment",
                  "assignment": assignment_entry(case, f["fov_id"]),
                  "grid": grid, "owns_points": True} for f in fovs],
        "counts": {"cells": population(cell_frame, CELL_ASSEMBLY_REASONS, "n_cells"),
                   "molecules": population(molecule_frame, MOLECULE_ASSEMBLY_REASONS, "n_molecules"),
                   "flags": {}},
        "files": files, "software": {"starfinder": "0.1.0"}})
    update_section(folder, case.sample, "assemble", section, status="written")


def cells_config() -> ReportConfig:
    """The configuration of report_cells: one marker group and two marker pairs of its genes."""
    return ReportConfig(markers=(MarkerGroup("compartments", ("G1", "G2", "G3")),),
                        marker_pairs=(("G1", "G2"), ("G4", "G5")))


def write_assignment_folder(case: ReportCase, fov) -> Path:
    """The saved assignment ``default`` of a FOV and its cell run ``cell``, in the layout of assignment/_persist.py.

    Hand-written: ``segmentation/cell/segmentation.json`` (a computed run, method ``watershed``) and
    ``assignment/default/`` with ``cells.csv`` and ``assignment.json``, whose ``inputs.cells`` holds the run's
    record and its record SHA-256 (``assignment/_assign.py``: ``_label_input``) and whose ``files`` the SHA-256
    of ``cells.csv``.
    """
    from starfinder._records import json_sha256 as _sha256_json
    run = case.checkpoint(fov) / "segmentation" / "cell"
    run.mkdir(parents=True, exist_ok=True)
    segmentation = {"format_version": 1, "stage": "segmentation", "run": "cell", "target": "cell",
                    "geometry": "volume", "methods": [{"stage": "segmentation", "method": "watershed"}],
                    "import": None, "operations": [], "outcome": "ok"}
    (run / "segmentation.json").write_text(json.dumps(segmentation, indent=2) + "\n")
    folder = case.checkpoint(fov) / "assignment" / "default"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "cells.csv").write_text("cell_id,status\n1,kept\n")
    record = {"format_version": 1, "stage": "assignment", "name": "default",
              "inputs": {"cells": {"run": "cell", "target": "cell", "geometry": "volume",
                                   "record_sha256": _sha256_json(segmentation), "saved_under_run": True,
                                   "file": {"path": "../../segmentation/cell/labels.tif", "sha256": None},
                                   "record": segmentation},
                         "nuclei": None},
              "files": {"cells": {"path": "cells.csv", "sha256": sha256((folder / "cells.csv").read_bytes()),
                                  "dtypes": {}}}}
    (folder / "assignment.json").write_text(json.dumps(record, indent=2) + "\n")
    return folder


def assignment_entry(case: ReportCase, fov) -> dict:
    """``assembly.fovs[*].assignment`` of a folder source, as ``assembly/_assemble.py``: ``_fov_record`` writes it."""
    from starfinder._records import file_sha256
    folder = case.checkpoint(fov) / "assignment" / "default"
    record = json.loads((folder / "assignment.json").read_text())
    return {"name": "default", "folder": str(folder), "source": "folder",
            "record_sha256": file_sha256(folder / "assignment.json"),
            "tables": {t: record["files"][t]["sha256"] for t in ("molecules", "cells", "counts")
                       if t in record["files"]}}


def report_cells(root, variant=None) -> ReportCase:
    """``report_records`` with the export folder of ``report_cells`` (or a variant); needs the anndata extra."""
    if variant is not None and variant not in CELL_VARIANTS:
        raise ValueError(f"unknown report_cells variant {variant!r}")
    case = report_records(root)
    if variant != "no_markers":
        case.config = cells_config()
    for fov in case.fov_ids:
        write_assignment_folder(case, fov)
    write_export(case, variant)
    return case


# --- The pages drawn with their canvases (test_report_figure_rules.py and test_report_second_review.py) -----------

#: Every case of the page: each fixture alone, then with each of its variants.
CASES = [("report_records", None)] + [("report_records", v) for v in RECORD_VARIANTS] + [
    ("report_cells", None)] + [("report_cells", v) for v in CELL_VARIANTS]
IDS = [f"{m}-{v}" for m, v in CASES]


class Drawn:
    """A collected case: its folder, record, page text and figures (with canvases)."""

    def __init__(self, case, folder, record, figures):
        self.case, self.folder, self.record, self.figures = case, Path(folder), record, figures
        self.text = (self.folder / "report.html").read_text(encoding="utf-8")


def collect_drawn(maker, variant, root, **patches):
    """A case collected with the figures kept as drawn; ``patches`` set attributes of ``_figures`` meanwhile."""
    if maker == "report_cells":
        from .test_report_collection import anndata_or_skip
        anndata_or_skip()
    found = {}
    original = _page.build_page

    def build(*args, **kwargs):
        text, figures = original(*args, **kwargs)
        found["figures"] = figures
        return text, figures
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_figures, "KEEP_CANVAS", True)
        for name, value in patches.items():
            patch.setattr(_figures, name, value)
        patch.setattr(_page, "build_page", build)
        case = (report_cells if maker == "report_cells" else report_records)(root, variant)
        folder, record = case.collect()
    return Drawn(case, folder, record, found["figures"])


_DRAWN = pytest.StashKey[dict]()


@pytest.fixture(scope="session")
def drawn(request, tmp_path_factory):
    """``drawn(maker, variant)``: the case collected once for the session by ``collect_drawn``. The modules that
    import this fixture share the cases (kept in the session's stash): their tests only read the folder and the
    figures; a test that changes either collects its own."""
    made = request.config.stash.setdefault(_DRAWN, {})

    def get(maker, variant):
        if (maker, variant) not in made:
            root = tmp_path_factory.mktemp(f"{maker}-{variant}") / "root"
            made[(maker, variant)] = collect_drawn(maker, variant, root)
        return made[(maker, variant)]
    yield get
    made.clear()


@pytest.fixture
def keep(monkeypatch):
    monkeypatch.setattr(_figures, "KEEP_CANVAS", True)

