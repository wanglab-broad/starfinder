"""Golden baseline of the records a FOV run writes today, before §2.11 adds to them (W-385).

docs/report-contract.md ("Records added to run.json") adds fields to ``run.json`` so that
the processing report can read the global transforms, the codebook file and the
correct-form count. This file pins what ``FOV.run`` writes at the base revision
(dc5fc3c) on one small synthetic FOV generated in the test (two rounds of four channels,
4 × 24 × 24 voxels, four molecules on a seeded Poisson background, round2 shifted by one
voxel in X; nothing outside the test is read and no image is committed), with the
checkpoint stages of the default ``CheckpointConfig`` and with ``candidates`` and
``pre_qc`` only:

(a) the content of ``run.json``, in the form "every key and value pinned here is still
    present": a dict may gain keys, a list keeps its length and order, and a pinned
    value must stay equal (floats to a relative 1e-9). Values that depend on the time,
    the host, the temporary folder or the installed versions are pinned by what they are
    (a timestamp, a duration, a SHA-256 of the named file, a key set) instead;
(b) the files the run writes in the FOV's checkpoint folder.

So the record additions of §2.11 pass, and a removed key or a changed value fails. The
pipeline is the recipe of W-384's synthetic runs on a smaller scene: min-max
normalization and a white top-hat, one translation step, local maxima, extraction with
the local background, WTA decoding, scoring, deduplication and a read filter with
``end_bases`` (whose correct-form count stays in the filter diagnostics today).
"""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from starfinder.barcode import (Codebook, DeduplicationConfig, NeighborhoodSumConfig, ReadFilterConfig,
                                ReadScoreConfig, WtaDecoderConfig)
from starfinder.dataset import (CheckpointConfig, Dataset, ExecutionConfig, PipelineConfig, RegistrationRecipe,
                                RegistrationStep, RoundState)
from starfinder.image import ImageMetadata
from starfinder.io import ImageLoadConfig, save_volume
from starfinder.preprocessing import (MinMaxNormalizationConfig, PreprocessingRecipe, PreprocessingStep,
                                      TophatConfig)
from starfinder.registration import TranslationConfig
from starfinder.spot_finding import LocalMaximaConfig

pytestmark = [pytest.mark.dataset, pytest.mark.golden]

ROUNDS = ("round1", "round2")
CHANNELS = ("ch00", "ch01", "ch02", "ch03")
SHAPE_ZYX = (4, 24, 24)
# (z, y, x) of each molecule in round1 and its channel in round1 and round2.
MOLECULES = [((1, 5, 6), (0, 1)), ((2, 12, 15), (1, 0)), ((1, 18, 8), (2, 3)), ((2, 8, 18), (3, 2))]
GENES = {"g1": "12", "g2": "21", "g3": "34", "g4": "43"}
ALL_STAGES = ("registered", "candidates", "pre_qc")
NO_REGISTERED = ("candidates", "pre_qc")
ROOT = "<root>"


# ---------------------------------------------------------------------------------------
# Pinned forms for values that depend on the time, the host or the installed versions.

class Pinned:
    """A value pinned by what it is; ``check(value, actual_record)`` is True when it matches."""

    def __init__(self, name, check):
        self.name, self.check = name, check

    def __repr__(self):
        return self.name


ANY = Pinned("any value", lambda value, record: True)
TIME = Pinned("a UTC ISO time", lambda value, record: isinstance(value, str)
              and datetime.fromisoformat(value).utcoffset().total_seconds() == 0)
SECONDS = Pinned("a duration in seconds", lambda value, record: isinstance(value, float) and value >= 0)


def keys(*names):
    """A dict with exactly these keys, values not pinned (host or version dependent)."""
    return Pinned(f"a dict with keys {sorted(names)}", lambda value, record: isinstance(value, dict)
                  and set(names) <= set(value))


def file_sha256(path):
    """The SHA-256 of the file at ``path`` (``<root>`` is the test's folder)."""
    def check(value, record):
        actual = Path(record["__root__"]) / path
        return value == hashlib.sha256(actual.read_bytes()).hexdigest()
    return Pinned(f"the SHA-256 of {path}", check)


def differences(pinned, actual, record=None, where="run.json"):
    """Where ``actual`` lacks a pinned key or differs from a pinned value; [] when it holds all of them."""
    record = actual if record is None else record
    if isinstance(pinned, Pinned):
        return [] if pinned.check(actual, record) else [f"{where}: {actual!r} is not {pinned!r}"]
    if isinstance(pinned, dict):
        if not isinstance(actual, dict):
            return [f"{where}: {actual!r} is not a dict"]
        found = []
        for key, value in pinned.items():
            if key not in actual:
                found.append(f"{where}: key {key!r} is missing")
            else:
                found += differences(value, actual[key], record, f"{where}.{key}")
        return found
    if isinstance(pinned, list):
        if not isinstance(actual, list) or len(actual) != len(pinned):
            return [f"{where}: {actual!r} is not a list of {len(pinned)} items"]
        return [d for i, (p, a) in enumerate(zip(pinned, actual)) for d in differences(p, a, record, f"{where}[{i}]")]
    if isinstance(pinned, float) and isinstance(actual, float) and not isinstance(actual, bool):
        return [] if math.isclose(pinned, actual, rel_tol=1e-9, abs_tol=1e-12) else [
            f"{where}: {actual!r} != {pinned!r}"]
    if type(pinned) is not type(actual) or pinned != actual:
        return [f"{where}: {actual!r} != {pinned!r}"]
    return []


# ---------------------------------------------------------------------------------------
# The synthetic FOV and its run.

def write_scene(root: Path):
    """Per-channel ZYX TIFFs of FOV_001 under ``root/in/<round>/FOV_001/``."""
    rng = np.random.default_rng(385)
    images = {name: rng.poisson(100, SHAPE_ZYX + (len(CHANNELS),)).astype(np.uint16) for name in ROUNDS}
    for (z, y, x), channels in MOLECULES:
        for name, channel in zip(ROUNDS, channels):
            images[name][z, y, x, channel] = 800
    images["round2"] = np.roll(images["round2"], 1, axis=2)
    for name, image in images.items():
        for index, channel in enumerate(CHANNELS):
            save_volume(image[..., index], root / "in" / name / "FOV_001" / f"{channel}.tif",
                        metadata=ImageMetadata("common", spacing_zyx=(2.0, 0.5, 0.5)))


def pipeline():
    return PipelineConfig(
        load=ImageLoadConfig(channel_labels=CHANNELS),
        preprocessing=PreprocessingRecipe((PreprocessingStep(MinMaxNormalizationConfig("uint8", (0, 255))),
                                           PreprocessingStep(TophatConfig(radius_yx=2)))),
        registration=RegistrationRecipe((RegistrationStep(TranslationConfig()),)),
        spot_finding=LocalMaximaConfig("adaptive", 0.2), extraction=NeighborhoodSumConfig((0, 1, 1)),
        decoding=WtaDecoderConfig(), scoring=ReadScoreConfig(), deduplication=DeduplicationConfig(),
        filtering=ReadFilterConfig(end_bases="CC"))


def run_fov(root: Path, stages):
    """Run FOV_001 with checkpoints; return run.json (the folder replaced by <root>) and the files written."""
    write_scene(root)
    dataset = Dataset(root / "in", root / "out", "golden", "sample", "run",
                      RoundState(list(ROUNDS), reference_round="round1"), CHANNELS)
    dataset.codebook = Codebook(pd.DataFrame({"gene_id": list(GENES), "color_sequence": list(GENES.values())}),
                                ROUNDS, CHANNELS)
    fov = dataset.fov("FOV_001").run(pipeline(), execution=ExecutionConfig(),
                                     checkpoints=CheckpointConfig(stages=stages))
    folder = fov.paths.checkpoint_dir
    record = json.loads((folder / "run.json").read_text().replace(str(root), ROOT))
    record["__root__"] = str(root)
    files = sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file())
    return record, files


@pytest.fixture(scope="module", params=[ALL_STAGES, NO_REGISTERED], ids=["all_stages", "no_registered"])
def written(request, tmp_path_factory):
    stages = request.param
    return stages, *run_fov(tmp_path_factory.mktemp("report_records"), stages)


# ---------------------------------------------------------------------------------------
# The pinned record.

def _channels():
    return [{"channel": c, "name": None, "wavelength": "unavailable"} for c in CHANNELS]


MIN_MAX = {"output_dtype": "uint8", "output_range": [0, 255], "scope": "per_channel", "snr_threshold": None,
           "rounding": "truncate"}
QC_POLICY = {"min_coverage": None, "min_ncc_gain": None, "max_fold_fraction": None, "max_translation_voxels": None,
             "projections": False}
QC_CONFIG = {"overlap": "pull point inside the closed box [0, n-1] on every axis", "ssim_policy": "mip",
             "ssim_window": 7, "ssim_erosion": 3, "data_range": 244.0, "qc": QC_POLICY}
QC_UNITS = {"coverage": "fraction", "ncc_before": "dimensionless", "ncc_after": "dimensionless",
            "ncc_gain": "dimensionless", "ssim_before": "dimensionless", "ssim_after": "dimensionless"}
THREADS = keys("torch_num_threads", "torch_num_interop_threads", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
               "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS")


def _preprocessing_round(fitted_max):
    return [
        {"index": 0, "stage": "steps", "step": "min_max_normalization", "config": MIN_MAX,
         "fitted": {"groups": [{"channel": c, "min": lo, "max": hi, "mode": "rescale"}
                               for c, (lo, hi) in enumerate(fitted_max)]},
         "diagnostics": {}, "input_dtype": "uint16", "output_dtype": "uint8", "save_as": None},
        {"index": 1, "stage": "steps", "step": "white_tophat", "config": {"radius_yx": 2}, "fitted": {},
         "diagnostics": {}, "input_dtype": "uint8", "output_dtype": "uint8", "save_as": None},
    ]


def _step(name, round_name=None, **extra):
    return {"name": name, "round": round_name, "seconds": SECONDS, "status": "succeeded", **extra}


def pinned_record(stages):
    registered = "registered" in stages
    detection = {
        "stage": "spot_finding", "method": "local_maxima", "config_type": "starfinder.spot_finding.LocalMaximaConfig",
        "implementation": "starfinder.spot_finding._methods._local_maxima",
        "config": {"threshold_mode": "adaptive", "threshold_value": 0.2, "min_distance_voxels": 1,
                   "exclude_border": True, "channel_labels": list(CHANNELS), "measure_peak_intensity": True,
                   "merge_radius_zyx": None, "method": "local_maxima"},
        "requires": {}, "artifacts": [], "execution": {"device": "cpu", "framework": None, "threads": THREADS}}
    steps = [
        _step("load_images"),
        _step("preprocess:min_max_normalization", "round1"), _step("preprocess:white_tophat", "round1"),
        *([_step("write_checkpoint:registered", "round1")] if registered else []),
        _step("find_spots", methods=[detection]),
        _step("extract_round", "round1"),
        _step("preprocess:min_max_normalization", "round2"), _step("preprocess:white_tophat", "round2"),
        _step("register", "round2"),
        *([_step("write_checkpoint:registered", "round2")] if registered else []),
        _step("extract_round", "round2"),
        *([_step("write_checkpoint:registered")] if registered else []),
        _step("write_checkpoint:candidates"), _step("decode_barcodes"), _step("score_reads"),
        _step("deduplicate_reads"), _step("write_checkpoint:pre_qc"), _step("filter_reads"),
    ]
    estimation_qc = {
        "status": "ok", "values": PINNED_QC_VALUES, "units": QC_UNITS, "counts": PINNED_QC_COUNTS, "reasons": {},
        "config": QC_CONFIG,
        "details": {"transform": {"kind": "translation", "displacement_zyx": [0.0, 0.0, 1.0]},
                    "optimizer": {"method": "translation", "backend": "scipy_fft", "converged": None,
                                  "iterations_completed": None, "final_metric_value": None, "stop_condition": None,
                                  "elapsed_iterations": None, "final_rms_change": None}}}
    application_qc = {
        "status": "ok", "values": PINNED_QC_VALUES, "units": QC_UNITS, "counts": PINNED_QC_COUNTS, "reasons": {},
        "config": QC_CONFIG,
        "details": {"transform": {"kind": "dense",
                                  "displacement_voxels": {"median": 1.0, "p95": 1.0, "max": 1.0},
                                  "displacement_physical": {"median": 0.5, "p95": 0.5, "max": 0.5},
                                  "fold_fraction": 0.0},
                    "optimizer": None}}
    checkpoints = {"candidates": ["candidates.csv", "candidates.json"], "pre_qc": ["pre_qc.csv", "pre_qc.json"]}
    if registered:
        checkpoints = {"registered": ["registered/round1.ome.tif", "registered/round2.ome.tif",
                                      "registered/transforms.json"], **checkpoints}
    return {
        "format_version": 1, "dataset_id": "golden", "sample_id": "sample", "fov_id": "FOV_001", "subtile_id": None,
        "status": "succeeded", "started_at": TIME, "ended_at": TIME, "error": None,
        "code": keys("version", "git_commit", "git_dirty"),
        "environment": {"python": ANY, "platform": ANY,
                        "packages": keys("starfinder", "numpy", "pandas", "scipy", "scikit-image", "tifffile",
                                         "pyarrow", "SimpleITK")},
        "config": {
            "pipeline": {
                "load": {"channel_labels": list(CHANNELS), "source_paths": None, "subdir": "", "source_axes": None,
                         "series_index": None, "time_index": None, "channel_index": None, "crop_policy": "error",
                         "conversion": None, "metadata": None},
                "rotation_degrees": None,
                "preprocessing": {"steps": [{"config": MIN_MAX, "save_as": None},
                                            {"config": {"radius_yx": 2}, "save_as": None}],
                                  "post_registration": [], "extraction_source": None, "registration_source": None,
                                  "supplied_statistics": None},
                "registration": {"steps": [{"config": {"backend": "scipy_fft", "fft_workers": 1,
                                                       "method": "translation"},
                                            "recovery": None, "signal": None}],
                                 "signal": {"mode": "max", "reference_channel": None, "moving_channel": None},
                                 "warp": None, "reference_round": None, "qc": QC_POLICY},
                "spot_finding": {"threshold_mode": "adaptive", "threshold_value": 0.2, "min_distance_voxels": 1,
                                 "exclude_border": True, "channel_labels": None, "measure_peak_intensity": True,
                                 "merge_radius_zyx": None, "method": "local_maxima"},
                "extraction": {"neighborhood_radius_zyx": [0, 1, 1], "sampling": "nearest", "boundary": "zero",
                               "background": {"inner_radius_zyx": [1, 3, 3], "outer_radius_zyx": [1, 6, 6],
                                              "min_voxels": 16}},
                "decoding": {"negative_policy": "reject", "diagnostics": False, "method": "wta"},
                "filtering": {"call_statuses": ["assigned"], "score_bounds": {}, "end_bases": "CC",
                              "start_base": "C", "exclude_invalid_endpoints": False, "exclude_duplicates": True},
                "scoring": {"method": "bgcorr_probability"},
                "deduplication": {"distance_voxels": 1.0, "compatibility": "same_sequence"}},
            "execution": {"mode": "batch", "retain_images": False, "device": "cpu"},
            "checkpoints": {"stages": list(stages), "directory": None, "table_format": "csv", "hash_inputs": True,
                            "overwrite": False},
            "readout_mode": "multiplexed",
            "encoding": {"method": "two_base", "reverse_bases": True, "pair_to_color": {
                "AA": "1", "AC": "2", "AG": "3", "AT": "4", "CA": "2", "CC": "1", "CG": "4", "CT": "3",
                "GA": "3", "GC": "4", "GG": "1", "GT": "2", "TA": "4", "TC": "3", "TG": "2", "TT": "1"}}},
        "channels": {"round1": _channels(), "round2": _channels()},
        "inputs": [{"path": f"{ROOT}/in/{r}/FOV_001/{c}.tif", "sha256": file_sha256(f"in/{r}/FOV_001/{c}.tif")}
                   for r in ROUNDS for c in CHANNELS],
        "steps": steps,
        "preprocessing": {
            "recipe": {"steps": ["min_max_normalization", "white_tophat"], "post_registration": [],
                       "extraction_source": None, "registration_source": None},
            "rounds": {name: _preprocessing_round(PINNED_FITTED[name]) for name in ROUNDS},
            "transforms": {"round1": {"detection": []},
                           "round2": {"detection": [{"result": 0, "method": "translation", "kind": "translation"}]}},
            "supplied_statistics": {"path": None, "sha256": None}},
        "registration": {"round2": [
            {"record": "estimation", "step": 0, "attempt": 0, "requested_method": "translation",
             "actual_method": "translation", "fallback": False, "backend": "scipy_fft", "backend_versions": {},
             "reference": "round1", "reference_sha256": None,
             "config": {"backend": "scipy_fft", "fft_workers": 1, "method": "translation"},
             "outcome": "succeeded", "failure": None, "qc": estimation_qc},
            {"record": "application", "outcome": "succeeded",
             "application_config": {"backend": "translation", "boundary_mode": "constant", "fill_value": 0,
                                    "output_dtype": "input", "integer_rounding": "nearest_even",
                                    "clip_to_dtype": True, "fft_workers": 1},
             "failure": None, "qc": application_qc}]},
        "counts": PINNED_COUNTS,
        "checkpoint_directory": f"{ROOT}/out/checkpoints/FOV_001",
        "checkpoints": checkpoints,
    }


# The values of this scene at the base revision (dc5fc3c, locked environment, one thread).
PINNED_FITTED = {"round1": [(68, 800), (57, 800), (69, 800), (69, 800)],
                 "round2": [(66, 800), (66, 800), (64, 800), (63, 800)]}
PINNED_QC_VALUES = {"coverage": 0.9583333333333334, "ncc_before": 0.0018992507837392752,
                    "ncc_after": 0.934932878901886, "ncc_gain": 0.9330336281181467,
                    "ssim_before": 0.38526367052047494, "ssim_after": 0.9287930480462401}
PINNED_QC_COUNTS = {"total": 2304, "valid": 2208, "valid_columns": 552, "ssim_columns": 306}
_MEDIANS = {"round1": {"ch00": (26.5, 3.0, 2.9652), "ch01": (39.5, 3.0, 3.7065), "ch02": (36.0, 3.0, 3.7065),
                       "ch03": (37.0, 3.0, 4.07715)},
            "round2": {"ch00": (34.5, 3.0, 2.9652), "ch01": (35.5, 3.0, 2.9652), "ch02": (40.5, 3.0, 2.9652),
                       "ch03": (39.5, 3.0, 4.4478)}}
PINNED_COUNTS = {
    "spots": 4, "intensities": 4, "call_status": {"assigned": 4},
    "scoring": {"total": 4, "scored": 4, "no_assignment": 0, "background_unavailable": 0},
    "deduplication": {"total": 4, "groups": 0, "merged_reads": 0, "conflicting_groups": 0},
    "filtering": {"total": 4, "accepted": 4, "rejected": 0},
    "summary": {
        "readout_mode": "multiplexed", "reads": 4,
        "call_status": {"assigned": 4, "ambiguous": 0, "no_signal": 0, "unmatched": 0},
        "failure_reason": {}, "call_type": {"exact": 4},
        "genes": {"g1": 1, "g2": 1, "g3": 1, "g4": 1}, "entries": {"12": 1, "21": 1, "34": 1, "43": 1},
        "qc_score": {"exact": {"n": 4, "q05": 0.11813371107852477, "q25": 0.14006466441611107,
                               "q50": 0.15844545181379877, "q75": 0.17378839153418724,
                               "q95": 0.1884285104462554}},
        "scoring": {"scored": 4, "no_assignment": 0, "background_unavailable": 0},
        "deduplication": {"total": 4, "groups": 0, "merged_reads": 0, "conflicting_groups": 0},
        "filtering": {"total": 4, "accepted": 4, "rejected": 0, "rejection_reasons": {}},
        "intensities": {
            "candidates": 4, "valid": {"round1": 4, "round2": 4}, "invalid": {"round1": 0, "round2": 0},
            "background_unavailable": {"round1": 0, "round2": 0},
            "medians": {r: {c: {"sum": s, "background": b, "noise": n} for c, (s, b, n) in channels.items()}
                        for r, channels in _MEDIANS.items()}},
        "cross_channel_pairs": {"distance_voxels": 1.0, "pairs": 0}}}


def pinned_files(stages):
    files = ["candidates.csv", "candidates.json", "pre_qc.csv", "pre_qc.json", "run.json"]
    if "registered" in stages:
        files += ["registered/round1.ome.tif", "registered/round2.ome.tif", "registered/transforms.json"]
    return sorted(files)


# ---------------------------------------------------------------------------------------

def test_run_record_holds_every_pinned_key_and_value(written):
    stages, record, _ = written
    assert differences(pinned_record(stages), record) == []


def test_checkpoint_folder_holds_every_pinned_file(written):
    stages, _, files = written
    assert [f for f in pinned_files(stages) if f not in files] == []


def test_the_comparison_accepts_additions_and_finds_removals_and_changes():
    pinned = {"a": 1, "b": {"c": [1.0, "x"], "d": None}, "e": True}
    added = {"a": 1, "b": {"c": [1.0, "x"], "d": None, "new": 2}, "e": True, "f": {"g": 3}}
    assert differences(pinned, added) == []
    assert differences(pinned, {"a": 1, "b": {"c": [1.0, "x"]}, "e": True}) == ["run.json.b: key 'd' is missing"]
    assert differences(pinned, {"a": 2, "b": {"c": [1.0, "x"], "d": None}, "e": True}) == ["run.json.a: 2 != 1"]
    assert differences(pinned, {"a": 1, "b": {"c": [1.0, "x"], "d": None}, "e": 1}) == ["run.json.e: 1 != True"]
    assert differences(pinned, {"a": 1, "b": {"c": [1.0], "d": None}, "e": True}) == [
        "run.json.b.c: [1.0] is not a list of 2 items"]
    assert differences(pinned, {"a": 1, "b": {"c": [1.0 + 1e-6, "x"], "d": None}, "e": True}) == [
        "run.json.b.c[0]: 1.000001 != 1.0"]
