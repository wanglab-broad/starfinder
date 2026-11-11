"""The worked example of the workflow contract on ``tiny_sample`` (design/validation/orchestration.md, "Fixtures").

``tiny_sample`` is W-642's fixture (``scripts/workflow_golden.py``, ``write_fixture``: ``sample-tiny-v1``, seed 42,
one sample of 2 × 2 FOVs of 5 × 36 × 36, four rounds of four channels, the reference stain ``ch04`` and the
additional round ``flamingo``). ``variants`` gives the contract's eight worked-example configurations
(design/specs/orchestration.md, "Worked example") in the Python shape, as W-643's prototype wrote them;
``after-fusion`` runs its processing tiled (``processing.mode: tiled``) with a tile size of 18 voxels, so that each FOV
of 36 × 36 voxels is the contract's four tiles (W-735 restored it; W-647 had planned it direct). ``write_supplied``
writes the supplied files the variants name (the BigStitcher result of ``supplied-xml`` and the import file of
``after-fusion``); their content is never read by the plan.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXPECTED = Path(__file__).parent / "data" / "expected" / "test_workflow_plan"
DATASET_ID = "sample-tiny-v1"
SAMPLE = "sample"
FOVS = ("FOV_001", "FOV_002", "FOV_003", "FOV_004")
VARIANTS = ("per-fov", "after-fusion", "molecules-only", "supplied-xml", "morphology-only", "reference-only")
REFUSED = ("refused-assignment", "refused-matlab")
#: The tasks per kind of each variant (the contract's table).
KINDS = {
    "per-fov": {"summarize": 4, "merge": 1, "fov": 4, "morphology": 4, "segment": 8, "assign": 4, "stitch": 1,
                "export": 1, "report": 1, "compare": 1},
    "after-fusion": {"tile_global": 4, "tile": 16, "tile_merge": 4, "morphology": 4, "bigstitcher_macro": 1,
                     "bigstitcher_engine": 1, "bigstitcher_import": 1, "export": 1, "report": 1, "compare": 1},
    "molecules-only": {"fov": 4, "stitch": 1, "export": 1, "report": 1, "compare": 1},
    "supplied-xml": {"fov": 4, "bigstitcher_import": 1, "export": 1, "report": 1, "compare": 1},
    "morphology-only": {"morphology": 4, "segment": 4},
    "reference-only": {"fov": 4, "morphology": 4, "segment": 8},
}
FIJI = "/stanley/WangLab/Tools/Fiji.app/ImageJ-linux64"

PROCESSING = {
    "load_codebook": {},
    "preprocessing": {"steps": [
        {"method": "min_max_normalization", "output_dtype": "uint8", "output_range": [0, 255],
         "rounding": "truncate"},
        {"method": "white_tophat", "radius_yx": 3}]},
    "registration": {"steps": [{"method": "translation"}], "signal": {"mode": "max"}},
    "spot_finding": {"method": "local_maxima", "exclude_border": False},
    "reads_extraction": {"voxel_size": [1, 2, 2]},
    "reads_filtration": {"end_base": "CC", "start_base": "C"},
    "decoding": {"method": "wta", "negative_policy": "clip_negative"},
    "deduplication": {},
    "checkpoints": ["candidates", "pre_qc"]}


def golden():
    """scripts/workflow_golden.py as a module (its fixture writer and comparison)."""
    spec = importlib.util.spec_from_file_location("workflow_golden", ROOT / "src" / "python" / "scripts" /
                                                  "workflow_golden.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def input_root(work):
    return Path(work) / "input" / DATASET_ID / SAMPLE


def output_root(work, output_id):
    return Path(work) / "output" / DATASET_ID / output_id


def dataset(work, output_id):
    """The dataset keys of every variant: W-642's fixture, its rounds, channels, stain and additional round."""
    work = Path(work)
    return {"backend": "python", "root_input_path": str(work / "input"), "dataset_id": DATASET_ID,
            "sample_id": SAMPLE, "root_output_path": str(work / "output"), "output_id": output_id,
            "n_rounds": 4, "ref_round": "round1", "ref_channel": "DAPI",
            "seq_channel_order": ["ch00", "ch01", "ch02", "ch03"],
            "reference_stains": [{"channel": "ch04", "name": "DAPI"}],
            "additional_round": [{"round_name": "flamingo", "channel_order": [
                {"channel": "ch04", "name": "DAPI"}, {"channel": "ch05", "name": "Flamingo"}]}],
            "rotate_angle": 0, "voxel_size_xy": 0.5, "voxel_size_z": 1.0}


def multiview(work):
    return {"method": "multiview_stitcher", "config": {"binning_zyx": [1, 2, 2], "quality_threshold": -1.0},
            "positions": {"source": "table", "table": str(input_root(work) / "positions.csv")}}


def grid():
    return {"source": "regular_grid", "regular_grid": {"columns": 2, "rows": 2, "order": "column_major_down_right",
                                                       "overlap": 1 / 3}}


def variants(work):
    """{variant: configuration} of the worked example on the fixture in ``work``."""
    work = Path(work)
    labels = str(work / "labels" / "{fov_id}.tif")
    out = {}
    processing = copy.deepcopy(PROCESSING)
    processing["preprocessing"]["steps"].insert(1, {"method": "percentile_normalization", "p_low": 1.0,
                                                    "p_high": 99.9, "fit": "supplied"})
    out["per-fov"] = dict(
        dataset(work, "example-per-fov"), processing=processing, morphology={},
        segmentation={"runs": [{"name": "nucleus", "target": "nucleus", "method": "import", "path": labels},
                               {"name": "cell", "target": "cell", "method": "import", "path": labels}]},
        assignment={"cells": "cell", "previews": True, "nuclei": "nucleus"},
        stitching=multiview(work), assembly={"assignment": "default"},
        export={"route": "per_fov", "images": ["reference_merged", "reference_stain"], "previews": True},
        report={}, execution={"tasks": {"fov": {"memory_mb": 16000}}})
    tiled = dict(copy.deepcopy(PROCESSING), mode="tiled", tiled={"tile_size_yx": 18})
    out["after-fusion"] = dict(
        dataset(work, "example-after-fusion"), processing=tiled, morphology={},
        stitching={"method": "bigstitcher", "bigstitcher_save_dir": str(work / "bigstitcher"), "positions": grid()},
        assembly={"assignment": "default"},
        export={"images": ["reference_merged", "reference_stain"], "segmentation": {"cells": "cell", "runs": [
            {"name": "cell", "target": "cell", "method": "import",
             "path": str(work / "labels" / "fused-cells.ome.tif")}]}},
        report={}, execution={"fiji": FIJI})
    out["molecules-only"] = dict(dataset(work, "example-molecules-only"), processing=copy.deepcopy(PROCESSING),
                                 stitching=multiview(work), assembly={"assignment": None},
                                 export={"images": ["reference_merged"]}, report={})
    out["supplied-xml"] = dict(
        dataset(work, "example-supplied-xml"), processing=copy.deepcopy(PROCESSING),
        stitching={"method": "bigstitcher", "dataset_xml": str(work / "supplied" / "{sample}" / "dataset.xml"),
                   "view_map": str(work / "supplied" / "{sample}" / "view_map.csv"), "positions": grid()},
        assembly={"assignment": None}, export={"images": ["reference_merged"]}, report={})
    out["morphology-only"] = dict(dataset(work, "example-morphology-only"), morphology={}, segmentation={
        "device": "cuda", "runs": [{"name": "nucleus", "target": "nucleus", "method": "stardist",
                                    "model": "2D_versatile_fluo", "scale": 1.0, "projection": True,
                                    "inputs": [{"role": "nuclear", "round": "reference_stain", "channel": "ch04"}]}]})
    reference = {k: v for k, v in copy.deepcopy(PROCESSING).items()
                 if k in ("load_codebook", "preprocessing", "registration", "checkpoints")}
    reference["checkpoints"] = []
    out["reference-only"] = dict(dataset(work, "example-reference-only"), processing=reference, morphology={},
                                 segmentation={"runs": [
                                     {"name": "nucleus", "target": "nucleus", "method": "import", "path": labels},
                                     {"name": "cell", "target": "cell", "method": "seeded_watershed",
                                      "seeds": "nucleus", "inputs": [
                                          {"role": "composite", "round": "reference_stain", "channel": "ch04",
                                           "prepare": {"function": "composite_nuclei_amplicon"}}]}]})
    out["refused-assignment"] = dict(dataset(work, "example-refused"), morphology={}, segmentation={"runs": [
        {"name": "cell", "target": "cell", "method": "import", "path": labels}]}, assignment={})
    refused = {k: v for k, v in dataset(work, "example-refused").items() if k != "backend"}
    out["refused-matlab"] = dict(refused, workflow_mode="direct", rules={"rsf_single_fov": {"run": True}})
    return out


def write_supplied(work):
    """The supplied files the variants name: supplied-xml's dataset.xml and view map, after-fusion's import file."""
    work = Path(work)
    folder = work / "supplied" / SAMPLE
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "dataset.xml").write_text("<SpimData/>\n")
    (folder / "view_map.csv").write_text("view,grid_column,grid_row,fov_id,kind\n")
    (work / "labels" / "fused-cells.ome.tif").write_bytes(b"supplied")


def write_config(work, name, config):
    import yaml
    path = Path(work) / "configs" / f"{name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


PLACEHOLDERS = {"mtime_ns": "<mtime_ns>", "bytes": "<bytes>"}
_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")


def normalized(manifest, work):
    """A manifest as its committed expected form: the work folder written ``<work>``, the inputs' sizes and times
    and every hash (which hold them, or the absolute paths of the configuration) written as placeholders."""
    def walk(value, key=None):
        if isinstance(value, dict):
            return {k: (PLACEHOLDERS[k] if k in PLACEHOLDERS else walk(v, k)) for k, v in value.items()}
        if isinstance(value, list):
            return [walk(v) for v in value]
        if isinstance(value, str):
            return "<sha256>" if _HASH.match(value) else value.replace(str(work), "<work>")
        return value
    return walk(json.loads(json.dumps(manifest)))


def single_fov(work, output_id="single", *, downstream=False, **execution):
    """One FOV of the fixture (FOV_001) with the molecules-only variant's processing block; with ``downstream`` the
    stitching, assembly, export and report blocks too (planned, not run); ``execution`` sets execution keys."""
    config = dict(dataset(work, output_id), selected_fovs=["FOV_001"], processing=copy.deepcopy(PROCESSING))
    if downstream:
        config.update(stitching=multiview(work), assembly={"assignment": None},
                      export={"images": ["reference_merged"]}, report={})
    if execution:
        config["execution"] = dict(execution)
    return config


#: A clean code identity of the package, for the tests of verified resume (a dirty tree refuses a resume).
CLEAN_CODE = {"version": "0.2.0", "version_source": "checkout", "git_commit": "0123456789abcdef0123456789abcdef01234567", "git_dirty": False}


def fix_code(monkeypatch, **code):
    """Make ``starfinder._records.code`` (the run records' and the plan's code identity) return ``code`` (by default
    CLEAN_CODE), its cache cleared, as test_workflow_orchestration_golden.py patches it."""
    import starfinder._records as records
    if hasattr(records.code, "cache_clear"):
        records.code.cache_clear()
    identity = dict(CLEAN_CODE, **code)
    monkeypatch.setattr(records, "code", lambda: dict(identity))


def run_task(manifest, task, *extra):
    """Run one task's command in this process (``starfinder workflow <command> --manifest M --task T``)."""
    from starfinder.__main__ import main
    from starfinder.workflow._shape import COMMANDS
    kind = next(t["kind"] for t in json.loads(Path(manifest).read_text())["tasks"] if t["id"] == task)
    return main(["workflow", COMMANDS[kind], "--manifest", str(manifest), "--task", task, *extra])


def write_expected(work, out):
    """The committed expected manifests of C1 (``data/expected/test_workflow_plan/<variant>.json``): the plan of each
    variant on a fixture in the new folder ``work``, with test_workflow_plan's package identity, normalized."""
    from unittest import mock
    from starfinder.workflow import _plan
    from .test_workflow_plan import CODE
    golden().write_fixture(work)
    write_supplied(work)
    Path(out).mkdir(parents=True, exist_ok=True)
    with mock.patch.object(_plan, "package_identity", lambda: dict(CODE)):
        for variant in VARIANTS:
            manifest = _plan.plan(write_config(work, variant, variants(work)[variant]))
            (Path(out) / f"{variant}.json").write_text(json.dumps(normalized(manifest, work), indent=1) + "\n")


if __name__ == "__main__":
    import sys
    write_expected(Path(sys.argv[1]), Path(sys.argv[2]))
