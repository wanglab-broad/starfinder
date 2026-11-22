"""The worked example's configurations of the workflow contract on ``tiny_sample`` (``workflow_fixture.py``).

``variants`` gives the contract's eight worked-example configurations (design/specs/orchestration.md, "Worked
example") in the Python shape, as W-643's prototype wrote them; ``after-fusion`` runs its processing tiled
(``processing.mode: tiled``) with a tile size of 18 voxels, so that each FOV of 36 × 36 voxels is the contract's four
tiles (W-735 restored it; W-647 had planned it direct). ``single_fov`` is one FOV of the fixture with the
molecules-only variant's processing block.
"""
from __future__ import annotations

import copy
from pathlib import Path

from .workflow_fixture import DATASET_ID, SAMPLE, input_root

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
    # The noise rule, named: the default before W-880 (adaptive 0.2 since), which the fixture's expected records pin.
    "spot_finding": {"method": "local_maxima", "threshold_mode": "noise", "threshold_value": 5.0, "exclude_border": False},
    "reads_extraction": {"voxel_size": [1, 2, 2]},
    "reads_filtration": {"end_base": "CC", "start_base": "C"},
    "decoding": {"method": "wta", "negative_policy": "clip_negative"},
    "deduplication": {},
    "checkpoints": ["candidates", "pre_qc"]}


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
                                    "model": "2D_versatile_fluo",
                                    "input_recipe": {"projection": {"method": "max"}, "model_input": {"factor": 1.0}},
                                    "inputs": [{"role": "nuclear", "round": "reference_stain", "channel": "ch04"}]}]})
    reference = {k: v for k, v in copy.deepcopy(PROCESSING).items()
                 if k in ("load_codebook", "preprocessing", "registration", "checkpoints")}
    reference["checkpoints"] = []
    out["reference-only"] = dict(dataset(work, "example-reference-only"), processing=reference, morphology={},
                                 segmentation={"runs": [
                                     {"name": "nucleus", "target": "nucleus", "method": "import", "path": labels},
                                     {"name": "cell", "target": "cell", "method": "seeded_watershed",
                                      "seeds": "nucleus", "input_recipe": {"combine": {"method": "composite"}},
                                      "inputs": [{"role": "nuclear", "round": "reference_stain", "channel": "ch04"},
                                                 {"role": "amplicon", "reference_merged": True}]}]})
    out["refused-assignment"] = dict(dataset(work, "example-refused"), morphology={}, segmentation={"runs": [
        {"name": "cell", "target": "cell", "method": "import", "path": labels}]}, assignment={})
    refused = {k: v for k, v in dataset(work, "example-refused").items() if k != "backend"}
    out["refused-matlab"] = dict(refused, workflow_mode="direct", rules={"rsf_single_fov": {"run": True}})
    return out


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
