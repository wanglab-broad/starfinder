"""C3 of design/validation/orchestration.md: the declared outputs of every task kind.

For each worked-example variant and each task, the declared outputs equal the contract's table
(design/specs/orchestration.md, "Declared outputs", with the tiled kinds' files of design/specs/orchestration-tiled.md,
"Declared files of the tiled task kinds") resolved through ``_layout``; and every task's outputs, record and log equal
its group's templates filled with its identity, the chunked checkpoint its group's one temporary output. The rows of
the morphology, segment and assign kinds (W-736) are checked on the variants and on configurations built here: the
channel images with ``morphology.channel_images``, the displacement field of a dense ``morphology.recipe``, the nucleus
table and the territories of an assignment with ``nuclei`` and ``expansion``, its previews with
``assignment.previews``. And the declarations of the old rule files that moved here (W-750): the reference image where
reference_image or reference_image_import names it, the position file of the placing tasks, the export task's inputs
by route and its preview folder, the assign task's saved folder, the report task's inputs, the run records of the fov
kinds and the morphology task's inputs; these plan configurations of the test's own on an input root whose raw images
are empty files (``workflow_tasks``). Where the old rule declared less, the task declares every file its command
reads (D2: the manifest lists each task's inputs), named in the test.
"""
import os

import pytest

from starfinder import _layout
from starfinder.workflow import plan

from . import workflow_examples as X
from . import workflow_tasks as W

pytestmark = [pytest.mark.workflow, pytest.mark.contract]

KINDS = ("summarize", "merge", "fov", "tile_global", "tile", "tile_merge", "morphology", "segment", "assign",
         "bigstitcher_macro", "bigstitcher_engine", "bigstitcher_import", "stitch", "export", "report", "compare")
#: The registration methods whose transform is a displacement field or a B-spline (the saved form's field.npz).
DENSE = ("bspline", "demons", "tps", "cpd")


@pytest.fixture(scope="module")
def manifests(tmp_path_factory):
    work = tmp_path_factory.mktemp("tiny") / "work"
    X.golden().write_fixture(work)
    X.write_supplied(work)
    return work, {v: plan(X.write_config(work, v, X.variants(work)[v])) for v in X.VARIANTS}


def _rel(path, root):
    path, root = os.path.abspath(path), os.path.abspath(root)
    return os.path.relpath(path, root).replace(os.sep, "/") if os.path.commonpath([path, root]) == root else path


def expected_outputs(task, config, out):
    """The contract's declared outputs of a task (its table's row for the task's kind), through _layout."""
    kind, identity = task["kind"], task["identity"]
    sample, fov = identity.get("sample"), identity.get("fov")
    processing = config.get("processing") or {}
    if kind == "summarize":
        paths = [_layout.statistics_dir(out, sample, identity["step"]) / "histograms" / f"{fov}.npz"]
    elif kind == "merge":
        paths = [_layout.statistics_dir(out, sample, identity["step"]) / "supplied.json"]
    elif kind == "fov":
        folder = _layout.fov_dir(out, fov)
        paths = [_layout.run_record(out, fov), _layout.reference_image_path(out, fov)]
        if all(k in processing for k in ("spot_finding", "reads_extraction", "reads_filtration")):
            paths.append(_layout.spot_table(out, fov))
        stages = processing.get("checkpoints", ["candidates", "pre_qc"])
        if "candidates" in stages and "spot_finding" in processing:
            paths.append(_layout.checkpoint_dir(folder) / "candidates.json")
        if "pre_qc" in stages and "reads_filtration" in processing:
            paths.append(_layout.checkpoint_dir(folder) / "pre_qc.json")
    elif kind == "tile_global":
        tiled = _layout.tiled_dir(_layout.fov_dir(out, fov))
        paths = [_layout.reference_image_path(out, fov), tiled / "registered.h5", tiled / "global.json",
                 tiled / "run.json"]
    elif kind == "tile":
        folder = _layout.tile_dir(_layout.fov_dir(out, fov), identity["tile"])
        paths = [folder / "candidates.csv", folder / "reads.csv", folder / "tile.json"]
    elif kind == "tile_merge":
        # As direct's fov task (D5): run_record, spot_table and the stage records, but not the reference image, which
        # the global task writes.
        folder = _layout.fov_dir(out, fov)
        paths = [_layout.run_record(out, fov), _layout.spot_table(out, fov)]
        stages = processing.get("checkpoints", ["candidates", "pre_qc"])
        if "candidates" in stages:
            paths.append(_layout.checkpoint_dir(folder) / "candidates.json")
        if "pre_qc" in stages:
            paths.append(_layout.checkpoint_dir(folder) / "pre_qc.json")
    elif kind == "morphology":
        block = config["morphology"]
        folder = _layout.fov_dir(out, fov)
        rounds = block.get("rounds", [e["round_name"] for e in config.get("additional_round") or []])
        steps = ((block.get("recipe") or {}).get("steps")) or []
        dense = any(step["method"] in DENSE for step in steps)
        paths = [_layout.nuclei_registration_record(out, fov)]
        for name in (["reference_stain"] if config.get("reference_stains") else []) + list(rounds):
            paths += [_layout.other_round_dir(folder, name) / "image.ome.tif",
                      _layout.other_round_dir(folder, name) / "registration.json"]
            if dense and name != "reference_stain":
                paths.append(_layout.other_round_dir(folder, name) / "field.npz")
        if block.get("channel_images") is not None:
            for entry in config.get("additional_round") or []:
                if entry["round_name"] in rounds:
                    paths += [_layout.image_file(out, fov, entry["round_name"], channel["channel"])
                              for channel in entry["channel_order"]]
    elif kind == "segment":
        (run,) = [r for r in config["segmentation"]["runs"] if r["name"] == identity["run"]]
        folder = _layout.segmentation_dir(_layout.fov_dir(out, fov), run["name"])
        paths = [folder / "labels.tif", folder / "segmentation.json"]
        if run["method"] != "import":
            paths.append(folder / "input.ome.tif")
    elif kind == "assign":
        block = config["assignment"]
        folder = _layout.assignment_dir(_layout.fov_dir(out, fov), block.get("name", "default"))
        paths = [folder / "molecules.csv", folder / "cells.csv", folder / "counts.csv", folder / "assignment.json"]
        if block.get("nuclei"):
            paths.append(folder / "nuclei.csv")
        if block.get("expansion"):
            paths.append(folder / "territories.tif")
        if block.get("previews"):
            paths.append(str(_layout.fov_previews_dir(out, fov)) + "/")
    elif kind in ("bigstitcher_macro", "bigstitcher_engine"):
        save = os.path.join(config["stitching"]["bigstitcher_save_dir"], sample)
        paths = ([os.path.join(save, "BigStitcher_macro.ijm"), os.path.join(save, "view_map.csv"),
                  os.path.join(save, "tiles") + "/"] if kind == "bigstitcher_macro"
                 else [os.path.join(save, "dataset.xml")])
    elif kind in ("bigstitcher_import", "stitch"):
        paths = [_layout.stitching_dir(out, sample) / name for name in ("placement.json", "fovs.csv", "links.csv")]
    elif kind == "export":
        paths = [_layout.export_dir(out, sample) / _layout.EXPORT_FILE]
        if (config.get("export") or {}).get("previews") and (config.get("export") or {}).get("route") == "per_fov":
            paths.append(str(_layout.export_previews_dir(out, sample)) + "/")
    elif kind == "report":
        paths = [_layout.report_dir(out, sample) / "report.json"]
    else:
        paths = [_layout.report_root(out) / "samples.csv", _layout.report_root(out) / "index.html"]
    return sorted(_rel(str(p).rstrip("/"), out) + ("/" if str(p).endswith("/") else "") for p in paths)


@pytest.mark.parametrize("variant", X.VARIANTS)
def test_the_declared_outputs_are_the_contracts(manifests, variant):
    work, planned = manifests
    manifest = planned[variant]
    config = X.variants(work)[variant]
    out = manifest["roots"]["output"]
    checked = 0
    for task in manifest["tasks"]:
        if task["kind"] not in KINDS:
            continue
        assert task["outputs"] == expected_outputs(task, config, out), task["id"]
        assert task["record"] == _rel(_layout.task_record(out, task["id"]), out)
        assert task["log"] == _rel(_layout.task_log(out, task["id"]), out)
        checked += 1
    assert checked == sum(n for kind, n in X.KINDS[variant].items() if kind in KINDS)


@pytest.mark.parametrize("variant", X.VARIANTS)
def test_every_task_fills_its_groups_templates(manifests, variant):
    _, planned = manifests
    manifest = planned[variant]
    groups = {group["name"]: group for group in manifest["groups"]}
    assert len(groups) == len(manifest["groups"])
    for task in manifest["tasks"]:
        group = groups[task["group"]]
        assert group["kind"] == task["kind"]
        assert group["wildcards"] == [k for k in ("sample", "step", "fov", "tile") if k in task["identity"]]
        fill = {k: task["identity"][k] for k in group["wildcards"]}
        assert sorted(t.format(**fill) for t in group["outputs"]) == task["outputs"], task["id"]
        assert group["record"].format(**fill) == task["record"]
        assert group["log"].format(**fill) == task["log"]
        # The chunked checkpoint is the one temporary output (the engine record), unless the stage registered is
        # kept; no worked-example variant keeps it.
        assert group["temporary"] == (["fovs/{fov}/tiled/registered.h5"] if group["kind"] == "tile_global" else [])
    assert {t["group"] for t in manifest["tasks"]} == set(groups)


def test_the_route_after_fusion_declares_the_bigstitcher_folder_outside_the_output_root(manifests):
    work, planned = manifests
    manifest = planned["after-fusion"]
    save = os.path.join(str(work / "bigstitcher"), "sample")
    engine = next(t for t in manifest["tasks"] if t["kind"] == "bigstitcher_engine")
    assert engine["inputs"] == sorted([os.path.join(save, "BigStitcher_macro.ijm"), os.path.join(save, "tiles") + "/",
                                       os.path.join(save, "view_map.csv")])
    assert engine["tool_command"] == ["{fiji}", f"-Djava.io.tmpdir={save}", "--ij2", "--headless", "--run",
                                      os.path.join(save, "BigStitcher_macro.ijm")]


def test_the_tile_merge_declares_the_files_of_direct(manifests):
    """D5: tile_merge declares the run record, the molecule table and the stage records of direct's fov task, and
    tile_global the reference image, so that their consumers read the same files whichever mode made them."""
    work, planned = manifests
    direct = {t["identity"]["fov"]: set(t["outputs"]) for t in planned["molecules-only"]["tasks"]
              if t["kind"] == "fov"}
    tasks = planned["after-fusion"]["tasks"]
    for fov, outputs in direct.items():
        merge = next(t for t in tasks if t["kind"] == "tile_merge" and t["identity"]["fov"] == fov)
        glob = next(t for t in tasks if t["kind"] == "tile_global" and t["identity"]["fov"] == fov)
        relocated = {o.replace("example-molecules-only", "example-after-fusion") for o in outputs}
        assert set(merge["outputs"]) | {o for o in glob["outputs"] if o.endswith("reference.tif")} == relocated
        tiles = [t for t in tasks if t["kind"] == "tile" and t["identity"]["fov"] == fov]
        assert sorted(t["identity"]["tile"] for t in tiles) == ["r0c0", "r0c1", "r1c0", "r1c1"]
        assert set(merge["needs"]) == {glob["id"], *(t["id"] for t in tiles)}


def _built(work, name, config):
    """The manifest of a configuration built here, on the variants' fixture."""
    return plan(X.write_config(work, name, config))


@pytest.mark.parametrize("case", ["channel images", "dense recipe", "nuclei and expansion", "no previews"])
def test_the_options_of_the_cell_kinds_declare_their_files(manifests, case):
    """The options of the contract's table, column "By option", of morphology and assign (W-736)."""
    work, _ = manifests
    config = X.variants(work)["per-fov"]
    if case == "channel images":
        config["morphology"] = {"channel_images": {"projection": "max"}}
    elif case == "dense recipe":
        config["morphology"] = {"recipe": {"steps": [{"method": "translation"}, {"method": "demons"}],
                                           "signal": {"mode": "channel", "reference_channel": "ch04",
                                                      "moving_channel": "ch04"}}}
    elif case == "nuclei and expansion":
        config["assignment"] = dict(config["assignment"], expansion={"distance": 1.0, "unit": "um", "mode": "planar"})
    else:
        config["assignment"] = dict(config["assignment"], previews=False)
    manifest = _built(work, case.replace(" ", "-"), config)
    out = manifest["roots"]["output"]
    tasks = [t for t in manifest["tasks"] if t["kind"] in ("morphology", "segment", "assign")]
    assert len(tasks) == 16
    for task in tasks:
        assert task["outputs"] == expected_outputs(task, config, out), task["id"]
    morphology = next(t for t in tasks if t["kind"] == "morphology")
    assign = next(t for t in tasks if t["kind"] == "assign")
    if case == "channel images":
        assert {o for o in morphology["outputs"] if o.startswith("images/")} == {
            "images/flamingo/ch04/FOV_001.tif", "images/flamingo/ch05/FOV_001.tif"}
    elif case == "dense recipe":
        assert "fovs/FOV_001/other_rounds/flamingo/field.npz" in morphology["outputs"]
        assert not any(o.endswith("reference_stain/field.npz") for o in morphology["outputs"])
    elif case == "nuclei and expansion":
        assert {"fovs/FOV_001/assignment/default/nuclei.csv", "fovs/FOV_001/assignment/default/territories.tif"} <= \
            set(assign["outputs"])
    else:
        assert not any("previews" in o for o in assign["outputs"])


# --- the declarations of the old rule files (W-750) ----------------------------------------------------------------

import copy  # noqa: E402
import re  # noqa: E402
from pathlib import Path  # noqa: E402

from starfinder.workflow._plan import PlanRefused  # noqa: E402

MERGED = {"method": "multiview_stitcher", "input": "reference_merged", "positions": X.grid()}
TWO = "sample,fov_id\nsample,F1\nsample,F2\n"


def _planned(work, name, config, fovs=TWO, **files):
    """Plan ``config`` on an input root with fovs.csv and empty raw images; ``files`` names more empty files."""
    W.write_inputs(config, fovs)
    for path in files.values():
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"")
    return W.plan(work, name, config)[1]


def _kinds(manifest, *kinds):
    return [t for t in manifest["tasks"] if t["kind"] in kinds]


def _cells(work, **assignment):
    """A segment run importing the cells and the assignment block reading it."""
    return {"segmentation": {"runs": [{"name": "cell", "target": "cell", "method": "import",
                                       "path": str(Path(work) / "labels" / "{fov_id}.tif")}]},
            "assignment": {"cells": "cell", **assignment}}


def _labels(work, fovs=("F1", "F2")):
    return {fov: Path(work) / "labels" / f"{fov}.tif" for fov in fovs}


@pytest.mark.parametrize("setting", ["default", "relative", "absolute", "imported relative", "imported absolute"])
def test_the_reference_image_is_declared_where_reference_image_names_it(tmp_path, setting):
    """From test_dataset_fov_layout.py (the declarations of the reference image and of an imported one) and
    test_stitching_reference_image.py, test_the_setting_defaults_to_the_present_location: the fov task declares the
    file reference_image names (relative to the output root, or absolute), none with reference_image_import; the
    tasks that read it (stitch, segment, assign) declare the same file as an input."""
    keys = {"default": {}, "relative": {"reference_image": "refs/{fov_id}_ref.tif"},
            "absolute": {"reference_image": str(tmp_path / "elsewhere" / "{fov_id}.tif")},
            "imported relative": {"reference_image_import": "imports/{fov_id}.tif"},
            "imported absolute": {"reference_image_import": str(tmp_path / "tool" / "{fov_id}.tif")}}[setting]
    declared = {"default": "fovs/F1/reference.tif", "relative": "refs/F1_ref.tif",
                "absolute": str(tmp_path / "elsewhere" / "F1.tif"), "imported relative": "imports/F1.tif",
                "imported absolute": str(tmp_path / "tool" / "F1.tif")}[setting]
    config = W.dataset_keys(tmp_path, processing=copy.deepcopy(X.PROCESSING), stitching=MERGED, **_cells(tmp_path),
                            **keys)
    manifest = _planned(tmp_path, "reference", config, fovs="sample,fov_id\nsample,F1\n",
                        **{k: str(v) for k, v in _labels(tmp_path, ("F1",)).items()})
    (fov,) = _kinds(manifest, "fov")
    assert (declared in fov["outputs"]) is not setting.startswith("imported")
    assert not [o for o in fov["outputs"] if o.endswith(".tif") and o != declared]
    readers = {t["kind"] for t in manifest["tasks"] if declared in t["inputs"]}
    assert {"stitch", "assign"} <= readers


#: The position file of each source: the configuration's positions, and the file the placing tasks declare.
POSITIONS = {"stage_file_path": ({"source": "stage_file", "stage_axes": {"x": "+x", "y": "+y"}}, "LN.maf"),
             "table": ({"source": "table", "unit": "um"}, "positions.csv"),
             "regular_grid": (X.grid(), None)}


def _positions(work, case, method):
    positions, name = POSITIONS[case]
    positions = copy.deepcopy(positions)
    if name is not None:
        positions[positions["source"]] = str(Path(work) / "stage" / name)
    block = dict(MERGED, positions=positions) if method == "multiview_stitcher" else {
        "positions": positions, "bigstitcher_save_dir": str(Path(work) / "bigstitcher")}
    config = W.dataset_keys(work, reference_image_import=str(Path(work) / "refs" / "{fov_id}.tif"),
                            voxel_size_xy=1.0, voxel_size_z=1.0, stitching=block)
    return config, None if name is None else str(Path(work) / "stage" / name)


@pytest.mark.parametrize("method", ["multiview_stitcher", "bigstitcher"])
@pytest.mark.parametrize("case", sorted(POSITIONS))
def test_the_placing_tasks_declare_the_position_file(tmp_path, case, method):
    """From test_stitching_workflow.py, test_wf3_placement_rules_declare_the_position_file and
    test_wf3_the_placement_rules_use_the_position_input_function: the stitch task, or the BigStitcher macro and import
    tasks, declare the stage file or the supplied table the block names, and nothing for a regular grid; the engine
    task (Fiji) does not."""
    from .stitching_fixtures import maf_text
    config, path = _positions(tmp_path, case, method)
    if path is not None:
        Path(path).parent.mkdir(parents=True)
        Path(path).write_text(maf_text(((1, "0.076", "0.048"), (2, "0.076", "0.048261"))) if path.endswith(".maf")
                              else "fov_id,nominal_z,nominal_y,nominal_x\nF1,0,0,0\nF2,0,1,0\n")
    manifest = _planned(tmp_path, case, config, fovs="sample,fov_id,position_id\nsample,F1,1\nsample,F2,2\n")
    placing = _kinds(manifest, "stitch", "bigstitcher_macro", "bigstitcher_import")
    assert [t["kind"] for t in placing] == (["stitch"] if method == "multiview_stitcher"
                                            else ["bigstitcher_macro", "bigstitcher_import"])
    for task in placing:
        assert [i for i in task["inputs"] if "/stage/" in i] == ([] if path is None else [path]), task["id"]
    for task in _kinds(manifest, "bigstitcher_engine"):
        assert not [i for i in task["inputs"] if "/stage/" in i]


@pytest.mark.parametrize("case", ["stage_file_path", "table"])
def test_the_declared_position_file_is_the_file_dataset_stitch_reads(tmp_path, monkeypatch, case):
    """From test_stitching_workflow.py, the test of the same name: table_from_positions, which the stitch command's
    Dataset.stitch calls, opens the file the stitch task declares."""
    import starfinder.dataset._stitching as positions_module
    from .stitching_fixtures import STAGE_SPACING, grid, maf_text
    from starfinder.io import save_volume
    import numpy as np
    config, path = _positions(tmp_path, case, "multiview_stitcher")
    Path(path).parent.mkdir(parents=True)
    Path(path).write_text(maf_text(((1, "0.076", "0.048"), (2, "0.076", "0.048261"))) if case == "stage_file_path"
                          else "fov_id,nominal_z,nominal_y,nominal_x\nF1,0,0,0\nF2,0,1,0\n")
    for fov in ("F1", "F2"):
        save_volume(np.zeros((2, 16, 24), np.uint16), tmp_path / "refs" / f"{fov}.tif",
                    metadata=grid((2, 16, 24), STAGE_SPACING).metadata)
    W.write_inputs(config, "sample,fov_id,position_id\nsample,F1,1\nsample,F2,2\n")
    manifest_path, manifest = W.plan(tmp_path, case, config)
    opened = []

    def spy(source, **kwargs):
        opened.append(str(source))
        raise StopIteration

    monkeypatch.setattr(positions_module, "positions_from_leica", spy)
    monkeypatch.setattr(positions_module, "positions_from_table", spy)
    with pytest.raises(StopIteration):
        W.call(manifest_path, "stitch/sample")
    (stitch,) = _kinds(manifest, "stitch")
    assert opened == [i for i in stitch["inputs"] if "/stage/" in i] == [path]


def _route_inputs(manifest, config, combination):
    """The export task's inputs a route declares (the contract's table, "Declared outputs"), built independently: the
    placement; per FOV, the saved assignment's files (route per FOV) or the molecule table and the reference image
    (the molecule routes, with the codebook); with fuse, each source image; after fusion, the imported segmentation
    files."""
    route, dimensionality, segment, fuse, segmentation, assignment = combination
    out = W.output_root(config)
    expected = {"stitching/sample/placement.json", "stitching/sample/fovs.csv", "stitching/sample/links.csv"}
    way = "molecules" if assignment is None else (route or ("after_fusion" if dimensionality == "projection"
                                                            else "per_fov"))
    for fov in ("F1", "F2"):
        if way == "per_fov":
            expected |= {f"fovs/{fov}/assignment/default/{n}" for n in ("assignment.json", "cells.csv", "counts.csv",
                                                                         "molecules.csv")}
        else:
            expected |= {f"fovs/{fov}/molecules.csv", f"fovs/{fov}/reference.tif", "input:genes.csv"}
        if fuse:
            expected |= {f"fovs/{fov}/other_rounds/reference_stain/image.ome.tif", f"fovs/{fov}/reference.tif",
                         str(out / "images" / "dapi" / f"{fov}.tif"),
                         str(Path(config["root_input_path"]).parent / "flamingo" / f"{fov}.tif")}
    if way == "after_fusion" and segmentation:
        expected |= {run["path"] for run in config["export"]["segmentation"]["runs"] if run["method"] == "import"}
    return expected


def _route_config(tmp_path, combination):
    from .test_workflow_plan_refusals import _route
    config = _route(tmp_path, combination)
    segmentation = config["export"].get("segmentation")
    if segmentation:
        nuclei = tmp_path / "nuclei.ome.tif"
        nuclei.write_bytes(b"")
        for run in segmentation["runs"]:
            if run["method"] == "import":
                run["path"] = str(nuclei)
    return config


def _combinations():
    from .test_sample_segmentation_workflow import COMBINATIONS
    return COMBINATIONS


@pytest.mark.parametrize("combination", _combinations(), ids=lambda c: "-".join(map(str, c)))
def test_the_export_task_declares_its_inputs_by_route(tmp_path, combination):
    """From test_sample_segmentation_workflow.py, test_wf1_the_rule_s_inputs_by_route, and test_stitching_workflow.py,
    test_wf3_export_sample_inputs_and_declared_output: on the route per FOV each FOV's saved assignment (its folder's
    files, no raw.h5ad or reads_assignment.csv), on the route after fusion the molecule tables, the reference images
    and the source images and no saved assignment, molecules only the molecule tables on either route; export.json is
    the one declared output. A configuration the route's check refuses by itself is refused by the plan, naming the
    key. (The old rule declared the record of each saved assignment alone, the molecule routes without the reference
    images and the codebook, and no source image of the route per FOV: the task declares every file its command reads,
    D2.)"""
    from .test_sample_segmentation_workflow import refused
    config = _route_config(tmp_path, combination)
    error = refused(*combination)
    if error is not None:
        with pytest.raises(PlanRefused, match=error):
            W.plan(tmp_path, "route", config)
        return
    _, manifest = W.plan(tmp_path, "route", config)
    (export,) = _kinds(manifest, "export")
    assert export["outputs"] == ["export/sample/export.json"]
    assert set(export["inputs"]) == _route_inputs(manifest, config, combination)
    assert not [i for i in export["inputs"] if "/expr/" in i or i.endswith(("raw.h5ad", "reads_assignment.csv"))]


def test_the_export_task_declares_the_reference_images_and_the_named_images(tmp_path):
    """From test_sample_segmentation_workflow.py, test_wf1_the_inputs_follow_reference_image_and_the_named_images: the
    reference images are the files reference_image names, reference_merged that file, listed once; with
    reference_stain alone its saved form is declared (a declared output of the morphology task, D6; the old rule
    declared no source image then) and no named file."""
    combination = ("after_fusion", "projection", True, True, True, "default")
    config = _route_config(tmp_path, combination)
    config["reference_image"] = "images/flamingo/DAPI/{fov_id}.tif"
    _, manifest = W.plan(tmp_path, "named", config)
    (export,) = _kinds(manifest, "export")
    for fov in ("F1", "F2"):
        assert export["inputs"].count(f"images/flamingo/DAPI/{fov}.tif") == 1
        assert f"fovs/{fov}/reference.tif" not in export["inputs"]
    config["export"]["images"] = ["reference_stain"]
    config["export"]["segmentation"]["runs"][1]["inputs"] = [{"role": "cytoplasm", "image": "reference_stain"}]
    _, manifest = W.plan(tmp_path, "stain", config)
    (export,) = _kinds(manifest, "export")
    assert {"fovs/F1/other_rounds/reference_stain/image.ome.tif",
            "fovs/F2/other_rounds/reference_stain/image.ome.tif"} <= set(export["inputs"])
    assert not [i for i in export["inputs"] if "/dapi/" in i or "/flamingo/F" in i]


def test_a_configuration_without_the_new_keys_keeps_its_route(tmp_path):
    """From test_sample_segmentation_workflow.py, test_wf1_existing_configurations_without_the_new_keys: the stitching
    and assembly blocks without the export keys of the routes: a stack or both dimensionalities plan the route per
    FOV, an assembly without an assignment the molecules, and a projection with an assignment and no
    export.segmentation is refused, naming export.segmentation and route: per_fov, which then plans the route per FOV.
    (The parse-time check of common.smk is the plan's.)"""
    base = W.dataset_keys(tmp_path, processing=copy.deepcopy(X.PROCESSING), stitching=MERGED, **_cells(tmp_path))
    for name, (export, assignment, expected) in enumerate((
            (None, "default", None), ({"dimensionality": "stack"}, "default", "per_fov"),
            ({"dimensionality": "both"}, "default", "per_fov"), (None, None, "molecules"))):
        config = dict(copy.deepcopy(base), output_id=f"case{name}", assembly={"assignment": assignment},
                      export=dict(export or {}, images=["reference_merged"]))
        if assignment is None:
            config.pop("assignment")
            config.pop("segmentation")
        W.write_inputs(config, TWO)
        for path in _labels(tmp_path).values():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"")
        if expected is None:
            with pytest.raises(PlanRefused, match=r"export\.segmentation.*route: per_fov"):
                W.plan(tmp_path, f"case{name}", config)
            config["export"]["route"] = "per_fov"
            expected = "per_fov"
        _, manifest = W.plan(tmp_path, f"case{name}", config)
        (task,) = _kinds(manifest, "export")
        saved = [i for i in task["inputs"] if i.endswith("assignment.json")]
        molecules = [i for i in task["inputs"] if re.fullmatch(r"fovs/F\d/molecules\.csv", i)]
        assert (saved, molecules) == ((["fovs/F1/assignment/default/assignment.json",
                                        "fovs/F2/assignment/default/assignment.json"], [])
                                      if expected == "per_fov" else ([], ["fovs/F1/molecules.csv",
                                                                           "fovs/F2/molecules.csv"])), name


@pytest.mark.parametrize("case, declared", [
    ("default", False), ("previews", True), ("molecules", False), ("no assemble", False), ("after fusion", False),
    ("projection", False)])
def test_the_export_task_declares_the_ownership_previews_on_request(tmp_path, case, declared):
    """From test_stitching_workflow.py, test_wf3_export_sample_declares_the_ownership_previews_on_request: with
    export.previews on (P5; the old rule read assembly.previews), an assignment name, the route per FOV and assemble
    on, the export task declares export/<sample>/previews/ (a folder) beside export.json; by default, with assignment
    null, after fusion (the route set, or the default of dimensionality projection) and with assemble off, nothing
    more."""
    export = {"default": {"route": "per_fov"}, "previews": {"route": "per_fov", "previews": True},
              "molecules": {"route": "per_fov", "previews": True},
              "no assemble": {"route": "per_fov", "assemble": False, "previews": True},
              "after fusion": {"route": "after_fusion", "previews": True},
              "projection": {"dimensionality": "projection", "previews": True}}[case]
    if case in ("after fusion", "projection"):
        export["segmentation"] = {"cells": "cell", "runs": [{"name": "cell", "target": "cell", "method": "import",
                                                             "path": str(tmp_path / "fused.tif")}]}
        (tmp_path / "fused.tif").write_bytes(b"")
    assignment = None if case == "molecules" else "default"
    config = W.dataset_keys(tmp_path, processing=copy.deepcopy(X.PROCESSING), stitching=MERGED,
                            assembly={"assignment": assignment}, export=dict(export, images=["reference_merged"]),
                            **({} if assignment is None else _cells(tmp_path)))
    manifest = _planned(tmp_path, case, config, **{k: str(v) for k, v in _labels(tmp_path).items()})
    (task,) = _kinds(manifest, "export")
    assert task["outputs"] == ["export/sample/export.json"] + (["export/sample/previews/"] if declared else [])


@pytest.mark.parametrize("name, previews", [("default", None), ("nuclei", True)])
def test_the_assign_task_declares_the_saved_folder_alone(tmp_path, name, previews):
    """From test_stitching_workflow.py, test_wf3_reads_assignment_declares_the_saved_folder_alone: on the route of a
    saved assignment the assign task declares the saved folder's files (and with assignment.previews the folder
    fovs/<fov>/previews/), never the legacy expr/ files; without an assignment (molecules only) no assign task is
    planned (the reads_assignment rule's molecules-only files are retired with its legacy route, D6)."""
    assignment = {"name": name, **({} if previews is None else {"previews": previews})}
    config = W.dataset_keys(tmp_path, processing=copy.deepcopy(X.PROCESSING), stitching=MERGED,
                            assembly={"assignment": name}, export={"route": "per_fov", "images": ["reference_merged"]},
                            **_cells(tmp_path, **assignment))
    manifest = _planned(tmp_path, name, config, **{k: str(v) for k, v in _labels(tmp_path).items()})
    for task in _kinds(manifest, "assign"):
        fov = task["identity"]["fov"]
        folder = f"fovs/{fov}/assignment/{name}"
        assert task["outputs"] == sorted([f"{folder}/{n}" for n in ("assignment.json", "cells.csv", "counts.csv",
                                                                     "molecules.csv")]
                                         + ([f"fovs/{fov}/previews/"] if previews else []))
    molecules = dict(copy.deepcopy(config), output_id="molecules", assembly={"assignment": None})
    molecules.pop("assignment")
    molecules.pop("segmentation")
    molecules["export"] = {"images": ["reference_merged"]}
    assert _kinds(_planned(tmp_path, "molecules", molecules), "assign") == []


@pytest.mark.parametrize("case", ["assembly", "stitching", "processing"])
def test_the_report_task_declares_its_inputs(tmp_path, case):
    """From test_report_workflow.py, test_wf1_report_sample_declares_the_last_per_sample_output and
    test_wf1_report_smk_declares_exactly_these: the report task of each sample declares its report.json, and reads
    the sample's export.json with an assembly block, else its placement.json with a stitching block, and the FOVs'
    run records (and stage records) in every case (the old rule declared the last per-sample output alone: the task
    declares every file the report reads, D2); the compare task reads every sample's report.json, in fovs.csv's
    order, and declares samples.csv and index.html."""
    blocks = {"assembly": dict(stitching=MERGED, assembly={"assignment": "default"},
                               export={"route": "per_fov", "fuse": False}, **_cells(tmp_path)),
              "stitching": dict(stitching=MERGED), "processing": {}}[case]
    config = W.dataset_keys(tmp_path, processing=copy.deepcopy(X.PROCESSING), report={}, **blocks)
    manifest = _planned(tmp_path, case, config, fovs="sample,fov_id\ns2,F1\ns1,F2\n",
                        **{k: str(v) for k, v in _labels(tmp_path).items()})
    reports = _kinds(manifest, "report")
    assert [t["outputs"] for t in reports] == [["report/s2/report.json"], ["report/s1/report.json"]]
    for task in reports:
        sample, fov = task["identity"]["sample"], {"s2": "F1", "s1": "F2"}[task["identity"]["sample"]]
        last = {"assembly": f"export/{sample}/export.json", "stitching": f"stitching/{sample}/placement.json",
                "processing": None}[case]
        assert f"fovs/{fov}/run.json" in task["inputs"]
        if last is not None:
            assert last in task["inputs"]
        assert not [i for i in task["inputs"] if ("export/" in i or "stitching/" in i) and i != last]
    (compare,) = _kinds(manifest, "compare")
    assert compare["inputs"] == ["report/s1/report.json", "report/s2/report.json"]
    assert compare["needs"] == ["report/s1", "report/s2"]
    assert compare["outputs"] == ["report/index.html", "report/samples.csv"]


@pytest.mark.parametrize("case", ["direct", "tiled", "morphology"])
def test_the_fov_kinds_declare_their_run_records_and_no_log(tmp_path, case):
    """From test_run_record_logs.py, test_the_final_targets_branch_on_the_backend (its Python cases): the fov task
    declares the FOV folder's run.json, its reference image and its molecule table; in processing.mode tiled the
    tile_merge task declares run.json and the molecule table (the subtile rules' records are retired, D5); the
    morphology task its registration record; no task declares a log/ file (its record and log are under workflow/,
    D8)."""
    processing = copy.deepcopy(X.PROCESSING)
    if case == "tiled":
        processing.update(mode="tiled", tiled={"tile_size_yx": 18})
    keys = (dict(processing=processing) if case != "morphology" else dict(
        reference_stains=[{"channel": "ch04", "name": "DAPI"}], morphology={},
        additional_round=[{"round_name": "morph", "channel_order": [{"channel": "ch00", "name": "DAPI"}]}]))
    config = W.dataset_keys(tmp_path, **keys)
    manifest = _planned_tiled(tmp_path, config) if case == "tiled" else _planned(tmp_path, case, config)
    kind = {"direct": "fov", "tiled": "tile_merge", "morphology": "morphology"}[case]
    for task in _kinds(manifest, kind):
        fov = task["identity"]["fov"]
        expected = {"direct": {f"fovs/{fov}/run.json", f"fovs/{fov}/reference.tif", f"fovs/{fov}/molecules.csv"},
                    "tiled": {f"fovs/{fov}/run.json", f"fovs/{fov}/molecules.csv"},
                    "morphology": {f"fovs/{fov}/nuclei_registration.json"}}[case]
        assert expected <= set(task["outputs"]), task["id"]
    assert not [o for t in manifest["tasks"] for o in t["outputs"] if o.startswith("log/") or "/log/" in o]


def _planned_tiled(work, config):
    """The tiled plan reads each FOV's image header: write the raw images as real one-plane TIFFs."""
    import numpy as np
    import tifffile
    W.write_inputs(config, TWO)
    for path in W.input_root(config).rglob("*.tif"):
        tifffile.imwrite(path, np.zeros((2, 36, 36), np.uint16))
    return W.plan(work, "tiled", config)[1]


def test_the_morphology_task_declares_its_rounds_files_and_no_json_copy(tmp_path):
    """From test_registration_other_rounds.py, test_the_python_workflow_rule_mirrors_the_matlab_outputs (its rule
    text): the morphology task declares the channel files of its rounds and the reference stain file (no JSON copy of
    the configuration, which only the MATLAB script reads), and its registration record, the saved forms and, with
    morphology.channel_images, the channel images, no log; MATLAB's nuclei_registration rule keeps the JSON copy,
    its log and its shift table."""
    config = W.dataset_keys(tmp_path, reference_stains=[{"channel": "ch04", "name": "DAPI"}],
                            additional_round=[{"round_name": "morph", "channel_order": [
                                {"channel": "ch00", "name": "DAPI"}, {"channel": "ch01", "name": "Nissl"}]}],
                            morphology={"channel_images": {"projection": None}})
    manifest = _planned(tmp_path, "morph", config, fovs="sample,fov_id\nsample,F1\n")
    (task,) = _kinds(manifest, "morphology")
    assert task["inputs"] == ["input:morph/F1/F1_ch00.tif", "input:morph/F1/F1_ch01.tif",
                              "input:round1/F1/F1_ch04.tif"]
    assert task["outputs"] == sorted(["fovs/F1/nuclei_registration.json", "images/morph/ch00/F1.tif",
                                      "images/morph/ch01/F1.tif"]
                                     + [f"fovs/F1/other_rounds/{n}/{f}" for n in ("morph", "reference_stain")
                                        for f in ("image.ome.tif", "registration.json")])
    matlab = (X.ROOT / "workflow" / "rules" / "registration.smk").read_text()
    block = matlab[matlab.index("rule nuclei_registration:"):]
    block = block[:block.index("    resources:")]
    inputs, outputs = block.split("    output:\n")
    assert inputs.endswith("    input:\n        config['config_path'].replace('.yaml', '.json'),\n"
                           '        expand("{input_dir}/{rounds}/{{fovID}}", input_dir=INPUT_DIR, '
                           "rounds=additional_round_names),\n")
    assert "/log/" in outputs and "matlab_script_name = 'nuclei_registration'" in matlab
