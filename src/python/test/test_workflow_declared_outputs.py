"""C3 of design/validation/orchestration.md: the declared outputs of every task kind.

For each worked-example variant and each task, the declared outputs equal the contract's table
(design/specs/orchestration.md, "Declared outputs", with the tiled kinds' files of design/specs/orchestration-tiled.md,
"Declared files of the tiled task kinds") resolved through ``_layout``; and every task's outputs, record and log equal
its group's templates filled with its identity, the chunked checkpoint its group's one temporary output. The rows of
the morphology, segment and assign kinds (W-736) are checked on the variants and on configurations built here: the
channel images with ``morphology.channel_images``, the displacement field of a dense ``morphology.recipe``, the nucleus
table and the territories of an assignment with ``nuclei`` and ``expansion``, its previews with
``assignment.previews``.
"""
import os

import pytest

from starfinder import _layout
from starfinder.workflow import plan

from . import workflow_examples as X

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

