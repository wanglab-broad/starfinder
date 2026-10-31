"""C3 of design/validation/orchestration.md for the kinds of W-647: the declared outputs of every route.

For each worked-example variant and each task of the kinds summarize, merge, fov, bigstitcher_macro,
bigstitcher_engine, bigstitcher_import, stitch, export, report and compare, the declared outputs equal the contract's
table (design/specs/orchestration.md, "Declared outputs") resolved through ``_layout``; and every task's outputs,
record and log equal its group's templates filled with its identity. The tiled kinds (W-735) and the morphology,
segment and assign kinds (W-736) add their rows with their issues.
"""
import os

import pytest

from starfinder import _layout
from starfinder.workflow import plan

from . import workflow_examples as X

pytestmark = [pytest.mark.workflow, pytest.mark.contract]

KINDS = ("summarize", "merge", "fov", "bigstitcher_macro", "bigstitcher_engine", "bigstitcher_import", "stitch",
         "export", "report", "compare")


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
    """The contract's declared outputs of a task (its table's rows for the kinds of W-647), through _layout."""
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
        assert group["temporary"] == []
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
