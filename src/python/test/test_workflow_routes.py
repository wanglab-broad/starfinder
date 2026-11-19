"""Both export routes in one output root (W-876).

Each export route writes its sample outputs into folders of its own, ``export/<sample>/<route>/`` and
``report/<sample>/<route>/``, with its own task records (``export/<sample>/<route>``, ``report/<sample>/<route>``),
saved export record and held page; the routes share every FOV-level task and the placement. The comparison page
lists the reports of every route the output root holds. The fast tests plan the two routes of one configuration on
one output root and run the export, report and compare commands with ``Dataset.export``, ``Dataset.report`` and
``Dataset.compare_samples`` replaced by stand-ins that write their folders; the slow test runs every task of both
routes on W-642's fixture by its command.
"""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from starfinder import _layout
from starfinder.__main__ import main
from starfinder.dataset import Dataset
from starfinder.workflow import plan, verify

from . import workflow_examples as X
from . import workflow_tasks as T

pytestmark = [pytest.mark.workflow]

ROUTES = ("per_fov", "after_fusion")
MERGED = {"method": "multiview_stitcher", "input": "reference_merged",
          "positions": {"source": "regular_grid", "regular_grid": {"columns": 2, "rows": 1,
                                                                   "order": "column_major_down_right",
                                                                   "overlap": 0.25}}}


def _tree(out, route, sample="s1"):
    """path → SHA-256 of every file of a route under the output root: its export and report folders, its saved export
    record and held page, and its tasks' records and logs."""
    folders = [out / "export" / sample / route, out / "report" / sample / route]
    files = [p for folder in folders if folder.is_dir() for p in folder.rglob("*") if p.is_file()]
    files += [_layout.workflow_export_record(out, sample, route), _layout.workflow_held_page(out, sample, route)]
    for kind in ("export", "report"):
        files += [_layout.task_record(out, f"{kind}/{sample}/{route}"), _layout.task_log(out, f"{kind}/{sample}/{route}")]
    return {p.relative_to(out).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(set(files)) if p.is_file()}


def _config(work, route):
    """The processing, segmentation (one imported cell run), assignment, stitching, assembly, export and report blocks
    of sample s1 (F1, F2); the route after fusion imports one label image on the fused grid."""
    labels = Path(work) / "labels"
    config = T.dataset_keys(work, output_id="routes", processing=copy.deepcopy(X.PROCESSING), stitching=MERGED,
                            segmentation={"runs": [{"name": "cell", "target": "cell", "method": "import",
                                                    "path": str(labels / "{fov_id}.tif")}]},
                            assignment={"cells": "cell"}, assembly={"assignment": "default"}, report={})
    config["export"] = ({"route": "per_fov", "fuse": False} if route == "per_fov" else {
        "route": "after_fusion", "images": ["reference_merged"], "segmentation": {"cells": "cell", "runs": [
            {"name": "cell", "target": "cell", "method": "import", "path": str(labels / "fused.tif")}]}})
    return config


def _plan(work, route):
    """The route's plan; the inputs are written once, so that both routes' plans list them with the same times."""
    config = _config(work, route)
    labels = Path(work) / "labels"
    if not labels.is_dir():
        labels.mkdir(parents=True)
        for name in ("F1.tif", "F2.tif", "fused.tif"):
            (labels / name).touch()
        T.write_inputs(config, "sample,fov_id\ns1,F1\ns1,F2\n")
    manifest, _ = T.plan(work, route, config)
    return manifest


def _complete(manifest, task_id):
    """A record of the task that verify finds complete, over stand-ins of the declared outputs it lacks."""
    from starfinder._records import file_sha256
    data = json.loads(Path(manifest).read_text())
    out = Path(data["roots"]["output"])
    entry = next(t for t in data["tasks"] if t["id"] == task_id)
    for declared in entry["outputs"]:
        path = out / declared
        if declared.endswith("/"):
            path.mkdir(parents=True, exist_ok=True)
        elif not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"stand-in of {task_id}\n")
    (out / entry["log"]).parent.mkdir(parents=True, exist_ok=True)
    (out / entry["log"]).write_text("")
    outputs = [{"path": o, "bytes": (out / o).stat().st_size, "sha256": "sha256:" + file_sha256(out / o)}
               for o in [*entry["outputs"], entry["log"]] if not o.endswith("/")]
    outputs += [{"path": o, "files": []} for o in entry["outputs"] if o.endswith("/")]
    _layout.task_record(out, task_id).parent.mkdir(parents=True, exist_ok=True)
    _layout.task_record(out, task_id).write_text(json.dumps({
        "id": task_id, "status": "succeeded", "outputs": outputs, "workflow": {"fingerprint": entry["fingerprint"]}}))


@pytest.fixture
def stand_ins(monkeypatch):
    """Dataset.export, report and compare_samples as stand-ins that write their folders: export.json and a table
    naming the route, report.json and report.html, and index.html and samples.csv with one row per sample and route
    (the calls are recorded)."""
    calls = []

    def export(self, sample, *, output_dir, config, **kwargs):
        calls.append(("export", sample, Path(output_dir)))
        (Path(output_dir) / "tables").mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "export.json").write_text(json.dumps({"sample": sample, "route": config.route}) + "\n")
        (Path(output_dir) / "tables" / "cells.csv").write_text(f"route\n{config.route}\n")

    def report(self, sample, fov_ids, *, output_dir, export_dir, **kwargs):
        calls.append(("report", sample, Path(output_dir), export_dir))
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "report.json").write_text(json.dumps({"export": str(export_dir)}) + "\n")
        (Path(output_dir) / "report.html").write_text(f"report of {export_dir}\n")

    def compare_samples(self, samples, *, report_root, routes):
        calls.append(("compare", list(samples), list(routes)))
        rows = [f"{s},{r}" for s in samples for r in routes if (Path(report_root) / s / r / "report.json").is_file()]
        (Path(report_root) / "index.html").write_text("\n".join(rows) + "\n")
        (Path(report_root) / "samples.csv").write_text("sample,route\n" + "\n".join(rows) + "\n")

    monkeypatch.setattr(Dataset, "export", export)
    monkeypatch.setattr(Dataset, "report", report)
    monkeypatch.setattr(Dataset, "compare_samples", compare_samples)
    return calls


def _run_route(manifest):
    """The route's FOV-level and placement tasks complete (stand-ins), then its export, report and compare commands;
    the ids that ran."""
    data = json.loads(Path(manifest).read_text())
    done = verify(manifest)
    for task in data["tasks"]:
        if task["kind"] not in ("export", "report", "compare") and not done["tasks"][task["id"]]["complete"]:
            _complete(manifest, task["id"])
    ran = []
    for task in data["tasks"]:
        if not verify(manifest, task=task["id"])["tasks"][task["id"]]["complete"]:
            assert T.run(manifest, task["id"]) == 0, task["id"]
            ran.append(task["id"])
    return ran


@pytest.mark.contract
def test_the_two_routes_plan_their_own_tasks_and_folders(tmp_path):
    """The two routes of one configuration planned on one output root: the export and report tasks have the route in
    their identity, id, record and every declared output, so that no declared path of one route is another's; every
    other task, the compare task and its outputs aside, has the same id, outputs and fingerprint in both plans."""
    manifests = {route: json.loads(_plan(tmp_path, route).read_text()) for route in ROUTES}
    owned = {}
    for route, manifest in manifests.items():
        assert manifest["export_route"] == route
        for kind in ("export", "report"):
            (task,) = [t for t in manifest["tasks"] if t["kind"] == kind]
            assert task["identity"] == {"sample": "s1", "route": route}
            assert task["id"] == f"{kind}/s1/{route}"
            assert task["record"] == f"workflow/tasks/{kind}.s1.{route}.json"
            assert task["outputs"] == [f"{kind}/s1/{route}/{kind}.json"]
            owned[route] = owned.get(route, set()) | set(task["outputs"]) | {task["record"], task["log"]}
    assert not owned["per_fov"] & owned["after_fusion"]
    shared = [{t["id"]: (t["outputs"], t["fingerprint"]) for t in m["tasks"] if t["kind"] not in
               ("export", "report", "compare")} for m in manifests.values()]
    assert shared[0] == shared[1] and {i.split("/")[0] for i in shared[0]} == {"fov", "segment", "assign", "stitch"}


@pytest.mark.contract
def test_both_routes_in_one_output_root(tmp_path, stand_ins):
    """The route per FOV, then the route after fusion planned on the same output root: the second runs only its
    export, report and compare tasks (its FOV-level and placement tasks are the first's, complete), each route's
    export and report read and write their own folders, the second's compare lists both routes' reports, and verify
    --final passes after each. The manifest and verify list one route with one route in the output root, and both
    after the second: the second plan's own route and the route per FOV found in the output root, which is no task
    of the second plan (no id, need or fingerprint of it) and which verify checks by its records."""
    first = _plan(tmp_path, "per_fov")
    assert json.loads(first.read_text())["routes"] == {
        "s1": [{"route": "per_fov", "planned": True, "tasks": ["export/s1/per_fov", "report/s1/per_fov"]}]}
    assert _run_route(first) == ["export/s1/per_fov", "report/s1/per_fov", "compare"]
    assert main(["workflow", "verify", "--manifest", str(first), "--final", "--hash"]) == 0
    one = verify(first, hash=True, final=True)
    assert one["passed"] and one["found_incomplete"] == [] and [(r["route"], r["planned"], r["complete"])
                                                                for r in one["routes"]["s1"]] == [("per_fov", True, True)]
    out = Path(json.loads(first.read_text())["roots"]["output"])
    second = _plan(tmp_path, "after_fusion")
    assert T.task(second, "compare")["inputs"] == ["report/s1/after_fusion/report.json",
                                                   "report/s1/per_fov/report.json"]
    planned = json.loads(second.read_text())
    assert planned["routes"] == {"s1": [
        {"route": "per_fov", "planned": False, "tasks": [
            {"id": f"{kind}/s1/per_fov", "kind": kind, "outputs": [f"{kind}/s1/per_fov/{kind}.json"],
             "record": f"workflow/tasks/{kind}.s1.per_fov.json", "log": f"workflow/logs/{kind}.s1.per_fov.log"}
            for kind in ("export", "report")]},
        {"route": "after_fusion", "planned": True, "tasks": ["export/s1/after_fusion", "report/s1/after_fusion"]}]}
    named = [t["id"] for t in planned["tasks"]] + [n for t in planned["tasks"] for n in t["needs"]]
    assert not [i for i in named if i.endswith("/per_fov")]
    stand_ins.clear()
    assert _run_route(second) == ["export/s1/after_fusion", "report/s1/after_fusion", "compare"]
    assert main(["workflow", "verify", "--manifest", str(second), "--final", "--hash"]) == 0
    both = verify(second, hash=True, final=True)
    assert both["passed"] and both["found_incomplete"] == [] and "export/s1/per_fov" not in both["tasks"]
    assert [(r["route"], r["planned"], r["complete"], sorted(r["tasks"])) for r in both["routes"]["s1"]] == [
        ("per_fov", False, True, ["export/s1/per_fov", "report/s1/per_fov"]),
        ("after_fusion", True, True, ["export/s1/after_fusion", "report/s1/after_fusion"])]
    folder = out / "export" / "s1" / "after_fusion"
    assert stand_ins == [("export", "s1", folder),
                         ("report", "s1", out / "report" / "s1" / "after_fusion", folder),
                         ("compare", ["s1"], ["per_fov", "after_fusion"])]
    assert sorted(p.name for p in (out / "export" / "s1").iterdir()) == ["after_fusion", "per_fov"]
    assert sorted(p.name for p in (out / "report" / "s1").iterdir()) == ["after_fusion", "per_fov"]
    assert (out / "report" / "samples.csv").read_text() == "sample,route\ns1,per_fov\ns1,after_fusion\n"
    for route in ROUTES:
        assert json.loads((out / "export" / "s1" / route / "export.json").read_text())["route"] == route
        assert _layout.task_record(out, f"export/s1/{route}").is_file()
        assert _layout.workflow_export_record(out, "s1", route).is_file()


@pytest.mark.contract
@pytest.mark.parametrize("rerun", ROUTES)
def test_a_rerun_of_one_route_leaves_the_other_routes_files_byte_identical(tmp_path, stand_ins, rerun):
    """With both routes in one output root, the export, report and compare commands of one route run again (planned
    again, and run by their commands whether complete or not): every file of the other route (its export and report
    folders, its saved export record and its tasks' records and logs) keeps its bytes, the rerun route's export and
    report are rewritten, and verify --final passes on the rerun route's plan."""
    manifests = {}
    for route in ROUTES:
        manifests[route] = _plan(tmp_path, route)
        _run_route(manifests[route])
    out = Path(json.loads(manifests[rerun].read_text())["roots"]["output"])
    other = next(r for r in ROUTES if r != rerun)
    before = _tree(out, other)
    assert before and any(p.startswith(f"export/s1/{other}/") for p in before)
    manifest = _plan(tmp_path, rerun)
    record = _layout.task_record(out, f"export/s1/{rerun}").read_bytes()
    for task_id in (f"export/s1/{rerun}", f"report/s1/{rerun}", "compare"):
        assert T.run(manifest, task_id) == 0, task_id
    assert _layout.task_record(out, f"export/s1/{rerun}").read_bytes() != record
    assert _tree(out, other) == before
    assert main(["workflow", "verify", "--manifest", str(manifest), "--final", "--hash"]) == 0
    assert (out / "report" / "samples.csv").read_text() == "sample,route\ns1,per_fov\ns1,after_fusion\n"


@pytest.mark.contract
@pytest.mark.parametrize("damage", ["report byte", "export byte", "report record"])
def test_verify_checks_the_found_route_by_its_records(tmp_path, stand_ins, capsys, damage):
    """With both routes in one output root, verify of the second plan checks the route found there (the route per FOV)
    by its records, without their fingerprint: one byte changed in a file its report or export record lists makes
    verify --hash --final fail and names that file (the size alone, without --hash, does not see it), and so does its
    report task's missing record; this plan's own tasks stay complete."""
    for route in ROUTES:
        manifest = _plan(tmp_path, route)
        _run_route(manifest)
    out = Path(json.loads(manifest.read_text())["roots"]["output"])
    assert verify(manifest, hash=True, final=True)["passed"]
    kind = "export" if damage == "export byte" else "report"
    task_id, path = f"{kind}/s1/per_fov", out / kind / "s1" / "per_fov" / f"{kind}.json"
    if damage == "report record":
        _layout.task_record(out, task_id).unlink()
        expected = ["no task record"]
    else:
        data = bytearray(path.read_bytes())
        data[0] ^= 1
        path.write_bytes(data)
        expected = [f"{kind}/s1/per_fov/{kind}.json has another SHA-256 than the record's"]
        assert verify(manifest, final=True)["passed"]                  # the same size: only the hash finds it
    result = verify(manifest, hash=True, final=True)
    assert not result["passed"] and result["incomplete"] == [] and result["found_incomplete"] == [task_id]
    (found,) = [r for r in result["routes"]["s1"] if not r["planned"]]
    assert found["route"] == "per_fov" and not found["complete"] and found["tasks"][task_id]["problems"] == expected
    capsys.readouterr()
    assert main(["workflow", "verify", "--manifest", str(manifest), "--final", "--hash"]) == 1
    printed = capsys.readouterr()
    assert f"found route s1/per_fov: {task_id}\tnot complete: {expected[0]}" in printed.err
    assert "per_fov" not in printed.out                                     # stdout: one line per task of this plan


@pytest.mark.contract
def test_the_export_of_one_route_removes_its_own_held_page_alone(tmp_path, stand_ins):
    """The saved export record and the held page are per route: the export of one route removes its own held page
    when it runs, never another route's, and saves its own record under workflow/exports/<sample>/<route>.json."""
    manifest = _plan(tmp_path, "per_fov")
    out = Path(json.loads(manifest.read_text())["roots"]["output"])
    pages = {route: _layout.workflow_held_page(out, "s1", route) for route in ROUTES}
    assert pages["per_fov"] == out / "workflow" / "held" / "s1" / "per_fov.html"
    for page in pages.values():
        page.parent.mkdir(parents=True, exist_ok=True)
        page.write_text("held\n")
    T.call(manifest, "export/s1/per_fov")
    assert not pages["per_fov"].exists() and pages["after_fusion"].read_text() == "held\n"
    saved = out / "workflow" / "exports" / "s1" / "per_fov.json"
    assert saved.read_bytes() == (out / "export" / "s1" / "per_fov" / "export.json").read_bytes()
    assert not _layout.workflow_export_record(out, "s1", "after_fusion").exists()


# --- the two routes end to end on W-642's fixture -------------------------------------------------------------------

def _routes_config(work, route):
    """The per-fov variant (four FOVs, one sample) in the output root ``routes``, the export fusing the reference
    images and, on the route per FOV, the cell labels at the fusion's scale; the route after fusion imports the
    per-FOV route's fused cell labels (their Z maximum on the fused projection's grid, as the golden invocation D
    does)."""
    config = copy.deepcopy(X.variants(work)["per-fov"])
    config["output_id"] = "routes"
    if route == "per_fov":
        config["export"] = dict(config["export"], fuse_labels=True, labels_scale_yx=0.5, previews=False)
    else:
        config["export"] = {"route": "after_fusion", "images": ["reference_merged", "reference_stain"],
                            "segmentation": {"cells": "cell", "runs": [{
                                "name": "cell", "target": "cell", "method": "import",
                                "path": str(Path(work) / "labels" / "fused-cells.ome.tif")}]}}
    return config


@pytest.mark.slow
@pytest.mark.integration
def test_both_routes_run_end_to_end_in_one_output_root(tmp_path):
    """W-642's fixture through the route per FOV and then the route after fusion into one output root, every task
    by its command and no copy step: the second plan reruns only its export, report and compare tasks, verify
    --final passes with both and lists both routes (this plan's and the one found in the output root), the comparison
    page lists both routes' reports, and the route per FOV run again (its export, report and compare commands) leaves
    every file of the route after fusion byte-identical. One byte changed in the found route's report makes verify
    --hash --final fail, naming the file; restored, verify passes again."""
    if importlib.util.find_spec("multiview_stitcher") is None or importlib.util.find_spec("anndata") is None:
        pytest.skip("the stitching and the export need the multiview-stitcher and anndata extras")
    work = tmp_path / "work"
    X.golden().write_fixture(work)
    X.write_supplied(work)
    first = plan(X.write_config(work, "per_fov", _routes_config(work, "per_fov")))
    path = Path(first["roots"]["output"]) / "workflow" / "manifest.json"
    out = path.parents[1]
    for task in first["tasks"]:
        assert X.run_task(path, task["id"]) == 0, task["id"]
    one = verify(path, hash=True, final=True)
    assert one["passed"] and [(r["route"], r["planned"]) for r in one["routes"][X.SAMPLE]] == [("per_fov", True)]
    golden = X.golden()
    golden._fused_cells(out / "export" / X.SAMPLE / "per_fov", work / "labels" / "fused-cells.ome.tif")
    per_fov = _tree(out, "per_fov", X.SAMPLE)
    second = plan(X.write_config(work, "after_fusion", _routes_config(work, "after_fusion")))
    stale = verify(path)
    assert set(stale["incomplete"]) == {f"export/{X.SAMPLE}/after_fusion", f"report/{X.SAMPLE}/after_fusion",
                                        "compare"}
    for task in second["tasks"]:
        if task["id"] in stale["incomplete"]:
            assert X.run_task(path, task["id"]) == 0, task["id"]
    assert main(["workflow", "verify", "--manifest", str(path), "--final", "--hash"]) == 0
    _both_routes_verified(path, "after_fusion")
    assert _tree(out, "per_fov", X.SAMPLE) == per_fov
    samples = pd.read_csv(out / "report" / "samples.csv")
    assert list(zip(samples["sample"], samples["route"], samples["report"])) == [
        (X.SAMPLE, "per_fov", "present"), (X.SAMPLE, "after_fusion", "present")]
    page = (out / "report" / "index.html").read_text(encoding="utf-8")
    for route in ROUTES:
        assert f'<a href="{X.SAMPLE}/{route}/report.html">{X.SAMPLE}</a>' in page
        record = json.loads((out / "report" / X.SAMPLE / route / "report.json").read_text())
        assert record["route"] == route
        assert {e["path"] for e in record["inputs"] if e["path"].startswith("export/")} <= {
            p for p in _tree(out, route, X.SAMPLE) if p.startswith(f"export/{X.SAMPLE}/{route}/")}
    # The route per FOV again: its export, report and compare commands, whether complete or not.
    after_fusion = _tree(out, "after_fusion", X.SAMPLE)
    again = plan(X.write_config(work, "per_fov", _routes_config(work, "per_fov")))
    assert verify(path)["incomplete"] == ["compare"]
    for task_id in (f"export/{X.SAMPLE}/per_fov", f"report/{X.SAMPLE}/per_fov", "compare"):
        assert X.run_task(path, task_id) == 0, task_id
    assert [t["id"] for t in again["tasks"]] == [t["id"] for t in first["tasks"]]
    assert main(["workflow", "verify", "--manifest", str(path), "--final", "--hash"]) == 0
    _both_routes_verified(path, "per_fov")
    assert _tree(out, "after_fusion", X.SAMPLE) == after_fusion
    report = out / "report" / X.SAMPLE / "after_fusion" / "report.json"
    data = report.read_bytes()
    report.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
    broken = verify(path, hash=True, final=True)
    assert not broken["passed"] and broken["incomplete"] == []
    assert broken["found_incomplete"] == [f"report/{X.SAMPLE}/after_fusion"]
    (found,) = [r for r in broken["routes"][X.SAMPLE] if not r["planned"]]
    assert found["tasks"][f"report/{X.SAMPLE}/after_fusion"]["problems"] == [
        f"report/{X.SAMPLE}/after_fusion/report.json has another SHA-256 than the record's"]
    report.write_bytes(data)
    assert main(["workflow", "verify", "--manifest", str(path), "--final", "--hash"]) == 0


def _both_routes_verified(manifest, planned):
    """verify --hash --final of the plan of route ``planned`` with both routes in the output root: it passes and lists
    both routes of the sample, the other found in the output root (its export and report checked by their records);
    the manifest names both, the found route's tasks being none of the plan's."""
    data = json.loads(Path(manifest).read_text())
    other = next(r for r in ROUTES if r != planned)
    found = [f"export/{X.SAMPLE}/{other}", f"report/{X.SAMPLE}/{other}"]
    assert [(r["route"], r["planned"]) for r in data["routes"][X.SAMPLE]] == [
        ("per_fov", planned == "per_fov"), ("after_fusion", planned == "after_fusion")]
    assert not set(found) & {t["id"] for t in data["tasks"]}
    result = verify(manifest, hash=True, final=True)
    assert result["passed"] and result["found_incomplete"] == [] and result["incomplete"] == []
    listed = {r["route"]: r for r in result["routes"][X.SAMPLE]}
    assert sorted(listed) == sorted(ROUTES) and all(r["complete"] for r in listed.values())
    assert sorted(listed[other]["tasks"]) == found and not listed[other]["planned"]
    return result


# --- the two routes under each engine (extended) --------------------------------------------------------------------

def _engine_route(run, route):
    """The engine run's configuration (the per-fov variant with small resources) on one route, as _routes_config."""
    config = copy.deepcopy(run.config)
    config["export"] = _routes_config(run.work, route)["export"]
    return config


def _engine_routes(tmp_path, engine):
    """W-642's fixture through the route per FOV and then the route after fusion into one output root by the engine,
    no copy step between them: the second invocation runs only the second route's export, report and compare, every
    file of the first route keeps its bytes, and verify --final passes after each (EngineRun.check, which also finds
    the route's link on the comparison page), listing both routes once both are in the output root. The first route
    then runs again: only compare runs, the second route's files keep their bytes, and the page lists both routes.
    Neither engine ever runs the route found in the output root."""
    from . import workflow_engine_validation as E
    run = E.EngineRun(tmp_path / f"routes-{engine}", engine)
    run.replan(_engine_route(run, "per_fov"))
    run.invoke({t["id"] for t in run.data["tasks"]})
    run.check()
    X.golden()._fused_cells(run.out / "export" / X.SAMPLE / "per_fov", run.work / "labels" / "fused-cells.ome.tif")
    trees = {"per_fov": _tree(run.out, "per_fov", X.SAMPLE)}
    run.replan(_engine_route(run, "after_fusion"))
    ran = run.invoke({f"export/{X.SAMPLE}/after_fusion", f"report/{X.SAMPLE}/after_fusion", "compare"})
    run.check()
    _both_routes_verified(run.manifest, "after_fusion")
    assert _tree(run.out, "per_fov", X.SAMPLE) == trees["per_fov"]
    trees["after_fusion"] = _tree(run.out, "after_fusion", X.SAMPLE)
    run.replan(_engine_route(run, "per_fov"))
    again = run.invoke({"compare"})
    run.check()
    _both_routes_verified(run.manifest, "per_fov")
    # The engines never ran the found route: neither invocation's own list of the tasks it ran names it, and its
    # records keep their bytes (the trees hold them).
    assert not [t for t in ran if t.endswith("/per_fov")] and not [t for t in again if t.endswith("/after_fusion")]
    assert _tree(run.out, "after_fusion", X.SAMPLE) == trees["after_fusion"]
    assert _tree(run.out, "per_fov", X.SAMPLE) == trees["per_fov"]
    samples = pd.read_csv(run.out / "report" / "samples.csv")
    assert list(zip(samples["route"], samples["report"])) == [("per_fov", "present"), ("after_fusion", "present")]
    E.evidence(run.work, "routes.json", {"engine": engine, "second_route_ran": sorted(ran),
                                         "first_route_again_ran": sorted(again), "trees": trees,
                                         "samples": samples.astype(object).where(samples.notna(), None)
                                         .to_dict("records")})


@pytest.mark.slow
@pytest.mark.extended
@pytest.mark.integration
def test_both_routes_in_one_output_root_under_snakemake(tmp_path):
    _engine_routes(tmp_path, "snakemake")


@pytest.mark.slow
@pytest.mark.extended
@pytest.mark.integration
@pytest.mark.nextflow
def test_both_routes_in_one_output_root_under_nextflow(tmp_path):
    _engine_routes(tmp_path, "nextflow")
