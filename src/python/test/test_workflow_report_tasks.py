"""The report and compare commands (moved by W-750 from the tests of the report_sample and report_dataset adapters in
test_report_workflow.py, WF1 of design/specs/report-algorithms.md).

Each test plans a configuration in the Python shape with a ``report`` block on an input root whose raw images are
empty files (``workflow_tasks``); the report's function is called on its task's ``Run`` with ``Dataset.report`` and
``Dataset.compare_samples`` spied, as the adapter tests spied them, or runs for real on ``report_records`` of
``report_fixtures.py`` written into the plan's output root. An invalid report block is refused by the plan before any
task.
"""
import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from starfinder.dataset import Dataset
from starfinder.report import ReportConfig, load_report
from starfinder.workflow import plan
from starfinder.workflow._plan import PlanRefused

from . import workflow_examples as X
from . import workflow_tasks as T
from .test_report_workflow import REPORT, REPORT_CONFIG, SAMPLE

pytestmark = [pytest.mark.workflow, pytest.mark.contract]

MERGED = {"method": "multiview_stitcher", "input": "reference_merged",
          "positions": {"source": "regular_grid", "regular_grid": {"columns": 2, "rows": 2,
                                                                   "order": "column_major_down_right",
                                                                   "overlap": 0.25}}}
SEGMENT = {"runs": [{"name": "cell", "target": "cell", "method": "import", "path": "{labels}"}]}


def spy(monkeypatch, name):
    calls = []
    monkeypatch.setattr(Dataset, name, lambda self, *args, **kwargs: calls.append((self, args, kwargs)))
    return calls


def membership(rows):
    """fovs.csv: the FOVs Position<i:03d> of each (sample, start, end)."""
    return "sample,fov_id\n" + "".join(f"{s},Position{i:03d}\n" for s, a, b in rows for i in range(a, b + 1))


def report_config(work, report, blocks):
    """The processing block, the report block and the blocks of a case; an assignment name of the assembly block gets
    the assignment block and the segment run it reads (the route per FOV)."""
    config = T.dataset_keys(work, processing=copy.deepcopy(X.PROCESSING), report=report, **copy.deepcopy(blocks))
    assembly = config.get("assembly")
    if assembly is not None:
        config["export"] = {"fuse": False}           # the export's fusion would need the morphology block
    if assembly is not None and assembly.get("assignment", "default") is not None:
        labels = work / "labels" / "{fov_id}.tif"
        segmentation = copy.deepcopy(SEGMENT)
        segmentation["runs"][0]["path"] = str(labels)
        config.update(segmentation=segmentation, export={"route": "per_fov", "fuse": False},
                      assignment={"cells": "cell", "name": assembly.get("assignment", "default")})
    return config


#: The cases of test_wf1_report_sample_calls_dataset_report: the blocks and the assignment the report reads.
CASES = [
    ({"stitching": MERGED, "assembly": {"assignment": "default"}}, "default"),
    ({"stitching": MERGED, "assembly": {"assignment": "cells_v2"}}, "cells_v2"),
    ({"stitching": MERGED, "assembly": {"assignment": None}}, None),
    ({"stitching": MERGED, "assembly": {}}, "default"),
    ({"stitching": MERGED}, None),
    ({}, None),
]


@pytest.mark.parametrize("blocks, assignment", CASES)
@pytest.mark.parametrize("block", ["contract", "empty"])
def test_the_report_command_calls_dataset_report(tmp_path, monkeypatch, blocks, assignment, block):
    """From test_wf1_report_sample_calls_dataset_report: the sample's FOVs from fovs.csv, the assembly block's
    assignment (none without the block, where no task saves an assignment) and the report block's configuration."""
    calls = spy(monkeypatch, "report")
    config = report_config(tmp_path, copy.deepcopy(REPORT) if block == "contract" else {}, blocks)
    rows = [("sample0", 1, 2), (SAMPLE, 3, 5), ("sample2", 6, 6)]
    T.write_inputs(config, membership(rows))
    for i in range(1, 7):
        (tmp_path / "labels").mkdir(exist_ok=True)
        (tmp_path / "labels" / f"Position{i:03d}.tif").write_bytes(b"")
    manifest, _ = T.plan(tmp_path, "report", config)
    T.call(manifest, f"report/{SAMPLE}")
    out = T.output_root(config)
    (dataset, args, kwargs), = calls
    assert dataset.output_root == out and dataset.dataset_id == "data"
    assert args == (SAMPLE, ["Position003", "Position004", "Position005"])
    assert kwargs == {"assignment": assignment, "config": REPORT_CONFIG if block == "contract" else ReportConfig(),
                      "output_dir": out / "report" / SAMPLE}


#: The cases of test_the_report_command_names_the_assignment_block_without_an_assembly_block: the assignment block and
#: the name the report reads (W-807).
ASSIGNMENT_ONLY = [({}, "default"), ({"name": "cells_v2"}, "cells_v2")]


def _assignment_only_config(work, assignment, report):
    """report_config of a configuration with segmentation, assignment and report blocks and no stitching, assembly or
    export block (W-649's U4), over empty label files."""
    labels = work / "labels" / "{fov_id}.tif"
    segmentation = copy.deepcopy(SEGMENT)
    segmentation["runs"][0]["path"] = str(labels)
    return report_config(work, report, {"segmentation": segmentation, "assignment": copy.deepcopy(assignment)})


@pytest.mark.parametrize("assignment, name", ASSIGNMENT_ONLY)
def test_the_report_command_names_the_assignment_block_without_an_assembly_block(tmp_path, monkeypatch, assignment,
                                                                                name):
    """W-807: without an assembly block the assign tasks' assignment, the assignment block's name, is the report's
    (it was None, a sample without cells, although the report task waits for each FOV's assign task)."""
    calls = spy(monkeypatch, "report")
    config = _assignment_only_config(tmp_path, assignment, copy.deepcopy(REPORT))
    assert "assembly" not in config and "export" not in config
    T.write_inputs(config, membership([(SAMPLE, 3, 5)]))
    (tmp_path / "labels").mkdir()
    for i in range(3, 6):
        (tmp_path / "labels" / f"Position{i:03d}.tif").write_bytes(b"")
    manifest, _ = T.plan(tmp_path, "assignment-only", config)
    task = T.task(manifest, "report", sample=SAMPLE)
    assert [i for i in task["inputs"] if i.endswith("assignment.json")] == [
        f"fovs/Position{i:03d}/assignment/{name}/assignment.json" for i in range(3, 6)]
    T.call(manifest, f"report/{SAMPLE}")
    (_, args, kwargs), = calls
    assert args == (SAMPLE, ["Position003", "Position004", "Position005"])
    assert kwargs["assignment"] == name


def test_the_report_of_a_run_without_an_assembly_block_shows_its_cells(tmp_path):
    """W-807, on report_fixtures' hand-written records (report_records of F1 to F4 and write_assignment_folder's saved
    assignment and cell run of each FOV, over empty label files, and no export): the real report task names the
    assignment, reports every FOV's cell run and the per-FOV segment and assign_molecules steps as run. The same case
    on records that the tasks themselves write is test_the_report_task_reads_what_the_fov_segment_and_assign_tasks_wrote."""
    from .report_fixtures import DATASET, FOVS, SAMPLE_ID, report_records, write_assignment_folder
    from .report_fixtures import SAMPLE as REPORT_SAMPLE
    config = _assignment_only_config(tmp_path, {}, {})
    config.update(dataset_id=DATASET, sample_id=SAMPLE_ID, output_id="output")
    T.write_inputs(config, "sample,fov_id\n" + "".join(f"{REPORT_SAMPLE},{fov}\n" for fov in FOVS))
    (tmp_path / "labels").mkdir()
    for fov in FOVS:
        (tmp_path / "labels" / f"{fov}.tif").write_bytes(b"")
    manifest, _ = T.plan(tmp_path, "cells", config)
    out = T.output_root(config)
    case = report_records(out)
    for fov in FOVS:
        write_assignment_folder(case, fov)
    T.call(manifest, f"report/{REPORT_SAMPLE}")
    folder = out / "report" / REPORT_SAMPLE
    record = load_report(folder)
    assert record["assignment"]["name"] == "default"
    assert sorted(record["assignment"]["folders"]) == list(FOVS)
    assert record["segmentation"]["runs"] == {fov: {"cells": "cell", "nuclei": None} for fov in FOVS}
    steps = pd.read_csv(folder / "tables" / "steps.csv")
    cell_steps = steps[steps["name"].isin(["segment", "assign_molecules"])]
    assert sorted(zip(cell_steps["fov_id"], cell_steps["name"], cell_steps["status"])) == sorted(
        (fov, name, "succeeded") for fov in FOVS for name in ("segment", "assign_molecules"))


@pytest.fixture(scope="module")
def tiny_sample(tmp_path_factory):
    """The workflow tests' synthetic fixture (W-642's tiny_sample), as test_workflow_cell_tasks.py builds it."""
    work = tmp_path_factory.mktemp("tiny") / "work"
    X.golden().write_fixture(work)
    X.write_supplied(work)
    return work


def test_the_report_task_reads_what_the_fov_segment_and_assign_tasks_wrote(tiny_sample):
    """W-807 on tiny_sample, with no record written by hand: a configuration with processing, segmentation (an import
    run of the fixture's labels), assignment and report blocks and no stitching, assembly or export block, FOV_001 and
    FOV_002. The fov, segment, assign and report tasks all exit 0, and the report names the assignment ``default`` with
    a folder for each FOV, lists the cell run, and has the segment and assign_molecules steps of each FOV succeeded."""
    work = tiny_sample
    run = {"name": "cell", "target": "cell", "method": "import", "path": str(work / "labels" / "{fov_id}.tif")}
    fovs = ["FOV_001", "FOV_002"]
    config = dict(X.dataset(work, "report-without-assembly"), selected_fovs=fovs,
                  processing=copy.deepcopy(X.PROCESSING), segmentation={"runs": [run]},
                  assignment={"cells": "cell"}, report={})
    assert not {"stitching", "assembly", "export"} & set(config)
    manifest = plan(X.write_config(work, "report-without-assembly", config))
    path = Path(manifest["roots"]["output"]) / "workflow" / "manifest.json"
    planned = [t for t in manifest["tasks"] if t["kind"] in ("fov", "segment", "assign", "report")]
    assert sorted({t["kind"] for t in planned}) == ["assign", "fov", "report", "segment"]
    for kind in ("fov", "segment", "assign", "report"):
        for task in planned:
            if task["kind"] == kind:
                assert X.run_task(path, task["id"]) == 0, task["id"]
    folder = Path(manifest["roots"]["output"]) / "report" / X.SAMPLE
    record = load_report(folder)
    assert record["fov_ids"] == fovs
    assert record["assignment"]["name"] == "default"
    assert sorted(record["assignment"]["folders"]) == fovs
    assert record["segmentation"]["runs"] == {fov: {"cells": "cell", "nuclei": None} for fov in fovs}
    steps = pd.read_csv(folder / "tables" / "steps.csv")
    cell_steps = steps[steps["name"].isin(["segment", "assign_molecules"])]
    assert sorted(zip(cell_steps["fov_id"], cell_steps["name"], cell_steps["status"])) == sorted(
        (fov, name, "succeeded") for fov in fovs for name in ("segment", "assign_molecules"))


def test_the_compare_command_calls_compare_samples_on_every_sample(tmp_path, monkeypatch):
    """From test_wf1_report_dataset_calls_compare_samples_on_every_sample: every sample's report, in fovs.csv's
    order, into report/."""
    calls = spy(monkeypatch, "compare_samples")
    config = report_config(tmp_path, {}, {})
    T.write_inputs(config, membership([("sample0", 1, 1), (SAMPLE, 2, 2), ("sample2", 3, 3)]))
    manifest, _ = T.plan(tmp_path, "compare", config)
    T.call(manifest, "compare")
    out = T.output_root(config)
    (dataset, args, kwargs), = calls
    assert dataset.output_root == out
    assert args == (["sample0", SAMPLE, "sample2"],) and kwargs == {"report_root": out / "report"}


def test_the_compare_command_keeps_the_order_of_fovs_csv(tmp_path, monkeypatch):
    """From test_wf1_report_sample_declares_the_last_per_sample_output and test_wf1_report_smk_declares_exactly_these
    (the comparison took the samples in fovs.csv's order, s2 before s1): compare passes ["s2", "s1"] to
    compare_samples, although the task's declared inputs are sorted (s1 first), so that no manifest changes."""
    calls = spy(monkeypatch, "compare_samples")
    config = report_config(tmp_path, {}, {})
    T.write_inputs(config, "sample,fov_id\ns2,F1\ns1,F2\ns2,F3\n")
    manifest, planned = T.plan(tmp_path, "order", config)
    assert T.task(manifest, "compare")["inputs"] == ["report/s1/report.json", "report/s2/report.json"]
    assert list(planned["membership"]["samples"]) == ["s2", "s1"]
    T.call(manifest, "compare")
    (_, args, kwargs), = calls
    assert args == (["s2", "s1"],) and kwargs == {"report_root": T.output_root(config) / "report"}


def test_an_invalid_report_block_is_refused_before_any_task(tmp_path, monkeypatch):
    """From test_wf1_the_rules_raise_before_calling_the_entries: the plan refuses the block, naming the unknown key,
    and neither entry is called."""
    calls = (spy(monkeypatch, "report"), spy(monkeypatch, "compare_samples"))
    config = report_config(tmp_path, {"title": "LN"}, {})
    T.write_inputs(config, membership([(SAMPLE, 1, 2)]))
    with pytest.raises(PlanRefused, match=r"unknown report keys: \['title'\]"):
        T.plan(tmp_path, "refused", config)
    assert calls == ([], []) and not T.output_root(config).exists()


def test_the_report_and_compare_commands_write_the_report_and_the_comparison_page(tmp_path):
    """From test_wf1_the_rules_write_the_report_and_the_comparison_page: the real entries through both commands on
    report_records (hand-written run records and placement of sample s1, FOVs F1 to F4): the report task writes
    report/s1/ with its report.json, the compare task report/samples.csv and report/index.html."""
    from .report_fixtures import DATASET, FOVS, SAMPLE_ID, report_records
    from .report_fixtures import SAMPLE as REPORT_SAMPLE
    config = report_config(tmp_path, {"marker_pairs": [["G1", "G2"]]}, {"stitching": MERGED})
    config.update(dataset_id=DATASET, sample_id=SAMPLE_ID, output_id="output")
    T.write_inputs(config, "sample,fov_id\n" + "".join(f"{REPORT_SAMPLE},{fov}\n" for fov in FOVS))
    manifest, _ = T.plan(tmp_path, "records", config)
    out = T.output_root(config)
    case = report_records(out)
    assert list(case.fov_ids) == list(FOVS)
    task = T.task(manifest, "report", sample=REPORT_SAMPLE)
    assert f"stitching/{REPORT_SAMPLE}/placement.json" in task["inputs"]
    assert (out / "stitching" / REPORT_SAMPLE / "placement.json").is_file()
    T.call(manifest, f"report/{REPORT_SAMPLE}")
    record = load_report(out / "report" / REPORT_SAMPLE)
    assert (out / "report" / REPORT_SAMPLE / "report.json").is_file()
    assert (out / "report" / REPORT_SAMPLE / "report.html").is_file()
    assert record["sample"] == REPORT_SAMPLE and record["fov_ids"] == list(FOVS)
    T.call(manifest, "compare")
    samples = pd.read_csv(out / "report" / "samples.csv")
    assert samples.iloc[:, 0].tolist() == [REPORT_SAMPLE] and "fovs_expected" in samples
    assert (out / "report" / "index.html").is_file()


def _report_rerun_case(tmp_path):
    from .report_fixtures import DATASET, FOVS, SAMPLE_ID, report_records
    from .report_fixtures import SAMPLE as sample
    config = report_config(tmp_path, {}, {"stitching": MERGED})
    config.update(dataset_id=DATASET, sample_id=SAMPLE_ID, output_id="output")
    T.write_inputs(config, "sample,fov_id\n" + "".join(f"{sample},{fov}\n" for fov in FOVS))
    manifest, _ = T.plan(tmp_path, "rerun", config)
    out = T.output_root(config)
    report_records(out)
    # Earlier tasks' outputs are the hand-written report fixture, as in the real-entry test above.
    T.succeed(manifest, "fov", "stitch")
    for relative in T.task(manifest, "report", sample=sample)["inputs"]:
        path = out / relative
        if not path.exists():
            assert "/checkpoints/" in relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}")
    folder = out / "report" / sample
    (folder / "tables").mkdir(parents=True)
    (folder / "report.html").write_text("old page")
    (folder / "tables/stale.csv").write_text("old table")
    return manifest, sample, folder


def test_report_task_replaces_page_and_tables_without_record(tmp_path):
    manifest, sample, folder = _report_rerun_case(tmp_path)
    assert T.run(manifest, f"report/{sample}") == 0
    record = load_report(folder)
    assert {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()} == {
        "report.json", *(entry["path"] for entry in record["files"])}


def test_report_task_preserves_and_names_a_foreign_file(tmp_path, capsys):
    manifest, sample, folder = _report_rerun_case(tmp_path)
    foreign = folder / "foreign.txt"
    foreign.write_text("owner file")
    assert T.run(manifest, f"report/{sample}") == 1
    assert "foreign.txt" in capsys.readouterr().err
    assert foreign.read_text() == "owner file"


def test_report_command_removes_nothing_when_record_exists(tmp_path, monkeypatch):
    manifest, sample, folder = _report_rerun_case(tmp_path)
    (folder / "report.json").write_text("record owned by Dataset.report")
    before = {p.relative_to(folder): p.read_bytes() for p in folder.rglob("*") if p.is_file()}
    calls = spy(monkeypatch, "report")
    T.call(manifest, f"report/{sample}")
    assert len(calls) == 1
    assert {p.relative_to(folder): p.read_bytes() for p in folder.rglob("*") if p.is_file()} == before


@pytest.fixture
def preview_export(tmp_path):
    import importlib.util
    if any(importlib.util.find_spec(module) is None for module in ("anndata", "multiview_stitcher")):
        pytest.skip("export preview checks need the anndata and multiview-stitcher extras")
    import dask
    from .test_workflow_export_tasks import scene_export
    with dask.config.set(scheduler="synchronous"):
        folder, manifest = scene_export(tmp_path, {"route": "per_fov", "fuse": False})
        yield folder, manifest


@pytest.mark.integration
def test_export_task_removes_earlier_previews_when_key_is_off(tmp_path, preview_export):
    from .test_workflow_export_tasks import SCENE_SAMPLE, plan_scene
    folder, _ = preview_export
    manifest = plan_scene(tmp_path, folder, {"route": "per_fov", "fuse": False}, export_previews=False)
    previews = folder.export_dir / "previews"
    previews.mkdir(parents=True)
    (previews / "stale.png").write_bytes(b"earlier figure")
    assert T.run(manifest, f"export/{SCENE_SAMPLE}") == 0
    assert not previews.exists()


@pytest.mark.integration
def test_export_task_rebuilds_current_figures_without_stale_files(preview_export):
    from .test_workflow_export_tasks import SCENE_SAMPLE
    folder, manifest = preview_export
    previews = folder.export_dir / "previews"
    previews.mkdir(parents=True)
    stale = previews / "stale.txt"
    stale.write_text("not a current figure")
    assert T.run(manifest, f"export/{SCENE_SAMPLE}") == 0
    assert not stale.exists()
    expected = {f"{fov}/{name}" for fov in folder.scene.ids
                for name in ("assignment.png", "assignment_single_layer.png")}
    assert {p.relative_to(previews).as_posix() for p in previews.rglob("*") if p.is_file()} == expected
    assert all((previews / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for name in expected)


@pytest.mark.integration
def test_export_task_preserves_foreign_file_outside_previews(tmp_path, preview_export):
    from .test_workflow_export_tasks import SCENE_SAMPLE, plan_scene
    folder, _ = preview_export
    manifest = plan_scene(tmp_path, folder, {"route": "per_fov", "fuse": False}, export_previews=False)
    folder.export_dir.mkdir(parents=True, exist_ok=True)
    foreign = folder.export_dir / "owner.txt"
    foreign.write_text("owner file")
    assert T.run(manifest, f"export/{SCENE_SAMPLE}") == 0
    assert foreign.read_text() == "owner file"


def _held_export_case(tmp_path, monkeypatch):
    """report_records under a plan with stitching, assembly (molecules only), export and report blocks: the fov and
    stitch tasks succeeded, and the placement needs review (F3 metadata_only and F4 failed: unplaced_fovs)."""
    from starfinder.stitching import load_placement, review_placement, save_placement
    from .report_fixtures import DATASET, FOVS, SAMPLE_ID, report_records
    from .report_fixtures import SAMPLE as sample
    config = report_config(tmp_path, {}, {"stitching": MERGED, "assembly": {"assignment": None}})
    config.update(dataset_id=DATASET, sample_id=SAMPLE_ID, output_id="output")
    T.write_inputs(config, "sample,fov_id\n" + "".join(f"{sample},{fov}\n" for fov in FOVS))
    manifest, _ = T.plan(tmp_path, "held", config)
    out = T.output_root(config)
    report_records(out, "placement_states")
    folder = out / "stitching" / sample
    save_placement(review_placement(load_placement(folder)), folder)
    T.succeed(manifest, "fov", "stitch")
    # The export refuses on the review before it reads a molecule table or an image: empty stand-ins of those it
    # declares, and the fixture's codebook is a header alone.
    for relative in T.task(manifest, "export", sample=sample)["inputs"]:
        if not (out / relative).exists():
            (out / relative).write_bytes(b"")
    monkeypatch.setattr(Dataset, "load_codebook", lambda self, *args, **kwargs: None)
    return manifest, sample, out


def test_a_held_export_makes_the_report_and_exits_4(tmp_path, monkeypatch, capsys):
    """W-814: an export whose placement needs review fails with exit status 4 after making the sample report from the
    FOVs' records and placement.json; nothing records it, or the report, as succeeded."""
    from starfinder import _layout
    from starfinder.workflow import verify
    manifest, sample, out = _held_export_case(tmp_path, monkeypatch)
    export_id, report_id = f"export/{sample}", f"report/{sample}"
    assert export_id in T.task(manifest, "report", sample=sample)["needs"]     # the plan is unchanged
    assert T.run(manifest, export_id) == 4
    assert f"export held for review: {export_id}: TaskHeld:" in capsys.readouterr().err
    failed = json.loads(_layout.failed_record(out, export_id).read_text())
    assert failed["status"] == "held" and failed["error"]["type"] == "TaskHeld"
    assert "approve_placement(result, reviewer=..., note=...)" in failed["error"]["message"]
    assert not (out / "export" / sample / "export.json").exists()
    assert not _layout.task_record(out, export_id).exists() and not _layout.task_record(out, report_id).exists()
    record = load_report(out / "report" / sample)
    assert record["fov_ids"] == ["F1", "F2", "F3", "F4"]
    assert [entry["path"] for entry in record["inputs"] if entry["path"].startswith("stitching/")] == [
        f"stitching/{sample}/placement.json", f"stitching/{sample}/fovs.csv"]
    page = (out / "report" / sample / "report.html").read_text(encoding="utf-8")
    assert "<td>Flagged FOVs</td><td>unplaced_fovs: F3, F4</td>" in page
    log = (out / T.task(manifest, "export", sample=sample)["log"]).read_text()
    assert "made the report of the held export" in log
    result = verify(manifest, final=True)
    assert not result["passed"] and {export_id, report_id, "compare"} <= set(result["incomplete"])
    assert result["tasks"][export_id]["problems"] == [
        "held for review (a failed record that says held) and no task record"]
    # The report task itself still refuses before the export succeeds.
    assert T.run(manifest, report_id) == 2


def test_a_held_export_without_its_report_needs_makes_no_report(tmp_path, monkeypatch):
    # A need of the report that the export does not have (the assign tasks of the route after fusion), unfinished:
    # written into the manifest here, since every need of this plan's report is also the export's.
    manifest, sample, out = _held_export_case(tmp_path, monkeypatch)
    data = json.loads(manifest.read_text())
    next(t for t in data["tasks"] if t["kind"] == "report")["needs"].append("assign/s1/F1/default")
    manifest.write_text(json.dumps(data))
    assert T.run(manifest, f"export/{sample}") == 4
    assert not (out / "report" / sample).exists()
    log = (out / T.task(manifest, "export", sample=sample)["log"]).read_text()
    assert f"the report of sample {sample} is not made: report/{sample} also needs assign/s1/F1/default" in log


def test_a_stitch_rerun_keeps_the_approvals_of_unchanged_tables(tmp_path):
    """W-814: approve_placement and save_placement change placement.json; the rerun of the placement task keeps each
    earlier approval whose table SHA-256 equal the new tables' (review_placement's rule), and only those."""
    from types import SimpleNamespace
    from starfinder.stitching import ReviewPolicy, approve_placement, load_placement, review_placement, save_placement
    from starfinder.workflow._commands import _keep_approvals, earlier_approvals
    from .report_fixtures import FOVS, placement_result
    from .report_fixtures import SAMPLE as sample
    folder = tmp_path / "stitching" / sample
    run = SimpleNamespace(out=tmp_path, identity={"sample": sample}, approvals=())
    assert earlier_approvals(run) == ()
    placed = review_placement(placement_result(FOVS, statuses={"F4": "failed"}))
    save_placement(approve_placement(placed, reviewer="A. Reviewer", note="checked"), folder)
    run.approvals = earlier_approvals(run)
    assert [a["reviewer"] for a in run.approvals] == ["A. Reviewer"]
    # The rerun writes the same tables: the approval is kept and the export's review gate passes.
    save_placement(placed, folder)
    _keep_approvals(run, ReviewPolicy())
    kept = load_placement(folder)
    assert kept.record["review"]["approvals"] == list(run.approvals)
    from starfinder.stitching._review import require_approval
    require_approval(kept)
    # The rerun places the FOVs otherwise: no approval is kept and placement.json stays as written.
    other = review_placement(placement_result(FOVS, statuses={"F3": "failed"}))
    save_placement(other, folder)
    before = (folder / "placement.json").read_bytes()
    _keep_approvals(run, ReviewPolicy())
    assert (folder / "placement.json").read_bytes() == before
    assert load_placement(folder).record["review"]["approvals"] == []
