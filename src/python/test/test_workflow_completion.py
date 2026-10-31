"""C7 of design/validation/orchestration.md: the completion rule and the verify command.

On one FOV of ``tiny_sample``: an API call that raises exits 1, removes the attempt's declared outputs, writes
``<task>.failed.json`` and no record; a declared output missing after the call exits 1; a success writes the record
last, listing every declared output but itself with its SHA-256 and the closed log. ``verify`` reports the task
complete, and not complete after an output is deleted or truncated, after the record is deleted, with a failed record
and with a record whose fingerprint differs from the manifest's; its token for a complete task is the record's
``workflow.token`` and for any other task ``rerun:`` with an id that differs between two invocations. A command that
reads a block outside its kind's map fails.
"""
import hashlib
import json
import shutil

import pytest

from starfinder import _layout
from starfinder.dataset.fov import FOV
from starfinder.workflow import plan, verify

from . import workflow_examples as X

pytestmark = [pytest.mark.workflow, pytest.mark.contract]

FOV_TASK = "fov/sample/FOV_001"
OUTPUTS = ["fovs/FOV_001/checkpoints/candidates.json", "fovs/FOV_001/checkpoints/pre_qc.json",
           "fovs/FOV_001/molecules.csv", "fovs/FOV_001/reference.tif", "fovs/FOV_001/run.json"]


@pytest.fixture(scope="module")
def fixture_work(tmp_path_factory):
    work = tmp_path_factory.mktemp("tiny") / "work"
    X.golden().write_fixture(work)
    return work


@pytest.fixture
def manifest(fixture_work, tmp_path, monkeypatch):
    work = tmp_path / "work"
    shutil.copytree(fixture_work, work, symlinks=True)
    X.fix_code(monkeypatch)
    plan(X.write_config(work, "single", X.single_fov(work)))
    return X.output_root(work, "single") / "workflow" / "manifest.json"


@pytest.fixture
def done(manifest):
    assert X.run_task(manifest, FOV_TASK, "--cache-token", "token-1") == 0
    return manifest


def _sha(path):
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _task(manifest):
    return next(t for t in json.loads(manifest.read_text())["tasks"] if t["id"] == FOV_TASK)


def test_the_task_declares_the_fov_kinds_outputs(manifest):
    assert _task(manifest)["outputs"] == OUTPUTS


def test_an_api_call_that_raises_removes_the_attempts_outputs_and_writes_a_failed_record(manifest, monkeypatch):
    def broken(self, *args, **kwargs):
        raise RuntimeError("the export broke")
    monkeypatch.setattr(FOV, "export_spots", broken)
    out = manifest.parents[1]
    assert X.run_task(manifest, FOV_TASK) == 1
    assert not _layout.task_record(out, FOV_TASK).exists()
    assert [o for o in OUTPUTS if (out / o).exists()] == []
    failed = json.loads(_layout.failed_record(out, FOV_TASK).read_text())
    assert failed["status"] == "failed" and failed["error"] == {"type": "RuntimeError", "message": "the export broke"}
    assert sorted(failed["removed"]) == sorted(o for o in OUTPUTS if o != "fovs/FOV_001/molecules.csv")
    log = _layout.task_log(out, FOV_TASK)
    assert failed["log"] == {"path": "workflow/logs/fov.sample.FOV_001.log", "sha256": _sha(log)}
    text = log.read_text()
    assert "Traceback" in text and "RuntimeError: the export broke" in text


def test_a_declared_output_missing_after_the_call_fails_the_task(manifest, monkeypatch, capsys):
    monkeypatch.setattr(FOV, "export_spots", lambda self, *args, **kwargs: None)
    out = manifest.parents[1]
    assert X.run_task(manifest, FOV_TASK) == 1
    assert "without the declared outputs fovs/FOV_001/molecules.csv" in capsys.readouterr().err
    assert not _layout.task_record(out, FOV_TASK).exists()
    assert [o for o in OUTPUTS if (out / o).exists()] == []
    assert _layout.failed_record(out, FOV_TASK).is_file()


def test_a_success_writes_the_record_last_listing_every_output_and_the_closed_log(done):
    out = done.parents[1]
    path = _layout.task_record(out, FOV_TASK)
    record = json.loads(path.read_text())
    assert record["format"] == "starfinder.workflow.task/1" and record["status"] == "succeeded"
    assert record["id"] == FOV_TASK and record["kind"] == "fov"
    listed = {entry["path"]: entry for entry in record["outputs"]}
    assert sorted(listed) == sorted([*OUTPUTS, "workflow/logs/fov.sample.FOV_001.log"])
    for name, entry in listed.items():
        assert entry["sha256"] == _sha(out / name) and entry["bytes"] == (out / name).stat().st_size, name
    assert all(path.stat().st_mtime_ns >= (out / name).stat().st_mtime_ns for name in listed)
    assert not _layout.failed_record(out, FOV_TASK).exists()
    task = _task(done)
    workflow = record["workflow"]
    assert workflow["fingerprint"] == task["fingerprint"] and workflow["blocks_sha256"] == task["blocks_sha256"]
    assert workflow["token"] == "token-1" and workflow["code"] == X.CLEAN_CODE
    assert workflow["manifest_sha256"] == _sha(done)
    assert workflow["execution_context"]["engine"] == "manual" and workflow["execution_context"]["attempt"] == 1
    assert workflow["backend"] == {"name": "python", "device": "cpu"} and workflow["fallback"] is None
    assert workflow["wall_s"] > 0 and workflow["cpu_s"] > 0 and workflow["peak_rss_kib"] > 0
    run = json.loads((out / "fovs" / "FOV_001" / "run.json").read_text())
    assert run["status"] == "succeeded" and run["workflow"]["fingerprint"] == task["fingerprint"]
    assert run["workflow_config"] == {"path": "workflow/resolved.json", "sha256": _sha(done.parent / "resolved.json")}
    lines = (out / "workflow" / "logs" / "fov.sample.FOV_001.log").read_text().splitlines()
    assert lines[0].endswith(f"starfinder.workflow {FOV_TASK} task {FOV_TASK} (fov), attempt 1")
    assert " INFO starfinder " in "\n".join(lines) and lines[-1].split(" ", 3)[2] == "starfinder.workflow"


def test_verify_reports_a_complete_task_with_its_records_token(done):
    result = verify(done, hash=True, final=True)
    assert result["tasks"][FOV_TASK] == {"complete": True, "problems": [], "token": "token-1"}
    assert result["passed"] and result["incomplete"] == []
    written = json.loads((done.parent / "verify.json").read_text())
    assert written["tasks"] == result["tasks"] and written["invocation"] == result["invocation"]


def _damage(out, how):
    if how == "deleted output":
        (out / "fovs" / "FOV_001" / "molecules.csv").unlink()
    elif how == "truncated output":
        path = out / "fovs" / "FOV_001" / "molecules.csv"
        path.write_text("".join(path.read_text().splitlines(keepends=True)[:-1]))
    elif how == "deleted record":
        _layout.task_record(out, FOV_TASK).unlink()
    elif how == "failed record":
        _layout.task_record(out, FOV_TASK).unlink()
        _layout.failed_record(out, FOV_TASK).write_text(json.dumps({"status": "failed"}))
    elif how == "another fingerprint":
        path = _layout.task_record(out, FOV_TASK)
        record = json.loads(path.read_text())
        record["workflow"]["fingerprint"] = "sha256:" + "0" * 64
        path.write_text(json.dumps(record))


PROBLEMS = {"deleted output": "fovs/FOV_001/molecules.csv is missing",
            "truncated output": "fovs/FOV_001/molecules.csv has",
            "deleted record": "no task record", "failed record": "a failed record and no task record",
            "another fingerprint": "the record's fingerprint sha256:0000"}


@pytest.mark.parametrize("how", sorted(PROBLEMS))
def test_verify_reports_a_damaged_or_stale_task_as_not_complete_with_a_fresh_rerun_token(done, how, capsys):
    from starfinder.__main__ import main
    _damage(done.parents[1], how)
    first, second = verify(done), verify(done)
    for result in (first, second):
        entry = result["tasks"][FOV_TASK]
        assert not entry["complete"] and any(PROBLEMS[how] in p for p in entry["problems"]), entry
        assert result["incomplete"] == [FOV_TASK] and result["passed"]
    tokens = [r["tasks"][FOV_TASK]["token"] for r in (first, second)]
    assert all(t.startswith("rerun:") for t in tokens) and tokens[0] != tokens[1]
    assert main(["workflow", "verify", "--manifest", str(done), "--final"]) == 1
    assert "not complete" in capsys.readouterr().out


def test_a_command_that_reads_a_block_outside_its_map_fails(manifest, monkeypatch):
    from starfinder.workflow import _commands
    monkeypatch.setitem(_commands.FUNCTIONS, "fov", lambda run: run.configuration.get("export"))
    out = manifest.parents[1]
    assert X.run_task(manifest, FOV_TASK) == 1
    failed = json.loads(_layout.failed_record(out, FOV_TASK).read_text())
    assert failed["error"]["type"] == "BlockAccessError"
    assert "the fov command read the export block" in failed["error"]["message"]
