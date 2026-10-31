"""C6 of design/validation/orchestration.md: the negative checks of a task command.

On one FOV of ``tiny_sample`` with the processing block (and, planned only, the downstream blocks): an unknown task,
a resolved configuration whose hash differs, a ``needs`` task without a succeeded record, a missing declared input
and a translation that differs from ``api`` each exit 2 and write nothing; a verified resume whose ``run.json``
differs in ``blocks_sha256``, an input's hash or the package version, or whose record or package is dirty, exits 3
naming every differing field and writes nothing.
"""
import hashlib
import json
import shutil

import pytest

from starfinder import _layout
from starfinder.dataset.fov import FOV
from starfinder.workflow import plan

from . import workflow_examples as X

pytestmark = [pytest.mark.workflow, pytest.mark.contract]

FOV_TASK = "fov/sample/FOV_001"


@pytest.fixture(scope="module")
def fixture_work(tmp_path_factory):
    work = tmp_path_factory.mktemp("tiny") / "work"
    X.golden().write_fixture(work)
    return work


@pytest.fixture
def work(fixture_work, tmp_path, monkeypatch):
    target = tmp_path / "work"
    shutil.copytree(fixture_work, target, symlinks=True)
    X.fix_code(monkeypatch)
    return target


def _plan(work, config):
    plan(X.write_config(work, "single", config))
    return X.output_root(work, "single") / "workflow" / "manifest.json"


def _tree(root):
    """Every file under ``root`` with its bytes' SHA-256 (what a refused command must leave unchanged)."""
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def _refused(manifest, task, capsys, status, *parts):
    root = manifest.parents[1]
    before = _tree(root)
    assert X.run_task(manifest, task) == status
    message = capsys.readouterr().err
    for part in parts:
        assert part in message, (part, message)
    assert _tree(root) == before
    return message


def test_an_unknown_task_is_refused(work, capsys):
    from starfinder.__main__ import main
    manifest = _plan(work, X.single_fov(work))
    before = _tree(manifest.parents[1])
    assert main(["workflow", "fov", "--manifest", str(manifest), "--task", "fov/sample/FOV_009"]) == 2
    assert "has no task 'fov/sample/FOV_009'" in capsys.readouterr().err
    assert _tree(manifest.parents[1]) == before


def test_a_resolved_configuration_whose_hash_differs_is_refused(work, capsys):
    manifest = _plan(work, X.single_fov(work))
    resolved = manifest.parent / "resolved.json"
    resolved.write_bytes(resolved.read_bytes() + b" ")
    _refused(manifest, FOV_TASK, capsys, 2, "resolved configuration", "plan the run again")


def test_a_needs_task_without_a_succeeded_record_is_refused(work, capsys):
    manifest = _plan(work, X.single_fov(work, downstream=True))
    _refused(manifest, "stitch/sample", capsys, 2, "needs fov/sample/FOV_001", "no record that says succeeded")


def test_a_missing_declared_input_is_refused(work, capsys):
    manifest = _plan(work, X.single_fov(work))
    (X.input_root(work) / "round3" / "FOV_001" / "ch02.ome.tif").unlink()
    _refused(manifest, FOV_TASK, capsys, 2, "input:round3/FOV_001/ch02.ome.tif", "do not exist")


def test_a_translation_that_differs_from_the_plans_is_refused(work, capsys):
    # The commands carry no default the plan did not resolve: a task translates its block again and compares.
    manifest = _plan(work, X.single_fov(work))
    resolved = manifest.parent / "resolved.json"
    data = json.loads(resolved.read_text())
    data["configuration"]["api"]["processing"]["pipeline"]["spot_finding"]["threshold_value"] = 9.0
    text = json.dumps(data, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    resolved.write_bytes(text)
    record = json.loads(manifest.read_text())
    record["configuration"]["sha256"] = "sha256:" + hashlib.sha256(text).hexdigest()
    manifest.write_text(json.dumps(record, indent=1) + "\n")
    _refused(manifest, FOV_TASK, capsys, 2, "translates the processing block differently from the plan",
             "api.processing differs in pipeline")


# --- the refusals of a verified resume (exit 3) ------------------------------------------------------------------

class Kill(BaseException):
    """A stop of the task in the middle of FOV.run (as a scheduler's kill): nothing is cleaned up."""


@pytest.fixture
def interrupted(work, monkeypatch):
    """A fov task stopped after its pre_qc stage (in filter_reads), planned with resume_from_checkpoint."""
    config = X.single_fov(work, resume_from_checkpoint=True)
    manifest = _plan(work, config)
    with monkeypatch.context() as patch:
        def stop(self, **kwargs):
            raise Kill()
        patch.setattr(FOV, "filter_reads", stop)
        with pytest.raises(Kill):
            X.run_task(manifest, FOV_TASK)
    out = manifest.parents[1]
    record = json.loads(_layout.run_record(out, "FOV_001").read_text())
    # FOV.run records an interruption it sees as such; a kill it never sees leaves "running".
    assert record["status"] == "interrupted" and set(record["checkpoints"]) == {"candidates", "pre_qc"}
    assert record["workflow"]["blocks_sha256"] == json.loads(manifest.read_text())["tasks"][0]["blocks_sha256"]
    assert not _layout.task_record(out, FOV_TASK).exists()
    return config, manifest


def test_a_resume_after_a_changed_processing_key_is_refused(work, interrupted, capsys):
    config, _ = interrupted
    config["processing"]["spot_finding"]["threshold_value"] = 6.0
    manifest = _plan(work, config)
    message = _refused(manifest, FOV_TASK, capsys, 3, "resume_from_checkpoint refused for fov/sample/FOV_001",
                       "fovs/FOV_001/run.json does not match this task", "workflow.blocks_sha256 is sha256:")
    assert "(recorded)" in message and "(now)" in message


def test_a_resume_after_a_changed_input_is_refused(work, interrupted, capsys):
    _, manifest = interrupted
    image = X.input_root(work) / "round2" / "FOV_001" / "ch01.ome.tif"
    data = bytearray(image.read_bytes())
    data[-1] ^= 1
    image.write_bytes(bytes(data))
    _refused(manifest, FOV_TASK, capsys, 3, "inputs round2/FOV_001/ch01.ome.tif sha256 differs")


def test_a_resume_under_another_package_version_is_refused(work, interrupted, capsys, monkeypatch):
    _, manifest = interrupted
    X.fix_code(monkeypatch, version="0.2.1")
    _refused(manifest, FOV_TASK, capsys, 3, "code.version is 0.2.0 (recorded), 0.2.1 (now)")


def test_a_resume_of_a_dirty_record_is_refused(work, interrupted, capsys):
    _, manifest = interrupted
    path = _layout.run_record(manifest.parents[1], "FOV_001")
    record = json.loads(path.read_text())
    record["code"]["git_dirty"] = True
    path.write_text(json.dumps(record, indent=2) + "\n")
    _refused(manifest, FOV_TASK, capsys, 3, "code.git_dirty is true (recorded)")


def test_a_resume_by_a_dirty_package_is_refused(work, interrupted, capsys, monkeypatch):
    _, manifest = interrupted
    X.fix_code(monkeypatch, git_dirty=True)
    _refused(manifest, FOV_TASK, capsys, 3, "code.git_dirty is true (now)")


def test_a_refused_resume_names_every_differing_field(work, interrupted, capsys, monkeypatch):
    config, _ = interrupted
    config["processing"]["spot_finding"]["threshold_value"] = 6.0
    manifest = _plan(work, config)
    X.fix_code(monkeypatch, version="0.2.1", git_dirty=True)
    _refused(manifest, FOV_TASK, capsys, 3, "workflow.blocks_sha256", "code.git_dirty is true (now)",
             "code.version is 0.2.0 (recorded), 0.2.1 (now)")
