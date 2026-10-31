"""X8 of design/validation/orchestration.md: verified resume of the fov task.

On one FOV of ``tiny_sample`` with ``execution.resume_from_checkpoint``: after a stop at ``pre_qc`` (or at
``candidates``) the fov command resumes and its outputs equal those of a full run; with a changed ``processing`` key,
a changed input or another package version (whichever way ``execution.code_version`` points) it refuses with exit 3
naming every differing field; with a changed ``export`` key, a block the fov task does not read, it resumes; a
version-1 ``run.json`` is never resumed. The code identity is patched (``starfinder._records.code``), so that the
tests do not depend on the state of the checkout.
"""
import hashlib
import json
import shutil

import pytest

from starfinder import _layout
from starfinder.dataset.fov import FOV
from starfinder.workflow import plan

from . import workflow_examples as X

pytestmark = [pytest.mark.workflow, pytest.mark.dataset, pytest.mark.contract, pytest.mark.extended]

FOV_TASK = "fov/sample/FOV_001"
#: The fov task's outputs compared byte for byte with the full run, and the stage tables they list.
FILES = ["molecules.csv", "reference.tif", "checkpoints/candidates.json", "checkpoints/pre_qc.json",
         "checkpoints/candidates.csv", "checkpoints/pre_qc.csv"]


class Kill(BaseException):
    """A stop of the task in the middle of FOV.run, as a scheduler's kill: the command cleans nothing up."""


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


def _config(work, **execution):
    return X.single_fov(work, downstream=True, resume_from_checkpoint=True, **execution)


def _plan(work, config):
    plan(X.write_config(work, "single", config))
    return X.output_root(work, "single") / "workflow" / "manifest.json"


def _folder(manifest):
    return _layout.fov_dir(manifest.parents[1], "FOV_001")


def _digests(folder):
    return {name: hashlib.sha256((folder / name).read_bytes()).hexdigest() for name in FILES}


def _record(folder):
    """run.json as the comparison reads it: without where, when and with which software it was made, its times, the
    workflow section (the execution context and the resume) and the resolved configuration's hash (execution, which
    holds resume_from_checkpoint, is part of the file)."""
    record = json.loads((folder / "run.json").read_text())
    for key in ("started_at", "ended_at", "code", "environment", "software", "workflow", "workflow_config"):
        record.pop(key, None)
    for step in record["steps"]:
        step.pop("seconds")
    return record


def _stop(manifest, monkeypatch, method):
    """Run the fov task until ``method`` (filter_reads: after pre_qc; decode_reads: after candidates) and stop."""
    with monkeypatch.context() as patch:
        def stop(self, **kwargs):
            raise Kill()
        patch.setattr(FOV, method, stop)
        with pytest.raises(Kill):
            X.run_task(manifest, FOV_TASK)
    record = json.loads((_folder(manifest) / "run.json").read_text())
    assert record["status"] == "interrupted"
    return record


@pytest.fixture
def full(work):
    """A full run of the task: its outputs' digests and its normalized run.json, then the outputs removed."""
    manifest = _plan(work, _config(work))
    assert X.run_task(manifest, FOV_TASK) == 0
    folder = _folder(manifest)
    result = (_digests(folder), _record(folder))
    shutil.rmtree(folder)
    _layout.task_record(manifest.parents[1], FOV_TASK).unlink()
    return manifest, result


@pytest.mark.parametrize("stage, method", [("pre_qc", "filter_reads"), ("candidates", "decode_reads")])
def test_a_resume_after_a_stop_equals_a_full_run(full, monkeypatch, stage, method):
    manifest, (digests, record) = full
    stopped = _stop(manifest, monkeypatch, method)
    assert list(stopped["checkpoints"])[-1] == stage
    assert X.run_task(manifest, FOV_TASK) == 0
    folder = _folder(manifest)
    task_record = json.loads(_layout.task_record(manifest.parents[1], FOV_TASK).read_text())
    assert task_record["workflow"]["resumed_from"] == stage
    log = _layout.task_log(manifest.parents[1], FOV_TASK).read_text()
    assert "kept from the earlier attempt: fovs/FOV_001/checkpoints/candidates.json" in log
    assert _digests(folder) == digests
    assert _record(folder) == record
    assert json.loads((folder / "run.json").read_text())["workflow"]["resumed_from"] == stage


def test_a_resume_after_a_changed_export_key_resumes(full, monkeypatch, work):
    manifest, (digests, record) = full
    _stop(manifest, monkeypatch, "filter_reads")
    config = _config(work)
    config["export"]["fuse_labels"] = True
    before = json.loads(manifest.read_text())["tasks"][0]
    manifest = _plan(work, config)
    after = json.loads(manifest.read_text())["tasks"][0]
    assert after["id"] == FOV_TASK and after["blocks_sha256"] == before["blocks_sha256"]
    assert X.run_task(manifest, FOV_TASK) == 0
    assert _digests(_folder(manifest)) == digests and _record(_folder(manifest)) == record


def _refused(manifest, capsys, *parts):
    folder = _folder(manifest)
    before = {p.relative_to(folder).as_posix(): p.read_bytes() for p in folder.rglob("*") if p.is_file()}
    assert X.run_task(manifest, FOV_TASK) == 3
    message = capsys.readouterr().err
    assert message.startswith(f"resume_from_checkpoint refused for {FOV_TASK}: fovs/FOV_001/run.json does not "
                              "match this task: ")
    for part in parts:
        assert part in message, (part, message)
    assert {p.relative_to(folder).as_posix(): p.read_bytes() for p in folder.rglob("*") if p.is_file()} == before
    assert not _layout.task_record(manifest.parents[1], FOV_TASK).exists()


def test_a_resume_after_a_changed_processing_key_is_refused(full, monkeypatch, work, capsys):
    manifest, _ = full
    _stop(manifest, monkeypatch, "filter_reads")
    config = _config(work)
    config["processing"]["reads_filtration"]["start_base"] = "A"
    _refused(_plan(work, config), capsys, "workflow.blocks_sha256 is sha256:")


def test_a_resume_after_a_changed_input_is_refused(full, monkeypatch, work, capsys):
    manifest, _ = full
    _stop(manifest, monkeypatch, "filter_reads")
    image = X.input_root(work) / "round3" / "FOV_001" / "ch02.ome.tif"
    data = bytearray(image.read_bytes())
    data[-1] ^= 1
    image.write_bytes(bytes(data))
    _refused(_plan(work, _config(work)), capsys, "inputs round3/FOV_001/ch02.ome.tif sha256 differs")


@pytest.mark.parametrize("code_version", ["keep", "rerun"])
def test_a_resume_under_another_package_version_is_refused(work, monkeypatch, capsys, code_version):
    config = _config(work, code_version=code_version)
    manifest = _plan(work, config)
    _stop(manifest, monkeypatch, "filter_reads")
    X.fix_code(monkeypatch, version="0.3.0")
    manifest = _plan(work, config)
    _refused(manifest, capsys, "code.version is 0.2.0 (recorded), 0.3.0 (now)")


def test_a_resume_after_changes_names_every_differing_field(full, monkeypatch, work, capsys):
    manifest, _ = full
    _stop(manifest, monkeypatch, "filter_reads")
    config = _config(work)
    config["processing"]["reads_filtration"]["start_base"] = "A"
    image = X.input_root(work) / "round2" / "FOV_001" / "ch00.ome.tif"
    image.write_bytes(image.read_bytes()[::-1])
    X.fix_code(monkeypatch, version="0.3.0")
    _refused(_plan(work, config), capsys, "workflow.blocks_sha256", "inputs round2/FOV_001/ch00.ome.tif sha256 differs",
             "code.version is 0.2.0 (recorded), 0.3.0 (now)")


def test_a_version_1_run_record_is_never_resumed(full, monkeypatch, capsys):
    manifest, _ = full
    _stop(manifest, monkeypatch, "filter_reads")
    path = _folder(manifest) / "run.json"
    record = json.loads(path.read_text())
    record["format_version"] = 1
    path.write_text(json.dumps(record, indent=2) + "\n")
    _refused(manifest, capsys, "format_version is 1 (recorded), 2 (now): a version-1 record is never resumed")


def test_without_resume_from_checkpoint_the_task_reruns_from_scratch(full, monkeypatch, work):
    manifest, (digests, record) = full
    _stop(manifest, monkeypatch, "filter_reads")
    manifest = _plan(work, X.single_fov(work, downstream=True))
    assert X.run_task(manifest, FOV_TASK) == 0
    task_record = json.loads(_layout.task_record(manifest.parents[1], FOV_TASK).read_text())
    assert task_record["workflow"]["resumed_from"] is None
    assert _digests(_folder(manifest)) == digests and _record(_folder(manifest)) == record

