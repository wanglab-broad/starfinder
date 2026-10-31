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



#: A segmentation input of a sequencing round makes the plan add the stage registered and declare the round's image.
REGISTERED_RUN = {"runs": [{"name": "nucleus", "target": "nucleus", "method": "stardist", "model": "2D_versatile_fluo",
                            "scale": 1.0, "inputs": [{"role": "nuclear", "round": "round2", "channel": "ch00"}]}]}
REGISTERED_FILES = ["checkpoints/registered/round2.ome.tif", "checkpoints/registered/transforms.json"]


def test_a_resume_from_the_registered_stage_equals_a_full_run(work, monkeypatch):
    config = dict(_config(work), segmentation=REGISTERED_RUN)
    manifest = _plan(work, config)
    task = json.loads(manifest.read_text())["tasks"][0]
    assert "fovs/FOV_001/checkpoints/registered/round2.ome.tif" in task["outputs"]
    assert X.run_task(manifest, FOV_TASK) == 0
    folder = _folder(manifest)
    names = FILES + REGISTERED_FILES
    digests = {name: hashlib.sha256((folder / name).read_bytes()).hexdigest() for name in names}
    full_record = _record(folder)
    shutil.rmtree(folder)
    _layout.task_record(manifest.parents[1], FOV_TASK).unlink()
    stopped = _stop(manifest, monkeypatch, "_assemble_intensities")
    assert list(stopped["checkpoints"]) == ["registered"]
    assert X.run_task(manifest, FOV_TASK) == 0
    assert {name: hashlib.sha256((folder / name).read_bytes()).hexdigest() for name in names} == digests
    resumed = _record(folder)
    # A resume from registered runs the detection and extraction after the loop of rounds, so its steps come in
    # another order than a full run's; the same steps, and every other field, are equal.
    steps = lambda record: sorted((s["name"], s["round"] or "") for s in record.pop("steps"))  # noqa: E731
    assert steps(resumed) == steps(full_record) and resumed == full_record


def test_a_resumed_attempt_that_fails_leaves_no_declared_output(full, monkeypatch):
    """A failed attempt leaves no declared output, resumed or not: run.json, which the resume rewrote, and the stages
    it kept go too; the next attempt finds no run.json, runs from scratch and equals a full run."""
    manifest, (digests, record) = full
    _stop(manifest, monkeypatch, "filter_reads")
    out = manifest.parents[1]
    with monkeypatch.context() as patch:
        def broken(self, *args, **kwargs):
            assert json.loads((_folder(manifest) / "run.json").read_text())["workflow"]["resumed_from"] == "pre_qc"
            raise RuntimeError("the molecule table cannot be written")
        patch.setattr(FOV, "export_spots", broken)
        assert X.run_task(manifest, FOV_TASK) == 1
    task = json.loads(manifest.read_text())["tasks"][0]
    assert [o for o in task["outputs"] if (out / o).exists()] == []
    failed = json.loads(_layout.failed_record(out, FOV_TASK).read_text())
    assert {"fovs/FOV_001/run.json", "fovs/FOV_001/checkpoints/candidates.json",
            "fovs/FOV_001/checkpoints/pre_qc.json", "fovs/FOV_001/reference.tif"} <= set(failed["removed"])
    assert _layout.task_log(out, FOV_TASK).is_file() and not _layout.task_record(out, FOV_TASK).exists()
    assert X.run_task(manifest, FOV_TASK) == 0
    task_record = json.loads(_layout.task_record(out, FOV_TASK).read_text())
    assert task_record["workflow"]["resumed_from"] is None
    assert _digests(_folder(manifest)) == digests and _record(_folder(manifest)) == record


def test_a_record_with_no_stage_runs_from_scratch(full, monkeypatch):
    """An attempt killed before its first stage (in load_images, its final write never made, as after a kill) leaves a
    run.json with no stage, no input and no preprocessing, which the check would refuse. It is not a resume: the stage
    is found before the record is checked, and the task runs from scratch."""
    from starfinder.dataset._run_record import _RunRecord
    manifest, (digests, record) = full
    with monkeypatch.context() as patch:
        def stop(self, **kwargs):
            raise Kill()
        patch.setattr(FOV, "load_images", stop)
        patch.setattr(_RunRecord, "finish", lambda self, *args, **kwargs: None)
        with pytest.raises(Kill):
            X.run_task(manifest, FOV_TASK)
    stopped = json.loads((_folder(manifest) / "run.json").read_text())
    assert stopped["status"] == "running" and stopped["checkpoints"] == {} and stopped["preprocessing"] is None
    assert stopped["inputs"] == []
    assert X.run_task(manifest, FOV_TASK) == 0
    task_record = json.loads(_layout.task_record(manifest.parents[1], FOV_TASK).read_text())
    assert task_record["workflow"]["resumed_from"] is None
    assert _digests(_folder(manifest)) == digests and _record(_folder(manifest)) == record


# --- every row of the table "Verified resume", each side (design/specs/orchestration.md) -------------------------

SUPPLIED_STEP = {"method": "percentile_normalization", "p_low": 1.0, "p_high": 99.9, "fit": "supplied"}


class Stopped:
    """An attempt of the fov task stopped after its pre_qc stage, with what a case needs to change it."""

    def __init__(self, work, monkeypatch, supplied=False):
        self.work, self.monkeypatch = work, monkeypatch
        self.config = _config(work)
        if supplied:
            self.config["processing"]["preprocessing"]["steps"].insert(1, dict(SUPPLIED_STEP))
        self.manifest = _plan(work, self.config)
        for task in json.loads(self.manifest.read_text())["tasks"]:
            if task["kind"] in ("summarize", "merge"):
                assert X.run_task(self.manifest, task["id"]) == 0, task["id"]
        _stop(self.manifest, monkeypatch, "filter_reads")
        self.path = _folder(self.manifest) / "run.json"

    def edit(self, change):
        record = json.loads(self.path.read_text())
        change(record)
        self.path.write_text(json.dumps(record, indent=2) + "\n")
        return self.manifest

    def replan(self, change):
        change(self.config)
        self.manifest = _plan(self.work, self.config)
        return self.manifest

    def code(self, **code):
        X.fix_code(self.monkeypatch, **code)
        return self.manifest

    def touch(self, relative, root=None):
        path = (root or X.input_root(self.work)) / relative
        path.write_bytes(path.read_bytes() + b"\n")
        return self.manifest


def _set_record(*path, value):
    def change(record):
        node = record
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value(node.get(path[-1])) if callable(value) else value
    return change


def _without_decoding(config):
    """The processing block without decoding (so the task loads no codebook), and no block that reads molecules."""
    for key in ("reads_filtration", "decoding", "deduplication"):
        config["processing"].pop(key)
    for key in ("stitching", "assembly", "export", "report"):
        config.pop(key)


SUPPLIED_FILE = "statistics/sample/percentile_normalization/supplied.json"
#: Each row of the table and side: (the base attempt with a supplied step or not, the change, what the message names).
TABLE = {
    "format_version": (False, lambda s: s.edit(_set_record("format_version", value=1)), "format_version is 1"),
    "dataset_id": (False, lambda s: s.edit(_set_record("dataset_id", value="other")), "dataset_id is 'other'"),
    "sample_id": (False, lambda s: s.edit(_set_record("sample_id", value="other")), "sample_id is 'other'"),
    "fov_id": (False, lambda s: s.edit(_set_record("fov_id", value="FOV_009")), "fov_id is 'FOV_009'"),
    "subtile_id": (False, lambda s: s.edit(_set_record("subtile_id", value=1)), "subtile_id is 1"),
    "workflow.blocks_sha256": (False, lambda s: s.edit(_set_record("workflow", "blocks_sha256", value="sha256:0")),
                               "workflow.blocks_sha256 is sha256:0 (recorded)"),
    "code.version": (False, lambda s: s.code(version="0.2.1"), "code.version is 0.2.0 (recorded), 0.2.1 (now)"),
    "code.git_commit": (False, lambda s: s.code(git_commit="f" * 40), "code.git_commit is 0123456789abcdef"),
    "git_dirty recorded": (False, lambda s: s.edit(_set_record("code", "git_dirty", value=True)),
                           "code.git_dirty is true (recorded)"),
    "git_dirty now": (False, lambda s: s.code(git_dirty=True), "code.git_dirty is true (now)"),
    "a dirty build's version": (False, lambda s: s.code(version="0.2.0.dev3+g0123456.d20261031"),
                                "is a build with uncommitted changes (now)"),
    "a version that cannot be established": (False, lambda s: s.code(version=None),
                                             "code.version cannot be established (now)"),
    "an input missing from the record": (False, lambda s: s.edit(_set_record(
        "inputs", value=lambda v: [e for e in v if e["path"] != "round2/FOV_001/ch01.ome.tif"])),
        "inputs round2/FOV_001/ch01.ome.tif is not recorded"),
    "an extra input in the record": (False, lambda s: s.edit(_set_record(
        "inputs", value=lambda v: v + [{"path": "round9/FOV_001/ch00.ome.tif", "sha256": "0" * 64}])),
        "inputs round9/FOV_001/ch00.ome.tif is recorded and no input of this task"),
    "an input with another hash": (False, lambda s: s.touch("round4/FOV_001/ch03.ome.tif"),
                                   "inputs round4/FOV_001/ch03.ome.tif sha256 differs"),
    "the codebook's hash": (False, lambda s: s.touch("genes.csv"), "codebook genes.csv sha256 differs"),
    "a codebook only in the record": (False, lambda s: s.replan(_without_decoding),
                                      "codebook is genes.csv (recorded), none (now)"),
    "a codebook only in the task": (False, lambda s: s.edit(_set_record("codebook", value=None)),
                                    "codebook is not recorded (recorded)"),
    "the supplied statistics' hash": (True, lambda s: s.touch(SUPPLIED_FILE, X.output_root(s.work, "single")),
                                      "preprocessing.supplied_statistics", ),
    "supplied statistics only in the record": (False, lambda s: s.edit(_set_record(
        "preprocessing", "supplied_statistics", value={"path": "/elsewhere/supplied.json", "sha256": "0" * 64})),
        "preprocessing.supplied_statistics is /elsewhere/supplied.json (recorded), none (now)"),
    "supplied statistics only in the task": (True, lambda s: s.edit(_set_record(
        "preprocessing", "supplied_statistics", value={"path": None, "sha256": None})),
        "preprocessing.supplied_statistics is not recorded (recorded)"),
    "a stage record's hash": (False, lambda s: s.touch("checkpoints/candidates.json", _folder(s.manifest)),
                              "checkpoints.candidates checkpoints/candidates.json sha256 differs"),
    "a payload's hash": (False, lambda s: s.touch("checkpoints/pre_qc.csv", _folder(s.manifest)),
                         "checkpoints.pre_qc file pre_qc.csv sha256 differs"),
}


@pytest.mark.parametrize("case", list(TABLE))
def test_each_row_of_the_resume_table_refuses_a_mismatch(work, monkeypatch, capsys, case):
    supplied, change, named = TABLE[case]
    stopped = Stopped(work, monkeypatch, supplied)
    manifest = change(stopped)
    capsys.readouterr()
    _refused(manifest, capsys, named)


def test_the_supplied_statistics_case_resumes_when_nothing_differs(work, monkeypatch):
    """The base of the supplied statistics' cases resumes: the record holds the file this task reads."""
    stopped = Stopped(work, monkeypatch, supplied=True)
    record = json.loads(stopped.path.read_text())
    assert record["preprocessing"]["supplied_statistics"]["path"].endswith(SUPPLIED_FILE)
    assert X.run_task(stopped.manifest, FOV_TASK) == 0
