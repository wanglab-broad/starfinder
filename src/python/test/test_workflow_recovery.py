"""X7, X8, X10 and X11: real-engine recovery and stale results, and verified FOV resume.

X8, on one FOV of ``tiny_sample`` with ``execution.resume_from_checkpoint``: after a stop at ``pre_qc`` (or at
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
    "a recorded commit and none now": (False, lambda s: s.code(git_commit=None),
                                       "code.git_commit is 0123456789abcdef01234567… (recorded), None (now)"),
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


def test_a_record_without_a_commit_resumes_with_a_package_that_has_one(work, monkeypatch):
    """code.git_commit is compared when recorded: a record that holds none resumes under a package that has one."""
    stopped = Stopped(work, monkeypatch)
    stopped.edit(_set_record("code", "git_commit", value=None))
    assert X.CLEAN_CODE["git_commit"]
    assert X.run_task(stopped.manifest, FOV_TASK) == 0
    task_record = json.loads(_layout.task_record(stopped.manifest.parents[1], FOV_TASK).read_text())
    assert task_record["workflow"]["resumed_from"] == "pre_qc"


def test_the_supplied_statistics_case_resumes_when_nothing_differs(work, monkeypatch):
    """The base of the supplied statistics' cases resumes: the record holds the file this task reads."""
    stopped = Stopped(work, monkeypatch, supplied=True)
    record = json.loads(stopped.path.read_text())
    assert record["preprocessing"]["supplied_statistics"]["path"].endswith(SUPPLIED_FILE)
    assert X.run_task(stopped.manifest, FOV_TASK) == 0


# X7, X10 and X11 use real engine jobs; the X8 tests above retain their inputs
# and assertions, including their in-process stop injection.
from . import workflow_engine_validation as E

ENGINES = ["snakemake", pytest.param("nextflow", marks=pytest.mark.nextflow)]


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("engine", ENGINES)
def test_x7_sigkill_partial_task_recovers_with_only_its_downstream(tmp_path, engine):
    """Kill a real CLI task with SIGKILL immediately after its molecule table is written."""
    import os
    import subprocess
    import sys
    run = E.EngineRun(tmp_path / "kill", engine)
    run.invoke({t["id"] for t in run.data["tasks"]})
    uninterrupted = E.snapshot(run)
    task = next(t for t in run.data["tasks"] if t["id"] == FOV_TASK)
    for name in [*task["outputs"], task["record"]]:
        path = run.out / name
        shutil.rmtree(path) if path.is_dir() else path.unlink(missing_ok=True)
    hook = tmp_path / "kill-hook"
    hook.mkdir()
    marker = hook / "killed.json"
    (hook / "sitecustomize.py").write_text(
        "import sys\n"
        "if '--task' in sys.argv and sys.argv[sys.argv.index('--task') + 1] == 'fov/sample/FOV_001':\n"
        " import os, signal, json, hashlib\n"
        " from pathlib import Path\n"
        " from starfinder.dataset.fov import FOV\n"
        " original = FOV.export_spots\n"
        " def kill(self, *args, **kwargs):\n"
        "  result = original(self, *args, **kwargs)\n"
        f"  p = Path({str(run.out / 'fovs/FOV_001/molecules.csv')!r})\n"
        f"  Path({str(marker)!r}).write_text(json.dumps({{'pid': os.getpid(), 'bytes': p.stat().st_size, "
        "'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}))\n"
        "  os.kill(os.getpid(), signal.SIGKILL)\n"
        " FOV.export_spots = kill\n")
    env = dict(os.environ, PYTHONPATH=str(hook) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    command = [sys.executable, "-m", "starfinder", *task["command"][1:], "--attempt", "1"]
    killed = subprocess.run(command, env=env, capture_output=True, text=True, timeout=180)
    assert killed.returncode == -9, killed.stdout + killed.stderr
    assert marker.is_file() and (run.out / "fovs/FOV_001/molecules.csv").is_file()
    assert not (run.out / task["record"]).exists()
    E.evidence(run.work, "x7-killed.json", {"command": command, "returncode": killed.returncode,
                "partial_output": json.loads(marker.read_text()), "record_exists": False})
    run.invoke(E.downstream(run.data, {FOV_TASK}))
    E.compare(run, uninterrupted, E.snapshot(run), name="x7-recovered.json")


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.nextflow
@pytest.mark.parametrize("damage", ["deleted-molecules", "truncated-molecules", "deleted-index", "deleted-task"])
def test_x10_nextflow_repairs_only_damaged_tasks_and_their_downstream(tmp_path, damage):
    run = E.EngineRun(tmp_path / damage, "nextflow")
    run.invoke({t["id"] for t in run.data["tasks"]})
    uninterrupted = E.snapshot(run)
    before = run.records()
    run.invoke(set())
    assert run.records() == before
    target = FOV_TASK if "molecules" in damage else "compare" if damage == "deleted-index" else "assign/sample/FOV_001/default"
    task = next(t for t in run.data["tasks"] if t["id"] == target)
    if damage == "deleted-task":
        for name in [*task["outputs"], task["record"], task["log"]]:
            path = run.out / name
            shutil.rmtree(path) if path.is_dir() else path.unlink()
    elif damage == "deleted-index":
        (run.out / "report/index.html").unlink()
    else:
        path = run.out / "fovs/FOV_001/molecules.csv"
        if damage == "truncated-molecules":
            payload = path.read_bytes().splitlines(keepends=True)
            assert len(payload) > 1
            path.write_bytes(b"".join(payload[:-1]))
        else:
            path.unlink()
    run.invoke(E.downstream(run.data, {target}))
    E.compare(run, uninterrupted, E.snapshot(run), name=f"x10-{damage}.json")


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("case", ["processing", "export", "restored-twice", "raw-size", "raw-older", "fingerprint"])
def test_x11_stale_results_rerun_only_readers_and_downstream(tmp_path, engine, case):
    import copy
    import os
    from starfinder.__main__ import main
    from starfinder.workflow import verify
    run = E.EngineRun(tmp_path / case, engine)
    run.invoke({t["id"] for t in run.data["tasks"]})
    if case == "fingerprint":
        task = next(t for t in run.data["tasks"] if t["id"] == FOV_TASK)
        path = run.out / task["record"]
        record = json.loads(path.read_text())
        record["workflow"]["fingerprint"] = "sha256:" + "0" * 64
        path.write_text(json.dumps(record))
        assert main(["workflow", "verify", "--manifest", str(run.manifest), "--final"]) == 1
        result = verify(run.manifest, final=True)
        E.evidence(run.work, "x11-edited-fingerprint.json", result)
        assert not result["passed"] and FOV_TASK in result["incomplete"]
        assert "fingerprint" in " ".join(result["tasks"][FOV_TASK]["problems"])
        return
    original = copy.deepcopy(run.config)
    failures = []
    for iteration, changed in enumerate([True, False, True, False] if case == "restored-twice" else [True]):
        config = copy.deepcopy(original)
        if case.startswith("raw-"):
            raw = X.input_root(run.work) / "round2/FOV_002/ch01.ome.tif"
            declaring = {t["id"] for t in run.data["tasks"] if "input:round2/FOV_002/ch01.ome.tif" in t["inputs"]}
            assert declaring == {"summarize/sample/percentile_normalization/FOV_002", "fov/sample/FOV_002"}
            affected = E.downstream(run.data, declaring)
            stat = raw.stat()
            if case == "raw-size":
                raw.write_bytes(raw.read_bytes() + b"\n")
                assert raw.stat().st_size != stat.st_size
            else:
                import tifffile
                with tifffile.TiffFile(raw) as tif:
                    assert int(tif.pages[0].compression) == 1
                    offset = tif.pages[0].dataoffsets[0]
                payload = bytearray(raw.read_bytes())
                payload[offset] ^= 1
                raw.write_bytes(payload)
                os.utime(raw, ns=(stat.st_atime_ns, stat.st_mtime_ns - 86_400 * 10**9))
                assert raw.stat().st_size == stat.st_size and raw.stat().st_mtime_ns < stat.st_mtime_ns
        else:
            block = "export" if case == "export" else "processing"
            affected = E.downstream(run.data, {t["id"] for t in run.data["tasks"] if block in t["blocks"]})
            if block == "export":
                assert affected == {"export/sample", "report/sample", "compare"}
                config["export"]["previews"] = False
            elif changed:
                config["processing"]["spot_finding"]["threshold_value"] = 6.0
        run.replan(config)
        stale = verify(run.manifest)
        E.evidence(run.work, f"x11-{iteration}-stale.json", stale)
        assert set(stale["incomplete"]) == affected
        assert all("fingerprint" in " ".join(stale["tasks"][task]["problems"]) for task in affected)
        run.invoke(affected)
        failures.extend(E.fresh_comparison(run, E.snapshot(run), name=f"x11-{iteration}-fresh.json", rerun_tasks=affected))
    assert not failures, sorted(set(failures))


# W-760: preserve both the completed output tree and Snakemake's metadata at the
# same paths, so each damage case starts from one identical real-engine run.
@pytest.fixture(scope="module")
def snakemake_damage_base(tmp_path_factory):
    run = E.EngineRun(tmp_path_factory.mktemp("snakemake-damage") / "work", "snakemake")
    run.invoke({t["id"] for t in run.data["tasks"]})
    uninterrupted = E.snapshot(run)
    saved = run.work / "undamaged"
    shutil.copytree(run.out, saved / "output", symlinks=True)
    shutil.copytree(run.work / "snakemake", saved / "engine", symlinks=True)
    return run, saved, uninterrupted


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("triggers", ["default", "profile"])
@pytest.mark.parametrize("damage", ["none", "deleted-molecules", "deleted-fov", "deleted-index",
                                    "deleted-directory", "deleted-record", "deleted-log",
                                    "truncated-newer", "truncated-preserved"])
def test_x10_snakemake_missing_outputs_and_damage_limits(snakemake_damage_base, damage, triggers):
    """Missing permanent outputs rerun their producer; logs and resized files remain verify's job."""
    import os
    import re
    import time
    from collections import Counter
    from starfinder.__main__ import main
    from starfinder.workflow import verify
    run, saved, uninterrupted = snakemake_damage_base
    for source, target in ((saved / "output", run.out), (saved / "engine", run.work / "snakemake")):
        shutil.rmtree(target)
        shutil.copytree(source, target, symlinks=True)
    task = next(t for t in run.data["tasks"] if t["id"] == FOV_TASK)
    target = FOV_TASK
    expected = set()
    if damage == "deleted-index":
        (run.out / "report/index.html").unlink()
        target = "compare"
    elif damage == "deleted-directory":
        target = "assign/sample/FOV_001/default"
        directory_task = next(t for t in run.data["tasks"] if t["id"] == target)
        directory = next(o for o in directory_task["outputs"] if o.endswith("/"))
        shutil.rmtree(run.out / directory)
    elif damage == "deleted-fov":
        shutil.rmtree(run.out / "fovs/FOV_001")
    elif damage == "deleted-record":
        (run.out / task["record"]).unlink()
    elif damage == "deleted-log":
        (run.out / task["log"]).unlink()
    elif damage != "none":
        path = run.out / "fovs/FOV_001/molecules.csv"
        if damage == "deleted-molecules":
            path.unlink()
        else:
            stat = path.stat()
            lines = path.read_bytes().splitlines(keepends=True)
            assert len(lines) > 1
            path.write_bytes(b"".join(lines[:-1]))
            assert path.stat().st_size < stat.st_size
            # Explicitly newer than every completed output, without a sleep or
            # dependence on the filesystem's timestamp resolution.
            stamp = max(time.time_ns(), max(p.stat().st_mtime_ns for p in run.out.rglob("*")
                                            if p.is_file())) + 2_000_000_000
            os.utime(path, ns=(stat.st_atime_ns, stamp if damage == "truncated-newer" else stat.st_mtime_ns))
    if damage.startswith("deleted-") and damage != "deleted-log":
        roots = {target}
        if damage == "deleted-fov":
            roots = {t["id"] for t in run.data["tasks"] if any(
                o.startswith("fovs/FOV_001/") for o in t["outputs"])}
        expected = E.downstream(run.data, roots)
    elif damage == "truncated-newer":
        readers = {t["id"] for t in run.data["tasks"] if "fovs/FOV_001/molecules.csv" in t["inputs"]}
        expected = E.downstream(run.data, readers)
        assert FOV_TASK not in expected
    before = {t["id"]: (run.out / t["record"]).read_bytes() for t in run.data["tasks"]
              if (run.out / t["record"]).is_file()}
    done = E.snakemake(run.work, run.path, "--rerun-incomplete", *(E.TRIGGERS if triggers == "profile" else []))
    tag = f"x10-snakemake-{damage}-{triggers}"
    # Save the engine's full output even on the base revision where scheduling
    # assertions fail; limits deliberately do not use EngineRun.invoke's check.
    E.evidence(run.work, tag + "-engine.json", {"returncode": done.returncode, "stdout": done.stdout,
               "stderr": done.stderr, "expected": sorted(expected)})
    assert done.returncode == 0, done.stdout[-6000:] + done.stderr[-6000:]
    tasks = Counter(re.findall(r" --task (\S+) ", done.stdout + done.stderr))
    assert set(tasks) == expected and all(count == 1 for count in tasks.values()), (tasks, expected)
    for task_id, payload in before.items():
        path = run.out / next(t["record"] for t in run.data["tasks"] if t["id"] == task_id)
        assert (path.read_bytes() != payload) == (task_id in expected), task_id
    limited = damage in ("deleted-log", "truncated-newer", "truncated-preserved")
    status = main(["workflow", "verify", "--manifest", str(run.manifest), "--final"])
    result = verify(run.manifest, final=True)
    E.evidence(run.work, tag + "-verify.json", result)
    assert status == (1 if limited else 0) and result["passed"] is (not limited), result
    if limited:
        assert FOV_TASK in result["incomplete"]
        problems = " ".join(result["tasks"][FOV_TASK]["problems"])
        assert ("is missing" if damage == "deleted-log" else "bytes") in problems
    else:
        E.compare(run, uninterrupted, E.snapshot(run), name=tag + "-comparison.json")


# W-758: keep one complete run per engine and restore copies of its output tree
# between cases; only export/report/compare need real jobs after that setup.
@pytest.fixture(scope="module", params=ENGINES)
def export_engine_base(tmp_path_factory, request):
    run = E.EngineRun(tmp_path_factory.mktemp("export-record") / "work", request.param)
    run.invoke({t["id"] for t in run.data["tasks"]})
    saved = run.out.with_name(run.out.name + "-baseline")
    shutil.copytree(run.out, saved, symlinks=True)
    return run, saved


@pytest.fixture
def export_engine(export_engine_base):
    import copy
    run, saved = export_engine_base
    shutil.rmtree(run.out)
    shutil.copytree(saved, run.out, symlinks=True)
    run.replan(copy.deepcopy(run.config))
    yield run


EXPORT_DOWNSTREAM = {"export/sample", "report/sample", "compare"}


def _export_files(run):
    return {p for p in E.tree_names(run.out) if p.startswith("export/") and not E.is_marker(run, p, run.engine)}


@pytest.mark.integration
@pytest.mark.slow
def test_export_record_part_off_matches_a_fresh_export(export_engine):
    import copy
    run = export_engine
    config = copy.deepcopy(run.config)
    config["export"]["fuse"] = False
    run.replan(config)
    run.invoke(EXPORT_DOWNSTREAM)
    rerun = _export_files(run)
    # Fresh export on the same completed upstream results, with no saved export
    # record in either location; do not repeat the complete FOV pipeline.
    shutil.rmtree(run.out / "export")
    (run.out / "workflow/exports/sample.json").unlink(missing_ok=True)
    for task_id in EXPORT_DOWNSTREAM:
        _layout.task_record(run.out, task_id).unlink()
    run.invoke(EXPORT_DOWNSTREAM)
    fresh = _export_files(run)
    E.evidence(run.work, "w758-part-off.json", {"engine": run.engine, "rerun": sorted(rerun),
               "fresh": sorted(fresh), "differences": sorted(rerun ^ fresh)})
    assert rerun == fresh


@pytest.mark.integration
@pytest.mark.slow
def test_export_record_failed_attempt_is_kept_for_retry(export_engine, tmp_path):
    import copy
    import os
    import subprocess
    import sys
    from starfinder.export._record import PARTS, listed_files
    run = export_engine
    config = copy.deepcopy(run.config)
    config["export"].update(images=[f"attempt=file:{run.out}/fovs/{{fov_id}}/reference.tif"], previews=False)
    run.replan(config)
    hook = tmp_path / "fail-hook"
    hook.mkdir()
    (hook / "sitecustomize.py").write_text(
        "import sys\n"
        "if '--task' in sys.argv and sys.argv[sys.argv.index('--task') + 1] == 'export/sample':\n"
        " import starfinder.dataset._export as export\n"
        " original = export.fuse_images\n"
        " def fail(*args, **kwargs):\n"
        "  original(*args, **kwargs)\n"
        "  raise RuntimeError('W-758 injected failure after fusion')\n"
        " export.fuse_images = fail\n")
    task = next(t for t in run.data["tasks"] if t["id"] == "export/sample")
    command = [sys.executable, "-m", "starfinder", *task["command"][1:], "--engine", run.engine]
    environment = dict(os.environ, PYTHONPATH=str(hook) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    failed = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=180)
    folder = run.out / "export/sample"
    log = _layout.task_log(run.out, "export/sample").read_text()
    E.evidence(run.work, "w758-failed-attempt.json", {"command": command, "returncode": failed.returncode,
               "stdout": failed.stdout, "stderr": failed.stderr, "task_log": log})
    assert failed.returncode == 1 and "W-758 injected failure after fusion" in log
    assert not (folder / "export.json").exists()
    assert list(folder.glob("attempt*.tif")), "failure must leave a newly named, previously listed fused image"
    foreign = folder / "foreign.txt"
    foreign.write_bytes(b"not owned by Dataset.export")
    config["export"]["fuse"] = False
    run.replan(config)
    run.invoke(EXPORT_DOWNSTREAM)
    record = json.loads((folder / "export.json").read_text())
    listed = {"export.json", *(name for part in PARTS for name in listed_files(record, part))}
    unlisted = {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()} - listed
    E.evidence(run.work, "w758-retry.json", {"engine": run.engine, "listed": sorted(listed),
               "unlisted": sorted(unlisted), "foreign": foreign.read_text()})
    assert unlisted == {"foreign.txt"}
    assert foreign.read_bytes() == b"not owned by Dataset.export"


@pytest.mark.integration
@pytest.mark.slow
def test_export_record_saved_assembly_is_reused(export_engine):
    import copy
    run = export_engine
    folder = run.out / "export/sample"
    before = {p.relative_to(folder).as_posix(): p.read_bytes() for p in (folder / "tables").glob("*")}
    assert before and (folder / "sample.h5ad").is_file()
    before["sample.h5ad"] = (folder / "sample.h5ad").read_bytes()
    config = copy.deepcopy(run.config)
    config["export"].update(assemble=False, fuse=False, fuse_labels=True, previews=False)
    run.replan(config)
    run.invoke(EXPORT_DOWNSTREAM)
    record = json.loads((folder / "export.json").read_text())
    E.evidence(run.work, "w758-reused-assembly.json", record)
    assert record["parts"]["assemble"] == "reused" and record["parts"]["fuse_labels"] == "written"
    assert {name: (folder / name).read_bytes() for name in before} == before
