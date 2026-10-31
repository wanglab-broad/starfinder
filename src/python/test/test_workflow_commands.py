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
    X.write_supplied(work)
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


def _edit_resolved(manifest, change):
    """Change the resolved configuration with ``change(data)`` and give the manifest its new hash, as a package that
    translates differently from the one that planned would see it."""
    resolved = manifest.parent / "resolved.json"
    data = json.loads(resolved.read_text())
    change(data)
    text = json.dumps(data, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    resolved.write_bytes(text)
    record = json.loads(manifest.read_text())
    record["configuration"]["sha256"] = "sha256:" + hashlib.sha256(text).hexdigest()
    manifest.write_text(json.dumps(record, indent=1) + "\n")


def _set(*path, value):
    def change(data):
        node = data["configuration"]["api"]
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value(node[path[-1]])
    return change


#: Each kind with a command and a translator, one block its map names and a value of that block's api changed: the
#: task refuses (exit 2) before its needs and inputs are looked at, naming the block. The export's case changes
#: api.processing's codebook encoding, which only the molecule routes read.
KIND_REFUSALS = {
    "summarize": ("per-fov", "summarize/sample/percentile_normalization/FOV_001",
                  _set("processing", "pipeline", "spot_finding", "threshold_value", value=lambda v: 9.0), "processing"),
    "merge": ("per-fov", "merge/sample/percentile_normalization",
              _set("processing", "checkpoints", "stages", value=lambda v: ["pre_qc"]), "processing"),
    "fov": ("per-fov", "fov/sample/FOV_002", _set("processing", "execution", "mode", value=lambda v: "streaming"),
            "processing"),
    "morphology": ("per-fov", "morphology/sample/FOV_001",
                   _set("morphology", "rotation_degrees", value=lambda v: 90.0), "morphology"),
    "segment": ("per-fov", "segment/sample/FOV_001/cell", _set("segmentation", "device", value=lambda v: "cuda"),
                "segmentation"),
    "assign": ("per-fov", "assign/sample/FOV_001/default", _set("assignment", "previews", value=lambda v: not v),
               "assignment"),
    "stitch": ("per-fov", "stitch/sample", _set("stitching", "config", "quality_threshold", value=lambda v: 0.5),
               "stitching"),
    "bigstitcher_macro": ("after-fusion", "bigstitcher_macro/sample",
                          _set("stitching", "input", value=lambda v: "reference_stain"), "stitching"),
    "bigstitcher_engine": ("after-fusion", "bigstitcher_engine/sample",
                           _set("stitching", "fallback", value=lambda v: "nominal"), "stitching"),
    "bigstitcher_import": ("after-fusion", "bigstitcher_import/sample",
                           _set("stitching", "review", "max_residual_um", value=lambda v: 2.0), "stitching"),
    "export": ("molecules-only", "export/sample",
               _set("processing", "codebook", "encoding", "reverse_bases", value=lambda v: not v), "processing"),
    "export (its own block)": ("per-fov", "export/sample", _set("export", "route", value=lambda v: "after_fusion"),
                               "export"),
    "report": ("per-fov", "report/sample", _set("report", "config", "positive_threshold", value=lambda v: 3), "report"),
    "compare": ("per-fov", "compare", _set("report", "config", "jpeg_quality", value=lambda v: 50), "report"),
}


@pytest.mark.parametrize("case", sorted(KIND_REFUSALS))
def test_each_kind_refuses_a_translation_that_differs_from_the_plans(work, capsys, case):
    variant, task, change, block = KIND_REFUSALS[case]
    plan(X.write_config(work, variant, X.variants(work)[variant]))
    manifest = X.output_root(work, f"example-{variant}") / "workflow" / "manifest.json"
    _edit_resolved(manifest, change)
    _refused(manifest, task, capsys, 2, f"translates the {block} block differently from the plan",
             f"api.{block} differs in")


def test_the_report_refuses_a_changed_assembly_assignment(work, capsys):
    """The report reads assembly.assignment from the resolved configuration, which no translator owns: a changed
    value fails the common check of the file's hash."""
    plan(X.write_config(work, "per-fov", X.variants(work)["per-fov"]))
    manifest = X.output_root(work, "example-per-fov") / "workflow" / "manifest.json"
    resolved = manifest.parent / "resolved.json"
    data = json.loads(resolved.read_text())
    assert data["configuration"]["assembly"]["assignment"] == "default"
    data["configuration"]["assembly"]["assignment"] = "other"
    resolved.write_text(json.dumps(data, sort_keys=True, separators=(",", ":")) + "\n")
    _refused(manifest, "report/sample", capsys, 2, "resolved configuration", "plan the run again")


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
