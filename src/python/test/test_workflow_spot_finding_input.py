"""The saved spot-finding input (W-889): ``processing.spot_finding_input`` on one FOV of ``tiny_sample``.

The stage ``registered`` holds each round after its preprocessing, registration and post-registration steps, the
stack that spot finding reads. A run with ``processing.checkpoints: [registered, ...]`` saves it (the set); a later
run whose processing block names that run's output root in ``spot_finding_input`` loads the FOV's registered stage
and runs spot finding and the steps after it. Its outputs equal those of a run from the raw images with the same
configuration, also when the spot-finding settings differ from the set's run, and under another package version. A set
whose preprocessing, registration, rounds, channels or code identity does not fit refuses with exit 2, and the plan
refuses the settings the key excludes. The code identity is patched (``starfinder._records.code``), so that the tests
do not depend on the state of the checkout.
"""
import copy
import hashlib
import json
import shutil

import pytest

from starfinder import _layout
from starfinder.workflow import plan
from starfinder.workflow._plan import PlanRefused

from . import workflow_examples as X

pytestmark = [pytest.mark.workflow, pytest.mark.dataset, pytest.mark.contract]

FOV = "FOV_001"
FOV_TASK = f"fov/sample/{FOV}"
#: The fov task's outputs compared byte for byte with a run from the raw images.
FILES = ["molecules.csv", "reference.tif", "checkpoints/candidates.json", "checkpoints/pre_qc.json",
         "checkpoints/candidates.csv", "checkpoints/pre_qc.csv"]
#: A post-registration step, so that the saved stage is the stack after it, as spot finding reads it.
POST = [{"method": "reconstruction", "radius_yx": 2}]


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


def _config(work, output_id, *, checkpoints=("candidates", "pre_qc"), saved=None, **processing):
    """One FOV with the molecules-only processing block, a reconstruction after registration, ``checkpoints``,
    ``spot_finding_input: <saved>`` when given and the processing keys ``processing`` (merged into mappings)."""
    config = X.single_fov(work, output_id)
    p = config["processing"]
    p["preprocessing"]["post_registration"] = copy.deepcopy(POST)
    p["checkpoints"] = list(checkpoints)
    for key, value in processing.items():
        p[key] = dict(p[key], **value) if isinstance(value, dict) and isinstance(p.get(key), dict) else value
    if saved is not None:
        p["spot_finding_input"] = str(saved)
    return config


def _run(work, name, config):
    """Plan ``config`` and run its fov task; (exit status, manifest, FOV folder)."""
    plan(X.write_config(work, name, config))
    manifest = X.output_root(work, config["output_id"]) / "workflow" / "manifest.json"
    return X.run_task(manifest, FOV_TASK), manifest, _layout.fov_dir(manifest.parents[1], FOV)


def _digests(folder):
    return {name: hashlib.sha256((folder / name).read_bytes()).hexdigest() for name in FILES}


def _saved(work, **processing):
    """The set: a run from the raw images that saves the stage registered besides candidates and pre_qc; its output
    root and FOV folder."""
    status, manifest, folder = _run(work, "set", _config(work, "set", checkpoints=("registered", "candidates",
                                                                                     "pre_qc"), **processing))
    assert status == 0
    return manifest.parents[1], folder


def test_a_run_from_the_saved_set_equals_a_run_from_the_raw_images(work):
    root, full = _saved(work)
    header = json.loads((full / "checkpoints" / "registered" / "transforms.json").read_text())
    assert header["preprocessing"]["recipe"]["post_registration"] == ["reconstruction"]
    status, manifest, folder = _run(work, "from-set", _config(work, "from-set", saved=root))
    assert status == 0
    assert _digests(folder) == _digests(full)
    task = next(t for t in json.loads(manifest.read_text())["tasks"] if t["id"] == FOV_TASK)
    # The task declares the set's run.json and registered stage record of its FOV in place of the raw images.
    assert sorted(i for i in task["inputs"] if not i.endswith("genes.csv")) == sorted([
        str(full / "run.json"), str(full / "checkpoints" / "registered" / "transforms.json")])
    record, saved = (json.loads((f / "run.json").read_text()) for f in (folder, full))
    assert not (folder / "checkpoints" / "registered").exists() and list(record["checkpoints"]) == [
        "candidates", "pre_qc"]
    assert record["spot_finding_input"] == {
        "path": str(root), "run_record": {"path": str(full / "run.json"),
                                          "sha256": hashlib.sha256((full / "run.json").read_bytes()).hexdigest()},
        "registered": {"path": str(full / "checkpoints" / "registered" / "transforms.json"),
                       "sha256": saved["checkpoints"]["registered"]["sha256"]},
        "code": saved["code"], "ended_at": saved["ended_at"]}
    for key in ("inputs", "preprocessing", "registration", "transforms", "counts"):
        assert record[key] == saved[key], key
    assert record["config"]["pipeline"] == saved["config"]["pipeline"]
    # The set's steps that made the stack, then this run's: the same steps as the set's run, but its writes of the
    # registered stage.
    steps = lambda r: sorted((s["name"], s["round"] or "") for s in r["steps"])  # noqa: E731
    assert steps(record) == [s for s in steps(saved) if s[0] != "write_checkpoint:registered"]


def test_a_run_from_the_saved_set_with_other_spot_finding_settings_equals_a_run_from_the_raw_images(work):
    """The use the set is kept for: a change of spot finding runs from the set, not from the raw images."""
    root, _ = _saved(work)
    changed = {"spot_finding": {"threshold_value": 4.0}}
    status, _, full = _run(work, "full", _config(work, "full", **changed))
    assert status == 0
    status, _, folder = _run(work, "from-set", _config(work, "from-set", saved=root, **changed))
    assert status == 0
    assert _digests(folder) == _digests(full)
    record = json.loads((folder / "run.json").read_text())
    assert record["config"]["pipeline"]["spot_finding"]["threshold_value"] == 4.0


def test_a_set_made_by_another_package_version_is_read_and_its_code_identity_recorded(work, monkeypatch):
    root, full = _saved(work)
    X.fix_code(monkeypatch, version="0.2.1.dev3+gfedcba9", git_commit="fedcba9876543210fedcba9876543210fedcba98")
    status, _, folder = _run(work, "from-set", _config(work, "from-set", saved=root))
    assert status == 0
    assert _digests(folder) == _digests(full)
    record = json.loads((folder / "run.json").read_text())
    assert record["spot_finding_input"]["code"]["version"] == X.CLEAN_CODE["version"]
    assert record["code"]["version"] == "0.2.1.dev3+gfedcba9"


@pytest.mark.parametrize("case", ["preprocessing", "registration", "dirty"])
def test_a_set_that_does_not_fit_the_task_is_refused(work, monkeypatch, capsys, case):
    if case == "dirty":
        X.fix_code(monkeypatch, git_dirty=True)
    root, full = _saved(work)
    X.fix_code(monkeypatch)
    changes, expected = {
        "preprocessing": ({"preprocessing": {"post_registration": [{"method": "reconstruction", "radius_yx": 1}]}},
                          "config.pipeline.preprocessing.post_registration.0.config.radius_yx is 2 (recorded), "
                          "1 (now)"),
        "registration": ({"registration": {"signal": {"mode": "sum"}}},
                         "config.pipeline.registration.signal.mode is max (recorded), sum (now)"),
        "dirty": ({}, "code.git_dirty is true (recorded)")}[case]
    status, manifest, folder = _run(work, "from-set", _config(work, "from-set", saved=root, **changes))
    assert status == 2
    message = capsys.readouterr().err
    assert f"processing.spot_finding_input {full / 'run.json'} does not match this task: " in message
    assert expected in message
    assert not folder.exists()


def test_the_plan_refuses_a_set_without_the_files_of_a_fov(work, tmp_path):
    with pytest.raises(PlanRefused, match=r"processing.spot_finding_input, FOV FOV_001: the file .*run.json does not "
                                          r"exist"):
        plan(X.write_config(work, "from-set", _config(work, "from-set", saved=tmp_path / "empty")))


@pytest.mark.parametrize("case, expected", [
    ("tiled", "processing.spot_finding_input is read in processing.mode direct only"),
    ("no_spot_finding", "add processing.spot_finding"),
    ("registered", "remove registered from processing.checkpoints"),
    ("segmentation", "and the segmentation inputs [('nucleus', 'round2')], which read registered sequencing rounds"),
    ("resume", "set execution.resume_from_checkpoint: false"),
])
def test_the_plan_refuses_the_settings_the_key_excludes(work, case, expected):
    config = _config(work, "from-set", saved=work / "set")
    p = config["processing"]
    if case == "tiled":
        p["mode"] = "tiled"
    elif case == "no_spot_finding":
        for key in ("spot_finding", "reads_extraction", "reads_filtration", "decoding", "deduplication"):
            p.pop(key)
    elif case == "registered":
        p["checkpoints"] = ["registered", "candidates"]
    elif case == "segmentation":
        config["segmentation"] = {"runs": [{"name": "nucleus", "target": "nucleus", "method": "stardist",
                                            "model": "2D_versatile_fluo",
                                            "input_recipe": {"model_input": {"factor": 1.0}},
                                            "inputs": [{"role": "nuclear", "round": "round2", "channel": "ch00"}]}]}
    else:
        config["execution"] = {"resume_from_checkpoint": True}
    with pytest.raises(PlanRefused, match="processing.spot_finding_input") as refused:
        plan(X.write_config(work, "from-set", config))
    assert expected in str(refused.value)


def test_without_the_key_the_resolved_processing_block_has_none(work):
    """The key has no default the plan writes, so the hashes of the configurations without it are unchanged."""
    plan(X.write_config(work, "plain", _config(work, "plain")))
    resolved = json.loads((X.output_root(work, "plain") / "workflow" / "resolved.json").read_text())
    assert "spot_finding_input" not in resolved["configuration"]["processing"]
