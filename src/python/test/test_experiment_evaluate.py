"""``starfinder experiment evaluate``: saved workflow outputs, checkpoints and trial outputs, read without processing.

design/specs/benchmark.md, "Proposed evaluate-only contract"; design/validation/benchmark.md, "Evaluation, records and
retirement checks". One FOV of W-642's fixture (``workflow_examples.tiny_sample``: 5 × 36 × 36, four rounds of four
channels) is run once by the workflow with the ``registered``, ``candidates`` and ``pre_qc`` checkpoints, and one
translation trial on an 8 × 24 × 24 pair. Every evaluation runs with the processing entry points patched to raise; the
bytes of every source file are hashed before and after. Corrupt, missing and version-2 sources, a failed trial, an
incomplete task, a metric that raises and an undefined metric keep their distinct outcomes.
"""
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest
import yaml

from starfinder.__main__ import main

from . import workflow_examples as X

pytestmark = [pytest.mark.benchmark, pytest.mark.contract]

FOV_TASK = "fov/sample/FOV_001"
CHECKPOINTS = "fovs/FOV_001/checkpoints"
MOLECULES = "fovs/FOV_001/molecules.csv"
#: The processing entry points that evaluation must never call: (module, attribute).
PROCESSING = [
    ("starfinder.dataset.fov", "FOV.run"),
    ("starfinder.registration", "estimate_transform"), ("starfinder.registration._api", "estimate_transform"),
    ("starfinder.registration", "apply_transform"), ("starfinder.registration._api", "apply_transform"),
    ("starfinder.preprocessing.steps", "run_step"),
    ("starfinder.spot_finding", "find_spots"),
    ("starfinder.segmentation", "segment"), ("starfinder.segmentation._segment", "segment"),
    *(("starfinder.readout", name) for name in ("extract_intensities", "decode_reads", "decode_direct",
                                                  "filter_reads", "score_reads", "deduplicate_reads")),
    ("starfinder.assignment", "assign_molecules"),
    *(("starfinder.synthetic", name) for name in ("generate_amplicon_dataset", "generate_amplicon_scene",
                                                    "generate_registration_pair", "generate_sample_scene",
                                                    "generate_cellular_scene")),
    ("starfinder.benchmark._trial", "_run"), ("starfinder.benchmark._trial", "_main"),
    ("starfinder.workflow._cli", "run"),
]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _tree(*roots):
    """{path: SHA-256} of every file under the roots."""
    return {str(p): _sha(p) for root in roots for p in sorted(Path(root).rglob("*")) if p.is_file()}


def _no_processing(monkeypatch):
    import importlib

    def boom(*args, **kwargs):
        raise AssertionError("evaluation called a processing entry point")
    for module, attribute in PROCESSING:
        target = importlib.import_module(module)
        *owner, name = attribute.split(".")
        for part in owner:
            target = getattr(target, part)
        monkeypatch.setattr(target, name, boom)


@pytest.fixture(scope="module")
def workflow_run(tmp_path_factory):
    """One FOV run by the workflow with every checkpoint stage, once for the module."""
    work = tmp_path_factory.mktemp("evaluate") / "work"
    X.golden().write_fixture(work)
    with pytest.MonkeyPatch.context() as patch:
        X.fix_code(patch)
        config = X.single_fov(work)
        config["processing"]["checkpoints"] = ["registered", "candidates", "pre_qc"]
        from starfinder.workflow import plan
        plan(X.write_config(work, "single", config))
        manifest = X.output_root(work, "single") / "workflow" / "manifest.json"
        assert X.run_task(manifest, FOV_TASK) == 0
    return work


@pytest.fixture
def work(workflow_run, tmp_path):
    """A copy of the module's run in this test's folder, the manifest's roots moved with it."""
    copy = tmp_path / "work"
    shutil.copytree(workflow_run, copy, symlinks=True)
    manifest = X.output_root(copy, "single") / "workflow" / "manifest.json"
    manifest.write_text(manifest.read_text().replace(str(workflow_run), str(copy)))
    return copy


def _output(work):
    return X.output_root(work, "single")


@pytest.fixture
def trial(tmp_path):
    """A succeeded translation trial (moving = reference rolled by one voxel in X), run by ``starfinder trial run``."""
    folder = tmp_path / "trial"
    folder.mkdir()
    z, y, x = np.indices((8, 24, 24))
    image = (100 * np.exp(-((z - 3)**2 / 2 + (y - 9)**2 / 8 + (x - 10)**2 / 8))
             + 60 * np.exp(-((z - 5)**2 / 2 + (y - 17)**2 / 5 + (x - 17)**2 / 5))).astype(np.float32)
    np.save(folder / "reference.npy", image)
    np.save(folder / "moving.npy", np.roll(image, 1, axis=2))
    config = folder / "trial.json"
    config.write_text(json.dumps(dict(method=dict(module="registration", method="translation"),
                                      inputs=dict(reference="reference.npy", moving="moving.npy"),
                                      metadata=dict(frame_id="trial/reference"), seed=3,
                                      root_output_path="out")))
    assert main(["trial", "run", "--config", str(config)]) == 0
    return folder / "out"


def _write(path, value):
    Path(path).write_text(yaml.safe_dump(value, sort_keys=False))
    return Path(path)


def _truth(tmp_path, name, frame):
    path = tmp_path / "truth" / name
    path.parent.mkdir(exist_ok=True)
    frame.to_csv(path, index=False)
    return path


VOXEL = {"metadata": {"frame_id": "evaluation/voxel"}}


def _matching(units="voxel"):
    return {"policy": "greedy", "threshold": 0.5, "boundary": "inclusive", "units": units}


def _all_sources(work, trial, tmp_path):
    """The five sources and the metric that reads each, with their known answers built from the saved files."""
    out = _output(work)
    molecules = pd.read_csv(out / MOLECULES)
    candidates = pd.read_csv(out / CHECKPOINTS / "candidates.csv")
    truth_molecules = _truth(tmp_path, "molecules.csv", molecules[["z", "y", "x", "gene_id"]].head(5))
    truth_spots = _truth(tmp_path, "spots.csv", candidates[["z", "y", "x"]].head(3))
    shift = tmp_path / "truth" / "shift.json"
    shift.write_text(json.dumps({"displacement": [0.0, 0.0, 1.0]}))
    files = {key: {"path": str(path), "sha256": _sha(path)}
             for key, path in (("molecules", truth_molecules), ("spots", truth_spots), ("shift", shift))}
    sources = {"sources": [
        {"name": "run", "kind": "workflow", "manifest": str(out / "workflow" / "manifest.json")},
        *({"name": stage, "kind": "checkpoint", "path": str(out / CHECKPOINTS), "stage": stage}
          for stage in ("registered", "candidates", "pre_qc")),
        {"name": "trial", "kind": "trial", "path": str(trial)},
        {"name": "truth", "kind": "files", "files": files}]}
    metrics = {"metrics": [
        {"name": "molecules", "metric": "M70", "function": "matching.evaluate_molecules", "parameters": _matching(),
         "arguments": {"reference": {"source": "truth", "file": "molecules"},
                       "observed": {"source": "run", "task": FOV_TASK, "output": MOLECULES},
                       "reference_metadata": VOXEL, "observed_metadata": VOXEL}},
        {"name": "spots", "metric": "M03", "function": "spot_finding.f1_score", "parameters": _matching(),
         "arguments": {"detected": {"source": "candidates", "key": "spot_result.spots", "columns": ["z", "y", "x"]},
                       "truth": {"source": "truth", "file": "spots", "columns": ["z", "y", "x"]},
                       "reference_metadata": VOXEL, "observed_metadata": VOXEL}},
        {"name": "reads", "metric": "M17", "function": "readout.read_status_summary",
         "arguments": {"reads": {"source": "pre_qc", "key": "decoding_result.table"}}},
        {"name": "registered", "function": "registration.normalized_cross_correlation",
         "arguments": {"ref": {"source": "registered", "key": "images.round1"},
                       "observed": {"source": "registered", "key": "images.round1"}}},
        {"name": "translation", "function": "registration.evaluate_translation",
         "parameters": {"units": "voxel", "tolerance": 0.5},
         "arguments": {"detected": {"mapping": {"moving": {"source": "trial", "output": "transform.json",
                                                            "key": "transform.displacement_zyx"}}},
                       "truth": {"mapping": {"moving": {"source": "truth", "file": "shift", "key": "displacement"}}},
                       "reference_metadata": {"source": "trial", "metadata": True},
                       "observed_metadata": {"source": "trial", "metadata": True}}}]}
    answers = {"molecules": dict(reference_fraction=1.0, observed_fraction=5 / len(molecules)),
               "spots": dict(recall=1.0, precision=3 / len(candidates)),
               "reads": dict(assigned=int(pd.read_csv(out / CHECKPOINTS / "pre_qc.csv").call_status.eq(
                   "assigned").sum()))}
    return sources, metrics, answers


def _evaluate(tmp_path, sources, metrics, name="evaluation"):
    status = main(["experiment", "evaluate", "--sources", str(_write(tmp_path / f"{name}-sources.yaml", sources)),
                   "--metrics", str(_write(tmp_path / f"{name}-metrics.yaml", metrics)),
                   "--output", str(tmp_path / name)])
    record = json.loads((tmp_path / name / "evaluation.json").read_text())
    return status, record, {m["name"]: m for m in record["metrics"]}


def test_saved_only_and_record_identity(work, trial, tmp_path, monkeypatch):
    sources, metrics, answers = _all_sources(work, trial, tmp_path)
    before = _tree(work, trial.parent, tmp_path / "truth")
    _no_processing(monkeypatch)
    status, record, by_name = _evaluate(tmp_path, sources, metrics)
    assert _tree(work, trial.parent, tmp_path / "truth") == before
    assert not (_output(work) / "workflow" / "verify.json").exists()
    assert status == 0 and record["status"] == "complete"
    assert record["format"] == "starfinder.experiment.evaluation/1"
    assert record["counts"] == {"sources": 6, "failed_sources": 0, "metrics": 5, "computed": 5, "skipped": 0,
                                "unavailable": 0, "failed": 0}
    assert set(record["software"]) == {"code", "environment"}
    for key in ("sources", "metrics"):
        assert record["inputs"][key]["sha256"] == _sha(tmp_path / f"evaluation-{key}.yaml")
    # Every consumed file is listed with the SHA-256 of its bytes.
    consumed = [item for source in record["sources"] for item in source["consumed"]]
    assert consumed and all(item["sha256"] == _sha(item["path"]) for item in consumed)
    paths = {Path(item["path"]).name for item in consumed}
    assert {"manifest.json", "fov.sample.FOV_001.json", "molecules.csv", "candidates.json", "candidates.csv",
            "pre_qc.json", "pre_qc.csv", "transforms.json", "round1.ome.tif", "record.json",
            "transform.json"} <= paths
    run = next(s for s in record["sources"] if s["name"] == "run")
    assert run["tasks"][FOV_TASK]["complete"] and run["incomplete"] == []
    checkpoints = {s["name"]: s for s in record["sources"] if s["kind"] == "checkpoint"}
    assert {s["format_version"] for s in checkpoints.values()} == {3}
    assert all(s["limitations"] == [] for s in checkpoints.values())
    assert set(checkpoints["registered"]["frames"]) == {"round1", "round2", "round3", "round4"}
    trial_source = next(s for s in record["sources"] if s["name"] == "trial")
    assert trial_source["trial_status"] == "succeeded" and "truth" in trial_source["limitations"][0]
    for name, values in answers.items():
        assert by_name[name]["outcome"] == "computed"
        for key, value in values.items():
            assert by_name[name]["values"][key] == pytest.approx(value, rel=1e-9, abs=1e-12)
    assert by_name["registered"]["values"]["ncc"] == pytest.approx(1.0, rel=1e-9, abs=1e-12)
    translation = by_name["translation"]
    assert translation["values"]["passed"] is True and translation["values"]["max_error"] == 0.0
    assert translation["arguments"]["detected"]["mapping"]["moving"]["output"] == "transform.json"
    assert by_name["spots"]["parameters"] == _matching() and by_name["spots"]["metric"] == "M03"
    table = pd.read_csv(tmp_path / "evaluation" / "values.csv")
    assert set(table.metric) == set(by_name) and set(table.outcome) == {"computed"}
    for name in by_name:
        assert (tmp_path / "evaluation" / by_name[name]["details_file"]).is_file()


def _corrupt(path):
    data = bytearray(Path(path).read_bytes())
    data[len(data) // 2] ^= 1
    Path(path).write_bytes(bytes(data))


@pytest.mark.parametrize("damage,source,outcome", [
    ("workflow-output", "run", "skipped"),
    ("workflow-record", "run", "skipped"),
    ("checkpoint-payload", "candidates", "skipped"),
    ("checkpoint-missing", "candidates", "skipped"),
    ("checkpoint-header", "candidates", "skipped"),
    ("trial-output", "trial", "skipped"),
    ("trial-record", "trial", "skipped"),
    ("truth", "truth", "skipped"),
])
def test_integrity_and_source_identity(work, trial, tmp_path, monkeypatch, damage, source, outcome):
    sources, metrics, _ = _all_sources(work, trial, tmp_path)
    out = _output(work)
    if damage == "workflow-output":
        _corrupt(out / MOLECULES)
    elif damage == "workflow-record":
        (out / "workflow" / "tasks" / "fov.sample.FOV_001.json").unlink()
    elif damage == "checkpoint-payload":
        _corrupt(out / CHECKPOINTS / "candidates.csv")
    elif damage == "checkpoint-missing":
        (out / CHECKPOINTS / "candidates.csv").unlink()
    elif damage == "checkpoint-header":
        header = json.loads((out / CHECKPOINTS / "candidates.json").read_text())
        (out / CHECKPOINTS / "candidates.json").write_text(json.dumps(dict(header, format_version=99)))
    elif damage == "trial-output":
        _corrupt(trial / "transform.json")
    elif damage == "trial-record":
        record = json.loads((trial / "record.json").read_text())
        record["configuration"]["seed"] = 4
        (trial / "record.json").write_text(json.dumps(record))
    else:
        _corrupt(tmp_path / "truth" / "spots.csv")
    before = _tree(work, trial.parent, tmp_path / "truth")
    _no_processing(monkeypatch)
    status, record, by_name = _evaluate(tmp_path, sources, metrics)
    assert _tree(work, trial.parent, tmp_path / "truth") == before
    assert status == 1 and record["status"] == "incomplete"
    summary = next(s for s in record["sources"] if s["name"] == source)
    if source == "run":
        # A damaged task stays visible: the workflow source is read, its task is not complete.
        assert summary["status"] == "verified" and summary["incomplete"] == [FOV_TASK]
        assert summary["tasks"][FOV_TASK]["problems"]
    else:
        assert summary["status"] == "failed" and summary["error"]["message"]
    reader = {"run": ["molecules"], "candidates": ["spots"], "trial": ["translation"],
              "truth": ["molecules", "spots", "translation"]}[source]
    if damage == "checkpoint-header":
        # candidates.json is also a declared output of the FOV task, so the task is no longer complete either.
        reader = ["molecules", "spots"]
        assert next(s for s in record["sources"] if s["name"] == "run")["incomplete"] == [FOV_TASK]
    for name, metric in by_name.items():
        assert metric["outcome"] == (outcome if name in reader else "computed"), (name, metric)
        if name in reader:
            assert "values" not in metric and metric["reason"]
    rows = pd.read_csv(tmp_path / "evaluation" / "values.csv")
    skipped = rows[rows.outcome == outcome]
    assert set(skipped.metric) == set(reader) and skipped.value.isna().all()


def _version_2(folder):
    """Rewrite the candidates header as version 2: no payload hashes."""
    header = json.loads((folder / "candidates.json").read_text())
    header.pop("files")
    (folder / "candidates.json").write_text(json.dumps(dict(header, format_version=2)))


def test_checkpoint_versions_1_and_2_need_an_external_inventory(work, trial, tmp_path, monkeypatch):
    sources, metrics, answers = _all_sources(work, trial, tmp_path)
    folder = _output(work) / CHECKPOINTS
    metrics["metrics"] = [m for m in metrics["metrics"] if m["name"] == "spots"]
    sources["sources"] = [s for s in sources["sources"] if s["name"] in ("candidates", "truth")]
    _version_2(folder)
    _no_processing(monkeypatch)
    status, record, by_name = _evaluate(tmp_path, sources, metrics, "without")
    candidates = next(s for s in record["sources"] if s["name"] == "candidates")
    assert status == 1 and candidates["status"] == "failed" and candidates["format_version"] == 2
    assert "no payload hashes" in candidates["error"]["message"] and by_name["spots"]["outcome"] == "skipped"
    assert "version-2" in candidates["limitations"][0]
    inventory = [{"path": name, "sha256": _sha(folder / name)} for name in ("candidates.json", "candidates.csv")]
    sources["sources"][0]["inventory"] = inventory[:1]
    status, record, _ = _evaluate(tmp_path, sources, metrics, "partial")
    candidates = next(s for s in record["sources"] if s["name"] == "candidates")
    assert status == 1 and "candidates.csv" in candidates["error"]["message"]
    sources["sources"][0]["inventory"] = inventory
    before = _tree(folder)
    status, record, by_name = _evaluate(tmp_path, sources, metrics, "inventoried")
    assert _tree(folder) == before
    candidates = next(s for s in record["sources"] if s["name"] == "candidates")
    assert status == 0 and candidates["status"] == "verified"
    assert candidates["payload_check"] == "the supplied external inventory"
    assert "only against the supplied external inventory" in candidates["limitations"][0]
    assert {Path(i["path"]).name: i["checked_against"] for i in candidates["consumed"]} == {
        "candidates.json": "inventory", "candidates.csv": "inventory"}
    for key, value in answers["spots"].items():
        assert by_name["spots"]["values"][key] == pytest.approx(value, rel=1e-9, abs=1e-12)
    _corrupt(folder / "candidates.csv")
    status, record, _ = _evaluate(tmp_path, sources, metrics, "corrupt")
    assert status == 1 and next(s for s in record["sources"] if s["name"] == "candidates")["status"] == "failed"


def test_failed_recovered_and_undefined(work, trial, tmp_path, monkeypatch):
    """A failed trial, an incomplete task, a missing file, a metric that raises and an undefined metric: five
    outcomes, none of them a zero."""
    sources, metrics, _ = _all_sources(work, trial, tmp_path)
    failed = tmp_path / "failed"
    failed.mkdir()
    (failed / "trial.json").write_text(json.dumps(dict(method=dict(module="registration", method="translation"),
                                                       inputs=dict(reference="missing.npy", moving="missing.npy"),
                                                       seed=3, root_output_path="out")))
    assert main(["trial", "run", "--config", str(failed / "trial.json")]) == 1
    empty = _truth(tmp_path, "empty.csv", pd.DataFrame(columns=["z", "y", "x"]))
    sources["sources"] += [{"name": "failed", "kind": "trial", "path": str(failed / "out")},
                           {"name": "empty", "kind": "files", "files": {"spots": {"path": str(empty),
                                                                                   "sha256": _sha(empty)}}}]
    (_output(work) / "workflow" / "tasks" / "fov.sample.FOV_001.json").unlink()
    detected = {"source": "candidates", "key": "spot_result.spots", "columns": ["z", "y", "x"]}
    metrics["metrics"] = [
        {"name": "failed-trial", "function": "registration.evaluate_translation",
         "parameters": {"units": "voxel", "tolerance": 0.5},
         "arguments": {"detected": {"mapping": {"moving": {"source": "failed", "output": "transform.json",
                                                            "key": "transform.displacement_zyx"}}},
                       "truth": {"mapping": {"moving": {"source": "truth", "file": "shift", "key": "displacement"}}},
                       "reference_metadata": VOXEL, "observed_metadata": VOXEL}},
        next(m for m in metrics["metrics"] if m["name"] == "molecules"),
        {"name": "missing", "function": "segmentation.label_size_summary",
         "arguments": {"labels": {"source": "trial", "output": "labels.npy"}}},
        {"name": "raises", "function": "spot_finding.f1_score", "parameters": _matching(),
         "arguments": {"detected": dict(detected, columns=["y", "x"]),
                       "truth": {"source": "truth", "file": "spots", "columns": ["z", "y", "x"]},
                       "reference_metadata": VOXEL, "observed_metadata": VOXEL}},
        {"name": "undefined", "function": "spot_finding.f1_score", "parameters": _matching(),
         "arguments": {"detected": detected, "truth": {"source": "empty", "file": "spots", "columns": ["z", "y", "x"]},
                       "reference_metadata": VOXEL, "observed_metadata": VOXEL}}]
    _no_processing(monkeypatch)
    status, record, by_name = _evaluate(tmp_path, sources, metrics)
    assert status == 1
    assert next(s for s in record["sources"] if s["name"] == "failed")["trial_status"] == "failed"
    assert {name: m["outcome"] for name, m in by_name.items()} == {
        "failed-trial": "skipped", "molecules": "skipped", "missing": "unavailable", "raises": "failed",
        "undefined": "computed"}
    assert "status 'failed'" in by_name["failed-trial"]["reason"]
    assert "not complete" in by_name["molecules"]["reason"] and "no task record" in by_name["molecules"]["reason"]
    assert "labels.npy" in by_name["missing"]["reason"]
    assert by_name["raises"]["error"]["type"] == "ValueError"
    undefined = by_name["undefined"]
    assert undefined["status"] == "undefined" and undefined["values"]["f1"] is None
    assert undefined["values"]["recall"] is None and undefined["reasons"]["f1"]
    assert record["counts"] == {"sources": 8, "failed_sources": 0, "metrics": 5, "computed": 1, "skipped": 2,
                                "unavailable": 1, "failed": 1}
    rows = pd.read_csv(tmp_path / "evaluation" / "values.csv")
    f1 = rows[(rows.metric == "undefined") & (rows.value_name == "f1")]
    assert f1.value.isna().all() and f1.reason.notna().all()
    assert rows[rows.metric != "undefined"].value.isna().all()


def test_missing_sources_and_unsafe_paths(work, trial, tmp_path, capsys):
    sources, metrics, _ = _all_sources(work, trial, tmp_path)
    good = {"sources": sources, "metrics": metrics}

    def refused(output, **change):
        value = dict(good, **change)
        before = _tree(work, trial.parent)
        with pytest.raises(SystemExit) as error:
            main(["experiment", "evaluate",
                  "--sources", str(_write(tmp_path / "s.yaml", value["sources"])),
                  "--metrics", str(_write(tmp_path / "m.yaml", value["metrics"])), "--output", str(output)])
        assert error.value.code == 2
        assert _tree(work, trial.parent) == before
        return capsys.readouterr().err

    assert "inside source 'trial'" in refused(trial / "evaluation")
    assert "inside source 'run'" in refused(_output(work) / "evaluation")
    assert not (trial / "evaluation").exists() and not (_output(work) / "evaluation").exists()
    (tmp_path / "used").mkdir()
    (tmp_path / "used" / "file").write_text("")
    assert "new or empty" in refused(tmp_path / "used")
    target = tmp_path / "new"
    assert "kind" in refused(target, sources={"sources": [{"name": "x", "kind": "folder", "path": "."}]})
    assert "public function" in refused(target, metrics={"metrics": [dict(metrics["metrics"][1],
                                                                         function="spot_finding._points")]})
    for function in ("os.system", "workflow.values_match"):
        assert "with a module of" in refused(target, metrics={"metrics": [dict(metrics["metrics"][0],
                                                                               function=function)]})
    bad = dict(metrics["metrics"][2], arguments={"reads": {"source": "nowhere"}})
    assert "names a source" in refused(target, metrics={"metrics": [bad]})
    bad = dict(metrics["metrics"][2], parameters={"threshold": 1})
    assert "unexpected keyword" in refused(target, metrics={"metrics": [bad]})
    assert "nonempty list 'metrics'" in refused(target, metrics={"metrics": []})
    assert not target.exists()


def test_a_file_with_its_recorded_hash_that_cannot_be_read_fails_its_source(work, trial, tmp_path, monkeypatch):
    sources, metrics, _ = _all_sources(work, trial, tmp_path)
    shift = tmp_path / "truth" / "shift.json"
    shift.write_text("{not json")
    truth = next(s for s in sources["sources"] if s["name"] == "truth")
    truth["files"]["shift"]["sha256"] = _sha(shift)
    _no_processing(monkeypatch)
    status, record, by_name = _evaluate(tmp_path, sources, metrics)
    assert status == 1
    summary = next(s for s in record["sources"] if s["name"] == "truth")
    assert summary["status"] == "failed" and "cannot be read as json" in summary["error"]["message"]
    assert by_name["translation"]["outcome"] == "skipped" and "failed while it was read" in by_name["translation"]["reason"]
    # The metrics that read the truth before it failed were computed from verified files.
    assert by_name["molecules"]["outcome"] == by_name["spots"]["outcome"] == "computed"


def _truth_entry(sources, key):
    return next(s for s in sources["sources"] if s["name"] == "truth")["files"][key]


def _record_output(trial, name, change):
    record = json.loads((trial / "record.json").read_text())
    change(next(o for o in record["outputs"] if o["path"] == name))
    (trial / "record.json").write_text(json.dumps(record))


def _record_input(trial, role, change):
    record = json.loads((trial / "record.json").read_text())
    change(record["inputs"][role])
    (trial / "record.json").write_text(json.dumps(record))


# (case, source that fails, the field the message names, how to make the omission). Each required SHA-256 or byte
# count that is missing or malformed fails its source as a SourceFailure before any metric reads it.
_REQUIRED = [
    ("files-null", "truth", "file 'spots' field sha256", lambda s, w, t: _truth_entry(s, "spots").update(sha256=None)),
    ("files-malformed", "truth", "file 'spots' field sha256",
     lambda s, w, t: _truth_entry(s, "spots").update(sha256="sha256:" + "A" * 64)),
    ("files-short", "truth", "file 'spots' field sha256", lambda s, w, t: _truth_entry(s, "spots").update(sha256="ab12")),
    ("files-missing", "truth", "file 'spots' field sha256", lambda s, w, t: _truth_entry(s, "spots").pop("sha256")),
    ("trial-output-sha256", "trial", "output 'transform.json' field sha256",
     lambda s, w, t: _record_output(t, "transform.json", lambda o: o.pop("sha256"))),
    ("trial-output-bytes", "trial", "output 'transform.json' field bytes",
     lambda s, w, t: _record_output(t, "transform.json", lambda o: o.pop("bytes"))),
    ("trial-output-bytes-text", "trial", "output 'image.npy' field bytes",
     lambda s, w, t: _record_output(t, "image.npy", lambda o: o.update(bytes=str(o["bytes"])))),
    ("trial-input-sha256", "trial", "input 'moving' field sha256",
     lambda s, w, t: _record_input(t, "moving", lambda i: i.pop("sha256"))),
    ("trial-record-null", "trial", "source 'trial' field sha256",
     lambda s, w, t: next(x for x in s["sources"] if x["name"] == "trial").update(sha256=None)),
    ("manifest-malformed", "run", "source 'run' field sha256",
     lambda s, w, t: next(x for x in s["sources"] if x["name"] == "run").update(sha256="not-a-digest")),
]


@pytest.mark.parametrize("case,source,field,omit", _REQUIRED, ids=[c[0] for c in _REQUIRED])
def test_required_hashes_and_sizes_fail_their_source_when_missing_or_malformed(work, trial, tmp_path, monkeypatch,
                                                                                 case, source, field, omit):
    sources, metrics, _ = _all_sources(work, trial, tmp_path)
    omit(sources, work, trial)
    before = _tree(work, trial.parent, tmp_path / "truth")
    _no_processing(monkeypatch)
    status, record, by_name = _evaluate(tmp_path, sources, metrics)
    assert _tree(work, trial.parent, tmp_path / "truth") == before
    assert status == 1 and record["status"] == "incomplete"
    failed = {s["name"]: s for s in record["sources"] if s["status"] == "failed"}
    assert set(failed) == {source}
    assert failed[source]["error"]["type"] == "SourceFailure"
    assert field in failed[source]["error"]["message"] and "is required" in failed[source]["error"]["message"]
    readers = {"truth": {"molecules", "spots", "translation"}, "trial": {"translation"}, "run": {"molecules"}}[source]
    for name, metric in by_name.items():
        assert metric["outcome"] == ("skipped" if name in readers else "computed"), (case, name, metric)


def test_an_inventory_digest_is_required_for_a_version_2_checkpoint(work, trial, tmp_path, monkeypatch):
    sources, metrics, _ = _all_sources(work, trial, tmp_path)
    folder = _output(work) / CHECKPOINTS
    _version_2(folder)
    metrics["metrics"] = [m for m in metrics["metrics"] if m["name"] == "spots"]
    sources["sources"] = [s for s in sources["sources"] if s["name"] in ("candidates", "truth")]
    sources["sources"][0]["inventory"] = [{"path": "candidates.json", "sha256": _sha(folder / "candidates.json")},
                                          {"path": "candidates.csv", "sha256": None}]
    before = _tree(folder)
    _no_processing(monkeypatch)
    status, record, by_name = _evaluate(tmp_path, sources, metrics)
    assert _tree(folder) == before
    candidates = next(s for s in record["sources"] if s["name"] == "candidates")
    assert status == 1 and candidates["status"] == "failed" and candidates["error"]["type"] == "SourceFailure"
    assert "inventory 'candidates.csv' field sha256" in candidates["error"]["message"]
    assert by_name["spots"]["outcome"] == "skipped"


# W-803: a version-3 header must hold its payload inventory, which names every file of the stage folder. Each case
# edits one stage header of the workflow's run; the source fails before it reads a payload file or a metric reads it.
HEADERS = {"registered": "registered/transforms.json", "candidates": "candidates.json", "pre_qc": "pre_qc.json"}
READERS = {"registered": "registered", "candidates": "spots", "pre_qc": "reads"}
TABLES = {"registered": "round2.ome.tif", "candidates": "candidates.csv", "pre_qc": "pre_qc.csv"}


def _header_digest(value):
    return lambda header, stage, folder: header["files"][0].update(sha256=value)


def _unnamed(header, stage, folder):
    """The stage table's entry names another file, so the list stays nonempty."""
    next(e for e in header["files"] if e["path"] == TABLES[stage]).update(path=f"earlier-{TABLES[stage]}")


def _stray(header, stage, folder):
    """A file in the stage folder that the header does not name."""
    (folder / ("extra.npz" if stage == "registered" else f"{stage}.parquet")).write_bytes(b"stray")


_INVENTORY = {
    "missing": (lambda header, stage, folder: header.pop("files"), "needs a nonempty files list"),
    "null": (lambda header, stage, folder: header.update(files=None), "needs a nonempty files list"),
    "empty": (lambda header, stage, folder: header.update(files=[]), "needs a nonempty files list"),
    "unnamed": (_unnamed, "files does not name"),
    "unlisted": (_stray, "files do not name the stage files"),
    "malformed-digest": (_header_digest("sha256:" + "A" * 64), "needs a SHA-256 of 64 lowercase hex"),
    "null-digest": (_header_digest(None), "needs a SHA-256 of 64 lowercase hex"),
}


@pytest.mark.parametrize("stage", list(HEADERS))
@pytest.mark.parametrize("case", list(_INVENTORY))
def test_a_version_3_checkpoint_needs_its_inventory_of_every_stage_file(work, trial, tmp_path, monkeypatch, stage,
                                                                         case):
    sources, metrics, _ = _all_sources(work, trial, tmp_path)
    sources["sources"] = [s for s in sources["sources"] if s["name"] in (stage, "truth")]
    metrics["metrics"] = [m for m in metrics["metrics"] if m["name"] == READERS[stage]]
    header_file = _output(work) / CHECKPOINTS / HEADERS[stage]
    header = json.loads(header_file.read_text())
    edit, message = _INVENTORY[case]
    edit(header, stage, header_file.parent)
    header_file.write_text(json.dumps(header))
    before = _tree(work, tmp_path / "truth")
    _no_processing(monkeypatch)
    import starfinder.dataset

    def read(*args, **kwargs):
        raise AssertionError("load_checkpoint read the checkpoint")
    monkeypatch.setattr(starfinder.dataset, "load_checkpoint", read)
    status, record, by_name = _evaluate(tmp_path, sources, metrics)
    assert _tree(work, tmp_path / "truth") == before
    summary = next(s for s in record["sources"] if s["name"] == stage)
    assert status == 1 and summary["status"] == "failed" and summary["error"]["type"] == "SourceFailure"
    assert f"source {stage!r}" in summary["error"]["message"] and message in summary["error"]["message"]
    assert f"{stage} checkpoint" in summary["error"]["message"]
    # Only the header was read; no payload file and no metric.
    assert [Path(item["path"]) for item in summary["consumed"]] == [header_file]
    assert by_name[READERS[stage]]["outcome"] == "skipped" and "values" not in by_name[READERS[stage]]
