"""X4 of design/validation/orchestration.md: the refactored per-FOV core against the original's values.

The ``fov`` command on ``session_dataset`` (the pytest session's small synthetic dataset) reproduces ``core_values``
(``data/expected/test_workflow_core_golden/``, recorded by W-734 and updated only for W-790's removed spots): the molecule
tables, ``run.json`` and the checkpoint stage records, the stage tables and the reference images, compared by
``workflow_golden.compare`` with ``same_environment=False``, so that the result depends on neither the caller nor the
host. Each command runs in its own process with every variable of ``_execution.THREAD_VARIABLES`` set to "1". The
contract's exceptions (design/specs/orchestration.md, "The exception of the golden comparison") are applied to the
values, each named by its decision: OC3, the rotation by 0 is skipped (no ``rotate_round`` steps, frame ids without
``/rotate:0``, ``rotation_degrees`` null); OC4, the paths recorded under the work folder (written ``<work>`` on both
sides); D8, the new ``workflow`` section of ``run.json``.
"""
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from starfinder._execution import THREAD_VARIABLES
from starfinder.workflow import plan

from . import workflow_examples as X

pytestmark = [pytest.mark.workflow, pytest.mark.dataset, pytest.mark.golden]

EXPECTED = Path(__file__).parent / "data" / "expected" / "test_workflow_core_golden"
FOVS = ("FOV_001", "FOV_002")


def _recorder():
    spec = importlib.util.spec_from_file_location("record_core_values", EXPECTED / "record_core_values.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configuration(work):
    """W-734's configuration of the core values (record_core_values.configuration) in the Python shape: the
    processing block of W-642's golden configuration A without its run flags, on the session dataset's layout."""
    return {"root_input_path": str(work / "input"), "root_output_path": str(work / "output"), "dataset_id": "session",
            "sample_id": "small", "output_id": "core", "n_rounds": 4, "ref_round": "round1", "rotate_angle": 0,
            "voxel_size_xy": 0.5, "voxel_size_z": 1.0, "seq_channel_order": ["ch00", "ch01", "ch02", "ch03"],
            "image_dir_template": "{fov}/{round}", "codebook": "codebook.csv",
            "processing": dict(X.PROCESSING, checkpoints=["candidates", "pre_qc"])}


def write_work(work, dataset):
    """The work folder of record_core_values: the session dataset as the input root (copied, so that the frame ids
    name the work folder as W-734's did) with the workflow's fovs.csv, and fixture.json listing its files."""
    rec = _recorder()
    root = work / "input" / "session" / "small"
    shutil.copytree(dataset, root)
    generation = json.loads((root / "generation.json").read_text())
    fov_ids = list(generation["fov_ids"])
    assert fov_ids == list(FOVS)
    (root / "fovs.csv").write_text("sample,fov_id\n" + "".join(f"small,{fov}\n" for fov in fov_ids))
    files = {}
    for path in sorted(p for p in work.rglob("*") if p.is_file()):
        name = path.relative_to(work).as_posix()
        files[name] = {"bytes": path.stat().st_size}
        if path.name not in rec.FIXTURE_VARYING:
            files[name]["sha256"] = rec.G.sha256_file(path)
    (work / "fixture.json").write_text(json.dumps({"fov_ids": fov_ids, "files": files}, indent=1) + "\n")


def _strip_rotation(value):
    """OC3 on an expected record: rotate_angle 0 is no rotation, so no rotate_round step, no ``/rotate:0`` in a frame
    id and a null rotation_degrees."""
    if isinstance(value, dict):
        out = {k: _strip_rotation(v) for k, v in value.items()}
        if out.get("rotation_degrees") == 0:
            out["rotation_degrees"] = None
        return out
    if isinstance(value, list):
        return [_strip_rotation(v) for v in value if not (isinstance(v, dict) and v.get("name") == "rotate_round")]
    if isinstance(value, str):
        return value.replace("/rotate:0", "")
    return value


def expected_with_exceptions(target):
    """A copy of the core values with OC3 applied to their records."""
    shutil.copytree(EXPECTED, target)
    for path in (target / "records").rglob("*.json"):
        path.write_text(json.dumps(_strip_rotation(json.loads(path.read_text())), indent=1) + "\n")
    return target


def actual_with_exceptions(folder):
    """D2 and D8 on the refactored values: the plan's files, the task records and logs (workflow/) and the new
    workflow section of run.json are left out."""
    values = json.loads((folder / "values.json").read_text())
    values = {k: v for k, v in values.items() if not k.startswith("file:workflow/")}
    (folder / "values.json").write_text(json.dumps(values, indent=1, sort_keys=True) + "\n")
    for path in (folder / "records").rglob("run.json"):
        record = json.loads(path.read_text())
        record.pop("workflow")
        path.write_text(json.dumps(record, indent=1) + "\n")
    return folder


def run_core(work, extra_environment=None):
    """Plan the core configuration and run each FOV's fov task in its own process, one numerical thread."""
    plan(X.write_config(work, "core", configuration(work)))
    manifest = work / "output" / "session" / "core" / "workflow" / "manifest.json"
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", **dict.fromkeys(THREAD_VARIABLES, "1"),
                       **(extra_environment or {}))
    environment["PYTHONPATH"] = os.pathsep.join([str(Path(__file__).resolve().parents[1]),
                                                 environment.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    for fov in FOVS:
        command = [sys.executable, "-m", "starfinder", "workflow", "fov", "--manifest", str(manifest), "--task",
                   f"fov/small/{fov}"]
        done = subprocess.run(command, env=environment, capture_output=True, text=True)
        assert done.returncode == 0, done.stderr
    return manifest


@pytest.mark.extended
@pytest.mark.slow
def test_the_fov_command_reproduces_the_core_values(small_dataset, tmp_path):
    work = tmp_path / "work"
    write_work(work, small_dataset)
    run_core(work)
    _recorder().extract(work, tmp_path / "actual")
    actual = actual_with_exceptions(tmp_path / "actual")
    expected = expected_with_exceptions(tmp_path / "expected")
    differences = _recorder().compare(expected, actual, same_environment=False)
    assert differences == [], "\n".join(differences)
