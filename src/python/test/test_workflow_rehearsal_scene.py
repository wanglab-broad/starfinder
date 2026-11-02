"""The mid-size scene and configuration writer of the workflow's local rehearsal (scripts/workflow_rehearsal_scene.py,
W-761): its configuration plans, on a scene of the same kind at 8 × 64 × 64."""
import importlib.util
from collections import Counter
from pathlib import Path

import pytest

from starfinder.workflow import plan

pytestmark = pytest.mark.workflow

ROOT = Path(__file__).resolve().parents[3]


def _script():
    spec = importlib.util.spec_from_file_location("workflow_rehearsal_scene",
                                                  ROOT / "src" / "python" / "scripts" / "workflow_rehearsal_scene.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_rehearsal_configuration_plans_every_block_of_the_full_example(tmp_path):
    script = _script()
    work = tmp_path / "scene"
    record = script.write_scene(work, (8, 64, 64), 4)
    assert record["fov_ids"] == ["FOV_001", "FOV_002", "FOV_003", "FOV_004"]
    assert record["shape_zyx"] == [8, 64, 64] and record["counts"]["cells"] > 0
    root = script.input_root(work)
    assert (root / "fovs.csv").read_text().splitlines()[1:] == [f"sample,{fov}" for fov in record["fov_ids"]]
    data = plan(script.write_config(work, tmp_path / "output"))
    # The 29 tasks of the full example's plan on tiny_sample (the workflow guide's Quick start).
    assert Counter(t["kind"] for t in data["tasks"]) == {
        "summarize": 4, "merge": 1, "fov": 4, "morphology": 4, "segment": 8, "assign": 4, "stitch": 1,
        "export": 1, "report": 1, "compare": 1}
    assert Path(data["roots"]["output"]) == tmp_path / "output" / script.DATASET_ID / script.OUTPUT_ID
    assert all(t["execution"]["threads"] == 1 for t in data["tasks"])
