"""The tiled task kinds run by their commands (C3 for the tiled kinds; design/specs/orchestration-tiled.md).

On one FOV of ``tiny_sample`` (5 × 36 × 36) planned with ``processing.mode: tiled`` and a tile size of 18 (2 × 2
tiles), ``tile-global``, the four ``tile`` commands and ``tile-merge`` each exit 0 and write a succeeded record; the
merge writes ``run.json`` (version 2, ``succeeded``, with its ``tiled`` section and the workflow section),
``molecules.csv`` and the stage records: the declared files of direct (D5), equal in their tables to a direct ``fov``
task's on the same FOV. The chunked checkpoint is removed after the merge unless the stage ``registered`` is kept, and
``verify`` reports every task complete (the removed checkpoint is the group's temporary output). The global task's
verified resume keeps its files when its ``run.json`` matches and its chunked checkpoint hashes as recorded.
"""
import json
import shutil

import pandas as pd
import pytest

from starfinder import _layout
from starfinder.workflow import plan, verify

from . import workflow_examples as X

pytestmark = [pytest.mark.workflow, pytest.mark.dataset, pytest.mark.contract]

FOV = "FOV_001"


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


def _plan(work, output_id, *, tiled=True, checkpoints=None, **execution):
    config = X.single_fov(work, output_id, **execution)
    if tiled:
        config["processing"].update(mode="tiled", tiled={"tile_size_yx": 18})
    if checkpoints is not None:
        config["processing"]["checkpoints"] = checkpoints
    manifest = plan(X.write_config(work, output_id, config))
    return X.output_root(work, output_id) / "workflow" / "manifest.json", manifest


def _run_all(path, manifest):
    for task in manifest["tasks"]:
        assert X.run_task(path, task["id"]) == 0, task["id"]


def _table(path):
    return pd.read_csv(path, keep_default_na=False)


def test_the_tiled_tasks_write_the_declared_files_of_direct(work):
    path, manifest = _plan(work, "tiled")
    assert [t["kind"] for t in manifest["tasks"]] == ["tile_global", "tile", "tile", "tile", "tile", "tile_merge"]
    _run_all(path, manifest)
    out = X.output_root(work, "tiled")
    folder = _layout.fov_dir(out, FOV)
    for task in manifest["tasks"]:
        record = json.loads(_layout.task_record(out, task["id"]).read_text())
        assert record["status"] == "succeeded", task["id"]
        assert [e["path"] for e in record["outputs"]] == [*task["outputs"], task["log"]]
    merge = next(t for t in manifest["tasks"] if t["kind"] == "tile_merge")
    assert merge["outputs"] == ["fovs/FOV_001/checkpoints/candidates.json", "fovs/FOV_001/checkpoints/pre_qc.json",
                                "fovs/FOV_001/molecules.csv", "fovs/FOV_001/run.json"]
    run = json.loads((folder / "run.json").read_text())
    assert run["format_version"] == 2 and run["status"] == "succeeded" and run["subtile_id"] is None
    assert run["workflow"]["task"] == {"id": merge["id"], "kind": "tile_merge"}
    assert set(run["checkpoints"]) == {"candidates", "pre_qc"}
    assert set(run["outputs"]) == {"molecules", "reference_image"}
    tiled = run["tiled"]
    assert tiled["grid"] == {"tile_size_yx": 18, "n_yx": [2, 2], "halo": 6, "chunk_zyxc": [5, 36, 36, 4],
                             "local_registration": None}
    assert [t["tile"] for t in tiled["tiles"]] == ["r0c0", "r0c1", "r1c0", "r1c1"]
    assert sum(t["candidates"] for t in tiled["tiles"]) == run["counts"]["spots"]
    assert tiled["checkpoint"]["kept"] is False and sorted(tiled["checkpoint"]["sha256"]) == \
        ["round1", "round2", "round3", "round4"]
    # The chunked checkpoint is removed after the merge; verify exempts it (the group's temporary output).
    assert not (_layout.tiled_dir(folder) / "registered.h5").exists()
    assert verify(path, hash=True, final=True)["passed"]
    # The same FOV direct: the tables equal (this fixture's scores are equal to the last bit, as W-644 measured).
    direct_path, direct = _plan(work, "direct", tiled=False)
    _run_all(direct_path, direct)
    other = _layout.fov_dir(X.output_root(work, "direct"), FOV)
    for name in ("molecules.csv", "checkpoints/candidates.csv", "checkpoints/pre_qc.csv"):
        pd.testing.assert_frame_equal(_table(folder / name), _table(other / name), check_exact=False, rtol=1e-9,
                                      atol=1e-12)
    assert run["counts"]["filtering"] == json.loads((other / "run.json").read_text())["counts"]["filtering"]
    assert (folder / "reference.tif").read_bytes() == (other / "reference.tif").read_bytes()


def test_the_chunked_checkpoint_stays_when_the_stage_registered_is_kept(work):
    path, manifest = _plan(work, "kept", checkpoints=["registered", "candidates", "pre_qc"])
    group = next(g for g in manifest["groups"] if g["kind"] == "tile_global")
    assert group["temporary"] == []
    _run_all(path, manifest)
    folder = _layout.fov_dir(X.output_root(work, "kept"), FOV)
    assert (_layout.tiled_dir(folder) / "registered.h5").is_file()
    assert json.loads((folder / "run.json").read_text())["tiled"]["checkpoint"]["kept"] is True


def test_the_global_task_resumes_from_its_chunked_checkpoint(work):
    path, manifest = _plan(work, "resume", resume_from_checkpoint=True)
    task = manifest["tasks"][0]["id"]
    assert X.run_task(path, task) == 0
    folder = _layout.tiled_dir(_layout.fov_dir(X.output_root(work, "resume"), FOV))
    before = {name: (folder / name).stat().st_mtime_ns for name in ("registered.h5", "global.json")}
    assert X.run_task(path, task, "--attempt", "2") == 0
    record = json.loads(_layout.task_record(X.output_root(work, "resume"), task).read_text())
    assert record["workflow"]["resumed_from"] == "registered"
    assert {name: (folder / name).stat().st_mtime_ns for name in before} == before
    # A damaged chunked checkpoint is not resumed: the task runs again from scratch.
    with open(folder / "registered.h5", "r+b") as handle:
        handle.seek(-64, 2)
        handle.write(b"\0" * 64)
    assert X.run_task(path, task, "--attempt", "3") == 0
    record = json.loads(_layout.task_record(X.output_root(work, "resume"), task).read_text())
    assert record["workflow"]["resumed_from"] is None


def test_the_subtile_api_is_removed():
    """The gate's choice A (design/specs/orchestration-tiled.md, "Choice: the package's subtile API"): the subtile API
    is gone without aliases; FOV.tile replaces FOV.subtile_id."""
    import dataclasses

    import starfinder.dataset as dataset
    from starfinder.dataset import FOV, Dataset
    for name in ("SubtileConfig", "CropWindow"):
        assert not hasattr(dataset, name) and name not in dataset.__all__
    for name in ("save_subtiles", "from_subtile", "subtile_id"):
        assert not hasattr(FOV, name)
    assert "tile" in {f.name for f in dataclasses.fields(FOV)}
    assert "subtile" not in {f.name for f in dataclasses.fields(Dataset)}
    assert not hasattr(FOV(Dataset("/in", "/out", "d", "s", "o"), "F").paths, "subtile_dir")
