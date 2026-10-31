"""The statistics and sample commands of the workflow on ``tiny_sample``: each task of a variant run by its command.

The two-pass statistics of the ``per-fov`` variant (four ``summarize`` tasks, their ``merge`` and a ``fov`` task that
reads the merged file), and with a second supplied step summarized on the first step's file; the whole ``molecules-only`` variant (four ``fov`` tasks, ``stitch``, ``export``, ``report``
and ``compare``), after which ``verify --hash --final`` passes; and the BigStitcher macro task of ``after-fusion``,
then ``bigstitcher-check`` without Fiji's ``dataset.xml``, which fails (exit 1) as Fiji's failed macro would.
"""
import importlib.util
import json
import shutil

import pytest

from starfinder import _layout
from starfinder.workflow import plan, verify

from . import workflow_examples as X

pytestmark = [pytest.mark.workflow, pytest.mark.integration, pytest.mark.contract]


@pytest.fixture
def work(tmp_path):
    work = tmp_path / "work"
    X.golden().write_fixture(work)
    X.write_supplied(work)
    return work


def _manifest(work, variant):
    plan(X.write_config(work, variant, X.variants(work)[variant]))
    return X.output_root(work, f"example-{variant}") / "workflow" / "manifest.json"


def _tasks(manifest, *kinds):
    return [t["id"] for t in json.loads(manifest.read_text())["tasks"] if t["kind"] in kinds]


def test_the_two_pass_statistics_feed_the_fov_tasks(work):
    from starfinder.preprocessing import load_supplied_statistics
    manifest = _manifest(work, "per-fov")
    for task in _tasks(manifest, "summarize", "merge"):
        assert X.run_task(manifest, task) == 0, task
    out = manifest.parents[1]
    supplied = out / "statistics" / "sample" / "percentile_normalization" / "supplied.json"
    document = load_supplied_statistics(supplied)
    assert document["fovs_used"] == list(X.FOVS) and list(document["steps"]) == ["percentile_normalization"]
    assert X.run_task(manifest, "fov/sample/FOV_003") == 0
    record = json.loads((out / "fovs" / "FOV_003" / "run.json").read_text())
    assert record["preprocessing"]["supplied_statistics"]["path"] == str(supplied)
    assert verify(manifest, task="fov/sample/FOV_003", hash=True)["tasks"]["fov/sample/FOV_003"]["complete"]


def test_a_later_supplied_step_is_summarized_on_the_earlier_steps_file(work):
    from starfinder.preprocessing import load_supplied_statistics
    config = X.variants(work)["per-fov"]
    config["processing"]["preprocessing"]["steps"].insert(
        2, {"method": "histogram_matching", "fit": "supplied", "reference_channel": 0})
    plan(X.write_config(work, "two-steps", config))
    manifest = X.output_root(work, "example-per-fov") / "workflow" / "manifest.json"
    first = "statistics/sample/percentile_normalization/supplied.json"
    for task in json.loads(manifest.read_text())["tasks"]:
        if task["kind"] == "summarize" and task["identity"]["step"] == "histogram_matching":
            assert task["needs"] == ["merge/sample/percentile_normalization"] and first in task["inputs"]
    for task in _tasks(manifest, "summarize", "merge") + ["fov/sample/FOV_002"]:
        assert X.run_task(manifest, task) == 0, task
    out = manifest.parents[1]
    last = load_supplied_statistics(out / "statistics" / "sample" / "histogram_matching" / "supplied.json")
    assert list(last["steps"]) == ["percentile_normalization", "histogram_matching"]
    assert last["steps"]["percentile_normalization"] == load_supplied_statistics(out / first)["steps"][
        "percentile_normalization"]
    record = json.loads((out / "fovs" / "FOV_002" / "run.json").read_text())
    assert record["preprocessing"]["supplied_statistics"]["path"].endswith("histogram_matching/supplied.json")


@pytest.mark.slow
def test_the_molecules_only_variant_runs_to_its_comparison_page(work):
    if importlib.util.find_spec("multiview_stitcher") is None or importlib.util.find_spec("anndata") is None:
        pytest.skip("the stitching and the export of this variant need the multiview-stitcher and anndata extras")
    manifest = _manifest(work, "molecules-only")
    for task in json.loads(manifest.read_text())["tasks"]:
        assert X.run_task(manifest, task["id"], "--cache-token", "t1") == 0, task["id"]
    result = verify(manifest, hash=True, final=True)
    assert result["passed"] and result["incomplete"] == []
    assert {entry["token"] for entry in result["tasks"].values()} == {"t1"}
    out = manifest.parents[1]
    assert (out / "report" / "index.html").is_file() and (out / "export" / "sample" / "export.json").is_file()
    assert json.loads((out / "export" / "sample" / "export.json").read_text())["assembly"]["mode"] == "molecules_only"


def test_the_bigstitcher_check_fails_without_fijis_dataset_xml(work, capsys):
    manifest = _manifest(work, "after-fusion")
    for task in _tasks(manifest, "fov") + _tasks(manifest, "bigstitcher_macro"):
        assert X.run_task(manifest, task) == 0, task
    folder = work / "bigstitcher" / "sample"
    assert (folder / "BigStitcher_macro.ijm").is_file() and (folder / "view_map.csv").is_file()
    assert sorted(p.name for p in (folder / "tiles").iterdir())
    assert X.run_task(manifest, "bigstitcher_engine/sample") == 1
    assert "wrote no dataset.xml" in capsys.readouterr().err
    out = manifest.parents[1]
    assert not _layout.task_record(out, "bigstitcher_engine/sample").exists()
    assert _layout.failed_record(out, "bigstitcher_engine/sample").is_file()
    assert X.run_task(manifest, "bigstitcher_import/sample") == 2
    shutil.rmtree(folder)
