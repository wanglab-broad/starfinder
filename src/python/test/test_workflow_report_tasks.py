"""The report and compare commands (moved by W-750 from the tests of the report_sample and report_dataset adapters in
test_report_workflow.py, WF1 of design/specs/report-algorithms.md).

Each test plans a configuration in the Python shape with a ``report`` block on an input root whose raw images are
empty files (``workflow_tasks``); the report's function is called on its task's ``Run`` with ``Dataset.report`` and
``Dataset.compare_samples`` spied, as the adapter tests spied them, or runs for real on ``report_records`` of
``report_fixtures.py`` written into the plan's output root. An invalid report block is refused by the plan before any
task.
"""
import copy

import pandas as pd
import pytest

from starfinder.dataset import Dataset
from starfinder.report import ReportConfig, load_report
from starfinder.workflow._plan import PlanRefused

from . import workflow_examples as X
from . import workflow_tasks as T
from .test_report_workflow import REPORT, REPORT_CONFIG, SAMPLE

pytestmark = [pytest.mark.workflow, pytest.mark.contract]

MERGED = {"method": "multiview_stitcher", "input": "reference_merged",
          "positions": {"source": "regular_grid", "regular_grid": {"columns": 2, "rows": 2,
                                                                   "order": "column_major_down_right",
                                                                   "overlap": 0.25}}}
SEGMENT = {"runs": [{"name": "cell", "target": "cell", "method": "import", "path": "{labels}"}]}


def spy(monkeypatch, name):
    calls = []
    monkeypatch.setattr(Dataset, name, lambda self, *args, **kwargs: calls.append((self, args, kwargs)))
    return calls


def membership(rows):
    """fovs.csv: the FOVs Position<i:03d> of each (sample, start, end)."""
    return "sample,fov_id\n" + "".join(f"{s},Position{i:03d}\n" for s, a, b in rows for i in range(a, b + 1))


def report_config(work, report, blocks):
    """The processing block, the report block and the blocks of a case; an assignment name of the assembly block gets
    the assignment block and the segment run it reads (the route per FOV)."""
    config = T.dataset_keys(work, processing=copy.deepcopy(X.PROCESSING), report=report, **copy.deepcopy(blocks))
    assembly = config.get("assembly")
    if assembly is not None:
        config["export"] = {"fuse": False}           # the export's fusion would need the morphology block
    if assembly is not None and assembly.get("assignment", "default") is not None:
        labels = work / "labels" / "{fov_id}.tif"
        segmentation = copy.deepcopy(SEGMENT)
        segmentation["runs"][0]["path"] = str(labels)
        config.update(segmentation=segmentation, export={"route": "per_fov", "fuse": False},
                      assignment={"cells": "cell", "name": assembly.get("assignment", "default")})
    return config


#: The cases of test_wf1_report_sample_calls_dataset_report: the blocks and the assignment the report reads.
CASES = [
    ({"stitching": MERGED, "assembly": {"assignment": "default"}}, "default"),
    ({"stitching": MERGED, "assembly": {"assignment": "cells_v2"}}, "cells_v2"),
    ({"stitching": MERGED, "assembly": {"assignment": None}}, None),
    ({"stitching": MERGED, "assembly": {}}, "default"),
    ({"stitching": MERGED}, None),
    ({}, None),
]


@pytest.mark.parametrize("blocks, assignment", CASES)
@pytest.mark.parametrize("block", ["contract", "empty"])
def test_the_report_command_calls_dataset_report(tmp_path, monkeypatch, blocks, assignment, block):
    """From test_wf1_report_sample_calls_dataset_report: the sample's FOVs from fovs.csv, the assembly block's
    assignment (none without the block, where no task saves an assignment) and the report block's configuration."""
    calls = spy(monkeypatch, "report")
    config = report_config(tmp_path, copy.deepcopy(REPORT) if block == "contract" else {}, blocks)
    rows = [("sample0", 1, 2), (SAMPLE, 3, 5), ("sample2", 6, 6)]
    T.write_inputs(config, membership(rows))
    for i in range(1, 7):
        (tmp_path / "labels").mkdir(exist_ok=True)
        (tmp_path / "labels" / f"Position{i:03d}.tif").write_bytes(b"")
    manifest, _ = T.plan(tmp_path, "report", config)
    T.call(manifest, f"report/{SAMPLE}")
    out = T.output_root(config)
    (dataset, args, kwargs), = calls
    assert dataset.output_root == out and dataset.dataset_id == "data"
    assert args == (SAMPLE, ["Position003", "Position004", "Position005"])
    assert kwargs == {"assignment": assignment, "config": REPORT_CONFIG if block == "contract" else ReportConfig(),
                      "output_dir": out / "report" / SAMPLE}


def test_the_compare_command_calls_compare_samples_on_every_sample(tmp_path, monkeypatch):
    """From test_wf1_report_dataset_calls_compare_samples_on_every_sample: every sample's report, in fovs.csv's
    order, into report/."""
    calls = spy(monkeypatch, "compare_samples")
    config = report_config(tmp_path, {}, {})
    T.write_inputs(config, membership([("sample0", 1, 1), (SAMPLE, 2, 2), ("sample2", 3, 3)]))
    manifest, _ = T.plan(tmp_path, "compare", config)
    T.call(manifest, "compare")
    out = T.output_root(config)
    (dataset, args, kwargs), = calls
    assert dataset.output_root == out
    assert args == (["sample0", SAMPLE, "sample2"],) and kwargs == {"report_root": out / "report"}


def test_an_invalid_report_block_is_refused_before_any_task(tmp_path, monkeypatch):
    """From test_wf1_the_rules_raise_before_calling_the_entries: the plan refuses the block, naming the unknown key,
    and neither entry is called."""
    calls = (spy(monkeypatch, "report"), spy(monkeypatch, "compare_samples"))
    config = report_config(tmp_path, {"title": "LN"}, {})
    T.write_inputs(config, membership([(SAMPLE, 1, 2)]))
    with pytest.raises(PlanRefused, match=r"unknown report keys: \['title'\]"):
        T.plan(tmp_path, "refused", config)
    assert calls == ([], []) and not T.output_root(config).exists()


def test_the_report_and_compare_commands_write_the_report_and_the_comparison_page(tmp_path):
    """From test_wf1_the_rules_write_the_report_and_the_comparison_page: the real entries through both commands on
    report_records (hand-written run records and placement of sample s1, FOVs F1 to F4): the report task writes
    report/s1/ with its report.json, the compare task report/samples.csv and report/index.html."""
    from .report_fixtures import DATASET, FOVS, SAMPLE_ID, report_records
    from .report_fixtures import SAMPLE as REPORT_SAMPLE
    config = report_config(tmp_path, {"marker_pairs": [["G1", "G2"]]}, {"stitching": MERGED})
    config.update(dataset_id=DATASET, sample_id=SAMPLE_ID, output_id="output")
    T.write_inputs(config, "sample,fov_id\n" + "".join(f"{REPORT_SAMPLE},{fov}\n" for fov in FOVS))
    manifest, _ = T.plan(tmp_path, "records", config)
    out = T.output_root(config)
    case = report_records(out)
    assert list(case.fov_ids) == list(FOVS)
    task = T.task(manifest, "report", sample=REPORT_SAMPLE)
    assert f"stitching/{REPORT_SAMPLE}/placement.json" in task["inputs"]
    assert (out / "stitching" / REPORT_SAMPLE / "placement.json").is_file()
    T.call(manifest, f"report/{REPORT_SAMPLE}")
    record = load_report(out / "report" / REPORT_SAMPLE)
    assert (out / "report" / REPORT_SAMPLE / "report.json").is_file()
    assert (out / "report" / REPORT_SAMPLE / "report.html").is_file()
    assert record["sample"] == REPORT_SAMPLE and record["fov_ids"] == list(FOVS)
    T.call(manifest, "compare")
    samples = pd.read_csv(out / "report" / "samples.csv")
    assert samples.iloc[:, 0].tolist() == [REPORT_SAMPLE] and "fovs_expected" in samples
    assert (out / "report" / "index.html").is_file()
