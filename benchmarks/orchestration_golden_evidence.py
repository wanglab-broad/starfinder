"""G3, the refactored workflow against the golden values of the workflow before the dispatcher, as an evidence script.

What it protects: W-647's refactor of the Snakemake workflow. G3 of design/validation/orchestration.md runs the
refactored workflow (``workflow/Snakefile``, the dispatcher over the plan's manifest) on W-642's fixture with A to D
migrated to the Python shape (``scripts/workflow_golden.py``, ``migrated_configurations``) and compares the outputs
with the expected values that W-642 (invocations A, A2, B and C, G1) and W-643 (invocation D, G2) recorded with the
workflow before the dispatcher, except the contract's exceptions (design/specs/orchestration.md, "The exception of the
golden comparison"), each difference named by its decision (``workflow_golden.compare_migrated``: OC3, D6, D8, IR,
W-909).
The tests of ``compare_migrated`` check that each exception explains only its own difference. W-898 moved these tests
out of the suite (category K7 of design/proposals/test-suite-slimming.md): a pin of a finished refactor, whose task
commands' values ``test_workflow_core_golden.py``, ``test_workflow_cell_golden.py`` and
``test_workflow_stitching_golden.py`` pin in the suite, and which every change of a default breaks. The tests are those
of ``src/python/test/test_workflow_orchestration_golden.py`` at ``ORIGIN_REVISION``, as pytest tests with their
markers, parameters, fixtures and assertions as they were; the helpers that the suite's comparison tests share
(``_golden``, ``_record``, ``_set``, ``_values``, ``_table_value``, ``OTHER`` and the expected folders) are imported
from there. The expected folders stay in ``src/python/test/data/expected/`` (``test_workflow_orchestration_golden/`` and
``…_golden_export/``), which the suite's comparison tests and test_workflow_cell_golden.py read too.

Software: Snakemake 9 beside the interpreter (the ``workflow`` dependency group); G2's part needs the ``anndata`` and
``multiview-stitcher`` extras. A missing one skips with the reason (exit status 2). It reads no real data: the runs
write W-642's fixture into their temporary folder. About 2.5 minutes on one CPU for the two runs (W-880's measured
times), a few seconds for the comparison tests; under 4 GiB.

When it runs: when the engine script does, that is, at the end of a batch whose changes touch the workflow package
(``src/python/starfinder/workflow/``) or ``nextflow/``; also after a change to ``workflow/`` or
``scripts/workflow_golden.py``; and with every evidence script before a tag or a release (Jiahao, 2026-11-19). The
coordinating session runs it on UGER, in the image with the commit's code laid over it.

    cd src/python
    uv run python ../../benchmarks/orchestration_golden_evidence.py --output <run dir>/orchestration-golden
    uv run python ../../benchmarks/orchestration_golden_evidence.py --list

``-k`` selects cases as pytest's ``-k`` does, and ``--case <test>[<parameter>]`` names one (repeatable);
``evidence_runner.py`` describes the output and the exit status. Do not
run with ``python -O``, which drops the assertions.
"""
import importlib.util
import json
from pathlib import Path
import re
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "python"
if str(PACKAGE) not in sys.path[:1]:
    sys.path.insert(0, str(PACKAGE))

from test.test_workflow_orchestration_golden import (  # noqa: E402
    EXPECTED, EXPECTED_EXPORT, OTHER, _golden, _record, _set, _table_value, _values)

pytestmark = [pytest.mark.workflow]

# The last revision whose test suite held these tests (W-898's starting revision).
ORIGIN_REVISION = "472d2f6ffaaec6f2764873ad971784e1fbec2df2"
HERE = "benchmarks/orchestration_golden_evidence.py"
ORIGIN = "src/python/test/test_workflow_orchestration_golden.py"
NO_SNAKEMAKE = "the golden comparison runs the workflow with Snakemake 9, which this environment does not have"
#: The decisions that may explain a difference (the contract's exceptions; OC4 explains none here: the records'
#: absolute paths under the work folder are written <work> on both sides). W-909: the two keys of the codec in run.json
#: (design/specs/compressed-image-outputs.md, migration M3).
DECISIONS = {"OC3", "OC4", "D6", "D8", "IR", "W-909"}


def _snakemake():
    """The Snakemake executable of this interpreter's environment (the ``workflow`` group), or a skip."""
    executable = Path(sys.executable).with_name("snakemake")
    if importlib.util.find_spec("snakemake") is None or not executable.is_file():
        pytest.skip(NO_SNAKEMAKE)
    return executable


def _explained(unexplained, explained):
    assert unexplained == [], "\n".join(unexplained)
    assert set(explained) <= DECISIONS, set(explained)


@pytest.mark.golden
@pytest.mark.slow
def test_the_refactored_workflow_reproduces_the_golden_values(tmp_path):
    """G3 on G1's values: the migrated A, B and C (the fov, morphology, report, compare, segment import, assign and
    BigStitcher macro tasks)."""
    snakemake = _snakemake()
    G = _golden()
    work = tmp_path / "work"
    G.write_fixture(work)
    G.run_migrated(work, snakemake, ROOT, names=("A", "B", "C"))
    G.extract(work, tmp_path / "actual", migrated=True)
    _explained(*G.compare_migrated(EXPECTED, tmp_path / "actual", tmp_path / "expected"))


@pytest.mark.golden
@pytest.mark.slow
def test_the_refactored_workflow_reproduces_the_golden_values_of_the_export(tmp_path):
    """G3 on G2's values: the migrated invocation D (the stitch, export, report and compare tasks on three routes)."""
    snakemake = _snakemake()
    if importlib.util.find_spec("multiview_stitcher") is None or importlib.util.find_spec("anndata") is None:
        pytest.skip("invocation D needs the multiview-stitcher and anndata extras")
    G = _golden()
    work = tmp_path / "work"
    G.write_fixture(work)
    G.run_migrated(work, snakemake, ROOT, names=("D-per-fov", "D-molecules-only", "D-after-fusion"))
    G.extract_export(work, tmp_path / "actual")
    _explained(*G.compare_migrated(EXPECTED_EXPORT, tmp_path / "actual", tmp_path / "expected"))
    # W-879: the segment, assign_molecules, stitch and assembly records hold their wall time on every route, so the
    # report of the 4-FOV run names no unavailable step time (the expected tables hold no such row either).
    for route in ("per_fov", "molecules_only", "after_fusion"):
        table = tmp_path / "actual" / f"{route}-report-sample-tables-unavailable.csv"
        assert "workflow.time" not in table.read_text(), route


# G3's exceptions (workflow_golden.compare_migrated): each case changes one value of a copy of an expected folder with
# OC3 applied (the refactored workflow's form) and names the decision that explains it, or None.
MIGRATED_CASES = {
    "reference image digest": (EXPECTED, _record("fovs/FOV_001/run.json", _set("outputs", "reference_image", "sha256",
                                                                               value=OTHER)), "OC3"),
    "report input digest of a reference image": (EXPECTED, _record("report/sample/report.json",
                                                                   _set("inputs", 12, "sha256", value=OTHER)), "OC3"),
    "report input digest of a raw image": (EXPECTED, _record("report/sample/report.json",
                                                             _set("inputs", 4, "sha256", value=OTHER)), None),
    "new task record": (EXPECTED, _values(lambda v: v.update({"file:workflow/tasks/fov.sample.FOV_001.json": {
        "value": "0" * 64, "compare": "same_environment"}})), "D8"),
    "retired image": (EXPECTED, _values(lambda v: v.pop("images/DAPI/FOV_001.tif:sum")), "D6"),
    "table value": (EXPECTED, _table_value, None),
    "record integer": (EXPECTED, _record("fovs/FOV_001/run.json", _set("outputs", "molecules", "rows",
                                                                         value=lambda v: v + 1)), None),
    "a step of the run": (EXPECTED, _record("fovs/FOV_001/run.json", lambda r: r["steps"].pop()), None),
    "an export image statistic": (EXPECTED_EXPORT, _values(_set("per_fov/export/sample/cell_labels.ome.tif:max",
                                                      "value", value=lambda v: v + 1)), None),
    "the cell labels' path in the assignment record": (EXPECTED, _record(
        "fovs/FOV_001/assignment/default/assignment.json",
        _set("inputs", "cells", "file", "path", value=lambda _: "../../segmentation/stardist_segmentation/labels.tif")),
        "D6"),
    "another file as the cell labels' path": (EXPECTED, _record(
        "fovs/FOV_001/assignment/default/assignment.json",
        _set("inputs", "cells", "file", "path", value=lambda _: "../../segmentation/stardist_segmentation/other.tif")),
        None),
    "the imported label file's path": (EXPECTED, _record(
        "fovs/FOV_001/assignment/default/assignment.json",
        _set("inputs", "cells", "record", "import", "path", value=lambda _: "<work>/labels/FOV_001.tif")), "D6"),
    "another FOV's imported label file": (EXPECTED, _record(
        "fovs/FOV_001/assignment/default/assignment.json",
        _set("inputs", "cells", "record", "import", "path", value=lambda _: "<work>/labels/FOV_002.tif")), None),
    "the cell run's record of format version 2": (EXPECTED, _record(
        "fovs/FOV_001/assignment/default/assignment.json",
        _set("inputs", "cells", "record", "format_version", value=lambda _: 2)), "IR"),
    "the cell run's record of another format version": (EXPECTED, _record(
        "fovs/FOV_001/assignment/default/assignment.json",
        _set("inputs", "cells", "record", "format_version", value=lambda _: 3)), None),
    "the export's cell record of format version 2": (EXPECTED_EXPORT, _record(
        "after_fusion/export/sample/export.json",
        _set("assembly", "assignment", "inputs", "cells", "record", "format_version", value=lambda _: 2)), "IR"),
    "the export run's recipe": (EXPECTED_EXPORT, _record(
        "after_fusion/export/sample/export.json",
        lambda r: r["segmentation"]["runs"][0].update(recipe=None)), "IR"),
    "another key of the export run": (EXPECTED_EXPORT, _record(
        "after_fusion/export/sample/export.json",
        lambda r: r["segmentation"]["runs"][0].update(colour="red")), None),
}


@pytest.mark.contract
@pytest.mark.parametrize("case", sorted(MIGRATED_CASES))
def test_the_migrated_comparison_explains_only_the_contracts_exceptions(tmp_path, case):
    G = _golden()
    expected, change, decision = MIGRATED_CASES[case]
    actual = G.expected_with_exceptions(expected, tmp_path / "actual")
    assert G.compare_migrated(expected, actual, tmp_path / "unchanged") == ([], {})
    change(actual)
    unexplained, explained = G.compare_migrated(expected, actual, tmp_path / "expected")
    lines = unexplained + [line for found in explained.values() for line in found]
    assert len(lines) == 1, lines
    assert (unexplained == [] and list(explained) == [decision]) if decision else unexplained == lines


# The report's inputs (workflow_golden._report_input_differences): a copy of the per-FOV route's expected values with
# the label entries swapped as the refactored workflow swaps them (each FOV's copied cell_labels.tif becomes the
# segment task's labels.tif, and every FOV's segmentation.json joins), so that the lists differ in length as they do
# on that route; then one change per case.
REPORT = "per_fov/report/sample/report.json"
SAVED_RUN = "fovs/{fov}/segmentation/stardist_segmentation/{name}"


def _swap_labels(inputs):
    out = []
    for entry in inputs:
        match = re.match(r"fovs/(FOV_\d+)/assignment/[^/]+/cell_labels\.tif$", entry["path"])
        out.append(dict(entry, path=SAVED_RUN.format(fov=match[1], name="labels.tif")) if match else entry)
    records = [{"path": SAVED_RUN.format(fov=f"FOV_00{n}", name="segmentation.json"), "kind": "segmentation_record",
                "sha256": "1" * 64, "bytes": 2752, "verified": True} for n in range(1, 5)]
    return out[:4] + records + out[4:]


def _entry(inputs, end):
    return next(entry for entry in inputs if entry["path"].endswith(end))


def _exchange(inputs):
    i, j = (inputs.index(_entry(inputs, end)) for end in ("tables/molecules.csv", "tables/cells.csv"))
    inputs[i], inputs[j] = inputs[j], inputs[i]


#: case: (change of the swapped inputs, the unexplained lines, the explained lines by decision).
REPORT_INPUT_CASES = {
    "the label swap alone": (lambda i: None, 0, {"D6": 12}),
    "a raw image's digest": (lambda i: _entry(i, "round1/FOV_001/ch00.ome.tif").update(sha256="0" * 64), 1, {"D6": 12}),
    "a retained input's size": (lambda i: _entry(i, "stitching/sample/fovs.csv").update(bytes=1), 1, {"D6": 12}),
    "a retained input's kind": (lambda i: _entry(i, "export/sample/tables/fovs.csv").update(kind="placement_table"),
                                1, {"D6": 12}),
    "a retained input's verified flag": (lambda i: _entry(i, "tables/cells.csv").update(verified=False), 1, {"D6": 12}),
    "a retained input removed": (lambda i: i.remove(_entry(i, "stitching/sample/links.csv")), 1, {"D6": 12}),
    "a new input that is no label entry": (lambda i: i.append({"path": "fovs/FOV_001/other.tif", "kind": "label_image",
                                                               "verified": True}), 1, {"D6": 12}),
    "two retained inputs exchanged": (_exchange, 6, {"D6": 12}),
    "a labels.tif of a FOV without copied labels": (lambda i: i.append(dict(
        _entry(i, "FOV_001/segmentation/stardist_segmentation/labels.tif"),
        path=SAVED_RUN.format(fov="FOV_005", name="labels.tif"))), 1, {"D6": 13}),
    "copied labels without a labels.tif": (lambda i: i.remove(_entry(i, "FOV_002/segmentation/stardist_segmentation/"
                                                                        "labels.tif")), 1, {"D6": 11}),
    "a label entry not verified": (lambda i: _entry(i, "FOV_001/segmentation/stardist_segmentation/labels.tif").update(
        verified=False), 1, {"D6": 11}),
    "a reference image's digest": (lambda i: _entry(i, "FOV_001/reference.tif").update(sha256="0" * 64), 0,
                                   {"D6": 12, "OC3": 1}),
    "a reference image's size": (lambda i: _entry(i, "FOV_002/reference.tif").update(bytes=7672), 0,
                                 {"D6": 12, "OC3": 1}),
}


@pytest.mark.contract
@pytest.mark.parametrize("case", sorted(REPORT_INPUT_CASES))
def test_the_migrated_comparison_compares_the_report_inputs_entry_by_entry(tmp_path, case):
    G = _golden()
    change, unexplained, explained = REPORT_INPUT_CASES[case]
    actual = G.expected_with_exceptions(EXPECTED_EXPORT, tmp_path / "actual")
    record = json.loads((actual / "records" / REPORT).read_text())
    record["inputs"] = _swap_labels(record["inputs"])
    change(record["inputs"])
    (actual / "records" / REPORT).write_text(json.dumps(record, indent=1) + "\n")
    lines, decided = G.compare_migrated(EXPECTED_EXPORT, actual, tmp_path / "expected")
    assert len(lines) == unexplained, lines
    assert {d: len(found) for d, found in decided.items()} == explained, decided
    assert all(line.startswith(f"records/{REPORT}.inputs") for found in decided.values() for line in found)
    if case == "the label swap alone":
        named = sorted(re.search(r": (\S+) only in", line)[1] for line in decided["D6"])
        assert named == sorted([f"fovs/FOV_00{n}/assignment/default/cell_labels.tif" for n in range(1, 5)]
                               + [SAVED_RUN.format(fov=f"FOV_00{n}", name="labels.tif") for n in range(1, 5)]
                               + [SAVED_RUN.format(fov=f"FOV_00{n}", name="segmentation.json") for n in range(1, 5)])

def _moved():
    """Each case's node ID here: (its node ID in the suite at ORIGIN_REVISION, its group K7)."""
    names = ["test_the_refactored_workflow_reproduces_the_golden_values",
             "test_the_refactored_workflow_reproduces_the_golden_values_of_the_export"]
    names += [f"test_the_migrated_comparison_explains_only_the_contracts_exceptions[{case}]"
              for case in sorted(MIGRATED_CASES)]
    names += [f"test_the_migrated_comparison_compares_the_report_inputs_entry_by_entry[{case}]"
              for case in sorted(REPORT_INPUT_CASES)]
    return {f"{HERE}::{name}": (f"{ORIGIN}::{name}", "K7") for name in names}


def main(argv=None):
    spec = importlib.util.spec_from_file_location("evidence_runner", Path(__file__).resolve().parent / "evidence_runner.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    return runner.run(__file__, _moved(), ORIGIN_REVISION, argv,
                      description="Run G3 of design/validation/orchestration.md (the refactored workflow against the "
                                  "golden values of W-642 and W-643) and the tests of its comparison, which W-898 "
                                  "moved out of the test suite. Needs Snakemake 9; G2's part also the anndata and "
                                  "multiview-stitcher extras.",
                      epilog="Run from src/python: uv run python ../../benchmarks/orchestration_golden_evidence.py "
                             "--output <run dir>/orchestration-golden")


if __name__ == "__main__":
    sys.exit(main())
