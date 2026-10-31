"""The golden baseline of the Snakemake workflow (design/baselines/orchestration.md, "The golden baseline").

G3 of design/validation/orchestration.md: the refactored workflow (``workflow/Snakefile``, the dispatcher over the
plan's manifest) on W-642's fixture and A to D migrated to the Python shape (``scripts/workflow_golden.py``,
``migrated_configurations``) reproduces the expected values that W-642 (invocations A, A2, B and C, G1) and W-643
(invocation D, G2) recorded with the workflow before the dispatcher, except the contract's exceptions
(design/specs/orchestration.md, "The exception of the golden comparison"), each difference named by its decision
(``workflow_golden.compare_migrated``: OC3, D6, D8). G1 and G2 ran that workflow, which no longer exists for
``backend: python``; G3 compares with the same two expected folders, which no run of this workflow re-records. The
runs use Snakemake 9 from the ``workflow`` dependency group, the one beside the interpreter, and skip with a stated
reason where it is absent. The file digests and the stage-record digests are identities of the recording environment
only (the expected folders' README.md), so the comparison leaves them out.
"""
import importlib.util
import json
import shutil
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pytest

pytestmark = [pytest.mark.workflow]

ROOT = Path(__file__).resolve().parents[3]
EXPECTED = Path(__file__).parent / "data" / "expected" / "test_workflow_orchestration_golden"
EXPECTED_EXPORT = Path(__file__).parent / "data" / "expected" / "test_workflow_orchestration_golden_export"
EXPECTED_CORE = Path(__file__).parent / "data" / "expected" / "test_workflow_core_golden"
NO_SNAKEMAKE = "the golden comparison runs the workflow with Snakemake 9, which this environment does not have"
#: The decisions that may explain a difference (the contract's exceptions; OC4 explains none here: the records'
#: absolute paths under the work folder are written <work> on both sides).
DECISIONS = {"OC3", "OC4", "D6", "D8"}


def _snakemake():
    """The Snakemake executable of this interpreter's environment (the ``workflow`` group), or a skip."""
    executable = Path(sys.executable).with_name("snakemake")
    if importlib.util.find_spec("snakemake") is None or not executable.is_file():
        pytest.skip(NO_SNAKEMAKE)
    return executable


def _golden():
    spec = importlib.util.spec_from_file_location("workflow_golden", ROOT / "src" / "python" / "scripts" /
                                                  "workflow_golden.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def _set_json(path, change):
    data = json.loads(path.read_text())
    change(data)
    path.write_text(json.dumps(data, indent=1) + "\n")


def _values(change):
    return lambda folder: _set_json(folder / "values.json", change)


def _record(name, change):
    return lambda folder: _set_json(folder / "records" / name, change)


def _fixture_digest(values):
    values[next(key for key in sorted(values) if key.startswith("fixture:"))]["value"] = "0" * 64


def _table_value(folder):
    table = folder / "FOV_001-molecules.csv"
    text = table.read_text()
    assert text.count(",0.1807337788883871\n") == 1
    table.write_text(text.replace(",0.1807337788883871\n", ",0.5\n"))


def _core_table_value(folder):
    table = folder / "FOV_001-molecules.csv"
    text = table.read_text()
    assert text.count(",0.22604338288727557\n") == 1
    table.write_text(text.replace(",0.22604338288727557\n", ",0.5\n"))


def _set(*path, value):
    def change(data):
        for key in path[:-1]:
            data = data[key]
        data[path[-1]] = value(data[path[-1]])
    return change


OTHER = lambda _: "0" * 64  # noqa: E731
# The skips of a comparison across environments (W-751): the fixture's digests and the report files' digests, at any
# depth of the records, and the output-file digests the per-FOV records hold of pre_qc.csv and molecules.csv (W-647).
# Each case changes one value of a copy of a committed expected folder.
SKIPPED = {
    "fixture digest": (EXPECTED, _values(_fixture_digest), "fixture:"),
    "report file digest": (EXPECTED, _record("report/sample/report.json", _set("files", 2, "sha256", value=OTHER)),
                           "records/report/sample/report.json.files[2].sha256: "),
    "report file digest of a route": (EXPECTED_EXPORT, _record("per_fov/report/sample/report.json",
                                                               _set("files", 2, "sha256", value=OTHER)),
                                      "records/per_fov/report/sample/report.json.files[2].sha256: "),
    "run output digest": (EXPECTED, _record("fovs/FOV_001/run.json", _set("outputs", "molecules", "sha256",
                                                                            value=OTHER)),
                          "records/fovs/FOV_001/run.json.outputs.molecules.sha256: "),
    "core run output digest": (EXPECTED_CORE, _record("fovs/FOV_001/run.json", _set("outputs", "molecules", "sha256",
                                                                                     value=OTHER)),
                               "records/fovs/FOV_001/run.json.outputs.molecules.sha256: "),
    "core pre_qc table digest": (EXPECTED_CORE, _record("fovs/FOV_002/checkpoints/pre_qc.json",
                                                        _set("files", 0, "sha256", value=OTHER)),
                                 "records/fovs/FOV_002/checkpoints/pre_qc.json.files[0].sha256: "),
}
# What the skips leave in: each change is reported in both comparisons.
REPORTED = {
    "table value": (EXPECTED, _table_value, "FOV_001-molecules.csv: "),
    "image statistic": (EXPECTED, _values(_set("bigstitcher/sample/tiles/tile_0.tif:max", "value",
                                               value=lambda v: v + 1)),
                        "bigstitcher/sample/tiles/tile_0.tif:max: "),
    "record integer": (EXPECTED, _record("fovs/FOV_001/run.json", _set("outputs", "molecules", "rows",
                                                                         value=lambda v: v + 1)),
                       "records/fovs/FOV_001/run.json.outputs.molecules.rows: "),
    "report input digest": (EXPECTED, _record("report/sample/report.json", _set("inputs", 4, "sha256", value=OTHER)),
                            "records/report/sample/report.json.inputs[4].sha256: "),
    "core table value": (EXPECTED_CORE, _core_table_value, "FOV_001-molecules.csv: "),
    "core stage count": (EXPECTED_CORE, _record("fovs/FOV_001/run.json", _set("counts", "spots", value=lambda v: v + 1)),
                         "records/fovs/FOV_001/run.json.counts.spots: "),
    "core candidates table digest": (EXPECTED_CORE, _record("fovs/FOV_001/checkpoints/candidates.json",
                                                            _set("files", 0, "sha256", value=OTHER)),
                                     "records/fovs/FOV_001/checkpoints/candidates.json.files[0].sha256: "),
    "core input digest": (EXPECTED_CORE, _record("fovs/FOV_001/run.json", _set("inputs", 0, "sha256", value=OTHER)),
                          "records/fovs/FOV_001/run.json.inputs[0].sha256: "),
}


def _changed(tmp_path, expected, change):
    actual = tmp_path / "actual"
    shutil.copytree(expected, actual)
    change(actual)
    return actual


@pytest.mark.contract
@pytest.mark.parametrize("case", sorted(SKIPPED))
def test_a_comparison_across_environments_skips_only_the_fixture_report_and_output_file_digests(tmp_path, case):
    G = _golden()
    expected, change, prefix = SKIPPED[case]
    actual = _changed(tmp_path, expected, change)
    assert G.compare(expected, actual, same_environment=False) == []
    lines = G.compare(expected, actual, same_environment=True)
    assert len(lines) == 1 and lines[0].startswith(prefix), lines


@pytest.mark.contract
@pytest.mark.parametrize("case", sorted(REPORTED))
def test_a_comparison_across_environments_reports_every_other_change(tmp_path, case):
    G = _golden()
    expected, change, prefix = REPORTED[case]
    actual = _changed(tmp_path, expected, change)
    for same_environment in (False, True):
        lines = G.compare(expected, actual, same_environment=same_environment)
        assert len(lines) == 1 and lines[0].startswith(prefix), (same_environment, lines)


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


@pytest.mark.contract
def test_the_snakemake_runs_have_one_numerical_thread_whatever_the_caller_sets(tmp_path, monkeypatch):
    G = _golden()
    for name in G.THREAD_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "8")
    environment = G._environment(tmp_path)
    assert {name: environment[name] for name in G.THREAD_VARIABLES + ("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS",)} == \
        dict.fromkeys(G.THREAD_VARIABLES + ("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS",), "1")


@pytest.mark.contract
def test_the_code_identity_reads_the_package_metadata(tmp_path, monkeypatch):
    # The golden runs leave `code` out of every record; what it holds is the contract's code identity
    # (design/specs/orchestration.md, "The code identity"): the metadata's version in a checkout and outside one,
    # the live commit only in a checkout.
    import starfinder
    import starfinder._records as records
    try:
        expected = version("starfinder")
    except PackageNotFoundError:
        expected = None
    assert starfinder.__version__ == (expected or "0+unknown")
    records.code.cache_clear()
    try:
        checkout = records.code()
    finally:
        records.code.cache_clear()
    assert checkout["version"] == expected
    # An installed wheel: the package outside any checkout.
    monkeypatch.setattr(records, "__file__", str(tmp_path / "site-packages" / "starfinder" / "_records.py"))
    (tmp_path / "site-packages" / "starfinder").mkdir(parents=True)
    try:
        installed = records.code()
    finally:
        records.code.cache_clear()
    assert installed == {"version": expected, "git_commit": None, "git_dirty": None}
