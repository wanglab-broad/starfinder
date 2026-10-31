"""The golden baseline of the current workflow (design/baselines/orchestration.md, "The golden baseline").

G1 and G2 of design/validation/orchestration.md: today's Snakemake workflow (``backend: python``) on W-642's fixture
reproduces the expected values that W-642 (invocations A, A2, B and C) and W-643 (invocation D) recorded with
``scripts/workflow_golden.py``. They run Snakemake 9 from the ``workflow`` dependency group, the one beside the
interpreter, and skip with a stated reason where it is absent. The file digests and the stage-record digests are
identities of the recording environment only (the expected folders' README.md), so the comparison leaves them out.
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
NO_SNAKEMAKE = "the golden baseline runs the workflow with Snakemake 9, which this environment does not have"


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


@pytest.mark.golden
@pytest.mark.slow
def test_the_workflow_reproduces_the_golden_values(tmp_path):
    snakemake = _snakemake()
    G = _golden()
    work = tmp_path / "work"
    G.write_fixture(work)
    G.run_workflow(work, snakemake, ROOT)
    G.extract(work, tmp_path / "actual")
    assert G.compare(EXPECTED, tmp_path / "actual", same_environment=False) == []


@pytest.mark.golden
@pytest.mark.slow
def test_the_workflow_reproduces_the_golden_values_of_the_export(tmp_path):
    snakemake = _snakemake()
    if importlib.util.find_spec("multiview_stitcher") is None or importlib.util.find_spec("anndata") is None:
        pytest.skip("invocation D needs the multiview-stitcher and anndata extras")
    G = _golden()
    work = tmp_path / "work"
    G.write_fixture(work)
    G.run_export(work, snakemake, ROOT)
    G.extract_export(work, tmp_path / "actual")
    assert G.compare(EXPECTED_EXPORT, tmp_path / "actual", same_environment=False) == []


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


def _set(*path, value):
    def change(data):
        for key in path[:-1]:
            data = data[key]
        data[path[-1]] = value(data[path[-1]])
    return change


OTHER = lambda _: "0" * 64  # noqa: E731
# The skips of a comparison across environments (W-751): the fixture's digests and the report files' digests, at any
# depth of the records. Each case changes one value of a copy of a committed expected folder.
SKIPPED = {
    "fixture digest": (EXPECTED, _values(_fixture_digest), "fixture:"),
    "report file digest": (EXPECTED, _record("report/sample/report.json", _set("files", 2, "sha256", value=OTHER)),
                           "records/report/sample/report.json.files[2].sha256: "),
    "report file digest of a route": (EXPECTED_EXPORT, _record("per_fov/report/sample/report.json",
                                                               _set("files", 2, "sha256", value=OTHER)),
                                      "records/per_fov/report/sample/report.json.files[2].sha256: "),
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
    "run output digest": (EXPECTED, _record("fovs/FOV_001/run.json", _set("outputs", "molecules", "sha256",
                                                                            value=OTHER)),
                          "records/fovs/FOV_001/run.json.outputs.molecules.sha256: "),
}


def _changed(tmp_path, expected, change):
    actual = tmp_path / "actual"
    shutil.copytree(expected, actual)
    change(actual)
    return actual


@pytest.mark.contract
@pytest.mark.parametrize("case", sorted(SKIPPED))
def test_a_comparison_across_environments_skips_only_the_fixture_and_report_file_digests(tmp_path, case):
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
