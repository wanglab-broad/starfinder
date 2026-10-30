"""The golden baseline of the current workflow (design/baselines/orchestration.md, "The golden baseline").

G1 and G2 of design/validation/orchestration.md: today's Snakemake workflow (``backend: python``) on W-642's fixture
reproduces the expected values that W-642 (invocations A, A2, B and C) and W-643 (invocation D) recorded with
``scripts/workflow_golden.py``. They run Snakemake 9 from the ``workflow`` dependency group, the one beside the
interpreter, and skip with a stated reason where it is absent. The file digests and the stage-record digests are
identities of the recording environment only (the expected folders' README.md), so the comparison leaves them out.
"""
import importlib.util
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
