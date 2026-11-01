"""X1–X3: real engines on tiny_sample, completion hashes and strict output comparison."""
import pytest

from . import workflow_engine_validation as E

pytestmark = [pytest.mark.workflow, pytest.mark.integration, pytest.mark.extended, pytest.mark.slow]


@pytest.mark.parametrize("variant", ["per-fov", "molecules-only"])
def test_x1_snakemake_end_to_end(tmp_path, variant):
    run = E.EngineRun(tmp_path / variant, "snakemake", variant)
    run.invoke({t["id"] for t in run.data["tasks"]})


@pytest.mark.nextflow
@pytest.mark.parametrize("variant", ["per-fov", "molecules-only", "supplied-xml"])
def test_x2_nextflow_end_to_end(tmp_path, variant):
    run = E.EngineRun(tmp_path / variant, "nextflow", variant)
    run.invoke({t["id"] for t in run.data["tasks"]})


@pytest.mark.nextflow
@pytest.mark.parametrize("variant", ["per-fov", "molecules-only"])
def test_x3_engines_have_byte_equal_declared_outputs(tmp_path, variant):
    # Same input bytes, mtimes, resolved configuration and output path in one environment.
    # Each engine starts with no output tree and its own clean work/cache folder.
    E.require("nextflow")
    run = E.EngineRun(tmp_path / variant, "snakemake", variant)
    run.invoke({t["id"] for t in run.data["tasks"]})
    first = E.snapshot(run)
    import shutil
    shutil.rmtree(run.out)
    run.engine = "nextflow"
    E.require(run.engine)
    run.invoke({t["id"] for t in run.data["tasks"]})
    E.compare(run, first, E.snapshot(run), name=f"x3-{variant}.json", left_engine="snakemake", right_engine="nextflow")
