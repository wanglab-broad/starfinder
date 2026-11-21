"""The workflow engine checks that W-898 moved out of the test suite, as an evidence script.

What it protects: the reruns of the two workflow engines on a complete ``tiny_sample`` per-fov run, compared with a
fresh run on the changed configuration or inputs, and the declared spacing of an uncalibrated run's grids
(design/validation/orchestration.md, X11 and the grid checks of ``test_workflow_engines_e2e.py``; the Nextflow cache
regression of W-650). The tests are those of the suite at ``ORIGIN_REVISION``, as pytest tests with their markers,
parameters, fixtures and assertions as they were; they import the suite's helpers from ``src/python/test``. W-893's
proposal (design/proposals/test-suite-slimming.md, "The workflow recovery and cache tests") names the groups:

* K1: X11 ``restored-twice``, ``raw-size`` and ``raw-older`` on Nextflow (the A, B, A, B sequence and the two raw
  replacements, which the suite's stub cache test makes on Nextflow);
* O2: X11 ``restored-twice`` on Snakemake (the suite keeps X11 ``processing`` on both engines);
* K2: the cache test on real tasks (the suite keeps its stub variant);
* K3: the uncalibrated run on Nextflow (the suite keeps the Snakemake one, which asserts the same grids on the same
  commands).

Each case makes its own complete run from scratch, as it did in the suite: about 4 to 12 minutes per case on one
CPU, about 40 minutes for the six (W-880's measured times, W-893's inventory), under 1 GiB beside Nextflow's JVM.

Software: Snakemake 9 beside the interpreter (the ``workflow`` dependency group), Nextflow 26.10.4 or newer on
``PATH``, running on Java 17 or newer, and the ``anndata`` and ``multiview-stitcher`` extras. A case whose engine or
extra is missing skips with the reason (exit status 2). It reads no real data: every case writes ``tiny_sample``
(``scripts/workflow_golden.py``) into its temporary folder.

When it runs: at the end of a batch whose changes touch the workflow package (``src/python/starfinder/workflow/``) or
``nextflow/``, and with every evidence script before a tag or a release (Jiahao, 2026-11-19); the coordinating session
runs it on UGER, in the image with the commit's code laid over it.

    cd src/python
    uv run python ../../benchmarks/workflow_engine_evidence.py --output <run dir>/workflow-engines
    uv run python ../../benchmarks/workflow_engine_evidence.py --list

``-k`` selects cases as pytest's ``-k`` does, and ``--case <test>[<parameter>]`` names one (repeatable);
``evidence_runner.py`` describes the output and the exit status. Do not
run with ``python -O``, which drops the assertions.
"""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "python"
if str(PACKAGE) not in sys.path[:1]:
    sys.path.insert(0, str(PACKAGE))

from test import workflow_engine_validation as E  # noqa: E402
from test import workflow_fixture as F  # noqa: E402
from test import workflow_nextflow as N  # noqa: E402
from test import workflow_variants as V  # noqa: E402
from test.test_workflow_engines_e2e import _assert_declared_grids, _uncalibrated_run  # noqa: E402
from test.test_workflow_nextflow_cache import downstream  # noqa: E402
from test.test_workflow_recovery import FOV_TASK  # noqa: E402

# The last revision whose test suite held these tests (W-898's starting revision).
ORIGIN_REVISION = "472d2f6ffaaec6f2764873ad971784e1fbec2df2"
HERE = "benchmarks/workflow_engine_evidence.py"
RECOVERY = "src/python/test/test_workflow_recovery.py"
ENGINES_E2E = "src/python/test/test_workflow_engines_e2e.py"
CACHE = "src/python/test/test_workflow_nextflow_cache.py"
X11 = "test_x11_stale_results_rerun_only_readers_and_downstream"
#: Each case's node ID here: (its node ID in the suite at ORIGIN_REVISION, its group of W-893's proposal).
MOVED = {
    f"{HERE}::{X11}[restored-twice-snakemake]": (f"{RECOVERY}::{X11}[restored-twice-snakemake]", "O2"),
    f"{HERE}::{X11}[restored-twice-nextflow]": (f"{RECOVERY}::{X11}[restored-twice-nextflow]", "K1"),
    f"{HERE}::{X11}[raw-size-nextflow]": (f"{RECOVERY}::{X11}[raw-size-nextflow]", "K1"),
    f"{HERE}::{X11}[raw-older-nextflow]": (f"{RECOVERY}::{X11}[raw-older-nextflow]", "K1"),
    f"{HERE}::test_fingerprints_and_fresh_tokens_rerun_only_affected_tasks[real]": (
        f"{CACHE}::test_fingerprints_and_fresh_tokens_rerun_only_affected_tasks[real]", "K2"),
    f"{HERE}::test_uncalibrated_tiny_sample_nextflow": (f"{ENGINES_E2E}::test_uncalibrated_tiny_sample_nextflow", "K3"),
}


# --- K1 and O2: from test_workflow_recovery.py (its module markers: workflow, dataset, contract, extended) ----------

@pytest.mark.workflow
@pytest.mark.dataset
@pytest.mark.contract
@pytest.mark.extended
@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("case, engine", [
    ("restored-twice", "snakemake"),
    *(pytest.param(case, "nextflow", marks=pytest.mark.nextflow) for case in ("restored-twice", "raw-size", "raw-older"))])
def test_x11_stale_results_rerun_only_readers_and_downstream(tmp_path, engine, case):
    import copy
    import os
    from starfinder.__main__ import main
    from starfinder.workflow import verify
    run = E.EngineRun(tmp_path / case, engine)
    run.invoke({t["id"] for t in run.data["tasks"]})
    if case == "fingerprint":
        task = next(t for t in run.data["tasks"] if t["id"] == FOV_TASK)
        path = run.out / task["record"]
        record = json.loads(path.read_text())
        record["workflow"]["fingerprint"] = "sha256:" + "0" * 64
        path.write_text(json.dumps(record))
        assert main(["workflow", "verify", "--manifest", str(run.manifest), "--final"]) == 1
        result = verify(run.manifest, final=True)
        E.evidence(run.work, "x11-edited-fingerprint.json", result)
        assert not result["passed"] and FOV_TASK in result["incomplete"]
        assert "fingerprint" in " ".join(result["tasks"][FOV_TASK]["problems"])
        return
    original = copy.deepcopy(run.config)
    failures = []
    for iteration, changed in enumerate([True, False, True, False] if case == "restored-twice" else [True]):
        config = copy.deepcopy(original)
        if case.startswith("raw-"):
            raw = F.input_root(run.work) / "round2/FOV_002/ch01.ome.tif"
            declaring = {t["id"] for t in run.data["tasks"] if "input:round2/FOV_002/ch01.ome.tif" in t["inputs"]}
            assert declaring == {"summarize/sample/percentile_normalization/FOV_002", "fov/sample/FOV_002"}
            affected = E.downstream(run.data, declaring)
            stat = raw.stat()
            if case == "raw-size":
                raw.write_bytes(raw.read_bytes() + b"\n")
                assert raw.stat().st_size != stat.st_size
            else:
                import tifffile
                with tifffile.TiffFile(raw) as tif:
                    assert int(tif.pages[0].compression) == 1
                    offset = tif.pages[0].dataoffsets[0]
                payload = bytearray(raw.read_bytes())
                payload[offset] ^= 1
                raw.write_bytes(payload)
                os.utime(raw, ns=(stat.st_atime_ns, stat.st_mtime_ns - 86_400 * 10**9))
                assert raw.stat().st_size == stat.st_size and raw.stat().st_mtime_ns < stat.st_mtime_ns
        else:
            block = "export" if case == "export" else "processing"
            affected = E.downstream(run.data, {t["id"] for t in run.data["tasks"] if block in t["blocks"]})
            if block == "export":
                assert affected == {"export/sample/per_fov", "report/sample/per_fov", "compare"}
                config["export"]["previews"] = False
            elif changed:
                config["processing"]["spot_finding"]["threshold_value"] = 6.0
        run.replan(config)
        stale = verify(run.manifest)
        E.evidence(run.work, f"x11-{iteration}-stale.json", stale)
        assert set(stale["incomplete"]) == affected
        assert all("fingerprint" in " ".join(stale["tasks"][task]["problems"]) for task in affected)
        run.invoke(affected)
        failures.extend(E.fresh_comparison(run, E.snapshot(run), name=f"x11-{iteration}-fresh.json", rerun_tasks=affected))
    assert not failures, sorted(set(failures))


# --- K2: from test_workflow_nextflow_cache.py (its module markers: workflow, contract, nextflow, slow) --------------

@pytest.mark.workflow
@pytest.mark.contract
@pytest.mark.nextflow
@pytest.mark.slow
@pytest.mark.extended
@pytest.mark.parametrize("stub", [False], ids=["real"])
def test_fingerprints_and_fresh_tokens_rerun_only_affected_tasks(tmp_path, stub):
    """A,B,A,B,A for two independent blocks, two raw replacements, and a memory-only cache hit.

    The stub variant also runs in CI's minimal extras. The real variant exercises scientific commands;
    both VERIFY processes execute the package's completion rule on every invocation.
    """
    N.executable()
    if not stub:
        import importlib.util
        if any(importlib.util.find_spec(m) is None for m in ("anndata", "multiview_stitcher")):
            pytest.skip("the real per-fov cache checks need the anndata and multiview-stitcher extras")
    work = tmp_path / "cache"
    F.golden().write_fixture(work)
    config = N.small_resources(V.variants(work)["per-fov"])
    path = F.write_config(work, "cache", config)

    def invoke(expected, *, resume=True):
        done, rows = N.run(work, path, *(["-resume"] if resume else []), stub=stub)
        assert done.returncode == 0, done.stdout + done.stderr
        actual = N.tasks(rows)
        manifest = N.manifest(path)
        assert {r["tag"] for r in actual} == {t["id"] for t in manifest["tasks"]}
        assert {r["tag"] for r in actual if r["status"] == "COMPLETED"} == expected
        assert all(r["status"] in ("COMPLETED", "CACHED") for r in actual)
        checks = [r for r in rows if r["name"] in ("PLAN", "VERIFY_BEFORE", "VERIFY_AFTER")]
        assert len(checks) == 3 and all(r["status"] == "COMPLETED" for r in checks)
        records = {t["id"]: json.loads((Path(manifest["roots"]["output"]) / t["record"]).read_text())
                   for t in manifest["tasks"]}
        return manifest, records

    from starfinder.workflow import plan
    data = plan(path)
    data, records = invoke({t["id"] for t in data["tasks"]}, resume=False)
    unchanged, kept = invoke(set())
    assert kept == records
    for block in ("processing", "export"):
        affected = downstream(data, [t["id"] for t in data["tasks"] if block in t["blocks"]])
        for changed in (True, False, True, False):
            edit = copy.deepcopy(config)
            if changed and block == "processing":
                edit["processing"]["spot_finding"]["threshold_value"] = 6.0
            if changed and block == "export":
                edit["export"]["previews"] = False
            F.write_config(work, "cache", edit)
            _, new_records = invoke(affected)
            for task_id, previous in records.items():
                if task_id in affected:
                    assert new_records[task_id]["workflow"]["token"] != previous["workflow"]["token"]
                else:
                    assert new_records[task_id] == previous
            records = new_records
    # Per-kind requests live outside cache inputs and the rendered scientific command.
    edit = copy.deepcopy(config)
    edit["execution"]["tasks"]["fov"]["memory_mb"] = 768
    F.write_config(work, "cache", edit)
    _, kept = invoke(set())
    assert kept == records
    F.write_config(work, "cache", config)
    for replacement in ("older-time", "another-size"):
        raw = F.input_root(work) / "round2/FOV_002/ch01.ome.tif"
        declared = "input:" + raw.relative_to(F.input_root(work)).as_posix()
        affected = downstream(data, [t["id"] for t in data["tasks"] if declared in t["inputs"]])
        assert affected and len(affected) < len(data["tasks"])
        if replacement == "older-time":
            stat = raw.stat()
            import tifffile
            with tifffile.TiffFile(raw) as tif:
                assert int(tif.pages[0].compression) == 1
                offset = tif.pages[0].dataoffsets[0]
            payload = bytearray(raw.read_bytes())
            payload[offset] ^= 1
            raw.write_bytes(payload)
            os.utime(raw, ns=(stat.st_atime_ns, stat.st_mtime_ns - 86_400 * 10**9))
        else:
            raw.write_bytes(raw.read_bytes() + b"\n")
        _, new_records = invoke(affected)
        for task_id in records:
            if task_id in affected:
                assert new_records[task_id]["workflow"]["token"] != records[task_id]["workflow"]["token"]
            else:
                assert new_records[task_id] == records[task_id]
        records = new_records


# --- K3: from test_workflow_engines_e2e.py (its module markers: workflow, integration, extended, slow) ---------------

@pytest.mark.workflow
@pytest.mark.integration
@pytest.mark.extended
@pytest.mark.slow
@pytest.mark.nextflow
def test_uncalibrated_tiny_sample_nextflow(tmp_path):
    run = _uncalibrated_run(tmp_path, "nextflow")
    assert len(run.data["tasks"]) == 29
    run.invoke({t["id"] for t in run.data["tasks"]})
    _assert_declared_grids(run)


def main(argv=None):
    spec = importlib.util.spec_from_file_location("evidence_runner", Path(__file__).resolve().parent / "evidence_runner.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    return runner.run(__file__, MOVED, ORIGIN_REVISION, argv,
                      description="Run the workflow engine checks that W-898 moved out of the test suite (X11's "
                                  "restored-twice, and raw-size and raw-older on Nextflow; the Nextflow cache test on "
                                  "real tasks; the uncalibrated Nextflow run). Needs Snakemake 9, Nextflow 26.10.4 or "
                                  "newer on Java 17 or newer, and the anndata and multiview-stitcher extras.",
                      epilog="Run from src/python: uv run python ../../benchmarks/workflow_engine_evidence.py "
                             "--output <run dir>/workflow-engines")


if __name__ == "__main__":
    sys.exit(main())
