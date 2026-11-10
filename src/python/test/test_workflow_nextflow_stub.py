"""N1 to N3: actual Nextflow 26.10.4 stub execution, graph refusals and exit-status retries."""
from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path

import pytest

from . import workflow_examples as X
from . import workflow_nextflow as N

pytestmark = [pytest.mark.workflow, pytest.mark.contract, pytest.mark.nextflow, pytest.mark.slow]


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    N.executable()
    work = tmp_path_factory.mktemp("nextflow-stub")
    X.golden().write_fixture(work)
    X.write_supplied(work)
    return work


def configuration(work, variant):
    if variant == "empty":
        return N.small_resources(X.dataset(work, "nextflow-empty"))
    config = X.variants(work)["per-fov" if variant == "two-supplied-steps" else variant]
    config["output_id"] = "nextflow-" + variant
    if variant == "two-supplied-steps":
        config["processing"]["preprocessing"]["steps"].insert(
            2, {"method": "histogram_matching", "fit": "supplied", "reference_channel": 0})
    if variant == "after-fusion":
        config["stitching"] = copy.deepcopy(X.variants(work)["supplied-xml"]["stitching"])
    return N.small_resources(config)


@pytest.mark.parametrize("variant", [*X.VARIANTS, "two-supplied-steps", "empty"])
@pytest.mark.parametrize("profile", ["local", "google-batch,local"])
def test_n1_every_task_runs_once_after_its_needs(tiny, variant, profile):
    settings = configuration(tiny, variant)
    name = variant
    arguments = []
    environment = None
    if profile == "google-batch,local":
        name += "-google-batch"
        settings["output_id"] += "-google-batch"
        arguments = ["-work-dir", str(tiny / name / "work")]
        environment = {"NXF_OFFLINE": "true", "NXF_PLUGINS_DEFAULT": "false"}
    config = X.write_config(tiny, name, settings)
    done, rows = N.run(tiny / name, config, *arguments, environment=environment, profile=profile)
    assert done.returncode == 0, done.stdout + done.stderr
    manifest = N.manifest(config)
    checks = [r for r in rows if r["name"] in ("PLAN", "VERIFY_BEFORE", "VERIFY_AFTER")]
    assert len(checks) == 3 and all(r["status"] == "COMPLETED" for r in checks)
    rows = N.tasks(rows)
    assert Counter(r["tag"] for r in rows) == Counter(t["id"] for t in manifest["tasks"])
    by_id = {r["tag"]: r for r in rows}
    for task in manifest["tasks"]:
        row = by_id[task["id"]]
        assert row["status"] == "COMPLETED"
        for need in task["needs"]:
            assert int(row["start"]) >= int(by_id[need]["complete"]), (task["id"], need)
        record = json.loads((Path(manifest["roots"]["output"]) / task["record"]).read_text())
        assert record["stub"] and record["workflow"]["token"].startswith("rerun:")
    if variant == "per-fov":
        assert sum("merge/sample/percentile_normalization" in t["needs"] and t["kind"] == "fov"
                   for t in manifest["tasks"]) == 4
    if variant == "after-fusion":
        globals_ = [t for t in manifest["tasks"] if t["kind"] == "tile_global"]
        assert len(globals_) == 4
        for producer in globals_:
            assert sum(producer["id"] in t["needs"] and t["kind"] == "tile" for t in manifest["tasks"]) == 4
    if variant == "two-supplied-steps":
        assert {r["name"].split(" (")[0] for r in rows if r["tag"].startswith(("summarize/", "merge/"))} == {
            "SUMMARIZE_0", "MERGE_0", "SUMMARIZE_1", "MERGE_1"}


@pytest.mark.parametrize("status", [1, 2, 3])
def test_n3_only_failures_are_retried(tiny, tmp_path, status):
    config = configuration(tiny, "reference-only")
    config["output_id"] = f"retry-{status}"
    config["selected_fovs"] = ["FOV_001"]
    config = X.write_config(tmp_path, "retry", config)
    stub = tmp_path / "fault.sh"
    stub.write_text('#!/usr/bin/env bash\nset -eu\n'
                    f'if [[ "$2" == "fov/sample/FOV_001" && "$4" == 1 ]]; then exit {status}; fi\n'
                    f'exec "{X.ROOT / "nextflow/bin/stub_task.sh"}" "$@"\n')
    stub.chmod(0o755)
    done, rows = N.run(tmp_path, config, "--stub_runner", str(stub))
    attempts = [r for r in N.tasks(rows) if r["tag"] == "fov/sample/FOV_001"]
    assert len(attempts) == (2 if status == 1 else 1), done.stdout + done.stderr
    assert [int(r["exit"]) for r in sorted(attempts, key=lambda r: int(r["attempt"]))] == ([1, 0] if status == 1 else [status])
    assert (done.returncode == 0) == (status == 1), done.stdout + done.stderr


@pytest.mark.parametrize("max_retries, attempts", [(0, 1), (1, 2)])
def test_n3_max_retries_bounds_the_attempts(tiny, tmp_path, max_retries, attempts):
    """`--max_retries 0` runs a failing task once (Nextflow reads 0 as unset); 1 allows one retry, never more."""
    config = configuration(tiny, "reference-only")
    config["output_id"] = f"max-retries-{max_retries}"
    config["selected_fovs"] = ["FOV_001"]
    config = X.write_config(tmp_path, "max-retries", config)
    stub = tmp_path / "fault.sh"
    stub.write_text('#!/usr/bin/env bash\nset -eu\n'
                    'if [[ "$2" == "fov/sample/FOV_001" ]]; then exit 1; fi\n'
                    f'exec "{X.ROOT / "nextflow/bin/stub_task.sh"}" "$@"\n')
    stub.chmod(0o755)
    done, rows = N.run(tmp_path, config, "--stub_runner", str(stub), "--max_retries", str(max_retries))
    tries = [r for r in N.tasks(rows) if r["tag"] == "fov/sample/FOV_001"]
    assert sorted(int(r["attempt"]) for r in tries) == list(range(1, attempts + 1)), done.stdout + done.stderr
    assert {int(r["exit"]) for r in tries} == {1}, done.stdout + done.stderr
    assert done.returncode != 0, done.stdout + done.stderr


@pytest.mark.parametrize("case, message", [
    ("fiji", "D3: Nextflow runs no Fiji"),
    ("seed-chain", "seed chain deeper than three at segment/sample/FOV_001/seed_3"),
    ("four-supplied-steps", "more than three supplied steps at summarize/sample/fourth_step/FOV_001"),
    ("unknown-kind", "no implementation for kind future_task"),
])
def test_n2_unsupported_graphs_are_refused_before_any_task(tiny, tmp_path, case, message):
    """Four supplied methods cannot currently be configured (only three support fit=supplied).

    For bounds beyond today's configuration shape, extend an actual plan at the CLI boundary with a stand-in
    plan command; the engine must refuse these graph shapes rather than silently omitting tasks.
    """
    import os
    import sys
    from starfinder.workflow import plan

    variant = "after-fusion" if case == "fiji" else ("reference-only" if case == "seed-chain" else "per-fov")
    config = X.variants(tiny)[variant]
    config["output_id"] = "refuse-" + case
    if case == "seed-chain":
        runs = config["segmentation"]["runs"]
        runs[0]["name"] = "seed_0"
        template = copy.deepcopy(runs[1])
        config["segmentation"]["runs"] = [runs[0]]
        for i in range(1, 4):
            config["segmentation"]["runs"].append(dict(template, name=f"seed_{i}", seeds=f"seed_{i - 1}"))
    config = X.write_config(tmp_path, case, N.small_resources(config))
    environment = None
    if case in ("four-supplied-steps", "unknown-kind"):
        data = plan(config)
        if case == "four-supplied-steps":
            templates = {kind: next(t for t in data["tasks"] if t["kind"] == kind)
                         for kind in ("summarize", "merge")}
            data["tasks"] = []
            parent = None
            for step in ("percentile_normalization", "histogram_matching", "scalar_background", "fourth_step"):
                summary = copy.deepcopy(templates["summarize"])
                summary["id"] = f"summarize/sample/{step}/FOV_001"
                summary["needs"] = [parent] if parent else []
                merge = copy.deepcopy(templates["merge"])
                merge["id"] = f"merge/sample/{step}"
                merge["needs"] = [summary["id"]]
                data["tasks"].extend([summary, merge])
                parent = merge["id"]
        else:
            data["tasks"][0]["kind"] = "future_task"
        manifest = Path(data["roots"]["output"]) / "workflow" / "manifest.json"
        manifest.write_text(json.dumps(data))
        # Only PLAN is replaced, not the Nextflow graph, processes, or verify command.
        folder = tmp_path / "bin"
        folder.mkdir()
        wrapper = folder / "starfinder"
        wrapper.write_text(f'#!{sys.executable}\nimport json, os, sys\n'
                           'if sys.argv[1:3] == ["workflow", "plan"]:\n'
                           f'    print(json.dumps({{"manifest": {str(manifest)!r}}}))\n'
                           'else:\n'
                           f'    os.execv({str(Path(sys.executable).with_name("starfinder"))!r}, '
                           '["starfinder", *sys.argv[1:]])\n')
        wrapper.chmod(0o755)
        environment = {"PATH": str(folder) + os.pathsep + os.environ["PATH"]}
    done, rows = N.run(tmp_path, config, environment=environment)
    assert done.returncode != 0
    assert message in done.stdout + done.stderr, done.stdout + done.stderr
    assert N.tasks(rows) == []
