"""W-650 cache regressions on the real engine; full X11 fresh-output comparisons belong to W-652."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest

from . import workflow_examples as X
from . import workflow_nextflow as N

pytestmark = [pytest.mark.workflow, pytest.mark.contract, pytest.mark.nextflow, pytest.mark.slow]


def downstream(manifest, selected):
    selected = set(selected)
    for task in manifest["tasks"]:
        if selected.intersection(task["needs"]):
            selected.add(task["id"])
    return selected


@pytest.mark.extended
@pytest.mark.parametrize("stub", [True, False], ids=["stub", "real"])
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
    X.golden().write_fixture(work)
    config = N.small_resources(X.variants(work)["per-fov"])
    path = X.write_config(work, "cache", config)

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
            X.write_config(work, "cache", edit)
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
    X.write_config(work, "cache", edit)
    _, kept = invoke(set())
    assert kept == records
    X.write_config(work, "cache", config)
    for replacement in ("older-time", "another-size"):
        raw = X.input_root(work) / "round2/FOV_002/ch01.ome.tif"
        declared = "input:" + raw.relative_to(X.input_root(work)).as_posix()
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
