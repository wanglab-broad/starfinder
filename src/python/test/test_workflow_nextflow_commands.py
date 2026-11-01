"""Small real CLI jobs through Nextflow's script paths, separate from the stub-only graph checks."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from . import workflow_examples as X
from . import workflow_nextflow as N

pytestmark = [pytest.mark.workflow, pytest.mark.contract, pytest.mark.nextflow, pytest.mark.slow]


def assert_completed(config, done, rows, expected_kinds):
    assert done.returncode == 0, done.stdout + done.stderr
    manifest = N.manifest(config)
    assert {t["kind"] for t in manifest["tasks"]} == expected_kinds
    actual = N.tasks(rows)
    assert {r["tag"] for r in actual} == {t["id"] for t in manifest["tasks"]}
    assert all(r["status"] == "COMPLETED" for r in actual)
    for task in manifest["tasks"]:
        record = json.loads((Path(manifest["roots"]["output"]) / task["record"]).read_text())
        assert not record.get("stub", False)
        assert record["status"] == "succeeded"
        assert record["workflow"]["execution_context"]["engine"] == "nextflow"
        assert record["workflow"]["token"].startswith("rerun:")
        assert record["workflow"]["fingerprint"] == task["fingerprint"]


def test_real_per_fov_commands(tmp_path):
    N.executable()
    if any(importlib.util.find_spec(m) is None for m in ("anndata", "multiview_stitcher")):
        pytest.skip("the real per-fov route needs the anndata and multiview-stitcher extras")
    work = tmp_path / "real-per-fov"
    X.golden().write_fixture(work)
    config = N.small_resources(X.variants(work)["per-fov"])
    path = X.write_config(work, "real", config)
    done, rows = N.run(work, path, stub=False)
    assert_completed(path, done, rows, {"summarize", "merge", "fov", "morphology", "segment", "assign", "stitch",
                                        "export", "report", "compare"})


def test_real_tiled_commands(tmp_path):
    N.executable()
    work = tmp_path / "real-tiled"
    X.golden().write_fixture(work)
    config = X.single_fov(work)
    config["processing"].update(mode="tiled", tiled={"tile_size_yx": 18})
    path = X.write_config(work, "tiled", N.small_resources(config))
    done, rows = N.run(work, path, stub=False)
    assert_completed(path, done, rows, {"tile_global", "tile", "tile_merge"})
    manifest = N.manifest(path)
    assert not (Path(manifest["roots"]["output"]) / "fovs/FOV_001/tiled/registered.h5").exists()
    # Temporary outputs removed by the merge must still allow every task's cache to be reused.
    done, rows = N.run(work, path, "-resume", stub=False)
    assert done.returncode == 0, done.stdout + done.stderr
    assert all(r["status"] == "CACHED" for r in N.tasks(rows))


def test_real_supplied_bigstitcher_import_command(tmp_path):
    """A supplied, hand-written placement enters the real import command without running Fiji."""
    N.executable()
    import numpy as np
    from . import workflow_tasks as T
    from .stitching_fixtures import LN_SPACING
    from .test_stitching_bigstitcher import LINK_IDS, LINK_SHAPE, links_view_map, links_xml
    from .test_workflow_stitch_tasks import GRID, SAMPLE, imported, reference, rows

    work = tmp_path / "real-import"
    work.mkdir()
    supplied = work / SAMPLE
    supplied.mkdir()
    xml = supplied / "dataset.xml"
    xml.write_text(links_xml())
    views = supplied / "view_map.csv"
    links_view_map().to_csv(views, index=False)
    config = imported(work, stitching={"method": "bigstitcher", "positions": GRID,
                                      "dataset_xml": str(work / "{sample}" / "dataset.xml"),
                                      "view_map": str(work / "{sample}" / "view_map.csv")})
    T.write_inputs(config, rows(SAMPLE, LINK_IDS))
    for index, fov in enumerate(LINK_IDS):
        reference(work, fov, np.full(LINK_SHAPE, index + 1, np.uint16), LN_SPACING)
    path = X.write_config(work, "supplied", N.small_resources(config))
    done, actual = N.run(work, path, stub=False)
    assert_completed(path, done, actual, {"bigstitcher_import"})
