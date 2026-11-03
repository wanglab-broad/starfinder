"""The morphology and assign commands against W-642's golden values (design/baselines/orchestration.md).

W-790 updates only identities and downstream values resulting from collapsed regional maxima;
every kept spot's extracted and decoded values are unchanged, as independently audited in that issue.

W-642's invocations A and B (``scripts/workflow_golden.py``, ``configurations``) migrated to the Python shape on their
fixture (``tiny_sample``): A's processing block without its run flags, a morphology block with
``morphology.channel_images`` (the images the nuclei_registration rule wrote), one segment run per FOV that imports the
truth cell labels with ``{fov_id}`` in its path (OP3), which B copied as the legacy label file, and B's assignment (no
expansion), named ``default``. Each task runs by its command. The morphology task's ``nuclei_registration.json``
equals the golden record, and its channel images' shape, dtype, minimum, maximum and sum equal ``values.json``; the
assign task's tables equal B's: integers and strings exactly, float64 within ``rtol=1e-9``, ``atol=1e-12``, a float
column that holds integers exactly (``workflow_golden.compare``'s rule). Of the contract's exceptions
(design/specs/orchestration.md, "The exception of the golden comparison") none touches a compared value: OC3 and OC4
change the frame ids and the recorded paths, which the record and the tables do not hold, D6 the retired rules'
images, D8 the checkpoint stages, which the migrated configuration states as A did.
"""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from starfinder.workflow import plan

from . import expected_values as EV
from . import workflow_examples as X

pytestmark = [pytest.mark.workflow, pytest.mark.golden, pytest.mark.slow]

EXPECTED = Path(__file__).parent / "data" / "expected" / "test_workflow_orchestration_golden"
OUTPUT_ID = "golden"
STAINS = ("ch04", "ch05")


def configuration(work):
    """Invocations A and B of W-642 in the Python shape (see the module docstring)."""
    work = Path(work)
    return dict(X.dataset(work, OUTPUT_ID), processing=copy.deepcopy(X.PROCESSING),
                morphology={"channel_images": {"projection": None}},
                segmentation={"runs": [{"name": "cell", "target": "cell", "method": "import",
                                        "path": str(work / "labels" / "{fov_id}.tif")}]},
                assignment={"cells": "cell"})


@pytest.fixture(scope="module")
def golden_run(tmp_path_factory):
    """(workflow_golden, work folder, output root) after every task of the migrated configuration ran."""
    G = X.golden()
    work = (tmp_path_factory.mktemp("golden") / "work").resolve()
    G.write_fixture(work)
    plan(X.write_config(work, "golden", configuration(work)))
    manifest = X.output_root(work, OUTPUT_ID) / "workflow" / "manifest.json"
    kinds = [t["kind"] for t in json.loads(manifest.read_text())["tasks"]]
    assert {k: kinds.count(k) for k in set(kinds)} == {"fov": 4, "morphology": 4, "segment": 4, "assign": 4}
    for task in json.loads(manifest.read_text())["tasks"]:
        assert X.run_task(manifest, task["id"]) == 0, task["id"]
    return G, work, manifest.parents[1]


def test_the_morphology_command_reproduces_the_golden_records(golden_run):
    G, work, out = golden_run
    for fov in X.FOVS:
        actual = G._normalized(json.loads((out / "fovs" / fov / "nuclei_registration.json").read_text()), str(work))
        expected = json.loads((EXPECTED / "records" / "fovs" / fov / "nuclei_registration.json").read_text())
        assert G.record_differences(actual, expected) == [], fov


def test_the_morphology_channel_images_reproduce_the_golden_images(golden_run):
    G, _, out = golden_run
    values = json.loads((EXPECTED / "values.json").read_text())
    compared = 0
    for fov in X.FOVS:
        for label in STAINS:
            name = f"images/flamingo/{label}/{fov}.tif"
            pixels = G._image(out / name)
            actual = {"shape": list(pixels.shape), "dtype": str(pixels.dtype), "min": pixels.min().item(),
                      "max": pixels.max().item(), "sum": float(pixels.sum(dtype=np.float64))}
            for statistic, value in actual.items():
                entry = values[f"{name}:{statistic}"]
                assert G._close(value, entry["value"], entry["compare"]), (name, statistic, value, entry)
                compared += 1
    assert compared == len(X.FOVS) * len(STAINS) * 5


def test_the_assign_command_reproduces_the_golden_assignment_tables(golden_run, tmp_path):
    G, _, out = golden_run
    for fov in X.FOVS:
        for table in ("cells", "counts", "molecules"):
            name = f"{fov}-assignment-{table}.csv"
            dtypes = EV.write_csv(G._table(out / "fovs" / fov / "assignment" / "default" / f"{table}.csv"),
                                  tmp_path / name)
            actual, expected = EV.read_csv(tmp_path / name, dtypes), EV.read_table(EXPECTED / name)
            close = [c for c in expected if str(expected[c].dtype) == "float64" and not G._integral(expected[c])]
            assert EV.table_differences(actual, expected, close) == [], name
