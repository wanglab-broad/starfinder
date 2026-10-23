"""The §2.10 engineering validation: EE6, the full chain, and the table of implemented checks (W-373).

Rows of docs/stitching-assembly-validation.md ("Checks"). Every other row is in the module of
the issue that delivered its code; the section "Implemented checks (W-373)" of that page maps
every row to its tests, and ``test_every_row_names_collected_tests`` keeps that map true:
the table has exactly the rows of the page's check tables (each case line of LB7 a row of its
own) and the RX rows; every row but an RX row names tests that pytest collects; none is marked
as an expected failure, none is skipped where the extras are importable, and the tier column
agrees with the ``slow`` marks. This module adds:

* EE6 on ``scene_small`` (``sample-small-v1``, seed 42): the placement of MV3
  (``multiview_stitcher`` on the reference-round images, maximum over channels,
  ``binning_zyx=(1, 1, 1)``, the link filter off), approved with ``approve_placement`` when its
  review is ``required``; the path that needs no segmenter (``import_labels(…, relabel=False)``,
  ``import_legacy_molecules_csv``, ``assign_molecules`` without expansion) and ``assemble`` on it;
  MV3's bound, EE2 (the oracle of ``test_stitching_assembly_scene.py`` with the stitched
  origins in place of the true ones) and EE3 on that assembly; and the observations whose
  status differs from the assembly on the truth placement in gauge G1 (``true_correction_*``
  through ``stitch`` with the test-only method of PL4, ``nominal_z`` 0), a measurement printed
  with ``-s``. The EE6 tests are ``slow`` and skip, naming the extra, where
  ``multiview_stitcher`` is not importable; dask runs with the synchronous scheduler.
"""
import ast
import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from starfinder.assembly import AssemblyInput, assemble
from starfinder.assignment import assign_molecules, import_legacy_molecules_csv
from starfinder.segmentation import ReferenceGrid, import_labels
from starfinder.stitching import (STITCHING_METHODS, MultiviewStitcherConfig, StitchingInput, approve_placement,
                                  positions_from_table, stitch)

from .assembly_fixtures import check_accounting
from .conftest import SUBSYSTEM_MARKERS
from .stitching_fixtures import STUB_SPEC, StubConfig
from .test_stitching_assembly_scene import molecule_rows, none, oracle

pytestmark = [pytest.mark.stitching, pytest.mark.validation]

TEST_ROOT = Path(__file__).resolve().parent
DESIGN = TEST_ROOT.parents[2] / "docs" / "stitching-assembly-validation.md"
SECTION = "## Implemented checks (W-373)"
ROW = r"(?:PL|MV|AS|SC|EE|EX|FI|LB|SD|WF|RX)\d+(?:\.\d+)?"
BATCH_EXTRAS = ("multiview_stitcher", "spatialdata", "anndata")
EXTRA = "the multiview-stitcher extra (starfinder[multiview-stitcher]) is not installed"
needs_library = pytest.mark.skipif(importlib.util.find_spec("multiview_stitcher") is None, reason=EXTRA)
ORIGIN = ["origin_z", "origin_y", "origin_x"]


# --- The table of the section "Implemented checks" ------------------------------------------------

def design_rows():
    """The row ids of the page's check tables and of the real examples, before the section."""
    text = DESIGN.read_text(encoding="utf-8").split(SECTION, 1)[0]
    return [m.group(1) for m in re.finditer(rf"^\| ({ROW}) \|", text, flags=re.MULTILINE)]


def implemented_checks():
    """{row: (the backticked names of its "Module and tests" cell, the cell, its tier)}."""
    section = DESIGN.read_text(encoding="utf-8").split(SECTION, 1)[1].split("\n## ", 1)[0]
    rows = {}
    for line in section.splitlines():
        match = re.match(rf"\| ({ROW}) \| ([^|]+) \| ([^|]+) \|$", line)
        if match:
            assert match.group(1) not in rows, match.group(1)
            rows[match.group(1)] = (re.findall(r"`([^`]+)`", match.group(2)), match.group(2).strip(),
                                    match.group(3).strip())
    return rows


def named_tests(names):
    """(module, test name) of a row, each name under the module named before it."""
    module, named = None, []
    for name in names:
        if name.endswith(".py"):
            module = name
        elif name.startswith("test_"):
            assert module is not None, name
            named.append((module, name))
    return named


def collected(session, module):
    """The items pytest collects from one test module (its classes included)."""
    items, nodes = [], [pytest.Module.from_parent(session, path=TEST_ROOT / module)]
    while nodes:
        node = nodes.pop()
        for child in node.collect():
            (items if isinstance(child, pytest.Item) else nodes).append(child)
    return items


def matches(item, name):
    if name.endswith("*"):
        return item.originalname.startswith(name[:-1])
    if "[" in name:
        return item.name == name
    return item.originalname == name


def runtime_skips(module):
    """The modules each ``importorskip`` call of a test module names, and whether it calls ``pytest.skip``."""
    imported, skips = set(), False
    for node in ast.walk(ast.parse((TEST_ROOT / module).read_text(encoding="utf-8"))):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "importorskip" and node.args and isinstance(node.args[0], ast.Constant):
                imported.add(node.args[0].value)
            skips |= node.func.attr == "skip"
    return imported, skips


def skip_reason(item):
    """The reason the marks of a collected item skip it here, or None."""
    for mark in item.iter_markers("skip"):
        return mark.kwargs.get("reason", "unconditional skip")
    for mark in item.iter_markers("skipif"):
        conditions = mark.args or (mark.kwargs["condition"],)
        assert not any(isinstance(c, str) for c in conditions), item.nodeid
        if any(conditions):
            return mark.kwargs.get("reason", "skipif")
    return None


def test_every_row_names_collected_tests(request):
    rows = implemented_checks()
    expected = design_rows()
    assert len(expected) == len(set(expected))
    assert {f"LB7.{k}" for k in range(1, 12)} <= set(expected)
    assert list(rows) == expected
    batch = all(importlib.util.find_spec(name) is not None for name in BATCH_EXTRAS)
    modules = sorted({module for names, _, _ in rows.values() for module, _ in named_tests(names)})
    items_of = {module: collected(request.session, module) for module in modules}
    for module in modules:
        imported, skips = runtime_skips(module)
        assert not skips, module
        missing = sorted(m for m in imported if importlib.util.find_spec(m) is None)
        assert not (batch and missing), (module, missing)
    for row, (names, cell, tier) in rows.items():
        named = named_tests(names)
        if row.startswith("RX"):
            assert named == [] and "real-example issue" in cell and tier == "real examples", row
            continue
        assert named, row
        items = []
        for module, name in named:
            found = [item for item in items_of[module] if matches(item, name)]
            assert found, (row, module, name)
            items += found
        for item in items:
            marks = {mark.name for mark in item.iter_markers()}
            assert "xfail" not in marks, item.nodeid
            assert SUBSYSTEM_MARKERS & marks, item.nodeid
            reason = skip_reason(item)
            if batch:
                assert reason is None, (row, item.nodeid, reason)
            elif reason is not None:
                assert "extra" in reason, (row, item.nodeid, reason)
        slow = [item.get_closest_marker("slow") is not None for item in items]
        if tier == "default":
            assert not any(slow), row
        elif tier == "`slow`":
            assert all(slow), row


# --- EE6: the full chain on scene_small ----------------------------------------------------------

@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    """``scene_small`` saved, assigned per FOV, stitched by MV3's configuration and by the truth in G1, assembled."""
    import dask

    from starfinder.synthetic import generate_sample_scene, sample_scene_preset, save_sample_scene
    book, config = sample_scene_preset("sample-small-v1")
    assert config.fov.seed == 42
    scene = generate_sample_scene(book, config=config)
    root = tmp_path_factory.mktemp("scene_small")
    save_sample_scene(scene, root)
    dataset, sample = config.fov.dataset_version, config.fov.sample_id
    table = positions_from_table(root / "positions.csv")
    truth = pd.read_csv(root / "placement_truth.csv", float_precision="round_trip").set_index("fov_id")
    grids = {fov: ReferenceGrid(config.acquisition.fov_shape_zyx, scene.fov_metadata[fov], "declared")
             for fov in table.fov_ids}
    reference = config.fov.geometry.reference_round or book.round_labels[0]
    inputs = [StitchingInput(fov, scene.fovs[fov].rounds[reference].max(axis=-1), grids[fov], "reference_merged")
              for fov in table.fov_ids]

    # MV3's placement: binning 1, the link filter off (NaN-quality links still dropped).
    with dask.config.set(scheduler="synchronous"):
        stitched = stitch(inputs, table, sample="scene_small",
                          config=MultiviewStitcherConfig(binning_zyx=(1, 1, 1), quality_threshold=-1.0))
    review = dict(state=stitched.record["review"]["state"], reasons=list(stitched.record["review"]["reasons"]))
    if review["state"] == "required":
        stitched = approve_placement(stitched, reviewer="test", note="EE6: the placement of MV3")

    # The comparison placement: the truth in gauge G1 through stitch with the test-only method of PL4.
    corrections = truth.loc[table.fov_ids, ["true_correction_z", "true_correction_y", "true_correction_x"]]
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(STITCHING_METHODS, StubConfig, STUB_SPEC)
        g1 = stitch(inputs, table, sample="scene_small", config=StubConfig(tuple(map(tuple, corrections.to_numpy()))))
    g1_review = dict(state=g1.record["review"]["state"], reasons=list(g1.record["review"]["reasons"]))
    if g1_review["state"] == "required":
        g1 = approve_placement(g1, reviewer="test", note="EE6: the truth placement in G1")

    results = {}
    for fov in table.fov_ids:
        labels = {}
        for target, file in (("cell", "cells.tif"), ("nucleus", "nuclei.tif")):
            namespace = json.dumps([dataset, sample, fov, None, "truth"])
            labels[target] = import_labels(root / "fovs" / fov / "labels" / file, grid=grids[fov], target=target,
                                           relabel=False, label_namespace=namespace)
        mols = import_legacy_molecules_csv(root / "fovs" / fov / "molecules.csv",
                                       spot_namespace=json.dumps([dataset, sample, fov, None]), genes=book.genes)
        results[fov] = assign_molecules(mols, labels["cell"], grid=grids[fov], nuclei=labels["nucleus"])
    assembly_inputs = [AssemblyInput(f, grids[f], assignment=results[f]) for f in table.fov_ids]
    tables = {name: pd.read_csv(root / f"{name}.csv", float_precision="round_trip")
              for name in ("molecules", "cells", "molecule_fovs", "cell_fovs")}
    return dict(scene=scene, config=config, truth=truth, ids=table.fov_ids, tables=tables, results=results,
                stitched=stitched, review=review, result=assemble(stitched, assembly_inputs),
                g1=g1, g1_review=g1_review, g1_result=assemble(g1, assembly_inputs))


def stitched_case(chain):
    """The EE2 case of ``chain`` with the stitched origins in place of the true ones."""
    truth = chain["truth"].copy()
    origins = chain["stitched"].fovs.set_index("fov_id").loc[chain["ids"], ORIGIN]
    for axis in "zyx":
        truth[f"true_origin_{axis}"] = origins[f"origin_{axis}"]
    return dict(chain, truth=truth)


@pytest.mark.slow
@needs_library
def test_ee6_the_stitched_placement_passes_mv3s_bound(chain):
    stitched, truth, config = chain["stitched"], chain["truth"], chain["config"]
    fovs = stitched.fovs
    assert list(fovs.status) == ["image_supported"] * 9
    assert (fovs.nominal_z == 0).all()
    expected = truth.loc[list(fovs.fov_id), ["true_origin_g1_z", "true_origin_g1_y", "true_origin_g1_x"]].to_numpy()
    errors = (fovs[ORIGIN].to_numpy() - expected) / np.asarray(config.acquisition.spacing_zyx)
    approval = stitched.record["review"].get("approval")
    print("\nEE6 " + json.dumps({"review": chain["review"], "approved": approval is not None,
                                  "max_abs_error_voxels_zyx": [float(v) for v in np.abs(errors).max(axis=0)]}))
    assert np.abs(errors).max() <= 0.5
    assert chain["review"]["state"] != "required" or approval is not None


@pytest.mark.slow
@needs_library
def test_ee6_ee2_holds_on_the_stitched_placement(chain):
    case = stitched_case(chain)
    result = chain["result"]
    cells, molecules = oracle(case)
    got = {(r.fov_id, int(r.cell_id)): (r.assembly_status, none(r.assembly_reason), none(r.owner_fov_id),
                                        "fov_edge" in r.assembly_flags.split(";"),
                                        "z_edge" in r.assembly_flags.split(";")) for r in result.cells.itertuples()}
    assert got == cells
    rows = molecule_rows(result)
    assert {k: (r.assembly_status, none(r.assembly_reason), none(r.owner_fov_id)) for k, r in rows.items()} \
        == molecules
    cell_fovs = chain["tables"]["cell_fovs"]
    for r in cell_fovs[cell_fovs.n_voxels > 0].itertuples():
        assert (bool(r.fov_edge), bool(r.z_edge)) == cells[(r.fov_id, int(r.cell_id))][3:]
    assert len(cell_fovs[cell_fovs.n_voxels > 0]) == len(cells)
    # The stitched Z origins reach the sample tables.
    truth = case["truth"]
    for fov in chain["ids"]:
        row = truth.loc[fov]
        mols = result.molecules[result.molecules.fov_id == fov]
        assert np.abs(mols.sample_z.to_numpy() - (row.true_origin_z + mols.z.to_numpy() * row.spacing_z)).max(
            initial=0.0) <= 1e-9
        fov_cells = result.cells[result.cells.fov_id == fov]
        assert np.abs(fov_cells.sample_centroid_z.to_numpy() - (row.true_origin_z + fov_cells.centroid_z.to_numpy()
                                                                * row.spacing_z)).max(initial=0.0) <= 1e-9


@pytest.mark.slow
@needs_library
def test_ee6_ee3_holds_on_the_stitched_placement(chain):
    check_accounting(chain["result"], chain["results"], chain["stitched"])


@pytest.mark.slow
@needs_library
def test_ee6_the_observations_whose_status_differs_from_the_truth_placement(chain):
    g1, truth = chain["g1"], chain["truth"]
    expected = truth.loc[chain["ids"], ["true_origin_g1_z", "true_origin_g1_y", "true_origin_g1_x"]].to_numpy()
    assert list(g1.fovs.status) == ["image_supported"] * 9 and (g1.fovs.nominal_z == 0).all()
    assert np.abs(g1.fovs[ORIGIN].to_numpy() - expected).max() <= 1e-9
    shift = np.abs(chain["stitched"].fovs[ORIGIN].to_numpy() - g1.fovs[ORIGIN].to_numpy()).max(axis=0)

    def outcome(row):
        return [row.assembly_status, none(row.assembly_reason), none(row.owner_fov_id)]

    differences, owners = {}, {}
    for kind, key in (("cells", lambda r: (r.fov_id, int(r.cell_id))),
                      ("molecules", lambda r: (r.fov_id, r.spot_id))):
        stitched = {key(r): outcome(r) for r in getattr(chain["result"], kind).itertuples()}
        reference = {key(r): outcome(r) for r in getattr(chain["g1_result"], kind).itertuples()}
        assert stitched.keys() == reference.keys()
        owners[kind] = sum(stitched[k][2] != reference[k][2] for k in stitched)
        differences[kind] = [{"fov_id": k[0], "id": k[1], "stitched": stitched[k], "truth_g1": reference[k]}
                             for k in sorted(stitched) if stitched[k][0] != reference[k][0]]
    print("\nEE6 " + json.dumps({"truth_g1_review": chain["g1_review"],
                                  "max_abs_origin_difference_um_zyx": [float(v) for v in shift],
                                  "rows": {kind: len(getattr(chain["result"], kind)) for kind in differences},
                                  "status_differs": {kind: len(rows) for kind, rows in differences.items()},
                                  "owner_differs": owners,
                                  "differences": differences}))
