"""AS3 to AS10 of docs/stitching-assembly-validation.md: retention, flags, accounting, special populations,
placeholder, missing and failed FOVs, and the entry's checks; and the ``assemble`` clauses of PL18.

Fixtures (``assembly_fixtures.py``): ``assembly_golden`` (3D, 2D, expanded), ``own_cases`` with
its expected rows, ``empty_cases``, ``status_cases``. The retained sets of AS3 are W-326
``tables/option_effects.json`` (O1), written here as literals.
"""
import dataclasses
import json
import os

import numpy as np
import pandas as pd
import pytest

from starfinder.assembly import (ASSEMBLY_STATUSES, CELL_ASSEMBLY_FLAGS, CELL_ASSEMBLY_REASONS, MOLECULE_ASSEMBLY_FLAGS,
                                 MOLECULE_ASSEMBLY_REASONS, AssemblyConfig, AssemblyInput, AssemblyResult, assemble)
from starfinder.assembly._assemble import _counts_record
from starfinder.image import IncompatibleGeometryError
from starfinder.segmentation import ReferenceGrid
from starfinder.stitching import PlacementReviewRequiredError, approve_placement, placement_from_table

from .assembly_fixtures import (GOLDEN_GENES, GOLDEN_IDS, OWN_DUPLICATE_CELLS, OWN_DUPLICATE_MOLECULES,
                                OWN_EXPECTED_CELLS, OWN_EXPECTED_MOLECULES, OWN_IDS,
                                OWN_SHAPE, TILE_ZYX, approved, cell_namespace, check_accounting, golden_molecule_tables,
                                golden_pieces, golden_placement, golden_results, inputs_of, metadata, molecules,
                                own_grids, own_inputs, own_placement, own_results, own_table, placeholder_result,
                                scene_cell, spot_namespace, unit_grid)

pytestmark = [pytest.mark.stitching, pytest.mark.validation]

# "2d" with the duplicate rule off; under the default rule FOV3's copy of scene cell 9 (its cell 5, 42 voxels on the
# sample grid) shares 38 voxels (0.90 of the smaller) with FOV1's copy (its cell 6, 87 voxels) and is excluded (W-400).
SCENE_RETAINED = {"3d": {"Position001": [1, 2, 5, 9], "Position002": [4, 6], "Position003": [3, 7, 8]},
                  "2d": {"Position001": [1, 2, 5, 9], "Position002": [4, 6], "Position003": [3, 7, 8, 9]},
                  "2d_rule": {"Position001": [1, 2, 5, 9], "Position002": [4, 6], "Position003": [3, 7, 8]}}
# Cell 9's molecule at global (4, 8, 64): local x 9 of FOV3 (offset 55), outside FOV1.
CELL9_MOLECULE = ("Position003", (4.0, 8.0, 9.0))
CELL_29 = ["cell_id", "status", "exclusion_reason", "size_voxels", "expanded_size_voxels", "size_physical",
           "expanded_size_physical", "centroid_z", "centroid_y", "centroid_x", "expanded_centroid_z",
           "expanded_centroid_y", "expanded_centroid_x", "n_molecules", "n_nuclei", "correspondence",
           "correspondence_flags", "compartments"]
SAMPLE_CELL_COLUMNS = (["fov_id", "cell_namespace", "cell_id", "obs_name"] + CELL_29[1:]
                       + [f"sample_{p}_{a}" for p in ("centroid", "expanded_centroid") for a in "zyx"]
                       + ["ownership_centroid"] + [f"sample_ownership_centroid_{a}" for a in "zyx"]
                       + ["owner_fov_id", "assembly_status", "assembly_reason", "assembly_flags", "conflict_with",
                          "duplicate_of"])
MOLECULE_29 = ["spot_namespace", "spot_id", "z", "y", "x", "gene_id", "voxel_z", "voxel_y", "voxel_x",
               "assignment_status", "cell_id", "in_expansion", "original_cell_id", "nucleus_id", "compartment"]
SAMPLE_TAIL = ["sample_z", "sample_y", "sample_x", "owner_fov_id", "assembly_status", "assembly_reason",
               "assembly_flags"]


def none(value):
    return None if pd.isna(value) else value


def retained_scene_cells(result):
    kept = {}
    for row in result.cells[result.cells.assembly_status == "retained"].itertuples():
        kept.setdefault(row.fov_id, []).append(scene_cell(row.fov_id, row.cell_id))
    return kept


def golden(dimensions="3d", *, config=AssemblyConfig(), **kw):
    results = golden_results(dimensions, **kw)
    placement = golden_placement()
    return assemble(placement, inputs_of(results), config=config), results, placement


def test_the_vocabularies():
    assert ASSEMBLY_STATUSES == ("retained", "excluded")
    assert CELL_ASSEMBLY_REASONS == ("not_owner", "fov_not_placed", "duplicate_observation")
    assert MOLECULE_ASSEMBLY_REASONS == ("cell_not_owner", "not_owner", "outside_grid", "fov_not_placed",
                                         "cell_duplicate_observation")
    assert CELL_ASSEMBLY_FLAGS == ("fov_edge", "z_edge", "conflict", "unclaimed")
    assert MOLECULE_ASSEMBLY_FLAGS == ("outside_owner_region", "cell_unclaimed")


# --- AS3: the retained sets of the golden fixture --------------------------------------------------------

@pytest.mark.parametrize("dimensions", ["3d", "2d", "2d_rule"])
def test_as3_retained_scene_cells(dimensions):
    rule = dimensions == "2d_rule"
    off = AssemblyConfig(duplicate_threshold=None)
    result, _, _ = golden(dimensions[:2], config=AssemblyConfig() if dimensions != "2d" else off)
    assert retained_scene_cells(result) == SCENE_RETAINED[dimensions]
    cells = result.cells.set_index(["fov_id", "cell_id"])
    conflicts = cells[cells.assembly_flags.str.contains("conflict")]
    if dimensions == "3d":
        assert conflicts.empty
        assert result.cells.duplicate_of.isna().all()
        assert_frame_equal_but_config(result, golden("3d", config=off)[0])
    elif rule:   # FOV3's copy of cell 9 is excluded naming FOV1's, which keeps no conflict; its molecule follows it
        assert conflicts.empty
        assert (cells.loc[("Position003", 5), "assembly_reason"], cells.loc[("Position003", 5), "duplicate_of"]) == \
            ("duplicate_observation", "sample1_Position001_6")
        assert result.cells.duplicate_of.notna().sum() == 1
    else:   # scene cell 9 retained twice, both copies flagged against each other
        assert sorted(conflicts.index) == [("Position001", 6), ("Position003", 5)]
        assert cells.loc[("Position001", 6), "conflict_with"] == "sample1_Position003_5"
        assert cells.loc[("Position003", 5), "conflict_with"] == "sample1_Position001_6"
    fov, position = CELL9_MOLECULE
    molecules = result.molecules
    row = molecules[(molecules.fov_id == fov) & (molecules.z == position[0]) & (molecules.y == position[1])
                    & (molecules.x == position[2])].iloc[0]
    if dimensions == "3d":
        assert (row.assembly_status, row.assembly_reason) == ("excluded", "cell_not_owner")
    elif rule:
        assert (row.assembly_status, row.assembly_reason) == ("excluded", "cell_duplicate_observation")
    else:
        assert (row.assembly_status, none(row.assembly_reason)) == ("retained", None)


def assert_frame_equal_but_config(result, off):
    """A result equals the same assembly with the rule off, apart from the config's threshold and the counts' keys
    of the rule's reasons (all 0)."""
    for name in ("fovs", "cells", "molecules", "counts"):
        pd.testing.assert_frame_equal(getattr(result, name), getattr(off, name), check_exact=True)
    counts = json.loads(json.dumps(result.record["counts"]))
    assert counts["cells"]["excluded"].pop("duplicate_observation") == 0
    assert counts["molecules"]["excluded"].pop("cell_duplicate_observation") == 0
    assert counts == off.record["counts"]


# --- AS4: retention rules and reasons, row order and columns ---------------------------------------------

@pytest.mark.parametrize("rule", [True, False])
def test_as4_own_cases_rows_equal_the_fixture(rule):
    results = own_results()
    config = AssemblyConfig() if rule else AssemblyConfig(duplicate_threshold=None)
    result = assemble(own_placement(), own_inputs(), config=config)
    assert list(result.cells.columns) == SAMPLE_CELL_COLUMNS
    # cell_namespace: the cell key's namespace of a molecule's cell (W-425; docs/sample-segmentation-contract.md, CK1).
    assert list(result.molecules.columns) == ["fov_id"] + MOLECULE_29 + ["cell_namespace"] + SAMPLE_TAIL
    got = {(r.fov_id, int(r.cell_id)): (r.status, r.assembly_status, none(r.assembly_reason), none(r.owner_fov_id),
                                        r.assembly_flags, r.conflict_with) for r in result.cells.itertuples()}
    assert got == ({**OWN_EXPECTED_CELLS, **OWN_DUPLICATE_CELLS} if rule else OWN_EXPECTED_CELLS)
    got = {(r.fov_id, r.spot_id): (r.assignment_status, none(r.cell_id), r.assembly_status, none(r.assembly_reason),
                                   none(r.owner_fov_id), r.assembly_flags) for r in result.molecules.itertuples()}
    assert got == ({**OWN_EXPECTED_MOLECULES, **OWN_DUPLICATE_MOLECULES} if rule else OWN_EXPECTED_MOLECULES)
    named = {(r.fov_id, int(r.cell_id)): r.duplicate_of for r in result.cells.itertuples() if not pd.isna(r.duplicate_of)}
    assert named == ({("FOV1", 2): "s1_FOV2_1"} if rule else {})
    # Order: placement table order, then increasing cell_id / input order.
    assert list(result.cells.fov_id) == ["FOV1"] * 4 + ["FOV2"] * 4
    assert list(result.cells.cell_id) == [1, 2, 3, 4] * 2
    assert list(result.molecules.spot_id) == [f"m{i}" for i in range(7)] + [f"m{i}" for i in range(8)]
    assert list(result.cells.obs_name) == [f"s1_{f}_{c}" for f in OWN_IDS for c in (1, 2, 3, 4)]
    # Every §2.9 column is copied unchanged; no molecule changes cell_id or assignment_status.
    for fov in OWN_IDS:
        pd.testing.assert_frame_equal(result.cells[result.cells.fov_id == fov][CELL_29].reset_index(drop=True),
                                      results[fov].cells[CELL_29], check_exact=True)
        pd.testing.assert_frame_equal(result.molecules[result.molecules.fov_id == fov][MOLECULE_29].reset_index(
            drop=True), results[fov].molecules[MOLECULE_29], check_exact=True)
    assert set(result.cells.cell_namespace) == {cell_namespace(f) for f in OWN_IDS}
    # Rule 7: the retained excluded_no_nucleus cell keeps its status, its excluded_cell molecule follows it.
    n = result.cells[(result.cells.fov_id == "FOV1") & (result.cells.cell_id == 4)].iloc[0]
    assert (n.status, n.assembly_status) == ("excluded_no_nucleus", "retained")
    # The expanded FOV's ownership centroid is the expanded one; FOV1's the original one.
    assert list(result.cells.ownership_centroid) == ["original"] * 4 + ["expanded"] * 4


def test_as4_rows_are_sorted_by_cell_id_whatever_the_input_order():
    results = own_results()
    shuffled = dataclasses.replace(results["FOV2"], cells=results["FOV2"].cells.iloc[::-1].reset_index(drop=True))
    result = assemble(own_placement(), inputs_of({"FOV1": results["FOV1"], "FOV2": shuffled}, own_grids()))
    assert list(result.cells.cell_id) == [1, 2, 3, 4] * 2


# --- AS5: flags --------------------------------------------------------------------------------------------

def test_as5_golden_flags():
    result, _, _ = golden("3d")
    cells = result.cells.set_index(["fov_id", "cell_id"])
    assert "fov_edge" in cells.loc[("Position001", 6), "assembly_flags"].split(";")
    assert cells.loc[("Position003", 5), "assembly_reason"] == "not_owner"
    assert "unclaimed" not in cells.loc[("Position003", 5), "assembly_flags"]
    # fov_edge and z_edge on exactly the cells whose label touches a Y or X border, or the first or last plane.
    for fov, (labels, _, _) in golden_pieces().items():
        yx = set(np.unique(np.concatenate([labels[:, [0, -1], :].ravel(), labels[:, :, [0, -1]].ravel()])))
        z = set(np.unique(labels[[0, -1]]))
        for cell_id in range(1, int(labels.max()) + 1):
            flags = set(cells.loc[(fov, cell_id), "assembly_flags"].split(";"))
            assert ("fov_edge" in flags) == (cell_id in yx) and ("z_edge" in flags) == (cell_id in z)


@pytest.mark.parametrize("rule", [True, False])
def test_as5_conflicts_false_leaves_the_two_flags_uncomputed(rule):
    threshold = 0.5 if rule else None
    with_flags = assemble(own_placement(), own_inputs(), config=AssemblyConfig(duplicate_threshold=threshold))
    without = assemble(own_placement(), own_inputs(), config=AssemblyConfig(conflicts=False,
                                                                            duplicate_threshold=threshold))
    assert without.cells.conflict_with.isna().all()
    flags = {f for v in without.cells.assembly_flags for f in v.split(";") if f}
    assert flags == {"fov_edge", "z_edge"}
    assert not without.molecules.assembly_flags.str.contains("cell_unclaimed").any()
    assert without.record["flags"] == {"computed": ["fov_edge", "z_edge", "outside_owner_region"],
                                       "conflicts": False}
    assert without.record["counts"]["flags"]["conflict"] is None
    assert without.record["counts"]["flags"]["unclaimed"] is None
    # W-400: under the default rule A (FOV1 cell 2) is excluded as a second observation of B (share 1.0), so neither
    # keeps conflict and A's molecule m1 is no longer a retained molecule outside its owner's region.
    assert with_flags.record["counts"]["flags"] == {"fov_edge": 3, "z_edge": 1, "conflict": 0 if rule else 2,
                                                    "unclaimed": 1, "outside_owner_region": 1 if rule else 2,
                                                    "cell_unclaimed": 1}
    for name in ("cells", "molecules"):
        a, b = getattr(with_flags, name), getattr(without, name)
        assert a.assembly_status.tolist() == b.assembly_status.tolist()
        assert a.assembly_reason.isna().tolist() == b.assembly_reason.isna().tolist()


def test_as5_a_placeholder_cell_no_owning_fov_covers_is_unclaimed():
    result = assemble(golden_placement(statuses={"Position002": "placeholder"}),
                      inputs_of({**golden_results(), "Position002": placeholder_result()}))
    cells = result.cells[result.cells.fov_id == "Position002"].set_index("cell_id")
    assert (cells.loc[1, "owner_fov_id"], cells.loc[1, "assembly_flags"]) == ("Position001", "")
    assert pd.isna(cells.loc[2, "owner_fov_id"]) and cells.loc[2, "assembly_flags"] == "unclaimed"


# --- AS6: accounting identities ----------------------------------------------------------------------------

@pytest.mark.parametrize("case", ["3d", "2d", "expanded"])
def test_as6_identities_on_the_golden_fixture(case):
    results = golden_results("3d" if case == "expanded" else case,
                             expand=("Position001", "Position003") if case == "expanded" else ())
    placement = golden_placement()
    result = assemble(placement, inputs_of(results))
    check_accounting(result, results, placement)
    matrix, rows = result.matrix()
    assert matrix.shape == (len(rows), len(GOLDEN_GENES))
    whole = result.counts[result.counts.compartment == "whole"]
    assert int(matrix.sum()) == int(whole["count"].sum())
    assert rows.obs_name.is_unique and result.cells.obs_name.is_unique


def test_as6_identities_on_own_cases():
    check_accounting(assemble(own_placement(), own_inputs()), own_results(), own_placement())


def test_as6_post_init_refuses_a_broken_result():
    result = assemble(own_placement(), own_inputs())
    with pytest.raises(ValueError, match="identity 1"):
        dataclasses.replace(result, cells=result.cells.drop(index=0))
    with pytest.raises(ValueError, match="identity 1"):
        dataclasses.replace(result, molecules=result.molecules.drop(index=3))
    changed = result.cells.copy()
    changed.loc[0, "assembly_status"], changed.loc[0, "assembly_reason"] = "excluded", "not_owner"
    with pytest.raises(ValueError, match="identity 2"):
        dataclasses.replace(result, cells=changed)
    changed = result.molecules.copy()
    changed.loc[4, "assembly_status"], changed.loc[4, "assembly_reason"] = "excluded", "not_owner"
    with pytest.raises(ValueError, match="identity 2"):
        dataclasses.replace(result, molecules=changed)
    # A status change that keeps the totals still breaks identity 3.
    changed = result.molecules.copy()
    retained, excluded = 4, 5      # FOV1 m4 (retained unassigned) and m5 (excluded not_owner)
    changed.loc[retained, ["assembly_status", "assembly_reason"]] = ["excluded", "not_owner"]
    changed.loc[excluded, ["assembly_status", "assembly_reason"]] = ["retained", pd.NA]
    with pytest.raises(ValueError, match="identity 3"):
        dataclasses.replace(result, molecules=changed)
    with pytest.raises(ValueError, match="identity 4"):
        dataclasses.replace(result, counts=result.counts.drop(index=0))


# --- AS7 and AS8: molecules only and empty inputs ---------------------------------------------------------

def molecules_only():
    tables = golden_molecule_tables()
    off = [((1, 2, 70), "A"), ((-0.6, 2, 3), "B")]      # outside the grid in x and in z
    rows = [r for r in golden_pieces()["Position001"][2]] + off
    tables["Position001"] = molecules("Position001", rows, GOLDEN_GENES)
    return tables


def test_as7_molecules_only():
    tables = molecules_only()
    placement = golden_placement()
    result = assemble(placement, inputs_of(tables, kind="molecules"))
    assert result.record["mode"] == "molecules_only" and result.cells.empty and result.counts.empty
    check_accounting(result, tables, placement)
    first = result.molecules[result.molecules.fov_id == "Position001"]
    assert first.assignment_status.isna().all()
    assert list(first.assembly_reason.iloc[-2:]) == ["outside_grid", "outside_grid"]
    for fov in GOLDEN_IDS:
        rows = result.molecules[result.molecules.fov_id == fov]
        inside = rows.assembly_reason.astype(object).fillna("") != "outside_grid"
        assert ((rows.assembly_status == "retained") == (rows.owner_fov_id == fov))[inside].all()


def test_as7_molecules_only_anndata(tmp_path):
    anndata = pytest.importorskip("anndata")
    from starfinder.assembly import write_sample_h5ad
    result = assemble(golden_placement(), inputs_of(molecules_only(), kind="molecules"))
    adata = anndata.read_h5ad(write_sample_h5ad(result, tmp_path / "sample.h5ad"))
    assert adata.shape == (0, len(GOLDEN_GENES)) and adata.var_names.tolist() == list(GOLDEN_GENES)
    retained = result.molecules[result.molecules.assembly_status == "retained"].gene_id.value_counts()
    assert adata.var["n_molecules_retained"].tolist() == [int(retained.get(g, 0)) for g in GOLDEN_GENES]
    import json
    assert json.loads(adata.uns["assembly"])["mode"] == "molecules_only"


def empty_variants():
    pieces = golden_pieces()
    background = {fov: [m for m in local if labels[tuple(int(v) for v in m[0])] == 0]
                  for fov, (labels, _, local) in pieces.items()}
    zero = {fov: np.zeros_like(labels) for fov, (labels, _, _) in pieces.items()}
    return {
        "fov_without_molecules": dict(molecules_of={"Position002": []}),
        "fov_without_cells": dict(labels={"Position003": zero["Position003"]}),
        "molecules_miss_every_cell": dict(molecules_of={"Position001": background["Position001"]}),
        "no_counted_cell": dict(nuclei={fov: np.zeros(TILE_ZYX, np.uint32) for fov in GOLDEN_IDS}),
        "no_cell_row": dict(labels=zero),
        "no_molecule": dict(molecules_of={fov: [] for fov in GOLDEN_IDS}),
    }


@pytest.mark.parametrize("case", list(empty_variants()))
def test_as8_empty_inputs_go_through_the_same_rules(case, tmp_path):
    results = golden_results(**empty_variants()[case])
    placement = golden_placement()
    result = assemble(placement, inputs_of(results))
    check_accounting(result, results, placement)
    assert result.fovs.fov_id.tolist() == GOLDEN_IDS
    assert result.fovs.n_cells.tolist() == [len(results[f].cells) for f in GOLDEN_IDS]
    if case == "no_counted_cell":
        assert result.cells.status.eq("excluded_no_nucleus").all() and result.counts.empty
        anndata = pytest.importorskip("anndata")
        from starfinder.assembly import write_sample_h5ad
        adata = anndata.read_h5ad(write_sample_h5ad(result, tmp_path / "sample.h5ad"))
        assert adata.shape == (0, len(GOLDEN_GENES))
    if case == "no_cell_row":
        assert result.cells.empty and result.counts.empty


@pytest.mark.parametrize("status", ["failed", "placeholder"])
def test_as8_a_placement_without_an_owning_fov_raises(status):
    statuses = {fov: status for fov in GOLDEN_IDS}
    placement = approved(golden_placement(statuses=statuses, review="required"))
    inputs = inputs_of(golden_results())
    with pytest.raises(ValueError, match="no FOV of the placement.*owns a point"):
        assemble(placement, inputs)
    with pytest.raises(ValueError, match="no FOV of the placement.*owns a point"):
        assemble(placement, [])


def no_fov_assembly():
    """empty_cases' no_fov_assembly: an AssemblyResult built by hand on the all-failed placement."""
    result, _, _ = golden()
    cells, mols = result.cells.copy(), result.molecules.copy()
    for frame, coordinates in ((cells, [c for c in cells.columns if c.startswith("sample_")]),
                               (mols, ["sample_z", "sample_y", "sample_x"])):
        frame[coordinates] = np.nan
        frame["owner_fov_id"] = pd.array([None] * len(frame), dtype="string")
        frame["assembly_status"] = "excluded"
        frame["assembly_reason"] = pd.array(["fov_not_placed"] * len(frame), dtype="string")
        frame["assembly_flags"] = ""
    cells["conflict_with"] = ""
    fovs = result.fovs.assign(placement_status="failed", origin_z=np.nan, origin_y=np.nan, origin_x=np.nan,
                              owns_points=False, n_cells_retained=0, n_molecules_retained=0)
    for column in ("footprint_y0", "footprint_y1", "footprint_x0", "footprint_x1", "centre_y", "centre_x"):
        fovs[column] = np.nan
    computed = result.record["flags"]["computed"]
    record = dict(result.record, counts=_counts_record(cells, mols, fovs, set(computed)))
    return AssemblyResult(result.sample, fovs, cells, mols, result.counts.iloc[:0], result.genes, record)


def test_as8_the_hand_built_no_fov_assembly_is_accepted():
    result = no_fov_assembly()
    assert set(result.cells.assembly_reason) == set(result.molecules.assembly_reason) == {"fov_not_placed"}
    assert result.record["counts"]["cells"]["retained"] == 0


# --- AS9: placeholder, missing and failed FOVs -----------------------------------------------------------

def test_as9_owning_fovs_without_an_input_raise():
    results = golden_results()
    with pytest.raises(ValueError, match="FOVs without one: Position002"):
        assemble(golden_placement(), inputs_of({f: r for f, r in results.items() if f != "Position002"}))
    metadata_only = golden_placement(statuses={"Position002": "metadata_only"})
    with pytest.raises(ValueError, match="FOVs without one: Position002"):
        assemble(metadata_only, inputs_of({f: r for f, r in results.items() if f != "Position002"}))


def test_as9_missing_and_failed_fovs():
    results = golden_results()
    missing = approved(golden_placement(statuses={"Position003": "missing"}, review="required"))
    result = assemble(missing, inputs_of({f: r for f, r in results.items() if f != "Position003"}))
    assert pd.isna(result.fovs.set_index("fov_id").loc["Position003", "kind"])
    check_accounting(result, {f: r for f, r in results.items() if f != "Position003"}, missing)
    failed = approved(golden_placement(statuses={"Position003": "failed"}, review="required"))
    result = assemble(failed, inputs_of(results))
    check_accounting(result, results, failed)
    for frame, columns in ((result.cells, [c for c in result.cells.columns if c.startswith("sample_")]),
                           (result.molecules, ["sample_z", "sample_y", "sample_x"])):
        rows = frame[frame.fov_id == "Position003"]
        assert len(rows) and (rows.assembly_reason == "fov_not_placed").all()
        assert rows[columns].isna().all().all() and rows.owner_fov_id.isna().all()
    # FOV1 now owns FOV3's former region: scene cell 3 is retained by FOV1.
    assert 3 in retained_scene_cells(result)["Position001"]


def test_as9_placeholder_fovs():
    results = golden_results()
    placement = golden_placement(statuses={"Position002": "placeholder"})
    without = assemble(placement, inputs_of({f: r for f, r in results.items() if f != "Position002"}))
    fovs = without.fovs.set_index("fov_id")
    assert fovs.owns_points.tolist() == [True, False, True]
    # The placeholder's region costs FOV1 no observation: scene cell 4 (centroid y 45) is retained by FOV1.
    assert retained_scene_cells(without)["Position001"] == [1, 2, 4, 5, 9]
    assert not without.cells.assembly_flags.str.contains("unclaimed").any()
    given = {**results, "Position002": placeholder_result()}
    result = assemble(placement, inputs_of(given))
    check_accounting(result, given, placement)
    cells = result.cells[result.cells.fov_id == "Position002"]
    assert (cells.assembly_reason == "not_owner").all()
    assert [none(v) for v in cells.owner_fov_id] == ["Position001", None]
    mols = result.molecules[result.molecules.fov_id == "Position002"]
    assert mols.assembly_reason.tolist() == ["cell_not_owner", "cell_not_owner", "not_owner", "not_owner",
                                             "outside_grid"]
    assert [none(v) for v in mols.owner_fov_id][:4] == ["Position001", None, "Position001", None]
    assert mols.assembly_flags.tolist()[:2] == ["", "cell_unclaimed"]


def test_as9_review_required_is_refused_until_approved():
    placement = golden_placement(statuses={"Position003": "missing"}, review="required")
    inputs = inputs_of({f: r for f, r in golden_results().items() if f != "Position003"})
    with pytest.raises(PlacementReviewRequiredError):
        assemble(placement, inputs)
    assemble(approved(placement), inputs)


# --- AS10: the entry's checks, in order ----------------------------------------------------------------------

def test_as10_check_1_config():
    for change in (dict(ownership="voronoi"), dict(centroid="mass"), dict(tie="random"), dict(conflicts="yes"),
                   dict(tie="half_open")):
        with pytest.raises(ValueError):
            AssemblyConfig(**change)
    AssemblyConfig(ownership="midpoint_box", tie="half_open")
    broken = AssemblyConfig()
    object.__setattr__(broken, "ownership", "voronoi")
    review = golden_placement(review="required")      # check 1 comes before check 2
    with pytest.raises(ValueError, match="ownership"):
        assemble(review, inputs_of(golden_results()), config=broken)
    with pytest.raises(TypeError, match="AssemblyConfig"):
        assemble(review, inputs_of(golden_results()), config={"ownership": "nearest_center"})


def test_as10_checks_2_to_8_each_raise_in_order():
    results = golden_results()
    placement = golden_placement()
    grid = unit_grid(TILE_ZYX)

    def fov_inputs(**changes):
        items = {f: AssemblyInput(f, grid, assignment=r) for f, r in results.items()}
        items.update(changes)
        return [v for v in items.values() if v is not None]

    # 2. Review, before the FOV identity (the inputs also repeat a FOV).
    repeated = fov_inputs() + [AssemblyInput("Position001", grid, assignment=results["Position001"])]
    with pytest.raises(PlacementReviewRequiredError):
        assemble(golden_placement(review="required"), repeated)
    # 3. FOV identity: repeated, unknown, owning FOV without an input; before the kind (both given).
    both = AssemblyInput("Position001", grid, assignment=results["Position001"],
                         molecules=golden_molecule_tables()["Position001"])
    with pytest.raises(ValueError, match="at most one input; repeated: Position001"):
        assemble(placement, repeated + [both])
    with pytest.raises(ValueError, match="not rows of the placement.*Position009"):
        assemble(placement, fov_inputs(Position009=AssemblyInput("Position009", grid, assignment=results[
            "Position001"])))
    # 4. Kind: both or neither, and mixed kinds; before the observation identity.
    with pytest.raises(ValueError, match="exactly one of assignment and molecules; FOVs Position001"):
        assemble(placement, fov_inputs(Position001=both))
    with pytest.raises(ValueError, match="exactly one of assignment and molecules; FOVs Position001"):
        assemble(placement, fov_inputs(Position001=AssemblyInput("Position001", grid)))
    mixed = fov_inputs(Position002=AssemblyInput("Position002", grid,
                                                 molecules=golden_molecule_tables()["Position001"]))
    with pytest.raises(ValueError, match="same kind"):
        assemble(placement, mixed)
    # 5. Observation identity: another FOV's observations, a subtile, another sample; before the grid.
    other = fov_inputs(Position002=AssemblyInput("Position002", unit_grid((8, 48, 60)),
                                                 assignment=results["Position001"]))
    with pytest.raises(ValueError, match="observation identity: Position002"):
        assemble(placement, other)
    tables = golden_molecule_tables()
    subtile = tables["Position001"]
    namespace = '["data","sample","Position001","t0"]'
    subtile = dataclasses.replace(subtile, table=subtile.table.assign(
        spot_namespace=pd.array([namespace] * len(subtile.table), dtype="string")),
        source={"spot_namespace": namespace})
    with pytest.raises(ValueError, match="subtile"):
        assemble(placement, inputs_of({**tables, "Position001": subtile}, kind="molecules"))
    elsewhere = '["data","other","Position001",null]'
    moved = dataclasses.replace(tables["Position001"], table=tables["Position001"].table.assign(
        spot_namespace=pd.array([elsewhere] * len(subtile.table), dtype="string")),
        source={"spot_namespace": elsewhere})
    with pytest.raises(ValueError, match="one dataset_id and sample_id"):
        assemble(placement, inputs_of({**tables, "Position001": moved}, kind="molecules"))
    # 6. Grid: not the placement's, not the assignment's, unknown spacing; before the genes.
    with pytest.raises(IncompatibleGeometryError, match="Position002.*not the grid the placement recorded"):
        assemble(placement, fov_inputs(Position002=AssemblyInput("Position002", unit_grid((8, 48, 60)),
                                                                 assignment=results["Position002"])))
    shifted = ReferenceGrid(TILE_ZYX, metadata(frame="other"), "declared")
    recorded = golden_placement(grids={f: shifted for f in GOLDEN_IDS})
    with pytest.raises(IncompatibleGeometryError, match="not the grid of its assignment record"):
        assemble(recorded, [AssemblyInput(f, shifted, assignment=r) for f, r in results.items()])
    unknown = ReferenceGrid(TILE_ZYX, dataclasses.replace(metadata(), spacing_zyx=None), "declared")
    uncalibrated = golden_placement(grids={f: unknown for f in GOLDEN_IDS})
    with pytest.raises(IncompatibleGeometryError, match="unknown spacing: Position001, Position002, Position003"):
        assemble(uncalibrated, [AssemblyInput(f, unknown, molecules=t) for f, t in tables.items()])
    # 7. Genes.
    fewer = {f: molecules(f, [m for m in golden_pieces()[f][2] if m[1] != "E"], GOLDEN_GENES[:4])
             for f in GOLDEN_IDS}
    fewer["Position001"] = tables["Position001"]
    with pytest.raises(ValueError, match="same genes"):
        assemble(placement, inputs_of(fewer, kind="molecules"))
    # 8. One layer.
    layered = golden_placement()
    fovs = layered.fovs.assign(nominal_z=[0.0, 0.0, 2.0])
    layered = dataclasses.replace(layered, fovs=fovs)
    with pytest.raises(ValueError, match="assemble handles one layer of tiles; FOVs Position001, Position002 have "
                                         "nominal z 0.0; Position003 have nominal z 2.0"):
        assemble(layered, inputs_of(results))
    # An approved placement passes.
    assert assemble(approved(golden_placement(review="required")), fov_inputs()).sample == "sample1"


def test_as10_assemble_writes_no_file(tmp_path, monkeypatch):
    results = golden_results()
    folder = tmp_path / "read_only"
    folder.mkdir()
    monkeypatch.chdir(folder)
    os.chmod(folder, 0o555)
    try:
        assemble(golden_placement(), inputs_of(results))
        assemble(own_placement(), own_inputs())
    finally:
        os.chmod(folder, 0o755)
    assert list(folder.iterdir()) == []


def test_assembly_input_types():
    grid = unit_grid(TILE_ZYX)
    for bad in (dict(fov_id=""), dict(grid="grid"), dict(assignment=3), dict(molecules="table")):
        with pytest.raises(TypeError):
            AssemblyInput(**{"fov_id": "Position001", "grid": grid, **bad})


# --- PL18: the assemble clauses ------------------------------------------------------------------------------

OWN_ORIGINS = {"FOV1": (0.3, 0.25, 1.5), "FOV2": (-0.7, -0.125, 15.0)}


@pytest.mark.parametrize("origins", [None, OWN_ORIGINS])
def test_pl18_a_supplied_placement_is_refused_until_approved(origins):
    plain = own_placement(origins=origins, approve=False)
    with pytest.raises(PlacementReviewRequiredError, match="supplied_placement"):
        assemble(plain, own_inputs())
    accepted = approve_placement(plain, reviewer="test", note="pl18")
    result = assemble(accepted, own_inputs())
    origin = ["origin_z", "origin_y", "origin_x"]
    correction = ["correction_z", "correction_y", "correction_x"]
    assert accepted.fovs[origin].to_numpy().tobytes() == plain.fovs[origin].to_numpy().tobytes()
    assert accepted.fovs[correction].to_numpy().tobytes() == plain.fovs[correction].to_numpy().tobytes()
    assert result.fovs[origin].to_numpy().tobytes() == plain.fovs[origin].to_numpy().tobytes()
    if origins is not None:
        assert result.fovs.origin_z.tolist() == [0.3, -0.7]
        fov2 = result.molecules[result.molecules.fov_id == "FOV2"]
        assert (fov2.sample_z.to_numpy() == fov2.z.to_numpy() - 0.7).all()


@pytest.mark.parametrize("origins", [None, {"FOV1": (0.5, 0.0, 0.0), "FOV2": (0.5, 0.0, 12.0)}])
def test_pl18_a_table_whose_nominal_z_differ_raises_from_check_8(origins):
    placement = placement_from_table(own_table(nominal_z=(0.0, 1.0)), own_grids(), sample="s1", origins=origins)
    with pytest.raises(PlacementReviewRequiredError):
        assemble(placement, own_inputs())
    with pytest.raises(ValueError, match="assemble handles one layer of tiles"):
        assemble(approve_placement(placement, reviewer="test", note="pl18"), own_inputs())


def test_the_own_cases_grid_is_unit_spaced():
    assert own_grids()["FOV1"].shape_zyx == OWN_SHAPE and spot_namespace("FOV1").startswith('["data"')
