"""The duplicate rule of the route per FOV (W-400): one of two observations of a cell stays retained.

docs/assembly-contract.md ("Retention", rule 8; "Statuses and reasons"; "Count accounting") and
docs/assembly-algorithms.md ("Duplicate rule"). Fixtures, built here in the style of
``assembly_fixtures.py`` (box cells painted on unit grids, assigned in memory, a placement of
``positions_from_grid``):

* ``share_cases``: two FOVs of 4×24×20 at x 0 and 10 (boundary x 14.5 under both ownership
  options), five pairs of a cell retained by FOV P1 and one retained by P2 whose territories
  share 0 %, 30 %, exactly 50 %, 80 % and 100 % of the smaller one, in 3D (boxes through planes 1
  and 2) and in 2D (``plane`` labels on the grid's Z maximum); the 50 % pair holds molecules.
* ``partner_cases``: one cell of P1 paired with two cells of P2, once as the largest and once
  as the smallest of the three.
* ``corner_case``: a 2 × 2 grid of 4×20×20 FOVs at steps of 10 px; one cell seen by all four
  FOVs, each FOV retaining its own observation.
* ``chain_cases``: three FOVs of 4×12×20 at x 0, 10 and 20; a cell of C1 paired with one of C2
  and that with one of C3, the C1 and C3 cells not paired, once with the middle cell largest and
  once with the first.

The shares the rule computed are captured from ``_duplicate_rule`` and compared with the
shares ``duplicate_oracle`` computes from the territories placed on one grid by the test's own
code. Nothing is read from a run directory.
"""
import importlib.util

import numpy as np
import pandas as pd
import pytest

import starfinder.assembly._assemble as assemble_module
from starfinder.assembly import AssemblyConfig, AssemblyInput, assemble, sample_anndata
from starfinder.assignment import assign_molecules
from starfinder.stitching import approve_placement, placement_from_table, positions_from_grid

from .assembly_fixtures import (check_accounting, duplicate_oracle, label_run, molecules, paint, placed_on_grid,
                                unit_grid)

pytestmark = [pytest.mark.stitching, pytest.mark.validation]

GENES = ("A", "B", "C")
OWNERSHIP = ("nearest_center", "midpoint_box")
needs_anndata = pytest.mark.skipif(importlib.util.find_spec("anndata") is None,
                                   reason="the anndata extra (starfinder[anndata]) is not installed")


def box(rows, columns, origin_yx, planes=(1, 2)):
    """A box cell in a FOV's local index coordinates from global (sample) rows and columns, inclusive."""
    oy, ox = int(origin_yx[0]), int(origin_yx[1])
    return planes, (rows[0] - oy, rows[1] - oy), (columns[0] - ox, columns[1] - ox)


def build(sample, ids, shape, columns, rows, cells, *, molecules_of=None, dimensions="3d"):
    """A sample of FOVs on a regular grid (overlap 0.5 in X, and in Y when rows > 1), with the box cells given.

    ``cells`` maps each FOV to ``{cell_id: (global rows, global columns)}``; ``molecules_of`` each FOV to its
    local ``((z, y, x), gene)`` rows. Returns the placement, the per-FOV results, the grids and the origins.
    """
    table = positions_from_grid(columns=columns, rows=rows, order="column_major_down_right",
                                tile_shape_yx=shape[1:], spacing_yx=(1.0, 1.0), overlap=0.5, fov_ids=ids)
    grids = {fov: unit_grid(shape) for fov in ids}
    origins = {fov: tuple(float(v) for v in nominal) for fov, nominal in zip(table.fov_ids,
                                                                              table.nominal_zyx().tolist())}
    placement = approve_placement(placement_from_table(table, grids, sample=sample), reviewer="test", note=sample)
    results = {}
    for fov in ids:
        boxes = {c: box(r, x, origins[fov][1:]) for c, (r, x) in cells.get(fov, {}).items()}
        labels = paint(shape, boxes)
        if dimensions == "2d":
            labels = labels.max(axis=0)[None]
        results[fov] = assign_molecules(molecules(fov, (molecules_of or {}).get(fov, []), GENES),
                                        label_run(labels, fov, grids[fov]), grid=grids[fov])
    return placement, results, grids, origins


def run(placement, results, grids, *, config=AssemblyConfig(), order=None):
    """assemble with the inputs in ``order`` (default the placement's), and what ``_duplicate_rule`` measured."""
    captured = []
    real = assemble_module._duplicate_rule

    def spy(*args, **kwargs):
        out = real(*args, **kwargs)
        captured.append(out)
        return out

    assemble_module._duplicate_rule = spy
    try:
        order = order or list(results)
        result = assemble(placement, [AssemblyInput(f, grids[f], assignment=results[f]) for f in order],
                          config=config)
    finally:
        assemble_module._duplicate_rule = real
    return result, (captured[0] if captured else ({}, {}, {}))


def measured_shares(captured):
    """{pair: share of the smaller} as the rule computed it, from the sizes and shared voxels it measured."""
    _, sizes, shared = captured
    return {pair: n / min(sizes[pair[0]], sizes[pair[1]]) for pair, n in shared.items()}


def retained_territories(result, results):
    """Per FOV with a retained cell: its territory (the cell labels; no expansion here) and its retained cells,
    those retained by their ownership centroid (the rule's input: retained now, or excluded by the rule)."""
    cells = result.cells
    before = cells[(cells.assembly_status == "retained") | (cells.assembly_reason == "duplicate_observation")]
    retained = {}
    for r in before.itertuples():
        retained.setdefault(r.fov_id, set()).add(int(r.cell_id))
    return {f: results[f].cell_labels for f in retained}, retained


def oracle(result, results, placement, origins):
    territories, retained = retained_territories(result, results)
    included = {f: origins[f] for f in placement.fovs.fov_id}
    return duplicate_oracle(territories, retained, included, (1.0, 1.0, 1.0), list(placement.fovs.fov_id))


def decisions(result):
    """{(fov, cell): kept obs_name} of the cells excluded duplicate_observation."""
    cells = result.cells
    rows = cells[cells.assembly_reason == "duplicate_observation"]
    return {(r.fov_id, int(r.cell_id)): r.duplicate_of for r in rows.itertuples()}


# --- share_cases ------------------------------------------------------------------------------------------------

SHARE_IDS = ["P1", "P2"]
# (global rows, P1's columns, P2's columns), inclusive; each P1 centroid at x < 14.5, each P2 centroid at x > 14.5.
SHARE_PAIRS = {
    0.0: ((1, 2), (8, 12), (16, 20)),     # 0 of 10 columns
    0.3: ((5, 6), (3, 12), (10, 21)),     # 3 of 10 (P1's 10 columns smaller than P2's 12)
    0.5: ((9, 10), (11, 14), (13, 18)),   # 2 of 4 (P1 smaller)
    0.8: ((13, 14), (9, 18), (11, 20)),   # 8 of 10, equal sizes
    1.0: ((17, 18), (12, 14), (12, 18)),  # 3 of 3 (P1 smaller, inside P2's)
}
# Cell IDs: P1's and P2's cell of the pair at share s.
SHARE_CELLS = {s: (k + 1, k + 1) for k, s in enumerate(SHARE_PAIRS)}
# The 50 % pair's molecules, global (z, y, x): two in P1's cell 3 (one of them in the shared columns), two in P2's
# cell 3, and unassigned ones around them on both sides of the boundary.
SHARE_MOLECULES = {
    "P1": [((1, 9, 11), "A"), ((1, 10, 14), "B"), ((1, 11, 12), "C"), ((1, 8, 16), "A"), ((2, 3, 18), "B")],
    "P2": [((1, 9, 17), "A"), ((1, 10, 14), "C"), ((1, 11, 16), "B"), ((1, 8, 12), "C"), ((2, 22, 25), "A")],
}


def share_case(dimensions="3d", ownership="nearest_center"):
    cells = {"P1": {}, "P2": {}}
    for share, (rows, a, b) in SHARE_PAIRS.items():
        ca, cb = SHARE_CELLS[share]
        cells["P1"][ca] = (rows, a)
        cells["P2"][cb] = (rows, b)
    local = {fov: [((z, y, x - (0 if fov == "P1" else 10)), g) for (z, y, x), g in rows]
             for fov, rows in SHARE_MOLECULES.items()}
    placement, results, grids, origins = build("pairs", SHARE_IDS, (4, 24, 20), 2, 1, cells, molecules_of=local,
                                               dimensions=dimensions)
    return placement, results, grids, origins, AssemblyConfig(ownership=ownership)


@pytest.mark.parametrize("ownership", OWNERSHIP)
@pytest.mark.parametrize("dimensions", ["3d", "2d"])
def test_the_measure_and_the_decision(dimensions, ownership):
    placement, results, grids, origins, config = share_case(dimensions, ownership)
    result, captured = run(placement, results, grids, config=config)
    # Every pair cell is retained by its ownership centroid, under both options.
    off, _ = run(placement, results, grids, config=AssemblyConfig(ownership=ownership, duplicate_threshold=None))
    assert set(off.cells.assembly_status) == {"retained"}
    # The shares: the rule's against the test's own placement of the two territories on one grid, exactly.
    sizes, shares, excluded = oracle(result, results, placement, origins)
    assert measured_shares(captured) == {pair: share for pair, (_, share) in shares.items()}
    expected = {(("P1", SHARE_CELLS[s][0]), ("P2", SHARE_CELLS[s][1])): s for s in SHARE_PAIRS if s > 0}
    assert measured_shares(captured) == expected
    assert captured[1] == sizes
    # The decision: the pairs at 0.5 and above lose their smaller observation (at 0.8, equal sizes, the one of P2,
    # second in placement table order); the pairs at 0 and 0.3 lose none.
    assert decisions(result) == {("P1", 3): "pairs_P2_3", ("P2", 4): "pairs_P1_4", ("P1", 5): "pairs_P2_5"}
    assert {k: f"pairs_{v[0]}_{v[1]}" for k, v in excluded.items()} == decisions(result)
    assert sizes[("P1", 4)] == sizes[("P2", 4)]
    cells = result.cells.set_index(["fov_id", "cell_id"])
    for key in decisions(result):
        assert (cells.loc[key, "assembly_status"], cells.loc[key, "owner_fov_id"]) == ("excluded", key[0])
    assert set(cells.assembly_status) == {"retained", "excluded"}
    assert cells.duplicate_of.notna().sum() == 3


# Cell IDs over the whole uint32 range (the §2.9 labels allow 1 to 2^32 − 1). Each pair: a P1 cell of 4 columns
# (x 11 to 14, 16 voxels in 3D) and a P2 cell of 6 columns (x 13 to 18, 24 voxels) sharing 2 columns (8 voxels):
# exactly 0.5 of the smaller, so P1's is excluded naming P2's, which stays.
TOP = 2 ** 32 - 1
LARGE_ID_CASES = {
    # the review's reproduction: P1's cell 2^31 against P2's cell 1
    "review": {"P1": {2 ** 31: ((9, 10), (11, 14))}, "P2": {1: ((9, 10), (13, 18))}},
    # both cells of each pair at or above 2^31, 2^32 − 1 once excluded (rows 9 to 10) and once kept (rows 13 to 14)
    "both_high": {"P1": {TOP: ((9, 10), (11, 14)), 2 ** 31: ((13, 14), (11, 14))},
                  "P2": {2 ** 31 + 5: ((9, 10), (13, 18)), TOP: ((13, 14), (13, 18))}},
}
LARGE_ID_EXCLUDED = {"review": {("P1", 2 ** 31): "large_P2_1"},
                     "both_high": {("P1", TOP): f"large_P2_{2 ** 31 + 5}", ("P1", 2 ** 31): f"large_P2_{TOP}"}}


@pytest.mark.parametrize("dimensions", ["3d", "2d"])
@pytest.mark.parametrize("case", list(LARGE_ID_CASES))
def test_the_measure_and_the_decision_keep_the_whole_uint32_range(case, dimensions):
    placement, results, grids, origins = build("large", SHARE_IDS, (4, 24, 20), 2, 1, LARGE_ID_CASES[case],
                                               dimensions=dimensions)
    off, _ = run(placement, results, grids, config=AssemblyConfig(duplicate_threshold=None))
    assert set(off.cells.assembly_status) == {"retained"}
    result, captured = run(placement, results, grids)
    sizes, shares, excluded = oracle(result, results, placement, origins)
    assert measured_shares(captured) == {pair: s for pair, (_, s) in shares.items()}
    pairs = [((("P1", a), ("P2", b))) for a, b in (((2 ** 31, 1),) if case == "review" else
                                                   ((TOP, 2 ** 31 + 5), (2 ** 31, TOP)))]
    assert measured_shares(captured) == {pair: 0.5 for pair in pairs}
    per_plane = 1 if dimensions == "2d" else 2
    assert {key: sizes[key] for pair in pairs for key in pair} == {
        key: (8 if key[0] == "P1" else 12) * per_plane for pair in pairs for key in pair}
    assert decisions(result) == LARGE_ID_EXCLUDED[case]
    assert {k: f"large_{v[0]}_{v[1]}" for k, v in excluded.items()} == LARGE_ID_EXCLUDED[case]
    cells = result.cells.set_index(["fov_id", "cell_id"])
    assert str(result.cells.cell_id.dtype) == "UInt32"
    for key, kept in LARGE_ID_EXCLUDED[case].items():
        assert (cells.loc[key, "assembly_status"], cells.loc[key, "assembly_reason"]) == \
            ("excluded", "duplicate_observation")
        kept_id = int(kept.rsplit("_", 1)[1])
        assert cells.loc[("P2", kept_id), "assembly_status"] == "retained"
        assert cells.loc[("P2", kept_id), "obs_name"] == kept


def test_placed_on_grid_is_the_nearest_voxel():
    """The test's own placement: a fractional origin shifts by the nearest voxel, and the grid voxel whose centre
    lies outside the FOV's voxel-centre extent shows nothing (here the FOV's first or last voxel is not shown)."""
    territory = np.arange(1, 7, dtype=np.int64).reshape(1, 1, 6)
    placed = placed_on_grid(territory, (0.0, 0.0, 0.3), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0), (1, 1, 7))
    assert placed.ravel().tolist() == [0, 2, 3, 4, 5, 6, 0]
    placed = placed_on_grid(territory, (0.0, 0.0, -0.6), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0), (1, 1, 6))
    assert placed.ravel().tolist() == [2, 3, 4, 5, 6, 0]


# --- partner_cases ----------------------------------------------------------------------------------------------

PARTNER_CELLS = {
    # one P1 cell (1) larger than two P2 cells (1, 2), each sharing 12 of its 14 voxels per plane with it
    "P1": {1: ((1, 6), (8, 18)), 2: ((9, 12), (11, 14))},
    # one P1 cell (2) smaller than two P2 cells (3, 4), sharing 8 of its 16 voxels per plane with each
    "P2": {1: ((1, 2), (13, 19)), 2: ((5, 6), (13, 19)), 3: ((9, 10), (11, 21)), 4: ((11, 12), (11, 20))},
}


def test_one_cell_paired_with_two_cells_of_another_fov():
    placement, results, grids, origins = build("partners", SHARE_IDS, (4, 24, 20), 2, 1, PARTNER_CELLS)
    result, captured = run(placement, results, grids)
    sizes, shares, _ = oracle(result, results, placement, origins)
    assert measured_shares(captured) == {pair: s for pair, (_, s) in shares.items()}
    assert measured_shares(captured) == {(("P1", 1), ("P2", 1)): 12 / 14, (("P1", 1), ("P2", 2)): 12 / 14,
                                         (("P1", 2), ("P2", 3)): 0.5, (("P1", 2), ("P2", 4)): 0.5}
    # Order by size: P1's 1 (132), P2's 3 (44), P2's 4 (40), P1's 2 (32), P2's 1 and 2 (28 each).
    # P1's 1 stays, so P2's 1 and 2 are excluded naming it. P2's 3 stays; P2's 4 has no partner that stays (P1's 2
    # comes later) and stays; P1's 2 pairs with both, which stay, and names the first in the order, P2's 3.
    assert decisions(result) == {("P2", 1): "partners_P1_1", ("P2", 2): "partners_P1_1", ("P1", 2): "partners_P2_3"}
    assert set(result.cells[result.cells.assembly_status == "retained"].obs_name) == {
        "partners_P1_1", "partners_P2_3", "partners_P2_4"}
    check_order_independence(placement, results, grids, result)


# --- corner_case ------------------------------------------------------------------------------------------------

CORNER_IDS = ["Q0", "Q1", "Q2", "Q3"]       # column-major: Q0 (y 0, x 0), Q1 (y 10, x 0), Q2 (y 0, x 10), Q3 (10, 10)
CORNER_CELLS = {"Q0": {1: ((11, 16), (11, 16))}, "Q1": {1: ((13, 18), (11, 16))},
                "Q2": {1: ((11, 16), (13, 18))}, "Q3": {1: ((12, 19), (12, 19))}}


def test_one_cell_seen_by_four_fovs_at_a_corner():
    placement, results, grids, origins = build("corner", CORNER_IDS, (4, 20, 20), 2, 2, CORNER_CELLS)
    off, _ = run(placement, results, grids, config=AssemblyConfig(duplicate_threshold=None))
    assert list(off.cells.assembly_status) == ["retained"] * 4       # each FOV retains its own observation
    result, captured = run(placement, results, grids)
    _, shares, _ = oracle(result, results, placement, origins)
    assert measured_shares(captured) == {pair: s for pair, (_, s) in shares.items()}
    assert measured_shares(captured) == {(("Q0", 1), ("Q1", 1)): 24 / 36, (("Q0", 1), ("Q2", 1)): 24 / 36,
                                         (("Q0", 1), ("Q3", 1)): 25 / 36, (("Q1", 1), ("Q2", 1)): 16 / 36,
                                         (("Q1", 1), ("Q3", 1)): 30 / 36, (("Q2", 1), ("Q3", 1)): 30 / 36}
    # Q3's observation is the largest (64 voxels per plane against 36) and stays although Q3 is last in table
    # order; Q0, Q1 and Q2 each pair with it above 0.5 and are excluded naming it.
    assert decisions(result) == {(f, 1): "corner_Q3_1" for f in ("Q0", "Q1", "Q2")}
    check_order_independence(placement, results, grids, result)


# --- chain_cases ------------------------------------------------------------------------------------------------

CHAIN_IDS = ["C1", "C2", "C3"]
CHAIN_CELLS = {
    # rows 1 to 2: C2's cell 1 is the largest; C1's 1 (6 of 10 shared) and C3's 1 (10 of 16) each pair with it
    # rows 5 to 8: C1's cell 2 is the largest; C2's 2 pairs with it (18 of 30) and C3's 2 with C2's 2 (10 of 18)
    "C1": {1: ((1, 2), (10, 14)), 2: ((5, 8), (3, 19))},
    "C2": {1: ((1, 2), (12, 26)), 2: ((5, 6), (11, 25))},
    "C3": {1: ((1, 2), (22, 29)), 2: ((5, 6), (21, 29))},
}


def test_a_chain_of_three_fovs():
    placement, results, grids, origins = build("chain", CHAIN_IDS, (4, 12, 20), 3, 1, CHAIN_CELLS)
    result, captured = run(placement, results, grids)
    _, shares, _ = oracle(result, results, placement, origins)
    assert measured_shares(captured) == {pair: s for pair, (_, s) in shares.items()}
    assert measured_shares(captured) == {(("C1", 1), ("C2", 1)): 0.6, (("C2", 1), ("C3", 1)): 10 / 16,
                                         (("C1", 2), ("C2", 2)): 0.6, (("C2", 2), ("C3", 2)): 10 / 18}
    # Rows 1 to 2: C2's 1 stays; C3's 1 and C1's 1 are excluded naming it.
    # Rows 5 to 8: C1's 2 stays; C2's 2 is excluded naming it; C3's 2 pairs only with C2's 2, which does not stay,
    # so it stays (C1's 2 and C3's 2 share no voxel).
    assert decisions(result) == {("C1", 1): "chain_C2_1", ("C3", 1): "chain_C2_1", ("C2", 2): "chain_C1_2"}
    retained = set(result.cells[result.cells.assembly_status == "retained"].obs_name)
    assert retained == {"chain_C2_1", "chain_C1_2", "chain_C3_2"}
    check_order_independence(placement, results, grids, result)


def check_order_independence(placement, results, grids, result):
    """The same tables and record counts for the inputs reversed and rotated."""
    ids = list(results)
    for order in (ids[::-1], ids[1:] + ids[:1]):
        other, _ = run(placement, results, grids, order=order)
        for name in ("fovs", "cells", "molecules", "counts"):
            pd.testing.assert_frame_equal(getattr(other, name), getattr(result, name), check_exact=True)
        assert other.record["counts"] == result.record["counts"]


# --- The molecules and the accounting ---------------------------------------------------------------------------

@needs_anndata
@pytest.mark.parametrize("dimensions", ["3d", "2d"])
def test_the_molecules_and_the_accounting(tmp_path, dimensions):
    import anndata
    placement, results, grids, _, _ = share_case(dimensions)
    result, _ = run(placement, results, grids)
    off, _ = run(placement, results, grids, config=AssemblyConfig(duplicate_threshold=None))
    check_accounting(result, results, placement)
    check_accounting(off, results, placement)
    mols, before = result.molecules, off.molecules
    excluded = {("P1", 3), ("P2", 4), ("P1", 5)}
    of_excluded = np.array([(f, int(c)) in excluded if not pd.isna(c) else False
                            for f, c in zip(mols.fov_id, mols.cell_id)])
    # The 50 % pair's molecules are in both observations, and unassigned ones lie around them.
    pair = mols[[(f, None if pd.isna(c) else int(c)) in {("P1", 3), ("P2", 3)} for f, c in zip(mols.fov_id,
                                                                                             mols.cell_id)]]
    assert sorted(pair.fov_id) == ["P1", "P1", "P2", "P2"] and (mols.assignment_status == "unassigned").sum() == 6
    # The molecules of the excluded observations carry the new reason; no other molecule changes status or reason.
    assert of_excluded.any()
    assert (mols.assembly_status[of_excluded] == "excluded").all()
    assert (mols.assembly_reason[of_excluded] == "cell_duplicate_observation").all()
    for column in ("assembly_status", "assembly_reason"):
        assert mols[column][~of_excluded].astype(object).fillna("").tolist() == \
            before[column][~of_excluded].astype(object).fillna("").tolist()
    assert (before.assembly_status[of_excluded] == "retained").all()
    # The record counts both reasons as the tables hold them; the populations sum to the inputs.
    counts = result.record["counts"]
    assert counts["cells"]["excluded"]["duplicate_observation"] == 3
    assert counts["molecules"]["excluded"]["cell_duplicate_observation"] == int(of_excluded.sum())
    for name, frame in (("cells", result.cells), ("molecules", mols)):
        assert counts[name]["input"] == len(frame) == counts[name]["retained"] + sum(counts[name]["excluded"].values())
    assert result.record["config"]["duplicate_threshold"] == 0.5
    # The kept cell's counts are unchanged; the excluded cells have none.
    key = ["fov_id", "cell_id", "gene_id", "compartment"]
    kept = result.counts[(result.counts.fov_id == "P2") & (result.counts.cell_id == 3)].sort_values(key)
    kept_off = off.counts[(off.counts.fov_id == "P2") & (off.counts.cell_id == 3)].sort_values(key)
    pd.testing.assert_frame_equal(kept.reset_index(drop=True), kept_off.reset_index(drop=True), check_exact=True)
    assert len(kept) and not result.counts[result.counts.fov_id.eq("P1") & result.counts.cell_id.eq(3)].shape[0]
    # sample.h5ad has one row for each pair at 0.5 and above, the kept observation's.
    adata = anndata.read_h5ad(sample_anndata(result, tmp_path / "sample.h5ad"))
    names = set(adata.obs_names)
    assert {"pairs_P2_3", "pairs_P1_4", "pairs_P2_5"} <= names
    assert not names & {"pairs_P1_3", "pairs_P2_4", "pairs_P1_5"}
    assert adata.n_obs == len(off.counted_cells()) - 3
    row = list(adata.obs_names).index("pairs_P2_3")
    whole = kept[kept.compartment == "whole"]
    assert {g: adata.X[row, list(adata.var_names).index(g)] for g in whole.gene_id} == \
        {g: float(n) for g, n in zip(whole.gene_id, whole["count"])}


# --- Refused values -----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("value", [0, 0.0, -0.25, 1.5, float("nan"), "0.5", True, False, np.bool_(True)])
def test_refused_thresholds(value):
    with pytest.raises(ValueError, match="AssemblyConfig.duplicate_threshold"):
        AssemblyConfig(duplicate_threshold=value)


@pytest.mark.parametrize("value, stored", [(None, None), (1, 1.0), (0.5, 0.5), (np.float64(0.25), 0.25), (1e-9, 1e-9)])
def test_accepted_thresholds(value, stored):
    config = AssemblyConfig(duplicate_threshold=value)
    assert config.duplicate_threshold == stored and (stored is None or type(config.duplicate_threshold) is float)
    assert AssemblyConfig().duplicate_threshold == 0.5


def test_the_rule_needs_one_spacing_among_the_fovs_with_a_retained_cell():
    placement, results, grids, _, _ = share_case()
    off = AssemblyConfig(duplicate_threshold=None)
    from starfinder.assembly._assemble import _duplicate_rule
    loaded = {f: assemble_module._Fov(f, unit_grid((4, 24, 20), (1.0, 1.0, 2.0) if f == "P2" else (1.0, 1.0, 1.0)),
                                      "assignment", result=results[f]) for f in results}
    state = {f: {"placed": True, "retained": {1: True}, "plane": False} for f in results}
    with pytest.raises(ValueError, match="share one spacing.*duplicate_threshold"):
        _duplicate_rule(placement, loaded, state, list(results), 0.5)
    run(placement, results, grids, config=off)
