"""PL2 (neighbour pairs), the box part of PL7 and the pair part of PL8 of design/validation/stitching-assembly.md.

Fixtures: ``stub_grid``, the ``stub_variants`` with irregular stage positions
(``maf_2x3_jitter`` unsnapped), unequal shapes and the L shape of ``maf_3x2_gap``, and an
8 × 8 regular grid of 1496 × 1496 px grids at 0.194398 µm and overlap 0.1 built by
``positions_from_grid`` (grids only, no image).
"""
import itertools

import pandas as pd
import pytest

from starfinder.stitching import NeighborPair, neighbor_pairs, positions_from_grid, positions_from_table

from .stitching_fixtures import (MAF_2X3_JITTER, MAF_3X2_GAP, STAGE_SPACING, STUB_SHAPE, STUB_SPACING, grid,
                                 grids_for, stage_table, stub_table)

pytestmark = [pytest.mark.stitching, pytest.mark.validation]


def rule_pairs(table, grids, tolerance=0.0):
    """The pairs by the contract's rule, written out per axis: positive overlap length of the grown boxes."""
    ids, frame = table.fov_ids, table.positions
    out = set()
    for a, b in itertools.combinations(range(len(ids)), 2):
        lengths = []
        for axis, column in enumerate(("nominal_z", "nominal_y", "nominal_x")):
            ga, gb = grids[ids[a]], grids[ids[b]]
            lo_a, lo_b = frame[column][a], frame[column][b]
            hi_a = lo_a + ga.shape_zyx[axis] * ga.metadata.spacing_zyx[axis]
            hi_b = lo_b + gb.shape_zyx[axis] * gb.metadata.spacing_zyx[axis]
            lengths.append(min(hi_a, hi_b) - max(lo_a, lo_b) + 2 * tolerance)
        volume = grids[ids[a]].shape_zyx[0] > 1 and grids[ids[b]].shape_zyx[0] > 1
        if lengths[1] > 0 and lengths[2] > 0 and (not volume or lengths[0] > 0):
            out.add((ids[a], ids[b]))
    return out


def as_set(pairs):
    return {(p.fov_a, p.fov_b) for p in pairs}


def test_stub_grid_pairs_and_boxes_use_each_grid_size():
    table = stub_table()
    grids = grids_for(table)
    pairs = neighbor_pairs(table, grids)
    expected = [("FOV0", "FOV1"), ("FOV0", "FOV2"), ("FOV0", "FOV3"), ("FOV1", "FOV2"), ("FOV1", "FOV3"),
                ("FOV2", "FOV3"), ("FOV2", "FOV4"), ("FOV2", "FOV5"), ("FOV3", "FOV4"), ("FOV3", "FOV5"),
                ("FOV4", "FOV5")]
    assert [(p.fov_a, p.fov_b) for p in pairs] == expected
    boxes = {(p.fov_a, p.fov_b): p.overlap_zyx for p in pairs}
    # Tiles are 4 µm tall (16 px) and 6 µm wide (24 px); steps 3 µm in Y and 4.5 µm in X.
    assert boxes["FOV0", "FOV1"] == ((0.0, 1.0), (3.0, 4.0), (0.0, 6.0))
    assert boxes["FOV0", "FOV2"] == ((0.0, 1.0), (0.0, 4.0), (4.5, 6.0))
    assert boxes["FOV0", "FOV3"] == ((0.0, 1.0), (3.0, 4.0), (4.5, 6.0))
    assert boxes["FOV1", "FOV2"] == ((0.0, 1.0), (3.0, 4.0), (4.5, 6.0))
    assert all(isinstance(p, NeighborPair) for p in pairs)
    assert [(p.index_a, p.index_b) for p in pairs][:3] == [(0, 1), (0, 2), (0, 3)]


def test_touching_boxes_pair_only_with_a_tolerance():
    frame = pd.DataFrame({"fov_id": ["a", "b"], "nominal_z": [0.0, 0.0], "nominal_y": [0.0, 0.0],
                          "nominal_x": [0.0, 6.0]})
    table = positions_from_table(frame)
    grids = {"a": grid(), "b": grid()}
    assert neighbor_pairs(table, grids) == []
    (pair,) = neighbor_pairs(table, grids, tolerance_um=0.25)
    assert (pair.fov_a, pair.fov_b) == ("a", "b") and pair.overlap_zyx[2] == (6.0, 6.0)
    with pytest.raises(ValueError, match="tolerance_um must be a finite number ≥ 0"):
        neighbor_pairs(table, grids, tolerance_um=-0.1)


def test_z_ranges_must_intersect_for_volumes_only():
    frame = pd.DataFrame({"fov_id": ["a", "b"], "nominal_z": [0.0, 1.0], "nominal_y": [0.0, 1.0],
                          "nominal_x": [0.0, 1.0]})
    table = positions_from_table(frame)
    assert neighbor_pairs(table, {"a": grid(), "b": grid()}) == []          # z boxes [0, 1] and [1, 2] touch
    planes = {"a": grid((1, 16, 24)), "b": grid((1, 16, 24))}
    assert as_set(neighbor_pairs(table, planes)) == {("a", "b")}


def test_pairs_do_not_depend_on_grid_cells_or_row_order():
    table = stub_table()
    order = [4, 1, 5, 0, 3, 2]
    permuted = positions_from_table(table.positions.iloc[order].drop(columns=["grid_column", "grid_row"]))
    grids = grids_for(table)
    original = neighbor_pairs(table, grids)
    shuffled = neighbor_pairs(permuted, grids)
    assert {frozenset(p) for p in as_set(original)} == {frozenset(p) for p in as_set(shuffled)}
    rank = {fov: k for k, fov in enumerate(permuted.fov_ids)}
    assert all(rank[p.fov_a] < rank[p.fov_b] for p in shuffled)
    boxes = {frozenset((p.fov_a, p.fov_b)): p.overlap_zyx for p in original}
    assert all(boxes[frozenset((p.fov_a, p.fov_b))] == p.overlap_zyx for p in shuffled)


def test_eight_by_eight_grid_has_210_pairs():
    ids = [f"Position{k:03d}" for k in range(64)]
    table = positions_from_grid(columns=8, rows=8, order="column_major_down_right", tile_shape_yx=(1496, 1496),
                                spacing_yx=(0.194398, 0.194398), overlap=0.1, fov_ids=ids)
    spacing = (0.346255, 0.194398, 0.194398)
    grids = {fov: grid((50, 1496, 1496), spacing) for fov in ids}
    pairs = neighbor_pairs(table, grids)
    cell = {fov: (c, r) for fov, c, r in zip(ids, table.positions.grid_column, table.positions.grid_row)}
    kinds = {"rows": 0, "columns": 0, "diagonal": 0}
    for p in pairs:
        (ca, ra), (cb, rb) = cell[p.fov_a], cell[p.fov_b]
        assert max(abs(ca - cb), abs(ra - rb)) == 1
        kinds["rows" if ra == rb else "columns" if ca == cb else "diagonal"] += 1
    assert len(pairs) == 210 and kinds == {"rows": 56, "columns": 56, "diagonal": 98}
    assert as_set(pairs) == rule_pairs(table, grids)


@pytest.mark.parametrize("name", ["maf_2x3_jitter", "maf_3x2_gap"])
def test_irregular_stage_positions_follow_the_rule(tmp_path, name):
    points = {"maf_2x3_jitter": MAF_2X3_JITTER, "maf_3x2_gap": MAF_3X2_GAP}[name]
    table = stage_table(tmp_path, points, name)
    grids = {fov: grid(STUB_SHAPE, STAGE_SPACING) for fov in table.fov_ids}
    pairs = neighbor_pairs(table, grids)
    assert as_set(pairs) == rule_pairs(table, grids)
    expected = {"maf_2x3_jitter": 11, "maf_3x2_gap": 6}[name]
    assert len(pairs) == expected


def test_unequal_shapes_use_each_grid():
    table = stub_table()
    narrow = {"FOV2": ((4, 16, 20), STUB_SPACING)}
    grids = grids_for(table, **narrow)
    pairs = neighbor_pairs(table, grids)
    assert as_set(pairs) == rule_pairs(table, grids)
    boxes = {(p.fov_a, p.fov_b): p.overlap_zyx for p in pairs}
    # FOV2 is 5 µm wide (20 px): its box ends at x 9.5, where FOV4 (x from 9.0) begins.
    assert boxes["FOV2", "FOV4"][2] == (9.0, 9.5)
    assert boxes["FOV0", "FOV2"][2] == (4.5, 6.0)
