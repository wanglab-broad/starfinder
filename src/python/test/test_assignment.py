"""The §2.9 assign entry: sampling, statuses, one expansion, correspondence, compartments and counts (W-314).

Rows L4, A1, A2 (on ``boxes``), A4 to A14, A16, A17, A19 and A20 of the engineering
validation design in docs/assignment-algorithms.md, the checks of docs/assignment-contract.md
("Checks the entry applies") and ``FOV.assign``. Rows A2 and A3 on ``assign_golden`` are in
``test_assignment_golden.py``. The fixtures are hand-built: ``boxes`` (cells and nuclei in
``segmentation_fixtures.py``, its molecules here), ``boxes_51_49``, ``boxes_nocal``,
``bounds``, ``plane`` and ``culture``. Every expected value is written out from the stated
geometry, computed by a formula in the test, or taken from a pinned golden output.
"""
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from skimage.segmentation import expand_labels as skimage_expand_labels

import starfinder.assignment._assign as assign_module
from starfinder.assignment import (ASSIGNMENT_STATUSES, CELL_CORRESPONDENCE, CELL_STATUSES, COMPARTMENT_STATES,
                                   NUCLEUS_STATUSES, AssignmentConfig, AssignmentResult, CorrespondenceConfig,
                                   MoleculeTable, assign_molecules, match_nuclei, molecule_table,
                                   molecule_table_from_csv, plot_assignment, sample_labels, summarize_assignment)
from starfinder.image import ImageMetadata, IncompatibleGeometryError
from starfinder.segmentation import (ExpandLabelsConfig, ReferenceGrid, SegmentationResult, ZExtensionConfig,
                                     expand_labels, extend_labels_through_z)

from .segmentation_fixtures import BOXES_METADATA, BOXES_SHAPE, CELL_BOXES, NUCLEUS_BOXES, boxes, paint
from .test_assignment_golden import MOLECULES as GOLDEN_MOLECULES
from .test_assignment_golden import SEG_LABELS, digest, label_fixture

pytestmark = [pytest.mark.segmentation, pytest.mark.validation]

FOV_IDENTITY = ["data", "sample", "FOV_001", None]
SPOT_NAMESPACE = json.dumps(FOV_IDENTITY, separators=(",", ":"))
GRID = ReferenceGrid(BOXES_SHAPE, BOXES_METADATA, "fov:round1")
GENES = ("A", "B", "C", "D")


def namespace(run, identity=FOV_IDENTITY):
    return json.dumps([*identity, run], separators=(",", ":"))


def label_run(labels, target, run, *, grid=GRID, geometry=None, **record):
    """A SegmentationResult with a minimal run record."""
    labels = np.ascontiguousarray(labels, np.uint32)
    geometry = geometry or ("plane" if labels.shape[0] == 1 else "volume")
    record = {"run": run, "target": target, "operations": [], "seeds": None, "input": None, **record}
    return SegmentationResult(labels, grid, target, geometry, namespace(run), record)


def molecules(rows, genes=GENES):
    """A MoleculeTable of ((z, y, x), gene) rows with the identities m0, m1, ..."""
    table = pd.DataFrame({
        "spot_namespace": pd.array([SPOT_NAMESPACE] * len(rows), dtype="string"),
        "spot_id": pd.array([f"m{i}" for i in range(len(rows))], dtype="string"),
        "z": np.array([p[0] for p, _ in rows], np.float64), "y": np.array([p[1] for p, _ in rows], np.float64),
        "x": np.array([p[2] for p, _ in rows], np.float64),
        "gene_id": pd.array([g for _, g in rows], dtype="string")})
    return MoleculeTable(table, tuple(genes), "final", {"spot_namespace": SPOT_NAMESPACE})


# --- The boxes fixture ------------------------------------------------------------------------------
# ((z, y, x), gene, cell, nucleus): every position is at z = 4 (or off the grid); cell and nucleus are the
# label values the geometry of segmentation_fixtures.py puts there (None off the grid).
BOXES_MOLECULES = (
    ((4, 1, 1), "A", 1, 0), ((4, 6, 6), "B", 1, 0), ((4, 3, 3), "C", 1, 11), ((4, 2, 4), "D", 1, 11),
    ((4, 1, 9), "A", 2, 0), ((4, 6, 16), "B", 2, 0), ((4, 3, 10), "C", 2, 21), ((4, 3, 11), "D", 2, 21),
    ((4, 3, 14), "C", 2, 22), ((4, 3, 15), "C", 2, 22),
    ((4, 9, 1), "A", 3, 0), ((4, 15, 2), "B", 3, 0), ((4, 12, 4), "C", 3, 31), ((4, 12, 5), "D", 3, 31),
    ((4, 9, 10), "A", 4, 0), ((4, 15, 9), "B", 4, 0), ((4, 12, 6), "C", 4, 31), ((4, 12, 7), "D", 4, 31),
    ((4, 9, 15), "A", 5, 0), ((4, 15, 18), "B", 5, 0), ((4, 12, 15), "C", 5, 51), ((4, 12, 16), "D", 5, 51),
    ((4, 19, 1), "A", 6, 0), ((4, 24, 8), "B", 6, 0), ((4, 22, 3), "C", 6, 61), ((4, 22, 5), "D", 6, 61),
    ((4, 19, 14), "A", 7, 0), ((4, 24, 15), "B", 7, 0), ((4, 22, 10), "C", 7, 61), ((4, 22, 11), "D", 7, 61),
    ((4, 20, 20), "A", 8, 0), ((4, 23, 23), "B", 8, 0),
    ((4, 28, 2), "A", 0, 0),       # background
    ((4, 30, 30), "B", 0, 91),     # inside nucleus 91, in the background
    ((4, 12, 21), "C", 0, 51),     # the background part of nucleus 51
    ((8, 4, 4), "A", None, None), ((4, -1, 4), "B", None, None), ((4, 4, 32), "C", None, None),  # off the grid
)
# cell -> (status, correspondence, flags, compartments, n_nuclei) with nuclei, by default (exclusion on)
BOXES_CELLS = {
    1: ("kept", "matched", "", "available", 1),
    2: ("kept", "matched", "several_nuclei", "available", 2),
    3: ("kept", "ambiguous", "ambiguous_nucleus", "withheld", 0),
    4: ("kept", "ambiguous", "ambiguous_nucleus", "withheld", 0),
    5: ("kept", "matched", "nucleus_outside_cell", "withheld", 1),
    6: ("kept", "matched", "nucleus_outside_cell", "withheld", 1),
    7: ("excluded_no_nucleus", "no_nucleus", "foreign_nucleus", "no_nucleus", 0),
    8: ("excluded_no_nucleus", "no_nucleus", "", "no_nucleus", 0),
}
# nucleus -> (status, cell, size, share_in_cell, share_background, n_cells_overlapped, outside)
BOXES_NUCLEI = {
    11: ("matched", 1, 27, 1.0, 0.0, 1, False),
    21: ("matched", 2, 18, 1.0, 0.0, 1, False),
    22: ("matched", 2, 18, 1.0, 0.0, 1, False),
    31: ("ambiguous", None, 100, 0.5, 0.0, 2, False),
    51: ("matched", 5, 10, 0.6, 0.4, 1, True),
    61: ("matched", 6, 10, 0.8, 0.0, 2, True),
    91: ("no_cell", None, 12, 0.0, 1.0, 0, False),
}


def boxes_inputs(*, metadata=BOXES_METADATA, nuclei_boxes=NUCLEUS_BOXES):
    grid = ReferenceGrid(BOXES_SHAPE, metadata, "fov:round1")
    cells, _ = boxes()
    return (molecules([(p, g) for p, g, _, _ in BOXES_MOLECULES]), label_run(cells, "cell", "cell", grid=grid),
            label_run(paint(nuclei_boxes), "nucleus", "nucleus", grid=grid), grid)


def boxes_51_49():
    """Nucleus 31 split 51 / 49: one voxel moves from its cell-4 half (x 7) into cell 3 (x 3), at z 6."""
    labels = paint(NUCLEUS_BOXES)
    assert labels[6, 14, 7] == 31 and labels[6, 14, 3] == 0
    labels[6, 14, 7], labels[6, 14, 3] = 0, 31
    return labels


def assign_boxes(**config):
    mols, cells, nuclei, grid = boxes_inputs()
    return assign_molecules(mols, cells, grid=grid, nuclei=nuclei, config=AssignmentConfig(**config))


def by_cell(result):
    return {int(row.cell_id): row for row in result.cells.itertuples()}


def expected_status(cell, excluded):
    if cell is None:
        return "outside_grid"
    if cell == 0:
        return "unassigned"
    return "excluded_cell" if cell in excluded else "assigned"


def check_accounting(result):
    """Identities 1-5 of docs/assignment-contract.md, recomputed from the tables (row A2)."""
    mols, cells, counts = result.molecules, result.cells, result.counts
    status = mols.assignment_status.astype(object)
    n = {s: int(status.eq(s).sum()) for s in ASSIGNMENT_STATUSES}
    assert sum(n.values()) == len(mols) and status.isin(ASSIGNMENT_STATUSES).all()
    whole = counts[counts.compartment.eq("whole").to_numpy(bool)]
    assert int(whole["count"].sum()) == n["assigned"]
    kept = cells[cells.status.eq("kept").to_numpy(bool)]
    for row in kept.itertuples():
        assert int(whole.loc[whole.cell_id.eq(row.cell_id).to_numpy(bool), "count"].sum()) == row.n_molecules
    for row in kept[kept.compartments.eq("available").to_numpy(bool)].itertuples():
        mine = counts[counts.cell_id.eq(row.cell_id).to_numpy(bool)]
        for gene in result.genes:
            part = {c: int(mine.loc[(mine.gene_id.eq(gene) & mine.compartment.eq(c)).to_numpy(bool), "count"].sum())
                    for c in ("whole", "nucleus", "cytoplasm")}
            assert part["whole"] == part["nucleus"] + part["cytoplasm"]
    unavailable = set(cells.cell_id[~cells.compartments.eq("available").to_numpy(bool)].astype(int))
    parts = counts[counts.compartment.isin(("nucleus", "cytoplasm")).to_numpy(bool)]
    assert not set(parts.cell_id.astype(int)) & unavailable
    assert int(mols.compartment.notna().sum()) == n["assigned"]
    excluded = cells[cells.status.eq("excluded_no_nucleus").to_numpy(bool)]
    assert int(excluded.n_molecules.sum()) == n["excluded_cell"]
    assert not counts.cell_id.isin(excluded.cell_id).any()
    assert counts.gene_id.isin(result.genes).all() and (counts["count"] > 0).all()


def test_the_vocabularies():
    assert ASSIGNMENT_STATUSES == ("assigned", "unassigned", "excluded_cell", "outside_grid")
    assert CELL_STATUSES == ("kept", "excluded_no_nucleus")
    assert CELL_CORRESPONDENCE == ("matched", "ambiguous", "no_nucleus", "unavailable")
    assert NUCLEUS_STATUSES == ("matched", "ambiguous", "no_cell")
    assert COMPARTMENT_STATES == ("available", "withheld", "no_nucleus", "unavailable")


def test_the_boxes_molecules_lie_where_stated():
    cells, nuclei = boxes()
    for (z, y, x), _, cell, nucleus in BOXES_MOLECULES:
        if cell is None:
            assert not (0 <= z < 8 and 0 <= y < 32 and 0 <= x < 32)
        else:
            assert (cells[z, y, x], nuclei[z, y, x]) == (cell, nucleus)
    assert (boxes_51_49() == 31).sum() == 100 and ((boxes_51_49() == 31) & (cells == 3)).sum() == 51


# --- L4: expand_labels ------------------------------------------------------------------------------

def test_l4_volumetric_expansion_in_um_labels_the_voxels_within_the_distance():
    seed = np.zeros((9, 21, 21), np.uint32)
    seed[4, 10, 10] = 1
    metadata = ImageMetadata("single", spacing_zyx=(0.3, 0.1, 0.1))
    labels, record = expand_labels(seed, metadata, config=ExpandLabelsConfig(0.5, "um", "volumetric"))
    z, y, x = np.indices(seed.shape)
    distance = np.sqrt(((z - 4) * 0.3) ** 2 + ((y - 10) * 0.1) ** 2 + ((x - 10) * 0.1) ** 2)
    assert labels.dtype == np.uint32 and np.array_equal(labels, (distance <= 0.5).astype(np.uint32))
    assert record["voxels_added"] == int((distance <= 0.5).sum()) - 1
    assert record["spacing"] == [0.3, 0.1, 0.1] and record["mode"] == "volumetric" and record["unit"] == "um"


@pytest.mark.parametrize("mode", ["planar", "volumetric"])
def test_l4_distance_zero_and_empty_images(mode):
    cells, _ = boxes()
    unchanged, record = expand_labels(cells, BOXES_METADATA, config=ExpandLabelsConfig(0, "um", mode))
    assert np.array_equal(unchanged, cells) and record["voxels_added"] == 0
    empty, _ = expand_labels(np.zeros(BOXES_SHAPE, np.uint16), None, config=ExpandLabelsConfig(3, "pixel", mode))
    assert empty.dtype == np.uint32 and not empty.any()


def test_l4_planar_expansion_follows_scikit_image_per_plane():
    """planar keeps each plane apart; in um it passes the Y, X spacing (0.25 µm is 2.5 pixels at 0.1 µm).

    No pixel offset lies at exactly 2.5 pixels (dy² + dx² is never 6.25), so the physical and the
    pixel distance select the same pixels.
    """
    cells, _ = boxes()
    labels, record = expand_labels(cells, BOXES_METADATA, config=ExpandLabelsConfig(0.25, "um", "planar"))
    oracle = np.stack([skimage_expand_labels(plane, distance=2.5) for plane in cells])
    assert np.array_equal(labels, oracle)
    assert record["spacing"] == [0.1, 0.1] and record["physical_distance_yx"] == [0.25, 0.25]
    assert not labels[0].any() and not labels[7].any()       # empty planes stay empty
    assert np.array_equal(labels[cells > 0], cells[cells > 0])


def test_l4_errors():
    cells, _ = boxes()
    with pytest.raises(ValueError, match="spacing_zyx"):
        expand_labels(cells, ImageMetadata("boxes"), config=ExpandLabelsConfig(0.5, "um", "planar"))
    with pytest.raises(ValueError, match="spacing_zyx"):
        expand_labels(cells, None, config=ExpandLabelsConfig(0.5, "um", "volumetric"))
    with pytest.raises(IncompatibleGeometryError):
        expand_labels(cells[0], None, config=ExpandLabelsConfig(1, "pixel", "planar"))
    with pytest.raises(TypeError):
        expand_labels(cells, None, config={"distance": 1})
    for fields, message in (((-1, "pixel", "planar"), "distance"), ((1, "voxel", "planar"), "unit"),
                            ((1, "pixel", "3d"), "mode"), ((float("nan"), "pixel", "planar"), "distance")):
        with pytest.raises(ValueError, match=message):
            ExpandLabelsConfig(*fields)


# --- A1: sampling at the bounds ---------------------------------------------------------------------

def test_a1_sampling_at_the_bounds():
    n = 8
    labels = np.ones((n, n, n), np.uint32)
    labels[2, 3, 3] = 0
    # coordinate -> expected voxel (None: outside_grid)
    cases = {-0.5: 0, -0.5 - 1e-9: None, 2.5: 3, 3.5: 4, n - 0.5 - 1e-9: n - 1, n - 0.5: None, 1e6: None}
    rows, expected = [], []
    for axis in range(3):
        for c, voxel in cases.items():
            position = [1.0, 1.0, 1.0]
            position[axis] = c
            rows.append((tuple(position), "A"))
            expected.append((axis, voxel))
    rows.append(((2.0, 3.0, 3.0), "A"))
    grid = ReferenceGrid((n, n, n), ImageMetadata("bounds"), "fov:round1")
    result = assign_molecules(molecules(rows), label_run(labels, "cell", "cell", grid=grid), grid=grid)
    table = result.molecules
    for i, (axis, voxel) in enumerate(expected):
        row = table.iloc[i]
        axes = ["voxel_z", "voxel_y", "voxel_x"]
        if voxel is None:
            assert row.assignment_status == "outside_grid" and pd.isna(row.cell_id)
            assert pd.isna(row[axes[axis]])
        else:
            assert row.assignment_status == "assigned" and row.cell_id == 1
            assert row[axes[axis]] == voxel
        assert all(row[a] == 1 for k, a in enumerate(axes) if k != axis)
    last = table.iloc[-1]
    assert (last.assignment_status, last.voxel_z, last.voxel_y, last.voxel_x) == ("unassigned", 2, 3, 3)
    assert pd.isna(last.cell_id)
    values, voxels, inside = sample_labels(labels, np.array([[2.5, -0.5, 7.5]]), geometry="volume")
    assert voxels.tolist() == [[3, 0, -1]] and not inside[0] and values[0] == 0


def test_sample_labels_on_a_plane_ignores_z():
    plane = np.arange(16, dtype=np.uint32).reshape(1, 4, 4)
    values, voxels, inside = sample_labels(plane, np.array([[5.0, 1.0, 2.0], [-9.0, 3.4, 0.6]]), geometry="plane")
    assert values.tolist() == [6, 13] and voxels.tolist() == [[0, 1, 2], [0, 3, 1]] and inside.all()
    with pytest.raises(ValueError, match="finite"):
        sample_labels(plane, np.array([[np.nan, 0, 0]]), geometry="plane")
    with pytest.raises(ValueError, match="disagrees"):
        sample_labels(plane, np.zeros((1, 3)), geometry="volume")


# --- A2 (boxes), A5-A11: statuses, correspondence, compartments, exclusion ---------------------------

def test_a2_boxes_statuses_and_accounting():
    result = assign_boxes()
    excluded = {c for c, row in BOXES_CELLS.items() if row[0] == "excluded_no_nucleus"}
    expected = [expected_status(cell, excluded) for _, _, cell, _ in BOXES_MOLECULES]
    assert result.molecules.assignment_status.tolist() == expected
    assert [None if pd.isna(v) else int(v) for v in result.molecules.cell_id] == \
        [c or None for _, _, c, _ in BOXES_MOLECULES]
    assert [None if pd.isna(v) else int(v) for v in result.molecules.nucleus_id] == \
        [n for _, _, _, n in BOXES_MOLECULES]
    counts = result.record["counts"]
    assert (counts["assigned"], counts["unassigned"], counts["excluded_cell"], counts["outside_grid"]) == \
        (26, 3, 6, 3)
    assert counts["molecules"] == len(BOXES_MOLECULES) == 38
    check_accounting(result)


def test_a5_to_a7_nucleus_table_and_cell_rows():
    result = assign_boxes()
    nuclei = {int(r.nucleus_id): r for r in result.nuclei.itertuples()}
    assert sorted(nuclei) == sorted(BOXES_NUCLEI)
    for n, (status, cell, size, share, background, overlapped, outside) in BOXES_NUCLEI.items():
        row = nuclei[n]
        assert (row.status, None if pd.isna(row.cell_id) else int(row.cell_id), row.size_voxels) == \
            (status, cell, size)
        assert (row.share_in_cell, row.share_background, row.n_cells_overlapped, row.outside) == \
            (share, background, overlapped, outside)
        assert pd.isna(row.seed_value_agrees)
    cells = by_cell(result)
    for c, (status, correspondence, flags, compartments, n_nuclei) in BOXES_CELLS.items():
        row = cells[c]
        assert (row.status, row.correspondence, row.correspondence_flags, row.compartments, row.n_nuclei) == \
            (status, correspondence, flags, compartments, n_nuclei), c
        if status == "kept":
            assert pd.isna(row.exclusion_reason)
        else:
            assert row.exclusion_reason == "no_matched_nucleus"


def test_a6_several_nuclei_count_both():
    result = assign_boxes()
    nucleus = result.counts[(result.counts.cell_id.eq(2) & result.counts.compartment.eq("nucleus")).to_numpy(bool)]
    # Cell 2: C, D in nucleus 21 and C, C in nucleus 22.
    assert dict(zip(nucleus.gene_id, nucleus["count"])) == {"C": 3, "D": 1}


def test_a7_a_nucleus_split_51_49():
    mols, cells, _, grid = boxes_inputs()
    nuclei = label_run(boxes_51_49(), "nucleus", "nucleus", grid=grid)
    result = assign_molecules(mols, cells, grid=grid, nuclei=nuclei)
    row = result.nuclei[result.nuclei.nucleus_id.eq(31).to_numpy(bool)].iloc[0]
    assert (row.status, row.cell_id, row.share_in_cell, row.outside) == ("matched", 3, 0.51, True)
    rows = by_cell(result)
    assert (rows[3].correspondence, rows[3].correspondence_flags, rows[3].compartments, rows[3].status) == \
        ("matched", "nucleus_outside_cell", "withheld", "kept")
    assert (rows[4].correspondence, rows[4].correspondence_flags, rows[4].compartments, rows[4].status) == \
        ("no_nucleus", "foreign_nucleus", "no_nucleus", "excluded_no_nucleus")


def test_a8_a_nucleus_outside_its_cell():
    strict, loose = assign_boxes(), assign_boxes(correspondence=CorrespondenceConfig(outside_tolerance=0.5))
    for result, outside, compartments in ((strict, True, "withheld"), (loose, False, "available")):
        row = result.nuclei[result.nuclei.nucleus_id.eq(51).to_numpy(bool)].iloc[0]
        assert (row.status, row.cell_id, bool(row.outside)) == ("matched", 5, outside)
        assert by_cell(result)[5].compartments == compartments
        background = result.molecules.iloc[34]
        assert (background.assignment_status, background.nucleus_id) == ("unassigned", 51)
        assert pd.isna(background.cell_id)


def test_a9_a_foreign_nucleus():
    result = assign_boxes(exclude_cells_without_nucleus=False)
    row = result.nuclei[result.nuclei.nucleus_id.eq(61).to_numpy(bool)].iloc[0]
    assert (row.status, row.cell_id, row.share_in_cell, bool(row.outside)) == ("matched", 6, 0.8, True)
    rows = by_cell(result)
    assert (rows[6].correspondence, rows[6].correspondence_flags, rows[6].compartments) == \
        ("matched", "nucleus_outside_cell", "withheld")
    assert (rows[7].correspondence, rows[7].correspondence_flags, rows[7].compartments, rows[7].status) == \
        ("no_nucleus", "foreign_nucleus", "no_nucleus", "kept")
    counts = result.counts
    for cell in (6, 7):
        mine = counts[counts.cell_id.eq(cell).to_numpy(bool)]
        assert set(mine.compartment) == {"whole"} and int(mine["count"].sum()) == 4


def test_a10_compartment_partition():
    result = assign_boxes()
    mols = result.molecules
    for i, (_, _, cell, nucleus) in enumerate(BOXES_MOLECULES):
        if cell in (1, 2):
            assert mols.compartment.iloc[i] == ("nucleus" if nucleus else "cytoplasm")
        elif cell in (3, 4, 5, 6):
            assert mols.compartment.iloc[i] == "withheld"
        else:
            assert pd.isna(mols.compartment.iloc[i])
    parts = result.counts[result.counts.compartment.isin(("nucleus", "cytoplasm")).to_numpy(bool)]
    assert set(parts.cell_id.astype(int)) == {1, 2}
    kept = assign_boxes(exclude_cells_without_nucleus=False)
    in_7_8 = [i for i, (_, _, cell, _) in enumerate(BOXES_MOLECULES) if cell in (7, 8)]
    assert set(kept.molecules.compartment.iloc[in_7_8]) == {"no_nucleus"}
    assert not kept.counts.cell_id.isin([7, 8]).to_numpy(bool)[kept.counts.compartment.ne("whole").to_numpy(bool)].any()
    mols, cells, _, grid = boxes_inputs()
    alone = assign_molecules(mols, cells, grid=grid)
    assigned = alone.molecules.assignment_status.eq("assigned").to_numpy(bool)
    assert set(alone.molecules.compartment[assigned]) == {"unavailable"}
    nucleus_matrix, rows = result.matrix("nucleus")
    assert rows.cell_id.tolist() == [1, 2]
    assert nucleus_matrix.tolist() == [[0, 0, 1, 1], [0, 0, 3, 1]]
    cytoplasm_matrix, _ = result.matrix("cytoplasm")
    assert cytoplasm_matrix.tolist() == [[1, 1, 0, 0], [1, 1, 0, 0]]
    whole, whole_rows = result.matrix()
    assert whole_rows.cell_id.tolist() == [1, 2, 3, 4, 5, 6]
    assert np.array_equal(whole[:2], nucleus_matrix + cytoplasm_matrix)


def test_a11_exclusion_and_its_statuses():
    default = assign_boxes()
    off = assign_boxes(exclude_cells_without_nucleus=False)
    mols, cells, _, grid = boxes_inputs()
    alone = assign_molecules(mols, cells, grid=grid)
    rows = by_cell(default)
    assert [c for c, r in rows.items() if r.status == "excluded_no_nucleus"] == [7, 8]
    assert {rows[c].exclusion_reason for c in (7, 8)} == {"no_matched_nucleus"}
    assert rows[3].status == rows[4].status == "kept"
    excluded_molecules = default.molecules[default.molecules.assignment_status.eq("excluded_cell").to_numpy(bool)]
    assert sorted(excluded_molecules.cell_id.astype(int).tolist()) == [7, 7, 7, 7, 8, 8]
    assert not default.counts.cell_id.isin([7, 8]).any()
    config = default.record["config"]
    assert (config["exclude_cells_without_nucleus"], config["exclusion_source"]) == (True, "default")
    assert config["exclusion_rationale"] == "possible cell residue; not a biological identity"
    assert off.record["config"]["exclusion_source"] == "config"
    # Totals before and after: the excluded cells and their molecules, exactly.
    before, _ = default.matrix(cells="all")
    after, _ = default.matrix()
    assert before.shape[0] - after.shape[0] == 2 and before.sum() - after.sum() == 6
    assert len(default.cells) == 8 and default.record["counts"]["cells_excluded"] == 2
    assert np.array_equal(off.matrix()[0], before)
    assert set(by_cell(off)[c].status for c in range(1, 9)) == {"kept"}
    assert {by_cell(off)[c].compartments for c in (7, 8)} == {"no_nucleus"}
    assert not alone.cells.status.eq("excluded_no_nucleus").any()
    assert set(alone.cells.correspondence) == {"unavailable"} and set(alone.cells.compartments) == {"unavailable"}
    assert alone.cells.correspondence_flags.isna().all() and alone.cells.n_nuclei.isna().all()
    assert alone.molecules.nucleus_id.isna().all() and alone.nuclei is None and alone.nucleus_labels is None
    assert (alone.record["config"]["exclude_cells_without_nucleus"], alone.record["inputs"]["nuclei"]) == \
        (False, None)
    for result in (default, off, alone):
        check_accounting(result)


def test_a12_supplied_correspondence_equal_to_the_derived_one():
    mols, cells, nuclei, grid = boxes_inputs()
    derived = assign_molecules(mols, cells, grid=grid, nuclei=nuclei)
    matched = derived.nuclei[derived.nuclei.status.eq("matched").to_numpy(bool)]
    table = pd.DataFrame({"nucleus_id": matched.nucleus_id.astype("int64").to_numpy(),
                          "cell_id": matched.cell_id.astype("int64").to_numpy()})
    supplied = assign_molecules(mols, cells, grid=grid, nuclei=nuclei, correspondence=table)
    for name in ("molecules", "cells", "counts", "nuclei"):
        assert_frame_equal(getattr(supplied, name), getattr(derived, name), check_exact=True)
    for name in ("cell_labels", "nucleus_labels"):
        assert np.array_equal(getattr(supplied, name), getattr(derived, name))
    assert supplied.territories is None and derived.territories is None
    record_s, record_d = (json.loads(json.dumps(r.record)) for r in (supplied, derived))
    assert record_s["inputs"].pop("correspondence")["source"] == "supplied"
    assert record_d["inputs"].pop("correspondence") == {"source": "overlap", "sha256": None}
    assert record_s == record_d
    bad = [(pd.DataFrame({"nucleus_id": [99], "cell_id": [1]}), "absent from the nucleus image: \\[99\\]"),
           (pd.DataFrame({"nucleus_id": [11], "cell_id": [42]}), "absent from the cell image: \\[42\\]"),
           (pd.DataFrame({"nucleus_id": [11, 11], "cell_id": [1, 2]}),
            "more than once in the supplied correspondence: \\[11\\]"),
           (pd.DataFrame({"nucleus_id": [11.5], "cell_id": [1]}), "nonnegative integers")]
    for frame, message in bad:
        with pytest.raises(ValueError, match=message):
            assign_molecules(mols, cells, grid=grid, nuclei=nuclei, correspondence=frame)
    with pytest.raises(ValueError, match="needs nuclei"):
        assign_molecules(mols, cells, grid=grid, correspondence=table)


def test_a12_a_partial_supplied_table_leaves_the_rest_to_overlap():
    """A nucleus absent from the table is ambiguous when it overlaps a cell, no_cell otherwise."""
    mols, cells, nuclei, grid = boxes_inputs()
    result = assign_molecules(mols, cells, grid=grid, nuclei=nuclei,
                              correspondence=pd.DataFrame({"nucleus_id": [31], "cell_id": [4]}))
    status = dict(zip(result.nuclei.nucleus_id.astype(int), result.nuclei.status))
    assert status == {11: "ambiguous", 21: "ambiguous", 22: "ambiguous", 31: "matched", 51: "ambiguous",
                      61: "ambiguous", 91: "no_cell"}
    rows = by_cell(result)
    assert (rows[4].correspondence, rows[4].correspondence_flags) == ("matched", "nucleus_outside_cell")
    assert (rows[3].correspondence, rows[3].correspondence_flags) == ("no_nucleus", "foreign_nucleus")


# --- A13, A14: cell metadata, grid and identity checks ----------------------------------------------

def test_a13_cell_metadata():
    result = assign_boxes()
    rows = by_cell(result)
    for c, ranges in CELL_BOXES.items():
        volume = int(np.prod([hi - lo for lo, hi in ranges]))
        centre = [(lo + hi - 1) / 2 for lo, hi in ranges]
        row = rows[c]
        assert row.size_voxels == volume
        assert row.size_physical == pytest.approx(volume * 0.35 * 0.1 * 0.1, rel=1e-12, abs=0)
        assert [row.centroid_z, row.centroid_y, row.centroid_x] == centre
        assert pd.isna(row.expanded_size_voxels) and np.isnan(row.expanded_centroid_x)
        assert np.isnan(row.expanded_size_physical)
    assert (result.record["calibration"], result.record["calibration_source"], result.record["size_unit"]) == \
        ("known", "grid", "micrometer^3")
    mols, cells, nuclei, grid = boxes_inputs(metadata=ImageMetadata("boxes"))
    nocal = assign_molecules(mols, cells, grid=grid, nuclei=nuclei)
    assert nocal.cells.size_physical.isna().all() and nocal.cells.size_voxels.tolist() == \
        result.cells.size_voxels.tolist()
    assert (nocal.record["calibration"], nocal.record["calibration_source"], nocal.record["size_unit"]) == \
        ("unknown", None, None)


def test_a14_grid_and_identity_checks_raise_before_sampling(monkeypatch):
    def never(*args, **kwargs):
        raise AssertionError("sampled before the checks")
    monkeypatch.setattr(assign_module, "sample_labels", never)
    mols, cells, nuclei, grid = boxes_inputs()
    shifted = ReferenceGrid(BOXES_SHAPE, replace(BOXES_METADATA, frame_id="boxes/shifted"), "fov:round1")
    other_shape = ReferenceGrid((8, 32, 30), BOXES_METADATA, "fov:round1")
    projected_sum = grid.projected(method="sum")
    plane_cells = label_run(boxes()[0][4:5], "cell", "cell", grid=projected_sum,
                            input={"projection": {"axis": "z", "method": "max"}})
    other_fov = SegmentationResult(cells.labels, grid, "cell", "volume",
                                   namespace("cell", ["data", "sample", "FOV_002", None]), cells.record)
    cases = [
        (lambda: assign_molecules(mols, label_run(cells.labels, "cell", "cell", grid=shifted), grid=grid),
         IncompatibleGeometryError, "boxes/shifted"),
        (lambda: assign_molecules(mols, label_run(cells.labels[:, :, :30], "cell", "cell", grid=other_shape),
                                  grid=grid), IncompatibleGeometryError, "30"),
        (lambda: assign_molecules(mols, plane_cells, grid=grid), IncompatibleGeometryError, "projection:max"),
        (lambda: assign_molecules(mols, cells, grid=grid,
                                  nuclei=label_run(nuclei.labels, "nucleus", "nucleus", grid=shifted)),
         IncompatibleGeometryError, "cell grid"),
        (lambda: assign_molecules(mols, other_fov, grid=grid), ValueError, "FOV_002"),
        (lambda: assign_molecules(mols, cells, grid=grid, nuclei=SegmentationResult(
            nuclei.labels, grid, "nucleus", "volume", namespace("nucleus", ["data", "other", "FOV_001", None]),
            nuclei.record)), ValueError, "other"),
    ]
    for build, error, message in cases:
        with pytest.raises(error, match=message):
            build()


def test_the_checks_of_the_entry():
    mols, cells, nuclei, grid = boxes_inputs()
    expanded = label_run(cells.labels, "cell", "cell",
                         operations=[{"operation": "expand_labels", "config": {"distance": 1}}])
    message = ("labels of run 'cell' were expanded by segmentation, which keeps no original mask; run segment "
               "without the expansion and set AssignmentConfig.expansion")
    cases = [
        (lambda: assign_molecules(mols, cells, grid=grid, config={}), TypeError, "AssignmentConfig"),
        (lambda: assign_molecules(mols.table, cells, grid=grid), TypeError, "MoleculeTable"),
        (lambda: assign_molecules(mols, cells.labels, grid=grid), TypeError, "SegmentationResult"),
        (lambda: assign_molecules(mols, cells, grid=BOXES_SHAPE), TypeError, "ReferenceGrid"),
        (lambda: assign_molecules(mols, cells, grid=grid, nuclei=cells), ValueError, "target nucleus"),
        (lambda: assign_molecules(mols, expanded, grid=grid), ValueError, message),
        (lambda: assign_molecules(mols, expanded, grid=grid, config=AssignmentConfig(
            expansion=ExpandLabelsConfig(0.1, "um", "planar"))), ValueError, "expanded by segmentation"),
        (lambda: assign_molecules(mols, cells, grid=grid, config=AssignmentConfig(
            expansion=ExpandLabelsConfig(1, "pixel", "planar"))), ValueError, "legacy_pixel_expansion"),
        (lambda: AssignmentConfig(expansion={"distance": 1}), TypeError, "ExpandLabelsConfig"),
        (lambda: AssignmentConfig(exclude_cells_without_nucleus=1), TypeError, "bool"),
        (lambda: AssignmentConfig(correspondence=0.5), TypeError, "CorrespondenceConfig"),
        (lambda: CorrespondenceConfig(match_fraction=0.4), ValueError, "match_fraction"),
        (lambda: CorrespondenceConfig(match_fraction=1.0), ValueError, "match_fraction"),
        (lambda: CorrespondenceConfig(outside_tolerance=-0.1), ValueError, "outside_tolerance"),
        (lambda: CorrespondenceConfig(outside_tolerance=1), ValueError, "outside_tolerance"),
    ]
    for build, error, text in cases:
        with pytest.raises(error, match=text if text != message else None) as raised:
            build()
        if text == message:
            assert str(raised.value) == message
    legacy = assign_molecules(mols, cells, grid=grid, config=AssignmentConfig(
        expansion=ExpandLabelsConfig(1, "pixel", "planar"), legacy_pixel_expansion=True))
    assert legacy.record["expansion"]["physical_distance_yx"] == [0.1, 0.1]
    assert legacy.record["config"]["legacy_pixel_expansion"] is True


def test_the_cell_run_may_be_a_nucleus_run_used_as_territories():
    mols, _, nuclei, grid = boxes_inputs()
    result = assign_molecules(mols, nuclei, grid=grid,
                              config=AssignmentConfig(expansion=ExpandLabelsConfig(0.2, "um", "volumetric")))
    assert result.record["territory_source"] == {"run": "nucleus", "target": "nucleus", "origin": "segmented",
                                                 "expanded_by_assign": True}
    assert result.cells.cell_id.astype(int).tolist() == sorted(NUCLEUS_BOXES)
    assert (result.territories[nuclei.labels > 0] == nuclei.labels[nuclei.labels > 0]).all()


# --- A4: one expansion on assign_golden (without the gene-Z molecule) -------------------------------

GOLDEN_GRID = ReferenceGrid((16, 64, 64), ImageMetadata("assign_golden"), "fov:round1")
GOLDEN_KEEP = [i for i, (_, gene) in enumerate(GOLDEN_MOLECULES) if gene != "Z"]


def golden_inputs(dimensions):
    labels = label_fixture()
    rows = [GOLDEN_MOLECULES[i] for i in GOLDEN_KEEP]
    mols = molecules(rows, genes=("A", "B", "C", "D", "E"))
    if dimensions == "3d":
        return mols, label_run(labels, "cell", "cell", grid=GOLDEN_GRID)
    plane_grid = GOLDEN_GRID.projected()
    return mols, label_run(labels.max(axis=0)[None], "cell", "cell", grid=plane_grid,
                           input={"projection": {"axis": "z", "method": "max"}})


@pytest.mark.parametrize("dimensions", ["3d", "2d"])
def test_a4_one_expansion_keeps_both_masks(dimensions):
    mols, cells = golden_inputs(dimensions)
    config = ExpandLabelsConfig(4, "pixel", "planar")
    result = assign_molecules(mols, cells, grid=GOLDEN_GRID, config=AssignmentConfig(expansion=config))
    oracle = np.stack([skimage_expand_labels(plane, distance=4) for plane in cells.labels])
    assert np.array_equal(result.territories, oracle) and result.territories.dtype == np.uint32
    assert np.array_equal(result.territories, expand_labels(cells.labels, cells.grid.metadata, config=config)[0])
    assert np.array_equal(result.cell_labels, cells.labels)
    before = [SEG_LABELS[(dimensions, False)][i] for i in GOLDEN_KEEP]
    after = [SEG_LABELS[(dimensions, True)][i] for i in GOLDEN_KEEP]
    band = [b == 0 and a > 0 for b, a in zip(before, after)]
    assert sum(band) == 3
    assert result.molecules.in_expansion.tolist() == band
    assert [int(v) for v in result.molecules.original_cell_id] == before
    assert [int(v) for v in result.molecules.original_cell_id[band]] == [0, 0, 0]
    entry = result.record["expansion"]
    added = int((oracle > 0).sum() - (cells.labels > 0).sum())
    assert entry["voxels_added"] == added and entry["mode"] == "planar" and entry["unit"] == "pixel"
    assert entry["original"] == {"path": None, "sha256": digest(cells.labels)}
    assert entry["expanded"] == {"path": None, "sha256": digest(oracle.astype(np.uint32))}
    assert result.record["territory_source"]["expanded_by_assign"] is True
    plain = assign_molecules(mols, cells, grid=GOLDEN_GRID)
    assert plain.territories is None and plain.record["expansion"] is None
    assert plain.molecules.in_expansion.isna().all() and plain.molecules.original_cell_id.isna().all()
    assert plain.cells.expanded_size_voxels.isna().all()


@pytest.mark.parametrize("expansion", [None, ExpandLabelsConfig(4, "pixel", "planar")])
def test_a4_a_cell_run_expanded_by_segmentation_raises(expansion):
    mols, cells = golden_inputs("3d")
    record = dict(cells.record, operations=[{"operation": "expand_labels",
                                             "config": {"distance": 4, "unit": "pixel", "mode": "planar"}}])
    expanded = replace(cells, record=record)
    with pytest.raises(ValueError, match="expanded by segmentation"):
        assign_molecules(mols, expanded, grid=GOLDEN_GRID, config=AssignmentConfig(expansion=expansion))


def test_a4_um_on_the_uncalibrated_grid_raises():
    mols, cells = golden_inputs("3d")
    with pytest.raises(ValueError, match="calibrated"):
        assign_molecules(mols, cells, grid=GOLDEN_GRID,
                         config=AssignmentConfig(expansion=ExpandLabelsConfig(0.4, "um", "planar")))


# --- A16, A17: Z=1 and culture ------------------------------------------------------------------------

def test_a16_a_plane_on_a_projected_grid():
    cells, _ = boxes()
    plane = label_run(cells[4:5], "cell", "cell", grid=GRID.projected(),
                      input={"projection": {"axis": "z", "method": "max"}})
    in_grid = [(p, g, c) for p, g, c, _ in BOXES_MOLECULES if c is not None]
    rows = [((i % 8, y, x), g) for i, ((_, y, x), g, _) in enumerate(in_grid)] + [((1000, 4, 4), "A")]
    result = assign_molecules(molecules(rows), plane, grid=GRID)
    table = result.molecules
    expected = [int(cells[4, y, x]) for (_, y, x), _ in rows[:-1]]
    assert [0 if pd.isna(v) else int(v) for v in table.cell_id[:-1]] == expected
    assert table.assignment_status[:-1].tolist() == ["assigned" if c else "unassigned" for c in expected]
    assert table.voxel_z[:-1].tolist() == [0] * len(expected)
    assert table.assignment_status.iloc[-1] == "outside_grid" and pd.isna(table.voxel_z.iloc[-1])
    assert (table.voxel_y.iloc[-1], table.voxel_x.iloc[-1]) == (4, 4)
    assert result.record["sampling"] == {"rule": "floor(c + 0.5)", "sampled_axes": "yx"}
    assert (result.record["calibration"], result.record["calibration_source"], result.record["size_unit"]) == \
        ("known", "projection_source", "micrometer^2")
    for c, ranges in CELL_BOXES.items():
        pixels = (ranges[1][1] - ranges[1][0]) * (ranges[2][1] - ranges[2][0])
        row = by_cell(result)[c]
        assert row.size_voxels == pixels and row.centroid_z == 0.0
        assert row.size_physical == pytest.approx(pixels * 0.1 * 0.1, rel=1e-12, abs=0)
    check_accounting(result)


CULTURE_CELLS = {1: ((0, 1), (2, 12), (2, 12)), 2: ((0, 1), (2, 12), (16, 28)), 3: ((0, 1), (16, 28), (4, 20))}
CULTURE_NUCLEI = {11: ((0, 1), (4, 8), (4, 8)), 21: ((0, 1), (5, 9), (18, 22)), 31: ((0, 1), (20, 24), (8, 12))}


def test_a17_culture_labels_extended_through_z():
    shape = (8, 32, 32)
    grid = ReferenceGrid(shape, BOXES_METADATA, "fov:round1")
    stain = np.full(shape, 10, np.uint8)
    stain[2:5] = 200
    config = ZExtensionConfig(median_um=0.1, threshold=100 / 255, min_area_um2=0.01, dilation_um=0.0,
                              fill_holes="once")
    cells_2d, nuclei_2d = paint(CULTURE_CELLS, (1, 32, 32)), paint(CULTURE_NUCLEI, (1, 32, 32))
    cells, _ = extend_labels_through_z(cells_2d, stain, BOXES_METADATA, config=config)
    nuclei, _ = extend_labels_through_z(nuclei_2d, stain, BOXES_METADATA, config=config)
    layer = np.zeros(shape, bool)
    layer[2:5] = True
    assert np.array_equal(cells, np.where(layer, cells_2d, 0)) and np.array_equal(nuclei, np.where(layer, nuclei_2d, 0))
    rows = []
    for z in (0, 2, 3, 4, 6):
        rows += [((z, 3, 3), "A"), ((z, 6, 6), "B"), ((z, 7, 20), "C"), ((z, 10, 26), "A"), ((z, 22, 10), "D"),
                 ((z, 17, 18), "B"), ((z, 30, 30), "C")]
    result = assign_molecules(molecules(rows), label_run(cells, "cell", "cell", grid=grid, geometry="extended"),
                              grid=grid, nuclei=label_run(nuclei, "nucleus", "nucleus", grid=grid,
                                                          geometry="extended"))
    table = result.molecules
    for i, ((z, y, x), _) in enumerate(rows):
        cell, nucleus = int(cells_2d[0, y, x]), int(nuclei_2d[0, y, x])
        if 2 <= z <= 4 and cell:
            assert table.assignment_status.iloc[i] == "assigned" and table.cell_id.iloc[i] == cell
            assert table.compartment.iloc[i] == ("nucleus" if nucleus else "cytoplasm")
        else:
            assert table.assignment_status.iloc[i] == "unassigned"
    assert set(result.cells.correspondence) == {"matched"} and set(result.cells.compartments) == {"available"}
    assert result.nuclei.cell_id.astype(int).tolist() == [1, 2, 3]
    check_accounting(result)


# --- A19, A20: determinism and row order --------------------------------------------------------------

HASH_SCRIPT = """
import json
import sys
sys.path.insert(0, {test_root!r})
from test.test_assignment import assign_boxes, golden_inputs, GOLDEN_GRID
from starfinder.assignment import AssignmentConfig, assign_molecules
from starfinder.assignment._assign import table_sha256
from starfinder.segmentation import ExpandLabelsConfig
results = {{"boxes": assign_boxes()}}
for dimensions in ("3d", "2d"):
    mols, cells = golden_inputs(dimensions)
    for expand in (False, True):
        config = AssignmentConfig(expansion=ExpandLabelsConfig(4, "pixel", "planar") if expand else None)
        results[f"golden_{{dimensions}}_{{expand}}"] = assign_molecules(mols, cells, grid=GOLDEN_GRID, config=config)
tables = ("molecules", "cells", "counts", "nuclei")
print(json.dumps({{key: {{name: table_sha256(getattr(result, name)) for name in tables
                        if getattr(result, name) is not None}} for key, result in results.items()}}, sort_keys=True))
"""


@pytest.mark.slow
def test_a19_three_single_thread_processes_give_identical_tables():
    root = str(Path(__file__).resolve().parents[1])
    script = HASH_SCRIPT.format(test_root=root)
    env = {**__import__("os").environ, **{v: "1" for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                                             "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                                                             "NUMBA_NUM_THREADS")}}
    outputs = [subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True, env=env,
                              cwd=root).stdout for _ in range(3)]
    hashes = [json.loads(o) for o in outputs]
    assert hashes[0] == hashes[1] == hashes[2]
    assert set(hashes[0]["boxes"]) == {"molecules", "cells", "counts", "nuclei"}


def test_a20_row_order():
    mols, cells, nuclei, grid = boxes_inputs()
    order = np.random.default_rng(100).permutation(len(mols.table))
    shuffled = MoleculeTable(mols.table.iloc[order].reset_index(drop=True), mols.genes, mols.population, mols.source)
    assert shuffled.sha256 == mols.sha256
    a = assign_molecules(mols, cells, grid=grid, nuclei=nuclei)
    b = assign_molecules(shuffled, cells, grid=grid, nuclei=nuclei)
    key = ["spot_namespace", "spot_id"]
    assert_frame_equal(a.molecules.sort_values(key).reset_index(drop=True),
                       b.molecules.sort_values(key).reset_index(drop=True), check_exact=True)
    for name in ("cells", "counts", "nuclei"):
        assert_frame_equal(getattr(a, name), getattr(b, name), check_exact=True)
    assert b.molecules.spot_id.tolist() == mols.table.spot_id.iloc[order].tolist()


# --- Molecule tables ------------------------------------------------------------------------------------

def test_molecule_table_validation():
    good = molecules([((1, 2, 3), "A")])
    assert len(good) == 1 and good.spot_namespace == SPOT_NAMESPACE and len(good.sha256) == 64
    table = good.table
    cases = [
        (table.drop(columns="gene_id"), {}, ValueError, "exactly the columns"),
        (table.assign(z=table.z.astype("float32")), {}, ValueError, "float64"),
        (table.assign(spot_id=table.spot_id.astype(object)), {}, ValueError, "string"),
        (pd.concat([table, table], ignore_index=True), {}, ValueError, "unique"),
        (table.assign(x=[np.inf]), {}, ValueError, "row 0"),
        (table.assign(gene_id=pd.array(["Q"], dtype="string")), {}, ValueError, "\\['Q'\\]"),
        (table.assign(gene_id=pd.array([None], dtype="string")), {}, ValueError, "gene_id"),
        (table.assign(spot_namespace=pd.array(['["x"]'], dtype="string")), {}, ValueError, "one FOV"),
        (table, {"genes": ("A", "A")}, ValueError, "distinct"),
        (table, {"population": "all"}, ValueError, "population"),
        (table, {"source": {}}, ValueError, "spot_namespace"),
    ]
    for frame, changes, error, message in cases:
        fields = {"genes": good.genes, "population": "final", "source": good.source, **changes}
        with pytest.raises(error, match=message):
            MoleculeTable(frame, fields["genes"], fields["population"], fields["source"])


def test_molecule_table_from_csv(tmp_path):
    path = tmp_path / "goodSpots.csv"
    path.write_text("x,y,z,gene\n1,2,3,A\n17.0,17.5,7.0,B\n")
    table = molecule_table_from_csv(path, spot_namespace=SPOT_NAMESPACE, genes=GENES)
    frame = table.table
    assert frame.spot_id.tolist() == ["csv:0", "csv:1"]
    assert frame[["z", "y", "x"]].to_numpy().tolist() == [[2.0, 1.0, 0.0], [6.0, 16.5, 16.0]]
    assert table.population == "final" and table.source["csv"] == str(path) and len(table.source["sha256"]) == 64
    path.write_text("x,y,z,gene\n1,2,3,Z\n")
    with pytest.raises(ValueError, match="\\['Z'\\]"):
        molecule_table_from_csv(path, spot_namespace=SPOT_NAMESPACE, genes=GENES)
    path.write_text("x,y,z,gene\n1,2,nan,A\n")
    with pytest.raises(ValueError, match="row 0"):
        molecule_table_from_csv(path, spot_namespace=SPOT_NAMESPACE, genes=GENES)
    path.write_text("x,y,gene\n1,2,A\n")
    with pytest.raises(ValueError, match="\\['z'\\]"):
        molecule_table_from_csv(path, spot_namespace=SPOT_NAMESPACE, genes=GENES)
    with pytest.raises(FileNotFoundError):
        molecule_table_from_csv(tmp_path / "missing.csv", spot_namespace=SPOT_NAMESPACE, genes=GENES)


def test_match_nuclei_on_arrays_equals_the_entry():
    result = assign_boxes()
    cells, nuclei = boxes()
    nucleus_table, per_cell = match_nuclei(nuclei, cells)
    assert_frame_equal(nucleus_table, result.nuclei, check_exact=True)
    assert per_cell.correspondence.tolist() == result.cells.correspondence.tolist()
    with pytest.raises(IncompatibleGeometryError):
        match_nuclei(nuclei[:4], cells)


def test_an_empty_territory_image_keeps_every_molecule():
    mols, _, _, grid = boxes_inputs()
    empty = label_run(np.zeros(BOXES_SHAPE, np.uint32), "cell", "cell")
    result = assign_molecules(mols, empty, grid=grid)
    assert result.record["outcome"] == "empty" and len(result.cells) == 0 and len(result.counts) == 0
    assert set(result.molecules.assignment_status) == {"unassigned", "outside_grid"}
    assert result.matrix()[0].shape == (0, 4)


def test_result_identities_are_checked_at_construction():
    result = assign_boxes()
    counts = result.counts.copy()
    counts.loc[0, "count"] += 1
    with pytest.raises(AssertionError, match="identity 2"):
        replace(result, counts=counts)
    with pytest.raises(ValueError, match="compartment"):
        result.matrix("membrane")
    with pytest.raises(ValueError, match="cells"):
        result.matrix(cells="excluded")


def test_summary_and_plot():
    result = assign_boxes()
    summary = summarize_assignment(result)
    assert summary["molecules_by_status"] == {"assigned": 26, "unassigned": 3, "excluded_cell": 6, "outside_grid": 3}
    assert summary["cells_by_flag"] == {"several_nuclei": 1, "ambiguous_nucleus": 2, "nucleus_outside_cell": 2,
                                        "foreign_nucleus": 1}
    assert summary["nuclei_by_status"] == {"matched": 5, "ambiguous": 1, "no_cell": 1}
    assert summary["exclusion"]["before"]["cells"] == 8 and summary["exclusion"]["after"]["cells"] == 6
    assert summary["exclusion"]["after"]["whole"] == 26
    assert summary["exclusion"]["after"]["nucleus"] + summary["exclusion"]["after"]["cytoplasm"] == 10
    assert summary["quantiles"]["size_voxels"]["0.0"] == 210.0
    json.dumps(summary)
    import matplotlib.pyplot as plt
    figure = plot_assignment(result, image=np.zeros(BOXES_SHAPE), z=4)
    assert len(figure.axes) >= 4
    plt.close(figure)
    with pytest.raises(ValueError, match="z must be"):
        plot_assignment(result, z=9)


# --- FOV.assign -------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def decoded_fov(tmp_path_factory):
    """FOV_001 of the development preset 'clean', size 'small', after FOV.run through filtering."""
    import warnings

    from starfinder.barcode import NeighborhoodSumConfig, ReadFilterConfig, WtaDecoderConfig
    from starfinder.dataset import (
        Dataset,
        PipelineConfig,
        RegistrationRecipe,
        RegistrationStep,
        RoundState,
    )
    from starfinder.io import ImageLoadConfig, save_volume
    from starfinder.registration import TranslationConfig
    from starfinder.spot_finding import LocalMaximaConfig
    from starfinder.synthetic import development_scene_preset, generate_formed_scene

    book, config = development_scene_preset("clean", size="small")
    scene = generate_formed_scene(book, config=config)
    root = tmp_path_factory.mktemp("assign_fov")
    for name, image in scene.rounds.items():
        metadata = replace(scene.round_metadata[name], spacing_zyx=(0.35, 0.1, 0.1), spatial_unit="micrometer")
        for c, channel in enumerate(scene.channel_labels):
            save_volume(image[..., c], root / name / "FOV_001" / f"{channel}.tif", metadata=metadata)
    rounds = RoundState(list(scene.round_labels), reference_round=scene.round_labels[0])
    dataset = Dataset(root, root / "out", "dev", "sample", "out", rounds, list(scene.channel_labels))
    dataset.codebook = book
    pipeline = PipelineConfig(load=ImageLoadConfig(channel_labels=tuple(scene.channel_labels)),
                              registration=RegistrationRecipe((RegistrationStep(TranslationConfig()),)),
                              spot_finding=LocalMaximaConfig(), extraction=NeighborhoodSumConfig(),
                              decoding=WtaDecoderConfig(), filtering=ReadFilterConfig())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fov = dataset.fov("FOV_001").run(pipeline)
    return fov, root


def fov_masks(fov):
    """Cell boxes and nucleus boxes on the FOV's grid that hold its accepted reads."""
    grid = fov.reference_grid()
    shape = grid.shape_zyx
    cells = paint({1: ((0, shape[0]), (0, shape[1] // 2), (0, shape[2])),
                   2: ((0, shape[0]), (shape[1] // 2, shape[1]), (0, shape[2]))}, shape)
    nuclei = paint({5: ((0, shape[0]), (2, 6), (2, 6)), 9: ((0, shape[0]), (20, 24), (20, 24))}, shape)
    return grid, cells, nuclei


@pytest.mark.dataset
def test_fov_assign_runs_the_entry_on_the_fov_results(decoded_fov, tmp_path):
    from starfinder.io import save_volume
    from starfinder.segmentation import LabelImportConfig, SegmentationPlan, SegmentationRun
    fov, root = decoded_fov
    grid, cells, nuclei = fov_masks(fov)
    for name, mask in (("cells", cells), ("nuclei", nuclei)):
        save_volume(mask, tmp_path / f"{name}.tif", metadata=grid.metadata)
    fov.segmentation_results.clear()
    fov.segment(SegmentationPlan((
        SegmentationRun("nucleus", "nucleus", (), LabelImportConfig(str(tmp_path / "nuclei.tif"), "nucleus")),
        SegmentationRun("cell", "cell", (), LabelImportConfig(str(tmp_path / "cells.tif"), "cell")))))
    files = sorted(p for p in root.rglob("*"))
    assert fov.assign(nuclei="nucleus") is fov
    assert sorted(p for p in root.rglob("*")) == files     # nothing is written
    result = fov.assignment_results["default"]
    mols = molecule_table(fov.spot_result, fov.filtering_result, genes=fov.codebook.genes)
    direct = assign_molecules(mols, fov.segmentation_results["cell"], grid=grid,
                              nuclei=fov.segmentation_results["nucleus"])
    for name in ("molecules", "cells", "counts", "nuclei"):
        assert_frame_equal(getattr(result, name), getattr(direct, name), check_exact=True)
    assert result.record["name"] == "default" and result.record["counts"] == summarize_assignment(direct)
    assert result.record["territory_source"]["origin"] == "imported"
    assert result.record["inputs"]["molecules"]["sha256"] == mols.sha256
    assert result.record["grid"]["check"] == "checked" and result.record["grid"]["sha256"] == grid.sha256
    assert len(result.molecules) == fov.filtering_result.counts["accepted"]
    imported = fov.segmentation_results["cell"]
    fov.assign(AssignmentConfig(exclude_cells_without_nucleus=False), cells=imported, name="all_cells",
               population="called")
    assert set(fov.assignment_results) == {"default", "all_cells"}
    assert fov.assignment_results["all_cells"].record["inputs"]["molecules"]["population"] == "called"
    errors = [(dict(checkpoints=object()), TypeError, "checkpoints"), (dict(name="Default"), ValueError, "snake_case"),
              (dict(cells="membrane"), ValueError, "membrane"), (dict(cells=3), TypeError, "segmentation run"),
              (dict(population="all"), ValueError, "population")]
    for kwargs, error, message in errors:
        with pytest.raises(error, match=message):
            fov.assign(**kwargs)


@pytest.mark.dataset
def test_molecule_table_agrees_with_export_spots(decoded_fov, tmp_path):
    from starfinder.io import export_spots
    fov, _ = decoded_fov
    path = export_spots(fov.spot_result, fov.filtering_result, tmp_path / "goodSpots.csv", accepted_only=True)
    from_csv = molecule_table_from_csv(path, spot_namespace=fov.spot_result.spot_namespace, genes=fov.codebook.genes)
    joined = molecule_table(fov.spot_result, fov.filtering_result, genes=fov.codebook.genes)
    assert len(joined.table) == len(from_csv.table) > 0
    for column in ("z", "y", "x"):
        assert np.allclose(joined.table[column], from_csv.table[column], rtol=0, atol=1e-9)
    assert joined.table.gene_id.tolist() == from_csv.table.gene_id.tolist()
    assert joined.source["reads"] == "ReadFilteringResult" and joined.source["population"] == "final"
    with pytest.raises(ValueError, match="ReadFilteringResult"):
        molecule_table(fov.spot_result, fov.decoding_result, genes=fov.codebook.genes)


@pytest.mark.dataset
def test_a_run_that_lists_expand_labels_is_refused_by_assign(decoded_fov, tmp_path):
    from starfinder.io import save_volume
    from starfinder.segmentation import (
        InputChannel,
        LabelImportConfig,
        SeededWatershedConfig,
        SegmentationPlan,
        SegmentationRun,
    )
    fov, _ = decoded_fov
    grid, _, nuclei = fov_masks(fov)
    save_volume(nuclei, tmp_path / "nuclei.tif", metadata=grid.metadata)
    expansion = ExpandLabelsConfig(0.2, "um", "planar")
    fov.segment(SegmentationPlan((
        SegmentationRun("nucleus", "nucleus", (), LabelImportConfig(str(tmp_path / "nuclei.tif"), "nucleus")),
        SegmentationRun("grown", "cell", (InputChannel("amplicon", reference_merged=True),),
                        SeededWatershedConfig(sigma_um=0.1), seeds="nucleus", operations=(expansion,)))))
    grown = fov.segmentation_results["grown"]
    (operation,) = grown.record["operations"]
    assert operation["operation"] == "expand_labels" and operation["config"] == {"distance": 0.2, "unit": "um",
                                                                                 "mode": "planar"}
    assert grown.record["labels"]["sha256"] == digest(grown.labels)
    with pytest.raises(ValueError, match="labels of run 'grown' were expanded by segmentation"):
        fov.assign(cells="grown", nuclei="nucleus")
    with pytest.raises(ValueError, match="takes no operations"):
        SegmentationRun("nucleus", "nucleus", (), LabelImportConfig("x.tif", "nucleus"), operations=(expansion,))


def test_assignment_result_is_frozen():
    result = assign_boxes()
    assert isinstance(result, AssignmentResult)
    with pytest.raises(AttributeError):
        result.genes = ()
