"""The §2.9 engineering validation of assignment: the A checks the task groups did not add (W-318).

Rows of the engineering validation design in docs/assignment-algorithms.md ("Checks"). Rows A1
to A20 are in the modules of their task groups; the section "Implemented checks" of that page
maps every row to its tests, and ``test_every_a_row_names_existing_tests`` keeps that map true.
This module adds A2 on ``assign_golden`` with every identity of docs/assignment-contract.md
("Count accounting", 1 to 5) recomputed from the tables, in 3D and 2D, without and with the
legacy expansion: the fixture without the gene-``Z`` molecule, its statuses from the matching
subset of the pinned ``SEG_LABELS`` (the unmodified fixture, with gene ``Z``, serves the
unknown-gene rejection and the legacy pins in ``test_assignment_golden.py``).

``assign_golden`` uses no random number.
"""
import pytest

from starfinder.assignment import AssignmentConfig, assign_molecules
from starfinder.segmentation import ExpandLabelsConfig

from .test_assignment import GOLDEN_GRID, GOLDEN_KEEP, check_accounting, golden_inputs
from .test_assignment_golden import MOLECULES, SEG_LABELS
from .test_segmentation_validation import check_rows

pytestmark = [pytest.mark.segmentation, pytest.mark.validation]


def test_every_a_row_names_existing_tests():
    check_rows("A", 20)


@pytest.mark.parametrize("expand", [False, True], ids=["no_expansion", "planar_pixel_4"])
@pytest.mark.parametrize("dimensions", ["3d", "2d"])
def test_a2_assign_golden_statuses_and_identities_1_to_5(dimensions, expand):
    mols, cells = golden_inputs(dimensions)
    assert len(mols) == len(MOLECULES) - 1 == 18 and "Z" not in set(mols.table.gene_id)
    config = AssignmentConfig(expansion=ExpandLabelsConfig(4, "pixel", "planar") if expand else None)
    result = assign_molecules(mols, cells, grid=GOLDEN_GRID, config=config)
    pinned = [SEG_LABELS[(dimensions, expand)][i] for i in GOLDEN_KEEP]
    assert result.molecules.assignment_status.tolist() == ["assigned" if c else "unassigned" for c in pinned]
    counts = result.record["counts"]
    n_assigned = sum(1 for c in pinned if c)
    assert (counts["assigned"], counts["unassigned"], counts["excluded_cell"], counts["outside_grid"]) == \
        (n_assigned, len(pinned) - n_assigned, 0, 0)
    check_accounting(result)
