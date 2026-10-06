"""Assignment (§2.9): molecules to cells, nucleus–cell correspondence, compartments and counts.

:func:`assign_molecules`, the third of the three calls per FOV (``FOV.run``,
``FOV.segment``, ``FOV.assign``), places every molecule of a :class:`MoleculeTable`
in the territory of a cell :class:`~starfinder.segmentation.SegmentationResult` that
holds its sampled voxel (``floor(c + 0.5)`` per axis) and gives it one status of
``ASSIGNMENT_STATUSES``. An optional expansion is applied once, here, with
:func:`~starfinder.segmentation.expand_labels`, and both masks are kept. With
nuclei, :func:`match_nuclei` relates nuclei to cells by overlap and flags every
doubtful case; nuclear and cytoplasmic counts exist only where the correspondence
allows; and cells without a matched nucleus are excluded by default. The result is
an :class:`AssignmentResult` with the molecule, cell, count and nucleus tables;
nothing is read or written. See docs/assignment-contract.md and
docs/assignment-algorithms.md.
"""
from ._assign import AssignmentResult, assign_molecules, sample_labels
from ._config import (ASSIGNMENT_STATUSES, CELL_CORRESPONDENCE, CELL_STATUSES, COMPARTMENT_STATES, NUCLEUS_STATUSES,
                      AssignmentConfig, CorrespondenceConfig)
from ._correspondence import match_nuclei
from ._diagnostics import plot_assignment, summarize_assignment
from ._molecules import MoleculeTable, molecule_table, molecule_table_from_csv

__all__ = ["assign_molecules", "AssignmentResult", "AssignmentConfig", "CorrespondenceConfig",
           "MoleculeTable", "molecule_table", "molecule_table_from_csv", "sample_labels", "match_nuclei",
           "ASSIGNMENT_STATUSES", "CELL_STATUSES", "CELL_CORRESPONDENCE", "NUCLEUS_STATUSES",
           "COMPARTMENT_STATES", "summarize_assignment", "plot_assignment"]
