"""Vocabularies, the per-FOV input and the config of the assemble entry (design/specs/assembly.md, "Names")."""
from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from pathlib import Path

import numpy as np

from starfinder.assignment import AssignmentResult, MoleculeTable
from starfinder.segmentation import ReferenceGrid

#: Whether assembly retained or excluded an observation.
ASSEMBLY_STATUSES = ("retained", "excluded")
#: Reasons a cell observation is excluded from the sample.
CELL_ASSEMBLY_REASONS = ("not_owner", "fov_not_placed", "duplicate_observation")
#: Reasons a molecule observation is excluded from the sample.
MOLECULE_ASSEMBLY_REASONS = ("cell_not_owner", "not_owner", "outside_grid", "fov_not_placed",
                             "cell_duplicate_observation")
# The reasons of the duplicate rule, which the record counts only when the rule is on (duplicate_threshold not None).
DUPLICATE_REASONS = ("duplicate_observation", "cell_duplicate_observation")
#: Diagnostic flags attached to cell observations during assembly.
CELL_ASSEMBLY_FLAGS = ("fov_edge", "z_edge", "conflict", "unclaimed")
#: Diagnostic flags attached to molecule observations during assembly.
MOLECULE_ASSEMBLY_FLAGS = ("outside_owner_region", "cell_unclaimed")

OWNERSHIP = ("nearest_center", "midpoint_box")
CENTROIDS = ("territory", "original")
TIES = ("table_order", "half_open")


@dataclass(frozen=True)
class AssemblyConfig:
    """How :func:`assemble` decides ownership and which flags it computes.

    ``ownership`` is ``nearest_center`` (O1: the owning FOV whose footprint
    contains the point and whose footprint centre is nearest in Y and X) or
    ``midpoint_box`` (O2: per FOV an axis-aligned box whose edge towards each
    grid neighbour is the middle of their overlap; it needs grid cells).
    ``centroid`` is the cell centroid ownership tests: ``territory`` (the
    territory assign sampled: the expanded one when assign expanded) or
    ``original``. ``tie`` is ``table_order`` (T1: an exact tie goes to the FOV
    first in the placement's table order) or ``half_open`` (T2, only with
    ``midpoint_box``: a point on a box edge belongs to the box that starts
    there). ``conflicts`` computes the ``conflict`` and ``unclaimed`` flags,
    which read the territories. See design/specs/assembly-algorithms.md.

    Raises
    ------
    ValueError
        An unknown value of a field, ``conflicts`` that is not a bool, or
        ``tie="half_open"`` with ``ownership="nearest_center"``.
    """

    ownership: str = "nearest_center"
    centroid: str = "territory"
    tie: str = "table_order"
    conflicts: bool = True
    duplicate_threshold: float | None = 0.5

    def __post_init__(self):
        for name, allowed in (("ownership", OWNERSHIP), ("centroid", CENTROIDS), ("tie", TIES)):
            if getattr(self, name) not in allowed:
                raise ValueError(f"AssemblyConfig.{name} must be one of {list(allowed)}; got {getattr(self, name)!r}")
        if not isinstance(self.conflicts, (bool, np.bool_)):
            raise ValueError(f"AssemblyConfig.conflicts must be True or False; got {self.conflicts!r}")
        if self.tie == "half_open" and self.ownership != "midpoint_box":
            raise ValueError("AssemblyConfig.tie='half_open' (T2) needs ownership='midpoint_box'; "
                             "nearest_center ties go by table_order")
        threshold = self.duplicate_threshold
        if threshold is not None:
            if isinstance(threshold, (bool, np.bool_)) or not isinstance(threshold, Real) or \
                    not math.isfinite(float(threshold)) or not 0.0 < float(threshold) <= 1.0:
                raise ValueError("AssemblyConfig.duplicate_threshold must be None (the duplicate rule off) or a number "
                                 f"above 0 and at most 1; got {threshold!r}")
            object.__setattr__(self, "duplicate_threshold", float(threshold))


@dataclass(frozen=True)
class AssemblyInput:
    """One FOV's input of :func:`assemble`: its reference grid and one of two kinds.

    ``assignment`` is the FOV's :class:`~starfinder.assignment.AssignmentResult`
    or the path of its saved folder (``fovs/<fov_id>/assignment/<name>/``,
    loaded with the assignment loader, which checks every recorded SHA-256);
    ``molecules`` is a :class:`~starfinder.assignment.MoleculeTable` alone, for a
    sample without segmentation. :func:`assemble` checks that exactly one is
    given and that every input of a sample gives the same kind. ``grid`` is the
    FOV's reference grid (the molecule run's), which places its index
    coordinates.

    Raises
    ------
    TypeError
        An empty or non-string ``fov_id``, a ``grid`` that is not a
        ReferenceGrid, or ``assignment`` or ``molecules`` of another type.
    """
    # §2.9

    fov_id: str
    grid: ReferenceGrid
    assignment: AssignmentResult | str | Path | None = None
    molecules: MoleculeTable | None = None

    def __post_init__(self):
        if not isinstance(self.fov_id, str) or not self.fov_id:
            raise TypeError(f"fov_id must be a nonempty string; got {self.fov_id!r}")
        if not isinstance(self.grid, ReferenceGrid):
            raise TypeError(f"grid of FOV {self.fov_id!r} must be a ReferenceGrid")
        if self.assignment is not None and not isinstance(self.assignment, (AssignmentResult, str, Path)):
            raise TypeError(f"assignment of FOV {self.fov_id!r} must be an AssignmentResult, a folder path or None")
        if self.molecules is not None and not isinstance(self.molecules, MoleculeTable):
            raise TypeError(f"molecules of FOV {self.fov_id!r} must be a MoleculeTable or None")
