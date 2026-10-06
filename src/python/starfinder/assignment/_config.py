"""The assignment configs and status vocabularies (docs/assignment-contract.md, "Names")."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Real

from starfinder.segmentation import ExpandLabelsConfig

#: What assign decided for one molecule.
ASSIGNMENT_STATUSES = ("assigned", "unassigned", "excluded_cell", "outside_grid")
#: Whether a cell is kept or excluded.
CELL_STATUSES = ("kept", "excluded_no_nucleus")
#: The correspondence status of a cell.
CELL_CORRESPONDENCE = ("matched", "ambiguous", "no_nucleus", "unavailable")
#: The status of a nucleus.
NUCLEUS_STATUSES = ("matched", "ambiguous", "no_cell")
#: Whether a cell's nuclear and cytoplasmic counts exist, and why not.
COMPARTMENT_STATES = ("available", "withheld", "no_nucleus", "unavailable")
#: The correspondence flags of a cell, in the order they are joined; the last three withhold compartments.
CORRESPONDENCE_FLAGS = ("several_nuclei", "ambiguous_nucleus", "nucleus_outside_cell", "foreign_nucleus")
WITHHOLDING_FLAGS = CORRESPONDENCE_FLAGS[1:]
SAMPLING_RULE = "floor(c + 0.5)"
EXCLUSION_REASON = "no_matched_nucleus"
EXCLUSION_RATIONALE = "possible cell residue; not a biological identity"


def _share(value, name, low, *, low_open=False):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) \
            or not (low < value if low_open else low <= value) or value >= 1:
        raise ValueError(f"{name} must be a number in [{low}, 1); got {value!r}")
    return float(value)


@dataclass(frozen=True)
class CorrespondenceConfig:
    """The overlap rule that relates nuclei to cells (option C2 of docs/assignment-contract.md).

    A nucleus is matched to the cell holding the largest share of its voxels when
    that share is strictly greater than ``match_fraction`` (in [0.5, 1); 0.5 is the
    smallest value that makes the match unique). A matched nucleus whose share
    outside its cell is strictly greater than ``outside_tolerance`` (in [0, 1);
    0.0 is exact containment) is flagged ``outside``. The default 0.1 is
    provisional, from one culture crop (W-320). Shares are voxel counts.

    Raises
    ------
    ValueError
        A value outside its range.
    """

    match_fraction: float = 0.5
    outside_tolerance: float = 0.1

    def __post_init__(self):
        object.__setattr__(self, "match_fraction", _share(self.match_fraction, "match_fraction", 0.5))
        object.__setattr__(self, "outside_tolerance", _share(self.outside_tolerance, "outside_tolerance", 0.0))


@dataclass(frozen=True)
class AssignmentConfig:
    """Settings of :func:`assign_molecules`.

    ``expansion`` is applied once, by assign, to the cell territories with
    :func:`~starfinder.segmentation.expand_labels`; both masks are kept. A
    ``pixel`` distance on a calibrated grid needs ``legacy_pixel_expansion=True``
    (the legacy adapter's ``dilation_distance``). ``correspondence`` sets the
    overlap rule. ``exclude_cells_without_nucleus`` excludes cells whose
    correspondence is ``no_nucleus``; None resolves to True when nuclei are given
    and to False without them (option X1).

    Raises
    ------
    TypeError
        expansion is neither an ExpandLabelsConfig nor None, correspondence is
        not a CorrespondenceConfig, or a flag is not a bool (or None).
    """

    expansion: ExpandLabelsConfig | None = None
    legacy_pixel_expansion: bool = False
    correspondence: CorrespondenceConfig = field(default_factory=CorrespondenceConfig)
    exclude_cells_without_nucleus: bool | None = None

    def __post_init__(self):
        if self.expansion is not None and not isinstance(self.expansion, ExpandLabelsConfig):
            raise TypeError("expansion must be an ExpandLabelsConfig or None")
        if not isinstance(self.legacy_pixel_expansion, bool):
            raise TypeError("legacy_pixel_expansion must be a bool")
        if not isinstance(self.correspondence, CorrespondenceConfig):
            raise TypeError("correspondence must be a CorrespondenceConfig")
        if self.exclude_cells_without_nucleus is not None and not isinstance(self.exclude_cells_without_nucleus, bool):
            raise TypeError("exclude_cells_without_nucleus must be a bool or None")
