"""AS6 of design/validation/stitching-assembly.md: every count identity refused twice.

design/specs/assembly.md ("Count accounting"): "The identities are checked exactly by
``AssemblyResult.__post_init__`` and by the tests." Each case below changes one thing in a
valid result of ``own_states`` (``own_cases`` with counted cells whose compartments are
``available``, ``withheld``, ``no_nucleus`` and ``unavailable``), ``own_cases`` or
``status_cases`` (the placeholder FOV2 with an input), and asserts both refusals: the
validator (``__post_init__``) raises ``ValueError`` naming the identity, and the independent
checker (``check_accounting``, which evaluates identities 1 to 5 from the inputs and the
sample tables without the validator) fails. ``VALIDATOR_CANNOT_DECIDE`` lists the cases the
result's own tables cannot decide (they need the per-FOV inputs): there the validator accepts
the result and the checker alone refuses it.
"""
import copy
import dataclasses
from functools import lru_cache

import pandas as pd
import pytest

from starfinder.assembly import AssemblyConfig, AssemblyResult, assemble

from .assembly_fixtures import (check_accounting, golden_placement, golden_results, inputs_of, own_grids,
                                own_placement, own_results, own_states_results, placeholder_result)

pytestmark = [pytest.mark.stitching, pytest.mark.validation]


@lru_cache(maxsize=None)
def valid(fixture):
    """(result, inputs, placement) of a valid assembly; callers change copies only.

    ``own_states_rule_off`` is ``own_states`` assembled with the duplicate rule off, for the cases that change FOV1's
    cell 2 (A), which the default rule excludes as a second observation of FOV2's B (share 1.0; W-400).
    """
    config = AssemblyConfig(duplicate_threshold=None) if fixture.endswith("_rule_off") else AssemblyConfig()
    fixture = fixture.removesuffix("_rule_off")
    if fixture == "own_states":
        inputs, placement = own_states_results(), own_placement()
    elif fixture == "own_cases":
        inputs, placement = own_results(), own_placement()
    else:   # status_cases: FOV2 a placeholder with its own input
        inputs = {**golden_results(), "Position002": placeholder_result()}
        placement = golden_placement(statuses={"Position002": "placeholder"})
    grids = own_grids() if fixture != "status_cases" else None
    return assemble(placement, inputs_of(inputs, grids), config=config), inputs, placement


def row(frame, fov, key, value):
    """The index of the row of fov whose column key equals value."""
    hit = frame.index[(frame.fov_id == fov) & (frame[key].astype(object) == value)]
    assert len(hit) == 1, (fov, key, value)
    return hit[0]


def molecule(frame, fov, spot):
    return row(frame, fov, "spot_id", spot)


def set_rows(frame, index, **values):
    for column, value in values.items():
        frame.loc[index, column] = value


def swap(frame, a, b, columns=("assembly_status", "assembly_reason")):
    """Exchange the status and reason of two rows (the totals stay equal)."""
    columns = list(columns)
    first, second = frame.loc[a, columns].copy(), frame.loc[b, columns].copy()
    frame.loc[a, columns], frame.loc[b, columns] = second.to_numpy(), first.to_numpy()


def add_count(counts, cells, fov, cell, gene, compartment, n):
    namespace = cells[(cells.fov_id == fov) & (cells.cell_id == cell)].cell_namespace.iloc[0]
    extra = pd.DataFrame({"fov_id": [fov], "cell_namespace": [namespace], "cell_id": [cell], "gene_id": [gene],
                          "compartment": [compartment], "count": [n]}).astype(counts.dtypes.to_dict())
    return pd.concat([counts, extra], ignore_index=True)


def count_row(counts, fov, cell, gene, compartment):
    hit = counts.index[(counts.fov_id == fov) & (counts.cell_id == cell) & (counts.gene_id == gene)
                       & (counts.compartment == compartment)]
    assert len(hit) == 1
    return hit[0]


# --- The cases: (identity, fixture, change). A change receives copies of the four tables and the record and
# returns the changed ones as a dict.

def c_repeated_cell_key(t):
    set_rows(t["cells"], row(t["cells"], "FOV1", "cell_id", 2), cell_id=1)


def c_repeated_molecule_key(t):
    set_rows(t["molecules"], molecule(t["molecules"], "FOV1", "m1"), spot_id="m0")


def c_missing_cell_row(t):
    t["cells"] = t["cells"].drop(index=row(t["cells"], "FOV2", "cell_id", 4))


def c_missing_molecule_row(t):
    t["molecules"] = t["molecules"].drop(index=molecule(t["molecules"], "FOV1", "m5"))


def c_cell_status_outside_vocabulary(t):
    set_rows(t["cells"], row(t["cells"], "FOV1", "cell_id", 1), assembly_status="kept")


def c_molecule_status_outside_vocabulary(t):
    set_rows(t["molecules"], molecule(t["molecules"], "FOV1", "m4"), assembly_status="dropped")


def c_molecule_key_not_an_input(t):
    set_rows(t["molecules"], molecule(t["molecules"], "FOV1", "m4"), spot_id="m99")


def c_record_total_changed(t):
    t["record"]["counts"]["molecules"]["retained"] += 1


def c_record_reason_total_changed(t):
    t["record"]["counts"]["cells"]["excluded"]["not_owner"] += 1


def c_cell_moved_without_its_count(t):
    set_rows(t["cells"], row(t["cells"], "FOV1", "cell_id", 1), assembly_status="excluded",
             assembly_reason="not_owner")


def c_molecule_moved_without_its_count(t):
    set_rows(t["molecules"], molecule(t["molecules"], "FOV1", "m5"), assembly_status="retained",
             assembly_reason=pd.NA)


def c_assigned_status_differs_from_cell(t):
    m = t["molecules"]
    swap(m, molecule(m, "FOV2", "m2"), molecule(m, "FOV2", "m5"))   # FOV2 m2 retained, its cell E2 excluded


def c_cell_not_owner_reason_replaced(t):
    m = t["molecules"]
    swap(m, molecule(m, "FOV2", "m2"), molecule(m, "FOV2", "m6"), columns=("assembly_reason",))


def c_unassigned_retained_owner_elsewhere(t):
    m = t["molecules"]
    swap(m, molecule(m, "FOV2", "m6"), molecule(m, "FOV2", "m5"))   # m6's position is owned by FOV1


def c_outside_grid_retained(t):
    m = t["molecules"]
    swap(m, molecule(m, "FOV2", "m7"), molecule(m, "FOV2", "m5"))


def c_excluded_cell_molecule_leaves_its_cell(t):
    m = t["molecules"]
    swap(m, molecule(m, "FOV1", "m3"), molecule(m, "FOV1", "m5"))   # own_cases: m3 excluded_cell in retained N


def c_placeholder_owns_a_position(t):
    m = t["molecules"]
    set_rows(m, molecule(m, "Position002", "m2"), owner_fov_id="Position002")


def c_placeholder_owns_a_centroid(t):
    c = t["cells"]
    set_rows(c, row(c, "Position002", "cell_id", 1), owner_fov_id="Position002")   # was Position001


def c_count_rows_for_an_uncounted_cell(t):
    t["counts"] = add_count(t["counts"], t["cells"], "FOV2", 2, "C", "whole", 1)


def c_count_rows_for_a_retained_excluded_no_nucleus_cell(t):
    t["counts"] = add_count(t["counts"], t["cells"], "FOV1", 4, "A", "whole", 1)   # own_cases: N


def c_whole_count_changed_by_one(t):
    i = count_row(t["counts"], "FOV1", 2, "B", "whole")
    t["counts"].loc[i, "count"] += 1


def c_counted_cell_row_missing(t):
    t["counts"] = t["counts"].drop(index=count_row(t["counts"], "FOV2", 1, "A", "whole"))


def c_n_molecules_changed(t):
    i = row(t["cells"], "FOV2", "cell_id", 4)
    t["cells"].loc[i, "n_molecules"] += 1


def c_whole_ne_nucleus_plus_cytoplasm(t):
    i = count_row(t["counts"], "FOV1", 1, "A", "nucleus")
    t["counts"].loc[i, "count"] += 1


def c_compartment_row_for_a_withheld_cell(t):
    t["counts"] = add_count(t["counts"], t["cells"], "FOV1", 3, "C", "nucleus", 1)


def c_compartment_row_for_a_no_nucleus_cell(t):
    t["counts"] = add_count(t["counts"], t["cells"], "FOV1", 4, "A", "cytoplasm", 1)


def c_compartment_row_for_an_unavailable_cell(t):
    t["counts"] = add_count(t["counts"], t["cells"], "FOV2", 1, "A", "nucleus", 1)


def c_retained_assigned_molecule_without_compartment(t):
    set_rows(t["molecules"], molecule(t["molecules"], "FOV2", "m0"), compartment=pd.NA)   # the reviewer's case 1


def c_compartment_not_allowed_in_an_unavailable_cell(t):
    set_rows(t["molecules"], molecule(t["molecules"], "FOV2", "m0"), compartment="nucleus")


def c_compartment_not_allowed_in_an_available_cell(t):
    set_rows(t["molecules"], molecule(t["molecules"], "FOV1", "m0"), compartment="withheld")


def c_compartment_on_an_unassigned_molecule(t):
    set_rows(t["molecules"], molecule(t["molecules"], "FOV1", "m4"), compartment="nucleus")


def c_nucleus_and_cytoplasm_labels_swapped(t):
    c = t["counts"]                                                     # the reviewer's case 2
    a, b = count_row(c, "FOV1", 1, "A", "nucleus"), count_row(c, "FOV1", 2, "B", "cytoplasm")
    c.loc[a, "compartment"], c.loc[b, "compartment"] = "cytoplasm", "nucleus"


def c_duplicate_of_names_an_excluded_cell(t):
    set_rows(t["cells"], row(t["cells"], "FOV1", "cell_id", 2), duplicate_of="s1_FOV2_2")    # E2: not_owner


def c_duplicate_of_on_a_retained_cell(t):
    set_rows(t["cells"], row(t["cells"], "FOV1", "cell_id", 1), duplicate_of="s1_FOV2_1")


def c_cell_duplicate_observation_reason_replaced(t):
    m = t["molecules"]     # m1 of FOV1's A (excluded duplicate_observation) and m5 (excluded not_owner)
    swap(m, molecule(m, "FOV1", "m1"), molecule(m, "FOV1", "m5"), columns=("assembly_reason",))


CASES = {
    "repeated_cell_key": (1, "own_states", c_repeated_cell_key),
    "repeated_molecule_key": (1, "own_states", c_repeated_molecule_key),
    "missing_cell_row": (1, "own_states", c_missing_cell_row),
    "missing_molecule_row": (1, "own_states", c_missing_molecule_row),
    "cell_status_outside_vocabulary": (1, "own_states", c_cell_status_outside_vocabulary),
    "molecule_status_outside_vocabulary": (1, "own_states", c_molecule_status_outside_vocabulary),
    "molecule_key_not_an_input": (1, "own_states", c_molecule_key_not_an_input),
    "record_total_changed": (2, "own_states", c_record_total_changed),
    "record_reason_total_changed": (2, "own_states", c_record_reason_total_changed),
    "cell_moved_without_its_count": (2, "own_states", c_cell_moved_without_its_count),
    "molecule_moved_without_its_count": (2, "own_states", c_molecule_moved_without_its_count),
    "assigned_status_differs_from_cell": (3, "own_states", c_assigned_status_differs_from_cell),
    "cell_not_owner_reason_replaced": (3, "own_states", c_cell_not_owner_reason_replaced),
    "unassigned_retained_owner_elsewhere": (3, "own_states", c_unassigned_retained_owner_elsewhere),
    "outside_grid_retained": (3, "own_states", c_outside_grid_retained),
    "excluded_cell_molecule_leaves_its_cell": (3, "own_cases", c_excluded_cell_molecule_leaves_its_cell),
    "placeholder_owns_a_position": (3, "status_cases", c_placeholder_owns_a_position),
    "placeholder_owns_a_centroid": (3, "status_cases", c_placeholder_owns_a_centroid),
    "count_rows_for_an_uncounted_cell": (4, "own_states", c_count_rows_for_an_uncounted_cell),
    "count_rows_for_a_retained_excluded_no_nucleus_cell": (4, "own_cases",
                                                           c_count_rows_for_a_retained_excluded_no_nucleus_cell),
    "whole_count_changed_by_one": (4, "own_states_rule_off", c_whole_count_changed_by_one),
    "counted_cell_row_missing": (4, "own_states", c_counted_cell_row_missing),
    "n_molecules_changed": (4, "own_states", c_n_molecules_changed),
    "whole_ne_nucleus_plus_cytoplasm": (5, "own_states", c_whole_ne_nucleus_plus_cytoplasm),
    "compartment_row_for_a_withheld_cell": (5, "own_states", c_compartment_row_for_a_withheld_cell),
    "compartment_row_for_a_no_nucleus_cell": (5, "own_states", c_compartment_row_for_a_no_nucleus_cell),
    "compartment_row_for_an_unavailable_cell": (5, "own_states", c_compartment_row_for_an_unavailable_cell),
    "retained_assigned_molecule_without_compartment": (5, "own_states",
                                                       c_retained_assigned_molecule_without_compartment),
    "compartment_not_allowed_in_an_unavailable_cell": (5, "own_states",
                                                       c_compartment_not_allowed_in_an_unavailable_cell),
    "compartment_not_allowed_in_an_available_cell": (5, "own_states",
                                                     c_compartment_not_allowed_in_an_available_cell),
    "compartment_on_an_unassigned_molecule": (5, "own_states", c_compartment_on_an_unassigned_molecule),
    "nucleus_and_cytoplasm_labels_swapped": (5, "own_states_rule_off", c_nucleus_and_cytoplasm_labels_swapped),
    "duplicate_of_names_an_excluded_cell": (2, "own_cases", c_duplicate_of_names_an_excluded_cell),
    "duplicate_of_on_a_retained_cell": (2, "own_cases", c_duplicate_of_on_a_retained_cell),
    "cell_duplicate_observation_reason_replaced": (3, "own_cases", c_cell_duplicate_observation_reason_replaced),
}
# A key that is not an input's: only the inputs say which keys exist, so only the checker can refuse it.
VALIDATOR_CANNOT_DECIDE = {"molecule_key_not_an_input"}


def broken(name):
    """(the changed tables and record, the valid result, its inputs and placement) of one case."""
    identity, fixture, change = CASES[name]
    result, inputs, placement = valid(fixture)
    tables = {k: getattr(result, k).copy() for k in ("fovs", "cells", "molecules", "counts")}
    tables["record"] = copy.deepcopy(dict(result.record))
    change(tables)
    return tables, result, inputs, placement


def unchecked(result, tables):
    """An AssemblyResult holding the changed tables without running __post_init__ (for the checker)."""
    out = object.__new__(AssemblyResult)
    for field in dataclasses.fields(AssemblyResult):
        value = tables.get(field.name, getattr(result, field.name))
        object.__setattr__(out, field.name, value.reset_index(drop=True) if isinstance(value, pd.DataFrame)
                           else value)
    return out


def refusals(name):
    """(validator message or None, checker refused) for one case."""
    tables, result, inputs, placement = broken(name)
    try:
        dataclasses.replace(result, **tables)
        validator = None
    except ValueError as error:
        validator = str(error)
    try:
        check_accounting(unchecked(result, tables), inputs, placement)
        checker = False
    except AssertionError:
        checker = True
    return validator, checker


@pytest.mark.parametrize("fixture", ["own_states", "own_states_rule_off", "own_cases", "status_cases"])
def test_the_valid_results_pass_both(fixture):
    result, inputs, placement = valid(fixture)
    check_accounting(result, inputs, placement)
    dataclasses.replace(result)


def test_own_states_has_counted_cells_in_every_compartments_state():
    result, _, _ = valid("own_states")
    counted = result.counted_cells()
    assert set(counted.compartments) == {"available", "withheld", "no_nucleus", "unavailable"}


@pytest.mark.parametrize("name", list(CASES))
def test_as6_a_broken_result_is_refused(name):
    identity = CASES[name][0]
    tables, result, inputs, placement = broken(name)
    if name in VALIDATOR_CANNOT_DECIDE:
        dataclasses.replace(result, **tables)          # the result's own tables are consistent
    else:
        with pytest.raises(ValueError, match=f"identity {identity}:"):
            dataclasses.replace(result, **tables)
    with pytest.raises(AssertionError):
        check_accounting(unchecked(result, tables), inputs, placement)


def test_the_changes_leave_the_valid_results_alone():
    for name in CASES:
        broken(name)
    for fixture in ("own_states", "own_states_rule_off", "own_cases", "status_cases"):
        result, inputs, placement = valid(fixture)
        check_accounting(result, inputs, placement)
        dataclasses.replace(result)
