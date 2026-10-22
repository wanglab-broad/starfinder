"""The assemble entry: sample coordinates, retention, flags, accounting and the sample tables.

docs/assembly-contract.md ("The assemble entry", "Sample coordinates", "Retention",
"Statuses and reasons", "Flags", "Count accounting", "Sample tables") and
docs/assembly-algorithms.md.
"""
from __future__ import annotations

import json
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from starfinder._layout import ASSIGNMENT_RECORD
from starfinder._records import json_sha256, software
from starfinder.assignment import AssignmentResult
from starfinder.assignment._assign import COMPARTMENTS, assignment_table_sha256
from starfinder.image import ImageMetadata, IncompatibleGeometryError
from starfinder.segmentation import ReferenceGrid
from starfinder.stitching import PlacementResult
from starfinder.stitching._placement import PLACED, grid_problems, same_frame_grid, table_sha256
from starfinder.stitching._positions import _names
from starfinder.stitching._review import require_approval

from ._config import (ASSEMBLY_STATUSES, CELL_ASSEMBLY_FLAGS, CELL_ASSEMBLY_REASONS, DUPLICATE_REASONS,
                      MOLECULE_ASSEMBLY_FLAGS, MOLECULE_ASSEMBLY_REASONS, AssemblyConfig, AssemblyInput)
from ._ownership import OWNING, OwnershipMap, ownership_map, recorded_grid

FORMAT_VERSION = 1
STAGE = "assembly"
AXES = ("z", "y", "x")

# The §2.9 tables as assign_molecules writes them (docs/assignment-contract.md), copied unchanged.
CELL_DTYPES_29 = {"cell_id": "UInt32", "status": "string", "exclusion_reason": "string", "size_voxels": "int64",
                  "expanded_size_voxels": "Int64", "size_physical": "float64", "expanded_size_physical": "float64",
                  "centroid_z": "float64", "centroid_y": "float64", "centroid_x": "float64",
                  "expanded_centroid_z": "float64", "expanded_centroid_y": "float64",
                  "expanded_centroid_x": "float64", "n_molecules": "int64", "n_nuclei": "Int64",
                  "correspondence": "string", "correspondence_flags": "string", "compartments": "string"}
MOLECULE_DTYPES_29 = {"spot_namespace": "string", "spot_id": "string", "z": "float64", "y": "float64",
                      "x": "float64", "gene_id": "string", "voxel_z": "Int64", "voxel_y": "Int64",
                      "voxel_x": "Int64", "assignment_status": "string", "cell_id": "UInt32",
                      "in_expansion": "boolean", "original_cell_id": "UInt32", "nucleus_id": "UInt32",
                      "compartment": "string"}
MOLECULE_ONLY_DTYPES = {"spot_namespace": "string", "spot_id": "string", "z": "float64", "y": "float64",
                        "x": "float64", "gene_id": "string", "assignment_status": "string"}
# The namespace of a molecule's cell, so that (cell_namespace, cell_id) names the cell in the sample (choice 7, CK1 of
# docs/sample-segmentation-contract.md); null for a molecule without a cell_id.
CELL_KEY = ["cell_namespace", "cell_id"]
CELL_KEY_DTYPES = {"cell_namespace": "string"}
CELL_DTYPES = {"fov_id": "string", "cell_namespace": "string", "cell_id": "UInt32", "obs_name": "string",
               **{c: d for c, d in CELL_DTYPES_29.items() if c != "cell_id"},
               **{f"sample_centroid_{a}": "float64" for a in AXES},
               **{f"sample_expanded_centroid_{a}": "float64" for a in AXES},
               "ownership_centroid": "string",
               **{f"sample_ownership_centroid_{a}": "float64" for a in AXES},
               "owner_fov_id": "string", "assembly_status": "string", "assembly_reason": "string",
               "assembly_flags": "string", "conflict_with": "string", "duplicate_of": "string"}
TAIL_DTYPES = {"sample_z": "float64", "sample_y": "float64", "sample_x": "float64", "owner_fov_id": "string",
               "assembly_status": "string", "assembly_reason": "string", "assembly_flags": "string"}
COUNT_DTYPES = {"fov_id": "string", "cell_namespace": "string", "cell_id": "UInt32", "gene_id": "string",
                "compartment": "string", "count": "int64"}
FOV_DTYPES = {"fov_id": "string", "placement_status": "string", "origin_z": "float64", "origin_y": "float64",
              "origin_x": "float64", "kind": "string", "footprint_y0": "float64", "footprint_y1": "float64",
              "footprint_x0": "float64", "footprint_x1": "float64", "centre_y": "float64", "centre_x": "float64",
              "owns_points": "bool", "n_cells": "int64", "n_cells_retained": "int64", "n_molecules": "int64",
              "n_molecules_retained": "int64"}
MODES = {"assignment": "cells", "molecules": "molecules_only"}
# The reason of a molecule of a cell excluded for a reason that is not fov_not_placed.
CELL_TO_MOLECULE_REASON = {"not_owner": "cell_not_owner", "duplicate_observation": "cell_duplicate_observation"}


def molecule_dtypes(mode: str) -> dict:
    """The sample molecule table's columns and dtypes in a mode (``cells`` or ``molecules_only``)."""
    return {"fov_id": "string", **(MOLECULE_DTYPES_29 if mode == "cells" else MOLECULE_ONLY_DTYPES),
            **CELL_KEY_DTYPES, **TAIL_DTYPES}


def fill_cell_namespace(molecules: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    """A sample molecule table saved without ``cell_namespace``, with the column filled from the cell table.

    A molecule with a ``cell_id`` takes the ``cell_namespace`` of the one cell row
    of its ``fov_id`` and ``cell_id`` (on the route per FOV a molecule's cell is a
    cell of its own FOV); a molecule without a ``cell_id``, or whose ``fov_id`` and
    ``cell_id`` name no cell row or more than one namespace, gets null; so every
    molecule of a table without ``cell_id`` (``molecules_only``) gets null. The
    column is inserted where :func:`molecule_dtypes` places it, before
    ``sample_z``. A table that has the column is returned unchanged.
    """
    if "cell_namespace" in molecules:
        return molecules
    has_cells = "cell_id" in molecules
    named = {}
    if has_cells:
        for fov, cell, namespace in zip(cells["fov_id"].astype(object), cells["cell_id"].astype("Int64"),
                                        cells["cell_namespace"].astype(object)):
            if not (pd.isna(fov) or pd.isna(cell) or pd.isna(namespace)):
                named.setdefault((fov, int(cell)), set()).add(namespace)
    cell_ids = molecules["cell_id"].astype("Int64") if has_cells else [None] * len(molecules)
    values = []
    for fov, cell in zip(molecules["fov_id"].astype(object), cell_ids):
        found = None if pd.isna(cell) else named.get((fov, int(cell)), set())
        values.append(next(iter(found)) if found is not None and len(found) == 1 else None)
    columns = list(molecules.columns)
    out = molecules.copy()
    out.insert(columns.index("sample_z") if "sample_z" in columns else len(columns), "cell_namespace",
               pd.array(values, dtype="string"))
    return out


def _empty(dtypes):
    return pd.DataFrame({c: pd.Series([], dtype=d) for c, d in dtypes.items()}, columns=list(dtypes))


def _frame(parts, dtypes):
    parts = [p for p in parts if len(p)]
    if not parts:
        return _empty(dtypes)
    frame = pd.concat([p[list(dtypes)].astype(dtypes) for p in parts], ignore_index=True)
    return frame.astype(dtypes).reset_index(drop=True)


def _flag_text(flags, vocabulary):
    return ";".join(f for f in vocabulary if f in flags)


def _split(text):
    return set() if pd.isna(text) or text == "" else set(str(text).split(";"))


# --- Count accounting -----------------------------------------------------------------------------------

def _fail(identity, what, key=None):
    raise ValueError(f"identity {identity}: {what}" + ("" if key is None else f"; first differing key {key}"))


def _first(keys):
    keys = sorted(keys, key=str)
    return keys[0] if keys else None


def _tally(frame, columns, weights=None):
    """{key tuple: count} over the rows of frame, each row weighing 1 or its ``weights`` value."""
    if not len(frame):
        return {}
    keys = frame[columns].astype(object)
    weight = np.ones(len(frame), np.int64) if weights is None else frame[weights].to_numpy(np.int64)
    grouped = keys.assign(_w=weight).groupby(columns, sort=False)["_w"].sum()
    return {tuple(int(v) if isinstance(v, np.integer) else v for v in (k if isinstance(k, tuple) else (k,))): int(n)
            for k, n in grouped.items()}


def _duplicates_on(record):
    """Whether the record's configuration has the duplicate rule on (a record without the field has it off)."""
    return (record.get("config") or {}).get("duplicate_threshold") is not None


def _counts_record(cells, molecules, fovs, flags_computed, duplicates=True):
    cell_flags = [_split(v) for v in cells["assembly_flags"]]
    molecule_flags = [_split(v) for v in molecules["assembly_flags"]]

    def population(frame, reasons, column):
        excluded = frame[frame["assembly_status"].eq("excluded").to_numpy(dtype=bool)]
        # The reasons of the duplicate rule are counted when it is on; with it off the record is as without the rule.
        reasons = [r for r in reasons if duplicates or r not in DUPLICATE_REASONS]
        return {"input": int(fovs[column].sum()),
                "retained": int(frame["assembly_status"].eq("retained").sum()),
                "excluded": {r: int(excluded["assembly_reason"].eq(r).sum()) for r in reasons}}

    flags = {}
    for flag, rows in (*((f, cell_flags) for f in CELL_ASSEMBLY_FLAGS),
                       *((f, molecule_flags) for f in MOLECULE_ASSEMBLY_FLAGS)):
        flags[flag] = sum(flag in r for r in rows) if flag in flags_computed else None
    return {"cells": population(cells, CELL_ASSEMBLY_REASONS, "n_cells"),
            "molecules": population(molecules, MOLECULE_ASSEMBLY_REASONS, "n_molecules"), "flags": flags}


def check_identities(result: AssemblyResult) -> None:
    """The identities 1 to 5 of docs/assembly-contract.md ("Count accounting") on a result.

    Cells are keyed by the cell key ``(cell_namespace, cell_id)``, and a molecule
    links to its cell, and counts in it, by the same key in its own columns.
    Raises ValueError naming the identity and the first differing key.
    """
    fovs, cells, molecules, counts = result.fovs, result.cells, result.molecules, result.counts
    record = result.record
    # 1: one assembly status each; one row per input row (the fovs table records the input sizes).
    for name, frame, key in (("cell", cells, ["cell_namespace", "cell_id"]),
                             ("molecule", molecules, ["spot_namespace", "spot_id"])):
        repeated = frame[frame.duplicated(key, keep=False)]
        if len(repeated):
            _fail(1, f"a {name} key appears more than once", tuple(repeated[key].iloc[0]))
        status = frame["assembly_status"].astype(object)
        bad = frame[status.isna().to_numpy(dtype=bool) | ~status.isin(ASSEMBLY_STATUSES).to_numpy(dtype=bool)]
        if len(bad):
            _fail(1, f"every {name} row has one status of {list(ASSEMBLY_STATUSES)}", tuple(bad[key].iloc[0]))
        per_fov = frame["fov_id"].value_counts().to_dict()
        column = "n_cells" if name == "cell" else "n_molecules"
        differ = [f for f, n in zip(fovs["fov_id"], fovs[column]) if per_fov.get(f, 0) != int(n)]
        differ += sorted(set(per_fov) - set(fovs["fov_id"]))
        if differ:
            _fail(1, f"the sample {name} table holds every input row exactly once (fovs.{column})", differ[0])
    # 1 on the route after fusion: one cell row per label of the cell run, as the sample assignment counted them.
    after_fusion = record.get("route") == "after_fusion" and record.get("mode") == "cells"
    if after_fusion:
        labelled = ((record.get("assignment") or {}).get("counts") or {}).get("cells")
        if labelled != len(cells):
            _fail(1, f"the sample cell table holds one row per label of the cell run ({labelled} labels, "
                     f"{len(cells)} rows)")
    # 2: the populations sum to the inputs, per FOV and in total; reasons belong to their status.
    for name, frame, reasons, key in (("cell", cells, CELL_ASSEMBLY_REASONS, ["cell_namespace", "cell_id"]),
                                      ("molecule", molecules, MOLECULE_ASSEMBLY_REASONS,
                                       ["spot_namespace", "spot_id"])):
        status = frame["assembly_status"].astype(object).to_numpy()
        reason = frame["assembly_reason"].astype(object).to_numpy()
        wrong = [i for i, (s, r) in enumerate(zip(status, reason))
                 if (s == "retained") != pd.isna(r) or (s == "excluded" and r not in reasons)]
        if wrong:
            _fail(2, f"a retained {name} has no reason and an excluded one a reason of {list(reasons)}",
                  tuple(frame[key].iloc[wrong[0]]))
        retained = frame[status == "retained"]["fov_id"].value_counts().to_dict()
        column = "n_cells_retained" if name == "cell" else "n_molecules_retained"
        differ = [f for f, n in zip(fovs["fov_id"], fovs[column]) if retained.get(f, 0) != int(n)]
        if differ:
            _fail(2, f"fovs.{column} equals the retained {name}s per FOV", differ[0])
    # 2 (continued): a cell excluded duplicate_observation names a retained cell of another FOV, and no other names one.
    if "duplicate_of" in cells:
        named = cells["duplicate_of"].astype(object)
        has = named.notna().to_numpy(dtype=bool)
        duplicate = cells["assembly_reason"].eq("duplicate_observation").fillna(False).to_numpy(dtype=bool)
        state = dict(zip(cells["obs_name"].astype(object), zip(cells["fov_id"].astype(object),
                                                               cells["assembly_status"].astype(object))))
        wrong = [i for i in np.flatnonzero(has | duplicate)
                 if not (has[i] and duplicate[i]) or state.get(named.iloc[i], (None, None))[1] != "retained"
                 or state[named.iloc[i]][0] == cells["fov_id"].iloc[i]]
        if wrong:
            _fail(2, "a cell excluded duplicate_observation, and only such a cell, names in duplicate_of a retained "
                     "cell of another FOV", tuple(cells[["cell_namespace", "cell_id"]].iloc[wrong[0]]))
    expected = _counts_record(cells, molecules, fovs, set(record.get("flags", {}).get("computed", [])),
                              _duplicates_on(record))
    recorded = record.get("counts")
    if recorded is not None:
        for population in ("cells", "molecules"):
            if recorded.get(population) != expected[population]:
                _fail(2, f"record['counts'][{population!r}] equals the {population} table's totals", population)
            got = expected[population]
            if got["input"] != got["retained"] + sum(got["excluded"].values()):
                _fail(2, f"the {population} sum to the inputs", population)
        if recorded.get("flags") != expected["flags"]:
            _fail(2, "record['counts']['flags'] equals the tables' flag counts", "flags")
    # 3: molecules follow their cell, the cell of their cell key (cell_namespace, cell_id); unassigned ones their own
    # position.
    cell_mode = "cell_id" in molecules
    if "cell_namespace" not in molecules:
        _fail(3, "a molecule names its cell by the cell key (cell_namespace, cell_id), but the molecule table has no "
                 "column cell_namespace")
    named = molecules["cell_namespace"].notna().to_numpy(dtype=bool)
    has_cell = molecules["cell_id"].notna().to_numpy(dtype=bool) if cell_mode else np.zeros(len(molecules), bool)
    wrong = np.flatnonzero(named != has_cell)
    if len(wrong):
        _fail(3, "a molecule has a cell_namespace exactly when it has a cell_id",
              tuple(molecules[["spot_namespace", "spot_id"]].iloc[int(wrong[0])]))
    placed = molecules["fov_id"].map(dict(zip(fovs["fov_id"], fovs["placement_status"].isin(PLACED))))
    placed = placed.fillna(False).to_numpy(dtype=bool)
    a_status = molecules["assignment_status"].astype(object).fillna("").to_numpy()
    status = molecules["assembly_status"].astype(object).to_numpy()
    reason = molecules["assembly_reason"].astype(object).fillna("").to_numpy()
    own = (molecules["owner_fov_id"].astype(object) == molecules["fov_id"].astype(object)).to_numpy(dtype=bool)
    in_cell = np.isin(a_status, ("assigned", "excluded_cell"))
    want_status = np.where(own, "retained", "excluded").astype(object)
    want_reason = np.where(own, "", "not_owner").astype(object)
    # A molecules_only row outside its grid has no §2.9 status and the reason outside_grid.
    no_status = pd.isna(molecules["assignment_status"]).to_numpy(dtype=bool) & (reason == "outside_grid")
    want_status[no_status], want_reason[no_status] = "excluded", "outside_grid"
    outside = a_status == "outside_grid"
    want_status[outside], want_reason[outside] = "excluded", "outside_grid"
    if in_cell.any():
        cell_state = pd.DataFrame({"cell_namespace": cells["cell_namespace"].astype(object),
                                   "cell_id": cells["cell_id"].astype("int64"),
                                   "_status": cells["assembly_status"].astype(object),
                                   "_reason": cells["assembly_reason"].astype(object).fillna("")})
        rows = pd.DataFrame({"cell_namespace": molecules["cell_namespace"].astype(object)[in_cell],
                             "cell_id": molecules["cell_id"].astype("Int64")[in_cell].astype("int64")})
        joined = rows.merge(cell_state, on=CELL_KEY, how="left")
        missing = joined["_status"].isna().to_numpy() | rows["cell_namespace"].isna().to_numpy()
        if missing.any():
            first = int(np.flatnonzero(in_cell)[np.flatnonzero(missing)[0]])
            _fail(3, "every assigned or excluded_cell molecule's cell is a row of the cell table",
                  tuple(molecules[["spot_namespace", "spot_id"]].iloc[first]))
        # On the route per FOV a molecule follows its cell; after fusion it keeps its own position's status.
        if not after_fusion:
            cell_reason = joined["_reason"].to_numpy(dtype=object)
            want_status[in_cell] = joined["_status"].to_numpy(dtype=object)
            want_reason[in_cell] = np.array([CELL_TO_MOLECULE_REASON.get(r, r) for r in cell_reason], dtype=object)
    if after_fusion:
        # Only the retained molecules are assigned; an excluded one has no sample assignment.
        assigned_excluded = (status == "excluded") & pd.notna(molecules["assignment_status"]).to_numpy(dtype=bool)
        if assigned_excluded.any():
            _fail(3, "on the route after fusion only retained molecules have a sample assignment",
                  tuple(molecules[["spot_namespace", "spot_id"]].iloc[int(np.flatnonzero(assigned_excluded)[0])]))
    want_status[~placed], want_reason[~placed] = "excluded", "fov_not_placed"
    wrong = np.flatnonzero((status != want_status) | (reason != want_reason))
    if len(wrong):
        _fail(3, "a molecule's status follows its cell, its own position or outside_grid",
              tuple(molecules[["spot_namespace", "spot_id"]].iloc[int(wrong[0])]))
    # 3 (continued): no position or centroid is owned by a FOV that owns no point (a placeholder, an unplaced FOV).
    owning = set(fovs["fov_id"][fovs["owns_points"].to_numpy(dtype=bool)])
    for name, frame, key in (("cell", cells, ["cell_namespace", "cell_id"]),
                             ("molecule", molecules, ["spot_namespace", "spot_id"])):
        owner = frame["owner_fov_id"].astype(object)
        foreign = (owner.notna() & ~owner.isin(owning)).to_numpy(dtype=bool)
        if foreign.any():
            _fail(3, f"a {name}'s owner_fov_id is an owning FOV (owns_points), never a placeholder",
                  tuple(frame[key].iloc[int(np.flatnonzero(foreign)[0])]))
    # 4: the counted cells' counts equal their retained assigned molecules; no other cell has rows.
    count_key = ["fov_id", "cell_namespace", "cell_id", "gene_id", "compartment"]
    repeated = counts[counts.duplicated(count_key, keep=False)]
    if len(repeated):
        _fail(4, "one count row per (cell, gene, compartment)", tuple(repeated[count_key].iloc[0]))
    compartment = counts["compartment"].astype(object)
    bad = (~compartment.isin(COMPARTMENTS) | ~counts["gene_id"].astype(object).isin(result.genes)
           | (counts["count"].to_numpy() <= 0)).to_numpy(dtype=bool)
    if bad.any():
        _fail(4, f"every count row has a compartment of {list(COMPARTMENTS)}, a gene of genes and a count above 0",
              tuple(counts[count_key].iloc[int(np.flatnonzero(bad)[0])]))
    counted = cells[(cells["assembly_status"].eq("retained") & cells["status"].eq("kept")).to_numpy(dtype=bool)]
    counted_keys = {(f, ns, int(c)) for f, ns, c in zip(counted["fov_id"], counted["cell_namespace"],
                                                        counted["cell_id"].astype("int64"))}
    count_keys = {(f, ns, int(c)) for f, ns, c in zip(counts["fov_id"], counts["cell_namespace"],
                                                      counts["cell_id"].astype("int64"))}
    if not count_keys <= counted_keys:
        _fail(4, "only counted cells (retained, §2.9 kept) have count rows", _first(count_keys - counted_keys))
    # The counts and the molecules meet at the cell key (cell_namespace, cell_id), so a molecule counts in its cell
    # whichever FOV observed it.
    tally_key = ["cell_namespace", "cell_id", "gene_id"]
    whole = counts[compartment.eq("whole").to_numpy(dtype=bool)]
    whole_tally = _tally(whole.assign(cell_id=whole["cell_id"].astype("int64")), tally_key, "count")
    if cell_mode:
        assigned = molecules[(molecules["assignment_status"].eq("assigned")
                              & molecules["assembly_status"].eq("retained")).to_numpy(dtype=bool)]
        assigned = assigned.assign(cell_id=assigned["cell_id"].astype("int64"))
        molecule_tally = _tally(assigned, tally_key)
    else:
        assigned = molecules.iloc[:0]
        molecule_tally = {}
    if whole_tally != molecule_tally:
        _fail(4, "the whole counts are the retained assigned molecules per cell and gene",
              _first(set(whole_tally.items()) ^ set(molecule_tally.items())))
    per_cell = {}
    for (ns, c, _), n in whole_tally.items():
        per_cell[(ns, c)] = per_cell.get((ns, c), 0) + n
    n_molecules = {(ns, int(c)): int(n) for ns, c, n in zip(counted["cell_namespace"],
                                                            counted["cell_id"].astype("int64"),
                                                            counted["n_molecules"]) if n}
    if per_cell != n_molecules:
        _fail(4, "the whole counts sum to n_molecules over the counted cells",
              _first(set(per_cell.items()) ^ set(n_molecules.items())))
    # 5: the §2.9 compartment identities over the counted cells and the retained assigned molecules.
    state = {(ns, int(c)): a for ns, c, a in zip(counted["cell_namespace"], counted["cell_id"].astype("int64"),
                                                 counted["compartments"].astype(object))}
    available = {k for k, a in state.items() if a == "available"}
    parts = counts[compartment.isin(("nucleus", "cytoplasm")).to_numpy(dtype=bool)]
    part_keys = {(ns, int(c)) for ns, c in zip(parts["cell_namespace"], parts["cell_id"].astype("int64"))}
    if not part_keys <= available:
        _fail(5, "only counted cells whose compartments are available have nucleus and cytoplasm rows",
              _first(part_keys - available))
    split = _tally(parts.assign(cell_id=parts["cell_id"].astype("int64")), tally_key, "count")
    expected_split = {k: v for k, v in whole_tally.items() if (k[0], k[1]) in available}
    if split != expected_split:
        _fail(5, "nucleus + cytoplasm = whole per available counted cell and gene",
              _first(set(split.items()) ^ set(expected_split.items())))
    if not cell_mode:
        return
    # Per molecule: a compartment exactly on the assigned molecules; nucleus or cytoplasm in an available cell,
    # else the cell's state; so the compartment populations sum to the retained assigned molecules.
    keys = molecules[["spot_namespace", "spot_id"]]
    has = molecules["compartment"].notna().to_numpy(dtype=bool)
    is_assigned = molecules["assignment_status"].eq("assigned").fillna(False).to_numpy(dtype=bool)
    if (has != is_assigned).any():
        _fail(5, "every assigned molecule, and only those, has a compartment",
              tuple(keys.iloc[int(np.flatnonzero(has != is_assigned)[0])]))
    cell_state = pd.Series([state.get((ns, int(c))) for ns, c in zip(assigned["cell_namespace"],
                                                                     assigned["cell_id"])], dtype=object)
    value = assigned["compartment"].astype(object).reset_index(drop=True)
    allowed = ((cell_state == "available") & value.isin(("nucleus", "cytoplasm"))) | (
        (cell_state != "available") & (value == cell_state))
    if not allowed.all():
        i = int(np.flatnonzero(~allowed.to_numpy(dtype=bool))[0])
        _fail(5, f"a retained assigned molecule's compartment is nucleus or cytoplasm in an available cell, else "
                 f"the cell's compartments state ({cell_state[i]}); got {value[i]}",
              tuple(assigned[["spot_namespace", "spot_id"]].iloc[i]))
    for name in ("nucleus", "cytoplasm"):
        rows = parts[parts["compartment"].eq(name).to_numpy(dtype=bool)]
        given = _tally(rows.assign(cell_id=rows["cell_id"].astype("int64")), tally_key, "count")
        held = _tally(assigned[assigned["compartment"].eq(name).to_numpy(dtype=bool)], tally_key)
        if given != held:
            _fail(5, f"the {name} counts are the retained assigned molecules whose compartment is {name}",
                  _first(set(given.items()) ^ set(held.items())))


@dataclass(frozen=True, eq=False)
class AssemblyResult:
    """The assembled sample: the four sample tables, the gene list and the record.

    ``fovs`` has one row per FOV of the placement; ``cells`` one row per cell of
    every input FOV, in placement table order then increasing ``cell_id``;
    ``molecules`` one row per molecule of every input FOV, in placement table
    order then input order, naming its cell by ``(cell_namespace, cell_id)``
    (null without a cell); ``counts`` the §2.9 long counts of the counted
    cells (retained, §2.9 ``kept``) with ``fov_id`` and ``cell_namespace``.
    ``record`` is the ``assembly`` section of ``export.json``. See
    docs/assembly-contract.md ("Sample tables", "What assemble records").

    Raises
    ------
    TypeError
        A table that is not a DataFrame or a record that is not a mapping.
    ValueError
        An identity of "Count accounting" fails, naming it and the first
        differing key.
    """

    sample: str
    fovs: pd.DataFrame
    cells: pd.DataFrame
    molecules: pd.DataFrame
    counts: pd.DataFrame
    genes: tuple[str, ...]
    record: Mapping[str, Any]

    def __post_init__(self):
        for name in ("fovs", "cells", "molecules", "counts"):
            frame = getattr(self, name)
            if not isinstance(frame, pd.DataFrame):
                raise TypeError(f"{name} must be a pandas DataFrame")
            object.__setattr__(self, name, frame.reset_index(drop=True))
        if not isinstance(self.record, Mapping):
            raise TypeError("record must be a mapping")
        object.__setattr__(self, "genes", tuple(self.genes))
        check_identities(self)

    @property
    def mode(self) -> str:
        """``cells`` or ``molecules_only``."""
        return self.record.get("mode", "cells")

    def counted_cells(self) -> pd.DataFrame:
        """The cell-table rows of the counted cells (retained, §2.9 ``kept``), in table order."""
        cells = self.cells
        return cells[(cells["assembly_status"].eq("retained") & cells["status"].eq("kept")).to_numpy(dtype=bool)
                     ].reset_index(drop=True)

    def matrix(self, compartment: str = "whole") -> tuple[np.ndarray, pd.DataFrame]:
        """A dense counted cells × genes ``int64`` matrix in ``genes`` order, with its cell-table rows.

        ``compartment`` is ``whole`` (every counted cell) or ``nucleus`` or
        ``cytoplasm`` (the counted cells whose compartments are ``available``).

        Raises
        ------
        ValueError
            An unknown compartment.
        """
        if compartment not in COMPARTMENTS:
            raise ValueError(f"compartment must be whole, nucleus or cytoplasm; got {compartment!r}")
        rows = self.counted_cells()
        if compartment != "whole":
            rows = rows[rows["compartments"].eq("available").to_numpy(dtype=bool)].reset_index(drop=True)
        row_of = {(ns, int(c)): i for i, (ns, c) in enumerate(zip(rows["cell_namespace"], rows["cell_id"]))}
        gene_of = {g: j for j, g in enumerate(self.genes)}
        matrix = np.zeros((len(rows), len(self.genes)), np.int64)
        counts = self.counts[self.counts["compartment"].eq(compartment).to_numpy(dtype=bool)]
        for ns, c, g, n in zip(counts["cell_namespace"], counts["cell_id"], counts["gene_id"], counts["count"]):
            matrix[row_of[(ns, int(c))], gene_of[g]] += int(n)
        return matrix, rows

    def __repr__(self):
        counts = self.record.get("counts", {})
        cells, molecules = counts.get("cells", {}), counts.get("molecules", {})
        return (f"AssemblyResult: sample {self.sample!r} ({self.mode}), {len(self.fovs)} FOVs, "
                f"{cells.get('retained')} of {len(self.cells)} cells and {molecules.get('retained')} of "
                f"{len(self.molecules)} molecules retained")


# --- The entry ------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class _Saved:
    """A saved assignment as assemble keeps it: its tables and record, without its label images.

    Its territory image is read again from the folder when it is needed
    (:func:`_territory`), so at most two territory images are in memory at a
    time (docs/assembly-algorithms.md, "Flags").
    """
    molecules: pd.DataFrame
    cells: pd.DataFrame
    counts: pd.DataFrame
    record: Mapping[str, Any]
    cell_namespace: str
    genes: tuple
    expanded: bool
    nuclei: bool


@dataclass
class _Fov:
    """One input FOV as assemble reads it."""
    fov_id: str
    grid: Any
    kind: str
    result: AssignmentResult | _Saved | None = None
    molecules: Any = None
    folder: Path | None = None
    name: str | None = None


def _load(item: AssemblyInput) -> _Fov:
    if item.assignment is None:
        return _Fov(item.fov_id, item.grid, "molecules", molecules=item.molecules)
    if isinstance(item.assignment, AssignmentResult):
        return _Fov(item.fov_id, item.grid, "assignment", result=item.assignment)
    from starfinder.assignment._persist import read_assignment
    folder = Path(os.path.abspath(item.assignment))
    loaded = read_assignment(folder, folder.name)
    saved = _Saved(loaded.molecules, loaded.cells, loaded.counts, loaded.record, loaded.cell_namespace, loaded.genes,
                   loaded.territories is not None, loaded.nucleus_labels is not None)
    return _Fov(item.fov_id, item.grid, "assignment", result=saved, folder=folder, name=folder.name)


def _expanded(fov: _Fov) -> bool:
    """Whether assign expanded the FOV's cells (its territories are then the expanded labels)."""
    result = fov.result
    return result.expanded if isinstance(result, _Saved) else result.territories is not None


def _territory(fov: _Fov) -> np.ndarray:
    """The territory image assign sampled: the expanded labels when it expanded, else the cell labels.

    In memory for an AssignmentResult; read again from a saved folder, with
    the SHA-256 of its labels checked as the §2.9 loader checks it.
    """
    result = fov.result
    if not isinstance(result, _Saved):
        return result.cell_labels if result.territories is None else result.territories
    from starfinder.assignment._persist import _labels
    record = result.record
    if result.expanded:
        return _labels(fov.folder, record["files"]["territories"], record["expansion"]["expanded"]["sha256"],
                       "territory file")
    return _labels(fov.folder, record["inputs"]["cells"]["file"], record["inputs"]["cells"]["labels_sha256"],
                   "cell label file")


def _identity(namespace, what):
    try:
        value = json.loads(namespace)
    except (TypeError, ValueError):
        value = None
    if not isinstance(value, list) or len(value) < 4:
        raise ValueError(f"{what} {namespace!r} is not a JSON list starting [dataset_id, sample_id, fov_id, "
                         "subtile_id]")
    return value[:4]


def _check_inputs(placement, inputs, config):
    """Checks 1 to 8 of docs/assembly-contract.md ("Checks the entry applies"), in order."""
    if not isinstance(config, AssemblyConfig):
        raise TypeError("config must be an AssemblyConfig")
    config.__post_init__()
    if not isinstance(placement, PlacementResult):
        raise TypeError("placement must be a PlacementResult")
    if isinstance(inputs, (str, bytes)) or not isinstance(inputs, Sequence):
        raise TypeError("inputs must be a sequence of AssemblyInput")
    inputs = list(inputs)
    wrong = [i for i, item in enumerate(inputs) if not isinstance(item, AssemblyInput)]
    if wrong:
        raise TypeError(f"inputs must be AssemblyInput; items {_names(wrong)} are not")
    # 2. Review.
    require_approval(placement)
    # 3. FOV identity.
    ids = [item.fov_id for item in inputs]
    repeated = sorted(k for k, n in Counter(ids).items() if n > 1)
    if repeated:
        raise ValueError(f"each FOV has at most one input; repeated: {_names(repeated)}")
    statuses = dict(zip(placement.fovs["fov_id"], placement.fovs["status"]))
    unknown = [fov for fov in ids if fov not in statuses]
    if unknown:
        raise ValueError(f"inputs name FOVs that are not rows of the placement of sample {placement.sample!r}: "
                         f"{_names(unknown)}")
    without = [fov for fov, status in statuses.items() if status in OWNING and fov not in ids]
    if without:
        raise ValueError(f"every image_supported or metadata_only FOV needs an input; FOVs without one: "
                         f"{_names(without)}")
    # 4. Kind.
    both = [item.fov_id for item in inputs if (item.assignment is None) == (item.molecules is None)]
    if both:
        raise ValueError(f"each input gives exactly one of assignment and molecules; FOVs {_names(both)} do not")
    kinds = {}
    for item in inputs:
        kinds.setdefault("assignment" if item.assignment is not None else "molecules", []).append(item.fov_id)
    if len(kinds) > 1:
        raise ValueError("every input of a sample gives the same kind; " +
                         "; ".join(f"{kind}: {_names(fovs)}" for kind, fovs in kinds.items()))
    # 5. Observation identity (a saved assignment is loaded here, checking its SHA-256).
    loaded = [_load(item) for item in inputs]
    samples, problems = {}, []
    for fov in loaded:
        if fov.kind == "assignment":
            spaces = [("spot_namespace", fov.result.record["inputs"]["molecules"]["source"]["spot_namespace"]),
                      ("cell_namespace", fov.result.cell_namespace)]
        else:
            spaces = [("spot_namespace", fov.molecules.spot_namespace)]
        for what, namespace in spaces:
            dataset, sample, fov_id, subtile = _identity(namespace, f"{what} of FOV {fov.fov_id}")
            if fov_id != fov.fov_id:
                problems.append(f"{fov.fov_id} ({what} names FOV {fov_id!r})")
            elif subtile is not None:
                problems.append(f"{fov.fov_id} ({what} names subtile {subtile!r}; subtiles are not assembled)")
            samples.setdefault((dataset, sample), []).append(fov.fov_id)
    if problems:
        raise ValueError(f"observation identity: {'; '.join(problems)}")
    if len(samples) > 1:
        raise ValueError("the inputs of a sample share one dataset_id and sample_id; " +
                         "; ".join(f"{list(key)}: {_names(sorted(set(f)))}" for key, f in samples.items()))
    # 6. Grid.
    for fov in loaded:
        recorded = recorded_grid(placement, fov.fov_id)
        if recorded is None or not same_frame_grid(recorded[0], recorded[1], fov.grid.shape_zyx, fov.grid.metadata):
            raise IncompatibleGeometryError(
                f"the grid of FOV {fov.fov_id} ({fov.grid.shape_zyx}, {fov.grid.metadata}) is not the grid the "
                f"placement recorded ({None if recorded is None else (recorded[0], recorded[1])})")
        if fov.kind == "assignment":
            grid = fov.result.record["grid"]
            assigned = ReferenceGrid(tuple(grid["shape_zyx"]), ImageMetadata(**grid["metadata"]), "declared")
            if not same_frame_grid(assigned.shape_zyx, assigned.metadata, fov.grid.shape_zyx, fov.grid.metadata):
                raise IncompatibleGeometryError(
                    f"the grid of FOV {fov.fov_id} ({fov.grid.shape_zyx}, {fov.grid.metadata}) is not the grid of "
                    f"its assignment record ({assigned.shape_zyx}, {assigned.metadata})")
    bad = {"unknown spacing": [], "a unit other than µm": [], "a direction other than the identity": []}
    for fov in loaded:
        for what in grid_problems(fov.grid.metadata):
            bad[what].append(fov.fov_id)
    found = [f"{what}: {_names(fovs)}" for what, fovs in bad.items() if fovs]
    if found:
        raise IncompatibleGeometryError("assemble needs known spacing, unit µm and identity direction on every "
                                        f"grid; FOVs with {'; '.join(found)}")
    # 7. Genes.
    genes = {}
    for fov in loaded:
        own = fov.result.genes if fov.kind == "assignment" else fov.molecules.genes
        genes.setdefault(tuple(own), []).append(fov.fov_id)
    if len(genes) > 1:
        lists = list(genes.items())
        first = lists[0][0]
        described = "; ".join(f"{_names(fovs)}: {list(key)}" + ("" if key == first else
                              f" (differs by {sorted(set(key) ^ set(first))})") for key, fovs in lists)
        raise ValueError(f"every input has the same genes; {described}")
    # 8. One layer.
    placed = placement.fovs[placement.fovs["status"].isin(PLACED).to_numpy(dtype=bool)]
    layers = {}
    for fov, z in zip(placed["fov_id"], placed["nominal_z"]):
        layers.setdefault(float(z), []).append(fov)
    if len(layers) > 1:
        raise ValueError("assemble handles one layer of tiles; FOVs " +
                         "; ".join(f"{_names(fovs)} have nominal z {z!r}" for z, fovs in layers.items()))
    return loaded, MODES[next(iter(kinds))] if kinds else "cells", (next(iter(genes)) if genes else ())


def _sample_territory(labels, plane, origin, spacing, points):
    """The territory value at sample points by the §2.9 rule in another FOV; 0 outside its grid."""
    index = np.floor((points - np.asarray(origin)) / np.asarray(spacing) + 0.5)
    values = np.zeros(len(points), np.int64)
    with np.errstate(invalid="ignore"):
        inside = np.isfinite(index[:, 1:]).all(axis=1)
        for axis in (1, 2):
            inside &= (index[:, axis] >= 0) & (index[:, axis] < labels.shape[axis])
        if plane:
            index[:, 0] = 0
        else:
            inside &= np.isfinite(index[:, 0]) & (index[:, 0] >= 0) & (index[:, 0] < labels.shape[0])
    z, y, x = index[inside].astype(np.int64).T
    values[inside] = labels[z, y, x]
    return values


def _edge_values(labels, plane):
    """(label values on the first and last Y row and X column; on the first and last Z plane)."""
    yx = np.unique(np.concatenate([labels[:, 0, :].ravel(), labels[:, -1, :].ravel(), labels[:, :, 0].ravel(),
                                   labels[:, :, -1].ravel()]))
    z = np.array([], labels.dtype) if plane else np.unique(np.concatenate([labels[0].ravel(), labels[-1].ravel()]))
    return set(yx[yx > 0].tolist()), set(z[z > 0].tolist())


def _duplicate_rule(placement, by_fov, state, order, threshold):
    """The duplicate rule (docs/assembly-algorithms.md, "Duplicate rule") over the cells retained so far.

    Returns ``(excluded, sizes, shared)``: ``{(fov_id, cell_id): (kept fov_id,
    kept cell_id)}``, each retained cell's territory voxels on the sample grid
    and the shared voxels of every pair of retained cells of two FOVs that
    share one (``_duplicates.measure``).
    """
    from starfinder.sample_output._grid import INCLUDED

    from . import _duplicates
    fovs = [f for f in order if state[f]["placed"] and any(state[f]["retained"].values())]
    if len(fovs) < 2:
        return {}, {}, {}
    spacings, kinds = {}, {}
    for f in fovs:
        grid = by_fov[f].grid
        spacings.setdefault(tuple(float(v) for v in grid.metadata.spacing_zyx), []).append(f)
        kinds.setdefault("plane" if state[f]["plane"] or grid.shape_zyx[0] == 1 else "volume", []).append(f)
    if len(spacings) > 1:
        raise ValueError("the duplicate rule places the territories on one sample grid, so the FOVs with a retained "
                         "cell must share one spacing (or set duplicate_threshold to None); " +
                         "; ".join(f"{_names(f)}: {key!r}" for key, f in spacings.items()))
    if len(kinds) > 1:
        raise ValueError("the duplicate rule needs the territories of the FOVs with a retained cell to be all plane or "
                         f"all volume (or set duplicate_threshold to None); plane: {_names(kinds['plane'])}, "
                         f"volume: {_names(kinds['volume'])}")
    rows = placement.fovs.set_index("fov_id")
    origin = {f: tuple(float(rows.loc[f, f"origin_{a}"]) for a in AXES) for f in rows.index}
    # The label export's grid at scale 1: the smallest origin of the included FOVs per axis, their spacing.
    included = [f for f, status in zip(placement.fovs["fov_id"], placement.fovs["status"]) if status in INCLUDED]
    grid = {"origin": tuple(min(origin[f][a] for f in included) for a in range(3)),
            "spacing": next(iter(spacings)), "two_d": next(iter(kinds)) == "plane"}
    items = [{"fov_id": f, "origin": origin[f], "shape": by_fov[f].grid.shape_zyx,
              "cells": np.array(sorted(c for c, kept in state[f]["retained"].items() if kept), np.int64)}
             for f in fovs]
    sizes, shared = _duplicates.measure(items, lambda f: _territory(by_fov[f]), grid)
    position = {f: k for k, f in enumerate(order)}
    return _duplicates.decide(sizes, shared, threshold, position), sizes, shared


def assemble(placement: PlacementResult, inputs: Sequence[AssemblyInput], *,
             config: AssemblyConfig = AssemblyConfig(), segmentation=None) -> AssemblyResult:
    """Assemble a sample from its placement and its per-FOV §2.9 results, or on the route after fusion.

    Applies the checks of docs/assembly-contract.md ("Checks the entry
    applies") in order, then maps every observation into sample coordinates
    (``origin + p ⊙ spacing``, float64, never rounded), computes the ownership
    regions from the placement (:func:`ownership_map`), gives every cell and
    molecule one assembly status with its reason, applies the duplicate rule,
    computes the flags, and copies the §2.9 counts of the counted cells. A cell
    is retained when the owner of its ownership centroid is its own FOV; the
    molecules of a cell follow it wherever they lie; an unassigned molecule is
    retained by its own position; ``outside_grid`` molecules and every
    observation of a FOV without an origin are excluded; a ``placeholder`` owns
    no point. Then, with ``config.duplicate_threshold`` set, two retained cells
    of different FOVs whose territories share at least that share of the
    smaller territory on the sample grid are two observations of one cell: the
    larger stays and the other is excluded ``duplicate_observation`` with its
    molecules, naming the kept cell in ``duplicate_of``. No cell is merged and
    no molecule moves to another cell. It reads no file except a saved
    assignment named by path, and writes none. The record's ``route`` is
    ``per_fov`` (``route_source`` null; ``Dataset.export`` records whether the
    route was set or taken from the default).

    With ``segmentation`` (the route after fusion,
    docs/sample-segmentation-contract.md) the inputs are molecule tables: it
    computes their ``molecules_only`` assembly by the rules above, converts
    its retained molecules to one molecule table on the segmentation grid
    (``spot_namespace`` ``[dataset_id, sample_id, null, null]``, ``z, y, x =
    (sample_zyx − O) ⊘ S``), calls
    :func:`~starfinder.assignment.assign_molecules` once on the sample with
    the cell run, the nucleus run and the AssignmentConfig of the
    segmentation's record (an expansion in pixels divided by the scale
    factor), and maps its rows back to the original keys by position. Every
    cell of the cell run is one retained row with its label as ``cell_id``;
    its ``fov_id`` is the owner of its ownership centroid or, where no owning
    footprint holds it, the owning FOV with the nearest footprint centre
    (``owner_fov_id`` null); no cell flag and no duplicate rule applies. A
    molecule keeps its rule (a position belongs to one FOV); a retained one
    outside the segmentation grid in Y or X is excluded ``outside_grid``. The
    record holds ``route`` ``after_fusion``, ``segmentation_files`` (each
    run's label file and SHA-256) and the sample assignment's record as
    ``assignment``; with an expansion :func:`save_assembly` writes the
    territories as ``territories.ome.tif``.

    Parameters
    ----------
    placement : PlacementResult
        Any placement; one whose review is required must hold an approval.
    inputs : Sequence[AssemblyInput]
        One per FOV of the placement that has observations; every
        ``image_supported`` and ``metadata_only`` FOV needs one.
    config : AssemblyConfig
        Ownership rule, centroid, tie rule, whether to compute the
        ``conflict`` and ``unclaimed`` flags, and the duplicate rule's
        threshold.
    segmentation : SampleSegmentation, optional
        The sample segmentation of the route after fusion
        (:func:`~starfinder.sample_output.segment_fused` or
        :func:`~starfinder.sample_output.load_sample_segmentation`), status
        ``written`` or ``reused``. Its record's ``assignment`` holds the
        assignment settings the tables are made with (``AssignmentConfig()``
        when it holds none); with a reused segmentation ``Dataset.export``
        sets them to the current export's. The result's record keeps the
        §2.9 record of the assignment made as ``assignment``.

    Returns
    -------
    AssemblyResult

    Raises
    ------
    TypeError
        An argument of the wrong type.
    PlacementReviewRequiredError
        Review is required and the placement holds no approval of its tables.
    ValueError
        A repeated or unknown FOV, an owning FOV without an input, inputs of
        both kinds or of neither, observations of another FOV, sample or a
        subtile, different gene lists, placed FOVs of different ``nominal_z``,
        a placement without an owning FOV, or a repeated ``obs_name``; with the
        duplicate rule on, FOVs with retained cells of different spacings or
        of mixed plane and volume territories; with a segmentation, inputs
        that are not molecule tables, a segmentation without cells, of
        another frame or placement, or whose labels are of another
        ``[dataset_id, sample_id]`` than the molecules.
    IncompatibleGeometryError
        An input grid that is not the grid the placement (or the assignment)
        recorded, or one without known spacing, unit µm and identity direction.
    """
    if segmentation is not None:
        from ._fused import _check, assemble_after_fusion
        base = assemble(placement, inputs, config=config)
        _check(placement, list(inputs), segmentation)
        return assemble_after_fusion(placement, list(inputs), config, segmentation, base)
    loaded, mode, genes = _check_inputs(placement, inputs, config)
    by_fov = {fov.fov_id: fov for fov in loaded}
    ownership = ownership_map(placement, {fov.fov_id: fov.grid for fov in loaded}, config=config)
    geometry = ownership.geometry.set_index("fov_id")
    sample = placement.sample
    fov_rows = {row.fov_id: row for row in placement.fovs.itertuples(index=False)}
    order = [fov for fov in placement.fovs["fov_id"] if fov in by_fov]
    computed = ["fov_edge", "z_edge", "outside_owner_region"] if mode == "cells" else []
    if mode == "cells" and config.conflicts:
        computed += ["conflict", "unclaimed", "cell_unclaimed"]
    computed = [f for f in (*CELL_ASSEMBLY_FLAGS, *MOLECULE_ASSEMBLY_FLAGS) if f in computed]

    cell_parts, molecule_parts, count_parts, state = [], [], [], {}
    for fov_id in order:
        fov, row = by_fov[fov_id], fov_rows[fov_id]
        placed = row.status in PLACED
        origin = np.array([row.origin_z, row.origin_y, row.origin_x], np.float64)
        spacing = np.asarray(fov.grid.metadata.spacing_zyx, np.float64)

        def to_sample(p):
            return origin + p * spacing if placed else np.full(p.shape, np.nan)

        # Molecules: sample position and its owner.
        table = fov.result.molecules if mode == "cells" else fov.molecules.table.assign(
            assignment_status=pd.array([None] * len(fov.molecules.table), dtype="string"))
        positions = table[["z", "y", "x"]].to_numpy(np.float64)
        if fov.grid.shape_zyx[0] == 1:
            # A one-plane grid samples y and x only: every molecule is on its plane, index z 0; the column z keeps
            # the recorded value.
            positions = positions.copy()
            positions[:, 0] = 0.0
        sample_zyx = to_sample(positions)
        molecule_owner = ownership.owner(sample_zyx) if placed else np.full(len(table), None, dtype=object)
        molecules = table.copy()
        molecules.insert(0, "fov_id", fov_id)
        for i, a in enumerate(AXES):
            molecules[f"sample_{a}"] = sample_zyx[:, i]
        molecules["owner_fov_id"] = pd.array(list(molecule_owner), dtype="string")

        if mode == "cells":
            result = fov.result
            labels = _territory(fov)
            plane = result.record["inputs"]["cells"]["geometry"] == "plane"
            expanded = _expanded(fov)
            cells = result.cells.sort_values("cell_id", kind="mergesort").reset_index(drop=True)
            cell_ids = cells["cell_id"].astype("int64").to_numpy()
            original = to_sample(cells[[f"centroid_{a}" for a in AXES]].to_numpy(np.float64))
            grown = to_sample(cells[[f"expanded_centroid_{a}" for a in AXES]].to_numpy(np.float64))
            if plane:
                original[:, 0] = np.nan
                grown[:, 0] = np.nan
            use_expanded = config.centroid == "territory" and expanded
            own = grown if use_expanded else original
            owner = ownership.owner(own) if placed else np.full(len(cells), None, dtype=object)
            if placed:
                retained = owner == fov_id
                reason = np.where(retained, None, "not_owner")
            else:
                retained = np.zeros(len(cells), bool)
                reason = np.full(len(cells), "fov_not_placed", dtype=object)
            yx_edge, z_edge = _edge_values(labels, plane)
            del labels
            flags = [{f for f, hit in (("fov_edge", c in yx_edge), ("z_edge", c in z_edge)) if hit}
                     for c in cell_ids.tolist()]
            front = pd.DataFrame({"fov_id": [fov_id] * len(cells),
                                  "cell_namespace": [result.cell_namespace] * len(cells),
                                  "cell_id": cells["cell_id"].array,
                                  "obs_name": [f"{sample}_{fov_id}_{c}" for c in cell_ids.tolist()]})
            frame = pd.concat([front, cells.drop(columns="cell_id")], axis=1)
            for i, a in enumerate(AXES):
                frame[f"sample_centroid_{a}"] = original[:, i]
            for i, a in enumerate(AXES):
                frame[f"sample_expanded_centroid_{a}"] = grown[:, i]
            frame["ownership_centroid"] = "expanded" if use_expanded else "original"
            for i, a in enumerate(AXES):
                frame[f"sample_ownership_centroid_{a}"] = own[:, i]
            frame["owner_fov_id"] = pd.array(list(owner), dtype="string")
            frame["assembly_status"] = np.where(retained, "retained", "excluded")
            frame["assembly_reason"] = pd.array(list(reason), dtype="string")
            state[fov_id] = {"frame": frame, "flags": flags, "plane": plane, "origin": origin,
                             "spacing": spacing, "retained": dict(zip(cell_ids.tolist(), retained.tolist())),
                             "row_of": {c: i for i, c in enumerate(cell_ids.tolist())},
                             "own": own, "placed": placed}
            cell_status = dict(zip(cell_ids.tolist(), zip(frame["assembly_status"], reason)))
            status = table["assignment_status"].astype(object).to_numpy()
            cell_of = table["cell_id"].astype(object).to_numpy()
            # The cell key of a molecule's cell: a §2.9 cell_id is a label of this FOV's cell run.
            molecules["cell_namespace"] = pd.array(
                np.where(table["cell_id"].notna().to_numpy(dtype=bool), result.cell_namespace, None),
                dtype="string")
        else:
            status = np.full(len(table), None, dtype=object)
            shape = np.asarray(fov.grid.shape_zyx, np.float64)
            index = np.floor(positions + 0.5)
            outside = ~((index >= 0) & (index < shape)).all(axis=1)
            status[outside] = "outside_grid"
            cell_of = np.full(len(table), None, dtype=object)
            cell_status = {}
            # No molecule of a molecules_only sample has a cell: its cell key is null.
            molecules["cell_namespace"] = pd.array([None] * len(table), dtype="string")
        # Molecule retention (rules 2 to 6).
        own_position = molecule_owner == fov_id
        m_status = np.where(own_position, "retained", "excluded").astype(object)
        m_reason = np.where(own_position, None, "not_owner").astype(object)
        in_cell = np.isin(status, ("assigned", "excluded_cell"))
        if in_cell.any():
            ids = [int(c) for c in cell_of[in_cell]]
            m_status[in_cell] = [cell_status[c][0] for c in ids]
            m_reason[in_cell] = [None if cell_status[c][0] == "retained" else "cell_not_owner" for c in ids]
        outside = status == "outside_grid"
        m_status[outside], m_reason[outside] = "excluded", "outside_grid"
        if not placed:
            m_status[:], m_reason[:] = "excluded", "fov_not_placed"
        molecules["assembly_status"] = m_status
        molecules["assembly_reason"] = pd.array(list(m_reason), dtype="string")
        state.setdefault(fov_id, {})["molecules"] = molecules
        state[fov_id]["molecule_cells"] = cell_of
        state[fov_id]["molecule_status"] = status

    # The duplicate rule, among the cells retained by their ownership centroid; the flags follow, among the cells
    # retained after it, and the molecules of an excluded observation are excluded with it.
    duplicate_of = {}
    if mode == "cells" and config.duplicate_threshold is not None:
        duplicate_of = _duplicate_rule(placement, by_fov, state, order, config.duplicate_threshold)[0]
    for fov_id, cell in duplicate_of:
        sf = state[fov_id]
        i = sf["row_of"][cell]
        frame = sf["frame"]
        frame.iloc[i, frame.columns.get_loc("assembly_status")] = "excluded"
        frame.iloc[i, frame.columns.get_loc("assembly_reason")] = "duplicate_observation"
        sf["retained"][cell] = False
        rows = np.isin(sf["molecule_status"], ("assigned", "excluded_cell"))
        rows[rows] = [int(c) == cell for c in sf["molecule_cells"][rows]]
        molecules = sf["molecules"]
        molecules.loc[rows, "assembly_status"] = "excluded"
        molecules.loc[rows, "assembly_reason"] = "cell_duplicate_observation"

    # Flags across FOVs: conflict and unclaimed (centroid tests in the other FOV's territory). The territory of
    # each FOV b is read once and every test in it is made then, so at most two territory images are in memory.
    conflict_with = {fov_id: [set() for _ in range(len(state[fov_id].get("frame", ())))] for fov_id in order}
    unclaimed = {fov_id: np.zeros(len(state[fov_id].get("frame", ())), bool) for fov_id in order}
    if mode == "cells" and config.conflicts:
        others = [f for f in order if state[f]["placed"]]
        retained = {}
        for a in others:
            keep = state[a]["frame"]["assembly_status"].eq("retained").to_numpy(dtype=bool)
            if keep.any():
                retained[a] = (np.flatnonzero(keep), state[a]["own"][keep])
        queries = {}
        for a in order:
            frame = state[a]["frame"]
            for i in np.flatnonzero(frame["assembly_reason"].eq("not_owner").fillna(False).to_numpy(dtype=bool)):
                owner = frame["owner_fov_id"].iloc[i]
                if pd.isna(owner):
                    unclaimed[a][i] = True
                else:
                    queries.setdefault(owner, []).append((a, i))
        for b in order:
            tests = [a for a in retained if a != b] if b in others else []
            if not tests and b not in queries:
                continue
            sb, labels = state[b], _territory(by_fov[b])
            gb = geometry.loc[b]
            for a in tests:
                rows, points = retained[a]
                inside = ((points[:, 1] >= gb.footprint_y0) & (points[:, 1] < gb.footprint_y1)
                          & (points[:, 2] >= gb.footprint_x0) & (points[:, 2] < gb.footprint_x1))
                if not inside.any():
                    continue
                values = _sample_territory(labels, sb["plane"], sb["origin"], sb["spacing"], points[inside])
                for i, v in zip(rows[inside], values.tolist()):
                    if v and sb["retained"].get(v, False):
                        j = sb["row_of"][v]
                        conflict_with[a][i].add((b, j))
                        conflict_with[b][j].add((a, i))
            for a, i in queries.get(b, []):
                v = int(_sample_territory(labels, sb["plane"], sb["origin"], sb["spacing"],
                                          state[a]["own"][i:i + 1])[0])
                unclaimed[a][i] = v == 0 or not sb["retained"].get(v, False)
            del labels
    position = {fov_id: k for k, fov_id in enumerate(order)}
    for fov_id in order:
        sf = state[fov_id]
        molecules = sf["molecules"]
        if mode == "cells":
            frame = sf["frame"]
            texts, partners = [], []
            for i, flags in enumerate(sf["flags"]):
                flags = set(flags)
                if conflict_with[fov_id][i]:
                    flags.add("conflict")
                if unclaimed[fov_id][i]:
                    flags.add("unclaimed")
                texts.append(_flag_text(flags, CELL_ASSEMBLY_FLAGS))
                pairs = sorted(conflict_with[fov_id][i], key=lambda p: (position[p[0]], p[1]))
                partners.append(";".join(state[b]["frame"]["obs_name"].iloc[j] for b, j in pairs)
                                if config.conflicts else None)
            frame["assembly_flags"] = texts
            frame["conflict_with"] = pd.array(partners, dtype="string")
            kept = [duplicate_of.get((fov_id, int(c))) for c in frame["cell_id"]]
            frame["duplicate_of"] = pd.array([None if k is None else state[k[0]]["frame"]["obs_name"].iloc[
                state[k[0]]["row_of"][k[1]]] for k in kept], dtype="string")
            cell_parts.append(frame)
            flagged = {int(c) for c, f in zip(frame["cell_id"], texts) if "unclaimed" in f.split(";")}
            # Counts of the counted cells.
            counted = frame[(frame["assembly_status"].eq("retained") & frame["status"].eq("kept")).to_numpy(
                dtype=bool)]["cell_id"].astype("int64")
            counts = by_fov[fov_id].result.counts
            counts = counts[counts["cell_id"].astype("int64").isin(set(counted.tolist())).to_numpy(dtype=bool)]
            count_parts.append(pd.concat([pd.DataFrame({"fov_id": [fov_id] * len(counts),
                                                        "cell_namespace": [by_fov[fov_id].result.cell_namespace]
                                                        * len(counts)}),
                                          counts.reset_index(drop=True)], axis=1))
        else:
            flagged = set()
        m_flags = []
        for a_status, cell, status, owner in zip(sf["molecule_status"], sf["molecule_cells"],
                                                 molecules["assembly_status"],
                                                 molecules["owner_fov_id"].astype(object)):
            flags = set()
            if status == "retained" and a_status in ("assigned", "excluded_cell") and not pd.isna(owner) \
                    and owner != fov_id:
                flags.add("outside_owner_region")
            if status == "excluded" and a_status in ("assigned", "excluded_cell") and int(cell) in flagged:
                flags.add("cell_unclaimed")
            m_flags.append(_flag_text(flags, MOLECULE_ASSEMBLY_FLAGS))
        molecules["assembly_flags"] = m_flags
        molecule_parts.append(molecules)

    cells = _frame(cell_parts, CELL_DTYPES)
    molecules = _frame(molecule_parts, molecule_dtypes(mode))
    counts = _frame(count_parts, COUNT_DTYPES)
    repeated = sorted(k for k, n in Counter(cells["obs_name"]).items() if n > 1)
    if repeated:
        raise ValueError(f"obs_name must be unique in the sample; repeated: {_names(repeated)}")

    fovs = _fov_table(placement, geometry, by_fov, cells, molecules)
    record = _record(placement, config, mode, genes, ownership, by_fov, fovs, cells, molecules, computed)
    return AssemblyResult(sample, fovs, cells, molecules, counts, genes, record)


def _fov_table(placement, geometry, by_fov, cells, molecules):
    rows = []
    n_cells = cells["fov_id"].value_counts().to_dict()
    n_molecules = molecules["fov_id"].value_counts().to_dict()
    kept_cells = cells[cells["assembly_status"].eq("retained").to_numpy(dtype=bool)]["fov_id"].value_counts()
    kept_molecules = molecules[molecules["assembly_status"].eq("retained").to_numpy(dtype=bool)]["fov_id"]
    kept_molecules = kept_molecules.value_counts()
    for row in placement.fovs.itertuples(index=False):
        g = geometry.loc[row.fov_id]
        rows.append({"fov_id": row.fov_id, "placement_status": row.status, "origin_z": row.origin_z,
                     "origin_y": row.origin_y, "origin_x": row.origin_x,
                     "kind": by_fov[row.fov_id].kind if row.fov_id in by_fov else None,
                     **{c: g[c] for c in ("footprint_y0", "footprint_y1", "footprint_x0", "footprint_x1",
                                          "centre_y", "centre_x")},
                     "owns_points": bool(g["owns_points"]), "n_cells": n_cells.get(row.fov_id, 0),
                     "n_cells_retained": int(kept_cells.get(row.fov_id, 0)),
                     "n_molecules": n_molecules.get(row.fov_id, 0),
                     "n_molecules_retained": int(kept_molecules.get(row.fov_id, 0))})
    return pd.DataFrame(rows, columns=list(FOV_DTYPES)).astype(FOV_DTYPES)


def _finite_or_none(values):
    return [float(v) if np.isfinite(v) else None for v in values]


def _fov_record(fov_row, entry, fov, config, mode):
    grid = fov.grid if fov is not None else None
    item = {"fov_id": fov_row.fov_id, "placement_status": fov_row.placement_status,
            "origin_zyx": _finite_or_none([fov_row.origin_z, fov_row.origin_y, fov_row.origin_x]),
            "kind": None if fov is None else fov.kind, "assignment": None, "molecules_sha256": None,
            "nuclei": False, "territories": {"path": None, "sha256": None}, "grid": None,
            "footprint_yx": None if not np.isfinite(fov_row.footprint_y0) else
            [[float(fov_row.footprint_y0), float(fov_row.footprint_y1)],
             [float(fov_row.footprint_x0), float(fov_row.footprint_x1)]],
            "centre_yx": _finite_or_none([fov_row.centre_y, fov_row.centre_x]) if np.isfinite(fov_row.centre_y)
            else None,
            "owns_points": bool(fov_row.owns_points)}
    if entry is not None and np.isfinite(entry.get("box_y0", np.nan)):
        item["box_yx"] = [[float(entry["box_y0"]), float(entry["box_y1"])],
                          [float(entry["box_x0"]), float(entry["box_x1"])]]
    if fov is None:
        return item
    item["grid"] = {"shape_zyx": list(grid.shape_zyx), "metadata": json.loads(json.dumps(asdict(grid.metadata))),
                    "sha256": grid.sha256, "source": grid.source,
                    "calibration_source": "declared" if grid.source == "declared" else "grid"}
    if fov.kind == "molecules":
        item["molecules_sha256"] = fov.molecules.sha256
        return item
    result = fov.result
    record = result.record
    item["molecules_sha256"] = record["inputs"]["molecules"]["sha256"]
    item["nuclei"] = result.nuclei if isinstance(result, _Saved) else result.nucleus_labels is not None
    if fov.folder is not None:
        from starfinder._records import file_sha256
        files = record["files"]
        item["assignment"] = {"name": fov.name, "folder": str(fov.folder), "source": "folder",
                              "record_sha256": file_sha256(fov.folder / ASSIGNMENT_RECORD),
                              "tables": {t: files[t]["sha256"] for t in ("molecules", "cells", "counts") if t in files}}
        if config.conflicts:
            entry_file = files["territories"] if "territories" in files else record["inputs"]["cells"]["file"]
            item["territories"] = {"path": os.path.normpath(fov.folder / entry_file["path"]),
                                   "sha256": entry_file["sha256"]}
    else:
        item["assignment"] = {"name": record.get("name"), "folder": None, "source": "memory",
                              "record_sha256": json_sha256(record),
                              "tables": {t: assignment_table_sha256(getattr(result, t))
                                         for t in ("molecules", "cells", "counts")}}
    return item


def _record(placement, config, mode, genes, ownership: OwnershipMap, by_fov, fovs, cells, molecules, computed):
    from starfinder._records import json_record
    review = placement.record.get("review") or {}
    hashes = table_sha256(placement.fovs), table_sha256(placement.links)
    approvals = [a for a in review.get("approvals", []) if (a.get("fovs_sha256"), a.get("links_sha256")) == hashes]
    method = placement.record.get("method") or {}
    geometry = {row["fov_id"]: row for row in ownership.geometry.to_dict("records")}
    return json_record({
        "format_version": FORMAT_VERSION, "stage": STAGE, "sample": placement.sample, "mode": mode,
        "route": "per_fov", "route_source": None,
        "frame": placement.record.get("frame"), "genes": list(genes),
        "placement": {"sample": placement.sample, "method": method.get("method"),
                      "implementation": method.get("implementation"), "config": method.get("config"),
                      "origins": placement.record.get("origins"), "gauge": placement.record.get("gauge"),
                      "review": review.get("state"), "reasons": review.get("reasons", []),
                      "approval": approvals[-1] if approvals else None,
                      "fovs_sha256": hashes[0], "links_sha256": hashes[1]},
        "config": asdict(config),
        "ownership": {"rule": config.ownership, "tie": config.tie, "axes": "yx",
                      "owning_fovs": list(ownership.owning)},
        "flags": {"computed": list(computed), "conflicts": bool(config.conflicts)},
        "fovs": [_fov_record(row, geometry.get(row.fov_id), by_fov.get(row.fov_id), config, mode)
                 for row in fovs.itertuples(index=False)],
        "counts": _counts_record(cells, molecules, fovs, set(computed), config.duplicate_threshold is not None),
        "files": {},
        "software": software(),
    })
