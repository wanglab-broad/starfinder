"""SO3 of design/validation/sample-segmentation.md: the column ``cell_namespace`` of the sample molecule
table on the route per FOV (choice 7, CK1 of design/specs/sample-segmentation.md), and the readers that
link a molecule to its cell by the cell key ``(cell_namespace, cell_id)``.

Fixtures: the assemblies of design/validation/stitching-assembly.md (``assembly_golden`` 3D, 2D and
expanded, ``own_cases`` with the duplicate rule on and off, ``empty_cases`` and ``status_cases``),
built by ``assembly_fixtures.py``, and ``big_ids``: ``assembly_golden`` (3D) with two label values at
and above 2^31.

Until W-574, ``test_so3_the_column_on_the_route_per_fov`` also compared every other column of the four
sample tables with the tables ``assemble`` gave at the base revision 725f24a, before the column
existed. That comparison guarded the moment the column was added and is dropped as history. The
values of the assembly tables are checked by the AS rows of design/validation/stitching-assembly.md:
``test_stitching_assembly_ownership.py`` (AS1 sample coordinates, AS2 owners),
``test_stitching_assembly_retention.py`` (AS3 to AS9: retained cells, rows, reasons and flags,
identities, molecules-only, empty and status cases), ``test_stitching_assembly_accounting.py`` (the
counts) and ``test_stitching_assembly_scene.py`` (EE2, the assembly against an independent oracle).
"""
import dataclasses
import json

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from starfinder.assembly import AssemblyConfig, AssemblyResult, assemble, load_assembly, save_assembly

from .assembly_fixtures import (GOLDEN_IDS, TILE_ZYX, approved, golden_molecule_tables, golden_pieces,
                                golden_placement, golden_results, inputs_of, own_placement, own_results,
                                placeholder_result, unit_grid)

pytestmark = [pytest.mark.stitching, pytest.mark.validation]

TABLES = ("fovs", "cells", "molecules", "counts")
KEY = "cell_namespace"
BIG = {"Position001": {1: 2 ** 32 - 1}, "Position003": {2: 2 ** 31}}


def _empty_variants():
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


def _big_labels():
    out = {}
    for fov, (labels, _, _) in golden_pieces().items():
        labels = labels.astype(np.uint32)
        for old, new in BIG.get(fov, {}).items():
            labels = np.where(labels == old, np.uint32(new), labels)
        out[fov] = labels
    return out


def _molecules_only():
    return golden_molecule_tables()


def build(case):
    """(AssemblyResult, the per-FOV inputs) of one SO3 case."""
    if case.startswith("golden_"):
        dimensions = case.split("_")[1]
        results = golden_results("3d" if dimensions == "expanded" else dimensions,
                                 expand=("Position001", "Position003") if dimensions == "expanded" else ())
        return assemble(golden_placement(), inputs_of(results)), results
    if case == "big_ids":
        results = golden_results(labels=_big_labels())
        return assemble(golden_placement(), inputs_of(results)), results
    if case.startswith("own_cases"):
        config = AssemblyConfig() if case == "own_cases" else AssemblyConfig(duplicate_threshold=None)
        results = own_results()
        inputs = inputs_of(results, {fov: unit_grid((4, 16, 16)) for fov in results})
        return assemble(own_placement(), inputs, config=config), results
    if case.startswith("empty_"):
        name = case[len("empty_"):]
        if name == "molecules_only":
            tables = _molecules_only()
            return assemble(golden_placement(), inputs_of(tables, kind="molecules")), tables
        results = golden_results(**_empty_variants()[name])
        return assemble(golden_placement(), inputs_of(results)), results
    results = golden_results()
    if case == "status_missing":
        placement = approved(golden_placement(statuses={"Position003": "missing"}, review="required"))
        results = {f: r for f, r in results.items() if f != "Position003"}
    elif case == "status_failed":
        placement = approved(golden_placement(statuses={"Position003": "failed"}, review="required"))
    elif case == "status_placeholder_without_input":
        placement = golden_placement(statuses={"Position002": "placeholder"})
        results = {f: r for f, r in results.items() if f != "Position002"}
    elif case == "status_placeholder_with_input":
        placement = golden_placement(statuses={"Position002": "placeholder"})
        results = {**results, "Position002": placeholder_result()}
    else:
        raise KeyError(case)
    return assemble(placement, inputs_of(results)), results


CASES = ["golden_3d", "golden_2d", "golden_expanded", "big_ids", "own_cases", "own_cases_rule_off",
         *(f"empty_{name}" for name in (*_empty_variants(), "molecules_only")),
         "status_missing", "status_failed", "status_placeholder_without_input", "status_placeholder_with_input"]


MOLECULE_COLUMNS_29 = ["spot_namespace", "spot_id", "z", "y", "x", "gene_id", "voxel_z", "voxel_y", "voxel_x",
                       "assignment_status", "cell_id", "in_expansion", "original_cell_id", "nucleus_id", "compartment"]
MOLECULE_ONLY_COLUMNS = ["spot_namespace", "spot_id", "z", "y", "x", "gene_id", "assignment_status"]
TAIL = ["sample_z", "sample_y", "sample_x", "owner_fov_id", "assembly_status", "assembly_reason", "assembly_flags"]


def namespace_of(inputs, fov):
    return inputs[fov].cell_namespace


def without_key(result):
    """The result with its molecule table as it was before the column (no identity check runs)."""
    out = object.__new__(AssemblyResult)
    for field in dataclasses.fields(AssemblyResult):
        object.__setattr__(out, field.name, getattr(result, field.name))
    object.__setattr__(out, "molecules", result.molecules.drop(columns=KEY, errors="ignore"))
    return out


def old_rule_links(result):
    """cell_instance_id by the rule before the column: a molecule's cell is (its fov_id, cell_id)."""
    cells, molecules = result.cells, result.molecules
    of = {(f, int(c)): i + 1 for i, (f, c) in enumerate(zip(cells.fov_id, cells.cell_id.astype("int64")))}
    return [None if pd.isna(c) else of[(f, int(c))] for f, c in zip(molecules.fov_id, molecules.cell_id)]


# --- SO3: the column on the route per FOV -------------------------------------------------------------------

@pytest.mark.parametrize("case", CASES)
def test_so3_the_column_on_the_route_per_fov(case):
    result, inputs = build(case)
    molecules = result.molecules
    assert str(molecules[KEY].dtype) == "string"
    if case == "empty_molecules_only":
        # A molecules_only table has no cell_id: no molecule has a cell, so the column is null on every row.
        assert list(molecules.columns) == ["fov_id", *MOLECULE_ONLY_COLUMNS, KEY, *TAIL]
        assert len(molecules) and molecules[KEY].isna().all()
    else:
        assert list(molecules.columns) == ["fov_id", *MOLECULE_COLUMNS_29, KEY, *TAIL]
        expected = [None if pd.isna(c) else namespace_of(inputs, f) for f, c in zip(molecules.fov_id,
                                                                                   molecules.cell_id)]
        assert [None if pd.isna(v) else v for v in molecules[KEY]] == expected
        assert molecules[KEY].notna().sum() == molecules.cell_id.notna().sum()
        if molecules.cell_id.notna().any():
            assert molecules[KEY].notna().any()
    if case == "big_ids":
        ids = set(result.molecules.cell_id.dropna().astype("int64")) & set(result.cells.cell_id.astype("int64"))
        assert {2 ** 32 - 1, 2 ** 31} <= ids


@pytest.mark.parametrize("table_format", ["csv", "parquet"])
@pytest.mark.parametrize("case", ["golden_3d", "big_ids", "own_cases", "empty_molecules_only"])
def test_so3_save_and_load_keep_the_column(case, table_format, tmp_path):
    pytest.importorskip("anndata", reason="the anndata extra (starfinder[anndata]) is not installed")
    result, _ = build(case)
    saved = save_assembly(result, tmp_path, table_format=table_format)
    assert saved.record["files"]["molecules"]["dtypes"][KEY] == "string"
    loaded = load_assembly(tmp_path)
    for name in TABLES:
        assert_frame_equal(getattr(loaded, name), getattr(result, name), check_exact=True)
    assert str(loaded.molecules[KEY].dtype) == "string"
    if case == "empty_molecules_only":
        assert loaded.molecules[KEY].isna().all()


def _as_saved_before_the_column(folder):
    """Rewrite a saved folder's molecule table without cell_namespace, its record as save_assembly wrote it then."""
    from starfinder._records import _read_table, _write_table, file_sha256, write_json
    sections = json.loads((folder / "export.json").read_text())
    entry = sections["assembly"]["files"]["molecules"]
    frame = _read_table(folder / entry["path"], entry["dtypes"]).drop(columns=KEY)
    name, dtypes = _write_table(frame, folder / "tables", "molecules", entry["path"].rsplit(".", 1)[1])
    sections["assembly"]["files"]["molecules"] = {"path": f"tables/{name}", "dtypes": dtypes,
                                                  "sha256": file_sha256(folder / "tables" / name)}
    write_json(sections, folder / "export.json")
    return frame


@pytest.mark.parametrize("table_format", ["csv", "parquet"])
@pytest.mark.parametrize("case", ["golden_3d", "golden_expanded", "big_ids", "own_cases",
                                  "status_placeholder_with_input", "status_failed", "empty_no_cell_row",
                                  "empty_molecules_only"])
def test_so3_a_folder_saved_before_the_column_loads(case, table_format, tmp_path):
    pytest.importorskip("anndata", reason="the anndata extra (starfinder[anndata]) is not installed")
    result, _ = build(case)
    save_assembly(result, tmp_path, table_format=table_format)
    old = _as_saved_before_the_column(tmp_path)
    assert KEY not in old
    loaded = load_assembly(tmp_path)
    # The column is filled from the cell table, at its place; the loaded tables equal those assemble gives now.
    for name in TABLES:
        assert_frame_equal(getattr(loaded, name), getattr(result, name), check_exact=True)
    assert_frame_equal(loaded.molecules.drop(columns=KEY), old, check_exact=True)


def test_so3_the_fill_rule():
    from starfinder.assembly._assemble import fill_cell_namespace
    cells = pd.DataFrame({"fov_id": pd.array(["F1", "F1", "F2", "F3", "F3"], dtype="string"),
                          "cell_namespace": pd.array(["n1", "n1", "n2", "n3a", "n3b"], dtype="string"),
                          "cell_id": pd.array([1, 2 ** 32 - 1, 1, 7, 7], dtype="UInt32")})
    molecules = pd.DataFrame({"fov_id": pd.array(["F1", "F1", "F2", "F2", "F3", "F1"], dtype="string"),
                              "cell_id": pd.array([2 ** 32 - 1, None, 1, 5, 7, 1], dtype="UInt32"),
                              "sample_z": [0.0] * 6})
    filled = fill_cell_namespace(molecules, cells)
    assert list(filled.columns) == ["fov_id", "cell_id", KEY, "sample_z"]
    # The cell of the molecule's own FOV; null without a cell_id, for a cell no row holds, and for two namespaces.
    assert [None if pd.isna(v) else v for v in filled[KEY]] == ["n1", None, "n2", None, None, "n1"]
    assert fill_cell_namespace(filled, cells) is filled
    # A molecules_only table (no cell_id): null on every row, at its place.
    only = fill_cell_namespace(molecules.drop(columns="cell_id"), cells.iloc[:0])
    assert list(only.columns) == ["fov_id", KEY, "sample_z"] and str(only[KEY].dtype) == "string"
    assert only[KEY].isna().all()


# --- The readers that link a molecule to its cell, by the cell key -------------------------------------------

@pytest.mark.parametrize("case", ["golden_3d", "golden_2d", "big_ids", "own_cases", "own_cases_rule_off",
                                  "status_placeholder_with_input", "status_failed", "empty_no_cell_row",
                                  "empty_molecules_only"])
def test_so3_cell_links_by_the_cell_key(case):
    from starfinder.export._spatialdata import cell_links
    result, _ = build(case)
    links = cell_links(result)
    if case == "empty_molecules_only":
        assert links.isna().all()
        return
    # On the route per FOV the cell key and the rule before it, (fov_id, cell_id), give every molecule one cell.
    assert [None if pd.isna(v) else int(v) for v in links] == old_rule_links(result)


def test_so3_cell_links_refuses_a_key_the_cell_table_does_not_hold():
    from starfinder.export._spatialdata import cell_links
    result, _ = build("big_ids")
    molecules = result.molecules.copy()
    row = int(np.flatnonzero(molecules.cell_id.notna().to_numpy())[0])
    molecules.loc[row, KEY] = '["data","sample","Position009",null,"cell"]'
    broken = without_key(result)
    object.__setattr__(broken, "molecules", molecules)
    with pytest.raises(ValueError, match="of namespace .*Position009.*which the sample cell table does not hold"):
        cell_links(broken)


@pytest.mark.parametrize("case", ["golden_3d", "big_ids", "own_cases", "status_placeholder_with_input"])
def test_so3_identities_link_by_the_cell_key(case):
    result, _ = build(case)
    molecules = result.molecules
    in_cell = molecules.assignment_status.isin(["assigned", "excluded_cell"]).fillna(False).to_numpy(dtype=bool)
    row = int(np.flatnonzero(in_cell)[0])
    # Another namespace: no cell row holds the key.
    other = molecules.copy()
    other.loc[row, KEY] = '["data","sample","elsewhere",null,"cell"]'
    with pytest.raises(ValueError, match="identity 3: every assigned or excluded_cell molecule's cell is a row"):
        dataclasses.replace(result, molecules=other)
    # A namespace without a cell_id, and a cell_id without a namespace.
    for column, value in ((KEY, None), ("cell_id", None)):
        broken = molecules.copy()
        target = row if column == KEY else int(np.flatnonzero(molecules.cell_id.isna().to_numpy())[0])
        if column == "cell_id":
            broken.loc[target, KEY] = molecules[KEY].dropna().iloc[0]
        else:
            broken.loc[target, KEY] = value
        with pytest.raises(ValueError, match="identity 3: a molecule has a cell_namespace exactly when"):
            dataclasses.replace(result, molecules=broken)
    # The column absent.
    with pytest.raises(ValueError, match="identity 3: .*no column cell_namespace"):
        dataclasses.replace(result, molecules=molecules.drop(columns=KEY))


def test_so3_identities_on_a_molecules_only_table():
    """A molecules_only table holds the column, null on every row: a namespace on a row, or no column, fails."""
    result, _ = build("empty_molecules_only")
    named = result.molecules.copy()
    named.loc[0, KEY] = '["data","sample","Position001",null,"cell"]'
    with pytest.raises(ValueError, match="identity 3: a molecule has a cell_namespace exactly when"):
        dataclasses.replace(result, molecules=named)
    with pytest.raises(ValueError, match="identity 3: .*no column cell_namespace"):
        dataclasses.replace(result, molecules=result.molecules.drop(columns=KEY))


def test_so3_identity_4_counts_a_molecule_in_the_cell_of_its_key():
    """Identity 4 tallies the retained assigned molecules by the cell key: moving one molecule's key to another
    counted cell of another FOV, with the same cell_id and status, breaks the counts though identity 3 holds."""
    result, _ = build("golden_3d")
    cells, molecules = result.cells, result.molecules
    counted = cells[(cells.assembly_status == "retained") & (cells.status == "kept")]
    pairs = counted.groupby("cell_id").cell_namespace.apply(list)
    cell_id, (first, second) = next((int(c), ns[:2]) for c, ns in pairs.items() if len(ns) >= 2)
    rows = np.flatnonzero(((molecules[KEY] == first) & (molecules.cell_id == cell_id)
                           & (molecules.assignment_status == "assigned")
                           & (molecules.assembly_status == "retained")).fillna(False).to_numpy(dtype=bool))
    moved = molecules.copy()
    moved.loc[int(rows[0]), KEY] = second
    with pytest.raises(ValueError, match="identity 4"):
        dataclasses.replace(result, molecules=moved)


@pytest.mark.parametrize("case", ["golden_3d", "big_ids", "own_cases", "status_placeholder_with_input",
                                  "empty_molecules_only"])
def test_so3_readers_that_do_not_read_the_key_give_the_same_results(case, tmp_path):
    """write_sample_h5ad, export_counts and fov_ownership on the table with the column and on the table before it."""
    anndata = pytest.importorskip("anndata", reason="the anndata extra (starfinder[anndata]) is not installed")
    from starfinder.assembly import fov_ownership, write_sample_h5ad
    from starfinder.export._spatialdata import export_counts
    result, _ = build(case)
    before = without_key(result)
    assert export_counts(result) == export_counts(before)
    now = anndata.read_h5ad(write_sample_h5ad(result, tmp_path / "now.h5ad"))
    then = anndata.read_h5ad(write_sample_h5ad(before, tmp_path / "then.h5ad"))
    assert_frame_equal(now.obs, then.obs, check_exact=True)
    assert_frame_equal(now.var, then.var, check_exact=True)
    assert np.array_equal(now.X, then.X, equal_nan=True) and dict(now.uns) == dict(then.uns)
    assert set(now.layers) == set(then.layers)
    if case.startswith("own_cases"):
        placement = own_placement()
    elif case == "status_placeholder_with_input":
        placement = golden_placement(statuses={"Position002": "placeholder"})
    else:
        placement = golden_placement()
    for fov in result.fovs.fov_id:
        a, b = fov_ownership(result, fov, placement=placement), fov_ownership(before, fov, placement=placement)
        assert np.array_equal(a.region, b.region) and a.excluded_cells == b.excluded_cells
        assert a.excluded_molecules == b.excluded_molecules


@pytest.mark.parametrize("assignment", ["default", None])
def test_so3_the_export_readers_on_a_folder_with_the_column(assignment, tmp_path):
    """Dataset.export of assembly_golden with the SpatialData store, with cells and molecules_only
    (assignment=None, the column null on every row): check_export passes S1 to S8 with the column (S2 compares
    it), the points carry it, cell_instance_id follows the cell key, and lookup_source finds a molecule (and a
    cell) as before."""
    import importlib.util
    for name, extra in (("multiview_stitcher", "multiview-stitcher"), ("spatialdata", "spatialdata"),
                        ("anndata", "anndata")):
        if importlib.util.find_spec(name) is None:
            pytest.skip(f"the {extra} extra (starfinder[{extra}]) is not installed")
    import dask
    import spatialdata

    from starfinder.assignment._persist import read_assignment
    from starfinder.export import ExportConfig, SpatialDataExportConfig, check_export, lookup_source
    from starfinder.export._spatialdata import cell_links

    from .assembly_fixtures import SAMPLE
    from .fusion_fixtures import export_folder
    f = export_folder(tmp_path)
    with dask.config.set(scheduler="synchronous"):
        f.dataset.export(SAMPLE, placement=f.placement_dir, output_dir=f.export_dir, assignment=assignment,
                         config=ExportConfig(route="per_fov", export_spatialdata=True,
                                             spatialdata=SpatialDataExportConfig()))
        check_export(f.export_dir)
        assembly = load_assembly(f.export_dir)
        assert json.loads((f.export_dir / "export.json").read_text())["assembly"]["files"]["molecules"][
            "dtypes"][KEY] == "string"
        with pd.option_context("future.infer_string", False):
            points = spatialdata.read_zarr(f.export_dir / "sample.zarr").points["molecules"].compute()
    assert assembly.mode == ("cells" if assignment else "molecules_only")
    points = points.set_index(["spot_namespace", "spot_id"])
    source = assembly.molecules.set_index(["spot_namespace", "spot_id"]).loc[points.index]
    assert [None if pd.isna(v) else v for v in points[KEY]] == [None if pd.isna(v) else v for v in source[KEY]]
    links = pd.Series(cell_links(assembly), index=assembly.molecules.set_index(["spot_namespace", "spot_id"]).index)
    assert [None if pd.isna(v) else int(v) for v in points["cell_instance_id"]] == \
        [None if pd.isna(v) else int(v) for v in links.loc[points.index]]
    fov = "Position002"
    if assignment is None:
        assert assembly.molecules[KEY].isna().all() and points[KEY].isna().all() and links.isna().all()
        rows = assembly.molecules[assembly.molecules.fov_id == fov]
        found = lookup_source(f.export_dir, spot=(rows.spot_namespace.iloc[0], rows.spot_id.iloc[0]))
        assert (found["fov_id"], found["row"], found["assignment_folder"]) == (fov, 0, None)
        return
    assert [None if pd.isna(v) else int(v) for v in links] == old_rule_links(assembly)
    loaded = read_assignment(f.assignment_dir(fov), "default")
    row = loaded.molecules.iloc[0]
    found = lookup_source(f.export_dir, spot=(row.spot_namespace, row.spot_id))
    assert (found["fov_id"], found["row"], found["assignment_folder"]) == (fov, 0, str(f.assignment_dir(fov)))
    found = lookup_source(f.export_dir, cell=(loaded.cell_namespace, int(loaded.cells.cell_id.iloc[-1])))
    assert (found["fov_id"], found["row"]) == (fov, len(loaded.cells) - 1)
