"""SO3 of docs/sample-segmentation-validation.md: the column ``cell_namespace`` of the sample molecule
table on the route per FOV (choice 7, CK1 of docs/sample-segmentation-contract.md), and the readers that
link a molecule to its cell by the cell key ``(cell_namespace, cell_id)``.

Fixtures: the assemblies of docs/stitching-assembly-validation.md (``assembly_golden`` 3D, 2D and
expanded, ``own_cases`` with the duplicate rule on and off, ``empty_cases`` and ``status_cases``),
built by ``assembly_fixtures.py``, and ``big_ids``: ``assembly_golden`` (3D) with two label values at
and above 2^31. ``BASE_TABLES`` holds, per case, the ``table_sha256`` (column names, dtypes and CSV
text) of each of the four sample tables that ``assemble`` gave at the base revision 725f24a, before
the column existed; the tables after the change, the new column dropped, must give the same values.
"""
import dataclasses
import json

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from starfinder.assembly import AssemblyConfig, AssemblyResult, assemble, load_assembly, save_assembly
from starfinder.assignment._assign import table_sha256

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


def digests(result):
    """table_sha256 of each sample table without the column cell_namespace of the molecule table."""
    return {name: table_sha256(getattr(result, name).drop(columns=[KEY] if name == "molecules" else [],
                                                         errors="ignore"))
            for name in TABLES}


# table_sha256 of the four tables at the base revision 725f24a (computed there with build() and digests()).
BASE_TABLES = {
    "golden_3d": {
        "fovs": "2b1601cd186f6f23ca2f9329b2cbf2106673379a892670f7a8baaa941f3b951e",
        "cells": "fe40a12f89c7590b67eca5c7417451fbe6ac413eb74cab97f4fb559331c075c3",
        "molecules": "2abccb529e40af15438550389de8070f8d78309c68e159cdf53585bf0608e260",
        "counts": "e2dd4489c82f6a1a0dc11821d4b7085ed807ec76b2d018a515b6147eb467df96",
    },
    "golden_2d": {
        "fovs": "2b1601cd186f6f23ca2f9329b2cbf2106673379a892670f7a8baaa941f3b951e",
        "cells": "e6d7f0afb48c3839d3bf9fe390fce6c6ae85a930bbde89e87a705a4a2843377f",
        "molecules": "95d1af210e94df97caf0b1e6b777aa9bc5657340dd190f397d29849f1f152c3c",
        "counts": "d07491230dcee5bfe3c72928eb430403f34ca50228d9c94a3d597c51ba765a12",
    },
    "golden_expanded": {
        "fovs": "2b1601cd186f6f23ca2f9329b2cbf2106673379a892670f7a8baaa941f3b951e",
        "cells": "83e33bb8aecd1a090c9a11d763eb301153ded5b2faa64ab3dbb60ccc4436d48a",
        "molecules": "92e0cf229bd75eaf447ac4a84c11223cfc69ef9b9af31d504b3144c18cab0b6e",
        "counts": "e2dd4489c82f6a1a0dc11821d4b7085ed807ec76b2d018a515b6147eb467df96",
    },
    "big_ids": {
        "fovs": "2b1601cd186f6f23ca2f9329b2cbf2106673379a892670f7a8baaa941f3b951e",
        "cells": "5de8376886cac9296971e6746204a47cb9bec4af12e43fd9fa4f6350ac4e92c6",
        "molecules": "5ac7cb4f8102c3d79ab2aac07dafe46d91e1eda65d87451e5b1f985bc4269bad",
        "counts": "495684a98bb3e1e47cb921c1e5ad6bef660cd434f55f499ef132b79262670f78",
    },
    "own_cases": {
        "fovs": "972662080ff135f78a7399c25367f86a8250e4f9e31cb808c7e4200f99ac118d",
        "cells": "7e527fd765b8d435b1b9751ef760375da0b6da033346c9a0f1d52a329fbf899f",
        "molecules": "7c928b342796af9c2c8d557d9489dcd5e677143904c0de4b246ba9129dc22bdc",
        "counts": "4543ea63f8f923c3b613b1668f114c36bdbe05b67b1188d5cdf403f9bb61891a",
    },
    "own_cases_rule_off": {
        "fovs": "c1cb0da9840dd18244eebdac3d5535e7ba163263c819b153edc1b444e276a983",
        "cells": "5ea2016229863916e81b893e1bfd9eab8f6f422088a9f053389f34eaf9b8ad32",
        "molecules": "acfe663f787bb56b351442a45fe8b6a318398df35b35acfc872178838b791a4a",
        "counts": "642c7a632ce81136513c810ca95ddc6671efaa0b9b24c168baf99df739cc5bbd",
    },
    "empty_fov_without_molecules": {
        "fovs": "62853a24d04af8185964f2561a0a83277db2b68bf8406ad82c3d1a8cec1cc16c",
        "cells": "2343cdfc03a7f4952767e529aba810164b6c9014ff39176dd8359fe16eadb785",
        "molecules": "c6a919964b8bfc775357f1b0c2458efc7aae81e6a2f387d8170d028d6cced62c",
        "counts": "6e6ae625493a82467cbf46138a66b7725c485a975c81b29085bce909c5132343",
    },
    "empty_fov_without_cells": {
        "fovs": "43e2fc939d6a6bc6b408bcd1ae0f818091700adb9e97a42924619cdb88debd14",
        "cells": "cffef9b36e93dbdf633a953a3b0799c04d69d2c28cada71ae69569cf37c1cd0a",
        "molecules": "eff7235c4a436f1851fd4de511032cdac98c519a4498373ea32e83b85b3edf74",
        "counts": "ba04ecca8849fcbb1faae82365e6f7e14d877a1df1aadc6cf58fc08ce3a10bb9",
    },
    "empty_molecules_miss_every_cell": {
        "fovs": "6f1f2e2d15d4837e9812f037ffb620f069764568072ef100cb6e4cdbff2cefe7",
        "cells": "3996c5e14ba6e4ecf53722b884bde0fb96de5759bb42ae0bd1bd00f019b7dc61",
        "molecules": "5477c0c9a692610eb27a433cb3ca188fd12c8bd68569e6c6e4905fb69a1ce8a6",
        "counts": "244e39b75c2ea061ab47dc3fa294f0c1ae752300bc4f83bd131949b97691828c",
    },
    "empty_no_counted_cell": {
        "fovs": "2b1601cd186f6f23ca2f9329b2cbf2106673379a892670f7a8baaa941f3b951e",
        "cells": "d5cc7e0c0303c715ddd32ed47868d2a26147a9a0e614383a4d3f2689d22578a5",
        "molecules": "4abbfee69662d78e2e4815471ff7d446774a3476e4452482aeee47f39492f931",
        "counts": "14e4b4ab220435ee57f03f44a6e2e5c4636f9b288d40149b6b5a8999ff030aa2",
    },
    "empty_no_cell_row": {
        "fovs": "212abc3642e08fbeee48ca38ac3b6f123a051a92decc1a9ecf667f49ac621f0f",
        "cells": "64c3cd5b4651f83889426329abff8aa4aeb42a95c1e30f757fe545c8ce7d5bc8",
        "molecules": "cc9afd6be8c4595cb022808aa8d59c62cd8e2b93e69fbc14cd70804360c2d29f",
        "counts": "14e4b4ab220435ee57f03f44a6e2e5c4636f9b288d40149b6b5a8999ff030aa2",
    },
    "empty_no_molecule": {
        "fovs": "0518a9a17d56f2d9e28122e6590bc0d5b971c10cdf1d990d0e010340e65ab2c7",
        "cells": "59d18733cd32f8f82a90b71ab6867395b9b0dde62b7001363d407df6f2407153",
        "molecules": "557012c85576cfd5f8d72dc52a18949b5ebd75d06c5963cd403aab5a73efa480",
        "counts": "14e4b4ab220435ee57f03f44a6e2e5c4636f9b288d40149b6b5a8999ff030aa2",
    },
    "empty_molecules_only": {
        "fovs": "ed78dc1bce3ed34e1486e62f621f9d7a95da5cbe65821ad0efbf26262789b630",
        "cells": "64c3cd5b4651f83889426329abff8aa4aeb42a95c1e30f757fe545c8ce7d5bc8",
        "molecules": "2986c7be752def1ca06194d52c66e235c269d23bf359df6055a75852fd12aa93",
        "counts": "14e4b4ab220435ee57f03f44a6e2e5c4636f9b288d40149b6b5a8999ff030aa2",
    },
    "status_missing": {
        "fovs": "1b8887634ae3b5772232dc24e8a4b6dcd2a5428b82dae0e409a1d7aeb3d7f4c0",
        "cells": "6cba77303286bcd1fb12dbff8651b75a75dc0a0790e4ea6fd3a383d1376e484d",
        "molecules": "3511a89a8fa44bfd7fc3835c786b8c43856fdf6c8ef20adbd5b70786165609e1",
        "counts": "750996bc8c9216af3d0ae4e3430f636d0ac705abe2b94cf6cec42eb41b29be86",
    },
    "status_failed": {
        "fovs": "058790e798e466cd854dd4cbb268156b8b0ac11969f186ca638bd1e2f578b92e",
        "cells": "67f733d7aa6c8672271492d621b71d5e309c518b71eec96d59e608baebe2c08c",
        "molecules": "690694c89c721b427bbaa4164bb8efcee847f5b86513692fe3d283db51167399",
        "counts": "750996bc8c9216af3d0ae4e3430f636d0ac705abe2b94cf6cec42eb41b29be86",
    },
    "status_placeholder_without_input": {
        "fovs": "c3f6590ed33489ebf9f44381b92a55699f5ce2a681377f56bed0f67e79caf9d4",
        "cells": "2edb1846651a1d3e84186427e484b13ea43771d5e13237c4afa0254cce95e30d",
        "molecules": "23253ff725a6b692f7017aff821e41dfd1e9e8ad56f24aa5c4da44f0991fbd82",
        "counts": "0a18f5637fb1670a9d7540cd6adf712cde9c8b9266205bc13990a1c5e9820860",
    },
    "status_placeholder_with_input": {
        "fovs": "70eb182cb5a23f468d3459ae9fe5d6f29431aa62648f4144aa72aecd5705da09",
        "cells": "b3210c802205662ea1c27aed8ca3a229c0894a36ec8115cd1f0a80d900b2c3ad",
        "molecules": "ebd20ab298928846d08ee9d70c4270f64c14860ce23611f726cd69eb0fe6bef9",
        "counts": "0a18f5637fb1670a9d7540cd6adf712cde9c8b9266205bc13990a1c5e9820860",
    },
}

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
    # Every other column of every table equals the result before the change.
    assert digests(result) == BASE_TABLES[case]
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
    from starfinder.io._checkpoint import _read_table, _write_table, write_json
    from starfinder.segmentation._persist import file_sha256
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
    from starfinder.sample_output._spatialdata import cell_links
    result, _ = build(case)
    links = cell_links(result)
    if case == "empty_molecules_only":
        assert links.isna().all()
        return
    # On the route per FOV the cell key and the rule before it, (fov_id, cell_id), give every molecule one cell.
    assert [None if pd.isna(v) else int(v) for v in links] == old_rule_links(result)


def test_so3_cell_links_refuses_a_key_the_cell_table_does_not_hold():
    from starfinder.sample_output._spatialdata import cell_links
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
    """sample_anndata, export_counts and fov_ownership on the table with the column and on the table before it."""
    anndata = pytest.importorskip("anndata", reason="the anndata extra (starfinder[anndata]) is not installed")
    from starfinder.assembly import fov_ownership, sample_anndata
    from starfinder.sample_output._spatialdata import export_counts
    result, _ = build(case)
    before = without_key(result)
    assert export_counts(result) == export_counts(before)
    now = anndata.read_h5ad(sample_anndata(result, tmp_path / "now.h5ad"))
    then = anndata.read_h5ad(sample_anndata(before, tmp_path / "then.h5ad"))
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
    from starfinder.sample_output import ExportConfig, SpatialDataExportConfig, check_export, lookup_source
    from starfinder.sample_output._spatialdata import cell_links

    from .assembly_fixtures import SAMPLE
    from .fusion_fixtures import export_folder
    f = export_folder(tmp_path)
    with dask.config.set(scheduler="synchronous"):
        f.dataset.export(SAMPLE, placement=f.placement_dir, output_dir=f.export_dir, assignment=assignment,
                         config=ExportConfig(export_spatialdata=True, spatialdata=SpatialDataExportConfig()))
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
