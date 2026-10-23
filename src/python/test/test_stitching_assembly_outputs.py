"""AS11 to AS14 of docs/stitching-assembly-validation.md: the sample-level AnnData, persistence in the
export folder, ``write_h5ad`` and the legacy adapter, and determinism.

Fixtures: ``assembly_golden`` (3D with an expanded FOV, 2D, molecules only), ``own_cases`` (a
retained ``excluded_no_nucleus`` cell; FOV1 with nuclei, FOV2 without), an annotation row, the
§2.9 boxes and golden assignments of ``test_assignment.py``; the three single-thread
processes of AS14 are ``slow``. ``frozen_write_raw_h5ad`` is the adapter's ``_write_raw_h5ad``
as it was before ``write_h5ad`` was factored out (W-366 revision), kept as the reference.
"""
import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import starfinder.assembly._persist as persist_module
from starfinder.assembly import AssemblyConfig, AssemblyInput, assemble, load_assembly, save_assembly, write_sample_h5ad
from starfinder.assignment import AssignmentConfig, assign_molecules, write_h5ad
from starfinder.assignment._persist import write_assignment
from starfinder._snakemake import _cell_obs, _in_box, _write_raw_h5ad
from starfinder.segmentation import ExpandLabelsConfig

from .assembly_fixtures import (GOLDEN_IDS, TILE_ZYX, golden_molecule_tables, golden_placement, golden_results,
                                inputs_of, own_inputs, own_placement, unit_grid, with_record)

pytestmark = [pytest.mark.stitching, pytest.mark.validation]

ROOT = Path(__file__).resolve().parents[1]
COORDINATES = [f"sample_{p}_{a}" for p in ("centroid", "expanded_centroid", "ownership_centroid") for a in "zyx"]
ANNOTATION = {"sample_id": "sample1", "tissue": "lymph node", "section": 3}


def counted(result):
    cells = result.cells
    return cells[(cells.assembly_status == "retained") & (cells.status == "kept")].reset_index(drop=True)


def dense(result, rows, compartment):
    """cells × genes float64 from the sample counts, written out from the long table."""
    out = np.zeros((len(rows), len(result.genes)))
    where = {(ns, int(c)): i for i, (ns, c) in enumerate(zip(rows.cell_namespace, rows.cell_id))}
    for r in result.counts[result.counts.compartment == compartment].itertuples():
        out[where[(r.cell_namespace, int(r.cell_id))], result.genes.index(r.gene_id)] += r.count
    return out


def check_anndata(adata, result, *, annotation=None):
    """Every condition of "A test can check it against the sample tables" (docs/assembly-contract.md)."""
    rows = counted(result)
    assert adata.n_obs == len(rows)
    assert adata.obs_names.tolist() == rows.obs_name.tolist()
    obs = adata.obs.reset_index(drop=True)
    assert obs.fov_id.astype(str).tolist() == rows.fov_id.tolist()
    assert obs.cell_namespace.astype(str).tolist() == rows.cell_namespace.tolist()
    assert obs.cell_id.tolist() == rows.cell_id.astype(int).tolist()
    others = set(result.cells.obs_name) - set(rows.obs_name)
    assert not others & set(adata.obs_names)
    assert adata.X.dtype == np.float64 and isinstance(adata.X, np.ndarray)
    assert np.array_equal(adata.X, dense(result, rows, "whole"))
    if adata.layers:
        available = (rows.compartments == "available").to_numpy()
        for name in ("nucleus", "cytoplasm"):
            layer = adata.layers[name]
            assert layer.dtype == np.float64 and np.isnan(layer[~available]).all()
            assert np.array_equal(layer[available], dense(result, rows, name)[available])
    for column in COORDINATES:
        assert np.array_equal(obs[column].to_numpy(np.float64), rows[column].to_numpy(np.float64), equal_nan=True)
    assert obs.ownership_centroid.astype(str).tolist() == rows.ownership_centroid.tolist()
    spatial = adata.obsm["spatial"]
    assert spatial.shape == (len(rows), 2) and spatial.dtype == np.float64
    assert np.array_equal(spatial[:, 0], rows.sample_ownership_centroid_x.to_numpy())
    assert np.array_equal(spatial[:, 1], rows.sample_ownership_centroid_y.to_numpy())
    expanded = (rows.ownership_centroid == "expanded").to_numpy()
    assert np.array_equal(spatial[expanded], rows[["sample_expanded_centroid_x",
                                                   "sample_expanded_centroid_y"]].to_numpy()[expanded])
    assert np.array_equal(spatial[~expanded], rows[["sample_centroid_x", "sample_centroid_y"]].to_numpy()[~expanded])
    meta = json.loads(adata.uns["spatial_centroid"])
    assert meta == {"columns": ["sample_ownership_centroid_x", "sample_ownership_centroid_y"],
                    "centroid": result.record["config"]["centroid"],
                    "rows": {"expanded": int(expanded.sum()), "original": int((~expanded).sum())}}
    assert adata.uns["spatial_unit"] == "µm" and adata.uns["frame_id"] == f"sample:{result.sample}"
    assert json.loads(adata.uns["assembly"])["counts"] == result.record["counts"]
    assert adata.var_names.tolist() == list(result.genes)
    for name, value in (annotation or {}).items():
        assert obs[name].tolist() == [value] * len(rows)


# --- AS11 ----------------------------------------------------------------------------------------------------

def test_as11_golden_with_an_expanded_fov(tmp_path):
    anndata = pytest.importorskip("anndata")
    result = assemble(golden_placement(), inputs_of(golden_results(expand=("Position003",))))
    path = write_sample_h5ad(result, tmp_path / "out" / "sample.h5ad", annotation=ANNOTATION)
    assert path == tmp_path / "out" / "sample.h5ad" and path.is_file()
    adata = anndata.read_h5ad(path)
    check_anndata(adata, result, annotation=ANNOTATION)
    assert not adata.layers
    third = (counted(result).fov_id == "Position003").to_numpy()
    assert third.any() and (adata.obs.ownership_centroid.astype(str).to_numpy()[third] == "expanded").all()


def test_as11_own_cases_with_a_retained_excluded_no_nucleus_cell(tmp_path):
    anndata = pytest.importorskip("anndata")
    result = assemble(own_placement(), own_inputs())
    adata = anndata.read_h5ad(write_sample_h5ad(result, tmp_path / "sample.h5ad", annotation=pd.Series(ANNOTATION)))
    check_anndata(adata, result, annotation=ANNOTATION)
    assert "s1_FOV1_4" in set(result.cells.obs_name) and "s1_FOV1_4" not in set(adata.obs_names)
    assert set(adata.layers) == {"nucleus", "cytoplasm"}
    fov2 = (adata.obs.fov_id.astype(str) == "FOV2").to_numpy()
    assert np.isnan(adata.layers["nucleus"][fov2]).all() and not np.isnan(adata.layers["nucleus"][~fov2]).any()
    with pytest.raises(ValueError, match="already an obs column"):
        write_sample_h5ad(result, tmp_path / "again.h5ad", annotation={"fov_id": "x"})


def test_as11_and_as13_without_anndata(monkeypatch, tmp_path):
    result = assemble(own_placement(), own_inputs())
    monkeypatch.setitem(sys.modules, "anndata", None)
    with pytest.raises(ModuleNotFoundError, match="'anndata' extra"):
        write_sample_h5ad(result, tmp_path / "sample.h5ad")
    with pytest.raises(ModuleNotFoundError, match="'anndata' extra"):
        write_h5ad(tmp_path / "x.h5ad", pd.DataFrame(index=pd.Index([], dtype=object)),
                   pd.DataFrame({"cell_id": []}), pd.DataFrame({"cell_id": [], "gene_id": [], "compartment": [],
                                                                "count": []}), ["A"], layers=False, uns={})
    assert not list(tmp_path.iterdir())


# --- AS13: write_h5ad and the adapter ----------------------------------------------------------------------------

def frozen_write_raw_h5ad(result, path, *, sample, fov_id, tile):
    """_write_raw_h5ad (now in starfinder/_snakemake.py) before write_h5ad (the frozen reference)."""
    import anndata

    from starfinder._records import jsonable as _jsonable
    matrix, rows = result.matrix()
    layers = {}
    if result.nucleus_labels is not None:
        row_of = {int(c): i for i, c in enumerate(rows.cell_id)}
        for compartment in ('nucleus', 'cytoplasm'):
            values, available = result.matrix(compartment)
            layers[compartment] = np.full(matrix.shape, np.nan)
            layers[compartment][[row_of[int(c)] for c in available.cell_id], :] = values
    with pd.option_context('future.infer_string', False):
        obs = _cell_obs(result, rows, sample=sample, fov_id=fov_id, tile=tile)
        keep = _in_box(obs.fov_x.to_numpy(), obs.fov_y.to_numpy(), tile)
        layers = {name: layer[keep] for name, layer in layers.items()}
        obs = obs[keep]
        obs.index = pd.Index([str(i) for i in range(len(obs))], dtype=object)
        adata = anndata.AnnData(X=matrix[keep].astype(np.float64), obs=obs,
                                var=pd.DataFrame(index=pd.Index(list(result.genes), dtype=object)), layers=layers)
        adata.uns['assignment'] = json.dumps(_jsonable(result.record), default=str)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        adata.write(path)
    return obs.seg_label.to_numpy()


def a18_results():
    """The §2.9 results of A18's configurations: the golden labels in 3D and 2D, with and without expansion,
    and the boxes with nuclei (compartment layers)."""
    from .test_assignment import GOLDEN_GRID, assign_boxes, golden_inputs
    out = {}
    for dimensions in ("3d", "2d"):
        for expand in (False, True):
            mols, cells = golden_inputs(dimensions)
            config = AssignmentConfig(expansion=ExpandLabelsConfig(1.0, "pixel", "planar") if expand else None,
                                      legacy_pixel_expansion=expand)
            out[f"{dimensions}-{expand}"] = assign_molecules(mols, cells, grid=GOLDEN_GRID, config=config)
    out["boxes"] = assign_boxes(exclude_cells_without_nucleus=False)
    return out


@pytest.mark.parametrize("box", [(0, 64, 0, 64), (10, 40, 5, 50)])
def test_as13_raw_h5ad_equals_the_frozen_helper(tmp_path, box):
    anndata = pytest.importorskip("anndata")
    tile = pd.Series({"x": 3, "y": 4, "z": 0, "start_x_norm": box[0], "end_x_norm": box[1],
                      "start_y_norm": box[2], "end_y_norm": box[3]})
    for name, result in a18_results().items():
        new = _write_raw_h5ad(result, tmp_path / name / "raw.h5ad", sample="sample1", fov_id="Position001",
                              tile=tile)
        old = frozen_write_raw_h5ad(result, tmp_path / name / "frozen.h5ad", sample="sample1", fov_id="Position001",
                                    tile=tile)
        assert np.array_equal(new, old)
        a, b = anndata.read_h5ad(tmp_path / name / "raw.h5ad"), anndata.read_h5ad(tmp_path / name / "frozen.h5ad")
        assert_frame_equal(a.obs, b.obs, check_exact=True)
        assert_frame_equal(a.var, b.var, check_exact=True)
        assert a.X.dtype == b.X.dtype == np.float64 and np.array_equal(a.X, b.X)
        assert set(a.layers) == set(b.layers)
        for layer in a.layers:
            assert np.array_equal(a.layers[layer], b.layers[layer], equal_nan=True)
        assert dict(a.uns) == dict(b.uns) and list(a.obsm) == list(b.obsm) == []


def test_as13_write_h5ad_cases(tmp_path):
    anndata = pytest.importorskip("anndata")
    genes = ("A", "B", "C")
    empty_counts = pd.DataFrame({"cell_id": pd.array([], dtype="UInt32"), "gene_id": pd.array([], dtype="string"),
                                 "compartment": pd.array([], dtype="string"), "count": np.array([], np.int64)})
    no_row = pd.DataFrame({"compartments": pd.Series([], dtype=object)}, index=pd.Index([], dtype=object))
    for layers in (False, True):
        path = write_h5ad(tmp_path / f"deep/{layers}/empty.h5ad", no_row, pd.DataFrame({"cell_id": []}),
                          empty_counts, genes, layers=layers, uns={"k": "v"})
        adata = anndata.read_h5ad(path)
        assert path.is_file() and adata.X.shape == (0, 3) and adata.uns["k"] == "v"
        assert set(adata.layers) == ({"nucleus", "cytoplasm"} if layers else set())
        for layer in adata.layers.values():
            assert layer.shape == (0, 3)
    obs = pd.DataFrame({"compartments": ["available", "withheld"], "name": pd.array(["a", None], dtype="string")},
                       index=pd.Index(["c1", "c2"], dtype=object))
    counts = pd.DataFrame({"cell_id": pd.array([1, 1, 2, 1], dtype="UInt32"),
                           "gene_id": pd.array(["B", "B", "A", "C"], dtype="string"),
                           "compartment": pd.array(["whole", "nucleus", "whole", "cytoplasm"], dtype="string"),
                           "count": np.array([2, 2, 5, 0], np.int64)})
    var = pd.DataFrame({"n": np.array([1, 2, 3]), "label": pd.array(["x", "y", "z"], dtype="string")})
    path = write_h5ad(tmp_path / "two.h5ad", obs, pd.DataFrame({"cell_id": [1, 2]}), counts, genes, layers=True,
                      uns={}, var=var, obsm={"spatial": np.array([[1.0, 2.0], [3.0, 4.0]])})
    adata = anndata.read_h5ad(path)
    assert adata.X.tolist() == [[0.0, 2.0, 0.0], [5.0, 0.0, 0.0]]
    assert adata.layers["nucleus"][0].tolist() == [0.0, 2.0, 0.0] and np.isnan(adata.layers["nucleus"][1]).all()
    assert adata.var["n"].tolist() == [1, 2, 3] and adata.var["label"].astype(str).tolist() == ["x", "y", "z"]
    assert adata.var_names.tolist() == list(genes) and adata.obsm["spatial"].tolist() == [[1.0, 2.0], [3.0, 4.0]]
    with pytest.raises(ValueError, match="keys has 1 rows and obs 2"):
        write_h5ad(tmp_path / "bad.h5ad", obs, pd.DataFrame({"cell_id": [1]}), counts, genes, layers=False, uns={})
    with pytest.raises(ValueError, match=r"counts lacks the key columns \['cell_namespace'\]"):
        write_h5ad(tmp_path / "bad.h5ad", obs, pd.DataFrame({"cell_namespace": ["a", "b"], "cell_id": [1, 2]}),
                   counts, genes, layers=False, uns={})
    assert not (tmp_path / "bad.h5ad").exists()


# --- AS12: persistence ----------------------------------------------------------------------------------------

def saved_assignments(tmp_path, results):
    """Write each FOV's §2.9 folder <tmp>/<fov>/assignment/default; return fov -> folder."""
    folders = {}
    for fov, result in results.items():
        folder = tmp_path / "checkpoints" / fov / "assignment" / "default"
        write_assignment(folder, with_record(result, name="default"), metadata=unit_grid(TILE_ZYX).metadata)
        folders[fov] = folder
    return folders


def results_for_saving(tmp_path):
    golden = {d: assemble(golden_placement(), inputs_of(golden_results(d))) for d in ("3d", "2d")}
    only = assemble(golden_placement(), inputs_of(golden_molecule_tables(), kind="molecules"))
    return {"golden_3d": golden["3d"], "golden_2d": golden["2d"], "molecules_only": only}


@pytest.mark.parametrize("table_format", ["csv", "parquet"])
def test_as12_save_and_load(tmp_path, table_format):
    pytest.importorskip("anndata")
    for name, result in results_for_saving(tmp_path).items():
        folder = tmp_path / table_format / name
        folder.mkdir(parents=True)
        fusion = {"fusion": {"status": "off", "note": "written by another part", "values": [1.5, None]}}
        (folder / "export.json").write_text(json.dumps(fusion))
        saved = save_assembly(result, folder, table_format=table_format)
        names = sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file())
        assert names == sorted(["export.json", "sample.h5ad"] + [f"tables/{t}.{table_format}" for t in
                                                                  ("cells", "counts", "fovs", "molecules")])
        export = json.loads((folder / "export.json").read_text())
        assert json.dumps(export["fusion"], sort_keys=True) == json.dumps(fusion["fusion"], sort_keys=True)
        section = export["assembly"]
        assert section == saved.record and section["format_version"] == 1 and section["table_format"] == table_format
        for key, entry in section["files"].items():
            assert hashlib.sha256((folder / entry["path"]).read_bytes()).hexdigest() == entry["sha256"], key
        loaded = load_assembly(folder)
        for table in ("fovs", "cells", "molecules", "counts"):
            assert_frame_equal(getattr(loaded, table), getattr(result, table), check_exact=True)
        assert loaded.record == saved.record and loaded.genes == result.genes
        # A changed file raises naming the path and both hashes.
        path = folder / section["files"]["cells"]["path"]
        original = path.read_bytes()
        path.write_bytes(original + b"\n")
        recorded = section["files"]["cells"]["sha256"]
        with pytest.raises(ValueError, match=rf"{path}.*SHA-256 [0-9a-f]{{64}}, the record {recorded}"):
            load_assembly(folder)
        path.write_bytes(original)
        load_assembly(folder)


def test_as12_creates_the_record_and_replaces_only_its_section(tmp_path):
    pytest.importorskip("anndata")
    result = results_for_saving(tmp_path)["golden_3d"]
    folder = tmp_path / "export" / "sample1"
    save_assembly(result, folder)
    first = json.loads((folder / "export.json").read_text())
    assert list(first) == ["assembly"]
    save_assembly(result, folder, table_format="parquet")
    assert not list((folder / "tables").glob("*.csv"))
    assert json.loads((folder / "export.json").read_text())["assembly"]["table_format"] == "parquet"


def test_as12_the_record(tmp_path):
    pytest.importorskip("anndata")
    results = golden_results(expand=("Position003",))
    folders = saved_assignments(tmp_path, results)
    placement = golden_placement()
    by_path = assemble(placement, [AssemblyInput(f, unit_grid(TILE_ZYX), assignment=folders[f]) for f in GOLDEN_IDS])
    in_memory = assemble(placement, inputs_of(results))
    for table in ("fovs", "cells", "molecules", "counts"):
        assert_frame_equal(getattr(by_path, table), getattr(in_memory, table), check_exact=True)
    saved = save_assembly(by_path, tmp_path / "export" / "sample1")
    record = saved.record
    assert record["placement"]["method"] == "constructed" and record["placement"]["gauge"] == "G1"
    assert record["placement"]["review"] == "not_needed"
    from starfinder.stitching._placement import table_sha256
    assert record["placement"]["fovs_sha256"] == table_sha256(placement.fovs)
    assert record["placement"]["links_sha256"] == table_sha256(placement.links)
    assert record["config"] == {"ownership": "nearest_center", "centroid": "territory", "tie": "table_order",
                                "conflicts": True, "duplicate_threshold": 0.5}
    assert record["counts"] == by_path.record["counts"] and record["format_version"] == 1
    for entry, row in zip(record["fovs"], by_path.fovs.itertuples()):
        assert entry["fov_id"] == row.fov_id and entry["placement_status"] == "image_supported"
        assert entry["origin_zyx"] == [row.origin_z, row.origin_y, row.origin_x] and entry["kind"] == "assignment"
        assert entry["footprint_yx"] == [[row.footprint_y0, row.footprint_y1], [row.footprint_x0, row.footprint_x1]]
        assert entry["centre_yx"] == [row.centre_y, row.centre_x] and entry["owns_points"] is True
        assert entry["grid"]["shape_zyx"] == list(TILE_ZYX) and entry["grid"]["calibration_source"] == "declared"
        folder = folders[row.fov_id]
        assigned = json.loads((folder / "assignment.json").read_text())
        assert entry["assignment"]["name"] == "default"
        assert entry["assignment"]["record_sha256"] == hashlib.sha256(
            (folder / "assignment.json").read_bytes()).hexdigest()
        assert entry["assignment"]["tables"] == {t: assigned["files"][t]["sha256"]
                                                 for t in ("molecules", "cells", "counts")}
        assert entry["molecules_sha256"] == assigned["inputs"]["molecules"]["sha256"]
        territory = Path(entry["territories"]["path"])
        assert territory.is_absolute() and territory.parent == folder
        assert territory.name == ("territories.tif" if row.fov_id == "Position003" else "cell_labels.tif")
        assert hashlib.sha256(territory.read_bytes()).hexdigest() == entry["territories"]["sha256"]
    nulls = [assemble(placement, [AssemblyInput(f, unit_grid(TILE_ZYX), assignment=folders[f]) for f in GOLDEN_IDS],
                      config=AssemblyConfig(conflicts=False)),
             in_memory,
             assemble(placement, inputs_of(golden_molecule_tables(), kind="molecules"))]
    for result in nulls:
        assert all(e["territories"] == {"path": None, "sha256": None} for e in result.record["fovs"])
    assert all(e["assignment"]["source"] == "memory" for e in in_memory.record["fovs"])


def test_as12_an_interrupted_save_leaves_complete_or_absent_files(tmp_path, monkeypatch):
    pytest.importorskip("anndata")
    result = results_for_saving(tmp_path)["golden_3d"]
    folder = tmp_path / "export" / "sample1"
    real = persist_module._write_table
    calls = []

    def hook(frame, directory, name, table_format):
        calls.append(name)
        if len(calls) == 2:
            raise RuntimeError("interrupted")
        return real(frame, directory, name, table_format)

    monkeypatch.setattr(persist_module, "_write_table", hook)
    with pytest.raises(RuntimeError, match="interrupted"):
        save_assembly(result, folder)
    files = sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file())
    assert files == ["tables/fovs.csv"]
    from starfinder._records import _dtype_map, _read_table
    assert_frame_equal(_read_table(folder / "tables/fovs.csv", _dtype_map(result.fovs)), result.fovs,
                       check_exact=True)


# --- AS14: determinism ----------------------------------------------------------------------------------------

def table_hashes():
    from starfinder.stitching._placement import table_sha256
    result = assemble(golden_placement(), inputs_of(golden_results()))
    return {t: table_sha256(getattr(result, t)) for t in ("fovs", "cells", "molecules", "counts")}


def per_key(result):
    cells = {(r.cell_namespace, int(r.cell_id)): (r.assembly_status, r.assembly_reason, r.owner_fov_id,
                                                  r.assembly_flags, r.conflict_with)
             for r in result.cells.astype(object).itertuples()}
    mols = {(r.spot_namespace, r.spot_id): (r.assembly_status, r.assembly_reason, r.owner_fov_id, r.assembly_flags)
            for r in result.molecules.astype(object).itertuples()}
    return cells, mols


def test_as14_permuted_molecule_rows_give_the_same_per_key_results():
    rng = np.random.default_rng(210)
    reference = assemble(golden_placement(), inputs_of(golden_results()))
    tables = golden_molecule_tables()
    permuted_results = {}
    from .assembly_fixtures import golden_pieces, label_run
    grid = unit_grid(TILE_ZYX)
    for fov, (labels, _, _) in golden_pieces().items():
        table = tables[fov]
        order = rng.permutation(len(table.table))
        shuffled = replace(table, table=table.table.iloc[order].reset_index(drop=True))
        permuted_results[fov] = assign_molecules(shuffled, label_run(labels, fov, grid), grid=grid)
    permuted = assemble(golden_placement(), inputs_of(permuted_results))
    assert per_key(permuted) == per_key(reference)
    assert permuted.molecules.spot_id.tolist() != reference.molecules.spot_id.tolist()
    assert_frame_equal(permuted.counts, reference.counts, check_exact=True)


@pytest.mark.slow
def test_as14_three_single_thread_processes_give_identical_tables():
    script = ("import json; from test.test_stitching_assembly_outputs import table_hashes; "
              "print(json.dumps(table_hashes()))")
    env = {**__import__("os").environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "PYTHONHASHSEED": "random"}
    outputs = [subprocess.run([sys.executable, "-c", script], cwd=ROOT, env=env, capture_output=True, text=True,
                              check=True).stdout.strip().splitlines()[-1] for _ in range(3)]
    assert outputs[0] == outputs[1] == outputs[2]
    assert json.loads(outputs[0]) == table_hashes()
