"""The sample measurements of the report (W-455).

Checks of docs/report-algorithms.md ("Validation design"): DF7 (the reads assigned to cells
and the check against ``sample.h5ad``), DF8 (MECR), DF9 with RC7 (the per-cell values and
the unknown calibration), DF10 (the reads along Z and the grid-edge lines of their figure),
RC6 (no marker), and on the synthetic chains (seed 42, ``slow``) the marker figures on the
stack, RT2 (the sample measurements on both routes) and Z1's parts of the sample
measurements. The hand-written checks run on ``report_cells`` of ``report_fixtures.py`` and
on hand tables. Expected values are written here by hand or computed here from the fixtures'
own tables, read with the checkpoint table reader, and from ``X`` read with ``anndata``
(not the report's h5py reader), never from the report's own output.
"""
from __future__ import annotations

import io
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from starfinder.report import MarkerGroup, ReportConfig
from starfinder.report._collect import report_measurements
from starfinder.report._figures import (EXAMPLE_TILE, SMALL_PLOT, grid_edges, histogram_bins,
                                        reads_along_z_figure)
from starfinder.report._measurements import (AnnDataArrays, _Measurements, mecr, read_h5ad, reads_along_z,
                                             size_unit_of)
from starfinder.report._values import Unavailable

from . import hash_oracles
from .report_fixtures import (FOVS, GENES, cells_config, report_cells, report_records, write_assignment_folder,
                              write_export)
from .report_scene import scene_chain, scene_chain_plane  # noqa: F401 (module-scoped fixtures)
from .report_sizes import READS_Z_WIDTH

pytestmark = [pytest.mark.report, pytest.mark.validation]

TOL = 1e-9
SAMPLE_PARTS = ("sample.per_cell", "sample.markers", "sample.marker_zoom", "sample.mecr", "sample.reads_along_z")


def anndata_or_skip():
    return pytest.importorskip("anndata", reason="report_cells writes sample.h5ad, which needs the anndata extra")


def collect(case, **changes):
    folder, record = case.collect(**changes)
    tables = {Path(e["path"]).stem: hash_oracles.read_table(Path(folder) / e["path"], e["dtypes"])
              for e in record["files"] if "dtypes" in e}
    return folder, record, tables


def keyed(frame):
    return [tuple(None if pd.isna(v) else v for v in r)
            for r in frame[["part", "table", "column", "row_key", "reason"]].itertuples(index=False)]


def figures(case, **changes):
    arguments = dict(sample=case.sample, fov_ids=case.fov_ids, dataset_id=case.dataset_id, sample_id=case.sample_id,
                     config=case.config) | changes
    return report_measurements(case.root, **arguments)


def export_table(root, sample, name):
    """A table of the export, read with the oracle of the documented table format and the dtypes export.json records."""
    record = json.loads((Path(root) / "export" / sample / "export.json").read_text())
    entry = record["assembly"]["files"][name]
    return hash_oracles.read_table(Path(root) / "export" / sample / entry["path"], entry["dtypes"])


def matrix(root, sample):
    """``sample.h5ad`` read with anndata: obs names, obs, var names and the dense ``X``."""
    anndata = anndata_or_skip()
    adata = anndata.read_h5ad(Path(root) / "export" / sample / "sample.h5ad")
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    return list(adata.obs_names), adata.obs, list(adata.var_names), np.asarray(X, dtype=np.float64)


def decoded(figure):
    from matplotlib.image import imread
    return imread(io.BytesIO(figure.data), format=figure.format).shape[:2]


def expected_reads_in_cells(molecules, cells, fovs):
    """Per FOV the retained ``assigned`` molecules whose ``(cell_namespace, cell_id)`` is a counted cell (retained
    and kept), counted row by row."""
    counted = cells[cells["assembly_status"].eq("retained").to_numpy(bool) & cells["status"].eq("kept").to_numpy(bool)]
    keys = {(ns, int(i)) for ns, i in zip(counted["cell_namespace"], counted["cell_id"])}
    out = {f: 0 for f in fovs}
    for row in molecules.itertuples(index=False):
        if row.assembly_status == "retained" and row.assignment_status == "assigned" and not pd.isna(row.cell_id) \
                and (row.cell_namespace, int(row.cell_id)) in keys:
            out[row.fov_id] += 1
    return out


# --- DF7: the reads assigned to cells and the check against sample.h5ad -------------------------------------------

@pytest.mark.parametrize("variant", [None, "mismatch"])
def test_df7_reads_assigned_to_cells(tmp_path, variant):
    anndata_or_skip()
    case = report_cells(tmp_path / "root", variant)
    folder, record, tables = collect(case)
    molecules, cells = (export_table(case.root, case.sample, n) for n in ("molecules", "cells"))
    expected = expected_reads_in_cells(molecules, cells, FOVS)
    # By hand: F1's 20 kept cells hold one molecule each, F3's 7 two each, F4's 14 one each, F2 has no counted cell.
    assert expected == {"F1": 20, "F2": 0, "F3": 14, "F4": 14}
    # The two cells left out each hold two molecules: F1's retained excluded_no_nucleus cell (retained, assigned)
    # and F2's not_owner cell (its molecules excluded cell_not_owner).
    for selected in (cells["status"].eq("excluded_no_nucleus"), cells["assembly_reason"].eq("not_owner")):
        (cell,) = cells[selected.fillna(False).to_numpy(bool)].itertuples(index=False)
        own = molecules[molecules["cell_namespace"].eq(cell.cell_namespace).fillna(False).to_numpy(bool)
                        & molecules["cell_id"].eq(cell.cell_id).fillna(False).to_numpy(bool)]
        assert len(own) == 2 and own["assignment_status"].eq("assigned").all()
    f1 = molecules[molecules["fov_id"].eq("F1").to_numpy(bool)]
    assert int((f1["assembly_status"].eq("retained") & f1["assignment_status"].eq("assigned")).sum()) == 22
    fovs = tables["fovs"].set_index("fov_id")
    assert {f: int(fovs.loc[f, "reads_in_cells"]) for f in FOVS} == expected
    for fov in FOVS:
        assert fovs.loc[fov, "reads_in_cells_fraction"] == pytest.approx(
            expected[fov] / fovs.loc[fov, "n_molecules_retained"], abs=TOL)
    total = sum(expected.values())
    _, _, _, X = matrix(case.root, case.sample)
    whole = int(np.rint(X.sum()))
    assert whole - total == (1 if variant == "mismatch" else 0)
    totals = tables["readout_totals"].set_index("level")
    assert (int(totals.loc["reads_in_cells", "count"]), int(totals.loc["reads_in_cells", "fovs"])) == (48, 4)
    assert totals.loc["reads_in_cells", "fraction_of_spots"] == pytest.approx(48 / 4000, abs=TOL)
    assert not totals.loc["reads_in_cells", "overlaps_counted_per_fov"]
    population = tables["population"].set_index("population")
    assert (int(population.loc["counted", "cells"]), int(population.loc["counted", "molecules"])) == (X.shape[0], 48)
    summary = record["summary"]
    assert (summary["reads_in_cells"], summary["h5ad_whole_count_sum"], summary["reads_in_cells_equal"]) == (
        48, whole, variant != "mismatch")
    assert summary["counted_cells"] == X.shape[0] == 41
    # The difference is recorded and the report is written: no row names the check or a value of the cells (the
    # fixture fuses no image, so only the marker parts have rows).
    assert (folder / "report.json").is_file()
    assert [r for r in keyed(tables["unavailable"]) if r[0].startswith("sample.") or r[1] == "summary"] == [
        ("sample.markers", None, None, "", "part_not_written"),
        ("sample.marker_zoom", None, None, "", "part_not_written")]


def test_reads_in_cells_without_the_inputs(tmp_path):
    """Without sample.h5ad and with the molecule table changed, each value takes its input's reason."""
    anndata_or_skip()
    case = report_cells(tmp_path / "root")
    (case.root / "export/s1/sample.h5ad").unlink()
    path = case.root / "export/s1/tables/molecules.csv"
    path.write_bytes(path.read_bytes() + b"\n")
    _, record, tables = collect(case)
    rows = keyed(tables["unavailable"])
    assert [r for r in rows if r[0].startswith("sample.")] == [
        ("sample.per_cell", None, None, "", "input_missing"), ("sample.markers", None, None, "", "part_not_written"),
        ("sample.marker_zoom", None, None, "", "part_not_written")] + [
        ("sample.mecr", "mecr", c, k, "input_missing") for c in ("cells_a", "cells_b", "both", "either", "mecr")
        for k in ("gene_a=G1;gene_b=G2", "gene_a=G4;gene_b=G5")] + [
        ("sample.reads_along_z", None, None, "", "input_changed")]
    assert [r for r in rows if r[1:3] == ("fovs", "reads_in_cells")] == [
        ("assembly.cells_per_fov", "fovs", "reads_in_cells", f"fov_id={f}", "input_changed") for f in FOVS]
    assert ("assembly.accounting", "population", "cells", "population=counted", "input_missing") in rows
    assert ("assembly.accounting", "population", "molecules", "population=counted", "input_changed") in rows
    assert [(k, v) for k, v in record["summary"].items() if v is None] == [
        (k, None) for k in ("counted_cells", "reads_per_cell_median", "genes_per_cell_median", "reads_in_cells",
                            "h5ad_whole_count_sum", "reads_in_cells_equal")]
    assert {r[2]: r[4] for r in rows if r[1] == "summary"} == {
        "counted_cells": "input_missing", "reads_per_cell_median": "input_missing",
        "genes_per_cell_median": "input_missing", "reads_in_cells": "input_changed",
        "h5ad_whole_count_sum": "input_missing", "reads_in_cells_equal": "input_missing"}
    assert tables["cells"].empty and tables["reads_along_z"].empty
    kinds = {e["path"]: (e["kind"], e["verified"]) for e in record["inputs"]}
    assert kinds["export/s1/tables/molecules.csv"] == ("export_table", True)
    assert "export/s1/sample.h5ad" not in kinds


def drop_cell_namespace(case):
    """The export of ``case`` as one saved before the molecule table had ``cell_namespace``: the table written again
    without the column by the checkpoint writer, and its entry in export.json given the new SHA-256 and dtypes."""
    from starfinder._records import _write_table
    from starfinder.export._record import update_section
    from starfinder._records import file_sha256
    folder = case.root / "export" / case.sample
    section = json.loads((folder / "export.json").read_text())["assembly"]
    molecules = export_table(case.root, case.sample, "molecules").drop(columns="cell_namespace")
    name, dtypes = _write_table(molecules, folder / "tables", "molecules", "csv")
    section["files"]["molecules"] = {"path": f"tables/{name}", "sha256": file_sha256(folder / "tables" / name),
                                     "dtypes": dtypes}
    update_section(folder, case.sample, "assemble", section, status="written")


def test_an_export_without_cell_namespace(tmp_path):
    """An export whose molecule table has no cell_namespace makes the report of the same export with the column."""
    anndata_or_skip()
    case, older = report_cells(tmp_path / "root"), report_cells(tmp_path / "older")
    drop_cell_namespace(older)
    header = (older.root / "export/s1/tables/molecules.csv").read_text().splitlines()[0].split(",")
    assert "cell_namespace" not in header and "cell_id" in header
    (_, record, tables), (_, older_record, older_tables) = collect(case), collect(older)
    assert list(older_tables) == list(tables)
    for name, frame in tables.items():
        pd.testing.assert_frame_equal(older_tables[name], frame, check_exact=False, atol=TOL, rtol=0)
    assert older_record["summary"] == record["summary"]
    fovs = older_tables["fovs"].set_index("fov_id")
    assert {f: int(fovs.loc[f, "reads_in_cells"]) for f in FOVS} == {"F1": 20, "F2": 0, "F3": 14, "F4": 14}
    assert older_record["summary"]["reads_in_cells"] == 48 and older_record["summary"]["reads_in_cells_equal"]
    assert len(older_tables["cells"]) == 41


# --- DF9 and RC7: the per-cell values and the unknown calibration -------------------------------------------------

def expected_cells(root, sample, units):
    names, obs, _, X = matrix(root, sample)
    physical = obs["size_physical"].to_numpy(np.float64)
    rows = []
    for i, name in enumerate(names):
        finite = bool(np.isfinite(physical[i]))
        fov = str(obs["fov_id"].iloc[i])
        rows.append({"obs_name": name, "fov_id": fov, "reads": int(np.rint(X[i].sum())), "genes": int((X[i] > 0).sum()),
                     "size": float(physical[i]) if finite else float(obs["size_voxels"].iloc[i]),
                     "size_unit": units[fov] if finite else "voxels"})
    dtypes = {"obs_name": "string", "fov_id": "string", "reads": "int64", "genes": "int64", "size": "float64",
              "size_unit": "string"}
    return pd.DataFrame(rows, columns=list(dtypes)).astype(dtypes)


@pytest.mark.parametrize("variant", [None, "nocal"])
def test_df9_rc7_cells_and_the_unknown_calibration(tmp_path, variant):
    anndata_or_skip()
    case = report_cells(tmp_path / "root", variant)
    _, record, tables = collect(case)
    # The fixture's cell runs are volumes (assignment.json inputs.cells.geometry), so a finite size is µm³.
    expected = expected_cells(case.root, case.sample, {f: "µm³" for f in FOVS})
    pd.testing.assert_frame_equal(tables["cells"], expected, check_exact=True)
    if variant == "nocal":
        assert tables["cells"]["size_unit"].eq("voxels").all()
        assert tables["cells"]["size"].tolist() == [8.0] * 41
    assert record["summary"]["reads_per_cell_median"] == float(np.median(expected["reads"]))
    assert record["summary"]["genes_per_cell_median"] == float(np.median(expected["genes"]))
    drawn = figures(case)
    names = [f.tiles[0]["name"] for f in drawn.per_cell]
    assert names == ["reads", "genes", "size:voxels" if variant == "nocal" else "size:µm³"]
    size = drawn.per_cell[2].tiles[0]
    assert ("(voxels)" if variant == "nocal" else "(µm³)") in size["xlabel"]
    for figure in drawn.per_cell:
        assert (figure.width, figure.height) == SMALL_PLOT and decoded(figure) == SMALL_PLOT[::-1]
        assert figure.format == "png"
    reads = drawn.per_cell[0].tiles[0]
    assert reads["median"] == float(np.median(expected["reads"])) and "(molecules)" in reads["xlabel"]


def volume_and_plane(root):
    """report_cells whose F3 assignment record says its cells are a plane (``size_unit`` ``µm^2``), the others a
    volume (``µm^3``), the export written after the records so that it holds their SHA-256."""
    case = report_records(root)
    case.config = cells_config()
    for fov in FOVS:
        folder = write_assignment_folder(case, fov)
        record = json.loads((folder / "assignment.json").read_text())
        record.update(calibration="known", size_unit="µm^2" if fov == "F3" else "µm^3")
        (folder / "assignment.json").write_text(json.dumps(record, indent=2) + "\n")
    write_export(case)
    return case


def test_df9_a_volume_and_a_plane_fov(tmp_path):
    anndata_or_skip()
    case = volume_and_plane(tmp_path / "root")
    _, _, tables = collect(case)
    units = {"F1": "µm³", "F2": "µm³", "F3": "µm²", "F4": "µm³"}
    pd.testing.assert_frame_equal(tables["cells"], expected_cells(case.root, case.sample, units), check_exact=True)
    assert tables["cells"].groupby("fov_id")["size_unit"].unique().map(list).to_dict() == {
        "F1": ["µm³"], "F3": ["µm²"], "F4": ["µm³"]}
    drawn = figures(case)
    assert [f.tiles[0]["name"] for f in drawn.per_cell] == ["reads", "genes", "size:µm³", "size:µm²"]
    assert "(µm³)" in drawn.per_cell[2].tiles[0]["xlabel"] and "(µm²)" in drawn.per_cell[3].tiles[0]["xlabel"]
    assert drawn.per_cell[3].tiles[0]["median"] == float(np.median(tables["cells"]["size"][
        tables["cells"]["size_unit"].eq("µm²").to_numpy(bool)]))


def test_size_unit_of_the_records():
    assert size_unit_of({"size_unit": "µm^3"}) == "µm³" and size_unit_of({"size_unit": "µm^2"}) == "µm²"
    assert size_unit_of({"inputs": {"cells": {"geometry": "plane"}}}) == "µm²"
    assert size_unit_of({"inputs": {"cells": {"geometry": "volume"}}}, 1) == "µm³"
    assert size_unit_of(None, 1) == "µm²" and size_unit_of(None, 4) == "µm³"


def test_histogram_bins():
    # Integers 0 to 99: p99 = 98.01, width ceil(99.01 / 30) = 4, bins [0, 4) … [96, 100) for v ≤ 98.01, one above.
    bins = histogram_bins(list(range(100)), integer=True)
    assert bins["p99"] == pytest.approx(98.01, abs=TOL) and bins["width"] == 4.0
    assert len(bins["counts"]) == 25 and bins["counts"][:24] == [4] * 24 and bins["counts"][24] == 3
    assert bins["above"] == 1 and sum(bins["counts"]) + bins["above"] == 100
    # Small integers: width 1, one bin per value.
    bins = histogram_bins([1, 1, 2, 5], integer=True)
    assert bins["width"] == 1.0 and bins["counts"] == [0, 2, 1, 0, 0] and bins["above"] == 1
    # Floats: 30 equal bins over [0, p99], the last closed.
    values = np.linspace(0.0, 1.0, 101)
    bins = histogram_bins(values, integer=False)
    assert len(bins["counts"]) == 30 and bins["edges"][-1] == pytest.approx(0.99, abs=TOL)
    assert sum(bins["counts"]) == int((values <= bins["p99"]).sum()) and bins["above"] == 1
    assert histogram_bins([], integer=True)["counts"] == []


# --- DF8: MECR ----------------------------------------------------------------------------------------------------

HAND = {"A": (2, 1, 0, 0, 0, 0), "B": (0, 3, 1, 1, 0, 0), "C": (0,) * 6, "D": (0,) * 6}


def test_df8_mecr_hand_tables():
    assert mecr(HAND["A"], HAND["B"], 1) == {"cells_a": 2, "cells_b": 3, "both": 1, "either": 4, "mecr": 0.25}
    assert mecr(HAND["A"], HAND["B"], 2) == {"cells_a": 1, "cells_b": 1, "both": 0, "either": 2, "mecr": 0.0}
    assert mecr(HAND["A"], HAND["C"], 1) == {"cells_a": 2, "cells_b": 0, "both": 0, "either": 2, "mecr": 0.0}
    assert mecr(HAND["C"], HAND["D"], 1) == {"cells_a": 0, "cells_b": 0, "both": 0, "either": 0,
                                             "mecr": Unavailable("undefined", "either")}
    # The rows of mecr.csv from a count matrix: one per pair in configuration order, the undefined pair alone null.
    arrays = AnnDataArrays(np.array([f"c{i}" for i in range(6)], dtype=object), {},
                           np.array(list(HAND), dtype=object), np.array([HAND[g] for g in HAND], float).T)
    config = ReportConfig(marker_pairs=(("A", "B"), ("A", "C"), ("C", "D")))
    measurements = _Measurements(SimpleNamespace(config=config))
    measurements._cache["anndata"] = arrays
    rows = measurements.mecr_rows()
    assert [(r["gene_a"], r["gene_b"], r["threshold"], r["cells_a"], r["cells_b"], r["both"], r["either"])
            for r in rows] == [("A", "B", 1, 2, 3, 1, 4), ("A", "C", 1, 2, 0, 0, 2), ("C", "D", 1, 0, 0, 0, 0)]
    assert [r["mecr"] for r in rows] == [0.25, 0.0, Unavailable("undefined", "either")]


def test_read_h5ad_dense_and_csr(tmp_path):
    """The h5py reader against anndata's own writer, for a dense and a CSR ``X`` with a categorical obs column."""
    anndata = anndata_or_skip()
    from scipy import sparse
    X = np.array([HAND[g] for g in HAND], float).T
    with pd.option_context("future.infer_string", False):  # object strings, as anndata writes them
        obs = pd.DataFrame({"fov_id": pd.Categorical(np.array(["F1", "F1", "F2", None, "F2", "F1"], dtype=object)),
                            "size_physical": [1.0, np.nan, 2.0, 3.0, 4.0, 5.0], "size_voxels": np.arange(6)},
                           index=pd.Index([f"c{i}" for i in range(6)], dtype=object))
        var = pd.DataFrame(index=pd.Index(list(HAND), dtype=object))
    for name, data in (("dense", X), ("csr", sparse.csr_matrix(X))):
        anndata.AnnData(X=data, obs=obs, var=var).write_h5ad(tmp_path / f"{name}.h5ad")
        arrays = read_h5ad(tmp_path / f"{name}.h5ad")
        assert list(arrays.obs_names) == list(obs.index) and list(arrays.var_names) == list(HAND)
        assert list(arrays.obs["fov_id"]) == ["F1", "F1", "F2", None, "F2", "F1"]
        np.testing.assert_array_equal(arrays.obs["size_physical"], obs["size_physical"].to_numpy())
        assert arrays.row_sums().tolist() == [2, 4, 1, 1, 0, 0] and arrays.row_genes().tolist() == [1, 2, 1, 1, 0, 0]
        assert arrays.total() == 8 and arrays.column("B").tolist() == list(HAND["B"])
        assert arrays.column("Z").tolist() == [0.0] * 6


def expected_mecr(root, sample, pairs, threshold):
    _, _, genes, X = matrix(root, sample)
    rows = []
    for a, b in pairs:
        pa, pb = X[:, genes.index(a)] >= threshold, X[:, genes.index(b)] >= threshold
        both, either = int((pa & pb).sum()), int((pa | pb).sum())
        rows.append({"gene_a": a, "gene_b": b, "threshold": threshold, "cells_a": int(pa.sum()),
                     "cells_b": int(pb.sum()), "both": both, "either": either,
                     "mecr": both / either if either else None})
    dtypes = {"gene_a": "string", "gene_b": "string", "threshold": "int64", "cells_a": "Int64", "cells_b": "Int64",
              "both": "Int64", "either": "Int64", "mecr": "float64"}
    return pd.DataFrame(rows, columns=list(dtypes)).astype(dtypes)


@pytest.mark.parametrize("threshold", [1, 100])
def test_df8_mecr_report_cells(tmp_path, threshold):
    anndata_or_skip()
    case = report_cells(tmp_path / "root")
    case.config = ReportConfig(markers=case.config.markers, marker_pairs=case.config.marker_pairs,
                               positive_threshold=threshold)
    _, _, tables = collect(case)
    pairs = [("G1", "G2"), ("G4", "G5")]
    pd.testing.assert_frame_equal(tables["mecr"], expected_mecr(case.root, case.sample, pairs, threshold),
                                  check_exact=True)
    rows = [r for r in keyed(tables["unavailable"]) if r[0] == "sample.mecr"]
    if threshold == 1:
        assert tables["mecr"]["mecr"].notna().all() and rows == []
    else:
        # No cell holds 100 molecules of a gene: the four counts are written (0) and MECR is undefined per pair.
        assert tables["mecr"][["cells_a", "cells_b", "both", "either"]].eq(0).all().all()
        assert rows == [("sample.mecr", "mecr", "mecr", f"gene_a={a};gene_b={b}", "undefined") for a, b in pairs]
        details = tables["unavailable"][tables["unavailable"]["part"].eq("sample.mecr")]["detail"].tolist()
        assert details == ["either", "either"]


# --- RC6: no marker -----------------------------------------------------------------------------------------------

def test_rc6_no_markers(tmp_path):
    anndata_or_skip()
    case = report_cells(tmp_path / "root", "no_markers")
    _, _, tables = collect(case)
    assert tables["mecr"].empty and list(tables["mecr"].columns) == ["gene_a", "gene_b", "threshold", "cells_a",
                                                                      "cells_b", "both", "either", "mecr"]
    rows = keyed(tables["unavailable"])
    assert [r for r in rows if r[0] == "sample.mecr"] == [("sample.mecr", None, None, "", "no_markers")]
    # The three genes with the most retained molecules, ties by the order of assembly.genes, counted here.
    molecules = export_table(case.root, case.sample, "molecules")
    counts = molecules[molecules["assembly_status"].eq("retained").to_numpy(bool)]["gene_id"].value_counts()
    top = sorted((g for g in GENES if g in counts), key=lambda g: (-int(counts[g]), GENES.index(g)))[:3]
    drawn = figures(case)
    assert [(g.name, g.title, g.genes) for g in drawn.groups] == [
        ("most_abundant", f"Most abundant genes (no markers supplied): {', '.join(top)}", tuple(top))]
    assert drawn.groups[0].colours == ("#E69F00", "#56B4E9", "#009E73")
    # report_cells exports no fused image: the figure is unavailable with the fuse part's status.
    assert drawn.markers == Unavailable("part_not_written", "fuse: off")
    assert [r for r in rows if r[0] in ("sample.markers", "sample.marker_zoom")] == [
        ("sample.markers", None, None, "", "part_not_written"),
        ("sample.marker_zoom", None, None, "", "part_not_written")]


# --- DF10: the reads along Z --------------------------------------------------------------------------------------

def test_df10_reads_along_z_report_cells(tmp_path):
    anndata_or_skip()
    case = report_cells(tmp_path / "root")
    _, _, tables = collect(case)
    molecules = export_table(case.root, case.sample, "molecules")
    plane = np.floor(molecules["z"].to_numpy(np.float64) + 0.5).astype(int)
    f1 = molecules["fov_id"].eq("F1").to_numpy(bool)
    # The written z values and their planes: −0.6 → −1, −0.5 → 0, 0.4999 → 0, 0.5 → 1, 3.5 → 4.
    for z, k in ((-0.6, -1), (-0.5, 0), (0.4999, 0), (0.5, 1), (3.5, 4)):
        assert int((molecules["z"].to_numpy() == z).sum()) == 1 and plane[molecules["z"].to_numpy() == z][0] == k
    others = molecules["z"][~molecules["z"].isin([-0.6, 3.5]).to_numpy(bool)]
    assert ((others >= -0.5) & (others < 3.5)).all()
    expected = []
    for fov in FOVS:
        own = plane[molecules["fov_id"].eq(fov).to_numpy(bool)]
        low, high = min(0, own.min()), max(3, own.max())
        expected += [{"fov_id": fov, "z": k, "reads": int((own == k).sum())} for k in range(low, high + 1)]
    frame = tables["reads_along_z"]
    pd.testing.assert_frame_equal(frame, pd.DataFrame(expected).astype({"fov_id": "string", "z": "int64",
                                                                          "reads": "int64"}), check_exact=True)
    rows = {f: frame[frame["fov_id"].eq(f).to_numpy(bool)] for f in FOVS}
    assert rows["F1"]["z"].tolist() == [-1, 0, 1, 2, 3, 4]
    assert rows["F1"].set_index("z").loc[[-1, 4], "reads"].tolist() == [1, 1]
    for fov in ("F2", "F3", "F4"):
        assert rows[fov]["z"].tolist() == [0, 1, 2, 3] and 0 in rows[fov]["reads"].tolist()
    for fov in FOVS:
        assert int(rows[fov]["reads"].sum()) == int(molecules["fov_id"].eq(fov).sum())
    assert int(rows["F1"]["reads"].sum()) == int(f1.sum())
    assert not frame.isna().any().any()
    # The figure: the columns −1 to 4 over all FOVs; each FOV's grid of 4 planes edged between −1 | 0 and 3 | 4.
    figure = figures(case).reads_along_z
    assert figure.width == READS_Z_WIDTH and decoded(figure) == (figure.height, READS_Z_WIDTH)
    for tile in figure.tiles:
        assert tile["columns"] == [-1, 0, 1, 2, 3, 4] and tile["edges"] == [[0, 1], [4, 5]]
    assert [t["rows"] for t in figure.tiles] == [[-1, 4], [0, 3], [0, 3], [0, 3]]


HAND_Z = {"A": (-1.2, 0.0, 0.3, 2.4, 2.6), "B": (2.0, 3.0), "C": ()}


def test_df10_reads_along_z_hand_tables():
    fov_of = [f for f, values in HAND_Z.items() for _ in values]
    z = [v for values in HAND_Z.values() for v in values]
    planes = {f: 1 for f in HAND_Z}
    rows = reads_along_z(list(HAND_Z), fov_of, z, planes)
    assert [(r["fov_id"], r["z"], r["reads"]) for r in rows] == [
        ("A", -1, 1), ("A", 0, 2), ("A", 1, 0), ("A", 2, 1), ("A", 3, 1),
        ("B", 0, 0), ("B", 1, 0), ("B", 2, 1), ("B", 3, 1), ("C", 0, 0)]
    assert {f: sum(r["reads"] for r in rows if r["fov_id"] == f) for f in HAND_Z} == {"A": 5, "B": 2, "C": 0}
    # The grid-edge lines (gate W-387): on grids of one plane and the columns −1 to 3, every FOV has both lines,
    # between the columns of z −1 and 0 (indices 0 | 1) and between those of 0 and 1 (indices 1 | 2).
    figure = reads_along_z_figure(rows, planes)
    assert [(t["fov_id"], t["columns"], t["edges"]) for t in figure.tiles] == [
        (f, [-1, 0, 1, 2, 3], [[0, 1], [1, 2]]) for f in HAND_Z]
    assert [t["rows"] for t in figure.tiles] == [[-1, 3], [0, 3], [0, 0]]
    assert decoded(figure) == (figure.height, READS_Z_WIDTH) and figure.format == "png"
    # Each line only where the figure has a column on both sides.
    assert grid_edges([0, 1, 2, 3], 4) == [] and grid_edges([0, 1, 2, 3, 4], 4) == [(3, 4)]
    assert grid_edges([-2, -1, 0], 1) == [(1, 2)] and grid_edges([-1, 0, 1], None) == []


# --- The marker figures on the stack, RT2 and Z1 (the synthetic chains, slow) -------------------------------------

def scene_config(root):
    """Two groups of the scene's genes and two pairs (CH3)."""
    genes = json.loads((Path(root) / "export/sample/export.json").read_text())["assembly"]["genes"]
    return ReportConfig(markers=(MarkerGroup("first", tuple(genes[:2])), MarkerGroup("second", tuple(genes[2:5]))),
                        marker_pairs=((genes[0], genes[1]), (genes[2], genes[3])))


def pixel_range(low, high, origin, spacing, size):
    start, stop = (int(math.floor((q - origin) / spacing + 0.5)) for q in (low, high))
    return max(0, min(size, start)), max(0, min(size, stop))


def chain_figures(chain, config):
    folder, record = chain.collect(config=config)
    tables = {Path(e["path"]).stem: hash_oracles.read_table(Path(folder) / e["path"], e["dtypes"])
              for e in record["files"] if "dtypes" in e}
    drawn = report_measurements(chain.root, chain.sample, chain.fov_ids, dataset_id=chain.dataset_id,
                                sample_id=chain.sample, config=config)
    return record, tables, drawn


def check_marker_figures(chain, config, record, drawn, *, stack):
    """Every marker figure decodes at its size; its title and genes are the group's; the whole view's points and
    each zoom-in region's are the molecules this test selects from molecules.csv, at the diameter of the rule."""
    export = json.loads((chain.root / "export/sample/export.json").read_text())
    outputs = [o for o in export["fusion"]["outputs"] if o["image"] == export["fusion"]["outputs"][0]["image"]]
    projection = next((o for o in outputs if o["dimensionality"] == "projection"), None)
    stack_entry = next((o for o in outputs if o["dimensionality"] == "stack"), None)
    whole_entry = projection or stack_entry
    plane_entry = stack_entry if stack else whole_entry
    molecules = export_table(chain.root, chain.sample, "molecules")
    footprints = export_table(chain.root, chain.sample, "fovs").set_index("fov_id")
    retained = molecules["assembly_status"].eq("retained").to_numpy(bool)
    genes = molecules["gene_id"].astype(object).to_numpy()
    x, y, z = (molecules[f"sample_{a}"].to_numpy(np.float64) for a in "xyz")
    width = config.figure_width_px
    assert [m.group.name for m in drawn.markers] == [g.name for g in config.markers]
    for figure, group in zip(drawn.markers, config.markers):
        assert figure.group.title == group.name and figure.group.genes == group.genes
        whole = figure.whole
        height = decoded(whole)
        assert whole.width == width and height == (whole.height, width) and whole.format == "jpeg"
        tile = whole.tiles[0]
        assert tile["title"] == group.name and list(tile["genes"]) == list(group.genes)
        assert tile["genes"] == {g: int((retained & (genes == g)).sum()) for g in group.genes}
        p = whole_entry["shape_zyx"][2] * whole_entry["spacing_zyx"][2] / width
        assert tile["point_diameter_px"] == pytest.approx(max(2.0, 1.5 / p), abs=TOL)
        mine = np.isin(genes, group.genes)
        for k, role in enumerate(("example_1", "example_2", "example_3"), start=1):
            fov = record["examples"][role]["fov_id"]
            y0, y1, x0, x1 = (float(footprints.loc[fov, c]) for c in ("footprint_y0", "footprint_y1",
                                                                       "footprint_x0", "footprint_x1"))
            region = retained & (y >= y0) & (y < y1) & (x >= x0) & (x < x1)
            (_, _, ox), (_, _, sx) = plane_entry["origin_zyx"], plane_entry["spacing_zyx"]
            c0, c1 = pixel_range(x0, x1, ox, sx, plane_entry["shape_zyx"][2])
            p = (c1 - c0) * sx / EXAMPLE_TILE
            xy = figure.zooms[(role, "xy")]
            assert xy.width == EXAMPLE_TILE and decoded(xy) == (xy.height, EXAMPLE_TILE)
            assert xy.tiles[0]["point_diameter_px"] == pytest.approx(max(2.0, 1.5 / p), abs=TOL)
            if stack:
                centre = float(np.median(z[region]))
                chosen = region & mine & (z >= centre - 1.5) & (z < centre + 1.5)
                assert xy.tiles[0]["z_c"] == pytest.approx(centre, abs=TOL)
                assert xy.tiles[0]["slab"] == pytest.approx([centre - 1.5, centre + 1.5], abs=TOL)
            else:
                chosen = region & mine  # Z = 1 or a projection: every retained molecule of the region
                assert xy.tiles[0]["projection"] and xy.tiles[0]["slab"] is None
                assert xy.tiles[0]["title"] == f"Region {k} · {fov} · projection"
            assert xy.tiles[0]["rows"] == np.flatnonzero(chosen).tolist(), (role, group.name)
            xz = figure.zooms[(role, "xz")]
            if not stack:
                assert xz == Unavailable("no_stack", "the export has no fused stack of more than one plane")
                continue
            cy = (y0 + y1) / 2
            band = retained & mine & (y >= cy - 1.5) & (y < cy + 1.5) & (x >= x0) & (x < x1)
            assert xz.tiles[0]["rows"] == np.flatnonzero(band).tolist(), (role, group.name)
            assert xz.width == EXAMPLE_TILE and decoded(xz) == (xz.height, EXAMPLE_TILE)
            depth = stack_entry["shape_zyx"][0] * stack_entry["spacing_zyx"][0]
            stretch = max(1.0, ((c1 - c0) * sx / 6) / depth)
            assert xz.tiles[0]["stretch"] == pytest.approx(stretch, abs=TOL)
            assert xz.tiles[0]["title"] == f"Region {k} · {fov} · XZ, Z × {stretch:.3g}"
            assert xz.tiles[0]["point_diameter_px"] == pytest.approx(max(2.0, 1.5 / p), abs=TOL)


def check_cells_and_mecr(chain, config, tables, route):
    units = {f: "µm²" if route != "stack" else "µm³" for f in chain.fov_ids}
    pd.testing.assert_frame_equal(tables["cells"], expected_cells(chain.root, chain.sample, units), check_exact=True)
    pd.testing.assert_frame_equal(tables["mecr"], expected_mecr(chain.root, chain.sample, config.marker_pairs,
                                                                config.positive_threshold), check_exact=True)


@pytest.mark.slow
def test_markers_on_the_stack(scene_chain):  # noqa: F811
    config = scene_config(scene_chain.root)
    record, tables, drawn = chain_figures(scene_chain, config)
    assert not [r for r in keyed(tables["unavailable"]) if r[0].startswith("sample.")]
    check_marker_figures(scene_chain, config, record, drawn, stack=True)
    check_cells_and_mecr(scene_chain, config, tables, "stack")
    # The reads check: reads_in_cells, Σ X and, by identity 4, Σ n_molecules over the counted cells.
    _, _, _, X = matrix(scene_chain.root, scene_chain.sample)
    cells = export_table(scene_chain.root, scene_chain.sample, "cells")
    counted = cells["assembly_status"].eq("retained").to_numpy(bool) & cells["status"].eq("kept").to_numpy(bool)
    summary = record["summary"]
    assert summary["reads_in_cells"] == summary["h5ad_whole_count_sum"] == int(np.rint(X.sum())) == \
        int(cells["n_molecules"][counted].sum()) and summary["reads_in_cells_equal"] is True
    molecules = export_table(scene_chain.root, scene_chain.sample, "molecules")
    assert summary["reads_in_cells"] == sum(expected_reads_in_cells(molecules, cells, scene_chain.fov_ids).values())
    # Every molecule on its own plane; the grids hold 5 planes.
    along = tables["reads_along_z"]
    assert int(along["reads"].sum()) == len(molecules) and set(along["z"]) == set(range(5))
    # Without markers: one figure of the three genes with the most retained molecules.
    plain = report_measurements(scene_chain.root, scene_chain.sample, scene_chain.fov_ids,
                                dataset_id=scene_chain.dataset_id, sample_id=scene_chain.sample,
                                config=ReportConfig())
    genes = json.loads((scene_chain.root / "export/sample/export.json").read_text())["assembly"]["genes"]
    counts = molecules[molecules["assembly_status"].eq("retained").to_numpy(bool)]["gene_id"].value_counts()
    top = sorted((g for g in genes if g in counts), key=lambda g: (-int(counts[g]), genes.index(g)))[:3]
    (figure,) = plain.markers
    assert figure.group.title == figure.whole.tiles[0]["title"] == \
        f"Most abundant genes (no markers supplied): {', '.join(top)}"
    assert list(figure.whole.tiles[0]["genes"]) == top
    assert decoded(figure.whole) == (figure.whole.height, 1600)


@pytest.mark.slow
@pytest.mark.parametrize("route", ["per_fov", "after_fusion"])
def test_rt2_z1_sample_measurements_on_both_routes(scene_chain_plane, route):  # noqa: F811
    chain = scene_chain_plane[route]
    config = scene_config(chain.root)
    record, tables, drawn = chain_figures(chain, config)
    assert record["route"] == route
    check_cells_and_mecr(chain, config, tables, route)
    # Z = 1: one row per FOV, plane 0, holding all its molecules.
    molecules = export_table(chain.root, chain.sample, "molecules")
    along = tables["reads_along_z"]
    assert along[["fov_id", "z"]].values.tolist() == [[f, 0] for f in chain.fov_ids]
    assert along["reads"].tolist() == [int(molecules["fov_id"].eq(f).sum()) for f in chain.fov_ids]
    # The XZ zooms: three rows view=xz with no_stack, one per region; the XY zooms drawn.
    roles = {role: record["examples"][role]["fov_id"] for role in ("example_1", "example_2", "example_3")}
    rows = [r for r in keyed(tables["unavailable"]) if r[0].startswith("sample.")]
    assert rows == [("sample.marker_zoom", None, None, f"fov_id={fov};role={role};view=xz", "no_stack")
                    for role, fov in roles.items()]
    check_marker_figures(chain, config, record, drawn, stack=False)
    for figure in drawn.per_cell:
        assert decoded(figure) == SMALL_PLOT[::-1]
    assert decoded(drawn.reads_along_z) == (drawn.reads_along_z.height, READS_Z_WIDTH)
    assert [t["edges"] for t in drawn.reads_along_z.tiles] == [[]] * len(chain.fov_ids)
