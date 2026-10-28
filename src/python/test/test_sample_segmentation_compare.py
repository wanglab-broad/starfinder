"""CR1 to CR5 of design/validation/sample-segmentation.md: ``compare_routes``, the consistency check of the two routes,
and that its measure is the duplicate rule's.

Fixtures (``sample_segmentation_fixtures.py``): ``fused_pair``, ``fused_pair_seams``, ``fused_quad``, ``scene_plane``
and (``slow``) ``scene_plane_small``, each exported on both routes by ``build`` of
``test_sample_segmentation_route.py`` (W-427), imported, not rebuilt: the route after fusion with ``truth_import`` at
``k = 1`` into ``export/<sample>/`` and the route per FOV (``per_fov_routes``) into ``export_per_fov/<sample>/``;
``fused_pair`` also at ``k = 2``, where the segmentation grid's spacing differs from a FOV's. The truth cell of a cell
is its ``cell_id`` (both routes import the truth labels with ``relabel: false``). No model is loaded or run; dask runs
with the synchronous scheduler.
"""
import dataclasses
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from starfinder.assembly import load_assembly
from starfinder.export import RouteComparison, compare_routes
from starfinder.stitching import approve_placement, placement_from_table, positions_from_table, save_placement

from .sample_segmentation_fixtures import (
    BIG,
    SAMPLE,
    SceneFolder,
    per_fov_routes,
    route_config,
    seam_spot_ids,
    write_folder,
)
from .test_sample_segmentation_route import build, export, export_grid, per_fov_dir, seam_cells, truth_config

pytestmark = [pytest.mark.stitching, pytest.mark.validation]
needs_library = pytest.mark.skipif(importlib.util.find_spec("multiview_stitcher") is None,
                                   reason="the multiview-stitcher extra (starfinder[multiview-stitcher]) is not "
                                          "installed")
needs_anndata = pytest.mark.skipif(importlib.util.find_spec("anndata") is None,
                                   reason="the anndata extra (starfinder[anndata]) is not installed")
CR_SCENES = ["fused_pair", "fused_pair_k2", "fused_pair_seams", "fused_quad", "scene_plane"]
CATEGORIES = ("same_cell", "different_cells", "cell_on_one_route", "no_cell_on_both", "retained_on_one_route",
              "excluded_on_both")
#: The counts of the summary that depend on the threshold (CR2); every other entry does not.
THRESHOLD_COUNTS = {"matched_pairs", "cells_in_no_matched_pair", "after_fusion_cells_sharing_with_several",
                    "per_fov_cells_sharing_with_several"}


@pytest.fixture(autouse=True)
def synchronous():
    import dask
    with dask.config.set(scheduler="synchronous"):
        yield


def build_k2(tmp_path, name):
    """``fused_pair`` on both routes, the route after fusion at ``k = 2`` (the truth imported on that grid)."""
    folder = write_folder(tmp_path, name)
    per_fov_routes(folder)
    export(folder, route_config(segmentation=truth_config(folder, export_grid(folder, 2)), k=2))
    export(folder, route_config(route="per_fov"), output_dir=per_fov_dir(folder))
    return SimpleNamespace(folder=folder, result=load_assembly(folder.export_dir),
                           per_fov_result=load_assembly(per_fov_dir(folder)))


@pytest.fixture(scope="module")
def cr(tmp_path_factory):
    """``cr(name)``: a fixture on both routes, built once for the module (the tests read the folders only)."""
    cache = {}

    def get(name):
        if name not in cache:
            tmp = tmp_path_factory.mktemp(name)
            cache[name] = build_k2(tmp, "fused_pair") if name == "fused_pair_k2" else build(tmp, name)
        return cache[name]
    return get


def folders(case):
    return per_fov_dir(case.folder), case.folder.export_dir


def digests(root) -> dict:
    """The SHA-256 of every file under a folder, by relative path."""
    root = Path(root)
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def cell_key(namespace, cell_id):
    return str(namespace), int(cell_id)


# --- CR1 ------------------------------------------------------------------------------------------------------

def cr1_conditions(case, report):
    """CR1's conditions on any fixture: every counted cell of the route per FOV that is the only observation of its
    truth cell is matched to that cell of the route after fusion; each molecule key has exactly one category, and
    the totals sum to the molecule rows of the two tables' key union."""
    per_fov, after = case.per_fov_result, case.result
    observations = per_fov.cells["cell_id"].astype("int64").value_counts()
    after_keys = {int(c): cell_key(ns, c) for ns, c in zip(after.cells["cell_namespace"], after.cells["cell_id"])}
    matched = report.pairs[report.pairs["matched"]]
    matched = {cell_key(a, b): cell_key(c, d) for a, b, c, d in zip(
        matched["per_fov_cell_namespace"], matched["per_fov_cell_id"], matched["after_fusion_cell_namespace"],
        matched["after_fusion_cell_id"])}
    only = [cell_key(ns, c) for ns, c in zip(per_fov.counted_cells()["cell_namespace"],
                                             per_fov.counted_cells()["cell_id"]) if observations[int(c)] == 1]
    # fused_pair_seams has no such cell: both of its cells are seen by both FOVs; its written outcome covers them.
    assert only or case.folder.scene.name == "fused_pair_seams", "a counted cell is the only observation of its cell"
    for key in only:
        assert matched.get(key) == after_keys[key[1]], (key, matched.get(key))
    keys = pd.MultiIndex.from_arrays([report.molecules["spot_namespace"], report.molecules["spot_id"]])
    union = pd.MultiIndex.from_arrays([per_fov.molecules["spot_namespace"], per_fov.molecules["spot_id"]]).union(
        pd.MultiIndex.from_arrays([after.molecules["spot_namespace"], after.molecules["spot_id"]]))
    assert not keys.has_duplicates and keys.sort_values().equals(union.sort_values())
    assert report.molecules["category"].isin(CATEGORIES).all()
    totals = report.summary["molecule_categories"]
    assert list(totals) == list(CATEGORIES)
    assert totals == {c: int((report.molecules["category"] == c).sum()) for c in CATEGORIES}
    assert sum(totals.values()) == len(union)
    return only, matched


def seams_outcome(case, report):
    """``fused_pair_seams``' written outcome, number by number."""
    cells = case.per_fov_result.cells.set_index(["fov_id", "cell_id"])
    assert cells.loc[("FOV1", 1), "assembly_status"] == "retained"
    assert cells.loc[("FOV2", 1), ["assembly_status", "assembly_reason", "duplicate_of"]].tolist() == [
        "excluded", "duplicate_observation", f"{SAMPLE}_FOV1_1"]
    assert cells.loc[("FOV1", 2), "assembly_status"] == cells.loc[("FOV2", 2), "assembly_status"] == "retained"
    ns = {fov: json.dumps(["data", SAMPLE, fov, None, "cell"], separators=(",", ":")) for fov in ("FOV1", "FOV2")}
    fused = json.dumps(["data", SAMPLE, None, None, "cell"], separators=(",", ":"))
    pairs = report.pairs
    assert list(zip(pairs["per_fov_cell_namespace"], pairs["per_fov_cell_id"], pairs["after_fusion_cell_id"],
                    pairs["share"], pairs["matched"])) == [
        (ns["FOV1"], 1, 1, 1.0, True), (ns["FOV1"], 2, 2, 1.0, True), (ns["FOV2"], 2, 2, 1.0, False)]
    assert (pairs["after_fusion_cell_namespace"] == fused).all()
    # The cells of the route after fusion: cell 1 of 78 pixels, cell 2 of 144; no row for FOV2's excluded cell 1.
    assert dict(zip(pairs["after_fusion_cell_id"], pairs["after_fusion_size"])) == {1: 78, 2: 144}
    summary = report.summary
    assert summary["cells"] == {"per_fov": 3, "after_fusion": 2}
    assert summary["matched_pairs"] == 2
    assert summary["cells_in_no_matched_pair"] == {"per_fov": 1, "after_fusion": 0}
    assert summary["after_fusion_cells_sharing_with_several"] == 1
    assert summary["per_fov_cells_sharing_with_several"] == 0
    assert report.counts[["per_fov_cell_namespace", "per_fov_cell_id", "after_fusion_cell_id", "per_fov_whole",
                          "after_fusion_whole", "gene_difference"]].values.tolist() == [
        [ns["FOV1"], 1, 1, 2, 3, 1], [ns["FOV1"], 2, 2, 2, 3, 1]]
    assert summary["molecule_categories"] == {"same_cell": 2, "different_cells": 2, "cell_on_one_route": 0,
                                              "no_cell_on_both": 1, "retained_on_one_route": 4,
                                              "excluded_on_both": 2}
    expected = {("FOV1", "m_a"): ("same_cell", None), ("FOV1", "m_d"): ("same_cell", None),
                ("FOV2", "m_e"): ("different_cells", None), ("FOV2", "m_f"): ("different_cells", None),
                ("FOV2", "m_g"): ("no_cell_on_both", None),
                ("FOV1", "m_b"): ("retained_on_one_route", "per_fov"),
                ("FOV1", "m_e"): ("retained_on_one_route", "per_fov"),
                ("FOV2", "m_b"): ("retained_on_one_route", "after_fusion"),
                ("FOV2", "m_c"): ("retained_on_one_route", "after_fusion"),
                ("FOV1", "m_g"): ("excluded_on_both", None), ("FOV2", "m_a"): ("excluded_on_both", None)}
    spot = seam_spot_ids()
    molecules = report.molecules.set_index(["fov_id", "spot_id"])
    assert len(molecules) == len(expected) == 11
    for (fov, name), (category, route) in expected.items():
        row = molecules.loc[(fov, spot[(fov, name)])]
        assert (row["category"], None if pd.isna(row["route"]) else row["route"]) == (category, route), (fov, name)


@needs_anndata
@needs_library
@pytest.mark.parametrize("scene", CR_SCENES)
def test_cr1_two_routes_from_one_truth(cr, scene):
    case = cr(scene)
    report = compare_routes(*folders(case))
    assert isinstance(report, RouteComparison)
    only, matched = cr1_conditions(case, report)
    if scene == "fused_pair_seams":
        seams_outcome(case, report)
    if scene == "fused_quad":
        # Identifiers at and above 2^31 come back whole: the cell at the crossing and the border cell.
        assert {BIG, 2 ** 32 - 1} <= set(report.pairs["after_fusion_cell_id"].astype("int64"))
        assert {BIG, 2 ** 32 - 1} <= {b[1] for b in matched.values()}
    if scene == "scene_plane":
        assert seam_cells(case.folder), "scene_plane has a cell across a seam"
    print(f"\nCR1 {scene}: {len(only)} counted cells of the route per FOV that are the only observation of their "
          f"truth cell, all matched; summary {json.dumps({k: v for k, v in report.summary.items() if k != 'share_bins'})}")


@needs_anndata
@needs_library
def test_cr1_the_placement_at_factor_2(cr):
    """At ``k = 2`` each counted cell of the route per FOV shows on the grid pixels ``v`` whose FOV pixel
    ``floor(c + ½)``, ``c = (O + v·S − origin) / s``, inside the FOV's voxel-centre extent, holds it in any plane;
    recomputed here pixel by pixel from the saved territories."""
    import tifffile
    case = cr("fused_pair_k2")
    report = compare_routes(*folders(case))
    record = json.loads((case.folder.export_dir / "export.json").read_text())
    grid = record["segmentation"]["inputs"]["grid"]
    (_, n_y, n_x), (_, sy, sx), (_, oy, ox) = grid["shape_zyx"], grid["spacing_zyx"], grid["origin_zyx"]
    assert (sy, sx) == (1.0, 1.0)
    fused = tifffile.imread(case.folder.export_dir / "segmentation_cell.ome.tif")
    entries = {e["fov_id"]: e for e in json.loads((per_fov_dir(case.folder) / "export.json").read_text())[
        "assembly"]["fovs"]}
    expected = {}
    for row in case.per_fov_result.counted_cells().itertuples(index=False):
        entry = entries[row.fov_id]
        territory = tifffile.imread(entry["territories"]["path"])
        n_fy, n_fx = territory.shape[-2:]
        shown = np.zeros((n_y, n_x), bool)
        for v in range(n_y):
            for u in range(n_x):
                cy = (oy + v * sy - entry["origin_zyx"][1]) / 0.5
                cx = (ox + u * sx - entry["origin_zyx"][2]) / 0.5
                if -1e-9 <= cy <= n_fy - 1 + 1e-9 and -1e-9 <= cx <= n_fx - 1 + 1e-9:
                    shown[v, u] = (territory[:, int(np.floor(cy + 0.5)), int(np.floor(cx + 0.5))] == row.cell_id).any()
        for cell in np.unique(fused[shown]).tolist():
            if cell:
                expected[(row.fov_id, int(row.cell_id), cell)] = (int(shown.sum()), int((fused[shown] == cell).sum()))
    got = {(name.split("_")[1], int(a), int(b)): (int(size), int(shared)) for name, a, b, size, shared in zip(
        report.pairs["per_fov_obs_name"], report.pairs["per_fov_cell_id"], report.pairs["after_fusion_cell_id"],
        report.pairs["per_fov_size"], report.pairs["shared"])}
    assert got == expected and got


# --- The measure is the duplicate rule's --------------------------------------------------------------------------

@needs_anndata
@needs_library
def test_cr_the_measure_is_the_duplicate_rule_s(cr, monkeypatch):
    """On the pairs of ``fused_pair_seams`` the share of ``compare_routes`` equals the share the duplicate rule of
    the assembly computes for the same two territories (``_duplicates.measure`` and ``_duplicates.share``, the
    pair placed as two FOVs on the sample grid at scale 1), and ``compare_routes`` takes every share from
    ``_duplicates.share``."""
    import tifffile

    from starfinder.assembly import _duplicates
    case = cr("fused_pair_seams")
    calls = []
    original = _duplicates.share

    def spy(sizes, pair, voxels):
        value = original(sizes, pair, voxels)
        calls.append((pair, voxels, value))
        return value

    monkeypatch.setattr(_duplicates, "share", spy)
    report = compare_routes(*folders(case))
    monkeypatch.undo()
    assert len(calls) == len(report.pairs) == 3
    record = json.loads((case.folder.export_dir / "export.json").read_text())
    grid = record["segmentation"]["inputs"]["grid"]
    fused = tifffile.imread(case.folder.export_dir / "segmentation_cell.ome.tif")[np.newaxis]
    entries = {e["fov_id"]: e for e in json.loads((per_fov_dir(case.folder) / "export.json").read_text())[
        "assembly"]["fovs"]}
    for row in report.pairs.itertuples(index=False):
        fov = row.per_fov_obs_name.split("_")[1]
        entry = entries[fov]
        territory = tifffile.imread(entry["territories"]["path"])
        items = [{"fov_id": fov, "origin": tuple(entry["origin_zyx"]), "shape": territory.shape,
                  "cells": np.array([row.per_fov_cell_id], np.int64)},
                 {"fov_id": "fused", "origin": tuple(grid["origin_zyx"]), "shape": fused.shape,
                  "cells": np.array([row.after_fusion_cell_id], np.int64)}]
        images = {fov: territory, "fused": fused}
        sizes, shared = _duplicates.measure(items, images.__getitem__, {
            "origin": (0.0, min(grid["origin_zyx"][1], entry["origin_zyx"][1]),
                       min(grid["origin_zyx"][2], entry["origin_zyx"][2])),
            "spacing": (1.0, 0.5, 0.5), "two_d": True})
        pair = ((fov, row.per_fov_cell_id), ("fused", row.after_fusion_cell_id))
        assert _duplicates.share(sizes, pair, shared[pair]) == row.share
        assert (sizes[pair[0]], sizes[pair[1]], shared[pair]) == (row.per_fov_size, row.after_fusion_size, row.shared)
    # The two observations of case high, as the duplicate rule measured them: 68 shared of 73, 0.9315.
    terr = {f: tifffile.imread(entries[f]["territories"]["path"]) for f in ("FOV1", "FOV2")}
    sizes, shared = _duplicates.measure(
        [{"fov_id": f, "origin": tuple(entries[f]["origin_zyx"]), "shape": terr[f].shape,
          "cells": np.array([1, 2], np.int64)} for f in ("FOV1", "FOV2")], terr.__getitem__,
        {"origin": (0.0, 0.0, 0.0), "spacing": (1.0, 0.5, 0.5), "two_d": False})
    pair = (("FOV1", 1), ("FOV2", 1))
    assert (shared[pair], sizes[pair[0]], round(_duplicates.share(sizes, pair, shared[pair]), 4)) == (
        3 * 68, 3 * 73, 0.9315)


# --- CR2 ------------------------------------------------------------------------------------------------------

@needs_anndata
@needs_library
@pytest.mark.parametrize("scene", ["fused_pair", "fused_pair_seams", "fused_quad", "scene_plane"])
def test_cr2_the_report_decides_nothing(cr, scene, tmp_path, monkeypatch):
    case = cr(scene)
    root = case.folder.root
    before = digests(root)
    read_only = tmp_path / "read_only"
    read_only.mkdir()
    read_only.chmod(0o555)
    monkeypatch.chdir(read_only)
    try:
        reports = {t: compare_routes(*folders(case), threshold=t) for t in (0.25, 0.5, 0.75)}
    finally:
        read_only.chmod(0o755)
    assert digests(root) == before
    assert not any(read_only.iterdir())
    base = reports[0.5]
    for threshold, report in reports.items():
        assert report.pairs.drop(columns="matched").equals(base.pairs.drop(columns="matched")), threshold
        for key, value in report.summary.items():
            if key == "molecule_categories":
                for category, n in value.items():
                    if category not in ("same_cell", "different_cells"):
                        assert n == base.summary[key][category], (threshold, category)
            elif key not in THRESHOLD_COUNTS:
                assert value == base.summary[key], (threshold, key)
        assert report.record["threshold"] == threshold
    # With output: the tables and the summary in a folder outside both export folders, and the folders unchanged.
    output = tmp_path / "report"
    report = compare_routes(*folders(case), output=output)
    assert digests(root) == before
    assert sorted(p.name for p in output.iterdir()) == ["counts.csv", "molecules.csv", "pairs.csv", "summary.json"]
    saved = json.loads((output / "summary.json").read_text())
    assert saved["summary"] == json.loads(json.dumps(report.summary))
    assert saved["record"]["threshold"] == 0.5
    for name in ("pairs", "counts", "molecules"):
        table = pd.read_csv(output / f"{name}.csv")
        assert list(table.columns) == list(getattr(report, name).columns)
        assert len(table) == len(getattr(report, name))


# --- CR3 ------------------------------------------------------------------------------------------------------

CR3_SCENES = ["fused_pair", "fused_pair_seams", "fused_quad", "scene_plane"]


def with_extra_gene(book):
    """The codebook with one more gene, ``extra_gene``, on the first colour sequence it does not use."""
    import itertools
    table = book.table
    used = set(table["color_sequence"])
    length = len(table["color_sequence"].iloc[0])
    sequence = next("".join(p) for p in itertools.product("1234", repeat=length) if "".join(p) not in used)
    row = table.iloc[[-1]].assign(entry_id="extra", gene_id="extra_gene", color_sequence=sequence)
    return dataclasses.replace(book, table=pd.concat([table, row], ignore_index=True))


def shifted_placement(folder, path):
    """The fixture's placement with its last FOV 0.5 µm further in x, approved and saved at ``path``."""
    origins = dict(folder.scene.origins)
    last = list(origins)[-1]
    z, y, x = origins[last]
    origins[last] = (z, y, x + 0.5)
    table = positions_from_table(pd.DataFrame({"fov_id": list(origins),
                                               "nominal_z": [o[0] for o in origins.values()],
                                               "nominal_y": [o[1] for o in origins.values()],
                                               "nominal_x": [o[2] for o in origins.values()]}))
    placement = placement_from_table(table, folder.grids, sample=SAMPLE, origins=origins)
    save_placement(approve_placement(placement, reviewer="test", note="CR3 another placement"), path)
    return path


@pytest.fixture(scope="module")
def cr3(cr, tmp_path_factory):
    """``cr3(scene)``: the fixture's two folders of ``cr`` (built once for the module) and folders with one difference
    each, built once per fixture beside them: another placement (the last FOV 0.5 µm further in x, the route per
    FOV), another gene list (the codebook with one more gene, the route per FOV) and another molecule key set (the
    last FOV's last molecules.csv row removed, the route after fusion; the table is restored afterwards, so the fixture's
    files are as ``cr`` built them)."""
    cache = {}

    def get(scene):
        if scene in cache:
            return cache[scene]
        case = cr(scene)
        folder = case.folder
        out = {"per_fov": per_fov_dir(folder), "after_fusion": folder.export_dir}
        # Another placement.
        shifted = shifted_placement(folder, tmp_path_factory.mktemp(f"{scene}_placement") / "stitching")
        out["placement"] = folder.root / "export_placement" / SAMPLE
        export(folder, route_config(route="per_fov"), output_dir=out["placement"], placement=shifted)
        # Another gene list.
        genes = SceneFolder(folder.scene, dataclasses.replace(folder.dataset,
                                                              codebook=with_extra_gene(folder.dataset.codebook)),
                            folder.placement, folder.grids)
        per_fov_routes(genes, name="genes")
        out["genes"] = folder.root / "export_genes" / SAMPLE
        genes.dataset.export(SAMPLE, placement=folder.placement_dir, assignment="genes",
                             config=route_config(route="per_fov"), output_dir=out["genes"])
        # Another molecule key set.
        csv = folder.root / "fovs" / folder.scene.ids[-1] / "molecules.csv"
        original = csv.read_bytes()
        lines = original.decode().splitlines(keepends=True)
        assert len(lines) >= 2  # the header and at least one molecule
        try:
            csv.write_text("".join(lines[:-1]))
            out["molecules"] = folder.root / "export_molecules" / SAMPLE
            export(folder, route_config(segmentation=case.segmentation), output_dir=out["molecules"])
        finally:
            csv.write_bytes(original)
        cache[scene] = SimpleNamespace(case=case, folders=out)
        return cache[scene]
    return get


CR3_CASES = {
    # case: (per_fov folder, after_fusion folder, output, the words the error names)
    "another_placement": ("placement", "after_fusion", None, "different placements: placement fovs_sha256"),
    "another_gene_list": ("genes", "after_fusion", None, "different gene lists"),
    "another_molecule_key_set": ("per_fov", "molecules", None, "different molecule populations"),
    "two_folders_per_fov": ("per_fov", "per_fov", None, "same route 'per_fov'"),
    "two_folders_after_fusion": ("after_fusion", "after_fusion", None, "same route 'after_fusion'"),
    "the_routes_swapped": ("after_fusion", "per_fov", None, "a folder of the route 'after_fusion'"),
    "output_in_the_per_fov_folder": ("per_fov", "after_fusion", ("per_fov", "report"), "inside the export folder"),
    "output_in_the_after_fusion_folder": ("per_fov", "after_fusion", ("after_fusion", "report"),
                                          "inside the export folder"),
    "output_the_folder_itself": ("per_fov", "after_fusion", ("after_fusion", ""), "inside the export folder"),
    "output_in_another_export_folder": ("per_fov", "after_fusion", ("genes", "report"), "inside the export folder"),
}


@needs_anndata
@needs_library
@pytest.mark.parametrize("case", CR3_CASES)
@pytest.mark.parametrize("scene", CR3_SCENES)
def test_cr3_refusals(cr3, scene, case, tmp_path):
    fixture = cr3(scene)
    a, b, output, words = CR3_CASES[case]
    root = fixture.case.folder.root
    before = digests(root)
    target = None if output is None else (fixture.folders[output[0]] / output[1] if output[1]
                                          else fixture.folders[output[0]])
    with pytest.raises(ValueError, match=words.replace("(", r"\(")):
        compare_routes(fixture.folders[a], fixture.folders[b], output=target)
    # Nothing written: every file of both folders (and of the whole fixture) keeps its SHA-256, and no file is added.
    assert digests(root) == before
    if target is not None and output[1]:
        assert not target.exists()
    # The same folders without the difference are accepted.
    compare_routes(fixture.folders["per_fov"], fixture.folders["after_fusion"], output=tmp_path / "accepted")


def test_cr3_a_threshold_outside_0_1_is_refused(tmp_path):
    for value in (0, -0.5, 1.5, float("nan"), True, "0.5"):
        with pytest.raises(ValueError, match="threshold must be a number"):
            compare_routes(tmp_path / "a", tmp_path / "b", threshold=value)
    assert not any(tmp_path.iterdir())


# --- CR4 ------------------------------------------------------------------------------------------------------

def reverse_cells(folder):
    """Rewrite the cell and count tables of an export folder with their rows reversed, and their SHA-256 in the
    record."""
    from starfinder._records import _write_table, file_sha256, write_json
    result = load_assembly(folder)
    record = json.loads((folder / "export.json").read_text())
    files = record["assembly"]["files"]
    for name in ("cells", "counts"):
        frame = getattr(result, name).iloc[::-1].reset_index(drop=True)
        file_name, dtypes = _write_table(frame, folder / "tables", name, "csv")
        assert dtypes == files[name]["dtypes"]
        files[name]["sha256"] = file_sha256(folder / "tables" / file_name)
    write_json(record, folder / "export.json")


@needs_anndata
@needs_library
def test_cr4_determinism(cr, tmp_path):
    case = cr("fused_quad")
    first, second = compare_routes(*folders(case)), compare_routes(*folders(case))
    for name in ("pairs", "counts", "molecules"):
        pd.testing.assert_frame_equal(getattr(first, name), getattr(second, name), check_exact=True)
    assert first.summary == second.summary
    copies = {}
    for name, folder in zip(("per_fov", "after_fusion"), folders(case)):
        copies[name] = tmp_path / name / SAMPLE
        shutil.copytree(folder, copies[name])
        reverse_cells(copies[name])
        reversed_cells = load_assembly(copies[name]).cells
        assert reversed_cells["cell_id"].tolist() == load_assembly(folder).cells["cell_id"].tolist()[::-1]
    third = compare_routes(copies["per_fov"], copies["after_fusion"])
    for name in ("pairs", "counts", "molecules"):
        pd.testing.assert_frame_equal(getattr(first, name), getattr(third, name), check_exact=True)
    assert first.summary == third.summary


# --- CR5 ------------------------------------------------------------------------------------------------------

@pytest.mark.slow
@needs_anndata
@needs_library
def test_cr5_the_whole_of_a_larger_scene(tmp_path):
    case = build(tmp_path, "scene_plane_small")
    seams = seam_cells(case.folder)
    crossing = [cell for cell, fovs in seams.items() if len(fovs) == 4]
    assert crossing, "scene_plane_small has a cell at the crossing of four FOVs"
    report = compare_routes(*folders(case))
    only, _ = cr1_conditions(case, report)
    after_ids = set(case.result.cells["cell_id"].astype("int64").tolist())
    assert set(crossing) <= after_ids
    print(f"\nCR5: {len(case.folder.scene.ids)} FOVs, {len(crossing)} truth cells at the crossing of four FOVs, "
          f"{len(only)} counted cells of the route per FOV that are the only observation of their truth cell, all "
          f"matched; summary {json.dumps({k: v for k, v in report.summary.items() if k != 'share_bins'})}")

