"""EE1 to EE5 of docs/stitching-assembly-validation.md: the synthetic sample scene through the path that
needs no segmenter, assembled on its truth placement.

Fixtures: ``scene_tiny`` (``sample-tiny-v1``, seed 42, J1 jitter in Z, Y and X),
``scene_tiny_flat`` (the same configuration with ``jitter_max_zyx`` ``(0.0, 1.5, 1.5)``) and
``scene_tiny_flat_inward`` (every FOV moved by the bound towards the grid centre in Y and X,
Z 0). Per FOV the truth labels enter through ``import_labels(…, relabel=False)`` and the
molecules through ``molecule_table_from_csv``; ``assign_molecules`` runs without expansion;
the placement is ``placement_from_table`` on ``positions.csv`` with ``origins`` from
``true_origin_z/y/x`` of ``placement_truth.csv``, approved. The oracle of EE2 is computed here
from the truth tables and the FOV truth label images by the rules of
docs/assembly-algorithms.md, never from ``assemble``'s output. EE5's table is a measurement:
it is printed (``-s``) and no accuracy is claimed.
"""
import dataclasses
import json

import numpy as np
import pandas as pd
import pytest

from starfinder.assembly import AssemblyInput, assemble
from starfinder.assignment import assign_molecules, molecule_table_from_csv
from starfinder.segmentation import ReferenceGrid, import_labels
from starfinder.stitching import approve_placement, placement_from_table, positions_from_table

from .assembly_fixtures import check_accounting

pytestmark = [pytest.mark.stitching, pytest.mark.validation]

SCENES = ("scene_tiny", "scene_tiny_flat", "scene_tiny_flat_inward")


def scene_config(name):
    from starfinder.synthetic import sample_scene_preset
    book, config = sample_scene_preset("sample-tiny-v1")
    if name == "scene_tiny":
        return book, config
    acquisition = dataclasses.replace(config.acquisition, jitter_max_zyx=(0.0, 1.5, 1.5))
    if name == "scene_tiny_flat_inward":
        j = acquisition.jitter_max_zyx
        ids = [f"FOV_{k + 1:03d}" for k in range(acquisition.columns * acquisition.rows)]
        inward = {}
        for k, fov in enumerate(ids):
            column, row = divmod(k, acquisition.rows)
            inward[fov] = (0.0, j[1] if row == 0 else -j[1], j[2] if column == 0 else -j[2])
        acquisition = dataclasses.replace(acquisition, jitter="supplied", jitter_zyx=inward)
    return book, dataclasses.replace(config, acquisition=acquisition)


@pytest.fixture(scope="module", params=SCENES)
def scene(request, tmp_path_factory):
    """The scene saved, assigned per FOV and assembled on its truth placement."""
    from starfinder.synthetic import generate_sample_scene, save_sample_scene
    book, config = scene_config(request.param)
    generated = generate_sample_scene(book, config=config)
    root = tmp_path_factory.mktemp(request.param)
    save_sample_scene(generated, root)
    dataset = config.fov.dataset_version
    sample = config.fov.sample_id
    table = positions_from_table(root / "positions.csv")
    truth = pd.read_csv(root / "placement_truth.csv", float_precision="round_trip").set_index("fov_id")
    grids = {fov: ReferenceGrid(config.acquisition.fov_shape_zyx, generated.fov_metadata[fov], "declared")
             for fov in table.fov_ids}
    origins = {fov: tuple(truth.loc[fov, ["true_origin_z", "true_origin_y", "true_origin_x"]]) for fov in
               table.fov_ids}
    placement = approve_placement(placement_from_table(table, grids, sample=request.param, origins=origins),
                                  reviewer="test", note="truth placement")
    results = {}
    for fov in table.fov_ids:
        labels = {}
        for target, file in (("cell", "cells.tif"), ("nucleus", "nuclei.tif")):
            namespace = json.dumps([dataset, sample, fov, None, "truth"])
            labels[target] = import_labels(root / "fovs" / fov / "labels" / file, grid=grids[fov], target=target,
                                           relabel=False, label_namespace=namespace)
        mols = molecule_table_from_csv(root / "fovs" / fov / "molecules.csv",
                                       spot_namespace=json.dumps([dataset, sample, fov, None]), genes=book.genes)
        results[fov] = assign_molecules(mols, labels["cell"], grid=grids[fov], nuclei=labels["nucleus"])
    result = assemble(placement, [AssemblyInput(f, grids[f], assignment=results[f]) for f in table.fov_ids])
    tables = {name: pd.read_csv(root / f"{name}.csv", float_precision="round_trip")
              for name in ("molecules", "cells", "molecule_fovs", "cell_fovs")}
    return dict(name=request.param, scene=generated, config=config, placement=placement, results=results,
                result=result, truth=truth, tables=tables, ids=table.fov_ids)


# --- The oracle: O1 and T1 from the true origins and grid shapes (Y and X), centroids from the truth labels.

def oracle_owner(points_yx, truth, ids):
    owners = []
    for y, x in points_yx:
        best, best_d = None, None
        for fov in ids:
            row = truth.loc[fov]
            oy, ox, ny, nx = row.true_origin_y, row.true_origin_x, row.shape_y, row.shape_x
            sy, sx = row.spacing_y, row.spacing_x
            if oy - sy / 2 <= y < oy + (ny - 0.5) * sy and ox - sx / 2 <= x < ox + (nx - 0.5) * sx:
                d = (y - (oy + (ny - 1) / 2 * sy)) ** 2 + (x - (ox + (nx - 1) / 2 * sx)) ** 2
                if best_d is None or d < best_d:
                    best, best_d = fov, d
        owners.append(best)
    return owners


def oracle(case):
    """Per (fov, cell) and per (fov, csv_row): the expected status, reason, owner, fov_edge and z_edge."""
    scene, truth, ids, tables = case["scene"], case["truth"], case["ids"], case["tables"]
    cells = {}
    for fov in ids:
        labels = scene.fov_labels[fov]["cell"]
        row = truth.loc[fov]
        origin = np.array([row.true_origin_z, row.true_origin_y, row.true_origin_x])
        spacing = np.array([row.spacing_z, row.spacing_y, row.spacing_x])
        for value in (int(v) for v in np.unique(labels) if v):
            centroid = np.argwhere(labels == value).mean(axis=0)
            point = origin + centroid * spacing
            owner = oracle_owner([point[1:]], truth, ids)[0]
            yx_edge = bool(np.isin(value, np.concatenate([labels[:, [0, -1], :].ravel(),
                                                          labels[:, :, [0, -1]].ravel()])))
            z_edge = labels.shape[0] > 1 and bool(np.isin(value, labels[[0, -1]].ravel()))
            cells[(fov, value)] = ("retained" if owner == fov else "excluded", None if owner == fov else "not_owner",
                                   owner, yx_edge, z_edge)
    molecules = {}
    for r in tables["molecule_fovs"].itertuples():
        row = truth.loc[r.fov_id]
        point = (row.true_origin_y + r.y * row.spacing_y, row.true_origin_x + r.x * row.spacing_x)
        owner = oracle_owner([point], truth, ids)[0]
        if not r.in_footprint:
            expected = ("excluded", "outside_grid")
        elif r.fov_cell_label > 0:
            status, reason = cells[(r.fov_id, int(r.fov_cell_label))][:2]
            expected = (status, None if status == "retained" else "cell_not_owner")
        else:
            expected = ("retained", None) if owner == r.fov_id else ("excluded", "not_owner")
        molecules[(r.fov_id, int(r.csv_row))] = (*expected, owner)
    return cells, molecules


def none(value):
    return None if pd.isna(value) else value


def molecule_rows(result):
    return {(r.fov_id, int(r.spot_id.split(":")[1])): r for r in result.molecules.itertuples()}


# --- EE1 -----------------------------------------------------------------------------------------------------

def test_ee1_per_fov_assignment_equals_the_sampled_truth(scene):
    rows = {}
    for fov, result in scene["results"].items():
        for r in result.molecules.itertuples():
            rows[(fov, int(r.spot_id.split(":")[1]))] = r
    disagree = {}
    for r in scene["tables"]["molecule_fovs"].itertuples():
        got = rows.pop((r.fov_id, int(r.csv_row)))
        if r.in_footprint:
            assert (0 if pd.isna(got.cell_id) else int(got.cell_id)) == int(r.fov_cell_label)
            assert got.assignment_status != "outside_grid"
            disagree[r.fov_id] = disagree.get(r.fov_id, 0) + (not r.label_agrees)
        else:
            assert got.assignment_status == "outside_grid"
    assert not rows
    print(f"\nEE1 {scene['name']}: label_agrees false per FOV {disagree}")


# --- EE2 and EE3 ---------------------------------------------------------------------------------------------

def test_ee2_assembly_equals_the_oracle(scene):
    result = scene["result"]
    cells, molecules = oracle(scene)
    got = {(r.fov_id, int(r.cell_id)): (r.assembly_status, none(r.assembly_reason), none(r.owner_fov_id),
                                        "fov_edge" in r.assembly_flags.split(";"),
                                        "z_edge" in r.assembly_flags.split(";")) for r in result.cells.itertuples()}
    assert got == cells
    rows = molecule_rows(result)
    assert {k: (r.assembly_status, none(r.assembly_reason), none(r.owner_fov_id)) for k, r in rows.items()} \
        == molecules
    cell_fovs = scene["tables"]["cell_fovs"]
    for r in cell_fovs[cell_fovs.n_voxels > 0].itertuples():
        assert (bool(r.fov_edge), bool(r.z_edge)) == cells[(r.fov_id, int(r.cell_id))][3:]
    assert len(cell_fovs[cell_fovs.n_voxels > 0]) == len(cells)


def test_ee2_the_sample_z_carries_the_true_z_offsets(scene):
    result, truth = scene["result"], scene["truth"]
    for fov in scene["ids"]:
        row = truth.loc[fov]
        mols = result.molecules[result.molecules.fov_id == fov]
        assert np.abs(mols.sample_z.to_numpy() - (row.true_origin_z + mols.z.to_numpy() * row.spacing_z)).max(
            initial=0.0) <= 1e-9
        cells = result.cells[result.cells.fov_id == fov]
        assert np.abs(cells.sample_centroid_z.to_numpy() - (row.true_origin_z + cells.centroid_z.to_numpy()
                                                            * row.spacing_z)).max(initial=0.0) <= 1e-9
    z = truth.true_origin_z
    assert (z != 0).any() if scene["name"] == "scene_tiny" else (z == 0).all()


def test_ee3_accounting_on_the_scene(scene):
    check_accounting(scene["result"], scene["results"], scene["placement"])


# --- EE4 -----------------------------------------------------------------------------------------------------

def test_ee4_guarantees_against_sample_level_truth(scene):
    result, truth, ids = scene["result"], scene["truth"], scene["ids"]
    tables = scene["tables"]
    rows = molecule_rows(result)
    copies = tables["molecule_fovs"].assign(
        status=[rows[(r.fov_id, int(r.csv_row))].assembly_status for r in tables["molecule_fovs"].itertuples()],
        assignment=[rows[(r.fov_id, int(r.csv_row))].assignment_status
                    for r in tables["molecule_fovs"].itertuples()],
        cell=[none(rows[(r.fov_id, int(r.csv_row))].cell_id) for r in tables["molecule_fovs"].itertuples()],
        assembly_flags=[rows[(r.fov_id, int(r.csv_row))].assembly_flags for r in tables["molecule_fovs"].itertuples()])
    molecules = tables["molecules"].set_index("amplicon_id")
    sample = molecules[molecules.in_sample]
    by_molecule = {k: g for k, g in copies.groupby("amplicon_id")}
    for amplicon, m in sample.iterrows():
        group = by_molecule[amplicon]
        # (a) at least one copy that is not outside_grid.
        assert (group.assignment != "outside_grid").any()
        # (b) the copy in the FOV owning the sample position samples 0: exactly one retained unassigned copy there.
        owner = oracle_owner([(m.sample_y, m.sample_x)], truth, ids)[0]
        own = group[group.fov_id == owner]
        if len(own) and own.fov_cell_label.iloc[0] == 0:
            retained = group[(group.status == "retained") & (group.assignment == "unassigned")]
            assert list(retained.fov_id) == [owner]
    for amplicon, group in by_molecule.items():
        retained = group[group.status == "retained"]
        # (c) r >= 2 retained copies: at least r - 1 flagged outside_owner_region.
        if len(retained) >= 2:
            assert retained.assembly_flags.str.contains("outside_owner_region").sum() >= len(retained) - 1
        # (d) a retained assigned copy disagrees with the truth exactly when label_agrees is false.
        true_cell = molecules.loc[amplicon, "true_cell_id"]
        for r in retained[retained.assignment == "assigned"].itertuples():
            differs = int(r.cell) != (0 if pd.isna(true_cell) else int(true_cell))
            assert differs == (not r.label_agrees)


# --- EE5 -----------------------------------------------------------------------------------------------------

def ee5_table(cell_status, molecule_status, tables):
    """Retained copies per sample cell and per sample molecule (sample and halo apart).

    cell_status maps (fov, cell) -> (status, flags); molecule_status (fov, csv_row) -> status.
    """
    cells = tables["cells"].set_index("cell_id")
    rows = []
    for population, in_sample in (("sample", True), ("halo", False)):
        per_cell = {}
        for (fov, cell), (status, flags) in cell_status.items():
            if status == "retained" and bool(cells.loc[cell, "in_sample"]) == in_sample:
                per_cell.setdefault(cell, []).append(flags)
        for bucket in ("0", "1", "2+"):
            ids = [c for c in cells.index[cells.in_sample == in_sample]
                   if {"0": 0, "1": 1}.get(bucket, 2) == min(len(per_cell.get(c, [])), 2)]
            flagged = {f: sum(any(f in flags for flags in per_cell.get(c, [])) for c in ids)
                       for f in ("fov_edge", "z_edge", "conflict")}
            rows.append({"population": population, "object": "cell", "retained_copies": bucket, "n": len(ids),
                         "truth_n_molecules": int(cells.loc[ids, "n_molecules"].sum()) if ids else 0, **flagged})
        molecules = tables["molecule_fovs"]
        retained = {}
        for r in molecules.itertuples():
            if bool(r.in_sample) == in_sample:
                retained[r.amplicon_id] = retained.get(r.amplicon_id, 0) + (molecule_status[(r.fov_id,
                                                                                            int(r.csv_row))]
                                                                            == "retained")
        unseen = tables["molecules"]
        unseen = set(unseen.amplicon_id[unseen.in_sample == in_sample]) - set(retained)
        retained.update({a: 0 for a in unseen})
        for bucket in ("0", "1", "2+"):
            n = sum({"0": 0, "1": 1}.get(bucket, 2) == min(v, 2) for v in retained.values())
            rows.append({"population": population, "object": "molecule", "retained_copies": bucket, "n": n,
                         "truth_n_molecules": None, "fov_edge": None, "z_edge": None, "conflict": None})
    return pd.DataFrame(rows).astype({c: "Int64" for c in ("n", "truth_n_molecules", "fov_edge", "z_edge",
                                                             "conflict")})


def test_ee5_comparison_with_sample_level_truth(scene):
    result = scene["result"]
    measured = ee5_table({(r.fov_id, int(r.cell_id)): (r.assembly_status, r.assembly_flags)
                          for r in result.cells.itertuples()},
                         {k: r.assembly_status for k, r in molecule_rows(result).items()}, scene["tables"])
    cells, molecules = oracle(scene)
    # The oracle's recount: its statuses, with the flags recomputed from the truth labels (edges) and the
    # conflict flag, a centroid test in the other FOV's truth labels.
    conflict = oracle_conflicts(scene, cells)
    recount = ee5_table({k: (v[0], ";".join(f for f, hit in (("fov_edge", v[3]), ("z_edge", v[4]),
                                                            ("conflict", k in conflict)) if hit))
                         for k, v in cells.items()}, {k: v[0] for k, v in molecules.items()}, scene["tables"])
    pd.testing.assert_frame_equal(measured, recount, check_exact=True)
    print(f"\nEE5 {scene['name']} (a measurement of the retention policy, no accuracy claim):\n"
          + measured.to_string(index=False))


def oracle_conflicts(case, cells):
    """(fov, cell) of retained cells whose centroid lies in another FOV's retained cell, or the converse."""
    scene, truth, ids = case["scene"], case["truth"], case["ids"]
    centroid = {}
    for fov in ids:
        labels = scene.fov_labels[fov]["cell"]
        row = truth.loc[fov]
        origin = np.array([row.true_origin_z, row.true_origin_y, row.true_origin_x])
        spacing = np.array([row.spacing_z, row.spacing_y, row.spacing_x])
        for value in (int(v) for v in np.unique(labels) if v):
            centroid[(fov, value)] = origin + np.argwhere(labels == value).mean(axis=0) * spacing
    flagged = set()
    for (a, cell), point in centroid.items():
        if cells[(a, cell)][0] != "retained":
            continue
        for b in ids:
            if b == a:
                continue
            row = truth.loc[b]
            origin = np.array([row.true_origin_z, row.true_origin_y, row.true_origin_x])
            spacing = np.array([row.spacing_z, row.spacing_y, row.spacing_x])
            index = np.floor((point - origin) / spacing + 0.5).astype(int)
            shape = scene.fov_labels[b]["cell"].shape
            footprint = all(origin[k] - spacing[k] / 2 <= point[k] < origin[k] + (shape[k] - 0.5) * spacing[k]
                            for k in (1, 2))
            if not footprint or not all(0 <= index[k] < shape[k] for k in range(3)):
                continue
            value = int(scene.fov_labels[b]["cell"][tuple(index)])
            if value and cells[(b, value)][0] == "retained":
                flagged |= {(a, cell), (b, value)}
    return flagged
