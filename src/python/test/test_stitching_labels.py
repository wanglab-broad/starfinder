"""LB1 to LB7, and the label parts of EX1, EX2 (check 6) and EX3 ((c), (e) to (h)), of
design/validation/stitching-assembly.md: the assembled label export ``fuse_labels``.

Fixtures (``label_fixtures``): ``label_quad`` and its volume variant, ``label_pair`` (W-359's
two-FOV fixture) with ``label_pair_conflict``, ``label_pair_empty`` and ``label_pair_merged``,
``label_scene`` (W-354's label fixture, seeds 354 and 355; ``slow``), a one-FOV sample whose
territory is all one cell, ``id_bound``, ``empty_cases`` and ``export_folder``. The expected
values are W-359's records (``tables/w354_summary.csv``, ``tables/two_fov_summary.csv``,
``tables/two_fov_cells.csv``, ``tables/case_placeholder_summary.csv``), stated here as literals.
The per-cell measure of W-359 ``worker-notes.md`` section 3.3 is recomputed here from the
territories: ``F_v`` by nearest-voxel sampling of the FOV's footprint, ``E_v`` by the library
fusing that FOV's territory alone on the same grid. dask runs with the synchronous scheduler.
"""
import importlib.util
import json
import re
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import tifffile

import starfinder.assembly
import starfinder.assignment._persist as assignment_persist
import starfinder.dataset._export as export_module
from starfinder.assembly import AssemblyConfig, AssemblyResult, assemble, load_assembly, save_assembly
from starfinder.io import save_volume
from starfinder.export import ExportConfig, ExportIncompleteWarning, LabelConfig, check_export, fuse_labels
from starfinder.export._labels import check_label_ids
from starfinder.io._ome import read_ome_layout
from starfinder.stitching import load_placement

from . import hash_oracles
from . import label_fixtures as lf
from .fusion_fixtures import canonical, export_folder, listing

EXTRA = "the multiview-stitcher extra (starfinder[multiview-stitcher]) is not installed"
pytestmark = [pytest.mark.stitching, pytest.mark.validation,
              pytest.mark.skipif(importlib.util.find_spec("multiview_stitcher") is None, reason=EXTRA)]
needs_anndata = pytest.mark.skipif(importlib.util.find_spec("anndata") is None,
                                   reason="the anndata extra (starfinder[anndata]) is not installed")
SCALES = (1, 0.25)


@pytest.fixture(autouse=True)
def synchronous():
    import dask
    with dask.config.set(scheduler="synchronous"):
        yield


# --- Running the part and reading its output ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def cases(tmp_path_factory):
    """Each fixture built once per module, on first use."""
    built = {}
    root = tmp_path_factory.mktemp("labels")

    def get(name, **kwargs):
        key = (name, tuple(sorted(kwargs.items())))
        if key not in built:
            if name.startswith("label_pair"):
                built[key] = lf.label_pair(root, name, **kwargs)
            else:
                built[key] = getattr(lf, name)(root, **kwargs)
        return built[key]

    return get


class Run(SimpleNamespace):
    """One label export of a case at one scale: the section, the image (ZYX) and its grid."""


def run(case, scale, *, folder=None):
    folder = folder or case.root / "export" / f"{case.name}_{scale}"
    if not (folder / "export.json").exists():
        save_assembly(case.assembly, folder)
    record = fuse_labels(case.placement, load_assembly(folder), config=LabelConfig(scale_yx=scale),
                         output_dir=folder)
    section = record.section
    output = section["outputs"][0] if section["outputs"] else None
    image = None
    if output is not None:
        image = tifffile.imread(folder / output["path"])
        image = image[None] if image.ndim == 2 else image
    return Run(case=case, scale=scale, folder=folder, record=record, section=section, output=output, image=image)


@pytest.fixture(scope="module")
def runs(cases):
    done = {}

    def get(name, scale, **kwargs):
        key = (name, scale, tuple(sorted(kwargs.items())))
        if key not in done:
            done[key] = run(cases(name, **kwargs), scale)
        return done[key]

    return get


def instance_map(case):
    """(fov, cell_id) → instance_id (1..N in cell table order), and the retained (fov, cell_id) rows."""
    cells = case.assembly.cells
    keys = list(zip(cells.fov_id.astype(str), cells.cell_id.astype("int64")))
    ids = {key: i + 1 for i, key in enumerate(keys)}
    retained = {key for key, status in zip(keys, cells.assembly_status) if status == "retained"}
    return ids, retained


def included(case):
    fovs = case.placement.fovs
    return list(fovs.fov_id[fovs.status.isin(["image_supported", "metadata_only"])])


def centres(r):
    o = r.output
    return [o["origin_zyx"][a] + np.arange(o["shape_zyx"][a]) * o["spacing_zyx"][a] for a in range(3)]


def nearest(r, fov):
    """Per axis the index ``floor((c − origin)/s + ½)`` of the FOV's voxel nearest each output voxel centre."""
    case = r.case
    origin, spacing = np.asarray(case.origins[fov], float), case.spacing(fov)
    index = [np.floor((c - origin[a]) / spacing[a] + 0.5).astype(np.int64) for a, c in enumerate(centres(r))]
    if case.territories[fov].shape[0] == 1:
        index[0] = np.zeros_like(index[0])
    return index


def footprint_view(r, fov):
    """The FOV's whole territory sampled at the output voxel centres by the nearest voxel over its footprint."""
    territory = r.case.territories[fov]
    index = nearest(r, fov)
    valid = [(i >= 0) & (i < n) for i, n in zip(index, territory.shape)]
    out = np.zeros(r.image.shape, np.int64)
    out[np.ix_(*valid)] = territory[np.ix_(*[i[v] for i, v in zip(index, valid)])]
    return out


def _first_valid_positive(transformed_views):
    out = np.full(transformed_views[0].shape, np.nan, dtype=transformed_views.dtype)
    for view in transformed_views:
        take = np.isnan(out) & ~np.isnan(view) & (view > 0)
        out[take] = view[take]
    return out


def library_single(r, fov, territory=None):
    """The FOV's territory (cell_id values) fused alone by the library at order 0 on the label image's grid."""
    from multiview_stitcher import fusion
    from multiview_stitcher import spatial_image_utils as si_utils
    case, o = r.case, r.output
    territory = case.territories[fov] if territory is None else territory
    two_d = o["inputs_2d"]
    dims = ["y", "x"] if two_d else ["z", "y", "x"]
    axes = [1, 2] if two_d else [0, 1, 2]
    array = territory[0] if two_d else territory
    spacing = case.spacing(fov)
    sim = si_utils.get_sim_from_array(array, dims=dims, scale={d: float(spacing[a]) for d, a in zip(dims, axes)},
                                      translation={d: float(case.origins[fov][a]) for d, a in zip(dims, axes)},
                                      transform_key="placed")
    grid = {"spacing": {d: o["spacing_zyx"][a] for d, a in zip(dims, axes)},
            "origin": {d: o["origin_zyx"][a] for d, a in zip(dims, axes)},
            "shape": {d: o["shape_zyx"][a] for d, a in zip(dims, axes)}}
    fused = fusion.fuse([sim], transform_key="placed", fusion_func=_first_valid_positive, interpolation_order=0,
                        output_stack_properties=grid)
    out = np.asarray(fused.data.compute())[0, 0].astype(np.int64)
    return out[None] if two_d else out


def counts(a):
    values, n = np.unique(a[a > 0], return_counts=True)
    return dict(zip(values.tolist(), n.tolist()))


def measure(r):
    """W-359's per-cell measure (``worker-notes.md`` section 3.3) on one label image; returns per-cell rows."""
    case = r.case
    ids, retained = instance_map(case)
    owner = {v: key for key, v in ids.items()}
    fovs = included(case)
    singles = {f: library_single(r, f) for f in fovs}
    feet = {f: footprint_view(r, f) for f in fovs}
    drawn = counts(r.image)
    rows = {}
    for g in sorted({c for _, c in ids}):
        shown = sorted(v for v in drawn if owner[v][1] == g)
        by = [key for key in retained if key[1] == g]
        if not shown:
            rows[g] = {"outcome": "absent", "retained_by": by}
            continue
        if len(shown) > 1:
            rows[g] = {"outcome": "two_or_more_ids", "ids": shown, "voxels": [drawn[v] for v in shown]}
            continue
        v = shown[0]
        fov, cell = owner[v]
        sampled, foot, here = singles[fov] == cell, feet[fov] == cell, r.image == v
        assert int((sampled & here).sum()) == drawn[v], (g, v)        # every drawn voxel was sampled for v
        assert int((sampled & foot).sum()) == int(sampled.sum()), (g, v)  # and lies in F_v
        overdrawn = int(sampled.sum()) - drawn[v]
        edge = int(foot.sum()) - int(sampled.sum())
        pixels = int((case.territories[fov] == cell).sum())
        cut = pixels < case.true_px.get((fov, cell), pixels)
        outcome = "whole_one_id" if not (overdrawn or edge or cut) else "incomplete"
        rows[g] = {"outcome": outcome, "ids": [v], "voxels": [drawn[v]], "overdrawn": overdrawn, "edge": edge,
                   "cut": cut, "fov": fov, "cell": cell, "foot": foot, "sampled": sampled}
    return rows


def summary(rows):
    outcomes = [row["outcome"] for row in rows.values()]
    incomplete = [row for row in rows.values() if row["outcome"] == "incomplete"]
    return {"cells": len(rows), "whole": outcomes.count("whole_one_id"), "incomplete": len(incomplete),
            "overdrawn": sum(bool(row["overdrawn"]) for row in incomplete),
            "edge": sum(bool(row["edge"]) for row in incomplete), "cut": sum(row["cut"] for row in incomplete),
            "two_or_more": outcomes.count("two_or_more_ids"), "absent": outcomes.count("absent")}


def recomputed_ids(r):
    """ids_input from the territories, the instance_id map and assembly_status; ids_output from the file."""
    ids, retained = instance_map(r.case)
    ids_input = set()
    for fov, cell in retained:
        if (r.case.territories[fov] == cell).any():
            ids_input.add(ids[(fov, cell)])
    ids_output = {int(v) for v in np.unique(r.image) if v > 0}
    return ids_input, ids_output


def check_no_new_value(r):
    ids, retained = instance_map(r.case)
    retained_ids = {ids[key] for key in retained}
    excluded_ids = set(ids.values()) - retained_ids
    ids_input, ids_output = recomputed_ids(r)
    assert ids_output <= retained_ids and ids_output <= ids_input
    assert not ids_output & excluded_ids
    assert len(ids_output - ids_input) == 0                    # values in no input: 0 (W-359)


def check_drawn_by_retained_cells(r):
    """(a) of LB3: at each drawn voxel of value v, v's FOV's territory sampled at floor(c + ½) holds v's cell."""
    ids, _ = instance_map(r.case)
    owner = {v: key for key, v in ids.items()}
    z, y, x = np.nonzero(r.image)
    values = r.image[z, y, x]
    checked = 0
    for v in np.unique(values):
        fov, cell = owner[int(v)]
        territory = r.case.territories[fov]
        index = nearest(r, fov)
        at = values == v
        iz, iy, ix = index[0][z[at]], index[1][y[at]], index[2][x[at]]
        inside = (iz >= 0) & (iz < territory.shape[0]) & (iy >= 0) & (iy < territory.shape[1]) & (ix >= 0) \
            & (ix < territory.shape[2])
        assert inside.all(), (fov, cell)
        assert (territory[iz, iy, ix] == cell).all(), (fov, cell)
        checked += int(at.sum())
    assert checked == len(values)


# --- LB1: the label export at scale 0.25 --------------------------------------------------------------------------

@needs_anndata
@pytest.mark.parametrize("variant", ["plane", "volume"])
def test_lb1_label_export_at_scale_025(cases, variant):
    case = cases("label_quad", volume=variant == "volume")
    config = ExportConfig(route="per_fov", fuse=False, fuse_labels=True)
    assert config.labels.scale_yx == 0.25
    exported = case.dataset.export(case.name, placement=case.placement_dir, config=config, output_dir=case.export_dir)
    section = exported.record["labels"]
    assert exported.parts["fuse_labels"] == "written" and section["status"] == "written"
    assert section["rule"] == "retained_cells" and section["labels_scale_yx"] == 0.25
    assert section["library"]["fusion_func"] == "_first_positive" and section["library"]["interpolation_order"] == 0
    assert section["library"]["route"] == "output_spacing"
    output = section["outputs"][0]
    path = case.export_dir / "cell_labels.ome.tif"
    assert output["path"] == "cell_labels.ome.tif" and output["dtype"] == "uint32"
    layout = read_ome_layout(path)
    assert layout["is_bigtiff"] and layout["is_ome"] and layout["n_series"] == 1 and layout["dtype"] == "uint32"
    assert layout["pixels"]["Type"] == "uint32" and layout["tile_yx"] == [512, 512] and layout["channels"] == 1
    # The formula grid in Y and X: spacing 4·s, origin o (not o + s·(k − 1)/2), size 18 × 18.
    n = int(np.floor((30.25 + 39 - 0.0) / 4 + 1e-9)) + 1
    assert n == 18
    if variant == "plane":
        expected = ([1.0, 4.0, 4.0], [0.0, 0.0, 0.0], [1, 18, 18])     # Z: the slab of these plane territories
        assert layout["axes"] == "YX" and layout["shape"] == [18, 18]
    else:
        expected = ([1.0, 4.0, 4.0], [0.0, 0.0, 0.0], [4, 18, 18])     # Z: the fusion grid's s_z, o_z, n0_z
        assert layout["axes"] == "ZYX" and layout["shape"] == [4, 18, 18]
    assert output["shape_zyx"] == expected[2]
    assert np.allclose(output["spacing_zyx"], expected[0], rtol=0, atol=1e-9)
    assert np.allclose(output["origin_zyx"], expected[1], rtol=0, atol=1e-9)
    physical = [float(layout["pixels"][f"PhysicalSize{a}"]) for a in "ZYX"]
    assert np.allclose(physical, expected[0], rtol=0, atol=1e-9)
    for i, plane in enumerate(layout["planes"]):
        position = [float(plane[f"Position{a}"]) for a in "ZYX"]
        assert np.allclose(position, [expected[1][0] + i * expected[0][0], 0.0, 0.0], rtol=0, atol=1e-9)
    grid = section["grid"]
    for a in ("y", "x"):
        assert grid["library"]["shape"][a] == grid["formula"]["shape"][a] == 18
        assert abs(grid["library"]["spacing"][a] - grid["formula"]["spacing"][a]) <= 1e-9
        assert abs(grid["library"]["origin"][a] - grid["formula"]["origin"][a]) <= 1e-9
    if variant == "volume":
        assert grid["library"]["shape"]["z"] == 4 and grid["library"]["spacing"]["z"] == 1.0
        assert grid["library"]["origin"]["z"] == 0.0
    check_export(case.export_dir)


@pytest.mark.parametrize("variant", ["plane", "volume"])
def test_lb1_label_export_at_scale_1(runs, variant):
    r = runs("label_quad", 1, volume=variant == "volume")
    assert r.output["shape_zyx"] == [1 if variant == "plane" else 4, 70, 70]
    assert r.section["labels_scale_yx"] == 1
    assert np.allclose(r.output["spacing_zyx"][1:], [1.0, 1.0], rtol=0, atol=1e-9)
    assert np.allclose(r.output["origin_zyx"], [0.0, 0.0, 0.0], rtol=0, atol=1e-9)
    assert r.section["grid"]["library"]["shape"] == {**({} if variant == "plane" else {"z": 4}), "y": 70, "x": 70}
    check_export(r.folder)


# --- LB2, LB3, LB4 on label_quad, label_pair and the one-FOV sample -----------------------------------------------

@pytest.mark.parametrize("scale", SCALES)
@pytest.mark.parametrize("volume", [False, True])
def test_lb2_lb3_label_quad(runs, scale, volume):
    r = runs("label_quad", scale, volume=volume)
    check_no_new_value(r)
    check_drawn_by_retained_cells(r)
    check_export(r.folder)


@pytest.mark.parametrize("scale", SCALES)
def test_lb2_lb3_lb4_label_pair(runs, scale):
    r = runs("label_pair", scale)
    check_no_new_value(r)
    check_drawn_by_retained_cells(r)
    rows = measure(r)
    expected = {1: {"cells": 13, "whole": 9, "incomplete": 2, "overdrawn": 1, "edge": 0, "cut": 1, "two_or_more": 1,
                    "absent": 1},
                0.25: {"cells": 13, "whole": 10, "incomplete": 1, "overdrawn": 0, "edge": 0, "cut": 1,
                       "two_or_more": 1, "absent": 1}}[scale]
    assert summary(rows) == expected
    # (c) The outcome of each cell (W-359 tables/two_fov_cells.csv, rows retained_cells) and its voxels.
    voxels = {1: {1: [81], 2: [81], 3: [49], 4: [49], 5: [62], 6: [49, 37], 8: [81], 9: [80], 10: [35], 11: [49],
                  12: [113], 13: [113]},
              0.25: {1: [5], 2: [5], 3: [2], 4: [2], 5: [5], 6: [2, 1], 8: [6], 9: [5], 10: [4], 11: [2], 12: [8],
                     13: [8]}}[scale]
    outcomes = {g: row["outcome"] for g, row in rows.items()}
    assert outcomes[6] == "two_or_more_ids" and outcomes[7] == "absent" and outcomes[5] == "whole_one_id"
    assert outcomes[10] == "incomplete" and rows[10]["cut"]
    assert outcomes[9] == ("incomplete" if scale == 1 else "whole_one_id")
    others = set(rows) - {6, 7, 9, 10}
    assert all(outcomes[g] == "whole_one_id" for g in others)
    assert {g: row["voxels"] for g, row in rows.items() if "voxels" in row} == voxels
    # LB4: 13, 13, 0; cell 7 (retained by no FOV) is in neither.
    ids_input, ids_output = recomputed_ids(r)
    section = r.section
    assert (section["n_ids_input"], section["n_ids_output"], section["absent_ids"]) == (13, 13, [])
    assert (len(ids_input), len(ids_output)) == (13, 13) and ids_output <= ids_input
    ids, _ = instance_map(r.case)
    assert not {ids[("A", 7)], ids[("B", 7)]} & (ids_input | ids_output)
    check_export(r.folder)


@pytest.mark.parametrize("scale", SCALES)
def test_lb4_label_quad(runs, scale):
    r = runs("label_quad", scale)
    ids_input, ids_output = recomputed_ids(r)
    absent = sorted(ids_input - ids_output)
    section = r.section
    assert section["n_ids_input"] == len(ids_input) and section["n_ids_output"] == len(ids_output)
    assert section["absent_ids"] == absent and section["n_absent_ids"] == len(absent)
    assert ids_output <= ids_input
    ids, _ = instance_map(r.case)
    owner = {v: key for key, v in ids.items()}
    if scale == 1:
        assert absent == []
    else:
        # At 0.25 only 1 × 1 px cells have no output voxel centre: sampling drops them.
        assert absent and all(owner[v][1] == lf.QUAD_DOT_ID for v in absent)


def test_lb4_one_fov_all_one_cell(cases):
    case = cases("one_cell")
    r = run(case, 0.25)
    assert r.section["n_ids_input"] == 1 and r.section["n_ids_output"] == 1 and r.section["absent_ids"] == []
    assert recomputed_ids(r) == ({1}, {1})
    assert set(np.unique(r.image).tolist()) == {1}
    check_export(r.folder)


# --- LB2, LB3, LB4, LB7.8 and LB7.9 on label_scene (slow) ----------------------------------------------------------

SCENE_EXPECTED = {1: {"cells": 391, "whole": 370, "incomplete": 21, "overdrawn": 3, "edge": 18, "cut": 0,
                      "two_or_more": 0, "absent": 0},
                  0.25: {"cells": 391, "whole": 366, "incomplete": 11, "overdrawn": 1, "edge": 10, "cut": 0,
                         "two_or_more": 0, "absent": 14}}
SCENE_IDS = {1: (391, 391, 0), 0.25: (391, 377, 14)}


@pytest.mark.slow
@pytest.mark.parametrize("scale", SCALES)
def test_label_scene(runs, scale):
    """LB2, LB3, LB4 and LB7.8 on W-354's label fixture rebuilt (seeds 354, 355), at scale 1 and 0.25."""
    r = runs("label_scene", scale)
    check_no_new_value(r)
    check_drawn_by_retained_cells(r)
    rows = measure(r)
    assert summary(rows) == SCENE_EXPECTED[scale]
    ids_input, ids_output = recomputed_ids(r)
    section = r.section
    assert (section["n_ids_input"], section["n_ids_output"], section["n_absent_ids"]) == SCENE_IDS[scale]
    assert (len(ids_input), len(ids_output)) == SCENE_IDS[scale][:2] and ids_output <= ids_input
    assert section["absent_ids"] == sorted(ids_input - ids_output)
    # LB7.8: every voxel of F_v that the library does not sample lies in one row or column just beyond the
    # FOV's voxel-centre extent, within half a voxel of its footprint.
    edge_cells = [row for row in rows.values() if row["outcome"] == "incomplete" and row["edge"]]
    assert len(edge_cells) == {1: 18, 0.25: 10}[scale]
    for row in edge_cells:
        fov = row["fov"]
        z, y, x = np.nonzero(row["foot"] & ~row["sampled"])
        assert len(y) == row["edge"]
        spacing, origin = r.case.spacing(fov), r.case.origins[fov]
        n = r.case.territories[fov].shape
        c = centres(r)
        iy, ix = (c[1][y] - origin[1]) / spacing[1], (c[2][x] - origin[2]) / spacing[2]
        beyond_y = ((iy > -0.5) & (iy < 0)) | ((iy > n[1] - 1) & (iy < n[1] - 0.5))
        beyond_x = ((ix > -0.5) & (ix < 0)) | ((ix > n[2] - 1) & (ix < n[2] - 0.5))
        assert (beyond_y | beyond_x).all()
    check_export(r.folder)


@pytest.mark.slow
def test_lb7_9_label_scene_with_a_placeholder(runs):
    """LB7.9 on label_scene with its centre FOV a placeholder, at scale 1 (W-359 case_placeholder_summary.csv)."""
    r = runs("label_scene", 1, placeholder=True)
    case = r.case
    _, retained = instance_map(case)
    assert len(retained) == 378 and not [key for key in retained if key[0] == "S4"]
    assert [e["fov_id"] for e in r.section["fovs"]] == [f for f in case.ids if f != "S4"]
    check_no_new_value(r)
    check_drawn_by_retained_cells(r)
    rows = measure(r)
    absent = [g for g, row in rows.items() if row["outcome"] == "absent"]
    assert len(absent) == 13 and all(not rows[g]["retained_by"] for g in absent)
    assert summary(rows)["two_or_more"] == 0
    check_export(r.folder)


# --- LB5: empty populations, the conflict count and the section ----------------------------------------------------

def conflict_count(r):
    """The cell rows flagged ``conflict`` whose instance_id is in the label image, recounted."""
    cells = r.case.assembly.cells
    flagged = cells.assembly_flags.fillna("").astype(str).str.split(";").map(lambda f: "conflict" in f)
    ids = np.arange(1, len(cells) + 1)[flagged.to_numpy(dtype=bool)]
    return int(np.isin(ids, np.unique(r.image)).sum())


def test_lb5_conflict_count(runs):
    pair, conflict = runs("label_pair", 1), runs("label_pair_conflict", 1)
    assert conflict_count(pair) == pair.section["n_conflict"] == 0
    # W-400: under the default duplicate rule B's view of cell 6 is excluded (share 0.53 with A's), so no cell keeps
    # the flag; with the rule off both views are retained and flagged, as before.
    assert conflict_count(conflict) == conflict.section["n_conflict"] == 0
    off = runs("label_pair_conflict", 1, duplicate_threshold=None)
    assert conflict_count(off) == off.section["n_conflict"] == 2


def test_lb5_section(runs):
    r = runs("label_quad", 0.25)
    section = r.section
    case = r.case
    _, retained = instance_map(case)
    by_fov = {}
    for fov, _ in retained:
        by_fov[fov] = by_fov.get(fov, 0) + 1
    assert section["assembly_sha256"] == hash_oracles.section_sha256(
        json.loads(json.dumps(load_assembly(r.folder).record)))
    record = json.loads((r.folder / "export.json").read_text())
    assert section["assembly_sha256"] == hash_oracles.section_sha256(record["assembly"])
    assert section["rule"] == "retained_cells"
    assert [e["fov_id"] for e in section["fovs"]] == [f for f in case.ids if f in by_fov]
    for entry in section["fovs"]:
        path = case.territory_path(entry["fov_id"])
        assert entry["path"] == str(path) and entry["source"] == "assembly_record"
        assert entry["sha256"] == hash_oracles.file_sha256(path)
        assert entry["n_retained"] == by_fov[entry["fov_id"]]
    library = section["library"]
    assert library["multiview_stitcher"] == "0.1.62" or isinstance(library["multiview_stitcher"], str)
    assert (library["fusion_func"], library["interpolation_order"], library["route"]) == \
        ("_first_positive", 0, "output_spacing")
    assert section["labels_scale_yx"] == 0.25 and set(section["grid"]) >= {"library", "formula"}
    output = section["outputs"][0]
    assert {"path", "sha256", "shape_zyx", "dtype"} <= set(output)
    assert output["sha256"] == hash_oracles.file_sha256(r.folder / output["path"])
    assert {"n_ids_input", "n_ids_output", "absent_ids", "n_absent_ids", "n_conflict"} <= set(section)
    assert all(section[k] >= 0 for k in ("wall_s", "cpu_s", "peak_rss_kib"))
    # A FOV without a retained cell has no entry, and its territory is not read (LB7.9 below).
    placeholder = runs("label_quad", 1, placeholder=True)
    assert [e["fov_id"] for e in placeholder.section["fovs"]] == ["L1", "L2", "L3"]


@pytest.mark.parametrize("scale", SCALES)
def test_lb5_label_pair_empty(runs, scale):
    r = runs("label_pair_empty", scale)
    assert r.section["status"] == "written" and not r.image.any()
    assert (r.section["n_ids_input"], r.section["n_ids_output"], r.section["absent_ids"]) == (0, 0, [])
    assert r.section["fovs"] == []
    check_export(r.folder)


@needs_anndata
def test_lb5_empty_cases(tmp_path):
    from .assembly_fixtures import golden_placement, golden_results, inputs_of
    from .test_stitching_assembly_retention import empty_variants, molecules_only, no_fov_assembly
    # A molecules_only sample: no file, empty, no_cells.
    placement = golden_placement()
    only = assemble(placement, inputs_of(molecules_only(), kind="molecules"))
    record = fuse_labels(placement, only, output_dir=tmp_path / "only")
    assert record.status == "empty" and record.section["reason"] == "no_cells" and record.paths == ()
    assert sorted(listing(tmp_path / "only")) == ["export.json"]
    # A cells sample whose FOVs have no cell row: an image of zeros, ids empty.
    none = assemble(placement, inputs_of(golden_results(**empty_variants()["no_cell_row"])))
    none = save_assembly(none, tmp_path / "none")
    record = fuse_labels(placement, none, output_dir=tmp_path / "none")
    image = tifffile.imread(tmp_path / "none" / "cell_labels.ome.tif")
    assert record.status == "written" and image.dtype == np.uint32 and not image.any()
    assert (record.section["n_ids_input"], record.section["n_ids_output"], record.section["absent_ids"]) == (0, 0, [])
    check_export(tmp_path / "none")
    # No included FOV: fuse_labels alone on no_fov_assembly's all-failed placement.
    from starfinder.stitching._placement import table_sha256

    from .assembly_fixtures import GOLDEN_IDS
    failed = golden_placement(statuses={fov: "failed" for fov in GOLDEN_IDS})
    hand = no_fov_assembly()
    hashes = {"fovs_sha256": table_sha256(failed.fovs), "links_sha256": table_sha256(failed.links)}
    hand = AssemblyResult(hand.sample, hand.fovs, hand.cells, hand.molecules, hand.counts, hand.genes,
                          dict(hand.record, placement=dict(hand.record["placement"], **hashes)))
    record = fuse_labels(failed, hand, output_dir=tmp_path / "failed")
    assert record.status == "empty" and record.section["reason"] == "no_included_fov" and record.paths == ()


# --- LB6: the ID bound --------------------------------------------------------------------------------------------

def test_lb6_id_bound():
    above = pd.DataFrame({"cell_id": np.zeros(2 ** 24 + 1, np.uint32)})
    with pytest.raises(ValueError, match="16777217 rows"):
        check_label_ids(above)
    check_label_ids(above.iloc[:2 ** 24])                       # 2^24 = 16,777,216 rows pass
    del above


@needs_anndata
def test_lb6_id_bound_through_the_entry(tmp_path, monkeypatch):
    """Check 6 of the entry refuses 16,777,217 cell rows before anything is written."""
    folder = export_folder(tmp_path)
    stand_in = SimpleNamespace(cells=pd.DataFrame({"cell_id": np.zeros(2 ** 24 + 1, np.uint32)}))
    monkeypatch.setattr(starfinder.assembly, "assemble", lambda *a, **k: stand_in)
    before = listing(folder.root)
    with pytest.raises(ValueError, match="16777217 rows"):
        folder.dataset.export("sample1", placement=folder.placement_dir,
                              config=ExportConfig(route="per_fov", fuse=False, fuse_labels=True),
                              output_dir=folder.export_dir)
    assert listing(folder.root) == before and not folder.export_dir.exists()


# --- LB7: the cases of "What the label image shows" ---------------------------------------------------------------

def shows(r, fov, cell):
    """The output voxels whose value is the instance_id of (fov, cell)."""
    ids, _ = instance_map(r.case)
    return r.image == ids[(fov, cell)]


def test_lb7_1_a_retained_cell(runs):
    r = runs("label_pair", 1)
    for fov, cell in (("A", 1), ("B", 2), ("A", 3), ("B", 4), ("A", 12), ("B", 13)):
        assert np.array_equal(shows(r, fov, cell), footprint_view(r, fov) == cell), (fov, cell)
    # label_quad: every retained cell is exactly F_v, except where a later row of the table applies: a retained
    # cell of a FOV earlier in table order lies on it (FOV 1's 1 × 1 px cell at (34, 34) on FOV 4's box 1).
    quad = runs("label_quad", 1)
    _, retained = instance_map(quad.case)
    order = quad.case.ids
    feet = {fov: footprint_view(quad, fov) for fov in order}
    kept = {fov: np.isin(feet[fov], [c for f, c in retained if f == fov]) & (feet[fov] > 0) for fov in order}
    exceptions = set()
    for fov, cell in retained:
        own = feet[fov] == cell
        under = any((own & kept[f]).any() for f in order[:order.index(fov)])
        if under:
            exceptions.add((fov, cell))
            continue
        assert np.array_equal(shows(quad, fov, cell), own), (fov, cell)
    # W-400: under the default duplicate rule FOV 1's 1 × 1 px cell lies whole on FOV 4's box 1 (share 1.0) and is
    # excluded, so box 1 is drawn whole; with the rule off it is retained and drawn over box 1, as before.
    assert exceptions == set() and ("L1", lf.QUAD_DOT_ID) not in retained
    assert shows(quad, "L4", 1).sum() == (feet["L4"] == 1).sum()
    off = runs("label_quad", 1, duplicate_threshold=None)
    _, retained = instance_map(off.case)
    assert ("L1", lf.QUAD_DOT_ID) in retained
    assert shows(off, "L4", 1).sum() == (footprint_view(off, "L4") == 1).sum() - 1


def test_lb7_2_an_excluded_cell(runs):
    r = runs("label_pair", 1)
    ids, retained = instance_map(r.case)
    ids_input, _ = recomputed_ids(r)
    for fov, cell in (("A", 4), ("A", 9), ("B", 3), ("B", 5), ("B", 8), ("B", 11)):
        assert (fov, cell) not in retained and ids[(fov, cell)] not in ids_input
        assert not shows(r, fov, cell).any()
        voxels = library_single(r, fov) == cell
        other = "B" if fov == "A" else "A"
        others = {ids[key] for key in retained if key[0] == other}
        assert set(np.unique(r.image[voxels]).tolist()) <= others | {0}
    on_a4 = library_single(r, "A") == 4
    assert (r.image[on_a4] == ids[("B", 4)]).all()
    quad = runs("label_quad", 1)
    ids, retained = instance_map(quad.case)
    assert ("L1", lf.QUAD_EXTRA_ID) not in retained and not shows(quad, "L1", lf.QUAD_EXTRA_ID).any()
    extra = library_single(quad, "L1") == lf.QUAD_EXTRA_ID
    assert extra.sum() == 16 and (quad.image[extra] == ids[("L2", 1)]).all()      # FOV 2's box at rows 4 to 9


def test_lb7_3_two_retained_cells_whose_territories_overlap(runs):
    r = runs("label_pair", 1)
    a8, b9 = library_single(r, "A") == 8, library_single(r, "B") == 9
    shared = a8 & b9
    assert shared.sum() == 1 and (r.image[shared] == instance_map(r.case)[0][("A", 8)]).all()
    assert shows(r, "B", 9).sum() == 80 and b9.sum() == 81


@pytest.mark.parametrize("scale", SCALES)
def test_lb7_4_a_cell_retained_by_two_fovs(runs, scale):
    r = runs("label_pair", scale)
    _, retained = instance_map(r.case)
    assert {("A", 6), ("B", 6)} <= retained
    a, b = shows(r, "A", 6), shows(r, "B", 6)
    assert a.any() and b.any()
    if scale == 1:
        assert np.array_equal(a, footprint_view(r, "A") == 6) and a.sum() == 49
        assert np.array_equal(b, (library_single(r, "B") == 6) & ~a) and b.sum() == 37
    # W-400: with the rule off, label_pair_conflict retains both views of cell 6, flagged conflict and both drawn;
    # under the default rule B's view (share 0.53, equal sizes, B second in table order) is excluded.
    conflict = runs("label_pair_conflict", 1, duplicate_threshold=None)
    flags = conflict.case.assembly.cells.set_index(["fov_id", "cell_id"]).assembly_flags
    assert all("conflict" in flags[(fov, 6)].split(";") for fov in ("A", "B"))
    assert shows(conflict, "A", 6).any() and shows(conflict, "B", 6).any()
    ruled = runs("label_pair_conflict", 1)
    cells = ruled.case.assembly.cells.set_index(["fov_id", "cell_id"])
    assert (cells.loc[("B", 6), "assembly_reason"], cells.loc[("B", 6), "duplicate_of"]) == \
        ("duplicate_observation", "label_pair_conflict_A_6")
    assert "conflict" not in cells.loc[("A", 6), "assembly_flags"].split(";")
    assert shows(ruled, "A", 6).any() and not shows(ruled, "B", 6).any()


@pytest.mark.parametrize("scale", SCALES)
def test_lb7_5_a_cell_retained_by_no_fov(runs, scale):
    r = runs("label_pair", scale)
    cells = r.case.assembly.cells.set_index(["fov_id", "cell_id"])
    for fov in ("A", "B"):
        row = cells.loc[(fov, 7)]
        assert row.assembly_status == "excluded" and row.assembly_reason == "not_owner"
        assert "unclaimed" in row.assembly_flags.split(";")
        assert not shows(r, fov, 7).any()
        assert not r.image[library_single(r, fov) == 7].any()


def test_lb7_6_a_centroid_on_an_ownership_boundary(runs):
    r = runs("label_pair", 1)
    cells = r.case.assembly.cells.set_index(["fov_id", "cell_id"])
    assert cells.loc[("A", 5)].assembly_status == "retained"
    assert cells.loc[("B", 5)].assembly_status == "excluded" and cells.loc[("B", 5)].assembly_reason == "not_owner"
    a5 = shows(r, "A", 5)
    assert a5.sum() == 62 and np.array_equal(a5, footprint_view(r, "A") == 5)
    x = centres(r)[2][np.nonzero(a5)[2]] / lf.LN_XY
    assert (x > 79.5).any()                                     # drawn also beyond x 79.5, in B's region


def test_lb7_7_a_cell_cut_by_its_fovs_border(runs):
    r = runs("label_pair", 1)
    cells = r.case.assembly.cells.set_index(["fov_id", "cell_id"])
    assert cells.loc[("B", 10)].assembly_status == "retained"
    assert "fov_edge" in cells.loc[("B", 10)].assembly_flags.split(";")
    assert shows(r, "B", 10).sum() == 35 == int((r.case.territories["B"] == 10).sum())
    assert cells.loc[("B", 11)].assembly_status == "excluded"
    assert "fov_edge" in cells.loc[("B", 11)].assembly_flags.split(";")
    assert np.array_equal(shows(r, "A", 11), footprint_view(r, "A") == 11) and shows(r, "A", 11).sum() == 49


def test_lb7_9_a_placeholder_fov(runs, cases, monkeypatch):
    case = cases("label_quad", placeholder=True)
    read = []
    original = assignment_persist.read_labels
    monkeypatch.setattr(assignment_persist, "read_labels", lambda path: read.append(str(path)) or original(path))
    r = run(case, 1, folder=case.root / "export" / "placeholder_counted")
    assert str(case.territory_path("L4")) not in read
    assert sorted(read) == sorted(str(case.territory_path(f)) for f in ("L1", "L2", "L3"))
    ids, retained = instance_map(case)
    assert not [key for key in retained if key[0] == "L4"]
    l4 = {v for key, v in ids.items() if key[0] == "L4"}
    assert not set(np.unique(r.image).tolist()) & l4
    # Output voxels whose centre lies only in L4's footprint hold 0.
    c = centres(r)
    inside = {}
    for fov in case.ids:
        n, o = case.territories[fov].shape, case.origins[fov]
        inside[fov] = ((c[1][:, None] >= o[1] - 0.5) & (c[1][:, None] < o[1] + n[1] - 0.5)
                       & (c[2][None, :] >= o[2] - 0.5) & (c[2][None, :] < o[2] + n[2] - 0.5))
    only = inside["L4"] & ~inside["L1"] & ~inside["L2"] & ~inside["L3"]
    assert only.any() and not r.image[0][only].any()
    # A retained cell of FOVs 1 to 3 that reaches into L4's footprint is drawn as its FOV holds it.
    reaching = 0
    for fov, cell in retained:
        here = shows(r, fov, cell)
        assert np.array_equal(here, footprint_view(r, fov) == cell)
        reaching += bool((here[0] & inside["L4"]).any())
    assert reaching > 0
    check_no_new_value(r)
    check_drawn_by_retained_cells(r)


def test_lb7_10_background(runs):
    for name in ("label_pair", "label_pair_empty", "label_quad"):
        r = runs(name, 1)
        ids, retained = instance_map(r.case)
        drawn = np.zeros(r.image.shape, bool)
        for fov in included(r.case):
            single = library_single(r, fov)
            mine = [cell for f, cell in retained if f == fov]
            drawn |= np.isin(single, mine) & (single > 0)
        assert not r.image[~drawn].any(), name
    # label_quad: every voxel uncovered by F3's rule applied to the label grid holds 0.
    r = runs("label_quad", 1)
    c = centres(r)
    covered = np.zeros(r.image.shape[1:], bool)
    for fov in included(r.case):
        n, o = r.case.territories[fov].shape, r.case.origins[fov]
        covered |= ((c[1][:, None] >= o[1] - 1e-6) & (c[1][:, None] <= o[1] + n[1] - 1 + 1e-6)
                    & (c[2][None, :] >= o[2] - 1e-6) & (c[2][None, :] <= o[2] + n[2] - 1 + 1e-6))
    assert (~covered).any() and not r.image[0][~covered].any()


@pytest.mark.parametrize("name", ["label_pair", "label_quad"])
def test_lb7_11_an_assembly_made_without_conflicts(runs, name):
    with_flags, without = runs(name, 1), runs(name, 1, conflicts=False)
    assert without.case.assembly.record["flags"]["conflicts"] is False
    assert all(e["territories"] == {"path": None, "sha256": None} for e in without.case.assembly.record["fovs"])
    assert np.array_equal(with_flags.image, without.image)
    assert without.section["n_conflict"] is None
    for entry in without.section["fovs"]:
        path = without.case.territory_path(entry["fov_id"])
        assert entry["source"] == "assignment_folder" and entry["path"] == str(path)
        assert entry["sha256"] == hash_oracles.file_sha256(path)
    check_export(without.folder)


def test_lb7_12_a_cell_retained_by_no_fov_without_a_flag(runs):
    """label_pair_merged: A segments cells 31 and 32 apart; B segments them as one cell 32 and retains it."""
    r = runs("label_pair_merged", 1)
    cells = r.case.assembly.cells.set_index(["fov_id", "cell_id"])
    rows = cells.loc[[key for key in cells.index if key[1] == 31]]
    assert list(rows.index) == [("A", 31)]
    assert (rows.assembly_status == "excluded").all() and (rows.assembly_reason == "not_owner").all()
    assert not any("unclaimed" in flags.split(";") for flags in rows.assembly_flags)
    assert cells.loc[("B", 32)].assembly_status == "retained"
    assert not shows(r, "A", 31).any() and not shows(r, "A", 32).any()
    # Its area shows the owner's merged cell.
    a31 = library_single(r, "A") == 31
    assert (r.image[a31] == instance_map(r.case)[0][("B", 32)]).all()
    check_export(r.folder)


# --- W-400: the duplicate rule, the label image and the flags ------------------------------------------------------

def page_flags(case):
    """(fov_edge, unclaimed) of every cell by the definitions of design/specs/assembly.md ("Flags"), over the cells
    the cell table retains (after the duplicate rule): fov_edge from the territory's first and last Y and X index;
    unclaimed for a cell excluded not_owner whose ownership centroid has no owner, or whose owner's territory there
    holds 0 or a cell the owner does not retain (sampled at floor(c + ½), Y and X for these plane territories)."""
    cells = case.assembly.cells
    status = {(f, int(c)): s for f, c, s in zip(cells.fov_id, cells.cell_id, cells.assembly_status)}
    out = {}
    for r in cells.itertuples():
        t = case.territories[r.fov_id]
        edge = int(r.cell_id) in set(np.unique(np.concatenate([t[:, [0, -1], :].ravel(), t[:, :, [0, -1]].ravel()])))
        unclaimed = False
        if not pd.isna(r.assembly_reason) and r.assembly_reason == "not_owner":
            unclaimed = True
            if not pd.isna(r.owner_fov_id):
                owner = case.territories[r.owner_fov_id]
                origin, spacing = np.asarray(case.origins[r.owner_fov_id], float), case.spacing(r.owner_fov_id)
                q = np.array([r.sample_ownership_centroid_y, r.sample_ownership_centroid_x])
                iy, ix = np.floor((q - origin[1:]) / spacing[1:] + 0.5).astype(int)
                inside = 0 <= iy < owner.shape[1] and 0 <= ix < owner.shape[2]
                value = int(owner[0, iy, ix]) if inside else 0
                unclaimed = value == 0 or status.get((r.owner_fov_id, value)) != "retained"
        out[(r.fov_id, int(r.cell_id))] = (edge, unclaimed)
    return out


@pytest.mark.parametrize("name", ["label_pair", "label_pair_conflict"])
def test_w400_the_excluded_observation_is_not_drawn(runs, name):
    """label_pair (no pair at 0.5 or above) and label_pair_conflict (cell 6's views share 0.53; B's is excluded)."""
    r, off = runs(name, 1), runs(name, 1, duplicate_threshold=None)
    ids, retained = instance_map(r.case)
    cells = r.case.assembly.cells.set_index(["fov_id", "cell_id"])
    duplicate = cells.assembly_reason.eq("duplicate_observation").fillna(False)
    excluded = [key for key in ids if duplicate[key]]
    assert excluded == ([("B", 6)] if name == "label_pair_conflict" else [])
    drawn = {int(v) for v in np.unique(r.image) if v > 0}
    ids_input, _ = recomputed_ids(r)
    assert r.section["n_ids_input"] == len(ids_input) == len(retained)
    assert drawn == ids_input - set(r.section["absent_ids"])
    for key in excluded:
        assert ids[key] not in drawn and ids[key] not in ids_input and ids[key] not in r.section["absent_ids"]
        assert not shows(r, *key).any() and shows(off, *key).any()
    n_retained = {e["fov_id"]: e["n_retained"] for e in r.section["fovs"]}
    assert n_retained == {f: n - sum(k[0] == f for k in excluded)
                          for f, n in ((e["fov_id"], e["n_retained"]) for e in off.section["fovs"])}
    # The image changes only where the excluded observation was drawn, which now shows nothing.
    changed = r.image != off.image
    assert (off.image[changed] == ids[("B", 6)]).all() if excluded else not changed.any()
    assert (r.image[changed] == 0).all()
    # conflict no longer marks the kept cell; fov_edge and unclaimed are as the pages define them after the rule.
    assert r.section["n_conflict"] == 0
    assert "conflict" not in cells.loc[("A", 6), "assembly_flags"].split(";")
    got = {key: ("fov_edge" in cells.loc[key, "assembly_flags"].split(";"),
                 "unclaimed" in cells.loc[key, "assembly_flags"].split(";")) for key in ids}
    assert got == page_flags(r.case)
    check_export(r.folder)


def test_w400_cell_ids_over_the_whole_uint32_range(tmp_path):
    """The two pairs of the duplicate test's ``both_high`` case written as label files (1×24×20, unit spacing, x origins 0
    and 10): P1's cells 2^32 − 1 and 2^31 (4 px wide) each share half of their pixels with P2's cells 2^31 + 5 and
    2^32 − 1 (6 px wide). The default rule excludes both P1 cells; the label image shows the two P2 cells only."""
    top = 2 ** 32 - 1
    territories = {f: np.zeros((1, 24, 20), np.uint32) for f in ("P1", "P2")}
    territories["P1"][0, 9:11, 11:15], territories["P1"][0, 13:15, 11:15] = top, 2 ** 31
    territories["P2"][0, 9:11, 3:9], territories["P2"][0, 13:15, 3:9] = 2 ** 31 + 5, top
    case = lf.build(tmp_path, "large_ids", territories, {"P1": (0.0, 0.0, 0.0), "P2": (0.0, 0.0, 10.0)})
    r = run(case, 1)
    ids, retained = instance_map(case)
    cells = case.assembly.cells.set_index(["fov_id", "cell_id"])
    assert retained == {("P2", 2 ** 31 + 5), ("P2", top)}
    assert cells.loc[("P1", top), "duplicate_of"] == f"large_ids_P2_{2 ** 31 + 5}"
    assert cells.loc[("P1", 2 ** 31), "duplicate_of"] == f"large_ids_P2_{top}"
    assert {int(v) for v in np.unique(r.image) if v > 0} == {ids[key] for key in retained}
    for key in retained:
        assert shows(r, *key).sum() == 12
    check_export(r.folder)


# --- EX1 and EX3: the label part inside the entry ------------------------------------------------------------------

DEFAULT_FILES = ["export.json", "reference_stain_projection.ome.tif", "sample.h5ad", "tables/cells.csv",
                 "tables/counts.csv", "tables/fovs.csv", "tables/molecules.csv"]
TABLE_FILES = ["export.json", "sample.h5ad", "tables/cells.csv", "tables/counts.csv", "tables/fovs.csv",
               "tables/molecules.csv"]


def export(f, config=None, **kwargs):
    return f.dataset.export("sample1", placement=kwargs.pop("placement", f.placement_dir),
                            config=ExportConfig(route="per_fov") if config is None else config,
                            output_dir=kwargs.pop("output_dir", f.export_dir), **kwargs)


def record_of(f):
    return json.loads((f.export_dir / "export.json").read_text())


@needs_anndata
def test_ex1_the_label_part_runs_third(tmp_path, monkeypatch):
    folder = export_folder(tmp_path)
    calls = []
    real_assemble, real_fuse, real_labels = starfinder.assembly.assemble, export_module.fuse_images, export_module.fuse_labels
    monkeypatch.setattr(starfinder.assembly, "assemble", lambda *a, **k: calls.append("assemble") or
                        real_assemble(*a, **k))
    monkeypatch.setattr(export_module, "fuse_images", lambda *a, **k: calls.append("fuse") or real_fuse(*a, **k))
    monkeypatch.setattr(export_module, "fuse_labels", lambda *a, **k: calls.append("fuse_labels") or
                        real_labels(*a, **k))
    record = export(folder, ExportConfig(route="per_fov", fuse_labels=True))
    assert calls == ["assemble", "fuse", "fuse_labels"]
    assert sorted(listing(folder.export_dir)) == sorted(DEFAULT_FILES + ["cell_labels.ome.tif"])
    assert record.parts == {"assemble": "written", "fuse": "written", "segment": "off", "fuse_labels": "written",
                            "export_spatialdata": "off"} and record.complete
    labels = record.record["labels"]
    assert labels["assembly_sha256"] == hash_oracles.section_sha256(record.record["assembly"])
    assert [o["path"] for o in labels["outputs"]] == ["cell_labels.ome.tif"]
    check_export(folder.export_dir)
    # Called alone with the saved files of the parts before it: its section only changes.
    before = record_of(folder)
    placement = load_placement(folder.placement_dir)
    fuse_labels(placement, load_assembly(folder.export_dir), config=LabelConfig(scale_yx=1),
                output_dir=folder.export_dir)
    after = record_of(folder)
    for name in ("assembly", "fusion", "spatialdata"):
        assert canonical(after[name]) == canonical(before[name])
    assert after["labels"]["labels_scale_yx"] == 1 and after["labels"]["assembly_sha256"] == \
        hash_oracles.section_sha256(after["assembly"])
    check_export(folder.export_dir)


def territory_of(folder, fov="Position002"):
    return folder.assignment_dir(fov) / "cell_labels.tif"


def replace_territory(folder, fov="Position002"):
    target = territory_of(folder, fov)
    labels = tifffile.imread(territory_of(folder, "Position001"))
    save_volume(labels.astype(np.uint32), target)
    return target


@needs_anndata
@pytest.mark.parametrize("change", ["removed", "replaced"])
def test_ex3_c_e_the_saved_assembly_meets_a_changed_territory(tmp_path, change):
    """(c) and (e): with conflicts true and assemble off, check 4's load_assembly meets the fault first."""
    folder = export_folder(tmp_path)
    export(folder, ExportConfig(route="per_fov", fuse=False))
    if change == "removed":
        territory_of(folder).unlink()
        match = f"{re.escape(str(territory_of(folder)))} is missing"
    else:
        replace_territory(folder)
        match = f"{re.escape(str(territory_of(folder)))} has SHA-256 [0-9a-f]{{64}}, the record [0-9a-f]{{64}}"
    before = listing(folder.root)
    with pytest.raises(ValueError, match=match):
        export(folder, ExportConfig(route="per_fov", assemble=False, fuse=False, fuse_labels=True))
    assert listing(folder.root) == before


@needs_anndata
def test_ex3_f_the_assemble_part_meets_a_changed_territory(tmp_path):
    folder = export_folder(tmp_path)
    export(folder, ExportConfig(route="per_fov", fuse=False))
    territory_of(folder).unlink()
    tables = listing(folder.export_dir)
    with pytest.raises(ValueError, match=re.escape(str(territory_of(folder)))):
        export(folder, ExportConfig(route="per_fov", fuse=False, fuse_labels=True))
    assert listing(folder.export_dir) == tables


@needs_anndata
def test_ex3_g_a_territory_missing_without_conflicts(tmp_path):
    folder = export_folder(tmp_path)
    export(folder, ExportConfig(route="per_fov", fuse=False, assembly=AssemblyConfig(conflicts=False)))
    assert all(e["territories"] == {"path": None, "sha256": None} for e in record_of(folder)["assembly"]["fovs"])
    territory_of(folder).unlink()
    with pytest.warns(ExportIncompleteWarning, match="fuse_labels could not run: territory_missing") as caught:
        record = export(folder, ExportConfig(route="per_fov", assemble=False, fuse=False, fuse_labels=True))
    assert len([w for w in caught if w.category is ExportIncompleteWarning]) == 1
    assert record.parts == {"assemble": "reused", "fuse": "off", "segment": "off", "fuse_labels": "not_run",
                            "export_spatialdata": "off"}
    labels = record.record["labels"]
    assert labels["reason"] == "territory_missing" and labels["outputs"] == []
    assert labels["missing"] == [{"fov_id": "Position002", "path": str(territory_of(folder))}]
    assert record.record["complete"] is False and sorted(listing(folder.export_dir)) == TABLE_FILES
    check_export(folder.export_dir)


@needs_anndata
def test_ex3_h_a_replaced_territory_without_conflicts(tmp_path):
    folder = export_folder(tmp_path)
    export(folder, ExportConfig(route="per_fov", fuse=False, assembly=AssemblyConfig(conflicts=False)))
    target = replace_territory(folder)
    with pytest.raises(ValueError, match=f"{re.escape(str(target))} has SHA-256"):
        export(folder, ExportConfig(route="per_fov", assemble=False, fuse=False, fuse_labels=True))
    exported = record_of(folder)
    assert exported["parts"]["assemble"] == "reused" and exported["parts"]["fuse_labels"] == "failed"
    assert exported["labels"]["error"]["type"] == "ValueError"
    assert str(target) in exported["labels"]["error"]["message"] and exported["complete"] is False
    assert sorted(listing(folder.export_dir)) == TABLE_FILES
    check_export(folder.export_dir)


@pytest.mark.contract
def test_fuse_labels_without_the_extra(cases, monkeypatch, tmp_path):
    """fuse_labels alone without multiview-stitcher: not_run, extra_missing, the warning names the extra."""
    case = cases("label_pair")
    folder = tmp_path / "no_extra"
    saved = save_assembly(case.assembly, folder)
    monkeypatch.setitem(sys.modules, "multiview_stitcher", None)
    with pytest.warns(ExportIncompleteWarning, match="fuse_labels could not run: extra_missing"):
        record = fuse_labels(case.placement, saved, output_dir=folder)
    assert record.status == "not_run" and record.section["extra"] == "multiview-stitcher" and record.paths == ()
    check_export(folder)
