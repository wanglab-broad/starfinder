"""SO2 and SO4 of docs/sample-segmentation-validation.md: the readers and the label export on the route after fusion.

SO2: ``load_assembly``, ``sample_anndata``, ``check_export`` (R1 to R3, F, L, S and T1 to T5), ``lookup_source``
and ``export_spatialdata`` with its round trip, on ``fused_pair`` exported on both routes (``per_fov_routes``), the
route after fusion at ``k = 1`` and at ``k = 2`` (where the segmentation grid's spacing differs from a FOV's, so the
radii tell the two apart), each built once; ``check_export`` fails T1 on a folder whose fused input was fused
again with other bytes after the segmentation, fails T5 on one whose label file the ``assembly`` section does not
record, and passes after a reuse with ``fuse`` on and off; on ``fused_quad`` under ``ownership="midpoint_box"`` with
each tie, T4 passes, and fails on a copy whose ``assembly`` section records the other tie (W-446). SO4: the label
export on the route after fusion, ``fused_pair`` and ``fused_quad``
with ``truth_import``, ``fuse_labels`` at ``labels_scale_yx`` 1, 0.5 and 0.25; the expected label image is
recomputed in the test from the sample's territories by ``floor(c + ½)``.

Fixtures and helpers: ``sample_segmentation_fixtures.py`` (W-425 to W-427) and the helpers of
``test_sample_segmentation_route.py`` (W-427), imported, not rebuilt. No model is loaded or run; dask runs with the
synchronous scheduler.
"""
import dataclasses
import importlib.util
import json
import math
import shutil

import numpy as np
import pandas as pd
import pytest
import tifffile

from starfinder.assembly import AssemblyConfig, load_assembly, ownership_map
from starfinder.assembly._anndata import sample_anndata
from starfinder.sample_output import LabelConfig, check_export, lookup_source, segment_fused
from starfinder.stitching import load_placement

from .fusion_fixtures import listing
from .sample_segmentation_fixtures import (
    BIG,
    DS,
    QUAD_CELLS,
    SAMPLE,
    per_fov_routes,
    quad_grid_placement,
    route_config,
    scene_truth_labels,
    seeded_runs,
    truth_import,
    truth_runs,
    write_folder,
)
from .test_sample_segmentation_route import EXPANSION, export, export_grid, record_of, truth_config

pytestmark = [pytest.mark.stitching, pytest.mark.validation]
needs_library = pytest.mark.skipif(importlib.util.find_spec("multiview_stitcher") is None,
                                   reason="the multiview-stitcher extra (starfinder[multiview-stitcher]) is not "
                                          "installed")
needs_anndata = pytest.mark.skipif(importlib.util.find_spec("anndata") is None,
                                   reason="the anndata extra (starfinder[anndata]) is not installed")
needs_spatialdata = pytest.mark.skipif(importlib.util.find_spec("spatialdata") is None,
                                       reason="the spatialdata extra (starfinder[spatialdata]) is not installed")
ROUTE_CASES = ("per_fov", "after_fusion_k1", "after_fusion_k2")
#: The tie of ``midpoint_box`` → the owner of ``fused_quad``'s cell at the crossing, whose ownership centroid (17.5,
#: 17.5) µm lies on FOV2's upper X box edge (closed under ``table_order``) and on FOV4's lower box edges.
CROSSING_OWNER = {"table_order": "FOV2", "half_open": "FOV4"}


@pytest.fixture(autouse=True)
def synchronous():
    import dask
    with dask.config.set(scheduler="synchronous"):
        yield


def with_outputs(config, *, labels=1.0, spatialdata=True):
    """A route configuration with the label export at ``labels_scale_yx`` and the SpatialData export on."""
    return dataclasses.replace(config, fuse_labels=labels is not None,
                               labels=LabelConfig(scale_yx=labels if labels is not None else 0.25),
                               export_spatialdata=spatialdata)


def instance_of(cells):
    """``(cell_namespace, cell_id)`` → ``instance_id``, ``1..N`` in the cell table's order, recomputed here."""
    return {(ns, int(c)): i + 1 for i, (ns, c) in enumerate(zip(cells["cell_namespace"].astype(object),
                                                                cells["cell_id"].astype("int64")))}


# --- SO2: readers -------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def so2(tmp_path_factory):
    """``so2(case)``: ``fused_pair`` exported with the label and SpatialData exports on one route (built once per
    case): the route after fusion at ``k = 1`` or ``k = 2`` with ``truth_import``, or the route per FOV
    (``per_fov_routes``)."""
    cache = {}

    def get(case):
        if case not in cache:
            folder = write_folder(tmp_path_factory.mktemp(case), "fused_pair")
            if case == "per_fov":
                per_fov_routes(folder)
                config = route_config(route="per_fov")
            else:
                k = 2 if case == "after_fusion_k2" else 1
                config = route_config(segmentation=truth_config(folder, export_grid(folder, k)), k=k)
            export(folder, with_outputs(config))
            cache[case] = folder
        return cache[case]
    return get


@needs_spatialdata
@needs_anndata
@needs_library
@pytest.mark.parametrize("case", ROUTE_CASES)
def test_so2_readers(so2, case, tmp_path):
    import anndata
    import spatialdata
    folder = so2(case)
    out = folder.export_dir
    record = record_of(out)
    after = case != "per_fov"
    assert record["complete"] is True
    assert record["parts"]["fuse_labels"] == "written" and record["parts"]["export_spatialdata"] == "written"
    assert record["parts"]["segment"] == ("written" if after else "off")
    assert record["labels"]["rule"] == ("segmentation" if after else "retained_cells")
    # check_export: R1 to R3, F, L, S and, on the route after fusion, T1 to T5.
    check_export(out)
    # load_assembly and sample_anndata.
    result = load_assembly(out)
    cells, molecules = result.cells, result.molecules
    assert result.record["route"] == ("after_fusion" if after else "per_fov") and len(cells) > 0
    written = sample_anndata(result, tmp_path / "sample.h5ad")
    adata = anndata.read_h5ad(written)
    assert sorted(adata.layers) == ["cytoplasm", "nucleus"]   # the nucleus run (after fusion) or the FOVs' nuclei
    # lookup_source of every cell and molecule.
    for i, row in enumerate(cells.itertuples(index=False)):
        found = lookup_source(out, cell=(row.cell_namespace, int(row.cell_id)))
        assert found["kind"] == "cell" and found["fov_id"] == row.fov_id
        if after:
            assert found["assignment_folder"] is None and found["assignment_record_sha256"] is None
            assert found["row"] == i
            assert found["label_file"] == "segmentation_cell.ome.tif" and found["label_value"] == int(row.cell_id)
            assert json.loads(row.cell_namespace)[-1] == "cell"
        else:
            assert found["assignment_folder"] is not None and "label_file" not in found and "label_value" not in found
    for row in molecules.itertuples(index=False):
        found = lookup_source(out, spot=(row.spot_namespace, row.spot_id))
        same_fov = molecules["fov_id"].astype(object).eq(row.fov_id).to_numpy()
        keys = list(zip(molecules["spot_namespace"][same_fov], molecules["spot_id"][same_fov]))
        assert found["fov_id"] == row.fov_id and found["row"] == keys.index((row.spot_namespace, row.spot_id))
        assert (found["assignment_folder"] is None) is after and "label_file" not in found
    # The SpatialData round trip: cell_instance_id from the cell key, radii from the grid the cells were counted on.
    with pd.option_context("future.infer_string", False):
        sdata = spatialdata.read_zarr(out / "sample.zarr")
    of = instance_of(cells)
    points = sdata.points["molecules"].compute().reset_index(drop=True)
    for row in points.itertuples(index=False):
        expected = None if pd.isna(row.cell_id) else of[(row.cell_namespace, int(row.cell_id))]
        got = None if pd.isna(row.cell_instance_id) else int(row.cell_instance_id)
        assert got == expected
    if after:
        grid = record["segmentation"]["inputs"]["grid"]
        sz, sy, sx = grid["spacing_zyx"]
        assert (sy, sx) == ((0.5, 0.5) if case == "after_fusion_k1" else (1.0, 1.0))
    else:
        sz, sy, sx = folder.grids["FOV1"].metadata.spacing_zyx
    placed = np.isfinite(cells[["sample_ownership_centroid_y", "sample_ownership_centroid_x"]].to_numpy(np.float64))
    rows = cells[placed.all(axis=1)]
    plane = ~np.isfinite(rows["sample_centroid_z"].to_numpy(np.float64))
    assert plane.all() if after else not plane.any()   # planes on the segmentation grid; the FOVs' volumes
    expected = np.array([math.sqrt(v * sy * sx / math.pi) if p else (3 * v * sz * sy * sx / (4 * math.pi)) ** (1 / 3)
                         for v, p in zip(rows["size_voxels"].to_numpy(np.float64), plane)])
    shapes = sdata.shapes["cell_centroids"]
    np.testing.assert_allclose(shapes["radius"].to_numpy(np.float64), expected, rtol=1e-12, atol=0)
    assert [int(v) for v in shapes.index] == [of[(ns, int(c))] for ns, c in zip(rows["cell_namespace"],
                                                                                rows["cell_id"].astype("int64"))]
    assert "cell_labels" in sdata.labels


@needs_anndata
@needs_library
def test_so2_check_export_fails_t1_on_a_fused_input_fused_again(tmp_path):
    """The cell run of ``seeded_runs`` reads the fused flamingo; after the export, flamingo is edited and ``fuse`` is
    called alone again, so the fused file has other bytes than the segmentation was made from (its record is
    consistent, so R1, R2 and F pass)."""
    from .sample_segmentation_fixtures import fuse_scene
    folder = write_folder(tmp_path, "fused_pair")
    grid = export_grid(folder)
    nuclei = truth_import(folder, grid, "nucleus", labels=scene_truth_labels(folder, grid, "nucleus"))
    export(folder, with_outputs(route_config(segmentation=seeded_runs(nuclei)), spatialdata=False))
    check_export(folder.export_dir)
    for fov in folder.scene.ids:
        path = folder.image_path("flamingo", fov)
        image = tifffile.imread(path)
        tifffile.imwrite(path, (image + 1).astype(image.dtype), photometric="minisblack", metadata={"axes": "ZYX"})
    fuse_scene(folder)
    with pytest.raises(ValueError, match="check_export T1:.*'flamingo'.*another fusion"):
        check_export(folder.export_dir)


@needs_anndata
@needs_library
def test_so2_check_export_fails_t5_on_a_label_file_the_assembly_does_not_record(tmp_path):
    """After the export, ``segment_fused`` is called alone with the truth cells less cell 3: the label file is
    rewritten and recorded by the ``segmentation`` section, but not by the ``assembly`` and ``labels`` sections."""
    folder = write_folder(tmp_path, "fused_pair")
    grid = export_grid(folder)
    export(folder, with_outputs(route_config(segmentation=truth_config(folder, grid)), spatialdata=False))
    check_export(folder.export_dir)
    labels = scene_truth_labels(folder, grid, "cell")
    labels[labels == 3] = 0
    cells = truth_import(folder, grid, "cell", path=folder.root / "truth" / "cell_less_3.ome.tif", labels=labels)
    segment_fused(folder.export_dir, config=truth_runs(cells, folder.root / "truth" / "nucleus.ome.tif"),
                  sample=SAMPLE, identity=(DS, SAMPLE))
    with pytest.raises(ValueError, match="check_export T5:.*assembly section's segmentation_files"):
        check_export(folder.export_dir)


@pytest.fixture(scope="module")
def so2_midpoint(tmp_path_factory):
    """``so2_midpoint(tie)``: ``fused_quad`` exported on the route after fusion at ``k = 1`` with ``truth_import`` and
    ``AssemblyConfig(ownership="midpoint_box", tie=tie)``, on the placement of ``quad_grid_placement`` (built once
    per tie). Returns the export folder, the scene folder and the placement folder."""
    cache = {}

    def get(tie):
        if "scene" not in cache:
            folder = write_folder(tmp_path_factory.mktemp("fused_quad_midpoint"), "fused_quad")
            cache["scene"] = (folder, quad_grid_placement(folder), truth_config(folder, export_grid(folder)))
        folder, placement, segmentation = cache["scene"]
        if tie not in cache:
            out = folder.root / f"export_{tie}" / SAMPLE
            config = dataclasses.replace(route_config(segmentation=segmentation),
                                         assembly=AssemblyConfig(ownership="midpoint_box", tie=tie))
            export(folder, config, output_dir=out, placement=placement)
            cache[tie] = out
        return cache[tie], folder, placement
    return get


@needs_anndata
@needs_library
@pytest.mark.parametrize("tie", list(CROSSING_OWNER))
def test_so2_check_export_t4_under_midpoint_box(so2_midpoint, tie):
    """T4 under ``midpoint_box``: ``check_export`` rebuilds the boxes from the ``assembly`` section, the upper edges
    set at a neighbour's midpoint closed under ``table_order`` and open under ``half_open``. The cell at the crossing
    has its ownership centroid inside all four footprints, so the boxes decide its owner, and the tie decides which."""
    out, folder, placement = so2_midpoint(tie)
    section = record_of(out)["assembly"]
    assert section["ownership"]["rule"] == "midpoint_box" and section["ownership"]["tie"] == tie
    check_export(out)
    cells = load_assembly(out).cells
    points = cells[["sample_ownership_centroid_z", "sample_ownership_centroid_y",
                    "sample_ownership_centroid_x"]].to_numpy(np.float64)
    # The half-open footprints the assembly section records holding each centroid.
    holding = np.zeros(len(cells), np.int64)
    for entry in section["fovs"]:
        (y0, y1), (x0, x1) = entry["footprint_yx"]
        holding += (points[:, 1] >= y0) & (points[:, 1] < y1) & (points[:, 2] >= x0) & (points[:, 2] < x1)
    assert (holding > 1).any()
    crossing = cells["cell_id"].astype("int64").eq(BIG).to_numpy()
    assert holding[crossing].tolist() == [4]
    # The owners the assembly gave are those of the ownership map of the placement it read, not of the record.
    owners = ownership_map(load_placement(placement), folder.grids,
                           config=AssemblyConfig(ownership="midpoint_box", tie=tie)).owner(points).tolist()
    assert None not in owners
    assert cells["owner_fov_id"].astype(object).tolist() == owners == cells["fov_id"].astype(object).tolist()
    assert cells["owner_fov_id"][crossing].tolist() == [CROSSING_OWNER[tie]]


@needs_anndata
@needs_library
@pytest.mark.parametrize("tie", list(CROSSING_OWNER))
def test_so2_check_export_fails_t4_on_a_cell_the_recorded_rule_gives_another_owner(so2_midpoint, tie, tmp_path):
    """A copy of the export of one tie whose ``assembly`` section is edited to record the other tie: the tables keep
    the owners the assembly gave, and the recorded rule gives the cell at the crossing the other FOV. Only
    ``export.json`` changes and no other section records the ``assembly`` section's SHA-256 (no label or SpatialData
    export), so R1 to R3, T1, T5, T2 and T3 pass and T4 is the first condition that fails."""
    out, _, _ = so2_midpoint(tie)
    copy = tmp_path / SAMPLE
    shutil.copytree(out, copy)
    check_export(copy)
    path = copy / "export.json"
    record = json.loads(path.read_text())
    assert all((record.get(name) or {}).get("assembly_sha256") is None for name in ("fusion", "labels", "spatialdata"))
    other = next(t for t in CROSSING_OWNER if t != tie)
    record["assembly"]["ownership"]["tie"] = other
    path.write_text(json.dumps(record, indent=2))
    got, rule = CROSSING_OWNER[tie], CROSSING_OWNER[other]
    with pytest.raises(ValueError, match=f"^check_export T4: cell {SAMPLE}_{got}_{BIG}: fov_id '{got}' and "
                                         f"owner_fov_id '{got}'; the rule of the cells gives '{rule}' and '{rule}'$"):
        check_export(copy)


@needs_anndata
@needs_library
@pytest.mark.parametrize("fuse_on", [True, False])
def test_so2_check_export_after_a_reuse(tmp_path, fuse_on):
    """``check_export`` passes on an untouched folder after a reuse (``segment: false``), with the fused files present
    (T1 compares them with the recorded inputs) and with ``fuse`` off (T1 checks the label files alone)."""
    folder = write_folder(tmp_path, "fused_pair")
    grid = export_grid(folder)
    nuclei = truth_import(folder, grid, "nucleus", labels=scene_truth_labels(folder, grid, "nucleus"))
    segmentation = seeded_runs(nuclei)
    export(folder, with_outputs(route_config(segmentation=segmentation), spatialdata=False))
    record = export(folder, with_outputs(route_config(segmentation=segmentation, segment=False, fuse=fuse_on),
                                         spatialdata=False))
    assert record["parts"]["segment"] == "reused" and record["complete"] is True
    assert record["parts"]["fuse"] == ("written" if fuse_on else "off")
    assert record["parts"]["fuse_labels"] == "written"
    assert any(e["image"] == "flamingo" for e in record["segmentation"]["inputs"]["images"])
    check_export(folder.export_dir)


@needs_anndata
@needs_library
def test_so2_lookup_source_of_identifiers_at_and_above_2_31(so4):
    """``lookup_source(cell=…)`` on the route after fusion keeps the cell IDs 2^31 and 2^32 − 1 of
    ``fused_quad``."""
    out = so4("fused_quad", 1.0)
    result = load_assembly(out)
    namespace = json.dumps([DS, SAMPLE, None, None, "cell"], separators=(",", ":"))
    for cell in (BIG, 2 ** 32 - 1):
        assert cell in QUAD_CELLS
        found = lookup_source(out, cell=(namespace, cell))
        row = result.cells.index[result.cells["cell_id"].astype("int64").eq(cell)][0]
        assert found["label_value"] == cell and found["row"] == int(row) and found["assignment_folder"] is None


# --- SO4: the label export on the route after fusion ---------------------------------------------------------------

@pytest.fixture(scope="module")
def so4(tmp_path_factory):
    """``so4(scene, scale, expansion=False)``: the folder of the scene's export on the route after fusion at
    ``k = 1`` with ``truth_import`` and ``fuse_labels`` at ``labels_scale_yx`` ``scale`` (built once)."""
    folders, cache = {}, {}

    def get(scene, scale, expansion=False):
        key = (scene, scale, expansion)
        if key not in cache:
            if (scene, expansion) not in folders:
                folder = write_folder(tmp_path_factory.mktemp(scene), scene)
                changes = {"assignment": EXPANSION} if expansion else {}
                folders[(scene, expansion)] = (folder, truth_config(folder, export_grid(folder), **changes))
            folder, segmentation = folders[(scene, expansion)]
            out = folder.root / f"export_{scale}" / SAMPLE
            export(folder, with_outputs(route_config(segmentation=segmentation), labels=scale, spatialdata=False),
                   output_dir=out)
            cache[key] = out
        return cache[key]
    return get


def expected_label_image(out, record):
    """The label image recomputed from the sample's territories (the cell run's labels, or territories.ome.tif):
    ``cell_id`` mapped to ``instance_id``, sampled at each output pixel's centre by ``floor(c + ½)``; with the grid
    of the label scale's spacing and the territories' origin."""
    grid = record["segmentation"]["inputs"]["grid"]
    files = record["assembly"]["files"]
    path = files["territories"]["path"] if "territories" in files else "segmentation_cell.ome.tif"
    territory = tifffile.imread(out / path)
    cells = load_assembly(out).cells
    of = {int(c): i + 1 for i, c in enumerate(cells["cell_id"].astype("int64"))}
    mapped = np.zeros(territory.shape, np.uint32)
    for cell, instance in of.items():
        mapped[territory == cell] = instance
    k = int(record["labels"]["factor_yx"])
    (_, sy, sx), (_, oy, ox), (_, ny, nx) = grid["spacing_zyx"], grid["origin_zyx"], grid["shape_zyx"]
    step = (k * 0.5, k * 0.5)   # k·s at the label scale, s the FOVs' 0.5 µm
    shape = (math.floor((ny - 1) * sy / step[0] + 1e-9) + 1, math.floor((nx - 1) * sx / step[1] + 1e-9) + 1)
    iy = np.floor((oy + np.arange(shape[0]) * step[0] - oy) / sy + 0.5).astype(np.int64)
    ix = np.floor((ox + np.arange(shape[1]) * step[1] - ox) / sx + 0.5).astype(np.int64)
    image = mapped[iy.clip(0, ny - 1)[:, None], ix.clip(0, nx - 1)[None, :]]
    image[~(((iy >= 0) & (iy < ny))[:, None] & ((ix >= 0) & (ix < nx))[None, :])] = 0
    return image, mapped, (list(step), [oy, ox], list(shape)), of


@needs_anndata
@needs_library
@pytest.mark.parametrize("scale", [1.0, 0.5, 0.25])
@pytest.mark.parametrize("scene", ["fused_pair", "fused_quad"])
def test_so4_the_label_export_on_the_route_after_fusion(so4, scene, scale):
    out = so4(scene, scale)
    record = record_of(out)
    section = record["labels"]
    assert record["parts"]["fuse_labels"] == "written" and section["rule"] == "segmentation"
    assert section["segmentation_files"] == record["assembly"]["segmentation_files"]
    assert section["labels_scale_yx"] == scale and section["fovs"] == []
    output = section["outputs"][0]
    # L1 (layout, uint32), L2 to L5 and T1 to T5 by check_export.
    check_export(out)
    labels = tifffile.imread(out / "cell_labels.ome.tif")
    assert labels.dtype == np.uint32 and output["dtype"] == "uint32"
    image, mapped, (step, origin, shape), of = expected_label_image(out, record)
    # L5: the label scale's spacing and the territories' origin; Z the segmentation grid's slab.
    grid = record["segmentation"]["inputs"]["grid"]
    assert output["shape_zyx"] == [1, *shape] and list(labels.shape) == shape
    assert output["spacing_zyx"] == pytest.approx([grid["spacing_zyx"][0], *step], abs=1e-9)
    assert output["origin_zyx"] == pytest.approx([grid["origin_zyx"][0], *origin], abs=1e-9)
    # L3 and L2: every pixel is the territories' cell at floor(c + ½), so no value is new.
    np.testing.assert_array_equal(labels, image)
    ids_input = sorted(int(v) for v in np.unique(mapped) if v > 0)
    ids_output = sorted(int(v) for v in np.unique(labels) if v > 0)
    assert set(ids_output) <= set(ids_input) <= set(of.values())
    # L4: the recorded IDs in, out and absent.
    absent = sorted(set(ids_input) - set(ids_output))
    assert (section["n_ids_input"], section["n_ids_output"], section["absent_ids"], section["n_absent_ids"]) == \
        (len(ids_input), len(ids_output), absent, len(absent))
    if scale == 1.0:
        # At scale 1 and k = 1 each cell shows one ID on every pixel of its territory, and nowhere else.
        territory = tifffile.imread(out / "segmentation_cell.ome.tif")
        for cell, instance in of.items():
            assert np.all(labels[territory == cell] == instance) and np.all(territory[labels == instance] == cell)
        assert absent == [] and ids_output == sorted(of.values())


@needs_anndata
@needs_library
def test_so4_with_the_expanded_territories(so4):
    """With an expansion of the assignment, the one view is ``territories.ome.tif``, the territories assign sampled."""
    out = so4("fused_pair", 1.0, expansion=True)
    record = record_of(out)
    assert "territories" in record["assembly"]["files"]
    territories = record["assembly"]["files"]["territories"]
    assert record["labels"]["segmentation_files"][-1] == {"run": None, "path": "territories.ome.tif",
                                                          "sha256": territories["sha256"]}
    check_export(out)
    image, mapped, _, of = expected_label_image(out, record)
    labels = tifffile.imread(out / "cell_labels.ome.tif")
    np.testing.assert_array_equal(labels, image)
    np.testing.assert_array_equal(labels, mapped)   # scale 1 at k = 1: the expanded territories themselves
    cells = tifffile.imread(out / "segmentation_cell.ome.tif")
    assert np.count_nonzero(labels) > np.count_nonzero(cells)


@needs_anndata
@needs_library
def test_so4_l6_the_id_bound(tmp_path, monkeypatch):
    """L6 on the route after fusion (the row SO4 as amended on 2026-10-10), the bound lowered to 1 cell row on
    ``fused_pair`` (3 cells). With a reused segmentation (``segment: false``, with ``assemble`` on and off) it is
    refused before anything is written: the folder is byte for byte unchanged. On a fresh export it is refused
    directly after ``segment``, before ``assemble`` writes: the fused files and the segmentation's files stay with
    their recorded SHA-256, and no table, ``sample.h5ad``, territories, ``cell_labels.ome.tif`` or store exists; the
    record names the part ``fuse_labels`` with the error, ``complete`` is false. The error is the one L6 raises on
    the route per FOV (``check_label_ids`` of the sample cell table)."""
    import starfinder.sample_output._labels as labels_module
    folder = write_folder(tmp_path, "fused_pair")
    segmentation = truth_config(folder, export_grid(folder))
    export(folder, route_config(segmentation=segmentation))
    cells = load_assembly(folder.export_dir).cells
    assert len(cells) == 3
    monkeypatch.setattr(labels_module, "LABEL_ID_BOUND", 1)
    with pytest.raises(ValueError) as per_fov:
        labels_module.check_label_ids(cells)
    # A reused segmentation: refused before anything is written.
    for assemble_on in (True, False):
        before = listing(folder.root)
        with pytest.raises(ValueError) as caught:
            export(folder, with_outputs(route_config(segmentation=segmentation, segment=False,
                                                     assemble=assemble_on)))
        assert str(caught.value) == str(per_fov.value)
        assert listing(folder.root) == before
    # A fresh export: refused after segment, before assemble writes.
    with pytest.raises(ValueError) as caught:
        export(folder, with_outputs(route_config(segmentation=segmentation)))
    assert str(caught.value) == str(per_fov.value)
    record = record_of(folder.export_dir)
    assert record["parts"]["fuse"] == "written" and record["parts"]["segment"] == "written"
    assert record["parts"]["fuse_labels"] == "failed" and record["complete"] is False
    assert record["labels"]["error"] == {"type": "ValueError", "message": str(per_fov.value)}
    for part in ("assemble", "export_spatialdata"):
        assert record["parts"][part] == "not_run" and record[{"assemble": "assembly"}.get(part, "spatialdata")] == \
            {"status": "not_run", "reason": "export_not_finished"}
    kept = {o["path"]: o["sha256"] for o in record["fusion"]["outputs"] + record["segmentation"]["outputs"]}
    assert any(p.startswith("segmentation_") for p in kept) and any(p.startswith("dapi_") for p in kept)
    assert {p: d for p, d in listing(folder.export_dir).items() if p != "export.json"} == kept
    assert not any((folder.export_dir / "tables").glob("*"))   # the earlier export's tables were removed
    for absent in ("sample.h5ad", "territories.ome.tif", "cell_labels.ome.tif", "sample.zarr"):
        assert not (folder.export_dir / absent).exists()
