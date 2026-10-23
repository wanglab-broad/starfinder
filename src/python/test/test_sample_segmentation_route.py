"""RS1, SG4, SG5, SA1 to SA8 and SO1 of docs/sample-segmentation-validation.md: the route after fusion in
``Dataset.export`` (the parts ``fuse``, ``segment`` and ``assemble`` with the sample segmentation), its statuses,
its reuse rule, the sample assignment and the outputs it writes, against the route per FOV of the same fixture.

Fixtures (``sample_segmentation_fixtures.py``): ``fused_pair``, ``fused_quad``, ``fused_quad_notch`` and
``scene_plane`` written as datasets, ``truth_import`` (``truth_runs``) and ``seeded_runs`` at ``k = 1``, and
``per_fov_routes`` (per FOV the truth labels cut at its origin, imported and assigned with the same
``AssignmentConfig``, saved as §2.9 folders and exported with ``route: per_fov`` into a second folder). The
conditions T2, T3 and T4 of docs/sample-segmentation-contract.md ("The record") are evaluated in the tests. The
label export and the SpatialData export are off: on this route they are W-428's. No model is loaded or run; dask
runs with the synchronous scheduler.
"""
import dataclasses
import importlib.util
import json
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import tifffile

from starfinder.assembly import AssemblyConfig, AssemblyResult, assemble, load_assembly, ownership_map
from starfinder.assignment import AssignmentConfig
from starfinder.export import (
    ExportIncompleteWarning,
    SampleSegmentationConfig,
    SampleSegmentationRun,
    fuse_images,
    fused_grid,
    segment_fused,
)
from starfinder.dataset._export import assembly_inputs
from starfinder.segmentation import CellposeConfig, ExpandLabelsConfig, LabelImportConfig, ReferenceGrid
from starfinder._records import file_sha256
from starfinder.stitching import PlacementResult, approve_placement, save_placement

from .fusion_fixtures import listing
from .sample_segmentation_fixtures import (
    BIG,
    DS,
    NOTCH_CELL,
    QUAD_CELLS,
    SAMPLE,
    fusion_config,
    model_files,
    per_fov_routes,
    route_config,
    scene_truth_labels,
    seeded_runs,
    truth_import,
    truth_runs,
    write_folder,
)

pytestmark = [pytest.mark.stitching, pytest.mark.validation]
EXTRA = "the multiview-stitcher extra (starfinder[multiview-stitcher]) is not installed"
needs_library = pytest.mark.skipif(importlib.util.find_spec("multiview_stitcher") is None, reason=EXTRA)
needs_anndata = pytest.mark.skipif(importlib.util.find_spec("anndata") is None,
                                   reason="the anndata extra (starfinder[anndata]) is not installed")
EXPANSION = AssignmentConfig(expansion=ExpandLabelsConfig(distance=1.0, unit="um", mode="planar"))
SA_SCENES = ["fused_pair", "fused_quad", "scene_plane"]
AXES = ("z", "y", "x")


@pytest.fixture(autouse=True)
def synchronous():
    import dask
    with dask.config.set(scheduler="synchronous"):
        yield


def record_of(folder):
    return json.loads((folder / "export.json").read_text())


def export_grid(folder, k=1) -> ReferenceGrid:
    """The segmentation grid of an export at ``1/k``: the projection level of ``fused_grid`` (known before any
    fusion; equal to the fusion section's output, SG1)."""
    from starfinder.image import ImageMetadata
    grid = fused_grid(folder.placement, folder.grids, config=fusion_config(k))
    spacing, origin, shape = grid.level("projection")
    metadata = ImageMetadata(f"sample:{SAMPLE}", spacing, origin, ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
                                                                    (0.0, 0.0, 1.0)), "µm")
    return ReferenceGrid(tuple(shape), metadata, "declared")


def truth_config(folder, grid, **changes) -> SampleSegmentationConfig:
    """``truth_import`` of the cells and nuclei on the grid, imported with ``relabel: false``."""
    cells = truth_import(folder, grid, "cell", labels=scene_truth_labels(folder, grid, "cell"))
    nuclei = truth_import(folder, grid, "nucleus", labels=scene_truth_labels(folder, grid, "nucleus"))
    return truth_runs(cells, nuclei, **changes)


def export(folder, config, *, assignment="default", output_dir=None, placement=None):
    return folder.dataset.export(SAMPLE, placement=placement or folder.placement_dir, assignment=assignment,
                                 config=config, output_dir=output_dir or folder.export_dir).record


def per_fov_dir(folder):
    return folder.root / "export_per_fov" / SAMPLE


def build(tmp_path, name, *, assignment=None, runs="truth"):
    """A fixture through both routes: the route after fusion into ``export/<sample>/`` (the call to
    ``assign_molecules`` captured), the route per FOV into ``export_per_fov/<sample>/``, and the ``molecules_only``
    assembly of the same molecule tables."""
    import starfinder.assembly._fused as fused_module
    folder = write_folder(tmp_path, name)
    assignment = AssignmentConfig() if assignment is None else assignment
    per_fov_routes(folder, assignment=assignment)
    grid = export_grid(folder)
    if runs == "truth":
        segmentation = truth_config(folder, grid, assignment=assignment)
    else:
        nuclei = truth_import(folder, grid, "nucleus", labels=scene_truth_labels(folder, grid, "nucleus"))
        segmentation = seeded_runs(nuclei, assignment=assignment)
    captured = {}
    original = fused_module.assign_molecules

    def capture(molecules, cells, **kwargs):
        result = original(molecules, cells, **kwargs)
        captured.update(molecules=molecules, cells=cells, kwargs=kwargs, result=result)
        return result

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(fused_module, "assign_molecules", capture)
        after = export(folder, route_config(segmentation=segmentation))
    per_fov = export(folder, route_config(route="per_fov"), output_dir=per_fov_dir(folder))
    inputs = assembly_inputs(folder.dataset, folder.placement, None)
    base = assemble(folder.placement, inputs)
    return SimpleNamespace(folder=folder, grid=grid, after=after, per_fov=per_fov, captured=captured, base=base,
                           result=load_assembly(folder.export_dir), per_fov_result=load_assembly(per_fov_dir(folder)),
                           segmentation=segmentation, assignment=assignment)


@pytest.fixture(scope="module")
def routes(tmp_path_factory):
    """``build`` once per fixture name and settings for the module (the tests read the folders only)."""
    cache = {}

    def get(name, **settings):
        key = (name, json.dumps({k: repr(v) for k, v in settings.items()}, sort_keys=True))
        if key not in cache:
            cache[key] = build(tmp_path_factory.mktemp(name), name, **settings)
        return cache[key]
    return get


# --- The conditions T2, T3, T4 (docs/sample-segmentation-contract.md, "The record"), evaluated here --------------

def label_image(folder, path):
    labels = tifffile.imread(folder / path)
    return labels[np.newaxis] if labels.ndim == 2 else labels


def territory_image(folder, record):
    """The territories the sample assignment sampled: ``territories.ome.tif`` when it expanded, else the cell run's
    label file."""
    files = record["assembly"]["files"]
    if "territories" in files:
        return label_image(folder, files["territories"]["path"])
    section = record["segmentation"]
    run = next(r for r in section["runs"] if r["name"] == section["cells"])
    return label_image(folder, run["path"])


def t2(folder, record, cells):
    """T2: the positive values of the cell run's labels are the sample cell table's cell_ids, one row each, all with
    the run's namespace."""
    section = record["segmentation"]
    run = next(r for r in section["runs"] if r["name"] == section["cells"])
    values = np.unique(label_image(folder, run["path"]))
    values = values[values > 0].astype(np.int64)
    ids = cells["cell_id"].astype("int64").to_numpy()
    assert sorted(ids.tolist()) == values.tolist() and len(set(ids.tolist())) == len(ids)
    assert set(cells["cell_namespace"]) <= {run["label_namespace"]}


def t3_failures(folder, record, molecules):
    """T3: the molecules whose cell_id is not the territory value at their pixel floor(j + ½), or that lie outside
    the grid and are not excluded outside_grid. Returns the failing (spot_namespace, spot_id) keys."""
    grid = record["segmentation"]["inputs"]["grid"]
    territories = territory_image(folder, record)
    origin, spacing = np.asarray(grid["origin_zyx"][1:]), np.asarray(grid["spacing_zyx"][1:])
    failures = []
    for row in molecules.itertuples(index=False):
        if not np.isfinite([row.sample_y, row.sample_x]).all():
            continue
        j = (np.array([row.sample_y, row.sample_x]) - origin) / spacing
        pixel = np.floor(j + 0.5).astype(np.int64)
        inside = bool(np.all(pixel >= 0) and np.all(pixel < territories.shape[1:]))
        if not inside:
            # Outside the grid: never retained; one its position's owner would retain is excluded outside_grid.
            own = not pd.isna(row.owner_fov_id) and row.owner_fov_id == row.fov_id
            if row.assembly_status == "retained" or (own and row.assembly_reason != "outside_grid"):
                failures.append((row.spot_namespace, row.spot_id))
            continue
        if row.assembly_status != "retained":
            continue
        value = int(territories[0, pixel[0], pixel[1]])
        cell = 0 if pd.isna(row.cell_id) else int(row.cell_id)
        if value != cell:
            failures.append((row.spot_namespace, row.spot_id))
    return failures


def nearest_owning_centre(folder, point_yx):
    """F1 outside every owning footprint, from the placement's origins and grids: the owning FOV whose footprint
    centre is nearest, ties by table order."""
    best, best_d = None, None
    for row in folder.placement.fovs.itertuples(index=False):
        if row.status not in ("image_supported", "metadata_only"):
            continue
        grid = folder.grids[row.fov_id]
        (_, sy, sx), (_, ny, nx) = grid.metadata.spacing_zyx, grid.shape_zyx
        centre = (row.origin_y + (ny - 1) / 2 * sy, row.origin_x + (nx - 1) / 2 * sx)
        d = (point_yx[0] - centre[0]) ** 2 + (point_yx[1] - centre[1]) ** 2
        if best_d is None or d < best_d:
            best, best_d = row.fov_id, d
    return best


def t4(folder, cells):
    """T4: every cell's owner_fov_id is the owner of its ownership centroid (ownership_map) and its fov_id that
    owner, or where none owns it the owning FOV with the nearest footprint centre."""
    ownership = ownership_map(folder.placement, folder.grids, config=AssemblyConfig())
    points = cells[[f"sample_ownership_centroid_{a}" for a in AXES]].to_numpy(np.float64)
    owners = ownership.owner(points)
    for owner, row, point in zip(owners, cells.itertuples(index=False), points):
        recorded = None if pd.isna(row.owner_fov_id) else row.owner_fov_id
        assert recorded == owner
        assert row.fov_id == (owner if owner is not None else nearest_owning_centre(folder, point[1:]))


def seam_cells(folder):
    """The truth cells across a seam: seen by two or more FOVs (a pixel of the cell in each FOV's labels)."""
    seen = {}
    for fov in folder.scene.ids:
        if folder.placement.fovs.set_index("fov_id").loc[fov, "status"] not in ("image_supported", "metadata_only"):
            continue
        from .sample_segmentation_fixtures import fov_label_image
        for value in np.unique(fov_label_image(folder, fov, "cell")).tolist():
            if value:
                seen.setdefault(int(value), set()).add(fov)
    return {cell: fovs for cell, fovs in seen.items() if len(fovs) >= 2}


# --- RS1: defaults and the recorded route ----------------------------------------------------------------------

RS1_CASES = {
    # case: (route, dimensionality, assignment, expected route, route_source, segment status, segment reason)
    "default_projection": (None, "projection", "default", "after_fusion", "default", "written", None),
    "set_after_fusion": ("after_fusion", "projection", "default", "after_fusion", "set", "written", None),
    "default_stack": (None, "stack", "default", "per_fov", "default", "off", "route_per_fov"),
    "default_both": (None, "both", "default", "per_fov", "default", "off", "route_per_fov"),
    "set_per_fov": ("per_fov", "projection", "default", "per_fov", "set", "off", "route_per_fov"),
    "no_cells_default": (None, "projection", None, "after_fusion", "default", "off", "no_cells"),
    "no_cells_set_after_fusion": ("after_fusion", "projection", None, "after_fusion", "set", "off", "no_cells"),
    "no_cells_set_per_fov": ("per_fov", "projection", None, "per_fov", "set", "off", "no_cells"),
}


@pytest.fixture(scope="module")
def rs1_folder(tmp_path_factory):
    folder = write_folder(tmp_path_factory.mktemp("rs1"), "fused_pair")
    per_fov_routes(folder)
    return folder, truth_config(folder, export_grid(folder))


@needs_anndata
@needs_library
@pytest.mark.parametrize("case", list(RS1_CASES))
def test_rs1_defaults_and_the_recorded_route(rs1_folder, case):
    folder, segmentation = rs1_folder
    route, dimensionality, assignment, expected, source, status, reason = RS1_CASES[case]
    config = route_config(route=route, dimensionality=dimensionality,
                          segmentation=segmentation if expected == "after_fusion" else None)
    output = folder.root / "rs1" / case
    record = export(folder, config, assignment=assignment, output_dir=output)
    assert record["parts"]["segment"] == status and record["segmentation"]["route"] == expected
    assert (record["assembly"]["route"], record["assembly"]["route_source"]) == (expected, source)
    if status == "off":
        assert record["segmentation"] == {"status": "off", "reason": reason, "route": expected, "route_source": source}
    mode = "molecules_only" if assignment is None else "cells"
    assert record["assembly"]["mode"] == mode and record["complete"] is True
    result = load_assembly(output)
    assert result.record["route"] == expected and result.mode == mode
    if assignment is None:
        assert len(result.cells) == 0 and result.record.get("segmentation_files") is None


# --- SG4: reuse -------------------------------------------------------------------------------------------------------

def first_export(tmp_path, k=1, runs="truth"):
    folder = write_folder(tmp_path, "fused_pair")
    grid = export_grid(folder, k)
    if runs == "truth":
        segmentation = truth_config(folder, grid)
    else:
        segmentation = seeded_runs(truth_import(folder, grid, "nucleus",
                                                labels=scene_truth_labels(folder, grid, "nucleus")))
    record = export(folder, route_config(segmentation=segmentation, k=k))
    return folder, segmentation, record


def t1(folder, record):
    """T1 of the contract: the label files with their SHA-256 on the recorded grid; with the fusion written, the
    fused input files equal the recorded inputs; otherwise the label files alone."""
    section = record["segmentation"]
    grid = section["inputs"]["grid"]
    for run in section["runs"]:
        path = folder / run["path"]
        assert path.is_file() and file_sha256(path) == run["sha256"]
        labels = tifffile.imread(path)
        assert labels.dtype == np.uint32 and [1, *labels.shape] == grid["shape_zyx"]
    if record["fusion"]["status"] == "written":
        outputs = {o["image"]: o for o in record["fusion"]["outputs"]
                   if o["dimensionality"] == "projection" and o["format"] == "ome_tiff"}
        for entry in section["inputs"]["images"]:
            assert file_sha256(folder / outputs[entry["image"]]["path"]) == entry["sha256"]
            assert outputs[entry["image"]]["shape_zyx"] == grid["shape_zyx"]


def t5(record):
    """T5: the assembly section's segmentation_files are the segmentation section's label files and SHA-256."""
    files = [{"run": r["name"], "path": r["path"], "sha256": r["sha256"]} for r in record["segmentation"]["runs"]]
    used = [e for e in record["assembly"]["segmentation_files"] if e["run"] is not None]
    assert used == files


SG4_REFUSALS = ["label_file_edited", "another_placement", "another_scale_yx", "no_saved_segmentation"]


@needs_anndata
@needs_library
@pytest.mark.parametrize("fuse_on", [True, False])
def test_sg4_reuse(tmp_path, fuse_on):
    folder, segmentation, first = first_export(tmp_path)
    before = listing(folder.export_dir)
    tables = load_assembly(folder.export_dir)
    record = export(folder, route_config(segmentation=segmentation, segment=False, fuse=fuse_on))
    assert record["parts"]["segment"] == "reused" and record["segmentation"]["status"] == "reused"
    assert record["parts"]["assemble"] == "written" and record["complete"] is True
    after = listing(folder.export_dir)
    fused = [o["path"] for o in first["fusion"]["outputs"]]
    if fuse_on:
        assert all(after[path] == before[path] for path in fused)
        assert record["fusion"]["status"] == "written"
    else:
        assert not any(path in after for path in fused) and record["fusion"] == {"status": "off"}
    for run in first["segmentation"]["runs"]:
        assert after[run["path"]] == before[run["path"]]
    reused = load_assembly(folder.export_dir)
    for name in ("fovs", "cells", "molecules", "counts"):
        pd.testing.assert_frame_equal(getattr(reused, name), getattr(tables, name), check_exact=True)
    t1(folder.export_dir, record)
    t5(record)
    # assemble: false with segment: false reuses the tables made from the reused segmentation.
    again = export(folder, route_config(segmentation=segmentation, segment=False, fuse=fuse_on, assemble=False))
    assert again["parts"]["assemble"] == "reused" and again["parts"]["segment"] == "reused"
    t5(again)


@needs_anndata
@needs_library
def test_sg4_reuse_with_changed_assignment_settings(tmp_path):
    """A reused segmentation takes the current export's assignment settings: the segmentation section's assignment
    holds them, and the tables are made with them (here an expansion of 1 µm, which the first export did not have)."""
    from starfinder.export import load_sample_segmentation
    folder, segmentation, first = first_export(tmp_path)
    assert first["segmentation"]["assignment"]["expansion"] is None
    labels = {run["path"]: run["sha256"] for run in first["segmentation"]["runs"]}
    changed = dataclasses.replace(segmentation, assignment=EXPANSION)
    record = export(folder, route_config(segmentation=changed, segment=False))
    assert record["parts"]["segment"] == "reused" and record["parts"]["assemble"] == "written"
    assert record["segmentation"]["assignment"] == json.loads(json.dumps(dataclasses.asdict(EXPANSION)))
    assert {run["path"]: run["sha256"] for run in record["segmentation"]["runs"]} == labels
    assert record["assembly"]["assignment"]["config"]["expansion"]["distance"] == 1.0
    assert "territories" in record["assembly"]["files"]
    t5(record)
    tables = load_assembly(folder.export_dir)
    assert tables.cells["expanded_size_voxels"].notna().all()
    expected = assemble(folder.placement, assembly_inputs(folder.dataset, folder.placement, None),
                        segmentation=load_sample_segmentation(folder.export_dir))
    for name in ("fovs", "cells", "molecules", "counts"):
        pd.testing.assert_frame_equal(getattr(tables, name), getattr(expected, name), check_exact=True)


@needs_anndata
@needs_library
@pytest.mark.parametrize("case", SG4_REFUSALS)
def test_sg4_reuse_refused_before_anything_is_written(tmp_path, case):
    folder, segmentation, first = first_export(tmp_path)
    config = route_config(segmentation=segmentation, segment=False)
    placement = None
    if case == "label_file_edited":
        path = folder.export_dir / "segmentation_cell.ome.tif"
        labels = tifffile.imread(path)
        labels[0, 0] = 99
        tifffile.imwrite(path, labels)
    elif case == "another_placement":
        moved = folder.placement.fovs.assign(origin_x=folder.placement.fovs.origin_x + [0.0, 0.5])
        other = approve_placement(PlacementResult(SAMPLE, moved, folder.placement.links, folder.placement.record),
                                  reviewer="test", note="moved")
        placement = folder.root / "stitching_moved" / SAMPLE
        save_placement(other, placement)
    elif case == "another_scale_yx":
        config = route_config(segmentation=segmentation, segment=False, k=2)
    elif case == "no_saved_segmentation":
        export(folder, route_config(route="per_fov"), assignment=None)
    before = listing(folder.root)
    with pytest.raises(ValueError) as caught:
        export(folder, config, placement=placement)
    message = str(caught.value)
    assert "segment: false" in message
    if case == "label_file_edited":
        recorded = first["segmentation"]["runs"][1]["sha256"]
        assert str(path) in message and recorded in message and file_sha256(path) in message
    elif case == "another_placement":
        assert "another placement" in message
    elif case == "another_scale_yx":
        assert "another grid" in message
    else:
        assert "holds no saved sample segmentation" in message
    assert listing(folder.root) == before


@needs_anndata
@needs_library
def test_sg4_reuse_refused_for_tables_of_another_segmentation(tmp_path):
    """assemble: false with segment: false is refused when the saved tables were not made from the reused
    segmentation (here: tables of the route per FOV)."""
    folder, segmentation, _ = first_export(tmp_path)
    per_fov_routes(folder)
    export(folder, route_config(route="per_fov"))
    segment_fused(folder.export_dir, config=segmentation, sample=SAMPLE, identity=(DS, SAMPLE))
    before = listing(folder.root)
    with pytest.raises(ValueError, match="segmentation_files"):
        export(folder, route_config(segmentation=segmentation, segment=False, assemble=False))
    assert listing(folder.root) == before


@needs_anndata
@needs_library
def test_sg4_a_changed_fused_input(tmp_path):
    """The cell run of seeded_runs reads the fused flamingo; a source image edited so that the new fusion differs."""
    folder, segmentation, first = first_export(tmp_path, runs="seeded")
    for fov in folder.scene.ids:
        path = folder.image_path("flamingo", fov)
        image = tifffile.imread(path)
        tifffile.imwrite(path, (image + 1).astype(image.dtype), photometric="minisblack", metadata={"axes": "ZYX"})
    with pytest.raises(ValueError, match="fused_input_changed") as caught:
        export(folder, route_config(segmentation=segmentation, segment=False))
    record = record_of(folder.export_dir)
    section = record["segmentation"]
    old = next(e["sha256"] for e in first["segmentation"]["inputs"]["images"] if e["image"] == "flamingo")
    new = next(o["sha256"] for o in record["fusion"]["outputs"] if o["image"] == "flamingo")
    assert old != new and old in str(caught.value) and new in str(caught.value)
    assert "'flamingo'" in str(caught.value)
    assert record["parts"]["segment"] == "failed" and section["reason"] == "fused_input_changed"
    assert (section["image"], section["recorded_sha256"], section["fused_sha256"]) == ("flamingo", old, new)
    assert record["parts"]["fuse"] == "written" and record["complete"] is False
    assert all((folder.export_dir / o["path"]).is_file() for o in record["fusion"]["outputs"])


# --- SG5: a part that cannot run -------------------------------------------------------------------------------------

def fallback_checks(folder, record):
    """The molecules_only fallback: the tables and sample.h5ad without a row, complete false."""
    import anndata
    assert record["parts"]["assemble"] == "written" and record["complete"] is False
    assert record["assembly"]["mode"] == "molecules_only" and record["assembly"]["route"] == "after_fusion"
    result = load_assembly(folder.export_dir)
    assert len(result.cells) == 0 and len(result.molecules) > 0
    assert anndata.read_h5ad(folder.export_dir / "sample.h5ad").n_obs == 0


@needs_anndata
def test_sg5_fuse_cannot_run(tmp_path, monkeypatch):
    folder = write_folder(tmp_path, "fused_pair")
    segmentation = truth_config(folder, export_grid(folder))
    monkeypatch.setitem(sys.modules, "multiview_stitcher", None)
    with pytest.warns(ExportIncompleteWarning) as caught:
        record = export(folder, route_config(segmentation=segmentation))
    messages = " ".join(str(w.message) for w in caught)
    assert "export part fuse" in messages and "export part segment" in messages
    assert (record["parts"]["fuse"], record["fusion"]["reason"]) == ("not_run", "extra_missing")
    assert (record["parts"]["segment"], record["segmentation"]["reason"]) == ("not_run", "part_not_run")
    assert record["assembly"]["reason"] == "segment_not_run"
    fallback_checks(folder, record)


@needs_anndata
@needs_library
def test_sg5_the_method_backend_cannot_run(tmp_path, monkeypatch):
    folder = write_folder(tmp_path, "fused_pair")
    grid = export_grid(folder)
    cellpose, _ = model_files(tmp_path)
    nuclei = truth_import(folder, grid, "nucleus", labels=scene_truth_labels(folder, grid, "nucleus"))
    runs = (SampleSegmentationRun("nucleus", "nucleus", (), LabelImportConfig(str(nuclei), "nucleus")),
            SampleSegmentationRun("cell", "cell", (("cytoplasm", "flamingo"), ("nuclear", "dapi")),
                                  CellposeConfig(diameter=24.0, model_path=str(cellpose))))
    monkeypatch.setitem(sys.modules, "cellpose", None)
    with pytest.warns(ExportIncompleteWarning, match="extra_missing"):
        record = export(folder, route_config(segmentation=SampleSegmentationConfig(runs, "cell", "nucleus")))
    section = record["segmentation"]
    assert (record["parts"]["segment"], section["reason"], section["extra"]) == ("not_run", "extra_missing",
                                                                                 "cellpose")
    assert record["parts"]["fuse"] == "written" and record["assembly"]["reason"] == "segment_not_run"
    fallback_checks(folder, record)


@needs_library
def test_sg5_no_included_fov(tmp_path):
    """Every FOV failed: fuse_images and segment_fused record empty (no_included_fov) when called alone.
    Dataset.export refuses such a placement before anything is written, by assemble's own check (no FOV owns a
    point)."""
    folder = write_folder(tmp_path, "fused_pair")
    p = folder.placement
    fovs = p.fovs.assign(status="failed", origin_z=np.nan, origin_y=np.nan, origin_x=np.nan)
    failed = approve_placement(PlacementResult(SAMPLE, fovs, p.links, p.record), reviewer="test", note="failed")
    record = fuse_images(failed, [], config=fusion_config(1), output_dir=folder.export_dir)
    assert record.status == "empty"
    nuclei = tmp_path / "nuclei.tif"
    result = segment_fused(folder.export_dir, config=seeded_runs(nuclei), sample=SAMPLE, identity=(DS, SAMPLE))
    assert (result.status, result.record["reason"]) == ("empty", "no_included_fov") and result.runs == {}
    saved = folder.root / "stitching_failed" / SAMPLE
    save_placement(failed, saved)
    before = listing(folder.root)
    with pytest.raises(ValueError, match="owns a point"):
        export(folder, route_config(segmentation=seeded_runs(nuclei)), placement=saved)
    assert listing(folder.root) == before


@needs_anndata
@needs_library
def test_sg5_a_method_that_raises(tmp_path, monkeypatch):
    import starfinder.export._segment as segment_module
    folder = write_folder(tmp_path, "fused_pair")
    grid = export_grid(folder)
    nuclei = truth_import(folder, grid, "nucleus", labels=scene_truth_labels(folder, grid, "nucleus"))

    def broken(*args, **kwargs):
        raise RuntimeError("the method failed")
    monkeypatch.setattr(segment_module, "segment", broken)
    with pytest.raises(RuntimeError, match="the method failed"):
        export(folder, route_config(segmentation=seeded_runs(nuclei)))
    record = record_of(folder.export_dir)
    assert record["parts"]["segment"] == "failed" and record["complete"] is False
    assert record["segmentation"]["error"] == {"type": "RuntimeError", "message": "the method failed"}
    assert record["parts"]["fuse"] == "written"
    assert all((folder.export_dir / o["path"]).is_file() for o in record["fusion"]["outputs"])


# --- SA1: the conversion (MT1) -------------------------------------------------------------------------------------

@needs_anndata
@needs_library
@pytest.mark.parametrize("scene", SA_SCENES)
def test_sa1_the_conversion(routes, scene):
    case = routes(scene)
    table = case.captured["molecules"].table
    base = case.base.molecules
    retained = np.flatnonzero(base["assembly_status"].eq("retained").to_numpy(dtype=bool))
    assert len(table) == len(retained) > 0
    assert table["spot_id"].tolist() == [str(i) for i in retained]
    assert set(table["spot_namespace"]) == {json.dumps([DS, SAMPLE, None, None], separators=(",", ":"))}
    origin = np.asarray(case.grid.metadata.origin_zyx)
    spacing = np.asarray(case.grid.metadata.spacing_zyx)
    expected = (base[[f"sample_{a}" for a in AXES]].to_numpy(np.float64)[retained] - origin) / spacing
    assert np.array_equal(table[["z", "y", "x"]].to_numpy(np.float64), expected)
    assert table["gene_id"].tolist() == base["gene_id"].astype(object).to_numpy()[retained].tolist()
    assert case.captured["kwargs"]["grid"].shape_zyx == case.grid.shape_zyx
    record = case.after["assembly"]["assignment"]["inputs"]["molecules"]
    assert record["source"]["conversion"] == "MT1" and record["sha256"] == case.captured["molecules"].sha256


# --- SA2: mapping back -----------------------------------------------------------------------------------------------

ASSIGNED = ["voxel_z", "voxel_y", "voxel_x", "assignment_status", "cell_id", "in_expansion", "original_cell_id",
            "nucleus_id", "compartment"]


@needs_anndata
@needs_library
@pytest.mark.parametrize("scene", SA_SCENES)
def test_sa2_mapping_back(routes, scene):
    case = routes(scene)
    molecules, base = case.result.molecules, case.base.molecules
    for column in ("fov_id", "spot_namespace", "spot_id", "z", "y", "x", "gene_id"):
        pd.testing.assert_series_equal(molecules[column], base[column], check_exact=True)
    assigned = case.captured["result"].molecules
    rows = [int(i) for i in case.captured["molecules"].table["spot_id"]]
    for position, row in enumerate(rows):
        if molecules["assembly_status"].iloc[row] != "retained":
            assert assigned["assignment_status"].iloc[position] == "outside_grid"
            continue
        for column in ASSIGNED:
            a, b = molecules[column].iloc[row], assigned[column].iloc[position]
            assert (pd.isna(a) and pd.isna(b)) or a == b, (row, column)
    excluded = molecules[molecules["assembly_status"].eq("excluded").to_numpy(dtype=bool)]
    assert excluded[ASSIGNED + ["cell_namespace"]].isna().all().all()
    # The temporary keys occur in no table.
    sample_namespace = json.dumps([DS, SAMPLE, None, None], separators=(",", ":"))
    for path in (case.folder.export_dir / "tables").iterdir():
        assert sample_namespace not in path.read_text()
    assert not molecules["spot_id"].isin([str(i) for i in rows]).any()


# --- SA3: each cell once and whole ------------------------------------------------------------------------------------

#: The written values of the hand-built fixtures: per cell its retained molecules of every FOV that owns a part of
#: it, as whole counts by gene. fused_pair: cell 1 receives FOV1's molecule at x 8.5 µm and FOV2's at x 11 µm.
#: fused_quad: four molecules per cell (genes alternate from the cell's position in QUAD_CELLS), the border cell's
#: second beyond the segmentation grid.
WRITTEN = {"fused_pair": {1: {"A": 2}, 2: {"B": 1}, 3: {"B": 1}},
           "fused_quad": {**{cell: {"A": 2, "B": 2} for cell in QUAD_CELLS if cell != 2 ** 32 - 1},
                          2 ** 32 - 1: {"B": 2, "A": 1}}}


def whole_counts(result):
    whole = result.counts[result.counts["compartment"].eq("whole").to_numpy(dtype=bool)]
    out = {}
    for cell, gene, count in zip(whole["cell_id"].astype("int64"), whole["gene_id"], whole["count"]):
        out.setdefault(int(cell), {})[gene] = int(count)
    return out


@needs_anndata
@needs_library
@pytest.mark.parametrize("scene", SA_SCENES)
def test_sa3_each_cell_once_and_whole(routes, scene):
    import anndata
    case = routes(scene)
    folder, cells = case.folder, case.result.cells
    t2(folder.export_dir, case.after, cells)
    truth = scene_truth_labels(folder, case.grid, "cell")
    ids = sorted(int(v) for v in np.unique(truth) if v)
    assert cells["cell_id"].astype("int64").tolist() == ids
    territories = territory_image(folder.export_dir, case.after)
    np.testing.assert_array_equal(territories, truth)
    seams = seam_cells(folder)
    assert seams, "the fixture has a cell across a seam"
    if scene == "fused_pair":
        assert set(seams) == {1}
    if scene == "fused_quad":
        assert {BIG, 2, 3} <= set(seams) and len(seams[BIG]) == 4
    for cell in seams:
        rows = cells[cells["cell_id"].astype("int64").eq(cell).to_numpy(dtype=bool)]
        assert len(rows) == 1
        assert int(rows["size_voxels"].iloc[0]) == int((truth == cell).sum())
    # The counts hold the retained molecules of every FOV that owns a part of the cell.
    if scene in WRITTEN:
        assert whole_counts(case.result) == WRITTEN[scene]
        n = dict(zip(cells["cell_id"].astype("int64"), cells["n_molecules"]))
        assert n == {cell: sum(genes.values()) for cell, genes in WRITTEN[scene].items()}
        owners = case.result.molecules[case.result.molecules["cell_id"].notna().to_numpy(dtype=bool)]
        if scene == "fused_pair":
            assert sorted(owners[owners["cell_id"].astype("int64").eq(1)]["fov_id"]) == ["FOV1", "FOV2"]
        else:
            assert len(set(owners[owners["cell_id"].astype("int64").eq(BIG)]["fov_id"])) >= 2
    # sample.h5ad: one row per counted cell, the cells of the table.
    adata = anndata.read_h5ad(folder.export_dir / "sample.h5ad")
    counted = cells[cells["status"].eq("kept").to_numpy(dtype=bool)]
    assert adata.obs_names.tolist() == counted["obs_name"].tolist()
    for cell in seams:
        assert int((adata.obs["cell_id"] == cell).sum()) == int(cell in set(counted["cell_id"].astype("int64")))


# --- SA4: once and whole with a method ----------------------------------------------------------------------------

@needs_anndata
@needs_library
@pytest.mark.parametrize("scene", ["fused_pair", "fused_quad"])
def test_sa4_once_and_whole_with_a_method(routes, scene):
    case = routes(scene, runs="seeded")
    folder, cells = case.folder, case.result.cells
    seeds = scene_truth_labels(folder, case.grid, "nucleus")
    seed_ids = sorted(int(v) for v in np.unique(seeds) if v)
    assert len(cells) == len(seed_ids) and (cells["size_voxels"] > 0).all()
    labels = territory_image(folder.export_dir, case.after)
    ownership = ownership_map(folder.placement, folder.grids, config=AssemblyConfig())
    origin, spacing = np.asarray(case.grid.metadata.origin_zyx), np.asarray(case.grid.metadata.spacing_zyx)
    truth = scene_truth_labels(folder, case.grid, "cell")
    for cell in seam_cells(folder):
        # The region of the seed inside the truth cell across a seam.
        values, counts = np.unique(labels[truth == cell], return_counts=True)
        region = int(values[values > 0][np.argmax(counts[values > 0])])
        index = np.argwhere(labels == region).astype(np.float64)
        points = origin + index * spacing
        owners = set(ownership.owner(points).tolist()) - {None}
        assert len(owners) >= 2, (cell, owners)


# --- SA5: assigned where it lies --------------------------------------------------------------------------------------

@needs_anndata
@needs_library
@pytest.mark.parametrize("scene", SA_SCENES)
def test_sa5_assigned_where_it_lies(routes, scene):
    case = routes(scene)
    molecules = case.result.molecules
    assert t3_failures(case.folder.export_dir, case.after, molecules) == []
    if scene == "fused_quad":
        assert molecules["assembly_reason"].eq("outside_grid").sum() == 1
    if scene != "scene_plane":
        return
    # The share of retained molecules in their true cell, reported; every difference is a molecule whose sampled
    # pixel's imported label is not its true cell.
    generated = case.folder.extra["scene"]
    rows = generated.molecule_fovs.set_index(["fov_id", "csv_row"])
    true_cell = dict(zip(generated.molecules.amplicon_id, generated.molecules.true_cell_id))
    labels = territory_image(case.folder.export_dir, case.after)
    retained = molecules[molecules["assembly_status"].eq("retained").to_numpy(dtype=bool)]
    same = 0
    for row in retained.itertuples(index=False):
        truth = true_cell[rows.loc[(row.fov_id, int(row.spot_id.split(":")[1])), "amplicon_id"]]
        truth = 0 if pd.isna(truth) else int(truth)   # an ambient molecule has no true cell
        cell = 0 if pd.isna(row.cell_id) else int(row.cell_id)
        if cell == truth:
            same += 1
            continue
        y, x = int(row.voxel_y), int(row.voxel_x)
        assert int(labels[0, y, x]) != truth
    print(f"\nSA5 scene_plane: {same} of {len(retained)} retained molecules in their true cell "
          f"({same / len(retained):.4f})")


# --- SA6: fov_id of a cell -------------------------------------------------------------------------------------------

@needs_anndata
@needs_library
@pytest.mark.parametrize("scene", ["fused_quad", "fused_quad_notch"])
def test_sa6_fov_id_of_a_cell(routes, scene):
    case = routes(scene)
    cells = case.result.cells
    t4(case.folder, cells)
    assert cells["obs_name"].is_unique
    assert cells["obs_name"].tolist() == [f"{SAMPLE}_{f}_{c}" for f, c in zip(cells["fov_id"], cells["cell_id"])]
    by_id = cells.set_index(cells["cell_id"].astype("int64"))
    if scene == "fused_quad":
        crossing = by_id.loc[BIG]
        ownership = ownership_map(case.folder.placement, case.folder.grids, config=AssemblyConfig())
        point = crossing[[f"sample_ownership_centroid_{a}" for a in AXES]].to_numpy(np.float64)
        assert crossing["fov_id"] == crossing["owner_fov_id"] == ownership.owner(point[None])[0]
    else:
        notch = by_id.loc[NOTCH_CELL]
        assert pd.isna(notch["owner_fov_id"]) and notch["fov_id"] == "FOV2"
        point = notch[[f"sample_ownership_centroid_{a}" for a in AXES]].to_numpy(np.float64)
        assert point[1] > 20.0 and point[2] > 20.0   # where no owning footprint covers it
        assert nearest_owning_centre(case.folder, point[1:]) == "FOV2"
        fovs = case.result.fovs.set_index("fov_id")
        assert fovs.loc["FOV4", "placement_status"] == "placeholder" and fovs.loc["FOV4", "n_cells"] == 0


# --- SA7: identities ----------------------------------------------------------------------------------------------

@needs_anndata
@needs_library
@pytest.mark.parametrize("scene", SA_SCENES + ["fused_quad_notch"])
def test_sa7_identities(routes, scene):
    case = routes(scene)
    for folder in (case.folder.export_dir, per_fov_dir(case.folder)):
        loaded = load_assembly(folder)   # AssemblyResult checks the identities on construction
        assert isinstance(loaded, AssemblyResult)
    result = case.result
    assert result.record["route"] == "after_fusion" and case.per_fov_result.record["route"] == "per_fov"

    def rebuilt(**tables):
        frames = {name: getattr(result, name) for name in ("fovs", "cells", "molecules", "counts")}
        frames.update(tables)
        return AssemblyResult(result.sample, frames["fovs"], frames["cells"], frames["molecules"],
                              frames["counts"], result.genes, result.record)

    molecules = result.molecules
    assigned = np.flatnonzero(molecules["assignment_status"].eq("assigned").fillna(False).to_numpy(dtype=bool))
    # A molecule that names a cell that does not hold its pixel: T3 finds it.
    row = int(assigned[0])
    other = int(next(c for c in result.cells["cell_id"].astype("int64") if c != int(molecules["cell_id"].iloc[row])))
    edited = molecules.copy()
    edited.loc[row, "cell_id"] = other
    failures = t3_failures(case.folder.export_dir, case.after, edited)
    assert failures == [tuple(molecules[["spot_namespace", "spot_id"]].iloc[row])]
    # A molecule that names a cell key the cell table does not hold: identity 3.
    edited = molecules.copy()
    edited.loc[row, "cell_namespace"] = json.dumps([DS, SAMPLE, None, None, "other"], separators=(",", ":"))
    with pytest.raises(ValueError, match="identity 3"):
        rebuilt(molecules=edited)
    # Identity 3 by position: an excluded molecule with a sample assignment.
    excluded = int(np.flatnonzero(molecules["assembly_status"].eq("excluded").to_numpy(dtype=bool))[0])
    edited = molecules.copy()
    edited.loc[excluded, "assignment_status"] = "unassigned"
    with pytest.raises(ValueError, match="identity 3"):
        rebuilt(molecules=edited)
    # Identity 1 against the labels: a cell row removed.
    cells = result.cells
    without = cells[cells["n_molecules"].eq(0).to_numpy(dtype=bool)].index
    if len(without):
        with pytest.raises(ValueError, match="identity 1"):
            rebuilt(cells=cells.drop(index=without[:1]))


# --- SA8: molecules keep their rule ---------------------------------------------------------------------------------

@needs_anndata
@needs_library
@pytest.mark.parametrize("scene", ["fused_pair", "fused_quad"])
def test_sa8_molecules_keep_their_rule(routes, scene):
    case = routes(scene)
    molecules, base = case.result.molecules, case.base.molecules
    changed = (molecules["assembly_status"].astype(object) != base["assembly_status"].astype(object)) | (
        molecules["assembly_reason"].astype(object).fillna("") != base["assembly_reason"].astype(object).fillna(""))
    changed = np.flatnonzero(changed.to_numpy(dtype=bool))
    expected_changes = 1 if scene == "fused_quad" else 0
    assert len(changed) == expected_changes
    for row in changed:
        assert base["assembly_status"].iloc[row] == "retained"
        assert (molecules["assembly_status"].iloc[row], molecules["assembly_reason"].iloc[row]) == \
            ("excluded", "outside_grid")
    pd.testing.assert_series_equal(molecules["owner_fov_id"], base["owner_fov_id"], check_exact=True)
    assert (molecules["assembly_flags"] == "").all()


# --- SO1: same columns, keys and files ----------------------------------------------------------------------------

def section_keys(record):
    return {name: sorted(section) if isinstance(section, dict) else None for name, section in record.items()
            if name not in ("format_version", "stage", "sample", "frame_id", "complete", "parts",
                            "viewer_qualification")}


@needs_anndata
@needs_library
@pytest.mark.parametrize("expansion", [False, True])
def test_so1_same_columns_keys_and_files(routes, expansion):
    import anndata
    case = routes("fused_pair", **({"assignment": EXPANSION} if expansion else {}))
    after, per_fov = case.result, case.per_fov_result
    # The four tables: names, order and dtypes.
    for name in ("fovs", "cells", "molecules", "counts"):
        a, b = getattr(after, name), getattr(per_fov, name)
        assert list(a.columns) == list(b.columns) and a.dtypes.to_dict() == b.dtypes.to_dict(), name
    # sample.h5ad.
    adata_a = anndata.read_h5ad(case.folder.export_dir / "sample.h5ad")
    adata_b = anndata.read_h5ad(per_fov_dir(case.folder) / "sample.h5ad")
    assert list(adata_a.obs.columns) == list(adata_b.obs.columns)
    assert adata_a.var_names.tolist() == adata_b.var_names.tolist() and list(adata_a.var) == list(adata_b.var)
    assert sorted(adata_a.layers) == sorted(adata_b.layers) == ["cytoplasm", "nucleus"]
    assert sorted(adata_a.obsm) == sorted(adata_b.obsm) and sorted(adata_a.uns) == sorted(adata_b.uns)
    # export.json: top-level keys; each section's keys, apart from those of this route.
    record_a, record_b = case.after, case.per_fov
    assert list(record_a) == list(record_b) and list(record_a["parts"]) == list(record_b["parts"])
    keys_a, keys_b = section_keys(record_a), section_keys(record_b)
    assert set(keys_a) == set(keys_b)
    for name in keys_a:
        if name == "segmentation":
            continue
        extra = {"segmentation_files", "assignment"} if name == "assembly" else set()
        assert set(keys_a[name]) - extra == set(keys_b[name]), name
    assert set(keys_a["assembly"]) - set(keys_b["assembly"]) == {"segmentation_files", "assignment"}
    # The files, apart from the label files of this route.
    files_a = {p for p in listing(case.folder.export_dir) if not p.startswith("segmentation_")
               and p != "territories.ome.tif"}
    assert files_a == set(listing(per_fov_dir(case.folder)))
    assert ("territories.ome.tif" in listing(case.folder.export_dir)) == expansion
    so1_column_by_column(case, adata_a, expansion)


def so1_column_by_column(case, adata, expansion):
    """Every row of "Outputs, column by column" on the route after fusion."""
    result, folder, grid = case.result, case.folder, case.grid
    record = case.after
    cells, molecules, counts, fovs = result.cells, result.molecules, result.counts, result.fovs
    section = record["segmentation"]
    run = next(r for r in section["runs"] if r["name"] == "cell")
    labels = label_image(folder.export_dir, run["path"])
    assigned = case.captured["result"]
    origin, spacing = np.asarray(grid.metadata.origin_zyx), np.asarray(grid.metadata.spacing_zyx)
    # The cells.
    t4(folder, cells)
    assert (cells["cell_namespace"] == run["label_namespace"]).all()
    assert cells["cell_namespace"].iloc[0] == json.dumps([DS, SAMPLE, None, None, "cell"], separators=(",", ":"))
    for column in ("status", "exclusion_reason", "size_voxels", "expanded_size_voxels", "size_physical",
                   "expanded_size_physical", "centroid_z", "centroid_y", "centroid_x", "expanded_centroid_z",
                   "expanded_centroid_y", "expanded_centroid_x", "n_molecules", "n_nuclei", "correspondence",
                   "correspondence_flags", "compartments"):
        pd.testing.assert_series_equal(cells[column].reset_index(drop=True), assigned.cells[column].reset_index(
            drop=True), check_exact=True, check_names=False, check_dtype=False)
    for cell, size in zip(cells["cell_id"].astype("int64"), cells["size_voxels"]):
        assert int((labels == cell).sum()) == int(size)
    np.testing.assert_array_equal(cells["size_physical"], cells["size_voxels"] * spacing[1] * spacing[2])
    assert (cells["centroid_z"] == 0.0).all()
    sample = origin + cells[["centroid_z", "centroid_y", "centroid_x"]].to_numpy() * spacing
    assert cells["sample_centroid_z"].isna().all()
    np.testing.assert_array_equal(cells[["sample_centroid_y", "sample_centroid_x"]].to_numpy(), sample[:, 1:])
    if expansion:
        grown = origin + cells[["expanded_centroid_z", "expanded_centroid_y", "expanded_centroid_x"]].to_numpy() \
            * spacing
        np.testing.assert_array_equal(cells[["sample_expanded_centroid_y", "sample_expanded_centroid_x"]].to_numpy(),
                                      grown[:, 1:])
        assert (cells["ownership_centroid"] == "expanded").all()
        own = cells[["sample_expanded_centroid_y", "sample_expanded_centroid_x"]].to_numpy()
    else:
        assert cells[[f"sample_expanded_centroid_{a}" for a in AXES]].isna().all().all()
        assert (cells["ownership_centroid"] == "original").all()
        own = cells[["sample_centroid_y", "sample_centroid_x"]].to_numpy()
    np.testing.assert_array_equal(cells[["sample_ownership_centroid_y", "sample_ownership_centroid_x"]].to_numpy(),
                                  own)
    assert (cells["assembly_status"] == "retained").all() and cells["assembly_reason"].isna().all()
    assert (cells["assembly_flags"] == "").all() and (cells["conflict_with"] == "").all()
    assert cells["duplicate_of"].isna().all()
    # The molecules.
    placed = molecules["assignment_status"].notna().to_numpy(dtype=bool)
    assert (molecules["voxel_z"][placed] == 0).all() and molecules["voxel_z"][~placed].isna().all()
    has_cell = molecules["cell_id"].notna().to_numpy(dtype=bool)
    assert (molecules["cell_namespace"][has_cell] == run["label_namespace"]).all()
    assert molecules["cell_namespace"][~has_cell].isna().all()
    # The counts.
    fov_of = dict(zip(cells["cell_id"].astype("int64"), cells["fov_id"]))
    assert [fov_of[int(c)] for c in counts["cell_id"]] == counts["fov_id"].tolist()
    assert (counts["cell_namespace"] == run["label_namespace"]).all()
    # The fovs.
    assert (fovs["kind"] == "molecules").all()
    per_fov = cells["fov_id"].value_counts()
    assert fovs["n_cells"].tolist() == [int(per_fov.get(f, 0)) for f in fovs["fov_id"]]
    assert fovs["n_cells_retained"].tolist() == fovs["n_cells"].tolist()
    assert fovs["n_molecules"].tolist() == case.base.fovs["n_molecules"].tolist()
    # sample.h5ad.
    counted = cells[cells["status"].eq("kept").to_numpy(dtype=bool)]
    assert adata.obs_names.tolist() == counted["obs_name"].tolist()
    np.testing.assert_array_equal(adata.obsm["spatial"],
                                  counted[["sample_ownership_centroid_x", "sample_ownership_centroid_y"]].to_numpy())
    uns = json.loads(adata.uns["assembly"])
    assert uns["route"] == "after_fusion" and uns["segmentation_files"] == record["assembly"]["segmentation_files"]
    # export.json.
    assembly = record["assembly"]
    assert (assembly["mode"], assembly["route"], assembly["route_source"]) == ("cells", "after_fusion", "default")
    assert assembly["config"]["duplicate_threshold"] is None
    assert assembly["flags"] == {"computed": [], "conflicts": True, "applicable": False}
    assert assembly["assignment"]["stage"] == "assignment"
    assert all(e["kind"] == "molecules" and e["assignment"] is None for e in assembly["fovs"])
    files = [{"run": r["name"], "path": r["path"], "sha256": r["sha256"]} for r in section["runs"]]
    if expansion:
        digest = file_sha256(folder.export_dir / "territories.ome.tif")
        files.append({"run": None, "path": "territories.ome.tif", "sha256": digest})
        assert assembly["files"]["territories"] == {"path": "territories.ome.tif", "sha256": digest}
        np.testing.assert_array_equal(label_image(folder.export_dir, "territories.ome.tif"), assigned.territories)
    assert assembly["segmentation_files"] == files
    assert section["assignment"] == json.loads(json.dumps(dataclasses.asdict(case.assignment)))


@needs_anndata
@needs_library
def test_so1_the_saved_territories_reload_and_save_again(routes, tmp_path):
    """A result loaded from a folder of the route with an expansion keeps its territories, so that save_assembly
    writes the same territories.ome.tif elsewhere."""
    from starfinder.assembly import save_assembly
    case = routes("fused_pair", assignment=EXPANSION)
    loaded = load_assembly(case.folder.export_dir)
    saved = save_assembly(loaded, tmp_path / "copy")
    assert saved.record["segmentation_files"] == case.after["assembly"]["segmentation_files"]
    np.testing.assert_array_equal(tifffile.imread(tmp_path / "copy" / "territories.ome.tif"),
                                  tifffile.imread(case.folder.export_dir / "territories.ome.tif"))


# --- Each cell once and whole, stated as the result (the worker notes) -----------------------------------------------

@needs_anndata
@needs_library
@pytest.mark.parametrize("scene", SA_SCENES)
def test_sa3_seam_cells_on_both_routes(routes, scene):
    """For each fixture: the truth cells across a seam, the rows the route per FOV has for them (with the duplicate
    rule on) and the rows the route after fusion has. Printed for the worker notes (``-s``)."""
    case = routes(scene)
    seams = seam_cells(case.folder)
    per_fov = case.per_fov_result.cells
    after = case.result.cells
    ids_per_fov = per_fov["cell_id"].astype("int64")
    rows = int(ids_per_fov.isin(list(seams)).sum())
    retained = int((ids_per_fov.isin(list(seams)) & per_fov["assembly_status"].eq("retained")).sum())
    after_rows = int(after["cell_id"].astype("int64").isin(list(seams)).sum())
    assert after_rows == len(seams)
    assert rows >= len(seams)
    print(f"\nSA3 {scene}: {len(seams)} truth cells across a seam; route per FOV {rows} rows "
          f"({retained} retained); route after fusion {after_rows} rows")


# --- SG2 (the assignment's expansion, the last row of the LS2 table) -------------------------------------------------

def test_sg2_the_assignment_expansion_at_factor_k():
    """The assignment's expansion distance: µm unchanged; pixels (with legacy_pixel_expansion) divided by k. The
    settings read back from the segmentation section's JSON equal the config."""
    from starfinder.assembly._fused import assignment_config, used_assignment
    pixels = AssignmentConfig(expansion=ExpandLabelsConfig(distance=6.0, unit="pixel", mode="planar"),
                              legacy_pixel_expansion=True)
    for k in (1, 2, 4):
        assert used_assignment(pixels, k).expansion.distance == 6.0 / k
        assert used_assignment(EXPANSION, k) == EXPANSION
    assert used_assignment(AssignmentConfig(), 4) == AssignmentConfig()
    for config in (pixels, EXPANSION, AssignmentConfig(exclude_cells_without_nucleus=False)):
        assert assignment_config(json.loads(json.dumps(dataclasses.asdict(config)))) == config
    assert assignment_config(None) == AssignmentConfig()


@needs_anndata
@needs_library
def test_sg2_the_assignment_expansion_in_the_export(tmp_path):
    """At k = 2 an expansion of 2 pixels of the acquired grid is 1 pixel of the segmentation grid; the sample
    assignment's record holds the distance used, the segmentation section the one given."""
    folder = write_folder(tmp_path, "fused_pair")
    pixels = AssignmentConfig(expansion=ExpandLabelsConfig(distance=2.0, unit="pixel", mode="planar"),
                              legacy_pixel_expansion=True)
    segmentation = truth_config(folder, export_grid(folder, 2), assignment=pixels)
    record = export(folder, route_config(segmentation=segmentation, k=2))
    assert record["segmentation"]["assignment"]["expansion"]["distance"] == 2.0
    assert record["assembly"]["assignment"]["config"]["expansion"]["distance"] == 1.0
    assert "territories" in record["assembly"]["files"]
